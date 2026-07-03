"""
backfill_rag.py — Seed the analog RAG database from historical signal CSVs.

Reads every trading_logs/signals_*.csv, finds rows where pnl is filled in
(either by the bot's exit logic or manually by the trader), extracts indicator
values from the AI reasoning text, and inserts them into analog_history.db.

Run once to seed the DB, then run again any time after manually entering PnL
to pick up new rows. Duplicate rows (same symbol + original timestamp) are
silently skipped.

Usage:
    python backfill_rag.py               # process all CSV files
    python backfill_rag.py --dry-run     # show what would be inserted, no writes
    python backfill_rag.py --date 2026-06-10  # single date only
"""
import sys
import re
import csv
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "trading_logs"
DB_PATH = ROOT / "analog_history.db"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Symbol → security_id (for optional candle-data fallback) ─────────────────
def _load_security_ids() -> dict:
    try:
        from constant import (FNO_UNIVERSE, ETF_LIQUID,
                              FILTERED_FNO_UNIVERSE, NIFTY50_UNIVERSE)
        from dhan_integration import VWAP_RECLAIM_STOCKS
        m = {**FNO_UNIVERSE, **ETF_LIQUID,
             **FILTERED_FNO_UNIVERSE, **VWAP_RECLAIM_STOCKS, **NIFTY50_UNIVERSE}
        return {sym: str(sid) for sym, sid in m.items()}
    except Exception:
        return {}


# ── Indicator extraction ──────────────────────────────────────────────────────

def _extract_from_reasoning(reasoning: str, entry_price: float,
                             stop_loss: float) -> dict:
    """
    Pull rsi/adx/volume_ratio/mfi from the AI reasoning text.
    ATR% is estimated from the stop_loss column (SL = 1.5x ATR by bot convention).
    Falls back to neutral defaults for anything not found.
    """
    def _find(pattern, text, default):
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return default

    rsi          = _find(r'RSI[^0-9]{0,5}(\d+\.?\d*)',          reasoning, 50.0)
    adx          = _find(r'ADX[^0-9]{0,5}(\d+\.?\d*)',          reasoning, 22.0)
    volume_ratio = _find(r'volume_ratio[^0-9]{0,5}(\d+\.?\d*)', reasoning, 1.0)
    mfi          = _find(r'MFI[^0-9]{0,5}(\d+\.?\d*)',          reasoning, 50.0)

    # ATR% from stop_loss: SL = 1.5 × ATR  →  ATR% = (SL distance %) / 1.5
    atr_pct = 0.3
    if entry_price > 0 and stop_loss > 0:
        sl_pct = abs(entry_price - stop_loss) / entry_price * 100
        if sl_pct > 0:
            atr_pct = round(sl_pct / 1.5, 4)

    return {
        "rsi":          max(0.0, min(100.0, rsi)),
        "adx":          max(0.0, adx),
        "volume_ratio": max(0.0, volume_ratio),
        "mfi":          max(0.0, min(100.0, mfi)),
        "atr_pct":      max(0.0, atr_pct),
    }


def _candle_indicators(symbol: str, signal_ts: str,
                       security_ids: dict) -> dict | None:
    """
    Optional: load the saved 3-minute CSV for the signal date and recompute
    indicators. Returns None when no candle file is found.
    """
    try:
        from indicators import calculate_technical_indicators
        import pandas as pd

        date_str = signal_ts[:10]             # "2026-06-10"
        sid = security_ids.get(symbol)
        if not sid:
            return None

        candle_path = ROOT / "kronos_integrated_bot" / "data" / date_str / f"{sid}_3minute.csv"
        if not candle_path.exists():
            return None

        df = pd.read_csv(candle_path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")

        # Keep bars up to (and including) the signal bar
        try:
            sig_dt = pd.Timestamp(signal_ts).tz_localize("Asia/Kolkata")
        except Exception:
            sig_dt = pd.Timestamp(signal_ts)
        df = df[df.index <= sig_dt]

        if len(df) < 14:          # need at least 14 bars for RSI/ADX
            return None

        ind = calculate_technical_indicators(df)
        return ind if ind else None
    except Exception as exc:
        logger.debug("Candle indicator fallback failed for %s: %s", symbol, exc)
        return None


# ── Deduplication ─────────────────────────────────────────────────────────────

def _existing_keys(db_path: str) -> set:
    """Return set of (symbol, ts) already in the DB."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT symbol, ts FROM setups").fetchall()
        conn.close()
        return {(r[0], r[1]) for r in rows}
    except Exception:
        return set()


# ── CSV reader ────────────────────────────────────────────────────────────────

def _load_csv_rows(date_filter: str | None = None) -> list[dict]:
    rows = []
    for path in sorted(LOG_DIR.glob("signals_*.csv")):
        date_str = path.stem.replace("signals_", "")
        if date_filter and date_str != date_filter:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["_file_date"] = date_str
                    rows.append(row)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def backfill(dry_run: bool = False, date_filter: str | None = None,
             reenrich: bool = False) -> int:
    """
    Returns the number of new rows inserted (or that would be inserted
    in dry-run mode).
    """
    # Ensure the DB schema exists (AnalogRAG.__init__ runs CREATE TABLE IF NOT EXISTS)
    try:
        from analog_rag import AnalogRAG
        AnalogRAG(db_path=DB_PATH)
    except Exception as exc:
        logger.warning("Could not init AnalogRAG schema: %s", exc)

    security_ids = _load_security_ids()
    existing = _existing_keys(str(DB_PATH))
    rows = _load_csv_rows(date_filter)

    eligible = [
        r for r in rows
        if r.get("pnl", "").strip()
        and r.get("direction", "") in ("BUY", "SELL")
    ]

    logger.info("Found %d/%d rows with PnL across all CSVs", len(eligible), len(rows))

    inserted = 0
    updated = 0
    skipped_dup = 0
    skipped_bad = 0

    conn = None if dry_run else sqlite3.connect(str(DB_PATH))

    for row in eligible:
        symbol    = row.get("symbol", "").strip()
        direction = row.get("direction", "").strip()
        ts_raw    = row.get("timestamp", "").strip()

        # Normalise timestamp to ISO format (used as dedup key)
        try:
            ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S").isoformat()
        except ValueError:
            ts = ts_raw  # keep whatever string is there

        if not symbol or not ts:
            skipped_bad += 1
            continue

        # Dedup: an existing (symbol, ts) is normally skipped. In --reenrich mode
        # we instead fall through to backfill the context columns the original
        # insert left blank (trend_15m / trend_1h / sector_trend).
        is_existing = (symbol, ts) in existing
        if is_existing and not reenrich:
            skipped_dup += 1
            continue

        try:
            pnl          = float(row["pnl"].strip())
            entry_price  = float(row.get("entry_price", "0").strip() or 0)
            exit_price_s = row.get("exit_price", "").strip()
            exit_price   = float(exit_price_s) if exit_price_s else None
            quantity     = int(float(row.get("quantity", "1").strip() or 1))
            confidence   = int(float(row.get("confidence", "0").strip() or 0))
            stop_loss    = float(row.get("stop_loss", "0").strip() or 0)
            reasoning    = row.get("reasoning", "")
            market_reg   = row.get("market_regime", "").strip().lower()
            sector_reg   = row.get("sector_regime", "").strip()
            # Context columns the original backfill never wrote. mtf_* are
            # UPPERCASE (match EnhancedIntradayBot._tf_trend). sector_trend is the
            # trend token after '=' in "SECTOR=TREND", lowercased to match the
            # live path (sector_data.trend is lowercase). matrix_score /
            # matrix_breakdown / candle_against / analog_wr are absent from the
            # CSV, so they stay at their column defaults.
            trend_15m    = row.get("mtf_15m", "").strip()
            trend_1h     = row.get("mtf_1h", "").strip()
            sector_trend = (sector_reg.split("=", 1)[1].strip().lower()
                            if "=" in sector_reg else "")
            # kronos_* are not in the current CSVs — write NULL (honest "unknown")
            # rather than a hardcoded 0 that analysis would read as "conflicted".
            kronos_dir   = row.get("kronos_direction", "").strip()
            _kal         = row.get("kronos_aligned", "").strip()
            _kpr         = row.get("kronos_pred_return", "").strip()
            kronos_aligned     = int(float(_kal)) if _kal else None
            kronos_pred_return = float(_kpr) if _kpr else None
            # Leading microstructure — present only in CSVs written after OFI
            # logging was added; older rows leave these NULL (honest "unknown").
            _ofi         = row.get("ofi", "").strip()
            _ofi_trend   = row.get("ofi_trend", "").strip()
            ofi          = float(_ofi) if _ofi else None
            ofi_trend    = float(_ofi_trend) if _ofi_trend else None
        except (ValueError, KeyError) as exc:
            logger.debug("Skip row %s %s — bad value: %s", symbol, ts_raw, exc)
            skipped_bad += 1
            continue

        # Candle-derived indicators (rsi/adx/volume_ratio/mfi/ofi/ofi_trend) —
        # computed once here so both the reenrich path and the new-insert path
        # below can use it. Rows from CSVs written before OFI logging existed
        # have no ofi/ofi_trend column at all; recomputing from the saved
        # candle file recovers it retroactively instead of leaving it NULL.
        ind = _candle_indicators(symbol, ts_raw, security_ids)
        if ind:
            if ofi is None and ind.get("ofi") is not None:
                ofi = float(ind["ofi"])
            if ofi_trend is None and ind.get("ofi_trend") is not None:
                ofi_trend = float(ind["ofi_trend"])

        # --reenrich: existing row — update only the derivable context columns,
        # never pnl/outcome/entry (the ground truth), and skip the indicator work.
        if is_existing:
            if dry_run:
                print(f"  [DRY-REENRICH] {symbol:12s} {ts_raw[:16]}  "
                      f"15m={trend_15m} 1h={trend_1h} sector={sector_trend or 'NULL'}"
                      f" ofi={ofi if ofi is not None else 'NULL'}")
            else:
                # COALESCE keeps any existing ofi if the CSV row has none, so a
                # re-enrich never wipes a previously captured value.
                conn.execute(
                    "UPDATE setups SET trend_15m=?, trend_1h=?, sector_trend=?, "
                    "ofi=COALESCE(?, ofi), ofi_trend=COALESCE(?, ofi_trend) "
                    "WHERE symbol=? AND ts=?",
                    (trend_15m, trend_1h, sector_trend, ofi, ofi_trend, symbol, ts))
            updated += 1
            continue

        # pnl_pct: prefer (exit-entry)/entry, fall back to pnl/(entry*qty)
        if exit_price and entry_price > 0:
            if direction == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
        elif entry_price > 0 and quantity > 0:
            pnl_pct = pnl / (entry_price * quantity) * 100
        else:
            pnl_pct = 0.0

        # Indicator values — try candle file first, then parse reasoning
        if ind:
            src = "candle"
            rsi          = float(ind.get("rsi", 50))
            adx          = float(ind.get("adx", 22))
            volume_ratio = float(ind.get("volume_ratio", 1.0))
            mfi          = float(ind.get("mfi", 50))
            # ATR still from stop_loss (more reliable than computed ATR on partial bars)
            atr_pct = (abs(entry_price - stop_loss) / entry_price * 100 / 1.5
                       if entry_price > 0 and stop_loss > 0 else 0.3)
        else:
            src = "reasoning"
            parsed = _extract_from_reasoning(reasoning, entry_price, stop_loss)
            rsi          = parsed["rsi"]
            adx          = parsed["adx"]
            volume_ratio = parsed["volume_ratio"]
            mfi          = parsed["mfi"]
            atr_pct      = parsed["atr_pct"]

        outcome = "WIN" if pnl > 0 else "LOSS"

        if dry_run:
            print(f"  [DRY] {symbol:12s} {direction:4s} {ts_raw[:16]}  "
                  f"pnl={pnl:+.2f} ({pnl_pct:+.2f}%)  outcome={outcome}  "
                  f"RSI={rsi:.0f} ADX={adx:.0f} vol={volume_ratio:.2f} "
                  f"MFI={mfi:.0f} ATR%={atr_pct:.3f}  src={src}")
        else:
            try:
                conn.execute("""
                    INSERT INTO setups
                        (ts, symbol, rsi, adx, volume_ratio, mfi, atr_pct,
                         kronos_aligned, kronos_direction, kronos_pred_return,
                         nifty_trend, market_regime,
                         signal_type, confidence, pnl, pnl_pct, outcome,
                         trend_15m, trend_1h, sector_trend,
                         ofi, ofi_trend)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ts, symbol,
                    rsi, adx, volume_ratio, mfi, atr_pct,
                    kronos_aligned, kronos_dir, kronos_pred_return,
                    market_reg, sector_reg,
                    direction, confidence,
                    pnl, round(pnl_pct, 4), outcome,
                    trend_15m, trend_1h, sector_trend,
                    ofi, ofi_trend,
                ))
                existing.add((symbol, ts))  # prevent duplicate within this run
                logger.info("Inserted: %s %s %s pnl=%.2f outcome=%s src=%s",
                            symbol, direction, ts_raw[:16], pnl, outcome, src)
            except sqlite3.IntegrityError:
                skipped_dup += 1
                continue

        inserted += 1

    if not dry_run and conn:
        conn.commit()
        conn.close()

    logger.info(
        "Done. inserted=%d  updated=%d  skipped_dup=%d  skipped_bad=%d",
        inserted, updated, skipped_dup, skipped_bad,
    )
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill RAG DB from signal CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show rows that would be inserted without writing")
    parser.add_argument("--date", default=None,
                        help="Only process a specific date (e.g. 2026-06-10)")
    parser.add_argument("--reenrich", action="store_true",
                        help="Also update existing rows' context columns "
                             "(trend_15m/trend_1h/sector_trend) from the CSVs")
    args = parser.parse_args()

    n = backfill(dry_run=args.dry_run, date_filter=args.date, reenrich=args.reenrich)
    if args.dry_run:
        print(f"\nDry run complete — {n} rows would be inserted"
              f"{' (existing rows re-enriched separately above)' if args.reenrich else ''}.")
    else:
        print(f"\nBackfill complete — {n} new rows inserted into {DB_PATH}.")
