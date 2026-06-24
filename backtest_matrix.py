"""
backtest_matrix.py — Validate the matrix-authoritative confidence against REAL fills.

Recomputes _compute_score_matrix (the now-authoritative confidence) for every
closed trade in analog_history.db, adds the point-in-time analog penalty, then
reports:
  [A] win-rate / PnL by recomputed-score band
  [B] pass-rate / win% / PnL at candidate MIN_CONFIDENCE thresholds
  [C] per-penalty contribution (does each penalty actually mark losers?)
  [D] same, split by direction (BUY / SELL)

SCOPE / HONESTY: analog_history.db stores only the 3m indicators
(rsi/adx/volume_ratio/mfi/atr_pct) plus nifty_trend — it does NOT store the
15m/1h trends, VWAP/SMA, sector, intraday move, or the entry candles. So the
recomputed matrix here covers the *dominant* deterministic penalties
(volume / ADX / MFI / volume-bonus) + analog. MTF, regime, and candle penalties
are NOT reconstructable from the DB and are therefore excluded. This is a
calibration sketch on the score's biggest movers, not the full live matrix.

Run:  python backtest_matrix.py            (uses analog_history.db)
      python backtest_matrix.py --db path  (alternate DB)
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from kronos_integrated_bot.enhanced_bot import EnhancedIntradayBot as E  # noqa: E402

MIN_CONFIDENCE_LIVE = 82  # current kronos_strategy.yaml value


def load_fills(db_path: str) -> list[dict]:
    c = sqlite3.connect(db_path)
    rows = c.execute("""
        SELECT ts, symbol, rsi, adx, volume_ratio, mfi, atr_pct,
               nifty_trend, signal_type, confidence, pnl, pnl_pct, outcome
        FROM setups
        WHERE outcome IS NOT NULL
        ORDER BY ts ASC
    """).fetchall()
    c.close()
    fills = []
    for r in rows:
        (ts, sym, rsi, adx, vol, mfi, atr_pct, nifty, sig, conf,
         pnl, pnl_pct, outcome) = r
        fills.append({
            "ts": ts, "symbol": sym,
            "rsi": rsi, "adx": adx, "volume_ratio": vol, "mfi": mfi,
            "atr_pct": atr_pct, "nifty_trend": nifty, "signal_type": sig,
            "old_conf": conf, "pnl": pnl or 0.0, "pnl_pct": pnl_pct or 0.0,
            "outcome": outcome, "win": outcome == "WIN",
        })
    return fills


def point_in_time_analog(fills: list[dict], idx: int, std: np.ndarray, n: int = 5):
    """Top-n analog win-rate using ONLY earlier fills (no look-ahead)."""
    cur = fills[idx]
    earlier = fills[:idx]
    if len(earlier) < 3:
        return None
    qv = np.array([cur["rsi"], cur["adx"], cur["volume_ratio"],
                   cur["mfi"], cur["atr_pct"]], dtype=float) / std
    dists = []
    for e in earlier:
        ev = np.array([e["rsi"], e["adx"], e["volume_ratio"],
                       e["mfi"], e["atr_pct"]], dtype=float) / std
        dists.append(np.linalg.norm(ev - qv))
    order = np.argsort(dists)[:n]
    top = [earlier[i] for i in order]
    wins = sum(1 for t in top if t["win"])
    return wins / len(top) * 100


def recompute(fill: dict, analog_wr):
    """Recompute the DB-reconstructable part of the matrix-authoritative score.
    Returns (score, fired) where fired is a dict of penalty -> points."""
    ind = {
        "rsi": fill["rsi"], "adx": fill["adx"],
        "volume_ratio": fill["volume_ratio"], "mfi": fill["mfi"],
        # close/sma20/vwap unknown -> trend_3m/price_above_vwap default NEUTRAL,
        # 15m/1h empty -> NEUTRAL, so MTF/regime-bonus contribute nothing here.
    }
    regime = {"nifty": {"trend": (fill["nifty_trend"] or "neutral")}, "sector": None}
    score, breakdown = E._compute_score_matrix(ind, regime, {}, {}, None)

    fired = {}
    # parse the breakdown for per-penalty attribution
    for part in breakdown.split(" | "):
        if "vol=" in part and "-" in part:
            fired["volume"] = part
        elif "ADX=" in part:
            fired["adx"] = part
        elif "MFI=" in part:
            fired["mfi"] = part

    # analog (matches _analyze): <35 -> -10, >=65 -> +5
    if analog_wr is not None and score > 0:
        if analog_wr < 35:
            score -= 10
            fired["analog"] = f"analog {analog_wr:.0f}%<35: -10"
        elif analog_wr >= 65:
            score += 5
            fired["analog"] = f"analog {analog_wr:.0f}%>=65: +5"
    return max(0, min(100, score)), fired


def pct(w, n):
    return f"{w/n*100:4.0f}%" if n else "  - "


def report_bands(fills):
    print("\n[A] WIN-RATE / PnL BY RECOMPUTED-SCORE BAND")
    print(f"  {'band':>10} | {'n':>4} | {'win%':>5} | {'tot pnl':>9} | {'avg pnl':>8}")
    print("  " + "-" * 52)
    bands = [(0, 76), (76, 80), (80, 82), (82, 84), (84, 86), (86, 101)]
    for lo, hi in bands:
        grp = [f for f in fills if lo <= f["score"] < hi]
        if not grp:
            continue
        n = len(grp)
        w = sum(f["win"] for f in grp)
        tot = sum(f["pnl"] for f in grp)
        print(f"  {f'{lo}-{hi-1}':>10} | {n:>4} | {pct(w,n)} | {tot:>9.1f} | {tot/n:>8.2f}")


def report_thresholds(fills, label=""):
    print(f"\n[B] PASS-RATE / WIN% / PnL AT CANDIDATE MIN_CONFIDENCE {label}")
    print(f"  {'thresh':>6} | {'pass':>4} | {'win%':>5} | {'tot pnl':>9} | {'avg pnl':>8}")
    print("  " + "-" * 50)
    for t in [76, 78, 80, 82, 84, 86]:
        grp = [f for f in fills if f["score"] >= t]
        n = len(grp)
        if not n:
            print(f"  {t:>6} | {0:>4} |   -   |       -  |      - ")
            continue
        w = sum(f["win"] for f in grp)
        tot = sum(f["pnl"] for f in grp)
        star = "  <- live" if t == MIN_CONFIDENCE_LIVE else ""
        print(f"  {t:>6} | {n:>4} | {pct(w,n)} | {tot:>9.1f} | {tot/n:>8.2f}{star}")


def report_penalties(fills):
    print("\n[C] PER-PENALTY CONTRIBUTION (did the penalty actually mark losers?)")
    print(f"  {'penalty':>14} | {'fired n':>7} | {'win% fired':>10} | {'win% NOT':>9}")
    print("  " + "-" * 52)

    def show(label, pred):
        fired = [f for f in fills if pred(f)]
        notf = [f for f in fills if not pred(f)]
        if not fired:
            return
        wf = sum(f["win"] for f in fired)
        wn = sum(f["win"] for f in notf)
        print(f"  {label:>14} | {len(fired):>7} | {pct(wf,len(fired))}      | {pct(wn,len(notf))}")

    show("volume(-)", lambda f: "volume" in f["fired"])
    show("adx(-)", lambda f: "adx" in f["fired"])
    show("mfi(-)", lambda f: "mfi" in f["fired"])
    show("analog -10", lambda f: "<35" in f["fired"].get("analog", ""))
    show("analog +5", lambda f: ">=65" in f["fired"].get("analog", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "analog_history.db"))
    args = ap.parse_args()

    fills = load_fills(args.db)
    print(f"Loaded {len(fills)} real fills with outcomes from {Path(args.db).name}")
    if len(fills) < 5:
        print("Not enough fills to analyze.")
        return

    feats = np.array([[f["rsi"], f["adx"], f["volume_ratio"], f["mfi"], f["atr_pct"]]
                      for f in fills], dtype=float)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0

    for i, f in enumerate(fills):
        wr = point_in_time_analog(fills, i, std)
        f["analog_wr"] = wr
        f["score"], f["fired"] = recompute(f, wr)

    base_w = sum(f["win"] for f in fills)
    base_pnl = sum(f["pnl"] for f in fills)
    print(f"Baseline (all fills): win% {base_w/len(fills)*100:.0f}  "
          f"total pnl {base_pnl:.1f}  avg {base_pnl/len(fills):.2f}")

    report_bands(fills)
    report_thresholds(fills, "(all)")
    report_penalties(fills)

    for d in ("SELL", "BUY"):
        grp = [f for f in fills if f["signal_type"] == d]
        if len(grp) >= 5:
            print(f"\n[D] DIRECTION = {d}  (n={len(grp)})")
            report_thresholds(grp, f"({d})")


if __name__ == "__main__":
    main()
