"""
SQLite-backed analog retrieval system.

Stores completed trade setups (technical features + outcome) and queries
the most similar past setups before each new trade decision, giving the
AI model concrete historical evidence rather than just aggregate stats.
"""
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parent / "analog_history.db"


class AnalogRAG:
    """
    Feature vector: [rsi, adx, volume_ratio, mfi, atr_pct]
    Similarity: Euclidean distance in stddev-normalised feature space.
    """

    FEATURE_COLS = ["rsi", "adx", "volume_ratio", "mfi", "atr_pct"]

    def __init__(self, db_path=None):
        self.db_path = str(db_path or _DEFAULT_DB)
        self._init_db()

    # ── schema ────────────────────────────────────────────────────────────────

    # Columns added after the original schema. Persisted at entry so a backtest
    # can reconstruct the FULL matrix-authoritative confidence (the 3m-only
    # columns above cannot reconstruct MTF/regime/candle penalties). Migrated
    # onto existing DBs via ALTER TABLE (new rows fill them; old rows stay NULL).
    _EXTRA_COLUMNS = (
        ("matrix_score",     "INTEGER"),
        ("matrix_breakdown", "TEXT    DEFAULT ''"),
        ("trend_15m",        "TEXT    DEFAULT ''"),
        ("trend_1h",         "TEXT    DEFAULT ''"),
        ("sector_trend",     "TEXT    DEFAULT ''"),
        ("candle_against",   "INTEGER DEFAULT 0"),
        ("analog_wr",        "REAL"),
        ("kronos_pred_return", "REAL"),  # signed forecast return (fraction); +ve=bullish
    )

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS setups (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                rsi             REAL,
                adx             REAL,
                volume_ratio    REAL,
                mfi             REAL,
                atr_pct         REAL,
                kronos_aligned  INTEGER DEFAULT 0,
                kronos_direction TEXT   DEFAULT '',
                nifty_trend     TEXT    DEFAULT '',
                market_regime   TEXT    DEFAULT '',
                signal_type     TEXT    NOT NULL,
                confidence      INTEGER,
                pnl             REAL,
                pnl_pct         REAL,
                outcome         TEXT
            )
        """)
        existing = {r[1] for r in conn.execute("PRAGMA table_info(setups)")}
        for col, decl in self._EXTRA_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE setups ADD COLUMN {col} {decl}")
                logger.info("AnalogRAG DB migrated: added column %s", col)
        conn.commit()
        conn.close()
        logger.debug("AnalogRAG DB ready: %s", self.db_path)

    # ── write ─────────────────────────────────────────────────────────────────

    def store_setup(
        self,
        symbol: str,
        indicators: dict,
        kronos_conf: dict | None,
        nifty_trend: str,
        market_regime: str,
        signal_type: str,
        confidence: int,
        pnl: float,
        pnl_pct: float,
        matrix_score: int = 0,
        matrix_breakdown: str = "",
        trend_15m: str = "",
        trend_1h: str = "",
        sector_trend: str = "",
        candle_against: bool = False,
        analog_wr: float | None = None,
    ):
        """Persist a completed trade setup with its outcome.

        The matrix_* / trend_* / sector / candle / analog_wr fields capture the
        full matrix-authoritative decision context at entry so a later backtest
        can reconstruct the complete confidence (not just the 3m-only part)."""
        outcome = "WIN" if pnl > 0 else "LOSS"
        kronos_aligned = 0
        kronos_direction = ""
        kronos_pred_return = None
        if kronos_conf:
            kronos_aligned = 0 if kronos_conf.get("conflict") else 1
            kronos_direction = str(kronos_conf.get("pred_direction", ""))
            kronos_pred_return = float(kronos_conf.get("pred_return", 0.0))

        row = (
            datetime.utcnow().isoformat(),
            symbol,
            float(indicators.get("rsi", 50)),
            float(indicators.get("adx", 20)),
            float(indicators.get("volume_ratio", 1.0)),
            float(indicators.get("mfi", 50)),
            float(indicators.get("atr_pct", 0.5)),
            int(kronos_aligned),
            kronos_direction,
            str(nifty_trend),
            str(market_regime),
            signal_type,
            int(confidence),
            float(pnl),
            float(pnl_pct),
            outcome,
            int(matrix_score),
            str(matrix_breakdown),
            str(trend_15m),
            str(trend_1h),
            str(sector_trend),
            int(bool(candle_against)),
            (float(analog_wr) if analog_wr is not None else None),
            kronos_pred_return,
        )
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO setups
                    (ts, symbol, rsi, adx, volume_ratio, mfi, atr_pct,
                     kronos_aligned, kronos_direction, nifty_trend,
                     market_regime, signal_type, confidence, pnl, pnl_pct, outcome,
                     matrix_score, matrix_breakdown, trend_15m, trend_1h,
                     sector_trend, candle_against, analog_wr, kronos_pred_return)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)
            conn.commit()
            conn.close()
            logger.debug("AnalogRAG stored: %s %s -> %s (P&L=%.2f)", symbol, signal_type, outcome, pnl)
        except Exception as exc:
            logger.warning("AnalogRAG store failed: %s", exc)

    # ── read ──────────────────────────────────────────────────────────────────

    def _similar_rows(self, indicators: dict, n: int) -> list:
        """Return the n most similar stored setups (raw DB rows) by normalized
        Euclidean distance. Empty list if fewer than 3 setups are stored or on
        any DB error. Shared by query_similar (formatting) and analog_stats
        (numeric scoring) so both use identical neighbour selection."""
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute("""
                SELECT rsi, adx, volume_ratio, mfi, atr_pct,
                       kronos_aligned, signal_type, confidence,
                       pnl, pnl_pct, outcome, symbol, ts
                FROM setups
                WHERE outcome IS NOT NULL
                ORDER BY ts DESC LIMIT 1000
            """).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("AnalogRAG query failed: %s", exc)
            return []

        if len(rows) < 3:
            return []

        query_vec = np.array([
            indicators.get("rsi", 50),
            indicators.get("adx", 20),
            indicators.get("volume_ratio", 1.0),
            indicators.get("mfi", 50),
            indicators.get("atr_pct", 0.5),
        ], dtype=float)

        db_vecs = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
        feat_std = db_vecs.std(axis=0)
        feat_std[feat_std < 1e-6] = 1.0

        query_norm = query_vec / feat_std
        db_norm = db_vecs / feat_std
        dists = np.linalg.norm(db_norm - query_norm, axis=1)
        top_idx = np.argsort(dists)[:n]
        return [rows[i] for i in top_idx]

    def analog_stats(self, indicators: dict, n: int = 5) -> dict:
        """Numeric analog evidence for programmatic confidence scoring.

        Returns {"n": int, "win_rate": float|None, "avg_pnl": float}.
        win_rate is None when there is insufficient history (< 3 setups), so
        callers can distinguish 'no signal' from a genuine 0% win rate."""
        top_rows = self._similar_rows(indicators, n)
        if not top_rows:
            return {"n": 0, "win_rate": None, "avg_pnl": 0.0}
        winners = [r for r in top_rows if r[10] == "WIN"]
        return {
            "n": len(top_rows),
            "win_rate": len(winners) / len(top_rows) * 100,
            "avg_pnl": float(np.mean([r[8] for r in top_rows])),
        }

    def query_similar(
        self,
        indicators: dict,
        kronos_conf: dict | None = None,
        nifty_trend: str = "",
        market_regime: str = "",
        signal_type: str = "",
        n: int = 5,
    ) -> str:
        """
        Return a formatted string of the n most similar past setups.
        Empty string if fewer than 3 setups are stored.
        """
        top_rows = self._similar_rows(indicators, n)
        if not top_rows:
            return ""

        winners = [r for r in top_rows if r[10] == "WIN"]

        win_rate = len(winners) / len(top_rows) * 100 if top_rows else 0
        avg_pnl = float(np.mean([r[8] for r in top_rows])) if top_rows else 0.0

        lines = [f"ANALOG SETUPS ({len(top_rows)} most similar past trades):"]
        for rank, row in enumerate(top_rows, 1):
            rsi, adx, vol, mfi, atr, kal, sig, conf, pnl, pnl_pct, outcome, sym, ts = row
            lines.append(
                f"  [{rank}] {sym} {sig} RSI={rsi:.0f} ADX={adx:.0f} Vol={vol:.2f}"
                f" Kronos={'ALIGN' if kal else 'CONF'}"
                f" conf={conf} -> {outcome} P&L={pnl:+.2f} ({pnl_pct:+.2f}%)"
            )

        lines.append(
            f"  Analog win rate: {win_rate:.0f}% ({len(winners)}/{len(top_rows)})"
            f"  avg P&L={avg_pnl:+.2f}"
        )
        if win_rate < 35:
            lines.append("  ANALOG WARNING: Similar setups lose money historically. Use extra caution.")
        elif win_rate >= 65:
            lines.append("  ANALOG EDGE: Similar setups win historically. Slight confidence boost justified.")

        return "\n".join(lines)

    # ── utility ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            n = conn.execute("SELECT COUNT(*) FROM setups WHERE outcome IS NOT NULL").fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0

    def recent_stats(self, last_n=50) -> dict:
        """Return win/loss stats for the last N completed trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute("""
                SELECT outcome, pnl FROM setups
                WHERE outcome IS NOT NULL
                ORDER BY ts DESC LIMIT ?
            """, (last_n,)).fetchall()
            conn.close()
        except Exception:
            return {}
        if not rows:
            return {}
        wins = [r for r in rows if r[0] == "WIN"]
        return {
            "total": len(rows),
            "wins": len(wins),
            "losses": len(rows) - len(wins),
            "win_rate": len(wins) / len(rows) * 100,
            "total_pnl": sum(r[1] for r in rows),
        }
