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
    ):
        """Persist a completed trade setup with its outcome."""
        outcome = "WIN" if pnl > 0 else "LOSS"
        kronos_aligned = 0
        kronos_direction = ""
        if kronos_conf:
            kronos_aligned = 0 if kronos_conf.get("conflict") else 1
            kronos_direction = str(kronos_conf.get("pred_direction", ""))

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
        )
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO setups
                    (ts, symbol, rsi, adx, volume_ratio, mfi, atr_pct,
                     kronos_aligned, kronos_direction, nifty_trend,
                     market_regime, signal_type, confidence, pnl, pnl_pct, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)
            conn.commit()
            conn.close()
            logger.debug("AnalogRAG stored: %s %s -> %s (P&L=%.2f)", symbol, signal_type, outcome, pnl)
        except Exception as exc:
            logger.warning("AnalogRAG store failed: %s", exc)

    # ── read ──────────────────────────────────────────────────────────────────

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
            return ""

        if len(rows) < 3:
            return ""

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

        top_rows = [rows[i] for i in top_idx]
        winners = [r for r in top_rows if r[10] == "WIN"]

        win_rate = len(winners) / len(top_rows) * 100 if top_rows else 0
        avg_pnl = float(np.mean([r[8] for r in top_rows])) if top_rows else 0.0

        lines = [f"ANALOG SETUPS ({len(top_rows)} most similar past trades):"]
        for rank, (idx, row) in enumerate(zip(top_idx, top_rows), 1):
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
