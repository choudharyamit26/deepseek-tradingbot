"""Regression tests for backfill_rag.py — specifically the OFI (order-flow
imbalance) carry from the signals CSV into analog_history.db.

OFI reaches the DB only through this path when orders are placed manually (the
live store_setup path never runs then). These tests lock that carry in so it
can't silently break.
"""
import csv
import sqlite3

import pytest

import backfill_rag
from signal_logger import _CSV_FIELDS


def _write_csv(path, rows, fields=_CSV_FIELDS):
    """Write a signals CSV with the given field list and row dicts."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _base_row(**overrides):
    row = {
        "timestamp": "2026-07-02 09:45:00",
        "symbol": "TESTSYM",
        "signal_type": "ENTRY-SHORT",
        "direction": "SELL",
        "entry_price": "100.00",
        "exit_price": "101.00",
        "quantity": "1",
        "stop_loss": "99.00",
        "trailing_stop": "99.00",
        "target": "97.00",
        "confidence": "90",
        "reasoning": "RSI 40 ADX 25 test setup",
        "pnl": "-1.00",
        "mode": "LIVE",
        "market_regime": "NEUTRAL",
        "sector_regime": "X=NEUTRAL",
        "mtf_3m": "BEARISH",
        "mtf_15m": "BEARISH",
        "mtf_1h": "BEARISH",
        "kronos_direction": "SELL",
        "kronos_pred_return": "-0.100",
        "kronos_aligned": "1",
        "ofi": "-0.4200",
        "ofi_trend": "-0.1500",
    }
    row.update(overrides)
    return row


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect backfill_rag's LOG_DIR and DB_PATH to a temp sandbox."""
    logdir = tmp_path / "trading_logs"
    logdir.mkdir()
    db = tmp_path / "analog_history.db"
    monkeypatch.setattr(backfill_rag, "LOG_DIR", logdir)
    monkeypatch.setattr(backfill_rag, "DB_PATH", db)
    return logdir, db


def _fetch_ofi(db, symbol="TESTSYM"):
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT ofi, ofi_trend FROM setups WHERE symbol=?", (symbol,)
    ).fetchone()
    conn.close()
    return row


def test_ofi_carried_csv_to_db(env):
    """A CSV row with ofi/ofi_trend lands in the DB as the correct floats."""
    logdir, db = env
    _write_csv(logdir / "signals_2026-07-02.csv", [_base_row()])

    inserted = backfill_rag.backfill(date_filter="2026-07-02")

    assert inserted == 1
    assert _fetch_ofi(db) == pytest.approx((-0.42, -0.15))


def test_missing_ofi_columns_null(env):
    """An older CSV lacking the ofi columns inserts NULL, not 0 (honest unknown)."""
    logdir, db = env
    legacy_fields = [f for f in _CSV_FIELDS if f not in ("ofi", "ofi_trend")]
    _write_csv(logdir / "signals_2026-07-02.csv", [_base_row()], fields=legacy_fields)

    inserted = backfill_rag.backfill(date_filter="2026-07-02")

    assert inserted == 1
    assert _fetch_ofi(db) == (None, None)


def test_reenrich_preserves_existing_ofi(env):
    """Re-enrich with a blank-ofi CSV must not wipe a previously captured value."""
    logdir, db = env
    csv_path = logdir / "signals_2026-07-02.csv"

    # First pass: insert with OFI populated.
    _write_csv(csv_path, [_base_row()])
    backfill_rag.backfill(date_filter="2026-07-02")
    assert _fetch_ofi(db) == pytest.approx((-0.42, -0.15))

    # Second pass: same row but OFI now blank, run with --reenrich.
    _write_csv(csv_path, [_base_row(ofi="", ofi_trend="")])
    backfill_rag.backfill(date_filter="2026-07-02", reenrich=True)

    # COALESCE keeps the original value rather than nulling it.
    assert _fetch_ofi(db) == pytest.approx((-0.42, -0.15))
