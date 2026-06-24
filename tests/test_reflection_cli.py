"""Tests for the operator-agent tool surface (reflection_cli).

These lock in the guardrails that an external agent (Hermes) must not be able to
bypass: the evidence gate, the locked-param veto, and bounds/step clamping. The
data loaders are monkeypatched so nothing here touches the real DB, strategy
file, or network.
"""
from types import SimpleNamespace

import pytest

from kronos_integrated_bot import reflect
from kronos_integrated_bot import reflection_cli


STRATEGY = {
    "version": "kronos-vTEST",
    "goal": {"min_trades_per_change": 30, "min_eval_trades": 20},
    "params": {"min_adx_trending": 22, "risk_per_trade_pct": 2.0},
}


def _patch(monkeypatch, *, n_trades, hypotheses=None):
    """Make _load_context return a deterministic window with n_trades closed."""
    trades = [{"pnl": 1.0, "market_regime": "", "direction": "SELL"}
              for _ in range(n_trades)]
    monkeypatch.setattr(reflect, "load_strategy", lambda: dict(STRATEGY))
    monkeypatch.setattr(reflect, "load_hypotheses", lambda: list(hypotheses or []))
    monkeypatch.setattr(reflect, "last_change_time", lambda h: None)
    monkeypatch.setattr(reflect, "load_closed_trades_from_db",
                        lambda **k: list(trades))
    monkeypatch.setattr(reflect, "load_closed_trades", lambda **k: [])
    # replay over an empty data store returns None (skip); be explicit anyway.
    monkeypatch.setattr(reflect, "replay_validate", lambda *a, **k: None)


def test_state_reports_gate_closed(monkeypatch):
    _patch(monkeypatch, n_trades=17)
    out = reflection_cli.cmd_state(SimpleNamespace())
    assert out["ok"] is True
    assert out["evidence_gate"]["open"] is False
    assert out["evidence_gate"]["closed_trades_since_last_change"] == 17


def test_apply_refused_when_gate_closed(monkeypatch):
    _patch(monkeypatch, n_trades=17)
    args = SimpleNamespace(param="min_adx_trending", value=24.0,
                           reason="x", confirm=True)
    out = reflection_cli.cmd_apply(args)
    assert out["ok"] is False
    assert out["refused"] == "evidence_gate_closed"


def test_apply_refused_for_locked_param(monkeypatch):
    # Gate open, but risk_per_trade_pct is a locked capital-protection knob.
    _patch(monkeypatch, n_trades=40)
    args = SimpleNamespace(param="risk_per_trade_pct", value=1.5,
                           reason="x", confirm=True)
    out = reflection_cli.cmd_apply(args)
    assert out["ok"] is False
    assert out["refused"] == "invalid_proposal"


def test_propose_clamps_to_max_step(monkeypatch):
    # min_adx_trending current=22, max_step=2, max=30: a jump to 30 must clamp
    # to 24 (current + step), and the clamp must be reported.
    _patch(monkeypatch, n_trades=40)
    args = SimpleNamespace(param="min_adx_trending", value=30.0, replay=False)
    out = reflection_cli.cmd_propose(args)
    assert out["ok"] is True
    assert out["valid"] is True
    assert out["clamped_value"] == 24
    assert out["was_clamped"] is True
    assert out["replay_ran"] is False


def test_apply_dry_preview_without_confirm(monkeypatch):
    _patch(monkeypatch, n_trades=40)
    args = SimpleNamespace(param="min_adx_trending", value=24.0,
                           reason="tighten trend gate", confirm=False)
    out = reflection_cli.cmd_apply(args)
    assert out["ok"] is True
    assert out["applied"] is False
    assert out["would_apply"] is True
    assert out["new_value"] == 24


# ── Read-only DB query tool ──────────────────────────────────────────────────
def test_query_simple_select():
    out = reflection_cli.cmd_query(SimpleNamespace(sql="SELECT 1 AS x, 2 AS y", limit=10))
    assert out["ok"] is True
    assert out["columns"] == ["x", "y"]
    assert out["rows"] == [{"x": 1, "y": 2}]


def test_query_allows_cte():
    out = reflection_cli.cmd_query(
        SimpleNamespace(sql="WITH t AS (SELECT 1 AS a) SELECT a FROM t", limit=10))
    assert out["ok"] is True
    assert out["rows"] == [{"a": 1}]


@pytest.mark.parametrize("sql", [
    "DELETE FROM setups",
    "UPDATE setups SET pnl=0",
    "DROP TABLE setups",
    "INSERT INTO setups (id) VALUES (999)",
    "SELECT 1; DROP TABLE setups",
])
def test_query_refuses_non_readonly(sql):
    out = reflection_cli.cmd_query(SimpleNamespace(sql=sql, limit=10))
    assert out["ok"] is False
    assert "error" in out


def test_query_truncates_to_limit():
    # setups has many rows; a limit of 1 must return exactly one and flag truncation.
    out = reflection_cli.cmd_query(SimpleNamespace(sql="SELECT * FROM setups", limit=1))
    assert out["ok"] is True
    assert out["row_count"] == 1
    assert out["truncated"] is True


def test_schema_reports_tables_and_fill():
    out = reflection_cli.cmd_schema(SimpleNamespace())
    assert out["ok"] is True
    assert "setups" in out["tables"]
    cols = {c["name"]: c for c in out["tables"]["setups"]["columns"]}
    assert "pnl" in cols and cols["pnl"]["fill_pct"] == 100.0
    assert out["setups_summary"]["ts_range"][0] is not None
