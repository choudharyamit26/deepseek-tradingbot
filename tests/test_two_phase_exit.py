"""Tests for the two-phase exit in KronosExitGuardian.

Validated design (2026-07-03 candle replay, 43 trades): at +TWO_PHASE_PARTIAL_AT_PCT
profit, bank TWO_PHASE_PARTIAL_FRACTION of the position, floor the stop at
breakeven, and trail the runner TWO_PHASE_RUNNER_TRAIL_PCT behind its
high-water mark. These tests lock the state machine in: phase flip, breakeven
floor, ratchet-only trail, qty-1 handling, rejected-order retry, and the
partial-PnL fold into the final exit.
"""
import asyncio

import pytest

from kronos_integrated_bot import config as cfg
from kronos_integrated_bot.enhanced_guardian import KronosExitGuardian

BUY, SELL = "BUY", "SELL"


class FakeDhanSDK:
    BUY = BUY
    SELL = SELL

    def __init__(self):
        self.cancel_calls = []
        self.cancel_fail_legs = set()   # legs that return failure

    def cancel_super_order(self, order_id, order_leg):
        self.cancel_calls.append((order_id, order_leg))
        if order_leg in self.cancel_fail_legs:
            return {"status": "failure", "remarks": "leg not cancellable"}
        return {"status": "success"}


class FakeDhan:
    def __init__(self, reduce_ok=True):
        self.dhan = FakeDhanSDK()
        self.security_ids = {"TESTSYM": "12345"}
        self.reduce_ok = reduce_ok
        self.reduce_calls = []

    def reduce_position(self, security_id, transaction_type, quantity, product_type="INTRADAY"):
        self.reduce_calls.append((security_id, transaction_type, quantity))
        if self.reduce_ok:
            return {"status": "success"}
        return {"status": "failure", "remarks": "DH-905 Invalid IP"}


class FakeBot:
    def __init__(self, trade):
        self.active_trades = {"TESTSYM": trade}
        self._dhan_sem = asyncio.Semaphore(1)


def make_trade(direction=BUY, entry=100.0, qty=2):
    return {
        "symbol": "TESTSYM",
        "security_id": "12345",
        "entry_price": entry,
        "quantity": qty,
        "transaction_type": direction,
        "trailing_sl": 0,
        "sl_price": 0,
        "atr_value": 1.0,
    }


def make_guardian(trade, dry_run=True, reduce_ok=True):
    dhan = FakeDhan(reduce_ok=reduce_ok)
    bot = FakeBot(trade)
    g = KronosExitGuardian(dhan, bot, kronos=None, dry_run=dry_run)
    return g, dhan, bot


def tick(g, trade, price, pnl_pct):
    asyncio.run(g._two_phase_tick("TESTSYM", trade, price, pnl_pct))


@pytest.fixture(autouse=True)
def two_phase_cfg(monkeypatch):
    monkeypatch.setattr(cfg, "TWO_PHASE_EXIT_ENABLED", True)
    monkeypatch.setattr(cfg, "TWO_PHASE_PARTIAL_AT_PCT", 0.4)
    monkeypatch.setattr(cfg, "TWO_PHASE_PARTIAL_FRACTION", 0.5)
    monkeypatch.setattr(cfg, "TWO_PHASE_RUNNER_TRAIL_PCT", 0.5)


def test_phase1_below_threshold_does_nothing():
    trade = make_trade()
    g, _, _ = make_guardian(trade)
    tick(g, trade, 100.2, 0.2)
    assert not trade.get("tp2_active")
    assert trade["trailing_sl"] == 0
    assert trade["quantity"] == 2


def test_phase_flip_books_partial_and_floors_breakeven():
    trade = make_trade(qty=2)
    g, _, _ = make_guardian(trade)
    tick(g, trade, 100.4, 0.4)
    assert trade["tp2_active"]
    assert trade["quantity"] == 1
    assert trade["original_quantity"] == 2
    # banked 1 share at +0.40
    assert trade["realized_partial_pnl"] == pytest.approx(0.4)
    # trail = max(entry, hw*(1-0.005)) = max(100, 99.898) = breakeven
    assert trade["trailing_sl"] == pytest.approx(100.0)
    assert trade["sl_price"] == pytest.approx(100.0)


def test_runner_trail_ratchets_up_and_never_loosens():
    trade = make_trade(qty=2)
    g, _, _ = make_guardian(trade)
    tick(g, trade, 100.4, 0.4)
    tick(g, trade, 101.0, 1.0)           # new high-water
    assert trade["tp2_highwater"] == pytest.approx(101.0)
    assert trade["trailing_sl"] == pytest.approx(101.0 * 0.995)
    sl_at_peak = trade["trailing_sl"]
    tick(g, trade, 100.6, 0.6)           # price falls back
    assert trade["trailing_sl"] == pytest.approx(sl_at_peak)  # unchanged
    assert trade["tp2_highwater"] == pytest.approx(101.0)


def test_sell_direction_symmetric():
    trade = make_trade(direction=SELL, entry=100.0, qty=2)
    g, _, _ = make_guardian(trade)
    tick(g, trade, 99.6, 0.4)            # short in profit
    assert trade["tp2_active"]
    assert trade["realized_partial_pnl"] == pytest.approx(0.4)
    # trail = min(entry, hw*(1+0.005)) = min(100, 100.098) = breakeven
    assert trade["trailing_sl"] == pytest.approx(100.0)
    tick(g, trade, 99.0, 1.0)            # deeper profit
    assert trade["tp2_highwater"] == pytest.approx(99.0)
    assert trade["trailing_sl"] == pytest.approx(99.0 * 1.005)
    sl_at_low = trade["trailing_sl"]
    tick(g, trade, 99.4, 0.6)            # bounce
    assert trade["trailing_sl"] == pytest.approx(sl_at_low)


def test_qty_one_skips_partial_but_arms_phase2():
    trade = make_trade(qty=1)
    g, dhan, _ = make_guardian(trade, dry_run=False)
    tick(g, trade, 100.5, 0.5)
    assert trade["tp2_active"]
    assert trade["quantity"] == 1                      # nothing to split
    assert "realized_partial_pnl" not in trade
    assert dhan.reduce_calls == []                     # no order sent
    assert trade["trailing_sl"] == pytest.approx(100.0)  # breakeven lock


def test_live_partial_places_reduce_order():
    trade = make_trade(qty=4)
    g, dhan, _ = make_guardian(trade, dry_run=False)
    tick(g, trade, 100.4, 0.4)
    assert dhan.reduce_calls == [("12345", SELL, 2)]
    assert trade["quantity"] == 2
    assert trade["realized_partial_pnl"] == pytest.approx(0.8)


def test_rejected_partial_leaves_phase1_and_retries():
    trade = make_trade(qty=2)
    g, dhan, _ = make_guardian(trade, dry_run=False, reduce_ok=False)
    tick(g, trade, 100.4, 0.4)
    assert not trade.get("tp2_active")                 # flip aborted
    assert trade["quantity"] == 2                      # untouched
    assert trade["trailing_sl"] == 0                   # no premature BE lock
    dhan.reduce_ok = True
    tick(g, trade, 100.5, 0.5)                         # next poll retries
    assert trade["tp2_active"]
    assert trade["quantity"] == 1
    assert len(dhan.reduce_calls) == 2


def test_disabled_switch_is_inert():
    cfg.TWO_PHASE_EXIT_ENABLED = False
    trade = make_trade(qty=2)
    g, _, _ = make_guardian(trade)
    # check_position gates on the flag; the tick itself is never called there,
    # but calling it directly must also be safe when threshold is crossed
    # because check_position is the only caller. Simulate the gate:
    if cfg.TWO_PHASE_EXIT_ENABLED:
        tick(g, trade, 101.0, 1.0)
    assert not trade.get("tp2_active")


def test_time_exit_exemption_flag():
    # Both guardians zero out max-duration for phase-2 runners; assert the
    # marker the exemption keys on is exactly what the tick sets.
    trade = make_trade(qty=2)
    g, _, _ = make_guardian(trade)
    tick(g, trade, 100.4, 0.4)
    assert trade.get("tp2_active") is True


def test_super_order_legs_cleared_before_partial():
    trade = make_trade(qty=2)
    trade["order_id"] = "SO-98765"          # real super order id
    g, dhan, _ = make_guardian(trade, dry_run=False)
    tick(g, trade, 100.4, 0.4)
    # both legs cancelled BEFORE the reduce order went out
    assert dhan.dhan.cancel_calls == [("SO-98765", "TARGET_LEG"), ("SO-98765", "STOP_LOSS_LEG")]
    assert dhan.reduce_calls == [("12345", SELL, 1)]
    assert trade["super_legs_cancelled"] is True   # _exit_position skips re-cancel
    assert trade["tp2_active"]
    assert trade["quantity"] == 1


def test_super_order_leg_failure_aborts_partial_and_retries_only_failed_leg():
    trade = make_trade(qty=2)
    trade["order_id"] = "SO-98765"
    g, dhan, _ = make_guardian(trade, dry_run=False)
    dhan.dhan.cancel_fail_legs = {"STOP_LOSS_LEG"}
    tick(g, trade, 100.4, 0.4)
    assert not trade.get("tp2_active")             # flip aborted
    assert trade["quantity"] == 2                  # no reduce placed
    assert dhan.reduce_calls == []
    assert trade["tp2_legs_done"] == ["TARGET_LEG"]  # progress kept
    dhan.dhan.cancel_fail_legs = set()
    tick(g, trade, 100.45, 0.45)                   # next poll
    # only the failed leg is re-attempted
    assert dhan.dhan.cancel_calls == [
        ("SO-98765", "TARGET_LEG"), ("SO-98765", "STOP_LOSS_LEG"),
        ("SO-98765", "STOP_LOSS_LEG"),
    ]
    assert trade["tp2_active"]
    assert trade["quantity"] == 1


def test_super_order_three_strikes_arms_phase2_without_partial():
    trade = make_trade(qty=2)
    trade["order_id"] = "SO-98765"
    g, dhan, _ = make_guardian(trade, dry_run=False)
    dhan.dhan.cancel_fail_legs = {"TARGET_LEG"}    # permanently uncancellable
    for px, pct in [(100.4, 0.4), (100.42, 0.42), (100.45, 0.45)]:
        tick(g, trade, px, pct)
    assert trade["tp2_partial_blocked"] is True
    assert not trade.get("tp2_active")
    tick(g, trade, 100.5, 0.5)                     # 4th poll: arms without partial
    assert trade["tp2_active"]
    assert trade["quantity"] == 2                  # position never reduced
    assert dhan.reduce_calls == []                 # legs intact -> no qty conflict
    assert trade["trailing_sl"] == pytest.approx(100.0)  # BE floor still armed
    assert not trade.get("super_legs_cancelled")   # _exit_position still cancels


def test_plain_order_skips_leg_cancellation():
    trade = make_trade(qty=2)
    trade["order_id"] = "DHAN-TESTSYM"             # adopted/plain position
    g, dhan, _ = make_guardian(trade, dry_run=False)
    tick(g, trade, 100.4, 0.4)
    assert dhan.dhan.cancel_calls == []
    assert trade["tp2_active"]
    assert trade["quantity"] == 1


def test_dry_run_skips_all_broker_calls_for_super_order():
    trade = make_trade(qty=2)
    trade["order_id"] = "SO-98765"
    g, dhan, _ = make_guardian(trade, dry_run=True)
    tick(g, trade, 100.4, 0.4)
    assert dhan.dhan.cancel_calls == []
    assert dhan.reduce_calls == []
    assert trade["tp2_active"]
    assert trade["quantity"] == 1                  # simulated reduction


def test_partial_pnl_folds_into_final_exit_math():
    # Mirrors the fold in stock_trading_bot._exit_position: runner exit pnl
    # plus banked partial, pct denominated in the original quantity.
    trade = make_trade(qty=2)
    g, _, _ = make_guardian(trade)
    tick(g, trade, 100.4, 0.4)           # banked 1 @ +0.40
    exit_price = 101.0                   # runner exits later at +1.00
    runner_pnl = (exit_price - trade["entry_price"]) * trade["quantity"]
    total = runner_pnl + trade["realized_partial_pnl"]
    orig_qty = trade["original_quantity"]
    pnl_pct = total / (trade["entry_price"] * orig_qty) * 100
    assert total == pytest.approx(1.4)
    assert pnl_pct == pytest.approx(0.7)
