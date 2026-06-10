from risk_manager import RiskManager


def make_rm(**kw):
    return RiskManager(initial_capital=100000, **kw)


def test_position_size_basic():
    rm = make_rm(risk_per_trade_percent=2)
    # risk 2000, SL 1% of 500 = 5/share -> 400 shares, capped at 200 affordable
    qty = rm.calculate_position_size(100000, stop_loss_percent=1.0, entry_price=500)
    assert qty == 200  # affordability cap binds


def test_position_size_risk_cap_binds():
    rm = make_rm(risk_per_trade_percent=1)
    # risk 1000, SL 2% of 100 = 2/share -> 500 shares; affordable 1000 -> 500
    assert rm.calculate_position_size(100000, 2.0, 100) == 500


def test_position_size_invalid_inputs():
    rm = make_rm()
    assert rm.calculate_position_size(100000, 0, 100) == 0
    assert rm.calculate_position_size(100000, -1, 100) == 0
    assert rm.calculate_position_size(100000, 1, 0) == 0


def test_daily_limits():
    rm = make_rm(max_daily_trades=2, max_daily_loss_percent=2)
    assert rm.check_daily_trade_limit()
    rm.record_trade()
    rm.record_trade()
    assert not rm.check_daily_trade_limit()

    assert rm.check_daily_loss_limit()
    rm.record_pnl(-2500)  # -2.5% of 100k
    assert not rm.check_daily_loss_limit()

    rm.reset_daily()
    assert rm.daily_trade_count == 0 and rm.daily_pnl == 0
