from conftest import make_ohlcv

from kronos_integrated_bot.replay import (
    entry_direction,
    passes_prefilter,
    run_replay,
    simulate_day,
)

PARAMS = {
    "min_adx_trending": 18,
    "min_prefilter_volume_ratio": 0.15,
    "min_prefilter_atr_pct": 0.05,
    "min_volume_ratio_trending": 0.1,
    "min_rr_ratio": 1.8,
    "rsi_ob_limit": 70,
    "rsi_os_limit": 30,
    "max_signals_per_stock_per_day": 2,
    "max_daily_signals": 10,
    "cooldown_seconds": 0,
    "trailing_sl_activation_pct": 3.0,
    "trailing_sl_distance_atr": 2.0,
    "max_trade_duration_minutes": 180,
    "market_close_exit_minutes": 15,
}


def test_prefilter_gates():
    good = {"adx": 25, "volume_ratio": 1.0, "atr": 1.0, "close": 100, "rsi": 55}
    assert passes_prefilter(good, PARAMS)
    assert not passes_prefilter({**good, "adx": 10}, PARAMS)
    assert not passes_prefilter({**good, "rsi": 80}, PARAMS)
    assert not passes_prefilter({**good, "volume_ratio": 0.01}, PARAMS)


def test_entry_direction():
    buy_ind = {"close": 105, "vwap": 100, "sma_20": 101, "rsi": 60, "volume_ratio": 1.0}
    assert entry_direction(buy_ind, PARAMS) == "BUY"
    sell_ind = {"close": 95, "vwap": 100, "sma_20": 99, "rsi": 45, "volume_ratio": 1.0}
    assert entry_direction(sell_ind, PARAMS) == "SELL"
    flat_ind = {"close": 100, "vwap": 100, "sma_20": 100, "rsi": 50, "volume_ratio": 1.0}
    assert entry_direction(flat_ind, PARAMS) is None


def test_simulate_day_closes_all_positions():
    df = make_ohlcv(n=100, trend=0.1)
    trades = simulate_day("TEST", df, PARAMS, {"total_entries": 0})
    for t in trades:
        assert t["exit"] is not None
        assert t["exit_reason"] in ("SL", "TP", "TIME", "EOD", "REVERSAL", "EOD-FORCED")
        assert t["direction"] in ("BUY", "SELL")


def test_run_replay_no_data(tmp_path):
    metrics, trades = run_replay(["2020-01-01"], PARAMS, data_dir=tmp_path)
    assert metrics["closed_trades"] == 0
    assert trades == []
