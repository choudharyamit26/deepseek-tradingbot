from indicators import calculate_technical_indicators

EXPECTED_KEYS = {
    "close", "high", "low", "rsi", "macd", "macd_signal", "sma_20", "ema_9",
    "bb_percent_b", "atr", "vwap", "vwap_distance_pct", "volume_ratio",
    "support", "resistance", "adx", "mfi",
}


def test_returns_all_keys(ohlcv):
    ind = calculate_technical_indicators(ohlcv)
    assert set(ind.keys()) == EXPECTED_KEYS


def test_sane_ranges(ohlcv):
    ind = calculate_technical_indicators(ohlcv)
    assert 0 <= ind["rsi"] <= 100
    assert 0 <= ind["mfi"] <= 100
    assert ind["atr"] > 0
    assert ind["support"] <= ind["close"] <= ind["resistance"] * 1.01
    assert abs(ind["vwap_distance_pct"]) < 50


def test_short_frame_returns_empty(ohlcv):
    assert calculate_technical_indicators(ohlcv.head(3)) == {}
    assert calculate_technical_indicators(None) == {}


def test_input_not_mutated(ohlcv):
    cols_before = list(ohlcv.columns)
    calculate_technical_indicators(ohlcv)
    assert list(ohlcv.columns) == cols_before
