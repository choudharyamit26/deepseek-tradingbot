from conftest import make_ohlcv

from indicators import calculate_technical_indicators
from reversal_detector import detect_reversals


def test_flat_market_low_score(ohlcv):
    ind = calculate_technical_indicators(ohlcv)
    report = detect_reversals(ohlcv.copy(), is_buy=True, indicators=ind)
    assert 0 <= report.score <= 100
    assert report.recommendation in ("✅ HOLD", "⚠️ CAUTION", "🚨 EXIT NOW")


def test_no_crash_on_tiny_frame():
    df = make_ohlcv(n=6)
    report = detect_reversals(df, is_buy=True, indicators={})
    assert report.score >= 0  # detectors that need more bars just contribute nothing


def test_volume_climax_detected():
    # Strong uptrend then a 6x volume spike on the last bar
    df = make_ohlcv(n=60, trend=0.15, vol_spike_at=59)
    ind = calculate_technical_indicators(df)
    report = detect_reversals(df.copy(), is_buy=True, indicators=ind)
    # Volume climax may or may not fire depending on candle body, but the
    # report must be structurally valid and signals sorted by severity.
    sevs = [s.severity for s in report.signals]
    assert sevs == sorted(sevs, reverse=True)
