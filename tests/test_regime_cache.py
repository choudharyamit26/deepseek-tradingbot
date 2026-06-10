import time

from regime_filter import RegimeFilter


class _StubSdk:
    def __init__(self):
        self.daily_calls = 0
        self.quote_calls = 0
        self.INDEX = "IDX"

    def historical_daily_data(self, **kw):
        self.daily_calls += 1
        return {"status": "success",
                "data": {"open": [100] * 15, "high": [101] * 15, "low": [99] * 15,
                         "close": [100 + i * 0.1 for i in range(15)],
                         "volume": [1000] * 15,
                         "timestamp": [time.time() - 86400 * (15 - i) for i in range(15)]}}

    def quote_data(self, securities=None):
        self.quote_calls += 1
        return {"status": "success",
                "data": {"data": {"IDX_I": {"13": {"last_price": 101.5}}}}}


class _StubDhan:
    def __init__(self):
        self.dhan = _StubSdk()


def test_daily_and_live_fetches_are_cached():
    rf = RegimeFilter(_StubDhan())
    r1 = rf.get_regime("RELIANCE")
    daily_after_first = rf.dhan.dhan.daily_calls
    quote_after_first = rf.dhan.dhan.quote_calls
    assert r1["nifty"]["trend"] in ("bullish", "bearish", "neutral")

    # 50 more symbols in the same scan must not refetch index data
    for _ in range(50):
        rf.get_regime("RELIANCE")
    assert rf.dhan.dhan.daily_calls == daily_after_first
    assert rf.dhan.dhan.quote_calls == quote_after_first


def test_cache_expires():
    rf = RegimeFilter(_StubDhan())
    rf.DAILY_CACHE_TTL = 0
    rf.LIVE_PRICE_TTL = 0
    rf.get_regime("RELIANCE")
    rf.get_regime("RELIANCE")
    assert rf.dhan.dhan.daily_calls >= 2
