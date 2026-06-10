import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_ohlcv(n=60, start_price=100.0, trend=0.0, vol_spike_at=None, seed=42):
    """Synthetic 3-min OHLCV frame with a tz-aware intraday index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-10 09:15", periods=n, freq="3min", tz="Asia/Kolkata")
    closes = start_price + np.cumsum(rng.normal(trend, 0.3, n))
    opens = np.concatenate([[start_price], closes[:-1]])
    highs = np.maximum(opens, closes) + rng.uniform(0.05, 0.3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.05, 0.3, n)
    volume = rng.uniform(8000, 12000, n)
    if vol_spike_at is not None:
        volume[vol_spike_at] *= 6
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv()
