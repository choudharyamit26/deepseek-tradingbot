"""No-lookahead + signal-sanity for the ema_rsi study strategies (not in
REGISTRY, so the batch-wide guard doesn't cover them)."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategies.ema_rsi import (EmaRsiFilteredLong, EmaRsiFilteredShort,
                                EmaRsiLong, EmaRsiShort)
from tests.test_engine import synth_day


def test_no_lookahead_ema_rsi():
    df = synth_day(n_days=5)
    cut = len(df) - 80
    strats = (EmaRsiLong(), EmaRsiShort(),
              EmaRsiFilteredLong(), EmaRsiFilteredShort())
    for strat, trig in itertools.product(strats, ("ema_cross", "rsi_entry")):
        p = {**strat.default_params(), "trigger": trig,
             "use_vwap": 1, "adx_th": 20, "use_st": 1}
        full = np.asarray(strat.generate(df, p, {}))
        part = np.asarray(strat.generate(df.iloc[:cut], p, {}))
        same = (full[:cut] == part).mean()
        assert same > 0.995, f"{strat.name}/{trig} leaks future ({same:.3f})"


def test_supertrend_flips_both_ways():
    """Regression: NaN ATR warm-up used to freeze the direction at +1."""
    from strategies.base import supertrend_dir
    df = synth_day(n_days=5)
    d = np.unique(supertrend_dir(df, 10, 3.0))
    assert set(d) == {1, -1}, f"supertrend stuck: {d}"


def test_sides_are_one_directional():
    # dip then rally: forces an EMA cross-up mid-series with elevated RSI
    import pandas as pd
    down = synth_day(n_days=2, drift=-0.001, seed=3)
    up = synth_day(n_days=3, drift=0.0008, seed=4,
                   base=float(down["close"].iloc[-1]))
    up.index = up.index + pd.Timedelta(days=7)
    from engine import backtester
    df = backtester.prepare(pd.concat([down, up])[
        ["open", "high", "low", "close", "volume"]])
    fired = 0
    for trig in ("ema_cross", "rsi_entry"):
        long = np.asarray(EmaRsiLong().generate(
            df, {**EmaRsiLong().default_params(), "trigger": trig}, {}))
        short = np.asarray(EmaRsiShort().generate(
            df, {**EmaRsiShort().default_params(), "trigger": trig}, {}))
        assert set(np.unique(long)) <= {0, 1}
        assert set(np.unique(short)) <= {0, -1}
        fired += long.any()
    assert fired, "no long signals on dip-then-rally synth"
