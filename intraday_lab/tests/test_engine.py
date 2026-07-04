"""Engine honesty tests: known-answer trade, cost math, no-lookahead guard."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from engine import backtester, costs
from strategies import REGISTRY


def synth_day(n_days=3, base=100.0, drift=0.0, seed=7):
    """Synthetic 5-min sessions 09:15-15:25, 75 bars/day."""
    rng = np.random.default_rng(seed)
    frames = []
    px = base
    for d in range(n_days):
        ts = pd.date_range(f"2026-01-{5 + d:02d} 09:15", periods=75, freq="5min")
        rets = rng.normal(drift, 0.001, 75)
        close = px * np.cumprod(1 + rets)
        open_ = np.concatenate([[px], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        vol = rng.integers(1000, 5000, 75).astype(float)
        frames.append(pd.DataFrame({"open": open_, "high": high, "low": low,
                                    "close": close, "volume": vol}, index=ts))
        px = close[-1]
    return backtester.prepare(pd.concat(frames))


def test_known_answer_tp_hit():
    """Hand-built path: entry at next open 100, ATR-bracket TP at +2*ATR must
    fill exactly at the bracket with correct net math."""
    ts = pd.date_range("2026-01-05 09:15", periods=75, freq="5min")
    open_ = np.full(75, 100.0)
    close = np.full(75, 100.0)
    high = np.full(75, 100.4)
    low = np.full(75, 99.8)
    # bar 30 rockets so TP (2*ATR above entry) trades intrabar
    high[30] = 103.0
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": np.full(75, 1000.0)}, index=ts)
    df = backtester.prepare(df)
    sig = np.zeros(75, dtype=np.int8)
    sig[20] = 1                                   # entry at open of bar 21
    tr = backtester.run(df, sig, {"sl_atr": 5.0, "tp_atr": 2.0}, "T")
    assert len(tr) == 1
    t = tr.iloc[0]
    atr = df["atr"].values[20]
    entry = costs.slip(100.0, True)
    tp = entry + 2.0 * atr
    assert t["reason"] == "TP"
    assert t["exit"] == pytest.approx(costs.slip(tp, False), abs=0.01)
    expected_net = (t["exit"] - t["entry"]) * t["qty"] - costs.round_trip(
        t["entry"], t["exit"], t["qty"])
    assert t["net"] == pytest.approx(expected_net, abs=0.5)


def test_two_phase_banks_partial_and_trails():
    ts = pd.date_range("2026-01-05 09:15", periods=75, freq="5min")
    base = np.full(75, 100.0)
    df = pd.DataFrame({"open": base.copy(), "high": base * 1.001,
                       "low": base * 0.999, "close": base.copy(),
                       "volume": np.full(75, 1000.0)}, index=ts)
    df.iloc[30:40, df.columns.get_loc("high")] = 101.5   # runs +1.5%
    df.iloc[45, df.columns.get_loc("low")] = 99.0        # then collapses
    df = backtester.prepare(df)
    sig = np.zeros(75, dtype=np.int8)
    sig[20] = 1
    tr = backtester.run(df, sig, {"sl_atr": 8.0, "tp_mode": "2ph",
                                  "partial_pct": 0.4, "trail_pct": 0.5}, "T")
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["reason"] == "TRAIL"
    assert t["net"] > 0        # banked partial + BE-floored runner can't lose here


def test_sl_before_tp_same_bar():
    """When one bar touches both bracket sides, the loss is taken (SL first)."""
    ts = pd.date_range("2026-01-05 09:15", periods=75, freq="5min")
    base = np.full(75, 100.0)
    df = pd.DataFrame({"open": base.copy(), "high": base * 1.0005,
                       "low": base * 0.9995, "close": base.copy(),
                       "volume": np.full(75, 1000.0)}, index=ts)
    df.iloc[25, df.columns.get_loc("high")] = 110.0
    df.iloc[25, df.columns.get_loc("low")] = 90.0
    df = backtester.prepare(df)
    sig = np.zeros(75, dtype=np.int8)
    sig[20] = 1
    tr = backtester.run(df, sig, {"sl_atr": 1.0, "tp_atr": 1.0}, "T")
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "SL"


def test_no_lookahead_all_strategies():
    """Truncating the future must not change past signals, for every strategy."""
    df = synth_day(n_days=5)
    cut = len(df) - 80
    for strat in REGISTRY:
        p = strat.default_params()
        full = np.asarray(strat.generate(df, p, {}))
        part = np.asarray(strat.generate(df.iloc[:cut], p, {}))
        same = (full[:cut] == part).mean()
        assert same > 0.995, f"{strat.name} leaks future data ({same:.3f} match)"


def test_costs_sane():
    rt = costs.round_trip(100.0, 101.0, 1000)     # 1L notional each side
    assert 60 < rt < 200                          # realistic Indian intraday
    assert costs.side_cost(100_000, is_buy=True) < costs.side_cost(100_000, is_buy=False)


def test_registry_has_20():
    assert len(REGISTRY) == 68, [s.name for s in REGISTRY]
