"""EMA fast/slow + RSI momentum-zone strategy (user-requested study).

Base spec: EMA13 crosses above EMA34 with RSI(14) in (60, 80) -> long.
Two trigger styles (both close-of-bar, no lookahead):
  - "ema_cross": the EMA cross bar itself, RSI already in the zone
  - "rsi_entry": EMAs already aligned, RSI crosses INTO the zone from below
Short variant mirrors everything (RSI zone reflected to 100-hi .. 100-lo).

Optional confirmation filters (round 2 of the study), applied on the signal
bar's close:
  - use_vwap: longs need close > vwap, shorts close < vwap
  - adx_th:   ADX(14) > threshold; 0 disables
  - use_st:   supertrend(10, 3) direction must agree with the trade side

rsi_hi=101 means "no upper cap" so the grid can test whether the 80 ceiling
helps or just throws away the strongest trends.

NOT @register'ed: these run through their own study script
(ema_rsi_study.py), not the batch pipeline.
"""
import numpy as np
import pandas as pd

from strategies.base import (Strategy, adx, cross_dn, cross_up, ema, rsi,
                             sig_array, supertrend_dir)

_ST_CACHE: dict = {}   # (symbol, ts0, ts1, n) -> int8 direction array


def _supertrend(df, ctx):
    key = (ctx.get("symbol", "?"), df.index[0], df.index[-1], len(df))
    if key not in _ST_CACHE:
        _ST_CACHE[key] = supertrend_dir(df, 10, 3.0).values.astype(np.int8)
    return _ST_CACHE[key]


def _signals(df, p, side, ctx):
    f = ema(df["close"], p["fast"])
    s = ema(df["close"], p["slow"])
    r = rsi(df["close"], p.get("rsi_len", 14))
    lo, hi = p["rsi_lo"], p["rsi_hi"]
    ok_time = df["tod"] >= p.get("entry_after", 570)

    if side > 0:
        if p["trigger"] == "ema_cross":
            raw = cross_up(f, s) & (r >= lo) & (r <= hi)
        else:  # rsi_entry: trend already up, RSI pushes into the zone
            raw = (f > s) & (r >= lo) & (r.shift(1) < lo) & (r <= hi)
    else:
        m_lo, m_hi = 100 - hi, 100 - lo   # mirrored zone
        if p["trigger"] == "ema_cross":
            raw = cross_dn(f, s) & (r >= m_lo) & (r <= m_hi)
        else:
            raw = (f < s) & (r <= m_hi) & (r.shift(1) > m_hi) & (r >= m_lo)

    raw = raw & ok_time
    if p.get("use_vwap", 0):
        raw = raw & ((df["close"] > df["vwap"]) if side > 0
                     else (df["close"] < df["vwap"]))
    if p.get("adx_th", 0):
        raw = raw & (adx(df) > p["adx_th"])
    if p.get("use_st", 0):
        raw = raw & pd.Series(_supertrend(df, ctx) == side, index=df.index)

    none = raw & False
    return sig_array(df, raw if side > 0 else none, none if side > 0 else raw)


class EmaRsiLong(Strategy):
    name = "ema_rsi_long"
    space = {
        "fast": [9, 13, 21],
        "slow": [34, 55],
        "rsi_lo": [55, 60, 65],
        "rsi_hi": [80, 101],
        "trigger": ["ema_cross", "rsi_entry"],
        "tp_mode": ["atr", "2ph"],
    }

    def generate(self, df, p, ctx):
        return _signals(df, p, +1, ctx)


class EmaRsiShort(Strategy):
    name = "ema_rsi_short"
    space = dict(EmaRsiLong.space)

    def generate(self, df, p, ctx):
        return _signals(df, p, -1, ctx)


class EmaRsiFilteredLong(EmaRsiLong):
    """Round-2 grid: EMA/RSI core (best round-1 region) x filter toggles."""
    name = "ema_rsi_filt_long"
    space = {
        "fast": [9, 13],
        "slow": [34],
        "rsi_lo": [60, 65],
        "rsi_hi": [80, 101],
        "trigger": ["ema_cross"],
        "use_vwap": [0, 1],
        "adx_th": [0, 25],
        "use_st": [0, 1],
        "tp_mode": ["atr", "2ph"],
    }


class EmaRsiFilteredShort(EmaRsiShort):
    name = "ema_rsi_filt_short"
    space = dict(EmaRsiFilteredLong.space)
