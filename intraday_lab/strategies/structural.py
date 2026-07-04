"""Structural / time-of-day family (s19-s20)."""
import numpy as np
from strategies.base import Strategy, register, ofi, sig_array


@register
class S19_AfternoonTrend(Strategy):
    """At 14:00, join a day that has already trended, ride into the close."""
    name = "s19_afternoon_trend"
    space = {"d_pct": [0.5, 1.0], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.5, 0.8]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        dret = (df["close"] / df["day_open"] - 1) * 100
        at2pm = (df["tod"] >= 14 * 60) & (df["tod"].shift(1) < 14 * 60)
        long = at2pm & (dret > p["d_pct"]) & (df["close"] > df["vwap"]) & (f > 0)
        short = at2pm & (dret < -p["d_pct"]) & (df["close"] < df["vwap"]) & (f < 0)
        return sig_array(df, long, short)


@register
class S20_PdhRetest(Strategy):
    """Break of the previous day's high/low that HOLDS for b bars (retest
    filter kills the one-bar fakeout that plagued naive breakout systems)."""
    name = "s20_pdh_retest"
    space = {"hold": [2, 3], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        pdh = df["day"].map(g["high"].max().shift(1))
        pdl = df["day"].map(g["low"].min().shift(1))
        b = p["hold"]
        above = (df["close"] > pdh).rolling(b).sum() == b
        below = (df["close"] < pdl).rolling(b).sum() == b
        ok = df["tod"] >= 10 * 60 + 15
        long = ok & above & ~above.shift(1).fillna(False)
        short = ok & below & ~below.shift(1).fillna(False)
        return sig_array(df, long, short)
