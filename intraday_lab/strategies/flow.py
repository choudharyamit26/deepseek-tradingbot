"""Order-flow family (s16-s18): OFI as the primary signal, not a filter."""
import numpy as np
from strategies.base import Strategy, register, ofi, sig_array

LATE = 10 * 60 + 15


@register
class S16_OfiMomentum(Strategy):
    name = "s16_ofi_momentum"
    space = {"th": [0.25, 0.4], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        ok = df["tod"] >= LATE
        long = ok & (f > p["th"]) & (f.shift(1) <= p["th"]) & (df["close"] > df["vwap"])
        short = ok & (f < -p["th"]) & (f.shift(1) >= -p["th"]) & (df["close"] < df["vwap"])
        return sig_array(df, long, short)


@register
class S17_VolSpikeOfi(Strategy):
    """Participation spike + decisive bar + flow all pointing the same way."""
    name = "s17_vol_spike_ofi"
    space = {"vol_x": [2.5, 3.5], "body_min": [0.5, 0.65],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        volx = df["volume"] / df["volume"].rolling(10).mean().clip(lower=1)
        rng = (df["high"] - df["low"]).clip(lower=1e-9)
        body = (df["close"] - df["open"]) / rng
        ok = (df["tod"] >= LATE) & (volx > p["vol_x"])
        long = ok & (body > p["body_min"]) & (f > 0) & (df["close"] > df["vwap"])
        short = ok & (body < -p["body_min"]) & (f < 0) & (df["close"] < df["vwap"])
        return sig_array(df, long, short)


@register
class S18_OfiDivReversal(Strategy):
    """Price prints a fresh 40-bar extreme but flow never confirmed — reverse."""
    name = "s18_ofi_div_reversal"
    space = {"n": [40, 60], "sl_atr": [1.0, 1.5], "tp_atr": [1.5, 2.5]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        hh = df["high"].rolling(p["n"]).max().shift(1)
        ll = df["low"].rolling(p["n"]).min().shift(1)
        ok = df["tod"] >= LATE
        short = ok & (df["high"] > hh) & (f < 0) & (df["close"] < df["open"])
        long = ok & (df["low"] < ll) & (f > 0) & (df["close"] > df["open"])
        return sig_array(df, long, short)
