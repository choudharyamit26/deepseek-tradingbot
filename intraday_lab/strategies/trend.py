"""CONTROL GROUP part 1 (s01-s04): classic, context-free strategies.

The opencode lab already ran this menu on the same data/window with realistic
costs and got 0/5 profitable walk-forward folds on every one (gross edge ~0,
costs ~-Rs140/trade). They stay in this lab as engine cross-validation: if
these DON'T lose here too, one of the two engines is broken.
"""
import numpy as np
from strategies.base import (Strategy, register, ema, adx, cross_up, cross_dn,
                             sig_array)


@register
class S01_ORB(Strategy):
    name = "s01_orb_ctl"
    space = {"n_or": [3, 6], "buf_bps": [0, 10],
             "sl_atr": [1.0, 1.5], "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        n_or = p["n_or"]
        buf = 1 + p["buf_bps"] / 10000
        or_h = df["high"].where(df["bar_no"] < n_or).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < n_or).groupby(df["day"]).transform("min")
        ok = (df["bar_no"] >= n_or) & (df["tod"] <= 13 * 60)
        lvl_h, lvl_l = or_h * buf, or_l / buf
        long = ok & (df["close"] > lvl_h) & (df["close"].shift(1) <= lvl_h)
        short = ok & (df["close"] < lvl_l) & (df["close"].shift(1) >= lvl_l)
        return sig_array(df, long, short)


@register
class S02_VwapPullback(Strategy):
    name = "s02_vwap_pullback_ctl"
    space = {"band_bps": [5, 15], "sl_atr": [1.0, 1.5], "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        band = p["band_bps"] / 10000
        up = ema(df["close"], 20) > ema(df["close"], 50)
        long = up & (df["low"] <= df["vwap"] * (1 + band)) & \
            (df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))
        short = ~up & (df["high"] >= df["vwap"] * (1 - band)) & \
            (df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))
        return sig_array(df, long, short)


@register
class S03_EmaAdx(Strategy):
    name = "s03_ema_adx_ctl"
    space = {"fast": [5, 9], "slow": [21, 34], "adx_th": [20, 25],
             "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        f, s = ema(df["close"], p["fast"]), ema(df["close"], p["slow"])
        trending = adx(df) > p["adx_th"]
        return sig_array(df, cross_up(f, s) & trending, cross_dn(f, s) & trending)


@register
class S04_Donchian(Strategy):
    name = "s04_donchian_ctl"
    space = {"n": [20, 40], "sl_atr": [1.0, 1.5], "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        hh = df["high"].rolling(p["n"]).max().shift(1)
        ll = df["low"].rolling(p["n"]).min().shift(1)
        long = (df["close"] > hh) & (df["close"].shift(1) <= hh.shift(1))
        short = (df["close"] < ll) & (df["close"].shift(1) >= ll.shift(1))
        return sig_array(df, long, short)
