"""Conditioned strategies (s07-s15): the classics' entry ideas, but gated by
the live-bot's empirically validated filters — OFI confirmation (62% vs 25% WR
on shorts), post-10:15 entries (market-open decay), context alignment — and
exited via the two-phase scheme (partial + breakeven + trail) that flipped the
live sample from -1.4% to +7% in candle replay. tp_mode='2ph' engages it.
"""
import numpy as np
from strategies.base import (Strategy, register, ema, adx, rsi, bollinger,
                             keltner, ofi, sig_array)

LATE = 10 * 60 + 15   # 10:15 — no fresh entries before this (validated bleed)


@register
class S07_OrbOfi(Strategy):
    """ORB break that order flow actually confirms, after the open chop."""
    name = "s07_orb_ofi_2ph"
    space = {"n_or": [6, 12], "ofi_th": [0.15, 0.3], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        or_h = df["high"].where(df["bar_no"] < p["n_or"]).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < p["n_or"]).groupby(df["day"]).transform("min")
        ok = (df["bar_no"] >= p["n_or"]) & (df["tod"] >= LATE) & (df["tod"] <= 14 * 60)
        long = ok & (df["close"] > or_h) & (df["close"].shift(1) <= or_h) & (f > p["ofi_th"])
        short = ok & (df["close"] < or_l) & (df["close"].shift(1) >= or_l) & (f < -p["ofi_th"])
        return sig_array(df, long, short)


@register
class S08_VwapReclaimOfi(Strategy):
    """VWAP reclaim with flow + participation behind it."""
    name = "s08_vwap_reclaim_ofi"
    space = {"ofi_th": [0.15, 0.3], "vol_x": [1.2, 1.8], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        volx = df["volume"] / df["volume"].rolling(10).mean().clip(lower=1)
        ok = df["tod"] >= LATE
        up = (df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))
        dn = (df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))
        long = ok & up & (f > p["ofi_th"]) & (volx > p["vol_x"])
        short = ok & dn & (f < -p["ofi_th"]) & (volx > p["vol_x"])
        return sig_array(df, long, short)


@register
class S09_TrendDayRider(Strategy):
    """Established trend day (move since open) + right side of VWAP + flow."""
    name = "s09_trend_day_rider"
    space = {"d_pct": [0.5, 0.8], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        dret = (df["close"] / df["day_open"] - 1) * 100
        ok = df["tod"] >= LATE
        long = ok & (dret > p["d_pct"]) & (dret.shift(1) <= p["d_pct"]) & \
            (df["close"] > df["vwap"]) & (f > 0)
        short = ok & (dret < -p["d_pct"]) & (dret.shift(1) >= -p["d_pct"]) & \
            (df["close"] < df["vwap"]) & (f < 0)
        return sig_array(df, long, short)


@register
class S10_NiftyRS(Strategy):
    """Intraday relative strength vs NIFTY: trade the outlier, with VWAP side."""
    name = "s10_nifty_rs"
    space = {"th": [0.4, 0.7], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        nifty = ctx.get("nifty")
        if nifty is None:
            return np.zeros(len(df), dtype=np.int8)
        nc = nifty["close"].reindex(df.index).ffill()
        no = nifty["day_open"].reindex(df.index).ffill()
        rs = ((df["close"] / df["day_open"]) - (nc / no)) * 100
        ok = df["tod"] >= LATE
        long = ok & (rs > p["th"]) & (rs.shift(1) <= p["th"]) & (df["close"] > df["vwap"])
        short = ok & (rs < -p["th"]) & (rs.shift(1) >= -p["th"]) & (df["close"] < df["vwap"])
        return sig_array(df, long, short)


@register
class S11_SqueezeOfi(Strategy):
    """BB-inside-KC compression, resolved in the direction flow confirms."""
    name = "s11_squeeze_ofi"
    space = {"m": [6, 10], "ofi_th": [0.15, 0.3],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        blo, _, bhi = bollinger(df["close"], 20, 2.0)
        klo, _, khi = keltner(df, 20, 1.5)
        squeeze = (bhi < khi) & (blo > klo)
        primed = squeeze.rolling(p["m"]).sum() >= p["m"]
        ok = df["tod"] >= LATE
        long = ok & primed.shift(1).fillna(False) & (df["close"] > bhi) & (f > p["ofi_th"])
        short = ok & primed.shift(1).fillna(False) & (df["close"] < blo) & (f < -p["ofi_th"])
        return sig_array(df, long, short)


@register
class S12_GapFadeConfirm(Strategy):
    """Fade a gap only after it demonstrably fails to extend (bar-3 check),
    with flow already leaning against the gap."""
    name = "s12_gap_fade_confirm"
    space = {"g": [0.5, 1.0], "sl_atr": [1.0, 1.5], "tp_atr": [1.5, 2.5]}

    def generate(self, df, p, ctx):
        gap = (df["day_open"] / df["prev_close"] - 1) * 100
        f = ofi(df, 3)
        bar1_h = df["high"].where(df["bar_no"] == 0).groupby(df["day"]).transform("max")
        bar1_l = df["low"].where(df["bar_no"] == 0).groupby(df["day"]).transform("min")
        at3 = df["bar_no"] == 3
        short = at3 & (gap >= p["g"]) & (df["close"] < bar1_h) & (f < 0)
        long = at3 & (gap <= -p["g"]) & (df["close"] > bar1_l) & (f > 0)
        return sig_array(df, long, short)


@register
class S13_VwapZOfiDiv(Strategy):
    """Price stretched off VWAP while flow disagrees with the stretch — fade."""
    name = "s13_vwapz_ofi_div"
    space = {"z_th": [1.5, 2.0], "adx_max": [20, 25],
             "sl_atr": [1.0, 1.5], "tp_atr": [1.0, 2.0]}

    def generate(self, df, p, ctx):
        dev = df["close"] - df["vwap"]
        z = dev / dev.rolling(30).std().replace(0, np.nan)
        f = ofi(df)
        quiet = adx(df) < p["adx_max"]
        ok = df["tod"] >= LATE
        th = p["z_th"]
        long = ok & quiet & (z < -th) & (z.shift(1) >= -th) & (f > 0)
        short = ok & quiet & (z > th) & (z.shift(1) <= th) & (f < 0)
        return sig_array(df, long, short)


@register
class S14_RsiZoneShort(Strategy):
    """The live bot's single validated entry zone: shorts with RSI 35-45 below
    VWAP with selling flow (RSI<35 shorts snap back; 35-45 was the only
    profitable bucket, payoff 1.41). Short-only by design."""
    name = "s14_rsi_zone_short"
    space = {"rsi_len": [14], "ofi_th": [0.0, 0.15], "sl_atr": [1.0, 1.5],
             "tp_mode": ["2ph"], "trail_pct": [0.4, 0.6]}

    def generate(self, df, p, ctx):
        r = rsi(df["close"], p["rsi_len"])
        f = ofi(df)
        ok = df["tod"] >= LATE
        entered = (r < 45) & (r.shift(1) >= 45)
        short = ok & entered & (r >= 35) & (df["close"] < df["vwap"]) & (f < -p["ofi_th"])
        return sig_array(df, short & False, short)


@register
class S15_RangeFadeDiv(Strategy):
    """Late-session test of the day's extreme without flow support — fade it."""
    name = "s15_range_fade_div"
    space = {"after": [11 * 60, 12 * 60], "sl_atr": [1.0, 1.5], "tp_atr": [1.5, 2.5]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        g = df.groupby("day", sort=False)
        day_h = g["high"].cummax().shift(1)
        day_l = g["low"].cummin().shift(1)
        ok = df["tod"] >= p["after"]
        short = ok & (df["high"] >= day_h) & (f < 0) & (df["close"] < df["open"])
        long = ok & (df["low"] <= day_l) & (f > 0) & (df["close"] > df["open"])
        return sig_array(df, long, short)
