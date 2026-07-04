"""Batch 4 (s61-s68): Smart Money Concepts, mechanically defined.

ICT/SMC vocabulary -> testable rules (all backward-looking, pivots confirmed
k bars later): liquidity sweeps, fair value gaps (FVG), order blocks (OB),
break of structure (BOS), change of character (CHoCH), premium/discount.
Exits: hold-to-close or two-phase (the only economics that neared the cost
frontier). Entries gated >=10:15 except sweep setups that need the open.
"""
import numpy as np
import pandas as pd
from strategies.base import Strategy, register, ofi, sig_array
from strategies.batch2 import day_stats, H2C, LATE


def swing_levels(df, k=3):
    """Last CONFIRMED swing high/low (pivot needs k bars each side; level
    becomes known k bars after the pivot -> shift(k) kills lookahead)."""
    hi = df["high"]
    lo = df["low"]
    win = 2 * k + 1
    piv_h = hi.rolling(win, center=True).max() == hi
    piv_l = lo.rolling(win, center=True).min() == lo
    swing_h = hi.where(piv_h).shift(k).ffill()
    swing_l = lo.where(piv_l).shift(k).ffill()
    return swing_h, swing_l


def displacement(df, x=1.2):
    body = (df["close"] - df["open"]).abs()
    return body > x * df["atr"]


@register
class S61_SweepReclaim(Strategy):
    """Liquidity sweep of prev-day low/high with displacement reclaim."""
    name = "s61_smc_sweep_reclaim_h2c"
    space = {"disp_x": [1.0, 1.4], "sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        y = day_stats(df)
        disp = displacement(df, p["disp_x"])
        swept_lo = (df["low"] < y["l"]) & (df["close"] > y["l"]) & \
            (df["close"] > df["open"]) & disp
        swept_hi = (df["high"] > y["h"]) & (df["close"] < y["h"]) & \
            (df["close"] < df["open"]) & disp
        ok = df["tod"] >= LATE
        return sig_array(df, ok & swept_lo, ok & swept_hi)


@register
class S62_FvgRetrace(Strategy):
    """Bullish FVG forms (low > high 2 bars back), price retraces into the
    gap and closes back above it, in the day-trend direction."""
    name = "s62_smc_fvg_retrace_h2c"
    space = {"sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        bull = df["low"] > df["high"].shift(2)
        bear = df["high"] < df["low"].shift(2)
        g = df["day"]
        fvg_top_b = df["low"].where(bull).groupby(g).ffill()
        fvg_bot_s = df["high"].where(bear).groupby(g).ffill()
        up = df["close"] > df["vwap"]
        ok = df["tod"] >= LATE
        long = ok & up & (df["low"] <= fvg_top_b) & (df["close"] > fvg_top_b)
        short = ok & ~up & (df["high"] >= fvg_bot_s) & (df["close"] < fvg_bot_s)
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))


@register
class S63_OrderBlock(Strategy):
    """Retest of the last opposite candle before a displacement move."""
    name = "s63_smc_order_block_h2c"
    space = {"disp_x": [1.2, 1.6], "sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        disp = displacement(df, p["disp_x"])
        red_prev = df["close"].shift(1) < df["open"].shift(1)
        grn_prev = df["close"].shift(1) > df["open"].shift(1)
        bull_ob = disp & (df["close"] > df["open"]) & red_prev
        bear_ob = disp & (df["close"] < df["open"]) & grn_prev
        g = df["day"]
        ob_top = df["high"].shift(1).where(bull_ob).groupby(g).ffill()
        ob_bot = df["low"].shift(1).where(bear_ob).groupby(g).ffill()
        ok = df["tod"] >= LATE
        long = ok & (df["low"] <= ob_top) & (df["close"] > ob_top) & \
            (df["close"] > df["open"])
        short = ok & (df["high"] >= ob_bot) & (df["close"] < ob_bot) & \
            (df["close"] < df["open"])
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))


@register
class S64_Bos(Strategy):
    """Break of structure: close beyond the last confirmed swing point."""
    name = "s64_smc_bos_h2c"
    space = {"k": [3, 5], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        sh, sl = swing_levels(df, p["k"])
        ok = df["tod"] >= LATE
        long = ok & (df["close"] > sh) & (df["close"].shift(1) <= sh.shift(1))
        short = ok & (df["close"] < sl) & (df["close"].shift(1) >= sl.shift(1))
        return sig_array(df, long, short)


@register
class S65_Choch(Strategy):
    """Change of character: day trending one way, close breaks the last
    swing AGAINST that structure -> reversal."""
    name = "s65_smc_choch_h2c"
    space = {"k": [3, 5], "d_pct": [0.3, 0.5], "sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        sh, sl = swing_levels(df, p["k"])
        dret = (df["close"] / df["day_open"] - 1) * 100
        ok = df["tod"] >= LATE
        long = ok & (dret < -p["d_pct"]) & (df["close"] > sh) & \
            (df["close"].shift(1) <= sh.shift(1))
        short = ok & (dret > p["d_pct"]) & (df["close"] < sl) & \
            (df["close"].shift(1) >= sl.shift(1))
        return sig_array(df, long, short)


@register
class S66_SessionSweep(Strategy):
    """Sweep of the MORNING session extreme after 11:00, with flow reversal."""
    name = "s66_smc_session_sweep_h2c"
    space = {"sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        f = ofi(df)
        am = df["tod"] <= 11 * 60
        g = df["day"]
        am_h = df["high"].where(am).groupby(g).transform("max")
        am_l = df["low"].where(am).groupby(g).transform("min")
        ok = df["tod"] >= 11 * 60 + 15
        short = ok & (df["high"] > am_h) & (df["close"] < am_h) & (f < 0)
        long = ok & (df["low"] < am_l) & (df["close"] > am_l) & (f > 0)
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))


@register
class S67_Discount(Strategy):
    """Premium/discount: on up-trend days buy only in the discount half of
    the day's range (and mirror), with a rejection bar."""
    name = "s67_smc_discount_h2c"
    space = {"zone": [0.35, 0.5], "sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        dh = g["high"].cummax()
        dl = g["low"].cummin()
        pos = (df["close"] - dl) / (dh - dl).clip(lower=1e-9)
        up = df["close"] > df["vwap"]
        ok = df["tod"] >= LATE
        long = ok & up & (pos < p["zone"]) & (df["close"] > df["open"])
        short = ok & ~up & (pos > 1 - p["zone"]) & (df["close"] < df["open"])
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))


@register
class S68_SweepFvgConfluence(Strategy):
    """Full SMC confluence: PDL/PDH sweep, then FVG forms in the reversal
    direction -> enter on the FVG close."""
    name = "s68_smc_confluence_h2c"
    space = {"sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        y = day_stats(df)
        g = df["day"]
        swept_lo = ((df["low"] < y["l"]).groupby(g).cummax()).astype(bool)
        swept_hi = ((df["high"] > y["h"]).groupby(g).cummax()).astype(bool)
        bull_fvg = df["low"] > df["high"].shift(2)
        bear_fvg = df["high"] < df["low"].shift(2)
        ok = df["tod"] >= LATE
        long = ok & swept_lo & bull_fvg & (df["close"] > y["l"])
        short = ok & swept_hi & bear_fvg & (df["close"] < y["h"])
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))