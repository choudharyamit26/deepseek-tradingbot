"""Batch 2 (s21-s40): attack the cost hurdle, not the entry pattern.

Batch 1 + the opencode lab proved mechanical 5-min entries gross ~Rs5-30/trade
vs Rs75-140 costs. These 20 change the economics instead:
  - hold-to-close (H2C) one-shot daily trades: amortize cost over the day move
  - cross-sectional: ONE best trade per day across the whole book
  - 15m/60m signals executed on 5m bars (fewer, larger swings)
  - day-type conditioning: only trade days structurally suited to the setup
H2C params: tp_atr=99 (bracket never caps), max_hold=75 (square-off exits).
"""
import numpy as np
import pandas as pd
from strategies.base import Strategy, register, ema, ofi, sig_array

LATE = 10 * 60 + 15
H2C = {"tp_atr": [99.0], "max_hold": [75]}


def day_stats(df):
    """Per-day OHLC aggregates mapped back onto the 5-min frame, shifted one
    day (yesterday's stats, known today)."""
    g = df.groupby("day", sort=False)
    d = pd.DataFrame({"h": g["high"].max(), "l": g["low"].min(),
                      "o": g["open"].first(), "c": g["close"].last()})
    d["rng"] = d["h"] - d["l"]
    d["ret"] = (d["c"] / d["o"] - 1) * 100
    prev = d.shift(1)
    return {k: df["day"].map(prev[k]) for k in prev.columns}


def htf_close_signal(df, rule, fn):
    """Evaluate fn on a completed higher-timeframe frame; emit the signal on
    the LAST 5-min bar of each bucket (bucket data == info at that bar close)."""
    b = df.index.floor(rule)
    H = pd.DataFrame({
        "open": df["open"].groupby(b).first(),
        "high": df["high"].groupby(b).max(),
        "low": df["low"].groupby(b).min(),
        "close": df["close"].groupby(b).last(),
        "volume": df["volume"].groupby(b).sum()})
    sig_h = fn(H)                                    # +1/-1/0 per bucket
    last_of_bucket = np.r_[b[1:] != b[:-1], True]
    out = np.zeros(len(df), dtype=np.int8)
    vals = sig_h.reindex(b).values
    out[last_of_bucket] = np.nan_to_num(vals[last_of_bucket]).astype(np.int8)
    return out


def at_1015(df):
    return df["tod"] == LATE


# ── hold-to-close one-shots (s21-s25) ────────────────────────────────────────

@register
class S21_FirstHourTrendH2C(Strategy):
    name = "s21_first_hour_trend_h2c"
    space = {"d_pct": [0.4, 0.7], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        dret = (df["close"] / df["day_open"] - 1) * 100
        ok = at_1015(df)
        long = ok & (dret > p["d_pct"]) & (df["close"] > df["vwap"])
        short = ok & (dret < -p["d_pct"]) & (df["close"] < df["vwap"])
        return sig_array(df, long, short)


@register
class S22_GapGoH2C(Strategy):
    name = "s22_gap_go_h2c"
    space = {"g": [0.4, 0.8], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        gap = (df["day_open"] / df["prev_close"] - 1) * 100
        ok = at_1015(df)
        long = ok & (gap > p["g"]) & (df["close"] > df["day_open"])
        short = ok & (gap < -p["g"]) & (df["close"] < df["day_open"])
        return sig_array(df, long, short)


@register
class S23_PrevTrendFollow(Strategy):
    """Yesterday trended hard -> today join on the first VWAP touch."""
    name = "s23_prev_trend_follow_h2c"
    space = {"y_ret": [1.0, 1.5], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        y = day_stats(df)
        ok = df["tod"] >= LATE
        touch = (df["low"] <= df["vwap"]) & (df["close"] > df["vwap"])
        touch_s = (df["high"] >= df["vwap"]) & (df["close"] < df["vwap"])
        long = ok & (y["ret"] > p["y_ret"]) & touch
        short = ok & (y["ret"] < -p["y_ret"]) & touch_s
        return sig_array(df, long, short)


@register
class S24_Nr7ExpansionH2C(Strategy):
    """Yesterday = narrowest range of last 7 days -> trade today's 30-min break."""
    name = "s24_nr7_expansion_h2c"
    space = {"sl_atr": [1.5, 2.5], "n_or": [6], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        rng = g["high"].max() - g["low"].min()
        nr7 = rng == rng.rolling(7).min()
        was_nr7 = df["day"].map(nr7.shift(1)).fillna(False)
        or_h = df["high"].where(df["bar_no"] < p["n_or"]).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < p["n_or"]).groupby(df["day"]).transform("min")
        ok = was_nr7 & (df["bar_no"] >= p["n_or"]) & (df["tod"] <= 13 * 60)
        long = ok & (df["close"] > or_h) & (df["close"].shift(1) <= or_h)
        short = ok & (df["close"] < or_l) & (df["close"].shift(1) >= or_l)
        return sig_array(df, long, short)


@register
class S25_InsideDayBreak(Strategy):
    name = "s25_inside_day_break_h2c"
    space = {"sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        y = day_stats(df)
        g = df.groupby("day", sort=False)
        d_h, d_l = g["high"].max(), g["low"].min()
        inside = (d_h < d_h.shift(1)) & (d_l > d_l.shift(1))
        was_inside = df["day"].map(inside.shift(1)).fillna(False)
        ok = was_inside & (df["tod"] >= LATE)
        long = ok & (df["close"] > y["h"]) & (df["close"].shift(1) <= y["h"])
        short = ok & (df["close"] < y["l"]) & (df["close"].shift(1) >= y["l"])
        return sig_array(df, long, short)


# ── cross-sectional: one trade/day across the book (s26-s29) ─────────────────

def _xs(ctx, key):
    return ctx.get("xs", {}).get(key)


@register
class S26_XsRsWinner(Strategy):
    """At 10:15 the single strongest name in the book, long to close."""
    name = "s26_xs_rs_winner_h2c"
    space = {"min_ret": [0.3, 0.6], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        r = _xs(ctx, "ret1015")
        sym = ctx.get("symbol")
        if r is None or sym not in r.columns:
            return np.zeros(len(df), dtype=np.int8)
        top = r.idxmax(axis=1)
        best = r.max(axis=1)
        my_days = set(top.index[(top == sym) & (best > p["min_ret"])])
        long = at_1015(df) & df["day"].isin(my_days)
        return sig_array(df, long, long & False)


@register
class S27_XsRsLoser(Strategy):
    name = "s27_xs_rs_loser_short_h2c"
    space = {"min_ret": [0.3, 0.6], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        r = _xs(ctx, "ret1015")
        sym = ctx.get("symbol")
        if r is None or sym not in r.columns:
            return np.zeros(len(df), dtype=np.int8)
        bot = r.idxmin(axis=1)
        worst = r.min(axis=1)
        my_days = set(bot.index[(bot == sym) & (worst < -p["min_ret"])])
        short = at_1015(df) & df["day"].isin(my_days)
        return sig_array(df, short & False, short)


@register
class S28_XsGapExtend(Strategy):
    """Largest gap in the book, still extending at 10:15 -> with-gap H2C."""
    name = "s28_xs_gap_extend_h2c"
    space = {"min_gap": [0.5, 1.0], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        gp = _xs(ctx, "gap")
        sym = ctx.get("symbol")
        if gp is None or sym not in gp.columns:
            return np.zeros(len(df), dtype=np.int8)
        ag = gp.abs()
        top = ag.idxmax(axis=1)
        mine = set(top.index[(top == sym) & (ag.max(axis=1) > p["min_gap"])])
        onday = at_1015(df) & df["day"].isin(mine)
        gap = (df["day_open"] / df["prev_close"] - 1) * 100
        long = onday & (gap > 0) & (df["close"] > df["day_open"])
        short = onday & (gap < 0) & (df["close"] < df["day_open"])
        return sig_array(df, long, short)


@register
class S29_RangeCompressBreak(Strategy):
    """3-day range compression vs its own history -> today's ORB break H2C."""
    name = "s29_range_compress_break_h2c"
    space = {"q": [0.5, 0.7], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        rng3 = (g["high"].max().rolling(3).max() - g["low"].min().rolling(3).min())
        med14 = rng3.rolling(14).median()
        tight = (rng3 < p["q"] * med14).shift(1)
        onday = df["day"].map(tight).fillna(False)
        or_h = df["high"].where(df["bar_no"] < 6).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < 6).groupby(df["day"]).transform("min")
        ok = onday & (df["bar_no"] >= 6) & (df["tod"] <= 13 * 60)
        long = ok & (df["close"] > or_h) & (df["close"].shift(1) <= or_h)
        short = ok & (df["close"] < or_l) & (df["close"].shift(1) >= or_l)
        return sig_array(df, long, short)


# ── higher-timeframe signals on 5m execution (s30-s33) ───────────────────────

@register
class S30_Donchian15m(Strategy):
    name = "s30_15m_donchian_h2c"
    space = {"n": [16, 26], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        n = p["n"]

        def fn(H):
            hh = H["high"].rolling(n).max().shift(1)
            ll = H["low"].rolling(n).min().shift(1)
            s = pd.Series(0, index=H.index, dtype=np.int8)
            s[(H["close"] > hh) & (H["close"].shift(1) <= hh.shift(1))] = 1
            s[(H["close"] < ll) & (H["close"].shift(1) >= ll.shift(1))] = -1
            return s
        return htf_close_signal(df, "15min", fn)


@register
class S31_EmaTrend15m(Strategy):
    name = "s31_15m_ema_trend"
    space = {"fast": [5, 9], "sl_atr": [1.5, 2.5],
             "tp_mode": ["2ph"], "trail_pct": [0.6, 1.0]}

    def generate(self, df, p, ctx):
        f = p["fast"]

        def fn(H):
            a, b = ema(H["close"], f), ema(H["close"], 21)
            s = pd.Series(0, index=H.index, dtype=np.int8)
            s[(a > b) & (a.shift(1) <= b.shift(1))] = 1
            s[(a < b) & (a.shift(1) >= b.shift(1))] = -1
            return s
        return htf_close_signal(df, "15min", fn)


@register
class S32_Breakout60m(Strategy):
    name = "s32_60m_breakout_h2c"
    space = {"n": [5, 8], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        n = p["n"]

        def fn(H):
            hh = H["high"].rolling(n).max().shift(1)
            ll = H["low"].rolling(n).min().shift(1)
            s = pd.Series(0, index=H.index, dtype=np.int8)
            s[H["close"] > hh] = 1
            s[H["close"] < ll] = -1
            return s
        return htf_close_signal(df, "60min", fn)


@register
class S33_VwapTrend15m(Strategy):
    """15m VWAP-side flip confirmed by 15m OFI, ridden with a wide trail."""
    name = "s33_15m_vwap_ofi_2ph"
    space = {"ofi_th": [0.1, 0.25], "sl_atr": [1.5, 2.5],
             "tp_mode": ["2ph"], "trail_pct": [0.6, 1.0]}

    def generate(self, df, p, ctx):
        f5 = ofi(df, 30)
        vw = df["vwap"]
        ok = df["tod"] >= LATE
        long = ok & (df["close"] > vw) & (df["close"].shift(3) <= vw.shift(3)) & \
            (f5 > p["ofi_th"])
        short = ok & (df["close"] < vw) & (df["close"].shift(3) >= vw.shift(3)) & \
            (f5 < -p["ofi_th"])
        return sig_array(df, long, short)


# ── day-type conditioned (s34-s38) ───────────────────────────────────────────

@register
class S34_HighVolDayOrb(Strategy):
    """ORB break, but only on days whose recent realized vol is elevated."""
    name = "s34_high_vol_day_orb_h2c"
    space = {"vq": [1.1, 1.3], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        drng = (g["high"].max() - g["low"].min()) / g["close"].last() * 100
        hot = (drng.rolling(3).mean() > p["vq"] * drng.rolling(14).median()).shift(1)
        onday = df["day"].map(hot).fillna(False)
        or_h = df["high"].where(df["bar_no"] < 6).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < 6).groupby(df["day"]).transform("min")
        ok = onday & (df["bar_no"] >= 6) & (df["tod"] <= 13 * 60)
        long = ok & (df["close"] > or_h) & (df["close"].shift(1) <= or_h)
        short = ok & (df["close"] < or_l) & (df["close"].shift(1) >= or_l)
        return sig_array(df, long, short)


@register
class S35_OpenDrive(Strategy):
    """First 3 bars monotonic with volume = conviction open; join at bar 4."""
    name = "s35_open_drive_h2c"
    space = {"vol_x": [1.2, 1.6], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        c = df["close"]
        up3 = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (c.shift(2) > df["day_open"])
        dn3 = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < df["day_open"])
        volx = df["volume"].rolling(3).mean() / \
            df["volume"].rolling(30).mean().clip(lower=1)
        at4 = df["bar_no"] == 3
        long = at4 & up3 & (volx > p["vol_x"])
        short = at4 & dn3 & (volx > p["vol_x"])
        return sig_array(df, long, short)


@register
class S36_LunchReversal(Strategy):
    """Midday poke to a fresh day extreme that closes back inside -> fade it."""
    name = "s36_lunch_reversal_h2c"
    space = {"sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        day_h = g["high"].cummax().shift(1)
        day_l = g["low"].cummin().shift(1)
        lunch = (df["tod"] >= 12 * 60 + 30) & (df["tod"] <= 13 * 60 + 30)
        short = lunch & (df["high"] > day_h) & (df["close"] < day_h) & \
            (df["close"] < df["open"])
        long = lunch & (df["low"] < day_l) & (df["close"] > day_l) & \
            (df["close"] > df["open"])
        return sig_array(df, long, short)


@register
class S37_SecondDayMomo(Strategy):
    """Yesterday broke the prior 20-day high -> buy today's first VWAP touch."""
    name = "s37_second_day_momo_h2c"
    space = {"n_days": [20], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        dh, dl, dc = g["high"].max(), g["low"].min(), g["close"].last()
        broke_up = (dc > dh.shift(1).rolling(p["n_days"]).max()).shift(1)
        broke_dn = (dc < dl.shift(1).rolling(p["n_days"]).min()).shift(1)
        up_day = df["day"].map(broke_up).fillna(False)
        dn_day = df["day"].map(broke_dn).fillna(False)
        ok = df["tod"] >= LATE
        touch_l = (df["low"] <= df["vwap"]) & (df["close"] > df["vwap"])
        touch_s = (df["high"] >= df["vwap"]) & (df["close"] < df["vwap"])
        return sig_array(df, ok & up_day & touch_l, ok & dn_day & touch_s)


@register
class S38_FailedBalance(Strategy):
    """Flat open near prev close, then a decisive 10:15 departure -> follow."""
    name = "s38_failed_balance_h2c"
    space = {"open_tol": [0.3], "dep": [0.5, 0.8], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        gap = ((df["day_open"] / df["prev_close"] - 1) * 100).abs()
        dret = (df["close"] / df["day_open"] - 1) * 100
        ok = at_1015(df) & (gap < p["open_tol"])
        long = ok & (dret > p["dep"])
        short = ok & (dret < -p["dep"])
        return sig_array(df, long, short)


# ── vol-scaled / ensemble (s39-s40) ──────────────────────────────────────────

@register
class S39_CompressedOrbAtr(Strategy):
    """ORB only when the opening range is tiny vs the daily range norm —
    compression opens leave room for expansion."""
    name = "s39_compressed_orb_h2c"
    space = {"frac": [0.25, 0.35], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        day_rng = (g["high"].max() - g["low"].min())
        norm = day_rng.rolling(14).median().shift(1)
        norm_map = df["day"].map(norm)
        or_h = df["high"].where(df["bar_no"] < 6).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < 6).groupby(df["day"]).transform("min")
        small = (or_h - or_l) < p["frac"] * norm_map
        ok = small & (df["bar_no"] >= 6) & (df["tod"] <= 13 * 60)
        long = ok & (df["close"] > or_h) & (df["close"].shift(1) <= or_h)
        short = ok & (df["close"] < or_l) & (df["close"].shift(1) >= or_l)
        return sig_array(df, long, short)


@register
class S40_EnsembleVote(Strategy):
    """Composite regime vote: side of VWAP, EMA trend, OFI, day return, RS vs
    NIFTY. Enter when the vote flips to >=4 of 5 aligned."""
    name = "s40_ensemble_vote_2ph"
    space = {"need": [4, 5], "sl_atr": [1.5, 2.5],
             "tp_mode": ["2ph"], "trail_pct": [0.5, 0.8]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        votes = (
            (df["close"] > df["vwap"]).astype(int)
            + (ema(df["close"], 9) > ema(df["close"], 21)).astype(int)
            + (f > 0).astype(int)
            + (df["close"] > df["day_open"]).astype(int))
        nifty = ctx.get("nifty")
        if nifty is not None:
            nc = nifty["close"].reindex(df.index).ffill()
            no = nifty["day_open"].reindex(df.index).ffill()
            rs = (df["close"] / df["day_open"]) - (nc / no)
            votes = votes + (rs > 0).astype(int)
            total = 5
        else:
            total = 4
        need = min(p["need"], total)
        ok = df["tod"] >= LATE
        long = ok & (votes >= need) & (votes.shift(1) < need)
        short = ok & (votes <= total - need) & (votes.shift(1) > total - need)
        return sig_array(df, long, short)
