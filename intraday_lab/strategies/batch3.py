"""Batch 3 (s41-s60): calendar/expiry, trap reversals, volume-profile levels,
pairs/relative value, regime switching, close-auction flows. All H2C or wide
trails (batch 2 showed only day-scale holds approach the cost frontier)."""
import numpy as np
import pandas as pd
from strategies.base import Strategy, register, ema, ofi, sig_array
from strategies.batch2 import day_stats, at_1015, H2C, LATE

PEERS = {
    "SHRIRAMFIN": ["CHOLAFIN", "BAJFINANCE", "MUTHOOTFIN"],
    "CHOLAFIN": ["SHRIRAMFIN", "BAJFINANCE", "MUTHOOTFIN"],
    "BAJFINANCE": ["SHRIRAMFIN", "CHOLAFIN", "MUTHOOTFIN"],
    "MUTHOOTFIN": ["SHRIRAMFIN", "CHOLAFIN", "BAJFINANCE"],
    "CANBK": ["PNB", "BANKBARODA"], "PNB": ["CANBK", "BANKBARODA"],
    "BANKBARODA": ["CANBK", "PNB"],
    "INDUSINDBK": ["BANDHANBNK"], "BANDHANBNK": ["INDUSINDBK"],
    "ADANIENT": ["ADANIPORTS"], "ADANIPORTS": ["ADANIENT"],
    "LT": ["BHEL", "M&M"], "BHEL": ["LT", "M&M"], "M&M": ["LT", "BHEL"],
    "TRENT": ["DIXON"], "DIXON": ["TRENT"],
}


def _peer_ret(df, ctx):
    """Peer-basket return since open, aligned to df's index (NaN if no peers)."""
    px = ctx.get("prices")
    sym = ctx.get("symbol")
    if px is None or sym not in PEERS:
        return None
    peers = [p for p in PEERS[sym] if p in px.columns]
    if not peers:
        return None
    c = px[peers].reindex(df.index).ffill()
    o = c.groupby(df["day"].values).transform("first")
    return ((c / o - 1) * 100).mean(axis=1)


def _poc(df):
    """Prev-day point of control (volume-modal price), mapped onto today."""
    step = df.groupby("day", sort=False)["close"].transform("first") * 0.002
    binned = (df["close"] / step).round() * step
    va = pd.DataFrame({"day": df["day"].values, "bin": binned.values,
                       "vol": df["volume"].values})
    poc = va.groupby(["day", "bin"])["vol"].sum().reset_index() \
        .sort_values("vol").groupby("day").last()["bin"]
    return df["day"].map(poc.shift(1))


# ── calendar / structural (s41-s44) ──────────────────────────────────────────

@register
class S41_ExpiryFade(Strategy):
    name = "s41_expiry_day_fade_h2c"
    space = {"d_pct": [0.4, 0.7], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        thu = pd.Series(df.index.dayofweek == 3, index=df.index)
        dret = (df["close"] / df["day_open"] - 1) * 100
        ok = at_1015(df) & thu
        return sig_array(df, ok & (dret < -p["d_pct"]), ok & (dret > p["d_pct"]))


@register
class S42_FridayTrend(Strategy):
    name = "s42_friday_trend_h2c"
    space = {"d_pct": [0.3, 0.6], "sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        fri = pd.Series(df.index.dayofweek == 4, index=df.index)
        dret = (df["close"] / df["day_open"] - 1) * 100
        ok = at_1015(df) & fri
        return sig_array(df, ok & (dret > p["d_pct"]), ok & (dret < -p["d_pct"]))


@register
class S43_MondayGapRev(Strategy):
    name = "s43_monday_gap_rev_h2c"
    space = {"g": [0.4, 0.8], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        mon = pd.Series(df.index.dayofweek == 0, index=df.index)
        gap = (df["day_open"] / df["prev_close"] - 1) * 100
        ok = at_1015(df) & mon
        return sig_array(df, ok & (gap < -p["g"]), ok & (gap > p["g"]))


@register
class S44_MonthEndLong(Strategy):
    name = "s44_month_end_long_h2c"
    space = {"sl_atr": [2.0, 3.0], **H2C}

    def generate(self, df, p, ctx):
        days = df["day"].drop_duplicates()
        month = pd.PeriodIndex(days, freq="M")
        last2 = set()
        for m in month.unique():
            md = days[month == m]
            last2.update(md.iloc[-2:])
        onday = df["day"].isin(last2)
        long = at_1015(df) & onday & (df["close"] > df["vwap"])
        return sig_array(df, long, long & False)


# ── trap reversals (s45-s48) ─────────────────────────────────────────────────

@register
class S45_FailedOrbRev(Strategy):
    name = "s45_failed_orb_rev_h2c"
    space = {"n_or": [6], "within": [3, 5], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        or_h = df["high"].where(df["bar_no"] < p["n_or"]).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < p["n_or"]).groupby(df["day"]).transform("min")
        broke_up = (df["close"].shift(1) > or_h.shift(1))
        rej_up = broke_up.rolling(p["within"]).max().astype(bool) & (df["close"] < or_h)
        broke_dn = (df["close"].shift(1) < or_l.shift(1))
        rej_dn = broke_dn.rolling(p["within"]).max().astype(bool) & (df["close"] > or_l)
        ok = (df["bar_no"] >= p["n_or"]) & (df["tod"] <= 13 * 60)
        edge_s = rej_up & ~rej_up.shift(1).fillna(False)
        edge_l = rej_dn & ~rej_dn.shift(1).fillna(False)
        return sig_array(df, ok & edge_l, ok & edge_s)


@register
class S46_FailedPdhRev(Strategy):
    name = "s46_failed_pdh_rev_h2c"
    space = {"within": [2, 4], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        y = day_stats(df)
        w = p["within"]
        broke_up = df["close"].shift(1) > y["h"].shift(1)
        fail_up = broke_up.rolling(w).max().astype(bool) & (df["close"] < y["h"])
        broke_dn = df["close"].shift(1) < y["l"].shift(1)
        fail_dn = broke_dn.rolling(w).max().astype(bool) & (df["close"] > y["l"])
        ok = df["tod"] >= LATE
        return sig_array(df, ok & fail_dn & ~fail_dn.shift(1).fillna(False),
                         ok & fail_up & ~fail_up.shift(1).fillna(False))


@register
class S47_GapTrap(Strategy):
    name = "s47_gap_trap_h2c"
    space = {"g": [0.4, 0.8], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        gap = (df["day_open"] / df["prev_close"] - 1) * 100
        g = df.groupby("day", sort=False)
        day_l = g["low"].cummin().shift(1)
        day_h = g["high"].cummax().shift(1)
        ok = (df["tod"] >= LATE) & (df["tod"] <= 12 * 60)
        short = ok & (gap > p["g"]) & (df["close"] < day_l)
        long = ok & (gap < -p["g"]) & (df["close"] > day_h)
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))


@register
class S48_Spring(Strategy):
    name = "s48_spring_reversal_h2c"
    space = {"within": [4, 6], "sl_atr": [1.5, 2.0], **H2C}

    def generate(self, df, p, ctx):
        y = day_stats(df)
        w = p["within"]
        pierced = df["low"].shift(1) < y["l"].shift(1)
        reclaimed = pierced.rolling(w).max().astype(bool) & (df["close"] > y["l"])
        pierced_h = df["high"].shift(1) > y["h"].shift(1)
        rej = pierced_h.rolling(w).max().astype(bool) & (df["close"] < y["h"])
        ok = df["tod"] >= LATE
        return sig_array(df, ok & reclaimed & ~reclaimed.shift(1).fillna(False),
                         ok & rej & ~rej.shift(1).fillna(False))


# ── volume profile (s49-s51) ─────────────────────────────────────────────────

@register
class S49_PocMagnet(Strategy):
    name = "s49_poc_magnet"
    space = {"dist": [0.7, 1.0], "sl_atr": [1.5, 2.5], "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        poc = _poc(df)
        dev = (df["close"] / poc - 1) * 100
        ok = at_1015(df)
        return sig_array(df, ok & (dev < -p["dist"]), ok & (dev > p["dist"]))


@register
class S50_PocBreak(Strategy):
    name = "s50_poc_break_h2c"
    space = {"vol_x": [1.3, 1.8], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        poc = _poc(df)
        volx = df["volume"] / df["volume"].rolling(10).mean().clip(lower=1)
        ok = (df["tod"] >= LATE) & (volx > p["vol_x"])
        long = ok & (df["close"] > poc) & (df["close"].shift(1) <= poc.shift(1))
        short = ok & (df["close"] < poc) & (df["close"].shift(1) >= poc.shift(1))
        return sig_array(df, long, short)


@register
class S51_PocBounce(Strategy):
    name = "s51_poc_bounce"
    space = {"tol": [0.15], "sl_atr": [1.0, 1.5], "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        poc = _poc(df)
        near = (df["low"] <= poc * (1 + p["tol"] / 100)) & (df["close"] > poc)
        near_s = (df["high"] >= poc * (1 - p["tol"] / 100)) & (df["close"] < poc)
        ok = df["tod"] >= LATE
        from_above = df["open"] > poc
        from_below = df["open"] < poc
        return sig_array(df, ok & near & from_above & (df["close"] > df["open"]),
                         ok & near_s & from_below & (df["close"] < df["open"]))


# ── pairs / relative value (s52-s54) ─────────────────────────────────────────

@register
class S52_PairZ(Strategy):
    name = "s52_pair_zscore"
    space = {"z": [1.5, 2.0], "sl_atr": [1.5, 2.5], "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        pr = _peer_ret(df, ctx)
        if pr is None:
            return np.zeros(len(df), dtype=np.int8)
        mine = (df["close"] / df["day_open"] - 1) * 100
        spread = mine - pr
        z = (spread - spread.rolling(75).mean()) / spread.rolling(75).std().replace(0, np.nan)
        ok = df["tod"] >= LATE
        long = ok & (z < -p["z"]) & (z.shift(1) >= -p["z"])
        short = ok & (z > p["z"]) & (z.shift(1) <= p["z"])
        return sig_array(df, long, short)


@register
class S53_BetaGapRv(Strategy):
    """Gapped against NIFTY's direction -> converge trade."""
    name = "s53_beta_gap_rv_h2c"
    space = {"g": [0.4, 0.7], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        nifty = ctx.get("nifty")
        if nifty is None:
            return np.zeros(len(df), dtype=np.int8)
        gap = (df["day_open"] / df["prev_close"] - 1) * 100
        ng = ((nifty["day_open"] / nifty["prev_close"] - 1) * 100) \
            .reindex(df.index).ffill()
        ok = at_1015(df)
        long = ok & (gap < -p["g"]) & (ng > 0)
        short = ok & (gap > p["g"]) & (ng < 0)
        return sig_array(df, long, short)


@register
class S54_SectorLaggard(Strategy):
    name = "s54_sector_laggard_h2c"
    space = {"lead": [0.7, 1.0], "lag_max": [0.2], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        pr = _peer_ret(df, ctx)
        if pr is None:
            return np.zeros(len(df), dtype=np.int8)
        mine = (df["close"] / df["day_open"] - 1) * 100
        ok = at_1015(df)
        long = ok & (pr > p["lead"]) & (mine.abs() < p["lag_max"])
        short = ok & (pr < -p["lead"]) & (mine.abs() < p["lag_max"])
        return sig_array(df, long, short)


# ── regime switching (s55-s57) ───────────────────────────────────────────────

@register
class S55_VolRegimeSwitch(Strategy):
    """High-vol regime -> ORB momentum; low-vol regime -> VWAP fade."""
    name = "s55_vol_regime_switch"
    space = {"sl_atr": [1.5, 2.0], "tp_atr": [2.0, 3.0]}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        drng = ((g["high"].max() - g["low"].min()) / g["close"].last() * 100)
        hot = (drng.rolling(3).mean() > drng.rolling(14).median()).shift(1)
        is_hot = df["day"].map(hot).fillna(False)
        or_h = df["high"].where(df["bar_no"] < 6).groupby(df["day"]).transform("max")
        or_l = df["low"].where(df["bar_no"] < 6).groupby(df["day"]).transform("min")
        ok = (df["bar_no"] >= 6) & (df["tod"] >= LATE)
        mom_l = ok & is_hot & (df["close"] > or_h) & (df["close"].shift(1) <= or_h)
        mom_s = ok & is_hot & (df["close"] < or_l) & (df["close"].shift(1) >= or_l)
        dev = (df["close"] - df["vwap"]) / df["vwap"] * 100
        fade_l = ok & ~is_hot & (dev < -0.5) & (dev.shift(1) >= -0.5)
        fade_s = ok & ~is_hot & (dev > 0.5) & (dev.shift(1) <= 0.5)
        return sig_array(df, mom_l | fade_l, mom_s | fade_s)


@register
class S56_TrendRegimePullback(Strategy):
    name = "s56_trend_regime_pullback_h2c"
    space = {"n_days": [5], "sl_atr": [1.5, 2.5], **H2C}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        dc = g["close"].last()
        up = (dc > dc.shift(p["n_days"])).shift(1)
        dn = (dc < dc.shift(p["n_days"])).shift(1)
        up_day = df["day"].map(up).fillna(False)
        dn_day = df["day"].map(dn).fillna(False)
        ok = df["tod"] >= LATE
        touch_l = (df["low"] <= df["vwap"]) & (df["close"] > df["vwap"])
        touch_s = (df["high"] >= df["vwap"]) & (df["close"] < df["vwap"])
        return sig_array(df, ok & up_day & touch_l, ok & dn_day & touch_s)


@register
class S57_RangeExhaustFade(Strategy):
    """Day already stretched vs its norm by 11:00 -> fade the extension."""
    name = "s57_range_exhaust_fade"
    space = {"x": [1.2, 1.5], "sl_atr": [1.5, 2.0], "tp_atr": [1.5, 2.5]}

    def generate(self, df, p, ctx):
        g = df.groupby("day", sort=False)
        norm = ((g["high"].max() - g["low"].min()).rolling(14).median()).shift(1)
        norm_map = df["day"].map(norm)
        cur_rng = g["high"].cummax() - g["low"].cummin()
        stretched = cur_rng > p["x"] * norm_map
        ok = (df["tod"] >= 11 * 60) & stretched
        dret = df["close"] > df["day_open"]
        short = ok & dret & (df["close"] < df["open"])
        long = ok & ~dret & (df["close"] > df["open"])
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))


# ── close-auction flows (s58-s60) ────────────────────────────────────────────

@register
class S58_MocMomentum(Strategy):
    name = "s58_moc_momentum"
    space = {"d_pct": [0.6, 1.0], "sl_atr": [1.5, 2.5],
             "tp_atr": [99.0], "max_hold": [75]}

    def generate(self, df, p, ctx):
        at230 = (df["tod"] >= 14 * 60 + 30) & (df["tod"].shift(1) < 14 * 60 + 30)
        dret = (df["close"] / df["day_open"] - 1) * 100
        long = at230 & (dret > p["d_pct"]) & (df["close"] > df["vwap"])
        short = at230 & (dret < -p["d_pct"]) & (df["close"] < df["vwap"])
        return sig_array(df, long, short)


@register
class S59_LateBreakout(Strategy):
    name = "s59_late_breakout"
    space = {"sl_atr": [1.0, 1.5], "tp_atr": [99.0], "max_hold": [75]}

    def generate(self, df, p, ctx):
        f = ofi(df)
        g = df.groupby("day", sort=False)
        day_h = g["high"].cummax().shift(1)
        day_l = g["low"].cummin().shift(1)
        late = df["tod"] >= 14 * 60
        long = late & (df["close"] > day_h) & (f > 0)
        short = late & (df["close"] < day_l) & (f < 0)
        return sig_array(df, long & ~long.shift(1).fillna(False),
                         short & ~short.shift(1).fillna(False))


@register
class S60_VwapConvergence(Strategy):
    """Institutional VWAP benchmarking pulls late-day price toward VWAP."""
    name = "s60_vwap_convergence"
    space = {"dist": [0.5, 0.8], "sl_atr": [1.5, 2.0],
             "tp_atr": [99.0], "max_hold": [75]}

    def generate(self, df, p, ctx):
        at230 = (df["tod"] >= 14 * 60 + 30) & (df["tod"].shift(1) < 14 * 60 + 30)
        dev = (df["close"] - df["vwap"]) / df["vwap"] * 100
        return sig_array(df, at230 & (dev < -p["dist"]), at230 & (dev > p["dist"]))
