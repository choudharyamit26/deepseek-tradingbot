"""CONTROL GROUP part 2 (s05-s06): classic mean reversion (see trend.py note)."""
from strategies.base import Strategy, register, rsi, adx, bollinger, sig_array


@register
class S05_Rsi2(Strategy):
    name = "s05_rsi2_fade_ctl"
    space = {"lo": [5, 10], "sl_atr": [1.0, 1.5], "tp_atr": [1.0, 2.0]}

    def generate(self, df, p, ctx):
        r = rsi(df["close"], 2)
        hi = 100 - p["lo"]
        long = (r < p["lo"]) & (r.shift(1) >= p["lo"])
        short = (r > hi) & (r.shift(1) <= hi)
        return sig_array(df, long, short)


@register
class S06_BollFade(Strategy):
    name = "s06_boll_fade_ctl"
    space = {"k": [2.0, 2.5], "adx_max": [20, 25], "tp_atr": [1.0, 2.0]}

    def generate(self, df, p, ctx):
        lo, _, hi = bollinger(df["close"], 20, p["k"])
        quiet = adx(df) < p["adx_max"]
        long = quiet & (df["close"] < lo) & (df["close"].shift(1) >= lo.shift(1))
        short = quiet & (df["close"] > hi) & (df["close"].shift(1) <= hi.shift(1))
        return sig_array(df, long, short)
