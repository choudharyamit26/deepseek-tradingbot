"""Strategy interface + shared vectorized indicator helpers.

Contract: generate(df, params, ctx) returns an int8 array (len == len(df)) of
{-1, 0, +1} evaluated at each bar's CLOSE. The engine fills at the next bar's
open — strategies must never read forward (guarded by tests/test_no_lookahead).
"""
import numpy as np
import pandas as pd

REGISTRY: list = []


def register(cls):
    REGISTRY.append(cls())
    return cls


class Strategy:
    name = "base"
    space: dict = {}

    def generate(self, df, params, ctx) -> np.ndarray:
        raise NotImplementedError

    def default_params(self):
        return {k: v[0] for k, v in self.space.items()}


# ── indicator helpers (all backward-looking) ─────────────────────────────────

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def bollinger(close, n, k):
    m = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return m - k * sd, m, m + k * sd


def keltner(df, n, mult):
    m = ema(df["close"], n)
    return m - mult * df["atr"], m, m + mult * df["atr"]


def supertrend_dir(df, period, mult):
    """+1/-1 regime series (classic supertrend flip logic, backward-only)."""
    hl2 = (df["high"] + df["low"]) / 2
    atr = df["atr"] if period == 14 else (
        pd.concat([df["high"] - df["low"],
                   (df["high"] - df["close"].shift()).abs(),
                   (df["low"] - df["close"].shift()).abs()], axis=1)
        .max(axis=1).rolling(period).mean())
    ub = (hl2 + mult * atr).values
    lb = (hl2 - mult * atr).values
    c = df["close"].values
    n = len(df)
    d = np.ones(n, dtype=np.int8)
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, n):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
    return pd.Series(d, index=df.index)


def macd(close, fast, slow, sig):
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, sig)


def ofi(df, n=10):
    """CLV-weighted volume imbalance in [-1, +1] (same construction the live
    bot validated as GATE-READY)."""
    rng = (df["high"] - df["low"]).clip(lower=1e-9)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    sv = (clv * df["volume"]).rolling(n).sum()
    vv = df["volume"].rolling(n).sum().clip(lower=1e-9)
    return (sv / vv).clip(-1, 1)


def cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def sig_array(df, long_cond, short_cond):
    s = np.zeros(len(df), dtype=np.int8)
    s[np.asarray(long_cond.fillna(False), dtype=bool)] = 1
    s[np.asarray(short_cond.fillna(False), dtype=bool)] = -1
    return s
