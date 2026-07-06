from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .indicators import (
    atr,
    between_times,
    bollinger,
    ema,
    first_bar_levels,
    intraday_vwap,
    macd,
    opening_range,
    previous_day_levels,
    rsi,
    supertrend,
)

SignalFunc = Callable[[pd.DataFrame, dict[str, float | int | str]], pd.Series]


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    description: str
    default_params: dict[str, float | int | str]
    param_grid: dict[str, list[float | int | str]]
    signal_func: SignalFunc

    def generate(self, df: pd.DataFrame, params: dict[str, float | int | str] | None = None) -> pd.Series:
        merged = dict(self.default_params)
        if params:
            merged.update(params)
        signal = self.signal_func(df.copy(), merged).fillna(0).astype(int)
        return signal.where(signal.isin([-1, 0, 1]), 0)


COMMON_EXIT_GRID: dict[str, list[float | int]] = {
    "stop_pct": [0.35, 0.60],
    "target_pct": [0.70, 1.20],
    "max_hold_bars": [12, 24],
}


def _signal(index: pd.Index, long_mask: pd.Series, short_mask: pd.Series) -> pd.Series:
    out = pd.Series(0, index=index, dtype=int)
    out.loc[long_mask.fillna(False)] = 1
    out.loc[short_mask.fillna(False)] = -1
    return out


def _cross_above(series: pd.Series, level: pd.Series | float) -> pd.Series:
    return (series > level) & (series.shift(1) <= level if np.isscalar(level) else series.shift(1) <= level.shift(1))


def _cross_below(series: pd.Series, level: pd.Series | float) -> pd.Series:
    return (series < level) & (series.shift(1) >= level if np.isscalar(level) else series.shift(1) >= level.shift(1))


def _with_common(params: dict[str, list[float | int | str]]) -> dict[str, list[float | int | str]]:
    out: dict[str, list[float | int | str]] = dict(params)
    out.update(COMMON_EXIT_GRID)
    return out


def opening_range_breakout(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    high, low, active = opening_range(df, int(p["or_minutes"]))
    buffer = float(p["breakout_buffer_pct"]) / 100
    long = active & (df["close"] > high * (1 + buffer))
    short = active & (df["close"] < low * (1 - buffer))
    return _signal(df.index, long, short)


def opening_range_fade(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    high, low, active = opening_range(df, int(p["or_minutes"]))
    buffer = float(p["fade_buffer_pct"]) / 100
    long = active & (df["low"] < low * (1 - buffer)) & (df["close"] > low)
    short = active & (df["high"] > high * (1 + buffer)) & (df["close"] < high)
    return _signal(df.index, long, short)


def vwap_pullback(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    vwap = intraday_vwap(df)
    trend = ema(df["close"], int(p["trend_ema"]))
    tolerance = float(p["pullback_pct"]) / 100
    long = (df["close"] > trend) & (df["low"] <= vwap * (1 + tolerance)) & (df["close"] > vwap)
    short = (df["close"] < trend) & (df["high"] >= vwap * (1 - tolerance)) & (df["close"] < vwap)
    return _signal(df.index, long, short)


def vwap_trend_continuation(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    vwap = intraday_vwap(df)
    slope = vwap.diff(int(p["slope_bars"]))
    vol_avg = df["volume"].rolling(int(p["volume_period"]), min_periods=int(p["volume_period"])).mean()
    volume_ok = df["volume"] > vol_avg * float(p["volume_mult"])
    distance = float(p["min_distance_pct"]) / 100
    long = (slope > 0) & volume_ok & (df["close"] > vwap * (1 + distance))
    short = (slope < 0) & volume_ok & (df["close"] < vwap * (1 - distance))
    return _signal(df.index, long, short)


def ema_crossover(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    fast = ema(df["close"], int(p["fast_ema"]))
    slow = ema(df["close"], int(p["slow_ema"]))
    return _signal(df.index, _cross_above(fast, slow), _cross_below(fast, slow))


def ema_pullback_trend(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    trend = ema(df["close"], int(p["trend_ema"]))
    pullback = ema(df["close"], int(p["pullback_ema"]))
    long = (df["close"] > trend) & (df["low"] <= pullback) & (df["close"] > pullback)
    short = (df["close"] < trend) & (df["high"] >= pullback) & (df["close"] < pullback)
    return _signal(df.index, long, short)


def supertrend_continuation(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    _, direction = supertrend(df, int(p["atr_period"]), float(p["multiplier"]))
    long = (direction == 1) & (direction.shift(1) == -1)
    short = (direction == -1) & (direction.shift(1) == 1)
    return _signal(df.index, long, short)


def rsi_mean_reversion(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    value = rsi(df["close"], int(p["rsi_period"]))
    long = _cross_above(value, float(p["oversold"]))
    short = _cross_below(value, float(p["overbought"]))
    return _signal(df.index, long, short)


def rsi_trend_pullback(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    trend = ema(df["close"], int(p["trend_ema"]))
    value = rsi(df["close"], int(p["rsi_period"]))
    long = (df["close"] > trend) & _cross_above(value, float(p["pullback_low"]))
    short = (df["close"] < trend) & _cross_below(value, float(p["pullback_high"]))
    return _signal(df.index, long, short)


def bollinger_mean_reversion(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    upper, _, lower = bollinger(df["close"], int(p["bb_period"]), float(p["std_dev"]))
    long = _cross_above(df["close"], lower)
    short = _cross_below(df["close"], upper)
    return _signal(df.index, long, short)


def bollinger_squeeze_breakout(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    upper, middle, lower = bollinger(df["close"], int(p["bb_period"]), float(p["std_dev"]))
    width = (upper - lower) / middle.replace(0, np.nan)
    threshold = width.rolling(int(p["squeeze_lookback"]), min_periods=int(p["squeeze_lookback"])).quantile(
        float(p["squeeze_quantile"])
    )
    squeezed = width <= threshold
    long = squeezed.shift(1).fillna(False) & _cross_above(df["close"], upper)
    short = squeezed.shift(1).fillna(False) & _cross_below(df["close"], lower)
    return _signal(df.index, long, short)


def donchian_breakout(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    period = int(p["channel_period"])
    high = df["high"].rolling(period, min_periods=period).max().shift(1)
    low = df["low"].rolling(period, min_periods=period).min().shift(1)
    long = df["close"] > high
    short = df["close"] < low
    return _signal(df.index, long, short)


def keltner_breakout(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    mid = ema(df["close"], int(p["ema_period"]))
    spread = atr(df, int(p["atr_period"])) * float(p["multiplier"])
    upper = mid + spread
    lower = mid - spread
    return _signal(df.index, _cross_above(df["close"], upper), _cross_below(df["close"], lower))


def macd_momentum(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    line, signal_line, _ = macd(df["close"], int(p["fast"]), int(p["slow"]), int(p["signal"]))
    return _signal(df.index, _cross_above(line, signal_line), _cross_below(line, signal_line))


def atr_volatility_breakout(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    spread = atr(df, int(p["atr_period"])) * float(p["atr_mult"])
    previous_close = df["close"].shift(1)
    upper = previous_close + spread
    lower = previous_close - spread
    return _signal(df.index, _cross_above(df["close"], upper), _cross_below(df["close"], lower))


def gap_and_go(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    prev_high, prev_low, prev_close = previous_day_levels(df)
    first_high, first_low, active = first_bar_levels(df, int(p["confirm_bars"]))
    gap = (df["open"] - prev_close) / prev_close.replace(0, np.nan) * 100
    threshold = float(p["gap_pct"])
    long = active & (gap >= threshold) & (df["close"] > first_high) & (df["close"] > prev_high)
    short = active & (gap <= -threshold) & (df["close"] < first_low) & (df["close"] < prev_low)
    return _signal(df.index, long, short)


def gap_fill_reversal(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    _, _, prev_close = previous_day_levels(df)
    first_high, first_low, active = first_bar_levels(df, int(p["confirm_bars"]))
    gap = (df["open"] - prev_close) / prev_close.replace(0, np.nan) * 100
    threshold = float(p["gap_pct"])
    long = active & (gap <= -threshold) & (df["close"] > first_high)
    short = active & (gap >= threshold) & (df["close"] < first_low)
    return _signal(df.index, long, short)


def previous_day_breakout(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    prev_high, prev_low, _ = previous_day_levels(df)
    buffer = float(p["breakout_buffer_pct"]) / 100
    vol_avg = df["volume"].rolling(int(p["volume_period"]), min_periods=int(p["volume_period"])).mean()
    volume_ok = df["volume"] > vol_avg * float(p["volume_mult"])
    long = volume_ok & (df["close"] > prev_high * (1 + buffer))
    short = volume_ok & (df["close"] < prev_low * (1 - buffer))
    return _signal(df.index, long, short)


def volume_spike_breakout(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    vol_avg = df["volume"].rolling(int(p["volume_period"]), min_periods=int(p["volume_period"])).mean()
    range_ok = (df["high"] - df["low"]) > atr(df, int(p["atr_period"])) * float(p["range_atr_mult"])
    volume_ok = df["volume"] > vol_avg * float(p["volume_mult"])
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    close_location = (df["close"] - df["low"]) / candle_range
    long = volume_ok & range_ok & (close_location >= float(p["close_location"]))
    short = volume_ok & range_ok & (close_location <= 1 - float(p["close_location"]))
    return _signal(df.index, long, short)


def time_of_day_reversal(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    allowed = between_times(df["timestamp"], str(p["start_time"]), str(p["end_time"]))
    vwap = intraday_vwap(df)
    value = rsi(df["close"], int(p["rsi_period"]))
    stretch = float(p["vwap_stretch_pct"]) / 100
    long = allowed & (df["close"] < vwap * (1 - stretch)) & (value < float(p["oversold"]))
    short = allowed & (df["close"] > vwap * (1 + stretch)) & (value > float(p["overbought"]))
    return _signal(df.index, long, short)


STRATEGIES: dict[str, StrategyDefinition] = {
    "opening_range_breakout": StrategyDefinition(
        "opening_range_breakout",
        "Breaks the initial session range with a small buffer.",
        {"or_minutes": 30, "breakout_buffer_pct": 0.03, "stop_pct": 0.50, "target_pct": 1.00, "max_hold_bars": 18},
        _with_common({"or_minutes": [15, 30, 45], "breakout_buffer_pct": [0.02, 0.05]}),
        opening_range_breakout,
    ),
    "opening_range_fade": StrategyDefinition(
        "opening_range_fade",
        "Fades failed extensions beyond the opening range.",
        {"or_minutes": 30, "fade_buffer_pct": 0.05, "stop_pct": 0.45, "target_pct": 0.80, "max_hold_bars": 12},
        _with_common({"or_minutes": [15, 30, 45], "fade_buffer_pct": [0.03, 0.08]}),
        opening_range_fade,
    ),
    "vwap_pullback": StrategyDefinition(
        "vwap_pullback",
        "Trades trend pullbacks into intraday VWAP.",
        {"trend_ema": 50, "pullback_pct": 0.05, "stop_pct": 0.45, "target_pct": 0.90, "max_hold_bars": 18},
        _with_common({"trend_ema": [34, 50], "pullback_pct": [0.03, 0.08]}),
        vwap_pullback,
    ),
    "vwap_trend_continuation": StrategyDefinition(
        "vwap_trend_continuation",
        "Follows VWAP slope with volume confirmation.",
        {
            "slope_bars": 6,
            "volume_period": 20,
            "volume_mult": 1.4,
            "min_distance_pct": 0.05,
            "stop_pct": 0.50,
            "target_pct": 1.00,
            "max_hold_bars": 18,
        },
        _with_common({"slope_bars": [4, 8], "volume_mult": [1.2, 1.6], "min_distance_pct": [0.03, 0.08]}),
        vwap_trend_continuation,
    ),
    "ema_crossover": StrategyDefinition(
        "ema_crossover",
        "Trades fast and slow EMA crosses.",
        {"fast_ema": 9, "slow_ema": 21, "stop_pct": 0.45, "target_pct": 0.90, "max_hold_bars": 24},
        _with_common({"fast_ema": [5, 9, 13], "slow_ema": [21, 34]}),
        ema_crossover,
    ),
    "ema_pullback_trend": StrategyDefinition(
        "ema_pullback_trend",
        "Uses a long EMA trend filter and short EMA pullback trigger.",
        {"trend_ema": 55, "pullback_ema": 13, "stop_pct": 0.45, "target_pct": 1.00, "max_hold_bars": 18},
        _with_common({"trend_ema": [34, 55, 89], "pullback_ema": [9, 13, 21]}),
        ema_pullback_trend,
    ),
    "supertrend_continuation": StrategyDefinition(
        "supertrend_continuation",
        "Trades Supertrend direction flips.",
        {"atr_period": 10, "multiplier": 3.0, "stop_pct": 0.55, "target_pct": 1.10, "max_hold_bars": 24},
        _with_common({"atr_period": [7, 10, 14], "multiplier": [2.0, 3.0]}),
        supertrend_continuation,
    ),
    "rsi_mean_reversion": StrategyDefinition(
        "rsi_mean_reversion",
        "Fades intraday RSI extremes after RSI crosses back inside.",
        {"rsi_period": 14, "oversold": 30, "overbought": 70, "stop_pct": 0.40, "target_pct": 0.70, "max_hold_bars": 12},
        _with_common({"rsi_period": [7, 14], "oversold": [25, 30], "overbought": [70, 75]}),
        rsi_mean_reversion,
    ),
    "rsi_trend_pullback": StrategyDefinition(
        "rsi_trend_pullback",
        "Buys RSI recovery in uptrends and shorts RSI weakness in downtrends.",
        {
            "trend_ema": 55,
            "rsi_period": 14,
            "pullback_low": 45,
            "pullback_high": 55,
            "stop_pct": 0.45,
            "target_pct": 0.90,
            "max_hold_bars": 18,
        },
        _with_common({"trend_ema": [34, 55], "rsi_period": [7, 14], "pullback_low": [40, 45], "pullback_high": [55, 60]}),
        rsi_trend_pullback,
    ),
    "bollinger_mean_reversion": StrategyDefinition(
        "bollinger_mean_reversion",
        "Fades closes that revert back inside Bollinger Bands.",
        {"bb_period": 20, "std_dev": 2.0, "stop_pct": 0.40, "target_pct": 0.75, "max_hold_bars": 12},
        _with_common({"bb_period": [20, 30], "std_dev": [1.8, 2.2]}),
        bollinger_mean_reversion,
    ),
    "bollinger_squeeze_breakout": StrategyDefinition(
        "bollinger_squeeze_breakout",
        "Breaks out of low Bollinger bandwidth regimes.",
        {
            "bb_period": 20,
            "std_dev": 2.0,
            "squeeze_lookback": 80,
            "squeeze_quantile": 0.20,
            "stop_pct": 0.50,
            "target_pct": 1.10,
            "max_hold_bars": 24,
        },
        _with_common({"bb_period": [20, 30], "squeeze_lookback": [60, 100], "squeeze_quantile": [0.15, 0.25]}),
        bollinger_squeeze_breakout,
    ),
    "donchian_breakout": StrategyDefinition(
        "donchian_breakout",
        "Trades close breaks of a shifted Donchian channel.",
        {"channel_period": 30, "stop_pct": 0.55, "target_pct": 1.15, "max_hold_bars": 24},
        _with_common({"channel_period": [20, 30, 45]}),
        donchian_breakout,
    ),
    "keltner_breakout": StrategyDefinition(
        "keltner_breakout",
        "Trades breaks beyond Keltner channels.",
        {"ema_period": 20, "atr_period": 14, "multiplier": 1.5, "stop_pct": 0.50, "target_pct": 1.00, "max_hold_bars": 18},
        _with_common({"ema_period": [20, 34], "atr_period": [10, 14], "multiplier": [1.3, 1.8]}),
        keltner_breakout,
    ),
    "macd_momentum": StrategyDefinition(
        "macd_momentum",
        "Trades MACD line and signal line momentum crosses.",
        {"fast": 12, "slow": 26, "signal": 9, "stop_pct": 0.45, "target_pct": 0.95, "max_hold_bars": 18},
        _with_common({"fast": [8, 12], "slow": [21, 26], "signal": [5, 9]}),
        macd_momentum,
    ),
    "atr_volatility_breakout": StrategyDefinition(
        "atr_volatility_breakout",
        "Uses previous close plus or minus ATR expansion levels.",
        {"atr_period": 14, "atr_mult": 0.7, "stop_pct": 0.50, "target_pct": 1.05, "max_hold_bars": 18},
        _with_common({"atr_period": [10, 14], "atr_mult": [0.5, 0.9]}),
        atr_volatility_breakout,
    ),
    "gap_and_go": StrategyDefinition(
        "gap_and_go",
        "Continues large opening gaps after confirmation bars.",
        {"gap_pct": 0.60, "confirm_bars": 2, "stop_pct": 0.55, "target_pct": 1.10, "max_hold_bars": 18},
        _with_common({"gap_pct": [0.40, 0.80], "confirm_bars": [1, 2, 3]}),
        gap_and_go,
    ),
    "gap_fill_reversal": StrategyDefinition(
        "gap_fill_reversal",
        "Fades large opening gaps when early price action reverses.",
        {"gap_pct": 0.60, "confirm_bars": 2, "stop_pct": 0.45, "target_pct": 0.90, "max_hold_bars": 18},
        _with_common({"gap_pct": [0.40, 0.80], "confirm_bars": [1, 2, 3]}),
        gap_fill_reversal,
    ),
    "previous_day_breakout": StrategyDefinition(
        "previous_day_breakout",
        "Breaks previous day high or low with volume confirmation.",
        {
            "breakout_buffer_pct": 0.03,
            "volume_period": 20,
            "volume_mult": 1.3,
            "stop_pct": 0.50,
            "target_pct": 1.00,
            "max_hold_bars": 18,
        },
        _with_common({"breakout_buffer_pct": [0.02, 0.06], "volume_mult": [1.1, 1.5]}),
        previous_day_breakout,
    ),
    "volume_spike_breakout": StrategyDefinition(
        "volume_spike_breakout",
        "Trades directional wide-range candles with volume spikes.",
        {
            "volume_period": 20,
            "volume_mult": 2.0,
            "atr_period": 14,
            "range_atr_mult": 0.8,
            "close_location": 0.75,
            "stop_pct": 0.55,
            "target_pct": 1.15,
            "max_hold_bars": 12,
        },
        _with_common({"volume_mult": [1.6, 2.2], "range_atr_mult": [0.6, 1.0], "close_location": [0.70, 0.80]}),
        volume_spike_breakout,
    ),
    "time_of_day_reversal": StrategyDefinition(
        "time_of_day_reversal",
        "Fades late-morning VWAP stretches with RSI confirmation.",
        {
            "start_time": "10:30",
            "end_time": "14:30",
            "rsi_period": 14,
            "vwap_stretch_pct": 0.35,
            "oversold": 30,
            "overbought": 70,
            "stop_pct": 0.45,
            "target_pct": 0.85,
            "max_hold_bars": 12,
        },
        _with_common({"vwap_stretch_pct": [0.25, 0.45], "rsi_period": [7, 14], "oversold": [25, 30], "overbought": [70, 75]}),
        time_of_day_reversal,
    ),
}


BATCH3_EXIT_GRID: dict[str, list[float | int]] = {
    "stop_pct": [0.75, 1.25],
    "target_pct": [99.0],
    "max_hold_bars": [75],
}


def _with_h2c(params: dict[str, list[float | int | str]]) -> dict[str, list[float | int | str]]:
    out: dict[str, list[float | int | str]] = dict(params)
    out.update(BATCH3_EXIT_GRID)
    return out


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"]).dt.date


def _bar_no(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.groupby(_dates(df), sort=False).cumcount().to_numpy(), index=df.index, dtype=int)


def _day_open(df: pd.DataFrame) -> pd.Series:
    return df.groupby(_dates(df), sort=False)["open"].transform("first")


def _daily_context(df: pd.DataFrame) -> dict[str, pd.Series]:
    dates = _dates(df)
    daily = df.groupby(dates, sort=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    )
    daily["range"] = daily["high"] - daily["low"]
    daily["ret"] = (daily["close"] / daily["open"] - 1) * 100
    daily["close_pos"] = (daily["close"] - daily["low"]) / daily["range"].replace(0, np.nan)
    daily["prev2_high"] = daily["high"].rolling(2).max().shift(1)
    daily["prev2_low"] = daily["low"].rolling(2).min().shift(1)
    daily["prev3_ret"] = daily["ret"].rolling(3).sum().shift(1)
    daily["prev5_ret"] = daily["ret"].rolling(5).sum().shift(1)
    daily["high20"] = daily["high"].rolling(20).max().shift(1)
    daily["low20"] = daily["low"].rolling(20).min().shift(1)
    daily["range_med10"] = daily["range"].rolling(10).median().shift(1)
    daily["range_med20"] = daily["range"].rolling(20).median().shift(1)
    daily["range_min4"] = daily["range"].rolling(4).min().shift(1)
    daily["volume_med20"] = daily["volume"].rolling(20).median().shift(1)
    prev = daily.shift(1)
    context: dict[str, pd.Series] = {}
    for column in ["open", "high", "low", "close", "volume", "range", "ret", "close_pos"]:
        context[f"prev_{column}"] = pd.Series(dates.map(prev[column]).to_numpy(), index=df.index, dtype=float)
    for column in ["prev2_high", "prev2_low", "prev3_ret", "prev5_ret", "high20", "low20", "range_med10", "range_med20", "range_min4", "volume_med20"]:
        context[column] = pd.Series(dates.map(daily[column]).to_numpy(), index=df.index, dtype=float)
    return context


def _nth_close(df: pd.DataFrame, n: int) -> pd.Series:
    dates = _dates(df)
    values = df.groupby(dates, sort=False)["close"].nth(int(n))
    return pd.Series(dates.map(values).to_numpy(), index=df.index, dtype=float)


def _first_n_volume_mean(df: pd.DataFrame, n: int) -> pd.Series:
    dates = _dates(df)
    bar_no = _bar_no(df)
    values = df.loc[bar_no < int(n)].groupby(dates.loc[bar_no < int(n)], sort=False)["volume"].mean()
    return pd.Series(dates.map(values).to_numpy(), index=df.index, dtype=float)


def _day_cummax(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    return series.groupby(_dates(df), sort=False).cummax()


def _day_cummin(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    return series.groupby(_dates(df), sort=False).cummin()


def b3_prev_day_sweep_reversal(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    prev_high, prev_low, _ = previous_day_levels(df)
    bar = _bar_no(df)
    sweep = float(p["sweep_pct"]) / 100
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & (df["low"] < prev_low * (1 - sweep)) & (df["close"] > prev_low) & (df["close"] > df["open"])
    short = ok & (df["high"] > prev_high * (1 + sweep)) & (df["close"] < prev_high) & (df["close"] < df["open"])
    return _signal(df.index, long, short)


def b3_open_gap_stall_fade(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    _, _, prev_close = previous_day_levels(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    vwap = intraday_vwap(df)
    gap = (day_open / prev_close.replace(0, np.nan) - 1) * 100
    at_decision = bar == int(p["decision_bar"])
    threshold = float(p["gap_pct"])
    long = at_decision & (gap <= -threshold) & (df["close"] > day_open) & (df["close"] > vwap)
    short = at_decision & (gap >= threshold) & (df["close"] < day_open) & (df["close"] < vwap)
    return _signal(df.index, long, short)


def b3_pivot_reclaim_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    prev_high, prev_low, prev_close = previous_day_levels(df)
    pivot = (prev_high + prev_low + prev_close) / 3
    day_open = _day_open(df)
    bar = _bar_no(df)
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & (day_open < pivot) & _cross_above(df["close"], pivot)
    short = ok & (day_open > pivot) & _cross_below(df["close"], pivot)
    return _signal(df.index, long, short)


def b3_cpr_escape_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    prev_high, prev_low, prev_close = previous_day_levels(df)
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = 2 * pivot - bc
    cpr_low = pd.concat([bc, tc], axis=1).min(axis=1)
    cpr_high = pd.concat([bc, tc], axis=1).max(axis=1)
    day_open = _day_open(df)
    bar = _bar_no(df)
    volume_avg = df["volume"].rolling(20, min_periods=20).mean()
    width_pct = (cpr_high - cpr_low) / prev_close.replace(0, np.nan) * 100
    open_inside = (day_open >= cpr_low) & (day_open <= cpr_high)
    ok = (bar >= int(p["start_bar"])) & open_inside & (width_pct <= float(p["max_width_pct"])) & (df["volume"] > volume_avg * float(p["volume_mult"]))
    long = ok & _cross_above(df["close"], cpr_high)
    short = ok & _cross_below(df["close"], cpr_low)
    return _signal(df.index, long, short)


def b3_two_day_extreme_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    ctx = _daily_context(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    buffer = float(p["buffer_pct"]) / 100
    day_ret = (df["close"] / day_open - 1) * 100
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & (day_ret > float(p["min_day_ret"])) & (df["close"] > ctx["prev2_high"] * (1 + buffer))
    short = ok & (day_ret < -float(p["min_day_ret"])) & (df["close"] < ctx["prev2_low"] * (1 - buffer))
    return _signal(df.index, long, short)


def b3_three_day_reversal_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    ctx = _daily_context(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    threshold = float(p["three_day_ret_pct"])
    long = ok & (ctx["prev3_ret"] <= -threshold) & _cross_above(df["close"], day_open)
    short = ok & (ctx["prev3_ret"] >= threshold) & _cross_below(df["close"], day_open)
    return _signal(df.index, long, short)


def b3_nr4_false_break_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    ctx = _daily_context(df)
    or_high, or_low, active = opening_range(df, int(p["or_minutes"]))
    bar = _bar_no(df)
    was_nr4 = ctx["prev_range"] <= ctx["range_min4"]
    ok = active & was_nr4 & (bar <= int(p["end_bar"]))
    long = ok & (df["low"] < or_low) & (df["close"] > or_low)
    short = ok & (df["high"] > or_high) & (df["close"] < or_high)
    return _signal(df.index, long, short)


def b3_morning_midpoint_trap(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    or_high, or_low, active = opening_range(df, int(p["or_minutes"]))
    midpoint = (or_high + or_low) / 2
    dates = _dates(df)
    broke_high = ((df["high"] > or_high) & active).groupby(dates, sort=False).cummax().shift(1).fillna(False)
    broke_low = ((df["low"] < or_low) & active).groupby(dates, sort=False).cummax().shift(1).fillna(False)
    bar = _bar_no(df)
    ok = active & (bar <= int(p["end_bar"]))
    long = ok & broke_low & _cross_above(df["close"], midpoint)
    short = ok & broke_high & _cross_below(df["close"], midpoint)
    return _signal(df.index, long, short)


def b3_opening_drive_pullback_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    day_open = _day_open(df)
    fourth_close = _nth_close(df, int(p["drive_bars"]) - 1)
    first_volume = _first_n_volume_mean(df, int(p["drive_bars"]))
    normal_volume = df["volume"].rolling(30, min_periods=30).mean()
    pullback = ema(df["close"], int(p["pullback_ema"]))
    bar = _bar_no(df)
    drive_ret = (fourth_close / day_open - 1) * 100
    volume_ok = first_volume > normal_volume * float(p["volume_mult"])
    ok = (bar >= int(p["drive_bars"]) + 2) & (bar <= int(p["end_bar"])) & volume_ok
    long = ok & (drive_ret >= float(p["drive_pct"])) & (df["low"] <= pullback) & (df["close"] > pullback)
    short = ok & (drive_ret <= -float(p["drive_pct"])) & (df["high"] >= pullback) & (df["close"] < pullback)
    return _signal(df.index, long, short)


def b3_vwap_compression_release(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    vwap = intraday_vwap(df)
    distance = (df["close"] - vwap) / vwap.replace(0, np.nan) * 100
    lookback = int(p["lookback"])
    compressed = distance.abs().rolling(lookback, min_periods=lookback).max() <= float(p["max_dist_pct"])
    high = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    low = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    slope = vwap.diff(int(p["slope_bars"]))
    long = compressed.shift(1).fillna(False) & (slope > 0) & (df["close"] > high)
    short = compressed.shift(1).fillna(False) & (slope < 0) & (df["close"] < low)
    return _signal(df.index, long, short)


def b3_inside_bar_expansion_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    bar = _bar_no(df)
    inside_previous = (df["high"].shift(1) < df["high"].shift(2)) & (df["low"].shift(1) > df["low"].shift(2))
    range_now = df["high"] - df["low"]
    range_ok = range_now > atr(df, int(p["atr_period"])) * float(p["range_atr_mult"])
    close_pos = (df["close"] - df["low"]) / range_now.replace(0, np.nan)
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"])) & inside_previous & range_ok
    long = ok & (df["close"] > df["high"].shift(1)) & (close_pos >= float(p["close_location"]))
    short = ok & (df["close"] < df["low"].shift(1)) & (close_pos <= 1 - float(p["close_location"]))
    return _signal(df.index, long, short)


def b3_pinbar_reversal_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    bar = _bar_no(df)
    body = (df["close"] - df["open"]).abs().replace(0, np.nan)
    upper_wick = df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    lower_wick = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
    day_high = _day_cummax(df["high"], df).shift(1)
    day_low = _day_cummin(df["low"], df).shift(1)
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    wick_mult = float(p["wick_body_mult"])
    long = ok & (df["low"] <= day_low) & (lower_wick > body * wick_mult) & (df["close"] > df["open"])
    short = ok & (df["high"] >= day_high) & (upper_wick > body * wick_mult) & (df["close"] < df["open"])
    return _signal(df.index, long, short)


def b3_ema_slope_acceleration_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    slow = ema(df["close"], int(p["slow_ema"]))
    fast = ema(df["close"], int(p["fast_ema"]))
    slope = slow.diff(int(p["slope_bars"])) / slow.replace(0, np.nan) * 100
    impulse = df["close"].pct_change(int(p["impulse_bars"])) * 100
    bar = _bar_no(df)
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & (fast > slow) & (slope > float(p["slope_pct"])) & _cross_above(impulse, float(p["impulse_pct"]))
    short = ok & (fast < slow) & (slope < -float(p["slope_pct"])) & _cross_below(impulse, -float(p["impulse_pct"]))
    return _signal(df.index, long, short)


def b3_realized_vol_contraction_break(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    ctx = _daily_context(df)
    bar = _bar_no(df)
    lookback = int(p["lookback"])
    rolling_high = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    rolling_low = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    contraction = ctx["prev_range"] < ctx["range_med20"] * float(p["range_frac"])
    ok = contraction & (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & (df["close"] > rolling_high)
    short = ok & (df["close"] < rolling_low)
    return _signal(df.index, long, short)


def b3_afternoon_range_resolve_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    bar = _bar_no(df)
    dates = _dates(df)
    start_bar = int(p["range_start_bar"])
    end_bar = int(p["range_end_bar"])
    in_range = (bar >= start_bar) & (bar <= end_bar)
    range_high_by_day = df["high"].where(in_range).groupby(dates, sort=False).max()
    range_low_by_day = df["low"].where(in_range).groupby(dates, sort=False).min()
    range_high = pd.Series(dates.map(range_high_by_day).to_numpy(), index=df.index, dtype=float)
    range_low = pd.Series(dates.map(range_low_by_day).to_numpy(), index=df.index, dtype=float)
    ok = bar > end_bar
    long = ok & _cross_above(df["close"], range_high)
    short = ok & _cross_below(df["close"], range_low)
    return _signal(df.index, long, short)


def b3_afternoon_failed_extreme_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    bar = _bar_no(df)
    day_high = _day_cummax(df["high"], df).shift(1)
    day_low = _day_cummin(df["low"], df).shift(1)
    vwap = intraday_vwap(df)
    ok = bar >= int(p["start_bar"])
    long = ok & (df["low"] < day_low) & (df["close"] > day_low) & (df["close"] > vwap)
    short = ok & (df["high"] > day_high) & (df["close"] < day_high) & (df["close"] < vwap)
    return _signal(df.index, long, short)


def b3_prev_close_magnet_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    _, _, prev_close = previous_day_levels(df)
    day_open = _day_open(df)
    vwap = intraday_vwap(df)
    bar = _bar_no(df)
    gap = (day_open / prev_close.replace(0, np.nan) - 1) * 100
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"])) & (gap.abs() >= float(p["gap_pct"]))
    long = ok & (gap < 0) & (df["close"] > vwap) & _cross_above(df["close"], day_open)
    short = ok & (gap > 0) & (df["close"] < vwap) & _cross_below(df["close"], day_open)
    return _signal(df.index, long, short)


def b3_weekday_first_hour_bias_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    timestamps = pd.to_datetime(df["timestamp"])
    weekday = timestamps.dt.weekday
    day_open = _day_open(df)
    bar = _bar_no(df)
    ret = (df["close"] / day_open - 1) * 100
    at_decision = bar == int(p["decision_bar"])
    selected_day = weekday == int(p["weekday"])
    threshold = float(p["first_hour_pct"])
    mode = str(p["mode"])
    trend_long = selected_day & at_decision & (ret > threshold)
    trend_short = selected_day & at_decision & (ret < -threshold)
    if mode == "fade":
        return _signal(df.index, trend_short, trend_long)
    return _signal(df.index, trend_long, trend_short)


def b3_opening_range_inside_value_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    prev_high, prev_low, prev_close = previous_day_levels(df)
    or_high, or_low, active = opening_range(df, int(p["or_minutes"]))
    day_open = _day_open(df)
    inside_prev_value = (or_high < prev_high) & (or_low > prev_low) & (((day_open / prev_close.replace(0, np.nan) - 1) * 100).abs() < float(p["open_gap_max_pct"]))
    ok = active & inside_prev_value
    long = ok & _cross_above(df["close"], or_high)
    short = ok & _cross_below(df["close"], or_low)
    return _signal(df.index, long, short)


def b3_late_liquidity_run_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    bar = _bar_no(df)
    volume_avg = df["volume"].rolling(30, min_periods=30).mean()
    high = df["high"].rolling(int(p["lookback"]), min_periods=int(p["lookback"])).max().shift(1)
    low = df["low"].rolling(int(p["lookback"]), min_periods=int(p["lookback"])).min().shift(1)
    day_open = _day_open(df)
    day_ret = (df["close"] / day_open - 1) * 100
    ok = (bar >= int(p["start_bar"])) & (df["volume"] > volume_avg * float(p["volume_mult"]))
    long = ok & (day_ret > float(p["min_day_ret"])) & (df["close"] > high)
    short = ok & (day_ret < -float(p["min_day_ret"])) & (df["close"] < low)
    return _signal(df.index, long, short)


STRATEGIES.update(
    {
        "b3_prev_day_sweep_reversal": StrategyDefinition(
            "b3_prev_day_sweep_reversal",
            "Fades sweeps of the previous day high or low after price closes back inside.",
            {"sweep_pct": 0.05, "start_bar": 12, "end_bar": 55, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"sweep_pct": [0.03, 0.08], "start_bar": [9, 12], "end_bar": [45, 60]}),
            b3_prev_day_sweep_reversal,
        ),
        "b3_open_gap_stall_fade": StrategyDefinition(
            "b3_open_gap_stall_fade",
            "Fades opening gaps that have stalled by the first-hour decision bar.",
            {"gap_pct": 0.6, "decision_bar": 12, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"gap_pct": [0.4, 0.8], "decision_bar": [9, 12]}),
            b3_open_gap_stall_fade,
        ),
        "b3_pivot_reclaim_h2c": StrategyDefinition(
            "b3_pivot_reclaim_h2c",
            "Trades reclaim or rejection of the previous-day floor pivot.",
            {"start_bar": 9, "end_bar": 42, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"start_bar": [6, 12], "end_bar": [36, 50]}),
            b3_pivot_reclaim_h2c,
        ),
        "b3_cpr_escape_h2c": StrategyDefinition(
            "b3_cpr_escape_h2c",
            "Trades escapes from a narrow prior-day central pivot range when the open starts inside it.",
            {"start_bar": 8, "max_width_pct": 0.5, "volume_mult": 1.2, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"max_width_pct": [0.35, 0.60], "volume_mult": [1.0, 1.4], "start_bar": [6, 10]}),
            b3_cpr_escape_h2c,
        ),
        "b3_two_day_extreme_h2c": StrategyDefinition(
            "b3_two_day_extreme_h2c",
            "Follows breaks of the prior two-day extreme only after the current session already confirms direction.",
            {"buffer_pct": 0.03, "min_day_ret": 0.35, "start_bar": 18, "end_bar": 55, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"buffer_pct": [0.02, 0.06], "min_day_ret": [0.25, 0.50], "start_bar": [15, 24]}),
            b3_two_day_extreme_h2c,
        ),
        "b3_three_day_reversal_h2c": StrategyDefinition(
            "b3_three_day_reversal_h2c",
            "Fades extended three-day moves when the session reclaims or loses its open.",
            {"three_day_ret_pct": 3.0, "start_bar": 12, "end_bar": 45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"three_day_ret_pct": [2.0, 3.5], "start_bar": [9, 12], "end_bar": [36, 50]}),
            b3_three_day_reversal_h2c,
        ),
        "b3_nr4_false_break_h2c": StrategyDefinition(
            "b3_nr4_false_break_h2c",
            "Fades failed opening-range breaks after a prior narrow-range day.",
            {"or_minutes": 30, "end_bar": 48, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"or_minutes": [20, 30], "end_bar": [42, 55]}),
            b3_nr4_false_break_h2c,
        ),
        "b3_morning_midpoint_trap": StrategyDefinition(
            "b3_morning_midpoint_trap",
            "Reverses failed morning range breaks after price crosses the opening-range midpoint.",
            {"or_minutes": 30, "end_bar": 45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"or_minutes": [20, 30, 45], "end_bar": [36, 50]}),
            b3_morning_midpoint_trap,
        ),
        "b3_opening_drive_pullback_h2c": StrategyDefinition(
            "b3_opening_drive_pullback_h2c",
            "Joins a strong opening drive only after a later pullback to a short EMA.",
            {"drive_bars": 4, "drive_pct": 0.45, "volume_mult": 1.1, "pullback_ema": 13, "end_bar": 36, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"drive_pct": [0.30, 0.60], "volume_mult": [0.9, 1.3], "pullback_ema": [9, 13]}),
            b3_opening_drive_pullback_h2c,
        ),
        "b3_vwap_compression_release": StrategyDefinition(
            "b3_vwap_compression_release",
            "Breaks out after price compresses tightly around intraday VWAP.",
            {"lookback": 18, "max_dist_pct": 0.18, "slope_bars": 6, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"lookback": [12, 18], "max_dist_pct": [0.12, 0.22], "slope_bars": [4, 8]}),
            b3_vwap_compression_release,
        ),
        "b3_inside_bar_expansion_h2c": StrategyDefinition(
            "b3_inside_bar_expansion_h2c",
            "Trades expansion out of a completed 5-minute inside bar only when the expansion range is large.",
            {"atr_period": 14, "range_atr_mult": 0.7, "close_location": 0.72, "start_bar": 12, "end_bar": 60, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"range_atr_mult": [0.5, 0.9], "close_location": [0.68, 0.78], "start_bar": [9, 15]}),
            b3_inside_bar_expansion_h2c,
        ),
        "b3_pinbar_reversal_h2c": StrategyDefinition(
            "b3_pinbar_reversal_h2c",
            "Fades large wick rejection candles at fresh session extremes.",
            {"wick_body_mult": 2.2, "start_bar": 12, "end_bar": 58, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"wick_body_mult": [1.8, 2.6], "start_bar": [9, 15], "end_bar": [45, 62]}),
            b3_pinbar_reversal_h2c,
        ),
        "b3_ema_slope_acceleration_h2c": StrategyDefinition(
            "b3_ema_slope_acceleration_h2c",
            "Trades acceleration only when fast EMA alignment and slow EMA slope agree.",
            {"fast_ema": 8, "slow_ema": 34, "slope_bars": 10, "slope_pct": 0.08, "impulse_bars": 4, "impulse_pct": 0.25, "start_bar": 10, "end_bar": 55, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"fast_ema": [8, 13], "slow_ema": [34, 55], "impulse_pct": [0.20, 0.35]}),
            b3_ema_slope_acceleration_h2c,
        ),
        "b3_realized_vol_contraction_break": StrategyDefinition(
            "b3_realized_vol_contraction_break",
            "Trades intraday range breaks only after yesterday's realized range contracted versus its 20-day norm.",
            {"lookback": 24, "range_frac": 0.75, "start_bar": 14, "end_bar": 55, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"lookback": [18, 30], "range_frac": [0.65, 0.85], "start_bar": [12, 18]}),
            b3_realized_vol_contraction_break,
        ),
        "b3_afternoon_range_resolve_h2c": StrategyDefinition(
            "b3_afternoon_range_resolve_h2c",
            "Trades the resolution of the late-morning/midday range after it is fully formed.",
            {"range_start_bar": 21, "range_end_bar": 45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"range_start_bar": [18, 24], "range_end_bar": [42, 48]}),
            b3_afternoon_range_resolve_h2c,
        ),
        "b3_afternoon_failed_extreme_h2c": StrategyDefinition(
            "b3_afternoon_failed_extreme_h2c",
            "Fades late-session failed fresh highs or lows only when price also crosses back through VWAP.",
            {"start_bar": 54, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"start_bar": [48, 54, 60]}),
            b3_afternoon_failed_extreme_h2c,
        ),
        "b3_prev_close_magnet_h2c": StrategyDefinition(
            "b3_prev_close_magnet_h2c",
            "Fades large gaps once price reclaims VWAP and the session open toward the previous close magnet.",
            {"gap_pct": 0.6, "start_bar": 12, "end_bar": 48, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"gap_pct": [0.4, 0.8], "start_bar": [9, 15], "end_bar": [42, 55]}),
            b3_prev_close_magnet_h2c,
        ),
        "b3_weekday_first_hour_bias_h2c": StrategyDefinition(
            "b3_weekday_first_hour_bias_h2c",
            "Tests whether first-hour continuation or fade has a stable day-of-week edge.",
            {"weekday": 2, "mode": "trend", "decision_bar": 12, "first_hour_pct": 0.35, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"weekday": [0, 1, 2, 3, 4], "mode": ["trend", "fade"], "first_hour_pct": [0.25, 0.50]}),
            b3_weekday_first_hour_bias_h2c,
        ),
        "b3_opening_range_inside_value_h2c": StrategyDefinition(
            "b3_opening_range_inside_value_h2c",
            "Trades breaks from an opening range that formed completely inside the previous day's range.",
            {"or_minutes": 45, "open_gap_max_pct": 0.35, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"or_minutes": [30, 45], "open_gap_max_pct": [0.25, 0.50]}),
            b3_opening_range_inside_value_h2c,
        ),
        "b3_late_liquidity_run_h2c": StrategyDefinition(
            "b3_late_liquidity_run_h2c",
            "Follows late-session liquidity runs when direction, volume, and range extension align.",
            {"lookback": 24, "start_bar": 57, "volume_mult": 1.3, "min_day_ret": 0.6, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"lookback": [18, 30], "start_bar": [54, 60], "volume_mult": [1.1, 1.5], "min_day_ret": [0.4, 0.8]}),
            b3_late_liquidity_run_h2c,
        ),
    }
)


def _first_n_ohlcv(df: pd.DataFrame, n: int) -> dict[str, pd.Series]:
    dates = _dates(df)
    bar = _bar_no(df)
    mask = bar < int(n)
    grouped = df.loc[mask].groupby(dates.loc[mask], sort=False)
    stats = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    return {column: pd.Series(dates.map(stats[column]).to_numpy(), index=df.index, dtype=float) for column in stats.columns}


def _daily_flag_map(df: pd.DataFrame, flag: pd.Series) -> pd.Series:
    dates = _dates(df)
    return pd.Series(dates.map(flag).fillna(False).to_numpy(), index=df.index, dtype=bool)


def b4_prev_close_location_follow_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    ctx = _daily_context(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    day_ret = (df["close"] / day_open - 1) * 100
    at_decision = bar == int(p["decision_bar"])
    close_pos_high = ctx["prev_close_pos"] >= float(p["close_pos"])
    close_pos_low = ctx["prev_close_pos"] <= 1 - float(p["close_pos"])
    long = at_decision & close_pos_high & (day_ret >= float(p["confirm_pct"])) & (df["close"] > ctx["prev_close"])
    short = at_decision & close_pos_low & (day_ret <= -float(p["confirm_pct"])) & (df["close"] < ctx["prev_close"])
    return _signal(df.index, long, short)


def b4_wide_range_day_reversal_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    ctx = _daily_context(df)
    bar = _bar_no(df)
    wide = ctx["prev_range"] > ctx["range_med20"] * float(p["range_mult"])
    ok = wide & (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & (ctx["prev_close_pos"] <= 1 - float(p["close_pos"])) & _cross_above(df["close"], ctx["prev_close"])
    short = ok & (ctx["prev_close_pos"] >= float(p["close_pos"])) & _cross_below(df["close"], ctx["prev_close"])
    return _signal(df.index, long, short)


def b4_opening_efficiency_drive_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    stats = _first_n_ohlcv(df, int(p["open_bars"]))
    ctx = _daily_context(df)
    bar = _bar_no(df)
    net = stats["close"] - stats["open"]
    opening_range = (stats["high"] - stats["low"]).replace(0, np.nan)
    efficiency = net.abs() / opening_range
    volume_ratio = stats["volume"] / ctx["volume_med20"].replace(0, np.nan)
    at_decision = bar == int(p["open_bars"])
    ok = at_decision & (efficiency >= float(p["efficiency"])) & (volume_ratio >= float(p["volume_ratio"]))
    long = ok & (net > 0) & (df["close"] > intraday_vwap(df))
    short = ok & (net < 0) & (df["close"] < intraday_vwap(df))
    return _signal(df.index, long, short)


def b4_opening_climax_fade_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    stats = _first_n_ohlcv(df, int(p["open_bars"]))
    ctx = _daily_context(df)
    bar = _bar_no(df)
    opening_range = stats["high"] - stats["low"]
    net = stats["close"] - stats["open"]
    range_large = opening_range > ctx["range_med20"] * float(p["range_frac"])
    volume_ratio = stats["volume"] / ctx["volume_med20"].replace(0, np.nan)
    at_decision = bar == int(p["decision_bar"])
    long = at_decision & range_large & (volume_ratio >= float(p["volume_ratio"])) & (net < 0) & (df["close"] > stats["low"] + opening_range * float(p["reclaim_frac"]))
    short = at_decision & range_large & (volume_ratio >= float(p["volume_ratio"])) & (net > 0) & (df["close"] < stats["high"] - opening_range * float(p["reclaim_frac"]))
    return _signal(df.index, long, short)


def b4_relative_volume_trend_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    stats = _first_n_ohlcv(df, int(p["decision_bar"]))
    ctx = _daily_context(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    rel_volume = stats["volume"] / ctx["volume_med20"].replace(0, np.nan)
    day_ret = (df["close"] / day_open - 1) * 100
    at_decision = bar == int(p["decision_bar"])
    long = at_decision & (rel_volume >= float(p["volume_ratio"])) & (day_ret >= float(p["move_pct"])) & (df["close"] > intraday_vwap(df))
    short = at_decision & (rel_volume >= float(p["volume_ratio"])) & (day_ret <= -float(p["move_pct"])) & (df["close"] < intraday_vwap(df))
    return _signal(df.index, long, short)


def b4_low_volume_drift_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    stats = _first_n_ohlcv(df, int(p["decision_bar"]))
    ctx = _daily_context(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    rel_volume = stats["volume"] / ctx["volume_med20"].replace(0, np.nan)
    day_ret = (df["close"] / day_open - 1) * 100
    vwap = intraday_vwap(df)
    at_decision = bar == int(p["decision_bar"])
    long = at_decision & (rel_volume <= float(p["max_volume_ratio"])) & (day_ret >= float(p["move_pct"])) & (df["close"] > vwap)
    short = at_decision & (rel_volume <= float(p["max_volume_ratio"])) & (day_ret <= -float(p["move_pct"])) & (df["close"] < vwap)
    return _signal(df.index, long, short)


def b4_gap_fill_reject_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    _, _, prev_close = previous_day_levels(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    gap = (day_open / prev_close.replace(0, np.nan) - 1) * 100
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"])) & (gap.abs() >= float(p["gap_pct"]))
    long = ok & (gap > 0) & (df["low"] <= prev_close) & (df["close"] > prev_close) & (df["close"] > df["open"])
    short = ok & (gap < 0) & (df["high"] >= prev_close) & (df["close"] < prev_close) & (df["close"] < df["open"])
    return _signal(df.index, long, short)


def b4_gap_fill_accept_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    _, _, prev_close = previous_day_levels(df)
    day_open = _day_open(df)
    bar = _bar_no(df)
    gap = (day_open / prev_close.replace(0, np.nan) - 1) * 100
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"])) & (gap.abs() >= float(p["gap_pct"]))
    long = ok & (gap < 0) & _cross_above(df["close"], prev_close)
    short = ok & (gap > 0) & _cross_below(df["close"], prev_close)
    return _signal(df.index, long, short)


def b4_prior20_breakout_failure_fade_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    dates = _dates(df)
    daily = df.groupby(dates, sort=False).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    high20 = daily["high"].rolling(20).max().shift(1)
    low20 = daily["low"].rolling(20).min().shift(1)
    failed_up = ((daily["high"] > high20) & (daily["close"] < high20)).shift(1)
    failed_down = ((daily["low"] < low20) & (daily["close"] > low20)).shift(1)
    fail_up_today = _daily_flag_map(df, failed_up)
    fail_down_today = _daily_flag_map(df, failed_down)
    day_open = _day_open(df)
    bar = _bar_no(df)
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & fail_down_today & _cross_above(df["close"], day_open)
    short = ok & fail_up_today & _cross_below(df["close"], day_open)
    return _signal(df.index, long, short)


def b4_prior20_breakout_followthrough_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    dates = _dates(df)
    daily = df.groupby(dates, sort=False).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    high20 = daily["high"].rolling(20).max().shift(1)
    low20 = daily["low"].rolling(20).min().shift(1)
    closed_up = (daily["close"] > high20).shift(1)
    closed_down = (daily["close"] < low20).shift(1)
    up_today = _daily_flag_map(df, closed_up)
    down_today = _daily_flag_map(df, closed_down)
    ctx = _daily_context(df)
    bar = _bar_no(df)
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & up_today & _cross_above(df["close"], ctx["prev_close"])
    short = ok & down_today & _cross_below(df["close"], ctx["prev_close"])
    return _signal(df.index, long, short)


def b4_midday_vwap_bandwalk_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    vwap = intraday_vwap(df)
    bar = _bar_no(df)
    lookback = int(p["lookback"])
    above = (df["close"] > vwap).rolling(lookback, min_periods=lookback).sum() >= lookback * float(p["side_frac"])
    below = (df["close"] < vwap).rolling(lookback, min_periods=lookback).sum() >= lookback * float(p["side_frac"])
    slope = vwap.diff(int(p["slope_bars"]))
    at_decision = bar == int(p["decision_bar"])
    long = at_decision & above & (slope > 0)
    short = at_decision & below & (slope < 0)
    return _signal(df.index, long, short)


def b4_vwap_reversion_acceptance_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    vwap = intraday_vwap(df)
    bar = _bar_no(df)
    distance = (df["close"] - vwap) / vwap.replace(0, np.nan) * 100
    was_high = distance.rolling(int(p["lookback"]), min_periods=int(p["lookback"])).max().shift(1) >= float(p["stretch_pct"])
    was_low = distance.rolling(int(p["lookback"]), min_periods=int(p["lookback"])).min().shift(1) <= -float(p["stretch_pct"])
    accepted_inside = distance.abs().rolling(2, min_periods=2).max() <= float(p["accept_pct"])
    ok = (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"])) & accepted_inside
    long = ok & was_low & (df["close"] > df["open"])
    short = ok & was_high & (df["close"] < df["open"])
    return _signal(df.index, long, short)


def b4_morning_range_quartile_reversal_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    or_high, or_low, active = opening_range(df, int(p["or_minutes"]))
    range_size = or_high - or_low
    q25 = or_low + range_size * 0.25
    q75 = or_low + range_size * 0.75
    bar = _bar_no(df)
    ok = active & (bar >= int(p["start_bar"])) & (bar <= int(p["end_bar"]))
    long = ok & (df["low"] < q25) & (df["close"] > q25) & (df["close"] > df["open"])
    short = ok & (df["high"] > q75) & (df["close"] < q75) & (df["close"] < df["open"])
    return _signal(df.index, long, short)


def b4_late_compression_break_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    dates = _dates(df)
    bar = _bar_no(df)
    compression_end = int(p["compression_end_bar"])
    compression = bar <= compression_end
    comp_high_by_day = df["high"].where(compression).groupby(dates, sort=False).max()
    comp_low_by_day = df["low"].where(compression).groupby(dates, sort=False).min()
    comp_high = pd.Series(dates.map(comp_high_by_day).to_numpy(), index=df.index, dtype=float)
    comp_low = pd.Series(dates.map(comp_low_by_day).to_numpy(), index=df.index, dtype=float)
    ctx = _daily_context(df)
    small = (comp_high - comp_low) < ctx["range_med20"] * float(p["range_frac"])
    ok = (bar >= int(p["start_bar"])) & small
    long = ok & _cross_above(df["close"], comp_high)
    short = ok & _cross_below(df["close"], comp_low)
    return _signal(df.index, long, short)


def b4_prior_trend_first_hour_reversal_h2c(df: pd.DataFrame, p: dict[str, float | int | str]) -> pd.Series:
    ctx = _daily_context(df)
    stats = _first_n_ohlcv(df, int(p["open_bars"]))
    day_open = _day_open(df)
    bar = _bar_no(df)
    vwap = intraday_vwap(df)
    first_hour_ret = (stats["close"] / day_open - 1) * 100
    ok = bar == int(p["decision_bar"])
    long = ok & (ctx["prev5_ret"] > float(p["trend_pct"])) & (first_hour_ret <= -float(p["counter_pct"])) & (df["close"] > vwap)
    short = ok & (ctx["prev5_ret"] < -float(p["trend_pct"])) & (first_hour_ret >= float(p["counter_pct"])) & (df["close"] < vwap)
    return _signal(df.index, long, short)


STRATEGIES.update(
    {
        "b4_prev_close_location_follow_h2c": StrategyDefinition(
            "b4_prev_close_location_follow_h2c",
            "Follows a prior-day strong close only when the current first hour confirms beyond previous close.",
            {"close_pos": 0.75, "confirm_pct": 0.35, "decision_bar": 12, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"close_pos": [0.70, 0.82], "confirm_pct": [0.25, 0.50], "decision_bar": [9, 12]}),
            b4_prev_close_location_follow_h2c,
        ),
        "b4_wide_range_day_reversal_h2c": StrategyDefinition(
            "b4_wide_range_day_reversal_h2c",
            "Fades yesterday's wide-range extreme close after price crosses the previous close in the opposite direction.",
            {"range_mult": 1.4, "close_pos": 0.75, "start_bar": 9, "end_bar": 45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"range_mult": [1.2, 1.6], "close_pos": [0.70, 0.82], "start_bar": [9, 15]}),
            b4_wide_range_day_reversal_h2c,
        ),
        "b4_opening_efficiency_drive_h2c": StrategyDefinition(
            "b4_opening_efficiency_drive_h2c",
            "Follows high-efficiency directional opening auctions with elevated relative volume.",
            {"open_bars": 9, "efficiency": 0.75, "volume_ratio": 0.18, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"open_bars": [6, 9, 12], "efficiency": [0.65, 0.80], "volume_ratio": [0.12, 0.22]}),
            b4_opening_efficiency_drive_h2c,
        ),
        "b4_opening_climax_fade_h2c": StrategyDefinition(
            "b4_opening_climax_fade_h2c",
            "Fades opening-climax moves when a large first-hour range is partly reclaimed.",
            {"open_bars": 9, "decision_bar": 12, "range_frac": 0.35, "volume_ratio": 0.18, "reclaim_frac": 0.35, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"range_frac": [0.25, 0.45], "reclaim_frac": [0.30, 0.45], "volume_ratio": [0.12, 0.22]}),
            b4_opening_climax_fade_h2c,
        ),
        "b4_relative_volume_trend_h2c": StrategyDefinition(
            "b4_relative_volume_trend_h2c",
            "Follows first-hour trend only when cumulative opening volume is high relative to prior daily volume.",
            {"decision_bar": 12, "volume_ratio": 0.20, "move_pct": 0.45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"decision_bar": [9, 12], "volume_ratio": [0.14, 0.24], "move_pct": [0.30, 0.60]}),
            b4_relative_volume_trend_h2c,
        ),
        "b4_low_volume_drift_h2c": StrategyDefinition(
            "b4_low_volume_drift_h2c",
            "Tests whether low-participation first-hour drifts continue through the session.",
            {"decision_bar": 12, "max_volume_ratio": 0.10, "move_pct": 0.25, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"max_volume_ratio": [0.08, 0.12], "move_pct": [0.20, 0.35], "decision_bar": [9, 12]}),
            b4_low_volume_drift_h2c,
        ),
        "b4_gap_fill_reject_h2c": StrategyDefinition(
            "b4_gap_fill_reject_h2c",
            "Follows the original gap direction only after the previous close is tested and rejected.",
            {"gap_pct": 0.5, "start_bar": 9, "end_bar": 45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"gap_pct": [0.35, 0.70], "start_bar": [6, 12], "end_bar": [36, 55]}),
            b4_gap_fill_reject_h2c,
        ),
        "b4_gap_fill_accept_h2c": StrategyDefinition(
            "b4_gap_fill_accept_h2c",
            "Trades acceptance through the previous close after a large opening gap fills.",
            {"gap_pct": 0.5, "start_bar": 9, "end_bar": 45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"gap_pct": [0.35, 0.70], "start_bar": [6, 12], "end_bar": [36, 55]}),
            b4_gap_fill_accept_h2c,
        ),
        "b4_prior20_breakout_failure_fade_h2c": StrategyDefinition(
            "b4_prior20_breakout_failure_fade_h2c",
            "Fades stocks the day after a failed 20-day high or low breakout.",
            {"start_bar": 9, "end_bar": 50, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"start_bar": [9, 15], "end_bar": [42, 55]}),
            b4_prior20_breakout_failure_fade_h2c,
        ),
        "b4_prior20_breakout_followthrough_h2c": StrategyDefinition(
            "b4_prior20_breakout_followthrough_h2c",
            "Follows stocks the day after a close beyond a prior 20-day high or low.",
            {"start_bar": 9, "end_bar": 50, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"start_bar": [9, 15], "end_bar": [42, 55]}),
            b4_prior20_breakout_followthrough_h2c,
        ),
        "b4_midday_vwap_bandwalk_h2c": StrategyDefinition(
            "b4_midday_vwap_bandwalk_h2c",
            "Follows a midday VWAP band-walk when most recent bars stayed on one side of VWAP.",
            {"lookback": 18, "side_frac": 0.85, "slope_bars": 8, "decision_bar": 42, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"lookback": [12, 18], "side_frac": [0.75, 0.90], "decision_bar": [36, 45]}),
            b4_midday_vwap_bandwalk_h2c,
        ),
        "b4_vwap_reversion_acceptance_h2c": StrategyDefinition(
            "b4_vwap_reversion_acceptance_h2c",
            "Fades earlier VWAP stretch only after price accepts back near VWAP for two bars.",
            {"lookback": 18, "stretch_pct": 0.75, "accept_pct": 0.20, "start_bar": 18, "end_bar": 55, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"stretch_pct": [0.50, 0.90], "accept_pct": [0.15, 0.25], "start_bar": [15, 24]}),
            b4_vwap_reversion_acceptance_h2c,
        ),
        "b4_morning_range_quartile_reversal_h2c": StrategyDefinition(
            "b4_morning_range_quartile_reversal_h2c",
            "Fades pushes into the outer quartile of the formed morning range.",
            {"or_minutes": 45, "start_bar": 18, "end_bar": 55, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"or_minutes": [30, 45], "start_bar": [15, 24], "end_bar": [48, 60]}),
            b4_morning_range_quartile_reversal_h2c,
        ),
        "b4_late_compression_break_h2c": StrategyDefinition(
            "b4_late_compression_break_h2c",
            "Trades late-session breaks only after the day stayed compressed versus prior 20-day range.",
            {"compression_end_bar": 48, "start_bar": 54, "range_frac": 0.45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"compression_end_bar": [42, 48], "start_bar": [54, 60], "range_frac": [0.35, 0.55]}),
            b4_late_compression_break_h2c,
        ),
        "b4_prior_trend_first_hour_reversal_h2c": StrategyDefinition(
            "b4_prior_trend_first_hour_reversal_h2c",
            "Uses a five-day prior trend and fades the first-hour countertrend move after VWAP reclaim/rejection.",
            {"open_bars": 12, "decision_bar": 15, "trend_pct": 3.0, "counter_pct": 0.45, "stop_pct": 1.0, "target_pct": 99.0, "max_hold_bars": 75},
            _with_h2c({"trend_pct": [2.0, 3.5], "counter_pct": [0.30, 0.60], "decision_bar": [12, 18]}),
            b4_prior_trend_first_hour_reversal_h2c,
        ),
    }
)


def all_strategies() -> list[StrategyDefinition]:
    return list(STRATEGIES.values())


def get_strategy(name: str) -> StrategyDefinition:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}") from exc
