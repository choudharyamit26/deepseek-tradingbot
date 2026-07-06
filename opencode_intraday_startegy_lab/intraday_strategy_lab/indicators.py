from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False, min_periods=max(1, int(period))).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(int(period), min_periods=int(period)).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / int(period), adjust=False, min_periods=int(period)).mean()
    average_loss = loss.ewm(alpha=1 / int(period), adjust=False, min_periods=int(period)).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["timestamp"]).dt.date
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3
    volume = df["volume"].astype(float).clip(lower=0)
    pv = typical * volume
    cumulative_pv = pv.groupby(dates).cumsum()
    cumulative_volume = volume.groupby(dates).cumsum().replace(0, np.nan)
    return cumulative_pv / cumulative_volume


def bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.astype(float).rolling(int(period), min_periods=int(period)).mean()
    spread = close.astype(float).rolling(int(period), min_periods=int(period)).std(ddof=0)
    upper = middle + float(std_dev) * spread
    lower = middle - float(std_dev) * spread
    return upper, middle, lower


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, int(fast)) - ema(close, int(slow))
    signal_line = ema(line, int(signal))
    histogram = line - signal_line
    return line, signal_line, histogram


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    close = df["close"].astype(float).reset_index(drop=True)
    average_true_range = atr(df.reset_index(drop=True), int(period)).reset_index(drop=True)
    hl2 = (high + low) / 2
    basic_upper = hl2 + float(multiplier) * average_true_range
    basic_lower = hl2 - float(multiplier) * average_true_range
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend_line = pd.Series(np.nan, index=close.index, dtype=float)
    direction = pd.Series(1, index=close.index, dtype=int)

    for i in range(1, len(close)):
        if pd.isna(average_true_range.iloc[i]):
            direction.iloc[i] = direction.iloc[i - 1]
            continue
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]
        if close.iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        trend_line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return trend_line.reindex(df.index), direction.reindex(df.index)


def opening_range(df: pd.DataFrame, minutes: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    timestamps = pd.to_datetime(df["timestamp"])
    high_out = pd.Series(np.nan, index=df.index, dtype=float)
    low_out = pd.Series(np.nan, index=df.index, dtype=float)
    active = pd.Series(False, index=df.index, dtype=bool)
    for _, group in df.groupby(timestamps.dt.date, sort=False):
        start = pd.to_datetime(group["timestamp"]).iloc[0]
        cutoff = start + pd.Timedelta(minutes=int(minutes))
        mask = pd.to_datetime(group["timestamp"]) < cutoff
        or_slice = group.loc[mask]
        if or_slice.empty:
            continue
        high_out.loc[group.index] = float(or_slice["high"].max())
        low_out.loc[group.index] = float(or_slice["low"].min())
        active.loc[group.index] = pd.to_datetime(group["timestamp"]) >= cutoff
    return high_out, low_out, active


def previous_day_levels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    timestamps = pd.to_datetime(df["timestamp"])
    daily = df.groupby(timestamps.dt.date).agg(
        previous_high=("high", "max"), previous_low=("low", "min"), previous_close=("close", "last")
    )
    shifted = daily.shift(1)
    dates = timestamps.dt.date
    prev_high = dates.map(shifted["previous_high"])
    prev_low = dates.map(shifted["previous_low"])
    prev_close = dates.map(shifted["previous_close"])
    return (
        pd.Series(prev_high.to_numpy(), index=df.index, dtype=float),
        pd.Series(prev_low.to_numpy(), index=df.index, dtype=float),
        pd.Series(prev_close.to_numpy(), index=df.index, dtype=float),
    )


def first_bar_levels(df: pd.DataFrame, bars: int = 1) -> tuple[pd.Series, pd.Series, pd.Series]:
    timestamps = pd.to_datetime(df["timestamp"])
    high_out = pd.Series(np.nan, index=df.index, dtype=float)
    low_out = pd.Series(np.nan, index=df.index, dtype=float)
    active = pd.Series(False, index=df.index, dtype=bool)
    for _, group in df.groupby(timestamps.dt.date, sort=False):
        first = group.head(int(bars))
        if first.empty:
            continue
        high_out.loc[group.index] = float(first["high"].max())
        low_out.loc[group.index] = float(first["low"].min())
        active.loc[group.index] = False
        active.loc[group.index[int(bars) :]] = True
    return high_out, low_out, active


def parse_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time value: {value}")
    return time(int(parts[0]), int(parts[1]))


def between_times(timestamps: pd.Series, start: str, end: str) -> pd.Series:
    start_time = parse_time(start)
    end_time = parse_time(end)
    times = pd.to_datetime(timestamps).dt.time
    return (times >= start_time) & (times <= end_time)
