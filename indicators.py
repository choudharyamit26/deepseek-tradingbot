"""Shared technical-indicator stack.

Single source of truth for the 14-indicator snapshot used by the entry bot,
the exit guardian, and the replay engine. Previously duplicated (and already
diverging) between stock_trading_bot.py and position_exit_guardian.py.
"""

import pandas as pd
import talib


def calculate_technical_indicators(df: pd.DataFrame, min_bars: int = 5) -> dict:
    """Compute the indicator snapshot for the latest bar of an OHLCV frame.

    Returns {} if there are fewer than min_bars rows. The input frame is not
    modified.
    """
    if df is None or len(df) < min_bars:
        return {}
    df = df.copy()

    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
    df["sma_20"] = talib.SMA(df["close"], timeperiod=20)
    df["ema_9"] = talib.EMA(df["close"], timeperiod=9)
    upper, _, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    df["bb_percent_b"] = (df["close"] - lower) / (upper - lower)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
    df["mfi"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)

    # VWAP with daily reset
    if hasattr(df.index, "date"):
        vwap_parts = []
        for _date, group in df.groupby(df.index.date):
            cum_vol = group["volume"].cumsum().clip(lower=1e-9)
            cum_vp = (group["close"] * group["volume"]).cumsum()
            vwap_parts.append(cum_vp / cum_vol)
        df["vwap"] = pd.concat(vwap_parts)
    else:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum().clip(lower=1e-9)

    # 10-bar volume lookback: a 20-bar avg inflated by the morning session
    # made afternoon ratios read ~0.00x
    vol_lookback = min(10, len(df))
    vol_ma = df["volume"].rolling(window=vol_lookback).mean().clip(lower=1e-9)
    df["volume_ratio"] = df["volume"] / vol_ma
    df["resistance"] = df["high"].rolling(window=20).max()
    df["support"] = df["low"].rolling(window=20).min()

    # ── Order-Flow Imbalance (leading / microstructure) ─────────────────────
    # A pressure proxy the lagging stack (RSI/ADX/MFI/VWAP) does not capture.
    # Close Location Value: where each bar closes within its range, in [-1,+1]
    # (+1 = closed on the high = net buying, -1 = closed on the low = net
    # selling). Weight by volume → signed volume per bar. The rolling sum of
    # signed volume divided by rolling total volume gives a normalized
    # accumulation/distribution imbalance in [-1,+1] that leads price because
    # participants position before the move shows up in close-to-close returns.
    rng = (df["high"] - df["low"]).clip(lower=1e-9)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng  # [-1,+1]
    signed_vol = clv * df["volume"]
    ofi_lookback = min(10, len(df))
    signed_sum = signed_vol.rolling(window=ofi_lookback).sum()
    vol_sum = df["volume"].rolling(window=ofi_lookback).sum().clip(lower=1e-9)
    df["ofi"] = (signed_sum / vol_sum).clip(-1, 1)
    # Short-horizon slope of OFI: is pressure building or fading right now?
    df["ofi_trend"] = df["ofi"] - df["ofi"].shift(3)

    latest = df.iloc[-1]
    close_val = round(latest["close"], 2)
    vwap_val = round(latest["vwap"], 2) if not pd.isna(latest["vwap"]) else close_val
    vwap_distance_pct = round(((close_val - vwap_val) / vwap_val) * 100, 3) if vwap_val > 0 else 0

    return {
        "close": close_val,
        "high": round(latest["high"], 2),
        "low": round(latest["low"], 2),
        "rsi": round(latest["rsi"], 2) if not pd.isna(latest["rsi"]) else 50,
        "macd": round(latest["macd"], 2) if not pd.isna(latest["macd"]) else 0,
        "macd_signal": round(latest["macd_signal"], 2) if not pd.isna(latest["macd_signal"]) else 0,
        "sma_20": round(latest["sma_20"], 2) if not pd.isna(latest["sma_20"]) else close_val,
        "ema_9": round(latest["ema_9"], 2) if not pd.isna(latest["ema_9"]) else close_val,
        "bb_percent_b": round(latest["bb_percent_b"], 3) if not pd.isna(latest["bb_percent_b"]) else 0.5,
        "atr": round(latest["atr"], 2) if not pd.isna(latest["atr"]) else 1,
        "vwap": vwap_val,
        "vwap_distance_pct": vwap_distance_pct,
        "volume_ratio": round(latest["volume_ratio"], 2) if not pd.isna(latest["volume_ratio"]) else 1,
        "support": round(latest["support"], 2) if not pd.isna(latest["support"]) else latest["low"],
        "resistance": round(latest["resistance"], 2) if not pd.isna(latest["resistance"]) else latest["high"],
        "adx": round(latest["adx"], 2) if not pd.isna(latest["adx"]) else 20,
        "mfi": round(latest["mfi"], 2) if not pd.isna(latest["mfi"]) else 50,
        "ofi": round(latest["ofi"], 4) if not pd.isna(latest["ofi"]) else 0.0,
        "ofi_trend": round(latest["ofi_trend"], 4) if not pd.isna(latest["ofi_trend"]) else 0.0,
    }
