"""Opening Range Breakout (ORB) signal logic.

Checks a single stock on each 3-min scan tick and returns a signal dict
(or None). All AI / Kronos / regime logic is intentionally absent — this
is a pure price-action / volume momentum strategy.

Entry rules
-----------
BUY  : latest 3-min bar CLOSES above OR high + buffer
        AND 3-min volume > ENTRY_VOLUME_MULTIPLIER × rolling-bar-avg
        AND RSI < 78 (not already overbought)
        AND not already in a position for this symbol

SELL : latest 3-min bar CLOSES below OR low − buffer
        AND 3-min volume > ENTRY_VOLUME_MULTIPLIER × rolling-bar-avg
        AND RSI > 22 (not already oversold)
        AND not already in a position for this symbol

Stop / target
-------------
BUY  : stop = OR low  (clamped by MAX/MIN_STOP_PCT)
        target = entry + RR_RATIO × (entry − stop)

SELL : stop = OR high (clamped by MAX/MIN_STOP_PCT)
        target = entry − RR_RATIO × (stop − entry)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import pandas as pd
import talib

from momentum_bot import config as cfg
from momentum_bot.scanner import OpeningRange

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol:           str
    direction:        str          # "BUY" | "SELL"
    entry_price:      float        # market price at signal time
    stop_price:       float
    target_price:     float
    stop_pct:         float        # stop distance as % of entry
    target_pct:       float
    or_high:          float
    or_low:           float
    volume_ratio:     float        # entry bar vol / avg bar vol
    rsi:              float
    reasoning:        str


def check_entry(
    symbol: str,
    sid: str,
    orb: OpeningRange,
    direction: str,          # "BUY" | "SELL" — from sector scanner
    dhan,
    already_traded: set[str],
) -> Signal | None:
    """Return a Signal if the ORB entry condition is met, else None."""
    if symbol in already_traded:
        return None

    df = dhan.get_historical_data(sid, interval="3minute", min_bars=25)
    if df is None or len(df) < 10:
        logger.debug("%s  insufficient 3-min bars (%d)", symbol, len(df) if df is not None else 0)
        return None

    # Keep only today's bars
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    if hasattr(df.index, "date"):
        df = df[df.index.date == today]
    if len(df) < 3:
        return None

    latest = df.iloc[-1]
    close  = float(latest["close"])
    volume = float(latest["volume"])

    # -- RSI ----------------------------------------------------------------
    rsi_series = talib.RSI(df["close"].values.astype(float), timeperiod=14)
    rsi = float(rsi_series[-1]) if not pd.isna(rsi_series[-1]) else 50.0

    # -- Volume ratio: this bar vs rolling 20-bar mean ----------------------
    vol_mean = float(df["volume"].rolling(20, min_periods=5).mean().iloc[-1])
    vol_ratio = volume / vol_mean if vol_mean > 0 else 1.0

    # -- Entry thresholds ---------------------------------------------------
    buy_trigger  = orb.or_high * (1 + cfg.BREAKOUT_BUFFER_PCT / 100.0)
    sell_trigger = orb.or_low  * (1 - cfg.BREAKOUT_BUFFER_PCT / 100.0)

    # -- Check direction ----------------------------------------------------
    if direction == "BUY":
        if close < buy_trigger:
            return None
        if rsi > 78:
            logger.info("%s  BUY skipped — RSI %.1f overbought", symbol, rsi)
            return None
        if vol_ratio < cfg.ENTRY_VOLUME_MULTIPLIER:
            logger.info("%s  BUY skipped — vol %.1fx below %.1fx", symbol, vol_ratio, cfg.ENTRY_VOLUME_MULTIPLIER)
            return None

        entry  = close
        raw_stop   = orb.or_low
        stop_dist  = entry - raw_stop
        stop_pct   = stop_dist / entry * 100.0

        # clamp stop distance
        stop_pct   = max(cfg.MIN_STOP_PCT, min(cfg.MAX_STOP_PCT, stop_pct))
        stop_price = entry * (1 - stop_pct / 100.0)
        tgt_price  = entry + cfg.RR_RATIO * (entry - stop_price)
        tgt_pct    = (tgt_price - entry) / entry * 100.0

        reason = (
            f"ORB BUY: close {close:.2f} > OR high {orb.or_high:.2f}+buf | "
            f"vol {vol_ratio:.1f}x | RSI {rsi:.1f}"
        )

    else:  # SELL
        if close > sell_trigger:
            return None
        if rsi < 22:
            logger.info("%s  SELL skipped — RSI %.1f oversold", symbol, rsi)
            return None
        if vol_ratio < cfg.ENTRY_VOLUME_MULTIPLIER:
            logger.info("%s  SELL skipped — vol %.1fx below %.1fx", symbol, vol_ratio, cfg.ENTRY_VOLUME_MULTIPLIER)
            return None

        entry  = close
        raw_stop   = orb.or_high
        stop_dist  = raw_stop - entry
        stop_pct   = stop_dist / entry * 100.0

        stop_pct   = max(cfg.MIN_STOP_PCT, min(cfg.MAX_STOP_PCT, stop_pct))
        stop_price = entry * (1 + stop_pct / 100.0)
        tgt_price  = entry - cfg.RR_RATIO * (stop_price - entry)
        tgt_pct    = (entry - tgt_price) / entry * 100.0

        reason = (
            f"ORB SELL: close {close:.2f} < OR low {orb.or_low:.2f}-buf | "
            f"vol {vol_ratio:.1f}x | RSI {rsi:.1f}"
        )

    logger.info(
        "SIGNAL  %-12s  %s  entry=%.2f  stop=%.2f(%.2f%%)  "
        "target=%.2f(%.2f%%)  vol=%.1fx  RSI=%.1f",
        symbol, direction, entry,
        stop_price, stop_pct, tgt_price, tgt_pct,
        vol_ratio, rsi,
    )

    return Signal(
        symbol=symbol, direction=direction,
        entry_price=entry, stop_price=stop_price, target_price=tgt_price,
        stop_pct=stop_pct, target_pct=tgt_pct,
        or_high=orb.or_high, or_low=orb.or_low,
        volume_ratio=vol_ratio, rsi=rsi, reasoning=reason,
    )
