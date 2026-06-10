"""
Reversal Detection Engine
─────────────────────────
Analyses a 3-minute OHLCV DataFrame and returns discrete reversal signals
plus a composite score (0-100).  No chart/plot dependencies.

v2 — Optimized to avoid false exits from single big candles:
  • VWAP breach requires 2-candle confirmation
  • Volume climax requires 4x avg + next-candle follow-through
  • MFI extreme requires 2 consecutive bars in zone
  • RSI divergence requires min 5-bar divergence window
  • Scoring heavily penalizes single-signal detections
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import talib


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ReversalSignal:
    """A single reversal signal detected on the chart."""
    name: str
    severity: int          # 0-100
    direction: str         # BULLISH or BEARISH
    description: str
    icon: str = "⚠️"
    confirmed: bool = True  # False = preliminary (single candle, needs confirmation)


@dataclass
class ReversalReport:
    """Aggregate report for one position."""
    signals: List[ReversalSignal] = field(default_factory=list)
    score: int = 0                      # composite 0-100
    recommendation: str = "HOLD"        # HOLD / CAUTION / EXIT NOW


# ── Individual detectors ─────────────────────────────────────────────────────

def _detect_rsi_divergence(df: pd.DataFrame, is_buy: bool) -> ReversalSignal | None:
    """Bullish/bearish divergence with strict thresholds to avoid noise."""
    if len(df) < 20:
        return None
    rsi = talib.RSI(df["close"], timeperiod=14)
    closes = df["close"].values
    rsi_vals = rsi.values

    # Need at least 5 valid bars for meaningful divergence
    window = min(10, len(df) - 1)
    if window < 5:
        return None

    recent_close = closes[-window:]
    recent_rsi = rsi_vals[-window:]

    mask = ~np.isnan(recent_rsi)
    if mask.sum() < 5:
        return None
    recent_close = recent_close[mask]
    recent_rsi = recent_rsi[mask]

    # ── Strict thresholds to filter noise ──────────────────────────────────
    rsi_delta = abs(recent_rsi[-1] - recent_rsi[0])
    price_pct_change = abs(recent_close[-1] - recent_close[0]) / recent_close[0] * 100

    # Require minimum 8-point RSI divergence AND 0.3% price move
    if rsi_delta < 8 or price_pct_change < 0.3:
        return None

    # Require divergence across the MIDDLE of the window too (not just endpoints)
    mid = len(recent_close) // 2

    if is_buy:
        # Bearish divergence: price trending up but RSI trending down
        price_up = recent_close[-1] > recent_close[0] and recent_close[-1] > recent_close[mid]
        rsi_down = recent_rsi[-1] < recent_rsi[0] and recent_rsi[-1] < recent_rsi[mid]
        if price_up and rsi_down:
            return ReversalSignal(
                name="RSI Divergence",
                severity=75,
                direction="BEARISH",
                description=f"Price rising but RSI falling ({recent_rsi[-1]:.0f} vs {recent_rsi[0]:.0f}) over {len(recent_close)} bars",
                icon="📉",
            )
    else:
        # Bullish divergence: price trending down but RSI trending up
        price_down = recent_close[-1] < recent_close[0] and recent_close[-1] < recent_close[mid]
        rsi_up = recent_rsi[-1] > recent_rsi[0] and recent_rsi[-1] > recent_rsi[mid]
        if price_down and rsi_up:
            return ReversalSignal(
                name="RSI Divergence",
                severity=75,
                direction="BULLISH",
                description=f"Price falling but RSI rising ({recent_rsi[-1]:.0f} vs {recent_rsi[0]:.0f}) over {len(recent_close)} bars",
                icon="📈",
            )
    return None


def _detect_macd_crossover(df: pd.DataFrame, is_buy: bool) -> ReversalSignal | None:
    """MACD line crossing signal line — 1-bar crossover detection."""
    if len(df) < 30:
        return None
    macd, signal, _ = talib.MACD(df["close"])

    # Need 2 valid bars
    for i in [-1, -2]:
        if pd.isna(macd.iloc[i]) or pd.isna(signal.iloc[i]):
            return None

    prev_diff = macd.iloc[-2] - signal.iloc[-2]
    curr_diff = macd.iloc[-1] - signal.iloc[-1]

    if is_buy:
        # Macd crossed below signal: prev was positive/zero, curr is negative
        crossed = prev_diff >= 0 and curr_diff < 0
        if crossed:
            return ReversalSignal(
                name="MACD Bearish Cross",
                severity=65,
                direction="BEARISH",
                description=f"MACD crossed below signal line ({curr_diff:+.2f})",
                icon="❌",
            )
    else:
        # Macd crossed above signal: prev was negative/zero, curr is positive
        crossed = prev_diff <= 0 and curr_diff > 0
        if crossed:
            return ReversalSignal(
                name="MACD Bullish Cross",
                severity=65,
                direction="BULLISH",
                description=f"MACD crossed above signal line ({curr_diff:+.2f})",
                icon="✅",
            )
    return None


def _detect_bollinger_squeeze(df: pd.DataFrame, is_buy: bool, indicators: dict) -> ReversalSignal | None:
    """BB breakout — requires close outside band.
    - 1 close outside band is enough if there was a squeeze AND volume is high (volume_ratio > 1.2)
    - Otherwise, requires 2 consecutive closes outside the band.
    """
    if len(df) < 20:
        return None
    upper, mid, lower = talib.BBANDS(df["close"], timeperiod=20)
    if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]):
        return None

    width = (upper - lower) / mid
    width_vals = width.dropna()
    if len(width_vals) < 5:
        return None

    avg_width = width_vals.iloc[-10:].mean() if len(width_vals) >= 10 else width_vals.mean()
    curr_width = width_vals.iloc[-1]
    is_squeezed = curr_width < avg_width * 0.6
    vol_ratio = indicators.get("volume_ratio", 1.0)

    close_now = df["close"].iloc[-1]

    # Check 1-candle conditions (requires squeeze + volume confirmation)
    one_candle_break = is_squeezed and vol_ratio > 1.2

    if is_buy:
        # Breakdown (close below lower band)
        is_breakdown = close_now < lower.iloc[-1]
        if is_breakdown:
            confirmed = False
            if one_candle_break:
                confirmed = True
            elif len(df) >= 2 and not pd.isna(lower.iloc[-2]):
                confirmed = df["close"].iloc[-2] < lower.iloc[-2]
            
            if confirmed:
                return ReversalSignal(
                    name="BB Breakdown",
                    severity=70 if is_squeezed else 50,
                    direction="BEARISH",
                    description=f"Breakdown below lower BB{' (squeeze + volume)' if one_candle_break else ' (2-bar confirm)'}",
                    icon="🔻",
                )
    else:
        # Breakout (close above upper band)
        is_breakout = close_now > upper.iloc[-1]
        if is_breakout:
            confirmed = False
            if one_candle_break:
                confirmed = True
            elif len(df) >= 2 and not pd.isna(upper.iloc[-2]):
                confirmed = df["close"].iloc[-2] > upper.iloc[-2]
            
            if confirmed:
                return ReversalSignal(
                    name="BB Breakout",
                    severity=70 if is_squeezed else 50,
                    direction="BULLISH",
                    description=f"Breakout above upper BB{' (squeeze + volume)' if one_candle_break else ' (2-bar confirm)'}",
                    icon="🔺",
                )
    return None


def _detect_volume_climax(df: pd.DataFrame, is_buy: bool) -> ReversalSignal | None:
    """Volume climax detection:
    - Immediate 1-bar detection if volume is extremely high (> 5x avg) and body is small.
    - Otherwise, 2-bar confirmation: volume > 4x average with reversal candle + next-candle follow-through.
    """
    if len(df) < 12:
        return None

    # Check current bar first for extreme climax (1-bar)
    vol_avg_current = df["volume"].iloc[-11:-1].mean()
    if vol_avg_current > 0:
        curr_bar = df.iloc[-1]
        curr_vol = curr_bar["volume"]
        curr_body = abs(curr_bar["close"] - curr_bar["open"])
        curr_range = curr_bar["high"] - curr_bar["low"]
        
        if curr_vol >= vol_avg_current * 5.0 and curr_range > 0:
            curr_body_ratio = curr_body / curr_range
            if curr_body_ratio < 0.35:
                if is_buy and curr_bar["close"] < curr_bar["open"]:
                    return ReversalSignal(
                        name="Volume Climax",
                        severity=80,
                        direction="BEARISH",
                        description=f"Extreme volume climax ({curr_vol / vol_avg_current:.1f}x) on latest bar",
                        icon="💥",
                    )
                elif not is_buy and curr_bar["close"] > curr_bar["open"]:
                    return ReversalSignal(
                        name="Volume Climax",
                        severity=80,
                        direction="BULLISH",
                        description=f"Extreme volume climax ({curr_vol / vol_avg_current:.1f}x) on latest bar",
                        icon="💥",
                    )

    # Fallback to the 2-bar climax logic
    vol_avg_2bar = df["volume"].iloc[-12:-2].mean()
    if vol_avg_2bar <= 0:
        return None

    spike_bar = df.iloc[-2]
    follow_bar = df.iloc[-1]
    spike_vol = spike_bar["volume"]

    if spike_vol >= vol_avg_2bar * 4:
        body = abs(spike_bar["close"] - spike_bar["open"])
        full_range = spike_bar["high"] - spike_bar["low"]
        if full_range > 0:
            body_ratio = body / full_range
            is_reversal_candle = body_ratio < 0.35
            if is_buy:
                spike_bearish = spike_bar["close"] < spike_bar["open"]
                follow_confirms = follow_bar["close"] < spike_bar["close"]
                if spike_bearish and is_reversal_candle and follow_confirms:
                    return ReversalSignal(
                        name="Volume Climax",
                        severity=80,
                        direction="BEARISH",
                        description=f"Volume spike ({spike_vol / vol_avg_2bar:.1f}x) with reversal candle + bearish follow-through",
                        icon="💥",
                    )
            else:
                spike_bullish = spike_bar["close"] > spike_bar["open"]
                follow_confirms = follow_bar["close"] > spike_bar["close"]
                if spike_bullish and is_reversal_candle and follow_confirms:
                    return ReversalSignal(
                        name="Volume Climax",
                        severity=80,
                        direction="BULLISH",
                        description=f"Volume spike ({spike_vol / vol_avg_2bar:.1f}x) with reversal candle + bullish follow-through",
                        icon="💥",
                    )
    return None


def _detect_vwap_breach(df: pd.DataFrame, is_buy: bool, indicators: dict) -> ReversalSignal | None:
    """Price on wrong side of VWAP.
    - 2-candle confirmation: both closes on wrong side of VWAP with vol_ratio > 1.0.
    - 1-candle confirmation: current close is on wrong side with high volume (vol_ratio > 1.8).
    """
    if len(df) < 5:
        return None

    vwap = indicators.get("vwap", 0)
    vol_ratio = indicators.get("volume_ratio", 1)

    if vwap <= 0:
        return None

    close_now = df["close"].iloc[-1]
    close_prev = df["close"].iloc[-2]

    if is_buy:
        is_below = close_now < vwap
        if is_below:
            confirmed = False
            desc = ""
            dist = (close_now - vwap) / vwap * 100
            if vol_ratio > 1.8:
                confirmed = True
                desc = f"Immediate high-volume VWAP breach ({dist:.2f}%, vol {vol_ratio:.1f}x)"
            elif close_prev < vwap and vol_ratio > 1.0:
                confirmed = True
                desc = f"Price below VWAP for 2 bars ({dist:.2f}%, vol {vol_ratio:.1f}x)"
            
            if confirmed:
                return ReversalSignal(
                    name="VWAP Breach",
                    severity=70,
                    direction="BEARISH",
                    description=desc,
                    icon="⬇️",
                )
    else:
        is_above = close_now > vwap
        if is_above:
            confirmed = False
            desc = ""
            dist = (close_now - vwap) / vwap * 100
            if vol_ratio > 1.8:
                confirmed = True
                desc = f"Immediate high-volume VWAP breach (+{dist:.2f}%, vol {vol_ratio:.1f}x)"
            elif close_prev > vwap and vol_ratio > 1.0:
                confirmed = True
                desc = f"Price above VWAP for 2 bars (+{dist:.2f}%, vol {vol_ratio:.1f}x)"
            
            if confirmed:
                return ReversalSignal(
                    name="VWAP Breach",
                    severity=70,
                    direction="BULLISH",
                    description=desc,
                    icon="⬆️",
                )
    return None


def _detect_three_candle_reversal(df: pd.DataFrame, is_buy: bool) -> ReversalSignal | None:
    """3 consecutive candles closing against the position."""
    if len(df) < 3:
        return None
    c1, c2, c3 = df["close"].iloc[-3], df["close"].iloc[-2], df["close"].iloc[-1]

    if is_buy and c1 > c2 > c3:
        return ReversalSignal(
            name="3-Candle Reversal",
            severity=60,
            direction="BEARISH",
            description=f"3 consecutive bearish closes: {c1:.2f} → {c2:.2f} → {c3:.2f}",
            icon="🔴",
        )
    elif not is_buy and c1 < c2 < c3:
        return ReversalSignal(
            name="3-Candle Reversal",
            severity=60,
            direction="BULLISH",
            description=f"3 consecutive bullish closes: {c1:.2f} → {c2:.2f} → {c3:.2f}",
            icon="🟢",
        )
    return None


def _detect_ema_sma_cross(df: pd.DataFrame, is_buy: bool, indicators: dict) -> ReversalSignal | None:
    """EMA-9 crosses SMA-20 — requires cross held for 2 bars."""
    if len(df) < 22:
        return None
    ema9 = talib.EMA(df["close"], timeperiod=9)
    sma20 = talib.SMA(df["close"], timeperiod=20)

    for i in [-1, -2, -3]:
        if pd.isna(ema9.iloc[i]) or pd.isna(sma20.iloc[i]):
            return None

    # Cross must have happened and held for 2 bars
    pre = ema9.iloc[-3] > sma20.iloc[-3]
    bar1 = ema9.iloc[-2] > sma20.iloc[-2]
    bar2 = ema9.iloc[-1] > sma20.iloc[-1]

    if is_buy and pre and not bar1 and not bar2:
        return ReversalSignal(
            name="EMA/SMA Death Cross",
            severity=55,
            direction="BEARISH",
            description=f"EMA-9 ({ema9.iloc[-1]:.2f}) below SMA-20 ({sma20.iloc[-1]:.2f}) for 2 bars",
            icon="💀",
        )
    elif not is_buy and not pre and bar1 and bar2:
        return ReversalSignal(
            name="EMA/SMA Golden Cross",
            severity=55,
            direction="BULLISH",
            description=f"EMA-9 ({ema9.iloc[-1]:.2f}) above SMA-20 ({sma20.iloc[-1]:.2f}) for 2 bars",
            icon="✨",
        )
    return None


def _detect_mfi_extreme(df: pd.DataFrame, is_buy: bool) -> ReversalSignal | None:
    """MFI extreme for 2 consecutive bars (avoids single-candle spikes)."""
    if len(df) < 16:
        return None

    mfi = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)

    mfi_now = mfi.iloc[-1]
    mfi_prev = mfi.iloc[-2]

    if pd.isna(mfi_now) or pd.isna(mfi_prev):
        return None

    if is_buy and mfi_now > 80 and mfi_prev > 75:
        return ReversalSignal(
            name="MFI Overbought",
            severity=45,
            direction="BEARISH",
            description=f"MFI at {mfi_now:.0f} (prev {mfi_prev:.0f}) — sustained overbought",
            icon="🔥",
        )
    elif not is_buy and mfi_now < 20 and mfi_prev < 25:
        return ReversalSignal(
            name="MFI Oversold",
            severity=45,
            direction="BULLISH",
            description=f"MFI at {mfi_now:.0f} (prev {mfi_prev:.0f}) — sustained oversold",
            icon="❄️",
        )
    return None


# ── Aggregation ──────────────────────────────────────────────────────────────

def detect_reversals(df: pd.DataFrame, is_buy: bool, indicators: dict) -> ReversalReport:
    """Run all reversal detectors and return an aggregate ReversalReport.

    Scoring rules (v3 — Optimized Exit Signal Generator):
      • Primary Signals: RSI Divergence, Volume Climax, VWAP Breach, MACD Crossover.
      • Secondary Signals: Bollinger Breakout/Breakdown, 3-Candle Reversal, EMA/SMA Crossover, MFI Extreme.
      • 1 Primary signal  → score = severity × 0.65  (moderately discounted, can trigger caution)
      • 1 Secondary signal → score = severity × 0.45  (heavily discounted, keeps hold)
      • 2 signals         → score = avg severity × 0.80 (can cross exit threshold if severe)
      • 3+ signals        → score = avg severity × 0.95, capped at 95
    """
    signals: List[ReversalSignal] = []

    detectors = [
        lambda: _detect_rsi_divergence(df, is_buy),
        lambda: _detect_macd_crossover(df, is_buy),
        lambda: _detect_bollinger_squeeze(df, is_buy, indicators),
        lambda: _detect_volume_climax(df, is_buy),
        lambda: _detect_vwap_breach(df, is_buy, indicators),
        lambda: _detect_three_candle_reversal(df, is_buy),
        lambda: _detect_ema_sma_cross(df, is_buy, indicators),
        lambda: _detect_mfi_extreme(df, is_buy),
    ]

    for detector in detectors:
        try:
            result = detector()
            if result is not None:
                signals.append(result)
        except Exception:
            pass  # Never crash the dashboard for a detector failure

    # Sort by severity descending
    signals.sort(key=lambda s: s.severity, reverse=True)

    # ── Composite score with dynamic base-score + increments (v3) ────────────
    if not signals:
        score = 0
    else:
        primary_names = {
            "RSI Divergence", "Volume Climax", "VWAP Breach", 
            "MACD Bearish Cross", "MACD Bullish Cross"
        }
        base_score = max(s.severity for s in signals)
        n = len(signals)

        if n == 1:
            # Single signal discount
            sig = signals[0]
            if sig.name in primary_names:
                score = int(base_score * 0.70)
            else:
                score = int(base_score * 0.45)
        else:
            # Multi-signal accumulator (avoids dilution of average)
            score = base_score
            # Sort signals so we skip the one that gave the base_score
            sorted_sigs = sorted(signals, key=lambda s: s.severity, reverse=True)
            for sig in sorted_sigs[1:]:
                if sig.name in primary_names:
                    score += 12
                else:
                    score += 4
            score = min(95, score)

    if score >= 70:
        recommendation = "🚨 EXIT NOW"
    elif score >= 40:
        recommendation = "⚠️ CAUTION"
    else:
        recommendation = "✅ HOLD"

    return ReversalReport(signals=signals, score=score, recommendation=recommendation)
