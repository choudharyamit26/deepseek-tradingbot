"""Opening Range scanner.

Phase 1 (runs once at 09:30):
  - Fetch first 15-min candle (the OR) for every stock in the sector map.
  - Score each sector by the volume-weighted average % move of its stocks.
  - Three quality gates before a sector qualifies:
      1. Neutral zone — if signed avg_move < MIN_SECTOR_DIRECTION_PCT, the
         opening move is noise; skip the sector (prevents e.g. Banking -0.18%
         being called BEAR when multi-TF it is Bullish).
      2. 5-day trend alignment — OR direction must agree with the sector's
         5-day price trend. Prevents taking BUY in IT which gapped up today
         but is -1.57%/1W, -4.80%/1M in a downtrend.
      3. Stock-direction alignment — individual stocks whose own OR move
         opposes the sector direction are excluded from the watchlist
         (e.g. MANAPPURAM -0.96% inside a BULL FINANCIALS sector).
  - Pick top-N sectors, then top-M stocks per sector.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import pandas as pd

from momentum_bot import config as cfg
from momentum_bot.sector_map import SYMBOL_TO_SECTOR

logger = logging.getLogger(__name__)


@dataclass
class OpeningRange:
    symbol:          str
    sector:          str
    or_open:         float
    or_high:         float
    or_low:          float
    or_close:        float
    or_volume:       float    # total volume in OR window
    avg_vol_15:      float    # baseline: avg 15-min volume (daily avg / 25)
    pct_move:        float    # (or_close - or_open) / or_open * 100
    vol_ratio:       float    # or_volume / avg_vol_15
    score:           float    # abs(pct_move) * vol_ratio — momentum rank
    five_day_return: float    # (close_today - close_5d_ago) / close_5d_ago * 100


@dataclass
class SectorResult:
    sector:           str
    direction:        str       # "BULL" | "BEAR"
    score:            float     # weighted mean abs(pct_move) of stocks
    avg_move:         float     # signed mean pct_move (confirms direction)
    five_day_return:  float     # avg 5d return across sector stocks
    stocks:           list[OpeningRange] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────

def fetch_opening_range(symbol: str, sid: str, dhan) -> OpeningRange | None:
    """Fetch the first 15-min candle of the day and return an OpeningRange."""
    sector = SYMBOL_TO_SECTOR.get(symbol)
    if not sector:
        return None

    # -- 15-min bar: first bar of today is 09:15-09:30 ─────────────────────
    df15 = dhan.get_historical_data(sid, interval="15minute", min_bars=1)
    if df15 is None or df15.empty:
        logger.debug("%s  no 15-min data", symbol)
        return None

    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    if hasattr(df15.index, "date"):
        df15 = df15[df15.index.date == today]
    if df15.empty:
        logger.debug("%s  no today bars in 15-min frame", symbol)
        return None

    or_bar  = df15.iloc[0]
    or_open = float(or_bar["open"])
    or_high = float(or_bar["high"])
    or_low  = float(or_bar["low"])
    or_close= float(or_bar["close"])
    or_vol  = float(or_bar["volume"])
    if or_open <= 0:
        return None

    # -- Daily bars: volume baseline + 5-day trend ──────────────────────────
    avg_daily_vol, five_day_ret = _fetch_daily_context(symbol, sid, dhan)
    avg_vol_15  = (avg_daily_vol / 25.0) if avg_daily_vol > 0 else 1.0
    vol_ratio   = or_vol / avg_vol_15 if avg_vol_15 > 0 else 1.0
    pct_move    = (or_close - or_open) / or_open * 100.0
    score       = abs(pct_move) * vol_ratio

    return OpeningRange(
        symbol=symbol, sector=sector,
        or_open=or_open, or_high=or_high, or_low=or_low, or_close=or_close,
        or_volume=or_vol, avg_vol_15=avg_vol_15,
        pct_move=pct_move, vol_ratio=vol_ratio, score=score,
        five_day_return=five_day_ret,
    )


def _fetch_daily_context(symbol: str, sid: str, dhan) -> tuple[float, float]:
    """Return (avg_daily_volume_10d, five_day_return_pct).

    Reuses the single daily-bar fetch to compute both volume baseline and
    5-day trend — no extra API call.
    """
    try:
        df = dhan.get_historical_data(sid, interval="1day", min_bars=6)
        if df is not None and len(df) >= 6:
            avg_vol     = float(df["volume"].tail(10).mean())
            five_day    = (
                (float(df["close"].iloc[-1]) - float(df["close"].iloc[-6]))
                / float(df["close"].iloc[-6]) * 100.0
            )
            return avg_vol, five_day
        elif df is not None and len(df) >= 2:
            return float(df["volume"].tail(10).mean()), 0.0
    except Exception as e:
        logger.debug("%s  daily context fetch failed: %s", symbol, e)
    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────

def rank_sectors(all_or: list[OpeningRange]) -> list[SectorResult]:
    """Group by sector, apply quality gates, return qualifying sectors sorted."""
    by_sector: dict[str, list[OpeningRange]] = {}
    for r in all_or:
        by_sector.setdefault(r.sector, []).append(r)

    results: list[SectorResult] = []

    for sector, records in by_sector.items():
        if len(records) < cfg.MIN_SECTOR_STOCKS:
            continue

        avg_move = sum(r.pct_move for r in records) / len(records)

        # ── Gate 1: weighted score floor ─────────────────────────────────
        total_vr      = sum(r.vol_ratio for r in records) or 1.0
        weighted_score= sum(abs(r.pct_move) * r.vol_ratio for r in records) / total_vr
        if weighted_score < cfg.MIN_SECTOR_MOVE_PCT:
            logger.debug("SECTOR %-14s  score %.3f < MIN → skip", sector, weighted_score)
            continue

        # ── Gate 2: neutral zone — direction must be decisive ─────────────
        # Small OR moves (|avg_move| < threshold) are noise, not a signal.
        # Example: BANKING was -0.18% in the OR but is Bullish multi-TF.
        # We refuse to label it BEAR; we skip it instead.
        if abs(avg_move) < cfg.MIN_SECTOR_DIRECTION_PCT:
            logger.info(
                "SECTOR %-14s  neutral zone (avg_move=%+.2f%% inside ±%.2f%%) → skip",
                sector, avg_move, cfg.MIN_SECTOR_DIRECTION_PCT,
            )
            continue

        direction     = "BULL" if avg_move > 0 else "BEAR"
        sector_5d     = sum(r.five_day_return for r in records) / len(records)
        sector_5d_dir = "BULL" if sector_5d > 0 else "BEAR"

        # ── Gate 3: OR direction must agree with 5-day trend ─────────────
        # Prevents buying IT which gapped up today (+0.52% OR) but is in a
        # 1W -1.57%, 1M -4.80% downtrend (RSI 39).
        if cfg.REQUIRE_TREND_ALIGNMENT and direction != sector_5d_dir:
            logger.info(
                "SECTOR %-14s  trend mismatch: OR=%s(%.2f%%) vs 5d=%s(%.2f%%) → skip",
                sector, direction, avg_move, sector_5d_dir, sector_5d,
            )
            continue

        results.append(SectorResult(
            sector=sector, direction=direction,
            score=weighted_score, avg_move=avg_move,
            five_day_return=sector_5d,
            stocks=sorted(records, key=lambda r: r.score, reverse=True),
        ))

    return sorted(results, key=lambda s: s.score, reverse=True)


def select_watchlist(
    ranked_sectors: list[SectorResult],
) -> tuple[list[str], dict[str, OpeningRange], dict[str, str]]:
    """Return (watchlist_symbols, or_data_map, symbol_direction_map).

    Picks top-N sectors and top-M stocks per sector. Stocks whose own OR
    direction opposes the sector direction are silently skipped — this
    catches e.g. MANAPPURAM -0.96% inside a BULL FINANCIALS sector.
    """
    watchlist: list[str] = []
    or_map:    dict[str, OpeningRange] = {}
    dir_map:   dict[str, str]          = {}

    for sector_result in ranked_sectors[: cfg.TOP_N_SECTORS]:
        count = 0
        for orb in sector_result.stocks:
            if count >= cfg.TOP_STOCKS_PER_SECTOR:
                break

            # ── OR width floor ────────────────────────────────────────────
            or_width_pct = (orb.or_high - orb.or_low) / orb.or_close * 100.0
            if or_width_pct < cfg.MIN_OR_WIDTH_PCT:
                logger.info(
                    "SKIP %-12s  OR width %.2f%% < %.2f%% minimum",
                    orb.symbol, or_width_pct, cfg.MIN_OR_WIDTH_PCT,
                )
                continue

            # ── Stock-direction alignment ─────────────────────────────────
            # Individual stock's OR must move in the same direction as the
            # sector to confirm it is a genuine participant in the theme.
            stock_bull = orb.pct_move >= 0
            if sector_result.direction == "BULL" and not stock_bull:
                logger.info(
                    "SKIP %-12s  stock OR %+.2f%% opposes BULL sector — not a sector participant",
                    orb.symbol, orb.pct_move,
                )
                continue
            if sector_result.direction == "BEAR" and stock_bull:
                logger.info(
                    "SKIP %-12s  stock OR %+.2f%% opposes BEAR sector — not a sector participant",
                    orb.symbol, orb.pct_move,
                )
                continue

            entry_dir = "BUY" if sector_result.direction == "BULL" else "SELL"
            watchlist.append(orb.symbol)
            or_map[orb.symbol]  = orb
            dir_map[orb.symbol] = entry_dir
            count += 1

            logger.info(
                "WATCH  %-12s  sector=%-14s  dir=%s  OR=%.2f-%.2f  "
                "move=%+.2f%%  5d=%+.2f%%  vol=%.1fx",
                orb.symbol, sector_result.sector, entry_dir,
                orb.or_low, orb.or_high,
                orb.pct_move, orb.five_day_return, orb.vol_ratio,
            )

    return watchlist, or_map, dir_map
