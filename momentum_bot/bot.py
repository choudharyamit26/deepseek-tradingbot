"""Main async loop for the Opening Range Breakout momentum bot.

Timeline each trading day:
  09:15          Market opens. Bot sleeps until OR_END_TIME.
  09:30          Opening-range scan fires: fetch 15-min bars for all watched
                 symbols, score sectors, build a focused watchlist (4-6 stocks).
  09:30 – 14:00  Trading loop: every 3 minutes check each watchlist stock for
                 an ORB breakout entry. Execute qualifying signals.
  14:00          Entry window closes. Hold existing positions.
  14:45          Force-exit all open positions. Day done.

The bot is completely self-contained. It imports DhanStockTradingBot directly
so it has no shared state with the main enhanced_bot process.
"""

from __future__ import annotations
import asyncio
import logging
import sys
from datetime import datetime, time as dtime

import pandas as pd

from momentum_bot import config as cfg
from momentum_bot.scanner import (
    fetch_opening_range, rank_sectors, select_watchlist, OpeningRange,
)
from momentum_bot.signals import check_entry
from momentum_bot.executor import MomentumExecutor

logger = logging.getLogger(__name__)

# ── Timing helpers ────────────────────────────────────────────────────────────

def _now_ist() -> datetime:
    return datetime.now(tz=None)   # server must be in IST; or add pytz if needed


def _time_now() -> dtime:
    return _now_ist().time().replace(second=0, microsecond=0)


def _parse_time(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(h, m)


def _seconds_until(target: dtime) -> float:
    now = _now_ist()
    t   = now.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    delta = (t - now).total_seconds()
    return max(delta, 0.0)


# ── Bot ───────────────────────────────────────────────────────────────────────

class MomentumBot:
    def __init__(self, dhan, dry_run: bool = False):
        """
        Parameters
        ----------
        dhan     : DhanStockTradingBot instance (caller creates it)
        dry_run  : if True, signals are computed but no orders are placed
        """
        self._dhan     = dhan
        self._executor = MomentumExecutor(dhan, dry_run=dry_run)
        self._dry_run  = dry_run

        # Populated after the opening scan
        self._watchlist:  list[str]                = []
        self._or_map:     dict[str, OpeningRange]  = {}
        self._dir_map:    dict[str, str]            = {}   # symbol → "BUY"|"SELL"
        self._sid_map:    dict[str, str]            = {}   # symbol → Dhan security_id

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        mode = "DRY-RUN" if self._dry_run else "LIVE"
        logger.info("=" * 60)
        logger.info("MomentumBot starting  [%s]", mode)
        logger.info("=" * 60)

        or_end   = _parse_time(cfg.OR_END_TIME)
        entry_end = _parse_time(cfg.ENTRY_END_TIME)
        exit_all  = _parse_time(cfg.EXIT_ALL_TIME)

        tnow = _time_now()

        # ── Wait for opening range to seal ────────────────────────────────────
        if tnow < or_end:
            wait = _seconds_until(or_end)
            logger.info("Waiting %.0fs for OR to seal at %s …", wait, cfg.OR_END_TIME)
            await asyncio.sleep(wait)
        else:
            logger.info("Already past OR_END_TIME (%s) — scanning immediately", cfg.OR_END_TIME)

        # ── Opening range scan ────────────────────────────────────────────────
        await self._opening_scan()

        if not self._watchlist:
            logger.warning("No qualifying stocks found in opening scan — exiting for today")
            return

        # ── Trading loop ──────────────────────────────────────────────────────
        logger.info(
            "Trading loop starts. Watchlist: %s",
            ", ".join(self._watchlist),
        )

        while True:
            tnow = _time_now()

            if tnow >= _parse_time(cfg.EXIT_ALL_TIME):
                break

            # Per-position time exit: close any trade held > TIME_EXIT_MINUTES,
            # regardless of how far away EXIT_ALL_TIME is.
            self._check_position_timeouts()

            if tnow < _parse_time(cfg.ENTRY_END_TIME):
                await self._scan_entries()

            await asyncio.sleep(180)   # 3-minute tick

        # ── Time exit ─────────────────────────────────────────────────────────
        # Distinguish positions already closed by SL/target (server-side) from
        # those still open (neither triggered) — the latter get a TIME-EXIT
        # market close, Telegram alert, and a separate log row.
        logger.info("EXIT_ALL_TIME reached — running time-exit check")
        self._executor.check_and_time_exit()
        logger.info("MomentumBot finished for today")

    # ── Opening range scan ────────────────────────────────────────────────────

    async def _opening_scan(self) -> None:
        logger.info("--- Opening Range Scan ---")
        security_ids = self._dhan.security_ids   # dict: symbol → sid

        all_or: list[OpeningRange] = []

        for symbol, sid in security_ids.items():
            try:
                orb = fetch_opening_range(symbol, str(sid), self._dhan)
                if orb is not None:
                    all_or.append(orb)
                    self._sid_map[symbol] = str(sid)
            except Exception as exc:
                logger.debug("%s  OR fetch error: %s", symbol, exc)
            await asyncio.sleep(0.1)   # gentle rate limiting

        logger.info("Opening ranges fetched: %d / %d stocks", len(all_or), len(security_ids))

        if not all_or:
            logger.error("Zero opening ranges — data issue, aborting scan")
            return

        ranked = rank_sectors(all_or)
        if not ranked:
            logger.warning("No sector cleared the minimum move threshold (%.2f%%)", cfg.MIN_SECTOR_MOVE_PCT)
            return

        logger.info("Sector ranking (top 5):")
        for r in ranked[:5]:
            logger.info(
                "  %-14s  dir=%-4s  score=%.3f  avg_move=%+.2f%%  stocks=%d",
                r.sector, r.direction, r.score, r.avg_move, len(r.stocks),
            )

        self._watchlist, self._or_map, self._dir_map = select_watchlist(ranked)
        logger.info(
            "Watchlist (%d stocks): %s",
            len(self._watchlist), ", ".join(self._watchlist),
        )

    # ── Per-tick entry scan ───────────────────────────────────────────────────

    async def _scan_entries(self) -> None:
        if self._executor.open_count >= cfg.MAX_OPEN_POSITIONS:
            logger.debug("Max positions open (%d) — skipping scan tick", cfg.MAX_OPEN_POSITIONS)
            return

        for symbol in self._watchlist:
            if symbol in self._executor.traded_symbols:
                continue
            sid = self._sid_map.get(symbol)
            if not sid:
                continue
            orb = self._or_map.get(symbol)
            if not orb:
                continue
            direction = self._dir_map.get(symbol, "BUY")

            try:
                signal = check_entry(
                    symbol=symbol, sid=sid, orb=orb,
                    direction=direction, dhan=self._dhan,
                    already_traded=self._executor.traded_symbols,
                )
                if signal:
                    self._executor.execute(signal, sid)
            except Exception as exc:
                logger.error("%s  signal check error: %s", symbol, exc)

            await asyncio.sleep(0.2)   # gentle rate limiting between stocks

    # ── Per-position time exit ────────────────────────────────────────────────

    def _check_position_timeouts(self) -> None:
        """Exit any position held longer than TIME_EXIT_MINUTES (mid-session)."""
        stale = self._executor.timed_out_positions()
        for pos in stale:
            logger.info(
                "Per-position time exit firing: %s held > %d min",
                pos.symbol, cfg.TIME_EXIT_MINUTES,
            )
            self._executor.time_exit_one(pos)
