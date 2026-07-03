import sys
import asyncio
import logging
import time as time_module
from pathlib import Path
from datetime import time as dtime, datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parent.parent))
from position_exit_guardian import PositionExitGuardian

from . import config as cfg
from .kronos_integration import KronosIntegration

logger = logging.getLogger(__name__)

_MARKET_OPEN_T = dtime(9, 15)
_MARKET_CLOSE_T = dtime(15, 30)
IST = ZoneInfo("Asia/Kolkata")


class KronosExitGuardian(PositionExitGuardian):
    def __init__(self, dhan_bot, bot_instance, kronos: KronosIntegration,
                 send_telegram=None, format_signal_msg=None,
                 enable_telegram=False, dry_run=True):
        super().__init__(dhan_bot, bot_instance, send_telegram,
                         format_signal_msg, enable_telegram, dry_run)
        self.kronos = kronos

    async def check_position(self, symbol, trade, current_price, pnl_pct):
        # ── Two-phase exit: partial + breakeven at threshold, then runner trail ──
        # Runs before everything else so a threshold crossing is never missed;
        # it only books partials and tightens stops — never closes the position.
        if cfg.TWO_PHASE_EXIT_ENABLED:
            await self._two_phase_tick(symbol, trade, current_price, pnl_pct)

        await super().check_position(symbol, trade, current_price, pnl_pct)
        if symbol not in self.bot_instance.active_trades:
            return  # base guardian closed it

        # ── Trailing SL activation ──────────────────────────────────────────
        # Only activate trailing when profit reaches activation_pct
        activation_pct = trade.get("trail_activation_pct", 3.0)
        if activation_pct > 0 and trade.get("trailing_sl", 0) == 0:
            if pnl_pct >= activation_pct:
                atr = trade.get("atr_value", 1)
                trail_dist = trade.get("trail_distance_atr", cfg.TRAILING_SL_DISTANCE_ATR)
                is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY
                if is_buy:
                    new_sl = current_price - trail_dist * atr
                    if new_sl > trade.get("entry_price", 0):
                        self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
                        self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
                        logger.info("%s trailing SL ACTIVATED at %.2f (pnl=%.2f%%, dist=%.1f*ATR)",
                                    symbol, new_sl, pnl_pct, trail_dist)
                else:
                    new_sl = current_price + trail_dist * atr
                    old_sl = trade.get("sl_price", 0)
                    if old_sl == 0 or new_sl < old_sl:
                        self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
                        self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
                        logger.info("%s trailing SL ACTIVATED at %.2f (pnl=%.2f%%, dist=%.1f*ATR)",
                                    symbol, new_sl, pnl_pct, trail_dist)

        # ── Time-based exit ─────────────────────────────────────────────────
        # Phase-2 runners are exempt: half is banked and the stop is at
        # breakeven or better, so time carries no risk — only upside.
        max_duration = 0 if trade.get("tp2_active") else cfg.MAX_TRADE_DURATION_MINUTES
        if max_duration > 0:
            entry_time = trade.get("entry_time")
            if entry_time:
                now = self._now_ist()
                if isinstance(entry_time, datetime):
                    elapsed = (now - entry_time).total_seconds() / 60
                else:
                    elapsed = 0
                if elapsed >= max_duration:
                    logger.info("%s time-based exit: held %.0f min (max %d min)",
                                symbol, elapsed, max_duration)
                    await self._intraday_exit(symbol, trade, current_price, f"TIME-EXIT {int(elapsed)}min")
                    return

        # ── Market close exit ──────────────────────────────────────────────
        close_exit_min = cfg.MARKET_CLOSE_EXIT_MINUTES
        if close_exit_min > 0:
            now = self._now_ist()
            close_cutoff = now.replace(hour=15, minute=30 - close_exit_min, second=0)
            if now.time() >= close_cutoff.time():
                logger.info("%s market close exit: time=%s cutoff=%02d:%02d",
                            symbol, now.time(), close_cutoff.hour, close_cutoff.minute)
                await self._intraday_exit(symbol, trade, current_price, f"CLOSE-EXIT {close_exit_min}min-before")
                return

        if not cfg.KRONOS_ENABLED or not self.kronos.ready:
            return

        security_id = self.dhan.security_ids.get(symbol)
        if not security_id:
            return

        try:
            async with self.bot_instance._dhan_sem:
                hist_3m = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "3minute", 20)
            if len(hist_3m) < 20:
                return

            hist_15m = hist_3m.resample("15min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

            if len(hist_15m) < 5:
                return

            pred_df = await asyncio.to_thread(self.kronos.predict, hist_15m, symbol=symbol)
            if pred_df is None:
                return

            entry_price = trade.get("entry_price", 0)
            side = "BUY" if trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY else "SELL"
            exit_signal = self.kronos.get_exit_signal(pred_df, entry_price, side)

            if exit_signal["exit"]:
                urgency = exit_signal.get("urgency", 50)
                pred_return = exit_signal.get("pred_return", 0) * 100
                logger.info(
                    "%s Kronos exit: urgency=%d pred_return=%.2f%% pnl=%.2f%%",
                    symbol, urgency, pred_return, pnl_pct,
                )

                wants_full_exit = (
                    urgency >= self._get_base_exit(symbol, trade, pnl_pct)
                    or (urgency >= 40 and pnl_pct < -1.5)
                )

                # ── Let winners run ─────────────────────────────────────────
                # A position in real profit ("runner") with only a MODEST Kronos
                # reversal is NOT harvested. We lock a profit-protecting trailing
                # stop and let it ride toward target. KRONOS-EXIT keeps its edge
                # on flat/losing trades (below the profit floor) and still
                # full-exits a runner on a STRONG reversal (urgency >= hard-exit).
                if (wants_full_exit and cfg.KRONOS_LET_WINNERS_RUN
                        and pnl_pct >= cfg.KRONOS_RUN_PROFIT_PCT
                        and urgency < cfg.KRONOS_HARD_EXIT_URGENCY):
                    locked = self._lock_runner_trail(symbol, trade, current_price)
                    logger.info(
                        "%s LET-RUN: winner (pnl=%.2f%%) held on modest reversal "
                        "(u=%d < hard=%d); trail locked at %.2f",
                        symbol, pnl_pct, urgency, cfg.KRONOS_HARD_EXIT_URGENCY, locked,
                    )
                    return

                if wants_full_exit:
                    await self._kronos_exit(symbol, trade, current_price, f"Kronos u={urgency} r={pred_return:+.2f}%")
                    return

                if urgency >= 40:
                    atr = trade.get("atr_value", 0)
                    if atr > 0:
                        old_sl = trade.get("trailing_sl", 0)
                        is_buy = side == "BUY"
                        if is_buy:
                            new_sl = pred_df["low"].min() - 0.3 * atr
                            if new_sl > old_sl:
                                self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
                                self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
                                logger.info("%s Kronos tightened SL: %.2f -> %.2f", symbol, old_sl, new_sl)
                        else:
                            new_sl = pred_df["high"].max() + 0.3 * atr
                            if old_sl == 0 or new_sl < old_sl:
                                self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
                                self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
                                logger.info("%s Kronos tightened SL: %.2f -> %.2f", symbol, old_sl, new_sl)

        except Exception as e:
            logger.warning("%s Kronos guardian error: %s", symbol, e)

    async def _intraday_exit(self, symbol, trade, current_price, reason):
        quantity = trade.get("quantity", 0)
        is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY

        if self.dry_run:
            logger.info("%s DRY-RUN %s qty=%d reason=%s", symbol, reason, quantity, reason)

        # _exit_position is the SOLE order-placement site — it sends the single
        # closing order. Do NOT also call reduce_position here or every exit
        # fires twice (double-close into an opposite position).
        closed = await self.bot_instance._exit_position(symbol, reason)
        if not closed:
            # Order rejected (or nothing to close): don't fake PnL / consec-loss.
            return

        pnl = pnl_pct = None
        if trade.get("entry_price", 0) > 0:
            ltp = current_price
            if ltp > 0:
                if is_buy:
                    pnl = (ltp - trade["entry_price"]) * quantity
                    pnl_pct = (ltp - trade["entry_price"]) / trade["entry_price"] * 100
                else:
                    pnl = (trade["entry_price"] - ltp) * quantity
                    pnl_pct = (trade["entry_price"] - ltp) / trade["entry_price"] * 100

        # Track consecutive losses
        bot = self.bot_instance
        if pnl is not None and pnl < 0:
            bot._consecutive_losses = getattr(bot, "_consecutive_losses", 0) + 1
            max_consec = cfg.MAX_CONSECUTIVE_LOSSES
            if max_consec > 0 and bot._consecutive_losses >= max_consec:
                logger.warning("CONSECUTIVE LOSS BREAKER: %d consecutive losses (max %d) — stopping new entries",
                               bot._consecutive_losses, max_consec)
        elif pnl is not None and pnl > 0:
            bot._consecutive_losses = 0

        self._notify_telegram(symbol, reason, "EXIT", quantity, trade.get("entry_price", 0),
                              pnl=pnl, pnl_pct=pnl_pct, reason=reason)

    def _lock_runner_trail(self, symbol, trade, current_price):
        """Lock a profit-protecting trailing stop on a winning position instead
        of market-exiting it. Trails KRONOS_RUN_TRAIL_ATR behind price but never
        looser than breakeven, so a runner can trail out with profit (or at worst
        breakeven-minus-costs) while keeping upside open toward target.

        Returns the SL level now in force (for logging). The bot's monitor loop
        (stock_trading_bot._monitor_one_position) enforces it as a TRAILING-SL.
        """
        atr = trade.get("atr_value", 0) or 0
        entry = trade.get("entry_price", 0)
        trail = cfg.KRONOS_RUN_TRAIL_ATR * atr
        is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY
        old_sl = trade.get("sl_price", 0) or 0

        if is_buy:
            new_sl = current_price - trail
            new_sl = max(new_sl, entry)              # never worse than breakeven
            if new_sl >= current_price:              # degenerate (tiny/zero ATR)
                new_sl = entry
            tighten = old_sl == 0 or new_sl > old_sl  # long: raise the floor
        else:
            new_sl = current_price + trail
            new_sl = min(new_sl, entry)              # never worse than breakeven
            if new_sl <= current_price:
                new_sl = entry
            tighten = old_sl == 0 or new_sl < old_sl  # short: lower the ceiling

        if tighten and new_sl > 0:
            self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
            self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
            return new_sl
        return old_sl

    async def _two_phase_tick(self, symbol, trade, current_price, pnl_pct):
        """Two-phase exit state machine, one step per guardian poll.

        Phase 1 (pnl below TWO_PHASE_PARTIAL_AT_PCT): do nothing — the existing
        loss-cutting stack (hard SL, Kronos exits, reversal exits) owns the trade.

        Phase flip (pnl crosses the threshold): book TWO_PHASE_PARTIAL_FRACTION
        of the position at market, then arm the runner with a breakeven-floored
        percent trail. A rejected partial order leaves the phase unflipped so it
        retries next poll. Positions too small to split (qty 1) skip the partial
        but still get the breakeven lock + trail.

        Phase 2: ratchet the trail behind the high-water mark. Enforcement is
        the monitor loop's existing TRAILING-SL check; this only sets levels.
        """
        entry = trade.get("entry_price", 0) or 0
        if entry <= 0 or current_price <= 0:
            return

        if not trade.get("tp2_active"):
            if pnl_pct < cfg.TWO_PHASE_PARTIAL_AT_PCT:
                return
            qty = trade.get("quantity", 0)
            part = int(qty * cfg.TWO_PHASE_PARTIAL_FRACTION)
            if 1 <= part < qty and not trade.get("tp2_partial_blocked"):
                booked = await self._book_partial(symbol, trade, current_price, part)
                if not booked:
                    return  # order rejected / legs pending — retry the flip next poll
            trade["tp2_active"] = True
            trade["tp2_highwater"] = current_price
            trade.setdefault("original_quantity", qty)
            logger.info("%s TWO-PHASE: phase 2 armed at %.2f (pnl=%.2f%%, runner qty=%d)",
                        symbol, current_price, pnl_pct, trade.get("quantity", 0))
        else:
            is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY
            hw = trade.get("tp2_highwater", current_price)
            trade["tp2_highwater"] = max(hw, current_price) if is_buy else min(hw, current_price)

        self._tp2_update_trail(symbol, trade)

    def _tp2_update_trail(self, symbol, trade):
        """Set the runner stop to high-water ∓ TWO_PHASE_RUNNER_TRAIL_PCT,
        never worse than breakeven, tighten-only (coexists with the ATR-based
        locks, which are also tighten-only — the tighter stop wins)."""
        entry = trade.get("entry_price", 0)
        hw = trade.get("tp2_highwater", 0)
        if entry <= 0 or hw <= 0:
            return
        is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY
        trail = cfg.TWO_PHASE_RUNNER_TRAIL_PCT / 100.0
        old_sl = trade.get("trailing_sl", 0) or 0
        if is_buy:
            new_sl = max(entry, hw * (1 - trail))
            tighten = old_sl == 0 or new_sl > old_sl
        else:
            new_sl = min(entry, hw * (1 + trail))
            tighten = old_sl == 0 or new_sl < old_sl
        if tighten:
            trade["trailing_sl"] = new_sl
            trade["sl_price"] = new_sl
            logger.info("%s TWO-PHASE: runner trail -> %.2f (hw=%.2f)", symbol, new_sl, hw)

    async def _clear_super_legs(self, symbol, trade):
        """Cancel a super order's pending TARGET/STOP_LOSS legs before a partial.

        The legs were placed for the FULL entry quantity; reducing the position
        while they stay working means a later leg trigger over-exits into an
        opposite position. Per-leg progress is tracked on the trade so a retry
        only re-attempts the leg that failed. Returns True when no legs remain
        (also for non-super orders, which have nothing to clear)."""
        order_id = trade.get("order_id", "")
        if not order_id or order_id.startswith(("DRY-", "DHAN-")):
            return True  # plain order / adopted position — no broker legs
        if trade.get("super_legs_cancelled"):
            return True
        done = trade.setdefault("tp2_legs_done", [])
        for leg in ("TARGET_LEG", "STOP_LOSS_LEG"):
            if leg in done:
                continue
            try:
                async with self.bot_instance._dhan_sem:
                    resp = await asyncio.to_thread(
                        self.dhan.dhan.cancel_super_order, order_id, leg)
                if isinstance(resp, dict) and resp.get("status") == "failure":
                    logger.error("%s TWO-PHASE: cancel %s failed: %s",
                                 symbol, leg, resp.get("remarks"))
                    return False
                done.append(leg)
            except Exception as exc:
                logger.error("%s TWO-PHASE: cancel %s raised: %s", symbol, leg, exc)
                return False
        trade["super_legs_cancelled"] = True  # _exit_position skips its re-cancel
        logger.info("%s TWO-PHASE: super order legs cleared (%s)", symbol, order_id)
        return True

    async def _book_partial(self, symbol, trade, current_price, part_qty):
        """Bank part_qty at market. Accumulates the realized PnL on the trade
        (folded into the final exit row by _exit_position) instead of calling
        log_exit, which would consume the trade's single CSV row and orphan the
        runner's exit. Returns True when the reduction is in effect.

        Super orders: broker-side legs are cleared first (see _clear_super_legs).
        While legs can't be cleared, the partial is aborted and retried — the
        position keeps its full broker-side protection meanwhile. After
        3 failed polls the partial is abandoned for this trade
        (tp2_partial_blocked) and phase 2 arms without it: breakeven floor and
        runner trail still apply, and with no reduction placed the intact legs
        cannot over-exit. Once legs are cleared, the runner is protected by the
        software trail only (the same model every other post-modification exit
        path in this codebase uses)."""
        is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY
        exit_trans = self.dhan.dhan.SELL if is_buy else self.dhan.dhan.BUY
        security_id = trade.get("security_id") or self.dhan.security_ids.get(symbol)

        if not self.dry_run:
            if not security_id:
                return False
            if not await self._clear_super_legs(symbol, trade):
                fails = trade.get("tp2_leg_cancel_fails", 0) + 1
                trade["tp2_leg_cancel_fails"] = fails
                if fails >= 3:
                    trade["tp2_partial_blocked"] = True
                    logger.error(
                        "%s TWO-PHASE: legs uncancellable after %d polls — "
                        "arming phase 2 WITHOUT partial (broker legs left intact)",
                        symbol, fails)
                return False
            async with self.bot_instance._dhan_sem:
                order = await asyncio.to_thread(
                    self.dhan.reduce_position, security_id, exit_trans, part_qty)
            ok = isinstance(order, dict) and order.get("status") == "success"
            if not ok:
                remarks = order.get("remarks") if isinstance(order, dict) else order
                logger.error("%s TWO-PHASE partial REJECTED (qty=%d): %s — will retry",
                             symbol, part_qty, remarks)
                return False

        entry = trade["entry_price"]
        pnl_part = (current_price - entry) * part_qty if is_buy else (entry - current_price) * part_qty
        trade.setdefault("original_quantity", trade.get("quantity", 0))
        trade["quantity"] = trade.get("quantity", 0) - part_qty
        trade["realized_partial_pnl"] = trade.get("realized_partial_pnl", 0.0) + pnl_part
        logger.info("%s TWO-PHASE: banked %d @ %.2f (pnl=%+.2f), runner qty=%d%s",
                    symbol, part_qty, current_price, pnl_part, trade["quantity"],
                    " [DRY-RUN]" if self.dry_run else "")
        self._notify_telegram(symbol, "PARTIAL-EXIT", "EXIT", part_qty, current_price,
                              pnl=pnl_part, reason=f"two-phase partial @ +{cfg.TWO_PHASE_PARTIAL_AT_PCT}%")
        return True

    def _get_base_exit(self, symbol, trade, pnl_pct):
        from datetime import time as dtime
        now = self._now_ist()
        is_late = now.time() >= dtime(15, 0)
        base = 75
        if is_late:
            base -= 10
        if pnl_pct < -1.5:
            base = 55
        return base

    async def _kronos_exit(self, symbol, trade, current_price, reason):
        quantity = trade.get("quantity", 0)
        is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY

        tag = "KRONOS-EXIT"

        if self.dry_run:
            logger.info("%s DRY-RUN %s qty=%d reason=%s", symbol, tag, quantity, reason)

        # _exit_position is the SOLE order-placement site — it sends the single
        # closing order. Do NOT also call reduce_position here or every exit
        # fires twice (double-close into an opposite position).
        closed = await self.bot_instance._exit_position(symbol, tag)
        if not closed:
            # Order rejected (or nothing to close): don't fake PnL / consec-loss.
            return

        pnl = pnl_pct = None
        if trade.get("entry_price", 0) > 0:
            ltp = current_price
            if ltp > 0:
                if is_buy:
                    pnl = (ltp - trade["entry_price"]) * quantity
                    pnl_pct = (ltp - trade["entry_price"]) / trade["entry_price"] * 100
                else:
                    pnl = (trade["entry_price"] - ltp) * quantity
                    pnl_pct = (trade["entry_price"] - ltp) / trade["entry_price"] * 100

        # Track consecutive losses
        bot = self.bot_instance
        if pnl is not None and pnl < 0:
            bot._consecutive_losses = getattr(bot, "_consecutive_losses", 0) + 1
            max_consec = cfg.MAX_CONSECUTIVE_LOSSES
            if max_consec > 0 and bot._consecutive_losses >= max_consec:
                logger.warning("CONSECUTIVE LOSS BREAKER: %d consecutive losses (max %d) — stopping new entries",
                               bot._consecutive_losses, max_consec)
        elif pnl is not None and pnl > 0:
            bot._consecutive_losses = 0

        self._notify_telegram(symbol, tag, "EXIT", quantity, trade.get("entry_price", 0),
                              pnl=pnl, pnl_pct=pnl_pct, reason=reason)

    async def run(self):
        logger.info("Kronos-enhanced exit guardian started (live position sync enabled)")
        # Delay first tick so the bot's initial quote burst finishes before we
        # hit the Dhan API — avoids DH-901 false-positives from rate limiting.
        await asyncio.sleep(15)
        while True:
            try:
                if not self.bot_instance.is_market_hours():
                    await asyncio.sleep(self.poll_interval)
                    continue

                # ── Fetch live positions from Dhan (source of truth in live mode) ──
                dhan_positions = {}
                if not self.dry_run:
                    async with self.bot_instance._dhan_sem:
                        dhan_positions = await asyncio.to_thread(self.dhan.fetch_positions)

                    # Sync: remove trades no longer open on Dhan (SL/TP filled, manual exit)
                    for symbol in list(self.bot_instance.active_trades.keys()):
                        if symbol not in dhan_positions:
                            logger.info(
                                "Guardian: %s no longer in Dhan positions — removing from tracking",
                                symbol,
                            )
                            del self.bot_instance.active_trades[symbol]
                            # Clean up guardian cooldown state too
                            if symbol in self.cooldown_state:
                                del self.cooldown_state[symbol]

                    # Sync: discover positions on Dhan that bot doesn't know about
                    for symbol, pos in dhan_positions.items():
                        if symbol not in self.bot_instance.active_trades:
                            logger.info(
                                "Guardian: discovered live position %s on Dhan "
                                "(qty=%d, entry=%.2f, pnl=%.2f) — adding to tracking",
                                symbol, pos["quantity"], pos["entry_price"],
                                pos.get("unrealized_pnl", 0),
                            )
                            self.bot_instance.active_trades[symbol] = {
                                "symbol": symbol,
                                "security_id": pos.get("security_id", self.dhan.security_ids.get(symbol, "")),
                                "entry_price": pos["entry_price"],
                                "quantity": pos["quantity"],
                                "transaction_type": pos["transaction_type"],
                                "order_id": f"DHAN-{symbol}",
                                "entry_time": self.bot_instance._now_ist(),
                                "trailing_sl": 0,
                                "atr_value": 1,
                                "stop_loss_percent": 0,
                                "target_percent": 0,
                            }

                active_trades = self.bot_instance.active_trades
                if not active_trades:
                    if not self.dry_run and dhan_positions:
                        logger.info("Guardian: Dhan returned %d positions but none matched known symbols",
                                    len(dhan_positions))
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Pre-cache live quotes for all active positions in one batch
                active_sids = [
                    self.dhan.security_ids[sym]
                    for sym in active_trades
                    if sym in self.dhan.security_ids
                ]
                if active_sids:
                    async with self.bot_instance._dhan_sem:
                        await asyncio.to_thread(self.dhan.cache_live_quotes, active_sids)

                for symbol, trade in list(active_trades.items()):
                    security_id = self.dhan.security_ids.get(symbol)
                    if not security_id:
                        continue
                    async with self.bot_instance._dhan_sem:
                        live = await asyncio.to_thread(self.dhan.fetch_live_data, security_id)
                    current_price = live.get("last_price")
                    if not current_price:
                        continue

                    if not self.dry_run:
                        dhan_pnl = dhan_positions.get(symbol, {}).get("unrealized_pnl")
                        if dhan_pnl is not None:
                            entry = trade["entry_price"]
                            pnl_pct = (dhan_pnl / (entry * trade["quantity"]) * 100) if entry > 0 and trade["quantity"] > 0 else 0
                        else:
                            _, pnl_pct = self.bot_instance._calc_pnl(trade, current_price)
                    else:
                        _, pnl_pct = self.bot_instance._calc_pnl(trade, current_price)

                    await self.check_position(symbol, trade, current_price, pnl_pct)

            except Exception as e:
                logger.exception("Kronos guardian error: %s", e)

            await asyncio.sleep(self.poll_interval)
