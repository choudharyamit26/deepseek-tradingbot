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
        await super().check_position(symbol, trade, current_price, pnl_pct)

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
                    if old_sl := trade.get("sl_price", 0) == 0 or new_sl < old_sl:
                        self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
                        self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
                        logger.info("%s trailing SL ACTIVATED at %.2f (pnl=%.2f%%, dist=%.1f*ATR)",
                                    symbol, new_sl, pnl_pct, trail_dist)

        # ── Time-based exit ─────────────────────────────────────────────────
        max_duration = cfg.MAX_TRADE_DURATION_MINUTES
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
                logger.info("%s market close exit: time=%.2f cutoff=%02d:%02d",
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
                    "%s Kronos exit: urgency=%d pred_return=%.2f%%",
                    symbol, urgency, pred_return,
                )

                if urgency >= self._get_base_exit(symbol, trade, pnl_pct) or \
                   (urgency >= 40 and pnl_pct < -1.5):
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
        trans_type = self.dhan.dhan.SELL if is_buy else self.dhan.dhan.BUY
        security_id = trade.get("security_id", self.dhan.security_ids.get(symbol, ""))

        if self.dry_run:
            logger.info("%s DRY-RUN %s qty=%d reason=%s", symbol, reason, quantity, reason)
        else:
            if security_id:
                result = await asyncio.to_thread(self.dhan.reduce_position, security_id, trans_type, quantity)
                logger.info("%s %s result: %s", symbol, reason, result)

        await self.bot_instance._exit_position(symbol, reason)

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
        trans_type = self.dhan.dhan.SELL if is_buy else self.dhan.dhan.BUY
        security_id = trade.get("security_id", self.dhan.security_ids.get(symbol, ""))

        tag = "KRONOS-EXIT"

        if self.dry_run:
            logger.info("%s DRY-RUN %s qty=%d reason=%s", symbol, tag, quantity, reason)
        else:
            if security_id:
                result = await asyncio.to_thread(self.dhan.reduce_position, security_id, trans_type, quantity)
                logger.info("%s %s result: %s", symbol, tag, result)

        await self.bot_instance._exit_position(symbol, tag)

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
