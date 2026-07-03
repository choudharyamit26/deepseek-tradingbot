import asyncio
import logging
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import talib

from indicators import calculate_technical_indicators
from reversal_detector import detect_reversals

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

class PositionExitGuardian:
    def __init__(self, dhan_bot, bot_instance, send_telegram=None, format_signal_msg=None, enable_telegram=False, dry_run=True):
        self.dhan = dhan_bot
        self.bot_instance = bot_instance  # Ref to IntradayStockBot to share dry-run state
        self.send_telegram = send_telegram
        self.format_signal_msg = format_signal_msg
        self.enable_telegram = enable_telegram
        self.dry_run = dry_run
        
        self.poll_interval = 30  # 30 seconds
        self.cooldown_state = {} # symbol -> {"time": ts, "score": score}

    def _now_ist(self):
        return datetime.now(IST)

    def _notify_telegram(self, symbol, tag, direction, quantity, price, pnl=None, pnl_pct=None, reason=""):
        if not self.enable_telegram or not self.send_telegram:
            return
        msg = self.format_signal_msg(symbol, tag, direction, quantity, price, pnl=pnl, pnl_pct=pnl_pct)
        if reason:
            msg += f"\nReason: {reason}"
        self.send_telegram(msg)
        
    def _calculate_technical_indicators(self, df):
        return calculate_technical_indicators(df, min_bars=5)

    async def check_position(self, symbol, trade, current_price, pnl_pct):
        security_id = self.dhan.security_ids.get(symbol)
        if not security_id:
            return
            
        is_buy = trade.get("transaction_type", self.dhan.dhan.BUY) == self.dhan.dhan.BUY

        # ── Trailing SL activation ──────────────────────────────────────────
        activation_pct = trade.get("trail_activation_pct", 3.0) if "trail_activation_pct" in trade else 0
        if activation_pct > 0 and trade.get("trailing_sl", 0) == 0 and pnl_pct >= activation_pct:
            atr = trade.get("atr_value", 1)
            trail_dist = trade.get("trail_distance_atr", 2.0) if "trail_distance_atr" in trade else 2.0
            if is_buy:
                new_sl = current_price - trail_dist * atr
                if new_sl > trade.get("entry_price", 0):
                    self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
                    self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
                    logger.info("%s trailing SL ACTIVATED at %.2f (pnl=%.2f%%)", symbol, new_sl, pnl_pct)
            else:
                new_sl = current_price + trail_dist * atr
                old_sl = trade.get("sl_price", 0)
                if old_sl == 0 or new_sl < old_sl:
                    self.bot_instance.active_trades[symbol]["trailing_sl"] = new_sl
                    self.bot_instance.active_trades[symbol]["sl_price"] = new_sl
                    logger.info("%s trailing SL ACTIVATED at %.2f (pnl=%.2f%%)", symbol, new_sl, pnl_pct)

        # ── Time-based exit check ──────────────────────────────────────────
        entry_time = trade.get("entry_time")
        if entry_time:
            now = self._now_ist()
            if isinstance(entry_time, datetime):
                elapsed = (now - entry_time).total_seconds() / 60
            else:
                elapsed = 0
            max_dur = getattr(self.bot_instance, "max_trade_duration_minutes", 0) or 0
            # Phase-2 runners (two-phase exit) are breakeven-floored — exempt
            # from the time exit; only their trail or market close ends them.
            if trade.get("tp2_active"):
                max_dur = 0
            if max_dur > 0 and elapsed >= max_dur:
                logger.info("%s time-based exit: held %.0f min (max %d min)", symbol, elapsed, max_dur)
                await self.bot_instance._exit_position(symbol, f"TIME-EXIT {int(elapsed)}min")
                return

        # Dual timeframe analysis
        async with self.bot_instance._dhan_sem:
            hist_3m, hist_5m = await asyncio.gather(
                asyncio.to_thread(self.dhan.get_historical_data, security_id, "3minute", min_bars=5),
                asyncio.to_thread(self.dhan.get_historical_data, security_id, "5minute", min_bars=5)
            )
        
        if len(hist_3m) < 5 or len(hist_5m) < 5:
            return
            
        ind_3m = self._calculate_technical_indicators(hist_3m)
        ind_5m = self._calculate_technical_indicators(hist_5m)
        
        if not ind_3m or not ind_5m:
            return
            
        rev_3m = detect_reversals(hist_3m, is_buy, ind_3m)
        rev_5m = detect_reversals(hist_5m, is_buy, ind_5m)
        
        # Determine exit thresholds
        now = self._now_ist()
        is_late = now.time() >= dtime(15, 0)
        
        base_exit = 75
        base_partial = 60
        base_tighten = 40
        
        if is_late:
            base_exit -= 10
            base_partial -= 10
            
        if pnl_pct < -1.5:
            base_exit = 55
        elif pnl_pct > 0.5:
            base_partial -= 5
            
        score_3m = rev_3m.score
        score_5m = rev_5m.score
        
        # Confirm dual timeframe agreement for strong exits
        tf_agree_exit = score_3m >= base_exit and score_5m >= base_exit
        
        max_score = max(score_3m, score_5m)
        
        # Cooldown check
        last_check = self.cooldown_state.get(symbol)
        
        async def execute_exit(reason_desc):
            logger.info("Guardian triggering exit for %s. Reason: %s", symbol, reason_desc)
            
            # Extract specific reversal reason if available
            exit_tag = "REV-GUARDIAN"
            if rev_3m.signals:
                exit_tag = "REV-" + rev_3m.signals[0].name.upper().replace(" ", "-")
            elif rev_5m.signals:
                exit_tag = "REV-" + rev_5m.signals[0].name.upper().replace(" ", "-")
                
            await self.bot_instance._exit_position(symbol, exit_tag)
            if symbol in self.cooldown_state:
                del self.cooldown_state[symbol]
                
        if score_3m >= base_exit or tf_agree_exit or pnl_pct < -2.0:
            if last_check and (time.time() - last_check["time"]) < 60:
                if score_3m >= base_exit or tf_agree_exit:
                    await execute_exit(f"Confirmed Full Exit (Score: {score_3m})")
                else:
                    self.cooldown_state[symbol] = {"time": time.time(), "score": score_3m}
            else:
                logger.info("Guardian armed full exit for %s, waiting for confirmation", symbol)
                self.cooldown_state[symbol] = {"time": time.time(), "score": score_3m}
                
        elif max_score >= base_partial and pnl_pct > 0:
             new_sl = current_price - ind_3m.get("atr", 1) if is_buy else current_price + ind_3m.get("atr", 1)
             old_sl = trade.get("trailing_sl", 0)
             if is_buy and new_sl > old_sl:
                 trade["trailing_sl"] = new_sl
             elif not is_buy and (old_sl == 0 or new_sl < old_sl):
                 trade["trailing_sl"] = new_sl
        elif max_score >= base_tighten:
            new_sl = current_price - ind_3m.get("atr", 1) if is_buy else current_price + ind_3m.get("atr", 1)
            old_sl = trade.get("trailing_sl", 0)
            if is_buy and new_sl > old_sl:
                trade["trailing_sl"] = new_sl
            elif not is_buy and (old_sl == 0 or new_sl < old_sl):
                trade["trailing_sl"] = new_sl
        else:
            if symbol in self.cooldown_state:
                del self.cooldown_state[symbol]

    async def run(self):
        logger.info("Starting Position Exit Guardian (live position sync enabled)...")
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

                    # Sync: remove trades no longer open on Dhan
                    for symbol in list(self.bot_instance.active_trades.keys()):
                        if symbol not in dhan_positions:
                            logger.info(
                                "Guardian: %s no longer in Dhan positions — removing",
                                symbol,
                            )
                            del self.bot_instance.active_trades[symbol]
                            if symbol in self.cooldown_state:
                                del self.cooldown_state[symbol]

                    # Sync: discover positions on Dhan not tracked by bot
                    for symbol, pos in dhan_positions.items():
                        if symbol not in self.bot_instance.active_trades:
                            logger.info(
                                "Guardian: discovered live position %s "
                                "(qty=%d, entry=%.2f, pnl=%.2f) — adding",
                                symbol, pos["quantity"], pos["entry_price"],
                                pos.get("unrealized_pnl", 0),
                            )
                            self.bot_instance.active_trades[symbol] = {
                                "symbol": symbol,
                                "entry_price": pos["entry_price"],
                                "quantity": pos["quantity"],
                                "transaction_type": pos["transaction_type"],
                                "order_id": f"DHAN-{symbol}",
                                "entry_time": self.bot_instance._now_ist(),
                                "trailing_sl": 0,
                                "atr_value": 1,
                            }

                active_trades = self.bot_instance.active_trades
                if not active_trades:
                    await asyncio.sleep(self.poll_interval)
                    continue

                for symbol, trade in list(active_trades.items()):
                    security_id = self.dhan.security_ids.get(symbol)
                    if not security_id: continue
                    async with self.bot_instance._dhan_sem:
                        live = await asyncio.to_thread(self.dhan.fetch_live_data, security_id)
                    current_price = live.get("last_price")
                    if not current_price: continue
                    
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
                logger.exception("Guardian error: %s", e)
                
            await asyncio.sleep(self.poll_interval)

