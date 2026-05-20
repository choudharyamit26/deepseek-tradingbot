import asyncio
import logging
import time
import pandas as pd
import numpy as np
import talib
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from signal_logger import SignalLogger
from regime_filter import RegimeFilter

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# ── Timing constants ───────────────────────────────────────────────────────────
# Delay entry to 9:45 to skip opening volatility and allow indicators to warm up
FIRST_ENTRY_HOUR, FIRST_ENTRY_MIN = 9, 30
LAST_ENTRY_HOUR, LAST_ENTRY_MIN = 15, 0
MAX_SIGNALS_PER_STOCK_PER_DAY = 2
MAX_CONCURRENT_POSITIONS = 3
# Align scan interval with 3-minute candle timeframe (was 60s → duplicate signals)
SCAN_INTERVAL = 180

# ── Quality gates ──────────────────────────────────────────────────────────────
MIN_PREFILTER_VOLUME_RATIO = 0.3     # Was 0.5 — captures early-morning/low-volume periods
MIN_PREFILTER_ATR_PCT = 0.2          # Was 0.3 — 0.2% is enough for quick scalps
NEUTRAL_RSI_LOW, NEUTRAL_RSI_HIGH = 45, 55  # Was 40-60 — tighter dead zone, lets more through
MIN_RR_RATIO = 1.8                   # Minimum reward:risk ratio
MIN_ADX_TRENDING = 18                # ADX below this = choppy/ranging market

_ENTRY_START_T = dtime(FIRST_ENTRY_HOUR, FIRST_ENTRY_MIN)
_ENTRY_END_T = dtime(LAST_ENTRY_HOUR, LAST_ENTRY_MIN)
_MARKET_OPEN_T = dtime(9, 15)
_MARKET_CLOSE_T = dtime(15, 30)


class IntradayStockBot:
    def __init__(self, dhan_bot, ai_analyzer, risk_manager, watchlist=None,
                 send_telegram=None, format_signal_msg=None,
                 enable_telegram=False, dry_run=True):
        self.dhan = dhan_bot
        self.ai = ai_analyzer
        self.risk = risk_manager
        self.watchlist = watchlist or list(self.dhan.security_ids.keys())
        self.active_trades: dict = {}
        self.last_signal_time: dict[str, float] = {}
        # Increased from 60s to 900s (15 min) to prevent duplicate signals
        self.cooldown_seconds = 900
        self.send_telegram = send_telegram
        self.format_signal_msg = format_signal_msg
        self.enable_telegram = enable_telegram
        self.dry_run = dry_run
        self.signal_log = SignalLogger()
        self.regime = RegimeFilter(dhan_bot)
        self._api_sem = asyncio.Semaphore(3)
        self._last_date: str = datetime.now(IST).strftime("%Y-%m-%d")

    def _now_ist(self):
        return datetime.now(IST)

    def _reset_daily_if_needed(self):
        today = self._now_ist().strftime("%Y-%m-%d")
        if today != self._last_date:
            self._last_date = today
            self.last_signal_time.clear()
            self.signal_log.reset_daily()

    @staticmethod
    def _time_in_range(start: dtime, end: dtime, x: dtime) -> bool:
        return start <= x <= end

    def is_entry_allowed(self):
        now = self._now_ist()
        if now.weekday() >= 5:
            return False
        return self._time_in_range(_ENTRY_START_T, _ENTRY_END_T, now.time())

    def is_market_hours(self):
        now = self._now_ist()
        if now.weekday() >= 5:
            return False
        return self._time_in_range(_MARKET_OPEN_T, _MARKET_CLOSE_T, now.time())

    # Increased from 5 to 20 to let RSI-14 and SMA-20 stabilize properly
    MIN_BARS = 5
    MIN_BARS_15M = 3
    MIN_BARS_1H = 2

    # ── IMPROVEMENT #9 & #10: Fixed VWAP + Added ADX/MFI ──────────────────────
    def calculate_technical_indicators(self, df):
        if len(df) < self.MIN_BARS:
            return {}

        df["rsi"] = talib.RSI(df["close"], timeperiod=14)
        df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
        df["sma_20"] = talib.SMA(df["close"], timeperiod=20)
        df["ema_9"] = talib.EMA(df["close"], timeperiod=9)
        upper, _, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
        df["bb_percent_b"] = (df["close"] - lower) / (upper - lower)
        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

        # IMPROVEMENT #10: ADX (trend strength) and MFI (money flow)
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
        df["mfi"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)

        # IMPROVEMENT #9: VWAP with daily reset
        if hasattr(df.index, 'date'):
            vwap_parts = []
            for _date, group in df.groupby(df.index.date):
                cum_vol = group["volume"].cumsum().clip(lower=1e-9)
                cum_vp = (group["close"] * group["volume"]).cumsum()
                vwap_parts.append(cum_vp / cum_vol)
            df["vwap"] = pd.concat(vwap_parts)
        else:
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum().clip(lower=1e-9)

        # Fix #4: Shortened from 20→10 to avoid skewed afternoon ratios
        # (morning high-volume bars inflated the avg, making PM ratios ~0.00x)
        vol_lookback = min(10, len(df))
        vol_ma = df["volume"].rolling(window=vol_lookback).mean().clip(lower=1e-9)
        df["volume_ratio"] = df["volume"] / vol_ma
        df["resistance"] = df["high"].rolling(window=20).max()
        df["support"] = df["low"].rolling(window=20).min()

        latest = df.iloc[-1]
        close_val = round(latest["close"], 2)
        vwap_val = round(latest["vwap"], 2) if not pd.isna(latest["vwap"]) else close_val
        # VWAP distance as a percentage — key context for the LLM
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
        }

    def _build_mtf_summary(self, i3m, i15m, i1h) -> str:
        lines = []
        for label, ind in [("3-Min", i3m), ("15-Min", i15m), ("1-Hour", i1h)]:
            if not ind:
                continue
            close = ind.get("close", 0)
            sma = ind.get("sma_20", close)
            trend = "BULLISH" if close > sma else "BEARISH"
            rsi = ind.get("rsi", 50)
            adx = ind.get("adx", 20)
            lines.append(f"{label}: RSI={rsi}, ADX={adx}, Price={close:.2f}, SMA20={sma:.2f} -> {trend}")
        if len(lines) >= 2:
            return "Multi-Timeframe Analysis:\n" + "\n".join(lines)
        return ""

    # ── IMPROVEMENT #1: Hard technical pre-filters BEFORE calling AI ───────────
    def _passes_prefilter(self, indicators: dict, regime_data: dict) -> tuple[bool, str]:
        """Gate the AI call with hard rules. Returns (pass, reason)."""
        rsi = indicators.get("rsi", 50)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        atr = indicators.get("atr", 0)
        close = indicators.get("close", 0)
        adx = indicators.get("adx", 20)

        # 1. Reject dead-zone RSI (40-60) with no volume spike — nothing is happening
        if NEUTRAL_RSI_LOW < rsi < NEUTRAL_RSI_HIGH and volume_ratio < 1.3:
            return False, f"RSI neutral ({rsi:.0f}) + no volume spike ({volume_ratio:.1f}x)"

        # 2. Reject if volume is dead
        if volume_ratio < MIN_PREFILTER_VOLUME_RATIO:
            return False, f"Volume too low ({volume_ratio:.2f}x avg)"

        # 3. Reject if ATR is too small (stock not moving enough for intraday)
        atr_pct = (atr / close * 100) if close > 0 else 0
        if atr_pct < MIN_PREFILTER_ATR_PCT:
            return False, f"ATR too low ({atr_pct:.2f}%)"

        # 4. Reject if ADX shows no trend (choppy/ranging market)
        if adx < MIN_ADX_TRENDING:
            return False, f"ADX too low ({adx:.0f}) — market is ranging"

        return True, "OK"

    # ── IMPROVEMENT #7: Hard MTF alignment veto AFTER AI signal ────────────────
    @staticmethod
    def _validate_mtf_alignment(sig_type: str, i3m: dict, i15m: dict, i1h: dict) -> tuple[bool, str]:
        """Veto signals that contradict multi-timeframe alignment."""
        def _trend(ind):
            if not ind:
                return "NEUTRAL"
            c = ind.get("close", 0)
            s = ind.get("sma_20", c)
            return "BULLISH" if c > s else "BEARISH" if c < s else "NEUTRAL"

        t3, t15, t1h = _trend(i3m), _trend(i15m), _trend(i1h)

        if sig_type == "BUY":
            if t1h == "BEARISH" and t15 == "BEARISH":
                return False, f"BUY vetoed: 1H={t1h}, 15m={t15} both bearish"
        elif sig_type == "SELL":
            if t1h == "BULLISH" and t15 == "BULLISH":
                return False, f"SELL vetoed: 1H={t1h}, 15m={t15} both bullish"
        return True, f"MTF OK: 3m={t3}, 15m={t15}, 1H={t1h}"

    def _notify_telegram(self, symbol, tag, direction, quantity, price,
                         sl_price=0.0, tp1_price=0.0, tp2_price=0.0):
        if not self.enable_telegram or not self.send_telegram:
            return
        msg = self.format_signal_msg(symbol, tag, direction, quantity, price,
                                     sl_price, tp1_price, tp2_price)
        self.send_telegram(msg)

    async def analyze_stock(self, symbol):
        async with self._api_sem:
            await self._analyze(symbol)

    async def _analyze(self, symbol):
        self._reset_daily_if_needed()
        logger.info("Analyzing %s...", symbol)

        security_id = self.dhan.security_ids.get(symbol)
        if not security_id:
            logger.warning("Security ID not found for %s", symbol)
            return

        if not self.is_entry_allowed():
            logger.info("%s -- entry window closed", symbol)
            return

        if not self.signal_log.can_trade(symbol, MAX_SIGNALS_PER_STOCK_PER_DAY):
            logger.info("%s -- daily signal limit reached", symbol)
            return

        # ── IMPROVEMENT #2: Move cooldown check BEFORE AI call ─────────────────
        last_time = self.last_signal_time.get(symbol, 0)
        if time.time() - last_time < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (time.time() - last_time)
            logger.info("%s -- cooldown active (%.0fs remaining)", symbol, remaining)
            return

        historical = self.dhan.get_historical_data(security_id, "3minute", min_bars=self.MIN_BARS)
        if len(historical) < self.MIN_BARS:
            logger.info("%s -- insufficient 3m bars (%d, need %d)", symbol, len(historical), self.MIN_BARS)
            return

        indicators_3m = self.calculate_technical_indicators(historical)
        if not indicators_3m:
            return

        # ── Fetch regime data early (needed for prefilter) ─────────────────────
        regime_data = self.regime.get_regime(symbol)

        # ── IMPROVEMENT #1: Pre-filter before AI call ──────────────────────────
        passed, reason = self._passes_prefilter(indicators_3m, regime_data)
        if not passed:
            logger.info("%s -- pre-filter rejected: %s", symbol, reason)
            return

        historical_15m = self.dhan.get_historical_data(security_id, "15minute", min_bars=self.MIN_BARS_15M)
        indicators_15m = self.calculate_technical_indicators(historical_15m) if len(historical_15m) >= self.MIN_BARS_15M else {}

        historical_1h = self.dhan.get_historical_data(security_id, "60minute", min_bars=self.MIN_BARS_1H)
        indicators_1h = self.calculate_technical_indicators(historical_1h) if len(historical_1h) >= self.MIN_BARS_1H else {}

        live = self.dhan.fetch_live_data(security_id)
        ltp = live.get("last_price") or historical["close"].iloc[-1]
        market_data = {
            "ltp": ltp,
            "high_3m": live.get("high_price") or historical["high"].iloc[-1],
            "low_3m": live.get("low_price") or historical["low"].iloc[-1],
            "volume": live.get("volume") or historical["volume"].iloc[-1],
            "avg_volume_3m": historical["volume"].tail(5).mean(),
        }

        regime_context = self.regime.format_regime_context(symbol, regime_data)
        mtf_summary = self._build_mtf_summary(indicators_3m, indicators_15m, indicators_1h)
        full_context = regime_context + "\n\n" + mtf_summary if mtf_summary else regime_context

        # ── IMPROVEMENT #5: Pass recent bars for candle-history context ────────
        recent_bars = historical.tail(10) if len(historical) >= 10 else historical
        signal = self.ai.get_trading_signal(symbol, market_data, indicators_3m,
                                            full_context, recent_bars=recent_bars)
        logger.info("%s - AI Signal: %s (conf=%s)", symbol,
                    signal.get("signal", "UNKNOWN"), signal.get("confidence", "?"))

        current_trade = self.active_trades.get(symbol)
        sig_type = signal.get("signal", "HOLD")
        confidence = signal.get("confidence", 0)
        reasoning = signal.get("reasoning", "")

        if sig_type == "EXIT" and current_trade:
            await self._exit_position(symbol)
            return

        if sig_type not in ("BUY", "SELL") or confidence < self.risk.min_confidence:
            return

        # ── IMPROVEMENT #7: Hard MTF alignment veto ────────────────────────────
        mtf_ok, mtf_reason = self._validate_mtf_alignment(sig_type, indicators_3m, indicators_15m, indicators_1h)
        if not mtf_ok:
            logger.info("%s -- %s", symbol, mtf_reason)
            return

        if not self.risk.check_daily_trade_limit() or not self.risk.check_daily_loss_limit():
            logger.info("Daily trade or loss limit hit")
            return

        if len(self.active_trades) >= MAX_CONCURRENT_POSITIONS:
            logger.info("%s -- max concurrent positions (%d) reached", symbol, MAX_CONCURRENT_POSITIONS)
            return

        if symbol in self.active_trades:
            logger.info("%s -- already in position", symbol)
            return

        atr_value = indicators_3m.get("atr", 1) if isinstance(indicators_3m.get("atr"), (int, float)) else 1
        atr_pct = (atr_value / ltp * 100) if ltp > 0 else 1.0
        sl_percent = signal.get("stop_loss_percent", round(atr_pct * 1.5, 2))
        target_percent = signal.get("target_percent", round(atr_pct * 3.0, 2))

        # ── IMPROVEMENT #8: Enforce minimum R:R ratio ──────────────────────────
        if sl_percent > 0 and target_percent < sl_percent * MIN_RR_RATIO:
            logger.info("%s -- R:R too low (target=%.2f%% / SL=%.2f%% = %.1fx, need %.1fx)",
                        symbol, target_percent, sl_percent,
                        target_percent / sl_percent if sl_percent > 0 else 0, MIN_RR_RATIO)
            return

        capital = self.risk.current_capital
        quantity = self.risk.calculate_position_size(capital, sl_percent, ltp)

        # ── IMPROVEMENT #3: Block quantity=0 BEFORE logging/Telegram ───────────
        if quantity < 1:
            # Fix #3: Set cooldown even on qty=0 to prevent duplicate AI calls
            # (LTIM got 2 SELL signals 25 min apart because cooldown was never set)
            self.last_signal_time[symbol] = time.time()
            logger.info("%s %s pre-filtered: quantity=0 (price too high for capital)",
                        symbol, sig_type)
            return

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        trans_type = self.dhan.dhan.BUY if sig_type == "BUY" else self.dhan.dhan.SELL
        sl_price = ltp * (1 - sl_percent / 100) if sig_type == "BUY" else ltp * (1 + sl_percent / 100)
        target_price = ltp * (1 + target_percent / 100) if sig_type == "BUY" else ltp * (1 - target_percent / 100)

        tag = "ENTRY-LONG" if sig_type == "BUY" else "ENTRY-SHORT"

        nifty_regime = regime_data.get("nifty", {}).get("trend", "")
        sector_name = regime_data.get("sector_name", "")
        sector_regime = ""
        if regime_data.get("sector"):
            s_trend = regime_data["sector"].get("trend", "")
            sector_regime = f"{sector_name}={s_trend.upper()}" if sector_name else s_trend.upper()

        def _tf_trend(ind):
            c = ind.get("close", 0)
            s = ind.get("sma_20", c)
            return "BULLISH" if c > s else "BEARISH" if c < s else "NEUTRAL"

        mtf_3m = _tf_trend(indicators_3m)
        mtf_15m = _tf_trend(indicators_15m) if indicators_15m else ""
        mtf_1h = _tf_trend(indicators_1h) if indicators_1h else ""

        self.signal_log.log_signal(
            symbol=symbol, signal_type=tag, direction=sig_type,
            entry_price=ltp, quantity=quantity, stop_loss=sl_price,
            target=target_price, confidence=confidence,
            reasoning=reasoning, mode=mode,
            market_regime=nifty_regime.upper(), sector_regime=sector_regime.upper(),
            mtf_3m=mtf_3m, mtf_15m=mtf_15m, mtf_1h=mtf_1h,
        )

        self._notify_telegram(symbol, tag, sig_type, quantity, ltp, sl_price=sl_price, tp1_price=target_price)

        if self.dry_run:
            self.last_signal_time[symbol] = time.time()
            self.active_trades[symbol] = {
                "symbol": symbol, "entry_price": ltp, "quantity": quantity,
                "transaction_type": trans_type, "order_id": f"DRY-{symbol}-{int(time.time())}",
                "stop_loss_percent": sl_percent, "target_percent": target_percent,
                "entry_time": self._now_ist(),
            }
            logger.info("%s %s signal generated: %d shares @ %.2f | R:R=%.1f:1 (DRY-RUN)",
                        symbol, sig_type, quantity, ltp,
                        target_percent / sl_percent if sl_percent > 0 else 0)
            return

        order = self.dhan.place_super_order(
            security_id=security_id, transaction_type=trans_type,
            quantity=quantity, entry_price=ltp,
            sl_percent=sl_percent, target_percent=target_percent,
            symbol=symbol,
        )

        if order and order.get("status") == "success":
            self.last_signal_time[symbol] = time.time()
            self.active_trades[symbol] = {
                "symbol": symbol, "entry_price": ltp, "quantity": quantity,
                "transaction_type": trans_type, "order_id": order.get("data", {}).get("orderId"),
                "stop_loss_percent": sl_percent, "target_percent": target_percent,
                "entry_time": self._now_ist(),
            }
            logger.info("%s %s super order placed: %d shares @ %.2f (SL=%s)",
                        symbol, sig_type, quantity, ltp, sl_price)
            self.risk.record_trade()
            self._notify_telegram(symbol, "ORDER-PLACED", sig_type, quantity, ltp,
                                 sl_price=sl_price, tp1_price=target_price)
        else:
            remarks = order.get("remarks", "unknown") if order else "no response"
            logger.warning("%s %s super order FAILED: %s", symbol, sig_type, remarks)

    async def _exit_position(self, symbol):
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        exit_trans = (self.dhan.dhan.SELL if trade["transaction_type"] == self.dhan.dhan.BUY
                      else self.dhan.dhan.BUY)
        security_id = self.dhan.security_ids.get(symbol)
        if not security_id:
            return

        exit_price = trade["entry_price"]
        pnl = 0.0

        if not self.dry_run:
            order = self.dhan.place_equity_order(
                security_id=security_id, transaction_type=exit_trans,
                quantity=trade["quantity"], product_type="INTRA",
            )
            if not order:
                return

        self.signal_log.log_exit(symbol, exit_price, pnl, "EXIT")
        self._notify_telegram(symbol, "EXIT", "EXIT", trade["quantity"], exit_price)
        logger.info("Closed position for %s", symbol)
        del self.active_trades[symbol]

    async def monitor_open_positions(self):
        now = self._now_ist()
        close_cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
        if now < close_cutoff:
            return
        for symbol in list(self.active_trades.keys()):
            await self._exit_position(symbol)
        logger.info("Market closing - all positions closed")

    async def run(self):
        logger.info("Starting Intraday Stock Trading Bot")
        logger.info("Entry window: %d:%02d AM - %d:%02d PM IST | Max %d signals/stock/day | Scan every %ds",
                    FIRST_ENTRY_HOUR, FIRST_ENTRY_MIN,
                    LAST_ENTRY_HOUR, LAST_ENTRY_MIN,
                    MAX_SIGNALS_PER_STOCK_PER_DAY, SCAN_INTERVAL)
        while True:
            try:
                self._reset_daily_if_needed()
                if not self.is_market_hours():
                    logger.info("Outside market hours -- sleeping %ds", SCAN_INTERVAL)
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                # ── IMPROVEMENT #11: Align scans with 3-min candle boundaries ──
                now = self._now_ist()
                seconds_into_candle = (now.minute % 3) * 60 + now.second
                if seconds_into_candle < 5:
                    # We're right at a candle boundary — wait a few seconds for data
                    await asyncio.sleep(5 - seconds_into_candle)

                logger.info("Scanning %d stocks...", len(self.watchlist))
                tasks = [self.analyze_stock(s) for s in self.watchlist]
                await asyncio.gather(*tasks)
                await self.monitor_open_positions()
                await asyncio.sleep(SCAN_INTERVAL)
            except Exception as e:
                logger.exception("Unexpected error in main loop: %s", e)
                await asyncio.sleep(SCAN_INTERVAL)
