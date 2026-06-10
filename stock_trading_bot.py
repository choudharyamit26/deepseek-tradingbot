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
from reversal_detector import detect_reversals

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
MIN_PREFILTER_VOLUME_RATIO = 0.15    # Floor for truly dead volume bars
MIN_PREFILTER_ATR_PCT = 0.30         # Raised to 0.30 based on reflection agent to filter out low-volatility choppy stocks
NEUTRAL_RSI_LOW, NEUTRAL_RSI_HIGH = 38, 62  # Dead zone — bypassed when trending
MIN_RR_RATIO = 1.8                   # Minimum reward:risk ratio
MIN_ADX_TRENDING = 18                # ADX below this = choppy/ranging market
RSI_OB_LIMIT = 70                    # Overbought RSI limit (avoid Buy)
RSI_OS_LIMIT = 30                    # Oversold RSI limit (avoid Sell)
RSI_EXTREME_OB = 78                  # Pre-AI gate: skip AI if RSI above this (any direction)
RSI_EXTREME_OS = 22                  # Pre-AI gate: skip AI if RSI below this (any direction)
SECTOR_REGIME_PENALTY = 12           # Confidence penalty when signal opposes sector trend
MAX_VWAP_DISTANCE_PCT = 1.0          # Max price distance from VWAP to prevent chasing overextended trends

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
        self._dhan_sem = asyncio.Semaphore(2)
        self._ai_sem = asyncio.Semaphore(3)
        self._exit_lock = asyncio.Lock()
        self._last_date: str = datetime.now(IST).strftime("%Y-%m-%d")
        self._atr_thresholds: dict[str, float | None] = {}
        self._atr_current: dict[str, float | None] = {}
        self._atr_building: set[str] = set()
        self._atr_prewarmed: bool = False
        self.filter_stats = {"atr_blocked": 0, "volume_blocked": 0, "reversal_blocked": 0, "total_scans": 0}

    async def _build_atr_profile(self, symbol: str, security_id: int):
        if symbol in self._atr_building:
            return
        self._atr_building.add(symbol)
        try:
            async with self._dhan_sem:
                daily = await asyncio.to_thread(
                    self.dhan.get_historical_data, security_id, "1day", min_bars=25)
            if daily is not None and len(daily) >= 25:
                daily["atr"] = talib.ATR(daily["high"], daily["low"], daily["close"], timeperiod=14)
                daily["atr_pct"] = daily["atr"] / daily["close"] * 100
                valid = daily["atr_pct"].dropna()
                self._atr_current[symbol] = round(float(valid.iloc[-1]), 4) if len(valid) > 0 else None
                if len(valid) >= 5:
                    self._atr_thresholds[symbol] = round(float(valid.quantile(0.20)), 4)
                    logger.debug("ATR profile for %s: p20=%.4f%%, current=%.4f%%", symbol, self._atr_thresholds[symbol], self._atr_current.get(symbol, 0))
                    return
            self._atr_thresholds[symbol] = None
        except Exception as e:
            logger.debug("ATR profile failed for %s: %s", symbol, e)
            self._atr_thresholds[symbol] = None
        finally:
            self._atr_building.discard(symbol)

    @staticmethod
    def _check_volume_exhaustion(df) -> bool:
        if df is None or len(df) < 5:
            return False
        recent = df.tail(5)
        bullish = recent[recent["close"] >= recent["open"]]
        bearish = recent[recent["close"] < recent["open"]]
        if len(bullish) < 2 or len(bearish) < 2:
            return False
        bv = bullish["volume"].tolist()
        bev = bearish["volume"].tolist()
        b_decl = bv[0] > bv[-1] and bv[-1] / bv[0] < 0.80
        be_rise = bev[0] < bev[-1] and bev[-1] / bev[0] > 1.25
        be_decl = bev[0] > bev[-1] and bev[-1] / bev[0] < 0.80
        b_rise = bv[0] < bv[-1] and bv[-1] / bv[0] > 1.25
        return (b_decl and be_rise) or (be_decl and b_rise)

    def _now_ist(self):
        return datetime.now(IST)

    def _reset_daily_if_needed(self):
        today = self._now_ist().strftime("%Y-%m-%d")
        if today != self._last_date:
            self._last_date = today
            self.last_signal_time.clear()
            self.signal_log.reset_daily()
            self.risk.reset_daily()

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
    MIN_BARS_15M = 25
    MIN_BARS_1H = 15

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
            if close > sma:
                trend = "BULLISH"
            elif close < sma:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
            rsi = ind.get("rsi", 50)
            adx = ind.get("adx", 20)
            lines.append(f"{label}: RSI={rsi}, ADX={adx}, Price={close:.2f}, SMA20={sma:.2f} -> {trend}")
        if len(lines) >= 2:
            return "Multi-Timeframe Analysis:\n" + "\n".join(lines)
        return ""

    # ── IMPROVEMENT #1: Hard technical pre-filters BEFORE calling AI ───────────
    def _passes_prefilter(self, indicators: dict, regime_data: dict) -> tuple[bool, str]:
        """Gate the AI call with hard rules. Returns (pass, reason).

        Design: Balances two market types:
          - CHOPPY days: strict RSI/volume/ATR filters block noise → fewer API calls
          - TRENDING days: if price > VWAP & SMA20 with ADX confirming direction,
            relax RSI dead-zone and volume checks → don't miss smooth trends
        ADX >= 18 is the universal safety net against ranging markets.
        """
        rsi = indicators.get("rsi", 50)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        atr = indicators.get("atr", 0)
        close = indicators.get("close", 0)
        adx = indicators.get("adx", 20)
        vwap = indicators.get("vwap", 0)
        sma20 = indicators.get("sma_20", close)
        ema9 = indicators.get("ema_9", close)

        # ── Hard gate: RSI extremes (direction-agnostic) ────────────────────
        if rsi > RSI_EXTREME_OB:
            return False, f"RSI overbought ({rsi:.0f} > {RSI_EXTREME_OB}) — overextended"
        if rsi < RSI_EXTREME_OS:
            return False, f"RSI oversold ({rsi:.0f} < {RSI_EXTREME_OS}) — overextended"

        # ── Hard gate: prevent chasing overextended trends ─────────────────
        vwap_dist = indicators.get("vwap_distance_pct", 0)
        if abs(vwap_dist) > MAX_VWAP_DISTANCE_PCT:
            return False, f"Price too far from VWAP ({vwap_dist:.2f}% > {MAX_VWAP_DISTANCE_PCT}%) — overextended"

        # ── Hard gate: ADX must confirm directional movement UNLESS strong volume spike ──
        if adx < MIN_ADX_TRENDING and volume_ratio < 1.8:
            return False, f"ADX too low ({adx:.0f}) and no volume spike ({volume_ratio:.2f}x) — ranging market"

        # ── Trend detection: price above both VWAP and SMA20/EMA9 = uptrend ─────
        #    price below both = downtrend. Either counts as "trending".
        is_trending = (close > vwap and (close > sma20 or close > ema9)) or \
                      (close < vwap and (close < sma20 or close < ema9))

        # ── Volume floor: reject truly dead bars (0.00x - 0.15x) always ────
        if volume_ratio < MIN_PREFILTER_VOLUME_RATIO:
            return False, f"Volume too low ({volume_ratio:.2f}x avg)"

        if is_trending:
            # TRENDING MODE: relaxed filters — the directional context
            # (VWAP + SMA20 + ADX >= 18) already confirms the stock is moving.
            # Only block if ATR is truly microscopic (< 0.05%)
            atr_pct = (atr / close * 100) if close > 0 else 0
            if atr_pct < 0.05:
                return False, f"ATR too low even for trend ({atr_pct:.2f}%)"
            # Volume floor even for trends — reject dead-volume bars
            if volume_ratio < 0.40:
                return False, f"Volume too low for trending ({volume_ratio:.2f}x avg)"
            # RSI dead zone is BYPASSED — a smooth trend naturally sits at RSI 50
            return True, "OK (trending)"

        # ── CHOPPY MODE: strict filters — no clear direction, avoid noise ──
        # 1. Reject dead-zone RSI with no volume spike
        if NEUTRAL_RSI_LOW < rsi < NEUTRAL_RSI_HIGH and volume_ratio < 1.3:
            return False, f"RSI neutral ({rsi:.0f}) + no volume spike ({volume_ratio:.1f}x)"

        # 2. Reject if ATR is too small for a non-trending stock
        atr_pct = (atr / close * 100) if close > 0 else 0
        if atr_pct < MIN_PREFILTER_ATR_PCT:
            return False, f"ATR too low ({atr_pct:.2f}%)"

        return True, "OK"

    # ── IMPROVEMENT #7: Hard MTF alignment veto AFTER AI signal ────────────────
    @staticmethod
    def _validate_mtf_alignment(sig_type: str, i3m: dict, i15m: dict, i1h: dict) -> tuple[bool, str]:
        """Veto signals that contradict multi-timeframe alignment."""
        def _trend_3m_1h(ind):
            if not ind:
                return "NEUTRAL"
            c = ind.get("close", 0)
            s = ind.get("sma_20", c)
            return "BULLISH" if c > s else "BEARISH" if c < s else "NEUTRAL"

        def _trend_15m(ind):
            if not ind:
                return "NEUTRAL"
            c = ind.get("close", 0)
            e = ind.get("ema_9", c)
            return "BULLISH" if c > e else "BEARISH"

        t3, t1h = _trend_3m_1h(i3m), _trend_3m_1h(i1h)
        t15 = _trend_15m(i15m)

        if sig_type == "BUY":
            if t15 != "BULLISH":
                return False, f"BUY vetoed: 15m trend is {t15} (need BULLISH)"
            if t1h == "BEARISH":
                return False, f"BUY vetoed: 1H trend is BEARISH"
        elif sig_type == "SELL":
            if t15 != "BEARISH":
                return False, f"SELL vetoed: 15m trend is {t15} (need BEARISH)"
            if t1h == "BULLISH":
                return False, f"SELL vetoed: 1H trend is BULLISH"
        return True, f"MTF OK: 3m={t3}, 15m={t15}, 1H={t1h}"

    def _notify_telegram(self, symbol, tag, direction, quantity, price,
                         sl_price=0.0, tp1_price=0.0, tp2_price=0.0,
                         trailing_sl=0.0, pnl=None, pnl_pct=None):
        if not self.enable_telegram or not self.send_telegram:
            return
        msg = self.format_signal_msg(symbol, tag, direction, quantity, price,
                                     sl_price, tp1_price, tp2_price,
                                     trailing_sl=trailing_sl, pnl=pnl, pnl_pct=pnl_pct)
        if asyncio.iscoroutinefunction(self.send_telegram):
            asyncio.create_task(self.send_telegram(msg))
        else:
            asyncio.create_task(asyncio.to_thread(self.send_telegram, msg))

    async def analyze_stock(self, symbol):
        await self._analyze(symbol)

    async def _analyze(self, symbol):
        self.filter_stats["total_scans"] += 1
        if not self._atr_prewarmed:
            self._atr_prewarmed = True
            for sym, sid in self.dhan.security_ids.items():
                if sym in self.watchlist:
                    asyncio.create_task(self._build_atr_profile(sym, sid))
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

        async with self._dhan_sem:
            historical = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "3minute", min_bars=self.MIN_BARS)
        if len(historical) < self.MIN_BARS:
            logger.info("%s -- insufficient 3m bars (%d, need %d)", symbol, len(historical), self.MIN_BARS)
            return

        indicators_3m = self.calculate_technical_indicators(historical)
        if not indicators_3m:
            return

        # ── Stock-specific ATR check (daily ATR% vs daily p20 floor) ───────
        if symbol not in self._atr_thresholds:
            asyncio.create_task(self._build_atr_profile(symbol, security_id))
        else:
            threshold = self._atr_thresholds.get(symbol)
            daily_atr = self._atr_current.get(symbol)
            if threshold is not None and daily_atr is not None and daily_atr > 0 and daily_atr < threshold:
                self.filter_stats["atr_blocked"] += 1
                logger.info("%s -- daily ATR %.3f%% below stock floor %.3f%%, skipping", symbol, daily_atr, threshold)
                return

        # ── Fetch regime data early (needed for prefilter) ─────────────────────
        regime_data = self.regime.get_regime(symbol)

        # ── IMPROVEMENT #1: Pre-filter before AI call ──────────────────────────
        passed, reason = self._passes_prefilter(indicators_3m, regime_data)
        if not passed:
            logger.info("%s -- pre-filter rejected: %s", symbol, reason)
            return

        async with self._dhan_sem:
            historical_15m = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "15minute", min_bars=self.MIN_BARS_15M)
        indicators_15m = self.calculate_technical_indicators(historical_15m) if len(historical_15m) >= self.MIN_BARS_15M else {}

        async with self._dhan_sem:
            historical_1h = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "60minute", min_bars=self.MIN_BARS_1H)
        indicators_1h = self.calculate_technical_indicators(historical_1h) if len(historical_1h) >= self.MIN_BARS_1H else {}

        async with self._dhan_sem:
            live = await asyncio.to_thread(self.dhan.fetch_live_data, security_id)
        ltp = live.get("last_price") or historical["close"].iloc[-1]
        market_data = {
            "ltp": ltp,
            "high_3m": live.get("high_price") or historical["high"].iloc[-1],
            "low_3m": live.get("low_price") or historical["low"].iloc[-1],
            "volume": live.get("volume") or historical["volume"].iloc[-1],
            "avg_volume_3m": historical["volume"].tail(5).mean(),
        }

        # ── Volume exhaustion check ─────────────────────────────────────────
        if self._check_volume_exhaustion(historical):
            self.filter_stats["volume_blocked"] += 1
            logger.info("%s -- volume exhaustion detected, skipping AI", symbol)
            return

        regime_context = self.regime.format_regime_context(symbol, regime_data)
        mtf_summary = self._build_mtf_summary(indicators_3m, indicators_15m, indicators_1h)
        full_context = regime_context + "\n\n" + mtf_summary if mtf_summary else regime_context

        # ── Pre-AI quality gate: skip DeepSeek for obvious no-trade situations ──
        rsi_3m = indicators_3m.get("rsi", 50)
        if rsi_3m > RSI_EXTREME_OB:
            logger.info("%s -- RSI overbought (%.0f > %d), skipping AI", symbol, rsi_3m, RSI_EXTREME_OB)
            return
        if rsi_3m < RSI_EXTREME_OS:
            logger.info("%s -- RSI oversold (%.0f < %d), skipping AI", symbol, rsi_3m, RSI_EXTREME_OS)
            return

        adx_15m = indicators_15m.get("adx", 20)
        adx_1h = indicators_1h.get("adx", 20)
        adx_3m = indicators_3m.get("adx", 20)
        if adx_15m < MIN_ADX_TRENDING and adx_1h < MIN_ADX_TRENDING and adx_3m < MIN_ADX_TRENDING:
            logger.info("%s -- all TFs ranging (3m=%.0f, 15m=%.0f, 1h=%.0f), skipping AI",
                        symbol, adx_3m, adx_15m, adx_1h)
            return

        # ── IMPROVEMENT #5: Pass recent bars for candle-history context ────────
        recent_bars = historical.tail(10) if len(historical) >= 10 else historical
        async with self._ai_sem:
            signal = await self.ai.get_trading_signal(symbol, market_data, indicators_3m,
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

        # ── Sector regime conflict penalty ─────────────────────────────────────
        sector_trend = (regime_data.get("sector") or {}).get("trend", "").upper()
        sector_name = regime_data.get("sector_name", "")
        if sector_trend:
            conflict = (sig_type == "SELL" and sector_trend == "BULLISH") or \
                       (sig_type == "BUY" and sector_trend == "BEARISH")
            if conflict:
                old_conf = confidence
                confidence -= SECTOR_REGIME_PENALTY
                reasoning += f" | SECTOR PENALTY: {sig_type} vs {sector_name}={sector_trend}, conf {old_conf}→{confidence}"
                logger.info("%s -- sector conflict penalty: %s vs %s=%s, confidence %d→%d",
                            symbol, sig_type, sector_name, sector_trend, old_conf, confidence)
                if confidence < self.risk.min_confidence:
                    logger.info("%s -- post-penalty confidence %d < min %d, skipping",
                                symbol, confidence, self.risk.min_confidence)
                    return

        # ── RSI Overbought/Oversold validation ───────────────────────────
        rsi_3m = indicators_3m.get("rsi", 50)
        if sig_type == "BUY" and rsi_3m >= RSI_OB_LIMIT:
            logger.info("%s BUY vetoed: RSI is overbought (%.2f >= %d)",
                        symbol, rsi_3m, RSI_OB_LIMIT)
            return
        elif sig_type == "SELL" and rsi_3m <= RSI_OS_LIMIT:
            logger.info("%s SELL vetoed: RSI is oversold (%.2f <= %d)",
                        symbol, rsi_3m, RSI_OS_LIMIT)
            return

        # ── Reversal check on entry ─────────────────────────────────────────
        rev = detect_reversals(historical, is_buy=(sig_type == "BUY"), indicators=indicators_3m)
        if rev.score >= 40:
            self.filter_stats["reversal_blocked"] += 1
            logger.info("%s entry vetoed: reversal score %d (>= 40)", symbol, rev.score)
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
        trailing_sl = ltp - (2.0 * atr_value) if sig_type == "BUY" else ltp + (2.0 * atr_value)

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
        mtf_15m = _tf_trend(indicators_15m)
        mtf_1h = _tf_trend(indicators_1h)

        await self.signal_log.log_signal(
            symbol=symbol, signal_type=tag, direction=sig_type,
            entry_price=ltp, quantity=quantity, stop_loss=sl_price,
            trailing_stop=trailing_sl, target=target_price, confidence=confidence,
            reasoning=reasoning, mode=mode,
            market_regime=nifty_regime.upper(), sector_regime=sector_regime.upper(),
            mtf_3m=mtf_3m, mtf_15m=mtf_15m, mtf_1h=mtf_1h,
        )

        self._notify_telegram(symbol, tag, sig_type, quantity, ltp, sl_price=sl_price, tp1_price=target_price, trailing_sl=trailing_sl)

        if self.dry_run:
            self.last_signal_time[symbol] = time.time()
            self.active_trades[symbol] = {
                "symbol": symbol, "entry_price": ltp, "quantity": quantity,
                "transaction_type": trans_type, "order_id": f"DRY-{symbol}-{int(time.time())}",
                "stop_loss_percent": sl_percent, "target_percent": target_percent,
                "entry_time": self._now_ist(),
                "trailing_sl": trailing_sl, "atr_value": atr_value,
            }
            logger.info("%s %s signal generated: %d shares @ %.2f | R:R=%.1f:1 (DRY-RUN)",
                        symbol, sig_type, quantity, ltp,
                        target_percent / sl_percent if sl_percent > 0 else 0)
            return

        async with self._dhan_sem:
            order = await asyncio.to_thread(self.dhan.place_super_order,
                security_id=security_id, transaction_type=trans_type,
                quantity=quantity, entry_price=ltp,
                sl_percent=sl_percent, target_percent=target_percent,
                symbol=symbol, atr_value=atr_value,
            )

        if order and order.get("status") == "success":
            self.last_signal_time[symbol] = time.time()
            self.active_trades[symbol] = {
                "symbol": symbol, "entry_price": ltp, "quantity": quantity,
                "transaction_type": trans_type, "order_id": order.get("data", {}).get("orderId"),
                "stop_loss_percent": sl_percent, "target_percent": target_percent,
                "entry_time": self._now_ist(),
                "trailing_sl": trailing_sl, "atr_value": atr_value,
            }
            logger.info("%s %s super order placed: %d shares @ %.2f (SL=%s)",
                        symbol, sig_type, quantity, ltp, sl_price)
            self.risk.record_trade()
            self._notify_telegram(symbol, "ORDER-PLACED", sig_type, quantity, ltp,
                                 sl_price=sl_price, tp1_price=target_price,
                                 trailing_sl=trailing_sl)
        else:
            remarks = order.get("remarks", "unknown") if order else "no response"
            logger.warning("%s %s super order FAILED: %s", symbol, sig_type, remarks)

    def _calc_pnl(self, trade, exit_price):
        is_buy = trade["transaction_type"] == self.dhan.dhan.BUY
        entry = trade["entry_price"]
        if is_buy:
            pnl = (exit_price - entry) * trade["quantity"]
            pnl_pct = (exit_price - entry) / entry * 100 if entry > 0 else 0
        else:
            pnl = (entry - exit_price) * trade["quantity"]
            pnl_pct = (entry - exit_price) / entry * 100 if entry > 0 else 0
        return pnl, pnl_pct

    async def _exit_position(self, symbol, reason="EXIT"):
        async with self._exit_lock:
            if symbol not in self.active_trades:
                return
            trade = self.active_trades.get(symbol)
            if not trade:
                return

            exit_trans = (self.dhan.dhan.SELL if trade["transaction_type"] == self.dhan.dhan.BUY
                          else self.dhan.dhan.BUY)
            security_id = self.dhan.security_ids.get(symbol)
            if not security_id:
                return

            async with self._dhan_sem:
                live = await asyncio.to_thread(self.dhan.fetch_live_data, security_id)
            exit_price = live.get("last_price") or trade["entry_price"]
            pnl, pnl_pct = self._calc_pnl(trade, exit_price)

            if not self.dry_run:
                # Cleanly close super order pending legs if this was a super order
                order_id = trade.get("order_id", "")
                if order_id and not order_id.startswith(("DRY-", "DHAN-")):
                    try:
                        logger.info("%s - Cancelling Super Order pending legs before exit...", symbol)
                        await asyncio.to_thread(self.dhan.dhan.cancel_super_order, order_id, "TARGET_LEG")
                        await asyncio.to_thread(self.dhan.dhan.cancel_super_order, order_id, "STOP_LOSS_LEG")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.warning("%s - Error cancelling super order legs: %s", symbol, e)

                async with self._dhan_sem:
                    order = await asyncio.to_thread(self.dhan.place_equity_order,
                        security_id=security_id, transaction_type=exit_trans,
                        quantity=trade["quantity"], product_type="INTRA",
                    )
                if not order:
                    return

            await self.signal_log.log_exit(symbol, exit_price, pnl, reason)
            self.risk.record_pnl(pnl)
            self._notify_telegram(symbol, reason, "EXIT", trade["quantity"],
                                  exit_price, pnl=pnl, pnl_pct=pnl_pct)
            logger.info("Closed %s: PnL=%.2f (%.2f%%) [%s]", symbol, pnl, pnl_pct, reason)
            del self.active_trades[symbol]

    async def _close_at_market_end(self):
        """Close all remaining positions at market end."""
        now = self._now_ist()
        close_cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
        if now < close_cutoff:
            return
        for symbol in list(self.active_trades.keys()):
            await self._exit_position(symbol, "MARKET-CLOSE")
        logger.info("Market closing - all positions closed")

    async def _monitor_positions(self):
        """Monitor open positions: live PnL, trailing stop, reversal detection.
        Uses Dhan API positions as source of truth in live mode.
        """
        if not self.active_trades:
            return

        dhan_positions = {}
        if not self.dry_run:
            async with self._dhan_sem:
                dhan_positions = await asyncio.to_thread(self.dhan.fetch_positions)
            # Remove trades no longer open on Dhan
            for symbol in list(self.active_trades.keys()):
                if symbol not in dhan_positions:
                    logger.info("%s no longer in Dhan positions — removing from tracking", symbol)
                    del self.active_trades[symbol]
            # Add trades found on Dhan but not tracked
            for symbol, pos in dhan_positions.items():
                if symbol not in self.active_trades:
                    logger.info("%s found in Dhan positions but untracked — adding", symbol)
                    self.active_trades[symbol] = {
                        "symbol": symbol, "entry_price": pos["entry_price"],
                        "quantity": pos["quantity"],
                        "transaction_type": pos["transaction_type"],
                        "order_id": f"DHAN-{symbol}",
                        "entry_time": self._now_ist(),
                        "trailing_sl": 0,
                        "atr_value": 1,
                    }

        for symbol in list(self.active_trades.keys()):
            try:
                dhan_pnl = dhan_positions.get(symbol, {}).get("unrealized_pnl") if dhan_positions else None
                await self._monitor_one_position(symbol, dhan_pnl=dhan_pnl)
            except Exception as e:
                logger.exception("Error monitoring %s: %s", symbol, e)

    async def _monitor_one_position(self, symbol, dhan_pnl=None):
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        security_id = self.dhan.security_ids.get(symbol)
        if not security_id:
            return

        async with self._dhan_sem:
            live = await asyncio.to_thread(self.dhan.fetch_live_data, security_id)
        current_price = live.get("last_price")
        if not current_price:
            return

        # Calculate live PnL (use Dhan PnL if available)
        if dhan_pnl is not None:
            pnl = dhan_pnl
            entry = trade["entry_price"]
            pnl_pct = (pnl / (entry * trade["quantity"]) * 100) if entry > 0 and trade["quantity"] > 0 else 0
        else:
            pnl, pnl_pct = self._calc_pnl(trade, current_price)

        logger.info("%s PnL=%.2f (%.2f%%) price=%.2f entry=%.2f",
                    symbol, pnl, pnl_pct, current_price, trade["entry_price"])

        is_buy = trade["transaction_type"] == self.dhan.dhan.BUY

        # Check trailing stop
        trailing_sl = trade.get("trailing_sl", 0)
        if trailing_sl > 0:
            hit = (is_buy and current_price <= trailing_sl) or (not is_buy and current_price >= trailing_sl)
            if hit:
                logger.info("%s trailing stop hit at %.2f (trail=%.2f)", symbol, current_price, trailing_sl)
                await self._exit_position(symbol, "TRAILING-SL")
                return

        # Fetch fresh 3m indicators for reversal check
        async with self._dhan_sem:
            historical = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "3minute", min_bars=5)
        if len(historical) < 5:
            return

        indicators = self.calculate_technical_indicators(historical)
        if not indicators:
            return

        # Compute reversal report using reversal detector
        reversal_report = detect_reversals(historical, is_buy, indicators)

        # Log caution state
        if reversal_report.recommendation == "⚠️ CAUTION":
            signals_desc = ", ".join(f"{s.name} ({s.description})" for s in reversal_report.signals)
            logger.warning("%s caution state (score=%d): %s", symbol, reversal_report.score, signals_desc)

        # Trigger exit on EXIT NOW
        if reversal_report.recommendation == "🚨 EXIT NOW":
            exit_reason = "REV-" + (reversal_report.signals[0].name.upper().replace(" ", "-") if reversal_report.signals else "EXIT")
            logger.info("%s reversal exit triggered (score=%d, PnL=%.2f%%): %s", 
                        symbol, reversal_report.score, pnl_pct, exit_reason)
            await self._exit_position(symbol, exit_reason)

    async def run(self, scan_interval=180, single_run=False):
        logger.info("Starting Intraday Stock Trading Bot")
        logger.info("Entry window: %d:%02d AM - %d:%02d PM IST | Max %d signals/stock/day | Scan every %ds",
                    FIRST_ENTRY_HOUR, FIRST_ENTRY_MIN,
                    LAST_ENTRY_HOUR, LAST_ENTRY_MIN,
                    MAX_SIGNALS_PER_STOCK_PER_DAY, scan_interval)
        while True:
            try:
                self._reset_daily_if_needed()
                if not self.is_market_hours() and not single_run:
                    logger.info("Outside market hours -- sleeping %ds", scan_interval)
                    await asyncio.sleep(scan_interval)
                    continue

                # ── IMPROVEMENT #11: Align scans with 3-min candle boundaries ──
                now = self._now_ist()
                seconds_into_candle = (now.minute % 3) * 60 + now.second
                if seconds_into_candle < 5 and not single_run:
                    # We're right at a candle boundary — wait a few seconds for data
                    await asyncio.sleep(5 - seconds_into_candle)

                # Clear caches at start of scan loop
                self.dhan.clear_historical_cache()
                self.dhan.clear_live_quotes_cache()

                # Pre-cache live quotes for all watchlist stocks at start of scan loop
                watchlist_sids = [self.dhan.security_ids[sym] for sym in self.watchlist if sym in self.dhan.security_ids]
                if watchlist_sids:
                    logger.info("Pre-caching live quotes for %d watchlist stocks...", len(watchlist_sids))
                    async with self._dhan_sem:
                        await asyncio.to_thread(self.dhan.cache_live_quotes, watchlist_sids)

                logger.info("Scanning %d stocks...", len(self.watchlist))
                tasks = [self.analyze_stock(s) for s in self.watchlist]
                await asyncio.gather(*tasks)
                await self._close_at_market_end()

                # Pre-cache live quotes for all active positions before monitoring
                active_sids = [self.dhan.security_ids[sym] for sym in self.active_trades.keys() if sym in self.dhan.security_ids]
                if active_sids:
                    logger.info("Pre-caching live quotes for %d active positions...", len(active_sids))
                    async with self._dhan_sem:
                        await asyncio.to_thread(self.dhan.cache_live_quotes, active_sids)

                await self._monitor_positions()
                
                if single_run:
                    logger.info("Single run completed.")
                    break
                    
                await asyncio.sleep(scan_interval)
            except Exception as e:
                logger.exception("Unexpected error in main loop: %s", e)
                if single_run:
                    break
                await asyncio.sleep(scan_interval)
