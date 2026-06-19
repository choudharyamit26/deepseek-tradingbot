import sys
import os
import csv
import time
import logging
import asyncio
from pathlib import Path
from datetime import time as dtime, datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parent.parent))
from stock_trading_bot import IntradayStockBot
from reversal_detector import detect_reversals
from analog_rag import AnalogRAG
from risk_controls import normalize_stop_loss_percent, stop_loss_floor_percent

from . import config as cfg
from .kronos_integration import KronosIntegration

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_ENTRY_START_T = dtime(cfg.FIRST_ENTRY_HOUR, cfg.FIRST_ENTRY_MIN)
_ENTRY_END_T = dtime(cfg.LAST_ENTRY_HOUR, cfg.LAST_ENTRY_MIN)
_MARKET_OPEN_T = dtime(9, 15)
_MARKET_CLOSE_T = dtime(15, 30)

RSI_EXTREME_OB = 78
RSI_EXTREME_OS = 22

class EnhancedIntradayBot(IntradayStockBot):
    def __init__(self, dhan_bot, ai_analyzer, risk_manager, kronos: KronosIntegration,
                 watchlist=None, send_telegram=None, format_signal_msg=None,
                 enable_telegram=False, dry_run=True):
        super().__init__(dhan_bot, ai_analyzer, risk_manager, watchlist,
                         send_telegram, format_signal_msg, enable_telegram, dry_run)
        self.kronos = kronos
        self.cfg = {
            "enabled": cfg.KRONOS_ENABLED,
            "pred_len": cfg.KRONOS_PRED_LEN,
            "lookback": cfg.KRONOS_LOOKBACK,
            "temperature": cfg.KRONOS_TEMPERATURE,
            "sample_count": cfg.KRONOS_SAMPLE_COUNT,
            "top_p": cfg.KRONOS_TOP_P,
            "confidence_weight": cfg.KRONOS_CONFIDENCE_WEIGHT,
            "penalty_conflict": cfg.KRONOS_PENALTY_CONFLICT,
            "bonus_align": cfg.KRONOS_BONUS_ALIGN,
            "exit_threshold": cfg.KRONOS_EXIT_THRESHOLD,
            "min_predicted_move": cfg.KRONOS_MIN_PREDICTED_MOVE,
            "trailing_sl_activation_pct": cfg.TRAILING_SL_ACTIVATION_PCT,
            "trailing_sl_distance_atr": cfg.TRAILING_SL_DISTANCE_ATR,
            "max_trade_duration_minutes": cfg.MAX_TRADE_DURATION_MINUTES,
            "market_open_skip_minutes": cfg.MARKET_OPEN_SKIP_MINUTES,
            "market_close_exit_minutes": cfg.MARKET_CLOSE_EXIT_MINUTES,
            "max_consecutive_losses": cfg.MAX_CONSECUTIVE_LOSSES,
            "partial_profit_pct": cfg.PARTIAL_PROFIT_PCT,
            "position_confidence_scalar": cfg.POSITION_CONFIDENCE_SCALAR,
        }
        self.kronos_cfg = {
            "model_name": cfg.KRONOS_MODEL,
            "tokenizer_name": cfg.KRONOS_TOKENIZER,
            "max_context": cfg.KRONOS_MAX_CONTEXT,
            "device": cfg.KRONOS_DEVICE,
            "pred_len": self.cfg["pred_len"],
            "lookback": self.cfg["lookback"],
            "temperature": self.cfg["temperature"],
            "sample_count": self.cfg["sample_count"],
            "top_p": self.cfg["top_p"],
            "penalty_conflict": self.cfg["penalty_conflict"],
            "bonus_align": self.cfg["bonus_align"],
            "exit_threshold": self.cfg["exit_threshold"],
        }
        self._last_signals: dict[str, dict] = {}   # {symbol: {signal, confidence, direction, timestamp, ...}}
        self._track_record_cache: str | None = None
        self._track_record_date: str | None = None
        self._track_record_lock = asyncio.Lock()
        self._consecutive_losses: int = 0
        self._last_exit_was_loss: bool = False
        self._partial_profit_booked: dict[str, bool] = {}
        # Feature 1: batch Kronos state (populated in _pre_scan_batch)
        self._batch_kronos_lock = asyncio.Lock()
        self._batch_kronos_hist: dict[str, object] = {}  # symbol -> 3m df
        # Feature 3: analog RAG
        self.rag = AnalogRAG()
        logger.info("AnalogRAG ready: %d stored setups", self.rag.count())
        # Feature 2: circuit breaker proximity tracking
        self._circuit_blocked: set[str] = set()

    # ── Feature 1: Batch Kronos prediction ────────────────────────────────────

    async def _pre_scan_batch(self):
        """
        Fetch 3m history for all watchlist stocks and run a single
        KronosPredictor.predict_batch() call instead of 191 separate ones.
        Results are stored in self.kronos._prediction_cache so that
        per-stock _analyze() calls hit the cache for free.
        """
        if not self.cfg["enabled"] or not self.kronos.ready:
            return

        # On CPU, batch=99 is slower than 99 concurrent batch=1 calls from _analyze().
        # Skip and let per-stock predict() run in parallel via asyncio.gather.
        if self.kronos.device == "cpu":
            logger.info("Kronos batch: CPU device — skipping batch, using per-stock inference")
            return

        now_ist = self._now_ist()
        if now_ist.weekday() >= 5 or not (_ENTRY_START_T <= now_ist.time() <= _ENTRY_END_T):
            return

        logger.info("Kronos batch: fetching 3m history for %d stocks...", len(self.watchlist))
        KRONOS_MIN_3M = 30

        async def _fetch_one(symbol):
            security_id = self.dhan.security_ids.get(symbol)
            if not security_id:
                return symbol, None
            try:
                async with self._dhan_sem:
                    df = await asyncio.to_thread(
                        self.dhan.get_historical_data, security_id, "3minute", KRONOS_MIN_3M
                    )
                if df is None or len(df) < KRONOS_MIN_3M:
                    return symbol, None
                return symbol, df
            except Exception as exc:
                logger.debug("Batch fetch failed for %s: %s", symbol, exc)
                return symbol, None

        results = await asyncio.gather(*[_fetch_one(s) for s in self.watchlist])
        stock_data = {sym: df for sym, df in results if df is not None}
        self._batch_kronos_hist = stock_data

        if not stock_data:
            logger.info("Kronos batch: no stocks with enough 3m bars yet")
            return

        logger.info("Kronos batch: running predict_batch() on %d stocks...", len(stock_data))
        await asyncio.to_thread(
            self.kronos.predict_batch_for_stocks, stock_data, KRONOS_MIN_3M
        )
        logger.info("Kronos batch: complete, %d predictions cached", len(stock_data))

    # ── Feature 2: Circuit breaker proximity check ─────────────────────────

    def _near_circuit_limit(self, symbol: str, historical: object, ltp: float) -> bool:
        """
        Return True if the stock has moved >= 8% intraday from today's open,
        meaning it may be approaching a 10% circuit limit and will be illiquid.
        Uses today's first 3m bar as a proxy for the opening price.
        """
        if historical is None or len(historical) < 1:
            return False
        try:
            today_open = float(historical["open"].iloc[0])
            if today_open <= 0:
                return False
            intraday_move_pct = abs(ltp - today_open) / today_open * 100
            CIRCUIT_PROXIMITY_THRESHOLD = 8.0  # warn if within 2% of 10% circuit
            if intraday_move_pct >= CIRCUIT_PROXIMITY_THRESHOLD:
                logger.info(
                    "%s near circuit limit: intraday move=%.2f%% from open=%.2f (ltp=%.2f)",
                    symbol, intraday_move_pct, today_open, ltp,
                )
                return True
        except Exception:
            pass
        return False

    async def _build_track_record(self) -> str:
        """Build a summary of recent closed trades for AI feedback.
        Loads last 5 trading sessions of signal CSVs and computes win/loss stats.
        Cached per day to avoid re-reading CSVs on every stock analysis."""
        today = datetime.now(IST).strftime("%Y-%m-%d")

        async with self._track_record_lock:
            if self._track_record_date == today and self._track_record_cache is not None:
                return self._track_record_cache

            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_logs")
            closed_trades = []

            def _read_csvs():
                trades = []
                for days_back in range(14):
                    d = (datetime.now(IST) - timedelta(days=days_back)).strftime("%Y-%m-%d")
                    csv_path = os.path.join(log_dir, f"signals_{d}.csv")
                    if not os.path.isfile(csv_path):
                        continue
                    try:
                        with open(csv_path, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                pnl_str = row.get("pnl", "").strip()
                                exit_str = row.get("exit_price", "").strip()
                                if pnl_str and exit_str:  # only closed trades
                                    conf_str = row.get("confidence", "").strip()
                                    trades.append({
                                        "symbol": row["symbol"],
                                        "direction": row.get("direction", ""),
                                        "confidence": int(conf_str) if conf_str else 0,
                                        "pnl": float(pnl_str),
                                        "entry": float(row.get("entry_price", 0) or 0),
                                        "exit": float(exit_str),
                                        "date": d,
                                        "market_regime": row.get("market_regime", ""),
                                    })
                    except Exception as e:
                        logger.warning("Track record: error reading %s: %s", csv_path, e)
                return trades

            closed_trades = await asyncio.to_thread(_read_csvs)

            if len(closed_trades) < 3:
                self._track_record_cache = ""
                self._track_record_date = today
                return ""

            # Compute statistics
            winners = [t for t in closed_trades if t["pnl"] > 0]
            losers = [t for t in closed_trades if t["pnl"] <= 0]
            total = len(closed_trades)
            win_rate = len(winners) / total * 100
            total_pnl = sum(t["pnl"] for t in closed_trades)

            avg_conf_winners = sum(t["confidence"] for t in winners) / len(winners) if winners else 0
            avg_conf_losers = sum(t["confidence"] for t in losers) / len(losers) if losers else 0
            avg_pnl_winners = sum(t["pnl"] for t in winners) / len(winners) if winners else 0
            avg_pnl_losers = sum(t["pnl"] for t in losers) / len(losers) if losers else 0

            # Find worst high-confidence losers (confidence >= 80 but lost money)
            high_conf_losers = sorted(
                [t for t in losers if t["confidence"] >= 80],
                key=lambda t: t["pnl"]
            )[:5]  # worst 5

            # Per-regime and per-direction stats
            regime_stats = {}
            dir_stats = {}
            for t in closed_trades:
                r = t.get("market_regime") or "UNKNOWN"
                d = t["direction"]
                for bucket, key in [(regime_stats, r), (dir_stats, d)]:
                    bucket.setdefault(key, {"count": 0, "pnl": 0.0, "winners": 0})
                    bucket[key]["count"] += 1
                    bucket[key]["pnl"] += t["pnl"]
                    if t["pnl"] > 0:
                        bucket[key]["winners"] += 1

            trading_days = sorted(set(t["date"] for t in closed_trades))

            lines = [
                f"YOU TRADED {total} TIMES IN LAST {len(trading_days)} SESSIONS. HERE IS TRUTH:",
                f"  Win rate: {len(winners)}/{total} = {len(winners)/total*100:.0f}%",
                f"  Total P&L: {total_pnl:+.2f}",
                f"  Avg confidence on WINNERS: {avg_conf_winners:.0f} (avg P&L: {avg_pnl_winners:+.2f})",
                f"  Avg confidence on LOSERS:  {avg_conf_losers:.0f} (avg P&L: {avg_pnl_losers:+.2f})",
            ]

            # CRITICAL: Detect when confidence is noise (winner/loser avg within 5 pts)
            conf_gap = abs(avg_conf_winners - avg_conf_losers)
            if conf_gap <= 5:
                lines.append(
                    f"\n  *** YOUR CONFIDENCE SCORE IS BROKEN ***"
                    f"\n  Winners avg conf = {avg_conf_winners:.0f}, Losers avg conf = {avg_conf_losers:.0f}."
                    f"\n  Gap is only {conf_gap:.0f} points. This means your confidence predicts NOTHING."
                    f"\n  A coin flip would be just as good. STOP giving everything 80-85."
                    f"\n  If you are not VERY sure, give 65-75. Save 85+ for PERFECT setups only."
                )

            # Direction breakdown
            for d, s in sorted(dir_stats.items()):
                wr = s["winners"] / s["count"] * 100 if s["count"] else 0
                pnl = s["pnl"]
                if wr < 45 or pnl < -10:
                    lines.append(f"  {d} trades: {s['count']}x, WR={wr:.0f}%, P&L={pnl:+.2f} <-- BAD. Think twice before {d}ing.")
                else:
                    lines.append(f"  {d} trades: {s['count']}x, WR={wr:.0f}%, P&L={pnl:+.2f}")

            # Regime breakdown
            for r, s in sorted(regime_stats.items()):
                wr = s["winners"] / s["count"] * 100 if s["count"] else 0
                pnl = s["pnl"]
                if pnl < -20:
                    lines.append(f"  In {r} regime: {s['count']}x, WR={wr:.0f}%, P&L={pnl:+.2f} <-- LOSING MONEY in this regime. Be extra careful.")
                else:
                    lines.append(f"  In {r} regime: {s['count']}x, WR={wr:.0f}%, P&L={pnl:+.2f}")

            if high_conf_losers:
                lines.append(f"\n  LOOK AT THESE. YOU SAID 80+ CONFIDENCE AND LOST MONEY:")
                for t in high_conf_losers:
                    lines.append(
                        f"    {t['symbol']} {t['direction']} conf={t['confidence']} "
                        f"P&L={t['pnl']:+.2f} regime={t['market_regime']} ({t['date']})"
                    )
                lines.append(
                    f"  {len(high_conf_losers)} of {len(losers)} losers had confidence >= 80."
                    f" You are OVERCONFIDENT. Lower your scores or keep losing money."
                )

            record = "\n        ".join(lines)
            self._track_record_cache = record
            self._track_record_date = today
            logger.info("Track record loaded: %d trades, %.0f%% win rate, P&L=%+.2f",
                        total, win_rate, total_pnl)
            return record

    def _compute_score_matrix(self, indicators: dict, regime_data: dict,
                               indicators_15m: dict, indicators_1h: dict,
                               kronos_conf: dict | None = None) -> tuple[int, str]:
        """Compute confidence ceiling from raw indicators. Returns (score, breakdown_str).

        This is the AUTHORITATIVE confidence score. The AI's self-assessed score
        will be capped at this value. No negotiation, no rationalization."""
        score = 100
        penalties = []
        bonuses = []

        # ── Extract raw values ───────────────────────────────────────────────
        adx = indicators.get("adx", 20)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        rsi = indicators.get("rsi", 50)
        mfi = indicators.get("mfi", 50)
        vwap = indicators.get("vwap", 0)
        close = indicators.get("close", 0)
        sma20 = indicators.get("sma_20", close)

        # MTF trends
        # 15m uses EMA-9 to match the hard MTF veto (_validate_mtf_alignment._trend_15m)
        # and the MTF summary shown to the AI; 3m/1h use SMA-20.
        close_15m = indicators_15m.get("close", 0) if indicators_15m else 0
        ema9_15m = indicators_15m.get("ema_9", close_15m) if indicators_15m else close_15m
        trend_15m = "BULLISH" if close_15m > ema9_15m else "BEARISH" if close_15m else "NEUTRAL"

        close_1h = indicators_1h.get("close", 0) if indicators_1h else 0
        sma20_1h = indicators_1h.get("sma_20", close_1h) if indicators_1h else close_1h
        trend_1h = "BULLISH" if close_1h > sma20_1h else "BEARISH" if close_1h else "NEUTRAL"

        trend_3m = "BULLISH" if close > sma20 else "BEARISH"
        price_above_vwap = close > vwap if vwap > 0 else True

        # Nifty/sector regime
        nifty_data_m = regime_data.get("nifty", {}) if regime_data else {}
        nifty_trend = nifty_data_m.get("trend", "neutral")
        nifty_intraday = nifty_data_m.get("intraday_chg_pct", 0)
        nifty_session = nifty_data_m.get("session_trend", "neutral")
        sector_data = regime_data.get("sector") if regime_data else None
        sector_trend = sector_data.get("trend", "neutral") if sector_data else "neutral"

        # ── CRITICAL: instant HOLD ───────────────────────────────────────────
        if adx < 18:
            return 0, f"HOLD: ADX={adx:.1f} < 18 (ranging market)"
        if volume_ratio < 0.3:
            return 0, f"HOLD: Volume ratio={volume_ratio:.2f} < 0.3 (dead market)"

        # ── MAJOR PENALTIES ──────────────────────────────────────────────────
        if 0.3 <= volume_ratio < 0.5:
            score -= 12
            penalties.append(f"vol={volume_ratio:.2f} very weak: -12")
        elif 0.5 <= volume_ratio < 0.8:
            score -= 8
            penalties.append(f"vol={volume_ratio:.2f} below avg: -8")

        # Kronos conflict
        if kronos_conf and kronos_conf.get("conflict"):
            score -= 8
            penalties.append("Kronos CONFLICT: -8")

        # 15m disagrees with 3m
        if trend_15m != "NEUTRAL" and trend_15m != trend_3m:
            score -= 7
            penalties.append(f"15m={trend_15m} vs 3m={trend_3m}: -7")

        # 1h disagrees with 3m. Softened -5 -> -3: the 1h trend only became a
        # real (non-NEUTRAL) signal after the MIN_BARS_1H / multi-day-fetch fix
        # (AC.6), so it has zero validated performance history. The only TF-
        # alignment evidence we have (15m) shows alignment is non-predictive /
        # slightly negative, so a freshly-live signal should nudge, not dominate.
        if trend_1h != "NEUTRAL" and trend_1h != trend_3m:
            score -= 3
            penalties.append(f"1h={trend_1h} vs 3m={trend_3m}: -3")

        # MFI extremes with potential stalling
        if mfi > 80:
            score -= 12
            penalties.append(f"MFI={mfi:.0f} overbought: -12")
        elif mfi < 20:
            score -= 12
            penalties.append(f"MFI={mfi:.0f} oversold: -12")

        # ── MINOR PENALTIES ──────────────────────────────────────────────────
        if 18 <= adx < 22:
            score -= 5
            penalties.append(f"ADX={adx:.0f} borderline: -5")

        if 0.8 <= volume_ratio < 1.0:
            score -= 3
            penalties.append(f"vol={volume_ratio:.2f} slightly low: -3")

        if kronos_conf and kronos_conf.get("pred_range_pct", 1.0) < 0.2:
            score -= 3
            penalties.append(f"Kronos range={kronos_conf.get('pred_range_pct', 0):.2f}% low conviction: -3")

        # ── BONUSES ──────────────────────────────────────────────────────────
        bonus_total = 0
        if volume_ratio >= 2.0:
            bonus_total += 3
            bonuses.append(f"vol={volume_ratio:.2f} exceptional: +3")
        elif volume_ratio >= 1.5:
            bonus_total += 2
            bonuses.append(f"vol={volume_ratio:.2f} strong: +2")

        # All-TF-aligned bonus, softened +3 -> +1. This bonus was effectively
        # never granted before the 1h fix (1h was always NEUTRAL). Since 15m
        # alignment was shown non-predictive/slightly negative, granting a large
        # bonus would over-promote the exact trades that did slightly worse
        # (and, via position_confidence_scalar at conf>=85, oversize them).
        # Keep it a small nudge until live 1h-aligned trades justify more.
        if (trend_15m == trend_3m and trend_1h == trend_3m and
                trend_15m != "NEUTRAL" and trend_1h != "NEUTRAL"):
            bonus_total += 1
            bonuses.append(f"All TFs aligned ({trend_3m}): +1")

        if kronos_conf and not kronos_conf.get("conflict") and kronos_conf.get("pred_range_pct", 0) > 0.5:
            bonus_total += 2
            bonuses.append(f"Kronos strongly aligned: +2")

        # Nifty intraday momentum bonus: if session trend aligns with 3m direction
        if trend_3m == "BULLISH" and nifty_intraday >= 1.0:
            bonus_total += 3
            bonuses.append(f"Nifty intraday {nifty_intraday:+.2f}% aligns with BUY: +3")
        elif trend_3m == "BEARISH" and nifty_intraday <= -1.0:
            bonus_total += 3
            bonuses.append(f"Nifty intraday {nifty_intraday:+.2f}% aligns with SELL: +3")
        elif trend_3m == "BULLISH" and nifty_intraday >= 0.5:
            bonus_total += 1
            bonuses.append(f"Nifty intraday {nifty_intraday:+.2f}% supports BUY: +1")
        elif trend_3m == "BEARISH" and nifty_intraday <= -0.5:
            bonus_total += 1
            bonuses.append(f"Nifty intraday {nifty_intraday:+.2f}% supports SELL: +1")

        score += min(bonus_total, 10)  # cap bonuses at +10 (was +8, raised for intraday bonus)

        # ── Build breakdown string ───────────────────────────────────────────
        parts = [f"START=100"]
        if penalties:
            parts.extend(penalties)
        if bonuses:
            parts.extend(bonuses)
        parts.append(f"FINAL={score}")

        breakdown = " | ".join(parts)

        return max(score, 0), breakdown

    async def _analyze(self, symbol: str):
        self.filter_stats["total_scans"] += 1
        if not self._atr_prewarmed:
            self._atr_prewarmed = True
            for sym, sid in self.dhan.security_ids.items():
                if sym in self.watchlist:
                    asyncio.create_task(self._build_atr_profile(sym, sid))
        self._reset_daily_if_needed()
        logger.info("Enhanced: Analyzing %s...", symbol)

        security_id = self.dhan.security_ids.get(symbol)
        if not security_id:
            logger.warning("Security ID not found for %s", symbol)
            return

        now_ist = self._now_ist()
        if now_ist.weekday() >= 5 or not (_ENTRY_START_T <= now_ist.time() <= _ENTRY_END_T):
            logger.debug("%s -- outside entry window", symbol)
            return

        # ── Market open skip: avoid opening volatility ─────────────────────
        skip_min = self.cfg.get("market_open_skip_minutes", 0)
        if skip_min > 0:
            open_skip_cutoff = dtime(9, 15 + skip_min)
            if now_ist.time() < open_skip_cutoff:
                logger.info("%s -- market open skip active (until %02d:%02d)", symbol, open_skip_cutoff.hour, open_skip_cutoff.minute)
                return

        # ── Late exit cutoff: stop new entries near market close ──────────
        close_exit_min = self.cfg.get("market_close_exit_minutes", 15)
        if close_exit_min > 0:
            close_cutoff = dtime(15, 30 - close_exit_min)
            if now_ist.time() >= close_cutoff:
                logger.info("%s -- market close cutoff active (after %02d:%02d)", symbol, close_cutoff.hour, close_cutoff.minute)
                return

        # ── Early cap pre-filter (approximate — avoids wasted API calls) ────
        # NOTE: These are non-atomic reads, so concurrent coroutines may slip through.
        # The authoritative gate is reserve_daily_slot() called later before order placement.
        if not self.signal_log.can_trade(symbol, cfg.MAX_SIGNALS_PER_STOCK_PER_DAY):
            logger.info("%s -- daily signal limit", symbol)
            return
        if self.signal_log.get_total_daily_count() >= cfg.MAX_DAILY_SIGNALS:
            logger.info("%s -- global daily signal cap (%d) reached", symbol, cfg.MAX_DAILY_SIGNALS)
            return

        last_time = self.last_signal_time.get(symbol, 0)
        if time.time() - last_time < self.cooldown_seconds:
            return

        async with self._dhan_sem:
            historical = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "3minute", self.MIN_BARS_3M_WARMUP)
        # Holiday/empty guard stays on the small floor: a symbol with a handful
        # of real bars is tradeable; MIN_BARS_3M_WARMUP only drives how much
        # history the fetch backfills for indicator warmup (esp. ADX-14).
        if len(historical) < self.MIN_BARS:
            logger.info("%s -- insufficient 3m bars (%d) - possible holiday", symbol, len(historical))
            return

        logger.debug("%s 3m data: %d bars (%s -> %s)", symbol,
                     len(historical), historical.index[0], historical.index[-1])

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

        # Off the event loop: get_regime makes blocking HTTP calls on cache miss
        async with self._dhan_sem:
            regime_data = await asyncio.to_thread(self.regime.get_regime, symbol)

        passed, reason = self._passes_prefilter(indicators_3m, regime_data)
        if not passed:
            logger.info("%s pre-filter: %s", symbol, reason)
            return

        async with self._dhan_sem:
            historical_15m = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "15minute", self.MIN_BARS_15M)
        indicators_15m = self.calculate_technical_indicators(historical_15m) if len(historical_15m) >= self.MIN_BARS_15M else {}

        async with self._dhan_sem:
            historical_1h = await asyncio.to_thread(self.dhan.get_historical_data, security_id, "60minute", self.MIN_BARS_1H)
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

        # ── Feature 2: Circuit breaker proximity check ──────────────────────
        if self._near_circuit_limit(symbol, historical, ltp):
            self.filter_stats["atr_blocked"] += 1  # reuse counter for blocked stocks
            self._circuit_blocked.add(symbol)
            logger.info("%s -- near circuit limit, skipping (illiquid/gap-prone)", symbol)
            return
        self._circuit_blocked.discard(symbol)

        # ── Volume exhaustion check ─────────────────────────────────────────
        if self._check_volume_exhaustion(historical):
            self.filter_stats["volume_blocked"] += 1
            logger.info("%s -- volume exhaustion detected, skipping AI", symbol)
            return

        regime_context = self.regime.format_regime_context(symbol, regime_data)
        mtf_summary = self._build_mtf_summary(indicators_3m, indicators_15m, indicators_1h)
        full_context = regime_context + "\n\n" + mtf_summary if mtf_summary else regime_context

        # ── Kronos prediction (injected into AI context) ────────────────────
        kronos_ratio = 1.0
        kronos_conf = None
        kronos_conf_pred = None
        if self.cfg["enabled"] and self.kronos.ready:
            try:
                # Primary: 3-min bars (same timeframe as bot decisions).
                # Kronos will predict next 10 x 3-min = 30 min ahead — directly
                # relevant to entry/exit timing.
                # Fallback: multi-day 15m only for early-morning sessions (<30 3m bars).
                KRONOS_MIN_3M  = 30   # 30 x 3-min = 90 min context; available after ~10:45
                KRONOS_MIN_15M = 30   # 30 x 15-min = 7.5h context; always available via multi-day fetch

                if len(historical) >= KRONOS_MIN_3M:
                    kronos_input = historical
                    logger.debug("%s Kronos using %d 3-min bars (primary)", symbol, len(historical))
                else:
                    # Early-morning: not enough 3m bars yet — fall back to 15m multi-day
                    async with self._dhan_sem:
                        kronos_15m = await asyncio.to_thread(
                            self.dhan.get_kronos_history, security_id, "15minute", 7
                        )
                    if len(kronos_15m) >= KRONOS_MIN_15M:
                        kronos_input = kronos_15m
                        logger.debug("%s Kronos fallback: %d 15-min bars (3m=%d, need %d)",
                                     symbol, len(kronos_15m), len(historical), KRONOS_MIN_3M)
                    else:
                        kronos_input = None
                        logger.info("%s Kronos skipped: 3m=%d (need %d), 15m=%d (need %d)",
                                    symbol, len(historical), KRONOS_MIN_3M,
                                    len(kronos_15m), KRONOS_MIN_15M)

                if kronos_input is not None:
                    pred_df = await asyncio.to_thread(self.kronos.predict, kronos_input, symbol=symbol)
                    if pred_df is not None:
                        kronos_conf_pred = {"pred_df": pred_df, "historical_15m": kronos_input}
                        kronos_section = self.kronos.build_prompt_section(pred_df, ltp)
                        full_context += (
                            "\n\n" + kronos_section +
                            "\n\nIMPORTANT: The Kronos forecast above is a supplementary signal."
                            "\nFactor it into your decision but do NOT follow it blindly."
                            "\nYour primary technical rules (VWAP, RSI, ADX, volume, MTF) still apply."
                        )
                        logger.debug("%s Kronos prompt injected into AI context", symbol)
            except Exception as e:
                logger.warning("%s Kronos error: %s", symbol, e)

        recent_bars = historical.tail(10) if len(historical) >= 10 else historical
        track_record = await self._build_track_record()

        # Pre-compute scoring matrix ceiling based on 3m trend direction
        trend_3m = "BULLISH" if indicators_3m.get("close", 0) > indicators_3m.get("sma_20", 0) else "BEARISH"
        direction_3m = "BUY" if trend_3m == "BULLISH" else "SELL"
        pre_kronos_conf = None
        if kronos_conf_pred:
            try:
                pred_df = kronos_conf_pred.get("pred_df")
                hist_15m = kronos_conf_pred.get("historical_15m")
                pre_kronos_conf = self.kronos.compute_confirmation(
                    direction_3m, pred_df, ltp, historical_df=hist_15m,
                )
                pre_kronos_conf["pred_df"] = pred_df
            except Exception as e:
                logger.warning("%s Pre-AI Kronos conf error: %s", symbol, e)

        matrix_score, matrix_breakdown = self._compute_score_matrix(
            indicators_3m, regime_data, indicators_15m, indicators_1h, pre_kronos_conf
        )
        logger.info("%s Pre-computed scoring matrix ceiling: %d (%s)", symbol, matrix_score, matrix_breakdown)

        # ── Matrix gate: skip DeepSeek if pre-computed ceiling < MIN_CONFIDENCE ──
        if matrix_score < cfg.MIN_CONFIDENCE:
            logger.info("%s matrix ceiling %d < MIN_CONFIDENCE %d, skipping AI", symbol, matrix_score, cfg.MIN_CONFIDENCE)
            return

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
        if adx_15m < cfg.MIN_ADX_TRENDING and adx_1h < cfg.MIN_ADX_TRENDING and adx_3m < cfg.MIN_ADX_TRENDING:
            logger.info("%s -- all TFs ranging (3m=%.0f, 15m=%.0f, 1h=%.0f), skipping AI",
                        symbol, adx_3m, adx_15m, adx_1h)
            return

        # ── Feature 3: RAG/analog context ───────────────────────────────────
        nifty_trend_for_rag = (regime_data.get("nifty", {}) or {}).get("trend", "") if regime_data else ""
        regime_str_for_rag = (regime_data.get("nifty", {}) or {}).get("trend", "") if regime_data else ""
        try:
            atr_value_raw = indicators_3m.get("atr", 1)
            atr_pct_for_rag = (float(atr_value_raw) / ltp * 100) if ltp > 0 and atr_value_raw else 0.5
            indicators_for_rag = dict(indicators_3m)
            indicators_for_rag["atr_pct"] = atr_pct_for_rag
            analog_context = self.rag.query_similar(
                indicators_for_rag, pre_kronos_conf,
                nifty_trend_for_rag, regime_str_for_rag,
                signal_type=direction_3m, n=5,
            )
            if analog_context:
                full_context += "\n\n" + analog_context
                logger.debug("%s RAG analog context injected", symbol)
        except Exception as exc:
            logger.debug("%s RAG query failed: %s", symbol, exc)

        async with self._ai_sem:
            signal = await self.ai.get_trading_signal(
                symbol, market_data, indicators_3m,
                full_context, recent_bars=recent_bars,
                track_record=track_record if track_record else None,
                matrix_score=matrix_score,
                matrix_breakdown=matrix_breakdown,
            )
        logger.info("%s AI: %s (conf=%s)", symbol,
                    signal.get("signal", "?"), signal.get("confidence", "?"))

        current_trade = self.active_trades.get(symbol)
        sig_type = signal.get("signal", "HOLD")
        confidence = signal.get("confidence", 0)
        reasoning = signal.get("reasoning", "")

        nifty_data = regime_data.get("nifty", {}) if regime_data else {}
        nifty_trend = nifty_data.get("trend", "neutral")
        nifty_intraday_chg = nifty_data.get("intraday_chg_pct", 0)
        nifty_session_trend = nifty_data.get("session_trend", "neutral")
        sector_data = regime_data.get("sector") if regime_data else None
        sector_trend = sector_data.get("trend", "neutral") if sector_data else "neutral"

        # Programmatically cap trade confidence at the pre-computed ceiling.
        # The daily-SMA-based regime penalty (-6) is scaled linearly by how much
        # the Nifty has moved intraday (vs yesterday's close):
        #   intraday_chg = 0%    → full -6 penalty
        #   intraday_chg = +0.5% → -3 penalty
        #   intraday_chg = +1.0% → 0 penalty (fully waived)
        #   intraday_chg > +1.0% → 0 penalty (capped, doesn't become a bonus)
        # INTRADAY_SCALE_PCT is the move at which the penalty fully disappears.
        # Symmetric logic applies for bullish daily trend vs SELL signals.
        INTRADAY_SCALE_PCT = 1.0
        if sig_type in ("BUY", "SELL"):
            if nifty_trend == "bearish" and sig_type == "BUY":
                scale = min(max(nifty_intraday_chg, 0.0), INTRADAY_SCALE_PCT) / INTRADAY_SCALE_PCT
                nifty_pen = -round(6 * (1 - scale))
                if nifty_pen < 0:
                    confidence += nifty_pen
                    reasoning += (f" | Nifty bearish (daily) penalty scaled by intraday {nifty_intraday_chg:+.2f}%"
                                  f" ({nifty_pen:+d})")
                else:
                    reasoning += (f" | Nifty bearish (daily) but intraday {nifty_intraday_chg:+.2f}%"
                                  f" >= {INTRADAY_SCALE_PCT:.1f}% — BUY penalty fully waived")
            elif nifty_trend == "bullish" and sig_type == "SELL":
                scale = min(max(-nifty_intraday_chg, 0.0), INTRADAY_SCALE_PCT) / INTRADAY_SCALE_PCT
                nifty_pen = -round(6 * (1 - scale))
                if nifty_pen < 0:
                    confidence += nifty_pen
                    reasoning += (f" | Nifty bullish (daily) penalty scaled by intraday {nifty_intraday_chg:+.2f}%"
                                  f" ({nifty_pen:+d})")
                else:
                    reasoning += (f" | Nifty bullish (daily) but intraday {nifty_intraday_chg:+.2f}%"
                                  f" <= -{INTRADAY_SCALE_PCT:.1f}% — SELL penalty fully waived")

            if sector_trend == "bearish" and sig_type == "BUY":
                confidence -= 4
                reasoning += " | Sector bearish vs BUY penalty (-4)"
            elif sector_trend == "bullish" and sig_type == "SELL":
                confidence -= 4
                reasoning += " | Sector bullish vs SELL penalty (-4)"

            if confidence > matrix_score:
                logger.info("%s AI confidence %d capped by matrix ceiling %d (%s)",
                            symbol, confidence, matrix_score, matrix_breakdown)
                confidence = matrix_score
            signal["confidence"] = confidence

        # ── HARD GATE: BUY only in a confirmed bullish regime ──────────────
        # Empirical: across 106 closed trades BUY is structurally unprofitable
        # — 14 trades, 28.6% win rate, payoff 0.42, accounting for -87 of the
        # -123 total pnl, and it loses in every Nifty trend bucket. Permit a
        # BUY only when BOTH the Nifty and the stock's sector are bullish;
        # block every other BUY outright (a hard gate, not a soft penalty).
        #
        # Intraday override (symmetric with the SELL gate below): "bullish" is
        # the *daily* (10-day-SMA) trend, so a day that is strongly green
        # intraday can still read daily-NON-bullish (e.g. Nifty up sharply but
        # still below its 10-day SMA after a downtrend). Treat the Nifty leg as
        # satisfied when the daily trend is bullish OR the *session* is bullish
        # (intraday >= +0.5%, see regime_filter._calc_regime) — the intraday
        # tape pointing the BUY's way. The sector leg stays daily-only (sector
        # session_trend isn't tracked, and the AC.2 sector-tailwind requirement
        # is deliberately strict). session_trend defaults to "neutral" on a data
        # gap, preserving the conservative (gated) default. NOTE BUY is the
        # weakest side empirically, so this is the riskier of the two overrides
        # — flagged for the reflection agent to validate.
        nifty_ok = nifty_trend == "bullish" or nifty_session_trend == "bullish"
        if sig_type == "BUY" and not (nifty_ok and sector_trend == "bullish"):
            logger.warning(
                "%s BUY HARD-GATED: requires nifty(daily|session) AND sector bullish "
                "(nifty=%s, intraday=%+.2f%%, session=%s, sector=%s, conf=%d)",
                symbol, nifty_trend, nifty_intraday_chg, nifty_session_trend, sector_trend, confidence,
            )
            return

        # ── HARD GATE: block counter-trend SELLs (SELL into a bullish Nifty) ──
        # Empirical: SELL while the Nifty is bullish lost -19.55 over 9 trades
        # (payoff 0.44). NOTE the gate is Nifty-only — it deliberately does NOT
        # look at sector: SELLs while the *sector* is bullish are the strategy's
        # best edge (+60.25 over 10 trades, payoff 3.73, mean-reversion shorts
        # on extended sector names), so gating on sector would destroy profit.
        #
        # Intraday override: "bullish" here is the *daily* (10-day-SMA) trend,
        # so a session that is sharply red intraday can still read daily-bullish
        # (e.g. Nifty down 200 pts but still above its 10-day SMA). Selling into
        # a Nifty whose *session* is already bearish (intraday <= -0.5%, see
        # regime_filter._calc_regime) is not "selling into strength", so lift the
        # gate there. session_trend defaults to "neutral" when no live price is
        # available, so the conservative (gated) behaviour is preserved on a data
        # gap. Flagged for the reflection agent to validate once such trades
        # accumulate.
        if (sig_type == "SELL" and nifty_trend == "bullish"
                and nifty_session_trend != "bearish"):
            logger.warning(
                "%s SELL HARD-GATED: counter-trend (nifty bullish daily, intraday=%+.2f%%, session=%s, sector=%s, conf=%d)",
                symbol, nifty_intraday_chg, nifty_session_trend, sector_trend, confidence,
            )
            return

        # ── Kronos: position scaler (never blocks, only adjusts size) ──────
        kronos_ratio = 1.0
        kronos_conf = pre_kronos_conf
        if kronos_conf_pred and sig_type in ("BUY", "SELL"):
            try:
                pred_df = kronos_conf_pred.get("pred_df")
                hist_15m = kronos_conf_pred.get("historical_15m")
                if sig_type != direction_3m:
                    kronos_conf = self.kronos.compute_confirmation(
                        sig_type, pred_df, ltp, historical_df=hist_15m,
                    )
                    kronos_conf["pred_df"] = pred_df
                kronos_ratio = kronos_conf["adjustment"]
                logger.info(
                    "%s Kronos: %s (mag=%.3f, ratio=%.2f, conf=%d, range=%.2f%%)",
                    symbol, "ALIGN" if not kronos_conf["conflict"] else "CONFLICT",
                    kronos_conf.get("magnitude", 0), kronos_ratio, confidence,
                    kronos_conf.get("pred_range_pct", 0),
                )
            except Exception as e:
                logger.warning("%s Kronos conf error: %s", symbol, e)
                kronos_conf = None

        if sig_type == "EXIT" and current_trade:
            await self._exit_position(symbol)
            return

        if sig_type not in ("BUY", "SELL") or confidence < cfg.MIN_CONFIDENCE:
            if sig_type in ("BUY", "SELL"):
                logger.warning(
                    "%s %s REGIME-KILLED: AI conf=%d → after regime penalty → %d < %d "
                    "(nifty=%s intraday=%+.2f%%, sector=%s)",
                    symbol, sig_type, signal.get("confidence", 0), confidence, cfg.MIN_CONFIDENCE,
                    nifty_trend, nifty_intraday_chg, sector_trend,
                )
            return

        # ── Same-direction deduplication ─────────────────────────────────────
        last_sig = self._last_signals.get(symbol)
        if last_sig and last_sig.get("direction") == sig_type:
            elapsed = time.time() - last_sig.get("timestamp", 0)
            if elapsed < cfg.SAME_DIRECTION_COOLDOWN:
                logger.info("%s same-direction repeat (%s) blocked — %ds since last, need %ds",
                            symbol, sig_type, int(elapsed), cfg.SAME_DIRECTION_COOLDOWN)
                return

        mtf_ok, mtf_reason = self._validate_mtf_alignment(sig_type, indicators_3m, indicators_15m, indicators_1h)
        if not mtf_ok:
            logger.info("%s MTF veto: %s", symbol, mtf_reason)
            return

        # ── RSI Overbought/Oversold validation ───────────────────────────
        rsi_3m = indicators_3m.get("rsi", 50)
        if sig_type == "BUY" and rsi_3m >= cfg.RSI_OB_LIMIT:
            logger.info("%s BUY vetoed: RSI is overbought (%.2f >= %d)",
                        symbol, rsi_3m, cfg.RSI_OB_LIMIT)
            return
        elif sig_type == "SELL" and rsi_3m <= cfg.RSI_OS_LIMIT:
            logger.info("%s SELL vetoed: RSI is oversold (%.2f <= %d)",
                        symbol, rsi_3m, cfg.RSI_OS_LIMIT)
            return

        # ── RSI quality gate: don't short into deep oversold ─────────────────
        # Empirical (106 trades): SELL with RSI<35 had payoff 0.58 (chasing the
        # bottom into snap-back risk); the RSI 35-45 zone was the only profitable
        # bucket (payoff 1.41, +22.67). Block shorts below the floor.
        min_rsi_short = getattr(cfg, "MIN_RSI_FOR_SHORT", 0)
        if sig_type == "SELL" and min_rsi_short and rsi_3m < min_rsi_short:
            logger.info("%s SELL RSI-GATED: RSI %.2f < %d (deep-oversold short, poor-payoff zone)",
                        symbol, rsi_3m, min_rsi_short)
            return

        # ── Reversal check on entry ─────────────────────────────────────────
        rev = detect_reversals(historical, is_buy=(sig_type == "BUY"), indicators=indicators_3m)
        if rev.score >= 40:
            self.filter_stats["reversal_blocked"] += 1
            logger.info("%s entry vetoed: reversal score %d (>= 40)", symbol, rev.score)
            return

        if not self.risk.check_daily_trade_limit() or not self.risk.check_daily_loss_limit():
            logger.info("Daily limit hit")
            return

        # ── Consecutive losses circuit breaker ──────────────────────────────
        max_consec = self.cfg.get("max_consecutive_losses", 3)
        if max_consec > 0 and self._consecutive_losses >= max_consec:
            logger.info("%s consecutive loss breaker hit (%d/%d), skipping", symbol, self._consecutive_losses, max_consec)
            return

        if len(self.active_trades) >= cfg.MAX_CONCURRENT_POSITIONS:
            logger.info("%s max concurrent", symbol)
            return

        if symbol in self.active_trades:
            return

        # ── Feature 5: Cash buffer enforcement (20% must remain undeployed) ──
        deployed_capital = sum(
            t.get("entry_price", 0) * t.get("quantity", 0)
            for t in self.active_trades.values()
        )
        if not self.risk.check_cash_buffer(deployed_capital):
            logger.info("%s blocked: cash buffer enforced (deployed=%.2f)", symbol, deployed_capital)
            return

        atr_value = indicators_3m.get("atr", 1) if isinstance(indicators_3m.get("atr"), (int, float)) else 1
        atr_pct = (atr_value / ltp * 100) if ltp > 0 else 1.0
        default_sl_percent = stop_loss_floor_percent(
            atr_value, ltp, cfg.STOP_LOSS_ATR_MULTIPLIER, cfg.MIN_STOP_LOSS_PCT
        )
        raw_sl_percent = signal.get("stop_loss_percent", default_sl_percent)
        sl_percent = normalize_stop_loss_percent(
            raw_sl_percent, atr_value, ltp, cfg.STOP_LOSS_ATR_MULTIPLIER, cfg.MIN_STOP_LOSS_PCT
        )
        # ── Hard ceiling on the stop ────────────────────────────────────────
        # ATR-based sizing has only a FLOOR (MIN_STOP_LOSS_PCT); high-ATR names
        # (ATR up to ~2%) otherwise get stops as wide as ~3%, producing the fat
        # loss tail that sank the book (trades worse than -1% were -66 of -123).
        # Cap the per-trade stop; combined with the 25%-of-buying-power position
        # cap this bounds worst-case loss to ~(0.25*buying_power)*MAX_STOP_LOSS_PCT.
        max_sl = getattr(cfg, "MAX_STOP_LOSS_PCT", 0)
        if max_sl and sl_percent > max_sl:
            logger.info("%s SL capped to ceiling: %.2f%% -> %.2f%%", symbol, sl_percent, max_sl)
            sl_percent = max_sl
        try:
            raw_sl_for_log = float(raw_sl_percent)
        except (TypeError, ValueError):
            raw_sl_for_log = default_sl_percent
        if sl_percent > raw_sl_for_log:
            logger.info("%s SL raised to execution floor: %.2f%% -> %.2f%%",
                        symbol, raw_sl_for_log, sl_percent)
        # Fallback target must clear the R:R gate below; a plain 2x SL default
        # is auto-rejected whenever MIN_RR_RATIO is tuned above 2.0.
        fallback_target = round(max(atr_pct * 3.0, sl_percent * cfg.MIN_RR_RATIO), 2)
        target_percent = signal.get("target_percent", fallback_target)

        # ── R:R check must happen BEFORE Kronos tightening ─────────────────
        if sl_percent > 0 and target_percent < sl_percent * cfg.MIN_RR_RATIO:
            logger.info("%s R:R too low (Target: %.2f%%, SL: %.2f%%, Min RR: %.2f)", symbol, target_percent, sl_percent, cfg.MIN_RR_RATIO)
            return

        # ── Kronos dynamic SL based on predicted range ─────────────────────
        if kronos_conf and kronos_conf.get("pred_df") is not None:
            pred_df = kronos_conf["pred_df"]
            range_score = kronos_conf.get("range_score", 0)
            if 0 < range_score <= 0.3:
                tight_factor = 0.5 + 0.5 * range_score
                proposed_sl = round(sl_percent * tight_factor, 2)
                sl_floor = stop_loss_floor_percent(
                    atr_value, ltp, cfg.STOP_LOSS_ATR_MULTIPLIER, cfg.MIN_STOP_LOSS_PCT
                )
                sl_percent = max(proposed_sl, sl_floor)
                logger.info("%s Kronos range=%.2f tightened SL: %.2f%% -> %.2f%% (floor=%.2f%%)",
                            symbol, range_score, proposed_sl, sl_percent, sl_floor)

        capital = self.risk.current_capital
        quantity = self.risk.calculate_position_size(capital, sl_percent, ltp)

        # ── Kronos position scaling ─────────────────────────────────────────
        if kronos_ratio != 1.0:
            base_qty = quantity
            quantity = max(1, int(quantity * kronos_ratio))
            logger.info("%s Kronos position scale: %d -> %d (ratio=%.2f)",
                        symbol, base_qty, quantity, kronos_ratio)

        # ── Confidence-based position scaling ───────────────────────────────
        conf_scalar = self.cfg.get("position_confidence_scalar", 1.0)
        if conf_scalar > 1.0 and confidence >= 85:
            base_qty = quantity
            quantity = max(1, int(quantity * conf_scalar))
            logger.info("%s confidence scale (conf=%d): %d -> %d (scalar=%.2f)",
                        symbol, confidence, base_qty, quantity, conf_scalar)

        if quantity < 1:
            self.last_signal_time[symbol] = time.time()
            logger.info("%s qty=0", symbol)
            return

        # ── Atomic slot reservation (authoritative daily cap gate) ──────────
        # This is the TOCTOU-safe gate: atomically checks both per-stock and
        # global caps, and increments the counter in a single call.
        if not self.signal_log.reserve_daily_slot(
            symbol, cfg.MAX_DAILY_SIGNALS, cfg.MAX_SIGNALS_PER_STOCK_PER_DAY
        ):
            logger.info("%s -- daily signal cap reached (atomic check)", symbol)
            return
        _slot_reserved = True

        trans_type = self.dhan.dhan.BUY if sig_type == "BUY" else self.dhan.dhan.SELL
        sl_price = ltp * (1 - sl_percent / 100) if sig_type == "BUY" else ltp * (1 + sl_percent / 100)
        target_price = ltp * (1 + target_percent / 100) if sig_type == "BUY" else ltp * (1 - target_percent / 100)
        trail_dist_atr = self.cfg.get("trailing_sl_distance_atr", 2.0)
        trailing_sl = ltp - (trail_dist_atr * atr_value) if sig_type == "BUY" else ltp + (trail_dist_atr * atr_value)
        tag = "ENTRY-LONG" if sig_type == "BUY" else "ENTRY-SHORT"

        nifty_regime = regime_data.get("nifty", {}).get("trend", "")
        sector_regime = ""
        if regime_data.get("sector"):
            s_trend = regime_data["sector"].get("trend", "")
            sector_name = regime_data.get("sector_name", "")
            sector_regime = f"{sector_name}={s_trend.upper()}" if sector_name else s_trend.upper()

        reasoning_extra = ""
        if kronos_conf:
            reasoning_extra = " | Kronos: ratio={:.2f} {} range={:.2f}%".format(
                kronos_ratio,
                "CONFLICT" if kronos_conf["conflict"] else "ALIGN",
                kronos_conf.get("pred_range_pct", 0),
            )

        self.last_signal_time[symbol] = time.time()
        self._last_signals[symbol] = {
            "signal": sig_type, "confidence": confidence,
            "direction": sig_type, "timestamp": time.time(),
            "setup_type": signal.get("setup_type"), "reasoning": reasoning,
        }
        trade_data = {
            "symbol": symbol,
            "security_id": security_id,
            "signal_type": sig_type,
            "transaction_type": trans_type,
            "entry_price": ltp,
            "quantity": quantity,
            "sl_price": sl_price,
            "stop_loss_percent": sl_percent,
            "target_percent": target_percent,
            "trailing_sl": trailing_sl,
            "target_price": target_price,
            "confidence": confidence,
            "reasoning": reasoning + reasoning_extra,
            "entry_time": self._now_ist(),
            "setup_type": signal.get("setup_type", "NONE"),
            "atr_value": atr_value,
            "trail_activation_pct": self.cfg.get("trailing_sl_activation_pct", 3.0),
            "partial_profit_pct": self.cfg.get("partial_profit_pct", 0.0),
            # Feature 3: RAG snapshot for post-exit storage
            "_entry_indicators": {
                "rsi": indicators_3m.get("rsi", 50),
                "adx": indicators_3m.get("adx", 20),
                "volume_ratio": indicators_3m.get("volume_ratio", 1.0),
                "mfi": indicators_3m.get("mfi", 50),
                "atr_pct": atr_pct,
            },
            "_entry_kronos_conf": kronos_conf,
            "_entry_nifty": nifty_regime,
            "_entry_regime": regime_str_for_rag,
        }

        fail_remarks = ""
        if self.dry_run:
            trade_data["order_id"] = f"DRY-{symbol}-{int(time.time())}"
            self.active_trades[symbol] = trade_data
            logger.info("%s DRY-RUN %s qty=%d entry=%.2f SL=%.2f target=%.2f",
                        symbol, sig_type, quantity, ltp, sl_price, target_price)
            order_failed = False
        else:
            async with self._dhan_sem:
                result = await asyncio.to_thread(
                    self.dhan.place_super_order,
                    security_id, trans_type, quantity, ltp,
                    sl_percent, target_percent, symbol=symbol, atr_value=atr_value,
                )
            logger.info("%s order result: %s", symbol, result)
            if result and result.get("status") == "success":
                trade_data["order_id"] = result.get("data", {}).get("orderId")
                self.active_trades[symbol] = trade_data
                order_failed = False
            else:
                fail_remarks = str(result.get("remarks", "unknown") if result else "no response")
                logger.warning("%s %s super order FAILED: %s", symbol, sig_type, fail_remarks)
                # Release the reserved slot since order failed
                self.signal_log.release_daily_slot(symbol)
                order_failed = True

        # ── Always send telegram notification ────────────────────────────────
        if order_failed:
            tag = f"ORDER-FAILED-{'LONG' if sig_type == 'BUY' else 'SHORT'}"
        self._notify_telegram(symbol, tag,
                              "LONG" if sig_type == "BUY" else "SHORT",
                              quantity, ltp,
                              sl_price=sl_price, tp1_price=target_price,
                              trailing_sl=trailing_sl)

        # ── Always log signal to CSV ─────────────────────────────────────────
        def _tf_trend(ind):
            if not ind:
                return "NEUTRAL"
            c = ind.get("close", 0)
            s = ind.get("sma_20", c)
            return "BULLISH" if c > s else "BEARISH" if c < s else "NEUTRAL"

        mtf_3m = _tf_trend(indicators_3m)
        mtf_15m = _tf_trend(indicators_15m)
        mtf_1h = _tf_trend(indicators_1h)

        log_tag = tag if not order_failed else f"FAILED-{'LONG' if sig_type == 'BUY' else 'SHORT'}"
        log_reasoning = reasoning + reasoning_extra
        if order_failed:
            log_reasoning += f" | ORDER FAILED: {fail_remarks}"

        await self.signal_log.log_signal(
            symbol=symbol,
            signal_type=log_tag,
            direction=sig_type,
            entry_price=ltp,
            quantity=quantity,
            stop_loss=sl_price,
            trailing_stop=trailing_sl,
            target=target_price,
            confidence=confidence,
            reasoning=log_reasoning,
            mode="DRY-RUN" if self.dry_run else ("LIVE-FAILED" if order_failed else "LIVE"),
            market_regime=nifty_regime.upper(),
            sector_regime=sector_regime.upper(),
            mtf_3m=mtf_3m,
            mtf_15m=mtf_15m,
            mtf_1h=mtf_1h,
            slot_reserved=not order_failed,  # only count slot if order succeeded
        )

    # ── Feature 3: RAG — store trade outcome on exit ─────────────────────────

    async def _exit_position(self, symbol, reason="EXIT"):
        """Override to capture trade outcome into the analog RAG database."""
        trade = self.active_trades.get(symbol)
        if not trade:
            await super()._exit_position(symbol, reason)
            return

        # Snapshot indicators stored at entry (before parent deletes the trade)
        entry_indicators = trade.get("_entry_indicators", {})
        entry_kronos_conf = trade.get("_entry_kronos_conf")
        entry_nifty = trade.get("_entry_nifty", "")
        entry_regime = trade.get("_entry_regime", "")
        signal_type = trade.get("signal_type", "")
        confidence = trade.get("confidence", 0)

        # Estimate exit price and PnL before parent closes the trade.
        # Only store to RAG if we actually get a price — pnl=0 would record
        # every exit as LOSS (since 0 is not > 0 in the WIN check).
        pnl_est = None
        pnl_pct_est = None
        try:
            security_id = self.dhan.security_ids.get(symbol)
            if security_id:
                async with self._dhan_sem:
                    live = await asyncio.to_thread(self.dhan.fetch_live_data, security_id)
                exit_price = live.get("last_price")
                if exit_price:
                    pnl_est, pnl_pct_est = self._calc_pnl(trade, exit_price)
        except Exception as exc:
            logger.debug("%s RAG exit price estimate failed: %s", symbol, exc)

        # Call parent to do the actual exit (fetches live price again, sends telegram, etc.)
        await super()._exit_position(symbol, reason)

        # Store to RAG only when we have a valid PnL estimate
        if entry_indicators and signal_type in ("BUY", "SELL") and pnl_est is not None:
            try:
                self.rag.store_setup(
                    symbol=symbol,
                    indicators=entry_indicators,
                    kronos_conf=entry_kronos_conf,
                    nifty_trend=entry_nifty,
                    market_regime=entry_regime,
                    signal_type=signal_type,
                    confidence=int(confidence),
                    pnl=pnl_est,
                    pnl_pct=pnl_pct_est,
                )
                logger.debug("RAG stored: %s %s P&L=%.2f (%.2f%%)", symbol, signal_type, pnl_est, pnl_pct_est)
            except Exception as exc:
                logger.warning("RAG store_setup failed for %s: %s", symbol, exc)
