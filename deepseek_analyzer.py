# deepseek_analyzer.py
import re
import aiohttp
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


class DeepSeekStockAnalyzer:
    # Sane bounds for LLM-suggested SL/TP percentages; values outside are
    # dropped so the caller falls back to its ATR-based defaults.
    SL_TP_MIN_PCT = 0.2
    SL_TP_MAX_PCT = 5.0

    def __init__(self, api_key, alert_cb=None, min_confidence=60, min_adx=18,
                 rsi_ob=70, rsi_os=30, min_rr_ratio=1.8):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-v4-pro"
        self.alert_cb = alert_cb  # e.g. send_telegram; called on billing/quota errors
        self._billing_alerted = False
        # Strategy thresholds injected into the prompt so the model reasons
        # against the SAME rulebook the system enforces (these are tuned by
        # the reflection agent via kronos_strategy.yaml).
        self.min_confidence = min_confidence
        self.min_adx = min_adx
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os
        self.min_rr = min_rr_ratio

    def prepare_market_context(self, symbol, market_data, technical_indicators,
                               recent_bars=None, previous_signal=None):
        """Build the user prompt with market data, indicators, and candle history."""
        context = f"""
        Stock: {symbol}
        Current Price: {market_data.get('ltp')}
        3-min High/Low: {market_data.get('high_3m')}/{market_data.get('low_3m')}
        Volume (latest 3min bar): {market_data.get('volume')}
        Last 5 bar avg volume: {market_data.get('avg_volume_3m')}
        
        Technical Indicators (3-min timeframe):
        - RSI(14): {technical_indicators.get('rsi')}
        - MACD line: {technical_indicators.get('macd')} | Signal: {technical_indicators.get('macd_signal')}
        - 20-period SMA: {technical_indicators.get('sma_20')}
        - 9-period EMA: {technical_indicators.get('ema_9')}
        - Bollinger %B: {technical_indicators.get('bb_percent_b')}
        - ATR (volatility): {technical_indicators.get('atr')}
        - ADX (trend strength): {technical_indicators.get('adx', 'N/A')}
        - MFI (money flow): {technical_indicators.get('mfi', 'N/A')}
        - Support (S1): {technical_indicators.get('support')} | Resistance (R1): {technical_indicators.get('resistance')}
        - VWAP: {technical_indicators.get('vwap')}
        - VWAP Distance: {technical_indicators.get('vwap_distance_pct', 0):.3f}%
        - Volume Spike (vs 20-period avg): {technical_indicators.get('volume_ratio')}
        """

        if recent_bars is not None and len(recent_bars) >= 5:
            last5 = recent_bars.tail(5)
            bar_lines = []
            for i, (_, row) in enumerate(last5.iterrows()):
                bar_lines.append(
                    f"  Bar-{5-i}: O={row['open']:.2f} H={row['high']:.2f} "
                    f"L={row['low']:.2f} C={row['close']:.2f} V={int(row['volume'])}"
                )
            context += "\n\nLast 5 candles (3-min, oldest to newest):\n" + "\n".join(bar_lines)

            price_change = (last5['close'].iloc[-1] - last5['close'].iloc[0])
            price_change_pct = price_change / last5['close'].iloc[0] * 100
            context += f"\n5-bar price change: {price_change_pct:+.2f}%"

            latest_candle = last5.iloc[-1]
            body = latest_candle['close'] - latest_candle['open']
            full_range = latest_candle['high'] - latest_candle['low']
            if full_range > 0:
                body_ratio = abs(body) / full_range
                candle_type = "BULLISH" if body > 0 else "BEARISH"
                context += f"\nLatest candle: {candle_type}, body/range={body_ratio:.0%}"

        # Previous decision context — removed to avoid cascading state divergence.
        # CoT protocol provides systematic reasoning each cycle independently.

        return context

    async def get_trading_signal(self, symbol, market_data, indicators,
                           regime_context=None, recent_bars=None,
                           previous_signal=None, track_record=None,
                           matrix_score=None, matrix_breakdown=None):
        """
        Call DeepSeek API to generate intraday stock trading signal.
        Args:
            regime_context: optional string with Nifty & sector regime info.
            recent_bars: optional DataFrame with last N candles for price action context.
        Returns JSON with action, confidence, stop loss %, target %, and reasoning.
        """
        rules_extra = ""
        if regime_context:
            rules_extra = (
                "- Account for market & sector regime: avoid BUY in bearish sectors, prefer SELL in overbought markets.\n"
                "- 'Nifty trend' is the DAILY (10-day SMA) macro trend. 'intraday_chg' is today's session move vs yesterday close.\n"
                "- The system applies a Nifty regime penalty that scales linearly with intraday_chg: 0% move = -6, +0.5% = -3, +1.0% or more = 0 (fully waived).\n"
                "- Do NOT apply Nifty or sector penalties yourself — the system adds them after your output and self-subtracting double-counts them.\n"
                "- On a strong intraday rally (intraday_chg >= +1%), treat bearish Nifty daily trend as neutral — do not over-penalise BUY signals.\n"
                "- On a strong intraday drop (intraday_chg <= -1%), treat bullish Nifty daily trend as neutral for SELL signals."
            )

        min_rr = self.min_rr

        # Compute ATR% for this specific stock so the prompt shows realistic
        # SL/TP examples instead of hardcoded 1.2% that the AI anchors on.
        atr_raw = (indicators or {}).get("atr", 0)
        ltp_for_atr = (market_data or {}).get("ltp", 0) or 1000
        if atr_raw and ltp_for_atr and float(ltp_for_atr) > 0:
            atr_pct = round(float(atr_raw) / float(ltp_for_atr) * 100, 3)
            example_sl = max(round(atr_pct * 1.5, 2), 0.25)  # 1.5x ATR, floor 0.25%
        else:
            atr_pct = 0.0
            example_sl = 0.8
        example_tp = round(example_sl * min_rr * 1.1, 2)
        min_tp_floor = round(example_sl * min_rr, 2)
        atr_sl_hint = (f"ATR = {atr_pct:.3f}% → 1.5×ATR = {example_sl:.2f}%"
                       if atr_pct > 0 else "use ATR from Technical Indicators")

        # System prompt with Chain-of-Thought reasoning protocol
        system_prompt = f"""You are an expert intraday equity trader for the Indian stock market (NSE).
        Analyze the provided data and return a trading decision in JSON format.
        You may suggest LONG (BUY) or SHORT (SELL) trades on liquid stocks.
        Return ONLY valid JSON, no extra text. CRITICAL: NEVER use unescaped double quotes inside string values. Use apostrophe or rephrase to avoid quotes inside strings — broken JSON cannot be parsed.

        Output format:
        {{
            "signal": "BUY" | "SELL" | "HOLD",
            "confidence": 0-100,
            "reasoning": "Step-by-step reasoning showing your evaluation and penalty deductions",
            "stop_loss_percent": {example_sl},
            "target_percent": {example_tp},
            "setup_type": "VWAP_RECLAIM" | "BREAKOUT" | "REVERSAL" | "MOMENTUM" | "NONE",
            "penalty_breakdown": "e.g. -10 MTF, -10 Kronos = -20 total"
        }}

        REASONING PROTOCOL — Follow each step IN ORDER before deciding:
        1. VIABILITY: Check mandatory rules for each direction.
           BUY: price above VWAP, RSI < {self.rsi_ob}, ADX > {self.min_adx}.
           SELL: price below VWAP, RSI > {self.rsi_os}, ADX > {self.min_adx}.
           If neither direction passes all its rules -> HOLD immediately.
        2. VOLUME: Evaluate volume_ratio with context.
           Trending (price above VWAP+SMA20, ADX>{self.min_adx + 4}): vol >= 0.5 ok.
           Breakout/Reversal: vol > 1.2 required.
           vol < 0.3 always -> HOLD (see penalty matrix below).
        3. CANDLES: Check last 5 candles — is price moving with or against your signal?
        4. MTF: Check multi-timeframe alignment. 15-min/1-hour disagree? Apply penalty.
        5. KRONOS: Read the KRONOS TIME-SERIES FORECAST section carefully.
           KRONOS CONFLICT definition: total forecast return is NEGATIVE for BUY signals,
           or POSITIVE for SELL signals — regardless of magnitude.
           If conflict -> -8 penalty. If aligned AND pred_range_pct > 0.5% -> +2 bonus.
           Kronos predicts the next 30 minutes. Use pred_range_pct to tune SL/TP:
           if pred_range_pct < 0.2% (tight range), tighten your stop_loss_percent.
           if pred_range_pct > 0.6% (wide range), consider widening target_percent.
        6. ANALOG EVIDENCE: If an ANALOG SETUPS section is present in the context:
           - win rate < 35% among similar past trades -> apply -10 confidence penalty
           - win rate >= 65% among similar past trades -> apply +5 confidence bonus
           - Always include analog win rate in your penalty_breakdown.
           If no ANALOG SETUPS section present, skip this step.
        7. R:R: Set stop_loss_percent = 1.5 x ATR% for this stock ({atr_sl_hint}).
           Target must be >= {min_rr:g}x SL (min_tp_floor = {min_tp_floor}%).
           The system HARD-REJECTS any trade with target below {min_rr:g}x the stop loss.
           Do NOT copy example values — calculate from the actual ATR in Technical Indicators.
           Each stock has different volatility.
        8. CONFIDENCE: Apply the PENALTY MATRIX below. Start at 100 and SUBTRACT.
           Compute final score, then:
           - If score < {self.min_confidence}: set signal=HOLD, setup_type=NONE,
             confidence=computed score (NOT zero — always report the actual number).
           - If score >= {self.min_confidence}: you MUST output the BUY/SELL signal.
             NO further overrides allowed after the math passes.

        ═══════════════════════════════════════════════════════════
        CONFIDENCE PENALTY MATRIX (MANDATORY — apply ALL that match)
        ═══════════════════════════════════════════════════════════
        Start at 100. Subtract each applicable penalty:

        CRITICAL (any one -> signal=HOLD, confidence=computed score):
        - ADX < {self.min_adx}                                    -> HOLD
        - Volume ratio < 0.3 in any market             -> HOLD
        - BUY when price below VWAP                    -> HOLD
        - SELL when price above VWAP                   -> HOLD

        MAJOR PENALTIES:
        - Volume ratio 0.3-0.5 (very weak volume)     -> -12 points
        - Volume ratio 0.5-0.8 (below average)        -> -8 points
        - Kronos forecast CONFLICTS with signal        -> -8 points
        - 15-min timeframe disagrees with signal       -> -7 points
        - 1-hour timeframe disagrees with signal       -> -5 points
        - Last 3+ candles move AGAINST signal dir.     -> -8 points
        - MFI > 80 with stalling price (for BUY)      -> -12 points
        - MFI < 20 with stalling price (for SELL)      -> -12 points
        - Analog similar-setup win rate < 35%          -> -10 points

        MINOR PENALTIES:
        - ADX borderline ({self.min_adx}-{self.min_adx + 4})                        -> -5 points
        - Volume ratio 0.8-1.0 (slightly below avg)   -> -3 points
        - Kronos pred_range_pct < 0.2% (low forecast conviction) -> -3 points
        - R:R ratio barely meets minimum               -> -3 points

        NOTE: Nifty/sector regime penalties (-6/-4) are applied automatically
        by the system AFTER your output. Do NOT subtract them here.

        BONUSES (max +10 total):
        - Volume ratio > 2.0 (exceptional volume)      -> +3 points
        - Volume ratio > 1.5 (strong volume)           -> +2 points
        - All 3 timeframes aligned with signal         -> +3 points
        - Kronos aligned AND pred_range_pct > 0.5%    -> +2 points
        - Analog similar-setup win rate >= 65%         -> +5 points

        Final confidence = 100 - (sum of penalties) + (sum of bonuses, capped at +10)
        If final confidence < {self.min_confidence} -> set signal=HOLD.
        If final confidence >= {self.min_confidence} -> you MUST output the active signal (BUY or SELL).
        Once the math passes the threshold, do NOT second-guess, add qualitative concerns,
        or downgrade to HOLD. The numeric score is BINDING. If the number passes, the trade passes.
        The system discards any signal below {self.min_confidence}; do not
        round up to sneak past the threshold.
        When signal=HOLD: always set setup_type=NONE.

        PENALTY MATH EXAMPLES (pass threshold = {self.min_confidence}):
        - Perfect setup: 100 + 5 bonuses = 105 -> capped at 100, strong PASS -> output BUY/SELL
        - Good setup, slightly low volume: 100 - 8 = 92 -> {"PASS -> output BUY/SELL" if 92 >= self.min_confidence else "FAIL -> HOLD"}
        - Weak volume + Kronos conflict: 100 - 12 - 8 = 80 -> {"PASS -> output BUY/SELL" if 80 >= self.min_confidence else "FAIL -> HOLD"}
        - Very weak volume + MTF disagree: 100 - 12 - 7 = 81 -> {"PASS -> output BUY/SELL" if 81 >= self.min_confidence else "FAIL -> HOLD"}
        - Marginal SELL, vol ok, 15m opposes: 100 - 8 - 7 = 85 -> {"PASS -> output SELL, do NOT change to HOLD" if 85 >= self.min_confidence else "FAIL -> HOLD"}
        - Weak vol + bad analogs + Kronos conflict: 100 - 12 - 10 - 8 = 70 -> {"PASS -> output BUY/SELL" if 70 >= self.min_confidence else "FAIL -> HOLD"}
        - Strong volume + all aligned + good analogs: 100 + 3 + 3 + 5 = 111 -> capped 100, strong PASS -> output BUY/SELL

        You MUST list each penalty/bonus applied in your penalty_breakdown field.
        Do NOT rationalize away violations. If volume is 0.48, it is < 0.5, PERIOD.
        Do NOT round up borderline values. 0.48 is not "approximately 0.5".
        ═══════════════════════════════════════════════════════════

        {rules_extra}
        """

        user_prompt = self.prepare_market_context(symbol, market_data, indicators,
                                                    recent_bars=recent_bars,
                                                    previous_signal=previous_signal)
        if regime_context:
            user_prompt += f"\n\n        Context:\n        {regime_context}"

        # Inject recent P&L feedback — blunt caveman language to override AI's optimism bias
        if track_record:
            user_prompt += f"""
        === YOU LOOK AT TRADING RECORD NOW ===
        PAST SINS OF AI:
        {track_record}
        
        CAVEMAN EXPLAIN: 
        AI make trade before.
        AI think he smart, give 80+ confidence.
        But Winners avg confidence and Losers avg confidence almost SAME.
        This mean AI confidence is RANDOM NOISE. Just coin flip!
        AI must stop lying to self.
        If similar setup lost money before, confidence MUST be low.
        If you want to give 82+ confidence, you ask self: is this REALLY different or am I just dumb caveman?
        """

        # Inject algorithmic confidence ceiling
        if matrix_score is not None:
            user_prompt += f"""
        === HARD CONFIDENCE CEILING ===
        System calculate max possible confidence score for this setup is: {matrix_score}
        Reason: {matrix_breakdown}
        
        CAVEMAN RULE: 
        You CANNOT output confidence higher than {matrix_score}. 
        If you write > {matrix_score}, system force-cap it to {matrix_score}.
        Explain in reasoning how the ceiling limits your optimism.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,   # Deterministic sampling
            "top_p": 1.0,         # Full nucleus — no additional truncation
            "max_tokens": 3000,   # Must accommodate verbose penalty-matrix reasoning
            "response_format": {"type": "json_object"},
            # V4 Pro: disable thinking mode for JSON output
            # (thinking mode ignores temperature/top_p params)
            "thinking": {"type": "disabled"},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status in (402, 429):
                        reason = "quota/payment exhausted (402)" if response.status == 402 else "rate limited (429)"
                        logger.error("DeepSeek %s — signals disabled until resolved", reason)
                        if self.alert_cb and not self._billing_alerted:
                            self._billing_alerted = True
                            try:
                                self.alert_cb(f"DEEPSEEK API ERROR: {reason}. Bot is returning HOLD for all stocks.")
                            except Exception:
                                logger.exception("Alert callback failed")
                        return {"signal": "HOLD", "confidence": 0, "reasoning": f"DeepSeek {reason}"}
                    response.raise_for_status()
                    resp_json = await response.json()
                    message = resp_json['choices'][0]['message']

            # V4 Pro may return reasoning_content alongside content
            reasoning_content = message.get('reasoning_content', '')
            content = message.get('content', '')

            if reasoning_content:
                logger.debug("%s V4 Pro reasoning: %s", symbol, reasoning_content[:200])

            # Handle empty content (documented V4 Pro edge case with JSON mode)
            if not content or content.strip() == '':
                logger.warning("%s V4 Pro returned empty content", symbol)
                return {"signal": "HOLD", "confidence": 0, "reasoning": "Empty AI response"}

            result = json.loads(content)

            # Log the raw AI response for transparency
            logger.info("%s raw AI response: %s", symbol, result)

            return self._validate_signal_fields(result, symbol)
        except asyncio.TimeoutError:
            return {"signal": "HOLD", "confidence": 0, "reasoning": "API Timeout"}
        except aiohttp.ClientError as e:
            return {"signal": "HOLD", "confidence": 0, "reasoning": f"API Error: {e}"}
        except json.JSONDecodeError as e:
            logger.warning("%s JSON parse error: %s | raw: %s", symbol, e, content[:200] if content else 'empty')
            # Attempt to salvage signal & confidence from truncated JSON
            return self._repair_truncated_json(content, symbol)
        except Exception as e:
            return {"signal": "HOLD", "confidence": 0, "reasoning": f"API Error: {e}"}

    @classmethod
    def _validate_signal_fields(cls, result: dict, symbol: str) -> dict:
        """Sanitize and range-check LLM output before it reaches order math."""
        try:
            conf = int(float(result.get("confidence", 0)))
        except (TypeError, ValueError):
            conf = 0
        result["confidence"] = max(0, min(100, conf))

        if result.get("signal") not in ("BUY", "SELL", "HOLD", "EXIT"):
            result["signal"] = "HOLD"

        # SL/TP must be sane percentages; otherwise drop so the caller's
        # ATR-based defaults take over instead of poisoning SL/target prices.
        for key in ("stop_loss_percent", "target_percent"):
            if key not in result:
                continue
            try:
                val = float(result[key])
            except (TypeError, ValueError):
                logger.warning("%s invalid %s=%r from AI — dropped", symbol, key, result[key])
                del result[key]
                continue
            if not (cls.SL_TP_MIN_PCT <= val <= cls.SL_TP_MAX_PCT):
                logger.warning("%s out-of-range %s=%.2f from AI — dropped (bounds %.1f-%.1f)",
                               symbol, key, val, cls.SL_TP_MIN_PCT, cls.SL_TP_MAX_PCT)
                del result[key]
            else:
                result[key] = val

        return result

    @classmethod
    def _repair_truncated_json(cls, raw: str, symbol: str) -> dict:
        """Extract signal & confidence from truncated JSON via regex."""
        result = {"signal": "HOLD", "confidence": 0, "reasoning": "Repaired from truncated response"}
        if not raw:
            return result

        sig_match = re.search(r'"signal"\s*:\s*"(BUY|SELL|HOLD)"', raw)
        conf_match = re.search(r'"confidence"\s*:\s*(\d+)', raw)
        sl_match = re.search(r'"stop_loss_percent"\s*:\s*([\d.]+)', raw)
        tp_match = re.search(r'"target_percent"\s*:\s*([\d.]+)', raw)
        setup_match = re.search(r'"setup_type"\s*:\s*"(\w+)"', raw)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)(?="\s*,\s*"stop_loss_percent"|"\s*,\s*"target_percent"|"\s*,\s*"setup_type"|"\s*,\s*"penalty_breakdown"|\Z)', raw, re.DOTALL)

        if sig_match:
            result["signal"] = sig_match.group(1)
        if conf_match:
            try:
                result["confidence"] = int(conf_match.group(1).rstrip('.'))
            except ValueError:
                pass
        if sl_match:
            try:
                result["stop_loss_percent"] = float(sl_match.group(1).rstrip('.'))
            except ValueError:
                pass
        if tp_match:
            try:
                result["target_percent"] = float(tp_match.group(1).rstrip('.'))
            except ValueError:
                pass
        if setup_match:
            result["setup_type"] = setup_match.group(1)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip() + " [repaired]"

        logger.info("%s repaired truncated JSON -> %s (conf=%d)",
                    symbol, result["signal"], result["confidence"])
        return cls._validate_signal_fields(result, symbol)