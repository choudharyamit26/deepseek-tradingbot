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
    SL_TP_MIN_PCT = 0.25
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
            "reasoning": "Step-by-step reasoning for your DIRECTION choice (advisory)",
            "stop_loss_percent": {example_sl},
            "target_percent": {example_tp},
            "setup_type": "VWAP_RECLAIM" | "BREAKOUT" | "REVERSAL" | "MOMENTUM" | "NONE"
        }}

        YOUR JOB: choose the DIRECTION (BUY / SELL / HOLD) and the risk levels
        (stop_loss_percent, target_percent). You do NOT score the trade. The
        SYSTEM computes the authoritative confidence (volume, ADX, MTF, Kronos,
        MFI, analog history, Nifty/sector regime) and decides whether it passes.
        Your `confidence` field is ADVISORY ONLY — report your honest qualitative
        conviction, but NEVER downgrade a viable BUY/SELL to HOLD because you
        think the number looks low. Direction is your job; scoring is the system's.

        REASONING PROTOCOL — follow each step IN ORDER:
        1. VIABILITY (this step decides the direction):
           BUY:  price above VWAP, RSI < {self.rsi_ob}, ADX > {self.min_adx}.
           SELL: price below VWAP, RSI > {self.rsi_os}, ADX > {self.min_adx}.
           Pick the direction that passes ALL of its rules. If neither passes -> HOLD.
        2. VOLUME context: trending (price above VWAP+SMA20, ADX>{self.min_adx + 4})
           wants vol >= 0.5; a fresh breakout/reversal wants vol > 1.2. Thin
           volume weakens the case but the system applies the volume penalty —
           you do not need to subtract anything.
        3. CANDLES: do the last 5 candles move WITH your chosen direction? If they
           clearly move against it, prefer HOLD.
        4. MTF: note whether the 15-min / 1-hour trends agree (informational; the
           system scores alignment).
        5. KRONOS: read the KRONOS TIME-SERIES FORECAST section. Use it ONLY to
           tune the risk levels — pred_range_pct < 0.2% (tight range) -> lower
           conviction, keep stop_loss_percent at the ATR floor; pred_range_pct
           > 0.6% (wide range) -> consider a wider target_percent.
        6. ANALOG EVIDENCE: if an ANALOG SETUPS section is present and similar
           setups lose money historically, prefer HOLD or a cleaner setup. The
           system applies the numeric analog penalty/bonus itself.
        7. RISK LEVELS: stop_loss_percent = 1.5 x ATR% for this stock ({atr_sl_hint}).
           target_percent must be >= {min_rr:g}x SL (floor = {min_tp_floor}%).
           The system HARD-REJECTS any trade with target below {min_rr:g}x the stop.
           Calculate from the actual ATR in Technical Indicators — each stock has
           different volatility. NEVER copy the example numbers.

        DECISION RULES:
        - Output BUY or SELL whenever its viability rules (step 1) pass and the
          setup is not flatly contradicted by candles (step 3) or analog history.
        - Output HOLD only when neither direction is viable.
        - Do NOT compute a 0-100 threshold, and do NOT talk yourself out of a
          viable trade with extra qualitative worries — the system scores and
          gates it. When signal=HOLD, set setup_type=NONE.
        - Do NOT rationalize away a viability violation. If price is above VWAP,
          SELL is not viable, PERIOD. Do not round borderline values.

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

        # NOTE (C1 + matrix-authoritative, 2026-06-24): the algorithmic score is
        # NOT injected into the prompt, and the AI's confidence number is no longer
        # used for the trade gate at all. The AI's self-scored confidence proved
        # unstable (same input -> 83 then 76; emitted field decoupled from its own
        # reasoning), so it now only chooses DIRECTION + risk levels. The binding
        # confidence is computed in enhanced_bot._analyze() as the deterministic
        # _compute_score_matrix value (plus analog/candle/regime penalties). The AI's
        # `confidence` field is advisory/logged only. matrix_score / matrix_breakdown
        # are kept in the signature for logging/back-compat.

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

            result = json.loads(self._sanitize_json_content(content))

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

    @staticmethod
    def _sanitize_json_content(content: str) -> str:
        """Escape literal control characters that appear inside JSON string values.

        The model occasionally emits unescaped newlines / tabs / carriage returns
        inside the "reasoning" field, making json.loads() raise JSONDecodeError.
        This walks the content character-by-character, tracks whether we are
        inside a string literal, and replaces bare control chars with their
        JSON escape sequences so the parser never sees them raw.
        """
        _ESCAPE = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}
        result = []
        in_string = False
        escape_next = False
        for ch in content:
            if escape_next:
                result.append(ch)
                escape_next = False
            elif ch == '\\' and in_string:
                result.append(ch)
                escape_next = True
            elif ch == '"':
                in_string = not in_string
                result.append(ch)
            elif in_string and ord(ch) < 0x20:
                result.append(_ESCAPE.get(ch, f'\\u{ord(ch):04x}'))
            else:
                result.append(ch)
        return ''.join(result)

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
