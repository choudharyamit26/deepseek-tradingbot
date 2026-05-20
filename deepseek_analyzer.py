# deepseek_analyzer.py
import requests
import json
import logging

logger = logging.getLogger(__name__)


class DeepSeekStockAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        # V4 Pro: base URL without /v1 suffix per official docs
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-v4-pro"

    def prepare_market_context(self, symbol, market_data, technical_indicators,
                               recent_bars=None):
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

        # IMPROVEMENT #5: Feed last 5-10 candles so the LLM sees price action direction
        if recent_bars is not None and len(recent_bars) >= 5:
            last5 = recent_bars.tail(5)
            bar_lines = []
            for i, (_, row) in enumerate(last5.iterrows()):
                bar_lines.append(
                    f"  Bar-{5-i}: O={row['open']:.2f} H={row['high']:.2f} "
                    f"L={row['low']:.2f} C={row['close']:.2f} V={int(row['volume'])}"
                )
            context += "\n\nLast 5 candles (3-min, oldest to newest):\n" + "\n".join(bar_lines)

            # Directional context
            price_change = (last5['close'].iloc[-1] - last5['close'].iloc[0])
            price_change_pct = price_change / last5['close'].iloc[0] * 100
            context += f"\n5-bar price change: {price_change_pct:+.2f}%"

            # Candle structure hints
            latest_candle = last5.iloc[-1]
            body = latest_candle['close'] - latest_candle['open']
            full_range = latest_candle['high'] - latest_candle['low']
            if full_range > 0:
                body_ratio = abs(body) / full_range
                candle_type = "BULLISH" if body > 0 else "BEARISH"
                context += f"\nLatest candle: {candle_type}, body/range={body_ratio:.0%}"

        return context

    def get_trading_signal(self, symbol, market_data, indicators,
                           regime_context=None, recent_bars=None):
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
                "- Account for market & sector regime: avoid BUY in bearish sectors, prefer SELL in overbought markets,"
                " scale down confidence when sector volatility is high.\n"
                "- If Nifty trend is bearish, be very cautious with BUY signals.\n"
                "- If Nifty trend is bullish, be very cautious with SELL signals."
            )

        # IMPROVEMENT #4: Reworked system prompt with stricter rules and higher conviction bar
        system_prompt = f"""You are an expert intraday equity trader for the Indian stock market (NSE).
        Analyze the provided data and return a trading decision in JSON format.
        You may suggest LONG (BUY) or SHORT (SELL) trades on liquid stocks.
        Return ONLY valid JSON, no extra text.
        
        Output format:
        {{
            "signal": "BUY" | "SELL" | "HOLD",
            "confidence": 0-100,
            "reasoning": "2-3 sentences explaining the specific setup",
            "stop_loss_percent": 1.5,
            "target_percent": 3.0,
            "setup_type": "VWAP_RECLAIM" | "BREAKOUT" | "REVERSAL" | "MOMENTUM" | "NONE"
        }}
        
        STRICT RULES — violations MUST result in HOLD:
        1. HOLD unless you have genuine high conviction. Confidence of 70 means "barely worth trading" — be honest.
           - 75-80 = Good setup with partial confirmation
           - 80-85 = Strong setup with volume confirmation
           - 85-95 = Textbook setup, all indicators aligned
           - Below 75 = HOLD (not worth the risk)
        2. target_percent MUST be >= 2x stop_loss_percent (minimum 2:1 reward-to-risk).
           Base stop_loss on ~1.5x ATR%. If ATR is very small, the stock is not moving — HOLD.
        3. BUY requires ALL of:
           - Price above VWAP (VWAP distance > 0)
           - RSI NOT overbought (RSI < 70)
           - Volume ratio > 1.2 (volume confirmation)
           - ADX > 18 (trending, not ranging)
        4. SELL requires ALL of:
           - Price below VWAP (VWAP distance < 0)
           - RSI NOT oversold (RSI > 30)
           - Volume ratio > 1.2 (volume confirmation)
           - ADX > 18 (trending, not ranging)
        5. MULTI-TIMEFRAME ALIGNMENT is mandatory:
           - BUY blocked when: 1-hour AND 15-min are both BEARISH (price below SMA).
           - SELL blocked when: 1-hour AND 15-min are both BULLISH (price above SMA).
           - When 3-min disagrees with 15-min: HOLD (conflicting momentum).
        6. "RSI overbought" alone is NEVER a SELL trigger. You need:
           price below VWAP + bearish candle structure + volume confirmation.
        7. "RSI oversold" alone is NEVER a BUY trigger. You need:
           price above VWAP + reversal candle + volume spike.
        8. Check MFI (Money Flow Index) for volume-price divergence:
           - MFI > 80 with price stalling = exhaustion (avoid BUY)
           - MFI < 20 with price stalling = capitulation (avoid SELL)
        9. Look at the last 5 candles for DIRECTION. If price is moving against your signal, HOLD.
        10. If ADX < 18, the market is ranging — HOLD regardless of other indicators.
        {rules_extra}
        """

        user_prompt = self.prepare_market_context(symbol, market_data, indicators,
                                                   recent_bars=recent_bars)
        if regime_context:
            user_prompt += f"\n\n        Context:\n        {regime_context}"

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
            "temperature": 0.2,   # Low for consistency
            "max_tokens": 800,    # Increased for V4 Pro's richer responses
            "response_format": {"type": "json_object"},
            # V4 Pro: disable thinking mode for JSON output
            # (thinking mode ignores temperature/top_p params)
            "thinking": {"type": "disabled"},
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=45  # V4 Pro may take slightly longer
            )
            response.raise_for_status()
            resp_json = response.json()
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

            # Log the raw AI response for debugging
            logger.debug("%s raw AI response: %s", symbol, result)

            # Sanitize: ensure confidence is int, signal is valid
            result["confidence"] = int(result.get("confidence", 0))
            if result.get("signal") not in ("BUY", "SELL", "HOLD", "EXIT"):
                result["signal"] = "HOLD"

            return result
        except requests.exceptions.Timeout:
            return {"signal": "HOLD", "confidence": 0, "reasoning": "API Timeout"}
        except json.JSONDecodeError as e:
            logger.warning("%s JSON parse error: %s | raw: %s", symbol, e, content[:200] if content else 'empty')
            return {"signal": "HOLD", "confidence": 0, "reasoning": f"JSON parse error: {e}"}
        except Exception as e:
            return {"signal": "HOLD", "confidence": 0, "reasoning": f"API Error: {e}"}