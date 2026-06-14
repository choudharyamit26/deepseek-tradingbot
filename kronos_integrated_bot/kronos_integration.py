import sys
import time
import logging
from pathlib import Path
from threading import Lock

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor

logger = logging.getLogger(__name__)


class KronosIntegration:
    def __init__(self, config: dict):
        self.cfg = config
        self._model = None
        self._tokenizer = None
        self._predictor = None
        self._lock = Lock()
        self._prediction_cache = {}  # symbol -> (timestamp, pred_df)
        self._cache_ttl = 180

    def load(self):
        if self._predictor is not None:
            return
        logger.info("Loading Kronos model: %s", self.cfg["model_name"])
        self._tokenizer = KronosTokenizer.from_pretrained(self.cfg["tokenizer_name"])
        self._model = Kronos.from_pretrained(self.cfg["model_name"])
        device = self.cfg["device"]
        if device == "cuda" and not __import__("torch").cuda.is_available():
            device = "cpu"
        self._model.to(device)
        self._model.eval()
        self._predictor = KronosPredictor(
            self._model, self._tokenizer,
            device=device, max_context=self.cfg["max_context"],
        )
        logger.info("Kronos loaded on %s", device)

    @property
    def ready(self) -> bool:
        return self._predictor is not None

    def predict(self, df: pd.DataFrame, symbol: str = "",
                force: bool = False) -> pd.DataFrame | None:
        if not self.ready:
            return None

        if df is None or len(df) < 5:
            logger.warning("Kronos predict: too few bars (%d) for %s", len(df) if df is not None else 0, symbol)
            return None

        now = time.time()
        cache_key = symbol if symbol else id(df)
        if not force and cache_key in self._prediction_cache:
            ts, cached = self._prediction_cache[cache_key]
            if now - ts < self._cache_ttl:
                return cached

        pred_len = self.cfg["pred_len"]
        lookback = min(self.cfg["lookback"], len(df), self.cfg["max_context"])

        x_df = df.tail(lookback).copy()
        required = {"open", "high", "low", "close"}
        if not required.issubset(x_df.columns):
            logger.warning("Missing OHLC columns for Kronos prediction")
            return None
        if "volume" not in x_df.columns:
            x_df["volume"] = 0
        if "amount" not in x_df.columns:
            x_df["amount"] = x_df["close"] * x_df["volume"]
        x_df = x_df[["open", "high", "low", "close", "volume", "amount"]]

        x_ts = pd.Series(x_df.index, index=x_df.index)
        last_ts = x_ts.iloc[-1]
        if len(x_df) >= 2:
            diffs = x_df.index.to_series().diff().dropna()
            gap_minutes = max(1, int(round(diffs.median().total_seconds() / 60)))
        else:
            gap_minutes = 5
        y_ts = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=gap_minutes),
            periods=pred_len, freq=f"{gap_minutes}min", tz=last_ts.tz,
        )
        y_ts = pd.Series(y_ts)

        try:
            pred = self._predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len,
                T=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                sample_count=self.cfg["sample_count"],
                verbose=False,
            )
        except Exception as e:
            import traceback
            logger.error("Kronos predict failed: %s\n%s", e, traceback.format_exc())
            return None

        self._prediction_cache[cache_key] = (now, pred)
        return pred

    def compute_confirmation(self, signal_type: str, pred_df: pd.DataFrame,
                             last_price: float,
                             historical_df: pd.DataFrame | None = None) -> dict:
        if pred_df is None or pred_df.empty:
            return {"agreement": 0.0, "adjustment": 1.0, "conflict": False}

        pred_close = pred_df["close"].iloc[-1] if len(pred_df) > 1 else pred_df["close"].iloc[0]
        pred_open = pred_df["open"].iloc[0]
        pred_return = (pred_close - pred_open) / pred_open if pred_open > 0 else 0

        pred_direction = "BUY" if pred_return > 0 else "SELL"
        conflict = pred_direction != signal_type if signal_type in ("BUY", "SELL") else False

        # Volatility-normalized magnitude: measure predicted return in recent stddev units
        if historical_df is not None and len(historical_df) > 15:
            returns = historical_df["close"].pct_change().dropna()
            recent_vol = returns.tail(20).std()
            if recent_vol > 0:
                magnitude = min(abs(pred_return) / (recent_vol * 2), 1.0)
            else:
                magnitude = min(abs(pred_return) / 0.01, 1.0)
        else:
            magnitude = min(abs(pred_return) / 0.01, 1.0)

        # Predicted range as additional signal quality metric
        pred_range = pred_df["high"].max() - pred_df["low"].min()
        range_pct = (pred_range / last_price * 100) if last_price > 0 else 0
        range_score = min(range_pct / 0.5, 1.0)  # 0.5% range = full score

        if signal_type in ("BUY", "SELL"):
            if conflict:
                adjustment = max(0.1, 1.0 - self.cfg.get("penalty_conflict", 0.50) * magnitude)
                agreement = -magnitude
            else:
                adjustment = 1.0 + self.cfg.get("bonus_align", 1.10) * magnitude
                agreement = magnitude
        else:
            adjustment = 1.0
            agreement = 0.0

        return {
            "agreement": agreement,
            "adjustment": adjustment,
            "conflict": conflict,
            "pred_return": pred_return,
            "pred_direction": pred_direction,
            "pred_close_final": pred_close,
            "pred_close_first": pred_df["close"].iloc[0],
            "pred_range_pct": range_pct,
            "range_score": range_score,
            "magnitude": magnitude,
        }

    def build_prompt_section(self, pred_df: pd.DataFrame, last_price: float) -> str:
        if pred_df is None or pred_df.empty:
            return ""

        pred_close = pred_df["close"].iloc[-1] if len(pred_df) > 1 else pred_df["close"].iloc[0]
        pred_return = (pred_close - last_price) / last_price * 100 if last_price > 0 else 0
        pred_high = pred_df["high"].max()
        pred_low = pred_df["low"].min()

        pred_open = pred_df["open"].iloc[0]
        pred_first_return = (pred_df["close"].iloc[0] - pred_open) / pred_open * 100

        range_pct = (pred_high - pred_low) / last_price * 100 if last_price > 0 else 0

        # Detect candle interval so DeepSeek knows the time horizon
        if len(pred_df) >= 2:
            gap_min = int(round((pred_df.index[1] - pred_df.index[0]).total_seconds() / 60))
        else:
            gap_min = 15
        horizon_min = len(pred_df) * gap_min

        lines = [
            "KRONOS TIME-SERIES FORECAST (next {} x {}-min candles = {} min ahead):".format(
                len(pred_df), gap_min, horizon_min),
            "  Predicted path: open={:.2f} -> close[{:d}]={:.2f} ({:+.2f}%)".format(
                pred_open, len(pred_df)-1, pred_close, pred_return),
            "  Predicted range: {:.2f} - {:.2f} ({:.2f}% of price)".format(
                pred_low, pred_high, range_pct),
            "  Immediate next {}-min candle: {:+.2f}%".format(gap_min, pred_first_return),
        ]

        if range_pct > 0.5:
            lines.append("  Kronos predicts elevated volatility (range {:.2f}%)".format(range_pct))
        elif range_pct < 0.1:
            lines.append("  Kronos predicts tight range ({:.2f}%) - low volatility expected".format(range_pct))

        if abs(pred_return) > 0.3:
            direction_str = "UP" if pred_return > 0 else "DOWN"
            lines.append("  Kronos predicts a significant {} move over next {:.0f} min ({:.2f}%)".format(
                direction_str, horizon_min, abs(pred_return)))

        return "\n".join(lines)

    def predicted_sl_price(self, pred_df: pd.DataFrame, side: str,
                           atr_value: float) -> float | None:
        if pred_df is None or pred_df.empty:
            return None
        if side == "BUY":
            return pred_df["low"].min() - 0.5 * atr_value
        else:
            return pred_df["high"].max() + 0.5 * atr_value

    def predict_batch_for_stocks(
        self,
        stock_data: dict,
        min_bars: int = 30,
    ) -> dict:
        """
        Run a single Kronos batch prediction for multiple stocks simultaneously.

        stock_data: {symbol: df (3m OHLCV, DatetimeIndex)}
        Returns:    {symbol: pred_df} — only for stocks with sufficient bars.

        All series are truncated to the same length (min_bars or the minimum
        available across qualifying stocks) so KronosPredictor.predict_batch()
        can stack them into one tensor.
        """
        if not self.ready:
            return {}

        pred_len = self.cfg["pred_len"]
        lookback = min(self.cfg["lookback"], self.cfg["max_context"])

        # Step 1: keep only stocks that have enough bars
        valid: dict[str, pd.DataFrame] = {}
        for symbol, df in stock_data.items():
            if df is None or len(df) < min_bars:
                continue
            x_df = df.tail(lookback).copy()
            required = {"open", "high", "low", "close"}
            if not required.issubset(x_df.columns):
                continue
            if "volume" not in x_df.columns:
                x_df["volume"] = 0.0
            if "amount" not in x_df.columns:
                x_df["amount"] = x_df["close"] * x_df["volume"]
            x_df = x_df[["open", "high", "low", "close", "volume", "amount"]]
            valid[symbol] = x_df

        if not valid:
            logger.debug("predict_batch_for_stocks: no qualifying stocks (min_bars=%d)", min_bars)
            return {}

        # Step 2: align to the same length required by predict_batch()
        min_len = min(len(df) for df in valid.values())
        symbols = list(valid.keys())

        df_list, x_ts_list, y_ts_list = [], [], []
        for symbol in symbols:
            df = valid[symbol].tail(min_len)
            x_ts = pd.Series(df.index, index=df.index)
            last_ts = df.index[-1]
            diffs = df.index.to_series().diff().dropna()
            gap_min = max(1, int(round(diffs.median().total_seconds() / 60))) if len(diffs) else 3
            y_ts = pd.Series(pd.date_range(
                start=last_ts + pd.Timedelta(minutes=gap_min),
                periods=pred_len, freq=f"{gap_min}min", tz=last_ts.tz,
            ))
            df_list.append(df)
            x_ts_list.append(x_ts)
            y_ts_list.append(y_ts)

        logger.info("Kronos batch predict: %d stocks x %d bars -> %d preds each",
                    len(symbols), min_len, pred_len)
        try:
            pred_dfs = self._predictor.predict_batch(
                df_list, x_ts_list, y_ts_list, pred_len,
                T=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                sample_count=self.cfg["sample_count"],
                verbose=False,
            )
        except Exception as exc:
            import traceback
            logger.error("Kronos predict_batch failed: %s\n%s", exc, traceback.format_exc())
            return {}

        now = time.time()
        results = {}
        for symbol, pred_df in zip(symbols, pred_dfs):
            self._prediction_cache[symbol] = (now, pred_df)
            results[symbol] = pred_df

        logger.info("Kronos batch predict done: %d/%d stocks succeeded", len(results), len(symbols))
        return results

    def get_exit_signal(self, pred_df: pd.DataFrame,
                        entry_price: float, side: str) -> dict:
        if pred_df is None or pred_df.empty:
            return {"exit": False, "urgency": 0}

        pred_close = pred_df["close"].iloc[-1] if len(pred_df) > 1 else pred_df["close"].iloc[0]
        pred_return = (pred_close - entry_price) / entry_price if entry_price > 0 else 0

        exit_threshold = self.cfg.get("exit_threshold", -0.008)
        signal = False
        urgency = 0

        if side == "BUY":
            if pred_return < exit_threshold:
                signal = True
                urgency = min(abs(pred_return) / abs(exit_threshold), 1.0) * 100
        else:
            if -pred_return < exit_threshold:
                signal = True
                urgency = min(abs(pred_return) / abs(exit_threshold), 1.0) * 100

        recent_lows = pred_df["low"].tail(3)
        recent_highs = pred_df["high"].tail(3)
        if side == "BUY" and len(recent_lows) >= 2:
            if recent_lows.is_monotonic_decreasing:
                signal = True
                urgency = max(urgency, 60)
        if side == "SELL" and len(recent_highs) >= 2:
            if recent_highs.is_monotonic_increasing:
                signal = True
                urgency = max(urgency, 60)

        return {"exit": signal, "urgency": int(urgency), "pred_return": pred_return}
