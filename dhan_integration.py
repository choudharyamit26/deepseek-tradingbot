import os
import threading
import time
import random
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from dhanhq import dhanhq, DhanContext
import logging

logger = logging.getLogger(__name__)

load_dotenv(override=True)

try:
    from constant import FNO_UNIVERSE, ETF_LIQUID, FILTERED_FNO_UNIVERSE, NIFTY50_UNIVERSE
except ImportError:
    FNO_UNIVERSE = {}
    ETF_LIQUID = {}
    FILTERED_FNO_UNIVERSE = {}
    NIFTY50_UNIVERSE = {}

VWAP_RECLAIM_STOCKS: dict[str, str] = {
    # ── Existing 41 stocks ──────────────────────────────────────────────────
    "AUROPHARMA": "275", "BANKINDIA": "4745", "PNB": "10666",
    "KAYNES": "12092", "KFINTECH": "13359", "SAMMAANCAP": "30125",
    "BHEL": "438", "NAUKRI": "13751", "NBCC": "31415",
    "POLICYBZR": "6656", "DELHIVERY": "9599", "PPLPHARMA": "11571",
    "UPL": "11287", "WAAREEENER": "25907", "MCX": "31181",
    "POLYCAB": "9590", "TATATECH": "20293", "VMM": "27969",
    "M&M": "2031", "ABCAPITAL": "21614", "MOTHERSON": "4204",
    "HINDZINC": "1424", "COLPAL": "15141", "NESTLEIND": "17963",
    "BAJFINANCE": "317", "BPCL": "526", "AMBUJACEM": "1270",
    "HUDCO": "20825", "RVNL": "9552", "YESBANK": "11915",
    "SAIL": "2963", "BOSCHLTD": "2181", "PGEL": "25358",
    "INFY": "1594", "NHPC": "17400", "TORNTPOWER": "13786",
    "DIXON": "21690", "DABUR": "772", "MANAPPURAM": "19061",
    "HDFCAMC": "4244", "HDFCLIFE": "467",
    # ── FNO Universe (high liquidity) ──────────────────────────────────────
    "RELIANCE": "2885", "TCS": "11536", "HDFCBANK": "1333",
    "ICICIBANK": "4963", "KOTAKBANK": "1922", "AXISBANK": "5900",
    "SBIN": "3045", "TATASTEEL": "3499", "WIPRO": "3787",
    "HCLTECH": "7229", "SUNPHARMA": "3351", "DRREDDY": "881",
    "MARUTI": "10999", "ASIANPAINT": "236", "TITAN": "3506",
    "ADANIPORTS": "15083", "LTIM": "17818", "TECHM": "13538",
    "POWERGRID": "14977", "NTPC": "11630", "ONGC": "2475",
    "COALINDIA": "20374", "JSWSTEEL": "11723", "HINDALCO": "1363",
    "VEDL": "3063", "BAJAJFINSV": "16675",
    # ── Liquid ETFs ─────────────────────────────────────────────────────────
    "NIFTYBEES": "10576", "BANKBEES": "11439", "ITBEES": "19084",
    "GOLDBEES": "14428", "SILVERBEES": "8080", "JUNIORBEES": "10939",
    # ── Expanded universe (2026-06-01) ─────────────────────────────────────────
    "ADANIGREEN": "3563", "ADANIPOWER": "17388",
    "ASTRAL": "14418", "AUBANK": "21238", "BANDHANBNK": "2263",
    "BANKBARODA": "4668", "BHARATFORG": "422", "BIOCON": "11373",
    "CANBK": "10794", "CHOLAFIN": "685", "COFORGE": "11543",
    "CUMMINSIND": "1901", "DMART": "19913", "FEDERALBNK": "1023",
    "GAIL": "4717", "GODREJCP": "10099", "HAL": "2303",
    "HAVELLS": "9819", "ICICIGI": "21770", "ICICIPRULI": "18652",
    "IDFCFIRSTB": "11184", "IEX": "220", "INDIGO": "11195",
    "IOC": "1624", "JUBLFOOD": "18096", "LUPIN": "10440",
    "MARICO": "4067", "MPHASIS": "4503", "PERSISTENT": "18365",
    "PIDILITIND": "2664", "SHRIRAMFIN": "4306", "SIEMENS": "3150",
    "TATAELXSI": "3411", "TATAPOWER": "3426", "TRENT": "1964",
    "ZYDUSLIFE": "7929",
    "MANKIND": "543904",
    # ── Momentum bot additions (resolved from scrip master 2026-06-22) ─────────
    "BDL": "541143",       # Bharat Dynamics Limited  [DEFENCE]
    "IRFC": "543257",      # Indian Railway Finance Co  [DEFENCE]
    "TVSMOTOR": "532343",  # TVS Motor Company  [AUTO]
}

TICK_SIZE_MAP: dict[str, float] = {
    # ── Existing 41 stocks ──────────────────────────────────────────────────
    "AUROPHARMA": 0.05, "BANKINDIA": 0.05, "PNB": 0.05,
    "KAYNES": 0.05, "KFINTECH": 0.05, "SAMMAANCAP": 0.05,
    "BHEL": 0.05, "NAUKRI": 0.10, "NBCC": 0.05,
    "POLICYBZR": 0.05, "DELHIVERY": 0.05, "PPLPHARMA": 0.05,
    "UPL": 0.05, "WAAREEENER": 0.10, "MCX": 0.10,
    "POLYCAB": 0.05, "TATATECH": 0.05, "VMM": 0.05,
    "M&M": 0.10, "ABCAPITAL": 0.05, "MOTHERSON": 0.05,
    "HINDZINC": 0.05, "COLPAL": 0.10, "NESTLEIND": 0.10,
    "BAJFINANCE": 0.05, "BPCL": 0.05, "AMBUJACEM": 0.05,
    "HUDCO": 0.05, "RVNL": 0.05, "YESBANK": 0.05,
    "SAIL": 0.05, "BOSCHLTD": 0.10, "PGEL": 0.10,
    "INFY": 0.10, "NHPC": 0.05, "TORNTPOWER": 0.05,
    "DIXON": 0.10, "DABUR": 0.05, "MANAPPURAM": 0.05,
    "HDFCAMC": 0.10, "HDFCLIFE": 0.05,
    # ── FNO Universe ────────────────────────────────────────────────────────
    "RELIANCE": 0.05, "TCS": 0.05, "HDFCBANK": 0.05, "ICICIBANK": 0.05,
    "KOTAKBANK": 0.05, "AXISBANK": 0.05, "SBIN": 0.05,
    "TATASTEEL": 0.05, "WIPRO": 0.05, "HCLTECH": 0.05,
    "SUNPHARMA": 0.05, "DRREDDY": 0.05, "MARUTI": 0.05,
    "ASIANPAINT": 0.05, "TITAN": 0.05, "ADANIPORTS": 0.05,
    "LTIM": 0.05, "TECHM": 0.05, "POWERGRID": 0.05,
    "NTPC": 0.05, "ONGC": 0.05, "COALINDIA": 0.05,
    "JSWSTEEL": 0.05, "HINDALCO": 0.05, "VEDL": 0.05,
    "BAJAJFINSV": 0.05,
    # ── Nifty 50 Additional Constituents ────────────────────────────────────
    "ADANIENT": 0.05, "APOLLOHOSP": 0.05, "BAJAJ-AUTO": 0.05, "BEL": 0.05,
    "BHARTIARTL": 0.05, "BRITANNIA": 0.05, "CIPLA": 0.05, "DIVISLAB": 0.05,
    "GRASIM": 0.05, "HEROMOTOCO": 0.05, "HINDUNILVR": 0.05, "INDUSINDBK": 0.05,
    "ITC": 0.05, "LT": 0.05, "SBILIFE": 0.05, "TATACONSUM": 0.05,
    "ULTRACEMCO": 0.05, "ZOMATO": 0.05, "ETERNAL": 0.05, "TATAMOTORS": 0.05,
    "TMCV": 0.05, "TMPV": 0.05, "LTM": 0.05,
    # ── Liquid ETFs ─────────────────────────────────────────────────────────
    "NIFTYBEES": 0.05, "BANKBEES": 0.05, "ITBEES": 0.05,
    "GOLDBEES": 0.05, "SILVERBEES": 0.05, "JUNIORBEES": 0.05,
    # ── Momentum bot additions ───────────────────────────────────────────────
    "BDL": 0.05, "IRFC": 0.05, "TVSMOTOR": 0.05,
}


def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to nearest valid tick value (e.g., tick=5 -> 1145 -> 1145, 1142.3 -> 1140)."""
    return round(price / tick_size) * tick_size

_INTERVAL_MAP = {"1minute": 1, "3minute": 3, "5minute": 5, "10minute": 10,
                 "15minute": 15, "30minute": 30, "60minute": 60}

_RESAMPLE_MAP = {1: "1min", 3: "3min", 5: "5min", 10: "10min",
                 15: "15min", 30: "30min", 60: "60min"}


class DhanStockTradingBot:
    def __init__(self):
        self.client_id = os.getenv("DHAN_CLIENT_ID")
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN")
        self._dhan_ctx = DhanContext(self.client_id, self.access_token)
        self.dhan = dhanhq(self._dhan_ctx)
        self.security_ids = {**FNO_UNIVERSE, **ETF_LIQUID, **FILTERED_FNO_UNIVERSE, **VWAP_RECLAIM_STOCKS, **NIFTY50_UNIVERSE}
        self.historical_cache = {}  # key -> (timestamp, df)
        self.historical_cache_ttl = 120  # seconds (default / 3-minute bars)
        # Higher timeframes barely change between 3-min scans — no point
        # refetching 15m/1h history every cycle for every stock.
        self.historical_cache_ttl_by_interval = {
            "3minute": 120,
            "15minute": 600,
            "60minute": 1800,
            "1day": 3600,
        }
        self.live_quotes_cache = {}  # key -> (timestamp, qdata)
        self.live_quotes_cache_ttl = 10  # seconds
        # Index (IDX_I) push ticks are kept in a SEPARATE cache so index
        # security IDs (13, 25, 29, ...) can never collide with NSE equity IDs.
        self.index_quotes_cache = {}  # sid_str -> (timestamp, qdata)
        # Optional candle persistence: when set to a directory, every fetched
        # OHLCV frame is also written to <dir>/<date>/<sid>_<interval>.csv so
        # replay/backtests don't need to re-hit the API.
        self.data_store_dir = None
        # Circuit breaker: after N consecutive API failures, stop hammering
        # the API for a cooldown window instead of retrying every call.
        self.breaker_threshold = 8
        self.breaker_cooldown = 120  # seconds
        self._consec_failures = 0
        self._breaker_open_until = 0.0
        self.alert_cb = None  # optional, e.g. send_telegram
        # Calls run in asyncio.to_thread workers, so cross-thread state
        # (breaker counters, candle-store file writes) needs real locks.
        self._breaker_lock = threading.Lock()
        self._persist_lock = threading.Lock()
        # MarketFeed WebSocket (push-based live quotes, replaces REST polling)
        self._feed = None
        self._feed_thread = None

    # ── MarketFeed WebSocket ─────────────────────────────────────────────────

    def start_live_feed(self, security_ids: list, exchange: int = None):
        """Start a MarketFeed Quote WebSocket in a background daemon thread.

        After this call, live_quotes_cache is kept fresh by push ticks and
        cache_live_quotes() / fetch_live_data_multi() become no-ops — all
        callers automatically get sub-second prices from cache.

        Args:
            security_ids: List of integer or string security IDs to subscribe.
            exchange:     MarketFeed exchange constant (default: MarketFeed.NSE = 1).
        """
        from dhanhq import MarketFeed

        if self._feed is not None:
            logger.warning("start_live_feed: feed already running; use subscribe_live() to add symbols.")
            return

        if exchange is None:
            exchange = MarketFeed.NSE

        instruments = [
            (exchange, str(int(sid)), MarketFeed.Quote)
            for sid in security_ids if sid
        ]
        if not instruments:
            logger.warning("start_live_feed: no valid security_ids provided.")
            return

        def _on_message(feed, data):
            if not isinstance(data, dict):
                return
            tick_type = data.get("type")
            if tick_type not in ("Quote Data", "Ticker Data"):
                return
            sid = str(data.get("security_id", ""))
            if not sid:
                return
            def _f(v):
                try: return float(v)
                except (TypeError, ValueError): return 0.0
            if tick_type == "Quote Data":
                qdata = {
                    "last_price": _f(data.get("LTP")),
                    "high_price": _f(data.get("high")),
                    "low_price": _f(data.get("low")),
                    "volume": int(data.get("volume") or 0),
                }
            else:
                qdata = {"last_price": _f(data.get("LTP")), "high_price": 0.0, "low_price": 0.0, "volume": 0}
            # exchange_segment 0 == IDX_I (indices) -> route to the index cache so
            # index IDs don't overwrite an equity with the same numeric ID.
            if data.get("exchange_segment") == 0:
                self.index_quotes_cache[sid] = (time.time(), qdata)
            else:
                self.live_quotes_cache[sid] = (time.time(), qdata)

        def _on_error(feed, exc):
            logger.error("MarketFeed error: %s", exc)

        def _on_close(feed):
            logger.info("MarketFeed connection closed")

        self._feed = MarketFeed(
            self._dhan_ctx, instruments, version="v2",
            on_message=_on_message, on_error=_on_error, on_close=_on_close,
        )
        self._feed_thread = self._feed.start()
        logger.info("MarketFeed started: %d instruments subscribed", len(instruments))

    def subscribe_live(self, security_ids: list, exchange: int = None):
        """Add more symbols to a running MarketFeed without restarting it."""
        from dhanhq import MarketFeed
        if self._feed is None:
            self.start_live_feed(security_ids, exchange)
            return
        if exchange is None:
            exchange = MarketFeed.NSE
        new_instruments = [
            (exchange, str(int(sid)), MarketFeed.Quote)
            for sid in security_ids if sid
        ]
        if new_instruments:
            self._feed.subscribe_symbols(new_instruments)
            logger.debug("MarketFeed: subscribed %d more instruments", len(new_instruments))

    def stop_live_feed(self):
        """Close the MarketFeed WebSocket connection."""
        if self._feed is not None:
            try:
                self._feed.close_connection()
            except Exception as exc:
                logger.warning("stop_live_feed error: %s", exc)
            finally:
                self._feed = None
                self._feed_thread = None
            logger.info("MarketFeed stopped")

    def is_feed_active(self) -> bool:
        return self._feed is not None and self._feed_thread is not None and self._feed_thread.is_alive()

    def get_index_ltps_from_feed(self, security_ids: list, max_stale: float = 180.0) -> dict:
        """Return {sid_str: ltp} for index IDs present and fresh in the push-feed
        cache. Empty dict when the feed is inactive or an index hasn't ticked yet
        — the caller then falls back to the REST quote endpoint.
        """
        if not self.is_feed_active():
            return {}
        out = {}
        now = time.time()
        for sid in security_ids:
            sid_str = str(sid)
            entry = self.index_quotes_cache.get(sid_str)
            if entry and now - entry[0] < max_stale:
                ltp = entry[1].get("last_price", 0) or 0
                if ltp > 0:
                    out[sid_str] = ltp
        return out

    # ── Circuit breaker ──────────────────────────────────────────────────────

    def _breaker_is_open(self) -> bool:
        return time.time() < self._breaker_open_until

    def _record_api_success(self):
        with self._breaker_lock:
            self._consec_failures = 0

    def _record_api_failure(self, context: str):
        with self._breaker_lock:
            self._consec_failures += 1
            tripped = self._consec_failures >= self.breaker_threshold
            if tripped:
                self._breaker_open_until = time.time() + self.breaker_cooldown
                self._consec_failures = 0
        if tripped:
            logger.error("Dhan circuit breaker OPEN after %d consecutive failures (%s). "
                         "Pausing API calls for %ds.", self.breaker_threshold, context,
                         self.breaker_cooldown)
            if self.alert_cb:
                try:
                    self.alert_cb(f"DHAN API CIRCUIT BREAKER OPEN ({context}). "
                                  f"Pausing calls for {self.breaker_cooldown}s.")
                except Exception:
                    logger.exception("Alert callback failed")

    def clear_historical_cache(self, intervals=None):
        """Clear cached candles. With intervals (e.g. ("3minute",)), only
        those timeframes are dropped — higher TFs expire via their TTLs."""
        if intervals is None:
            self.historical_cache.clear()
            return
        for key in [k for k in self.historical_cache if k[1] in intervals]:
            del self.historical_cache[key]

    def clear_live_quotes_cache(self):
        if not self.is_feed_active():
            self.live_quotes_cache.clear()

    def _fetch_intraday(self, security_id, date_str, interval_int, exchange_segment=None):
        return self._fetch_intraday_range(security_id, date_str, date_str, interval_int, exchange_segment=exchange_segment)

    def _fetch_intraday_range(self, security_id, from_date, to_date, interval_int, retries=2, exchange_segment=None):
        if exchange_segment is None:
            exchange_segment = self.dhan.NSE
        if self._breaker_is_open():
            return None
        for attempt in range(retries + 1):
            time.sleep(0.3)
            try:
                resp = self.dhan.intraday_minute_data(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument_type="EQUITY",
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval_int,
                )
            except Exception as e:
                logger.warning("intraday_minute_data raised for %s: %s", security_id, e)
                resp = None
            if isinstance(resp, dict) and resp.get("status") == "success":
                self._record_api_success()
                break

            # Classify error type from remarks or top-level response
            is_rate_limit = False
            is_auth_error = False
            remarks = ""
            if isinstance(resp, dict):
                remarks = resp.get("remarks", "")
                if isinstance(remarks, dict):
                    error_type = remarks.get("error_type", "")
                    error_code = remarks.get("error_code", "")
                    if error_type == "Rate_Limit" or error_code == "DH-904":
                        is_rate_limit = True
                    if error_type == "Invalid_Authentication" or error_code == "DH-901":
                        is_auth_error = True
                elif "Rate_Limit" in str(remarks) or "DH-904" in str(remarks):
                    is_rate_limit = True
                elif "Invalid_Authentication" in str(remarks) or "DH-901" in str(remarks):
                    is_auth_error = True

                if resp.get("error_type") == "Rate_Limit" or resp.get("error_code") == "DH-904":
                    is_rate_limit = True
                if resp.get("error_type") == "Invalid_Authentication" or resp.get("error_code") == "DH-901":
                    is_auth_error = True

            if attempt == retries:
                logger.warning("API error for %s (int=%d, final attempt, rate_limit=%s, auth_error=%s): %s",
                               security_id, interval_int, is_rate_limit, is_auth_error,
                               resp.get("remarks", resp) if isinstance(resp, dict) else resp)
            else:
                logger.debug("API error for %s (int=%d, attempt=%d, rate_limit=%s, auth_error=%s): %s",
                               security_id, interval_int, attempt, is_rate_limit, is_auth_error,
                               resp.get("remarks", resp) if isinstance(resp, dict) else resp)

            # Auth errors: allow one retry (transient server hiccup) then abort
            if is_auth_error:
                if attempt == 0:
                    logger.debug("Auth error for %s — retrying once after 3s (may be transient)...", security_id)
                    time.sleep(3.0)
                    continue
                else:
                    logger.error("Auth error persists for %s — check DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN. Skipping.", security_id)
                    return None

            if attempt < retries:
                if is_rate_limit:
                    logger.debug("Rate limit hit. Backing off for %ds...", 3 * (attempt + 1))
                    time.sleep(3.0 * (attempt + 1))
                else:
                    time.sleep(1.5)
        else:
            self._record_api_failure(f"intraday data {security_id}")
            return None
        
        # Hard throttle: force a 0.2s delay on successful API calls to prevent concurrency bursts
        time.sleep(0.2)
        
        data = resp.get("data", {})
        if not data or "open" not in data or not data["open"]:
            return None
        opens = data["open"]
        if not opens or len(opens) == 0:
            return None
        df = pd.DataFrame({
            "open": opens, "high": data.get("high", []),
            "low": data.get("low", []), "close": data.get("close", []),
            "volume": data.get("volume", []),
        })
        if "timestamp" in data and data["timestamp"]:
            df["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
            df.set_index("timestamp", inplace=True)
        return df

    def _prev_trading_day(self):
        t = datetime.now()
        offset = 3 if t.weekday() == 0 else 1
        return (t - timedelta(days=offset)).strftime("%Y-%m-%d")

    def _prev_n_trading_days(self, n: int) -> list:
        """Return date strings for the last n trading days (Mon-Fri), most-recent first."""
        days = []
        t = datetime.now()
        offset = 1
        while len(days) < n:
            d = t - timedelta(days=offset)
            if d.weekday() < 5:
                days.append(d.strftime("%Y-%m-%d"))
            offset += 1
        return days

    def get_kronos_history(self, security_id, interval="15minute", n_days=7, exchange_segment=None):
        """Fetch N trading days of intraday OHLCV for the Kronos context window.

        Tries a single multi-day range call first; falls back to day-by-day
        fetches if the range call fails. Cached for 5 minutes independently
        of the normal trading-data cache.
        """
        cache_key = (str(security_id), f"kronos_{interval}_{n_days}")
        now = time.time()
        if cache_key in self.historical_cache:
            ts, df = self.historical_cache[cache_key]
            if now - ts < 300 and df is not None and len(df) > 0:
                return df

        interval_int = _INTERVAL_MAP.get(interval, 15)
        today_str = datetime.now().strftime("%Y-%m-%d")
        prev_days = self._prev_n_trading_days(n_days - 1)
        from_date = prev_days[-1] if prev_days else today_str

        dfs = []
        range_df = self._fetch_intraday_range(
            security_id, from_date, today_str, interval_int,
            exchange_segment=exchange_segment,
        )
        if range_df is not None and len(range_df) > 0:
            dfs.append(range_df)
        else:
            for day_str in [today_str] + prev_days:
                day_df = self._fetch_intraday(security_id, day_str, interval_int,
                                              exchange_segment=exchange_segment)
                if day_df is not None and len(day_df) > 0:
                    dfs.append(day_df)

        if not dfs:
            result = pd.DataFrame()
        else:
            result = pd.concat(dfs)
            result = result[~result.index.duplicated(keep="last")]
            result.sort_index(inplace=True)

        self.historical_cache[cache_key] = (now, result)
        return result

    def get_historical_data(self, security_id, interval="3minute", min_bars=5, exchange_segment=None):
        # Key on (sid, interval) only: the bot (min_bars=20) and guardian
        # (min_bars=5) previously kept separate entries for identical data,
        # doubling API fetches for every symbol with an open position. A
        # cached frame serves any caller it has enough rows for; an empty
        # frame is a fresh negative result (failed fetch) and is returned
        # as-is rather than hammering the API.
        cache_key = (str(security_id), interval)
        now = time.time()
        ttl = self.historical_cache_ttl_by_interval.get(interval, self.historical_cache_ttl)
        if cache_key in self.historical_cache:
            ts, df = self.historical_cache[cache_key]
            if now - ts < ttl and (df is None or len(df) == 0 or len(df) >= min_bars):
                return df

        df = self._get_historical_data_uncached(security_id, interval, min_bars, exchange_segment=exchange_segment)
        self.historical_cache[cache_key] = (now, df)
        if interval != "1day":  # daily bars aren't useful in the replay store
            self._persist_candles(security_id, interval, df)
        return df

    def _persist_candles(self, security_id, interval, df):
        """Append fetched candles to the on-disk store, one file per
        (date, security, interval), deduplicated by timestamp."""
        if self.data_store_dir is None or df is None or len(df) == 0:
            return
        if not hasattr(df.index, "date"):
            return
        try:
            # Lock: bot and guardian threads can fetch the same symbol
            # concurrently; unsynchronized read-modify-write corrupts the CSV.
            with self._persist_lock:
                for d, group in df.groupby(df.index.date):
                    day_dir = os.path.join(str(self.data_store_dir), str(d))
                    os.makedirs(day_dir, exist_ok=True)
                    path = os.path.join(day_dir, f"{security_id}_{interval}.csv")
                    if os.path.isfile(path):
                        old = pd.read_csv(path, index_col=0, parse_dates=True)
                        group = pd.concat([old, group])
                        group = group[~group.index.duplicated(keep="last")].sort_index()
                    group.to_csv(path)
        except Exception:
            logger.exception("Failed to persist candles for %s/%s", security_id, interval)

    def _get_historical_data_uncached(self, security_id, interval="3minute", min_bars=5, exchange_segment=None):
        time.sleep(random.uniform(0.1, 0.3))
        if interval == "1day":
            # "1day" used to silently fall through _INTERVAL_MAP to 3-minute
            # bars, so the ATR-floor profile was computed on intraday data.
            return self._fetch_daily_history(security_id, exchange_segment=exchange_segment)
        interval_int = _INTERVAL_MAP.get(interval, 3)
        resample_rule = _RESAMPLE_MAP.get(interval_int, "3min")
        today_str = time.strftime("%Y-%m-%d")

        today_df = self._fetch_intraday(security_id, today_str, interval_int, exchange_segment=exchange_segment)

        # If today alone satisfies min_bars, return immediately
        if today_df is not None and len(today_df) >= min_bars:
            return today_df

        dfs = [today_df] if today_df is not None and len(today_df) > 0 else []

        # Try resampling from 1-min data if native interval didn't give enough
        if interval_int != 1:
            today_1m = self._fetch_intraday(security_id, today_str, 1, exchange_segment=exchange_segment)
            if today_1m is not None and len(today_1m) > 0:
                resampled = today_1m.resample(resample_rule).agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
                dfs.append(resampled)

        # Fetch previous trading days until min_bars is met. Higher timeframes
        # yield few bars per session (60-minute ≈ 6/day), so SMA-20 / RSI-14 /
        # ADX-14 need several prior sessions of context. Fetching only ONE prior
        # day left the 1-hour frame with ~12 bars — below the gate and far below
        # the 20 SMA-20 needs — so indicators_1h was empty/degenerate and the 1h
        # trend logged NEUTRAL on every signal. The early-break leaves 3m/15m
        # (already satisfied by today's data) untouched.
        for prev in self._prev_n_trading_days(7):
            have = pd.concat(dfs) if dfs else None
            if have is not None and len(have[~have.index.duplicated()]) >= min_bars:
                break
            prev_df = self._fetch_intraday(security_id, prev, interval_int, exchange_segment=exchange_segment)
            if prev_df is not None and len(prev_df) > 0:
                dfs.append(prev_df)
            elif interval_int != 1:
                # Fallback: fetch that day's 1-min bars and resample
                prev_1m = self._fetch_intraday(security_id, prev, 1, exchange_segment=exchange_segment)
                if prev_1m is not None and len(prev_1m) > 0:
                    resampled = prev_1m.resample(resample_rule).agg({
                        "open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum",
                    }).dropna()
                    dfs.append(resampled)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined

    def _fetch_daily_history(self, security_id, calendar_days=120, exchange_segment=None):
        """Fetch daily OHLCV bars (~80 trading days for 120 calendar days)."""
        if exchange_segment is None:
            exchange_segment = self.dhan.NSE
        if self._breaker_is_open():
            return pd.DataFrame()
        end = datetime.now()
        start = end - timedelta(days=calendar_days)
        try:
            resp = self.dhan.historical_daily_data(
                security_id=str(security_id),
                exchange_segment=exchange_segment,
                instrument_type="EQUITY",
                from_date=start.strftime("%Y-%m-%d"),
                to_date=end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            logger.warning("historical_daily_data raised for %s: %s", security_id, e)
            resp = None
        if not isinstance(resp, dict) or resp.get("status") != "success":
            self._record_api_failure(f"daily history {security_id}")
            return pd.DataFrame()
        self._record_api_success()
        data = resp.get("data", {})
        if not data or not data.get("open"):
            return pd.DataFrame()
        df = pd.DataFrame({
            "open": data["open"], "high": data.get("high", []),
            "low": data.get("low", []), "close": data.get("close", []),
            "volume": data.get("volume", []),
        })
        if "timestamp" in data and data["timestamp"]:
            df["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
            df.set_index("timestamp", inplace=True)
        return df

    def place_equity_order(self, security_id, transaction_type, quantity,
                           order_type="MARKET", product_type="INTRA"):
        return self.dhan.place_order(
            security_id=security_id,
            exchange_segment=self.dhan.NSE,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=0,
        )

    def reduce_position(self, security_id, transaction_type, quantity, product_type="INTRA"):
        """Place a market order to reduce an existing position by `quantity` shares."""
        return self.place_equity_order(security_id, transaction_type, quantity, 
                                       order_type="MARKET", product_type=product_type)

    def fetch_live_data_multi(self, security_ids: list):
        """Return live quotes for multiple security IDs.

        When MarketFeed is active, reads directly from the push-populated cache
        (no REST call, no rate-limit exposure). Falls back to chunked quote_data()
        REST calls when the feed is not running.

        Returns dict: str(security_id) -> {last_price, high_price, low_price, volume}
        """
        if self.is_feed_active():
            results = {}
            for sid in security_ids:
                sid_str = str(sid)
                entry = self.live_quotes_cache.get(sid_str)
                if entry:
                    results[sid_str] = entry[1]
            return results

        # ── REST fallback (feed not running) ──────────────────────────────────
        int_ids = []
        for sid in security_ids:
            try:
                int_ids.append(int(sid))
            except (ValueError, TypeError):
                continue

        if not int_ids:
            return {}

        results = {}
        chunk_size = 50
        for i in range(0, len(int_ids), chunk_size):
            chunk = int_ids[i:i + chunk_size]
            if i > 0:
                time.sleep(0.3)
            try:
                resp = self.dhan.quote_data(securities={self.dhan.NSE: chunk})
                if isinstance(resp, dict) and resp.get("status") == "success":
                    data_dict = resp.get("data", {}).get("data", {})
                    for sid_int in chunk:
                        sid_str = str(sid_int)
                        found_data = {}
                        for segment, seg_data in data_dict.items():
                            if sid_str in seg_data:
                                found_data = seg_data[sid_str]
                                break
                        if found_data:
                            ohlc = found_data.get("ohlc", {})
                            results[sid_str] = {
                                "last_price": found_data.get("last_price") or 0.0,
                                "high_price": ohlc.get("high") or 0.0,
                                "low_price": ohlc.get("low") or 0.0,
                                "volume": found_data.get("volume") or 0,
                            }
            except Exception as e:
                logger.error("Error in fetch_live_data_multi for chunk %s: %s", chunk, e)

        return results

    def cache_live_quotes(self, security_ids: list):
        """Pre-populate live_quotes_cache for the given security IDs.

        No-op when MarketFeed is active — the feed already keeps the cache
        current via push ticks, so the REST bulk-fetch is unnecessary.
        """
        if self.is_feed_active():
            return
        quotes = self.fetch_live_data_multi(security_ids)
        now = time.time()
        for sid, qdata in quotes.items():
            self.live_quotes_cache[str(sid)] = (now, qdata)

    def fetch_live_data(self, security_id):
        sid_str = str(security_id)
        now = time.time()
        if sid_str in self.live_quotes_cache:
            ts, qdata = self.live_quotes_cache[sid_str]
            if now - ts < self.live_quotes_cache_ttl:
                return qdata

        # Fallback to single quote API call
        try:
            int_id = int(security_id)
            resp = self.dhan.quote_data(securities={self.dhan.NSE: [int_id]})
            if isinstance(resp, dict) and resp.get("status") == "success":
                data_dict = resp.get("data", {}).get("data", {})
                found_data = {}
                for segment, seg_data in data_dict.items():
                    if sid_str in seg_data:
                        found_data = seg_data[sid_str]
                        break
                if found_data:
                    ohlc = found_data.get("ohlc", {})
                    qdata = {
                        "last_price": found_data.get("last_price") or 0.0,
                        "high_price": ohlc.get("high") or 0.0,
                        "low_price": ohlc.get("low") or 0.0,
                        "volume": found_data.get("volume") or 0,
                    }
                    self.live_quotes_cache[sid_str] = (now, qdata)
                    return qdata
        except Exception as e:
            logger.error("Error in fallback fetch_live_data for %s: %s", security_id, e)
        return {}

    def get_available_balance(self):
        resp = self.dhan.get_fund_limits()
        if isinstance(resp, dict) and resp.get("status") == "success":
            bal = float(resp["data"].get("availabelBalance", 0))
            if bal == 0:
                import time
                time.sleep(1)
                resp = self.dhan.get_fund_limits()
                if isinstance(resp, dict) and resp.get("status") == "success":
                    bal = float(resp["data"].get("availabelBalance", 0))
            return bal
        return 0.0

    def get_positions(self):
        return self.dhan.get_positions()

    def fetch_positions(self):
        """Get current open positions from Dhan API.
        Returns dict: symbol -> {entry_price, quantity, transaction_type, unrealized_pnl, security_id}
        """
        if self._breaker_is_open():
            return {}
        resp = None
        for attempt in range(5):
            try:
                resp = self.dhan.get_positions()
            except Exception as e:
                logger.warning("get_positions raised: %s", e)
                resp = None
            if isinstance(resp, dict) and resp.get("status") == "success":
                self._record_api_success()
                break
            remarks = resp.get("remarks", resp) if isinstance(resp, dict) else resp

            # Dhan often misreports rate limits as DH-901 or DH-906.
            # Only log as warning on the final attempt to reduce console spam.
            if attempt == 4:
                logger.warning("fetch_positions API error (final attempt): %s", remarks)
            else:
                logger.debug("fetch_positions API error (attempt %d/5): %s", attempt + 1, remarks)
                # Exponential backoff: 4s, 8s, 12s, 16s
                time.sleep(4 * (attempt + 1))
        else:
            self._record_api_failure("fetch_positions")
            return {}

        data = resp.get("data", [])
        if not data:
            return {}

        # Build reverse map: security_id -> symbol
        sid_to_symbol = {sid: sym for sym, sid in self.security_ids.items()}

        positions = {}
        for pos in data:
            try:
                security_id = str(pos.get("securityId", ""))
                net_qty = int(pos.get("netQty", 0))  # >0 = LONG, <0 = SHORT
                if net_qty == 0:
                    continue

                symbol = sid_to_symbol.get(security_id)
                if not symbol:
                    # Fallback: BSE-listed or off-watchlist positions report tradingSymbol
                    trading_symbol = str(pos.get("tradingSymbol") or "").strip()
                    if not trading_symbol:
                        logger.debug("Skipping position with unknown security_id %s", security_id)
                        continue
                    symbol = trading_symbol

                exchange_segment = str(pos.get("exchangeSegment") or self.dhan.NSE)
                is_buy = net_qty > 0
                # netAvg is the correct average cost; fall back to buyAvg/sellAvg
                # if netAvg is missing or zero (can happen for externally-placed orders).
                net_avg = float(pos.get("netAvg", 0))
                if net_avg == 0:
                    net_avg = float(pos.get("buyAvg", 0) if is_buy else pos.get("sellAvg", 0))
                entry_price = net_avg
                quantity = abs(net_qty)
                unrealized_pnl = float(pos.get("unrealizedProfit", 0))
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning("Malformed position entry skipped: %s (%s)", pos, e)
                continue

            positions[symbol] = {
                "symbol": symbol,
                "entry_price": entry_price,
                "quantity": quantity,
                "transaction_type": self.dhan.BUY if is_buy else self.dhan.SELL,
                "unrealized_pnl": unrealized_pnl,
                "security_id": security_id,
                "exchange_segment": exchange_segment,
            }

        return positions

    def get_tick_size(self, symbol: str) -> float:
        return TICK_SIZE_MAP.get(symbol, 5.0)

    def place_super_order(self, security_id, transaction_type, quantity,
                          entry_price, sl_percent, target_percent, symbol=None, atr_value=None):
        tick = self.get_tick_size(symbol) if symbol else 0.05
        is_buy = transaction_type == self.dhan.BUY
        sl_raw = entry_price * (1 - sl_percent / 100) if is_buy else entry_price * (1 + sl_percent / 100)
        target_raw = entry_price * (1 + target_percent / 100) if is_buy else entry_price * (1 - target_percent / 100)
        sl_price = round_to_tick(sl_raw, tick)
        target_price = round_to_tick(target_raw, tick)

        # Calculate trailing jump equal to ATR rounded to nearest tick size
        if atr_value is not None and atr_value > 0:
            trailing_jump = round_to_tick(atr_value, tick)
        else:
            trailing_jump = 0.0

        # Bypass the SDK's place_super_order method to avoid strict price validation
        # for MARKET orders (which requires price > 0, leading to 'Invalid Price for orderType')
        payload = {
            "transactionType": transaction_type.upper(),
            "exchangeSegment": self.dhan.NSE.upper(),
            "productType": self.dhan.INTRA.upper(),  # Must be INTRADAY for Super Orders (not BO)
            "orderType": self.dhan.MARKET.upper(),
            "securityId": str(security_id),
            "quantity": int(quantity),
            "price": None,  # Must be null/None for MARKET orders
            "targetPrice": float(target_price),
            "stopLossPrice": float(sl_price),
            "trailingJump": float(trailing_jump)
        }
        return self.dhan.dhan_http.post('/super/orders', payload)
