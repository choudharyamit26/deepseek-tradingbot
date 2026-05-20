import os
import time
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from dhanhq import dhanhq, DhanContext
import logging

logger = logging.getLogger(__name__)

load_dotenv()

try:
    from constant import FNO_UNIVERSE, ETF_LIQUID, FILTERED_FNO_UNIVERSE
except ImportError:
    FNO_UNIVERSE = {}
    ETF_LIQUID = {}
    FILTERED_FNO_UNIVERSE = {}

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
    # ── Liquid ETFs ─────────────────────────────────────────────────────────
    "NIFTYBEES": 0.05, "BANKBEES": 0.05, "ITBEES": 0.05,
    "GOLDBEES": 0.05, "SILVERBEES": 0.05, "JUNIORBEES": 0.05,
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
        self.dhan = dhanhq(DhanContext(self.client_id, self.access_token))
        self.security_ids = {**FNO_UNIVERSE, **ETF_LIQUID, **FILTERED_FNO_UNIVERSE, **VWAP_RECLAIM_STOCKS}

    def _fetch_intraday(self, security_id, date_str, interval_int):
        return self._fetch_intraday_range(security_id, date_str, date_str, interval_int)

    def _fetch_intraday_range(self, security_id, from_date, to_date, interval_int, retries=2):
        for attempt in range(retries + 1):
            resp = self.dhan.intraday_minute_data(
                security_id=security_id,
                exchange_segment=self.dhan.NSE,
                instrument_type="EQUITY",
                from_date=from_date,
                to_date=to_date,
                interval=interval_int,
            )
            if isinstance(resp, dict) and resp.get("status") == "success":
                break
            if attempt < retries:
                time.sleep(1)
        else:
            return None
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

    def get_historical_data(self, security_id, interval="3minute", min_bars=5):
        interval_int = _INTERVAL_MAP.get(interval, 3)
        resample_rule = _RESAMPLE_MAP.get(interval_int, "3min")
        today_str = time.strftime("%Y-%m-%d")

        today_df = self._fetch_intraday(security_id, today_str, interval_int)
        if today_df is not None and len(today_df) >= min_bars:
            return today_df

        dfs = [today_df] if today_df is not None and len(today_df) > 0 else []

        today_1m = self._fetch_intraday(security_id, today_str, 1)
        if today_1m is not None and len(today_1m) > 0:
            resampled = today_1m.resample(resample_rule).agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            dfs.append(resampled)

        prev = self._prev_trading_day()
        prev_1m = self._fetch_intraday(security_id, prev, 1)
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

    def fetch_live_data(self, security_id):
        resp = self.dhan.quote_data(securities={self.dhan.NSE: [security_id]})
        if isinstance(resp, dict):
            data = resp.get(str(security_id), {})
            return {
                "last_price": data.get("LTP"),
                "high_price": data.get("high"),
                "low_price": data.get("low"),
                "volume": data.get("volume"),
            }
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

    def get_tick_size(self, symbol: str) -> float:
        return TICK_SIZE_MAP.get(symbol, 5.0)

    def place_super_order(self, security_id, transaction_type, quantity,
                          entry_price, sl_percent, target_percent, symbol=None):
        tick = self.get_tick_size(symbol) if symbol else 0.05
        is_buy = transaction_type == self.dhan.BUY
        sl_raw = entry_price * (1 - sl_percent / 100) if is_buy else entry_price * (1 + sl_percent / 100)
        target_raw = entry_price * (1 + target_percent / 100) if is_buy else entry_price * (1 - target_percent / 100)
        sl_price = round_to_tick(sl_raw, tick)
        target_price = round_to_tick(target_raw, tick)

        return self.dhan.place_super_order(
            security_id=security_id,
            exchange_segment=self.dhan.NSE,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=self.dhan.MARKET,
            product_type=self.dhan.BO,
            price=entry_price,
            stopLossPrice=sl_price,
            targetPrice=target_price,
        )
