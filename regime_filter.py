import logging
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# NSE index security IDs (from dhanhq security list)
NIFTY_50 = "13"           # NIFTY
BANKNIFTY = "25"          # BANKNIFTY
FINNIFTY = "27"           # FINNIFTY
NIFTY_IT = "29"           # NIFTYIT
NIFTY_PHARMA = "32"       # NIFTY PHARMA
NIFTY_AUTO = "14"         # NIFTY AUTO
NIFTY_FMCG = "28"         # NIFTY FMCG
NIFTY_METAL = "31"        # NIFTY METAL
NIFTY_OIL_GAS = "470"     # NIFTY OIL AND GAS
NIFTY_PSU_BANK = "33"     # NIFTY PSU BANK
NIFTY_PVT_BANK = "15"     # NIFTY PVT BANK
NIFTY_REALTY = "34"       # NIFTY REALTY
NIFTY_MEDIA = "30"        # NIFTY MEDIA
NIFTY_HEALTHCARE = "447"  # NIFTY HEALTHCARE
NIFTY_CONSR_DURBL = "466" # NIFTY CONSR DURBL
NIFTY_FINSRV = "469"      # NIFTY FINSRV25 50
NIFTY_ENERGY = "42"       # NIFTY ENERGY
NIFTY_INFRA = "43"        # NIFTY INFRA
MIDCPNIFTY = "442"        # MIDCPNIFTY

SECTOR_NAMES: dict[str, str] = {
    NIFTY_50: "NIFTY 50",
    BANKNIFTY: "BANK NIFTY",
    FINNIFTY: "FIN NIFTY",
    NIFTY_IT: "NIFTY IT",
    NIFTY_PHARMA: "NIFTY PHARMA",
    NIFTY_AUTO: "NIFTY AUTO",
    NIFTY_FMCG: "NIFTY FMCG",
    NIFTY_METAL: "NIFTY METAL",
    NIFTY_OIL_GAS: "NIFTY OIL & GAS",
    NIFTY_PSU_BANK: "NIFTY PSU BANK",
    NIFTY_PVT_BANK: "NIFTY PVT BANK",
    NIFTY_REALTY: "NIFTY REALTY",
    NIFTY_MEDIA: "NIFTY MEDIA",
    NIFTY_HEALTHCARE: "NIFTY HEALTHCARE",
    NIFTY_CONSR_DURBL: "NIFTY CONSUMER DURABLES",
    NIFTY_FINSRV: "NIFTY FINANCIAL SERVICES",
    NIFTY_ENERGY: "NIFTY ENERGY",
    NIFTY_INFRA: "NIFTY INFRA",
    MIDCPNIFTY: "NIFTY MIDCAP 100",
}

STOCK_SECTOR_MAP = {
    # ── IT / Tech ────────────────────────────────────────────────────────────
    "INFY": NIFTY_IT, "TCS": NIFTY_IT, "NAUKRI": NIFTY_IT,
    "HCLTECH": NIFTY_IT, "WIPRO": NIFTY_IT, "TECHM": NIFTY_IT,
    "LTIM": NIFTY_IT, "LTM": NIFTY_IT, "TATATECH": NIFTY_IT,
    "DELHIVERY": NIFTY_IT, "COFORGE": NIFTY_IT, "MPHASIS": NIFTY_IT,
    "PERSISTENT": NIFTY_IT, "TATAELXSI": NIFTY_IT,

    # ── Banks (Bank Nifty) ───────────────────────────────────────────────────
    "HDFCBANK": BANKNIFTY, "SBIN": BANKNIFTY, "ICICIBANK": BANKNIFTY,
    "KOTAKBANK": BANKNIFTY, "AXISBANK": BANKNIFTY, "INDUSINDBK": BANKNIFTY,
    "YESBANK": BANKNIFTY, "AUBANK": BANKNIFTY, "BANDHANBNK": BANKNIFTY,
    "FEDERALBNK": BANKNIFTY, "IDFCFIRSTB": BANKNIFTY,

    # ── PSU Banks ────────────────────────────────────────────────────────────
    "BANKINDIA": NIFTY_PSU_BANK, "PNB": NIFTY_PSU_BANK,
    "BANKBARODA": NIFTY_PSU_BANK, "CANBK": NIFTY_PSU_BANK,

    # ── Financial Services ───────────────────────────────────────────────────
    "BAJFINANCE": NIFTY_FINSRV, "BAJAJFINSV": NIFTY_FINSRV,
    "KFINTECH": NIFTY_FINSRV, "SAMMAANCAP": NIFTY_FINSRV,
    "POLICYBZR": NIFTY_FINSRV, "MCX": NIFTY_FINSRV,
    "ABCAPITAL": NIFTY_FINSRV, "MANAPPURAM": NIFTY_FINSRV,
    "HDFCAMC": NIFTY_FINSRV, "HDFCLIFE": NIFTY_FINSRV,
    "HUDCO": NIFTY_FINSRV, "SBILIFE": NIFTY_FINSRV,
    "CHOLAFIN": NIFTY_FINSRV, "SHRIRAMFIN": NIFTY_FINSRV,
    "ICICIGI": NIFTY_FINSRV, "ICICIPRULI": NIFTY_FINSRV,
    "IEX": NIFTY_FINSRV,

    # ── Auto ─────────────────────────────────────────────────────────────────
    "M&M": NIFTY_AUTO, "MOTHERSON": NIFTY_AUTO, "BOSCHLTD": NIFTY_AUTO,
    "MARUTI": NIFTY_AUTO, "TATAMOTORS": NIFTY_AUTO, "TMCV": NIFTY_AUTO,
    "TMPV": NIFTY_AUTO, "BAJAJ-AUTO": NIFTY_AUTO, "EICHERMOT": NIFTY_AUTO,
    "HEROMOTOCO": NIFTY_AUTO, "BHARATFORG": NIFTY_AUTO,

    # ── Pharma / Healthcare ──────────────────────────────────────────────────
    "AUROPHARMA": NIFTY_PHARMA, "PPLPHARMA": NIFTY_PHARMA,
    "SUNPHARMA": NIFTY_PHARMA, "DRREDDY": NIFTY_PHARMA,
    "CIPLA": NIFTY_PHARMA, "DIVISLAB": NIFTY_PHARMA,
    "LUPIN": NIFTY_PHARMA, "BIOCON": NIFTY_PHARMA,
    "ZYDUSLIFE": NIFTY_PHARMA, "MANKIND": NIFTY_PHARMA,
    "APOLLOHOSP": NIFTY_HEALTHCARE,

    # ── FMCG ─────────────────────────────────────────────────────────────────
    "COLPAL": NIFTY_FMCG, "NESTLEIND": NIFTY_FMCG, "DABUR": NIFTY_FMCG,
    "VMM": NIFTY_FMCG, "HINDUNILVR": NIFTY_FMCG, "ITC": NIFTY_FMCG,
    "BRITANNIA": NIFTY_FMCG, "MARICO": NIFTY_FMCG,
    "GODREJCP": NIFTY_FMCG, "TATACONSUM": NIFTY_FMCG,
    "DMART": NIFTY_FMCG, "JUBLFOOD": NIFTY_FMCG,

    # ── Metal ────────────────────────────────────────────────────────────────
    "HINDZINC": NIFTY_METAL, "SAIL": NIFTY_METAL,
    "TATASTEEL": NIFTY_METAL, "JSWSTEEL": NIFTY_METAL,
    "HINDALCO": NIFTY_METAL, "VEDL": NIFTY_METAL,

    # ── Oil & Gas ────────────────────────────────────────────────────────────
    "BPCL": NIFTY_OIL_GAS, "RELIANCE": NIFTY_OIL_GAS,
    "ONGC": NIFTY_OIL_GAS, "IOC": NIFTY_OIL_GAS,
    "GAIL": NIFTY_OIL_GAS,

    # ── Energy / Power ───────────────────────────────────────────────────────
    "BHEL": NIFTY_ENERGY, "TORNTPOWER": NIFTY_ENERGY,
    "WAAREEENER": NIFTY_ENERGY, "POLYCAB": NIFTY_ENERGY,
    "PGEL": NIFTY_ENERGY, "NHPC": NIFTY_ENERGY,
    "NTPC": NIFTY_ENERGY, "POWERGRID": NIFTY_ENERGY,
    "COALINDIA": NIFTY_ENERGY, "TATAPOWER": NIFTY_ENERGY,
    "ADANIGREEN": NIFTY_ENERGY, "ADANIPOWER": NIFTY_ENERGY,

    # ── Infra / Cement / Realty ──────────────────────────────────────────────
    "AMBUJACEM": NIFTY_INFRA, "RVNL": NIFTY_INFRA,
    "NBCC": NIFTY_REALTY, "LT": NIFTY_INFRA,
    "ULTRACEMCO": NIFTY_INFRA, "GRASIM": NIFTY_INFRA,
    "PIDILITIND": NIFTY_INFRA, "SIEMENS": NIFTY_INFRA,
    "CUMMINSIND": NIFTY_INFRA,

    # ── Consumer Durables ────────────────────────────────────────────────────
    "DIXON": NIFTY_CONSR_DURBL, "TITAN": NIFTY_CONSR_DURBL,
    "HAVELLS": NIFTY_CONSR_DURBL, "ASIANPAINT": NIFTY_CONSR_DURBL,
    "ASTRAL": NIFTY_CONSR_DURBL,

    # ── Telecom / Conglomerate ───────────────────────────────────────────────
    "BHARTIARTL": NIFTY_IT,  # Nifty IT constituent
    "ADANIENT": NIFTY_INFRA, "ADANIPORTS": NIFTY_INFRA,

    # ── Miscellaneous / Midcap ───────────────────────────────────────────────
    "KAYNES": MIDCPNIFTY, "UPL": NIFTY_OIL_GAS,
    "INDIGO": MIDCPNIFTY, "HAL": MIDCPNIFTY,
    "BEL": MIDCPNIFTY, "TRENT": NIFTY_FMCG,
    "ZOMATO": MIDCPNIFTY, "ETERNAL": MIDCPNIFTY,

    # ── Liquid ETFs (map to their underlying index) ──────────────────────────
    "NIFTYBEES": NIFTY_50, "BANKBEES": BANKNIFTY,
    "ITBEES": NIFTY_IT, "GOLDBEES": NIFTY_50,
    "SILVERBEES": NIFTY_50, "JUNIORBEES": NIFTY_50,
}


class RegimeFilter:
    def __init__(self, dhan_api):
        self.dhan = dhan_api
        # get_regime() is called once per stock per scan; without caching the
        # same Nifty/sector daily history was refetched ~100x per cycle.
        self._daily_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._live_price_cache: dict[str, tuple[float, float]] = {}
        self._cache_lock = threading.Lock()

    REGIME_LOOKBACK_TRADING_DAYS = 15  # ~3 weeks calendar for intraday regime
    REGIME_SMA_WINDOW = 10  # 10-day SMA (~2 trading weeks)
    DAILY_CACHE_TTL = 300   # daily index bars barely change intraday
    LIVE_PRICE_TTL = 60     # index LTP for trend-vs-SMA comparison

    def _fetch_daily(self, security_id: str) -> pd.DataFrame:
        now = time.time()
        with self._cache_lock:
            cached = self._daily_cache.get(security_id)
            if cached and now - cached[0] < self.DAILY_CACHE_TTL:
                return cached[1]
        df = self._fetch_daily_uncached(security_id)
        with self._cache_lock:
            self._daily_cache[security_id] = (now, df)
        return df

    def _fetch_daily_uncached(self, security_id: str) -> pd.DataFrame:
        end = datetime.now(IST)
        start = end - timedelta(days=self.REGIME_LOOKBACK_TRADING_DAYS * 2)
        resp = self.dhan.dhan.historical_daily_data(
            security_id=security_id,
            exchange_segment=self.dhan.dhan.INDEX,
            instrument_type="INDEX",
            from_date=start.strftime("%Y-%m-%d"),
            to_date=end.strftime("%Y-%m-%d"),
        )
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return pd.DataFrame()
        data = resp.get("data", {})
        if not data or "close" not in data or not data["close"]:
            return pd.DataFrame()
        ohlc = data.get("open", []), data.get("high", []), data.get("low", []), data.get("close", []), data.get("volume", [])
        df = pd.DataFrame({"open": ohlc[0], "high": ohlc[1], "low": ohlc[2], "close": ohlc[3], "volume": ohlc[4]})
        if "timestamp" in data and data["timestamp"]:
            df["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
            df.set_index("timestamp", inplace=True)
        return df

    def _fetch_live_index_prices(self, security_ids: list[str]) -> dict[str, float]:
        """Fetch live index LTPs via IDX_I segment, cached. Returns {security_id: ltp}.

        Indices absent from the quote response are negatively cached (price 0)
        so a permanently-missing index doesn't force a refetch on every call.
        """
        now = time.time()
        with self._cache_lock:
            entries = {sid: self._live_price_cache.get(sid) for sid in security_ids}
        if all(e is not None and now - e[0] < self.LIVE_PRICE_TTL for e in entries.values()):
            return {sid: e[1] for sid, e in entries.items() if e[1] > 0}
        fresh = self._fetch_live_index_prices_uncached(security_ids)
        with self._cache_lock:
            for sid in security_ids:
                self._live_price_cache[sid] = (now, fresh.get(sid, 0.0))
        return fresh

    def _fetch_live_index_prices_uncached(self, security_ids: list[str]) -> dict[str, float]:
        int_ids = []
        for sid in security_ids:
            try:
                int_ids.append(int(sid))
            except (ValueError, TypeError):
                continue
        if not int_ids:
            return {}
        try:
            resp = self.dhan.dhan.quote_data(securities={"IDX_I": int_ids})
            if isinstance(resp, dict) and resp.get("status") == "success":
                idx_data = resp.get("data", {}).get("data", {}).get("IDX_I", {})
                result = {}
                for sid in security_ids:
                    if sid in idx_data:
                        ltp = idx_data[sid].get("last_price", 0)
                        if ltp > 0:
                            result[sid] = ltp
                return result
        except Exception as e:
            logger.warning("Failed to fetch live index prices: %s", e)
        return {}

    def _calc_regime(self, df: pd.DataFrame, live_price: float = 0) -> dict:
        if df.empty or len(df) < 5:
            return {"trend": "neutral", "volatility": "normal", "strength": 0,
                    "current": 0, "sma": 0, "sma_window": self.REGIME_SMA_WINDOW,
                    "volatility_pct": 0, "intraday_chg_pct": 0, "session_trend": "neutral"}
        close = df["close"]
        # Use live price if available, otherwise fall back to last daily close
        current = live_price if live_price > 0 else close.iloc[-1]
        sma = close.rolling(window=self.REGIME_SMA_WINDOW, min_periods=5).mean().iloc[-1]
        atr14 = (df["high"] - df["low"]).rolling(window=14, min_periods=7).mean().iloc[-1]
        atr_pct = (atr14 / current * 100) if current > 0 else 0
        trend = "bullish" if current > sma * 1.005 else ("bearish" if current < sma * 0.995 else "neutral")
        vol = "high" if atr_pct > 2.0 else ("low" if atr_pct < 0.5 else "normal")
        strength = round(((current / sma) - 1) * 100, 2)
        vol_pct = round(atr_pct, 2)
        # Intraday movement: live price vs yesterday's (last completed) daily close
        prev_close = close.iloc[-1]
        intraday_chg_pct = round((current - prev_close) / prev_close * 100, 2) if live_price > 0 and prev_close > 0 else 0
        session_trend = "bullish" if intraday_chg_pct >= 0.5 else ("bearish" if intraday_chg_pct <= -0.5 else "neutral")
        return {"trend": trend, "volatility": vol, "strength": strength, "volatility_pct": vol_pct,
                "current": round(current, 2), "sma": round(sma, 2), "sma_window": self.REGIME_SMA_WINDOW,
                "intraday_chg_pct": intraday_chg_pct, "session_trend": session_trend}

    def get_regime(self, symbol: str) -> dict:
        # Determine which indices we need
        sector_id = STOCK_SECTOR_MAP.get(symbol)
        ids_to_fetch = [NIFTY_50]
        if sector_id:
            ids_to_fetch.append(sector_id)

        # Fetch live prices for all needed indices in one API call
        live_prices = self._fetch_live_index_prices(ids_to_fetch)

        nifty_df = self._fetch_daily(NIFTY_50)
        nifty = self._calc_regime(nifty_df, live_price=live_prices.get(NIFTY_50, 0))

        sector = {}
        sector_name = ""
        if sector_id:
            sector_df = self._fetch_daily(sector_id)
            sector = self._calc_regime(sector_df, live_price=live_prices.get(sector_id, 0))
            sector_name = SECTOR_NAMES.get(sector_id, f"IDX-{sector_id}")

        return {
            "nifty": nifty,
            "sector": sector or None,
            "sector_name": sector_name,
        }

    def format_regime_context(self, symbol: str, regime_data: dict | None = None) -> str:
        reg = regime_data if regime_data else self.get_regime(symbol)
        lines = []
        n = reg["nifty"]
        intraday_str = f", intraday_chg={n['intraday_chg_pct']:+.2f}% (session={n['session_trend']})" if n.get("intraday_chg_pct") else ""
        lines.append(f"Nifty 50: trend={n['trend']}, volatility={n['volatility']}, "
                     f"strength={n['strength']}%, current={n['current']}, SMA{n['sma_window']}={n['sma']}{intraday_str}")
        if reg["sector"]:
            s = reg["sector"]
            lines.append(f"{reg['sector_name']}: trend={s['trend']}, volatility={s['volatility']}, "
                         f"strength={s['strength']}%, current={s['current']}, SMA{s['sma_window']}={s['sma']}")
        return "\n".join(lines)

