import logging
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
    "INFY": NIFTY_IT, "TCS": NIFTY_IT, "NAUKRI": NIFTY_IT,
    "HDFCBANK": BANKNIFTY, "SBIN": BANKNIFTY, "BANKINDIA": NIFTY_PSU_BANK,
    "PNB": NIFTY_PSU_BANK, "YESBANK": BANKNIFTY,
    "BAJFINANCE": NIFTY_FINSRV, "KFINTECH": NIFTY_FINSRV,
    "SAMMAANCAP": NIFTY_FINSRV, "POLICYBZR": NIFTY_FINSRV,
    "MCX": NIFTY_FINSRV, "ABCAPITAL": NIFTY_FINSRV,
    "MANAPPURAM": NIFTY_FINSRV, "HDFCAMC": NIFTY_FINSRV,
    "HDFCLIFE": NIFTY_FINSRV,
    "M&M": NIFTY_AUTO, "MOTHERSON": NIFTY_AUTO, "BOSCHLTD": NIFTY_AUTO,
    "AUROPHARMA": NIFTY_PHARMA, "PPLPHARMA": NIFTY_PHARMA,
    "COLPAL": NIFTY_FMCG, "NESTLEIND": NIFTY_FMCG, "DABUR": NIFTY_FMCG,
    "HINDZINC": NIFTY_METAL, "SAIL": NIFTY_METAL,
    "BPCL": NIFTY_OIL_GAS,
    "BHEL": NIFTY_ENERGY, "TORNTPOWER": NIFTY_ENERGY,
    "NBCC": NIFTY_REALTY,
    "DIXON": NIFTY_CONSR_DURBL,
    "RELIANCE": NIFTY_OIL_GAS,
    "TATATECH": NIFTY_IT,
    "KAYNES": MIDCPNIFTY,
    "UPL": NIFTY_OIL_GAS,
    "WAAREEENER": NIFTY_ENERGY,
    "POLYCAB": NIFTY_ENERGY,
    "VMM": NIFTY_FMCG,
    "DELHIVERY": NIFTY_IT,
    "AMBUJACEM": NIFTY_INFRA,
    "HUDCO": NIFTY_FINSRV,
    "RVNL": NIFTY_INFRA,
    "PGEL": NIFTY_ENERGY,
    "NHPC": NIFTY_ENERGY,
}


class RegimeFilter:
    def __init__(self, dhan_api):
        self.dhan = dhan_api

    REGIME_LOOKBACK_TRADING_DAYS = 15  # ~3 weeks calendar for intraday regime
    REGIME_SMA_WINDOW = 10  # 10-day SMA (~2 trading weeks)

    def _fetch_daily(self, security_id: str) -> pd.DataFrame:
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

    def _calc_regime(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 5:
            return {"trend": "neutral", "volatility": "normal", "strength": 0,
                    "current": 0, "sma": 0, "sma_window": self.REGIME_SMA_WINDOW,
                    "volatility_pct": 0}
        close = df["close"]
        current = close.iloc[-1]
        sma = close.rolling(window=self.REGIME_SMA_WINDOW, min_periods=5).mean().iloc[-1]
        atr14 = (df["high"] - df["low"]).rolling(window=14, min_periods=7).mean().iloc[-1]
        atr_pct = (atr14 / current * 100) if current > 0 else 0
        trend = "bullish" if current > sma * 1.005 else ("bearish" if current < sma * 0.995 else "neutral")
        vol = "high" if atr_pct > 2.0 else ("low" if atr_pct < 0.5 else "normal")
        strength = round(((current / sma) - 1) * 100, 2)
        vol_pct = round(atr_pct, 2)
        return {"trend": trend, "volatility": vol, "strength": strength, "volatility_pct": vol_pct,
                "current": round(current, 2), "sma": round(sma, 2), "sma_window": self.REGIME_SMA_WINDOW}

    def get_regime(self, symbol: str) -> dict:
        nifty_df = self._fetch_daily(NIFTY_50)
        nifty = self._calc_regime(nifty_df)
        sector_id = STOCK_SECTOR_MAP.get(symbol)
        sector = {}
        sector_name = ""
        if sector_id:
            sector_df = self._fetch_daily(sector_id)
            sector = self._calc_regime(sector_df)
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
        lines.append(f"Nifty 50: trend={n['trend']}, volatility={n['volatility']}, "
                     f"strength={n['strength']}%, current={n['current']}, SMA{n['sma_window']}={n['sma']}")
        if reg["sector"]:
            s = reg["sector"]
            lines.append(f"{reg['sector_name']}: trend={s['trend']}, volatility={s['volatility']}, "
                         f"strength={s['strength']}%, current={s['current']}, SMA{s['sma_window']}={s['sma']}")
        return "\n".join(lines)
