from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dhan credentials ─────────────────────────────────────────────────────────
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# ── Mode ─────────────────────────────────────────────────────────────────────
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# ── Kronos model settings ────────────────────────────────────────────────────
KRONOS_MODEL = "NeoQuasar/Kronos-base"  # or Kronos-base / Kronos-mini
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_MAX_CONTEXT = 512
KRONOS_DEVICE = "cuda"

# ── Kronos integration params (tunable by self-improvement agent) ────────────
KRONOS_ENABLED = True
KRONOS_PRED_LEN = 10  # candles to forecast
KRONOS_LOOKBACK = 200  # historical bars for prediction
KRONOS_TEMPERATURE = 0.5  # sampling temperature
KRONOS_SAMPLE_COUNT = 5  # ensemble paths to average
KRONOS_TOP_P = 0.9
KRONOS_CONFIDENCE_WEIGHT = 0.35  # how much Kronos opinion affects final conf
KRONOS_PENALTY_CONFLICT = 0.50  # multiplier when Kronos contradicts signal
KRONOS_BONUS_ALIGN = 1.10  # multiplier when Kronos agrees with signal
KRONOS_EXIT_THRESHOLD = -0.008  # predicted return below this → tighten exit
KRONOS_MIN_PREDICTED_MOVE = 0.003  # minimum predicted move to override SL

# ── Bot trading params ───────────────────────────────────────────────────────
FIRST_ENTRY_HOUR, FIRST_ENTRY_MIN = 9, 30
LAST_ENTRY_HOUR, LAST_ENTRY_MIN = 15, 0
MAX_SIGNALS_PER_STOCK_PER_DAY = 1  # one entry per stock per day
MAX_CONCURRENT_POSITIONS = 3
SCAN_INTERVAL = 180  # seconds between scan cycles
COOLDOWN_SECONDS = 1800  # 30-min cooldown per stock after a signal
MIN_CONFIDENCE = 80  # raised floor — reject borderline setups
MIN_RR_RATIO = 1.8
STOP_LOSS_ATR_MULTIPLIER = 1.5
MIN_STOP_LOSS_PCT = 0.25
MIN_ADX_TRENDING = 18
MIN_PREFILTER_VOLUME_RATIO = 0.15
MIN_PREFILTER_ATR_PCT = 0.30
MIN_VOLUME_RATIO_TRENDING = 0.40  # even trending stocks need minimum volume
RISK_PER_TRADE_PCT = 2.0
MAX_DAILY_TRADES = 5
MAX_DAILY_LOSS_PCT = 2.0
MAX_DAILY_SIGNALS = 20  # hard cap on total entry signals per day
SAME_DIRECTION_COOLDOWN = 3600  # 1-hour block for repeat same-direction signal
RSI_OB_LIMIT = 70
RSI_OS_LIMIT = 30

# ── Intraday-specific params (tunable by self-improvement agent) ────────────
TRAILING_SL_ACTIVATION_PCT = 3.0   # Profit % at which trailing SL activates
TRAILING_SL_DISTANCE_ATR = 2.0     # How many ATRs to keep trail behind price
MAX_TRADE_DURATION_MINUTES = 180   # Max time a position can stay open
MARKET_OPEN_SKIP_MINUTES = 0       # Skip N min after market open before entries
MARKET_CLOSE_EXIT_MINUTES = 15     # Exit all positions N min before market close
MAX_CONSECUTIVE_LOSSES = 3         # Stop trading after N consecutive losses
PARTIAL_PROFIT_PCT = 0.0           # % of position to book at first target (0=off)
POSITION_CONFIDENCE_SCALAR = 1.0   # Position size multiplier when confidence>=85
MAX_DAILY_TRADES = 5               # Max trades per day
RISK_PER_TRADE_PCT = 2.0           # % of capital risked per trade
CASH_BUFFER_PCT = 20.0             # % of capital always kept undeployed
MAX_POSITION_CAPITAL_PCT = 100.0   # max % of capital in a single position (100 = no cap)

# ── Strategy params for self-improvement ─────────────────────────────────────
STRATEGY_FILE = Path(__file__).parent / "kronos_strategy.yaml"
STATE_DIR = Path(__file__).parent / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Watchlist ────────────────────────────────────────────────────────────────
WATCHLIST = [
    # ── Nifty 50 Core (29) ──────────────────────────────────────────────
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "BAJFINANCE",
    "KOTAKBANK",
    "AXISBANK",
    "LT",
    "TATASTEEL",
    "WIPRO",
    "HCLTECH",
    "SUNPHARMA",
    "DRREDDY",
    "MARUTI",
    "ADANIPORTS",
    "M&M",
    "NTPC",
    "POWERGRID",
    "ONGC",
    "COALINDIA",
    "JSWSTEEL",
    "HINDALCO",
    "TITAN",
    "ASIANPAINT",
    "BAJAJFINSV",
    "TECHM",
    "LTIM",
    # ── Additional Nifty 50 (21) ─────────────────────────────────────────
    "BHARTIARTL",
    "ITC",
    "HINDUNILVR",
    "TATAMOTORS",
    "ULTRACEMCO",
    "BAJAJ-AUTO",
    "BRITANNIA",
    "NESTLEIND",
    "TRENT",
    "APOLLOHOSP",
    "CIPLA",
    "DIVISLAB",
    "EICHERMOT",
    "GRASIM",
    "HEROMOTOCO",
    "TATACONSUM",
    "BPCL",
    "BEL",
    "HDFCLIFE",
    "SBILIFE",
    "INDUSINDBK",
    # ── Banking & Financials (14) ────────────────────────────────────────
    "PNB",
    "BANKBARODA",
    "CANBK",
    "IDFCFIRSTB",
    "FEDERALBNK",
    "BANDHANBNK",
    "YESBANK",
    "SHRIRAMFIN",
    "ICICIPRULI",
    "AUBANK",
    "CHOLAFIN",
    "ICICIGI",
    # ── Pharma & Healthcare (4) ──────────────────────────────────────────
    "LUPIN",
    "ZYDUSLIFE",
    "BIOCON",
    # ── Auto & Industrials (8) ──────────────────────────────────────────
    "HAVELLS",
    "MARICO",
    "MOTHERSON",
    "BHARATFORG",
    "TATAPOWER",
    "TORNTPOWER",
    "SIEMENS",
    "CUMMINSIND",
    # ── Energy & Commodities (9) ─────────────────────────────────────────
    "VEDL",
    "IEX",
    "ADANIENT",
    "ADANIGREEN",
    "ADANIPOWER",
    "MANKIND",
    "GAIL",
    "IOC",
    "HINDZINC",
    # ── Consumer & FMCG (6) ──────────────────────────────────────────────
    "DABUR",
    "COLPAL",
    "GODREJCP",
    "JUBLFOOD",
    "PIDILITIND",
    # ── Tech & New Economy (6) ───────────────────────────────────────────
    "ZOMATO",
    "DMART",
    "PERSISTENT",
    "COFORGE",
    "INDIGO",
    "TATAELXSI",
    "MPHASIS",
    "ASTRAL",
    "AMBUJACEM",
    "ABCAPITAL",
    "DIXON",
    "POLYCAB",
    "HAL",
]
