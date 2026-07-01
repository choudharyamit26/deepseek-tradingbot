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

# ── "Let winners run" exit policy ─────────────────────────────────────────────
# 1-month post-mortem: planned R:R 2.0 collapsed to 0.81 realized because
# KRONOS-EXIT market-closed winners at ~+0.10% avg (only 6% of wins reached
# target). KRONOS-EXIT's value is CUTTING LOSERS (+4.0% net from flat/losing
# trades); its harm is decapitating winners. This policy keeps the loss-cutting
# and lets winners ride: once a trade is in real profit, a *modest* Kronos
# reversal only tightens a trailing stop instead of full-exiting. Full exit on a
# winner requires a STRONG reversal (urgency >= hard-exit) or the trail hitting.
KRONOS_LET_WINNERS_RUN = True       # master switch (flip false to restore old harvest behavior)
KRONOS_RUN_PROFIT_PCT = 0.5         # profit % above which a trade is "running" — protect, don't harvest
KRONOS_HARD_EXIT_URGENCY = 90       # even a runner full-exits at/above this Kronos urgency
KRONOS_RUN_TRAIL_ATR = 1.5          # trail distance (in ATRs) applied when locking a runner's profit

# ── Bot trading params ───────────────────────────────────────────────────────
FIRST_ENTRY_HOUR, FIRST_ENTRY_MIN = 9, 30
LAST_ENTRY_HOUR, LAST_ENTRY_MIN = 15, 0
MAX_SIGNALS_PER_STOCK_PER_DAY = 1  # one entry per stock per day
MAX_CONCURRENT_POSITIONS = 3
SCAN_INTERVAL = 180  # seconds between scan cycles
COOLDOWN_SECONDS = 1800  # 30-min cooldown per stock after a signal
MIN_CONFIDENCE = 80  # raised floor — reject borderline setups
MIN_RR_RATIO = 1.8
BUY_ENABLED = True  # master kill-switch for long entries (BUY is the worst
# performer historically: 28.6% WR, -87 of -133 total pnl, confidence
# anti-predictive). Left enabled; flip to false to suppress all BUYs.
STOP_LOSS_ATR_MULTIPLIER = 1.5
MIN_STOP_LOSS_PCT = 0.25
MAX_STOP_LOSS_PCT = 1.0  # hard ceiling on per-trade stop %; caps the fat loss tail from wide ATR stops on high-ATR names (0 = no cap)
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
MIN_RSI_FOR_SHORT = 35  # block SELLs with 3m RSI below this (RSI<35 shorts had payoff 0.58; RSI 35-45 zone is the only profitable bucket, payoff 1.41)

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
CASH_BUFFER_PCT = 20.0             # % of buying power always kept undeployed
LEVERAGE = 5.0                     # intraday MIS margin multiplier (5x = Rs1000 -> Rs5000 buying power)
MAX_POSITION_CAPITAL_PCT = 25.0    # max % of buying power in a single position (25% of 5x = 1.25x cash)

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
    # ── High-beta additions (2026-07-01) ─────────────────────────────────
    # Routinely move 2-4% intraday so the trade's move clears the ~0.15%
    # roundtrip cost. Old universe averaged only 0.41% move/trade (too small
    # to beat costs — see entry-features-nondiscriminative post-mortem).
    "PFC",
    "RECLTD",
    "NATIONALUM",
    "NMDC",
    "HINDCOPPER",
    "IREDA",
    "PAYTM",
    "JIOFIN",
    "GMRAIRPORT",
    "KALYANKJIL",
    "OFSS",
]
