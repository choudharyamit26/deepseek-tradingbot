import asyncio
import os
import sys
import logging
import requests as _requests
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from dhan_integration import DhanStockTradingBot
from deepseek_analyzer import DeepSeekStockAnalyzer
from stock_trading_bot import IntradayStockBot
from risk_manager import RiskManager
from signal_logger import SignalLogger, LOG_DIR

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def setup_date_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"bot_{today}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(name)-16s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logging.getLogger().info("Logging to %s", log_path)
    return fh


def send_telegram(message: str) -> None:
    if not TELEGRAM_ENABLED:
        return
    try:
        _requests.post(
            _TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN),
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as exc:
        logging.getLogger("TELEGRAM").warning("Telegram send failed: %s", exc)


def format_signal_msg(symbol: str, tag: str, direction: str, quantity: int,
                      price: float, sl_price: float = 0.0,
                      tp1_price: float = 0.0, tp2_price: float = 0.0) -> str:
    emoji = {
        "ENTRY-LONG": "\U0001f7e2", "ENTRY-SHORT": "\U0001f534",
        "TP1-LONG": "\U0001f3af", "TP1-SHORT": "\U0001f3af",
        "TP2-LONG": "\U0001f4b0", "TP2-SHORT": "\U0001f4b0",
        "SL-LONG": "\U0001f6d1", "SL-SHORT": "\U0001f6d1",
        "EXIT": "\U0001f504", "SQUAREOFF": "\u23f0",
        "ORDER-PLACED": "\u2705",
    }.get(tag, "\U0001f4ca")

    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    now_str = datetime.now(IST).strftime("%H:%M:%S")
    lines = [
        f"{emoji} <b>{tag}</b>  [{mode}]",
        f"<b>{symbol}</b>  {direction} x {quantity}",
        f"Price: \u20b9{price:,.2f}",
    ]
    if sl_price > 0:
        lines.append(f"SL: \u20b9{sl_price:,.2f}")
    if tp1_price > 0:
        lines.append(f"TP1: \u20b9{tp1_price:,.2f}")
    if tp2_price > 0:
        lines.append(f"TP2: \u20b9{tp2_price:,.2f}")
    lines.append(f"Time: {now_str} IST")
    return "\n".join(lines)


async def main():
    _ = setup_date_logging()
    log = logging.getLogger("MAIN")

    dhan_bot = DhanStockTradingBot()
    ai_analyzer = DeepSeekStockAnalyzer(api_key=os.getenv("DEEPSEEK_API_KEY"))
    risk_mgr = RiskManager(
        dhan_api=dhan_bot, max_daily_trades=5,
        max_daily_loss_percent=2, risk_per_trade_percent=2,
        min_confidence=75,  # Raised from 65 — eliminates floor-riding signals
    )

    watchlist = [
        # ── Core 41 stocks (existing) ───────────────────────────────────────
        "AUROPHARMA", "BANKINDIA", "PNB", "KAYNES", "KFINTECH",
        "SAMMAANCAP", "BHEL", "NAUKRI", "NBCC", "POLICYBZR",
        "DELHIVERY", "PPLPHARMA", "UPL", "WAAREEENER", "MCX",
        "POLYCAB", "TATATECH", "VMM", "M&M", "ABCAPITAL",
        "MOTHERSON", "HINDZINC", "COLPAL", "NESTLEIND", "BAJFINANCE",
        "BPCL", "AMBUJACEM", "HUDCO", "RVNL", "YESBANK",
        "SAIL", "BOSCHLTD", "PGEL", "INFY", "NHPC",
        "TORNTPOWER", "DIXON", "DABUR", "MANAPPURAM", "HDFCAMC",
        "HDFCLIFE",
        # ── FNO Universe (liquid large-caps) ────────────────────────────────
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "KOTAKBANK",
        "AXISBANK", "SBIN", "TATASTEEL", "WIPRO", "HCLTECH",
        "SUNPHARMA", "DRREDDY", "MARUTI", "ASIANPAINT", "TITAN",
        "ADANIPORTS", "LTIM", "TECHM", "POWERGRID", "NTPC",
        "ONGC", "COALINDIA", "JSWSTEEL", "HINDALCO", "VEDL",
        "BAJAJFINSV",
        # ── Liquid ETFs ─────────────────────────────────────────────────────
        "NIFTYBEES", "BANKBEES", "ITBEES", "GOLDBEES", "SILVERBEES",
        "JUNIORBEES",
    ]
    log.info("Watchlist: %d stocks (41 core + 26 FNO + 6 ETFs)", len(watchlist))

    if TELEGRAM_ENABLED:
        mode_str = "DRY-RUN" if DRY_RUN else "LIVE"
        send_telegram(
            f"\U0001f4e1 <b>Trading Bot Started</b>  [{mode_str}]\n"
            f"Watchlist: {len(watchlist)} stocks | Max 2 signals/stock/day\n"
            f"Entry window: 9:45 AM - 3:00 PM IST | Min confidence: 75"
        )

    bot = IntradayStockBot(
        dhan_bot, ai_analyzer, risk_mgr, watchlist,
        send_telegram=send_telegram, format_signal_msg=format_signal_msg,
        enable_telegram=TELEGRAM_ENABLED, dry_run=DRY_RUN,
    )
    await bot.run()

    if TELEGRAM_ENABLED:
        send_telegram("\U0001f6d1 <b>Trading Bot Stopped</b>")


if __name__ == "__main__":
    asyncio.run(main())
