"""
dry_run_test.py -- Test full pipeline: fetch -> indicators -> AI signal -> CSV -> Telegram.
Runs a subset of stocks (fast) with Telegram notifications enabled.
"""
import asyncio
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import sys
import logging
import time
import requests as _requests
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhan_integration import DhanStockTradingBot
from deepseek_analyzer import DeepSeekStockAnalyzer
from stock_trading_bot import IntradayStockBot
from risk_manager import RiskManager
from signal_logger import SignalLogger, LOG_DIR
from regime_filter import RegimeFilter

IST = ZoneInfo("Asia/Kolkata")
DRY_RUN = True

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-16s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "dry_run_test.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("DRY_TEST")

WATCHLIST = [
    "INFY", "RELIANCE", "TCS", "HDFCBANK", "SBIN",
    "BAJFINANCE", "M&M", "TATATECH", "KAYNES", "YESBANK",
]

signals_generated = 0


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
        log.warning("Telegram send failed: %s", exc)


def format_signal_msg(symbol: str, tag: str, direction: str, quantity: int,
                      price: float, sl_price: float = 0.0,
                      tp1_price: float = 0.0, tp2_price: float = 0.0) -> str:
    emoji = {
        "ENTRY-LONG": "\U0001f7e2", "ENTRY-SHORT": "\U0001f534",
        "EXIT": "\U0001f504", "SQUAREOFF": "\u23f0",
        "ORDER-PLACED": "\u2705",
    }.get(tag, "\U0001f4ca")
    now_str = datetime.now(IST).strftime("%H:%M:%S")
    lines = [
        f"{emoji} <b>{tag}</b>  [DRY-RUN]",
        f"<b>{symbol}</b>  {direction} x {quantity}",
        f"Price: \u20b9{price:,.2f}",
    ]
    if sl_price > 0:
        lines.append(f"SL: \u20b9{sl_price:,.2f}")
    if tp1_price > 0:
        lines.append(f"TP1: \u20b9{tp1_price:,.2f}")
    lines.append(f"Time: {now_str} IST")
    return "\n".join(lines)


async def test_stock(symbol: str, dhan: DhanStockTradingBot,
                     analyzer: DeepSeekStockAnalyzer, risk: RiskManager,
                     signal_log: SignalLogger, api_sem: asyncio.Semaphore,
                     regime: RegimeFilter) -> None:
    global signals_generated

    async with api_sem:
        security_id = dhan.security_ids.get(symbol)
        if not security_id:
            log.warning("  [SKIP] %s  --  no security ID", symbol)
            return

        try:
            historical = dhan.get_historical_data(security_id, "3minute")
        except Exception as e:
            log.warning("  [API ERR] %s  --  %s", symbol, e)
            return

        if len(historical) < 20:
            log.info("  [SKIP] %s  --  only %d bars", symbol, len(historical))
            return

        log.info("  [DATA] %s  --  %d bars (%.2f-%.2f)",
                 symbol, len(historical),
                 historical["low"].min(), historical["high"].max())

        bot = IntradayStockBot(dhan, analyzer, risk, [symbol], dry_run=True)
        indicators = bot.calculate_technical_indicators(historical)
        if not indicators:
            log.info("  [SKIP] %s  --  no indicators", symbol)
            return

        live = dhan.fetch_live_data(security_id)
        ltp = live.get("last_price") or historical["close"].iloc[-1]
        market_data = {
            "ltp": ltp,
            "high_3m": live.get("high_price") or historical["high"].iloc[-1],
            "low_3m": live.get("low_price") or historical["low"].iloc[-1],
            "volume": live.get("volume") or historical["volume"].iloc[-1],
            "avg_volume_3m": historical["volume"].tail(5).mean(),
        }

        regime_context = regime.format_regime_context(symbol)
        log.info("  [REGIME] %s -- nifty=%s, sector=%s",
                 symbol, regime_context.split("Nifty")[1].split("Sector")[0].strip() if "Sector" in regime_context else regime_context,
                 regime_context.split("Sector", 1)[1].strip() if "Sector" in regime_context else "none")
        # Pass recent bars for candle-history context (improvement #5)
        recent_bars = historical.tail(10) if len(historical) >= 10 else historical
        try:
            signal = analyzer.get_trading_signal(symbol, market_data, indicators,
                                                  regime_context, recent_bars=recent_bars)
        except Exception as e:
            log.error("  [AI ERR] %s  --  %s", symbol, e)
            return
        sig_type = signal.get("signal", "HOLD")
        confidence = signal.get("confidence", 0)
        reasoning = signal.get("reasoning", "")

        log.info("  [SIGNAL] %s -> %s (confidence=%d, reason=%s)",
                 symbol, sig_type, confidence, reasoning[:80] if reasoning else "none")

        if sig_type in ("BUY", "SELL") and confidence >= risk.min_confidence:
            sl_percent = signal.get("stop_loss_percent", 1.5)
            target_percent = signal.get("target_percent", 3.0)
            quantity = risk.calculate_position_size(risk.current_capital, sl_percent, ltp)

            if quantity >= 1:
                signals_generated += 1
                sl_price = ltp * (1 - sl_percent / 100) if sig_type == "BUY" else ltp * (1 + sl_percent / 100)
                target_price = ltp * (1 + target_percent / 100) if sig_type == "BUY" else ltp * (1 - target_percent / 100)
                tag = "ENTRY-LONG" if sig_type == "BUY" else "ENTRY-SHORT"

                signal_log.log_signal(
                    symbol=symbol, signal_type=tag, direction=sig_type,
                    entry_price=ltp, quantity=quantity, stop_loss=sl_price,
                    target=target_price, confidence=confidence,
                    reasoning=reasoning, mode="DRY-RUN",
                )

                msg = format_signal_msg(symbol, tag, sig_type, quantity, ltp, sl_price=sl_price)
                send_telegram(msg)
                log.info("  >>> %s %s x%d @ %.2f | SL=%.2f | Target=%.2f | Telegram sent",
                         symbol, sig_type, quantity, ltp, sl_price, target_price)
        else:
            log.info("  [NO TRADE] %s  --  signal=%s confidence=%d (min=%d)",
                     symbol, sig_type, confidence, risk.min_confidence)


async def main():
    log.info("=" * 72)
    log.info("  DRY RUN TEST  --  Full Pipeline Validation")
    log.info("  Stocks : %d (subset for quick test)", len(WATCHLIST))
    log.info("  Date   : %s", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
    log.info("  Mode   : DRY-RUN (no real orders)")
    log.info("  Telegram : %s", "ENABLED" if TELEGRAM_ENABLED else "DISABLED")
    log.info("=" * 72)

    dhan = DhanStockTradingBot()
    analyzer = DeepSeekStockAnalyzer(api_key=os.getenv("DEEPSEEK_API_KEY"))
    risk = RiskManager(dhan_api=dhan, max_daily_trades=5,
                       max_daily_loss_percent=2, risk_per_trade_percent=2,
                       min_confidence=75)  # Raised from 65 to match production
    signal_log = SignalLogger()
    regime = RegimeFilter(dhan)
    api_sem = asyncio.Semaphore(3)

    if TELEGRAM_ENABLED:
        send_telegram(
            f"<b>Dry Run Test Started</b>  [DRY-RUN]\n"
            f"Testing {len(WATCHLIST)} stocks | 3-min timeframe | Full pipeline"
        )

    start = time.perf_counter()
    tasks = [test_stock(s, dhan, analyzer, risk, signal_log, api_sem, regime) for s in WATCHLIST]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    log.info("=" * 72)
    log.info("  RESULTS")
    log.info("  Tested             : %d stocks", len(WATCHLIST))
    log.info("  Signals generated  : %d", signals_generated)
    log.info("  Time elapsed       : %.1fs", elapsed)
    log.info("  Signal CSV         : %s",
             signal_log._csv_path(datetime.now(IST).strftime("%Y-%m-%d")))
    log.info("=" * 72)

    summary = (
        f"[DRY RUN COMPLETE] Stocks: {len(WATCHLIST)}, Signals: {signals_generated}, Time: {elapsed:.0f}s"
    )
    print(summary)
    if TELEGRAM_ENABLED:
        send_telegram(summary)


if __name__ == "__main__":
    asyncio.run(main())
