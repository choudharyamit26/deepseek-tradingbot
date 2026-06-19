#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

from dhan_integration import DhanStockTradingBot
from deepseek_analyzer import DeepSeekStockAnalyzer
from risk_manager import RiskManager
from signal_logger import LOG_DIR

from . import config as cfg
from .kronos_integration import KronosIntegration
from .enhanced_bot import EnhancedIntradayBot
from .enhanced_guardian import KronosExitGuardian

logger = logging.getLogger("enhanced_main")
IST = ZoneInfo("Asia/Kolkata")
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ── Logging setup ────────────────────────────────────────────────────────────
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(cfg.STATE_DIR, exist_ok=True)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"enhanced_bot_{today}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("model").setLevel(logging.WARNING)

    logger.info("Enhanced bot logging to %s", log_path)
    return fh


# ── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    if not cfg.TELEGRAM_ENABLED:
        return
    try:
        requests.post(
            TELEGRAM_API.format(token=cfg.TELEGRAM_BOT_TOKEN),
            json={"chat_id": cfg.TELEGRAM_CHAT_ID, "text": message,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)


def format_signal_msg(symbol: str, tag: str, direction: str, quantity: int,
                      price: float, sl_price: float = 0.0,
                      tp1_price: float = 0.0, tp2_price: float = 0.0,
                      trailing_sl: float = 0.0, pnl: float = None,
                      pnl_pct: float = None) -> str:
    mode = "DRY-RUN" if cfg.DRY_RUN else "LIVE"
    now_str = datetime.now(IST).strftime("%H:%M:%S")
    lines = [
        f"<b>{tag}</b>  [{mode}]",
        f"<b>{symbol}</b>  {direction} x {quantity}",
        f"Price: Rs{price:,.2f}",
    ]
    if pnl is not None and pnl_pct is not None:
        lines.append(f"{'GREEN' if pnl >= 0 else 'RED'} PnL: Rs{pnl:+,.2f} ({pnl_pct:+.2f}%)")
    if sl_price > 0:
        lines.append(f"SL: Rs{sl_price:,.2f}")
    if trailing_sl > 0 and trailing_sl != sl_price:
        lines.append(f"Trail SL: Rs{trailing_sl:,.2f}")
    if tp1_price > 0:
        lines.append(f"TP1: Rs{tp1_price:,.2f}")
    if tp2_price > 0:
        lines.append(f"TP2: Rs{tp2_price:,.2f}")
    lines.append(f"Time: {now_str} IST")
    return "\n".join(lines)


# ── Self-improvement agent (reflection) ──────────────────────────────────────
# Strategy I/O and the reflection cycle live in reflect.py; run it between
# sessions via: python -m kronos_integrated_bot.run_reflection
from .reflect import load_strategy, save_strategy  # noqa: E402


def apply_strategy_to_config(strategy: dict):
    """Push YAML strategy params into cfg AND the base-class module globals.

    The inherited prefilter/MTF gates in stock_trading_bot read module-level
    constants, not cfg — without the module patch below, several tuned
    parameters (min_rr_ratio, min_adx_trending, prefilter floors) were
    silently inert while the reflection log believed they had changed.
    """
    import stock_trading_bot as base

    params = strategy.get("params", {})
    cfg.KRONOS_ENABLED = params.get("kronos_enabled", True)
    cfg.KRONOS_PRED_LEN = params.get("kronos_pred_len", 10)
    cfg.KRONOS_SAMPLE_COUNT = params.get("kronos_sample_count", 5)
    cfg.KRONOS_TEMPERATURE = params.get("kronos_temperature", 0.5)
    cfg.KRONOS_CONFIDENCE_WEIGHT = params.get("kronos_confidence_weight", 0.35)
    cfg.KRONOS_PENALTY_CONFLICT = params.get("kronos_penalty_conflict", 0.50)
    cfg.KRONOS_BONUS_ALIGN = params.get("kronos_bonus_align", 1.10)
    cfg.KRONOS_EXIT_THRESHOLD = params.get("kronos_exit_threshold", -0.008)
    cfg.KRONOS_MIN_PREDICTED_MOVE = params.get("kronos_min_predicted_move", 0.003)
    cfg.COOLDOWN_SECONDS = params.get("cooldown_seconds", 1800)
    cfg.MIN_CONFIDENCE = params.get("min_confidence", 80)
    cfg.RSI_OB_LIMIT = params.get("rsi_ob_limit", 70)
    cfg.RSI_OS_LIMIT = params.get("rsi_os_limit", 30)
    cfg.MAX_DAILY_SIGNALS = params.get("max_daily_signals", 10)
    cfg.MAX_SIGNALS_PER_STOCK_PER_DAY = params.get("max_signals_per_stock_per_day", 1)
    cfg.SAME_DIRECTION_COOLDOWN = params.get("same_direction_cooldown", 3600)
    cfg.MIN_VOLUME_RATIO_TRENDING = params.get("min_volume_ratio_trending", 0.40)
    cfg.MIN_RR_RATIO = params.get("min_rr_ratio", 1.8)
    cfg.MIN_ADX_TRENDING = params.get("min_adx_trending", 18)
    cfg.MIN_PREFILTER_VOLUME_RATIO = params.get("min_prefilter_volume_ratio", 0.15)
    cfg.MIN_PREFILTER_ATR_PCT = params.get("min_prefilter_atr_pct", 0.30)
    cfg.MAX_CONCURRENT_POSITIONS = params.get("max_concurrent_positions", 3)
    cfg.BUY_ENABLED = params.get("buy_enabled", True)

    # Patch the base-class module globals used by the inherited gates.
    base.MIN_ADX_TRENDING = cfg.MIN_ADX_TRENDING
    base.MIN_PREFILTER_VOLUME_RATIO = cfg.MIN_PREFILTER_VOLUME_RATIO
    base.MIN_PREFILTER_ATR_PCT = cfg.MIN_PREFILTER_ATR_PCT
    base.MIN_VOLUME_RATIO_TRENDING = cfg.MIN_VOLUME_RATIO_TRENDING
    base.MIN_RR_RATIO = cfg.MIN_RR_RATIO
    base.RSI_OB_LIMIT = cfg.RSI_OB_LIMIT
    base.RSI_OS_LIMIT = cfg.RSI_OS_LIMIT

    # Intraday-specific params
    cfg.TRAILING_SL_ACTIVATION_PCT = params.get("trailing_sl_activation_pct", 3.0)
    cfg.TRAILING_SL_DISTANCE_ATR = params.get("trailing_sl_distance_atr", 2.0)
    cfg.MAX_TRADE_DURATION_MINUTES = params.get("max_trade_duration_minutes", 180)
    cfg.MARKET_OPEN_SKIP_MINUTES = params.get("market_open_skip_minutes", 0)
    cfg.MARKET_CLOSE_EXIT_MINUTES = params.get("market_close_exit_minutes", 15)
    cfg.MAX_CONSECUTIVE_LOSSES = params.get("max_consecutive_losses", 3)
    cfg.PARTIAL_PROFIT_PCT = params.get("partial_profit_pct", 0.0)
    cfg.POSITION_CONFIDENCE_SCALAR = params.get("position_confidence_scalar", 1.0)
    cfg.MAX_DAILY_TRADES = params.get("max_daily_trades", 5)
    cfg.RISK_PER_TRADE_PCT = params.get("risk_per_trade_pct", 2.0)
    cfg.MAX_DAILY_LOSS_PCT = params.get("max_daily_loss_pct", 2.0)


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    _ = setup_logging()

    strategy = load_strategy()
    apply_strategy_to_config(strategy)
    logger.info("Loaded strategy v%s", strategy.get("version", "?"))

    # Init Dhan
    dhan_bot = DhanStockTradingBot()
    dhan_bot.data_store_dir = cfg.DATA_DIR  # persist candles for replay/backtests
    if cfg.TELEGRAM_ENABLED:
        dhan_bot.alert_cb = send_telegram

    # Init Kronos
    kronos_integration = KronosIntegration({
        "model_name": cfg.KRONOS_MODEL,
        "tokenizer_name": cfg.KRONOS_TOKENIZER,
        "max_context": cfg.KRONOS_MAX_CONTEXT,
        "device": cfg.KRONOS_DEVICE,
        "pred_len": cfg.KRONOS_PRED_LEN,
        "lookback": cfg.KRONOS_LOOKBACK,
        "temperature": cfg.KRONOS_TEMPERATURE,
        "sample_count": cfg.KRONOS_SAMPLE_COUNT,
        "top_p": cfg.KRONOS_TOP_P,
        "penalty_conflict": cfg.KRONOS_PENALTY_CONFLICT,
        "bonus_align": cfg.KRONOS_BONUS_ALIGN,
        "exit_threshold": cfg.KRONOS_EXIT_THRESHOLD,
    })
    kronos_integration.load()

    # Init AI
    ai_analyzer = DeepSeekStockAnalyzer(
        api_key=cfg.DEEPSEEK_API_KEY,
        alert_cb=send_telegram if cfg.TELEGRAM_ENABLED else None,
        # Prompt rulebook tracks the tuned strategy (applied above from YAML)
        min_confidence=cfg.MIN_CONFIDENCE,
        min_adx=cfg.MIN_ADX_TRENDING,
        rsi_ob=cfg.RSI_OB_LIMIT,
        rsi_os=cfg.RSI_OS_LIMIT,
        min_rr_ratio=cfg.MIN_RR_RATIO,
    )

    # Init Risk Manager
    risk_mgr = RiskManager(
        dhan_api=dhan_bot,
        max_daily_trades=cfg.MAX_DAILY_TRADES,
        max_daily_loss_percent=cfg.MAX_DAILY_LOSS_PCT,
        risk_per_trade_percent=cfg.RISK_PER_TRADE_PCT,
        min_confidence=cfg.MIN_CONFIDENCE,
        cash_buffer_pct=cfg.CASH_BUFFER_PCT,
        max_position_capital_pct=cfg.MAX_POSITION_CAPITAL_PCT,
        leverage=cfg.LEVERAGE,
    )

    # Init Enhanced Bot
    bot = EnhancedIntradayBot(
        dhan_bot, ai_analyzer, risk_mgr, kronos_integration,
        watchlist=cfg.WATCHLIST,
        send_telegram=send_telegram, format_signal_msg=format_signal_msg,
        enable_telegram=cfg.TELEGRAM_ENABLED, dry_run=cfg.DRY_RUN,
    )

    # Init Enhanced Guardian
    guardian = KronosExitGuardian(
        dhan_bot, bot, kronos_integration,
        send_telegram=send_telegram, format_signal_msg=format_signal_msg,
        enable_telegram=cfg.TELEGRAM_ENABLED, dry_run=cfg.DRY_RUN,
    )

    mode_str = "DRY-RUN" if cfg.DRY_RUN else "LIVE"
    telegram_msg = (
        f"KRONOS-ENHANCED Bot Started  [{mode_str}]\n"
        f"Watchlist: {len(cfg.WATCHLIST)} stocks\n"
        f"Kronos: {cfg.KRONOS_MODEL} | pred_len={cfg.KRONOS_PRED_LEN} | T={cfg.KRONOS_TEMPERATURE}\n"
        f"Strategy: v{strategy.get('version', '?')}"
    )
    if cfg.TELEGRAM_ENABLED:
        send_telegram(telegram_msg)
    logger.info(telegram_msg)

    await asyncio.gather(bot.run(), guardian.run())

    if cfg.TELEGRAM_ENABLED:
        send_telegram("KRONOS-ENHANCED Bot Stopped")


if __name__ == "__main__":
    asyncio.run(main())
