"""Telegram notification helper for the momentum bot.

Reads the same environment variables used by the main kronos bot so a single
bot/chat pair receives alerts from both strategies.

    TELEGRAM_BOT_TOKEN — set in .env or shell
    TELEGRAM_CHAT_ID   — set in .env or shell
"""

from __future__ import annotations
import logging
import os

import requests

logger = logging.getLogger(__name__)

_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")
_ENABLED  = bool(_TOKEN and _CHAT_ID)
_API_URL  = "https://api.telegram.org/bot{token}/sendMessage"


def send(message: str) -> None:
    if not _ENABLED:
        logger.debug("Telegram disabled (no token/chat_id) — message: %s", message[:80])
        return
    try:
        requests.post(
            _API_URL.format(token=_TOKEN),
            json={
                "chat_id":                  _CHAT_ID,
                "text":                     message,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)


def time_exit_msg(
    symbol:      str,
    direction:   str,
    quantity:    int,
    entry_price: float,
    exit_price:  float,
    pnl:         float,
    exit_time:   str,
) -> str:
    sign   = "+" if pnl >= 0 else ""
    arrow  = "BUY" if direction == "BUY" else "SELL"
    return (
        f"<b>[MOMENTUM] TIME EXIT</b>\n"
        f"Symbol   : <b>{symbol}</b>\n"
        f"Direction: {arrow}\n"
        f"Qty      : {quantity}\n"
        f"Entry    : {entry_price:.2f}\n"
        f"Exit     : {exit_price:.2f}\n"
        f"PnL      : <b>{sign}{pnl:.2f}</b>\n"
        f"Time     : {exit_time}\n"
        f"Reason   : Time exit (neither SL nor target hit)"
    )
