"""Entry point for the Opening Range Breakout momentum strategy.

Usage:
    python momentum_main.py            # live mode
    python momentum_main.py --dry-run  # compute signals, skip order placement

This is a completely separate process from the main enhanced_bot. Running
both simultaneously is fine — they use independent Dhan sessions.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs("trading_logs", exist_ok=True)
_date = datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"trading_logs/momentum_{_date}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("momentum_main")

# ── Imports after logging ─────────────────────────────────────────────────────
from dhan_integration import DhanStockTradingBot
from momentum_bot.bot import MomentumBot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORB momentum bot")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute signals but do not place real orders",
    )
    return p.parse_args()


async def _main(dry_run: bool) -> None:
    logger.info("Initialising Dhan connection …")
    dhan = DhanStockTradingBot()

    bot = MomentumBot(dhan, dry_run=dry_run)
    await bot.run()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(_main(dry_run=args.dry_run))
    except KeyboardInterrupt:
        logger.info("Interrupted by user — stopping")
