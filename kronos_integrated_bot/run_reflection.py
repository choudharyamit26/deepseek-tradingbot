#!/usr/bin/env python3
"""CLI for the reflection agent.

Usage:
    python -m kronos_integrated_bot.run_reflection [--dry-run]

--dry-run: run the full cycle (revert check, evidence gate, LLM proposal)
but do not archive, save, or append anything.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kronos_integrated_bot import config as cfg
from kronos_integrated_bot.reflect import REFLECTION_LOG, run_reflection


def main():
    parser = argparse.ArgumentParser(description="Run one reflection cycle.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze and propose but do not modify the strategy.")
    args = parser.parse_args()

    cfg.STATE_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REFLECTION_LOG, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers)

    result = run_reflection(dry_run=args.dry_run)
    logging.getLogger("reflection").info("Reflection result: %s", result)


if __name__ == "__main__":
    main()
