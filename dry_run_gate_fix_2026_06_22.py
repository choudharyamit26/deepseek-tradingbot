#!/usr/bin/env python3
"""
Dry-run replay of 2026-06-22 log under the fixed SELL gate.

What changed in enhanced_bot.py:
  1. Removed -4 penalty for "sector bullish vs SELL" in post-AI regime block.
  2. Added `and sector_trend != "bullish"` to the SELL hard gate.
     → SELL into a bullish sector (mean-reversion shorts) now passes through.
     → All other counter-trend SELLs (sector neutral/bearish, Nifty bullish) stay gated.

This script replays every SELL HARD-GATED entry from today's log, recalculates
confidence and gate outcome under the new rules, and prints a signal table.
"""

import re, sys
from datetime import datetime

LOG = r"trading_logs\enhanced_bot_2026-06-22.log"
MIN_CONF = 82
NIFTY_PENALTY = 6   # fixed (intraday was 0.00% all day → scale=0 → pen=6)
SECTOR_PENALTY = 4  # REMOVED in fix

# ── Gate entries: all SELL HARD-GATED where sector=bullish ─────────────────
# Each tuple: (time, symbol, matrix_score, ai_conf_before_cap_or_approx)
# Derived from log: gate_conf = min(ai_conf, matrix) - NIFTY_PEN - SECTOR_PEN
# → ai_after_cap = gate_conf + NIFTY_PEN + SECTOR_PEN
# → new_conf = ai_after_cap - NIFTY_PEN  (no sector pen)

gate_hits = [
    # (time,       symbol,       matrix, ai_conf, sector)
    ("10:18:55", "DIXON",       98,     85,      "bullish"),
    ("11:01:30", "POLYCAB",     85,     85,      "bullish"),
    ("11:10:02", "COLPAL",      91,     91,      "bullish"),
    ("11:16:05", "COLPAL",      89,     89,      "bullish"),
    ("11:46:07", "COLPAL",      89,     89,      "bullish"),
    ("11:55:02", "SHRIRAMFIN",  86,     86,      "bullish"),
    ("12:15:59", "SHRIRAMFIN",  86,     86,      "bullish"),
    ("12:58:17", "SBILIFE",    100,     82,      "bullish"),
    ("13:01:16", "AMBUJACEM",   86,     86,      "bullish"),
    ("13:16:04", "TATACONSUM",  90,     90,      "bullish"),
    ("13:58:14", "SBILIFE",     86,     86,      "bullish"),
    ("14:25:21", "YESBANK",     90,     82,      "bullish"),
]

# Track per-stock signal counts (existing BUY signal for SBILIFE counts)
daily_signals: dict[str, int] = {"SBILIFE": 1}  # had BUY signal at 10:37

print("=" * 100)
print("  DRY-RUN REPLAY — 2026-06-22  |  Fixed SELL gate (sector-bullish exemption)")
print("  Nifty was daily-BULLISH all day, session=neutral, intraday~0%")
print("  Rule: SELL gated when (nifty_bullish AND session!=bearish AND sector!=bullish)")
print("=" * 100)
print()
print(f"  {'Time':8s}  {'Symbol':12s}  {'Sector':8s}  {'Matrix':7s}  {'AI conf':8s}  "
      f"{'Old conf':9s}  {'Old outcome':14s}  {'New conf':9s}  {'New outcome':20s}  Note")
print(f"  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*8}  "
      f"{'-'*9}  {'-'*14}  {'-'*9}  {'-'*20}  {'-'*30}")

new_signals = []

for time, sym, matrix, ai_conf, sector in gate_hits:
    # old path: both penalties, then cap, then gate (sector-blind)
    ai_capped     = min(ai_conf, matrix)
    old_conf      = ai_capped - NIFTY_PENALTY - SECTOR_PENALTY
    old_outcome   = "GATED (counter-trend)"

    # new path: only Nifty penalty (sector pen removed), then cap, then gate skipped
    new_conf_raw  = ai_capped - NIFTY_PENALTY          # no sector penalty
    new_conf      = min(new_conf_raw, matrix)           # cap still applies

    # gate exempted since sector=bullish
    daily_signals[sym] = daily_signals.get(sym, 0)

    if daily_signals[sym] >= 2:
        new_outcome = "BLOCKED (daily max 2)"
        note = f"already {daily_signals[sym]} signals today"
    elif new_conf < MIN_CONF:
        new_outcome = f"BLOCKED (conf {new_conf}<{MIN_CONF})"
        note = "sector exempts gate but conf still too low"
    else:
        daily_signals[sym] += 1
        new_signals.append((time, sym, new_conf))
        new_outcome = f"** SELL SIGNAL conf={new_conf}"
        note = f"signal #{daily_signals[sym]} for {sym}"

    print(f"  {time:8s}  {sym:12s}  {sector:8s}  {matrix:7d}  {ai_conf:8d}  "
          f"{old_conf:9d}  {old_outcome:14s}  {new_conf:9d}  {new_outcome:20s}  {note}")

print()
print("=" * 100)
print(f"  SIGNALS GENERATED: {len(new_signals)}")
print()
if new_signals:
    for t, s, c in new_signals:
        print(f"    {t}  {s:<12s}  SELL  conf={c}")
else:
    print("    (none)")

print()
print("  SECTOR-NEUTRAL SELL gate hits (unchanged — still gated by fix):")
still_gated = [
    # stocks that were gated with sector=neutral/bearish — fix does not help
    ("09:33:48", "DIXON",       "neutral", 82),
    ("09:39:07", "DIXON",       "neutral", 85),
    ("09:55:50", "DIXON",       "neutral", 89),
    ("10:30:53", "COFORGE",     "neutral", 92),
    ("10:37:18", "LTIM",        "bearish", 82),
    ("11:01:30", "SHRIRAMFIN",  "neutral", 85),
    ("11:28:05", "YESBANK",     "neutral", 85),
    ("11:43:08", "YESBANK",     "neutral", 90),
    ("12:01:33", "DIXON",       "neutral", 86),
    ("12:15:41", "TATACONSUM",  "neutral", 82),
    ("12:15:52", "DIXON",       "neutral", 85),
    ("12:15:55", "COLPAL",      "neutral", 86),
    ("12:22:04", "WIPRO",       "bearish", 86),
    ("12:25:07", "TORNTPOWER",  "neutral", 82),
    ("12:31:15", "TATAELXSI",   "bearish", 89),
    ("12:43:05", "WIPRO",       "bearish", 87),
    ("12:46:31", "YESBANK",     "neutral", 82),
    ("12:49:05", "YESBANK",     "neutral", 82),
    ("12:52:16", "YESBANK",     "neutral", 89),
    ("12:52:17", "SBILIFE",     "neutral", 89),
    ("12:55:04", "SBILIFE",     "neutral", 89),
]
print(f"  {'Time':8s}  {'Symbol':12s}  {'Sector':8s}  {'AI conf':8s}  Outcome")
print(f"  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*30}")
for t, s, sec, c in still_gated:
    print(f"  {t:8s}  {s:12s}  {sec:8s}  {c:8d}  still gated (no change)")

print()
print("=" * 100)
print("  SUMMARY")
print(f"    Sector=bullish gate hits today : {len(gate_hits)}")
print(f"    Would generate signals         : {len(new_signals)}")
print(f"    Still blocked (conf < 82)      : {len(gate_hits) - len(new_signals) - sum(1 for t,s,c in new_signals if False)}")
print(f"    Sector=neutral still gated     : {len(still_gated)} (correct, fix doesn't touch these)")
print()
print("  NOTE: All signal executions would have hit the same Dhan DH-905 'Invalid IP'")
print("  error as the SBILIFE BUY at 10:37 — that is a separate infrastructure issue.")
print("=" * 100)
