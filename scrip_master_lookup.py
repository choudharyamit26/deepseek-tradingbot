"""Download Dhan scrip master and resolve security IDs for all sector_map symbols.

Usage:
    python scrip_master_lookup.py

Outputs:
  1. Which sector_map symbols are already mapped (with their current ID)
  2. Which are MISSING — with the ID found in the scrip master (ready to paste)
  3. Which could not be found in the scrip master at all
  4. A ready-to-paste MOMENTUM_UNIVERSE dict for dhan_integration.py
"""

import csv
import io
import sys
import urllib.request
from collections import defaultdict

# ── Dhan scrip master URL ─────────────────────────────────────────────────────
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# ── All symbols from momentum_bot/sector_map.py ───────────────────────────────
from momentum_bot.sector_map import SECTORS, SYMBOL_TO_SECTOR

ALL_SECTOR_SYMBOLS = set(SYMBOL_TO_SECTOR.keys())

# ── All currently mapped symbols (from dhan_integration + constant.py) ────────
# Import the combined watchlist the same way DhanStockTradingBot builds it
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from constant import FNO_UNIVERSE, ETF_LIQUID, FILTERED_FNO_UNIVERSE, NIFTY50_UNIVERSE
from dhan_integration import VWAP_RECLAIM_STOCKS

CURRENT_MAPPING: dict[str, str] = {
    **FNO_UNIVERSE,
    **ETF_LIQUID,
    **FILTERED_FNO_UNIVERSE,
    **VWAP_RECLAIM_STOCKS,
    **NIFTY50_UNIVERSE,
}

# ── Download scrip master ─────────────────────────────────────────────────────
print("Downloading Dhan scrip master …", end=" ", flush=True)
try:
    req = urllib.request.Request(
        SCRIP_MASTER_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    print(f"OK ({len(raw):,} bytes)")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

# ── Parse scrip master ────────────────────────────────────────────────────────
# Build two lookups:
#   by_symbol[trading_symbol] → list of rows (may have BSE + NSE entries)
#   We prefer NSE_EQ EQUITY rows.
reader    = csv.DictReader(io.StringIO(raw))
by_symbol = defaultdict(list)

for row in reader:
    sym  = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
    seg  = (row.get("SEM_SEGMENT") or "").strip().upper()
    inst = (row.get("SEM_INSTRUMENT_NAME") or "").strip().upper()
    sid  = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
    name = (row.get("SM_SYMBOL_NAME") or "").strip()
    if sym and sid:
        by_symbol[sym].append({
            "sid": sid, "segment": seg, "instrument": inst, "name": name,
        })

print(f"Scrip master: {len(by_symbol):,} unique symbols\n")


def best_sid(symbol: str) -> tuple[str, str]:
    """Return (security_id, full_name) for the NSE equity row, else best guess."""
    rows = by_symbol.get(symbol, [])
    # Prefer NSE_EQ EQUITY
    for r in rows:
        if r["segment"] == "NSE_EQ" and r["instrument"] == "EQUITY":
            return r["sid"], r["name"]
    # Fall back to any NSE_EQ
    for r in rows:
        if r["segment"] == "NSE_EQ":
            return r["sid"], r["name"]
    # Any row
    if rows:
        return rows[0]["sid"], rows[0]["name"]
    return "", ""


# ── Classify each sector_map symbol ──────────────────────────────────────────
already_mapped = {}
missing_found  = {}   # in sector_map, not yet in our mapping, but found in scrip master
not_found      = []   # not in our mapping AND not in scrip master

for sym in sorted(ALL_SECTOR_SYMBOLS):
    if sym in CURRENT_MAPPING:
        already_mapped[sym] = CURRENT_MAPPING[sym]
    else:
        sid, name = best_sid(sym)
        if sid:
            missing_found[sym] = (sid, name)
        else:
            not_found.append(sym)

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"ALREADY MAPPED  ({len(already_mapped)} symbols)")
print("=" * 70)
for sym in sorted(already_mapped):
    sec = SYMBOL_TO_SECTOR[sym]
    print(f"  {sym:<14}  {already_mapped[sym]:<8}  [{sec}]")

print()
print("=" * 70)
print(f"MISSING — found in scrip master  ({len(missing_found)} symbols)")
print("=" * 70)
for sym in sorted(missing_found):
    sid, name = missing_found[sym]
    sec = SYMBOL_TO_SECTOR[sym]
    print(f"  {sym:<14}  {sid:<8}  [{sec}]  {name}")

print()
print("=" * 70)
print(f"NOT FOUND in scrip master  ({len(not_found)} symbols)")
print("=" * 70)
for sym in not_found:
    sec = SYMBOL_TO_SECTOR[sym]
    print(f"  {sym:<14}  [{sec}]")

# ── Ready-to-paste MOMENTUM_UNIVERSE dict ─────────────────────────────────────
if missing_found:
    print()
    print("=" * 70)
    print("PASTE THIS INTO dhan_integration.py  (MOMENTUM_UNIVERSE)")
    print("=" * 70)
    print("MOMENTUM_UNIVERSE: dict[str, str] = {")
    for sym in sorted(missing_found):
        sid, name = missing_found[sym]
        sec = SYMBOL_TO_SECTOR[sym]
        print(f'    "{sym}": "{sid}",  # {name}  [{sec}]')
    print("}")

    print()
    print("=" * 70)
    print("TICK SIZE ENTRIES  (add to TICK_SIZE_MAP in dhan_integration.py)")
    print("=" * 70)
    for sym in sorted(missing_found):
        sid, name = missing_found[sym]
        # Default tick for equities is 0.05; stocks > Rs 5000 typically 0.05 too
        print(f'    "{sym}": 0.05,  # {name}')
