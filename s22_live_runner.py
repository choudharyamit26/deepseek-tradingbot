"""s22 gap-and-go live/paper runner (holdout-validated 2026-07-04).

Rule (frozen, do not re-tune): gap open >= 0.8% vs prev close AND price still
beyond day-open in gap direction at the 10:15 bar close -> enter with-gap at
~10:20, SL 3 x ATR(14, 5-min), no target, square off 15:10.

Runs once per session (start before 10:15, e.g. via scheduler). DRY_RUN=true
(default) records signals without orders. Entries/exits are appended to
trading_logs/signals_YYYY-MM-DD.csv in the standard schema, and backfill_rag
ingests them into analog_history.db after square-off.

Usage:  python s22_live_runner.py            # paper (DRY_RUN honored from .env)
        S22_LIVE=1 python s22_live_runner.py # force live orders
"""
import csv
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv(override=True)

from dhan_integration import DhanStockTradingBot
from signal_logger import _CSV_FIELDS
import backfill_rag

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).parent
LOGS = ROOT / "trading_logs"

GAP_MIN = 0.8          # frozen holdout params
SL_ATR = 3.0
CAPITAL = float(os.getenv("S22_CAPITAL", "100000"))
DRY = not os.getenv("S22_LIVE") and os.getenv("DRY_RUN", "true").lower() == "true"
UNIVERSE = ["SHRIRAMFIN", "INDIGO", "ADANIENT", "CHOLAFIN", "DIXON", "PAYTM",
            "BAJFINANCE", "ADANIPORTS", "BANDHANBNK", "M&M", "IDEA", "LT",
            "INDUSINDBK", "MUTHOOTFIN", "CANBK", "BPCL", "PNB", "BANKBARODA",
            "BHEL", "TRENT"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("s22")


def now():
    return datetime.now(IST)


def csv_path():
    LOGS.mkdir(exist_ok=True)
    return LOGS / f"signals_{now():%Y-%m-%d}.csv"


def append_row(**kw):
    p = csv_path()
    new = not p.exists()
    row = {k: "" for k in _CSV_FIELDS}
    row.update(kw)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def fill_exit(symbol, exit_price, pnl, tag):
    p = csv_path()
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in reversed(rows):
        if r["symbol"] == symbol and not r["exit_price"]:
            r["exit_price"] = f"{exit_price:.2f}"
            r["pnl"] = f"{pnl:.2f}"
            r["signal_type"] = tag
            break
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def wait_until(h, m):
    while now().time() < datetime(2000, 1, 1, h, m).time():
        time.sleep(15)


def main():
    bot = DhanStockTradingBot()
    log.info("s22 runner %s | capital/trade Rs%.0f", "DRY-RUN" if DRY else "LIVE", CAPITAL)
    wait_until(10, 20)          # 10:15 bar completes at 10:20

    positions = {}
    for sym in UNIVERSE:
        sid = bot.security_ids.get(sym)
        if not sid:
            continue
        try:
            df = bot.get_historical_data(sid, "5minute", min_bars=120)
            today = df[df.index.normalize() == df.index[-1].normalize()]
            prev = df[df.index.normalize() < df.index[-1].normalize()]
            if len(today) < 13 or prev.empty:
                continue
            day_open = float(today["open"].iloc[0])
            prev_close = float(prev["close"].iloc[-1])
            gap = (day_open / prev_close - 1) * 100
            bar1015 = today.between_time("10:15", "10:15")
            if bar1015.empty or abs(gap) < GAP_MIN:
                continue
            c = float(bar1015["close"].iloc[0])
            d = "BUY" if (gap > 0 and c > day_open) else \
                "SELL" if (gap < 0 and c < day_open) else None
            if not d:
                continue
            tr = (df["high"] - df["low"]).rolling(14).mean()
            atr = float(tr.iloc[-1])
            ltp = bot.fetch_live_data(sid).get("last_price") or c
            qty = max(1, int(CAPITAL // ltp))
            sl_pct = SL_ATR * atr / ltp * 100
            if not DRY:
                bot.place_super_order(sid, d, qty, ltp, sl_pct, 5.0,
                                      symbol=sym, atr_value=atr)
            sl_price = ltp * (1 - sl_pct / 100) if d == "BUY" else ltp * (1 + sl_pct / 100)
            positions[sym] = dict(sid=sid, d=d, entry=ltp, qty=qty, sl=sl_price)
            append_row(timestamp=f"{now():%Y-%m-%d %H:%M:%S}", symbol=sym,
                       signal_type=f"ENTRY-{'LONG' if d=='BUY' else 'SHORT'}",
                       direction=d, entry_price=f"{ltp:.2f}", quantity=qty,
                       stop_loss=f"{sl_price:.2f}", target="", confidence=80,
                       reasoning=f"s22 gap-and-go: gap={gap:+.2f}% still-extended "
                                 f"at 10:15 (close {c:.2f} vs open {day_open:.2f}); "
                                 f"SL=3xATR({atr:.2f}); hold-to-close. "
                                 f"Holdout-validated PF 1.10.",
                       mode="DRY-S22" if DRY else "LIVE-S22")
            log.info("%s %s ENTRY %s qty=%d gap=%+.2f%% sl=%.2f",
                     sym, "DRY" if DRY else "LIVE", d, qty, gap, sl_price)
        except Exception as exc:
            log.warning("%s skipped: %s", sym, exc)

    # manage to square-off
    while positions and now().time() < datetime(2000, 1, 1, 15, 10).time():
        time.sleep(30)
        for sym, p in list(positions.items()):
            try:
                ltp = bot.fetch_live_data(p["sid"]).get("last_price")
                if not ltp:
                    continue
                hit = ltp <= p["sl"] if p["d"] == "BUY" else ltp >= p["sl"]
                if hit:
                    _close(bot, sym, p, ltp, "SL-EXIT")
                    del positions[sym]
            except Exception as exc:
                log.warning("%s monitor: %s", sym, exc)
    for sym, p in positions.items():
        ltp = bot.fetch_live_data(p["sid"]).get("last_price") or p["entry"]
        _close(bot, sym, p, ltp, "MARKET-CLOSE")

    backfill_rag.backfill(date_filter=f"{now():%Y-%m-%d}")
    log.info("done; signals csv + analog_history.db updated")


def _close(bot, sym, p, ltp, tag):
    if not DRY:
        bot.reduce_position(p["sid"], "SELL" if p["d"] == "BUY" else "BUY", p["qty"])
    pnl = (ltp - p["entry"]) * p["qty"] * (1 if p["d"] == "BUY" else -1)
    fill_exit(sym, ltp, pnl, tag)
    log.info("%s EXIT %s @ %.2f pnl=%+.2f", sym, tag, ltp, pnl)


if __name__ == "__main__":
    main()
