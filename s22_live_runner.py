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
from momentum_bot.telegram import send as tg_send
import backfill_rag

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).parent
LOGS = ROOT / "trading_logs"

GAP_MIN = 0.8          # frozen holdout params
SL_ATR = 3.0
CAPITAL = float(os.getenv("S22_CAPITAL", "100000"))
DRY = not os.getenv("S22_LIVE") and os.getenv("DRY_RUN", "true").lower() == "true"
# holdout-validated universe. 20-name high-beta expansion tested 2026-07-06
# and REVERTED: transfer failed (PF 0.82, 5/20 profitable, s22_new20.json).
# Last 4 names = expansion survivors that passed a pre-registered 2024-25
# holdout (PF 1.45-2.05, s22_new20_holdout.json); TATAMOTORS untestable
# post-demerger, dropped.
UNIVERSE = ["SHRIRAMFIN", "INDIGO", "ADANIENT", "CHOLAFIN", "DIXON", "PAYTM",
            "BAJFINANCE", "ADANIPORTS", "BANDHANBNK", "M&M", "IDEA", "LT",
            "INDUSINDBK", "MUTHOOTFIN", "CANBK", "BPCL", "PNB", "BANKBARODA",
            "BHEL", "TRENT",
            "VEDL", "RECLTD", "RVNL", "PFC"]
# NSE EQ security ids for universe names absent from the production bot's
# security_ids map (source: Dhan scrip master, verified 2026-07-06)
S22_SIDS = {"SHRIRAMFIN": "4306", "INDIGO": "11195", "CHOLAFIN": "685",
            "DIXON": "21690", "PAYTM": "6705", "BANDHANBNK": "2263",
            "IDEA": "14366", "MUTHOOTFIN": "23650", "CANBK": "10794",
            "PNB": "10666", "BANKBARODA": "4668", "BHEL": "438",
            "TRENT": "1964",
            "RVNL": "9552", "RECLTD": "15355", "PFC": "14299"}


def get_sid(bot, sym):
    sid = bot.security_ids.get(sym) or S22_SIDS.get(sym)
    if not sid:
        log.warning("%s: no security id, skipped", sym)
    return sid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("s22")

SESSION_PNL = []   # realized pnl per closed position, for the end-of-day summary


def notify(msg):
    """Telegram alert prefixed with the runner mode; never raises."""
    tg_send(f"<b>[S22{'-DRY' if DRY else ''}]</b> {msg}")


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


OVERNIGHT = bool(os.getenv("S22_OVERNIGHT"))   # BTST: CNC gap-up longs, sell next open
STATE = ROOT / "s22_overnight_state.json"


def overnight_main(bot):
    """Sell yesterday's CNC holdings at the open, buy today's gap-ups at 10:20.
    Validated: buy 10:20 gap>=0.8% extended, sell NEXT open (PF 1.23 / 1.52)."""
    import json as _j
    wait_until(9, 16)
    held = _j.loads(STATE.read_text()) if STATE.exists() else {}
    for sym, p in held.items():
        ltp = bot.fetch_live_data(p["sid"]).get("last_price") or p["entry"]
        if not DRY:
            bot.place_equity_order(p["sid"], "SELL", p["qty"], product_type="CNC")
        pnl = (ltp - p["entry"]) * p["qty"]
        fill_exit(sym, ltp, pnl, "BTST-OPEN-EXIT")
        log.info("%s BTST EXIT @ %.2f pnl=%+.2f", sym, ltp, pnl)
        notify(f"BTST EXIT <b>{sym}</b> @ {ltp:.2f}\nPnL: <b>{pnl:+.2f}</b>")
    if held:
        backfill_rag.backfill(date_filter=f"{now():%Y-%m-%d}")
    STATE.write_text("{}")
    wait_until(10, 20)
    new = {}
    for sym in UNIVERSE:
        sid = get_sid(bot, sym)
        if not sid:
            continue
        try:
            df = bot.get_historical_data(sid, "5minute", min_bars=120)
            today = df[df.index.normalize() == df.index[-1].normalize()]
            prev = df[df.index.normalize() < df.index[-1].normalize()]
            day_open = float(today["open"].iloc[0])
            gap = (day_open / float(prev["close"].iloc[-1]) - 1) * 100
            bar = today.between_time("10:15", "10:15")
            if bar.empty or gap < GAP_MIN or float(bar["close"].iloc[0]) <= day_open:
                continue
            ltp = bot.fetch_live_data(sid).get("last_price") or float(bar["close"].iloc[0])
            qty = max(1, int(CAPITAL // ltp))
            if not DRY:
                bot.place_equity_order(sid, "BUY", qty, product_type="CNC")
            new[sym] = dict(sid=sid, entry=ltp, qty=qty)
            append_row(timestamp=f"{now():%Y-%m-%d %H:%M:%S}", symbol=sym,
                       signal_type="ENTRY-LONG", direction="BUY",
                       entry_price=f"{ltp:.2f}", quantity=qty, confidence=80,
                       reasoning=f"s22-overnight BTST: gap={gap:+.2f}% extended at "
                                 f"10:15; CNC hold to next open (PF 1.23/1.52).",
                       mode="DRY-S22ON" if DRY else "LIVE-S22ON")
            log.info("%s BTST ENTRY qty=%d gap=%+.2f%%", sym, qty, gap)
            notify(f"BTST ENTRY <b>{sym}</b>\n"
                   f"Qty: {qty} @ {ltp:.2f}\n"
                   f"Gap: {gap:+.2f}% (CNC, sell tomorrow open)")
        except Exception as exc:
            log.warning("%s skipped: %s", sym, exc)
    STATE.write_text(_j.dumps(new))
    log.info("overnight positions: %d (sell tomorrow 09:16)", len(new))
    notify(f"overnight scan done: {len(new)} positions"
           + (" — " + ", ".join(new) if new else "") + " (sell tomorrow 09:16)")


def main():
    bot = DhanStockTradingBot()
    log.info("s22 runner %s%s | capital/trade Rs%.0f",
             "DRY-RUN" if DRY else "LIVE", " OVERNIGHT" if OVERNIGHT else "", CAPITAL)
    notify(f"runner started {'DRY-RUN' if DRY else 'LIVE'}"
           f"{' OVERNIGHT' if OVERNIGHT else ''} | capital/trade Rs{CAPITAL:.0f}")
    if OVERNIGHT:
        overnight_main(bot)
        return
    wait_until(10, 20)          # 10:15 bar completes at 10:20

    positions = {}
    for sym in UNIVERSE:
        sid = get_sid(bot, sym)
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
            if bar1015.empty:
                log.warning("%s: 10:15 bar missing from history, skipped", sym)
                continue
            if abs(gap) < GAP_MIN:
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
            notify(f"ENTRY {d} <b>{sym}</b>\n"
                   f"Qty: {qty} @ {ltp:.2f}\n"
                   f"SL: {sl_price:.2f} (3xATR)\n"
                   f"Gap: {gap:+.2f}%")
        except Exception as exc:
            log.warning("%s skipped: %s", sym, exc)
    log.info("scan complete: %d/%d symbols, %d entries",
             sum(1 for s in UNIVERSE if bot.security_ids.get(s) or S22_SIDS.get(s)),
             len(UNIVERSE), len(positions))
    notify(f"scan complete: {len(positions)} entries"
           + (" — " + ", ".join(positions) if positions else ""))

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
    if SESSION_PNL:
        notify(f"session done: {len(SESSION_PNL)} closed, "
               f"total PnL <b>{sum(SESSION_PNL):+.2f}</b>")


def _close(bot, sym, p, ltp, tag):
    if not DRY:
        bot.reduce_position(p["sid"], "SELL" if p["d"] == "BUY" else "BUY", p["qty"])
    pnl = (ltp - p["entry"]) * p["qty"] * (1 if p["d"] == "BUY" else -1)
    fill_exit(sym, ltp, pnl, tag)
    SESSION_PNL.append(pnl)
    log.info("%s EXIT %s @ %.2f pnl=%+.2f", sym, tag, ltp, pnl)
    notify(f"EXIT {tag} <b>{sym}</b> @ {ltp:.2f}\nPnL: <b>{pnl:+.2f}</b>")


if __name__ == "__main__":
    main()
