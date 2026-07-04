"""Thin Dhan v2 REST client for historical data. Deliberately independent of
the live bot's dhan_integration (which load_dotenv(override=True)s and drags in
trading state). Read-only endpoints, live credentials."""
import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

load_dotenv(cfg.REPO / ".env")
BASE = "https://api.dhan.co/v2"
HEADERS = {
    "access-token": os.getenv("DHAN_ACCESS_TOKEN", ""),
    "client-id": os.getenv("DHAN_CLIENT_ID", ""),
    "Content-Type": "application/json",
    "Accept": "application/json",
}
logger = logging.getLogger(__name__)


def _post(path, payload, retries=4):
    for i in range(retries):
        try:
            r = requests.post(BASE + path, headers=HEADERS, json=payload, timeout=40)
            if r.status_code == 429:
                time.sleep(2.0 * (i + 1))
                continue
            if r.status_code != 200:
                logger.warning("%s %s -> %s %s", path, payload.get("securityId"),
                               r.status_code, r.text[:120])
                time.sleep(1.0 * (i + 1))
                continue
            return r.json()
        except requests.RequestException as exc:
            logger.warning("%s retry %d: %s", path, i + 1, exc)
            time.sleep(1.5 * (i + 1))
    return None


def _to_ist(ts_epoch):
    """Dhan epochs: detect whether they're true UTC or already-IST-naive by
    checking which interpretation puts the first bars at 09:15."""
    utc = pd.to_datetime(ts_epoch, unit="s", utc=True)
    ist = utc.tz_convert("Asia/Kolkata").tz_localize(None)
    naive = utc.tz_localize(None)
    def score(idx):
        t = idx[: min(len(idx), 500)]
        return sum(1 for x in t if (x.hour, x.minute) == (9, 15))
    return ist if score(ist) >= score(naive) else naive


def _frame(d):
    if not d or not d.get("timestamp"):
        return pd.DataFrame()
    df = pd.DataFrame({k: d[k] for k in ("open", "high", "low", "close", "volume")})
    df.index = _to_ist(d["timestamp"])
    df.index.name = "ts"
    return df[~df.index.duplicated()].sort_index()


def fetch_intraday(security_id, start, end, interval=cfg.INTERVAL, segment="NSE_EQ",
                   instrument="EQUITY"):
    """Chunked 5-min history [start, end] inclusive, IST-naive index."""
    frames = []
    s = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    while s <= stop:
        e = min(s + timedelta(days=cfg.CHUNK_DAYS), stop)
        d = _post("/charts/intraday", {
            "securityId": str(security_id), "exchangeSegment": segment,
            "instrument": instrument, "interval": interval,
            "fromDate": s.strftime("%Y-%m-%d"), "toDate": e.strftime("%Y-%m-%d")})
        frames.append(_frame(d))
        time.sleep(cfg.THROTTLE_S)
        s = e + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return df[~df.index.duplicated()].sort_index()


def fetch_daily(security_id, start, end, segment="NSE_EQ", instrument="EQUITY"):
    d = _post("/charts/historical", {
        "securityId": str(security_id), "exchangeSegment": segment,
        "instrument": instrument, "expiryCode": 0,
        "fromDate": start, "toDate": end})
    time.sleep(cfg.THROTTLE_S)
    df = _frame(d)
    if not df.empty:
        df.index = df.index.normalize()
    return df


def store_path(symbol, interval=cfg.INTERVAL):
    return cfg.STORE / f"{symbol}_{interval}min.parquet"


def load_bars(symbol, interval=cfg.INTERVAL):
    p = store_path(symbol, interval)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def fetch_and_store(symbol, security_id, segment="NSE_EQ", instrument="EQUITY"):
    """Idempotent: skips if stored data already spans the window."""
    p = store_path(symbol)
    if p.exists():
        df = pd.read_parquet(p)
        if len(df) and df.index[0].strftime("%Y-%m-%d") <= cfg.START \
                and df.index[-1].strftime("%Y-%m-%d") >= cfg.END:
            return df
    df = fetch_intraday(security_id, cfg.START, cfg.END,
                        segment=segment, instrument=instrument)
    if df.empty:
        logger.error("no intraday data for %s (%s)", symbol, security_id)
        return df
    df.to_parquet(p)
    return df


def qa_report(symbol, df):
    """Session-coverage sanity for one symbol's 5-min frame."""
    if df.empty:
        return {"symbol": symbol, "sessions": 0, "ok": False, "note": "EMPTY"}
    by_day = df.groupby(df.index.normalize())
    sessions = len(by_day)
    bars_med = int(by_day.size().median())
    first_ok = (by_day.head(1).index.time == pd.Timestamp("09:15").time()).mean()
    closes = by_day["close"].last()
    opens = by_day["open"].first()
    jumps = (opens.values[1:] / closes.values[:-1] - 1)
    max_jump = float(abs(jumps).max()) if len(jumps) else 0.0
    ok = sessions >= cfg.MIN_SESSIONS and bars_med >= 70 and max_jump <= cfg.MAX_OVERNIGHT_JUMP
    return {"symbol": symbol, "sessions": sessions, "median_bars": bars_med,
            "first_bar_0915_pct": round(100 * float(first_ok)), "max_overnight_jump_pct":
            round(100 * max_jump, 1), "ok": ok, "note": ""}
