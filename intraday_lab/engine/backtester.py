"""Bar-replay backtester shared by all strategies.

Honesty rules (enforced here, tested in tests/):
  - signals are read at bar close; entry fills at the NEXT bar's open + slippage
  - SL/TP checked intrabar with conservative ordering: SL first when both touch
  - fresh entries only 09:30-14:45; forced square-off at first bar >= 15:10
  - one position per symbol; fixed Rs notional; all results NET of costs
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from engine import costs, metrics


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add engine columns (ATR, minutes-of-day, day id). Done once per symbol."""
    out = df.copy()
    h, l, c = out["high"], out["low"], out["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(cfg.ATR_LEN).mean()
    out["tod"] = out.index.hour * 60 + out.index.minute
    out["day"] = out.index.normalize()
    g = out.groupby("day", sort=False)
    out["bar_no"] = g.cumcount()
    out["day_open"] = g["open"].transform("first")
    cum_v = g["volume"].cumsum().clip(lower=1e-9)
    cum_pv = (out["close"] * out["volume"]).groupby(out["day"], observed=True).cumsum()
    out["vwap"] = cum_pv / cum_v
    day_close = g["close"].last()
    out["prev_close"] = out["day"].map(day_close.shift(1))
    return out


def run(df: pd.DataFrame, signal: np.ndarray, params: dict, symbol: str) -> pd.DataFrame:
    """Simulate one symbol. `signal[i]` in {-1,0,+1} decided at close of bar i.

    Exit modes:
      tp_mode="atr" (default): ATR bracket (sl_atr / tp_atr) + max_hold.
      tp_mode="2ph": two-phase — bank half at +partial_pct, stop -> breakeven,
        runner trails trail_pct behind high-water (the live bot's validated
        exit); hard ATR stop applies before the flip. TIME exit skipped after
        the flip (runner is breakeven-floored), square-off still applies.
    """
    sl_atr = float(params.get("sl_atr", 1.5))
    tp_atr = float(params.get("tp_atr", 3.0))
    max_hold = int(params.get("max_hold", cfg.MAX_HOLD_BARS))
    two_ph = params.get("tp_mode", "atr") == "2ph"
    partial_pct = float(params.get("partial_pct", 0.4)) / 100.0
    trail_pct = float(params.get("trail_pct", 0.5)) / 100.0

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["atr"].values
    tod = df["tod"].values
    day = df["day"].values
    n = len(df)
    idx = df.index

    cand = np.nonzero(signal != 0)[0]  # bars whose close carries a signal
    trades = []
    busy_until = -1
    day_count: dict = {}

    for s_i in cand:
        if s_i < busy_until or s_i + 1 >= n:
            continue
        e_i = s_i + 1                                   # entry bar (next open)
        if day[e_i] != day[s_i]:                        # signal can't cross days
            continue
        if not (cfg.ENTRY_START <= tod[e_i] <= cfg.ENTRY_END):
            continue
        if day_count.get(day[e_i], 0) >= cfg.MAX_TRADES_PER_DAY:
            continue
        a = atr[s_i]
        if not np.isfinite(a) or a <= 0:
            continue
        d = int(signal[s_i])
        entry = costs.slip(o[e_i], direction_hurts_up=(d > 0))
        qty = int(cfg.CAPITAL_PER_TRADE // entry)
        if qty < 1:
            continue
        sl = entry - d * sl_atr * a
        tp = entry + d * tp_atr * a
        part_lvl = entry * (1 + d * partial_pct)

        banked = 0.0                # realized partial pnl, net of its exit cost
        run_qty = qty
        flipped = False
        highwater = entry
        exit_px, exit_i, reason = None, None, None

        for j in range(e_i, n):
            if day[j] != day[e_i]:
                exit_px, exit_i, reason = c[j - 1], j - 1, "EOD"
                break
            if tod[j] >= cfg.SQUARE_OFF:
                exit_px, exit_i, reason = o[j], j, "SQUAREOFF"
                break
            if not flipped:
                hit_sl = l[j] <= sl if d > 0 else h[j] >= sl
                if hit_sl:                              # conservative: SL first
                    exit_px, exit_i, reason = sl, j, "SL"
                    break
                if two_ph:
                    hit_p = h[j] >= part_lvl if d > 0 else l[j] <= part_lvl
                    if hit_p:
                        pq = run_qty // 2
                        if pq >= 1:
                            px = costs.slip(part_lvl, direction_hurts_up=(d < 0))
                            banked += d * (px - entry) * pq - costs.side_cost(px * pq, d < 0)
                            run_qty -= pq
                        flipped = True
                        highwater = part_lvl
                else:
                    hit_tp = h[j] >= tp if d > 0 else l[j] <= tp
                    if hit_tp:
                        exit_px, exit_i, reason = tp, j, "TP"
                        break
                if not flipped and j - e_i + 1 >= max_hold:
                    exit_px, exit_i, reason = c[j], j, "TIME"
                    break
            else:
                stop = max(entry, highwater * (1 - trail_pct)) if d > 0 \
                    else min(entry, highwater * (1 + trail_pct))
                hit = l[j] <= stop if d > 0 else h[j] >= stop
                if hit:                                 # stop checked pre-ratchet
                    exit_px, exit_i, reason = stop, j, "TRAIL"
                    break
                highwater = max(highwater, h[j]) if d > 0 else min(highwater, l[j])
        if exit_px is None:
            exit_px, exit_i, reason = c[n - 1], n - 1, "EOD"

        exit_px = costs.slip(exit_px, direction_hurts_up=(d < 0))
        gross = d * (exit_px - entry) * run_qty + banked
        chg = costs.side_cost(entry * qty, d > 0) + costs.side_cost(exit_px * run_qty, d < 0)
        trades.append((symbol, idx[e_i], idx[exit_i], d, round(entry, 2),
                       round(exit_px, 2), qty, round(gross, 2), round(chg, 2),
                       round(gross - chg, 2), exit_i - e_i + 1, reason))
        busy_until = exit_i + 1
        day_count[day[e_i]] = day_count.get(day[e_i], 0) + 1

    return pd.DataFrame(trades, columns=metrics.TRADE_COLS)


def run_strategy(strategy, data: dict, params: dict, ctx: dict | None = None,
                 start=None, end=None) -> pd.DataFrame:
    """Run one strategy over {symbol: prepared df}, optional date slice."""
    all_trades = []
    for sym, df in data.items():
        d = df
        if start or end:
            d = df.loc[start or df.index[0]: end or df.index[-1]]
        if len(d) < 100:
            continue
        c2 = dict(ctx or {})
        c2["symbol"] = sym
        sig = strategy.generate(d, params, c2)
        sig = np.asarray(sig, dtype=np.int8)
        assert len(sig) == len(d), f"{strategy.name}: signal length mismatch"
        all_trades.append(run(d, sig, params, sym))
    if not all_trades:
        return metrics.empty_trades()
    return pd.concat(all_trades, ignore_index=True)
