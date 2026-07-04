"""Performance metrics computed from a trades DataFrame (net of costs)."""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

TRADE_COLS = ["symbol", "entry_ts", "exit_ts", "dir", "entry", "exit",
              "qty", "gross", "costs", "net", "bars", "reason"]


def empty_trades():
    return pd.DataFrame(columns=TRADE_COLS)


def compute(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "net": 0.0, "wr": 0.0, "pf": 0.0, "sharpe": 0.0,
                "maxdd_pct": 0.0, "exp_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    net = trades["net"].values
    wins, losses = net[net > 0], net[net < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")
    # daily pnl series -> annualized Sharpe on capital
    daily = trades.groupby(pd.to_datetime(trades["exit_ts"]).dt.normalize())["net"].sum()
    dr = daily / cfg.CAPITAL_PER_TRADE
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 5 and dr.std() > 0 else 0.0
    eq = daily.cumsum()
    dd = (eq - eq.cummax()) / cfg.CAPITAL_PER_TRADE
    notional = (trades["entry"] * trades["qty"]).values
    return {
        "trades": n,
        "net": round(float(net.sum()), 0),
        "wr": round(100 * len(wins) / n, 1),
        "pf": round(min(pf, 99.0), 2),
        "sharpe": round(sharpe, 2),
        "maxdd_pct": round(100 * float(dd.min()) if len(dd) else 0.0, 2),
        "exp_pct": round(100 * float((net / notional).mean()), 4),
        "avg_win": round(float(wins.mean()) if len(wins) else 0.0, 0),
        "avg_loss": round(float(losses.mean()) if len(losses) else 0.0, 0),
    }
