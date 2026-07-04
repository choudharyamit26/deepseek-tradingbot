"""Central config for the intraday strategy lab. All dates IST."""
from pathlib import Path

ROOT = Path(__file__).parent
REPO = ROOT.parent
STORE = ROOT / "data" / "store"
RESULTS = ROOT / "results"

# ── Study window ──────────────────────────────────────────────────────────────
START = "2025-07-04"
END = "2026-07-03"
IS_END = "2026-03-31"          # in-sample: START .. IS_END (~9 months)
OOS_START = "2026-04-01"       # out-of-sample: OOS_START .. END (~3 months)

INTERVAL = "5"                 # 5-minute bars
CHUNK_DAYS = 85                # Dhan intraday cap ~90 days/request
THROTTLE_S = 0.35              # be polite to the data API

# ── Universe selection ────────────────────────────────────────────────────────
N_STOCKS = 20
MIN_ADV_RS = 25e7              # Rs 25 cr min avg daily turnover
MIN_SESSIONS = 200             # require data coverage
MAX_OVERNIGHT_JUMP = 0.15      # corporate-action screen
NIFTY_SID = "13"               # NIFTY 50 index (IDX_I)

# ── Backtest execution ────────────────────────────────────────────────────────
CAPITAL_PER_TRADE = 100_000.0  # fixed notional, no compounding
SLIPPAGE_PCT = 0.02            # per side, in %
ENTRY_START = 9 * 60 + 30      # 09:30 (minutes from midnight)
ENTRY_END = 14 * 60 + 45       # last fresh entry 14:45
SQUARE_OFF = 15 * 60 + 10      # force-exit at first bar >= 15:10
MAX_HOLD_BARS = 36             # 3 hours of 5-min bars
MAX_TRADES_PER_DAY = 2         # per symbol — the opencode lab showed cost drag
                               # scales with frequency (10+/day -> -Rs140/trade)
ATR_LEN = 14

# ── Validation ────────────────────────────────────────────────────────────────
MIN_IS_TRADES = 100            # combos below this are rejected
WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1

# Survivor criteria (fixed up front)
SURVIVOR = dict(oos_pf=1.2, oos_sharpe=1.0, oos_trades=30, max_decay=0.5,
                wf_folds_min=5, wf_pf=1.0)  # WF gates added for batch>=3 (tightening only)
