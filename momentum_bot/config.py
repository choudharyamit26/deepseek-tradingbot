"""All tunable parameters for the momentum bot. Edit here, nowhere else."""

# ── Market timing (IST) ───────────────────────────────────────────────────────
OR_START_TIME        = "09:15"   # NSE open — opening range begins
OR_END_TIME          = "09:30"   # First 15-min bar closes; scanner fires
ENTRY_START_TIME     = "09:30"   # Earliest entry after OR is sealed
ENTRY_END_TIME       = "14:00"   # No new entries after this
EXIT_ALL_TIME        = "14:45"   # Hard force-close of every open position

# ── Sector selection ──────────────────────────────────────────────────────────
TOP_N_SECTORS           = 2      # Number of sectors to trade each day
MIN_SECTOR_MOVE_PCT     = 0.20   # Sector score (weighted abs move) must clear this
MIN_SECTOR_STOCKS       = 2      # Need at least N stocks with data to score sector

# Neutral zone: if the sector's signed avg OR move is inside ±this%, direction
# is ambiguous noise (e.g. Banking was -0.18% today but is Bullish multi-TF).
# We refuse to call it BULL or BEAR and skip the sector entirely.
MIN_SECTOR_DIRECTION_PCT = 0.20  # e.g. -0.20% ≤ avg_move ≤ +0.20% → NEUTRAL, skip

# 5-day trend alignment: the sector's OR direction must agree with its 5-day
# price trend (last-close vs close-5-days-ago). Prevents trading IT BUY on a
# gap-up day when IT is in a -1.57%/1W, -4.80%/1M downtrend.
REQUIRE_TREND_ALIGNMENT = True

# ── Stock selection (within each selected sector) ─────────────────────────────
TOP_STOCKS_PER_SECTOR = 2        # Watch the top-N movers per selected sector
MIN_OR_WIDTH_PCT      = 0.15     # OR (high-low)/close must be ≥ 0.15% — skip flat opens

# ── Entry filter ─────────────────────────────────────────────────────────────
BREAKOUT_BUFFER_PCT   = 0.05     # Price must close > ORH + 0.05% (avoid false breaks)
ENTRY_VOLUME_MULTIPLIER = 1.5    # 3-min entry bar volume must be > N × bar-avg

# ── Risk / position sizing ────────────────────────────────────────────────────
RR_RATIO              = 1.5      # Target distance = RR × OR width from entry
MAX_STOP_PCT          = 1.5      # Cap stop at 1.5% of entry price regardless of OR width
MIN_STOP_PCT          = 0.15     # Stop must be at least 0.15% (avoids tick-wide stops)
POSITION_SIZE_PCT     = 0.20     # Fraction of available cash per trade (20%)
MAX_OPEN_POSITIONS    = 3        # Hard cap on concurrent open positions

# ── Per-position time exit ─────────────────────────────────────────────────────
# If neither SL nor target is hit within this many minutes of entry,
# the position is force-closed mid-session (not waiting until EXIT_ALL_TIME).
# EXIT_ALL_TIME is still the hard backstop for anything still open at day-end.
TIME_EXIT_MINUTES     = 60       # Minutes after entry before time exit fires

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR               = "trading_logs"
