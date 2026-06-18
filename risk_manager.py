# risk_manager.py
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        dhan_api=None,
        initial_capital=None,
        max_daily_trades=5,
        max_daily_loss_percent=2,
        risk_per_trade_percent=2,
        min_confidence=65,
        max_position_capital_pct=100.0,  # max % of BUYING POWER in one position
        cash_buffer_pct=20.0,
        leverage=1.0,  # intraday leverage multiplier (e.g. 5.0 = 5x MIS margin)
    ):
        if dhan_api is not None and initial_capital is None:
            initial_capital = dhan_api.get_available_balance()
            logger.info("Fetched balance from Dhan API: %.2f", initial_capital)
        if initial_capital is None:
            initial_capital = 100000
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.dhan_api = dhan_api
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_percent = max_daily_loss_percent
        self.risk_per_trade_percent = risk_per_trade_percent
        self.min_confidence = min_confidence
        # Feature 5: explicit cash buffer controls
        self.max_position_capital_pct = max_position_capital_pct  # max % of buying power in one position
        self.cash_buffer_pct = cash_buffer_pct                    # % of buying power always kept undeployed
        self.leverage = leverage if leverage and leverage > 0 else 1.0  # intraday margin multiplier
        self.daily_trade_count = 0
        self.daily_pnl = 0

    def check_daily_trade_limit(self):
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
        return True

    def check_daily_loss_limit(self):
        loss_percent = (
            (self.initial_capital - self.current_capital) / self.initial_capital * 100
        )
        if loss_percent > self.max_daily_loss_percent:
            logger.warning("Daily loss limit exceeded: %.2f%%", loss_percent)
            return False
        return True

    def check_cash_buffer(self, deployed_capital: float) -> bool:
        """
        Returns True if buying power is available to open another position.
        `deployed_capital` is the GROSS value of currently open positions.
        Blocks new positions when deployed >= (100 - cash_buffer_pct)% of total
        buying power (current_capital x leverage).
        Default: max 80% of buying power deployed, always keep 20% undeployed.
        """
        buying_power = self.current_capital * self.leverage
        max_deployable = buying_power * (1.0 - self.cash_buffer_pct / 100.0)
        if deployed_capital >= max_deployable:
            logger.warning(
                "Cash buffer enforced: deployed=%.2f >= limit=%.2f (%.0f%% of buying power %.2f = %.2f x %.1fx, buffer=%.0f%%)",
                deployed_capital, max_deployable,
                100 - self.cash_buffer_pct, buying_power,
                self.current_capital, self.leverage, self.cash_buffer_pct,
            )
            return False
        return True

    def calculate_position_size(self, capital, stop_loss_percent, entry_price):
        if stop_loss_percent <= 0 or entry_price <= 0:
            return 0
        # Risk-based sizing uses REAL cash: a stop-out loses actual money, not
        # leveraged money, so risk_per_trade is a % of unleveraged capital.
        risk_amount = capital * (self.risk_per_trade_percent / 100)
        risk_per_share = entry_price * (stop_loss_percent / 100)
        quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

        # Intraday leverage expands buying power: with 5x MIS margin a Rs1,000
        # account can hold positions worth up to Rs5,000 gross.
        buying_power = capital * self.leverage

        # Hard cap: no single position > max_position_capital_pct% of BUYING POWER.
        # e.g. capital=1000, leverage=5, cap=25% -> max 1250 gross in one position.
        max_position_value = buying_power * (self.max_position_capital_pct / 100.0)
        max_by_cap = int(max_position_value / entry_price) if entry_price > 0 else quantity

        max_afford = int(buying_power / entry_price) if entry_price > 0 else 0
        return min(max(quantity, 0), max_afford, max_by_cap)

    def record_trade(self):
        self.daily_trade_count += 1

    def record_pnl(self, pnl):
        self.current_capital += pnl
        self.daily_pnl += pnl

    def update_capital(self, new_capital):
        self.current_capital = new_capital

    def reset_daily(self):
        """Call at start of each trading day"""
        self.daily_trade_count = 0
        self.daily_pnl = 0
        if self.dhan_api is not None:
            balance = self.dhan_api.get_available_balance()
            if balance > 0:
                self.current_capital = balance
                logger.info("Reset capital from Dhan API: %.2f", balance)
