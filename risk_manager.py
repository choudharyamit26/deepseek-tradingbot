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
        self.risk_per_trade_percent = (
            risk_per_trade_percent  # % of capital risked per trade
        )
        self.min_confidence = min_confidence
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
            logger.warning(f"Daily loss limit exceeded: {loss_percent:.2f}%")
            return False
        return True

    def calculate_position_size(self, capital, stop_loss_percent, entry_price):
        risk_amount = capital * (self.risk_per_trade_percent / 100)
        risk_per_share = entry_price * (stop_loss_percent / 100)
        quantity = int(risk_amount / risk_per_share)
        max_afford = int(capital / entry_price)
        half_cap = int(capital * 0.5 / entry_price)
        max_quantity = min(max_afford, max(half_cap, 1))
        return min(max(quantity, 0), max_quantity)

    def record_trade(self):
        self.daily_trade_count += 1

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
