from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from llm_trading_agent.config import AlpacaConfig, StrategyConfig
from llm_trading_agent.models import SignalRecord, TradeDecision
from llm_trading_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PaperBroker:
    broker_config: AlpacaConfig
    strategy_config: StrategyConfig

    def __post_init__(self) -> None:
        self.client = TradingClient(
            api_key=self.broker_config.api_key,
            secret_key=self.broker_config.secret_key,
            paper=self.broker_config.paper,
        )

    def build_trade_decision(self, signal: SignalRecord) -> TradeDecision:
        account = self.client.get_account()
        buying_power = float(account.buying_power)
        notional = buying_power * self.strategy_config.position_size_fraction
        qty = max(int(notional // signal.close), 0)
        if qty <= 0:
            raise ValueError("Calculated quantity is zero; reduce price or increase buying power.")

        stop_loss_price = round(signal.close * (1.0 - self.strategy_config.stop_loss_pct), 2)
        take_profit_price = round(signal.close * (1.0 + self.strategy_config.take_profit_pct), 2)
        side = "buy" if signal.action == "BUY" else "sell"
        return TradeDecision(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            metadata={"signal_reason": signal.reason, "conviction": signal.conviction},
        )

    def submit_trade(self, decision: TradeDecision):
        side = OrderSide.BUY if decision.side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=decision.symbol,
            qty=decision.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            stop_loss=StopLossRequest(stop_price=decision.stop_loss_price),
            take_profit=TakeProfitRequest(limit_price=decision.take_profit_price),
        )
        logger.info("Submitting %s %s x %s", decision.side.upper(), decision.symbol, decision.qty)
        return self.client.submit_order(order_data=request)
