from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_trading_agent.config import DEFAULT_CONFIG
from llm_trading_agent.execution.paper_broker import PaperBroker
from llm_trading_agent.pipeline import TradingPipeline


def main() -> None:
    pipeline = TradingPipeline(DEFAULT_CONFIG)
    result = pipeline.prepare_live_signal()
    signal = result["signal"]
    print(f"Signal for {signal.symbol}: {signal.action} | reason={signal.reason}")
    if signal.action == "HOLD":
        print("No order submitted.")
        return

    broker = PaperBroker(DEFAULT_CONFIG.alpaca, DEFAULT_CONFIG.strategy)
    decision = broker.build_trade_decision(signal)
    order = broker.submit_trade(decision)
    print("Order submitted:")
    print(order)


if __name__ == "__main__":
    main()
