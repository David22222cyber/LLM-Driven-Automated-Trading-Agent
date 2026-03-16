from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pprint import pprint

from llm_trading_agent.config import DEFAULT_CONFIG
from llm_trading_agent.pipeline import TradingPipeline


def main() -> None:
    pipeline = TradingPipeline(DEFAULT_CONFIG)
    result = pipeline.prepare_live_signal()
    signal = result["signal"]
    print("\n=== LIVE SIGNAL SUMMARY ===")
    print(f"Symbol: {signal.symbol}")
    print(f"Action: {signal.action}")
    print(f"Close: {signal.close:.2f}")
    print(f"SMA({DEFAULT_CONFIG.strategy.sma_window}): {signal.sma:.2f}")
    print(f"Avg Sentiment: {signal.avg_sentiment:.4f}")
    print(f"Conviction: {signal.conviction:.2f}/10")
    print(f"Reason: {signal.reason}\n")

    print("Recent headlines:")
    for item in result["news"]:
        print(f"- [{item.source}] {item.headline}")

    print("\nSentiment rows:")
    for row in result["sentiments"]:
        pprint(row)


if __name__ == "__main__":
    main()
