from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llm_trading_agent.backtest.backtester import SimpleBacktester
from llm_trading_agent.config import DATA_DIR, DEFAULT_CONFIG
from llm_trading_agent.data.market_data import MarketDataHandler
from llm_trading_agent.pipeline import TradingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment-csv", type=str, default=None, help="Optional CSV with columns: date,signed_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TradingPipeline(DEFAULT_CONFIG)
    backtest_df = pipeline.build_backtest_frame(args.sentiment_csv)
    backtester = SimpleBacktester(DEFAULT_CONFIG.strategy)
    result = backtester.run(backtest_df)

    benchmark = MarketDataHandler(DEFAULT_CONFIG.data).fetch_benchmark_history()
    benchmark = benchmark.loc[backtest_df.index.min(): backtest_df.index.max()].copy()
    benchmark["equity"] = DEFAULT_CONFIG.strategy.initial_cash * (1 + benchmark["close"].pct_change().fillna(0.0)).cumprod()

    metrics = pd.DataFrame(
        [
            {"strategy": "LLM+Trend", **result.metrics},
            {
                "strategy": DEFAULT_CONFIG.data.benchmark_symbol + " Buy&Hold",
                "total_return": round(float(benchmark["equity"].iloc[-1] / benchmark["equity"].iloc[0] - 1), 4),
                "max_drawdown": round(float((benchmark["equity"] / benchmark["equity"].cummax() - 1).min()), 4),
                "sharpe_ratio": round(float((benchmark["equity"].pct_change().fillna(0.0).mean() / benchmark["equity"].pct_change().fillna(0.0).std(ddof=0)) * (252 ** 0.5)), 4),
            },
        ]
    )
    print("\n=== BACKTEST METRICS ===")
    print(metrics.to_string(index=False))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(DATA_DIR / "equity_curve.csv")
    result.trades.to_csv(DATA_DIR / "trades.csv", index=False)
    metrics.to_csv(DATA_DIR / "metrics.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(result.equity_curve.index, result.equity_curve["equity"], label="LLM+Trend")
    plt.plot(benchmark.index, benchmark["equity"], label=DEFAULT_CONFIG.data.benchmark_symbol + " Buy&Hold")
    plt.legend()
    plt.title(f"Equity Curve: {DEFAULT_CONFIG.strategy.symbol} Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "equity_curve.png", dpi=150)
    print(f"\nArtifacts written to: {Path(DATA_DIR).resolve()}")


if __name__ == "__main__":
    main()
