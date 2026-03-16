from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from llm_trading_agent.config import StrategyConfig
from llm_trading_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class SimpleBacktester:
    config: StrategyConfig

    def run(self, df: pd.DataFrame) -> BacktestResult:
        data = df.copy().dropna(subset=["sma", "sentiment_signal"])
        cash = self.config.initial_cash
        shares = 0
        entry_price = None
        entry_date = None
        equity_rows = []
        trade_rows = []

        for dt, row in data.iterrows():
            close = float(row["close"])
            signal = int(row["sentiment_signal"])
            price_above_sma = bool(row["price_above_sma"])

            if shares > 0 and entry_price is not None:
                holding_days = (dt - entry_date).days if entry_date is not None else 0
                stop_hit = close <= entry_price * (1.0 - self.config.stop_loss_pct)
                tp_hit = close >= entry_price * (1.0 + self.config.take_profit_pct)
                timeout = holding_days >= self.config.max_holding_days
                trend_break = not price_above_sma
                negative_news = signal < 0
                if stop_hit or tp_hit or timeout or trend_break or negative_news:
                    cash += shares * close
                    trade_rows.append(
                        {
                            "date": dt,
                            "action": "SELL",
                            "price": close,
                            "shares": shares,
                            "reason": "exit_rule",
                        }
                    )
                    shares = 0
                    entry_price = None
                    entry_date = None

            if shares == 0 and price_above_sma and signal > 0:
                budget = cash * self.config.position_size_fraction
                buy_qty = int(budget // close)
                if buy_qty > 0:
                    cash -= buy_qty * close
                    shares = buy_qty
                    entry_price = close
                    entry_date = dt
                    trade_rows.append(
                        {
                            "date": dt,
                            "action": "BUY",
                            "price": close,
                            "shares": buy_qty,
                            "reason": "trend_plus_sentiment",
                        }
                    )

            equity = cash + shares * close
            equity_rows.append({"date": dt, "equity": equity, "cash": cash, "shares": shares, "close": close})

        equity_df = pd.DataFrame(equity_rows).set_index("date")
        trades_df = pd.DataFrame(trade_rows)
        metrics = self._compute_metrics(equity_df)
        return BacktestResult(equity_curve=equity_df, trades=trades_df, metrics=metrics)

    def _compute_metrics(self, equity_df: pd.DataFrame) -> dict[str, float]:
        returns = equity_df["equity"].pct_change().fillna(0.0)
        total_return = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1.0
        running_max = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] / running_max - 1.0
        max_drawdown = drawdown.min()
        daily_vol = returns.std(ddof=0)
        sharpe = np.sqrt(252) * returns.mean() / daily_vol if daily_vol > 0 else 0.0
        return {
            "total_return": round(float(total_return), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe_ratio": round(float(sharpe), 4),
        }
