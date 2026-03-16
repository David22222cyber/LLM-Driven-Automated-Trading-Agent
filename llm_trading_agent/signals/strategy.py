from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llm_trading_agent.config import StrategyConfig
from llm_trading_agent.models import SentimentRecord, SignalRecord


@dataclass
class TradingStrategy:
    config: StrategyConfig

    def summarize_sentiment(self, sentiments: list[SentimentRecord]) -> dict[str, float | str]:
        if not sentiments:
            return {"avg_sentiment": 0.0, "avg_conviction": 0.0, "direction": "NEUTRAL"}
        avg_sentiment = sum(x.signed_score for x in sentiments) / len(sentiments)
        avg_conviction = sum(x.conviction_score for x in sentiments) / len(sentiments)
        direction = "NEUTRAL"
        if avg_sentiment >= self.config.positive_threshold:
            direction = "POSITIVE"
        elif avg_sentiment <= self.config.negative_threshold:
            direction = "NEGATIVE"
        return {
            "avg_sentiment": round(avg_sentiment, 4),
            "avg_conviction": round(avg_conviction, 2),
            "direction": direction,
        }

    def generate_live_signal(self, symbol: str, feature_df: pd.DataFrame, sentiments: list[SentimentRecord]) -> SignalRecord:
        latest = feature_df.iloc[-1]
        s = self.summarize_sentiment(sentiments)
        price_above_sma = bool(latest["price_above_sma"])
        action = "HOLD"
        reason = "Conditions not aligned."

        if price_above_sma and s["avg_sentiment"] >= self.config.positive_threshold and s["avg_conviction"] >= self.config.min_sentiment_score:
            action = "BUY"
            reason = "Trend positive and sentiment strong."
        elif self.config.allow_short and (not price_above_sma) and s["avg_sentiment"] <= self.config.negative_threshold and s["avg_conviction"] >= self.config.min_sentiment_score:
            action = "SELL"
            reason = "Trend weak and sentiment strongly negative."

        return SignalRecord(
            timestamp=feature_df.index[-1].to_pydatetime(),
            symbol=symbol,
            close=float(latest["close"]),
            sma=float(latest["sma"]),
            price_above_sma=price_above_sma,
            avg_sentiment=float(s["avg_sentiment"]),
            conviction=float(s["avg_conviction"]),
            action=action,
            reason=reason,
        )
