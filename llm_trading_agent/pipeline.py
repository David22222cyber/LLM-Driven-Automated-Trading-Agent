from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llm_trading_agent.config import AppConfig
from llm_trading_agent.data.market_data import MarketDataHandler
from llm_trading_agent.data.news_data import NewsFetcher
from llm_trading_agent.sentiment.finbert_agent import LocalFinBERTAgent
from llm_trading_agent.sentiment.openai_agent import OpenAISentimentAgent
from llm_trading_agent.signals.strategy import TradingStrategy
from llm_trading_agent.signals.technical import compute_technical_features
from llm_trading_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TradingPipeline:
    config: AppConfig

    def __post_init__(self) -> None:
        self.market_data = MarketDataHandler(self.config.data)
        self.news_fetcher = NewsFetcher(self.config.data)
        self.strategy = TradingStrategy(self.config.strategy)
        if self.config.sentiment.mode == "openai":
            self.sentiment_agent = OpenAISentimentAgent(self.config.openai)
        else:
            self.sentiment_agent = LocalFinBERTAgent(self.config.sentiment)

    def prepare_live_signal(self):
        symbol = self.config.strategy.symbol
        prices = self.market_data.fetch_price_history(symbol)
        features = compute_technical_features(prices, self.config.strategy.sma_window)
        news = self.news_fetcher.fetch(symbol)
        sentiments = self.sentiment_agent.analyze(symbol, news)
        signal = self.strategy.generate_live_signal(symbol, features, sentiments)
        return {
            "signal": signal,
            "news": news,
            "sentiments": sentiments,
            "features": features,
        }

    def build_backtest_frame(self, sentiment_csv: str | None = None) -> pd.DataFrame:
        symbol = self.config.strategy.symbol
        prices = self.market_data.fetch_price_history(symbol, period="12mo")
        features = compute_technical_features(prices, self.config.strategy.sma_window)
        if sentiment_csv is None:
            logger.warning("No historical sentiment CSV provided. Using a neutral placeholder series.")
            features["avg_sentiment"] = 0.0
        else:
            sent = pd.read_csv(sentiment_csv)
            sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
            sent = sent.groupby("date", as_index=False)["signed_score"].mean().rename(columns={"signed_score": "avg_sentiment"})
            features = features.reset_index().rename(columns={"Date": "date", "index": "date"})
            if "date" not in features.columns:
                features = features.rename(columns={features.columns[0]: "date"})
            features["date"] = pd.to_datetime(features["date"]).dt.normalize()
            features = features.merge(sent, on="date", how="left")
            features = features.set_index("date")
            features["avg_sentiment"] = features["avg_sentiment"].fillna(0.0)

        features["sentiment_rolling"] = features["avg_sentiment"].rolling(self.config.strategy.sentiment_lookback_days).mean()
        features["sentiment_signal"] = 0
        features.loc[
            features["sentiment_rolling"] >= self.config.strategy.positive_threshold,
            "sentiment_signal",
        ] = 1
        if self.config.strategy.allow_short:
            features.loc[
                features["sentiment_rolling"] <= self.config.strategy.negative_threshold,
                "sentiment_signal",
            ] = -1
        return features
