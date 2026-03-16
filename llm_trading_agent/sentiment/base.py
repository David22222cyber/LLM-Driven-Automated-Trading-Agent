from __future__ import annotations

from abc import ABC, abstractmethod

from llm_trading_agent.models import NewsItem, SentimentRecord


class BaseSentimentAgent(ABC):
    @abstractmethod
    def analyze(self, symbol: str, news_items: list[NewsItem]) -> list[SentimentRecord]:
        raise NotImplementedError
