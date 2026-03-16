from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline

from llm_trading_agent.config import SentimentConfig
from llm_trading_agent.models import NewsItem, SentimentRecord
from llm_trading_agent.sentiment.base import BaseSentimentAgent
from llm_trading_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LocalFinBERTAgent(BaseSentimentAgent):
    config: SentimentConfig

    def __post_init__(self) -> None:
        logger.info("Loading local sentiment model: %s", self.config.local_model_name)
        self.classifier = pipeline(
            "text-classification",
            model=self.config.local_model_name,
            tokenizer=self.config.local_model_name,
            truncation=True,
        )

    @staticmethod
    def _signed_score(label: str, score: float) -> float:
        label = label.lower()
        if "positive" in label:
            return score
        if "negative" in label:
            return -score
        return 0.0

    @staticmethod
    def _conviction_from_signed(signed_score: float) -> float:
        # maps [-1, 1] to [0, 10]
        return round((abs(signed_score) * 10.0), 2)

    def analyze(self, symbol: str, news_items: list[NewsItem]) -> list[SentimentRecord]:
        texts = [item.headline for item in news_items[: self.config.max_headlines_per_call]]
        preds = self.classifier(texts)
        records: list[SentimentRecord] = []
        for item, pred in zip(news_items, preds):
            signed = self._signed_score(pred["label"], float(pred["score"]))
            records.append(
                SentimentRecord(
                    symbol=symbol,
                    headline=item.headline,
                    label=pred["label"].upper(),
                    raw_score=float(pred["score"]),
                    signed_score=signed,
                    conviction_score=self._conviction_from_signed(signed),
                    rationale="Local financial sentiment model classification.",
                    source=item.source,
                    published_at=item.published_at,
                )
            )
        return records
