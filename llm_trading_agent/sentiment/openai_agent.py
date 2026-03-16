from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from llm_trading_agent.config import OpenAIConfig
from llm_trading_agent.models import NewsItem, SentimentRecord
from llm_trading_agent.sentiment.base import BaseSentimentAgent


SCHEMA = {
    "name": "financial_sentiment_batch",
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string"},
                        "label": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
                        "signed_score": {"type": "number"},
                        "conviction_score": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["headline", "label", "signed_score", "conviction_score", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}


@dataclass
class OpenAISentimentAgent(BaseSentimentAgent):
    config: OpenAIConfig

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.config.api_key)

    def analyze(self, symbol: str, news_items: list[NewsItem]) -> list[SentimentRecord]:
        headlines = [item.headline for item in news_items]
        prompt = (
            f"You are a financial sentiment analyst. For ticker {symbol}, classify each headline as "
            "POSITIVE, NEGATIVE, or NEUTRAL for near-term stock impact. Also return a signed_score in [-1, 1] "
            "and a conviction_score in [0, 10]. Keep rationale very short. Headline list:\n" +
            "\n".join(f"- {h}" for h in headlines)
        )

        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": SCHEMA["name"], "schema": SCHEMA["schema"], "strict": True},
            },
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        output: list[SentimentRecord] = []
        by_headline = {item.headline: item for item in news_items}
        for row in parsed["items"]:
            item = by_headline[row["headline"]]
            output.append(
                SentimentRecord(
                    symbol=symbol,
                    headline=row["headline"],
                    label=row["label"],
                    raw_score=abs(float(row["signed_score"])),
                    signed_score=float(row["signed_score"]),
                    conviction_score=float(row["conviction_score"]),
                    rationale=row["rationale"],
                    source=item.source,
                    published_at=item.published_at,
                )
            )
        return output
