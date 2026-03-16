from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import quote_plus

import feedparser
import requests

from llm_trading_agent.config import DataConfig
from llm_trading_agent.models import NewsItem
from llm_trading_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class NewsFetcher:
    config: DataConfig

    def fetch(self, symbol: str, company_hint: str | None = None) -> list[NewsItem]:
        query = f'{symbol} stock'
        if company_hint:
            query = f'{symbol} {company_hint} stock'
        google_rss = f'https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en'
        logger.info("Fetching news via RSS for %s", symbol)
        parsed = feedparser.parse(google_rss)
        items: list[NewsItem] = []
        cutoff = datetime.utcnow() - timedelta(days=self.config.news_days)

        for entry in parsed.entries[: self.config.news_limit * 3]:
            published = None
            if getattr(entry, 'published', None):
                try:
                    published = parsedate_to_datetime(entry.published).replace(tzinfo=None)
                except Exception:
                    published = None
            if published and published < cutoff:
                continue

            source = "Google News RSS"
            if getattr(entry, "source", None):
                source = entry.source.get("title", source)

            items.append(
                NewsItem(
                    symbol=symbol,
                    headline=entry.title,
                    source=source,
                    published_at=published,
                    summary=getattr(entry, "summary", None),
                    url=getattr(entry, "link", None),
                )
            )
            if len(items) >= self.config.news_limit:
                break

        if not items:
            raise ValueError(f"No recent news found for {symbol}")
        return items


@dataclass
class AlpacaNewsFetcher:
    api_key: str
    secret_key: str
    base_url: str = "https://data.alpaca.markets/v1beta1/news"

    def fetch(self, symbols: Iterable[str], limit: int = 10) -> list[NewsItem]:
        symbol_csv = ",".join(symbols)
        response = requests.get(
            self.base_url,
            params={"symbols": symbol_csv, "limit": limit},
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        items: list[NewsItem] = []
        for article in payload.get("news", []):
            created_at = article.get("created_at")
            published = None
            if created_at:
                published = datetime.fromisoformat(created_at.replace("Z", "+00:00")).replace(tzinfo=None)
            items.append(
                NewsItem(
                    symbol=(article.get("symbols") or [""])[0],
                    headline=article.get("headline", ""),
                    source=article.get("source", "Alpaca News"),
                    published_at=published,
                    summary=article.get("summary"),
                    url=article.get("url"),
                )
            )
        return items
