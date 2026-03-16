from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NewsItem:
    symbol: str
    headline: str
    source: str
    published_at: datetime | None = None
    summary: str | None = None
    url: str | None = None


@dataclass
class SentimentRecord:
    symbol: str
    headline: str
    label: str
    raw_score: float
    signed_score: float
    conviction_score: float
    rationale: str | None = None
    source: str | None = None
    published_at: datetime | None = None


@dataclass
class SignalRecord:
    timestamp: datetime
    symbol: str
    close: float
    sma: float
    price_above_sma: bool
    avg_sentiment: float
    conviction: float
    action: str
    reason: str


@dataclass
class TradeDecision:
    symbol: str
    side: str
    qty: int
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
