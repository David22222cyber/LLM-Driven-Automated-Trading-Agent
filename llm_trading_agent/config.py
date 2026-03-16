from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str = "YOUR API KEY"
    secret_key: str = "YOUR SECRET KEY"
    paper: bool = True


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str = "YOUR OPENAI API KEY"
    model: str = "gpt-4.1-mini"


@dataclass(frozen=True)
class DataConfig:
    benchmark_symbol: str = "SPY"
    lookback_period: str = "18mo"
    interval: str = "1d"
    news_limit: int = 12
    news_days: int = 7


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "AAPL"
    sma_window: int = 50
    sentiment_lookback_days: int = 3
    min_sentiment_score: float = 6.5
    positive_threshold: float = 0.15
    negative_threshold: float = -0.15
    position_size_fraction: float = 0.95
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.08
    max_holding_days: int = 10
    initial_cash: float = 100_000.0
    allow_short: bool = False


@dataclass(frozen=True)
class SentimentConfig:
    mode: Literal["local", "openai"] = "local"
    local_model_name: str = "ProsusAI/finbert"
    max_headlines_per_call: int = 8


@dataclass(frozen=True)
class AppConfig:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)


DEFAULT_CONFIG = AppConfig()
