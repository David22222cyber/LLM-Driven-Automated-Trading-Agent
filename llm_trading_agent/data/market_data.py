from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from llm_trading_agent.config import DataConfig
from llm_trading_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MarketDataHandler:
    config: DataConfig

    def fetch_price_history(self, symbol: str, period: str | None = None, interval: str | None = None) -> pd.DataFrame:
        period = period or self.config.lookback_period
        interval = interval or self.config.interval
        logger.info("Downloading price history for %s | period=%s interval=%s", symbol, period, interval)
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No market data returned for {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        rename_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename_map)
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {symbol}: {missing}")

        out = df[required].copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        out["return"] = out["close"].pct_change().fillna(0.0)
        return out

    def fetch_benchmark_history(self) -> pd.DataFrame:
        return self.fetch_price_history(self.config.benchmark_symbol)
