from __future__ import annotations

import pandas as pd


def compute_technical_features(price_df: pd.DataFrame, sma_window: int) -> pd.DataFrame:
    df = price_df.copy()
    df["sma"] = df["close"].rolling(sma_window).mean()
    df["price_above_sma"] = df["close"] > df["sma"]
    df["daily_return"] = df["close"].pct_change().fillna(0.0)
    return df
