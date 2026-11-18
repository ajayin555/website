from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SENTIMENT_COLS = [
    "Buzz", "Mentions", "Emotion_vs_Fact", "Sentiment",
    "Optimism", "Pessimism", "Volatility_lang", "Uncertainty", "Fear",
]


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def load_sentiment(asset: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    root = _find_project_root()
    path = root / "data" / "raw" / "sentiment" / f"{asset.replace(' ', '_')}.parquet"
    if not path.exists():
        logger.warning("Sentiment file not found for %s — returning zeros", asset)
        return pd.DataFrame(0.0, index=dates, columns=SENTIMENT_COLS)

    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(dates)
    df = df.ffill(limit=5)
    df = df.fillna(0.0)
    for col in SENTIMENT_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df[SENTIMENT_COLS]


def normalize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    std = df.std()
    std = std.replace(0, 1)
    return (df - df.mean()) / std
