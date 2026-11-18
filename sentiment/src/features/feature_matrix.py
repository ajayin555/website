from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.technical import compute_technical_indicators
from src.features.sentiment_features import load_sentiment
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def _normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def build_feature_matrix(
    asset: str,
    regime: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    root = _find_project_root()
    norm = _normalize_name(asset)
    eq_path = root / "data" / "raw" / "equities" / f"{norm}.parquet"
    if not eq_path.exists():
        raise FileNotFoundError(f"Equity parquet not found: {eq_path}")

    df = pd.read_parquet(eq_path)
    df.index = pd.DatetimeIndex(df.index)

    # Apply regime date filter
    reg_cfg = config.get("regimes", {}).get(regime, {})
    start = reg_cfg.get("full_sample_start")
    end = reg_cfg.get("full_sample_end")
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    df = compute_technical_indicators(df)

    sentiment_df = load_sentiment(asset, df.index)
    df = df.join(sentiment_df, how="left")

    # Target: next-day return
    target = df["Return"].shift(-1)

    # Drop last row (no target) and rows with NaN
    df = df.iloc[:-1]
    target = target.iloc[:-1]

    feature_cols = [
        "Return", "SMA_14", "EMA_14", "RSI", "CCI", "MFI", "ROC",
        "Williams_R", "OBV", "AD", "MACD", "ATR",
        "Buzz", "Mentions", "Emotion_vs_Fact", "Sentiment",
        "Optimism", "Pessimism", "Volatility_lang", "Uncertainty", "Fear",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].dropna()
    y = target.loc[X.index]

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    mask_test = (X.index >= test_start) & (X.index <= test_end)
    mask_train = X.index < test_start

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    return X_train, y_train, X_test, y_test
