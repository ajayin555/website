"""
Sentiment data fetcher.

Production use requires a live news archive or licensed sentiment feed.
Synthetic fallback used for historical backtesting scaffold.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils.io import load_config, save_parquet
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SENTIMENT_COLS = [
    "Buzz", "Mentions", "Emotion_vs_Fact", "Sentiment",
    "Optimism", "Pessimism", "Volatility_lang", "Uncertainty", "Fear",
]

_FINBERT_PIPE: Any = None


def _get_finbert_pipeline():
    global _FINBERT_PIPE
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE
    try:
        from transformers import pipeline as hf_pipeline
        _FINBERT_PIPE = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT pipeline loaded")
    except Exception as exc:
        logger.warning("FinBERT unavailable (%s) — using synthetic fallback", exc)
        _FINBERT_PIPE = None
    return _FINBERT_PIPE


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def scrape_headlines(entity: str, date: str) -> list[str]:
    url = (
        f"https://economictimes.indiatimes.com/rssfeedsearch.cms"
        f"?query={requests.utils.quote(entity)}&language=EN&format=atom"
    )
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
        if not titles:
            titles = re.findall(r"<title>(.*?)</title>", resp.text)[1:]  # skip feed title
        return [t.strip() for t in titles if t.strip()]
    except Exception:
        return []


def _text_stats(texts: list[str], entity: str) -> dict[str, float]:
    if not texts:
        return {c: 0.0 for c in ["Buzz", "Mentions", "Emotion_vs_Fact", "Volatility_lang", "Uncertainty"]}
    combined = " ".join(texts).lower()
    entity_lower = entity.lower().split()[0]
    buzz = float(len(texts))
    mentions = float(combined.count(entity_lower))
    emotion_words = {"feel", "fear", "hope", "worry", "great", "terrible", "love", "hate", "excited", "panic"}
    fact_words = {"report", "data", "announced", "percent", "quarter", "result", "revenue"}
    n_emotion = sum(combined.count(w) for w in emotion_words)
    n_fact = sum(combined.count(w) for w in fact_words)
    emotion_vs_fact = n_emotion / max(n_fact, 1)
    vol_words = {"volatile", "crash", "surge", "spike", "plunge", "rally", "swing"}
    volatility_lang = float(sum(combined.count(w) for w in vol_words))
    uncert_words = {"uncertain", "unclear", "doubt", "maybe", "could", "might", "risk"}
    uncertainty = float(sum(combined.count(w) for w in uncert_words))
    return {
        "Buzz": buzz,
        "Mentions": mentions,
        "Emotion_vs_Fact": emotion_vs_fact,
        "Volatility_lang": volatility_lang,
        "Uncertainty": uncertainty,
    }


def score_with_finbert(texts: list[str], entity: str) -> dict[str, float]:
    stats = _text_stats(texts, entity)
    pipe = _get_finbert_pipeline()
    if pipe is None or not texts:
        return {
            **stats,
            "Sentiment": 0.0,
            "Optimism": 0.0,
            "Pessimism": 0.0,
            "Fear": stats["Uncertainty"] * 0.5,
        }

    pos_scores, neg_scores, neu_scores = [], [], []
    for text in texts[:20]:  # cap at 20 to keep it fast
        try:
            result = pipe(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "positive":
                pos_scores.append(score)
            elif label == "negative":
                neg_scores.append(score)
            else:
                neu_scores.append(score)
        except Exception:
            continue

    n = max(len(texts), 1)
    pos_avg = np.mean(pos_scores) if pos_scores else 0.0
    neg_avg = np.mean(neg_scores) if neg_scores else 0.0

    return {
        **stats,
        "Sentiment": pos_avg - neg_avg,
        "Optimism": pos_avg,
        "Pessimism": neg_avg,
        "Fear": neg_avg * 0.7 + stats["Uncertainty"] * 0.3,
    }


def _synthetic_row(rng: np.random.Generator) -> dict[str, float]:
    return {
        "Buzz": float(rng.integers(0, 5)),
        "Mentions": float(rng.integers(0, 3)),
        "Emotion_vs_Fact": float(rng.uniform(0.5, 1.5)),
        "Sentiment": float(rng.normal(0, 0.1)),
        "Optimism": float(max(rng.normal(0.5, 0.15), 0)),
        "Pessimism": float(max(rng.normal(0.5, 0.15), 0)),
        "Volatility_lang": float(rng.uniform(0, 2)),
        "Uncertainty": float(rng.uniform(0, 2)),
        "Fear": float(max(rng.normal(0.4, 0.15), 0)),
    }


def run(
    assets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    live_scrape: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generate sentiment data for all sector assets.

    By default uses synthetic data for the full historical window (fast, ~seconds).
    Pass live_scrape=True to attempt Economic Times RSS scraping for recent dates only
    (last 30 calendar days); historical dates always fall back to synthetic data.
    """
    root = _find_project_root()
    config = load_config(root / "config" / "config.yaml")
    assets_cfg = load_config(root / "config" / "assets.yaml")

    regimes = config.get("regimes", {})
    starts = [r["full_sample_start"] for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_start")]
    ends_list = [r["full_sample_end"] for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_end")]
    effective_start = start or min(starts)
    effective_end = end or max(ends_list)

    if assets is None:
        sectors = assets_cfg.get("sectoral", {}).get("sectors", [])
        assets = [s["name"] for s in sectors]

    trading_days = pd.bdate_range(effective_start, effective_end)
    live_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    rng = np.random.default_rng(42)
    result: dict[str, pd.DataFrame] = {}

    for asset in assets:
        out_path = root / "data" / "raw" / "sentiment" / f"{asset.replace(' ', '_')}.parquet"
        entity_query = asset.replace("Nifty ", "").strip()
        rows = []

        for day in trading_days:
            if live_scrape and day >= live_cutoff:
                headlines = scrape_headlines(entity_query, day.strftime("%Y-%m-%d"))
                scores = score_with_finbert(headlines, entity_query) if headlines else _synthetic_row(rng)
            else:
                scores = _synthetic_row(rng)
            rows.append({"date": day, **scores})

        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "Date"
        save_parquet(df, out_path)
        result[asset] = df
        logger.info("Sentiment saved for %s (%d rows, mode=%s)", asset, len(df), "live+synthetic" if live_scrape else "synthetic")

    return result
