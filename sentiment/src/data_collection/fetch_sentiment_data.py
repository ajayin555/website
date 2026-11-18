"""
Sentiment data fetcher using NewsData.io.

Historical data: fetches via /archive endpoint, batched weekly to stay within
API credit limits, with raw article cache to avoid re-fetching.

Live data: fetches via /latest endpoint for the most recent articles.

Fallback: any day with no fetched articles gets a synthetic row so the
feature matrix is always fully populated.

API key is read from the NEWSDATA_API_KEY environment variable.
Create a .env file (see .env.example) — it is gitignored.
"""
from __future__ import annotations

import os
import time
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

# NewsData.io endpoints
_ARCHIVE_URL = "https://newsdata.io/api/1/archive"
_LATEST_URL  = "https://newsdata.io/api/1/latest"

# Base query params applied to every request
_BASE_PARAMS = {
    "country":  "in",
    "language": "en",
    "category": "breaking,business,technology,world,politics",
}

# Seconds between API requests — stay well under rate limits
_REQUEST_DELAY = 1.0

_FINBERT_PIPE: Any = None


# ---------------------------------------------------------------------------
# FinBERT pipeline
# ---------------------------------------------------------------------------

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
        logger.warning("FinBERT unavailable (%s) — keyword scoring only", exc)
        _FINBERT_PIPE = None
    return _FINBERT_PIPE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def _load_api_key() -> str | None:
    # Try environment first, then .env file
    key = os.environ.get("NEWSDATA_API_KEY")
    if key:
        return key
    root = _find_project_root()
    env_file = root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            key = os.environ.get("NEWSDATA_API_KEY")
        except ImportError:
            # dotenv not installed — parse manually
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("NEWSDATA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    return key or None


# ---------------------------------------------------------------------------
# NewsData.io fetchers
# ---------------------------------------------------------------------------

def _fetch_page(url: str, params: dict) -> tuple[list[dict], str | None]:
    """Single API call. Returns (articles, next_page_token)."""
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 422:
            # Date range not supported on free plan — archive not available
            return [], None
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("results") or []
        next_page = data.get("nextPage")
        return articles, next_page
    except Exception as exc:
        logger.debug("NewsData.io request failed: %s", exc)
        return [], None


def fetch_articles_archive(
    entity: str,
    from_date: str,
    to_date: str,
    api_key: str,
    max_pages: int = 3,
) -> list[dict]:
    """Fetch articles from /archive for a date range (paid plan feature)."""
    params = {
        **_BASE_PARAMS,
        "apikey":    api_key,
        "q":         entity,
        "from_date": from_date,
        "to_date":   to_date,
    }
    articles: list[dict] = []
    next_page = None
    for _ in range(max_pages):
        if next_page:
            params["page"] = next_page
        batch, next_page = _fetch_page(_ARCHIVE_URL, params)
        articles.extend(batch)
        if not next_page:
            break
        time.sleep(_REQUEST_DELAY)
    return articles


def fetch_articles_latest(entity: str, api_key: str) -> list[dict]:
    """Fetch the most recent articles from /latest."""
    params = {**_BASE_PARAMS, "apikey": api_key, "q": entity}
    articles, _ = _fetch_page(_LATEST_URL, params)
    return articles


# ---------------------------------------------------------------------------
# Article cache  (data/raw/sentiment/cache/{entity}_{from}_{to}.parquet)
# ---------------------------------------------------------------------------

def _cache_path(root: Path, entity: str, from_date: str, to_date: str) -> Path:
    safe = entity.replace(" ", "_").replace("/", "-")
    return root / "data" / "raw" / "sentiment" / "cache" / f"{safe}_{from_date}_{to_date}.parquet"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _articles_to_df(articles: list[dict]) -> pd.DataFrame:
    rows = []
    for a in articles:
        pub = a.get("pubDate") or a.get("pubDateTZ") or ""
        title = a.get("title") or ""
        desc  = a.get("description") or ""
        text  = f"{title}. {desc}".strip()
        rows.append({"pubDate": pub, "text": text})
    if not rows:
        return pd.DataFrame(columns=["pubDate", "text"])
    df = pd.DataFrame(rows)
    df["pubDate"] = pd.to_datetime(df["pubDate"], utc=True, errors="coerce")
    df = df.dropna(subset=["pubDate"])
    df["date"] = df["pubDate"].dt.tz_localize(None).dt.normalize()
    return df[["date", "text"]]


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------

def _text_stats(texts: list[str], entity: str) -> dict[str, float]:
    if not texts:
        return {c: 0.0 for c in ["Buzz", "Mentions", "Emotion_vs_Fact", "Volatility_lang", "Uncertainty"]}
    combined = " ".join(texts).lower()
    entity_kw = entity.lower().split()[0]
    buzz     = float(len(texts))
    mentions = float(combined.count(entity_kw))
    emotion_words = {"feel", "fear", "hope", "worry", "great", "terrible", "love", "hate", "excited", "panic"}
    fact_words    = {"report", "data", "announced", "percent", "quarter", "result", "revenue"}
    n_emotion = sum(combined.count(w) for w in emotion_words)
    n_fact    = sum(combined.count(w) for w in fact_words)
    vol_words   = {"volatile", "crash", "surge", "spike", "plunge", "rally", "swing"}
    uncert_words = {"uncertain", "unclear", "doubt", "maybe", "could", "might", "risk"}
    return {
        "Buzz":            buzz,
        "Mentions":        mentions,
        "Emotion_vs_Fact": n_emotion / max(n_fact, 1),
        "Volatility_lang": float(sum(combined.count(w) for w in vol_words)),
        "Uncertainty":     float(sum(combined.count(w) for w in uncert_words)),
    }


def score_with_finbert(texts: list[str], entity: str) -> dict[str, float]:
    stats = _text_stats(texts, entity)
    pipe  = _get_finbert_pipeline()
    if pipe is None or not texts:
        return {
            **stats,
            "Sentiment": 0.0,
            "Optimism":  0.0,
            "Pessimism": 0.0,
            "Fear":      stats["Uncertainty"] * 0.5,
        }
    pos_scores, neg_scores = [], []
    for text in texts[:20]:
        try:
            result = pipe(text[:512])[0]
            label  = result["label"].lower()
            score  = result["score"]
            if label == "positive":
                pos_scores.append(score)
            elif label == "negative":
                neg_scores.append(score)
        except Exception:
            continue
    pos_avg = float(np.mean(pos_scores)) if pos_scores else 0.0
    neg_avg = float(np.mean(neg_scores)) if neg_scores else 0.0
    return {
        **stats,
        "Sentiment": pos_avg - neg_avg,
        "Optimism":  pos_avg,
        "Pessimism": neg_avg,
        "Fear":      neg_avg * 0.7 + stats["Uncertainty"] * 0.3,
    }


def _synthetic_row(rng: np.random.Generator) -> dict[str, float]:
    return {
        "Buzz":            float(rng.integers(0, 5)),
        "Mentions":        float(rng.integers(0, 3)),
        "Emotion_vs_Fact": float(rng.uniform(0.5, 1.5)),
        "Sentiment":       float(rng.normal(0, 0.1)),
        "Optimism":        float(max(rng.normal(0.5, 0.15), 0)),
        "Pessimism":       float(max(rng.normal(0.5, 0.15), 0)),
        "Volatility_lang": float(rng.uniform(0, 2)),
        "Uncertainty":     float(rng.uniform(0, 2)),
        "Fear":            float(max(rng.normal(0.4, 0.15), 0)),
    }


# ---------------------------------------------------------------------------
# Per-asset builder
# ---------------------------------------------------------------------------

def _build_asset_sentiment(
    asset: str,
    trading_days: pd.DatetimeIndex,
    api_key: str | None,
    rng: np.random.Generator,
    root: Path,
) -> pd.DataFrame:
    """
    Build daily sentiment DataFrame for one asset.

    Strategy:
    - If api_key available: fetch weekly batches from /archive, cache raw
      articles, aggregate texts per trading day, score with FinBERT.
    - Any day with no articles (API unavailable, date too old for free plan,
      or no relevant news): fill with synthetic row.
    - Latest articles (/latest) are fetched once and merged for recent days.
    """
    entity = asset.replace("Nifty ", "").strip()
    rows: dict[pd.Timestamp, dict] = {}

    if api_key:
        # Fetch latest articles once and map to recent trading days
        logger.info("  [%s] Fetching latest articles from NewsData.io ...", entity)
        latest_articles = fetch_articles_latest(entity, api_key)
        latest_df = _articles_to_df(latest_articles)
        time.sleep(_REQUEST_DELAY)

        # Fetch historical in weekly batches
        weeks = pd.date_range(trading_days.min(), trading_days.max(), freq="W-FRI")
        total_fetched = 0

        for week_end in weeks:
            week_start = week_end - pd.Timedelta(days=6)
            from_str = week_start.strftime("%Y-%m-%d")
            to_str   = week_end.strftime("%Y-%m-%d")
            cache_p  = _cache_path(root, entity, from_str, to_str)

            cached = _load_cache(cache_p)
            if cached is not None:
                week_df = cached
            else:
                articles = fetch_articles_archive(entity, from_str, to_str, api_key)
                week_df  = _articles_to_df(articles)
                _save_cache(cache_p, week_df)
                time.sleep(_REQUEST_DELAY)

            total_fetched += len(week_df)
            # Merge latest into week_df for the most recent week
            if not latest_df.empty:
                week_df = pd.concat([week_df, latest_df], ignore_index=True)

            # Aggregate per trading day in this week
            week_trading = [d for d in trading_days if week_start <= d <= week_end]
            for day in week_trading:
                day_texts = week_df.loc[week_df["date"] == day, "text"].tolist() if not week_df.empty else []
                rows[day] = score_with_finbert(day_texts, entity) if day_texts else None

        logger.info(
            "  [%s] NewsData.io: %d articles fetched across %d weeks",
            entity, total_fetched, len(weeks),
        )

    # Fill remaining days (no API key, archive unavailable, or no articles) with synthetic
    n_synthetic = 0
    result_rows = []
    for day in trading_days:
        scores = rows.get(day)
        if scores is None:
            scores = _synthetic_row(rng)
            n_synthetic += 1
        result_rows.append({"date": day, **scores})

    if n_synthetic:
        logger.info("  [%s] %d / %d days filled with synthetic data", entity, n_synthetic, len(trading_days))

    df = pd.DataFrame(result_rows).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    assets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    live_scrape: bool = False,   # kept for CLI compatibility; ignored (NewsData.io always used)
) -> dict[str, pd.DataFrame]:
    """
    Fetch / build sentiment data for all sector assets.

    Reads NEWSDATA_API_KEY from the environment or .env file.
    Falls back to fully synthetic data if the key is not set.

    Raw article batches are cached under data/raw/sentiment/cache/ so
    re-running does not consume additional API credits.
    """
    root = _find_project_root()
    config     = load_config(root / "config" / "config.yaml")
    assets_cfg = load_config(root / "config" / "assets.yaml")

    regimes    = config.get("regimes", {})
    starts     = [r["full_sample_start"] for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_start")]
    ends_list  = [r["full_sample_end"]   for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_end")]
    effective_start = start or min(starts)
    effective_end   = end   or max(ends_list)

    if assets is None:
        sectors = assets_cfg.get("sectoral", {}).get("sectors", [])
        assets  = [s["name"] for s in sectors]

    api_key = _load_api_key()
    if api_key:
        logger.info("NewsData.io API key found — will fetch real articles (archive + latest)")
    else:
        logger.warning(
            "NEWSDATA_API_KEY not set — using synthetic sentiment. "
            "Create a .env file with your key (see .env.example)."
        )

    trading_days = pd.bdate_range(effective_start, effective_end)
    rng = np.random.default_rng(42)
    result: dict[str, pd.DataFrame] = {}

    for asset in assets:
        logger.info("Building sentiment for: %s", asset)
        out_path = root / "data" / "raw" / "sentiment" / f"{asset.replace(' ', '_')}.parquet"
        df = _build_asset_sentiment(asset, trading_days, api_key, rng, root)
        save_parquet(df, out_path)
        result[asset] = df
        logger.info("Sentiment saved for %s (%d rows)", asset, len(df))

    return result
