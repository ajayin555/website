from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.io import load_config, save_parquet
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Tickers that need remapping before appending .NS
_TICKER_MAP = {
    "M&M": "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "LTM": "LTIM.NS",
    "TMPV": "TVSMOTOR.NS",
}


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root (config/config.yaml not found)")


def _normalize_sector_name(name: str) -> str:
    name = name.replace("Nifty ", "").strip()
    return name.replace(" ", "_")


def _ns_ticker(raw: str) -> str:
    if raw in _TICKER_MAP:
        return _TICKER_MAP[raw]
    if raw.endswith(".NS") or raw.endswith(".BO") or "=" in raw:
        return raw
    return raw + ".NS"


def _fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    caps: dict[str, float] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            mc = getattr(info, "market_cap", None)
            if mc and mc > 0:
                caps[t] = float(mc)
        except Exception:
            pass
    return caps


def _build_basket(sector_cfg: dict, start: str, end: str) -> pd.DataFrame | None:
    raw_tickers: list[str] = sector_cfg.get("constituents", [])
    ns_tickers = [_ns_ticker(str(t)) for t in raw_tickers]

    logger.info("Downloading %d tickers for %s", len(ns_tickers), sector_cfg["name"])
    try:
        raw = yf.download(
            ns_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance download failed for %s: %s", sector_cfg["name"], exc)
        return None

    if raw.empty:
        logger.warning("No data returned for %s", sector_cfg["name"])
        return None

    # Normalize multi/single ticker column layout
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].dropna(axis=1, how="all")
        open_ = raw["Open"].dropna(axis=1, how="all")
        high = raw["High"].dropna(axis=1, how="all")
        low = raw["Low"].dropna(axis=1, how="all")
        volume = raw["Volume"].dropna(axis=1, how="all")
    else:
        close = raw[["Close"]].rename(columns={"Close": ns_tickers[0]})
        open_ = raw[["Open"]].rename(columns={"Open": ns_tickers[0]})
        high = raw[["High"]].rename(columns={"High": ns_tickers[0]})
        low = raw[["Low"]].rename(columns={"Low": ns_tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": ns_tickers[0]})

    valid_tickers = close.columns.tolist()
    if not valid_tickers:
        logger.warning("No valid close prices for %s", sector_cfg["name"])
        return None

    # Market-cap weights (fallback to equal weight)
    caps = _fetch_market_caps(valid_tickers)
    if len(caps) >= 2:
        total = sum(caps.values())
        weights = {t: caps.get(t, 0) / total for t in valid_tickers}
        missing = [t for t in valid_tickers if t not in caps]
        if missing:
            eq = (1 - sum(weights.values())) / len(missing)
            for t in missing:
                weights[t] = max(eq, 0)
    else:
        n = len(valid_tickers)
        weights = {t: 1.0 / n for t in valid_tickers}

    w = np.array([weights[t] for t in valid_tickers])

    basket_close = (close[valid_tickers] * w).sum(axis=1)
    basket_open = (open_.reindex(columns=valid_tickers).fillna(method="ffill") * w).sum(axis=1)
    basket_high = (high.reindex(columns=valid_tickers).fillna(method="ffill") * w).sum(axis=1)
    basket_low = (low.reindex(columns=valid_tickers).fillna(method="ffill") * w).sum(axis=1)
    basket_volume = volume.reindex(columns=valid_tickers).fillna(0).sum(axis=1)

    # Normalize Close to 100 at first valid date
    first_valid = basket_close.first_valid_index()
    if first_valid is not None:
        norm_factor = basket_close[first_valid]
        basket_close = basket_close / norm_factor * 100
        basket_open = basket_open / norm_factor * 100
        basket_high = basket_high / norm_factor * 100
        basket_low = basket_low / norm_factor * 100

    basket = pd.DataFrame({
        "Open": basket_open,
        "High": basket_high,
        "Low": basket_low,
        "Close": basket_close,
        "Volume": basket_volume,
    }).dropna(subset=["Close"])

    basket.index.name = "Date"
    return basket


def run(start: str | None = None, end: str | None = None, force: bool = False) -> pd.DataFrame:
    root = _find_project_root()
    assets_cfg = load_config(root / "config" / "assets.yaml")
    config = load_config(root / "config" / "config.yaml")

    regimes = config.get("regimes", {})
    dates = [
        r.get("full_sample_start") or r.get("test_start")
        for r in regimes.values()
        if isinstance(r, dict)
    ]
    ends = [
        r.get("full_sample_end") or r.get("test_end")
        for r in regimes.values()
        if isinstance(r, dict)
    ]

    effective_start = start or min(d for d in dates if d)
    effective_end = end or max(d for d in ends if d)

    sectors = assets_cfg.get("sectoral", {}).get("sectors", [])
    rows = []

    for sector in sectors:
        name = sector["name"]
        norm_name = _normalize_sector_name(name)
        out_path = root / "data" / "raw" / "equities" / f"{norm_name}.parquet"

        if out_path.exists() and not force:
            logger.info("Skipping %s — parquet already exists (use --force to re-download)", name)
            existing = pd.read_parquet(out_path)
            rows.append({
                "sector": name,
                "n_tickers": len(sector.get("constituents", [])),
                "start_date": str(existing.index.min().date()),
                "end_date": str(existing.index.max().date()),
                "n_rows": len(existing),
                "output_file": str(out_path.relative_to(root)),
            })
            continue

        basket = _build_basket(sector, effective_start, effective_end)
        if basket is None or basket.empty:
            logger.warning("No basket built for %s — skipping", name)
            continue

        save_parquet(basket, out_path)
        logger.info("Saved %s -> %s (%d rows)", name, out_path.name, len(basket))
        rows.append({
            "sector": name,
            "n_tickers": len(sector.get("constituents", [])),
            "start_date": str(basket.index.min().date()),
            "end_date": str(basket.index.max().date()),
            "n_rows": len(basket),
            "output_file": str(out_path.relative_to(root)),
        })

    return pd.DataFrame(rows)
