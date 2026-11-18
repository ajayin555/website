"""
Commodity data fetcher using international yfinance proxy tickers (USD-denominated).
Replace with MCX/NCDEX bhavcopy continuous series for production use.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.io import load_config, save_parquet
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

COMMODITY_TICKERS = {
    # Energy
    "Crude_Oil": "CL=F",
    "Natural_Gas": "NG=F",
    "Brent_Crude": "BZ=F",
    "Gasoline": "RB=F",
    # Metals
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Zinc": "ZNC=F",
    # Agriculture
    "Soybean": "ZS=F",
    "Cotton": "CT=F",
    # Mustard Seed and Guar Seed have no reliable yfinance proxy; skip
}

ZINC_FALLBACK = "ZINC.NS"


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def _download_one(name: str, ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty and name == "Zinc":
            logger.info("ZNC=F empty, trying %s", ZINC_FALLBACK)
            df = yf.download(ZINC_FALLBACK, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception as exc:
        logger.warning("Failed to download %s (%s): %s", name, ticker, exc)
        return None


def run(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    root = _find_project_root()
    config = load_config(root / "config" / "config.yaml")
    regimes = config.get("regimes", {})
    dates = [r.get("full_sample_start") for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_start")]
    ends = [r.get("full_sample_end") for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_end")]

    effective_start = start or min(dates)
    effective_end = end or max(ends)

    rows = []
    for name, ticker in COMMODITY_TICKERS.items():
        out_path = root / "data" / "raw" / "commodities" / f"{name}.parquet"
        df = _download_one(name, ticker, effective_start, effective_end)
        if df is None or df.empty:
            logger.warning("No data for %s — skipping", name)
            rows.append({"commodity": name, "ticker": ticker, "status": "failed", "n_rows": 0})
            continue
        save_parquet(df, out_path)
        logger.info("Saved %s -> %s (%d rows)", name, out_path.name, len(df))
        rows.append({
            "commodity": name,
            "ticker": ticker,
            "status": "ok",
            "n_rows": len(df),
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "output_file": str(out_path.relative_to(root)),
        })

    return pd.DataFrame(rows)
