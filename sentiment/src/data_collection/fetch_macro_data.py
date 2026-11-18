from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.io import load_config, save_parquet
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ^IRX is the 13-week US T-bill annualised yield (%).
# Used as a proxy; replace with RBI 91-day T-bill auction data for production.
_RF_TICKER = "^IRX"
_VIX_TICKER = "^INDIAVIX"
_NIFTY_TICKER = "^NSEI"


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def _download(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        return df if not df.empty else None
    except Exception as exc:
        logger.warning("Download failed for %s: %s", ticker, exc)
        return None


def run(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    root = _find_project_root()
    config = load_config(root / "config" / "config.yaml")
    regimes = config.get("regimes", {})
    starts = [r["full_sample_start"] for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_start")]
    ends = [r["full_sample_end"] for r in regimes.values() if isinstance(r, dict) and r.get("full_sample_end")]

    effective_start = start or min(starts)
    effective_end = end or max(ends)

    rows = []

    # India VIX
    vix = _download(_VIX_TICKER, effective_start, effective_end)
    if vix is not None:
        vix_path = root / "data" / "raw" / "macro" / "india_vix.parquet"
        save_parquet(vix[["Close"]].rename(columns={"Close": "india_vix"}), vix_path)
        rows.append({"series": "india_vix", "n_rows": len(vix), "status": "ok"})
    else:
        rows.append({"series": "india_vix", "n_rows": 0, "status": "failed"})

    # Nifty 50
    nifty = _download(_NIFTY_TICKER, effective_start, effective_end)
    if nifty is not None:
        nifty_path = root / "data" / "raw" / "macro" / "nifty50.parquet"
        save_parquet(nifty[["Open", "High", "Low", "Close", "Volume"]], nifty_path)
        rows.append({"series": "nifty50", "n_rows": len(nifty), "status": "ok"})
    else:
        rows.append({"series": "nifty50", "n_rows": 0, "status": "failed"})

    # Risk-free rate (US T-bill proxy -> daily)
    rf = _download(_RF_TICKER, effective_start, effective_end)
    if rf is not None:
        rf_daily = rf[["Close"]].rename(columns={"Close": "rf_annual_pct"})
        rf_daily["rf_daily"] = rf_daily["rf_annual_pct"] / 100 / 252
        rf_path = root / "data" / "raw" / "macro" / "india_tbill_91d.parquet"
        save_parquet(rf_daily[["rf_daily"]], rf_path)
        rows.append({"series": "rf_rate", "n_rows": len(rf_daily), "status": "ok"})
    else:
        rows.append({"series": "rf_rate", "n_rows": 0, "status": "failed"})

    return pd.DataFrame(rows)
