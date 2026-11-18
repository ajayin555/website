"""
End-to-end data pipeline entry point:
  1. Fetch equities — market-cap-weighted sector baskets (src/data_collection/fetch_equity_data.py)
  2. Fetch commodity futures via yfinance proxies (src/data_collection/fetch_commodity_data.py)
  3. Fetch macro data: India VIX, Nifty 50, risk-free rate (src/data_collection/fetch_macro_data.py)
  4. Build technical + sentiment features (src/features/) [optional stage]
  5. Synchronize into a single panel and split into regimes (src/regimes/) [optional stage]

Usage:
  python -m scripts.run_data_pipeline
  python -m scripts.run_data_pipeline --stage equities
  python -m scripts.run_data_pipeline --start 2020-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse

from src.data_collection import fetch_equity_data, fetch_commodity_data, fetch_macro_data, fetch_sentiment_data
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

STAGES = ["equities", "commodities", "macro", "sentiment"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="Run a single stage only (default: all)")
    parser.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if parquet files exist")
    parser.add_argument("--live-scrape", action="store_true",
                        help="Attempt live ET RSS scraping for recent dates (last 30 days); "
                             "historical dates always use synthetic data")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else STAGES

    if "equities" in stages:
        logger.info("=== Stage: equities ===")
        summary = fetch_equity_data.run(start=args.start, end=args.end, force=args.force)
        print(summary.to_string(index=False))

    if "commodities" in stages:
        logger.info("=== Stage: commodities ===")
        summary = fetch_commodity_data.run(start=args.start, end=args.end)
        print(summary.to_string(index=False))

    if "macro" in stages:
        logger.info("=== Stage: macro ===")
        summary = fetch_macro_data.run(start=args.start, end=args.end)
        print(summary.to_string(index=False))

    if "sentiment" in stages:
        logger.info("=== Stage: sentiment ===")
        result = fetch_sentiment_data.run(start=args.start, end=args.end, live_scrape=args.live_scrape)
        logger.info("Sentiment built for %d assets", len(result))

    logger.info("Data pipeline finished for stage(s): %s", stages)


if __name__ == "__main__":
    main()
