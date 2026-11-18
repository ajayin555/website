"""
End-to-end data pipeline entry point:
  1. Fetch equities [IMPLEMENTED], commodities, sentiment, macro data (src/data_collection/)
  2. Build technical + sentiment features (src/features/) [NOT YET IMPLEMENTED]
  3. Synchronize into a single panel and split into regimes (src/regimes/) [NOT YET IMPLEMENTED]

Usage:
  python scripts/run_data_pipeline.py            # runs everything currently implemented (equities only)
  python scripts/run_data_pipeline.py --stage equities
"""

from __future__ import annotations

import argparse

from src.data_collection import fetch_equity_data
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

STAGES = ["equities"]  # "commodities", "sentiment", "macro" land here once implemented


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run a single stage only (default: all implemented stages)")
    parser.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else STAGES

    if "equities" in stages:
        logger.info("=== Stage: equities ===")
        summary = fetch_equity_data.run(start=args.start, end=args.end)
        print(summary.to_string(index=False))

    logger.info(
        "Data pipeline finished for stage(s) %s. Commodities, sentiment, and macro "
        "fetchers, and the feature/regime-split stages, are not yet implemented "
        "(see src/data_collection/, src/features/, src/regimes/).",
        stages,
    )


if __name__ == "__main__":
    main()
