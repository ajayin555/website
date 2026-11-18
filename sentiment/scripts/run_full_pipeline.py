"""
Runs the full pipeline end-to-end for both regimes:
  run_data_pipeline -> run_regime_split -> run_forecasting_pipeline -> run_portfolio_pipeline

Usage:
  python -m scripts.run_full_pipeline
  python -m scripts.run_full_pipeline --trials 5 --models LR,RF
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.warning("Command exited with code %d", result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--models", default=None)
    parser.add_argument("--force", action="store_true", help="Force re-download data")
    args = parser.parse_args()

    py = sys.executable
    root = Path(__file__).resolve().parent.parent

    # Stage 1: data
    data_cmd = [py, "-m", "scripts.run_data_pipeline"]
    if args.force:
        data_cmd.append("--force")
    _run(data_cmd)

    # Stages 2+3: forecasting + portfolio per regime
    for regime in ["high_volatility", "normal"]:
        forecast_cmd = [py, "-m", "scripts.run_forecasting_pipeline",
                        "--regime", regime, "--trials", str(args.trials)]
        if args.models:
            forecast_cmd += ["--models", args.models]
        _run(forecast_cmd)

        portfolio_cmd = [py, "-m", "scripts.run_portfolio_pipeline", "--regime", regime]
        _run(portfolio_cmd)

    logger.info("Full pipeline complete.")


if __name__ == "__main__":
    main()
