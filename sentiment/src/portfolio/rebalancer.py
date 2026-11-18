from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_backtest(
    returns: pd.DataFrame,
    forecasts: pd.DataFrame,
    rf_rate: pd.Series,
    optimizer_fn: Callable,
    config: dict,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    tc = transaction_cost_bps / 10_000
    dates = returns.index.intersection(forecasts.index)
    dates = sorted(dates)

    prev_weights = pd.Series(dtype=float)
    rows = []

    for date in dates:
        daily_forecasts = forecasts.loc[date]
        try:
            new_weights = optimizer_fn(
                returns=returns.loc[:date].iloc[:-1],
                forecasted_returns=daily_forecasts,
            )
        except Exception as exc:
            logger.debug("Optimizer failed on %s: %s", date, exc)
            new_weights = prev_weights.copy()

        if new_weights.empty:
            new_weights = prev_weights.copy()

        if new_weights.empty:
            rows.append({"date": date, "portfolio_return": 0.0, "weights": {}})
            continue

        # Align with available assets
        available = [a for a in new_weights.index if a in returns.columns]
        new_weights = new_weights[available]
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()

        # Transaction cost
        if not prev_weights.empty:
            aligned_prev = prev_weights.reindex(new_weights.index).fillna(0)
            turnover = (new_weights - aligned_prev).abs().sum()
            tc_cost = tc * float(turnover)
        else:
            tc_cost = 0.0

        # Portfolio return
        if date in returns.index:
            asset_returns = returns.loc[date, new_weights.index]
            port_ret = float(new_weights.dot(asset_returns.fillna(0))) - tc_cost
        else:
            port_ret = 0.0

        rows.append({
            "date": date,
            "portfolio_return": port_ret,
            "weights": new_weights.to_dict(),
        })
        prev_weights = new_weights

    return pd.DataFrame(rows).set_index("date")
