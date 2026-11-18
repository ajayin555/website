from __future__ import annotations

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from pypfopt.hierarchical_portfolio import HRPOpt
    _HAS_PYPFOPT = True
except ImportError:
    _HAS_PYPFOPT = False


def equal_weight_portfolio(assets: list[str]) -> pd.Series:
    n = len(assets)
    return pd.Series(1.0 / n, index=assets)


def hrp_portfolio(returns: pd.DataFrame) -> pd.Series:
    if _HAS_PYPFOPT:
        try:
            hrp = HRPOpt(returns=returns)
            hrp.optimize()
            w = pd.Series(hrp.clean_weights())
            w = w[w > 1e-5]
            return w / w.sum()
        except Exception as exc:
            logger.warning("HRP failed (%s), falling back to equal weight", exc)
    return equal_weight_portfolio(returns.columns.tolist())


def market_benchmark(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = weights.reindex(returns.columns).fillna(0)
    aligned_weights = aligned_weights / aligned_weights.sum()
    return returns.dot(aligned_weights)
