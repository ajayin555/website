from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.risk_measures import compute_cvar, compute_max_drawdown, compute_sortino


def compute_portfolio_metrics(
    portfolio_returns: pd.Series,
    rf_rate: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    if portfolio_returns.empty:
        return {k: np.nan for k in ["AMR", "ASD", "Sharpe", "Sortino", "CVaR_95", "MaxDrawdown"]}

    rf_aligned = rf_rate.reindex(portfolio_returns.index).fillna(0)
    excess = portfolio_returns - rf_aligned

    amr = float(portfolio_returns.mean() * 252)
    asd = float(portfolio_returns.std() * np.sqrt(252))
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan
    sortino = compute_sortino(portfolio_returns, rf=float(rf_aligned.mean()))
    cvar = compute_cvar(portfolio_returns)
    mdd = compute_max_drawdown(portfolio_returns)

    return {
        "AMR": amr,
        "ASD": asd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "CVaR_95": cvar,
        "MaxDrawdown": mdd,
    }


def compute_all_metrics(
    portfolios: dict[str, pd.Series],
    rf: pd.Series,
    benchmark: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, rets in portfolios.items():
        metrics = compute_portfolio_metrics(rets, rf, benchmark)
        rows.append({"portfolio": name, **metrics})
    return pd.DataFrame(rows).set_index("portfolio")
