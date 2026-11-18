from __future__ import annotations

import numpy as np
import pandas as pd

RISK_MEASURES = ["MV", "MSV", "MAD", "FLPM", "CVaR", "CDaR", "EVaR"]

RM_MAP = {
    "Variance": "MV",
    "MSV": "MSV",
    "MAD": "MAD",
    "FLPM": "FLPM",
    "CVaR": "CVaR",
    "CDaR": "CDaR",
    "EVaR": "EVaR",
}


def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    sorted_r = returns.sort_values()
    cutoff = int(np.floor((1 - alpha) * len(sorted_r)))
    tail = sorted_r.iloc[: max(cutoff, 1)]
    return float(-tail.mean())


def compute_max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min())


def compute_sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    return float(excess.mean() / downside.std() * np.sqrt(252))
