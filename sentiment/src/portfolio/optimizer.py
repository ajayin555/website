from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import riskfolio as rp
    _HAS_RISKFOLIO = True
except ImportError:
    _HAS_RISKFOLIO = False

try:
    from pypfopt import BlackLittermanModel, risk_models, expected_returns as _ep
    from pypfopt.efficient_frontier import EfficientFrontier
    _HAS_PYPFOPT = True
except ImportError:
    _HAS_PYPFOPT = False


def _top_n_assets(forecasted_returns: pd.Series, n: int = 6) -> list[str]:
    sorted_ = forecasted_returns.sort_values(ascending=False)
    top = sorted_.head(n)
    return top[top > 0].index.tolist() if top.max() > 0 else []


def _equal_weight(assets: list[str]) -> pd.Series:
    n = len(assets)
    return pd.Series(1.0 / n, index=assets)


def optimize_fmsr(
    returns: pd.DataFrame,
    forecasted_returns: pd.Series,
    risk_measure: str,
    config: dict,
) -> pd.Series:
    n_assets = config.get("portfolio", {}).get("n_assets_per_portfolio", 6)
    max_weight = config.get("portfolio", {}).get("max_weight_per_asset", 0.30)
    suspend = config.get("portfolio", {}).get("suspend_if_all_negative_forecast", True)

    top_assets = _top_n_assets(forecasted_returns, n_assets)
    if not top_assets:
        if suspend:
            return pd.Series(dtype=float)
        top_assets = forecasted_returns.nlargest(n_assets).index.tolist()

    ret_sub = returns[top_assets].dropna()
    if len(ret_sub) < 20:
        return _equal_weight(top_assets)

    if _HAS_RISKFOLIO:
        try:
            port = rp.Portfolio(returns=ret_sub)
            port.assets_stats(method_mu="hist", method_cov="hist")
            rm_code = risk_measure if risk_measure in ("MV", "MSV", "MAD", "FLPM", "CVaR", "CDaR", "EVaR") else "MV"
            w = port.optimization(model="Classic", rm=rm_code, obj="Sharpe", rf=0.0, l=0)
            if w is not None and not w.empty:
                w = w.squeeze().clip(lower=0)
                w = w / w.sum()
                w = w.clip(upper=max_weight)
                w = w / w.sum()
                return w
        except Exception as exc:
            logger.warning("riskfolio optimization failed (%s), falling back to equal weight", exc)

    if _HAS_PYPFOPT:
        try:
            mu = _ep.mean_historical_return(ret_sub)
            S = risk_models.sample_cov(ret_sub)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
            ef.max_sharpe()
            cleaned = ef.clean_weights()
            w = pd.Series(cleaned).dropna()
            w = w[w > 1e-5]
            if w.sum() > 0:
                return w / w.sum()
        except Exception as exc:
            logger.warning("pypfopt optimization failed (%s), falling back", exc)

    return _equal_weight(top_assets)


def optimize_fbl(
    returns: pd.DataFrame,
    forecasted_returns: pd.Series,
    config: dict,
) -> pd.Series:
    if not _HAS_PYPFOPT:
        return _equal_weight(forecasted_returns.nlargest(6).index.tolist())

    bl_cfg = config.get("black_litterman", {})
    tau = bl_cfg.get("tau", 0.05)
    delta = bl_cfg.get("risk_aversion_delta", 1.0)
    view_conf = bl_cfg.get("view_confidence", 0.99)
    max_weight = config.get("portfolio", {}).get("max_weight_per_asset", 0.30)

    assets = forecasted_returns.index.intersection(returns.columns).tolist()
    if not assets:
        return pd.Series(dtype=float)

    ret_sub = returns[assets].dropna()
    if len(ret_sub) < 20:
        return _equal_weight(assets)

    try:
        S = risk_models.sample_cov(ret_sub)
        # Equal market weights as prior
        mkt_weights = pd.Series(1.0 / len(assets), index=assets)
        pi = delta * S.dot(mkt_weights)

        # Views: each asset view = forecasted return
        views = forecasted_returns[assets]
        # Omega: diagonal confidence matrix
        omega = np.diag([(1 - view_conf) / view_conf * S.loc[a, a] for a in assets])

        bl = BlackLittermanModel(S, pi=pi, absolute_views=views.to_dict(),
                                  omega=omega, tau=tau)
        ret_bl = bl.bl_returns()
        ef = EfficientFrontier(ret_bl, S, weight_bounds=(0, max_weight))
        ef.max_sharpe()
        cleaned = ef.clean_weights()
        w = pd.Series(cleaned).dropna()
        w = w[w > 1e-5]
        if w.sum() > 0:
            return w / w.sum()
    except Exception as exc:
        logger.warning("Black-Litterman failed (%s), fallback to equal weight", exc)

    return _equal_weight(assets[:6])
