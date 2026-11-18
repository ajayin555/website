"""
End-to-end portfolio pipeline entry point:
  1. Load selected-model forecasts (results/models/{regime}/)
  2. Load returns from data/raw/equities/ and data/raw/commodities/
  3. Load macro data (risk-free rate)
  4. Run all portfolio configurations (F-MSR-X, F-BL, MSR-X, EWP, HRP, benchmark)
  5. Compute performance metrics and save results

Usage:
  python -m scripts.run_portfolio_pipeline --regime normal
  python -m scripts.run_portfolio_pipeline --regime high_volatility
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.report import save_metrics_table, plot_cumulative_returns
from src.portfolio.benchmarks import equal_weight_portfolio, hrp_portfolio, market_benchmark
from src.portfolio.optimizer import optimize_fmsr, optimize_fbl
from src.portfolio.rebalancer import run_backtest
from src.portfolio.risk_measures import RISK_MEASURES
from src.utils.io import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def _load_returns(root: Path, subdir: str) -> pd.DataFrame:
    frames = {}
    for f in sorted((root / "data" / "raw" / subdir).glob("*.parquet")):
        try:
            df = pd.read_parquet(f)[["Close"]]
            df.index = pd.DatetimeIndex(df.index)
            frames[f.stem] = df["Close"].pct_change()
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")


def _load_forecasts(root: Path, regime: str) -> pd.DataFrame:
    forecast_dir = root / "results" / "models" / regime
    frames = {}
    for f in sorted(forecast_dir.glob("*_forecasts.parquet")):
        asset = f.stem.replace("_forecasts", "")
        try:
            df = pd.read_parquet(f)
            frames[asset] = df["y_pred"]
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["high_volatility", "normal"], default="normal")
    args = parser.parse_args()

    root = _find_project_root()
    config = load_config(root / "config" / "config.yaml")
    tc_bps = config.get("portfolio", {}).get("transaction_cost_bps", 5)

    # Load data
    eq_returns = _load_returns(root, "equities")
    com_returns = _load_returns(root, "commodities")
    all_returns = pd.concat([eq_returns, com_returns], axis=1) if not com_returns.empty else eq_returns

    forecasts = _load_forecasts(root, args.regime)
    if forecasts.empty:
        logger.error("No forecasts found in results/models/%s/ — run run_forecasting_pipeline first", args.regime)
        return

    # Risk-free rate
    rf_path = root / "data" / "raw" / "macro" / "india_tbill_91d.parquet"
    if rf_path.exists():
        rf = pd.read_parquet(rf_path)["rf_daily"]
    else:
        rf = pd.Series(0.0, index=all_returns.index, name="rf_daily")

    # Align index to test period
    reg_cfg = config["regimes"][args.regime]
    test_start = reg_cfg.get("test_start")
    test_end = reg_cfg.get("full_sample_end")
    test_idx = all_returns.index[(all_returns.index >= test_start) & (all_returns.index <= test_end)]
    forecasts = forecasts.reindex(test_idx)
    rf_test = rf.reindex(test_idx).fillna(0)

    portfolios: dict[str, pd.Series] = {}

    # Benchmark: Nifty 50
    nifty_path = root / "data" / "raw" / "macro" / "nifty50.parquet"
    if nifty_path.exists():
        nifty_df = pd.read_parquet(nifty_path)
        nifty_df.index = pd.DatetimeIndex(nifty_df.index)
        portfolios["Nifty50"] = nifty_df["Close"].pct_change().reindex(test_idx)

    # EWP
    sect_assets = [c for c in eq_returns.columns if c in all_returns.columns]
    if sect_assets:
        ewp_w = equal_weight_portfolio(sect_assets)
        portfolios["EWP"] = all_returns[sect_assets].reindex(test_idx).dot(ewp_w.reindex(sect_assets, fill_value=0))

    # HRP
    if sect_assets:
        hrp_w = hrp_portfolio(all_returns[sect_assets].dropna())
        portfolios["HRP"] = all_returns[hrp_w.index].reindex(test_idx).dot(hrp_w)

    # F-MSR-X for each risk measure
    for rm in RISK_MEASURES:
        def _optimizer(returns, forecasted_returns, rm=rm):
            return optimize_fmsr(returns, forecasted_returns, rm, config)
        try:
            bt = run_backtest(
                returns=all_returns.reindex(test_idx),
                forecasts=forecasts,
                rf_rate=rf_test,
                optimizer_fn=_optimizer,
                config=config,
                transaction_cost_bps=tc_bps,
            )
            portfolios[f"F-MSR-{rm}"] = bt["portfolio_return"]
        except Exception as exc:
            logger.warning("F-MSR-%s failed: %s", rm, exc)

    # F-BL
    def _fbl_optimizer(returns, forecasted_returns):
        return optimize_fbl(returns, forecasted_returns, config)
    try:
        bt_bl = run_backtest(
            returns=all_returns.reindex(test_idx),
            forecasts=forecasts,
            rf_rate=rf_test,
            optimizer_fn=_fbl_optimizer,
            config=config,
            transaction_cost_bps=tc_bps,
        )
        portfolios["F-BL"] = bt_bl["portfolio_return"]
    except Exception as exc:
        logger.warning("F-BL failed: %s", exc)

    # Compute metrics
    benchmark_ret = portfolios.get("Nifty50", pd.Series(0.0, index=test_idx))
    metrics_df = compute_all_metrics(portfolios, rf_test, benchmark_ret)

    out_dir = root / "results"
    save_metrics_table(metrics_df, args.regime, str(out_dir / "tables"))
    plot_cumulative_returns(portfolios, args.regime, str(out_dir / "figures"))
    logger.info("Portfolio pipeline done.")


if __name__ == "__main__":
    main()
