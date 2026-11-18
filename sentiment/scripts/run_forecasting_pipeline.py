"""
End-to-end forecasting pipeline entry point:
  1. Load regime-split, model-ready data (data/processed/)
  2. Train + tune all forecasting models per asset per regime (src/forecasting/)
  3. Evaluate and select the best model; save forecasts for the portfolio stage.

Usage:
  python -m scripts.run_forecasting_pipeline --regime high_volatility
  python -m scripts.run_forecasting_pipeline --regime normal --models LR,RF --trials 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.evaluator import evaluate_all_models, select_best_model, predict
from src.forecasting.trainer import train_all_models
from src.regimes.regime_split import run_regime_split
from src.utils.io import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["high_volatility", "normal"], default="normal")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names (default: all)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Optuna trials per model (default: 10)")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Rebuild regime-split processed data before training")
    args = parser.parse_args()

    root = _find_project_root()
    config = load_config(root / "config" / "config.yaml")
    hparams_config = load_config(root / "config" / "model_hparams.yaml")

    if args.rebuild_features:
        logger.info("Rebuilding regime-split feature matrices …")
        summary = run_regime_split(config)
        print(summary.to_string(index=False))

    regime_dir = root / "data" / "processed" / f"{args.regime}_regime"
    if not regime_dir.exists() or not any(regime_dir.glob("*_train.parquet")):
        logger.info("No processed data found, running regime split …")
        summary = run_regime_split(config)
        print(summary.to_string(index=False))

    train_files = sorted(regime_dir.glob("*_train.parquet"))
    test_files = sorted(regime_dir.glob("*_test.parquet"))
    asset_names = [f.stem.replace("_train", "") for f in train_files]

    out_dir = root / "results" / "models" / args.regime
    out_dir.mkdir(parents=True, exist_ok=True)

    model_filter = set(args.models.split(",")) if args.models else None
    all_rows = []

    for asset, train_f, test_f in zip(asset_names, train_files, test_files):
        logger.info("--- Asset: %s ---", asset)
        try:
            train_df = pd.read_parquet(train_f)
            test_df = pd.read_parquet(test_f)
            y_col = "__target__"
            X_train = train_df.drop(columns=[y_col]).values.astype(np.float32)
            y_train = train_df[y_col].values.astype(np.float32)
            X_test = test_df.drop(columns=[y_col]).values.astype(np.float32)
            y_test = test_df[y_col].values.astype(np.float32)
        except Exception as exc:
            logger.error("Failed to load data for %s: %s", asset, exc)
            continue

        if model_filter:
            from src.forecasting.models import MODEL_REGISTRY
            subset = {k: v for k, v in MODEL_REGISTRY.items() if k in model_filter}
            trained = {}
            for name, factory in subset.items():
                try:
                    from src.forecasting.trainer import tune_and_train
                    trained[name] = tune_and_train(name, X_train, y_train, hparams_config, args.trials)
                except Exception as exc:
                    logger.warning("  %s failed: %s", name, exc)
        else:
            trained = train_all_models(X_train, y_train, hparams_config, n_trials=args.trials)

        eval_df = evaluate_all_models(trained, X_test, y_test)
        best_name = select_best_model(eval_df)
        logger.info("Best model for %s: %s", asset, best_name)

        # Save per-asset forecasts from best model
        best_model = trained[best_name]
        is_dl = _HAS_TORCH and hasattr(best_model, "parameters")
        preds = predict(best_model, X_test, is_dl=is_dl)
        forecast_df = pd.DataFrame({
            "y_true": y_test[-len(preds):],
            "y_pred": preds,
        }, index=pd.read_parquet(test_f).index[-len(preds):])
        forecast_df.to_parquet(out_dir / f"{asset}_forecasts.parquet")

        for _, row in eval_df.iterrows():
            all_rows.append({"asset": asset, **row.to_dict(), "best": row["model"] == best_name})

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "forecast_summary.csv", index=False)
    print("\n=== Forecast Summary ===")
    print(summary.to_string(index=False))
    logger.info("Forecasting pipeline done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
