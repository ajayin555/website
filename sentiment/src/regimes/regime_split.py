from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_matrix import build_feature_matrix, split_train_test
from src.utils.io import load_config, save_parquet
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root")


def load_regime_config(config_path: str) -> dict:
    cfg = load_config(config_path)
    return cfg.get("regimes", {})


def split_by_regime(
    df: pd.DataFrame,
    regime: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = config.get("regimes", {}).get(regime, {})
    full_start = reg.get("full_sample_start")
    full_end = reg.get("full_sample_end")
    test_start = reg.get("test_start")

    mask_full = (df.index >= full_start) & (df.index <= full_end) if full_start else pd.Series(True, index=df.index)
    df_full = df[mask_full]
    train_df = df_full[df_full.index < test_start] if test_start else df_full
    test_df = df_full[df_full.index >= test_start] if test_start else pd.DataFrame()
    return train_df, test_df


def save_regime_data(
    asset: str,
    regime: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    root = _find_project_root()
    out_dir = root / "data" / "processed" / f"{regime}_regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_key = asset.replace(" ", "_")
    train_df = X_train.copy()
    train_df["__target__"] = y_train
    test_df = X_test.copy()
    test_df["__target__"] = y_test

    save_parquet(train_df, out_dir / f"{asset_key}_train.parquet")
    save_parquet(test_df, out_dir / f"{asset_key}_test.parquet")
    logger.info("Saved regime data for %s / %s", asset, regime)


def run_regime_split(config: dict) -> pd.DataFrame:
    root = _find_project_root()
    assets_cfg = load_config(root / "config" / "assets.yaml")
    sectors = assets_cfg.get("sectoral", {}).get("sectors", [])
    asset_names = [s["name"] for s in sectors]

    rows = []
    for regime in config.get("regimes", {}).keys():
        reg_cfg = config["regimes"][regime]
        test_start = reg_cfg.get("test_start")
        test_end = reg_cfg.get("full_sample_end")

        for asset in asset_names:
            try:
                X, y = build_feature_matrix(asset, regime, config)
                X_train, y_train, X_test, y_test = split_train_test(X, y, test_start, test_end)
                save_regime_data(asset, regime, X_train, y_train, X_test, y_test)
                rows.append({
                    "asset": asset, "regime": regime,
                    "train_rows": len(X_train), "test_rows": len(X_test),
                    "features": len(X.columns), "status": "ok",
                })
            except Exception as exc:
                logger.error("Failed %s / %s: %s", asset, regime, exc)
                rows.append({"asset": asset, "regime": regime, "status": str(exc)})

    return pd.DataFrame(rows)
