from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecasting.trainer import create_sequences
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

SEQ_LEN = 20


def predict(
    model: Any,
    X_test: np.ndarray,
    seq_len: int = SEQ_LEN,
    is_dl: bool = False,
) -> np.ndarray:
    if is_dl:
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for DL inference")
        scaler = getattr(model, "_scaler", None)
        X_sc = scaler.transform(X_test) if scaler is not None else X_test
        Xs, _ = create_sequences(X_sc, np.zeros(len(X_sc)), seq_len)
        if len(Xs) == 0:
            return np.array([])
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(Xs)).squeeze().numpy()
        return preds
    else:
        scaler = getattr(model, "_scaler", None)
        X_sc = scaler.transform(X_test) if scaler is not None else X_test
        return model.predict(X_sc)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "DA": np.nan}
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[-n:], y_pred[-n:]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    da = float(np.mean(np.sign(yt) == np.sign(yp)))
    return {"MAE": mae, "RMSE": rmse, "DA": da}


def evaluate_all_models(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        is_dl = _HAS_TORCH and hasattr(model, "parameters")
        try:
            preds = predict(model, X_test, seq_len=SEQ_LEN, is_dl=is_dl)
            metrics = compute_metrics(y_test, preds)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", name, exc)
            metrics = {"MAE": np.nan, "RMSE": np.nan, "DA": np.nan}
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows)


def select_best_model(eval_df: pd.DataFrame, criterion: str = "MAE") -> str:
    df = eval_df.dropna(subset=[criterion])
    if df.empty:
        return eval_df["model"].iloc[0]
    if criterion == "DA":
        return df.loc[df[criterion].idxmax(), "model"]
    return df.loc[df[criterion].idxmin(), "model"]
