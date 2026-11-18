from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_sklearn_model(model, X_train: np.ndarray, y_train: np.ndarray, trial=None):
    model.fit(X_train, y_train)
    return model


def train_dl_model(
    ModelClass,
    X_train: np.ndarray,
    y_train: np.ndarray,
    hparams: dict,
    device: str = "cpu",
):
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for DL models")

    seq_len = hparams.get("seq_len", 20)
    units = hparams.get("units", 100)
    n_layers = hparams.get("n_layers", 2)
    dropout = hparams.get("dropout", 0.2)
    lr = hparams.get("learning_rate", 1e-3)
    optimizer_name = hparams.get("optimizer", "adam")
    loss_name = hparams.get("loss", "huber")
    batch_size = hparams.get("batch_size", 64)
    max_epochs = hparams.get("max_epochs", 100)
    patience = hparams.get("early_stopping_patience", 20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Validation split (last 20%)
    val_split = max(1, int(len(X_scaled) * 0.2))
    X_tr, X_val = X_scaled[:-val_split], X_scaled[-val_split:]
    y_tr, y_val = y_train[:-val_split], y_train[-val_split:]

    Xtr_seq, ytr_seq = create_sequences(X_tr, y_tr, seq_len)
    Xval_seq, yval_seq = create_sequences(X_val, y_val, seq_len)

    if len(Xtr_seq) == 0:
        raise ValueError("Not enough training data for sequences")

    input_size = Xtr_seq.shape[2]
    model = ModelClass(input_size=input_size, units=units, n_layers=n_layers, dropout=dropout).to(device)

    opt_map = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
    optimizer = opt_map.get(optimizer_name, torch.optim.Adam)(model.parameters(), lr=lr)
    criterion = nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr_seq), torch.tensor(ytr_seq).unsqueeze(1)),
        batch_size=batch_size, shuffle=False,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(Xval_seq).to(device)
            yv = torch.tensor(yval_seq).unsqueeze(1).to(device)
            val_loss = criterion(model(xv), yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug("Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    model._scaler = scaler  # attach scaler for inference
    return model


def tune_and_train(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    hparams_config: dict,
    n_trials: int = 50,
):
    from src.forecasting.models import MODEL_REGISTRY, get_lr_model, get_rf_model, get_svr_model

    if model_name == "LR":
        return train_sklearn_model(get_lr_model(), X_train, y_train)

    if not _HAS_OPTUNA or n_trials <= 1:
        if model_name == "RF":
            return train_sklearn_model(get_rf_model(), X_train, y_train)
        if model_name == "SVR":
            return train_sklearn_model(get_svr_model(), X_train, y_train)
        if _HAS_TORCH and model_name in MODEL_REGISTRY:
            return train_dl_model(MODEL_REGISTRY[model_name], X_train, y_train, {})
        raise ValueError(f"Unknown model: {model_name}")

    from sklearn.model_selection import cross_val_score, TimeSeriesSplit

    if model_name == "RF":
        def objective(trial):
            m = get_rf_model(trial)
            scores = cross_val_score(m, X_train, y_train, cv=TimeSeriesSplit(5),
                                     scoring="neg_mean_absolute_error")
            return -scores.mean()
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return train_sklearn_model(get_rf_model(study.best_trial), X_train, y_train)

    if model_name == "SVR":
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_train)
        def objective(trial):
            m = get_svr_model(trial)
            scores = cross_val_score(m, X_sc, y_train, cv=TimeSeriesSplit(5),
                                     scoring="neg_mean_absolute_error")
            return -scores.mean()
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        m = get_svr_model(study.best_trial)
        m.fit(X_sc, y_train)
        m._scaler = scaler
        return m

    if _HAS_TORCH and model_name in MODEL_REGISTRY:
        hp_ranges = hparams_config.get("recurrent_models", {}) if model_name not in ("GRU-LSTM", "BiGRU-GRU") \
            else hparams_config.get("hybrid_models", {})
        def objective(trial):
            hparams = {
                "units": trial.suggest_categorical("units", hp_ranges.get("units", [100])),
                "n_layers": trial.suggest_int("n_layers", *hp_ranges.get("n_layers", [1, 2])),
                "dropout": trial.suggest_categorical("dropout", hp_ranges.get("dropout", [0.2])),
                "optimizer": trial.suggest_categorical("optimizer", hp_ranges.get("optimizer", ["adam"])),
                "loss": trial.suggest_categorical("loss", hp_ranges.get("loss", ["huber"])),
                "learning_rate": trial.suggest_float("learning_rate", *hp_ranges.get("learning_rate", [1e-4, 1e-2]), log=True),
                "max_epochs": 30,
            }
            try:
                m = train_dl_model(MODEL_REGISTRY[model_name], X_train, y_train, hparams)
                seq_len = hparams.get("seq_len", 20)
                scaler = m._scaler
                X_sc = scaler.transform(X_train)
                Xs, ys = create_sequences(X_sc, y_train, seq_len)
                if len(Xs) == 0:
                    return float("inf")
                m.eval()
                with torch.no_grad():
                    preds = m(torch.tensor(Xs)).squeeze().numpy()
                return float(np.mean(np.abs(preds - ys)))
            except Exception:
                return float("inf")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=min(n_trials, 10), show_progress_bar=False)
        best = study.best_params
        best["max_epochs"] = 100
        return train_dl_model(MODEL_REGISTRY[model_name], X_train, y_train, best)

    raise ValueError(f"Unknown model: {model_name}")


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hparams_config: dict,
    n_trials: int = 10,
) -> dict[str, Any]:
    from src.forecasting.models import MODEL_REGISTRY
    models: dict[str, Any] = {}
    for name in MODEL_REGISTRY:
        logger.info("Training %s ...", name)
        try:
            models[name] = tune_and_train(name, X_train, y_train, hparams_config, n_trials)
            logger.info("  %s done", name)
        except Exception as exc:
            logger.warning("  %s failed: %s", name, exc)
    return models
