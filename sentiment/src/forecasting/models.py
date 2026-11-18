from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ── ML models ────────────────────────────────────────────────────────────────

def get_lr_model():
    return LinearRegression()


def get_rf_model(trial=None):
    if trial is not None:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    else:
        params = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    return RandomForestRegressor(**params, random_state=42, n_jobs=-1)


def get_svr_model(trial=None):
    if trial is not None:
        params = {
            "C": trial.suggest_float("C", 0.1, 1000, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.001, 1.0),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        }
    else:
        params = {"C": 10.0, "gamma": "scale", "kernel": "rbf"}
    return SVR(**params)


# ── DL models (PyTorch) ───────────────────────────────────────────────────────

if _HAS_TORCH:
    class GRUModel(nn.Module):
        def __init__(self, input_size: int, units: int = 100, n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(input_size, units, num_layers=n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(units, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.drop(out[:, -1, :]))

    class LSTMModel(nn.Module):
        def __init__(self, input_size: int, units: int = 100, n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, units, num_layers=n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(units, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    class BiGRUModel(nn.Module):
        def __init__(self, input_size: int, units: int = 100, n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(input_size, units, num_layers=n_layers, batch_first=True,
                              bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(units * 2, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.drop(out[:, -1, :]))

    class BiLSTMModel(nn.Module):
        def __init__(self, input_size: int, units: int = 100, n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, units, num_layers=n_layers, batch_first=True,
                                bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(units * 2, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    class GRU_LSTM_Model(nn.Module):
        def __init__(self, input_size: int, units: int = 100, n_layers: int = 1, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(input_size, units, num_layers=n_layers, batch_first=True)
            self.lstm = nn.LSTM(units, units, num_layers=n_layers, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(units, 1)

        def forward(self, x):
            gru_out, _ = self.gru(x)
            lstm_out, _ = self.lstm(gru_out)
            return self.fc(self.drop(lstm_out[:, -1, :]))

    class BiGRU_GRU_Model(nn.Module):
        def __init__(self, input_size: int, units: int = 100, n_layers: int = 1, dropout: float = 0.2):
            super().__init__()
            self.bigru = nn.GRU(input_size, units, num_layers=n_layers, batch_first=True, bidirectional=True)
            self.gru = nn.GRU(units * 2, units, num_layers=n_layers, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(units, 1)

        def forward(self, x):
            bigru_out, _ = self.bigru(x)
            gru_out, _ = self.gru(bigru_out)
            return self.fc(self.drop(gru_out[:, -1, :]))

    MODEL_REGISTRY = {
        "LR": get_lr_model,
        "RF": get_rf_model,
        "SVR": get_svr_model,
        "GRU": GRUModel,
        "LSTM": LSTMModel,
        "BiGRU": BiGRUModel,
        "BiLSTM": BiLSTMModel,
        "GRU-LSTM": GRU_LSTM_Model,
        "BiGRU-GRU": BiGRU_GRU_Model,
    }

else:
    # Torch unavailable: only sklearn models
    MODEL_REGISTRY = {
        "LR": get_lr_model,
        "RF": get_rf_model,
        "SVR": get_svr_model,
    }
