from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as pta
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False


def _rsi_manual(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _cci_manual(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def _mfi_manual(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfi = 100 - 100 / (1 + pos_sum / neg_sum.replace(0, np.nan))
    return mfi


def _roc_manual(close: pd.Series, period: int = 10) -> pd.Series:
    return close.pct_change(period) * 100


def _williams_r_manual(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def _obv_manual(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _ad_manual(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv * volume).cumsum()


def _macd_manual(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()


def _atr_manual(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    df["Return"] = c.pct_change()

    if _HAS_PANDAS_TA:
        df.ta.sma(length=14, append=True)
        df.ta.ema(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.roc(length=10, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.obv(append=True)
        df.ta.ad(append=True)
        macd_df = df.ta.macd(fast=12, slow=26, signal=9, append=False)
        if macd_df is not None:
            df["MACD"] = macd_df.iloc[:, 0]
        df.ta.atr(length=14, append=True)

        # Normalize column names to our standard
        rename = {}
        for col in df.columns:
            cu = col.upper()
            if cu.startswith("SMA_14"):
                rename[col] = "SMA_14"
            elif cu.startswith("EMA_14"):
                rename[col] = "EMA_14"
            elif cu.startswith("RSI_14"):
                rename[col] = "RSI"
            elif cu.startswith("CCI_20"):
                rename[col] = "CCI"
            elif cu.startswith("MFI_14"):
                rename[col] = "MFI"
            elif cu.startswith("ROC_10"):
                rename[col] = "ROC"
            elif cu.startswith("WILLR_14"):
                rename[col] = "Williams_R"
            elif cu == "OBV":
                rename[col] = "OBV"
            elif cu == "AD":
                rename[col] = "AD"
            elif cu.startswith("ATR_14"):
                rename[col] = "ATR"
        df = df.rename(columns=rename)
    else:
        df["SMA_14"] = c.rolling(14).mean()
        df["EMA_14"] = c.ewm(span=14, adjust=False).mean()
        df["RSI"] = _rsi_manual(c)
        df["CCI"] = _cci_manual(h, l, c)
        df["MFI"] = _mfi_manual(h, l, c, v)
        df["ROC"] = _roc_manual(c)
        df["Williams_R"] = _williams_r_manual(h, l, c)
        df["OBV"] = _obv_manual(c, v)
        df["AD"] = _ad_manual(h, l, c, v)
        df["MACD"] = _macd_manual(c)
        df["ATR"] = _atr_manual(h, l, c)

    expected = ["SMA_14", "EMA_14", "RSI", "CCI", "MFI", "ROC", "Williams_R", "OBV", "AD", "MACD", "ATR"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    return df.dropna(subset=expected)
