from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict

def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[time_col])
    df = df.copy()
    # Hour-of-day cyclical
    hour = dt.dt.hour.values
    df['hour_sin'] = np.sin(2*np.pi*hour/24.0)
    df['hour_cos'] = np.cos(2*np.pi*hour/24.0)
    # Day-of-week cyclical
    dow = dt.dt.dayofweek.values
    df['dow_sin'] = np.sin(2*np.pi*dow/7.0)
    df['dow_cos'] = np.cos(2*np.pi*dow/7.0)
    return df

def make_windows(X: np.ndarray, y: np.ndarray, lookback: int, horizon: int):
    """Return (X_windows, y_windows)
    X: (T, F), y: (T,)
    X_windows: (N, lookback, F), y_windows: (N, horizon)
    """
    Xs, Ys = [], []
    T = len(y)
    for t in range(lookback, T - horizon + 1):
        Xs.append(X[t - lookback:t])
        Ys.append(y[t:t + horizon])
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

def time_series_split(T: int, val_ratio: float, test_ratio: float):
    test_size = int(T * test_ratio)
    val_size = int(T * val_ratio)
    train_end = T - (val_size + test_size)
    val_end = T - test_size
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, T)

def prepare_data(csv_path: str, time_col: str, target_col: str, feature_cols: List[str],
                 lookback: int, horizon: int, val_ratio: float, test_ratio: float):
    df = pd.read_csv(csv_path)
    if time_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"CSV must contain '{time_col}' and '{target_col}'.")

    df = df.sort_values(time_col).reset_index(drop=True)
    df = add_time_features(df, time_col)

    # Build feature matrix
    extra = feature_cols or []
    used_cols = extra + ['hour_sin','hour_cos','dow_sin','dow_cos']
    X = df[used_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Split by time
    train_idx, val_idx, test_idx = time_series_split(len(df), val_ratio, test_ratio)

    # Fit scalers on TRAIN only
    x_scaler = StandardScaler().fit(X[train_idx])
    y_scaler = StandardScaler().fit(y[train_idx].reshape(-1,1))

    Xs = x_scaler.transform(X)
    ys = y_scaler.transform(y.reshape(-1,1)).ravel()

    # Windowing on the FULL series, then split by indices of the *end* of windows
    Xw, Yw = make_windows(Xs, ys, lookback, horizon)

    # Map each window to its ending index (end at t)
    end_indices = np.arange(lookback, lookback + len(Xw))

    def idx_to_mask(idx_slice: slice):
        return (end_indices > idx_slice.start) & (end_indices <= idx_slice.stop)

    train_mask = idx_to_mask(train_idx)
    val_mask = idx_to_mask(val_idx)
    test_mask = idx_to_mask(test_idx)

    data = {
        'X_train': Xw[train_mask], 'Y_train': Yw[train_mask],
        'X_val':   Xw[val_mask],   'Y_val':   Yw[val_mask],
        'X_test':  Xw[test_mask],  'Y_test':  Yw[test_mask],
        'x_scaler_mean': x_scaler.mean_.tolist(),
        'x_scaler_scale': x_scaler.scale_.tolist(),
        'y_scaler_mean': float(y_scaler.mean_[0]),
        'y_scaler_scale': float(y_scaler.scale_[0]),
        'used_features': used_cols,
    }
    return data