from __future__ import annotations
import numpy as np

def seasonal_naive(y: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Return the last 'season' values repeated to forecast 'horizon' steps.
    y shape: (T,)
    """
    if len(y) < season:
        raise ValueError("Series shorter than one season.")
    pattern = y[-season:]
    reps = int(np.ceil(horizon / season))
    forecast = np.tile(pattern, reps)[:horizon]
    return forecast

def last_value_naive(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) == 0:
        raise ValueError("Empty series.")
    return np.full(horizon, y[-1], dtype=float)