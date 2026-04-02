"""
Shared metric helpers, batch prediction utilities, and series-key utilities.

These functions are imported at module level by parallel.py so that worker
processes can use them without needing to redefine anything locally.
"""
import numpy as np


def series_key_str(grp_key: tuple) -> str:
    """Return a filesystem-safe string key from a GROUP_COLS value tuple."""
    return "__".join(str(v).replace("/", "-").replace(" ", "_") for v in grp_key)


def wma(series, window: int) -> float:
    """Weighted moving average with linearly increasing weights."""
    s = series.tail(window).values
    if len(s) == 0:
        return 0.0
    w = np.arange(1, len(s) + 1, dtype=float)
    return float(np.dot(w, s) / w.sum())


def wmape_safe(y_true, y_pred, cap: float = 5.0) -> float:
    """Weighted MAPE clipped at `cap`, robust against near-zero actuals."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return np.nan
    threshold = np.abs(yt).max() * 0.001 if np.abs(yt).max() > 0 else 1.0
    mask = np.abs(yt) > threshold
    if mask.sum() < 3:
        return np.nan
    yt_m, yp_m = yt[mask], yp[mask]
    weights = np.abs(yt_m)
    pct_errs = np.clip(np.abs(yt_m - yp_m) / weights, 0, cap)
    return float(np.average(pct_errs, weights=weights) * 100)


def medape_safe(y_true, y_pred) -> float:
    """Median APE, robust against near-zero actuals."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return np.nan
    max_val = np.abs(yt).max() if len(yt) > 0 else 0
    mask = (
        np.abs(yt) > max_val * 0.001
        if max_val > 0
        else np.ones(len(yt), dtype=bool)
    )
    if mask.sum() < 3:
        return np.nan
    return float(np.median(np.abs(yt[mask] - yp[mask]) / np.abs(yt[mask])) * 100)


def batch_predict(model, X, batch_size: int = 512) -> np.ndarray:
    """Run model.predict in mini-batches to limit peak memory."""
    if len(X) == 0:
        return np.array([])
    out = np.empty(len(X), dtype=float)
    for s in range(0, len(X), batch_size):
        e = min(s + batch_size, len(X))
        out[s:e] = model.predict(X[s:e])
    return out


def batch_predict_proba(model, X, batch_size: int = 512) -> np.ndarray:
    """Run model.predict_proba in mini-batches, returning P(class=1)."""
    if len(X) == 0:
        return np.array([])
    out = np.empty(len(X), dtype=float)
    for s in range(0, len(X), batch_size):
        e = min(s + batch_size, len(X))
        proba = model.predict_proba(X[s:e])
        if proba.ndim == 1:
            out[s:e] = proba
        elif proba.shape[1] == 1:
            classes = getattr(model, "classes_", np.array([1]))
            out[s:e] = 1.0 if classes[0] == 1 else 0.0
        else:
            out[s:e] = proba[:, 1]
    return out
