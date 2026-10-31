"""Point-forecast error metrics and grouped scoring.

* MAPE  = 100 / n * sum(|y - f| / |y|)
* SMAPE = 100 / n * sum(2 |y - f| / (|y| + |f|))      (range 0..200)
* MAE   = 1 / n * sum(|y - f|)
* RMSE  = sqrt(1 / n * sum((y - f)^2))

MAPE requires non-zero actuals; the cost indices are strictly positive (A12).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ta_forecasting.config import Config

ACTUAL = "actual"
PREDICTION = "prediction"
PERCENT = 100.0

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _arrays(actual: Sequence[float], forecast: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if y.shape != f.shape or y.size == 0:
        raise ValueError("actual and forecast must be non-empty and of equal shape.")
    if not (np.isfinite(y).all() and np.isfinite(f).all()):
        raise ValueError("actual and forecast must be finite.")
    return y, f


def mape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Mean absolute percentage error in percent."""
    y, f = _arrays(actual, forecast)
    if (y == 0).any():
        raise ValueError("MAPE is undefined for zero actuals.")
    return float(PERCENT * np.mean(np.abs(y - f) / np.abs(y)))


def smape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Symmetric MAPE in percent (0..200). Pairs with y = f = 0 count as zero error."""
    y, f = _arrays(actual, forecast)
    denom = np.abs(y) + np.abs(f)
    ratio = np.divide(2.0 * np.abs(y - f), denom, out=np.zeros_like(y), where=denom > 0)
    return float(PERCENT * np.mean(ratio))


def mae(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Mean absolute error (index units)."""
    y, f = _arrays(actual, forecast)
    return float(np.mean(np.abs(y - f)))


def rmse(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Root mean squared error (index units)."""
    y, f = _arrays(actual, forecast)
    return float(np.sqrt(np.mean((y - f) ** 2)))


METRICS: dict[str, MetricFn] = {"mape": mape, "smape": smape, "mae": mae, "rmse": rmse}


def metric_names(cfg: Config) -> list[str]:
    """Primary metric first, then the secondary ones, as configured."""
    names = [cfg.metrics.primary, *cfg.metrics.secondary]
    unknown = sorted(set(names) - set(METRICS))
    if unknown:
        raise ValueError(f"Unknown metrics in config: {unknown}; available: {sorted(METRICS)}")
    return list(dict.fromkeys(names))


def score(frame: pd.DataFrame, by: list[str], names: list[str]) -> pd.DataFrame:
    """Compute ``names`` per group of ``by`` on pooled rows; adds ``n`` (row count)."""
    rows = []
    for key, group in frame.groupby(by, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, keys, strict=True))
        y, f = group[ACTUAL].to_numpy(), group[PREDICTION].to_numpy()
        row.update({name: METRICS[name](y, f) for name in names})
        row["n"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)
