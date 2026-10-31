"""Empirical prediction intervals, identical in construction for every model.

For model ``m`` and step ``h`` the ratios ``actual / prediction`` of the long backtest are
pooled over series and origins. The central ``interval_level`` interval of a new forecast
``f`` is ``[f * q_lo, f * q_hi]`` with ``q_lo, q_hi`` the ``(1 ± level) / 2`` quantiles of
those ratios. A multiplicative band suits index levels whose errors scale with the level.

Coverage check: quantiles from origins in ``calibration_split`` only, coverage measured on
origins in ``coverage_split`` (out of sample for ETS, ARIMA and naive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN
from ta_forecasting.evaluation.backtest import MODEL
from ta_forecasting.evaluation.metrics import ACTUAL
from ta_forecasting.models.base import PREDICTION, STEP

RATIO = "ratio"
Q_LO = "ratio_lo"
Q_HI = "ratio_hi"
N = "n"
LOWER = "lower"
UPPER = "upper"
COVERAGE = "coverage"


def _bounds(cfg: Config) -> tuple[float, float]:
    level = cfg.forecast.interval_level
    return (1 - level) / 2, (1 + level) / 2


def ratio_quantiles(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """``[model, step, ratio_lo, ratio_hi, n]`` from scored backtest forecasts."""
    lo, hi = _bounds(cfg)
    ratios = pred.assign(**{RATIO: pred[ACTUAL] / pred[PREDICTION]})
    grouped = ratios.groupby([MODEL, STEP])[RATIO]
    out = pd.DataFrame({
        Q_LO: grouped.quantile(lo),
        Q_HI: grouped.quantile(hi),
        N: grouped.size(),
    }).reset_index()
    if not np.isfinite(out[[Q_LO, Q_HI]].to_numpy()).all():
        raise ValueError("Interval ratios are not finite (a prediction of 0?).")
    return out


def apply_intervals(forecast: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    """Add ``lower`` / ``upper`` to forecasts ``[model, step, prediction, ...]``."""
    merged = forecast.merge(quantiles[[MODEL, STEP, Q_LO, Q_HI]], on=[MODEL, STEP],
                            how="left", validate="m:1")
    if merged[[Q_LO, Q_HI]].isna().any().any():
        missing = merged.loc[merged[Q_LO].isna(), [MODEL, STEP]].drop_duplicates()
        raise ValueError(f"No backtest errors for {missing.to_dict('records')}.")
    lo = merged[PREDICTION] * merged[Q_LO]
    hi = merged[PREDICTION] * merged[Q_HI]
    merged[LOWER] = np.minimum(lo, hi)
    merged[UPPER] = np.maximum(lo, hi)
    return merged.drop(columns=[Q_LO, Q_HI])


def coverage_check(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Coverage on ``coverage_split`` origins of intervals calibrated on ``calibration_split``.

    Only steps with origins in both splits are reported.
    """
    f = cfg.forecast
    calib = pred[pred[SPLIT_COLUMN] == f.calibration_split]
    check = pred[pred[SPLIT_COLUMN] == f.coverage_split]
    quantiles = ratio_quantiles(calib, cfg)
    scored = check.merge(quantiles, on=[MODEL, STEP], how="inner", suffixes=("", "_calib"))
    lo = scored[PREDICTION] * scored[Q_LO]
    hi = scored[PREDICTION] * scored[Q_HI]
    inside = (scored[ACTUAL] >= np.minimum(lo, hi)) & (scored[ACTUAL] <= np.maximum(lo, hi))
    out = (
        scored.assign(_inside=inside)
        .groupby([MODEL, STEP])
        .agg(**{COVERAGE: ("_inside", "mean"), N: ("_inside", "size")})
        .reset_index()
    )
    out[COVERAGE] = 100.0 * out[COVERAGE]
    out["nominal"] = 100.0 * f.interval_level
    return out
