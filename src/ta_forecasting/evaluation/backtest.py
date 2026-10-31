"""Rolling-origin backtest with identical origins for every model.

For a split with time_idx range [first, last] and horizon H the origins are
``first .. last - H + 1``; every forecast target lies inside the split. At origin ``o`` the
model receives ``history = panel[time_idx < o]`` (all actuals before the origin) and
nothing else. The panel itself never holds quarters after ``last_actual_quarter``.

Statistical models are independent per series, so their series are split into chunks that
run in parallel (``evaluation.n_jobs``); the results are identical to a serial run.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN, TIME_IDX, split_bounds
from ta_forecasting.evaluation.metrics import ACTUAL
from ta_forecasting.models.base import ORIGIN, PREDICTION, STEP, Forecaster, check_no_projection
from ta_forecasting.models.statistical import PerSeriesForecaster

logger = logging.getLogger(__name__)

MODEL = "model"
MODE = "mode"
MAIN = "main"
REFIT = "refit"


@dataclass
class BacktestResult:
    """Forecasts of one model in one mode, plus timing and refit bookkeeping."""

    model: str
    mode: str
    predictions: pd.DataFrame
    seconds: float
    forecast_log: Counter[str] = field(default_factory=Counter)


def split_origins(panel: pd.DataFrame, cfg: Config) -> dict[str, list[int]]:
    """Rolling origins per configured split (first quarter .. last full-horizon start)."""
    horizon = cfg.prep.max_prediction_length
    bounds = split_bounds(panel)
    out: dict[str, list[int]] = {}
    for name in cfg.evaluation.splits:
        first, last = bounds[name]
        origins = list(range(first, last - horizon + 2))
        if not origins:
            raise ValueError(f"Split {name!r} is shorter than the horizon {horizon}.")
        out[name] = origins
    return out


def _run_origins(
    model: Forecaster, panel: pd.DataFrame, origins: dict[str, list[int]], horizon: int
) -> tuple[pd.DataFrame, Counter[str]]:
    frames = []
    for split, split_origins_ in origins.items():
        for origin in split_origins_:
            history = panel[panel[TIME_IDX] < origin]
            if history[TIME_IDX].max() != origin - 1:
                raise RuntimeError(f"History before origin {origin} is incomplete.")
            out = model.forecast(history, horizon)
            out[SPLIT_COLUMN] = split
            frames.append(out)
    log = getattr(model, "forecast_log", Counter())
    return pd.concat(frames, ignore_index=True), Counter(log)


def _subset(model: PerSeriesForecaster, ids: list[str]) -> PerSeriesForecaster:
    part = copy.copy(model)
    part.fits = {k: model.fits[k] for k in ids}
    part.forecast_log = Counter()
    return part


def run_backtest(
    model: Forecaster,
    panel: pd.DataFrame,
    cfg: Config,
    mode: str = MAIN,
) -> BacktestResult:
    """Forecast every origin of every configured split with ``model``."""
    check_no_projection(panel, cfg)
    horizon = cfg.prep.max_prediction_length
    origins = split_origins(panel, cfg)
    sid = cfg.prep.columns.series_id

    start = time.perf_counter()
    n_jobs = cfg.evaluation.n_jobs
    if isinstance(model, PerSeriesForecaster) and n_jobs != 1:
        ids = sorted(model.fits)
        chunks = [
            [str(x) for x in c] for c in np.array_split(ids, max(1, min(len(ids), abs(n_jobs))))
        ]
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_run_origins)(
                _subset(model, chunk), panel[panel[sid].isin(chunk)], origins, horizon
            )
            for chunk in chunks
            if chunk
        )
        predictions = pd.concat([p for p, _ in parts], ignore_index=True)
        log: Counter[str] = sum((c for _, c in parts), Counter())
    else:
        predictions, log = _run_origins(model, panel, origins, horizon)
    seconds = time.perf_counter() - start

    actual = panel[[sid, TIME_IDX, cfg.prep.target_column]].rename(
        columns={cfg.prep.target_column: ACTUAL}
    )
    predictions = predictions.merge(actual, on=[sid, TIME_IDX], how="left", validate="m:1")
    predictions.insert(0, MODE, mode)
    predictions.insert(0, MODEL, model.name)
    order = [MODEL, MODE, SPLIT_COLUMN, sid, ORIGIN, STEP, TIME_IDX, PREDICTION, ACTUAL]
    predictions = predictions[order].sort_values([SPLIT_COLUMN, sid, ORIGIN, STEP])
    logger.info("Backtest %s (%s): %d rows in %.1f s", model.name, mode, len(predictions), seconds)
    return BacktestResult(model.name, mode, predictions.reset_index(drop=True), seconds, log)


def validate_predictions(predictions: pd.DataFrame, panel: pd.DataFrame, cfg: Config) -> None:
    """Check row counts, split membership of every target and identical origins per model.

    Raises
    ------
    ValueError
        On any violation.
    """
    sid = cfg.prep.columns.series_id
    horizon = cfg.prep.max_prediction_length
    origins = split_origins(panel, cfg)
    bounds = split_bounds(panel)
    n_series = panel[sid].nunique()

    if predictions[[PREDICTION, ACTUAL]].isna().any().any():
        raise ValueError("Predictions or actuals contain missing values.")
    if not np.isfinite(predictions[[PREDICTION, ACTUAL]].to_numpy(dtype=float)).all():
        raise ValueError("Predictions or actuals are not finite.")

    reference: dict[str, set[tuple[str, int]]] = {}
    for (model, mode, split), group in predictions.groupby([MODEL, MODE, SPLIT_COLUMN]):
        expected = n_series * len(origins[split]) * horizon
        if len(group) != expected:
            raise ValueError(f"{model}/{mode}/{split}: {len(group)} rows, expected {expected}.")
        first, last = bounds[split]
        if group[TIME_IDX].min() < first or group[TIME_IDX].max() > last:
            raise ValueError(f"{model}/{mode}/{split}: a target lies outside the split.")
        if sorted(group[ORIGIN].unique()) != origins[split]:
            raise ValueError(f"{model}/{mode}/{split}: origins differ from the protocol.")
        if group.duplicated([sid, ORIGIN, STEP]).any():
            raise ValueError(f"{model}/{mode}/{split}: duplicate forecasts.")
        keys = set(zip(group[sid], group[ORIGIN], strict=True))
        if split in reference and keys != reference[split]:
            raise ValueError(f"{model}/{mode}/{split}: (series, origin) set differs.")
        reference.setdefault(split, keys)
