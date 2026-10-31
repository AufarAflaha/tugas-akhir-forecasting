"""Long-horizon rolling-origin backtest (additional to the Phase 5 comparison).

Origins run over every quarter from the start of ``forecast.backtest_first_split`` to the
last actual quarter. At origin ``o`` each model forecasts ``model_steps`` quarters from
``history = panel[time_idx < o]`` (Phase 5 protocol, ``refit_statistical = false``). Only
targets that are actuals (``time_idx <= last``) are scored, so horizon ``h`` is scored on
the origins ``first .. last - h + 1``: short horizons have more origins than long ones.
Each prediction keeps the split of its origin (val / test) for the coverage check.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN, TIME_IDX, split_bounds
from ta_forecasting.evaluation.backtest import MAIN, MODEL, run_backtest
from ta_forecasting.evaluation.evaluate import load_model
from ta_forecasting.evaluation.metrics import ACTUAL, metric_names, score
from ta_forecasting.forecasting.horizon import model_steps
from ta_forecasting.models.base import ORIGIN, PREDICTION, STEP, check_no_projection

logger = logging.getLogger(__name__)

ORIGIN_SET = "origins"
ALL_ORIGINS = "all"
N_ORIGINS = "n_origins"


def long_origins(panel: pd.DataFrame, cfg: Config) -> dict[str, list[int]]:
    """Origins grouped by the split they fall in, from ``backtest_first_split`` onwards."""
    bounds = split_bounds(panel)
    first = bounds[cfg.forecast.backtest_first_split][0]
    last = int(panel[TIME_IDX].max())
    out: dict[str, list[int]] = {}
    for name, (lo, hi) in bounds.items():
        origins = [o for o in range(max(lo, first), hi + 1) if o <= last]
        if origins:
            out[name] = origins
    return out


def run_long_backtest(cfg: Config, panel: pd.DataFrame) -> pd.DataFrame:
    """Scored forecasts ``[model, split, ID, origin, step, time_idx, prediction, actual]``."""
    check_no_projection(panel, cfg)
    local = cfg.model_copy(deep=True)
    local.evaluation.n_jobs = cfg.forecast.n_jobs
    if local.rolling.refit_statistical:
        raise ValueError("The long backtest uses the main protocol (refit_statistical = false).")
    origins = long_origins(panel, cfg)
    last = int(panel[TIME_IDX].max())
    frames = []
    for name in cfg.forecast.models:
        model = load_model(name, local, panel)
        result = run_backtest(model, panel, local, MAIN, origins=origins,
                              horizon=model_steps(name, cfg))
        pred = result.predictions
        beyond = pred[TIME_IDX] > last
        if pred.loc[~beyond, ACTUAL].isna().any():
            raise ValueError(f"{name}: an in-sample target has no actual value.")
        frames.append(pred[~beyond])
        logger.info("Long backtest %s: %d scored forecasts in %.1f s", name,
                    int((~beyond).sum()), result.seconds)
    out = pd.concat(frames, ignore_index=True)
    validate_long_backtest(out, panel, cfg)
    return out


def validate_long_backtest(pred: pd.DataFrame, panel: pd.DataFrame, cfg: Config) -> None:
    """Row counts per (model, step) and finite values; every target is an actual."""
    sid = cfg.prep.columns.series_id
    n_series = panel[sid].nunique()
    first = min(min(v) for v in long_origins(panel, cfg).values())
    last = int(panel[TIME_IDX].max())
    if not np.isfinite(pred[[PREDICTION, ACTUAL]].to_numpy(dtype=float)).all():
        raise ValueError("Long backtest has non-finite predictions or actuals.")
    if (pred[TIME_IDX] > last).any():
        raise ValueError("Long backtest scored a target after the last actual quarter.")
    for (model, step), group in pred.groupby([MODEL, STEP]):
        expected = n_series * (last - step + 1 - first + 1)
        if len(group) != expected:
            raise ValueError(f"{model}/h={step}: {len(group)} rows, expected {expected}.")
        if group.duplicated([sid, ORIGIN]).any():
            raise ValueError(f"{model}/h={step}: duplicate forecasts.")


def horizon_metrics(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Metrics per model and step over all origins and per origin split."""
    names = metric_names(cfg)
    parts = []
    overall = score(pred, [MODEL, STEP], names)
    overall.insert(1, ORIGIN_SET, ALL_ORIGINS)
    overall[N_ORIGINS] = pred.groupby([MODEL, STEP])[ORIGIN].nunique().to_numpy()
    parts.append(overall)
    by_split = score(pred, [MODEL, SPLIT_COLUMN, STEP], names)
    by_split[N_ORIGINS] = (
        pred.groupby([MODEL, SPLIT_COLUMN, STEP])[ORIGIN].nunique().to_numpy()
    )
    parts.append(by_split.rename(columns={SPLIT_COLUMN: ORIGIN_SET}))
    out = pd.concat(parts, ignore_index=True)
    rank = out[MODEL].map({m: i for i, m in enumerate(cfg.forecast.models)})
    out = out.assign(_rank=rank).sort_values([ORIGIN_SET, "_rank", STEP])
    return out.drop(columns="_rank").reset_index(drop=True)
