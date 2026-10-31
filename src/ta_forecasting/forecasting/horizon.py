"""Forecast horizon bookkeeping: steps to ``forecast.end_quarter`` and per-model limits."""

from __future__ import annotations

import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import QUARTER_LABEL, TIME_IDX


def total_steps(cfg: Config) -> int:
    """Quarters from the one after ``last_actual_quarter`` up to ``forecast.end_quarter``."""
    last = pd.Period(cfg.project.last_actual_quarter, freq="Q")
    end = pd.Period(cfg.forecast.end_quarter, freq="Q")
    steps = (end - last).n
    if steps < 1:
        raise ValueError("forecast.end_quarter must lie after project.last_actual_quarter.")
    return steps


def model_steps(name: str, cfg: Config) -> int:
    """Forecast steps of ``name``: limited models stop at their training horizon."""
    steps = total_steps(cfg)
    if name in cfg.forecast.max_horizon_models:
        return min(steps, cfg.prep.max_prediction_length)
    return steps


def validated_steps(cfg: Config) -> int:
    """Steps covered by the Phase 5 comparison (the 'validated' horizon)."""
    return cfg.prep.max_prediction_length


def future_quarters(last_time_idx: int, steps: int, cfg: Config) -> pd.DataFrame:
    """``time_idx`` and quarter label of the ``steps`` quarters after ``last_time_idx``."""
    last = pd.Period(cfg.project.last_actual_quarter, freq="Q")
    return pd.DataFrame(
        {
            TIME_IDX: [last_time_idx + h for h in range(1, steps + 1)],
            QUARTER_LABEL: [str(last + h) for h in range(1, steps + 1)],
        }
    )


def check_panel_ends_at_last_actual(panel: pd.DataFrame, cfg: Config) -> int:
    """Return the last ``time_idx``; raise unless it is ``last_actual_quarter`` for all series."""
    sid = cfg.prep.columns.series_id
    last_rows = panel.sort_values(TIME_IDX).groupby(sid).tail(1)
    labels = set(last_rows[QUARTER_LABEL].astype(str))
    if labels != {cfg.project.last_actual_quarter}:
        raise ValueError(
            "Every series must end at last_actual_quarter "
            f"{cfg.project.last_actual_quarter}; found {sorted(labels)}."
        )
    ends = set(last_rows[TIME_IDX].astype(int))
    if len(ends) != 1:
        raise ValueError("Series end at different time_idx values.")
    return ends.pop()
