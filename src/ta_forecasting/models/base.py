"""Uniform forecaster interface shared by ETS, ARIMA and TFT.

Every model works on the long-format panel produced in Phase 3 (one row per series and
quarter, columns ``ID``, ``time_idx``, ``quarter_label``, ``target``, ...).

* ``fit(train, val=None)``      learns from the training rows (``val`` only where a model
  needs it for model selection, i.e. TFT early stopping).
* ``predict(horizon)``          forecasts ``horizon`` quarters after the end of ``train``.
* ``forecast(history, horizon)`` forecasts ``horizon`` quarters after the end of ``history``
  (rolling origin). ``history`` must contain only actuals observed before the origin; models
  never look at anything else.
* ``save(path)`` / ``load(path)``.

Forecasts are returned as a long DataFrame with columns
``[ID, origin, step, time_idx, prediction]`` where ``origin`` is the first forecast
``time_idx`` and ``step`` runs from 1 to ``horizon``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import QUARTER_LABEL, TIME_IDX

logger = logging.getLogger(__name__)

ORIGIN = "origin"
STEP = "step"
PREDICTION = "prediction"


@dataclass(frozen=True)
class SeriesHistory:
    """Contiguous observed values of one series starting at ``start`` (a ``time_idx``)."""

    start: int
    values: np.ndarray

    @property
    def end(self) -> int:
        """Last observed ``time_idx``."""
        return self.start + len(self.values) - 1


def validate_horizon(horizon: int) -> int:
    """Return ``horizon`` as int or raise if it is not a positive integer."""
    if int(horizon) != horizon or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")
    return int(horizon)


def check_no_projection(df: pd.DataFrame, cfg: Config) -> None:
    """Raise if ``df`` holds a quarter after ``project.last_actual_quarter``."""
    if QUARTER_LABEL not in df.columns:
        raise ValueError(f"Panel needs a {QUARTER_LABEL!r} column to guard against projections.")
    last_actual = pd.Period(cfg.project.last_actual_quarter, freq="Q")
    quarters = pd.PeriodIndex(df[QUARTER_LABEL].astype(str), freq="Q")
    if (quarters > last_actual).any():
        raise ValueError(
            f"Data contains quarters after last_actual_quarter={cfg.project.last_actual_quarter}."
        )


def split_series(df: pd.DataFrame, cfg: Config) -> dict[str, SeriesHistory]:
    """Validate a long panel and return one contiguous history per series.

    Raises
    ------
    ValueError
        If columns are missing, a series has gaps or duplicates, the target is not finite,
        or a quarter lies after ``last_actual_quarter``.
    """
    sid = cfg.prep.columns.series_id
    target = cfg.prep.target_column
    missing = {sid, TIME_IDX, target} - set(df.columns)
    if missing:
        raise ValueError(f"Panel is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Panel is empty.")
    check_no_projection(df, cfg)

    out: dict[str, SeriesHistory] = {}
    for key, group in df.sort_values([sid, TIME_IDX]).groupby(sid, sort=True):
        idx = group[TIME_IDX].to_numpy(dtype=int)
        if len(idx) > 1 and not (np.diff(idx) == 1).all():
            raise ValueError(f"Series {key!r} has gaps or duplicate time_idx values.")
        values = group[target].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Series {key!r} has non-finite target values.")
        out[str(key)] = SeriesHistory(start=int(idx[0]), values=values)
    return out


def forecast_frame(
    forecasts: dict[str, np.ndarray], origins: dict[str, int], cfg: Config
) -> pd.DataFrame:
    """Assemble per-series forecast arrays into the common long output format."""
    rows = []
    for key in sorted(forecasts):
        values = np.asarray(forecasts[key], dtype=float)
        steps = np.arange(1, len(values) + 1)
        rows.append(
            pd.DataFrame(
                {
                    cfg.prep.columns.series_id: key,
                    ORIGIN: origins[key],
                    STEP: steps,
                    TIME_IDX: origins[key] + steps - 1,
                    PREDICTION: values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class Forecaster(ABC):
    """Abstract base for every model in the comparison."""

    name: ClassVar[str]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.fit_info: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        """True once ``fit`` (or ``load``) has completed."""
        return bool(self.fit_info)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted.")

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> Self:
        """Fit on ``train`` (and ``val`` for model selection where applicable)."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Forecast ``horizon`` quarters after the end of the training data."""

    @abstractmethod
    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast ``horizon`` quarters after the end of ``history`` (rolling origin)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Restore a model written by :meth:`save`."""
