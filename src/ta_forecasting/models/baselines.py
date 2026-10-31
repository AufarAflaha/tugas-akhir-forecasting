"""Reference baselines behind the common forecaster interface (no parameters).

* ``naive``:          f(T + h) = y(T)
* ``seasonal_naive``: f(T + h) = y(T + h - m * ceil(h / m)), m = ``prep.quarter_period``
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.models.base import (
    Forecaster,
    SeriesHistory,
    forecast_frame,
    split_series,
    validate_horizon,
)


class _Baseline(Forecaster):
    """Shared logic: forecasts depend only on the history passed in."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._train: pd.DataFrame | None = None

    def _forecast_values(self, hist: SeriesHistory, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> Self:
        split_series(train, self.cfg)
        self._train = train.copy()
        self.fit_info = {"n_series": int(train[self.cfg.prep.columns.series_id].nunique())}
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        self._require_fitted()
        assert self._train is not None
        return self.forecast(self._train, horizon)

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        self._require_fitted()
        horizon = validate_horizon(horizon)
        series = split_series(history, self.cfg)
        forecasts = {k: self._forecast_values(h, horizon) for k, h in series.items()}
        origins = {k: h.end + 1 for k, h in series.items()}
        return forecast_frame(forecasts, origins, self.cfg)

    def save(self, path: Path) -> None:
        self._require_fitted()
        with Path(path).open("wb") as fh:
            pickle.dump({"config": self.cfg.model_dump(), "train": self._train}, fh)

    @classmethod
    def load(cls, path: Path) -> Self:
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - trusted local artefact
        return cls(Config(**payload["config"])).fit(payload["train"])


class NaiveForecaster(_Baseline):
    """Last observed value for every step."""

    name: ClassVar[str] = "naive"

    def _forecast_values(self, hist: SeriesHistory, horizon: int) -> np.ndarray:
        return np.full(horizon, hist.values[-1], dtype=float)


class SeasonalNaiveForecaster(_Baseline):
    """Value of the same quarter in the last observed season."""

    name: ClassVar[str] = "seasonal_naive"

    def _forecast_values(self, hist: SeriesHistory, horizon: int) -> np.ndarray:
        m = self.cfg.prep.quarter_period
        if len(hist.values) < m:
            raise ValueError(f"Seasonal naive needs at least {m} observations.")
        n = len(hist.values)
        idx = [n - 1 + h - m * math.ceil(h / m) for h in range(1, horizon + 1)]
        return hist.values[idx].astype(float)
