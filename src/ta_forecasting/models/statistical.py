"""Shared logic for per-series statistical models (ETS and ARIMA).

One model is fitted per series on the training rows. At a rolling origin the model either
keeps its training parameters and only runs its filter over the history
(``rolling.refit_statistical = false``) or re-estimates the parameters on the history with
the configuration chosen on train (``true``). If such a re-estimation fails, the training
parameters are used for that origin and the event is counted in ``forecast_log``.
"""

from __future__ import annotations

import logging
import pickle
import time
import warnings
from abc import abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, TypeVar

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ta_forecasting.config import Config
from ta_forecasting.models.base import (
    Forecaster,
    SeriesHistory,
    forecast_frame,
    split_series,
    validate_horizon,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SeriesFit:
    """Fitted model of one series plus bookkeeping (no data values besides the result)."""

    series_id: str
    train_start: int
    train_end: int
    spec: dict[str, Any]
    result: Any
    converged: bool
    fallback: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitAttempt:
    """Outcome of one fitting call with warnings captured."""

    result: Any | None
    converged: bool
    error: str | None


@contextmanager
def captured_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Record all warnings raised inside the block instead of printing them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


FIT_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError, IndexError)


def attempt_fit(fit: Callable[[], T], is_converged: Callable[[T], bool]) -> FitAttempt:
    """Run ``fit``; report failure (exception or non-finite output) and convergence."""
    with captured_warnings() as caught:
        try:
            result = fit()
        except FIT_ERRORS as exc:
            return FitAttempt(None, False, f"{type(exc).__name__}: {exc}")
    if not _finite_result(result):
        return FitAttempt(None, False, "non-finite parameters or fitted values")
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return FitAttempt(result, bool(is_converged(result)) and not warned, None)


def _numeric_params(params: Any) -> np.ndarray:
    """Flatten numeric parameters (arrays, Series or a dict such as Holt-Winters').

    In the Holt-Winters dict, components that the model does not use (e.g. the seasonal
    smoothing of a non-seasonal model) are NaN by design and are skipped.
    """
    if not isinstance(params, dict):
        return np.asarray(params, dtype=float).ravel()
    parts = []
    for value in params.values():
        if value is None or isinstance(value, bool):
            continue
        arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        if arr.size == 1 and np.isnan(arr[0]):
            continue
        parts.append(arr)
    return np.concatenate(parts) if parts else np.empty(0)


def _finite_result(result: Any) -> bool:
    checks = (
        _numeric_params(result.params),
        np.asarray(result.fittedvalues, dtype=float),
        np.asarray(result.forecast(1), dtype=float),
    )
    return all(bool(np.isfinite(arr).all()) for arr in checks)


class PerSeriesForecaster(Forecaster):
    """A dictionary of independent per-series models behind the common interface."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.fits: dict[str, SeriesFit] = {}
        self.forecast_log: Counter[str] = Counter()

    # -- to implement ------------------------------------------------------------------

    @abstractmethod
    def _fit_all(self, series: dict[str, SeriesHistory]) -> dict[str, SeriesFit]:
        """Fit one model per series on the training histories."""

    @abstractmethod
    def _summarize(self) -> dict[str, Any]:
        """Aggregate, data-free description of the fitted models."""

    @abstractmethod
    def _forecast_fixed(self, fit: SeriesFit, history: SeriesHistory, horizon: int) -> np.ndarray:
        """Forecast with the training parameters, filtering over ``history``."""

    @abstractmethod
    def _refit(self, fit: SeriesFit, history: SeriesHistory) -> FitAttempt:
        """Re-estimate the parameters on ``history`` with the train configuration."""

    # -- interface ---------------------------------------------------------------------

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> Self:
        """Fit one model per series on ``train``. ``val`` is not used by statistical models."""
        series = split_series(train, self.cfg)
        start = time.perf_counter()
        self.fits = self._fit_all(series)
        seconds = time.perf_counter() - start
        self.fit_info = {"n_series": len(self.fits), "fit_seconds": round(seconds, 3)}
        self.fit_info.update(self._summarize())
        logger.info("%s fitted on %d series in %.1f s", self.name, len(self.fits), seconds)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Forecast ``horizon`` quarters after the end of the training data."""
        self._require_fitted()
        horizon = validate_horizon(horizon)
        forecasts = {
            key: np.asarray(fit.result.forecast(horizon), dtype=float)
            for key, fit in self.fits.items()
        }
        origins = {key: fit.train_end + 1 for key, fit in self.fits.items()}
        return forecast_frame(forecasts, origins, self.cfg)

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast ``horizon`` quarters after the last observation of ``history``."""
        self._require_fitted()
        horizon = validate_horizon(horizon)
        series = split_series(history, self.cfg)
        unknown = sorted(set(series) - set(self.fits))
        if unknown:
            raise ValueError(f"{len(unknown)} series in history were not fitted.")

        forecasts: dict[str, np.ndarray] = {}
        for key, hist in series.items():
            fit = self.fits[key]
            if hist.start != fit.train_start:
                raise ValueError(
                    f"History of {key!r} must start at the training start (time_idx "
                    f"{fit.train_start}); got {hist.start}."
                )
            forecasts[key] = self._forecast_one(fit, hist, horizon)
        origins = {key: hist.end + 1 for key, hist in series.items()}
        return forecast_frame(forecasts, origins, self.cfg)

    def _forecast_one(self, fit: SeriesFit, history: SeriesHistory, horizon: int) -> np.ndarray:
        if not self.cfg.rolling.refit_statistical:
            self.forecast_log["fixed_parameters"] += 1
            return self._forecast_fixed(fit, history, horizon)
        attempt = self._refit(fit, history)
        if attempt.result is None:
            self.forecast_log["refit_failed_used_train_parameters"] += 1
            logger.warning("%s refit failed for %s: %s", self.name, fit.series_id, attempt.error)
            return self._forecast_fixed(fit, history, horizon)
        self.forecast_log["refit" if attempt.converged else "refit_not_converged"] += 1
        return np.asarray(attempt.result.forecast(horizon), dtype=float)

    def save(self, path: Path) -> None:
        """Pickle the fitted models, configuration and fit summary."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.name,
            "config": self.cfg.model_dump(),
            "fit_info": self.fit_info,
            "fits": self.fits,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a model pickled by :meth:`save`. Only load files this project wrote."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - trusted local artefact
        if payload.get("model") != cls.name:
            raise ValueError(f"{path} holds a {payload.get('model')!r} model, not {cls.name!r}.")
        obj = cls(Config(**payload["config"]))
        obj.fits = payload["fits"]
        obj.fit_info = payload["fit_info"]
        return obj
