"""ETS — Holt-Winters exponential smoothing (statsmodels ``ExponentialSmoothing``).

One model per series with trend / seasonal / damping settings from ``cfg.ets``. A series
whose optimiser reports non-convergence keeps its model and is only recorded. The fallback
(``ets.fallback_seasonal``, by default no seasonal component) is used only when the primary
fit raises or returns non-finite output.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ta_forecasting.models.base import SeriesHistory
from ta_forecasting.models.statistical import (
    FitAttempt,
    PerSeriesForecaster,
    SeriesFit,
    attempt_fit,
)

logger = logging.getLogger(__name__)


def _holt_winters_converged(result: Any) -> bool:
    retvals = getattr(result, "mle_retvals", None)
    return bool(getattr(retvals, "success", True)) if retvals is not None else True


class ETSForecaster(PerSeriesForecaster):
    """Per-series additive damped Holt-Winters model."""

    name: ClassVar[str] = "ets"

    def _spec(self, seasonal: str | None) -> dict[str, Any]:
        e = self.cfg.ets
        return {
            "trend": e.trend,
            "damped_trend": e.damped_trend if e.trend else False,
            "seasonal": seasonal,
            "seasonal_periods": e.seasonal_periods if seasonal else None,
        }

    def _fit_spec(self, values: np.ndarray, spec: dict[str, Any]) -> FitAttempt:
        def run() -> Any:
            model = ExponentialSmoothing(
                values, initialization_method=self.cfg.ets.initialization_method, **spec
            )
            return model.fit()

        return attempt_fit(run, _holt_winters_converged)

    def _fit_all(self, series: dict[str, SeriesHistory]) -> dict[str, SeriesFit]:
        fits: dict[str, SeriesFit] = {}
        for key, hist in series.items():
            spec = self._spec(self.cfg.ets.seasonal)
            attempt = self._fit_spec(hist.values, spec)
            fallback = False
            if attempt.result is None:
                logger.warning("ETS fit failed for %s (%s); using fallback", key, attempt.error)
                spec = self._spec(self.cfg.ets.fallback_seasonal)
                attempt = self._fit_spec(hist.values, spec)
                fallback = True
                if attempt.result is None:
                    raise RuntimeError(f"ETS fallback also failed for series {key!r}.")
            if not attempt.converged:
                logger.warning("ETS optimiser did not converge for %s", key)
            fits[key] = SeriesFit(
                series_id=key,
                train_start=hist.start,
                train_end=hist.end,
                spec=spec,
                result=attempt.result,
                converged=attempt.converged,
                fallback=fallback,
            )
        return fits

    def _summarize(self) -> dict[str, Any]:
        non_converged = sorted(k for k, f in self.fits.items() if not f.converged)
        fallback = sorted(k for k, f in self.fits.items() if f.fallback)
        return {
            "spec": self._spec(self.cfg.ets.seasonal),
            "fallback_spec": self._spec(self.cfg.ets.fallback_seasonal),
            "initialization_method": self.cfg.ets.initialization_method,
            "n_non_converged": len(non_converged),
            "n_fallback": len(fallback),
            "n_failed": 0,
            "non_converged_ids": non_converged,
            "fallback_ids": fallback,
        }

    def _forecast_fixed(self, fit: SeriesFit, history: SeriesHistory, horizon: int) -> np.ndarray:
        """Run the training parameters and initial states over the full history."""
        p = fit.result.params
        spec = fit.spec
        init: dict[str, Any] = {"initial_level": p["initial_level"]}
        smoothing: dict[str, Any] = {"smoothing_level": p["smoothing_level"]}
        if spec["trend"]:
            init["initial_trend"] = p["initial_trend"]
            smoothing["smoothing_trend"] = p["smoothing_trend"]
            if spec["damped_trend"]:
                smoothing["damping_trend"] = p["damping_trend"]
        if spec["seasonal"]:
            init["initial_seasonal"] = p["initial_seasons"]
            smoothing["smoothing_seasonal"] = p["smoothing_seasonal"]
        model = ExponentialSmoothing(history.values, initialization_method="known", **init, **spec)
        result = model.fit(optimized=False, **smoothing)
        return np.asarray(result.forecast(horizon), dtype=float)

    def _refit(self, fit: SeriesFit, history: SeriesHistory) -> FitAttempt:
        return self._fit_spec(history.values, fit.spec)
