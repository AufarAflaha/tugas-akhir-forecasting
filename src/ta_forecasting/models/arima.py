"""ARIMA — seasonal ARIMA per series (statsmodels ``SARIMAX``).

Order search per series, on the training rows only, with every bound from ``cfg.arima``:

1. ``D``: seasonal differencing while the STL seasonal strength
   ``F_s = max(0, 1 - Var(R) / Var(S + R))`` exceeds ``seasonal_strength_threshold``
   (up to ``max_D``), as in ``forecast::nsdiffs``.
2. ``d``: KPSS test on the seasonally differenced series; difference while the null of
   stationarity is rejected at ``kpss_alpha`` (up to ``max_d``), as in ``forecast::ndiffs``.
3. Exhaustive grid ``p, q <= max_p, max_q`` and ``P, Q <= max_P, max_Q``; the converged
   candidate with the lowest information criterion wins. A constant (drift when
   ``d + D = 1``) is included only when ``d + D <= constant_max_total_diff``.

Order selection is independent per series, so it runs in parallel (``n_jobs``) without
affecting the result.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

from ta_forecasting.config import ARIMAConfig
from ta_forecasting.models.base import SeriesHistory
from ta_forecasting.models.statistical import (
    FitAttempt,
    PerSeriesForecaster,
    SeriesFit,
    attempt_fit,
    captured_warnings,
)

logger = logging.getLogger(__name__)

NO_TREND = "n"
CONSTANT = "c"


@dataclass(frozen=True)
class ArimaSpec:
    """SARIMAX structure of one series."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str

    def label(self) -> str:
        p, d, q = self.order
        sp, sd, sq, m = self.seasonal_order
        return f"({p},{d},{q})({sp},{sd},{sq})[{m}] trend={self.trend}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "trend": self.trend,
        }


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of the order search for one series (no data values)."""

    spec: ArimaSpec
    result: Any
    converged: bool
    fallback: bool
    n_candidates: int
    n_converged: int
    n_failed: int


# ---------------------------------------------------------------------------
# Differencing tests
# ---------------------------------------------------------------------------


def seasonal_strength(values: np.ndarray, period: int) -> float:
    """STL seasonal strength in [0, 1] (Wang, Smith & Hyndman, 2006)."""
    if len(values) < 2 * period + 1 or np.ptp(values) == 0:
        return 0.0
    decomposition = STL(values, period=period).fit()
    resid = np.asarray(decomposition.resid)
    denom = float(np.var(resid + np.asarray(decomposition.seasonal)))
    if denom <= 0:
        return 0.0
    return max(0.0, 1.0 - float(np.var(resid)) / denom)


def seasonal_diff_order(values: np.ndarray, cfg: ARIMAConfig) -> int:
    """Number of seasonal differences ``D`` (0..max_D)."""
    m = cfg.seasonal_period
    x = np.asarray(values, dtype=float)
    order = 0
    while order < cfg.max_D and seasonal_strength(x, m) > cfg.seasonal_strength_threshold:
        x = x[m:] - x[:-m]
        order += 1
    return order


def kpss_diff_order(values: np.ndarray, cfg: ARIMAConfig) -> int:
    """Number of regular differences ``d`` (0..max_d) from repeated KPSS tests."""
    x = np.asarray(values, dtype=float)
    order = 0
    while order < cfg.max_d and len(x) > 2 and np.ptp(x) > 0:
        with captured_warnings():  # p-value outside the lookup table -> InterpolationWarning
            p_value = kpss(x, regression=cfg.kpss_regression, nlags=cfg.kpss_nlags)[1]
        if p_value >= cfg.kpss_alpha:
            break
        x = np.diff(x)
        order += 1
    return order


def differencing_orders(values: np.ndarray, cfg: ARIMAConfig) -> tuple[int, int]:
    """Return ``(d, D)``: ``D`` first, then ``d`` on the seasonally differenced series."""
    m = cfg.seasonal_period
    big_d = seasonal_diff_order(values, cfg)
    x = np.asarray(values, dtype=float)
    for _ in range(big_d):
        x = x[m:] - x[:-m]
    return kpss_diff_order(x, cfg), big_d


# ---------------------------------------------------------------------------
# Candidate grid and fitting
# ---------------------------------------------------------------------------


def candidate_specs(d: int, big_d: int, cfg: ARIMAConfig) -> list[ArimaSpec]:
    """All (p, q, P, Q) combinations within the configured bounds, in a fixed order."""
    trend = CONSTANT if d + big_d <= cfg.constant_max_total_diff else NO_TREND
    specs = []
    for p, q, sp, sq in itertools.product(
        range(cfg.max_p + 1), range(cfg.max_q + 1), range(cfg.max_P + 1), range(cfg.max_Q + 1)
    ):
        seasonal = (sp, big_d, sq, cfg.seasonal_period) if sp or big_d or sq else (0, 0, 0, 0)
        specs.append(ArimaSpec((p, d, q), seasonal, trend))
    return specs


def _sarimax_converged(result: Any) -> bool:
    return bool(result.mle_retvals.get("converged", False)) if result.mle_retvals else True


def fit_spec(
    values: np.ndarray,
    spec: ArimaSpec,
    cfg: ARIMAConfig,
    start_params: np.ndarray | None = None,
) -> FitAttempt:
    """Fit one SARIMAX specification by maximum likelihood."""

    def run() -> Any:
        model = SARIMAX(
            values,
            order=spec.order,
            seasonal_order=spec.seasonal_order,
            trend=spec.trend,
            enforce_stationarity=cfg.enforce_stationarity,
            enforce_invertibility=cfg.enforce_invertibility,
        )
        return model.fit(
            start_params=start_params, method=cfg.fit_method, maxiter=cfg.maxiter, disp=False
        )

    return attempt_fit(run, _sarimax_converged)


def _criterion(result: Any, cfg: ARIMAConfig) -> float:
    value = float(getattr(result, cfg.information_criterion))
    return value if np.isfinite(value) else np.inf


def select_order(values: np.ndarray, cfg: ARIMAConfig) -> SelectionResult:
    """Choose ``(d, D)`` by tests and ``(p, q, P, Q)`` by information criterion."""
    d, big_d = differencing_orders(values, cfg)
    specs = candidate_specs(d, big_d, cfg)

    best: tuple[float, int, ArimaSpec, Any] | None = None
    best_any: tuple[float, int, ArimaSpec, Any] | None = None
    n_converged = n_failed = 0
    for rank, spec in enumerate(specs):
        attempt = fit_spec(values, spec, cfg)
        if attempt.result is None:
            n_failed += 1
            continue
        score = (_criterion(attempt.result, cfg), rank, spec, attempt.result)
        if best_any is None or score[:2] < best_any[:2]:
            best_any = score
        if attempt.converged:
            n_converged += 1
            if best is None or score[:2] < best[:2]:
                best = score

    if best is not None:
        _, _, spec, result = best
        return SelectionResult(spec, result, True, False, len(specs), n_converged, n_failed)
    if best_any is not None:  # nothing converged: keep the best finite candidate, flagged
        _, _, spec, result = best_any
        return SelectionResult(spec, result, False, False, len(specs), 0, n_failed)

    # every candidate failed: pure differencing model (with constant if allowed)
    spec = ArimaSpec((0, d, 0), specs[0].seasonal_order, specs[0].trend)
    attempt = fit_spec(values, spec, cfg)
    if attempt.result is None:
        raise RuntimeError(f"ARIMA fallback {spec.label()} failed: {attempt.error}")
    return SelectionResult(spec, attempt.result, attempt.converged, True, len(specs), 0, n_failed)


def _select_for_series(key: str, values: np.ndarray, cfg: ARIMAConfig) -> tuple[str, Any]:
    try:
        return key, select_order(values, cfg)
    except RuntimeError as exc:
        raise RuntimeError(f"Series {key!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class ARIMAForecaster(PerSeriesForecaster):
    """Per-series SARIMAX with orders chosen on the training data."""

    name: ClassVar[str] = "arima"

    def _fit_all(self, series: dict[str, SeriesHistory]) -> dict[str, SeriesFit]:
        a = self.cfg.arima
        jobs = Parallel(n_jobs=a.n_jobs)(
            delayed(_select_for_series)(key, hist.values, a) for key, hist in series.items()
        )
        fits: dict[str, SeriesFit] = {}
        for key, sel in sorted(jobs):
            hist = series[key]
            if sel.fallback:
                logger.warning("ARIMA: all candidates failed for %s; fallback used", key)
            elif not sel.converged:
                logger.warning("ARIMA: no converged candidate for %s", key)
            fits[key] = SeriesFit(
                series_id=key,
                train_start=hist.start,
                train_end=hist.end,
                spec=sel.spec.as_dict(),
                result=sel.result,
                converged=sel.converged,
                fallback=sel.fallback,
                extra={
                    "n_candidates": sel.n_candidates,
                    "n_converged_candidates": sel.n_converged,
                    "n_failed_candidates": sel.n_failed,
                },
            )
        return fits

    @staticmethod
    def _spec_of(fit: SeriesFit) -> ArimaSpec:
        s = fit.spec
        return ArimaSpec(tuple(s["order"]), tuple(s["seasonal_order"]), s["trend"])

    def order_table(self) -> dict[str, dict[str, Any]]:
        """Selected structure per series (no data values or criterion values)."""
        return {
            key: {**fit.spec, "converged": fit.converged, "fallback": fit.fallback, **fit.extra}
            for key, fit in sorted(self.fits.items())
        }

    def _summarize(self) -> dict[str, Any]:
        fits = self.fits.values()
        orders = [self._spec_of(f) for f in fits]

        def count(values: list[Any]) -> dict[str, int]:
            return {str(k): v for k, v in sorted(Counter(values).items())}

        a = self.cfg.arima
        return {
            "search": {
                "max_p": a.max_p,
                "max_d": a.max_d,
                "max_q": a.max_q,
                "max_P": a.max_P,
                "max_D": a.max_D,
                "max_Q": a.max_Q,
                "seasonal_period": a.seasonal_period,
                "information_criterion": a.information_criterion,
                "kpss_alpha": a.kpss_alpha,
                "seasonal_strength_threshold": a.seasonal_strength_threshold,
            },  # fmt: skip
            "n_candidate_fits": sum(f.extra["n_candidates"] for f in fits),
            "n_candidate_fits_non_converged": sum(
                f.extra["n_candidates"]
                - f.extra["n_converged_candidates"]
                - f.extra["n_failed_candidates"]
                for f in fits
            ),
            "n_candidate_fits_failed": sum(f.extra["n_failed_candidates"] for f in fits),
            "n_non_converged": sum(not f.converged for f in fits),
            "n_fallback": sum(f.fallback for f in fits),
            "non_converged_ids": sorted(f.series_id for f in fits if not f.converged),
            "fallback_ids": sorted(f.series_id for f in fits if f.fallback),
            "distribution": {
                "d": count([o.order[1] for o in orders]),
                "D": count([o.seasonal_order[1] for o in orders]),
                "p": count([o.order[0] for o in orders]),
                "q": count([o.order[2] for o in orders]),
                "P": count([o.seasonal_order[0] for o in orders]),
                "Q": count([o.seasonal_order[2] for o in orders]),
                "trend": count([o.trend for o in orders]),
                "model": count([o.label() for o in orders]),
            },
        }

    def _forecast_fixed(self, fit: SeriesFit, history: SeriesHistory, horizon: int) -> np.ndarray:
        """Kalman-filter the history with the training parameters, then forecast."""
        applied = fit.result.apply(history.values)
        return np.asarray(applied.forecast(horizon), dtype=float)

    def _refit(self, fit: SeriesFit, history: SeriesHistory) -> FitAttempt:
        return fit_spec(
            history.values, self._spec_of(fit), self.cfg.arima, start_params=fit.result.params
        )
