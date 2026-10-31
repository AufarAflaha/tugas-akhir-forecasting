"""Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

For forecast horizon ``h`` and loss differential ``d_t = L(e_a,t) - L(e_b,t)`` over the
``n`` rolling origins:

    var(d̄) = (γ0 + 2 Σ_{k=1}^{h-1} γk) / n        (γk: autocovariance, divisor n)
    DM     = d̄ / sqrt(var(d̄))
    DM*    = DM * sqrt((n + 1 - 2h + h (h - 1) / n) / n),   p = 2 P(t_{n-1} > |DM*|)

If the long-run variance is not positive, γ0 alone is used (Diebold, 1995 practice). A
negative ``mean_diff`` means model ``a`` has the lower loss.

Two levels are reported on ``evaluation.dm_split``:
* per series and horizon (one test per series, counted, with FDR correction over series),
* panel level: the loss averaged over series at each origin, one test per horizon.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN
from ta_forecasting.evaluation.backtest import MAIN, MODE, MODEL
from ta_forecasting.evaluation.metrics import ACTUAL, PERCENT
from ta_forecasting.models.base import ORIGIN, PREDICTION, STEP

LOSS = "loss"


@dataclass(frozen=True)
class DMResult:
    """Outcome of one Diebold-Mariano test (statistic is HLN-corrected)."""

    statistic: float
    p_value: float
    mean_diff: float
    n: int
    variance_fallback: bool


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> DMResult:
    """Two-sided HLN-corrected DM test of equal expected loss."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 2 or horizon < 1:
        raise ValueError("Need at least two loss differentials and horizon >= 1.")
    mean = float(d.mean())
    centred = d - mean
    gammas = [float(np.dot(centred[k:], centred[: n - k]) / n) for k in range(horizon)]
    long_run = gammas[0] + 2.0 * sum(gammas[1:])
    fallback = long_run <= 0
    if fallback:
        long_run = gammas[0]
    if long_run <= 0:  # constant differential: test undefined
        return DMResult(math.nan, math.nan, mean, n, fallback)
    dm = mean / math.sqrt(long_run / n)
    correction = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = dm * correction
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return DMResult(statistic, p_value, mean, n, fallback)


def loss(actual: np.ndarray, forecast: np.ndarray, kind: str) -> np.ndarray:
    """Per-forecast loss: absolute percentage error, absolute error or squared error."""
    err = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    if kind == "ape":
        return PERCENT * np.abs(err) / np.abs(np.asarray(actual, dtype=float))
    if kind == "ae":
        return np.abs(err)
    if kind == "se":
        return err**2
    raise ValueError(f"Unknown DM loss {kind!r}.")


def _loss_table(predictions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    e = cfg.evaluation
    sid = cfg.prep.columns.series_id
    part = predictions[
        (predictions[SPLIT_COLUMN] == e.dm_split)
        & (predictions[MODE] == MAIN)
        & (predictions[MODEL].isin(e.main_models))
    ].copy()
    part[LOSS] = loss(part[ACTUAL].to_numpy(), part[PREDICTION].to_numpy(), e.dm_loss)
    return part.pivot_table(index=[sid, STEP, ORIGIN], columns=MODEL, values=LOSS)


def diebold_mariano(predictions: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-series summary, panel-level tests) for every pair of main models."""
    e = cfg.evaluation
    sid = cfg.prep.columns.series_id
    table = _loss_table(predictions, cfg)
    summary_rows, panel_rows = [], []
    for a, b in itertools.combinations(e.main_models, 2):
        for step, by_step in table.groupby(level=STEP):
            results = [
                dm_test(g[a].to_numpy(), g[b].to_numpy(), int(step))
                for _, g in by_step.groupby(level=sid)
            ]
            p = np.array([r.p_value for r in results])
            diff = np.array([r.mean_diff for r in results])
            defined = ~np.isnan(p)
            adjusted = np.full_like(p, np.nan)
            if defined.any():
                adjusted[defined] = multipletests(
                    p[defined], alpha=e.dm_alpha, method=e.dm_fdr_method
                )[1]
            sig = defined & (p < e.dm_alpha)
            sig_adj = defined & (adjusted < e.dm_alpha)
            summary_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    STEP: int(step),
                    "n_series": len(results),
                    "n_undefined": int((~defined).sum()),
                    "n_variance_fallback": int(sum(r.variance_fallback for r in results)),
                    "n_sig_a_better": int((sig & (diff < 0)).sum()),
                    "n_sig_b_better": int((sig & (diff > 0)).sum()),
                    "n_sig_a_better_fdr": int((sig_adj & (diff < 0)).sum()),
                    "n_sig_b_better_fdr": int((sig_adj & (diff > 0)).sum()),
                    "n_origins": results[0].n,
                }
            )
            avg = by_step.groupby(level=ORIGIN)[[a, b]].mean()
            res = dm_test(avg[a].to_numpy(), avg[b].to_numpy(), int(step))
            panel_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    STEP: int(step),
                    "mean_loss_diff": res.mean_diff,
                    "dm_hln": res.statistic,
                    "p_value": res.p_value,
                    "significant": bool(res.p_value < e.dm_alpha)
                    if not math.isnan(res.p_value)
                    else False,
                    "better": (a if res.mean_diff < 0 else b) if res.p_value < e.dm_alpha else "-",
                    "n_origins": res.n,
                    "variance_fallback": res.variance_fallback,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(panel_rows)
