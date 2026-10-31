"""Per-series, per-model summary of the production forecasts (no single recommendation).

Two windows are summarised:

* ``short``: the validated horizon (steps 1 .. ``prep.max_prediction_length``), where every
  model has a forecast;
* ``full``:  every step the model forecasts (up to ``forecast.end_quarter``).

For each series, model and window: the cheapest forecast quarter (ties: earliest), its
change against the last actual value, the model's long-backtest MAPE at that step, whether
the change is smaller than that error, and how many models in the window pick the same
quarter. A path whose relative range is below ``forecast.flat_tolerance`` (e.g. naive) has no
cheapest quarter and is excluded from the agreement count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import QUARTER_LABEL
from ta_forecasting.evaluation.backtest import MODEL
from ta_forecasting.forecasting.horizon import validated_steps
from ta_forecasting.forecasting.long_backtest import ALL_ORIGINS, ORIGIN_SET
from ta_forecasting.models.base import PREDICTION, STEP

WINDOW = "window"
SHORT = "short"
FULL = "full"
LAST_ACTUAL = "last_actual"
CHEAPEST_QUARTER = "cheapest_quarter"
CHEAPEST_STEP = "cheapest_step"
CHEAPEST_VALUE = "cheapest_value"
CHANGE_PCT = "change_pct"
ERROR_AT_STEP = "mape_at_step"
WITHIN_ERROR = "within_error"
FLAT = "flat"
N_SAME = "n_models_same_quarter"
N_IN_WINDOW = "n_models_in_window"
CONSENSUS_QUARTER = "consensus_quarter"
CONSENSUS_COUNT = "consensus_count"
MIN_CONSENSUS = 2  # a consensus needs at least two models (definition, not a tuning knob)


def summarize(forecast: pd.DataFrame, last_actual: pd.Series, metrics: pd.DataFrame,
              cfg: Config) -> pd.DataFrame:
    """One row per (series, model, window). ``last_actual`` is indexed by series ID."""
    sid = cfg.prep.columns.series_id
    primary = cfg.metrics.primary
    errors = metrics[metrics[ORIGIN_SET] == ALL_ORIGINS].set_index([MODEL, STEP])[primary]
    rows = []
    for window, max_step in ((SHORT, validated_steps(cfg)), (FULL, None)):
        part = forecast if max_step is None else forecast[forecast[STEP] <= max_step]
        for (key, model), group in part.groupby([sid, MODEL], sort=True):
            g = group.sort_values(STEP)
            values = g[PREDICTION].to_numpy(dtype=float)
            last = float(last_actual.loc[key])
            flat = bool(np.ptp(values) <= cfg.forecast.flat_tolerance * abs(values).max())
            row = {sid: key, MODEL: model, WINDOW: window, LAST_ACTUAL: last, FLAT: flat,
                   "n_steps": len(values)}
            if flat:
                row.update({CHEAPEST_QUARTER: None, CHEAPEST_STEP: np.nan,
                            CHEAPEST_VALUE: float(values[0]),
                            CHANGE_PCT: 100.0 * (values[0] / last - 1), ERROR_AT_STEP: np.nan,
                            WITHIN_ERROR: np.nan})
            else:
                i = int(np.argmin(values))
                step = int(g[STEP].iloc[i])
                change = 100.0 * (values[i] / last - 1)
                err = float(errors.loc[(model, step)])
                row.update({CHEAPEST_QUARTER: str(g[QUARTER_LABEL].iloc[i]),
                            CHEAPEST_STEP: step, CHEAPEST_VALUE: float(values[i]),
                            CHANGE_PCT: change, ERROR_AT_STEP: err,
                            WITHIN_ERROR: bool(abs(change) < err)})
            rows.append(row)
    out = pd.DataFrame(rows)
    return _agreement(out, cfg)


def _agreement(out: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Agreement among non-flat models of the same series and window."""
    sid = cfg.prep.columns.series_id
    keys = [sid, WINDOW]
    picked = out[~out[FLAT]]
    counts = picked.groupby([*keys, CHEAPEST_QUARTER]).size().rename(N_SAME).reset_index()
    out = out.merge(counts, on=[*keys, CHEAPEST_QUARTER], how="left")
    out[N_SAME] = out[N_SAME].fillna(0).astype(int)
    n_window = picked.groupby(keys).size().rename(N_IN_WINDOW).reset_index()
    out = out.merge(n_window, on=keys, how="left")
    out[N_IN_WINDOW] = out[N_IN_WINDOW].fillna(0).astype(int)
    # consensus: most frequent quarter (ties: earliest quarter)
    top = counts.sort_values([*keys, N_SAME, CHEAPEST_QUARTER], ascending=[True, True, False,
                                                                           True])
    top = top.groupby(keys).head(1).rename(
        columns={CHEAPEST_QUARTER: CONSENSUS_QUARTER, N_SAME: CONSENSUS_COUNT})
    out = out.merge(top, on=keys, how="left")
    out[CONSENSUS_COUNT] = out[CONSENSUS_COUNT].fillna(0).astype(int)
    # no consensus when no quarter is picked by at least two models
    out.loc[out[CONSENSUS_COUNT] < MIN_CONSENSUS, CONSENSUS_QUARTER] = None
    order = {m: i for i, m in enumerate(cfg.forecast.models)}
    out = out.assign(_r=out[MODEL].map(order)).sort_values([sid, WINDOW, "_r"])
    return out.drop(columns="_r").reset_index(drop=True)


def cheapest_counts(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate: number of series per model, window and cheapest quarter (plus flat)."""
    label = summary[CHEAPEST_QUARTER].fillna("datar")
    return (
        summary.assign(**{CHEAPEST_QUARTER: label})
        .groupby([MODEL, WINDOW, CHEAPEST_QUARTER]).size().rename("n_series").reset_index()
    )


def summary_stats(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate per model and window: share within error, flat, and agreement."""
    rows = []
    for (model, window), g in summary.groupby([MODEL, WINDOW]):
        picked = g[~g[FLAT]]
        rows.append({
            MODEL: model, WINDOW: window, "n_series": len(g), "n_flat": int(g[FLAT].sum()),
            "n_within_error": int(picked[WITHIN_ERROR].astype(bool).sum()),
            "n_cheapest_is_step1": int((picked[CHEAPEST_STEP] == 1).sum()),
            "median_change_pct": float(g[CHANGE_PCT].median()),
            "n_agree_with_consensus": int(
                (picked[CHEAPEST_QUARTER] == picked[CONSENSUS_QUARTER]).sum()),
        })
    order = {m: i for i, m in enumerate(cfg.forecast.models)}
    out = pd.DataFrame(rows)
    return out.assign(_r=out[MODEL].map(order)).sort_values([WINDOW, "_r"]).drop(
        columns="_r").reset_index(drop=True)
