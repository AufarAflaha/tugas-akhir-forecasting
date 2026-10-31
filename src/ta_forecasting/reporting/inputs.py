"""Load and validate the inputs of the final report.

Allowed inputs (nothing else is read):

* ``<reports_dir>/<prefix>_*.csv`` — Phase 5 aggregate tables;
* ``<models_dir>/training_summary.json`` — training times, counts, package versions;
* ``<models_dir>/arima_orders.json`` — keyed by series ID, so it is reduced to counts here
  and the IDs never leave this module;
* ``<processed_dir>/split_summary.json`` — split boundaries and row counts.

Every table is checked for completeness (all models, splits, horizons and model pairs) and
for columns that would carry series-level values (``report.forbidden_columns``).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN
from ta_forecasting.evaluation.backtest import MODE, MODEL
from ta_forecasting.evaluation.metrics import metric_names
from ta_forecasting.models.base import STEP

TABLE_KEYS = (
    "overall",
    "per_horizon",
    "per_series_distribution",
    "wins",
    "dm_panel",
    "dm_per_series_summary",
    "timing",
)
SENSITIVITY_KEY = "sensitivity"
PAIR_A = "model_a"
PAIR_B = "model_b"


class ReportInputError(ValueError):
    """An input of the report is missing, incomplete or not aggregate-only."""


@dataclass(frozen=True)
class ArimaOrderCounts:
    """Counts over the per-series ARIMA orders (no series IDs)."""

    n_series: int
    d: dict[int, int]
    seasonal_d: dict[int, int]
    n_seasonal_terms: int
    n_converged: int
    n_fallback: int


@dataclass(frozen=True)
class ReportInputs:
    """Everything the report may use."""

    tables: dict[str, pd.DataFrame]
    training: dict[str, Any]
    split_summary: dict[str, Any]
    arima: ArimaOrderCounts
    sources: list[Path] = field(default_factory=list)
    # Series IDs, held only to check that no output contains them; never written.
    series_ids: frozenset[str] = field(default_factory=frozenset, repr=False)


def table_path(cfg: Config, key: str) -> Path:
    """Path of the Phase 5 CSV ``key``."""
    return Path(cfg.output.reports_dir) / f"{cfg.evaluation.report_prefix}_{key}.csv"


def input_paths(cfg: Config) -> dict[str, Path]:
    """Every file the report reads, by key."""
    paths = {key: table_path(cfg, key) for key in TABLE_KEYS}
    if cfg.evaluation.run_sensitivity:
        paths[SENSITIVITY_KEY] = table_path(cfg, SENSITIVITY_KEY)
    models = Path(cfg.output.models_dir)
    paths["training_summary"] = models / cfg.output.training_summary_file
    paths["arima_orders"] = models / cfg.output.arima_orders_file
    paths["split_summary"] = Path(cfg.data.processed_dir) / cfg.prep.summary_file
    return paths


def main_models(cfg: Config) -> list[str]:
    """Main models in report order."""
    return [m for m in cfg.report.model_order if m in cfg.evaluation.main_models]


def all_models(cfg: Config) -> list[str]:
    """Main models and baselines in report order."""
    wanted = {*cfg.evaluation.main_models, *cfg.evaluation.baselines}
    missing = wanted - set(cfg.report.model_order)
    if missing:
        raise ReportInputError(f"report.model_order lacks {sorted(missing)}.")
    return [m for m in cfg.report.model_order if m in wanted]


def model_pairs(cfg: Config) -> list[tuple[str, str]]:
    """Main-model pairs in the order used by the Diebold-Mariano tables."""
    return list(combinations(cfg.evaluation.main_models, 2))


def load_inputs(cfg: Config) -> ReportInputs:
    """Read and validate all report inputs."""
    paths = input_paths(cfg)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise ReportInputError(
            "Missing report input(s): " + ", ".join(missing) + ". Run `evaluate` first."
        )
    table_keys = [k for k in (*TABLE_KEYS, SENSITIVITY_KEY) if k in paths]
    tables = {key: pd.read_csv(paths[key]) for key in table_keys}
    for key, frame in tables.items():
        check_aggregate_only(frame, cfg, key)
    validate_tables(tables, cfg)

    training = json.loads(paths["training_summary"].read_text(encoding="utf-8"))
    split_summary = json.loads(paths["split_summary"].read_text(encoding="utf-8"))
    orders = json.loads(paths["arima_orders"].read_text(encoding="utf-8"))
    validate_metadata(training, split_summary, cfg)
    return ReportInputs(
        tables=tables,
        training=training,
        split_summary=split_summary,
        arima=count_arima_orders(orders),
        sources=list(paths.values()),
        series_ids=frozenset(map(str, orders)),
    )


def check_no_series_ids(text: str, inputs: ReportInputs, name: str) -> None:
    """Raise if any series ID appears in ``text`` (the ID itself is not echoed)."""
    hits = sum(1 for sid in inputs.series_ids if sid and sid in text)
    if hits:
        raise ReportInputError(f"{name}: contains {hits} series ID(s); refusing to write.")


def check_aggregate_only(frame: pd.DataFrame, cfg: Config, name: str) -> None:
    """Reject tables with series IDs, actuals, forecasts, time indices or origins."""
    forbidden = {*cfg.report.forbidden_columns, cfg.prep.columns.series_id}
    found = sorted(forbidden & set(frame.columns))
    if found:
        raise ReportInputError(f"{name}: non-aggregate column(s) {found}.")


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ReportInputError(f"{name}: missing column(s) {missing}.")


def _require_keys(frame: pd.DataFrame, keys: list[str], expected: set[tuple], name: str) -> None:
    present = set(frame[keys].itertuples(index=False, name=None))
    missing = expected - present
    if missing:
        raise ReportInputError(f"{name}: missing {keys} combination(s) {sorted(missing)}.")
    if frame.duplicated(keys).any():
        raise ReportInputError(f"{name}: duplicated {keys} rows.")


def validate_tables(tables: dict[str, pd.DataFrame], cfg: Config) -> None:
    """Every model, split, horizon and pair the report needs is present exactly once."""
    r, e = cfg.report, cfg.evaluation
    names = metric_names(cfg)
    models = all_models(cfg)
    steps = range(1, cfg.prep.max_prediction_length + 1)
    if r.split not in e.splits:
        raise ReportInputError(f"report.split {r.split!r} is not in evaluation.splits.")

    overall = tables["overall"]
    _require_columns(overall, [MODE, MODEL, SPLIT_COLUMN, *names, "n"], "overall")
    main_keys = {(r.main_mode, m, s) for m in models for s in e.splits}
    _require_keys(overall[overall[MODE] == r.main_mode], [MODE, MODEL, SPLIT_COLUMN],
                  main_keys, "overall")

    step = tables["per_horizon"]
    _require_columns(step, [MODE, MODEL, SPLIT_COLUMN, STEP, *names, "n"], "per_horizon")
    step_keys = {(r.main_mode, m, s, h) for m in models for s in e.splits for h in steps}
    _require_keys(step[step[MODE] == r.main_mode], [MODE, MODEL, SPLIT_COLUMN, STEP],
                  step_keys, "per_horizon")

    dist = tables["per_series_distribution"]
    pct = [f"p{q:g}" for q in e.percentiles]
    _require_columns(dist, [MODE, MODEL, SPLIT_COLUMN, "metric", "mean", *pct, "n_series"],
                     "per_series_distribution")
    primary = cfg.metrics.primary
    dist_keys = {(r.main_mode, m, s, primary) for m in models for s in e.splits}
    _require_keys(dist[dist[MODE] == r.main_mode], [MODE, MODEL, SPLIT_COLUMN, "metric"],
                  dist_keys, "per_series_distribution")

    wins = tables["wins"]
    _require_columns(wins, [SPLIT_COLUMN, "metric", "n_series", *e.main_models, "tie"], "wins")
    _require_keys(wins, [SPLIT_COLUMN, "metric"], {(s, primary) for s in e.splits}, "wins")

    pair_keys = {(a, b, h) for a, b in model_pairs(cfg) for h in steps}
    dm = tables["dm_panel"]
    _require_columns(dm, [PAIR_A, PAIR_B, STEP, "p_value", "significant", "better"], "dm_panel")
    _require_keys(dm, [PAIR_A, PAIR_B, STEP], pair_keys, "dm_panel")

    dms = tables["dm_per_series_summary"]
    _require_columns(dms, [PAIR_A, PAIR_B, STEP, "n_series", "n_undefined",
                           "n_sig_a_better_fdr", "n_sig_b_better_fdr", "n_origins"],
                     "dm_per_series_summary")
    _require_keys(dms, [PAIR_A, PAIR_B, STEP], pair_keys, "dm_per_series_summary")

    timing = tables["timing"]
    _require_columns(timing, [MODEL, MODE, "seconds"], "timing")
    _require_keys(timing, [MODEL, MODE], {(m, r.main_mode) for m in models}, "timing")

    if SENSITIVITY_KEY in tables:
        sens = tables[SENSITIVITY_KEY]
        cols = [f"{n}_{s}" for n in names for s in ("main", "refit", "delta")]
        _require_columns(sens, [MODEL, SPLIT_COLUMN, *cols], SENSITIVITY_KEY)
        sens_keys = {(m, s) for m in e.sensitivity_models for s in e.splits}
        _require_keys(sens, [MODEL, SPLIT_COLUMN], sens_keys, SENSITIVITY_KEY)
        refit_keys = {(r.sensitivity_mode, m, s) for m in e.sensitivity_models for s in e.splits}
        _require_keys(overall[overall[MODE] == r.sensitivity_mode],
                      [MODE, MODEL, SPLIT_COLUMN], refit_keys, "overall (refit)")
        _require_keys(timing[timing[MODE] == r.sensitivity_mode], [MODEL, MODE],
                      {(m, r.sensitivity_mode) for m in e.sensitivity_models}, "timing (refit)")


def validate_metadata(
    training: dict[str, Any], split_summary: dict[str, Any], cfg: Config
) -> None:
    """The JSON summaries describe the same setup as the config."""
    for name in cfg.evaluation.main_models:
        if name not in training.get("models", {}):
            raise ReportInputError(f"training_summary: no entry for model {name!r}.")
    for split in ("train", *cfg.evaluation.splits):
        if split not in split_summary.get("splits", {}):
            raise ReportInputError(f"split_summary: no entry for split {split!r}.")
    if training.get("seed") != cfg.project.seed:
        raise ReportInputError("training_summary: seed differs from project.seed.")
    if split_summary.get("last_actual_quarter") != cfg.project.last_actual_quarter:
        raise ReportInputError("split_summary: last_actual_quarter differs from the config.")


def count_arima_orders(orders: dict[str, dict[str, Any]]) -> ArimaOrderCounts:
    """Reduce the per-series ARIMA orders to counts; the series IDs are dropped."""
    records = list(orders.values())
    if not records:
        raise ReportInputError("arima_orders: no series.")
    return ArimaOrderCounts(
        n_series=len(records),
        d=dict(sorted(Counter(int(r["order"][1]) for r in records).items())),
        seasonal_d=dict(sorted(Counter(int(r["seasonal_order"][1]) for r in records).items())),
        n_seasonal_terms=sum(
            1 for r in records if int(r["seasonal_order"][0]) or int(r["seasonal_order"][2])
        ),
        n_converged=sum(1 for r in records if r["converged"]),
        n_fallback=sum(1 for r in records if r["fallback"]),
    )
