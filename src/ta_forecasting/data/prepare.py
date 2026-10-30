"""Phase 3 — data preparation.

Turns the interim ``Data`` and ``ID_lookup`` tables into a model-ready panel:

1. ``select_targets``  keeps one series type and attaches static metadata.
2. ``add_features``    adds ``time_idx`` and calendar features.
3. ``split_temporal``  assigns every quarter to train / val / test (no shuffling).
4. ``fit_scaler``      learns per-series mean/std from train rows only.
5. ``build_timeseries_datasets`` builds the pytorch-forecasting datasets for the TFT.

Only quarters up to ``project.last_actual_quarter`` are ever accepted. S&P projections
beyond that quarter must never reach fitting, validation or testing.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ta_forecasting.config import Config

if TYPE_CHECKING:
    from pytorch_forecasting import TimeSeriesDataSet

logger = logging.getLogger(__name__)

SPLIT_COLUMN = "split"
TRAIN = "train"
VAL = "val"
TEST = "test"
SPLIT_ORDER: tuple[str, ...] = (TRAIN, VAL, TEST)

TIME_IDX = "time_idx"
QUARTER = "quarter"
QUARTER_SIN = "quarter_sin"
QUARTER_COS = "quarter_cos"
QUARTER_LABEL = "quarter_label"

SCALER_MEAN = "mean"
SCALER_STD = "std"


@dataclass(frozen=True)
class PrepResult:
    """Outcome of :func:`run_prep`. Holds paths and aggregate counts only."""

    panel_path: Path
    scaler_path: Path
    summary_path: Path
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# 1. Target selection
# ---------------------------------------------------------------------------


def _dedupe_lookup(lookup: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep exactly one metadata row per series ID, chosen deterministically.

    Preference: the row with the most non-null metadata fields, then alphabetical order of
    concept and region. Row order of the input does not influence the result.
    """
    c = cfg.prep.columns
    meta_cols = [c.concept, c.region, c.series_type]
    ranked = lookup.assign(_n_meta=lookup[meta_cols].notna().sum(axis=1))
    ranked = ranked.sort_values(
        [c.lookup_id, "_n_meta", c.concept, c.region, c.series_type],
        ascending=[True, False, True, True, True],
        na_position="last",
        kind="mergesort",
    )
    return ranked.drop_duplicates(subset=c.lookup_id, keep="first").drop(columns="_n_meta")


def select_targets(data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Select target series and attach static metadata.

    Parameters
    ----------
    data : pd.DataFrame
        Interim long-format time series (date, series id, value, series type).
    lookup : pd.DataFrame
        Interim series metadata (series id, concept, region, series type).
    cfg : Config
        Project configuration.

    Returns
    -------
    pd.DataFrame
        Columns: series id, date, quarter label, target, concept, region, series type.
        Sorted by series id and date.

    Raises
    ------
    ValueError
        If the selection is empty, contains a quarter after ``last_actual_quarter``,
        has missing static metadata, or has a non-finite target.
    """
    c = cfg.prep.columns
    target_col = cfg.prep.target_column

    selected = data.loc[
        data[c.series_type] == cfg.prep.target_series_type, [c.series_id, c.date, c.value]
    ].copy()
    if selected.empty:
        raise ValueError(f"No series of type {cfg.prep.target_series_type!r} in the data.")

    selected[c.date] = pd.to_datetime(selected[c.date])
    periods = selected[c.date].dt.to_period("Q")
    last_actual = pd.Period(cfg.project.last_actual_quarter, freq="Q")
    if (periods > last_actual).any():
        raise ValueError(
            f"Found quarters after last_actual_quarter={cfg.project.last_actual_quarter}. "
            "S&P projections must not enter the pipeline."
        )
    selected[QUARTER_LABEL] = periods.astype(str)

    meta = _dedupe_lookup(lookup, cfg)[
        [c.lookup_id, c.concept, c.region, c.series_type]
    ].rename(columns={c.lookup_id: c.series_id})
    selected = selected.merge(meta, on=c.series_id, how="left", validate="many_to_one")

    missing_meta = selected[[c.concept, c.region, c.series_type]].isna().any(axis=1)
    if missing_meta.any():
        n_ids = selected.loc[missing_meta, c.series_id].nunique()
        raise ValueError(f"{n_ids} target series lack Concept/Region/Series type metadata.")

    selected = selected.rename(columns={c.value: target_col})
    if not np.isfinite(selected[target_col].to_numpy(dtype=float)).all():
        raise ValueError("Target contains NaN or infinite values.")

    ordered = [
        c.series_id,
        c.date,
        QUARTER_LABEL,
        target_col,
        c.concept,
        c.region,
        c.series_type,
    ]
    return selected[ordered].sort_values([c.series_id, c.date]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add ``time_idx`` and calendar features.

    ``time_idx`` is a global integer (0, 1, 2, ...) with one step per quarter, identical for
    every series. The panel must be balanced and free of calendar gaps.

    Raises
    ------
    ValueError
        If quarters are not consecutive or a series misses some quarters.
    """
    c = cfg.prep.columns
    out = df.copy()

    dates = np.sort(out[c.date].unique())
    ordinals = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="Q").asi8
    if len(ordinals) > 1 and not (np.diff(ordinals) == 1).all():
        raise ValueError("Quarters are not consecutive; time_idx would be misleading.")

    counts = out.groupby(c.series_id)[c.date].nunique()
    if (counts != len(dates)).any():
        raise ValueError(f"{int((counts != len(dates)).sum())} series do not cover all quarters.")
    if out.duplicated([c.series_id, c.date]).any():
        raise ValueError("Duplicate (series, date) rows found.")

    out[TIME_IDX] = pd.Index(dates).get_indexer(pd.DatetimeIndex(out[c.date])).astype(int)
    out[QUARTER] = out[c.date].dt.quarter.astype(int)
    angle = 2.0 * np.pi * out[QUARTER] / cfg.prep.quarter_period
    out[QUARTER_SIN] = np.sin(angle)
    out[QUARTER_COS] = np.cos(angle)
    return out


# ---------------------------------------------------------------------------
# 3. Temporal split
# ---------------------------------------------------------------------------


def split_lengths(n_quarters: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """Return (train, val, test) lengths in quarters.

    train = ceil(train_ratio * n), val = floor(val_ratio * n), test = the remainder.
    Ratios are converted through their decimal string and evaluated with exact fractions,
    so ``0.7 * 105`` is exactly 73.5 and not 73.49999.
    """
    train_frac = Fraction(str(train_ratio))
    val_frac = Fraction(str(val_ratio))
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Need train_ratio > 0, val_ratio > 0 and train_ratio + val_ratio < 1.")

    n_train = math.ceil(train_frac * n_quarters)
    n_val = math.floor(val_frac * n_quarters)
    n_test = n_quarters - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Cannot split {n_quarters} quarters into non-empty train/val/test.")
    return n_train, n_val, n_test


def split_temporal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign each row to train, val or test by time only (no shuffling)."""
    v = cfg.validation
    ratio_sum = Fraction(str(v.train_ratio)) + Fraction(str(v.val_ratio))
    if ratio_sum + Fraction(str(v.test_ratio)) != 1:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.")

    n_quarters = int(df[TIME_IDX].nunique())
    n_train, n_val, _ = split_lengths(n_quarters, v.train_ratio, v.val_ratio)

    out = df.copy()
    out[SPLIT_COLUMN] = np.select(
        [out[TIME_IDX] < n_train, out[TIME_IDX] < n_train + n_val],
        [TRAIN, VAL],
        default=TEST,
    )
    return out


def split_bounds(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Return (first, last) ``time_idx`` of each split."""
    grouped = df.groupby(SPLIT_COLUMN)[TIME_IDX].agg(["min", "max"])
    return {
        name: (int(grouped.loc[name, "min"]), int(grouped.loc[name, "max"]))
        for name in SPLIT_ORDER
    }


def summarize_splits(df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Aggregate description of the split: quarter bounds and counts. No data rows."""
    c = cfg.prep.columns
    splits: dict[str, Any] = {}
    for name in SPLIT_ORDER:
        part = df[df[SPLIT_COLUMN] == name]
        splits[name] = {
            "start_quarter": str(part.loc[part[TIME_IDX].idxmin(), QUARTER_LABEL]),
            "end_quarter": str(part.loc[part[TIME_IDX].idxmax(), QUARTER_LABEL]),
            "start_time_idx": int(part[TIME_IDX].min()),
            "end_time_idx": int(part[TIME_IDX].max()),
            "n_quarters": int(part[TIME_IDX].nunique()),
            "n_rows": int(len(part)),
        }
    return {
        "target_series_type": cfg.prep.target_series_type,
        "last_actual_quarter": cfg.project.last_actual_quarter,
        "n_series": int(df[c.series_id].nunique()),
        "n_quarters": int(df[TIME_IDX].nunique()),
        "n_rows": int(len(df)),
        "splits": splits,
    }


# ---------------------------------------------------------------------------
# 4. Scaler (train rows only)
# ---------------------------------------------------------------------------


def fit_scaler(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fit a per-series standard scaler on **train rows only**.

    Returns
    -------
    pd.DataFrame
        Indexed by series id with columns ``mean`` and ``std`` (sample std, ddof=1).
    """
    c = cfg.prep.columns
    train = df[df[SPLIT_COLUMN] == TRAIN]
    stats = train.groupby(c.series_id)[cfg.prep.target_column].agg(
        **{SCALER_MEAN: "mean", SCALER_STD: "std"}
    )
    if not np.isfinite(stats.to_numpy()).all() or (stats[SCALER_STD] <= 0).any():
        raise ValueError("Scaler statistics are not finite and positive for every series.")
    return stats


def apply_scaler(df: pd.DataFrame, scaler: pd.DataFrame, cfg: Config) -> pd.Series:
    """Standardise the target of ``df`` with statistics from ``scaler``."""
    c = cfg.prep.columns
    mean = df[c.series_id].map(scaler[SCALER_MEAN])
    std = df[c.series_id].map(scaler[SCALER_STD])
    if mean.isna().any() or std.isna().any():
        raise ValueError("Scaler has no statistics for some series.")
    return (df[cfg.prep.target_column] - mean) / std


def save_scaler(scaler: pd.DataFrame, path: Path, cfg: Config) -> None:
    """Write the scaler as JSON (deterministic key order)."""
    payload = {
        "method": cfg.prep.normalizer_method,
        "fitted_on": TRAIN,
        "stats": {
            str(sid): {SCALER_MEAN: float(row[SCALER_MEAN]), SCALER_STD: float(row[SCALER_STD])}
            for sid, row in scaler.iterrows()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_scaler(path: Path, cfg: Config) -> pd.DataFrame:
    """Read a scaler written by :func:`save_scaler`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame.from_dict(payload["stats"], orient="index")
    frame.index.name = cfg.prep.columns.series_id
    return frame[[SCALER_MEAN, SCALER_STD]]


# ---------------------------------------------------------------------------
# 5. TimeSeriesDataSet for the TFT
# ---------------------------------------------------------------------------


def build_timeseries_datasets(
    panel: pd.DataFrame, cfg: Config
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """Build the train, validation and test ``TimeSeriesDataSet`` objects.

    * The training set (and its per-series ``GroupNormalizer``) sees train rows only.
    * Validation and test sets reuse the training encoders and normalizer via
      ``from_dataset``. Their encoder windows may reach back into earlier splits (that is
      past information), but ``min_prediction_idx`` guarantees every predicted target lies
      inside the split itself, and the data is cut at the end of that split.
    * For validation and test ``min_encoder_length`` is raised to ``max_encoder_length``.
      Otherwise pytorch-forecasting appends shortened-encoder copies of the final window,
      which would count the last targets several times in the evaluation.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    c = cfg.prep.columns
    p = cfg.prep
    bounds = split_bounds(panel)

    frame = panel.copy()
    static_categoricals = [c.series_id, c.concept, c.region, c.series_type]
    for col in static_categoricals:
        frame[col] = frame[col].astype(str)

    train_df = frame[frame[SPLIT_COLUMN] == TRAIN]
    training = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_IDX,
        target=p.target_column,
        group_ids=[c.series_id],
        min_encoder_length=p.min_encoder_length,
        max_encoder_length=p.max_encoder_length,
        min_prediction_length=p.max_prediction_length,
        max_prediction_length=p.max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=[TIME_IDX, QUARTER_SIN, QUARTER_COS],
        time_varying_unknown_reals=[p.target_column],
        target_normalizer=GroupNormalizer(groups=[c.series_id], method=p.normalizer_method),
    )

    val_start, val_end = bounds[VAL]
    validation = TimeSeriesDataSet.from_dataset(
        training,
        frame[frame[TIME_IDX] <= val_end],
        min_prediction_idx=val_start,
        min_encoder_length=p.max_encoder_length,
        stop_randomization=True,
    )

    test_start, test_end = bounds[TEST]
    test = TimeSeriesDataSet.from_dataset(
        training,
        frame[frame[TIME_IDX] <= test_end],
        min_prediction_idx=test_start,
        min_encoder_length=p.max_encoder_length,
        stop_randomization=True,
    )
    return training, validation, test


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_panel(data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run target selection, feature engineering and the temporal split."""
    return split_temporal(add_features(select_targets(data, lookup, cfg), cfg), cfg)


def run_prep(cfg: Config) -> PrepResult:
    """Read interim data, build the panel and write all artefacts to ``processed_dir``."""
    interim = Path(cfg.data.interim_dir)
    processed = Path(cfg.data.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(interim / f"{cfg.prep.source_dataset}.parquet")
    lookup = pd.read_parquet(interim / f"{cfg.prep.lookup_dataset}.parquet")

    panel = build_panel(data, lookup, cfg)
    scaler = fit_scaler(panel, cfg)
    summary = summarize_splits(panel, cfg)

    panel_path = processed / cfg.prep.panel_file
    scaler_path = processed / cfg.prep.scaler_file
    summary_path = processed / cfg.prep.summary_file

    panel.to_parquet(panel_path, engine="pyarrow", index=False)
    save_scaler(scaler, scaler_path, cfg)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("Prepared panel: %d rows, %d series", summary["n_rows"], summary["n_series"])
    return PrepResult(panel_path, scaler_path, summary_path, summary)
