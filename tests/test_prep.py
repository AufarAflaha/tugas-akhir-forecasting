"""Tests for Phase 3 data preparation. All data here is synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from typer.testing import CliRunner

from ta_forecasting.cli import app
from ta_forecasting.config import Config, set_global_seed
from ta_forecasting.data.prepare import (
    SCALER_MEAN,
    SCALER_STD,
    SPLIT_COLUMN,
    TEST,
    TIME_IDX,
    TRAIN,
    VAL,
    add_features,
    apply_scaler,
    build_panel,
    build_timeseries_datasets,
    fit_scaler,
    load_scaler,
    run_prep,
    save_scaler,
    select_targets,
    split_bounds,
    split_lengths,
    split_temporal,
)

N_QUARTERS = 40
FIRST_DATE = "2010-03-31"
LAST_QUARTER = "2019Q4"  # 40 quarters starting 2010Q1
TARGET_IDS = ["A", "B", "C"]
OTHER_ID = "X"
OTHER_TYPE = "Exchange Rate"
SEED = 42
# pytorch-forecasting's TorchNormalizer stores scale = std + float16 machine epsilon.
PF_SCALE_EPS = float(np.finfo(np.float16).eps)


@pytest.fixture
def cfg(config: Config) -> Config:
    """Project config with ``last_actual_quarter`` matching the synthetic data."""
    out = config.model_copy(deep=True)
    out.project.last_actual_quarter = LAST_QUARTER
    return out


@pytest.fixture
def raw_data(cfg: Config) -> pd.DataFrame:
    """Synthetic long-format table: three Cost Index series and one unrelated series."""
    c = cfg.prep.columns
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(FIRST_DATE, periods=N_QUARTERS, freq="QE-DEC")
    frames = []
    for sid in TARGET_IDS:
        values = 100.0 + np.cumsum(rng.normal(size=N_QUARTERS))
        frames.append(pd.DataFrame({c.date: dates, c.series_id: sid, c.value: values}))
    huge = pd.DataFrame({c.date: dates, c.series_id: OTHER_ID, c.value: 1e15})
    data = pd.concat([*frames, huge], ignore_index=True)
    data[c.series_type] = np.where(data[c.series_id] == OTHER_ID, OTHER_TYPE, "Cost Index")
    return data.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


@pytest.fixture
def lookup(cfg: Config) -> pd.DataFrame:
    """Synthetic metadata with one deliberately duplicated, poorer row for series B."""
    c = cfg.prep.columns
    rows = [
        {c.lookup_id: "A", c.concept: "Alpha", c.region: "R1", c.series_type: "Cost Index"},
        {c.lookup_id: "B", c.concept: "Beta", c.region: "R2", c.series_type: "Cost Index"},
        {c.lookup_id: "B", c.concept: "Zeta", c.region: None, c.series_type: "Cost Index"},
        {c.lookup_id: "C", c.concept: "Gamma", c.region: "R1", c.series_type: "Cost Index"},
        {c.lookup_id: OTHER_ID, c.concept: "Fx", c.region: "R3", c.series_type: OTHER_TYPE},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def panel(raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return build_panel(raw_data, lookup, cfg)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


class TestSelectTargets:
    def test_keeps_only_target_series_type(
        self, raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config
    ) -> None:
        out = select_targets(raw_data, lookup, cfg)
        assert set(out[cfg.prep.columns.series_id]) == set(TARGET_IDS)
        assert len(out) == len(TARGET_IDS) * N_QUARTERS

    def test_rejects_quarters_after_last_actual(
        self, raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config
    ) -> None:
        c = cfg.prep.columns
        projection = raw_data[raw_data[c.series_id] == "A"].iloc[[0]].copy()
        projection[c.date] = pd.Timestamp("2020-03-31")  # 2020Q1 > 2019Q4
        with pytest.raises(ValueError, match="last_actual_quarter"):
            select_targets(pd.concat([raw_data, projection]), lookup, cfg)

    def test_duplicate_lookup_resolved_deterministically(
        self, raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config
    ) -> None:
        c = cfg.prep.columns
        first = select_targets(raw_data, lookup, cfg)
        second = select_targets(raw_data, lookup.iloc[::-1].reset_index(drop=True), cfg)
        pd.testing.assert_frame_equal(first, second)
        concept_b = first.loc[first[c.series_id] == "B", c.concept].unique()
        assert list(concept_b) == ["Beta"]  # the row with complete metadata wins

    def test_missing_metadata_is_rejected(
        self, raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config
    ) -> None:
        c = cfg.prep.columns
        with pytest.raises(ValueError, match="metadata"):
            select_targets(raw_data, lookup[lookup[c.lookup_id] != "A"], cfg)

    def test_unknown_series_type_is_rejected(
        self, raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config
    ) -> None:
        other = cfg.model_copy(deep=True)
        other.prep.target_series_type = "Does Not Exist"
        with pytest.raises(ValueError, match="No series"):
            select_targets(raw_data, lookup, other)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_time_idx_is_contiguous_per_series(self, panel: pd.DataFrame, cfg: Config) -> None:
        for _, g in panel.groupby(cfg.prep.columns.series_id):
            idx = g.sort_values(TIME_IDX)[TIME_IDX].to_numpy()
            assert idx[0] == 0
            assert (np.diff(idx) == 1).all()
            assert len(idx) == N_QUARTERS

    def test_cyclic_features(self, panel: pd.DataFrame) -> None:
        q = panel["quarter"].to_numpy()
        assert set(q) == {1, 2, 3, 4}
        assert np.allclose(panel["quarter_sin"] ** 2 + panel["quarter_cos"] ** 2, 1.0)
        expected = {1: (1.0, 0.0), 2: (0.0, -1.0), 3: (-1.0, 0.0), 4: (0.0, 1.0)}
        for quarter, (sin, cos) in expected.items():
            rows = panel[panel["quarter"] == quarter]
            assert np.allclose(rows["quarter_sin"], sin, atol=1e-9)
            assert np.allclose(rows["quarter_cos"], cos, atol=1e-9)

    def test_gap_in_calendar_is_rejected(
        self, raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config
    ) -> None:
        c = cfg.prep.columns
        dates = sorted(raw_data[c.date].unique())
        selected = select_targets(raw_data, lookup, cfg)
        gapped = selected[selected[c.date] != dates[10]]  # drop one quarter for all series
        with pytest.raises(ValueError, match="consecutive"):
            add_features(gapped, cfg)

    def test_unbalanced_panel_is_rejected(
        self, raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config
    ) -> None:
        c = cfg.prep.columns
        selected = select_targets(raw_data, lookup, cfg)
        drop = selected[(selected[c.series_id] == "A")].index[-1]
        with pytest.raises(ValueError, match="do not cover"):
            add_features(selected.drop(index=drop), cfg)

    def test_no_quarter_after_last_actual_in_panel(self, panel: pd.DataFrame) -> None:
        assert pd.Period(panel["quarter_label"].max(), freq="Q") <= pd.Period(LAST_QUARTER, "Q")
        assert panel["quarter_label"].max() == LAST_QUARTER


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_lengths_for_105_quarters(self) -> None:
        assert split_lengths(105, 0.7, 0.15) == (74, 15, 16)

    def test_lengths_use_exact_arithmetic(self) -> None:
        # In floating point 0.57 * 100 = 56.99999999999999, so floor() would give 56.
        assert 0.57 * 100 < 57
        assert split_lengths(100, 0.2, 0.57) == (20, 57, 23)

    @pytest.mark.parametrize(("train", "val"), [(0.0, 0.5), (0.6, 0.0), (0.7, 0.3)])
    def test_invalid_ratios_are_rejected(self, train: float, val: float) -> None:
        with pytest.raises(ValueError, match="ratio"):
            split_lengths(100, train, val)

    def test_split_sizes_and_order(self, panel: pd.DataFrame, cfg: Config) -> None:
        v = cfg.validation
        n_train, n_val, n_test = split_lengths(N_QUARTERS, v.train_ratio, v.val_ratio)
        bounds = split_bounds(panel)
        assert bounds[TRAIN] == (0, n_train - 1)
        assert bounds[VAL] == (n_train, n_train + n_val - 1)
        assert bounds[TEST] == (n_train + n_val, N_QUARTERS - 1)
        assert n_train + n_val + n_test == N_QUARTERS

    def test_no_temporal_leakage_between_splits(self, panel: pd.DataFrame, cfg: Config) -> None:
        t = panel[TIME_IDX]
        s = panel[SPLIT_COLUMN]
        assert t[s == TRAIN].max() < t[s == VAL].min()
        assert t[s == VAL].max() < t[s == TEST].min()
        # every time step belongs to exactly one split
        assert panel.groupby(TIME_IDX)[SPLIT_COLUMN].nunique().eq(1).all()
        # every series has the same quarters in each split
        sizes = panel.groupby([cfg.prep.columns.series_id, SPLIT_COLUMN]).size().unstack()
        assert (sizes.nunique() == 1).all()

    def test_split_does_not_depend_on_row_order(self, panel: pd.DataFrame, cfg: Config) -> None:
        shuffled = panel.drop(columns=SPLIT_COLUMN).sample(frac=1.0, random_state=SEED)
        again = split_temporal(shuffled, cfg).sort_values(["ID", TIME_IDX]).reset_index(drop=True)
        expected = panel.sort_values(["ID", TIME_IDX]).reset_index(drop=True)
        assert (again[SPLIT_COLUMN] == expected[SPLIT_COLUMN]).all()


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


class TestScaler:
    def test_matches_train_statistics(self, panel: pd.DataFrame, cfg: Config) -> None:
        scaler = fit_scaler(panel, cfg)
        train = panel[panel[SPLIT_COLUMN] == TRAIN]
        target = cfg.prep.target_column
        for sid, g in train.groupby("ID"):
            assert scaler.loc[sid, SCALER_MEAN] == pytest.approx(g[target].mean())
            assert scaler.loc[sid, SCALER_STD] == pytest.approx(g[target].std(ddof=1))

    def test_ignores_validation_and_test_rows(self, panel: pd.DataFrame, cfg: Config) -> None:
        baseline = fit_scaler(panel, cfg)
        tampered = panel.copy()
        later = tampered[SPLIT_COLUMN] != TRAIN
        tampered.loc[later, cfg.prep.target_column] *= 1000.0
        pd.testing.assert_frame_equal(baseline, fit_scaler(tampered, cfg))

    def test_reacts_to_train_rows(self, panel: pd.DataFrame, cfg: Config) -> None:
        baseline = fit_scaler(panel, cfg)
        tampered = panel.copy()
        tampered.loc[tampered[SPLIT_COLUMN] == TRAIN, cfg.prep.target_column] *= 2.0
        assert not baseline.equals(fit_scaler(tampered, cfg))

    def test_standardises_train_rows(self, panel: pd.DataFrame, cfg: Config) -> None:
        scaler = fit_scaler(panel, cfg)
        train = panel[panel[SPLIT_COLUMN] == TRAIN].copy()
        train["z"] = apply_scaler(train, scaler, cfg)
        assert np.allclose(train.groupby("ID")["z"].mean(), 0.0, atol=1e-9)
        assert np.allclose(train.groupby("ID")["z"].std(ddof=1), 1.0, atol=1e-9)

    def test_save_and_load_roundtrip(
        self, panel: pd.DataFrame, cfg: Config, tmp_path: Path
    ) -> None:
        scaler = fit_scaler(panel, cfg)
        path = tmp_path / "scaler.json"
        save_scaler(scaler, path, cfg)
        loaded = load_scaler(path, cfg)
        pd.testing.assert_frame_equal(scaler, loaded, check_names=False)
        assert json.loads(path.read_text())["fitted_on"] == TRAIN


# ---------------------------------------------------------------------------
# TimeSeriesDataSet
# ---------------------------------------------------------------------------


def _collect(dataset, key: str) -> torch.Tensor:  # noqa: ANN001
    loader = dataset.to_dataloader(train=False, batch_size=1024, num_workers=0)
    return torch.cat([x[key] for x, _ in loader])


class TestTimeSeriesDataSets:
    def test_window_counts_and_bounds(self, panel: pd.DataFrame, cfg: Config) -> None:
        p = cfg.prep
        train, val, test = build_timeseries_datasets(panel, cfg)
        bounds = split_bounds(panel)
        n_series = panel["ID"].nunique()

        for ds, (first, last) in ((val, bounds[VAL]), (test, bounds[TEST])):
            expected = n_series * (last - first + 1 - p.max_prediction_length + 1)
            assert len(ds) == expected
            decoder = _collect(ds, "decoder_time_idx")
            assert decoder.shape[1] == p.max_prediction_length
            assert int(decoder.min()) >= first  # targets only inside own split
            assert int(decoder.max()) <= last
            assert (_collect(ds, "encoder_lengths") == p.max_encoder_length).all()

        train_decoder = _collect(train, "decoder_time_idx")
        assert int(train_decoder.max()) <= bounds[TRAIN][1]  # no val/test target in training
        # each training target window appears exactly once (no shortened-encoder copies)
        assert (_collect(train, "encoder_lengths") == p.max_encoder_length).all()
        n_train = bounds[TRAIN][1] - bounds[TRAIN][0] + 1
        expected_train = n_series * (
            n_train - p.max_encoder_length - p.max_prediction_length + 1
        )
        assert len(train) == expected_train

    def test_encoder_context_reaches_into_earlier_split(
        self, panel: pd.DataFrame, cfg: Config
    ) -> None:
        _, val, _ = build_timeseries_datasets(panel, cfg)
        first_val = split_bounds(panel)[VAL][0]
        decoder_start = _collect(val, "decoder_time_idx")[:, 0]
        encoder_start = decoder_start - _collect(val, "encoder_lengths")
        assert int(encoder_start.min()) < first_val  # past information is allowed as context

    def test_static_and_dynamic_variables(self, panel: pd.DataFrame, cfg: Config) -> None:
        train, _, _ = build_timeseries_datasets(panel, cfg)
        c = cfg.prep.columns
        assert train.static_categoricals == [c.series_id, c.concept, c.region, c.series_type]
        assert train.time_varying_known_reals == [TIME_IDX, "quarter_sin", "quarter_cos"]
        assert train.time_varying_unknown_reals == [cfg.prep.target_column]
        assert train.max_encoder_length == cfg.prep.max_encoder_length
        assert train.max_prediction_length == cfg.prep.max_prediction_length

    def test_target_normalizer_is_fitted_on_train_only(
        self, panel: pd.DataFrame, cfg: Config
    ) -> None:
        train, _, _ = build_timeseries_datasets(panel, cfg)
        scaler = fit_scaler(panel, cfg)
        norm = train.target_normalizer.norm_
        centers = np.sort(np.asarray(norm["center"], dtype=float))
        scales = np.sort(np.asarray(norm["scale"], dtype=float)) - PF_SCALE_EPS
        assert np.allclose(centers, np.sort(scaler[SCALER_MEAN].to_numpy()), atol=1e-6)
        assert np.allclose(scales, np.sort(scaler[SCALER_STD].to_numpy()), atol=1e-6)

        tampered = panel.copy()
        tampered.loc[tampered[SPLIT_COLUMN] != TRAIN, cfg.prep.target_column] *= 1000.0
        train_2, _, _ = build_timeseries_datasets(tampered, cfg)
        assert np.allclose(
            np.sort(np.asarray(train_2.target_normalizer.norm_["center"], dtype=float)), centers
        )

    def test_deterministic_with_seed(self, panel: pd.DataFrame, cfg: Config) -> None:
        batches = []
        for _ in range(2):
            set_global_seed(cfg.project.seed)
            train, val, _ = build_timeseries_datasets(panel, cfg)
            loader = train.to_dataloader(train=True, batch_size=8, num_workers=0)
            x, _ = next(iter(loader))
            batches.append((x["encoder_target"], _collect(val, "decoder_target")))
        assert torch.equal(batches[0][0], batches[1][0])
        assert torch.equal(batches[0][1], batches[1][1])


# ---------------------------------------------------------------------------
# Pipeline and CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(
    raw_data: pd.DataFrame, lookup: pd.DataFrame, cfg: Config, tmp_path: Path
) -> tuple[Config, Path]:
    """Write synthetic interim parquet files and return a config pointing at tmp_path."""
    interim = tmp_path / "interim"
    interim.mkdir()
    raw_data.to_parquet(interim / f"{cfg.prep.source_dataset}.parquet", index=False)
    lookup.to_parquet(interim / f"{cfg.prep.lookup_dataset}.parquet", index=False)

    local = cfg.model_copy(deep=True)
    local.data.interim_dir = str(interim)
    local.data.processed_dir = str(tmp_path / "processed")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(local.model_dump()), encoding="utf-8")
    return local, config_path


class TestPipeline:
    def test_run_prep_writes_artefacts(self, workspace: tuple[Config, Path]) -> None:
        local, _ = workspace
        result = run_prep(local)
        processed = Path(local.data.processed_dir)
        assert (processed / local.prep.panel_file).exists()
        assert (processed / local.prep.scaler_file).exists()
        assert (processed / local.prep.summary_file).exists()

        panel = pd.read_parquet(result.panel_path)
        assert len(panel) == len(TARGET_IDS) * N_QUARTERS
        assert {SPLIT_COLUMN, TIME_IDX, "quarter_sin", "quarter_cos", "target"} <= set(panel)

        summary = json.loads(result.summary_path.read_text())
        assert summary["n_series"] == len(TARGET_IDS)
        assert summary["splits"]["train"]["start_quarter"] == "2010Q1"
        assert summary["splits"]["test"]["end_quarter"] == LAST_QUARTER
        assert sum(s["n_rows"] for s in summary["splits"].values()) == summary["n_rows"]

    def test_summary_contains_no_row_level_data(self, workspace: tuple[Config, Path]) -> None:
        local, _ = workspace
        summary = run_prep(local).summary
        allowed_top = {
            "target_series_type", "last_actual_quarter", "n_series", "n_quarters",
            "n_rows", "splits",
        }  # fmt: skip
        assert set(summary) == allowed_top
        for info in summary["splits"].values():
            assert set(info) == {
                "start_quarter", "end_quarter", "start_time_idx", "end_time_idx",
                "n_quarters", "n_rows",
            }  # fmt: skip

    def test_cli_prep(self, workspace: tuple[Config, Path]) -> None:
        local, config_path = workspace
        result = CliRunner().invoke(app, ["prep", "--config", str(config_path)])
        assert result.exit_code == 0, result.output
        assert "TimeSeriesDataSet windows" in result.output
        assert (Path(local.data.processed_dir) / local.prep.panel_file).exists()
