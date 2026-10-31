"""Tests for Phase 4 models. All data is synthetic and the config is shrunk for speed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ta_forecasting.cli import app
from ta_forecasting.config import Config, load_config
from ta_forecasting.data.prepare import (
    SPLIT_COLUMN,
    TEST,
    TIME_IDX,
    TRAIN,
    VAL,
    build_evaluation_dataset,
    build_panel,
    build_training_dataset,
    split_bounds,
)
from ta_forecasting.models.arima import (
    ARIMAForecaster,
    candidate_specs,
    differencing_orders,
    seasonal_strength,
)
from ta_forecasting.models.base import ORIGIN, PREDICTION, STEP, Forecaster
from ta_forecasting.models.ets import ETSForecaster
from ta_forecasting.models.statistical import FitAttempt
from ta_forecasting.models.tft import TFTForecaster, build_predict_frame, resolve_accelerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_QUARTERS = 40
FIRST_DATE = "2010-03-31"
LAST_QUARTER = "2019Q4"
SERIES = {"A": ("Alpha", "R1"), "B": ("Beta", "R2"), "C": ("Gamma", "R1")}
HORIZON = 4
TAMPER_FACTOR = 1000.0


def mini_config(tmp: Path) -> Config:
    """Project config shrunk for tests: tiny grid, tiny TFT, 2 epochs, paths in ``tmp``."""
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml").model_copy(deep=True)
    cfg.project.last_actual_quarter = LAST_QUARTER
    cfg.prep.max_encoder_length = 8
    cfg.prep.min_encoder_length = 8
    cfg.prep.max_prediction_length = HORIZON
    cfg.arima.max_p = cfg.arima.max_q = 1
    cfg.arima.n_jobs = 1
    cfg.tft.hidden_size = 8
    cfg.tft.attention_head_size = 1
    cfg.tft.hidden_continuous_size = 4
    cfg.tft.max_epochs = 2
    cfg.tft.batch_size = 16
    cfg.tft.num_threads = 1
    cfg.data.processed_dir = str(tmp / "processed")
    cfg.output.models_dir = str(tmp / "models")
    cfg.output.logs_dir = str(tmp / "logs")
    return cfg


def synthetic_panel(cfg: Config) -> pd.DataFrame:
    """Three trending, seasonal series built through the Phase 3 pipeline."""
    c = cfg.prep.columns
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(FIRST_DATE, periods=N_QUARTERS, freq="QE-DEC")
    t = np.arange(N_QUARTERS)
    frames = []
    for i, sid in enumerate(SERIES):
        season = (2.0 + i) * np.sin(2 * np.pi * t / 4)
        values = 100.0 + (0.4 + 0.1 * i) * t + season + np.cumsum(rng.normal(0, 0.5, N_QUARTERS))
        frames.append(pd.DataFrame({c.date: dates, c.series_id: sid, c.value: values}))
    data = pd.concat(frames, ignore_index=True)
    data[c.series_type] = cfg.prep.target_series_type
    lookup = pd.DataFrame(
        [
            {c.lookup_id: sid, c.concept: concept, c.region: region,
             c.series_type: cfg.prep.target_series_type}
            for sid, (concept, region) in SERIES.items()
        ]
    )  # fmt: skip
    return build_panel(data, lookup, cfg)


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, pd.DataFrame]:
    cfg = mini_config(tmp_path_factory.mktemp("mini"))
    return cfg, synthetic_panel(cfg)


def _split(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    return panel[panel[SPLIT_COLUMN] == name].reset_index(drop=True)


def _fit(name: str, cfg: Config, panel: pd.DataFrame) -> Forecaster:
    train = _split(panel, TRAIN)
    if name == "ets":
        return ETSForecaster(cfg).fit(train)
    if name == "arima":
        return ARIMAForecaster(cfg).fit(train)
    return TFTForecaster(cfg).fit(train, _split(panel, VAL))


@pytest.fixture(scope="module")
def fitted(mini: tuple[Config, pd.DataFrame]) -> dict[str, Forecaster]:
    cfg, panel = mini
    return {name: _fit(name, cfg, panel) for name in ("ets", "arima", "tft")}


MODEL_NAMES = ["ets", "arima", "tft"]


def _values(frame: pd.DataFrame) -> np.ndarray:
    return frame.sort_values(["ID", STEP])[PREDICTION].to_numpy()


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODEL_NAMES)
class TestInterface:
    def test_predict_shape(
        self, name: str, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        _, panel = mini
        out = fitted[name].predict(HORIZON)
        train_end = int(_split(panel, TRAIN)[TIME_IDX].max())
        assert len(out) == len(SERIES) * HORIZON
        for _, g in out.groupby("ID"):
            assert list(g[STEP]) == list(range(1, HORIZON + 1))
            assert list(g[TIME_IDX]) == list(range(train_end + 1, train_end + 1 + HORIZON))
            assert (g[ORIGIN] == train_end + 1).all()
        assert np.isfinite(out[PREDICTION]).all()

    def test_forecast_shape_at_rolling_origin(
        self, name: str, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        _, panel = mini
        origin = split_bounds(panel)[TEST][0]
        out = fitted[name].forecast(panel[panel[TIME_IDX] < origin], HORIZON)
        assert len(out) == len(SERIES) * HORIZON
        assert set(out["ID"]) == set(SERIES)
        assert (out[ORIGIN] == origin).all()
        assert out[TIME_IDX].min() == origin
        assert out[TIME_IDX].max() == origin + HORIZON - 1
        assert np.isfinite(out[PREDICTION]).all()

    def test_shorter_horizon(self, name: str, fitted: dict[str, Forecaster]) -> None:
        out = fitted[name].predict(2)
        assert len(out) == len(SERIES) * 2
        assert np.allclose(_values(out), _values(fitted[name].predict(HORIZON).query("step <= 2")))

    def test_save_load_identical(
        self,
        name: str,
        fitted: dict[str, Forecaster],
        mini: tuple[Config, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        _, panel = mini
        model = fitted[name]
        path = tmp_path / f"{name}.bin"
        model.save(path)
        loaded = type(model).load(path)
        history = panel[panel[TIME_IDX] < split_bounds(panel)[VAL][1]]
        pd.testing.assert_frame_equal(model.predict(HORIZON), loaded.predict(HORIZON))
        pd.testing.assert_frame_equal(
            model.forecast(history, HORIZON), loaded.forecast(history, HORIZON)
        )
        assert loaded.fit_info == model.fit_info

    def test_forecast_uses_only_history_before_origin(
        self, name: str, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        cfg, panel = mini
        origin = split_bounds(panel)[TEST][0]
        tampered = panel.copy()
        future = tampered[TIME_IDX] >= origin
        tampered.loc[future, cfg.prep.target_column] *= TAMPER_FACTOR

        model = fitted[name]
        clean = model.forecast(panel[panel[TIME_IDX] < origin], HORIZON)
        dirty = model.forecast(tampered[tampered[TIME_IDX] < origin], HORIZON)
        pd.testing.assert_frame_equal(clean, dirty)

        # the last value before the origin does matter, so the history really is used
        nudged = panel[panel[TIME_IDX] < origin].copy()
        nudged.loc[nudged[TIME_IDX] == origin - 1, cfg.prep.target_column] += 10.0
        assert not np.allclose(_values(clean), _values(model.forecast(nudged, HORIZON)))

    def test_rejects_projection_quarters(
        self, name: str, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        _, panel = mini
        projected = panel.copy()
        projected.loc[projected[TIME_IDX] == projected[TIME_IDX].max(), "quarter_label"] = (
            "2020Q1"  # after LAST_QUARTER
        )
        with pytest.raises(ValueError, match="last_actual_quarter"):
            fitted[name].forecast(projected, HORIZON)

    def test_invalid_horizon(self, name: str, fitted: dict[str, Forecaster]) -> None:
        with pytest.raises(ValueError, match="horizon"):
            fitted[name].predict(0)


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ets", "arima"])
class TestStatistical:
    def test_forecast_from_train_equals_predict(
        self, name: str, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        _, panel = mini
        model = fitted[name]
        from_history = model.forecast(_split(panel, TRAIN), HORIZON)
        assert np.allclose(_values(from_history), _values(model.predict(HORIZON)), atol=1e-8)

    def test_deterministic(self, name: str, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, panel = mini
        first = _fit(name, cfg, panel).predict(HORIZON)
        second = _fit(name, cfg, panel).predict(HORIZON)
        pd.testing.assert_frame_equal(first, second)

    def test_refit_mode(
        self, name: str, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        cfg, panel = mini
        model = fitted[name]
        model.cfg = cfg.model_copy(deep=True)
        model.cfg.rolling.refit_statistical = True
        model.forecast_log.clear()
        try:
            origin = split_bounds(panel)[TEST][0]
            out = model.forecast(panel[panel[TIME_IDX] < origin], HORIZON)
        finally:
            model.cfg = cfg
        assert len(out) == len(SERIES) * HORIZON
        assert np.isfinite(out[PREDICTION]).all()
        assert "fixed_parameters" not in model.forecast_log
        assert sum(model.forecast_log.values()) == len(SERIES)

    def test_history_must_start_at_train_start(
        self, name: str, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        _, panel = mini
        with pytest.raises(ValueError, match="training start"):
            fitted[name].forecast(panel[panel[TIME_IDX] >= 1], HORIZON)

    def test_summary_is_aggregate(self, name: str, fitted: dict[str, Forecaster]) -> None:
        info = fitted[name].fit_info
        assert info["n_series"] == len(SERIES)
        assert info["n_fallback"] == 0
        assert info["fit_seconds"] >= 0
        json.dumps(info)  # serialisable, no arrays of data


class TestETS:
    def test_fallback_only_when_fit_fails(
        self, mini: tuple[Config, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, panel = mini
        original = ETSForecaster._fit_spec

        def fail_seasonal(self: ETSForecaster, values: np.ndarray, spec: dict) -> FitAttempt:
            if spec["seasonal"] is not None:
                return FitAttempt(None, False, "forced failure")
            return original(self, values, spec)

        monkeypatch.setattr(ETSForecaster, "_fit_spec", fail_seasonal)
        model = ETSForecaster(cfg).fit(_split(panel, TRAIN))
        assert model.fit_info["n_fallback"] == len(SERIES)
        assert sorted(model.fit_info["fallback_ids"]) == sorted(SERIES)
        assert all(f.spec["seasonal"] is None for f in model.fits.values())
        assert np.isfinite(model.predict(HORIZON)[PREDICTION]).all()

    def test_no_fallback_for_healthy_series(self, fitted: dict[str, Forecaster]) -> None:
        model = fitted["ets"]
        assert all(f.spec["seasonal"] == "add" for f in model.fits.values())


class TestARIMA:
    def test_differencing_tests(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, _ = mini
        rng = np.random.default_rng(SEED)
        noise = rng.normal(size=80)
        walk = np.cumsum(rng.normal(size=80))
        seasonal = 10 * np.tile([1.0, -1.0, 2.0, -2.0], 20) + rng.normal(0, 0.1, 80)
        assert differencing_orders(noise, cfg.arima) == (0, 0)
        assert differencing_orders(walk, cfg.arima)[0] >= 1
        assert differencing_orders(seasonal, cfg.arima)[1] == 1
        assert seasonal_strength(seasonal, 4) > cfg.arima.seasonal_strength_threshold

    def test_bounds_come_from_config(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, _ = mini
        a = cfg.arima
        specs = candidate_specs(1, 0, a)
        assert len(specs) == (a.max_p + 1) * (a.max_q + 1) * (a.max_P + 1) * (a.max_Q + 1)
        assert max(s.order[0] for s in specs) == a.max_p
        assert max(s.seasonal_order[0] for s in specs) == a.max_P
        assert {s.trend for s in specs} == {"c"}  # d + D = 1 -> drift
        assert {s.trend for s in candidate_specs(1, 1, a)} == {"n"}

    def test_order_table_has_no_data(self, fitted: dict[str, Forecaster]) -> None:
        model = fitted["arima"]
        assert isinstance(model, ARIMAForecaster)
        table = model.order_table()
        assert set(table) == set(SERIES)
        for row in table.values():
            assert set(row) == {
                "order", "seasonal_order", "trend", "converged", "fallback",
                "n_candidates", "n_converged_candidates", "n_failed_candidates",
            }  # fmt: skip
        assert sum(model.fit_info["distribution"]["d"].values()) == len(SERIES)


# ---------------------------------------------------------------------------
# TFT
# ---------------------------------------------------------------------------


class TestTFT:
    def test_fit_info(self, fitted: dict[str, Forecaster]) -> None:
        info = fitted["tft"].fit_info
        assert 1 <= info["epochs_run"] <= info["max_epochs"]
        assert 1 <= info["best_epoch"] <= info["epochs_run"]
        assert np.isfinite(info["best_val_loss"])
        assert info["accelerator"] in {"cpu", "gpu", "mps"}

    def test_forecast_matches_evaluation_window(
        self, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        """forecast(history < origin) equals the dataset window at that origin, even when
        the actual future targets in that window are tampered with."""
        cfg, panel = mini
        model = fitted["tft"]
        assert isinstance(model, TFTForecaster)
        origin = split_bounds(panel)[TEST][0]
        tampered = panel.copy()
        tampered.loc[tampered[TIME_IDX] >= origin, cfg.prep.target_column] *= TAMPER_FACTOR

        training = build_training_dataset(_split(panel, TRAIN), cfg)
        window = build_evaluation_dataset(training, tampered, origin, origin + HORIZON - 1, cfg)
        values, index = model.predict_windows(window)
        expected = values[np.argsort(index["ID"].to_numpy())].ravel()
        got = _values(model.forecast(panel[panel[TIME_IDX] < origin], HORIZON))
        assert np.allclose(got, expected, atol=1e-5)

    def test_predict_frame_has_only_calendar_future(
        self, mini: tuple[Config, pd.DataFrame]
    ) -> None:
        cfg, panel = mini
        origin = split_bounds(panel)[TEST][0]
        frame = build_predict_frame(panel[panel[TIME_IDX] < origin], cfg)
        per_series = cfg.prep.max_encoder_length + cfg.prep.max_prediction_length
        assert (frame.groupby("ID").size() == per_series).all()
        assert frame[TIME_IDX].max() == origin + cfg.prep.max_prediction_length - 1
        future = frame[frame[TIME_IDX] >= origin]
        expected_q = panel.set_index(["ID", TIME_IDX])["quarter"]
        got_q = future.set_index(["ID", TIME_IDX])["quarter"]
        assert (got_q == expected_q.loc[got_q.index]).all()

    def test_horizon_longer_than_training_is_rejected(self, fitted: dict[str, Forecaster]) -> None:
        with pytest.raises(ValueError, match="horizon"):
            fitted["tft"].predict(HORIZON + 1)

    def test_deterministic_with_seed(
        self, fitted: dict[str, Forecaster], mini: tuple[Config, pd.DataFrame]
    ) -> None:
        cfg, panel = mini
        again = _fit("tft", cfg, panel)
        pd.testing.assert_frame_equal(fitted["tft"].predict(HORIZON), again.predict(HORIZON))
        assert again.fit_info["best_val_loss"] == fitted["tft"].fit_info["best_val_loss"]

    def test_requires_validation(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, panel = mini
        with pytest.raises(ValueError, match="validation"):
            TFTForecaster(cfg).fit(_split(panel, TRAIN))

    def test_accelerator_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_accelerator("auto") == "cpu"
        assert resolve_accelerator("cuda") == "cpu"
        assert resolve_accelerator("mps") == "cpu"
        assert resolve_accelerator("cpu") == "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_train_all(tmp_path: Path) -> None:
    cfg = mini_config(tmp_path)
    cfg.tft.max_epochs = 1
    processed = Path(cfg.data.processed_dir)
    processed.mkdir(parents=True)
    synthetic_panel(cfg).to_parquet(processed / cfg.prep.panel_file, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")

    result = CliRunner().invoke(app, ["train", "--model", "all", "--config", str(config_path)])
    assert result.exit_code == 0, result.output

    models_dir = Path(cfg.output.models_dir)
    o = cfg.output
    for file in (o.ets_file, o.arima_file, o.arima_orders_file, o.tft_file):
        assert (models_dir / file).exists()
    summary = json.loads((models_dir / o.training_summary_file).read_text())
    assert set(summary["models"]) == {"ets", "arima", "tft"}
    assert summary["data_scope"]["test_rows_used"] is False
    assert summary["models"]["tft"]["epochs_run"] == 1

    # training a single model keeps the other entries
    result = CliRunner().invoke(app, ["train", "--model", "ets", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    summary = json.loads((models_dir / o.training_summary_file).read_text())
    assert set(summary["models"]) == {"ets", "arima", "tft"}
