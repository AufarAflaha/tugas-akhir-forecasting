"""Tests for Phase 5 evaluation. All data is synthetic; expected values are hand-computed."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats
from typer.testing import CliRunner

from ta_forecasting.cli import app
from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN, TEST, TIME_IDX, VAL
from ta_forecasting.evaluation.backtest import (
    MAIN,
    MODEL,
    run_backtest,
    split_origins,
    validate_predictions,
)
from ta_forecasting.evaluation.evaluate import aggregate, run_backtests
from ta_forecasting.evaluation.metrics import (
    ACTUAL,
    mae,
    mape,
    metric_names,
    rmse,
    score,
    smape,
)
from ta_forecasting.evaluation.report import to_markdown, wins
from ta_forecasting.evaluation.significance import dm_test
from ta_forecasting.models.base import ORIGIN, PREDICTION, STEP
from ta_forecasting.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
from ta_forecasting.models.ets import ETSForecaster
from ta_forecasting.models.train import train_models
from tests.test_models import HORIZON, SERIES, _split, mini_config, synthetic_panel

# ---------------------------------------------------------------------------
# Metrics (hand-computed)
# ---------------------------------------------------------------------------

Y = [100.0, 200.0]
F = [110.0, 190.0]


class TestMetrics:
    def test_mape(self) -> None:
        # (10/100 + 10/200) / 2 * 100 = 7.5
        assert mape(Y, F) == pytest.approx(7.5)

    def test_smape(self) -> None:
        # (2*10/210 + 2*10/390) / 2 * 100 = 7.326007326...
        assert smape(Y, F) == pytest.approx(7.3260073260)
        assert smape([0.0, 10.0], [0.0, 10.0]) == 0.0

    def test_mae_and_rmse(self) -> None:
        assert mae(Y, F) == pytest.approx(10.0)
        assert rmse(Y, F) == pytest.approx(10.0)
        # errors 3 and 4: sqrt((9 + 16) / 2) = 3.5355339
        assert rmse([10.0, 10.0], [13.0, 6.0]) == pytest.approx(3.5355339059)
        assert mae([10.0, 10.0], [13.0, 6.0]) == pytest.approx(3.5)

    def test_perfect_forecast(self) -> None:
        for fn in (mape, smape, mae, rmse):
            assert fn(Y, Y) == 0.0

    def test_invalid_input(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            mape([0.0, 1.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="shape"):
            mae([1.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="finite"):
            rmse([1.0], [np.nan])

    def test_names_follow_config(self, config: Config) -> None:
        assert metric_names(config) == ["mape", "rmse", "mae", "smape"]

    def test_score_groups(self) -> None:
        frame = pd.DataFrame(
            {"g": ["a", "a", "b"], ACTUAL: [100.0, 200.0, 50.0], PREDICTION: [110.0, 190.0, 50.0]}
        )
        out = score(frame, ["g"], ["mape", "mae"]).set_index("g")
        assert out.loc["a", "mape"] == pytest.approx(7.5)
        assert out.loc["b", "mae"] == 0.0
        assert list(out["n"]) == [2, 1]


# ---------------------------------------------------------------------------
# Diebold-Mariano (hand-computed)
# ---------------------------------------------------------------------------


class TestDieboldMariano:
    D = np.array([1.0, 2.0, 3.0, 4.0])

    def test_horizon_one(self) -> None:
        # mean 2.5, gamma0 = 1.25, DM = 2.5 / sqrt(1.25/4) = 4.472136,
        # HLN factor sqrt((4 + 1 - 2) / 4) = 0.8660254 -> 3.8729833
        res = dm_test(self.D, np.zeros(4), horizon=1)
        assert res.statistic == pytest.approx(3.8729833462)
        assert res.p_value == pytest.approx(2 * stats.t.sf(3.8729833462, df=3))
        assert res.mean_diff == pytest.approx(2.5)
        assert not res.variance_fallback

    def test_horizon_two_uses_autocovariance(self) -> None:
        # gamma1 = 0.3125, long-run = 1.875, DM = 2.5 / sqrt(1.875/4) = 3.6514837,
        # HLN factor sqrt((4 + 1 - 4 + 2/4) / 4) = 0.6123724 -> sqrt(5) = 2.2360680
        res = dm_test(self.D, np.zeros(4), horizon=2)
        assert res.statistic == pytest.approx(math.sqrt(5.0))

    def test_antisymmetric(self) -> None:
        rng = np.random.default_rng(42)
        a, b = rng.random(13), rng.random(13)
        assert dm_test(a, b, 3).statistic == pytest.approx(-dm_test(b, a, 3).statistic)
        assert dm_test(a, b, 3).p_value == pytest.approx(dm_test(b, a, 3).p_value)

    def test_identical_losses_are_undefined(self) -> None:
        res = dm_test(self.D, self.D, 1)
        assert math.isnan(res.statistic) and math.isnan(res.p_value)


# ---------------------------------------------------------------------------
# Baselines, origins, backtest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, pd.DataFrame]:
    cfg = mini_config(tmp_path_factory.mktemp("eval"))
    cfg.tft.max_epochs = 1
    cfg.evaluation.n_jobs = 1
    return cfg, synthetic_panel(cfg)


def _history(values: list[float], cfg: Config) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "ID": "A",
            TIME_IDX: range(n),
            "quarter_label": [str(pd.Period("2010Q1", "Q") + i) for i in range(n)],
            cfg.prep.target_column: values,
        }
    )


class TestBaselines:
    def test_naive(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, _ = mini
        hist = _history([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], cfg)
        out = NaiveForecaster(cfg).fit(hist).forecast(hist, 4)
        assert list(out[PREDICTION]) == [6.0] * 4
        assert list(out[TIME_IDX]) == [6, 7, 8, 9]

    def test_seasonal_naive(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, _ = mini
        hist = _history([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], cfg)
        out = SeasonalNaiveForecaster(cfg).fit(hist).forecast(hist, 4)
        # targets t = 6..9 take the values at t - 4 = 2..5
        assert list(out[PREDICTION]) == [3.0, 4.0, 5.0, 6.0]


class TestOrigins:
    def test_origins_cover_full_horizon_inside_split(
        self, mini: tuple[Config, pd.DataFrame]
    ) -> None:
        cfg, panel = mini
        origins = split_origins(panel, cfg)
        for split in (VAL, TEST):
            idx = panel.loc[panel[SPLIT_COLUMN] == split, TIME_IDX]
            assert origins[split][0] == idx.min()
            assert origins[split][-1] + HORIZON - 1 == idx.max()

    def test_real_protocol_origins(self, config: Config) -> None:
        # 105 quarters: val 74-88 -> origins 74..85, test 89-104 -> origins 89..101
        times = pd.DataFrame({TIME_IDX: range(105)})
        times[SPLIT_COLUMN] = np.select(
            [times[TIME_IDX] < 74, times[TIME_IDX] < 89], ["train", VAL], default=TEST
        )
        origins = split_origins(times, config)
        assert (origins[VAL][0], origins[VAL][-1], len(origins[VAL])) == (74, 85, 12)
        assert (origins[TEST][0], origins[TEST][-1], len(origins[TEST])) == (89, 101, 13)


class TestBacktest:
    def test_rows_and_validation(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, panel = mini
        model = ETSForecaster(cfg).fit(_split(panel, "train"))
        result = run_backtest(model, panel, cfg)
        n_origins = sum(len(o) for o in split_origins(panel, cfg).values())
        assert len(result.predictions) == len(SERIES) * n_origins * HORIZON
        validate_predictions(result.predictions, panel, cfg)

    def test_validation_catches_violations(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, panel = mini
        preds = run_backtest(NaiveForecaster(cfg).fit(_split(panel, "train")), panel, cfg)
        good = preds.predictions
        with pytest.raises(ValueError, match="rows"):
            validate_predictions(good.iloc[1:], panel, cfg)
        shifted = good.copy()
        shifted.loc[shifted[SPLIT_COLUMN] == VAL, TIME_IDX] += HORIZON
        with pytest.raises(ValueError, match="outside"):
            validate_predictions(shifted, panel, cfg)
        other = good.copy()
        other.loc[other[SPLIT_COLUMN] == VAL, ORIGIN] += 1
        with pytest.raises(ValueError, match="origins"):
            validate_predictions(other, panel, cfg)

    def test_parallel_equals_serial(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, panel = mini
        model = ETSForecaster(cfg).fit(_split(panel, "train"))
        serial = run_backtest(model, panel, cfg).predictions
        par_cfg = cfg.model_copy(deep=True)
        par_cfg.evaluation.n_jobs = 2
        parallel = run_backtest(model, panel, par_cfg).predictions
        pd.testing.assert_frame_equal(serial, parallel)

    def test_val_forecasts_ignore_test_values(self, mini: tuple[Config, pd.DataFrame]) -> None:
        cfg, panel = mini
        model = ETSForecaster(cfg).fit(_split(panel, "train"))
        tampered = panel.copy()
        tampered.loc[tampered[SPLIT_COLUMN] == TEST, cfg.prep.target_column] *= 1000.0
        clean = run_backtest(model, panel, cfg).predictions
        dirty = run_backtest(model, tampered, cfg).predictions
        val = clean[SPLIT_COLUMN] == VAL
        pd.testing.assert_series_equal(clean.loc[val, PREDICTION], dirty.loc[val, PREDICTION])


# ---------------------------------------------------------------------------
# Aggregation and the full pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(mini: tuple[Config, pd.DataFrame]) -> tuple[Config, pd.DataFrame, Path]:
    cfg, panel = mini
    processed = Path(cfg.data.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(processed / cfg.prep.panel_file, index=False)
    train_models(cfg, ["ets", "arima", "tft"])
    return cfg, panel, processed


class TestAggregation:
    def test_wins_and_tables(self, trained: tuple[Config, pd.DataFrame, Path]) -> None:
        cfg, panel, _ = trained
        results = run_backtests(cfg, panel)
        predictions = pd.concat([r.predictions for r in results], ignore_index=True)
        validate_predictions(predictions, panel, cfg)
        tables, per_series = aggregate(predictions, cfg)

        main = cfg.evaluation.main_models
        win = tables["wins"].set_index(SPLIT_COLUMN)
        assert (win[[*main, "tie"]].sum(axis=1) == len(SERIES)).all()
        assert set(tables["per_horizon"][STEP]) == set(range(1, HORIZON + 1))
        assert set(tables["sensitivity"][MODEL]) == set(cfg.evaluation.sensitivity_models)
        dm = tables["dm_per_series_summary"]
        assert len(dm) == 3 * HORIZON  # 3 pairs x 4 horizons
        assert (dm["n_series"] == len(SERIES)).all()
        assert set(tables["overall"][MODEL]) == {*main, *cfg.evaluation.baselines}
        assert "ID" in per_series.columns  # per-series metrics stay out of reports/

    def test_wins_counts_ties(self) -> None:
        frame = pd.DataFrame(
            {
                "mode": MAIN,
                SPLIT_COLUMN: TEST,
                "ID": ["a", "a", "b", "b"],
                MODEL: ["x", "y", "x", "y"],
                "mape": [1.0, 2.0, 3.0, 3.0],
            }
        )
        out = wins(frame, ["x", "y"], "mape", "ID").iloc[0]
        assert (out["x"], out["y"], out["tie"]) == (1, 0, 1)

    def test_markdown_table(self) -> None:
        text = to_markdown(pd.DataFrame({"a": [1.23456], "b": ["x"]}), digits=2)
        assert text.splitlines() == ["| a | b |", "|---|---|", "| 1.23 | x |"]


def test_cli_evaluate(trained: tuple[Config, pd.DataFrame, Path], tmp_path: Path) -> None:
    cfg, _, _ = trained
    local = cfg.model_copy(deep=True)
    local.evaluation.output_dir = str(tmp_path / "eval")
    local.output.reports_dir = str(tmp_path / "reports")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(local.model_dump()), encoding="utf-8")

    result = CliRunner().invoke(app, ["evaluate", "--config", str(config_path)])
    assert result.exit_code == 0, result.output

    reports = Path(local.output.reports_dir)
    prefix = local.evaluation.report_prefix
    assert (reports / f"{prefix}_evaluation.md").exists()
    assert (Path(local.evaluation.output_dir) / local.evaluation.predictions_file).exists()
    for csv in reports.glob("*.csv"):
        columns = set(pd.read_csv(csv).columns)
        # reports hold aggregates only: no series ids, actuals or individual forecasts
        assert not columns & {"ID", ACTUAL, PREDICTION, TIME_IDX, ORIGIN}, csv.name
    assert len(list(Path(local.evaluation.output_dir).glob("*.png"))) == 5
