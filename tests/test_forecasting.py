"""Tests for Phase 7a forecasting. Synthetic panel, shrunk models; expected values by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ta_forecasting.cli import app
from ta_forecasting.config import Config
from ta_forecasting.data.prepare import QUARTER_LABEL, SPLIT_COLUMN, TIME_IDX, split_bounds
from ta_forecasting.evaluation.backtest import MODEL
from ta_forecasting.evaluation.metrics import ACTUAL
from ta_forecasting.forecasting import intervals, summary
from ta_forecasting.forecasting.horizon import (
    check_panel_ends_at_last_actual,
    future_quarters,
    model_steps,
    total_steps,
)
from ta_forecasting.forecasting.long_backtest import (
    ALL_ORIGINS,
    N_ORIGINS,
    ORIGIN_SET,
    horizon_metrics,
    long_origins,
    run_long_backtest,
)
from ta_forecasting.forecasting.run import production_forecast, run_forecast
from ta_forecasting.models.base import ORIGIN, PREDICTION, STEP
from ta_forecasting.models.train import train_models
from tests.test_models import HORIZON, LAST_QUARTER, SERIES, mini_config, synthetic_panel

END_QUARTER = "2021Q4"  # 8 quarters after LAST_QUARTER (2019Q4)
STEPS = 8


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, pd.DataFrame]:
    tmp = tmp_path_factory.mktemp("forecast")
    cfg = mini_config(tmp)
    cfg.tft.max_epochs = 1
    cfg.evaluation.n_jobs = 1
    cfg.forecast.n_jobs = 1
    cfg.forecast.end_quarter = END_QUARTER
    cfg.output.reports_dir = str(tmp / "reports")
    panel = synthetic_panel(cfg)
    processed = Path(cfg.data.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(processed / cfg.prep.panel_file, index=False)
    train_models(cfg, ["ets", "arima", "tft"])
    return cfg, panel


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------


def test_steps(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg, panel = trained
    assert total_steps(cfg) == STEPS
    assert model_steps("ets", cfg) == STEPS
    assert model_steps("tft", cfg) == HORIZON
    last = check_panel_ends_at_last_actual(panel, cfg)
    q = future_quarters(last, 3, cfg)
    assert q[QUARTER_LABEL].tolist() == ["2020Q1", "2020Q2", "2020Q3"]
    assert q[TIME_IDX].tolist() == [last + 1, last + 2, last + 3]


def test_end_quarter_must_follow_last_actual(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg = trained[0].model_copy(deep=True)
    cfg.forecast.end_quarter = LAST_QUARTER
    with pytest.raises(ValueError, match="end_quarter"):
        total_steps(cfg)


# ---------------------------------------------------------------------------
# No data after last_actual_quarter
# ---------------------------------------------------------------------------


def _with_projection_row(panel: pd.DataFrame) -> pd.DataFrame:
    extra = panel[panel[TIME_IDX] == panel[TIME_IDX].max()].copy()
    extra[TIME_IDX] += 1
    extra[QUARTER_LABEL] = str(pd.Period(LAST_QUARTER, "Q") + 1)
    return pd.concat([panel, extra], ignore_index=True)


def test_projection_rows_are_rejected(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg, panel = trained
    tampered = _with_projection_row(panel)
    with pytest.raises(ValueError, match="last_actual_quarter"):
        production_forecast(cfg, tampered)
    with pytest.raises(ValueError, match="last_actual_quarter"):
        run_long_backtest(cfg, tampered)


def test_panel_must_end_at_last_actual(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg, panel = trained
    with pytest.raises(ValueError, match="end at last_actual_quarter"):
        check_panel_ends_at_last_actual(panel[panel[TIME_IDX] < panel[TIME_IDX].max()], cfg)


# ---------------------------------------------------------------------------
# Long backtest
# ---------------------------------------------------------------------------


def test_long_backtest(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg, panel = trained
    pred = run_long_backtest(cfg, panel)
    last = int(panel[TIME_IDX].max())
    first = split_bounds(panel)[cfg.forecast.backtest_first_split][0]
    assert pred[TIME_IDX].max() <= last
    assert set(pred[MODEL]) == set(cfg.forecast.models)
    assert pred.loc[pred[MODEL] == "tft", STEP].max() == HORIZON
    assert pred.loc[pred[MODEL] == "ets", STEP].max() == STEPS
    # actuals are the panel values at the target quarter
    actual = panel.set_index(["ID", TIME_IDX])[cfg.prep.target_column]
    looked_up = actual.loc[list(zip(pred["ID"], pred[TIME_IDX], strict=True))].to_numpy()
    assert np.allclose(pred[ACTUAL].to_numpy(), looked_up)
    # each origin keeps the split it falls in
    origins = long_origins(panel, cfg)
    for split, values in origins.items():
        assert set(pred.loc[pred[SPLIT_COLUMN] == split, ORIGIN]) <= set(values)

    metrics = horizon_metrics(pred, cfg)
    allo = metrics[metrics[ORIGIN_SET] == ALL_ORIGINS].set_index([MODEL, STEP])
    for h in range(1, STEPS + 1):
        assert allo.loc[("ets", h), N_ORIGINS] == last - h + 1 - first + 1
        assert allo.loc[("ets", h), "n"] == len(SERIES) * (last - h + 1 - first + 1)


# ---------------------------------------------------------------------------
# Intervals (hand-computed)
# ---------------------------------------------------------------------------


def _ratios_frame(ratios: list[float], split: str = "val") -> pd.DataFrame:
    return pd.DataFrame({MODEL: "m", STEP: 1, SPLIT_COLUMN: split,
                         PREDICTION: 100.0, ACTUAL: [100.0 * r for r in ratios]})


def test_ratio_quantiles_and_apply(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg = trained[0].model_copy(deep=True)
    cfg.forecast.interval_level = 0.8
    q = intervals.ratio_quantiles(_ratios_frame([0.9, 1.0, 1.1, 1.2, 1.3]), cfg)
    # numpy linear quantiles: q(0.1) = 0.9 + 0.4 * 0.1 = 0.94, q(0.9) = 1.2 + 0.6 * 0.1 = 1.26
    assert q[intervals.Q_LO].iloc[0] == pytest.approx(0.94)
    assert q[intervals.Q_HI].iloc[0] == pytest.approx(1.26)
    fc = pd.DataFrame({MODEL: ["m"], STEP: [1], PREDICTION: [50.0]})
    out = intervals.apply_intervals(fc, q)
    assert out[intervals.LOWER].iloc[0] == pytest.approx(47.0)
    assert out[intervals.UPPER].iloc[0] == pytest.approx(63.0)
    with pytest.raises(ValueError, match="No backtest errors"):
        intervals.apply_intervals(fc.assign(**{STEP: 2}), q)


def test_coverage_check(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg = trained[0].model_copy(deep=True)
    cfg.forecast.interval_level = 0.8
    calib = _ratios_frame([0.9, 1.0, 1.1, 1.2, 1.3], "val")  # band [0.94, 1.26]
    check = _ratios_frame([0.95, 1.0, 1.25, 1.3], "test")    # 3 of 4 inside
    cov = intervals.coverage_check(pd.concat([calib, check]), cfg)
    assert cov["coverage"].iloc[0] == pytest.approx(75.0)
    assert cov["n"].iloc[0] == 4


# ---------------------------------------------------------------------------
# Summary (hand-computed)
# ---------------------------------------------------------------------------


def _path(model: str, sid: str, values: list[float]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        MODEL: model, "ID": sid, STEP: range(1, n + 1),
        QUARTER_LABEL: [str(pd.Period("2026Q2", "Q") + i) for i in range(n)],
        PREDICTION: values,
    })


def test_summary(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg = trained[0].model_copy(deep=True)
    cfg.prep.max_prediction_length = 4
    cfg.forecast.models = ["ets", "arima", "tft", "naive"]
    fc = pd.concat([
        _path("ets", "X", [99.0, 98.0, 97.0, 99.0, 90.0, 95.0]),
        _path("arima", "X", [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]),
        _path("tft", "X", [99.5, 98.5, 96.0, 97.0]),
        _path("naive", "X", [100.0] * 6),
    ], ignore_index=True)
    metrics = pd.DataFrame([
        {MODEL: m, ORIGIN_SET: ALL_ORIGINS, STEP: h, "mape": 2.0 * h}
        for m in ("ets", "arima", "tft", "naive") for h in range(1, 7)
    ])
    out = summary.summarize(fc, pd.Series({"X": 100.0}), metrics, cfg).set_index(
        [MODEL, summary.WINDOW])

    ets_short = out.loc[("ets", summary.SHORT)]
    assert ets_short[summary.CHEAPEST_QUARTER] == "2026Q4"
    assert ets_short[summary.CHANGE_PCT] == pytest.approx(-3.0)
    assert ets_short[summary.ERROR_AT_STEP] == pytest.approx(6.0)
    assert bool(ets_short[summary.WITHIN_ERROR]) is True        # |-3| < 6
    assert ets_short[summary.N_SAME] == 2                       # TFT also picks 2026Q4
    assert ets_short[summary.CONSENSUS_QUARTER] == "2026Q4"

    assert out.loc[("ets", summary.FULL), summary.CHEAPEST_QUARTER] == "2027Q2"   # step 5
    assert out.loc[("arima", summary.SHORT), summary.CHEAPEST_STEP] == 1
    assert bool(out.loc[("naive", summary.SHORT), summary.FLAT]) is True
    assert out.loc[("naive", summary.SHORT), summary.CHEAPEST_QUARTER] is None
    assert out.loc[("ets", summary.SHORT), summary.N_IN_WINDOW] == 3   # naive excluded

    # full window: ETS 2027Q2, ARIMA 2026Q2, TFT 2026Q4 -> no quarter picked twice
    full = out.xs(summary.FULL, level=summary.WINDOW)
    assert full[summary.CONSENSUS_QUARTER].isna().all()
    assert (full[summary.CONSENSUS_COUNT] == 1).all()

    stats = summary.summary_stats(out.reset_index(), cfg)
    assert set(stats[MODEL]) == {"ets", "arima", "tft", "naive"}
    counts = summary.cheapest_counts(out.reset_index(), cfg)
    assert "datar" in set(counts[summary.CHEAPEST_QUARTER])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_run_forecast(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg, panel = trained
    result = run_forecast(cfg)
    fc = result.forecast
    last = int(panel[TIME_IDX].max())
    assert (fc[ORIGIN] == last + 1).all()
    for name, group in fc.groupby(MODEL):
        assert len(group) == len(SERIES) * model_steps(name, cfg)
        assert group[QUARTER_LABEL].max() == (END_QUARTER if name != "tft" else "2020Q4")
    assert fc["validated"].eq(fc[STEP] <= HORIZON).all()
    assert (fc[intervals.LOWER] <= fc[intervals.UPPER]).all()
    assert np.isfinite(fc[[PREDICTION, intervals.LOWER, intervals.UPPER]].to_numpy()).all()

    reports = Path(cfg.output.reports_dir)
    for path in result.files:
        assert path.exists()
        if path.parent == reports and path.suffix == ".csv":
            columns = set(pd.read_csv(path).columns)
            assert not columns & {"ID", ACTUAL, PREDICTION, TIME_IDX, ORIGIN}, path.name
    text = (reports / f"{cfg.forecast.report_prefix}_long_horizon.md").read_text()
    for sid in SERIES:
        assert f"| {sid} |" not in text
    assert Path(cfg.data.processed_dir) / cfg.forecast.forecast_file in result.files


def test_forecast_is_deterministic(trained: tuple[Config, pd.DataFrame]) -> None:
    cfg, _ = trained
    first = run_forecast(cfg).forecast
    second = run_forecast(cfg).forecast
    pd.testing.assert_frame_equal(first, second)


def test_cli_forecast(trained: tuple[Config, pd.DataFrame], tmp_path: Path) -> None:
    cfg, _ = trained
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(app, ["forecast", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "tft" in result.output
