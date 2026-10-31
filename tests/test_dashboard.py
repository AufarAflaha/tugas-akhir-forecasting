"""Tests for the Phase 7b dashboard. All inputs are synthetic and written to tmp_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from ta_forecasting.config import Config, DashboardConfig, load_config
from ta_forecasting.dashboard import charts
from ta_forecasting.dashboard import data as dd
from ta_forecasting.dashboard.app import CONFIG_ENV, PAGE_ABOUT, PAGE_OVERVIEW
from ta_forecasting.forecasting import summary as fs
from ta_forecasting.forecasting.intervals import LOWER, UPPER
from tests.conftest import PROJECT_ROOT

APP = PROJECT_ROOT / "src" / "ta_forecasting" / "dashboard" / "app.py"
LAST = "2019Q4"
END = "2021Q4"
STEPS = 8
MODELS = ["ets", "arima", "tft", "naive"]
SERIES = {"S1": ("Absorber", "US"), "S2": ("Absorber", "EU"), "S3": ("Boiler", "US")}


def make_cfg(tmp: Path) -> Config:
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml").model_copy(deep=True)
    cfg.project.last_actual_quarter = LAST
    cfg.forecast.end_quarter = END
    cfg.dashboard.default_base_year = 2012
    cfg.dashboard.default_start_year = 2010
    cfg.dashboard.interval_max_step = 6
    cfg.dashboard.error_steps = [1, 4, 8]
    cfg.data.processed_dir = str(tmp / "processed")
    cfg.data.interim_dir = str(tmp / "interim")
    cfg.output.reports_dir = str(tmp / "reports")
    return cfg


def write_inputs(cfg: Config) -> None:
    """Synthetic panel 2010Q1–2019Q4, forecasts 2020Q1–2021Q4 and aggregate reports."""
    c = cfg.prep.columns
    for path in (cfg.data.processed_dir, cfg.data.interim_dir, cfg.output.reports_dir):
        Path(path).mkdir(parents=True, exist_ok=True)
    quarters = pd.period_range("2010Q1", LAST, freq="Q")
    rows = []
    for i, (sid, (concept, region)) in enumerate(SERIES.items()):
        for t, q in enumerate(quarters):
            # 2012 values are 50, 52, 54, 56 for S1 (mean 53)
            rows.append({c.series_id: sid, "quarter_label": str(q),
                         cfg.prep.target_column: 50.0 + 2.0 * (t - 8) + 10.0 * i + 42.0,
                         c.concept: concept, c.region: region})
    panel = pd.DataFrame(rows)
    panel.to_parquet(Path(cfg.data.processed_dir) / cfg.prep.panel_file, index=False)

    future = pd.period_range("2020Q1", END, freq="Q")
    fc = []
    for m in MODELS:
        n = 4 if m == "tft" else STEPS
        for sid in SERIES:
            last = float(panel.loc[panel[c.series_id] == sid, cfg.prep.target_column].iloc[-1])
            for h in range(1, n + 1):
                pred = last if m == "naive" else last - h + (h == 3) * -5.0
                fc.append({"model": m, c.series_id: sid, "origin": 40, "step": h,
                           "time_idx": 39 + h, "prediction": pred,
                           "quarter_label": str(future[h - 1]), "validated": h <= 4,
                           LOWER: pred * 0.9, UPPER: pred * 1.1})
    fc = pd.DataFrame(fc)
    fc.to_parquet(Path(cfg.data.processed_dir) / cfg.forecast.forecast_file, index=False)

    last_actual = panel.groupby(c.series_id)[cfg.prep.target_column].last()
    metrics = pd.DataFrame([{"model": m, "origins": "all", "step": h, "mape": 1.0 + h}
                            for m in MODELS for h in range(1, (4 if m == "tft" else STEPS) + 1)])
    summ = fs.summarize(fc, last_actual, metrics, cfg)
    summ.to_parquet(Path(cfg.data.processed_dir) / cfg.forecast.summary_file, index=False)

    reports = Path(cfg.output.reports_dir)
    errs = metrics.assign(rmse=1.0, mae=1.0, smape=1.0, n=10, n_origins=5)
    errs.to_csv(reports / f"{cfg.forecast.report_prefix}_long_horizon.csv", index=False)
    pd.DataFrame([{"model": m, "step": h, "coverage": 80.0, "n": 10, "nominal": 80.0}
                  for m in MODELS for h in (1, 2)]).to_csv(
        reports / f"{cfg.forecast.report_prefix}_coverage.csv", index=False)
    pd.DataFrame([{"mode": "main", "model": m, "split": s, "mape": 3.0, "rmse": 1.0, "mae": 1.0,
                   "smape": 3.0, "n": 10} for m in [*MODELS, "seasonal_naive"]
                  for s in ("val", "test")]).to_csv(
        reports / f"{cfg.evaluation.report_prefix}_overall.csv", index=False)

    lookup = pd.DataFrame([{c.lookup_id: sid, cfg.dashboard.lookup_label: f"Label {sid}",
                            cfg.dashboard.lookup_currency: "USD", "other": 1} for sid in SERIES])
    lookup.to_parquet(Path(cfg.data.interim_dir) / f"{cfg.prep.lookup_dataset}.parquet")
    desc = pd.DataFrame([{cfg.dashboard.description_id: "S1",
                          cfg.dashboard.description_text: "Deskripsi sintetis."}])
    desc.to_parquet(Path(cfg.data.interim_dir) / f"{cfg.dashboard.description_dataset}.parquet")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    out = make_cfg(tmp_path)
    write_inputs(out)
    return out


LABELS = {"ets": "ETS", "arima": "ARIMA", "tft": "TFT", "naive": "Naive"}


# ---------------------------------------------------------------------------
# Data functions
# ---------------------------------------------------------------------------


def test_load_and_select(cfg: Config) -> None:
    data = dd.load_data(cfg)
    assert dd.concepts(data, cfg) == ["Absorber", "Boiler"]
    assert dd.regions(data, "Absorber", cfg) == ["EU", "US"]
    assert dd.series_id(data, "Absorber", "US", cfg) == "S1"
    assert dd.base_years(data) == list(range(2010, 2020))
    assert dd.year_bounds(data, cfg) == (2010, 2021)
    meta = data.meta.set_index("ID")
    assert meta.loc["S1", "Description"] == "Deskripsi sintetis."
    assert pd.isna(meta.loc["S2", "Description"])


def test_missing_inputs_are_reported(cfg: Config) -> None:
    (Path(cfg.data.processed_dir) / cfg.forecast.forecast_file).unlink()
    with pytest.raises(dd.DashboardDataError, match="forecast"):
        dd.load_data(cfg)


def test_rejects_projection_quarters(cfg: Config) -> None:
    path = Path(cfg.data.processed_dir) / cfg.prep.panel_file
    panel = pd.read_parquet(path)
    extra = panel.tail(1).assign(quarter_label="2020Q1")
    pd.concat([panel, extra]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="last_actual_quarter"):
        dd.load_data(cfg)


def test_rejects_forecast_outside_range(cfg: Config) -> None:
    path = Path(cfg.data.processed_dir) / cfg.forecast.forecast_file
    fc = pd.read_parquet(path)
    fc.loc[0, "quarter_label"] = LAST
    fc.to_parquet(path, index=False)
    with pytest.raises(dd.DashboardDataError, match="rentang"):
        dd.load_data(cfg)


def test_rebase_and_intervals(cfg: Config) -> None:
    data = dd.load_data(cfg)
    act, fc = dd.series_view(data, "S1", ["ets", "naive"], 2012, cfg)
    # 2012 raw values: 92, 94, 96, 98 -> mean 95 -> factor 100 / 95
    assert act.loc[act["year"] == 2012, "prediction"].mean() == pytest.approx(100.0)
    raw_last = 92.0 + 2.0 * 31
    assert act["prediction"].iloc[-1] == pytest.approx(raw_last * 100 / 95)
    assert act["qoq_pct"].iloc[1] == pytest.approx(
        (act["prediction"].iloc[1] / act["prediction"].iloc[0] - 1) * 100)
    assert act["qoq_pct"].iloc[1] == pytest.approx((78.0 / 76.0 - 1) * 100)  # rebase-free
    naive = fc[fc["model"] == "naive"]
    assert naive["prediction"].iloc[0] == pytest.approx(raw_last * 100 / 95)
    assert fc.loc[fc["step"] > 6, [LOWER, UPPER]].isna().all().all()
    assert fc.loc[fc["step"] <= 6, [LOWER, UPPER]].notna().all().all()
    assert fc["model"].drop_duplicates().tolist() == ["ets", "naive"]
    with pytest.raises(dd.DashboardDataError, match="Tahun dasar"):
        dd.series_view(data, "S1", ["ets"], 2020, cfg)


def test_comparison_table(cfg: Config) -> None:
    data = dd.load_data(cfg)
    table = dd.comparison_table(data, "S1", ["ets", "tft", "naive"], cfg, LABELS).set_index(
        "Model")
    assert table.loc["ETS", "MAPE h=1 (%)"] == "2,0"
    assert table.loc["TFT", "MAPE h=8 (%)"] == dd.MISSING
    # ETS path: last - h, with an extra -5 at h = 3 -> cheapest in 1 year = h 3 (2020Q3)
    assert table.loc["ETS", "Termurah (1 tahun)"].startswith("2020Q3 (2/2 model)")
    # h 3 and h 8 tie at last - 8 -> the earliest quarter wins (A60)
    assert table.loc["ETS", "Termurah (s.d. 2021Q4)"].startswith("2020Q3")
    assert "ramalan hanya 4 kuartal" in table.loc["TFT", "Termurah (s.d. 2021Q4)"]
    assert table.loc["Naive", "Termurah (1 tahun)"] == "datar"
    assert table.loc["Naive", "Selisih (1 tahun)"] == "+0,0%"


def test_annual_table(cfg: Config) -> None:
    data = dd.load_data(cfg)
    act, fc = dd.series_view(data, "S1", ["ets", "tft"], 2012, cfg)
    table, fc_cols = dd.annual_table(act, fc, 2018, 2021, LABELS, cfg)
    assert fc_cols == ["2020", "2021"]
    rows = table.set_index("Baris")
    assert rows.loc["Aktual", "2018"] != dd.MISSING
    assert rows.loc["Aktual", "2020"] == dd.MISSING
    assert rows.loc["Ramalan ETS", "2019"] == dd.MISSING        # model rows: forecast years
    assert rows.loc["Ramalan ETS", "2021"] != dd.MISSING
    assert rows.loc["Ramalan TFT", "2020"] != dd.MISSING        # 4 quarters of TFT
    assert rows.loc["Ramalan TFT", "2021"] == dd.MISSING        # TFT stops at 2020Q4
    expected = act.loc[act["year"] == 2019, "prediction"].mean()
    assert table.iloc[0][str(2019)] == f"{expected:.1f}".replace(".", ",")


def test_annual_table_combines_partial_year(cfg: Config, tmp_path: Path) -> None:
    """A last actual quarter inside a year: that year = actual quarters + forecast quarters."""
    data = dd.load_data(cfg)
    act, fc = dd.series_view(data, "S1", ["ets"], 2012, cfg)
    act = act[act["quarter_label"] != "2019Q4"]
    fc4 = fc[fc["year"] == 2020].copy()
    fc4["year"] = 2019
    fc_partial = fc4.head(1)  # one forecast quarter completes 2019 (Q1-Q3 actual + Q4)
    local = cfg.model_copy(deep=True)
    local.project.last_actual_quarter = "2019Q3"
    table, cols = dd.annual_table(act, fc_partial, 2019, 2019, LABELS, local)
    assert cols == ["2019"]
    q = list(act.loc[act["year"] == 2019, "prediction"]) + [fc_partial["prediction"].iloc[0]]
    assert table.set_index("Baris").loc["Ramalan ETS", "2019"] == (
        f"{np.mean(q):.1f}".replace(".", ","))


def test_overview_and_consensus(cfg: Config) -> None:
    data = dd.load_data(cfg)
    table = dd.overview_table(data, fs.SHORT, ["ets", "arima", "naive"], cfg, LABELS)
    assert len(table) == len(SERIES)
    assert {"Cost model", "Region", "Kesepakatan", "ETS: termurah"} <= set(table.columns)
    assert "ID" not in table.columns
    assert (table["Naive: termurah"] == "datar").all()
    assert table["Kesepakatan"].str.startswith("2020Q3 (2/2 model)").all()
    only_one = dd.overview_table(data, fs.SHORT, ["ets"], cfg, LABELS)
    assert (only_one["Kesepakatan"] == "tidak ada").all()


def test_accuracy_table(cfg: Config) -> None:
    data = dd.load_data(cfg)
    table = dd.accuracy_table(data, cfg, LABELS)
    assert table["MAPE (%)"].eq("3,0").all()
    assert len(table) == len(MODELS) + 1


def test_main_chart(cfg: Config) -> None:
    data = dd.load_data(cfg)
    act, fc = dd.series_view(data, "S1", ["ets", "tft"], 2012, cfg)
    fig = charts.main_chart(act, fc, 2010, 2021, 2012, LABELS, cfg)
    names = [t.name for t in fig.data]
    assert names[:2] == ["Perubahan QoQ aktual (%)", "Aktual"]
    assert "Ramalan ETS" in names and "Interval 80% TFT" in names
    assert fig.layout.yaxis2.overlaying == "y" and fig.layout.yaxis2.side == "right"
    texts = [a.text for a in fig.layout.annotations]
    assert any("tervalidasi" in t for t in texts) and any("jangka panjang" in t for t in texts)
    # forecast line starts at the last actual point
    ets = next(t for t in fig.data if t.name == "Ramalan ETS")
    assert ets.y[0] == pytest.approx(act["prediction"].iloc[-1])


def test_no_outbound_settings(cfg: Config) -> None:
    assert cfg.dashboard.host == "127.0.0.1"
    with pytest.raises(ValueError, match="gather_usage_stats"):
        DashboardConfig(gather_usage_stats=True)
    toml = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text()
    assert "gatherUsageStats = false" in toml and 'address = "127.0.0.1"' in toml


# ---------------------------------------------------------------------------
# App (headless)
# ---------------------------------------------------------------------------


def _app(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(path))
    return AppTest.from_file(str(APP), default_timeout=60)


def test_app_pages_render(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(cfg, tmp_path, monkeypatch).run()
    assert not at.exception
    assert at.title[0].value == "Absorber"
    assert len(at.get("plotly_chart")) == 1
    assert len(at.dataframe) == 2
    assert any("S&P Global" in c.value for c in at.caption)
    assert any("internal" in c.value for c in at.caption)

    at.sidebar.multiselect(key="detail_models").set_value(MODELS).run()
    assert not at.exception
    assert any("TFT" in w.value for w in at.warning)

    at.sidebar.selectbox(key="concept").set_value("Boiler").run()
    assert not at.exception and at.title[0].value == "Boiler"

    at.sidebar.radio(key="page").set_value(PAGE_OVERVIEW).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == len(SERIES)
    at.sidebar.radio(key="window").set_value(fs.FULL).run()
    assert not at.exception

    at.sidebar.radio(key="page").set_value(PAGE_ABOUT).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1


def test_app_shows_error_without_inputs(cfg: Config, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    (Path(cfg.data.processed_dir) / cfg.forecast.summary_file).unlink()
    at = _app(cfg, tmp_path, monkeypatch).run()
    assert not at.exception
    assert at.error and "forecast" in at.error[0].value
