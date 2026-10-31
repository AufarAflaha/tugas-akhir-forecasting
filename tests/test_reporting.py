"""Tests for the Phase 6 report. Inputs are synthetic aggregate tables written to tmp_path."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ta_forecasting.cli import app
from ta_forecasting.config import Config, load_config
from ta_forecasting.reporting.build import run_report
from ta_forecasting.reporting.document import build_report
from ta_forecasting.reporting.figures import _decimal_axis, make_figures
from ta_forecasting.reporting.formatting import Formatter
from ta_forecasting.reporting.inputs import (
    ReportInputError,
    check_aggregate_only,
    check_no_series_ids,
    input_paths,
    load_inputs,
    table_path,
)
from tests.conftest import PROJECT_ROOT

MODELS = ["ets", "arima", "tft", "naive", "seasonal_naive"]
MAINS = ["ets", "arima", "tft"]
SPLITS = ["val", "test"]
STEPS = [1, 2, 3, 4]
N_SERIES = 6
N_ORIGINS = {"val": 2, "test": 3}
# Base MAPE per model and split; ETS best on test, ARIMA best on val, TFT worst.
BASE = {
    ("ets", "test"): 3.1, ("arima", "test"): 4.2, ("tft", "test"): 15.3,
    ("naive", "test"): 3.6, ("seasonal_naive", "test"): 4.6,
    ("ets", "val"): 5.4, ("arima", "val"): 5.2, ("tft", "val"): 7.3,
    ("naive", "val"): 5.5, ("seasonal_naive", "val"): 6.9,
}
FAKE_IDS = ["SYNTHSERIESID01", "SYNTHSERIESID02", "SYNTHSERIESID03",
            "SYNTHSERIESID04", "SYNTHSERIESID05", "SYNTHSERIESID06"]


def _metrics(base: float) -> dict[str, float]:
    return {"mape": base, "rmse": base * 3.7, "mae": base * 2.4, "smape": base * 0.98}


def write_inputs(cfg: Config, base: dict[tuple[str, str], float] | None = None) -> None:
    """Write a complete, consistent set of synthetic report inputs."""
    base = base or BASE
    reports = Path(cfg.output.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    n = {s: N_SERIES * N_ORIGINS[s] * len(STEPS) for s in SPLITS}

    overall, step, dist = [], [], []
    for (m, s), b in base.items():
        overall.append({"mode": "main", "model": m, "split": s, **_metrics(b), "n": n[s]})
        for h in STEPS:
            step.append({"mode": "main", "model": m, "split": s, "step": h,
                         **_metrics(b * (0.55 + 0.3 * h)), "n": n[s] // len(STEPS)})
        for name, value in _metrics(b).items():
            dist.append({"mode": "main", "model": m, "split": s, "metric": name,
                         "mean": value, "p10": value * 0.5, "p25": value * 0.7,
                         "p50": value * 0.9, "p75": value * 1.2, "p90": value * 1.6,
                         "n_series": N_SERIES})
    for m, delta in (("ets", 0.31), ("arima", -0.29)):
        for s in SPLITS:
            b = base[(m, s)] + delta
            overall.append({"mode": "refit", "model": m, "split": s, **_metrics(b), "n": n[s]})
    sens = []
    for m in ("arima", "ets"):
        for s in SPLITS:
            main, refit = _metrics(base[(m, s)]), _metrics(
                base[(m, s)] + (0.31 if m == "ets" else -0.29))
            row = {"model": m, "split": s}
            for k in main:
                row.update({f"{k}_main": main[k], f"{k}_refit": refit[k],
                            f"{k}_delta": refit[k] - main[k]})
            sens.append(row)
    dm, dms = [], []
    for a, b in combinations(MAINS, 2):
        for h in STEPS:
            p = 0.2 if (a, b) == ("ets", "arima") and h >= 3 else 0.00001
            better = min((a, b), key=lambda x: base[(x, "test")])
            dm.append({"model_a": a, "model_b": b, "step": h, "mean_loss_diff": -1.5,
                       "dm_hln": -4.0, "p_value": p, "significant": p < 0.05,
                       "better": better if p < 0.05 else "-", "n_origins": N_ORIGINS["test"],
                       "variance_fallback": False})
            dms.append({"model_a": a, "model_b": b, "step": h, "n_series": N_SERIES,
                        "n_undefined": 0, "n_variance_fallback": 0, "n_sig_a_better": 4,
                        "n_sig_b_better": 1, "n_sig_a_better_fdr": 3, "n_sig_b_better_fdr": 1,
                        "n_origins": N_ORIGINS["test"]})
    wins = [{"split": "test", "metric": "mape", "n_series": N_SERIES, "ets": 4, "arima": 1,
             "tft": 1, "tie": 0},
            {"split": "val", "metric": "mape", "n_series": N_SERIES, "ets": 1, "arima": 4,
             "tft": 1, "tie": 0}]
    forecasts = N_SERIES * sum(N_ORIGINS.values())
    timing = [{"model": m, "mode": "main", "seconds": 1.0 + i, "n_series_origin_forecasts":
               forecasts, "log_refit_not_converged": 0.0} for i, m in enumerate(MODELS)]
    timing += [{"model": "ets", "mode": "refit", "seconds": 30.0,
                "n_series_origin_forecasts": forecasts, "log_refit_not_converged": 0.0},
               {"model": "arima", "mode": "refit", "seconds": 40.0,
                "n_series_origin_forecasts": forecasts, "log_refit_not_converged": 2.0}]
    frames = {"overall": overall, "per_horizon": step, "per_series_distribution": dist,
              "wins": wins, "dm_panel": dm, "dm_per_series_summary": dms, "timing": timing,
              "sensitivity": sens}
    for key, rows in frames.items():
        pd.DataFrame(rows).round(4).to_csv(table_path(cfg, key), index=False)

    models = Path(cfg.output.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    training = {
        "seed": cfg.project.seed,
        "environment": {"python": "3.12.3", "torch": "2.2.2"},
        "models": {
            "ets": {"n_series": N_SERIES, "fit_seconds": 1.5, "n_fallback": 0,
                    "spec": {"trend": "add", "damped_trend": True, "seasonal": "add",
                             "seasonal_periods": 4},
                    "non_converged_ids": [FAKE_IDS[0]]},
            "arima": {"n_series": N_SERIES, "fit_seconds": 9.5, "n_fallback": 0,
                      "search": {"seasonal_strength_threshold": 0.64, "kpss_alpha": 0.05,
                                 "max_p": 3, "max_q": 3, "max_P": 1, "max_Q": 1,
                                 "information_criterion": "aic"}},
            "tft": {"n_series": N_SERIES, "fit_seconds": 20.0, "n_fallback": 0,
                    "n_parameters": 5000, "n_train_windows": 100, "best_epoch": 2,
                    "epochs_run": 7, "loss": "QuantileLoss", "quantiles": [0.1, 0.5, 0.9],
                    "accelerator": "cpu"},
        },
    }
    (models / cfg.output.training_summary_file).write_text(json.dumps(training))
    orders = {sid: {"order": [1, 1 if i else 0, 1], "seasonal_order": [i % 2, 0, 0, 4],
                    "converged": True, "fallback": False, "trend": "c"}
              for i, sid in enumerate(FAKE_IDS)}
    (models / cfg.output.arima_orders_file).write_text(json.dumps(orders))
    processed = Path(cfg.data.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    splits = {"train": ("2000Q1", "2004Q4", 20), "val": ("2005Q1", "2006Q1", 5),
              "test": ("2006Q2", "2007Q3", 6)}
    summary = {
        "last_actual_quarter": cfg.project.last_actual_quarter, "n_series": N_SERIES,
        "n_quarters": 31, "target_series_type": "Cost Index",
        "splits": {k: {"start_quarter": a, "end_quarter": b, "n_quarters": q,
                       "n_rows": q * N_SERIES} for k, (a, b, q) in splits.items()},
    }
    (processed / cfg.prep.summary_file).write_text(json.dumps(summary))


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    out = load_config(PROJECT_ROOT / "configs" / "config.yaml").model_copy(deep=True)
    out.output.reports_dir = str(tmp_path / "reports")
    out.output.models_dir = str(tmp_path / "models")
    out.output.logs_dir = str(tmp_path / "logs")
    out.data.processed_dir = str(tmp_path / "processed")
    out.report.figures_dir = str(tmp_path / "figures")
    write_inputs(out)
    return out


# ---------------------------------------------------------------------------
# Helpers: parse the Markdown tables back out of the report
# ---------------------------------------------------------------------------


def num(value: float, digits: int = 2) -> str:
    """Independent re-implementation of the report number format."""
    return f"{value:.{digits}f}".replace(".", ",")


def table_after(text: str, caption_start: str) -> pd.DataFrame:
    """The Markdown table that follows the caption starting with ``caption_start``."""
    lines = text.splitlines()
    idx = next(i for i, line in enumerate(lines)
               if re.match(r"\*\*Tabel \d+\.\*\* " + re.escape(caption_start), line))
    rows = []
    for line in lines[idx + 2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    header, body = rows[0], rows[2:]
    return pd.DataFrame(body, columns=header)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_report_sections_and_figures(cfg: Config) -> None:
    result = run_report(cfg)
    text = result.report_path.read_text(encoding="utf-8")
    for heading in ["## 1. Ringkasan Eksekutif", "## 2. Ringkasan Desain Eksperimen",
                    "## 3. Hasil", "### 3.4 Uji signifikansi Diebold-Mariano",
                    "### 3.5 Analisis sensitivitas", "## 4. Pembahasan",
                    "## 5. Jawaban Pertanyaan Penelitian", "**RQ1.", "**RQ2.", "**RQ3.",
                    "## 6. Keterbatasan dan Saran", "## 7. Reprodusibilitas"]:
        assert heading in text, heading
    assert set(result.figures) == {"overall", "horizon", "boxplot", "dm_heatmap"}
    w, h = cfg.report.figure.size
    dpi = cfg.report.figure.dpi
    for path in result.figures.values():
        assert path.exists()
        assert Path(cfg.report.figures_dir) in path.parents
        assert mpimg.imread(path).shape[:2] == (round(h * dpi), round(w * dpi))
        assert f"]({Path('..') / 'figures' / path.name})" in text
    assert "python -m ta_forecasting.cli report --config" in text


def test_tables_complete(cfg: Config) -> None:
    text = run_report(cfg).report_path.read_text(encoding="utf-8")
    labels = [cfg.report.model_labels[m] for m in MODELS]
    for split in ("uji", "validasi"):
        overall = table_after(text, f"Metrik keseluruhan pada data {split}")
        assert overall["Model"].tolist() == labels
        horizon = table_after(text, f"MAPE (%) per horizon pada data {split}")
        assert horizon.columns.tolist() == ["Model", *[f"h = {h}" for h in STEPS]]
        assert horizon["Model"].tolist() == labels
        dist = table_after(text, f"Distribusi MAPE per seri (%; data {split})")
        assert dist.columns.tolist() == ["Model", "Rata-rata", "p10", "p25", "p50", "p75",
                                         "p90", "n seri"]
    long = table_after(text, "Semua metrik per horizon")
    assert len(long) == len(MAINS) * len(STEPS)
    dm = table_after(text, "Uji Diebold-Mariano (HLN) tingkat panel")
    assert len(dm) == 3 * len(STEPS)
    dms = table_after(text, "Uji Diebold-Mariano per seri")
    assert len(dms) == 3 * len(STEPS)
    assert len(table_after(text, "Sensitivitas")) == 2 * len(SPLITS)
    assert table_after(text, "Waktu komputasi")["Model"].tolist() == labels
    assert not re.search(r"\bnan\b", text)


def test_numbers_match_phase5_csv(cfg: Config) -> None:
    text = run_report(cfg).report_path.read_text(encoding="utf-8")
    overall = pd.read_csv(table_path(cfg, "overall"))
    step = pd.read_csv(table_path(cfg, "per_horizon"))
    dist = pd.read_csv(table_path(cfg, "per_series_distribution"))
    labels = cfg.report.model_labels
    for split, word in (("test", "uji"), ("val", "validasi")):
        table = table_after(text, f"Metrik keseluruhan pada data {word}").set_index("Model")
        for row in overall[(overall["mode"] == "main") & (overall["split"] == split)].itertuples():
            cells = table.loc[labels[row.model]]
            assert cells["MAPE (%)"] == num(row.mape)
            assert cells["RMSE"] == num(row.rmse)
            assert cells["MAE"] == num(row.mae)
            assert cells["SMAPE (%)"] == num(row.smape)
            assert cells["n prediksi"].replace(".", "") == str(row.n)
        pivot = table_after(text, f"MAPE (%) per horizon pada data {word}").set_index("Model")
        for row in step[(step["mode"] == "main") & (step["split"] == split)].itertuples():
            assert pivot.loc[labels[row.model], f"h = {row.step}"] == num(row.mape)
        d = table_after(text, f"Distribusi MAPE per seri (%; data {word})").set_index("Model")
        part = dist[(dist["mode"] == "main") & (dist["split"] == split)
                    & (dist["metric"] == "mape")]
        for row in part.itertuples():
            for col in ("p10", "p25", "p50", "p75", "p90"):
                assert d.loc[labels[row.model], col] == num(getattr(row, col))
    dm = pd.read_csv(table_path(cfg, "dm_panel"))
    table = table_after(text, "Uji Diebold-Mariano (HLN) tingkat panel")
    for i, row in enumerate(dm.itertuples()):
        expected = "< 0,0001" if row.p_value < 0.00005 else num(row.p_value, 4)
        assert table.iloc[i]["p-value"] == expected
        assert table.iloc[i]["Kesimpulan"].startswith(
            "Signifikan" if row.significant else "Tidak signifikan")
    # headline sentence uses the best test model and its CSV value
    assert "**ETS** (MAPE 3,10%" in text


def test_no_raw_values_in_outputs(cfg: Config) -> None:
    result = run_report(cfg)
    text = result.report_path.read_text(encoding="utf-8")
    for sid in FAKE_IDS:
        assert sid not in text
    forbidden = set(cfg.report.forbidden_columns)
    for t in result.tables:
        assert not forbidden & set(t.frame.columns), t.key
    assert "time_idx" not in text and "prediction" not in text.lower()


def test_rejects_non_aggregate_input(cfg: Config) -> None:
    path = table_path(cfg, "wins")
    frame = pd.read_csv(path)
    frame["ID"] = "x"
    frame.to_csv(path, index=False)
    with pytest.raises(ReportInputError, match="non-aggregate"):
        load_inputs(cfg)
    with pytest.raises(ReportInputError):
        check_aggregate_only(pd.DataFrame({"actual": [1.0]}), cfg, "t")


def test_refuses_to_write_series_ids(cfg: Config) -> None:
    inputs = load_inputs(cfg)
    with pytest.raises(ReportInputError, match="series ID"):
        check_no_series_ids(f"text {FAKE_IDS[2]} text", inputs, "x")


def test_rejects_incomplete_tables(cfg: Config) -> None:
    path = table_path(cfg, "per_horizon")
    frame = pd.read_csv(path)
    frame[~((frame["model"] == "tft") & (frame["step"] == 4))].to_csv(path, index=False)
    with pytest.raises(ReportInputError, match="per_horizon"):
        load_inputs(cfg)


def test_rejects_missing_file(cfg: Config) -> None:
    input_paths(cfg)["split_summary"].unlink()
    with pytest.raises(ReportInputError, match="Missing"):
        load_inputs(cfg)


def test_arima_orders_are_only_counted(cfg: Config) -> None:
    counts = load_inputs(cfg).arima
    assert counts.n_series == N_SERIES
    assert counts.d == {0: 1, 1: 5}
    assert counts.n_seasonal_terms == 3
    assert FAKE_IDS[0] not in repr(counts)


def test_text_follows_the_numbers(cfg: Config) -> None:
    """If TFT had the lowest test MAPE, the report must say so instead of a fixed story."""
    flipped = dict(BASE)
    flipped[("tft", "test")] = 2.0
    write_inputs(cfg, flipped)
    text = run_report(cfg).report_path.read_text(encoding="utf-8")
    assert "**TFT** (MAPE 2,00%" in text
    assert re.search(r"\*\*RQ2\..*?\n\nYa, terhadap ETS dan ARIMA", text, re.S)
    assert "Dugaan penyebab utama" not in text
    original = run_report_text(cfg, BASE)
    assert re.search(r"\*\*RQ2\..*?\n\nTidak\.", original, re.S)
    assert "Naive yang kuat: ARIMA dan TFT" in original


def run_report_text(cfg: Config, base: dict[tuple[str, str], float]) -> str:
    write_inputs(cfg, base)
    return run_report(cfg).report_path.read_text(encoding="utf-8")


def test_report_is_deterministic(cfg: Config) -> None:
    first = run_report(cfg).report_path.read_text(encoding="utf-8")
    second = run_report(cfg).report_path.read_text(encoding="utf-8")
    assert first == second


def test_formatter(cfg: Config) -> None:
    f = Formatter(cfg)
    assert f.num(3.14159) == "3,14"
    assert f.pval(0.00001) == "< 0,0001"
    assert f.pval(0.0665) == "0,0665"
    assert f.signed(-0.3) == "-0,30"
    assert f.count(6032) == "6.032"
    assert f.join(["a", "b", "c"]) == "a, b, dan c"
    assert f.mval("mape", 3.2) == "3,20%" and f.mval("rmse", 3.2) == "3,20"


def test_axis_uses_decimal_comma(cfg: Config) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0.0, 2.5])
    _decimal_axis(ax, cfg)
    assert ax.yaxis.get_major_formatter()(2.5, 0) == "2,5"
    plt.close(fig)


def test_cli_report(cfg: Config, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(app, ["report", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert (Path(cfg.output.reports_dir) / cfg.report.output_file).exists()


REAL_CFG = PROJECT_ROOT / "configs" / "config.yaml"


def _real_inputs_present() -> bool:
    real = load_config(REAL_CFG)
    return all((PROJECT_ROOT / p).exists() for p in input_paths(real).values())


@pytest.mark.skipif(not _real_inputs_present(), reason="Phase 5 outputs not available locally")
def test_real_inputs_match_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the real Phase 5 tables: every MAPE in the test tables equals the CSV value."""
    monkeypatch.chdir(PROJECT_ROOT)
    real = load_config(REAL_CFG).model_copy(deep=True)
    real.report.figures_dir = str(tmp_path / "figures")  # do not touch outputs/report
    inputs = load_inputs(real)
    figures = make_figures(inputs, real)
    text, _ = build_report(inputs, real, figures, tmp_path / "FINAL_REPORT.md")
    check_no_series_ids(text, inputs, "real report")
    overall = inputs.tables["overall"]
    table = table_after(text, "Metrik keseluruhan pada data uji").set_index("Model")
    for row in overall[(overall["mode"] == "main") & (overall["split"] == "test")].itertuples():
        assert table.loc[real.report.model_labels[row.model], "MAPE (%)"] == num(row.mape)
