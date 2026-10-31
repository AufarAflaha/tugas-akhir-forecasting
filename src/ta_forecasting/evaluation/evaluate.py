"""Phase 5 — rolling-origin evaluation of ETS, ARIMA and TFT (plus reference baselines).

1. Load the Phase 4 artefacts (no retraining). Baselines are "fitted" on train rows.
2. Backtest every model on the val and test origins with ``refit_statistical = false``.
3. Sensitivity: ETS and ARIMA again with ``refit_statistical = true`` (reported separately).
4. Validate the forecasts, then score them overall, per horizon and per series, count wins,
   and run Diebold-Mariano tests on the test split.

Forecasts (with actuals) and per-series metrics go to ``evaluation.output_dir`` (not
committed). Only aggregate tables go to ``output.reports_dir``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN, TRAIN
from ta_forecasting.evaluation import report
from ta_forecasting.evaluation.backtest import (
    MAIN,
    MODE,
    MODEL,
    REFIT,
    BacktestResult,
    run_backtest,
    split_origins,
    validate_predictions,
)
from ta_forecasting.evaluation.metrics import metric_names, score
from ta_forecasting.evaluation.significance import diebold_mariano
from ta_forecasting.models.base import STEP, Forecaster, check_no_projection
from ta_forecasting.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
from ta_forecasting.models.train import MODELS, artefact_path

logger = logging.getLogger(__name__)

BASELINES: dict[str, type[Forecaster]] = {
    NaiveForecaster.name: NaiveForecaster,
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
}


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate tables plus the paths of everything written."""

    tables: dict[str, pd.DataFrame]
    timing: pd.DataFrame
    files: list[Path]


def load_panel(cfg: Config) -> pd.DataFrame:
    """Processed panel (actuals only, guarded against projections)."""
    panel = pd.read_parquet(Path(cfg.data.processed_dir) / cfg.prep.panel_file)
    check_no_projection(panel, cfg)
    return panel


def load_model(name: str, cfg: Config, panel: pd.DataFrame) -> Forecaster:
    """Load a Phase 4 artefact (or fit a baseline on train) and check its config."""
    if name in BASELINES:
        return BASELINES[name](cfg).fit(panel[panel[SPLIT_COLUMN] == TRAIN])
    model = MODELS[name].load(artefact_path(name, cfg))
    for section in ("prep", "project"):
        if getattr(model.cfg, section) != getattr(cfg, section):
            raise ValueError(f"{name}: artefact was trained with a different `{section}` config.")
    return model


def _refit_config(model: Forecaster) -> Config:
    out = model.cfg.model_copy(deep=True)
    out.rolling.refit_statistical = True
    return out


def run_backtests(cfg: Config, panel: pd.DataFrame) -> list[BacktestResult]:
    """Main backtests for all models and baselines, then the refit sensitivity runs."""
    e = cfg.evaluation
    results = []
    for name in [*e.main_models, *e.baselines]:
        model = load_model(name, cfg, panel)
        if model.cfg.rolling.refit_statistical:
            raise ValueError(f"{name}: the main run requires rolling.refit_statistical = false.")
        results.append(run_backtest(model, panel, cfg, MAIN))
    if e.run_sensitivity:
        for name in e.sensitivity_models:
            model = load_model(name, cfg, panel)
            model.cfg = _refit_config(model)
            results.append(run_backtest(model, panel, cfg, REFIT))
    return results


def _timing(results: list[BacktestResult], panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    origins = split_origins(panel, cfg)
    n_forecasts = (
        sum(len(o) for o in origins.values()) * panel[cfg.prep.columns.series_id].nunique()
    )
    rows = []
    for r in results:
        row: dict[str, Any] = {
            MODEL: r.model,
            MODE: r.mode,
            "seconds": round(r.seconds, 2),
            "n_series_origin_forecasts": n_forecasts,
        }
        row.update({f"log_{k}": v for k, v in sorted(r.forecast_log.items())})
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def _sensitivity(overall: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    refit = overall[overall[MODE] == REFIT]
    if refit.empty:
        return pd.DataFrame()
    main = overall[overall[MODE] == MAIN]
    merged = main.merge(refit, on=[MODEL, SPLIT_COLUMN], suffixes=("_main", "_refit"))
    cols = [MODEL, SPLIT_COLUMN]
    for name in names:
        merged[f"{name}_delta"] = merged[f"{name}_refit"] - merged[f"{name}_main"]
        cols += [f"{name}_main", f"{name}_refit", f"{name}_delta"]
    return merged[cols]


def aggregate(
    predictions: pd.DataFrame, cfg: Config
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """All aggregate tables, and the per-series metrics (kept out of reports/)."""
    names = metric_names(cfg)
    sid = cfg.prep.columns.series_id
    e = cfg.evaluation
    overall = score(predictions, [MODE, MODEL, SPLIT_COLUMN], names)
    per_step = score(predictions, [MODE, MODEL, SPLIT_COLUMN, STEP], names)
    per_series = score(predictions, [MODE, MODEL, SPLIT_COLUMN, sid], names)
    dm_series, dm_panel = diebold_mariano(predictions, cfg)
    tables = {
        "overall": overall,
        "per_horizon": per_step,
        "per_series_distribution": report.distribution(per_series, names, e.percentiles),
        "wins": report.wins(per_series, e.main_models, cfg.metrics.primary, sid),
        "dm_per_series_summary": dm_series,
        "dm_panel": dm_panel,
        "sensitivity": _sensitivity(overall, names),
    }
    return tables, per_series


def run_evaluation(cfg: Config) -> EvaluationResult:
    """Run the full Phase 5 evaluation and write predictions, tables and plots."""
    e = cfg.evaluation
    panel = load_panel(cfg)
    results = run_backtests(cfg, panel)
    predictions = pd.concat([r.predictions for r in results], ignore_index=True)
    validate_predictions(predictions, panel, cfg)

    out_dir = Path(e.output_dir)
    reports_dir = Path(cfg.output.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    pred_path = out_dir / e.predictions_file
    predictions.to_parquet(pred_path, index=False)
    files.append(pred_path)

    tables, per_series = aggregate(predictions, cfg)
    series_path = out_dir / e.per_series_file
    per_series.to_csv(series_path, index=False)
    files.append(series_path)

    timing = _timing(results, panel, cfg)
    tables["timing"] = timing
    for key, table in tables.items():
        if table.empty:
            continue
        path = reports_dir / f"{e.report_prefix}_{key}.csv"
        table.round(report.FLOAT_DIGITS).to_csv(path, index=False)
        files.append(path)

    primary = cfg.metrics.primary
    models = [*e.main_models, *e.baselines]
    files.append(
        report.plot_overall(tables["overall"], primary, out_dir / f"overall_{primary}.png")
    )
    for split in e.splits:
        files.append(
            report.plot_per_horizon(
                tables["per_horizon"], primary, split, out_dir / f"horizon_{primary}_{split}.png"
            )
        )
        files.append(
            report.plot_series_boxplot(
                per_series, primary, split, models, out_dir / f"series_{primary}_{split}.png"
            )
        )

    md_path = reports_dir / f"{e.report_prefix}_evaluation.md"
    md_path.write_text(_markdown(tables, cfg, panel), encoding="utf-8")
    files.append(md_path)

    meta_path = out_dir / "evaluation_meta.json"
    meta = {
        "origins": {k: [v[0], v[-1], len(v)] for k, v in split_origins(panel, cfg).items()},
        "horizon": cfg.prep.max_prediction_length,
        "n_prediction_rows": len(predictions),
        "metrics": metric_names(cfg),
        "dm": {
            "split": e.dm_split,
            "loss": e.dm_loss,
            "alpha": e.dm_alpha,
            "fdr": e.dm_fdr_method,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files.append(meta_path)
    return EvaluationResult(tables, timing, files)


def _markdown(tables: dict[str, pd.DataFrame], cfg: Config, panel: pd.DataFrame) -> str:
    e = cfg.evaluation
    origins = split_origins(panel, cfg)
    origin_text = ", ".join(
        f"{k}: time_idx {v[0]}–{v[-1]} ({len(v)} origin)" for k, v in origins.items()
    )
    main = tables["overall"][tables["overall"][MODE] == MAIN].drop(columns=MODE)
    step = tables["per_horizon"]
    step = step[(step[MODE] == MAIN) & step[MODEL].isin(e.main_models)].drop(columns=MODE)
    dist = tables["per_series_distribution"]
    dist = dist[(dist[MODE] == MAIN) & (dist["metric"] == cfg.metrics.primary)].drop(columns=MODE)
    sections = [
        "# Fase 5 — Hasil Evaluasi (agregat)",
        "Berkas ini dibuat otomatis oleh `python -m ta_forecasting.cli evaluate`. "
        "Hanya berisi statistik agregat.",
        f"Protokol: rolling origin, horizon {cfg.prep.max_prediction_length} kuartal, "
        f"origin sama untuk semua model ({origin_text}). Hasil utama memakai "
        "`refit_statistical = false`. Catatan: val dipakai TFT untuk early stopping, "
        "sehingga skor val TFT tidak sepenuhnya out-of-sample; perbandingan utama ada di test.",
        "## Metrik keseluruhan (pooled seri × origin × horizon)",
        report.to_markdown(main),
        "## Metrik per horizon (model utama)",
        report.to_markdown(step),
        f"## Distribusi {cfg.metrics.primary.upper()} per seri",
        report.to_markdown(dist),
        f"## Jumlah seri yang dimenangkan ({cfg.metrics.primary.upper()} terendah)",
        report.to_markdown(tables["wins"]),
        f"## Diebold-Mariano (HLN), split {e.dm_split}, loss {e.dm_loss}, α = {e.dm_alpha}",
        "Tingkat panel (loss dirata-rata antar seri per origin):",
        report.to_markdown(tables["dm_panel"]),
        f"Per seri (jumlah seri signifikan; kolom `_fdr` dengan koreksi {e.dm_fdr_method}):",
        report.to_markdown(tables["dm_per_series_summary"]),
    ]
    if not tables["sensitivity"].empty:
        sections += [
            "## Analisis sensitivitas: `refit_statistical = true` (terpisah dari hasil utama)",
            report.to_markdown(tables["sensitivity"]),
        ]
    sections += ["## Waktu komputasi backtest", report.to_markdown(tables["timing"])]
    return "\n\n".join(sections) + "\n"
