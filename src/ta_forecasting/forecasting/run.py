"""Phase 7a entry point: long-horizon backtest -> intervals -> production forecasts.

Series-level outputs (forecasts, per-series summary) go to ``data.processed_dir`` and are not
committed. Only aggregate tables go to ``output.reports_dir``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import QUARTER_LABEL, TIME_IDX
from ta_forecasting.evaluation.backtest import MODEL
from ta_forecasting.evaluation.evaluate import load_model, load_panel
from ta_forecasting.evaluation.report import to_markdown
from ta_forecasting.forecasting import intervals, summary
from ta_forecasting.forecasting.horizon import (
    check_panel_ends_at_last_actual,
    future_quarters,
    model_steps,
    total_steps,
    validated_steps,
)
from ta_forecasting.forecasting.long_backtest import (
    ALL_ORIGINS,
    N_ORIGINS,
    ORIGIN_SET,
    horizon_metrics,
    run_long_backtest,
)
from ta_forecasting.models.base import ORIGIN, PREDICTION, STEP, check_no_projection

logger = logging.getLogger(__name__)

VALIDATED = "validated"


@dataclass(frozen=True)
class ForecastResult:
    """Everything written by ``forecast`` plus the tables (for tests and the CLI)."""

    forecast: pd.DataFrame
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    files: list[Path]
    seconds: dict[str, float]


def production_forecast(cfg: Config, panel: pd.DataFrame) -> pd.DataFrame:
    """Forecast from all actuals up to ``last_actual_quarter`` with every configured model."""
    check_no_projection(panel, cfg)
    last_idx = check_panel_ends_at_last_actual(panel, cfg)
    sid = cfg.prep.columns.series_id
    frames = []
    for name in cfg.forecast.models:
        model = load_model(name, cfg, panel)
        if model.cfg.rolling.refit_statistical:
            raise ValueError(f"{name}: production forecasts require refit_statistical = false.")
        steps = model_steps(name, cfg)
        out = model.forecast(panel, steps)
        out.insert(0, MODEL, name)
        frames.append(out)
    fc = pd.concat(frames, ignore_index=True)
    quarters = future_quarters(last_idx, total_steps(cfg), cfg)
    fc = fc.merge(quarters, on=TIME_IDX, how="left", validate="m:1")
    if fc[QUARTER_LABEL].isna().any() or (fc[ORIGIN] != last_idx + 1).any():
        raise ValueError("Production forecasts do not start right after the last actual.")
    if not np.isfinite(fc[PREDICTION].to_numpy(dtype=float)).all():
        raise ValueError("Production forecasts are not finite.")
    n_series = panel[sid].nunique()
    for name, group in fc.groupby(MODEL):
        if len(group) != n_series * model_steps(name, cfg):
            raise ValueError(f"{name}: incomplete production forecast.")
    fc[VALIDATED] = fc[STEP] <= validated_steps(cfg)
    return fc


def run_forecast(cfg: Config) -> ForecastResult:
    """Run Phase 7a and write forecasts, summary and aggregate report tables."""
    f = cfg.forecast
    panel = load_panel(cfg)
    sid = cfg.prep.columns.series_id
    seconds: dict[str, float] = {}

    start = time.perf_counter()
    backtest = run_long_backtest(cfg, panel)
    seconds["long_backtest"] = time.perf_counter() - start
    metrics = horizon_metrics(backtest, cfg)
    quantiles = intervals.ratio_quantiles(backtest, cfg)
    coverage = intervals.coverage_check(backtest, cfg)

    start = time.perf_counter()
    fc = production_forecast(cfg, panel)
    fc = intervals.apply_intervals(fc, quantiles)
    seconds["production_forecast"] = time.perf_counter() - start

    last_actual = panel.sort_values(TIME_IDX).groupby(sid).tail(1).set_index(sid)[
        cfg.prep.target_column]
    summ = summary.summarize(fc, last_actual, metrics, cfg)
    tables = {
        "long_horizon": metrics,
        "intervals": quantiles,
        "coverage": coverage,
        "cheapest_counts": summary.cheapest_counts(summ, cfg),
        "summary_stats": summary.summary_stats(summ, cfg),
    }

    processed = Path(cfg.data.processed_dir)
    reports = Path(cfg.output.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    files = []
    fc_path = processed / f.forecast_file
    fc.to_parquet(fc_path, index=False)
    sum_path = processed / f.summary_file
    summ.to_parquet(sum_path, index=False)
    files += [fc_path, sum_path]
    for key, table in tables.items():
        path = reports / f"{f.report_prefix}_{key}.csv"
        table.round(4).to_csv(path, index=False)
        files.append(path)
    md = reports / f"{f.report_prefix}_long_horizon.md"
    md.write_text(_markdown(tables, cfg, panel, seconds), encoding="utf-8")
    files.append(md)
    logger.info("Forecast: %d rows, summary %d rows", len(fc), len(summ))
    return ForecastResult(fc, summ, tables, files, seconds)


LABELS = {"step": "h", "n_origin": "n origin", "origins": "split origin", "model": "model",
          "window": "jendela", "ratio_lo": "rasio_lo", "ratio_hi": "rasio_hi", "n": "n"}


def _md(frame: pd.DataFrame, digits: int) -> str:
    """Markdown table with fixed decimals and '–' for missing cells."""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "–" if pd.isna(x) else f"{x:.{digits}f}")
    return to_markdown(out.rename(columns=lambda c: LABELS.get(c, c)))


def _findings(tables: dict[str, pd.DataFrame], cfg: Config) -> list[str]:
    """Observations derived from the tables (only stated when the numbers show them)."""
    f, v = cfg.forecast, validated_steps(cfg)
    out = []
    share = 100.0 * (1 + f.interval_level) / 2
    q = tables["intervals"]
    above = q[q[intervals.Q_LO] > 1]
    below = q[q[intervals.Q_HI] < 1]
    for rule, part, side in (("rasio_lo > 1", above, "di atas"),
                             ("rasio_hi < 1", below, "di bawah")):
        for model, g in part.groupby(MODEL):
            steps = ", ".join(str(int(h)) for h in sorted(g[STEP]))
            out.append(
                f"- {model}: pada h = {steps} seluruh interval berada {side} ramalan titik "
                f"({rule}). Pada backtest, setidaknya {share:.0f}% aktual di horizon ini "
                f"berada {side} ramalan, jadi galatnya bias satu arah, bukan acak.")
    cov = tables["coverage"]
    nominal = 100.0 * f.interval_level
    low = cov[cov[intervals.COVERAGE] < nominal - 10]
    for model, g in low.groupby(MODEL):
        out.append(f"- {model}: cakupan uji lebih dari 10 poin di bawah nominal pada h = "
                   + ", ".join(str(int(h)) for h in sorted(g[STEP])) + ".")
    m = tables["long_horizon"]
    allo = m[m[ORIGIN_SET] == ALL_ORIGINS]
    p = cfg.metrics.primary
    for model, g in allo.groupby(MODEL):
        g = g.sort_values(STEP)
        if len(g) > v and g[p].iloc[-1] < (1 - f.composition_note_drop) * g[p].max():
            peak = int(g.loc[g[p].idxmax(), STEP])
            out.append(f"- {model}: {p.upper()} tertinggi pada h = {peak} lalu turun. Horizon "
                       "panjang hanya dinilai pada origin awal (sedikit origin), jadi penurunan "
                       "ini kemungkinan efek komposisi origin, bukan model yang makin akurat.")
    return out


def _markdown(tables: dict[str, pd.DataFrame], cfg: Config, panel: pd.DataFrame,
              seconds: dict[str, float]) -> str:
    f = cfg.forecast
    p = cfg.metrics.primary
    m = tables["long_horizon"]
    v = validated_steps(cfg)
    last = int(panel[TIME_IDX].max())
    models = list(f.models)
    allo = m[m[ORIGIN_SET] == ALL_ORIGINS]
    wide = allo.pivot(index=STEP, columns=MODEL, values=p)[
        [x for x in models if x in set(allo[MODEL])]].reset_index()
    n_orig = allo.groupby(STEP)[N_ORIGINS].max().rename("n_origin").reset_index()
    wide = wide.merge(n_orig, on=STEP)
    by_split = m[(m[ORIGIN_SET] != ALL_ORIGINS) & (m[STEP] <= v)]
    split_wide = by_split.pivot_table(index=[ORIGIN_SET, STEP], columns=MODEL, values=p)
    split_wide = split_wide[[x for x in models if x in split_wide.columns]].reset_index()
    cov = tables["coverage"].pivot(index=STEP, columns=MODEL, values=intervals.COVERAGE)
    cov = cov[[x for x in models if x in cov.columns]].reset_index()
    q = tables["intervals"]
    q_short = q[q[STEP].isin([1, v, total_steps(cfg)])]
    findings = _findings(tables, cfg)
    sections = [
        f"# Fase 7a — Galat per horizon dan ramalan sampai {f.end_quarter} (agregat)",
        "Dibuat otomatis oleh `python -m ta_forecasting.cli forecast`. Hanya berisi statistik "
        "agregat; ramalan per seri ada di `data/processed/` dan tidak di-commit.",
        "Protokol sama dengan hasil utama Fase 5 (`refit_statistical = false`, model Fase 4 "
        f"hanya di-load). Origin: setiap kuartal dari awal split {f.backtest_first_split} "
        "sampai kuartal aktual terakhir; horizon h dinilai pada origin yang targetnya masih "
        "aktual, sehingga jumlah origin turun untuk h besar. Horizon 1–"
        f"{v} adalah horizon tervalidasi Fase 5; di atasnya adalah jangka panjang. TFT hanya "
        f"sampai h = {v}. Origin pada split validasi tidak sepenuhnya out-of-sample untuk TFT "
        "(early stopping).",
        f"Angka h = 1–{v} di sini berbeda dari Fase 5 karena himpunan origin berbeda "
        "(Fase 5: origin dan target di dalam split yang sama). Hasil Fase 5 tidak berubah.",
        f"## {p.upper()} (%) per horizon, semua origin",
        _md(wide, 2),
        f"## {p.upper()} (%) per horizon menurut split origin (h ≤ {v})",
        _md(split_wide, 2),
        f"## Interval prediksi empiris {f.interval_level:.0%} (rasio aktual / ramalan)",
        "Interval = ramalan × [rasio_lo, rasio_hi], per model dan horizon, dari semua origin "
        f"backtest, digabung antar seri. Cuplikan h = 1, {v}, dan {total_steps(cfg)}; tabel "
        f"lengkap di `{f.report_prefix}_intervals.csv`.",
        _md(q_short, 4),
        "## Uji cakupan interval (out-of-sample)",
        f"Kuantil dihitung hanya dari origin {f.calibration_split}, lalu cakupan (%) diukur "
        f"pada origin {f.coverage_split}. Nominal {f.interval_level:.0%}. Hanya horizon yang "
        "memiliki origin di kedua split.",
        _md(cov, 1),
        "## Catatan pembacaan",
        "\n".join(findings) if findings else "Tidak ada catatan khusus.",
        "## Ringkasan kuartal termurah (jumlah seri)",
        f"`short` = h 1–{v} (semua model), `full` = semua langkah tiap model sampai "
        f"{f.end_quarter} (TFT tetap {v} langkah). `n_within_error` = seri yang selisih "
        "ramalan termurah terhadap aktual terakhir lebih kecil dari "
        f"{p.upper()} model pada horizon itu. Naive datar sehingga tidak memiliki kuartal "
        "termurah. `n_agree_with_consensus` = seri yang kuartal termurahnya sama dengan kuartal "
        "yang paling banyak dipilih model.",
        _md(tables["summary_stats"], 2),
        "## Waktu komputasi (detik)",
        _md(pd.DataFrame([{k: round(s, 1) for k, s in seconds.items()}]), 1),
        f"Kuartal aktual terakhir: {cfg.project.last_actual_quarter} (time_idx {last}). "
        "Proyeksi S&P setelahnya tidak dibaca.",
    ]
    return "\n\n".join(sections) + "\n"
