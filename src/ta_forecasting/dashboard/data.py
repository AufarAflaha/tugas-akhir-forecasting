"""Data access and table building for the dashboard (pure functions, no Streamlit).

Inputs: the processed panel (actuals only), the Phase 7a forecasts and summary, the interim
``Cost_Model_Desc`` and ``ID_lookup`` sheets, and the aggregate Phase 5 / 7a reports. The raw
workbook is never read and nothing leaves the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ta_forecasting.config import Config, DashboardConfig
from ta_forecasting.data.prepare import QUARTER_LABEL
from ta_forecasting.evaluation.backtest import MODE, MODEL
from ta_forecasting.forecasting import summary as fs
from ta_forecasting.forecasting.horizon import total_steps
from ta_forecasting.forecasting.intervals import LOWER, UPPER
from ta_forecasting.forecasting.long_backtest import ALL_ORIGINS, ORIGIN_SET
from ta_forecasting.forecasting.run import VALIDATED
from ta_forecasting.models.base import PREDICTION, STEP, check_no_projection

YEAR = "year"
QUARTER_END = "quarter_end"
KIND = "kind"
ACTUAL_KIND = "aktual"
QOQ = "qoq_pct"
MISSING = "–"


class DashboardDataError(ValueError):
    """A dashboard input is missing or inconsistent."""


@dataclass(frozen=True)
class DashboardData:
    """Everything the dashboard shows."""

    actual: pd.DataFrame        # ID, quarter_label, target, Concept, Region
    forecast: pd.DataFrame      # model, ID, step, quarter_label, prediction, lower, upper, ...
    summary: pd.DataFrame       # Phase 7a per-series summary
    errors: pd.DataFrame        # Phase 7a MAPE etc. per model and step (all origins)
    coverage: pd.DataFrame      # Phase 7a interval coverage (aggregate)
    phase5: pd.DataFrame        # Phase 5 overall metrics (aggregate)
    meta: pd.DataFrame          # ID, Concept, Region, label, currency, description


def input_paths(cfg: Config) -> dict[str, Path]:
    processed, interim = Path(cfg.data.processed_dir), Path(cfg.data.interim_dir)
    reports = Path(cfg.output.reports_dir)
    f, e = cfg.forecast, cfg.evaluation
    return {
        "panel": processed / cfg.prep.panel_file,
        "forecast": processed / f.forecast_file,
        "summary": processed / f.summary_file,
        "description": interim / f"{cfg.dashboard.description_dataset}.parquet",
        "lookup": interim / f"{cfg.prep.lookup_dataset}.parquet",
        "errors": reports / f"{f.report_prefix}_long_horizon.csv",
        "coverage": reports / f"{f.report_prefix}_coverage.csv",
        "phase5": reports / f"{e.report_prefix}_overall.csv",
    }


def load_data(cfg: Config) -> DashboardData:
    """Read and check all dashboard inputs."""
    paths = input_paths(cfg)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise DashboardDataError(
            "Berkas belum ada: " + ", ".join(missing)
            + ". Jalankan `prep`, `train`, `evaluate`, lalu `forecast` terlebih dahulu.")
    c = cfg.prep.columns
    d = cfg.dashboard
    panel = pd.read_parquet(paths["panel"])
    check_no_projection(panel, cfg)
    actual = panel[[c.series_id, QUARTER_LABEL, cfg.prep.target_column, c.concept, c.region]]

    forecast = pd.read_parquet(paths["forecast"])
    last = pd.Period(cfg.project.last_actual_quarter, freq="Q")
    end = pd.Period(cfg.forecast.end_quarter, freq="Q")
    q = pd.PeriodIndex(forecast[QUARTER_LABEL].astype(str), freq="Q")
    if (q <= last).any() or (q > end).any():
        raise DashboardDataError("Ramalan berada di luar rentang setelah kuartal aktual terakhir.")
    if set(forecast[c.series_id]) != set(actual[c.series_id]):
        raise DashboardDataError("Seri pada ramalan dan data aktual tidak sama.")

    errors = pd.read_csv(paths["errors"])
    errors = errors[errors[ORIGIN_SET] == ALL_ORIGINS].drop(columns=ORIGIN_SET)
    phase5 = pd.read_csv(paths["phase5"])
    phase5 = phase5[phase5[MODE] == cfg.report.main_mode].drop(columns=MODE)

    lookup = pd.read_parquet(paths["lookup"])
    lookup = lookup[[c.lookup_id, d.lookup_label, d.lookup_currency]].drop_duplicates(
        c.lookup_id).rename(columns={c.lookup_id: c.series_id})
    desc = pd.read_parquet(paths["description"])
    desc = desc[[d.description_id, d.description_text]].drop_duplicates(d.description_id)
    desc = desc.rename(columns={d.description_id: c.series_id})
    meta = (actual[[c.series_id, c.concept, c.region]].drop_duplicates(c.series_id)
            .merge(lookup, on=c.series_id, how="left")
            .merge(desc, on=c.series_id, how="left")
            .sort_values([c.concept, c.region]).reset_index(drop=True))
    return DashboardData(actual, forecast, pd.read_parquet(paths["summary"]), errors,
                         pd.read_csv(paths["coverage"]), phase5, meta)


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def concepts(data: DashboardData, cfg: Config) -> list[str]:
    return sorted(data.meta[cfg.prep.columns.concept].unique())


def regions(data: DashboardData, concept: str, cfg: Config) -> list[str]:
    c = cfg.prep.columns
    return sorted(data.meta.loc[data.meta[c.concept] == concept, c.region].unique())


def series_id(data: DashboardData, concept: str, region: str, cfg: Config) -> str:
    c = cfg.prep.columns
    rows = data.meta[(data.meta[c.concept] == concept) & (data.meta[c.region] == region)]
    if len(rows) != 1:
        raise DashboardDataError(f"Cost model {concept!r} / {region!r} tidak unik.")
    return str(rows[c.series_id].iloc[0])


def base_years(data: DashboardData) -> list[int]:
    """Years with four actual quarters (valid rebasing years)."""
    years = pd.PeriodIndex(data.actual[QUARTER_LABEL].drop_duplicates(), freq="Q").year
    counts = pd.Series(years).value_counts()
    return sorted(int(y) for y, n in counts.items() if n == 4)


def year_bounds(data: DashboardData, cfg: Config) -> tuple[int, int]:
    first = pd.Period(data.actual[QUARTER_LABEL].min(), freq="Q").year
    return int(first), pd.Period(cfg.forecast.end_quarter, freq="Q").year


# ---------------------------------------------------------------------------
# One series
# ---------------------------------------------------------------------------


def _quarter_frame(labels: pd.Series) -> pd.DataFrame:
    q = pd.PeriodIndex(labels.astype(str), freq="Q")
    return pd.DataFrame({YEAR: q.year, QUARTER_END: q.to_timestamp(how="end").normalize()})


def series_view(data: DashboardData, sid: str, models: list[str], base_year: int,
                cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebased actuals (with QoQ %) and rebased forecasts of the selected models.

    Rebasing multiplies every value (actual, forecast, interval) of the series by
    ``100 / mean(actual in base_year)``; percentage changes are unaffected.
    """
    c = cfg.prep.columns
    act = data.actual[data.actual[c.series_id] == sid].sort_values(QUARTER_LABEL)
    act = act[[QUARTER_LABEL, cfg.prep.target_column]].rename(
        columns={cfg.prep.target_column: PREDICTION}).reset_index(drop=True)
    act = pd.concat([act, _quarter_frame(act[QUARTER_LABEL])], axis=1)
    base = act.loc[act[YEAR] == base_year, PREDICTION]
    if len(base) != 4:
        raise DashboardDataError(f"Tahun dasar {base_year} tidak memiliki 4 kuartal aktual.")
    factor = 100.0 / float(base.mean())
    act[PREDICTION] *= factor
    act[QOQ] = act[PREDICTION].pct_change() * 100.0
    act[KIND] = ACTUAL_KIND

    fc = data.forecast[(data.forecast[c.series_id] == sid) & data.forecast[MODEL].isin(models)]
    fc = fc[[MODEL, STEP, QUARTER_LABEL, PREDICTION, LOWER, UPPER, VALIDATED]].copy()
    fc[[PREDICTION, LOWER, UPPER]] *= factor
    hide = fc[STEP] > cfg.dashboard.interval_max_step
    fc.loc[hide, [LOWER, UPPER]] = np.nan
    fc = pd.concat([fc.reset_index(drop=True), _quarter_frame(fc[QUARTER_LABEL])
                    .reset_index(drop=True)], axis=1)
    rank = fc[MODEL].map({m: i for i, m in enumerate(models)})
    fc = fc.assign(_r=rank).sort_values(["_r", STEP]).drop(columns="_r")
    return act, fc.reset_index(drop=True)


def comparison_table(data: DashboardData, sid: str, models: list[str], cfg: Config,
                     labels: dict[str, str]) -> pd.DataFrame:
    """Per selected model: MAPE at chosen steps, cheapest quarter, change, agreement."""
    d, c = cfg.dashboard, cfg.prep.columns
    primary = cfg.metrics.primary
    err = data.errors.set_index([MODEL, STEP])[primary]
    summ = data.summary[data.summary[c.series_id] == sid].set_index([MODEL, fs.WINDOW])
    # agreement among the selected, non-flat models only (matches the overview page)
    picked = summ[~summ[fs.FLAT].astype(bool) & summ.index.get_level_values(MODEL).isin(models)]
    agree = picked.groupby([fs.WINDOW, fs.CHEAPEST_QUARTER]).size()
    n_window = picked.groupby(fs.WINDOW).size()
    rows = []
    for m in models:
        row: dict[str, object] = {"Model": labels.get(m, m)}
        for h in d.error_steps:
            value = err.get((m, h), np.nan)
            row[f"MAPE h={h} (%)"] = MISSING if pd.isna(value) else fmt_num(value, d)
        windows = ((fs.SHORT, "1 tahun"), (fs.FULL, f"s.d. {cfg.forecast.end_quarter}"))
        for window, title in windows:
            if (m, window) not in summ.index:
                row.update({f"Termurah ({title})": MISSING, f"Selisih ({title})": MISSING,
                            f"Bermakna? ({title})": MISSING})
                continue
            s = summ.loc[(m, window)]
            if bool(s[fs.FLAT]):
                row[f"Termurah ({title})"] = "datar"
                row[f"Selisih ({title})"] = fmt_signed_pct(s[fs.CHANGE_PCT], d)
                row[f"Bermakna? ({title})"] = MISSING
                continue
            same = int(agree.get((window, s[fs.CHEAPEST_QUARTER]), 0))
            row[f"Termurah ({title})"] = (
                f"{s[fs.CHEAPEST_QUARTER]} ({same}/{int(n_window.get(window, 0))} model)")
            if window == fs.FULL and int(s["n_steps"]) < total_steps(cfg):
                row[f"Termurah ({title})"] += f", ramalan hanya {int(s['n_steps'])} kuartal"
            row[f"Selisih ({title})"] = fmt_signed_pct(s[fs.CHANGE_PCT], d)
            row[f"Bermakna? ({title})"] = (
                "tidak (selisih < galat)" if bool(s[fs.WITHIN_ERROR]) else "ya")
        rows.append(row)
    return pd.DataFrame(rows)


def annual_table(actual: pd.DataFrame, forecast: pd.DataFrame, start: int, end: int,
                 labels: dict[str, str], cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Annual mean index and YoY % (rows) by year (columns), like the S&P annual sheet.

    Years after the last actual quarter combine actual quarters of that year with the
    model's forecast quarters; a year without four quarters is shown as '–'. Returns the
    table and the list of forecast-year columns.
    """
    d = cfg.dashboard
    last_q = pd.Period(cfg.project.last_actual_quarter, freq="Q")
    first_fc_year = (last_q + 1).year  # first year that holds forecast quarters
    years = list(range(start, end + 1))
    stats = actual.groupby(YEAR)[PREDICTION].agg(["mean", "size"])
    actual_means = stats.loc[stats["size"] == 4, "mean"]
    rows: list[list[str]] = []

    def add(name: str, means: pd.Series, shown: set[int]) -> None:
        """Value row and YoY row; YoY uses the previous year even if it is not shown."""
        values, yoy = [], []
        for y in years:
            v, prev = means.get(y, np.nan), means.get(y - 1, np.nan)
            ok = y in shown
            values.append(fmt_num(v, d) if ok and pd.notna(v) else MISSING)
            change = (v / prev - 1) * 100 if pd.notna(v) and pd.notna(prev) else np.nan
            yoy.append(fmt_signed_pct(change, d) if ok and pd.notna(change) else MISSING)
        rows.append([name, *values])
        rows.append(["  perubahan YoY", *yoy])

    add("Aktual", actual_means, {int(y) for y in actual_means.index})
    partial = actual.loc[actual[YEAR] >= first_fc_year, [YEAR, PREDICTION]]
    for model, g in forecast.groupby(MODEL, sort=False):
        quarters = pd.concat([partial, g[[YEAR, PREDICTION]]])
        per_year = quarters.groupby(YEAR)[PREDICTION].agg(["mean", "size"])
        fc_means = per_year.loc[per_year["size"] == 4, "mean"]
        means = pd.concat([actual_means[actual_means.index < first_fc_year], fc_means])
        add(f"Ramalan {labels.get(model, model)}", means,
            {int(y) for y in fc_means.index if y >= first_fc_year})
    columns = ["Baris", *[str(y) for y in years]]
    forecast_cols = [str(y) for y in years if y >= first_fc_year]
    return pd.DataFrame(rows, columns=columns), forecast_cols


# ---------------------------------------------------------------------------
# Overview of all series
# ---------------------------------------------------------------------------


def overview_table(data: DashboardData, window: str, models: list[str], cfg: Config,
                   labels: dict[str, str]) -> pd.DataFrame:
    """One row per cost model: cheapest quarter and change per model, plus consensus."""
    c, d = cfg.prep.columns, cfg.dashboard
    summ = data.summary[(data.summary[fs.WINDOW] == window) & data.summary[MODEL].isin(models)]
    base = data.meta[[c.series_id, c.concept, c.region]].copy()
    for m in models:
        part = summ[summ[MODEL] == m].set_index(c.series_id)
        name = labels.get(m, m)
        quarter = part[fs.CHEAPEST_QUARTER].where(~part[fs.FLAT], "datar")
        base[f"{name}: termurah"] = base[c.series_id].map(quarter).fillna(MISSING)
        base[f"{name}: selisih (%)"] = base[c.series_id].map(part[fs.CHANGE_PCT]).round(
            d.table_decimals)
        within = part[fs.WITHIN_ERROR].map(
            lambda x: MISSING if pd.isna(x) else ("tidak" if x else "ya"))
        base[f"{name}: bermakna?"] = base[c.series_id].map(within).fillna(MISSING)
    cons = _consensus(summ, c.series_id)
    base["Kesepakatan"] = base[c.series_id].map(cons).fillna("tidak ada")
    return base.drop(columns=c.series_id).rename(
        columns={c.concept: "Cost model", c.region: "Region"})


def _consensus(summ: pd.DataFrame, sid: str) -> pd.Series:
    """Quarter picked by most of the selected non-flat models (needs at least two)."""
    picked = summ[~summ[fs.FLAT]]
    out = {}
    for key, g in picked.groupby(sid):
        counts = g[fs.CHEAPEST_QUARTER].value_counts()
        top = counts[counts == counts.max()].sort_index()
        if counts.max() >= fs.MIN_CONSENSUS:
            out[key] = f"{top.index[0]} ({int(counts.max())}/{len(g)} model)"
    return pd.Series(out, dtype=object)


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------


def accuracy_table(data: DashboardData, cfg: Config, labels: dict[str, str]) -> pd.DataFrame:
    """Phase 5 test-split metrics of every model (aggregate)."""
    d = cfg.dashboard
    part = data.phase5[data.phase5["split"] == cfg.report.split].copy()
    order = {m: i for i, m in enumerate(cfg.report.model_order)}
    part = part.sort_values(MODEL, key=lambda s: s.map(order))
    out = pd.DataFrame({"Model": part[MODEL].map(lambda m: labels.get(m, m))})
    for name in [cfg.metrics.primary, *cfg.metrics.secondary]:
        out[cfg.report.metric_labels.get(name, name.upper())] = part[name].map(
            lambda v: fmt_num(v, d)).to_numpy()
    return out.reset_index(drop=True)


def fmt_num(value: float, d: DashboardConfig) -> str:
    return f"{float(value):.{d.table_decimals}f}".replace(".", ",")


def fmt_signed_pct(value: float, d: DashboardConfig) -> str:
    v = float(value)
    return ("+" if v >= 0 else "-") + fmt_num(abs(v), d) + "%"
