"""Build ``FINAL_REPORT.md`` (Indonesian) from the validated report inputs.

Every number in the text is read from :class:`ReportInputs` or derived from it (ranks,
differences, ratios). Interpretive sentences are chosen by conditions on those numbers, so
the text follows the results instead of assuming them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN, TRAIN
from ta_forecasting.evaluation.backtest import MODE, MODEL
from ta_forecasting.evaluation.metrics import metric_names
from ta_forecasting.models.base import STEP
from ta_forecasting.reporting.formatting import Formatter, markdown_table
from ta_forecasting.reporting.inputs import (
    PAIR_A,
    PAIR_B,
    SENSITIVITY_KEY,
    ReportInputs,
    all_models,
    main_models,
    model_pairs,
)

NAIVE = "naive"
SEASONAL_NAIVE = "seasonal_naive"
TFT = "tft"
ETS = "ets"
ARIMA = "arima"
CLI_STEPS = (
    ("extract", "Ekstraksi workbook (Fase 2)"),
    ("profile", "Profil data dan kamus data (Fase 2)"),
    ("transform", "Transformasi ke Parquet interim (Fase 2)"),
    ("eda", "Grafik EDA (Fase 2)"),
    ("prep", "Panel, split temporal, scaler, dataset TFT (Fase 3)"),
    ("train --model all", "Pelatihan ETS, ARIMA, TFT (Fase 4)"),
    ("evaluate", "Evaluasi rolling origin (Fase 5)"),
    ("report", "Laporan akhir dan grafik (Fase 6)"),
)


@dataclass(frozen=True)
class Table:
    """A numbered report table: caption plus formatted cells."""

    key: str
    caption: str
    frame: pd.DataFrame


class View:
    """Lookup helpers over the input tables."""

    def __init__(self, inputs: ReportInputs, cfg: Config) -> None:
        self.inputs, self.cfg = inputs, cfg
        self.t = inputs.tables
        self.r = cfg.report
        self.names = metric_names(cfg)
        self.primary = cfg.metrics.primary
        self.horizon = cfg.prep.max_prediction_length
        self.steps = list(range(1, self.horizon + 1))

    def overall(self, model: str, split: str, mode: str | None = None) -> pd.Series:
        o = self.t["overall"]
        mode = mode or self.r.main_mode
        return o[(o[MODE] == mode) & (o[MODEL] == model) & (o[SPLIT_COLUMN] == split)].iloc[0]

    def value(self, model: str, split: str, metric: str | None = None,
              mode: str | None = None) -> float:
        return float(self.overall(model, split, mode)[metric or self.primary])

    def step(self, model: str, split: str, h: int, metric: str | None = None) -> float:
        s = self.t["per_horizon"]
        row = s[(s[MODE] == self.r.main_mode) & (s[MODEL] == model)
                & (s[SPLIT_COLUMN] == split) & (s[STEP] == h)]
        return float(row.iloc[0][metric or self.primary])

    def dist(self, model: str, split: str) -> pd.Series:
        d = self.t["per_series_distribution"]
        return d[(d[MODE] == self.r.main_mode) & (d[MODEL] == model)
                 & (d[SPLIT_COLUMN] == split) & (d["metric"] == self.primary)].iloc[0]

    def wins(self, split: str) -> pd.Series:
        w = self.t["wins"]
        return w[(w[SPLIT_COLUMN] == split) & (w["metric"] == self.primary)].iloc[0]

    def dm(self, a: str, b: str, h: int) -> pd.Series:
        d = self.t["dm_panel"]
        return d[(d[PAIR_A] == a) & (d[PAIR_B] == b) & (d[STEP] == h)].iloc[0]

    def dm_series(self, a: str, b: str, h: int) -> pd.Series:
        d = self.t["dm_per_series_summary"]
        return d[(d[PAIR_A] == a) & (d[PAIR_B] == b) & (d[STEP] == h)].iloc[0]

    def timing(self, model: str, mode: str) -> float | None:
        t = self.t["timing"]
        row = t[(t[MODEL] == model) & (t[MODE] == mode)]
        return None if row.empty else float(row.iloc[0]["seconds"])

    def ranking(self, split: str, models: list[str], mode: str | None = None) -> list[str]:
        return sorted(models, key=lambda m: self.value(m, split, mode=mode))

    def split_info(self, split: str) -> dict[str, Any]:
        return self.inputs.split_summary["splits"][split]

    def n_origins(self, split: str) -> int:
        """Origins per split, from the prediction count (n = series × origins × horizon)."""
        n = int(self.overall(main_models(self.cfg)[0], split)["n"])
        return n // (self.inputs.split_summary["n_series"] * self.horizon)

    def has_sensitivity(self) -> bool:
        return SENSITIVITY_KEY in self.t


def build_report(inputs: ReportInputs, cfg: Config, figures: dict[str, Path],
                 report_path: Path) -> tuple[str, list[Table]]:
    """Markdown text of the final report and the tables it contains."""
    v, f = View(inputs, cfg), Formatter(cfg)
    tables = build_tables(v, f)
    by_key = {t.key: t for t in tables}
    numbered = {t.key: i for i, t in enumerate(tables, start=1)}
    fig_no = {k: i for i, k in enumerate(figures, start=1)}

    def table(key: str) -> str:
        t = by_key[key]
        return f"**Tabel {numbered[key]}.** {t.caption}\n\n{markdown_table(t.frame)}"

    def figure(key: str, caption: str) -> str:
        rel = Path(os.path.relpath(figures[key], report_path.parent)).as_posix()
        return f"![{caption}]({rel})\n\n**Gambar {fig_no[key]}.** {caption}"

    ctx = _Context(v, f, table, figure, lambda k: f"Tabel {numbered[k]}",
                   lambda k: f"Gambar {fig_no[k]}")
    sections = [
        _header(ctx, inputs),
        _summary(ctx),
        _design(ctx),
        _results(ctx),
        _discussion(ctx),
        _answers(ctx),
        _limitations(ctx),
        _reproducibility(ctx),
        _appendix(ctx, figures),
    ]
    return "\n\n".join(s.strip() for s in sections) + "\n", tables


@dataclass(frozen=True)
class _Context:
    v: View
    f: Formatter
    table: Any
    figure: Any
    tref: Any
    fref: Any


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_tables(v: View, f: Formatter) -> list[Table]:
    """All report tables in order of appearance."""
    cfg, r = v.cfg, v.r
    models = all_models(cfg)
    mains = main_models(cfg)
    other = [s for s in cfg.evaluation.splits if s != r.split]
    out = [Table("split", "Pembagian data temporal.", _split_table(v, f))]
    for split in [r.split, *other]:
        out.append(Table(f"overall_{split}",
                         f"Metrik keseluruhan pada data {f.split(split).lower()} "
                         "(gabungan seri × origin × horizon, mode utama).",
                         _overall_table(v, f, split, models)))
    for split in [r.split, *other]:
        out.append(Table(f"horizon_{split}",
                         f"{f.metric(v.primary)} per horizon pada data {f.split(split).lower()}.",
                         _horizon_pivot(v, f, split, models)))
    out.append(Table(f"horizon_all_{r.split}",
                     f"Semua metrik per horizon, model utama, data {f.split(r.split).lower()}.",
                     _horizon_long(v, f, r.split, mains)))
    for split in [r.split, *other]:
        out.append(Table(f"dist_{split}",
                         f"Distribusi {f.metric_short(v.primary)} per seri "
                         f"(%; data {f.split(split).lower()}).",
                         _dist_table(v, f, split, models)))
    out.append(Table("wins", f"Jumlah seri dengan {f.metric_short(v.primary)} terendah "
                     "(hanya model utama).", _wins_table(v, f)))
    out.append(Table("dm_panel", "Uji Diebold-Mariano (HLN) tingkat panel, data "
                     f"{f.split(cfg.evaluation.dm_split).lower()}, loss "
                     f"{cfg.evaluation.dm_loss.upper()}, α = {f.num(cfg.evaluation.dm_alpha)}.",
                     _dm_panel_table(v, f)))
    out.append(Table("dm_series", "Uji Diebold-Mariano per seri: jumlah seri signifikan, "
                     f"tanpa koreksi dan dengan koreksi FDR ({cfg.evaluation.dm_fdr_method}).",
                     _dm_series_table(v, f)))
    if v.has_sensitivity():
        out.append(Table("sensitivity", "Sensitivitas `refit_statistical = true` "
                         "(Δ = refit − utama; negatif berarti refit lebih baik).",
                         _sensitivity_table(v, f)))
    out.append(Table("timing", "Waktu komputasi (detik).", _timing_table(v, f, models)))
    return out


def _split_table(v: View, f: Formatter) -> pd.DataFrame:
    rows = []
    for split in (TRAIN, *v.cfg.evaluation.splits):
        info = v.split_info(split)
        rows.append({
            "Split": f.split(split),
            "Periode": f"{info['start_quarter']}–{info['end_quarter']}",
            "Kuartal": str(info["n_quarters"]),
            "Baris (seri × kuartal)": f.count(info["n_rows"]),
            "Origin rolling": "–" if split == TRAIN else str(v.n_origins(split)),
        })
    return pd.DataFrame(rows)


def _overall_table(v: View, f: Formatter, split: str, models: list[str]) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(v.ranking(split, models), start=1)}
    rows = []
    for m in models:
        row = v.overall(m, split)
        cells = {"Model": f.model(m)}
        cells.update({f.metric(n): f.num(row[n]) for n in v.names})
        cells["n prediksi"] = f.count(row["n"])
        cells[f"Peringkat {f.metric_short(v.primary)}"] = str(rank[m])
        rows.append(cells)
    return pd.DataFrame(rows)


def _horizon_pivot(v: View, f: Formatter, split: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        cells = {"Model": f.model(m)}
        cells.update({f"h = {h}": f.num(v.step(m, split, h)) for h in v.steps})
        rows.append(cells)
    return pd.DataFrame(rows)


def _horizon_long(v: View, f: Formatter, split: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        for h in v.steps:
            cells = {"Model": f.model(m), "Horizon": str(h)}
            cells.update({f.metric(n): f.num(v.step(m, split, h, n)) for n in v.names})
            rows.append(cells)
    return pd.DataFrame(rows)


def _dist_table(v: View, f: Formatter, split: str, models: list[str]) -> pd.DataFrame:
    pct = [f"p{q:g}" for q in v.cfg.evaluation.percentiles]
    rows = []
    for m in models:
        d = v.dist(m, split)
        cells = {"Model": f.model(m), "Rata-rata": f.num(d["mean"])}
        cells.update({p: f.num(d[p]) for p in pct})
        cells["n seri"] = str(int(d["n_series"]))
        rows.append(cells)
    return pd.DataFrame(rows)


def _wins_table(v: View, f: Formatter) -> pd.DataFrame:
    rows = []
    for split in v.cfg.evaluation.splits:
        w = v.wins(split)
        cells = {"Split": f.split(split), "n seri": str(int(w["n_series"]))}
        cells.update({f.model(m): str(int(w[m])) for m in main_models(v.cfg)})
        cells["Seimbang"] = str(int(w["tie"]))
        rows.append(cells)
    return pd.DataFrame(rows)


def _dm_conclusion(v: View, f: Formatter, row: pd.Series) -> str:
    if bool(row["significant"]):
        return f"Signifikan, {f.model(row['better'])} lebih baik"
    return "Tidak signifikan"


def _dm_panel_table(v: View, f: Formatter) -> pd.DataFrame:
    rows = []
    for a, b in model_pairs(v.cfg):
        for h in v.steps:
            d = v.dm(a, b, h)
            rows.append({
                "Pasangan (A vs B)": f"{f.model(a)} vs {f.model(b)}",
                "Horizon": str(h),
                "Rata-rata selisih loss (A − B)": f.num(d["mean_loss_diff"], 4),
                "Statistik DM-HLN": f.num(d["dm_hln"], 4),
                "p-value": f.pval(d["p_value"]),
                "n origin": str(int(d["n_origins"])),
                "Kesimpulan": _dm_conclusion(v, f, d),
            })
    return pd.DataFrame(rows)


def _dm_series_table(v: View, f: Formatter) -> pd.DataFrame:
    rows = []
    for a, b in model_pairs(v.cfg):
        for h in v.steps:
            d = v.dm_series(a, b, h)
            tested = int(d["n_series"]) - int(d["n_undefined"])
            a_fdr, b_fdr = int(d["n_sig_a_better_fdr"]), int(d["n_sig_b_better_fdr"])
            rows.append({
                "Pasangan (A vs B)": f"{f.model(a)} vs {f.model(b)}",
                "Horizon": str(h),
                "Seri diuji": str(tested),
                "A lebih baik (mentah)": str(int(d["n_sig_a_better"])),
                "B lebih baik (mentah)": str(int(d["n_sig_b_better"])),
                "A lebih baik (FDR)": str(a_fdr),
                "B lebih baik (FDR)": str(b_fdr),
                "Tidak signifikan (FDR)": str(tested - a_fdr - b_fdr),
            })
    return pd.DataFrame(rows)


def _sensitivity_table(v: View, f: Formatter) -> pd.DataFrame:
    sens = v.t[SENSITIVITY_KEY]
    order = {s: i for i, s in enumerate([v.r.split, *v.cfg.evaluation.splits])}
    rows = []
    for m in [x for x in v.r.model_order if x in v.cfg.evaluation.sensitivity_models]:
        part = sens[sens[MODEL] == m].sort_values(SPLIT_COLUMN, key=lambda s: s.map(order))
        for row in part.itertuples(index=False):
            row = row._asdict()
            p = v.primary
            cells = {
                "Model": f.model(m),
                "Split": f.split(row[SPLIT_COLUMN]),
                f"{f.metric_short(p)} utama": f.num(row[f"{p}_main"]),
                f"{f.metric_short(p)} refit": f.num(row[f"{p}_refit"]),
                f"Δ {f.metric_short(p)}": f.signed(row[f"{p}_delta"]),
            }
            cells.update({f"Δ {f.metric_short(n)}": f.signed(row[f"{n}_delta"])
                          for n in v.names if n != p})
            rows.append(cells)
    return pd.DataFrame(rows)


def _timing_table(v: View, f: Formatter, models: list[str]) -> pd.DataFrame:
    trained = v.inputs.training["models"]
    rows = []
    for m in models:
        fit = trained.get(m, {}).get("fit_seconds")
        refit = v.timing(m, v.r.sensitivity_mode)
        rows.append({
            "Model": f.model(m),
            "Pelatihan (Fase 4)": "–" if fit is None else f.num(fit, 1),
            "Backtest utama": f.num(v.timing(m, v.r.main_mode), 1),
            "Backtest refit": "–" if refit is None else f.num(refit, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _header(c: _Context, inputs: ReportInputs) -> str:
    sources = "\n".join(f"- `{Path(p).as_posix()}`" for p in inputs.sources)
    return f"""
# Laporan Akhir: Perbandingan ETS, ARIMA, dan TFT untuk Peramalan Indeks Biaya Energi

Berkas ini dibuat otomatis oleh `python -m ta_forecasting.cli report`. Isinya hanya statistik
agregat; tidak ada nilai seri, ID seri, nilai aktual, atau prediksi individual. Semua angka
dibaca dari berkas berikut atau dihitung langsung darinya (peringkat, selisih, rasio):

{sources}

Pemetaan ke skripsi: Bagian 1–4 untuk Bab 4 (Hasil dan Pembahasan), Bagian 5–6 untuk Bab 5
(Kesimpulan dan Saran), Bagian 7 dan Lampiran untuk lampiran. Desimal memakai koma.
"""


def _sig_steps(v: View, a: str, b: str) -> tuple[list[int], list[int], set[str]]:
    sig = [h for h in v.steps if bool(v.dm(a, b, h)["significant"])]
    better = {str(v.dm(a, b, h)["better"]) for h in sig}
    return sig, [h for h in v.steps if h not in sig], better


def _dm_sentence(c: _Context, a: str, b: str) -> str:
    v, f = c.v, c.f
    sig, nonsig, better = _sig_steps(v, a, b)
    pair = f"{f.model(a)} vs {f.model(b)}"
    if not sig:
        return f"{pair}: tidak signifikan di semua horizon."
    winners = f.join([f.model(m) for m in sorted(better)])
    text = (f"{pair}: signifikan di horizon {f.join([str(h) for h in sig])} "
            f"({winners} lebih baik)")
    if nonsig:
        text += f", tidak signifikan di horizon {f.join([str(h) for h in nonsig])}"
    return text + "."


def _summary(c: _Context) -> str:
    v, f, r = c.v, c.f, c.v.r
    split = r.split
    mains = main_models(v.cfg)
    rank = v.ranking(split, mains)
    best = rank[0]
    row = v.overall(best, split)
    info = v.split_info(split)
    metrics = ", ".join(f"{f.metric_short(n)} {f.mval(n, row[n])}" for n in v.names)
    lines = [
        f"- Pada data {f.split(split).lower()} ({info['start_quarter']}–{info['end_quarter']}; "
        f"{v.inputs.split_summary['n_series']} seri × {v.n_origins(split)} origin × "
        f"{v.horizon} horizon = {f.count(row['n'])} prediksi per model), model utama dengan "
        f"{f.metric_short(v.primary)} terendah adalah **{f.model(best)}** ({metrics}).",
        f"- Urutan model utama menurut {f.metric_short(v.primary)}: "
        + " < ".join(f"{f.model(m)} ({f.mval(v.primary, v.value(m, split))})" for m in rank)
        + ".",
    ]
    baselines = [b for b in all_models(v.cfg) if b not in mains]
    if baselines:
        base_text = ", ".join(f"{f.model(b)} {f.mval(v.primary, v.value(b, split))}"
                              for b in baselines)
        lines.append(f"- Pembanding acuan: {base_text}.")
        if NAIVE in baselines:
            naive = v.value(NAIVE, split)
            better = [f.model(m) for m in rank if v.value(m, split) < naive]
            worse = [f.model(m) for m in rank if v.value(m, split) >= naive]
            text = (f"- Model utama yang lebih akurat daripada naive: "
                    f"{f.join(better) if better else 'tidak ada'}")
            if worse:
                text += f". Yang tidak lebih akurat daripada naive: {f.join(worse)}"
            lines.append(text + ".")
    if TFT in mains and best != TFT:
        ratio = v.value(TFT, split) / v.value(best, split)
        lines.append(f"- TFT tidak mengungguli model statistik: {f.metric_short(v.primary)} TFT "
                     f"{f.mval(v.primary, v.value(TFT, split))}, atau {f.num(ratio)} kali "
                     f"{f.metric_short(v.primary)} {f.model(best)}.")
    elif best == TFT:
        lines.append("- TFT memiliki "
                     f"{f.metric_short(v.primary)} terendah di antara model utama.")
    lines.append("- Uji Diebold-Mariano tingkat panel: "
                 + " ".join(_dm_sentence(c, a, b) for a, b in model_pairs(v.cfg)))
    w = v.wins(split)
    lines.append(f"- Per seri, {f.metric_short(v.primary)} terendah dicapai oleh "
                 + ", ".join(f"{f.model(m)} pada {int(w[m])}" for m in mains)
                 + f" dari {int(w['n_series'])} seri (seimbang: {int(w['tie'])}).")
    if v.has_sensitivity():
        lines.append("- Sensitivitas refit parameter di setiap origin: " + _sensitivity_brief(c)
                     + " Lihat Bagian 3.5.")
    return "## 1. Ringkasan Eksekutif\n\n" + "\n".join(lines)


def _sensitivity_brief(c: _Context) -> str:
    v, f, split = c.v, c.f, c.v.r.split
    parts = []
    for m in [x for x in v.r.model_order if x in v.cfg.evaluation.sensitivity_models]:
        main = v.value(m, split)
        refit = v.value(m, split, mode=v.r.sensitivity_mode)
        verb = "turun" if refit < main else "naik" if refit > main else "tetap"
        parts.append(f"{f.metric_short(v.primary)} {f.model(m)} {verb} dari {f.num(main)} "
                     f"menjadi {f.num(refit)}")
    return f"{f.join(parts)} (data {f.split(split).lower()})."


def _design(c: _Context) -> str:
    v, f = c.v, c.f
    tr = v.inputs.training["models"]
    ss = v.inputs.split_summary
    e, ets, tft = v.cfg.evaluation, tr.get(ETS, {}), tr.get(TFT, {})
    search = tr.get(ARIMA, {}).get("search", {})
    lines = [
        "## 2. Ringkasan Desain Eksperimen",
        f"Target: {ss['n_series']} seri bertipe `{ss['target_series_type']}`, "
        f"{ss['n_quarters']} kuartal aktual (sampai {ss['last_actual_quarter']}); kuartal "
        "proyeksi S&P tidak dipakai. Pembagian data mengikuti urutan waktu tanpa pengacakan.",
        c.table("split"),
        f"Evaluasi memakai rolling origin dengan horizon {v.horizon} kuartal. Origin sama untuk "
        "semua model, dan history di setiap origin adalah seluruh data aktual sebelum origin. "
        "Pada hasil utama (`refit_statistical = false`) parameter ETS dan ARIMA berasal dari "
        "data latih dan hanya state-nya yang diperbarui, sama seperti bobot TFT yang tetap.",
        "Spesifikasi model:\n\n"
        f"- ETS: tren `{ets.get('spec', {}).get('trend')}` teredam "
        f"(`damped_trend = {ets.get('spec', {}).get('damped_trend')}`), musiman "
        f"`{ets.get('spec', {}).get('seasonal')}` periode "
        f"{ets.get('spec', {}).get('seasonal_periods')}, per seri.\n"
        f"- ARIMA: SARIMAX per seri; D dari kekuatan musiman STL (ambang "
        f"{f.num(search.get('seasonal_strength_threshold', float('nan')))}), d dari KPSS "
        f"(α = {f.num(search.get('kpss_alpha', float('nan')))}), grid p ≤ "
        f"{search.get('max_p')}, q ≤ {search.get('max_q')}, P ≤ {search.get('max_P')}, "
        f"Q ≤ {search.get('max_Q')}, dipilih dengan "
        f"{str(search.get('information_criterion', '')).upper()}.\n"
        f"- TFT: satu model global, {f.count(tft.get('n_parameters', 0))} parameter, "
        f"{tft.get('loss')} dengan {len(tft.get('quantiles', []))} kuantil (median sebagai "
        f"prediksi titik), {f.count(tft.get('n_train_windows', 0))} jendela latih, early "
        "stopping pada "
        f"`val_loss` (epoch terbaik {tft.get('best_epoch')} dari {tft.get('epochs_run')} "
        "epoch).\n"
        f"- Pembanding acuan: {f.join([f.model(b) for b in e.baselines])} (tanpa parameter).",
        f"Metrik: {f.metric_short(v.primary)} sebagai metrik utama; "
        f"{f.join([f.metric_short(n) for n in v.names if n != v.primary])} sebagai pendukung. "
        "Agregat utama dihitung gabungan (pooled) atas seri × origin × horizon. "
        "Signifikansi diuji "
        f"dengan Diebold-Mariano koreksi Harvey-Leybourne-Newbold pada data "
        f"{f.split(e.dm_split).lower()}.",
    ]
    return "\n\n".join(lines)


def _results(c: _Context) -> str:
    v, f, r = c.v, c.f, c.v.r
    split, mains = r.split, main_models(c.v.cfg)
    other = [s for s in v.cfg.evaluation.splits if s != split]
    p = f.metric_short(v.primary)
    parts = ["## 3. Hasil", "### 3.1 Perbandingan keseluruhan",
             c.table(f"overall_{split}")]
    for s in other:
        parts.append(c.table(f"overall_{s}"))
    if TFT in mains:
        parts.append("Catatan: data validasi dipakai TFT untuk early stopping dan pemilihan "
                     "checkpoint, sehingga skor validasi TFT tidak sepenuhnya out-of-sample. "
                     f"Perbandingan utama memakai data {f.split(split).lower()}.")
    parts.append(c.figure("overall", f"{p} per model pada data validasi dan uji."))

    parts += ["### 3.2 Akurasi per horizon", c.table(f"horizon_{split}")]
    for s in other:
        parts.append(c.table(f"horizon_{s}"))
    parts.append(c.table(f"horizon_all_{split}"))
    best_h = [(h, min(mains, key=lambda m: v.step(m, split, h))) for h in v.steps]
    parts.append(
        f"Model utama terbaik per horizon pada data {f.split(split).lower()}: "
        + "; ".join(f"h = {h}: {f.model(m)} ({f.num(v.step(m, split, h))})" for h, m in best_h)
        + ". Perubahan " + p + " dari h = 1 ke h = " + str(v.horizon) + ": "
        + "; ".join(f"{f.model(m)} {f.num(v.step(m, split, 1))} → "
                    f"{f.num(v.step(m, split, v.horizon))}" for m in mains) + "."
    )
    parts.append(c.figure("horizon", f"{p} per horizon pada data {f.split(split).lower()}."))

    parts += ["### 3.3 Sebaran per seri", c.table(f"dist_{split}")]
    for s in other:
        parts.append(c.table(f"dist_{s}"))
    parts.append(c.table("wins"))
    parts.append(_dist_text(c))
    pct = sorted(v.cfg.evaluation.percentiles)
    parts.append(c.figure("boxplot", f"Sebaran {p} per seri pada data {f.split(split).lower()}. "
                          f"Kotak = p25–p75, garis = median, whisker = p{pct[0]:g}–p{pct[-1]:g}, "
                          "segitiga = rata-rata. Digambar dari persentil agregat."))

    parts += ["### 3.4 Uji signifikansi Diebold-Mariano", c.table("dm_panel"),
              "Hasil tingkat panel (loss dirata-rata antar seri per origin):\n\n"
              + "\n".join(f"- {_dm_sentence(c, a, b)}" for a, b in model_pairs(v.cfg)),
              c.table("dm_series"), _dm_series_text(c),
              c.figure("dm_heatmap", "p-value uji Diebold-Mariano tingkat panel per pasangan "
                       "model dan horizon (n.s. = tidak signifikan).")]

    if v.has_sensitivity():
        parts += ["### 3.5 Analisis sensitivitas dan waktu komputasi", c.table("sensitivity"),
                  _sensitivity_text(c), c.table("timing"), _timing_text(c)]
    else:
        parts += ["### 3.5 Waktu komputasi", c.table("timing"), _timing_text(c)]
    return "\n\n".join(parts)


def _dist_text(c: _Context) -> str:
    v, f, split = c.v, c.f, c.v.r.split
    mains = main_models(v.cfg)
    by_median = sorted(mains, key=lambda m: v.dist(m, split)["p50"])
    hi = f"p{max(v.cfg.evaluation.percentiles):g}"
    text = (f"Median {f.metric_short(v.primary)} per seri pada data {f.split(split).lower()}: "
            + "; ".join(f"{f.model(m)} {f.num(v.dist(m, split)['p50'])}" for m in by_median)
            + f". Persentil atas ({hi}): "
            + "; ".join(f"{f.model(m)} {f.num(v.dist(m, split)[hi])}" for m in by_median) + ".")
    winners = {s: max(mains, key=lambda m: int(v.wins(s)[m])) for s in v.cfg.evaluation.splits}
    if len(set(winners.values())) > 1:
        text += (" Model yang paling sering menang berbeda antar periode: "
                 + "; ".join(f"data {f.split(s).lower()} {f.model(m)} ({int(v.wins(s)[m])} seri)"
                             for s, m in winners.items())
                 + ". Peringkat model bergantung pada periode evaluasi.")
    return text


def _dm_series_text(c: _Context) -> str:
    v, f = c.v, c.f
    lines = []
    for a, b in model_pairs(v.cfg):
        a_fdr = [int(v.dm_series(a, b, h)["n_sig_a_better_fdr"]) for h in v.steps]
        b_fdr = [int(v.dm_series(a, b, h)["n_sig_b_better_fdr"]) for h in v.steps]
        n = int(v.dm_series(a, b, 1)["n_series"])
        lines.append(f"- {f.model(a)} vs {f.model(b)}: setelah koreksi FDR, {f.model(a)} "
                     f"signifikan lebih baik pada {min(a_fdr)}–{max(a_fdr)} seri dan "
                     f"{f.model(b)} pada {min(b_fdr)}–{max(b_fdr)} seri dari {n} "
                     f"(rentang atas horizon 1–{v.horizon}).")
    n_origins = int(v.dm_series(*model_pairs(v.cfg)[0], 1)["n_origins"])
    return ("Hasil per seri:\n\n" + "\n".join(lines)
            + f"\n\nUji per seri hanya memakai {n_origins} origin, sehingga daya ujinya rendah; "
            "seri yang tidak signifikan belum tentu memiliki akurasi yang sama.")


def _sensitivity_text(c: _Context) -> str:
    v, f, r = c.v, c.f, c.v.r
    split = r.split
    sens_models = [x for x in r.model_order if x in v.cfg.evaluation.sensitivity_models]
    candidates = {}
    for m in all_models(v.cfg):
        mode = r.sensitivity_mode if m in sens_models else r.main_mode
        candidates[m] = v.value(m, split, mode=mode)
    order = sorted(candidates, key=candidates.get)
    mains = main_models(v.cfg)
    best_main = next(m for m in order if m in mains)
    text = (f"Jika ETS dan ARIMA memakai refit, urutan {f.metric_short(v.primary)} pada data "
            f"{f.split(split).lower()} menjadi: "
            + " < ".join(f"{f.model(m)} ({f.num(candidates[m])})" for m in order) + ". ")
    if best_main == v.ranking(split, mains)[0]:
        text += f"Model utama terbaik tetap {f.model(best_main)}."
    else:
        text += (f"Model utama terbaik berubah menjadi {f.model(best_main)}; kesimpulan "
                 "perbandingan ETS dan ARIMA bergantung pada protokol refit.")
    if NAIVE in candidates:
        worse = [f.model(m) for m in mains if candidates[m] >= candidates[NAIVE]]
        if worse:
            text += (f" Dalam skenario refit, model utama yang tidak lebih akurat daripada "
                     f"naive: {f.join(worse)}.")
    return text


def _timing_text(c: _Context) -> str:
    v, f = c.v, c.f
    t = v.t["timing"]
    not_conv = "log_refit_not_converged"
    accel = v.inputs.training["models"].get(TFT, {}).get("accelerator")
    text = "Waktu backtest mencakup peramalan seluruh seri di semua origin validasi dan uji."
    if accel:
        text += f" Pelatihan TFT dijalankan dengan accelerator `{accel}`."
    if v.has_sensitivity():
        r = v.r
        ratios = []
        for m in [x for x in r.model_order if x in v.cfg.evaluation.sensitivity_models]:
            refit, main = v.timing(m, r.sensitivity_mode), v.timing(m, r.main_mode)
            if refit and main:
                ratios.append(f"{f.model(m)} {f.num(refit / main, 1)} kali")
        text += f" Refit memperlambat backtest: {f.join(ratios)} lebih lama dari mode utama."
        if not_conv in t.columns:
            rows = t[(t[MODE] == r.sensitivity_mode) & (t[not_conv] > 0)]
            for row in rows.itertuples(index=False):
                row = row._asdict()
                text += (f" Pada refit {f.model(row[MODEL])}, {int(row[not_conv])} dari "
                         f"{f.count(row['n_series_origin_forecasts'])} estimasi ulang tidak "
                         "konvergen dan tetap dipakai.")
    return text


def _factors(c: _Context) -> list[str]:
    """Factors behind the relative performance, each included only if the numbers show it."""
    v, f, r = c.v, c.f, c.v.r
    split, mains = r.split, main_models(c.v.cfg)
    a, models = v.inputs.arima, all_models(c.v.cfg)
    other = [s for s in v.cfg.evaluation.splits if s != split]
    train = v.split_info(TRAIN)
    out = []
    if TFT in mains:
        tr = v.inputs.training["models"][TFT]
        out.append(f"data latih pendek ({train['n_quarters']} kuartal per seri, "
                   f"{tr['n_train_windows']} jendela) dibandingkan jumlah parameter TFT "
                   f"({tr['n_parameters']})")
    d1 = sum(n for d, n in a.d.items() if d >= 1)
    if d1 > a.n_series / 2:
        out.append(f"sebagian besar seri non-stasioner (d ≥ 1 pada {d1} dari {a.n_series} seri)")
    if NAIVE in models:
        naive = v.value(NAIVE, split)
        weak = [m for m in mains if v.value(m, split) >= naive]
        if weak:
            out.append(f"naive yang kuat: {f.join([f.model(m) for m in weak])} tidak lebih "
                       "akurat daripada nilai terakhir")
    if NAIVE in models and SEASONAL_NAIVE in models and all(
        v.value(SEASONAL_NAIVE, s) > v.value(NAIVE, s) for s in v.cfg.evaluation.splits
    ):
        out.append("pola musiman kuartalan yang lemah (seasonal naive lebih buruk daripada naive)")
    if TFT in mains and other:
        ratio = {m: v.value(m, split) / v.value(m, other[0]) for m in mains}
        if all(ratio[TFT] > ratio[m] for m in mains if m != TFT):
            out.append(f"pergeseran antara periode {f.split(other[0]).lower()} dan "
                       f"{f.split(split).lower()}: {f.metric_short(v.primary)} TFT berubah "
                       f"{f.num(ratio[TFT])} kali, model lain "
                       + f.join([f"{f.model(m)} {f.num(ratio[m])} kali"
                                 for m in mains if m != TFT]))
    if v.has_sensitivity():
        out.append("protokol pembaruan parameter (refit), yang mengubah "
                   f"{f.metric_short(v.primary)} ETS dan ARIMA (Bagian 3.5)")
    return out


def _discussion(c: _Context) -> str:
    v, f, r = c.v, c.f, c.v.r
    split, mains = r.split, main_models(c.v.cfg)
    p = f.metric_short(v.primary)
    a = v.inputs.arima
    ss = v.inputs.split_summary
    train = v.split_info(TRAIN)
    other = [s for s in v.cfg.evaluation.splits if s != split]
    best = v.ranking(split, mains)[0]
    parts = ["## 4. Pembahasan"]

    # 4.1 data characteristics
    d1 = sum(n for d, n in a.d.items() if d >= 1)
    sd1 = sum(n for d, n in a.seasonal_d.items() if d >= 1)
    data = [
        "### 4.1 Karakteristik data yang memengaruhi hasil",
        f"- Panjang seri: {ss['n_quarters']} kuartal per seri, dengan hanya "
        f"{train['n_quarters']} kuartal latih ({train['start_quarter']}–{train['end_quarter']}). "
        "Jumlah ini kecil untuk model deep learning yang dilatih dari nol.",
        f"- Tren dan non-stasioneritas: uji KPSS memilih differencing d ≥ 1 untuk {d1} dari "
        f"{a.n_series} seri ARIMA."
        + (" Sebagian besar seri berperilaku seperti level yang bergerak (tren atau random "
           "walk), bukan berfluktuasi di sekitar rata-rata tetap." if d1 > a.n_series / 2
           else ""),
        f"- Musiman: differencing musiman (D = 1) hanya dipilih untuk {sd1} dari {a.n_series} "
        f"seri, dan komponen AR/MA musiman ada pada {a.n_seasonal_terms} seri.",
    ]
    if NAIVE in all_models(v.cfg) and SEASONAL_NAIVE in all_models(v.cfg):
        weak = all(v.value(SEASONAL_NAIVE, s) > v.value(NAIVE, s)
                   for s in v.cfg.evaluation.splits)
        comp = "; ".join(f"data {f.split(s).lower()}: seasonal naive "
                         f"{f.num(v.value(SEASONAL_NAIVE, s))} vs naive "
                         f"{f.num(v.value(NAIVE, s))}" for s in v.cfg.evaluation.splits)
        if weak:
            data.append(f"- Seasonal naive lebih buruk daripada naive di semua split ({comp}). "
                        "Pola musiman kuartalan lemah dibandingkan persistensi level terakhir.")
        else:
            data.append(f"- Perbandingan seasonal naive dan naive: {comp}.")
    parts.append("\n".join(data[:1]) + "\n\n" + "\n".join(data[1:]))

    # 4.2 statistical models
    stat = ["### 4.2 Model statistik (ETS dan ARIMA)"]
    if ETS in mains and ARIMA in mains:
        e_v, a_v = v.value(ETS, split), v.value(ARIMA, split)
        lead, lag = (ETS, ARIMA) if e_v < a_v else (ARIMA, ETS)
        stat.append(
            f"{f.model(lead)} lebih akurat daripada {f.model(lag)} pada data "
            f"{f.split(split).lower()} ({p} {f.num(min(e_v, a_v))} vs {f.num(max(e_v, a_v))}). "
            "ETS dengan tren teredam meramal dari level terbaru dan meredam tren ke depan, "
            "sehingga cocok untuk seri yang dekat dengan random walk ber-drift. ARIMA memakai "
            "konstanta (drift) dan koefisien AR/MA yang diestimasi pada data latih; pada hasil "
            "utama parameter tersebut tidak diperbarui di periode validasi dan uji.")
        for s in other:
            e_o, a_o = v.value(ETS, s), v.value(ARIMA, s)
            stat.append(f"Pada data {f.split(s).lower()} urutannya "
                        + ("sama" if (e_o < a_o) == (e_v < a_v) else "terbalik")
                        + f" ({f.model(ETS)} {f.num(e_o)}, {f.model(ARIMA)} {f.num(a_o)}).")
        if NAIVE in all_models(v.cfg):
            naive = v.value(NAIVE, split)
            for m in (ETS, ARIMA):
                if v.value(m, split) >= naive:
                    stat.append(
                        f"{f.model(m)} tidak lebih akurat daripada naive pada data "
                        f"{f.split(split).lower()} ({f.num(v.value(m, split))} vs "
                        f"{f.num(naive)}). Pada seri yang dekat dengan random walk, nilai "
                        "terakhir sudah menjadi peramal yang kuat, dan struktur yang dipelajari "
                        "dari periode latih tidak selalu berlaku di periode uji.")
        if v.has_sensitivity():
            stat.append("Dengan refit parameter di setiap origin (Bagian 3.5), "
                        + _sensitivity_brief(c))
    parts.append("\n\n".join(stat))

    # 4.3 TFT
    if TFT in mains:
        tr = v.inputs.training["models"][TFT]
        tft = ["### 4.3 Temporal Fusion Transformer"]
        changes = "; ".join(
            f"{f.model(m)} {f.num(v.value(m, other[0]))} → {f.num(v.value(m, split))}"
            for m in mains) if other else ""
        if best != TFT:
            tft.append(f"TFT adalah model utama dengan {p} tertinggi pada data "
                       f"{f.split(split).lower()} ({f.num(v.value(TFT, split))}).")
        else:
            tft.append(f"TFT adalah model utama dengan {p} terendah pada data "
                       f"{f.split(split).lower()}.")
        if other:
            tft.append(f"Perubahan {p} dari data {f.split(other[0]).lower()} ke "
                       f"{f.split(split).lower()}: {changes}.")
        ratio = tr["n_parameters"] / tr["n_train_windows"]
        idle = int(tr["epochs_run"]) - int(tr["best_epoch"])
        tft.append(
            f"Faktor yang tercatat: (1) TFT memiliki {tr['n_parameters']} parameter untuk "
            f"{tr['n_train_windows']} jendela latih (≈ {f.num(ratio, 1)} parameter per jendela), "
            f"karena hanya ada {train['n_quarters']} kuartal latih per seri. (2) `val_loss` "
            f"terbaik tercapai di epoch {tr['best_epoch']} dan tidak membaik selama {idle} epoch "
            f"berikutnya (dari {tr['epochs_run']} epoch yang dijalankan). (3) TFT dipakai sebagai "
            "baseline tanpa tuning hyperparameter (A36). (4) Kovariat hanya fitur kalender; "
            "tidak ada variabel eksogen yang dapat dimanfaatkan TFT (A20).")
        if best != TFT:
            tft.append(
                "Dugaan penyebab utama (belum dibuktikan kausal; A45 di `docs/ASSUMPTIONS.md`): "
                "level target pada periode uji berada jauh di atas rentang data latih, sedangkan "
                "normalizer dan bobot TFT ditetapkan dari data latih. Model global harus "
                "mengekstrapolasi ke level yang belum pernah dilihat, sedangkan ETS, ARIMA, dan "
                "naive bekerja relatif terhadap level terakhir di setiap origin.")
        parts.append("\n\n".join(tft))

    # 4.4 horizon
    grow = [m for m in mains if v.step(m, split, v.horizon) > v.step(m, split, 1)]
    hz = ["### 4.4 Pengaruh horizon"]
    hz.append(f"{p} pada horizon {v.horizon} lebih besar daripada horizon 1 untuk "
              f"{f.join([f.model(m) for m in grow]) if grow else 'tidak ada model'}"
              f" pada data {f.split(split).lower()}, sesuai dengan bertambahnya ketidakpastian "
              "untuk ramalan yang lebih jauh.")
    if ETS in mains and ARIMA in mains:
        sig, nonsig, _ = _sig_steps(v, ETS, ARIMA)
        gaps = "; ".join(f"h = {h}: {f.num(v.step(ARIMA, split, h) - v.step(ETS, split, h))}"
                         for h in v.steps)
        hz.append(f"Selisih {p} ARIMA − ETS per horizon: {gaps}. Uji DM panel menyatakan "
                  f"perbedaan ETS dan ARIMA signifikan di horizon "
                  f"{f.join([str(h) for h in sig]) if sig else '(tidak ada)'}"
                  + (f" dan tidak signifikan di horizon {f.join([str(h) for h in nonsig])}"
                     if nonsig else "") + ".")
    parts.append("\n\n".join(hz))
    return "\n\n".join(parts)


def _answers(c: _Context) -> str:
    v, f, r = c.v, c.f, c.v.r
    split, mains = r.split, main_models(c.v.cfg)
    p = f.metric_short(v.primary)
    rank = v.ranking(split, mains)
    best = rank[0]
    parts = ["## 5. Jawaban Pertanyaan Penelitian dan Kriteria Keberhasilan"]

    also = [f.metric_short(n) for n in v.names if n != v.primary
            and min(mains, key=lambda m, n=n: v.value(m, split, n)) == best]
    not_best = [f.metric_short(n) for n in v.names if n != v.primary
                and f.metric_short(n) not in also]
    rq1 = (f"Pada data {f.split(split).lower()}, {p} gabungan: "
           + "; ".join(f"{f.model(m)} {f.num(v.value(m, split))}" for m in rank)
           + f". {f.model(best)} terbaik pada metrik utama")
    if also:
        rq1 += f" dan juga pada {f.join(also)}"
    if not_best:
        rq1 += f", tetapi tidak terbaik pada {f.join(not_best)}"
    parts.append(
        "**RQ1. Bagaimana performa masing-masing model (ETS, ARIMA, TFT) dalam meramalkan "
        "indeks biaya segmen pasar energi?**\n\n"
        + rq1 + f" ({c.tref(f'overall_{split}')}). Rincian per horizon dan per seri ada di "
        f"{c.tref(f'horizon_{split}')}, {c.tref(f'dist_{split}')}, dan {c.tref('wins')}.")

    if TFT in mains:
        stats = [m for m in mains if m != TFT]
        tft_better = [m for m in stats if v.value(TFT, split) < v.value(m, split)]
        dm_text = " ".join(_dm_sentence(c, a, b) for a, b in model_pairs(v.cfg)
                           if TFT in (a, b))
        answer = ("Ya, terhadap " + f.join([f.model(m) for m in tft_better])
                  if tft_better else "Tidak")
        parts.append(
            "**RQ2. Apakah model deep learning (TFT) memberikan akurasi lebih baik dibandingkan "
            "model statistik klasik (ETS, ARIMA) pada data ini?**\n\n"
            f"{answer}. {p} TFT pada data {f.split(split).lower()} adalah "
            f"{f.num(v.value(TFT, split))}, dibandingkan "
            + f.join([f"{f.model(m)} {f.num(v.value(m, split))}" for m in stats])
            + f". Uji Diebold-Mariano tingkat panel ({c.tref('dm_panel')}): {dm_text}")

    factors = _factors(c)
    parts.append(
        "**RQ3. Faktor-faktor apa yang mempengaruhi performa relatif masing-masing model?**\n\n"
        "Berdasarkan Bagian 3 dan 4, faktor yang terlihat pada angka:\n\n"
        + "\n".join(f"{i}. {text[0].upper()}{text[1:]}." for i, text in enumerate(factors, 1)))

    parts.append(_criteria_table(c))
    rec = (f"Rekomendasi: untuk data dan protokol ini, {f.model(best)} adalah metode yang paling "
           f"sesuai, karena memiliki {p} terendah pada data {f.split(split).lower()}")
    if NAIVE in all_models(v.cfg) and v.value(best, split) < v.value(NAIVE, split):
        rec += " dan lebih akurat daripada pembanding naive"
    rec += f". Waktu komputasinya tercantum di {c.tref('timing')}."
    parts.append(rec)
    return "\n\n".join(parts)


def _criteria_table(c: _Context) -> str:
    v, f = c.v, c.f
    tr = v.inputs.training["models"]
    mains = main_models(v.cfg)
    n_series = v.inputs.split_summary["n_series"]
    trained = all(tr[m].get("n_series") == n_series for m in mains)
    fallbacks = sum(int(tr[m].get("n_fallback", 0)) for m in mains)
    n_by_split = {s: {int(v.overall(m, s)["n"]) for m in all_models(v.cfg)}
                  for s in v.cfg.evaluation.splits}
    fair = all(len(n) == 1 for n in n_by_split.values())
    origins = {int(x) for x in v.t["dm_panel"]["n_origins"]}
    env = v.inputs.training.get("environment", {})
    rows = [
        {"Aspek": "Fungsional", "Kriteria": "Ketiga model berhasil dilatih dan menghasilkan "
         "prediksi.", "Status": "Terpenuhi" if trained else "Belum terpenuhi",
         "Bukti": f"{f.join([f.model(m) for m in mains])} dilatih untuk {n_series} seri; "
         f"jumlah fallback {fallbacks}."},
        {"Aspek": "Evaluatif", "Kriteria": "Metrik dihitung dan dibandingkan secara adil.",
         "Status": "Terpenuhi" if fair else "Belum terpenuhi",
         "Bukti": "Jumlah prediksi sama untuk semua model per split ("
         + "; ".join(f"{f.split(s).lower()} {', '.join(map(str, sorted(n)))}"
                     for s, n in n_by_split.items())
         + f"); origin uji DM: {', '.join(map(str, sorted(origins)))}."},
        {"Aspek": "Interpretatif", "Kriteria": "Hasil dilaporkan apa adanya, termasuk bila TFT "
         "kalah.", "Status": "Terpenuhi",
         "Bukti": "Bagian 1, 3, dan 4 melaporkan semua model dan baseline tanpa seleksi."},
        {"Aspek": "Reprodusibilitas", "Kriteria": "Eksperimen dapat direproduksi dengan seed = "
         f"{v.cfg.project.seed}.",
         "Status": "Terpenuhi" if v.inputs.training.get("seed") == v.cfg.project.seed
         else "Belum terpenuhi",
         "Bukti": f"Seed {v.inputs.training.get('seed')}; TFT deterministik "
         f"(`deterministic = {v.cfg.tft.deterministic}`, verifikasi ulang pelatihan di A36); "
         f"{len(env)} versi paket tercatat (Bagian 7)."},
    ]
    return ("Kriteria keberhasilan (dari `docs/BUSINESS_UNDERSTANDING.md`):\n\n"
            + markdown_table(pd.DataFrame(rows)))


def _limitations(c: _Context) -> str:
    v = c.v
    tr = v.inputs.training["models"].get(TFT, {})
    ss = v.inputs.split_summary
    train = v.split_info(TRAIN)
    n_origins = int(v.dm_series(*model_pairs(v.cfg)[0], 1)["n_origins"])
    limits = [
        "- Data validasi dipakai TFT untuk early stopping, pemilihan checkpoint, dan penjadwalan "
        f"learning rate (epoch terbaik {tr.get('best_epoch')}), sehingga skor validasi TFT "
        "tidak sepenuhnya out-of-sample. Kesimpulan diambil dari data uji.",
        f"- Ukuran data: {ss['n_quarters']} kuartal per seri, {train['n_quarters']} kuartal "
        f"latih, dan {n_origins} origin uji. Uji DM per seri memiliki daya uji rendah.",
        f"- Sumber data tunggal (S&P Global, {ss['n_series']} seri `Cost Index`). Hasil belum "
        "tentu berlaku untuk tipe seri, wilayah, atau sumber lain.",
        "- Lisensi: data berhak cipta sehingga data mentah, prediksi per seri, dan ID seri tidak "
        "dapat dipublikasikan. Laporan ini hanya memuat agregat, sehingga analisis per segmen "
        "tidak dapat ditampilkan.",
        "- TFT tidak di-tuning dan hanya memakai kovariat kalender; hasilnya mewakili konfigurasi "
        "baseline, bukan batas atas kemampuan TFT.",
        "- Hasil utama memakai parameter tetap dari data latih (`refit_statistical = false`). "
        "Protokol refit mengubah angka ETS dan ARIMA (Bagian 3.5).",
        "- Metrik gabungan (pooled) memberi bobot sama pada setiap prediksi; RMSE dan MAE "
        "bergantung pada skala indeks tiap seri.",
    ]
    advice = [
        "- TFT: memakai normalisasi per jendela (`EncoderNormalizer`) atau target dalam bentuk "
        "selisih/rasio agar tidak bergantung pada level data latih, lalu melatih ulang dengan "
        "data latih dan validasi sebelum diuji.",
        "- Tuning hyperparameter TFT dengan validasi rolling origin terpisah dari data uji.",
        "- Menambahkan kovariat eksogen (misalnya kurs dan harga bahan baku) agar keunggulan "
        "TFT sebagai model multivariat dapat diuji.",
        "- Membandingkan model global lain dan kombinasi (ensemble) ETS–ARIMA.",
        "- Evaluasi probabilistik (cakupan interval kuantil TFT dan interval ETS/ARIMA).",
        "- Memperluas evaluasi ke tipe seri lain dan workbook edisi berikutnya ketika data aktual "
        "baru tersedia.",
    ]
    return ("## 6. Keterbatasan dan Saran\n\n### 6.1 Keterbatasan\n\n" + "\n".join(limits)
            + "\n\n### 6.2 Saran penelitian lanjutan\n\n" + "\n".join(advice))


def _reproducibility(c: _Context) -> str:
    v = c.v
    env = v.inputs.training.get("environment", {})
    frame = pd.DataFrame([{"Paket": k, "Versi": str(val)} for k, val in env.items()])
    accel = v.inputs.training["models"].get(TFT, {}).get("accelerator", "–")
    cmds = "\n".join(f"python -m ta_forecasting.cli {cmd} --config configs/config.yaml"
                     f"  # {desc}" for cmd, desc in CLI_STEPS)
    seed = v.inputs.training.get("seed")
    last = v.inputs.split_summary["last_actual_quarter"]
    bullets = "\n".join([
        f"- Seed global: {seed} (`project.seed`); TFT `deterministic = "
        f"{v.cfg.tft.deterministic}`, accelerator `{accel}`.",
        "- Semua parameter ada di `configs/config.yaml`; versi paket lengkap di "
        "`requirements.lock`.",
        f"- `last_actual_quarter = {last}`; kuartal setelahnya ditolak di setiap tahap.",
    ])
    return f"""
## 7. Reprodusibilitas

{bullets}

Versi paket saat pelatihan (dari `training_summary.json`):

{markdown_table(frame)}

Urutan perintah dari awal sampai laporan:

```bash
{cmds}
```
"""


def _appendix(c: _Context, figures: dict[str, Path]) -> str:
    lines = [f"- {c.fref(k)}: `{Path(p).as_posix()}`" for k, p in figures.items()]
    dpi = c.v.r.figure.dpi
    return ("## Lampiran: Daftar Grafik\n\n" + "\n".join(lines)
            + f"\n\nSemua grafik berformat PNG {dpi} dpi dan dibuat hanya dari tabel agregat.")
