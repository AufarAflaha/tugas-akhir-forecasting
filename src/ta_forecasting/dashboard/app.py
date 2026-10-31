"""Streamlit entry point: ``python -m ta_forecasting.cli dashboard`` (localhost only).

Three pages: detail per cost model, overview of all cost models, and model information. The
user chooses which models to compare; the dashboard gives no single recommendation.
"""

from __future__ import annotations

import os

import streamlit as st

from ta_forecasting.config import Config, load_config
from ta_forecasting.dashboard import charts
from ta_forecasting.dashboard import data as dd
from ta_forecasting.forecasting import summary as fs
from ta_forecasting.forecasting.horizon import total_steps, validated_steps

CONFIG_ENV = "TA_FORECASTING_CONFIG"
DEFAULT_CONFIG = "configs/config.yaml"
PAGE_DETAIL = "Detail cost model"
PAGE_OVERVIEW = "Ringkasan semua cost model"
PAGE_ABOUT = "Tentang model"
PAGES = (PAGE_DETAIL, PAGE_OVERVIEW, PAGE_ABOUT)
FORECAST_STYLE = "background-color: #fff4e5"


@st.cache_resource
def get_config(path: str) -> Config:
    return load_config(path)


@st.cache_data(show_spinner="Memuat data ...")
def get_data(path: str) -> dd.DashboardData:
    return dd.load_data(load_config(path))


def footer(cfg: Config) -> None:
    st.divider()
    st.caption(cfg.dashboard.source_note)
    st.caption("⚠️ " + cfg.dashboard.license_note)


def model_picker(cfg: Config, labels: dict[str, str], key: str) -> list[str]:
    options = list(cfg.forecast.models)
    default = [m for m in cfg.dashboard.default_models if m in options]
    return st.sidebar.multiselect("Model peramalan", options, default=default,
                                  format_func=lambda m: labels.get(m, m), key=key)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_detail(cfg: Config, data: dd.DashboardData, labels: dict[str, str]) -> None:
    d, c = cfg.dashboard, cfg.prep.columns
    sb = st.sidebar
    sb.subheader("Pilih cost model")
    options = dd.concepts(data, cfg)
    default = options.index(d.default_concept) if d.default_concept in options else 0
    concept = sb.selectbox("Cost model", options, index=default, key="concept")
    region_options = dd.regions(data, concept, cfg)
    region = sb.selectbox("Region", region_options, key="region")
    models = model_picker(cfg, labels, "detail_models")
    years = dd.base_years(data)
    base = sb.selectbox("Tahun dasar (= 100)", years,
                        index=years.index(d.default_base_year) if d.default_base_year in years
                        else 0, key="base_year")
    lo, hi = dd.year_bounds(data, cfg)
    all_years = list(range(lo, hi + 1))
    start = sb.selectbox("Tahun awal grafik", all_years,
                         index=all_years.index(max(lo, d.default_start_year)), key="start")
    end = sb.selectbox("Tahun akhir grafik", [y for y in all_years if y >= start],
                       index=len([y for y in all_years if y >= start]) - 1, key="end")

    sid = dd.series_id(data, concept, region, cfg)
    meta = data.meta[data.meta[c.series_id] == sid].iloc[0]
    sb.subheader("Deskripsi cost model")
    text = meta.get(d.description_text)
    sb.write(text if isinstance(text, str) and text.strip() else "Deskripsi tidak tersedia.")

    st.title(f"{concept}")
    currency = meta.get(d.lookup_currency)
    st.caption(f"Region: {region} · Mata uang lokal: "
               f"{currency if isinstance(currency, str) else '–'} · "
               f"{d.value_label} ({base} = 100) · aktual sampai "
               f"{cfg.project.last_actual_quarter}")
    if not models:
        st.info("Pilih minimal satu model peramalan di panel kiri.")
    for m in models:
        if m in d.model_warnings:
            st.warning(f"**{labels.get(m, m)}**: {d.model_warnings[m]}")

    actual, forecast = dd.series_view(data, sid, models, base, cfg)
    st.plotly_chart(charts.main_chart(actual, forecast, start, end, base, labels, cfg),
                    width="stretch")
    v = validated_steps(cfg)
    st.caption(
        f"Garis putus-putus = ramalan; area hijau = horizon tervalidasi utama (h 1–{v}), area "
        f"abu-abu = jangka panjang. Interval {cfg.forecast.interval_level:.0%} hanya "
        f"ditampilkan sampai h = {d.interval_max_step}; di atas itu cakupan uji interval turun "
        "jauh di bawah nominal sehingga tidak ditampilkan. Batang = perubahan QoQ aktual.")

    if models:
        st.subheader("Perbandingan model")
        st.dataframe(dd.comparison_table(data, sid, models, cfg, labels), hide_index=True,
                     width="stretch")
        st.caption(
            f"MAPE = galat rata-rata backtest pada horizon h (semua cost model). Termurah = "
            "kuartal dengan ramalan terendah, beserta jumlah model yang memilih kuartal yang "
            f"sama. Selisih = terhadap aktual {cfg.project.last_actual_quarter}. 'Tidak "
            "bermakna' berarti selisihnya lebih kecil dari galat model pada horizon itu. "
            "Keputusan pengadaan tetap di tangan pengguna.")

    st.subheader("Data tahunan (rata-rata kuartal)")
    table, fc_cols = dd.annual_table(actual, forecast, start, end, labels, cfg)
    styled = table.style.map(lambda _: FORECAST_STYLE, subset=fc_cols) if fc_cols else table
    st.dataframe(styled, hide_index=True, width="stretch")
    st.caption("Kolom berwarna = tahun ramalan (tahun pertama menggabungkan kuartal aktual dan "
               "ramalan). '–' = tahun tidak lengkap atau model tidak meramal sejauh itu.")


def page_overview(cfg: Config, data: dd.DashboardData, labels: dict[str, str]) -> None:
    c = cfg.prep.columns
    sb = st.sidebar
    models = model_picker(cfg, labels, "overview_models")
    window_label = {fs.SHORT: f"1 tahun (h 1–{validated_steps(cfg)})",
                    fs.FULL: f"Sampai {cfg.forecast.end_quarter}"}
    window = sb.radio("Jendela ramalan", list(window_label), format_func=window_label.get,
                      key="window")
    concept_filter = sb.multiselect("Filter Concept", dd.concepts(data, cfg), key="f_concept")
    region_filter = sb.multiselect("Filter Region", sorted(data.meta[c.region].unique()),
                                   key="f_region")

    st.title("Ringkasan semua cost model")
    if not models:
        st.info("Pilih minimal satu model peramalan di panel kiri.")
        return
    table = dd.overview_table(data, window, models, cfg, labels)
    if concept_filter:
        table = table[table["Cost model"].isin(concept_filter)]
    if region_filter:
        table = table[table["Region"].isin(region_filter)]
    st.caption(f"{len(table)} dari {len(data.meta)} cost model. Klik judul kolom untuk "
               "mengurutkan. Selisih (%) terhadap aktual "
               f"{cfg.project.last_actual_quarter}. TFT hanya meramal "
               f"{validated_steps(cfg)} kuartal.")
    st.dataframe(table, hide_index=True, width="stretch", height=600)
    st.download_button("Unduh CSV (penggunaan internal)",
                       table.to_csv(index=False).encode("utf-8"),
                       file_name=f"ringkasan_{window}.csv", mime="text/csv")


def page_about(cfg: Config, data: dd.DashboardData, labels: dict[str, str]) -> None:
    primary = cfg.metrics.primary
    st.title("Tentang model")
    st.markdown(
        "Dashboard ini membandingkan ramalan **indeks biaya** (bukan harga) dari empat model. "
        "Parameter model diestimasi pada data latih 2000Q1–2018Q2, lalu state-nya diperbarui "
        f"dengan semua data aktual sampai {cfg.project.last_actual_quarter} (prosedur yang sama "
        "dengan evaluasi skripsi). Proyeksi S&P tidak ditampilkan dan tidak dipakai.")

    st.subheader(f"Akurasi pada data uji ({cfg.report.split}, horizon 1–{validated_steps(cfg)})")
    st.dataframe(dd.accuracy_table(data, cfg, labels), hide_index=True)
    st.caption("Sumber: reports/FINAL_REPORT.md (Fase 5, rolling origin 2022Q2–2026Q1).")

    st.subheader(f"{primary.upper()} backtest per horizon (sampai h = {total_steps(cfg)})")
    st.plotly_chart(charts.error_chart(data.errors, list(cfg.forecast.models), labels, cfg),
                    width="stretch")
    st.caption("Sumber: reports/phase7_long_horizon.md. Horizon panjang dinilai dari sedikit "
               "origin (9 untuk h = 23), sehingga angkanya kurang stabil.")

    cov = data.coverage.pivot(index="step", columns="model", values="coverage")
    cov = cov[[m for m in cfg.forecast.models if m in cov.columns]]
    cov.columns = [labels.get(m, m) for m in cov.columns]
    st.subheader(f"Cakupan interval {cfg.forecast.interval_level:.0%} pada data uji (%)")
    st.dataframe(cov.round(1).reset_index().rename(columns={"step": "h"}), hide_index=True)

    err = data.errors.set_index(["model", "step"])[primary]
    lines = [f"- Indeks biaya ({cfg.dashboard.value_label}), bukan harga dalam mata uang."]
    for m in cfg.forecast.models:
        steps = sorted(s for (mm, s) in err.index if mm == m)
        if steps:
            first, last = steps[0], steps[-1]
            lines.append(
                f"- {labels.get(m, m)}: {primary.upper()} backtest "
                f"{dd.fmt_num(err[(m, first)], cfg.dashboard)}% pada h = {first} dan "
                f"{dd.fmt_num(err[(m, last)], cfg.dashboard)}% pada h = {last}.")
    lines += [
        f"- TFT hanya dilatih untuk {validated_steps(cfg)} kuartal; tidak ada ramalan TFT "
        "setelahnya.",
        f"- Interval hanya ditampilkan sampai h = {cfg.dashboard.interval_max_step} karena "
        "cakupannya turun di horizon yang lebih panjang (tabel di atas).",
        "- Keputusan pengadaan tetap di tangan pengguna; dashboard tidak memberi rekomendasi "
        "tunggal.",
    ]
    st.subheader("Keterbatasan")
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    path = os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)
    cfg = get_config(path)
    st.set_page_config(page_title=cfg.dashboard.title, layout="wide",
                       menu_items={"Get help": None, "Report a bug": None,
                                   "About": cfg.dashboard.license_note})
    labels = cfg.report.model_labels
    st.sidebar.title(cfg.dashboard.title)
    page = st.sidebar.radio("Halaman", PAGES, key="page")
    try:
        data = get_data(path)
    except (dd.DashboardDataError, FileNotFoundError) as exc:
        st.error(str(exc))
        footer(cfg)
        return
    {PAGE_DETAIL: page_detail, PAGE_OVERVIEW: page_overview, PAGE_ABOUT: page_about}[page](
        cfg, data, labels)
    st.sidebar.caption("⚠️ " + cfg.dashboard.license_note)
    footer(cfg)


if __name__ == "__main__":
    main()
