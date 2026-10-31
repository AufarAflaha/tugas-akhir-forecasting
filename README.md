# Tugas Akhir: Perbandingan ETS, ARIMA, dan TFT untuk Peramalan Indeks Biaya Energi

## Deskripsi

Proyek tugas akhir (skripsi) Sistem Informasi, Universitas Telkom.
Membandingkan tiga algoritma peramalan — **Exponential Smoothing (ETS)**, **ARIMA**, dan **Temporal Fusion Transformer (TFT)** — pada data indeks biaya segmen pasar energi dari S&P Global.

## Metodologi

Proyek ini mengikuti kerangka kerja **CRISP-DM** dengan fase-fase berikut:

| Fase | Nama | Status |
|------|------|--------|
| 0 | Project Setup | ✅ |
| 1 | Business Understanding | ✅ |
| 2 | Data Understanding & ELT | ✅ |
| 3 | Data Preparation | ✅ |
| 4 | Modeling | ✅ |
| 5 | Evaluation | ✅ |
| 6 | Reporting | ✅ |


## Struktur Proyek

```
├── configs/           # Parameter konfigurasi (config.yaml)
├── data/
│   ├── raw/           # Data mentah (read-only, tidak di-commit)
│   ├── interim/       # Data intermediate
│   ├── processed/     # Data siap model
│   └── db/            # Database lokal
├── docs/              # Dokumentasi proyek
├── models/            # Artefak model
├── outputs/           # Grafik dan tabel hasil
├── reports/           # Laporan akhir
├── src/ta_forecasting/ # Kode sumber utama
└── tests/             # Unit & integration tests
```

## Instalasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Versi paket yang sudah diuji tersimpan di `requirements.lock`
(`pip install -r requirements.lock` untuk mereproduksi lingkungan yang sama).
`numpy` sengaja dibatasi `<2` karena `torch 2.2.x` (versi terakhir untuk macOS x86_64)
dibangun terhadap NumPy 1.x. `lightning` dan `pytorch-lightning` harus bernomor versi sama (2.4.0).

## Penggunaan CLI
```bash
# Lihat semua perintah
python -m ta_forecasting.cli --help

# Fase 2: Ekstraksi data
python -m ta_forecasting.cli extract --config configs/config.yaml

# Fase 2: Profiling data (menghasilkan docs/DATA_DICTIONARY.md)
python -m ta_forecasting.cli profile --config configs/config.yaml

# Fase 2: Transformasi & simpan ke data/interim/ (Parquet)
python -m ta_forecasting.cli transform --config configs/config.yaml

# Fase 2: Pembuatan visualisasi EDA ke outputs/eda/
python -m ta_forecasting.cli eda --config configs/config.yaml

# Fase 3: Persiapan data (panel, split temporal, scaler, dataset TFT) ke data/processed/
python -m ta_forecasting.cli prep --config configs/config.yaml

# Fase 4: Pelatihan model (ets | arima | tft | all) ke models/
python -m ta_forecasting.cli train --model all --config configs/config.yaml

# Fase 5: Evaluasi rolling origin (tabel agregat ke reports/, prediksi dan plot ke outputs/eval/)
python -m ta_forecasting.cli evaluate --config configs/config.yaml

# Fase 6: Laporan akhir (reports/FINAL_REPORT.md) dan grafik skripsi (outputs/report/)
python -m ta_forecasting.cli report --config configs/config.yaml
```

## Fase 3: Persiapan Data

Modul: `src/ta_forecasting/data/prepare.py`. Semua parameter ada di blok `prep` pada `configs/config.yaml`.

| Langkah | Keterangan |
|---------|------------|
| Seleksi target | Panel seri bertipe `Cost Index` (116 seri, 105 kuartal: 2000Q1–2026Q1). Metadata statis (`Concept`, `Region`, `Series type`) diambil dari `ID_lookup`. Kuartal setelah `last_actual_quarter` ditolak. |
| Fitur | `time_idx` (0..104, satu langkah per kuartal), `quarter`, `quarter_sin`, `quarter_cos` (periode 4). |
| Split temporal | Train = ceil(0,7·n), Val = floor(0,15·n), Test = sisa. Untuk n = 105: **74 / 15 / 16** kuartal, yaitu Train 2000Q1–2018Q2, Val 2018Q3–2022Q1, Test 2022Q2–2026Q1. Tanpa pengacakan. |
| Scaler | Rata-rata dan simpangan baku per `ID`, dihitung **hanya dari baris train** (`scaler.json`). `GroupNormalizer` pada TFT juga hanya di-fit pada train. |
| Dataset TFT | `TimeSeriesDataSet` (encoder 16, prediksi 4). Val dan Test dibuat lewat `from_dataset`, dengan target yang diprediksi hanya di dalam split masing-masing. |

Keluaran di `data/processed/`: `panel.parquet`, `scaler.json`, `split_summary.json`
(hanya batas kuartal dan jumlah baris). Folder ini tidak di-commit.

## Fase 4: Modeling

Modul: `src/ta_forecasting/models/`. Parameter ada di blok `rolling`, `ets`, `arima`, `tft`,
dan `output` pada `configs/config.yaml`.

| Model | Berkas | Ringkasan |
|-------|--------|-----------|
| Antarmuka | `base.py`, `statistical.py` | `fit(train, val=None)`, `predict(horizon)`, `forecast(history, horizon)`, `save(path)`, `load(path)`. Keluaran: `[ID, origin, step, time_idx, prediction]`. Kuartal setelah `last_actual_quarter` ditolak. |
| ETS | `ets.py` | `ExponentialSmoothing` per seri: tren aditif teredam, musiman aditif (periode 4). Seri yang tidak konvergen hanya dicatat. Fallback tanpa musiman hanya bila fit gagal. |
| ARIMA | `arima.py` | `SARIMAX` per seri. `D` dari kekuatan musiman STL (> 0,64), `d` dari KPSS (α = 0,05), lalu grid p, q ≤ 3 dan P, Q ≤ 1 dipilih dengan AIC. Order terpilih per seri disimpan. |
| TFT | `tft.py` | Satu model global, `QuantileLoss` (median sebagai titik prediksi), early stopping dan checkpoint terbaik pada `val_loss`, `ReduceLROnPlateau`, seed 42, deterministik, CPU (deteksi otomatis dengan fallback ke CPU). |

Data yang dipakai: ETS dan ARIMA hanya baris train (2000Q1–2018Q2). TFT dilatih pada train,
dengan baris val (2018Q3–2022Q1) untuk early stopping dan pemilihan checkpoint. Baris test tidak
dibaca pada tahap ini.

Protokol rolling origin (dipakai di Fase 5): horizon 4 kuartal dan origin yang sama untuk ketiga
model. Dengan `rolling.refit_statistical: false` (default), ETS dan ARIMA memakai parameter train
dan hanya memperbarui state dengan history sebelum origin, sama seperti TFT yang bobotnya tetap.

Artefak di `models/`:

| Berkas | Isi |
|--------|-----|
| `ets.pkl`, `arima.pkl` | Model per seri (pickle; berisi turunan data, tidak di-commit) |
| `arima_orders.json` | Order, order musiman, tren, status konvergensi per seri (tanpa nilai data) |
| `tft.ckpt` | Checkpoint TFT terbaik (tidak di-commit) |
| `training_summary.json` | Waktu latih, jumlah seri tidak konvergen atau fallback, distribusi order, epoch, `val_loss` terbaik, versi paket |

Log pelatihan: `outputs/logs/train_<model>.log`. Keputusan dan asumsi Fase 4: A24–A36 di
`docs/ASSUMPTIONS.md`.

## Fase 5: Evaluasi

Modul: `src/ta_forecasting/evaluation/` (`metrics.py`, `backtest.py`, `significance.py`,
`report.py`, `evaluate.py`) dan `src/ta_forecasting/models/baselines.py`. Parameter ada di
blok `metrics` dan `evaluation` pada `configs/config.yaml`. Model Fase 4 hanya di-load, tidak
dilatih ulang.

| Langkah | Keterangan |
|---------|------------|
| Rolling origin | Horizon 4 kuartal, origin sama untuk semua model: val time_idx 74–85 (12 origin), test 89–101 (13 origin). History = semua data aktual sebelum origin. Hasil utama: `refit_statistical = false`. |
| Validasi | Jumlah baris = 116 × origin × 4, semua target di dalam split-nya, origin identik antar model. |
| Metrik | MAPE (utama), RMSE, MAE, SMAPE: keseluruhan, per horizon, per split, distribusi per seri, dan jumlah seri yang dimenangkan tiap model. |
| Signifikansi | Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold per pasangan model di test, per horizon: tingkat panel dan per seri (dengan koreksi FDR). |
| Sensitivitas | ETS dan ARIMA diulang dengan `refit_statistical = true`, dilaporkan terpisah beserta waktu komputasinya. |
| Baseline | Naive dan seasonal naive sebagai pembanding acuan. |

Keluaran:

| Lokasi | Isi | Di-commit? |
|--------|-----|------------|
| `reports/phase5_*.csv`, `reports/phase5_evaluation.md` | Tabel agregat saja | Ya (setelah disetujui) |
| `outputs/eval/*.png` | Grafik metrik per model, per horizon, boxplot metrik per seri | Tidak |
| `outputs/eval/predictions.parquet`, `per_series_metrics.csv` | Prediksi beserta aktual, metrik per seri | Tidak |

Asumsi Fase 5: A37–A45 di `docs/ASSUMPTIONS.md`.

## Fase 6: Reporting

Modul: `src/ta_forecasting/reporting/` (`inputs.py`, `formatting.py`, `figures.py`,
`document.py`, `build.py`). Parameter ada di blok `report` pada `configs/config.yaml`
(path, dpi, ukuran grafik, urutan dan label model, label split dan metrik, jumlah desimal).
Tidak ada model yang dilatih atau dievaluasi ulang.

| Langkah | Keterangan |
|---------|------------|
| Input | Hanya `reports/phase5_*.csv`, `models/training_summary.json`, `models/arima_orders.json` (diringkas menjadi hitungan), dan `data/processed/split_summary.json`. |
| Validasi | Semua model, split, horizon 1–4, dan pasangan DM harus lengkap. CSV dengan kolom ID, aktual, prediksi, `time_idx`, atau origin ditolak. Laporan tidak ditulis bila memuat ID seri. |
| Laporan | `reports/FINAL_REPORT.md` (Bahasa Indonesia, koma desimal): ringkasan eksekutif, desain eksperimen, hasil keseluruhan, per horizon, per seri, uji Diebold-Mariano, sensitivitas refit dan waktu komputasi, pembahasan, jawaban RQ1–RQ3 dan kriteria keberhasilan, keterbatasan dan saran, reprodusibilitas. |
| Grafik | `outputs/report/`: MAPE per model, MAPE per horizon, boxplot MAPE per seri (dari persentil agregat), heatmap p-value DM. PNG 300 dpi. |

Kalimat interpretatif di laporan dipilih berdasarkan angka input, sehingga teks ikut berubah
bila hasil berubah. Pemetaan ke skripsi: Bagian 1–4 untuk Bab 4, Bagian 5–6 untuk Bab 5.

| Lokasi | Isi | Di-commit? |
|--------|-----|------------|
| `reports/FINAL_REPORT.md` | Laporan akhir, agregat saja | Ya (setelah disetujui) |
| `outputs/report/*.png` | Grafik skripsi, agregat saja | Tidak (`outputs/` di-ignore) |

Asumsi Fase 6: A46–A54 di `docs/ASSUMPTIONS.md`.


## Catatan Penting

- Data S&P Global bersifat **proprietary** dan **tidak di-commit** ke repository.
- `LAST_ACTUAL_QUARTER = 2026Q1` — data setelah ini tidak dipakai untuk modeling.
- Seed global: `42` untuk reprodusibilitas.
