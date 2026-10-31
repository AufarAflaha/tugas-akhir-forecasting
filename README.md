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
| 4 | Modeling | ✅ (menunggu persetujuan gate) |
| 5 | Evaluation | ⬜ |
| 6 | Reporting | ⬜ |


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

Log pelatihan: `outputs/logs/train_<model>.log`. Keputusan dan asumsi Fase 4: A24–A35 di
`docs/ASSUMPTIONS.md`.


## Catatan Penting

- Data S&P Global bersifat **proprietary** dan **tidak di-commit** ke repository.
- `LAST_ACTUAL_QUARTER = 2026Q1` — data setelah ini tidak dipakai untuk modeling.
- Seed global: `42` untuk reprodusibilitas.
