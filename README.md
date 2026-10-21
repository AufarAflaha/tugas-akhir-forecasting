# Tugas Akhir: Perbandingan ETS, ARIMA, dan TFT untuk Peramalan Indeks Biaya Energi

## Deskripsi

Proyek tugas akhir (skripsi) Sistem Informasi, Universitas Telkom.
Membandingkan tiga algoritma peramalan — **Exponential Smoothing (ETS)**, **ARIMA**, dan **Temporal Fusion Transformer (TFT)** — pada data indeks biaya segmen pasar energi dari S&P Global.

## Metodologi

Proyek ini mengikuti kerangka kerja **CRISP-DM** dengan fase-fase berikut:

| Fase | Nama | Status |
|------|------|--------|
| 0 | Project Setup | ✅ |
| 1 | Business Understanding | 🔄 |
| 2 | Data Understanding & ELT | ⬜ |
| 3 | Data Preparation | ⬜ |
| 4 | Modeling | ⬜ |
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

## Penggunaan

```bash
# Lihat semua perintah
ta-forecasting --help

# Contoh: extract data
ta-forecasting extract --config configs/config.yaml
```

## Catatan Penting

- Data S&P Global bersifat **proprietary** dan **tidak di-commit** ke repository.
- `LAST_ACTUAL_QUARTER = 2026Q1` — data setelah ini tidak dipakai untuk modeling.
- Seed global: `42` untuk reprodusibilitas.
