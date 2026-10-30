# Business Understanding

Dokumen ini merangkum pemahaman bisnis (Business Understanding) sesuai kerangka kerja CRISP-DM
untuk proyek tugas akhir: **Perbandingan ETS, ARIMA, dan TFT untuk Peramalan Indeks Biaya Energi**.

---

## 1. Latar Belakang

Industri energi membutuhkan peramalan indeks biaya yang akurat untuk:
- **Perencanaan anggaran** — proyeksi biaya konstruksi dan operasional.
- **Analisis risiko** — identifikasi potensi lonjakan biaya pada segmen tertentu.
- **Pengambilan keputusan investasi** — evaluasi kelayakan proyek berdasarkan tren biaya.

S&P Global menyediakan _Market Segment Cost Models_ yang berisi indeks biaya per segmen pasar
energi secara kuartalan. Data ini menjadi dasar bagi banyak perusahaan untuk merencanakan
pengeluaran modal (capex) dan operasional (opex).

## 2. Tujuan Bisnis

> Membandingkan performa tiga metode peramalan — **ETS**, **ARIMA**, dan **TFT** — dalam
> meramalkan indeks biaya segmen pasar energi, guna menentukan metode mana yang paling sesuai
> untuk konteks data ini.

### Tujuan Spesifik

1. Membangun model peramalan ETS, ARIMA, dan TFT pada data indeks biaya kuartalan.
2. Mengevaluasi performa masing-masing model menggunakan metrik yang konsisten.
3. Menganalisis kelebihan dan keterbatasan setiap metode pada karakteristik data ini.
4. Memberikan rekomendasi metode peramalan yang paling cocok.

## 3. Pertanyaan Penelitian

| # | Pertanyaan |
|---|-----------|
| RQ1 | Bagaimana performa masing-masing model (ETS, ARIMA, TFT) dalam meramalkan indeks biaya segmen pasar energi? |
| RQ2 | Apakah model deep learning (TFT) memberikan akurasi lebih baik dibandingkan model statistik klasik (ETS, ARIMA) pada data ini? |
| RQ3 | Faktor-faktor apa yang mempengaruhi performa relatif masing-masing model? |

## 4. Algoritma yang Digunakan

### 4.1 Exponential Smoothing (ETS)

- **Kategori**: Model statistik univariat.
- **Prinsip**: Mendekomposisi time series menjadi komponen _error_ (E), _trend_ (T), dan _seasonality_ (S).
- **Kekuatan**: Sederhana, cepat, interpretatif, cocok untuk data dengan pola tren dan musiman yang jelas.
- **Batasan**: Hanya univariat, tidak bisa memanfaatkan variabel eksogen.

### 4.2 ARIMA (AutoRegressive Integrated Moving Average)

- **Kategori**: Model statistik univariat.
- **Prinsip**: Menggabungkan autoregression (AR), differencing (I), dan moving average (MA) untuk menangkap pola temporal.
- **Kekuatan**: Fleksibel dalam menangkap berbagai pola autokorelasi, didukung metode pemilihan orde otomatis.
- **Batasan**: Asumsi linearitas, membutuhkan stasioneritas, univariat.

### 4.3 Temporal Fusion Transformer (TFT)

- **Kategori**: Model deep learning multivariate.
- **Prinsip**: Arsitektur transformer yang dirancang khusus untuk time series forecasting, mampu menangani input multi-horizon dan variabel eksogen.
- **Kekuatan**: Dapat menangkap hubungan non-linear dan interaksi antar-variabel, menghasilkan _interpretable attention weights_.
- **Batasan**: Membutuhkan lebih banyak data, komputasi lebih berat, risiko overfitting pada data kecil.

## 5. Metrik Evaluasi

Semua metrik dihitung pada **data test** (temporal split, bukan random split).

| Metrik | Singkatan | Formula | Kegunaan |
|--------|-----------|---------|----------|
| Mean Absolute Percentage Error | **MAPE** | $\frac{1}{n}\sum\frac{\|y_i - \hat{y}_i\|}{\|y_i\|} \times 100$ | **Metrik utama** — mudah diinterpretasi sebagai persentase kesalahan |
| Root Mean Squared Error | **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Sensitif terhadap outlier, memberikan bobot lebih pada error besar |
| Mean Absolute Error | **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust terhadap outlier |
| Symmetric MAPE | **SMAPE** | $\frac{100}{n}\sum\frac{2\|y_i - \hat{y}_i\|}{\|y_i\| + \|\hat{y}_i\|}$ | Mengatasi kelemahan MAPE pada nilai mendekati nol |

**Metrik utama**: MAPE (karena paling intuitif untuk stakeholder).
**Metrik pendukung**: RMSE, MAE, SMAPE (untuk analisis komprehensif).

## 6. Strategi Validasi

```
┌─────────────────────────────────┬──────────────┬──────────────┐
│          TRAIN (70%)            │  VAL (15%)   │  TEST (15%)  │
│  Data historis awal             │  Tuning      │  Evaluasi    │
│                                 │  hyperparam  │  akhir       │
└─────────────────────────────────┴──────────────┴──────────────┘
                          ← waktu →
```

- **Strategi**: _Temporal split_ — data dibagi berdasarkan urutan waktu, bukan acak.
- **Rasio**: 70% train / 15% validation / 15% test.
- **Batasan**: Data setelah `LAST_ACTUAL_QUARTER` (2026Q1) **tidak digunakan** — dianggap proyeksi S&P.

## 7. Kriteria Keberhasilan

| Aspek | Kriteria |
|-------|---------|
| **Fungsional** | Ketiga model berhasil dilatih dan menghasilkan prediksi. |
| **Evaluatif** | Metrik evaluasi dihitung dan dibandingkan secara adil. |
| **Interpretatif** | Hasil dilaporkan apa adanya, termasuk jika TFT kalah dari model statistik. |
| **Reprodusibilitas** | Semua eksperimen dapat direproduksi dengan seed = 42. |

## 8. Asumsi Bisnis

Lihat [`docs/ASSUMPTIONS.md`](file:///Users/macbookpro/Documents/tugas-akhir-forecasting/docs/ASSUMPTIONS.md) untuk daftar lengkap asumsi.

## 9. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Data kuartalan terlalu sedikit untuk TFT | TFT underperform | Laporkan apa adanya; gunakan regularisasi dan early stopping |
| Outlier atau structural break pada data | Semua model terdampak | Analisis residual; pertimbangkan robust methods |
| Overfitting pada data kecil | Metrik test buruk | Cross-validation temporal; regularisasi |
| Pola non-stasioner | ARIMA kurang akurat | Differencing; uji stasioneritas (ADF, KPSS) |
