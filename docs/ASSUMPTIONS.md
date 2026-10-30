# Asumsi Proyek

Dokumen ini mencatat semua asumsi yang dibuat selama pengerjaan proyek.
Setiap asumsi baru harus ditambahkan dengan tanggal dan fase terkait.

---

## Fase 0 — Setup

| # | Asumsi | Alasan | Tanggal |
|---|--------|--------|---------|
| A1 | **LAST_ACTUAL_QUARTER = 2026Q1** | Data setelah kuartal ini dianggap proyeksi S&P dan dikecualikan dari fitting, validasi, dan testing. Ditetapkan oleh pengguna. | 2026-10-21 |
| A2 | **Seed global = 42** | Untuk reprodusibilitas semua eksperimen. Standar umum di riset ML. | 2026-10-21 |
| A3 | **Frekuensi data = kuartalan** | Asumsi awal berdasarkan nama workbook "Q2 2026". Diverifikasi di Fase 2. | 2026-10-21 |
| A4 | **Sumber data tunggal** | Hanya workbook S&P Global Energy Market Segment Cost Models Q2 2026. | 2026-10-21 |
| A5 | **Python 3.12** | Versi Python yang tersedia di environment. Kompatibel dengan semua dependensi. | 2026-10-21 |

---

## Fase 1 — Business Understanding

| # | Asumsi | Alasan | Tanggal |
|---|--------|--------|---------|
| A6 | **Metrik utama: MAPE** | Metrik persentase paling mudah dipahami stakeholder bisnis untuk perbandingan multi-metode (ETS, ARIMA, TFT). Didukung RMSE, MAE, dan SMAPE. | 2026-10-28 |
| A7 | **Validasi: Temporal Split (70/15/15)** | Data deret waktu tidak boleh diacak (no random split) untuk mencegah kebocoran data masa depan (data leakage). | 2026-10-28 |

---

## Fase 2 — Data Understanding & ELT

| # | Temuan & Asumsi | Fakta / Verifikasi Empiris | Tanggal |
|---|-----------------|-----------------------------|---------|
| A8 | **Frekuensi kuartalan terverifikasi** | Seluruh data tanggal adalah akhir kuartal (31 Maret, 30 Juni, 30 September, 31 Desember) dari 2000Q1 hingga 2031Q4. | 2026-10-28 |
| A9 | **Pemisahan aktual vs proyeksi S&P** | Data aktual (<= 2026Q1): 85.155 baris (105 kuartal). Data proyeksi S&P (> 2026Q1): 18.653 baris (23 kuartal) telah disaring dan dibuang dari pipeline pelatihan. | 2026-10-28 |
| A10 | **Dataset primer dan metadata** | Sheet `Data` (811 unique ID) adalah deret waktu primer. Sheet `ID lookup` menyediakan taksonomi (Concept, Region, Currency). Sheet `Cost Model Desc` memuat 103 segmen model. Sheet `IO Weight` memuat relasi bobot input-output. | 2026-10-28 |
| A11 | **Format penyimpanan Parquet interim** | Data interim disimpan dalam format Parquet di `data/interim/` dengan penanganan tipe data terstandarisasi untuk efisiensi komputasi Fase 3–5. | 2026-10-28 |

---

## Fase 3 — Data Preparation

| # | Asumsi / Keputusan | Alasan dan verifikasi | Tanggal |
|---|--------------------|------------------------|---------|
| A12 | **Target = panel seri bertipe `Cost Index`** (116 ID, 105 kuartal lengkap). | Satu-satunya tipe seri dengan skala wajar dan sebanding (nilai 65,5–348,2, semua positif). Keputusan pengguna. ETS dan ARIMA di-fit per seri, TFT satu model global. | 2026-10-30 |
| A13 | **Tipe seri lain tidak dipakai sebagai target**, termasuk `Assembly / Fabrication`, `Management / engineering`, dan `Exchange Rate`. | Rentang nilai tidak wajar untuk indeks biaya (hingga ~1e15–1e17, minimum −1,7e10) dan 141 baris bernilai ≤ 0. Dugaan artefak sumber atau ekstraksi; belum ditelusuri ke workbook. `Cost Index with Operating Margins (Price Index)` (105 seri) wajar tetapi di luar keputusan pengguna, jadi tidak diikutkan. | 2026-10-30 |
| A14 | **Pembulatan split: train = ceil, val = floor, test = sisa**, dihitung dengan `Fraction` dari string rasio. | Menghindari kesalahan floating point (contoh: `0.57*100 = 56.99999…`). Untuk n = 105 hasilnya 74/15/16 (train 70,5%, val 14,3%, test 15,2%), sama dengan contoh pada rencana. Satu pembagian kuartal global dipakai untuk semua seri. | 2026-10-30 |
| A15 | **Duplikat `Series ID` di `ID_lookup` dipilih deterministik**: baris dengan metadata terlengkap, lalu urut alfabet Concept dan Region. | Ada 2 `Series ID` duplikat dengan metadata berbeda; keduanya **bukan** seri target (0 dari 116), sehingga tidak memengaruhi hasil. Aturan tetap dibuat agar pipeline tidak bergantung pada urutan baris. | 2026-10-30 |
| A16 | **`numpy` dibatasi `>=1.26,<2`; `lightning` dan `pytorch-lightning` dikunci sama (2.4.0).** Terpasang: numpy 1.26.4, scipy 1.17.1, contourpy 1.3.3, torch 2.2.2, pytorch-forecasting 1.8.0. Daftar lengkap di `requirements.lock`. | `torch 2.2.2` (versi terakhir untuk macOS x86_64) dibangun terhadap NumPy 1.x; dengan NumPy 2.5.4 konversi tensor–numpy gagal ("Numpy is not available"). `pytorch-lightning 2.6.6` tidak sesuai dengan `lightning 2.4.0`. Diverifikasi: interop NumPy–torch, `pip check` bersih, forward pass dan fit 1 epoch TFT pada data sintetis. | 2026-10-30 |
| A17 | **Jendela TFT: `max_encoder_length=16`, `min_encoder_length=16`, `max_prediction_length=4`.** | Keputusan pengguna (horizon 4 kuartal = 1 tahun). Train hanya 74 kuartal, jadi encoder lebih panjang akan mengurangi jumlah jendela latih. `min_encoder_length` disamakan dengan maksimum (lihat A22). | 2026-10-30 |
| A18 | **Val dan Test memakai `min_encoder_length = max_encoder_length`.** | Jika tidak, pytorch-forecasting menambahkan salinan jendela terakhir dengan encoder lebih pendek, sehingga target terakhir val dan test terhitung berulang (val 2320 dan test 2436 jendela, bukan 1392 dan 1508) dan metrik menjadi bias. Setelah perbaikan: val = 116×12, test = 116×13 jendela. | 2026-10-30 |
| A19 | **Konteks encoder val dan test boleh melewati batas split; target yang dihitung tidak.** | Konteks adalah masa lalu yang sudah diketahui saat peramalan. `min_prediction_idx` dan pemotongan data di akhir split menjamin seluruh target prediksi ada di dalam split itu (diverifikasi: indeks decoder val 74–88, test 89–104, train maks. 73). | 2026-10-30 |
| A20 | **Kovariat: hanya target dan fitur kalender.** Fitur statis: `ID`, `Concept`, `Region`, `Series type`. Known real: `time_idx`, `quarter_sin`, `quarter_cos`. Unknown real: target. Tanpa seri lain (kurs, bahan baku). | Keputusan pengguna. Menjaga perbandingan dengan ETS dan ARIMA (univariat) tetap adil. Catatan: `time_idx` sebagai known real akan berada di luar rentang train untuk val dan test (ekstrapolasi `StandardScaler`). | 2026-10-30 |
| A21 | **Normalisasi: rata-rata dan simpangan baku per `ID`, hanya dari baris train.** `scaler.json` menyimpan std tanpa epsilon; `GroupNormalizer` pytorch-forecasting menambahkan epsilon float16 (~0,000977) pada skalanya. | Mencegah lookahead bias. Konsistensi keduanya diuji di `tests/test_prep.py`. Panel (`panel.parquet`) menyimpan nilai asli, bukan nilai ter-scale. | 2026-10-30 |
| A22 | **Train memakai `min_encoder_length=16` (= maksimum): 55 jendela per seri, 6.380 total, semua encoder 16, tiap jendela target muncul sekali.** | Dengan nilai awal 8, pytorch-forecasting menghasilkan 71 jendela per seri: 55 jendela penuh, 8 jendela awal deret dengan encoder 8–15, dan 8 salinan jendela terakhir (target mulai time_idx 70), sehingga jendela terakhir train berbobot 9× saat pelatihan. Pengguna memilih menyamakan minimum dengan maksimum. Konsekuensi: 8 jendela awal deret (target mulai time_idx 8–15) tidak dipakai sebagai target latih, tetapi kuartal itu tetap dipakai sebagai konteks encoder. | 2026-10-30 |
| A23 | **TFT dilatih di CPU.** | Mesin macOS x86_64 dengan torch 2.2.2 tidak memiliki CUDA maupun MPS. Keputusan pengguna. Deteksi perangkat di Fase 4 tetap dibuat otomatis dengan fallback ke CPU, tetapi eksperimen skripsi dijalankan di CPU. Waktu latih dicatat sebagai bagian hasil. | 2026-10-30 |
