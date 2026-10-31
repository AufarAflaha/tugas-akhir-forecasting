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

---

## Fase 4 — Modeling

| # | Asumsi / Keputusan | Alasan dan verifikasi | Tanggal |
|---|--------------------|------------------------|---------|
| A24 | **Antarmuka seragam** (`models/base.py`): `fit(train, val=None)`, `predict(horizon)`, `forecast(history, horizon)`, `save(path)`, `load(path)`. Keluaran: tabel panjang `[ID, origin, step, time_idx, prediction]`. `history` wajib berisi `quarter_label`; kuartal > `last_actual_quarter`, celah `time_idx`, atau nilai non-finite ditolak. | Satu format untuk ketiga model memudahkan rolling origin dan evaluasi Fase 5. Penolakan kuartal proyeksi diuji untuk ketiga model. | 2026-10-31 |
| A25 | **Rolling origin model statistik, default `rolling.refit_statistical: false`**: order dan konfigurasi dipilih di train, parameter juga dari train. Di tiap origin model hanya menjalankan filter (state) atas seluruh history sebelum origin. Pilihan `true` mengestimasi ulang parameter di tiap origin dengan order tetap; bila gagal, parameter train dipakai dan kejadiannya dihitung. | Setara dengan TFT, yang bobotnya tetap setelah dilatih dan hanya membaca history sebagai input. Dengan begitu ketiga model hanya belajar dari train. Konsekuensi: ETS/ARIMA tidak menyesuaikan parameter terhadap perubahan rezim di val/test. **Perlu konfirmasi pengguna sebelum Fase 5.** Diuji: `forecast(train)` = `predict()`, dan mode refit berjalan. | 2026-10-31 |
| A26 | **History statistik harus dimulai di awal train (time_idx 0).** | Mode parameter tetap memakai state awal hasil estimasi train (ETS: `initialization_method="known"`; ARIMA: `results.apply`). Sesuai protokol "history = semua data aktual sebelum origin". | 2026-10-31 |
| A27 | **ETS**: `ExponentialSmoothing` aditif, tren teredam, musiman aditif periode 4, `initialization_method="estimated"`. Seri yang optimizer-nya tidak konvergen (`mle_retvals.success=False` atau `ConvergenceWarning`) **tetap dipakai dan hanya dicatat**. Fallback `fallback_seasonal: null` (tanpa musiman, tren teredam tetap) **hanya** bila fit melempar exception atau menghasilkan parameter/fitted/forecast non-finite. Jika fallback juga gagal, pelatihan berhenti dengan error. | Sesuai instruksi. Parameter komponen yang tidak dipakai (mis. smoothing musiman pada model tanpa musiman) bernilai NaN di statsmodels dan tidak dianggap kegagalan. | 2026-10-31 |
| A28 | **ARIMA — differencing**: `D` ditentukan dulu dengan kekuatan musiman STL, `F_s = max(0, 1 − Var(R)/Var(S+R)) > 0,64` (batas `max_D`). Lalu `d` dengan KPSS (`regression="c"`, `nlags="auto"`, α = 0,05, batas `max_d`) pada seri yang sudah di-difference musiman. | Mengikuti `forecast::nsdiffs` dan `forecast::ndiffs` (Hyndman & Khandakar, 2008; Wang, Smith & Hyndman, 2006). Uji OCSB/Canova-Hansen butuh paket tambahan (pmdarima) yang tidak ada di lingkungan terkunci. | 2026-10-31 |
| A29 | **Grid ARIMA diperkecil: p, q ≤ 3 dan P, Q ≤ 1** (sebelumnya `max_p=5`, `max_q=5`, P=2, Q=2). | (1) Train hanya 74 kuartal; setelah differencing musiman tersisa ±69 observasi, dan model hingga 5+5+2·4+2·4 lag mudah overfit serta sering gagal konvergen. (2) Grid lengkap lama = 6·6·3·3 = 324 kandidat/seri (37.584 fit), grid baru 4·4·2·2 = 64/seri (7.424 fit). (3) Kuartalan dengan musim 4: P, Q = 1 sudah mencakup lag 4; lag 8 jarang bermakna. Tetap exhaustive (bukan stepwise), jadi hasil tidak bergantung urutan pencarian. | 2026-10-31 |
| A30 | **ARIMA — pemilihan**: kandidat diurutkan dengan AIC; hanya kandidat konvergen yang boleh menang; seri dengan AIC sama dipilih berdasarkan urutan grid (deterministik). Konstanta (drift bila `d + D = 1`) disertakan bila `d + D ≤ constant_max_total_diff` (1). Bila tidak ada kandidat konvergen: kandidat finite dengan AIC terendah dipakai dan ditandai. Bila semua kandidat gagal: fallback `(0,d,0)(0,D,0)[4]`. `enforce_stationarity` dan `enforce_invertibility` aktif, `lbfgs`, `maxiter=200`. | AIC hanya dibandingkan di antara model dengan `d` dan `D` yang sama (ditetapkan sebelum grid), jadi perbandingan valid. Aturan konstanta mengikuti `auto.arima`. Pada `SARIMAX`, tren `"c"` berada pada persamaan setelah differencing sehingga berarti drift. | 2026-10-31 |
| A31 | **ARIMA dijalankan paralel per seri** (`n_jobs=4`, joblib). | Seri independen, sehingga hasil identik dengan eksekusi serial. Hanya memengaruhi waktu latih. `joblib` sudah ada di `requirements.lock` (1.6.0) dan kini dicantumkan eksplisit di `pyproject.toml`; tidak ada versi paket yang berubah. | 2026-10-31 |
| A32 | **TFT**: `from_dataset` dengan hidden 64, head 4, dropout 0,1, hidden continuous 16, lr 0,001, batch 64, `gradient_clip_val=0.1`, `QuantileLoss` 7 kuantil (median = titik prediksi). `EarlyStopping(val_loss, patience=10)`, `ModelCheckpoint` terbaik pada `val_loss`, `LearningRateMonitor`, `ReduceLROnPlateau` bawaan pytorch-forecasting (patience 5, faktor 0,5, cooldown 5, lr minimum 1e-5). `seed_everything(42, workers=True)`, `deterministic=True`, `num_workers=0`, `torch.set_num_threads(4)`, accelerator `auto` → CPU di mesin ini. | Sesuai instruksi. Val hanya dipakai untuk early stopping, pemilihan checkpoint, dan penjadwalan lr. Faktor, cooldown, dan lr minimum adalah default pustaka (tidak diatur di config). | 2026-10-31 |
| A33 | **`val_loss` TFT = QuantileLoss dalam satuan indeks asli** (prediksi dikembalikan ke skala asli sebelum loss dihitung), rata-rata 2 × pinball loss atas 7 kuantil, 4 langkah, dan 1.392 jendela val. `epochs_run` = epoch yang selesai; `best_epoch` dihitung mulai 1. | Diverifikasi di kode pytorch-forecasting 1.8.0 (`transform_output` sebelum `self.loss`). Angka ini bukan MAPE dan tidak dipakai untuk membandingkan model. | 2026-10-31 |
| A34 | **`forecast(history)` TFT**: encoder = 16 kuartal terakhir per seri; 4 baris masa depan hanya berisi kovariat kalender (`time_idx`, `quarter_sin`, `quarter_cos`) dan target placeholder (nilai terakhir). Horizon > 4 ditolak. Semua seri harus berakhir di origin yang sama. | TFT tidak membaca target decoder sebagai input. Diuji: hasil `forecast(history < origin)` sama dengan jendela dataset evaluasi di origin itu walaupun target masa depan dikali 1000. | 2026-10-31 |
| A35 | **Artefak model berisi turunan data**: hasil statsmodels menyimpan seri train (`ets.pkl`, `arima.pkl`), dan `tft.ckpt` menyimpan 16 kuartal terakhir train per seri untuk `predict()`. File ini hanya lokal dan sudah di-`.gitignore`. `load()` memakai pickle sehingga hanya boleh memuat berkas buatan proyek ini. `arima_orders.json` dan `training_summary.json` hanya berisi ID seri, order, hitungan, waktu, epoch, dan `val_loss` (tanpa nilai data). | Aturan data AGENTS.md no. 1–4. | 2026-10-31 |
| A36 | **Keputusan gate Fase 4 (pengguna)**: `refit_statistical: false` menjadi hasil utama; `true` dijalankan di Fase 5 sebagai analisis sensitivitas. TFT diterima sebagai baseline tanpa tuning. `models/*.json` tidak di-commit karena berisi ID seri dari workbook berlisensi. | Determinisme TFT diverifikasi pada data asli: pelatihan ulang menghasilkan epoch, epoch terbaik, `val_loss` terbaik, lr akhir, serta prediksi `predict(4)` dan `forecast` di origin 85 yang identik (selisih maksimum 0). | 2026-10-31 |
