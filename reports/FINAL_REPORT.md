# Laporan Akhir: Perbandingan ETS, ARIMA, dan TFT untuk Peramalan Indeks Biaya Energi

Berkas ini dibuat otomatis oleh `python -m ta_forecasting.cli report`. Isinya hanya statistik
agregat; tidak ada nilai seri, ID seri, nilai aktual, atau prediksi individual. Semua angka
dibaca dari berkas berikut atau dihitung langsung darinya (peringkat, selisih, rasio):

- `reports/phase5_overall.csv`
- `reports/phase5_per_horizon.csv`
- `reports/phase5_per_series_distribution.csv`
- `reports/phase5_wins.csv`
- `reports/phase5_dm_panel.csv`
- `reports/phase5_dm_per_series_summary.csv`
- `reports/phase5_timing.csv`
- `reports/phase5_sensitivity.csv`
- `models/training_summary.json`
- `models/arima_orders.json`
- `data/processed/split_summary.json`

Pemetaan ke skripsi: Bagian 1–4 untuk Bab 4 (Hasil dan Pembahasan), Bagian 5–6 untuk Bab 5
(Kesimpulan dan Saran), Bagian 7 dan Lampiran untuk lampiran. Desimal memakai koma.

## 1. Ringkasan Eksekutif

- Pada data uji (2022Q2–2026Q1; 116 seri × 13 origin × 4 horizon = 6.032 prediksi per model), model utama dengan MAPE terendah adalah **ETS** (MAPE 3,20%, RMSE 11,76, MAE 7,70, SMAPE 3,16%).
- Urutan model utama menurut MAPE: ETS (3,20%) < ARIMA (4,07%) < TFT (15,81%).
- Pembanding acuan: Naive 3,47%, Seasonal naive 4,38%.
- Model utama yang lebih akurat daripada naive: ETS. Yang tidak lebih akurat daripada naive: ARIMA dan TFT.
- TFT tidak mengungguli model statistik: MAPE TFT 15,81%, atau 4,94 kali MAPE ETS.
- Uji Diebold-Mariano tingkat panel: ETS vs ARIMA: signifikan di horizon 1 dan 2 (ETS lebih baik), tidak signifikan di horizon 3 dan 4. ETS vs TFT: signifikan di horizon 1, 2, 3, dan 4 (ETS lebih baik). ARIMA vs TFT: signifikan di horizon 1, 2, 3, dan 4 (ARIMA lebih baik).
- Per seri, MAPE terendah dicapai oleh ETS pada 85, ARIMA pada 28, TFT pada 3 dari 116 seri (seimbang: 0).
- Sensitivitas refit parameter di setiap origin: MAPE ETS naik dari 3,20 menjadi 3,52 dan MAPE ARIMA turun dari 4,07 menjadi 3,77 (data uji). Lihat Bagian 3.5.

## 2. Ringkasan Desain Eksperimen

Target: 116 seri bertipe `Cost Index`, 105 kuartal aktual (sampai 2026Q1); kuartal proyeksi S&P tidak dipakai. Pembagian data mengikuti urutan waktu tanpa pengacakan.

**Tabel 1.** Pembagian data temporal.

| Split | Periode | Kuartal | Baris (seri × kuartal) | Origin rolling |
|---|---|---|---|---|
| Latih | 2000Q1–2018Q2 | 74 | 8.584 | – |
| Validasi | 2018Q3–2022Q1 | 15 | 1.740 | 12 |
| Uji | 2022Q2–2026Q1 | 16 | 1.856 | 13 |

Evaluasi memakai rolling origin dengan horizon 4 kuartal. Origin sama untuk semua model, dan history di setiap origin adalah seluruh data aktual sebelum origin. Pada hasil utama (`refit_statistical = false`) parameter ETS dan ARIMA berasal dari data latih dan hanya state-nya yang diperbarui, sama seperti bobot TFT yang tetap.

Spesifikasi model:

- ETS: tren `add` teredam (`damped_trend = True`), musiman `add` periode 4, per seri.
- ARIMA: SARIMAX per seri; D dari kekuatan musiman STL (ambang 0,64), d dari KPSS (α = 0,05), grid p ≤ 3, q ≤ 3, P ≤ 1, Q ≤ 1, dipilih dengan AIC.
- TFT: satu model global, 236.062 parameter, QuantileLoss dengan 7 kuantil (median sebagai prediksi titik), 6.380 jendela latih, early stopping pada `val_loss` (epoch terbaik 2 dari 12 epoch).
- Pembanding acuan: Naive dan Seasonal naive (tanpa parameter).

Metrik: MAPE sebagai metrik utama; RMSE, MAE, dan SMAPE sebagai pendukung. Agregat utama dihitung gabungan (pooled) atas seri × origin × horizon. Signifikansi diuji dengan Diebold-Mariano koreksi Harvey-Leybourne-Newbold pada data uji.

## 3. Hasil

### 3.1 Perbandingan keseluruhan

**Tabel 2.** Metrik keseluruhan pada data uji (gabungan seri × origin × horizon, mode utama).

| Model | MAPE (%) | RMSE | MAE | SMAPE (%) | n prediksi | Peringkat MAPE |
|---|---|---|---|---|---|---|
| ETS | 3,20 | 11,76 | 7,70 | 3,16 | 6.032 | 1 |
| ARIMA | 4,07 | 14,46 | 9,74 | 3,99 | 6.032 | 3 |
| TFT | 15,81 | 42,75 | 38,32 | 17,40 | 6.032 | 5 |
| Naive | 3,47 | 12,21 | 8,36 | 3,46 | 6.032 | 2 |
| Seasonal naive | 4,38 | 14,98 | 10,45 | 4,37 | 6.032 | 4 |

**Tabel 3.** Metrik keseluruhan pada data validasi (gabungan seri × origin × horizon, mode utama).

| Model | MAPE (%) | RMSE | MAE | SMAPE (%) | n prediksi | Peringkat MAPE |
|---|---|---|---|---|---|---|
| ETS | 5,41 | 18,35 | 11,56 | 5,62 | 5.568 | 2 |
| ARIMA | 5,30 | 17,24 | 11,19 | 5,44 | 5.568 | 1 |
| TFT | 7,25 | 21,88 | 15,06 | 7,46 | 5.568 | 5 |
| Naive | 5,43 | 19,19 | 11,74 | 5,70 | 5.568 | 3 |
| Seasonal naive | 6,83 | 22,94 | 14,68 | 7,21 | 5.568 | 4 |

Catatan: data validasi dipakai TFT untuk early stopping dan pemilihan checkpoint, sehingga skor validasi TFT tidak sepenuhnya out-of-sample. Perbandingan utama memakai data uji.

![MAPE per model pada data validasi dan uji.](../outputs/report/fig_mape_per_model.png)

**Gambar 1.** MAPE per model pada data validasi dan uji.

### 3.2 Akurasi per horizon

**Tabel 4.** MAPE (%) per horizon pada data uji.

| Model | h = 1 | h = 2 | h = 3 | h = 4 |
|---|---|---|---|---|
| ETS | 1,99 | 2,87 | 3,52 | 4,41 |
| ARIMA | 2,97 | 4,48 | 4,17 | 4,64 |
| TFT | 12,98 | 16,01 | 16,70 | 17,53 |
| Naive | 2,32 | 3,26 | 3,78 | 4,54 |
| Seasonal naive | 4,68 | 4,10 | 4,21 | 4,54 |

**Tabel 5.** MAPE (%) per horizon pada data validasi.

| Model | h = 1 | h = 2 | h = 3 | h = 4 |
|---|---|---|---|---|
| ETS | 2,64 | 4,57 | 6,38 | 8,03 |
| ARIMA | 2,56 | 4,60 | 6,21 | 7,82 |
| TFT | 4,01 | 6,61 | 8,41 | 9,95 |
| Naive | 2,61 | 4,55 | 6,34 | 8,20 |
| Seasonal naive | 5,44 | 6,32 | 7,36 | 8,20 |

**Tabel 6.** Semua metrik per horizon, model utama, data uji.

| Model | Horizon | MAPE (%) | RMSE | MAE | SMAPE (%) |
|---|---|---|---|---|---|
| ETS | 1 | 1,99 | 8,03 | 4,82 | 1,97 |
| ETS | 2 | 2,87 | 10,71 | 6,82 | 2,82 |
| ETS | 3 | 3,52 | 12,16 | 8,43 | 3,48 |
| ETS | 4 | 4,41 | 15,04 | 10,71 | 4,38 |
| ARIMA | 1 | 2,97 | 11,33 | 7,13 | 2,94 |
| ARIMA | 2 | 4,48 | 15,88 | 10,68 | 4,39 |
| ARIMA | 3 | 4,17 | 14,24 | 9,97 | 4,08 |
| ARIMA | 4 | 4,64 | 15,90 | 11,20 | 4,56 |
| TFT | 1 | 12,98 | 35,73 | 31,42 | 14,09 |
| TFT | 2 | 16,01 | 42,47 | 38,56 | 17,62 |
| TFT | 3 | 16,70 | 44,55 | 40,41 | 18,44 |
| TFT | 4 | 17,53 | 47,40 | 42,88 | 19,46 |

Model utama terbaik per horizon pada data uji: h = 1: ETS (1,99); h = 2: ETS (2,87); h = 3: ETS (3,52); h = 4: ETS (4,41). Perubahan MAPE dari h = 1 ke h = 4: ETS 1,99 → 4,41; ARIMA 2,97 → 4,64; TFT 12,98 → 17,53.

![MAPE per horizon pada data uji.](../outputs/report/fig_mape_per_horizon.png)

**Gambar 2.** MAPE per horizon pada data uji.

### 3.3 Sebaran per seri

**Tabel 7.** Distribusi MAPE per seri (%; data uji).

| Model | Rata-rata | p10 | p25 | p50 | p75 | p90 | n seri |
|---|---|---|---|---|---|---|---|
| ETS | 3,20 | 1,92 | 2,22 | 2,87 | 3,72 | 5,06 | 116 |
| ARIMA | 4,07 | 1,92 | 2,58 | 3,29 | 4,90 | 6,71 | 116 |
| TFT | 15,81 | 7,41 | 12,13 | 16,36 | 19,18 | 22,36 | 116 |
| Naive | 3,47 | 2,41 | 2,58 | 3,10 | 3,84 | 5,09 | 116 |
| Seasonal naive | 4,38 | 2,60 | 2,95 | 4,07 | 4,80 | 6,91 | 116 |

**Tabel 8.** Distribusi MAPE per seri (%; data validasi).

| Model | Rata-rata | p10 | p25 | p50 | p75 | p90 | n seri |
|---|---|---|---|---|---|---|---|
| ETS | 5,41 | 2,51 | 3,60 | 5,12 | 6,79 | 8,45 | 116 |
| ARIMA | 5,30 | 2,30 | 3,53 | 4,90 | 6,91 | 8,25 | 116 |
| TFT | 7,25 | 3,99 | 4,88 | 6,27 | 8,36 | 11,24 | 116 |
| Naive | 5,43 | 2,76 | 3,79 | 5,08 | 6,83 | 8,28 | 116 |
| Seasonal naive | 6,83 | 3,48 | 4,93 | 6,29 | 8,19 | 10,05 | 116 |

**Tabel 9.** Jumlah seri dengan MAPE terendah (hanya model utama).

| Split | n seri | ETS | ARIMA | TFT | Seimbang |
|---|---|---|---|---|---|
| Validasi | 116 | 23 | 67 | 26 | 0 |
| Uji | 116 | 85 | 28 | 3 | 0 |

Median MAPE per seri pada data uji: ETS 2,87; ARIMA 3,29; TFT 16,36. Persentil atas (p90): ETS 5,06; ARIMA 6,71; TFT 22,36. Model yang paling sering menang berbeda antar periode: data validasi ARIMA (67 seri); data uji ETS (85 seri). Peringkat model bergantung pada periode evaluasi.

![Sebaran MAPE per seri pada data uji. Kotak = p25–p75, garis = median, whisker = p10–p90, segitiga = rata-rata. Digambar dari persentil agregat.](../outputs/report/fig_mape_per_series_boxplot.png)

**Gambar 3.** Sebaran MAPE per seri pada data uji. Kotak = p25–p75, garis = median, whisker = p10–p90, segitiga = rata-rata. Digambar dari persentil agregat.

### 3.4 Uji signifikansi Diebold-Mariano

**Tabel 10.** Uji Diebold-Mariano (HLN) tingkat panel, data uji, loss APE, α = 0,05.

| Pasangan (A vs B) | Horizon | Rata-rata selisih loss (A − B) | Statistik DM-HLN | p-value | n origin | Kesimpulan |
|---|---|---|---|---|---|---|
| ETS vs ARIMA | 1 | -0,9834 | -4,5478 | 0,0007 | 13 | Signifikan, ETS lebih baik |
| ETS vs ARIMA | 2 | -1,6114 | -5,5817 | 0,0001 | 13 | Signifikan, ETS lebih baik |
| ETS vs ARIMA | 3 | -0,6510 | -2,0184 | 0,0665 | 13 | Tidak signifikan |
| ETS vs ARIMA | 4 | -0,2365 | -0,4928 | 0,6310 | 13 | Tidak signifikan |
| ETS vs TFT | 1 | -10,9901 | -19,8023 | < 0,0001 | 13 | Signifikan, ETS lebih baik |
| ETS vs TFT | 2 | -13,1427 | -13,0769 | < 0,0001 | 13 | Signifikan, ETS lebih baik |
| ETS vs TFT | 3 | -13,1824 | -9,4803 | < 0,0001 | 13 | Signifikan, ETS lebih baik |
| ETS vs TFT | 4 | -13,1209 | -7,2390 | < 0,0001 | 13 | Signifikan, ETS lebih baik |
| ARIMA vs TFT | 1 | -10,0067 | -14,5838 | < 0,0001 | 13 | Signifikan, ARIMA lebih baik |
| ARIMA vs TFT | 2 | -11,5313 | -9,8376 | < 0,0001 | 13 | Signifikan, ARIMA lebih baik |
| ARIMA vs TFT | 3 | -12,5314 | -7,8419 | < 0,0001 | 13 | Signifikan, ARIMA lebih baik |
| ARIMA vs TFT | 4 | -12,8844 | -5,9835 | 0,0001 | 13 | Signifikan, ARIMA lebih baik |

Hasil tingkat panel (loss dirata-rata antar seri per origin):

- ETS vs ARIMA: signifikan di horizon 1 dan 2 (ETS lebih baik), tidak signifikan di horizon 3 dan 4.
- ETS vs TFT: signifikan di horizon 1, 2, 3, dan 4 (ETS lebih baik).
- ARIMA vs TFT: signifikan di horizon 1, 2, 3, dan 4 (ARIMA lebih baik).

**Tabel 11.** Uji Diebold-Mariano per seri: jumlah seri signifikan, tanpa koreksi dan dengan koreksi FDR (fdr_bh).

| Pasangan (A vs B) | Horizon | Seri diuji | A lebih baik (mentah) | B lebih baik (mentah) | A lebih baik (FDR) | B lebih baik (FDR) | Tidak signifikan (FDR) |
|---|---|---|---|---|---|---|---|
| ETS vs ARIMA | 1 | 116 | 41 | 1 | 7 | 1 | 108 |
| ETS vs ARIMA | 2 | 116 | 50 | 2 | 35 | 1 | 80 |
| ETS vs ARIMA | 3 | 116 | 12 | 3 | 3 | 2 | 111 |
| ETS vs ARIMA | 4 | 116 | 4 | 12 | 0 | 6 | 110 |
| ETS vs TFT | 1 | 116 | 108 | 0 | 108 | 0 | 8 |
| ETS vs TFT | 2 | 116 | 107 | 1 | 107 | 1 | 8 |
| ETS vs TFT | 3 | 116 | 102 | 0 | 102 | 0 | 14 |
| ETS vs TFT | 4 | 116 | 96 | 0 | 96 | 0 | 20 |
| ARIMA vs TFT | 1 | 116 | 103 | 1 | 102 | 1 | 13 |
| ARIMA vs TFT | 2 | 116 | 99 | 4 | 99 | 4 | 13 |
| ARIMA vs TFT | 3 | 116 | 98 | 2 | 98 | 2 | 16 |
| ARIMA vs TFT | 4 | 116 | 94 | 0 | 94 | 0 | 22 |

Hasil per seri:

- ETS vs ARIMA: setelah koreksi FDR, ETS signifikan lebih baik pada 0–35 seri dan ARIMA pada 1–6 seri dari 116 (rentang atas horizon 1–4).
- ETS vs TFT: setelah koreksi FDR, ETS signifikan lebih baik pada 96–108 seri dan TFT pada 0–1 seri dari 116 (rentang atas horizon 1–4).
- ARIMA vs TFT: setelah koreksi FDR, ARIMA signifikan lebih baik pada 94–102 seri dan TFT pada 0–4 seri dari 116 (rentang atas horizon 1–4).

Uji per seri hanya memakai 13 origin, sehingga daya ujinya rendah; seri yang tidak signifikan belum tentu memiliki akurasi yang sama.

![p-value uji Diebold-Mariano tingkat panel per pasangan model dan horizon (n.s. = tidak signifikan).](../outputs/report/fig_dm_pvalue_heatmap.png)

**Gambar 4.** p-value uji Diebold-Mariano tingkat panel per pasangan model dan horizon (n.s. = tidak signifikan).

### 3.5 Analisis sensitivitas dan waktu komputasi

**Tabel 12.** Sensitivitas `refit_statistical = true` (Δ = refit − utama; negatif berarti refit lebih baik).

| Model | Split | MAPE utama | MAPE refit | Δ MAPE | Δ RMSE | Δ MAE | Δ SMAPE |
|---|---|---|---|---|---|---|---|
| ETS | Validasi | 5,41 | 5,43 | +0,02 | +0,17 | +0,05 | +0,03 |
| ETS | Uji | 3,20 | 3,52 | +0,33 | +2,36 | +0,82 | +0,28 |
| ARIMA | Validasi | 5,30 | 5,29 | -0,00 | +0,21 | +0,02 | +0,01 |
| ARIMA | Uji | 4,07 | 3,77 | -0,30 | -0,70 | -0,67 | -0,31 |

Jika ETS dan ARIMA memakai refit, urutan MAPE pada data uji menjadi: Naive (3,47) < ETS (3,52) < ARIMA (3,77) < Seasonal naive (4,38) < TFT (15,81). Model utama terbaik tetap ETS. Dalam skenario refit, model utama yang tidak lebih akurat daripada naive: ETS, ARIMA, dan TFT.

**Tabel 13.** Waktu komputasi (detik).

| Model | Pelatihan (Fase 4) | Backtest utama | Backtest refit |
|---|---|---|---|
| ETS | 9,8 | 11,7 | 182,7 |
| ARIMA | 625,4 | 9,6 | 154,9 |
| TFT | 280,9 | 10,6 | – |
| Naive | – | 4,7 | – |
| Seasonal naive | – | 5,1 | – |

Waktu backtest mencakup peramalan seluruh seri di semua origin validasi dan uji. Pelatihan TFT dijalankan dengan accelerator `cpu`. Refit memperlambat backtest: ETS 15,6 kali dan ARIMA 16,1 kali lebih lama dari mode utama. Pada refit ARIMA, 7 dari 2.900 estimasi ulang tidak konvergen dan tetap dipakai.

## 4. Pembahasan

### 4.1 Karakteristik data yang memengaruhi hasil

- Panjang seri: 105 kuartal per seri, dengan hanya 74 kuartal latih (2000Q1–2018Q2). Jumlah ini kecil untuk model deep learning yang dilatih dari nol.
- Tren dan non-stasioneritas: uji KPSS memilih differencing d ≥ 1 untuk 112 dari 116 seri ARIMA. Sebagian besar seri berperilaku seperti level yang bergerak (tren atau random walk), bukan berfluktuasi di sekitar rata-rata tetap.
- Musiman: differencing musiman (D = 1) hanya dipilih untuk 1 dari 116 seri, dan komponen AR/MA musiman ada pada 29 seri.
- Seasonal naive lebih buruk daripada naive di semua split (data validasi: seasonal naive 6,83 vs naive 5,43; data uji: seasonal naive 4,38 vs naive 3,47). Pola musiman kuartalan lemah dibandingkan persistensi level terakhir.

### 4.2 Model statistik (ETS dan ARIMA)

ETS lebih akurat daripada ARIMA pada data uji (MAPE 3,20 vs 4,07). ETS dengan tren teredam meramal dari level terbaru dan meredam tren ke depan, sehingga cocok untuk seri yang dekat dengan random walk ber-drift. ARIMA memakai konstanta (drift) dan koefisien AR/MA yang diestimasi pada data latih; pada hasil utama parameter tersebut tidak diperbarui di periode validasi dan uji.

Pada data validasi urutannya terbalik (ETS 5,41, ARIMA 5,30).

ARIMA tidak lebih akurat daripada naive pada data uji (4,07 vs 3,47). Pada seri yang dekat dengan random walk, nilai terakhir sudah menjadi peramal yang kuat, dan struktur yang dipelajari dari periode latih tidak selalu berlaku di periode uji.

Dengan refit parameter di setiap origin (Bagian 3.5), MAPE ETS naik dari 3,20 menjadi 3,52 dan MAPE ARIMA turun dari 4,07 menjadi 3,77 (data uji).

### 4.3 Temporal Fusion Transformer

TFT adalah model utama dengan MAPE tertinggi pada data uji (15,81).

Perubahan MAPE dari data validasi ke uji: ETS 5,41 → 3,20; ARIMA 5,30 → 4,07; TFT 7,25 → 15,81.

Faktor yang tercatat: (1) TFT memiliki 236062 parameter untuk 6380 jendela latih (≈ 37,0 parameter per jendela), karena hanya ada 74 kuartal latih per seri. (2) `val_loss` terbaik tercapai di epoch 2 dan tidak membaik selama 10 epoch berikutnya (dari 12 epoch yang dijalankan). (3) TFT dipakai sebagai baseline tanpa tuning hyperparameter (A36). (4) Kovariat hanya fitur kalender; tidak ada variabel eksogen yang dapat dimanfaatkan TFT (A20).

Dugaan penyebab utama (belum dibuktikan kausal; A45 di `docs/ASSUMPTIONS.md`): level target pada periode uji berada jauh di atas rentang data latih, sedangkan normalizer dan bobot TFT ditetapkan dari data latih. Model global harus mengekstrapolasi ke level yang belum pernah dilihat, sedangkan ETS, ARIMA, dan naive bekerja relatif terhadap level terakhir di setiap origin.

### 4.4 Pengaruh horizon

MAPE pada horizon 4 lebih besar daripada horizon 1 untuk ETS, ARIMA, dan TFT pada data uji, sesuai dengan bertambahnya ketidakpastian untuk ramalan yang lebih jauh.

Selisih MAPE ARIMA − ETS per horizon: h = 1: 0,98; h = 2: 1,61; h = 3: 0,65; h = 4: 0,24. Uji DM panel menyatakan perbedaan ETS dan ARIMA signifikan di horizon 1 dan 2 dan tidak signifikan di horizon 3 dan 4.

## 5. Jawaban Pertanyaan Penelitian dan Kriteria Keberhasilan

**RQ1. Bagaimana performa masing-masing model (ETS, ARIMA, TFT) dalam meramalkan indeks biaya segmen pasar energi?**

Pada data uji, MAPE gabungan: ETS 3,20; ARIMA 4,07; TFT 15,81. ETS terbaik pada metrik utama dan juga pada RMSE, MAE, dan SMAPE (Tabel 2). Rincian per horizon dan per seri ada di Tabel 4, Tabel 7, dan Tabel 9.

**RQ2. Apakah model deep learning (TFT) memberikan akurasi lebih baik dibandingkan model statistik klasik (ETS, ARIMA) pada data ini?**

Tidak. MAPE TFT pada data uji adalah 15,81, dibandingkan ETS 3,20 dan ARIMA 4,07. Uji Diebold-Mariano tingkat panel (Tabel 10): ETS vs TFT: signifikan di horizon 1, 2, 3, dan 4 (ETS lebih baik). ARIMA vs TFT: signifikan di horizon 1, 2, 3, dan 4 (ARIMA lebih baik).

**RQ3. Faktor-faktor apa yang mempengaruhi performa relatif masing-masing model?**

Berdasarkan Bagian 3 dan 4, faktor yang terlihat pada angka:

1. Data latih pendek (74 kuartal per seri, 6380 jendela) dibandingkan jumlah parameter TFT (236062).
2. Sebagian besar seri non-stasioner (d ≥ 1 pada 112 dari 116 seri).
3. Naive yang kuat: ARIMA dan TFT tidak lebih akurat daripada nilai terakhir.
4. Pola musiman kuartalan yang lemah (seasonal naive lebih buruk daripada naive).
5. Pergeseran antara periode validasi dan uji: MAPE TFT berubah 2,18 kali, model lain ETS 0,59 kali dan ARIMA 0,77 kali.
6. Protokol pembaruan parameter (refit), yang mengubah MAPE ETS dan ARIMA (Bagian 3.5).

Kriteria keberhasilan (dari `docs/BUSINESS_UNDERSTANDING.md`):

| Aspek | Kriteria | Status | Bukti |
|---|---|---|---|
| Fungsional | Ketiga model berhasil dilatih dan menghasilkan prediksi. | Terpenuhi | ETS, ARIMA, dan TFT dilatih untuk 116 seri; jumlah fallback 0. |
| Evaluatif | Metrik dihitung dan dibandingkan secara adil. | Terpenuhi | Jumlah prediksi sama untuk semua model per split (validasi 5568; uji 6032); origin uji DM: 13. |
| Interpretatif | Hasil dilaporkan apa adanya, termasuk bila TFT kalah. | Terpenuhi | Bagian 1, 3, dan 4 melaporkan semua model dan baseline tanpa seleksi. |
| Reprodusibilitas | Eksperimen dapat direproduksi dengan seed = 42. | Terpenuhi | Seed 42; TFT deterministik (`deterministic = True`, verifikasi ulang pelatihan di A36); 8 versi paket tercatat (Bagian 7). |

Rekomendasi: untuk data dan protokol ini, ETS adalah metode yang paling sesuai, karena memiliki MAPE terendah pada data uji dan lebih akurat daripada pembanding naive. Waktu komputasinya tercantum di Tabel 13.

## 6. Keterbatasan dan Saran

### 6.1 Keterbatasan

- Data validasi dipakai TFT untuk early stopping, pemilihan checkpoint, dan penjadwalan learning rate (epoch terbaik 2), sehingga skor validasi TFT tidak sepenuhnya out-of-sample. Kesimpulan diambil dari data uji.
- Ukuran data: 105 kuartal per seri, 74 kuartal latih, dan 13 origin uji. Uji DM per seri memiliki daya uji rendah.
- Sumber data tunggal (S&P Global, 116 seri `Cost Index`). Hasil belum tentu berlaku untuk tipe seri, wilayah, atau sumber lain.
- Lisensi: data berhak cipta sehingga data mentah, prediksi per seri, dan ID seri tidak dapat dipublikasikan. Laporan ini hanya memuat agregat, sehingga analisis per segmen tidak dapat ditampilkan.
- TFT tidak di-tuning dan hanya memakai kovariat kalender; hasilnya mewakili konfigurasi baseline, bukan batas atas kemampuan TFT.
- Hasil utama memakai parameter tetap dari data latih (`refit_statistical = false`). Protokol refit mengubah angka ETS dan ARIMA (Bagian 3.5).
- Metrik gabungan (pooled) memberi bobot sama pada setiap prediksi; RMSE dan MAE bergantung pada skala indeks tiap seri.

### 6.2 Saran penelitian lanjutan

- TFT: memakai normalisasi per jendela (`EncoderNormalizer`) atau target dalam bentuk selisih/rasio agar tidak bergantung pada level data latih, lalu melatih ulang dengan data latih dan validasi sebelum diuji.
- Tuning hyperparameter TFT dengan validasi rolling origin terpisah dari data uji.
- Menambahkan kovariat eksogen (misalnya kurs dan harga bahan baku) agar keunggulan TFT sebagai model multivariat dapat diuji.
- Membandingkan model global lain dan kombinasi (ensemble) ETS–ARIMA.
- Evaluasi probabilistik (cakupan interval kuantil TFT dan interval ETS/ARIMA).
- Memperluas evaluasi ke tipe seri lain dan workbook edisi berikutnya ketika data aktual baru tersedia.

## 7. Reprodusibilitas

- Seed global: 42 (`project.seed`); TFT `deterministic = True`, accelerator `cpu`.
- Semua parameter ada di `configs/config.yaml`; versi paket lengkap di `requirements.lock`.
- `last_actual_quarter = 2026Q1`; kuartal setelahnya ditolak di setiap tahap.

Versi paket saat pelatihan (dari `training_summary.json`):

| Paket | Versi |
|---|---|
| lightning | 2.4.0 |
| machine | x86_64 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| python | 3.12.3 |
| pytorch-forecasting | 1.8.0 |
| statsmodels | 0.15.0 |
| torch | 2.2.2 |

Urutan perintah dari awal sampai laporan:

```bash
python -m ta_forecasting.cli extract --config configs/config.yaml  # Ekstraksi workbook (Fase 2)
python -m ta_forecasting.cli profile --config configs/config.yaml  # Profil data dan kamus data (Fase 2)
python -m ta_forecasting.cli transform --config configs/config.yaml  # Transformasi ke Parquet interim (Fase 2)
python -m ta_forecasting.cli eda --config configs/config.yaml  # Grafik EDA (Fase 2)
python -m ta_forecasting.cli prep --config configs/config.yaml  # Panel, split temporal, scaler, dataset TFT (Fase 3)
python -m ta_forecasting.cli train --model all --config configs/config.yaml  # Pelatihan ETS, ARIMA, TFT (Fase 4)
python -m ta_forecasting.cli evaluate --config configs/config.yaml  # Evaluasi rolling origin (Fase 5)
python -m ta_forecasting.cli report --config configs/config.yaml  # Laporan akhir dan grafik (Fase 6)
```

## Lampiran: Daftar Grafik

- Gambar 1: `outputs/report/fig_mape_per_model.png`
- Gambar 2: `outputs/report/fig_mape_per_horizon.png`
- Gambar 3: `outputs/report/fig_mape_per_series_boxplot.png`
- Gambar 4: `outputs/report/fig_dm_pvalue_heatmap.png`

Semua grafik berformat PNG 300 dpi dan dibuat hanya dari tabel agregat.
