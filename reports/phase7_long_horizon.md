# Fase 7a — Galat per horizon dan ramalan sampai 2031Q4 (agregat)

Dibuat otomatis oleh `python -m ta_forecasting.cli forecast`. Hanya berisi statistik agregat; ramalan per seri ada di `data/processed/` dan tidak di-commit.

Protokol sama dengan hasil utama Fase 5 (`refit_statistical = false`, model Fase 4 hanya di-load). Origin: setiap kuartal dari awal split val sampai kuartal aktual terakhir; horizon h dinilai pada origin yang targetnya masih aktual, sehingga jumlah origin turun untuk h besar. Horizon 1–4 adalah horizon tervalidasi Fase 5; di atasnya adalah jangka panjang. TFT hanya sampai h = 4. Origin pada split validasi tidak sepenuhnya out-of-sample untuk TFT (early stopping).

Angka h = 1–4 di sini berbeda dari Fase 5 karena himpunan origin berbeda (Fase 5: origin dan target di dalam split yang sama). Hasil Fase 5 tidak berubah.

## MAPE (%) per horizon, semua origin

| h | ets | arima | tft | naive | n origin |
|---|---|---|---|---|---|
| 1 | 2.36 | 2.70 | 10.09 | 2.51 | 31 |
| 2 | 3.87 | 4.49 | 12.85 | 4.09 | 30 |
| 3 | 5.14 | 5.21 | 13.64 | 5.32 | 29 |
| 4 | 6.28 | 6.23 | 14.25 | 6.46 | 28 |
| 5 | 7.20 | 7.22 | – | 7.46 | 27 |
| 6 | 8.05 | 7.77 | – | 8.41 | 26 |
| 7 | 8.96 | 8.35 | – | 9.47 | 25 |
| 8 | 9.72 | 8.88 | – | 10.44 | 24 |
| 9 | 10.29 | 9.15 | – | 11.35 | 23 |
| 10 | 10.89 | 9.37 | – | 12.29 | 22 |
| 11 | 11.56 | 9.66 | – | 13.25 | 21 |
| 12 | 12.21 | 9.75 | – | 14.09 | 20 |
| 13 | 12.72 | 9.85 | – | 14.83 | 19 |
| 14 | 12.91 | 9.79 | – | 15.30 | 18 |
| 15 | 12.92 | 9.44 | – | 15.69 | 17 |
| 16 | 13.02 | 8.78 | – | 16.19 | 16 |
| 17 | 13.12 | 8.11 | – | 16.67 | 15 |
| 18 | 13.61 | 7.98 | – | 17.46 | 14 |
| 19 | 14.24 | 8.03 | – | 18.43 | 13 |
| 20 | 14.77 | 8.04 | – | 19.21 | 12 |
| 21 | 15.36 | 8.35 | – | 19.88 | 11 |
| 22 | 15.66 | 8.34 | – | 20.19 | 10 |
| 23 | 15.54 | 8.36 | – | 20.37 | 9 |

## MAPE (%) per horizon menurut split origin (h ≤ 4)

| split origin | h | ets | arima | tft | naive |
|---|---|---|---|---|---|
| test | 1 | 2.04 | 2.90 | 14.05 | 2.33 |
| test | 2 | 3.05 | 4.41 | 16.85 | 3.36 |
| test | 3 | 3.74 | 4.21 | 17.20 | 4.00 |
| test | 4 | 4.41 | 4.64 | 17.53 | 4.54 |
| val | 1 | 2.70 | 2.48 | 5.86 | 2.71 |
| val | 2 | 4.70 | 4.58 | 8.84 | 4.82 |
| val | 3 | 6.44 | 6.15 | 10.32 | 6.55 |
| val | 4 | 7.90 | 7.60 | 11.41 | 8.12 |

## Interval prediksi empiris 80% (rasio aktual / ramalan)

Interval = ramalan × [rasio_lo, rasio_hi], per model dan horizon, dari semua origin backtest, digabung antar seri. Cuplikan h = 1, 4, dan 23; tabel lengkap di `phase7_intervals.csv`.

| model | h | rasio_lo | rasio_hi | n |
|---|---|---|---|---|
| arima | 1 | 0.9640 | 1.0431 | 3596 |
| arima | 4 | 0.9206 | 1.1196 | 3248 |
| arima | 23 | 0.9912 | 1.1938 | 1044 |
| ets | 1 | 0.9728 | 1.0410 | 3596 |
| ets | 4 | 0.9340 | 1.1428 | 3248 |
| ets | 23 | 1.0782 | 1.3237 | 1044 |
| naive | 1 | 0.9753 | 1.0468 | 3596 |
| naive | 4 | 0.9479 | 1.1559 | 3248 |
| naive | 23 | 1.1534 | 1.3873 | 1044 |
| tft | 1 | 0.9781 | 1.2624 | 3596 |
| tft | 4 | 0.9412 | 1.3307 | 3248 |

## Uji cakupan interval (out-of-sample)

Kuantil dihitung hanya dari origin val, lalu cakupan (%) diukur pada origin test. Nominal 80%. Hanya horizon yang memiliki origin di kedua split.

| h | ets | arima | tft | naive |
|---|---|---|---|---|
| 1 | 89.4 | 78.4 | 50.4 | 86.0 |
| 2 | 87.3 | 78.5 | 64.8 | 86.4 |
| 3 | 90.4 | 86.9 | 70.3 | 87.9 |
| 4 | 90.7 | 90.1 | 72.3 | 90.8 |
| 5 | 89.7 | 88.4 | – | 88.9 |
| 6 | 88.2 | 88.0 | – | 87.7 |
| 7 | 88.4 | 86.5 | – | 86.6 |
| 8 | 87.3 | 86.6 | – | 87.2 |
| 9 | 81.9 | 83.0 | – | 80.7 |
| 10 | 74.3 | 77.6 | – | 74.4 |
| 11 | 67.5 | 68.0 | – | 65.4 |
| 12 | 65.0 | 66.9 | – | 64.8 |
| 13 | 67.0 | 70.3 | – | 65.9 |
| 14 | 63.8 | 66.7 | – | 62.6 |
| 15 | 57.8 | 57.8 | – | 53.4 |
| 16 | 62.9 | 67.2 | – | 64.7 |

## Catatan pembacaan

- arima: pada h = 20, 21 seluruh interval berada di atas ramalan titik (rasio_lo > 1). Pada backtest, setidaknya 90% aktual di horizon ini berada di atas ramalan, jadi galatnya bias satu arah, bukan acak.
- ets: pada h = 16, 17, 18, 19, 20, 21, 22, 23 seluruh interval berada di atas ramalan titik (rasio_lo > 1). Pada backtest, setidaknya 90% aktual di horizon ini berada di atas ramalan, jadi galatnya bias satu arah, bukan acak.
- naive: pada h = 15, 16, 17, 18, 19, 20, 21, 22, 23 seluruh interval berada di atas ramalan titik (rasio_lo > 1). Pada backtest, setidaknya 90% aktual di horizon ini berada di atas ramalan, jadi galatnya bias satu arah, bukan acak.
- arima: cakupan uji lebih dari 10 poin di bawah nominal pada h = 11, 12, 14, 15, 16.
- ets: cakupan uji lebih dari 10 poin di bawah nominal pada h = 11, 12, 13, 14, 15, 16.
- naive: cakupan uji lebih dari 10 poin di bawah nominal pada h = 11, 12, 13, 14, 15, 16.
- tft: cakupan uji lebih dari 10 poin di bawah nominal pada h = 1, 2.
- arima: MAPE tertinggi pada h = 13 lalu turun. Horizon panjang hanya dinilai pada origin awal (sedikit origin), jadi penurunan ini kemungkinan efek komposisi origin, bukan model yang makin akurat.

## Ringkasan kuartal termurah (jumlah seri)

`short` = h 1–4 (semua model), `full` = semua langkah tiap model sampai 2031Q4 (TFT tetap 4 langkah). `n_within_error` = seri yang selisih ramalan termurah terhadap aktual terakhir lebih kecil dari MAPE model pada horizon itu. Naive datar sehingga tidak memiliki kuartal termurah. `n_agree_with_consensus` = seri yang kuartal termurahnya sama dengan kuartal yang paling banyak dipilih model.

| model | jendela | n_series | n_flat | n_within_error | n_cheapest_is_step1 | median_change_pct | n_agree_with_consensus |
|---|---|---|---|---|---|---|---|
| ets | full | 116 | 0 | 114 | 20 | -0.14 | 112 |
| arima | full | 116 | 0 | 94 | 76 | 2.16 | 42 |
| tft | full | 116 | 0 | 11 | 2 | -25.99 | 98 |
| naive | full | 116 | 116 | 0 | 0 | 0.00 | 0 |
| ets | short | 116 | 0 | 114 | 20 | -0.14 | 112 |
| arima | short | 116 | 0 | 96 | 76 | 2.16 | 44 |
| tft | short | 116 | 0 | 11 | 2 | -25.99 | 98 |
| naive | short | 116 | 116 | 0 | 0 | 0.00 | 0 |

## Waktu komputasi (detik)

| long_backtest | production_forecast |
|---|---|
| 39.2 | 3.0 |

Kuartal aktual terakhir: 2026Q1 (time_idx 104). Proyeksi S&P setelahnya tidak dibaca.
