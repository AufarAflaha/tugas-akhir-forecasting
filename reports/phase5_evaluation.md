# Fase 5 — Hasil Evaluasi (agregat)

Berkas ini dibuat otomatis oleh `python -m ta_forecasting.cli evaluate`. Hanya berisi statistik agregat.

Protokol: rolling origin, horizon 4 kuartal, origin sama untuk semua model (val: time_idx 74–85 (12 origin), test: time_idx 89–101 (13 origin)). Hasil utama memakai `refit_statistical = false`. Catatan: val dipakai TFT untuk early stopping, sehingga skor val TFT tidak sepenuhnya out-of-sample; perbandingan utama ada di test.

## Metrik keseluruhan (pooled seri × origin × horizon)

| model | split | mape | rmse | mae | smape | n |
|---|---|---|---|---|---|---|
| arima | test | 4.0672 | 14.4574 | 9.7442 | 3.9906 | 6032 |
| arima | val | 5.2963 | 17.2425 | 11.1924 | 5.4368 | 5568 |
| ets | test | 3.1967 | 11.7609 | 7.6961 | 3.1624 | 6032 |
| ets | val | 5.4070 | 18.3484 | 11.5564 | 5.6235 | 5568 |
| naive | test | 3.4716 | 12.2115 | 8.3603 | 3.4560 | 6032 |
| naive | val | 5.4261 | 19.1936 | 11.7399 | 5.6988 | 5568 |
| seasonal_naive | test | 4.3822 | 14.9796 | 10.4490 | 4.3722 | 6032 |
| seasonal_naive | val | 6.8313 | 22.9426 | 14.6848 | 7.2126 | 5568 |
| tft | test | 15.8057 | 42.7543 | 38.3157 | 17.4019 | 6032 |
| tft | val | 7.2468 | 21.8787 | 15.0591 | 7.4618 | 5568 |

## Metrik per horizon (model utama)

| model | split | step | mape | rmse | mae | smape | n |
|---|---|---|---|---|---|---|---|
| arima | test | 1 | 2.9747 | 11.3335 | 7.1291 | 2.9352 | 1508 |
| arima | test | 2 | 4.4803 | 15.8759 | 10.6770 | 4.3921 | 1508 |
| arima | test | 3 | 4.1719 | 14.2424 | 9.9671 | 4.0776 | 1508 |
| arima | test | 4 | 4.6420 | 15.8973 | 11.2036 | 4.5576 | 1508 |
| arima | val | 1 | 2.5585 | 7.3480 | 5.0620 | 2.5714 | 1392 |
| arima | val | 2 | 4.5961 | 13.7227 | 9.4143 | 4.6571 | 1392 |
| arima | val | 3 | 6.2090 | 19.1504 | 13.2096 | 6.3930 | 1392 |
| arima | val | 4 | 7.8217 | 24.0866 | 17.0839 | 8.1256 | 1392 |
| ets | test | 1 | 1.9913 | 8.0260 | 4.8204 | 1.9685 | 1508 |
| ets | test | 2 | 2.8689 | 10.7120 | 6.8227 | 2.8168 | 1508 |
| ets | test | 3 | 3.5209 | 12.1564 | 8.4339 | 3.4811 | 1508 |
| ets | test | 4 | 4.4055 | 15.0443 | 10.7073 | 4.3831 | 1508 |
| ets | val | 1 | 2.6383 | 7.5563 | 5.2372 | 2.6606 | 1392 |
| ets | val | 2 | 4.5732 | 14.2843 | 9.4806 | 4.6858 | 1392 |
| ets | val | 3 | 6.3825 | 20.3268 | 13.7180 | 6.6539 | 1392 |
| ets | val | 4 | 8.0341 | 25.9295 | 17.7898 | 8.4937 | 1392 |
| tft | test | 1 | 12.9814 | 35.7298 | 31.4218 | 14.0910 | 1508 |
| tft | test | 2 | 16.0116 | 42.4684 | 38.5591 | 17.6169 | 1508 |
| tft | test | 3 | 16.7033 | 44.5485 | 40.4067 | 18.4440 | 1508 |
| tft | test | 4 | 17.5264 | 47.4021 | 42.8750 | 19.4557 | 1508 |
| tft | val | 1 | 4.0145 | 11.4708 | 7.9025 | 4.0697 | 1392 |
| tft | val | 2 | 6.6096 | 19.2078 | 13.4679 | 6.7844 | 1392 |
| tft | val | 3 | 8.4117 | 24.2006 | 17.5408 | 8.6636 | 1392 |
| tft | val | 4 | 9.9515 | 28.7841 | 21.3251 | 10.3297 | 1392 |

## Distribusi MAPE per seri

| model | split | metric | mean | p10 | p25 | p50 | p75 | p90 | n_series |
|---|---|---|---|---|---|---|---|---|---|
| arima | test | mape | 4.0672 | 1.9197 | 2.5804 | 3.2931 | 4.9005 | 6.7055 | 116 |
| arima | val | mape | 5.2963 | 2.3038 | 3.5257 | 4.9018 | 6.9117 | 8.2504 | 116 |
| ets | test | mape | 3.1967 | 1.9166 | 2.2176 | 2.8651 | 3.7197 | 5.0569 | 116 |
| ets | val | mape | 5.4070 | 2.5099 | 3.5973 | 5.1241 | 6.7931 | 8.4542 | 116 |
| naive | test | mape | 3.4716 | 2.4057 | 2.5836 | 3.0960 | 3.8419 | 5.0876 | 116 |
| naive | val | mape | 5.4261 | 2.7579 | 3.7861 | 5.0830 | 6.8349 | 8.2799 | 116 |
| seasonal_naive | test | mape | 4.3822 | 2.5953 | 2.9528 | 4.0684 | 4.8001 | 6.9106 | 116 |
| seasonal_naive | val | mape | 6.8313 | 3.4816 | 4.9341 | 6.2903 | 8.1863 | 10.0455 | 116 |
| tft | test | mape | 15.8057 | 7.4122 | 12.1324 | 16.3606 | 19.1794 | 22.3607 | 116 |
| tft | val | mape | 7.2468 | 3.9911 | 4.8762 | 6.2693 | 8.3615 | 11.2423 | 116 |

## Jumlah seri yang dimenangkan (MAPE terendah)

| split | metric | n_series | ets | arima | tft | tie |
|---|---|---|---|---|---|---|
| test | mape | 116 | 85 | 28 | 3 | 0 |
| val | mape | 116 | 23 | 67 | 26 | 0 |

## Diebold-Mariano (HLN), split test, loss ape, α = 0.05

Tingkat panel (loss dirata-rata antar seri per origin):

| model_a | model_b | step | mean_loss_diff | dm_hln | p_value | significant | better | n_origins | variance_fallback |
|---|---|---|---|---|---|---|---|---|---|
| ets | arima | 1 | -0.9834 | -4.5478 | 0.0007 | True | ets | 13 | False |
| ets | arima | 2 | -1.6114 | -5.5817 | 0.0001 | True | ets | 13 | False |
| ets | arima | 3 | -0.6510 | -2.0184 | 0.0665 | False | - | 13 | False |
| ets | arima | 4 | -0.2365 | -0.4928 | 0.6310 | False | - | 13 | False |
| ets | tft | 1 | -10.9901 | -19.8023 | 0.0000 | True | ets | 13 | False |
| ets | tft | 2 | -13.1427 | -13.0769 | 0.0000 | True | ets | 13 | False |
| ets | tft | 3 | -13.1824 | -9.4803 | 0.0000 | True | ets | 13 | False |
| ets | tft | 4 | -13.1209 | -7.2390 | 0.0000 | True | ets | 13 | False |
| arima | tft | 1 | -10.0067 | -14.5838 | 0.0000 | True | arima | 13 | False |
| arima | tft | 2 | -11.5313 | -9.8376 | 0.0000 | True | arima | 13 | False |
| arima | tft | 3 | -12.5314 | -7.8419 | 0.0000 | True | arima | 13 | False |
| arima | tft | 4 | -12.8844 | -5.9835 | 0.0001 | True | arima | 13 | False |

Per seri (jumlah seri signifikan; kolom `_fdr` dengan koreksi fdr_bh):

| model_a | model_b | step | n_series | n_undefined | n_variance_fallback | n_sig_a_better | n_sig_b_better | n_sig_a_better_fdr | n_sig_b_better_fdr | n_origins |
|---|---|---|---|---|---|---|---|---|---|---|
| ets | arima | 1 | 116 | 0 | 0 | 41 | 1 | 7 | 1 | 13 |
| ets | arima | 2 | 116 | 0 | 4 | 50 | 2 | 35 | 1 | 13 |
| ets | arima | 3 | 116 | 0 | 9 | 12 | 3 | 3 | 2 | 13 |
| ets | arima | 4 | 116 | 0 | 10 | 4 | 12 | 0 | 6 | 13 |
| ets | tft | 1 | 116 | 0 | 0 | 108 | 0 | 108 | 0 | 13 |
| ets | tft | 2 | 116 | 0 | 0 | 107 | 1 | 107 | 1 | 13 |
| ets | tft | 3 | 116 | 0 | 0 | 102 | 0 | 102 | 0 | 13 |
| ets | tft | 4 | 116 | 0 | 1 | 96 | 0 | 96 | 0 | 13 |
| arima | tft | 1 | 116 | 0 | 0 | 103 | 1 | 102 | 1 | 13 |
| arima | tft | 2 | 116 | 0 | 1 | 99 | 4 | 99 | 4 | 13 |
| arima | tft | 3 | 116 | 0 | 0 | 98 | 2 | 98 | 2 | 13 |
| arima | tft | 4 | 116 | 0 | 0 | 94 | 0 | 94 | 0 | 13 |

## Analisis sensitivitas: `refit_statistical = true` (terpisah dari hasil utama)

| model | split | mape_main | mape_refit | mape_delta | rmse_main | rmse_refit | rmse_delta | mae_main | mae_refit | mae_delta | smape_main | smape_refit | smape_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arima | test | 4.0672 | 3.7674 | -0.2998 | 14.4574 | 13.7539 | -0.7035 | 9.7442 | 9.0772 | -0.6670 | 3.9906 | 3.6838 | -0.3068 |
| arima | val | 5.2963 | 5.2930 | -0.0033 | 17.2425 | 17.4571 | 0.2146 | 11.1924 | 11.2127 | 0.0203 | 5.4368 | 5.4495 | 0.0127 |
| ets | test | 3.1967 | 3.5217 | 0.3250 | 11.7609 | 14.1197 | 2.3588 | 7.6961 | 8.5120 | 0.8160 | 3.1624 | 3.4401 | 0.2777 |
| ets | val | 5.4070 | 5.4279 | 0.0208 | 18.3484 | 18.5170 | 0.1685 | 11.5564 | 11.6028 | 0.0464 | 5.6235 | 5.6551 | 0.0316 |

## Waktu komputasi backtest

| model | mode | seconds | n_series_origin_forecasts | log_fixed_parameters | log_refit | log_refit_not_converged |
|---|---|---|---|---|---|---|
| ets | main | 11.7400 | 2900 | 2900.0000 | 0.0000 | 0.0000 |
| arima | main | 9.6200 | 2900 | 2900.0000 | 0.0000 | 0.0000 |
| tft | main | 10.6100 | 2900 | 0.0000 | 0.0000 | 0.0000 |
| naive | main | 4.6900 | 2900 | 0.0000 | 0.0000 | 0.0000 |
| seasonal_naive | main | 5.1400 | 2900 | 0.0000 | 0.0000 | 0.0000 |
| ets | refit | 182.6700 | 2900 | 0.0000 | 2900.0000 | 0.0000 |
| arima | refit | 154.9100 | 2900 | 0.0000 | 2893.0000 | 7.0000 |
