# Aturan proyek: perbandingan ETS, ARIMA, dan TFT untuk peramalan indeks biaya (Tugas Akhir)

## Konteks
Skripsi Sistem Informasi, Universitas Telkom. Metodologi CRISP-DM. Tiga algoritma dibandingkan: ETS, ARIMA, Temporal Fusion Transformer (TFT). Data: workbook S&P Global Energy Market Segment Cost Models Q2 2026 (berlisensi dan berhak cipta).

## Aturan data (WAJIB)
1. Berkas .xlsx hanya berada di data/raw/. Jangan menyalin, mengubah, atau men-commit.
2. Jangan membuka berkas .xlsx di editor atau viewer untuk membaca isinya. Proses data hanya lewat skrip Python.
3. Jangan menampilkan, menempel, atau merangkum baris data mentah di chat, artifact, laporan, atau log. Yang boleh: skema, jumlah baris, dan statistik agregat.
4. Jangan mengirim data atau turunan datanya ke layanan eksternal. Tidak ada panggilan jaringan selain instalasi paket dari PyPI.
5. Nilai proyeksi S&P (setelah LAST_ACTUAL_QUARTER) tidak boleh dipakai untuk fitting, validasi, atau pengujian model.
6. Pengecualian aturan no. 3, khusus dashboard (Fase 7): dashboard boleh menampilkan nilai seri asli dan ramalan per seri kepada pengguna internal. Dashboard hanya berjalan lokal atau di jaringan internal, tidak di-deploy publik, dan datanya tidak di-commit. Pengecualian ini tidak berlaku untuk chat, laporan, artifact, atau log; aturan lain tetap berlaku.

## Standar kode
- Python 3.11+, type hints, ruff, pytest, tata letak src/.
- Semua parameter di configs/config.yaml; tidak ada angka ajaib di kode.
- Seed global 42; hasil harus dapat direproduksi.
- Setiap tahap dapat dijalankan dari CLI. Dependensi dikunci setelah memeriksa kompatibilitas pytorch-forecasting, lightning, dan torch.
- Jangan menghapus berkas di luar models/, reports/, outputs/, dan data/interim, data/processed, data/db. Jangan git push.

## Cara kerja
- Kerjakan per fase. Berhenti di setiap gerbang dan tunggu persetujuan saya.
- Jangan mengarang hasil atau angka. Catat setiap asumsi di docs/ASSUMPTIONS.md.
- Laporkan hasil apa adanya, termasuk bila TFT kalah dari ETS atau ARIMA.
- Sebelum menyatakan selesai: jalankan pytest dan smoke test, lalu laporkan ringkasan hasilnya (bukan datanya).
- Bahasa: kode, nama variabel, dan komentar dalam bahasa Inggris. README, laporan, dan ringkasan kepada saya dalam Bahasa Indonesia.