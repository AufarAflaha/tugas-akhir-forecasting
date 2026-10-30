# Asumsi Proyek

Dokumen ini mencatat semua asumsi yang dibuat selama pengerjaan proyek.
Setiap asumsi baru harus ditambahkan dengan tanggal dan fase terkait.

---

## Fase 0 — Setup

| # | Asumsi | Alasan | Tanggal |
|---|--------|--------|---------|
| A1 | **LAST_ACTUAL_QUARTER = 2026Q1** | Data setelah kuartal ini dianggap proyeksi S&P dan dikecualikan dari fitting, validasi, dan testing. Ditetapkan oleh pengguna. | 2026-10-21 |
| A2 | **Seed global = 42** | Untuk reprodusibilitas semua eksperimen. Standar umum di riset ML. | 2026-10-21 |
| A3 | **Frekuensi data = kuartalan** | Asumsi awal berdasarkan nama workbook "Q2 2026". Akan diverifikasi di Fase 2. | 2026-10-21 |
| A4 | **Sumber data tunggal** | Hanya workbook S&P Global Energy Market Segment Cost Models Q2 2026. | 2026-10-21 |
| A5 | **Python 3.12** | Versi Python yang tersedia di environment. Kompatibel dengan semua dependensi. | 2026-10-21 |
