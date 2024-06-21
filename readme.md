# Setup Proyek
Proyek ini membutuhkan beberapa langkah konfigurasi sebelum Anda dapat menjalankannya. Ikuti petunjuk di bawah ini untuk menyiapkan dan menjalankan proyek.

Langkah-langkah Konfigurasi
1. Menyalin dan Mengkonfigurasi File db.py
Salin file db.example.py ke db.py:

```cp db.example.py db.py```

Buka file db.py dan sesuaikan pengaturan di dalamnya dengan konfigurasi database Anda.
2. Menyalin dan Mengkonfigurasi File hikvision.py
Salin file hikvision.example.py ke hikvision.py:

```cp hikvision.example.py hikvision.py```

Buka file hikvision.py dan sesuaikan pengaturan di dalamnya dengan konfigurasi kamera Hikvision Anda.
3. Menjalankan Skrip Instalasi
Jalankan skrip install.py untuk membuat direktori yang diperlukan, menginstal dependensi tambahan, dan mengunduh file model dan .weights dari Google Drive.

```python install.py```

Skrip ini akan melakukan hal-hal berikut:

Membuat direktori results
Membuat direktori models
Menginstal paket dari requirements.txt
Mengunduh file model dari Google Drive
Mengunduh file .weights dari Google Drive
Menjalankan Proyek
Setelah menyelesaikan langkah-langkah di atas, Anda dapat menjalankan proyek dengan menjalankan skrip main.py.

```python main.py```

Skrip ini akan memproses gambar dari kamera Hikvision dan mendeteksi objek di dalamnya. Hasil deteksi akan disimpan di direktori results.