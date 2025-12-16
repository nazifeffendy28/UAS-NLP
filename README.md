# UAS-NLP
# 🛍️ Analisis Sentimen Hybrid & SVM Ulasan Produk Tokopedia

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/Model-SVM-orange)
![Status](https://img.shields.io/badge/Status-Prototype-green)

> **Aplikasi Analisis Sentimen Hybrid Ulasan Produk E-Commerce Tokopedia Menggunakan Algoritma Support Vector Machine (SVM) Berbasis Web Dashboard.**

---

## 👥 Tim Pengembang

Project ini disusun sebagai bagian dari penelitian akademik oleh:

| Nama Mahasiswa | NIM | Peran |
| :--- | :--- | :--- |
| **Nazif Hamza Effendy** | 2411501527 | *Lead Developer & Researcher* |
| **Shelly Ananda** | 2411500891 | *Data Analyst & Documentation* |
| **Muhammad Richo Irsyad Faiz** | 2411500958 | *UI/UX Designer & Testing* |

---

## 📖 Latar Belakang & Ringkasan

Pertumbuhan e-commerce di Indonesia menghasilkan volume ulasan produk yang masif namun seringkali tidak terstruktur. Masalah utama yang sering terjadi adalah ketidakkonsistenan antara **teks ulasan** dan **rating bintang** (misalnya: ulasan sarkas atau salah klik rating).

Aplikasi ini hadir untuk memecahkan masalah tersebut dengan pendekatan **Hybrid**:
1.  **Lexicon Based:** Menggunakan kamus kata positif/negatif.
2.  **Rating Correction:** Mengoreksi label sentimen jika terjadi ketidaksesuaian dengan rating (Hybrid Logic).
3.  **Machine Learning (SVM):** Melatih model untuk klasifikasi otomatis yang lebih akurat pada data baru.

**Target Luaran:** Aplikasi berbasis web (Streamlit) yang membantu penjual (Seller) dan pembeli melihat performa produk secara objektif.

---

## 🚀 Fitur Utama

Aplikasi ini (`app2.py`) memiliki fitur lengkap:

### 1. 🧠 Hybrid Sentiment Logic
Sistem tidak menelan mentah-mentah teks ulasan. Logika koreksi diterapkan:
* **Netral/Sarkas:** Jika Teks Positif tapi Rating 1-3.
* **Netral/Bias:** Jika Teks Negatif tapi Rating 4-5.
* **Murni:** Jika Teks dan Rating selaras.

### 2. 🤖 Support Vector Machine (SVM)
* Implementasi algoritma SVM dengan Kernel Linear.
* **TF-IDF Vectorization** untuk pembobotan kata.
* Fitur pelatihan model (*Train*) dan pengujian (*Test*) secara real-time di dashboard.
* Evaluasi model menggunakan **Confusion Matrix** dan **Classification Report**.

### 3. 📊 Business Intelligence Dashboard
* **Top Products:** Grafik produk terlaris (Sold) dan paling banyak diulas.
* **Sentiment Distribution:** Pie chart persentase sentimen.
* **WordCloud:** Visualisasi kata yang paling sering muncul.
* **Top Keywords:** 10 kata paling berpengaruh berdasarkan skor TF-IDF.

### 4. 🛠️ Advanced Preprocessing
* Pembersihan tanda baca & *case folding*.
* **Slang Normalization:** Mengubah bahasa gaul Tokopedia (e.g., "brg", "gan", "mantul", "blm") menjadi bahasa baku Indonesia.
* **Stopwords Removal:** Menghapus kata hubung yang tidak relevan.

---

## Dataset
Format CSV yang diharapkan aplikasi agar berjalan lancar:
text: Isi review/ulasan pembeli.
rating: Angka 1-5.
category: Kategori produk (opsional, untuk filter).
product_name: Nama barang (untuk fitur Top Produk).

## 📂 Struktur Project

```text
├── app2.py                 # File Utama Aplikasi (Source Code)
├── requirements.txt        # Daftar library python
├── README.md               # Dokumentasi User
├── dataset/
│   └── tokopedia-reviews.csv  # Contoh dataset
└── images/                 # Aset gambar (jika ada)

---

## Dataset
Format CSV yang diharapkan aplikasi agar berjalan lancar:
text: Isi review/ulasan pembeli.
rating: Angka 1-5.
category: Kategori produk (opsional, untuk filter).
product_name: Nama barang (untuk fitur Top Produk).
