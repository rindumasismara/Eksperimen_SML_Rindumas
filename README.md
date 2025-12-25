# Eksperimen Sistem Machine Learning – Online Payment Fraud Detection

Repositori ini mendokumentasikan proses end-to-end pembangunan **sistem Machine Learning** untuk mendeteksi penipuan pada transaksi pembayaran online.  
Proyek ini tidak hanya berhenti pada pembuatan model, tetapi mencakup **alur MLOps lengkap**, mulai dari preprocessing data, training model, eksperimen terkelola, otomatisasi workflow, hingga monitoring dan alerting sistem setelah model di-*deploy*.

Tujuan utama dari proyek ini adalah membangun sistem Machine Learning yang **andal, terukur, dan siap dipantau di lingkungan nyata**.

---

## Latar Belakang

Kasus **online payment fraud** merupakan permasalahan klasik dengan karakteristik:
- Data sangat besar dan tidak seimbang (fraud sangat jarang terjadi)
- Pola fraud cenderung berubah seiring waktu (*concept drift*)
- Sistem harus tetap stabil meskipun menerima banyak request

Karena itu, membangun model saja tidak cukup.  
Diperlukan pendekatan **Machine Learning Operations (MLOps)** agar model dapat:
- Dilatih secara konsisten
- Dievaluasi dan dilacak eksperimennya
- Dijalankan sebagai service
- Dipantau performa dan infrastrukturnya secara real-time

---

## Dataset

Dataset yang digunakan adalah **Online Payment Fraud Detection Dataset** yang bersifat *open-source* dan tersedia di Kaggle.

Dataset ini berisi informasi transaksi seperti:
- Jenis transaksi
- Nominal transaksi
- Saldo pengirim dan penerima
- Label fraud (`isFraud`)

Dataset ini sangat tidak seimbang, sehingga cocok untuk studi kasus fraud detection dan evaluasi model yang lebih realistis.

---

## Alur Proyek Secara Umum

Berikut gambaran besar tahapan yang dilakukan dalam proyek ini:

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Modeling & Evaluasi
4. Otomatisasi Preprocessing & Training
5. Experiment Tracking dengan MLflow
6. Workflow CI/CD
7. Model Serving (Inference API)
8. Monitoring & Logging (Prometheus & Grafana)
9. Alerting Otomatis dengan Grafana

---

## Exploratory Data Analysis (EDA)

Tahap EDA dilakukan untuk memahami karakteristik data, meliputi:
- Statistik deskriptif fitur numerik
- Distribusi target `isFraud`
- Analisis korelasi antar fitur

Hasil EDA menunjukkan bahwa:
- Dataset mengalami **class imbalance yang signifikan**
- Tidak ada korelasi linear kuat langsung terhadap target
- Beberapa fitur saldo memiliki korelasi sangat tinggi satu sama lain

Temuan ini menjadi dasar pemilihan teknik preprocessing dan model yang digunakan.

---

## Data Preprocessing

Tahapan preprocessing yang dilakukan antara lain:
- Menghapus data kosong dan duplikat
- Menghapus fitur ID transaksi yang tidak relevan
- Encoding fitur kategorikal (`type`)
- Standardisasi fitur numerik
- Deteksi dan pembersihan outlier pada nominal transaksi
- Binning nominal transaksi untuk analisis tambahan

Hasil preprocessing disimpan dalam bentuk dataset baru agar pipeline lebih terstruktur dan dapat direproduksi.

---

## Modeling dan Evaluasi

Model Machine Learning dilatih menggunakan data hasil preprocessing dengan skema:
- Train-test split
- Penanganan class imbalance menggunakan SMOTE
- Evaluasi menggunakan metrik yang relevan untuk fraud detection:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

Pendekatan ini dipilih agar model tidak hanya akurat, tetapi juga mampu mendeteksi kasus fraud yang jarang terjadi.

---

## Otomatisasi Preprocessing & Training

Untuk menghindari proses manual yang berulang, preprocessing dan training dibuat dalam bentuk **script Python modular**, sehingga:
- Dapat dijalankan ulang dengan data baru
- Konsisten antar eksperimen
- Mudah diintegrasikan ke workflow otomatis

---

## Experiment Tracking dengan MLflow

MLflow digunakan untuk:
- Mencatat parameter model
- Menyimpan metrik evaluasi
- Menyimpan artefak model
- Membandingkan hasil eksperimen

Dengan MLflow, seluruh proses eksperimen menjadi **terlacak dan transparan**, sehingga memudahkan analisis performa model dari waktu ke waktu.

---

## Workflow CI/CD

Repositori ini menggunakan **GitHub Actions** untuk menjalankan workflow otomatis, seperti:
- Menjalankan preprocessing script
- Menjalankan training model
- Menjaga konsistensi environment

Pendekatan ini membantu memastikan bahwa setiap perubahan kode tetap dapat dijalankan dan direproduksi dengan baik.

---

## Model Serving

Model yang telah dilatih disajikan melalui API berbasis Python (Flask), sehingga:
- Model dapat diakses melalui endpoint HTTP
- Dapat digunakan oleh sistem lain
- Menjadi dasar untuk monitoring performa runtime

Endpoint inference juga mengekspor metrik performa menggunakan Prometheus client.

---

## Monitoring & Logging

### Prometheus
Prometheus digunakan untuk mengumpulkan metrik seperti:
- Jumlah request inference
- Latency model
- Error rate
- Penggunaan CPU sistem

Metrik di-*scrape* langsung dari endpoint `/metrics`.

### Grafana
Grafana digunakan untuk:
- Visualisasi metrik dalam bentuk dashboard
- Analisis tren performa model dan sistem
- Monitoring real-time

Dashboard mencakup metrik seperti:
- Model Latency
- Request Count
- System CPU Usage
- Error Rate

---

## Alerting Otomatis

Grafana Alerting digunakan untuk membuat sistem peringatan otomatis, seperti:
- Alert jika jumlah request melebihi threshold tertentu
- Alert jika rata-rata latency meningkat
- Alert jika penggunaan CPU sistem terlalu tinggi

Notifikasi dapat dikirim melalui email atau media lain yang didukung Grafana, sehingga sistem menjadi **proaktif**, bukan reaktif.

---

## Struktur Folder (Ringkas)
SMSML_Rindumas-Ismara-Putri
- Membangun_model/
    |- modelling.py
    |- modelling_tuning.py
    |- mlruns/
    |- DagsHub.txt
    |- onlinepaymentfraud_preprocessing.csv
    |- requirements.txt
    |- screenshot_artifak-mlflow.jpeg
    |- screenshot_dashboard-mlflow.jpeg
- Monitoring_dan_Logging/
    |- inference.py
    |- prometheus.yml
    |- prometheus_exporter.py
    |- inference.log
    |- bukti monitoring/
    |- bukti alerting/
    |- bukti_serving.jpeg
- .github/workflows/
    |- preprocessing_workflow.yml
- requirements.txt
- Workflow-CI.txt
- README.md


---

## Catatan dan Pengembangan Lanjutan

Proyek ini masih dapat dikembangkan lebih jauh, misalnya dengan:
- Retraining otomatis saat terjadi data drift
- Monitoring kualitas prediksi
- Integrasi notifikasi ke Slack / Telegram
- Deployment menggunakan container (Docker)

Eksplorasi lanjutan sangat terbuka.

---

## Kontribusi

Jika kamu tertarik mengembangkan proyek ini:
1. Fork repository
2. Buat branch baru
3. Ajukan pull request

Setiap ide dan perbaikan sangat diapresiasi.

---

## Pemilik

**Rindumas Ismara Putri**  
GitHub: https://github.com/rindumasismara