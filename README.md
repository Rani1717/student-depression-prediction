# 🧠 Prediksi Depresi Mahasiswa — Student Depression ML Project

**Fast Track Bengkel Koding Data Science | Universitas Dian Nuswantoro**

---

## 📋 Deskripsi Proyek
Proyek ini membangun model machine learning untuk memprediksi status depresi mahasiswa berdasarkan faktor akademik, gaya hidup, dan kondisi mental. Dataset yang digunakan adalah *Student Depression Dataset* dengan 28.008 data dan 18 fitur.

---

## 🗂️ Struktur Repository

```
├── student_depression.ipynb   # Notebook utama (EDA → Modeling → Tuning)
├── app.py                     # Aplikasi Streamlit
├── requirements.txt           # Dependensi Python
├── model/
│   ├── best_model.pkl         # Model terbaik (hasil tuning)
│   ├── scaler.pkl             # StandardScaler
│   ├── selected_features.pkl  # Daftar fitur terpilih
│   └── le_degree.pkl          # LabelEncoder untuk kolom Degree
└── README.md
```

---

## 🔬 Tahapan Proyek

### 1. EDA (Exploratory Data Analysis)
- Identifikasi missing values, distribusi data, anomali
- Visualisasi distribusi target dan hubungan fitur terhadap Depression
- **3 Insight Utama:**
  1. Pikiran bunuh diri adalah prediktor terkuat depresi
  2. Durasi tidur berkorelasi dengan status depresi
  3. Financial Stress dan Academic Pressure berkorelasi positif dengan depresi

### 2. Direct Modeling (Tanpa Preprocessing)
7 model dibangun sebagai baseline: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, Gradient Boosting, XGBoost

### 3. Modeling dengan Preprocessing
Data dibersihkan (imputasi, encoding, scaling) lalu model yang sama dilatih ulang untuk mengukur peningkatan performa.

### 4. Feature Selection
Menggunakan 3 metode: Korelasi Pearson, ANOVA F-test (SelectKBest), dan Feature Importance dari Random Forest.

### 5. Hyperparameter Tuning
RandomizedSearchCV pada Random Forest, XGBoost, dan Gradient Boosting.

### 6. Deployment
Aplikasi Streamlit di-deploy ke Streamlit Cloud.

---

## 🚀 Cara Menjalankan Lokal

```bash
# Clone repository
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME

# Install dependensi
pip install -r requirements.txt

# Jalankan Streamlit
streamlit run app.py
```

> **Catatan:** Pastikan folder `model/` sudah terisi dengan file `.pkl` hasil menjalankan notebook terlebih dahulu.

---

## 📊 Dataset
- **Sumber:** Student Depression Dataset (Kaggle)
- **Jumlah data:** 28.008 baris
- **Fitur:** 17 fitur + 1 target (Depression)
- **Target:** Binary — 0 (Tidak Depresi), 1 (Depresi)

---

## 🛠️ Tech Stack
- Python 3.10+
- scikit-learn, XGBoost
- pandas, numpy
- matplotlib, seaborn
- Streamlit

---

## ⚠️ Disclaimer
Aplikasi ini **bukan** alat diagnosis klinis. Hanya bersifat indikatif. Untuk penanganan lebih lanjut, konsultasikan dengan tenaga profesional kesehatan mental.
