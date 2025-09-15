# 🛫 Travel Insurance Claim Prediction

## 1. Project Overview

Perusahaan asuransi perjalanan menghadapi tantangan dalam mengelola risiko klaim yang dapat berdampak pada profitabilitas. Sebagian besar nasabah tidak mengajukan klaim, namun ketika klaim terjadi, biayanya bisa signifikan.

Oleh karena itu, perusahaan membutuhkan model prediksi klaim yang dapat membantu mengidentifikasi faktor risiko dan mendukung proses pengambilan keputusan bisnis.

**Key Objectives:**

1. Membangun model prediksi klaim asuransi perjalanan.
2. Mengevaluasi performa model dengan metrik klasifikasi dan analisis cost-benefit.
3. Menginterpretasi faktor-faktor yang paling memengaruhi klaim.
4. Memberikan insight bisnis dari hasil analisis model.

---

## 2. Dataset Overview

Dataset setelah proses data cleaning terdiri dari **38,892 baris** dengan fitur sebagai berikut:

* **Fitur Kategorikal**:

  * Agency
  * Agency Type
  * Distribution Channel
  * Product Name
  * Destination

* **Fitur Numerikal**:

  * Duration
  * Net Sales
  * Commission
  * Age

* **Target Variable**:

  * `Claim` → 0 (*No Claim*), 1 (*Claim*)

Tantangan utama pada dataset adalah **imbalance class** (mayoritas tidak melakukan klaim).

---

## 3. Technologies Used

* **Programming Language**: Python
* **Machine Learning**: scikit-learn, imbalanced-learn
* **Visualization**: Matplotlib, Seaborn
* **Interactive App**: Streamlit
* **Development**: Jupyter Notebook

---

## 4. Project Structure

```bash
📁 travel-insurance-claim/
├── 📂 dataset/                           # dataset folder
    ├── cleaned_travel_insurance.csv
    ├── data_travel_insurance.csv          # raw data
├── 📓 Travel_Insurance_Model_Report.ipynb # analysis & modeling report
├── 📝 README.md                          # project documentation
├── 📄 requirements.txt                   # dependencies
├── 💻 streamlit_travel_insurance.py      # streamlit app
├── 📦 travel_insurance_logreg.sav        # saved logistic regression model
```

### Deliverables

* \[Interactive App with Streamlit](https://travel-insurance-model-wcxsbxbd7vakhjepuvtmqc.streamlit.app/)
* \[Presentation Link] 

---

## 5. Summary of Findings

### 5.1 Business Insights

* Mayoritas data tidak melakukan klaim → imbalance yang cukup signifikan.
* Faktor yang cukup berpengaruh terhadap klaim adalah **Product Name**, **Duration**, dan **Net Sales**.
* Logistic Regression dapat menjadi baseline yang baik karena sifatnya interpretable.
* Namun performa masih terbatas pada kelas minoritas (Claim).

---

### 5.2 Actionable Recommendations

**Dari sisi Model**

* Eksperimen lebih lanjut dengan model non-linear seperti Random Forest, Gradient Boosting, atau XGBoost untuk menangkap pola yang lebih kompleks.
* Mencoba menggunakan teknik handling imbalance lain seperti ADASYN atau Cost-sensitive learning.
* Melakukan feature engineering untuk membuat variabel baru atau melakukan grouping data destinasi menjadi kelompok regional sehingga dapat meningkatkan kualitas prediksi.

**Dari sisi Bisnis**

* Perusahaan dapat menggunakan model sebagai alat bantu screening awal dalam membantu proses pengambilan keputusan strategi perusahaan.
* Melakukan integrasi model dengan sistem operasional. Dengan memberikan flag/tanda pada nasabah dengan risiko tinggi supaya dilakukan pengecekan lebih lanjut.
* Mengembangkan kebijakan premi yang adaptif sehingga disesuaikan berdasarkan tingkat risiko klaim.
* Melakukan pengumpulan data tambahan, misal frekuensi klaim dari nasabah atau dari sisi demografi nasabah untuk memperkaya data training model.
* Monitoring performa model, dengan evaluasi secara periodik. Hal ini untuk memastikan model tetap relevan seiring perubahan kondisi saat ini.

---

## 6. Conclusion

Proyek ini menunjukkan bahwa model machine learning, meskipun sederhana seperti Logistic Regression, mampu memberikan insight yang berguna dalam memprediksi klaim asuransi perjalanan. Hasil ini dapat menjadi fondasi awal sebelum mengembangkan model yang lebih kompleks dan robust untuk kebutuhan bisnis nyata.

---
