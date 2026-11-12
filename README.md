<div align="center">

# 💵 Banknote Authentication using Logistic Regression

**Automated Counterfeit Currency Detection via Wavelet Transform Features & Threshold Optimization**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)

---

*Developed by **Shreyas J** (`4NI24CI104`) • Dept. of CSE (AI & ML), The National Institute of Engineering, Mysuru*

</div>

---

## 📌 Overview

Counterfeit currency detection is a vital problem in automated financial systems and high-speed banknote sorting machines. This project builds an end-to-end, interpretable **Binary Classification** pipeline using **Logistic Regression** to classify banknotes as **Genuine** (`0`) or **Forged** (`1`) based on statistical features extracted from Continuous Wavelet Transform (CWT) image scans.

> ⚠️ **Academic Disclaimer:** *This project is built strictly for educational and research purposes using a public benchmark dataset. It is not intended for production deployment in banking hardware or ATMs.*

---

## ✨ Key Highlights

- 🛡️ **Leakage-Free Preprocessing:** `StandardScaler` fitted strictly on the training split ($80\%$) before transforming unseen test data ($20\%$).
- 📊 **High Baseline Accuracy:** Achieves **$98.55\%$ Accuracy** and **$98.19\%$ $F_1$-Score** at the standard $\tau = 0.50$ probability threshold.
- 🎯 **Zero False-Negative Security Mode:** Calibrating the decision threshold to $\tau = 0.30$ reaches **$100.00\%$ Recall**, ensuring zero counterfeit notes pass undetected.
- 📈 **Rich Visualizations:** Includes class balance plots, correlation heatmaps, confusion matrices, and precision-recall threshold curves.

---

## 🗂️ Dataset Summary

Sourced from the **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)**, containing **$1,372$ instances** ($762$ Genuine, $610$ Forged) with zero missing values:

| Feature | Type | Description |
| :--- | :---: | :--- |
| `Variance` | `float64` | Variance of the wavelet-transformed image |
| `Skewness` | `float64` | Skewness of the wavelet-transformed image |
| `Curtosis` | `float64` | Kurtosis (peakedness) of the wavelet-transformed image |
| `Entropy` | `float64` | Entropy of the digitized banknote image |
| `Class` | `int64` | Target label (`0` = Genuine, `1` = Forged) |

---

## 📊 Model Performance

### 1️⃣ Test Set Metrics ($\tau = 0.50$, $n = 275$)

| Metric | Score | Status |
| :--- | :---: | :---: |
| **Accuracy** | `98.55%` | 🟢 High |
| **Precision (Forged)** | `97.30%` | 🟢 High |
| **Recall (Forged)** | `99.10%` | 🟢 High |
| **$F_1$-Score (Forged)** | `98.19%` | 🟢 High |

### 2️⃣ Threshold Sensitivity Comparison

| Threshold ($\tau$) | Accuracy | Precision | Recall | $F_1$-Score | Best Use Case |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`0.30`** | `97.82%` | `94.12%` | **`100.00%`** | `96.97%` | 🔒 Maximum security (zero counterfeit leakage) |
| **`0.50`** | **`98.55%`** | `97.30%` | `99.10%` | **`98.19%`** | ⚖️ Balanced classification |
| **`0.70`** | `97.45%` | **`99.07%`** | `94.59%` | `96.78%` | 🚦 Minimizing false alarms |

---

## 📁 Repository Structure

```text
banknote-authentication-logreg/
├── Banknote_Authentication.ipynb   # Complete Jupyter/Colab notebook with code & plots
├── Banknote_Report.pdf             # Detailed academic project report
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignored system & cache files
├── LICENSE                         # MIT License
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/banknote-authentication-logreg.git
cd banknote-authentication-logreg
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Open `Banknote_Authentication.ipynb` in **Jupyter Notebook**, **VS Code**, or **Google Colab** and run all cells sequentially.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ using Python & Scikit-Learn • Star ⭐ this repo if you found it helpful!</sub>
</div>