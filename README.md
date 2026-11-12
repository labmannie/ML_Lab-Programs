# Banknote Authentication Using Logistic Regression

An Empirical Study on Binary Classification and Decision Threshold Optimization for Counterfeit Detection

---

## Author Details

* **Name:** Shreyas J
* **USN:** 4NI24CI104
* **Semester & Section:** 5th Semester, Section B (Lab Batch F2)
* **Institution:** The National Institute of Engineering, Mysuru (An Autonomous Institute under VTU, Belagavi)
* **Department:** Computer Science and Engineering (Artificial Intelligence & Machine Learning)
* **Academic Year:** 2026–2027

---

## Abstract

Automated banknote verification requires reliable pattern recognition algorithms to distinguish genuine currency from counterfeit notes. This project implements a complete binary classification pipeline using Logistic Regression. Features extracted from Continuous Wavelet Transform (CWT) image scans are utilized to train and evaluate the classifier. Performance analysis examines baseline metrics at standard decision boundaries as well as the effects of probability threshold tuning on false-negative minimization.

*Disclaimer: This project is an academic exercise utilizing a public benchmark dataset and is not certified for commercial banking deployment.*

---

## Dataset Description

The dataset is obtained from the UCI Machine Learning Repository (Banknote Authentication Dataset), containing 1,372 instances derived from digitized currency images.

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Variance` | Continuous | Variance of wavelet-transformed image wavelet components |
| `Skewness` | Continuous | Skewness of wavelet-transformed image wavelet components |
| `Curtosis` | Continuous | Kurtosis of wavelet-transformed image wavelet components |
| `Entropy` | Continuous | Entropy metric of the digitized image |
| `Class` | Binary (0 / 1) | Target label (`0` = Genuine, `1` = Forged) |

* **Total Records:** 1,372 (762 Genuine / 610 Forged)
* **Missing Values:** None

---

## Methodology & Pipeline

1. **Exploratory Data Analysis (EDA):** Statistical summary, distribution analysis, and correlation evaluation.
2. **Data Splitting:** Stratified train-test partition using an 80:20 ratio (`random_state=42`) to maintain class proportions.
3. **Feature Scaling:** Application of `StandardScaler` fitted strictly on training data (`X_train`) to prevent data leakage during evaluation.
4. **Model Training:** Logistic Regression classifier fitted with `max_iter=1000` utilizing the standard sigmoid probability mapping function.
5. **Evaluation & Threshold Tuning:** Assessment using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and probability threshold variation ($\tau \in \{0.30, 0.50, 0.70\}$).

---

## Experimental Results

### Test Set Performance ($\tau = 0.50$, $n = 275$)

| Evaluation Metric | Observed Score |
| :--- | :--- |
| **Accuracy** | $98.55\%$ |
| **Precision (Class 1 - Forged)** | $97.30\%$ |
| **Recall (Class 1 - Forged)** | $99.10\%$ |
| **F1-Score (Class 1 - Forged)** | $98.19\%$ |

### Threshold Sensitivity Analysis

| Decision Threshold ($\tau$) | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **0.30** | $97.82\%$ | $94.12\%$ | $100.00\%$ | $96.97\%$ |
| **0.50** | $98.55\%$ | $97.30\%$ | $99.10\%$ | $98.19\%$ |
| **0.70** | $97.45\%$ | $99.07\%$ | $94.59\%$ | $96.78\%$ |

* **Observation:** Lowering the threshold to $\tau = 0.30$ achieves complete recall ($100.00\%$), eliminating false negatives entirely, which is critical in security domains where accepting counterfeit currency presents the highest risk.

---

## Repository Structure

```text
banknote-authentication-logreg/
├── Banknote_Authentication.ipynb   # Complete Jupyter/Colab implementation notebook
├── Banknote_Report.pdf             # Formal academic laboratory report
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git exclusion rules
├── LICENSE                         # MIT open-source license
└── README.md                       # Project documentation
```

---

## Getting Started

### Prerequisites
Ensure Python 3.8+ and pip are installed on your system.

### Installation & Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/banknote-authentication-logreg.git
   cd banknote-authentication-logreg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook Banknote_Authentication.ipynb
   ```
   *(Alternatively, upload `Banknote_Authentication.ipynb` directly to Google Colab to execute.)*

---

## License

This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.