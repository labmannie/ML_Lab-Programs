<div align="center">

# 🧪 Machine Learning Laboratory
### *Statistical Learning Paradigms, Empirical Diagnostics & Production-Grade Workflows*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab%20%2F%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](#)

<br/>

> **A standardized, reproducible coursework archive featuring convex optimization, parametric and non-parametric learning, validation protocols, and comprehensive diagnostic laboratory reports.**

---

### **Investigator & Academic Credentials**

| Attribute | Profile Record |
| :--- | :--- |
| **Candidate Name** | **Shreyas J** |
| **University Seat Number (USN)** | `4NI24CI104` |
| **Curricular Standing** | 5th Semester — Section B *(Laboratory Batch F2)* |
| **Department** | Computer Science and Engineering *(Artificial Intelligence & Machine Learning)* |
| **Institutional Affiliation** | **The National Institute of Engineering (NIE), Mysuru**<br>*(An Autonomous Institute under Visvesvaraya Technological University, Belagavi)* |
| **Academic Session** | 2026 – 2027 |

---

</div>

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Pipeline Architecture & Protocol](#-pipeline-architecture--protocol)
- [Curricular Experiments Index](#-curricular-experiments-index)
- [Empirical Experiment Dossier](#-empirical-experiment-dossier)
  - [Exp 01: Banknote Authentication (Binary Logistic Regression)](#exp-01-banknote-authentication-binary-logistic-regression)
- [Repository Directory Schema](#-repository-directory-schema)
- [Technical Stack & Environment Provisioning](#-technical-stack--environment-provisioning)
- [Scientific Rigor & Validation Standards](#-scientific-rigor--validation-standards)
- [Academic Integrity & Licensing](#-academic-integrity--licensing)

---

## 🔬 Executive Summary

This repository represents the official laboratory record for the **Machine Learning Laboratory** coursework. It bridges mathematical theory with production-grade engineering principles. Every experiment is structured as an isolated, end-to-end investigation—progressing from raw statistical distributions through mathematical formulation, exploratory diagnostics, leak-free preprocessing, cross-validation, and decision-boundary threshold calibration.

### Core Curricular Pillars
* **Convex & Non-Convex Optimization:** Gradient descent, iteratively reweighted least squares, and margin maximization.
* **Leakage-Invariant Data Processing:** Strict encapsulation of inductive biases within training folds.
* **Cost-Sensitive Decision Theory:** Optimization of operational decision thresholds ($\tau^*$) based on non-symmetric false-positive vs. false-negative penalty matrices.
* **Comprehensive Residual Diagnostics:** ROC-AUC, Precision-Recall curves, Brier score calibration, and confusion matrices.

---

## 🏛 Pipeline Architecture & Protocol

Every module in this repository adheres to a strict linear pipeline ensuring reproducible outcomes and mathematical integrity:

```mermaid
flowchart LR
    A[Data Ingestion<br/><i>UCI / OpenML Archives</i>] --> B[Exploratory Diagnostics<br/><i>Skewness, Kurtosis, Heatmaps</i>]
    B --> C[Stratified Splitting<br/><i>Seed: 42 | No Leakage</i>]
    C --> D[Feature Conditioning<br/><i>Fit on Train, Apply to Test</i>]
    D --> E[Parametric Estimator<br/><i>Loss Optimization & Regularization</i>]
    E --> F[Decision Calibration<br/><i>Threshold Tuning & Trade-offs</i>]
    F --> G[Comprehensive Reporting<br/><i>Matrices, Curves & Viva Notes</i>]

    style A fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style B fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style C fill:#0f766e,stroke:#14b8a6,stroke-width:1px,color:#f8fafc
    style D fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style E fill:#0f766e,stroke:#14b8a6,stroke-width:1px,color:#f8fafc
    style F fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style G fill:#0369a1,stroke:#38bdf8,stroke-width:1px,color:#f8fafc
```

### Protocol Invariants
1. **No Data Leakage:** Imputation, scaling parameters $(\mu, \sigma)$, and vectorizers are learned solely on $\mathcal{D}_{\text{train}}$ and frozen prior to inferencing on $\mathcal{D}_{\text{test}}$.
2. **Determinism:** Random seeds are anchored globally (`random_state=42`) across array operations, estimators, and partition generators.
3. **Multi-Metric Assessment:** No single metric is evaluated in isolation; classification tasks require simultaneous inspection of Precision, Recall, $F_{\beta}$, ROC-AUC, and Matthews Correlation Coefficient (MCC).

---

## 📊 Curricular Experiments Index

| # | Experiment Module | Underlying Paradigm | Target Dataset | Primary Evaluation Metric | Directory Reference | Status |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: |
| **01** | **Banknote Authentication** | Logistic Regression ($\ell_2$-regularized, Sigmoid Activation, Decision Optimization) | UCI Banknote Authentication ($N=1,372$) | ROC-AUC / Precision-Recall / $F_1$ | [`/Exp01_Banknote_Authentication`](./Exp01_Banknote_Authentication) | ![Completed](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square) |
| **02** | *Upcoming Experiment 02* | *Linear / Ridge / Lasso Regression or $k$-NN* | *Benchmark Corpus* | *MSE / $R^2$ / MAPE* | [`/Exp02_Directory_Name`](./Exp02_Directory_Name) | ![Planned](https://img.shields.io/badge/Status-Planned-64748b?style=flat-square) |
| **03** | *Upcoming Experiment 03* | *Support Vector Classifier / Kernel Trick* | *Benchmark Corpus* | *Hinge Loss / Margin / Accuracy* | [`/Exp03_Directory_Name`](./Exp03_Directory_Name) | ![Planned](https://img.shields.io/badge/Status-Planned-64748b?style=flat-square) |
| **04** | *Upcoming Experiment 04* | *Decision Trees / Random Forests / Bagging* | *Benchmark Corpus* | *Gini Impurity / Out-of-Bag Error* | [`/Exp04_Directory_Name`](./Exp04_Directory_Name) | ![Planned](https://img.shields.io/badge/Status-Planned-64748b?style=flat-square) |
| **05** | *Upcoming Experiment 05* | *Unsupervised Learning / $k$-Means / PCA* | *Benchmark Corpus* | *Silhouette Score / Inertia / EVR* | [`/Exp05_Directory_Name`](./Exp05_Directory_Name) | ![Planned](https://img.shields.io/badge/Status-Planned-64748b?style=flat-square) |

---

## 🔬 Empirical Experiment Dossier

### Exp 01: Banknote Authentication (Binary Logistic Regression)

* **Objective:** Distinguish between genuine and counterfeit banknotes using continuous features extracted from Continuous Wavelet Transform (CWT) digitized images ($400 \times 400$ px).
* **Mathematical Formulation:**
  $$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + \exp\left(-(\mathbf{w}^T \mathbf{x} + b)\right)}$$
  $$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i)\ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
* **Features Analyzed:**
  1. *Variance* of Wavelet Transformed image ($x_1$)
  2. *Skewness* of Wavelet Transformed image ($x_2$)
  3. *Curtosis* of Wavelet Transformed image ($x_3$)
  4. *Entropy* of the image ($x_4$)
* **Key Findings:** 
  * Features $x_1$ (variance) and $x_2$ (skewness) provide the strongest linear discriminative hyperplane.
  * Standardizing features via $z = \frac{x - \mu}{\sigma}$ accelerated gradient convergence and avoided vanishing gradients.
  * Optimal operational decision threshold $\tau^* \approx 0.48$ maximizes the $F_1$-score while penalizing counterfeit false approvals.
* **Deliverables:** Executable Colab Notebook, Confusion Matrix visualizer, Precision-Recall & ROC Curves, and Formal Academic PDF.

---

## 📂 Repository Directory Schema

```text
Machine-Learning-Lab/
├── .github/                               # CI workflows and issue templates
├── Exp01_Banknote_Authentication/         # Experiment 01 Root
│   ├── Banknote_Authentication.ipynb      # Annotated, executable experiment notebook
│   ├── Banknote_Report.pdf                # Formal technical documentation & viva guide
│   └── README.md                          # Experiment-level empirical methodology
│
├── Exp02_Directory_Name/                  # Experiment 02 Root (Staged)
│   └── ...
│
├── Exp03_Directory_Name/                  # Experiment 03 Root (Staged)
│   └── ...
│
├── .gitignore                             # OS and IDE ignore artifacts (.ipynb_checkpoints, etc.)
├── LICENSE                                # Open-source MIT Licensing
├── requirements.txt                       # Locked dependencies manifest
└── README.md                              # Institutional master documentation
```

---

## 🛠 Technical Stack & Environment Provisioning

### Runtime Dependencies

| Domain | Library | Version Target | Functional Purpose |
| :--- | :--- | :--- | :--- |
| **Engine** | `Python` | `>= 3.8.10` | Base interpreted runtime |
| **Array Math** | `numpy` | `>= 1.20.0` | Matrix vectorization and linear algebra operations |
| **DataFrames** | `pandas` | `>= 1.3.0` | Ingestion, feature profiling, and aggregation |
| **Modeling** | `scikit-learn` | `>= 0.24.0` | Estimators, transformers, and validation metrics |
| **Visualization** | `matplotlib` | `>= 3.4.0` | Publication-ready scalar plots, ROC-AUC curves |
| **Statistical Viz** | `seaborn` | `>= 0.11.0` | Bivariate distributions, correlation heatmaps |

### Local Reproducibility Protocol

Follow these commands to configure an isolated execution environment on Linux, macOS, or Windows:

```bash
# 1. Clone the master repository
git clone https://github.com/<your-username>/Machine-Learning-Lab.git
cd Machine-Learning-Lab

# 2. Initialize a dedicated virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (cmd / PowerShell):
.venv\Scripts\activate

# 4. Upgrade packaging toolchains
pip install --upgrade pip setuptools wheel

# 5. Install locked production dependencies
pip install -r requirements.txt

# 6. Launch the local interactive server
jupyter lab
```

---

## 📐 Scientific Rigor & Validation Standards

Every laboratory submission cataloged within this repository is audited against three mandatory validation principles:

```text
+----------------------------------------------------------------------------------------+
|                               SCIENTIFIC AUDIT CRITERIA                                |
+----------------------------------------------------------------------------------------+
|  [01] LEAK-PROOF BOUNDARIES                                                           |
|       - Transformations (e.g., StandardScaler, SimpleImputer) calculate moments (μ, σ) |
|         exclusively on the training subset.                                           |
|       - No global scaling prior to cross-validation or train/test splitting.           |
|                                                                                        |
|  [02] SEED-ANCHORED DETERMINISM                                                        |
|       - PRNG seeds are declared globally: SEED = 42.                                   |
|       - Stratification preserves empirical class priors: y_train ≈ y_test.             |
|                                                                                        |
|  [03] BEYOND RAW ACCURACY METRICS                                                      |
|       - Precision: TP / (TP + FP)  |  Recall: TP / (TP + FN)                          |
|       - Balanced F_beta scores, Confusion Matrices, ROC-AUC, and PR-AUC curves.        |
+----------------------------------------------------------------------------------------+
```

---

## ⚖️ Academic Integrity & Licensing

* **Honor Code & Academic Integrity:** This repository is published to demonstrate statistical learning workflows, algorithmic experiments, and technical documentation conducted as part of the **5th Semester CSE (AI & ML)** curriculum at **The National Institute of Engineering, Mysuru**. Current students are advised to consult their departmental lab guidelines and avoid direct code submission without individual derivation.
* **Licensing:** The underlying codebase and Jupyter notebooks are distributed under the **[MIT License](./LICENSE)**. You are free to inspect, modify, and reference this code with appropriate academic attribution.

<div align="center">

---

**Department of Computer Science and Engineering (Artificial Intelligence & Machine Learning)**  
**The National Institute of Engineering, Mysuru — 570008**  
*Autonomous Institute under VTU, Belagavi | Approved by AICTE, New Delhi*

</div>
