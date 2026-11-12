<div align="center">

# 🧪 Machine Learning Laboratory
### *Statistical Learning Paradigms, Empirical Diagnostics & Production-Grade Workflows*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](#)

<br/>

> **A standardized, reproducible coursework archive featuring convex optimization, parametric and non-parametric learning, validation protocols, and comprehensive diagnostic laboratory implementations.**

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
- [Curricular Notebooks & Experiments](#-curricular-notebooks--experiments)
- [Theoretical & Mathematical Formulations](#-theoretical--mathematical-formulations)
- [Repository File Hierarchy](#-repository-file-hierarchy)
- [Technical Stack & Environment Setup](#-technical-stack--environment-setup)
- [Scientific Rigor & Validation Standards](#-scientific-rigor--validation-standards)
- [Academic Integrity & Licensing](#-academic-integrity--licensing)

---

## 🔬 Executive Summary

This repository houses the official implementations and empirical analyses for the **Machine Learning Laboratory** coursework. Each notebook represents an isolated, end-to-end investigation covering core classical machine learning domains: ordinary least squares regression, L1 sparsity regularization, non-linear polynomial basis expansion, logistic classification with decision threshold optimization, and k-fold cross-validation pipelines.

### Core Curricular Pillars
* **Convex Optimization:** Normal equations, iteratively reweighted least squares, and coordinate descent for penalized objectives.
* **Leakage-Invariant Preprocessing:** Strict separation of training moments (mean μ, standard deviation σ) from validation and test partitions.
* **Resampling Diagnostics:** K-fold cross-validation for variance estimation and out-of-sample risk assessment.
* **Cost-Sensitive Decision Boundaries:** Optimal thresholding (tau*) on probability outputs beyond default 0.5 cutoffs.

---

## 🏛 Pipeline Architecture & Protocol

All implementations adhere to a standard leakage-free learning workflow:

```mermaid
flowchart LR
    A["Data Ingestion<br/>(Raw Corpus / UCI)"] --> B["Exploratory Diagnostics<br/>(Distributions & Correlation)"]
    B --> C["Stratified Partitioning<br/>(Train / Test Split - Seed: 42)"]
    C --> D["Feature Engineering<br/>(Scaling & Basis Expansion)"]
    D --> E["Model Estimation<br/>(Loss Minimization & Regularization)"]
    E --> F["Validation & Diagnostics<br/>(K-Fold CV, Residuals & ROC)"]
    F --> G["Final Evaluation<br/>(Metrics & Threshold Tuning)"]

    style A fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style B fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style C fill:#0f766e,stroke:#14b8a6,stroke-width:1px,color:#f8fafc
    style D fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style E fill:#0f766e,stroke:#14b8a6,stroke-width:1px,color:#f8fafc
    style F fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style G fill:#0369a1,stroke:#38bdf8,stroke-width:1px,color:#f8fafc
```

---

## 📊 Curricular Notebooks & Experiments

| Notebook File | Paradigm / Method | Key Mathematical Concepts | Primary Metrics | Status |
| :--- | :--- | :--- | :--- | :---: |
| [`Banknote_LogisticRegression.ipynb`](./Banknote_LogisticRegression.ipynb) | **Logistic Regression** | Sigmoid Activation, Binary Cross-Entropy, Decision Threshold Tuning (tau*) | Accuracy, Precision, Recall, F1-Score, ROC-AUC | ![Completed](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square) |
| [`LinearRegression.ipynb`](./LinearRegression.ipynb) | **Ordinary Least Squares** | Normal Equation w = (XᵀX)⁻¹Xᵀy, Gradient Descent Optimization | MSE, RMSE, MAE, R² Score | ![Completed](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square) |
| [`OLSLassoReg.ipynb`](./OLSLassoReg.ipynb) | **Lasso Regression (L1)** | OLS vs. L1 Regularization, Coordinate Descent, Feature Sparsity | Sparsity Ratio, MSE, Residual Plots | ![Completed](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square) |
| [`polyregression.ipynb`](./polyregression.ipynb) | **Polynomial Regression** | Polynomial Basis Expansions φ(x), Bias-Variance Tradeoff, Degree Tuning | Training vs. Testing MSE, Overfitting Analysis | ![Completed](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square) |
| [`k_fold_lin_reg.ipynb`](./k_fold_lin_reg.ipynb) | **K-Fold Cross-Validation** | Resampling Schemes, Out-of-Fold Error Estimation, Variance Diagnostics | Mean CV Score, Standard Deviation (σ) | ![Completed](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square) |
| [`endtoendMLProject.ipynb`](./endtoendMLProject.ipynb) | **End-to-End ML Pipeline** | Pipeline Composition, Feature Transforms, Hyperparameter Search, Inference | Comprehensive Metrics Suite | ![Completed](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square) |

---

## 🔬 Theoretical & Mathematical Formulations

### 1. Banknote Authentication (`Banknote_LogisticRegression.ipynb`)
Binary classification model predicting banknote validity based on Continuous Wavelet Transform (CWT) features:

```text
Hypothesis Function:
  p̂ = σ(wᵀx + b) = 1 / ( 1 + exp(-(wᵀx + b)) )

Loss Function (Binary Cross-Entropy with L2 Regularization):
  L(w, b) = - (1/N) ∑ [ yᵢ ln(p̂ᵢ) + (1 - yᵢ) ln(1 - p̂ᵢ) ] + (λ / 2) ||w||₂²
```

### 2. Linear Regression & Ordinary Least Squares (`LinearRegression.ipynb`)
Closed-form parameter estimation minimizing the residual sum of squares:

```text
Parameter Optimization:
  w* = argmin || y - Xw ||₂²

Closed-Form Normal Equation:
  w* = (Xᵀ X)⁻¹ Xᵀ y
```

### 3. Regularized Lasso Modeling (`OLSLassoReg.ipynb`)
Sparsity-inducing L1 penalty driving non-informative feature weights to zero:

```text
Lasso Objective Function:
  L_Lasso(w) = (1 / 2N) || y - Xw ||₂² + α ||w||₁

where:
  ||w||₁ = ∑ |wⱼ|  (L1 norm over all D feature weights)
```

### 4. Non-Linear Feature Mapping (`polyregression.ipynb`)
Higher-degree polynomial transformations modeling non-linear continuous relationships:

```text
Basis Function Transformation:
  φ(x) = [ 1,  x,  x²,  x³,  ...,  xᵈ ]ᵀ

Predicted Response:
  ŷ = wᵀ φ(x)
```

### 5. Resampling via K-Fold Validation (`k_fold_lin_reg.ipynb`)
Partitioning dataset D into K mutually disjoint validation folds D₁, D₂, ..., Dₖ:

```text
Cross-Validation Performance Estimator:
  CV_(K) = (1 / K) ∑ MSEₖ

Individual Fold Error:
  MSEₖ = (1 / |Dₖ|) ∑ ( yᵢ - f̂⁽⁻ᵏ⁾(xᵢ) )²   for all i ∈ Dₖ
```

---

## 📂 Repository File Hierarchy

```text
Machine-Learning-Lab/
├── Banknote_LogisticRegression.ipynb   # Logistic regression on banknote authentication
├── LinearRegression.ipynb             # Ordinary least squares & linear regression
├── OLSLassoReg.ipynb                  # OLS vs. Lasso (L1) regularization comparison
├── endtoendMLProject.ipynb            # Complete industrial ML pipeline implementation
├── k_fold_lin_reg.ipynb               # K-Fold cross-validation on regression models
├── polyregression.ipynb               # Polynomial regression and degree curve fitting
├── LICENSE.md                         # Open-source MIT License terms
└── README.md                          # Repository documentation & laboratory index
```

---

## 🛠 Technical Stack & Environment Setup

### Environment Dependencies

| Package | Version Target | Purpose |
| :--- | :--- | :--- |
| `python` | `>= 3.8` | Core runtime environment |
| `numpy` | `>= 1.20.0` | Matrix vectorization and linear algebra |
| `pandas` | `>= 1.3.0` | Data frame manipulation and statistical profiling |
| `scikit-learn` | `>= 0.24.0` | Machine learning models, pipelines, and metrics |
| `matplotlib` | `>= 3.4.0` | Visual diagnostics and scalar curve plotting |
| `seaborn` | `>= 0.11.0` | Statistical distributions and correlation heatmaps |

### Installation & Execution Guide

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Machine-Learning-Lab.git
cd Machine-Learning-Lab

# 2. Create and activate a Python virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (cmd / PowerShell):
.venv\Scripts\activate

# 3. Upgrade package installer
pip install --upgrade pip

# 4. Install essential scientific libraries
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# 5. Launch the Jupyter Notebook dashboard
jupyter notebook
```

---

## 📐 Scientific Rigor & Validation Standards

Every implementation in this repository follows three mandatory experimental protocols:

```text
+----------------------------------------------------------------------------------------+
|                               SCIENTIFIC AUDIT CRITERIA                                |
+----------------------------------------------------------------------------------------+
|  [01] LEAK-PROOF BOUNDARIES                                                           |
|       - Scaling parameters (μ, σ) are computed strictly on the training partition.    |
|       - Transformations are applied downstream to test partitions without refitting.  |
|                                                                                        |
|  [02] SEED-ANCHORED DETERMINISM                                                        |
|       - PRNG seeds are declared globally: random_state = 42.                           |
|       - Ensures identical data splits and reproducible numerical outcomes.             |
|                                                                                        |
|  [03] COMPREHENSIVE DIAGNOSTICS                                                        |
|       - Regression: Evaluated on MSE, RMSE, MAE, R², and residual error plots.         |
|       - Classification: Evaluated on Precision, Recall, F1, ROC-AUC, and Confusion    |
|         Matrices beyond simple raw accuracy.                                           |
+----------------------------------------------------------------------------------------+
```

---

## ⚖️ Academic Integrity & Licensing

* **Academic Honor Code:** This repository contains coursework developed for the **5th Semester CSE (AI & ML)** program at **The National Institute of Engineering, Mysuru**. It is published for educational demonstration, reference, and peer review. Direct plagiarism without derivation is strictly discouraged.
* **Licensing:** Distributed under the **[MIT License](./LICENSE.md)**. You are free to study, fork, and build upon this repository with proper attribution.

<div align="center">

---

**Department of Computer Science and Engineering (Artificial Intelligence & Machine Learning)**  
**The National Institute of Engineering, Mysuru — 570008**  
*Autonomous Institute under VTU, Belagavi | Approved by AICTE, New Delhi*

</div>
