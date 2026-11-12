# Machine Learning Laboratory Repository

**A Comprehensive Collection of Supervised and Unsupervised Machine Learning Algorithms, Empirical Analyses, and Laboratory Documentation**

---

## Author & Institutional Information

* **Student Name:** Shreyas J
* **USN:** 4NI24CI104
* **Semester & Section:** 5th Semester, Section B (Lab Batch F2)
* **Institution:** The National Institute of Engineering, Mysuru (An Autonomous Institute under VTU, Belagavi)
* **Department:** Computer Science and Engineering (Artificial Intelligence & Machine Learning)
* **Academic Year:** 2026–2027

---

## Repository Overview

This repository serves as a centralized archive for the **Machine Learning Laboratory** coursework. It houses end-to-end Python implementations, Jupyter/Google Colab notebooks, exploratory data analyses, model evaluations, and formal academic laboratory reports covering core paradigms of statistical learning.

Each experiment is organized into its dedicated directory containing executable code, analytical visualizations, and formal documentation.

---

## Standard Machine Learning Workflow Architecture

All experiments in this repository adhere to a standardized, leakage-free machine learning pipeline:

```text
+-----------------------+     +-----------------------+     +-----------------------+
|   Data Acquisition    | --> | Exploratory Analysis  | --> |  Stratified Split     |
|  (UCI / Standard DS)  |     |  (EDA & Distributions)|     |   (Train / Test)      |
+-----------------------+     +-----------------------+     +-----------------------+
                                                                        |
                                                                        v
+-----------------------+     +-----------------------+     +-----------------------+
| Threshold & Hyper-    | <-- |   Model Evaluation    | <-- | Feature Preprocessing |
| parameter Tuning      |     | (Metrics & Matrices)  |     | (Scaling / Encoding)  |
+-----------------------+     +-----------------------+     +-----------------------+
```

---

## Index of Laboratory Experiments

| Exp. No. | Experiment Title | Algorithm / Paradigm | Dataset | Directory | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **01** | **Banknote Authentication** | Logistic Regression (Binary Classification & Threshold Optimization) | UCI Banknote Authentication ($N = 1,372$) | [`/Exp01_Banknote_Authentication`](./Exp01_Banknote_Authentication) | Completed |
| **02** | *Experiment 02 (Placeholder)* | *Update Algorithm Name* | *Update Dataset* | [`/Exp02_Directory_Name`](./Exp02_Directory_Name) | Planned |
| **03** | *Experiment 03 (Placeholder)* | *Update Algorithm Name* | *Update Dataset* | [`/Exp03_Directory_Name`](./Exp03_Directory_Name) | Planned |
| **04** | *Experiment 04 (Placeholder)* | *Update Algorithm Name* | *Update Dataset* | [`/Exp04_Directory_Name`](./Exp04_Directory_Name) | Planned |
| **05** | *Experiment 05 (Placeholder)* | *Update Algorithm Name* | *Update Dataset* | [`/Exp05_Directory_Name`](./Exp05_Directory_Name) | Planned |

*(Note: Update rows 02–05 as additional laboratory experiments are completed and uploaded.)*

---

## Repository Directory Structure

```text
Machine-Learning-Lab/
│
├── Exp01_Banknote_Authentication/
│   ├── Banknote_Authentication.ipynb      # Executable Jupyter/Colab notebook
│   ├── Banknote_Report.pdf                # Formal laboratory report & viva notes
│   └── README.md                          # Experiment-specific documentation
│
├── Exp02_Directory_Name/                  # Future lab experiment directory
│   └── ...
│
├── requirements.txt                       # Unified Python environment dependencies
├── .gitignore                             # Git exclusion rules for checkpoints & cache
├── LICENSE                                # MIT open-source license
└── README.md                              # Master repository documentation
```

---

## Technical Stack & Environment Setup

### Core Dependencies
* **Language:** Python 3.8+
* **Numerical & Tabular Computation:** `numpy`, `pandas`
* **Machine Learning & Preprocessing:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Interactive Environment:** `jupyter`, Google Colaboratory

### Local Installation & Execution

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/Machine-Learning-Lab.git
   cd Machine-Learning-Lab
   ```

2. **Create and Activate a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Linux/macOS
   venv\Scripts\activate           # On Windows
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Navigate to any experiment folder (e.g., `Exp01_Banknote_Authentication/`) and open the corresponding `.ipynb` file.

---

## Evaluation & Documentation Standards

Every laboratory submission in this repository follows strict empirical standards:
1. **Zero Data Leakage:** Feature scalers, encoders, and imputers are fitted exclusively on training partitions (`X_train`) prior to transforming test sets (`X_test`).
2. **Reproducibility:** Fixed random seeds (`random_state=42`) are enforced across train-test splits and stochastic estimators.
3. **Comprehensive Metrics:** Models are evaluated beyond raw accuracy using Precision, Recall, F1-Score, Confusion Matrices, ROC-AUC curves, or Regression Error metrics ($R^2$, MSE, RMSE) as appropriate to the task.

---

## Disclaimer & License

* **Academic Disclaimer:** All implementations and reports in this repository are developed strictly for educational and academic coursework purposes using public benchmark datasets.
* **License:** Distributed under the **MIT License**. See the [`LICENSE`](./LICENSE) file for complete terms and conditions.