# Predicting Employee Attrition Using Behavioural Data

A machine learning system that identifies employees at risk of leaving by analyzing behavioural patterns. Built with a full data science pipeline — from synthetic data generation to an interactive Flask web application with SHAP-based interpretability.
1

## Overview

Employee attrition disrupts business stability and drives up operational costs. This project uses machine learning on synthetic behavioural employee data to predict attrition risk and provide interpretable, actionable insights for HR teams.

**Key Features:**
- Synthetic dataset generation for privacy-safe modeling
- 8 ML models trained and compared head-to-head
- SHAP and Permutation Importance for model explainability
- Flask web app supporting both single-entry and bulk CSV predictions
- Full evaluation suite: accuracy, precision, recall, F1, ROC, PR curves

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| ML & Data | scikit-learn, XGBoost, pandas, NumPy |
| Interpretability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Notebook | Jupyter |
| Version Control | Git / GitHub |

---

## Project Structure

```
employee-attrition/
│
├── data/
│   └── employees.csv              # Generated synthetic dataset
│
├── figures/                       # EDA and evaluation plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── shap_summary.png
│   └── permutation_importance_processed.png
│
├── models/                        # Trained model pickle files
│   └── *.pkl
│
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
│
├── results/
│   └── evaluation.csv             # Model comparison metrics
│
├── src/
│   ├── generate_data.py           # Synthetic dataset generation
│   ├── preprocess.py              # Data cleaning & feature engineering
│   ├── visualize.py               # EDA visualizations
│   ├── train_models.py            # Model training
│   ├── evaluate.py                # Model evaluation
│   └── interpret.py               # SHAP & permutation importance
│
├── templates/                     # Flask HTML templates
├── app.py                         # Flask application entry point
├── requirements.txt
└── README.md
```

---

## Setup & Installation

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/your-username/employee-attrition.git
cd employee-attrition

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run each step in sequence, or run only the steps you need.

### Step 1 — Generate Data
```bash
python src/generate_data.py
# Output: data/employees.csv
```

### Step 2 — Preprocess
```bash
python src/preprocess.py
```

### Step 3 — Exploratory Data Analysis
```bash
jupyter notebook notebooks/eda.ipynb
# Or run the script version:
python src/visualize.py
```

### Step 4 — Train Models
```bash
python src/train_models.py
# Output: models/*.pkl
```

### Step 5 — Evaluate
```bash
python src/evaluate.py
# Output: results/evaluation.csv + figures/
```

### Step 6 — Interpret
```bash
python src/interpret.py
# Output: figures/shap_summary.png, figures/permutation_importance_processed.png
```

### Step 7 — Launch Web App
```bash
python app.py
# Visit: http://127.0.0.1:5000
```

---

## ML Pipeline

### Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| K-Nearest Neighbors (KNN) | Distance-based |
| Support Vector Machine (SVM) | Kernel-based classifier |
| Decision Tree | Interpretable tree model |
| Random Forest | Ensemble of trees |
| Gradient Boosting | Sequential boosting |
| AdaBoost | Adaptive boosting |
| XGBoost | Optimized gradient boosting |

### Pipeline Steps

1. **Data Generation** — Synthetic employee dataset with behavioural features
2. **Preprocessing** — Missing value handling, categorical encoding, feature scaling
3. **EDA** — Distribution analysis, correlation heatmaps, attrition breakdown
4. **Training** — All 8 models trained on preprocessed data
5. **Evaluation** — Compared on accuracy, precision, recall, F1, ROC-AUC
6. **Interpretability** — SHAP values and permutation importance for the best model
7. **Deployment** — Flask app for interactive prediction

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home / landing page |
| `GET` | `/predict` | Single employee prediction form |
| `POST` | `/predict` | Submit employee data, returns prediction |
| `POST` | `/predict/bulk` | Upload CSV for batch predictions |
| `GET` | `/results` | View latest prediction results |
| `GET` | `/analysis` | SHAP and feature importance visualizations |

### CSV Format for Bulk Upload

```csv
age,department,job_role,monthly_income,overtime,years_at_company,...
34,Sales,Sales Executive,5000,Yes,3,...
```

> Column names must match the feature names used during training. Refer to `data/employees.csv` for the full schema.

---

## Results & Evaluation

Model performance is stored in `results/evaluation.csv`. Key visualizations generated:

- **Confusion Matrix** — `figures/confusion_matrix.png`
- **ROC Curve** — `figures/roc_curve.png`
- **Precision-Recall Curve** — `figures/pr_curve.png`
- **SHAP Summary Plot** — `figures/shap_summary.png`
- **Permutation Importance** — `figures/permutation_importance_processed.png`

---

## 📸 Screenshots

### 🏠 Homepage
![Homepage](figures/homepage.png)

### 📝 Prediction form Page
![Data Input Page](figures/data_input.png)

### 📊 Dashboard Chart
![Dashboard Chart](figures/dashboard_chart.png)

### 📋 Attrition Table
![Attrition Table](figures/attrition_table.png)

---

## Future Scope

- **Cloud Deployment** — Host on AWS / GCP for remote access and scalability
- **Role-Based Access Control** — Separate views for HR managers and admins
- **Real-Time Predictions** — Connect to live HR systems for continuous monitoring
- **Sentiment Analysis** — Incorporate employee survey and email data
- **Retention Strategy Module** — Suggest actions based on predicted attrition drivers

---

*This project uses synthetic data and does not involve any real employee information.*
