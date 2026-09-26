# Kaggle Playground Series S6E3 — Predict Customer Churn

**Competition:** [PS S6E3 — Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3/overview)  
**Task:** Binary classification → `Yes` / `No` (will the customer cancel?)  
**Metric:** ROC-AUC  
**Dataset:** ~594,000 rows, telecom customer data  
**Result:** Score **0.91822** — Top 4%

> My first Kaggle Playground competition. Focused on learning the full EDA pipeline, building my first engineered features, and understanding the end-to-end ML workflow.

---

## Project Structure

```
predict_customer_churn_s6e3/
├── README.md
└── notebooks/
    ├── 01_EDA.ipynb          # Exploratory Data Analysis (Denis)
    └── 02_ML_Modeling.ipynb  # ML Pipeline (Denis + Claude AI)
```

Data: download from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e3/data) and place in `data/raw/`.

---

## Role Split

**Denis** — EDA and feature engineering (notebook 01, independent work). Business hypotheses, segment analysis, and all engineered features.  
**ML pipeline** — built with significant AI assistance (Claude AI). I worked through each step to understand it, but the modeling decisions were not fully independent.

---

## Dataset

| Column | Type | Notes |
|--------|------|-------|
| `tenure` | Numeric | Months as a customer — strongest single predictor (corr = −0.42) |
| `MonthlyCharges` | Numeric | Monthly bill — churned customers pay ~$20 more on average |
| `TotalCharges` | Numeric | Near-redundant: corr(tenure × MonthlyCharges, TotalCharges) = 0.99 — dropped |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaymentMethod` | Categorical | Electronic check is a strong churn signal |
| `InternetService` | Categorical | Fiber optic = highest churn rate (41.5%) |
| `SeniorCitizen` | Binary | 11% of customers, 50% churn rate |
| `Partner`, `Dependents` | Binary | Family structure — key retention signal |
| `Churn` | Target | Yes (22.5%) / No (77.5%) |

---

## EDA Key Findings

### 1. Senior citizens churn at 2.6× the rate of younger customers

| Segment | Churn Rate | Share of Base | Revenue Share |
|---------|-----------|---------------|---------------|
| Younger | 19.0% | 88.6% | 87.3% |
| Senior | 50.0% | 11.4% | 12.7% |

Seniors pay more on average ($2,775 vs. $2,458 total charges) but have shorter median tenure (25 months vs. 37 months). The driver is not age itself — it's the combination of expensive services, no tech support, and short-term contracts.

### 2. Family status is the strongest retention signal

| Family Level | Churn Rate |
|-------------|-----------|
| Alone (no partner, no dependents) | 33.3% |
| Partner or kids | 20.8% |
| Full family (partner + dependents) | 6.6% |

26.7 percentage point gap between alone and full family. A person with family responsibilities is much less likely to switch providers.

### 3. Contract type drives the largest churn differences

| Contract | Churn Rate |
|----------|-----------|
| Month-to-month | 42.1% |
| One year | 5.8% |
| Two year | 1.0% |

Electronic check payment adds extra risk: 48.9% churn vs. 6.9–8.0% for automatic payment methods.

### 4. The first 12 months are the "death zone"

| Tenure Group | Churn Rate |
|-------------|-----------|
| 0–1 year | 49.4% |
| 1–2 years | 28.6% |
| 2–4 years | 17.1% |
| 4+ years | 5.3% |

Customers who survive the first year become significantly more stable. This is consistent with the fiber optic pattern: 85.5% of seniors are on fiber optic (vs. 40.7% for younger), and fiber optic has a 41.5% churn rate — the highest of any internet service.

### 5. Gender has no meaningful predictive value

Female vs. male churn rates differ by less than 0.5 pp in both segments. Gender was dropped from the feature set.

---

## Feature Engineering

Five features derived from EDA findings, each validated by churn rate comparison before inclusion.

| Feature | Type | Logic | Validation |
|---------|------|-------|------------|
| `price_per_service` | Ratio | `MonthlyCharges / (services_count + 1)` — cost per add-on service; high value signals dissatisfaction | Permutation importance: 0.103 — top feature |
| `no_help_fiber` | Binary flag | `1` if Fiber optic AND no TechSupport — expensive internet without support is a churn risk | Churn: 48.9% (flag=1) vs. 8.5% (flag=0) |
| `senior_churn_trigger` | Binary flag | `1` if Senior AND Electronic check AND Month-to-month — three risk factors combined | Churn: 67.0% (flag=1) vs. 19.1% (flag=0) |
| `is_auto_pay` | Binary flag | `1` if payment method contains "automatic" — auto-pay customers churn far less | Churn: 7.3% (flag=1) vs. 34.0% (flag=0) |
| `family_level` | Ordinal | 0 / 1 / 2 based on Partner + Dependents | Permutation importance: 0.007 |

`TotalCharges` dropped — near-perfectly correlated with `tenure × MonthlyCharges` (r = 0.99), adds no new information.

---

## ML Pipeline

### Models

**LightGBM · XGBoost · CatBoost** — three gradient boosting frameworks, each tuned with Optuna.

### Ensemble approach

- Multi-seed ensembling: 3 seeds × 3 models (9 base models)
- LGBM mega-ensemble: 5 seeds × 50 folds for variance reduction (OOF ~0.91525)
- GNN (GraphSAGE): KNN-based graph with 25 features, 9-model ensemble — OOF 0.91687, but final blend showed diminishing returns
- Public predictions blend: averaged top public submissions (top-2 blend: 0.91732)

### Submission history

| Submission | What Changed | Public Score |
|-----------|-------------|-------------|
| Baseline (LGBM + CatBoost) | First submission | 0.91060 |
| + XGBoost | Third model added | 0.91074 |
| + Optuna tuning | Full optimization across all models | 0.91491 |
| **+ Feature engineering** | **ORIG_proba, N-grams, Distribution features** | **0.91445 → biggest jump** |
| + Multi-seed ensemble | 3 seeds × 3 models | 0.91458 |
| + LGBM mega | 5 seeds × 50 folds | ~0.91525 OOF |
| + GNN ensemble | GraphSAGE, 9 models | 0.91594 blend |
| + Public blend | Top public predictions averaged | 0.91732 |
| 🏁 **Final (private LB)** | **100% test data** | **0.91822** |

The biggest single improvement came from feature engineering — adding features derived from the original IBM Telco dataset (churn probabilities per category, n-gram categorical combinations, distribution distance features). This one step added ~0.004 to the score.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| pandas, numpy | Data processing |
| matplotlib, seaborn | EDA visualization |
| scikit-learn | Permutation importance, preprocessing |
| LightGBM, XGBoost, CatBoost | Primary models |
| Optuna | Hyperparameter optimization |
| PyTorch Geometric | GraphSAGE GNN implementation |
