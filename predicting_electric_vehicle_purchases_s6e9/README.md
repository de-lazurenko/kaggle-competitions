# Predicting Electric Vehicle Purchases — Kaggle Playground Series S6E9

**Task:** Binary classification — predict whether a person will buy an electric vehicle.  
**Metric:** ROC-AUC  
**Competition deadline:** September 30, 2026  
**Current best public score:** 0.94592

I performed the data analysis and the entire feature engineering logic independently. The technical implementation of the features and the configuration of the training model were developed in collaboration with AI. The main goal of the project is training in data analysis and data science.

---

## Project Structure

```
predicting_electric_vehicle_purchases_s6e9/
├── data/
│   ├── raw/                  # Original competition files
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/            # Engineered features, saved by features.ipynb
│       ├── X_train.csv
│       ├── X_test.csv
│       └── y_train.csv
├── eda.ipynb                 # Exploratory data analysis
├── features.ipynb            # Feature engineering pipeline
└── baseline.ipynb            # Model training and submission
```

---

## Dataset

The dataset is synthetic but generated from a real-world EV survey (10,000 original rows,  
scaled to ~280K rows for the competition). Each row represents one person.

**Key columns:**

| Column | Type | Description |
|---|---|---|
| `annual_income_usd` | numeric | Annual income in USD |
| `daily_commute_km` | numeric | Daily commute distance |
| `environmental_concern_level` | numeric (1–10) | Self-reported concern about environment |
| `range_anxiety_level` | categorical | Low / Medium / High |
| `subsidy_available` | binary | Whether EV subsidy is available |
| `home_charging_possible` | binary | Whether home charging is possible |
| `charging_stations_near_home/work` | numeric | Number of nearby charging stations |
| `city_type` | categorical | Rural / Suburban / Urban |
| `will_buy_ev` | target | Yes / No |

---

## EDA Key Findings

The original dataset was generated using a deterministic scoring formula.  
Reverse-engineering that formula was the central insight of the EDA.

### Strong predictors

| Feature | Purchase rate (low → high) | Role |
|---|---|---|
| `environmental_concern_level` | 0.6% → 51.8% | Direct driver |
| `subsidy_available` | 0.6% → 27.5% | Direct driver |
| `range_anxiety_level` | 0.1% (High) → 18.9% (Low) | Direct driver |
| `annual_income_usd` | 6.4% → 35.1% | Direct driver |

### Weak predictors

`age`, `gender`, `number_of_cars_owned`, `city_type`, `current_car_type` — all showed  
less than 4% spread in purchase rate and have no direct connection to the target formula.  
They were kept as features but not used in manual scores.

### Key insight — hidden scoring formula

Analysis of purchase patterns revealed two underlying formulas used to generate the data:

**Buy score** — measures financial and motivational readiness to buy an EV:
```
buy_score = 1.2 × (income / 100K)
          + 0.6 × environmental_concern
          + 2.0 × (subsidy == Yes)
          - 1.0 × (anxiety == Medium)
          - 3.0 × (anxiety == High)
```

**Worry score** — measures practical friction around EV charging:
```
worry_score = daily_commute_km
            - 5 × charging_stations_near_home
            - 5 × charging_stations_near_work
            - 150 × (home_charging_possible == Yes)
```

A simple threshold classifier using `buy_score` alone reached AUC ≈ 0.91, confirming  
these formulas capture most of the decision logic.

---

## Feature Engineering

**Total: 22 features** across four groups.

### Numeric (raw) — 7 features
Used as-is: `age`, `annual_income_usd`, `daily_commute_km`, `number_of_cars_owned`,  
`charging_stations_near_home`, `charging_stations_near_work`, `environmental_concern_level`.

### Categorical encoding — 6 features
- **Ordinal:** `range_anxiety_level` (Low=0, Medium=1, High=2), `city_type` (Rural=0, Suburban=1, Urban=2)
- **Binary:** `subsidy_available`, `home_charging_possible` → 0/1
- **Label encoding:** `gender`, `current_car_type`

### Manual scores — 2 features
`buy_score` and `worry_score` from the formulas above.  
These directly encode domain knowledge about what drives EV purchase decisions.

### Interaction features — 7 features

| Feature | Logic |
|---|---|
| `income_x_subsidy` | Financial capacity when subsidy is available — amplifies subsidy effect |
| `concern_x_subsidy` | Environmentally motivated AND has subsidy — doubly motivated buyers |
| `income_per_car` | Household wealth relative to number of cars owned |
| `commute_per_charger` | Commute burden vs available charging infrastructure |
| `buy_x_worry` | High desire + high friction = conflicted buyer |
| `income_x_concern` | Income AND motivation — most likely buyers |
| `age_x_income` | Life-stage wealth proxy |

---

## Model

### Algorithm — LightGBM

LightGBM was chosen for its efficiency on tabular data, native handling of categorical  
features, and compatibility with the cross-validation setup used for target encoding.

### Hyperparameter tuning — Optuna

Bayesian search with 100 trials, 5-fold cross-validation, optimizing ROC-AUC.

**Final parameters:**

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 1069 | Found by Optuna with early stopping |
| `learning_rate` | 0.03359 | Slow learning, more trees |
| `num_leaves` | 23 | Shallow trees reduce overfitting on synthetic data |
| `min_child_samples` | 90 | Large minimum leaf size, additional regularization |
| `subsample` | 0.6799 | Row subsampling per tree |
| `colsample_bytree` | 0.9624 | Feature subsampling per tree |

---

## Nested Target Encoding

Standard target encoding computes `mean(target)` per group using the full training set —  
each row then "sees" its own target, causing data leakage and overfitting.

**Nested TE** removes this leakage by using inner cross-validation:

1. Split fold-train into 5 inner folds
2. For each inner-val row — compute TE statistics from inner-train rows only
3. For val/test rows — use full fold-train statistics (safe, since their targets are unseen)

**Bayesian smoothing** stabilizes estimates for rare groups:
```
te = (sum + m × global_mean) / (count + m)
```
- Small group → result pulled toward `global_mean`
- Large group → result ≈ observed group mean
- Two smoothing strengths: `m=10` (soft) and `m=100` (strong)

**TE keys — 6 grouping variables:**

| Key | Logic |
|---|---|
| `income_exact` | Exact income value (rounded) |
| `income_floor100` | Income bucketed to hundreds |
| `income_floor1000` | Income bucketed to thousands |
| `commute_exact` | Exact commute distance |
| `commute_integer` | Commute rounded to km |
| `switch` | `subsidy_available` × `range_anxiety_level` combined |

6 keys × 2 smoothing values = **12 TE features added per fold**

---

## Training Setup

- **Outer CV:** 5-fold StratifiedKFold
- **Multi-seed bagging:** 3 seeds (42, 0, 123) → 15 total model trainings
- **Final prediction:** Mean of all 15 test predictions

---

## Results

| Experiment | OOF AUC | Public AUC |
|---|---|---|
| Baseline (LightGBM, no TE) | ~0.940 | 0.9418 |
| + Nested Target Encoding | 0.9456 | **0.94592** |
| + Multi-seed bagging (3 seeds) | 0.94586 | 0.94588 |

Nested TE gave a clear improvement (+0.004 AUC) by providing additional signal about  
income and commute patterns without leaking target information.

Multi-seed bagging produced no meaningful gain — the signal in this dataset is  
already well-captured by a single-seed model with nested TE.

**Final leaderboard result:** to be updated after September 30, 2026.


