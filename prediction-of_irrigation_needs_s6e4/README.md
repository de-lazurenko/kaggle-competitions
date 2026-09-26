# Kaggle Playground Series S6E4 — Predicting Irrigation Need

**Competition:** [PS S6E4 — Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4)  
**Task:** Multiclass classification → `Low` / `Medium` / `High`  
**Metric:** Balanced Accuracy  
**Dataset:** 630,000 synthetic rows generated from a 10,000-row original  
**Result:** 🥇 **Top 0.7% — 29th out of 4,315 teams**

---

## Project Structure

```
prediction_of_irrigation_needs_s6e4/
├── README.md
└── notebooks/
    ├── 01_EDA_and_Feature_Engineering.ipynb   # EDA + feature engineering (Denis)
    ├── 02_XGBoost_Seed_Ensemble.ipynb         # XGBoost 3-seed ensemble (ML specialist)
    └── 03_XGB_RealMLP_Ensemble.ipynb          # XGBoost + RealMLP stacking (ML specialist)
```

Data: download from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e4/data) and place in `data/raw/`.

---

## Role Split

**Denis** — EDA and feature engineering (notebook 01, independent work).  
**ML specialist** — XGBoost and RealMLP training pipelines, cross-validation, ensembling. Denis contributed the threshold optimization logic on the ML side.

---

## Dataset

| Column | Type | Notes |
|--------|------|-------|
| `Soil_Moisture` | Numeric | Strongest predictor (MI = 0.205) |
| `Rainfall_mm` | Numeric | Second strongest predictor (MI = 0.188); non-linear relationship with target |
| `Temperature_C`, `Wind_Speed_kmh` | Numeric | Combined drying effect |
| `Mulching_Used` | Categorical | Strongest categorical signal (Chi² = 28,569) |
| `Crop_Growth_Stage`, `Season` | Categorical | Key groupby dimensions |
| `Soil_pH`, `Organic_Carbon`, `Electrical_Conductivity` | Numeric | Suspiciously low unique counts → synthetic artifact |
| `Irrigation_Requirement` | Target | Low (58.7%) / Medium (37.9%) / High (3.3%) |

---

## EDA Key Findings

### 1. Severe class imbalance

| Class | Count | Share |
|-------|-------|-------|
| Low | 369,917 | 58.7% |
| Medium | 239,074 | 37.9% |
| **High** | **21,009** | **3.3%** |

`High` is **18× rarer** than `Low`. Since the metric is Balanced Accuracy — which weights each class equally — ignoring the minority class is directly penalized.

Solution: `class_weight='balanced'` in all models + threshold optimization after training.

### 2. Two numerical features dominate

`Soil_Moisture` (MI = 0.205) and `Rainfall_mm` (MI = 0.188) are the strongest predictors by a large margin. The `High` class concentrates where soil moisture is low (10–25%) and rainfall is minimal — crops need irrigation most when they're dry.

`Rainfall_mm` had near-zero Pearson correlation with the target (−0.11) but high Mutual Information. This confirms a non-linear relationship. Correlation alone would have underweighted this feature at the engineering stage.

### 3. `Mulching_Used` is the most powerful categorical feature

Chi² score of **28,569** — more than 35× higher than the second-ranked categorical (`Water_Source`: 791). Mulch retains soil moisture; crops without it need significantly more irrigation.

### 4. Synthetic data artifacts

Three features showed suspiciously low unique counts relative to dataset size:
- `Organic_Carbon`: 131 unique values out of 630k rows
- `Soil_pH`: exactly 341 unique values
- `Electrical_Conductivity`: exactly 341 unique values

This pattern comes from the synthetic generator adding continuous noise on top of discrete original values. **Snap Features** (mapping each value back to the nearest original discrete value) were tested but dropped — all four candidate columns showed 0% non-zero differences after snapping, so there was nothing to recover.

`Rainfall_mm` had a near-zero spike that exists only in the synthetic data, not in the original 10k rows. This artifact turned out to be useful — it strongly separates classes.

---

## Feature Engineering

All features were validated with Mutual Information before inclusion. Features with no MI improvement over the originals were dropped.

### Engineered features kept (8 total)

| Feature | Type | Logic | MI |
|---------|------|-------|----|
| `is_dry` | Binary flag | `1` if `Rainfall_mm < 500` — captures the synthetic zero-spike; when `is_dry=1`, High class rate jumps from 2.2% to 28.7% | High |
| `Temp_x_Wind` | Interaction | `Temperature_C × Wind_Speed_kmh` — both dry the soil; combined effect is stronger than each alone | — |
| `Moisture_x_Rain` | Interaction | `Soil_Moisture × Rainfall_mm` — product of the two strongest predictors | — |
| `Moisture_div_Rain` | Ratio | `Soil_Moisture / (Rainfall_mm + 1)` — relative dryness signal | 0.101 |
| `Crop_Growth_Stage_Season_mean_target` | GroupBy aggregate | Mean-encoded target per growth stage × season | ~0.073 |
| `Crop_Growth_Stage_Season_Rainfall_mm_std` | GroupBy aggregate | Rainfall variability within each growth stage × season group | 0.176 |
| `Crop_Growth_Stage_Season_Soil_Moisture_std` | GroupBy aggregate | Moisture variability within each growth stage × season group | 0.173 |
| `Mulching_Used_Season_Soil_Moisture_mean` | GroupBy aggregate | Mean soil moisture per mulching status × season — combines the strongest categorical signal with season | 0.058 |

### Dropped during validation

Several groupby features were tested and dropped (no MI improvement):
- `Crop_Type_Season`, `Region_Season`, `Soil_Type_Season` mean targets
- `Irrigation_Type_Season_Rainfall_mm_mean`
- `Temp_div_Humidity`

Snap features (`*_snap`, `*_snap_diff`) — tested on 4 columns, showed 0% non-zero differences, dropped entirely.

### Final feature set

**19 features total**: 11 original columns + 8 engineered features.

---

## ML Pipeline

*This section is based on the approach designed and implemented by the ML specialist.*

### Models

**XGBoost** — trained with 3 independent random seeds (0, 1, 2), stratified k-fold cross-validation. Out-of-fold (OOF) predictions saved for each seed.

**RealMLP** — neural network optimized for tabular data, same seed × fold structure as XGBoost.

### Ensemble

Two-level stacking:

1. **Level 1:** XGBoost × 3 seeds + RealMLP × 3 seeds → 6 sets of OOF probability arrays (shape: `n_samples × 3`)
2. **Level 2:** Logistic Regression meta-learner trained on the concatenated OOF arrays (shape: `n_samples × 18`) — learns optimal blending weights from data

### Threshold Optimization

Standard `argmax` on probabilities doesn't handle class imbalance well when the metric is Balanced Accuracy. After training, per-class decision thresholds were optimized using coordinate descent on OOF predictions:

- Sweep thresholds per class independently while holding others fixed
- Fine-tune around the best found values
- Compare against cost-sensitive argmax (scaling class probabilities by learned weights)
- Apply the better-performing method to test predictions

This step gave a measurable improvement over the default argmax baseline on OOF Balanced Accuracy.

---

## Results

**Final leaderboard position: 29th / 4,315 teams — top 0.7%**

Key drivers of the result:

| Factor | Impact |
|--------|--------|
| `is_dry` flag | Identified from synthetic artifact; High class rate 2.2% → 28.7% when flag is 1 |
| GroupBy std features | MI ~0.17 — captured variability patterns invisible to raw features |
| Multi-seed XGBoost | Reduced variance from individual runs |
| XGBoost + RealMLP stacking | Diversified model types |
| Threshold optimization | Tuned directly to Balanced Accuracy on OOF data |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| pandas, numpy | Data processing |
| scikit-learn | KS test, Mutual Information, Chi², preprocessing |
| XGBoost | Primary model |
| RealMLP | Neural network for tabular data |
| scipy | `cKDTree` (snap feature testing), statistical tests |
| matplotlib, seaborn | EDA visualization |
