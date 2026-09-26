# Kaggle Playground Series S6E4 — Predicting Irrigation Need
 
**Competition:** [PS S6E4 — Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4)  
**Task:** Multiclass classification → `Low` / `Medium` / `High`  
**Metric:** Balanced Accuracy  
**Dataset:** 630,000 synthetic rows generated from the [Irrigation Water Requirement Prediction Dataset](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset)  
**Result:** 🥇 **Top 0.7% — 29th out of 4,315 teams**
 
---
 
## Overview
 
The goal was to predict agricultural irrigation need (Low / Medium / High) based on soil, climate, and crop features. The challenge had two technical complications that shaped the entire approach:
 
1. **Severe class imbalance** — `High` class represented only 3.3% of training data (18× rarer than `Low`), and the metric was Balanced Accuracy, which penalizes ignoring minority classes directly
2. **Synthetic data artifacts** — the dataset was generated from a small 10,000-row original, introducing discrete value patterns and zero-spike artifacts that could be exploited via feature engineering
This was a collaborative project. I was responsible for EDA and feature engineering; the ML modeling was done together with an ML specialist.
 
---
 
## Project Structure
 
```
kaggle-playground-irrigation-prediction/
├── README.md
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb   # EDA + features (Denis)
│   ├── 02_XGBoost_Seed_Ensemble.ipynb         # XGBoost 3-seed ensemble
│   └── 03_XGB_RealMLP_Ensemble.ipynb          # Final XGB + RealMLP stacking
└── .gitignore
```
 
Data: Download from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e4/data) and place in `data/raw/`.
 
---
 
## My Role & Approach
 
### EDA & Feature Engineering — Denis (independent work)
 
The EDA notebook is fully my own work. My process: start with the competition metric, understand what it penalizes, then look for structure in the data that the model can exploit.
 
Specific steps:
 
- Structural audit: shape, dtypes, missing values, unique counts per column — found three features with suspiciously low cardinality (`Organic_Carbon`: 131 unique, `Soil_pH` and `Electrical_Conductivity`: both exactly 341 unique out of 630k rows)
- Distribution analysis: histograms, KDE by class, categorical target rates — identified the strongest signal features before touching any model
- Original vs. synthetic comparison: loaded the source dataset (10k rows) and compared distributions to detect generator artifacts
- Train/test distribution shift check via Kolmogorov-Smirnov test — confirmed no shift (expected for Playground format)
- Statistical feature importance: Mutual Information for numerical features, Chi-squared for categorical
- Feature engineering: designed and validated 8 new features based on EDA findings (see below)
### ML Pipeline — collaboration with ML specialist
 
The modeling was done jointly. My ML partner built the XGBoost and RealMLP training pipelines with cross-validation and multi-seed ensembling. My contribution on the ML side was threshold optimization logic — a coordinate descent algorithm to find per-class decision thresholds that directly maximize Balanced Accuracy on OOF predictions.
 
**Stack:** XGBoost · RealMLP · Logistic Regression (meta-learner) · Scikit-learn · NumPy
 
---
 
## Key Findings from EDA
 
### 1. Severe class imbalance is the core challenge
 
| Class | Count | Share |
|-------|-------|-------|
| Low | 369,917 | 58.7% |
| Medium | 239,074 | 37.9% |
| **High** | **21,009** | **3.3%** |
 
`High` is **18× rarer** than `Low`. A model trained without correction ignores it almost entirely. Since Balanced Accuracy weights each class equally, this is catastrophic.  
→ Solution: `class_weight='balanced'` in all models + threshold optimization post-training.
 
### 2. Two numerical features dominate
 
`Soil_Moisture` (MI = 0.205) and `Rainfall_mm` (MI = 0.188) are the strongest predictors by a wide margin. The `High` class concentrates where soil moisture is low (10–25%) and rainfall is minimal — intuitively, crops need irrigation most when they're dry.
 
Critically, `Rainfall_mm` had near-zero Pearson correlation with the target (−0.11), but high Mutual Information — confirming a non-linear relationship that correlation alone would miss.
 
### 3. `Mulching_Used` is the most powerful categorical feature
 
Chi² score of **28,569** — more than 35× higher than the second-ranked categorical (`Water_Source`: 791). Mulch retains soil moisture; crops without it require significantly more irrigation.
 
### 4. Synthetic data artifacts are exploitable
 
- `Rainfall_mm` — spike near zero exists **only in synthetic data**, not in the original → binary flag candidate
- `Organic_Carbon`, `Soil_pH`, `Electrical_Conductivity`, `Sunlight_Hours` — all have suspiciously low unique counts → the original dataset used discrete/rounded values, and the synthetic generator added continuous noise on top → **Snap Features** applicable (map each value back to the nearest original discrete value)
---
 
## Feature Engineering
 
All features were validated with Mutual Information before inclusion.
 
| Feature | Type | Logic |
|---------|------|-------|
| `is_dry` | Binary flag | `1` if `Rainfall_mm < 500` — captures the synthetic zero-spike artifact |
| `Temp_x_Wind` | Interaction | `Temperature_C × Wind_Speed_kmh` — both dry the soil; combined effect is stronger |
| `Moisture_x_Rain` | Interaction | `Soil_Moisture × Rainfall_mm` — product of the two strongest predictors |
| `Moisture_div_Rain` | Ratio | `Soil_Moisture / (Rainfall_mm + 1)` — captures relative dryness signal (MI = 0.101) |
| `Crop_Growth_Stage_Season_mean_target` | GroupBy aggregate | Mean encoded target per growth stage × season combination |
| `Crop_Growth_Stage_Season_Rainfall_mm_std` | GroupBy aggregate | Rainfall variability within each growth stage × season group |
| `Crop_Growth_Stage_Season_Soil_Moisture_std` | GroupBy aggregate | Moisture variability within each growth stage × season group |
| `Mulching_Used_Season_Soil_Moisture_mean` | GroupBy aggregate | Mean soil moisture per mulching status × season — combines the top categorical signal with season |
 
Snap Features (`*_snap`, `*_snap_diff`) were tested but dropped after validation showed no MI improvement over the originals.
 
**Final feature set: 19 features** (11 original + 8 engineered).
 
---
 
## ML Pipeline
 
### Models
 
**XGBoost** — trained with 3 independent random seeds (0, 1, 2), stratified k-fold cross-validation, OOF predictions saved for each seed.
 
**RealMLP** — neural network baseline optimized for tabular data, same seed × fold structure as XGBoost.
 
### Ensemble
 
Final predictions come from a two-level stack:
 
1. **Level 1:** XGBoost × 3 seeds + RealMLP × 3 seeds → 6 sets of OOF probability arrays (shape: `n_samples × 3`)
2. **Level 2:** Logistic Regression meta-learner trained on the concatenated OOF arrays (shape: `n_samples × 18`) — learns optimal blending weights from data
### Threshold Optimization
 
Standard `argmax` on probabilities doesn't handle class imbalance well for Balanced Accuracy. Post-training, we optimized per-class decision thresholds using coordinate descent on OOF predictions:
 
- Sweep thresholds per class independently while holding others fixed
- Fine-tune around the best found values
- Compare against cost-sensitive argmax (scaling class probabilities by learned weights)
- Apply the better-performing method to test predictions
This step gave a measurable improvement over the default argmax baseline on OOF Balanced Accuracy.
 
---
 
## Results
 
**Final leaderboard position: 29th / 4,315 teams — top 0.7%**
 
The key drivers of the result:
- Feature engineering that captured both domain logic (mulching × moisture interactions) and synthetic data artifacts (zero-spike flag)
- Multi-seed ensembling to reduce variance from individual XGBoost runs
- Stacking XGBoost and RealMLP to diversify model types
- Threshold optimization tuned directly to Balanced Accuracy on OOF data
---
 
## What I Learned
 
**Mutual Information over correlation for imbalanced targets.** `Rainfall_mm` had Pearson correlation of −0.11 with the target but MI of 0.188 — the second-strongest predictor. Relying on correlation alone would have underweighted it at the feature engineering stage.
 
**Artifacts in synthetic data are features, not noise.** The zero-spike in `Rainfall_mm` was introduced by the synthetic generator and isn't in the original dataset — but it's still a real pattern that separates classes. The `is_dry` flag encoding it was one of the stronger engineered features.
 
**Threshold optimization is a concrete lever for imbalanced classification.** When the metric is Balanced Accuracy, the decision boundary matters as much as the model itself. Coordinate descent on OOF predictions let us tune thresholds in a rigorous, data-driven way without touching the model training.
 
**Division of roles works in practice.** Having one person own EDA and feature logic while another owns the training pipeline meant both got done well. The boundary — features go in, probabilities come out — is clean enough that the collaboration required very little coordination overhead.
 
---
 
## Closing Thoughts
 
The competition was a good test of whether EDA work translates into ML performance. The features I built from the class imbalance analysis and synthetic artifact investigation ended up in the final pipeline and contributed to the result. That was the clearest validation I've had so far that the analysis side of the work is meaningful beyond just understanding the data.
 
The threshold optimization was also new ground for me — building an algorithm to search the decision boundary rather than accepting the default argmax. It's a technique I'll carry forward to any classification problem where class distribution is uneven.
 
---
 
**LinkedIn:** [linkedin.com/in/de-lazurenko](https://linkedin.com/in/de-lazurenko)  
