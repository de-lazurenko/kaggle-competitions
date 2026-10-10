# Kaggle Playground Series — Solutions & Experiments

Solutions for [Kaggle Playground Series](https://www.kaggle.com/competitions?hostSegment=playground) competitions on tabular data. The main goal is practice: analysing data, engineering features, and learning to think through a dataset before touching a model. Machine learning is a second, growing focus.

## How the work is split

| Part | Who | Notes |
|---|---|---|
| EDA, hypotheses, feature ideas and their testing | **Me** | Every idea is tested one at a time against a fixed baseline; rejected ideas are documented, too |
| Validation design and decisions | **Me** | Fixed folds, out-of-fold predictions, nested CV for stacking; final submissions chosen by CV, not by the public leaderboard |
| Model training, tuning, ensembling | Me, with an AI assistant and public notebooks | Model recipes, parameters and public predictions are taken from public work where it helps; every source is credited in the competition README |

## Competitions

| Season | Competition | Target (metric) | Key techniques | Score | Result |
|--------|-------------|-----------------|----------------|-------|--------|
| S6E10 | [Airline Passenger Satisfaction](predicting_airline_satisfaction_s6e10/) | Binary (ROC-AUC) | Route-profile and original-data features, GBDT + RealMLP + TabPFN stack, nested CV | 0.96132 | in progress |
| S6E9 | [EV Purchase Prediction](predicting_electric_vehicle_purchases_s6e9/) | Binary (ROC-AUC) | Nested target encoding, LightGBM, Optuna | 0.94592 |  1,272 / 3,575 — Top 36% |
| S6E4 | [Irrigation Need Prediction](predicting_of_irrigation_needs_s6e4/) | Multiclass (Balanced Accuracy) | XGBoost + RealMLP stacking, threshold optimisation, synthetic-artifact features | 0.98092 | 29 / 4,315 — Top 0.7% |
| S6E3 | [Customer Churn Prediction](predicting_of_cusmoters_churn_s6e3/) | Binary (ROC-AUC) | Multi-seed ensemble, GNN, domain-driven feature engineering | 0.91822 | 179 / 4,142 — Top 4% |

## Inside a competition folder

Each folder has its own README: project map, pipeline, results, lessons learned, credits. Notebooks are written to be read — each step says what is done and why, and ends with the decisions taken. A good starting point is the most recent competition.

**Stack:** Python 3.12 · pandas · scikit-learn · LightGBM · XGBoost · CatBoost · PyTorch · Optuna
