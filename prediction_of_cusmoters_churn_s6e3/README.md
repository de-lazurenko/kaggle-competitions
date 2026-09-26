# Predict Customer Churn — Kaggle Playground S6E3
 
> My first Kaggle Playground competition. A hands-on learning project focused on EDA, feature engineering, and understanding the end-to-end ML pipeline.
 
![Score](https://img.shields.io/badge/Kaggle%20Score-0.91822-blue)
![Top](https://img.shields.io/badge/Leaderboard-Top%204%25-brightgreen )
 
---
 
## Overview
 
**Competition:** [Kaggle Playground Series S6E3 — Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3/overview)
 
The task is to predict which telecom customers are likely to cancel their subscription, based on demographic data, service usage, and billing information.
 
This was my first Kaggle Playground competition — and I came in with three specific goals in mind:
 
- Practice exploratory data analysis end-to-end on a real dataset
- Get first-hand experience with the hackathon format under time pressure
- Build a general understanding of the ML pipeline: structure, tooling, and architecture
 
One honest note on timing: the competition started March 1st, but I joined five days before the deadline and had to work fast. That turned out to be less of a problem than it sounds — my background in marketing strategy involves working under pressure with incomplete information, and that mindset transferred directly here. The main constraint was the 10-submission daily limit, which compressed experimentation, but I made the most of what was available.
 
---
 
## Project Structure
 
```
telco-churn-kaggle-s6e3/
├── README.md
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis (Denis)
│   └── 02_ML_Modeling.ipynb  # ML Pipeline (coming after deadline)
└── .gitignore
```
 
**Data:** Download from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e3/data) and place in `data/raw/`.
 
---
 
## My Role & Approach
 
### EDA — Denis (independent work)
 
The EDA notebook is fully my own work. I approached it the way I approach strategy problems: start with the business question, form hypotheses, then look for evidence in the data.
 
My process:
1. Defined the key question — *who churns and why?*
2. Segmented the customer base by age, family status, contract type, and service usage
3. Formed and tested hypotheses for each segment
4. Translated confirmed findings into engineered features for the model
 
Feature engineering was the hardest part — it was my first time doing it formally. I relied on Claude AI to help me understand the process and think through edge cases, but the hypotheses and business logic behind each feature were mine.
 
### ML Pipeline — built with Claude AI
 
I'll be honest: the modeling part was done with significant AI assistance. I worked through it step by step to understand what was happening and why, but I wouldn't claim this as independent ML work.
 
What I focused on learning:
- How to structure a modeling pipeline from train/test split to final submission
- Why ensemble methods outperform single models
- How hyperparameter tuning actually works in practice
- What the leaderboard score means and how public vs. private test sets differ
 
My goal for the next competition is to be comfortable enough with these processes to work with a human partner — rather than relying on AI to drive the modeling decisions.
 
**Stack:** XGBoost · LightGBM · CatBoost · Optuna · Multi-seed ensembling · Kaggle submission pipeline
 
*Full breakdown in* `02_ML_Modeling.ipynb` *(coming soon)*
 
---
 
## Key Findings from EDA
 
Four findings that shaped the entire analysis:
 
**1. Senior citizens churn at 2.6x the rate of younger customers**
50% churn rate vs. 19% — despite representing only 11% of the customer base and contributing 13% of revenue. The demographic gap is driven not by age alone, but by the combination of expensive services and lack of support.
 
**2. Family status is the strongest retention signal**
Customers with both a partner and dependents churn at roughly 5x lower rates than customers who are alone. A single engineered feature — `family_level` (0, 1, or 2) — captured this cleanly and became one of the more useful inputs for the model.
 
**3. Month-to-month contracts are the primary churn driver**
61% churn rate on monthly contracts vs. under 5% on two-year contracts. This isn't surprising — but the magnitude of the gap is. Combined with Electronic Check payment, the risk compounds further.
 
**4. The first 12 months are critical**
Churn is heavily front-loaded. Customers who survive the first year become significantly more stable. This shaped the `tenure_group` feature and confirmed that new customer retention should be a top business priority.
 
---
 
## Results
 
| Submission | What Changed | Public Score |
|---|---|---|
| Baseline (LGBM + CatBoost) | First submission | 0.91060 |
| + XGBoost | Added third model | 0.91074 |
| + Tuned XGBoost | Optuna optimization | 0.91089 |
| + GPU training | Moved to local RTX 3070 Ti | 0.91097 |
| + Feature engineering | ORIG_proba, N-grams, Distribution features (173 total) | 0.91445 |
| + Multi-seed ensemble | 3 seeds × 3 models | 0.91458 |
| + Optuna all models | Full optimization across LGBM/XGB/CatBoost | 0.91491 |
| + LGBM mega | 5 seeds × 50 folds | ~0.91525 OOF |
| + GNN v1 | GraphSAGE, 25 features, KNN graph | 0.91368 |
| + GNN ensemble | 9 models (base/wide/knn16 × 3 seeds × 10 folds), OOF 0.91687 | — |
| + LGBM + GNN blend | LGBM 60% + GNN ensemble 40% | 0.91594 |
| + Public predictions blend | H-blend from nina2025/ps-s6e3-29 (top 5 public notebooks) | 0.91718 |
| + Top-2 blend | 0.91732 + 0.91731 averaged | 0.91732 |
| + 0.91733.rr.csv | Nina's rounded+amplified prediction | 0.91733 |
| 🏁 **Final result** | **Private leaderboard (100% test data)** | **0.91822** |
 
The biggest single jump came from feature engineering — adding features derived from the original IBM Telco dataset (churn probabilities per category, distribution distance features, and n-gram categorical combinations). This alone added ~0.004 to the score.
 
---
 
## What I Learned
 
**Feature engineering is harder than it looks.**
It's not just about creating new columns — it's about translating business understanding into mathematical signals the model can use. This was my first time doing it formally, and I made mistakes. Having a marketing background helped with the "what matters to customers" part, but the "how to encode that as a feature" part required a lot of iteration.
 
**Good visualization takes deliberate thought.**
I moved away from plotting raw data toward a pattern I found more natural: calculate first, print the numbers, then confirm visually. It made the analysis cleaner and forced me to actually understand what I was looking at before drawing conclusions.
 
**There's always more to try.**
Several techniques I learned about during this project ended up in the Appendix rather than the notebook — SHAP values, survival analysis, OOF stacking. Not because they weren't interesting, but because time ran out. They're on the list for next time.
 
---
 
## Closing Thoughts
 
This was an unusual experience for me — working at the intersection of data analysis and machine learning under real competition conditions. I learned a lot, and not just about the technical side. The format itself was valuable: a fixed deadline, a public leaderboard, and the constant question of whether the next experiment is worth the submission slot.
 
I'm planning to keep participating in Kaggle Playground competitions. The next step is finding a partner for these projects — someone to own the ML side while I focus on analysis, and to learn from along the way. If that sounds like something you'd be interested in, feel free to reach out.

Final private leaderboard score: **0.91822** — top 4% overall.
 
**LinkedIn:** [linkedin.com/in/de-lazurenko](https://linkedin.com/in/de-lazurenko)  

---
 
*Kaggle notebook and final submission details will be added after the competition closes on March 31, 2026.*
