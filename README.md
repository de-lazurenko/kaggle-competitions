# Kaggle Playground Series — Solutions & Experiments

A personal collection of my solutions for [Kaggle Playground Series](https://www.kaggle.com/competitions?hostSegment=playground) competitions. This repository documents my learning journey in data analytics and machine learning — from EDA and feature engineering to model building and evaluation.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightgrey)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)

## Competitions

| Season | Competition | Target | Key Techniques | Score | Top % |
|--------|-------------|--------|----------------|-------|-------|
| S6E9 | [EV Purchase Prediction](predicting_electric_vehicle_purchases_s6e9/) | Binary classification (ROC-AUC) | Nested Target Encoding, LightGBM, Optuna | 0.94592 | TBD |

*Table updates with each new competition.*

## Overview

Each competition folder contains the full analytical pipeline:

- **EDA** — understanding the data, distributions, target analysis, key feature patterns
- **Feature Engineering** — manual scores, interaction features, categorical encoding
- **Modeling** — cross-validation, hyperparameter tuning, target encoding without leakage
- **Evaluation** — metric tracking, experiment comparison, submission

## Repository Structure

```
README.md
predicting_electric_vehicle_purchases_s6e9/
    eda.ipynb
    features.ipynb
    baseline.ipynb
    data/
        raw/
        processed/
    README.md
```

## Author

Denis Lazurenko | Data Analyst

- Kaggle: [kaggle.com/de-lazurenko](https://www.kaggle.com/de-lazurenko)
- LinkedIn: [linkedin.com/in/de-lazurenko](https://www.linkedin.com/in/de-lazurenko)
- GitHub: [github.com/de-lazurenko](https://github.com/de-lazurenko)
