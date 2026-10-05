# Kaggle Playground S6E10 — Airline Passenger Satisfaction

Binary classification (satisfied / not), metric **ROC-AUC**. Tabular data, ~700k train rows plus the original survey dataset as an extra source.

**Status (2026-10-05):** stack of 10 models, nested CV **0.96178**, public LB **0.96132** (124th of 734; top-1 0.96176). Deadline: 31 Oct 2026.

## Folder structure

```
.
├── README.md
├── 00_eda.ipynb                     # EDA
├── 00_baseline.ipynb                # untuned LightGBM reference
├── 00_features_experiments.ipynb    # feature experiments (E1-E15)
├── 01_final_features.ipynb          # frozen feature set X_final
├── modelling_notebooks/             # 02-09: models, features for NNs, stack
├── data/
│   ├── raw/          # train.csv, test.csv, sample_submission.csv, orig_data.csv (original survey)
│   ├── external/     # s6e10-tabpfn-member/  (public TabPFN predictions, goodpjw2008)
│   └── processed/    # X_final_*, X_rich_*, y_train (parquet/csv)
├── preds/            # OOF + test predictions per model (oof_*.npy, test_*.npy), fold checkpoints, stack submissions
├── submissions/      # files actually sent to Kaggle
└── archive/          # old drafts and backups (safe to delete)
```

Each notebook is run from **its own folder** (Jupyter default): the root notebooks (`00_*`, `01`) read `data/...`, the ones in `modelling_notebooks/` read `../data`, `../preds`. Both layouts are resolved automatically; on Kaggle the paths fall back to `/kaggle/input` and `/kaggle/working`.

## Pipeline

### Analysis and features (root folder)

| # | Notebook | What it does | CV AUC |
|---|----------|--------------|--------|
| 00 | `00_eda` | Explores data quality, distributions and the link of every feature to the target; turns findings into modelling decisions | – |
| 00 | `00_baseline` | Untuned LightGBM on the 21 raw columns: the reference score every later idea must beat | 0.95882 |
| 00 | `00_features_experiments` | Tests ~15 feature ideas one at a time against the baseline; keeps frequency and target-encoding features | 0.96015 |
| 01 | `01_final_features` | Builds and saves the frozen 45-column feature set `X_final` (frequencies, route profile, original-data model signal) | 0.96072 (tuned LightGBM) |

### Modelling (`modelling_notebooks/`)

| # | Notebook | What it does | CV AUC |
|---|----------|--------------|--------|
| 02 | `02_lgbm_optuna` | Tunes LightGBM with Optuna on `X_final`; saves its predictions for the stack | 0.96088 |
| 03 | `03_cat_xgb` | CatBoost (two variants) and XGBoost on `X_final` | 0.96064 / 0.96062 / 0.96066 |
| 04 | `04_mlp` | Own PyTorch neural network, built and tuned step by step | 0.96043 |
| 05 | `05_stack` | Combines all models' out-of-fold predictions with logistic regression and evaluates the result with nested CV (**run last**) | 0.96178 |
| 06 | `06_realmlp` | RealMLP (pytabkit) with the community recipe on `X_final`, 3 seeds | 0.96082 |
| 07 | `07_tabpfn_import` | Imports public TabPFN predictions after verifying they match our folds | 0.96158 / 0.96089 |
| 08 | `08_features_rich` | Builds `X_rich` (92 columns): digit features, delays, GPT-2 token keys, original-survey rates, second original-data model | – |
| 09 | `09_realmlp_rich` | RealMLP on `X_rich` with in-fold target encoding, 3 seeds | 0.96129 |

Numbering is chronological (the order in which the work was done), not execution order: the stack (`05`) is rerun after every new model. **Run order from scratch:** 00_eda → 00_baseline → 00_features_experiments → 01 → 02 → 03 → 04 → 06 → 07 → 08 → 09 → **05 last**. New notebooks continue from 10.

## How the notebooks are written

Every notebook follows the same layout, so any of them can be read on its own:

1. **Header table:** pipeline position, input, output, runtime, headline result, and a glossary of the terms used.
2. **Sections with a "why":** each step says what is done and the reason for the decision, not only the code.
3. **Comments in code** at non-obvious places (leakage guards, fold logic, scale conventions).
4. **"Results & decisions" / "Summary" at the end:** the key numbers, what was accepted or rejected, and what goes to the next notebook.

Long notebooks (`00_features_experiments`, `01_final_features`, `02_lgbm_optuna`) are stored without cell outputs; their key numbers are written in the markdown conclusions.

## Results

| Model | CV | LB |
|-------|----|----|
| LGBM tuned | 0.96088 | |
| CatBoost / XGBoost | 0.96064 / 0.96066 | |
| own MLP | 0.96043 | |
| RealMLP (3 seeds) | 0.96082 | |
| RealMLP-rich (3 seeds) | 0.96129 | |
| TabPFN members | 0.96158 | |
| Stack, 6 models | | 0.96118 |
| Stack, 8 models | | 0.96124 |
| **Stack, 10 models** | **0.96178 (nested)** | **0.96132** |

Final two Kaggle submissions are chosen by nested CV, not by the public LB (noise ±0.0003–0.0005).

## Conventions

- 5-fold `StratifiedKFold(shuffle=True, random_state=42)` everywhere, so OOF predictions are stackable.
- A new idea is accepted only if the stack's nested CV gains ≥ +0.00003 and is positive on ≥4 of 5 folds (test on fold 0 first).
- Long jobs checkpoint per fold into `preds/*_folds*/`; rerunning a notebook resumes from the checkpoints. Delete the folder to retrain.
- `preds/` is deliberately flat: notebooks write and resume from these names.

## Environment

conda `kaggle-playground` (Python 3.12): lightgbm, xgboost, catboost, optuna, pytabkit, tiktoken (for 08), scikit-learn. Hardware: MacBook Air M4 (CPU), home PC RTX 3070 Ti, Kaggle GPU.

## Git hygiene

Suggested `.gitignore`: `data/`, `preds/`, `submissions/`, `*.csv`, `*.parquet`, `*.npy`, `*.pkl`, `catboost_info/`, `lightning_logs/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`.

## Credits

goodpjw2008 (TabPFN member dataset), Flight Log V3 notebook, Demidov (RealMLP recipe).
