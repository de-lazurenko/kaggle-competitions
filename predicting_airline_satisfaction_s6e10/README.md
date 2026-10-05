# Kaggle Playground S6E10 — Airline Passenger Satisfaction

Binary classification (satisfied / not), metric **ROC-AUC**. Tabular data, ~700k train rows plus the original survey dataset as an extra source.

**Status (2026-10-05):** stack of 10 models, nested CV **0.96178**, public LB **0.96132** (124th of 734; top-1 0.96176). Deadline: 31 Oct 2026.

## Folder structure

```
.
├── README.md
├── notebooks/        # the whole pipeline, numbered in the order the work was done
├── data/
│   ├── raw/          # train.csv, test.csv, sample_submission.csv, orig_data.csv (original survey)
│   ├── external/     # s6e10-tabpfn-member/  (public TabPFN predictions, goodpjw2008)
│   └── processed/    # X_final_*, X_rich_*, y_train (parquet/csv)
├── preds/            # OOF + test predictions per model (oof_*.npy, test_*.npy), fold checkpoints, stack submissions
├── submissions/      # files actually sent to Kaggle
└── archive/          # old drafts and a backup of the layout before reorganisation (safe to delete)
```

All notebooks are run **from `notebooks/`** (Jupyter default). Paths resolve to `../data`, `../preds` automatically; on Kaggle they fall back to `/kaggle/input` and `/kaggle/working`.

## Pipeline

| # | Notebook | Purpose | Reads | Writes | CV AUC |
|---|----------|---------|-------|--------|--------|
| 00 | `00_eda` | EDA | raw | – | – |
| 00 | `00_baseline` | LGBM baseline | raw | – | 0.95882 |
| 00 | `00_features_experiments` | feature experiments (E1–E9) | raw | X_train/X_test.csv | ~0.96015 |
| 01 | `01_final_features` | final 45-column set `X_final` (frequencies, route-profile, orig_prob/logit) | raw, orig | X_final_*.parquet | – |
| 02 | `02_lgbm_optuna` | tuned LightGBM (Kaggle, Optuna) | X_final | oof/test_lgbm_tuned_final | 0.96088 |
| 03 | `03_cat_xgb` | CatBoost v1/v2, XGBoost | X_final | oof/test_cat_v1, cat_v2, xgb | 0.96064 / 0.96062 / 0.96066 |
| 04 | `04_mlp` | own MLP | X_final | oof/test_mlp | 0.96043 |
| 06 | `06_realmlp` | RealMLP (pytabkit), 3 seeds | X_final | oof/test_realmlp[_avg] | 0.96082 |
| 07 | `07_tabpfn_import` | import of public TabPFN members | data/external | oof/test_tabpfn_avg, tabpfn_raw_avg | 0.96158 / 0.96089 |
| 08 | `08_features_rich` | `X_rich` (92 columns: digits, delays, GPT-2 token keys, original-survey rates) | raw, orig, X_final | X_rich_*.parquet | – |
| 09 | `09_realmlp_rich` | RealMLP on X_rich + in-fold target encoding, 3 seeds | X_rich | oof/test_realmlp_rich[_avg] | 0.96129 |
| 05 | `05_stack` | nested-CV stack, logistic regression (C=10) on logits | all preds | submission_stack_{N}models.csv | 0.96178 |

Numbering is chronological, not execution order. **Run order from scratch:** 00_eda → 00_baseline → 00_features_experiments → 01 → 02 → 03 → 04 → 06 → 07 → 08 → 09 → **05 last**. New notebooks continue from 10.

## How the notebooks are written

Every notebook follows the same layout, so any of them can be read on its own:

1. **Header table:** pipeline position, input, output, runtime, headline result, and a glossary of the terms used.
2. **Sections with a "why":** each step says what is done and the reason for the decision, not only the code.
3. **Comments in code** at non-obvious places (leakage guards, fold logic, scale conventions).
4. **"Results & decisions" / "Summary" at the end:** the key numbers, what was accepted or rejected, and what goes to the next notebook.

Long notebooks (`00_features_experiments`, `02_lgbm_optuna`, `01_final_features`) are stored without cell outputs; their key numbers are written in the markdown conclusions.

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
