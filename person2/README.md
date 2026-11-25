# Person 2 — Baselines, Trees, and PCA (`T-2h`)

This folder contains the standalone, Kaggle/Colab-ready Person 2 modelling pipeline for the **Flight Departure Delay (T-2h)** project.

## Files
- `person2_pipeline.py`: Full 4-fold expanding chronological pipeline implementing:
  - Constant-score baseline (`DummyClassifier(strategy="prior")`)
  - Decision Tree (tuned `max_depth`, `min_samples_leaf`, `ccp_alpha`)
  - Random Forest (tuned `max_depth`, `min_samples_leaf`)
  - AdaBoost (tuned `n_estimators`, `learning_rate`, base tree `max_depth`)
  - PCA comparison (`LogisticRegression` without PCA vs. with PCA, recording retained components and explained variance)
- `person2_kaggle_colab.ipynb`: Ready-to-run notebook for Google Colab or Kaggle.
- `test_smoke.py`: Fast synthetic smoke test verifying folds, preprocessing, metrics, and Person 4 handoff outputs.

## Run Locally
```bash
python person2/person2_pipeline.py --data-dir final_csv --out-dir person2_outputs
```

## Outputs for Person 4 (`person2_outputs/`)
- `person2_summary_scores.csv`: `mean ± std` of AP (PR-AUC) and ROC-AUC across the 4 folds (with seeds `(11, 23, 42)` averaged per fold).
- `person2_fold_scores.csv`: Per-fold AP and ROC-AUC for every model.
- `person2_pca_stats.csv`: Retained PCA components and explained variance ratio sum per fold.
- `person2_oof_predictions.parquet`: Out-of-fold probability predictions keyed by `row_id`, `FlightDate`, `fold`.
- `person2_chosen_settings.json`: Selected hyperparameters, seeds, and exact reproducible command.
