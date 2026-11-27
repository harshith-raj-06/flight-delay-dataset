# Person 3 - T-2h boosting benchmarks

This folder holds Person 3's reproducible model entry points and completed benchmark artifacts. Every run uses the shared `preprocessing.py` contract: all 35 approved T-2h predictors, tuning on folds 1-2 with seed 11, locked evaluation on folds 3-4 with seeds 11, 23, and 42, and no December test data.

## Completed model results

The finished full-data artifacts are committed under `results/`:

- `results/lightgbm/`
- `results/xgboost/`
- `results/catboost/`
- `results/histgb/` (the documented scalable sklearn boosting substitute)

Each completed model folder contains tuning records, locked fold-3/4 evaluation scores, per-seed probability predictions, settings, and a refitted development model. `feature_importances.csv` is present when the estimator exposes native importances; HistGradientBoosting does not.

## Local setup

Keep the large development CSVs local rather than committing them. The expected layout is:

```text
flight-delay-dataset/
├── final_csv/
│   ├── train_T2h.csv
│   └── validation_T2h.csv
└── person3/
```

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt xgboost lightgbm catboost joblib
brew install libomp  # required by LightGBM on macOS
```

Verify the data and chronological folds before a new run:

```bash
python preprocessing.py
```

## Model commands

Each command requires a new, empty output directory. Existing `results/*` folders are completed artifacts and must not be reused for a rerun.

```bash
caffeinate -i python models/train_gradient_boosting.py --out results/gradient_boosting_rerun
caffeinate -i python models/train_xgboost.py --out results/xgboost_rerun
caffeinate -i python models/train_lightgbm.py --out results/lightgbm_rerun
caffeinate -i python models/train_catboost.py --out results/catboost_rerun
caffeinate -i python models/train_histgb.py --out results/histgb_rerun
```

AdaBoost is maintained with the shared team model scripts under `models/adaboost.py`, so it does not have a duplicate Person 3 entry point.

## Required artifact set

Every complete model folder contains:

- `tuning.csv` and `tuning_ranking.csv`
- `learning_curve.csv`
- `evaluation.csv`
- `predictions.parquet`
- `settings.json`
- `model.joblib`
- `feature_importances.csv` when supported by the estimator

For each seed, predictions must contain 49,882 rows for fold 3 and 58,899 rows for fold 4.
