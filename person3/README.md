# Person 3 - Local T-2h model benchmarks

These commands follow the team protocol: 35 approved T-2h features, tuning only on folds 1-2 with seed 11, then locked evaluation on folds 3-4 with seeds 11, 23 and 42. December is never loaded.

## One-time local setup

From the repository root, place `train_T2h.csv` and `validation_T2h.csv` in `final_csv/`, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt xgboost lightgbm catboost joblib
```

Keep the laptop awake during a run:

```bash
caffeinate -i
```

## Run one model at a time

Each command requires a new, empty output directory. Do not start more than one of these full-data benchmarks at once on the same laptop.

```bash
python person3/train_adaboost.py --out runs/adaboost
python person3/train_gradient_boosting.py --out runs/gradient_boosting
python person3/train_xgboost.py --out runs/xgboost
python person3/train_lightgbm.py --out runs/lightgbm
python person3/train_catboost.py --out runs/catboost
```

`train_histgb.py` is the documented scalable substitute for classic sklearn gradient boosting. Use it only if the team agrees that the classic gradient-boosting run is impractical, and record that substitution in the handoff:

```bash
python person3/train_histgb.py --out runs/histgb
```

## Required output in each model folder

Each successful run produces:

- `tuning.csv` - all 25 configurations on each tuning fold, including failures.
- `tuning_ranking.csv` and `learning_curve.csv` - model-selection evidence.
- `evaluation.csv` - one row per fold/seed for folds 3-4.
- `predictions.parquet` - `row_id`, fold, seed, target, and probability.
- `settings.json` - grid, selected settings, command, package versions and code commit.
- `model.joblib` - development refit using seed 11.
- `feature_importances.csv` when the estimator exposes native importances.

Before sharing a folder, verify that each seed has 49,882 predictions for fold 3 and 58,899 for fold 4.
