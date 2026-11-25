# Training the two-hour departure-delay model

Predict `LATE` (departure delay ≥15 minutes) for ATL, ORD and LAX two hours before scheduled departure. Run commands from the repository root. The shared entry point is `train.py`. It reuses the Person 2 preprocessing and estimator builders and uses the same four forward folds for every team member.

## Completed reference runs

All eleven model choices passed the continuous 20,000-row smoke check. Full-size four-fold reference runs are available for the constant baseline and schedule-only HistGB in [benchmarks/schedule](benchmarks/schedule), and for 35-feature HistGB in [benchmarks/full](benchmarks/full). Both use default parameters and seeds 11, 23 and 42. The folders include results, per-fold and per-seed runtimes, and column audits. The full folder also contains five recomputed history windows and their contributing IDs. The remaining model choices have smoke results only in this run.

## 1. Set up the environment and inputs

Tested with Python 3.12.9. On Apple Silicon install the OpenMP runtime with `brew install libomp` before installing the boosting libraries. On Debian/Ubuntu the equivalent runtime is `libgomp1`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip show scikit-learn
```

Use the existing final export or reproduce it using the three build commands in [README.md](README.md). Training needs **train_T2h.csv**, **validation_T2h.csv** and **fresh_2025/output_with_weather.parquet** from the same build. The Parquet supplies actual departure times for fold eligibility and independent history checks. For other directory layouts pass `--data-dir PATH --events FILE`. Development runs and the refit command read only development CSVs and request pre-December source events. December CSVs are unnecessary for these commands.

## 2. Person 1: verify the data and run the first benchmark

```bash
python train.py --smoke --models constant histgb --stage schedule --seeds 11 --out runs/quick_schedule
python train.py --smoke --models decision_tree random_forest logreg pca_logreg adaboost gradient histgb xgboost lightgbm catboost --seeds 11 --out runs/quick_full
python train.py --models constant histgb --stage schedule --seeds 11 --out runs/schedule_reference
```

The smoke test takes the first 20,000 development rows in checkpoint order, places the split near row 16,000, keeps equal timestamps together and removes training labels still unknown at the first validation checkpoint. It checks execution. Use four-fold results for comparisons. Each command requires a fresh or empty output directory.

The trainer checks the 41-column schema, unique row IDs, two-hour timestamps, label agreement with BTS, weather age, condition-code flags and the five historical activity values for five fixed validation flights. Inspect `history_checks.csv` for the source row IDs included in each window. It prints weather-code availability, unmatched weather and actual train/validation late rates. Class balance can vary substantially across months. The holiday feature is already present and is taken directly from the export. `wx_obs_time` is the audit timestamp.

## 3. Persons 2 and 3: run comparable benchmarks

| Owner | Development command |
|---|---|
| Person 2: tree models | `python train.py --models decision_tree random_forest adaboost --out runs/trees` |
| Person 2: PCA comparison | `python train.py --models logreg pca_logreg --out runs/pca` |
| Person 3: sklearn boosting | `python train.py --models gradient histgb --out runs/sklearn_boost` |
| Person 3: external boosting | `python train.py --models xgboost lightgbm catboost --out runs/native_boost` |

The default is the full 35-feature set, all eligible training rows, four validation folds and seeds 11, 23 and 42. Use `--jobs 2` on a smaller machine. Each score records elapsed fit-plus-predict time. Use the shared entry point for comparable scores. The earlier standalone Person 2 script applies a scheduled-time buffer to later folds; the shared entry point checks actual label availability directly.

**Preprocessing by family**

- Tree, RF, AdaBoost, sklearn Gradient Boosting and logistic regression use the fold-trained Person 2 transformer. It adds four hour/month cyclic terms, groups destinations outside the fold’s top 20, one-hot encodes categoricals, median-imputes numeric columns and scales the matrix. PCA and plain logistic regression share this preprocessing and the same classifier. PCA retains 95% of training variance and logs the actual component count.
- HistGradientBoosting uses ordinal encoding learned on the training fold, up to 254 categories per field, explicitly marked categorical inputs and native numeric missing values. Unknown categories map to missing. Automatic early stopping is disabled.
- XGBoost uses pandas categories and categorical histogram splits. LightGBM receives fold-aligned pandas categories. CatBoost receives named string categoricals, a missing-category token and chronological training order. The benchmark uses fixed iteration budgets and no random validation split.
- The allowlist excludes the target, IDs, timestamps and current-flight outcomes. The full feature set includes the approved past-airport mean delay. The output column audit records exactly which predictors were used in each run.

**Tune using development only.** Defaults are starting values. Use a single model per parameter override. The accepted keys are its entries in `DEFAULTS` in `train.py`. For example, compare tree depths of 8 and 14 with the same pruning value. For a slow search, use the most recent contiguous 100,000 eligible training rows per fold. Validation still covers the entire fold.

```bash
python train.py --models decision_tree --params '{"max_depth":8,"ccp_alpha":0.0001}' --train-limit 100000 --out runs/dt8
python train.py --models decision_tree --params '{"max_depth":14,"ccp_alpha":0.0001}' --train-limit 100000 --out runs/dt14
```

Repeat the chosen configuration without the row cap before comparing it with other full-fold runs. Keep a record of attempted settings. A validation score used to select settings is a development estimate.

## 4. Person 4: compare scores, run ablations and lock the final model

The validation blocks are Sep 1–21, Sep 22–Oct 12, Oct 13–Nov 2 and Nov 3–28, 2025. A fold trains on earlier local flight dates whose checkpoints and actual departure times precede the first validation checkpoint. Earlier validation flights can join a later training fold once eligible. Chronological gaps and old ATL holdout exclusions from the export still apply.

Rank models by **average precision (AP)**. Use **ROC-AUC** as the secondary metric. The constant training-prior baseline gives validation AP equal to the late fraction and ROC-AUC of 0.5. Report observed seed variation, including zeros. Bin sampling, estimator settings and row counts can affect reproducibility. A high schedule-only score warrants checking the allowlist and split; a score threshold alone cannot establish leakage.

For the selected estimator, run the four stages with identical parameters, folds and seeds. Replace histgb below if another model is selected.

```bash
for stage in schedule route weather full; do
  python train.py --models histgb --stage "$stage" --out "runs/ablation_$stage"
done
```

Stages use 7, 10, 30 and 35 predictors. Cloud cover enters with weather. For sensitivity to historical delay, append `--drop prior_mean_delay_60min_T2h`. To exclude prior delay results also drop `late_rate_60min_T2h`. Removing all outcome-derived activity also requires dropping `n_departed_60min_T2h` and `has_history_T2h`. Schedule-only demand can stay.

**What each owner hands over**

- `results.csv` has mean and sample standard deviation across folds, after averaging seed scores within each fold. A smoke run has one fold and hence no fold standard deviation.
- `fold_scores.csv` contains fold/seed metrics, class balances, row counts, runtimes and PCA statistics where applicable.
- `pred_MODEL_FOLD_SEED.parquet` contains row IDs, dates, labels and out-of-fold probabilities for that particular estimator and seed. Use validation predictions for any threshold choice.
- `config.json`, `column_audit.csv` and `history_checks.csv` identify settings, inputs and historical event checks. Share small results with the team; local prediction artifacts can stay under `runs/`.

**After model selection only:** lock the feature set, parameters, final seed and any threshold. Example command for a single-seed final pipeline:

```bash
python train.py --refit --models histgb --stage full --seeds 11 --out runs/locked_refit
```

Pass the actual chosen parameters and drop list. Refit uses all 726,979 development flights and writes `model.joblib` plus its configuration. It produces no test scores. For the one agreed final evaluation, load the trusted model with `joblib.load` from the repository environment, select the ordered `features` from its adjacent config, and use `predict_proba(X)[:, 1]` on the 68,146 December flights. Record AP, ROC-AUC, the December 1 test start and the chosen threshold if reporting thresholded metrics. Earlier prototypes examined overlapping December flights; disclose that history in the report. Earlier December flights can contribute to a later flight’s observed history under the current specification.

Once authorized for the final evaluation, this is the scoring code to run from the repository root. Set the folder to the chosen refit before running it.

```python
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

folder = Path("runs/locked_refit")
features = json.loads((folder / "config.json").read_text())["features"]
model = joblib.load(folder / "model.joblib")
test = pd.read_csv("final_csv/test_T2h.csv", usecols=["FlightDate", "LATE", *features])
assert len(test) == 68146 and test.FlightDate.min() == "2025-12-01"
prob = model.predict_proba(test[features])[:, 1]
print("AP", average_precision_score(test.LATE, prob))
print("ROC-AUC", roc_auc_score(test.LATE, prob))
```
