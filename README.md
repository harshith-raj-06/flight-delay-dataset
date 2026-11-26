# Flight departure delay at ATL, ORD and LAX

Predict whether a flight leaves the gate at least 15 minutes late, using only information available two hours before scheduled departure (checkpoint T−2h). Data: 2025 BTS on-time performance plus IEM airport weather.

## 1. Setup

Use Python 3.12 or later in a virtual environment, from the repository root.

```
python -m pip install -r requirements.txt
```

**Build the data (fresh clone).** Interrupted downloads resume if you rerun the commands.

```
python pipeline/bts.py
python pipeline/iem.py
python build_features.py
```

This writes the modelling tables to `final_csv/`: `train_T2h.csv`, `validation_T2h.csv`, `test_T2h.csv` and the T6h versions. Raw files, intermediate tables and CSV exports are local generated artifacts and are not committed. Source URLs and request records are saved in `fresh_2025/*log.txt`.

**Check that it works:** `python preprocessing.py` should print the four folds with the dates in section 3.

**Do not open or load the test file during development.**

`bench.py` and its results are deprecated (older five-monthly-fold protocol starting in July). Do not use them or mix their numbers into any table.

## 2. Task and inputs

Label LATE = departure delay ≥ 15 minutes (equals BTS DepDel15). Cancelled flights and flights missing a departure label or delay are removed. Diversions with a valid departure outcome are kept.

All models use the same 35 approved predictors, each known or observed by T−2h:

| Group | Inputs | Count |
|---|---|---|
| A. Schedule and calendar | month, dow, dep_hour, arr_hour, CRSElapsedTime, Distance, is_holiday | 7 |
| B. Route and carrier | Origin, Dest, Reporting_Airline | 3 |
| C. Weather at T (latest IEM report at or before T, age ≤ 180 min) | wx_* fields, missing flags | 20 |
| D. Airport activity | departures in the last 60 min, their mean delay and late rate, has_history, scheduled departures in the next 60 min | 5 |

Only the encoding differs between models (section 4). The column definitions are in `COLUMN_AUDIT.md`.

**Airport activity (group D)** is a rolling operational observation. At checkpoint T, only other same-origin flights that actually departed in (T−60 min, T] supply a delay or late indicator. This also applies in December: a prediction may use earlier December departures, as an airport observer could. Never compute a December-wide delay statistic, fit on December labels or tune during December. Empty history means count 0 and mean/share missing. Scheduled demand counts all flights scheduled in (T, T+60 min], including cancellations.

## 3. Data splits and protocol (do not change)

Splits by origin-local scheduled flight date, 2025:

| Export | Dates | Purpose |
|---|---|---|
| train_T2h.csv, train_T6h.csv | Jan 1 – Aug 30 | Model fitting |
| validation_T2h.csv, validation_T6h.csv | Sep 1 – Nov 28 | Tuning, ablations, threshold choice |
| test_T2h.csv, test_T6h.csv | Dec 1 – Dec 31 | Final evaluation |

August 31 and November 29–30 are gap days. The November 25, 2026 build downloaded 831,933 scheduled 2025 flights and 31,994 IEM reports (including weather warm-up). Each horizon exports 519,275 training, 207,704 validation and 68,146 test flights (795,125 unique). The exact test start is December 1, 2025.

Late-maturing training or validation labels are excluded if their actual departure reaches the first checkpoint of the next evaluation period. The January 27–31 and May 13–29 ATL prototype holdouts are excluded from training and historical outcomes; where a history window overlaps a protected date, the four observed-history fields are missing. December 2024 rows warm up history only.

Development uses four expanding chronological folds, with one full empty local calendar day between training and validation:

| Fold | Validation dates | Role |
|---|---|---|
| 1 | 2025-09-01 to 2025-09-21 | tuning |
| 2 | 2025-09-22 to 2025-10-12 | tuning |
| 3 | 2025-10-13 to 2025-11-02 | reported |
| 4 | 2025-11-03 to 2025-11-28 | reported |

- Tune hyperparameters on folds 1–2 only, selecting by average precision. Never tune on folds 3–4.
- Evaluate the locked settings on folds 3–4. For stochastic models, repeat with seeds 11, 23 and 42.
- Report ROC-AUC and PR-AUC (average precision) as mean and standard deviation. Also report lift (PR-AUC ÷ late rate), because the late rate differs between periods.
- Include a majority-class baseline under the same folds.
- Time zones: origin local time with daylight saving, converted to UTC. BTS 2400 means midnight at the end of the local date. For the ambiguous fall-back hour the first occurrence is used; a nonexistent local time moves forward to the next valid time.
- Freeze the model, features, encoder, destination grouping, imputer, scaler, PCA and decision threshold before final evaluation. Fit all transformations inside each training fold. Optionally refit on all development flights once choices are fixed.

## 4. Adding your model

`preprocessing.py` is shared by everyone. It loads the development data, builds the chronological folds and selects the right encoding from your model key, fitting it inside each training fold. You supply only your estimator and your hyperparameter grid. Do not write your own folds, metrics or preprocessing.

`decision_tree.py` is one worked example of a model script. Copy its structure for your model.

| Model key | Preprocessing supplied by the pipeline |
|---|---|
| decision_tree, random_forest, adaboost, gradient_boosting | One-hot categories, top 20 destinations plus Other, training-median numeric imputation |
| logistic_regression | Same one-hot encoding, with numeric scaling fitted on training rows |
| pca_logistic_regression | Same scaled input, then PCA retaining 95% of training variance. Supply the same logistic classifier for both variants. |
| histgb | Ordinal categories marked categorical, numeric missing values preserved, automatic early stopping disabled |
| xgboost, lightgbm | Named pandas categorical fields with training-fold category levels. Unseen categories become missing. XGBoost categorical support is enabled. |
| catboost | Named string categories, explicit missing-category token, numeric missing values preserved |

Install the library for your estimator. Parameter names use the `model__` prefix so scikit-learn tunes the estimator inside the pipeline.

```python
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from preprocessing import FEATURES, TARGET, chronological_cv, load_development, model_pipeline

dev = load_development()
X, y = dev[FEATURES], dev[TARGET]
folds = chronological_cv(dev)
pipe = model_pipeline("decision_tree", DecisionTreeClassifier(random_state=11))
grid = {"model__max_depth": [6, 12], "model__min_samples_leaf": [20, 100]}

search = GridSearchCV(pipe, grid, scoring="average_precision", cv=folds[:2], refit=False)
search.fit(X, y)
pipe.set_params(**search.best_params_)
scores = cross_validate(pipe, X, y, cv=folds[2:],
                        scoring=["average_precision", "roc_auc"])
print(search.best_params_, scores["test_average_precision"], scores["test_roc_auc"])
```

Use the same tuning and evaluation blocks across teammates. Here `test_*` in sklearn's output means the later development validation folds, not the December test file. Use a fresh pipeline per run, and record parameters, seeds, fold scores and runtime. If you subsample training rows for speed (for example scikit-learn Gradient Boosting), record it and tell the team.

**Example: decision tree**

```
python decision_tree.py --out runs/decision_tree
```

Use an empty output directory. The script searches depth, minimum leaf size and pruning strength on folds 1–2, selects by average precision, and evaluates the locked settings on folds 3–4 with seeds 11, 23 and 42. It saves the fold scores, predictions and a fitted preprocessing-plus-tree pipeline. The final refit uses all development flights.

## 5. Rules

- Same 35 inputs, same folds, same seeds and same metrics for every model.
- The test CSV is never read during development.
- No tuning, feature selection, encoding choices or thresholds from folds 3–4 or from December.
- Fit every encoder, imputer, scaler and PCA inside the training fold (the pipeline does this).
- Every run gets a fresh pipeline.
- When comparing with older results, use identical validation dates. Older `bench.py` scripts used different folds, so keep only their estimator factory and grid.

## 6. Final December evaluation

December 2025 (68,146 flights, starting 2025-12-01) is reserved for the final project evaluation. It is scored once per model, after all tuning is finished and all choices are frozen. Do not score it, or look at December labels, while developing. Report the exact test start date and the number of test flights.

## 7. Limitations

- One year of data and three hubs, so results may not generalise to other airports or seasons.
- Only two reporting folds (Oct 13 – Nov 28), all in autumn.
- September to November were examined in earlier prototypes, so validation is slightly optimistic.
- Earlier prototype experiments already reported December results on overlapping flights, so a fresh download does not restore a truly unseen evaluation period.
- December has a higher late rate than the development period.
- BTS supplies a retrospective schedule and event times without schedule-version or live publication timestamps. The build assumes schedule fields were known and departures observable at their event time.
- Weather uses report observation time as a proxy for when a report became available.

## 8. Sources

- BTS Reporting Carrier On-Time Performance, https://www.transtats.bts.gov (US Government Works). Downloaded in the November 25, 2026 build.
- Iowa Environmental Mesonet ASOS observations, https://mesonet.agron.iastate.edu. Downloaded in the November 25, 2026 build.
- Airport time zones: airportsdata package (MIT licence).
