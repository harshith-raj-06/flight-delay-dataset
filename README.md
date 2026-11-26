# Flight departure delay at ATL, ORD and LAX

## Shared preprocessing for teammates

Each teammate supplies an estimator and a parameter grid. [preprocessing.py](preprocessing.py) selects its encoding, fits preprocessing inside each training fold, and uses all 35 approved T−2h predictors. The four expanding folds validate September 1–21, September 22–October 12, October 13–November 2 and November 3–28, with one full local calendar day between training and validation. December is reserved for the final evaluation.

### Example for a teammate's model script

The pattern below accepts the estimator returned by your own factory, such as `make_tree(seed)`. Parameter names use the `model__` prefix so sklearn tunes the estimator inside the pipeline.

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

Use the same tuning and evaluation blocks across teammates. In this example, sklearn's `test_*` outputs are scores on the later **development validation folds**. Earlier experiments already evaluated those months. Use a fresh pipeline per run and record the parameters, seeds, fold scores and runtime. For stochastic models repeat the locked settings with seeds 11, 23 and 42.

| Pass this model key | Preprocessing supplied by the pipeline |
|---|---|
| `decision_tree`, `random_forest`, `adaboost`, `gradient_boosting` | One-hot categories, top 20 destinations plus Other, training-median numeric imputation |
| `logistic_regression` | Same one-hot encoding, with numeric scaling fitted on training rows |
| `pca_logistic_regression` | Same scaled input, then PCA retaining 95% of training variance. Supply the same logistic classifier for both variants. |
| `histgb` | Ordinal categories marked categorical, numeric missing values preserved, automatic early stopping disabled |
| `xgboost`, `lightgbm` | Named pandas categorical fields with training-fold category levels. Unseen categories become missing. XGBoost categorical support is enabled. |
| `catboost` | Named string categories, explicit missing-category token and numeric missing values preserved |

Install the library for the estimator you choose. When moving an older script from `bench.py`, keep its estimator factory and grid, then use the pipeline and search pattern above. The older benchmark used five monthly folds starting in July. Use identical validation dates when comparing implementations.

## Run the decision tree

```bash
python decision_tree.py --out runs/decision_tree
```

Use an empty output directory. The script searches depth, minimum leaf size and pruning strength on the first two development folds, selects by average precision, and evaluates the locked settings on the last two folds with seeds 11, 23 and 42. It saves the fold scores, predictions and a fitted preprocessing-plus-tree pipeline. The final refit uses all development flights. December is reserved for final project evaluation.

## Reproduce

Use Python 3.12 or later in a virtual environment, from the repository root.

```shell
python -m pip install -r requirements.txt
python pipeline/bts.py
python pipeline/iem.py
python build_features.py
```

Start in a fresh clone for a fresh download. Interrupted downloads can be resumed by rerunning the commands. BTS fetches the 12 monthly national archives for 2025 and December 2024 for warm-up. It retains departures from Atlanta (ATL), Chicago O’Hare (ORD), and Los Angeles (LAX) plus a separate inbound intermediate for future work. IEM fetches routine and special reports for all three stations. Source URLs and request records are saved in `fresh_2025/*log.txt`. Only the files under `final_csv/` are modelling tables. Raw files, intermediate tables and CSV exports are local generated artifacts.

## Splits by origin-local scheduled flight date

| Export in final_csv/ | Dates in 2025 | Purpose |
|---|---|---|
| train_T2h.csv, train_T6h.csv | Jan 1 – Aug 30 | Model fitting |
| validation_T2h.csv, validation_T6h.csv | Sep 1 – Nov 28 | Tuning, ablations and threshold choice |
| test_T2h.csv, test_T6h.csv | Dec 1 – Dec 31 | Final evaluation |

The November 25, 2026 build downloaded 831,933 scheduled 2025 flights and 31,994 IEM reports including weather warm-up. Each horizon exports 519,275 training, 207,704 validation and 68,146 test flights (795,125 unique flights overall). The exact test start is December 1, 2025.

August 31 and November 29–30 are gap days. An entire final calendar month captures weekday and holiday variation. The January 27–31 and May 13–29 ATL prototype holdouts are excluded from training and historical outcomes. Late-maturing training or validation labels are excluded if their actual departure reaches the first checkpoint in the next evaluation period. Cancelled flights and flights missing a departure label or delay are ineligible. Diversions with valid departures stay. Additional December 2024 rows warm up history only.

## Checkpoint and evaluation

Both horizons use origin time zones with daylight saving time, converted to UTC. BTS 2400 means midnight at the end of the local date. For the ambiguous fall-back hour the converter chooses the first occurrence, and for a nonexistent local time it moves forward to the next valid time. Freeze the trained model, feature choices, encoder, destination grouping, imputer, scaler, PCA and decision threshold before final evaluation. Fit transformations within each chronological training fold. Optionally refit on training plus validation after choices are fixed. The exports contain raw categories and numeric missing values.

Airport history is a **rolling operational observation**. At each checkpoint T, only other same-origin flights that actually departed in (T−60 min, T] can supply a delay or late indicator. This rule applies in December too. Each prediction can use earlier December events after departure, as an airport observer could. Never compute a December-wide delay statistic, fit on December labels, or tune during December. Empty history means count zero, mean and share missing. Outcomes of protected ATL prototype holdouts are withheld; where a window overlaps a protected local date the four observed-history fields are missing. Scheduled demand counts all flights scheduled in (T, T+60 min], including cancellations. Earlier prototype experiments already reported December results on overlapping flights. Fresh downloads do not restore a previously unseen evaluation period.

The build checks eligibility/label agreement, horizon-specific weather age and disjoint split membership. At each horizon it independently recomputes history and scheduled demand for 300 development flights. Weather uses report observation time as a proxy for availability. BTS supplies a retrospective schedule and event times without schedule-version or live publication timestamps. The build assumes schedule fields were known and departures observable at their event time.

Sources: [BTS monthly archive](https://transtats.bts.gov/PREZIP/) and [IEM ASOS download](https://mesonet.agron.iastate.edu/request/download.phtml).
