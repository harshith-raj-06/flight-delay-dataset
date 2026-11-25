# Flight departure delay at ATL, ORD and LAX

## Train a model

Run the command for your model from the repository root. Click the model name to read its implementation. All commands use the same T−2h dataset and chronological validation folds. For installation and required data files, see [training setup](train.md#1-set-up-the-environment-and-inputs).

| Model code | Training command |
|---|---|
| [Constant baseline](person2/person2_pipeline.py#L239) | `python train.py --models constant --out runs/constant` |
| [Decision Tree](person2/person2_pipeline.py#L241) | `python train.py --models decision_tree --out runs/decision_tree` |
| [Random Forest](person2/person2_pipeline.py#L243) | `python train.py --models random_forest --out runs/random_forest` |
| [AdaBoost](person2/person2_pipeline.py#L245) | `python train.py --models adaboost --out runs/adaboost` |
| [Logistic Regression](person2/person2_pipeline.py#L252) | `python train.py --models logreg --out runs/logreg` |
| [PCA + Logistic Regression](train.py#L150) | `python train.py --models pca_logreg --out runs/pca_logreg` |
| [Gradient Boosting (sklearn)](train.py#L140) | `python train.py --models gradient --out runs/gradient` |
| [Histogram Gradient Boosting](train.py#L101) | `python train.py --models histgb --out runs/histgb` |
| [XGBoost](train.py#L112) | `python train.py --models xgboost --out runs/xgboost` |
| [LightGBM](train.py#L122) | `python train.py --models lightgbm --out runs/lightgbm` |
| [CatBoost](train.py#L128) | `python train.py --models catboost --out runs/catboost` |

Each command writes scores to its own folder under `runs/`. For preprocessing, tuning and the meaning of each output, see [train.md](train.md).

### What the folders contain

- `pipeline/` downloads BTS flights and joins IEM weather. Use it when rebuilding the dataset.
- `benchmarks/` contains saved validation scores and audits from completed reference runs.
- `person2/` and `person3/` contain earlier scripts and notebooks. The shared trainer imports preprocessing and estimator definitions from `person2/`. Run the commands above for any model, regardless of who is training it.

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

For training setup, individual model commands, chronological benchmarks and the final refit, see [train.md](train.md).
