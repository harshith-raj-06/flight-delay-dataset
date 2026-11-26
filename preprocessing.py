"""Common T-2h features, chronological folds and unfitted transformers."""

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "final_csv"
TARGET = "LATE"
METADATA = [
    "row_id",
    "FlightDate",
    "scheduled_dep_utc",
    "checkpoint_utc",
    "wx_obs_time",
]
SCHEDULE = [
    "month",
    "dow",
    "dep_hour",
    "arr_hour",
    "CRSElapsedTime",
    "Distance",
    "is_holiday",
]
ROUTE = ["Origin", "Dest", "Reporting_Airline"]
WEATHER = [
    "wx_skyc1_T2h",
    "wx_tmpf_T2h",
    "wx_dwpf_T2h",
    "wx_drct_T2h",
    "wx_sknt_T2h",
    "wx_gust_T2h",
    "wx_vsby_T2h",
    "wx_p01i_T2h",
    "wx_skyl1_T2h",
    "wx_obs_age_min_T2h",
    "wx_ts_T2h",
    "wx_fg_T2h",
    "wx_ra_T2h",
    "wx_sn_T2h",
    "wx_fz_T2h",
    "wx_br_hz_T2h",
    "wx_gust_missing_T2h",
    "wx_skyl1_missing_T2h",
    "codes_known_T2h",
    "wx_unmatched_T2h",
]
AIRPORT_ACTIVITY = [
    "n_departed_60min_T2h",
    "prior_mean_delay_60min_T2h",
    "late_rate_60min_T2h",
    "n_scheduled_next_60min_T2h",
    "has_history_T2h",
]
FEATURES = SCHEDULE + ROUTE + WEATHER + AIRPORT_ACTIVITY
CATEGORICAL = ["Origin", "Dest", "Reporting_Airline", "wx_skyc1_T2h"]
FOLDS = [
    (1, "2025-09-01", "2025-09-21"),
    (2, "2025-09-22", "2025-10-12"),
    (3, "2025-10-13", "2025-11-02"),
    (4, "2025-11-03", "2025-11-28"),
]
GAP_DAYS = 1
assert len(FEATURES) == len(set(FEATURES)) == 35


def load_development(data_dir=DATA_DIR):
    """Read and verify the development CSVs; the test CSV is outside this function."""
    parts = []
    for name, first, last in (
        ("train", "2025-01-01", "2025-08-30"),
        ("validation", "2025-09-01", "2025-11-28"),
    ):
        frame = pd.read_csv(
            Path(data_dir) / f"{name}_T2h.csv", parse_dates=METADATA[2:]
        )
        if list(frame) != METADATA + FEATURES + [TARGET]:
            raise ValueError(f"Unexpected columns in {name}_T2h.csv")
        if not frame.FlightDate.between(first, last).all():
            raise ValueError(f"Unexpected dates in {name}_T2h.csv")
        parts.append(frame)
    dev = pd.concat(parts, ignore_index=True).set_index("row_id")
    if not dev.index.is_unique or not dev[TARGET].isin((0, 1)).all():
        raise ValueError("Duplicate row IDs or invalid target")
    if not (dev.scheduled_dep_utc - dev.checkpoint_utc).eq(pd.Timedelta(hours=2)).all():
        raise ValueError("A checkpoint is not two hours before departure")
    age = (dev.checkpoint_utc - dev.wx_obs_time).dt.total_seconds() / 60
    if not age.dropna().between(0, 180).all():
        raise ValueError("Invalid weather observation time")
    return dev.sort_values("checkpoint_utc", kind="stable")


def development_folds(dev):
    """Expanding folds with one full local-date gap before each validation block."""
    for fold, first, last in FOLDS:
        train = dev[
            pd.to_datetime(dev.FlightDate)
            < pd.Timestamp(first) - pd.Timedelta(days=GAP_DAYS)
        ]
        validation = dev[dev.FlightDate.between(first, last)]
        if train.empty or validation.empty:
            raise ValueError(f"Fold {fold} has no rows")
        if train.checkpoint_utc.max() >= validation.checkpoint_utc.min():
            raise ValueError(f"Fold {fold} overlaps in checkpoint time")
        yield fold, train, validation


class TopDestinations(BaseEstimator, TransformerMixin):
    """Group destinations outside the training fold's most frequent 20."""

    def __init__(self, count=20):
        self.count = count

    def fit(self, values, y=None):
        self.keep_ = set(
            pd.DataFrame(values).iloc[:, 0].value_counts().head(self.count).index
        )
        return self

    def transform(self, values):
        column = pd.DataFrame(values).iloc[:, 0]
        return column.where(column.isin(self.keep_), "Other").to_numpy().reshape(-1, 1)


class EmptyColumnGuard(TransformerMixin, BaseEstimator):
    """Represent numeric inputs entirely missing during fit by a fixed constant."""

    def fit(self, X, y=None):
        self.empty_ = X.columns[X.isna().all()]
        return self

    def transform(self, X):
        return X.assign(**{col: 0.0 for col in self.empty_})


def make_preprocessor(kind, features=FEATURES):
    """Return an unfitted transformer: onehot, onehot_scaled, or ordinal."""
    if (
        not features
        or len(set(features)) != len(features)
        or set(features) - set(FEATURES)
    ):
        raise ValueError("Select distinct approved T-2h inputs")
    cats = [name for name in features if name in CATEGORICAL]
    nums = [name for name in features if name not in CATEGORICAL]
    fill = ("missing", SimpleImputer(strategy="constant", fill_value="__MISSING__"))
    if kind == "ordinal":
        parts = []
        if cats:
            parts.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            fill,
                            (
                                "encode",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                    max_categories=254,
                                ),
                            ),
                        ]
                    ),
                    cats,
                )
            )
        if nums:
            parts.append(("numeric", EmptyColumnGuard(), nums))
        return ColumnTransformer(parts, sparse_threshold=0)
    if kind not in ("onehot", "onehot_scaled"):
        raise ValueError("kind must be onehot, onehot_scaled, or ordinal")
    parts = []
    if "Dest" in cats:
        parts.append(
            (
                "destination",
                Pipeline(
                    [
                        fill,
                        ("top20", TopDestinations()),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                ["Dest"],
            )
        )
    other = [name for name in cats if name != "Dest"]
    if other:
        parts.append(
            (
                "categorical",
                Pipeline(
                    [
                        fill,
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                other,
            )
        )
    if nums:
        numeric_steps = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if kind == "onehot_scaled":
            numeric_steps.append(("scale", StandardScaler()))
        parts.append(("numeric", Pipeline(numeric_steps), nums))
    return ColumnTransformer(parts, sparse_threshold=0)


def categorical_frame(frame, features=FEATURES):
    """Copy selected inputs with string categories for native categorical APIs."""
    if (
        not features
        or len(set(features)) != len(features)
        or set(features) - set(FEATURES)
    ):
        raise ValueError("Select distinct approved T-2h inputs")
    result = frame.loc[:, features].copy()
    for name in CATEGORICAL:
        if name in result:
            result[name] = (
                result[name].astype("string").fillna("__MISSING__").astype("category")
            )
    return result


# Only the representation and input requirements live here. The caller supplies
# the estimator, hyperparameters and seed.
ENCODING = {
    "decision_tree": "onehot",
    "random_forest": "onehot",
    "adaboost": "onehot",
    "gradient_boosting": "onehot",
    "logistic_regression": "onehot_scaled",
    "pca_logistic_regression": "onehot_scaled",
    "histgb": "ordinal",
    "xgboost": "category",
    "lightgbm": "category",
    "catboost": "string",
}


class NativeInputs(TransformerMixin, BaseEstimator):
    """Preserve column names and learn a consistent category vocabulary per fold."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y=None):
        self.levels_ = {c: list(X[c].dropna().unique()) for c in CATEGORICAL}
        return self

    def transform(self, X):
        frame = X[FEATURES].copy()
        for col, levels in self.levels_.items():
            if self.kind == "string":
                frame[col] = (
                    frame[col].astype("string").fillna("__MISSING__").astype(object)
                )
            else:
                frame[col] = pd.Categorical(
                    frame[col].where(frame[col].isin(levels)), categories=levels
                )
        return frame


def model_pipeline(model_name, estimator):
    """Wrap a teammate's unfitted estimator with its agreed T-2h preprocessing."""
    if model_name not in ENCODING:
        raise ValueError(f"Choose a model key from {list(ENCODING)}")
    kind = ENCODING[model_name]
    prep = (
        NativeInputs(kind)
        if kind in ("category", "string")
        else make_preprocessor(kind)
    )
    model = clone(estimator)
    if model_name == "histgb":
        model.set_params(
            categorical_features=list(range(len(CATEGORICAL))), early_stopping=False
        )
    elif model_name == "xgboost":
        model.set_params(enable_categorical=True)
    elif model_name == "catboost":
        model.set_params(cat_features=list(CATEGORICAL))
    steps = [("prep", prep)]
    if model_name == "pca_logistic_regression":
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    return Pipeline([*steps, ("model", model)])


def chronological_cv(dev):
    """Positional fold indices for sklearn search and cross-validation on dev in this order."""
    return [
        (dev.index.get_indexer(train.index), dev.index.get_indexer(val.index))
        for _, train, val in development_folds(dev)
    ]


if __name__ == "__main__":
    development = load_development()
    for fold, train, validation in development_folds(development):
        print(
            f"Fold {fold}: {len(train):,} train, {len(validation):,} validation, "
            f"{validation.FlightDate.min()} to {validation.FlightDate.max()}"
        )
