"""Shared T-2h preprocessing. Run directly to audit the four development folds.

Custom estimators should use the same read_dev, raw_events, folds and feature
allowlist as train.py. Fit a new transformer on each fold's training rows.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from build_features import HISTORY, META, ROOT, SHARED, WX
from person2.person2_pipeline import CAT_COLS, FOLDS, FoldPreprocessor

SUFFIX = "_T2h"
WEATHER = ["wx_" + x + SUFFIX for x in WX] + ["codes_known_T2h", "wx_unmatched_T2h"]
ACTIVITY = [x + SUFFIX for x in HISTORY]
ALL = SHARED + WEATHER + ACTIVITY
LADDER = {
    "schedule": SHARED[:7],
    "route": SHARED,
    "weather": SHARED + WEATHER,
    "full": ALL,
}


class NativeCategories(TransformerMixin, BaseEstimator):
    """Fold-specific pandas categories or explicit missing strings for CatBoost."""

    def __init__(self, cats, as_strings=False):
        self.cats = cats
        self.as_strings = as_strings

    def fit(self, X, y=None):
        self.levels_ = {k: sorted(X[k].dropna().unique()) for k in self.cats}
        return self

    def transform(self, X):
        frame = X.copy()
        for col, levels in self.levels_.items():
            if self.as_strings:
                frame[col] = (
                    frame[col].astype("string").fillna("__MISSING__").astype(object)
                )
            else:
                frame[col] = pd.Categorical(
                    frame[col].where(frame[col].isin(levels)), categories=levels
                )
        return frame


def make_preprocessor(name, cols):
    """Return an unfitted transformer for a named model and approved input list."""
    if not cols or len(set(cols)) != len(cols) or set(cols) - set(ALL):
        raise ValueError("Expected a nonempty list of distinct T-2h predictors")
    cats = [c for c in CAT_COLS if c in cols]
    if name == "histgb":
        return ColumnTransformer(
            [
                (
                    "categorical",
                    OrdinalEncoder(
                        max_categories=254,
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                    ),
                    cats,
                )
            ],
            remainder="passthrough",
        )
    if name in ("xgboost", "lightgbm", "catboost"):
        return NativeCategories(cats, as_strings=name == "catboost")
    if name in (
        "constant",
        "decision_tree",
        "random_forest",
        "adaboost",
        "gradient",
        "logreg",
        "pca_logreg",
    ):
        return FoldPreprocessor()
    raise ValueError(f"Unknown model {name}")


def read_dev(directory):
    data = []
    for part, start, end in [
        ("train", "2025-01-01", "2025-08-30"),
        ("validation", FOLDS[0][1], FOLDS[-1][2]),
    ]:
        frame = pd.read_csv(directory / f"{part}_T2h.csv", parse_dates=META[2:])
        if (
            list(frame) != META + ALL + ["LATE"]
            or not frame.FlightDate.between(start, end).all()
        ):
            raise ValueError(
                f"Unexpected {part} dates or schema: {frame.columns.tolist()}"
            )
        data.append(frame)
    dev = (
        pd.concat(data, ignore_index=True)
        .set_index("row_id")
        .sort_values("checkpoint_utc", kind="stable")
    )
    if not dev.index.is_unique:
        raise ValueError("Development row IDs overlap or repeat")
    assert dev.LATE.isin([0, 1]).all()
    assert (dev.scheduled_dep_utc - dev.checkpoint_utc).eq(pd.Timedelta(hours=2)).all()
    age = (dev.checkpoint_utc - dev.wx_obs_time).dt.total_seconds() / 60
    np.testing.assert_allclose(age, dev.wx_obs_age_min_T2h, equal_nan=True)
    assert (
        age.dropna().between(0, 180).all() and age.isna().eq(dev.wx_unmatched_T2h).all()
    )
    assert (
        dev[["wx_" + k + SUFFIX for k in ["ts", "fg", "ra", "sn", "fz", "br_hz"]]]
        .notna()
        .eq(dev.codes_known_T2h, axis=0)
        .all()
        .all()
    )
    return dev


def raw_events(path, dev):
    # The dataset builder concatenates whole BTS months in order. Predicate pushdown
    # preserves original row positions for this pre-December prefix, including warm-up.
    cols = [
        "Origin",
        "FlightDate",
        "scheduled_dep_utc",
        "Cancelled",
        "DepDelay",
        "DepDel15",
    ]
    events = pd.read_parquet(
        path, columns=cols, filters=[("FlightDate", "<", "2025-12-01")]
    ).reset_index(drop=True)
    matched = events.loc[dev.index]
    for col in ["Origin", "FlightDate", "scheduled_dep_utc"]:
        np.testing.assert_array_equal(matched[col], dev[col])
    np.testing.assert_array_equal(matched.DepDel15, dev.LATE)
    events["actual"] = events.scheduled_dep_utc + pd.to_timedelta(
        events.DepDelay, unit="min"
    )
    return events


def history_check(dev, events):
    eligible = (
        events.Cancelled.eq(0) & events.DepDelay.notna() & events.DepDel15.notna()
    )
    blocked = events.Origin.eq("ATL") & (
        events.FlightDate.between("2025-01-27", "2025-01-31")
        | events.FlightDate.between("2025-05-13", "2025-05-29")
    )
    checks = []
    for idx, r in (
        dev[dev.FlightDate.ge("2025-09-01")].sample(5, random_state=42).iterrows()
    ):
        t = r.checkpoint_utc
        same = events.Origin.eq(r.Origin)
        past = events.loc[
            same
            & eligible
            & ~blocked
            & events.actual.gt(t - pd.Timedelta(hours=1))
            & events.actual.le(t)
        ]
        past = past.drop(idx, errors="ignore")
        demand = (
            same
            & events.scheduled_dep_utc.gt(t)
            & events.scheduled_dep_utc.le(t + pd.Timedelta(hours=1))
        ).sum()
        expected = [
            len(past),
            past.DepDelay.mean(),
            past.DepDelay.ge(15).mean(),
            demand,
            int(bool(len(past))),
        ]
        np.testing.assert_allclose(r[ACTIVITY].astype(float), expected, equal_nan=True)
        checks.append(
            dict(
                row_id=idx,
                origin=r.Origin,
                checkpoint=t,
                included_ids=" ".join(map(str, past.index)),
                **dict(zip(ACTIVITY, expected)),
            )
        )
    return pd.DataFrame(checks)


def folds(dev, actual, smoke=False):
    if smoke:
        block = dev.iloc[:20000]
        first = block.checkpoint_utc.iloc[16000]
        yield (
            0,
            block[(block.checkpoint_utc < first) & (actual.loc[block.index] < first)],
            block[block.checkpoint_utc >= first],
        )
        return
    for fold, start, end in FOLDS:
        val = dev[dev.FlightDate.between(start, end)]
        t = val.checkpoint_utc.min()
        train = dev[(dev.FlightDate < start) & (dev.checkpoint_utc < t) & (actual < t)]
        assert train.checkpoint_utc.max() < t and actual.loc[train.index].max() < t
        yield fold, train, val


if __name__ == "__main__":
    dev = read_dev(ROOT / "final_csv")
    events = raw_events(ROOT / "fresh_2025/output_with_weather.parquet", dev)
    history_check(dev, events)
    actual = events.actual.loc[dev.index]
    membership = pd.Index([])
    for fold, tr, va in folds(dev, actual, smoke=False):
        assert (
            membership.intersection(va.index).empty
            and tr.index.intersection(va.index).empty
        )
        membership = membership.append(va.index)
        for model in ("logreg", "histgb", "xgboost", "catboost"):
            prep = make_preprocessor(model, ALL)
            prep.fit(tr[ALL])
            matrix = prep.transform(va[ALL])
            assert matrix.shape[0] == len(va)
        print(
            f"Fold {fold}: {va.FlightDate.min()} to {va.FlightDate.max()}, "
            f"{len(tr):,} train, {len(va):,} validation",
            flush=True,
        )
    assert set(membership) == set(dev.index[dev.FlightDate >= FOLDS[0][1]])
