"""T-2h chronological development training. See train.md for commands."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from build_features import HISTORY, META, ROOT, SHARED, WX
from person2.person2_pipeline import (
    CAT_COLS,
    FOLDS,
    SEEDS,
    FoldPreprocessor,
    build_estimator,
)

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
DEFAULTS = {
    "constant": {},
    "decision_tree": {"max_depth": 12, "min_samples_leaf": 50, "ccp_alpha": 0.00001},
    "random_forest": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 12},
    "logreg": {"C": 1.0},
    "pca_logreg": {"C": 1.0},
    "adaboost": {"max_depth": 2, "n_estimators": 60, "learning_rate": 0.1},
    "gradient": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08},
    "histgb": {"max_iter": 120, "max_leaf_nodes": 31},
    "xgboost": {"n_estimators": 150, "max_depth": 6, "subsample": 0.85},
    "lightgbm": {
        "n_estimators": 150,
        "num_leaves": 31,
        "subsample": 0.85,
        "subsample_freq": 1,
    },
    "catboost": {"iterations": 150, "depth": 6},
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
                frame[col] = pd.Categorical(frame[col], categories=levels)
        return frame


def model_pipeline(name, cols, seed, jobs, overrides):
    params = DEFAULTS[name] | overrides
    cats = [c for c in CAT_COLS if c in cols]
    steps = []
    if name == "histgb":
        preprocess = ColumnTransformer(
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
        model = HistGradientBoostingClassifier(
            **params,
            categorical_features=list(range(len(cats))) or None,
            random_state=seed,
            early_stopping=False,
        )
    elif name in ("xgboost", "lightgbm", "catboost"):
        preprocess = NativeCategories(cats, as_strings=name == "catboost")
        if name == "xgboost":
            from xgboost import XGBClassifier

            model = XGBClassifier(
                **params,
                tree_method="hist",
                enable_categorical=True,
                random_state=seed,
                n_jobs=jobs,
            )
        elif name == "lightgbm":
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                **params, random_state=seed, n_jobs=jobs, verbosity=-1
            )
        else:
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                **params,
                cat_features=cats,
                random_seed=seed,
                thread_count=jobs,
                verbose=False,
                allow_writing_files=False,
                has_time=True,
            )
    else:
        preprocess = FoldPreprocessor()
        if name == "gradient":
            model = GradientBoostingClassifier(
                **params, random_state=seed, n_iter_no_change=None
            )
        else:
            if name == "random_forest":
                params["n_jobs"] = jobs
            model = build_estimator(
                "logreg" if name == "pca_logreg" else name, params, seed
            )
        if name == "pca_logreg":
            steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    return Pipeline([("preprocess", preprocess), *steps, ("classifier", model)])


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
        .set_index("row_id", verify_integrity=True)
        .sort_values("checkpoint_utc", kind="stable")
    )
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


def folds(dev, actual, smoke):
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


def column_audit():
    rows = []
    for line in (ROOT / "COLUMN_AUDIT.md").read_text().splitlines():
        if line.startswith("| ") and not line.startswith("| Input"):
            name, kind, reason = [x.strip() for x in line.strip("|").split("|")]
            rows.append(
                (name if name in SHARED else name + SUFFIX, kind, reason, "predictor")
            )
    assert {r[0] for r in rows} == set(ALL)
    rows.extend(
        (c, "audit", "Identifier or timestamp used only for verification.", "metadata")
        for c in META
    )
    rows.append(
        ("LATE", "outcome", "Departure delay of at least 15 minutes.", "target")
    )
    return pd.DataFrame(rows, columns=["column", "type", "reason", "role"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=list(DEFAULTS), default=["constant", "histgb"]
    )
    parser.add_argument("--stage", choices=LADDER, default="full")
    parser.add_argument("--drop", nargs="+", choices=ACTIVITY, default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Refit one locked configuration on all development rows and save its pipeline.",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON object of hyperparameter overrides for one estimator",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Latest N eligible training rows per fold; 0 uses all",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "final_csv")
    parser.add_argument(
        "--events", type=Path, default=ROOT / "fresh_2025/output_with_weather.parquet"
    )
    args = parser.parse_args()
    overrides = json.loads(args.params)
    if not isinstance(overrides, dict) or (
        overrides
        and (
            len(args.models) != 1
            or not overrides.keys() <= DEFAULTS[args.models[0]].keys()
        )
    ):
        parser.error(
            "Use one model and only its documented configuration keys for overrides."
        )
    if (
        args.jobs < 1
        or args.train_limit < 0
        or (args.out.exists() and any(args.out.iterdir()))
    ):
        parser.error(
            "Choose positive jobs, nonnegative train-limit, and an empty output directory."
        )
    if args.refit and (
        len(args.models) != 1 or len(args.seeds) != 1 or args.smoke or args.train_limit
    ):
        parser.error(
            "Refit requires exactly one model, one seed, and all development rows."
        )
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(
        args.seeds
    ):
        parser.error("Models and seeds must be unique within a run.")
    cols = [c for c in LADDER[args.stage] if c not in args.drop]
    dev = read_dev(args.data_dir)
    events = raw_events(args.events, dev)
    actual = events.actual.loc[dev.index]
    checks = history_check(dev, events)
    args.out.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.out / "history_checks.csv", index=False)
    audit = column_audit()
    audit["selected"] = audit.column.isin(cols)
    audit.to_csv(args.out / "column_audit.csv", index=False)
    print("Development schema:", META + ALL + ["LATE"], flush=True)
    print(
        "Development weather codes known:",
        dev.codes_known_T2h.mean(),
        "unmatched:",
        dev.wx_unmatched_T2h.sum(),
        flush=True,
    )
    print(checks.drop(columns="included_ids").to_string(index=False), flush=True)
    (args.out / "config.json").write_text(
        json.dumps(
            {"arguments": vars(args), "features": cols, "defaults": DEFAULTS},
            indent=2,
            default=str,
        )
    )
    if args.refit:
        # Classes in a saved pipeline must resolve by importable module name.
        from train import model_pipeline as saved_pipeline

        model = saved_pipeline(
            args.models[0], cols, args.seeds[0], args.jobs, overrides
        )
        start = time.perf_counter()
        with threadpool_limits(limits=args.jobs):
            model.fit(dev[cols], dev.LATE.astype(int))
        elapsed = time.perf_counter() - start
        joblib.dump(model, args.out / "model.joblib", compress=3)
        print(f"Refitted on {len(dev)} development rows in {elapsed:.1f}s.")
        return
    scores = []
    for fold, train, val in folds(dev, actual, args.smoke):
        if args.train_limit:
            train = train.tail(args.train_limit)
        print(
            f"Fold {fold} train={len(train)} late={train.LATE.mean():.4f} validation={len(val)} late={val.LATE.mean():.4f}",
            flush=True,
        )
        for name in args.models:
            for seed in args.seeds:
                pipe = model_pipeline(name, cols, seed, args.jobs, overrides)
                start = time.perf_counter()
                with threadpool_limits(limits=args.jobs):
                    pipe.fit(train[cols], train.LATE.astype(int))
                    prob = pipe.predict_proba(val[cols])[:, 1]
                seconds = time.perf_counter() - start
                result = {
                    "model": name,
                    "stage": args.stage,
                    "fold": fold,
                    "seed": seed,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "train_late": train.LATE.mean(),
                    "validation_late": val.LATE.mean(),
                    "ap": average_precision_score(val.LATE, prob),
                    "roc_auc": roc_auc_score(val.LATE, prob),
                    "seconds": seconds,
                }
                if "pca" in pipe.named_steps:
                    result.update(
                        components=pipe["pca"].n_components_,
                        explained_variance=pipe["pca"].explained_variance_ratio_.sum(),
                    )
                scores.append(result)
                val[["FlightDate", "LATE"]].assign(probability=prob).to_parquet(
                    args.out / f"pred_{name}_{fold}_{seed}.parquet"
                )
                pd.DataFrame(scores).to_csv(args.out / "fold_scores.csv", index=False)
                print(
                    name,
                    seed,
                    f"AP={result['ap']:.4f} ROC-AUC={result['roc_auc']:.4f} {seconds:.1f}s",
                    flush=True,
                )
    per_fold = pd.DataFrame(scores).groupby(["model", "fold"])[["ap", "roc_auc"]].mean()
    summary = per_fold.groupby("model").agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(args.out / "results.csv")
    print(summary)


if __name__ == "__main__":
    main()
