"""Person 2 modelling pipeline: Baselines, Decision Tree, Random Forest, AdaBoost, and PCA.

Implements the shared T-2h evaluation protocol:
- 4 expanding chronological validation folds (Sep 1-21, Sep 22-Oct 12, Oct 13-Nov 2, Nov 3-28)
- Strictly in-fold preprocessing (top-20 Dest + Other, one-hot, cyclic sin/cos, median imputation, scaling)
- Seeds (11, 23, 42) averaged within each fold, then mean +/- std across the 4 folds
- Primary metric: Average Precision (AP / PR-AUC); Secondary: ROC-AUC
- Optional 100,000-row continuous block for fast hyperparameter tuning, full validation scoring
- Outputs validation score table, OOF probability predictions, PCA stats, and chosen settings for Person 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEEDS = (11, 23, 42)
META_COLS = ["row_id", "FlightDate", "scheduled_dep_utc", "checkpoint_utc", "wx_obs_time"]
TARGET_COL = "LATE"

FOLDS = [
    (1, "2025-09-01", "2025-09-21"),
    (2, "2025-09-22", "2025-10-12"),
    (3, "2025-10-13", "2025-11-02"),
    (4, "2025-11-03", "2025-11-28"),
]

# 35 raw predictors (T-2h)
SCHEDULE_COLS = ["month", "dow", "dep_hour", "arr_hour", "CRSElapsedTime", "Distance", "is_holiday"]
ROUTE_CARRIER_COLS = ["Origin", "Dest", "Reporting_Airline"]
CAT_COLS = ["Origin", "Dest", "Reporting_Airline", "wx_skyc1_T2h"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the 35 predictor columns in order, excluding metadata and target."""
    excluded = set(META_COLS) | {TARGET_COL}
    cols = [c for c in df.columns if c not in excluded]
    return cols


def load_dev_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train_T2h.csv and validation_T2h.csv (never touches test_T2h.csv)."""
    train_path = data_dir / "train_T2h.csv"
    val_path = data_dir / "validation_T2h.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected train_T2h.csv and validation_T2h.csv in {data_dir}. "
            "Run the dataset pipeline or point --data-dir to your CSV folder."
        )
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    for frame in (train_df, val_df):
        frame["scheduled_dep_utc"] = pd.to_datetime(frame["scheduled_dep_utc"], utc=True)
        frame["checkpoint_utc"] = pd.to_datetime(frame["checkpoint_utc"], utc=True)
    return train_df, val_df


def build_expanding_folds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    observability_buffer_hours: int = 12,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Create the 4 expanding chronological folds.

    All rows in train_T2h.csv are already verified observable before Sep 1.
    Earlier validation rows added to later folds are filtered conservatively so their
    scheduled departure is at least `observability_buffer_hours` before the fold's
    first checkpoint_utc (unless Person 1's explicit fold/observability column is used).
    """
    folds = []
    for fold_id, start_date, end_date in FOLDS:
        val_mask = val_df["FlightDate"].between(start_date, end_date)
        val_fold = val_df.loc[val_mask].copy()
        if val_fold.empty:
            raise ValueError(f"Fold {fold_id} ({start_date}..{end_date}) has 0 validation rows.")

        first_ckpt = val_fold["checkpoint_utc"].min()
        cutoff = first_ckpt - pd.Timedelta(hours=observability_buffer_hours)

        earlier_val_mask = (val_df["FlightDate"] < start_date) & (val_df["scheduled_dep_utc"] <= cutoff)
        train_fold = pd.concat([train_df, val_df.loc[earlier_val_mask]], ignore_index=True)
        train_fold = train_fold.sort_values(["checkpoint_utc", "row_id"]).reset_index(drop=True)
        val_fold = val_fold.sort_values(["checkpoint_utc", "row_id"]).reset_index(drop=True)
        folds.append((fold_id, train_fold, val_fold))
    return folds


class FoldPreprocessor:
    """Fold-safe preprocessor matching the T-2h feature guide (~80 encoded features).

    - Learns top-20 Dest categories in the training fold (others -> 'Other')
    - Preserves missing/unknown categorical values as '__MISSING__'
    - Adds 4 cyclic sin/cos features for dep_hour and month
    - Median-imputes numeric features and standard-scales all features inside the fold
    """

    def __init__(self, top_k_dest: int = 20):
        self.top_k_dest = top_k_dest
        self.top_dests_: set[str] = set()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.num_cols_: list[str] = []
        self.feature_names_: list[str] = []

    def _prepare_frame(self, df: pd.DataFrame, fit: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        feat_cols = get_feature_columns(df)
        work = df[feat_cols].copy()

        # Cyclic features (4 extra columns)
        work["dep_hour_sin"] = np.sin(2 * np.pi * work["dep_hour"].astype(float) / 24.0)
        work["dep_hour_cos"] = np.cos(2 * np.pi * work["dep_hour"].astype(float) / 24.0)
        work["month_sin"] = np.sin(2 * np.pi * (work["month"].astype(float) - 1.0) / 12.0)
        work["month_cos"] = np.cos(2 * np.pi * (work["month"].astype(float) - 1.0) / 12.0)

        # Categorical preparation
        cat_df = pd.DataFrame(index=work.index)
        for col in CAT_COLS:
            s = work[col].astype("string").fillna("__MISSING__")
            if col == "Dest":
                if fit:
                    self.top_dests_ = set(s.value_counts().head(self.top_k_dest).index.tolist())
                s = s.where(s.isin(self.top_dests_), "Other")
            cat_df[col] = s

        num_cols = [c for c in work.columns if c not in CAT_COLS]
        if fit:
            self.num_cols_ = num_cols
        num_df = work[self.num_cols_].astype(np.float32)
        return cat_df, num_df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        cat_df, num_df = self._prepare_frame(df, fit=True)
        cat_arr = self.encoder.fit_transform(cat_df)
        num_arr = self.imputer.fit_transform(num_df)
        combined = np.hstack([num_arr, cat_arr]).astype(np.float32)
        scaled = self.scaler.fit_transform(combined).astype(np.float32)
        cat_names = list(self.encoder.get_feature_names_out(CAT_COLS))
        self.feature_names_ = self.num_cols_ + cat_names
        return scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cat_df, num_df = self._prepare_frame(df, fit=False)
        cat_arr = self.encoder.transform(cat_df)
        num_arr = self.imputer.transform(num_df)
        combined = np.hstack([num_arr, cat_arr]).astype(np.float32)
        return self.scaler.transform(combined).astype(np.float32)


def get_tuning_slice(X: np.ndarray, y: np.ndarray, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous chronological block (most recent `max_rows`) for fast tuning."""
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y
    return X[-max_rows:], y[-max_rows:]


def score_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "ap": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }


def tune_on_block(
    model_name: str,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    seed: int = 11,
) -> dict[str, Any]:
    """Internal chronological 80/20 split on the continuous tuning block."""
    split = int(len(X_tune) * 0.8)
    X_tr, y_tr = X_tune[:split], y_tune[:split]
    X_va, y_va = X_tune[split:], y_tune[split:]

    if model_name == "decision_tree":
        grid = [
            {"max_depth": d, "min_samples_leaf": leaf, "ccp_alpha": alpha}
            for d in (6, 8, 10, 12)
            for leaf in (100, 300)
            for alpha in (0.0, 1e-5, 1e-4)
        ]
        builder = lambda p: DecisionTreeClassifier(random_state=seed, **p)
    elif model_name == "random_forest":
        grid = [
            {"n_estimators": 100, "max_depth": d, "min_samples_leaf": leaf, "n_jobs": -1}
            for d in (10, 14, 18)
            for leaf in (50, 200)
        ]
        builder = lambda p: RandomForestClassifier(random_state=seed, **p)
    elif model_name == "adaboost":
        grid = [
            {"n_estimators": n, "learning_rate": lr, "max_depth": d}
            for n in (50, 100)
            for lr in (0.1, 0.5, 1.0)
            for d in (1, 2)
        ]
        builder = lambda p: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=p["max_depth"], random_state=seed),
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    best_ap = -1.0
    best_params = grid[0]
    for params in grid:
        clf = builder(params)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, probs)
        if ap > best_ap:
            best_ap = ap
            best_params = params
    return best_params


def build_estimator(model_name: str, params: dict[str, Any], seed: int):
    if model_name == "constant":
        return DummyClassifier(strategy="prior")
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if model_name == "adaboost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=params["max_depth"], random_state=seed),
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=seed,
        )
    if model_name == "logreg":
        return LogisticRegression(max_iter=500, C=params.get("C", 1.0), random_state=seed)
    raise ValueError(f"Unknown model_name: {model_name}")


def run_pipeline(
    data_dir: Path,
    out_dir: Path,
    models_to_run: list[str],
    tune_rows: int = 100_000,
    pca_variance: float = 0.95,
    max_train_rows: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_df, val_df = load_dev_data(data_dir)
    folds = build_expanding_folds(train_df, val_df)

    score_rows = []
    pca_rows = []
    chosen_settings: dict[str, Any] = {
        "seeds": list(SEEDS),
        "tune_rows": tune_rows,
        "pca_variance_threshold": pca_variance,
        "folds": FOLDS,
        "models": {},
    }

    oof_frames: dict[int, pd.DataFrame] = {}
    for fold_id, _, val_fold in folds:
        oof_frames[fold_id] = val_fold[["row_id", "FlightDate", "LATE"]].copy()
        oof_frames[fold_id]["fold"] = fold_id

    for fold_id, train_fold, val_fold in folds:
        f_t0 = time.time()
        if max_train_rows > 0 and len(train_fold) > max_train_rows:
            train_fold = train_fold.iloc[-max_train_rows:].reset_index(drop=True)

        prep = FoldPreprocessor(top_k_dest=20)
        X_train = prep.fit_transform(train_fold)
        y_train = train_fold[TARGET_COL].to_numpy(dtype=int)
        X_val = prep.transform(val_fold)
        y_val = val_fold[TARGET_COL].to_numpy(dtype=int)

        X_tune, y_tune = get_tuning_slice(X_train, y_train, tune_rows)
        print(
            f"\n[Fold {fold_id}] train={len(X_train):,} val={len(X_val):,} "
            f"features={X_train.shape[1]} pos_rate(val)={y_val.mean():.4f}"
        )

        # Tune once on Fold 1 (or per fold; here we tune on Fold 1 and lock, or tune per fold)
        for model_name in models_to_run:
            if model_name == "pca_logreg":
                # Compare the exact same classifier (LogisticRegression) without vs with PCA
                pca = PCA(n_components=pca_variance, svd_solver="full", random_state=SEEDS[0])
                X_train_pca = pca.fit_transform(X_train)
                X_val_pca = pca.transform(X_val)
                n_comp = int(pca.n_components_)
                exp_var = float(pca.explained_variance_ratio_.sum())

                for variant, X_tr_v, X_va_v in [
                    ("logreg_no_pca", X_train, X_val),
                    ("logreg_with_pca", X_train_pca, X_val_pca),
                ]:
                    seed_probs = []
                    seed_aps = []
                    seed_aucs = []
                    for seed in SEEDS:
                        clf = build_estimator("logreg", {"C": 1.0}, seed)
                        clf.fit(X_tr_v, y_train)
                        probs = clf.predict_proba(X_va_v)[:, 1]
                        m = score_probs(y_val, probs)
                        seed_probs.append(probs)
                        seed_aps.append(m["ap"])
                        seed_aucs.append(m["roc_auc"])
                    mean_probs = np.mean(seed_probs, axis=0)
                    oof_frames[fold_id][f"prob_{variant}"] = mean_probs
                    score_rows.append(
                        {
                            "model": variant,
                            "fold": fold_id,
                            "ap": float(np.mean(seed_aps)),
                            "roc_auc": float(np.mean(seed_aucs)),
                            "n_features": X_tr_v.shape[1],
                        }
                    )
                pca_rows.append(
                    {
                        "fold": fold_id,
                        "original_features": X_train.shape[1],
                        "retained_components": n_comp,
                        "explained_variance_ratio_sum": exp_var,
                    }
                )
                chosen_settings["models"]["pca_logreg"] = {
                    "classifier": "LogisticRegression(max_iter=500, C=1.0)",
                    "pca_variance_threshold": pca_variance,
                }
                print(
                    f"  pca_logreg: retained {n_comp}/{X_train.shape[1]} components "
                    f"(explained_var={exp_var:.4f})"
                )
                continue

            if model_name == "constant":
                params: dict[str, Any] = {}
            else:
                if model_name not in chosen_settings["models"]:
                    params = tune_on_block(model_name, X_tune, y_tune, seed=SEEDS[0])
                    chosen_settings["models"][model_name] = params
                    print(f"  {model_name} tuned params: {params}")
                else:
                    params = chosen_settings["models"][model_name]

            seed_probs = []
            seed_aps = []
            seed_aucs = []
            for seed in SEEDS:
                clf = build_estimator(model_name, params, seed)
                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_val)[:, 1]
                m = score_probs(y_val, probs)
                seed_probs.append(probs)
                seed_aps.append(m["ap"])
                seed_aucs.append(m["roc_auc"])

            mean_probs = np.mean(seed_probs, axis=0)
            oof_frames[fold_id][f"prob_{model_name}"] = mean_probs
            ap_fold = float(np.mean(seed_aps))
            auc_fold = float(np.mean(seed_aucs))
            score_rows.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "ap": ap_fold,
                    "roc_auc": auc_fold,
                    "n_features": X_train.shape[1],
                }
            )
            print(f"  {model_name}: AP={ap_fold:.5f}, ROC-AUC={auc_fold:.5f}")

        print(f"  Fold {fold_id} finished in {time.time() - f_t0:.1f}s")

    scores_df = pd.DataFrame(score_rows)
    summary_df = (
        scores_df.groupby("model", as_index=False)
        .agg(
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
        )
        .sort_values("ap_mean", ascending=False)
    )

    scores_df.to_csv(out_dir / "person2_fold_scores.csv", index=False)
    summary_df.to_csv(out_dir / "person2_summary_scores.csv", index=False)
    if pca_rows:
        pd.DataFrame(pca_rows).to_csv(out_dir / "person2_pca_stats.csv", index=False)

    oof_all = pd.concat([oof_frames[fid] for fid, _, _ in FOLDS], ignore_index=True)
    try:
        oof_all.to_parquet(out_dir / "person2_oof_predictions.parquet", index=False)
    except ImportError:
        oof_all.to_csv(out_dir / "person2_oof_predictions.csv", index=False)

    chosen_settings["command"] = " ".join(sys.argv)
    chosen_settings["elapsed_seconds"] = round(time.time() - t0, 2)
    with (out_dir / "person2_chosen_settings.json").open("w", encoding="utf-8") as f:
        json.dump(chosen_settings, f, indent=2)

    print("\n=== Person 2 Summary (mean +/- std across 4 folds) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved Person 4 handoff files to: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Person 2 T-2h Flight Delay Pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("final_csv"), help="Folder containing train_T2h.csv and validation_T2h.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("person2_outputs"), help="Output folder for Person 4 handoff files")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["constant", "decision_tree", "random_forest", "adaboost", "pca_logreg"],
        choices=["constant", "decision_tree", "random_forest", "adaboost", "pca_logreg"],
        help="Models to run",
    )
    parser.add_argument("--tune-rows", type=int, default=100_000, help="Continuous block size inside training fold for tuning (0 = full fold)")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="Variance threshold for PCA")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional cap on training rows per fold (0 = all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        models_to_run=args.models,
        tune_rows=args.tune_rows,
        pca_variance=args.pca_variance,
        max_train_rows=args.max_train_rows,
    )
