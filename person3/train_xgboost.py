"""Run Person 3's XGBoost benchmark locally."""

import argparse
from pathlib import Path

from xgboost import XGBClassifier

from benchmark_runner import run_benchmark

GRID = {
    "n_estimators": [200, 400, 600, 900, 1200],
    "learning_rate": [0.02, 0.035, 0.05, 0.08, 0.12],
}


def make_estimator(seed, params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_depth=7,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=seed,
        **params,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/xgboost"))
    args = parser.parse_args()
    run_benchmark(
        model_key="xgboost",
        model_name="XGBClassifier",
        make_estimator=make_estimator,
        grid=GRID,
        iteration_parameter="n_estimators",
        output=args.out,
        package_names=["xgboost", "scikit-learn", "pandas", "numpy", "pyarrow", "joblib"],
        notes="Native categorical XGBoost input supplied by preprocessing.py; no early stopping.",
    )
