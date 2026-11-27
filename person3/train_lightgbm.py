"""Run Person 3's LightGBM benchmark locally."""

import argparse
from pathlib import Path

from lightgbm import LGBMClassifier

from benchmark_runner import run_benchmark

GRID = {
    "n_estimators": [200, 400, 600, 900, 1200],
    "learning_rate": [0.02, 0.035, 0.05, 0.08, 0.12],
}


def make_estimator(seed, params):
    return LGBMClassifier(
        objective="binary",
        num_leaves=63,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
        **params,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/lightgbm"))
    args = parser.parse_args()
    run_benchmark(
        model_key="lightgbm",
        model_name="LGBMClassifier",
        make_estimator=make_estimator,
        grid=GRID,
        iteration_parameter="n_estimators",
        output=args.out,
        package_names=["lightgbm", "scikit-learn", "pandas", "numpy", "pyarrow", "joblib"],
        notes="Native categorical LightGBM input supplied by preprocessing.py; no early stopping.",
    )
