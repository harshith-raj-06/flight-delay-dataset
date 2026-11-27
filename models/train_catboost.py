"""Run Person 3's CatBoost benchmark locally."""

import argparse
from pathlib import Path

from catboost import CatBoostClassifier

from benchmark_runner import run_benchmark

GRID = {
    "iterations": [200, 400, 600, 900, 1200],
    "learning_rate": [0.02, 0.035, 0.05, 0.08, 0.12],
}


def make_estimator(seed, params):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
        **params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/catboost"))
    args = parser.parse_args()
    run_benchmark(
        model_key="catboost",
        model_name="CatBoostClassifier",
        make_estimator=make_estimator,
        grid=GRID,
        iteration_parameter="iterations",
        output=args.out,
        package_names=["catboost", "scikit-learn", "pandas", "numpy", "pyarrow", "joblib"],
        notes="Native string categorical CatBoost input supplied by preprocessing.py; no early stopping.",
    )
