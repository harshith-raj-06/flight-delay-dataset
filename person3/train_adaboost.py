"""Run Person 3's AdaBoost benchmark locally."""

import argparse
from pathlib import Path

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from benchmark_runner import run_benchmark

GRID = {
    "n_estimators": [50, 100, 200, 350, 500],
    "learning_rate": [0.025, 0.05, 0.1, 0.2, 0.4],
}


def make_estimator(seed, params):
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
        random_state=seed,
        **params,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/adaboost"))
    args = parser.parse_args()
    run_benchmark(
        model_key="adaboost",
        model_name="AdaBoostClassifier with decision stumps",
        make_estimator=make_estimator,
        grid=GRID,
        iteration_parameter="n_estimators",
        output=args.out,
        package_names=["scikit-learn", "pandas", "numpy", "pyarrow", "joblib"],
        notes="All 35 approved T-2h features; no early stopping.",
    )
