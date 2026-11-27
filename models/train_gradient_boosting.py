"""Run Person 3's classic sklearn GradientBoosting benchmark locally."""

import argparse
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

from benchmark_runner import run_benchmark

GRID = {
    "n_estimators": [50, 100, 200, 350, 500],
    "learning_rate": [0.025, 0.05, 0.1, 0.2, 0.4],
}


def make_estimator(seed, params):
    return GradientBoostingClassifier(
        max_depth=2,
        min_samples_leaf=100,
        subsample=1.0,
        random_state=seed,
        **params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/gradient_boosting"))
    args = parser.parse_args()
    run_benchmark(
        model_key="gradient_boosting",
        model_name="GradientBoostingClassifier",
        make_estimator=make_estimator,
        grid=GRID,
        iteration_parameter="n_estimators",
        output=args.out,
        package_names=["scikit-learn", "pandas", "numpy", "pyarrow", "joblib"],
        notes="Classic sklearn gradient boosting, as required by the original model plan; no early stopping.",
    )
