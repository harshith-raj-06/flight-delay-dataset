"""Run Person 3's scalable sklearn HistGradientBoosting benchmark locally."""

import argparse
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_runner import run_benchmark

GRID = {
    "max_iter": [200, 350, 500, 700, 900],
    "learning_rate": [0.03, 0.05, 0.08, 0.12, 0.2],
}


def make_estimator(seed, params):
    return HistGradientBoostingClassifier(
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
        **params,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/histgb"))
    args = parser.parse_args()
    run_benchmark(
        model_key="histgb",
        model_name="HistGradientBoostingClassifier",
        make_estimator=make_estimator,
        grid=GRID,
        iteration_parameter="max_iter",
        output=args.out,
        package_names=["scikit-learn", "pandas", "numpy", "pyarrow", "joblib"],
        notes="Recorded substitution for classic sklearn GradientBoostingClassifier because the full-data protocol makes the histogram implementation practical.",
    )
