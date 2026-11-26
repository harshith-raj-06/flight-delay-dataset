"""Tune a T-2h random forest on folds 1-2, evaluate on 3-4 and save a development refit."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

from preprocessing import (
    FEATURES,
    TARGET,
    development_folds,
    load_development,
    model_pipeline,
)

SEEDS = (11, 23, 42)
GRID = {
    "n_estimators": [100, 200],
    "max_depth": [12, 16],
    "min_samples_leaf": [12, 50],
}


def make_forest(seed, params):
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)


def run(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dev = load_development()
    tuning, evaluation = [], []
    best = None
    for fold, train, val in development_folds(dev):
        prep = model_pipeline("random_forest", make_forest(SEEDS[0], {}))[0]
        X = prep.fit_transform(train[FEATURES]).astype(np.float32)
        V = prep.transform(val[FEATURES]).astype(np.float32)
        y, target = train[TARGET], val[TARGET]
        if fold <= 2:
            configs, seeds = ParameterGrid(GRID), SEEDS[:1]
        else:
            if best is None:
                raise ValueError("The first two tuning folds must precede evaluation.")
            configs, seeds = [best], SEEDS
        for params in configs:
            for seed in seeds:
                start = perf_counter()
                model = make_forest(seed, params).fit(X, y)
                prob = model.predict_proba(V)[:, 1]
                row = dict(
                    fold=fold,
                    seed=seed,
                    **params,
                    train_rows=len(y),
                    validation_rows=len(target),
                    late_rate=target.mean(),
                    ap=average_precision_score(target, prob),
                    roc_auc=roc_auc_score(target, prob),
                    seconds=perf_counter() - start,
                )
                (tuning if fold <= 2 else evaluation).append(row)
                if fold > 2:
                    val[[TARGET]].assign(probability=prob).to_parquet(
                        output / f"predictions_{fold}_{seed}.parquet"
                    )
        print(
            f"Finished fold {fold}: {len(y):,} training, {len(target):,} validation.",
            flush=True,
        )
        if fold == 2:
            table = pd.DataFrame(tuning)
            table.to_csv(output / "tuning.csv", index=False)
            ranking = table.groupby(list(GRID), as_index=False)[
                ["ap", "roc_auc"]
            ].mean()
            winner = ranking.sort_values(
                ["ap", "roc_auc", "n_estimators"], ascending=[False, False, True]
            ).iloc[0]
            best = {
                "n_estimators": int(winner.n_estimators),
                "max_depth": int(winner.max_depth),
                "min_samples_leaf": int(winner.min_samples_leaf),
            }
            print("Selected", best, flush=True)
        del X, V
    results = pd.DataFrame(evaluation)
    results.to_csv(output / "evaluation.csv", index=False)
    means = results.groupby("fold")[["ap", "roc_auc", "late_rate"]].mean()
    print(means.to_string(), flush=True)
    print("Mean across evaluation folds", means.mean().to_dict(), flush=True)
    final = model_pipeline("random_forest", make_forest(SEEDS[0], best)).fit(
        dev[FEATURES], dev[TARGET]
    )
    joblib.dump(final, output / "random_forest.joblib", compress=3)
    (output / "settings.json").write_text(
        json.dumps(
            {
                "params": best,
                "refit_seed": SEEDS[0],
                "features": FEATURES,
                "development_rows": len(dev),
                "mean_validation": means.mean().to_dict(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/random_forest"))
    run(parser.parse_args().out)
