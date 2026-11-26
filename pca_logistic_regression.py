"""Tune T-2h PCA logistic regression on folds 1-2, evaluate on 3-4 and save a development refit."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
    "C": [0.01, 0.1, 1.0, 10.0],
}


def make_logreg(seed, params):
    return LogisticRegression(
        random_state=seed, max_iter=1000, solver="lbfgs", **params
    )


def run(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dev = load_development()
    tuning, evaluation = [], []
    best = None
    for fold, train, val in development_folds(dev):
        prep = model_pipeline(
            "pca_logistic_regression", make_logreg(SEEDS[0], {})
        )[0]
        X_scaled = prep.fit_transform(train[FEATURES]).astype(np.float32)
        pca = PCA(n_components=0.95, svd_solver="full")
        X = pca.fit_transform(X_scaled)
        V_scaled = prep.transform(val[FEATURES]).astype(np.float32)
        V = pca.transform(V_scaled)
        n_components = pca.n_components_
        explained_var = float(pca.explained_variance_ratio_.sum())
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
                model = make_logreg(seed, params).fit(X, y)
                prob = model.predict_proba(V)[:, 1]
                row = dict(
                    fold=fold,
                    seed=seed,
                    **params,
                    n_components=n_components,
                    explained_variance=explained_var,
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
            f"Finished fold {fold}: {len(y):,} training, {len(target):,} validation, "
            f"{n_components} PCA components ({explained_var:.4f} variance).",
            flush=True,
        )
        if fold == 2:
            table = pd.DataFrame(tuning)
            table.to_csv(output / "tuning.csv", index=False)
            ranking = table.groupby(list(GRID), as_index=False)[
                ["ap", "roc_auc"]
            ].mean()
            winner = ranking.sort_values(
                ["ap", "roc_auc", "C"], ascending=[False, False, True]
            ).iloc[0]
            best = {"C": float(winner.C)}
            print("Selected", best, flush=True)
        del X, V, X_scaled, V_scaled
    results = pd.DataFrame(evaluation)
    results.to_csv(output / "evaluation.csv", index=False)
    means = results.groupby("fold")[["ap", "roc_auc", "late_rate"]].mean()
    print(means.to_string(), flush=True)
    print("Mean across evaluation folds", means.mean().to_dict(), flush=True)
    final = model_pipeline(
        "pca_logistic_regression", make_logreg(SEEDS[0], best)
    ).fit(dev[FEATURES], dev[TARGET])
    joblib.dump(final, output / "pca_logistic_regression.joblib", compress=3)
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
    parser.add_argument("--out", type=Path, default=Path("runs/pca_logistic_regression"))
    run(parser.parse_args().out)
