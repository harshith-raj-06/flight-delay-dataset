"""Tune T-2h logistic regression (without PCA) on folds 1-2, evaluate on 3-4, and save artifacts."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models._common import (
    SEEDS,
    finalize_eval_outputs,
    prediction_frame,
    save_feature_importances,
    write_settings,
)
from preprocessing import (
    FEATURES,
    TARGET,
    development_folds,
    load_development,
    model_pipeline,
)

GRID = {
    "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
}
C_BOUNDARIES = (0.0003, 100.0)


def make_logreg(seed, params):
    """Build an unfitted LogisticRegression estimator."""
    return LogisticRegression(
        random_state=seed, max_iter=1000, solver="lbfgs", **params
    )


def _select_winner(tuning_df):
    valid = tuning_df.dropna(subset=["ap", "roc_auc"])
    ranking = valid.groupby(list(GRID), as_index=False)[["ap", "roc_auc"]].mean()
    winner = ranking.sort_values(
        ["ap", "roc_auc", "C"], ascending=[False, False, True]
    ).iloc[0]
    return {"C": float(winner.C)}


def _evaluate_candidates(configs, fold, X, V, y, target, seeds):
    rows = []
    late_rate = float(target.mean())
    for params in configs:
        for seed in seeds:
            start = perf_counter()
            err = ""
            try:
                model = make_logreg(seed, params).fit(X, y)
                prob = model.predict_proba(V)[:, 1]
                ap = float(average_precision_score(target, prob))
                auc = float(roc_auc_score(target, prob))
                lift = ap / late_rate
            except Exception as exc:  # noqa: BLE001
                prob = None
                ap, auc, lift = np.nan, np.nan, np.nan
                err = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    **params,
                    "train_rows": len(y),
                    "validation_rows": len(target),
                    "late_rate": late_rate,
                    "ap": ap,
                    "roc_auc": auc,
                    "lift": lift,
                    "seconds": round(perf_counter() - start, 4),
                    "error": err,
                }
            )
    return rows


def run(output):
    """Tune on folds 1-2, evaluate on folds 3-4, refit on dev, and save standardized artifacts."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dev = load_development()

    tuning_rows = []
    fold_cache = {}
    configs = list(ParameterGrid(GRID))
    for fold, train, val in development_folds(dev):
        if fold > 2:
            continue
        prep = model_pipeline("logistic_regression", make_logreg(SEEDS[0], {}))["prep"]
        X = prep.fit_transform(train[FEATURES]).astype(np.float32)
        V = prep.transform(val[FEATURES]).astype(np.float32)
        y = train[TARGET].to_numpy()
        target = val[TARGET].to_numpy()
        fold_cache[fold] = (X, V, y, target)
        tuning_rows.extend(_evaluate_candidates(configs, fold, X, V, y, target, SEEDS[:1]))
        print(f"Finished tuning fold {fold}: {len(y):,} train, {len(target):,} val.", flush=True)

    tuning_df = pd.DataFrame(tuning_rows)
    best = _select_winner(tuning_df)
    print("Initial tuning winner:", best, flush=True)

    tested_extensions = []
    ext_c = []
    if best["C"] == min(GRID["C"]):
        ext_c.append({"C": C_BOUNDARIES[0]})
    if best["C"] == max(GRID["C"]):
        ext_c.append({"C": C_BOUNDARIES[1]})
    if ext_c:
        tested_extensions = ext_c
        for fold in (1, 2):
            X, V, y, target = fold_cache[fold]
            tuning_rows.extend(_evaluate_candidates(ext_c, fold, X, V, y, target, SEEDS[:1]))
        tuning_df = pd.DataFrame(tuning_rows)
        best = _select_winner(tuning_df)
        print("Final selected after boundary check:", best, flush=True)

    fold_cache.clear()
    tuning_df.to_csv(output / "tuning.csv", index=False)

    eval_rows, pred_frames = [], []
    for fold, train, val in development_folds(dev):
        if fold <= 2:
            continue
        prep = model_pipeline("logistic_regression", make_logreg(SEEDS[0], best))["prep"]
        X = prep.fit_transform(train[FEATURES]).astype(np.float32)
        V = prep.transform(val[FEATURES]).astype(np.float32)
        y = train[TARGET].to_numpy()
        target = val[TARGET].to_numpy()
        late_rate = float(target.mean())
        for seed in SEEDS:
            start = perf_counter()
            model = make_logreg(seed, best).fit(X, y)
            prob = model.predict_proba(V)[:, 1]
            ap = float(average_precision_score(target, prob))
            auc = float(roc_auc_score(target, prob))
            eval_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    **best,
                    "train_rows": len(y),
                    "validation_rows": len(target),
                    "late_rate": late_rate,
                    "ap": ap,
                    "roc_auc": auc,
                    "lift": ap / late_rate,
                    "seconds": round(perf_counter() - start, 4),
                }
            )
            pred_frames.append(prediction_frame(val.index.to_numpy(), fold, seed, target, prob))
        del X, V
        print(f"Finished evaluation fold {fold}: {len(y):,} train, {len(target):,} val.", flush=True)

    _, predictions, fold_means, fold_stds, overall = finalize_eval_outputs(
        output, eval_rows, pred_frames
    )

    t_refit = perf_counter()
    final = model_pipeline("logistic_regression", make_logreg(SEEDS[0], best)).fit(
        dev[FEATURES], dev[TARGET]
    )
    refit_seconds = round(perf_counter() - t_refit, 2)
    joblib.dump(final, output / "model.joblib", compress=3)

    save_feature_importances(
        output,
        final.named_steps["prep"],
        final.named_steps["model"].coef_[0],
        col_name="coefficient",
        abs_group=True,
    )

    write_settings(
        output,
        model_key="logistic_regression",
        grid=GRID,
        best_params=best,
        fold_means=fold_means,
        fold_stds=fold_stds,
        overall=overall,
        development_rows=len(dev),
        predictions_rows=len(predictions),
        refit_seconds=refit_seconds,
        boundary_extensions=tested_extensions,
        importance_method="LogisticRegression.coef_[0] on standardized one-hot inputs (grouped by sum of absolute standardized weights)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/logistic_regression"))
    run(parser.parse_args().out)
