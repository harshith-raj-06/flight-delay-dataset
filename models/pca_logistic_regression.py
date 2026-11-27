"""Tune T-2h PCA + logistic regression on folds 1-2, evaluate on 3-4, and save variance analysis."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models._common import (
    SEEDS,
    finalize_eval_outputs,
    prediction_frame,
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
PCA_VARIANCE_TARGET = 0.95


def make_logreg(seed, params):
    """Build an unfitted LogisticRegression estimator (identical to logistic_regression.py)."""
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


def _evaluate_candidates(configs, fold, X, V, y, target, n_comp, exp_var, seeds):
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
                ap, auc, lift = np.nan, np.nan, np.nan
                err = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    **params,
                    "n_components": n_comp,
                    "explained_variance": exp_var,
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


def _save_pca_analysis(output, X_scaled_fold4, component_rows):
    """Save per-fold component counts plus full cumulative explained-variance curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd.DataFrame(component_rows).to_csv(output / "pca_components_per_fold.csv", index=False)

    pca_full = PCA(n_components=None, svd_solver="full").fit(X_scaled_fold4)
    indiv = pca_full.explained_variance_ratio_
    cum = np.cumsum(indiv)
    counts = np.arange(1, len(cum) + 1)
    pd.DataFrame(
        {
            "components": counts,
            "individual_variance": indiv,
            "cumulative_variance": cum,
        }
    ).to_csv(output / "pca_variance.csv", index=False)

    n95 = int(np.searchsorted(cum, PCA_VARIANCE_TARGET) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(counts, cum, marker=".", markersize=3)
    ax.axhline(PCA_VARIANCE_TARGET, color="gray", linestyle="--", label="95% variance threshold")
    ax.axvline(n95, color="crimson", linestyle=":", label=f"{n95} components at 95%")
    ax.set(
        xlabel="Principal components",
        ylabel="Cumulative explained variance",
        title="PCA cumulative explained variance (Fold 4 standardized inputs)",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pca_variance.png", dpi=160)
    plt.close(fig)


def run(output):
    """Tune on folds 1-2, evaluate on folds 3-4, save PCA analysis and refit."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dev = load_development()

    tuning_rows = []
    component_rows = []
    fold_cache = {}
    configs = list(ParameterGrid(GRID))

    for fold, train, val in development_folds(dev):
        if fold > 2:
            continue
        prep = model_pipeline("pca_logistic_regression", make_logreg(SEEDS[0], {}))[0]
        X_scaled = prep.fit_transform(train[FEATURES]).astype(np.float32)
        pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full")
        X = pca.fit_transform(X_scaled)
        V = pca.transform(prep.transform(val[FEATURES]).astype(np.float32))
        n_comp = int(pca.n_components_)
        exp_var = float(pca.explained_variance_ratio_.sum())
        y = train[TARGET].to_numpy()
        target = val[TARGET].to_numpy()
        component_rows.append(
            {
                "fold": fold,
                "input_features": X_scaled.shape[1],
                "n_components": n_comp,
                "explained_variance": exp_var,
            }
        )
        fold_cache[fold] = (X, V, y, target, n_comp, exp_var)
        tuning_rows.extend(
            _evaluate_candidates(configs, fold, X, V, y, target, n_comp, exp_var, SEEDS[:1])
        )
        print(
            f"Finished tuning fold {fold}: {n_comp}/{X_scaled.shape[1]} PCA components ({exp_var:.4f} var).",
            flush=True,
        )
        del X_scaled

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
            X, V, y, target, n_comp, exp_var = fold_cache[fold]
            tuning_rows.extend(
                _evaluate_candidates(ext_c, fold, X, V, y, target, n_comp, exp_var, SEEDS[:1])
            )
        tuning_df = pd.DataFrame(tuning_rows)
        best = _select_winner(tuning_df)
        print("Final selected after boundary check:", best, flush=True)

    fold_cache.clear()
    tuning_df.to_csv(output / "tuning.csv", index=False)

    eval_rows, pred_frames = [], []
    for fold, train, val in development_folds(dev):
        if fold <= 2:
            continue
        prep = model_pipeline("pca_logistic_regression", make_logreg(SEEDS[0], best))[0]
        X_scaled = prep.fit_transform(train[FEATURES]).astype(np.float32)
        pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full")
        X = pca.fit_transform(X_scaled)
        V = pca.transform(prep.transform(val[FEATURES]).astype(np.float32))
        n_comp = int(pca.n_components_)
        exp_var = float(pca.explained_variance_ratio_.sum())
        y = train[TARGET].to_numpy()
        target = val[TARGET].to_numpy()
        late_rate = float(target.mean())
        component_rows.append(
            {
                "fold": fold,
                "input_features": X_scaled.shape[1],
                "n_components": n_comp,
                "explained_variance": exp_var,
            }
        )
        if fold == 4:
            _save_pca_analysis(output, X_scaled, component_rows)
        del X_scaled

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
                    "n_components": n_comp,
                    "explained_variance": exp_var,
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
        print(
            f"Finished evaluation fold {fold}: {n_comp} PCA components ({exp_var:.4f} var).",
            flush=True,
        )

    _, predictions, fold_means, fold_stds, overall = finalize_eval_outputs(
        output, eval_rows, pred_frames
    )

    t_refit = perf_counter()
    final = model_pipeline("pca_logistic_regression", make_logreg(SEEDS[0], best)).fit(
        dev[FEATURES], dev[TARGET]
    )
    refit_seconds = round(perf_counter() - t_refit, 2)
    joblib.dump(final, output / "model.joblib", compress=3)

    write_settings(
        output,
        model_key="pca_logistic_regression",
        grid=GRID,
        best_params=best,
        fold_means=fold_means,
        fold_stds=fold_stds,
        overall=overall,
        development_rows=len(dev),
        predictions_rows=len(predictions),
        refit_seconds=refit_seconds,
        boundary_extensions=tested_extensions,
        extra={
            "pca_variance_threshold": PCA_VARIANCE_TARGET,
            "pca_components_per_fold": component_rows,
            "refit_n_components": int(final.named_steps["pca"].n_components_),
            "refit_explained_variance": float(
                final.named_steps["pca"].explained_variance_ratio_.sum()
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/pca_logistic_regression"))
    run(parser.parse_args().out)
