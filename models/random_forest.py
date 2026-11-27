"""Tune T-2h random forest on folds 1-2, evaluate on 3-4, and save standardized artifacts."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# 27 tree-structure configs at n_estimators=100, plus n_estimators sweep & boundaries.
GRID = {
    "n_estimators": [100],
    "max_depth": [12, 16, 20],
    "min_samples_leaf": [25, 75, 200],
    "max_features": ["sqrt", 0.3, 0.5],
}
N_ESTIMATORS_SWEEP = [50, 100, 150, 200]
BOUNDARIES = {
    "max_depth": (8, 24),
    "min_samples_leaf": (10, 400),
    "max_features": (None, 0.7),
    "n_estimators": (None, 300),
}


def make_forest(seed, params, jobs=4):
    """Build an unfitted RandomForestClassifier."""
    return RandomForestClassifier(random_state=seed, n_jobs=jobs, **params)


def _select_winner(tuning_df):
    valid = tuning_df.dropna(subset=["ap", "roc_auc"]).copy()
    valid["max_features_key"] = valid["max_features"].astype(str)
    group_cols = ["n_estimators", "max_depth", "min_samples_leaf", "max_features_key"]
    ranking = valid.groupby(group_cols, as_index=False)[["ap", "roc_auc"]].mean()
    winner = ranking.sort_values(
        ["ap", "roc_auc", "n_estimators", "max_depth"],
        ascending=[False, False, True, True],
    ).iloc[0]
    mf_raw = winner.max_features_key
    mf = mf_raw if mf_raw == "sqrt" else float(mf_raw)
    return {
        "n_estimators": int(winner.n_estimators),
        "max_depth": int(winner.max_depth),
        "min_samples_leaf": int(winner.min_samples_leaf),
        "max_features": mf,
    }


def _evaluate_candidates(configs, fold, X, V, y, target, seeds, jobs):
    rows = []
    late_rate = float(target.mean())
    for params in configs:
        for seed in seeds:
            start = perf_counter()
            err = ""
            try:
                model = make_forest(seed, params, jobs=jobs).fit(X, y)
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
            print(
                f"[rf f{fold}] {params} -> AP={ap:.5f} ({rows[-1]['seconds']:.1f}s)",
                flush=True,
            )
    return rows


def _write_tree_count_curve(output, tuning_df, best):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = tuning_df[
        (tuning_df["max_depth"] == best["max_depth"])
        & (tuning_df["min_samples_leaf"] == best["min_samples_leaf"])
        & (tuning_df["max_features"].astype(str) == str(best["max_features"]))
    ]
    curve = (
        sub.groupby("n_estimators", as_index=False)[["ap", "roc_auc"]]
        .mean()
        .sort_values("n_estimators")
    )
    curve.to_csv(output / "n_estimators_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(curve["n_estimators"], curve["ap"], marker="o")
    ax.axvline(best["n_estimators"], color="gray", linestyle="--", label=f"selected n={best['n_estimators']}")
    ax.set(
        xlabel="n_estimators",
        ylabel="Mean Folds 1-2 AP",
        title=f"Random Forest AP vs n_estimators (depth={best['max_depth']}, leaf={best['min_samples_leaf']})",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "n_estimators_curve.png", dpi=160)
    plt.close(fig)


def run(output, jobs=4):
    """Tune on folds 1-2, evaluate on folds 3-4, save feature importances and refit."""
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
        prep = model_pipeline("random_forest", make_forest(SEEDS[0], {}, jobs=jobs))["prep"]
        X = prep.fit_transform(train[FEATURES]).astype(np.float32)
        V = prep.transform(val[FEATURES]).astype(np.float32)
        y = train[TARGET].to_numpy()
        target = val[TARGET].to_numpy()
        fold_cache[fold] = (X, V, y, target)
        tuning_rows.extend(_evaluate_candidates(configs, fold, X, V, y, target, SEEDS[:1], jobs))

    tuning_df = pd.DataFrame(tuning_rows)
    best = _select_winner(tuning_df)
    print("Initial tree-structure winner:", best, flush=True)

    # Check structural boundaries plus n_estimators sweep on the winning structure.
    tested_extensions = []
    extra = []
    for param in ("max_depth", "min_samples_leaf"):
        low, high = BOUNDARIES[param]
        if best[param] == min(GRID[param]) and low is not None:
            extra.append({**best, param: low})
        if best[param] == max(GRID[param]) and high is not None:
            extra.append({**best, param: high})
    if best["max_features"] == 0.5 and BOUNDARIES["max_features"][1] is not None:
        extra.append({**best, "max_features": BOUNDARIES["max_features"][1]})

    seen_keys = {
        (int(r["n_estimators"]), int(r["max_depth"]), int(r["min_samples_leaf"]), str(r["max_features"]))
        for r in tuning_rows
    }
    if extra:
        filtered = [
            p
            for p in extra
            if (p["n_estimators"], p["max_depth"], p["min_samples_leaf"], str(p["max_features"]))
            not in seen_keys
        ]
        if filtered:
            tested_extensions.extend(filtered)
            for fold in (1, 2):
                X, V, y, target = fold_cache[fold]
                tuning_rows.extend(_evaluate_candidates(filtered, fold, X, V, y, target, SEEDS[:1], jobs))
            tuning_df = pd.DataFrame(tuning_rows)
            best = _select_winner(tuning_df)

    # Rebuild seen_keys after boundary checks so n_estimators=100 isn't re-run.
    seen_keys = {
        (int(r["n_estimators"]), int(r["max_depth"]), int(r["min_samples_leaf"]), str(r["max_features"]))
        for r in tuning_rows
    }
    # Sweep n_estimators on the winning tree structure (and check 300 if 200 wins).
    n_candidates = [
        {**best, "n_estimators": n}
        for n in N_ESTIMATORS_SWEEP
        if (n, best["max_depth"], best["min_samples_leaf"], str(best["max_features"])) not in seen_keys
    ]
    if n_candidates:
        tested_extensions.extend(n_candidates)
        for fold in (1, 2):
            X, V, y, target = fold_cache[fold]
            tuning_rows.extend(_evaluate_candidates(n_candidates, fold, X, V, y, target, SEEDS[:1], jobs))
        tuning_df = pd.DataFrame(tuning_rows)
        best = _select_winner(tuning_df)

    if best["n_estimators"] == max(N_ESTIMATORS_SWEEP):
        ext_n = {**best, "n_estimators": BOUNDARIES["n_estimators"][1]}
        tested_extensions.append(ext_n)
        for fold in (1, 2):
            X, V, y, target = fold_cache[fold]
            tuning_rows.extend(_evaluate_candidates([ext_n], fold, X, V, y, target, SEEDS[:1], jobs))
        tuning_df = pd.DataFrame(tuning_rows)
        best = _select_winner(tuning_df)

    fold_cache.clear()
    tuning_df.to_csv(output / "tuning.csv", index=False)
    _write_tree_count_curve(output, tuning_df, best)
    print("Final selected:", best, flush=True)

    eval_rows, pred_frames = [], []
    for fold, train, val in development_folds(dev):
        if fold <= 2:
            continue
        prep = model_pipeline("random_forest", make_forest(SEEDS[0], best, jobs=jobs))["prep"]
        X = prep.fit_transform(train[FEATURES]).astype(np.float32)
        V = prep.transform(val[FEATURES]).astype(np.float32)
        y = train[TARGET].to_numpy()
        target = val[TARGET].to_numpy()
        late_rate = float(target.mean())
        for seed in SEEDS:
            start = perf_counter()
            model = make_forest(seed, best, jobs=jobs).fit(X, y)
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
    final = model_pipeline("random_forest", make_forest(SEEDS[0], best, jobs=jobs)).fit(
        dev[FEATURES], dev[TARGET]
    )
    refit_seconds = round(perf_counter() - t_refit, 2)
    joblib.dump(final, output / "model.joblib", compress=3)

    save_feature_importances(
        output,
        final.named_steps["prep"],
        final.named_steps["model"].feature_importances_,
        col_name="importance",
        abs_group=False,
    )

    write_settings(
        output,
        model_key="random_forest",
        grid={**GRID, "n_estimators_sweep": N_ESTIMATORS_SWEEP},
        best_params=best,
        fold_means=fold_means,
        fold_stds=fold_stds,
        overall=overall,
        development_rows=len(dev),
        predictions_rows=len(predictions),
        refit_seconds=refit_seconds,
        boundary_extensions=tested_extensions,
        importance_method="RandomForestClassifier.feature_importances_ (mean decrease in Gini impurity across bootstrap trees)",
        extra={"bootstrap": True, "n_jobs": jobs},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/random_forest"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run(args.out, jobs=args.jobs)
