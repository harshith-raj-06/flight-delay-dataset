"""Tune AdaBoost on folds 1-2 using preprocessing.py, evaluate on 3-4, and save artifacts."""

import argparse
import gc
import json
import multiprocessing as mp
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import (
    FEATURES,
    TARGET,
    development_folds,
    load_development,
    model_pipeline,
)

SEEDS = (11, 23, 42)
N_ESTIMATORS_CHECKPOINTS = [20, 40, 60, 80, 100]
BASE_GRID = {
    "max_depth": [2, 3],
    "min_samples_leaf": [200, 500],
    "learning_rate": [0.05, 0.15, 0.4],
}
BOUNDARIES = {
    "max_depth": (1, 4),
    "min_samples_leaf": (50, 1000),
    "learning_rate": (0.02, 0.8),
    "n_estimators": (None, 175),
}
EXTENDED_CHECKPOINTS = [20, 40, 60, 80, 100, 125, 150, 175]


def make_adaboost(seed, params):
    """Build an unfitted AdaBoostClassifier with DecisionTreeClassifier base estimator."""
    base = DecisionTreeClassifier(
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=seed,
    )
    return AdaBoostClassifier(
        estimator=base,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        random_state=seed,
    )


def _run_staged_worker(task):
    """Fit one AdaBoost ensemble up to max(checkpoints) and score each checkpoint."""
    base_params, checkpoints, fold, seed, cache_dir = task
    cache = Path(cache_dir)
    X = np.load(cache / f"X_{fold}.npy", mmap_mode="r")
    V = np.load(cache / f"V_{fold}.npy", mmap_mode="r")
    y = np.load(cache / f"y_{fold}.npy", mmap_mode="r")
    target = np.load(cache / f"t_{fold}.npy", mmap_mode="r")

    train_rows = len(y)
    val_rows = len(target)
    late_rate = float(target.mean())
    max_n = max(checkpoints)
    ckpt_set = set(checkpoints)

    rows = []
    start = perf_counter()
    try:
        model = make_adaboost(seed, {**base_params, "n_estimators": max_n})
        model.fit(X, y)
        fit_elapsed = perf_counter() - start
        actual_iters = len(model.estimators_)

        pred_start = perf_counter()
        staged_probs = {}
        if actual_iters > 0:
            for step, proba in enumerate(model.staged_predict_proba(V), start=1):
                if step in ckpt_set:
                    staged_probs[step] = proba[:, 1].copy()
        pred_elapsed = perf_counter() - pred_start

        for k in checkpoints:
            used_k = min(k, actual_iters)
            if used_k == 0:
                rows.append(
                    {
                        **base_params,
                        "n_estimators": k,
                        "fold": fold,
                        "seed": seed,
                        "train_rows": train_rows,
                        "validation_rows": val_rows,
                        "late_rate": late_rate,
                        "ap": np.nan,
                        "roc_auc": np.nan,
                        "seconds": round(fit_elapsed + pred_elapsed, 4),
                        "error": "AdaBoost stopped with 0 estimators",
                    }
                )
                continue
            prob = staged_probs[used_k]
            ap = float(average_precision_score(target, prob))
            auc = float(roc_auc_score(target, prob))
            est_seconds = fit_elapsed * (used_k / max(actual_iters, 1)) + pred_elapsed / len(checkpoints)
            rows.append(
                {
                    **base_params,
                    "n_estimators": k,
                    "fold": fold,
                    "seed": seed,
                    "train_rows": train_rows,
                    "validation_rows": val_rows,
                    "late_rate": late_rate,
                    "ap": ap,
                    "roc_auc": auc,
                    "seconds": round(est_seconds, 4),
                    "error": "" if actual_iters >= k else f"stopped_early_at_{actual_iters}",
                }
            )
    except Exception as exc:  # noqa: BLE001
        elapsed = perf_counter() - start
        for k in checkpoints:
            rows.append(
                {
                    **base_params,
                    "n_estimators": k,
                    "fold": fold,
                    "seed": seed,
                    "train_rows": train_rows,
                    "validation_rows": val_rows,
                    "late_rate": late_rate,
                    "ap": np.nan,
                    "roc_auc": np.nan,
                    "seconds": round(elapsed, 4),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return base_params, fold, perf_counter() - start, rows


def _run_eval_worker(task):
    """Fit and score one (fold, seed) evaluation run using locked parameters."""
    fold, seed, params, cache_dir, out_dir = task
    cache = Path(cache_dir)
    X = np.load(cache / f"X_{fold}.npy", mmap_mode="r")
    V = np.load(cache / f"V_{fold}.npy", mmap_mode="r")
    y = np.load(cache / f"y_{fold}.npy", mmap_mode="r")
    target = np.load(cache / f"t_{fold}.npy", mmap_mode="r")
    row_ids = np.load(cache / f"id_{fold}.npy", mmap_mode="r")

    start = perf_counter()
    model = make_adaboost(seed, params).fit(X, y)
    prob = model.predict_proba(V)[:, 1].astype(np.float32)
    elapsed = perf_counter() - start
    ap = float(average_precision_score(target, prob))
    auc = float(roc_auc_score(target, prob))
    late_rate = float(target.mean())

    pd.DataFrame(
        {
            "row_id": np.asarray(row_ids),
            "fold": np.int32(fold),
            "seed": np.int32(seed),
            TARGET: np.asarray(target, dtype=np.int8),
            "p_late": prob,
        }
    ).to_parquet(Path(out_dir) / f"_pred_{fold}_{seed}.parquet", index=False)

    return {
        "fold": fold,
        "seed": seed,
        "train_rows": len(y),
        "validation_rows": len(target),
        "late_rate": late_rate,
        "ap": ap,
        "roc_auc": auc,
        "lift": ap / late_rate,
        "seconds": round(elapsed, 4),
    }


def _base_configs_from_grid():
    configs = []
    for d in BASE_GRID["max_depth"]:
        for leaf in BASE_GRID["min_samples_leaf"]:
            for lr in BASE_GRID["learning_rate"]:
                configs.append({"max_depth": d, "min_samples_leaf": leaf, "learning_rate": lr})
    return configs


def _select_winner(tuning_df):
    valid = tuning_df.dropna(subset=["ap", "roc_auc"]).copy()
    group_cols = ["max_depth", "min_samples_leaf", "learning_rate", "n_estimators"]
    ranking = valid.groupby(group_cols, as_index=False)[["ap", "roc_auc"]].mean()
    winner = ranking.sort_values(
        ["ap", "roc_auc", "n_estimators", "max_depth"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {
        "max_depth": int(winner.max_depth),
        "min_samples_leaf": int(winner.min_samples_leaf),
        "learning_rate": float(winner.learning_rate),
        "n_estimators": int(winner.n_estimators),
    }


def _boundary_extensions(best):
    """Return extra (base_params, checkpoints) tasks if winner hits a grid boundary."""
    extensions = []
    base = {k: best[k] for k in ("max_depth", "min_samples_leaf", "learning_rate")}
    for param in ("max_depth", "min_samples_leaf", "learning_rate"):
        vals = BASE_GRID[param]
        low, high = BOUNDARIES[param]
        if best[param] == min(vals) and low is not None:
            extensions.append(({**base, param: low}, N_ESTIMATORS_CHECKPOINTS))
        if best[param] == max(vals) and high is not None:
            extensions.append(({**base, param: high}, N_ESTIMATORS_CHECKPOINTS))
    if best["n_estimators"] == max(N_ESTIMATORS_CHECKPOINTS) and BOUNDARIES["n_estimators"][1]:
        extensions.append((base, EXTENDED_CHECKPOINTS))
    return extensions


def _feature_names_from_prep(prep):
    """Extract ordered output feature names from preprocessing.py's ColumnTransformer."""
    names = []
    for name, trans, cols in prep.transformers_:
        if name == "destination":
            names.extend(trans.named_steps["encode"].get_feature_names_out(["Dest"]).tolist())
        elif name == "categorical":
            names.extend(trans.named_steps["encode"].get_feature_names_out(cols).tolist())
        elif name == "numeric":
            names.extend(list(cols))
    return names


def _raw_feature_name(encoded_name):
    """Map an encoded feature name back to one of the 35 raw FEATURES."""
    if encoded_name in FEATURES:
        return encoded_name
    for cat in ("Dest", "Origin", "Reporting_Airline", "wx_skyc1_T2h"):
        if encoded_name.startswith(f"{cat}_"):
            return cat
    return encoded_name


def _write_boosting_analysis(output, tuning_df, best):
    """Save iteration curve CSV + PNG from tuning folds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = tuning_df[
        (tuning_df["max_depth"] == best["max_depth"])
        & (tuning_df["min_samples_leaf"] == best["min_samples_leaf"])
    ]
    curve = (
        sub.groupby(["learning_rate", "n_estimators"], as_index=False)[["ap", "roc_auc"]]
        .mean()
        .sort_values(["learning_rate", "n_estimators"])
    )
    curve.to_csv(output / "iteration_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr, grp in curve.groupby("learning_rate"):
        ax.plot(grp["n_estimators"], grp["ap"], marker="o", label=f"lr={lr}")
    ax.axvline(best["n_estimators"], color="gray", linestyle="--", label=f"selected n={best['n_estimators']}")
    ax.set(
        xlabel="n_estimators",
        ylabel="Mean Folds 1-2 AP",
        title=f"AdaBoost tuning AP vs n_estimators (depth={best['max_depth']}, leaf={best['min_samples_leaf']})",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "iteration_curve.png", dpi=160)
    plt.close(fig)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _prepare_fold_cache(cache, sample_params):
    """Prepare and mmap-cache all 4 folds once, then free raw DataFrame memory."""
    dev = load_development()
    for fold, train, val in development_folds(dev):
        if (cache / f"X_{fold}.npy").exists():
            continue
        prep = model_pipeline("adaboost", make_adaboost(SEEDS[0], sample_params))["prep"]
        X = prep.fit_transform(train[FEATURES]).astype(np.float32)
        V = prep.transform(val[FEATURES]).astype(np.float32)
        np.save(cache / f"X_{fold}.npy", X)
        np.save(cache / f"V_{fold}.npy", V)
        np.save(cache / f"y_{fold}.npy", train[TARGET].to_numpy(dtype=np.int8))
        np.save(cache / f"t_{fold}.npy", val[TARGET].to_numpy(dtype=np.int8))
        np.save(cache / f"id_{fold}.npy", val.index.to_numpy())
        print(f"Cached fold {fold}: {len(train):,} train, {len(val):,} val", flush=True)
        del X, V, prep
    dev_rows = len(dev)
    del dev
    gc.collect()
    return dev_rows


def _seen_keys(tuning_rows):
    return {
        (
            int(r["fold"]),
            int(r["max_depth"]),
            int(r["min_samples_leaf"]),
            float(r["learning_rate"]),
            int(r["n_estimators"]),
        )
        for r in tuning_rows
    }


def run(output, workers=2, resume=False):
    """Execute full AdaBoost tuning, evaluation, refit, and analysis."""
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory or pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    base_configs = _base_configs_from_grid()
    sample_params = {**base_configs[0], "n_estimators": 10}
    dev_rows = _prepare_fold_cache(cache, sample_params)

    tuning_path = output / "tuning.csv"
    tuning_rows = pd.read_csv(tuning_path).to_dict("records") if (resume and tuning_path.exists()) else []
    seen = _seen_keys(tuning_rows)

    # Tune on folds 1-2.
    tasks = []
    for fold in (1, 2):
        for cfg in base_configs:
            missing = [
                k
                for k in N_ESTIMATORS_CHECKPOINTS
                if (fold, int(cfg["max_depth"]), int(cfg["min_samples_leaf"]), float(cfg["learning_rate"]), int(k))
                not in seen
            ]
            if missing:
                tasks.append((cfg, N_ESTIMATORS_CHECKPOINTS, fold, SEEDS[0], str(cache)))

    if tasks:
        t0 = perf_counter()
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            for base_p, f_id, dt, batch in pool.imap_unordered(_run_staged_worker, tasks):
                for row in batch:
                    key = (f_id, int(row["max_depth"]), int(row["min_samples_leaf"]), float(row["learning_rate"]), int(row["n_estimators"]))
                    if key not in seen:
                        tuning_rows.append(row)
                        seen.add(key)
                pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
                best_row = max(batch, key=lambda r: r["ap"] if not np.isnan(r["ap"]) else -1.0)
                print(
                    f"[tune f{f_id}] {base_p} -> best n={best_row['n_estimators']} AP={best_row['ap']:.5f} ({dt:.1f}s)",
                    flush=True,
                )
        print(f"Finished base tuning batch in {perf_counter() - t0:.1f}s", flush=True)

    tuning_df = pd.read_csv(tuning_path)
    best = _select_winner(tuning_df)
    print("Initial tuning winner:", best, flush=True)

    # Check boundary extensions on folds 1-2 if winner hits a grid boundary.
    extensions = _boundary_extensions(best)
    tested_extensions = []
    if extensions:
        ext_tasks = []
        for base_p, ckpts in extensions:
            for fold in (1, 2):
                missing = [
                    k
                    for k in ckpts
                    if (fold, int(base_p["max_depth"]), int(base_p["min_samples_leaf"]), float(base_p["learning_rate"]), int(k))
                    not in seen
                ]
                if missing:
                    ext_tasks.append((base_p, missing, fold, SEEDS[0], str(cache)))
            tested_extensions.append({"base": base_p, "checkpoints": ckpts})

        if ext_tasks:
            print(f"Testing {len(ext_tasks)} boundary extension tasks...", flush=True)
            with ctx.Pool(processes=min(workers, len(ext_tasks))) as pool:
                for base_p, f_id, dt, batch in pool.imap_unordered(_run_staged_worker, ext_tasks):
                    for row in batch:
                        key = (f_id, int(row["max_depth"]), int(row["min_samples_leaf"]), float(row["learning_rate"]), int(row["n_estimators"]))
                        if key not in seen:
                            tuning_rows.append(row)
                            seen.add(key)
                    pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
                    best_row = max(batch, key=lambda r: r["ap"] if not np.isnan(r["ap"]) else -1.0)
                    print(
                        f"[ext f{f_id}] {base_p} -> best n={best_row['n_estimators']} AP={best_row['ap']:.5f} ({dt:.1f}s)",
                        flush=True,
                    )
        tuning_df = pd.read_csv(tuning_path)
        best = _select_winner(tuning_df)
        print("Final selected after boundary check:", best, flush=True)

    # Evaluate locked parameters on folds 3-4 across seeds 11, 23, 42.
    eval_path = output / "evaluation.csv"
    eval_rows = pd.read_csv(eval_path).to_dict("records") if (resume and eval_path.exists()) else []
    done_eval = {(int(r["fold"]), int(r["seed"])) for r in eval_rows}
    eval_tasks = [
        (fold, seed, best, str(cache), str(output))
        for fold in (3, 4)
        for seed in SEEDS
        if (fold, seed) not in done_eval or not (output / f"_pred_{fold}_{seed}.parquet").exists()
    ]
    if eval_tasks:
        with ctx.Pool(processes=min(workers, len(eval_tasks))) as pool:
            for row in pool.imap_unordered(_run_eval_worker, eval_tasks):
                eval_rows = [r for r in eval_rows if (int(r["fold"]), int(r["seed"])) != (int(row["fold"]), int(row["seed"]))]
                eval_rows.append(row)
                pd.DataFrame(eval_rows).sort_values(["fold", "seed"]).to_csv(eval_path, index=False)
                print(
                    f"[eval f{row['fold']} seed={row['seed']}] AP={row['ap']:.5f} AUC={row['roc_auc']:.5f} ({row['seconds']:.1f}s)",
                    flush=True,
                )

    eval_df = pd.read_csv(eval_path).sort_values(["fold", "seed"]).reset_index(drop=True)
    eval_df.to_csv(eval_path, index=False)

    pred_files = [output / f"_pred_{fold}_{seed}.parquet" for fold in (3, 4) for seed in SEEDS]
    predictions = pd.concat([pd.read_parquet(p) for p in pred_files], ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    for p in pred_files:
        p.unlink(missing_ok=True)
    for f in cache.iterdir():
        f.unlink(missing_ok=True)
    cache.rmdir()

    means = eval_df.groupby("fold")[["ap", "roc_auc", "lift", "late_rate"]].mean()
    overall = means.mean()
    print(means.to_string(), flush=True)
    print("Mean across folds 3-4:", overall.to_dict(), flush=True)

    # Refit complete preprocessing + AdaBoost pipeline on all development rows with seed 11.
    dev = load_development()
    t_refit = perf_counter()
    final_pipe = model_pipeline("adaboost", make_adaboost(SEEDS[0], best)).fit(
        dev[FEATURES], dev[TARGET]
    )
    refit_seconds = round(perf_counter() - t_refit, 2)
    joblib.dump(final_pipe, output / "model.joblib", compress=3)
    print(f"Finished full dev refit in {refit_seconds:.1f}s", flush=True)

    # Native feature importances from AdaBoostClassifier (weighted mean decrease in impurity).
    feat_names = _feature_names_from_prep(final_pipe.named_steps["prep"])
    importances = final_pipe.named_steps["model"].feature_importances_
    fi_df = (
        pd.DataFrame(
            {
                "feature": feat_names,
                "raw_feature": [_raw_feature_name(f) for f in feat_names],
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df.to_csv(output / "feature_importances.csv", index=False)
    fi_grouped = (
        fi_df.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_grouped.to_csv(output / "feature_importances_grouped.csv", index=False)

    _write_boosting_analysis(output, tuning_df, best)

    settings = {
        "model_key": "adaboost",
        "code_commit": _git_commit(),
        "run_command": " ".join(sys.argv),
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "features": FEATURES,
        "grid": {**BASE_GRID, "n_estimators": N_ESTIMATORS_CHECKPOINTS},
        "boundary_extensions_tested": tested_extensions,
        "best_params": best,
        "selected_n_estimators": best["n_estimators"],
        "seeds": list(SEEDS),
        "tuning_seed": SEEDS[0],
        "refit_seed": SEEDS[0],
        "refit_seconds": refit_seconds,
        "development_rows": dev_rows,
        "early_stopping": "none (staged_predict_proba evaluated exact iteration checkpoints on Folds 1-2 using 100% of training rows)",
        "approved_exceptions": [],
        "importance_method": "AdaBoostClassifier.feature_importances_ (normalized weighted Gini impurity reduction across DecisionTreeClassifier base estimators)",
        "fold_summary": {
            "fold_3": means.loc[3].to_dict(),
            "fold_4": means.loc[4].to_dict(),
            "mean_folds_3_4": overall.to_dict(),
        },
        "predictions_rows": int(len(predictions)),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/adaboost"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.out, workers=args.workers, resume=args.resume)
