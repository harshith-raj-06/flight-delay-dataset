"""Tune classic sklearn GradientBoostingClassifier on folds 1-2, evaluate on 3-4, and save artifacts."""

import argparse
import gc
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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

N_ESTIMATORS_CHECKPOINTS = [100, 150, 200, 300]
BASE_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.06, 0.12],
    "subsample": [0.6, 0.8],
    "max_features": [0.4],
}
MIN_SAMPLES_LEAF = 150
BOUNDARIES = {
    "max_depth": (2, 6),
    "learning_rate": (0.03, 0.2),
    "subsample": (0.5, 1.0),
    "n_estimators": (None, 400),
}
EXTENDED_CHECKPOINTS = [100, 150, 200, 300, 350, 400]


def make_gb(seed, params):
    """Build an unfitted classic sklearn GradientBoostingClassifier."""
    return GradientBoostingClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        max_features=float(params["max_features"]),
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=seed,
    )


def _run_staged_worker(task):
    """Fit one GradientBoostingClassifier up to max(checkpoints) and score each checkpoint."""
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
        model = make_gb(seed, {**base_params, "n_estimators": max_n})
        model.fit(X, y)
        fit_elapsed = perf_counter() - start
        actual_iters = len(model.estimators_)

        pred_start = perf_counter()
        staged_probs = {}
        for step, proba in enumerate(model.staged_predict_proba(V), start=1):
            if step in ckpt_set or step == actual_iters:
                staged_probs[step] = proba[:, 1].copy()
        pred_elapsed = perf_counter() - pred_start

        for k in checkpoints:
            used_k = min(k, actual_iters)
            prob = staged_probs[used_k]
            ap = float(average_precision_score(target, prob))
            auc = float(roc_auc_score(target, prob))
            est_seconds = fit_elapsed * (used_k / max(actual_iters, 1)) + pred_elapsed / len(checkpoints)
            rows.append(
                {
                    **base_params,
                    "min_samples_leaf": MIN_SAMPLES_LEAF,
                    "n_estimators": k,
                    "fold": fold,
                    "seed": seed,
                    "train_rows": train_rows,
                    "validation_rows": val_rows,
                    "late_rate": late_rate,
                    "ap": ap,
                    "roc_auc": auc,
                    "lift": ap / late_rate,
                    "seconds": round(est_seconds, 4),
                    "error": "" if actual_iters >= k else f"stopped_early_at_{actual_iters}",
                }
            )
    except Exception as exc:  # noqa: BLE001
        elapsed = perf_counter() - start
        for k in checkpoints[len(rows):]:
            rows.append(
                {
                    **base_params,
                    "min_samples_leaf": MIN_SAMPLES_LEAF,
                    "n_estimators": k,
                    "fold": fold,
                    "seed": seed,
                    "train_rows": train_rows,
                    "validation_rows": val_rows,
                    "late_rate": late_rate,
                    "ap": np.nan,
                    "roc_auc": np.nan,
                    "lift": np.nan,
                    "seconds": round(elapsed, 4),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return base_params, fold, perf_counter() - start, rows


def _run_eval_worker(task):
    """Fit and score one (fold, seed) evaluation run with locked settings."""
    fold, seed, params, cache_dir, out_dir = task
    cache = Path(cache_dir)
    X = np.load(cache / f"X_{fold}.npy", mmap_mode="r")
    V = np.load(cache / f"V_{fold}.npy", mmap_mode="r")
    y = np.load(cache / f"y_{fold}.npy", mmap_mode="r")
    target = np.load(cache / f"t_{fold}.npy", mmap_mode="r")
    row_ids = np.load(cache / f"id_{fold}.npy", mmap_mode="r")

    start = perf_counter()
    model = make_gb(seed, params).fit(X, y)
    prob = model.predict_proba(V)[:, 1].astype(np.float32)
    elapsed = perf_counter() - start
    ap = float(average_precision_score(target, prob))
    auc = float(roc_auc_score(target, prob))
    late_rate = float(target.mean())

    prediction_frame(row_ids, fold, seed, target, prob).to_parquet(
        Path(out_dir) / f"_pred_{fold}_{seed}.parquet", index=False
    )

    return {
        "fold": fold,
        "seed": seed,
        **params,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
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
        for lr in BASE_GRID["learning_rate"]:
            for sub in BASE_GRID["subsample"]:
                for mf in BASE_GRID["max_features"]:
                    configs.append(
                        {
                            "max_depth": d,
                            "learning_rate": lr,
                            "subsample": sub,
                            "max_features": mf,
                        }
                    )
    return configs


def _valid_scores(df):
    return df.loc[np.isfinite(df[["ap", "roc_auc"]]).all(axis=1)]


def _select_winner(tuning_df):
    valid = _valid_scores(tuning_df).copy()
    valid = valid[valid.fold.isin((1, 2))]
    group_cols = ["max_depth", "learning_rate", "subsample", "max_features", "n_estimators"]
    valid = valid.drop_duplicates(group_cols + ["fold"], keep="last")
    complete = valid.groupby(group_cols)["fold"].transform("count").eq(2)
    ranking = valid[complete].groupby(group_cols, as_index=False)[["ap", "roc_auc"]].mean()
    winner = ranking.sort_values(
        ["ap", "roc_auc", "n_estimators", "max_depth"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {
        "max_depth": int(winner.max_depth),
        "learning_rate": float(winner.learning_rate),
        "subsample": float(winner.subsample),
        "max_features": float(winner.max_features),
        "n_estimators": int(winner.n_estimators),
    }


def _boundary_extensions(best):
    """Check max_features neighbors plus any boundary coordinates of the initial winner."""
    extensions = []
    base = {k: best[k] for k in ("max_depth", "learning_rate", "subsample", "max_features")}
    # Check max_features neighbors 0.3 and 0.5 on the winning structure.
    for mf in (0.3, 0.5):
        if abs(base["max_features"] - mf) > 1e-9:
            extensions.append(({**base, "max_features": mf}, N_ESTIMATORS_CHECKPOINTS))
    for param in ("max_depth", "learning_rate", "subsample"):
        vals = BASE_GRID[param]
        low, high = BOUNDARIES[param]
        if best[param] == min(vals) and low is not None:
            extensions.append(({**base, param: low}, N_ESTIMATORS_CHECKPOINTS))
        if best[param] == max(vals) and high is not None:
            extensions.append(({**base, param: high}, N_ESTIMATORS_CHECKPOINTS))
    if best["n_estimators"] == max(N_ESTIMATORS_CHECKPOINTS) and BOUNDARIES["n_estimators"][1]:
        extensions.append((base, EXTENDED_CHECKPOINTS))
    return extensions


def _write_boosting_analysis(output, tuning_df, best):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = tuning_df[
        (tuning_df["max_depth"] == best["max_depth"])
        & (tuning_df["subsample"] == best["subsample"])
        & (tuning_df["max_features"] == best["max_features"])
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
        title=f"GradientBoosting tuning AP vs n_estimators (depth={best['max_depth']}, sub={best['subsample']})",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "iteration_curve.png", dpi=160)
    plt.close(fig)


def _prepare_fold_cache(cache, sample_params):
    dev = load_development()
    for fold, train, val in development_folds(dev):
        marker = cache / f"complete_{fold}"
        files = [cache / f"{k}_{fold}.npy" for k in ("X", "V", "y", "t", "id")]
        if marker.exists() and all(p.is_file() for p in files):
            continue
        marker.unlink(missing_ok=True)
        prep = model_pipeline("gradient_boosting", make_gb(SEEDS[0], sample_params))["prep"]
        X = prep.fit_transform(train[FEATURES]).astype(np.float32)
        V = prep.transform(val[FEATURES]).astype(np.float32)
        np.save(cache / f"X_{fold}.npy", X)
        np.save(cache / f"V_{fold}.npy", V)
        np.save(cache / f"y_{fold}.npy", train[TARGET].to_numpy(dtype=np.int8))
        np.save(cache / f"t_{fold}.npy", val[TARGET].to_numpy(dtype=np.int8))
        np.save(cache / f"id_{fold}.npy", val.index.to_numpy())
        marker.touch()
        print(f"Cached fold {fold}: {len(train):,} train, {len(val):,} val", flush=True)
        del X, V, prep
    dev_rows = len(dev)
    del dev
    gc.collect()
    return dev_rows


def _seen_keys(rows):
    return {
        (
            int(r["fold"]),
            int(r["max_depth"]),
            float(r["learning_rate"]),
            float(r["subsample"]),
            float(r["max_features"]),
            int(r["n_estimators"]),
        )
        for r in rows
    }


def run(output, workers=4, resume=False):
    """Execute full stochastic GradientBoosting tuning, evaluation, refit, and analysis."""
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
    tuning_rows = _valid_scores(pd.read_csv(tuning_path)).to_dict("records") if (resume and tuning_path.exists()) else []
    seen = _seen_keys(tuning_rows)

    tasks = []
    for fold in (1, 2):
        for cfg in base_configs:
            missing = [
                k
                for k in N_ESTIMATORS_CHECKPOINTS
                if (fold, int(cfg["max_depth"]), float(cfg["learning_rate"]), float(cfg["subsample"]), float(cfg["max_features"]), int(k))
                not in seen
            ]
            if missing:
                tasks.append((cfg, missing, fold, SEEDS[0], str(cache)))

    if tasks:
        t0 = perf_counter()
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            for base_p, f_id, dt, batch in pool.imap_unordered(_run_staged_worker, tasks):
                for row in batch:
                    key = (f_id, int(row["max_depth"]), float(row["learning_rate"]), float(row["subsample"]), float(row["max_features"]), int(row["n_estimators"]))
                    if key not in seen:
                        tuning_rows.append(row)
                        seen.add(key)
                pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
                best_row = max(batch, key=lambda r: r["ap"] if not np.isnan(r["ap"]) else -1.0)
                print(
                    f"[gb tune f{f_id}] {base_p} -> best n={best_row['n_estimators']} AP={best_row['ap']:.5f} ({dt:.1f}s)",
                    flush=True,
                )
        print(f"Finished base tuning batch in {perf_counter() - t0:.1f}s", flush=True)

    tuning_df = pd.read_csv(tuning_path)
    best = _select_winner(tuning_df)
    print("Initial tuning winner:", best, flush=True)

    extensions = _boundary_extensions(best)
    tested_extensions = []
    if extensions:
        ext_tasks = []
        for base_p, ckpts in extensions:
            for fold in (1, 2):
                missing = [
                    k
                    for k in ckpts
                    if (fold, int(base_p["max_depth"]), float(base_p["learning_rate"]), float(base_p["subsample"]), float(base_p["max_features"]), int(k))
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
                        key = (f_id, int(row["max_depth"]), float(row["learning_rate"]), float(row["subsample"]), float(row["max_features"]), int(row["n_estimators"]))
                        if key not in seen:
                            tuning_rows.append(row)
                            seen.add(key)
                    pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
                    best_row = max(batch, key=lambda r: r["ap"] if not np.isnan(r["ap"]) else -1.0)
                    print(
                        f"[gb ext f{f_id}] {base_p} -> best n={best_row['n_estimators']} AP={best_row['ap']:.5f} ({dt:.1f}s)",
                        flush=True,
                    )
        tuning_df = pd.read_csv(tuning_path)
        best = _select_winner(tuning_df)
        print("Final selected after boundary check:", best, flush=True)

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
                eval_rows.append(row)
                pd.DataFrame(eval_rows).sort_values(["fold", "seed"]).to_csv(eval_path, index=False)
                print(
                    f"[gb eval f{row['fold']} seed={row['seed']}] AP={row['ap']:.5f} AUC={row['roc_auc']:.5f} ({row['seconds']:.1f}s)",
                    flush=True,
                )

    pred_files = [output / f"_pred_{fold}_{seed}.parquet" for fold in (3, 4) for seed in SEEDS]
    pred_frames = [pd.read_parquet(p) for p in pred_files]
    _, predictions, fold_means, fold_stds, overall = finalize_eval_outputs(
        output, eval_rows, pred_frames
    )
    for p in pred_files:
        p.unlink(missing_ok=True)
    for f in cache.iterdir():
        f.unlink(missing_ok=True)
    cache.rmdir()

    dev = load_development()
    t_refit = perf_counter()
    final_pipe = model_pipeline("gradient_boosting", make_gb(SEEDS[0], best)).fit(
        dev[FEATURES], dev[TARGET]
    )
    refit_seconds = round(perf_counter() - t_refit, 2)
    joblib.dump(final_pipe, output / "model.joblib", compress=3)
    print(f"Finished full dev refit in {refit_seconds:.1f}s", flush=True)

    save_feature_importances(
        output,
        final_pipe.named_steps["prep"],
        final_pipe.named_steps["model"].feature_importances_,
        col_name="importance",
        abs_group=False,
    )
    _write_boosting_analysis(output, tuning_df, best)

    write_settings(
        output,
        model_key="gradient_boosting",
        grid={**BASE_GRID, "n_estimators": N_ESTIMATORS_CHECKPOINTS},
        best_params=best,
        fold_means=fold_means,
        fold_stds=fold_stds,
        overall=overall,
        development_rows=dev_rows,
        predictions_rows=len(predictions),
        refit_seconds=refit_seconds,
        boundary_extensions=tested_extensions,
        early_stopping="none (staged_predict_proba evaluated exact iteration checkpoints on Folds 1-2 using 100% of training rows)",
        importance_method="GradientBoostingClassifier.feature_importances_ (normalized mean decrease in Friedman MSE impurity across trees)",
        extra={
            "selected_n_estimators": best["n_estimators"],
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "stochastic_gradient_boosting": True,
            "workers": workers,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/gradient_boosting"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.out, workers=args.workers, resume=args.resume)
