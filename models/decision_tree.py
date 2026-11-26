"""Tune a T-2h decision tree on folds 1-2, evaluate on 3-4 and save a development refit."""

import argparse
import json
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
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
GRID = {
    "max_depth": [8, 12, 16],
    "min_samples_leaf": [150, 500],
    "ccp_alpha": [0.0, 1e-5],
}


def make_tree(seed, params):
    """Build an unfitted DecisionTreeClassifier for the given seed and parameters."""
    return DecisionTreeClassifier(random_state=seed, **params)


def _feature_names_from_prep(prep):
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
    if encoded_name in FEATURES:
        return encoded_name
    for cat in ("Dest", "Origin", "Reporting_Airline", "wx_skyc1_T2h"):
        if encoded_name.startswith(f"{cat}_"):
            return cat
    return encoded_name


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run(output):
    """Tune on folds 1-2, evaluate on folds 3-4, and save artifacts."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dev = load_development()
    tuning, evaluation, pred_frames = [], [], []
    best = None
    for fold, train, val in development_folds(dev):
        # All candidates use the same fold-local matrix. Float32 is the tree's input dtype.
        prep = model_pipeline("decision_tree", make_tree(SEEDS[0], {}))["prep"]
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
                tree = make_tree(seed, params).fit(X, y)
                prob = tree.predict_proba(V)[:, 1].astype(np.float32)
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
                    pred_frames.append(
                        pd.DataFrame(
                            {
                                "row_id": val.index.to_numpy(),
                                "fold": np.int32(fold),
                                "seed": np.int32(seed),
                                TARGET: val[TARGET].to_numpy(dtype=np.int8),
                                "p_late": prob,
                            }
                        )
                    )
        print(
            f"Finished fold {fold}: {len(y):,} training, {len(target):,} validation.",
            flush=True,
        )
        if fold == 2:
            table = pd.DataFrame(tuning)
            table.to_csv(output / "tuning.csv", index=False)
            table.groupby("max_depth", as_index=False)[["ap", "roc_auc"]].mean().to_csv(
                output / "depth_curve.csv", index=False
            )
            table.groupby("ccp_alpha", as_index=False)[["ap", "roc_auc"]].mean().to_csv(
                output / "pruning_curve.csv", index=False
            )
            # Prefer higher AP, then ROC-AUC, then shallower depth in a tie.
            ranking = table.groupby(list(GRID), as_index=False)[
                ["ap", "roc_auc"]
            ].mean()
            winner = ranking.sort_values(
                ["ap", "roc_auc", "max_depth"], ascending=[False, False, True]
            ).iloc[0]
            best = {
                "max_depth": int(winner.max_depth),
                "min_samples_leaf": int(winner.min_samples_leaf),
                "ccp_alpha": float(winner.ccp_alpha),
            }
            print("Selected", best, flush=True)
        del X, V

    results = pd.DataFrame(evaluation)
    results.to_csv(output / "evaluation.csv", index=False)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)

    means = results.groupby("fold")[["ap", "roc_auc", "late_rate"]].mean()
    print(means.to_string(), flush=True)
    print("Mean across evaluation folds", means.mean().to_dict(), flush=True)

    final = model_pipeline("decision_tree", make_tree(SEEDS[0], best)).fit(
        dev[FEATURES], dev[TARGET]
    )
    joblib.dump(final, output / "model.joblib", compress=3)

    tree_est = final.named_steps["model"]
    feat_names = _feature_names_from_prep(final.named_steps["prep"])
    fi = pd.DataFrame(
        {
            "feature": feat_names,
            "raw_feature": [_raw_feature_name(f) for f in feat_names],
            "importance": tree_est.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    fi.to_csv(output / "feature_importances.csv", index=False)
    fi.groupby("raw_feature", as_index=False)["importance"].sum().sort_values(
        "importance", ascending=False
    ).to_csv(output / "feature_importances_grouped.csv", index=False)

    (output / "settings.json").write_text(
        json.dumps(
            {
                "model_key": "decision_tree",
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
                "params": best,
                "seeds": list(SEEDS),
                "refit_seed": SEEDS[0],
                "features": FEATURES,
                "development_rows": len(dev),
                "final_depth": int(tree_est.get_depth()),
                "final_n_leaves": int(tree_est.get_n_leaves()),
                "early_stopping": "none",
                "approved_exceptions": [],
                "importance_method": "DecisionTreeClassifier.feature_importances_ (normalized Gini impurity reduction)",
                "predictions_rows": int(len(predictions)),
                "mean_validation": means.mean().to_dict(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/decision_tree"))
    run(parser.parse_args().out)
