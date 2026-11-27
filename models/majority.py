"""Evaluate the T-2h majority-class (prior) baseline on folds 3-4 and save standardized artifacts."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

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
    make_preprocessor,
)


def make_majority():
    """Return an unfitted preprocessing + DummyClassifier(strategy='prior') pipeline."""
    return Pipeline(
        [
            ("prep", make_preprocessor("onehot")),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def run(output):
    """Evaluate prior baseline on folds 3-4, refit on all dev rows, and save artifacts."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dev = load_development()

    # Majority baseline needs no tuning; record empty tuning table for schema completeness.
    pd.DataFrame(
        columns=[
            "fold",
            "seed",
            "strategy",
            "train_rows",
            "validation_rows",
            "late_rate",
            "ap",
            "roc_auc",
            "lift",
            "seconds",
            "error",
        ]
    ).to_csv(output / "tuning.csv", index=False)

    eval_rows, pred_frames = [], []
    for fold, train, val in development_folds(dev):
        if fold <= 2:
            continue
        y = train[TARGET].to_numpy()
        target = val[TARGET].to_numpy()
        late_rate = float(target.mean())
        for seed in SEEDS:
            start = perf_counter()
            model = DummyClassifier(strategy="prior").fit(
                train[FEATURES].iloc[:, :1], y
            )
            prob = model.predict_proba(val[FEATURES].iloc[:, :1])[:, 1]
            ap = float(average_precision_score(target, prob))
            auc = float(roc_auc_score(target, prob))
            eval_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "strategy": "prior",
                    "train_rows": len(y),
                    "validation_rows": len(target),
                    "late_rate": late_rate,
                    "ap": ap,
                    "roc_auc": auc,
                    "lift": ap / late_rate,
                    "seconds": round(perf_counter() - start, 4),
                }
            )
            pred_frames.append(
                prediction_frame(val.index.to_numpy(), fold, seed, target, prob)
            )
        print(
            f"Finished evaluation fold {fold}: {len(y):,} train, {len(target):,} val.",
            flush=True,
        )

    _, predictions, fold_means, fold_stds, overall = finalize_eval_outputs(
        output, eval_rows, pred_frames
    )

    t_refit = perf_counter()
    final = make_majority().fit(dev[FEATURES], dev[TARGET])
    refit_seconds = round(perf_counter() - t_refit, 2)
    joblib.dump(final, output / "model.joblib", compress=3)

    write_settings(
        output,
        model_key="majority",
        grid={},
        best_params={"strategy": "prior"},
        fold_means=fold_means,
        fold_stds=fold_stds,
        overall=overall,
        development_rows=len(dev),
        predictions_rows=len(predictions),
        refit_seconds=refit_seconds,
        early_stopping="none (majority-class prior baseline requires no tuning)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/majority"))
    run(parser.parse_args().out)
