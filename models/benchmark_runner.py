"""Shared implementation for Person 3's sealed T-2h model benchmarks."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

# `python models/train_*.py` starts with models/ on sys.path, so add the
# repository root explicitly before importing the shared teammate contract.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from preprocessing import FEATURES, TARGET, development_folds, load_development, model_pipeline

TUNING_SEED = 11
EVALUATION_SEEDS = (11, 23, 42)
EXPECTED_ROWS = {3: 49_882, 4: 58_899}

EstimatorFactory = Callable[[int, dict[str, Any]], BaseEstimator]


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions(names: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _probabilities(pipe, features: pd.DataFrame) -> np.ndarray:
    probabilities = pipe.predict_proba(features)
    return probabilities[:, 1]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _importance_table(final_pipeline, fallback_names: list[str]) -> pd.DataFrame | None:
    model = final_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    values = np.asarray(model.feature_importances_, dtype=float)
    preprocessor = final_pipeline.named_steps["prep"]
    names = fallback_names
    if hasattr(preprocessor, "get_feature_names_out"):
        names = list(preprocessor.get_feature_names_out())
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    return pd.DataFrame(
        {
            "feature": names,
            "importance": values,
            "method": "estimator.feature_importances_",
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def run_benchmark(
    *,
    model_key: str,
    model_name: str,
    make_estimator: EstimatorFactory,
    grid: dict[str, list[Any]],
    iteration_parameter: str,
    output: Path,
    package_names: list[str],
    notes: str,
) -> None:
    """Tune folds 1-2, lock the winner, evaluate folds 3-4, and save artifacts."""
    candidates = list(ParameterGrid(grid))
    if len(candidates) < 25:
        raise ValueError("The protocol requires at least 25 distinct tuning configurations.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    development = load_development()
    folds = list(development_folds(development))
    tuning_rows: list[dict[str, Any]] = []

    # Tuning is deliberately restricted to the first two chronological folds.
    for fold, train, validation in folds[:2]:
        for config_id, params in enumerate(candidates, start=1):
            started = perf_counter()
            row: dict[str, Any] = {
                "config_id": config_id,
                "fold": fold,
                "seed": TUNING_SEED,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "late_rate": float(validation[TARGET].mean()),
                **params,
            }
            try:
                pipeline = model_pipeline(
                    model_key, make_estimator(TUNING_SEED, params)
                )
                pipeline.fit(train[FEATURES], train[TARGET])
                probability = _probabilities(pipeline, validation[FEATURES])
                row.update(
                    status="ok",
                    ap=float(average_precision_score(validation[TARGET], probability)),
                    roc_auc=float(roc_auc_score(validation[TARGET], probability)),
                    error="",
                )
            except Exception as exc:  # Failures are protocol-required tuning records.
                row.update(status="error", ap=np.nan, roc_auc=np.nan, error=repr(exc))
            row["runtime_seconds"] = perf_counter() - started
            tuning_rows.append(row)
            _write_csv(tuning_rows, output / "tuning.csv")
            print(
                f"tuning fold={fold} config={config_id}/{len(candidates)} "
                f"status={row['status']}",
                flush=True,
            )

    tuning = pd.DataFrame(tuning_rows)
    successful = tuning.loc[tuning.status.eq("ok")].copy()
    completed = successful.groupby("config_id").size()
    complete_ids = completed[completed.eq(2)].index
    successful = successful[successful.config_id.isin(complete_ids)]
    if successful.empty:
        raise RuntimeError("No tuning configuration completed successfully on both folds.")
    ranking = (
        successful.groupby("config_id", as_index=False)
        .agg(mean_ap=("ap", "mean"), mean_roc_auc=("roc_auc", "mean"))
        .sort_values(["mean_ap", "mean_roc_auc"], ascending=False, ignore_index=True)
    )
    best_id = int(ranking.iloc[0].config_id)
    best_params = candidates[best_id - 1]
    ranking.to_csv(output / "tuning_ranking.csv", index=False)

    evaluation_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold, train, validation in folds[2:]:
        expected_rows = EXPECTED_ROWS[fold]
        if len(validation) != expected_rows:
            raise ValueError(
                f"Fold {fold} has {len(validation):,} rows; expected {expected_rows:,}."
            )
        for seed in EVALUATION_SEEDS:
            started = perf_counter()
            pipeline = model_pipeline(model_key, make_estimator(seed, best_params))
            pipeline.fit(train[FEATURES], train[TARGET])
            probability = _probabilities(pipeline, validation[FEATURES])
            evaluation_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "late_rate": float(validation[TARGET].mean()),
                    "ap": float(average_precision_score(validation[TARGET], probability)),
                    "roc_auc": float(roc_auc_score(validation[TARGET], probability)),
                    "runtime_seconds": perf_counter() - started,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_id": validation.index.to_numpy(),
                        "fold": fold,
                        "seed": seed,
                        TARGET: validation[TARGET].to_numpy(),
                        "p_late": probability,
                    }
                )
            )
            _write_csv(evaluation_rows, output / "evaluation.csv")
            print(f"evaluation fold={fold} seed={seed} complete", flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    for fold, expected_rows in EXPECTED_ROWS.items():
        actual = predictions.loc[predictions.fold.eq(fold)].groupby("seed").size()
        if not actual.eq(expected_rows).all():
            raise AssertionError(f"Unexpected prediction count for fold {fold}: {actual.to_dict()}")
    predictions.to_parquet(output / "predictions.parquet", index=False)

    # The only all-development fit: seed 11 and already-locked settings.
    final_pipeline = model_pipeline(
        model_key, make_estimator(TUNING_SEED, best_params)
    ).fit(development[FEATURES], development[TARGET])
    joblib.dump(final_pipeline, output / "model.joblib", compress=3)

    importance = _importance_table(final_pipeline, FEATURES)
    if importance is not None:
        importance.to_csv(output / "feature_importances.csv", index=False)
    curve = (
        successful.groupby(iteration_parameter, as_index=False)
        .agg(mean_ap=("ap", "mean"), mean_roc_auc=("roc_auc", "mean"))
        .sort_values(iteration_parameter)
    )
    curve.to_csv(output / "learning_curve.csv", index=False)

    evaluation = pd.DataFrame(evaluation_rows)
    evaluation_summary = (
        evaluation.groupby("fold", as_index=False)
        .agg(mean_ap=("ap", "mean"), mean_roc_auc=("roc_auc", "mean"))
    )
    settings = {
        "model_key": model_key,
        "model_name": model_name,
        "code_commit": _git_commit(),
        "run_command": " ".join(sys.argv),
        "package_versions": _package_versions(package_names),
        "features": FEATURES,
        "tuning_grid": grid,
        "tuning_configurations": len(candidates),
        "tuning_folds": [1, 2],
        "best_parameters": best_params,
        "selection_rule": "highest mean AP; mean ROC-AUC breaks ties",
        "evaluation_folds": [3, 4],
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "early_stopping": "not used; iteration count is selected by chronological tuning folds and refit on full folds",
        "approved_exceptions": [],
        "notes": notes,
        "evaluation_summary": evaluation_summary.to_dict(orient="records"),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(evaluation_summary.to_string(index=False), flush=True)
