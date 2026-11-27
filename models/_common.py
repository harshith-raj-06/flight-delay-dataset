"""Shared helpers for T-2h model training, evaluation, feature analysis, and metadata."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import FEATURES, TARGET  # noqa: E402

SEEDS = (11, 23, 42)


def git_commit():
    """Return current git HEAD hash or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def package_versions():
    """Return core runtime package versions for settings.json."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pyarrow": pyarrow.__version__,
    }


def feature_names_from_prep(prep):
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


def raw_feature_name(encoded_name):
    """Map an encoded feature name back to one of the 35 raw FEATURES."""
    if encoded_name in FEATURES:
        return encoded_name
    for cat in ("Dest", "Origin", "Reporting_Airline", "wx_skyc1_T2h"):
        if encoded_name.startswith(f"{cat}_"):
            return cat
    return encoded_name


def save_feature_importances(output, prep, values, col_name="importance", abs_group=False):
    """Save encoded and raw-feature-grouped importance/weight CSVs."""
    names = feature_names_from_prep(prep)
    table = pd.DataFrame(
        {
            "feature": names,
            "raw_feature": [raw_feature_name(name) for name in names],
            col_name: np.asarray(values, dtype=float),
        }
    )
    sort_col = table[col_name].abs() if abs_group else table[col_name]
    table = table.assign(_sort=sort_col).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    table.to_csv(output / f"feature_{col_name}s.csv" if col_name != "importance" else output / "feature_importances.csv", index=False)

    grouped_vals = table.assign(_val=table[col_name].abs() if abs_group else table[col_name])
    grouped = (
        grouped_vals.groupby("raw_feature", as_index=False)["_val"]
        .sum()
        .rename(columns={"_val": f"abs_{col_name}" if abs_group else col_name})
        .sort_values(f"abs_{col_name}" if abs_group else col_name, ascending=False)
        .reset_index(drop=True)
    )
    grouped_name = "feature_importances_grouped.csv" if col_name == "importance" else f"feature_{col_name}s_grouped.csv"
    grouped.to_csv(output / grouped_name, index=False)


def prediction_frame(row_ids, fold, seed, y_true, prob):
    """Build standard predictions.parquet slice matching team schema."""
    return pd.DataFrame(
        {
            "row_id": np.asarray(row_ids),
            "fold": np.int32(fold),
            "seed": np.int32(seed),
            TARGET: np.asarray(y_true, dtype=np.int8),
            "p_late": np.asarray(prob, dtype=np.float32),
        }
    )


def finalize_eval_outputs(output, eval_rows, pred_frames):
    """Write evaluation.csv and predictions.parquet, and return per-fold/overall means."""
    eval_df = pd.DataFrame(eval_rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    eval_df.to_csv(output / "evaluation.csv", index=False)

    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)

    fold_means = eval_df.groupby("fold")[["ap", "roc_auc", "lift", "late_rate"]].mean()
    fold_stds = eval_df.groupby("fold")[["ap", "roc_auc", "lift"]].std(ddof=1).fillna(0.0)
    overall = fold_means.mean()
    print(fold_means.to_string(), flush=True)
    print("Mean across folds 3-4:", overall.to_dict(), flush=True)
    return eval_df, predictions, fold_means, fold_stds, overall


def write_settings(
    output,
    *,
    model_key,
    grid,
    best_params,
    fold_means,
    fold_stds,
    overall,
    development_rows,
    predictions_rows,
    refit_seconds,
    early_stopping="none",
    approved_exceptions=None,
    boundary_extensions=None,
    importance_method=None,
    extra=None,
):
    """Write standardized settings.json matching team submission checklist."""
    payload = {
        "model_key": model_key,
        "code_commit": git_commit(),
        "run_command": " ".join(sys.argv),
        "package_versions": package_versions(),
        "features": FEATURES,
        "grid": grid,
        "boundary_extensions_tested": boundary_extensions or [],
        "best_params": best_params,
        "seeds": list(SEEDS),
        "tuning_seed": SEEDS[0],
        "refit_seed": SEEDS[0],
        "refit_seconds": refit_seconds,
        "development_rows": int(development_rows),
        "early_stopping": early_stopping,
        "approved_exceptions": approved_exceptions or [],
        "fold_summary": {
            "fold_3": {
                **fold_means.loc[3].to_dict(),
                "ap_seed_std": float(fold_stds.loc[3, "ap"]),
                "roc_auc_seed_std": float(fold_stds.loc[3, "roc_auc"]),
            },
            "fold_4": {
                **fold_means.loc[4].to_dict(),
                "ap_seed_std": float(fold_stds.loc[4, "ap"]),
                "roc_auc_seed_std": float(fold_stds.loc[4, "roc_auc"]),
            },
            "mean_folds_3_4": overall.to_dict(),
        },
        "predictions_rows": int(predictions_rows),
    }
    if importance_method is not None:
        payload["importance_method"] = importance_method
    if extra:
        payload.update(extra)
    (output / "settings.json").write_text(json.dumps(payload, indent=2) + "\n")
