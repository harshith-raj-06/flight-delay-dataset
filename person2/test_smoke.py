"""Synthetic smoke test for Person 2 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from person2_pipeline import FOLDS, run_pipeline


def make_synthetic_df(start_date: str, end_date: str, n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    chosen_dates = rng.choice(dates, size=n_rows)
    sched_utc = pd.to_datetime(chosen_dates, utc=True) + pd.to_timedelta(rng.integers(6, 22, size=n_rows), unit="h")
    ckpt_utc = sched_utc - pd.Timedelta(hours=2)

    dests = [f"D{i:02d}" for i in range(25)]
    df = pd.DataFrame(
        {
            "row_id": np.arange(n_rows) + (0 if start_date.startswith("2025-01") else 1_000_000),
            "FlightDate": pd.to_datetime(chosen_dates).strftime("%Y-%m-%d"),
            "scheduled_dep_utc": sched_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "checkpoint_utc": ckpt_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "wx_obs_time": (ckpt_utc - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "month": pd.to_datetime(chosen_dates).month,
            "dow": pd.to_datetime(chosen_dates).dayofweek,
            "dep_hour": rng.integers(0, 24, size=n_rows),
            "arr_hour": rng.integers(0, 24, size=n_rows),
            "CRSElapsedTime": rng.uniform(60, 320, size=n_rows),
            "Distance": rng.uniform(200, 2500, size=n_rows),
            "is_holiday": rng.integers(0, 2, size=n_rows),
            "Origin": rng.choice(["ATL", "ORD", "LAX"], size=n_rows),
            "Dest": rng.choice(dests, size=n_rows),
            "Reporting_Airline": rng.choice(["DL", "UA", "AA", "WN"], size=n_rows),
            "wx_skyc1_T2h": rng.choice(["CLR", "FEW", "SCT", "BKN", None], size=n_rows),
            "wx_tmpf_T2h": rng.normal(65, 15, size=n_rows),
            "wx_dwpf_T2h": rng.normal(50, 12, size=n_rows),
            "wx_drct_T2h": rng.uniform(0, 360, size=n_rows),
            "wx_sknt_T2h": rng.uniform(0, 25, size=n_rows),
            "wx_gust_T2h": np.where(rng.random(n_rows) < 0.7, np.nan, rng.uniform(15, 40, size=n_rows)),
            "wx_vsby_T2h": rng.uniform(1, 10, size=n_rows),
            "wx_p01i_T2h": rng.exponential(0.05, size=n_rows),
            "wx_skyl1_T2h": np.where(rng.random(n_rows) < 0.3, np.nan, rng.uniform(500, 12000, size=n_rows)),
            "wx_obs_age_min_T2h": rng.uniform(0, 180, size=n_rows),
            "wx_ts_T2h": rng.integers(0, 2, size=n_rows).astype(float),
            "wx_fg_T2h": rng.integers(0, 2, size=n_rows).astype(float),
            "wx_ra_T2h": rng.integers(0, 2, size=n_rows).astype(float),
            "wx_sn_T2h": rng.integers(0, 2, size=n_rows).astype(float),
            "wx_fz_T2h": rng.integers(0, 2, size=n_rows).astype(float),
            "wx_br_hz_T2h": rng.integers(0, 2, size=n_rows).astype(float),
            "codes_known_T2h": rng.integers(0, 2, size=n_rows),
            "wx_gust_missing_T2h": rng.integers(0, 2, size=n_rows),
            "wx_skyl1_missing_T2h": rng.integers(0, 2, size=n_rows),
            "wx_unmatched_T2h": rng.integers(0, 2, size=n_rows),
            "n_departed_60min_T2h": rng.integers(0, 40, size=n_rows).astype(float),
            "prior_mean_delay_60min_T2h": np.where(rng.random(n_rows) < 0.1, np.nan, rng.normal(5, 15, size=n_rows)),
            "late_rate_60min_T2h": np.where(rng.random(n_rows) < 0.1, np.nan, rng.uniform(0, 0.6, size=n_rows)),
            "n_scheduled_next_60min_T2h": rng.integers(0, 45, size=n_rows).astype(float),
            "has_history_T2h": rng.integers(0, 2, size=n_rows),
        }
    )
    logits = -1.5 + 0.03 * df["wx_sknt_T2h"] + 0.8 * df["late_rate_60min_T2h"].fillna(0.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    df["LATE"] = (rng.random(n_rows) < probs).astype(int)
    return df


def main() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "out"
        data_dir.mkdir()

        train_df = make_synthetic_df("2025-01-01", "2025-08-30", 1200, seed=1)
        val_df = make_synthetic_df("2025-09-01", "2025-11-28", 800, seed=2)
        train_df.to_csv(data_dir / "train_T2h.csv", index=False)
        val_df.to_csv(data_dir / "validation_T2h.csv", index=False)

        run_pipeline(
            data_dir=data_dir,
            out_dir=out_dir,
            models_to_run=["constant", "decision_tree", "random_forest", "adaboost", "pca_logreg"],
            tune_rows=800,
            pca_variance=0.95,
        )

        summary = pd.read_csv(out_dir / "person2_summary_scores.csv")
        folds = pd.read_csv(out_dir / "person2_fold_scores.csv")
        pca_stats = pd.read_csv(out_dir / "person2_pca_stats.csv")
        oof_parquet = out_dir / "person2_oof_predictions.parquet"
        oof_csv = out_dir / "person2_oof_predictions.csv"
        oof = pd.read_parquet(oof_parquet) if oof_parquet.exists() else pd.read_csv(oof_csv)
        with (out_dir / "person2_chosen_settings.json").open(encoding="utf-8") as f:
            settings = json.load(f)

        assert set(summary["model"]) == {
            "constant",
            "decision_tree",
            "random_forest",
            "adaboost",
            "logreg_no_pca",
            "logreg_with_pca",
        }
        const_auc = summary.loc[summary["model"] == "constant", "roc_auc_mean"].iloc[0]
        assert abs(const_auc - 0.5) < 1e-6, f"Expected constant ROC-AUC 0.5, got {const_auc}"
        assert len(folds) == 6 * len(FOLDS)
        assert len(pca_stats) == len(FOLDS)
        assert len(oof) == len(val_df)
        assert "decision_tree" in settings["models"]
        print("SMOKE TEST PASSED!")


if __name__ == "__main__":
    main()
