"""Create checkpoint-specific, chronological flight-delay modelling CSVs."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from pipeline import iem

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "final_csv"
WX = [
    "skyc1",
    "tmpf",
    "dwpf",
    "drct",
    "sknt",
    "gust",
    "vsby",
    "p01i",
    "skyl1",
    "obs_age_min",
    "ts",
    "fg",
    "ra",
    "sn",
    "fz",
    "br_hz",
    "gust_missing",
    "skyl1_missing",
]
SHARED = [
    "month",
    "dow",
    "dep_hour",
    "arr_hour",
    "CRSElapsedTime",
    "Distance",
    "is_holiday",
    "Origin",
    "Dest",
    "Reporting_Airline",
]
HISTORY = [
    "n_departed_60min",
    "prior_mean_delay_60min",
    "late_rate_60min",
    "n_scheduled_next_60min",
    "has_history",
]
META = ["row_id", "FlightDate", "scheduled_dep_utc", "checkpoint_utc", "wx_obs_time"]


def activity(frame, checkpoint, allowed, protected, actual):
    result = pd.DataFrame(index=frame.index, columns=HISTORY, dtype=float)
    for origin, group in frame.groupby("Origin"):
        rows = group.index
        t = checkpoint.loc[rows]
        departures = actual.loc[rows[allowed.loc[rows]]].sort_values()
        delay = frame.DepDelay.loc[departures.index]
        start = departures.searchsorted(t - pd.Timedelta(hours=1), side="right")
        end = departures.searchsorted(t, side="right")
        own = (
            allowed.loc[rows]
            & actual.loc[rows].gt(t - pd.Timedelta(hours=1))
            & actual.loc[rows].le(t)
        )
        count = end - start - own.astype(int)
        result.loc[rows, HISTORY[0]] = count
        result.loc[rows, "has_history"] = count.gt(0).astype(int)
        for col, values, own_values in [
            (HISTORY[1], delay, group.DepDelay),
            (HISTORY[2], delay.ge(15), group.DepDelay.ge(15)),
        ]:
            total = np.r_[0, values.to_numpy().cumsum()]
            result.loc[rows, col] = (
                total[end] - total[start] - own_values.astype(float).where(own, 0)
            ) / count.replace(0, np.nan)
        schedule = group.scheduled_dep_utc.sort_values()
        result.loc[rows, HISTORY[3]] = schedule.searchsorted(
            t + pd.Timedelta(hours=1), side="right"
        ) - schedule.searchsorted(t, side="right")
        # Withhold outcome-history fields when the window overlaps a sealed local date.
        # A partial count would incorrectly imply that held-out events were observed.
        blocked = protected.loc[rows]
        if blocked.any():
            zone = {
                "ATL": "America/New_York",
                "ORD": "America/Chicago",
                "LAX": "America/Los_Angeles",
            }[origin]
            local = t.dt.tz_convert(zone)
            sealed_dates = set(group.FlightDate[blocked])
            unknown = local.dt.strftime("%Y-%m-%d").isin(sealed_dates) | (
                (local - pd.Timedelta(hours=1))
                .dt.strftime("%Y-%m-%d")
                .isin(sealed_dates)
            )
            result.loc[rows[unknown], [HISTORY[i] for i in (0, 1, 2, 4)]] = np.nan
    return result


def main():
    frame = pd.read_parquet(iem.OUTPUT_PARQUET)
    dates = frame.FlightDate
    schedule = frame.scheduled_dep_utc
    assert schedule.notna().all()
    actual = schedule + pd.to_timedelta(frame.DepDelay, unit="min")
    eligible = frame.Cancelled.eq(0) & frame.DepDelay.notna() & frame.DepDel15.notna()
    assert (
        frame.loc[eligible, "DepDelay"].ge(15).eq(frame.loc[eligible, "DepDel15"]).all()
    )
    old_test = frame.Origin.eq("ATL") & (
        dates.between("2025-01-27", "2025-01-31")
        | dates.between("2025-05-13", "2025-05-29")
    )
    test = dates.between("2025-12-01", "2025-12-31")
    protected = old_test
    allowed = eligible & ~protected
    validation = eligible & dates.between("2025-09-01", "2025-11-28")
    train = allowed & dates.between("2025-01-01", "2025-08-30")
    # Label must have become observable before the earliest validation prediction.
    train &= actual.lt((schedule[validation] - pd.Timedelta(hours=6)).min())
    first_test_prediction = (schedule[test & eligible] - pd.Timedelta(hours=6)).min()
    validation &= actual.lt(first_test_prediction)
    masks = {"train": train, "validation": validation, "test": test & eligible}
    assert not (train & validation | train & test | validation & test).any()
    assert set(frame.Origin.unique()) == {"ATL", "ORD", "LAX"}
    assert schedule[train].max() < (schedule[validation] - pd.Timedelta(hours=6)).min()
    assert schedule[validation].max() < first_test_prediction
    assert dates[test & eligible].min() == "2025-12-01"
    assert dates[test & eligible].max() == "2025-12-31"
    day = pd.to_datetime(dates)
    shared = frame[
        ["CRSElapsedTime", "Distance", "Origin", "Dest", "Reporting_Airline"]
    ].copy()
    shared["month"], shared["dow"] = day.dt.month, day.dt.dayofweek
    shared["dep_hour"] = (frame.CRSDepTime // 100) % 24
    shared["arr_hour"] = (frame.CRSArrTime // 100) % 24
    holidays = USFederalHolidayCalendar().holidays(day.min(), day.max())
    shared["is_holiday"] = day.isin(holidays).astype(int)
    OUT.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=OUT) as temp:
        staged = Path(temp)
        for hours in (2, 6):
            suffix = f"_T{hours}h"
            t = schedule - pd.Timedelta(hours=hours)
            obs = frame["wx_obs_time" + suffix]
            age = (t - obs).dt.total_seconds() / 60
            assert (
                age.equals(frame["wx_obs_age_min" + suffix])
                and age.dropna().between(0, 180).all()
            )
            wx = frame[["wx_" + s + suffix for s in WX]].copy()
            wx["codes_known" + suffix] = frame["wx_ts" + suffix].notna().astype(int)
            wx["wx_unmatched" + suffix] = obs.isna().astype(int)
            history = activity(frame, t, allowed, protected, actual)
            # Independent event-window audit on development rows, including empty windows.
            for i in dates[train | validation].sample(300, random_state=42).index:
                same = frame.Origin.eq(frame.at[i, "Origin"])
                past = (
                    allowed
                    & same
                    & actual.gt(t[i] - pd.Timedelta(hours=1))
                    & actual.le(t[i])
                )
                past.at[i] = False
                delays = frame.DepDelay[past]
                demand = (
                    same & schedule.gt(t[i]) & schedule.le(t[i] + pd.Timedelta(hours=1))
                ).sum()
                assert history.at[i, HISTORY[3]] == demand
                if pd.notna(history.at[i, HISTORY[0]]):
                    expected = [
                        len(delays),
                        delays.mean(),
                        delays.ge(15).mean(),
                        int(len(delays) > 0),
                    ]
                    np.testing.assert_allclose(
                        history.loc[i, [HISTORY[k] for k in (0, 1, 2, 4)]],
                        expected,
                        equal_nan=True,
                    )
            inputs = pd.concat([shared[SHARED], wx, history.add_suffix(suffix)], axis=1)
            assert inputs.shape[1] == 35
            meta = pd.DataFrame(
                {
                    "row_id": frame.index,
                    "FlightDate": dates,
                    "scheduled_dep_utc": schedule,
                    "checkpoint_utc": t,
                    "wx_obs_time": obs,
                }
            )
            table = pd.concat([meta, inputs, frame.DepDel15.rename("LATE")], axis=1)
            for name, mask in masks.items():
                part = table.loc[mask].sort_values(["checkpoint_utc", "row_id"])
                part.to_csv(
                    staged / f"{name}{suffix}.csv", index=False, chunksize=30000
                )
                print(
                    name,
                    suffix,
                    len(part),
                    part.FlightDate.min(),
                    part.FlightDate.max(),
                    flush=True,
                )
        for path in staged.glob("*.csv"):
            path.replace(OUT / path.name)


if __name__ == "__main__":
    main()
