"""Build a 2025 departure cohort and a separate history of flights into its airports."""

import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

YEAR = 2025
AIRPORTS = {"ATL", "ORD", "LAX"}
ROOT = Path(__file__).resolve().parents[1] / "fresh_2025"
BASE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
FIELDS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "Distance",
]


def download_month(year, month):
    url = BASE.format(year=year, month=month)
    path = ROOT / "bts_archives" / f"{year}-{month:02}.zip"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return path
    partial = path.with_suffix(".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("CRC failure")
            partial.replace(path)
            with (ROOT / "bts_download_log.txt").open("a") as log:
                print(
                    datetime.now(UTC).isoformat(),
                    url,
                    path.stat().st_size,
                    file=log,
                )
            return path
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == 2:
                raise
            time.sleep(5 * 2**attempt)


def main():
    ROOT.mkdir(exist_ok=True)
    targets, inbound = [], []
    # Two lookback days also cover US local-date differences for a 24-hour history.
    earliest = (date(YEAR, 1, 1) - timedelta(days=2)).isoformat()
    months = [(YEAR - 1, 12)] + [(YEAR, m) for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(download_month, year, month) for year, month in months]
        for (year, month), job in zip(months, jobs):
            with zipfile.ZipFile(job.result()) as archive:
                (name,) = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                with archive.open(name) as stream:
                    for frame in pd.read_csv(
                        stream,
                        usecols=FIELDS,
                        chunksize=100_000,
                        dtype={"Tail_Number": "string"},
                    ):
                        active = frame.FlightDate.ge(earliest)
                        targets.append(
                            frame.loc[
                                active & frame.Origin.isin(AIRPORTS), FIELDS
                            ].copy()
                        )
                        inbound.append(
                            frame.loc[active & frame.Dest.isin(AIRPORTS), FIELDS].copy()
                        )
            print(f"Read {year}-{month:02d}", flush=True)
    with TemporaryDirectory(dir=ROOT) as temp:
        for parts, name in [
            (targets, "flights_raw.parquet"),
            (inbound, "inbound_history.parquet"),
        ]:
            destination = Path(temp) / name
            data = pd.concat(parts, ignore_index=True)
            data.to_parquet(destination, index=False)
            destination.replace(ROOT / name)
            print(name, len(data))


if __name__ == "__main__":
    main()
