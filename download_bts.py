"""Build a 2025 departure cohort and a separate history of flights into its airports."""
import io
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
import time
from tempfile import TemporaryDirectory
import zipfile

import pandas as pd
import requests

YEAR = 2025
AIRPORTS = {"ATL", "ORD", "LAX"}
ROOT = Path(__file__).resolve().parent
BASE = ("https://transtats.bts.gov/PREZIP/"
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
FIELDS = ["FlightDate", "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
          "Origin", "Dest", "CRSDepTime", "DepTime", "DepDelay", "DepDel15", "CRSArrTime",
          "ArrTime", "ArrDelay", "Cancelled", "Diverted", "CRSElapsedTime", "Distance"]


def download_month(year, month):
    url = BASE.format(year=year, month=month)
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(20, 180))
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                name, = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                with archive.open(name) as stream:
                    frame = pd.read_csv(stream, usecols=FIELDS, low_memory=False)
            with (ROOT / "bts_download_log.txt").open("a") as log:
                print(datetime.now(timezone.utc).isoformat(), url, len(frame), file=log)
            return frame[FIELDS]
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == 2:
                raise
            time.sleep(5 * 2**attempt)


def main():
    targets, inbound = [], []
    # Two lookback days also cover US local-date differences for a 24-hour history.
    earliest = (date(YEAR, 1, 1) - timedelta(days=2)).isoformat()
    for year, month in [(YEAR - 1, 12)] + [(YEAR, m) for m in range(1, 13)]:
        frame = download_month(year, month)
        if year == YEAR:
            targets.append(frame.loc[frame.Origin.isin(AIRPORTS)].copy())
        inbound.append(frame.loc[frame.Dest.isin(AIRPORTS) & frame.FlightDate.ge(earliest)].copy())
        print(f"Read {year}-{month:02d}", flush=True)
    with TemporaryDirectory(dir=ROOT) as temp:
        for parts, name in [(targets, "flights_raw.parquet"), (inbound, "inbound_history.parquet")]:
            destination = Path(temp) / name
            data = pd.concat(parts, ignore_index=True)
            data.to_parquet(destination, index=False)
            destination.replace(ROOT / name)
            print(name, len(data))


if __name__ == "__main__":
    main()
