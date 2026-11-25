"""Attach IEM airport observations to flights as of each prediction checkpoint."""

import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import airportsdata
import pandas as pd
import requests

CHECKPOINT_HOURS = [2, 6]
MAX_OBSERVATION_AGE = pd.Timedelta(hours=3)
REQUEST_PAUSE_SECONDS = 10
REQUEST_ATTEMPTS = 3

ROOT = Path(__file__).resolve().parents[1] / "fresh_2025"
FLIGHTS = ROOT / "flights_raw.parquet"
WEATHER_CACHE = ROOT / "weather_raw"
WEATHER_PARQUET = ROOT / "weather_raw.parquet"
DOWNLOAD_LOG = ROOT / "download_log.txt"
OUTPUT_CSV = ROOT / "output_with_weather.csv"
OUTPUT_PARQUET = ROOT / "output_with_weather.parquet"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_FIELDS = (
    "tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "p01i", "skyc1", "skyl1", "wxcodes",
)
SCHEDULE_FIELDS = ("FlightDate", "Origin", "CRSDepTime")
OUTCOME_FIELDS = {
    "DepTime", "DepDelay", "DepDel15", "ArrTime", "ArrDelay", "Cancelled", "Diverted",
}
UTC_DTYPE = "datetime64[ns, UTC]"


class DownloadError(RuntimeError):
    """IEM failed to return a valid CSV for the requested interval."""


def log_download(message):
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    with DOWNLOAD_LOG.open("a", encoding="utf-8") as log:
        print(f"{timestamp} {message}", file=log)


def read_station_csv(source, origin):
    raw = pd.read_csv(source, dtype="string", keep_default_na=False, comment="#")
    columns = ("station", "valid", *IEM_FIELDS)
    if not set(columns).issubset(raw.columns):
        raise ValueError(f"{origin}: expected an IEM CSV with {', '.join(columns)}")
    raw = raw.loc[:, list(columns)]
    if not raw["station"].isin((origin, f"K{origin}")).all():
        raise ValueError(f"{origin}: unexpected station IDs {raw['station'].unique().tolist()}")
    raw["valid"] = pd.to_datetime(raw["valid"], utc=True, errors="coerce").astype(UTC_DTYPE)
    if raw["valid"].isna().any():
        raise ValueError(f"{origin}: IEM CSV contains unparseable observation timestamps")
    return raw


def retry_after_seconds(response):
    header = response.headers.get("Retry-After") if response is not None else None
    if not header:
        return 0
    if header.isdecimal():
        return int(header)
    try:
        return max(0, (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError):
        return 0


class IEMDownloader:
    def __init__(self, session):
        self.session = session
        self.next_request = 0

    def fetch(self, origin, station, start, end):
        params = [
            ("station", station),
            *(("data", field) for field in IEM_FIELDS),
            ("year1", start.year), ("month1", start.month), ("day1", start.day),
            ("year2", end.year), ("month2", end.month), ("day2", end.day),
            ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
            ("missing", "M"), ("trace", "T"), ("report_type", 3), ("report_type", 4),
        ]
        url = requests.Request("GET", IEM_URL, params=params).prepare().url
        for attempt in range(REQUEST_ATTEMPTS):
            delay = self.next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            response = None
            log_download(f"GET attempt={attempt + 1} {url}")
            try:
                response = self.session.get(url, timeout=(15, 120))
                response.raise_for_status()
                raw = read_station_csv(BytesIO(response.content), origin)
            except (requests.RequestException, ValueError) as error:
                log_download(f"FAILED station={station} {type(error).__name__}: {error}")
                if attempt == REQUEST_ATTEMPTS - 1:
                    self.next_request = time.monotonic() + max(
                        REQUEST_PAUSE_SECONDS, retry_after_seconds(response)
                    )
                    raise DownloadError(f"{station} {start} through {end} (exclusive)") from error
                print(f"{station}: request attempt {attempt + 1} failed ({error}); retrying.")
                self.next_request = time.monotonic() + max(
                    REQUEST_PAUSE_SECONDS * 2**attempt, retry_after_seconds(response)
                )
            else:
                self.next_request = time.monotonic() + REQUEST_PAUSE_SECONDS
                log_download(f"OK rows={len(raw)} {response.url}")
                return response.content, raw

    def fetch_range(self, origin, station, start, end):
        try:
            return self.fetch(origin, station, start, end)
        except DownloadError:
            first_boundary = (pd.Timestamp(start) + pd.offsets.MonthBegin()).date()
            if first_boundary >= end:
                raise

        print(f"{station}: full-range download failed; requesting calendar months.")
        frames = []
        while start < end:
            boundary = min((pd.Timestamp(start) + pd.offsets.MonthBegin()).date(), end)
            _, raw = self.fetch(origin, station, start, boundary)
            frames.append(raw)
            start = boundary
        raw = pd.concat(frames, ignore_index=True)
        content = raw.to_csv(index=False, date_format="%Y-%m-%d %H:%M").encode("utf-8")
        return content, raw


def load_weather(stations, start, end):
    """Reuse the station CSVs cached for the flight dataset's date window."""
    cache_dir = WEATHER_CACHE / f"{start}_{end}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as session:
        downloader = IEMDownloader(session)
        for origin in stations:
            cache = cache_dir / f"{origin}.csv"
            try:
                raw = read_station_csv(cache, origin)
            except FileNotFoundError:
                content, raw = downloader.fetch_range(origin, origin, start, end)
                if raw.empty:
                    print(f"{origin}: no observations; trying K{origin}.")
                    content, raw = downloader.fetch_range(origin, f"K{origin}", start, end)
                with TemporaryDirectory(dir=WEATHER_CACHE) as temp_dir:
                    temporary = Path(temp_dir) / cache.name
                    temporary.write_bytes(content)
                    temporary.replace(cache)
                print(f"{origin}: cached {len(raw):,} observations.")
            else:
                print(f"{origin}: loaded {len(raw):,} observations from cache.")
            if raw.empty:
                print(f"{origin}: no IEM observations available in the downloaded range.")
            # Retain the IEM identifier and use the BTS airport code for the join.
            frames.append(raw.assign(Origin=origin))

    combined = pd.concat(frames, ignore_index=True)
    in_range = combined["valid"].ge(pd.Timestamp(start, tz="UTC")) & combined["valid"].lt(
        pd.Timestamp(end, tz="UTC")
    )
    return combined.loc[in_range].reset_index(drop=True)


def inspect_flights(flights):
    print("Columns:", flights.columns.tolist())
    print("Dtypes:")
    print(flights.dtypes.to_string())
    print("Head:")
    print(flights.head().to_string(index=False))
    print(f"Flight rows: {len(flights):,}")
    if not set(SCHEDULE_FIELDS).issubset(flights.columns):
        raise ValueError(f"Expected schedule columns {SCHEDULE_FIELDS}")

    dates = pd.to_datetime(flights["FlightDate"], format="%Y-%m-%d", errors="coerce")
    if dates.notna().sum() == 0:
        raise ValueError("FlightDate contains no valid dates for the IEM request.")
    first, last = dates.min().date(), dates.max().date()
    print(f"FlightDate range: {first} through {last}")
    stations = sorted(flights["Origin"].dropna().unique())
    if not stations:
        raise ValueError("Origin contains no airport codes for the IEM request.")
    if any(len(code) != 3 or not code.isascii() or not code.isalnum() for code in stations):
        raise ValueError(f"Expected three-character IATA origin codes; found {stations}")
    print("Origin stations:", ", ".join(stations))
    # IEM excludes its end date. Include the entire day after the last FlightDate.
    start = first - timedelta(days=2)
    end = last + timedelta(days=2)
    print(f"IEM window (UTC): {start} through {end - timedelta(days=1)}, inclusive")
    return stations, start, end


def scheduled_departures_utc(schedule):
    # Only flight information known in advance can determine the checkpoint.
    assert tuple(schedule.columns) == SCHEDULE_FIELDS
    assert OUTCOME_FIELDS.isdisjoint(schedule.columns)

    dates = pd.to_datetime(schedule["FlightDate"], format="%Y-%m-%d", errors="coerce")
    hhmm = pd.to_numeric(schedule["CRSDepTime"], errors="coerce")
    hours, minutes = hhmm // 100, hhmm % 100
    valid = hhmm.ge(0) & hhmm.mod(1).eq(0) & (hhmm.lt(2400) | hhmm.eq(2400)) & minutes.lt(60)
    # This treats e.g. 5 as 0005 and adds 24 hours for 2400.
    local = dates + pd.to_timedelta((hours * 60 + minutes).where(valid), unit="min")

    airports = airportsdata.load("IATA")
    departure = pd.Series(pd.NaT, index=schedule.index, dtype=UTC_DTYPE, name="scheduled_dep_utc")
    for origin, rows in schedule.groupby("Origin", sort=False).groups.items():
        tz = airports.get(origin, {}).get("tz")
        if not tz:
            print(f"{origin}: no timezone available for {len(rows):,} rows.")
            continue
        # Use the first (DST) occurrence of the repeated fall-back hour at US airports.
        # Advance a nonexistent spring-forward time to the first valid local time.
        converted = local.loc[rows].dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        departure.loc[rows] = converted.dt.tz_convert("UTC").astype(UTC_DTYPE)
    print(f"Scheduled departure UTC conversion failures: {departure.isna().sum():,} / {len(schedule):,}")
    return departure


def build_weather_features(raw):
    weather = raw.loc[:, ["Origin", "valid", *IEM_FIELDS]]
    assert OUTCOME_FIELDS.isdisjoint(weather.columns)
    features = weather.loc[:, ["Origin", "valid"]].copy()
    missing = {"M": pd.NA, "": pd.NA}
    for field in ("tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "p01i", "skyl1"):
        values = weather[field].replace(missing)
        if field == "p01i":
            # Keep the reported accumulation. IEM's numeric trace convention is 0.0001 inches.
            values = values.replace("T", "0.0001")
        features[f"wx_{field}"] = pd.to_numeric(values, errors="coerce").astype("float64")
    features["wx_skyc1"] = weather["skyc1"].replace(missing)

    for field in ("gust", "skyl1"):
        values = features[f"wx_{field}"]
        features[f"wx_{field}_missing"] = values.isna().astype("Int8")
        features[f"wx_{field}_filled"] = values.fillna(0)

    codes = weather["wxcodes"].replace(missing).str.upper()
    # A missing code field leaves the indicators missing on that report.
    for name, code in {"ts": "TS", "fg": "FG", "ra": "RA", "sn": "SN", "fz": "FZ", "br_hz": "BR|HZ"}.items():
        # Match METAR's two-letter codes, including combinations such as -FZRA and +TSRA.
        pattern = rf"(?:^|\s)[+-]?(?:[A-Z]{{2}})*(?:{code})"
        features[f"wx_{name}"] = codes.str.contains(pattern).astype("Int8")

    return features


def attach_weather(flights, raw):
    original_columns = flights.columns.tolist()
    original_rows = len(flights)
    schedule = flights.loc[:, list(SCHEDULE_FIELDS)]
    departure = scheduled_departures_utc(schedule)
    if "scheduled_dep_utc" in original_columns:
        raise ValueError("Input already contains scheduled_dep_utc")

    # merge_asof needs the time key globally sorted, even with a station grouping key.
    left = pd.DataFrame({"Origin": schedule["Origin"].astype("string"), "departure": departure})
    left = left.dropna(subset=["departure"]).sort_values("departure", kind="stable")
    left = left.reset_index(names="flight_row")
    # If IEM lists multiple reports at the same time, merge_asof uses the last listed report.
    weather = build_weather_features(raw).sort_values("valid", kind="stable").reset_index(drop=True)
    right = weather.loc[:, ["Origin", "valid"]].assign(weather_row=weather.index)
    right["Origin"] = right["Origin"].astype("string")
    values = weather.drop(columns=["Origin", "valid"])
    flights["scheduled_dep_utc"] = departure

    for hours in CHECKPOINT_HOURS:
        left["T"] = left["departure"] - pd.Timedelta(hours=hours)
        # Match only observations at or before T, at the same airport and within three hours.
        # Join the weather row ID first to keep the time-key join small.
        matches = pd.merge_asof(
            left.loc[:, ["flight_row", "Origin", "T"]], right,
            by="Origin", left_on="T", right_on="valid", direction="backward",
            tolerance=MAX_OBSERVATION_AGE, allow_exact_matches=True,
        )
        ages = matches["T"] - matches["valid"]
        if (ages.lt(pd.Timedelta(0)) | ages.gt(MAX_OBSERVATION_AGE)).any():
            raise AssertionError(f"T-{hours}h: observation outside the allowed time window")

        # Restore CSV row order, including rows with invalid schedules and unmatched weather.
        matched = matches.set_index("flight_row").reindex(flights.index)
        attached = values.reindex(matched["weather_row"].to_numpy()).set_axis(flights.index)
        attached["wx_obs_time"] = matched["valid"]
        attached["wx_obs_age_min"] = (matched["T"] - matched["valid"]).dt.total_seconds() / 60
        attached = attached.add_suffix(f"_T{hours}h")
        if set(attached.columns) & set(flights.columns):
            raise ValueError(f"T-{hours}h: output feature names overlap existing columns")
        for column in attached:
            flights[column] = attached[column]

        age_column = f"wx_obs_age_min_T{hours}h"
        sample = flights.loc[:, ["Origin", "scheduled_dep_utc", f"wx_obs_time_T{hours}h", age_column]].head().copy()
        sample.insert(2, "T", matched["T"].head())
        print(f"\nT-{hours}h timestamp audit sample:")
        print(sample.to_string(index=False))
        print(f"T-{hours}h flights with weather: {matched['valid'].notna().sum():,} / {original_rows:,}")
        print(f"T-{hours}h median weather age: {attached[age_column].median():.1f} min")
        print(f"T-{hours}h missing by weather column:")
        print(attached.isna().mean().mul(100).to_string(float_format=lambda value: f"{value:.3f}%"))

    assert len(flights) == original_rows
    assert flights.columns[:len(original_columns)].tolist() == original_columns
    print(f"\nFlight rows before / after: {original_rows:,} / {len(flights):,}")
    print(f"New columns: {len(flights.columns) - len(original_columns)}")
    return flights


def write_outputs(raw, flights):
    # Stage the files so a failed write cannot leave a partially written CSV or Parquet.
    with TemporaryDirectory(prefix=".weather-", dir=ROOT) as temp_dir:
        staging = Path(temp_dir)
        raw.to_parquet(staging / WEATHER_PARQUET.name, engine="pyarrow", index=False)
        flights.to_parquet(staging / OUTPUT_PARQUET.name, engine="pyarrow", index=False)
        flights.to_csv(
            staging / OUTPUT_CSV.name, index=False, chunksize=50_000,
            date_format="%Y-%m-%dT%H:%M:%SZ",
        )
        for output in (WEATHER_PARQUET, OUTPUT_PARQUET, OUTPUT_CSV):
            (staging / output.name).replace(output)
            print(f"Wrote {output.name}")


def main():
    flights = pd.read_parquet(FLIGHTS)
    stations, start, end = inspect_flights(flights)
    raw = load_weather(stations, start, end)
    print(f"IEM observations in the requested range: {len(raw):,}")
    attach_weather(flights, raw)
    print("Writing CSV and Parquet outputs...")
    write_outputs(raw, flights)


if __name__ == "__main__":
    main()
