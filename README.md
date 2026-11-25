# Flight departure delay dataset

Build the 2025 BTS cohort for departures from Atlanta (ATL), Chicago O’Hare (ORD), and Los Angeles (LAX), then attach Iowa Environmental Mesonet airport observations at two and six hours before scheduled departure.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_bts.py
python weather.py
```

The first command script writes flights_raw.parquet and inbound_history.parquet. The latter contains flights arriving at the three airports and includes a year-boundary lookback. The weather script caches station downloads and writes weather_raw.parquet, output_with_weather.parquet, and output_with_weather.csv. The enriched table retains all scheduled departures for subsequent modelling eligibility checks. Each weather match uses the latest same-origin report at or before its UTC checkpoint, within three hours. Download logs and generated data are local. If changing the data window, use a fresh station cache.

Sources: [BTS monthly archives](https://www.transtats.bts.gov/PREZIP/) and [IEM ASOS archive](https://mesonet.agron.iastate.edu/request/download.phtml).
