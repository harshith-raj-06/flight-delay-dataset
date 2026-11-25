# Input column audit

Each horizon uses the following 35 raw inputs. Append `_T2h` or `_T6h` to all weather and activity names. Weather measurements and their flags use the latest same-origin report at or before the checkpoint, up to 180 minutes old.

| Input | Type | Reason or meaning |
|---|---|---|
| month | Known in advance | Scheduled local flight month |
| dow | Known in advance | Scheduled local weekday, Monday = 0 |
| dep_hour | Known in advance | Scheduled origin-local departure hour |
| arr_hour | Known in advance | Scheduled destination-local arrival hour |
| CRSElapsedTime | Known in advance | Scheduled flight duration in minutes |
| Distance | Static | Published route distance in miles |
| is_holiday | Known in advance | Observed US federal holiday on the local flight date |
| Origin | Static | Departure airport |
| Dest | Known in advance | Scheduled destination |
| Reporting_Airline | Known in advance | Reporting carrier on the schedule |
| wx_skyc1 | Observed by T | First reported sky-cover layer |
| wx_tmpf | Observed by T | Temperature in °F |
| wx_dwpf | Observed by T | Dew point in °F |
| wx_drct | Observed by T | Wind direction in degrees true |
| wx_sknt | Observed by T | Wind speed in knots |
| wx_gust | Observed by T | Gust speed in knots |
| wx_vsby | Observed by T | Visibility in statute miles |
| wx_p01i | Observed by T | Reported precipitation in inches, trace = 0.0001; one report per prediction |
| wx_skyl1 | Observed by T | First cloud-layer base in feet |
| wx_obs_age_min | Observed by T | Minutes from report observation time to checkpoint |
| wx_ts | Observed by T | Thunderstorm in reported codes |
| wx_fg | Observed by T | Fog in reported codes |
| wx_ra | Observed by T | Rain in reported codes |
| wx_sn | Observed by T | Snow in reported codes |
| wx_fz | Observed by T | Freezing descriptor in reported codes |
| wx_br_hz | Observed by T | Mist or haze in reported codes |
| codes_known | Observed by T | Report contains a weather-code value; absent codes leave event flags missing |
| wx_gust_missing | Observed by T | Matched report lacks a gust measurement |
| wx_skyl1_missing | Observed by T | Matched report lacks a first-layer height |
| wx_unmatched | Observed by T | No valid same-origin report within three hours |
| n_departed_60min | Observed by T | Number of other same-origin departures in (T−60 min, T] |
| prior_mean_delay_60min | Observed by T | Mean signed departure delay in minutes among those departures |
| late_rate_60min | Observed by T | Fraction of those departures delayed at least 15 minutes |
| has_history | Observed by T | One when the eligible historical departure count is positive |
| n_scheduled_next_60min | Known in advance | Same-origin scheduled departures in (T, T+60 min], irrespective of actual operation |

Exclude five metadata columns from X: row_id (cross-horizon record identifier), FlightDate (split date), scheduled_dep_utc, checkpoint_utc and wx_obs_time (timestamp audit). LATE is the outcome. Current-flight realized times, delay causes, tail and flight numbers are excluded from all model matrices. For ablations group cloud coverage with weather.
