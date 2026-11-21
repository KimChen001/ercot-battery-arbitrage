"""Download every raw input used by the project into data/raw/.

ERCOT (public, no login):
  - Historical DAM Load Zone and Hub Prices, NP4-180-ER, one zip per year 2022-2025
  - Hourly native load archive, one zip per year 2022-2025
Open-Meteo (free for non-commercial use), hourly, UTC:
  - Dallas (HB_NORTH) and Midland (HB_WEST): near-observed weather (Historical Forecast API)
    plus 24 h and 48 h ahead GFS forecasts (Previous Runs API)
  - West Texas wind belt: 48 h ahead wind and solar forecasts, 2024-2025

Files that already exist are skipped (use --force to re-download). Zips are extracted
next to themselves. Open-Meteo archives are occasionally back-filled, so a fresh download
can differ slightly from the copy committed in this repository.

Usage: python -P src/download_data.py <project_root> [--force]
"""

import argparse
import os
import zipfile

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (course-project data download)"}
ERCOT_DAM = {2022: 886627599, 2023: 969803138, 2024: 1065468714, 2025: 1177667469}
ERCOT_LOAD = {
    2022: "https://www.ercot.com/files/docs/2022/02/08/Native_Load_2022.zip",
    2023: "https://www.ercot.com/files/docs/2023/02/09/Native_Load_2023.zip",
    2024: "https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip",
    2025: "https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip",
}
OBS_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,cloud_cover"
DAY1_VARS = ("temperature_2m_previous_day1,wind_speed_10m_previous_day1,"
             "shortwave_radiation_previous_day1")
SITES = {"dallas": (32.78, -96.80), "midland": (31.99, -102.08)}
HIST_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV_API = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch(url, path, force, params=None):
    if os.path.exists(path) and not force:
        print(f"  exists   {os.path.relpath(path)}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, params=params, headers=HEADERS, timeout=300)
    r.raise_for_status()
    with open(path, "wb") as fh:
        fh.write(r.content)
    print(f"  saved    {os.path.relpath(path)}  ({len(r.content) / 1e6:.1f} MB)")


def extract(path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not os.path.exists(os.path.join(os.path.dirname(path), name)):
                z.extract(name, os.path.dirname(path))


def open_meteo(api, site, start, end, hourly, models=None):
    lat, lon = site
    p = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
         "hourly": hourly, "timezone": "GMT", "format": "csv"}
    if models:
        p["models"] = models
    return api, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    raw = os.path.join(a.root, "data", "raw")

    print("ERCOT day-ahead hub prices")
    for y, doc in ERCOT_DAM.items():
        path = os.path.join(raw, "ercot_dam", f"DAMLZHBSPP_{y}.zip")
        fetch(f"https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc}", path, a.force)
        extract(path)
    print("ERCOT hourly load")
    for y, url in ERCOT_LOAD.items():
        path = os.path.join(raw, "ercot_load", f"Native_Load_{y}.zip")
        fetch(url, path, a.force)
        extract(path)

    print("Open-Meteo weather")
    wdir = os.path.join(raw, "weather")
    jobs = []
    for name, site in SITES.items():
        jobs += [
            (f"{name}_hist_forecast_2022_2025.csv",
             open_meteo(HIST_API, site, "2022-01-01", "2025-12-31", OBS_VARS)),
            (f"{name}_prev_day1_2022_2025.csv",
             open_meteo(PREV_API, site, "2022-01-01", "2025-12-31", DAY1_VARS, "gfs_seamless")),
            (f"{name}_prev_day2_2022_2025.csv",
             open_meteo(PREV_API, site, "2022-01-01", "2025-12-31",
                        "temperature_2m_previous_day2", "gfs_seamless")),
        ]
    jobs.append(("westtx_renew_prev_day2_2024_2025.csv",
                 open_meteo(PREV_API, (32.47, -100.41), "2024-01-01", "2025-12-31",
                            "wind_speed_100m_previous_day2,shortwave_radiation_previous_day2",
                            "gfs_seamless")))
    for fname, (api, params) in jobs:
        fetch(api, os.path.join(wdir, fname), a.force, params)


if __name__ == "__main__":
    main()
