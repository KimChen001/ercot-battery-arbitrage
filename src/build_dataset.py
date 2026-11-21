"""Build one clean hourly table for ERCOT HB_NORTH, 2022-2025.

Inputs (data/raw):
  ercot_dam/*.xlsx      ERCOT day-ahead hub prices (NP4-180-ER), one sheet per month
  ercot_load/*.xlsx     ERCOT hourly native load by weather zone
  weather/*.csv         Open-Meteo, Dallas, UTC timestamps

Output (data/processed):
  ercot_hb_north_hourly.csv   one row per delivery hour, keyed by UTC hour start
  qa_report.txt               data quality summary

Time convention: ERCOT reports "Hour Ending" in Central Prevailing Time. Hour ending
HE on date D covers local [D + (HE-1)h, D + HE h). On the fall-back day the repeated
hour is flagged (DAM: Repeated Hour Flag = Y, load: " DST" suffix); it is the second,
standard-time occurrence. Everything is converted to a UTC hour-start key `ts_utc`.

Usage: python -P src/build_dataset.py <project_root>
"""

import glob
import os
import re
import sys

import pandas as pd

TZ = "America/Chicago"
HUB = "HB_NORTH"


def to_utc(local_date, he, repeated):
    """Local operating date + hour ending (1-24) + repeated flag -> UTC hour start."""
    start = pd.to_datetime(local_date) + pd.to_timedelta(he - 1, unit="h")
    # ambiguous=True means "this is the daylight-time occurrence" (the first one).
    local = start.dt.tz_localize(TZ, ambiguous=(~repeated).to_numpy(), nonexistent="raise")
    return local.dt.tz_convert("UTC")


def load_prices(raw, hub=HUB):
    frames = []
    for path in sorted(glob.glob(os.path.join(raw, "ercot_dam", "*.xlsx"))):
        sheets = pd.read_excel(path, sheet_name=None)
        for df in sheets.values():
            frames.append(df[df["Settlement Point"] == hub])
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "date_local": pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y"),
            "he": df["Hour Ending"].str.slice(0, 2).astype(int),
            "repeated": df["Repeated Hour Flag"].eq("Y"),
            "price": df["Settlement Point Price"].astype(float),
        }
    )
    out["ts_utc"] = to_utc(out["date_local"], out["he"], out["repeated"])
    return out


LOAD_RE = re.compile(r"^(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d)( DST)?$")


def parse_load_stamp(v):
    """Return (local_date, hour_ending, repeated) for one 'Hour Ending' cell."""
    if isinstance(v, str):
        m = LOAD_RE.match(v.strip())
        if not m:
            raise ValueError(f"unparsed load timestamp: {v!r}")
        mm, dd, yyyy, hh, _, dst = m.groups()
        return pd.Timestamp(int(yyyy), int(mm), int(dd)), int(hh), dst is not None
    ts = pd.Timestamp(v)  # Excel stored the cell as a datetime
    if ts.hour == 0:  # 24:00 rolled over to the next day's 00:00
        return (ts - pd.Timedelta(days=1)).normalize(), 24, False
    return ts.normalize(), ts.hour, False


def load_load(raw):
    frames = []
    for path in sorted(glob.glob(os.path.join(raw, "ercot_load", "*.xlsx"))):
        df = pd.read_excel(path)
        parsed = df.iloc[:, 0].map(parse_load_stamp)
        frames.append(
            pd.DataFrame(
                {
                    "date_local": [p[0] for p in parsed],
                    "he": [p[1] for p in parsed],
                    "repeated": [p[2] for p in parsed],
                    "load_ercot": df["ERCOT"].astype(float),
                    "load_ncent": df["NCENT"].astype(float),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["ts_utc"] = to_utc(out["date_local"], out["he"], out["repeated"])
    return out[["ts_utc", "load_ercot", "load_ncent"]]


def read_open_meteo(path, rename):
    df = pd.read_csv(path, skiprows=3)
    df = df.rename(columns=lambda c: rename.get(c.split(" (")[0], c))
    # Open-Meteo stamps a value at the END of the hour it describes (radiation is the
    # preceding-hour mean), so shift back one hour to get the hour-start key.
    df["ts_utc"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC") - pd.Timedelta(hours=1)
    return df.drop(columns=["time"])


def main(root, hub=HUB, weather="dallas", raw=None):
    raw = raw or os.path.join(root, "data", "raw")
    out_dir = os.path.join(root, "data", "processed")

    prices = load_prices(raw, hub)
    load = load_load(raw)
    obs = read_open_meteo(
        os.path.join(raw, "weather", f"{weather}_hist_forecast_2022_2025.csv"),
        {
            "temperature_2m": "temp_obs",
            "relative_humidity_2m": "rh_obs",
            "wind_speed_10m": "wind_obs",
            "shortwave_radiation": "solar_obs",
            "cloud_cover": "cloud_obs",
        },
    )
    fc = read_open_meteo(
        os.path.join(raw, "weather", f"{weather}_prev_day1_2022_2025.csv"),
        {
            "temperature_2m_previous_day1": "temp_fc_d1",
            "wind_speed_10m_previous_day1": "wind_fc_d1",
            "shortwave_radiation_previous_day1": "solar_fc_d1",
        },
    )

    df = prices.merge(load, on="ts_utc", how="left").merge(obs, on="ts_utc", how="left")
    df = df.merge(fc, on="ts_utc", how="left")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(TZ)
    df = df.sort_values("ts_utc").reset_index(drop=True)
    cols = ["ts_utc", "ts_local", "date_local", "he", "repeated", "price", "load_ercot",
            "load_ncent", "temp_fc_d1", "wind_fc_d1", "solar_fc_d1", "temp_obs", "rh_obs",
            "wind_obs", "solar_obs", "cloud_obs"]
    df = df[cols]

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"ercot_{hub.lower()}_hourly.csv"), index=False)

    # ---- quality report ----
    lines = []
    add = lines.append
    add(f"Rows: {len(df):,}   Price rows: {len(prices):,}   Load rows: {len(load):,}")
    add(f"Range (local): {df['ts_local'].min()} -> {df['ts_local'].max()}")
    add(f"Duplicate UTC hours: {df['ts_utc'].duplicated().sum()}")
    full = pd.date_range(df["ts_utc"].min(), df["ts_utc"].max(), freq="h")
    add(f"Missing UTC hours in price series: {len(full.difference(df['ts_utc']))}")
    per_day = df.groupby("date_local").size()
    add(f"Hours per local day: {per_day.value_counts().sort_index().to_dict()}")
    add(f"  23-hour days: {[d.date().isoformat() for d in per_day[per_day == 23].index]}")
    add(f"  25-hour days: {[d.date().isoformat() for d in per_day[per_day == 25].index]}")
    add("Missing values per column:")
    for c in cols[5:]:
        n = int(df[c].isna().sum())
        first = df.loc[df[c].notna(), "ts_local"].min()
        add(f"  {c:12s} {n:6,d} missing   first value at {first}")
    p = df["price"]
    add("Price ($/MWh): " + ", ".join(
        f"{k}={v:,.2f}" for k, v in p.describe(percentiles=[0.01, 0.5, 0.99]).items()))
    add(f"  negative-price hours: {(p < 0).sum():,}   hours above $1,000: {(p > 1000).sum():,}")
    for y, g in df.groupby(df["ts_local"].dt.year):
        add(f"  {y}: mean {g['price'].mean():7.2f}  median {g['price'].median():6.2f}  "
            f"max {g['price'].max():8.2f}  negative {int((g['price'] < 0).sum()):4d}")
    daily = df.groupby("date_local")["price"].agg(lambda s: s.max() - s.min())
    add(f"Daily max-min spread: median {daily.median():.2f}, mean {daily.mean():.2f}, "
        f"max {daily.max():,.2f} on {daily.idxmax().date()}")
    lo = df["load_ercot"]
    add(f"ERCOT load (MW): min {lo.min():,.0f}  median {lo.median():,.0f}  max {lo.max():,.0f}")
    add(f"Corr(price, load_ercot) = {df['price'].corr(lo):.3f}   "
        f"Corr(price, temp_fc_d1) = {df['price'].corr(df['temp_fc_d1']):.3f}")
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "qa_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--hub", default="HB_NORTH")
    ap.add_argument("--weather", default="dallas", help="prefix of the Open-Meteo files")
    ap.add_argument("--raw", default=None, help="raw data folder (default <root>/data/raw)")
    a = ap.parse_args()
    main(a.root, a.hub, a.weather, a.raw)
