"""Leakage-safe feature table for day-ahead price forecasting at ERCOT HB_NORTH.

Decision timing: forecasts for operating day D are made on the morning of D-1, before
the day-ahead market closes. At that moment we know:
  - all day-ahead prices for D-1 (cleared on D-2) and earlier;
  - actual load and observed weather only up to D-2 (D-1 is still in progress);
  - weather forecasts issued 48 h before each hour of D (Open-Meteo previous_day2).

Each day is normalized to 24 hours: on fall-back days the repeated hour is dropped, on
spring-forward days the missing hour ending 3 is the mean of hours ending 2 and 4.

Usage: python -P src/features.py <project_root>
Output: data/processed/features.csv  (one row per operating day x hour ending)
"""

import os
import sys

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

BASE_TEMP_C = 18.3  # 65 F, the usual degree-day base


def to_day_matrix(df, col):
    """Return a (days x 24) DataFrame of `col`, indexed by operating day."""
    m = df.pivot_table(index="date_local", columns="he", values=col, aggfunc="first",
                       dropna=False)
    # keep every operating day, even ones where this column is entirely missing
    m = m.reindex(index=sorted(df["date_local"].unique()), columns=range(1, 25))
    if m[3].isna().any():  # spring-forward days have no hour ending 3
        m[3] = m[3].fillna((m[2] + m[4]) / 2)
    return m


def stack(m, name):
    s = m.stack(future_stack=True).rename(name)
    s.index = s.index.set_names(["date", "he"])
    return s


def main(root, hub="HB_NORTH", weather="dallas", raw=None):
    proc = os.path.join(root, "data", "processed")
    hourly = pd.read_csv(os.path.join(proc, f"ercot_{hub.lower()}_hourly.csv"),
                         parse_dates=["date_local"])
    hourly["ts_utc"] = pd.to_datetime(hourly["ts_utc"], utc=True)
    hourly = hourly[~hourly["repeated"]]

    raw = raw or os.path.join(root, "data", "raw")
    fc2 = pd.read_csv(os.path.join(raw, "weather", f"{weather}_prev_day2_2022_2025.csv"),
                      skiprows=3)
    fc2["ts_utc"] = (pd.to_datetime(fc2["time"]).dt.tz_localize("UTC")
                     - pd.Timedelta(hours=1))
    fc2 = fc2.rename(columns={fc2.columns[1]: "temp_fc_d2"})[["ts_utc", "temp_fc_d2"]]
    hourly = hourly.merge(fc2, on="ts_utc", how="left")

    P = to_day_matrix(hourly, "price")
    L = to_day_matrix(hourly, "load_ercot")
    T_fc = to_day_matrix(hourly, "temp_fc_d2")
    T_obs = to_day_matrix(hourly, "temp_obs").interpolate(axis=1, limit_direction="both")
    WIND = to_day_matrix(hourly, "wind_obs").interpolate(axis=1, limit_direction="both")
    SOLAR = to_day_matrix(hourly, "solar_obs").interpolate(axis=1, limit_direction="both")

    temp_filled = T_fc.isna()
    T_fc = T_fc.fillna(T_obs)  # 2023-12-29..2024-01-19 archive gap, flagged below

    days = P.index
    # ---- price history (all known on the morning of D-1) ----
    price_d1 = P.shift(1)
    feats = {
        "price_lag1d": price_d1,
        "price_lag2d": P.shift(2),
        "price_lag7d": P.shift(7),
        "price_samehour_7d_mean": sum(P.shift(k) for k in range(1, 8)) / 7,
    }
    day_level = {
        "price_d1_mean": price_d1.mean(axis=1),
        "price_d1_max": price_d1.max(axis=1),
        "price_d1_min": price_d1.min(axis=1),
        "price_d1_spread": price_d1.max(axis=1) - price_d1.min(axis=1),
        "price_7d_std": pd.concat([P.shift(k) for k in range(1, 8)], axis=1).std(axis=1),
    }
    # ---- load and observed weather: only through D-2 ----
    feats["load_lag2d"] = L.shift(2)
    feats["load_lag7d"] = L.shift(7)
    day_level["load_d2_mean"] = L.shift(2).mean(axis=1)
    day_level["load_d2_max"] = L.shift(2).max(axis=1)
    feats["wind_obs_lag2d"] = WIND.shift(2)
    feats["solar_obs_lag2d"] = SOLAR.shift(2)
    day_level["solar_obs_d2_mean"] = SOLAR.shift(2).mean(axis=1)
    # ---- weather forecast for D, issued 48 h ahead ----
    feats["temp_fc"] = T_fc
    feats["cooling_deg"] = (T_fc - BASE_TEMP_C).clip(lower=0)
    feats["heating_deg"] = (BASE_TEMP_C - T_fc).clip(lower=0)
    day_level["temp_fc_daymax"] = T_fc.max(axis=1)
    day_level["temp_fc_daymin"] = T_fc.min(axis=1)

    out = pd.concat([stack(m, k) for k, m in feats.items()], axis=1)
    out["price"] = stack(P, "price")
    out["temp_fc_filled"] = stack(temp_filled, "f").astype(int)
    out = out.reset_index()
    for k, s in day_level.items():
        out[k] = out["date"].map(s)

    # ---- calendar ----
    d = pd.to_datetime(out["date"])
    holidays = USFederalHolidayCalendar().holidays(days.min(), days.max())
    out["hour_sin"] = np.sin(2 * np.pi * (out["he"] - 1) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * (out["he"] - 1) / 24)
    out["dow"] = d.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_holiday"] = d.isin(holidays).astype(int)
    out["month"] = d.dt.month
    out["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)

    out["price_asinh"] = np.arcsinh(out["price"])
    out["split"] = np.select([d.dt.year <= 2023, d.dt.year == 2024], ["train", "val"], "test")
    out = out[out["date"] >= days[0] + pd.Timedelta(days=7)]  # first week lacks lags

    out.to_csv(os.path.join(proc, "features.csv"), index=False)
    feat_cols = [c for c in out.columns
                 if c not in ("date", "he", "price", "price_asinh", "split", "temp_fc_filled")]
    print(f"rows {len(out):,}  days {out['date'].nunique():,}  features {len(feat_cols)}")
    print("split rows:", out["split"].value_counts().to_dict())
    print("rows with any missing feature:", int(out[feat_cols].isna().any(axis=1).sum()))
    print("temp forecast filled from observed:", int(out["temp_fc_filled"].sum()), "hours")
    print("features:", feat_cols)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--hub", default="HB_NORTH")
    ap.add_argument("--weather", default="dallas", help="prefix of the Open-Meteo files")
    ap.add_argument("--raw", default=None, help="raw data folder (default <root>/data/raw)")
    a = ap.parse_args()
    main(a.root, a.hub, a.weather, a.raw)
