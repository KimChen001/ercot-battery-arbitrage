# Battery Storage Arbitrage in ERCOT — Group 6

15.774 / 15.780 AI for Operations, Fall 2026 · Zhixin Kim Chen, Howard Chuang, Achille
Jacheet, Michelle Lin, Joshua Yang

We forecast next-day hourly day-ahead electricity prices at ERCOT's North hub (HB_NORTH),
turn each forecast into a battery charge/discharge schedule with a mixed-integer linear
program (MILP), and settle the schedule at actual prices to measure realized profit:
**predict → decide → evaluate**, on 2022–2025 data with 2025 held out for testing.

## Headline results (2025, 100 MW / 200 MWh battery)

| Forecast + MILP | MAE ($/MWh) | 2025 profit | % of perfect foresight |
| --- | --- | --- | --- |
| Perfect foresight (upper bound) | – | $3.37M | 100% |
| Ensemble (LightGBM + LSTM) | 8.85 | $3.07M | 90.9% |
| LightGBM | 8.85 | $3.02M | 89.6% |
| LASSO | 9.39 | $2.97M | 88.1% |
| Naive | 12.18 | $2.74M | 81.2% |
| Simple rule with *perfect* prices | – | $2.52M | 74.8% |

Better forecasts earn significantly more, the optimization layer is worth more than a
perfect forecast, and ranking the hours correctly matters more than a low average error.
Full tables, statistical tests, sensitivity analysis, ablations and the HB_WEST robustness
check are in **[RESULTS.md](RESULTS.md)**.

## Repository layout

```
run_all.py              one command to rebuild everything
requirements.txt
RESULTS.md              all result tables, findings, figure list, caveats
src/
  download_data.py      ERCOT + Open-Meteo raw files
  build_dataset.py      merge prices, load, weather into one hourly table (UTC-aligned) + QA
  features.py           leakage-safe feature table (29 variables)
  pca_analysis.py       PCA of the features
  forecast_baselines.py naive + LASSO
  forecast_lgbm.py      LightGBM + SHAP
  forecast_xgb.py       XGBoost (robustness check)
  forecast_lstm.py      LSTM + LightGBM/LSTM ensemble
  battery.py            MILP (scipy HiGHS), simple rule, profit
  backtest.py           schedule every 2025 day from every forecast, settle at actual prices
  analysis.py           sensitivity table, Diebold-Mariano tests, bootstrap, error vs profit
  figures.py            report figures 0-6
  experiments.py        feature ablation (figure 7), PCA regression, renewable forecasts
data/raw/               ERCOT zips and Open-Meteo CSVs (as downloaded)
data/processed/         hourly table, feature table, QA and PCA summaries
results/                forecasts, backtest summaries, test statistics (CSV)
figures/                all charts (PNG)
robustness_hb_west/     the same pipeline at the West hub
```

## Quick start

```
pip install -r requirements.txt
python -P run_all.py --skip-download     # ~10-15 min on a laptop CPU, no GPU needed
```

`run_all.py` without `--skip-download` first fetches any missing raw files. The MILP uses
scipy's built-in HiGHS solver, so no Gurobi license is needed. Results are deterministic
except the LSTM, which can vary slightly across machines.

## Data

`data/processed/ercot_hb_north_hourly.csv` — one row per delivery hour, 2022-01-01 to
2025-12-31, 35,064 rows.

| Column | Meaning | Source |
| --- | --- | --- |
| `ts_utc` | Hour start in UTC (join key) | derived |
| `ts_local`, `date_local`, `he`, `repeated` | Central time hour start, operating day, ERCOT hour ending (1–24), repeated fall-back hour flag | ERCOT |
| `price` | HB_NORTH day-ahead settlement point price, $/MWh (target) | [ERCOT NP4-180-ER](https://www.ercot.com/mp/data-products/data-product-details?id=NP4-180-ER) |
| `load_ercot`, `load_ncent` | Actual hourly load, MW: ERCOT total and North Central zone (Dallas–Fort Worth) | [ERCOT native load archive](https://www.ercot.com/gridinfo/load/load_hist) |
| `temp_fc_d1` | Dallas temperature forecast issued 24 h before (GFS), °C. Late hours of day D are issued after the bid deadline, so `features.py` uses the 48 h forecast instead | [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) |
| `wind_fc_d1`, `solar_fc_d1` | Day-before wind and solar forecasts; only from 2024-01-19 | Open-Meteo Previous Runs API |
| `temp_obs`, `rh_obs`, `wind_obs`, `solar_obs`, `cloud_obs` | Near-observed weather (stitched first forecast hours); not known at bid time, used only lagged | [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) |

Weather points: Dallas 32.78 N, 96.80 W (HB_NORTH); Midland 31.99 N, 102.08 W (HB_WEST);
West Texas wind belt 32.47 N, 100.41 W (48 h wind/solar forecasts, 2024–2025 only).

**Leakage rule.** Forecasts for day D are made on the morning of D-1. At that time we know
day-ahead prices through D-1 (cleared on D-2), actual load and observed weather only
through D-2, and weather forecasts issued 48 h ahead.

## Features (`data/processed/features.csv`)

One row per operating day × hour ending (24 per day; daylight-saving days normalized),
34,896 rows, split train 2022–2023 / tune 2024 / test 2025. Target `price` (models are
fit on `price_asinh`).

| Group | Features |
| --- | --- |
| Price history (through D-1) | `price_lag1d`, `price_lag2d`, `price_lag7d`, `price_samehour_7d_mean`, `price_d1_mean`, `price_d1_max`, `price_d1_min`, `price_d1_spread`, `price_7d_std` |
| Load (through D-2) | `load_lag2d`, `load_lag7d`, `load_d2_mean`, `load_d2_max` |
| Temperature forecast for D (48 h ahead) | `temp_fc`, `cooling_deg`, `heating_deg`, `temp_fc_daymax`, `temp_fc_daymin` |
| Observed weather (D-2) | `wind_obs_lag2d`, `solar_obs_lag2d`, `solar_obs_d2_mean` |
| Calendar | `hour_sin`, `hour_cos`, `dow`, `is_weekend`, `is_holiday`, `month`, `doy_sin`, `doy_cos` |

`temp_fc_filled` = 1 for the 523 hours where the forecast archive gap was filled from
observed temperature. PCA and LASSO use the 27 continuous features (no raw `dow`, `month`).

## Data quality (`data/processed/qa_report.txt`)

- No missing or duplicate hours; daylight-saving days have 23 hours (4 days) or 25 hours (4 days).
- The 48 h temperature forecast is missing for 2023-12-29 to 2024-01-19 and the last hours
  of 2025; filled from observed temperature and flagged.
- Prices: median $26/MWh, max $4,200 (Aug 2023), 92 hours above $1,000, 6 negative hours.
- Spikes dominate: the top 5% of days hold 47% of all daily max–min spread, mostly the
  August 2023 heat wave, so results are reported with and without spike days.
