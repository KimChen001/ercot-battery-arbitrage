# Results — Battery Storage Arbitrage at ERCOT HB_NORTH (Group 6)

All numbers are for the 2025 test year (365 days), produced by the scripts in `src/`
(run order in `README.md`). Models were trained on 2022–2023, tuned on 2024, then refit
on 2022–2024 before forecasting 2025.

**Battery (base case):** 100 MW / 200 MWh (two hours), 90% round-trip efficiency, wear
cost $5 per MWh charged or discharged, starts and ends each day empty, price taker,
day-ahead energy arbitrage only.

## 1. Forecast accuracy (Phase A)

| Model | MAE ($/MWh) | RMSE ($/MWh) | Hour-ranking (Spearman) | Peak hour hit | Cheapest hour hit |
| --- | --- | --- | --- | --- | --- |
| Ensemble (LightGBM + LSTM) | 8.85 | 20.8 | 0.809 | 50.7% | 41.1% |
| LightGBM | 8.85 | 21.2 | 0.820 | 53.2% | 41.1% |
| XGBoost | 8.97 | 21.4 | 0.805 | 54.0% | 39.2% |
| LASSO | 9.39 | 22.6 | 0.780 | 54.2% | 35.6% |
| LSTM | 10.11 | 22.0 | 0.745 | 38.4% | 29.9% |
| Naive | 12.18 | 28.8 | 0.727 | 49.9% | 32.3% |

Naive = same hour last week on Mon/Sat/Sun, same hour yesterday on Tue–Fri. LASSO's tuned
penalty is close to zero, so it behaves like ordinary linear regression. MAPE is not
reported because prices can be near zero.

## 2. Decisions and profit (Phases B and C)

Each forecast feeds the MILP; the schedule is settled at actual prices.

| Forecast | Decision | 2025 profit | Mean / median per day | % of perfect foresight | % excl. 19 spike days | Loss days |
| --- | --- | --- | --- | --- | --- | --- |
| Perfect foresight | MILP | $3,372,141 | $9,239 / $6,754 | 100% | 100% | 0 |
| Ensemble | MILP | $3,065,563 | $8,399 / $6,062 | **90.9%** | 88.1% | 18 |
| LightGBM | MILP | $3,021,222 | $8,277 / $6,010 | 89.6% | 87.7% | 19 |
| LASSO | MILP | $2,969,472 | $8,136 / $5,795 | 88.1% | 84.6% | 18 |
| XGBoost | MILP | $2,950,408 | $8,083 / $5,950 | 87.5% | 85.3% | 20 |
| LSTM | MILP | $2,911,474 | $7,977 / $5,776 | 86.3% | 83.3% | 20 |
| Naive | MILP | $2,738,343 | $7,502 / $5,427 | 81.2% | 78.4% | 28 |
| Perfect foresight | Simple rule | $2,521,206 | $6,907 / $5,626 | 74.8% | 76.3% | 40 |
| LightGBM | Simple rule | $2,326,846 | — | 69.0% | 70.0% | 43 |
| Naive | Simple rule | $2,018,627 | — | 59.9% | 59.1% | 70 |

Simple rule = charge at full power in the 2 cheapest forecast hours, discharge in the 2
dearest. Spike days = top 5% of 2025 days by actual max–min price spread.

## 3. Are the differences real? (`results/stat_tests.csv`)

Diebold–Mariano test on daily MAE (Newey–West, 7-day lag); 95% intervals for the annual
profit difference from a 7-day moving-block bootstrap (5,000 draws).

| A vs B | More accurate? (DM p-value) | Extra 2025 profit of A | 95% interval |
| --- | --- | --- | --- |
| LASSO vs Naive | Yes, A (p < 0.001) | +$231k | +$121k to +$355k |
| LightGBM vs Naive | Yes, A (p < 0.001) | +$283k | +$184k to +$399k |
| LightGBM vs LASSO | Yes, A (p = 0.001) | +$52k | −$33k to +$118k (not significant) |
| XGBoost vs LightGBM | No difference (p = 0.22) | −$71k | −$117k to −$31k |
| LSTM vs LightGBM | No, B (p < 0.001) | −$110k | −$194k to −$23k |
| Ensemble vs LightGBM | No difference (p = 1.00) | +$44k | −$9k to +$119k (not significant) |

## 4. Does lower error mean more profit? (`results/error_vs_profit.csv`)

Rank correlations with daily regret (% of the perfect-foresight profit left on the table),
pooled over the six forecasts, days with perfect-foresight profit ≥ $1,000:

| Daily metric | Rank correlation with regret |
| --- | --- |
| Hour-ranking accuracy (Spearman) | **−0.68** |
| Peak + cheapest hour hits | −0.52 |
| MAE | +0.28 |

Across the six models, MAE and capture rate are also strongly related (−0.94), but
pairs of models break the pattern: XGBoost is more accurate than LASSO yet earns $19k
less, and the ensemble matches LightGBM's MAE yet earns $44k more.

## 5. Sensitivity (`results/sensitivity_table.csv`, MILP, % of perfect foresight)

| Scenario | Perfect-foresight profit | Ensemble | LightGBM | LASSO | XGBoost | LSTM | Naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base (2 h, 90%, $5 wear) | $3.4M | 90.9 | 89.6 | 88.1 | 87.5 | 86.3 | 81.2 |
| 1-hour battery | $1.9M | 87.0 | 86.8 | 85.3 | 85.3 | 81.2 | 77.2 |
| 4-hour battery | $4.9M | 85.4 | 87.6 | 85.5 | 86.0 | 76.0 | 78.7 |
| RTE 85% | $3.2M | 90.2 | 88.4 | 87.6 | 86.4 | 84.8 | 79.7 |
| RTE 95% | $3.6M | 91.6 | 90.3 | 89.0 | 88.4 | 87.5 | 82.5 |
| Wear $0/MWh | $4.4M | 93.4 | 92.3 | 91.0 | 91.3 | 90.7 | 85.8 |
| Wear $10/MWh | $2.6M | 84.9 | 82.6 | 83.6 | 81.7 | 77.8 | 71.4 |
| Wear $20/MWh | $1.4M | 59.8 | 58.7 | 40.1 | 55.2 | 46.3 | 29.9 |

## 6. What each data source is worth (`results/experiment_ablation.csv`, fig7)

LightGBM refit without one group of variables at a time (same settings), then MILP on 2025.

| Variant | MAE ($/MWh) | % of perfect foresight | Profit vs. all features |
| --- | --- | --- | --- |
| All 29 features | 8.85 | 89.6% | – |
| Without price history | 13.34 | 73.0% | −$559k |
| Without calendar | 8.98 | 85.2% | −$147k |
| Without temperature forecast | 9.86 | 88.4% | −$39k |
| Without load | 9.17 | 90.2% | +$20k |
| Without observed weather | 8.88 | 90.3% | +$22k |
| Price history only | 10.32 | 85.4% | −$143k |
| Price history + calendar | 9.99 | 86.6% | −$102k |

Recent prices are essential, the calendar (hour-of-day shape) comes second and the
temperature forecast adds about $39k. Load and observed weather from two days earlier add
nothing beyond these; the ±$20k changes are well inside the bootstrap noise in section 3.

## 7. PCA as a forecasting input (`results/experiment_pcr.csv`)

Linear regression on the first 10 principal components (90% of feature variance) reaches
MAE 10.22 and 85.7% of perfect foresight, vs. 9.39 and 88.1% for LASSO on all 27 features
(−$81k). Compressing the features discards price-relevant detail, so PCA is used for
understanding the data, not as a model input.

## 8. Renewable forecasts (`results/experiment_renewables.csv`)

Day-before (48 h) wind and solar forecasts for the West Texas wind belt (32.47 N, 100.41 W)
exist only from 2024, so both models here are trained on 2024 alone and tested on 2025.

| LightGBM trained on 2024 only | MAE | Hour-ranking | % of perfect foresight |
| --- | --- | --- | --- |
| Base 29 features | 9.35 | 0.81 | 88.6% |
| + wind and solar forecasts (4 features) | 8.89 | 0.84 | 89.9% (+$44k) |

With one year of training data, renewable forecasts lift profit by 1.3 points, bringing a
2024-only model level with the full 2022–2024 LightGBM (89.6%). A longer archive of
renewable forecasts is the most promising data improvement.

## 9. Robustness: the West hub (`robustness_hb_west/`)

Same pipeline at HB_WEST (wind-driven, 1,375 negative-price hours in 2022–2025, 341 in
2025), with Midland weather (31.99 N, 102.08 W). No XGBoost run here.

| Forecast | Decision | MAE ($/MWh) | 2025 profit | % of perfect foresight |
| --- | --- | --- | --- | --- |
| Perfect foresight | MILP | – | $3,808,362 | 100% |
| Ensemble (LightGBM + LSTM) | MILP | 12.57 | $3,332,147 | **87.5%** |
| LSTM | MILP | 12.42 | $3,319,862 | 87.2% |
| LightGBM | MILP | 14.05 | $3,159,278 | 83.0% |
| LASSO | MILP | 13.91 | $3,121,086 | 82.0% |
| Naive | MILP | 16.73 | $2,917,882 | 76.6% |
| Perfect foresight | Simple rule | – | $2,837,592 | 74.5% |

The main findings hold: every model beats naive, the ensemble is best, and the simple rule
with perfect prices still earns less than naive + MILP. The ranking among ML models does
change: at the more volatile West hub the LSTM beats LightGBM, so the ensemble is the
safest choice across markets.

## 10. Why a validation year? (`results/experiment_no_validation.csv`)

Plan A trains on 2022–2024 with untuned settings (library defaults, or a guessed number
of LSTM epochs) and goes straight to the 2025 test. Plan B, ours, first holds out 2024 to
choose the settings, then refits on 2022–2024. Both final models are trained on 2022–2024
and tested on 2025, so the only difference is where the settings come from.

| Model | Plan | Settings | MAE ($/MWh) | % of perfect foresight |
| --- | --- | --- | --- | --- |
| LASSO | A: no validation | Library default penalty (1.0) | 13.37 | 0.0% |
| LASSO | B: tuned on 2024 | Penalty 0.0001 | 9.39 | 88.1% |
| LightGBM | A: no validation | Library defaults (31 leaves, 100 trees) | 8.84 | 87.4% |
| LightGBM | B: tuned on 2024 | 63 leaves, 486 trees | 8.85 | 89.6% |
| LSTM | A: no validation | Guessed 100 epochs | 11.89 | 87.1% |
| LSTM | B: tuned on 2024 | 15 epochs (early stopping on 2024) | 10.11 | 86.3% |

Without a validation year, LASSO's default penalty shrinks every coefficient to zero: it
forecasts the same price for every hour, the MILP sees no spread and never trades. The
LSTM trained for a guessed 100 epochs overfits (MAE 11.89 vs. 10.11). LightGBM's defaults
happen to work, but that can only be discovered by looking at the test year, which would
then no longer be an honest test. Because settings were tuned for MAE rather than profit,
plan A sometimes earns a little more (the LSTM), another sign that error and profit
diverge; tuning directly on backtested profit is future work.

## 11. Key findings

1. **Better forecasts earn more, significantly.** LightGBM + MILP earns $283k (+10%) more
   than the naive forecast in 2025 and captures 89.6% of the perfect-foresight profit;
   the ensemble reaches 90.9%.
2. **Optimization matters more than forecast quality.** The simple rule with perfect
   price knowledge (74.8%) earns less than the naive forecast fed to the MILP (81.2%).
3. **Ranking the hours beats minimizing error.** Daily regret tracks hour-ranking accuracy
   (−0.68) far more than MAE (+0.28). On 2025-05-07 LASSO under-forecast the evening peak
   by about 40% but placed it in the right hour and earned 99% of the perfect-foresight
   profit; naive matched the peak price but one hour late and earned 73%.
4. **Statistically better accuracy is not always more money.** LightGBM is significantly
   more accurate than LASSO, but its extra profit (+$52k) is not statistically
   significant; deep learning (LSTM) loses to LightGBM on both counts with only 2–3 years
   of daily training samples.
5. **Price level matters once trading has a cost.** With high wear cost ($20/MWh) the
   predicted spread must clear the cost before the MILP trades, so models that compress
   peaks (LASSO) fall to 40% while LightGBM keeps 59%.
6. **Ranking is robust.** Naive trails LASSO, LightGBM and the ensemble in every battery
   scenario, and at the West hub too.
7. **Which ML model wins depends on the market.** LightGBM leads at HB_NORTH, the LSTM at
   the wind-driven HB_WEST; averaging them (the ensemble) is best at both hubs.
8. **Recent prices are the key data source.** Removing price history costs $559k; the
   calendar is worth $147k and the temperature forecast $39k, while load and older
   observed weather add nothing.
9. **Renewable forecasts are the next data to add.** Where they exist (2024 onward) they
   raise profit by 1.3 points.
10. **A validation year is necessary.** Without one, default LASSO settings earn nothing
    and the LSTM overfits; holding out 2024 also keeps the 2025 test honest.

## 12. Figures (`figures/`)

| File | What it shows |
| --- | --- |
| `fig0_price_timeseries.png` | Daily mean and maximum price 2022–2025 with the train / tune / test periods |
| `fig0b_price_distribution.png` | Distribution of hourly prices (log scale): median $26, 99th percentile $320 |
| `fig1_hockey_stick.png` | Price vs. ERCOT load: flat until load nears the peak, then spikes (why a non-linear model helps) |
| `fig2_capture_by_model.png` | % of perfect-foresight profit by forecast, MILP vs. simple rule |
| `fig3_error_vs_profit.png` | Daily regret vs. hour-ranking accuracy and vs. MAE |
| `fig4_example_day.png` | 2025-05-07: prices and MILP schedules for perfect foresight, LASSO and naive |
| `fig5_monthly_profit.png` | Monthly 2025 profit, perfect foresight vs. ensemble |
| `fig6_sensitivity.png` | Capture rate across battery duration, efficiency and wear-cost scenarios |
| `fig7_ablation.png` | Profit when each group of variables is removed |
| `shap_importance.png` | LightGBM feature importance (SHAP) |
| `pca_scree.png`, `pca_loadings.png` | PCA of the 27 continuous features |
| `robustness_hb_west/figures/` | SHAP chart for the HB_WEST LightGBM |

## 13. Caveats

Day-ahead energy arbitrage at one hub only (no real-time or ancillary-service revenue,
no grid fees or degradation over time); assumed battery parameters (covered by the
sensitivity runs); 2025 was calmer than the training years (mean price $33 vs. $56–65),
so absolute profits are year-specific while the model comparison is like-for-like; day
before wind/solar forecasts are unavailable before 2024, so weather features are the
48-hour temperature forecast plus observed weather from two days earlier; the forecast
archive gap (2023-12-29 to 2024-01-19) is filled from observed temperature.
