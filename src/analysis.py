"""Sensitivity table, statistical tests, and the forecast-error vs profit analysis.

Reads the backtest outputs in results/ and writes:
  results/sensitivity_table.csv   capture % by forecast for each battery scenario
  results/stat_tests.csv          Diebold-Mariano (forecast error) and block-bootstrap
                                  (annual profit) comparisons between model pairs
  results/error_vs_profit.csv     how well MAE vs rank accuracy explain daily regret

Usage: python -P src/analysis.py <project_root>
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

SCENARIOS = [("base", "Base: 2 h, RTE 90%, wear $5"), ("dur1h", "1-hour battery"),
             ("dur4h", "4-hour battery"), ("rte85", "RTE 85%"), ("rte95", "RTE 95%"),
             ("kappa0", "Wear $0/MWh"), ("kappa10", "Wear $10/MWh"), ("kappa20", "Wear $20/MWh")]
MODELS = ["ensemble", "lgbm", "lasso", "xgb", "lstm", "naive"]
PAIRS = [("lasso", "naive"), ("lgbm", "naive"), ("lgbm", "lasso"), ("xgb", "lgbm"),
         ("lstm", "lgbm"), ("ensemble", "lgbm")]


def dm_test(loss_a, loss_b, lag=7):
    """Diebold-Mariano test on daily losses with a Newey-West variance (weekly lag)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n, dbar = len(d), d.mean()
    dc = d - dbar
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = dbar / np.sqrt(lrv / n)
    return stat, 2 * (1 - norm.cdf(abs(stat)))


def block_bootstrap_sum(diff, block=7, reps=5000, seed=0):
    """Moving-block bootstrap of the annual total of a daily difference."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    return diff[idx].sum(axis=1)


def main(root):
    res = os.path.join(root, "results")

    # ---- sensitivity ----
    rows = []
    for tag, label in SCENARIOS:
        s = pd.read_csv(os.path.join(res, f"backtest_summary_{tag}.csv"))
        s = s[s["method"] == "milp"].set_index("forecast")
        row = {"scenario": label, "perfect_profit_$M": s.loc["perfect", "profit_total_$"] / 1e6}
        row.update({f"{m}_capture_%": s.loc[m, "capture_%"] for m in MODELS})
        best = max(MODELS, key=lambda m: s.loc[m, "capture_%"])
        row["best_model"] = best
        rows.append(row)
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(res, "sensitivity_table.csv"), index=False)
    print("Sensitivity (MILP capture % of perfect foresight):")
    print(sens.round(1).to_string(index=False))

    # ---- statistical tests (base scenario) ----
    daily = pd.read_csv(os.path.join(res, "backtest_daily_base.csv"), parse_dates=["date"])
    milp = daily[daily["method"] == "milp"]
    mae = milp.pivot(index="date", columns="forecast", values="mae")
    prof = milp.pivot(index="date", columns="forecast", values="profit")
    tests = []
    for a, b in PAIRS:
        stat, p = dm_test(mae[a], mae[b])
        boot = block_bootstrap_sum(prof[a] - prof[b])
        tests.append({"model_A": a, "model_B": b,
                      "MAE_A": mae[a].mean(), "MAE_B": mae[b].mean(),
                      "DM_stat (neg = A more accurate)": stat, "DM_p": p,
                      "profit_A_minus_B_$": (prof[a] - prof[b]).sum(),
                      "CI95_low_$": np.percentile(boot, 2.5),
                      "CI95_high_$": np.percentile(boot, 97.5),
                      "share_boot_<=0": (boot <= 0).mean()})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(res, "stat_tests.csv"), index=False)
    print("\nModel comparisons (2025):")
    print(tests.round(3).to_string(index=False))

    # ---- does accuracy or ranking explain profit? ----
    pf = prof["perfect"]
    d = milp[milp["forecast"] != "perfect"].copy()
    d["pf"] = d["date"].map(pf)
    d = d[d["pf"] >= 1000]  # skip near-flat days where regret % is unstable
    d["regret_pct"] = 100 * d["regret"] / d["pf"]
    rho_mae = spearmanr(d["mae"], d["regret_pct"]).statistic
    rho_rank = spearmanr(d["spearman"], d["regret_pct"]).statistic
    rho_peak = spearmanr(d["peak_hit"] + d["trough_hit"], d["regret_pct"]).statistic
    summ = pd.read_csv(os.path.join(res, "backtest_summary_base.csv"))
    summ = summ[(summ["method"] == "milp") & (summ["forecast"] != "perfect")]
    model_rho_mae = spearmanr(summ["MAE"], summ["capture_%"]).statistic
    model_rho_rank = spearmanr(summ["spearman"], summ["capture_%"]).statistic
    evp = pd.DataFrame([
        {"level": "day (all models pooled)", "metric": "daily MAE", "rank_corr_with_regret_%": rho_mae},
        {"level": "day (all models pooled)", "metric": "daily Spearman (hour ranking)", "rank_corr_with_regret_%": rho_rank},
        {"level": "day (all models pooled)", "metric": "peak + trough hits", "rank_corr_with_regret_%": rho_peak},
        {"level": "model (6 forecasts)", "metric": "MAE vs capture %", "rank_corr_with_regret_%": model_rho_mae},
        {"level": "model (6 forecasts)", "metric": "Spearman vs capture %", "rank_corr_with_regret_%": model_rho_rank},
    ])
    evp.to_csv(os.path.join(res, "error_vs_profit.csv"), index=False)
    print("\nError vs profit (rank correlations; regret should RISE with MAE and FALL with ranking skill):")
    print(evp.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
