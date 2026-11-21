"""Phase B + C: schedule the battery from each forecast and settle at actual prices.

For every 2025 day and every forecast in results/forecast_*.csv:
  1. decide: MILP (and, for comparison, the k-cheapest/k-dearest heuristic) on the
     forecast prices;
  2. evaluate: realized profit with the ACTUAL prices.
Perfect foresight = the same MILP fed the actual prices (upper bound).

Usage: python -P src/backtest.py <project_root> [--kappa 5] [--e-max 200] [--rte 0.9] [--tag base]
Writes results/backtest_daily_<tag>.csv, results/backtest_summary_<tag>.csv,
       results/schedules_<tag>.csv
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from battery import Battery, heuristic, profit, solve_milp


def day_matrix(df, col):
    return df.pivot(index="date", columns="he", values=col).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--e-max", type=float, default=200.0)
    ap.add_argument("--p-max", type=float, default=100.0)
    ap.add_argument("--rte", type=float, default=0.90)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--milp-only", action="store_true", help="skip the heuristic")
    args = ap.parse_args()
    methods = (("milp", solve_milp),) if args.milp_only else (
        ("milp", solve_milp), ("heuristic", heuristic))
    bat = Battery(p_max=args.p_max, e_max=args.e_max, rte=args.rte, kappa=args.kappa)
    res_dir = os.path.join(args.root, "results")

    feats = pd.read_csv(os.path.join(args.root, "data", "processed", "features.csv"))
    actual = day_matrix(feats[feats["split"] == "test"], "price")
    forecasts = {}
    for path in sorted(glob.glob(os.path.join(res_dir, "forecast_*.csv"))):
        name = os.path.basename(path)[len("forecast_"):-4]
        forecasts[name] = day_matrix(pd.read_csv(path), "pred").reindex(actual.index)

    sources = {"perfect": actual, **forecasts}
    daily, sched = [], []
    t0 = time.time()
    for day in actual.index:
        a = actual.loc[day].to_numpy()
        for src, mat in sources.items():
            f = mat.loc[day].to_numpy()
            for method, fn in methods:
                c, d, s = fn(f, bat)
                row = {"date": day, "forecast": src, "method": method,
                       "profit": profit(a, c, d, bat),
                       "mwh_discharged": float(d.sum())}
                if src != "perfect":
                    row.update(mae=float(np.mean(np.abs(f - a))),
                               rmse=float(np.sqrt(np.mean((f - a) ** 2))),
                               spearman=float(spearmanr(f, a).statistic),
                               peak_hit=int(np.argmax(f) == np.argmax(a)),
                               trough_hit=int(np.argmin(f) == np.argmin(a)))
                daily.append(row)
                if method == "milp":
                    sched.append(pd.DataFrame({"date": day, "he": range(1, 25), "forecast": src,
                                               "price_used": f, "actual": a,
                                               "charge": c, "discharge": d, "soc": s}))
    print(f"solved {len(daily):,} schedules in {time.time() - t0:.0f}s")

    daily = pd.DataFrame(daily)
    pf = daily[(daily["forecast"] == "perfect") & (daily["method"] == "milp")].set_index("date")["profit"]
    daily["regret"] = daily["date"].map(pf) - daily["profit"]
    spread = (actual.max(axis=1) - actual.min(axis=1))
    spike_days = set(spread[spread >= spread.quantile(0.95)].index)
    daily["spike_day"] = daily["date"].isin(spike_days)

    rows = []
    for (src, method), g in daily.groupby(["forecast", "method"], sort=False):
        calm = g[~g["spike_day"]]
        rows.append({
            "forecast": src, "method": method,
            "MAE": g["mae"].mean(), "RMSE": np.sqrt((g["rmse"] ** 2).mean()),
            "spearman": g["spearman"].mean(), "peak_hit_%": 100 * g["peak_hit"].mean(),
            "trough_hit_%": 100 * g["trough_hit"].mean(),
            "profit_total_$": g["profit"].sum(),
            "profit_mean_day_$": g["profit"].mean(),
            "profit_median_day_$": g["profit"].median(),
            "capture_%": 100 * g["profit"].sum() / pf.sum(),
            "capture_ex_spike_%": 100 * calm["profit"].sum() / pf[~pf.index.isin(spike_days)].sum(),
            "loss_days": int((g["profit"] < 0).sum()),
        })
    summary = pd.DataFrame(rows)
    order = {"perfect": 0}
    summary = summary.sort_values(["method", "profit_total_$"], ascending=[False, False],
                                  key=lambda s: s.map(lambda v: order.get(v, v)) if s.name == "method" else s)

    tag = args.tag
    daily.to_csv(os.path.join(res_dir, f"backtest_daily_{tag}.csv"), index=False)
    summary.to_csv(os.path.join(res_dir, f"backtest_summary_{tag}.csv"), index=False)
    pd.concat(sched).to_csv(os.path.join(res_dir, f"schedules_{tag}.csv"), index=False)

    pd.set_option("display.width", 200)
    fmt = {c: "{:,.0f}".format for c in summary.columns if "$" in c}
    fmt.update({c: "{:.1f}".format for c in ["MAE", "RMSE", "peak_hit_%", "trough_hit_%",
                                              "capture_%", "capture_ex_spike_%"]})
    fmt["spearman"] = "{:.3f}".format
    print(f"Battery {bat.p_max:.0f} MW / {bat.e_max:.0f} MWh, RTE {bat.rte:.0%}, "
          f"wear ${bat.kappa:.0f}/MWh; {len(actual)} test days, {len(spike_days)} spike days")
    print(summary.to_string(index=False, formatters=fmt, na_rep="-"))


if __name__ == "__main__":
    main()
