"""Report figures. Run after backtest.py (all scenarios) and analysis.py.

Usage: python -P src/figures.py <project_root>
Writes figures/fig1_hockey_stick.png ... fig6_sensitivity.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#a8a7a1"
COLOR = {"lgbm": "#2a78d6", "naive": "#eb6834", "ensemble": "#6250d6", "lasso": "#e87ba4",
         "milp": "#1baf7a", "heuristic": "#c3c2b7", "actual": INK, "perfect": NEUTRAL}
NAME = {"perfect": "Perfect foresight", "ensemble": "Ensemble (LightGBM + LSTM)",
        "lgbm": "LightGBM", "lasso": "LASSO", "xgb": "XGBoost", "lstm": "LSTM", "naive": "Naive"}
EXAMPLE_DAY = "2025-05-07"


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def title(ax, text, sub=None):
    ax.set_title(text if sub is None else f"{text}\n", loc="left", color=INK, fontsize=11)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def save(fig, root, name):
    fig.tight_layout()
    fig.savefig(os.path.join(root, "figures", name), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_timeseries(root):
    h = pd.read_csv(os.path.join(root, "data", "processed", "ercot_hb_north_hourly.csv"),
                    parse_dates=["date_local"])
    daily = h.groupby("date_local")["price"].agg(["mean", "max"])
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    style(ax)
    spans = [("2022-01-01", "2024-01-01", "Train 2022-2023"), ("2024-01-01", "2025-01-01", "Tune 2024"),
             ("2025-01-01", "2026-01-01", "Test 2025")]
    for i, (a, b, lab) in enumerate(spans):
        if i % 2 == 1:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.5, zorder=0, linewidth=0)
        ax.text(pd.Timestamp(a) + pd.Timedelta(days=12), 6000, lab, color=INK2, fontsize=8.5, va="top")
    ax.plot(daily.index, daily["max"], color=COLOR["naive"], linewidth=1, label="Daily maximum")
    ax.plot(daily.index, daily["mean"], color=COLOR["lgbm"], linewidth=1.4, label="Daily mean")
    ax.set_yscale("log")
    ax.set_ylim(5, 8000)
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(["$10", "$100", "$1,000"])
    ax.set_ylabel("HB_NORTH day-ahead price ($/MWh, log scale)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left", ncol=2)
    title(ax, "Spikes cluster in summer heat waves; 2025 had the fewest extreme spikes",
          "Daily mean and maximum hourly price, 2022-2025")
    save(fig, root, "fig0_price_timeseries.png")


def fig_distribution(root):
    h = pd.read_csv(os.path.join(root, "data", "processed", "ercot_hb_north_hourly.csv"))
    p = h.loc[h["price"] > 0, "price"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    bins = np.logspace(0, np.log10(5000), 60)
    ax.hist(p, bins=bins, color=COLOR["lgbm"], edgecolor=SURFACE, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["$1", "$10", "$100", "$1,000"])
    for q, lab in ((0.5, "median"), (0.99, "99th pct")):
        v = h["price"].quantile(q)
        ax.axvline(v, color=INK, linewidth=1, linestyle="--")
        ax.text(v * 1.08, ax.get_ylim()[1] * 0.92, f"{lab} ${v:,.0f}", color=INK, fontsize=8.5)
    ax.set_xlabel("Hourly day-ahead price ($/MWh, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("Hours", color=INK2, fontsize=9)
    title(ax, r"Most hours cost \$10-50, but a long tail reaches \$4,200",
          f"35,064 hours, 2022-2025; {int((h['price'] <= 0).sum())} hours at or below " + r"\$0 not shown")
    save(fig, root, "fig0b_price_distribution.png")


def fig_hockey(root):
    h = pd.read_csv(os.path.join(root, "data", "processed", "ercot_hb_north_hourly.csv"))
    h = h[h["price"] > 0]
    gw = h["load_ercot"] / 1000
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax)
    ax.scatter(gw, h["price"], s=3, color=COLOR["lgbm"], alpha=0.12, linewidths=0)
    bins = np.arange(30, 88, 2)
    med = h.groupby(pd.cut(gw, bins), observed=True)["price"].median()
    mids = [b.mid for b in med.index]
    ax.plot(mids, med.values, color=INK, linewidth=2)
    ax.text(mids[-1] + 0.5, med.values[-1], "median", color=INK, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_ylim(1, 6000)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["$1", "$10", "$100", "$1,000"])
    ax.set_xlabel("ERCOT load (GW)", color=INK2, fontsize=9)
    ax.set_ylabel("HB_NORTH day-ahead price ($/MWh, log scale)", color=INK2, fontsize=9)
    title(ax, "Prices stay flat until load nears the system peak, then spike",
          "Hourly, 2022-2025; each dot is one hour")
    save(fig, root, "fig1_hockey_stick.png")


def fig_capture(root):
    s = pd.read_csv(os.path.join(root, "results", "backtest_summary_base.csv"))
    order = ["perfect", "ensemble", "lgbm", "lasso", "xgb", "lstm", "naive"]
    piv = s.pivot(index="forecast", columns="method", values="capture_%").reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    style(ax, grid_axis="x")
    y = np.arange(len(order))[::-1]
    hgt = 0.36
    for k, (m, off) in enumerate((("milp", hgt / 2 + 0.02), ("heuristic", -hgt / 2 - 0.02))):
        vals = piv[m].to_numpy()
        ax.barh(y + off, vals, height=hgt, color=COLOR[m], edgecolor=SURFACE, linewidth=2,
                label="MILP optimization" if m == "milp" else "Simple rule (2 cheapest / 2 dearest hours)")
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[o] for o in order], color=INK, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("2025 profit as % of perfect-foresight MILP", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.45, -0.13), ncol=2)
    title(ax, "Optimization adds more than a perfect forecast does",
          "Even with perfect prices the simple rule earns 75%, below naive forecast + MILP (81%)")
    save(fig, root, "fig2_capture_by_model.png")


def fig_error_vs_profit(root):
    d = pd.read_csv(os.path.join(root, "results", "backtest_daily_base.csv"))
    d = d[(d["method"] == "milp") & (d["forecast"] != "perfect")]
    perfect = pd.read_csv(os.path.join(root, "results", "backtest_daily_base.csv"))
    perfect = perfect[(perfect["method"] == "milp") & (perfect["forecast"] == "perfect")]
    d = d.merge(perfect[["date", "profit"]].rename(columns={"profit": "pf"}), on="date")
    d = d[d["pf"] >= 1000]
    d["regret_pct"] = (100 * d["regret"] / d["pf"]).clip(upper=150)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE, sharey=True)
    panels = [("spearman", "Hour-ranking accuracy (daily Spearman)", np.linspace(-0.2, 1, 13)),
              ("mae", "Forecast error (daily MAE, $/MWh)", np.array([0, 2, 4, 6, 8, 10, 13, 17, 25, 40, 80]))]
    for ax, (col, label, bins) in zip(axes, panels):
        style(ax)
        x = d[col].clip(lower=bins[0], upper=bins[-1])
        ax.scatter(x, d["regret_pct"], s=4, color=COLOR["lgbm"], alpha=0.15, linewidths=0)
        grp = d.groupby(pd.cut(x, bins), observed=True)["regret_pct"]
        med = grp.median()[grp.size() >= 30]
        ax.plot([b.mid for b in med.index], med.values, color=INK, linewidth=2, marker="o",
                markersize=4)
        ax.set_xlabel(label, color=INK2, fontsize=9)
        r = d[col].corr(d["regret_pct"], method="spearman")
        ax.text(0.98, 0.95, f"rank corr. {r:+.2f}", transform=ax.transAxes, ha="right",
                color=INK, fontsize=9)
    axes[0].set_ylabel("Profit left on the table (% of perfect-foresight day)", color=INK2, fontsize=9)
    axes[0].set_title("Ranking the hours right matters more than a low error\n", loc="left",
                      color=INK, fontsize=11)
    axes[0].text(0, 1.02, "Each dot is one model-day in 2025 (6 forecasts); line = median",
                 transform=axes[0].transAxes, color=INK2, fontsize=9, va="bottom")
    save(fig, root, "fig3_error_vs_profit.png")


def fig_example(root):
    s = pd.read_csv(os.path.join(root, "results", "schedules_base.csv"))
    s = s[s["date"] == EXAMPLE_DAY]
    daily = pd.read_csv(os.path.join(root, "results", "backtest_daily_base.csv"))
    p = daily[(daily["date"] == EXAMPLE_DAY) & (daily["method"] == "milp")].set_index("forecast")["profit"]
    hours = np.arange(1, 25)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.4), facecolor=SURFACE, sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
    style(a1)
    act = s[s["forecast"] == "perfect"].set_index("he")["actual"]
    a1.plot(hours, act, color=COLOR["actual"], linewidth=2.4, label="Actual price")
    for f in ("lasso", "naive"):
        a1.plot(hours, s[s["forecast"] == f].set_index("he")["price_used"], color=COLOR[f],
                linewidth=2, linestyle="--", label=f"{NAME[f]} forecast")
    a1.set_ylabel("$/MWh", color=INK2, fontsize=9)
    a1.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    a1.set_title(f"One day, three schedules: timing beats price level ({EXAMPLE_DAY})\n",
                 loc="left", color=INK, fontsize=11)
    a1.text(0, 1.02, "Top: prices. Bottom: MILP schedule from each price series "
            "(up = discharge/sell, down = charge/buy)", transform=a1.transAxes, color=INK2,
            fontsize=9, va="bottom")
    style(a2)
    w = 0.27
    for k, f in enumerate(("perfect", "lasso", "naive")):
        g = s[s["forecast"] == f].set_index("he")
        a2.bar(hours + (k - 1) * w, g["discharge"] - g["charge"], width=w, color=COLOR[f],
               edgecolor=SURFACE, linewidth=0.5, label=f"{NAME[f]}: ${p[f]:,.0f}")
    a2.axhline(0, color=INK2, linewidth=0.8)
    a2.set_ylabel("Battery power (MW)", color=INK2, fontsize=9)
    a2.set_xlabel("Hour ending (Central time)", color=INK2, fontsize=9)
    a2.set_xticks(hours[::2])
    a2.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3)
    save(fig, root, "fig4_example_day.png")


def fig_monthly(root):
    d = pd.read_csv(os.path.join(root, "results", "backtest_daily_base.csv"), parse_dates=["date"])
    d = d[(d["method"] == "milp") & d["forecast"].isin(["perfect", "ensemble"])]
    m = d.groupby([d["date"].dt.month, "forecast"])["profit"].sum().unstack() / 1000
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    style(ax)
    x = np.arange(1, 13)
    w = 0.38
    for k, f in enumerate(("perfect", "ensemble")):
        ax.bar(x + (k - 0.5) * w, m[f], width=w, color=COLOR[f], edgecolor=SURFACE, linewidth=2,
               label=NAME[f])
    for xi, (a, b) in zip(x, zip(m["perfect"], m["ensemble"])):
        ax.text(xi + w / 2, b + 4, f"{100 * b / a:.0f}%", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec"])
    ax.set_ylabel("Monthly profit ($ thousand)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    title(ax, "Arbitrage profit peaks in spring and late summer",
          "2025, 100 MW / 200 MWh battery; labels = ensemble's share of perfect foresight")
    save(fig, root, "fig5_monthly_profit.png")


def fig_sensitivity(root):
    t = pd.read_csv(os.path.join(root, "results", "sensitivity_table.csv"))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURFACE)
    style(ax, grid_axis="x")
    y = np.arange(len(t))[::-1]
    for f in ("ensemble", "lgbm", "lasso", "naive"):
        ax.scatter(t[f"{f}_capture_%"], y, s=60, color=COLOR[f], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=NAME[f])
    for yi, (lo, hi) in zip(y, zip(t[["ensemble_capture_%", "lgbm_capture_%", "lasso_capture_%", "naive_capture_%"]].min(axis=1),
                                   t[["ensemble_capture_%", "lgbm_capture_%", "lasso_capture_%", "naive_capture_%"]].max(axis=1))):
        ax.plot([lo, hi], [yi, yi], color=GRID, linewidth=2, zorder=2)
    labels = [f"{s}  (${p:.1f}M max)".replace("$", r"\$") for s, p in zip(t["scenario"], t["perfect_profit_$M"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_xlim(25, 100)
    ax.set_xlabel("2025 profit as % of perfect foresight (MILP)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    title(ax, "The model ranking holds across battery assumptions",
          "Naive trails LASSO, LightGBM and the ensemble in every scenario; higher wear cost widens the gaps")
    save(fig, root, "fig6_sensitivity.png")


def main(root):
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    for f in (fig_timeseries, fig_distribution, fig_hockey, fig_capture, fig_error_vs_profit, fig_example, fig_monthly, fig_sensitivity):
        f(root)
        print("done", f.__name__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
