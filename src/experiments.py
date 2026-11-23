"""Extra experiments on the 2025 test year (base battery, MILP decisions only).

1. Feature-group ablation: LightGBM retrained without each group of variables, to put a
   dollar value on each data source.
2. Principal-component regression (PCR): linear regression on the first 10 principal
   components (90% of feature variance) vs. LASSO on all 27 features.
3. Renewable forecasts: day-before (48 h) wind and solar forecasts for the West Texas wind
   belt (32.47 N, 100.41 W) exist only from 2024, so two LightGBM models are trained on
   2024 alone - with and without them - and compared on 2025.
4. No validation: train on 2022-2024 with untuned (library default / guessed) settings and
   test on 2025, vs. our settings tuned on 2024.

Usage: python -P src/experiments.py <project_root>
Writes results/experiment_ablation.csv, experiment_pcr.csv, experiment_renewables.csv,
experiment_no_validation.csv and figures/fig7_ablation.png
"""

import os
import sys
import warnings

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from battery import Battery, profit, solve_milp  # noqa: E402
from features import to_day_matrix  # noqa: E402
from forecast_baselines import design as lasso_design  # noqa: E402
from forecast_lgbm import FEATURES  # noqa: E402
import forecast_lstm  # noqa: E402
from pca_analysis import FEATURES as PCA_FEATURES, PRICE_COLS  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
# a flat forecast (untuned LASSO) has no hour ranking; its Spearman is reported as NaN
warnings.filterwarnings("ignore", category=RuntimeWarning)
GROUPS = {
    "price history": FEATURES[0:9],
    "load": FEATURES[9:13],
    "temperature forecast": FEATURES[13:18],
    "observed weather": FEATURES[18:21],
    "calendar": FEATURES[21:29],
}
LGB_PARAMS = dict(num_leaves=63, min_child_samples=100, n_estimators=486, learning_rate=0.03,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                  random_state=0)  # the setting tuned in forecast_lgbm.py
SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


class Evaluator:
    """Scores a 2025 forecast: accuracy plus MILP profit as % of perfect foresight."""

    def __init__(self, root):
        f = pd.read_csv(os.path.join(root, "data", "processed", "features.csv"))
        te = f[f["split"] == "test"]
        self.keys = te[["date", "he"]].reset_index(drop=True)
        self.actual = te.pivot(index="date", columns="he", values="price").sort_index()
        d = pd.read_csv(os.path.join(root, "results", "backtest_daily_base.csv"))
        pf = d[(d["forecast"] == "perfect") & (d["method"] == "milp")]
        self.pf_total = pf["profit"].sum()
        self.bat = Battery()

    def score(self, pred):
        m = self.keys.assign(pred=np.asarray(pred)).pivot(index="date", columns="he",
                                                          values="pred").loc[self.actual.index]
        prof, rho = 0.0, []
        for day in self.actual.index:
            a, f = self.actual.loc[day].to_numpy(), m.loc[day].to_numpy()
            c, dch, _ = solve_milp(f, self.bat)
            prof += profit(a, c, dch, self.bat)
            rho.append(spearmanr(f, a).statistic)
        err = m.to_numpy() - self.actual.to_numpy()
        return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                "spearman": float(np.nanmean(rho)), "profit_$": prof,
                "capture_%": 100 * prof / self.pf_total}


def ablation(df, ev):
    trva, te = df[df["split"] != "test"], df[df["split"] == "test"]
    variants = {"All 29 features": FEATURES}
    for g, cols in GROUPS.items():
        variants[f"Without {g}"] = [c for c in FEATURES if c not in cols]
    variants["Price history only"] = GROUPS["price history"]
    variants["Price history + calendar"] = GROUPS["price history"] + GROUPS["calendar"]
    rows = []
    for name, cols in variants.items():
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(trva[cols], trva["price_asinh"])
        r = ev.score(np.sinh(m.predict(te[cols])))
        rows.append({"variant": name, "n_features": len(cols), **r})
        print(f"  {name:32s} MAE {r['MAE']:6.2f}  capture {r['capture_%']:5.1f}%")
    out = pd.DataFrame(rows)
    full = out.loc[0, "profit_$"]
    out["profit_change_vs_all_$"] = out["profit_$"] - full
    return out


def pcr(df, ev, k=10):
    trva, te = df[df["split"] != "test"], df[df["split"] == "test"]

    def design(part):
        X = part[PCA_FEATURES].copy()
        X[PRICE_COLS] = np.arcsinh(X[PRICE_COLS])
        return X

    m = make_pipeline(StandardScaler(), PCA(n_components=k), LinearRegression())
    m.fit(design(trva), trva["price_asinh"])
    r = ev.score(np.sinh(m.predict(design(te))))
    var = m[1].explained_variance_ratio_.sum()
    print(f"  PCR ({k} components, {100 * var:.0f}% of variance): MAE {r['MAE']:.2f}, capture {r['capture_%']:.1f}%")
    return pd.DataFrame([{"model": f"PCR, {k} principal components", "variance_kept_%": 100 * var, **r}])


def renewables(root, df, ev):
    hourly = pd.read_csv(os.path.join(root, "data", "processed", "ercot_hb_north_hourly.csv"),
                         parse_dates=["date_local"])
    hourly["ts_utc"] = pd.to_datetime(hourly["ts_utc"], utc=True)
    hourly = hourly[~hourly["repeated"]]
    rn = pd.read_csv(os.path.join(root, "data", "raw", "weather",
                                  "westtx_renew_prev_day2_2024_2025.csv"), skiprows=3)
    rn.columns = ["time", "wind_fc_w", "solar_fc_w"]
    rn["ts_utc"] = pd.to_datetime(rn["time"]).dt.tz_localize("UTC") - pd.Timedelta(hours=1)
    hourly = hourly.merge(rn[["ts_utc", "wind_fc_w", "solar_fc_w"]], on="ts_utc", how="left")
    extra = {}
    for col in ("wind_fc_w", "solar_fc_w"):
        mat = to_day_matrix(hourly, col)
        s = mat.stack(future_stack=True).rename(col)
        s.index = s.index.set_names(["date", "he"])
        extra[col] = s
        extra[col + "_daymean"] = s.groupby(level=0).transform("mean").rename(col + "_daymean")
    ext = pd.concat(extra.values(), axis=1).reset_index()
    ext["date"] = ext["date"].dt.strftime("%Y-%m-%d")
    d = df.merge(ext, on=["date", "he"], how="left")
    tr, te = d[d["split"] == "val"], d[d["split"] == "test"]  # train on 2024 only
    new = list(extra)
    params = {**LGB_PARAMS, "num_leaves": 31, "min_child_samples": 50, "n_estimators": 400}
    rows = []
    for name, cols in (("2024-only, base 29 features", FEATURES),
                       ("2024-only, + West Texas wind/solar forecasts", FEATURES + new)):
        m = lgb.LGBMRegressor(**params).fit(tr[cols], tr["price_asinh"])
        r = ev.score(np.sinh(m.predict(te[cols])))
        rows.append({"model": name, **r})
        print(f"  {name:46s} MAE {r['MAE']:6.2f}  capture {r['capture_%']:5.1f}%")
    return pd.DataFrame(rows)


def no_validation(root, df, ev):
    """Plan A (no tuning year, untuned settings) vs plan B (settings tuned on 2024)."""
    trva, te = df[df["split"] != "test"], df[df["split"] == "test"]
    rows = []

    def add(model, plan, setting, pred):
        r = ev.score(pred)
        rows.append({"model": model, "plan": plan, "setting": setting, **r})
        print(f"  {model:9s} {plan:18s} {setting:42s} MAE {r['MAE']:6.2f}  capture {r['capture_%']:5.1f}%")

    for plan, setting, params in (
            ("A: no validation", "library defaults (31 leaves, 100 trees)",
             dict(verbose=-1, random_state=0)),
            ("B: tuned on 2024", "63 leaves, 486 trees", LGB_PARAMS)):
        m = lgb.LGBMRegressor(**params).fit(trva[FEATURES], trva["price_asinh"])
        add("LightGBM", plan, setting, np.sinh(m.predict(te[FEATURES])))

    for plan, setting, alpha in (("A: no validation", "library default penalty 1.0", 1.0),
                                 ("B: tuned on 2024", "penalty 0.0001", 1e-4)):
        m = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))
        m.fit(lasso_design(trva), trva["price_asinh"])
        add("LASSO", plan, setting, np.sinh(m.predict(lasso_design(te))))

    L = forecast_lstm
    days, seq, static, y, split = L.build_samples(root)
    fit, tst = split != "test", split == "test"
    s_seq = L.Scaler().fit(seq[fit], axis=(0, 1))
    s_st, s_y = L.Scaler().fit(static[fit], axis=0), L.Scaler().fit(y[fit], axis=0)
    for plan, setting, epochs in (("A: no validation", "guessed 100 epochs", 100),
                                  ("B: tuned on 2024", "15 epochs (early stopping on 2024)", 15)):
        preds = []
        for seed in L.SEEDS:
            model, _, _ = L.train(s_seq(seq[fit]), s_st(static[fit]), s_y(y[fit]), seed, epochs)
            preds.append(s_y.mu + s_y.sd * L.predict(model, s_seq(seq[tst]), s_st(static[tst])))
        p = np.sinh(np.mean(preds, axis=0))
        out = pd.DataFrame({"date": np.repeat(pd.to_datetime(days[tst]).strftime("%Y-%m-%d"), 24),
                            "he": np.tile(np.arange(1, 25), tst.sum()), "pred": p.ravel()})
        add("LSTM", plan, setting, ev.keys.merge(out, on=["date", "he"], how="left")["pred"].to_numpy())
    return pd.DataFrame(rows)


def plot_ablation(abl, root):
    t = abl.iloc[::-1].reset_index(drop=True)
    full = abl.loc[0, "capture_%"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for y, row in t.iterrows():
        v, ch = row["capture_%"], row["profit_change_vs_all_$"]
        is_full = row["variant"] == "All 29 features"
        if not is_full:
            ax.plot([full, v], [y, y], color=GRID, linewidth=2.5, zorder=1)
        ax.scatter(v, y, s=70, color=INK2 if is_full else BLUE, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        lab = f"{v:.1f}%" + ("" if is_full else f"  ({ch / 1000:+,.0f}k)")
        ax.text(max(v, full) + 0.5, y, lab, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["variant"], color=INK, fontsize=9)
    ax.set_xlim(70, 96)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlabel("2025 profit as % of perfect foresight (LightGBM + MILP); "
                  "change vs. all features in brackets", color=INK2, fontsize=8.5)
    ax.set_title("Recent prices carry most of the value; calendar and temperature add the rest\n",
                 loc="left", color=INK, fontsize=11)
    ax.text(0, 1.02, "Dropping load or observed weather changes profit by under 1 point "
            "(within noise)", transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(root, "figures", "fig7_ablation.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(root):
    df = pd.read_csv(os.path.join(root, "data", "processed", "features.csv"))
    ev = Evaluator(root)
    res = os.path.join(root, "results")
    print("1) Feature-group ablation")
    abl = ablation(df, ev)
    abl.to_csv(os.path.join(res, "experiment_ablation.csv"), index=False)
    plot_ablation(abl, root)
    print("2) Principal-component regression")
    pcr(df, ev).to_csv(os.path.join(res, "experiment_pcr.csv"), index=False)
    print("3) Renewable forecasts (train 2024, test 2025)")
    renewables(root, df, ev).to_csv(os.path.join(res, "experiment_renewables.csv"), index=False)
    print("4) No validation year vs settings tuned on 2024")
    no_validation(root, df, ev).to_csv(os.path.join(res, "experiment_no_validation.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
