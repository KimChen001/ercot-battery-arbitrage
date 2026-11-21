"""LightGBM price forecast for 2025, plus SHAP feature importance.

Same 29 features as the README, target asinh(price). A small grid is tuned on 2024 with
early stopping (train 2022-2023); the best setting is refit on 2022-2024 and used for 2025.
SHAP values come from LightGBM's built-in TreeSHAP (predict(..., pred_contrib=True)), so
no extra package is needed; they are in asinh(price) units.

Usage: python -P src/forecast_lgbm.py <project_root>
Writes results/forecast_lgbm.csv, results/shap_importance.csv, figures/shap_importance.png
"""

import itertools
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = ["price_lag1d", "price_lag2d", "price_lag7d", "price_samehour_7d_mean",
            "price_d1_mean", "price_d1_max", "price_d1_min", "price_d1_spread", "price_7d_std",
            "load_lag2d", "load_lag7d", "load_d2_mean", "load_d2_max",
            "temp_fc", "cooling_deg", "heating_deg", "temp_fc_daymax", "temp_fc_daymin",
            "wind_obs_lag2d", "solar_obs_lag2d", "solar_obs_d2_mean",
            "hour_sin", "hour_cos", "dow", "is_weekend", "is_holiday", "month",
            "doy_sin", "doy_cos"]
LABELS = {
    "price_lag1d": "Price, same hour yesterday", "price_lag2d": "Price, same hour 2 days ago",
    "price_lag7d": "Price, same hour last week", "price_samehour_7d_mean": "Price, same hour 7-day mean",
    "price_d1_mean": "Yesterday's mean price", "price_d1_max": "Yesterday's max price",
    "price_d1_min": "Yesterday's min price", "price_d1_spread": "Yesterday's price spread",
    "price_7d_std": "Price volatility, 7 days", "load_lag2d": "Load, same hour 2 days ago",
    "load_lag7d": "Load, same hour last week", "load_d2_mean": "Load, mean 2 days ago",
    "load_d2_max": "Load, peak 2 days ago", "temp_fc": "Temperature forecast",
    "cooling_deg": "Cooling degrees (forecast)", "heating_deg": "Heating degrees (forecast)",
    "temp_fc_daymax": "Forecast daily max temp", "temp_fc_daymin": "Forecast daily min temp",
    "wind_obs_lag2d": "Wind, 2 days ago", "solar_obs_lag2d": "Solar, same hour 2 days ago",
    "solar_obs_d2_mean": "Solar, mean 2 days ago", "hour_sin": "Hour of day (sin)",
    "hour_cos": "Hour of day (cos)", "dow": "Day of week", "is_weekend": "Weekend",
    "is_holiday": "Holiday", "month": "Month", "doy_sin": "Day of year (sin)",
    "doy_cos": "Day of year (cos)",
}
SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def main(root):
    df = pd.read_csv(os.path.join(root, "data", "processed", "features.csv"))
    res_dir, fig_dir = os.path.join(root, "results"), os.path.join(root, "figures")
    os.makedirs(res_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))

    t0 = time.time()
    base = dict(objective="regression", learning_rate=0.03, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, n_estimators=4000, verbose=-1, random_state=0)
    best = None
    for leaves, min_child in itertools.product([15, 31, 63], [20, 100]):
        m = lgb.LGBMRegressor(num_leaves=leaves, min_child_samples=min_child, **base)
        m.fit(tr[FEATURES], tr["price_asinh"], eval_set=[(va[FEATURES], va["price_asinh"])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        score = mae(va["price"], np.sinh(m.predict(va[FEATURES])))
        print(f"  leaves {leaves:2d}  min_child {min_child:3d}  trees {m.best_iteration_:4d}  val MAE {score:.2f}")
        if best is None or score < best[0]:
            best = (score, leaves, min_child, m.best_iteration_)
    score, leaves, min_child, n_iter = best
    print(f"best: leaves {leaves}, min_child {min_child}, {n_iter} trees, val MAE {score:.2f} "
          f"(tuning took {time.time() - t0:.0f}s)")

    trva = pd.concat([tr, va])
    n_final = int(n_iter * len(trva) / len(tr))
    final = lgb.LGBMRegressor(num_leaves=leaves, min_child_samples=min_child,
                              **{**base, "n_estimators": n_final})
    t1 = time.time()
    final.fit(trva[FEATURES], trva["price_asinh"])
    pred = np.sinh(final.predict(te[FEATURES]))
    err = pred - te["price"].to_numpy()
    print(f"final fit on 2022-2024 ({n_final} trees) took {time.time() - t1:.1f}s; "
          f"test MAE {np.mean(np.abs(err)):.2f}  RMSE {np.sqrt(np.mean(err ** 2)):.2f}")
    pd.DataFrame({"date": te["date"], "he": te["he"], "pred": pred}).to_csv(
        os.path.join(res_dir, "forecast_lgbm.csv"), index=False)

    # ---- SHAP (TreeSHAP built into LightGBM) on the 2025 test rows ----
    contrib = final.predict(te[FEATURES], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    imp.rename("mean_abs_shap_asinh").to_csv(os.path.join(res_dir, "shap_importance.csv"))
    print("top SHAP features:\n" + imp.head(10).round(3).to_string())

    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([LABELS[f] for f in top.index], top.values, color=BLUE, height=0.6,
            edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=INK)
    ax.set_xlabel("Mean |SHAP value| on 2025 forecasts (asinh price units)", color=INK2, fontsize=9)
    ax.set_title("Yesterday's price and the heat forecast drive LightGBM's predictions",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "shap_importance.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
