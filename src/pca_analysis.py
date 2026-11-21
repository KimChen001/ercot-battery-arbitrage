"""PCA of the forecasting features, fit on the 2022-2023 training rows only.

Price features are heavy-tailed (spikes up to $4,200/MWh), so they enter on the same
inverse hyperbolic sine scale we use for the target; then every feature is standardized.
Raw `dow` and `month` integers are left out (calendar enters through the sin/cos and
0/1 columns instead).

Usage: python -P src/pca_analysis.py <project_root>
Outputs: figures/pca_scree.png, figures/pca_loadings.png, data/processed/pca_summary.txt
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GROUPS = {
    "Price history": ["price_lag1d", "price_lag2d", "price_lag7d", "price_samehour_7d_mean",
                      "price_d1_mean", "price_d1_max", "price_d1_min", "price_d1_spread",
                      "price_7d_std"],
    "Load (through D-2)": ["load_lag2d", "load_lag7d", "load_d2_mean", "load_d2_max"],
    "Weather forecast": ["temp_fc", "cooling_deg", "heating_deg", "temp_fc_daymax",
                         "temp_fc_daymin"],
    "Observed weather (D-2)": ["wind_obs_lag2d", "solar_obs_lag2d", "solar_obs_d2_mean"],
    "Calendar": ["hour_sin", "hour_cos", "is_weekend", "is_holiday", "doy_sin", "doy_cos"],
}
FEATURES = [c for cols in GROUPS.values() for c in cols]
PRICE_COLS = GROUPS["Price history"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a3a2", "#c43a39"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def main(root):
    proc = os.path.join(root, "data", "processed")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(proc, "features.csv"))
    train = df[df["split"] == "train"].copy()
    X = train[FEATURES].copy()
    X[PRICE_COLS] = np.arcsinh(X[PRICE_COLS])
    Z = StandardScaler().fit_transform(X)
    pca = PCA().fit(Z)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    scores = pca.transform(Z)
    y = train["price_asinh"].to_numpy()

    n80, n90, n95 = (int(np.searchsorted(cum, t) + 1) for t in (0.80, 0.90, 0.95))
    k = 6
    load = pd.DataFrame(pca.components_[:k].T, index=FEATURES,
                        columns=[f"PC{i + 1}" for i in range(k)])

    lines = [f"Training rows: {len(train):,}   features: {len(FEATURES)}",
             f"Components for 80% / 90% / 95% of variance: {n80} / {n90} / {n95}",
             "", "PC   var%   cum%   corr with asinh(price)   top loadings"]
    for i in range(10):
        r = np.corrcoef(scores[:, i], y)[0, 1]
        top = pd.Series(pca.components_[i], index=FEATURES).abs().nlargest(4).index
        tops = ", ".join(f"{f}({pca.components_[i][FEATURES.index(f)]:+.2f})" for f in top)
        lines.append(f"PC{i + 1:<3d}{100 * ev[i]:5.1f}  {100 * cum[i]:5.1f}   {r:+.2f}"
                     f"                     {tops}")
    raw_r = pd.Series({f: np.corrcoef(X[f], y)[0, 1] for f in FEATURES})
    lines += ["", "Strongest single-feature correlations with asinh(price):"]
    lines += [f"  {f:24s} {v:+.2f}" for f, v in raw_r.reindex(
        raw_r.abs().sort_values(ascending=False).index).head(8).items()]
    report = "\n".join(lines)
    with open(os.path.join(proc, "pca_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)

    # ---- scree: individual and cumulative share, one % axis ----
    m = 15
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    x = np.arange(1, m + 1)
    ax.bar(x, 100 * ev[:m], width=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2,
           label="Share of variance")
    ax.plot(x, 100 * cum[:m], color=INK2, linewidth=2, marker="o", markersize=5,
            label="Cumulative share")
    for t in (80, 90):
        ax.axhline(t, color=GRID, linewidth=1, zorder=0)
    ax.annotate(f"{n90} components reach 90%", xy=(n90, 100 * cum[n90 - 1]),
                xytext=(n90 + 1.2, 100 * cum[n90 - 1] - 14), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xticks(x)
    ax.set_xlabel("Principal component", color=INK2, fontsize=9)
    ax.set_ylabel("Variance explained (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("PC1 alone explains "
                 f"{100 * ev[0]:.0f}% of feature variance; {n90} PCs reach 90%",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pca_scree.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ---- loadings heatmap, features grouped ----
    fig, ax = plt.subplots(figsize=(7.2, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lim = np.abs(load.values).max()
    im = ax.imshow(load.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"PC{i + 1}\n{100 * ev[i]:.0f}%" for i in range(k)],
                       color=INK2, fontsize=9)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES, color=INK, fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(FEATURES)):
        for j in range(k):
            v = load.values[i, j]
            if abs(v) >= 0.25:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(v) > 0.6 * lim else INK)
    short = {"Price history": "Price\nhistory", "Load (through D-2)": "Load\n(to D-2)",
             "Weather forecast": "Temp.\nforecast", "Observed weather (D-2)": "Obs.\nweather\n(D-2)",
             "Calendar": "Calendar"}
    row = 0
    for name, cols in GROUPS.items():
        if row:
            ax.axhline(row - 0.5, color=SURFACE, linewidth=3)
        ax.text(k - 0.4, row + len(cols) / 2 - 0.5, short[name], va="center",
                ha="left", fontsize=8.5, color=INK2)
        row += len(cols)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.16)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.set_label("Loading", color=INK2, fontsize=9)
    ax.set_title("PC1 = heat and load, PC2 = recent prices\n"
                 "Loadings of the first six components (|value| >= 0.25 labeled)",
                 loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pca_loadings.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
