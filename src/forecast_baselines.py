"""Naive and LASSO price forecasts for the 2025 test year.

Naive: the standard electricity-price benchmark - same hour last week on Mondays,
Saturdays and Sundays, same hour yesterday on Tuesday to Friday.

LASSO: linear regression with an L1 penalty on the 27 standardized features (price
features on the asinh scale), predicting asinh(price). The penalty is tuned on 2024
after training on 2022-2023, then the model is refit on 2022-2024 and used for 2025.

Usage: python -P src/forecast_baselines.py <project_root>
Writes results/forecast_naive.csv, results/forecast_lasso.csv,
       results/lasso_coefficients.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from pca_analysis import FEATURES, PRICE_COLS  # same 27 features as the PCA


def design(df):
    X = df[FEATURES].copy()
    X[PRICE_COLS] = np.arcsinh(X[PRICE_COLS])
    return X


def errors(y, yhat):
    e = np.asarray(yhat) - np.asarray(y)
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))


def main(root):
    df = pd.read_csv(os.path.join(root, "data", "processed", "features.csv"))
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))

    # ---- naive ----
    use_week = df["dow"].isin([0, 5, 6])
    df["naive"] = np.where(use_week, df["price_lag7d"], df["price_lag1d"])
    for name, part in (("val", va), ("test", te)):
        mae, rmse = errors(part["price"], df.loc[part.index, "naive"])
        print(f"naive  {name}: MAE {mae:6.2f}  RMSE {rmse:7.2f}")
    df.loc[te.index, ["date", "he", "naive"]].rename(columns={"naive": "pred"}).to_csv(
        os.path.join(out_dir, "forecast_naive.csv"), index=False)

    # ---- LASSO: tune alpha on 2024 ----
    best = None
    for alpha in np.logspace(-4, -0.5, 15):
        m = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))
        m.fit(design(tr), tr["price_asinh"])
        mae, _ = errors(va["price"], np.sinh(m.predict(design(va))))
        if best is None or mae < best[1]:
            best = (alpha, mae)
    alpha = best[0]
    print(f"LASSO best alpha {alpha:.5f} (val MAE {best[1]:.2f})")

    trva = pd.concat([tr, va])
    m = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))
    m.fit(design(trva), trva["price_asinh"])
    pred = np.sinh(m.predict(design(te)))
    mae, rmse = errors(te["price"], pred)
    print(f"lasso  test: MAE {mae:6.2f}  RMSE {rmse:7.2f}")
    pd.DataFrame({"date": te["date"], "he": te["he"], "pred": pred}).to_csv(
        os.path.join(out_dir, "forecast_lasso.csv"), index=False)

    coef = pd.Series(m[-1].coef_, index=FEATURES)
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
    coef.rename("coef_std_asinh").to_csv(os.path.join(out_dir, "lasso_coefficients.csv"))
    print(f"non-zero coefficients: {(coef != 0).sum()} of {len(coef)}")
    print(coef.head(8).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
