"""XGBoost price forecast for 2025 - a robustness check on the LightGBM result.

Same 29 features and asinh(price) target as LightGBM; small grid tuned on 2024 with early
stopping, refit on 2022-2024, predict 2025.

Usage: python -P src/forecast_xgb.py <project_root>
Writes results/forecast_xgb.csv
"""

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from forecast_lgbm import FEATURES  # noqa: E402


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def main(root):
    df = pd.read_csv(os.path.join(root, "data", "processed", "features.csv"))
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    base = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                random_state=0, n_jobs=-1)
    t0 = time.time()
    best = None
    for depth, mcw in itertools.product([4, 6, 8], [1, 10]):
        m = xgb.XGBRegressor(max_depth=depth, min_child_weight=mcw, n_estimators=4000,
                             early_stopping_rounds=200, **base)
        m.fit(tr[FEATURES], tr["price_asinh"], eval_set=[(va[FEATURES], va["price_asinh"])],
              verbose=False)
        score = mae(va["price"], np.sinh(m.predict(va[FEATURES])))
        print(f"  depth {depth}  min_child_weight {mcw:2d}  trees {m.best_iteration + 1:4d}  val MAE {score:.2f}")
        if best is None or score < best[0]:
            best = (score, depth, mcw, m.best_iteration + 1)
    score, depth, mcw, n_iter = best
    print(f"best: depth {depth}, min_child_weight {mcw}, {n_iter} trees, val MAE {score:.2f} "
          f"(tuning took {time.time() - t0:.0f}s)")

    trva = pd.concat([tr, va])
    final = xgb.XGBRegressor(max_depth=depth, min_child_weight=mcw,
                             n_estimators=int(n_iter * len(trva) / len(tr)), **base)
    final.fit(trva[FEATURES], trva["price_asinh"], verbose=False)
    pred = np.sinh(final.predict(te[FEATURES]))
    err = pred - te["price"].to_numpy()
    print(f"test MAE {np.mean(np.abs(err)):.2f}  RMSE {np.sqrt(np.mean(err ** 2)):.2f}")
    pd.DataFrame({"date": te["date"], "he": te["he"], "pred": pred}).to_csv(
        os.path.join(root, "results", "forecast_xgb.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
