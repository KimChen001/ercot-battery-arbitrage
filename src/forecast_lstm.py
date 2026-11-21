"""LSTM day-ahead price forecast for 2025, and the LightGBM + LSTM ensemble.

One sample per operating day D:
  sequence  = the 7 previous days, each step one day with 48 inputs:
              24 hourly asinh prices of that day + 24 hourly loads of the day before it
              (so the last step uses prices through D-1 and load through D-2);
  static    = D's 24-hour temperature forecast (48 h ahead) + day of week, holiday,
              day-of-year sin/cos;
  output    = D's 24 hourly asinh prices.
Architecture: LSTM(48 -> 64) on the sequence, its last state joined with the static
inputs, then a small MLP to 24 outputs. Scalers are fit on 2022-2023. Early stopping on
2024 picks the epoch count; three seeds are refit on 2022-2024 and averaged for 2025.

Usage: python -P src/forecast_lstm.py <project_root>
Writes results/forecast_lstm.csv and results/forecast_ensemble.csv (mean of LightGBM, LSTM)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
from features import to_day_matrix  # noqa: E402

SEQ_DAYS, HIDDEN, SEEDS = 7, 64, (0, 1, 2)
torch.set_num_threads(max(1, os.cpu_count() or 1))


class PriceLSTM(nn.Module):
    def __init__(self, n_seq, n_static):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN + n_static, 64), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(64, 24))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], dim=1))


def build_samples(root, hub="HB_NORTH"):
    hourly = pd.read_csv(os.path.join(root, "data", "processed", f"ercot_{hub.lower()}_hourly.csv"),
                         parse_dates=["date_local"])
    hourly = hourly[~hourly["repeated"]]
    P = np.arcsinh(to_day_matrix(hourly, "price"))
    L = to_day_matrix(hourly, "load_ercot") / 1e4
    feats = pd.read_csv(os.path.join(root, "data", "processed", "features.csv"),
                        parse_dates=["date"])
    TF = feats.pivot(index="date", columns="he", values="temp_fc")
    days = [d for d in TF.index if d - pd.Timedelta(days=SEQ_DAYS + 1) >= P.index[0]]

    seq, static, target, split = [], [], [], []
    for d in days:
        steps = [d - pd.Timedelta(days=k) for k in range(SEQ_DAYS, 0, -1)]
        x = [np.concatenate([P.loc[s].to_numpy(), L.loc[s - pd.Timedelta(days=1)].to_numpy()])
             for s in steps]
        cal = np.zeros(7); cal[d.dayofweek] = 1
        doy = 2 * np.pi * d.dayofyear / 365.25
        hol = feats.loc[(feats["date"] == d) & (feats["he"] == 1), "is_holiday"].iloc[0]
        seq.append(np.stack(x))
        static.append(np.concatenate([TF.loc[d].to_numpy(), cal, [hol, np.sin(doy), np.cos(doy)]]))
        target.append(P.loc[d].to_numpy())
        split.append("train" if d.year <= 2023 else ("val" if d.year == 2024 else "test"))
    return (np.array(days), np.array(seq, dtype=np.float32), np.array(static, dtype=np.float32),
            np.array(target, dtype=np.float32), np.array(split))


class Scaler:
    def fit(self, a, axis):
        self.mu, self.sd = a.mean(axis=axis, keepdims=True), a.std(axis=axis, keepdims=True) + 1e-6
        return self

    def __call__(self, a):
        return (a - self.mu) / self.sd


def train(seq, static, y, seed, epochs, val=None, patience=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PriceLSTM(seq.shape[2], static.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    X1, X2, Y = map(torch.from_numpy, (seq, static, y))
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Y))
        for i in range(0, len(Y), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = nn.functional.huber_loss(model(X1[b], X2[b]), Y[b])
            loss.backward(); opt.step()
        if val is not None:
            score = val(model)
            if score < best:
                best, best_ep, wait = score, ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best


def predict(model, seq, static):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(seq), torch.from_numpy(static)).numpy()


def main(root, hub="HB_NORTH"):
    t0 = time.time()
    days, seq, static, y, split = build_samples(root, hub)
    tr, va, te = (split == s for s in ("train", "val", "test"))
    print(f"samples: train {tr.sum()}, val {va.sum()}, test {te.sum()} days")

    def scalers(mask):
        return (Scaler().fit(seq[mask], axis=(0, 1)), Scaler().fit(static[mask], axis=0),
                Scaler().fit(y[mask], axis=0))

    s_seq, s_st, s_y = scalers(tr)
    actual_va = np.sinh(y[va])

    def val_mae(model):
        pred = s_y.mu + s_y.sd * predict(model, s_seq(seq[va]), s_st(static[va]))
        return float(np.mean(np.abs(np.sinh(pred) - actual_va)))

    epochs = []
    for seed in SEEDS:
        _, ep, score = train(s_seq(seq[tr]), s_st(static[tr]), s_y(y[tr]), seed, 400, val=val_mae)
        epochs.append(ep)
        print(f"  seed {seed}: best epoch {ep}, val MAE {score:.2f}")
    n_ep = int(np.mean(epochs))

    trva = tr | va
    s_seq, s_st, s_y = scalers(trva)
    preds = []
    for seed in SEEDS:
        model, _, _ = train(s_seq(seq[trva]), s_st(static[trva]), s_y(y[trva]), seed, n_ep)
        preds.append(s_y.mu + s_y.sd * predict(model, s_seq(seq[te]), s_st(static[te])))
    pred = np.sinh(np.mean(preds, axis=0))
    err = pred - np.sinh(y[te])
    print(f"refit on 2022-2024 for {n_ep} epochs x {len(SEEDS)} seeds; "
          f"test MAE {np.mean(np.abs(err)):.2f}  RMSE {np.sqrt(np.mean(err ** 2)):.2f} "
          f"(total {time.time() - t0:.0f}s)")

    res = os.path.join(root, "results")
    out = pd.DataFrame({"date": np.repeat(pd.to_datetime(days[te]).strftime("%Y-%m-%d"), 24),
                        "he": np.tile(np.arange(1, 25), te.sum()), "pred": pred.ravel()})
    out.to_csv(os.path.join(res, "forecast_lstm.csv"), index=False)

    lgbm = pd.read_csv(os.path.join(res, "forecast_lgbm.csv"))
    ens = lgbm.merge(out, on=["date", "he"], suffixes=("_lgbm", "_lstm"))
    ens["pred"] = (ens["pred_lgbm"] + ens["pred_lstm"]) / 2
    ens[["date", "he", "pred"]].to_csv(os.path.join(res, "forecast_ensemble.csv"), index=False)
    feats = pd.read_csv(os.path.join(root, "data", "processed", "features.csv"))
    act = feats[feats["split"] == "test"][["date", "he", "price"]].merge(ens, on=["date", "he"])
    print(f"ensemble (LightGBM + LSTM) test MAE {np.mean(np.abs(act['pred'] - act['price'])):.2f}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    hub = sys.argv[sys.argv.index("--hub") + 1] if "--hub" in sys.argv else "HB_NORTH"
    main(root, hub)
