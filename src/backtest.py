# src/backtest.py
from __future__ import annotations
import argparse, os, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import add_time_features, make_windows
from model import LSTMRegressor

def rmse(a, b): return mean_squared_error(a, b, squared=False)

def build_features(df: pd.DataFrame, time_col: str, target_col: str, extra_feats):
    df = df.sort_values(time_col).reset_index(drop=True)
    df = add_time_features(df, time_col)
    used = (extra_feats or []) + ["hour_sin","hour_cos","dow_sin","dow_cos"]
    X = df[used].values.astype("float32")
    y = df[target_col].values.astype("float32")
    return df, X, y, used

def backtest(
    csv_path: str, time_col: str, target_col: str, feature_cols,
    lookback: int, horizon: int, epochs: int, batch_size: int, lr: float,
    initial_days: int, step_hours: int, folds: int, hidden_size: int, num_layers: int,
    dropout: float, bidirectional: bool, device: str = None
):
    df = pd.read_csv(csv_path)
    df, X, y, used_features = build_features(df, time_col, target_col, feature_cols)

    H = horizon
    L = lookback
    T = len(df)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # convert days→rows; clamp to allow at least one fold
    init_end = max(L + H, min(T - H - 1, initial_days * 24))

    # Make supervised windows once (then we’ll select masks per fold)
    Xw, Yw = make_windows(X, y, L, H)             # (N, L, F), (N, H)
    end_indices = np.arange(L, L + len(Xw))       # each window "ends" at this absolute index

    fold_results = []
    n_possible_folds = max(0, (T - init_end - H) // step_hours + 1)
    if folds > 0:
        n_folds = min(folds, n_possible_folds)
    else:
        n_folds = n_possible_folds

    if n_folds <= 0:
        raise ValueError("Not enough data for the requested backtest settings.")

    for i in range(n_folds):
        train_end = init_end + i * step_hours          # last observed index available for training
        test_endindex = train_end                      # we test on the window that ends at this index
        # TRAIN/VAL windows must NOT peek into the future (no leakage):
        # end_index + horizon <= train_end  ->  end_index <= train_end - horizon
        tr_mask_all = end_indices <= (train_end - H)
        # hold out the tail of training windows as validation (10%)
        tr_idx = np.where(tr_mask_all)[0]
        if len(tr_idx) < 10:
            raise ValueError("Training window too small; adjust initial_days or lookback.")
        split = int(0.9 * len(tr_idx))
        tr_sel = tr_idx[:split]
        va_sel = tr_idx[split:]

        # TEST window: the one that ends exactly at train_end (predict next H steps)
        te_sel = np.where(end_indices == test_endindex)[0]
        if len(te_sel) == 0:
            # if exact match missing due to step misalignment, pick the closest <= train_end
            te_sel = np.where(end_indices <= test_endindex)[0]
            if len(te_sel) == 0:
                continue
            te_sel = np.array([te_sel[-1]])

        # Fit scalers on ORIGINAL (non-windowed) data up to train_end
        xsc = StandardScaler().fit(X[:train_end])
        ysc = StandardScaler().fit(y[:train_end].reshape(-1,1))

        # Scale full series, then remake windows scaled (for selected indices)
        Xs = xsc.transform(X)
        ys = ysc.transform(y.reshape(-1,1)).ravel()
        Xw_s, Yw_s = make_windows(Xs, ys, L, H)

        Xtr = torch.tensor(Xw_s[tr_sel]); Ytr = torch.tensor(Yw_s[tr_sel])
        Xva = torch.tensor(Xw_s[va_sel]); Yva = torch.tensor(Yw_s[va_sel])
        Xte = torch.tensor(Xw_s[te_sel]); Yte = torch.tensor(Yw_s[te_sel])

        train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=batch_size)

        model = LSTMRegressor(
            input_size=Xtr.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            horizon=H
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_val = math.inf
        best_state = None

        for ep in range(1, epochs+1):
            model.train()
            tr_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                va_losses = []
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    va_losses.append(loss_fn(model(xb), yb).item())
            val_loss = float(np.mean(va_losses)) if va_losses else float("nan")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate on the test window
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            yhat = model(Xte.to(device)).cpu().numpy()  # (1, H)
        ytrue = Yte.numpy()

        # De-normalize
        yhat_den = yhat * ysc.scale_[0] + ysc.mean_[0]
        ytrue_den = ytrue * ysc.scale_[0] + ysc.mean_[0]

        # Metrics (t+1 and all-horizon)
        mae_1 = np.mean(np.abs(yhat_den[:,0] - ytrue_den[:,0]))
        rmse_1 = np.sqrt(np.mean((yhat_den[:,0] - ytrue_den[:,0])**2))
        mae_all = np.mean(np.abs(yhat_den - ytrue_den))
        rmse_all = np.sqrt(np.mean((yhat_den - ytrue_den)**2))

        fold_results.append({
            "fold": i,
            "train_end_index": int(train_end),
            "mae_t+1": float(mae_1),
            "rmse_t+1": float(rmse_1),
            "mae_all": float(mae_all),
            "rmse_all": float(rmse_all),
            "val_mse_best": float(best_val),
        })

    # Aggregate
    agg = {
        "folds": len(fold_results),
        "mae_t+1_mean": float(np.mean([r["mae_t+1"] for r in fold_results])),
        "mae_t+1_std": float(np.std([r["mae_t+1"] for r in fold_results])),
        "mae_all_mean": float(np.mean([r["mae_all"] for r in fold_results])),
        "mae_all_std": float(np.std([r["mae_all"] for r in fold_results])),
        "rmse_all_mean": float(np.mean([r["rmse_all"] for r in fold_results])),
        "rmse_all_std": float(np.std([r["rmse_all"] for r in fold_results])),
    }
    return fold_results, agg, used_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--target", default="temp")
    ap.add_argument("--features", default="")
    ap.add_argument("--lookback", type=int, default=72)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--initial-days", type=int, default=60, help="days used before first forecast")
    ap.add_argument("--step-hours", type=int, default=24, help="advance between folds")
    ap.add_argument("--folds", type=int, default=5, help="0 = use all possible")
    ap.add_argument("--out", default="runs/backtest_weather")
    args = ap.parse_args()

    features = [c.strip() for c in args.features.split(",") if c.strip()]

    folds, agg, used = backtest(
        csv_path=args.data,
        time_col=args.time_col,
        target_col=args.target,
        feature_cols=features,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        initial_days=args.initial_days,
        step_hours=args.step_hours,
        folds=args.folds,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "folds.json"), "w") as f:
        json.dump(folds, f, indent=2)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print("\n=== Backtest summary ===")
    print(json.dumps(agg, indent=2))
    print(f"\nSaved details to {args.out}/folds.json and {args.out}/summary.json")

if __name__ == "__main__":
    main()
