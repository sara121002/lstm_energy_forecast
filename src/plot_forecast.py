import os, json, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from joblib import load

# reuse helpers from your repo
from model import LSTMRegressor
from data import add_time_features

def load_run_config(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "used_features.json")) as f:
        used = json.load(f)["used_features"]
    xsc = load(os.path.join(run_dir, "x_scaler.joblib"))
    ysc = load(os.path.join(run_dir, "y_scaler.joblib"))
    return cfg, used, xsc, ysc

def build_last_window(df, time_col, used_features, xsc, lookback):
    df = df.sort_values(time_col).reset_index(drop=True)
    df = add_time_features(df, time_col)
    X = df[used_features].values.astype("float32")
    # standardize with saved scalers
    Xs = (X - np.array(xsc["x_mean"])) / np.array(xsc["x_scale"])
    if len(Xs) < lookback:
        raise ValueError("Not enough rows to form the lookback window.")
    return df, Xs[-lookback:], len(df)  # return also length for indexing

def forecast(run_dir, data_csv):
    cfg, used, xsc, ysc = load_run_config(run_dir)

    # recreate model
    input_size = len(used)
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        horizon=cfg["horizon"],
    )
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.eval()

    df = pd.read_csv(data_csv)
    df, last_window, nrows = build_last_window(
        df, cfg["time_col"], used, xsc, cfg["lookback"]
    )

    with torch.no_grad():
        yhat = model(torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)).numpy()[0]

    # de-normalize
    y = df[cfg["target"]].values
    yhat_denorm = yhat * load(os.path.join(run_dir, "y_scaler.joblib"))["y_scale"] \
                  + load(os.path.join(run_dir, "y_scaler.joblib"))["y_mean"]

    return df, y, yhat_denorm, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="e.g. runs/weather_temp")
    ap.add_argument("--data", required=True, help="path to the CSV used for training")
    ap.add_argument("--history", type=int, default=72, help="hours of history to plot")
    ap.add_argument("--save", default="", help="optional: path to save PNG")
    args = ap.parse_args()

    df, y, yhat_denorm, cfg = forecast(args.run_dir, args.data)

    hist = min(len(y), args.history)
    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    # indices for plotting
    x_hist = np.arange(-hist, 0)
    x_fore = np.arange(0, horizon)

    plt.figure(figsize=(10,4))
    plt.plot(x_hist, y[-hist:], label=f"Last {hist}h actual")
    plt.plot(x_fore, yhat_denorm, label=f"Next {horizon}h forecast", marker="o")
    plt.xlabel("Hours (negative = past, positive = future)")
    plt.ylabel(cfg["target"])
    plt.title(f"LSTM forecast of '{cfg['target']}'")
    plt.legend()
    plt.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        plt.savefig(args.save, dpi=150)
        print(f"Saved plot → {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
