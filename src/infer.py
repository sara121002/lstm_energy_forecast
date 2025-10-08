from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from joblib import load
from model import LSTMRegressor
from data import add_time_features

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data', default='data/sample_energy.csv', help='CSV to take the last lookback window from')
    p.add_argument('--time-col', default='timestamp')
    p.add_argument('--target', default='load')
    p.add_argument('--steps', type=int, default=24, help='Horizon to forecast')
    args = p.parse_args()

    with open(os.path.join(args.run_dir, 'config.json')) as f:
        cfg = json.load(f)

    # Recreate model
    input_size = 4 + len(cfg.get('features', '').split(',')) if cfg.get('features') else 4
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        bidirectional=cfg['bidirectional'],
        horizon=cfg['horizon']
    )
    model.load_state_dict(torch.load(os.path.join(args.run_dir, 'best.pt'), map_location='cpu'))
    model.eval()

    # Load scalers
    xsc = load(os.path.join(args.run_dir, 'x_scaler.joblib'))
    ysc = load(os.path.join(args.run_dir, 'y_scaler.joblib'))
    used = json.load(open(os.path.join(args.run_dir, 'used_features.json')))['used_features']

    df = pd.read_csv(args.data).sort_values(args.time_col).reset_index(drop=True)
    df = add_time_features(df, args.time_col)
    X = df[used].values.astype('float32')

    Xs = (X - np.array(xsc['x_mean'])) / np.array(xsc['x_scale'])

    lookback = cfg['lookback']
    if len(Xs) < lookback:
        raise ValueError("Not enough rows to build the lookback window.")
    last_window = Xs[-lookback:]  # (T, F)

    with torch.no_grad():
        yhat = model(torch.tensor(last_window, dtype=torch.float32).unsqueeze(0))  # (1, horizon)
    yhat = yhat.numpy()[0]

    # Denormalize
    yhat_denorm = yhat * ysc['y_scale'] + ysc['y_mean']

    print("Forecast (next {} steps):".format(cfg['horizon']))
    print(yhat_denorm.tolist())

if __name__ == '__main__':
    main()