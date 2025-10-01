from __future__ import annotations
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
import yaml

from data import prepare_data
from model import LSTMRegressor

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rmse(a, b): return mean_squared_error(a, b, squared=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to CSV')
    p.add_argument('--time-col', default='timestamp')
    p.add_argument('--target', default='load')
    p.add_argument('--features', default='', help='Comma-separated extra feature columns')
    p.add_argument('--lookback', type=int, default=48)
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--val-ratio', type=float, default=0.15)
    p.add_argument('--test-ratio', type=float, default=0.15)
    p.add_argument('--hidden-size', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='runs/exp1')
    p.add_argument('--config', default='')
    args = p.parse_args()

    # Load defaults from YAML if provided
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        for k, v in yaml_cfg.items():
            if getattr(args, k, None) == p.get_default(k):
                setattr(args, k, v)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    feature_cols = [c for c in args.features.split(',') if c.strip()]

    data = prepare_data(
        csv_path=args.data, time_col=args.time_col, target_col=args.target,
        feature_cols=feature_cols, lookback=args.lookback, horizon=args.horizon,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    Xtr = torch.tensor(data['X_train'])
    Ytr = torch.tensor(data['Y_train'])
    Xva = torch.tensor(data['X_val'])
    Yva = torch.tensor(data['Y_val'])
    Xte = torch.tensor(data['X_test'])
    Yte = torch.tensor(data['Y_test'])

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMRegressor(
        input_size=Xtr.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        horizon=args.horizon
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_path = os.path.join(args.out_dir, 'best.pt')

    for epoch in range(1, args.epochs+1):
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
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())
        val_loss = float(np.mean(va_losses))

        print(f"Epoch {epoch:03d} | train MSE {np.mean(tr_losses):.4f} | val MSE {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    # Reload best model, evaluate on test
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        yhat = model(Xte.to(device)).cpu().numpy()
    y_true = Yte.numpy()

    # Inverse-scale for metrics
    y_mean = data['y_scaler_mean']; y_scale = data['y_scaler_scale']
    yhat_denorm = yhat * y_scale + y_mean
    ytrue_denorm = y_true * y_scale + y_mean

    # Compute metrics on the first step of horizon and on the average across horizon
    mae_1 = np.mean(np.abs(yhat_denorm[:,0] - ytrue_denorm[:,0]))
    rmse_1 = np.sqrt(np.mean((yhat_denorm[:,0] - ytrue_denorm[:,0])**2))
    mae_all = np.mean(np.abs(yhat_denorm - ytrue_denorm))
    rmse_all = np.sqrt(np.mean((yhat_denorm - ytrue_denorm)**2))
    mape_all = np.mean(np.abs((ytrue_denorm - yhat_denorm) / (np.clip(np.abs(ytrue_denorm), 1e-6, None)))) * 100.0

    metrics = {
        'val_mse_best': best_val,
        'test_mae_t+1': float(mae_1),
        'test_rmse_t+1': float(rmse_1),
        'test_mae_all': float(mae_all),
        'test_rmse_all': float(rmse_all),
        'test_mape_all_%': float(mape_all),
    }

    # Save run artifacts
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Save scalers & feature usage
    dump({'x_mean': data['x_scaler_mean'], 'x_scale': data['x_scaler_scale']},
         os.path.join(args.out_dir, 'x_scaler.joblib'))
    dump({'y_mean': data['y_scaler_mean'], 'y_scale': data['y_scaler_scale']},
         os.path.join(args.out_dir, 'y_scaler.joblib'))
    with open(os.path.join(args.out_dir, 'used_features.json'), 'w') as f:
        json.dump({'used_features': data['used_features']}, f, indent=2)

    print("Saved:", args.out_dir)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()