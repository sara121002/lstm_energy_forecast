from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_run(run_dir: str):
    with open(os.path.join(run_dir, 'metrics.json')) as f:
        metrics = json.load(f)
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    return metrics, cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()

    metrics, cfg = load_run(args.run_dir)
    print("Run:", args.run_dir)
    print(json.dumps(metrics, indent=2))

    if args.plots:
        # If predictions.npy and truth.npy exist, plot a few examples
        pred_path = os.path.join(args.run_dir, 'predictions.npy')
        true_path = os.path.join(args.run_dir, 'truth.npy')
        if os.path.exists(pred_path) and os.path.exists(true_path):
            yhat = np.load(pred_path)
            ytrue = np.load(true_path)
            idx = np.random.choice(len(yhat), size=min(5, len(yhat)), replace=False)
            for i in idx:
                plt.figure()
                plt.plot(ytrue[i], label='truth')
                plt.plot(yhat[i], label='pred')
                plt.title(f'Example {i} (horizon={yhat.shape[1]})')
                plt.legend()
                plt.show()
        else:
            print("No saved predictions to plot (optional)." )

if __name__ == '__main__':
    main()