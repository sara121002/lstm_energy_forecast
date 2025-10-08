# src/ablation.py
import argparse, os, json, subprocess, sys, csv
from pathlib import Path

EXPERIMENTS = [
    # (run_name_suffix, features CSV string)
    ("temp_only", ""),
    ("temp_hum_wind", "humidity,wind_speed"),
    ("temp_hum_wind_press", "humidity,wind_speed,pressure"),
    ("all_features", "humidity,wind_speed,pressure,precip"),
]

def run_train(python_cmd, data, time_col, target, lookback, horizon, epochs, batch_size, base_out):
    rows = []
    base = Path(base_out)
    base.mkdir(parents=True, exist_ok=True)

    for name, features in EXPERIMENTS:
        out_dir = base / name
        cmd = [
            python_cmd, "src/train.py",
            "--data", data,
            "--time-col", time_col,
            "--target", target,
            "--lookback", str(lookback),
            "--horizon", str(horizon),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--out-dir", str(out_dir),
        ]
        if features:
            cmd += ["--features", features]

        print("\n=== Running:", name, "===")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # read metrics.json
        mpath = out_dir / "metrics.json"
        with open(mpath, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        rows.append({
            "exp": name,
            "features": features if features else "(none)",
            "val_mse_best": metrics.get("val_mse_best"),
            "test_mae_t+1": metrics.get("test_mae_t+1"),
            "test_rmse_t+1": metrics.get("test_rmse_t+1"),
            "test_mae_all": metrics.get("test_mae_all"),
            "test_rmse_all": metrics.get("test_rmse_all"),
            "test_mape_all_%": metrics.get("test_mape_all_%"),
        })

    # write a CSV report
    report_path = base / "ablation_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # pretty print
    print("\n=== Ablation Summary ===")
    for r in rows:
        print(f"{r['exp']:>16} | MAE_all={r['test_mae_all']:.3f} | RMSE_all={r['test_rmse_all']:.3f} | MAPE_all={r['test_mape_all_%']:.2f}% | features={r['features']}")

    print(f"\nSaved CSV report → {report_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--target", default="temp")
    ap.add_argument("--lookback", type=int, default=72)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-base", default="runs/ablation_weather")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = ap.parse_args()

    run_train(
        python_cmd=args.python,
        data=args.data,
        time_col=args.time_col,
        target=args.target,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_out=args.out_base,
    )

if __name__ == "__main__":
    main()
