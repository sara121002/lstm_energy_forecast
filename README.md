# lstm_energy_forecast

A clean, learn-by-doing project to forecast energy load (or generation) with an LSTM.  
Designed for step-by-step growth: start with a baseline, then upgrade to a full LSTM pipeline.

## 🔧 Quick start

```bash
# 1) Create & activate a virtualenv (any name is fine)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train on the provided sample data (synthetic hourly series)
python src/train.py --data data/sample_energy.csv --time-col timestamp --target load \
  --lookback 48 --horizon 24 --epochs 5 --batch-size 64 --out-dir runs/exp1

# 4) Evaluate & plot
python src/evaluate.py --run-dir runs/exp1 --plots

# 5) Forecast N future steps using the last lookback window from the dataset
python src/infer.py --run-dir runs/exp1 --steps 24
```

> Tip: Replace `data/sample_energy.csv` with your real dataset. Just keep a timestamp column and a target column.  
> Recommended minimum columns: `timestamp`, `load`. Extra features like `temp`, `wind` are optional.

## 🗂️ Project structure

```
lstm_energy_forecast/
├─ data/
│  └─ sample_energy.csv
├─ src/
│  ├─ __init__.py
│  ├─ baselines.py
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ infer.py
├─ configs/
│  └─ default.yaml
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## ✅ Step-by-step learning roadmap

1. **Baseline**: seasonal naive (yesterday-same-hour) + metrics (RMSE/MAE/MAPE).
2. **Windowing**: turn a time series into supervised samples (lookback → horizon).
3. **LSTM**: single-layer LSTM regressor → compare vs. baseline.
4. **Feature engineering**: add calendar features (hour sin/cos, dow), weather (temp/wind).
5. **Scaling**: fit scalers on train only; persist with joblib.
6. **Validation**: rolling-origin (time series split) instead of random shuffle.
7. **Multi-step decoding**: direct (many heads) vs iterative (one-step rolled).
8. **Tuning**: hidden size, layers, dropout, lookback/horizon.
9. **Packaging**: save model + config; `infer.py` CLI for fast forecasts.
10. **Deploy (optional)**: add a simple FastAPI endpoint later.

## 📝 Data format

CSV with at least:
- `timestamp` (string, parseable to datetime)
- `load` (float — your target)

You can add optional feature columns (e.g., `temp`, `wind`); they will be scaled and fed to the model.

## 📊 Metrics

- RMSE, MAE, MAPE. We compare LSTM vs a seasonal naive baseline.

## 🔄 Reproducibility

- We save `config.json`, scalers, and weights under `runs/<exp>/`.
- Set `--seed` for deterministic behavior (to the extent possible).

---

Made for learning: clear code, small surface area, and simple commands.