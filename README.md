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

## 🌍 Versatile Forecasting Pipeline

This LSTM pipeline is not limited to energy forecasting — it can handle any time-series prediction task with minimal changes.
By simply changing the --target argument and providing a suitable CSV, the same code can forecast:

Temperature, humidity, or wind speed (weather forecasting)

Energy load or generation (power systems)

Sensor signals or industrial trends 

```bash
python src/train.py --data data/weather_sample_120d_hourly.csv \
  --time-col timestamp \
  --target temp \
  --features humidity,wind_speed,pressure,precip \
  --lookback 72 --horizon 24 \
  --epochs 10 --batch-size 64 --out-dir runs/weather_temp
```
With this small change in the target column, the pipeline becomes a 24-hour weather predictor instead of an energy model — demonstrating its flexibility for any continuous time-series domain.

## 📊 Visualization
After training, you can easily visualize your model’s performance — plotting the last observed values together with the next predicted horizon.

Use the visualization script:
```bash
python src/plot_forecast.py --run-dir runs/weather_temp --data data/weather_sample_120d_hourly.csv --history 72 --save runs/weather_temp/forecast_plot.png
```
--run-dir → folder where the trained model and config are stored (e.g. runs/weather_temp)

--data → dataset used during training

--history → number of past hours to display before the forecast (default = 72)

--save → optional path to save the figure as PNG

The output graph shows:

Blue line → the last 72 hours of actual temperature data

Orange line (or dots) → the next 24 hours of LSTM forecasts

This gives a quick visual sense of how well the model captures short-term trends and daily cycles.

## 🔍 Feature Importance (Ablation Study)

To understand which input variables contribute most to forecast accuracy, several models were trained with different feature combinations.
Each run used the same LSTM architecture and parameters — only the input features changed.
Result Summary:
| Model               | MAE_all | RMSE_all | MAPE_all_% |
| ------------------- | ------- | -------- | ---------- |
| temp_only           | 0.719   | 0.896    | 15.47%     |
| temp_hum_wind       | 0.721   | 0.901    | 15.73%     |
| temp_hum_wind_press | 0.718   | 0.899    | 15.41%     |
| all_features        | 0.723   | 0.902    | 15.50%     |

Interpretation:
The model achieves nearly identical accuracy across all feature combinations.
This means that for short-term (24-hour) temperature forecasting, past temperature alone already contains enough information.
Adding humidity, wind, or pressure provides little additional benefit on this dataset — likely because the synthetic data has strong temporal patterns already captured by temperature itself.

You can reproduce this experiment with:
```bash
python src/ablation.py --data data/weather_sample_120d_hourly.csv \
  --time-col timestamp --target temp \
  --lookback 72 --horizon 24 --epochs 8 --batch-size 64 \
  --out-base runs/ablation_weather
```

The results are saved as:
```bash
runs/ablation_weather/ablation_report.csv
```
