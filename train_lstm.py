#!/usr/bin/env python3
"""
train_lstm.py — Minimal, runnable demo for next‑day stock price prediction with an LSTM.

Usage:
  python train_lstm.py --tickers AAPL,MSFT,NVDA --period 2y --window 60 --epochs 8 --outdir runs/demo
"""
import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_sequences(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i, 0])
        y.append(series[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def build_model(window: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

def train_for_ticker(ticker: str, period: str, window: int, epochs: int, batch_size: int, test_size: float, outdir: Path, verbose: int = 1):
    print(f"=== {ticker}: downloading {period} of data ===")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check symbol or period.")
    df = df[['Close']].copy().dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['Close']])
    X, y = make_sequences(scaled, window=window)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = build_model(window=window)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    print(f"{ticker}: RMSE = {rmse:.4f}")

    td = outdir / ticker
    td.mkdir(parents=True, exist_ok=True)
    model.save(td / "model.h5")
    import joblib
    joblib.dump(scaler, td / "scaler.pkl")
    with open(td / "metrics.json", "w") as f:
        json.dump({"ticker": ticker, "rmse": rmse, "n_samples": int(len(df)), "window": window, "epochs": epochs, "test_size": test_size}, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.title(f"{ticker} — Next‑Day Close Prediction (RMSE={rmse:.2f})")
    plt.xlabel("Test Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(td / "prediction_plot.png", dpi=150)
    plt.close()

    import pandas as pd
    pd.DataFrame({"actual": y_test_inv, "predicted": y_pred_inv}).to_csv(td / "pred_vs_actual.csv", index=False)
    return {"ticker": ticker, "rmse": rmse}

def parse_args():
    p = argparse.ArgumentParser(description="Train a minimal LSTM for next‑day stock price prediction.")
    p.add_argument("--tickers", type=str, required=True, help="Comma‑separated tickers, e.g., AAPL,MSFT,NVDA")
    p.add_argument("--period", type=str, default="2y", help="yfinance period (e.g., 6mo, 1y, 2y, 5y)")
    p.add_argument("--window", type=int, default=60, help="Sliding window length (days)")
    p.add_argument("--epochs", type=int, default=8, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--test_size", type=float, default=0.2, help="Fraction for test")
    p.add_argument("--outdir", type=str, default="runs/demo", help="Output dir for artifacts")
    p.add_argument("--verbose", type=int, default=1, help="Keras training verbosity (0/1/2)")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(42); np.random.seed(42)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    summary = []
    for t in tickers:
        try:
            summary.append(train_for_ticker(t, args.period, args.window, args.epochs, args.batch_size, args.test_size, outdir, args.verbose))
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Artifacts saved to:", outdir.resolve())

if __name__ == "__main__":
    main()
