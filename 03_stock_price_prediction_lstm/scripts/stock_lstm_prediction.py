import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception as e:
    raise ImportError("Install TensorFlow first: pip install tensorflow") from e

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/sample_stock_prices.csv", parse_dates=["date"]).sort_values("date")
out = ROOT/"outputs"
out.mkdir(exist_ok=True)

close_values = df[["close"]].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_values)

def make_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

WINDOW = 60
X, y = make_sequences(scaled, WINDOW)
X = X.reshape(X.shape[0], X.shape[1], 1)
split = int(len(X)*0.8)
X_train, X_valid, y_train, y_valid = X[:split], X[split:], y[:split], y[split:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW,1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_valid, y_valid), verbose=1)

pred_scaled = model.predict(X_valid)
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_valid.reshape(-1,1))
mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

dates = df["date"].iloc[WINDOW+split:].reset_index(drop=True)
result = pd.DataFrame({"date": dates, "actual_close": actual.flatten(), "predicted_close": pred.flatten()})
result.to_csv(out/"actual_vs_predicted.csv", index=False)
pd.DataFrame([{"mae":mae, "rmse":rmse, "window_size":WINDOW}]).to_csv(out/"model_metrics.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(result["date"], result["actual_close"], label="Actual Close")
plt.plot(result["date"], result["predicted_close"], label="Predicted Close")
plt.title("Stock Price Prediction: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig(out/"actual_vs_predicted.png", dpi=150)
print("LSTM complete. Metrics:", {"mae": mae, "rmse": rmse})
