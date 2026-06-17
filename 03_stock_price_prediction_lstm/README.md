# Stock Price Prediction using LSTM

## Goal
Build a time-series forecasting model that predicts stock closing prices using an LSTM neural network.

## What this proves
- Time-series preparation
- Scaling
- Rolling window sequences
- LSTM model training
- Actual vs predicted evaluation

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
python scripts/stock_lstm_prediction.py
```

## Interview explanation
I used closing price data, scaled it, created rolling 60-day windows, trained an LSTM model, and compared actual vs predicted values. The main learning was how sequence models use previous observations to forecast future movement.
