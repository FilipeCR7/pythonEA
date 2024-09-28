# src/algorithms/time_series_analysis.py

import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import text
from src.data.db_connection import fetch_historical_data
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge.")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # Output of the last time step
        return x

class TimeSeriesAnalysis:
    def __init__(self, df):
        self.df = df.copy()

        # Ensure 'close' column is numeric
        self.df['close'] = pd.to_numeric(self.df['close'], errors='coerce')
        self.df.dropna(subset=['close'], inplace=True)

        # Ensure 'timestamp' column exists and is datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            self.df.dropna(subset=['timestamp'], inplace=True)
            self.df.set_index('timestamp', inplace=True)
        elif 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df.dropna(subset=['date'], inplace=True)
            self.df.set_index('date', inplace=True)
            self.df.index.name = 'timestamp'
        elif isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index.name = 'timestamp'
        else:
            print("Error: 'timestamp' column not found in DataFrame, and index is not datetime.")
            raise ValueError("Timestamp data is required for time series analysis.")

        # Remove duplicate indices
        self.df = self.df[~self.df.index.duplicated(keep='last')]

        # Sort the DataFrame by index
        self.df.sort_index(inplace=True)

        # Ensure the index has a frequency
        if self.df.index.inferred_freq is None:
            # Try to infer frequency
            inferred_freq = pd.infer_freq(self.df.index)
            if inferred_freq:
                self.df = self.df.asfreq(inferred_freq)
            else:
                # Set a default frequency, adjust as needed
                self.df = self.df.asfreq('T')  # 'T' for minute data; adjust if needed

        # Handle missing data after setting frequency
        self.df['close'].interpolate(method='time', inplace=True)

        # Optional: Print index frequency for debugging
        # print("Index frequency:", self.df.index.freq)

        self.df['returns'] = self.df['close'].pct_change().fillna(0)

        self.arima_result = None  # Initialize ARIMA result

    def fit_arima(self, order=(5, 1, 0)):
        if len(self.df) < 10:  # Ensure there is enough data for ARIMA
            print("Insufficient data for ARIMA. Skipping model fitting.")
            return

        try:
            model = ARIMA(self.df['close'], order=order)
            self.arima_result = model.fit()
            arima_rmse = self.calculate_arima_rmse()
            print(f"ARIMA model fitted successfully with RMSE: {arima_rmse}")
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            self.arima_result = None

    def get_signal(self):
        try:
            if self.arima_result is None:
                print("ARIMA model is not fitted. Returning 'HOLD' signal.")
                return 'HOLD'
            # Forecast the next value using the ARIMA model
            forecast = self.arima_result.forecast(steps=1)
            next_prediction = forecast.iloc[0]  # Use iloc[0] to avoid FutureWarning
            last_close = self.df['close'].iloc[-1]
            if next_prediction > last_close:
                signal = 'BUY'
            elif next_prediction < last_close:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            return signal
        except Exception as e:
            print(f"Error generating signal from ARIMA model: {e}")
            return 'HOLD'

    def calculate_arima_rmse(self):
        if self.arima_result is None:
            return None
        d = self.arima_result.model_orders.get('d', 0)
        start = d
        end = len(self.df) - 1
        predictions = self.arima_result.predict(start=start, end=end, dynamic=False)
        actuals = self.df['close'][start:]
        rmse = np.sqrt(((actuals - predictions) ** 2).mean())
        return float(round(rmse, 4))  # Convert to standard float

    def calculate_performance_metrics(self):
        roi = self.calculate_roi()
        sharpe_ratio = self.calculate_sharpe_ratio()
        volatility_score = self.calculate_volatility_score()
        # Return None for metrics not used in this strategy
        return roi, sharpe_ratio, volatility_score, None, None, None

    def calculate_roi(self):
        initial_price = self.df['close'].iloc[0]
        final_price = self.df['close'].iloc[-1]
        roi = (final_price - initial_price) / initial_price
        return round(roi, 4)

    def calculate_sharpe_ratio(self):
        risk_free_rate = 0.01  # 1% annual risk-free return
        returns = self.df['returns'].dropna()
        excess_returns = returns - (risk_free_rate / 252)  # Daily adjustment
        if returns.std() == 0:
            return 0.0  # Avoid division by zero
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        return round(sharpe_ratio, 4)

    def calculate_volatility_score(self):
        volatility = self.df['returns'].rolling(window=20).std().mean() * np.sqrt(252)
        return round(volatility, 4)

    def insert_scorecard_data(self, **kwargs):
        # Implement this method if needed
        pass

if __name__ == "__main__":
    # For testing purposes
    df = fetch_historical_data()
    if df.empty:
        print("No data fetched from the database.")
    else:
        time_series = TimeSeriesAnalysis(df)
        time_series.fit_arima()
        signal = time_series.get_signal()
        roi, sharpe_ratio, volatility_score, _, _, _ = time_series.calculate_performance_metrics()
        arima_rmse = time_series.calculate_arima_rmse()
        print(f"Generated signal: {signal}")
        print(f"ROI: {roi}, Sharpe Ratio: {sharpe_ratio}, Volatility Score: {volatility_score}")
        print(f"ARIMA RMSE: {arima_rmse}")
