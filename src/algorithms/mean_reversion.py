# src/algorithms/mean_reversion.py

import pandas as pd
import numpy as np
from sqlalchemy import text
from src.data.db_connection import fetch_historical_data
import torch  # If used elsewhere in your code
import torch.nn as nn  # If used elsewhere in your code

class MeanReversionStrategy:
    def __init__(self, df, window=20, std_dev_multiplier=2):
        self.df = df.copy()
        self.window = window
        self.std_dev_multiplier = std_dev_multiplier

        # Ensure 'close' column is numeric
        self.df['close'] = pd.to_numeric(self.df['close'], errors='coerce')
        self.df.dropna(subset=['close'], inplace=True)

        # Ensure 'timestamp' column exists and is datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            self.df.dropna(subset=['timestamp'], inplace=True)
            self.df.set_index('timestamp', inplace=True)
        elif 'date' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df.dropna(subset=['timestamp'], inplace=True)
            self.df.set_index('timestamp', inplace=True)
        elif isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index.name = 'timestamp'
        else:
            print("Error: 'timestamp' column not found in DataFrame, and index is not datetime.")
            raise ValueError("Timestamp data is required for mean reversion analysis.")

        # Remove duplicate indices
        self.df = self.df[~self.df.index.duplicated(keep='last')]

        # Sort the DataFrame by index
        self.df.sort_index(inplace=True)

    def calculate_indicators(self):
        self.df[f'sma_{self.window}'] = self.df['close'].rolling(window=self.window).mean()
        self.df['std_dev'] = self.df['close'].rolling(window=self.window).std()
        self.df['lower_bound'] = self.df[f'sma_{self.window}'] - (self.std_dev_multiplier * self.df['std_dev'])
        self.df['upper_bound'] = self.df[f'sma_{self.window}'] + (self.std_dev_multiplier * self.df['std_dev'])
        self.current_price = self.df['close'].iloc[-1]
        self.lower_bound = self.df['lower_bound'].iloc[-1]
        self.upper_bound = self.df['upper_bound'].iloc[-1]

    def get_signal(self):
        self.calculate_indicators()
        if self.current_price <= self.lower_bound:
            return 'BUY'
        elif self.current_price >= self.upper_bound:
            return 'SELL'
        else:
            return 'HOLD'

    def calculate_performance_metrics(self):
        roi = self.calculate_roi()
        sharpe_ratio = self.calculate_sharpe_ratio()
        volatility_score = self.calculate_volatility_score()
        rsi_value = self.calculate_rsi()
        ma_cross_signal = self.calculate_moving_average_cross()
        win_loss_ratio_value = None  # Implement if you have trade data
        return roi, sharpe_ratio, volatility_score, rsi_value, ma_cross_signal, win_loss_ratio_value

    def calculate_roi(self):
        initial_price = self.df['close'].iloc[0]
        final_price = self.df['close'].iloc[-1]
        roi = (final_price - initial_price) / initial_price
        return round(roi, 4)

    def calculate_sharpe_ratio(self):
        risk_free_rate = 0.01  # 1% annual risk-free return
        returns = self.df['close'].pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        if returns.std() == 0:
            return 0.0  # Avoid division by zero
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        return round(sharpe_ratio, 4)

    def calculate_volatility_score(self):
        volatility = self.df['close'].pct_change().rolling(window=20).std().mean() * np.sqrt(252)
        return round(volatility, 4)

    def calculate_rsi(self, periods=14):
        delta = self.df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=periods - 1, adjust=False).mean()
        ema_down = down.ewm(com=periods - 1, adjust=False).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        return round(rsi.iloc[-1], 2)

    def calculate_moving_average_cross(self, short_window=20, long_window=50):
        self.df['ma_short'] = self.df['close'].rolling(window=short_window).mean()
        self.df['ma_long'] = self.df['close'].rolling(window=long_window).mean()
        if self.df['ma_short'].iloc[-1] > self.df['ma_long'].iloc[-1]:
            return 'BUY'
        elif self.df['ma_short'].iloc[-1] < self.df['ma_long'].iloc[-1]:
            return 'SELL'
        else:
            return 'HOLD'

    def insert_scorecard_data(self, **kwargs):
        # Implement this method if needed
        pass

if __name__ == "__main__":
    df = fetch_historical_data()
    if df.empty:
        print("No data fetched from the database.")
    else:
        strategy = MeanReversionStrategy(df)
        signal = strategy.get_signal()
        roi, sharpe_ratio, volatility_score, rsi_value, ma_cross_signal, win_loss_ratio_value = strategy.calculate_performance_metrics()
        print(f"Generated signal: {signal}")
        print(f"ROI: {roi}, Sharpe Ratio: {sharpe_ratio}, Volatility Score: {volatility_score}")
        print(f"RSI: {rsi_value}, Moving Average Cross Signal: {ma_cross_signal}")
