import datetime

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.db_connection import create_db_connection, close_db_connection
import warnings
import statsmodels

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

        # Ensure the dataframe index is a proper datetime index
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.set_index('timestamp', inplace=True)
        else:
            print("Timestamp column not found in DataFrame.")
            return

        # Remove duplicate indices
        self.df = self.df[~self.df.index.duplicated(keep='last')]

        # Sort the DataFrame by index
        self.df.sort_index(inplace=True)

        self.df['returns'] = self.df['close'].pct_change().fillna(0)

    def fit_arima(self, order=(5, 1, 0)):
        if len(self.df) < 10:  # Ensure there is enough data for ARIMA
            print("Insufficient data for ARIMA. Skipping model fitting.")
            return

        try:
            model = ARIMA(self.df['close'], order=order)
            self.arima_result = model.fit()
            arima_rmse = self.calculate_arima_rmse()
            self.insert_scorecard_data(strategy_name="ARIMA", arima_rmse=arima_rmse)
            print(f"ARIMA model fitted successfully with RMSE: {arima_rmse}")
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")

    def fit_garch(self):
        if len(self.df) < 10:
            print("Insufficient data for GARCH. Skipping model fitting.")
            return

        try:
            returns = self.df['returns'] * 100
            model = arch_model(returns, vol='Garch', p=1, q=1)
            self.garch_result = model.fit(disp='off')
            garch_volatility = self.garch_result.conditional_volatility.mean()
            self.insert_scorecard_data(strategy_name="GARCH", garch_volatility=garch_volatility)
            print(f"GARCH model fitted successfully with average volatility: {garch_volatility}")
        except Exception as e:
            print(f"Error fitting GARCH model: {e}")

    def fit_lstm(self, epochs=10, batch_size=32):
        X, y = self.prepare_lstm_data()
        if len(X) == 0:  # If no data is available for training
            print("Insufficient data for LSTM. Skipping model fitting.")
            return

        try:
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())

            for epoch in range(epochs):
                for inputs, targets in loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            self.lstm_model = model
            lstm_rmse = self.calculate_lstm_rmse()
            self.insert_scorecard_data(strategy_name="LSTM", lstm_rmse=lstm_rmse)
            print(f"LSTM model fitted successfully with RMSE: {lstm_rmse}")
        except Exception as e:
            print(f"Error fitting LSTM model: {e}")

    def prepare_lstm_data(self):
        series = self.df['close'].values
        X, y = [], []
        window_size = 60
        if len(series) <= window_size:
            return np.array([]), np.array([])

        for i in range(window_size, len(series)):
            X.append(series[i - window_size:i])
            y.append(series[i])
        X = np.array(X).reshape(-1, window_size, 1)
        y = np.array(y).reshape(-1, 1)
        return torch.tensor(X).float(), torch.tensor(y).float()

    def calculate_arima_rmse(self):
        predictions = self.arima_result.predict(start=self.arima_result.k_diff, end=len(self.df) - 1, dynamic=False)
        actuals = self.df['close'][self.arima_result.k_diff:]
        rmse = np.sqrt(((actuals - predictions) ** 2).mean())
        return round(rmse, 4)

    def calculate_lstm_rmse(self):
        X, y = self.prepare_lstm_data()
        if len(X) == 0:
            return None
        with torch.no_grad():
            predictions = self.lstm_model(X)
        rmse = torch.sqrt(torch.mean((predictions - y) ** 2)).item()
        return round(rmse, 4)

    def insert_scorecard_data(self, strategy_name, signal, roi, sharpe_ratio, volatility_score):
        connection = create_db_connection()
        query = text("""
            INSERT INTO scorecard (`timestamp`, strategy_name, `signal`, roi, sharpe_ratio, volatility_score)
            VALUES (:timestamp, :strategy_name, :signal, :roi, :sharpe_ratio, :volatility_score)
        """)
        try:
            params = {
                'timestamp': datetime.now(),
                'strategy_name': strategy_name,
                'signal': signal,
                'roi': roi,
                'sharpe_ratio': sharpe_ratio,
                'volatility_score': volatility_score
            }
            print(query)
            print(params)
            print({k: type(v) for k, v in params.items()})

            result = connection.execute(query, params)
            connection.commit()  # Commit the transaction

            print(f"Inserted {result.rowcount} rows.")
        except Exception as e:
            print(f"Error inserting scorecard data: {e}")
            connection.rollback()  # Roll back in case of error
        finally:
            close_db_connection(connection)

