import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data.db_connection import create_db_connection, close_db_connection
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
        self.df = df

        # Ensure the dataframe index is a proper datetime index with frequency
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            self.df.index = pd.to_datetime(self.df.index)

        # Remove duplicate indices
        self.df = self.df[~self.df.index.duplicated(keep='last')]

        if self.df.index.freq is None:
            self.df = self.df.asfreq('D')

        self.df['returns'] = self.df['close'].pct_change().fillna(0)

    def fit_arima(self, order=(5, 1, 0)):
        if len(self.df) < 10:  # Ensure there is enough data for ARIMA
            print("Insufficient data for ARIMA. Skipping model fitting.")
            return

        model = ARIMA(self.df['close'], order=order)
        self.arima_result = model.fit()
        arima_rmse = self.calculate_arima_rmse()
        self.insert_scorecard_data(strategy_name="ARIMA", arima_rmse=arima_rmse)

    def fit_garch(self):
        returns = self.df['returns'] * 100
        model = arch_model(returns, vol='Garch', p=1, q=1)
        self.garch_result = model.fit()
        garch_volatility = self.garch_result.conditional_volatility.mean()
        self.insert_scorecard_data(strategy_name="GARCH", garch_volatility=garch_volatility)

    def fit_lstm(self, epochs=50, batch_size=32):
        X, y = self.prepare_lstm_data()
        if len(X) == 0:  # If no data is available for training
            print("Insufficient data for LSTM. Skipping model fitting.")
            return

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

    def prepare_lstm_data(self):
        series = self.df['close'].values
        X, y = [], []
        window_size = 60
        for i in range(window_size, len(series)):
            X.append(series[i - window_size:i])
            y.append(series[i])
        X = np.array(X).reshape(-1, window_size, 1)
        y = np.array(y)
        return torch.tensor(X).float(), torch.tensor(y).float()

    def calculate_arima_rmse(self):
        predictions = self.arima_result.forecast(steps=len(self.df))
        predictions = pd.Series(predictions, index=self.df.index)
        rmse = np.sqrt(((self.df['close'] - predictions) ** 2).mean())
        return round(rmse, 4)

    def calculate_lstm_rmse(self):
        X, y = self.prepare_lstm_data()
        with torch.no_grad():
            predictions = self.lstm_model(X)
        rmse = torch.sqrt(torch.mean((predictions - y) ** 2)).item()
        return round(rmse, 4)

    def insert_scorecard_data(self, strategy_name, arima_rmse=None, garch_volatility=None, lstm_rmse=None):
        """
        Insert the strategy results into the scorecard table.
        """
        connection = create_db_connection()
        query = """
            INSERT INTO scorecard (timestamp, strategy_name, arima_rmse, garch_volatility, lstm_rmse)
            VALUES (NOW(), %s, %s, %s, %s)
        """
        data_tuple = (strategy_name, arima_rmse, garch_volatility, lstm_rmse)  # Ensure it's a tuple
        try:
            connection.execute(query, data_tuple)
        except Exception as e:
            print(f"Error inserting scorecard data: {e}")
        finally:
            close_db_connection(connection)

