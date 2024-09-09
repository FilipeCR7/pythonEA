import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # For GARCH model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data.db_connection import create_db_connection, close_db_connection  # Ensure correct import
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge.")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the LSTM model.
        :param input_dim: Number of input features.
        :param hidden_dim: Number of hidden units in the LSTM layer.
        :param output_dim: Number of output features.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the LSTM model.
        """
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # Output of the last time step
        return x

class TimeSeriesAnalysis:
    def __init__(self, df):
        """
        Initialize the TimeSeriesAnalysis class with historical data.
        :param df: DataFrame containing historical price data.
        """
        self.df = df

        # Ensure the dataframe index is a proper datetime index with frequency
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            self.df.index = pd.to_datetime(self.df.index)
        if self.df.index.freq is None:
            self.df = self.df.asfreq('D')  # Set frequency to daily; adjust as needed

        self.df['returns'] = self.df['close'].pct_change().fillna(0)

    def fit_arima(self, order=(5, 1, 0)):
        """
        Fit an ARIMA model on the time series data.
        :param order: ARIMA order (p, d, q).
        """
        model = ARIMA(self.df['close'], order=order)
        self.arima_result = model.fit()
        arima_rmse = self.calculate_arima_rmse()
        self.insert_scorecard_data(strategy_name="ARIMA", arima_rmse=arima_rmse)

    def fit_garch(self):
        """
        Fit a GARCH model on the returns data.
        """
        returns = self.df['returns'] * 100  # Scaling for stability
        model = arch_model(returns, vol='Garch', p=1, q=1)
        self.garch_result = model.fit()
        garch_volatility = self.garch_result.conditional_volatility.mean()
        self.insert_scorecard_data(strategy_name="GARCH", garch_volatility=garch_volatility)

    def fit_lstm(self, epochs=50, batch_size=32):
        """
        Train an LSTM model on the time series data.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for the training loop.
        """
        X, y = self.prepare_lstm_data()
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
        """
        Prepare data for LSTM model training.
        """
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
        """
        Calculate RMSE for ARIMA model predictions.
        """
        # Generate predictions
        predictions = self.arima_result.forecast(steps=len(self.df))

        # Align predictions with the original df['close'] index
        predictions = pd.Series(predictions, index=self.df.index)

        # Calculate RMSE (Root Mean Square Error)
        rmse = np.sqrt(((self.df['close'] - predictions) ** 2).mean())
        return round(rmse, 4)

    def calculate_lstm_rmse(self):
        """
        Calculate RMSE for LSTM model predictions.
        """
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
        try:
            connection.execute(query, (strategy_name, arima_rmse, garch_volatility, lstm_rmse))
        except Exception as e:
            print(f"Error inserting scorecard data: {e}")
        finally:
            close_db_connection(connection)

# Example usage
if __name__ == "__main__":
    df = fetch_historical_data()  # Assuming fetch_historical_data is defined elsewhere
    ts_analysis = TimeSeriesAnalysis(df)
    ts_analysis.fit_arima()
    ts_analysis.fit_garch()
    ts_analysis.fit_lstm()
