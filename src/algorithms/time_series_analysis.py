import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # For GARCH model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.data_utils import prepare_time_series_data


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        LSTM model for time series prediction.

        :param input_dim: Number of input features.
        :param hidden_dim: Number of hidden units in LSTM.
        :param output_dim: Output dimension (1 for univariate time series).
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        :param x: Input data tensor.
        :return: Output of the model.
        """
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # Take the output of the last time step
        return x


class TimeSeriesAnalysis:
    def __init__(self):
        """
        Initialize the time series analysis by fetching and preparing the data.
        """
        self.df = prepare_time_series_data()  # Load data
        self.df.index = pd.to_datetime(self.df.index)

        # Handle duplicate timestamps
        if self.df.index.duplicated().any():
            print("Duplicate timestamps found; removing duplicates.")
            self.df = self.df[~self.df.index.duplicated(keep='last')]

        # Ensure regular minute-level frequency in the data
        self.df = self.df.asfreq('T', method='ffill')

        # Normalize data for model input
        self.df_normalized = (self.df - self.df.mean()) / self.df.std()
        self.df['returns'] = self.df['close'].pct_change().fillna(0)

    def fit_arima(self, order=(5, 1, 0)):
        """
        Fit ARIMA model on the time series data.

        :param order: ARIMA model order (p, d, q).
        """
        model = ARIMA(self.df['close'], order=order)
        self.arima_result = model.fit()
        print("ARIMA Model fitted:", self.arima_result.summary())

    def fit_garch(self):
        """
        Fit GARCH model on the time series returns data.
        """
        scaled_returns = self.df['returns'] * 1e4  # Rescale returns for stability
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1)
        self.garch_result = model.fit(update_freq=5)
        print("GARCH Model fitted:", self.garch_result.summary())

    def _prepare_lstm_data(self, window_size=5):
        """
        Prepare the data for LSTM model training.

        :param window_size: Number of past time steps to consider as input.
        :return: Tensors for LSTM input (X) and output (y).
        """
        series = self.df_normalized['close'].values
        X, y = [], []
        for i in range(window_size, len(series)):
            X.append(series[i - window_size:i])
            y.append(series[i])
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
        return torch.tensor(X).float(), torch.tensor(y).float().view(-1, 1)

    def fit_lstm(self, epochs=50, batch_size=32):
        """
        Train the LSTM model on the time series data.

        :param epochs: Number of epochs to train the model.
        :param batch_size: Size of batches for training.
        """
        X, y = self._prepare_lstm_data()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        model.train()
        for epoch in range(epochs):
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        self.lstm_model = model  # Save the trained model


# Example usage
if __name__ == '__main__':
    ts_analysis = TimeSeriesAnalysis()

    # Fit ARIMA model
    ts_analysis.fit_arima()

    # Fit GARCH model
    ts_analysis.fit_garch()

    # Train the LSTM model
    ts_analysis.fit_lstm()
