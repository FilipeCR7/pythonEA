import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model  # For GARCH
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils.data_utils import prepare_time_series_data

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

class TimeSeriesAnalysis:
    def __init__(self, currency_pair):
        self.df = prepare_time_series_data(currency_pair)
        # Normalize data
        self.df_normalized = (self.df - self.df.mean()) / self.df.std()

    def fit_arima(self, order=(5, 1, 0)):
        model = ARIMA(self.df['close'], order=order)
        self.arima_result = model.fit(disp=0)
        print("ARIMA Model fitted:", self.arima_result.summary())

    def fit_garch(self):
        model = arch_model(self.df['returns'], vol='Garch', p=1, q=1)
        self.garch_result = model.fit(update_freq=5)
        print("GARCH Model fitted:", self.garch_result.summary())

    def _prepare_lstm_data(self, window_size=5):
        series = self.df_normalized['close'].values
        X, y = [], []
        for i in range(window_size, len(series)):
            X.append(series[i-window_size:i])
            y.append(series[i])
        X = np.array(X)
        y = np.array(y)
        # Reshaping for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return torch.tensor(X).float(), torch.tensor(y).float()

    def fit_lstm(self, epochs=50, batch_size=32):
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
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        self.lstm_model = model  # Storing the trained model

# Example usage
if __name__ == '__main__':
    currency_pair = 'USD/CAD'
    ts_analysis = TimeSeriesAnalysis(currency_pair)

    # Assuming you have a method to fetch your data correctly
    ts_analysis.fit_arima()
    ts_analysis.fit_garch()

    # Training the LSTM model
    ts_analysis.fit_lstm()
