import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model  # For GARCH
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller  # For ADF test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from keras.wrappers.scikit_learn import KerasRegressor  # Only if you still want to use this for hyperparameter tuning
# Other necessary imports

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_layer = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        predictions = self.linear(lstm_out.view(len(x), -1))
        return predictions[-1]

class TimeSeriesAnalysis:
    def __init__(self, df):
        self.df = df
        # Normalize data for LSTM
        self.df_normalized = (df - df.mean()) / df.std()

    def train_test_split(self, test_size=0.2):
        return train_test_split(self.df, test_size=test_size)

    # ARIMA Model
    def fit_arima(self, order=(5, 1, 0), train_data=None):
        model = ARIMA(train_data if train_data is not None else self.df['close'], order=order)
        self.arima_result = model.fit(disp=0)
        return self.arima_result

    # GARCH Model
    def fit_garch(self, train_data=None):
        model = arch_model(train_data if train_data is not None else self.df['returns'], vol='Garch', p=1, q=1)
        self.garch_result = model.fit()
        return self.garch_result

    # LSTM Model
    def create_lstm_model(self, neurons=50, optimizer='adam'):
        model = Sequential()
        model.add(LSTM(neurons, activation='relu', input_shape=(1, 1)))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

        # Redefine the LSTM model setup and training

    def fit_lstm(self, train_data, epochs=50, batch_size=32):
        # Prepare the data
        X, y = self._prepare_lstm_data(train_data['close'])

        # Convert to PyTorch tensors
        X_train = torch.tensor(X).float().view([len(X), 1, -1])
        y_train = torch.tensor(y).float()

        # Define the model
        model = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for i in range(len(X_train)):
                optimizer.zero_grad()
                output = model(X_train[i])
                loss = criterion(output, y_train[i])
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch} Loss: {loss.item()}')

        self.lstm_model = model

    def _prepare_lstm_data(self, data, window_size=5):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i].tolist())
            y.append(data[i])
        return np.array(X), np.array(y)

    # Example method for hyperparameter tuning for LSTM using GridSearchCV
    def tune_lstm(self, param_grid, cv=3):
        model = KerasRegressor(build_fn=self.create_lstm_model, epochs=100, batch_size=10, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        grid_result = grid.fit(self.df_normalized['close'].values, self.df_normalized['close'].shift(-1).dropna().values)
        return grid_result

# Example usage
# Assuming you have a DataFrame 'enriched_df' from your historical_data.py
ts_analysis = TimeSeriesAnalysis(enriched_df)

# Split data into training and testing sets
train_data, test_data = ts_analysis.train_test_split()

# Fit ARIMA model
arima_result = ts_analysis.fit_arima(order=(5, 1, 0), train_data=train_data['close'])
print("ARIMA Model fitted")

# Fit GARCH model
garch_result = ts_analysis.fit_garch(train_data=train_data['returns'])
print("GARCH Model fitted")

# Fit LSTM model
lstm_model = ts_analysis.fit_lstm(train_data, epochs=50, batch_size=32)
print("LSTM Model fitted")

# For LSTM hyperparameter tuning (example grid)
param_grid = {'neurons': [50, 100], 'optimizer': ['adam', 'rmsprop']}
grid_result = ts_analysis.tune_lstm(param_grid)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
