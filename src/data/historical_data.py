import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model  # For GARCH
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.db_connection import create_db_connection
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

class TimeSeriesAnalysis:
    def __init__(self, df):
        self.df = df
        self.df['returns'] = self.df['close_price'].pct_change()
        self.df.dropna(inplace=True)
        self.df_normalized = (self.df - self.df.mean()) / self.df.std()

    @staticmethod
    def fetch_historical_data_from_db():
        query = "SELECT * FROM historical_data ORDER BY timestamp ASC"
        connection = create_db_connection()
        if connection is not None:
            try:
                df = pd.read_sql(query, connection)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            finally:
                connection.close()
        return pd.DataFrame()

    def perform_eda(self):
        # Plot the closing prices
        self.df['close_price'].plot(title='Closing Prices')
        plt.show()

        # Augmented Dickey-Fuller test
        result = adfuller(self.df['close_price'].dropna())
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

    def fit_arima(self, order=(5, 1, 0)):
        model = ARIMA(self.df['close_price'], order=order)
        self.arima_model = model.fit(disp=0)
        print(self.arima_model.summary())

    def fit_garch(self):
        returns = 100 * self.df['returns'].dropna()
        model = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1)
        self.garch_model = model.fit(update_freq=5)
        print(self.garch_model.summary())

    def prepare_lstm_data(self, window=60):
        feature_cols = ['close_price']
        df_scaled = (self.df[feature_cols] - self.df[feature_cols].mean()) / self.df[feature_cols].std()
        X, y = [], []
        for i in range(window, len(df_scaled)):
            X.append(df_scaled.iloc[i - window:i].values)
            y.append(df_scaled.iloc[i].values[0])
        X, y = np.array(X), np.array(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def fit_lstm(self, input_dim=1, hidden_dim=50, num_layers=1, output_dim=1, epochs=20, batch_size=1):
        X, y = self.prepare_lstm_data()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        model.train()
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        self.lstm_model = model

    def evaluate_model(self):
        # Evaluate ARIMA Model
        if hasattr(self, 'arima_model'):
            arima_predictions = self.arima_model.forecast(steps=len(self.test_data))[0]
            arima_rmse = mean_squared_error(self.test_data['close_price'], arima_predictions, squared=False)
            print(f'ARIMA Model RMSE: {arima_rmse}')

        # Evaluate GARCH Model
        if hasattr(self, 'garch_model'):
            # GARCH model evaluation would be more about volatility forecasting
            print("GARCH Model Summary:")
            print(self.garch_model.summary())

        # Evaluate LSTM Model
        if hasattr(self, 'lstm_model'):
            self.lstm_model.eval()  # Set the model to evaluation mode
            X_test, y_test = self.prepare_lstm_data()  # Prepare your test data
            with torch.no_grad():
                lstm_predictions = self.lstm_model(X_test)
            lstm_rmse = torch.sqrt(torch.mean((lstm_predictions - y_test) ** 2)).item()
            print(f'LSTM Model RMSE: {lstm_rmse}')

    def generate_trading_signals(self):
        if hasattr(self, 'lstm_model'):
            self.lstm_model.eval()  # Ensure the model is in evaluation mode
            # Assuming you have a method to fetch the latest data:
            latest_data = self.fetch_latest_data()  # Implement this method
            X, _ = self.prepare_lstm_data(latest_data)  # Prepare the data for prediction
            with torch.no_grad():
                prediction = self.lstm_model(X[-1].unsqueeze(0))  # Predict the latest point
            # Implement your logic for signal generation based on the prediction
            if prediction > 0:  # Example condition
                return "BUY"
            else:
                return "SELL"
        return "HOLD"  # Default action

# Example usage
def main():
    df = TimeSeriesAnalysis.fetch_historical_data_from_db()
    if not df.empty:
        # Initialize the TimeSeriesAnalysis with the fetched data
        ts_analysis = TimeSeriesAnalysis(df)

        # Perform Exploratory Data Analysis
        ts_analysis.perform_eda()

        # Split the data into training and testing datasets
        # Note: This step assumes you have implemented a method for splitting your dataset
        train_df, test_df = ts_analysis.split_data()  # Implement this method as needed

        # Fit the models on the training dataset
        ts_analysis.fit_arima(train_df)
        ts_analysis.fit_garch(train_df)
        ts_analysis.fit_lstm(train_df)

        # Evaluate the models on the testing dataset
        ts_analysis.evaluate_model(test_df)

        # Generate trading signals based on the latest data or evaluated models
        signal = ts_analysis.generate_trading_signals()
        print(f"Generated Trading Signal: {signal}")

        # Here, 'split_data', 'evaluate_model', and 'generate_trading_signals' methods
        # need to be defined in your TimeSeriesAnalysis class with the appropriate logic.


if __name__ == "__main__":
    main()
