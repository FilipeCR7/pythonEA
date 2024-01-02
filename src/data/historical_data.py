import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model  # For GARCH
from keras.models import Sequential  # For LSTM
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller  # For ADF test
import matplotlib.pyplot as plt  # For plotting
from src.data.db_connection import create_db_connection
from time_series_models import TimeSeriesAnalysis  # CREATE THIS FILE

class TimeSeriesAnalysis:
    def __init__(self, df):
        self.df = df
        self.df['returns'] = self.df['close'].pct_change()  # Calculate returns
        self.df.dropna(inplace=True)
        self.df_normalized = (self.df - self.df.mean()) / self.df.std()  # Normalize for LSTM

    def fetch_historical_data_from_db():
        query = "SELECT * FROM historical_data"  # Adjust the query as needed
        connection = create_db_connection()
        if connection is not None:
            try:
                return pd.read_sql(query, connection)
            finally:
                connection.close()
        return pd.DataFrame()

    # Step 3: Exploratory Data Analysis (EDA)
    def perform_eda(self):
        # Plot the closing prices
        self.df['close'].plot(title='Closing Prices')
        plt.show()

        # Augmented Dickey-Fuller test
        result = adfuller(self.df['close'])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

    # Step 4: Feature Engineering
    def feature_engineering(self):
        self.df['moving_avg'] = self.df['close'].rolling(window=20).mean()
        self.df['rsi'] = self.calculate_rsi()

    def calculate_rsi(self, window=14):
        delta = self.df['close'].diff()
        gain = (delta.clip(lower=0)).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Step 5: Model Development (ARIMA, GARCH, LSTM)
    def fit_arima(self, order=(5, 1, 0)):
        model = ARIMA(self.df['returns'], order=order)
        self.arima_model = model.fit(disp=0)
        print(self.arima_model.summary())

    def fit_garch(self):
        garch = arch_model(self.df['returns'], vol='Garch', p=1, q=1)
        self.garch_model = garch.fit()
        print(self.garch_model.summary())

    def create_lstm_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit_lstm(self):
        X, y = self.prepare_lstm_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = self.create_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, batch_size=1, epochs=20)
        self.lstm_model = model

    def prepare_lstm_data(self, window=60):
        X = []
        y = []
        for i in range(window, len(self.df_normalized)):
            X.append(self.df_normalized['returns'][i-window:i])
            y.append(self.df_normalized['returns'][i])
        return np.array(X), np.array(y)

    # Example of Model Evaluation
    def evaluate_model(self):
        # ARIMA Model Evaluation
        # GARCH Model Evaluation
        # LSTM Model Evaluation
        pass

    # Example of Trading Signal Generation
    def generate_trading_signals(self):
        # Logic for trading signal generation based on models
        pass

# Example usage
def main():
    # Fetch historical data from the database
    df = fetch_historical_data_from_db()

    # Assuming df has the necessary columns (like 'close') for time series analysis
    if not df.empty:
        ts_analysis = TimeSeriesAnalysis(df)

        # Perform your time series analysis steps
        ts_analysis.perform_eda()
        ts_analysis.feature_engineering()
        ts_analysis.fit_arima()
        # ... other steps ...

        # Example: generating signals or evaluating models
        signals = ts_analysis.generate_trading_signals()
        print("Generated Trading Signals:", signals)

if __name__ == "__main__":
    main()