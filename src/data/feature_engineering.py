import pandas as pd
import numpy as np

def add_moving_average(df, window=20):
    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
    return df

def add_exponential_moving_average(df, window=20):
    df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    return df

def add_rsi(df, window=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()

    df['macd'] = short_ema - long_ema
    df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    return df

def add_bollinger_bands(df, window=20, std_multiplier=2):
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_multiplier)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_multiplier)
    return df

def add_historical_volatility(df, window=10):
    df['hist_volatility'] = df['close'].pct_change().rolling(window=window).std() * np.sqrt(window)
    return df

def add_cci(df, window=20):
    TP = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (TP - TP.rolling(window=window).mean()) / (0.015 * TP.rolling(window=window).std())
    return df

def add_obv(df):
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return df

def add_pivot_points(df):
    df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['resistance1'] = 2 * df['pivot_point'] - df['low']
    df['support1'] = 2 * df['pivot_point'] - df['high']
    # Add more levels if needed
    return df

# Additional functions for other indicators can be added here.
