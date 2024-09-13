import argparse
import importlib
import sys
import pandas as pd

# Add the 'src' directory to the system path
sys.path.insert(0, './src')

try:
    from algorithms.mean_reversion import MeanReversionStrategy
    from algorithms.time_series_analysis import TimeSeriesAnalysis
    from src.data.db_connection import fetch_historical_data
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)


def run_mean_reversion_test():
    # Fetch data from the database
    df = fetch_historical_data()

    if df is None or df.empty:
        print("No data fetched from the database.")
        return

    # Create a DataFrame with a simple moving average
    df['sma_20'] = df['close'].rolling(window=20).mean()

    # Fill the initial NaN values for the moving average using bfill()
    df['sma_20'].bfill(inplace=True)

    # Initialize and run the Mean Reversion Strategy
    strategy = MeanReversionStrategy(df, window=20, std_dev_multiplier=2)
    signal = strategy.generate_signals()

    print(f"Generated signal: {signal}")


def run_time_series_analysis_test():
    # Suppress specific warnings
    import warnings
    warnings.filterwarnings('ignore', message=".*A date index has been provided, but it has no associated frequency information.*")
    warnings.filterwarnings('ignore', message=".*y is poorly scaled, which may affect convergence of the optimizer.*")

    # Adjust logging level
    import logging
    logging.getLogger('statsmodels').setLevel(logging.ERROR)
    logging.getLogger('arch').setLevel(logging.ERROR)

    # Fetch data from the database
    df = fetch_historical_data()

    if df is None or df.empty:
        print("No data fetched from the database.")
        return

    # Data cleaning and type conversion
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close']).reset_index()

    # Initialize TimeSeriesAnalysis with the cleaned DataFrame
    ts_analysis = TimeSeriesAnalysis(df)

    # Fit ARIMA model
    try:
        ts_analysis.fit_arima()
    except Exception as e:
        pass  # Suppress error

    # Fit GARCH model
    try:
        ts_analysis.fit_garch()
    except Exception as e:
        pass  # Suppress error

    # Train the LSTM model
    try:
        ts_analysis.fit_lstm()
    except Exception as e:
        pass  # Suppress error



def main():
    parser = argparse.ArgumentParser(description="Run a script's main function")
    parser.add_argument('--script', help='Name of the script to run (without .py)', required=True)
    args = parser.parse_args()

    script_name = args.script

    if script_name == 'mean_reversion_test':
        run_mean_reversion_test()
    elif script_name == 'time_series_analysis_test':
        run_time_series_analysis_test()
    else:
        try:
            # Dynamically import the module
            module = importlib.import_module(f"algorithms.{script_name}")

            # Check if the module has a main() function and call it
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"The script {script_name} does not have a main() function.")
        except ModuleNotFoundError:
            print(f"Script {script_name} not found.")


if __name__ == "__main__":
    main()
