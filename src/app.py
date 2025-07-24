# src/app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import backtrader as bt
import sys
import os

# Adjust the system path to include your project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import your existing modules
from src.algorithms.combined_test import CombinedStrategy
from src.data.db_connection import fetch_historical_data

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/get_price_data', methods=['GET'])
def get_price_data():
    # Fetch data
    df = fetch_historical_data()
    if df.empty:
        return jsonify({'error': 'No data available.'})

    # Print the columns of the DataFrame for debugging
    print("DataFrame columns:", df.columns.tolist())

    # Prepare data
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df.sort_values('datetime', inplace=True)
    df = df[['datetime', 'open', 'high', 'low', 'close']]

    # Rename columns to match expected names in frontend
    df.rename(columns={
        'open': 'open_price',
        'high': 'high_price',
        'low': 'low_price',
        'close': 'close_price'
    }, inplace=True)

    df['datetime'] = df['datetime'].astype(str)

    # Convert DataFrame to dictionary
    price_data = df.to_dict(orient='records')

    # Return JSON response
    return jsonify({'price_data': price_data})

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    data = request.get_json()

    # Extract parameters from the request
    mean_rev_window = data.get('mean_rev_window', 20)
    std_dev_multiplier = data.get('std_dev_multiplier', 2)
    arima_order = tuple(data.get('arima_order', [5, 1, 0]))
    # Add more parameters as needed

    # Fetch data
    df = fetch_historical_data()
    if df.empty:
        return jsonify({'error': 'No data available for backtesting.'})

    # Prepare data for Backtrader (using your existing code)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df['openinterest'] = 0

    # Create data feed
    data_feed = bt.feeds.PandasData(dataname=df)

    # Set up Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0)

    # Add strategy with parameters
    cerebro.addstrategy(
        CombinedStrategy,
        mean_rev_window=mean_rev_window,
        std_dev_multiplier=std_dev_multiplier,
        arima_order=arima_order,
        printlog=False
    )

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Extract performance metrics
    total_return = cerebro.broker.getvalue() - 100000.0
    trades = strat.analyzers.trades.get_analysis()
    returns = strat.analyzers.timereturn.get_analysis()

    # Prepare equity curve data
    equity_curve = pd.Series(returns).cumprod() * 100000.0
    equity_curve = equity_curve.reset_index().rename(columns={'index': 'datetime', 0: 'equity'})
    equity_curve['datetime'] = equity_curve['datetime'].astype(str)

    # Prepare response
    response = {
        'total_return': total_return,
        'trades': trades,
        'equity_curve': equity_curve.to_dict(orient='records'),
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
