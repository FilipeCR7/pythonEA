# src/backtesting/backtest.py

import sys
import os

# Add the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import backtrader as bt
import pandas as pd
from src.algorithms.combined_test import CombinedStrategy
from src.data.db_connection import fetch_historical_data

# Rest of your backtest.py code...

def run_backtest(strategy, cash=100000.0, commission=0.0, printlog=False):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Fetch and prepare data
    df = fetch_historical_data()
    if df.empty:
        print("No data fetched from the database.")
        return

    # Prepare DataFrame for Backtrader
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
    df.rename(columns={
        'open_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'close_price': 'close',
    }, inplace=True)
    df['openinterest'] = 0  # Required column for Backtrader

    # Convert to numeric and drop NaNs
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    # Create Data Feed
    data = bt.feeds.PandasData(dataname=df)

    # Add data and strategy to Cerebro
    cerebro.adddata(data)
    cerebro.addstrategy(strategy, printlog=printlog)

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Extract analyzers
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    # Print analysis
    print(f"Sharpe Ratio (Analyzer): {sharpe.get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {drawdown.drawdown:.2f}%")
    print(f"Total Trades: {trades.total.closed}")
    print(f"Wins: {trades.won.total}, Losses: {trades.lost.total}")
    win_rate = (trades.won.total / trades.total.closed * 100) if trades.total.closed > 0 else 0
    print(f"Win Rate: {win_rate:.2f}%")

    # Plot results
    cerebro.plot()

if __name__ == '__main__':
    run_backtest(CombinedStrategy, printlog=True)
