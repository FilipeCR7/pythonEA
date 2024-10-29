import backtrader as bt
import pandas as pd
import numpy as np

class MeanReversionStrategy(bt.Strategy):
    params = (
        ('window', 20),
        ('std_dev_multiplier', 2),
        ('short_window', 20),
        ('long_window', 50),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close

        self.sma = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.window)
        self.std_dev = bt.indicators.StandardDeviation(self.dataclose, period=self.params.window)
        self.lower_bound = self.sma - (self.params.std_dev_multiplier * self.std_dev)
        self.upper_bound = self.sma + (self.params.std_dev_multiplier * self.std_dev)

        self.rsi = bt.indicators.RelativeStrengthIndex(self.dataclose, period=14)

        self.ma_short = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.short_window)
        self.ma_long = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.long_window)

        self.order = None
        self.returns = []
        self.wins = 0
        self.losses = 0

    def next(self):
        if self.order:
            return

        current_price = self.dataclose[0]
        lower_bound = self.lower_bound[0]
        upper_bound = self.upper_bound[0]

        if not self.position:
            if current_price <= lower_bound:
                self.order = self.buy()
            elif current_price >= upper_bound:
                self.order = self.sell()
        else:
            if self.position.size > 0 and current_price >= self.sma[0]:
                self.order = self.close()
            elif self.position.size < 0 and current_price <= self.sma[0]:
                self.order = self.close()

    def calculate_performance_metrics(self):
        roi = (self.broker.get_value() / self.broker.startingcash) - 1

        if len(self.returns) > 1:
            returns = np.array(self.returns)
            risk_free_rate = 0.01 / 252
            excess_returns = returns - risk_free_rate
            mean_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252) if std_excess_return != 0 else 0.0
        else:
            sharpe_ratio = 0.0

        daily_returns = pd.Series(self.dataclose.get(size=len(self.dataclose))).pct_change().dropna()
        volatility_score = daily_returns.rolling(window=20).std().mean() * np.sqrt(252)

        rsi_value = self.rsi[0]

        if self.ma_short[0] > self.ma_long[0]:
            ma_cross_signal = 'BUY'
        elif self.ma_short[0] < self.ma_long[0]:
            ma_cross_signal = 'SELL'
        else:
            ma_cross_signal = 'HOLD'

        total_trades = self.wins + self.losses
        win_loss_ratio = self.wins / total_trades if total_trades > 0 else None

        return {
            'roi': round(roi, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'volatility_score': round(volatility_score, 4),
            'rsi_value': round(rsi_value, 2),
            'ma_cross_signal': ma_cross_signal,
            'win_loss_ratio': win_loss_ratio,
        }

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        profit = trade.pnl
        self.returns.append(profit)
        if profit > 0:
            self.wins += 1
        else:
            self.losses += 1

    def stop(self):
        metrics = self.calculate_performance_metrics()
        print(f'ROI: {metrics["roi"]}, Sharpe Ratio: {metrics["sharpe_ratio"]}, '
              f'Volatility Score: {metrics["volatility_score"]}, RSI: {metrics["rsi_value"]}, '
              f'MA Cross Signal: {metrics["ma_cross_signal"]}, '
              f'Win/Loss Ratio: {metrics["win_loss_ratio"]}')

