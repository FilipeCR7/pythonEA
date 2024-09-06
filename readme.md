# Python EA

## Setup Instructions
To set up the database, run:
```bash
sh database/install.sh

algorithms/mean_reversion.py
Description:
This file contains the implementation of the Mean Reversion Strategy, a basic trading algorithm that generates buy/sell signals based on the price deviating from its moving average. This strategy is commonly used when prices tend to revert to the mean over time.

algorithms/statistical_arbitrage.py
Description:
Implements the Statistical Arbitrage Strategy, where two currency pairs are compared for spread deviations. The algorithm identifies buy/sell opportunities when the spread between two pairs deviates significantly from its historical mean.

algorithms/time_series_analysis.py
Description:
This file includes methods to perform Time Series Analysis on financial data. It provides models like ARIMA, GARCH, and LSTM to analyze and forecast price movements and volatility. It also includes methods for preparing and normalizing time-series data for modeling.

data/db_connection.py
Description:
Handles database connections using SQLAlchemy. The file provides functions to connect to a MySQL database, insert data into tables, and fetch historical financial data for analysis. It ensures proper database management throughout the application's lifecycle.

data/feature_engineering/indicators.py
Description:
Provides functions to add common technical indicators to financial data, such as moving averages, Bollinger Bands, RSI, and others. These indicators enhance trading strategies by identifying trends, momentum, and volatility in market data.

api/oanda_api.py
Description:
Defines a class to interact with the OANDA API, a popular trading platform. It allows retrieval of account details, real-time pricing, and the ability to place various orders (market, limit, stop). It also includes methods for fetching historical price data.

utils/data_utils.py
Description:
Prepares time-series data for analysis by fetching historical financial data from a MySQL database. The data is properly indexed and formatted into a Pandas DataFrame used for modeling and analysis across the project.

utils/scheduled_scraping.py
Description:
Schedules the web scraping of articles from a specified website (e.g., Reuters) every 15 minutes using the schedule library. After scraping the articles, it performs sentiment analysis on each article and logs the results.

utils/sentiment_analysis_utils.py
Description:
A utility module providing sentiment analysis functionality using the TextBlob library. It calculates the sentiment polarity (a value between -1 and 1) for a given text, helping determine if the text has a positive, neutral, or negative sentiment.

utils/web_scraping_utils.py
Description:
Contains functions to scrape news articles from a specified website (e.g., Reuters). It fetches article URLs and scrapes the content of each article. The scraped data can be used for news-based sentiment analysis or other trading strategies.

trader/bot.py
Description:
Implements the core logic of the trading bot, including signal handling for graceful shutdown. This file includes a placeholder for the main trading logic, which can be integrated with other strategies or models.

analysis/eda.py
Description:
Provides tools for exploratory data analysis (EDA) of financial data. It includes functions to plot time-series data and check for stationarity using the Augmented Dickey-Fuller (ADF) test. This helps understand the properties of financial time series before modeling.