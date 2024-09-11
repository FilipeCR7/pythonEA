-- Drop existing tables if they exist
DROP TABLE IF EXISTS `historical_data`;
DROP TABLE IF EXISTS `scorecard`;

-- Create the historical_data table
CREATE TABLE `historical_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime DEFAULT NULL,
  `open_price` decimal(10,5) DEFAULT NULL,
  `high_price` decimal(10,5) DEFAULT NULL,
  `low_price` decimal(10,5) DEFAULT NULL,
  `close_price` decimal(10,5) DEFAULT NULL,
  `volume` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Create the scorecard table
CREATE TABLE `scorecard` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    `signal` VARCHAR(10),  -- Escaping 'signal' to avoid conflict
    roi DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    win_loss_ratio DECIMAL(10, 4),
    rsi DECIMAL(5, 2),
    moving_average_cross VARCHAR(10),
    volatility_score DECIMAL(10, 4),
    lstm_rmse DECIMAL(10, 4),
    arima_rmse DECIMAL(10, 4),
    garch_volatility DECIMAL(10, 4),
    macd_signal VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
