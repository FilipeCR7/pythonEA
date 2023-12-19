DROP TABLE IF EXISTS `historical_data`;

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