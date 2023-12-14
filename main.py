from api.oanda_api import OandaAPI
# Import other necessary modules, like strategies or utility functions

def main():
    # Initialize the OandaAPI instance
    account_id = "101-004-27721570-001"  # Replace with your actual account ID
    oanda_api = OandaAPI(account_id)

    # Example: Get account details
    account_details = oanda_api.get_account_details()
    print(account_details)

    # Example: Get pricing for USD/CAD
    pricing_info = oanda_api.get_pricing("USD_CAD")
    print(pricing_info)

    # Here, you can also initialize and execute your trading strategies,
    # perform data analysis, backtesting, etc.
    # Initialization and existing code...

    # Test creating a market order
    oanda_api.create_market_order("USD_CAD", 1000)

    # Test creating a limit order
    oanda_api.create_limit_order("USD_CAD", 1000, 1.1500)

    # Test creating a stop order
    oanda_api.create_stop_order("USD_CAD", -1000, 1.1400)

if __name__ == "__main__":
    main()