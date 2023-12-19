import signal
import time

# Flag to control the main loop
running = True

def signal_handler(signum, frame):
    """
    Signal handler for graceful shutdown.
    """
    global running
    print("Signal received, shutting down...")
    running = False

def main_bot_logic():
    """
    Main trading bot logic goes here.
    """
    # Replace this with your bot's main logic
    print("Bot is running...")
    time.sleep(0.1)  # Simulate work with a sleep

def main():
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    while running:
        try:
            main_bot_logic()
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, break or continue based on the nature of the error

    print("Bot stopped.")

if __name__ == "__main__":
    main()
