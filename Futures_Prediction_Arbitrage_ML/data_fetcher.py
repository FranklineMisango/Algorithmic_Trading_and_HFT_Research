import websocket
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import threading
import os
import signal
import sys

# Configuration
SYMBOL = 'btcusdt'
DEPTH_LEVELS = 20
UPDATE_SPEED = 100  # ms
DATA_FILE = 'live_order_book_data.csv'
MAX_ROWS = 100000  # Keep only recent data to avoid file growing too large

ws_url = f'wss://fstream.binance.com/ws/{SYMBOL}@depth{DEPTH_LEVELS}@{UPDATE_SPEED}ms'

# Global variables
order_book_data = []
data_lock = threading.Lock()
running = True

def signal_handler(sig, frame):
    global running
    print('Stopping data collection...')
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def on_message(ws, message):
    global order_book_data, running
    if not running:
        return
    print(f"Received message, len={len(message)}")
    try:
        data = json.loads(message)
        with data_lock:
            order_book_data.append({
                'timestamp': datetime.now(),
                'bids': data['b'],
                'asks': data['a'],
                'lastUpdateId': data['u']
            })
            # Keep only recent data in memory
            if len(order_book_data) > 1000:
                order_book_data = order_book_data[-1000:]
    except Exception as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Connection closed")
    if running:
        print("Attempting to reconnect...")
        time.sleep(5)
        start_websocket()

def on_open(ws):
    print("WebSocket Connection opened")

def start_websocket():
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()

def save_data_periodically():
    global order_book_data, running
    while running:
        time.sleep(10)  # Save every 10 seconds for testing
        print(f"Save thread: checking data, len={len(order_book_data)}")
        with data_lock:
            if order_book_data:
                print(f"Saving {len(order_book_data)} records...")
                df = pd.DataFrame(order_book_data)
                df = pd.DataFrame(order_book_data)
                # Parse bids and asks
                def parse_levels(levels):
                    prices = [float(level[0]) for level in levels]
                    quantities = [float(level[1]) for level in levels]
                    return prices, quantities

                df['bid_prices'], df['bid_quantities'] = zip(*df['bids'].apply(parse_levels))
                df['ask_prices'], df['ask_quantities'] = zip(*df['asks'].apply(parse_levels))

                # Append to file or create new
                if os.path.exists(DATA_FILE):
                    existing_df = pd.read_csv(DATA_FILE)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    # Keep only recent MAX_ROWS
                    if len(combined_df) > MAX_ROWS:
                        combined_df = combined_df.tail(MAX_ROWS)
                    combined_df.to_csv(DATA_FILE, index=False)
                else:
                    df.to_csv(DATA_FILE, index=False)

                print(f"Saved {len(df)} new records to {DATA_FILE}. Total records: {len(combined_df) if 'combined_df' in locals() else len(df)}")
                order_book_data = []  # Clear after saving

if __name__ == "__main__":
    print("Starting background data fetcher for Binance Futures Order Book...")
    print(f"Symbol: {SYMBOL}, Depth: {DEPTH_LEVELS}, Update Speed: {UPDATE_SPEED}ms")
    print("Press Ctrl+C to stop.")

    # Start saving thread
    save_thread = threading.Thread(target=save_data_periodically, daemon=True)
    save_thread.start()

    # Start WebSocket
    start_websocket()