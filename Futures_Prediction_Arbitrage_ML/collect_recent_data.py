import websocket
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import threading

# WebSocket URL for Binance Futures Order Book
SYMBOL = 'btcusdt'
DEPTH_LEVELS = 20
UPDATE_SPEED = 50  # ms

ws_url = f'wss://fstream.binance.com/ws/{SYMBOL}@depth{DEPTH_LEVELS}@{UPDATE_SPEED}ms'

# Global variables to store data
order_book_data = []
data_lock = threading.Lock()

def on_message(ws, message):
    data = json.loads(message)
    with data_lock:
        order_book_data.append({
            'timestamp': datetime.now(),
            'bids': data['b'],
            'asks': data['a'],
            'lastUpdateId': data['u']
        })

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")

def start_websocket():
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()

# Start WebSocket in background thread
ws_thread = threading.Thread(target=start_websocket)
ws_thread.daemon = True
ws_thread.start()

# Collect data for a specified time (e.g., 1 hour)
collection_time = 3600  # seconds
start_time = time.time()

print(f"Collecting data for {collection_time} seconds...")
while time.time() - start_time < collection_time:
    time.sleep(1)  # Check every second
    with data_lock:
        if len(order_book_data) % 1000 == 0 and len(order_book_data) > 0:
            print(f"Collected {len(order_book_data)} snapshots...")

# Save data
with data_lock:
    df = pd.DataFrame(order_book_data)

def parse_levels(levels):
    prices = [float(level[0]) for level in levels]
    quantities = [float(level[1]) for level in levels]
    return prices, quantities

df['bid_prices'], df['bid_quantities'] = zip(*df['bids'].apply(parse_levels))
df['ask_prices'], df['ask_quantities'] = zip(*df['asks'].apply(parse_levels))

df.to_csv('recent_order_book_data.csv', index=False)
print(f"Saved {len(df)} snapshots to 'recent_order_book_data.csv'")