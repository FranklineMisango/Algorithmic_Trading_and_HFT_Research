import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import requests
import threading
import websocket
import time
from datetime import datetime, timedelta
import json
class MarketDataCollector:

    def __init__(self):
        self.combined_data = []

    def on_order_book_message(self, ws, message):
        try:
            data = json.loads(message)
            timestamp = datetime.fromtimestamp(data['E']/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            if data.get('b') and data.get('a'):
                self.combined_data.append({
                    'timestamp': timestamp,
                    'bid_price': float(data['b'][0][0]),
                    'ask_price': float(data['a'][0][0]),
                    'trade_price': None,
                    'volume': None
                })
        except Exception as e:
            print(f"Error processing order book: {e}")

    def on_trade_message(self, ws, message):
        try:
            data = json.loads(message)
            timestamp = datetime.fromtimestamp(data['E']/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            self.combined_data.append({
                'timestamp': timestamp,
                'bid_price': None,
                'ask_price': None,
                'trade_price': float(data['p']),
                'volume': float(data['q']) * float(data['p'])
            })
        except Exception as e:
            print(f"Error processing trade: {e}")

    def collect_data(self, symbol, duration_minutes):
        def run_order_book_stream():
            ws = websocket.WebSocketApp(
                f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth@100ms",
                on_message=self.on_order_book_message
            )
            ws.run_forever()

        def run_trade_stream():
            ws = websocket.WebSocketApp(
                f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade",
                on_message=self.on_trade_message
            )
            ws.run_forever()

        threads = [
            threading.Thread(target=run_order_book_stream),
            threading.Thread(target=run_trade_stream)
        ]

        for t in threads:
            t.daemon = True
            t.start()

        time.sleep(duration_minutes * 60)
        return self.process_data()

    def process_data(self):
        df = pd.DataFrame(self.combined_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Forward fill order book data and backward fill trade data
        df['bid_price'] = df['bid_price'].ffill()
        df['ask_price'] = df['ask_price'].ffill()
        df['trade_price'] = df['trade_price'].bfill()
        df['volume'] = df['volume'].bfill()

        # Resample to regular intervals (e.g., 100ms)
        df = df.resample('100ms').agg({
            'bid_price': 'first',
            'ask_price': 'first',
            'trade_price': 'last',
            'volume': 'sum'
        })
        # Use explicit ffill() instead of fillna(method='ffill')
        df = df.ffill()

        return df



# Fetch market data for a specific symbol and duration
if __name__ == "__main__":
    symbol = "ethusdt"  # Example symbol
    duration_minutes = 60
    collector = MarketDataCollector()
    df = collector.collect_data(symbol, duration_minutes)
    if df is not None:
        print(f"Collected {len(df)} records.")
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_data_{timestamp}.csv"
        df.to_csv(filename, index=True)
        print(f"Data saved to {filename}")

        print(df.head())
    else:
        print("No data collected.")