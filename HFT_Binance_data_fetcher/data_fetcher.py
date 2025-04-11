import json
import time
#import datetime
import pandas as pd
import numpy as np
import websocket
import threading
from tqdm import tqdm
from datetime import datetime as dt
from datetime import timedelta
import os

class MarketDataCollector:
    def __init__(self, data_path='crypto_data'):
        """Initialize with local data directory"""
        self.combined_data = []
        self.data_path = data_path

        # Create directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Created directory: {data_path}")
        else:
            print(f"Using existing directory: {data_path}")
            
    def save_data(self, df, filename):
        """
        Save DataFrame to local directory

        Args:
            df: DataFrame to save
            filename: Name of the file to save
        """
        full_path = os.path.join(self.data_path, filename)
        print(f"Saving data to {full_path}...")

        # Make sure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Save the DataFrame to CSV
        df.to_csv(full_path)
        print(f"Successfully saved {len(df)} records to {filename}")

    def on_order_book_message(self, ws, message):
        """Process order book WebSocket messages"""
        try:
            data = json.loads(message)
            timestamp = dt.fromtimestamp(data['E']/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

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
        """Process trade WebSocket messages"""
        try:
            data = json.loads(message)
            timestamp = dt.fromtimestamp(data['E']/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            self.combined_data.append({
                'timestamp': timestamp,
                'bid_price': None,
                'ask_price': None,
                'trade_price': float(data['p']),
                'volume': float(data['q']) * float(data['p'])
            })
        except Exception as e:
            print(f"Error processing trade: {e}")

    def collect_long_duration_data(self, symbol, duration_hours=10, checkpoint_minutes=5):
        """
        Collect data for many hours with periodic checkpoints to prevent data loss

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            duration_hours: How long to collect data (in hours)
            checkpoint_minutes: How often to save checkpoints (in minutes)
        """
        total_seconds = duration_hours * 3600
        checkpoint_seconds = checkpoint_minutes * 60
        start_time = time.time()
        last_checkpoint = start_time

        print(f"Starting {duration_hours}-hour data collection for {symbol}...")
        print(f"Will save checkpoints every {checkpoint_minutes} minutes")
        print(f"Expected completion: {dt.now() + timedelta(hours=duration_hours)}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_status_code} - {close_msg}")

        def on_open(ws):
            print(f"WebSocket connection established")

        # Define WebSocket functions with proper references to self
        def run_order_book_stream():
            while time.time() - start_time < total_seconds:
                try:
                    ws = websocket.WebSocketApp(
                        f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth@100ms",
                        on_message=self.on_order_book_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open
                    )
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                    print("Order book WebSocket disconnected. Reconnecting...")
                    time.sleep(3)  # Wait before reconnecting
                except Exception as e:
                    print(f"Order book WebSocket error: {e}")
                    time.sleep(3)  # Wait before reconnecting

        def run_trade_stream():
            while time.time() - start_time < total_seconds:
                try:
                    ws = websocket.WebSocketApp(
                        f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade",
                        on_message=self.on_trade_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open
                    )
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                    print("Trade WebSocket disconnected. Reconnecting...")
                    time.sleep(3)  # Wait before reconnecting
                except Exception as e:
                    print(f"Trade WebSocket error: {e}")
                    time.sleep(3)  # Wait before reconnecting

        threads = [
            threading.Thread(target=run_order_book_stream),
            threading.Thread(target=run_trade_stream)
        ]

        for t in threads:
            t.daemon = True
            t.start()

        # Show progress with tqdm
        pbar = tqdm(total=total_seconds)
        checkpoint_count = 0

        try:
            while time.time() - start_time < total_seconds:
                elapsed = time.time() - last_checkpoint
                if elapsed >= checkpoint_seconds and len(self.combined_data) > 0:
                    # Process and save a checkpoint
                    checkpoint_count += 1
                    print(f"\nCreating checkpoint {checkpoint_count}...")
                    df = self.process_data()
                    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{symbol}_checkpoint_{checkpoint_count}_{timestamp}.csv"
                    self.save_data(df, filename)
                    last_checkpoint = time.time()

                    # Keep the most recent data to preserve state but free memory
                    if len(self.combined_data) > 100000:
                        print(f"Trimming data array from {len(self.combined_data)} to 10000 records")
                        self.combined_data = self.combined_data[-10000:]

                # Update progress bar
                time_passed = int(time.time() - start_time)
                pbar.update(1)
                time.sleep(1)

                # Print data collection status every minute
                if time_passed % 60 == 0 and time_passed > 0:
                    print(f"\nData collected so far: {len(self.combined_data)} records")

        except KeyboardInterrupt:
            print("Collection interrupted by user!")
        finally:
            pbar.close()

            # Process final dataset
            if len(self.combined_data) > 0:
                print("Processing final dataset...")
                df = self.process_data()

                # Save final dataset
                final_timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"{symbol}_FINAL_{duration_hours}h_{final_timestamp}.csv"
                self.save_data(df, final_filename)

                return df
            else:
                print("No data collected!")
                return None

    def process_data(self):
        """Process and clean the collected data"""
        print(f"Processing {len(self.combined_data)} data points...")

        # Create DataFrame from collected data
        df = pd.DataFrame(self.combined_data)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Check for duplicates and make index unique if needed
        if df['timestamp'].duplicated().any():
            print(f"Found {df['timestamp'].duplicated().sum()} duplicate timestamps. Making index unique...")
            # Sort first to ensure proper order
            df = df.sort_values('timestamp')
            # Add a small increment to duplicated timestamps to make them unique
            # This creates a new column with adjusted timestamps
            mask = df['timestamp'].duplicated(keep='first')
            dup_count = mask.sum()
            increments = pd.Series(np.arange(1, dup_count + 1)) * pd.Timedelta(microseconds=1)
            df.loc[mask, 'adjusted_timestamp'] = df.loc[mask, 'timestamp'] + pd.Series(increments).values
            df.loc[~mask, 'adjusted_timestamp'] = df.loc[~mask, 'timestamp']
            # Use the adjusted timestamp as index
            df = df.set_index('adjusted_timestamp').sort_index()
        else:
            # No duplicates, proceed normally
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

        # Calculate mid price
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2

        print(f"Processed to {len(df)} data points")
        return df

# Example usage
if __name__ == "__main__":
    collector = MarketDataCollector(data_path="crypto_data")
    collector.collect_long_duration_data("btcusdt", duration_hours=2, checkpoint_minutes=5)