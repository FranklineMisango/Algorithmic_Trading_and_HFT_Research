import sys
import os
import pandas as pd
import numpy as np
import time
import queue
import threading
from datetime import datetime, timedelta
import json
import time
import pandas as pd
import websocket
import threading
from datetime import datetime as dt, timedelta
import os
import traceback

class MarketDataCollector:
    def __init__(self, data_path='crypto_data'):
        self.combined_data = []
        self.data_path = data_path
        self.running = True  
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Created directory: {data_path}")
        else:
            print(f"Using existing directory: {data_path}")

    def save_data(self, df, filename):
        """!IMPROVEMENT: Added error handling for file operations"""
        try:
            full_path = os.path.join(self.data_path, filename)
            print(f"Saving data to {full_path}...")
            df.to_csv(full_path, index=False)  #
            print(f"Successfully saved {len(df)} records")
        except Exception as e:
            print(f"Save failed: {str(e)[:200]}")

   
    def _ws_handler(self, message, data_type):
        try:
            data = json.loads(message)
            timestamp = dt.fromtimestamp(data['E']/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Add safety checks for order book data
            bid_price = None
            ask_price = None
            if data_type == 'depth':
                if len(data.get('b', [])) > 0:
                    bid_price = float(data['b'][0][0])
                if len(data.get('a', [])) > 0:
                    ask_price = float(data['a'][0][0])

            record = {
                'timestamp': timestamp,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'trade_price': float(data['p']) if data_type == 'trade' else None,
                'volume': float(data['q'])*float(data['p']) if data_type == 'trade' else None
            }
            self.combined_data.append(record)
        except Exception as e:
            print(f"Processing error: {str(e)[:200]}")


    def on_order_book_message(self, ws, message):
        self._ws_handler(message, 'depth')

    def on_trade_message(self, ws, message):
        self._ws_handler(message, 'trade')

   
    def _run_websocket(self, url, handler, stream_type):
        end_time = time.time() + (3600 * 24)  # Fail-safe timeout
        while time.time() < end_time and self.running:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=handler,
                    on_error=lambda ws, e: print(f"{stream_type} error: {str(e)[:200]}"),
                    on_close=lambda ws: print(f"{stream_type} closed"),
                    on_open=lambda ws: print(f"{stream_type} connected")
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"{stream_type} failure: {str(e)[:200]}")
            time.sleep(5) 

    def collect_long_duration_data(self, symbol, duration_hours=0.1, checkpoint_minutes=1):
       
        total_seconds = duration_hours * 3600
        checkpoint_seconds = checkpoint_minutes * 60
        start_time = time.time()
        end_time = start_time + total_seconds  
        last_checkpoint = start_time
        checkpoint_count = 0
        
        print(f"\n{'='*40}\nStarting {duration_hours}-hour collection for {symbol}")
        print(f"Checkpoints every {checkpoint_minutes} mins | Target end: {dt.fromtimestamp(end_time)}\n{'='*40}")

        
        threads = [
            threading.Thread(target=self._run_websocket,
                args=(f"wss://stream.binance.com:9443/ws/{symbol}@depth@100ms", 
                    self.on_order_book_message, "OrderBook")),
            threading.Thread(target=self._run_websocket,
                args=(f"wss://stream.binance.com:9443/ws/{symbol}@trade", 
                    self.on_trade_message, "Trades"))
        ]

        for t in threads:
            t.daemon = True
            t.start()

        alive_threads = sum(1 for t in threads if t.is_alive())
        print(f"Active connections: {alive_threads}/2 | Buffer size: {len(self.combined_data)}")

        try:
            while time.time() < end_time and self.running:
                # !IMPROVEMENT: Adaptive sleep management
                remaining = end_time - time.time()
                sleep_time = max(0, min(1, remaining))
                time.sleep(sleep_time)

                # Checkpoint handling
                if time.time() - last_checkpoint >= checkpoint_seconds:
                    checkpoint_count += 1
                    last_checkpoint = time.time()
                    self._process_checkpoint(symbol, checkpoint_count)

                # !IMPROVEMENT: Status updates
                if int(time.time() - start_time) % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = min(100, (elapsed / total_seconds) * 100)
                    print(f"Progress: {progress:.1f}% | Records: {len(self.combined_data)}")

        except KeyboardInterrupt:
            print("\nUser requested shutdown...")
        finally:
            self.running = False
            self._final_save(symbol, duration_hours)
            print("\nCollection completed" if time.time() >= end_time else "\nCollection aborted")

    def _process_checkpoint(self, symbol, count):
        try:
            print(f"\n{'='*20} Checkpoint {count} {'='*20}")
            original_count = len(self.combined_data)
            
            # Process COPY of data
            temp_data = self.combined_data.copy()
            df = self.process_data()
            
            if not df.empty:
                # Only clear original data AFTER successful processing
                self.combined_data = self.combined_data[original_count:]  # Keep unprocessed data
                timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_checkpoint_{count}_{timestamp}.csv"
                self.save_data(df, filename)
                
            # Memory management (preserve last 10k)
            if len(self.combined_data) > 10000:
                self.combined_data = self.combined_data[-10000:]
                
        except Exception as e:
            print(f"Checkpoint failed: {str(e)[:200]}")
            traceback.print_exc()


    def process_data(self):
        try:
            if not self.combined_data:
                return pd.DataFrame()

            batch = pd.DataFrame(self.combined_data.copy())
            batch['timestamp'] = pd.to_datetime(batch['timestamp'])
            batch = batch.drop_duplicates(subset=['timestamp'], keep='last')

            # ==== New Validation Checks ====
            required_columns = {'bid_price', 'ask_price', 'trade_price', 'volume'}
            if not required_columns.issubset(batch.columns):
                missing = required_columns - set(batch.columns)
                print(f"Missing columns: {missing}")
                return pd.DataFrame()
            
            if not batch.empty:
                time_span = batch['timestamp'].max() - batch['timestamp'].min()
                if time_span < pd.Timedelta('1s'):
                    print(f"Critical time range error: {time_span}")
                    return pd.DataFrame()
            # ==============================

            if not batch.empty:
                resampled = (
                    batch.set_index('timestamp')
                    .resample('100ms', origin='start')
                    .agg({
                        'bid_price': 'ffill',
                        'ask_price': 'ffill',
                        'trade_price': 'bfill',
                        'volume': 'sum'
                    })
                    .reset_index()
                )
                resampled['mid_price'] = (resampled['bid_price'] + resampled['ask_price']) / 2
                resampled = resampled.ffill().dropna(subset=['bid_price', 'ask_price'], how='all')
                
                print(f"Processed {len(batch)} records -> {len(resampled)} data points")
                return resampled
            return pd.DataFrame()
        except Exception as e:
            print(f"Processing error: {str(e)[:200]}")
            traceback.print_exc()
            return pd.DataFrame()


    def _final_save(self, symbol, duration):
      
        try:
            print("\nFinalizing collection...")
            df = self.process_data()
            if not df.empty:
                timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_FINAL_{duration}h_{timestamp}.csv"
                self.save_data(df, filename)
        except Exception as e:
            print(f"Final save failed: {str(e)[:200]}")



class DataProcessor:
    """
    Processes market data using the existing MarketDataCollector
    """
    
    def __init__(self, data_path, symbol):
        """
        Initialize data processor.
        
        Args:
            data_path (str): Path to data directory
            symbol (str): Trading symbol to process
        """
        self.data_path = data_path
        self.symbol = symbol.lower()
        
        # Create absolute path for data
        abs_data_path = os.path.abspath(os.path.expanduser(data_path))
        print(f"Using data path: {abs_data_path}")
        
        # Initialize the MarketDataCollector
        self.collector = MarketDataCollector(data_path=abs_data_path)
        
        # For storing the most recent data
        self.buffer = pd.DataFrame()
        self.buffer_max_size = 1000  # Maximum points to keep in buffer
    
    def process_live_data(self, data_queue, stop_event, update_frequency=0.1):
        """
        Process data in a continuous loop, putting results in the queue
        
        Args:
            data_queue (Queue): Queue to put processed data in
            stop_event (Event): Event to signal thread to stop
            update_frequency (float): Update frequency in seconds
        """
        print(f"Starting live data collection for {self.symbol} with {update_frequency}s update frequency")
        
        # Start the collector in a separate thread
        collector_thread = threading.Thread(
            target=self._run_collector,
            args=(stop_event,),
            daemon=True
        )
        collector_thread.start()
        
        # Process collected data and forward to the dashboard
        update_count = 0
        while not stop_event.is_set():
            try:
                # Get any new data
                if self.collector.combined_data:
                    # Copy and process the data
                    new_data = pd.DataFrame(self.collector.combined_data.copy())
                    if not new_data.empty:
                        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
                        new_data = new_data.drop_duplicates(subset=['timestamp'], keep='last')
                        
                        # Calculate mid price
                        new_data['mid_price'] = (pd.to_numeric(new_data['bid_price'], errors='coerce') + 
                                            pd.to_numeric(new_data['ask_price'], errors='coerce')) / 2
                        
                        # Add to our buffer
                        self.buffer = pd.concat([self.buffer, new_data]).drop_duplicates(subset=['timestamp'], keep='last')
                        
                        # Trim buffer to max size
                        if len(self.buffer) > self.buffer_max_size:
                            self.buffer = self.buffer.iloc[-self.buffer_max_size:]
                        
                        # Send to dashboard queue
                        try:
                            # Put data in queue but don't block if full
                            data_queue.put(self.buffer, block=False)
                            update_count += 1
                            if update_count % 10 == 0:  # Print every 10 updates
                                print(f"Updated data queue: {len(self.buffer)} records ({update_count} updates)")
                        except queue.Full:
                            print("Dashboard queue full, skipping update")
                        
                        # Clear collector's buffer periodically to prevent memory issues
                        if len(self.collector.combined_data) > 5000:
                            self.collector.combined_data = self.collector.combined_data[-1000:]
                
                # Sleep to maintain update frequency
                time.sleep(update_frequency)
        
            except Exception as e:
                print(f"Error in live data processing: {str(e)[:200]}")
                traceback.print_exc()
                time.sleep(1)  # Pause briefly if there's an error
        
    def _run_collector(self, stop_event):
        """Run the MarketDataCollector in the background"""
        try:
            # Modify the collector to run without time limit
            self.collector.running = True
            
            # Set up the websocket connections
            threads = [
                threading.Thread(
                    target=self.collector._run_websocket,
                    args=(f"wss://stream.binance.com:9443/ws/{self.symbol}@depth@100ms", 
                         self.collector.on_order_book_message, "OrderBook")),
                threading.Thread(
                    target=self.collector._run_websocket,
                    args=(f"wss://stream.binance.com:9443/ws/{self.symbol}@trade", 
                         self.collector.on_trade_message, "Trades"))
            ]
            
            for t in threads:
                t.daemon = True
                t.start()
            
            # Monitor threads and report status
            while not stop_event.is_set():
                alive_threads = sum(1 for t in threads if t.is_alive())
                if alive_threads < 2:
                    print(f"Warning: Only {alive_threads}/2 threads alive, attempting to restart")
                    # Could add thread restart logic here
                
                time.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            print(f"Collector error: {str(e)[:200]}")
        finally:
            self.collector.running = False