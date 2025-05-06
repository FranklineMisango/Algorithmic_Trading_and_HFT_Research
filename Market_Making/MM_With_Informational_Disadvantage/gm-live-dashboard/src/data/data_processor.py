import pandas as pd
import numpy as np
import time
import os
import glob
from datetime import datetime, timedelta

class DataProcessor:
    """
    Processes market data from real or simulated sources.
    """
    
    def __init__(self, data_path, symbol):
        """
        Initialize data processor.
        
        Args:
            data_path (str): Path to data directory
            symbol (str): Trading symbol to process
        """
        self.data_path = data_path
        self.symbol = symbol
        self.data_files = []
        self.current_data = None
        self.live_mode = False
        self.buffer_size = 200  # Number of data points to keep in memory
        
        # Initialize with any existing data
        self._load_data_files()
    
    def _load_data_files(self):
        """Load available data files from the data directory"""
        if os.path.exists(self.data_path):
            # Look for files matching the symbol pattern
            pattern = os.path.join(self.data_path, f"*{self.symbol}*.csv")
            self.data_files = sorted(glob.glob(pattern))
            print(f"Found {len(self.data_files)} data files for {self.symbol}")
        else:
            print(f"Data directory {self.data_path} does not exist")
    
    def _load_file_data(self, file_path):
        """Load data from a single file"""
        try:
            data = pd.read_csv(file_path)
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def get_batch_data(self, batch_size=100):
        """
        Get a batch of historical data for replay mode
        
        Args:
            batch_size (int): Number of data points to return
            
        Returns:
            pd.DataFrame: Batch of market data
        """
        if not self.data_files:
            return pd.DataFrame()
        
        # If no current data, load the first file
        if self.current_data is None or self.current_data.empty:
            self.current_data = self._load_file_data(self.data_files[0])
            self.current_index = 0
        
        # Return a batch of data
        if self.current_index + batch_size <= len(self.current_data):
            batch = self.current_data.iloc[self.current_index:self.current_index+batch_size]
            self.current_index += batch_size
            return batch
        else:
            # Need to load next file or restart
            remaining = batch_size - (len(self.current_data) - self.current_index)
            batch = self.current_data.iloc[self.current_index:]
            
            # Try to get next file or cycle back
            current_file_idx = self.data_files.index(self.current_file) if hasattr(self, 'current_file') else 0
            next_file_idx = (current_file_idx + 1) % len(self.data_files)
            self.current_file = self.data_files[next_file_idx]
            self.current_data = self._load_file_data(self.current_file)
            self.current_index = 0
            
            # Get remaining data from new file
            if not self.current_data.empty and remaining > 0:
                batch = pd.concat([batch, self.current_data.iloc[:remaining]])
                self.current_index = remaining
            
            return batch
    
    def process_live_data(self, data_queue, stop_event, update_frequency=1):
        """
        Process data in a continuous loop, putting results in the queue
        
        Args:
            data_queue (Queue): Queue to put processed data in
            stop_event (Event): Event to signal thread to stop
            update_frequency (float): Update frequency in seconds
        """
        print(f"Starting data processing thread with {update_frequency}s update frequency")
        
        # Check if we have real data files
        if self.data_files:
            print(f"Using {len(self.data_files)} data files for replay")
            self.current_file = self.data_files[0]
            self.current_data = self._load_file_data(self.current_file)
            self.current_index = 0
            
            # Process in chunks for real data
            while not stop_event.is_set():
                batch = self.get_batch_data(50)  # Get 50 data points at a time
                if not batch.empty:
                    data_queue.put(batch)
                    time.sleep(update_frequency)  # Pace the updates
                else:
                    time.sleep(0.1)  # Short sleep to prevent CPU spin
        else:
            # Generate synthetic data if no real data is available
            print("No data files found, generating synthetic data")
            start_time = datetime.now()
            base_price = 100.0
            
            while not stop_event.is_set():
                # Create synthetic data
                now = datetime.now()
                timestamps = [start_time + timedelta(seconds=i) for i in range((now - start_time).seconds)][-self.buffer_size:]
                
                # Simulate price movement
                num_points = len(timestamps)
                if num_points == 0:
                    time.sleep(0.1)
                    continue
                
                # Simple random walk for prices
                noise = np.random.normal(0, 0.01, num_points)
                trend = np.cumsum(noise)
                
                mid_price = base_price + trend
                spread = np.abs(np.random.normal(0.05, 0.02, num_points))
                
                data = pd.DataFrame({
                    'timestamp': timestamps,
                    'bid_price': mid_price - spread/2,
                    'ask_price': mid_price + spread/2,
                    'mid_price': mid_price,
                    'trade_price': [p + np.random.normal(0, 0.02) if np.random.random() > 0.7 else None for p in mid_price],
                    'volume': np.random.exponential(100, num_points)
                })
                
                # Update the base price for next iteration
                base_price = float(mid_price[-1])
                
                # Put data in queue
                data_queue.put(data)
                
                # Sleep until next update
                time.sleep(update_frequency)