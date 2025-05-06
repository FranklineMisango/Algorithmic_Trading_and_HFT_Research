import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import queue

from data_fetcher import MarketDataCollector
from kyle_model import KyleModel
from dashboard import KyleDashboard

class KyleModelLiveTrading:
    def __init__(self, symbol='btcusdt', model_update_interval=1.0, dashboard_port=8050):
        """
        Initialize the Kyle model live trading system.
        
        Args:
            symbol (str): Trading symbol
            model_update_interval (float): Interval between model updates in seconds
            dashboard_port (int): Port for the dashboard
        """
        self.symbol = symbol.lower()
        self.model_update_interval = model_update_interval
        self.dashboard_port = dashboard_port
        
        # Initialize data structures
        self.market_data = pd.DataFrame()
        self.data_queue = queue.Queue()
        self.running = True
        
        # Estimate initial model parameters based on typical crypto market behavior
        # These would normally be calibrated from historical data
        initial_price = 0.0  # Will be updated when first data arrives
        uncertainty_sigma0 = 0.0001  # Initial uncertainty about asset value
        noise_sigma_u = 0.005  # Standard deviation of noise trading
        
        # Create components
        self.data_collector = MarketDataCollector(data_path=f'crypto_data_{self.symbol}')
        self.kyle_model = KyleModel(initial_price, uncertainty_sigma0, noise_sigma_u)
        self.dashboard = KyleDashboard(update_interval=1000)  # 1000ms update interval
        
        # Initialize threads
        self.data_thread = None
        self.model_thread = None
        self.dashboard_thread = None
    
    def start(self):
        """Start the Kyle model live trading system."""
        print(f"Starting Kyle model live trading for {self.symbol}...")
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self._collect_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start model update thread
        self.model_thread = threading.Thread(target=self._update_model)
        self.model_thread.daemon = True
        self.model_thread.start()
        
        # Start dashboard
        self.dashboard_thread = threading.Thread(target=self._run_dashboard)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down Kyle model live trading...")
            self.running = False
    
    def _collect_data(self):
        """Collect market data using the data fetcher."""
        try:
            print(f"Starting data collection for {self.symbol}...")
            
            def order_book_handler(ws, message):
                self.data_collector.on_order_book_message(ws, message)
                self._process_collected_data()
            
            def trade_handler(ws, message):
                self.data_collector.on_trade_message(ws, message)
                self._process_collected_data()
            
            # Start the websocket connections
            threads = [
                threading.Thread(target=self.data_collector._run_websocket,
                    args=(f"wss://stream.binance.com:9443/ws/{self.symbol}@depth@100ms", 
                        order_book_handler, "OrderBook")),
                threading.Thread(target=self.data_collector._run_websocket,
                    args=(f"wss://stream.binance.com:9443/ws/{self.symbol}@trade", 
                        trade_handler, "Trades"))
            ]
            
            for t in threads:
                t.daemon = True
                t.start()
            
            # Keep the thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            print(f"Data collection error: {str(e)}")
            traceback.print_exc()
    
    def _process_collected_data(self):
        """Process collected data and put it in the queue."""
        try:
            if len(self.data_collector.combined_data) > 0:
                # Process the data
                df = self.data_collector.process_data()
                if not df.empty:
                    self.data_queue.put(df)
        except Exception as e:
            print(f"Data processing error: {str(e)}")
            traceback.print_exc()
    
    def _update_model(self):
        """Update the Kyle model with new market data."""
        try:
            print("Starting model updates...")
            
            last_update = time.time()
            startup_phase = True
            min_data_points = 10  # Minimum data points before starting model updates
            
            while self.running:
                current_time = time.time()
                
                # Check if it's time for an update
                if current_time - last_update >= self.model_update_interval:
                    # Process all available data
                    new_data = pd.DataFrame()
                    while not self.data_queue.empty():
                        try:
                            df = self.data_queue.get_nowait()
                            if new_data.empty:
                                new_data = df
                            else:
                                new_data = pd.concat([new_data, df]).drop_duplicates(subset=['timestamp'])
                        except queue.Empty:
                            break
                    
                    if not new_data.empty:
                        # Update the market data
                        if self.market_data.empty:
                            self.market_data = new_data
                            
                            # Initialize the model with the first price
                            initial_price = new_data['mid_price'].iloc[-1]
                            self.kyle_model.p0 = initial_price
                            self.kyle_model.current_price = initial_price
                            print(f"Model initialized with price: {initial_price}")
                        else:
                            self.market_data = pd.concat([self.market_data, new_data])
                            self.market_data = self.market_data.drop_duplicates(subset=['timestamp'])
                            
                            # Keep only recent data (last 1000 rows)
                            if len(self.market_data) > 1000:
                                self.market_data = self.market_data.tail(1000)
                        
                        # Only start model updates once we have enough data
                        if startup_phase and len(self.market_data) >= min_data_points:
                            startup_phase = False
                            print(f"Startup phase complete with {len(self.market_data)} data points")
                        
                        if not startup_phase:
                            # Update the model
                            model_metrics = self.kyle_model.update(self.market_data)
                            model_history = self.kyle_model.get_history()
                            
                            # Update the dashboard
                            self.dashboard.update_data(
                                market_data=self.market_data,
                                model_metrics=model_metrics,
                                model_history=model_history
                            )
                    
                    last_update = current_time
                
                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Model update error: {str(e)}")
            traceback.print_exc()
    
    def _run_dashboard(self):
        """Run the dashboard."""
        try:
            print(f"Starting dashboard on port {self.dashboard_port}...")
            self.dashboard.run_server(debug=False, port=self.dashboard_port)
        except Exception as e:
            print(f"Dashboard error: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    symbol = "suiusdt"
    kyle_system = KyleModelLiveTrading(
        symbol=symbol,
        model_update_interval=1.0,  # Update model every second
        dashboard_port=8050  # Dashboard will be available at http://localhost:8050
    )
    
    try:
        kyle_system.start()
    except KeyboardInterrupt:
        print("\nKyle model live trading stopped by user.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()