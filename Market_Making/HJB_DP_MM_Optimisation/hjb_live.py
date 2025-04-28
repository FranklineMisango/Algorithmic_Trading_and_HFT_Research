# --------------------------
# 1. Enhanced Imports
# --------------------------
import numpy as np
import pandas as pd
import cupy as cp
from numba import cuda
import websocket
import threading
import time
from datetime import datetime
import plotly.graph_objects as go
from tabulate import tabulate
from queue import Queue
import ccxt
import json


# Complete implementation of the HJB kernel
@cuda.jit
def hjb_kernel(d_V, d_V_next, d_S, d_I, dt, ds, di, params):
    i, j = cuda.grid(2)
    if 1 <= i < d_V.shape[0]-1 and 1 <= j < d_V.shape[1]-1:
        # Extract current state
        S = d_S[i]
        I = d_I[j]
        
        # Extract params
        sigma = params[0]
        kappa = params[1]
        gamma = params[2]
        rho = params[3]
        market_impact = params[4]
        best_bid = params[5]
        best_ask = params[6]
        
        # Calculate derivatives
        V_S_plus = d_V_next[i+1, j] 
        V_S_minus = d_V_next[i-1, j]
        V_I_plus = d_V_next[i, j+1]
        V_I_minus = d_V_next[i, j-1]
        
        V_S = (V_S_plus - V_S_minus) / (2 * ds)
        V_SS = (V_S_plus - 2 * d_V_next[i, j] + V_S_minus) / (ds**2)
        V_I = (V_I_plus - V_I_minus) / (2 * di)
        
        # Optimization over control space (bid/ask spread)
        V_optimal = -1e10  # Negative infinity
        
        # Discretized control space for bid/ask adjustments
        for bid_idx in range(5):  # -2*ds to +2*ds
            bid_change = (bid_idx - 2) * ds
            
            for ask_idx in range(5):  # -2*ds to +2*ds
                ask_change = (ask_idx - 2) * ds
                
                new_bid = best_bid + bid_change
                new_ask = best_ask + ask_change
                
                # Valid spread check
                if new_bid > 0 and new_ask > 0 and new_bid < new_ask:
                    # Order execution intensity model (simplified)
                    buy_intensity = cuda.max(0.0, dt * (1.0 - (new_bid / best_bid - 1.0) / market_impact))
                    sell_intensity = cuda.max(0.0, dt * (1.0 - (new_ask / best_ask - 1.0) / market_impact))
                    
                    # Expected P&L from trades
                    expected_pnl = new_bid * sell_intensity - new_ask * buy_intensity
                    
                    # Inventory risk penalty
                    inventory_cost = kappa * I * I * dt
                    
                    # Diffusion term from price process
                    diffusion = 0.5 * sigma * sigma * S * S * V_SS * dt
                    
                    # Candidate value
                    V_candidate = d_V_next[i, j] + expected_pnl - inventory_cost + diffusion
                    
                    # Update if better
                    if V_candidate > V_optimal:
                        V_optimal = V_candidate
        
        # Update value function
        d_V[i, j] = V_optimal


class HJBSolver:
    """Hamilton-Jacobi-Bellman equation solver for market making"""
    
    def __init__(self, S_min, S_max, I_min, I_max, N_S=101, N_I=101, 
                 sigma=0.2, kappa=0.001, gamma=0.0001, rho=0.01, market_impact=0.0001):
        # Grid parameters
        self.S_grid = np.linspace(S_min, S_max, N_S)
        self.I_grid = np.linspace(I_min, I_max, N_I)
        self.ds = (S_max - S_min) / (N_S - 1)
        self.di = (I_max - I_min) / (N_I - 1)
        
        # Model parameters
        self.params = np.array([sigma, kappa, gamma, rho, market_impact, 0.0, 0.0], dtype=np.float32)
        
        # Initialize value function
        self.V = np.zeros((N_S, N_I))
        self.V_next = np.zeros((N_S, N_I))
        
        # GPU memory allocation
        self.d_S = cuda.to_device(self.S_grid)
        self.d_I = cuda.to_device(self.I_grid)
        self.d_V = cuda.to_device(self.V)
        self.d_V_next = cuda.to_device(self.V_next)
        self.d_params = cuda.to_device(self.params)
        
        # CUDA grid configuration
        self.threadsperblock = (16, 16)
        blockspergrid_x = (N_S + self.threadsperblock[0] - 1) // self.threadsperblock[0]
        blockspergrid_y = (N_I + self.threadsperblock[1] - 1) // self.threadsperblock[1]
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)
        
    def update(self, bid_price, ask_price, dt=0.001):
        """Update value function for one time step"""
        # Update market parameters
        self.params[5] = bid_price
        self.params[6] = ask_price
        self.d_params = cuda.to_device(self.params)
        
        # Run HJB kernel
        hjb_kernel[self.blockspergrid, self.threadsperblock](
            self.d_V, self.d_V_next, self.d_S, self.d_I, 
            dt, self.ds, self.di, self.d_params
        )
        
        # Swap buffers
        self.d_V, self.d_V_next = self.d_V_next, self.d_V
        
        # Copy back results occasionally (not every step for performance)
        cuda.synchronize()
        self.d_V.copy_to_host(self.V)
        
    def get_optimal_quotes(self, current_price, inventory):
        """Get optimal bid/ask quotes for current state"""
        # Find closest grid points
        s_idx = np.argmin(np.abs(self.S_grid - current_price))
        i_idx = np.argmin(np.abs(self.I_grid - inventory))
        
        # Search for optimal spreads around current state
        optimal_bid_change = 0
        optimal_ask_change = 0
        max_value = -float('inf')
        
        for bid_change in np.arange(-2*self.ds, 2*self.ds + self.ds, self.ds):
            for ask_change in np.arange(-2*self.ds, 2*self.ds + self.ds, self.ds):
                # Lookup neighbor value in grid
                s_offset = int(round(bid_change / self.ds))
                i_offset = int(round(ask_change / self.ds))
                
                if 0 <= s_idx + s_offset < len(self.S_grid) and 0 <= i_idx + i_offset < len(self.I_grid):
                    value = self.V[s_idx + s_offset, i_idx + i_offset]
                    
                    if value > max_value:
                        max_value = value
                        optimal_bid_change = bid_change
                        optimal_ask_change = ask_change
        
        # Apply to current market prices
        optimal_bid = self.params[5] + optimal_bid_change
        optimal_ask = self.params[6] + optimal_ask_change
        
        return optimal_bid, optimal_ask


class DataEngine:
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.data_queue = Queue(maxsize=1000)
        self.latest_data = {
            'bid': None,
            'ask': None,
            'trade': None,
            'volume': None,
            'timestamp': None
        }
        self._running = True
        
        # Start data collection thread
        self.thread = threading.Thread(target=self._ws_thread)
        self.thread.daemon = True
        self.thread.start()
        
    def _ws_thread(self):
        def on_message(ws, message):
            print(f"Received message: {message[:100]}...")  # Print first 100 chars of message
            try:
                data = json.loads(message)
                ts = datetime.now().timestamp()
                
                # Orderbook update (depth stream)
                if isinstance(data, dict) and 'bids' in data and 'asks' in data and len(data['bids']) > 0 and len(data['asks']) > 0:
                    update = {
                        'type': 'book',
                        'bid': float(data['bids'][0][0]),
                        'ask': float(data['asks'][0][0]),
                        'timestamp': ts
                    }
                    print(f"Processed orderbook update: bid={update['bid']}, ask={update['ask']}")
                    self.data_queue.put(update)
                    self.latest_data['bid'] = update['bid']
                    self.latest_data['ask'] = update['ask']
                    self.latest_data['timestamp'] = ts
                
                # Kline/Candlestick update
                elif isinstance(data, dict) and 'e' in data and data['e'] == 'kline':
                    kline = data['k']
                    update = {
                        'type': 'trade',
                        'price': float(kline['c']),  # Close price
                        'volume': float(kline['v']),  # Volume
                        'timestamp': ts
                    }
                    print(f"Processed kline update: price={update['price']}, volume={update['volume']}")
                    self.data_queue.put(update)
                    self.latest_data['trade'] = update['price']
                    self.latest_data['volume'] = update['volume']
                    self.latest_data['timestamp'] = ts
                
                # For any other message type or format, log for debugging
                else:
                    print(f"Unhandled message format: {data}")
                
            except Exception as e:
                print(f"Error processing message: {e}")
                
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket connection closed: {close_status_code}, {close_msg}")
            if self._running:
                print("Attempting reconnection...")
                time.sleep(1)
                self._connect()
                
        def on_open(ws):
            print(f"WebSocket connection established for {self.symbol}")
            # Subscribe to both order book and kline streams
            # For combining multiple streams, Binance now recommends using a combined stream
            subscription = {
                "method": "SUBSCRIBE",
                "params": [
                    f"{self.symbol}@depth10@100ms",
                    f"{self.symbol}@kline_1m"
                ],
                "id": 1
            }
            ws.send(json.dumps(subscription))
            print(f"Subscribed to streams for {self.symbol}")
            
        def _connect():
            # Use the combined streams endpoint
            ws_url = "wss://stream.binance.com:9443/ws"
            print(f"Connecting to {ws_url}")
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws.run_forever()
        
        while self._running:
            try:
                _connect()
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                time.sleep(1)

class TradingDashboard:
    def __init__(self):
        # Create figure with subplots
        self.fig = go.Figure()
        self.fig.update_layout(
            title="Real-time Market Making Dashboard",
            height=800,
            width=1200,
            showlegend=True,
            grid={'rows': 2, 'columns': 2, 'pattern': 'independent'},
        )
        
        # Initialize traces
        self.price_trace = go.Scatter(x=[], y=[], name='Mid Price', line=dict(color='white'))
        self.bid_trace = go.Scatter(x=[], y=[], name='Bid Quote', line=dict(color='green'))
        self.ask_trace = go.Scatter(x=[], y=[], name='Ask Quote', line=dict(color='red'))
        self.inventory_trace = go.Scatter(x=[], y=[], name='Inventory', line=dict(color='yellow'))
        self.pnl_trace = go.Scatter(x=[], y=[], name='Cumulative P&L', line=dict(color='purple'))
        
        # Add traces to figure
        self.fig.add_trace(self.price_trace)
        self.fig.add_trace(self.bid_trace)
        self.fig.add_trace(self.ask_trace)
        
        # Containers for historical data
        self.timestamps = []
        self.mid_prices = []
        self.bid_prices = []
        self.ask_prices = []
        self.inventory_history = []
        self.pnl_history = []
        
        # Show figure
        self.fig.show()
        
    def update(self, strategy_state):
        # Get current timestamp
        now = datetime.now()
        
        # Update data containers
        self.timestamps.append(now)
        self.mid_prices.append((strategy_state['bid'] + strategy_state['ask'])/2)
        self.bid_prices.append(strategy_state['bid'])
        self.ask_prices.append(strategy_state['ask'])
        self.inventory_history.append(strategy_state['inventory'])
        self.pnl_history.append(strategy_state.get('pnl', 0))
        
        # Update plot traces
        with self.fig.batch_update():
            self.price_trace.x = self.timestamps
            self.price_trace.y = self.mid_prices
            
            self.bid_trace.x = self.timestamps
            self.bid_trace.y = self.bid_prices
            
            self.ask_trace.x = self.timestamps
            self.ask_trace.y = self.ask_prices
        
        # Print current status
        print(tabulate(
            [[f"{now}", f"{strategy_state['bid']:.2f}", f"{strategy_state['ask']:.2f}", 
              f"{strategy_state['inventory']:.2f}", f"{strategy_state.get('pnl', 0):.2f}"]],
            headers=['Time', 'Bid', 'Ask', 'Position', 'P&L']
        ))

def main(symbol='btcusdt'):
    print("Initializing HJB Market Making Strategy...")
    
    # Initialize components
    data_engine = DataEngine(symbol)
    
    # Wait for initial data with timeout
    wait_start = time.time()
    timeout = 30  # seconds
    while (data_engine.latest_data['bid'] is None or data_engine.latest_data['ask'] is None):
        print("Waiting for market data...")
        time.sleep(1)
        
        # Check if we've exceeded timeout
        if time.time() - wait_start > timeout:
            print(f"Timeout waiting for market data after {timeout} seconds.")
            print("Current data state:", data_engine.latest_data)
            print("Make sure your internet connection is working and Binance API is accessible.")
            return
    

    # Initialize solver with current price range
    current_price = (data_engine.latest_data['bid'] + data_engine.latest_data['ask']) / 2
    S_min = current_price * 0.9
    S_max = current_price * 1.1
    I_min = -100  # Max short position
    I_max = 100   # Max long position
    
    solver = HJBSolver(S_min, S_max, I_min, I_max)
    dashboard = TradingDashboard()
    
    # Trading state
    current_inventory = 0
    cumulative_pnl = 0
    filled_orders = []
    
    print(f"Starting market making with {symbol}...")
    
    while True:
        try:
            # Process data queue
            if not data_engine.data_queue.empty():
                update = data_engine.data_queue.get()
                
                # Only process book updates
                if update['type'] == 'book':
                    # Update HJB model
                    mid_price = (update['bid'] + update['ask']) / 2
                    solver.update(update['bid'], update['ask'], dt=0.001)
                    
                    # Get optimal quotes
                    optimal_bid, optimal_ask = solver.get_optimal_quotes(mid_price, current_inventory)
                    
                    # Update dashboard
                    dashboard.update({
                        'bid': optimal_bid,
                        'ask': optimal_ask,
                        'inventory': current_inventory,
                        'pnl': cumulative_pnl
                    })
            
            time.sleep(0.001)  # 1ms latency
            
        except KeyboardInterrupt:
            print("\nStopping market making strategy...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
