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
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.subplots as sp
from collections import deque
import threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random

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
        
        V_S_plus = d_V_next[i+1, j] 
        V_S_minus = d_V_next[i-1, j]
        V_I_plus = d_V_next[i, j+1]
        V_I_minus = d_V_next[i, j-1]
        
        V_S = (V_S_plus - V_S_minus) / (2 * ds)
        V_SS = (V_S_plus - 2 * d_V_next[i, j] + V_S_minus) / (ds**2)
        V_I = (V_I_plus - V_I_minus) / (2 * di)
        
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
        # Track connection status
        self.is_connected = False
        self.last_heartbeat = time.time()
        
        # Initialize websocket client with heartbeat checking
        websocket.enableTrace(True)  # Enable debugging
        
        # Start data collection thread
        self.thread = threading.Thread(target=self._ws_thread)
        self.thread.daemon = True
        self.thread.start()
        
        # Start heartbeat monitor thread
        self.heartbeat_thread = threading.Thread(target=self._check_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
    def _check_heartbeat(self):
        """Monitor WebSocket connection health"""
        while self._running:
            if time.time() - self.last_heartbeat > 10:  # No heartbeat in 10 seconds
                print("WebSocket connection appears to be down. Reconnecting...")
                self.is_connected = False
            time.sleep(5)  # Check every 5 seconds
        
    def _ws_thread(self):
        def on_message(ws, message):
            self.last_heartbeat = time.time()
            self.is_connected = True
            
            try:
                data = json.loads(message)
                ts = datetime.now().timestamp()
                
                # Process ping/pong messages
                if isinstance(data, dict) and 'ping' in data:
                    ws.send(json.dumps({"pong": data["ping"]}))
                    return
                    
                # Orderbook update (depth stream)
                if isinstance(data, dict) and 'bids' in data and 'asks' in data and len(data['bids']) > 0 and len(data['asks']) > 0:
                    update = {
                        'type': 'book',
                        'bid': float(data['bids'][0][0]),
                        'bid_size': float(data['bids'][0][1]),
                        'ask': float(data['asks'][0][0]),
                        'ask_size': float(data['asks'][0][1]),
                        'timestamp': ts
                    }
                    print(f"Orderbook: Bid: {update['bid']:.4f} ({update['bid_size']:.4f}) | Ask: {update['ask']:.4f} ({update['ask_size']:.4f})")
                    self.data_queue.put(update)
                    self.latest_data.update({k: v for k, v in update.items() if k in self.latest_data})
                
                # Kline/Candlestick update
                elif isinstance(data, dict) and 'e' in data and data['e'] == 'kline':
                    kline = data['k']
                    update = {
                        'type': 'trade',
                        'price': float(kline['c']),  # Close price
                        'volume': float(kline['v']),  # Volume 
                        'timestamp': ts
                    }
                    print(f"Kline: Price: {update['price']:.4f} | Vol: {update['volume']:.4f}")
                    self.data_queue.put(update)
                    self.latest_data.update({k: v for k, v in update.items() if k in self.latest_data})
                
                # For any other message type or format, log for debugging
                else:
                    print(f"Received: {json.dumps(data)[:100]}...")
                
            except Exception as e:
                print(f"Error processing message: {e}")
                
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            self.is_connected = False
            
        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket connection closed: {close_status_code}, {close_msg}")
            self.is_connected = False
            if self._running:
                print("Attempting reconnection...")
                time.sleep(1)
                _connect()
                
        def on_open(ws):
            print(f"WebSocket connection established for {self.symbol}")
            self.is_connected = True
            
            # Subscribe to both order book and kline streams
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
            ws.run_forever(ping_interval=30)  # Send ping every 30 seconds
        
        while self._running:
            try:
                _connect()
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                self.is_connected = False
                time.sleep(1)
class TradingDashboard:
    def __init__(self, update_interval=1000):
        self.update_interval = update_interval  # in milliseconds
        
        # Use deques with maxlen for efficient data storage
        self.max_points = 100  # Maximum number of points to display
        self.timestamps = deque(maxlen=self.max_points)
        self.mid_prices = deque(maxlen=self.max_points)
        self.bid_prices = deque(maxlen=self.max_points)
        self.ask_prices = deque(maxlen=self.max_points)
        self.inventory_history = deque(maxlen=self.max_points)
        self.pnl_history = deque(maxlen=self.max_points)
        
        # Current strategy state
        self.strategy_state = {
            'bid': 0,
            'ask': 0,
            'inventory': 0,
            'pnl': 0,
            'symbol': ''
        }
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        
        # Create app layout with multiple graphs
        self.app.layout = html.Div([
            html.H1("Real-time Market Making Dashboard"),
            
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
            
            html.Div([
                html.Div([
                    html.H3(id='symbol-header', children="Symbol: -"),
                    html.H3(id='status-header', children="Status: Running"),
                ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                
                html.Div([
                    dcc.Graph(id='price-chart', style={'height': '400px'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    dcc.Graph(id='position-chart', style={'height': '300px', 'width': '49%', 'display': 'inline-block'}),
                    dcc.Graph(id='pnl-chart', style={'height': '300px', 'width': '49%', 'display': 'inline-block'}),
                ]),
                
                html.Div(id='current-stats', style={'marginTop': '20px'})
            ])
        ])
        
        # Define callbacks
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('position-chart', 'figure'),
             Output('pnl-chart', 'figure'),
             Output('current-stats', 'children'),
             Output('symbol-header', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            # Time series for x-axis (convert to readable format)
            x_data = [t.strftime('%H:%M:%S') for t in self.timestamps] if self.timestamps else []
            
            # Price chart with bid/ask quotes
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=x_data, y=list(self.mid_prices), name='Mid Price', line=dict(color='white')))
            price_fig.add_trace(go.Scatter(x=x_data, y=list(self.bid_prices), name='Bid Quote', line=dict(color='green')))
            price_fig.add_trace(go.Scatter(x=x_data, y=list(self.ask_prices), name='Ask Quote', line=dict(color='red')))
            
            price_fig.update_layout(
                title="Price and Quotes",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode="x unified"
            )
            
            # Position/Inventory chart
            position_fig = go.Figure()
            position_fig.add_trace(go.Scatter(x=x_data, y=list(self.inventory_history), 
                                              name='Inventory', fill='tozeroy', line=dict(color='yellow')))
            position_fig.update_layout(
                title="Position/Inventory",
                xaxis_title="Time",
                yaxis_title="Units",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # PnL chart
            pnl_fig = go.Figure()
            pnl_fig.add_trace(go.Scatter(x=x_data, y=list(self.pnl_history), 
                                         name='Cumulative P&L', line=dict(color='purple')))
            pnl_fig.update_layout(
                title="Cumulative P&L",
                xaxis_title="Time",
                yaxis_title="P&L",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # Current stats table
            if self.timestamps:
                stats = html.Table([
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody([
                        html.Tr([html.Td("Current Bid"), html.Td(f"{self.strategy_state['bid']:.4f}")]),
                        html.Tr([html.Td("Current Ask"), html.Td(f"{self.strategy_state['ask']:.4f}")]),
                        html.Tr([html.Td("Spread"), html.Td(f"{(self.strategy_state['ask'] - self.strategy_state['bid']):.4f}")]),
                        html.Tr([html.Td("Current Position"), html.Td(f"{self.strategy_state['inventory']}")]),
                        html.Tr([html.Td("Current P&L"), html.Td(f"{self.strategy_state['pnl']:.4f}")]),
                    ])
                ], style={'width': '100%', 'border': '1px solid white'})
            else:
                stats = html.Div("Waiting for data...")
            
            symbol_header = f"Symbol: {self.strategy_state['symbol'].upper()}"
            
            return price_fig, position_fig, pnl_fig, stats, symbol_header
    
    def update(self, strategy_state):
        """Update dashboard with new data"""
        now = datetime.now()
        
        # Update strategy state
        self.strategy_state = strategy_state
        
        # Update data containers
        self.timestamps.append(now)
        self.mid_prices.append((strategy_state['bid'] + strategy_state['ask'])/2)
        self.bid_prices.append(strategy_state['bid'])
        self.ask_prices.append(strategy_state['ask'])
        self.inventory_history.append(strategy_state['inventory'])
        self.pnl_history.append(strategy_state.get('pnl', 0))
        
        # Print current status to console
        print(tabulate(
            [[f"{now}", f"{strategy_state['bid']:.4f}", f"{strategy_state['ask']:.4f}", 
              f"{strategy_state['inventory']:.2f}", f"{strategy_state.get('pnl', 0):.4f}"]],
            headers=['Time', 'Bid', 'Ask', 'Position', 'P&L']
        ))
    
    def run(self, port=8050):
        """Run dashboard server"""
        self.app.run(debug=False, port=port)
        
    def start(self):
        """Start dashboard in a separate thread"""
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        print(f"Dashboard started at http://localhost:8050")
# ...existing code...

class CryptoHeatScanner:
    def __init__(self, top_n=5, update_interval=60):
        """
        Scanner to identify "hot" cryptocurrencies
        
        Parameters:
        - top_n: Number of top cryptocurrencies to track
        - update_interval: How often to update rankings (seconds)
        """
        self.top_n = top_n
        self.update_interval = update_interval
        self.exchange = ccxt.binance()
        self.rankings = []
        self.running = True
        
        # Start scanner thread
        self.thread = threading.Thread(target=self._scanner_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def _scanner_thread(self):
        """Background thread to update cryptocurrency rankings"""
        while self.running:
            try:
                # Fetch tickers for all symbols
                tickers = self.exchange.fetch_tickers()
                
                # Calculate metrics for each symbol
                metrics = []
                for symbol, ticker in tickers.items():
                    # Only include USDT pairs for simplicity
                    if symbol.endswith('/USDT'):
                        # Skip symbols with missing data
                        if not all(k in ticker for k in ['quoteVolume', 'percentage']):
                            continue
                            
                        # Extract key metrics
                        volume_24h = ticker.get('quoteVolume', 0)
                        price_change_24h = abs(ticker.get('percentage', 0) or 0)  # Use absolute value for volatility
                        
                        # Skip entries with invalid data
                        if not isinstance(volume_24h, (int, float)) or not isinstance(price_change_24h, (int, float)):
                            continue
                            
                        # Skip entries with zero volume
                        if volume_24h <= 0:
                            continue
                            
                        # Calculate heat score (higher is better)
                        # Formula: volume × volatility (can be adjusted)
                        heat_score = volume_24h * price_change_24h
                        
                        metrics.append({
                            'symbol': symbol.replace('/USDT', '').lower() + 'usdt',  # Format for Binance WebSocket
                            'display_symbol': symbol,
                            'volume_24h': volume_24h,
                            'price_change_24h': price_change_24h,
                            'heat_score': heat_score
                        })
                
                # Sort by heat score
                metrics.sort(key=lambda x: x['heat_score'], reverse=True)
                
                # Select top N symbols
                self.rankings = metrics[:self.top_n]
                
                print("\n--- Top Hot Cryptocurrencies ---")
                for i, crypto in enumerate(self.rankings):
                    print(f"{i+1}. {crypto['display_symbol']}: Volume=${crypto['volume_24h']:,.0f}, Change: {crypto['price_change_24h']}%")
                print("-------------------------------\n")
                
                # Wait before next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error updating crypto heat map: {e}")
                time.sleep(self.update_interval)
    
    def get_top_symbol(self):
        """Get the hottest cryptocurrency symbol"""
        if self.rankings:
            return self.rankings[0]['symbol']
        return None
        
    def get_rankings(self):
        """Get current rankings of hot cryptocurrencies"""
        return self.rankings

def main():
    print("Initializing HJB Market Making Strategy...")
    
    # Initialize hot crypto scanner
    scanner = CryptoHeatScanner(top_n=5, update_interval=60)
    print("Waiting for hot cryptocurrency data...")
    
    # Wait for scanner to collect initial data
    while not scanner.get_rankings():
        time.sleep(1)
    
    # Get the hottest cryptocurrency
    symbol = scanner.get_top_symbol() 
    print(f"Selected {symbol} for market making based on activity")
    
    # Initialize components
    data_engine = DataEngine(symbol)
    
    # Initialize dashboard early (but don't show data yet)
    dashboard = TradingDashboard()
    dashboard.strategy_state['symbol'] = symbol
    dashboard.start()
    
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
    
    # Trading state
    current_inventory = 0
    cumulative_pnl = 0
    filled_orders = []
    last_symbol_check = time.time()
    symbol_check_interval = 300  # Check for new hot symbol every 5 minutes
    
    print(f"Starting market making with {symbol}...")
    
    while True:
        try:
            # Check periodically if a different cryptocurrency is now hotter
            if time.time() - last_symbol_check > symbol_check_interval:
                new_hot_symbol = scanner.get_top_symbol()
                if new_hot_symbol != symbol:
                    print(f"Switching from {symbol} to hotter cryptocurrency {new_hot_symbol}")
                    # Reset trading state when switching symbols
                    symbol = new_hot_symbol
                    data_engine = DataEngine(symbol)
                    dashboard.strategy_state['symbol'] = symbol
                    
                    # Wait for initial data
                    wait_start = time.time()
                    while (data_engine.latest_data['bid'] is None or data_engine.latest_data['ask'] is None):
                        if time.time() - wait_start > timeout:
                            print(f"Timeout waiting for market data for {symbol}")
                            break
                        time.sleep(0.1)
                    
                    if data_engine.latest_data['bid'] is not None:
                        # Reset solver with new price range
                        current_price = (data_engine.latest_data['bid'] + data_engine.latest_data['ask']) / 2
                        solver = HJBSolver(
                            current_price * 0.9, 
                            current_price * 1.1,
                            I_min, I_max
                        )
                
                last_symbol_check = time.time()
            
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
                    
                    # Simulate order executions (simple model)
                    if random.random() < 0.02:  # 2% chance of order execution
                        if random.random() < 0.5:  # Buy order filled
                            current_inventory += 1
                            cumulative_pnl -= optimal_ask  # Spent money to buy
                            print(f"BUY EXECUTED at {optimal_ask}")
                        else:  # Sell order filled
                            current_inventory -= 1
                            cumulative_pnl += optimal_bid  # Received money from sell
                            print(f"SELL EXECUTED at {optimal_bid}")
                    
                    # Update dashboard
                    dashboard.update({
                        'bid': optimal_bid,
                        'ask': optimal_ask,
                        'inventory': current_inventory,
                        'pnl': cumulative_pnl,
                        'symbol': symbol
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
