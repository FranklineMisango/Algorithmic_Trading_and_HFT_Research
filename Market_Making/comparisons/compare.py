import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
import queue
import os
import json
import random
import warnings
from collections import deque
import websocket
import asyncio
import ccxt

# Import existing strategy classes - these will be used directly or via adapters
# Import from HJB
from sys import path
path.append('/home/misango/code/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation')
from hjb_live import HJBSolver, DataEngine, PortfolioTracker, ResourceMonitor

# Import from Avellaneda-Stoikov
path.append('/home/misango/code/Algorithmic_Trading_and_HFT_Research/Market_Making/Avellaneda-Stoikov')
from Avellaneda_stoikov_simulation import AvellanedaStoikovMM, OrderBook

# Import from Grossman-Miller
path.append('/home/misango/code/Algorithmic_Trading_and_HFT_Research/Market_Making/Grossman-Miller-Model')
from The_Grossman_MIller_Market_Making_Model import GrossmanMillerModel


class UnifiedMarketData:
    """Provides identical market data to all strategies"""
    
    def __init__(self, symbol, from_file=None, data_queue_size=1000):
        self.symbol = symbol.lower()
        self.from_file = from_file
        self.data_queue = queue.Queue(maxsize=data_queue_size)
        self.latest_data = {
            'bid': None,
            'ask': None,
            'trade': None,
            'volume': None,
            'timestamp': None
        }
        self._running = True
        self.is_connected = False
        self.last_heartbeat = time.time()
        
        # If loading from file, start that process instead of websocket
        if from_file and os.path.exists(from_file):
            self.thread = threading.Thread(target=self._load_from_file)
            self.thread.daemon = True
            self.thread.start()
        else:
            # Start websocket connection for live data
            self.thread = threading.Thread(target=self._ws_thread)
            self.thread.daemon = True
            self.thread.start()
            
            # Heartbeat monitoring
            self.heartbeat_thread = threading.Thread(target=self._check_heartbeat)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
    
    def _load_from_file(self):
        """Load market data from a CSV file"""
        print(f"Loading market data from {self.from_file}")
        try:
            df = pd.read_csv(self.from_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate timespan to set playback speed
            if isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                timespan = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
                record_count = len(df)
                replay_speed = timespan / record_count / 0.1  # Aim for 10 records per second
            else:
                replay_speed = 0.1  # Default 100ms if timestamps not available
            
            # Start replay loop
            print(f"Replaying {len(df)} records at {1/replay_speed:.2f} records per second")
            for i, row in df.iterrows():
                if not self._running:
                    break
                    
                # Create update with available data
                update = {
                    'type': 'book' if 'bid_price' in row and 'ask_price' in row else 'trade',
                    'timestamp': row.get('timestamp', datetime.now()),
                    'bid': row.get('bid_price', row.get('bid', None)),
                    'ask': row.get('ask_price', row.get('ask', None)),
                    'trade': row.get('trade_price', row.get('price', None)),
                    'volume': row.get('volume', 0)
                }
                
                # Put in queue and update latest data
                self.data_queue.put(update)
                for key, value in update.items():
                    if key in self.latest_data and value is not None:
                        self.latest_data[key] = value
                
                # Control replay speed
                time.sleep(replay_speed)
                
                # Provide progress update
                if i % 100 == 0:
                    print(f"Replayed {i}/{len(df)} records ({i/len(df)*100:.1f}%)")
            
            print("Replay completed, looping...")
            # Loop back to beginning when done
            self._load_from_file()
                
        except Exception as e:
            print(f"Error loading from file: {e}")
            self._running = False
    
    def _check_heartbeat(self):
        """Monitor connection health"""
        while self._running:
            if time.time() - self.last_heartbeat > 10:  # No heartbeat in 10 seconds
                print("WebSocket connection appears to be down. Reconnecting...")
                self.is_connected = False
            time.sleep(5)  # Check every 5 seconds
    
    def _ws_thread(self):
        """Thread for WebSocket connection and processing"""
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
                    self.data_queue.put(update)
                    self.latest_data.update({k: v for k, v in update.items() if k in self.latest_data})
                
                # Kline/Candlestick update
                elif isinstance(data, dict) and 'e' in data and data['e'] == 'kline':
                    kline = data['k']
                    update = {
                        'type': 'trade',
                        'trade': float(kline['c']),  # Close price
                        'volume': float(kline['v']),  # Volume 
                        'timestamp': ts
                    }
                    self.data_queue.put(update)
                    self.latest_data.update({k: v for k, v in update.items() if k in self.latest_data})
                
                # For any other message type, log for debugging
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
    
    def get_next_update(self, timeout=None):
        """Get the next data update from the queue"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop data collection"""
        self._running = False


class StrategyAdapter:
    """Base class for strategy adapters to ensure consistent interfaces"""
    
    def __init__(self, strategy_name, initial_cash=100000.0, initial_inventory=0.0):
        self.strategy_name = strategy_name
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.inventory = initial_inventory
        self.cash = initial_cash
        
        # Performance tracking
        self.trades = []
        self.current_bid = None
        self.current_ask = None
        self.bid_history = []
        self.ask_history = []
        self.inventory_history = []
        self.cash_history = []
        self.pnl_history = []
        self.timestamp_history = []
        self.mid_price_history = []
        
    def calculate_quotes(self, market_data):
        """Calculate optimal quotes based on market data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def execute_trade(self, size, price, trade_type):
        """Record a trade and update portfolio metrics"""
        trade = {
            'type': trade_type,
            'size': abs(size),
            'price': price,
            'timestamp': datetime.now(),
        }
        self.trades.append(trade)
        
        # Update inventory and cash
        if trade_type == 'BUY':
            self.cash -= price * size
            self.inventory += size
        else:  # SELL
            self.cash += price * size
            self.inventory -= size
        
        return trade
    
    def update_metrics(self, mid_price, timestamp=None):
        """Update performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate P&L as cash + inventory value relative to initial portfolio
        portfolio_value = self.cash + (self.inventory * mid_price)
        initial_value = self.initial_cash + (self.initial_inventory * mid_price)
        pnl = portfolio_value - initial_value
        
        # Store history
        self.timestamp_history.append(timestamp)
        self.inventory_history.append(self.inventory)
        self.cash_history.append(self.cash)
        self.pnl_history.append(pnl)
        self.mid_price_history.append(mid_price)
        if self.current_bid is not None:
            self.bid_history.append(self.current_bid)
        if self.current_ask is not None:
            self.ask_history.append(self.current_ask)
    
    def get_metrics(self):
        """Get current portfolio metrics as a dictionary"""
        if not self.mid_price_history:
            return {
                'strategy': self.strategy_name,
                'cash': self.cash,
                'inventory': self.inventory,
                'pnl': 0.0,
                'portfolio_value': self.cash,
                'current_bid': self.current_bid or 0,
                'current_ask': self.current_ask or 0,
            }
        
        # Use latest mid price for calculations
        mid_price = self.mid_price_history[-1]
        portfolio_value = self.cash + (self.inventory * mid_price)
        initial_value = self.initial_cash + (self.initial_inventory * mid_price)
        pnl = portfolio_value - initial_value
        
        return {
            'strategy': self.strategy_name,
            'cash': self.cash,
            'inventory': self.inventory,
            'pnl': pnl,
            'portfolio_value': portfolio_value,
            'current_bid': self.current_bid or 0,
            'current_ask': self.current_ask or 0,
        }


class HJBStrategyAdapter(StrategyAdapter):
    """Adapter for HJB market making strategy"""
    
    def __init__(self, initial_cash=100000.0, initial_inventory=0.0, params=None):
        super().__init__("HJB", initial_cash, initial_inventory)
        
        # Wait for first data point to initialize price range
        self.initialized = False
        self.params = params or {}
        self.solver = None

    def initialize(self, mid_price):
        """Initialize the HJB solver with appropriate price range"""
        # Calculate price range around current mid price
        S_min = mid_price * 0.9
        S_max = mid_price * 1.1
        I_min = -100  # Max short position
        I_max = 100   # Max long position
        
        # Set grid size based on params or default
        grid_size = self.params.get('grid_size', 51)
        
        # Initialize solver with provided parameters
        self.solver = HJBSolver(
            S_min, S_max, I_min, I_max, 
            N_S=grid_size, N_I=grid_size,
            sigma=self.params.get('sigma', 0.2),
            kappa=self.params.get('kappa', 0.001),
            gamma=self.params.get('gamma', 0.0001),
            rho=self.params.get('rho', 0.01),
            market_impact=self.params.get('market_impact', 0.0001),
            jump_intensity=self.params.get('jump_intensity', 0.1),
            jump_mean=self.params.get('jump_mean', 0.0),
            jump_std=self.params.get('jump_std', 0.01)
        )
        
        self.initialized = True
        print(f"HJB strategy initialized with mid price: {mid_price}")
    
    def calculate_quotes(self, market_data):
        """Calculate optimal quotes using HJB approach"""
        bid = market_data.get('bid')
        ask = market_data.get('ask')
        trade = market_data.get('trade')
        
        if bid is None or ask is None:
            return None, None
        
        mid_price = (bid + ask) / 2
        
        # Initialize solver if needed
        if not self.initialized:
            self.initialize(mid_price)
        
        # Update solver with current market data
        self.solver.update(bid, ask, last_trade=trade, dt=0.001)
        
        # Get optimal quotes
        optimal_bid, optimal_ask = self.solver.get_optimal_quotes(mid_price, self.inventory)
        
        self.current_bid = optimal_bid
        self.current_ask = optimal_ask
        
        # Update metrics
        self.update_metrics(mid_price)
        
        return optimal_bid, optimal_ask


class AvellanedaStoikovAdapter(StrategyAdapter):
    """Adapter for Avellaneda-Stoikov market making strategy"""
    
    def __init__(self, initial_cash=100000.0, initial_inventory=0.0, params=None):
        super().__init__("Avellaneda-Stoikov", initial_cash, initial_inventory)
        params = params or {}
        
        # Initialize order book for AS model
        self.order_book = OrderBook()
        
        # Initialize AS model with parameters
        self.model = AvellanedaStoikovMM(
            symbol="ADAPTER",
            exchange="SIMULATION",
            sigma=params.get('sigma', 0.3),      # Volatility
            gamma=params.get('gamma', 0.1),      # Risk aversion
            k=params.get('k', 1.5),              # Order book liquidity
            c=params.get('c', 1.0),              # Intensity of order arrivals
            T=params.get('T', 1.0),              # Time horizon in days
            initial_cash=initial_cash,
            initial_inventory=initial_inventory,
            max_inventory=params.get('max_inventory', 5.0),
            order_size=params.get('order_size', 0.01),
            min_spread_pct=params.get('min_spread_pct', 0.001)
        )
        
        # Point model's order book to our instance
        self.model.order_book = self.order_book
        
        print("Avellaneda-Stoikov strategy initialized")
    
    def calculate_quotes(self, market_data):
        """Calculate optimal quotes using AS approach"""
        bid = market_data.get('bid')
        ask = market_data.get('ask')
        
        if bid is None or ask is None:
            return None, None
        
        # Update order book
        self.order_book.update([[bid, 1.0]], [[ask, 1.0]])
        
        # Sync inventory between adapter and model
        self.model.inventory = self.inventory
        self.model.cash = self.cash
        
        # Get optimal quotes
        optimal_bid, optimal_ask = self.model.calculate_optimal_quotes()
        
        self.current_bid = optimal_bid
        self.current_ask = optimal_ask
        
        # Update metrics with current mid price
        mid_price = (bid + ask) / 2
        self.update_metrics(mid_price)
        
        return optimal_bid, optimal_ask


class GrossmanMillerAdapter(StrategyAdapter):
    """Adapter for Grossman-Miller market making strategy"""
    
    def __init__(self, initial_cash=100000.0, initial_inventory=0.0, params=None):
        super().__init__("Grossman-Miller", initial_cash, initial_inventory)
        params = params or {}
        
        # Initialize Grossman-Miller model
        self.model = GrossmanMillerModel(
            num_mm=params.get('num_mm', 10),
            gamma=params.get('gamma', 1.0),
            sigma_squared=params.get('sigma_squared', 0.01)
        )
        
        # For trade size estimation
        self.volume_history = deque(maxlen=100)
        self.threshold = params.get('threshold', 1.0)
        
        print("Grossman-Miller strategy initialized")
    
    def calculate_quotes(self, market_data):
        """Calculate optimal quotes using GM approach"""
        bid = market_data.get('bid')
        ask = market_data.get('ask')
        volume = market_data.get('volume')
        
        if bid is None or ask is None:
            return None, None
        
        mid_price = (bid + ask) / 2
        
        # Track volume for threshold calculation
        if volume is not None and volume > 0:
            self.volume_history.append(volume)
        
        # Calculate trade size if we have enough volume history
        if len(self.volume_history) > 10:
            # Adaptive threshold based on volume history
            self.threshold = np.mean(self.volume_history)
            
            # Estimate trade size based on current volume vs threshold
            if volume is not None and volume > self.threshold:
                trade_size = (volume - self.threshold) / 100  # Scale down for reasonable sizes
            else:
                trade_size = 0
        else:
            trade_size = 0
        
        # Calculate price impact and determine optimal quotes
        price_impact, _ = self.model.calculate_price_impact(trade_size)
        
        # Apply price impact to determine quotes
        optimal_bid = mid_price + price_impact * 0.5  # Half impact to bid
        optimal_ask = mid_price - price_impact * 0.5  # Half impact to ask
        
        # Ensure bid < ask
        if optimal_bid >= optimal_ask:
            spread = mid_price * 0.001  # Minimum 0.1% spread
            optimal_bid = mid_price - spread/2
            optimal_ask = mid_price + spread/2
        
        self.current_bid = optimal_bid
        self.current_ask = optimal_ask
        
        # Update metrics
        self.update_metrics(mid_price)
        
        return optimal_bid, optimal_ask


class StrategyComparisonEngine:
    """Engine to run and compare different market making strategies"""
    
    def __init__(self, market_data, strategies, simulation_mode=True, execution_probability=0.05):
        self.market_data = market_data
        self.strategies = strategies
        self.simulation_mode = simulation_mode
        self.execution_probability = execution_probability  # probability of order execution in simulation
        self.running = False
        self.latest_mid_price = None
    
    def start(self):
        """Start the comparison engine"""
        self.running = True
        self.thread = threading.Thread(target=self._run_engine)
        self.thread.daemon = True
        self.thread.start()
        print("Strategy comparison engine started")
    
    def stop(self):
        """Stop the comparison engine"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("Strategy comparison engine stopped")
    
    def _run_engine(self):
        """Run the main engine loop"""
        while self.running:
            # Get next market data update
            update = self.market_data.get_next_update(timeout=0.1)
            
            if update is None:
                continue
            
            # Extract market data
            market_data = {
                'bid': update.get('bid'),
                'ask': update.get('ask'),
                'trade': update.get('trade'),
                'volume': update.get('volume'),
                'timestamp': update.get('timestamp')
            }
            
            # Skip if we don't have sufficient data
            if market_data['bid'] is None or market_data['ask'] is None:
                continue
            
            # Calculate mid price for execution probability calculation
            mid_price = (market_data['bid'] + market_data['ask']) / 2
            self.latest_mid_price = mid_price
            
            # Process each strategy
            for strategy in self.strategies:
                # Calculate optimal quotes
                try:
                    bid, ask = strategy.calculate_quotes(market_data)
                    
                    # Skip if quotes couldn't be calculated
                    if bid is None or ask is None:
                        continue
                    
                    # In simulation mode, simulate order executions
                    if self.simulation_mode:
                        # Check for buy execution (someone hits our bid)
                        if random.random() < self.execution_probability:
                            # More likely to execute if our bid is higher than most recent trade
                            if market_data['trade'] is not None and bid > market_data['trade']:
                                execution_probability_buy = self.execution_probability * 2
                            else:
                                execution_probability_buy = self.execution_probability * 0.5
                                
                            if random.random() < execution_probability_buy:
                                size = random.uniform(0.01, 0.1)  # Random size between 0.01 and 0.1
                                execution_price = bid * (1 + random.uniform(-0.0001, 0.0001))  # Small price noise
                                
                                # Execute the buy (someone sold to us)
                                strategy.execute_trade(size, execution_price, 'BUY')
                                print(f"{strategy.strategy_name} BUY executed: {size:.4f} @ {execution_price:.4f}")
                        
                        # Check for sell execution (someone lifts our ask)
                        if random.random() < self.execution_probability:
                            # More likely to execute if our ask is lower than most recent trade
                            if market_data['trade'] is not None and ask < market_data['trade']:
                                execution_probability_sell = self.execution_probability * 2
                            else:
                                execution_probability_sell = self.execution_probability * 0.5
                                
                            if random.random() < execution_probability_sell:
                                size = random.uniform(0.01, 0.1)  # Random size
                                execution_price = ask * (1 + random.uniform(-0.0001, 0.0001))  # Small price noise
                                
                                # Execute the sell (someone bought from us)
                                strategy.execute_trade(size, execution_price, 'SELL')
                                print(f"{strategy.strategy_name} SELL executed: {size:.4f} @ {execution_price:.4f}")
                    
                except Exception as e:
                    print(f"Error in {strategy.strategy_name} calculation: {e}")
                    continue
            
            # Control loop timing to prevent 100% CPU usage
            time.sleep(0.01)
    
    def get_comparison_results(self):
        """Get the current comparison results for all strategies"""
        results = []
        
        for strategy in self.strategies:
            metrics = strategy.get_metrics()
            results.append(metrics)
        
        # Sort by PnL for easier comparison
        results.sort(key=lambda x: x['pnl'], reverse=True)
        return results


class ComparisonDashboard:
    """Interactive dashboard for comparing market making strategies"""
    
    def __init__(self, comparison_engine, update_interval=500):
        self.comparison_engine = comparison_engine
        self.update_interval = update_interval
        
        # Initialize collections for time-series data
        self.timestamps = deque(maxlen=500)
        self.mid_prices = deque(maxlen=500)
        
        # Initialize app with layout
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Market Making Strategy Comparison", 
                        style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Div(id='status-indicator', 
                         style={'textAlign': 'center', 'marginBottom': '20px'}),
            ]),
            
            # Main dashboard area
            html.Div([
                # Left column - charts
                html.Div([
                    # Price and quotes chart
                    html.Div([
                        html.H3("Market Prices & Strategy Quotes", 
                               style={'textAlign': 'center', 'color': '#2c3e50'}),
                        dcc.Graph(id='price-quote-chart'),
                    ], style={'marginBottom': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                    
                    # PnL comparison chart
                    html.Div([
                        html.H3("P&L Comparison", 
                               style={'textAlign': 'center', 'color': '#2c3e50'}),
                        dcc.Graph(id='pnl-chart'),
                    ], style={'marginBottom': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                    
                    # Inventory comparison chart
                    html.Div([
                        html.H3("Inventory Comparison", 
                               style={'textAlign': 'center', 'color': '#2c3e50'}),
                        dcc.Graph(id='inventory-chart'),
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                # Right column - metrics and tables
                html.Div([
                    # Performance metrics table
                    html.Div([
                        html.H3("Performance Metrics", 
                               style={'textAlign': 'center', 'color': '#2c3e50'}),
                        html.Div(id='metrics-table'),
                    ], style={'marginBottom': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                    
                    # Latest quotes table
                    html.Div([
                        html.H3("Latest Quotes", 
                               style={'textAlign': 'center', 'color': '#2c3e50'}),
                        html.Div(id='quotes-table'),
                    ], style={'marginBottom': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                    
                    # Strategies configuration
                    html.Div([
                        html.H3("Strategy Configuration", 
                               style={'textAlign': 'center', 'color': '#2c3e50'}),
                        html.Div(id='strategy-config'),
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                ], style={'width': '29%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'}),
            ]),
            
            # Update interval component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
        ], style={'padding': '20px', 'fontFamily': 'Arial'})
    
    def setup_callbacks(self):
        """Set up the dashboard callbacks"""
        @self.app.callback(
            [Output('status-indicator', 'children'),
             Output('price-quote-chart', 'figure'),
             Output('pnl-chart', 'figure'),
             Output('inventory-chart', 'figure'),
             Output('metrics-table', 'children'),
             Output('quotes-table', 'children'),
             Output('strategy-config', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Update timestamp and price history
            now = datetime.now()
            self.timestamps.append(now)
            
            mid_price = self.comparison_engine.latest_mid_price
            if mid_price is not None:
                self.mid_prices.append(mid_price)
            elif self.mid_prices:
                # Use last known price if no update
                self.mid_prices.append(self.mid_prices[-1])
            else:
                # Default if no prices yet
                self.mid_prices.append(100.0)
            
            # Get current results for all strategies
            results = self.comparison_engine.get_comparison_results()
            
            # Create status indicator
            if results:
                status = html.Div([
                    html.Span("Status: ", style={'fontWeight': 'bold'}),
                    html.Span("Active", style={'color': 'green', 'fontWeight': 'bold'}),
                    html.Span(" | Current Price: ", style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                    html.Span(f"${self.mid_prices[-1]:.4f}" if self.mid_prices else "Unknown", 
                             style={'fontWeight': 'bold'})
                ])
            else:
                status = html.Div([
                    html.Span("Status: ", style={'fontWeight': 'bold'}),
                    html.Span("Waiting for data...", style={'color': 'orange'})
                ])
            
            # Create price and quotes chart
            price_fig = go.Figure()
            
            # Add mid price line
            x_data = [t.strftime('%H:%M:%S') for t in self.timestamps]
            price_fig.add_trace(go.Scatter(
                x=x_data, y=list(self.mid_prices),
                mode='lines',
                name='Mid Price',
                line=dict(color='black', width=2)
            ))
            
            # Add quotes for each strategy
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, strategy in enumerate(self.comparison_engine.strategies):
                if strategy.bid_history and strategy.ask_history:
                    # Ensure we only plot up to the available length
                    max_len = min(len(x_data), len(strategy.bid_history))
                    
                    # Bids
                    price_fig.add_trace(go.Scatter(
                        x=x_data[-max_len:], 
                        y=strategy.bid_history[-max_len:],
                        mode='lines',
                        name=f'{strategy.strategy_name} Bid',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                    ))
                    
                    # Asks
                    price_fig.add_trace(go.Scatter(
                        x=x_data[-max_len:], 
                        y=strategy.ask_history[-max_len:],
                        mode='lines',
                        name=f'{strategy.strategy_name} Ask',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                    ))
            
            price_fig.update_layout(
                title='Market Price and Strategy Quotes',
                xaxis_title='Time',
                yaxis_title='Price',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            # Create PnL comparison chart
            pnl_fig = go.Figure()
            
            for i, strategy in enumerate(self.comparison_engine.strategies):
                if strategy.pnl_history:
                    # Ensure we only plot up to the available length
                    max_len = min(len(x_data), len(strategy.pnl_history))
                    
                    pnl_fig.add_trace(go.Scatter(
                        x=x_data[-max_len:], 
                        y=strategy.pnl_history[-max_len:],
                        mode='lines',
                        name=f'{strategy.strategy_name} P&L',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            pnl_fig.update_layout(
                title='P&L Comparison',
                xaxis_title='Time',
                yaxis_title='P&L ($)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=300
            )
            
            # Create inventory comparison chart
            inv_fig = go.Figure()
            
            for i, strategy in enumerate(self.comparison_engine.strategies):
                if strategy.inventory_history:
                    # Ensure we only plot up to the available length
                    max_len = min(len(x_data), len(strategy.inventory_history))
                    
                    inv_fig.add_trace(go.Scatter(
                        x=x_data[-max_len:], 
                        y=strategy.inventory_history[-max_len:],
                        mode='lines',
                        name=f'{strategy.strategy_name} Inventory',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            inv_fig.update_layout(
                title='Inventory Comparison',
                xaxis_title='Time',
                yaxis_title='Inventory',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=300
            )
            
            # Create metrics table
            if results:
                metrics_table = html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th('Strategy', style={'textAlign': 'left', 'padding': '8px'}),
                            html.Th('P&L', style={'textAlign': 'right', 'padding': '8px'}),
                            html.Th('Cash', style={'textAlign': 'right', 'padding': '8px'}),
                            html.Th('Inventory', style={'textAlign': 'right', 'padding': '8px'}),
                            html.Th('Portfolio Value', style={'textAlign': 'right', 'padding': '8px'})
                        ], style={'backgroundColor': '#2c3e50', 'color': 'white'})
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(result['strategy'], style={'padding': '8px'}),
                            html.Td(f"${result['pnl']:.2f}", 
                                   style={'textAlign': 'right', 'padding': '8px', 
                                         'color': 'green' if result['pnl'] >= 0 else 'red'}),
                            html.Td(f"${result['cash']:.2f}", style={'textAlign': 'right', 'padding': '8px'}),
                            html.Td(f"{result['inventory']:.4f}", style={'textAlign': 'right', 'padding': '8px'}),
                            html.Td(f"${result['portfolio_value']:.2f}", style={'textAlign': 'right', 'padding': '8px'})
                        ], style={'backgroundColor': '#f2f2f2' if i % 2 else 'white'})
                        for i, result in enumerate(results)
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
            else:
                metrics_table = html.Div("Waiting for data...")
            
            # Create quotes table
            if results:
                quotes_table = html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th('Strategy', style={'textAlign': 'left', 'padding': '8px'}),
                            html.Th('Bid', style={'textAlign': 'right', 'padding': '8px'}),
                            html.Th('Ask', style={'textAlign': 'right', 'padding': '8px'}),
                            html.Th('Spread', style={'textAlign': 'right', 'padding': '8px'}),
                            html.Th('Spread %', style={'textAlign': 'right', 'padding': '8px'})
                        ], style={'backgroundColor': '#2c3e50', 'color': 'white'})
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(result['strategy'], style={'padding': '8px'}),
                            html.Td(f"${result['current_bid']:.4f}", style={'textAlign': 'right', 'padding': '8px'}),
                            html.Td(f"${result['current_ask']:.4f}", style={'textAlign': 'right', 'padding': '8px'}),
                            html.Td(f"${result['current_ask'] - result['current_bid']:.4f}", 
                                   style={'textAlign': 'right', 'padding': '8px'}),
                            html.Td(f"{((result['current_ask'] - result['current_bid']) / result['current_bid'] * 100):.2f}%", 
                                   style={'textAlign': 'right', 'padding': '8px'})
                        ], style={'backgroundColor': '#f2f2f2' if i % 2 else 'white'})
                        for i, result in enumerate(results)
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
            else:
                quotes_table = html.Div("Waiting for data...")
            
            # Create strategy configuration
            strategy_config = html.Div([
                html.P("Strategy parameters:", style={'fontWeight': 'bold'}),
                html.Ul([
                    html.Li([
                        html.Span(f"{strategy.strategy_name}: ", style={'fontWeight': 'bold'}),
                        html.Span(f"Initial cash: ${strategy.initial_cash:.2f}, Initial inventory: {strategy.initial_inventory:.4f}")
                    ]) for strategy in self.comparison_engine.strategies
                ])
            ])
            
            return status, price_fig, pnl_fig, inv_fig, metrics_table, quotes_table, strategy_config
    
    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)
    
    def start_background(self, port=8050):
        """Start the dashboard in a background thread"""
        self.thread = threading.Thread(target=lambda: self.app.run_server(debug=False, port=port, use_reloader=False))
        self.thread.daemon = True
        self.thread.start()
        print(f"Dashboard started at http://localhost:{port}")
        return self.thread


def run_strategy_comparison(symbol="btcusdt", 
                           data_file=None, 
                           duration_hours=1,
                           simulation_mode=True,
                           initial_cash=100000.0,
                           initial_inventory=0.0,
                           dashboard_port=8050):
    """
    Run a comparison of different market making strategies
    
    Args:
        symbol: Trading symbol to use
        data_file: Path to historical data file (optional)
        duration_hours: How long to run the simulation (if not using data_file)
        simulation_mode: Whether to simulate order executions
        initial_cash: Starting cash for each strategy
        initial_inventory: Starting inventory for each strategy
        dashboard_port: Port to run the dashboard on
    """
    print(f"Starting strategy comparison for {symbol}")
    
    # Initialize market data source
    market_data = UnifiedMarketData(symbol, from_file=data_file)
    
    # Initialize strategies with the same starting conditions
    strategies = [
        HJBStrategyAdapter(
            initial_cash=initial_cash,
            initial_inventory=initial_inventory,
            params={
                'sigma': 0.2,
                'kappa': 0.001,
                'gamma': 0.0001,
                'rho': 0.01,
                'market_impact': 0.0001,
                'jump_intensity': 0.1,
                'jump_mean': 0.0,
                'jump_std': 0.01,
                'grid_size': 51
            }
        ),
        AvellanedaStoikovAdapter(
            initial_cash=initial_cash,
            initial_inventory=initial_inventory,
            params={
                'sigma': 0.3,
                'gamma': 0.1,
                'k': 1.5,
                'c': 1.0,
                'T': 1.0,
                'max_inventory': 5.0,
                'order_size': 0.01,
                'min_spread_pct': 0.001
            }
        ),
        GrossmanMillerAdapter(
            initial_cash=initial_cash,
            initial_inventory=initial_inventory,
            params={
                'num_mm': 10,
                'gamma': 1.0,
                'sigma_squared': 0.01,
                'threshold': 1.0
            }
        )
    ]
    
    # Initialize comparison engine
    comparison_engine = StrategyComparisonEngine(
        market_data=market_data,
        strategies=strategies,
        simulation_mode=simulation_mode,
        execution_probability=0.05  # 5% chance of execution per update
    )
    
    # Start the engine
    comparison_engine.start()
    
    # Initialize and start dashboard
    dashboard = ComparisonDashboard(comparison_engine)
    dashboard.start_background(port=dashboard_port)
    
    # Run for specified duration or until user interrupts
    try:
        print(f"Running comparison for {duration_hours} hours. Press Ctrl+C to stop.")
        end_time = time.time() + (duration_hours * 3600)
        
        while time.time() < end_time:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping strategy comparison...")
    finally:
        # Stop the engine
        comparison_engine.stop()
        
        # Display final results
        results = comparison_engine.get_comparison_results()
        print("\nFinal Results:")
        print("-" * 80)
        for result in results:
            print(f"{result['strategy']} | PnL: ${result['pnl']:.2f} | Final Inventory: {result['inventory']:.4f}")
        print("-" * 80)
        
        # Wait for user to close dashboard
        print("\nYou can still view the dashboard. Press Ctrl+C again to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    # Example usage with historical data file
    data_file = "/home/misango/code/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation/sample_data.csv"
    
    run_strategy_comparison(
        symbol="btcusdt",
        data_file=data_file,  # Use None to fetch live data
        duration_hours=1,     # Run for 1 hour (ignored if using data_file)
        simulation_mode=True, # Simulate order executions
        initial_cash=100000.0,
        initial_inventory=0.0,
        dashboard_port=8050
    )