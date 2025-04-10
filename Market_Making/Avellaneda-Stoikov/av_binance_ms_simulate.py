import asyncio
import time
import datetime
from datetime import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt.async_support as ccxt
import json
import logging
from typing import Dict, List, Tuple, Optional
import websockets
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from threading import Thread
import threading
import websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_maker.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("CryptoMM")

# ...existing code...

class MarketDataCollector:
    def __init__(self):
        self.combined_data = []

    def on_order_book_message(self, ws, message):
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

class OrderBook:
    def __init__(self, max_depth=10):
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}  # price -> size
        self.last_update_time = None
        self.max_depth = max_depth
        
    def update(self, bids: List[List[float]], asks: List[List[float]]):
        """Update order book with new bids and asks."""
        # Update bids (price, size)
        for bid in bids:
            price, size = float(bid[0]), float(bid[1])
            if size > 0:
                self.bids[price] = size
            else:
                self.bids.pop(price, None)
                
        # Update asks (price, size)
        for ask in asks:
            price, size = float(ask[0]), float(ask[1])
            if size > 0:
                self.asks[price] = size
            else:
                self.asks.pop(price, None)
        
        self.last_update_time = time.time()
    
    def get_mid_price(self) -> float:
        """Get the mid price from the order book."""
        if not self.bids or not self.asks:
            return 0
        
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        
        return (best_bid + best_ask) / 2
    
    def get_best_bid_ask(self) -> Tuple[float, float]:
        """Get the best bid and ask prices."""
        if not self.bids or not self.asks:
            return 0, 0
        
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        
        return best_bid, best_ask
    
    def get_market_depth(self, depth=None) -> Tuple[List[List[float]], List[List[float]]]:
        """Get market depth up to specified level."""
        if depth is None:
            depth = self.max_depth
            
        # Sort bids in descending order
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:depth]
        # Sort asks in ascending order
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        
        return sorted_bids, sorted_asks
    
    def calculate_market_impact(self, size: float, side: str) -> float:
        """
        Calculate the price impact of a market order of given size.
        Returns the executed price after slippage.
        """
        if side.lower() == 'buy':
            # For buy orders, we walk up the ask book
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
            remaining_size = size
            weighted_price = 0
            total_executed = 0
            
            for price, available_size in sorted_asks:
                executed = min(remaining_size, available_size)
                weighted_price += executed * price
                total_executed += executed
                remaining_size -= executed
                
                if remaining_size <= 0:
                    break
            
            return weighted_price / total_executed if total_executed > 0 else None
            
        elif side.lower() == 'sell':
            # For sell orders, we walk down the bid book
            sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
            remaining_size = size
            weighted_price = 0
            total_executed = 0
            
            for price, available_size in sorted_bids:
                executed = min(remaining_size, available_size)
                weighted_price += executed * price
                total_executed += executed
                remaining_size -= executed
                
                if remaining_size <= 0:
                    break
            
            return weighted_price / total_executed if total_executed > 0 else None
        
        return None

class AvellanedaStoikovMM:
    def __init__(self, 
                 symbol: str,
                 exchange: str,
                 sigma: float,      # Volatility of the mid-price process
                 gamma: float,      # Risk aversion coefficient
                 k: float,          # Order book liquidity parameter
                 c: float,          # Intensity of order arrivals
                 T: float,          # Time horizon in days
                 initial_cash: float = 10000.0,
                 initial_inventory: float = 0.0,
                 max_inventory: float = 5.0,  # Maximum allowed inventory
                 order_size: float = 0.01,    # Size of each order
                 min_spread_pct: float = 0.001,  # Minimum spread as percentage
                 ):
        self.symbol = symbol
        self.exchange_name = exchange
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.c = c
        self.T = T
        
        # Position and risk parameters
        self.cash = initial_cash
        self.inventory = initial_inventory
        self.max_inventory = max_inventory
        self.order_size = order_size
        self.min_spread_pct = min_spread_pct
        
        # Trading state
        self.order_book = OrderBook()
        self.current_bid_id = None
        self.current_ask_id = None
        self.current_bid_price = None
        self.current_ask_price = None
        self.start_time = time.time()
        
        # Performance tracking
        self.trading_history = []
        self.pnl_history = []
        self.inventory_history = []
        self.quote_history = []
        self.mid_price_history = []
        self.timestamp_history = []
        
        logger.info(f"Initialized market maker for {symbol} on {exchange}")
    
    def calculate_volatility(self, price_history: List[float], window: int = 100) -> float:
        """Calculate rolling volatility from price history."""
        if len(price_history) < window:
            return self.sigma  # Use default if not enough data
        
        returns = np.diff(np.log(price_history[-window:]))
        return np.std(returns) * np.sqrt(86400)  # Scale to daily volatility
    
    def optimal_bid_spread(self, inventory: float, remaining_time: float, sigma: float) -> float:
        """Calculate the optimal bid spread using equation (23)."""
        return ((2*inventory + 1) * self.gamma * sigma**2 * remaining_time / 2 + 
                np.log(1 + self.gamma / self.k) / self.gamma)
    
    def optimal_ask_spread(self, inventory: float, remaining_time: float, sigma: float) -> float:
        """Calculate the optimal ask spread using equation (24)."""
        return ((1 - 2*inventory) * self.gamma * sigma**2 * remaining_time / 2 + 
                np.log(1 + self.gamma / self.k) / self.gamma)
    
    def apply_inventory_constraints(self, bid_price: float, ask_price: float) -> Tuple[float, float]:
        """Apply inventory constraints to quotes."""
        mid_price = self.order_book.get_mid_price()
        
        # If inventory is near max, reduce or remove bid
        if self.inventory >= self.max_inventory * 0.8:
            bid_price = 0  # Don't place bid
        elif self.inventory >= self.max_inventory * 0.5:
            # Reduce bid aggressiveness as inventory increases
            inventory_factor = (self.max_inventory - self.inventory) / self.max_inventory
            bid_price = mid_price - (mid_price - bid_price) / inventory_factor
        
        # If inventory is near min (negative max), reduce or remove ask
        if self.inventory <= -self.max_inventory * 0.8:
            ask_price = float('inf')  # Don't place ask
        elif self.inventory <= -self.max_inventory * 0.5:
            # Reduce ask aggressiveness as negative inventory increases
            inventory_factor = (self.max_inventory + self.inventory) / self.max_inventory
            ask_price = mid_price + (ask_price - mid_price) / inventory_factor
        
        return bid_price, ask_price
    
    def calculate_optimal_quotes(self) -> Tuple[float, float]:
        """Calculate optimal bid and ask prices based on current market state."""
        current_time = time.time()
        elapsed_days = (current_time - self.start_time) / 86400
        remaining_time = max(0.001, self.T - elapsed_days)
        
        # Get current mid price
        mid_price = self.order_book.get_mid_price()
        if mid_price == 0:
            logger.warning("Mid price is zero - cannot calculate quotes")
            return 0, 0
            
        # Calculate optimal spreads
        bid_spread = self.optimal_bid_spread(self.inventory, remaining_time, self.sigma)
        ask_spread = self.optimal_ask_spread(self.inventory, remaining_time, self.sigma)
        
        # Calculate optimal prices
        bid_price = mid_price - bid_spread
        ask_price = mid_price + ask_spread
        
        # Apply inventory constraints
        bid_price, ask_price = self.apply_inventory_constraints(bid_price, ask_price)
        
        # Ensure minimum spread
        min_spread = mid_price * self.min_spread_pct
        if 0 < bid_price < float('inf') and 0 < ask_price < float('inf'):
            if ask_price - bid_price < min_spread:
                bid_price = mid_price - min_spread/2
                ask_price = mid_price + min_spread/2
        
        # Round to appropriate precision
        decimals = 2
        if mid_price < 100:
            decimals = 3
        if mid_price < 10:
            decimals = 4
        if mid_price < 1:
            decimals = 6
        
        if bid_price > 0:
            bid_price = round(bid_price, decimals)
        if ask_price < float('inf'):
            ask_price = round(ask_price, decimals)
        
        # Log the quote calculation
        logger.info(f"Mid: {mid_price:.6f}, Inv: {self.inventory:.6f}, " +
                    f"Optimal Bid: {bid_price:.6f}, Ask: {ask_price:.6f}")
        
        return bid_price, ask_price
    
    def update_pnl_and_metrics(self, mid_price: float):
        """Update PnL and performance metrics."""
        mark_to_market = self.cash + self.inventory * mid_price
        
        timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mid_price_history.append(mid_price)
        self.pnl_history.append(mark_to_market)
        self.inventory_history.append(self.inventory)
        self.timestamp_history.append(timestamp)
        
        if self.current_bid_price and self.current_ask_price:
            self.quote_history.append({
                'timestamp': timestamp,
                'mid_price': mid_price,
                'bid_price': self.current_bid_price,
                'ask_price': self.current_ask_price,
                'inventory': self.inventory,
                'pnl': mark_to_market
            })
    
    def record_trade(self, side: str, price: float, size: float, is_taker: bool = False):
        """Record a completed trade."""
        trade_type = "Taker" if is_taker else "Maker"
        mid_price = self.order_book.get_mid_price()
        
        self.trading_history.append({
            'timestamp': dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            'side': side,
            'price': price,
            'size': size,
            'mid_price': mid_price,
            'trade_type': trade_type,
            'inventory_after': self.inventory,
            'cash_after': self.cash,
        })
        
        logger.info(f"Trade executed: {trade_type} {side} {size} @ {price}")
        
        # Update inventory and cash based on the trade
        if side == 'buy':
            self.inventory += size
            self.cash -= price * size
        else:  # sell
            self.inventory -= size
            self.cash += price * size
            
        # Update metrics
        self.update_pnl_and_metrics(mid_price)

class ExchangeAdapter:
    def __init__(self, exchange_id: str, api_key: str = None, secret: str = None):
        """Initialize exchange adapter with authentication if provided."""
        self.exchange_id = exchange_id
        
        # Initialize with ccxt
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        logger.info(f"Initialized {exchange_id} adapter")
        
    async def fetch_order_book(self, symbol: str, depth: int = 10):
        """Fetch order book from the exchange."""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, depth)
            return orderbook['bids'], orderbook['asks']
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return [], []
    
    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float):
        """Create a limit order on the exchange."""
        try:
            if not amount or not price or price <= 0:
                logger.warning(f"Invalid order parameters - side: {side}, amount: {amount}, price: {price}")
                return None
                
            order = await self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"Created {side} limit order: {amount} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Error creating limit order: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str):
        """Cancel an existing order."""
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return None
    
    async def fetch_balance(self):
        """Fetch account balance."""
        try:
            balance = await self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    async def close(self):
        """Close the exchange connection."""
        await self.exchange.close()
        logger.info(f"Closed {self.exchange_id} connection")

class LiveMarketMaker:
    def __init__(self, 
                 exchange_id: str,
                 symbol: str,
                 api_key: str = None, 
                 secret: str = None,
                 sigma: float = 0.3,
                 gamma: float = 0.1,
                 k: float = 1.5,
                 c: float = 1.0,
                 T: float = 1.0,  # 1 day
                 initial_cash: float = 10000.0,
                 initial_inventory: float = 0.0,
                 max_inventory: float = 5.0,
                 order_size: float = 0.01,
                 update_interval: float = 5.0,  # seconds
                 min_spread_pct: float = 0.001
                ):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.api_key = api_key
        self.secret = secret
        self.update_interval = update_interval
        
        # Initialize exchange adapter
        self.exchange = None
        
        # Initialize market maker strategy
        self.market_maker = AvellanedaStoikovMM(
            symbol=symbol,
            exchange=exchange_id,
            sigma=sigma,
            gamma=gamma,
            k=k,
            c=c,
            T=T,
            initial_cash=initial_cash,
            initial_inventory=initial_inventory,
            max_inventory=max_inventory,
            order_size=order_size,
            min_spread_pct=min_spread_pct
        )
        
        # Trading state
        self.is_running = False
        self.current_bid_id = None
        self.current_ask_id = None
        
        logger.info(f"Initialized live market maker for {symbol} on {exchange_id}")

    async def initialize(self):
        """Initialize the exchange connection."""
        self.exchange = ExchangeAdapter(self.exchange_id, self.api_key, self.secret)
        
        # Fetch initial balance
        if self.api_key and self.secret:
            balance = await self.exchange.fetch_balance()
            if balance:
                logger.info(f"Initial balance: {balance['total']}")
                
        # Fetch initial order book
        bids, asks = await self.exchange.fetch_order_book(self.symbol)
        self.market_maker.order_book.update(bids, asks)
        
    async def update_order_book(self):
        """Update the local order book with latest data."""
        bids, asks = await self.exchange.fetch_order_book(self.symbol)
        self.market_maker.order_book.update(bids, asks)
        
        # Calculate and record mid price
        mid_price = self.market_maker.order_book.get_mid_price()
        self.market_maker.update_pnl_and_metrics(mid_price)
        
        return mid_price
        
    async def refresh_quotes(self):
        """Calculate new quotes and update orders if necessary."""
        # Calculate optimal quotes
        bid_price, ask_price = self.market_maker.calculate_optimal_quotes()
        
        # Check if we need to update orders
        need_update_bid = self.current_bid_id is None or abs(bid_price - self.market_maker.current_bid_price) > 0.0001
        need_update_ask = self.current_ask_id is None or abs(ask_price - self.market_maker.current_ask_price) > 0.0001
        
        if need_update_bid or need_update_ask:
            # Cancel existing orders if needed
            if need_update_bid and self.current_bid_id:
                await self.exchange.cancel_order(self.current_bid_id, self.symbol)
                self.current_bid_id = None
                
            if need_update_ask and self.current_ask_id:
                await self.exchange.cancel_order(self.current_ask_id, self.symbol)
                self.current_ask_id = None
            
            # Place new bid if appropriate
            if bid_price > 0 and need_update_bid:
                order = await self.exchange.create_limit_order(
                    self.symbol, 'buy', self.market_maker.order_size, bid_price
                )
                if order:
                    self.current_bid_id = order['id']
                    self.market_maker.current_bid_price = bid_price
            
            # Place new ask if appropriate
            if ask_price < float('inf') and need_update_ask:
                order = await self.exchange.create_limit_order(
                    self.symbol, 'sell', self.market_maker.order_size, ask_price
                )
                if order:
                    self.current_ask_id = order['id']
                    self.market_maker.current_ask_price = ask_price
    
    async def run(self):
        """Main loop for the live market maker."""
        logger.info("Starting market maker")
        self.is_running = True
        
        try:
            await self.initialize()
            
            while self.is_running:
                # Update order book
                mid_price = await self.update_order_book()
                
                # Calculate and update quotes if appropriate
                await self.refresh_quotes()
                
                # Wait for next update cycle
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Error in market maker loop: {e}", exc_info=True)
        finally:
            # Clean up
            self.is_running = False
            
            # Cancel any open orders
            if self.current_bid_id:
                await self.exchange.cancel_order(self.current_bid_id, self.symbol)
            if self.current_ask_id:
                await self.exchange.cancel_order(self.current_ask_id, self.symbol)
                
            # Close exchange connection
            if self.exchange:
                await self.exchange.close()
            
            logger.info("Market maker stopped")
    
    def stop(self):
        """Signal the market maker to stop."""
        logger.info("Stopping market maker")
        self.is_running = False

class MarketMakerDashboard:
    def __init__(self, market_maker):
        """Initialize dashboard for visualizing market maker performance."""
        self.market_maker = market_maker
        self.app = dash.Dash(__name__)
        
        # Define dashboard layout
        self.app.layout = html.Div([
            html.H1(f"Market Maker Dashboard - {market_maker.symbol}"),
            
            html.Div([
                html.Div([
                    html.H3("Current Status"),
                    html.Table([
                        html.Tr([html.Th("Metric"), html.Th("Value")]),
                        html.Tr([html.Td("Mid Price"), html.Td(id='mid-price')]),
                        html.Tr([html.Td("Inventory"), html.Td(id='inventory')]),
                        html.Tr([html.Td("Cash"), html.Td(id='cash')]),
                        html.Tr([html.Td("PnL"), html.Td(id='pnl')]),
                        html.Tr([html.Td("Bid Quote"), html.Td(id='bid-quote')]),
                        html.Tr([html.Td("Ask Quote"), html.Td(id='ask-quote')]),
                    ], style={'width': '100%', 'border': '1px solid black'})
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Order Book"),
                    dcc.Graph(id='orderbook-graph')
                ], style={'width': '70%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.H3("Performance Charts"),
                dcc.Tabs([
                    dcc.Tab(label="Price & Quotes", children=[
                        dcc.Graph(id='price-quotes-graph')
                    ]),
                    dcc.Tab(label="Inventory", children=[
                        dcc.Graph(id='inventory-graph')
                    ]),
                    dcc.Tab(label="PnL", children=[
                        dcc.Graph(id='pnl-graph')
                    ]),
                    dcc.Tab(label="Recent Trades", children=[
                        html.Div(id='recent-trades-table')
                    ])
                ])
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=2*1000,  # Update every 2 seconds
                n_intervals=0
            )
        ])
        
        # Define callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for updating the UI."""
        @self.app.callback(
            [Output('mid-price', 'children'),
             Output('inventory', 'children'),
             Output('cash', 'children'),
             Output('pnl', 'children'),
             Output('bid-quote', 'children'),
             Output('ask-quote', 'children'),
             Output('orderbook-graph', 'figure'),
             Output('price-quotes-graph', 'figure'),
             Output('inventory-graph', 'figure'),
             Output('pnl-graph', 'figure'),
             Output('recent-trades-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Current status values
            mid_price = self.market_maker.order_book.get_mid_price()
            inventory = self.market_maker.inventory
            cash = self.market_maker.cash
            pnl = cash + inventory * mid_price
            bid_quote = self.market_maker.current_bid_price or "N/A"
            ask_quote = self.market_maker.current_ask_price or "N/A"
            
            # Order book visualization
            bid_data, ask_data = self.market_maker.order_book.get_market_depth()
            
            orderbook_fig = go.Figure()
            # Add bids
            if bid_data:
                prices, sizes = zip(*bid_data)
                orderbook_fig.add_trace(go.Bar(
                    x=prices, y=sizes, name='Bids',
                    marker_color='green'
                ))
            
            # Add asks
            if ask_data:
                prices, sizes = zip(*ask_data)
                orderbook_fig.add_trace(go.Bar(
                    x=prices, y=sizes, name='Asks',
                    marker_color='red'
                ))
                
            orderbook_fig.update_layout(
                title='Order Book Depth',
                xaxis_title='Price',
                yaxis_title='Size',
                barmode='group'
            )
            
            # Price and quotes chart
            price_quotes_df = pd.DataFrame({
                'timestamp': self.market_maker.timestamp_history,
                'mid_price': self.market_maker.mid_price_history,
            })
            
            # Add quotes where available
            quote_data = pd.DataFrame(self.market_maker.quote_history)
            
            price_quotes_fig = go.Figure()
            price_quotes_fig.add_trace(go.Scatter(
                x=price_quotes_df['timestamp'], 
                y=price_quotes_df['mid_price'],
                mode='lines',
                name='Mid Price'
            ))
            
            if not quote_data.empty:
                price_quotes_fig.add_trace(go.Scatter(
                    x=quote_data['timestamp'], 
                    y=quote_data['bid_price'],
                    mode='lines',
                    name='Bid Quote',
                    line=dict(color='green')
                ))
                
                price_quotes_fig.add_trace(go.Scatter(
                    x=quote_data['timestamp'], 
                    y=quote_data['ask_price'],
                    mode='lines',
                    name='Ask Quote',
                    line=dict(color='red')
                ))
            
            price_quotes_fig.update_layout(
                title='Price and Quotes',
                xaxis_title='Time',
                yaxis_title='Price'
            )
            
            # Inventory chart
            inventory_df = pd.DataFrame({
                'timestamp': self.market_maker.timestamp_history,
                'inventory': self.market_maker.inventory_history
            })
            
            inventory_fig = px.line(
                inventory_df, x='timestamp', y='inventory',
                title='Inventory Over Time'
            )
            
            # PnL chart
            pnl_df = pd.DataFrame({
                'timestamp': self.market_maker.timestamp_history,
                'pnl': self.market_maker.pnl_history
            })
            
            pnl_fig = px.line(
                pnl_df, x='timestamp', y='pnl',
                title='Mark-to-Market PnL'
            )
            
            # Recent trades table
            trades = self.market_maker.trading_history[-10:]  # Last 10 trades
            if trades:
                trades_table = html.Table(
                    # Header
                    [html.Tr([html.Th(col) for col in ['Time', 'Side', 'Price', 'Size', 'Type']])] +
                    # Rows
                    [html.Tr([
                        html.Td(trade['timestamp']),
                        html.Td(trade['side']),
                        html.Td(f"{trade['price']:.6f}"),
                        html.Td(f"{trade['size']:.6f}"),
                        html.Td(trade['trade_type'])
                    ]) for trade in trades]
                )
            else:
                trades_table = html.Div("No trades yet")
            
            return (
                f"{mid_price:.6f}",
                f"{inventory:.6f}",
                f"${cash:.2f}",
                f"${pnl:.2f}",
                f"{bid_quote if bid_quote != 'N/A' else 'N/A'}",
                f"{ask_quote if ask_quote != 'N/A' else 'N/A'}",
                orderbook_fig,
                price_quotes_fig,
                inventory_fig,
                pnl_fig,
                trades_table
            )
    
    def run(self, debug=False, port=8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port)
    
    def start_background(self, port=8050):
        """Start the dashboard server in a background thread."""
        dashboard_thread = Thread(target=lambda: self.app.run_server(debug=False, port=port))
        dashboard_thread.daemon = True
        dashboard_thread.start()
        return dashboard_thread

class WebSocketOrderBookHandler:
    """Handler for real-time order book data via WebSockets."""
    
    def __init__(self, exchange_id: str, symbol: str, market_maker: AvellanedaStoikovMM):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.market_maker = market_maker
        self.ws = None
        self.is_running = False
        
        # Select appropriate WebSocket URL and message format based on exchange
        if exchange_id.lower() == 'binance':
            self.ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower().replace('/', '')}@depth20@100ms"
            self.parse_message = self._parse_binance_message
        elif exchange_id.lower() == 'coinbasepro':
            self.ws_url = "wss://ws-feed.pro.coinbase.com"
            self.parse_message = self._parse_coinbase_message
        else:
            raise ValueError(f"WebSocket support not implemented for exchange: {exchange_id}")
    
    async def _parse_binance_message(self, msg):
        """Parse Binance WebSocket message."""
        data = json.loads(msg)
        bids = [[float(price), float(qty)] for price, qty in data['bids']]
        asks = [[float(price), float(qty)] for price, qty in data['asks']]
        return bids, asks
    
    async def _parse_coinbase_message(self, msg):
        """Parse Coinbase WebSocket message."""
        data = json.loads(msg)
        
        if data.get('type') == 'snapshot':
            bids = [[float(price), float(size)] for price, size in data.get('bids', [])]
            asks = [[float(price), float(size)] for price, size in data.get('asks', [])]
            return bids, asks
        elif data.get('type') == 'l2update':
            # This requires maintaining a local order book - simplified here
            # In a real implementation, you'd update the local book with changes
            return None, None
        
        return None, None
    
    async def _coinbase_subscribe(self, websocket):
        """Send subscription message to Coinbase."""
        await websocket.send(json.dumps({
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channels": ["level2"]
        }))
    
    async def connect(self):
        """Connect to WebSocket and start processing messages."""
        self.is_running = True
        logger.info(f"Connecting to WebSocket for {self.symbol} on {self.exchange_id}")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.ws = websocket
                
                # For Coinbase, send subscription message
                if self.exchange_id.lower() == 'coinbasepro':
                    await self._coinbase_subscribe(websocket)
                
                # Process messages
                while self.is_running:
                    message = await websocket.recv()
                    bids, asks = await self.parse_message(message)
                    
                    if bids and asks:
                        # Update the market maker's order book
                        self.market_maker.order_book.update(bids, asks)
                        
                        # Update metrics with new mid price
                        mid_price = self.market_maker.order_book.get_mid_price()
                        self.market_maker.update_pnl_and_metrics(mid_price)
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            self.is_running = False
    
    def stop(self):
        """Stop the WebSocket connection."""
        logger.info("Stopping WebSocket connection")
        self.is_running = False

class BacktestingEngine:
    """Engine for backtesting the market-making strategy."""
    
    def __init__(self, 
                 price_data: pd.DataFrame, 
                 order_flow_data: Optional[pd.DataFrame] = None,
                 symbol: str = "BTC/USD",
                 **mm_params):
        """
        Initialize backtesting engine.
        
        Args:
            price_data: DataFrame with timestamp and market data columns
            order_flow_data: Optional DataFrame with order flow information
            symbol: Trading symbol
            **mm_params: Parameters for AvellanedaStoikovMM
        """
        self.price_data = price_data
        self.order_flow_data = order_flow_data
        self.symbol = symbol
        self.mm_params = mm_params
        
        # Initialize market maker
        self.market_maker = AvellanedaStoikovMM(symbol=symbol, exchange="backtest", **mm_params)
        
        # Results storage
        self.results = None
    
    def generate_simulated_order_book(self, row, timestamp_idx, spread_bps=10, depth=10):
        """Generate a simulated order book or use actual data if available."""
        # Check if we have actual bid/ask data
        if 'bid_price' in row and 'ask_price' in row and pd.notna(row['bid_price']) and pd.notna(row['ask_price']):
            mid_price = (row['bid_price'] + row['ask_price']) / 2
            
            # Use actual top of book for first level
            bids = [[row['bid_price'], 1.0]]
            asks = [[row['ask_price'], 1.0]]
            
            # Generate additional levels
            current_bid = row['bid_price']
            current_ask = row['ask_price']
            
            for i in range(1, depth):
                current_bid -= np.random.exponential(current_bid * 0.0001) + 0.0001
                bids.append([current_bid, np.random.exponential(1.5) + 0.1])
                
                current_ask += np.random.exponential(current_ask * 0.0001) + 0.0001
                asks.append([current_ask, np.random.exponential(1.5) + 0.1])
                
            return bids, asks, mid_price
        else:
            # Fall back to using trade price or provided price
            if 'trade_price' in row and pd.notna(row['trade_price']):
                mid_price = row['trade_price']
            else:
                mid_price = row['price']
                
            half_spread = mid_price * spread_bps / 10000
            
            # Generate bids below mid price
            bids = []
            current_bid = mid_price - half_spread
            for i in range(depth):
                size = np.random.exponential(2) + 0.1
                bids.append([current_bid, size])
                current_bid -= np.random.exponential(half_spread / 3) + 0.0001
            
            # Generate asks above mid price
            asks = []
            current_ask = mid_price + half_spread
            for i in range(depth):
                size = np.random.exponential(2) + 0.1
                asks.append([current_ask, size])
                current_ask += np.random.exponential(half_spread / 3) + 0.0001
            
            return bids, asks, mid_price
    
    
    def simulate_order_arrivals(self, bid_price, ask_price, mid_price, dt):
        """Simulate arrival of market orders that might hit our quotes."""
        # Check if prices are valid
        if bid_price is None or ask_price is None or not np.isfinite(bid_price) or not np.isfinite(ask_price):
            return 0, 0  # No orders if prices are invalid
        
        # Calculate spreads safely
        bid_spread = max(0, mid_price - bid_price)  # Ensure non-negative
        ask_spread = max(0, ask_price - mid_price)  # Ensure non-negative
        
        # Cap spreads to prevent overflow in exp calculation
        MAX_SPREAD = 10  # Reasonable cap for spread calculation
        bid_spread = min(bid_spread, MAX_SPREAD)
        ask_spread = min(ask_spread, MAX_SPREAD)
        
        # Parameters for arrival processes with safe calculation
        lambda_buy = self.market_maker.c * np.exp(-self.market_maker.k * ask_spread)
        lambda_sell = self.market_maker.c * np.exp(-self.market_maker.k * bid_spread)
        
        # Cap lambda values to prevent Poisson overflow
        MAX_LAMBDA = 100  # Safe maximum for Poisson
        lambda_buy = min(lambda_buy * dt, MAX_LAMBDA)
        lambda_sell = min(lambda_sell * dt, MAX_LAMBDA)
        
        # Generate Poisson arrivals
        try:
            bid_hit = np.random.poisson(lambda_sell)
            ask_lift = np.random.poisson(lambda_buy)
        except ValueError:
            # Fallback if we still get an error
            logger.warning(f"Error in Poisson generation. Lambda values too large.")
            bid_hit, ask_lift = 0, 0
        
        # Limit by order size
        bid_hit = min(bid_hit, 1)  # At most 1 hit per step to simplify
        ask_lift = min(ask_lift, 1)  # At most 1 lift per step
        
        return bid_hit, ask_lift
    
    def run_backtest(self):
        """Run the backtest simulation."""
        logger.info(f"Running backtest for {self.symbol} with {len(self.price_data)} data points")
        
        # Reset market maker state
        self.market_maker.cash = self.mm_params.get('initial_cash', 10000.0)
        self.market_maker.inventory = self.mm_params.get('initial_inventory', 0.0)
        self.market_maker.start_time = 0
        
        # Clear history
        self.market_maker.trading_history = []
        self.market_maker.pnl_history = []
        self.market_maker.inventory_history = []
        self.market_maker.quote_history = []
        self.market_maker.mid_price_history = []
        self.market_maker.timestamp_history = []
        
        # Time step in simulation (in days)
        # If we have high-frequency data from Binance (100ms), adjust dt
        if self.price_data.index.freq == '100ms' or self.price_data.index.to_series().diff().median().total_seconds() <= 0.1:
            dt = 1/864000  # 100ms in days (100ms / 86400000ms)
            logger.info("Using high-frequency data mode with 100ms time step")
        else:
            dt = 1/1440  # Default to 1 minute in days
            logger.info(f"Using standard data mode with time step: {dt}")
        
        # Iterate through price data
        for idx, (timestamp, row) in enumerate(self.price_data.iterrows()):
            # Generate or use actual order book data
            bids, asks, mid_price = self.generate_simulated_order_book(row, idx)
            self.market_maker.order_book.update(bids, asks)
            
            # Calculate time elapsed in days
            elapsed_days = idx * dt
            self.market_maker.start_time = -elapsed_days  # Hack to make remaining time calculation work
                        
            # Calculate optimal quotes
            bid_price, ask_price = self.market_maker.calculate_optimal_quotes()
            
            # Update current quotes
            self.market_maker.current_bid_price = bid_price
            self.market_maker.current_ask_price = ask_price
            
            # Simulate market order arrivals
            bid_hit, ask_lift = self.simulate_order_arrivals(bid_price, ask_price, mid_price, dt)
            
            # Process trades
            if bid_hit > 0:
                self.market_maker.record_trade('buy', bid_price, self.market_maker.order_size)
            
            if ask_lift > 0:
                self.market_maker.record_trade('sell', ask_price, self.market_maker.order_size)
            
            # Update metrics with actual mid price
            self.market_maker.update_pnl_and_metrics(mid_price)
        
        
        # Calculate final results
        final_mid_price = self.price_data['price'].iloc[-1]
        cash = self.market_maker.cash
        inventory = self.market_maker.inventory
        final_pnl = cash + inventory * final_mid_price
        initial_cash = self.mm_params.get('initial_cash', 10000.0)
        returns = (final_pnl / initial_cash) - 1
        
        # Store results
        self.results = {
            'final_pnl': final_pnl,
            'returns': returns,
            'final_inventory': inventory,
            'num_trades': len(self.market_maker.trading_history),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'pnl_history': self.market_maker.pnl_history,
            'inventory_history': self.market_maker.inventory_history,
            'mid_price_history': self.market_maker.mid_price_history,
            'timestamp_history': self.market_maker.timestamp_history
        }
        
        logger.info(f"Backtest completed. Final PnL: ${final_pnl:.2f}, Returns: {returns:.2%}")
        return self.results
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from PnL history."""
        if len(self.market_maker.pnl_history) < 2:
            return 0
            
        returns = np.diff(self.market_maker.pnl_history) / np.array(self.market_maker.pnl_history[:-1])
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from PnL history."""
        if not self.market_maker.pnl_history:
            return 0
            
        peak = self.market_maker.pnl_history[0]
        max_dd = 0
        
        for pnl in self.market_maker.pnl_history:
            peak = max(peak, pnl)
            dd = (peak - pnl) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
    
    def plot_results(self):
        """Plot backtest results."""
        if not self.results:
            logger.warning("No backtest results to plot")
            return
            
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'timestamp': self.market_maker.timestamp_history,
            'mid_price': self.market_maker.mid_price_history,
            'pnl': self.market_maker.pnl_history,
            'inventory': self.market_maker.inventory_history
        })
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot price
        axs[0].plot(df['timestamp'], df['mid_price'])
        axs[0].set_title('Mid Price')
        axs[0].set_ylabel('Price')
        axs[0].grid(True)
        
        # Plot inventory
        axs[1].plot(df['timestamp'], df['inventory'])
        axs[1].set_title('Inventory')
        axs[1].set_ylabel('Quantity')
        axs[1].axhline(y=0, color='r', linestyle='-')
        axs[1].grid(True)
        
        # Plot PnL
        axs[2].plot(df['timestamp'], df['pnl'])
        axs[2].set_title('PnL')
        axs[2].set_ylabel('USD')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"Final PnL: ${self.results['final_pnl']:.2f}")
        print(f"Return: {self.results['returns']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Final Inventory: {self.results['final_inventory']:.6f}")
        print(f"Number of Trades: {self.results['num_trades']}")
    
    def parameter_sweep(self, param_grid):
        """
        Perform parameter sweep to find optimal parameters.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
        
        Returns:
            DataFrame with results for each parameter combination
        """
        from itertools import product
        
        # Generate all combinations of parameters
        param_names = list(param_grid.keys())
        param_values = list(product(*param_grid.values()))
        
        results = []
        
        for values in param_values:
            params = dict(zip(param_names, values))
            
            # Update market maker parameters
            for key, value in params.items():
                setattr(self.market_maker, key, value)
                
            # Run backtest with these parameters
            backtest_results = self.run_backtest()
            
            # Store results along with parameters
            result_row = {
                **params,
                'final_pnl': backtest_results['final_pnl'],
                'returns': backtest_results['returns'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'max_drawdown': backtest_results['max_drawdown']
            }
            results.append(result_row)
            
            logger.info(f"Completed parameter sweep for {params}")
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

async def main():
    config = {
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'api_key': 'your_api_key',   # Replace with your API key
        'secret': 'your_secret',     # Replace with your secret
        'sigma': 0.3,                # Volatility
        'gamma': 0.1,                # Risk aversion
        'k': 1.5,                    # Order book liquidity parameter
        'c': 1.0,                    # Base intensity of order arrivals
        'T': 1.0,                    # Time horizon (days)
        'initial_cash': 10000.0,
        'initial_inventory': 0.0,
        'max_inventory': 0.5,        # Max BTC inventory
        'order_size': 0.01,          # 0.01 BTC per order
        'update_interval': 5.0,      # Update every 5 seconds
        'min_spread_pct': 0.001,     # Minimum spread percentage
        'mode': 'backtest',          # 'live' or 'backtest'
        'dashboard_port': 8050,
        'data_source': 'binance',    # 'yfinance', 'binance', or 'csv'
        'data_duration_minutes': 1,  # For Binance real-time data collection
        'csv_path': ''               # For CSV data source
    }
    
    if config['mode'] == 'live':
        # Create live market maker
        market_maker = LiveMarketMaker(
            exchange_id=config['exchange'],
            symbol=config['symbol'],
            api_key=config['api_key'],
            secret=config['secret'],
            sigma=config['sigma'],
            gamma=config['gamma'],
            k=config['k'],
            c=config['c'],
            T=config['T'],
            initial_cash=config['initial_cash'],
            initial_inventory=config['initial_inventory'],
            max_inventory=config['max_inventory'],
            order_size=config['order_size'],
            update_interval=config['update_interval'],
            min_spread_pct=config['min_spread_pct']
        )
        
        # Create and start dashboard
        dashboard = MarketMakerDashboard(market_maker.market_maker)
        dashboard.start_background(port=config['dashboard_port'])
        logger.info(f"Dashboard started at http://localhost:{config['dashboard_port']}")
        
        # Run market maker
        try:
            await market_maker.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            market_maker.stop()
    
    elif config['mode'] == 'backtest':
        # Load market data based on data_source
        if config['data_source'] == 'yfinance':
            # Use yfinance for historical data
            import yfinance as yf
            ticker = "AAPL"  # Or use config['symbol'].split('/')[0]
            data = yf.download(f"{ticker}", period="7d", interval="1m")
            price_data = pd.DataFrame({
                'timestamp': data.index,
                'price': data['Close'].squeeze()
            })
            price_data = price_data.set_index('timestamp')
            
        elif config['data_source'] == 'binance':
            # Use Binance WebSocket for high-frequency data
            logger.info(f"Collecting {config['data_duration_minutes']} minutes of Binance data for {config['symbol']}...")
            symbol = config['symbol'].replace('/', '').lower()
            collector = MarketDataCollector()
            price_data = collector.collect_data(symbol, config['data_duration_minutes'])
            
            # Calculate mid price if not in the dataframe
            if 'price' not in price_data.columns:
                price_data['price'] = price_data[['bid_price', 'ask_price']].mean(axis=1)
            
            logger.info(f"Collected {len(price_data)} data points")
            
        elif config['data_source'] == 'csv':
            # Load from CSV file
            price_data = pd.read_csv(config['csv_path'], parse_dates=['timestamp'], index_col='timestamp')
            
        else:
            logger.error(f"Unknown data source: {config['data_source']}")
            return
        
        # Create and run backtest
        backtest = BacktestingEngine(
            price_data=price_data,
            symbol=config['symbol'],
            sigma=config['sigma'],
            gamma=config['gamma'],
            k=config['k'],
            c=config['c'],
            T=config['T'],
            initial_cash=config['initial_cash'],
            initial_inventory=config['initial_inventory'],
            max_inventory=config['max_inventory'],
            order_size=config['order_size'],
            min_spread_pct=config['min_spread_pct']
        )
        # Run single backtest and plot results
        backtest.run_backtest()
        backtest.plot_results()


if __name__ == "__main__":
    import asyncio
    
    # Run the main function in the asyncio event loop
    asyncio.run(main())