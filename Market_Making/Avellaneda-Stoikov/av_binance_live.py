import json
import time
import threading
import websocket
import pandas as pd
import numpy as np
import os
from datetime import datetime as dt, timedelta
from collections import deque
from tabulate import tabulate
from typing import Dict, List, Tuple, Optional, Any
import traceback

# Configure basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("market_maker_live.log"), logging.StreamHandler()]
)
logger = logging.getLogger("LiveMM")

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

class AvellanedaStoikovMM:
    def __init__(self, 
                 symbol: str,
                 sigma: float = 0.3,    # Volatility of the mid-price process
                 gamma: float = 0.1,    # Risk aversion coefficient
                 k: float = 1.5,        # Order book liquidity parameter
                 c: float = 1.0,        # Intensity of order arrivals
                 T: float = 1.0,        # Time horizon in days
                 initial_cash: float = 10000.0,
                 initial_inventory: float = 50.0,
                 max_inventory: float = 100.0,  # Maximum allowed inventory
                 order_size: float = 0.01,    # Size of each order
                 min_spread_pct: float = 0.001  # Minimum spread as percentage
                 ):
        self.symbol = symbol
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
        self.current_bid_price = None
        self.current_ask_price = None
        self.start_time = time.time()
        
        # Performance tracking
        self.trading_history = []
        self.pnl_history = deque(maxlen=1000)  # Store recent PnL
        self.inventory_history = deque(maxlen=1000)  # Store recent inventory
        self.quote_history = deque(maxlen=100)  # Store recent quotes
        self.mid_price_history = deque(maxlen=1000)  # Store recent mid prices
        self.timestamp_history = deque(maxlen=1000)  # Store recent timestamps
        
        logger.info(f"Initialized market maker for {symbol}")
    
    def optimal_bid_spread(self, inventory: float, remaining_time: float, sigma: float) -> float:
        """Calculate the optimal bid spread using Avellaneda-Stoikov formula."""
        return ((2*inventory + 1) * self.gamma * sigma**2 * remaining_time / 2 + 
                np.log(1 + self.gamma / self.k) / self.gamma)
    
    def optimal_ask_spread(self, inventory: float, remaining_time: float, sigma: float) -> float:
        """Calculate the optimal ask spread using Avellaneda-Stoikov formula."""
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

class LiveMarketMaker:
    def __init__(self, symbol, params=None, max_display_rows=15):
        default_params = {
            'sigma': 0.3,                # Volatility
            'gamma': 0.1,                # Risk aversion
            'k': 1.5,                    # Order book liquidity parameter
            'c': 1.0,                    # Base intensity of order arrivals
            'T': 1.0,                    # Time horizon (days)
            'initial_cash': 1000000.0,
            'initial_inventory': 50.0,
            'max_inventory': 100.0,        # Max inventory
            'order_size': 0.01,          # Size of each order
            'min_spread_pct': 0.001,     # Minimum spread percentage
        }
        
        # Use provided params or defaults
        self.params = default_params if params is None else {**default_params, **params}
        
        # Initialize strategy
        self.symbol = symbol.lower()
        self.market_maker = AvellanedaStoikovMM(
            symbol=symbol,
            **self.params
        )
        
        # Trading state
        self.running = True
        self.max_display_rows = max_display_rows
        self.display_lock = threading.Lock()
        self.recent_trades = deque(maxlen=20)
        self.simulated_trades = []
        self.last_bid_hit = 0
        self.last_ask_lift = 0
        
        # Initialize display thread
        self.display_thread = threading.Thread(target=self._update_display)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        logger.info(f"Initialized live market maker for {symbol}")
        
    def on_order_book_message(self, ws, message):
        try:
            data = json.loads(message)
            
            # Extract bid and ask data
            bids = [[float(price), float(qty)] for price, qty in data.get('b', [])]
            asks = [[float(price), float(qty)] for price, qty in data.get('a', [])]
            
            # Update orderbook
            if bids and asks:
                self.market_maker.order_book.update(bids, asks)
                
                # Calculate mid price and update metrics
                mid_price = self.market_maker.order_book.get_mid_price()
                self.market_maker.update_pnl_and_metrics(mid_price)
                
                # Calculate new quotes
                self._refresh_quotes()
                
                # Simulate trades
                self._simulate_trades()
                
        except Exception as e:
            logger.error(f"Order book processing error: {str(e)}")
            traceback.print_exc()

    def on_trade_message(self, ws, message):
        try:
            data = json.loads(message)
            timestamp = dt.fromtimestamp(data['E']/1000)
            trade_price = float(data['p'])
            trade_size = float(data['q'])
            
            # Store recent trade
            trade_record = {
                'timestamp': timestamp,
                'price': trade_price,
                'size': trade_size,
                'volume': trade_price * trade_size
            }
            self.recent_trades.append(trade_record)
            
        except Exception as e:
            logger.error(f"Trade processing error: {str(e)}")
    
    def _refresh_quotes(self):
        """Calculate new quotes and update the market maker"""
        bid_price, ask_price = self.market_maker.calculate_optimal_quotes()
        
        # Update current quotes
        self.market_maker.current_bid_price = bid_price
        self.market_maker.current_ask_price = ask_price
    
    def _simulate_trades(self):
        """Simulate the arrival of market orders based on the quotes"""
        mid_price = self.market_maker.order_book.get_mid_price()
        bid_price = self.market_maker.current_bid_price
        ask_price = self.market_maker.current_ask_price
        
        # Skip if prices are invalid
        if mid_price == 0 or bid_price == 0 or ask_price == float('inf'):
            return
        
        # Calculate spreads safely
        bid_spread = max(0, mid_price - bid_price)
        ask_spread = max(0, ask_price - mid_price)
        
        # Calculate arrival probabilities
        # Larger spreads → lower probability
        lambda_buy = self.market_maker.c * np.exp(-self.market_maker.k * ask_spread)
        lambda_sell = self.market_maker.c * np.exp(-self.market_maker.k * bid_spread)
        
        # Scale by time step (assuming ~100ms between updates)
        time_scale = 0.1 / 86400  # 100ms in days
        lambda_buy *= time_scale
        lambda_sell *= time_scale
        
        # Generate Poisson arrivals
        try:
            bid_hit = np.random.poisson(lambda_sell)
            ask_lift = np.random.poisson(lambda_buy)
            
            # Limit by order size
            bid_hit = min(bid_hit, 1)  # At most 1 hit per step
            ask_lift = min(ask_lift, 1)  # At most 1 lift per step
            
            # Process trades if they occur
            if bid_hit > 0:
                self.market_maker.record_trade('buy', bid_price, self.market_maker.order_size)
                self.last_bid_hit = time.time()
                self.simulated_trades.append({
                    'timestamp': dt.now(),
                    'side': 'buy',
                    'price': bid_price,
                    'size': self.market_maker.order_size
                })
            
            if ask_lift > 0:
                self.market_maker.record_trade('sell', ask_price, self.market_maker.order_size)
                self.last_ask_lift = time.time()
                self.simulated_trades.append({
                    'timestamp': dt.now(),
                    'side': 'sell',
                    'price': ask_price,
                    'size': self.market_maker.order_size
                })
                
        except Exception as e:
            logger.error(f"Error simulating trades: {str(e)}")
    
    def _run_websocket(self, url, handler, stream_type):
        """Run a websocket connection with reconnection logic"""
        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=handler,
                    on_error=lambda ws, e: logger.error(f"{stream_type} error: {str(e)[:200]}"),
                    on_close=lambda ws: logger.info(f"{stream_type} closed"),
                    on_open=lambda ws: logger.info(f"{stream_type} connected")
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"{stream_type} failure: {str(e)[:200]}")
            
            if self.running:
                logger.info(f"Reconnecting {stream_type} in 5 seconds...")
                time.sleep(5)

    def _update_display(self):
        """Thread to update the terminal display"""
        while self.running:
            try:
                with self.display_lock:
                    self._render_display()
                time.sleep(0.5)  # Update every half second
            except Exception as e:
                logger.error(f"Display error: {str(e)[:200]}")
                time.sleep(2)  # Longer pause on error

    def _render_display(self):
        """Render the terminal display with market making information"""
        os.system('clear')  # Clear terminal
        
        # Print header
        print("\n" + "="*100)
        print(f"  🤖  AVELLANEDA-STOIKOV MARKET MAKER - LIVE SIMULATION  🤖")
        print("="*100)
        
        # Print current market state
        mid_price = self.market_maker.order_book.get_mid_price()
        bid_price = self.market_maker.current_bid_price or "N/A"
        ask_price = self.market_maker.current_ask_price or "N/A"
        
        print(f"\nSymbol: {self.symbol.upper()}")
        print(f"Current Time: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Performance metrics table
        perf_table = [
            ["Mid Price", f"{mid_price:.8f}"],
            ["Inventory", f"{self.market_maker.inventory:.6f}"],
            ["Cash", f"${self.market_maker.cash:.2f}"],
            ["PnL", f"${self.market_maker.pnl_history[-1] if self.market_maker.pnl_history else 0:.2f}"],
            ["Bid Quote", f"{bid_price if bid_price != 'N/A' else 'N/A'}"],
            ["Ask Quote", f"{ask_price if ask_price != 'N/A' else 'N/A'}"]
        ]
        
        print("\nPerformance Metrics:")
        print(tabulate(perf_table, tablefmt="grid"))
        
        # Strategy parameters
        strategy_table = [
            ["Volatility (σ)", f"{self.params['sigma']:.4f}"],
            ["Risk Aversion (γ)", f"{self.params['gamma']:.4f}"],
            ["Market Liquidity (k)", f"{self.params['k']:.4f}"],
            ["Order Intensity (c)", f"{self.params['c']:.4f}"],
            ["Time Horizon", f"{self.params['T']} days"],
            ["Max Inventory", f"{self.params['max_inventory']}"],
            ["Order Size", f"{self.params['order_size']}"]
        ]
        
        print("\nStrategy Parameters:")
        print(tabulate(strategy_table, tablefmt="grid"))
        
        # Recent trades
        if self.market_maker.trading_history:
            print("\nRecent Simulated Trades:")
            trades = self.market_maker.trading_history[-self.max_display_rows:]
            
            trade_table = []
            for trade in trades:
                trade_table.append([
                    trade['timestamp'],
                    trade['side'].upper(),
                    f"{trade['price']:.8f}",
                    f"{trade['size']:.6f}",
                    f"{trade['inventory_after']:.6f}",
                    f"${trade['cash_after']:.2f}"
                ])
                
            headers = ["Timestamp", "Side", "Price", "Size", "Inventory", "Cash"]
            print(tabulate(trade_table, headers=headers, tablefmt="grid"))
        else:
            print("\nNo trades executed yet.")
        
        # PnL over time
        if len(self.market_maker.pnl_history) > 1:
            pnl_change = self.market_maker.pnl_history[-1] - self.market_maker.pnl_history[0]
            pnl_direction = "↑" if pnl_change >= 0 else "↓"
            print(f"\nPnL Trend: {pnl_direction} ${abs(pnl_change):.2f} ({pnl_change/self.market_maker.pnl_history[0]*100:.2f}%)")
        
        print("\nPress Ctrl+C to exit")

    def start(self):
        """Start the live market making simulation"""
        print(f"\nStarting live market making for {self.symbol}...")
        
        threads = [
            threading.Thread(target=self._run_websocket,
                args=(f"wss://stream.binance.com:9443/ws/{self.symbol}@depth@100ms", 
                    self.on_order_book_message, "OrderBook")),
            threading.Thread(target=self._run_websocket,
                args=(f"wss://stream.binance.com:9443/ws/{self.symbol}@trade", 
                    self.on_trade_message, "Trades"))
        ]

        for t in threads:
            t.daemon = True
            t.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down market maker...")
            self.running = False
            time.sleep(1)  # Allow threads to terminate
            print("Market maker stopped")

if __name__ == "__main__":
    params = {
        'sigma': 0.4,                # Volatility
        'gamma': 0.1,                # Risk aversion
        'k': 1.5,                    # Order book liquidity parameter
        'c': 1.0,                    # Base intensity of order arrivals
        'T': 1.0,                    # Time horizon (days)
        'initial_cash': 10000.0,
        'initial_inventory': 0.0,
        'max_inventory': 5.0,        # Max inventory
        'order_size': 0.01,          # Size of each order
        'min_spread_pct': 0.001,     # Minimum spread percentage
    }
    
    symbol = "btcusdt"  # Change to your desired trading pair
    market_maker = LiveMarketMaker(symbol, params)
    
    try:
        market_maker.start()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        traceback.print_exc()