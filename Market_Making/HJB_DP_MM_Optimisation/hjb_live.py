import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import timedelta
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
import random
import websocket
import time
import threading
from collections import deque
import pandas as pd
import warnings
import os
import psutil
try:
    import pynvml
    HAS_GPU_MONITORING = True
except ImportError:
    HAS_GPU_MONITORING = False
    print("GPU monitoring not available. Install pynvml for GPU metrics.")
import plotly.express as px
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
os.environ['NUMBA_CUDA_DRIVER'] = '/usr/lib/x86_64-linux-gnu/libcuda.so'

USE_GPU = True
try:
    import cupy as cp
    from numba import cuda
    from numba import config
    
    # Set explicit architecture target
    config.CUDA_TARGET_COMPUTE_CAPABILITY = (8, 2)
    
    # Test if CUDA is working properly
    @cuda.jit
    def test_kernel(x):
        i = cuda.grid(1)
        if i < x.shape[0]:
            x[i] *= 2
    
    # Try to execute a simple kernel
    test_array = np.ones(1, dtype=np.float32)
    d_test = cuda.to_device(test_array)
    test_kernel[1, 1](d_test)
    cuda.synchronize()
    
    print("CUDA initialization successful!")
    
except (ImportError, RuntimeError, Exception) as e:
    USE_GPU = False
    warnings.warn(f"CUDA initialization failed: {str(e)}. Falling back to CPU implementation.")
    print("Using CPU-based computation instead of GPU. Performance may be slower.")

class ExchangeComparator:
    """
    Fetches and compares price data from multiple exchanges for a given symbol
    """
    def __init__(self, symbol, exchanges=None, max_history=100, update_interval=5.0):
        """
        Initialize exchange comparator
        
        Args:
            symbol: Base trading pair (e.g. 'BTC/USDT')
            exchanges: List of exchange IDs to monitor (default: top 5 by volume)
            max_history: Maximum data points to keep in history
            update_interval: How often to update data in seconds
        """
        self.symbol = self.normalize_symbol(symbol)
        self.max_history = max_history
        self.update_interval = update_interval
        self.exchange_instances = {}
        self.price_data = {}
        self.timestamps = deque(maxlen=max_history)
        self.running = True
        
        # Default exchanges if none provided (top by volume)
        self.exchanges = exchanges or ['binance', 'coinbase', 'kraken', 'kucoin', 'okx']
        
        # Initialize exchange connections
        self._initialize_exchanges()
        
        # Start data collection thread
        self.thread = threading.Thread(target=self._update_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def normalize_symbol(self, symbol):
        """Convert symbol to standard format for comparison across exchanges"""
        # Handle 'btcusdt' format
        if '/' not in symbol:
            # Extract potential base/quote
            if symbol.endswith('usdt'):
                base = symbol[:-4].upper()
                return f"{base}/USDT"
            # Other naming patterns could be handled here
        return symbol.upper()
    
    def _initialize_exchanges(self):
        """Set up connections to all specified exchanges"""
        for exchange_id in self.exchanges:
            try:
                # Initialize the exchange with rate limiting parameters
                exchange_class = getattr(ccxt, exchange_id)
                self.exchange_instances[exchange_id] = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                self.price_data[exchange_id] = deque(maxlen=self.max_history)
                print(f"Initialized connection to {exchange_id}")
            except Exception as e:
                print(f"Error initializing {exchange_id}: {e}")
    
    def _update_thread(self):
        """Background thread to update price data from all exchanges"""
        while self.running:
            timestamp = datetime.now()
            self.timestamps.append(timestamp)
            
            exchange_updates = {}
            
            # Fetch data from each exchange
            for exchange_id, exchange in self.exchange_instances.items():
                try:
                    # Normalize symbol for this specific exchange
                    exchange_symbol = self.symbol
                    if exchange.markets:
                        # Find closest matching symbol
                        available_symbols = list(exchange.markets.keys())
                        base, quote = self.symbol.split('/')
                        
                        # Try different symbol formats
                        candidates = [
                            f"{base}/{quote}",
                            f"{base}{quote}",
                            f"{base}-{quote}",
                            f"{base.lower()}/{quote.lower()}",
                        ]
                        
                        for candidate in candidates:
                            if candidate in available_symbols:
                                exchange_symbol = candidate
                                break
                    
                    # Fetch ticker with bid/ask
                    ticker = exchange.fetch_ticker(exchange_symbol)
                    
                    # Extract and store data
                    data = {
                        'exchange': exchange_id,
                        'bid': ticker.get('bid'),
                        'ask': ticker.get('ask'),
                        'last': ticker.get('last'),
                        'volume': ticker.get('quoteVolume', ticker.get('volume', 0)),
                        'timestamp': timestamp
                    }
                    
                    self.price_data[exchange_id].append(data)
                    exchange_updates[exchange_id] = data
                    
                except Exception as e:
                    print(f"Error fetching data from {exchange_id}: {e}")
            
            # Print comparison summary
            if exchange_updates:
                self._print_comparison(exchange_updates)
            
            # Sleep until next update
            time.sleep(self.update_interval)
    
    def _print_comparison(self, updates):
        """Display a summary of current price comparison"""
        print("\n--- Exchange Price Comparison ---")
        data = []
        for exchange_id, update in updates.items():
            data.append([
                exchange_id.capitalize(),
                f"{update.get('bid', 'N/A'):.2f}" if update.get('bid') else 'N/A',
                f"{update.get('ask', 'N/A'):.2f}" if update.get('ask') else 'N/A',
                f"{update.get('last', 'N/A'):.2f}" if update.get('last') else 'N/A'
            ])
        
        # Calculate potential arbitrage opportunities
        if len(data) > 1:
            bids = [update.get('bid') for update in updates.values() if update.get('bid')]
            asks = [update.get('ask') for update in updates.values() if update.get('ask')]
            
            if bids and asks:
                max_bid = max(bids)
                min_ask = min(asks)
                spread = ((max_bid / min_ask) - 1) * 100 if min_ask > 0 else 0
                
                print(f"Max Bid: {max_bid:.2f} | Min Ask: {min_ask:.2f}")
                print(f"Cross-Exchange Spread: {spread:.4f}%")
                
                if max_bid > min_ask:
                    print(f"⚠️ ARBITRAGE OPPORTUNITY: {spread:.4f}% ⚠️")
        
        # Print as table
        headers = ["Exchange", "Bid", "Ask", "Last"]
        try:
            from tabulate import tabulate
            print(tabulate(data, headers=headers, tablefmt="simple"))
        except ImportError:
            # Fallback if tabulate is not available
            print(" | ".join(headers))
            for row in data:
                print(" | ".join(str(cell) for cell in row))
        
        print("--------------------------------")
    
    def get_current_data(self):
        """Get the most recent data from all exchanges"""
        result = {}
        for exchange_id, history in self.price_data.items():
            if history:
                result[exchange_id] = history[-1]
        return result
    
    def get_data_for_chart(self):
        """Get formatted data suitable for charting"""
        # Create dictionary of bid/ask prices by exchange
        chart_data = {
            'timestamps': list(self.timestamps),
            'exchanges': {}
        }
        
        for exchange_id, history in self.price_data.items():
            if history:
                chart_data['exchanges'][exchange_id] = {
                    'bids': [item.get('bid') for item in history if item.get('bid')],
                    'asks': [item.get('ask') for item in history if item.get('ask')],
                    'last': [item.get('last') for item in history if item.get('last')]
                }
        
        return chart_data
    
    def calculate_arbitrage_opportunities(self):
        """Calculate potential arbitrage opportunities between exchanges"""
        current_data = self.get_current_data()
        
        opportunities = []
        for buy_ex, buy_data in current_data.items():
            for sell_ex, sell_data in current_data.items():
                if buy_ex != sell_ex and buy_data.get('ask') and sell_data.get('bid'):
                    buy_price = buy_data['ask']
                    sell_price = sell_data['bid']
                    profit_pct = ((sell_price / buy_price) - 1) * 100
                    
                    if profit_pct > 0:
                        opportunities.append({
                            'buy_exchange': buy_ex,
                            'sell_exchange': sell_ex,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit_pct': profit_pct
                        })
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
        return opportunities
    
    def stop(self):
        """Stop the data collection thread"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)


@cuda.jit(device=True)
def jump_operator_device(V_next, S, i, j, ds, di, params, d_S):
    jump_term = 0.0
    jump_mean = params[8]
    jump_std = params[9]
    jump_intensity = params[7]
    for m in range(-2, 3):
        jump_size = jump_mean + m*jump_std
        S_jump = S * (1 + jump_size)
        idx = min(max(int((S_jump - d_S[0])/ds), 0), V_next.shape[0]-1)
        jump_term += (1/5) * V_next[idx,j]
    
    # λ(J - I)V
    return params[7] * (jump_term - V_next[i,j])

def hjb_update(V, V_next, S_grid, I_grid, dt, ds, di, params):
    if USE_GPU:
        
        @cuda.jit
        def hjb_kernel(d_V, d_V_next, d_S, d_I, dt, ds, di, params):
            i, j = cuda.grid(2)
            if 1 <= i < d_V.shape[0]-1 and 1 <= j < d_V.shape[1]-1:
                # Check inventory boundaries - enforce constraints
                if j == 0 or j == d_V.shape[1]-1:
                    d_V[i,j] = -1e20  # Numerical approximation of -∞
                    return
                
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
                
                # Calculate derivatives first
                V_S_plus = d_V_next[i+1, j] 
                V_S_minus = d_V_next[i-1, j]
                V_I_plus = d_V_next[i, j+1]
                V_I_minus = d_V_next[i, j-1]
                
                V_S = (V_S_plus - V_S_minus) / (2 * ds)
                V_SS = (V_S_plus - 2 * d_V_next[i, j] + V_S_minus) / (ds**2)
                V_I = (V_I_plus - V_I_minus) / (2 * di)
                
                # Diffusion term from price process - now V_SS is defined
                diffusion = 0.5 * sigma * sigma * S * S * V_SS * dt
                
                # Jump term
                jump_term = jump_operator_device(d_V_next, S, i, j, ds, di, params, d_S) * dt
                
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
                            buy_intensity = max(0.0, dt * (1.0 - (new_bid / best_bid - 1.0) / market_impact))
                            sell_intensity = max(0.0, dt * (1.0 - (new_ask / best_ask - 1.0) / market_impact))
                            
                            # Expected P&L from trades
                            expected_pnl = new_bid * sell_intensity - new_ask * buy_intensity
                            
                            # Inventory risk penalty
                            inventory_cost = kappa * I * I * dt
                            
                            # Don't recalculate diffusion - use the value calculated earlier
                            # diffusion is already set above
                            
                            # Candidate value with jump diffusion
                            V_candidate = d_V_next[i, j] + expected_pnl - inventory_cost + diffusion + jump_term
                            
                            # Update if better
                            if V_candidate > V_optimal:
                                V_optimal = V_candidate
                
                # Update value function
                d_V[i, j] = V_optimal
        # Launch kernel

        return hjb_kernel
    
    else:
        #CPU implementation (simplified but functional)
        for i in range(1, V.shape[0]-1):
            for j in range(1, V.shape[1]-1):
                # Check inventory boundaries
                if j == 0 or j == V.shape[1]-1:
                    V[i,j] = -1e20  # Enforce boundary conditions
                    continue
                
                # Extract current state
                S = S_grid[i]
                I = I_grid[j]
                
                # Extract params
                sigma = params[0]
                kappa = params[1]
                gamma = params[2]
                rho = params[3]
                market_impact = params[4]
                best_bid = params[5]
                best_ask = params[6]
                jump_intensity = params[7] if len(params) > 7 else 0.0
                jump_mean = params[8] if len(params) > 8 else 0.0
                jump_std = params[9] if len(params) > 9 else 0.0
                
                V_S_plus = V_next[i+1, j] 
                V_S_minus = V_next[i-1, j]
                V_I_plus = V_next[i, j+1]
                V_I_minus = V_next[i, j-1]
                
                V_S = (V_S_plus - V_S_minus) / (2 * ds)
                V_SS = (V_S_plus - 2 * V_next[i, j] + V_S_minus) / (ds**2)
                V_I = (V_I_plus - V_I_minus) / (2 * di)
                
                V_optimal = -1e10  # Negative infinity
                
                # Jump term for CPU implementation
                jump_term = 0.0
                if jump_intensity > 0:
                    for m in range(-2, 3):
                        jump_size = jump_mean + m*jump_std
                        S_jump = S * (1 + jump_size)
                        idx = min(max(int((S_jump - S_grid[0])/ds), 0), V_next.shape[0]-1)
                        jump_term += (1/5) * V_next[idx,j]
                    jump_term = jump_intensity * (jump_term - V_next[i,j]) * dt
                
                # Discretized control space for bid/ask adjustments (reduced search space for CPU)
                for bid_idx in range(3):  # -ds to +ds (reduced for CPU)
                    bid_change = (bid_idx - 1) * ds
                    
                    for ask_idx in range(3):  # -ds to +ds (reduced for CPU)
                        ask_change = (ask_idx - 1) * ds
                        
                        new_bid = best_bid + bid_change
                        new_ask = best_ask + ask_change
                        
                        # Valid spread check
                        if new_bid > 0 and new_ask > 0 and new_bid < new_ask:
                            # Order execution intensity model (simplified)
                            buy_intensity = max(0.0, dt * (1.0 - (new_bid / best_bid - 1.0) / market_impact))
                            sell_intensity = max(0.0, dt * (1.0 - (new_ask / best_ask - 1.0) / market_impact))
                            
                            # Expected P&L from trades
                            expected_pnl = new_bid * sell_intensity - new_ask * buy_intensity
                            
                            # Inventory risk penalty
                            inventory_cost = kappa * I * I * dt
                            
                            # Diffusion term from price process
                            diffusion = 0.5 * sigma * sigma * S * S * V_SS * dt
                            
                            # Candidate value with jump diffusion
                            V_candidate = V_next[i, j] + expected_pnl - inventory_cost + diffusion + jump_term
                            
                            # Update if better
                            if V_candidate > V_optimal:
                                V_optimal = V_candidate
                
                # Update value function
                V[i, j] = V_optimal
        
        return V


class ToxicityTracker:
    def __init__(self, window=100):
        """
        Order book toxicity tracker
        
        Args:
            window: Number of trades to consider for toxicity calculation
        """
        self.trade_imbalance = deque(maxlen=window)
        self.spread_history = deque(maxlen=window)
        
    def update_toxicity(self, bid, ask, last_trade):
        """Update toxicity metrics with latest market data"""
        mid = (bid + ask)/2
        direction = 1 if last_trade > mid else -1
        self.trade_imbalance.append(direction)
        self.spread_history.append(ask - bid)
        
    @property
    def toxicity(self):
        """
        Calculate order book toxicity score
        
        Returns:
            float: Toxicity score from -1.0 to 1.0
        """
        if len(self.trade_imbalance) < 10:
            return 0.0
        imbalance = np.mean(self.trade_imbalance)
        spread = np.mean(self.spread_history)
        # Toxicity increases with order imbalance and decreases with spread
        return np.clip(imbalance * (1/spread), -1.0, 1.0)

class ResourceMonitor:
    """Monitor system resources like CPU and GPU usage"""
    def __init__(self, history_length=100):
        self.history_length = history_length
        self.timestamps = deque(maxlen=history_length)
        self.cpu_usage = deque(maxlen=history_length)
        self.memory_usage = deque(maxlen=history_length)
        
        # GPU monitoring
        self.has_gpu = HAS_GPU_MONITORING and USE_GPU
        self.gpu_usage = deque(maxlen=history_length) if self.has_gpu else None
        self.gpu_memory = deque(maxlen=history_length) if self.has_gpu else None
        
        if self.has_gpu:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                if self.device_count > 0:
                    self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                else:
                    self.has_gpu = False
            except Exception as e:
                print(f"Error initializing GPU monitoring: {e}")
                self.has_gpu = False
    
    def update(self):
        """Update resource usage metrics"""
        now = datetime.now()
        self.timestamps.append(now)
        
        # CPU metrics
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
        
        # GPU metrics if available
        if self.has_gpu:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.gpu_usage.append(utilization.gpu)
                
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                memory_percent = (memory_info.used / memory_info.total) * 100
                self.gpu_memory.append(memory_percent)
            except Exception as e:
                print(f"Error updating GPU metrics: {e}")
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
    
    def cleanup(self):
        """Clean up GPU monitoring resources"""
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            
class HJBSolver:    
    def __init__(self, S_min, S_max, I_min, I_max, N_S=101, N_I=101, 
                 sigma=0.2, kappa=0.001, gamma=0.0001, rho=0.01, market_impact=0.0001,
                 jump_intensity=0.1, jump_mean=0.0, jump_std=0.01):
        # Grid parameters
        self.S_grid = np.linspace(S_min, S_max, N_S)
        self.I_grid = np.linspace(I_min, I_max, N_I)
        self.ds = (S_max - S_min) / (N_S - 1)
        self.di = (I_max - I_min) / (N_I - 1)
        
        # Initialize toxicity tracker
        self.toxicity_tracker = ToxicityTracker(window=100)
        
        # Model parameters (extended with jump diffusion parameters)
        self.params = np.array([
            sigma, kappa, gamma, rho, market_impact, 
            0.0, 0.0,  # bid/ask placeholders
            jump_intensity, jump_mean, jump_std
        ], dtype=np.float32)
        
        # Initialize value function
        self.V = np.zeros((N_S, N_I))
        self.V_next = np.zeros((N_S, N_I))
        
        if USE_GPU:
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
            
            # Get the kernel function
            self.hjb_kernel = hjb_update(self.V, self.V_next, self.S_grid, self.I_grid, 0.001, self.ds, self.di, self.params)
        
    def update(self, bid_price, ask_price, last_trade=None, dt=0.001):
        """Update value function for one time step"""
        # Add small random noise to ensure changes are visible
        noise = np.random.normal(0, 0.0001)
        self.params[5] = bid_price * (1 + noise)
        self.params[6] = ask_price * (1 + noise)
        
        # Update toxicity if we have trade data
        if last_trade is not None:
            self.toxicity_tracker.update_toxicity(bid_price, ask_price, last_trade)
            # Adjust market impact based on toxicity
            toxicity = self.toxicity_tracker.toxicity
            self.params[4] = self.params[4] * (1 + 2*toxicity)  # Impact scaling
        
        if USE_GPU:
            # GPU implementation
            try:
                self.d_params = cuda.to_device(self.params)
                
                # Run HJB kernel
                self.hjb_kernel[self.blockspergrid, self.threadsperblock](
                    self.d_V, self.d_V_next, self.d_S, self.d_I, 
                    dt, self.ds, self.di, self.d_params
                )
                
                # Swap buffers
                self.d_V, self.d_V_next = self.d_V_next, self.d_V
                
                # Copy back results
                cuda.synchronize()
                self.d_V.copy_to_host(self.V)
            except Exception as e:
                print(f"GPU computation failed: {str(e)}. Falling back to CPU.")
                # If GPU fails, fall back to CPU implementation for this update
                hjb_update(self.V, self.V_next, self.S_grid, self.I_grid, dt, self.ds, self.di, self.params)
                # Swap buffers
                self.V, self.V_next = self.V_next.copy(), self.V.copy()
        else:
            # CPU implementation
            hjb_update(self.V, self.V_next, self.S_grid, self.I_grid, dt, self.ds, self.di, self.params)
            # Swap buffers
            self.V, self.V_next = self.V_next.copy(), self.V.copy()
        
    def get_optimal_quotes(self, current_price, inventory):
        """Get optimal bid/ask quotes for current state with enhanced exploration"""
        s_idx = np.argmin(np.abs(self.S_grid - current_price))
        i_idx = np.argmin(np.abs(self.I_grid - inventory))
        
        # Increased exploration factor for more visible changes
        exploration_factor = 0.001
        optimal_bid_change = 0
        optimal_ask_change = 0
        max_value = -float('inf')
        
        # Wider range for more visible changes
        for bid_change in np.arange(-3*self.ds, 3*self.ds + self.ds, self.ds):
            for ask_change in np.arange(-3*self.ds, 3*self.ds + self.ds, self.ds):
                # Lookup neighbor value in grid
                s_offset = int(round(bid_change / self.ds))
                i_offset = int(round(ask_change / self.ds))
                
                if 0 <= s_idx + s_offset < len(self.S_grid) and 0 <= i_idx + i_offset < len(self.I_grid):
                    # Add small random noise to encourage exploration
                    value = self.V[s_idx + s_offset, i_idx + i_offset] + np.random.normal(0, exploration_factor)
                    
                    if value > max_value:
                        max_value = value
                        optimal_bid_change = bid_change
                        optimal_ask_change = ask_change
        
        # Apply to current market prices with increased randomness for visibility
        optimal_bid = self.params[5] + optimal_bid_change + np.random.normal(0, self.ds * 0.1)
        optimal_ask = self.params[6] + optimal_ask_change + np.random.normal(0, self.ds * 0.1)
        
        # Ensure spread is valid
        if optimal_bid >= optimal_ask:
            optimal_ask = optimal_bid * 1.001  # Ensure minimum spread
            
        return optimal_bid, optimal_ask

    def process_potential_executions(optimal_bid, optimal_ask, last_trade_price, portfolio):
        """Execute trades based on whether our quotes would have been hit by market trades"""
        if last_trade_price is None:
            return False
        
        trade_executed = False
        
        # If trade price is below or at our bid, we would have bought
        if last_trade_price <= optimal_bid:
            size = random.randint(1, 5)  # Still randomize size for simulation
            execution_price = optimal_bid  # We execute at our quoted price
            
            # Execute trade through portfolio tracker
            trade = portfolio.execute_trade(size, execution_price, 'BUY')
            current_inventory = portfolio.inventory
            
            print(f"BUY EXECUTED: {size} @ {execution_price:.4f} (Market traded at {last_trade_price:.4f})")
            return {'trade': trade, 'executed': True, 'type': 'BUY'}
            
        # If trade price is above or at our ask, we would have sold
        elif last_trade_price >= optimal_ask:
            size = random.randint(1, 5)  # Still randomize size for simulation
            execution_price = optimal_ask  # We execute at our quoted price
            
            # Execute trade through portfolio tracker
            trade = portfolio.execute_trade(size, execution_price, 'SELL')
            current_inventory = portfolio.inventory
            
            print(f"SELL EXECUTED: {size} @ {execution_price:.4f} (Market traded at {last_trade_price:.4f})")
            return {'trade': trade, 'executed': True, 'type': 'SELL'}
        
        return {'executed': False}


def hjb_shared_memory_kernel(V, V_next, S_grid, I_grid, dt, ds, di, params):
    """GPU implementation with shared memory tiling"""
    @cuda.jit
    def hjb_kernel_sm(d_V, d_V_next, d_S, d_I, dt, ds, di, params):
        # Shared memory allocation for the tile
        tile_dim_x = 32
        tile_dim_y = 32
        shared_V = cuda.shared.array(shape=(34, 34), dtype=np.float32)  # 32x32 tile + halo
        
        # Global position
        i, j = cuda.grid(2)
        
        # Local thread index
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        # Local position in shared memory (add 1 for halo)
        li = tx + 1
        lj = ty + 1
        
        # Load main tile data into shared memory
        if i < d_V.shape[0] and j < d_V.shape[1]:
            shared_V[li, lj] = d_V_next[i, j]
        else:
            shared_V[li, lj] = 0.0
        
        # Load halo regions
        if tx < 1 and i > 0:  # Left halo
            shared_V[li-1, lj] = d_V_next[i-1, j]
        if tx >= tile_dim_x-1 and i < d_V.shape[0]-1:  # Right halo
            shared_V[li+1, lj] = d_V_next[i+1, j]
        if ty < 1 and j > 0:  # Top halo
            shared_V[li, lj-1] = d_V_next[i, j-1]
        if ty >= tile_dim_y-1 and j < d_V.shape[1]-1:  # Bottom halo
            shared_V[li, lj+1] = d_V_next[i, j+1]
        
        # Ensure all threads finish loading shared memory
        cuda.syncthreads()
        
        # Only compute for valid grid points
        if 1 <= i < d_V.shape[0]-1 and 1 <= j < d_V.shape[1]-1:
            # Check inventory boundaries - enforce constraints
            if j == 0 or j == d_V.shape[1]-1:
                d_V[i,j] = -1e20  # Numerical approximation of -∞
                return
            
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
            jump_intensity = params[7]
            jump_mean = params[8]
            jump_std = params[9]
            
            # Use shared memory for derivatives
            V_S_plus = shared_V[li+1, lj] 
            V_S_minus = shared_V[li-1, lj]
            V_I_plus = shared_V[li, lj+1]
            V_I_minus = shared_V[li, lj-1]
            
            V_S = (V_S_plus - V_S_minus) / (2 * ds)
            V_SS = (V_S_plus - 2 * shared_V[li, lj] + V_S_minus) / (ds**2)
            V_I = (V_I_plus - V_I_minus) / (2 * di)
            
            V_optimal = -1e10  # Negative infinity
            
            # Compute jump term from global memory (can't fit jumps in shared)
            jump_term = 0.0
            if jump_intensity > 0:
                for m in range(-2, 3):
                    jump_size = jump_mean + m*jump_std
                    S_jump = S * (1 + jump_size)
                    idx = min(max(int((S_jump - d_S[0])/ds), 0), d_V_next.shape[0]-1)
                    jump_term += (1/5) * d_V_next[idx,j]
                jump_term = jump_intensity * (jump_term - d_V_next[i,j]) * dt
            
            # Discretized control space for bid/ask adjustments
            for bid_idx in range(5):  # -2*ds to +2*ds
                bid_change = (bid_idx - 2) * ds
                
                for ask_idx in range(5):  # -2*ds to +2*ds
                    ask_change = (ask_idx - 2) * ds
                    
                    new_bid = best_bid + bid_change
                    new_ask = best_ask + ask_change
                    
                    if new_bid > 0 and new_ask > 0 and new_bid < new_ask:
                        # Order execution intensity model (simplified)
                        buy_intensity = max(0.0, dt * (1.0 - (new_bid / best_bid - 1.0) / market_impact))
                        sell_intensity = max(0.0, dt * (1.0 - (new_ask / best_ask - 1.0) / market_impact))
                        
                        # Expected P&L from trades
                        expected_pnl = new_bid * sell_intensity - new_ask * buy_intensity
                        
                        # Inventory risk penalty
                        inventory_cost = kappa * I * I * dt
                        
                        # Diffusion term from price process
                        diffusion = 0.5 * sigma * sigma * S * S * V_SS * dt
                        
                        # Candidate value with jump diffusion
                        V_candidate = shared_V[li, lj] + expected_pnl - inventory_cost + diffusion + jump_term
                        
                        # Update if better
                        if V_candidate > V_optimal:
                            V_optimal = V_candidate
            
            # Update value function
            d_V[i, j] = V_optimal
    
    return hjb_kernel_sm

def validate_jump_model(S_min=90, S_max=110, N_S=101, N_I=21, iterations=1000):
    """
    Test Merton jump diffusion convergence
    
    Returns:
        tuple: (S_grid, option_values)
    """
    import matplotlib.pyplot as plt
    
    S_grid = np.linspace(S_min, S_max, N_S)
    I_grid = np.linspace(-10, 10, N_I)
    ds = (S_max - S_min) / (N_S - 1)
    di = 20 / (N_I - 1)
    
    # Initial condition: call option payoff at maturity
    K = 100  # Strike price
    V = np.zeros((N_S, N_I))
    for i in range(N_S):
        for j in range(N_I):
            V[i, j] = max(S_grid[i] - K, 0)
    
    V_next = V.copy()
    
    # Set up parameters
    dt = 0.001
    sigma = 0.2
    kappa = 0.001
    gamma = 0.0001
    rho = 0.01
    market_impact = 0.0001
    best_bid = 100
    best_ask = 101
    # Jump parameters
    jump_intensity = 0.5  
    jump_mean = 0.0
    jump_std = 0.02
    
    params = np.array([
        sigma, kappa, gamma, rho, market_impact, 
        best_bid, best_ask, jump_intensity, jump_mean, jump_std
    ], dtype=np.float32)
    
    # Run iterations
    print("Running jump diffusion validation...")
    for i in range(iterations):
        if i % 100 == 0:
            print(f"Iteration {i}/{iterations}")
        V = hjb_update(V, V_next, S_grid, I_grid, dt, ds, di, params)
        V, V_next = V_next.copy(), V.copy()
    
    # Extract mid-index for I dimension to get option value function
    option_values = V[:, N_I // 2]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(S_grid, option_values)
    plt.title('Option Values with Jump Diffusion')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    plt.grid(True)
    plt.savefig('jump_diffusion_validation.png')
    
    print("Jump diffusion validation complete. See jump_diffusion_validation.png")
    return S_grid, option_values

def get_user_ticker_choice():
    """Prompt user to choose between manual ticker selection or hot ticker scanner"""
    print("\n" + "="*60)
    print("  FRANKLINE & CO. HJB OPTIMAL MARKET MAKING STRATEGY")
    print("="*60)
    print("\n1. Select a specific ticker for market making")
    print("2. Use the hot ticker scanner to automatically select active markets")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == '1':
            ticker = input("\nEnter ticker symbol (e.g., btcusdt): ").strip().lower()
            if ticker:
                return ticker, False  # Return ticker and flag indicating manual selection
            else:
                print("Invalid ticker. Please try again.")
        elif choice == '2':
            return None, True  # Return None and flag indicating scanner usage
        else:
            print("Invalid choice. Please enter 1 or 2.")

def profile_performance(solver, bid_price, ask_price, iterations=1000):
    """
    Profile the performance of the HJB solver
    
    Args:
        solver: HJBSolver instance
        bid_price: Current bid price
        ask_price: Current ask price
        iterations: Number of iterations to run
    """
    try:
        from nvidia import nsight
        with nsight.Profile() as prof:
            start_time = time.time()
            for _ in range(iterations):
                solver.update(bid_price, ask_price)
            end_time = time.time()
            
        print(f"Performance profiling: {iterations} iterations in {end_time - start_time:.4f} seconds")
        print(f"Average time per iteration: {(end_time - start_time) / iterations * 1000:.4f} ms")
        print(prof.report())
    except ImportError:
        # Fallback if nsight is not available
        start_time = time.time()
        for _ in range(iterations):
            solver.update(bid_price, ask_price)
        end_time = time.time()
        
        print(f"Performance profiling: {iterations} iterations in {end_time - start_time:.4f} seconds")
        print(f"Average time per iteration: {(end_time - start_time) / iterations * 1000:.4f} ms")
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

class PortfolioTracker:
    """
    Tracks portfolio metrics including cash, inventory, and P&L
    """
    def __init__(self, initial_cash=100000.0, initial_inventory=0.0, symbol=""):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.initial_inventory = initial_inventory
        self.inventory = initial_inventory
        self.trades = []
        self.last_price = None
        
        # Metrics
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.portfolio_value = initial_cash
        
    def update_price(self, price):
        """Update the last known price and recalculate unrealized P&L"""
        if price is not None:
            self.last_price = price
            self._calculate_metrics()
    
    def execute_trade(self, size, price, trade_type):
        """
        Record a trade and update portfolio metrics
        
        Args:
            size: Quantity traded (positive for buys, negative for sells)
            price: Execution price
            trade_type: 'BUY' or 'SELL'
        """
        # Record the trade
        trade = {
            'type': trade_type,
            'size': abs(size),  # Store absolute size for clarity
            'price': price,
            'timestamp': datetime.now(),
            'symbol': self.symbol
        }
        self.trades.append(trade)
        
        # Update inventory and cash
        if trade_type == 'BUY':
            self.cash -= price * size
            self.inventory += size
        else:  # SELL
            self.cash += price * size
            self.inventory -= size
        
        # Recalculate metrics
        self._calculate_metrics()
        
        return trade
    
    def _calculate_metrics(self):
        """Calculate current portfolio metrics"""
        if self.last_price is None:
            return
            
        # Calculate unrealized P&L
        inventory_value = self.inventory * self.last_price
        
        # Calculate realized P&L based on cash flow changes
        self.realized_pnl = self.cash - self.initial_cash + \
                           (self.initial_inventory - self.inventory) * self.last_price
                           
        # Unrealized P&L is the market value of the current position
        # minus the initial inventory value (adjusted for any realized trades)
        self.unrealized_pnl = inventory_value - (self.initial_inventory * self.last_price)
        
        # Total P&L
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Current portfolio value
        self.portfolio_value = self.cash + inventory_value
    
    def get_metrics(self):
        """Get current portfolio metrics as a dictionary"""
        return {
            'cash': self.cash,
            'inventory': self.inventory,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'portfolio_value': self.portfolio_value,
            'symbol': self.symbol
        }

def get_user_portfolio_setup():
    """Prompt user to set up initial portfolio parameters"""
    print("\nINITIAL PORTFOLIO SETUP")
    print("-----------------------")
    
    # Get starting cash
    while True:
        try:
            cash_input = input("Enter starting cash (default: 100000): ").strip()
            if not cash_input:
                initial_cash = 100000.0
                break
            initial_cash = float(cash_input)
            if initial_cash < 0:
                print("Cash must be greater than or equal to 0. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get starting inventory
    while True:
        try:
            inventory_input = input("Enter starting inventory (default: 0): ").strip()
            if not inventory_input:
                initial_inventory = 0.0
                break
            initial_inventory = float(inventory_input)
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\nPortfolio initialized with ${initial_cash:.2f} cash and {initial_inventory:.4f} units of inventory.\n")
    return initial_cash, initial_inventory


class TradingDashboard:
    def __init__(self, update_interval=1000):
        self.strategy_state = {
            'bid': 0,
            'ask': 0,
            'inventory': 0,
            'cash': 0,
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'total_pnl': 0,
            'portfolio_value': 0,
            'pnl': 0,  # Added to ensure compatibility with older code
            'symbol': '',
            'initial_cash': 0
        }
        
        self.update_interval = update_interval  # in milliseconds
        
        # Use deques with maxlen for efficient data storage
        self.max_points = 100  # Maximum number of points to display
        self.timestamps = deque(maxlen=self.max_points)
        self.mid_prices = deque(maxlen=self.max_points)
        self.bid_prices = deque(maxlen=self.max_points)
        self.ask_prices = deque(maxlen=self.max_points)
        self.inventory_history = deque(maxlen=self.max_points)
        self.pnl_history = deque(maxlen=self.max_points)
        
        # Add multi-exchange comparison data
        self.exchange_data = {}
        self.exchange_timestamps = deque(maxlen=self.max_points)
        
        # Add quote performance tracking
        self.market_trades = deque(maxlen=self.max_points)
        self.quote_performance = deque(maxlen=self.max_points)  # % difference between mid quote and trade
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(history_length=self.max_points)
        
        # Initialize Dash app with enhanced styling
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[
                'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap'
            ]
        )
    
        self.app.layout = html.Div([
        html.Div([
            html.Div([
                html.H1("FRANKLINE & CO", style={'margin': '0', 'color': '#7DF9FF', 'fontWeight': '700'}),
                html.H3("HJB Optimal Market Making", style={'margin': '0', 'color': '#E6E6E6'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}),
            
            html.Div([
                html.Div([
                    html.Span("Status: ", style={'color': '#E6E6E6'}),
                    html.Span("ACTIVE", id='status-indicator', 
                            style={'color': '#00FF00', 'fontWeight': 'bold', 'marginRight': '20px'})
                ]),
                html.H3(id='symbol-header', children="Symbol: -", 
                        style={'color': '#7DF9FF', 'margin': '0'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={
            'display': 'flex', 
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'padding': '20px',
            'backgroundColor': '#1A1A1A',
            'borderBottom': '2px solid #333',
            'borderRadius': '8px 8px 0 0',
        }),
        
        # Main dashboard content in a grid layout
        html.Div([
            # Left column (60% width) - Main charts
            html.Div([
                # Market price chart (60% height of left column)
                html.Div([
                    html.Div([
                        html.H3("Market Prices & Quotes", 
                                style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                        html.Div(id='price-metrics', style={'fontSize': '13px', 'color': '#999'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                    dcc.Graph(id='price-chart', style={'height': '100%'}, config={'displayModeBar': False}),
                ], style={
                    'height': '40%',
                    'backgroundColor': '#222',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'marginBottom': '15px',
                    'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                }),
                
                # Quote performance chart
                html.Div([
                    html.Div([
                        html.H3("Quote Performance", 
                                style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                        html.Div(id='quote-metrics', style={'fontSize': '13px', 'color': '#999'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                    dcc.Graph(id='quote-performance-chart', style={'height': '100%'}, config={'displayModeBar': False}),
                ], style={
                    'height': '20%',
                    'backgroundColor': '#222',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'marginBottom': '15px',
                    'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                }),
                
                # Exchange comparison chart
                html.Div([
                    html.Div([
                        html.H3("Multi-Exchange Comparison", 
                                style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                        html.Div(id='exchange-metrics', style={'fontSize': '13px', 'color': '#999'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                    dcc.Graph(id='exchange-comparison-chart', style={'height': '100%'}, config={'displayModeBar': False}),
                ], style={
                    'height': '20%',
                    'backgroundColor': '#222',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'marginBottom': '15px',
                    'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                }),
                
                # Position and PnL in horizontal layout (40% height of left column)
                html.Div([
                    # Position chart
                    html.Div([
                        html.Div([
                            html.H3("Position/Inventory", 
                                    style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                            html.Div(id='inventory-metrics', style={'fontSize': '13px', 'color': '#999'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                        dcc.Graph(id='position-chart', style={'height': '100%'}, config={'displayModeBar': False}),
                    ], style={
                        'width': '50%',
                        'height': '100%',
                        'display': 'inline-block',
                        'backgroundColor': '#222',
                        'borderRadius': '8px',
                        'padding': '15px',
                        'boxSizing': 'border-box',
                        'marginRight': '10px',
                        'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                    }),
                    
                    # PnL chart
                    html.Div([
                        html.Div([
                            html.H3("P&L Performance", 
                                    style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                            html.Div(id='pnl-metrics', style={'fontSize': '13px', 'color': '#999'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                        dcc.Graph(id='pnl-chart', style={'height': '100%'}, config={'displayModeBar': False}),
                    ], style={
                        'width': 'calc(50% - 10px)',
                        'height': '100%',
                        'float': 'right',
                        'display': 'inline-block',
                        'backgroundColor': '#222',
                        'borderRadius': '8px',
                        'padding': '15px',
                        'boxSizing': 'border-box',
                        'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                    }),
                ], style={'height': '20%', 'display': 'flex'}),
            ], style={'width': '60%', 'display': 'inline-block', 'height': '100%', 'paddingRight': '15px', 'boxSizing': 'border-box'}),
            
            # Right column (40% width) - Stats and resources
            html.Div([
                # KPI Cards in grid layout
                html.Div([
                    # Row 1: Primary KPIs
                    html.Div([
                        # Card 1: Portfolio Value
                        html.Div([
                            html.Div("Portfolio Value", style={'fontSize': '14px', 'color': '#999'}),
                            html.Div(id='portfolio-value', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#7DF9FF', 'marginTop': '5px'}),
                            html.Div(id='portfolio-change', style={'fontSize': '13px', 'color': '#32CD32', 'marginTop': '5px'})
                        ], style={
                            'width': 'calc(50% - 7px)',
                            'backgroundColor': '#222',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxSizing': 'border-box',
                            'marginRight': '14px',
                            'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                        }),
                        
                        # Card 2: Total P&L
                        html.Div([
                            html.Div("Total P&L", style={'fontSize': '14px', 'color': '#999'}),
                            html.Div(id='total-pnl', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#32CD32', 'marginTop': '5px'}),
                            html.Div(id='pnl-change', style={'fontSize': '13px', 'color': '#32CD32', 'marginTop': '5px'})
                        ], style={
                            'width': 'calc(50% - 7px)',
                            'backgroundColor': '#222',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxSizing': 'border-box',
                            'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                        }),
                    ], style={'display': 'flex', 'marginBottom': '15px'}),
                    
                    # Row 2: Secondary KPIs
                    html.Div([
                        # Card 3: Current Position
                        html.Div([
                            html.Div("Current Position", style={'fontSize': '14px', 'color': '#999'}),
                            html.Div(id='current-position', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#E6E6E6', 'marginTop': '5px'}),
                            html.Div(id='position-value', style={'fontSize': '13px', 'color': '#999', 'marginTop': '5px'})
                        ], style={
                            'width': 'calc(50% - 7px)',
                            'backgroundColor': '#222',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxSizing': 'border-box',
                            'marginRight': '14px',
                            'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                        }),
                        
                        # Card 4: Cash Balance
                        html.Div([
                            html.Div("Cash Balance", style={'fontSize': '14px', 'color': '#999'}),
                            html.Div(id='cash-balance', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#E6E6E6', 'marginTop': '5px'}),
                            html.Div(id='cash-change', style={'fontSize': '13px', 'color': '#999', 'marginTop': '5px'})
                        ], style={
                            'width': 'calc(50% - 7px)',
                            'backgroundColor': '#222',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxSizing': 'border-box',
                            'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                        }),
                    ], style={'display': 'flex', 'marginBottom': '15px'}),
                ], style={'marginBottom': '15px'}),
                
                # Strategy metrics table
                html.Div([
                    html.Div([
                        html.H3("Strategy Metrics", 
                                style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                        html.Div(id='strategy-updated', style={'fontSize': '13px', 'color': '#999'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                    html.Div(id='current-stats', style={'color': '#E6E6E6', 'height': 'calc(100% - 30px)', 'overflowY': 'auto'})
                ], style={
                    'height': '25%',
                    'backgroundColor': '#222',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'marginBottom': '15px',
                    'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                }),
                
                # Exchange comparison table
                html.Div([
                    html.Div([
                        html.H3("Exchange Arbitrage", 
                                style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                        html.Div(id='arbitrage-updated', style={'fontSize': '13px', 'color': '#999'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                    html.Div(id='exchange-comparison-table', style={'color': '#E6E6E6', 'height': 'calc(100% - 30px)', 'overflowY': 'auto'})
                ], style={
                    'height': '30%',
                    'backgroundColor': '#222',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'marginBottom': '15px',
                    'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                }),
                
                # System resources chart
                html.Div([
                    html.Div([
                        html.H3("System Performance", 
                                style={'margin': '0', 'color': '#CCCCCC', 'fontSize': '16px'}),
                        html.Div(id='system-metrics', style={'fontSize': '13px', 'color': '#999'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
                    dcc.Graph(id='resource-chart', style={'height': 'calc(100% - 30px)'}, config={'displayModeBar': False}),
                ], style={
                    'height': 'calc(45% - 15px)',
                    'backgroundColor': '#222',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.2)'
                }),
            ], style={'width': '40%', 'float': 'right', 'display': 'inline-block', 'height': '100%', 'boxSizing': 'border-box'}),
        ], style={
            'padding': '15px',
            'backgroundColor': '#1A1A1A',
            'height': 'calc(100vh - 80px)',  # Adjust for header height
            'borderRadius': '0 0 8px 8px'
        }),
        
        dcc.Interval(
            id='interval-component',
            interval=self.update_interval,
            n_intervals=0
        ),
        ], style={
            'fontFamily': 'Roboto, sans-serif',
            'backgroundColor': '#121212',
            'margin': '20px auto',
            'maxWidth': '1600px',
            'height': 'calc(100vh - 40px)',  # Margin top + bottom
            'boxShadow': '0px 5px 15px rgba(0, 0, 0, 0.5)',
            'borderRadius': '8px'
        })

            # Define callbacks with enhanced visualizations
        @self.app.callback(
            [Output('price-chart', 'figure'),
            Output('position-chart', 'figure'),
            Output('pnl-chart', 'figure'),
            Output('resource-chart', 'figure'),
            Output('current-stats', 'children'),
            Output('symbol-header', 'children'),
            Output('quote-performance-chart', 'figure'),
            Output('exchange-comparison-chart', 'figure'),
            Output('exchange-comparison-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
            

        def update_graphs(n):
            # Update resource monitor
            self.resource_monitor.update()
            
            # Time series for x-axis (convert to readable format)
            x_data = [t.strftime('%H:%M:%S') for t in self.timestamps] if self.timestamps else []
            
            # Enhanced price chart with bid/ask quotes
            price_fig = go.Figure()
            if self.mid_prices:
                price_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.mid_prices), 
                    name='Mid Price', 
                    line=dict(color='#7DF9FF', width=2)
                ))
                price_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.bid_prices), 
                    name='Bid Quote', 
                    line=dict(color='#32CD32', width=1.5)
                ))
                price_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.ask_prices), 
                    name='Ask Quote', 
                    line=dict(color='#FF6347', width=1.5)
                ))
                
                # Add spread area
                price_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.ask_prices),
                    fill='tonexty', 
                    fillcolor='rgba(255, 99, 71, 0.1)',
                    line=dict(width=0),
                    showlegend=False
                ))
            
            price_fig.update_layout(
                title=None,
                xaxis_title=None,
                yaxis_title="Price",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=20, b=40),
                hovermode="x unified",
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#2A2A2A',
                font=dict(color='#E6E6E6')
            )
            
            # Enhanced position/inventory chart with gradient fill
            position_fig = go.Figure()
            if self.inventory_history:
                position_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.inventory_history), 
                    name='Inventory', 
                    fill='tozeroy', 
                    fillcolor='rgba(255, 215, 0, 0.2)',
                    line=dict(color='#FFD700', width=2)
                ))
                
                # Add zero line
                position_fig.add_shape(
                    type="line", line=dict(color="#777", width=1, dash="dot"),
                    x0=0, y0=0, x1=1, y1=0,
                    xref="paper", yref="y"
                )
            
            position_fig.update_layout(
                title=None,
                xaxis_title=None,
                yaxis_title="Units",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=20, b=40),
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#2A2A2A',
                font=dict(color='#E6E6E6')
            )
            
            # Enhanced PnL chart
            pnl_fig = go.Figure()
            if self.pnl_history:
                # Determine color gradient based on positive/negative PnL
                colors = ['#FF6347' if p < 0 else '#32CD32' for p in self.pnl_history]
                pnl_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.pnl_history), 
                    name='P&L', 
                    line=dict(color='#9370DB', width=2),
                    fill='tozeroy', 
                    fillcolor='rgba(147, 112, 219, 0.2)'
                ))
                
                # Add zero line
                pnl_fig.add_shape(
                    type="line", line=dict(color="#777", width=1, dash="dot"),
                    x0=0, y0=0, x1=1, y1=0,
                    xref="paper", yref="y"
                )
            
            pnl_fig.update_layout(
                title=None,
                xaxis_title=None,
                yaxis_title="P&L",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=20, b=40),
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#2A2A2A',
                font=dict(color='#E6E6E6')
            )
            
            # Resource usage chart
            resource_fig = go.Figure()
            
            # Only add traces if we have data
            if self.resource_monitor.timestamps:
                resource_x = [t.strftime('%H:%M:%S') for t in self.resource_monitor.timestamps]
                
                # CPU Usage
                resource_fig.add_trace(go.Scatter(
                    x=resource_x, 
                    y=list(self.resource_monitor.cpu_usage),
                    name='CPU %', 
                    line=dict(color='#00CED1', width=2)
                ))
                
                # Memory Usage
                resource_fig.add_trace(go.Scatter(
                    x=resource_x, 
                    y=list(self.resource_monitor.memory_usage),
                    name='RAM %', 
                    line=dict(color='#FF8C00', width=2)
                ))
                
                # GPU metrics if available
                if self.resource_monitor.has_gpu:
                    resource_fig.add_trace(go.Scatter(
                        x=resource_x, 
                        y=list(self.resource_monitor.gpu_usage),
                        name='GPU %', 
                        line=dict(color='#7FFF00', width=2)
                    ))
                    
                    resource_fig.add_trace(go.Scatter(
                        x=resource_x, 
                        y=list(self.resource_monitor.gpu_memory),
                        name='GPU RAM %', 
                        line=dict(color='#FF1493', width=2)
                    ))
            
            resource_fig.update_layout(
                title=None,
                xaxis_title=None,
                yaxis_title="Usage %",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=20, b=40),
                hovermode="x unified",
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#2A2A2A',
                font=dict(color='#E6E6E6')
            )
            
            # Set y-axis range for resource chart
            resource_fig.update_yaxes(range=[0, 105])
            
            # Quote performance chart
            quote_perf_fig = go.Figure()
            if self.market_trades and self.quote_performance:
                # Performance line
                quote_perf_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.quote_performance),
                    name='Quote Efficiency',
                    line=dict(color='#00FF00', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ))
                
                # Add zero reference line
                quote_perf_fig.add_shape(
                    type="line", line=dict(color="#777", width=1, dash="dot"),
                    x0=0, y0=0, x1=1, y1=0,
                    xref="paper", yref="y"
                )
                
                # Add comparison with market trades
                quote_perf_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.market_trades),
                    name='Market Trades',
                    line=dict(color='#FF00FF', width=1, dash='dash'),
                    opacity=0.7
                ))
                
            quote_perf_fig.update_layout(
                title=None,
                xaxis_title=None,
                yaxis_title="% Difference",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=20, b=40),
                hovermode="x unified",
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#2A2A2A',
                font=dict(color='#E6E6E6')
            )
            
            # Exchange comparison chart
            exchange_fig = go.Figure()
            if self.exchange_timestamps:
                x_data = [t.strftime('%H:%M:%S') for t in self.exchange_timestamps]
                
                # Add our optimal quotes
                exchange_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.bid_prices),
                    name='Our Bid',
                    line=dict(color='#32CD32', width=2, dash='dash')
                ))
                
                exchange_fig.add_trace(go.Scatter(
                    x=x_data, y=list(self.ask_prices),
                    name='Our Ask',
                    line=dict(color='#FF6347', width=2, dash='dash')
                ))
                
                # Add other exchange data
                colors = ['#7DF9FF', '#FF8C00', '#7FFF00', '#FF1493', '#9370DB']
                color_idx = 0
                
                for exchange, data in self.exchange_data.items():
                    if 'bids' in data and len(data['bids']) > 0:
                        exchange_fig.add_trace(go.Scatter(
                            x=x_data[:len(data['bids'])], y=data['bids'],
                            name=f"{exchange.capitalize()} Bid",
                            line=dict(color=colors[color_idx % len(colors)], width=1.5)
                        ))
                    
                    if 'asks' in data and len(data['asks']) > 0:
                        exchange_fig.add_trace(go.Scatter(
                            x=x_data[:len(data['asks'])], y=data['asks'],
                            name=f"{exchange.capitalize()} Ask",
                            line=dict(color=colors[color_idx % len(colors)], width=1.5, dash='dot')
                        ))
                    
                    color_idx += 1
            
            exchange_fig.update_layout(
                title=None,
                xaxis_title=None,
                yaxis_title="Price",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=20, b=40),
                hovermode="x unified",
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#2A2A2A',
                font=dict(color='#E6E6E6')
            )
            
            # Enhanced stats table
            if self.timestamps:
                # Create a styled table
                stats_table = html.Table([
                    html.Tr([
                        html.Td("Current P&L", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"${self.strategy_state.get('total_pnl', self.strategy_state.get('pnl', 0)):.4f}", 
                            style={'textAlign': 'right', 'padding': '10px', 
                                    'color': '#32CD32' if self.strategy_state.get('total_pnl', self.strategy_state.get('pnl', 0)) >= 0 else '#FF6347',
                                    'borderBottom': '1px solid #333'})
                    ]),
                    html.Tr([
                        html.Th("Metric", style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid #555'}),
                        html.Th("Value", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #555'})
                    ]),
                    # Data rows
                    html.Tr([
                        html.Td("Current Bid", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{self.strategy_state['bid']:.4f}", style={'textAlign': 'right', 'padding': '10px', 'color': '#32CD32', 'borderBottom': '1px solid #333'})
                    ]),
                    html.Tr([
                        html.Td("Current Ask", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{self.strategy_state['ask']:.4f}", style={'textAlign': 'right', 'padding': '10px', 'color': '#FF6347', 'borderBottom': '1px solid #333'})
                    ]),
                    html.Tr([
                        html.Td("Spread", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{(self.strategy_state['ask'] - self.strategy_state['bid']):.4f}", 
                            style={'textAlign': 'right', 'padding': '10px', 'color': '#9370DB', 'borderBottom': '1px solid #333'})
                    ]),
                    html.Tr([
                        html.Td("Current Position", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{self.strategy_state['inventory']}", 
                            style={'textAlign': 'right', 'padding': '10px', 
                                    'color': '#32CD32' if self.strategy_state['inventory'] >= 0 else '#FF6347',
                                    'borderBottom': '1px solid #333'})
                    ]),
                    html.Tr([
                        html.Td("Current P&L", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{self.strategy_state['pnl']:.4f}", 
                            style={'textAlign': 'right', 'padding': '10px', 
                                    'color': '#32CD32' if self.strategy_state['pnl'] >= 0 else '#FF6347',
                                    'borderBottom': '1px solid #333'})
                    ]),
                    # Add system resource data
                    html.Tr([
                        html.Td("CPU Usage", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{self.resource_monitor.cpu_usage[-1]:.1f}%", 
                            style={'textAlign': 'right', 'padding': '10px', 'color': '#00CED1', 'borderBottom': '1px solid #333'})
                    ]),
                    html.Tr([
                        html.Td("Memory Usage", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{self.resource_monitor.memory_usage[-1]:.1f}%", 
                            style={'textAlign': 'right', 'padding': '10px', 'color': '#FF8C00', 'borderBottom': '1px solid #333'})
                    ]),
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
                
                if self.resource_monitor.has_gpu:
                    stats_table.children.extend([
                        html.Tr([
                            html.Td("GPU Usage", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"{self.resource_monitor.gpu_usage[-1]:.1f}%", 
                                style={'textAlign': 'right', 'padding': '10px', 'color': '#7FFF00', 'borderBottom': '1px solid #333'})
                        ]),
                        html.Tr([
                            html.Td("GPU Memory", style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"{self.resource_monitor.gpu_memory[-1]:.1f}%", 
                                style={'textAlign': 'right', 'padding': '10px', 'color': '#FF1493', 'borderBottom': '1px solid #333'})
                        ])
                    ])
            else:
                stats_table = html.Div("Waiting for data...", 
                                    style={'textAlign': 'center', 'padding': '40px', 'color': '#999'})
            
            # Exchange comparison table
            if self.exchange_data:
                # Create exchange comparison table
                rows = []
                
                # Header row
                rows.append(html.Tr([
                    html.Th("Exchange", style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid #555'}),
                    html.Th("Bid", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #555'}),
                    html.Th("Ask", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #555'}),
                    html.Th("Last", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #555'}),
                    html.Th("Bid-Ask %", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #555'})
                ]))
                
                # Add our quotes
                our_bid = self.strategy_state['bid']
                our_ask = self.strategy_state['ask']
                our_spread_pct = ((our_ask / our_bid) - 1) * 100 if our_bid > 0 else 0
                
                rows.append(html.Tr([
                    html.Td("Our Quotes", style={'padding': '10px', 'borderBottom': '1px solid #333', 'fontWeight': 'bold'}),
                    html.Td(f"{our_bid:.4f}", style={'textAlign': 'right', 'padding': '10px', 'color': '#32CD32', 'borderBottom': '1px solid #333'}),
                    html.Td(f"{our_ask:.4f}", style={'textAlign': 'right', 'padding': '10px', 'color': '#FF6347', 'borderBottom': '1px solid #333'}),
                    html.Td("-", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                    html.Td(f"{our_spread_pct:.2f}%", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'})
                ]))
                
                # Add exchange data
                for exchange, data in self.exchange_data.items():
                    if 'bids' in data and 'asks' in data and len(data['bids']) > 0 and len(data['asks']) > 0:
                        bid = data['bids'][-1]
                        ask = data['asks'][-1]
                        last = data['last'][-1] if 'last' in data and len(data['last']) > 0 else "-"
                        
                        spread_pct = ((ask / bid) - 1) * 100 if bid > 0 else 0
                        
                        # Calculate if our quotes are competitive
                        bid_class = 'color: #32CD32' if our_bid > bid else 'color: #FF6347'
                        ask_class = 'color: #32CD32' if our_ask < ask else 'color: #FF6347'
                        
                        rows.append(html.Tr([
                            html.Td(exchange.capitalize(), style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"{bid:.4f}", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"{ask:.4f}", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"{last:.4f}" if isinstance(last, (int, float)) else last, 
                                style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"{spread_pct:.2f}%", style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'})
                        ]))
                
                # Add arbitrage opportunities
                if hasattr(self, 'arbitrage_opportunities') and self.arbitrage_opportunities:
                    rows.append(html.Tr([
                        html.Th("Arbitrage", style={'textAlign': 'left', 'padding': '10px', 'borderTop': '1px solid #555', 'borderBottom': '1px solid #555'}, colSpan=5)
                    ]))
                    
                    for opp in self.arbitrage_opportunities[:3]:  # Show top 3 opportunities
                        rows.append(html.Tr([
                            html.Td(f"Buy: {opp['buy_exchange'].capitalize()}", 
                                style={'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"@ {opp['buy_price']:.4f}", 
                                style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"Sell: {opp['sell_exchange'].capitalize()}", 
                                style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"@ {opp['sell_price']:.4f}", 
                                style={'textAlign': 'right', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                            html.Td(f"+{opp['profit_pct']:.2f}%", 
                                style={'textAlign': 'right', 'padding': '10px', 'color': '#32CD32', 'borderBottom': '1px solid #333'})
                        ]))
                
                exchange_table = html.Table(rows, style={'width': '100%', 'borderCollapse': 'collapse'})
            else:
                exchange_table = html.Div("Waiting for exchange data...", 
                                        style={'textAlign': 'center', 'padding': '40px', 'color': '#999'})
            
            # Update symbol header with more styling
            symbol_header = f"Symbol: {self.strategy_state['symbol'].upper()}"
            
            return price_fig, position_fig, pnl_fig, resource_fig, stats_table, symbol_header, quote_perf_fig, exchange_fig, exchange_table

            
    def start(self):
        if not self.timestamps:
            now = datetime.now()
            base_price = 1000.0  # Default starting price
            
            for i in range(20):
                # Add timestamps from 2 minutes ago to now
                timestamp = now - timedelta(seconds=120-i*6)
                self.timestamps.append(timestamp)
                
                # Create realistic price movements
                noise = random.normalvariate(0, 0.0005)  # Small price variations
                trend = 0.0001 * i  # Small upward trend
                
                mid_price = base_price * (1 + noise + trend)
                bid_price = mid_price * 0.999
                ask_price = mid_price * 1.001
                
                self.mid_prices.append(mid_price)
                self.bid_prices.append(bid_price)
                self.ask_prices.append(ask_price)
                
                # Add initial flat positions and P&L
                init_inventory = self.strategy_state.get('inventory', 0)
                self.inventory_history.append(init_inventory)
                
                init_pnl = 0.0
                self.pnl_history.append(init_pnl)
        
        # Start the dashboard thread
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        print(f"Dashboard started at http://localhost:8050")


    def update(self, strategy_state):
        """Update dashboard with new data"""
        now = datetime.now()
        
        # Update strategy state
        self.strategy_state = strategy_state
        
        # Make sure 'pnl' is always populated (for backward compatibility)
        if 'pnl' not in self.strategy_state and 'total_pnl' in self.strategy_state:
            self.strategy_state['pnl'] = self.strategy_state['total_pnl']
        
        # Update data containers with smoothing to avoid scattered appearance
        self.timestamps.append(now)
        
        # Calculate mid price and add some smoothing to avoid jumpy charts
        mid_price = (strategy_state['bid'] + strategy_state['ask'])/2
        if len(self.mid_prices) > 0:
            # Apply light smoothing (90% new value, 10% previous value)
            smoothed_mid = 0.9 * mid_price + 0.1 * self.mid_prices[-1]
            smoothed_bid = 0.9 * strategy_state['bid'] + 0.1 * self.bid_prices[-1]
            smoothed_ask = 0.9 * strategy_state['ask'] + 0.1 * self.ask_prices[-1]
        else:
            smoothed_mid = mid_price
            smoothed_bid = strategy_state['bid']
            smoothed_ask = strategy_state['ask']
        
        self.mid_prices.append(smoothed_mid)
        self.bid_prices.append(smoothed_bid)
        self.ask_prices.append(smoothed_ask)
        
        # Smooth inventory updates to avoid jumps
        inventory = strategy_state['inventory']
        if len(self.inventory_history) > 0:
            # Only smooth small changes, allow big jumps for trades
            if abs(inventory - self.inventory_history[-1]) < 1.0:
                inventory = 0.8 * inventory + 0.2 * self.inventory_history[-1]
        self.inventory_history.append(inventory)
        
        # Use 'total_pnl' if available, otherwise fallback to 'pnl'
        pnl_value = strategy_state.get('total_pnl', strategy_state.get('pnl', 0))
        
        # Smooth PnL updates
        if len(self.pnl_history) > 0:
            # Allow PnL to jump on trades but smooth market fluctuations
            if abs(pnl_value - self.pnl_history[-1]) < 10.0:
                pnl_value = 0.9 * pnl_value + 0.1 * self.pnl_history[-1]
        self.pnl_history.append(pnl_value)
        
        # Print current status to console in a nice format
        print(tabulate(
            [[f"{now}", f"{strategy_state['bid']:.4f}", f"{strategy_state['ask']:.4f}", 
            f"{strategy_state['inventory']:.2f}", f"{pnl_value:.4f}"]],
            headers=['Time', 'Bid', 'Ask', 'Position', 'P&L']
        ))


    def update_exchange_data(self, exchange_data, arbitrage_opportunities=None):
        """Update dashboard with multi-exchange data"""
        now = datetime.now()
        self.exchange_timestamps.append(now)
        
        # Update exchange data
        self.exchange_data = exchange_data
        
        # Update arbitrage opportunities
        if arbitrage_opportunities is not None:
            self.arbitrage_opportunities = arbitrage_opportunities
            
        # Calculate and track quote performance metrics if last trade price is available
        if 'last_trade_price' in self.strategy_state and self.strategy_state['last_trade_price'] is not None:
            self.market_trades.append(self.strategy_state['last_trade_price'])
            
            # Calculate how far our mid-quote is from the market trade (%)
            mid_quote = (self.strategy_state['bid'] + self.strategy_state['ask']) / 2
            if mid_quote > 0 and self.strategy_state['last_trade_price'] > 0:
                perf_pct = ((mid_quote / self.strategy_state['last_trade_price']) - 1) * 100
                self.quote_performance.append(perf_pct)
                
    def run(self):
        try:
            print("Starting Dash server on port 8050...")
            # Start the Dash app server with threaded=True for better performance
            self.app.run(debug=False, host='127.0.0.1', port=8050, threaded=True)
            print("Dash server started successfully")
        except Exception as e:
            print(f"Error starting dashboard server: {str(e)}")
            import traceback
            traceback.print_exc()



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
    
    # Get user ticker choice
    user_ticker, use_scanner = get_user_ticker_choice()
    
    # Initialize hot crypto scanner if user chose it
    scanner = None
    if use_scanner:
        scanner = CryptoHeatScanner(top_n=5, update_interval=60)
        print("Waiting for hot cryptocurrency data...")
        
        # Wait for scanner to collect initial data with a timeout
        start_time = time.time()
        while not scanner.get_rankings():
            time.sleep(1)
            if time.time() - start_time > 30:  # 30 seconds timeout
                print("Timeout waiting for crypto data. Using default BTC/USDT.")
                break
        
        # Get the hottest cryptocurrency
        symbol = scanner.get_top_symbol() or "btcusdt"  # Default to BTC if none found
        print(f"Selected {symbol} for market making based on activity")
    else:
        # Use user-specified ticker
        symbol = user_ticker
        print(f"Using user-selected ticker: {symbol}")
    
    # Get initial portfolio setup
    initial_cash, initial_inventory = get_user_portfolio_setup()
    
    # Initialize data engine
    data_engine = DataEngine(symbol)
    
    # Initialize dashboard with enhanced UI
    dashboard = TradingDashboard(update_interval=500)  # More frequent updates (500ms)
    dashboard.strategy_state['symbol'] = symbol
    dashboard.strategy_state['initial_cash'] = initial_cash
    dashboard.strategy_state['cash'] = initial_cash
    dashboard.strategy_state['inventory'] = initial_inventory
    dashboard.start()
    
    # Initialize exchange comparator
    comparator = ExchangeComparator(
        symbol=symbol,
        exchanges=['binance', 'coinbase', 'kraken', 'kucoin', 'okx'],
        update_interval=5.0  # Update every 5 seconds to avoid rate limits
    )

    # Wait for initial exchange data
    print("Waiting for exchange comparison data...")
    wait_start = time.time()
    timeout = 15  # shorter timeout for exchange data
    while not comparator.get_current_data() and time.time() - wait_start < timeout:
        time.sleep(1)
    
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
            print("Using synthetic data for testing...")
            # Generate synthetic data to test dashboard
            data_engine.latest_data = {
                'bid': 50000,
                'ask': 50100,
                'trade': 50050,
                'volume': 1.0,
                'timestamp': datetime.now().timestamp()
            }
            break
    
    # Initialize portfolio tracker
    portfolio = PortfolioTracker(
        initial_cash=initial_cash,
        initial_inventory=initial_inventory,
        symbol=symbol
    )
    
    # Initialize solver with current price range
    current_price = (data_engine.latest_data['bid'] + data_engine.latest_data['ask']) / 2
    
    # Update portfolio with initial price
    portfolio.update_price(current_price)
    
    S_min = current_price * 0.9
    S_max = current_price * 1.1
    I_min = -100  # Max short position
    I_max = 100   # Max long position
    
    grid_size = 51 if not USE_GPU else 101
    
    # Initialize solver with jump diffusion parameters
    solver = HJBSolver(
        S_min, S_max, I_min, I_max, 
        N_S=grid_size, N_I=grid_size,
        jump_intensity=0.1,  # Jump occurs with 10% probability per unit time
        jump_mean=0.0,       # Zero mean jump (symmetric)
        jump_std=0.01        # 1% standard deviation for jumps
    )
    
    # Update inventory to use the initial value
    current_inventory = initial_inventory
    
    filled_orders = []
    last_symbol_check = time.time()
    symbol_check_interval = 300  # Check for new hot symbol every 5 minutes (only if using scanner)
    last_update_time = time.time()
    update_interval = 0.1  # Update at least every 100ms
    
    print(f"Starting market making with {symbol}...")
    print(f"Using {'GPU' if USE_GPU else 'CPU'} for computations")
    print(f"Initial portfolio: ${initial_cash:.2f} cash, {initial_inventory:.4f} {symbol.upper()}")
    
    update_count = 0
    
    try:
        while True:
            try:
                current_time = time.time()
                update_count += 1
                        
                # Check periodically if a different cryptocurrency is now hotter (only if using scanner)
                if use_scanner and current_time - last_symbol_check > symbol_check_interval:
                    new_hot_symbol = scanner.get_top_symbol()
                    if new_hot_symbol != symbol and new_hot_symbol is not None:
                        print(f"Switching from {symbol} to hotter cryptocurrency {new_hot_symbol}")
                        
                        # Save portfolio metrics before switching
                        prev_metrics = portfolio.get_metrics()
                        print(f"Portfolio before switch: ${prev_metrics['portfolio_value']:.2f} "
                              f"(P&L: ${prev_metrics['total_pnl']:.2f})")
                        
                        # Set up the new symbol
                        symbol = new_hot_symbol
                        data_engine = DataEngine(symbol)
                        dashboard.strategy_state['symbol'] = symbol
                        
                        # Transfer portfolio value to new symbol
                        portfolio = PortfolioTracker(
                            initial_cash=prev_metrics['portfolio_value'],
                            initial_inventory=0.0,  # Start with no inventory in new symbol
                            symbol=symbol
                        )
                        current_inventory = 0
                        
                        # Wait for initial data
                        wait_start = time.time()
                        while (data_engine.latest_data['bid'] is None or data_engine.latest_data['ask'] is None):
                            if time.time() - wait_start > 5:  # Shorter timeout when switching
                                print(f"Using synthetic data for {symbol}")
                                # Generate synthetic data
                                data_engine.latest_data = {
                                    'bid': random.uniform(100, 50000),
                                    'ask': random.uniform(100, 50000) * 1.001,  # Ensure ask > bid
                                    'trade': random.uniform(100, 50000),
                                    'volume': random.uniform(0.1, 10),
                                    'timestamp': datetime.now().timestamp()
                                }
                                break
                            time.sleep(0.1)
                        
                        # Reset solver with new price range
                        current_price = (data_engine.latest_data['bid'] + data_engine.latest_data['ask']) / 2
                        portfolio.update_price(current_price)
                        
                        solver = HJBSolver(
                            current_price * 0.9, 
                            current_price * 1.1,
                            I_min, I_max,
                            N_S=grid_size, N_I=grid_size
                        )
                    
                    last_symbol_check = current_time
                
                # Process data queue 
                data_processed = False
                while not data_engine.data_queue.empty():
                    update = data_engine.data_queue.get()
                    data_processed = True
                    
                    # Process any updates
                    if update['type'] in ('book', 'trade'):
                        # Get bid/ask from update or use latest data
                        bid = update.get('bid', data_engine.latest_data['bid'])
                        ask = update.get('ask', data_engine.latest_data['ask'])
                        
                        if bid is not None and ask is not None:
                            # Update portfolio with latest price
                            mid_price = (bid + ask) / 2
                            portfolio.update_price(mid_price)
                            
                            # Update solver (less frequently for CPU mode)
                            if USE_GPU or update_count % 5 == 0:
                                solver.update(bid, ask, dt=0.001)
                            
                            optimal_bid, optimal_ask = solver.get_optimal_quotes(mid_price, current_inventory)
                            
                            # Execute trades based on real market data instead of random
                            if 'trade' in update or data_engine.latest_data['trade'] is not None:
                                last_trade_price = update.get('price', data_engine.latest_data['trade'])
                                if last_trade_price is not None:
                                    execution_result = process_potential_executions(optimal_bid, optimal_ask, last_trade_price, portfolio)
                                    if execution_result['executed']:
                                        # Add to filled orders
                                        filled_orders.append(execution_result['trade'])
                                        # Update current inventory
                                        current_inventory = portfolio.inventory
                
                if not data_processed and current_time - last_update_time > update_interval:
                    bid = data_engine.latest_data['bid']
                    ask = data_engine.latest_data['ask']
                    
                    if bid is not None and ask is not None:
                        # Add price fluctuations to simulate market movement
                        price_change = random.uniform(-0.001, 0.001)
                        bid *= (1 + price_change)
                        ask *= (1 + price_change)
                        
                        # Update data store
                        data_engine.latest_data['bid'] = bid
                        data_engine.latest_data['ask'] = ask
                        
                        # Update portfolio with new price
                        mid_price = (bid + ask) / 2
                        portfolio.update_price(mid_price)
                        
                        # Update solver less frequently in CPU mode
                        if USE_GPU or update_count % 5 == 0:
                            solver.update(bid, ask, dt=0.001)
                        
                        optimal_bid, optimal_ask = solver.get_optimal_quotes(mid_price, current_inventory)
                        
                        # Get metrics for dashboard update
                        portfolio_metrics = portfolio.get_metrics()
                        
                        # Update dashboard with new values including portfolio metrics
                        dashboard.update({
                            'bid': optimal_bid,
                            'ask': optimal_ask,
                            'inventory': portfolio_metrics['inventory'],
                            'cash': portfolio_metrics['cash'],
                            'realized_pnl': portfolio_metrics['realized_pnl'],
                            'unrealized_pnl': portfolio_metrics['unrealized_pnl'],
                            'total_pnl': portfolio_metrics['total_pnl'],
                            'portfolio_value': portfolio_metrics['portfolio_value'],
                            'symbol': symbol,
                            'last_trade_price': data_engine.latest_data['trade']  # Add the last trade price
                        })
                        
                        # Update dashboard with exchange comparison data
                        comparison_data = comparator.get_data_for_chart()['exchanges']
                        arbitrage_opps = comparator.calculate_arbitrage_opportunities()
                        dashboard.update_exchange_data(comparison_data, arbitrage_opps)
                    
                    last_update_time = current_time
                
                # Performance optimization: sleep longer if CPU mode
                time.sleep(0.01 if USE_GPU else 0.05)
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                print(f"Exception type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping market making strategy...")
        
        # Print final portfolio status
        if 'portfolio' in locals():
            metrics = portfolio.get_metrics()
            print("\n=== FINAL PORTFOLIO STATUS ===")
            print(f"Symbol: {symbol.upper()}")
            print(f"Cash: ${metrics['cash']:.2f}")
            print(f"Inventory: {metrics['inventory']:.4f} units")
            print(f"Portfolio Value: ${metrics['portfolio_value']:.2f}")
            print(f"Total P&L: ${metrics['total_pnl']:.2f}")
            print(f"Number of Trades: {len(portfolio.trades)}")
            print("============================\n")
            
    finally:
        # Clean up resources
        if hasattr(dashboard, 'resource_monitor'):
            dashboard.resource_monitor.cleanup()
        if 'comparator' in locals():
            comparator.stop()
        print("Resources cleaned up. Exiting.")
    
if __name__ == "__main__":
    main()