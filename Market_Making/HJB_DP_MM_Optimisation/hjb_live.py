import numpy as np
import pandas as pd
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
import random
import websocket
import warnings
import os

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
            html.H1("Frankline & Co LP HJB Strategy Market Making Dashboard"),
            
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
    
    # Initialize components
    data_engine = DataEngine(symbol)
    
    # Initialize dashboard early (but don't show data yet)
    dashboard = TradingDashboard(update_interval=500)  # More frequent updates (500ms)
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
    
    # Initialize solver with current price range
    current_price = (data_engine.latest_data['bid'] + data_engine.latest_data['ask']) / 2
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
    current_inventory = 0
    cumulative_pnl = 0
    filled_orders = []
    last_symbol_check = time.time()
    symbol_check_interval = 300  # Check for new hot symbol every 5 minutes
    last_update_time = time.time()
    update_interval = 0.1  # Update at least every 100ms
    
    print(f"Starting market making with {symbol}...")
    print(f"Using {'GPU' if USE_GPU else 'CPU'} for computations")
    
    update_count = 0
    
    while True:
        try:
            current_time = time.time()
            update_count += 1
                    
            # Check periodically if a different cryptocurrency is now hotter
            if current_time - last_symbol_check > symbol_check_interval:
                new_hot_symbol = scanner.get_top_symbol()
                if new_hot_symbol != symbol and new_hot_symbol is not None:
                    print(f"Switching from {symbol} to hotter cryptocurrency {new_hot_symbol}")
                    symbol = new_hot_symbol
                    data_engine = DataEngine(symbol)
                    dashboard.strategy_state['symbol'] = symbol
                    
                    # Reset state for new symbol
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
                        # Update solver (less frequently for CPU mode)
                        if USE_GPU or update_count % 5 == 0:
                            solver.update(bid, ask, dt=0.001)
                        
                        mid_price = (bid + ask) / 2
                        optimal_bid, optimal_ask = solver.get_optimal_quotes(mid_price, current_inventory)
                        
                        # Simulate order executions
                        if random.random() < 0.05:  # 5% chance of execution
                            if random.random() < 0.5:  # Buy execution
                                size = random.randint(1, 5)
                                current_inventory += size
                                execution_price = optimal_ask * (1 + random.uniform(-0.001, 0.001))
                                cumulative_pnl -= execution_price * size
                                print(f"BUY EXECUTED: {size} @ {execution_price:.4f}")
                            else:  # Sell execution
                                size = random.randint(1, 5)
                                current_inventory -= size
                                execution_price = optimal_bid * (1 + random.uniform(-0.001, 0.001))
                                cumulative_pnl += execution_price * size
                                print(f"SELL EXECUTED: {size} @ {execution_price:.4f}")
                        
                        # Update dashboard
                        dashboard.update({
                            'bid': optimal_bid,
                            'ask': optimal_ask,
                            'inventory': current_inventory,
                            'pnl': cumulative_pnl, 
                            'symbol': symbol
                        })
            
            # Generate periodic updates even without new data
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
                    
                    # Update solver less frequently in CPU mode
                    if USE_GPU or update_count % 5 == 0:
                        solver.update(bid, ask, dt=0.001)
                    
                    mid_price = (bid + ask) / 2
                    optimal_bid, optimal_ask = solver.get_optimal_quotes(mid_price, current_inventory)
                    
                    # Update dashboard with new values
                    dashboard.update({
                        'bid': optimal_bid,
                        'ask': optimal_ask,
                        'inventory': current_inventory,
                        'pnl': cumulative_pnl,
                        'symbol': symbol
                    })
                
                last_update_time = current_time
            
            # Performance optimization: sleep longer if CPU mode
            time.sleep(0.01 if USE_GPU else 0.05)
            
        except KeyboardInterrupt:
            print("\nStopping market making strategy...")
            break
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    main()
