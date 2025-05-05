import numpy as np
import pandas as pd
from datetime import datetime
import time

class KyleModel:
    def __init__(self, initial_price, uncertainty_sigma0, noise_sigma_u):
        """
        Initialize the Kyle model.
        
        Args:
            initial_price (float): Initial price of the asset (p₀)
            uncertainty_sigma0 (float): Prior uncertainty about asset value (Σ₀)
            noise_sigma_u (float): Standard deviation of noise trading (σᵤ)
        """
        self.p0 = initial_price
        self.sigma0 = uncertainty_sigma0
        self.sigma_u = noise_sigma_u
        
        # Calculate the lambda (price impact) parameter
        self.lambda_param = (1/2) * np.sqrt(self.sigma0 / self.sigma_u)
        
        # Initialize state variables
        self.current_price = initial_price
        self.time_series = []
        self.true_value = None  # Will be estimated from market data
        
        # For tracking model performance and state
        self.last_update_time = time.time()
        self.order_flow_history = []
        self.price_history = []
        self.insider_order_history = []
        self.noise_order_history = []
        self.info_revelation_metric = []
    
    def estimate_true_value(self, market_data, window=100):
        """
        Estimate the true value of the asset based on recent market data.
        In a real market, we don't know the true value, but we can approximate it
        using future prices or fundamental analysis.
        
        Args:
            market_data (pd.DataFrame): Recent market data
            window (int): Window of data to consider
        
        Returns:
            float: Estimated true value
        """
        if market_data.empty:
            return self.p0
            
        # Use mid price as an estimator for true value
        recent_data = market_data.tail(window)
        estimated_value = recent_data['mid_price'].mean()
        
        # Add some random component to simulate private information
        random_component = np.random.normal(0, self.sigma0 / 10)
        self.true_value = estimated_value + random_component
        
        return self.true_value
    
    def simulate_noise_trading(self):
        """
        Simulate noise trading based on the model parameters.
        
        Returns:
            float: Simulated noise trading volume
        """
        return np.random.normal(0, self.sigma_u)
    
    def simulate_insider_trading(self, true_value):
        """
        Simulate insider trading based on the model parameters and true value.
        
        Args:
            true_value (float): True value of the asset
        
        Returns:
            float: Simulated insider trading volume
        """
        # Insider's optimal order: x = (σᵤ/√Σ₀)(ν - p₀)
        return (self.sigma_u / np.sqrt(self.sigma0)) * (true_value - self.current_price)
    
    def calculate_price_update(self, total_order_flow):
        """
        Calculate the price update based on the observed order flow.
        
        Args:
            total_order_flow (float): Total observed order flow
        
        Returns:
            float: New price
        """
        # Equilibrium price: p = p₀ + λ(u + x)
        return self.p0 + self.lambda_param * total_order_flow
    
    def update(self, market_data):
        """
        Update the model state based on new market data.
        
        Args:
            market_data (pd.DataFrame): New market data
        
        Returns:
            dict: Updated model state and metrics
        """
        if market_data.empty:
            return {
                'price': self.current_price,
                'lambda': self.lambda_param,
                'market_depth': 1/self.lambda_param,
                'info_revelation': 0.0
            }
            
        # Estimate the true value based on market data
        true_value = self.estimate_true_value(market_data)
        
        # Simulate noise and insider trading
        noise_order = self.simulate_noise_trading()
        insider_order = self.simulate_insider_trading(true_value)
        total_order_flow = noise_order + insider_order
        
        # Calculate the new price
        new_price = self.calculate_price_update(total_order_flow)
        
        # Calculate information revelation (how much of insider info is reflected in price)
        # In Kyle model, half of the insider info should be reflected in price
        price_change = new_price - self.current_price
        theoretical_full_info_change = true_value - self.current_price
        info_revelation = 0.0 if theoretical_full_info_change == 0 else min(1.0, max(0.0, 
                                 abs(price_change / theoretical_full_info_change)))
        
        # Update state
        self.current_price = new_price
        self.order_flow_history.append(total_order_flow)
        self.price_history.append(new_price)
        self.insider_order_history.append(insider_order)
        self.noise_order_history.append(noise_order)
        self.info_revelation_metric.append(info_revelation)
        self.last_update_time = time.time()
        
        # Create model metrics for dashboard
        return {
            'price': new_price,
            'true_value': true_value,
            'total_order_flow': total_order_flow,
            'noise_order': noise_order,
            'insider_order': insider_order,
            'lambda': self.lambda_param,
            'market_depth': 1/self.lambda_param,
            'info_revelation': info_revelation,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
    
    def get_history(self):
        """
        Get the history of model updates for visualization.
        
        Returns:
            pd.DataFrame: History of model updates
        """
        df = pd.DataFrame({
            'timestamp': [datetime.now() - pd.Timedelta(seconds=i) for i in range(len(self.price_history))],
            'price': self.price_history,
            'order_flow': self.order_flow_history,
            'insider_order': self.insider_order_history,
            'noise_order': self.noise_order_history,
            'info_revelation': self.info_revelation_metric
        })
        
        return df.iloc[::-1]  # Reverse to have most recent first