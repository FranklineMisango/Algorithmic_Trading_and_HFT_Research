import numpy as np
import pandas as pd

class GlostenMilgromModelLive:
    """
    Implements the Glosten-Milgrom model with real-time data integration.
    """

    def __init__(self, v_high, v_low, p, alpha, c_dist):
        """
        Initializes the Glosten-Milgrom model.

        Args:
            v_high (float): High possible value of the asset.
            v_low (float): Low possible value of the asset.
            p (float): Probability of the asset having a high value.
            alpha (float): Proportion of informed traders.
            c_dist (function): Cumulative distribution function for
                              uninformed traders' urgency parameter c.
        """
        self.v_high = v_high
        self.v_low = v_low
        self.p = p
        self.alpha = alpha
        self.mu = p * v_high + (1 - p) * v_low
        self.c_dist = c_dist
        
        # For tracking recent price updates
        self.price_history = []
        self.max_history = 1000

    def update_price_belief(self, market_data):
        """
        Updates the model's belief about the asset price based on market data.
        
        Args:
            market_data (pd.DataFrame): Recent market data
            
        Returns:
            None: Updates internal state
        """
        # In a complete implementation, this would update p based on the observed
        # order flow, potentially using Bayesian updating
        
        # For now, we'll use a simple approach: adjust p based on recent price movements
        if market_data is not None and not market_data.empty:
            if 'mid_price' in market_data.columns:
                mid_prices = market_data['mid_price'].dropna().tolist()
            else:
                mid_prices = [(bid + ask) / 2 for bid, ask in 
                              zip(market_data['bid_price'], market_data['ask_price'])]
            
            # Add to history
            self.price_history.extend(mid_prices)
            if len(self.price_history) > self.max_history:
                self.price_history = self.price_history[-self.max_history:]
            
            # Update p if we have enough data
            if len(self.price_history) > 10:
                recent_avg = np.mean(self.price_history[-10:])
                overall_avg = np.mean(self.price_history)
                
                # Adjust p based on recent price movement direction
                drift = (recent_avg - overall_avg) / (self.v_high - self.v_low)
                
                # Constrain adjustment to maintain p between 0.1 and 0.9
                self.p = max(0.1, min(0.9, self.p + drift * 0.01))
                
                # Update mu after p changes
                self.mu = self.p * self.v_high + (1 - self.p) * self.v_low

    def calculate_spreads(self):
        """
        Calculates the bid-ask spread based on current model parameters.

        Returns:
            tuple: A tuple containing the ask-half-spread and bid-half-spread.
        """
        def spread_equations(delta_a, delta_b):
            f_delta_a = self.c_dist(delta_a)
            f_delta_b = self.c_dist(delta_b)

            new_delta_a = (1 / (1 + ((1 - self.alpha) / self.alpha) * 
                             ((1 - f_delta_a) / 2 / self.p))) * (self.v_high - self.mu)
            new_delta_b = (1 / (1 + ((1 - self.alpha) / self.alpha) * 
                             ((1 - f_delta_b) / 2 / (1 - self.p)))) * (self.mu - self.v_low)
            return new_delta_a, new_delta_b
        
        delta_a, delta_b = spread_equations(0.01, 0.01)  # initial values
        
        # Iterate to find a fixed point
        for _ in range(100):  # Maximum iterations to prevent infinite loop
            new_delta_a, new_delta_b = spread_equations(delta_a, delta_b)
            if abs(new_delta_a - delta_a) < 1e-6 and abs(new_delta_b - delta_b) < 1e-6:
                break
            delta_a, delta_b = new_delta_a, new_delta_b

        return delta_a, delta_b

    def calculate_spreads_from_data(self, market_data):
        """
        Updates model parameters based on market data and calculates spreads.
        
        Args:
            market_data (pd.DataFrame): Recent market data
            
        Returns:
            dict: Dictionary with calculated bid/ask values
        """
        # Update model beliefs based on market data
        self.update_price_belief(market_data)
        
        # Calculate spreads
        delta_a, delta_b = self.calculate_spreads()
        
        # Calculate model bid/ask prices
        gm_ask = self.mu + delta_a
        gm_bid = self.mu - delta_b
        
        # Create a smooth time series of spreads
        timestamps = market_data['timestamp'].tolist()
        n = len(timestamps)
        
        return {
            'delta_a': [delta_a] * n,
            'delta_b': [delta_b] * n,
            'gm_ask': [gm_ask] * n,
            'gm_bid': [gm_bid] * n
        }