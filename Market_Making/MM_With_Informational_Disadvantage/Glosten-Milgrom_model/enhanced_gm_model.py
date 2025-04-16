import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GlostenMilgromModelPriceSensitive:
    """
    Implements the Glosten-Milgrom model with price-sensitive
    uninformed traders.
    """

    def __init__(self, v_high, v_low, p, alpha, c_dist):
        """
        Initializes the Glosten-Milgrom model with price sensitivity.

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

    def calculate_spreads(self, delta_a_prev=None, delta_b_prev=None):
        """
        Calculates the bid-ask spread, potentially iteratively if needed.

        Args:
            delta_a_prev (float, optional): Previous ask-half-spread for iteration.
            delta_b_prev (float, optional): Previous bid-half-spread for iteration.

        Returns:
            tuple: A tuple containing the ask-half-spread and bid-half-spread.
        """
        def spread_equations(delta_a, delta_b):
            f_delta_a = self.c_dist(delta_a)
            f_delta_b = self.c_dist(delta_b)

            new_delta_a = (1 / (1 + ((1 - self.alpha) / self.alpha) * ((1 - f_delta_a) / 2 / self.p))) * (self.v_high - self.mu)
            new_delta_b = (1 / (1 + ((1 - self.alpha) / self.alpha) * ((1 - f_delta_b) / 2 / (1 - self.p))) ) * (self.mu - self.v_low)
            return new_delta_a, new_delta_b
        
        delta_a, delta_b = spread_equations(0.01,0.01) #initial values
        
        # Iterate to find a fixed point
        for _ in range(100):  # Maximum iterations to prevent infinite loop
            new_delta_a, new_delta_b = spread_equations(delta_a, delta_b)
            if abs(new_delta_a - delta_a) < 1e-6 and abs(new_delta_b - delta_b) < 1e-6:
                break
            delta_a, delta_b = new_delta_a, new_delta_b

        return delta_a, delta_b

    def simulate_trade(self):
        """
        Simulates a single trade with price-sensitive uninformed traders.

        Returns:
            tuple: A tuple containing the trader type, trade direction, and price.
        """
        trader_type = 'Informed' if np.random.rand() < self.alpha else 'Uninformed'
        delta_a, delta_b = self.calculate_spreads()
        
        if trader_type == 'Informed':
            if np.random.rand() < self.p:
                trade_direction = 'Buy'
            else:
                trade_direction = 'Sell'
        else:
            c = self.c_dist(np.random.rand())  # Draw urgency parameter
            if np.random.rand() < 0.5:  # Uninformed trader wants to buy
                if c > delta_a:
                    trade_direction = 'Buy'
                else:
                    return 'Uninformed', 'No Trade', None
            else:  # Uninformed trader wants to sell
                if c > delta_b:
                    trade_direction = 'Sell'
                else:
                    return 'Uninformed', 'No Trade', None

        price = self.mu + delta_a if trade_direction == 'Buy' else self.mu - delta_b if trade_direction != 'No Trade' else None
        return trader_type, trade_direction, price

    def simulate_multiple_trades(self, n_trades):
        """
        Simulates multiple trades.

        Args:
            n_trades (int): Number of trades to simulate.

        Returns:
            pandas.DataFrame: DataFrame containing the simulation results.
        """
        trades = []
        for _ in range(n_trades):
            trader_type, trade_direction, price = self.simulate_trade()
            trades.append([trader_type, trade_direction, price])
        return pd.DataFrame(trades, columns=['Trader_Type', 'Trade_Direction', 'Price'])

    def analyze_trades(self, trades_df):
        """
        Analyzes the simulated trades.

        Args:
            trades_df (pandas.DataFrame): DataFrame containing simulation results.

        Returns:
            dict: A dictionary containing trade statistics.
        """
        trades_df = trades_df[trades_df['Trade_Direction'] != 'No Trade'] # Filter out no-trade
        
        buy_trades = trades_df[trades_df['Trade_Direction'] == 'Buy']
        sell_trades = trades_df[trades_df['Trade_Direction'] == 'Sell']
        
        stats = {
            'total_trades': len(trades_df),
            'buy_trade_count': len(buy_trades),
            'sell_trade_count': len(sell_trades),
            'informed_buy_ratio': len(buy_trades[trades_df['Trader_Type'] == 'Informed']) / len(buy_trades) if len(buy_trades) > 0 else 0,
            'informed_sell_ratio': len(sell_trades[trades_df['Trader_Type'] == 'Informed']) / len(sell_trades) if len(sell_trades) > 0 else 0
        }
        return stats

    def visualize_trades(self, trades_df):
        """
        Visualizes the simulated trades.

        Args:
            trades_df (pandas.DataFrame): DataFrame containing simulation results.
        """
        trades_df = trades_df[trades_df['Price'].notna()]  # Filter out rows with None prices

        plt.figure(figsize=(8, 5))
        trades_df['Price'].plot()
        plt.xlabel('Trade Number')
        plt.ylabel('Price')
        plt.title('Glosten-Milgrom Model (Price Sensitive) Simulation')
        plt.show()

        plt.figure(figsize=(6, 4))
        trades_df['Trade_Direction'].value_counts().plot(kind='bar')
        plt.xlabel('Trade Direction')
        plt.ylabel('Frequency')
        plt.title('Trade Direction Distribution')
        plt.show()