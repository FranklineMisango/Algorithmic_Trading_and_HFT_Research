import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class KyleModel:
    """
    Implements the Kyle (1985) model of informed trading.
    """

    def __init__(self, mu, sigma_v, sigma_u, lambda_param):
        """
        Initializes the Kyle Model.

        Args:
            mu (float): Mean of the asset's fundamental value.
            sigma_v (float): Standard deviation of the asset's fundamental value.
            sigma_u (float): Standard deviation of the noise trader demand.
            lambda_param (float): Kyle's lambda (price impact parameter).
        """
        self.mu = mu
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.lambda_param = lambda_param
        self.beta = 0.5 / lambda_param  # Insider's trading intensity

    def informed_trader_demand(self, v):
        """
        Calculates the informed trader's demand.

        Args:
            v (float): The asset's fundamental value.

        Returns:
            float: The informed trader's demand.
        """
        return self.beta * (v - self.mu)

    def market_price(self, x, u):
        """
        Calculates the market price based on order flow.

        Args:
            x (float): Informed trader's demand.
            u (float): Noise trader demand.

        Returns:
            float: The market price.
        """
        return self.mu + self.lambda_param * (x + u)

    def simulate_trade(self, v, u):
        """
        Simulates a single trading round.

        Args:
            v (float): Asset's fundamental value.
            u (float): Noise trader demand.

        Returns:
            tuple: A tuple containing the informed trader's demand,
                   noise trader demand, and the resulting market price.
        """
        x = self.informed_trader_demand(v)
        price = self.market_price(x, u)
        return x, u, price

    def simulate_multiple_trades(self, n_trades):
        """
        Simulates multiple trading rounds.

        Args:
            n_trades (int): The number of trading rounds to simulate.

        Returns:
            pandas.DataFrame: A DataFrame containing the simulation results.
        """
        v_values = np.random.normal(self.mu, self.sigma_v, n_trades)
        u_values = np.random.normal(0, self.sigma_u, n_trades)  # Noise trader demand centered at 0
        
        trades = []
        for i in range(n_trades):
            x, u, price = self.simulate_trade(v_values[i], u_values[i])
            trades.append([v_values[i], x, u, price])
        
        return pd.DataFrame(trades, columns=['Fundamental_Value', 'Informed_Demand', 'Noise_Demand', 'Price'])

    def backtest_informed_trader(self, trades_df):
        """
        Backtests a simple strategy for the informed trader.

        Args:
            trades_df (pandas.DataFrame): DataFrame containing simulation results.

        Returns:
            pandas.DataFrame: DataFrame with added profit/loss information.
        """
        trades_df['Position'] = np.where(trades_df['Informed_Demand'] > 0, 1, -1)  # Long if demand > 0, else short
        trades_df['Profit'] = trades_df['Position'].shift(1) * (trades_df['Fundamental_Value'] - trades_df['Price'])
        trades_df['Cumulative_Profit'] = trades_df['Profit'].cumsum()
        return trades_df

    def visualize_simulation(self, trades_df):
        """
        Visualizes the simulation results.

        Args:
            trades_df (pandas.DataFrame): DataFrame containing simulation results.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(trades_df['Price'], label='Price', color='blue')
        plt.plot(trades_df['Fundamental_Value'], label='Fundamental Value', color='green', linestyle='--')
        plt.xlabel('Trade')
        plt.ylabel('Price/Value')
        plt.title('Kyle Model Simulation')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(trades_df['Cumulative_Profit'], label='Cumulative Profit', color='purple')
        plt.xlabel('Trade')
        plt.ylabel('Cumulative Profit')
        plt.title('Informed Trader Profitability')
        plt.legend()
        plt.show()

    
if __name__ == "__main__":
    # Parameters for the Kyle model
    mu = 100  # Mean fundamental value
    sigma_v = 10  # Standard deviation of fundamental value
    sigma_u = 5  # Standard deviation of noise trader demand
    lambda_param = 0.1  # Price impact parameter

    # Initialize the Kyle model
    kyle_model = KyleModel(mu, sigma_v, sigma_u, lambda_param)

    # Simulate trades
    n_trades = 25
    trades_df = kyle_model.simulate_multiple_trades(n_trades)

    # Backtest the informed trader strategy
    trades_df = kyle_model.backtest_informed_trader(trades_df)

    # Visualize the simulation results
    kyle_model.visualize_simulation(trades_df)