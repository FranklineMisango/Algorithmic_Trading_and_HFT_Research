import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class KyleModel:
    def __init__(self, v_mean=0, v_std=1, noise_std=1, num_periods=1):
        """
        Initialize the Kyle model for market making with information disadvantage.
        
        Parameters:
        v_mean (float): Mean of the prior distribution of the asset value
        v_std (float): Standard deviation of the prior distribution of the asset value
        noise_std (float): Standard deviation of noise trading
        num_periods (int): Number of trading periods
        """
        self.v_mean = v_mean
        self.v_std = v_std
        self.noise_std = noise_std
        self.num_periods = num_periods
        
        # Calculate the optimal Kyle lambda (price impact parameter)
        self.lambda_ = self.calculate_lambda()
        
    def calculate_lambda(self):
        """
        Calculate the optimal Kyle lambda (price impact parameter).
        In the basic Kyle model, lambda = v_std / (2 * noise_std)
        """
        return self.v_std / (2 * self.noise_std)
    
    def optimal_insider_trading(self, v, p0):
        """
        Calculate the optimal trading strategy for the informed trader.
        
        Parameters:
        v (float): The true value of the asset known to the informed trader
        p0 (float): The initial price set by the market maker
        
        Returns:
        float: The optimal quantity for the informed trader to trade
        """
        # In the Kyle model, the insider trades x = (v - p0) / (2 * lambda)
        return (v - p0) / (2 * self.lambda_)
    
    def market_maker_price(self, order_flow):
        """
        Calculate the optimal price set by the market maker based on the observed order flow.
        
        Parameters:
        order_flow (float): The total order flow observed by the market maker
        
        Returns:
        float: The optimal price set by the market maker
        """
        # Market maker sets price as a linear function of order flow
        return self.v_mean + self.lambda_ * order_flow
    
    def calculate_optimal_market_maker_strategy(self, observed_order_flow):
        """
        Calculate the optimal strategy for the market maker given observed order flow.
        
        Parameters:
        observed_order_flow (float): The total order flow observed by the market maker
        
        Returns:
        tuple: (optimal_price, optimal_inventory)
        """
        # Determine optimal price based on order flow
        optimal_price = self.market_maker_price(observed_order_flow)
        
        # Market maker takes the opposite side of the order flow
        optimal_inventory = -observed_order_flow
        
        return optimal_price, optimal_inventory
    
    def simulate_single_period(self):
        """
        Simulate a single period of trading in the Kyle model.
        
        Returns:
        dict: Results of the simulation
        """
        # Draw the true asset value from the prior distribution
        v = np.random.normal(self.v_mean, self.v_std)
        
        # Initial price is the prior mean
        p0 = self.v_mean
        
        # Calculate the informed trader's optimal order
        x = self.optimal_insider_trading(v, p0)
        
        # Draw the noise trading
        u = np.random.normal(0, self.noise_std)
        
        # Total order flow
        order_flow = x + u
        
        # Market maker sets the price based on order flow
        p = self.market_maker_price(order_flow)
        
        # Calculate profits for the informed trader
        informed_profit = x * (v - p)
        
        # Calculate the market maker's inventory and profit
        mm_inventory = -order_flow  # Market maker takes the opposite side
        mm_profit = mm_inventory * (v - p)
        
        return {
            'true_value': v,
            'informed_trade': x,
            'noise_trade': u,
            'order_flow': order_flow,
            'price': p,
            'informed_profit': informed_profit,
            'mm_inventory': mm_inventory,
            'mm_profit': mm_profit
        }
    
    def simulate_multiple_periods(self, num_simulations=1000):
        """
        Simulate multiple periods of trading in the Kyle model.
        
        Parameters:
        num_simulations (int): Number of simulations to run
        
        Returns:
        list: List of dictionaries containing results from each simulation
        """
        results = []
        for _ in range(num_simulations):
            results.append(self.simulate_single_period())
        return results
    
    def analyze_results(self, results):
        """
        Analyze the results of the simulations.
        
        Parameters:
        results (list): List of dictionaries containing results
        
        Returns:
        dict: Summary statistics of the simulations
        """
        informed_profits = [r['informed_profit'] for r in results]
        mm_profits = [r['mm_profit'] for r in results]
        mm_inventories = [r['mm_inventory'] for r in results]
        prices = [r['price'] for r in results]
        true_values = [r['true_value'] for r in results]
        
        return {
            'avg_informed_profit': np.mean(informed_profits),
            'avg_mm_profit': np.mean(mm_profits),
            'avg_mm_inventory': np.mean(mm_inventories),
            'avg_price': np.mean(prices),
            'avg_true_value': np.mean(true_values),
            'price_efficiency': 1 - np.var([p - v for p, v in zip(prices, true_values)]) / self.v_std**2
        }
    
    def plot_results(self, results):
        """
        Plot the results of the simulations.
        
        Parameters:
        results (list): List of dictionaries containing results
        """
        informed_profits = [r['informed_profit'] for r in results]
        mm_profits = [r['mm_profit'] for r in results]
        
        plt.figure(figsize=(15, 10))
        
        # Plot profit distributions
        plt.subplot(2, 2, 1)
        plt.hist(informed_profits, bins=30, alpha=0.7)
        plt.axvline(np.mean(informed_profits), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(informed_profits):.2f}')
        plt.title('Informed Trader Profits')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.hist(mm_profits, bins=30, alpha=0.7)
        plt.axvline(np.mean(mm_profits), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(mm_profits):.2f}')
        plt.title('Market Maker Profits')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot order flow vs price
        plt.subplot(2, 2, 3)
        order_flows = [r['order_flow'] for r in results]
        prices = [r['price'] for r in results]
        plt.scatter(order_flows, prices, alpha=0.3)
        plt.title('Order Flow vs Price')
        plt.xlabel('Order Flow')
        plt.ylabel('Price')
        
        # Plot true value vs market maker's price
        plt.subplot(2, 2, 4)
        true_values = [r['true_value'] for r in results]
        plt.scatter(true_values, prices, alpha=0.3)
        plt.plot([min(true_values), max(true_values)], 
                 [min(true_values), max(true_values)], 'r--', 
                 label='Perfect Pricing')
        plt.title('True Value vs Market Maker Price')
        plt.xlabel('True Value')
        plt.ylabel('Market Maker Price')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def demonstrate_kyle_model():
    """
    Demonstrate how to use the Kyle model for market making.
    """
    # Initialize the Kyle model with parameters
    model = KyleModel(v_mean=100, v_std=10, noise_std=5)
    
    print(f"Kyle lambda (price impact parameter): {model.lambda_:.4f}")
    
    # Simulate a single period
    result = model.simulate_single_period()
    
    print("\nSingle Period Simulation:")
    print(f"True asset value: {result['true_value']:.2f}")
    print(f"Informed trader's order: {result['informed_trade']:.2f}")
    print(f"Noise trading: {result['noise_trade']:.2f}")
    print(f"Total order flow: {result['order_flow']:.2f}")
    print(f"Market maker's price: {result['price']:.2f}")
    print(f"Informed trader's profit: {result['informed_profit']:.2f}")
    print(f"Market maker's inventory: {result['mm_inventory']:.2f}")
    print(f"Market maker's profit: {result['mm_profit']:.2f}")
    
    # Demonstrate how a market maker would use this model in practice
    print("\nMarket Maker's Decision Process:")
    observed_order_flow = 2.5  # Example order flow
    optimal_price, optimal_inventory = model.calculate_optimal_market_maker_strategy(observed_order_flow)
    
    print(f"Observed order flow: {observed_order_flow:.2f}")
    print(f"Optimal price to set: {optimal_price:.2f}")
    print(f"Resulting inventory position: {optimal_inventory:.2f}")
    
    # Simulate multiple periods and analyze results
    print("\nRunning multiple simulations...")
    results = model.simulate_multiple_periods(num_simulations=10000)
    summary = model.analyze_results(results)
    
    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    # Plot the results
    model.plot_results(results)


# Run the demonstration
if __name__ == "__main__":
    demonstrate_kyle_model()