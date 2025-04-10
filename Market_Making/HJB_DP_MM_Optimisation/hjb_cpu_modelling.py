import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class HJBMarketMaker:
    def __init__(self, sigma, gamma, k, c, T, 
                 I_max=10, S_min=80, S_max=120, 
                 dS=0.5, dt=0.01):
        """
        Implementation of the Avellaneda-Stoikov model using direct HJB PDE solving
        
        Parameters:
        - sigma: volatility of mid-price process
        - gamma: risk aversion coefficient
        - k: order book liquidity parameter
        - c: intensity of order arrivals
        - T: time horizon
        - I_max: maximum inventory (grid extends from -I_max to I_max)
        - S_min, S_max: price grid boundaries
        - dS: price grid step size
        - dt: time step size
        """
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.c = c
        self.T = T
        
        # Grid parameters
        self.I_max = I_max
        self.S_min = S_min
        self.S_max = S_max
        self.dS = dS
        self.dt = dt
        
        # Create grids
        self.I_grid = np.arange(-I_max, I_max+1)
        self.S_grid = np.arange(S_min, S_max+dS, dS)
        self.t_grid = np.arange(0, T+dt, dt)
        
        self.n_I = len(self.I_grid)
        self.n_S = len(self.S_grid)
        self.n_t = len(self.t_grid)
        
        # Initialize value function and theta
        self.theta = np.zeros((self.n_t, self.n_S, self.n_I))
        
        # Terminal condition: θ(T, S, I) = I·S
        for i, I in enumerate(self.I_grid):
            for j, S in enumerate(self.S_grid):
                self.theta[-1, j, i] = I * S
    
    def _idx(self, I):
        """Convert inventory value to index in I_grid."""
        return np.where(self.I_grid == I)[0][0]
    
    def solve_pde(self):
        """Solve the HJB PDE for θ using finite differences and backward induction."""
        # Solve backward in time
        for t_idx in range(self.n_t-2, -1, -1):
            t = self.t_grid[t_idx]
            remaining_time = self.T - t
            
            # For each inventory level
            for i, I in enumerate(self.I_grid):
                # Build the tridiagonal system for implicit finite difference
                a = np.zeros(self.n_S)  # subdiagonal
                b = np.zeros(self.n_S)  # diagonal
                c = np.zeros(self.n_S)  # superdiagonal
                d = np.zeros(self.n_S)  # right-hand side
                
                # Interior points
                for j in range(1, self.n_S-1):
                    S = self.S_grid[j]
                    
                    # Finite difference coefficients for ∂²θ/∂S²
                    a[j] = self.sigma**2 * S**2 / (2 * self.dS**2)
                    c[j] = self.sigma**2 * S**2 / (2 * self.dS**2)
                    b[j] = -a[j] - c[j] - 1/self.dt
                    
                    # Calculate optimal bid/ask spreads
                    if I < self.I_max:
                        Q_b = S - (2*I + 1) * self.gamma * self.sigma**2 * remaining_time / 2
                        delta_b = (2*I + 1) * self.gamma * self.sigma**2 * remaining_time / 2 + np.log(1 + self.gamma/self.k) / self.gamma
                        lambda_b = self.c * np.exp(-self.k * delta_b)
                    else:
                        lambda_b = 0
                    
                    if I > -self.I_max:
                        Q_a = S - (2*I - 1) * self.gamma * self.sigma**2 * remaining_time / 2
                        delta_a = (1 - 2*I) * self.gamma * self.sigma**2 * remaining_time / 2 + np.log(1 + self.gamma/self.k) / self.gamma
                        lambda_a = self.c * np.exp(-self.k * delta_a)
                    else:
                        lambda_a = 0
                    
                    # Source term contributions
                    source = 0
                    if I < self.I_max:
                        source += lambda_b * (self.theta[t_idx+1, j, self._idx(I+1)] - self.theta[t_idx+1, j, i])
                    if I > -self.I_max:
                        source += lambda_a * (self.theta[t_idx+1, j, self._idx(I-1)] - self.theta[t_idx+1, j, i])
                    
                    d[j] = -self.theta[t_idx+1, j, i]/self.dt + source
                
                # Boundary conditions (simplified Neumann)
                a[0] = 0
                b[0] = 1
                c[0] = 0
                d[0] = self.theta[t_idx+1, 0, i]
                
                a[-1] = 0
                b[-1] = 1
                c[-1] = 0
                d[-1] = self.theta[t_idx+1, -1, i]
                
                # Solve tridiagonal system
                diagonals = [a[1:], b, c[:-1]]
                offsets = [-1, 0, 1]
                A = diags(diagonals, offsets, shape=(self.n_S, self.n_S))
                self.theta[t_idx, :, i] = spsolve(A, d)
    
    def optimal_quotes(self, t, S, I):
        """Calculate optimal bid and ask quotes for given state."""
        if t >= self.T:
            return S, S  # No spread at terminal time
        
        remaining_time = self.T - t
        
        # Optimal spreads from the closed-form solution
        bid_spread = ((2*I + 1) * self.gamma * self.sigma**2 * remaining_time / 2 + 
                      np.log(1 + self.gamma/self.k) / self.gamma)
        ask_spread = ((1 - 2*I) * self.gamma * self.sigma**2 * remaining_time / 2 + 
                      np.log(1 + self.gamma/self.k) / self.gamma)
        
        # Optimal quotes
        bid = S - bid_spread
        ask = S + ask_spread
        
        return bid, ask
    
    def indifference_prices(self, t, S, I):
        """Calculate indifference bid, ask, and mid prices."""
        remaining_time = self.T - t
        
        Q_b = S - (2*I + 1) * self.gamma * self.sigma**2 * remaining_time / 2
        Q_a = S - (2*I - 1) * self.gamma * self.sigma**2 * remaining_time / 2
        Q_m = S - I * self.gamma * self.sigma**2 * remaining_time
        
        return Q_b, Q_a, Q_m
    
    def plot_theta(self):
        """Plot the theta function for different inventory levels."""
        plt.figure(figsize=(12, 8))
        mid_S_idx = len(self.S_grid) // 2
        
        # Plot theta at t=0 for different inventory levels
        for I in [-5, -2, 0, 2, 5]:
            if abs(I) <= self.I_max:
                i = self._idx(I)
                plt.plot(self.S_grid, self.theta[0, :, i], label=f'I = {I}')
        
        plt.title('Value Function θ(0, S, I) for Different Inventory Levels')
        plt.xlabel('Mid Price (S)')
        plt.ylabel('θ Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot theta at different times for I=0
        plt.figure(figsize=(12, 8))
        i_zero = self._idx(0)
        for t_idx in [0, self.n_t//4, self.n_t//2, 3*self.n_t//4, -1]:
            t = self.t_grid[t_idx]
            plt.plot(self.S_grid, self.theta[t_idx, :, i_zero], label=f't = {t:.2f}')
        
        plt.title('Value Function θ(t, S, I=0) at Different Times')
        plt.xlabel('Mid Price (S)')
        plt.ylabel('θ Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def simulate_market(self, n_steps=1000, S0=100.0):
        """Simulate market dynamics using the HJB solution."""
        dt_sim = self.T / n_steps
        t_values = np.linspace(0, self.T, n_steps+1)
        
        # Simulation arrays
        S = np.zeros(n_steps+1)
        I = np.zeros(n_steps+1, dtype=int)
        pnl = np.zeros(n_steps+1)
        bid_prices = np.zeros(n_steps+1)
        ask_prices = np.zeros(n_steps+1)
        
        S[0] = S0
        
        for i in range(n_steps):
            t = t_values[i]
            
            # Get optimal quotes
            bid, ask = self.optimal_quotes(t, S[i], I[i])
            bid_prices[i] = bid
            ask_prices[i] = ask
            
            # Calculate arrival rates
            bid_spread = S[i] - bid
            ask_spread = ask - S[i]
            lambda_b = self.c * np.exp(-self.k * bid_spread)
            lambda_a = self.c * np.exp(-self.k * ask_spread)
            
            # Simulate order arrivals
            bid_hit = np.random.poisson(lambda_b * dt_sim)
            ask_lift = np.random.poisson(lambda_a * dt_sim)
            
            # Limit to feasible inventory changes
            if I[i] + bid_hit > self.I_max:
                bid_hit = self.I_max - I[i]
            if I[i] - ask_lift < -self.I_max:
                ask_lift = I[i] + self.I_max
            
            # Update inventory
            I[i+1] = I[i] + bid_hit - ask_lift
            
            # Update PnL
            pnl[i+1] = pnl[i] + ask_lift * ask - bid_hit * bid
            
            # Simulate price movement
            S[i+1] = S[i] * np.exp(self.sigma * np.sqrt(dt_sim) * np.random.normal())
        
        # Final quotes
        bid, ask = self.optimal_quotes(self.T, S[-1], I[-1])
        bid_prices[-1] = bid
        ask_prices[-1] = ask
        
        # Calculate final PnL including inventory liquidation
        final_pnl = pnl[-1] + I[-1] * S[-1]
        
        # Plot results
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot 1: Price and quotes
        axes[0].plot(t_values, S, label='Mid Price', color='black')
        axes[0].plot(t_values, bid_prices, label='Bid Price', color='green')
        axes[0].plot(t_values, ask_prices, label='Ask Price', color='red')
        axes[0].set_title('Price and Quotes')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Inventory
        axes[1].plot(t_values, I, label='Inventory', color='blue')
        axes[1].set_title('Inventory')
        axes[1].set_ylabel('Quantity')
        axes[1].axhline(y=0, color='black', linestyle='--')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: PnL
        axes[2].plot(t_values, pnl, label='Trading PnL', color='green')
        # Add final PnL with inventory liquidation
        final_pnl_series = pnl.copy()
        for j in range(len(t_values)):
            final_pnl_series[j] += I[j] * S[j]
        axes[2].plot(t_values, final_pnl_series, label='Total PnL (with inventory)', color='purple')
        axes[2].set_title('Profit and Loss')
        axes[2].set_ylabel('PnL')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return final_pnl


# Example usage
def run_hjb_simulation():
    # Parameters
    sigma = 0.3     # Volatility
    gamma = 0.1     # Risk aversion
    k = 1.5         # Order book liquidity parameter
    c = 1.0         # Base intensity of order arrivals
    T = 1.0         # Time horizon (1 day)
    
    # Create and solve the HJB market maker
    mm = HJBMarketMaker(sigma=sigma, gamma=gamma, k=k, c=c, T=T)
    print("Solving HJB PDE...")
    mm.solve_pde()
    print("PDE solved. Plotting value function...")
    mm.plot_theta()
    
    print("Simulating market...")
    final_pnl = mm.simulate_market()
    print(f"Final PnL (including inventory liquidation): {final_pnl:.2f}")
    
    return mm

if __name__ == "__main__":
    mm = run_hjb_simulation()
