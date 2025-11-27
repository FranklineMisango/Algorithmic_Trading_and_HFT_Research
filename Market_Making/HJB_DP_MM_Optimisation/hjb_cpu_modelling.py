import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class HJBMarketMaker:
    def __init__(self, sigma, gamma, k, c, T, 
                 I_max=10, S_min=80, S_max=120, 
                 dS=0.5, dt=0.01,
                 jump_intensity=0.1, jump_mean=0.0, jump_std=0.02):
        """
        Implementation of the Avellaneda-Stoikov model using direct HJB PDE solving
        with jump diffusion extensions
        
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
        - jump_intensity: intensity of jump arrivals (lambda)
        - jump_mean: mean of jump size distribution
        - jump_std: standard deviation of jump size distribution
        """
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.c = c
        self.T = T
        
        # Jump diffusion parameters
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        
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
    
    def _gauss_hermite_quadrature(self, n_points=5):
        """
        Generate Gauss-Hermite quadrature points and weights for jump integral
        
        Returns:
        - points: quadrature points
        - weights: quadrature weights
        """
        # Gauss-Hermite quadrature points and weights for n=5
        # These approximate the integral ∫ f(x) e^(-x²) dx
        if n_points == 5:
            points = np.array([0.0, 0.9585724646148185, -0.9585724646148185, 2.0201828704560856, -2.0201828704560856])
            weights = np.array([0.9453087204829419, 0.3936193231522412, 0.3936193231522412, 0.0882357465858919, 0.0882357465858919])
        else:
            # Fallback to simpler quadrature
            points = np.linspace(-3, 3, n_points)
            weights = np.ones(n_points) / n_points * np.sqrt(np.pi)
        
        return points, weights
    
    def _jump_integral(self, V_next, i, j, t_idx):
        """
        Compute the jump integral using Gauss-Hermite quadrature
        
        ∫ [V(t, S(1+y), I) - V(t, S, I)] f(y) dy
        
        where f(y) is the jump size density
        """
        S = self.S_grid[j]  # Current price (j is price index)
        integral = 0.0
        
        # Gauss-Hermite quadrature for normal distribution
        points, weights = self._gauss_hermite_quadrature()
        
        for point, weight in zip(points, weights):
            # Transform to jump size distribution
            jump_size = self.jump_mean + self.jump_std * point
            
            # New price after jump
            S_jump = S * (1 + jump_size)
            
            # Find grid index for S_jump (with boundary handling)
            if S_jump <= self.S_min:
                idx = 0
            elif S_jump >= self.S_max:
                idx = self.n_S - 1
            else:
                idx = int((S_jump - self.S_min) / self.dS)
                idx = np.clip(idx, 0, self.n_S - 1)
            
            # Add to integral: E[V(S') - V(S)] where S' = S*(1+J)
            integral += weight * (V_next[idx] - V_next[j])
        
        # Scale by sqrt(pi) for the Hermite weight and jump intensity
        return self.jump_intensity * integral * np.sqrt(np.pi)
    
    def _idx(self, I):
        """Convert inventory value to index in I_grid."""
        return np.where(self.I_grid == I)[0][0]
    
    def solve_pde(self):
        """Solve the HJB PDE for θ using finite differences and backward induction."""
        print(f"Solving HJB PDE with jump diffusion (λ={self.jump_intensity}, μ={self.jump_mean}, σ={self.jump_std})...")
        
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
                    
                    # Add jump diffusion term
                    jump_term = self._jump_integral(self.theta[t_idx+1, :, i], j, i, t_idx+1)
                    
                    d[j] = -self.theta[t_idx+1, j, i]/self.dt + source + jump_term
                
                # Boundary conditions (Neumann with proper handling)
                # At S_min: ∂θ/∂S = 0 (reflecting boundary)
                a[0] = 0
                b[0] = 1
                c[0] = -1
                d[0] = 0  # dθ/dS = 0
                
                # At S_max: ∂θ/∂S = 0 (reflecting boundary)  
                a[-1] = 1
                b[-1] = -1
                c[-1] = 0
                d[-1] = 0  # dθ/dS = 0
                
                # Solve tridiagonal system
                diagonals = [a[1:], b, c[:-1]]
                offsets = [-1, 0, 1]
                A = diags(diagonals, offsets, shape=(self.n_S, self.n_S))
                self.theta[t_idx, :, i] = spsolve(A.tocsc(), d)
            
            # Print progress
            if t_idx % 50 == 0:
                print(f"  Time step {t_idx}/{self.n_t-1} completed")
        
        print("PDE solved successfully with jump diffusion terms.")
    
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
    
    def plot_theta(self, save_path=None):
        """Plot the theta function for different inventory levels."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        mid_S_idx = len(self.S_grid) // 2
        
        # Plot theta at t=0 for different inventory levels
        for I in [-5, -2, 0, 2, 5]:
            if abs(I) <= self.I_max:
                i = self._idx(I)
                axes[0].plot(self.S_grid, self.theta[0, :, i], label=f'I = {I}')
        
        axes[0].set_title('Value Function θ(0, S, I) for Different Inventory Levels')
        axes[0].set_xlabel('Mid Price (S)')
        axes[0].set_ylabel('θ Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot theta at different times for I=0
        i_zero = self._idx(0)
        for t_idx in [0, self.n_t//4, self.n_t//2, 3*self.n_t//4, -1]:
            t = self.t_grid[t_idx]
            axes[1].plot(self.S_grid, self.theta[t_idx, :, i_zero], label=f't = {t:.2f}')
        
        axes[1].set_title('Value Function θ(t, S, I=0) at Different Times')
        axes[1].set_xlabel('Mid Price (S)')
        axes[1].set_ylabel('θ Value')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def simulate_market(self, n_steps=1000, S0=100.0, save_path=None):
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
        
        jump_count = 0
        
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
            
            # Simulate price movement with jump diffusion
            # Brownian motion component
            dW = np.random.normal(0, np.sqrt(dt_sim))
            drift = -0.5 * self.sigma**2  # Risk-neutral drift
            diffusion_term = self.sigma * dW
            
            # Jump component
            jump = 0.0
            if np.random.poisson(self.jump_intensity * dt_sim) > 0:
                jump_size = np.random.normal(self.jump_mean, self.jump_std)
                jump = jump_size
                jump_count += 1
                print(f"Jump {jump_count} occurred at t={t:.3f}: size={jump_size:.4f}")
            
            S[i+1] = S[i] * np.exp(drift * dt_sim + diffusion_term + jump)
        
        # Final quotes
        bid, ask = self.optimal_quotes(self.T, S[-1], I[-1])
        bid_prices[-1] = bid
        ask_prices[-1] = ask
        
        # Calculate final PnL including inventory liquidation
        final_pnl = pnl[-1] + I[-1] * S[-1]
        
        print(f"Simulation completed with {jump_count} jumps")
        
        # Plot results
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot 1: Price and quotes
        axes[0].plot(t_values, S, label='Mid Price', color='black', linewidth=1.5)
        axes[0].plot(t_values, bid_prices, label='Bid Price', color='green', alpha=0.7)
        axes[0].plot(t_values, ask_prices, label='Ask Price', color='red', alpha=0.7)
        axes[0].set_title('Price and Quotes (with Jump Diffusion)')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Inventory
        axes[1].plot(t_values, I, label='Inventory', color='blue', linewidth=1.5)
        axes[1].set_title('Inventory')
        axes[1].set_ylabel('Quantity')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: PnL
        axes[2].plot(t_values, pnl, label='Trading PnL', color='green', linewidth=1.5)
        # Add final PnL with inventory liquidation
        final_pnl_series = pnl.copy()
        for j in range(len(t_values)):
            final_pnl_series[j] += I[j] * S[j]
        axes[2].plot(t_values, final_pnl_series, label='Total PnL (with inventory)', color='purple', linewidth=1.5)
        axes[2].set_title('Profit and Loss')
        axes[2].set_ylabel('PnL')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Simulation plot saved to {save_path}")
        else:
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
    
    # Jump diffusion parameters
    jump_intensity = 0.5  # Jump arrival rate
    jump_mean = 0.0       # Mean jump size
    jump_std = 0.05       # Jump size volatility
    
    # Create and solve the HJB market maker
    mm = HJBMarketMaker(sigma=sigma, gamma=gamma, k=k, c=c, T=T,
                       jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std)
    print("Solving HJB PDE...")
    mm.solve_pde()
    print("PDE solved. Plotting value function...")
    mm.plot_theta(save_path='hjb_value_function.png')
    
    print("Simulating market...")
    final_pnl = mm.simulate_market(save_path='hjb_simulation.png')
    print(f"Final PnL (including inventory liquidation): {final_pnl:.2f}")
    
    return mm

if __name__ == "__main__":
    mm = run_hjb_simulation()
