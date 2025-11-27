"""
GPU-accelerated HJB market making model with jump diffusion
"""

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags
from cupyx.scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy import stats
import time

class HJBGPUMarketMaker:
    """GPU-accelerated HJB market maker with jump diffusion"""

    def __init__(self, sigma=0.3, gamma=0.1, k=1.5, c=1.0, T=1.0,
                 S_min=50.0, S_max=150.0, n_S=100, n_t=100, I_max=5,
                 jump_intensity=0.1, jump_mean=0.0, jump_std=0.05):
        """
        Initialize HJB market maker

        Parameters:
        - sigma: volatility
        - gamma: risk aversion parameter
        - k: order book liquidity parameter
        - c: base intensity of order arrivals
        - T: time horizon
        - S_min, S_max: price grid bounds
        - n_S, n_t: grid sizes
        - I_max: maximum inventory
        - jump_intensity, jump_mean, jump_std: jump diffusion parameters
        """
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.c = c
        self.T = T
        self.S_min = S_min
        self.S_max = S_max
        self.n_S = n_S
        self.n_t = n_t
        self.I_max = I_max

        # Jump diffusion parameters
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        # Create grids
        self.S_grid = np.linspace(S_min, S_max, n_S)
        self.t_grid = np.linspace(0, T, n_t)
        self.dS = (S_max - S_min) / (n_S - 1)
        self.dt = T / (n_t - 1)

        # Inventory levels
        self.I_levels = np.arange(-I_max, I_max + 1)
        self.n_I = len(self.I_levels)

        # Value function θ(t, S, I)
        self.theta = np.zeros((n_t, n_S, self.n_I))

        # Terminal condition: θ(T, S, I) = 0
        self.theta[-1, :, :] = 0.0

        print(f"Initialized GPU HJB model: {n_S}x{n_t}x{self.n_I} grid")

    def _idx(self, I):
        """Get index for inventory level I"""
        return int(I + self.I_max)

    def solve_pde(self):
        """Solve the HJB PDE using backward induction with GPU acceleration"""
        print(f"Solving HJB PDE with jump diffusion (λ={self.jump_intensity:.6f}, μ={self.jump_mean:.6f}, σ={self.jump_std:.6f})...")

        start_time = time.time()

        # Pre-compute jump distribution for efficiency
        # For simplicity, use discrete approximation of normal jump
        n_jump_points = 5
        jump_values = np.linspace(self.jump_mean - 3*self.jump_std,
                                self.jump_mean + 3*self.jump_std, n_jump_points)
        jump_probs = stats.norm.pdf(jump_values, self.jump_mean, self.jump_std)
        jump_probs /= jump_probs.sum()  # Normalize

        # Convert to cupy arrays for GPU
        jump_values_cp = cp.asarray(jump_values)
        jump_probs_cp = cp.asarray(jump_probs)

        for t_idx in range(self.n_t - 2, -1, -1):  # Backward in time
            t = self.t_grid[t_idx]

            for i in range(self.n_I):
                I = self.I_levels[i]

                # Set up tridiagonal system for diffusion term
                a = np.zeros(self.n_S)  # Sub-diagonal
                b = np.zeros(self.n_S)  # Main diagonal
                c = np.zeros(self.n_S)  # Super-diagonal
                d = np.zeros(self.n_S)  # RHS

                # Interior points
                for j in range(1, self.n_S - 1):
                    S = self.S_grid[j]

                    # Diffusion coefficients
                    sigma_S = self.sigma * S
                    drift = 0.5 * sigma_S**2  # Risk-neutral drift

                    # Finite difference coefficients
                    a[j] = -0.5 * sigma_S**2 * self.dt / self.dS**2 + drift * self.dt / (2 * self.dS)
                    b[j] = 1 + sigma_S**2 * self.dt / self.dS**2 + self.dt * self._lambda_func(S, I)
                    c[j] = -0.5 * sigma_S**2 * self.dt / self.dS**2 - drift * self.dt / (2 * self.dS)

                    # RHS: current value + jump terms
                    d[j] = self.theta[t_idx + 1, j, i]

                    # Add jump diffusion terms
                    for k in range(n_jump_points):
                        jump_val = jump_values[k]
                        jump_prob = jump_probs[k]

                        # Interpolate value at S + jump
                        S_jump = S * np.exp(jump_val)
                        if S_jump <= self.S_min:
                            theta_jump = self.theta[t_idx + 1, 0, i]
                        elif S_jump >= self.S_max:
                            theta_jump = self.theta[t_idx + 1, -1, i]
                        else:
                            # Linear interpolation
                            j_jump = int((S_jump - self.S_min) / self.dS)
                            frac = (S_jump - self.S_grid[j_jump]) / self.dS
                            theta_jump = (1 - frac) * self.theta[t_idx + 1, j_jump, i] + \
                                       frac * self.theta[t_idx + 1, j_jump + 1, i]

                        d[j] += self.dt * self.jump_intensity * jump_prob * theta_jump

                # Boundary conditions
                # At S_min: ∂θ/∂S = 0 (reflecting boundary)
                b[0] = 1
                c[0] = -1
                d[0] = 0  # dθ/dS = 0

                # At S_max: ∂θ/∂S = 0 (reflecting boundary)
                a[-1] = 1
                b[-1] = -1
                c[-1] = 0
                d[-1] = 0  # dθ/dS = 0

                # Convert to cupy sparse matrix and solve on GPU
                diagonals = [cp.asarray(a[1:]), cp.asarray(b), cp.asarray(c[:-1])]
                offsets = [-1, 0, 1]
                A_cp = diags(diagonals, offsets, shape=(self.n_S, self.n_S), format='csr')
                d_cp = cp.asarray(d)

                # Solve on GPU
                theta_cp = spsolve(A_cp, d_cp)

                # Copy back to CPU
                self.theta[t_idx, :, i] = cp.asnumpy(theta_cp)

            # Print progress
            if t_idx % 20 == 0:
                elapsed = time.time() - start_time
                print(".1f")

        solve_time = time.time() - start_time
        print(".2f")
        print("PDE solved successfully with jump diffusion terms.")

    def _lambda_func(self, S, I):
        """Arrival rate function λ(S, I)"""
        # Optimal spreads from closed-form solution
        remaining_time = self.T  # Approximation
        bid_spread = ((2*I + 1) * self.gamma * self.sigma**2 * remaining_time / 2 +
                      np.log(1 + self.gamma/self.k) / self.gamma)
        ask_spread = ((1 - 2*I) * self.gamma * self.sigma**2 * remaining_time / 2 +
                      np.log(1 + self.gamma/self.k) / self.gamma)

        # Arrival rates
        lambda_b = self.c * np.exp(-self.k * bid_spread)
        lambda_a = self.c * np.exp(-self.k * ask_spread)

        return lambda_b + lambda_a

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
def run_gpu_hjb_simulation():
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

    # Create and solve the GPU HJB market maker
    mm = HJBGPUMarketMaker(sigma=sigma, gamma=gamma, k=k, c=c, T=T,
                          jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std)
    print("Solving HJB PDE on GPU...")
    mm.solve_pde()
    print("PDE solved. Plotting value function...")
    mm.plot_theta(save_path='hjb_gpu_value_function.png')

    print("Simulating market...")
    final_pnl = mm.simulate_market(save_path='hjb_gpu_simulation.png')
    print(f"Final PnL (including inventory liquidation): {final_pnl:.2f}")

    return mm

if __name__ == "__main__":
    mm = run_gpu_hjb_simulation()