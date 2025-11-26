"""
Base classes for structured products.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class StructuredProduct(ABC):
    """Base class for structured equity products."""

    def __init__(self, spot: float, strike: float, maturity: float, r: float = 0.02, vol: float = 0.2):
        """
        Initialize structured product.

        Args:
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity in years
            r: Risk-free rate
            vol: Volatility
        """
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.r = r
        self.vol = vol

    @abstractmethod
    def payoff(self, path: np.ndarray) -> float:
        """Calculate payoff for a given price path."""
        pass

    def price_mc(self, n_sim: int = 10000, n_steps: int = 252, seed: Optional[int] = None) -> float:
        """
        Price using Monte Carlo simulation.

        Args:
            n_sim: Number of simulations
            n_steps: Number of time steps per year
            seed: Random seed for reproducibility

        Returns:
            Monte Carlo price
        """
        if seed is not None:
            np.random.seed(seed)

        dt = self.maturity / n_steps
        payoffs = []

        for _ in range(n_sim):
            # Generate GBM path
            path = [self.spot]
            for _ in range(n_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                path.append(path[-1] + dS)
            path = np.array(path)

            payoff = self.payoff(path)
            payoffs.append(payoff)

        price = np.exp(-self.r * self.maturity) * np.mean(payoffs)
        return price

    def calculate_greeks(self, n_sim: int = 10000, eps: float = 0.01) -> Dict[str, float]:
        """
        Calculate Greeks using finite differences.

        Args:
            n_sim: Number of simulations
            eps: Finite difference epsilon

        Returns:
            Dictionary of Greeks
        """
        base_price = self.price_mc(n_sim=n_sim)

        # Delta
        self.spot += eps
        price_up = self.price_mc(n_sim=n_sim)
        self.spot -= 2 * eps
        price_down = self.price_mc(n_sim=n_sim)
        self.spot += eps
        delta = (price_up - price_down) / (2 * eps)

        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (eps ** 2)

        # Vega
        vol_orig = self.vol
        self.vol = vol_orig + eps
        price_vega_up = self.price_mc(n_sim=n_sim)
        vega = (price_vega_up - base_price) / eps
        self.vol = vol_orig

        # Theta (approximate using time decay)
        maturity_orig = self.maturity
        self.maturity = maturity_orig - eps/365  # One day
        price_theta = self.price_mc(n_sim=n_sim)
        theta = (price_theta - base_price) / (eps/365)
        self.maturity = maturity_orig

        # Rho
        r_orig = self.r
        self.r = r_orig + eps
        price_rho = self.price_mc(n_sim=n_sim)
        rho = (price_rho - base_price) / eps
        self.r = r_orig

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    def plot_payoff_diagram(self, spot_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Plot payoff diagram at maturity.

        Args:
            spot_range: Range of spot prices to plot (min, max)
        """
        if spot_range is None:
            spot_range = (self.spot * 0.5, self.spot * 1.5)

        spots = np.linspace(spot_range[0], spot_range[1], 100)
        payoffs = []

        for s in spots:
            # Simple payoff assuming no path dependency
            payoffs.append(self.payoff(np.array([self.spot, s])))

        plt.figure(figsize=(10, 6))
        plt.plot(spots, payoffs, 'b-', linewidth=2)
        plt.axvline(x=self.spot, color='r', linestyle='--', alpha=0.7, label='Current Spot')
        plt.xlabel('Spot Price at Maturity')
        plt.ylabel('Payoff')
        plt.title(f'{self.__class__.__name__} Payoff Diagram')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def scenario_analysis(self, scenarios: Dict[str, Dict]) -> Dict[str, float]:
        """
        Perform scenario analysis.

        Args:
            scenarios: Dictionary of scenario names to parameter changes

        Returns:
            Dictionary of scenario prices
        """
        results = {}
        original_params = {k: getattr(self, k) for k in ['spot', 'vol', 'r']}

        for scenario_name, changes in scenarios.items():
            # Apply changes
            for param, change in changes.items():
                if isinstance(change, (int, float)):
                    setattr(self, param, original_params[param] * (1 + change))
                else:
                    setattr(self, param, change)

            results[scenario_name] = self.price_mc()

            # Reset parameters
            for param, value in original_params.items():
                setattr(self, param, value)

        return results