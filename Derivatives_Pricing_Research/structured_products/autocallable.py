"""
Autocallable structured product implementation.
"""

import numpy as np
from typing import List, Optional
from .base_product import StructuredProduct


class Autocallable(StructuredProduct):
    """
    Autocallable note with periodic coupons and early redemption features.

    The note pays periodic coupons and can be called early if the underlying
    reaches certain levels. If not called, it pays a final coupon at maturity
    plus potential capital protection.
    """

    def __init__(self,
                 spot: float,
                 strike: float,
                 barrier: float,
                 coupon: float,
                 maturity: float,
                 autocall_levels: List[float],
                 coupon_frequency: int = 4,  # Quarterly
                 capital_protection: float = 1.0,  # 100% protection
                 r: float = 0.02,
                 vol: float = 0.2):
        """
        Initialize autocallable note.

        Args:
            spot: Current spot price
            strike: Strike price (typically spot)
            barrier: Barrier level for capital protection
            coupon: Annual coupon rate
            maturity: Time to maturity in years
            autocall_levels: List of autocall barrier levels (one per period)
            coupon_frequency: Number of coupon payments per year
            capital_protection: Capital protection level (0.9 = 90%)
            r: Risk-free rate
            vol: Volatility
        """
        super().__init__(spot, strike, maturity, r, vol)
        self.barrier = barrier
        self.coupon = coupon
        self.autocall_levels = autocall_levels
        self.coupon_frequency = coupon_frequency
        self.capital_protection = capital_protection

        # Calculate observation dates
        self.observation_dates = np.linspace(0, maturity, len(autocall_levels) + 1)[1:]

    def payoff(self, path: np.ndarray) -> float:
        """
        Calculate payoff for a given price path.

        Args:
            path: Price path array

        Returns:
            Payoff amount
        """
        dt = self.maturity / (len(path) - 1)
        coupon_payment = self.coupon / self.coupon_frequency
        total_payoff = 0

        # Check for early redemption
        for i, obs_date in enumerate(self.observation_dates):
            obs_idx = int(obs_date / dt)
            if obs_idx >= len(path):
                obs_idx = len(path) - 1

            price_at_obs = path[obs_idx]

            if price_at_obs >= self.autocall_levels[i]:
                # Early redemption
                time_to_obs = obs_date
                total_payoff = np.exp(-self.r * time_to_obs) * (1 + coupon_payment * (i + 1))
                return total_payoff

        # Not called early - check final payoff
        final_price = path[-1]
        final_coupon = self.coupon / self.coupon_frequency

        if final_price >= self.barrier:
            # Full capital + final coupon
            total_payoff = self.capital_protection + final_coupon
        else:
            # Protected capital + final coupon (but capital may be eroded)
            capital_return = min(final_price / self.strike, self.capital_protection)
            total_payoff = capital_return + final_coupon

        return np.exp(-self.r * self.maturity) * total_payoff

    def price_mc(self, n_sim: int = 10000, n_steps: Optional[int] = None, seed: Optional[int] = None) -> float:
        """
        Price using Monte Carlo with optimized time stepping for autocallables.

        Args:
            n_sim: Number of simulations
            n_steps: Number of time steps (defaults to observation frequency)
            seed: Random seed

        Returns:
            Monte Carlo price
        """
        if n_steps is None:
            n_steps = int(self.maturity * 365)  # Daily steps for accuracy

        return super().price_mc(n_sim=n_sim, n_steps=n_steps, seed=seed)

    def calculate_autocall_probability(self, n_sim: int = 10000) -> List[float]:
        """
        Calculate probability of autocall at each observation date.

        Args:
            n_sim: Number of simulations

        Returns:
            List of autocall probabilities for each period
        """
        dt = self.maturity / 252  # Daily
        autocall_counts = [0] * len(self.observation_dates)

        np.random.seed(42)
        for _ in range(n_sim):
            path = [self.spot]
            for _ in range(252):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                path.append(path[-1] + dS)

            for i, obs_date in enumerate(self.observation_dates):
                obs_idx = int(obs_date / dt)
                if obs_idx < len(path) and path[obs_idx] >= self.autocall_levels[i]:
                    autocall_counts[i] += 1
                    break  # Only count first autocall

        return [count / n_sim for count in autocall_counts]

    def plot_autocall_probabilities(self) -> None:
        """Plot autocall probabilities over time."""
        probs = self.calculate_autocall_probability()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(probs) + 1), probs, alpha=0.7, color='skyblue')
        plt.xlabel('Observation Period')
        plt.ylabel('Autocall Probability')
        plt.title('Autocall Probabilities by Period')
        plt.xticks(range(1, len(probs) + 1))
        plt.grid(True, alpha=0.3)
        plt.show()