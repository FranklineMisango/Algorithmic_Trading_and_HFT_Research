"""
Range accrual note implementation.
"""

import numpy as np
from typing import List, Tuple
from .base_product import StructuredProduct


class RangeAccrual(StructuredProduct):
    """
    Range accrual note that pays coupons based on the underlying asset
    staying within predefined ranges during observation periods.
    """

    def __init__(self,
                 spot: float,
                 strike: float,
                 ranges: List[Tuple[float, float]],
                 coupon_rates: List[float],
                 maturity: float,
                 observation_frequency: int = 252,  # Daily
                 capital_protection: float = 1.0,
                 r: float = 0.02,
                 vol: float = 0.2):
        """
        Initialize range accrual note.

        Args:
            spot: Current spot price
            strike: Strike price (typically spot)
            ranges: List of (lower, upper) bounds for each period
            coupon_rates: Coupon rates for each range period
            maturity: Time to maturity in years
            observation_frequency: Number of observations per year
            capital_protection: Capital protection level
            r: Risk-free rate
            vol: Volatility
        """
        super().__init__(spot, strike, maturity, r, vol)
        self.ranges = ranges
        self.coupon_rates = coupon_rates
        self.observation_frequency = observation_frequency
        self.capital_protection = capital_protection

        # Calculate observation dates
        n_periods = len(ranges)
        self.period_length = maturity / n_periods
        self.observation_dates = np.linspace(0, maturity, n_periods + 1)[1:]

    def payoff(self, path: np.ndarray) -> float:
        """
        Calculate payoff for a given price path.

        Args:
            path: Price path array

        Returns:
            Payoff amount
        """
        dt = self.maturity / (len(path) - 1)
        total_coupon = 0

        # Calculate coupons for each period
        for i, (lower, upper) in enumerate(self.ranges):
            period_start = int((i * self.period_length) / dt)
            period_end = int(((i + 1) * self.period_length) / dt)

            if period_end >= len(path):
                period_end = len(path) - 1

            period_prices = path[period_start:period_end + 1]

            # Count days in range
            days_in_range = np.sum((period_prices >= lower) & (period_prices <= upper))
            total_days = len(period_prices)

            # Coupon for this period
            if total_days > 0:
                accrual_rate = days_in_range / total_days
                period_coupon = self.coupon_rates[i] * accrual_rate * self.period_length
                total_coupon += period_coupon

        # Final payoff: capital protection + accrued coupons
        payoff = self.capital_protection + total_coupon

        return np.exp(-self.r * self.maturity) * payoff

    def calculate_accrual_probabilities(self, n_sim: int = 10000) -> List[float]:
        """
        Calculate probability of full accrual for each period.

        Args:
            n_sim: Number of simulations

        Returns:
            List of full accrual probabilities
        """
        dt = self.maturity / 252
        full_accrual_counts = [0] * len(self.ranges)

        np.random.seed(42)
        for _ in range(n_sim):
            path = [self.spot]
            for _ in range(252):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                path.append(path[-1] + dS)

            for i, (lower, upper) in enumerate(self.ranges):
                period_start = int((i * self.period_length) / dt)
                period_end = int(((i + 1) * self.period_length) / dt)

                if period_end >= len(path):
                    period_end = len(path) - 1

                period_prices = path[period_start:period_end + 1]
                in_range = np.all((period_prices >= lower) & (period_prices <= upper))

                if in_range:
                    full_accrual_counts[i] += 1

        return [count / n_sim for count in full_accrual_counts]

    def plot_range_accrual_efficiency(self, n_sim: int = 10000) -> None:
        """Plot the efficiency of coupon accrual."""
        dt = self.maturity / 252
        accrual_rates = []

        np.random.seed(42)
        for _ in range(n_sim):
            path = [self.spot]
            for _ in range(252):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                path.append(path[-1] + dS)

            total_accrual = 0
            for i, (lower, upper) in enumerate(self.ranges):
                period_start = int((i * self.period_length) / dt)
                period_end = int(((i + 1) * self.period_length) / dt)

                if period_end >= len(path):
                    period_end = len(path) - 1

                period_prices = path[period_start:period_end + 1]
                days_in_range = np.sum((period_prices >= lower) & (period_prices <= upper))
                total_days = len(period_prices)

                if total_days > 0:
                    accrual_rate = days_in_range / total_days
                    total_accrual += accrual_rate * self.coupon_rates[i] * self.period_length

            accrual_rates.append(total_accrual)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(accrual_rates, bins=30, alpha=0.7, color='green', density=True)
        plt.axvline(x=np.mean(accrual_rates), color='red', linestyle='--', label='Expected Accrual')
        plt.xlabel('Total Accrued Coupon')
        plt.ylabel('Density')
        plt.title('Accrual Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Plot range bounds
        time_points = np.linspace(0, self.maturity, len(self.ranges))
        lower_bounds = [r[0] for r in self.ranges]
        upper_bounds = [r[1] for r in self.ranges]

        plt.fill_between(time_points, lower_bounds, upper_bounds, alpha=0.3, color='blue', label='Accrual Ranges')
        plt.axhline(y=self.spot, color='red', linestyle='--', label='Current Spot')
        plt.xlabel('Time (years)')
        plt.ylabel('Price Level')
        plt.title('Accrual Ranges Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()