"""
Barrier option implementation.
"""

import numpy as np
from enum import Enum
from typing import Optional
from .base_product import StructuredProduct


class BarrierType(Enum):
    UP_AND_IN = "up_and_in"
    UP_AND_OUT = "up_and_out"
    DOWN_AND_IN = "down_and_in"
    DOWN_AND_OUT = "down_and_out"


class KnockDirection(Enum):
    UP = "up"
    DOWN = "down"


class BarrierOption(StructuredProduct):
    """
    Barrier option that becomes active (knock-in) or inactive (knock-out)
    when the underlying reaches a barrier level.
    """

    def __init__(self,
                 spot: float,
                 strike: float,
                 barrier: float,
                 maturity: float,
                 option_type: str = 'call',  # 'call' or 'put'
                 barrier_type: BarrierType = BarrierType.UP_AND_IN,
                 rebate: float = 0.0,  # Rebate paid if barrier hit
                 r: float = 0.02,
                 vol: float = 0.2):
        """
        Initialize barrier option.

        Args:
            spot: Current spot price
            strike: Strike price
            barrier: Barrier level
            maturity: Time to maturity in years
            option_type: 'call' or 'put'
            barrier_type: Type of barrier option
            rebate: Rebate amount paid if barrier is hit
            r: Risk-free rate
            vol: Volatility
        """
        super().__init__(spot, strike, maturity, r, vol)
        self.option_type = option_type.lower()
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.rebate = rebate

        # Determine knock direction and in/out
        if barrier_type in [BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT]:
            self.knock_direction = KnockDirection.UP
        else:
            self.knock_direction = KnockDirection.DOWN

        self.is_knock_in = barrier_type in [BarrierType.UP_AND_IN, BarrierType.DOWN_AND_IN]

    def payoff(self, path: np.ndarray) -> float:
        """
        Calculate payoff for a given price path.

        Args:
            path: Price path array

        Returns:
            Payoff amount
        """
        final_price = path[-1]

        # Check if barrier was hit
        if self.knock_direction == KnockDirection.UP:
            barrier_hit = np.any(path >= self.barrier)
        else:
            barrier_hit = np.any(path <= self.barrier)

        # Calculate vanilla option payoff
        if self.option_type == 'call':
            vanilla_payoff = max(final_price - self.strike, 0)
        else:
            vanilla_payoff = max(self.strike - final_price, 0)

        # Apply barrier logic
        if self.is_knock_in:
            # Knock-in: only pays if barrier hit
            if barrier_hit:
                payoff = vanilla_payoff
            else:
                payoff = self.rebate
        else:
            # Knock-out: pays unless barrier hit
            if not barrier_hit:
                payoff = vanilla_payoff
            else:
                payoff = self.rebate

        return np.exp(-self.r * self.maturity) * payoff

    def barrier_probability(self, n_sim: int = 10000) -> float:
        """
        Calculate probability of barrier being hit.

        Args:
            n_sim: Number of simulations

        Returns:
            Probability of barrier hit
        """
        dt = self.maturity / 252
        hit_count = 0

        np.random.seed(42)
        for _ in range(n_sim):
            path = [self.spot]
            hit = False

            for _ in range(252):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                new_price = path[-1] + dS
                path.append(new_price)

                if self.knock_direction == KnockDirection.UP:
                    if new_price >= self.barrier:
                        hit = True
                        break
                else:
                    if new_price <= self.barrier:
                        hit = True
                        break

            if hit:
                hit_count += 1

        return hit_count / n_sim

    def plot_barrier_paths(self, n_paths: int = 10) -> None:
        """Plot sample paths showing barrier hits."""
        dt = self.maturity / 252
        time_points = np.linspace(0, self.maturity, 253)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))

        hit_paths = []
        no_hit_paths = []

        np.random.seed(42)
        for _ in range(n_paths):
            path = [self.spot]
            hit = False
            hit_time = None

            for i in range(252):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                new_price = path[-1] + dS
                path.append(new_price)

                if not hit:
                    if self.knock_direction == KnockDirection.UP:
                        if new_price >= self.barrier:
                            hit = True
                            hit_time = time_points[i+1]
                    else:
                        if new_price <= self.barrier:
                            hit = True
                            hit_time = time_points[i+1]

            if hit:
                hit_paths.append((path, hit_time))
            else:
                no_hit_paths.append(path)

        # Plot hit paths
        for path, hit_time in hit_paths[:5]:  # Show first 5 hit paths
            plt.plot(time_points, path, 'r-', alpha=0.7, linewidth=1)
            plt.axvline(x=hit_time, color='red', linestyle='--', alpha=0.5)

        # Plot no-hit paths
        for path in no_hit_paths[:5]:  # Show first 5 no-hit paths
            plt.plot(time_points, path, 'b-', alpha=0.7, linewidth=1)

        # Plot barrier
        plt.axhline(y=self.barrier, color='black', linestyle='-', linewidth=2, label='Barrier')

        plt.xlabel('Time (years)')
        plt.ylabel('Price')
        plt.title(f'Barrier Option Sample Paths ({self.barrier_type.value})')
        plt.legend(['Barrier Hit', 'No Barrier Hit', 'Barrier Level'])
        plt.grid(True, alpha=0.3)
        plt.show()