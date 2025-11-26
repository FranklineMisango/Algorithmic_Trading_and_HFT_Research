"""
Reverse convertible structured product implementation.
"""

import numpy as np
from typing import Dict
from .base_product import StructuredProduct


class ReverseConvertible(StructuredProduct):
    """
    Reverse convertible note that provides capital protection and coupon payments,
    but exposes investors to downside risk of the underlying asset.
    """

    def __init__(self,
                 spot: float,
                 strike: float,
                 coupon: float,
                 maturity: float,
                 capital_protection: float = 1.0,
                 participation_rate: float = 1.0,
                 r: float = 0.02,
                 vol: float = 0.2):
        """
        Initialize reverse convertible.

        Args:
            spot: Current spot price
            strike: Strike price (typically spot)
            coupon: Annual coupon rate
            maturity: Time to maturity in years
            capital_protection: Capital protection level (0.9 = 90%)
            participation_rate: Upside participation rate (1.0 = 100%)
            r: Risk-free rate
            vol: Volatility
        """
        super().__init__(spot, strike, maturity, r, vol)
        self.coupon = coupon
        self.capital_protection = capital_protection
        self.participation_rate = participation_rate

    def payoff(self, path: np.ndarray) -> float:
        """
        Calculate payoff for a given price path.

        Returns the maximum of:
        1. Capital protection + coupon
        2. Participation in upside + coupon

        But if the underlying falls below strike, investor gets the underlying value + coupon.

        Args:
            path: Price path array

        Returns:
            Payoff amount
        """
        final_price = path[-1]

        # Coupon payment
        coupon_payment = self.coupon * self.maturity

        # Capital protection payoff
        protected_payoff = self.capital_protection + coupon_payment

        # Upside participation payoff
        upside = max(0, final_price - self.strike) * self.participation_rate
        participation_payoff = self.strike + upside + coupon_payment

        # Reverse convertible payoff: if final price >= strike, get participation
        # if final price < strike, get final price + coupon (no capital protection)
        if final_price >= self.strike:
            payoff = participation_payoff
        else:
            payoff = final_price + coupon_payment

        return np.exp(-self.r * self.maturity) * payoff

    def calculate_downside_risk(self, n_sim: int = 10000) -> Dict[str, float]:
        """
        Calculate downside risk metrics.

        Args:
            n_sim: Number of simulations

        Returns:
            Dictionary of risk metrics
        """
        dt = self.maturity / 252
        payoffs = []

        np.random.seed(42)
        for _ in range(n_sim):
            path = [self.spot]
            for _ in range(252):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                path.append(path[-1] + dS)

            payoff = self.payoff(np.array(path))
            payoffs.append(payoff)

        payoffs = np.array(payoffs)
        capital_protected = self.capital_protection * np.exp(-self.r * self.maturity)

        return {
            'expected_payoff': np.mean(payoffs),
            'payoff_volatility': np.std(payoffs),
            'prob_capital_loss': np.mean(payoffs < capital_protected),
            'worst_case_payoff': np.min(payoffs),
            'best_case_payoff': np.max(payoffs)
        }

    def plot_payoff_distribution(self, n_sim: int = 10000) -> None:
        """Plot the distribution of payoffs."""
        dt = self.maturity / 252
        payoffs = []

        np.random.seed(42)
        for _ in range(n_sim):
            path = [self.spot]
            for _ in range(252):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = self.r * path[-1] * dt + self.vol * path[-1] * dW
                path.append(path[-1] + dS)

            payoff = self.payoff(np.array(path))
            payoffs.append(payoff)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(payoffs, bins=50, alpha=0.7, color='blue', density=True)
        plt.axvline(x=self.capital_protection, color='r', linestyle='--', label='Capital Protection')
        plt.xlabel('Payoff')
        plt.ylabel('Density')
        plt.title('Payoff Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot(payoffs, vert=False)
        plt.xlabel('Payoff')
        plt.title('Payoff Box Plot')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()