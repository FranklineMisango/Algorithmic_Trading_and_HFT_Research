"""
Pricing engine with advanced pricing methods and risk calculations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
import pandas as pd


class PricingEngine:
    """Advanced pricing engine for structured products."""

    @staticmethod
    def black_scholes_price(spot: float,
                           strike: float,
                           maturity: float,
                           r: float,
                           vol: float,
                           option_type: str = 'call') -> float:
        """
        Black-Scholes pricing for European options.

        Args:
            spot: Spot price
            strike: Strike price
            maturity: Time to maturity
            r: Risk-free rate
            vol: Volatility
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * maturity) / (vol * np.sqrt(maturity))
        d2 = d1 - vol * np.sqrt(maturity)

        if option_type.lower() == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return price

    @staticmethod
    def calculate_var(payoffs: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk.

        Args:
            payoffs: Array of payoff values
            confidence_level: Confidence level (0.95 = 95%)

        Returns:
            VaR value
        """
        return -np.percentile(payoffs, (1 - confidence_level) * 100)

    @staticmethod
    def calculate_cvar(payoffs: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            payoffs: Array of payoff values
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        var = PricingEngine.calculate_var(payoffs, confidence_level)
        losses = -payoffs
        return -np.mean(losses[losses >= -var])

    @staticmethod
    def heston_model_price(spot: float,
                          strike: float,
                          maturity: float,
                          r: float,
                          kappa: float,
                          theta: float,
                          sigma: float,
                          rho: float,
                          v0: float,
                          option_type: str = 'call',
                          n_sim: int = 10000) -> float:
        """
        Price option using Heston stochastic volatility model.

        Args:
            spot: Spot price
            strike: Strike price
            maturity: Time to maturity
            r: Risk-free rate
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of variance
            rho: Correlation between spot and variance
            v0: Initial variance
            option_type: 'call' or 'put'
            n_sim: Number of simulations

        Returns:
            Option price
        """
        dt = maturity / 252
        payoffs = []

        np.random.seed(42)
        for _ in range(n_sim):
            s = spot
            v = v0

            for _ in range(252):
                # Correlated random numbers
                z1 = np.random.normal(0, 1)
                z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)

                # Heston dynamics
                ds = r * s * dt + np.sqrt(v) * s * z1 * np.sqrt(dt)
                dv = kappa * (theta - v) * dt + sigma * np.sqrt(v) * z2 * np.sqrt(dt)

                s = max(s + ds, 0.01)  # Floor at 0.01 to avoid negative prices
                v = max(v + dv, 0.01)  # Floor at 0.01

            if option_type.lower() == 'call':
                payoff = max(s - strike, 0)
            else:
                payoff = max(strike - s, 0)

            payoffs.append(payoff)

        return np.exp(-r * maturity) * np.mean(payoffs)

    @staticmethod
    def implied_volatility(market_price: float,
                          spot: float,
                          strike: float,
                          maturity: float,
                          r: float,
                          option_type: str = 'call',
                          tol: float = 1e-6,
                          max_iter: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson.

        Args:
            market_price: Market price of option
            spot: Spot price
            strike: Strike price
            maturity: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            tol: Tolerance for convergence
            max_iter: Maximum iterations

        Returns:
            Implied volatility
        """
        vol = 0.2  # Initial guess

        for _ in range(max_iter):
            price = PricingEngine.black_scholes_price(spot, strike, maturity, r, vol, option_type)
            diff = price - market_price

            if abs(diff) < tol:
                break

            # Vega for derivative
            d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * maturity) / (vol * np.sqrt(maturity))
            vega = spot * np.sqrt(maturity) * norm.pdf(d1)

            if vega > 0:
                vol = vol - diff / vega
                vol = max(vol, 0.001)  # Floor at 0.1%

        return vol

    @staticmethod
    def generate_market_scenarios(base_params: Dict,
                                 shocks: Dict[str, List[float]]) -> List[Dict]:
        """
        Generate market scenarios for stress testing.

        Args:
            base_params: Base market parameters
            shocks: Dictionary of parameter shocks

        Returns:
            List of scenario parameter sets
        """
        scenarios = []

        # Generate all combinations of shocks
        import itertools

        param_names = list(shocks.keys())
        shock_values = list(shocks.values())

        for combination in itertools.product(*shock_values):
            scenario = base_params.copy()
            for i, param in enumerate(param_names):
                if param in ['spot', 'vol', 'r']:
                    scenario[param] *= (1 + combination[i])
                else:
                    scenario[param] = combination[i]
            scenarios.append(scenario)

        return scenarios

    @staticmethod
    def calculate_risk_metrics(payoffs: np.ndarray,
                              confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        Calculate comprehensive risk metrics.

        Args:
            payoffs: Array of payoff values
            confidence_levels: Confidence levels for VaR/CVaR

        Returns:
            Dictionary of risk metrics
        """
        metrics = {
            'expected_payoff': np.mean(payoffs),
            'payoff_volatility': np.std(payoffs),
            'payoff_skewness': pd.Series(payoffs).skew(),
            'payoff_kurtosis': pd.Series(payoffs).kurtosis(),
            'max_payoff': np.max(payoffs),
            'min_payoff': np.min(payoffs),
            'median_payoff': np.median(payoffs)
        }

        for conf in confidence_levels:
            metrics[f'var_{int(conf*100)}'] = PricingEngine.calculate_var(payoffs, conf)
            metrics[f'cvar_{int(conf*100)}'] = PricingEngine.calculate_cvar(payoffs, conf)

        return metrics