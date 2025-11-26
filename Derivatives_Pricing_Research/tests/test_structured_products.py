"""
Unit tests for structured products library.
"""

import unittest
import numpy as np
from structured_products import Autocallable, ReverseConvertible, BarrierOption, RangeAccrual


class TestAutocallable(unittest.TestCase):
    def setUp(self):
        self.product = Autocallable(
            spot=100, strike=100, barrier=90, coupon=0.05,
            maturity=3, autocall_levels=[105, 110, 115]
        )

    def test_initialization(self):
        self.assertEqual(self.product.spot, 100)
        self.assertEqual(self.product.coupon, 0.05)
        self.assertEqual(len(self.product.autocall_levels), 3)

    def test_payoff_no_autocall(self):
        # Price stays below autocall levels
        path = np.array([100] + [95] * 252)  # Constant price below autocall
        payoff = self.product.payoff(path)
        expected = 1.0 + 0.05  # Capital + final coupon
        self.assertAlmostEqual(payoff, expected, places=2)

    def test_payoff_with_autocall(self):
        # Price hits first autocall level
        path = np.array([100] + [106] * 252)  # Constant price above first autocall
        payoff = self.product.payoff(path)
        expected = np.exp(-0.02 * 1) * (1 + 0.05)  # Early redemption at year 1
        self.assertAlmostEqual(payoff, expected, places=1)

    def test_greeks_calculation(self):
        greeks = self.product.calculate_greeks(n_sim=100)
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertTrue(-2 < greeks['delta'] < 2)  # Reasonable bounds


class TestReverseConvertible(unittest.TestCase):
    def setUp(self):
        self.product = ReverseConvertible(
            spot=100, strike=100, coupon=0.07, maturity=3,
            capital_protection=0.9
        )

    def test_payoff_above_strike(self):
        path = np.array([100] + [120] * 252)
        payoff = self.product.payoff(path)
        expected = 1.0 + 0.07  # Strike + coupon (no participation in upside)
        self.assertAlmostEqual(payoff, expected, places=2)

    def test_payoff_below_strike(self):
        path = np.array([100] + [80] * 252)
        payoff = self.product.payoff(path)
        expected = 80 + 0.07  # Final price + coupon
        self.assertAlmostEqual(payoff, expected, places=2)


class TestBarrierOption(unittest.TestCase):
    def setUp(self):
        self.product = BarrierOption(
            spot=100, strike=100, barrier=90, maturity=3,
            option_type='call', barrier_type='up_and_in'
        )

    def test_barrier_not_hit(self):
        path = np.array([100] + [85] * 252)  # Stays below barrier
        payoff = self.product.payoff(path)
        self.assertEqual(payoff, 0)  # Knock-in not triggered

    def test_barrier_hit(self):
        path = np.array([100] + [95] * 252)  # Hits barrier
        payoff = self.product.payoff(path)
        self.assertGreater(payoff, 0)  # Should have payoff


class TestRangeAccrual(unittest.TestCase):
    def setUp(self):
        self.product = RangeAccrual(
            spot=100, strike=100,
            ranges=[(90, 110), (85, 115)],
            coupon_rates=[0.06, 0.06],
            maturity=2
        )

    def test_payoff_in_range(self):
        # Price stays in range
        path = np.array([100] + [100] * 252)
        payoff = self.product.payoff(path)
        expected = 1.0 + 0.06 * 2  # Capital + full coupons
        self.assertAlmostEqual(payoff, expected, places=1)

    def test_payoff_out_of_range(self):
        # Price goes out of range
        path = np.array([100] + [120] * 252)  # Above upper bound
        payoff = self.product.payoff(path)
        expected = 1.0  # No coupons accrued
        self.assertAlmostEqual(payoff, expected, places=1)


if __name__ == '__main__':
    unittest.main()