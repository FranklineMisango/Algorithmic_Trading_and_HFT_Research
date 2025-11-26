# Structured Equity Products Library

This project provides a comprehensive Python library for pricing and simulating structured equity products, designed to demonstrate skills in derivative structuring for roles like EQD Structuring at BNP Paribas.

## Features

### Structured Equity Payoff Library
- **Autocallables**: Pricing and simulation of autocallable notes with periodic coupons and early redemption features
- **Reverse Convertibles**: Pricing of reverse convertible notes with capital protection and upside participation
- **Barrier Options**: Knock-in/knock-out barrier options with various barrier types
- **Range Accruals**: Range accrual notes with coupon payments based on asset staying within ranges

### Pricing Methods
- **Monte Carlo Simulation**: Path-dependent pricing for complex payoffs
- **Analytical Solutions**: Closed-form solutions where available
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho sensitivities
- **Scenario Analysis**: Stress testing under various market conditions

### Visualizations
- Payoff diagrams at maturity
- Sensitivity plots (Greeks)
- Probability distributions of returns
- Risk-reward scatter plots

### Client Pitch Simulator
- Client profile input (risk appetite, market view, investment horizon)
- Automated product recommendation engine
- Comparative analysis vs vanilla options/ETFs
- Pitch deck generation with key selling points

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Pricing Example

```python
from structured_products import Autocallable

# Create an autocallable note
note = Autocallable(
    spot=100,
    strike=100,
    barrier=90,
    coupon=0.05,
    maturity=3,
    autocall_levels=[105, 110, 115]
)

# Price using Monte Carlo
price = note.price_mc(n_sim=10000, r=0.02, vol=0.2)
print(f"Autocallable Price: {price:.2f}")

# Calculate Greeks
greeks = note.calculate_greeks()
print(f"Delta: {greeks['delta']:.4f}")
```

### Client Pitch Simulation

See the `client_pitch_simulator.ipynb` notebook for interactive examples.

## Project Structure

```
Derivatives_Pricing_Research/
├── structured_products/
│   ├── __init__.py
│   ├── base_product.py
│   ├── autocallable.py
│   ├── reverse_convertible.py
│   ├── barrier_option.py
│   ├── range_accrual.py
│   └── pricing_engine.py
├── notebooks/
│   ├── payoff_visualizations.ipynb
│   └── client_pitch_simulator.ipynb
├── tests/
│   ├── test_autocallable.py
│   └── test_pricing_engine.py
├── requirements.txt
├── README.md
└── setup.py
```

## Mathematical Framework

### Monte Carlo Pricing
For path-dependent products, we use geometric Brownian motion:
```
dS = rS dt + σS dW
```

### Greeks Calculation
Using finite differences and pathwise derivatives for efficiency.

### Risk Metrics
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Expected Shortfall

## Contributing

This project demonstrates:
- **Product Design**: Innovative payoff structures
- **Quantitative Skills**: Advanced pricing and risk modeling
- **Communication**: Clear visualization and client-focused analysis
- **Commercial Orientation**: Solving client problems with structured solutions

## License

MIT License