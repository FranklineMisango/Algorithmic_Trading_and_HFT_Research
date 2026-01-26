#!/bin/bash

# Currency Crash Prediction - Quick Start Script

echo "=========================================="
echo "Currency Crash Prediction Model Setup"
echo "=========================================="

# Create data directory
mkdir -p data

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run data fetcher
echo ""
echo "Fetching currency and interest rate data..."
python data_fetcher.py --start 1999-01-01 --end 2023-12-31

# Generate signals
echo ""
echo "Generating R-Zone signals..."
python signal_generator.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb"
echo "2. Analyze signals: jupyter notebook notebooks/02_signal_generation.ipynb"
echo "3. Run backtest: jupyter notebook notebooks/03_backtest_analysis.ipynb"
echo "4. LEAN backtest: cd lean && lean backtest main.py"
echo ""
