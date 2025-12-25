#!/bin/bash
# Script to run the Put-Futures Arbitrage backtest with Lean

echo "Setting up Lean backtesting environment..."

# Check if Lean is installed
if ! command -v lean &> /dev/null; then
    echo "Lean CLI not found. Please install QuantConnect Lean:"
    echo "https://www.quantconnect.com/docs/v2/lean-cli/installation"
    exit 1
fi

# Navigate to project directory
cd "$(dirname "$0")"

# Create data directory if it doesn't exist
mkdir -p ../../../data

# Fetch data
echo "Fetching historical data..."
python data_fetcher.py

# Run backtest
echo "Running backtest..."
python backtrader_backtest.py

# Generate report
echo "Generating performance report..."
python analyze_results.py

echo "Backtest complete!"