#!/bin/bash

# Data Pipeline Setup Script
# This script sets up the data pipeline environment and downloads sample data

echo "Setting up LEAN Data Pipeline..."
echo ""
echo "To use the data pipeline:"
echo "1. Set your API keys as environment variables:"
echo "   export ALPACA_API_KEY='your_alpaca_api_key'"
echo "   export ALPACA_SECRET_KEY='your_alpaca_secret_key'"
echo "   export BINANCE_API_KEY='your_binance_api_key'  # Optional for public data"
echo "   export BINANCE_SECRET_KEY='your_binance_secret_key'  # Optional for public data"
echo ""
echo "2. Run the pipeline:"
echo "   python main.py --test  # For testing with limited data"
echo "   python main.py --source alpaca --resolution daily  # For Alpaca daily data"
echo "   python main.py --source binance --resolution minute  # For Binance minute data"
echo "   python main.py --help  # For all options"
echo ""
echo "Note: Binance API keys are optional for downloading public market data"
