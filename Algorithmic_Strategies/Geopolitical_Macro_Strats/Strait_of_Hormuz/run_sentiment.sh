#!/bin/bash

echo "=========================================="
echo "Crisis Sentiment Analysis"
echo "=========================================="
echo ""
echo "This will fetch news articles and analyze them with AI"
echo "Expected runtime: 10-20 minutes"
echo "Cost: ~\$0.50-1.00 in OpenAI API calls"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    python fetch_crisis_sentiment.py
    echo ""
    echo "=========================================="
    echo "Complete! Results saved to:"
    echo "  - Data/crisis_sentiment_analysis.csv"
    echo "  - Data/crisis_sentiment_daily.csv"
    echo ""
    echo "Now open 01_shipping_deep_analysis.ipynb"
    echo "=========================================="
fi
