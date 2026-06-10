"""
YFinance API - Flask web service for Yahoo Finance data
Provides REST API endpoints for securities, news, and fundamentals data
"""

from flask import Flask, request, jsonify
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import logging
import os
from datetime import datetime, timedelta

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Use environment variables for API keys
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

@app.route('/securities', methods=['GET'])
def get_securities():
    tickers = request.args.get('tickers', 'AAPL,MSFT,GOOGL').split(',')
    data = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            fast = ticker.fast_info
            data.append({
                'symbol': t,
                'name': t,
                'exchange': fast.get('exchange', ''),
                'region': '',
                'lastPrice': fast.get('lastPrice', None)
            })
        except Exception as e:
            data.append({'symbol': t, 'error': str(e)})
    return jsonify(data)

@app.route('/news', methods=['GET'])
def get_news():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        app.logger.info(f"Fetching news for ticker: {ticker}")
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news

        if not news:
            app.logger.warning(f"No news found for ticker: {ticker}")
            return jsonify([])

        formatted_news = []
        for item in news:
            formatted_item = {
                'Title': item.get('title', ''),
                'Publisher': item.get('publisher', ''),
                'Link': item.get('link', ''),
                'ProviderPublishTime': item.get('providerPublishTime', 0),
                'Type': item.get('type', ''),
                'Thumbnail': item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '') if item.get('thumbnail') else '',
                'Summary': item.get('title', '')
            }
            formatted_news.append(formatted_item)

        app.logger.info(f"Successfully fetched {len(formatted_news)} news items for {ticker}")
        return jsonify(formatted_news)

    except Exception as e:
        app.logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/fundamentals', methods=['GET'])
def get_fundamentals():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        app.logger.info(f"Fetching fundamentals for ticker: {ticker}")

        fundamentals = {}

        # Use yfinance as primary source
        app.logger.info(f"Fetching from yfinance for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        yfinance_data = {
            'symbol': ticker,
            'currentPrice': info.get('currentPrice') or info.get('regularMarketPrice'),
            'marketCap': info.get('marketCap'),
            'enterpriseValue': info.get('enterpriseValue'),
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'pegRatio': info.get('pegRatio'),
            'priceToBook': info.get('priceToBook'),
            'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
            'debtToEquity': info.get('debtToEquity'),
            'returnOnEquity': info.get('returnOnEquity'),
            'dividendYield': info.get('dividendYield'),
            'payoutRatio': info.get('payoutRatio'),
            'freeCashflow': info.get('freeCashflow'),
            'operatingCashflow': info.get('operatingCashflow'),
            'revenueGrowth': info.get('revenueGrowth'),
            'earningsGrowth': info.get('earningsGrowth'),
            'sharesOutstanding': info.get('sharesOutstanding'),
            'beta': info.get('beta'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
            'averageVolume': info.get('averageVolume'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'longBusinessSummary': info.get('longBusinessSummary')
        }

        fundamentals.update(yfinance_data)

        # Try Alpaca API as supplement (after yfinance, so yfinance takes precedence)
        if ALPACA_API_KEY:
            alpaca_data = get_alpaca_data(ticker)
            if alpaca_data:
                # Only add Alpaca fields not already in yfinance data
                for k, v in alpaca_data.items():
                    if k not in fundamentals or fundamentals[k] is None:
                        fundamentals[k] = v

        app.logger.info(f"Successfully fetched fundamentals for {ticker}")
        return jsonify(fundamentals)

    except Exception as e:
        app.logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_alpaca_data(ticker):
    """Get data from Alpaca API using environment variables"""
    try:
        if not ALPACA_API_KEY:
            app.logger.warning("Alpaca API key not configured")
            return None

        headers = {
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
            'Content-Type': 'application/json'
        }

        quote_url = f"{ALPACA_BASE_URL}/v2/stocks/{ticker}/quotes/latest"
        response = requests.get(quote_url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'quote' in data:
                quote = data['quote']
                fundamentals = {
                    'symbol': ticker,
                    'currentPrice': float(quote.get('bp', 0)),
                    'source': 'alpaca'
                }
                app.logger.info(f"Alpaca data for {ticker}: {fundamentals}")
                return fundamentals
        return None

    except Exception as e:
        app.logger.error(f"Alpaca API error for {ticker}: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'yfinance-api'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
