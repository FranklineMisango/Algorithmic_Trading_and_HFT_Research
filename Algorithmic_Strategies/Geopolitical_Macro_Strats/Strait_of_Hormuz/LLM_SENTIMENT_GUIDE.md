# LLM Sentiment Analysis Guide

## Overview

This project uses LLM-based sentiment analysis to evaluate geopolitical risk from news articles about the Strait of Hormuz. The system fetches articles from NewsAPI and analyzes them using OpenAI or Anthropic models.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key in .env
OPENAI_API_KEY=your_key_here

# Run the strategy
python main.py
```

## Features

- **Multi-source news fetching**: NewsAPI with 150,000+ sources
- **Full article scraping**: Extracts complete article text when possible
- **LLM analysis**: Uses GPT-4o-mini or Claude for sentiment scoring
- **Calibrated scoring**: -1.0 (warfare) to +1.0 (peace)
- **Caching**: Avoids redundant API calls
- **Fallback**: Uses title/description when scraping fails

## Calibrated Scoring Scale

The LLM uses this calibrated scale for consistent scoring:

- **-1.0 to -0.8**: Active warfare, major attacks, complete blockade
- **-0.7 to -0.5**: Serious threats, military mobilization, sanctions announced
- **-0.4 to -0.2**: Tensions rising, diplomatic warnings, minor incidents
- **-0.1 to 0.1**: Neutral reporting, routine operations
- **0.2 to 0.4**: De-escalation talks, diplomatic progress
- **0.5 to 0.7**: Agreements reached, sanctions lifted
- **0.8 to 1.0**: Peace treaties, full normalization

## Configuration

All settings are in `appsettings.json`:

```json
{
  "NewsAPI": {
    "ApiKey": "your_newsapi_key"
  },
  "LLM": {
    "Provider": "openai",
    "Model": "gpt-4o-mini",
    "FetchFullArticles": true
  }
}
```

## Usage in Code

```python
from news_sentiment_llm import EnhancedNewsSentimentAnalyzer

analyzer = EnhancedNewsSentimentAnalyzer(
    config_path='appsettings.json',
    use_llm=True,
    fetch_full_articles=True
)

sentiment_df = analyzer.get_geopolitical_sentiment(
    start_date='2024-01-01',
    end_date='2025-03-05',
    keywords=['Strait of Hormuz', 'Iran', 'oil tanker'],
    context="Strait of Hormuz"
)
```

## Output Format

The analyzer returns a DataFrame with:

- `date`: Date of analysis
- `sentiment_mean`: Average sentiment score (-1 to 1)
- `sentiment_std`: Standard deviation of scores
- `risk_level`: Categorical risk (low/medium/high/critical)
- `article_count`: Number of articles analyzed
- `confidence`: Confidence score (0 to 1)

## Article Scraping

The system attempts to fetch full article text for better analysis:

- **Success rate**: ~40% (varies by source)
- **Fallback**: Uses title + description when scraping fails
- **Blocked sites**: Automatically skips consent pages and paywalls
- **Caching**: Stores fetched articles to avoid re-scraping

## API Costs

Approximate costs per analysis:

- **NewsAPI**: Free tier (100 requests/day)
- **OpenAI GPT-4o-mini**: ~$0.01 per 10 articles
- **Anthropic Claude**: ~$0.02 per 10 articles

For 250 articles/day: ~$0.25/day or $7.50/month

## Troubleshooting

**No articles found:**
- Check NewsAPI key in appsettings.json
- Verify date range (free tier: last 30 days)
- Try broader keywords

**Low scraping success:**
- Normal (40% is expected)
- System uses title/description as fallback
- Consider upgrading to paid scraping service

**High API costs:**
- Reduce article count (top_n parameter)
- Use caching (enabled by default)
- Switch to cheaper model

## Files

- `news_sentiment_llm.py`: Main LLM sentiment analyzer
- `data_acquisition.py`: Integrates sentiment into data pipeline
- `main.py`: Full trading strategy
- `appsettings.json`: Configuration
- `.env`: API keys

## Requirements

```
openai>=1.0.0
anthropic>=0.18.0
newsapi-python>=0.2.7
beautifulsoup4>=4.12.0
requests>=2.31.0
pandas>=2.0.0
```
