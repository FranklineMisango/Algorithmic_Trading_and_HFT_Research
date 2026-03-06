"""
Fetch news sentiment for crisis periods using GDELT + OpenAI LLM analysis.
"""

import pandas as pd
from datetime import datetime, timedelta
from news_sentiment_llm import EnhancedNewsSentimentAnalyzer
import json

crisis_periods = [
    {
        'start': '2019-05-12',
        'end': '2019-07-31',
        'event': 'Gulf of Oman Tanker Attacks',
        'sentiment': 'High Tension',
        'severity': 'High'
    },
    {
        'start': '2020-01-03',
        'end': '2020-02-15',
        'event': 'US-Iran Escalation (Soleimani)',
        'sentiment': 'Extreme Tension',
        'severity': 'Critical'
    },
    {
        'start': '2021-04-11',
        'end': '2021-05-20',
        'event': 'Israel-Iran Naval Incidents',
        'sentiment': 'Elevated Tension',
        'severity': 'Medium'
    },
    {
        'start': '2022-02-24',
        'end': '2022-04-30',
        'event': 'Ukraine War Impact on Energy',
        'sentiment': 'Supply Concerns',
        'severity': 'High'
    },
    {
        'start': '2023-10-07',
        'end': '2024-01-31',
        'event': 'Red Sea / Bab al-Mandab Crisis',
        'sentiment': 'Critical Disruption',
        'severity': 'Critical'
    },
    {
        'start': '2024-04-13',
        'end': '2024-05-15',
        'event': 'Iran-Israel Direct Exchange',
        'sentiment': 'Extreme Tension',
        'severity': 'Critical'
    },
    {
        'start': '2025-02-15',
        'end': '2025-02-25',
        'event': 'Hormuz Naval "Inspections" Warning',
        'sentiment': 'Market Signaling',
        'severity': 'Medium'
    },
    {
        'start': '2025-06-13',
        'end': '2025-06-24',
        'event': '12-Day Air War (US/ISR vs Iran)',
        'sentiment': 'Open State Warfare',
        'severity': 'Critical'
    },
    {
        'start': '2026-02-28',
        'end': '2026-03-06',
        'event': 'Operation Epic Fury / Full Blockade',
        'sentiment': 'Total Supply Shock',
        'severity': 'Critical'
    }
]

def fetch_sentiment_for_period(analyzer, period, max_articles=30):
    """Fetch sentiment for a single crisis period."""
    print(f"\n{'='*80}")
    print(f"FETCHING SENTIMENT FOR: {period['event']}")
    print(f"Period: {period['start']} to {period['end']}")
    print(f"{'='*80}\n")
    
    # Expand date range slightly to capture pre/post crisis sentiment
    start_dt = pd.to_datetime(period['start']) - timedelta(days=7)
    end_dt = pd.to_datetime(period['end']) + timedelta(days=7)
    
    keywords = [
        "Strait of Hormuz",
        "Iran military",
        "Persian Gulf",
        "oil tanker",
        period['event'].split()[0]  # First word of event name
    ]
    
    sentiment_df = analyzer.get_geopolitical_sentiment(
        start_date=start_dt.strftime('%Y-%m-%d'),
        end_date=end_dt.strftime('%Y-%m-%d'),
        keywords=keywords,
        max_articles=max_articles
    )
    
    if len(sentiment_df) > 0:
        sentiment_df['crisis_event'] = period['event']
        sentiment_df['crisis_start'] = period['start']
        sentiment_df['crisis_end'] = period['end']
        sentiment_df['expected_sentiment'] = period['sentiment']
        sentiment_df['expected_severity'] = period['severity']
    
    return sentiment_df


def main():
    """Fetch sentiment for all crisis periods."""
    print("="*80)
    print("CRISIS PERIOD SENTIMENT ANALYSIS")
    print("="*80)
    print(f"Total crisis periods to analyze: {len(crisis_periods)}")
    print(f"Date range: 2019-2026 (7 years of geopolitical events)")
    print("Using: GDELT + OpenAI GPT-4o-mini")
    print("="*80)
    
    # Initialize analyzer
    analyzer = EnhancedNewsSentimentAnalyzer(
        fetch_full_articles=True  # Fetch full articles for better analysis
    )
    
    all_sentiment = []
    
    for i, period in enumerate(crisis_periods, 1):
        print(f"\n\n{'#'*80}")
        print(f"CRISIS {i}/{len(crisis_periods)}")
        print(f"{'#'*80}")
        
        try:
            sentiment_df = fetch_sentiment_for_period(
                analyzer, 
                period, 
                max_articles=30  # Analyze up to 30 articles per crisis
            )
            
            if len(sentiment_df) > 0:
                all_sentiment.append(sentiment_df)
                print(f"\n✓ Successfully analyzed {len(sentiment_df)} articles")
                print(f"  Average sentiment: {sentiment_df['sentiment'].mean():.3f}")
                print(f"  Risk levels: {sentiment_df['risk_level'].value_counts().to_dict()}")
            else:
                print(f"\n⚠ No articles found for this period")
        
        except Exception as e:
            print(f"\n✗ Error analyzing period: {e}")
            continue
    
    # Combine all results
    if all_sentiment:
        combined_df = pd.concat(all_sentiment, ignore_index=True)
        
        # Save to CSV
        output_file = 'Data/crisis_sentiment_analysis.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total articles analyzed: {len(combined_df)}")
        print(f"Saved to: {output_file}")
        print(f"\nOverall statistics:")
        print(f"  Average sentiment: {combined_df['sentiment'].mean():.3f}")
        print(f"  Average confidence: {combined_df['confidence'].mean():.3f}")
        print(f"  Full articles fetched: {combined_df['used_full_article'].sum()}")
        print(f"\nRisk level distribution:")
        print(combined_df['risk_level'].value_counts())
        print(f"\nBy crisis event:")
        print(combined_df.groupby('crisis_event')['sentiment'].agg(['count', 'mean', 'std']).round(3))
        
        # Create daily summary
        daily_summary = combined_df.groupby(['date', 'crisis_event']).agg({
            'sentiment': ['mean', 'std', 'count'],
            'confidence': 'mean',
            'risk_level': lambda x: x.mode()[0] if len(x) > 0 else 'unknown'
        }).round(3)
        
        daily_summary.columns = ['sentiment_mean', 'sentiment_std', 'article_count', 
                                'confidence_mean', 'risk_level']
        daily_summary = daily_summary.reset_index()
        
        daily_file = 'Data/crisis_sentiment_daily.csv'
        daily_summary.to_csv(daily_file, index=False)
        print(f"\nDaily summary saved to: {daily_file}")
        
    else:
        print("\n✗ No sentiment data collected")


if __name__ == "__main__":
    main()
