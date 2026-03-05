"""
Enhanced news sentiment analysis using LLMs and full article content.
Supports RAG-style chunking and caching for efficiency.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import json
import os
from dotenv import load_dotenv
import hashlib
import pickle
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠ BeautifulSoup not installed. Run: pip install beautifulsoup4")
    print("   Full article fetching will be disabled.")
load_dotenv()

# Try importing optional libraries
try:
    import gdelt
    GDELT_AVAILABLE = True
except ImportError:
    GDELT_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

try:
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest
    ALPACA_NEWS_AVAILABLE = True
except ImportError:
    ALPACA_NEWS_AVAILABLE = False

# LLM imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠ Anthropic not installed. Run: pip install anthropic")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ OpenAI not installed. Run: pip install openai")


class ArticleFetcher:
    """Fetch and parse full article content from URLs."""
    
    def __init__(self, cache_dir: str = '.cache/articles'):
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 required for article fetching. Run: pip install beautifulsoup4")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        
        # Rotate user agents for better success rate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        ]
        self.session.headers.update({
            'User-Agent': self.user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.pkl"
    
    def fetch_article(self, url: str, max_retries: int = 2) -> Optional[Dict]:
        """Fetch and parse article content with caching."""
        # Skip known problematic domains
        if any(domain in url for domain in ['consent.yahoo.com', 'bbc.co.uk/sounds']):
            return None
            
        # Check cache first
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # Fetch from web
        for attempt in range(max_retries):
            try:
                # Rotate user agent on retries
                if attempt > 0:
                    self.session.headers['User-Agent'] = self.user_agents[attempt % len(self.user_agents)]
                
                response = self.session.get(url, timeout=10, allow_redirects=True)
                response.raise_for_status()
                
                # Check if we got redirected to a consent page
                if 'consent' in response.url.lower() or len(response.content) < 1000:
                    return None
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract text from article body
                article_text = self._extract_article_text(soup)
                
                if not article_text or len(article_text) < 100:
                    return None
                
                result = {
                    'url': url,
                    'text': article_text,
                    'length': len(article_text),
                    'fetched_at': datetime.now().isoformat()
                }
                
                # Cache result
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
                continue
        
        return None
    
    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """Extract main article text from HTML."""
        # Try common article containers
        article_selectors = [
            'article',
            '[role="article"]',
            '.article-content',
            '.article-body',
            '.post-content',
            '.entry-content',
            '.story-body',
            '.article__body',
            '[itemprop="articleBody"]',
            'main'
        ]
        
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text(separator=' ', strip=True) 
                               for elem in elements])
                if len(text) > 200:
                    return self._clean_text(text)
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common noise
        text = re.sub(r'(Cookie Policy|Privacy Policy|Terms of Service|Subscribe|Sign up).*?\.', '', text)
        return text.strip()


class TextChunker:
    """Chunk long articles for LLM processing."""
    
    def __init__(self, chunk_size: int = 4000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 200 chars
                chunk_text = text[start:end]
                last_period = chunk_text.rfind('. ')
                if last_period > self.chunk_size - 200:
                    end = start + last_period + 1
            
            chunks.append(text[start:end])
            start = end - self.overlap
        
        return chunks


class LLMSentimentAnalyzer:
    """LLM-based sentiment analysis for geopolitical news."""
    
    def __init__(self, provider: str = 'openai', model: str = None, api_key: str = None):
        self.provider = provider.lower()
        self.cache_dir = Path('.cache/sentiment')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic not installed")
            key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment or config")
            self.client = anthropic.Anthropic(api_key=key)
            self.model = model or 'claude-3-5-sonnet-20241022'
        elif self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed")
            key = api_key or os.getenv('OPENAI_API_KEY')
            if not key:
                raise ValueError("OPENAI_API_KEY not found in environment or config")
            self.client = openai.OpenAI(api_key=key)
            self.model = model or 'gpt-4o-mini'
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def analyze_sentiment(self, text: str, context: str = "Strait of Hormuz") -> Dict:
        """Analyze sentiment with LLM."""
        # Check cache
        cache_key = self._get_cache_key(text[:500])  # Use first 500 chars for key
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        prompt = f"""Analyze the geopolitical risk sentiment of this news article about {context}.

Article text:
{text[:6000]}  

Score the risk level using this calibrated scale:
-1.0 to -0.8: Active warfare, major attacks, complete blockade
-0.7 to -0.5: Serious threats, military mobilization, sanctions announced
-0.4 to -0.2: Tensions rising, diplomatic warnings, minor incidents
-0.1 to 0.1: Neutral reporting, routine operations, no change
0.2 to 0.4: De-escalation talks, diplomatic progress
0.5 to 0.7: Agreements reached, sanctions lifted
0.8 to 1.0: Peace treaties, full normalization

Provide your analysis in JSON format:
{{
    "sentiment_score": <float from -1.0 to 1.0, use the full range based on actual severity>,
    "risk_level": "<low|medium|high|critical>",
    "key_factors": ["list", "of", "key", "risk", "factors"],
    "confidence": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of why this specific score>"
}}

Be precise: differentiate between speculation (-0.3), confirmed threats (-0.6), and actual attacks (-0.9).
Focus on: military actions, trade disruptions, sanctions, diplomatic relations, economic impacts."""
        
        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
            
            # Parse JSON response
            result = json.loads(result_text)
            
            # Cache result
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            
            return result
            
        except Exception as e:
            print(f"    ⚠ LLM analysis error: {e}")
            return {
                'sentiment_score': 0.0,
                'risk_level': 'unknown',
                'key_factors': [],
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}'
            }
    
    def analyze_chunks(self, chunks: List[str], context: str = "Strait of Hormuz") -> Dict:
        """Analyze multiple chunks and aggregate results."""
        results = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk) < 100:  # Skip very short chunks
                continue
            result = self.analyze_sentiment(chunk, context)
            results.append(result)
        
        if not results:
            return {
                'sentiment_score': 0.0,
                'risk_level': 'unknown',
                'key_factors': [],
                'confidence': 0.0,
                'reasoning': 'No valid chunks to analyze'
            }
        
        # Aggregate results (weighted by confidence)
        total_confidence = sum(r['confidence'] for r in results)
        if total_confidence == 0:
            avg_sentiment = np.mean([r['sentiment_score'] for r in results])
        else:
            avg_sentiment = sum(r['sentiment_score'] * r['confidence'] for r in results) / total_confidence
        
        # Collect all key factors
        all_factors = []
        for r in results:
            all_factors.extend(r.get('key_factors', []))
        
        # Determine overall risk level
        risk_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4, 'unknown': 0}
        max_risk = max([risk_scores.get(r['risk_level'], 0) for r in results])
        risk_levels = {v: k for k, v in risk_scores.items()}
        
        return {
            'sentiment_score': float(avg_sentiment),
            'risk_level': risk_levels.get(max_risk, 'unknown'),
            'key_factors': list(set(all_factors))[:10],  # Top 10 unique factors
            'confidence': float(np.mean([r['confidence'] for r in results])),
            'reasoning': f'Aggregated from {len(results)} chunks',
            'num_chunks': len(results)
        }


class EnhancedNewsSentimentAnalyzer:
    """Enhanced multi-source news sentiment with LLM analysis."""
    
    def __init__(self, config_path: str = 'appsettings.json',
                 llm_provider: str = 'openai',
                 fetch_full_articles: bool = True):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.fetch_full_articles = fetch_full_articles
        
        # Initialize components
        self.article_fetcher = ArticleFetcher() if fetch_full_articles else None
        self.text_chunker = TextChunker()
        
        # Get LLM config from appsettings.json
        llm_config = self.config.get('OpenAI', {}) if llm_provider == 'openai' else self.config.get('Anthropic', {})
        api_key = llm_config.get('ApiKey') or os.getenv(f'{llm_provider.upper()}_API_KEY')
        model = llm_config.get('ModelId')
        
        self.llm_analyzer = LLMSentimentAnalyzer(
            provider=llm_provider,
            model=model,
            api_key=api_key
        )
        
        # Initialize news API clients (reuse from original)
        self.gdelt_client = None
        self.newsapi_client = None
        self.alpaca_news_client = None
        self.google_api_key = None
        self.google_search_engine_id = None
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize news API clients."""
        if GDELT_AVAILABLE:
            try:
                self.gdelt_client = gdelt.gdelt(version=2)
                print("✓ GDELT client initialized")
            except Exception as e:
                print(f"⚠ GDELT initialization failed: {e}")
        
        if NEWSAPI_AVAILABLE:
            api_key = os.getenv('NEWS_API_KEY') or self.config.get('NewsApi', {}).get('ApiKey')
            if api_key:
                try:
                    self.newsapi_client = NewsApiClient(api_key=api_key)
                    print("✓ NewsAPI client initialized")
                except Exception as e:
                    print(f"⚠ NewsAPI initialization failed: {e}")
        
        if ALPACA_NEWS_AVAILABLE:
            api_key = self.config.get('Alpaca', {}).get('ApiKey')
            secret_key = self.config.get('Alpaca', {}).get('SecretKey')
            if api_key and secret_key:
                try:
                    self.alpaca_news_client = NewsClient(api_key, secret_key)
                    print("✓ Alpaca News client initialized")
                except Exception as e:
                    print(f"⚠ Alpaca News initialization failed: {e}")
        
        google_config = self.config.get('GoogleSearch', {})
        self.google_api_key = google_config.get('ApiKey')
        self.google_search_engine_id = google_config.get('SearchEngineId')
        if self.google_api_key and self.google_search_engine_id:
            print("✓ Google Search API configured")
        
        print(f"✓ LLM Sentiment Analyzer initialized ({self.llm_analyzer.provider})")

    
    def fetch_newsapi_articles(self, start_date: str, end_date: str,
                               keywords: List[str]) -> pd.DataFrame:
        """Fetch articles from NewsAPI."""
        if not self.newsapi_client:
            return pd.DataFrame()
        
        print("  → Fetching NewsAPI articles...")
        all_processed = []
        
        for keyword in keywords[:3]:
            try:
                all_articles = self.newsapi_client.get_everything(
                    q=keyword,
                    from_param=start_date,
                    to=end_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=100
                )
                
                if all_articles and 'articles' in all_articles:
                    for article in all_articles['articles']:
                        all_processed.append({
                            'date': pd.to_datetime(article['publishedAt']),
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'source_name': article.get('source', {}).get('name', 'Unknown'),
                            'source': 'NewsAPI'
                        })
            except Exception as e:
                print(f"    ⚠ Error fetching keyword '{keyword}': {e}")
                continue
        
        if not all_processed:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_processed)
        df = df.drop_duplicates(subset=['url'], keep='first')
        print(f"    ✓ Fetched {len(df)} NewsAPI articles")
        
        return df
    
    def analyze_article(self, article: Dict, context: str = "Strait of Hormuz") -> Dict:
        """Analyze a single article with LLM."""
        url = article.get('url', '')
        
        # Fetch full article if enabled
        if self.fetch_full_articles and url:
            print(f"    → Fetching full article: {url[:60]}...")
            full_article = self.article_fetcher.fetch_article(url)
            
            if full_article and full_article['length'] > 500:
                # Chunk and analyze
                chunks = self.text_chunker.chunk_text(full_article['text'])
                print(f"      Split into {len(chunks)} chunks")
                analysis = self.llm_analyzer.analyze_chunks(chunks, context)
                analysis['used_full_article'] = True
                analysis['article_length'] = full_article['length']
                return analysis
            else:
                print(f"      ⚠ Could not fetch full article, using title/description")
        
        # Fallback: analyze title + description
        text = f"{article.get('title', '')} {article.get('description', '')}"
        analysis = self.llm_analyzer.analyze_sentiment(text, context)
        analysis['used_full_article'] = False
        analysis['article_length'] = len(text)
        
        return analysis
    
    def get_geopolitical_sentiment(self, start_date: str, end_date: str,
                                   keywords: List[str] = None,
                                   max_articles: int = 20) -> pd.DataFrame:
        """
        Get geopolitical sentiment with LLM analysis.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            keywords: Search keywords
            max_articles: Maximum articles to analyze (LLM calls are expensive)
        """
        if keywords is None:
            keywords = ["Strait of Hormuz", "Iran", "Persian Gulf"]
        
        print(f"\n{'='*60}")
        print(f"ENHANCED NEWS SENTIMENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Keywords: {', '.join(keywords)}")
        print(f"Max articles to analyze: {max_articles}")
        print(f"Full article fetching: {self.fetch_full_articles}")
        print(f"{'='*60}\n")
        
        # Fetch articles from NewsAPI
        articles_df = self.fetch_newsapi_articles(start_date, end_date, keywords)
        
        if len(articles_df) == 0:
            print("⚠ No articles found")
            return pd.DataFrame()
        
        print(f"\n✓ Found {len(articles_df)} articles")
        print(f"  Analyzing top {min(max_articles, len(articles_df))} articles with LLM...\n")
        
        # Analyze articles with LLM
        analyzed = []
        for idx, row in articles_df.head(max_articles).iterrows():
            print(f"  [{idx+1}/{min(max_articles, len(articles_df))}] {row['title'][:60]}...")
            
            analysis = self.analyze_article(row.to_dict(), context=keywords[0])
            
            analyzed.append({
                'date': row['date'],
                'title': row['title'],
                'url': row['url'],
                'source_name': row['source_name'],
                'sentiment': analysis['sentiment_score'],
                'risk_level': analysis['risk_level'],
                'confidence': analysis['confidence'],
                'key_factors': ', '.join(analysis.get('key_factors', [])[:5]),
                'used_full_article': analysis.get('used_full_article', False),
                'article_length': analysis.get('article_length', 0),
                'source': 'NewsAPI-LLM'
            })
        
        result_df = pd.DataFrame(analyzed)
        result_df['date'] = pd.to_datetime(result_df['date'], utc=True).dt.tz_localize(None).dt.date
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Articles analyzed: {len(result_df)}")
        print(f"Full articles fetched: {result_df['used_full_article'].sum()}")
        print(f"Average sentiment: {result_df['sentiment'].mean():.3f}")
        print(f"Average confidence: {result_df['confidence'].mean():.3f}")
        print(f"\nRisk level distribution:")
        print(result_df['risk_level'].value_counts())
        
        return result_df
    
    def get_daily_summary(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment by day."""
        if len(sentiment_df) == 0:
            return pd.DataFrame()
        
        daily = sentiment_df.groupby('date').agg({
            'sentiment': ['mean', 'std', 'count'],
            'confidence': 'mean',
            'risk_level': lambda x: x.mode()[0] if len(x) > 0 else 'unknown'
        }).round(3)
        
        daily.columns = ['sentiment_mean', 'sentiment_std', 'article_count', 
                        'confidence_mean', 'risk_level']
        
        # Calculate risk score (0-100)
        daily['risk_score'] = ((1 - daily['sentiment_mean']) / 2 * 100).round(1)
        
        return daily.sort_index()


if __name__ == "__main__":
    # Quick test
    print("Testing Enhanced News Sentiment Analyzer with LLM...\n")
    
    analyzer = EnhancedNewsSentimentAnalyzer(
        llm_provider='openai',  # Using OpenAI by default
        fetch_full_articles=True
    )
    
    # Test with recent dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    sentiment_df = analyzer.get_geopolitical_sentiment(
        start_date=start_date,
        end_date=end_date,
        keywords=["Strait of Hormuz", "Iran"],
        max_articles=5  # Start small for testing
    )
    
    if len(sentiment_df) > 0:
        print(f"\n{'='*60}")
        print("SAMPLE RESULTS")
        print(f"{'='*60}\n")
        print(sentiment_df[['date', 'title', 'sentiment', 'risk_level', 'confidence']].head())
        
        print(f"\n{'='*60}")
        print("DAILY SUMMARY")
        print(f"{'='*60}\n")
        daily = analyzer.get_daily_summary(sentiment_df)
        print(daily)
