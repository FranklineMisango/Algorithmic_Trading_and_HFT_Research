"""
BERT/BART Fast Local Scorer
Alternative to OpenAI GPT-4o-mini for scoring earnings transcripts.

Advantages:
- FREE (no API costs)
- FAST (local GPU/CPU processing)
- No rate limits
- Reproducible results

Uses:
- facebook/bart-large-cnn for text summarization
- mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis for sentiment

Based on: https://github.com/FranklineMisango/NLG_Sentiment_Analysis
"""

import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class BertBartScorer:
    """
    Fast local scorer using BERT for sentiment and BART for summarization.
    
    This scorer processes earnings transcripts in chunks, generates summaries,
    and analyzes sentiment to produce economy scores (1-5 scale).
    """
    
    def __init__(self, 
                 chunk_size: int = 4000,
                 max_summary_length: int = 150,
                 min_summary_length: int = 50,
                 use_gpu: bool = True):
        """
        Initialize BERT/BART scorer with local models.
        
        Args:
            chunk_size: Max characters per chunk for processing
            max_summary_length: Max tokens for BART summary
            min_summary_length: Min tokens for BART summary
            use_gpu: Whether to use GPU (if available)
        """
        self.chunk_size = chunk_size
        self.max_summary_length = max_summary_length
        self.min_summary_length = min_summary_length
        
        # Determine device
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        print(f"Initializing BERT/BART scorer on {device_name}...")
        
        # Initialize BART summarization pipeline
        print("Loading BART summarization model (facebook/bart-large-cnn)...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=self.device,
            framework="pt"  # Force PyTorch, avoid TensorFlow
        )
        
        # Initialize DistilRoBERTa financial sentiment pipeline
        print("Loading DistilRoBERTa financial sentiment model...")
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            device=self.device,
            framework="pt"  # Force PyTorch, avoid TensorFlow
        )
        
        # Sentiment to score mapping
        # Financial sentiment model returns: positive, negative, neutral
        self.sentiment_to_score = {
            'positive': 4.0,  # Optimistic outlook
            'neutral': 3.0,   # Balanced outlook
            'negative': 2.0   # Pessimistic outlook
        }
        
        print("✓ BERT/BART scorer initialized successfully")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into manageable chunks for processing.
        
        Uses sentence-based splitting to avoid breaking mid-sentence.
        
        Args:
            text: Full transcript text
            
        Returns:
            List of text chunks (filtered for quality)
        """
        # Normalize text
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())  # Remove excessive whitespace
        
        # Simple sentence-based chunking
        sentences = text.split('. ')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Skip very short "sentences" (likely artifacts)
            if len(sentence.strip()) < 20:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk and start new one
                chunk_text = '. '.join(current_chunk) + '.'
                # Only add chunks with sufficient content
                if len(chunk_text.split()) >= 50:  # At least 50 words
                    chunks.append(chunk_text)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            if len(chunk_text.split()) >= 50:
                chunks.append(chunk_text)
        
        # Fallback if no valid chunks created
        if not chunks and len(text) > 100:
            # Create one large chunk from middle section (most informative)
            start = len(text) // 4
            end = start + min(self.chunk_size, len(text) - start)
            chunks = [text[start:end]]
        
        return chunks if chunks else [text[:min(len(text), self.chunk_size)]]
    
    def clean_transcript(self, text: str) -> str:
        """
        Clean transcript text while preserving economic content.
        
        Minimal cleaning to preserve signal (learned from previous over-cleaning issues).
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        # Remove only truly problematic content
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common boilerplate (optional - keep minimal)
        boilerplate = [
            r'Forward-?looking statements.*?(?=\n|\.|$)',
            r'Safe harbor.*?(?=\n|\.|$)',
            r'©.*?(?=\n|\.|$)'
        ]
        
        for pattern in boilerplate:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def summarize_chunk(self, chunk: str) -> Optional[str]:
        """
        Generate summary of a text chunk using BART.
        
        Args:
            chunk: Text chunk to summarize
            
        Returns:
            Summary text or None if error
        """
        try:
            # Validate input
            if not chunk or len(chunk.strip()) < 50:
                return None  # Skip chunks that are too short
            
            # Clean chunk - remove any special chars that cause issues
            chunk = chunk.strip()
            chunk = ' '.join(chunk.split())  # Normalize whitespace
            
            # BART has strict token limits - ensure we're within bounds
            # BART tokenizer can handle ~1024 tokens max
            words = chunk.split()
            if len(words) > 800:  # Conservative limit (words != tokens)
                chunk = ' '.join(words[:800])
            elif len(words) < 20:  # Skip very short chunks
                return None
            
            # Adjust min_length based on input length
            input_words = len(chunk.split())
            adjusted_min_length = min(self.min_summary_length, max(20, input_words // 4))
            adjusted_max_length = min(self.max_summary_length, input_words)
            
            # Skip if input is too short for meaningful summarization
            if adjusted_max_length < 30:
                return chunk[:500]  # Return truncated original
            
            summary = self.summarizer(
                chunk,
                max_length=adjusted_max_length,
                min_length=adjusted_min_length,
                do_sample=False,
                truncation=True
            )
            
            if summary and len(summary) > 0 and summary[0].get('summary_text'):
                return summary[0]['summary_text']
            return None
            
        except Exception as e:
            # Silent skip for expected errors, but return None for robust handling
            return None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of text using DistilRoBERTa.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with 'label' (positive/negative/neutral) and 'score' (confidence)
        """
        try:
            # Validate input
            if not text or len(text.strip()) < 10:
                return {'label': 'neutral', 'confidence': 0.5}
            
            # Clean and truncate text
            text = text.strip()
            text = ' '.join(text.split())  # Normalize whitespace
            
            # DistilRoBERTa max 512 tokens - be conservative
            words = text.split()
            if len(words) > 400:
                text = ' '.join(words[:400])
            
            result = self.sentiment_analyzer(text, truncation=True, max_length=512)
            
            if result and len(result) > 0 and 'label' in result[0]:
                return {
                    'label': result[0]['label'].lower(),
                    'confidence': result[0]['score']
                }
            return {'label': 'neutral', 'confidence': 0.5}
            
        except Exception as e:
            # Silent fallback to neutral for robustness
            return {'label': 'neutral', 'confidence': 0.5}
    
    def score_transcript(self, 
                        transcript: Union[Dict, str],
                        use_md_a_only: bool = False) -> Dict:
        """
        Score an earnings transcript using BERT/BART pipeline.
        
        Process:
        1. Clean and chunk transcript
        2. Summarize each chunk with BART
        3. Analyze sentiment of summaries with DistilRoBERTa
        4. Aggregate to final economy score (1-5)
        
        Args:
            transcript: Dict with 'full_text' key or string
            use_md_a_only: Whether to use only MD&A section (not implemented yet)
            
        Returns:
            Dict with 'firm_score', 'final_score', 'confidence', 'sentiment_breakdown'
        """
        # Extract text
        if isinstance(transcript, dict):
            text = transcript.get('full_text', '')
        else:
            text = str(transcript)
        
        if not text or len(text) < 100:
            return {
                'firm_score': 3.0,
                'final_score': 3.0,
                'confidence': 0.0,
                'sentiment_breakdown': {'neutral': 1.0},
                'error': 'Empty or too short transcript'
            }
        
        # Clean and chunk
        cleaned_text = self.clean_transcript(text)
        chunks = self.chunk_text(cleaned_text)
        
        # Summarize chunks
        summaries = []
        for chunk in chunks:
            summary = self.summarize_chunk(chunk)
            if summary:
                summaries.append(summary)
        
        if not summaries:
            # Fallback: analyze raw chunks if summarization fails
            summaries = chunks[:5]  # Use first 5 chunks
        
        # Analyze sentiment of summaries
        sentiment_results = []
        for summary in summaries:
            sentiment = self.analyze_sentiment(summary)
            sentiment_results.append(sentiment)
        
        # Aggregate to final score
        scores = []
        confidences = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for result in sentiment_results:
            label = result['label']
            confidence = result['confidence']
            
            # Map sentiment to score
            base_score = self.sentiment_to_score.get(label, 3.0)
            
            # Adjust score based on confidence
            # High confidence positive/negative gets boosted/reduced
            if label == 'positive' and confidence > 0.8:
                adjusted_score = min(5.0, base_score + 0.5)
            elif label == 'negative' and confidence > 0.8:
                adjusted_score = max(1.0, base_score - 0.5)
            else:
                adjusted_score = base_score
            
            scores.append(adjusted_score)
            confidences.append(confidence)
            sentiment_counts[label] += 1
        
        # Calculate final score (mean with confidence weighting)
        if scores:
            final_score = np.average(scores, weights=confidences)
            avg_confidence = np.mean(confidences)
        else:
            final_score = 3.0
            avg_confidence = 0.0
        
        # Sentiment breakdown
        total_sentiments = sum(sentiment_counts.values())
        sentiment_breakdown = {
            k: v / total_sentiments for k, v in sentiment_counts.items()
        } if total_sentiments > 0 else {'neutral': 1.0}
        
        return {
            'firm_score': round(final_score, 3),
            'final_score': round(final_score, 3),
            'score': round(final_score, 3),  # Multiple keys for compatibility
            'confidence': round(avg_confidence, 3),
            'sentiment_breakdown': sentiment_breakdown,
            'num_chunks': len(chunks),
            'num_summaries': len(summaries),
            'method': 'bert_bart_local'
        }
    
    def score_multiple_transcripts(self, 
                                   transcripts: List[Dict],
                                   show_progress: bool = True) -> List[Dict]:
        """
        Score multiple transcripts in batch.
        
        Args:
            transcripts: List of transcript dicts
            show_progress: Whether to show progress bar
            
        Returns:
            List of scoring results
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(transcripts, desc="Scoring transcripts")
            except ImportError:
                iterator = transcripts
                print(f"Scoring {len(transcripts)} transcripts...")
        else:
            iterator = transcripts
        
        for i, transcript in enumerate(iterator):
            result = self.score_transcript(transcript)
            results.append(result)
            
            if not show_progress and (i + 1) % 100 == 0:
                print(f"Scored {i + 1}/{len(transcripts)} transcripts")
        
        return results


# Convenience function for notebook usage
def create_bert_bart_scorer(use_gpu: bool = True) -> BertBartScorer:
    """
    Create and return a BERT/BART scorer instance.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        BertBartScorer instance
    """
    return BertBartScorer(use_gpu=use_gpu)


if __name__ == "__main__":
    # Test the scorer
    print("Testing BERT/BART Scorer...\n")
    
    scorer = create_bert_bart_scorer(use_gpu=True)
    
    # Test transcript
    test_transcript = {
        'full_text': """
        CEO: I'm pleased to report strong financial performance this quarter.
        The US economy continues to show resilience despite some headwinds.
        We see positive momentum in consumer spending and business investment.
        Our outlook for the next quarter remains optimistic.
        
        CFO: Revenue growth was 15% year-over-year.
        Margins expanded due to operational efficiencies.
        We're seeing strong demand across all segments.
        
        Q&A:
        Q: What's your outlook on the economy?
        A: We remain cautiously optimistic about near-term growth.
        Consumer confidence is improving and business investment is recovering.
        """
    }
    
    print("Scoring test transcript...")
    result = scorer.score_transcript(test_transcript)
    
    print("\nResults:")
    print(f"  Score: {result['firm_score']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Sentiment Breakdown: {result['sentiment_breakdown']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Summaries: {result['num_summaries']}")
    print(f"  Method: {result['method']}")
    
    print("\n✓ Test complete!")
