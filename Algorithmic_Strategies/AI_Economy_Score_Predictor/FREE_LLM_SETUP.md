# Free LLM Setup Guide

This guide explains how to use **free** LLM models instead of paid APIs (OpenAI, Anthropic).

## Option 1: Hugging Face Models (Recommended for Beginners)

Hugging Face models run directly in Python without any additional setup.

### Setup
1. Already installed in `requirements.txt`
2. No API keys needed
3. Works out of the box

### Configuration
Edit `config.yaml`:
```yaml
llm:
  provider: "huggingface"
  model: "facebook/opt-1.3b"  # or "google/flan-t5-large", "mistralai/Mistral-7B-Instruct-v0.2"
```

### Recommended Models
- **facebook/opt-1.3b**: Fast, lightweight (1.3B parameters)
- **google/flan-t5-large**: Good instruction following (780M parameters)
- **mistralai/Mistral-7B-Instruct-v0.2**: Better quality, slower (7B parameters, needs GPU)

### Hardware Requirements
- **CPU only**: Use `facebook/opt-1.3b` or `google/flan-t5-large`
- **GPU (recommended)**: Can use `mistralai/Mistral-7B-Instruct-v0.2`

---

## Option 2: Ollama (Recommended for Best Quality)

Ollama provides local LLM inference with easy model management.

### Setup
1. Install Ollama: https://ollama.ai/download
   ```bash
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # macOS
   brew install ollama
   
   # Windows: Download from https://ollama.ai/download
   ```

2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Pull a model:
   ```bash
   # Lightweight (1.3B parameters)
   ollama pull llama3.2:1b
   
   # Balanced (3.2B parameters) - RECOMMENDED
   ollama pull llama3.2
   
   # High quality (7B parameters)
   ollama pull mistral
   ```

### Configuration
Edit `config.yaml`:
```yaml
llm:
  provider: "ollama"
  model: "llama3.2"  # or "llama3.2:1b", "mistral", "phi3"
  ollama_url: "http://localhost:11434"
```

### Available Models
- **llama3.2:1b**: Fastest, lightweight (1.3B parameters)
- **llama3.2**: Best balance of speed/quality (3.2B parameters) ⭐ **RECOMMENDED**
- **mistral**: High quality (7B parameters)
- **phi3**: Microsoft's efficient model (3.8B parameters)

See all models: https://ollama.ai/library

---

## Comparison

| Feature | Hugging Face | Ollama | OpenAI/Anthropic |
|---------|--------------|---------|------------------|
| **Cost** | Free ✅ | Free ✅ | Paid ❌ |
| **Setup** | None | Install Ollama | API key needed |
| **Quality** | Good | Better | Best |
| **Speed** | Slow (CPU) | Fast | Fastest |
| **Privacy** | Local ✅ | Local ✅ | Cloud ❌ |
| **GPU Needed** | Optional | Optional | No |

---

## Testing Your Setup

Run this test:
```python
from llm_scorer import LLMScorer

scorer = LLMScorer("config.yaml")
test_text = "The US economy is showing strong growth with robust consumer spending."
score = scorer.score_text(test_text)
print(f"Score: {score}")  # Should print 4 or 5
```

---

## Troubleshooting

### Hugging Face: Out of Memory
- Use a smaller model: `facebook/opt-1.3b`
- Reduce chunk size in `config.yaml`: `chunk_size: 1000`

### Ollama: Connection Error
- Make sure Ollama is running: `ollama serve`
- Check the URL in config: `ollama_url: "http://localhost:11434"`

### Ollama: Model Not Found
- Pull the model first: `ollama pull llama3.2`
- List available models: `ollama list`

---

## Performance Tips

1. **For speed**: Use Ollama with `llama3.2:1b`
2. **For quality**: Use Ollama with `llama3.2` or `mistral`
3. **For minimal setup**: Use Hugging Face with `facebook/opt-1.3b`
4. **With GPU**: Use Hugging Face with `mistralai/Mistral-7B-Instruct-v0.2`

---

## Next Steps

After configuring, run the pipeline:
```python
# In notebook or Python script
from llm_scorer import LLMScorer
scorer = LLMScorer("config.yaml")

# Score transcripts
scored_df = scorer.score_dataframe(transcripts)
```

The scorer will automatically use your configured free model! 🎉
