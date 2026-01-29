# Setup Instructions

## Configuration File Setup

1. **Copy the example config file:**
   ```bash
   cp config.yaml.example config.yaml
   ```

2. **Add your API keys to config.yaml:**
   
   ### FRED API Key
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up for a free account
   - Copy your API key and replace `YOUR_FRED_API_KEY_HERE` in config.yaml
   
   ### OpenAI API Key
   - Visit: https://platform.openai.com/api-keys
   - Create a new API key
   - Copy your API key and replace `YOUR_OPENAI_API_KEY_HERE` in config.yaml

3. **Security Notes:**
   - ⚠️ **NEVER commit config.yaml to git** (it's already in .gitignore)
   - Only commit config.yaml.example with placeholder values
   - Keep your API keys private and secure
   - Rotate API keys if they are accidentally exposed

## Files

- `config.yaml.example` - Template configuration with placeholder values (safe to commit)
- `config.yaml` - Your actual configuration with real API keys (DO NOT commit)
