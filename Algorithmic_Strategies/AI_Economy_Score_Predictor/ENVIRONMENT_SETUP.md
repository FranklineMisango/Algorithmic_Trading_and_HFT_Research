# Environment Setup

## API Keys Configuration

This project requires API keys for FRED and OpenAI. These should be stored as environment variables and **never** committed to git.

### Setup Steps

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```bash
   # FRED API Key
   FRED_API_KEY=your_actual_fred_api_key_here
   
   # OpenAI API Key
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Get your API keys:**
   - **FRED API**: Register at https://fred.stlouisfed.org/docs/api/api_key.html
   - **OpenAI API**: Get from https://platform.openai.com/api-keys

4. **Verify `.env` is in `.gitignore`:**
   The `.gitignore` file already includes `.env` to prevent accidental commits.

### How It Works

- The `config.yaml` file uses `${VARIABLE_NAME}` syntax for environment variables
- Python code loads `.env` using `python-dotenv` package
- Environment variables are expanded when the config is loaded
- Your actual API keys never appear in committed code

### Security Notes

⚠️ **Important:**
- Never commit `.env` to git
- Never share your API keys
- Rotate keys immediately if accidentally exposed
- Use the `.env.example` file to document required variables (without actual values)
