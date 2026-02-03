#!/bin/bash
# Quick setup script for Ollama (recommended free option)

echo "=================================================="
echo "  Ollama Setup for AI Economy Score Predictor"
echo "=================================================="
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
else
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed successfully"
fi

echo ""
echo "🚀 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 3

echo ""
echo "📥 Pulling recommended model (llama3.2)..."
echo "   This may take a few minutes on first run..."
ollama pull llama3.2

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Edit config.yaml and set:"
echo "   provider: 'ollama'"
echo "   model: 'llama3.2'"
echo ""
echo "2. Run your notebook or Python script"
echo ""
echo "Optional: Pull other models"
echo "  - Fast (1.3B):  ollama pull llama3.2:1b"
echo "  - Better (7B):  ollama pull mistral"
echo ""
echo "To stop Ollama: kill $OLLAMA_PID"
echo "=================================================="
