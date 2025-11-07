#!/bin/bash

# setup.sh - Installation script for Terminology-Aware MT System

echo "🔧 Setting up Terminology-Aware MT System..."

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Download spaCy models
echo "🗣️ Downloading spaCy language models..."
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p data/tmx
mkdir -p data/glossary

echo "✅ Setup complete!"
echo "🚀 Next steps:"
echo "   1. Place your TMX files in data/tmx/"
echo "   2. Place your glossary Excel file in data/glossary/"
echo "   3. Run: python3 advanced_mt.py"
echo "   4. For LLM features: Install Ollama and pull a model (e.g., ollama pull llama3.1)"