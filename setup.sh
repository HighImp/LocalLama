#!/bin/bash

# Check if ollama is installed
if ! command -v ollama &> /dev/null
then
  echo "Ollama is not installed. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "Ollama is already installed."
fi

# Create virtual environment
python3 -m venv .venv
source ./.venv/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
  mkdir -p data
  echo "# Local Information:" > data/example.txt
  echo "Hostname: $(hostname)" >> data/example.txt
  echo "OS: $(uname -s)" >> data/example.txt
  chmod -R 755 data  # Ensure the correct path is used here
fi

# Run the Python script
python starter.py 

# Notify that the setup is done
echo "Setup done"
