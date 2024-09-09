#!/bin/bash
curl -fsSL https://ollama.com/install.sh | sh


python3 -m venv .venv
source ./.venv/bin/activate

pip -r install requirements.txt


if [ ! -d "data" ]; then
  mkdir -p data
  echo "# Local Information:" > data/example.txt
  echo "Hostname: $(hostname)" >> data/example.txt
  echo "OS: $(uname -s)" >> data/example.txt
  chmod -R 755 /path/to/data
fi

python starter.py 
echo "Setup done"