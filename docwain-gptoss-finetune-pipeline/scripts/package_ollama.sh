#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python main.py export-adapter
python main.py package-ollama

echo "Packaging complete."
