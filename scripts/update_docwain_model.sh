#!/bin/bash
# Update DocWain private model on Ollama registry
# Run periodically to incorporate system prompt improvements
#
# Usage: ./scripts/update_docwain_model.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELFILE="$PROJECT_DIR/Modelfile"
MODEL_NAME="MuthuSubramanian/DocWain"

echo "=== DocWain Model Update ==="
echo "Base: qwen3:8b"
echo "Modelfile: $MODELFILE"
echo "Target: $MODEL_NAME"
echo ""

# Ensure base model is up to date
echo "[1/4] Pulling latest base model..."
ollama pull qwen3:8b

# Recreate from Modelfile (picks up any system prompt changes)
echo "[2/4] Creating model from Modelfile..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

# Push to registry
echo "[3/4] Pushing to Ollama registry..."
ollama push "$MODEL_NAME"

# Verify
echo "[4/4] Verifying..."
ollama list | grep -i docwain

echo ""
echo "=== Update complete ==="
echo "Model available at: https://ollama.com/$MODEL_NAME"
