#!/bin/sh

# This script starts the vLLM server with the base model, ready to accept LoRA requests.

set -e

# --- Configuration ---
# This should be the base model you used for fine-tuning.
BASE_MODEL="NousResearch/Llama-3.2-1B"

# The maximum rank of the LoRA adapters you will be loading.
# From your config.yaml, this is 64.
MAX_LORA_RANK=64

# The host and port for the server.
HOST="0.0.0.0"
PORT=8000
# --- End Configuration ---

echo "Starting vLLM server for base model: $BASE_MODEL"

# Allow dynamic loading/unloading of LoRA adapters at runtime.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank "$MAX_LORA_RANK" \
    --trust-remote-code
