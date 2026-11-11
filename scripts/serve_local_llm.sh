#!/usr/bin/env bash
# Serve one local open-weight model with vLLM on port 8000 (OpenAI API).
#   scripts/serve_local_llm.sh qwen3.5-9b | llama3.1-8b | gemma3-12b
# The vLLM install lives outside this repo's venv (it pins its own torch);
# set VLLM to its binary if it is not at the default below.
set -euo pipefail
VLLM=${VLLM:-vllm}
case "${1:?which model}" in
  qwen3.5-9b)  MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit; EXTRA=(--language-model-only) ;;
  llama3.1-8b) MODEL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4; EXTRA=() ;;
  # 12B weights leave ~0.8 GB for the KV cache at 0.88 utilisation; an fp8
  # cache halves it so the full 8192-token context fits (Gemma only).
  gemma3-12b)  MODEL=gaunernst/gemma-3-12b-it-int4-awq; EXTRA=(--language-model-only --kv-cache-dtype fp8) ;;
  *) echo "unknown model $1" >&2; exit 2 ;;
esac
# 12 GB laptop GPU: short context (prompts are ~1-2k tokens), eager mode
# to skip CUDA-graph memory, and a fixed seed for repeatable sampling.
exec "$VLLM" serve "$MODEL" --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 --enforce-eager --seed 0 "${EXTRA[@]}"
