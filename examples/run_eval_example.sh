#!/usr/bin/env bash
# Example: evaluate one method (`<method_name>`) on EntityBench.
# Replace the paths below with your own locations before running.

set -euo pipefail

# ---- VLM credentials (Azure-OpenAI-compatible endpoint) -------------------
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export GEMINI_API_KEYS="key1,key2,key3"            # comma-separated, rotated

# ---- Inputs ---------------------------------------------------------------
METHOD_NAME="my_method"
RESULTS_DIR="/path/to/generated_videos/${METHOD_NAME}"   # <ep>/<scene>_<shot>.mp4
SCRIPTS_DIR="../data/scripts"
SPLIT_JSON="../data/splits/final_split_validated_ids.json"
OUT_DIR="eval_results/${METHOD_NAME}"

# ---- Multi-GPU (recommended) ---------------------------------------------
python ../eval/run_eval_distributed.py \
  --n_gpus 8 \
  --results_dir "$RESULTS_DIR" \
  --scripts_dir "$SCRIPTS_DIR" \
  --split_json  "$SPLIT_JSON" \
  --out_dir     "$OUT_DIR" \
  --method_name "$METHOD_NAME" \
  --pillars 1,2,3 \
  --llm_concurrency 5 \
  --resume

# ---- Single GPU ----------------------------------------------------------
# python ../eval/evaluate_benchmark.py \
#   --results_dir "$RESULTS_DIR" --scripts_dir "$SCRIPTS_DIR" \
#   --split_json  "$SPLIT_JSON"  --out_dir "$OUT_DIR" \
#   --method_name "$METHOD_NAME" --pillars 1,2,3 \
#   --llm_concurrency 5 --resume
