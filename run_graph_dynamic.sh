#!/usr/bin/env bash
set -euo pipefail

# DEPRECATED (2026-05-31):
# Dynamic/on-the-fly KG experiment pipeline is archived due to severe quality regression.
# This script is intentionally blocked by default to avoid accidental reuse.
# Override only for archival/debug purposes:
#   ALLOW_DEPRECATED_DYNAMIC_KG=1 bash run_graph_dynamic.sh ...
if [ "${ALLOW_DEPRECATED_DYNAMIC_KG:-0}" != "1" ]; then
  echo "[DEPRECATED] run_graph_dynamic.sh has been archived and is disabled by default."
  echo "[DEPRECATED] Reason: dynamic KG pipeline produced unstable/degenerate graphs and large accuracy drop."
  echo "[DEPRECATED] If you must reproduce legacy behavior, run with:"
  echo "  ALLOW_DEPRECATED_DYNAMIC_KG=1 bash run_graph_dynamic.sh ..."
  exit 2
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: bash run_graph_dynamic.sh <dataset: csqa|obqa> <split: train|dev|test|all> [mode:auto|llm|heuristic] [limit]"
  echo "Example: bash run_graph_dynamic.sh csqa dev auto"
  echo "Example: OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini bash run_graph_dynamic.sh csqa test llm 200"
  exit 1
fi

dataset="$1"
split="$2"
mode="${3:-auto}"
limit="${4:--1}"
timeout_sec="${TIMEOUT_SEC:-90}"
retries="${RETRIES:-4}"
retry_sleep_sec="${RETRY_SLEEP_SEC:-3}"
checkpoint_every="${CHECKPOINT_EVERY:-10}"
allow_failures="${ALLOW_FAILURES:-0}"
log_to_file="${LOG_TO_FILE:-1}"
log_file="${LOG_FILE:-terminal.log}"

# Stream logs to both terminal and file by default.
# Disable with LOG_TO_FILE=0, or customize path with LOG_FILE=...
if [ "${log_to_file}" = "1" ] && [ -z "${__QAGNN_DYNAMIC_LOG_TEE:-}" ]; then
  export __QAGNN_DYNAMIC_LOG_TEE=1
  mkdir -p "$(dirname "${log_file}")"
  touch "${log_file}"
  exec > >(tee -a "${log_file}") 2>&1
  echo "[log] appending output to ${log_file}"
fi

# Auto-setup proxy for unstable DNS/TUN environments.
# Disable by setting AUTO_PROXY=0.
auto_proxy="${AUTO_PROXY:-1}"
proxy_addr="${CLASH_PROXY_ADDR:-127.0.0.1:8890}"
if [ "${auto_proxy}" = "1" ]; then
  if [ -z "${http_proxy:-}" ] && [ -z "${HTTP_PROXY:-}" ] && [ -z "${https_proxy:-}" ] && [ -z "${HTTPS_PROXY:-}" ]; then
    export http_proxy="http://${proxy_addr}"
    export https_proxy="http://${proxy_addr}"
    export HTTP_PROXY="${http_proxy}"
    export HTTPS_PROXY="${https_proxy}"
    export no_proxy="127.0.0.1,localhost"
    export NO_PROXY="${no_proxy}"
    echo "[net] proxy env was empty, auto-set to ${proxy_addr}"
  else
    echo "[net] using existing proxy env"
  fi
fi

concept_path="data/cpnet/concept.txt"
cache_dir=".cache/dynamic_kg/${dataset}"
out_dir="data/${dataset}/graph_dynamic"
mkdir -p "${out_dir}"

run_one() {
  local sp="$1"
  local grounded="data/${dataset}/grounded/${sp}.grounded.jsonl"
  local statement="data/${dataset}/statement/${sp}.statement.jsonl"
  local out_pk="${out_dir}/${sp}.graph.adj.pk"
  echo "[run] ${dataset}/${sp} mode=${mode} limit=${limit} timeout=${timeout_sec}s retries=${retries}"
  local cmd=(
    conda run --no-capture-output -n amem_env python -u scripts/generate_dynamic_graph_from_llm.py
    --grounded-path "${grounded}" \
    --statement-path "${statement}" \
    --concept-path "${concept_path}" \
    --output-path "${out_pk}" \
    --cache-dir "${cache_dir}" \
    --mode "${mode}" \
    --limit "${limit}" \
    --timeout-sec "${timeout_sec}" \
    --retries "${retries}" \
    --retry-sleep-sec "${retry_sleep_sec}" \
    --checkpoint-every "${checkpoint_every}" \
    --resume
  )
  if [ "${allow_failures}" = "1" ]; then
    cmd+=(--allow-failures)
  fi
  "${cmd[@]}"
}

if [ "${split}" = "all" ]; then
  run_one train
  run_one dev
  run_one test
else
  run_one "${split}"
fi

echo "[done] dynamic graphs saved in ${out_dir}"
