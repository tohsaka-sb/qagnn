#!/bin/bash

set -euo pipefail

dataset="${1:-csqa}"
split="${2:-train}"
nprocs="${3:-4}"

case "$dataset" in
  csqa|obqa)
    ;;
  *)
    echo "Unsupported dataset: $dataset"
    echo "Usage: $0 [csqa|obqa] [train|dev|test] [nprocs]"
    exit 1
    ;;
esac

statement_path="data/${dataset}/statement/${split}.statement.jsonl"
output_path="data/${dataset}/grounded/${split}.grounded.jsonl"
cpnet_vocab_path="data/cpnet/concept.txt"
pattern_path="data/cpnet/matcher_patterns.json"

mkdir -p "data/${dataset}/grounded"

/home/tsdhyan/miniconda3/bin/conda run --no-capture-output -n amem_env \
  python -u -c "from utils.grounding import ground; ground('${statement_path}', '${cpnet_vocab_path}', '${pattern_path}', '${output_path}', ${nprocs})"
