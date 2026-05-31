#!/bin/bash

set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ "${1:-}" == "cleanup" ]]; then
  dataset="${2:-csqa}"
  split="${3:-train}"
  output_dir="data/${dataset}/graph${GRAPH_DIR_SUFFIX:-}"
  output_path="${output_dir}/${split}.graph.adj.pk"
  cache_dir="${output_path}.cache"
  rm -rf "${cache_dir}" "${output_path}.tmp"
  echo "Removed graph cache: ${cache_dir}"
  exit 0
fi

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

grounded_path="data/${dataset}/grounded/${split}.grounded.jsonl"
cpnet_graph_path="data/cpnet/conceptnet.en.pruned.graph"
cpnet_vocab_path="data/cpnet/concept.txt"
output_dir="data/${dataset}/graph${GRAPH_DIR_SUFFIX:-}"
output_path="${output_dir}/${split}.graph.adj.pk"

mkdir -p "${output_dir}"

echo "GRAPH_DIR_SUFFIX=${GRAPH_DIR_SUFFIX:-}"
echo "QAGNN_PRUNE_AC=${QAGNN_PRUNE_AC:-0} QAGNN_PRUNE_AC_TOPK=${QAGNN_PRUNE_AC_TOPK:-2}"

/home/tsdhyan/miniconda3/bin/conda run --no-capture-output -n amem_env \
  python -u -c "from utils.graph import generate_adj_data_from_grounded_concepts__use_LM; generate_adj_data_from_grounded_concepts__use_LM('${grounded_path}', '${cpnet_graph_path}', '${cpnet_vocab_path}', '${output_path}', ${nprocs})"
