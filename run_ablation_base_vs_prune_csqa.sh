#!/bin/bash

set -euo pipefail

NPROCS="${1:-4}"
TOPK="${2:-2}"
UNFREEZE="${3:-1000}"
EPOCHS="${4:-8}"

echo "[ablation] NPROCS=${NPROCS} TOPK=${TOPK} UNFREEZE=${UNFREEZE} EPOCHS=${EPOCHS}"

echo "[1/4] Build/Reuse base graph"
export GRAPH_DIR_SUFFIX=""
unset QAGNN_PRUNE_AC
unset QAGNN_PRUNE_AC_TOPK
bash run_graph.sh csqa train "${NPROCS}"
bash run_graph.sh csqa dev "${NPROCS}"
bash run_graph.sh csqa test "${NPROCS}"

echo "[2/4] Train base graph"
UNFREEZE_EPOCH="${UNFREEZE}" N_EPOCHS="${EPOCHS}" GRAPH_DIR_SUFFIX="" bash run_qagnn__csqa.sh

echo "[3/4] Build/Reuse pruned graph"
export GRAPH_DIR_SUFFIX="_prune_ac"
export QAGNN_PRUNE_AC=1
export QAGNN_PRUNE_AC_TOPK="${TOPK}"
bash run_graph.sh csqa train "${NPROCS}"
bash run_graph.sh csqa dev "${NPROCS}"
bash run_graph.sh csqa test "${NPROCS}"

echo "[4/4] Train pruned graph"
UNFREEZE_EPOCH="${UNFREEZE}" N_EPOCHS="${EPOCHS}" GRAPH_DIR_SUFFIX="_prune_ac" bash run_qagnn__csqa.sh

echo "[done] Compare runs under logs/ and saved_models/csqa/"
