#!/bin/bash

set -euo pipefail

NPROCS="${1:-4}"
TOPK="${2:-2}"

export QAGNN_PRUNE_AC=1
export QAGNN_PRUNE_AC_TOPK="${TOPK}"
export GRAPH_DIR_SUFFIX="_prune_ac"

echo "[ablation] NPROCS=${NPROCS}"
echo "[ablation] QAGNN_PRUNE_AC=${QAGNN_PRUNE_AC} TOPK=${QAGNN_PRUNE_AC_TOPK}"
echo "[ablation] GRAPH_DIR_SUFFIX=${GRAPH_DIR_SUFFIX}"

bash run_graph.sh csqa train "${NPROCS}"
bash run_graph.sh csqa dev "${NPROCS}"
bash run_graph.sh csqa test "${NPROCS}"

bash run_qagnn__csqa.sh
