#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
dt=`date '+%Y%m%d_%H%M%S'`
PYTHON_BIN=${PYTHON_BIN:-/home/tsdhyan/miniconda3/envs/amem_env/bin/python}


dataset="csqa"
model='roberta-large'
args=$@
ent_emb_source="${ENT_EMB_SOURCE:-tzw}"
graph_dir="data/${dataset}/graph${GRAPH_DIR_SUFFIX:-}"


echo "******************************"
echo "dataset: $dataset"
echo "ent_emb: $ent_emb_source"
echo "graph_dir: $graph_dir"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref

if [[ "$ent_emb_source" == "tzw" ]]; then
  "$PYTHON_BIN" scripts/verify_cpnet_assets.py \
    --ent-path data/cpnet/tzw.ent.npy \
    --concept-path data/cpnet/concept.txt || {
      echo "[ERROR] data/cpnet/tzw.ent.npy is not a valid full embedding."
      echo "[ERROR] Please replace it with a complete original file before evaluation."
      echo "[ERROR] If you only want sanity check, use: ENT_EMB_SOURCE=tzw_repaired bash eval_qagnn__csqa.sh"
      exit 1
    }
fi

###### Eval ######
"$PYTHON_BIN" -u qagnn.py --dataset $dataset \
      --train_adj ${graph_dir}/train.graph.adj.pk \
      --dev_adj   ${graph_dir}/dev.graph.adj.pk \
      --test_adj  ${graph_dir}/test.graph.adj.pk \
      --ent_emb $ent_emb_source \
      --train_statements data/${dataset}/statement/train.statement.jsonl \
      --dev_statements   data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir saved_models \
      --mode eval_detail \
      --load_model_path saved_models/csqa_model_hf3.4.0.pt \
      $args
