#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
dt=`date '+%Y%m%d_%H%M%S'`
PYTHON_BIN=${PYTHON_BIN:-/home/tsdhyan/miniconda3/envs/amem_env/bin/python}


dataset="csqa"
model='roberta-large'
args=$@
graph_dir="data/${dataset}/graph${GRAPH_DIR_SUFFIX:-}"


elr="1e-5"
dlr="1e-3"
bs="${BS:-64}"
mbs="${MBS:-2}"
n_epochs="${N_EPOCHS:-15}"
unfreeze_epoch="${UNFREEZE_EPOCH:-4}"
ent_emb_source="${ENT_EMB_SOURCE:-tzw}"
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


k=5 #num of gnn layers
gnndim=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "ent_emb: $ent_emb_source"
echo "unfreeze_epoch: $unfreeze_epoch"
echo "graph_dir: $graph_dir"
echo "******************************"

if [[ "$ent_emb_source" == "tzw" ]]; then
  "$PYTHON_BIN" scripts/verify_cpnet_assets.py \
    --ent-path data/cpnet/tzw.ent.npy \
    --concept-path data/cpnet/concept.txt || {
      echo "[ERROR] data/cpnet/tzw.ent.npy is not a valid full embedding."
      echo "[ERROR] Please replace it with a complete original file before training."
      echo "[ERROR] If you only want a pipeline sanity run, use: ENT_EMB_SOURCE=tzw_repaired bash run_qagnn__csqa.sh"
      exit 1
    }
fi

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
mkdir -p logs

###### Training ######
for seed in 0; do
  graph_tag="${GRAPH_DIR_SUFFIX:-base}"
  graph_tag="${graph_tag//\//_}"
  run_prefix="enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__emb${ent_emb_source}__ufz${unfreeze_epoch}__g${graph_tag}"
  latest_run_dir=$(find ${save_dir_pref}/${dataset} -maxdepth 1 -mindepth 1 -type d -name "${run_prefix}__*" 2>/dev/null | sort | tail -n 1)
  if [[ -n "${RUN_DIR:-}" ]]; then
    save_dir="${RUN_DIR}"
  elif [[ -n "$latest_run_dir" && -f "$latest_run_dir/checkpoint_last.pt" ]]; then
    save_dir="$latest_run_dir"
  else
    save_dir="${save_dir_pref}/${dataset}/${run_prefix}__${dt}"
  fi

  mkdir -p "$save_dir"
  log_name=$(basename "$save_dir")
  log_file="logs/train_${dataset}__${log_name}.log.txt"
  resume_args=()
  if [[ -f "$save_dir/checkpoint_last.pt" ]]; then
    resume_args+=(--resume_checkpoint_path "$save_dir/checkpoint_last.pt")
  fi

  "$PYTHON_BIN" -u qagnn.py \
      --dataset $dataset \
      --unfreeze_epoch $unfreeze_epoch \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed \
      --ent_emb $ent_emb_source \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 10 \
      --train_adj ${graph_dir}/train.graph.adj.pk \
      --dev_adj   ${graph_dir}/dev.graph.adj.pk \
      --test_adj  ${graph_dir}/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir "$save_dir" "${resume_args[@]}" $args \
      >> "$log_file" 2>&1
done
