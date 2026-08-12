#!/bin/bash
# Launch hyperparam ablations on free GPUs (machine has 0,1,2 — use 1 and 2).
# Usage:
#   bash walrus/run_scripts/morphogenesis_sweep.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/scr/louisa/walrus/runs/morphogenesis}"
GPUS=(0 1 2 3)

# tag|hydra_override  (space-separated overrides OK after the |)
# Aimed at improving rollout_valid over n_steps=8:
#   1) lower LR to curb late overfitting
#   2/3) push context to 10 (alone and with lower LR)
#   4) stronger weight decay for the same reason as (1)
#   5) stop near where n_steps=8 peaked (~epoch 70)
TRIALS=(
  "lr5e-5|optimizer.lr=5e-5"
  "n_steps-10|data.module_parameters.n_steps_input=10"
  "n_steps-10-lr5e-5|data.module_parameters.n_steps_input=10 optimizer.lr=5e-5"
  "wd1e-3|optimizer.weight_decay=1e-3"
  "max_epoch-100|trainer.max_epoch=100"
)

mkdir -p "${EXPERIMENT_DIR}"

run_trial() {
  local gpu="$1"
  local tag="$2"
  local override="$3"

  local run_dir="${EXPERIMENT_DIR}/Walrus_ft_morpho_${tag}"
  if [[ -d "${run_dir}/checkpoints/best" ]] || [[ -d "${run_dir}/checkpoints/last" ]]; then
    run_dir="${run_dir}_$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "${run_dir}"

  read -ra override_args <<< "${override}"
  echo "GPU ${gpu}: ${tag} -> ${run_dir} (${override})"

  CUDA_VISIBLE_DEVICES="${gpu}" NGPUS=1 \
    EXPERIMENT_DIR="${EXPERIMENT_DIR}" \
    TORCHELASTIC_ERROR_FILE="${run_dir}/torch_worker_log.json" \
    bash "${SCRIPT_DIR}/morphogenesis_finetune.sh" \
      "name=Walrus_ft_morpho_${tag}" \
      auto_resume=False \
      "++folder_override=${run_dir}" \
      "${override_args[@]}" \
      >"${run_dir}/train.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    echo "FAILED GPU ${gpu}: ${tag} (exit ${rc}) — see ${run_dir}/train.log"
  else
    echo "DONE  GPU ${gpu}: ${tag}"
  fi
  return "${rc}"
}

# Launch in waves of |GPUS|; wait for the whole wave before starting the next.
idx=0
n=${#TRIALS[@]}
while (( idx < n )); do
  pids=()
  for gpu in "${GPUS[@]}"; do
    (( idx >= n )) && break
    trial="${TRIALS[$idx]}"
    tag="${trial%%|*}"
    override="${trial#*|}"
    run_trial "${gpu}" "${tag}" "${override}" &
    pids+=($!)
    ((++idx))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
done

echo "All trials finished."
