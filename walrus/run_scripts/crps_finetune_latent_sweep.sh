#!/bin/bash
# CRPS latent-noise hyperparam ablations on free GPUs.
# Wraps crps_finetune_latent_example.sh (experiment=crps_finetune_latent).
#
# Usage:
#   bash walrus/run_scripts/crps_finetune_latent_sweep.sh
# Optional:
#   GPUS="1 2" bash walrus/run_scripts/crps_finetune_latent_sweep.sh
#   DET_RUN=/path/to/det/run CKPT_STEP=step_40 bash walrus/run_scripts/crps_finetune_latent_sweep.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/runs/morphogenesis_crps_sweep}"
# Override with e.g. GPUS="1 2" if 0/3 are busy.
read -ra GPUS <<< "${GPUS:-1 2 3}"

# tag|hydra_override  (space-separated overrides OK after the |)
# 1) baseline LRs (same as crps_finetune_latent_example.sh defaults)
# 2) hotter new-params LR, common kept at 1e-4
# 3) cooler both LRs
# 4) denser AdaLN: every other block (20 blocks) — more mem than stride-5 default
# 5) AdaLN on all 40 blocks — heaviest; may OOM (drop NUM_SAMPLES if needed)
# All trials use trainer.max_epoch=100 (example script default is 50).
TRIALS=(
  "lr-common5e-5-new1e-4|++optimizer.common_params_lr=5e-5 ++optimizer.new_params_lr=1e-4"
  "lr-common1e-4-new5e-4|++optimizer.common_params_lr=1e-4 ++optimizer.new_params_lr=5e-4"
  "lr-common1e-5-new5e-5|++optimizer.common_params_lr=1e-5 ++optimizer.new_params_lr=5e-5"
  "noise-every-2|model.noise_blocks=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]"
  "noise-all-40|model.noise_blocks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]"
)

mkdir -p "${EXPERIMENT_DIR}"

run_trial() {
  local gpu="$1"
  local tag="$2"
  local override="$3"

  local run_dir="${EXPERIMENT_DIR}/Walrus_crps_morph_latent_${tag}"
  if [[ -d "${run_dir}/checkpoints/best" ]] || [[ -d "${run_dir}/checkpoints/last" ]] \
    || [[ -d "${run_dir}/finetune" ]]; then
    run_dir="${run_dir}_$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "${run_dir}"

  read -ra override_args <<< "${override}"
  echo "GPU ${gpu}: ${tag} -> ${run_dir} (${override})"

  CUDA_VISIBLE_DEVICES="${gpu}" NGPUS=1 \
    EXPERIMENT_DIR="${EXPERIMENT_DIR}" \
    TORCHELASTIC_ERROR_FILE="${run_dir}/torch_worker_log.json" \
    bash "${SCRIPT_DIR}/crps_finetune_latent_example.sh" \
      "name=Walrus_crps_morph_latent_${tag}" \
      auto_resume=False \
      trainer.max_epoch=100 \
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
