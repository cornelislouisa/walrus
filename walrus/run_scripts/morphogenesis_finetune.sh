#!/bin/bash -l

# Local finetuning script for parka.
# Usage:
#   bash run_scripts/morphogenesis_finetune.sh
# Optional:
#   CUDA_VISIBLE_DEVICES=1 NGPUS=2 bash run_scripts/morphogenesis_finetune.sh
#   SKIP_DOWNLOAD=1 bash run_scripts/morphogenesis_finetune.sh
#   CHECKPOINT_PATH=/path/to/walrus.pt bash run_scripts/morphogenesis_finetune.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NGPUS="${NGPUS:-1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/runs/morphogenesis}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/walrus.pt}"
CONFIG_PATH="${CONFIG_PATH:-${CHECKPOINT_DIR}/extended_config.yaml}"
HF_REPO="polymathic-ai/walrus"
HF_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=torch_worker_log.json
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/bin:${PATH}"

cd "${SCRIPT_DIR}/.."

mkdir -p "${EXPERIMENT_DIR}" "${CHECKPOINT_DIR}"

download_file() {
  local dest="$1"
  local filename="$2"

  if [[ -f "${dest}" ]]; then
    echo "Found ${dest}"
    return 0
  fi

  if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    echo "ERROR: ${dest} not found and SKIP_DOWNLOAD=1."
    exit 1
  fi

  echo "Downloading ${filename} from HuggingFace (${HF_REPO})..."
  mkdir -p "$(dirname "${dest}")"

  if command -v hf >/dev/null 2>&1; then
    hf download "${HF_REPO}" "${filename}" --local-dir "${CHECKPOINT_DIR}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c "${HF_BASE_URL}/${filename}" -O "${dest}"
  else
    echo "ERROR: need 'hf' or 'wget' to download ${filename}."
    exit 1
  fi

  if [[ ! -f "${dest}" ]]; then
    echo "ERROR: failed to download ${filename} to ${dest}"
    exit 1
  fi
}

echo "Checking pretrained Walrus checkpoint..."
download_file "${CHECKPOINT_PATH}" "walrus.pt"
download_file "${CONFIG_PATH}" "extended_config.yaml"

CHECKPOINT_ARGS=(
  checkpoint=finetune
  experiment=finetune_example
  finetuning_mods=all
  auto_resume=False
  checkpoint.checkpoint_frequency=10
  "++checkpoint.coalesced_checkpoint_path=${CHECKPOINT_PATH}"
  "++config_override=${CONFIG_PATH}"
)

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPUS}" \
  train.py \
  distribution=local \
  server=local \
  model=isotropic_model \
  name=Walrus_ft_morph_WT_no_myosin \
  trainer=globalnorm \
  trainer.grad_acc_steps=1 \
  optimizer=adam \
  optimizer.lr=1.e-4 \
  logger.wandb_project_name="morphogenesis_no_myosin" \
  trainer.enable_amp=False \
  model.gradient_checkpointing_freq=0 \
  trainer.log_interval=10 \
  trainer.clip_gradient=10 \
  data.module_parameters.batch_size=1 \
  data.module_parameters.n_steps_input=10 \
  data.module_parameters.n_steps_output=1 \
  model/processor/space_mixing=full_spatial_attention \
  model.projection_dim=48 \
  model.intermediate_dim=352 \
  model.hidden_dim=1408 \
  model.groups=16 \
  model.processor_blocks=40 \
  model.drop_path=0.0 \
  model.processor.space_mixing.num_heads=16 \
  model.processor.time_mixing.num_heads=16 \
  model.causal_in_time=True \
  model.jitter_patches=True \
  data.module_parameters.max_samples=200 \
  trainer.short_validation_length=20 \
  trainer.max_rollout_steps=80 \
  lr_scheduler=inv_sqrt_w_sqrt_ramps \
  trainer.val_frequency=5 \
  trainer.rollout_val_frequency=5 \
  trainer.video_validation=True \
  trainer.image_validation=False \
  data.module_parameters.min_dt_stride=1 \
  data.module_parameters.max_dt_stride=1 \
  trainer.prediction_type="delta" \
  data=morphogenesis_WT \
  trainer.max_epoch=200 \
  data_workers=10 \
  model.override_dimensionality=0 \
  ++model.use_periodic_fixed_jitter=True \
  ++model.input_field_drop=0 \
  ++trainer.skip_spectral_metrics=True \
  ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
  ++experiment_dir="${EXPERIMENT_DIR}" \
  "${CHECKPOINT_ARGS[@]}" \
  "$@"
