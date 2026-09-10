#!/bin/bash -l

# Finetune Poseidon-L on morphogenesis WT (velocity-only, same data as FFNO/SineNet).
# Loads a local HuggingFace snapshot from checkpoints/Poseidon-L/ (downloads if missing).
#
# ScOT is natively 2D, so the loader is told to keep the data 2D
# (pad_cartesian_data_to_d=2) rather than inflating it to 3D like Walrus needs.
#
# Usage:
#   bash run_scripts/baselines/poseidon_morphogenesis.sh
# Optional:
#   CUDA_VISIBLE_DEVICES=1 NGPUS=1 bash run_scripts/baselines/poseidon_morphogenesis.sh
#   SKIP_DOWNLOAD=1 bash run_scripts/baselines/poseidon_morphogenesis.sh
#   POSEIDON_CKPT=/path/to/Poseidon-L bash run_scripts/baselines/poseidon_morphogenesis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NGPUS="${NGPUS:-1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/runs/morphogenesis_baselines}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
POSEIDON_CKPT="${POSEIDON_CKPT:-${CHECKPOINT_DIR}/Poseidon-L}"
HF_REPO="camlab-ethz/Poseidon-L"
HF_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"
# Full Poseidon-L safetensors is ~2.5 GB; reject truncated / LFS-pointer stubs.
MIN_WEIGHT_BYTES="${MIN_WEIGHT_BYTES:-2000000000}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=torch_worker_log.json
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/bin:${PATH}"

cd "${SCRIPT_DIR}/../.."

mkdir -p "${EXPERIMENT_DIR}" "${POSEIDON_CKPT}"

download_poseidon_file() {
  local dest="$1"
  local filename="$2"

  if [[ -f "${dest}" ]]; then
    local size
    size="$(stat -c%s "${dest}" 2>/dev/null || stat -f%z "${dest}")"
    if [[ "${filename}" == "model.safetensors" || "${filename}" == "pytorch_model.bin" ]]; then
      if (( size < MIN_WEIGHT_BYTES )); then
        echo "WARNING: ${dest} is only ${size} bytes (expected >= ${MIN_WEIGHT_BYTES})."
        echo "         Likely still uploading / incomplete. Re-download or wait for transfer to finish."
        if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
          echo "ERROR: incomplete weights and SKIP_DOWNLOAD=1."
          exit 1
        fi
        rm -f "${dest}"
      else
        echo "Found ${dest} (${size} bytes)"
        return 0
      fi
    else
      echo "Found ${dest}"
      return 0
    fi
  fi

  if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    echo "ERROR: ${dest} not found and SKIP_DOWNLOAD=1."
    exit 1
  fi

  echo "Downloading ${filename} from HuggingFace (${HF_REPO})..."
  mkdir -p "$(dirname "${dest}")"

  if command -v hf >/dev/null 2>&1; then
    hf download "${HF_REPO}" "${filename}" --local-dir "${POSEIDON_CKPT}"
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

echo "Checking Poseidon-L checkpoint at ${POSEIDON_CKPT}..."
download_poseidon_file "${POSEIDON_CKPT}/config.json" "config.json"
if [[ -f "${POSEIDON_CKPT}/model.safetensors" ]] || [[ ! -f "${POSEIDON_CKPT}/pytorch_model.bin" ]]; then
  download_poseidon_file "${POSEIDON_CKPT}/model.safetensors" "model.safetensors"
else
  download_poseidon_file "${POSEIDON_CKPT}/pytorch_model.bin" "pytorch_model.bin"
fi

WEIGHT_SIZE="$(stat -c%s "${POSEIDON_CKPT}/model.safetensors" 2>/dev/null || \
  stat -c%s "${POSEIDON_CKPT}/pytorch_model.bin" 2>/dev/null || \
  stat -f%z "${POSEIDON_CKPT}/model.safetensors" 2>/dev/null || \
  stat -f%z "${POSEIDON_CKPT}/pytorch_model.bin")"
if (( WEIGHT_SIZE < MIN_WEIGHT_BYTES )); then
  echo "ERROR: Poseidon-L weights still incomplete (${WEIGHT_SIZE} bytes)."
  echo "       Wait for upload/download to finish (~2.5 GB), then re-run."
  exit 1
fi

echo "Poseidon-L finetune on morphogenesis WT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NGPUS=${NGPUS}"
echo "POSEIDON_CKPT=${POSEIDON_CKPT}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPUS}" \
  train.py \
  distribution=local \
  server=local \
  model=poseidon_morph \
  name=PoseidonL_ft_morph_WT \
  finetune=False \
  auto_resume=False \
  checkpoint=defaults \
  checkpoint.checkpoint_frequency=10 \
  experiment=no_overrides \
  trainer=globalnorm \
  trainer.grad_acc_steps=1 \
  optimizer=adam \
  optimizer.lr=1.e-4 \
  logger.wandb_project_name="morphogenesis" \
  trainer.enable_amp=False \
  trainer.log_interval=10 \
  trainer.clip_gradient=10 \
  data.module_parameters.batch_size=1 \
  data.module_parameters.n_steps_input=1 \
  data.module_parameters.n_steps_output=1 \
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
  trainer.prediction_type="full" \
  data=morphogenesis_WT \
  ++data.module_parameters.dataset_kws.pad_cartesian_data_to_d=2 \
  trainer.max_epoch=100 \
  data_workers=10 \
  ++model.from_pretrained="${POSEIDON_CKPT}" \
  ++model.gradient_checkpointing_freq=1 \
  ++trainer.skip_spectral_metrics=True \
  ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
  ++experiment_dir="${EXPERIMENT_DIR}" \
  "$@"
