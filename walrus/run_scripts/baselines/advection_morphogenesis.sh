#!/bin/bash -l

# Velocity self-advection baseline on morphogenesis WT (same data as FFNO/SineNet).
# This model has no differentiable parameters, so invoke this script with
# validation_mode=True and an explicit folder_override (see MORPHOGENESIS.md).
#
# The baseline is natively 2D, so the loader is told to keep the data 2D
# (pad_cartesian_data_to_d=2) rather than inflating it to 3D like Walrus needs.
#
# Usage:
#   bash run_scripts/baselines/advection_morphogenesis.sh \
#     validation_mode=True auto_resume=False \
#     ++folder_override=/path/to/runs/Advection_morph_WT_eval
# Optional:
#   CUDA_VISIBLE_DEVICES=1 NGPUS=1 bash run_scripts/baselines/advection_morphogenesis.sh
#   model.dt=0.5 bash run_scripts/baselines/advection_morphogenesis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NGPUS="${NGPUS:-1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/runs/morphogenesis_baselines}"

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

mkdir -p "${EXPERIMENT_DIR}"

echo "Advection baseline on morphogenesis WT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NGPUS=${NGPUS}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPUS}" \
  train.py \
  distribution=local \
  server=local \
  model=advection \
  name=Advection_morph_WT \
  finetune=False \
  auto_resume=False \
  checkpoint=defaults \
  checkpoint.checkpoint_frequency=1 \
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
  data.module_parameters.n_steps_input=10 \
  data.module_parameters.n_steps_output=1 \
  data.module_parameters.max_samples=200 \
  trainer.short_validation_length=20 \
  trainer.max_rollout_steps=80 \
  lr_scheduler=inv_sqrt_w_sqrt_ramps \
  trainer.val_frequency=1 \
  trainer.rollout_val_frequency=1 \
  trainer.video_validation=True \
  trainer.image_validation=False \
  data.module_parameters.min_dt_stride=1 \
  data.module_parameters.max_dt_stride=1 \
  trainer.prediction_type="full" \
  data=morphogenesis_WT \
  ++data.module_parameters.dataset_kws.pad_cartesian_data_to_d=2 \
  trainer.max_epoch=1 \
  data_workers=10 \
  ++trainer.skip_spectral_metrics=True \
  ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
  ++experiment_dir="${EXPERIMENT_DIR}" \
  "$@"
