#!/bin/bash -l

# CRPS-finetune the deterministically finetuned WT-myosin Walrus checkpoint
# using latent noise + AdaLN conditioning.
#
# Starts from the best rollout_valid VRMSE checkpoint of:
#   Walrus_ft_morph_WT_myosin-morph-delta-Isotr[Space-Adapt-]-AdamW-0.0001
#   (wandb project morphogenesis_myosin) -> step_60, full_VRMSE_T=all_mean=0.879
#
# Usage:
#   bash run_scripts/crps_finetune_latent_example.sh
# Optional:
#   CUDA_VISIBLE_DEVICES=0 NGPUS=1 bash run_scripts/crps_finetune_latent_example.sh
#   DET_RUN=/path/to/det/run CKPT_STEP=best bash run_scripts/crps_finetune_latent_example.sh
#   NUM_SAMPLES=2 bash run_scripts/crps_finetune_latent_example.sh   # if you OOM
#   EXPERIMENT=crps_finetune_latent_cond bash run_scripts/crps_finetune_latent_example.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default to GPU 1 (override with CUDA_VISIBLE_DEVICES=...).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

NGPUS="${NGPUS:-1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/scr/louisa/walrus/runs/morphogenesis_myosin_crps}"

# Deterministic WT-myosin finetune run we are starting from.
DET_RUN="${DET_RUN:-/scr/louisa/walrus/runs/morphogenesis/Walrus_ft_morph_WT_myosin-morph-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/3}"
# Prefer best-by-rollout_valid VRMSE (epoch 60), not the one-step `best` symlink (epoch 200).
CKPT_STEP="${CKPT_STEP:-step_60}"
COALESCED_CKPT="${COALESCED_CKPT:-${DET_RUN}/checkpoints/${CKPT_STEP}/full_checkpoint.pt}"
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-${DET_RUN}/extended_config.yaml}"

DATA_NAME="${DATA_NAME:-morphogenesis_WT_myosin}"
EXPERIMENT="${EXPERIMENT:-crps_finetune_latent}"

# Ensemble size used for the CRPS training loss. Every member is a full forward
# pass, so this multiplies activation memory.
NUM_SAMPLES="${NUM_SAMPLES:-4}"
VAL_ENSEMBLE="${VAL_ENSEMBLE:-4}"
# Which processor blocks get noise conditioning. The det model has 40 blocks and
# each conditioned block adds ~12M AdaLN params, so condition a strided subset.
NOISE_BLOCKS="${NOISE_BLOCKS:-[0,5,10,15,20,25,30,35]}"

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

for f in "${COALESCED_CKPT}" "${CONFIG_OVERRIDE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: ${f} not found."
    exit 1
  fi
done

mkdir -p "${EXPERIMENT_DIR}"

echo "CRPS starting from ${COALESCED_CKPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NGPUS=${NGPUS}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPUS}" \
  train.py \
  experiment="${EXPERIMENT}" \
  server=local \
  distribution=local \
  name=Walrus_crps_morph_WT_myosin_latent \
  finetune=True \
  auto_resume=False \
  checkpoint=finetune \
  checkpoint.checkpoint_frequency=10 \
  "++checkpoint.coalesced_checkpoint_path='${COALESCED_CKPT}'" \
  ++checkpoint.load_chkpt_after_finetuning_expansion=True \
  "++config_override='${CONFIG_OVERRIDE}'" \
  finetuning_mods=all \
  logger.wandb_project_name="morphogenesis_myosin" \
  trainer=globalnorm \
  trainer.loss_fn._target_=walrus.metrics.crps.CRPS \
  trainer.enable_amp=False \
  trainer.grad_acc_steps=1 \
  trainer.clip_gradient=10 \
  trainer.log_interval=10 \
  trainer.max_epoch=50 \
  trainer.val_frequency=5 \
  trainer.rollout_val_frequency=5 \
  trainer.short_validation_length=20 \
  trainer.max_rollout_steps=80 \
  trainer.prediction_type="delta" \
  trainer.video_validation=True \
  trainer.image_validation=False \
  "++trainer.validation_ensemble_size=${VAL_ENSEMBLE}" \
  ++trainer.max_num_samples=4 \
  ++trainer.skip_spectral_metrics=True \
  lr_scheduler=inv_sqrt_w_sqrt_ramps \
  optimizer=adam \
  optimizer.lr=1e-4 \
  ++optimizer.new_params_lr=1e-4 \
  ++optimizer.common_params_lr=5e-5 \
  data="${DATA_NAME}" \
  data_workers=10 \
  data.module_parameters.batch_size=1 \
  data.module_parameters.n_steps_input=10 \
  data.module_parameters.n_steps_output=1 \
  data.module_parameters.min_dt_stride=1 \
  data.module_parameters.max_dt_stride=1 \
  data.module_parameters.max_samples=200 \
  ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
  model=isotropic_model_with_noise \
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
  model.override_dimensionality=0 \
  model.gradient_checkpointing_freq=1 \
  ++model.use_periodic_fixed_jitter=True \
  ++model.input_field_drop=0 \
  "model.num_samples=${NUM_SAMPLES}" \
  model.noise_type=latent \
  model.noise_mode=global \
  model.noise_dim=32 \
  model.mlp_layers=2 \
  model.noise_layernorm=True \
  "model.noise_blocks=${NOISE_BLOCKS}" \
  model.processor.noise_cond_dim=32 \
  model.processor.norm_cond_dim=0 \
  ++experiment_dir="${EXPERIMENT_DIR}" \
  "$@"
