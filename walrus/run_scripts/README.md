## Run script examples

These scripts show how to train, finetune, and evaluate Walrus with Hydra. Some are
Slurm templates; the morphogenesis scripts use local `torchrun`.

General examples:

- `eval_onegpu_example_walrus.sh`: evaluate pretrained weights with one Slurm GPU.
- `finetuning_example_distributed_walrus.sh`: distributed finetuning template.
- `pretrain_example_distributed_walrus.sh`: distributed pretraining template.
- `run_local_*`: local examples for Walrus and external models.

Morphogenesis experiments:

- `morphogenesis_finetune.sh`: published Walrus checkpoint finetuned on WT; Hydra
  overrides select myosin or scratch variants.
- `morphogenesis_sweep.sh`: deterministic Walrus hyperparameter sweep.
- `morphogenesis_resume_orig.sh`: portable legacy recipe for the historical
  128-by-128 WT data and three-frame context; not used for current experiments.
- `crps_finetune_latent_example.sh`: latent-noise/AdaLN CRPS finetuning from a
  deterministic morphogenesis checkpoint.
- `crps_finetune_latent_sweep.sh`: CRPS ablations across available GPUs.
- `baselines/ffno_morphogenesis.sh`: FFNO trained from scratch.
- `baselines/sinenet_morphogenesis.sh`: SineNet trained from scratch.
- `baselines/poseidon_morphogenesis.sh`: pretrained Poseidon-L finetuning.
- `baselines/mean_field_morphogenesis.sh`: constant source-training mean baseline.
- `baselines/advection_morphogenesis.sh`: autoregressive self-advection baseline.

See the repository-level [morphogenesis handoff guide](../../MORPHOGENESIS.md) for
data preparation, exact launch commands, W&B projects, analysis, and testing.