# Morphogenesis experiments

This fork adds the morphogenesis training, baseline, zero-shot evaluation, rollout
cache, video, and flow-analysis code used for the WT, temperature, and mutation
experiments. This document is the handoff checklist for reproducing those runs.

Generated checkpoints, caches, figures, videos, and notebook outputs are deliberately
not versioned. The repository contains the code and configurations needed to recreate
them.

## 1. Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[test,external_models]"
wandb login
```

The launch scripts also require `ffmpeg` on `PATH` for rollout videos. Training the
1.3B-parameter Walrus models requires a large-memory GPU. Set `CUDA_VISIBLE_DEVICES`
and `NGPUS` explicitly in the examples below.

The scripts activate `<repo>/.venv` themselves, so use that location for the virtual
environment unless you edit the activation line.

## 2. Data location and Well layout

The committed Hydra configurations expect the datasets under:

```text
/data/lcornelis/morphogenesis_data/
├── WT/
├── WT_myosin/
├── WT_small/
├── WT_17_degrees/
├── WT_17_degrees_myosin/
├── WT_27_degrees/
├── WT_27_degrees_myosin/
├── even_skipped_r13/
├── halo_twist_ey53/
└── spaetzle_a/
```

Each dataset must already have been converted to the Well HDF5 schema and must use
this layout:

```text
<dataset>/
├── data/
│   ├── train/
│   │   └── *.hdf5
│   ├── valid/
│   │   └── *.hdf5
│   └── test/
│       └── *.hdf5
└── stats.yaml
```

`compute_statistics.py` computes normalization statistics; it does **not** convert raw
movies or arrays into Well HDF5 files. Every HDF5 file must therefore already contain
valid Well metadata, field names, time-varying attributes, grids, and trajectories.
Velocity-only data should expose the two flattened velocity channels. Myosin data
should additionally expose the flattened components of `myosin_tensor`.
Files are discovered directly inside each split directory (nested directories are not
scanned), and each trajectory needs at least 11 frames for the default 10-frame
context plus one target.

You can validate a converted file before computing statistics:

```bash
python /home/louisa/code/the_well/scripts/check_thewell_formatting.py \
  /data/lcornelis/morphogenesis_data/<dataset>/data/train/<file>.hdf5
```

All three split directories are required by `MixedWellDataModule`. Use a genuine
held-out validation split for training and checkpoint selection. For evaluation-only
data that genuinely has no validation split, a `valid` link to `train` can satisfy the
loader, but it must not be used to claim held-out validation performance.

To store data elsewhere, either edit the `path` in the corresponding file under
`walrus/configs/data/`, or override it at launch:

```bash
data.module_parameters.well_dataset_info.morphodynamic_atlas.path=/absolute/path/to/dataset
```

## 3. Generate `stats.yaml` with The Well

The local Well checkout used for these experiments is:

```text
/home/louisa/code/the_well
```

The relevant files are:

- `/home/louisa/code/the_well/scripts/compute_stats.sh`
- `/home/louisa/code/the_well/scripts/compute_statistics.py`

The generated file is named **`stats.yaml`**, not `stats.sh`.

### Configure the statistics script

In `compute_statistics.py`, set `WELL_DATASETS` near the bottom to the dataset
directory names that need statistics. For example:

```python
WELL_DATASETS = [
    "WT",
    "WT_myosin",
    "WT_17_degrees",
    "WT_17_degrees_myosin",
]
```

The script reads:

```text
<THE_WELL_DIR>/<name>/data/train/
```

and writes:

```text
<THE_WELL_DIR>/<name>/stats.yaml
```

It computes `mean`, `std`, and `rms`, plus `mean_delta`, `std_delta`, and
`rms_delta`, from the **training split only**. It refuses to overwrite an existing
statistics file. Move or remove a stale `stats.yaml` deliberately before recomputing
it.

### Configure and run the shell wrapper

In `compute_stats.sh`, set:

```bash
THE_WELL_DIR="/data/lcornelis/morphogenesis_data/"
```

Then run:

```bash
cd /home/louisa/code/the_well/scripts
bash compute_stats.sh
```

The wrapper currently invokes one statistics worker. The equivalent direct command,
with optional parallelism across dataset names, is:

```bash
cd /home/louisa/code/the_well/scripts
python compute_statistics.py /data/lcornelis/morphogenesis_data -n 4
```

For a single dataset already laid out as `<root>/<name>/{data/train,valid,test}`,
this clone also has `scripts/get_one_stats.py`:

```bash
python scripts/get_one_stats.py /data/lcornelis/morphogenesis_data/WT stats.yaml
```

Before training, verify that every configured dataset has non-empty `train`, `valid`,
and `test` directories and that `stats.yaml` contains statistics for every predicted
field. In particular, myosin datasets need both `velocity` and `myosin_tensor`
entries, including their delta statistics.

## 4. Data configurations

The Hydra data name is the YAML filename without `.yaml`:

| Experiment data | Hydra name |
| --- | --- |
| WT velocity | `morphogenesis_WT` |
| WT velocity + myosin | `morphogenesis_WT_myosin` |
| Small WT subset | `morphogenesis_WT_small` |
| 17-degree velocity | `morphogenesis_WT_17_degrees` |
| 17-degree velocity + myosin | `morphogenesis_WT_17_degrees_myosin` |
| 27-degree velocity | `morphogenesis_WT_27_degrees` |
| 27-degree velocity + myosin | `morphogenesis_WT_27_degrees_myosin` |
| even-skipped mutation | `morphogenesis_even_skipped_r13` |
| halo/twist mutation | `morphogenesis_halo_twist_ey53` |
| spaetzle mutation | `morphogenesis_spaetzle_a` |

`morphogenesis_WT_old` and its bundled `WT_old_stats.yaml` exist only to reproduce
the historical 128-by-128 run. The `repo://` normalization path is resolved against
the clone root, so it is portable. This is not the current WT dataset. If HDF5
`dataset_name` attributes need rewriting, dry-run then apply
`scripts/fix_wt_old_dataset_name.py`. To check that a saved run still matches its
logged validation scores:

```bash
python scripts/verify_wt_old_matches_run.py --run-dir /path/to/saved/run --epoch 50
```

## 5. Train the models

Run these commands from `<repo>/walrus`. Replace the experiment directory with a
writable location. Hydra overrides appended to a script select another dataset,
project, run name, or training length without editing the script. Existing
launchers log to historically named W&B projects (`morphogenesis`,
`morphogenesis_myosin`, `morphogenesis_no_myosin`); override
`logger.wandb_project_name` if you want a different board.

### Pretrained Walrus finetuning

Velocity only:

```bash
CUDA_VISIBLE_DEVICES=0 NGPUS=1 \
EXPERIMENT_DIR=/path/to/runs/morphogenesis \
bash run_scripts/morphogenesis_finetune.sh
```

Velocity and myosin:

```bash
CUDA_VISIBLE_DEVICES=1 NGPUS=1 \
EXPERIMENT_DIR=/path/to/runs/morphogenesis \
bash run_scripts/morphogenesis_finetune.sh \
  data=morphogenesis_WT_myosin \
  name=Walrus_ft_morph_WT_myosin \
  logger.wandb_project_name=morphogenesis_myosin
```

The script downloads the published Walrus checkpoint into `<repo>/checkpoints` unless
`CHECKPOINT_PATH` and `CONFIG_PATH` point to local copies.

### Hyperparameter sweep

`run_scripts/morphogenesis_sweep.sh` wraps `morphogenesis_finetune.sh` and launches
a small grid of deterministic Walrus finetunes in parallel, one trial per GPU.
Each trial is the same pretrained WT recipe with a single Hydra override (or a
pair, for context length plus learning rate). The built-in trials are:

| Tag | Override |
| --- | --- |
| `lr5e-5` | `optimizer.lr=5e-5` |
| `n_steps-10` | `data.module_parameters.n_steps_input=10` |
| `n_steps-10-lr5e-5` | context 10 and `optimizer.lr=5e-5` |
| `wd1e-3` | `optimizer.weight_decay=1e-3` |
| `max_epoch-100` | `trainer.max_epoch=100` |

They were chosen to probe rollout validation after the default 8-frame setup:
lower learning rate or stronger weight decay against late overfitting, longer
context, and a shorter training budget near where the 8-frame run peaked.

Runs are named `Walrus_ft_morpho_<tag>` under `EXPERIMENT_DIR` (default
`<repo>/runs/morphogenesis`). If that directory already has `checkpoints/best` or
`checkpoints/last`, the script appends a timestamp instead of overwriting.
Stdout and stderr go to `<run_dir>/train.log`. GPUs are taken in waves of
`GPUS` (default `0 1 2 3`); the next wave starts only after the current wave
finishes.

```bash
cd <repo>/walrus
EXPERIMENT_DIR=/path/to/runs/morphogenesis \
GPUS="0 1 2" \
bash run_scripts/morphogenesis_sweep.sh
```

This is not the CRPS ablation launcher. That is
`run_scripts/crps_finetune_latent_sweep.sh`.

### Walrus from scratch

Use the same architecture and training settings while disabling the pretrained
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 NGPUS=1 \
EXPERIMENT_DIR=/path/to/runs/morphogenesis \
bash run_scripts/morphogenesis_finetune.sh \
  name=Walrus_scratch_morph_WT_no_myosin \
  logger.wandb_project_name=morphogenesis_no_myosin \
  finetune=False auto_resume=False experiment=no_overrides checkpoint=defaults \
  checkpoint.checkpoint_frequency=10 \
  ++checkpoint.coalesced_checkpoint_path=null \
  ++config_override=
```

For the myosin version, additionally pass:

```text
data=morphogenesis_WT_myosin
name=Walrus_scratch_morph_WT_myosin
logger.wandb_project_name=morphogenesis_myosin
```

Confirm that the log does not say `Loading coalesced checkpoint`.

### CRPS Walrus

CRPS training starts from a completed deterministic Walrus run:

```bash
CUDA_VISIBLE_DEVICES=0 NGPUS=1 \
EXPERIMENT_DIR=/path/to/runs/morphogenesis_crps \
DET_RUN=/path/to/deterministic/walrus/run \
CKPT_STEP=step_50 \
bash run_scripts/crps_finetune_latent_example.sh
```

The checkpoint is expected at
`$DET_RUN/checkpoints/$CKPT_STEP/full_checkpoint.pt`, with the resolved training
configuration at `$DET_RUN/extended_config.yaml`. Reduce `NUM_SAMPLES` if the
stochastic ensemble does not fit in memory. The ablation launcher is
`run_scripts/crps_finetune_latent_sweep.sh`.

### Learned baselines

```bash
CUDA_VISIBLE_DEVICES=0 NGPUS=1 EXPERIMENT_DIR=/path/to/runs/baselines \
  bash run_scripts/baselines/ffno_morphogenesis.sh

CUDA_VISIBLE_DEVICES=1 NGPUS=1 EXPERIMENT_DIR=/path/to/runs/baselines \
  bash run_scripts/baselines/sinenet_morphogenesis.sh

CUDA_VISIBLE_DEVICES=2 NGPUS=1 EXPERIMENT_DIR=/path/to/runs/baselines \
  bash run_scripts/baselines/poseidon_morphogenesis.sh
```

Poseidon-L downloads its published weights to `<repo>/checkpoints/Poseidon-L`.

For myosin, also override the dataset, run/project names, and channel counts:

```text
data=morphogenesis_WT_myosin
model.in_channels=6
model.out_channels=6
```

Poseidon uses `model.num_channels=6 model.num_out_channels=6` instead.

### Non-learned baselines

These models do not produce a differentiable training loss, so launch them in
validation mode and give each one an explicit fresh folder:

```bash
CUDA_VISIBLE_DEVICES=0 NGPUS=1 EXPERIMENT_DIR=/path/to/runs/baselines \
  bash run_scripts/baselines/mean_field_morphogenesis.sh \
  validation_mode=True auto_resume=False \
  ++folder_override=/path/to/runs/baselines/MeanField_morph_WT_eval

CUDA_VISIBLE_DEVICES=1 NGPUS=1 EXPERIMENT_DIR=/path/to/runs/baselines \
  bash run_scripts/baselines/advection_morphogenesis.sh \
  validation_mode=True auto_resume=False \
  ++folder_override=/path/to/runs/baselines/Advection_morph_WT_eval
```

The zero-shot analysis code identifies the mean-field model, computes its frozen mean
from the run's original training data, and then applies that source mean to the target
dataset. This avoids refitting the baseline on zero-shot data. Self-advection has no
fitted state: it applies the same fixed advection rule to the target's last context
frame and is autoregressive during rollout. For myosin, pass
`model.in_channels=6 model.out_channels=6` so all six channels are emitted. Use a new
`folder_override` for each data/channel variant.

## 6. Analyze checkpoints and zero-shot transfer

Start Jupyter from `demo_notebooks`:

```bash
cd <repo>/demo_notebooks
jupyter lab
```

- `analyze_checkpoints.ipynb` compares checkpoints and in-distribution runs.
- `zero_shot_eval.ipynb` evaluates a source-project checkpoint on another data config,
  caches physical-unit trajectories, and produces VRMSE, spectral, flow, distribution,
  video, temperature, mutation, and VF-aligned paper-residual analyses.

In `zero_shot_eval.ipynb`:

1. Set `PROJECT` to the source W&B project (`morphogenesis_no_myosin` or
   `morphogenesis_myosin`).
2. Set `RUNS` to the desired source runs, or execute the discovery cell.
3. Set `DATA` to a target Hydra data name and `ORIGINAL_DATA` to the matching WT
   source data.
4. Run `compare_zero_shot` once to populate the cache.
5. Run the plotting cells. Cache hits do not reload a model or rerun inference.

The cache is isolated by W&B entity, project, model, run, dataset, split, checkpoint,
and inference settings under `demo_notebooks/_analysis_cache`. Set
`REFRESH_CACHE=True` only when intentionally replacing a cache entry.

VRMSE averages normalized whole-field spatial error over predicted frames, channels,
and embryos. A velocity-only model therefore averages two velocity channels; a
velocity-plus-myosin model also includes all flattened myosin channels. Spectral plots
report normalized error in low-, medium-, and high-frequency bins. RMS-velocity plots
use velocity only and collapse each frame to a spatially averaged speed.

## 7. Verification

Run the focused tests before handing off changes:

```bash
cd <repo>
source .venv/bin/activate
pytest -q \
  tests/test_checkpoint_analysis.py \
  tests/test_rollout_cache.py \
  tests/test_rollout_video.py \
  tests/test_flow_metrics.py \
  tests/test_paper_residual.py \
  tests/test_simple_baselines.py \
  tests/test_ffno_baseline.py \
  tests/test_sinenet_baseline.py \
  tests/test_poseidon_baseline.py \
  tests/test_dataset.py
```

Generated files belong in the ignored `checkpoints/`, `runs/`,
`demo_notebooks/_analysis_cache/`, `demo_notebooks/figures/`, and `_analysis_viz/`
directories. Notebook outputs are cleared before committing; rerunning the notebooks
recreates them locally.
