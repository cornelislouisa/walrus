# Morphogenesis analysis modules

The notebooks import their reusable logic from this package:

- `checkpoint_analysis.py`: W&B run lookup, checkpoint selection, trainer
  reconstruction, cached evaluation, zero-shot comparisons, and VF-aligned cohorts.
- `rollout_cache.py`: manifest-verified physical-unit rollout artifacts, isolated by
  entity, project, model, run, dataset, split, checkpoint, and inference settings.
- `rollout_video.py`: cached rollout video layouts and shared rendering defaults.
- `flow_metrics.py`: RMS velocity, flow autocorrelation, temporal landmarks, and
  grouped plots.
- `paper_residual.py`: the paper-style normalized velocity residual and cohort
  aggregation.
- `plot_style.py`: common publication figure sizes and typography.

`demo_notebooks/analyze_checkpoints.ipynb` is the in-distribution entry point.
`demo_notebooks/zero_shot_eval.ipynb` is the cross-dataset entry point. Plotting
should consume `CachedRollout` objects instead of invoking a model again.

See [`MORPHOGENESIS.md`](../../MORPHOGENESIS.md) for the complete data, training,
evaluation, and handoff workflow.
