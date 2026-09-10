# Morphogenesis baselines

These wrappers adapt comparison models to the Walrus trainer tensor convention:

- `ffno/`: factorized FNO, 10 input frames, residual target.
- `sinenet/`: SineNet, 10 input frames, residual target.
- `advection/`: non-learned semi-Lagrangian self-advection. It uses the last context
  frame and feeds predictions back autoregressively during rollout.
- `mean_field/`: non-learned constant mean field. The zero-shot analysis installs a
  mean computed from the source run's original training data before applying it to a
  target dataset.

Poseidon-L uses `ScOTWrapper` in `walrus/models/baseline_wrappers.py`. It takes one
input frame and predicts the absolute next state.

Model defaults live in `walrus/configs/model/`; launchers live in
`walrus/run_scripts/baselines/`. See [`MORPHOGENESIS.md`](../../MORPHOGENESIS.md) for
data layouts, channel overrides, and commands.
