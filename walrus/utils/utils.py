"""Shared utility helpers."""

import torch


def load_common_weights(model, checkpoint_state_dict, strict=False, verbose=True):
    """
    Load common weights from a checkpoint into a model, ignoring missing or extra parameters.

    Used for CRPS finetuning when new stochastic layers are added on top of a
    deterministic checkpoint.

    Returns:
        dict with 'loaded', 'missing', 'unexpected', and 'size_mismatches' keys
    """
    model_state_dict = model.state_dict()
    model_param_names = set(model_state_dict.keys())
    checkpoint_param_names = set(checkpoint_state_dict.keys())

    common_keys = model_param_names & checkpoint_param_names
    missing_keys = model_param_names - checkpoint_param_names
    unexpected_keys = checkpoint_param_names - model_param_names

    loaded_keys = []
    size_mismatches = []

    with torch.no_grad():
        for key in common_keys:
            checkpoint_param = checkpoint_state_dict[key]
            model_param = model_state_dict[key]

            if model_param.shape == checkpoint_param.shape:
                model_param.copy_(checkpoint_param)
                loaded_keys.append(key)
            else:
                size_mismatches.append(key)
                if verbose:
                    print(
                        f"Size mismatch for {key}: model {model_param.shape} "
                        f"vs checkpoint {checkpoint_param.shape}"
                    )

    if verbose:
        print(f"Loaded {len(loaded_keys)} common parameters")
        print(f"Missing in checkpoint: {len(missing_keys)} parameters")
        print(f"Extra in checkpoint: {len(unexpected_keys)} parameters")
        print(f"Size mismatches: {len(size_mismatches)} parameters")
        if missing_keys:
            print("Missing parameters (will be randomly initialized):")
            for key in sorted(missing_keys):
                print(f"  - {key}")

    if strict and (missing_keys or unexpected_keys or size_mismatches):
        raise RuntimeError(
            f"strict load failed: missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}, mismatches={len(size_mismatches)}"
        )

    return {
        "loaded": loaded_keys,
        "missing": list(missing_keys),
        "unexpected": list(unexpected_keys),
        "size_mismatches": size_mismatches,
    }


def has_additional_params(cfg_node) -> bool:
    """Return True if a model subconfig enables noise / conditional-norm params."""
    if cfg_node is None:
        return False
    for attr in ("noise_cond_dim", "norm_cond_dim"):
        if getattr(cfg_node, attr, 0):
            return True
    return False
