"""CRPS helper functions."""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class CRPS(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) loss for ensemble predictions.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, predictions, target, metadata, mem_efficient: Optional[bool] = False
    ):
        """
        Compute CRPS loss for ensemble predictions

        Args:
            predictions: [B, N_ensemble, L,  H, [W], [D], C]
            target: [B, N_ensemble, L,  H, [W], [D], C]
            metadata: Metadata for the dataset
            mem_efficient: Whether to use memory-efficient computation (computes one time slice at a time)
        Returns:
            CRPS loss [B, L, C] averaged over spatial dimensions
        """
        if mem_efficient:
            return compute_crps_loss_mem_efficient(
                predictions,
                target,
                metadata,
            )
        else:
            return compute_crps_loss(
                predictions,
                target,
                metadata,
            )


class WeightedCRPS(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) loss for ensemble predictions.
    Computed as a weighted combination between a spatial and frequency component.
    """

    def __init__(self, frequency_weight: float = 0.5):
        super().__init__()
        self.frequency_weight = frequency_weight
        assert 0.0 <= self.frequency_weight <= 1.0, "frequency_weight must be in [0, 1]"

    def forward(
        self, predictions, target, metadata, mem_efficient: Optional[bool] = False
    ):
        """
        Compute CRPS loss for ensemble predictions

        Args:
            predictions: [B, N_ensemble, L,  H, [W], [D], C]
            target: [B, N_ensemble, L,  H, [W], [D], C]
            metadata: Metadata for the dataset
            mem_efficient: Whether to use memory-efficient computation (computes one time slice at a time)
        Returns:
            CRPS loss [B, L, C] averaged over spatial dimensions
        """
        if mem_efficient:
            spatial_crps = compute_crps_loss_mem_efficient(
                predictions,
                target,
                metadata,
            )
            frequency_crps = compute_frequency_crps_loss_mem_efficient(
                predictions,
                target,
                metadata,
            )
            return (
                1 - self.frequency_weight
            ) * spatial_crps + self.frequency_weight * frequency_crps
        else:
            spatial_crps = compute_crps_loss(
                predictions,
                target,
                metadata,
            )
            frequency_crps = compute_frequency_crps_loss(
                predictions,
                target,
                metadata,
            )
            return (
                1 - self.frequency_weight
            ) * spatial_crps + self.frequency_weight * frequency_crps


def compute_crps_loss(predictions, target, metadata):
    """
    Compute CRPS loss for ensemble predictions

    Args:
        predictions: [B, N_ensemble, L, H, [W], [D], C] - Ensemble predictions
        target: [B, N_ensemble, L, H, [W], [D], C] - Ground truth (same shape as predictions)
        metadata: Metadata for the dataset

    Returns:
        CRPS loss [B, L, C] averaged over spatial dimensions
    """

    deterministic = False
    # Verify input shapes
    assert (
        predictions.shape == target.shape
    ), f"Predictions {predictions.shape} and target {target.shape} must have same shape"
    assert (
        predictions.ndim >= 5
    ), f"Expected at least 5D tensor [B, N, L, H, C], got {predictions.ndim}D"

    # If we have deterministic predictions (N=1), add ensemble dimension
    num_spatial_dims = len(metadata.spatial_resolution)
    if len(predictions.shape) != 4 + num_spatial_dims:
        assert (
            len(predictions.shape) == 3 + num_spatial_dims
        ), f"Unexpected predictions shape {predictions.shape}"
        predictions = predictions.unsqueeze(1)
        target = target.unsqueeze(1)
        deterministic = True

    B, N, L = predictions.shape[:3]
    C = predictions.shape[-1]

    # Flatten spatial dimensions: [B, N, L, C, spatial_flat]
    predictions_flat = rearrange(predictions, "B N L ... C -> B N L C (...)")
    target_flat = rearrange(target, "B N L ... C -> B N L C (...)")

    # CRPS calculation
    # Term 1: E[|X - Y|] - expected absolute difference between prediction and target
    abs_diff = torch.abs(predictions_flat - target_flat)
    term1 = torch.mean(abs_diff, dim=1)  # Average over ensemble: [B, L, C, spatial]
    if deterministic or predictions.shape[1] == 1:  # Just MAE
        return term1.mean(dim=tuple(range(3, term1.ndim)))

    # Term 2: 0.5 * E[|X - X'|] - ensemble spread penalty
    # Note: Here, we are actually using the fairCRPS formula (Eq (2) from https://arxiv.org/pdf/2412.15832v1)
    # Create all pairwise differences within ensemble
    pred_expanded_1 = predictions_flat.unsqueeze(2)  # [B, N, 1, L, C, spatial]
    pred_expanded_2 = predictions_flat.unsqueeze(1)  # [B, 1, N, L, C, spatial]
    ensemble_diff = torch.abs(
        pred_expanded_1 - pred_expanded_2
    )  # [B, N, N, L, C, spatial]
    # The (N * (N-1)) comes from using the fairCRPS formula
    term2 = (
        0.5 * 1 / (N * (N - 1)) * torch.sum(ensemble_diff, dim=(1, 2))
    )  # Average over both ensemble dims: [B, L, C, spatial]

    # Final CRPS
    # Average over all spatial dimensions (dims 3 and beyond)
    crps = term1 - term2  # [B, L, C, spatial]
    crps = crps.mean(dim=tuple(range(3, crps.ndim)))

    return crps


def compute_crps_loss_mem_efficient(predictions, target, metadata):
    """
    Compute CRPS loss for ensemble predictions - memory efficient version where we compute one time slice at a time

    Args:
        predictions: [B, N_ensemble, L, H, [W], [D], C] - Ensemble predictions
        target: [B, N_ensemble, L, H, [W], [D], C] - Ground truth (same shape as predictions)
        metadata: Metadata for the dataset

    Returns:
        CRPS loss [B, L, C] averaged over spatial dimensions
    """

    deterministic = False
    # Verify input shapes
    assert (
        predictions.shape == target.shape
    ), f"Predictions {predictions.shape} and target {target.shape} must have same shape"
    assert (
        predictions.ndim >= 5
    ), f"Expected at least 5D tensor [B, N, L, H, C], got {predictions.ndim}D"

    # If we have deterministic predictions (N=1), add ensemble dimension
    num_spatial_dims = len(metadata.spatial_resolution)
    if len(predictions.shape) != 4 + num_spatial_dims:
        assert (
            len(predictions.shape) == 3 + num_spatial_dims
        ), f"Unexpected predictions shape {predictions.shape}"
        predictions = predictions.unsqueeze(1)
        target = target.unsqueeze(1)
        deterministic = True
    elif predictions.shape[1] == 1:
        deterministic = True

    B, N, L = predictions.shape[:3]
    C = predictions.shape[-1]

    # Flatten spatial dimensions: [B, N, L, C, spatial_flat]
    predictions_flat = rearrange(predictions, "B N L ... C -> B N L C (...)")
    target_flat = rearrange(target, "B N L ... C -> B N L C (...)")

    # Initialize list to store per-timestep CRPS values
    crps_per_timestep = []

    # Process each timestep separately to save memory
    for t in range(L):
        # Extract current timestep: [B, N, C, spatial]
        pred_t = predictions_flat[:, :, t, :, :]  # [B, N, C, spatial]
        target_t = target_flat[:, :, t, :, :]  # [B, N, C, spatial]

        # Term 1: E[|X - Y|] for this timestep
        abs_diff_t = torch.abs(pred_t - target_t)
        term1_t = torch.mean(
            abs_diff_t, dim=1
        )  # Average over ensemble: [B, C, spatial]

        # Term 2: 0.5 * E[|X - X'|] for this timestep (ensemble spread)
        if not deterministic:
            # Create pairwise differences for this timestep only
            pred_expanded_1 = pred_t.unsqueeze(2)  # [B, N, 1, C, spatial]
            pred_expanded_2 = pred_t.unsqueeze(1)  # [B, 1, N, C, spatial]

            ensemble_diff_t = torch.abs(
                pred_expanded_1 - pred_expanded_2
            )  # [B, N, N, C, spatial]
            term2_t = (
                0.5 * 1 / (N * (N - 1)) * torch.sum(ensemble_diff_t, dim=(1, 2))
            )  # [B, C, spatial]

            # Clear intermediate tensors to free memory
            del pred_expanded_1, pred_expanded_2, ensemble_diff_t
        else:
            term2_t = torch.zeros_like(term1_t)

        # CRPS for this timestep: [B, C, spatial]
        crps_t = term1_t - term2_t
        # Average over spatial dimensions: [B, C]
        crps_t = crps_t.mean(dim=tuple(range(2, crps_t.ndim)))

        # Store this timestep's CRPS
        crps_per_timestep.append(crps_t)

    # Stack all timesteps to get [B, L, C]
    crps = torch.stack(crps_per_timestep, dim=1)  # [B, L, C]

    return crps


def compute_frequency_crps_loss(predictions, target, metadata):
    """
    Compute CRPS loss in frequency domain using complex FFT representations.

    Args:
        predictions: [B, N_ensemble, L, H, [W], [D], C] - Ensemble predictions
        target: [B, N_ensemble, L, H, [W], [D], C] - Ground truth
        metadata: Metadata for the dataset

    Returns:
        CRPS loss [B, L, C] averaged over frequency components
    """
    deterministic = False

    # Verify input shapes
    assert predictions.shape == target.shape
    assert predictions.ndim >= 5

    num_spatial_dims = len(metadata.spatial_resolution)

    # Handle deterministic predictions
    if len(predictions.shape) != 4 + num_spatial_dims:
        predictions = predictions.unsqueeze(1)
        target = target.unsqueeze(1)
        deterministic = True

    assert num_spatial_dims == 2 or (
        num_spatial_dims == 3 and predictions.shape[5] == 1
    ), "Frequency CRPS currently only supports 2D spatial data"

    predictions = predictions[..., 0, :] if num_spatial_dims == 3 else predictions
    target = target[..., 0, :] if num_spatial_dims == 3 else target
    B, N, L, H, W, C = predictions.shape

    # Rearrange to [B, N, L, C, H, W]
    predictions_channels = rearrange(predictions, "B N L H W C -> B N L C H W")
    target_channels = rearrange(target, "B N L H W C -> B N L C H W")

    # Compute FFT along spatial dimensions
    pred_fft = torch.fft.rfft2(predictions_channels, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.rfft2(target_channels, dim=(-2, -1), norm="ortho")

    # Flatten frequency dimensions: [B, N, L, C, freq_flat]
    pred_fft_flat = rearrange(pred_fft, "B N L C ... -> B N L C (...)")
    target_fft_flat = rearrange(target_fft, "B N L C ... -> B N L C (...)")

    # CRPS calculation using complex arithmetic
    # Term 1: E[|X - Y|] where |.| is complex modulus
    diff = pred_fft_flat - target_fft_flat
    term1 = torch.mean(torch.abs(diff), dim=1)  # [B, L, C, freq]

    if deterministic or predictions.shape[1] == 1:
        return term1.mean(dim=3)  # [B, L, C]

    # Term 2: 0.5 * E[|X - X'|]
    pred_expanded_1 = pred_fft_flat.unsqueeze(2)  # [B, N, 1, L, C, freq]
    pred_expanded_2 = pred_fft_flat.unsqueeze(1)  # [B, 1, N, L, C, freq]
    ensemble_diff = torch.abs(pred_expanded_1 - pred_expanded_2)

    term2 = (
        0.5 * 1 / (N * (N - 1)) * torch.sum(ensemble_diff, dim=(1, 2))
    )  # [B, L, C, freq]

    # Final CRPS
    crps_freq = term1 - term2
    crps_freq = crps_freq.mean(dim=3)  # Average over frequencies: [B, L, C]

    return crps_freq


def compute_frequency_crps_loss_mem_efficient(predictions, target, metadata):
    """
    Compute CRPS loss in frequency domain - memory efficient version that processes one timestep at a time.

    Args:
        predictions: [B, N_ensemble, L, H, [W], [D], C] - Ensemble predictions
        target: [B, N_ensemble, L, H, [W], [D], C] - Ground truth
        metadata: Metadata for the dataset

    Returns:
        CRPS loss [B, L, C] averaged over frequency components
    """
    deterministic = False

    # Verify input shapes
    assert predictions.shape == target.shape
    assert predictions.ndim >= 5

    num_spatial_dims = len(metadata.spatial_resolution)
    # Handle deterministic predictions
    if len(predictions.shape) != 4 + num_spatial_dims:
        predictions = predictions.unsqueeze(1)
        target = target.unsqueeze(1)
        deterministic = True
    elif predictions.shape[1] == 1:
        deterministic = True

    assert num_spatial_dims == 2 or (
        num_spatial_dims == 3 and predictions.shape[5] == 1
    ), "Frequency CRPS currently only supports 2D spatial data"

    predictions = predictions[..., 0, :] if num_spatial_dims == 3 else predictions
    target = target[..., 0, :] if num_spatial_dims == 3 else target
    B, N, L, H, W, C = predictions.shape

    # Rearrange to [B, N, L, C, H, W]
    predictions_channels = rearrange(predictions, "B N L H W C -> B N L C H W")
    target_channels = rearrange(target, "B N L H W C -> B N L C H W")

    # Initialize list to store per-timestep CRPS values
    crps_per_timestep = []

    # Process each timestep separately to save memory
    for t in range(L):
        # Extract current timestep: [B, N, C, H, W]
        pred_t = predictions_channels[:, :, t, :, :, :]
        target_t = target_channels[:, :, t, :, :, :]

        # Compute FFT for this timestep
        pred_fft_t = torch.fft.rfft2(
            pred_t, dim=(-2, -1), norm="ortho"
        )  # [B, N, C, H, W//2+1]
        target_fft_t = torch.fft.rfft2(target_t, dim=(-2, -1), norm="ortho")

        # Flatten frequency dimensions: [B, N, C, freq_flat]
        pred_fft_flat_t = rearrange(pred_fft_t, "B N C ... -> B N C (...)")
        target_fft_flat_t = rearrange(target_fft_t, "B N C ... -> B N C (...)")

        # Term 1: E[|X - Y|] for this timestep
        diff_t = pred_fft_flat_t - target_fft_flat_t
        term1_t = torch.mean(torch.abs(diff_t), dim=1)  # [B, C, freq]

        # Term 2: 0.5 * E[|X - X'|] for this timestep
        if not deterministic:
            pred_expanded_1 = pred_fft_flat_t.unsqueeze(2)  # [B, N, 1, C, freq]
            pred_expanded_2 = pred_fft_flat_t.unsqueeze(1)  # [B, 1, N, C, freq]

            ensemble_diff_t = torch.abs(
                pred_expanded_1 - pred_expanded_2
            )  # [B, N, N, C, freq]
            term2_t = (
                0.5 * 1 / (N * (N - 1)) * torch.sum(ensemble_diff_t, dim=(1, 2))
            )  # [B, C, freq]

            # Clear intermediate tensors to free memory
            del pred_expanded_1, pred_expanded_2, ensemble_diff_t
        else:
            term2_t = torch.zeros_like(term1_t)

        # CRPS for this timestep: [B, C, freq]
        crps_t = term1_t - term2_t
        # Average over frequency dimensions: [B, C]
        crps_t = crps_t.mean(dim=2)

        # Store this timestep's CRPS
        crps_per_timestep.append(crps_t)

        # Clear FFT tensors to free memory
        del pred_fft_t, target_fft_t, pred_fft_flat_t, target_fft_flat_t

    # Stack all timesteps to get [B, L, C]
    crps_freq = torch.stack(crps_per_timestep, dim=1)

    return crps_freq
    return crps_freq
