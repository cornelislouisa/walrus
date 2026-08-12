import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer that can handle:
      - x: [T, B, C, ...]
      - x: [T*B, C, ...]
    Conditioning cond should be [T, B, cond_dim, ...].

    Args:
        in_features: Number of input features
        cond_dim: Dimension of conditioning input
        use_gamma: If True, applies gamma*x + beta. If False, applies x + beta. Default: True
    """

    def __init__(
        self,
        in_features,
        cond_dim,
        use_gamma=True,
        pre_norm=False,
        post_norm=False,
        div_2=False,
    ):
        super().__init__()
        self.use_gamma = use_gamma
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.div_2 = div_2
        out_features = 2 * in_features if use_gamma else in_features
        self.fc = nn.Linear(cond_dim, out_features)
        if self.pre_norm:
            self.norm = nn.RMSNorm(in_features)
        if self.post_norm:
            self.post_norm_layer = nn.RMSNorm(in_features)

    def forward(self, x, cond):
        # Determine if x is flattened (T*B) or unflattened (T, B)
        if x.dim() < 3:
            raise ValueError(f"x must have at least 3 dims, got {x.shape}")

        # Flatten T and B if needed
        flattened = False
        if x.shape[0] != cond.shape[0] or cond.shape[1] == 1:
            # assume x is [T*B, C, ...]
            T, B = cond.shape[:2]
            x = x.view(T, B, *x.shape[1:])
            flattened = True

        # Apply pre-normalization if enabled
        if self.pre_norm:
            # x is [T, B, C, H, W, D] - move C to last dimension
            # [T, B, C, ...] -> [T, B, ..., C]
            x = x.movedim(2, -1)
            x = self.norm(x)
            # [T, B, ..., C] -> [T, B, C, ...]
            x = x.movedim(-1, 2)

        # Apply FiLM per timestep
        T, B = cond.shape[:2]
        # Move the 3rd dimension (cond_dim) to the last position if needed
        if cond.dim() > 3:
            # [T, B, C, ...] -> [T, B, ..., C]
            cond = cond.movedim(2, -1)  # [T, B, ..., C]

        if self.use_gamma:
            gamma, beta = self.fc(cond).chunk(2, dim=-1)  # [T, B, ..., 2*C]
            if cond.dim() > 3:
                # [T, B, ..., C] -> [T, B, C, ...]
                gamma = gamma.movedim(-1, 2)
                beta = beta.movedim(-1, 2)
            else:
                # reshape for broadcasting
                gamma = gamma.view(T, B, -1, *([1] * (x.dim() - 3)))
                beta = beta.view(T, B, -1, *([1] * (x.dim() - 3)))
            x = gamma * x + beta
        else:
            beta = self.fc(cond)  # [T, B, ..., C]
            if cond.dim() > 3:
                # [T, B, ..., C] -> [T, B, C, ...]
                beta = beta.movedim(-1, 2)
            else:
                # reshape for broadcasting
                beta = beta.view(T, B, -1, *([1] * (x.dim() - 3)))
            if self.div_2:
                x = (x + beta) / 2
            else:
                x = x + beta

        if self.post_norm:
            # x is [T, B, C, H, W, D] - move C to last dimension
            # [T, B, C, ...] -> [T, B, ..., C]
            x = x.movedim(2, -1)
            x = self.post_norm_layer(x)
            # [T, B, ..., C] -> [T, B, C, ...]
            x = x.movedim(-1, 2)

        # Flatten back if needed
        if flattened:
            x = x.view(T * B, *x.shape[2:])
        return x
