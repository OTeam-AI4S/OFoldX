r"""
Fourier feature embeddings.

Random Fourier features lift a scalar (e.g. a diffusion timestep) into a higher-dimensional
periodic embedding that a network can use for diffusion timesteps and other scalar conditions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenFourierEmbedding(nn.Module):
    r"""
    Frozen random Fourier features ``cos(2*pi * projection(x))``.

    A single ``1 -> dim`` linear layer with standard-normal weight and bias is drawn once, frozen
    (``requires_grad_(False)``), and reused as a fixed random projection; taking the cosine of the
    scaled projection yields ``dim`` periodic features per scalar input.

    Args:
        dim: Number of Fourier feature channels.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> embedding = FrozenFourierEmbedding(4)
        >>> embedding(torch.tensor([0.0, 1.0])).shape
        torch.Size([2, 4])
        >>> sum(p.numel() for p in embedding.parameters() if p.requires_grad)
        0
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, dim)
        self.reset_parameters()
        self.projection.requires_grad_(False)

    def reset_parameters(self) -> None:
        r"""
        Re-draw the frozen projection weight and bias from a standard normal distribution.

        Examples:
            >>> import torch
            >>> embedding = FrozenFourierEmbedding(4)
            >>> _ = torch.manual_seed(0)
            >>> embedding.reset_parameters()
            >>> embedding.projection.weight.shape
            torch.Size([4, 1])
        """
        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.projection.bias, mean=0.0, std=1.0)

    def forward(self, times: Tensor) -> Tensor:
        r"""
        Return Fourier features for scalar times.

        Args:
            times: ``[...]`` scalar values (e.g. diffusion timesteps) to embed; a trailing
                channel axis is added internally before the projection.

        Returns:
            ``[..., dim]`` Fourier features ``cos(2*pi * projection(times))``.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> embedding = FrozenFourierEmbedding(4)
            >>> features = embedding(torch.tensor([0.0, 1.0, 2.0]))
            >>> features.shape
            torch.Size([3, 4])
            >>> bool(features.abs().le(1.0).all())
            True
        """

        projected = self.projection(times.unsqueeze(-1))
        return torch.cos(2 * torch.pi * projected)
