r"""
Shared diffusion conditioning components.

Provides the reusable [`PairwiseConditioning`][ofoldx.modules.diffusion.conditioning.PairwiseConditioning] module
and lower-level functional helpers for combining trunk representations with noise-level and relative-position
features.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from ..backend import LayerNorm
from ..transition import Transition


class PairwiseConditioning(nn.Module):
    r"""
    Condition a pair representation with relative-position features.

    Args:
        pair_size: Width of the pair representation.
        relative_position_size: Width of the relative-position features concatenated with the pair.
        num_transitions: Number of transition blocks refining the conditioned pair.
        expansion: Inner-width expansion factor of each transition block.
        operator_backend: Public backend request forwarded to every transition.
    """

    def __init__(
        self,
        pair_size: int = 128,
        relative_position_size: int = 128,
        num_transitions: int = 2,
        expansion: int = 2,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.init_projection = nn.Sequential(
            LayerNorm(pair_size + relative_position_size, operator_backend=operator_backend),
            nn.Linear(pair_size + relative_position_size, pair_size, bias=False),
        )
        self.transitions = nn.ModuleList(
            Transition(pair_size, expansion=expansion, operator_backend=operator_backend)
            for _ in range(num_transitions)
        )

    def forward(self, pair_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
        r"""
        Project concatenated pair features and refine them with residual transitions.

        Args:
            pair_trunk: Trunk pair representation ``[..., N_token, N_token, pair_size]``.
            relative_position_encoding: Relative-position features
                ``[..., N_token, N_token, relative_position_size]``.

        Returns:
            Conditioned pair representation ``[..., N_token, N_token, pair_size]``.
        """
        return condition_pair(
            pair_trunk,
            relative_position_encoding,
            norm=self.init_projection[0],
            projection=self.init_projection[1],
            transitions=self.transitions,
        )


def condition_single(
    single_a: Tensor,
    single_b: Tensor,
    scaled_noise: Tensor,
    *,
    norm: nn.Module,
    projection: nn.Module,
    noise_fourier: nn.Module,
    noise_norm: nn.Module,
    noise_projection: nn.Module,
    transitions: Iterable[nn.Module],
    sample_axis: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""
    Noise-condition a single representation.

    Concatenate the two single inputs, norm + project, add projected-normalized Fourier noise features, then apply
    ``transitions``. ``scaled_noise`` is already caller-scaled. ``sample_axis`` inserts an explicit diffusion-sample
    axis for callers that keep samples separate from the batch axis. Returns ``(single, normed_noise_features)``.

    Args:
        single_a: First ``[..., N_token, C]`` single representation.
        single_b: Second ``[..., N_token, C]`` single representation.
        scaled_noise: Noise-level tensor after caller-side scaling.
        norm: Normalization module applied to the concatenated single representation.
        projection: Projection module from concatenated single width to output single width.
        noise_fourier: Fourier/noise embedding module.
        noise_norm: Normalization module for noise features.
        noise_projection: Projection module from noise features to output single width.
        transitions: Transition modules applied residually to the conditioned single representation.
        sample_axis: If ``True``, insert a diffusion-sample axis before the token axis.

    Returns:
        A ``(single, normed_noise)`` tuple: the conditioned single representation of shape `[..., N_token, C]` (or
        `[..., N_sample, N_token, C]` when ``sample_axis`` is set) and the normalized noise features of shape
        `[..., C]`.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> _ = torch.manual_seed(0)
        >>> single_a = torch.zeros(1, 2, 4)
        >>> single_b = torch.zeros(1, 2, 4)
        >>> single, noise = condition_single(
        ...     single_a,
        ...     single_b,
        ...     torch.zeros(1, 1),
        ...     norm=nn.LayerNorm(8),
        ...     projection=nn.Linear(8, 4),
        ...     noise_fourier=nn.Linear(1, 4),
        ...     noise_norm=nn.LayerNorm(4),
        ...     noise_projection=nn.Linear(4, 4),
        ...     transitions=[],
        ... )
        >>> single.shape, noise.shape
        (torch.Size([1, 2, 4]), torch.Size([1, 4]))
    """
    single = projection(norm(torch.cat([single_a, single_b], dim=-1)))
    normed_noise = noise_norm(noise_fourier(scaled_noise).to(single.dtype))
    base = single.unsqueeze(-3) if sample_axis else single
    single = base + noise_projection(normed_noise).unsqueeze(-2)
    for transition in transitions:
        single = single + transition(single)
    return single, normed_noise


def condition_pair(
    pair_trunk: Tensor,
    relative_position_encoding: Tensor,
    *,
    norm: nn.Module,
    projection: nn.Module,
    transitions: Iterable[nn.Module],
) -> Tensor:
    r"""
    Relative-position-condition a pair representation.

    Concatenate the trunk pair with the (already embedded) relative-position encoding, norm + project, then apply
    the ``transitions``.

    Args:
        pair_trunk: ``[..., N_token, N_token, C_pair]`` trunk pair representation.
        relative_position_encoding: Embedded relative-position features with matching pair axes.
        norm: Normalization module for concatenated pair features.
        projection: Projection module to the output pair width.
        transitions: Transition modules applied residually.

    Returns:
        Conditioned pair representation of shape `[..., N_token, N_token, C_pair]`.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> _ = torch.manual_seed(0)
        >>> pair_trunk = torch.zeros(1, 2, 2, 4)
        >>> relative_position_encoding = torch.zeros(1, 2, 2, 4)
        >>> pair = condition_pair(
        ...     pair_trunk,
        ...     relative_position_encoding,
        ...     norm=nn.LayerNorm(8),
        ...     projection=nn.Linear(8, 4),
        ...     transitions=[],
        ... )
        >>> pair.shape
        torch.Size([1, 2, 2, 4])
    """
    pair = projection(norm(torch.cat([pair_trunk, relative_position_encoding], dim=-1)))
    for transition in transitions:
        pair = pair + transition(pair)
    return pair
