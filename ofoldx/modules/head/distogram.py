r"""
Distogram head.
"""

from __future__ import annotations

from torch import Tensor, nn


class DistogramHead(nn.Module):
    r"""
    Project pair representations to distance-bin logits.

    The pair representation is symmetrised across the two token axes. ``symmetrize_before_projection``
    selects where: ``False`` projects then adds the transpose, while ``True`` symmetrises the pair first and
    projects once. The two choices place the linear bias differently.

    Args:
        pair_size: Pair representation width.
        num_bins: Number of distance bins.
        symmetrize_before_projection: Whether to symmetrize pair features before the output projection.
        bias: Whether the output projection carries a learnable bias.
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_bins: int = 64,
        symmetrize_before_projection: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(pair_size, num_bins, bias=bias)
        self.symmetrize_before_projection = symmetrize_before_projection

    def forward(self, pair: Tensor) -> Tensor:
        r"""
        Return pairwise distance-bin logits.

        Args:
            pair: `[..., N_token, N_token, pair_size]` pair representation.

        Returns:
            `[..., N_token, N_token, num_bins]` symmetric distogram logits.

        Examples:
            >>> import torch
            >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
            <torch._C.Generator object at ...>
            >>> head = DistogramHead(pair_size=8, num_bins=4)
            >>> pair = torch.randn(1, 2, 2, 8)
            >>> head(pair).shape
            torch.Size([1, 2, 2, 4])
        """

        if self.symmetrize_before_projection:
            return self.projection(pair + pair.transpose(-2, -3))
        logits = self.projection(pair)
        return logits + logits.transpose(-2, -3)
