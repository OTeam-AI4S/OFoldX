r"""
Distogram head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..backend import resolve_config_operator_backend
from ..backend.cross_entropy import cross_entropy as dispatch_cross_entropy
from .frames import gather_atoms_by_index


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
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_bins: int = 64,
        symmetrize_before_projection: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(pair_size, num_bins, bias=bias)
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
            return self.linear(pair + pair.transpose(-2, -3))
        logits = self.linear(pair)
        return logits + logits.transpose(-2, -3)


class DistogramLoss(nn.Module):
    r"""
    Cross-entropy loss for distogram logits.

    Args:
        min_distance: Lower edge used for distance binning.
        max_distance: Upper edge used for distance binning.
        operator_backend: Which operator backend to run; see [`backend`][ofoldx.modules.backend].
    """

    def __init__(
        self,
        min_distance: float = 2.3125,
        max_distance: float = 21.6875,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.operator_backend = resolve_config_operator_backend(
            operator_backend, supported=("oops", "eager"), public_default="eager"
        )

    def forward(
        self,
        logits: Tensor,
        target_positions: Tensor,
        target_mask: Tensor,
        token_to_rep_atom: Tensor,
    ) -> Tensor:
        r"""
        Return masked pairwise distance-bin cross entropy.

        Each token's representative-atom position is gathered by ``token_to_rep_atom``; the pairwise
        distances between those atoms are binned and compared against ``logits`` with a cross entropy
        masked to the pairs whose both tokens have a resolved target.

        Args:
            logits: `[..., N_token, N_token, num_bins]` distogram logits.
            target_positions: `[..., N_atom, 3]` target atom coords.
            target_mask: `[..., N_atom]` valid-target mask.
            token_to_rep_atom: `[..., N_token]` representative atom indices.

        Returns:
            Scalar masked cross-entropy loss.

        Examples:
            >>> import torch
            >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
            <torch._C.Generator object at ...>
            >>> loss = DistogramLoss(operator_backend="eager")
            >>> logits = torch.randn(1, 2, 2, 64)
            >>> target_positions = torch.randn(1, 4, 3)
            >>> target_mask = torch.ones(1, 4, dtype=torch.bool)
            >>> token_to_rep_atom = torch.tensor([[0, 2]])
            >>> out = loss(logits, target_positions, target_mask, token_to_rep_atom)
            >>> out.shape
            torch.Size([])
        """

        target_positions = target_positions.to(device=logits.device, dtype=logits.dtype)
        target_mask = target_mask.to(device=logits.device).bool()
        # Gather each token's representative-atom position/mask by the per-token global atom index (a
        # per-element gather replacing a boolean mask over the atom axis). ``target_positions`` is
        # ``[..., N_atom, 3]`` with the same leading dims as ``token_to_rep_atom`` (``[..., N_token]``).
        rep_index = token_to_rep_atom.to(device=logits.device, dtype=torch.long)
        representative_positions = gather_atoms_by_index(target_positions, rep_index)
        representative_mask = torch.gather(target_mask, -1, rep_index)

        # Pairwise representative-atom distances.
        rep = representative_positions.float()
        distances = torch.cdist(rep, rep).to(logits.dtype)
        labels = distance_bin_indices(
            distances,
            num_bins=logits.shape[-1],
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )
        pair_mask = representative_mask.unsqueeze(-1) & representative_mask.unsqueeze(-2)
        return masked_cross_entropy(logits, labels, pair_mask, operator_backend=self.operator_backend)


def distance_bin_indices(distances: Tensor, *, num_bins: int, min_distance: float, max_distance: float) -> Tensor:
    r"""
    Return integer bin labels for distances.

    The binning uses ``num_bins - 1`` evenly spaced boundaries between ``min_distance`` and
    ``max_distance``, so distances below the first edge fall in bin ``0`` and distances above the
    last edge fall in bin ``num_bins - 1``.

    Args:
        distances: Pairwise distances of any shape.
        num_bins: Number of output bins.
        min_distance: Lower binning range edge.
        max_distance: Upper binning range edge.

    Returns:
        Integer labels in ``[0, num_bins)`` with the same shape as ``distances``.

    Examples:
        >>> import torch
        >>> distances = torch.tensor([[0.0, 5.0, 25.0]])
        >>> distance_bin_indices(distances, num_bins=4, min_distance=2.0, max_distance=20.0).tolist()
        [[0, 1, 3]]
    """

    boundaries = torch.linspace(
        min_distance,
        max_distance,
        steps=num_bins - 1,
        device=distances.device,
        dtype=distances.dtype,
    )
    return torch.bucketize(distances, boundaries)


def masked_cross_entropy(logits: Tensor, labels: Tensor, mask: Tensor, *, operator_backend: str = "auto") -> Tensor:
    r"""
    Return cross entropy averaged over true mask entries.

    The class axis stays last so OOps can use its integer-label fused path when the current shape is in-domain.

    Args:
        logits: Class logits with the class axis last.
        labels: Integer class labels matching ``logits.shape[:-1]``.
        mask: Boolean or numeric mask matching ``labels``; entries are used as averaging weights.
        operator_backend: Which operator backend to run; see [`backend`][ofoldx.modules.backend].

    Returns:
        Scalar cross entropy averaged over true mask entries.

    Examples:
        >>> import torch
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> logits = torch.randn(2, 3)
        >>> labels = torch.tensor([0, 1])
        >>> mask = torch.tensor([1.0, 0.0])
        >>> round(float(masked_cross_entropy(logits, labels, mask, operator_backend="eager")), 4)
        0.1689
    """

    labels = labels.to(device=logits.device, dtype=torch.long)
    mask_flat = mask.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    losses = dispatch_cross_entropy(logits, labels, operator_backend=operator_backend).reshape(-1)
    return (losses * mask_flat).sum() / mask_flat.sum().clamp_min(1)
