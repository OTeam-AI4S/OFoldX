r"""
Shared atom-pair encoding arithmetic.
"""

from __future__ import annotations

from torch import Tensor, nn


def encode_atom_pair(
    module: nn.Module,
    distance_vectors: Tensor,
    query_single: Tensor,
    key_single: Tensor,
    *,
    distance_term_mask: Tensor | float,
    inverse_term_mask: Tensor | float,
    valid_mask: Tensor,
    pair_bias: Tensor | None = None,
    prefix_ndim: int,
) -> Tensor:
    r"""
    Windowed atom-pair arithmetic shared by dense and local atom encoders.

    ``module`` supplies the six bias-free projections by their standard attribute names
    (``distance_vector_projection``, ``inverse_distance_projection``, ``distance_mask_projection``,
    ``left_atom_projection``, ``right_atom_projection``, ``feed_forward``). ``distance_vectors`` is the canonical
    displacement ``d = query - key`` used by AF3/Protenix atom attention. The masks are
    caller-defined so dense and local encoders can share the same arithmetic. ``prefix_ndim`` is the number of
    leading batch/sample axes before the pair/window axes.

    Args:
        module: Module supplying the expected atom-pair projection attributes.
        distance_vectors: ``[..., N_query, N_key, 3]`` canonical query-minus-key displacement vectors.
        query_single: ``[..., N_query, C]`` query atom features.
        key_single: ``[..., N_key, C]`` key atom features.
        distance_term_mask: Scalar or tensor mask for distance-vector terms.
        inverse_term_mask: Scalar or tensor mask for inverse-distance terms.
        valid_mask: ``[..., N_query, N_key, 1]`` valid pair mask.
        pair_bias: Optional precomputed pair bias to add.
        prefix_ndim: Number of leading prefix axes before pair/window axes.

    Returns:
        ``[..., N_query, N_key, C]`` encoded atom-pair features (same prefix and pair axes as
        ``distance_vectors``, with ``C`` set by the module's projection width).

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from ofoldx.modules.atom.pair_encoding import encode_atom_pair
        >>> class Tiny(nn.Module):
        ...     def __init__(self, channels=2):
        ...         super().__init__()
        ...         self.distance_vector_projection = nn.Linear(3, channels)
        ...         self.inverse_distance_projection = nn.Linear(1, channels)
        ...         self.distance_mask_projection = nn.Linear(1, channels)
        ...         self.left_atom_projection = nn.Linear(channels, channels)
        ...         self.right_atom_projection = nn.Linear(channels, channels)
        ...         self.feed_forward = nn.Linear(channels, channels)
        >>> _ = torch.manual_seed(0)
        >>> module = Tiny(channels=2)
        >>> distance_vectors = torch.zeros(1, 3, 3, 3)
        >>> query_single = torch.zeros(1, 3, 2)
        >>> key_single = torch.zeros(1, 3, 2)
        >>> valid_mask = torch.ones(1, 3, 3, 1)
        >>> out = encode_atom_pair(
        ...     module,
        ...     distance_vectors,
        ...     query_single,
        ...     key_single,
        ...     distance_term_mask=1.0,
        ...     inverse_term_mask=1.0,
        ...     valid_mask=valid_mask,
        ...     prefix_ndim=1,
        ... )
        >>> out.shape
        torch.Size([1, 3, 3, 2])
    """

    inverse_distance = 1 / (1 + (distance_vectors * distance_vectors).sum(dim=-1, keepdim=True))
    pair = module.distance_vector_projection(distance_vectors) * distance_term_mask
    pair = pair + module.inverse_distance_projection(inverse_distance) * inverse_term_mask
    pair = pair + module.distance_mask_projection(valid_mask)
    # Add the per-atom single terms before prefix-matching so the fully-formed pair (which shares the pair axes
    # of distance_vectors, no sample axis) is reshaped once to align with a possibly sample-prefixed pair_bias.
    pair = pair + module.left_atom_projection(query_single) + module.right_atom_projection(key_single)
    if pair_bias is not None:
        missing_prefix_dims = pair_bias.dim() - pair.dim()
        pair = pair.reshape(
            *pair.shape[:prefix_ndim],
            *((1,) * missing_prefix_dims),
            *pair.shape[prefix_ndim:],
        )
        pair = pair + pair_bias
    return pair + module.feed_forward(pair)
