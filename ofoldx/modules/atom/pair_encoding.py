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
    Encode atom pairs from displacement and single-atom features.

    ``module`` supplies ``distance_vector_projection``,
    ``inverse_distance_projection``, ``distance_mask_projection``,
    ``left_atom_projection``, ``right_atom_projection``, and the
    ``feed_forward(pair)`` residual-update method.

    Args:
        module: Owner of the atom-pair projections.
        distance_vectors: Query-minus-key displacements with shape
            ``(..., num_query_atoms, num_key_atoms, 3)``.
        query_single: Query features with shape
            ``(..., num_query_atoms, atom_hidden_size)``.
        key_single: Key features with shape
            ``(..., num_key_atoms, atom_hidden_size)``.
        distance_term_mask: Mask broadcast over projected displacement features.
        inverse_term_mask: Mask broadcast over projected inverse-distance features.
        valid_mask: Pair-validity values broadcastable to
            ``(..., num_query_atoms, num_key_atoms, 1)``.
        pair_bias: Optional additive pair representation.
        prefix_ndim: Number of leading axes that precede pair or window axes.

    Returns:
        Encoded pair features with the prefix and pair axes of ``distance_vectors``.
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
