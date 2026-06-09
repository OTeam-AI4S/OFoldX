r"""
Shared atom-pair encoding arithmetic.
"""

from __future__ import annotations

import torch
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
    prefix_ndim: int | None = None,
) -> Tensor:
    r"""
    Windowed atom-pair arithmetic shared by dense and local atom encoders.

    ``module`` supplies the six bias-free projections by their standard attribute names
    (``distance_vector_projection``, ``inverse_distance_projection``, ``distance_mask_projection``,
    ``left_atom_projection``, ``right_atom_projection``, ``feed_forward``). ``distance_vectors`` is the canonical
    displacement ``d = query - key`` used by AF3/Protenix atom attention. The masks are
    caller-defined so dense and local encoders can share the same arithmetic. ``prefix_ndim`` is the number of
    leading batch/sample axes before the pair/window axes; it is used only when ``pair_bias`` carries extra prefix
    axes.

    Args:
        module: Module supplying the expected atom-pair projection attributes.
        distance_vectors: ``[..., N_query, N_key, 3]`` canonical query-minus-key displacement vectors.
        query_single: ``[..., N_query, C]`` query atom features.
        key_single: ``[..., N_key, C]`` key atom features.
        distance_term_mask: Scalar or tensor mask for distance-vector terms.
        inverse_term_mask: Scalar or tensor mask for inverse-distance terms.
        valid_mask: ``[..., N_query, N_key, 1]`` valid pair mask.
        pair_bias: Optional precomputed pair bias to add.
        prefix_ndim: Optional number of leading prefix axes before pair/window axes.

    Returns:
        ``[..., N_query, N_key, C]`` encoded atom-pair features (same prefix and pair axes as
        ``distance_vectors``, with ``C`` set by the module's projection width).

    Raises:
        ValueError: If ``pair_bias`` cannot be broadcast onto the encoded pair tensor.

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
        ... )
        >>> out.shape
        torch.Size([1, 3, 3, 2])
    """

    inverse_distance = 1 / (1 + (distance_vectors * distance_vectors).sum(dim=-1, keepdim=True))
    pair = module.distance_vector_projection(distance_vectors) * distance_term_mask
    pair = pair + module.inverse_distance_projection(inverse_distance) * inverse_term_mask
    pair = pair + module.distance_mask_projection(valid_mask)
    if pair_bias is not None:
        pair = _match_prefix(pair, pair_bias, prefix_ndim=prefix_ndim) + pair_bias
    pair = pair + module.left_atom_projection(query_single) + module.right_atom_projection(key_single)
    return pair + module.feed_forward(pair)


def _match_prefix(source: Tensor, target: Tensor, *, prefix_ndim: int | None) -> Tensor:
    r"""
    Reshape ``source`` so ``target``'s extra leading prefix axes broadcast against it.

    Args:
        source: The encoded pair tensor to align.
        target: The ``pair_bias`` tensor whose (possibly higher) rank drives the alignment.
        prefix_ndim: Number of leading prefix axes on ``source``; inferred from the matching
            suffix axes when ``None``.

    Returns:
        ``source`` reshaped with singleton axes inserted so it broadcasts to ``target``'s shape.

    Raises:
        ValueError: If ``target`` has lower rank than ``source``, if ``prefix_ndim`` is out of
            range, or if the shapes are otherwise not broadcast-compatible.
    """
    missing_prefix_dims = target.dim() - source.dim()
    if missing_prefix_dims < 0:
        raise ValueError(f"pair_bias rank {target.dim()} cannot be smaller than pair rank {source.dim()}")
    if missing_prefix_dims == 0:
        return source
    if prefix_ndim is None:
        prefix_ndim = source.dim() - _matching_suffix_dims(source, target)
    if prefix_ndim < 0 or prefix_ndim > source.dim():
        raise ValueError(f"prefix_ndim={prefix_ndim} is invalid for pair rank {source.dim()}")
    reshaped = source.reshape(
        *source.shape[:prefix_ndim],
        *((1,) * missing_prefix_dims),
        *source.shape[prefix_ndim:],
    )
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(reshaped.shape), tuple(target.shape))
    except RuntimeError as error:
        raise ValueError(
            f"pair_bias shape {tuple(target.shape)} is incompatible with pair shape {tuple(source.shape)}"
        ) from error
    if broadcast_shape != tuple(target.shape):
        raise ValueError(f"pair_bias shape {tuple(target.shape)} cannot receive pair shape {tuple(source.shape)}")
    return reshaped


def _matching_suffix_dims(source: Tensor, target: Tensor) -> int:
    r"""
    Count the trailing axes ``source`` and ``target`` share, used to infer the prefix split.

    Args:
        source: The encoded pair tensor.
        target: The ``pair_bias`` tensor being aligned onto ``source``.

    Returns:
        The number of matching trailing axes (at least one).

    Raises:
        ValueError: If no trailing axis matches, so no valid prefix split exists.
    """
    suffix_dims = 0
    for offset in range(1, min(source.dim(), target.dim()) + 1):
        if source.shape[-offset] != target.shape[-offset]:
            break
        suffix_dims = offset
    if suffix_dims == 0:
        raise ValueError(f"pair_bias shape {tuple(target.shape)} is incompatible with pair shape {tuple(source.shape)}")
    return suffix_dims
