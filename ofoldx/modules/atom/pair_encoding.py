r"""Shared atom-pair encoding module."""

from __future__ import annotations

from torch import Tensor, nn


class AtomPairEncoding(nn.Module):
    r"""Encode atom pairs from geometry and per-atom states.

    This module owns the projections and residual feed-forward equation shared by
    the dense and packed atom encoders. Callers remain responsible only for
    constructing their geometry layout.

    Args:
        atom_hidden_size: Width of the query and key atom states.
        atom_pair_hidden_size: Width of the encoded atom-pair states.
    """

    def __init__(self, atom_hidden_size: int = 128, atom_pair_hidden_size: int = 16) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.left_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.right_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.feed_forward_input_projection = nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False)
        self.feed_forward_hidden_projection = nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False)
        self.feed_forward_output_projection = nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False)

    def forward(
        self,
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
        r"""Return encoded pair states with the pair axes of ``distance_vectors``."""

        inverse_distance = 1 / (1 + (distance_vectors * distance_vectors).sum(dim=-1, keepdim=True))
        pair = self.distance_vector_projection(distance_vectors) * distance_term_mask
        pair = pair + self.inverse_distance_projection(inverse_distance) * inverse_term_mask
        pair = pair + self.distance_mask_projection(valid_mask)
        pair = pair + self.left_atom_projection(query_single) + self.right_atom_projection(key_single)
        if pair_bias is not None:
            missing_prefix_dims = pair_bias.dim() - pair.dim()
            pair = pair.reshape(
                *pair.shape[:prefix_ndim],
                *((1,) * missing_prefix_dims),
                *pair.shape[prefix_ndim:],
            )
            pair = pair + pair_bias
        return pair + self._feed_forward(pair)

    def _feed_forward(self, pair: Tensor) -> Tensor:
        pair = self.feed_forward_input_projection(self.activation(pair))
        pair = self.feed_forward_hidden_projection(self.activation(pair))
        return self.feed_forward_output_projection(self.activation(pair))
