# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Atom-to-token encoders and token-to-atom decoders.
"""

from __future__ import annotations

from functools import partial
from math import prod
from typing import TypeAlias

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from ..backend import LayerNorm
from ..backend.normalization import layer_norm_projection
from ..feed_forward import FeedForward
from ..transformer import (
    AtomCrossAttentionTransformerEncoderBlock,
    ConditionedTransformerEncoder,
    ConditionedTransformerEncoderBlock,
)
from .coordinates import project_coordinates
from .nested import gather_packed_atom_windows, gather_packed_window_keys
from .transfer import AtomToTokenMean, TokenPairToLocalAtomPairBroadcast, TokenToAtomBroadcast
from .window import LocalAtomGeometry, LocalAtomWindowInfo, gather_key_windows

#: Block classes a local atom transformer can stack.
AtomTransformerBlock: TypeAlias = (
    type[ConditionedTransformerEncoderBlock] | type[AtomCrossAttentionTransformerEncoderBlock]
)


class AtomConditioner(nn.Module):
    r"""
    Build per-atom conditioning states from reference position, charge, and feature inputs.

    Projects each reference input to the atom hidden width and sums them, giving the shared per-atom
    conditioning consumed by the atom-pair encoder and atom self-attention. Formal charges are squashed
    with ``arcsinh`` before their projection.

    Args:
        atom_feature_size: Width of the raw reference atom-feature vector.
        atom_hidden_size: Width of the produced conditioning channel.
    """

    def __init__(self, atom_feature_size: int = 385, atom_hidden_size: int = 128) -> None:
        super().__init__()
        self.reference_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.reference_charge_projection = nn.Linear(1, atom_hidden_size, bias=False)
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=False)

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
    ) -> Tensor:
        r"""
        Sum projections of reference position, charge, and atom features into per-atom conditioning.

        The masked-out atoms have their conditioning zeroed rather than merely down-weighted, so absent
        atoms contribute nothing downstream.

        Args:
            atom_features: `[..., N_atom, atom_feature_size]` raw reference atom features.
            reference_positions: `[..., N_atom, 3]` reference coordinates.
            reference_charges: `[..., N_atom, 1]` reference formal charges.
            reference_mask: `[..., N_atom]` validity mask; absent atoms are zeroed.

        Returns:
            Per-atom conditioning of shape `[..., N_atom, atom_hidden_size]`.
        """

        absent = ~reference_mask.bool().unsqueeze(-1)
        reference_positions = project_coordinates(
            reference_positions, reference_mask.to(torch.bool), dtype=self.reference_position_projection.weight.dtype
        )
        reference_charges = reference_charges.masked_fill(absent, 0)
        atom_features = atom_features.masked_fill(absent, 0)
        conditioning = self.reference_position_projection(
            reference_positions.to(self.reference_position_projection.weight)
        )
        conditioning = conditioning + self.reference_charge_projection(
            torch.arcsinh(reference_charges).to(self.reference_charge_projection.weight)
        )
        conditioning = conditioning + self.atom_feature_projection(
            atom_features.to(self.atom_feature_projection.weight)
        )
        if isinstance(conditioning, NestedTensor):
            if not isinstance(reference_mask, NestedTensor):
                raise ValueError("packed atom conditioning requires a NestedTensor reference_mask")
            mask_values = reference_mask.concat.reshape(-1, 1).to(
                device=conditioning.device,
                dtype=conditioning.dtype,
            )
            return conditioning.packed_with_static_tail(conditioning.concat * mask_values)
        if isinstance(reference_mask, NestedTensor):
            raise ValueError("dense atom conditioning requires a dense reference_mask")
        return conditioning * reference_mask.reshape(*conditioning.shape[:-1], 1).to(
            device=conditioning.device,
            dtype=conditioning.dtype,
        )


class AtomPairEncoder(nn.Module):
    r"""
    Encode the local atom pairs from their reference geometry and the per-atom states.

    Each pair encodes its query-minus-key reference displacement, the reciprocal of one plus its squared
    length, and its validity (both atoms valid and in one reference space), together with the ReLU'd
    states of both atoms and an optional pair input, followed by a residual feed-forward. The pair
    states keep the packed batch topology of the atom states.

    Args:
        atom_hidden_size: Width of the per-atom states.
        atom_pair_hidden_size: Width of the encoded atom-pair states.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import AtomPairEncoder, LocalAtomWindowing
        >>> geometry = LocalAtomWindowing(4, 8)(
        ...     torch.randn(1, 10, 3), torch.zeros(1, 10, dtype=torch.long), torch.ones(1, 10, dtype=torch.bool)
        ... )
        >>> AtomPairEncoder(6, 5)(torch.randn(1, 10, 6), geometry).shape
        torch.Size([1, 3, 4, 8, 5])
    """

    def __init__(self, atom_hidden_size: int = 128, atom_pair_hidden_size: int = 16) -> None:
        super().__init__()
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.left_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.right_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.feed_forward = FeedForward(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            num_hidden_layers=2,
            bias=False,
            activation="relu",
        )

    def forward(
        self,
        atom_states: Tensor | NestedTensor,
        geometry: LocalAtomGeometry,
        pair: Tensor | NestedTensor | None = None,
    ) -> Tensor | NestedTensor:
        r"""
        Encode the local atom pairs of ``geometry``.

        Leading axes the states, geometry, and pair input do not share follow their shared prefix and
        broadcast.

        Args:
            atom_states: `[..., N_atom, atom_hidden_size]` per-atom states.
            geometry: The windowed reference geometry of the atom axis.
            pair: Optional `[..., N_window, num_query_atoms, num_key_atoms, atom_pair_hidden_size]` pair input.

        Returns:
            Local pair states `[..., N_window, num_query_atoms, num_key_atoms, atom_pair_hidden_size]`.
        """

        info = geometry.window_info
        query, key = info.gather(atom_states)
        distance_vectors = geometry.distance_vectors
        valid = geometry.distance_mask
        prefix_ndim = max(query.dim() - 3, distance_vectors.dim() - 4)
        distance_vectors = _insert_prefix_axes(distance_vectors, distance_vectors.dim() - 4, prefix_ndim)
        valid = _insert_prefix_axes(valid, valid.dim() - 4, prefix_ndim)
        query = _insert_prefix_axes(query, query.dim() - 3, prefix_ndim)
        key = _insert_prefix_axes(key, key.dim() - 3, prefix_ndim)

        inverse_distance = 1 / (1 + (distance_vectors * distance_vectors).sum(dim=-1, keepdim=True))
        state = self.distance_vector_projection(distance_vectors.to(self.distance_vector_projection.weight))
        state = state * valid.to(state)
        inverse = self.inverse_distance_projection(inverse_distance.to(self.inverse_distance_projection.weight))
        state = state + inverse * valid.to(inverse)
        state = state + self.distance_mask_projection(valid.to(self.distance_mask_projection.weight))
        state = (
            state
            + self.left_atom_projection(F.relu(query).unsqueeze(-2))
            + self.right_atom_projection(F.relu(key).unsqueeze(-3))
        )
        if pair is not None:
            pair_values = info.values(pair)
            pair_prefix_ndim = pair_values.dim() - 4
            target_ndim = max(prefix_ndim, pair_prefix_ndim)
            state = _insert_prefix_axes(state, prefix_ndim, target_ndim)
            state = state + _insert_prefix_axes(pair_values, pair_prefix_ndim, target_ndim)
        state = state + self.feed_forward(F.relu(state))
        return info.pack_like(atom_states, state)


class TrunkToAtomConditioning(nn.Module):
    r"""
    Condition per-atom states and local atom pairs on the trunk single and pair representations.

    Each atom adds the normalized, projected single state of its token; each local atom pair takes the
    normalized, projected pair state of its two atoms' tokens.

    Args:
        single_size: Width of the trunk single representation.
        pair_size: Width of the trunk pair representation.
        atom_hidden_size: Width of the per-atom states.
        atom_pair_hidden_size: Width of the local atom-pair states.
        bias: Whether the two layer normalizations have a learnable bias.
        backend: Backend selector forwarded to the normalizations and the token-to-atom transfers.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import LocalAtomWindowing, TrunkToAtomConditioning
        >>> window_info = LocalAtomWindowing(4, 8).window_info(torch.ones(1, 10, dtype=torch.bool))
        >>> conditioning, pair = TrunkToAtomConditioning(5, 3, 8, 4)(
        ...     torch.randn(1, 10, 8),
        ...     torch.randn(1, 5, 5),
        ...     torch.randn(1, 5, 5, 3),
        ...     torch.arange(10).div(2, rounding_mode="floor").unsqueeze(0),
        ...     window_info,
        ... )
        >>> conditioning.shape, pair.shape
        (torch.Size([1, 10, 8]), torch.Size([1, 3, 4, 8, 4]))
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        *,
        bias: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_norm = LayerNorm(single_size, bias=bias, backend=backend)
        self.single_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.pair_norm = LayerNorm(pair_size, bias=bias, backend=backend)
        self.pair_projection = nn.Linear(pair_size, atom_pair_hidden_size, bias=False)
        self.token_pair_to_local_atom_pair_broadcast = TokenPairToLocalAtomPairBroadcast(backend=backend)

    def forward(
        self,
        atom_states: Tensor | NestedTensor,
        single: Tensor | NestedTensor,
        pair: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        window_info: LocalAtomWindowInfo,
    ) -> tuple[Tensor | NestedTensor, Tensor | NestedTensor]:
        r"""
        Args:
            atom_states: Per-atom states `[..., N_atom, atom_hidden_size]`.
            single: Trunk single representation `[..., N_token, single_size]`.
            pair: Trunk pair representation `[..., N_token, N_token, pair_size]`.
            atom_to_token: Token index of each atom `[..., N_atom]`.
            window_info: The local window partition of the atom axis.

        Returns:
            atom_conditioning: Trunk-conditioned atom states `[..., N_atom, atom_hidden_size]`; a sample axis
                of ``single`` is inserted before the atom axis of ``atom_states``.
            local_pair: Local atom-pair input `[..., N_window, Q, K, atom_pair_hidden_size]`.
        """

        single = self.token_to_atom_broadcast(
            layer_norm_projection(single, norm=self.single_norm, projection=self.single_projection), atom_to_token
        )
        if single.dim() == atom_states.dim() + 1:
            atom_states = atom_states.unsqueeze(-3)
        atom_conditioning = atom_states + single
        pair = layer_norm_projection(pair, norm=self.pair_norm, projection=self.pair_projection)
        return atom_conditioning, self.token_pair_to_local_atom_pair_broadcast(pair, atom_to_token, window_info)


class LocalAtomAttention(nn.Module):
    r"""
    Run a conditioned Transformer encoder over local atom windows under a packed per-block pair bias.

    Flattens arbitrary leading (batch[, sample]) axes into one batch axis, splits the atom axis into the
    query windows of ``window_info``, attends every window to its key neighbourhood, and restores the
    atom axis. ``packed_bias`` stores every block's heads on its last axis.

    Args:
        num_blocks: Number of Transformer-encoder blocks.
        num_attention_heads: Attention heads per block.
        hidden_size: Width of the atom hidden state.
        conditioning_size: Width of the per-atom conditioning.
        backend: Backend selector forwarded to every block.
        block: Block class of every block.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import LocalAtomAttention, LocalAtomWindowing
        >>> from ofoldx.modules.transformer import ConditionedTransformerEncoderBlock
        >>> attention = LocalAtomAttention(2, 2, 8, 8, block=ConditionedTransformerEncoderBlock)
        >>> atom_mask = torch.ones(1, 10, dtype=torch.bool)
        >>> window_info = LocalAtomWindowing(4, 8).window_info(atom_mask)
        >>> hidden = torch.randn(1, 10, 8)
        >>> attention(hidden, hidden, torch.randn(1, 3, 4, 8, 4), atom_mask, window_info).shape
        torch.Size([1, 10, 8])
    """

    def __init__(
        self,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        hidden_size: int = 128,
        conditioning_size: int = 128,
        backend: str = "auto",
        *,
        block: AtomTransformerBlock,
    ) -> None:
        super().__init__()
        self.transformer = ConditionedTransformerEncoder(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            conditioning_size=conditioning_size,
            backend=backend,
            block=block,
        )

    def forward(
        self,
        atom_hidden: Tensor | NestedTensor,
        conditioning: Tensor | NestedTensor,
        packed_bias: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor,
        window_info: LocalAtomWindowInfo,
    ) -> Tensor | NestedTensor:
        r"""
        Apply local-window atom attention.

        Args:
            atom_hidden: `[..., N_atom, hidden_size]` atom query states.
            conditioning: `[..., N_atom, conditioning_size]` per-atom conditioning.
            packed_bias: Local pair bias `[..., N_window, Q, K, num_blocks * num_attention_heads]`.
            atom_mask: `[..., N_atom]` valid-atom mask; invalid atoms are no keys.
            window_info: The local window partition of the atom axis.

        Returns:
            Updated atom hidden states of shape `[..., N_atom, hidden_size]`.
        """

        layout = window_info.packed_layout
        if isinstance(atom_hidden, NestedTensor):
            if not isinstance(conditioning, NestedTensor) or not isinstance(packed_bias, NestedTensor):
                raise ValueError("packed atom attention requires NestedTensor hidden, conditioning, and bias")
            if not isinstance(atom_mask, NestedTensor):
                raise ValueError("packed atom attention requires a NestedTensor atom_mask")
            if layout is None:
                raise ValueError("packed atom attention requires a packed window partition")

            sampled = atom_hidden.concat.dim() == 3

            def to_query_windows(values: Tensor, operand: str) -> Tensor:
                windows = gather_packed_atom_windows(values, layout)[0]
                if values.dim() == 2:
                    return windows.unsqueeze(1)
                if values.dim() == 3:
                    return windows.movedim(2, 1)
                raise ValueError(f"packed atom attention {operand} must have shape (A_i, C) or (S, A_i, C)")

            query = to_query_windows(atom_hidden.concat, "hidden")
            conditioning_query = to_query_windows(conditioning.concat, "conditioning")
            # OOps gives rank-4 head-major QKV the mathematical meaning
            # ``(sample, head, token, channel)``. Local atom windows are
            # independent execution batches, not samples sharing one bias, so
            # retain an explicit unit sample axis instead of letting the dynamic
            # block count become the bias-sharing axis.
            if packed_bias.concat.dim() == 4:
                bias = packed_bias.concat.unsqueeze(1)
            elif packed_bias.concat.dim() == 5:
                bias = packed_bias.concat
            else:
                raise ValueError("packed atom attention bias must be shared or sample-specific")
            mask = gather_packed_atom_windows(atom_mask.concat.unsqueeze(-1), layout)[0]
            mask = mask.squeeze(-1).to(dtype=query.dtype)
            mask = mask.unsqueeze(1) if mask.dim() == 2 else mask.movedim(2, 1)

            def to_packed_keys_windowed(hidden: Tensor) -> Tensor:
                hidden = hidden.movedim(1, 2)
                return gather_packed_window_keys(hidden, layout).movedim(2, 1)

            output = self.transformer(query, conditioning_query, bias, mask, to_packed_keys_windowed)
            output_values = output.movedim(1, 2).flatten(0, 1).index_select(0, layout.atom_window_index)
            if not sampled:
                output_values = output_values.squeeze(1)
            return atom_hidden.packed_with_static_tail(output_values)
        if (
            isinstance(conditioning, NestedTensor)
            or isinstance(packed_bias, NestedTensor)
            or isinstance(atom_mask, NestedTensor)
        ):
            raise ValueError("dense atom attention requires dense conditioning, bias, and atom_mask tensors")
        if layout is not None:
            raise ValueError("dense atom attention requires a dense window partition")

        num_query_atoms = window_info.num_query_atoms
        num_key_atoms = window_info.num_key_atoms
        prefix = tuple(atom_hidden.shape[:-2])
        num_atoms = atom_hidden.shape[-2]
        query = _flatten_batch(atom_hidden, prefix, 2)
        conditioning = _flatten_batch(conditioning, prefix, 2)
        bias = _flatten_batch(packed_bias, prefix, 4)
        mask = _flatten_batch(atom_mask, prefix, 1).float()
        key_index = window_info.key_index
        if key_index is not None:
            key_index = _flatten_batch(key_index, prefix, 2)

        padding = (-num_atoms) % num_query_atoms
        if padding:
            query = F.pad(query, (0, 0, 0, padding))
            conditioning = F.pad(conditioning, (0, 0, 0, padding))
            mask = F.pad(mask, (0, padding))
        batch_size = query.shape[0]
        num_windows = query.shape[-2] // num_query_atoms
        query = query.reshape(batch_size * num_windows, num_query_atoms, -1)
        conditioning = conditioning.reshape(batch_size * num_windows, num_query_atoms, -1)
        mask = mask.reshape(batch_size * num_windows, num_query_atoms)
        bias = bias.reshape(batch_size * num_windows, num_query_atoms, num_key_atoms, -1)
        to_keys = partial(
            gather_key_windows,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            key_index=key_index,
        )

        def to_keys_windowed(hidden: Tensor) -> Tensor:
            r"""
            Map a per-window query pane to its gathered key pane for the local attention transformer.

            Takes ``hidden`` ``(batch * num_windows, num_query_atoms, C)``, flattens the windows back to the
            contiguous atom axis, gathers the overlapping key window around each query window, and returns
            ``(batch * num_windows, num_key_atoms, C)``.

            Args:
                hidden: Per-window query pane, shape `(batch * num_windows, num_query_atoms, C)`.

            Returns:
                Gathered key pane, shape `(batch * num_windows, num_key_atoms, C)`.
            """

            return to_keys(hidden.reshape(batch_size, num_windows * num_query_atoms, -1)).reshape(
                batch_size * num_windows, num_key_atoms, -1
            )

        output = self.transformer(query, conditioning, bias, mask, to_keys_windowed)
        output = output.reshape(batch_size, num_windows * num_query_atoms, -1)
        if padding:
            output = output[..., :num_atoms, :]
        return output.reshape(*prefix, num_atoms, output.shape[-1])


class AtomToTokenEncoder(nn.Module):
    r"""
    Attend locally over atom states and mean-pool them to tokens.

    The atom states are both the attention queries and their conditioning, under a pair bias already
    packed per block. The attended atoms are projected, ReLU'd, and averaged over the valid atoms of each
    token.

    Args:
        atom_hidden_size: Width of the atom states.
        token_hidden_size: Width of the produced token representation.
        num_blocks: Number of Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the transformer and the pooling.
        block: Block class of every block.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import AtomToTokenEncoder, LocalAtomWindowing
        >>> from ofoldx.modules.transformer import ConditionedTransformerEncoderBlock
        >>> encoder = AtomToTokenEncoder(
        ...     8, 6, num_blocks=1, num_attention_heads=2, block=ConditionedTransformerEncoderBlock
        ... )
        >>> atom_mask = torch.ones(1, 10, dtype=torch.bool)
        >>> window_info = LocalAtomWindowing(4, 8).window_info(atom_mask)
        >>> atom_to_token = torch.arange(10).div(2, rounding_mode="floor").unsqueeze(0)
        >>> encoder(
        ...     torch.randn(1, 10, 8),
        ...     torch.randn(1, 3, 4, 8, 2),
        ...     atom_mask,
        ...     atom_to_token,
        ...     torch.zeros(1, 5, 1),
        ...     window_info,
        ... ).shape
        torch.Size([1, 5, 6])
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        token_hidden_size: int = 384,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        backend: str = "auto",
        *,
        block: AtomTransformerBlock,
    ) -> None:
        super().__init__()
        self.atom_transformer = LocalAtomAttention(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            backend=backend,
            block=block,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

    def forward(
        self,
        atom_states: Tensor | NestedTensor,
        atom_pair_bias: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        token_reference: Tensor | NestedTensor,
        window_info: LocalAtomWindowInfo,
    ) -> Tensor | NestedTensor:
        r"""
        Args:
            atom_states: Atom states `[..., N_atom, atom_hidden_size]`.
            atom_pair_bias: Per-block attention bias `[..., N_window, Q, K, num_blocks * num_attention_heads]`.
            atom_mask: Valid-atom mask `[..., N_atom]`.
            atom_to_token: Token index of each atom `[..., N_atom]`.
            token_reference: Token-axis tensor whose topology defines the output; its values are not read.
            window_info: The local window partition of the atom axis.

        Returns:
            Token representation `[..., N_token, token_hidden_size]`.
        """

        atom_states = self.atom_transformer(atom_states, atom_states, atom_pair_bias, atom_mask, window_info)
        return self.atom_to_token_mean(
            F.relu(self.token_projection(atom_states)),
            atom_to_token,
            token_reference,
            atom_mask,
        )


class CoordinateAtomToTokenEncoder(nn.Module):
    r"""
    Add embedded noisy atom coordinates to the atom queries, attend locally, and mean-pool to tokens.

    The embedded noisy coordinates
    ([`embed_noisy_positions`][ofoldx.modules.atom.encoders.CoordinateAtomToTokenEncoder.embed_noisy_positions])
    are added to the atom queries of every sample, which attend under a pair bias already packed per block.
    The attended atoms are projected, ReLU'd, and averaged over the valid atoms of each token.

    Args:
        atom_hidden_size: Width of the atom query and conditioning states.
        token_hidden_size: Width of the produced token representation.
        num_blocks: Number of Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the transformer and the pooling.
        block: Block class of every block.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import CoordinateAtomToTokenEncoder, LocalAtomWindowing
        >>> from ofoldx.modules.transformer import ConditionedTransformerEncoderBlock
        >>> encoder = CoordinateAtomToTokenEncoder(
        ...     8, 6, num_blocks=1, num_attention_heads=2, block=ConditionedTransformerEncoderBlock
        ... )
        >>> atom_mask = torch.ones(1, 10, dtype=torch.bool)
        >>> window_info = LocalAtomWindowing(4, 8).window_info(atom_mask)
        >>> atom_to_token = torch.arange(10).div(2, rounding_mode="floor").unsqueeze(0)
        >>> token, atom_query = encoder(
        ...     torch.randn(1, 2, 10, 3),
        ...     torch.randn(1, 10, 8),
        ...     torch.randn(1, 10, 8),
        ...     torch.randn(1, 3, 4, 8, 2),
        ...     atom_mask,
        ...     atom_to_token,
        ...     torch.zeros(1, 5, 1),
        ...     window_info,
        ... )
        >>> token.shape, atom_query.shape
        (torch.Size([1, 2, 5, 6]), torch.Size([1, 2, 10, 8]))
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        backend: str = "auto",
        *,
        block: AtomTransformerBlock,
    ) -> None:
        super().__init__()
        self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_transformer = LocalAtomAttention(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            backend=backend,
            block=block,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

    def embed_noisy_positions(self, noisy_positions: Tensor | NestedTensor) -> Tensor | NestedTensor:
        r"""
        Embed noisy atom coordinates at the atom hidden width.

        Args:
            noisy_positions: Scaled noisy coordinates `[..., N_sample, N_atom, 3]`.

        Returns:
            The coordinate embedding `[..., N_sample, N_atom, atom_hidden_size]`.
        """

        noisy_positions = project_coordinates(
            noisy_positions,
            torch.ones_like(noisy_positions[..., 0], dtype=torch.bool),
            dtype=self.noisy_position_projection.weight.dtype,
        )
        return self.noisy_position_projection(noisy_positions)

    def forward(
        self,
        noisy_positions: Tensor | NestedTensor,
        atom_query: Tensor | NestedTensor,
        atom_conditioning: Tensor | NestedTensor,
        atom_pair_bias: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        token_reference: Tensor | NestedTensor,
        window_info: LocalAtomWindowInfo,
    ) -> tuple[Tensor | NestedTensor, Tensor | NestedTensor]:
        r"""
        Args:
            noisy_positions: Scaled noisy coordinates `[..., N_sample, N_atom, 3]`.
            atom_query: Atom query states `[..., N_atom, atom_hidden_size]`, optionally with the sample axis.
            atom_conditioning: Atom conditioning `[..., N_atom, atom_hidden_size]`, optionally with the
                sample axis.
            atom_pair_bias: Per-block attention bias `[..., N_window, Q, K, num_blocks * num_attention_heads]`.
            atom_mask: Valid-atom mask `[..., N_atom]`.
            atom_to_token: Token index of each atom `[..., N_atom]`.
            token_reference: Token-axis tensor whose topology defines the output; its values are not read.
            window_info: The local window partition of the atom axis.

        Returns:
            token: Token representation `[..., N_sample, N_token, token_hidden_size]`.
            atom_query: Attended atom states `[..., N_sample, N_atom, atom_hidden_size]`.
        """

        observed = atom_mask.to(torch.bool)
        while observed.dim() < noisy_positions.dim() - 1:
            observed = observed.unsqueeze(-2)
        noisy_positions = project_coordinates(
            noisy_positions,
            observed & torch.ones_like(noisy_positions[..., 0], dtype=torch.bool),
            dtype=self.noisy_position_projection.weight.dtype,
        )
        if noisy_positions.dim() == atom_query.dim() + 1:
            atom_query = atom_query.unsqueeze(-3)
        if noisy_positions.dim() == atom_conditioning.dim() + 1:
            atom_conditioning = atom_conditioning.unsqueeze(-3)
        atom_query = atom_query + self.embed_noisy_positions(noisy_positions)
        atom_query = self.atom_transformer(atom_query, atom_conditioning, atom_pair_bias, atom_mask, window_info)
        token = self.atom_to_token_mean(
            F.relu(self.token_projection(atom_query.to(self.token_projection.weight))),
            atom_to_token,
            token_reference,
            atom_mask,
        )
        return token, atom_query


class TokenToAtomDecoder(nn.Module):
    r"""
    Broadcast refined token states onto attended atom states and predict per-atom coordinate updates.

    The projected token states are added to the atom states of their atoms, which attend locally under a
    pair bias already packed per block
    ([`decode_atoms`][ofoldx.modules.atom.encoders.TokenToAtomDecoder.decode_atoms]); a normalized
    projection gives each atom's coordinate update.

    Args:
        token_hidden_size: Width of the refined token representation.
        atom_hidden_size: Width of the atom query and conditioning states.
        num_blocks: Number of Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the transformer, the broadcast, and the normalization.
        block: Block class of every block.
        bias: Whether the output layer normalization has a learnable bias.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import LocalAtomWindowing, TokenToAtomDecoder
        >>> from ofoldx.modules.transformer import ConditionedTransformerEncoderBlock
        >>> decoder = TokenToAtomDecoder(
        ...     6, 8, num_blocks=1, num_attention_heads=2, block=ConditionedTransformerEncoderBlock
        ... )
        >>> atom_mask = torch.ones(1, 10, dtype=torch.bool)
        >>> window_info = LocalAtomWindowing(4, 8).window_info(atom_mask)
        >>> atom_to_token = torch.arange(10).div(2, rounding_mode="floor").unsqueeze(0)
        >>> decoder(
        ...     torch.randn(1, 2, 5, 6),
        ...     torch.randn(1, 2, 10, 8),
        ...     torch.randn(1, 10, 8),
        ...     torch.randn(1, 3, 4, 8, 2),
        ...     atom_mask,
        ...     atom_to_token,
        ...     window_info,
        ... ).shape
        torch.Size([1, 2, 10, 3])
    """

    def __init__(
        self,
        token_hidden_size: int = 768,
        atom_hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        backend: str = "auto",
        *,
        block: AtomTransformerBlock,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(token_hidden_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.atom_transformer = LocalAtomAttention(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            backend=backend,
            block=block,
        )
        self.position_norm = LayerNorm(atom_hidden_size, bias=bias, backend=backend)
        self.position_update_projection = nn.Linear(atom_hidden_size, 3, bias=False)

    def decode_atoms(
        self,
        token: Tensor | NestedTensor,
        atom_query: Tensor | NestedTensor,
        atom_conditioning: Tensor | NestedTensor,
        atom_pair_bias: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        window_info: LocalAtomWindowInfo,
    ) -> Tensor | NestedTensor:
        r"""
        Decode refined token states into attended atom states.

        Args:
            token: Refined token representation `[..., N_sample, N_token, token_hidden_size]`.
            atom_query: Attended atom states of the encoder `[..., N_sample, N_atom, atom_hidden_size]`.
            atom_conditioning: Atom conditioning `[..., N_atom, atom_hidden_size]`, optionally with the
                sample axis.
            atom_pair_bias: Per-block attention bias `[..., N_window, Q, K, num_blocks * num_attention_heads]`.
            atom_mask: Valid-atom mask `[..., N_atom]`.
            atom_to_token: Token index of each atom `[..., N_atom]`.
            window_info: The local window partition of the atom axis.

        Returns:
            Attended atom states `[..., N_sample, N_atom, atom_hidden_size]`.
        """

        atom_query = atom_query + self.token_to_atom_broadcast(self.token_projection(token), atom_to_token)
        return self.atom_transformer(atom_query, atom_conditioning, atom_pair_bias, atom_mask, window_info)

    def forward(
        self,
        token: Tensor | NestedTensor,
        atom_query: Tensor | NestedTensor,
        atom_conditioning: Tensor | NestedTensor,
        atom_pair_bias: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        window_info: LocalAtomWindowInfo,
    ) -> Tensor | NestedTensor:
        r"""
        Args:
            token: Refined token representation `[..., N_sample, N_token, token_hidden_size]`.
            atom_query: Attended atom states of the encoder `[..., N_sample, N_atom, atom_hidden_size]`.
            atom_conditioning: Atom conditioning `[..., N_atom, atom_hidden_size]`, optionally with the
                sample axis.
            atom_pair_bias: Per-block attention bias `[..., N_window, Q, K, num_blocks * num_attention_heads]`.
            atom_mask: Valid-atom mask `[..., N_atom]`.
            atom_to_token: Token index of each atom `[..., N_atom]`.
            window_info: The local window partition of the atom axis.

        Returns:
            Per-atom coordinate updates `[..., N_sample, N_atom, 3]`.
        """

        atom_query = self.decode_atoms(
            token,
            atom_query,
            atom_conditioning,
            atom_pair_bias,
            atom_mask,
            atom_to_token,
            window_info,
        )
        return layer_norm_projection(atom_query, norm=self.position_norm, projection=self.position_update_projection)


def _insert_prefix_axes(tensor: Tensor, prefix_ndim: int, target_prefix_ndim: int) -> Tensor:
    r"""Insert singleton axes after the leading ``prefix_ndim`` axes until ``target_prefix_ndim`` lead."""

    missing = target_prefix_ndim - prefix_ndim
    if missing <= 0:
        return tensor
    return tensor.reshape(*tensor.shape[:prefix_ndim], *((1,) * missing), *tensor.shape[prefix_ndim:])


def _flatten_batch(tensor: Tensor, prefix: tuple[int, ...], payload_ndim: int) -> Tensor:
    r"""
    Broadcast a tensor to the full leading ``prefix`` and collapse that prefix into one batch axis.

    The atom stack runs over arbitrary leading (batch[, sample]) dims: in diffusion the query/conditioning/pair
    carry a noise-sample axis the mask does not. The conditioned Transformer encoder wants one batch axis, so each input
    is broadcast to the query's prefix and flattened.

    Args:
        tensor: Input whose leading dims are a left-prefix of ``prefix`` followed by ``payload_ndim`` axes.
        prefix: Full target leading shape (e.g. the query's ``(batch[, sample])`` dims).
        payload_ndim: Number of trailing payload axes to leave untouched.

    Returns:
        The tensor broadcast to ``prefix`` and reshaped to `[prod(prefix), *payload]`.
    """
    matched = tensor.dim() - payload_ndim
    missing = len(prefix) - matched
    if missing:
        tensor = tensor.reshape(*tensor.shape[:matched], *(1,) * missing, *tensor.shape[matched:])
    tensor = tensor.expand(*prefix, *tensor.shape[len(prefix) :])
    return tensor.reshape(prod(prefix), *tensor.shape[len(prefix) :])
