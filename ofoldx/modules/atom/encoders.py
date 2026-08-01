r"""
Atom-to-token encoders and token-to-atom decoders.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from ..backend import LayerNorm
from ..transformer import (
    AtomCrossAttentionTransformerEncoder,
    NormalizedKeyAtomCrossAttentionTransformerEncoder,
)
from .context import AtomToTokenOutput
from .nested import (
    PackedLocalAtomLayout,
    gather_packed_atom_windows,
    gather_packed_window_keys,
    packed_local_atom_layout,
)
from .pair_encoding import AtomPairEncoding
from .transfer import (
    AtomToTokenMean,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)
from .window import LocalAtomGeometry, LocalAtomGeometryBuilder, gather_key_windows


class AtomConditioner(nn.Module):
    r"""
    Build per-atom conditioning states from reference position, charge, and feature inputs.

    Projects each reference input to the atom hidden width and sums them, giving the shared per-atom
    conditioning consumed by the atom-pair encoder and atom self-attention.

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
            reference_charges: `[..., N_atom, 1]` reference charges.
            reference_mask: `[..., N_atom]` validity mask; absent atoms are zeroed.

        Returns:
            Per-atom conditioning of shape `[..., N_atom, atom_hidden_size]`.
        """

        conditioning = self.reference_position_projection(reference_positions)
        conditioning = conditioning + self.reference_charge_projection(reference_charges)
        conditioning = conditioning + self.atom_feature_projection(atom_features.to(dtype=conditioning.dtype))
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
    Build local-window atom-pair representations from atom geometry and conditioning.

    Combines distance-vector / inverse-distance / mask projections with left/right projections of the atom
    conditioning through the shared atom-pair encoding equation over precomputed local geometry.

    Args:
        atom_hidden_size: Width of the incoming atom conditioning channel.
        atom_pair_hidden_size: Width of the produced atom-pair channel.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
    ) -> None:
        super().__init__()
        self.pair_encoding = AtomPairEncoding(atom_hidden_size, atom_pair_hidden_size)
        self.atom_pair_hidden_size = atom_pair_hidden_size
        self.local_geometry_builder = LocalAtomGeometryBuilder(
            num_query_atoms=num_query_atoms, num_key_atoms=num_key_atoms
        )

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        r"""
        Precompute the windowed atom-pair geometry for the local encoding layout.

        Args:
            reference_positions: `[..., N_atom, 3]` reference coordinates.
            reference_space_id: `[..., N_atom]` reference-space ids grouping atoms that may interact.
            atom_mask: Optional valid-atom mask selecting row-specific key windows.

        Returns:
            A [`LocalAtomGeometry`][ofoldx.modules.atom.window.LocalAtomGeometry] to pass to this encoder.
        """

        return self.local_geometry_builder(reference_positions, reference_space_id, atom_mask)

    def forward(
        self,
        atom_conditioning: Tensor | NestedTensor,
        *,
        local_geometry: LocalAtomGeometry,
        pair: Tensor | NestedTensor | None = None,
    ) -> Tensor | NestedTensor:
        r"""
        Encode the local atom-pair representation from precomputed geometry.

        Uses the geometry's per-window `query - key` distance vectors and validity/window masks, windows the
        ``atom_conditioning`` into query/key panes, and combines them with an optional local ``pair`` bias.

        Args:
            atom_conditioning: `[..., N_atom, atom_hidden_size]` per-atom conditioning.
            local_geometry: Precomputed [`LocalAtomGeometry`][ofoldx.modules.atom.window.LocalAtomGeometry]
                carrying window distance vectors and masks.
            pair: Optional `[..., N_window, num_query_atoms, num_key_atoms, atom_pair_hidden_size]` pair bias.

        Returns:
            The local pair representation of shape
                `[..., N_window, num_query_atoms, num_key_atoms, atom_pair_hidden_size]`.
        """

        packed_layout = local_geometry.window_info.packed_layout
        if isinstance(atom_conditioning, NestedTensor):
            if packed_layout is None:
                raise ValueError("packed atom conditioning requires packed LocalAtomGeometry")
            conditioning_queries, conditioning_keys = gather_packed_atom_windows(
                atom_conditioning.concat,
                packed_layout,
            )
            pair_bias = pair
            if isinstance(pair, NestedTensor):
                pair_bias = pair.concat
            pair_values = self.pair_encoding(
                local_geometry.distance_vectors,
                F.relu(conditioning_queries[..., None, :]),
                F.relu(conditioning_keys[..., None, :, :]),
                distance_term_mask=(
                    local_geometry.distance_mask.to(dtype=local_geometry.distance_vectors.dtype)
                    * local_geometry.window_info.attention_mask.to(
                        device=local_geometry.distance_vectors.device,
                        dtype=local_geometry.distance_vectors.dtype,
                    ).unsqueeze(-1)
                ),
                inverse_term_mask=local_geometry.distance_mask.to(dtype=local_geometry.distance_vectors.dtype),
                valid_mask=local_geometry.distance_mask.to(dtype=local_geometry.distance_vectors.dtype),
                pair_bias=pair_bias,
                prefix_ndim=1,
            )
            return atom_conditioning.packed_with_lengths(pair_values, packed_layout.block_lengths)
        if packed_layout is not None or isinstance(pair, NestedTensor):
            raise ValueError("dense atom conditioning requires dense LocalAtomGeometry and pair tensors")

        distance_vectors = local_geometry.distance_vectors
        distance_mask = local_geometry.distance_mask
        attention_mask = local_geometry.window_info.attention_mask
        geometry_prefix_ndim = distance_vectors.dim() - 4
        atom_prefix_ndim = atom_conditioning.dim() - 2
        if atom_prefix_ndim == geometry_prefix_ndim + 1:
            distance_vectors = distance_vectors.unsqueeze(geometry_prefix_ndim)
            distance_mask = distance_mask.unsqueeze(geometry_prefix_ndim)
            attention_mask = attention_mask.unsqueeze(geometry_prefix_ndim)

        distance_mask = distance_mask.to(dtype=distance_vectors.dtype)
        window_mask = attention_mask.to(device=distance_vectors.device, dtype=distance_vectors.dtype).unsqueeze(-1)
        windowing = self.local_geometry_builder.windowing
        num_windows = distance_vectors.shape[-4]
        query_padding = num_windows * windowing.num_query_atoms - atom_conditioning.shape[-2]
        conditioning_queries = F.pad(atom_conditioning, (0, 0, 0, query_padding)).reshape(
            *atom_conditioning.shape[:-2],
            num_windows,
            windowing.num_query_atoms,
            atom_conditioning.shape[-1],
        )
        conditioning_keys = gather_key_windows(
            atom_conditioning,
            windowing.num_query_atoms,
            windowing.num_key_atoms,
            local_geometry.window_info.key_index,
        )
        return self.pair_encoding(
            distance_vectors,
            F.relu(conditioning_queries[..., None, :]),
            F.relu(conditioning_keys[..., None, :, :]),
            distance_term_mask=distance_mask * window_mask,
            inverse_term_mask=distance_mask,
            valid_mask=distance_mask,
            pair_bias=pair,
            prefix_ndim=atom_conditioning.dim() - 2,
        )


class PackedAtomAttention(nn.Module):
    r"""
    Apply a concrete atom transformer over an already-packed local pair bias.

    ``packed_bias`` stores all block/head pair-bias projections on the last axis, so the underlying
    [`ConditionedTransformerEncoder`][ofoldx.modules.transformer.ConditionedTransformerEncoder]
    can slice its per-block bias without re-projecting.

    Args:
        transformer: Concrete atom transformer whose block/head biases are packed on the last axis.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
    """

    def __init__(
        self,
        transformer: nn.Module,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms

    def forward(
        self,
        atom_hidden: Tensor | NestedTensor,
        conditioning: Tensor | NestedTensor,
        packed_bias: Tensor | NestedTensor,
        *,
        atom_mask: Tensor | NestedTensor | None = None,
        key_index: Tensor | None = None,
        packed_layout: PackedLocalAtomLayout | None = None,
    ) -> Tensor | NestedTensor:
        r"""
        Apply local-window atom attention.

        Flattens arbitrary leading (batch[, sample]) dims into one batch axis, runs the packed-bias
        transformer, and reshapes back. The atom axis is padded up to a whole number of query windows,
        split into windows, attended with a windowed key gather, then unpadded.

        Args:
            atom_hidden: `[..., N_atom, atom_hidden_size]` atom query states.
            conditioning: `[..., N_atom, conditioning_size]` per-atom conditioning.
            packed_bias: Local pair bias of shape
                `[..., N_window, num_query_atoms, num_key_atoms, num_blocks * num_attention_heads]`.
            atom_mask: Optional `[..., N_atom]` attention mask; defaults to all-valid.
            key_index: Optional row-specific atom indices for the local key panes.

        Returns:
            Updated atom hidden states of shape `[..., N_atom, atom_hidden_size]`.
        """

        if isinstance(atom_hidden, NestedTensor):
            if not isinstance(conditioning, NestedTensor) or not isinstance(packed_bias, NestedTensor):
                raise ValueError("packed atom attention requires NestedTensor hidden, conditioning, and bias")
            if atom_mask is not None and not isinstance(atom_mask, NestedTensor):
                raise ValueError("packed atom attention requires a NestedTensor atom_mask")
            if packed_layout is None:
                if atom_mask is None:
                    raise ValueError("packed atom attention requires atom_mask or a PackedLocalAtomLayout")
                packed_layout = packed_local_atom_layout(
                    atom_mask.unsqueeze(-1),
                    None,
                    self.num_query_atoms,
                    self.num_key_atoms,
                )
            if key_index is not None:
                raise ValueError("packed atom attention reconstructs row-specific keys from packed layout metadata")

            sampled = atom_hidden.concat.dim() == 3

            def to_query_windows(values: Tensor, operand: str) -> Tensor:
                windows = gather_packed_atom_windows(values, packed_layout)[0]
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
            if atom_mask is None:
                mask = packed_layout.query_valid.to(dtype=query.dtype).unsqueeze(1)
            else:
                mask = gather_packed_atom_windows(atom_mask.concat.unsqueeze(-1), packed_layout)[0]
                mask = mask.squeeze(-1).to(dtype=query.dtype)
                mask = mask.unsqueeze(1) if mask.dim() == 2 else mask.movedim(2, 1)

            def to_packed_keys_windowed(hidden: Tensor) -> Tensor:
                hidden = hidden.movedim(1, 2)
                return gather_packed_window_keys(hidden, packed_layout).movedim(2, 1)

            output = self.transformer(query, conditioning_query, bias, mask, to_packed_keys_windowed)
            output_values = output.movedim(1, 2).flatten(0, 1).index_select(0, packed_layout.atom_window_index)
            if not sampled:
                output_values = output_values.squeeze(1)
            return atom_hidden.packed_with_static_tail(output_values)
        if (
            isinstance(conditioning, NestedTensor)
            or isinstance(packed_bias, NestedTensor)
            or isinstance(atom_mask, NestedTensor)
        ):
            raise ValueError("dense atom attention requires dense conditioning, bias, and atom_mask tensors")
        if packed_layout is not None:
            raise ValueError("dense atom attention does not accept a PackedLocalAtomLayout")

        prefix = tuple(atom_hidden.shape[:-2])
        num_atoms = atom_hidden.shape[-2]
        if atom_mask is None:
            atom_mask = atom_hidden.new_ones(*prefix, num_atoms)

        query = _flatten_batch(atom_hidden, prefix, 2)
        conditioning = _flatten_batch(conditioning, prefix, 2)
        bias = _flatten_batch(packed_bias, prefix, 4)
        mask = _flatten_batch(atom_mask, prefix, 1).float()
        if key_index is not None:
            key_index = _flatten_batch(key_index, prefix, 2)

        num_query_atoms = self.num_query_atoms
        padding = (-num_atoms) % num_query_atoms
        if padding:
            query = F.pad(query, (0, 0, 0, padding))
            conditioning = F.pad(conditioning, (0, 0, 0, padding))
            mask = F.pad(mask, (0, padding))
        batch = query.shape[0]
        num_windows = query.shape[-2] // num_query_atoms
        query = query.reshape(batch * num_windows, num_query_atoms, -1)
        conditioning = conditioning.reshape(batch * num_windows, num_query_atoms, -1)
        mask = mask.reshape(batch * num_windows, num_query_atoms)
        bias = bias.reshape(batch * num_windows, num_query_atoms, self.num_key_atoms, -1)
        to_keys = partial(
            gather_key_windows,
            num_query_atoms=num_query_atoms,
            num_key_atoms=self.num_key_atoms,
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

            return to_keys(hidden.reshape(batch, num_windows * num_query_atoms, -1)).reshape(
                batch * num_windows, self.num_key_atoms, -1
            )

        output = self.transformer(query, conditioning, bias, mask, to_keys_windowed)
        output = output.reshape(batch, num_windows * num_query_atoms, -1)
        if padding:
            output = output[..., :num_atoms, :]
        return output.reshape(*prefix, num_atoms, output.shape[-1])


class AtomSelfAttention(nn.Module):
    r"""Apply local atom self-attention with one pair-bias projection per block."""

    def __init__(
        self,
        atom_hidden_size: int = 128,
        conditioning_size: int = 128,
        pair_size: int = 16,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norms = nn.ModuleList(LayerNorm(pair_size, bias=False, backend=backend) for _ in range(num_blocks))
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )
        self.atom_attention = PackedAtomAttention(
            AtomCrossAttentionTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=conditioning_size,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )

    def project_pair_bias(self, pair: Tensor | NestedTensor) -> Tensor | NestedTensor:
        r"""Project local atom pairs to packed per-block attention biases."""

        return torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections, strict=True)],
            dim=-1,
        )

    def forward_with_packed_bias(
        self,
        atom_hidden: Tensor | NestedTensor,
        conditioning: Tensor | NestedTensor,
        packed_bias: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor | None = None,
        key_index: Tensor | None = None,
        packed_layout: PackedLocalAtomLayout | None = None,
    ) -> Tensor | NestedTensor:
        r"""Apply atom attention from an already-projected packed pair bias."""

        return self.atom_attention(
            atom_hidden,
            conditioning,
            packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
            packed_layout=packed_layout,
        )

    def forward(
        self,
        atom_hidden: Tensor | NestedTensor,
        conditioning: Tensor | NestedTensor,
        pair: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor | None = None,
        key_index: Tensor | None = None,
        packed_layout: PackedLocalAtomLayout | None = None,
    ) -> Tensor | NestedTensor:
        return self.forward_with_packed_bias(
            atom_hidden,
            conditioning,
            self.project_pair_bias(pair),
            atom_mask=atom_mask,
            key_index=key_index,
            packed_layout=packed_layout,
        )


class AtomToTokenEncoder(nn.Module):
    r"""Encode local reference-atom features and geometry into token states."""

    def __init__(
        self,
        token_hidden_size: int,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        atom_feature_size: int = 385,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_conditioner = AtomConditioner(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
        )
        self.atom_pair_encoder = AtomPairEncoder(
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_self_attention = AtomSelfAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.atom_to_token_mean = AtomToTokenMean(backend=backend, denominator_offset=None)

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        return self.atom_pair_encoder.build_local_geometry(reference_positions, reference_space_id, atom_mask)

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        local_geometry: LocalAtomGeometry,
        atom_mask: Tensor | None = None,
        atom_to_token: Tensor,
        token_reference: Tensor,
    ) -> AtomToTokenOutput:
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        atom_pair = self.atom_pair_encoder(
            atom_conditioning,
            local_geometry=local_geometry,
        )
        atom_query = self.atom_self_attention(
            atom_conditioning,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            key_index=local_geometry.window_info.key_index,
            packed_layout=local_geometry.window_info.packed_layout,
        )
        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = self.atom_to_token_mean(
            token_hidden,
            atom_to_token,
            token_reference,
            atom_mask,
        )
        return AtomToTokenOutput(
            token_hidden=token_hidden,
            atom_query=atom_query,
            atom_pair=atom_pair,
            atom_conditioning=atom_conditioning,
        )


class CoordinateAtomToTokenEncoder(nn.Module):
    r"""Encode local noisy-atom coordinates and trunk context into token states."""

    def __init__(
        self,
        token_hidden_size: int,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        atom_feature_size: int = 385,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_conditioner = AtomConditioner(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
        )
        self.single_norm = LayerNorm(single_size, bias=False, backend=backend)
        self.single_to_atom_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.pair_norm = LayerNorm(pair_size, bias=False, backend=backend)
        self.pair_projection = nn.Linear(pair_size, atom_pair_hidden_size, bias=False)
        self.token_pair_to_local_atom_pair_broadcast = TokenPairToLocalAtomPairBroadcast(
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.atom_pair_encoder = AtomPairEncoder(
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_self_attention = AtomSelfAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.atom_to_token_mean = AtomToTokenMean(backend=backend, denominator_offset=None)

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        return self.atom_pair_encoder.build_local_geometry(reference_positions, reference_space_id, atom_mask)

    def prepare_context(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        single: Tensor,
        pair: Tensor,
        local_geometry: LocalAtomGeometry,
        atom_to_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Prepare atom conditioning and local pair features shared across samples."""

        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        single_conditioning = self.single_to_atom_projection(self.single_norm(single))
        single_conditioning = self.token_to_atom_broadcast(single_conditioning, atom_to_token)
        if single_conditioning.dim() == atom_conditioning.dim() + 1:
            atom_conditioning = atom_conditioning.unsqueeze(-3)
        atom_conditioning = atom_conditioning + single_conditioning

        pair = self.pair_projection(self.pair_norm(pair))
        local_pair = self.token_pair_to_local_atom_pair_broadcast(
            pair,
            atom_to_token,
            key_index=local_geometry.window_info.key_index,
            valid_extent=local_geometry.window_info.valid_extent,
        )
        atom_pair = self.atom_pair_encoder(
            atom_conditioning,
            local_geometry=local_geometry,
            pair=local_pair,
        )
        return atom_conditioning, atom_pair

    def forward_prepared(
        self,
        noisy_coords: Tensor,
        atom_conditioning: Tensor,
        atom_packed_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_reference: Tensor,
        key_index: Tensor | None,
        packed_layout: PackedLocalAtomLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Encode one or more coordinate samples from prepared atom tensors."""

        if noisy_coords.dim() == atom_conditioning.dim() + 1:
            atom_conditioning = atom_conditioning.unsqueeze(-3)
        atom_query = atom_conditioning + self.noisy_position_projection(noisy_coords)
        atom_query = self.atom_self_attention.forward_with_packed_bias(
            atom_query,
            atom_conditioning,
            atom_packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
            packed_layout=packed_layout,
        )
        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = self.atom_to_token_mean(
            token_hidden,
            atom_to_token,
            token_reference,
            atom_mask,
        )
        return token_hidden, atom_query

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        single: Tensor,
        pair: Tensor,
        noisy_coords: Tensor,
        local_geometry: LocalAtomGeometry,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_reference: Tensor,
    ) -> AtomToTokenOutput:
        atom_conditioning, atom_pair = self.prepare_context(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
            single=single,
            pair=pair,
            local_geometry=local_geometry,
            atom_to_token=atom_to_token,
        )
        token_hidden, atom_query = self.forward_prepared(
            noisy_coords,
            atom_conditioning,
            self.atom_self_attention.project_pair_bias(atom_pair),
            atom_mask,
            atom_to_token,
            token_reference,
            local_geometry.window_info.key_index,
            local_geometry.window_info.packed_layout,
        )
        return AtomToTokenOutput(
            token_hidden=token_hidden,
            atom_query=atom_query,
            atom_pair=atom_pair,
            atom_conditioning=atom_conditioning,
        )


class TokenToAtomDecoder(nn.Module):
    r"""Decode refined token states into local per-atom coordinate updates."""

    def __init__(
        self,
        token_hidden_size: int = 768,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(token_hidden_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.atom_self_attention = AtomSelfAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.query_norm = LayerNorm(atom_hidden_size, bias=False, backend=backend)
        self.output_projection = nn.Linear(atom_hidden_size, 3, bias=False)

    def forward_prepared(
        self,
        atom_query: Tensor,
        atom_conditioning: Tensor,
        atom_packed_bias: Tensor,
        token_hidden: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        key_index: Tensor | None,
    ) -> Tensor:
        r"""Decode coordinate updates from prepared atom tensors."""

        atom_query = atom_query + self.token_to_atom_broadcast(
            self.token_projection(token_hidden),
            atom_to_token,
        )
        atom_query = self.atom_self_attention.forward_with_packed_bias(
            atom_query,
            conditioning=atom_conditioning,
            packed_bias=atom_packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
        )
        return self.output_projection(self.query_norm(atom_query))

    def forward(
        self,
        atom_output: AtomToTokenOutput,
        token_hidden: Tensor,
        *,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        key_index: Tensor | None,
    ) -> Tensor:
        return self.forward_prepared(
            atom_output.atom_query,
            atom_output.atom_conditioning,
            self.atom_self_attention.project_pair_bias(atom_output.atom_pair),
            token_hidden,
            atom_to_token,
            atom_mask,
            key_index,
        )


class NormalizedKeyAtomSelfAttention(nn.Module):
    r"""
    Atom self-attention whose gathered keys are normalized from query-normalized states.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        conditioning_size: int = 128,
        pair_size: int = 16,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norms = nn.ModuleList(LayerNorm(pair_size, bias=False, backend=backend) for _ in range(num_blocks))
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )
        self.atom_attention = PackedAtomAttention(
            NormalizedKeyAtomCrossAttentionTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=conditioning_size,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )

    def project_pair_bias(self, pair: Tensor | NestedTensor) -> Tensor | NestedTensor:
        r"""Project local atom pairs to packed per-block attention biases."""

        return torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections, strict=True)],
            dim=-1,
        )

    def forward_with_packed_bias(
        self,
        atom_hidden: Tensor | NestedTensor,
        conditioning: Tensor | NestedTensor,
        packed_bias: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor | None = None,
        key_index: Tensor | None = None,
        packed_layout: PackedLocalAtomLayout | None = None,
    ) -> Tensor | NestedTensor:
        r"""Apply atom attention from an already-projected packed pair bias."""

        return self.atom_attention(
            atom_hidden,
            conditioning,
            packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
            packed_layout=packed_layout,
        )

    def forward(
        self,
        atom_hidden: Tensor | NestedTensor,
        conditioning: Tensor | NestedTensor,
        pair: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor | None = None,
        key_index: Tensor | None = None,
        packed_layout: PackedLocalAtomLayout | None = None,
    ) -> Tensor | NestedTensor:
        return self.forward_with_packed_bias(
            atom_hidden,
            conditioning,
            self.project_pair_bias(pair),
            atom_mask=atom_mask,
            key_index=key_index,
            packed_layout=packed_layout,
        )


class NormalizedKeyAtomToTokenEncoder(nn.Module):
    r"""
    Reference-atom encoder with query-normalized gathered keys.
    """

    def __init__(
        self,
        token_hidden_size: int,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        atom_feature_size: int = 385,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_conditioner = AtomConditioner(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
        )
        self.atom_pair_encoder = AtomPairEncoder(
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_self_attention = NormalizedKeyAtomSelfAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.atom_to_token_mean = AtomToTokenMean(backend=backend, denominator_offset=None)

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        return self.atom_pair_encoder.build_local_geometry(reference_positions, reference_space_id, atom_mask)

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        local_geometry: LocalAtomGeometry,
        atom_mask: Tensor | None = None,
        atom_to_token: Tensor,
        token_reference: Tensor,
    ) -> AtomToTokenOutput:
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        atom_pair = self.atom_pair_encoder(
            atom_conditioning,
            local_geometry=local_geometry,
        )
        atom_query = self.atom_self_attention(
            atom_conditioning,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            key_index=local_geometry.window_info.key_index,
            packed_layout=local_geometry.window_info.packed_layout,
        )
        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = self.atom_to_token_mean(
            token_hidden,
            atom_to_token,
            token_reference,
            atom_mask,
        )
        return AtomToTokenOutput(
            token_hidden=token_hidden,
            atom_query=atom_query,
            atom_pair=atom_pair,
            atom_conditioning=atom_conditioning,
        )


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
    return tensor.reshape(-1, *tensor.shape[len(prefix) :])
