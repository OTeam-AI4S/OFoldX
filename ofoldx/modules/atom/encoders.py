r"""
Atom-to-token encoders and token-to-atom decoders.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..backend import LayerNorm
from ..transformer import (
    AtomCrossAttentionTransformerEncoder,
    ConditionedTransformerEncoder,
    NormalizedKeyAtomCrossAttentionTransformerEncoder,
)
from .context import AtomToTokenOutput
from .pair_encoding import AtomPairEncoding
from .transfer import (
    AtomToTokenMean,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
    dense_mean_to_tokens,
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
        atom_conditioning: Tensor,
        *,
        local_geometry: LocalAtomGeometry,
        pair: Tensor | None = None,
    ) -> Tensor:
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


class PackedReferenceAtomEncoder(nn.Module):
    r"""
    Project reference atom features and geometry into packed atom single and pair states.
    """

    def __init__(
        self,
        atom_feature_size: int = 388,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
    ) -> None:
        super().__init__()
        self.pair_encoding = AtomPairEncoding(atom_hidden_size, atom_pair_hidden_size)
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)

    def forward(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        atom_query = self.atom_feature_projection(atom_features)
        atom_single = atom_query

        num_query_atoms, num_key_atoms = self.num_query_atoms, self.num_key_atoms
        batch, length = atom_single.shape[:2]
        padding = (-length) % num_query_atoms
        num_windows = (length + padding) // num_query_atoms
        to_keys = partial(
            gather_key_windows,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )

        def pad_atoms(value: Tensor) -> Tensor:
            r"""
            Zero-pad the atom axis (dim 1) up to a whole number of query windows; no-op when already aligned.

            Args:
                value: Tensor whose dim-1 atom axis is padded from ``length`` to ``length + padding``.

            Returns:
                ``value`` with its atom axis right-padded with zeros, or ``value`` unchanged when ``padding`` is 0.
            """

            if not padding:
                return value
            return torch.nn.functional.pad(value, [0, 0] * (value.dim() - 2) + [0, padding])

        positions = pad_atoms(reference_positions)
        positions_queries = positions.view(batch, num_windows, num_query_atoms, 1, 3)
        positions_keys = to_keys(positions).view(batch, num_windows, 1, num_key_atoms, 3)
        distance_vectors = positions_keys - positions_queries

        mask = pad_atoms(atom_mask).bool()
        mask_queries = mask.view(batch, num_windows, num_query_atoms, 1)
        mask_keys = to_keys(mask.unsqueeze(-1).float()).view(batch, num_windows, 1, num_key_atoms).bool()
        space_id = pad_atoms(reference_space_id)
        uid_queries = space_id.view(batch, num_windows, num_query_atoms, 1)
        uid_keys = to_keys(space_id.unsqueeze(-1)).view(batch, num_windows, 1, num_key_atoms)
        valid = (mask_queries & mask_keys & (uid_queries == uid_keys)).to(distance_vectors.dtype).unsqueeze(-1)

        single_windows = F.relu(pad_atoms(atom_single))
        atom_pair = self.pair_encoding(
            distance_vectors,
            single_windows.view(batch, num_windows, num_query_atoms, 1, -1),
            to_keys(single_windows).view(batch, num_windows, 1, num_key_atoms, -1),
            distance_term_mask=valid,
            inverse_term_mask=valid,
            valid_mask=valid,
            pair_bias=None,
            prefix_ndim=1,
        )
        return atom_query, atom_single, atom_pair


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
        atom_hidden: Tensor,
        conditioning: Tensor,
        packed_bias: Tensor,
        *,
        atom_mask: Tensor | None = None,
        key_index: Tensor | None = None,
    ) -> Tensor:
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

    def project_pair_bias(self, pair: Tensor) -> Tensor:
        r"""Project local atom pairs to packed per-block attention biases."""

        return torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections, strict=True)],
            dim=-1,
        )

    def forward_with_packed_bias(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        packed_bias: Tensor,
        atom_mask: Tensor | None = None,
        key_index: Tensor | None = None,
    ) -> Tensor:
        r"""Apply atom attention from an already-projected packed pair bias."""

        return self.atom_attention(
            atom_hidden,
            conditioning,
            packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
        )

    def forward(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        atom_mask: Tensor | None = None,
        key_index: Tensor | None = None,
    ) -> Tensor:
        return self.forward_with_packed_bias(
            atom_hidden,
            conditioning,
            self.project_pair_bias(pair),
            atom_mask=atom_mask,
            key_index=key_index,
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
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

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
        num_tokens: int,
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
        )
        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = self.atom_to_token_mean(
            token_hidden,
            atom_to_token,
            num_tokens,
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
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

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
        num_tokens: int,
        key_index: Tensor | None,
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
        )
        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = self.atom_to_token_mean(
            token_hidden,
            atom_to_token,
            num_tokens,
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
        num_tokens: int,
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
            num_tokens,
            local_geometry.window_info.key_index,
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


class PackedAtomToTokenEncoder(nn.Module):
    r"""
    Run packed local atom attention and mean-pool the result to token representations.

    Args:
        atom_hidden_size: Width of the atom hidden and conditioning channels.
        token_hidden_size: Width of the produced token representation.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the conditioned Transformer encoder.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        token_hidden_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_transformer = PackedAtomAttention(
            ConditionedTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=atom_hidden_size,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.token_activation = nn.ReLU()

    def forward(
        self,
        atom_single: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
    ) -> Tensor:
        atom_query = self.atom_transformer(
            atom_single,
            atom_conditioning,
            atom_pair_bias,
            atom_mask=atom_mask.bool(),
        )
        token_from_atom = self.token_activation(self.token_projection(atom_query))
        return dense_mean_to_tokens(
            token_from_atom,
            atom_to_token,
            num_tokens,
            atom_mask=atom_mask,
            count_eps=1e-6,
        )


class PackedDenoisingAtomToTokenEncoder(nn.Module):
    r"""
    Add noisy coordinates to packed atom states, attend locally, and pool to denoiser tokens.

    Args:
        atom_hidden_size: Width of the atom hidden and conditioning channels.
        token_hidden_size: Half the width of the produced denoiser token representation.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the conditioned Transformer encoder.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        token_hidden_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        transformer_size: int | None = None,
        post_layer_norm: bool = False,
        use_qk_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        transformer_size = 2 * token_hidden_size if transformer_size is None else transformer_size
        self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            ConditionedTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=atom_hidden_size,
                post_layer_norm=post_layer_norm,
                use_qk_norm=use_qk_norm,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.token_projection = nn.Linear(atom_hidden_size, transformer_size, bias=False)
        self.token_activation = nn.ReLU()

    def forward(
        self,
        atom_single: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        noisy_positions: Tensor,
        multiplicity: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        atom_query = atom_single.repeat_interleave(multiplicity, 0)
        atom_query = atom_query + self.noisy_position_projection(noisy_positions)
        conditioning = atom_conditioning.repeat_interleave(multiplicity, 0)
        atom_pair_bias = atom_pair_bias.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning,
            atom_pair_bias,
            atom_mask=atom_mask.bool(),
        )
        token_from_atom = self.token_activation(self.token_projection(atom_query))
        token_repr = dense_mean_to_tokens(
            token_from_atom,
            atom_to_token,
            num_tokens,
            atom_mask=atom_mask,
            count_eps=1e-6,
        )
        return token_repr, atom_query, conditioning


class PackedTokenToAtomDecoder(nn.Module):
    r"""
    Broadcast token states to packed atom queries and predict per-atom coordinate updates.

    Args:
        token_hidden_size: Half the width of the incoming denoiser token representation.
        atom_hidden_size: Width of the atom hidden and conditioning channels.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the conditioned Transformer encoder.
    """

    def __init__(
        self,
        token_hidden_size: int = 384,
        atom_hidden_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        transformer_size: int | None = None,
        use_qk_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        transformer_size = 2 * token_hidden_size if transformer_size is None else transformer_size
        self.token_projection = nn.Linear(transformer_size, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            ConditionedTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=atom_hidden_size,
                use_qk_norm=use_qk_norm,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.position_norm = LayerNorm(atom_hidden_size, backend=backend)
        self.position_update_projection = nn.Linear(atom_hidden_size, 3, bias=False)

    def forward(
        self,
        token_repr: Tensor,
        atom_skip: Tensor,
        conditioning_skip: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        token_index = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_pair_bias = atom_pair_bias.repeat_interleave(multiplicity, 0)
        token_projection = self.token_projection(token_repr)
        atom_query = atom_skip + token_projection.gather(
            1,
            token_index.unsqueeze(-1).expand(-1, -1, token_projection.shape[-1]),
        )
        mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning_skip,
            atom_pair_bias,
            atom_mask=mask.bool(),
        )
        return self.position_update_projection(self.position_norm(atom_query))


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

    def project_pair_bias(self, pair: Tensor) -> Tensor:
        r"""Project local atom pairs to packed per-block attention biases."""

        return torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections, strict=True)],
            dim=-1,
        )

    def forward_with_packed_bias(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        packed_bias: Tensor,
        atom_mask: Tensor | None = None,
        key_index: Tensor | None = None,
    ) -> Tensor:
        r"""Apply atom attention from an already-projected packed pair bias."""

        return self.atom_attention(
            atom_hidden,
            conditioning,
            packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
        )

    def forward(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        atom_mask: Tensor | None = None,
        key_index: Tensor | None = None,
    ) -> Tensor:
        return self.forward_with_packed_bias(
            atom_hidden,
            conditioning,
            self.project_pair_bias(pair),
            atom_mask=atom_mask,
            key_index=key_index,
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
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

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
        num_tokens: int,
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
        )
        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = self.atom_to_token_mean(
            token_hidden,
            atom_to_token,
            num_tokens,
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
