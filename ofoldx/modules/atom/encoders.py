r"""
Atom-to-token encoders and token-to-atom decoders.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..backend import LayerNorm
from ..transformer import (
    AtomCrossAttentionTransformerEncoder,
    ConditionedTransformerEncoder,
    NormalizedKeyAtomCrossAttentionTransformerEncoder,
)
from .context import AtomContext, AtomToTokenOutput
from .pair_encoding import encode_atom_pair
from .transfer import (
    AtomToTokenMean,
    TokenPairToAtomPairBroadcast,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
    dense_mean_to_tokens,
)
from .window import LocalAtomGeometry, LocalAtomGeometryBuilder, gather_key_windows

#: Supported atom-attention layouts.
AttentionLayout = Literal["dense", "local"]


class AtomToTokenEncoder(nn.Module):
    r"""
    Encode reference atom features and geometry into token hidden states.

    The input-embedding path of the atom module: it builds per-atom conditioning from the raw reference
    features, encodes the atom-pair representation (dense or local-window), runs the atom self-attention
    stack, and mask-means the result into tokens. See
    [`CoordinateAtomToTokenEncoder`][ofoldx.modules.atom.encoders.CoordinateAtomToTokenEncoder] for the
    diffusion variant that additionally folds in noisy coordinates and trunk context.

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
        atom_mask: Tensor | None = None,
        atom_to_token: Tensor,
        num_tokens: int,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> AtomToTokenOutput:
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        atom_pair = self.atom_pair_encoder(
            atom_conditioning,
            reference_positions=reference_positions,
            local_geometry=local_geometry,
        )
        attention_layout = "dense" if local_geometry is None else "local"
        atom_query = self.atom_self_attention(
            atom_conditioning,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            key_index=None if local_geometry is None else local_geometry.window_info.key_index,
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
            atom_context=AtomContext(pair=atom_pair, conditioning=atom_conditioning),
        )


class CoordinateAtomToTokenEncoder(nn.Module):
    r"""
    Encode noisy atom coordinates and trunk context into token hidden states (diffusion path).

    Broadcast trunk ``single`` augments the atom conditioning, broadcast trunk ``pair`` fuses into the
    atom-pair representation, and a projection of the noisy coordinates seeds the attention query.
    """

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
        self.token_pair_to_atom_pair_broadcast = TokenPairToAtomPairBroadcast()
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
        atom_mask: Tensor | None = None,
        atom_to_token: Tensor,
        num_tokens: int,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> AtomToTokenOutput:
        r"""
        Encode noisy atom coordinates plus trunk ``single``/``pair`` context into tokens.

        Extends the base encoding: reference conditioning is augmented by the broadcast trunk ``single``,
        the atom-pair representation fuses the broadcast trunk ``pair``, and the attention query adds a
        projection of ``noisy_coords``. ``local_geometry`` selects dense vs. local-window layout.

        Args:
            atom_features: `[..., N_atom, atom_feature_size]` raw reference atom features.
            reference_positions: `[..., N_atom, 3]` reference coordinates.
            reference_charges: `[..., N_atom, 1]` reference charges.
            reference_mask: `[..., N_atom]` validity mask zeroing conditioning for absent atoms.
            single: `[..., N_token, single_size]` trunk single representation, broadcast onto atoms.
            pair: `[..., N_token, N_token, pair_size]` trunk pair representation, fused into the atom pair.
            noisy_coords: `[..., N_atom, 3]` current noisy coordinates seeding the attention query.
            atom_mask: Optional `[..., N_atom]` attention/aggregation mask.
            atom_to_token: `[..., N_atom]` integer index mapping each atom to its token.
            num_tokens: Number of tokens to aggregate into.
            local_geometry: Optional precomputed local geometry; when given, selects local-window attention.

        Returns:
            An [`AtomToTokenOutput`][ofoldx.modules.atom.context.AtomToTokenOutput] with ``token_hidden``
            `[..., num_tokens, token_hidden_size]`, ``atom_query`` `[..., N_atom, atom_hidden_size]`, and the
            [`AtomContext`][ofoldx.modules.atom.context.AtomContext] for the coordinate decoder.
        """

        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        key_index = None if local_geometry is None else local_geometry.window_info.key_index
        atom_conditioning = self._add_single_conditioning(atom_conditioning, single, atom_to_token)
        atom_pair = self._build_atom_pair(
            atom_conditioning,
            pair=pair,
            atom_to_token=atom_to_token,
            reference_positions=reference_positions,
            local_geometry=local_geometry,
        )
        atom_query = atom_conditioning + self.noisy_position_projection(noisy_coords)
        attention_layout = "dense" if local_geometry is None else "local"
        atom_query = self.atom_self_attention(
            atom_query,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            key_index=key_index,
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
            atom_context=AtomContext(pair=atom_pair, conditioning=atom_conditioning),
        )

    def _add_single_conditioning(self, atom_conditioning: Tensor, single: Tensor, atom_to_token: Tensor) -> Tensor:
        r"""
        Broadcast the projected trunk ``single`` onto atoms and add it to the atom conditioning.

        Args:
            atom_conditioning: `[..., N_atom, atom_hidden_size]` per-atom conditioning.
            single: `[..., N_token, single_size]` trunk single representation.
            atom_to_token: `[..., N_atom]` atom-to-token index used to broadcast token states to atoms.

        Returns:
            Updated conditioning `[..., N_atom, atom_hidden_size]` (with an extra sample axis when ``single``
            carries one that the conditioning does not).
        """
        single_conditioning = self.single_to_atom_projection(self.single_norm(single))
        single_conditioning = self.token_to_atom_broadcast(single_conditioning, atom_to_token)
        if single_conditioning.dim() == atom_conditioning.dim() + 1:
            return atom_conditioning.unsqueeze(-3) + single_conditioning
        return atom_conditioning + single_conditioning

    def _build_atom_pair(
        self,
        atom_conditioning: Tensor,
        *,
        pair: Tensor,
        atom_to_token: Tensor,
        reference_positions: Tensor,
        local_geometry: LocalAtomGeometry | None,
    ) -> Tensor:
        r"""
        Fuse the projected trunk ``pair`` into the atom-pair representation (dense or local layout).

        Args:
            atom_conditioning: `[..., N_atom, atom_hidden_size]` per-atom conditioning.
            pair: `[..., N_token, N_token, pair_size]` trunk pair representation.
            atom_to_token: `[..., N_atom]` atom-to-token index used to broadcast token pairs to atom pairs.
            reference_positions: `[..., N_atom, 3]` reference coordinates (used for the dense layout).
            local_geometry: Precomputed local geometry, or ``None`` for the dense all-pairs layout.

        Returns:
            The atom-pair representation: dense `[..., N_atom, N_atom, atom_pair_hidden_size]`, or local
            `[..., N_window, num_query_atoms, num_key_atoms, atom_pair_hidden_size]`.
        """
        pair = self.pair_projection(self.pair_norm(pair))
        if local_geometry is None:
            pair = self.token_pair_to_atom_pair_broadcast(pair, atom_to_token)
            return self.atom_pair_encoder(
                atom_conditioning,
                reference_positions=reference_positions,
                pair=pair,
            )
        local_pair = self.token_pair_to_local_atom_pair_broadcast(
            pair,
            atom_to_token,
            key_index=local_geometry.window_info.key_index,
        )
        return self.atom_pair_encoder(atom_conditioning, local_geometry=local_geometry, pair=local_pair)


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
    Build dense or local-window atom-pair representations from atom geometry and conditioning.

    Combines distance-vector / inverse-distance / mask projections with left/right projections of the atom
    conditioning through a small feed-forward stack (via
    [`encode_atom_pair`][ofoldx.modules.atom.pair_encoding.encode_atom_pair]). The dense path forms the
    full all-pairs grid; the local path consumes precomputed windowed geometry.

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
        self.atom_pair_hidden_size = atom_pair_hidden_size
        self.local_geometry_builder = LocalAtomGeometryBuilder(
            num_query_atoms=num_query_atoms, num_key_atoms=num_key_atoms
        )
        self.activation = nn.ReLU()
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.left_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.right_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.feed_forward_input_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_hidden_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_output_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )

    def feed_forward(self, pair: Tensor) -> Tensor:
        r"""
        Apply the three projected ReLU stages of the atom-pair residual update.

        Args:
            pair: Atom-pair representation ending in ``atom_pair_hidden_size``.

        Returns:
            The residual update with the same shape as ``pair``.
        """

        pair = self.feed_forward_input_projection(self.activation(pair))
        pair = self.feed_forward_hidden_projection(self.activation(pair))
        return self.feed_forward_output_projection(self.activation(pair))

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
            A [`LocalAtomGeometry`][ofoldx.modules.atom.window.LocalAtomGeometry] to pass to
            [`encode_local`][ofoldx.modules.atom.encoders.AtomPairEncoder.encode_local].
        """

        return self.local_geometry_builder(reference_positions, reference_space_id, atom_mask)

    def forward(
        self,
        atom_conditioning: Tensor,
        *,
        reference_positions: Tensor | None = None,
        local_geometry: LocalAtomGeometry | None = None,
        pair: Tensor | None = None,
    ) -> Tensor:
        r"""
        Encode an atom-pair representation, dispatching to the dense or local layout.

        Uses the local layout when ``local_geometry`` is provided, otherwise the dense all-pairs layout
        (which requires ``reference_positions``).

        Args:
            atom_conditioning: `[..., N_atom, atom_hidden_size]` per-atom conditioning.
            reference_positions: `[..., N_atom, 3]` reference coordinates; required for the dense layout.
            local_geometry: Precomputed local geometry selecting the local-window layout.
            pair: Optional pair bias to fold in, matching the chosen layout's shape.

        Returns:
            The atom-pair representation: dense `[..., N_atom, N_atom, atom_pair_hidden_size]`, or local
            `[..., N_window, num_query_atoms, num_key_atoms, atom_pair_hidden_size]`.

        Raises:
            ValueError: If ``local_geometry`` is ``None`` and ``reference_positions`` is also ``None``.
        """

        if local_geometry is None:
            if reference_positions is None:
                raise ValueError("reference_positions is required for dense atom-pair encoding")
            return self.encode_dense(atom_conditioning, reference_positions=reference_positions, pair=pair)
        return self.encode_local(atom_conditioning, local_geometry=local_geometry, pair=pair)

    def encode_dense(
        self,
        atom_conditioning: Tensor,
        *,
        reference_positions: Tensor,
        pair: Tensor | None = None,
    ) -> Tensor:
        r"""
        Encode the dense all-pairs atom representation from reference coordinates.

        Forms the full `query - key` distance-vector grid `[..., N_atom, N_atom, 3]` and combines it (via
        [`encode_atom_pair`][ofoldx.modules.atom.pair_encoding.encode_atom_pair]) with the left/right
        relu'd ``atom_conditioning`` and an optional ``pair`` bias. Every position is treated as valid.

        Args:
            atom_conditioning: `[..., N_atom, atom_hidden_size]` per-atom conditioning.
            reference_positions: `[..., N_atom, 3]` reference coordinates.
            pair: Optional `[..., N_atom, N_atom, atom_pair_hidden_size]` pair bias.

        Returns:
            The dense pair representation of shape `[..., N_atom, N_atom, atom_pair_hidden_size]`.
        """

        distance_vectors = reference_positions.unsqueeze(-2) - reference_positions.unsqueeze(-3)
        relu_single = self.activation(atom_conditioning)
        valid = atom_conditioning.new_ones((*distance_vectors.shape[:-1], 1))
        return encode_atom_pair(
            self,
            distance_vectors,
            relu_single.unsqueeze(-2),
            relu_single.unsqueeze(-3),
            distance_term_mask=1.0,
            inverse_term_mask=1.0,
            valid_mask=valid,
            pair_bias=pair,
            prefix_ndim=atom_conditioning.dim() - 2,
        )

    def encode_local(
        self,
        atom_conditioning: Tensor,
        *,
        local_geometry: LocalAtomGeometry,
        pair: Tensor | None = None,
    ) -> Tensor:
        r"""
        Encode the windowed local atom-pair representation from precomputed geometry.

        Uses the geometry's per-window `query - key` distance vectors and validity/window masks, windows the
        ``atom_conditioning`` into query/key panes, and combines them (via
        [`encode_atom_pair`][ofoldx.modules.atom.pair_encoding.encode_atom_pair]) with an optional local
        ``pair`` bias.

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
        distance_mask = local_geometry.distance_mask.to(dtype=distance_vectors.dtype)
        window_mask = local_geometry.window_info.attention_mask.to(
            device=distance_vectors.device, dtype=distance_vectors.dtype
        ).unsqueeze(-1)
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
        return encode_atom_pair(
            self,
            distance_vectors,
            self.activation(conditioning_queries[..., None, :]),
            self.activation(conditioning_keys[..., None, :, :]),
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
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.left_atom_projection = nn.Linear(
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.right_atom_projection = nn.Linear(
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_input_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_hidden_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_output_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )

    def feed_forward(self, pair: Tensor) -> Tensor:
        pair = self.feed_forward_input_projection(F.relu(pair))
        pair = self.feed_forward_hidden_projection(F.relu(pair))
        return self.feed_forward_output_projection(F.relu(pair))

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
        atom_pair = encode_atom_pair(
            self,
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
    Apply a concrete atom transformer over an already-packed per-block pair bias.

    ``packed_bias`` stores all block/head pair-bias projections on the last axis, so the underlying
    [`ConditionedTransformerEncoder`][ofoldx.modules.transformer.ConditionedTransformerEncoder]
    can slice its per-block
    bias without re-projecting. Supports the dense atom layout (global attention over all atoms) and the
    local query/key window layout used by atom-level attention blocks.

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
        attention_layout: AttentionLayout = "dense",
        key_index: Tensor | None = None,
    ) -> Tensor:
        r"""
        Apply atom attention in the dense or local-window layout.

        Flattens arbitrary leading (batch[, sample]) dims into one batch axis, runs the packed-bias
        transformer, and reshapes back. In ``"local"`` layout the atom axis is padded up to a whole number
        of query windows, split into windows, attended with a windowed key gather, then unpadded.

        Args:
            atom_hidden: `[..., N_atom, atom_hidden_size]` atom query states.
            conditioning: `[..., N_atom, conditioning_size]` per-atom conditioning.
            packed_bias: Packed per-block/head bias whose last axis is ``num_blocks * num_attention_heads``;
                dense `[..., N_atom, N_atom, depth * heads]` or local
                `[..., N_window, num_query_atoms, num_key_atoms, depth * heads]`.
            atom_mask: Optional `[..., N_atom]` attention mask; defaults to all-valid.
            attention_layout: Either ``"dense"`` (global) or ``"local"`` (windowed).
            key_index: Optional row-specific atom indices for the local key panes.

        Returns:
            Updated atom hidden states of shape `[..., N_atom, atom_hidden_size]`.

        Raises:
            ValueError: If ``attention_layout`` is not ``"dense"`` or ``"local"``.
        """

        if attention_layout not in ("dense", "local"):
            raise ValueError(f"unsupported attention_layout: {attention_layout!r}")
        prefix = tuple(atom_hidden.shape[:-2])
        num_atoms = atom_hidden.shape[-2]
        if atom_mask is None:
            atom_mask = atom_hidden.new_ones(*prefix, num_atoms)

        query = _flatten_batch(atom_hidden, prefix, 2)
        conditioning = _flatten_batch(conditioning, prefix, 2)
        bias = _flatten_batch(packed_bias, prefix, 4 if attention_layout == "local" else 3)
        mask = _flatten_batch(atom_mask, prefix, 1).float()
        if key_index is not None:
            key_index = _flatten_batch(key_index, prefix, 2)

        if attention_layout == "dense":
            output = self.transformer(query, conditioning, bias, mask, to_keys=None)
            return output.reshape(*prefix, num_atoms, output.shape[-1])

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
            attention_layout="local",
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
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning,
            atom_pair_bias,
            atom_mask=atom_mask.bool(),
            attention_layout="local",
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
            attention_layout="local",
        )
        return self.position_update_projection(self.position_norm(atom_query))


class AtomSelfAttention(nn.Module):
    r"""
    Atom self-attention with a separately projected pair bias for each repeated block.

    Supports both a ``"dense"`` layout (global attention over all atoms) and a ``"local"`` layout (windowed
    local attention), selected per forward. The per-block pair-bias projection lives here; the packed attention
    execution is shared through [`PackedAtomAttention`][ofoldx.modules.atom.encoders.PackedAtomAttention].

    Args:
        atom_hidden_size: Width of the atom hidden channel.
        conditioning_size: Width of the per-atom conditioning channel.
        pair_size: Width of the incoming pair representation to project into per-head bias logits.
        num_blocks: Number of atom self-attention blocks.
        num_attention_heads: Number of attention heads per block.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        backend: Backend selector for the underlying attention/transition implementations.
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
        # The pair is projected to per-head attention logits independently in each block.
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

    def forward(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        atom_mask: Tensor | None = None,
        attention_layout: AttentionLayout = "dense",
        key_index: Tensor | None = None,
    ) -> Tensor:
        packed_bias = torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections)],
            dim=-1,
        )
        return self.atom_attention(
            atom_hidden,
            conditioning,
            packed_bias,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            key_index=key_index,
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

    def forward(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        atom_mask: Tensor | None = None,
        attention_layout: AttentionLayout = "dense",
        key_index: Tensor | None = None,
    ) -> Tensor:
        packed_bias = torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections)],
            dim=-1,
        )
        return self.atom_attention(
            atom_hidden,
            conditioning,
            packed_bias,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
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
        atom_mask: Tensor | None = None,
        atom_to_token: Tensor,
        num_tokens: int,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> AtomToTokenOutput:
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        atom_pair = self.atom_pair_encoder(
            atom_conditioning,
            reference_positions=reference_positions,
            local_geometry=local_geometry,
        )
        attention_layout = "dense" if local_geometry is None else "local"
        atom_query = self.atom_self_attention(
            atom_conditioning,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            key_index=None if local_geometry is None else local_geometry.window_info.key_index,
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
            atom_context=AtomContext(pair=atom_pair, conditioning=atom_conditioning),
        )


class NormalizedKeyCoordinateAtomToTokenEncoder(nn.Module):
    r"""
    Coordinate atom encoder with query-normalized gathered keys.
    """

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
        self.token_pair_to_atom_pair_broadcast = TokenPairToAtomPairBroadcast()
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
        single: Tensor,
        pair: Tensor,
        noisy_coords: Tensor,
        atom_mask: Tensor | None = None,
        atom_to_token: Tensor,
        num_tokens: int,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> AtomToTokenOutput:
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        key_index = None if local_geometry is None else local_geometry.window_info.key_index
        atom_conditioning = self._add_single_conditioning(atom_conditioning, single, atom_to_token)
        atom_pair = self._build_atom_pair(
            atom_conditioning,
            pair=pair,
            atom_to_token=atom_to_token,
            reference_positions=reference_positions,
            local_geometry=local_geometry,
        )
        atom_query = atom_conditioning + self.noisy_position_projection(noisy_coords)
        attention_layout = "dense" if local_geometry is None else "local"
        atom_query = self.atom_self_attention(
            atom_query,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            key_index=key_index,
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
            atom_context=AtomContext(pair=atom_pair, conditioning=atom_conditioning),
        )

    def _add_single_conditioning(self, atom_conditioning: Tensor, single: Tensor, atom_to_token: Tensor) -> Tensor:
        single_conditioning = self.single_to_atom_projection(self.single_norm(single))
        single_conditioning = self.token_to_atom_broadcast(single_conditioning, atom_to_token)
        if single_conditioning.dim() == atom_conditioning.dim() + 1:
            return atom_conditioning.unsqueeze(-3) + single_conditioning
        return atom_conditioning + single_conditioning

    def _build_atom_pair(
        self,
        atom_conditioning: Tensor,
        *,
        pair: Tensor,
        atom_to_token: Tensor,
        reference_positions: Tensor,
        local_geometry: LocalAtomGeometry | None,
    ) -> Tensor:
        pair = self.pair_projection(self.pair_norm(pair))
        if local_geometry is None:
            pair = self.token_pair_to_atom_pair_broadcast(pair, atom_to_token)
            return self.atom_pair_encoder(
                atom_conditioning,
                reference_positions=reference_positions,
                pair=pair,
            )
        local_pair = self.token_pair_to_local_atom_pair_broadcast(
            pair,
            atom_to_token,
            key_index=local_geometry.window_info.key_index,
        )
        return self.atom_pair_encoder(atom_conditioning, local_geometry=local_geometry, pair=local_pair)


class TokenToAtomDecoder(nn.Module):
    r"""
    Decode token hidden states into per-atom coordinate updates.

    The output side of the atom module: it broadcasts the token states back onto the encoder's atom query,
    reuses the encoder's precomputed pair/conditioning through a further atom self-attention pass, and
    projects to a 3-vector coordinate delta per atom.

    Args:
        token_hidden_size: Width of the incoming token hidden states.
        atom_hidden_size: Width of the atom hidden channel.
        atom_pair_hidden_size: Width of the atom-pair representation feeding the attention bias.
        num_blocks: Number of atom self-attention blocks.
        num_attention_heads: Number of attention heads per block.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        backend: Backend selector passed down to atom self-attention.
    """

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

    def forward(
        self,
        token_hidden: Tensor,
        *,
        atom_to_token: Tensor,
        atom_query: Tensor,
        atom_context: AtomContext,
        atom_mask: Tensor | None = None,
        attention_layout: AttentionLayout = "dense",
        key_index: Tensor | None = None,
    ) -> Tensor:
        atom_query = atom_query + self.token_to_atom_broadcast(self.token_projection(token_hidden), atom_to_token)
        atom_query = self.atom_self_attention(
            atom_query,
            conditioning=atom_context.conditioning,
            pair=atom_context.pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            key_index=key_index,
        )
        return self.output_projection(self.query_norm(atom_query))


class NormalizedKeyTokenToAtomDecoder(nn.Module):
    r"""
    Token-to-atom decoder with query-normalized gathered keys.
    """

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
        self.query_norm = LayerNorm(atom_hidden_size, bias=False, backend=backend)
        self.output_projection = nn.Linear(atom_hidden_size, 3, bias=False)

    def forward(
        self,
        token_hidden: Tensor,
        *,
        atom_to_token: Tensor,
        atom_query: Tensor,
        atom_context: AtomContext,
        atom_mask: Tensor | None = None,
        attention_layout: AttentionLayout = "dense",
        key_index: Tensor | None = None,
    ) -> Tensor:
        atom_query = atom_query + self.token_to_atom_broadcast(self.token_projection(token_hidden), atom_to_token)
        atom_query = self.atom_self_attention(
            atom_query,
            conditioning=atom_context.conditioning,
            pair=atom_context.pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            key_index=key_index,
        )
        return self.output_projection(self.query_norm(atom_query))


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
