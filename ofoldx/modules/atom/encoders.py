r"""
Atom-to-token encoders and the token-to-atom decoder.

The atom encoder/decoder web contains the conditioning builder, the dense/local atom-pair encoder, the windowed
self-attention stack, and the encoders/decoder that move atom states to/from token states. Each piece delegates the
heavy attention to the shared [`PackedAtomAttention`][ofoldx.modules.atom.encoders.PackedAtomAttention] and the
shared pair arithmetic to [`encode_atom_pair`][ofoldx.modules.atom.pair_encoding.encode_atom_pair].
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..backend import LayerNorm, resolve_config_operator_backend
from ..transformer import ConditionedTransformer, CrossAttentionKeySource, TransitionKind
from .context import AtomContext, AtomToTokenOutput
from .pair_encoding import encode_atom_pair
from .transfer import (
    TokenPairToAtomPairBroadcast,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
    scatter_mean_to_tokens,
)
from .window import LocalAtomGeometry, LocalAtomGeometryBuilder, gather_key_windows

AttentionLayout = Literal["dense", "local"]


class AtomToTokenEncoderBase(nn.Module):
    r"""
    Shared machinery for the atom-to-token encoders (reference and coordinate paths).

    Holds the atom conditioner, the atom-pair encoder, the atom self-attention stack, and the token
    projection, and exposes the pieces of the encode-then-aggregate pipeline that both concrete encoders
    reuse: [`build_local_geometry`][ofoldx.modules.atom.encoders.AtomToTokenEncoderBase.build_local_geometry]
    to precompute the windowed geometry and
    [`aggregate_to_tokens`][ofoldx.modules.atom.encoders.AtomToTokenEncoderBase.aggregate_to_tokens] to
    reduce per-atom states down to token states. Subclasses supply the ``forward`` that wires them together.

    Args:
        token_hidden_size: Width of the produced token hidden states.
        atom_hidden_size: Width of the per-atom conditioning / query channel.
        atom_pair_hidden_size: Width of the atom-pair representation feeding the attention bias.
        num_blocks: Number of atom self-attention blocks.
        num_attention_heads: Number of attention heads per block.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        atom_feature_size: Width of the raw reference atom-feature vector.
        operator_backend: Backend selector passed down to the attention stack (``"auto"`` picks the
            OOps-owned implementation when available, else the public implementation).

    Examples:
        >>> import torch
        >>> encoder = AtomToTokenEncoder(
        ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
        ...     num_blocks=1, num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
        ...     atom_feature_size=6,
        ... )
        >>> isinstance(encoder, AtomToTokenEncoderBase)
        True
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
        operator_backend: str = "auto",
        cross_attention_key_source: CrossAttentionKeySource = "raw",
    ) -> None:
        super().__init__()
        self.operator_backend = resolve_config_operator_backend(
            operator_backend,
            supported=("oops", "eager"),
            public_default="eager",
        )
        self.atom_conditioner = AtomConditioner(atom_feature_size=atom_feature_size, atom_hidden_size=atom_hidden_size)
        self.atom_pair_encoder = AtomPairEncoder(
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_attention_stack = AtomSelfAttentionStack(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=self.operator_backend,
            cross_attention_key_source=cross_attention_key_source,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.relu = nn.ReLU()

    def build_local_geometry(
        self, reference_positions: Tensor, reference_space_id: Tensor, num_valid_atoms: int | None = None
    ) -> LocalAtomGeometry:
        r"""
        Precompute the windowed atom-pair geometry for local-layout attention.

        Delegates to the atom-pair encoder's window builder so the geometry uses this encoder's
        ``num_query_atoms``/``num_key_atoms`` layout. Pass the result as ``local_geometry`` to a subclass
        ``forward`` to select local-window attention instead of dense all-pairs attention.

        Args:
            reference_positions: `[..., N_atom, 3]` reference coordinates.
            reference_space_id: `[..., N_atom]` reference-space ids grouping atoms that may interact.
            num_valid_atoms: Number of real atoms; enables the DeepMind AF3 clamped/shifted key window
                (see [`gather_key_windows`][ofoldx.modules.atom.window.gather_key_windows]). Left unset by the
                non-diffusion input-embedder path, which keeps the fixed zero-padded window.

        Returns:
            A [`LocalAtomGeometry`][ofoldx.modules.atom.window.LocalAtomGeometry] carrying per-window
            `query - key` displacement vectors, the same-space validity mask, and the window padding metadata.

        Examples:
            >>> import torch
            >>> encoder = AtomToTokenEncoder(
            ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
            ...     num_blocks=1, num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
            ...     atom_feature_size=6,
            ... )
            >>> reference_positions = torch.zeros(1, 4, 3)
            >>> reference_space_id = torch.zeros(1, 4, dtype=torch.long)
            >>> geometry = encoder.build_local_geometry(reference_positions, reference_space_id)
            >>> geometry.distance_vectors.shape
            torch.Size([1, 2, 2, 4, 3])
        """

        return self.atom_pair_encoder.build_local_geometry(reference_positions, reference_space_id, num_valid_atoms)

    def aggregate_to_tokens(
        self,
        atom_query: Tensor,
        *,
        atom_to_token: Tensor,
        num_tokens: int,
        atom_mask: Tensor | None,
        atom_context: AtomContext,
    ) -> AtomToTokenOutput:
        r"""
        Project the atom query to token width and mask-mean atoms into their tokens.

        Applies the ReLU'd token projection to ``atom_query`` and averages the resulting per-atom vectors
        into ``num_tokens`` tokens according to ``atom_to_token`` (respecting ``atom_mask`` if given), then
        packages the token states together with the untouched ``atom_query`` and ``atom_context`` for the
        downstream coordinate decoder.

        Args:
            atom_query: `[..., N_atom, atom_hidden_size]` per-atom query states from the attention stack.
            atom_to_token: `[..., N_atom]` integer index mapping each atom to its token.
            num_tokens: Number of tokens to scatter into (size of the output token axis).
            atom_mask: Optional `[..., N_atom]` validity mask; absent atoms are excluded from the mean.
            atom_context: Precomputed pair/conditioning tensors to carry into the output.

        Returns:
            An [`AtomToTokenOutput`][ofoldx.modules.atom.context.AtomToTokenOutput] whose
            ``token_hidden`` has shape `[..., num_tokens, token_hidden_size]`.

        Examples:
            >>> import torch
            >>> from ofoldx.modules.atom.context import AtomContext
            >>> encoder = AtomToTokenEncoder(
            ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
            ...     num_blocks=1, num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
            ...     atom_feature_size=6,
            ... )
            >>> atom_query = torch.zeros(1, 4, 8)
            >>> atom_to_token = torch.tensor([[0, 0, 1, 1]])
            >>> context = AtomContext(pair=torch.zeros(1, 4, 4, 4), conditioning=torch.zeros(1, 4, 8))
            >>> output = encoder.aggregate_to_tokens(
            ...     atom_query, atom_to_token=atom_to_token, num_tokens=2, atom_mask=None,
            ...     atom_context=context,
            ... )
            >>> output.token_hidden.shape
            torch.Size([1, 2, 8])
        """

        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = scatter_mean_to_tokens(
            token_hidden,
            atom_to_token,
            num_tokens,
            atom_mask,
            operator_backend=self.operator_backend,
        )
        return AtomToTokenOutput(token_hidden=token_hidden, atom_query=atom_query, atom_context=atom_context)


class AtomToTokenEncoder(AtomToTokenEncoderBase):
    r"""
    Encode reference atom features and geometry into token hidden states.

    The input-embedding path of the atom module: it builds per-atom conditioning from the raw reference
    features, encodes the atom-pair representation (dense or local-window), runs the atom self-attention
    stack, and mask-means the result into tokens. See
    [`CoordinateAtomToTokenEncoder`][ofoldx.modules.atom.encoders.CoordinateAtomToTokenEncoder] for the
    diffusion variant that additionally folds in noisy coordinates and trunk context.

    Constructor arguments are inherited from
    [`AtomToTokenEncoderBase`][ofoldx.modules.atom.encoders.AtomToTokenEncoderBase].

    Examples:
        >>> import torch
        >>> encoder = AtomToTokenEncoder(
        ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
        ...     num_blocks=1, num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
        ...     atom_feature_size=6,
        ... )
        >>> sum(p.numel() for p in encoder.parameters()) > 0
        True
    """

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
        r"""
        Encode reference atom features plus geometry into token hidden states.

        Builds per-atom conditioning, encodes the atom-pair representation (dense when ``local_geometry`` is
        ``None``, else local-window), runs the self-attention stack, then projects and mask-means atoms into
        ``num_tokens`` tokens.

        Args:
            atom_features: `[..., N_atom, atom_feature_size]` raw reference atom features.
            reference_positions: `[..., N_atom, 3]` reference coordinates.
            reference_charges: `[..., N_atom, 1]` reference charges.
            reference_mask: `[..., N_atom]` validity mask zeroing conditioning for absent atoms.
            atom_mask: Optional `[..., N_atom]` attention/aggregation mask.
            atom_to_token: `[..., N_atom]` integer index mapping each atom to its token.
            num_tokens: Number of tokens to aggregate into.
            local_geometry: Optional precomputed local geometry; when given, selects local-window attention.

        Returns:
            An [`AtomToTokenOutput`][ofoldx.modules.atom.context.AtomToTokenOutput] with ``token_hidden``
            `[..., num_tokens, token_hidden_size]`, ``atom_query`` `[..., N_atom, atom_hidden_size]`, and the
            [`AtomContext`][ofoldx.modules.atom.context.AtomContext] (pair + conditioning) for the decoder.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> encoder = AtomToTokenEncoder(
            ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
            ...     num_blocks=1, num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
            ...     atom_feature_size=6,
            ... )
            >>> output = encoder(
            ...     torch.randn(1, 4, 6),
            ...     reference_positions=torch.randn(1, 4, 3),
            ...     reference_charges=torch.randn(1, 4, 1),
            ...     reference_mask=torch.ones(1, 4),
            ...     atom_to_token=torch.tensor([[0, 0, 1, 1]]),
            ...     num_tokens=2,
            ... )
            >>> output.token_hidden.shape
            torch.Size([1, 2, 8])
            >>> output.atom_query.shape
            torch.Size([1, 4, 8])
        """

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
        atom_query = self.atom_attention_stack(
            atom_conditioning,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            num_valid_atoms=None if local_geometry is None else local_geometry.window_info.num_valid_atoms,
        )
        return self.aggregate_to_tokens(
            atom_query,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
            atom_mask=atom_mask,
            atom_context=AtomContext(pair=atom_pair, conditioning=atom_conditioning),
        )


class CoordinateAtomToTokenEncoder(AtomToTokenEncoderBase):
    r"""
    Encode noisy atom coordinates and trunk context into token hidden states (diffusion path).

    Extends [`AtomToTokenEncoderBase`][ofoldx.modules.atom.encoders.AtomToTokenEncoderBase] with the
    trunk-conditioning it needs inside the diffusion module: broadcast trunk ``single`` augments the atom
    conditioning, broadcast trunk ``pair`` fuses into the atom-pair representation, and a projection of the
    noisy coordinates seeds the attention query.

    Args:
        token_hidden_size: Width of the produced token hidden states.
        atom_hidden_size: Width of the per-atom conditioning / query channel.
        atom_pair_hidden_size: Width of the atom-pair representation feeding the attention bias.
        single_size: Width of the trunk single (per-token) representation.
        pair_size: Width of the trunk pair (per-token-pair) representation.
        num_blocks: Number of atom self-attention blocks.
        num_attention_heads: Number of attention heads per block.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        atom_feature_size: Width of the raw reference atom-feature vector.
        operator_backend: Backend selector passed down to the attention stack.

    Examples:
        >>> import torch
        >>> encoder = CoordinateAtomToTokenEncoder(
        ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
        ...     single_size=6, pair_size=5, num_blocks=1, num_attention_heads=2,
        ...     num_query_atoms=2, num_key_atoms=4, atom_feature_size=6,
        ... )
        >>> isinstance(encoder, AtomToTokenEncoderBase)
        True
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
        operator_backend: str = "auto",
        cross_attention_key_source: CrossAttentionKeySource = "raw",
    ) -> None:
        super().__init__(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            atom_feature_size=atom_feature_size,
            operator_backend=operator_backend,
            cross_attention_key_source=cross_attention_key_source,
        )
        self.token_to_atom_broadcast = TokenToAtomBroadcast()
        self.token_pair_to_atom_pair_broadcast = TokenPairToAtomPairBroadcast()
        self.token_pair_to_local_atom_pair_broadcast = TokenPairToLocalAtomPairBroadcast(
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.single_norm = LayerNorm(single_size, bias=False)
        self.single_to_atom_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.norm_pair = LayerNorm(pair_size, bias=False)
        self.pair_projection = nn.Linear(pair_size, atom_pair_hidden_size, bias=False)
        self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)

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

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> encoder = CoordinateAtomToTokenEncoder(
            ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
            ...     single_size=6, pair_size=5, num_blocks=1, num_attention_heads=2,
            ...     num_query_atoms=2, num_key_atoms=4, atom_feature_size=6,
            ... )
            >>> output = encoder(
            ...     torch.randn(1, 4, 6),
            ...     reference_positions=torch.randn(1, 4, 3),
            ...     reference_charges=torch.randn(1, 4, 1),
            ...     reference_mask=torch.ones(1, 4),
            ...     single=torch.randn(1, 2, 6),
            ...     pair=torch.randn(1, 2, 2, 5),
            ...     noisy_coords=torch.randn(1, 4, 3),
            ...     atom_to_token=torch.tensor([[0, 0, 1, 1]]),
            ...     num_tokens=2,
            ... )
            >>> output.token_hidden.shape
            torch.Size([1, 2, 8])
        """

        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        # The clamped/shifted key window (DeepMind AF3) is selected by the valid-atom count carried on the
        # precomputed geometry; the pair broadcast and the attention key gather must use the same count so
        # their key panes stay aligned. ``None`` (dense layout or fixed geometry) keeps the padded window.
        num_valid_atoms = None if local_geometry is None else local_geometry.window_info.num_valid_atoms
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
        atom_query = self.atom_attention_stack(
            atom_query,
            atom_conditioning,
            atom_pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            num_valid_atoms=num_valid_atoms,
        )
        return self.aggregate_to_tokens(
            atom_query=atom_query,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
            atom_mask=atom_mask,
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
        pair = self.pair_projection(self.norm_pair(pair))
        if local_geometry is None:
            pair = self.token_pair_to_atom_pair_broadcast(pair, atom_to_token)
            return self.atom_pair_encoder(
                atom_conditioning,
                reference_positions=reference_positions,
                pair=pair,
            )
        local_pair = self.token_pair_to_local_atom_pair_broadcast(
            pair, atom_to_token, num_valid_atoms=local_geometry.window_info.num_valid_atoms
        )
        return self.atom_pair_encoder(atom_conditioning, local_geometry=local_geometry, pair=local_pair)


class AtomConditioner(nn.Module):
    r"""
    Build per-atom conditioning states from reference position, charge, and feature inputs.

    Projects each reference input to the atom hidden width and sums them, giving the shared per-atom
    conditioning consumed by the atom-pair encoder and the attention stack.

    Args:
        atom_feature_size: Width of the raw reference atom-feature vector.
        atom_hidden_size: Width of the produced conditioning channel.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> conditioner = AtomConditioner(atom_feature_size=6, atom_hidden_size=8)
        >>> conditioning = conditioner(
        ...     torch.randn(2, 6),
        ...     reference_positions=torch.randn(2, 3),
        ...     reference_charges=torch.randn(2, 1),
        ...     reference_mask=torch.ones(2),
        ... )
        >>> conditioning.shape
        torch.Size([2, 8])
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

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> conditioner = AtomConditioner(atom_feature_size=6, atom_hidden_size=8)
            >>> reference_mask = torch.tensor([1.0, 0.0])
            >>> conditioning = conditioner(
            ...     torch.randn(2, 6),
            ...     reference_positions=torch.randn(2, 3),
            ...     reference_charges=torch.randn(2, 1),
            ...     reference_mask=reference_mask,
            ... )
            >>> conditioning[1].abs().sum().tolist()
            0.0
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

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> encoder = AtomPairEncoder(
        ...     atom_hidden_size=8, atom_pair_hidden_size=4, num_query_atoms=2, num_key_atoms=4,
        ... )
        >>> pair = encoder(torch.randn(3, 8), reference_positions=torch.randn(3, 3))
        >>> pair.shape
        torch.Size([3, 3, 4])
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
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.left_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.right_atom_projection = nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False)
        self.feed_forward = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
        )
        self.relu = nn.ReLU()

    def build_local_geometry(
        self, reference_positions: Tensor, reference_space_id: Tensor, num_valid_atoms: int | None = None
    ) -> LocalAtomGeometry:
        r"""
        Precompute the windowed atom-pair geometry for the local encoding layout.

        Args:
            reference_positions: `[..., N_atom, 3]` reference coordinates.
            reference_space_id: `[..., N_atom]` reference-space ids grouping atoms that may interact.
            num_valid_atoms: Number of real atoms; enables the DeepMind AF3 clamped/shifted key window.

        Returns:
            A [`LocalAtomGeometry`][ofoldx.modules.atom.window.LocalAtomGeometry] to pass to
            [`encode_local`][ofoldx.modules.atom.encoders.AtomPairEncoder.encode_local].

        Examples:
            >>> import torch
            >>> encoder = AtomPairEncoder(
            ...     atom_hidden_size=8, atom_pair_hidden_size=4, num_query_atoms=2, num_key_atoms=4,
            ... )
            >>> geometry = encoder.build_local_geometry(
            ...     torch.zeros(1, 4, 3), torch.zeros(1, 4, dtype=torch.long)
            ... )
            >>> geometry.distance_vectors.shape
            torch.Size([1, 2, 2, 4, 3])
        """

        return self.local_geometry_builder(reference_positions, reference_space_id, num_valid_atoms)

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

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> encoder = AtomPairEncoder(
            ...     atom_hidden_size=8, atom_pair_hidden_size=4, num_query_atoms=2, num_key_atoms=4,
            ... )
            >>> pair = encoder(torch.randn(3, 8), reference_positions=torch.randn(3, 3))
            >>> pair.shape
            torch.Size([3, 3, 4])
            >>> encoder(torch.randn(3, 8))
            Traceback (most recent call last):
                ...
            ValueError: reference_positions is required for dense atom-pair encoding
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

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> encoder = AtomPairEncoder(
            ...     atom_hidden_size=8, atom_pair_hidden_size=4, num_query_atoms=2, num_key_atoms=4,
            ... )
            >>> pair = encoder.encode_dense(torch.randn(3, 8), reference_positions=torch.randn(3, 3))
            >>> pair.shape
            torch.Size([3, 3, 4])
        """

        distance_vectors = reference_positions.unsqueeze(-2) - reference_positions.unsqueeze(-3)
        relu_single = self.relu(atom_conditioning)
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

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> encoder = AtomPairEncoder(
            ...     atom_hidden_size=8, atom_pair_hidden_size=4, num_query_atoms=2, num_key_atoms=4,
            ... )
            >>> geometry = encoder.build_local_geometry(
            ...     torch.zeros(1, 4, 3), torch.zeros(1, 4, dtype=torch.long)
            ... )
            >>> encoder.encode_local(torch.randn(1, 4, 8), local_geometry=geometry).shape
            torch.Size([1, 2, 2, 4, 4])
        """

        distance_vectors = local_geometry.distance_vectors
        distance_mask = local_geometry.distance_mask.to(dtype=distance_vectors.dtype)
        window_mask = local_geometry.window_info.attention_mask.to(
            device=distance_vectors.device, dtype=distance_vectors.dtype
        ).unsqueeze(-1)
        conditioning_windows = self.local_geometry_builder.windowing(
            atom_conditioning, local_geometry.window_info.num_valid_atoms
        )
        return encode_atom_pair(
            self,
            distance_vectors,
            self.relu(conditioning_windows.query[..., None, :]),
            self.relu(conditioning_windows.key[..., None, :, :]),
            distance_term_mask=distance_mask * window_mask,
            inverse_term_mask=distance_mask,
            valid_mask=distance_mask,
            pair_bias=pair,
            prefix_ndim=atom_conditioning.dim() - 2,
        )


class PackedAtomAttention(nn.Module):
    r"""
    Run atom attention over an already-packed per-layer pair bias.

    ``packed_bias`` stores all block/head pair-bias projections on the last axis, so the underlying
    [`ConditionedTransformer`][ofoldx.modules.transformer.ConditionedTransformer] can slice its per-block
    bias without re-projecting. Supports the dense atom layout (global attention over all atoms) and the
    local query/key window layout used by atom-level attention blocks.

    Args:
        atom_hidden_size: Width of the atom hidden channel.
        conditioning_size: Width of the per-atom conditioning channel.
        num_blocks: Number of transformer blocks.
        num_attention_heads: Number of attention heads per block.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        operator_backend: Backend selector for the transformer's attention/transition implementations.
        cross_attention_mode: Whether the transformer runs in cross-attention mode (query/key panes differ).
        transition: Transition-block kind used inside each transformer block.

    Examples:
        >>> import torch
        >>> attention = PackedAtomAttention(
        ...     atom_hidden_size=8, conditioning_size=8, num_blocks=1, num_attention_heads=2,
        ...     num_query_atoms=2, num_key_atoms=4,
        ... )
        >>> attention.num_key_atoms
        4
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        conditioning_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        operator_backend: str = "auto",
        cross_attention_mode: bool = True,
        cross_attention_key_source: CrossAttentionKeySource = "raw",
        transition: TransitionKind = "double",
    ) -> None:
        super().__init__()
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.transformer = ConditionedTransformer(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            hidden_size=atom_hidden_size,
            conditioning_size=conditioning_size,
            operator_backend=operator_backend,
            cross_attention_mode=cross_attention_mode,
            cross_attention_key_source=cross_attention_key_source,
            transition=transition,
        )

    def forward(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        packed_bias: Tensor,
        *,
        atom_mask: Tensor | None = None,
        attention_layout: AttentionLayout = "dense",
        num_valid_atoms: int | None = None,
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
            num_valid_atoms: Number of real atoms; enables the DeepMind AF3 clamped/shifted key gather so the
                windowed key panes match the pair-bias windows. Unset keeps the fixed zero-padded gather.

        Returns:
            Updated atom hidden states of shape `[..., N_atom, atom_hidden_size]`.

        Raises:
            ValueError: If ``attention_layout`` is not ``"dense"`` or ``"local"``.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> attention = PackedAtomAttention(
            ...     atom_hidden_size=8, conditioning_size=8, num_blocks=1, num_attention_heads=2,
            ...     num_query_atoms=2, num_key_atoms=4,
            ... )
            >>> out = attention(
            ...     torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 4, 2),
            ...     attention_layout="dense",
            ... )
            >>> out.shape
            torch.Size([1, 4, 8])
            >>> attention(torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 4, 2), attention_layout="x")
            Traceback (most recent call last):
                ...
            ValueError: unsupported attention_layout: 'x'
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
            num_valid_atoms=num_valid_atoms,
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


class AtomSelfAttentionStack(nn.Module):
    r"""
    Stack of atom self-attention blocks over a projected pair bias.

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
        operator_backend: Backend selector for the underlying attention/transition implementations.

    Examples:
        >>> import torch
        >>> stack = AtomSelfAttentionStack(
        ...     atom_hidden_size=8, conditioning_size=8, pair_size=4, num_blocks=1,
        ...     num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
        ... )
        >>> len(stack.bias_projections)
        1
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
        operator_backend: str = "auto",
        cross_attention_key_source: CrossAttentionKeySource = "raw",
    ) -> None:
        super().__init__()
        # The pair is projected to per-head attention logits independently in each block.
        self.pair_norms = nn.ModuleList(LayerNorm(pair_size, bias=False) for _ in range(num_blocks))
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )
        self.atom_attention = PackedAtomAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=conditioning_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_mode=True,
            cross_attention_key_source=cross_attention_key_source,
            transition="double",
        )

    def forward(
        self,
        atom_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        atom_mask: Tensor | None = None,
        attention_layout: AttentionLayout = "dense",
        num_valid_atoms: int | None = None,
    ) -> Tensor:
        r"""
        Project the ``pair`` to a packed per-block bias and run the atom self-attention stack.

        Args:
            atom_hidden: `[..., N_atom, atom_hidden_size]` atom query states driving the blocks.
            conditioning: `[..., N_atom, conditioning_size]` per-atom conditioning.
            pair: Dense `[..., N_atom, N_atom, pair_size]` or local-window pair representation, projected
                per block into per-head bias logits.
            atom_mask: Optional `[..., N_atom]` attention mask.
            attention_layout: Either ``"dense"`` or ``"local"`` execution.
            num_valid_atoms: Number of real atoms; enables the DeepMind AF3 clamped/shifted key gather.

        Returns:
            Updated atom hidden states of shape `[..., N_atom, atom_hidden_size]`.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> stack = AtomSelfAttentionStack(
            ...     atom_hidden_size=8, conditioning_size=8, pair_size=4, num_blocks=1,
            ...     num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
            ... )
            >>> out = stack(
            ...     torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 4, 4),
            ...     attention_layout="dense",
            ... )
            >>> out.shape
            torch.Size([1, 4, 8])
        """

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
            num_valid_atoms=num_valid_atoms,
        )


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
        operator_backend: Backend selector passed down to the attention stack.

    Examples:
        >>> import torch
        >>> decoder = TokenToAtomDecoder(
        ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
        ...     num_blocks=1, num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
        ... )
        >>> isinstance(decoder, TokenToAtomDecoder)
        True
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
        operator_backend: str = "auto",
        cross_attention_key_source: CrossAttentionKeySource = "raw",
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(token_hidden_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast()
        self.atom_attention_stack = AtomSelfAttentionStack(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_key_source=cross_attention_key_source,
        )
        self.query_norm = LayerNorm(atom_hidden_size, bias=False)
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
        num_valid_atoms: int | None = None,
    ) -> Tensor:
        r"""
        Decode token hidden states into per-atom coordinate updates.

        Projects ``token_hidden`` to atom width, broadcasts it onto the encoder ``atom_query`` via
        ``atom_to_token``, runs the atom self-attention stack (reusing ``atom_context``'s pair/conditioning,
        dense or local per ``attention_layout``), and projects to a coordinate delta per atom.

        Args:
            token_hidden: `[..., N_token, token_hidden_size]` token hidden states to decode.
            atom_to_token: `[..., N_atom]` integer index mapping each atom to its token.
            atom_query: `[..., N_atom, atom_hidden_size]` encoder atom query states to update.
            atom_context: Precomputed [`AtomContext`][ofoldx.modules.atom.context.AtomContext] with the
                pair representation and conditioning to reuse.
            atom_mask: Optional `[..., N_atom]` attention mask.
            attention_layout: Either ``"dense"`` or ``"local"`` execution.
            num_valid_atoms: Number of real atoms; enables the DeepMind AF3 clamped/shifted key gather so the
                decoder attends over the same key panes the encoder built the reused pair bias for.

        Returns:
            Per-atom coordinate updates of shape `[..., N_atom, 3]`.

        Examples:
            >>> import torch
            >>> from ofoldx.modules.atom.context import AtomContext
            >>> _ = torch.manual_seed(0)
            >>> decoder = TokenToAtomDecoder(
            ...     token_hidden_size=8, atom_hidden_size=8, atom_pair_hidden_size=4,
            ...     num_blocks=1, num_attention_heads=2, num_query_atoms=2, num_key_atoms=4,
            ... )
            >>> context = AtomContext(pair=torch.randn(1, 4, 4, 4), conditioning=torch.randn(1, 4, 8))
            >>> updates = decoder(
            ...     torch.randn(1, 2, 8),
            ...     atom_to_token=torch.tensor([[0, 0, 1, 1]]),
            ...     atom_query=torch.randn(1, 4, 8),
            ...     atom_context=context,
            ... )
            >>> updates.shape
            torch.Size([1, 4, 3])
        """

        atom_query = atom_query + self.token_to_atom_broadcast(self.token_projection(token_hidden), atom_to_token)
        atom_query = self.atom_attention_stack(
            atom_query,
            conditioning=atom_context.conditioning,
            pair=atom_context.pair,
            atom_mask=atom_mask,
            attention_layout=attention_layout,
            num_valid_atoms=num_valid_atoms,
        )
        return self.output_projection(self.query_norm(atom_query))


def _flatten_batch(tensor: Tensor, prefix: tuple[int, ...], payload_ndim: int) -> Tensor:
    r"""
    Broadcast a tensor to the full leading ``prefix`` and collapse that prefix into one batch axis.

    The atom stack runs over arbitrary leading (batch[, sample]) dims: in diffusion the query/conditioning/pair
    carry a noise-sample axis the mask does not. The conditioned transformer wants one batch axis, so each input
    is broadcast to the query's prefix and flattened.

    Args:
        tensor: Input whose leading dims are a left-prefix of ``prefix`` followed by ``payload_ndim`` axes.
        prefix: Full target leading shape (e.g. the query's ``(batch[, sample])`` dims).
        payload_ndim: Number of trailing payload axes to leave untouched.

    Returns:
        The tensor broadcast to ``prefix`` and reshaped to `[prod(prefix), *payload]`.

    Examples:
        >>> import torch
        >>> _flatten_batch(torch.zeros(2, 5, 8), (2,), 2).shape
        torch.Size([2, 5, 8])
        >>> _flatten_batch(torch.zeros(2, 5, 8), (2, 3), 2).shape
        torch.Size([6, 5, 8])
    """
    matched = tensor.dim() - payload_ndim
    missing = len(prefix) - matched
    if missing:
        tensor = tensor.reshape(*tensor.shape[:matched], *(1,) * missing, *tensor.shape[matched:])
        tensor = tensor.expand(*prefix, *tensor.shape[len(prefix) :])
    return tensor.reshape(-1, *tensor.shape[len(prefix) :])
