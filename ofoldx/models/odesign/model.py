r"""The released ODesign core graph and its canonical task methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import Self

from ...data.bonds import SymbolicBond
from ...data.conditions import Contact as _GeometricContact
from ...data.conditions import Hotspot as _Hotspot
from ...data.evidence import MSA, MSACollection

# §3.7.2 keeps the ODesign spatial/placement/placement condition API family-owned and still
# under investigation (not frozen). These records drive the family-owned hard-condition
# satisfaction check below and are re-homed family-locally in ``.conditions`` (the removed
# generic Condition base / validate_conditions of §3.7.3 stay gone).
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem, PolymerType
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
    SymbolicAtomSite,
    SymbolicResidueSite,
)
from ...data.placement import resolve as _resolve_site
from ...data.placement import validate_placement
from ...data.structure import Structure
from ...modules.atom import (
    AtomToTokenEncoder,
    LocalAtomGeometry,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)
from ...modules.attention import GatedMultiheadAttention, TriangleMultiplication
from ...modules.backend import LayerNorm
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import CoordinateDenoiseContext, CoordinateEdmSampler
from ...modules.embedding import (
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask
from ...modules.head import DistogramHead
from ...modules.transformer import ConditionedTransformerLayer, ProtenixConditionedTransformerLayer
from ...modules.transition import Transition
from ...modules.trunk import MsaTrunk, PairformerStack
from ...modules.trunk.msa import MsaPairWeightedAveraging, OuterProductMean
from ...results import DesignCandidate, Metric
from .config import ODesignConfig

# ======================================================================================
# Family-owned ODesign spatial and placement condition records (folded from the former
# ``.conditions`` submodule) and the ODesign hotspot target record (folded from the former
# ``.hotspot`` submodule). Defined ABOVE ``from .processor import _realize_ranges`` because
# ``processor`` imports these records back from ``.model`` — the names must already be bound
# when that submodule loads.
#
# DESIGN 3.7.2 ("Spatial conditioning is family-owned") freezes only the *negative*
# cross-family contract: there is no shared ``Condition`` base, union endpoint, or generic
# condition bag. Each concrete family owns its exact positive records. The data-model
# migration removed these records from the shared ``ofoldx.data.conditions`` module (which now
# keeps only the frozen 3.7.1 identity records ``AllowedResidueIdentities`` /
# ``TiedResidueIdentities``), so the ones ODesign consumes are re-homed here as
# ODesign-family-owned dataclasses. This mirrors the already-wired ``rfdiffusion`` /
# ``pxdesign`` families.
#
# The generation-topology placement records (:class:`~ofoldx.data.placement.GeneratedSegment`,
# :class:`~ofoldx.data.placement.ExistingSpan`, :class:`~ofoldx.data.placement.SymbolicResidueSite`,
# :class:`~ofoldx.data.placement.SymbolicAtomSite`) now come from the FROZEN §3.7 PlacementSpec
# placement core in ``ofoldx.data.placement`` (owner verdicts ratified 2026-07-18). Identity is
# placement-keyed: a placement's identity is the globally-unique ``placement_id`` inner key of
# the core :data:`~ofoldx.data.placement.PlacementSpec` mapping, so :class:`GeneratedSegment` carries NO
# embedded id and a pre-realization reference is exactly the two fields
# ``SymbolicResidueSite(placement_id, offset)``. This family only wraps that core with its own
# :class:`ODesignPlacements` condition, which re-homes the family-owned per-chain
# ``chain_lengths`` and the hard/soft ``intent`` (DESIGN 3.7.3 rejects both as shared
# invariants) alongside the core placement mapping.
#
# §3.7.4 constraint freeze: the covalent-link core (:class:`~ofoldx.data.bonds.SymbolicBond`
# over the promoted :data:`~ofoldx.data.bonds.SymbolicBondSite` endpoint union) and the
# target-conditioning core (the distance-free :class:`~ofoldx.data.conditions.Hotspot` label
# and the strictly-larger geometric :class:`~ofoldx.data.conditions.Contact`) are frozen and
# shared. This family imports them rather than re-declaring the placeholders it used to own
# (the deleted ``DesiredCovalentLink(site_a, site_b)`` two-atom record and the two-sided
# ``Contact`` that conflated the learned hotspot label with the §B verification cutoff).
#
# ``@dataclass(frozen=True)`` denotes runtime immutability, not DESIGN-frozen status.
# ======================================================================================


# --------------------------------------------------------------------------------------
# Family-owned PlacementSpec condition (wraps the FROZEN §3.7 core placement mapping).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ODesignPlacements:
    r"""One ODesign generation-topology request over the frozen §3.7 PlacementSpec core.

    ``placements`` is the frozen core :data:`~ofoldx.data.placement.PlacementSpec` mapping
    ``declaration_group -> placement_id -> value``; the inner ``placement_id`` is globally
    unique across the whole mapping and is the SOLE pre-realization reference target
    (validated at construction). Each declaration group is one ODesign output chain and its
    declared order is the chain layout order.

    ``chain_lengths`` is a family-owned per-chain exact-length field (DESIGN 3.7.3 rejects it
    as a shared invariant) aligned with ``placements`` group order; ``intent`` is read by the
    family hard-condition gate in ``model._require_hard_conditions``. Both re-home here as
    family-owned fields, never on the shared core.
    """

    placements: PlacementSpec
    chain_lengths: tuple[int | None, ...] | None = None
    intent: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        # Frozen §3.7: a placement_id must be globally unique across the whole PlacementSpec; this is
        # the construction-time uniqueness check (raises ValueError on any duplicate).
        validate_placement(self.placements)


# --------------------------------------------------------------------------------------
# Family-owned spatial conditions (DESIGN 3.7.2, family-owned & freezable target).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCoordinates:
    r"""Fixed/constraint coordinates for a set of EXISTING input atoms (§A1).

    ``intent == "hard"`` fixes the coordinates during EDM sampling; the constraint mask
    always feeds the learned constraint distogram. ``weight`` must be ``None`` / ``1.0``
    (the released signal is unweighted); the processor enforces that inline. Owner target
    name: ``ODesignFixedCoordinates``.
    """

    sites: tuple[AtomSite, ...]
    tolerance_angstrom: float
    intent: Literal["hard", "soft", "preference"] = "hard"
    weight: float | None = None


@dataclass(frozen=True)
class Hotspot(_Hotspot):
    r"""The §A2 learned per-residue hotspot label (family envelope over the frozen record).

    Extends the frozen distance-free :class:`~ofoldx.data.conditions.Hotspot` (owner verdict
    D3): ``residues`` names the EXISTING target residues written to
    ``token_features[..., 452]`` by ``processor._hotspot_tokens``; the generated partner is
    implicit (the whole design). This is the record the released two-sided ``Contact``
    soft-hotspot lowering is split onto — it carries NO ``cutoff``/``tolerance`` (those live
    on the geometric :class:`Contact`). The hard/soft ``intent`` rides on this family
    envelope, never on the frozen record; the released channel is always ``"soft"`` (the
    processor enforces it inline).

    An endpoint is a ``ComponentReference`` (existing residue), a bare ``placement_id``
    (whole generated placement), a :class:`SymbolicResidueSite` / :class:`SymbolicAtomSite`
    (generated residue/atom), or an :class:`~ofoldx.data.molecular.AtomSite` (existing atom).
    """

    intent: Literal["hard", "soft"] = "soft"


@dataclass(frozen=True)
class Contact(_GeometricContact):
    r"""The §B geometric hard-distance verification record (family envelope over the frozen
    :class:`~ofoldx.data.conditions.Contact`).

    Extends the frozen two-sided geometric ``Contact`` (owner verdict D3, identity is the
    real ``cutoff_angstrom``): ``model._require_hard_conditions`` reads ``side_a`` /
    ``side_b`` / ``cutoff_angstrom`` / ``tolerance_angstrom`` and requires the min
    cross-side distance to satisfy the cutoff. ``weight`` and hard/soft ``intent`` are the
    family-owned envelope fields the frozen record deliberately omits; ``weight`` is kept
    only so the processor can assert the released ``None`` / ``1.0`` no-weight contract.

    Endpoints are the shared :data:`~ofoldx.data.conditions.Endpoint` union (existing
    ``ComponentReference`` / :class:`~ofoldx.data.molecular.AtomSite`, a bare generated
    ``placement_id``, a :class:`SymbolicResidueSite` / :class:`SymbolicAtomSite`). This
    record is NO LONGER the learned hotspot signal — that is now :class:`Hotspot`.
    """

    weight: float | None = None
    intent: Literal["hard", "soft"] = "soft"


# --------------------------------------------------------------------------------------
# Covalent link envelope over the frozen §3.7.4 SymbolicBond core.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DesiredCovalentLink:
    r"""A requested covalent bond (head-to-tail cyclization) as a family envelope.

    The structural payload is the frozen :class:`~ofoldx.data.bonds.SymbolicBond` (owner
    verdicts D1/D2): its endpoints reference the placement model through the promoted
    :data:`~ofoldx.data.bonds.SymbolicBondSite` union, and its chemistry
    (``order``/``aromatic``/``stereo``) lowers losslessly to a
    :class:`~ofoldx.data.molecular.SystemBond`. ``intent`` (hard/soft) rides on THIS family
    envelope, never on the frozen record. ``_materialize_cyclic_links`` resolves the bond,
    synthesizes the ``SystemBond`` for a legal head-to-tail closure, and marks
    ``cyclic_instances`` → ``cyclic_period`` (behavior unchanged from the deleted two-atom
    placeholder — ODesign's previously hardcoded order ``1.0`` / aromatic ``False`` are now
    carried explicitly by the ``SymbolicBond`` defaults).
    """

    bond: SymbolicBond
    intent: Literal["hard", "soft"] = "hard"


# --------------------------------------------------------------------------------------
# design_mask <-> PlacementSpec projection (CONTIG_GENERALITY_AUDIT.md §3, family boundary).
# --------------------------------------------------------------------------------------
#
# ODesign natively selects fixed-vs-generated positions per output residue through a
# per-residue design mask (the ``res`` condition-mask driven by the ``sequence`` tokens:
# an ``N-M`` / ``N-N`` run designs a de-novo polymer interval, an ``X/A-B`` token copies a
# source span). These two functions are the lossless bridge between that native
# per-residue design-mask surface and the FROZEN §3.7 :data:`~ofoldx.data.placement.PlacementSpec`
# run-length placement core, per the projection contract in ``CONTIG_GENERALITY_AUDIT.md`` §3.

#: The core :data:`~ofoldx.data.molecular.PolymerType` values a generated polymer placement
#: may take. A de-novo *ligand* run (odesign ``{"chain_type":"ligand","sequence":"N-N"}`` =
#: ``num_atoms``) is GAP-1 in the audit: it has no ``PlacementValue`` and is rejected, not
#: silently dropped (the atom count stays a family argument; DESIGN §3.7 "does not justify a
#: shared ``GeneratedMolecule``").
_CORE_POLYMER_TYPES: frozenset[str] = frozenset(get_args(PolymerType))


@dataclass(frozen=True)
class DesignedResidue:
    r"""One realized output residue generated de novo — a ``design_mask`` ``True`` row.

    Carries only its ``polymer_type``; consecutive designed residues of one polymer type in
    one declaration group run-length-collapse into a single :class:`GeneratedSegment`.
    """

    polymer_type: PolymerType


@dataclass(frozen=True)
class FixedResidue:
    r"""One realized output residue copied verbatim from a source residue — a ``design_mask``
    ``False`` row.

    ``(instance_id, source_index)`` names the SOURCE input residue; consecutive fixed
    residues drawn contiguously (increasing ``source_index``) from one source instance
    run-length-collapse into a single half-open :class:`ExistingSpan`.
    """

    instance_id: str
    source_index: int


#: One per-output-residue design-mask cell: either a de-novo designed residue
#: (:class:`DesignedResidue`) or a fixed copy of a source residue (:class:`FixedResidue`).
ResidueSpec = DesignedResidue | FixedResidue


def from_design_mask(
    residues: Sequence[ResidueSpec],
    *,
    group_of: Callable[[int], str],
    placement_id_of: Callable[[int], str],
) -> PlacementSpec:
    r"""Project a per-output-residue design mask onto a run-length :data:`PlacementSpec`.

    ``residues`` is the realized output layout in order; ``group_of(i)`` labels residue ``i``'s
    declaration group (its ODesign output chain) and ``placement_id_of(i)`` its placement.
    Maximal consecutive runs that share one ``(group, placement_id)`` collapse into one
    placement: a run of :class:`DesignedResidue` of one ``polymer_type`` becomes
    ``GeneratedSegment(polymer_type, n, n)`` (exact realized length), and a run of
    :class:`FixedResidue` drawn contiguously from one source instance becomes the half-open
    ``ExistingSpan(instance_id, start, stop)``. A run also breaks when ``polymer_type`` changes,
    when the source instance changes, or when ``source_index`` is not the previous ``+ 1`` — so
    interleaving and discontiguous crops are preserved as ordered placements within the group.

    Losslessness: lossless over ODesign's polymer surface — the per-residue designed flag,
    each designed residue's ``polymer_type``, and each fixed residue's exact source residue are
    all recoverable (see :func:`to_design_mask` for the inverse). The one named loss is GAP-1:
    a designed non-polymer (ligand ``num_atoms``) residue has no ``PlacementValue`` and is
    rejected here rather than silently dropped. The partition is exhaustive — every input
    residue belongs to exactly one placement, so no realized row is unaddressable.
    """

    placement: PlacementSpec = {}
    seen: set[str] = set()

    current_group: str | None = None
    current_placement_id: str | None = None
    current_polymer_type: str | None = None
    current_instance_id: str | None = None
    run_start_source: int = 0
    run_next_source: int = 0
    run_length: int = 0

    def flush() -> None:
        if current_placement_id is None:
            return
        if current_polymer_type is not None:
            value: GeneratedSegment | ExistingSpan = GeneratedSegment(current_polymer_type, run_length, run_length)
        else:
            value = ExistingSpan(current_instance_id, run_start_source, run_next_source)
        placement.setdefault(current_group, {})[current_placement_id] = value

    for index, residue in enumerate(residues):
        group = group_of(index)
        placement_id = placement_id_of(index)
        if type(residue) is DesignedResidue:
            if residue.polymer_type not in _CORE_POLYMER_TYPES:
                raise ValueError(
                    f"designed residue {index} has non-core polymer_type "
                    f"{residue.polymer_type!r}; a de-novo non-polymer (ligand num_atoms) has no "
                    f"PlacementValue (GAP-1) and cannot be projected onto the placement core"
                )
            extends = (
                group == current_group
                and placement_id == current_placement_id
                and current_polymer_type == residue.polymer_type
            )
            if extends:
                run_length += 1
                continue
            flush()
            if placement_id in seen:
                raise ValueError(
                    f"placement_id {placement_id!r} reused by a non-contiguous run at residue "
                    f"{index}; a placement_id must be globally unique across the PlacementSpec"
                )
            seen.add(placement_id)
            current_group = group
            current_placement_id = placement_id
            current_polymer_type = residue.polymer_type
            current_instance_id = None
            run_length = 1
        else:
            extends = (
                group == current_group
                and placement_id == current_placement_id
                and current_polymer_type is None
                and current_instance_id == residue.instance_id
                and run_next_source == residue.source_index
            )
            if extends:
                run_next_source = residue.source_index + 1
                continue
            flush()
            if placement_id in seen:
                raise ValueError(
                    f"placement_id {placement_id!r} reused by a non-contiguous run at residue "
                    f"{index}; a placement_id must be globally unique across the PlacementSpec"
                )
            seen.add(placement_id)
            current_group = group
            current_placement_id = placement_id
            current_polymer_type = None
            current_instance_id = residue.instance_id
            run_start_source = residue.source_index
            run_next_source = residue.source_index + 1

    flush()
    # Frozen §3.7 construction-time check: placement_id globally unique across the PlacementSpec.
    validate_placement(placement)
    return placement


def to_design_mask(
    placement: PlacementSpec,
    placement_map: PlacementMap,
) -> tuple[list[bool], list[str]]:
    r"""Project a realized :data:`PlacementSpec` back onto a per-output-residue design mask.

    Returns ``(design_mask, groups)`` over the realized output rows in declared order: a
    :class:`GeneratedSegment` occurrence (its realized length read from ``placement_map``, so
    ranged ``min != max`` segments resolve to their sampled length) contributes ``True`` rows,
    an :class:`ExistingSpan` contributes ``stop - start`` ``False`` rows, and each row carries
    its declaration group (the outer key). This is the exact inverse of :func:`from_design_mask`
    over the polymer surface. Coverage is exhaustive: exactly one mask row is emitted per
    realized residue with no gap or overlap (frozen "every declared placement appears;
    zero-length -> ()"), so ``design_mask`` and ``groups`` partition the whole realized layout.
    """

    design_mask: list[bool] = []
    groups: list[str] = []
    for group, chain in placement.items():
        for placement_id, value in chain.items():
            if type(value) is GeneratedSegment:
                length = len(placement_map[group][placement_id])
                designed = True
            else:
                length = value.stop - value.start
                designed = False
            design_mask.extend([designed] * length)
            groups.extend([group] * length)
    return design_mask, groups


# --------------------------------------------------------------------------------------
# Family-owned ODesign hotspot target record (§3.7 / §4.11.5), folded from ``.hotspot``.
# --------------------------------------------------------------------------------------
#
# The released ODesign hotspot channel is a per-residue label lowered into
# ``token_features[..., 452]`` by ``processor._hotspot_tokens``. Crucially the released
# channel accepts **ligand** ``ComponentReference`` endpoints (the upstream ligand-design
# example uses ``hotspot: "E/505"``, a ligand chain), so an ODesign hotspot is emphatically
# NOT the synthetic fixed-distance protein-only ``Contact`` used by other families — it is a
# family-owned learned signal that may point at ligand components (DESIGN §3.7:1406-1407).
#
# :class:`ODesignHotspot` is the owner-gated **target** record for that signal (§4.11.5,
# §4.11.9 flag 3): a single-sided tuple of protein-or-ligand residue endpoints, with no
# ``cutoff``/``tolerance`` (those belong to the verification-only ``Contact``, not the learned
# hotspot). The owner's target split is:
#
#   * ``ODesignHotspot(residues)``            — the learned per-residue hotspot label; and
#   * a verification-only ``Contact``          — the §B post-generation hard-distance check.
#
# Until that split is owner-ratified, the *running* lowering stays on the two-sided
# :class:`Contact` soft-hotspot path (``processor.__call__`` / ``processor._hotspot_tokens``);
# this record is defined and exported as the design target but is **not** yet swapped into the
# running lowering. Importing it does not change behavior.


@dataclass(frozen=True)
class ODesignHotspot:
    r"""A learned per-residue ODesign hotspot label over protein OR ligand components.

    ``residues`` are canonical ``ComponentReference`` endpoints; unlike the synthetic
    protein-only fixed-distance ``Contact``, ligand components are first-class endpoints
    (§3.7 family-owned hotspot). Owner-gated target — NOT yet wired into the running
    ``Contact`` lowering (§4.11.9 flag 3).
    """

    residues: tuple[ComponentReference, ...]


class ODesignModel(nn.Module):
    r"""Released all-atom ODesign coordinate-design model."""

    config_class = ODesignConfig

    def __init__(self, config: ODesignConfig | None = None) -> None:
        config = ODesignConfig() if config is None else config
        super().__init__()
        self.config = config
        self.relative_position_embedder = RelativePositionEmbedder(
            config.relative_position_size,
            config.pair_size,
        )
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.input_embedder = InputEmbedder(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
        )
        self.msa_trunk = MsaTrunk(
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            num_blocks=config.trunk.num_msa_blocks,
            msa_dropout=config.trunk.msa_dropout,
            pair_dropout=config.trunk.pair_dropout,
        )
        self.constraint_distogram_embedder = ConstraintDistogramEmbedder(
            pair_size=config.pair_size,
            hidden_size=config.constraint.hidden_size,
            num_blocks=config.constraint.num_blocks,
            num_bins=config.constraint.num_bins,
            min_distance=config.constraint.min_distance,
            max_distance=config.constraint.max_distance,
            dropout=config.constraint.dropout,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=config.trunk.num_pairformer_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
            dropout=config.trunk.pairformer_dropout,
        )
        self.coordinate_denoiser = CoordinateDenoiser(
            num_encoder_blocks=config.denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=config.denoiser.num_token_encoder_attention_heads,
            sigma_data=config.denoiser.sigma_data,
            atom_hidden_size=config.denoiser.atom_hidden_size,
            atom_pair_hidden_size=config.denoiser.atom_pair_hidden_size,
            token_hidden_size=config.denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
        )
        self.coordinate_sampler = CoordinateEdmSampler(reverse_augmentation=True)
        self.distogram_head = DistogramHead(config.pair_size, config.pairwise.num_distogram_bins)
        self.token_bond_type_head = TokenBondTypeHead(
            pair_size=config.pair_size,
            hidden_size=config.pairwise.bond_hidden_size,
            num_bond_types=config.pairwise.num_bond_types,
        )
        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.distogram_head.linear.weight)
        if self.distogram_head.linear.bias is not None:
            nn.init.zeros_(self.distogram_head.linear.bias)
        self.recycling_steps = config.recycling_steps
        # ODesign inherits Protenix's ``AttentionPairBias`` double-adaLN convention: its cross-attention
        # key/value adaptive norm consumes the *already query-normed* activation, not the raw hidden
        # (ODesign/src/model/modules/transformer.py:207-212 -- ``layernorm_kv(a=a)`` runs on the ``a`` that
        # ``layernorm_a`` just overwrote). The shared atom encoder/decoder build the AF3-canonical base
        # layer, so re-cast every cross-attention atom transformer onto the Protenix-lineage subclass.
        for module in self.modules():
            if isinstance(module, ConditionedTransformerLayer) and module.cross_attention_mode:
                module.__class__ = ProtenixConditionedTransformerLayer
        self.set_backend("auto")

    def set_backend(self, backend: str) -> Self:
        r"""Select the public PyTorch or OOps operator path."""

        if backend not in {"auto", "torch", "oops"}:
            raise ValueError("backend must be exactly one of: auto, torch, oops")
        self.backend = backend
        operator_backend = "eager" if backend == "torch" else backend
        for module in self.modules():
            if isinstance(
                module,
                (
                    GatedMultiheadAttention,
                    MsaPairWeightedAveraging,
                    OuterProductMean,
                    TokenPairToLocalAtomPairBroadcast,
                    TokenToAtomBroadcast,
                    Transition,
                    TriangleMultiplication,
                ),
            ):
                module.operator_backend = operator_backend
        return self

    def forward(
        self,
        batch: ODesignInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        recycling_steps: int | None = None,
    ) -> ODesignOutput:
        r"""Denoise caller-supplied coordinates at one caller-supplied EDM noise level."""

        steps = self.recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        token_embedding, single, pair, relative_features = self._run_trunk(batch, recycling_steps=steps)
        context = self._denoise_context(batch, token_embedding, single, pair, relative_features)

        def denoise(noisy_chunk: Tensor, noise_chunk: Tensor) -> Tensor:
            return self.coordinate_denoiser(
                context.to_denoising_input(
                    noisy_coords=noisy_chunk,
                    noise_level=noise_chunk,
                )
            )

        coordinate_chunks = []
        for noisy_chunk, noise_chunk in zip(
            noisy_coordinates.split(4, dim=-3),
            noise_level.split(4, dim=-1),
        ):
            if self.training and torch.is_grad_enabled():
                coordinate_chunks.append(
                    checkpoint(
                        denoise,
                        noisy_chunk,
                        noise_chunk,
                        use_reentrant=False,
                    )
                )
            else:
                coordinate_chunks.append(denoise(noisy_chunk, noise_chunk))
        coordinates = torch.cat(coordinate_chunks, dim=-3)
        return ODesignOutput(
            coordinates=coordinates,
            distogram_logits=self.distogram_head(pair),
            token_bond_type_logits=self.token_bond_type_head(pair),
        )

    @torch.inference_mode()
    def design(
        self,
        batch: ODesignInput,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> ODesignDesignResult:
        r"""Generate coordinate-bearing canonical Structure design candidates.

        The caller-supplied ``generator`` (resolved once at the Pipeline boundary, one seed
        per ``design`` call per §4.11.3/§4.11.4) is the sole random stream; per-sample
        diffusion sampling remains model-intrinsic.
        """

        # Lazy import breaks the model<->processor cycle: ``.processor`` imports the folded-in
        # ODesignInput/condition contracts from ``.model`` at its module top, so ``.model`` must
        # not import ``.processor`` at module scope; ``_realize_ranges`` is only needed at call time.
        from .processor import _realize_ranges

        batch = _realize_ranges(batch, generator)
        output = self.sample(
            batch,
            num_samples=num_candidates,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            num_model_seeds=1,
            partial_diffusion_snr=partial_diffusion_snr,
            generator=generator,
        )
        structures = tuple(
            _structure_from_coordinates(output.coordinates[0, 0, index], batch.layout)
            for index in range(num_candidates)
        )
        for structure in structures:
            _require_hard_conditions(structure, batch)
        # §5.3.1/§5.3.3: the frozen shared DesignCandidate is coordinate-free (system,
        # scores) and must NOT be widened. ODesign is coordinate-generating, so each sample
        # is carried by the family-local ODesignDesignCandidate subtype, which adds the
        # decoded Structure and a per-sample metrics map (empty: the released core exposes no
        # per-sample confidence head). ``system = structure.system`` per the §5.3.1 producer
        # rule. Raw distogram/token-bond logits stay on the ODesignOutput; run provenance and
        # generated-identity maps are not carried here.
        # §4.11.3/§4.11.4: one Model.design call is one seed's M backbones (num_model_seeds=1).
        # The container is the family-owned ODesignDesignResult (§5.3.1/§5.3.3: coordinate-
        # producing families own a concrete family Result, not the widened shared DesignResult),
        # matching the already-emitted ODesignDesignCandidate subtype. Backbone-only stage, so
        # ``scores`` is empty and ``num_sequences=0``; the OInvFold sequence step (K axis) is
        # composed downstream by ODesignPipeline. ``backbone_index`` tags the M axis here;
        # ``seed_index`` stays 0 (the S loop is owned by the Pipeline, one seed per Model.design).
        candidates = tuple(
            ODesignDesignCandidate(
                system=structure.system,
                structure=structure,
                backbone_index=index,
            )
            for index, structure in enumerate(structures)
        )
        return ODesignDesignResult(
            candidates=candidates,
            num_seeds=1,
            num_backbones=num_candidates,
            num_sequences=0,
        )

    @torch.inference_mode()
    def sample(
        self,
        batch: ODesignInput,
        *,
        num_samples: int = 1,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        num_model_seeds: int = 1,
        partial_diffusion_snr: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ODesignOutput:
        r"""Run the released multi-seed sampler without discarding its pairwise heads."""

        steps = self.recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        partial_diffusion = partial_diffusion_snr is not None
        if partial_diffusion and not bool(batch.observed_mask.all()):
            raise ValueError("partial diffusion requires observed coordinates for every model atom")
        max_noise_level = 1 / math.sqrt(partial_diffusion_snr + 1e-6) if partial_diffusion_snr is not None else None
        coordinates = []
        distograms = []
        bond_logits = []
        for _ in range(num_model_seeds):
            token_embedding, single, pair, relative_features = self._run_trunk(batch, recycling_steps=steps)
            distograms.append(self.distogram_head(pair))
            bond_logits.append(self.token_bond_type_head(pair))
            coordinates.append(
                self.coordinate_sampler(
                    self.coordinate_denoiser,
                    context=self._denoise_context(batch, token_embedding, single, pair, relative_features),
                    num_samples=num_samples,
                    num_sampling_steps=num_sampling_steps,
                    sample_chunk_size=1,
                    initial_coords=batch.observed_positions if partial_diffusion else None,
                    condition_coords=batch.observed_positions,
                    condition_mask=batch.fixed_coordinate_mask,
                    max_noise_level=max_noise_level,
                    generator=generator,
                )
            )
        return ODesignOutput(
            coordinates=torch.stack(coordinates),
            distogram_logits=torch.stack(distograms),
            token_bond_type_logits=torch.stack(bond_logits),
        )

    def _run_trunk(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor, RelativePositionBuckets]:
        local_geometry = self.input_embedder.build_local_geometry(
            batch.reference_positions,
            batch.reference_space_id,
        )
        token_embedding = self.input_embedder(
            batch.atom_features,
            reference_positions=batch.reference_positions,
            reference_charges=batch.reference_charges,
            reference_mask=batch.reference_mask,
            token_features=batch.token_features,
            atom_to_token=batch.atom_to_token,
            num_tokens=batch.token_features.shape[-2],
            atom_mask=batch.atom_mask,
            token_mask=batch.token_mask,
            local_geometry=local_geometry,
        )
        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            cyclic_period=batch.cyclic_period,
            dtype=single_init.dtype,
        )
        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        pair_init = self._initial_pair(single_init, batch, relative_features)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        grad_enabled = torch.is_grad_enabled()
        for cycle in range(recycling_steps):
            with torch.set_grad_enabled(grad_enabled and (not self.training or cycle == recycling_steps - 1)):
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                pair = pair + self.constraint_distogram_embedder(
                    pair,
                    batch.constraint_distogram,
                    pair_mask=pair_mask,
                )
                _, pair = self.msa_trunk(
                    batch.msa_features,
                    pair,
                    token_embedding,
                    msa_mask=batch.msa_mask,
                    pair_mask=pair_mask,
                )
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        return token_embedding, single, pair, relative_features

    def _initial_pair(
        self,
        single: Tensor,
        batch: ODesignInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        pair = self.pair_left_projection(single).unsqueeze(-2)
        pair = pair + self.pair_right_projection(single).unsqueeze(-3)
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(batch.token_bonds.unsqueeze(-1))
        return pair * pair_mask_from_token_mask(batch.token_mask).unsqueeze(-1).to(pair.dtype)

    def _denoise_context(
        self,
        batch: ODesignInput,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> CoordinateDenoiseContext:
        return CoordinateDenoiseContext(
            atom_features=batch.atom_features,
            atom_to_token=batch.atom_to_token,
            reference_positions=batch.reference_positions,
            reference_charges=batch.reference_charges,
            reference_mask=batch.reference_mask,
            atom_mask=batch.atom_mask,
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                batch.reference_positions,
                batch.reference_space_id,
            ),
        )


class InputEmbedder(nn.Module):
    r"""Aggregate reference atoms and append ODesign token features."""

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 386,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.atom_encoder = AtomToTokenEncoder(
            token_hidden_size=single_size,
            atom_feature_size=atom_feature_size,
        )

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        token_features: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        atom_mask: Tensor,
        token_mask: Tensor,
        local_geometry: LocalAtomGeometry,
    ) -> Tensor:
        encoded = self.atom_encoder(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=torch.arcsinh(reference_charges),
            reference_mask=reference_mask,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
            local_geometry=local_geometry,
        ).token_hidden
        output = torch.cat(
            [encoded, token_features[..., self.single_size :].to(encoded)],
            dim=-1,
        )
        return output * token_mask.unsqueeze(-1).to(output.dtype)

    def build_local_geometry(self, reference_positions: Tensor, reference_space_id: Tensor) -> LocalAtomGeometry:
        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)


class TokenBondTypeHead(nn.Module):
    r"""Predict whether each ODesign token pair is covalently connected."""

    def __init__(self, *, pair_size: int = 128, hidden_size: int = 128, num_bond_types: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(pair_size, hidden_size),
            nn.Linear(hidden_size, num_bond_types),
        )

    def forward(self, pair: Tensor) -> Tensor:
        return self.classifier(pair)


class ConstraintDistogramEmbedder(nn.Module):
    r"""Embed pairwise target distances into the ODesign pair representation."""

    def __init__(
        self,
        *,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        num_bins: int = 39,
        min_distance: float = 3.25,
        max_distance: float = 50.75,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.pair_norm = LayerNorm(pair_size)
        self.distogram_projection = nn.Linear(num_bins, hidden_size, bias=False)
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=0,
            pair_size=hidden_size,
            dropout=dropout,
        )
        self.hidden_norm = LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(self, pair: Tensor, distances: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        boundaries = torch.linspace(
            self.min_distance,
            self.max_distance,
            self.num_bins - 1,
            device=pair.device,
            dtype=pair.dtype,
        )
        bins = (distances.to(pair.dtype) > boundaries).sum(dim=-1)
        distogram = F.one_hot(bins, self.num_bins).to(pair.dtype)
        hidden = self.pair_projection(self.pair_norm(pair)) + self.distogram_projection(distogram)
        _, hidden = self.pairformer_stack(None, hidden, pair_mask=pair_mask)
        return self.output_projection(self.activation(self.hidden_norm(hidden)))


def _structure_from_coordinates(coordinates: Tensor, layout: ODesignLayout) -> Structure:
    values = coordinates.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
    site_index = {site: index for index, site in enumerate(sites)}
    complete = np.zeros((len(sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(sites), dtype=np.bool_)
    for predicted_index, site in enumerate(layout.sites):
        output_index = site_index[site]
        complete[output_index] = values[predicted_index]
        coordinate_mask[output_index] = True
    return Structure(
        system=layout.system,
        sites=sites,
        coordinates=complete,
        coordinate_mask=coordinate_mask,
    )


def _require_hard_conditions(structure: Structure, batch: ODesignInput) -> None:
    r"""Raise when a coordinate-generated candidate violates a hard condition."""

    source_residues = batch.layout.source_residue_map
    placement_map = batch.layout.placement_map
    instance_of = batch.layout.instance_of
    source_atoms = batch.layout.source_atom_map
    coordinates = {site: structure.coordinates[index] for index, site in enumerate(structure.sites)}
    reference = {
        site: batch.observed_positions[0, index].detach().to(device="cpu", dtype=torch.float32).numpy()
        for index, site in enumerate(batch.layout.sites)
    }
    for index, condition in enumerate(batch.layout.conditions):
        satisfied: bool | None
        # TODO(§3.7↔§5 reopened): `value` is a real per-condition constraint measurement
        # (angstrom RMSD / min-distance) with no frozen per-condition home; only the
        # hard-condition raise survives here until a §3.7↔§5 sub-freeze defines a carrier.
        value: float | None = None
        if type(condition) is ReferenceCoordinates:
            sites = tuple(
                _resolve_atom(site, source_residues, placement_map, instance_of, source_atoms)
                for site in condition.sites
            )
            predicted = np.stack([coordinates[site] for site in sites])
            target = np.stack([reference[site] for site in sites])
            predicted_centered = predicted - predicted.mean(axis=0)
            target_centered = target - target.mean(axis=0)
            left, _, right = np.linalg.svd(predicted_centered.T @ target_centered)
            if np.linalg.det(left @ right) < 0:
                left[:, -1] *= -1
            aligned = predicted_centered @ (left @ right) + target.mean(axis=0)
            distances = np.linalg.norm(aligned - target, axis=-1)
            value = float(distances.max())
            satisfied = value <= condition.tolerance_angstrom
        elif type(condition) is Contact:
            left = _contact_sites(
                condition.side_a,
                batch.layout,
                source_residues,
                placement_map,
                instance_of,
                source_atoms,
            )
            right = _contact_sites(
                condition.side_b,
                batch.layout,
                source_residues,
                placement_map,
                instance_of,
                source_atoms,
            )
            value = min(float(np.linalg.norm(coordinates[a] - coordinates[b])) for a in left for b in right)
            satisfied = value <= condition.cutoff_angstrom + condition.tolerance_angstrom
        elif type(condition) in (ODesignPlacements, DesiredCovalentLink):
            satisfied = True
        # TODO(§3.7.2 reopened): `condition.intent` is now a field of the family-owned ODesign
        # ReferenceCoordinates/Contact records, not the removed universal condition protocol (§3.7.3);
        # this hard-condition satisfaction gate is family-owned and preserved until the API freezes.
        if condition.intent == "hard" and not satisfied:
            raise RuntimeError(f"ODesign candidate violated hard conditions[{index}]")


def _resolve_residue(
    reference: tuple[str, int] | SymbolicResidueSite,
    source_residues,
    placement_map,
    instance_of,
) -> tuple[str, int]:
    if type(reference) is tuple:
        return source_residues[reference]
    return _resolve_site(reference, placement_map, instance_of)


def _resolve_atom(reference, source_residues, placement_map, instance_of, source_atoms) -> AtomSite:
    if type(reference) is AtomSite:
        return source_atoms[reference]
    instance_id, monomer_index = _resolve_residue(
        reference.residue,
        source_residues,
        placement_map,
        instance_of,
    )
    return AtomSite(instance_id, monomer_index, reference.atom_id)


def _contact_sites(
    references,
    layout: ODesignLayout,
    source_residues,
    placement_map,
    instance_of,
    source_atoms,
) -> tuple[AtomSite, ...]:
    sites: list[AtomSite] = []
    for reference in references:
        if type(reference) in (AtomSite, SymbolicAtomSite):
            sites.append(_resolve_atom(reference, source_residues, placement_map, instance_of, source_atoms))
            continue
        residues = (
            placement_map[instance_of[reference]][reference]
            if type(reference) is str
            else (_resolve_residue(reference, source_residues, placement_map, instance_of),)
        )
        for residue in residues:
            beta = AtomSite(residue[0], residue[1], "CB")
            sites.append(beta if beta in layout.sites else AtomSite(residue[0], residue[1], "CA"))
    return tuple(sites)


# ---------------------------------------------------------------------------
# Tensor and semantic contracts owned by ODesign (folded in from contract.py).
# ---------------------------------------------------------------------------


@dataclass
class ODesignLayout:
    r"""Canonical identity needed to turn ODesign coordinates into Results.

    ``placement_map`` is the frozen §3.7 realized output surface for generated placements
    (``declaration_group -> placement_id -> ordered ComponentReference occurrences``); it is
    resolved together with ``instance_of`` (``placement_id -> declaration_group``, derived
    once from the PlacementSpec) via ``ofoldx.data.placement.resolve``. There is exactly one canonical
    realized map and one resolution helper — no parallel positive/negative alias table.
    """

    system: MolecularSystem
    sites: tuple[AtomSite, ...]
    conditions: tuple[Contact | Hotspot | ODesignPlacements | DesiredCovalentLink | ReferenceCoordinates, ...]
    placement_map: PlacementMap = field(default_factory=dict)
    instance_of: dict[str, str] = field(default_factory=dict)
    source_residue_map: dict[tuple[str, int], tuple[str, int]] = field(default_factory=dict)
    source_atom_map: dict[AtomSite, AtomSite] = field(default_factory=dict)
    msa: MSA | MSACollection | None = None


@dataclass(kw_only=True)
class ODesignInput:
    r"""Compute tensors plus canonical semantic layout for the ODesign core graph.

    ``B`` is batch size, ``T`` token count, ``A`` model-atom count, and ``R`` MSA
    row count. Index and mask tensors use ``int64`` and ``bool`` respectively.
    ``token_features`` is ``[B, T, 453]``; ``token_bonds`` is ``[B, T, T]``;
    ``atom_features`` is ``[B, A, 386]``; atom positions are ``[B, A, 3]``;
    charges are ``[B, A, 1]``; ``constraint_distogram`` is ``[B, T, T, 39]``;
    and MSA features/mask are ``[B, R, T, 34]`` and ``[B, R, T]``. All floating
    tensors use the Processor's requested dtype and device.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    cyclic_period: Tensor
    token_features: Tensor
    token_bonds: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    observed_positions: Tensor
    observed_mask: Tensor
    fixed_coordinate_mask: Tensor
    constraint_distogram: Tensor
    msa_features: Tensor
    msa_mask: Tensor
    layout: ODesignLayout


@dataclass
class ODesignOutput:
    r"""The three predictions computed by the released core graph.

    A one-step forward returns coordinates ``[B, S, A, 3]``, distogram logits
    ``[B, T, T, 64]``, and token-bond logits ``[B, T, T, 2]``. Sampling adds a
    leading model-seed axis to all three tensors.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_type_logits: Tensor


@dataclass(eq=False)
class ODesignDesignCandidate(DesignCandidate):
    r"""One coordinate-bearing ODesign design sample.

    ODesign is a coordinate-*generating* design family, so its candidate carries a decoded
    :class:`~ofoldx.data.structure.Structure` and per-sample ``metrics`` alongside the
    shared coordinate-free ``system`` / ``scores``. It subclasses the frozen §5.3.1
    :class:`~ofoldx.results.DesignCandidate` (rather than widening it, which §5.3.3 forbids)
    so :class:`~ofoldx.results.DesignResult` can carry it unchanged, and mirrors
    ``RFdiffusionDesignCandidate`` / ``PxDesignDesignCandidate``. This is a family-local
    carrier pending the §5.3 coordinate-bearing design sub-freeze — NOT a frozen public
    shape. The released ODesign core exposes no per-sample confidence head, so ``metrics``
    is empty today; the distogram / token-bond logits stay on :class:`ODesignOutput`.
    """

    structure: Structure | None = None
    metrics: dict[str, Metric] = field(default_factory=dict)
    seed_index: int = 0
    backbone_index: int = 0
    sequence_index: int = 0


@dataclass(eq=False)
class ODesignDesignResult:
    r"""The ODesign family Result for the released M×S×K design grid (§4.11.3).

    ODesign is a coordinate-*producing* design family, so §5.3.1/§5.3.3 require it to own a
    concrete family Result rather than widen the frozen coordinate-free
    :class:`~ofoldx.results.DesignResult`. ``candidates`` is generation-ordered
    (seed → backbone → sequence); ``num_seeds`` (S), ``num_backbones`` (M, per-seed
    diffusion samples), and ``num_sequences`` (K, OInvFold top-k per backbone) record the
    released loop structure. The ``K`` sequence variants of a given ``(seed, backbone)``
    share one backbone coordinate set. A backbone-only ``design`` call sets
    ``num_sequences=0``.

    This is a family-local carrier pending the §5.3 coordinate-bearing design sub-freeze —
    NOT a frozen public shape (§4.11.9, owner-gated flag 7).
    """

    candidates: tuple[ODesignDesignCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best(self, score: str) -> ODesignDesignCandidate:
        r"""Return the earliest best candidate under one explicit score.

        Reproduces the frozen §5.3.1 ranking contract verbatim (explicit ``preferred``
        direction, one identical unit, one identical ``frozenset`` scope, NaN rejection,
        earliest tie), mirroring :meth:`ofoldx.results.DesignResult.best`. There is no
        shared ranking abstraction and no default score.
        """

        scores = tuple(candidate.scores[score] for candidate in self.candidates)
        directions = {value.preferred for value in scores}
        if directions not in ({"higher"}, {"lower"}):
            raise ValueError(f"score {score!r} must consistently prefer higher or lower values")
        if len({value.unit for value in scores}) != 1:
            raise ValueError(f"score {score!r} must use one identical unit")
        if len({value.scope for value in scores}) != 1:
            raise ValueError(f"score {score!r} must use one identical scope")
        if any(math.isnan(value.value) for value in scores):
            raise ValueError(f"score {score!r} contains NaN and cannot be ranked")
        if directions == {"higher"}:
            return max(self.candidates, key=lambda candidate: candidate.scores[score].value)
        return min(self.candidates, key=lambda candidate: candidate.scores[score].value)
