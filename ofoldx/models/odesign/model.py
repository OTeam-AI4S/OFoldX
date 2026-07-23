r"""The ODesign core graph and its canonical task methods."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ...data.bonds import SymbolicBond
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MSACollection

# ODesign's spatial and placement records drive the hard-condition checks below.
from ...data.molecular import (
    AtomSite,
    MolecularSystem,
    PolymerType,
)
from ...data.placement import (
    PlacementMap,
    PlacementSpec,
    SymbolicResidueSite,
)
from ...data.placement import resolve as _resolve_site
from ...data.placement import validate_placement
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import (
    AtomToTokenEncoder,
    LocalAtomGeometry,
)
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
from ...modules.trunk import MsaTrunk, PairformerStack
from ...results import Metric, Score, best
from .config import ODesignConfig


class ODesignModel(Model):
    r"""All-atom ODesign coordinate-design model."""

    config_class = ODesignConfig
    config: ODesignConfig

    def __init__(self, config: ODesignConfig | None = None) -> None:
        config = ODesignConfig() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self.input_embeddings = ODesignInputEmbeddings(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
        )
        self.constraint_embeddings = ODesignConstraintEmbeddings(
            pair_size=config.pair_size,
            hidden_size=config.constraint_embeddings.hidden_size,
            num_blocks=config.constraint_embeddings.num_blocks,
            num_bins=config.constraint_embeddings.num_bins,
            min_distance=config.constraint_embeddings.min_distance,
            max_distance=config.constraint_embeddings.max_distance,
            dropout=config.constraint_embeddings.dropout,
            operator_backend="auto",
        )
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(
            config.relative_position_size,
            config.pair_size,
        )
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.msa_trunk = MsaTrunk(
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            num_blocks=config.trunk.num_msa_blocks,
            msa_dropout=config.trunk.msa_dropout,
            pair_dropout=config.trunk.pair_dropout,
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
            cross_attention_key_source="normalized",
        )
        self.coordinate_sampler = CoordinateEdmSampler(reverse_augmentation=True)
        self.distogram_head = DistogramHead(config.pair_size, config.pairwise.num_distogram_bins)
        self.token_bond_type_head = ODesignTokenBondTypeHead(
            pair_size=config.pair_size,
            hidden_size=config.pairwise.bond_hidden_size,
            num_bond_types=config.pairwise.num_bond_types,
        )
        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.distogram_head.projection.weight)
        if self.distogram_head.projection.bias is not None:
            nn.init.zeros_(self.distogram_head.projection.bias)
        self.post_init()

    def forward(
        self,
        input: ODesignInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        recycling_steps: int | None = None,
    ) -> ODesignOutput:
        r"""Denoise caller-supplied coordinates at one caller-supplied EDM noise level."""

        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        trunk = self._run_trunk(input, recycling_steps=steps)
        context = self._denoise_context(input, trunk)

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
            distogram_logits=self.distogram_head(trunk.pair),
            token_bond_type_logits=self.token_bond_type_head(trunk.pair),
        )

    @torch.inference_mode()
    def design(
        self,
        input: ODesignInput,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> ODesignDesignResult:
        r"""Generate coordinate-bearing canonical Structure design candidates.

        The caller-supplied ``generator`` is resolved once at the Pipeline boundary and is the
        sole random stream for one ``design`` call; per-sample
        diffusion sampling remains model-intrinsic.
        """

        output = self.sample(
            input,
            num_samples=num_candidates,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            num_model_seeds=1,
            partial_diffusion_snr=partial_diffusion_snr,
            generator=generator,
        )
        structures = tuple(
            _structure_from_coordinates(output.coordinates[0, 0, index], input.layout)
            for index in range(num_candidates)
        )
        for structure in structures:
            _require_hard_conditions(structure, input)
        # ODesign candidates own the decoded Structure and per-sample metrics. Raw
        # distogram/token-bond logits stay on ODesignOutput; run provenance and
        # generated-identity maps are not duplicated on each candidate.
        # One Model.design call produces one seed's M backbones (num_model_seeds=1). The
        # ODesignDesignResult carries these coordinate-bearing candidates.
        # This is the backbone-only stage, so
        # ``scores`` is empty and ``num_sequences=0``; the OInvFold sequence step (K axis) is
        # composed downstream by ODesignPipeline. ``backbone_index`` tags the M axis here;
        # ``seed_index`` stays 0 (the S loop is owned by the Pipeline, one seed per Model.design).
        candidates = tuple(
            ODesignDesignCandidate(
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
        input: ODesignInput,
        *,
        num_samples: int = 1,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        num_model_seeds: int = 1,
        partial_diffusion_snr: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ODesignOutput:
        r"""Run the multi-seed sampler without discarding its pairwise heads."""

        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        partial_diffusion = partial_diffusion_snr is not None
        if partial_diffusion and not bool(input.observed_mask.all()):
            raise ValueError("partial diffusion requires observed coordinates for every model atom")
        max_noise_level = 1 / math.sqrt(partial_diffusion_snr + 1e-6) if partial_diffusion_snr is not None else None
        coordinates = []
        distograms = []
        bond_logits = []
        for _ in range(num_model_seeds):
            trunk = self._run_trunk(input, recycling_steps=steps)
            distograms.append(self.distogram_head(trunk.pair))
            bond_logits.append(self.token_bond_type_head(trunk.pair))
            coordinates.append(
                self.coordinate_sampler(
                    self.coordinate_denoiser,
                    context=self._denoise_context(input, trunk),
                    num_samples=num_samples,
                    num_sampling_steps=num_sampling_steps,
                    sample_chunk_size=1,
                    initial_coords=input.observed_positions if partial_diffusion else None,
                    condition_coords=input.observed_positions,
                    condition_mask=input.fixed_coordinate_mask,
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
        input: ODesignInput,
        *,
        recycling_steps: int,
    ) -> _ODesignTrunkOutput:
        local_geometry = self.input_embeddings.build_local_geometry(
            input.reference_positions,
            input.reference_space_id,
        )
        token_embedding = self.input_embeddings(
            input.atom_features,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            token_features=input.token_features,
            atom_to_token=input.atom_to_token,
            num_tokens=input.token_features.shape[-2],
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=local_geometry,
        )
        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            cyclic_period=input.cyclic_period,
            dtype=single_init.dtype,
        )
        pair_mask = pair_mask_from_token_mask(input.token_mask)
        pair_init = self._initial_pair(single_init, input, relative_features)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        grad_enabled = torch.is_grad_enabled()
        for cycle in range(recycling_steps):
            with torch.set_grad_enabled(grad_enabled and (not self.training or cycle == recycling_steps - 1)):
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                pair = pair + self.constraint_embeddings(
                    pair,
                    input.constraint_distogram,
                    pair_mask=pair_mask,
                )
                _, pair = self.msa_trunk(
                    input.msa_features,
                    pair,
                    token_embedding,
                    msa_mask=input.msa_mask,
                    pair_mask=pair_mask,
                )
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        return _ODesignTrunkOutput(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_position_features=relative_features,
        )

    def _initial_pair(
        self,
        single: Tensor,
        input: ODesignInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        pair = self.pair_left_projection(single).unsqueeze(-2)
        pair = pair + self.pair_right_projection(single).unsqueeze(-3)
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(input.token_bonds.unsqueeze(-1))
        return pair * pair_mask_from_token_mask(input.token_mask).unsqueeze(-1).to(pair.dtype)

    def _denoise_context(
        self,
        input: ODesignInput,
        trunk: _ODesignTrunkOutput,
    ) -> CoordinateDenoiseContext:
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=trunk.token_embedding,
            single=trunk.single,
            pair=trunk.pair,
            relative_features=trunk.relative_position_features,
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                input.reference_positions,
                input.reference_space_id,
            ),
        )


class ODesignInputEmbeddings(nn.Module):
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
            cross_attention_key_source="normalized",
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


class ODesignConstraintEmbeddings(nn.Module):
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
        operator_backend: str = "auto",
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
            operator_backend=operator_backend,
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


class ODesignTokenBondTypeHead(nn.Module):
    r"""Predict whether each ODesign token pair is covalently connected."""

    def __init__(self, *, pair_size: int = 128, hidden_size: int = 128, num_bond_types: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(pair_size, hidden_size),
            nn.Linear(hidden_size, num_bond_types),
        )

    def forward(self, pair: Tensor) -> Tensor:
        return self.classifier(pair)


def _structure_from_coordinates(coordinates: Tensor, layout: ODesignLayout) -> Structure:
    values = coordinates.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    sites = tuple(site for site, atom in layout.system.iter_atoms() if atom.element != "H")
    site_index = {site: index for index, site in enumerate(sites)}
    complete = np.zeros((len(sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(sites), dtype=np.bool_)
    for predicted_index, site in enumerate(layout.atom_sites):
        output_index = site_index[site]
        complete[output_index] = values[predicted_index]
        coordinate_mask[output_index] = True
    return Structure(
        system=layout.system,
        sites=sites,
        coordinates=complete,
        coordinate_mask=coordinate_mask,
    )


def _require_hard_conditions(structure: Structure, input: ODesignInput) -> None:
    r"""Raise when a candidate violates fixed-coordinate evidence."""

    source_residues = input.layout.source_residue_map
    placement_map = input.layout.placement_map
    instance_of = input.layout.instance_of
    source_atoms = input.layout.source_atom_map
    coordinates = {site: structure.coordinates[index] for index, site in enumerate(structure.sites)}
    reference = {
        site: input.observed_positions[0, index].detach().to(device="cpu", dtype=torch.float32).numpy()
        for index, site in enumerate(input.layout.atom_sites)
    }
    for index, condition in enumerate(input.layout.conditions):
        if not isinstance(condition, ODesignReferenceCoordinates) or condition.intent != "hard":
            continue
        sites = tuple(
            _resolve_atom(site, source_residues, placement_map, instance_of, source_atoms) for site in condition.sites
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
        if float(distances.max()) > condition.tolerance_angstrom:
            raise RuntimeError(f"ODesign candidate violated hard conditions[{index}]")


def _resolve_residue(
    reference: tuple[str, int] | SymbolicResidueSite,
    source_residues,
    placement_map,
    instance_of,
) -> tuple[str, int]:
    if isinstance(reference, tuple):
        return source_residues[reference]
    return _resolve_site(reference, placement_map, instance_of)


def _resolve_atom(reference, source_residues, placement_map, instance_of, source_atoms) -> AtomSite:
    if isinstance(reference, AtomSite):
        return source_atoms[reference]
    instance_id, monomer_index = _resolve_residue(
        reference.residue,
        source_residues,
        placement_map,
        instance_of,
    )
    return AtomSite(instance_id, monomer_index, reference.atom_id)


# ---------------------------------------------------------------------------
# Typed tensor and semantic records for ODesign.
# ---------------------------------------------------------------------------


@dataclass
class _ODesignTrunkOutput:
    r"""Named representations produced by the ODesign trunk."""

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_position_features: RelativePositionBuckets


@dataclass(frozen=True)
class ODesignPlacements:
    r"""One ODesign generation-topology request over the shared PlacementSpec.

    ``placements`` is the :data:`~ofoldx.data.placement.PlacementSpec` mapping
    ``declaration_group -> placement_id -> value``; the inner ``placement_id`` is globally
    unique across the whole mapping and is the SOLE pre-realization reference target
    (validated at construction). Each declaration group is one ODesign output chain and its
    declared order is the chain layout order.

    ``chain_lengths`` is a per-chain exact-length field aligned with
    ``placements`` group order.
    """

    placements: PlacementSpec
    chain_lengths: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        # A placement_id must be globally unique across the whole PlacementSpec.
        validate_placement(self.placements)


@dataclass(frozen=True)
class ODesignReferenceCoordinates:
    r"""Fixed or constrained coordinates for a set of existing input atoms.

    ``intent == "hard"`` fixes the coordinates during EDM sampling; both modes feed the
    learned constraint distogram.
    """

    sites: tuple[AtomSite, ...]
    tolerance_angstrom: float
    intent: Literal["hard", "soft"] = "hard"


@dataclass(frozen=True)
class ODesignDesignedResidue:
    r"""One realized output residue generated de novo — a ``design_mask`` ``True`` row.

    Carries only its ``polymer_type``; consecutive designed residues of one polymer type in
    one declaration group run-length-collapse into a single :class:`GeneratedSegment`.
    """

    polymer_type: PolymerType


@dataclass(frozen=True)
class ODesignFixedResidue:
    r"""One realized output residue copied verbatim from a source residue — a ``design_mask``
    ``False`` row.

    ``(instance_id, source_index)`` names the SOURCE input residue; consecutive fixed
    residues drawn contiguously (increasing ``source_index``) from one source instance
    run-length-collapse into a single half-open :class:`ExistingSpan`.
    """

    instance_id: str
    source_index: int


ODesignResidueSpec = ODesignDesignedResidue | ODesignFixedResidue


@dataclass
class ODesignLayout:
    r"""Canonical identity needed to turn ODesign coordinates into Results.

    ``placement_map`` is the realized output surface for generated placements
    (``declaration_group -> placement_id -> ordered ComponentReference occurrences``); it is
    resolved together with ``instance_of`` (``placement_id -> declaration_group``, derived
    once from the PlacementSpec) via ``ofoldx.data.placement.resolve``. There is exactly one canonical
    realized map and one resolution helper — no parallel positive/negative alias table.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    placement: ODesignPlacements | None
    conditions: tuple[Hotspot | SymbolicBond | ODesignReferenceCoordinates, ...]
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
    r"""The three predictions computed by the core graph.

    A one-step forward returns coordinates ``[B, S, A, 3]``, distogram logits
    ``[B, T, T, 64]``, and token-bond logits ``[B, T, T, 2]``. Sampling adds a
    leading model-seed axis to all three tensors.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_type_logits: Tensor


@dataclass
class ODesignDesignCandidate:
    r"""One coordinate-bearing ODesign design sample.

    The decoded :class:`~ofoldx.data.structure.Structure` is the single owner of
    molecular identity and coordinates. The core exposes no per-sample
    confidence head, so ``metrics`` is empty today; distogram and token-bond logits
    stay on :class:`ODesignOutput`.
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    seed_index: int = 0
    backbone_index: int = 0
    sequence_index: int = 0


@dataclass(eq=False)
class ODesignDesignResult:
    r"""The ODesign Result for the M×S×K design grid.

    ODesign is coordinate-producing, so this result carries its coordinate-bearing candidate
    shape rather than the coordinate-free :class:`~ofoldx.results.DesignResult`.
    ``candidates`` is generation-ordered
    (seed → backbone → sequence); ``num_seeds`` (S), ``num_backbones`` (M, per-seed
    diffusion samples), and ``num_sequences`` (K, OInvFold top-k per backbone) record the
    loop structure. The ``K`` sequence variants of a given ``(seed, backbone)``
    share one backbone coordinate set. A backbone-only ``design`` call sets
    ``num_sequences=0``.
    """

    candidates: tuple[ODesignDesignCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best(self, score: str) -> ODesignDesignCandidate:
        r"""Return the earliest best candidate under one explicit score.

        Applies the shared ranking semantics: explicit ``preferred``
        direction, one identical unit, one identical ``frozenset`` scope, NaN rejection,
        earliest tie), mirroring :meth:`ofoldx.results.DesignResult.best`. There is no
        shared ranking abstraction and no default score.
        """

        return best(self.candidates, score)
