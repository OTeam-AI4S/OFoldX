r"""The released ODesign core graph and its canonical task methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, cast, get_args

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import Self

from ...data.bonds import SymbolicBond
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MSACollection

# ODesign's spatial and placement records drive the hard-condition checks below.
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    SystemBond,
)
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
from ...modules.adaptive_norm import AdaptiveLayerNorm
from ...modules.atom import (
    AtomToTokenEncoder,
    AtomToTokenEncoderBase,
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
from ...modules.transition import Transition
from ...modules.trunk import MsaTrunk, PairformerStack
from ...modules.trunk.msa import MsaPairWeightedAveraging, OuterProductMean
from ...results import DesignCandidate, Metric
from .config import ODesignConfig


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
            cross_attention_key_source="normalized",
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
                    AdaptiveLayerNorm,
                    AtomToTokenEncoderBase,
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
        input: ODesignInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        recycling_steps: int | None = None,
    ) -> ODesignOutput:
        r"""Denoise caller-supplied coordinates at one caller-supplied EDM noise level."""

        steps = self.recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        token_embedding, single, pair, relative_features = self._run_trunk(input, recycling_steps=steps)
        context = self._denoise_context(input, token_embedding, single, pair, relative_features)

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

        result, _ = self._design_with_input(
            input,
            num_candidates=num_candidates,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            partial_diffusion_snr=partial_diffusion_snr,
        )
        return result

    @torch.inference_mode()
    def _design_with_input(
        self,
        input: ODesignInput,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> tuple[ODesignDesignResult, ODesignInput]:
        r"""Design candidates and retain the realized variable-length semantic input."""

        input = _realize_ranges(input, generator)
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
        # The shared DesignCandidate is coordinate-free. ODesign is coordinate-generating, so each sample
        # is carried by ODesignDesignCandidate, which adds the
        # decoded Structure and a per-sample metrics map (empty: the released core exposes no
        # per-sample confidence head). ``system`` mirrors ``structure.system``. Raw
        # distogram/token-bond logits stay on the ODesignOutput; run provenance and
        # generated-identity maps are not carried here.
        # One Model.design call produces one seed's M backbones (num_model_seeds=1). The
        # ODesignDesignResult carries these coordinate-bearing candidates.
        # This is the backbone-only stage, so
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
        return (
            ODesignDesignResult(
                candidates=candidates,
                num_seeds=1,
                num_backbones=num_candidates,
                num_sequences=0,
            ),
            input,
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
        r"""Run the released multi-seed sampler without discarding its pairwise heads."""

        steps = self.recycling_steps if recycling_steps is None else recycling_steps
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
            token_embedding, single, pair, relative_features = self._run_trunk(input, recycling_steps=steps)
            distograms.append(self.distogram_head(pair))
            bond_logits.append(self.token_bond_type_head(pair))
            coordinates.append(
                self.coordinate_sampler(
                    self.coordinate_denoiser,
                    context=self._denoise_context(input, token_embedding, single, pair, relative_features),
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
    ) -> tuple[Tensor, Tensor, Tensor, RelativePositionBuckets]:
        local_geometry = self.input_embedder.build_local_geometry(
            input.reference_positions,
            input.reference_space_id,
        )
        token_embedding = self.input_embedder(
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
                pair = pair + self.constraint_distogram_embedder(
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
        return token_embedding, single, pair, relative_features

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
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> CoordinateDenoiseContext:
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                input.reference_positions,
                input.reference_space_id,
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
    sites = tuple(site for site, atom in layout.system.iter_atoms() if atom.element != "H")
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


def _require_hard_conditions(structure: Structure, input: ODesignInput) -> None:
    r"""Raise when a candidate violates fixed-coordinate evidence."""

    source_residues = input.layout.source_residue_map
    placement_map = input.layout.placement_map
    instance_of = input.layout.instance_of
    source_atoms = input.layout.source_atom_map
    coordinates = {site: structure.coordinates[index] for index, site in enumerate(structure.sites)}
    reference = {
        site: input.observed_positions[0, index].detach().to(device="cpu", dtype=torch.float32).numpy()
        for index, site in enumerate(input.layout.sites)
    }
    for index, condition in enumerate(input.layout.conditions):
        if not isinstance(condition, ReferenceCoordinates) or condition.intent != "hard":
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
class ODesignLayout:
    r"""Canonical identity needed to turn ODesign coordinates into Results.

    ``placement_map`` is the realized output surface for generated placements
    (``declaration_group -> placement_id -> ordered ComponentReference occurrences``); it is
    resolved together with ``instance_of`` (``placement_id -> declaration_group``, derived
    once from the PlacementSpec) via ``ofoldx.data.placement.resolve``. There is exactly one canonical
    realized map and one resolution helper — no parallel positive/negative alias table.
    """

    system: MolecularSystem
    sites: tuple[AtomSite, ...]
    placement: ODesignPlacements | None
    conditions: tuple[Hotspot | SymbolicBond | ReferenceCoordinates, ...]
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

    ODesign generates coordinates, so its candidate carries a decoded
    :class:`~ofoldx.data.structure.Structure` and per-sample ``metrics`` alongside the
    shared coordinate-free ``system`` and ``scores``. It subclasses
    :class:`~ofoldx.results.DesignCandidate` so :class:`~ofoldx.results.DesignResult` can carry
    it unchanged. The released ODesign core exposes no per-sample confidence head, so ``metrics``
    is empty today; the distogram / token-bond logits stay on :class:`ODesignOutput`.
    """

    structure: Structure | None = None
    metrics: dict[str, Metric] = field(default_factory=dict)
    seed_index: int = 0
    backbone_index: int = 0
    sequence_index: int = 0


@dataclass(eq=False)
class ODesignDesignResult:
    r"""The ODesign Result for the released M×S×K design grid.

    ODesign is coordinate-producing, so this result carries its coordinate-bearing candidate
    shape rather than the coordinate-free :class:`~ofoldx.results.DesignResult`.
    ``candidates`` is generation-ordered
    (seed → backbone → sequence); ``num_seeds`` (S), ``num_backbones`` (M, per-seed
    diffusion samples), and ``num_sequences`` (K, OInvFold top-k per backbone) record the
    released loop structure. The ``K`` sequence variants of a given ``(seed, backbone)``
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


# There is no shared ``Condition`` base, union endpoint, or generic condition bag. ODesign owns
# the concrete spatial records it consumes; shared identity conditions remain in
# ``ofoldx.data.conditions``.
#
# The generation-topology placement records (:class:`~ofoldx.data.placement.GeneratedSegment`,
# :class:`~ofoldx.data.placement.ExistingSpan`, :class:`~ofoldx.data.placement.SymbolicResidueSite`,
# :class:`~ofoldx.data.placement.SymbolicAtomSite`) come from ``ofoldx.data.placement``. Identity is
# placement-keyed: a placement's identity is the globally-unique ``placement_id`` inner key of
# the core :data:`~ofoldx.data.placement.PlacementSpec` mapping, so :class:`GeneratedSegment` carries NO
# embedded id and a pre-realization reference is exactly the two fields
# ``SymbolicResidueSite(placement_id, offset)``. :class:`ODesignPlacements` adds per-chain
# ``chain_lengths`` alongside the core placement mapping.
#
# The covalent-link record (:class:`~ofoldx.data.bonds.SymbolicBond`
# over the promoted :data:`~ofoldx.data.bonds.SymbolicBondSite` endpoint union) and the
# target-conditioning core (the distance-free :class:`~ofoldx.data.conditions.Hotspot` label)
# are shared and imported directly.
# ======================================================================================


# --------------------------------------------------------------------------------------
# ODesign placement condition.
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# ODesign spatial conditions.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCoordinates:
    r"""Fixed or constrained coordinates for a set of existing input atoms.

    ``intent == "hard"`` fixes the coordinates during EDM sampling; both modes feed the
    learned constraint distogram.
    """

    sites: tuple[AtomSite, ...]
    tolerance_angstrom: float
    intent: Literal["hard", "soft"] = "hard"


# --------------------------------------------------------------------------------------
# design_mask <-> PlacementSpec projection.
# --------------------------------------------------------------------------------------
#
# ODesign natively selects fixed-vs-generated positions per output residue through a
# per-residue design mask (the ``res`` condition-mask driven by the ``sequence`` tokens:
# an ``N-M`` / ``N-N`` run designs a de-novo polymer interval, an ``X/A-B`` token copies a
# source span). These two functions are the lossless bridge between that native
# per-residue design-mask surface and the run-length
# :data:`~ofoldx.data.placement.PlacementSpec` representation.

#: The core :data:`~ofoldx.data.molecular.PolymerType` values a generated polymer placement
#: may take. Small-molecule generation is deliberately absent because the shared placement core
#: represents polymers; ODesign's released placeholder-ligand rows are not a chemical graph.
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
    all recoverable (see :func:`to_design_mask` for the inverse). A designed non-polymer
    (ligand ``num_atoms``) residue has no ``PlacementValue`` and is
    rejected here rather than silently dropped. The partition is exhaustive — every input
    residue belongs to exactly one placement, so no realized row is unaddressable.
    """

    placement: PlacementSpec = {}
    seen: set[str] = set()

    current_group: str | None = None
    current_placement_id: str | None = None
    current_polymer_type: PolymerType | None = None
    current_instance_id: str | None = None
    run_start_source: int = 0
    run_next_source: int = 0
    run_length: int = 0

    def flush() -> None:
        if current_placement_id is None:
            return
        assert current_group is not None
        if current_polymer_type is not None:
            value: GeneratedSegment | ExistingSpan = GeneratedSegment(current_polymer_type, run_length, run_length)
        else:
            assert current_instance_id is not None
            value = ExistingSpan(current_instance_id, run_start_source, run_next_source)
        placement.setdefault(current_group, {})[current_placement_id] = value

    for index, residue in enumerate(residues):
        group = group_of(index)
        placement_id = placement_id_of(index)
        if isinstance(residue, DesignedResidue):
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
            assert isinstance(residue, FixedResidue)
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
    # Enforce globally unique placement IDs across the PlacementSpec.
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
    realized residue with no gap or overlap; zero-length placements contribute no rows. Thus
    ``design_mask`` and ``groups`` partition the whole realized layout.
    """

    design_mask: list[bool] = []
    groups: list[str] = []
    for group, chain in placement.items():
        for placement_id, value in chain.items():
            if isinstance(value, GeneratedSegment):
                length = len(placement_map[group][placement_id])
                designed = True
            else:
                length = value.stop - value.start
                designed = False
            design_mask.extend([designed] * length)
            groups.extend([group] * length)
    return design_mask, groups


ODESIGN_RESTYPE_INDEX: dict[str, int] = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
    "-": 31,
    "-P": 32,
    "-N": 33,
    "-L": 34,
}
ODESIGN_STANDARD_POLYMER_COMPONENTS: frozenset[str] = frozenset(ODESIGN_RESTYPE_INDEX) - {
    "-",
    "-P",
    "-N",
    "-L",
}


def materialize_odesign_cyclic_links(
    system: MolecularSystem,
    conditions: tuple[Hotspot | SymbolicBond | ReferenceCoordinates, ...],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
    source_atom_map: dict[AtomSite, AtomSite],
) -> tuple[MolecularSystem, frozenset[str], frozenset[frozenset[AtomSite]]]:
    r"""Materialize requested head-to-tail polymer closures."""

    source_atoms = source_atom_map
    bonds = set(system.bonds)
    cyclic_instances = set()
    cyclic_links = set()
    for existing_bond in bonds:
        if existing_bond.site_a.instance_id != existing_bond.site_b.instance_id:
            continue
        instance_id = existing_bond.site_a.instance_id
        entity = system.entities[system.instances[instance_id]]
        if not isinstance(entity, PolymerSequence):
            continue
        first = entity.components[0].previous_attachment
        last = entity.components[-1].next_attachment
        if first is None or last is None:
            continue
        expected = frozenset(
            (
                AtomSite(instance_id, 0, first.atom_id),
                AtomSite(instance_id, len(entity) - 1, last.atom_id),
            )
        )
        if frozenset((existing_bond.site_a, existing_bond.site_b)) == expected:
            cyclic_instances.add(instance_id)
            cyclic_links.add(expected)
    for condition in conditions:
        if not isinstance(condition, SymbolicBond):
            continue
        # Endpoints ride on the SymbolicBond (site_a/site_b over the
        # promoted SymbolicBondSite union). An existing-atom endpoint (AtomSite) is remapped
        # source -> output through source_atom_map; a generated-atom endpoint (SymbolicAtomSite)
        # resolves its SymbolicResidueSite through the shared placement resolver.
        symbolic_bond = condition
        endpoints: list[AtomSite] = []
        for reference in (symbolic_bond.site_a, symbolic_bond.site_b):
            if isinstance(reference, AtomSite):
                site = source_atoms.get(reference)
                if site is None:
                    raise ValueError(f"ODesign atom layout does not contain covalent-link site {reference!r}")
            else:
                assert isinstance(reference, SymbolicAtomSite)
                residue = _resolve_site(reference.residue, placement_map, instance_of)
                site = AtomSite(residue[0], residue[1], reference.atom_id)
            system.atom(site)
            endpoints.append(site)

        site_a, site_b = endpoints
        if site_a.instance_id != site_b.instance_id:
            raise ValueError("released ODesign supports only head-to-tail cyclic polymer links")
        instance_id = site_a.instance_id
        entity = system.entities[system.instances[instance_id]]
        if not isinstance(entity, PolymerSequence):
            raise ValueError("released ODesign supports only head-to-tail cyclic polymer links")
        previous = entity.components[0].previous_attachment
        following = entity.components[-1].next_attachment
        if previous is None or following is None:
            raise ValueError("cyclic polymer endpoints require previous and next attachments")
        expected = frozenset(
            (
                AtomSite(instance_id, 0, previous.atom_id),
                AtomSite(instance_id, len(entity) - 1, following.atom_id),
            )
        )
        endpoint_set = frozenset((site_a, site_b))
        if endpoint_set != expected:
            raise ValueError("released ODesign supports only head-to-tail cyclic polymer links")

        existing_bonds = tuple(
            candidate for candidate in bonds if frozenset((candidate.site_a, candidate.site_b)) == endpoint_set
        )
        requested_chemistry = (symbolic_bond.order, symbolic_bond.aromatic, symbolic_bond.stereo, None)
        for candidate in existing_bonds:
            if (candidate.order, candidate.aromatic, candidate.stereo, candidate.stereo_sites) != requested_chemistry:
                raise ValueError("conflicting ODesign cyclic bond already exists at the requested endpoints")
        if not existing_bonds:
            bonds.add(
                SystemBond(
                    site_a,
                    site_b,
                    order=symbolic_bond.order,
                    aromatic=symbolic_bond.aromatic,
                    stereo=symbolic_bond.stereo,
                    origin="user",
                )
            )
        cyclic_instances.add(instance_id)
        cyclic_links.add(endpoint_set)

    if not cyclic_instances:
        return system, frozenset(), frozenset()
    return (
        MolecularSystem(
            system.entities,
            system.instances,
            frozenset(bonds),
        ),
        frozenset(cyclic_instances),
        frozenset(cyclic_links),
    )


def _realize_ranges(input: ODesignInput, generator: torch.Generator | None) -> ODesignInput:
    placement = input.layout.placement
    if placement is None:
        return input
    ranged = {
        placement_id: part
        for chain in placement.placements.values()
        for placement_id, part in chain.items()
        if isinstance(part, GeneratedSegment) and part.min_length != part.max_length
    }
    if not ranged:
        return input
    lengths = {}
    chain_lengths = placement.chain_lengths or (None,) * len(placement.placements)
    for chain, chain_length in zip(placement.placements.values(), chain_lengths):
        chain_ranges = [
            (placement_id, part)
            for placement_id, part in chain.items()
            if isinstance(part, GeneratedSegment) and placement_id in ranged
        ]
        if chain_length is None:
            for placement_id, segment in chain_ranges:
                lengths[placement_id] = int(
                    torch.randint(
                        segment.min_length,
                        segment.max_length + 1,
                        (),
                        device=input.reference_positions.device,
                        generator=generator,
                    )
                )
            continue

        for placement_id, segment in chain_ranges:
            lengths[placement_id] = segment.min_length
        remaining = chain_length - sum(
            part.min_length if isinstance(part, GeneratedSegment) else part.stop - part.start for part in chain.values()
        )
        for _ in range(remaining):
            available = [
                (placement_id, segment)
                for placement_id, segment in chain_ranges
                if lengths[placement_id] < segment.max_length
            ]
            selected = int(
                torch.randint(
                    len(available),
                    (),
                    device=input.reference_positions.device,
                    generator=generator,
                )
            )
            lengths[available[selected][0]] += 1
    if all(length == ranged[placement_id].max_length for placement_id, length in lengths.items()):
        return input

    polymer_instances = [
        instance_id
        for instance_id, entity_id in input.layout.system.instances.items()
        if isinstance(input.layout.system.entities[entity_id], PolymerSequence)
    ]
    residue_map: dict[tuple[str, int], tuple[str, int]] = {}
    placement_map: PlacementMap = {}
    realized_sequences: list[tuple[str, str, PolymerSequence]] = []
    sequence_by_signature: dict[
        tuple[PolymerType, tuple[ChemicalComponent, ...]],
        tuple[str, PolymerSequence],
    ] = {}
    used_entity_ids = {
        entity_id for entity_id, entity in input.layout.system.entities.items() if isinstance(entity, Molecule)
    }
    # Walk declaration groups in order, indexing the realized polymer instances by declaration
    # position among NON-EMPTY groups only: a zero-length declaration group mints no realized
    # chain (matching _materialize_placement), so it is recorded as () and skipped here too.
    chain_position = 0
    for group, chain in placement.placements.items():
        if all(
            part.max_length == 0 if isinstance(part, GeneratedSegment) else part.stop == part.start
            for part in chain.values()
        ):
            for placement_id, part in chain.items():
                if isinstance(part, GeneratedSegment):
                    placement_map.setdefault(group, {})[placement_id] = ()
            continue
        instance_id = polymer_instances[chain_position]
        chain_position += 1
        old_entity_id = input.layout.system.instances[instance_id]
        old_entity = cast(PolymerSequence, input.layout.system.entities[old_entity_id])
        old_index = 0
        new_index = 0
        components: list[ChemicalComponent] = []
        for placement_id, part in chain.items():
            old_length = part.max_length if isinstance(part, GeneratedSegment) else part.stop - part.start
            new_length = (
                lengths.get(placement_id, part.max_length) if isinstance(part, GeneratedSegment) else old_length
            )
            occurrences: list[tuple[str, int]] = []
            for offset in range(new_length):
                residue_map[(instance_id, old_index + offset)] = (instance_id, new_index + offset)
                components.append(old_entity.components[old_index + offset])
                if isinstance(part, GeneratedSegment):
                    occurrences.append((instance_id, new_index + offset))
            if isinstance(part, GeneratedSegment):
                placement_map.setdefault(group, {})[placement_id] = tuple(occurrences)
            old_index += old_length
            new_index += new_length

        signature = (old_entity.polymer_type, tuple(components))
        existing_entity = sequence_by_signature.get(signature)
        if existing_entity is None:
            entity_id = old_entity_id
            while entity_id in used_entity_ids:
                entity_id += "_"
            sequence = PolymerSequence(old_entity.polymer_type, tuple(components))
            sequence_by_signature[signature] = (entity_id, sequence)
            used_entity_ids.add(entity_id)
        else:
            entity_id, sequence = existing_entity
        realized_sequences.append((instance_id, entity_id, sequence))

    molecule_entities = {
        entity_id: entity for entity_id, entity in input.layout.system.entities.items() if isinstance(entity, Molecule)
    }
    molecule_instances = {
        instance_id: entity_id
        for instance_id, entity_id in input.layout.system.instances.items()
        if isinstance(input.layout.system.entities[entity_id], Molecule)
    }
    instances: dict[str, str] = {
        **{instance_id: entity_id for instance_id, entity_id, _ in realized_sequences},
        **molecule_instances,
    }
    entities: dict[str, PolymerSequence | Molecule] = {
        **{entity_id: sequence for _, entity_id, sequence in realized_sequences},
        **molecule_entities,
    }

    def map_site(site: AtomSite) -> AtomSite | None:
        if isinstance(input.layout.system.entities[input.layout.system.instances[site.instance_id]], Molecule):
            return site
        residue = residue_map.get((site.instance_id, site.component_index))
        return None if residue is None else AtomSite(residue[0], residue[1], site.atom_id)

    bonds = []
    for bond in input.layout.system.bonds:
        site_a = map_site(bond.site_a)
        site_b = map_site(bond.site_b)
        if site_a is None or site_b is None:
            continue
        stereo_sites = None
        if bond.stereo_sites is not None:
            stereo_a = map_site(bond.stereo_sites[0])
            stereo_b = map_site(bond.stereo_sites[1])
            if stereo_a is None or stereo_b is None:
                continue
            stereo_sites = (stereo_a, stereo_b)
        bonds.append(
            SystemBond(
                site_a,
                site_b,
                order=bond.order,
                aromatic=bond.aromatic,
                stereo=bond.stereo,
                stereo_sites=stereo_sites,
                origin=bond.origin,
            )
        )
    system = MolecularSystem(entities, instances, frozenset(bonds))
    source_residue_map = {
        source: residue_map[output]
        for source, output in input.layout.source_residue_map.items()
        if output in residue_map
    }
    source_atom_map = {
        source: mapped
        for source, output in input.layout.source_atom_map.items()
        if (mapped := map_site(output)) is not None
    }
    system, cyclic_instances, cyclic_links = materialize_odesign_cyclic_links(
        system,
        input.layout.conditions,
        placement_map,
        input.layout.instance_of,
        source_atom_map,
    )

    old_to_new_site = {site: mapped for site in input.layout.sites if (mapped := map_site(site)) is not None}
    sites = tuple(old_to_new_site.values())
    new_to_old_site = {new: old for old, new in old_to_new_site.items()}
    old_atom_index = {site: index for index, site in enumerate(input.layout.sites)}
    atom_indices = torch.tensor(
        [old_atom_index[new_to_old_site[site]] for site in sites],
        dtype=torch.int64,
        device=input.atom_mask.device,
    )
    token_sites, token_residues, token_entities = build_odesign_tokens(system, sites)
    old_token_indices = torch.tensor(
        [int(input.atom_to_token[0, old_atom_index[new_to_old_site[group[0]]]]) for group in token_sites],
        dtype=torch.int64,
        device=input.token_mask.device,
    )
    atom_to_token = torch.tensor(
        [token_index for token_index, group in enumerate(token_sites) for _ in group],
        dtype=torch.int64,
        device=input.atom_to_token.device,
    ).unsqueeze(0)
    site_to_token = {site: token_index for token_index, group in enumerate(token_sites) for site in group}
    chain_index, entity_index, symmetry_index, residue_index = (
        value.unsqueeze(0).to(input.residue_index.device)
        for value in build_odesign_token_indices(system, token_sites, token_residues, token_entities)
    )
    selected_reference_spaces = input.reference_space_id.index_select(1, atom_indices)
    reference_space_map: dict[int, int] = {}
    reference_space_id: list[int] = []
    for value in selected_reference_spaces[0].tolist():
        if value not in reference_space_map:
            reference_space_map[value] = len(reference_space_map)
        reference_space_id.append(reference_space_map[value])
    reference_space_id = torch.tensor(
        reference_space_id,
        dtype=torch.int64,
        device=input.reference_space_id.device,
    ).unsqueeze(0)
    observed_positions = input.observed_positions.index_select(1, atom_indices)
    observed_mask = input.observed_mask.index_select(1, atom_indices)
    fixed_coordinate_mask = input.fixed_coordinate_mask.index_select(1, atom_indices)
    token_bonds = build_odesign_token_bonds(
        system,
        token_sites,
        site_to_token,
        excluded=cyclic_links,
    ).to(input.token_bonds)
    constraint_distogram = input.constraint_distogram.index_select(1, old_token_indices).index_select(
        2,
        old_token_indices,
    )
    layout = replace(
        input.layout,
        system=system,
        sites=sites,
        placement_map=placement_map,
        source_residue_map=source_residue_map,
        source_atom_map=source_atom_map,
    )
    return replace(
        input,
        token_mask=input.token_mask.index_select(1, old_token_indices),
        residue_index=residue_index,
        chain_index=chain_index,
        entity_index=entity_index,
        symmetry_index=symmetry_index,
        token_index=torch.arange(len(token_sites), device=input.token_index.device).unsqueeze(0),
        cyclic_period=build_odesign_cyclic_period(token_sites, cyclic_instances)
        .unsqueeze(0)
        .to(input.cyclic_period.device),
        token_features=input.token_features.index_select(1, old_token_indices),
        token_bonds=token_bonds.unsqueeze(0),
        atom_features=input.atom_features.index_select(1, atom_indices),
        atom_to_token=atom_to_token,
        atom_mask=input.atom_mask.index_select(1, atom_indices),
        reference_positions=input.reference_positions.index_select(1, atom_indices),
        reference_space_id=reference_space_id,
        reference_mask=input.reference_mask.index_select(1, atom_indices),
        reference_charges=input.reference_charges.index_select(1, atom_indices),
        observed_positions=observed_positions,
        observed_mask=observed_mask,
        fixed_coordinate_mask=fixed_coordinate_mask,
        constraint_distogram=constraint_distogram,
        msa_features=input.msa_features.index_select(2, old_token_indices),
        msa_mask=input.msa_mask.index_select(2, old_token_indices),
        layout=layout,
    )


def build_odesign_tokens(
    system: MolecularSystem,
    sites: tuple[AtomSite, ...],
) -> tuple[
    tuple[tuple[AtomSite, ...], ...],
    tuple[tuple[str, int] | None, ...],
    tuple[PolymerSequence | Molecule, ...],
]:
    r"""Build the ODesign token groups and their molecular identities."""

    sites_by_residue: dict[tuple[str, int], list[AtomSite]] = {}
    for site in sites:
        if isinstance(system.entities[system.instances[site.instance_id]], PolymerSequence):
            sites_by_residue.setdefault((site.instance_id, site.component_index), []).append(site)
    token_sites: list[tuple[AtomSite, ...]] = []
    token_residues: list[tuple[str, int] | None] = []
    token_entities: list[PolymerSequence | Molecule] = []
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            for component_index in range(len(entity)):
                residue = (instance_id, component_index)
                residue_sites = tuple(sites_by_residue[residue])
                if entity.components[component_index].component_id in ODESIGN_STANDARD_POLYMER_COMPONENTS:
                    token_sites.append(residue_sites)
                    token_residues.append(residue)
                    token_entities.append(entity)
                else:
                    for site in residue_sites:
                        token_sites.append((site,))
                        token_residues.append(residue)
                        token_entities.append(entity)
        else:
            for site in sites:
                if site.instance_id == instance_id:
                    token_sites.append((site,))
                    token_residues.append(None)
                    token_entities.append(entity)
    return tuple(token_sites), tuple(token_residues), tuple(token_entities)


def build_odesign_token_indices(
    system: MolecularSystem,
    token_sites: Sequence[Sequence[AtomSite]],
    token_residues: Sequence[tuple[str, int] | None],
    token_entities: Sequence[PolymerSequence | Molecule],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Build ODesign chain, entity, symmetry, and residue index tensors."""

    chain_by_instance = {instance_id: index for index, instance_id in enumerate(system.instances)}
    entity_ids = []
    for entity_id in system.instances.values():
        if entity_id not in entity_ids:
            entity_ids.append(entity_id)
    entity_index_by_id = {value: index for index, value in enumerate(entity_ids)}
    seen_by_entity: dict[str, int] = {}
    symmetry_by_instance = {}
    for instance_id, entity_id in system.instances.items():
        symmetry_by_instance[instance_id] = seen_by_entity.get(entity_id, 0)
        seen_by_entity[entity_id] = symmetry_by_instance[instance_id] + 1
    chain = []
    entity = []
    symmetry = []
    residue = []
    for group, reference, _definition in zip(token_sites, token_residues, token_entities):
        instance_id = reference[0] if reference is not None else group[0].instance_id
        chain.append(chain_by_instance[instance_id])
        entity.append(entity_index_by_id[system.instances[instance_id]])
        symmetry.append(symmetry_by_instance[instance_id])
        if reference is None:
            residue.append(0)
        else:
            residue.append(reference[1])
    return tuple(torch.tensor(values, dtype=torch.int64) for values in (chain, entity, symmetry, residue))


def build_odesign_token_bonds(
    system: MolecularSystem,
    token_sites: Sequence[Sequence[AtomSite]],
    site_to_token: dict[AtomSite, int],
    *,
    excluded: frozenset[frozenset[AtomSite]],
) -> Tensor:
    r"""Build the symmetric ODesign token-bond adjacency tensor."""

    bonds = torch.zeros(len(token_sites), len(token_sites), dtype=torch.float32)
    for bond in system.iter_bonds():
        if frozenset((bond.site_a, bond.site_b)) in excluded:
            continue
        if bond.site_a not in site_to_token or bond.site_b not in site_to_token:
            continue
        left, right = site_to_token[bond.site_a], site_to_token[bond.site_b]
        if left != right:
            bonds[left, right] = bonds[right, left] = 1.0
    return bonds


def build_odesign_cyclic_period(
    token_sites: Sequence[Sequence[AtomSite]],
    cyclic_instances: frozenset[str],
) -> Tensor:
    r"""Build the cyclic-polymer period associated with each ODesign token."""

    counts = {
        instance_id: sum(group[0].instance_id == instance_id for group in token_sites)
        for instance_id in cyclic_instances
    }
    return torch.tensor(
        [counts.get(group[0].instance_id, 0) for group in token_sites],
        dtype=torch.int64,
    )
