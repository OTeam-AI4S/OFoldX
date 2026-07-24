r"""
ODesign structure-generation model and tensor interfaces.
"""

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
    LocalAtomGeometry,
    NormalizedKeyAtomToTokenEncoder,
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
from ...modules.trunk import MsaTrunk, PairformerStack, TrianglePairUpdateStack
from ...results import Metric, Score, best
from .config import ODesignConfig


class ODesignModel(Model):
    r"""
    Generate all-atom biomolecular structures with ODesign.

    Accepts [`ODesignInput`][ofoldx.models.odesign.model.ODesignInput] and returns
    [`ODesignOutput`][ofoldx.models.odesign.model.ODesignOutput].

    Args:
        config: Model architecture and inference configuration.
    """

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
        r"""
        Denoise atom coordinates at specified EDM noise levels.

        ``B`` is batch size, ``S`` is the coordinate-sample axis, ``N`` is token
        count, and ``A`` is atom count.

        Args:
            input: Processed structure-design input.
            noisy_coordinates: Noisy atom coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: Diffusion noise levels with shape ``(B, S)``.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.

        Returns:
            An [`ODesignOutput`][] with denoised ``coordinates`` of shape
            ``(B, S, A, 3)``, ``distogram_logits`` of shape ``(B, N, N, D)``, and
            ``token_bond_type_logits`` of shape ``(B, N, N, D_b)``.
        """

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
        r"""
        Generate decoded structure candidates for one processed design problem.

        Args:
            input: Processed design input with batch size one and a realized fixed-length
                layout.
            num_candidates: Number of backbone candidates.
            generator: Random generator used by coordinate sampling.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of coordinate-diffusion steps.
            partial_diffusion_snr: Signal-to-noise ratio for partial diffusion from
                ``input.observed_positions``. ``None`` starts from noise.

        Returns:
            An [`ODesignDesignResult`][] containing ``num_candidates`` structures in
            sampling order. Coordinates are in angstroms.
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
        # Sequence assignment and candidate scoring are added by ODesignPipeline.
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
        r"""
        Sample coordinates while retaining the model's pairwise predictions.

        ``K`` is ``num_model_seeds``, ``B`` is batch size, ``S`` is ``num_samples``,
        ``N`` is token count, and ``A`` is atom count.

        Args:
            input: Processed design input.
            num_samples: Number of coordinate samples per model seed.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of coordinate-diffusion steps.
            num_model_seeds: Number of independent trunk evaluations.
            partial_diffusion_snr: Signal-to-noise ratio for partial diffusion from
                ``input.observed_positions``. ``None`` starts from noise.
            generator: Random generator used by coordinate sampling.

        Returns:
            An [`ODesignOutput`][] with ``coordinates`` of shape
            ``(K, B, S, A, 3)`` in angstroms, ``distogram_logits`` of shape
            ``(K, B, N, N, D)``, and ``token_bond_type_logits`` of shape
            ``(K, B, N, N, D_b)``.
        """

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
    r"""
    Aggregate reference atoms and append ODesign token features.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 386,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.atom_encoder = NormalizedKeyAtomToTokenEncoder(
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


class ODesignConstraintEmbeddings(nn.Module):
    r"""
    Embed pairwise target distances into the ODesign pair representation.
    """

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
        self.pair_norm = LayerNorm(pair_size, operator_backend=operator_backend)
        self.distogram_projection = nn.Linear(num_bins, hidden_size, bias=False)
        self.pair_update_stack = TrianglePairUpdateStack(
            num_blocks=num_blocks,
            pair_size=hidden_size,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.hidden_norm = LayerNorm(hidden_size, operator_backend=operator_backend)
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
        hidden = self.pair_update_stack(hidden, pair_mask=pair_mask)
        return self.output_projection(self.activation(self.hidden_norm(hidden)))


class ODesignTokenBondTypeHead(nn.Module):
    r"""
    Predict whether each ODesign token pair is covalently connected.
    """

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
    r"""
    Raise when a candidate violates fixed-coordinate evidence.
    """

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
    r"""
    Named representations produced by the ODesign trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_position_features: RelativePositionBuckets


@dataclass(frozen=True)
class ODesignPlacements:
    r"""
    Declare the generated and retained segments of an ODesign output.

    Each declaration group represents one output chain.

    Args:
        placements: Ordered placement declaration. Placement IDs must be globally
            unique.
        chain_lengths: Optional exact output length for each declaration group.
    """

    placements: PlacementSpec
    chain_lengths: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        # A placement_id must be globally unique across the whole PlacementSpec.
        validate_placement(self.placements)


@dataclass(frozen=True)
class ODesignReferenceCoordinates:
    r"""
    Fixed or constrained coordinates for a set of existing input atoms.

    Args:
        sites: Existing atom occurrences carrying the reference coordinates.
        tolerance_angstrom: Accepted coordinate deviation in angstroms.
        intent: ``"hard"`` fixes selected atoms during sampling; ``"soft"`` only
            supplies geometric conditioning.
    """

    sites: tuple[AtomSite, ...]
    tolerance_angstrom: float
    intent: Literal["hard", "soft"] = "hard"


@dataclass(frozen=True)
class ODesignDesignedResidue:
    r"""
    Describe one generated polymer residue in an ODesign mask.

    Args:
        polymer_type: Chemistry of the generated polymer.
    """

    polymer_type: PolymerType


@dataclass(frozen=True)
class ODesignFixedResidue:
    r"""
    Describe one output residue copied from an input structure.

    Args:
        instance_id: Source molecular instance.
        source_index: Zero-based component index within the source instance.
    """

    instance_id: str
    source_index: int


#: One designed or fixed residue in an ODesign residue mask.
ODesignResidueSpec = ODesignDesignedResidue | ODesignFixedResidue


@dataclass
class ODesignLayout:
    r"""
    Molecular identities and conditions aligned to ODesign tensor axes.
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
    r"""
    Tensor and molecular-layout input for ODesign.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, and ``M`` MSA depth.

    Args:
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        token_index: Token indices within each residue with shape ``(B, N)``.
        cyclic_period: Optional cyclic-position periods with shape ``(B, N)``.
        token_features: Per-token input features with shape ``(B, N, 453)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        atom_features: Reference atom features with shape ``(B, A, 386)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        atom_mask: Valid-atom mask with shape ``(B, A)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        observed_positions: Observed atom coordinates with shape ``(B, A, 3)`` in
            angstroms.
        observed_mask: Observed-coordinate validity mask with shape ``(B, A)``.
        fixed_coordinate_mask: Atoms fixed during sampling with shape ``(B, A)``.
        constraint_distogram: Geometric condition features with shape
            ``(B, N, N, 39)``.
        msa_features: MSA features with shape ``(B, M, N, 34)``.
        msa_mask: MSA validity mask with shape ``(B, M, N)``.
        layout: Molecular identities and conditions aligned to the token and atom axes.
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
    r"""
    Predictions from one ODesign denoising or sampling call.

    Args:
        coordinates: Denoised atom coordinates in angstroms. ``forward`` returns
            shape ``(B, S, A, 3)``; ``sample`` returns ``(K, B, S, A, 3)`` with a
            leading model-seed axis.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``
            for ``forward`` or ``(K, B, N, N, D)`` for ``sample``.
        token_bond_type_logits: Token-pair bond-type logits with shape
            ``(B, N, N, D_b)`` for ``forward`` or ``(K, B, N, N, D_b)`` for
            ``sample``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_type_logits: Tensor


@dataclass
class ODesignDesignCandidate:
    r"""
    One ODesign sample and its aligned scientific values.

    Args:
        structure: Designed molecular structure.
        scores: Scalar scores keyed by stable scientific names.
        metrics: One-axis metrics keyed by stable scientific names.
        seed_index: Zero-based generation-seed index.
        backbone_index: Zero-based backbone index within the seed.
        sequence_index: Zero-based sequence index within the backbone.
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    seed_index: int = 0
    backbone_index: int = 0
    sequence_index: int = 0


@dataclass(eq=False)
class ODesignDesignResult:
    r"""
    The seed-by-backbone-by-sequence ODesign result.

    Args:
        candidates: Candidates ordered by seed, backbone, then sequence.
        num_seeds: Number of generation seeds.
        num_backbones: Number of sampled backbones per seed.
        num_sequences: Number of sequence variants per backbone. Zero denotes a
            backbone-only result.
    """

    candidates: tuple[ODesignDesignCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best(self, score: str) -> ODesignDesignCandidate:
        r"""
        Return the earliest best candidate under a named score.

        Args:
            score: Comparable score name present on every candidate.

        Returns:
            The best candidate, with generation order breaking ties.
        """

        return best(self.candidates, score)
