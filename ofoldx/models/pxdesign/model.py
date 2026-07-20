r"""Native implementation of the released PxDesign v0.1 model graph."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Self

from ...data import Hotspot, PlacementMap, PlacementSpec
from ...data.molecular import AtomSite, MolecularSystem
from ...data.structure import Structure
from ...modules.adaptive_norm import AdaptiveLayerNorm
from ...modules.atom import AtomToTokenEncoder, TokenPairToLocalAtomPairBroadcast, TokenToAtomBroadcast
from ...modules.attention import GatedMultiheadAttention
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import CoordinateDenoiseContext, EdmScheduler
from ...modules.embedding import RelativePositionBuckets, generate_relative_position_features
from ...modules.geometry import uniform_random_rotation
from ...modules.transformer import ConditionedTransformerLayer, ProtenixConditionedTransformerLayer
from ...modules.transition import Transition
from ...results import Metric, Score
from .config import PxDesignConfig


class PxDesignModel(nn.Module):
    r"""PxDesign's released condition embedder and coordinate-diffusion model."""

    config_class = PxDesignConfig

    def __init__(self, config: PxDesignConfig | None = None) -> None:
        config = PxDesignConfig() if config is None else config
        super().__init__()
        self.config = config
        self.condition_embedder = PxDesignConditionEmbedder(config)

        denoiser = config.denoiser
        self.coordinate_denoiser = CoordinateDenoiser(
            num_encoder_blocks=denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=denoiser.num_token_encoder_attention_heads,
            sigma_data=denoiser.sigma_data,
            atom_hidden_size=denoiser.atom_hidden_size,
            atom_pair_hidden_size=denoiser.atom_pair_hidden_size,
            token_hidden_size=denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=denoiser.num_query_atoms,
            num_key_atoms=denoiser.num_key_atoms,
        )
        self.coordinate_sampler = PxDesignSampler(config)
        # PXDesign drives Protenix's own ``DiffusionModule``/``AtomAttentionEncoder`` upstream
        # (PXDesign/pxdesign/model/pxdesign.py:20,45; embedders.py:21), so it follows Protenix's
        # double-adaLN cross-attention: the key/value adaptive norm consumes the already query-normed
        # activation. The shared atom encoder/decoder build the AF3-canonical base layer, so re-cast
        # every cross-attention atom transformer onto the Protenix-lineage subclass.
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
                    AdaptiveLayerNorm,
                    GatedMultiheadAttention,
                    TokenPairToLocalAtomPairBroadcast,
                    TokenToAtomBroadcast,
                    Transition,
                ),
            ):
                module.operator_backend = operator_backend
        return self

    def forward(
        self,
        batch: PxDesignInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
    ) -> PxDesignOutput:
        r"""Denoise caller-supplied coordinates at a caller-supplied EDM noise level."""

        token_embedding, single, pair, relative_features = self.encode_condition(batch)
        context = self._denoise_context(batch, token_embedding, single, pair, relative_features)
        coordinates = self.coordinate_denoiser(
            context.to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return PxDesignOutput(coordinates)

    @torch.inference_mode()
    def sample(
        self,
        batch: PxDesignInput,
        *,
        num_samples: int = 100,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        generator: torch.Generator | None = None,
    ) -> PxDesignOutput:
        r"""Run the released PxDesign EDM sampler and return raw model coordinates."""

        batch = _augment_reference_conformers(batch, generator)
        token_embedding, single, pair, relative_features = self.encode_condition(batch)
        context = self._denoise_context(batch, token_embedding, single, pair, relative_features)
        coordinates = self.coordinate_sampler(
            self.coordinate_denoiser,
            context=context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
        return PxDesignOutput(coordinates)

    @torch.inference_mode()
    def design(
        self,
        batch: PxDesignInput,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
    ) -> PxDesignResult:
        r"""Generate canonical target--binder backbone Structure candidates.

        The caller-supplied ``generator`` (resolved once at the Pipeline boundary) is the
        sole random stream; per-sample EDM sampling remains model-intrinsic.
        """

        if batch.reference_positions.shape[:-2] != (1,):
            raise ValueError("PxDesignModel.design requires exactly one biological input")
        output = self.sample(
            batch,
            num_samples=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
        candidates = tuple(
            PxDesignCandidate(
                structure=Structure(
                    system=batch.layout.system,
                    sites=batch.layout.sites,
                    coordinates=output.coordinates[0, index].detach().to(device="cpu", dtype=torch.float32).numpy(),
                    coordinate_mask=np.ones(len(batch.layout.sites), dtype=np.bool_),
                ),
            )
            for index in range(num_candidates)
        )
        # §5.3 reopened + LOCKED: PxDesign is coordinate-producing, so each backbone sample
        # is wired into a standalone family-local PxDesignCandidate (carrying the decoded
        # Structure, which owns its MolecularSystem) collected into the family-owned
        # PxDesignResult, NOT the frozen coordinate-free shared DesignResult (whose §5.3.1
        # fixed-length slice excludes PXDesign) and NOT by widening the shared record (§5.3.3).
        # The released sampler decodes no per-sample score or metric (its plddt input is
        # zeros and it has no confidence head), so scores/metrics stay empty. Per-condition
        # PlacementSpec satisfaction flags carried no measured value and drove no hard-constraint
        # raise here, so they are dropped with the removed ConditionOutcome carrier.
        return PxDesignResult(candidates=candidates)

    def encode_condition(
        self,
        batch: PxDesignInput,
    ) -> tuple[Tensor, Tensor, Tensor, RelativePositionBuckets]:
        r"""Compute the released token, zero-single, pair, and relative conditioning."""

        token_embedding, pair = self.condition_embedder(batch)
        single = token_embedding.new_zeros(*token_embedding.shape[:-1], self.config.single_size)
        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            dtype=token_embedding.dtype,
        )
        return token_embedding, single, pair, relative_features

    def _denoise_context(
        self,
        batch: PxDesignInput,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> CoordinateDenoiseContext:
        r"""Assemble the conditioning shared by ``forward`` and ``sample``."""

        local_geometry = self.coordinate_denoiser.build_local_geometry(
            batch.reference_positions,
            batch.reference_space_id,
        )
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
            local_geometry=local_geometry,
        )


class PxDesignConditionEmbedder(nn.Module):
    r"""Released PxDesign atom, token, and conditional-template embedder."""

    def __init__(self, config: PxDesignConfig) -> None:
        super().__init__()
        embedder = config.condition_embedder
        self.atom_encoder = AtomToTokenEncoder(
            token_hidden_size=embedder.token_hidden_size,
            atom_hidden_size=embedder.atom_hidden_size,
            atom_pair_hidden_size=embedder.atom_pair_hidden_size,
            num_blocks=embedder.num_atom_encoder_blocks,
            num_attention_heads=embedder.num_atom_encoder_attention_heads,
            num_query_atoms=embedder.num_query_atoms,
            num_key_atoms=embedder.num_key_atoms,
            atom_feature_size=config.atom_feature_size,
        )
        input_size = (
            embedder.token_hidden_size
            + embedder.vocab_size
            + embedder.plddt_size
            + embedder.hotspot_size
            + 2 * embedder.add_feature_size
        )
        self.input_map = nn.Linear(input_size, config.token_feature_size)
        self.template_embedder = PxDesignConditionalTemplateEmbedder(
            num_template_bins=config.num_template_bins,
            pair_size=config.pair_size,
        )
        self.add_feature_size = embedder.add_feature_size

    def forward(self, batch: PxDesignInput) -> tuple[Tensor, Tensor]:
        r"""Return the released token and conditional-template pair embeddings."""

        local_geometry = self.atom_encoder.build_local_geometry(
            batch.reference_positions,
            batch.reference_space_id,
        )
        atom_output = self.atom_encoder(
            batch.atom_features,
            reference_positions=batch.reference_positions,
            reference_charges=torch.arcsinh(batch.reference_charges),
            reference_mask=batch.reference_mask,
            atom_mask=batch.atom_mask,
            atom_to_token=batch.atom_to_token,
            num_tokens=batch.restype.shape[-2],
            local_geometry=local_geometry,
        )
        add_feature = F.one_hot(
            torch.zeros(batch.restype.shape[:-1], device=batch.restype.device, dtype=torch.long),
            num_classes=self.add_feature_size,
        ).to(dtype=atom_output.token_hidden.dtype)
        token_embedding = self.input_map(
            torch.cat(
                (
                    atom_output.token_hidden,
                    batch.restype.to(atom_output.token_hidden),
                    batch.plddt.to(atom_output.token_hidden),
                    batch.hotspot.to(atom_output.token_hidden),
                    add_feature,
                    add_feature,
                ),
                dim=-1,
            )
        )
        pair = self.template_embedder(batch.conditional_templ, batch.conditional_templ_mask)
        return token_embedding, pair.to(token_embedding)


class PxDesignConditionalTemplateEmbedder(nn.Module):
    r"""Embed released conditional-template distance-bin IDs."""

    def __init__(self, *, num_template_bins: int = 64, pair_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_template_bins + 1, pair_size)

    def forward(self, conditional_templ: Tensor, conditional_templ_mask: Tensor) -> Tensor:
        r"""Use ID zero for unconditioned pairs and IDs 1--64 for released bins."""

        embedding_ids = conditional_templ_mask.to(dtype=torch.long) * (conditional_templ.to(dtype=torch.long) + 1)
        return self.embedding(embedding_ids)


class PxDesignSampler(nn.Module):
    r"""The released PxDesign EDM predictor-corrector sampling loop."""

    def __init__(self, config: PxDesignConfig) -> None:
        super().__init__()
        sampler = config.sampler
        self.scheduler = EdmScheduler(
            max_noise_level=sampler.max_noise_level,
            min_noise_level=sampler.min_noise_level,
            rho=sampler.rho,
            sigma_data=config.denoiser.sigma_data,
        )
        self.gamma0 = sampler.gamma0
        self.gamma_min = sampler.gamma_min
        self.noise_scale = sampler.noise_scale
        self.step_scale_min = sampler.step_scale_min
        self.step_scale_max = sampler.step_scale_max
        self.step_scale_schedule = sampler.step_scale_schedule

    def forward(
        self,
        denoise: nn.Module,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        num_sampling_steps: int,
        sample_chunk_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""Sample ``num_samples`` coordinate sets, optionally in memory-bounded chunks."""

        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if sample_chunk_size is not None and sample_chunk_size < 1:
            raise ValueError("sample_chunk_size must be positive")
        noise_schedule = self.scheduler(
            num_sampling_steps=num_sampling_steps,
            device=context.device,
            dtype=context.dtype,
        )
        chunk_size = num_samples if sample_chunk_size is None else sample_chunk_size
        chunks: list[Tensor] = []
        for sample_start in range(0, num_samples, chunk_size):
            chunk_num_samples = min(chunk_size, num_samples - sample_start)
            reference = context.reference_positions.to(device=context.device, dtype=context.dtype)
            noise = torch.randn(
                *context.batch_shape,
                chunk_num_samples,
                reference.shape[-2],
                3,
                device=context.device,
                dtype=context.dtype,
                generator=generator,
            )
            coordinates = noise_schedule[0] * noise
            schedule_length = noise_schedule.shape[0]

            for step_index, (sigma_prev, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
                coordinates = coordinates - coordinates.mean(dim=-2, keepdim=True)
                batch_shape = coordinates.shape[:-2]
                rotations = uniform_random_rotation(
                    math.prod(batch_shape),
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                    generator=generator,
                ).reshape(*batch_shape, 3, 3)
                translations = torch.randn(
                    *batch_shape,
                    3,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                    generator=generator,
                )
                coordinates = torch.einsum("...ij,...nj->...ni", rotations, coordinates)
                coordinates = coordinates + translations.unsqueeze(-2)

                gamma = torch.where(
                    sigma_next > self.gamma_min,
                    sigma_next.new_tensor(self.gamma0),
                    sigma_next.new_zeros(()),
                )
                sigma_hat = sigma_prev * (gamma + 1)
                churn = torch.randn(
                    coordinates.shape,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                    generator=generator,
                )
                noisy_coordinates = coordinates + (self.noise_scale * torch.sqrt(sigma_hat**2 - sigma_prev**2) * churn)
                leading_shape = noisy_coordinates.shape[:-3]
                sample_count = noisy_coordinates.shape[-3]
                noise_level = sigma_hat.reshape((1,) * (len(leading_shape) + 1)).expand(
                    *leading_shape,
                    sample_count,
                )
                denoised_coordinates = denoise(
                    context.to_denoising_input(
                        noisy_coords=noisy_coordinates,
                        noise_level=noise_level.to(dtype=noisy_coordinates.dtype),
                    )
                )

                fraction = step_index / schedule_length
                minimum = float(self.step_scale_min)
                maximum = float(self.step_scale_max)
                if self.step_scale_schedule == "const":
                    step_scale = minimum
                elif self.step_scale_schedule == "linear":
                    step_scale = minimum + (maximum - minimum) * fraction
                elif self.step_scale_schedule == "poly":
                    step_scale = minimum + (maximum - minimum) * fraction**2
                elif self.step_scale_schedule == "cos":
                    step_scale = minimum + 0.5 * (maximum - minimum) * (1 - math.cos(math.pi * fraction))
                elif self.step_scale_schedule == "piecewise":
                    step_scale = minimum if fraction < 0.5 else maximum
                elif self.step_scale_schedule == "piecewise_65":
                    step_scale = minimum if fraction < 0.65 else maximum
                elif self.step_scale_schedule == "piecewise_70":
                    step_scale = minimum if fraction < 0.70 else maximum
                else:
                    raise ValueError(f"unsupported PxDesign step_scale_schedule: {self.step_scale_schedule!r}")

                derivative = (noisy_coordinates - denoised_coordinates) / noise_level[..., None, None]
                coordinates = noisy_coordinates + step_scale * (sigma_next - sigma_hat) * derivative
            chunks.append(coordinates)

        return torch.cat(chunks, dim=-3)


def _augment_reference_conformers(
    batch: PxDesignInput,
    generator: torch.Generator | None,
) -> PxDesignInput:
    r"""Apply the released independent per-residue reference augmentation."""

    positions = batch.reference_positions.clone()
    flat_positions = positions.reshape(-1, positions.shape[-2], 3)
    flat_spaces = batch.reference_space_id.reshape(-1, batch.reference_space_id.shape[-1])
    for sample_positions, sample_spaces in zip(flat_positions, flat_spaces):
        for space_id in torch.unique(sample_spaces):
            selected = sample_spaces == space_id
            values = sample_positions[selected]
            augmentation_dtype = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else values.dtype
            values = values.to(augmentation_dtype)
            values = values - values.mean(dim=0)
            translation = (
                2
                * torch.rand(
                    3,
                    device=values.device,
                    dtype=augmentation_dtype,
                    generator=generator,
                )
                - 1
            )
            rotation = uniform_random_rotation(
                1,
                device=values.device,
                dtype=augmentation_dtype,
                generator=generator,
            )[0]
            augmented = (values + translation) @ rotation.transpose(-1, -2)
            sample_positions[selected] = augmented.to(sample_positions.dtype)
    return replace(batch, reference_positions=positions)


# ============================================================================
# PxDesign released tensor boundary (folded in from contract.py)
# ============================================================================
# ``Score`` / ``Metric`` are the shared scientific value types. PXDesign is
# coordinate-producing, so per the LOCKED §5.3 decision (§5.3.1 freezes the shared
# fixed-length carrier and explicitly excludes PXDesign and every coordinate-generating
# design operation; §5.3.3 forbids widening that shared record for such families) it defines
# its OWN standalone family ``PxDesignCandidate`` / ``PxDesignResult`` below and reuses
# neither the shared ``DesignCandidate`` nor the coordinate-free ``DesignResult``.
#
# The placement core is frozen (DESIGN §3.7 "placement core", ratified
# 2026-07-18): ``PlacementSpec`` and ``PlacementMap`` are the shared records in ``ofoldx.data``;
# ``Hotspot`` is the frozen §3.7.4 distance-free per-residue label PXDesign consumes.


@dataclass(frozen=True)
class PxDesignLayout:
    r"""Canonical identities needed to decode the released coordinate samples."""

    system: MolecularSystem
    sites: tuple[AtomSite, ...]
    conditions: tuple[PlacementSpec | Hotspot, ...]
    placement_map: PlacementMap


@dataclass(kw_only=True)
class PxDesignInput:
    r"""Tensors consumed by the released PxDesign condition and diffusion graph.

    ``restype`` is the released 36-channel one-hot representation. ``plddt`` and
    ``hotspot`` are explicit ``[..., N_token, 1]`` tensors; callers represent an
    absent condition with zeros rather than changing the input schema.
    """

    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    restype: Tensor
    plddt: Tensor
    hotspot: Tensor
    conditional_templ: Tensor
    conditional_templ_mask: Tensor
    layout: PxDesignLayout


@dataclass
class PxDesignOutput:
    r"""Coordinates computed by one PxDesign denoising or sampling call."""

    coordinates: Tensor


@dataclass(eq=False)
class PxDesignCandidate:
    r"""One ranked, evaluated PXDesign target--binder backbone design sample.

    PXDesign is a coordinate-*producing* family, so per the LOCKED §5.3 decision this
    candidate is a STANDALONE family-local record — NOT a subclass of the shared §5.3.1
    :class:`~ofoldx.results.DesignCandidate`, whose frozen fixed-length slice explicitly
    excludes PXDesign and every coordinate-generating design operation, and NOT a widening of
    that shared record, which §5.3.3 forbids for coordinate-producing and variable-topology
    families. It mirrors the sibling coordinate-producing
    :class:`~ofoldx.models.rfdiffusion.model.RFdiffusionDesignCandidate`: each candidate owns
    the decoded backbone :class:`~ofoldx.data.structure.Structure` (which owns its
    ``MolecularSystem`` via ``Structure.system``), so no separate ``system`` field is carried.
    The family carrier for these coordinate-bearing candidates is the sibling
    :class:`PxDesignResult` (NOT the shared coordinate-free ``DesignResult``). This is a
    family-local carrier pending the §5.3 coordinate-bearing design sub-freeze — NOT a frozen
    public shape.

    Fields:

    - ``structure`` — the decoded backbone Structure (truthful ``UNK`` binder identity,
      GLY/xpb backbone atom contract). Present for every generator sample.
    - ``scores`` — the evaluation Scores (§3): AF2-IG / Protenix confidence, RMSD, geometry,
      success flags, and the ranking ``rank`` / ``bucket`` / ``pass_*`` Scores stamped by the
      suite. Empty for raw generator output.
    - ``metrics`` — per-residue evaluation metrics (e.g. AF2 pLDDT). Currently empty for raw
      generator output: the released sampler emits only coordinates (its ``plddt``
      conditioning input is zeros and there is no confidence head), so no per-sample
      scientific metric is decoded from the generator itself.
    - ``sequence`` — the ProteinMPNN-assigned binder sequence per instance once the suite
      sequencing stage has run; ``None`` for a raw generator backbone whose binder is a
      truthful ``UNK`` placeholder with no assigned sequence.

    ``chosen_struct_type`` / ``chosen_struct_path`` provenance from the suite selection are
    carried as plain attributes when the suite sets them, not as Scores.
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    sequence: dict[str, tuple[str, ...]] | None = None


@dataclass(eq=False)
class PxDesignResult:
    r"""One generation-ordered collection of ranked, evaluated PXDesign candidates.

    This is PXDesign's OWN family Result. PXDesign is coordinate-producing, so per the
    LOCKED §5.3 decision it defines this carrier in its own dir rather than reusing the
    frozen, coordinate-free shared :class:`~ofoldx.results.DesignResult` (§5.3.1). It is a
    family-local carrier — NOT itself DESIGN-frozen — pending the §5.3 coordinate-bearing
    design sub-freeze, and is explicitly not the shared ``DesignResult``.

    ``.best`` / ``.ranked`` replicate the frozen single-Score validation at
    results/values.py:95-110 (single unit, single scope, consistent ``preferred``,
    NaN-reject). ``ranked("af2_ipae")`` (preview) / ``ranked("ptx_iptm")`` (extended)
    reproduce the released within-bucket tie ordering; the full released two-tier
    bucket+fusion selection is performed by the suite Pipeline, which stamps
    ``bucket`` / ``rank`` / ``pass_*`` Scores that this carrier only needs to order over.
    """

    candidates: tuple[PxDesignCandidate, ...]

    def _validated_scores(self, score: str) -> tuple[Score, ...]:
        r"""Return the per-candidate Scores after the frozen single-Score checks."""

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
        return scores

    def best(self, score: str) -> PxDesignCandidate:
        r"""Return the earliest best candidate under one explicit score."""

        scores = self._validated_scores(score)
        if {value.preferred for value in scores} == {"higher"}:
            return max(self.candidates, key=lambda candidate: candidate.scores[score].value)
        return min(self.candidates, key=lambda candidate: candidate.scores[score].value)

    def ranked(self, score: str) -> tuple[PxDesignCandidate, ...]:
        r"""Return the candidates ordered best-first under one explicit score.

        The ordering is stable, so candidates that tie on ``score`` keep their generation
        order — reproducing the released within-bucket tie ordering.
        """

        scores = self._validated_scores(score)
        reverse = {value.preferred for value in scores} == {"higher"}
        return tuple(
            sorted(
                self.candidates,
                key=lambda candidate: candidate.scores[score].value,
                reverse=reverse,
            )
        )
