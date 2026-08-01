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
ODesign structure-generation model and tensor interfaces.
"""

from __future__ import annotations

import math
import operator
from typing import NamedTuple, cast

import torch
import torch.nn.functional as F
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data.tokenization import TOKENS
from ...errors import RequestRef, fail
from ...model import Model, TensorCollection
from ...modules.atom import (
    AtomConditioner,
    AtomPairEncoder,
    AtomToTokenEncoder,
    LocalAtomGeometry,
    LocalAtomWindowing,
)
from ...modules.attention import PerBlockPairBiasProjection
from ...modules.backend import LayerNorm
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateEdmSampler,
    denoise_coordinates,
)
from ...modules.diffusion.edm import lognormal_training_noise
from ...modules.diffusion.sampling import draw_normal, draw_normal_like
from ...modules.embedding import (
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask, weighted_mean
from ...modules.geometry import uniform_random_rotation
from ...modules.head import DistogramHead
from ...modules.msa import MsaEncoder
from ...modules.pair import TrianglePairUpdateStack
from ...modules.pairformer import PairformerStack
from ...modules.transformer import NormalizedKeyAtomCrossAttentionTransformerEncoderBlock
from ..processing import (
    project_model_categories,
    project_model_coordinates,
    project_model_counts,
    project_model_features,
)
from ..vocabulary import VOCAB_MAPPING
from .config import ODesignConfig

_ATOM_FEATURE_SIZE = 386
#: The canonical vocabulary is every model's token axis: identity, MSA profile, deletion mean and hotspot.
_VOCAB_SIZE = len(TOKENS)
_TOKEN_FEATURE_SIZE = 2 * _VOCAB_SIZE + 2
#: The upstream ODesign conditioner alphabet in canonical spelling: 21 protein classes, 5 RNA, 5 DNA and the gap.
ODESIGN_SOURCE_TOKEN_INDEX = {
    token: index
    for index, token in enumerate(
        (*"ARNDCQEGHILKMFPSTWYVX", *(f"RNA_{x}" for x in "AGCUN"), *(f"DNA_{x}" for x in "AGCTN"), "-")
    )
}
#: Canonical classes the alphabet reads as another of its rows: each unknown class is the alphabet's unknown class of
#: its monomer type (upstream gives every per-atom token, ligands included, its UNK protein token), so a generated
#: position's hidden class is one row and the generated channel cancels it.
ODESIGN_SOURCE_ALIASES = {"OTHER": "X", "RNA_X": "RNA_N", "DNA_X": "DNA_N"}
_DESIGN_FEATURE_SIZE = 4
_SOURCE_TOKEN_FEATURE_SIZE = 35 + 32 + 2
_NUM_BOND_CLASSES = 2
_DEFAULT_NUM_RECYCLES = 10


def odesign_source_map() -> Tensor:
    r"""
    ``[32, 72]``: per canonical class (column), its coefficients on the upstream ODesign alphabet — a class the
    alphabet holds is itself, an alias the class it is read as, a class whose members it holds is their mean, and
    every other class is zero (the conditioner keeps its upstream width).
    """

    mapping = torch.zeros(len(ODESIGN_SOURCE_TOKEN_INDEX), _VOCAB_SIZE)
    for column, token in enumerate(TOKENS):
        token = ODESIGN_SOURCE_ALIASES.get(token, token)
        members = (token,) if token in ODESIGN_SOURCE_TOKEN_INDEX else VOCAB_MAPPING.get(token, ())
        rows = [ODESIGN_SOURCE_TOKEN_INDEX[member] for member in members if member in ODESIGN_SOURCE_TOKEN_INDEX]
        if rows and len(rows) == len(members):
            mapping[rows, column] = 1 / len(rows)
    return mapping


class ODesignModel(Model):
    r"""
    Generate all-atom biomolecular structures with ODesign.

    Accepts [`ODesignInput`][ofoldx.models.odesign.model.ODesignInput] and returns
    [`ODesignOutput`][ofoldx.models.odesign.model.ODesignOutput].

    Args:
        config: Model architecture configuration.
    """

    config_class = ODesignConfig
    config: ODesignConfig

    def __init__(self, config: ODesignConfig | None = None) -> None:
        config = ODesignConfig() if config is None else config
        super().__init__(config)
        self.input_embeddings = ODesignInputEmbeddings(
            single_size=config.single_size,
        )
        token_embedding_size = config.single_size + _TOKEN_FEATURE_SIZE + _DESIGN_FEATURE_SIZE
        self.single_init_projection = nn.Linear(token_embedding_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(
            4 * config.max_relative_residue + 2 * config.max_relative_chain + 7,
            config.pair_size,
        )
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.constraint_embeddings = ODesignConstraintEmbeddings(
            pair_size=config.pair_size,
            hidden_size=config.constraint_embeddings.hidden_size,
            num_blocks=config.constraint_embeddings.num_blocks,
            num_bins=config.constraint_embeddings.num_distogram_bins,
            min_distance=config.constraint_embeddings.min_distance,
            max_distance=config.constraint_embeddings.max_distance,
            dropout=config.constraint_embeddings.dropout,
            backend="auto",
        )
        self.msa_trunk = MsaEncoder(
            pair_size=config.pair_size,
            msa_vocab_size=_VOCAB_SIZE,
            token_feature_size=token_embedding_size,
            num_stages=config.trunk.num_msa_stages,
            msa_dropout=config.trunk.msa_dropout,
            pair_dropout=config.trunk.pair_dropout,
        )
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
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
            token_feature_size=config.single_size + _SOURCE_TOKEN_FEATURE_SIZE,
            atom_feature_size=_ATOM_FEATURE_SIZE,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
        )
        self.register_buffer("token_feature_map", odesign_source_map(), persistent=False)
        self.coordinate_sampler = ODesignEdmSampler()
        self.distogram_head = DistogramHead(config.pair_size, config.pairwise.num_distogram_bins)
        self.token_bond_head = ODesignTokenBondHead(
            pair_size=config.pair_size,
            hidden_size=config.pairwise.bond_hidden_size,
        )
        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.distogram_head.projection.weight)
        if self.distogram_head.projection.bias is not None:
            nn.init.zeros_(self.distogram_head.projection.bias)
        self.post_init()

    def forward(
        self,
        input: ODesignInput | ODesignLigandInput,
        target: ODesignStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        num_recycles: int | None = None,
        use_conditioning: bool = True,
        generator: torch.Generator | None = None,
    ) -> ODesignOutput:
        r"""
        Denoise randomly corrupted target coordinates.

        ``B`` is batch size, ``S`` is the coordinate-sample axis, ``N`` is token
        count, and ``A`` is atom count.

        Args:
            input: Processed structure-design input.
            target: Coordinates already aligned to the Model atom axis.
            num_samples: Number ``S`` of independently corrupted copies per input.
            sample_chunk_size: Maximum samples per input evaluated by one
                denoiser call. ``None`` evaluates all samples together.
            synchronize_noise_levels: Whether samples from one input item share one
                noise level.
            coordinate_augmentation: Whether to rotate and translate the centered
                target independently for each sample.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            num_recycles: Number of trunk recycling passes. ``None`` uses the
                released default of 10.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.
            generator: Optional generator for rigid augmentation, noise levels, and
                coordinate noise.

        Returns:
            Denoised coordinates in the Target's original frame with shape
                ``(B, S, A, 3)``, noise levels with shape ``(B, S)``, and pairwise
                logits computed once from the trunk.
        """

        input = normalize_odesign_input(input, dtype=self.single_init_projection.weight.dtype)
        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        if sample_chunk_size is None:
            sample_chunk_size = num_samples
        else:
            sample_chunk_size = operator.index(sample_chunk_size)
            if sample_chunk_size < 1:
                fail("value_invalid", RequestRef("sample_chunk_size", str(sample_chunk_size)))
            sample_chunk_size = min(sample_chunk_size, num_samples)
        num_recycles = _DEFAULT_NUM_RECYCLES if num_recycles is None else num_recycles
        trunk = self._run_trunk(input, num_recycles=num_recycles)
        target_mask = target.coordinate_mask.to(device=input.atom_features.device).bool() & input.atom_mask.bool()
        target_coordinates = project_model_coordinates(
            target.coordinates.to(device=input.atom_features.device),
            target_mask,
            dtype=input.atom_features.dtype,
            feature="ODesign target coordinates",
        )
        batch_size = target_coordinates.shape[0]
        augmented_target_coordinates, augmentation_transform = _augment_odesign_target(
            target_coordinates,
            target_mask,
            num_samples=num_samples,
            augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        num_noise_levels = batch_size if synchronize_noise_levels else batch_size * num_samples
        noise_level = lognormal_training_noise(
            num_noise_levels,
            sigma_data=self.coordinate_denoiser.preconditioner.sigma_data,
            log_mean=noise_log_mean,
            log_std=noise_log_std,
            device=target_coordinates.device,
            dtype=target_coordinates.dtype,
            generator=generator,
        )
        if synchronize_noise_levels:
            noise_level = noise_level.repeat_interleave(num_samples)
        noise_level = noise_level.reshape(batch_size, num_samples)
        coordinate_noise = draw_normal_like(augmented_target_coordinates, generator)
        fixed_coordinate_mask = target_mask & input.fixed_coordinate_mask.bool()
        fixed_coordinate_mask = fixed_coordinate_mask.unsqueeze(-2).expand(-1, num_samples, -1)
        coordinate_noise = coordinate_noise.masked_fill(fixed_coordinate_mask.unsqueeze(-1), 0)
        noisy_coordinates = augmented_target_coordinates + noise_level[..., None, None] * coordinate_noise
        raw_output = self._denoise_from_trunk(
            input,
            trunk,
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
            sample_chunk_size=sample_chunk_size,
        )
        augmented_coordinates = torch.where(
            fixed_coordinate_mask.unsqueeze(-1),
            augmented_target_coordinates,
            raw_output.coordinates,
        )
        coordinates = _reverse_odesign_augmentation(
            augmented_coordinates,
            augmentation_transform,
        )
        coordinates = torch.where(
            fixed_coordinate_mask.unsqueeze(-1),
            target_coordinates.unsqueeze(-3).expand(-1, num_samples, -1, -1),
            coordinates,
        )
        return ODesignOutput(
            coordinates=coordinates,
            distogram_logits=raw_output.distogram_logits,
            token_bond_logits=raw_output.token_bond_logits,
            noise_level=noise_level,
            batch_size=[coordinates.shape[0]],
        )

    def denoise(
        self,
        input: ODesignInput | ODesignLigandInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        num_recycles: int | None = None,
        use_conditioning: bool = True,
    ) -> ODesignRawOutput:
        r"""
        Denoise atom coordinates at specified EDM noise levels.

        Args:
            input: Processed structure-design input.
            noisy_coordinates: Noisy atom coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: Diffusion noise levels with shape ``(B, S)``.
            num_recycles: Number of trunk recycling passes. ``None`` uses the
                released default of 10.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.

        Returns:
            Denoised coordinates and pairwise logits.
        """

        input = normalize_odesign_input(input, dtype=self.single_init_projection.weight.dtype)
        noisy_coordinates = project_model_coordinates(
            noisy_coordinates,
            input.atom_mask[:, None, :].expand(noisy_coordinates.shape[:-1]),
            dtype=input.atom_features.dtype,
            feature="ODesign noisy coordinates",
        )
        num_recycles = _DEFAULT_NUM_RECYCLES if num_recycles is None else num_recycles
        trunk = self._run_trunk(input, num_recycles=num_recycles)
        return self._denoise_from_trunk(
            input,
            trunk,
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

    @torch.inference_mode()
    def sample(
        self,
        input: ODesignInput | ODesignLigandInput,
        *,
        num_samples: int = 1,
        num_recycles: int | None = None,
        num_sampling_steps: int = 200,
        num_trunk_evaluations: int = 1,
        partial_diffusion_snr: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ODesignSamplingOutput:
        r"""
        Sample coordinates while retaining the model's pairwise predictions.

        ``K`` is ``num_trunk_evaluations``, ``B`` is batch size, ``S`` is ``num_samples``,
        ``N`` is token count, and ``A`` is atom count.

        Args:
            input: Processed design input.
            num_samples: Number of coordinate samples per trunk evaluation.
            num_recycles: Number of trunk recycling passes. ``None`` uses the
                released default of 10.
            num_sampling_steps: Number of coordinate-diffusion steps.
            num_trunk_evaluations: Number of independent trunk evaluations.
            partial_diffusion_snr: Signal-to-noise ratio for partial diffusion from
                ``input.observed_positions``. ``None`` starts from noise.
            generator: Random generator used by coordinate sampling.

        Returns:
            An [`ODesignSamplingOutput`][] with ``coordinates`` of shape
                ``(B, K, S, A, 3)`` in angstroms, ``distogram_logits`` of shape
                ``(B, K, N, N, D)``, and ``token_bond_logits`` of shape
                ``(B, K, N, N, 2)``.
        """

        input = normalize_odesign_input(input, dtype=self.single_init_projection.weight.dtype)
        num_recycles = _DEFAULT_NUM_RECYCLES if num_recycles is None else num_recycles
        partial_diffusion = partial_diffusion_snr is not None
        if partial_diffusion and not bool((input.observed_mask | ~input.atom_mask).all()):
            fail("value_invalid", RequestRef("partial_diffusion", "unobserved model atoms"))
        max_noise_level = 1 / math.sqrt(partial_diffusion_snr + 1e-6) if partial_diffusion_snr is not None else None
        coordinates = []
        distograms = []
        bond_logits = []
        for _ in range(num_trunk_evaluations):
            trunk = self._run_trunk(input, num_recycles=num_recycles)
            distograms.append(self.distogram_head(trunk.pair))
            bond_logits.append(self.token_bond_head(trunk.pair))
            context = self._denoise_context(input, trunk)
            coordinates.append(
                self.coordinate_sampler(
                    self.coordinate_denoiser.prepare(context),
                    context=context,
                    num_samples=num_samples,
                    num_sampling_steps=num_sampling_steps,
                    initial_coords=input.observed_positions if partial_diffusion else None,
                    condition_coords=input.observed_positions,
                    condition_mask=input.fixed_coordinate_mask,
                    max_noise_level=max_noise_level,
                    generator=generator,
                )
            )
        return ODesignSamplingOutput(
            coordinates=torch.stack(coordinates, dim=1),
            distogram_logits=torch.stack(distograms, dim=1),
            token_bond_logits=torch.stack(bond_logits, dim=1),
            batch_size=[input.atom_features.shape[0]],
        )

    def _denoise_from_trunk(
        self,
        input: ODesignInput | ODesignLigandInput,
        trunk: _ODesignTrunkOutput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        use_conditioning: bool,
        sample_chunk_size: int | None = None,
    ) -> ODesignRawOutput:
        context = self._denoise_context(
            input,
            trunk,
            use_conditioning=use_conditioning,
        )
        coordinates = denoise_coordinates(
            self.coordinate_denoiser.prepare(context),
            context=context,
            noisy_coords=noisy_coordinates,
            noise_level=noise_level,
            sample_chunk_size=sample_chunk_size,
        )
        return ODesignRawOutput(
            coordinates=coordinates,
            distogram_logits=self.distogram_head(trunk.pair),
            token_bond_logits=self.token_bond_head(trunk.pair),
            batch_size=[coordinates.shape[0]],
        )

    def _run_trunk(
        self,
        input: ODesignInput | ODesignLigandInput,
        *,
        num_recycles: int,
    ) -> _ODesignTrunkOutput:
        local_geometry = self.input_embeddings.windowing(
            input.reference_positions,
            input.reference_space_id,
            input.atom_mask,
            input.reference_mask,
        )
        token_embedding = self.input_embeddings(
            input.atom_features,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            token_features=input.token_features,
            design_features=input.design_features,
            atom_to_token=input.atom_to_token,
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
            max_relative_residue=self.config.max_relative_residue,
            max_relative_chain=self.config.max_relative_chain,
            dtype=single_init.dtype,
        )
        pair_mask = pair_mask_from_token_mask(input.token_mask, backend=self.backend)
        pair_init = self._initial_pair(single_init, input, relative_features)
        deletion_counts = input.msa_deletion_counts.to(torch.float64)
        msa_continuous_features = torch.stack(
            (
                (deletion_counts > 0).to(deletion_counts.dtype),
                (2 / math.pi) * torch.atan(deletion_counts / 3),
            ),
            dim=-1,
        ).to(token_embedding.dtype)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        grad_enabled = torch.is_grad_enabled()
        for cycle in range(num_recycles):
            with torch.set_grad_enabled(grad_enabled and cycle == num_recycles - 1):
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                pair = pair + self.constraint_embeddings(
                    pair,
                    input.constraint_distogram,
                    pair_mask=pair_mask,
                )
                _, pair = self.msa_trunk(
                    input.msa_tokens,
                    msa_continuous_features,
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
            relative_features=relative_features,
        )

    def _initial_pair(
        self,
        single: Tensor,
        input: ODesignInput | ODesignLigandInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        pair = self.pair_left_projection(single).unsqueeze(-2)
        pair = pair + self.pair_right_projection(single).unsqueeze(-3)
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(input.token_bonds.unsqueeze(-1))
        return pair * pair_mask_from_token_mask(input.token_mask, backend=self.backend).unsqueeze(-1).to(pair.dtype)

    def _denoise_context(
        self,
        input: ODesignInput | ODesignLigandInput,
        trunk: _ODesignTrunkOutput,
        *,
        use_conditioning: bool = True,
    ) -> CoordinateDenoiseContext:
        conditioning_single = trunk.single if use_conditioning else 0 * trunk.single
        conditioning_pair = trunk.pair if use_conditioning else 0 * trunk.pair
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=_odesign_source_token_embedding(trunk.token_embedding, self.token_feature_map),
            single=trunk.single,
            conditioning_single=conditioning_single,
            conditioning_pair=conditioning_pair,
            relative_features=trunk.relative_features,
            local_geometry=self.coordinate_denoiser.windowing(
                input.reference_positions,
                input.reference_space_id,
                input.atom_mask,
                input.reference_mask,
            ),
        )


def _odesign_source_token_embedding(token_embedding: Tensor, source_map: Tensor) -> Tensor:
    r"""
    Map the canonical token features onto the upstream conditioner axis.

    The conditioner keeps its upstream width: its residue and profile planes are the canonical planes read through
    ``source_map``, and the three generated channels take the design-state planes. A generated position's hidden
    class (``X``, ``RNA_X``, ``DNA_X``) is read as one alphabet row (``ODESIGN_SOURCE_ALIASES``), which the
    generated channel subtracts back out: upstream states no residue where it states a generated position.
    """

    single_size = token_embedding.shape[-1] - _TOKEN_FEATURE_SIZE - _DESIGN_FEATURE_SIZE
    single, residue, profile, scalars, design = token_embedding.split(
        (single_size, _VOCAB_SIZE, _VOCAB_SIZE, 2, _DESIGN_FEATURE_SIZE), dim=-1
    )
    mapping = source_map.to(residue).transpose(-1, -2)
    generated_protein, generated_rna, generated_dna, generated_ligand = design.unbind(dim=-1)
    channels = list((residue @ mapping).unbind(dim=-1))
    for row, generated in (
        (ODESIGN_SOURCE_TOKEN_INDEX["X"], generated_protein + generated_ligand),
        (ODESIGN_SOURCE_TOKEN_INDEX["RNA_N"], generated_rna),
        (ODESIGN_SOURCE_TOKEN_INDEX["DNA_N"], generated_dna),
    ):
        channels[row] = channels[row] - generated
    source_residue = torch.stack(
        (*channels, generated_protein, generated_rna + generated_dna, generated_ligand), dim=-1
    )
    return torch.cat((single, source_residue, profile @ mapping, scalars), dim=-1)


class ODesignInputEmbeddings(nn.Module):
    r"""
    Aggregate reference atoms and append ODesign token features.

    The centred [`windowing`][ofoldx.modules.atom.window.LocalAtomWindowing] builds the local atom geometry
    the embeddings consume. The reference atoms are conditioned, their local pairs encoded and projected to
    every block's attention bias, and the attended atoms mean-pooled to tokens.
    """

    def __init__(
        self,
        single_size: int = 384,
    ) -> None:
        super().__init__()
        self.windowing = LocalAtomWindowing(32, 128)
        self.atom_conditioner = AtomConditioner(_ATOM_FEATURE_SIZE, 128)
        self.atom_pair_encoder = AtomPairEncoder(128, 16)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=16,
            num_blocks=3,
            num_attention_heads=4,
            bias=False,
        )
        self.atom_encoder = AtomToTokenEncoder(
            128,
            single_size,
            3,
            4,
            block=NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
        )

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        token_features: Tensor,
        design_features: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        token_mask: Tensor,
        local_geometry: LocalAtomGeometry,
    ) -> Tensor:
        atom_features = project_model_features(
            atom_features, atom_mask.bool()[..., None], feature="ODesign atom features"
        )
        reference_positions = project_model_coordinates(
            reference_positions, reference_mask.bool() & atom_mask.bool(), feature="ODesign reference positions"
        )
        reference_charges = project_model_features(
            reference_charges, atom_mask.bool()[..., None], feature="ODesign reference charges"
        )
        token_features = project_model_features(
            token_features, token_mask.bool()[..., None], feature="ODesign token features"
        )
        design_features = project_model_features(
            design_features, token_mask.bool()[..., None], feature="ODesign design features"
        )
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        atom_pair = self.atom_pair_encoder(atom_conditioning, local_geometry)
        encoded = self.atom_encoder(
            atom_conditioning,
            self.atom_pair_bias_projection(atom_pair),
            atom_mask,
            atom_to_token,
            token_features,
            local_geometry.window_info,
        )
        output = torch.cat(
            [encoded, token_features.to(encoded), design_features.to(encoded)],
            dim=-1,
        )
        return output * token_mask.unsqueeze(-1).to(output.dtype)


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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.distogram_projection = nn.Linear(num_bins, hidden_size, bias=False)
        self.pair_update_stack = TrianglePairUpdateStack(
            num_blocks=num_blocks,
            pair_size=hidden_size,
            dropout=dropout,
            backend=backend,
        )
        self.hidden_norm = LayerNorm(hidden_size, backend=backend)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(self, pair: Tensor, distances: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        if pair_mask is not None:
            distances = project_model_features(
                distances, pair_mask.bool()[..., None], feature="ODesign constraint distances"
            )
            pair = project_model_features(pair, pair_mask.bool()[..., None], feature="ODesign pair features")
        boundaries = torch.linspace(
            self.min_distance,
            self.max_distance,
            self.num_bins - 1,
            device=distances.device,
            dtype=distances.dtype,
        )
        bins = (distances > boundaries).sum(dim=-1)
        distogram = F.one_hot(bins, self.num_bins).to(self.distogram_projection.weight)
        hidden = self.pair_projection(self.pair_norm(pair)) + self.distogram_projection(distogram)
        hidden = self.pair_update_stack(hidden, pair_mask=pair_mask)
        return self.output_projection(self.activation(self.hidden_norm(hidden)))


class ODesignEdmSampler(CoordinateEdmSampler):
    r"""
    ODesign EDM sampler: every step's random rotation and translation is undone after the update.

    The released ODesign generator reverses each step's augmentation, so sampling continues in the frame of the
    initial noise.
    """

    def _unaugment(self, coords: Tensor, transform: tuple[Tensor, Tensor, Tensor] | None) -> Tensor:
        return coords if transform is None else self._reverse_augmentation(coords, transform)


class ODesignTokenBondHead(nn.Module):
    r"""
    Predict whether each ODesign token pair is covalently connected.
    """

    def __init__(self, *, pair_size: int = 128, hidden_size: int = 128) -> None:
        super().__init__()
        self.input_projection = nn.Linear(pair_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, _NUM_BOND_CLASSES)

    def forward(self, pair: Tensor) -> Tensor:
        return self.output_projection(self.input_projection(pair))


def _augment_odesign_target(
    coordinates: Tensor,
    coordinate_mask: Tensor,
    *,
    num_samples: int,
    augmentation: bool,
    translation_scale: float,
    generator: torch.Generator | None,
) -> tuple[Tensor, tuple[Tensor | None, Tensor, Tensor]]:
    coordinates = project_model_coordinates(coordinates, coordinate_mask, feature="ODesign augmentation coordinates")
    center = weighted_mean(coordinates.transpose(-1, -2), coordinate_mask.unsqueeze(-2)).unsqueeze(-2)
    center = torch.where(coordinate_mask.any(-1, keepdim=True).unsqueeze(-1), center, 0)
    centered = (coordinates - center).unsqueeze(-3).expand(-1, num_samples, -1, -1)
    batch_size = coordinates.shape[0]
    rotation: Tensor | None
    if augmentation:
        rotation = uniform_random_rotation(
            batch_size * num_samples,
            device=coordinates.device,
            dtype=coordinates.dtype,
            generator=generator,
        ).reshape(batch_size, num_samples, 3, 3)
        translation = translation_scale * draw_normal(
            (batch_size, num_samples, 3),
            device=coordinates.device,
            dtype=coordinates.dtype,
            generator=generator,
        )
        augmented = torch.einsum("bsad,bsde->bsae", centered, rotation)
        augmented = augmented + translation.unsqueeze(-2)
    else:
        rotation = None
        translation = torch.zeros(
            batch_size,
            num_samples,
            3,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        augmented = centered
    augmented = torch.where(coordinate_mask[:, None, :, None], augmented, 0)
    return augmented, (rotation, translation, center)


def _reverse_odesign_augmentation(
    coordinates: Tensor,
    transform: tuple[Tensor | None, Tensor, Tensor],
) -> Tensor:
    rotation, translation, center = transform
    coordinates = coordinates - translation.unsqueeze(-2)
    if rotation is not None:
        coordinates = torch.einsum(
            "bsad,bsde->bsae",
            coordinates,
            rotation.transpose(-1, -2),
        )
    return coordinates + center.unsqueeze(-3)


# ===========================================================================
# Typed tensor and semantic records for ODesign.
# ===========================================================================


class _ODesignTrunkOutput(NamedTuple):
    r"""
    Named representations produced by the ODesign trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets


@tensorclass
class ODesignInput:
    r"""
    Tensor and molecular-layout input for ODesign.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, and ``M`` MSA depth.
    Processors produce ``B=1``. A larger explicit ``B`` represents repeated
    tensors for the same ``system`` and layout, not heterogeneous structures.

    Attributes:
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        token_index: Token indices within each residue with shape ``(B, N)``.
        cyclic_period: Optional cyclic-position periods with shape ``(B, N)``.
        token_features: Canonical residue, MSA-profile, deletion, and hotspot
            features with shape ``(B, N, 2 * len(TOKENS) + 2)``.
        design_features: Unconditioned protein, RNA, DNA, and ligand identity
            states with shape ``(B, N, 4)``. The Model prepends
            its atom embedding to both feature tensors.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        atom_features: Reference atom features with shape ``(B, A, 386)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        atom_mask: Valid-atom mask with shape ``(B, A)``.
        atom_monomer: ``MonomerType`` of each atom's token with shape ``(B, A)``; the loss weights read it, and a
            ligand or provisional atom is ``NONE``.
        atom_token_mask: Tokens that are one atom each with shape ``(B, N)``.
        generated_atom_mask: Atom rows bound to no canonical atom with shape ``(B, A)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        observed_positions: Observed atom coordinates with shape ``(B, A, 3)`` in
            angstroms.
        observed_mask: Observed-coordinate validity mask with shape ``(B, A)``.
        fixed_coordinate_mask: Atoms fixed during sampling with shape ``(B, A)``.
        constraint_distogram: Raw observed pair distances with shape
            ``(B, N, N, 1)``; the Model performs released 39-bin encoding.
        msa_tokens: Compact MSA token indices with shape ``(B, M, N)``.
        msa_deletion_counts: Raw int64 MSA deletion counts with shape ``(B, M, N)``.
        msa_mask: MSA validity mask with shape ``(B, M, N)``.
        token_representative_atom_index: Representative atom row of each token,
            used by the distogram target, with shape ``(B, N)``.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    cyclic_period: Tensor
    token_features: Tensor
    design_features: Tensor
    token_bonds: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    atom_monomer: Tensor
    atom_token_mask: Tensor
    generated_atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    observed_positions: Tensor
    observed_mask: Tensor
    fixed_coordinate_mask: Tensor
    constraint_distogram: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_mask: Tensor
    token_representative_atom_index: Tensor


@tensorclass
class ODesignLigandInput:
    r"""
    ODesign tensor input with canonical context and provisional generated rows.

    The equation fields deliberately repeat [`ODesignInput`][] because ligand
    output has no canonical generated chemistry. The atom rows of
    ``generated_atom_mask`` are the ordered provisional slots: they state coordinates and element probabilities, not
    a chemical vertex.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    cyclic_period: Tensor
    token_features: Tensor
    design_features: Tensor
    token_bonds: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    atom_monomer: Tensor
    atom_token_mask: Tensor
    generated_atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    observed_positions: Tensor
    observed_mask: Tensor
    fixed_coordinate_mask: Tensor
    constraint_distogram: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_mask: Tensor
    token_representative_atom_index: Tensor


@tensorclass
class ODesignStructureTarget:
    r"""
    Ground-truth coordinates already aligned to the ODesign Model atom axis.

    Attributes:
        coordinates: Target coordinates with shape ``(B, A, 3)``.
        coordinate_mask: Valid-coordinate mask with shape ``(B, A)``.
        covalent_bond_index: Undirected covalent target edges on the Model atom
            axis, stored once with shape ``(B, E, 2)``.
        covalent_pair_mask: Known binary covalent status with shape ``(B, A, A)``.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    covalent_bond_index: Tensor
    covalent_pair_mask: Tensor


class ODesignRawOutput(TensorCollection):
    r"""
    Predictions from one explicit ODesign denoising call.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        token_bond_logits: Binary token-pair bond logits with shape
            ``(B, N, N, 2)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_logits: Tensor


class ODesignOutput(ODesignRawOutput):
    r"""
    Predictions from stochastic target corruption and denoising.

    Attributes:
        coordinates: Denoised atom coordinates in the Target's original frame,
            with shape ``(B, S, A, 3)``.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        token_bond_logits: Binary token-pair bond logits with shape
            ``(B, N, N, 2)``.
        noise_level: Sampled EDM noise levels with shape ``(B, S)``.
    """

    noise_level: Tensor


class ODesignSamplingOutput(TensorCollection):
    r"""
    Predictions from one ODesign sampling call.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, K, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape
            ``(B, K, N, N, D)``.
        token_bond_logits: Binary token-pair bond logits with shape
            ``(B, K, N, N, 2)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_logits: Tensor


def normalize_odesign_input(
    input: ODesignInput | ODesignLigandInput, *, dtype: torch.dtype | None = None
) -> ODesignInput | ODesignLigandInput:
    """Project available geometry and valid model features without changing facts.

    Provisional slots with canonical index -1 remain active model rows. Reference
    geometry, source observations and chemical model features use separate masks.
    """
    result = cast(ODesignInput | ODesignLigandInput, cast(TensorClass, input).clone(recurse=False))
    atom_mask, token_mask = input.atom_mask.bool(), input.token_mask.bool()
    valid_fixed = (~input.fixed_coordinate_mask.bool() | (input.observed_mask.bool() & atom_mask)).all()
    if torch.compiler.is_compiling():
        torch._assert_async(valid_fixed, "fixed ODesign atoms require available source coordinates")
    elif not bool(valid_fixed):
        fail("value_invalid", RequestRef("fixed_atoms", "unobserved source coordinates"))
    result.reference_positions = project_model_coordinates(
        input.reference_positions,
        input.reference_mask.bool() & atom_mask,
        dtype=dtype,
        feature="ODesign reference positions",
    )
    result.observed_positions = project_model_coordinates(
        input.observed_positions,
        input.observed_mask.bool() & atom_mask,
        dtype=dtype,
        feature="ODesign observed positions",
    )
    for name in ("atom_features", "reference_charges"):
        setattr(
            result,
            name,
            project_model_features(getattr(input, name), atom_mask[..., None], dtype=dtype, feature=f"ODesign {name}"),
        )
    for name in ("token_features", "design_features"):
        setattr(
            result,
            name,
            project_model_features(getattr(input, name), token_mask[..., None], dtype=dtype, feature=f"ODesign {name}"),
        )
    pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    result.token_bonds = project_model_features(
        input.token_bonds, pair_mask, dtype=dtype, feature="ODesign token relations"
    )
    result.constraint_distogram = project_model_features(
        input.constraint_distogram, pair_mask[..., None], dtype=dtype, feature="ODesign constraint distances"
    )
    msa_available = input.msa_mask.bool() & token_mask.unsqueeze(-2)
    result.msa_tokens = project_model_categories(
        input.msa_tokens, msa_available, num_classes=_VOCAB_SIZE, feature="ODesign MSA tokens"
    )
    result.msa_deletion_counts = project_model_counts(
        input.msa_deletion_counts, msa_available, feature="ODesign MSA deletion counts"
    )
    result.msa_mask = msa_available
    return result
