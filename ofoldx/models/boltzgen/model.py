r"""
BoltzGen structure-design and inverse-folding models with their typed tensor contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.bonds import SymbolicBond
from ...data.conditions import Contact, Hotspot
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem
from ...data.placement import (
    PlacementMap,
    PlacementSpec,
    SymbolicResidueSite,
    validate_placement,
)
from ...model import Model
from ...modules.atom import (
    PackedAtomToTokenEncoder,
    PackedDenoisingAtomToTokenEncoder,
    PackedReferenceAtomEncoder,
    PackedTokenToAtomDecoder,
)
from ...modules.atom.pair_encoding import encode_atom_pair
from ...modules.atom.window import gather_key_windows
from ...modules.attention import GatedMultiheadAttention, PairBiasedSelfAttention
from ...modules.backend import LayerNorm
from ...modules.contact import BoltzContactConditioning
from ...modules.diffusion.conditioning import PairwiseConditioning, condition_single
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.dropout import RowwiseDropout
from ...modules.embedding.factorized import FactorizedPairLinear
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.feed_forward import SwiGluFeedForward
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import PairSymmetrizedDistogramHead
from ...modules.transformer import (
    ConditionedTransformerEncoder,
    ConditionedTransformerEncoderBlock,
    TripleProductConditionedTransition,
)
from ...modules.transition import ConditionedTransition, Transition
from ...modules.trunk.msa import (
    CategoricalMsaTrunk,
    MsaPairWeightedAveraging,
    OuterProductMean,
    TrianglePairUpdateBlock,
)
from ...modules.trunk.pairformer import PairformerStack
from .config import BoltzGenConfig, BoltzGenInverseFoldingConfig

_NUM_BOND_TYPES = 6

# Molecular categories are protein, DNA, RNA, and non-polymer.
_NUM_MOLECULE_TYPES = 4

_NUM_RESIDUE_TYPES = 33  # const.num_tokens

_CANONICALS_OFFSET = 2  # const.canonicals_offset: the 20 canonical amino acids occupy ids [2, 22)

NUM_CANONICAL_TOKENS = 20  # len(const.canonical_tokens)

# The 20 canonical amino acids (three-letter CCD codes) in vocabulary order (residue-token ids ``[2, 22)``);
# matches ``const.canonical_tokens``. Indexed by ``inverse_fold_restriction`` and ``aa_constraint_mask``.
CANONICAL_TOKENS: tuple[str, ...] = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)


class BoltzGenModel(Model):
    r"""
    Generate conditioned biomolecular structures with BoltzGen.

    Accepts [`BoltzGenInput`][ofoldx.models.boltzgen.model.BoltzGenInput] and returns
    [`BoltzGenOutput`][ofoldx.models.boltzgen.model.BoltzGenOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = BoltzGenConfig
    config: BoltzGenConfig

    def __init__(
        self,
        config: BoltzGenConfig | None = None,
    ) -> None:
        config = BoltzGenConfig() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        single_size, pair_size = config.single_size, config.pair_size
        input_embeddings_config, trunk, denoiser_config, sampler = (
            config.input_embeddings,
            config.trunk,
            config.denoiser,
            config.sampler,
        )
        design = config.design

        self.input_embeddings = BoltzGenInputEmbeddings(
            single_size=single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=input_embeddings_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=input_embeddings_config.num_atom_encoder_attention_heads,
            vocab_size=input_embeddings_config.vocab_size,
            msa_profile_size=input_embeddings_config.msa_profile_size,
            num_method_types=input_embeddings_config.num_method_types,
            num_molecule_types=input_embeddings_config.num_molecule_types,
            add_binding_specification=input_embeddings_config.add_binding_specification,
            add_ss_specification=input_embeddings_config.add_ss_specification,
            backend="auto",
        )

        self.single_init_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(config.num_bond_types, pair_size)
        self.contact_conditioning = BoltzContactConditioning(
            pair_size, cutoff_min=config.conditioning_cutoff_min, cutoff_max=config.conditioning_cutoff_max
        )
        self.single_recycle_norm = LayerNorm(single_size)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        template_embedder_type = (
            BoltzGenMiniformerTemplateEmbedder if config.template.miniformer_blocks else BoltzGenTemplateEmbedder
        )
        self.template_embedder = template_embedder_type(
            pair_size=pair_size,
            template_size=design.template_size,
            num_blocks=design.num_template_blocks,
            vocab_size=input_embeddings_config.vocab_size,
            num_bins=config.template.num_distogram_bins,
            min_dist=config.template.min_distance,
            max_dist=config.template.max_distance,
            triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.template.num_triangle_attention_heads,
            backend="auto",
        )
        self.token_distance_embedder = (
            BoltzGenTokenDistanceEmbedder(
                pair_size=pair_size,
                token_distance_size=design.token_distance_size,
                num_blocks=design.num_token_distance_blocks,
                num_bins=config.template.num_distogram_bins,
                distance_gaussian_dim=design.distance_gaussian_dim,
                min_dist=config.template.min_distance,
                max_dist=config.template.max_distance,
                triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
                num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                backend="auto",
            )
            if design.use_token_distance
            else None
        )

        self.msa_trunk = (
            CategoricalMsaTrunk(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                msa_feature_size=trunk.msa_feature_size,
                num_blocks=trunk.num_msa_blocks,
                vocab_size=input_embeddings_config.vocab_size,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.pair_dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                backend="auto",
            )
            if not trunk.msa_miniformer_blocks
            else BoltzGenMiniformerMsaTrunk(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                msa_feature_size=trunk.msa_feature_size,
                num_blocks=trunk.num_msa_blocks,
                vocab_size=input_embeddings_config.vocab_size,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.pair_dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                backend="auto",
            )
        )
        self.trunk_stack = (
            BoltzGenMiniformerStack(
                num_blocks=trunk.num_trunk_blocks,
                single_size=single_size,
                pair_size=pair_size,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.pair_dropout,
                backend="auto",
            )
            if trunk.use_miniformer
            else PairformerStack(
                num_blocks=trunk.num_trunk_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.pair_dropout,
                backend="auto",
            )
        )
        self.distogram_head = PairSymmetrizedDistogramHead(pair_size=pair_size, num_bins=config.num_distogram_bins)

        self.denoiser_conditioning = BoltzGenDenoiserConditioning(
            single_size=single_size,
            pair_size=pair_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            atom_pair_hidden_size=denoiser_config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
        )
        denoiser = BoltzGenDenoiser(
            single_size=single_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            fourier_embedding_size=config.fourier_embedding_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
            backend="auto",
        )
        self.coordinate_diffusion = BoltzGenCoordinateDiffusion(
            denoiser,
            sigma_data=denoiser_config.sigma_data,
            min_noise_level=sampler.min_noise_level,
            max_noise_level=sampler.max_noise_level,
            rho=sampler.rho,
            churn_rate=sampler.churn_rate,
            noise_scale=sampler.noise_scale,
            step_scale=sampler.step_scale,
            num_sampling_steps=sampler.num_sampling_steps,
            alignment_reverse_diff=denoiser_config.alignment_reverse_diff,
            sampling_schedule=sampler.sampling_schedule,
            time_dilation=sampler.time_dilation,
            time_dilation_start=sampler.time_dilation_start,
            time_dilation_end=sampler.time_dilation_end,
        )
        self.bfactor_head = nn.Linear(single_size, config.num_distogram_bins)
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Apply the BoltzGen residual-update initialization.
        """

        _zero_initialize_boltzgen_updates(self)

    def forward(
        self,
        input: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenOutput:
        r"""
        Run one stochastic denoising step for structure training.

        ``B`` is batch size, ``N`` token count, and ``A`` atom count.

        Args:
            input: Model input with ``target_positions`` of shape ``(B, A, 3)`` in
                angstroms. ``atom_mask`` controls coordinate centering; ``target_mask``
                identifies resolved target atoms for the training objective.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.
            generator: Random generator used for rigid augmentation, noise-level
                sampling, and coordinate noise.

        Returns:
            A [`BoltzGenOutput`][] with denoised ``coordinates`` of shape
            ``(B, A, 3)``, ``distogram_logits`` of shape ``(B, N, N, D)``,
            per-token ``bfactor_logits`` of shape ``(B, N, D_b)``, augmented
            targets of shape ``(B, A, 3)``, and one ``noise_level`` per batch item.
        """

        trunk = self._run_trunk(input, recycling_steps=recycling_steps)
        conditioning = self.denoiser_conditioning(
            trunk.single,
            trunk.pair,
            trunk.relative_position_encoding,
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
        )
        target = cast(Tensor, input.target_positions)
        atom_mask = input.atom_mask
        augmented_target_coordinates = _center_random_augmentation(target, atom_mask, generator=generator)
        sigmas = self.coordinate_diffusion.noise_distribution(
            atom_mask.shape[0],
            device=augmented_target_coordinates.device,
            dtype=augmented_target_coordinates.dtype,
            generator=generator,
        )
        noise = torch.randn(
            augmented_target_coordinates.shape,
            device=augmented_target_coordinates.device,
            dtype=augmented_target_coordinates.dtype,
            generator=generator,
        )
        noisy_coordinates = augmented_target_coordinates + sigmas.reshape(-1, 1, 1) * noise
        denoised_coordinates = self.coordinate_diffusion.preconditioned_network_forward(
            noisy_coordinates,
            sigmas,
            multiplicity=1,
            atom_mask=atom_mask,
            single_inputs=trunk.single_inputs,
            single_trunk=trunk.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        return BoltzGenOutput(
            coordinates=denoised_coordinates,
            distogram_logits=trunk.distogram_logits,
            bfactor_logits=self.bfactor_head(trunk.single),
            augmented_target_coordinates=augmented_target_coordinates,
            noise_level=sigmas,
        )

    def design(
        self,
        input: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> BoltzGenDesignOutput:
        r"""
        Sample atom coordinates for a processed design problem.

        ``B`` is batch size, ``S`` is ``num_samples``, ``N`` is token count, and
        ``A`` is atom count.

        Args:
            input: Processed structure-design input.
            num_samples: Number of coordinate samples.
            num_sampling_steps: Number of EDM sampling steps. ``None`` uses the
                configured value.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.
            generator: Random generator used by coordinate sampling.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured
                value.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured
                value.

        Returns:
            A [`BoltzGenDesignOutput`][] with ``coordinates`` of shape
            ``(B, S, A, 3)`` in angstroms, ``distogram_logits`` of shape
            ``(B, N, N, D)``, and ``bfactor_logits`` of shape
            ``(B, S, N, D_b)``.
        """

        return self._design(
            input,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
        )

    @torch.no_grad()
    def _design(
        self,
        input: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> BoltzGenDesignOutput:
        r"""
        Draw reverse-diffusion structure samples from a preprocessed input.

        Args:
            input: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional per-call sampler step-scale override.
            noise_scale: Optional per-call sampler noise-scale override.

        """

        trunk = self._run_trunk(input, recycling_steps=recycling_steps)
        conditioning = self.denoiser_conditioning(
            trunk.single,
            trunk.pair,
            trunk.relative_position_encoding,
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
        )
        coordinates = self.coordinate_diffusion.sample(
            input.atom_mask,
            num_sampling_steps=num_sampling_steps,
            multiplicity=num_samples,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
            single_inputs=trunk.single_inputs,
            single_trunk=trunk.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        batch_size = input.token_mask.shape[0]
        return BoltzGenDesignOutput(
            coordinates=coordinates.reshape(batch_size, num_samples, *coordinates.shape[1:]),
            distogram_logits=trunk.distogram_logits,
            bfactor_logits=self.bfactor_head(trunk.single).unsqueeze(1).expand(-1, num_samples, -1, -1),
        )

    def _run_trunk(
        self,
        input: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
    ) -> _BoltzGenTrunkOutput:
        r"""
        Run the structure trunk.

        Args:
            input: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.

        Returns:
            Trunk representations and distogram logits.

        """

        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
            binding_type=input.binding_type,
            design_mask=input.design_mask,
            ss_type=input.ss_type,
        )

        single_init = self.single_init_projection(single_inputs)
        pair_init = self.pair_left_projection(single_inputs).unsqueeze(-2)
        pair_init = pair_init + self.pair_right_projection(single_inputs).unsqueeze(-3)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            cyclic_period=input.cyclic_period,
            dtype=single_init.dtype,
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        pair_init = pair_init + relative_position_encoding
        pair_init = pair_init + self.token_bond_projection(input.token_bonds.to(pair_init.dtype))
        pair_init = pair_init + self.token_bond_type_embedding(input.type_bonds.long())
        pair_init = pair_init + self.contact_conditioning(
            input.contact_conditioning.to(pair_init.dtype), input.contact_threshold.to(pair_init.dtype)
        )

        token_mask = input.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(steps + 1):
            grad_enabled = torch.is_grad_enabled() and cycle == steps
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if self.token_distance_embedder is not None:
                    pair = pair + self.token_distance_embedder(
                        pair,
                        input,
                        relative_position_encoding,
                        pair_mask=pair_mask,
                    )
                if input.template_residue_type_features is not None:
                    pair = pair + self.template_embedder(pair, input, pair_mask=pair_mask)
                if input.msa_tokens is not None:
                    pair = pair + self.msa_trunk(
                        pair,
                        single_inputs,
                        msa_tokens=input.msa_tokens,
                        has_deletion=input.has_deletion,
                        deletion_value=input.deletion_value,
                        is_paired=input.msa_paired,
                        msa_mask=input.msa_mask,
                        token_mask=token_mask,
                    )
                single, pair = self.trunk_stack(single, pair, pair_mask=pair_mask)

        return _BoltzGenTrunkOutput(
            single_inputs=single_inputs,
            single=single,
            pair=pair,
            relative_position_encoding=relative_position_encoding,
            distogram_logits=self.distogram_head(pair),
        )


class BoltzGenInverseFoldingModel(Model):
    r"""
    Design sequences for fixed biomolecular structures with BoltzGen.

    Accepts [`BoltzGenInverseFoldingInput`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldingInput] and returns
    [`BoltzGenInverseFoldingOutput`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldingOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = BoltzGenInverseFoldingConfig
    config: BoltzGenInverseFoldingConfig

    def __init__(self, config: BoltzGenInverseFoldingConfig | None = None) -> None:
        config = BoltzGenInverseFoldingConfig() if config is None else config
        super().__init__(config)
        inverse_folding = config.inverse_folding
        self.input_embeddings = BoltzGenInputEmbeddings(
            single_size=config.single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=inverse_folding.num_query_atoms,
            num_key_atoms=inverse_folding.num_key_atoms,
            num_atom_encoder_blocks=inverse_folding.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=inverse_folding.num_atom_encoder_attention_heads,
            vocab_size=config.input_embeddings.vocab_size,
            msa_profile_size=config.input_embeddings.msa_profile_size,
            num_method_types=config.input_embeddings.num_method_types,
            num_molecule_types=config.input_embeddings.num_molecule_types,
            backend="auto",
        )
        self.inverse_folding_encoder = BoltzGenInverseFoldingEncoder(
            single_size=config.single_size,
            node_size=inverse_folding.node_size,
            edge_size=inverse_folding.edge_size,
            hidden_size=inverse_folding.hidden_size,
            num_blocks=inverse_folding.num_encoder_blocks,
            num_attention_heads=inverse_folding.num_attention_heads,
            num_neighbors=inverse_folding.num_neighbors,
            dropout=inverse_folding.dropout,
            softmax_dropout=inverse_folding.softmax_dropout,
            coordinate_noise=inverse_folding.coordinate_noise,
        )
        self.inverse_folding_decoder = BoltzGenInverseFoldingDecoder(
            node_size=inverse_folding.node_size,
            edge_size=inverse_folding.edge_size,
            hidden_size=inverse_folding.hidden_size,
            num_blocks=inverse_folding.num_decoder_blocks,
            num_attention_heads=inverse_folding.num_attention_heads,
            dropout=inverse_folding.dropout,
            softmax_dropout=inverse_folding.softmax_dropout,
        )
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Apply the BoltzGen residual-update initialization.
        """

        _zero_initialize_boltzgen_updates(self)

    def forward(
        self,
        input: BoltzGenInverseFoldingInput,
        *,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingOutput:
        r"""
        Predict residue-type logits from a fixed structure.

        Args:
            input: Processed inverse-folding input with ``B`` batch items and ``N``
                tokens.
            generator: Random generator used for teacher-forcing visibility.

        Returns:
            A [`BoltzGenInverseFoldingOutput`][] whose ``logits`` have shape
            ``(B, N, V)``, where ``V`` is the residue vocabulary size.
        """

        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
        )
        edge_index, valid_mask, node, edge = self.inverse_folding_encoder(input, single_inputs=single_inputs)
        return BoltzGenInverseFoldingOutput(
            logits=self.inverse_folding_decoder(node, edge, edge_index, valid_mask, input, generator=generator),
        )

    def design(
        self,
        input: BoltzGenInverseFoldingInput,
        *,
        temperature: float | None = None,
        inverse_fold_restriction: tuple[str, ...] = (),
        tie_symmetric_sequences: bool = False,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingDesignOutput:
        r"""
        Sample residue identities for a fixed structure.

        Args:
            input: Processed inverse-folding input with ``B`` batch items and ``N``
                tokens.
            temperature: Categorical sampling temperature. ``None`` selects the
                highest-logit residue at each decoded position.
            inverse_fold_restriction: Canonical amino acids to exclude from this call.
            tie_symmetric_sequences: Whether symmetric copies share one sample.
            generator: Random generator used for decode order and categorical sampling.

        Returns:
            A [`BoltzGenInverseFoldingDesignOutput`][] with ``logits`` of shape
            ``(B, N, V)`` and sampled ``tokens`` of shape ``(B, N)``.
        """

        return self._design(
            input,
            temperature=temperature,
            inverse_fold_restriction=inverse_fold_restriction,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )

    @torch.no_grad()
    def _design(
        self,
        input: BoltzGenInverseFoldingInput,
        *,
        temperature: float | None = None,
        inverse_fold_restriction: tuple[str, ...] = (),
        tie_symmetric_sequences: bool = False,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingDesignOutput:
        r"""
        Autoregressively design residue tokens from the conditioned structure.

        Args:
            input: Typed model input.
            temperature: Optional categorical sampling temperature. ``None`` selects argmax decoding.
            inverse_fold_restriction: Canonical amino acids to exclude from this design call.
            tie_symmetric_sequences: Whether matching positions across symmetric copies share one sample.
            generator: Optional PyTorch random generator for decode order and categorical sampling.

        Returns:
            Design-generator output carrying sampled sequence tokens and logits.
        """

        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
        )
        edge_index, valid_mask, node, edge = self.inverse_folding_encoder(input, single_inputs=single_inputs)
        logits, residue_type = self.inverse_folding_decoder.sample(
            node,
            edge,
            edge_index,
            valid_mask,
            input,
            temperature=temperature,
            inverse_fold_restriction=inverse_fold_restriction,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )
        return BoltzGenInverseFoldingDesignOutput(
            logits=logits,
            tokens=residue_type.argmax(dim=-1),
        )


class BoltzGenInputEmbeddings(nn.Module):
    r"""
    Compose atom-derived token states with token-level encodings and conditioning embeddings.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 388,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        vocab_size: int = 33,
        msa_profile_size: int = 34,
        num_method_types: int = 12,
        num_molecule_types: int = 4,
        add_method_conditioning: bool = True,
        add_modified_flag: bool = True,
        add_cyclic_flag: bool = True,
        add_mol_type_feat: bool = True,
        add_binding_specification: bool = False,
        add_ss_specification: bool = False,
        num_binding_types: int = 3,
        num_ss_types: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_encoder = PackedReferenceAtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_pair_norm = LayerNorm(atom_pair_hidden_size, backend=backend)
        self.atom_pair_bias_projection = nn.Linear(
            atom_pair_hidden_size,
            num_atom_encoder_blocks * num_atom_encoder_attention_heads,
            bias=False,
        )
        self.atom_attention_encoder = PackedAtomToTokenEncoder(
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )

        self.residue_type_projection = nn.Linear(vocab_size, single_size, bias=False)
        self.msa_profile_projection = nn.Linear(msa_profile_size, single_size, bias=False)
        self.method_embedding: nn.Embedding | None = None
        self.modified_embedding: nn.Embedding | None = None
        self.cyclic_projection: nn.Linear | None = None
        self.molecule_type_embedding: nn.Embedding | None = None
        self.binding_specification_embedding: nn.Embedding | None = None
        self.design_mask_embedding: nn.Embedding | None = None
        self.ss_specification_embedding: nn.Embedding | None = None
        if add_method_conditioning:
            self.method_embedding = nn.Embedding(num_method_types, single_size)
        if add_modified_flag:
            self.modified_embedding = nn.Embedding(2, single_size)
        if add_cyclic_flag:
            self.cyclic_projection = nn.Linear(1, single_size, bias=False)
        if add_mol_type_feat:
            self.molecule_type_embedding = nn.Embedding(num_molecule_types, single_size)
        if add_binding_specification:
            self.binding_specification_embedding = nn.Embedding(num_binding_types, single_size)
            self.design_mask_embedding = nn.Embedding(2, single_size)
        if add_ss_specification:
            self.ss_specification_embedding = nn.Embedding(num_ss_types, single_size)
        for module in (
            self.method_embedding,
            self.modified_embedding,
            self.molecule_type_embedding,
            self.binding_specification_embedding,
            self.design_mask_embedding,
            self.ss_specification_embedding,
        ):
            if module is not None:
                nn.init.zeros_(module.weight)
        if self.cyclic_projection is not None:
            nn.init.zeros_(self.cyclic_projection.weight)

    def forward(
        self,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        residue_type_features: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        method_feature: Tensor | None = None,
        modified: Tensor | None = None,
        cyclic_period: Tensor | None = None,
        molecule_type: Tensor | None = None,
        binding_type: Tensor | None = None,
        design_mask: Tensor | None = None,
        ss_type: Tensor | None = None,
    ) -> Tensor:
        atom_query, atom_conditioning, atom_pair = self.atom_encoder(
            atom_features, reference_positions, atom_mask, reference_space_id
        )
        atom_pair_bias = self.atom_pair_bias_projection(self.atom_pair_norm(atom_pair))
        token_from_atom = self.atom_attention_encoder(
            atom_query,
            atom_conditioning,
            atom_pair_bias,
            atom_mask,
            atom_to_token,
            residue_type_features.shape[-2],
        )

        single = token_from_atom
        single = single + self.residue_type_projection(residue_type_features.to(single.dtype))
        msa_features = torch.cat([msa_profile, deletion_mean.unsqueeze(-1)], dim=-1)
        single = single + self.msa_profile_projection(msa_features.to(single.dtype))

        if self.method_embedding is not None:
            single = single + self.method_embedding(cast(Tensor, method_feature))
        if self.modified_embedding is not None:
            single = single + self.modified_embedding(cast(Tensor, modified))
        if self.cyclic_projection is not None:
            single = single + self.cyclic_projection(
                cast(Tensor, cyclic_period).clamp(max=1.0).unsqueeze(-1).to(single.dtype)
            )
        if self.molecule_type_embedding is not None:
            single = single + self.molecule_type_embedding(cast(Tensor, molecule_type))
        if self.binding_specification_embedding is not None:
            single = single + self.binding_specification_embedding(cast(Tensor, binding_type))
        if self.design_mask_embedding is not None:
            single = single + self.design_mask_embedding(cast(Tensor, design_mask).int())
        if self.ss_specification_embedding is not None:
            single = single + self.ss_specification_embedding(cast(Tensor, ss_type))
        return single


class BoltzGenDenoiserAtomEncoder(nn.Module):
    r"""
    Build trunk-conditioned atom states and atom-pair biases for coordinate denoising.
    """

    def __init__(
        self,
        atom_feature_size: int = 388,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        single_size: int = 384,
        pair_size: int = 128,
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
        self.single_trunk_norm = LayerNorm(single_size)
        self.single_trunk_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.pair_trunk_norm = LayerNorm(pair_size)
        self.pair_trunk_projection = nn.Linear(
            pair_size,
            atom_pair_hidden_size,
            bias=False,
        )
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
        *,
        atom_to_token: Tensor,
        single_trunk: Tensor,
        pair_trunk: Tensor,
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

        single_proj = self.single_trunk_projection(self.single_trunk_norm(single_trunk))
        atom_single = atom_single + single_proj.gather(
            1, atom_to_token.unsqueeze(-1).expand(-1, -1, single_proj.shape[-1])
        )
        pair_from_trunk = self.pair_trunk_projection(self.pair_trunk_norm(pair_trunk))
        token_windows = pad_atoms(atom_to_token)
        query_token = token_windows.view(batch, num_windows, num_query_atoms)
        key_token = to_keys(token_windows.unsqueeze(-1)).reshape(batch, num_windows, num_key_atoms)
        batch_index = torch.arange(batch, device=atom_single.device)[:, None, None, None]
        pair_gathered = pair_from_trunk[batch_index, query_token[:, :, :, None], key_token[:, :, None, :]]
        pair_bias = pair_gathered * mask_keys.unsqueeze(-1).to(distance_vectors.dtype)

        single_windows = F.relu(pad_atoms(atom_single))
        atom_pair = encode_atom_pair(
            self,
            distance_vectors,
            single_windows.view(batch, num_windows, num_query_atoms, 1, -1),
            to_keys(single_windows).view(batch, num_windows, 1, num_key_atoms, -1),
            distance_term_mask=valid,
            inverse_term_mask=valid,
            valid_mask=valid,
            pair_bias=pair_bias,
            prefix_ndim=1,
        )
        return atom_query, atom_single, atom_pair


class BoltzGenInverseFoldingEncoder(nn.Module):
    r"""
    kNN-graph encoder over tokens (BoltzGen inverse folding).
    """

    def __init__(
        self,
        single_size: int = 384,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_neighbors: int = 30,
        max_relative_position: int = 32,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        coordinate_noise: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_neighbors = num_neighbors
        self.max_relative_position = max_relative_position
        self.coordinate_noise = coordinate_noise
        # geo (4x4 backbone-atom distances x 16 RBF = 256)
        # + edge attrs (2*max_relative_position+2 + same_chain + same_residue)
        # + bond (token_bonds + one-hot bond type).
        self.edge_input_size = 256 + (2 * max_relative_position + 2) + 2 + 1 + (_NUM_BOND_TYPES + 1)

        self.node_input_projection = nn.Linear(single_size, node_size)
        self.edge_input_projection = nn.Linear(self.edge_input_size, edge_size)
        self.blocks = nn.ModuleList(
            BoltzGenInverseFoldingGraphEncoderBlock(
                node_size=node_size,
                edge_size=edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
            )
            for _ in range(num_blocks)
        )
        self.distance_smearing = BoltzGenGaussianSmearing(start=0.0, stop=20.0, num_gaussians=16)

    @torch.no_grad()
    def build_knn_graph(self, input: BoltzGenInverseFoldingInput) -> tuple[Tensor, Tensor, Tensor]:
        valid_mask = input.token_mask.bool()
        batch_size, num_tokens = valid_mask.shape
        valid_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        packed_index = (valid_mask.flatten().cumsum(dim=0) - 1).view(batch_size, num_tokens)
        num_neighbors = min(self.num_neighbors, num_tokens)

        distances = torch.cdist(input.center_coordinates, input.center_coordinates)
        distances = distances.masked_fill(~valid_pair, float("inf"))
        src = torch.topk(distances, num_neighbors, largest=False).indices  # (B, N, num_neighbors)
        dst = torch.arange(num_tokens, device=src.device)[None, :, None].expand(batch_size, -1, num_neighbors)

        # token_bonds carries a trailing singleton (B, N, N, 1); drop it so the gather index (B, N, num_neighbors)
        # matches its rank (the BoltzGen reference does the same, asserting shape[-1] == 1). type_bonds is (B, N, N).
        token_bonds = torch.gather(input.token_bonds[..., 0], 2, src)
        type_bonds = torch.gather(input.type_bonds, 2, src)

        src, dst = src.flatten(start_dim=1), dst.flatten(start_dim=1)
        token_bonds, type_bonds = token_bonds.flatten(), type_bonds.flatten()
        edge_valid = torch.gather(valid_mask, 1, src).flatten() & torch.gather(valid_mask, 1, dst).flatten()

        src = torch.gather(packed_index, 1, src).flatten()[edge_valid]
        dst = torch.gather(packed_index, 1, dst).flatten()[edge_valid]
        edge_index = torch.stack([src, dst], dim=0)

        token_bonds, type_bonds = token_bonds[edge_valid], type_bonds[edge_valid]
        type_bonds = F.one_hot(type_bonds, num_classes=_NUM_BOND_TYPES + 1)
        bond = torch.cat([token_bonds[..., None], type_bonds], dim=-1)
        return edge_index, valid_mask, bond

    @torch.no_grad()
    def edge_geometry(self, input: BoltzGenInverseFoldingInput, edge_index: Tensor, valid_mask: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        batch_size, num_tokens = valid_mask.shape
        coords = input.atom_coordinates
        if self.training and self.coordinate_noise > 0:
            coords = coords + torch.randn_like(coords) * self.coordinate_noise
        backbone = _gather_backbone4(
            input.token_to_bb4_atoms,
            coords,
            batch_size,
            num_tokens,
        )[valid_mask]
        distance = torch.norm(backbone[src, None] - backbone[dst, :, None], dim=-1).flatten(start_dim=1)
        return self.distance_smearing(distance)

    @torch.no_grad()
    def edge_attributes(self, input: BoltzGenInverseFoldingInput, edge_index: Tensor, valid_mask: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        chain_index = input.chain_index[valid_mask]
        same_chain = chain_index[src] == chain_index[dst]
        residue = input.residue_index[valid_mask]
        same_residue = residue[src] == residue[dst]
        delta = torch.clip(residue[dst] - residue[src] + self.max_relative_position, 0, 2 * self.max_relative_position)
        delta = torch.where(same_chain, delta, torch.zeros_like(delta) + 2 * self.max_relative_position + 1)
        relative = F.one_hot(delta, 2 * self.max_relative_position + 2)
        return torch.cat([relative.float(), same_chain[..., None].float(), same_residue[..., None].float()], dim=-1)

    def forward(
        self,
        input: BoltzGenInverseFoldingInput,
        *,
        single_inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_index, valid_mask, bond = self.build_knn_graph(input)
        geometry = self.edge_geometry(input, edge_index, valid_mask)
        edge_attr = self.edge_attributes(input, edge_index, valid_mask)

        node = self.node_input_projection(single_inputs[valid_mask])
        edge = self.edge_input_projection(torch.cat([geometry, edge_attr, bond], dim=-1))
        for block in self.blocks:
            node, edge = block(node, edge, edge_index)
        return edge_index, valid_mask, node, edge


class BoltzGenInverseFoldingGraphFeedForward(nn.Module):
    r"""
    GELU feed-forward computation for packed inverse-folding graph features.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.hidden_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)
        )
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.activation(self.input_projection(features))
        for projection in self.hidden_projections:
            hidden = self.activation(projection(hidden))
        return self.output_projection(hidden)


class BoltzGenInverseFoldingGraphAttention(nn.Module):
    r"""
    Aggregate packed edge context with destination-grouped attention.
    """

    def __init__(
        self,
        node_size: int,
        context_size: int,
        hidden_size: int,
        num_attention_heads: int,
        softmax_dropout: float,
    ) -> None:
        super().__init__()
        self.softmax_dropout = softmax_dropout
        self.weight_feed_forward = BoltzGenInverseFoldingGraphFeedForward(
            node_size + context_size,
            hidden_size,
            num_attention_heads,
            num_hidden_layers=2,
        )
        self.value_feed_forward = BoltzGenInverseFoldingGraphFeedForward(
            context_size,
            hidden_size,
            node_size,
            num_hidden_layers=2,
        )
        self.output_projection = nn.Linear(node_size * num_attention_heads, node_size)

    def forward(self, node: Tensor, context: Tensor, destination_index: Tensor) -> Tensor:
        weight = self.weight_feed_forward(torch.cat([node[destination_index], context], dim=1))
        value = self.value_feed_forward(context)
        if self.training and self.softmax_dropout > 0:
            weight = _softmax_dropout(weight, destination_index, node.shape[0], self.softmax_dropout)
        weight = _scatter_softmax(weight, destination_index, dim=0, dim_size=node.shape[0])
        message = _scatter_sum(
            weight.unsqueeze(-1) * value.unsqueeze(1),
            destination_index,
            dim=0,
            dim_size=node.shape[0],
        ).flatten(start_dim=1)
        return self.output_projection(message)


class BoltzGenInverseFoldingGraphEncoderBlock(nn.Module):
    r"""
    One MLP-attention GNN block with node and edge updates.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.edge_feed_forward = BoltzGenInverseFoldingGraphFeedForward(
            node_size * 2 + edge_size,
            hidden_size,
            edge_size,
        )
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_norm = nn.BatchNorm1d(edge_size)
        self.attention = BoltzGenInverseFoldingGraphAttention(
            node_size,
            node_size + edge_size,
            hidden_size,
            num_attention_heads,
            softmax_dropout,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.BatchNorm1d(node_size)
        self.node_feed_forward = BoltzGenInverseFoldingGraphFeedForward(
            node_size,
            hidden_size,
            node_size,
        )
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = nn.BatchNorm1d(node_size)

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        edge_update = self.edge_feed_forward(torch.cat([node[src], node[dst], edge], dim=1))
        edge = edge + self.edge_norm(self.edge_dropout(edge_update))
        attention_update = self.attention(node, torch.cat([node[src], edge], dim=1), dst)
        node = node + self.attention_norm(self.attention_dropout(attention_update))
        feed_forward_update = self.node_feed_forward(node)
        return node + self.node_norm(self.node_dropout(feed_forward_update)), edge


class BoltzGenInverseFoldingDecoder(nn.Module):
    r"""
    Graph decoder predicting residue-type logits (BoltzGen inverse folding).
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        vocab_size: int = _NUM_RESIDUE_TYPES,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        # Each block's edge state is cat[encoder_edge, neighbour_node + residue_rep] = edge_size + node_size.
        decoder_edge_size = edge_size + node_size
        self.blocks = nn.ModuleList(
            BoltzGenInverseFoldingGraphDecoderBlock(
                node_size=node_size,
                edge_size=decoder_edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
            )
            for _ in range(num_blocks)
        )
        self.residue_type_projection = nn.Linear(vocab_size, node_size)
        self.residue_type_head = nn.Linear(node_size, vocab_size, bias=False)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
        valid_mask: Tensor,
        input: BoltzGenInverseFoldingInput,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        with torch.no_grad():
            # Teacher forcing with random per-edge visibility: an edge reveals its source residue type only when
            # rand[src] < rand[dst], so each node sees a random subset of neighbour identities.
            rand = torch.rand(node.shape[0], device=node.device, generator=generator)
            visibility = rand[src] < rand[dst]
            known = input.residue_type_features.bool()[valid_mask][src] * visibility[:, None]
            known = known.to(edge.dtype)
        residue_rep = self.residue_type_projection(known)
        neighbours = torch.cat([edge, node[src] + residue_rep], dim=-1)

        for block in self.blocks:
            node = block(node, neighbours, edge_index)

        logits = self.residue_type_head(node)
        batch_size, num_tokens = valid_mask.shape
        dense = logits.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense[valid_mask] = logits
        return dense

    @torch.no_grad()
    def sample(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
        valid_mask: Tensor,
        input: BoltzGenInverseFoldingInput,
        *,
        temperature: float | None = None,
        inverse_fold_restriction: tuple[str, ...] = (),
        tie_symmetric_sequences: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = node.shape[0]
        design_mask = cast(Tensor, input.design_mask).bool()[valid_mask]
        decoded = node.new_zeros(num_nodes, self.vocab_size)
        logits = node.new_zeros(num_nodes, self.vocab_size)

        constraint_mask = None if input.aa_constraint_mask is None else input.aa_constraint_mask[valid_mask]
        per_residue_mask = _build_constraint_logit_mask(
            num_nodes=num_nodes,
            aa_constraint_mask=constraint_mask,
            inverse_fold_restriction=inverse_fold_restriction,
            device=node.device,
            dtype=node.dtype,
        )

        order = torch.randperm(num_nodes, device=node.device, generator=generator).tolist()
        if (~design_mask).any():
            known = input.residue_type_features[valid_mask][~design_mask].to(decoded.dtype)
            decoded[~design_mask] = known
            logits[~design_mask] = known
            fixed = set(torch.where(~design_mask)[0].tolist())
            order = [position for position in order if position not in fixed]

        symmetry_groups, position_to_group = (
            self._build_symmetric_groups(input, valid_mask, design_mask)
            if tie_symmetric_sequences and input.symmetric_group is not None
            else ({}, {})
        )
        sampled: set[int] = set()

        for position in order:
            if position in sampled:
                continue
            positions = (
                symmetry_groups.get(position_to_group[position], [position])
                if position in position_to_group
                else [position]
            )

            aggregated_logits: Tensor | None = None
            for pos in positions:
                incoming = dst == pos
                neighbours = torch.cat(
                    [edge[incoming], node[src[incoming]] + self.residue_type_projection(decoded[src[incoming]])],
                    dim=-1,
                )
                state = node[pos : pos + 1]
                for block in self.blocks:
                    state = block.sample(state, neighbours)
                position_logits = self.residue_type_head(state)
                aggregated_logits = (
                    position_logits if aggregated_logits is None else aggregated_logits + position_logits
                )
            aggregated_logits = cast(Tensor, aggregated_logits) / len(positions)

            canonical = (
                aggregated_logits[:, _CANONICALS_OFFSET : _CANONICALS_OFFSET + NUM_CANONICAL_TOKENS]
                + per_residue_mask[position : position + 1]
            )
            if temperature is None or temperature == 0:
                chosen = canonical.argmax(dim=-1)
            else:
                chosen = torch.multinomial(
                    torch.softmax(canonical / temperature, dim=-1),
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            one_hot = torch.nn.functional.one_hot(chosen + _CANONICALS_OFFSET, self.vocab_size).to(decoded.dtype)
            for pos in positions:
                decoded[pos] = one_hot
                logits[pos] = aggregated_logits
                sampled.add(pos)

        batch_size, num_tokens = valid_mask.shape
        dense_logits = logits.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense_logits[valid_mask] = logits
        dense_res_type = decoded.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense_res_type[valid_mask] = decoded
        return dense_logits, dense_res_type

    def _build_symmetric_groups(
        self, input: BoltzGenInverseFoldingInput, valid_mask: Tensor, design_mask: Tensor
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        metadata = torch.stack(
            [design_mask, cast(Tensor, input.symmetric_group)[valid_mask], input.residue_index[valid_mask]], dim=-1
        ).tolist()
        key_to_positions: dict[tuple[int, int], list[int]] = {}
        for position, (designable, group, residue_index) in enumerate(metadata):
            if not designable:
                continue
            if group <= 0:  # 0 = no symmetry group.
                continue
            key = (group, residue_index)
            key_to_positions.setdefault(key, []).append(position)

        groups: dict[int, list[int]] = {}
        position_to_group: dict[int, int] = {}
        group_id = 0
        for positions in key_to_positions.values():
            if len(positions) > 1:
                groups[group_id] = positions
                for position in positions:
                    position_to_group[position] = group_id
                group_id += 1
        return groups, position_to_group


class BoltzGenInverseFoldingGraphDecoderBlock(nn.Module):
    r"""
    One MLP-attention GNN decoder block updating nodes only.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 256,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention = BoltzGenInverseFoldingGraphAttention(
            node_size,
            edge_size,
            hidden_size,
            num_attention_heads,
            softmax_dropout,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.BatchNorm1d(node_size)
        self.node_feed_forward = BoltzGenInverseFoldingGraphFeedForward(
            node_size,
            hidden_size,
            node_size,
        )
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = nn.BatchNorm1d(node_size)

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> Tensor:
        return self._update(node, edge, edge_index[1])

    def sample(self, node: Tensor, edge: Tensor) -> Tensor:
        destination_index = torch.zeros(edge.shape[0], dtype=torch.long, device=edge.device)
        return self._update(node, edge, destination_index)

    def _update(self, node: Tensor, context: Tensor, destination_index: Tensor) -> Tensor:
        attention_update = self.attention(node, context, destination_index)
        node = node + self.attention_norm(self.attention_dropout(attention_update))
        feed_forward_update = self.node_feed_forward(node)
        return node + self.node_norm(self.node_dropout(feed_forward_update))


class BoltzGenTokenDistanceEmbedder(nn.Module):
    r"""
    Encode inter-token backbone geometry into an additive pair update.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_distance_size: int = 64,
        num_blocks: int = 2,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        distance_gaussian_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.hidden_norm = LayerNorm(token_distance_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, token_distance_size, bias=False)
        self.distance_feature_projection = nn.Linear(num_bins + 4 * pair_size, token_distance_size, bias=False)
        self.output_projection = nn.Linear(token_distance_size, pair_size, bias=False)
        self.blocks = nn.ModuleList(
            TrianglePairUpdateBlock(
                pair_size=token_distance_size,
                triangle_multiplication_hidden_size=token_distance_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                backend=backend,
            )
            for _ in range(num_blocks)
        )
        self.distance_encoder = BoltzGenDistanceTokenEncoder(
            distance_gaussian_dim=distance_gaussian_dim,
            pair_size=pair_size,
            output_size=pair_size,
        )

    def forward(
        self,
        pair: Tensor,
        input: BoltzGenInput,
        relative_position_encoding: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        coords = input.center_coordinates
        distances = torch.cdist(coords, coords)
        boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1, device=distances.device)
        distogram = F.one_hot((distances[..., None] > boundaries).sum(dim=-1).long(), num_classes=self.num_bins)

        distance_features = torch.cat([distogram, self.distance_encoder(relative_position_encoding, input)], dim=-1)
        distance_features = distance_features * input.token_distance_mask.unsqueeze(-1)
        value = self.pair_projection(self.pair_norm(pair)) + self.distance_feature_projection(
            distance_features.to(pair.dtype)
        )
        refined = value
        for block in self.blocks:
            refined = block(refined, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined)  # one outer residual around the (internally-residual) stack
        return self.output_projection(self.relu(value))


class BoltzGenTemplateEmbedder(nn.Module):
    r"""
    Embed template geometry as an additive pair update.
    """

    def __init__(
        self,
        pair_size: int = 128,
        template_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = 33,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.hidden_norm = LayerNorm(template_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, template_size, bias=False)
        self.template_feature_projection = FactorizedPairLinear(
            pair_size=num_bins + 5,
            token_size=vocab_size,
            output_size=template_size,
            bias=False,
            backend=backend,
        )
        self.output_projection = nn.Linear(template_size, pair_size, bias=False)
        self.blocks = nn.ModuleList(
            TrianglePairUpdateBlock(
                pair_size=template_size,
                triangle_multiplication_hidden_size=template_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    def forward(self, pair: Tensor, input: BoltzGenInput, pair_mask: Tensor | None = None) -> Tensor:
        residue_type_features = cast(Tensor, input.template_residue_type_features)
        frame_mask = cast(Tensor, input.template_frame_mask)
        frame_rotations = cast(Tensor, input.template_frame_rotations)
        frame_translations = cast(Tensor, input.template_frame_translations)
        beta_carbon_positions = cast(Tensor, input.template_beta_carbon_positions)
        alpha_carbon_positions = cast(Tensor, input.template_alpha_carbon_positions)
        beta_carbon_mask = cast(Tensor, input.template_beta_carbon_mask)
        template_mask = cast(Tensor, input.template_mask)
        visibility_index = cast(Tensor, input.template_visibility_index)
        present = template_mask.any(dim=2).float()
        count = present.sum(dim=1).clamp(min=1)
        batch_size, num_templates, num_tokens = residue_type_features.shape[:3]

        pair_beta_carbon_mask = (beta_carbon_mask[..., :, None] * beta_carbon_mask[..., None, :]).unsqueeze(-1)
        pair_frame_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        same_visibility = (visibility_index[..., :, None] == visibility_index[..., None, :]).float()

        beta_carbon_distances = torch.cdist(
            beta_carbon_positions,
            beta_carbon_positions,
        )
        boundaries = torch.linspace(
            self.min_dist,
            self.max_dist,
            self.num_bins - 1,
            device=beta_carbon_distances.device,
        )
        distogram = F.one_hot(
            (beta_carbon_distances[..., None] > boundaries).sum(dim=-1).long(),
            num_classes=self.num_bins,
        )

        rotation = frame_rotations.unsqueeze(2).transpose(-1, -2)
        translation = frame_translations.unsqueeze(2).unsqueeze(-1)
        targets = alpha_carbon_positions.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(rotation, targets - translation)
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector)).squeeze(-1)

        geometry = torch.cat(
            [
                distogram.to(pair.dtype),
                pair_beta_carbon_mask,
                unit_vector,
                pair_frame_mask,
            ],
            dim=-1,
        )
        geometry = geometry * same_visibility.unsqueeze(-1)
        template_residue_features = residue_type_features.to(pair.dtype)
        feature_value = self.template_feature_projection(geometry.to(pair.dtype), template_residue_features)
        value = self.pair_projection(self.pair_norm(pair[:, None])) + feature_value
        value = value.reshape(batch_size * num_templates, num_tokens, num_tokens, -1)
        if pair_mask is not None:
            pair_mask = (
                pair_mask[:, None]
                .expand(-1, num_templates, -1, -1)
                .reshape(batch_size * num_templates, num_tokens, num_tokens)
            )
        refined = value
        for block in self.blocks:
            refined = block(refined, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined).reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

        aggregated = (value * present[:, :, None, None, None]).sum(dim=1) / count[:, None, None, None].to(value.dtype)
        return self.output_projection(self.relu(aggregated))


class BoltzGenMiniformerTemplateEmbedder(BoltzGenTemplateEmbedder):
    r"""
    Embed template geometry with homogeneous BoltzGen Miniformer pair updates.
    """

    def __init__(
        self,
        pair_size: int = 128,
        template_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = 33,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.hidden_norm = LayerNorm(template_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, template_size, bias=False)
        self.template_feature_projection = FactorizedPairLinear(
            pair_size=num_bins + 5,
            token_size=vocab_size,
            output_size=template_size,
            bias=False,
            backend=backend,
        )
        self.output_projection = nn.Linear(template_size, pair_size, bias=False)
        self.blocks = nn.ModuleList(
            BoltzGenMiniTrianglePairUpdateBlock(pair_size=template_size, backend=backend) for _ in range(num_blocks)
        )


class BoltzGenDistanceTokenEncoder(nn.Module):
    r"""
    Per-backbone-atom pairwise distance features conditioned on the relative-position encoding.
    """

    def __init__(self, distance_gaussian_dim: int = 32, pair_size: int = 128, output_size: int = 128) -> None:
        super().__init__()
        self.distance_smearing = BoltzGenBroadcastGaussianSmearing(
            start=0.0,
            stop=2.0,
            num_gaussians=distance_gaussian_dim,
        )
        input_size = distance_gaussian_dim + 1 + pair_size
        self.distance_norm = LayerNorm(input_size)
        self.distance_feed_forward = SwiGluFeedForward(
            input_size,
            pair_size,
            output_size,
        )

    def forward(self, relative_position_encoding: Tensor, input: BoltzGenInput) -> Tensor:
        batch_size, num_tokens = relative_position_encoding.shape[:2]
        backbone = _gather_backbone4(
            input.token_to_bb4_atoms,
            input.atom_coordinates,
            batch_size,
            num_tokens,
        ).permute(0, 2, 1, 3)
        distance = (backbone.unsqueeze(-2) - backbone.unsqueeze(-3)).norm(dim=-1).unsqueeze(-1)  # (B, 4, N, N, 1)
        gaussian = self.distance_smearing(distance)  # (B, 4, N, N, gaussians)
        relative = relative_position_encoding.view(batch_size, 1, num_tokens, num_tokens, -1).expand(-1, 4, -1, -1, -1)
        bias_input = torch.cat([gaussian, distance, relative], dim=-1)
        bias = self.distance_feed_forward(self.distance_norm(bias_input))
        bias = bias.permute(0, 2, 3, 4, 1).reshape(batch_size, num_tokens, num_tokens, -1)
        return bias


class BoltzGenMiniformerMsaTrunk(nn.Module):
    r"""
    MSA-module variant whose pair track uses compact pair-only Miniformer blocks.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        msa_feature_size: int = 36,
        num_blocks: int = 3,
        vocab_size: int = 33,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.use_paired_feature = use_paired_feature
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.msa_projection = nn.Linear(msa_feature_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                BoltzGenMiniformerMsaBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_head_dim=msa_head_dim,
                    expansion=expansion,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        *,
        msa_tokens: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        is_paired: Tensor,
        msa_mask: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        features = [
            F.one_hot(msa_tokens.long(), num_classes=self.vocab_size).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa_representation = self.msa_projection(torch.cat(features, dim=-1))
        msa_representation = msa_representation + self.token_projection(single_inputs).unsqueeze(1)
        pair_mask = pair_mask_from_token_mask(token_mask)
        for block in self.blocks:
            pair, msa_representation = block(pair, msa_representation, pair_mask, msa_mask)
        return pair


class BoltzGenMiniformerMsaBlock(nn.Module):
    r"""
    Update MSA and pair features in a Miniformer block.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_dim=msa_head_dim,
            backend=backend,
        )
        self.msa_transition = Transition(msa_size, expansion=expansion, backend=backend)
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            divide_before_projection=True,
            backend=backend,
        )
        self.triangle_pair_update = BoltzGenMiniTrianglePairUpdateBlock(
            pair_size=pair_size,
            expansion=expansion,
            dropout=pair_dropout,
            backend=backend,
        )
        self.msa_dropout = RowwiseDropout(msa_dropout)

    def forward(
        self,
        pair: Tensor,
        msa_representation: Tensor,
        pair_mask: Tensor,
        msa_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        msa_representation = msa_representation + self.msa_dropout(
            self.pair_weighted_averaging(msa_representation, pair, pair_mask=pair_mask)
        )
        msa_representation = msa_representation + self.msa_transition(msa_representation)
        pair = pair + self.outer_product_mean(msa_representation, msa_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return pair, msa_representation


class BoltzGenMiniformerStack(nn.Module):
    r"""
    Stack of BoltzGen Miniformer blocks (the trunk variant selected by ``use_miniformer``).
    """

    def __init__(
        self,
        num_blocks: int = 12,
        single_size: int = 384,
        pair_size: int = 128,
        num_pair_attention_heads: int = 16,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            [
                BoltzGenMiniformerBlock(
                    single_size=single_size,
                    pair_size=pair_size,
                    num_pair_attention_heads=num_pair_attention_heads,
                    dropout=dropout,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            single, pair = block(single, pair, pair_mask=pair_mask)
        return single, pair


class BoltzGenMiniformerBlock(nn.Module):
    r"""
    BoltzGen Miniformer block: compact pair update plus pair-biased single attention.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication = BoltzGenBidirectionalTriangleMultiplication(pair_size)
        self.rowwise_dropout = RowwiseDropout(dropout)
        self.pair_transition = Transition(pair_size, expansion=expansion, backend=backend)
        self.attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        pair = pair + self.rowwise_dropout(self.triangle_multiplication(pair, pair_mask=pair_mask))
        pair = pair + self.pair_transition(pair)
        single = single + self.attention(single, pair, pair_mask=pair_mask)
        return single + self.single_transition(single), pair


class BoltzGenMiniTrianglePairUpdateBlock(nn.Module):
    r"""
    Pair-only Miniformer block used in structure-trained-small MSA / template stacks.
    """

    def __init__(
        self,
        pair_size: int = 128,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication = BoltzGenBidirectionalTriangleMultiplication(pair_size)
        self.rowwise_dropout = RowwiseDropout(dropout)
        self.pair_transition = Transition(pair_size, expansion=expansion, backend=backend)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        pair = pair + self.rowwise_dropout(self.triangle_multiplication(pair, pair_mask=pair_mask))
        return pair + self.pair_transition(pair)


class BoltzGenBidirectionalTriangleMultiplication(nn.Module):
    r"""
    Compact bidirectional triangular update used by BoltzGen Miniformer artifacts.
    """

    def __init__(self, pair_size: int = 128) -> None:
        super().__init__()
        self.input_norm = LayerNorm(pair_size, eps=1e-5)
        self.input_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.input_gate_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.output_norm = LayerNorm(pair_size // 2)
        self.output_projection = nn.Linear(pair_size // 2, pair_size, bias=False)
        self.output_gate_projection = nn.Linear(pair_size // 2, pair_size, bias=False)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        normalized = self.input_norm(pair)
        update = self.input_projection(normalized) * torch.sigmoid(self.input_gate_projection(normalized))
        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1).to(dtype=update.dtype, device=update.device)
        with torch.autocast("cuda", enabled=False):
            left_out, right_out, left_in, right_in = torch.chunk(update.float(), 4, dim=-1)
            outgoing = torch.einsum("...ikd,...jkd->...ijd", left_out, right_out)
            incoming = torch.einsum("...kid,...kjd->...ijd", left_in, right_in)
            update = torch.cat((outgoing, incoming), dim=-1).to(pair.dtype)
        update = self.output_norm(update)
        return self.output_projection(update) * torch.sigmoid(self.output_gate_projection(update))


class BoltzGenDenoiserConditioning(nn.Module):
    r"""
    Precompute the diffusion conditioning shared across denoiser calls.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 16,
        num_conditioning_transitions: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.atom_encoder = BoltzGenDenoiserAtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_encoder_pair_norms = nn.ModuleList(
            LayerNorm(atom_pair_hidden_size, backend=backend) for _ in range(num_atom_encoder_blocks)
        )
        self.atom_encoder_pair_bias_projections = nn.ModuleList(
            nn.Linear(atom_pair_hidden_size, num_atom_encoder_attention_heads, bias=False)
            for _ in range(num_atom_encoder_blocks)
        )
        self.atom_decoder_pair_norms = nn.ModuleList(
            LayerNorm(atom_pair_hidden_size, backend=backend) for _ in range(num_atom_decoder_blocks)
        )
        self.atom_decoder_pair_bias_projections = nn.ModuleList(
            nn.Linear(atom_pair_hidden_size, num_atom_decoder_attention_heads, bias=False)
            for _ in range(num_atom_decoder_blocks)
        )
        self.token_encoder_pair_norms = nn.ModuleList(
            LayerNorm(pair_size, backend=backend) for _ in range(num_token_encoder_blocks)
        )
        self.token_encoder_pair_bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_token_encoder_attention_heads, bias=False) for _ in range(num_token_encoder_blocks)
        )

    def forward(
        self,
        single_trunk: Tensor,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
    ) -> dict[str, Tensor]:
        pair = self.pairwise_conditioner(pair_trunk, relative_position_encoding)
        atom_query, atom_conditioning, atom_pair = self.atom_encoder(
            atom_features,
            reference_positions,
            atom_mask,
            reference_space_id,
            atom_to_token=atom_to_token,
            single_trunk=single_trunk,
            pair_trunk=pair,
        )
        atom_enc_bias = torch.cat(
            [
                projection(norm(atom_pair))
                for norm, projection in zip(
                    self.atom_encoder_pair_norms,
                    self.atom_encoder_pair_bias_projections,
                )
            ],
            dim=-1,
        )
        atom_dec_bias = torch.cat(
            [
                projection(norm(atom_pair))
                for norm, projection in zip(
                    self.atom_decoder_pair_norms,
                    self.atom_decoder_pair_bias_projections,
                )
            ],
            dim=-1,
        )
        token_encoder_bias = torch.cat(
            [
                projection(norm(pair))
                for norm, projection in zip(
                    self.token_encoder_pair_norms,
                    self.token_encoder_pair_bias_projections,
                )
            ],
            dim=-1,
        )
        return {
            "atom_query": atom_query,
            "atom_conditioning": atom_conditioning,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_encoder_bias": token_encoder_bias,
        }


class BoltzGenDenoiser(nn.Module):
    r"""
    Predict per-atom position updates for EDM sampling.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        fourier_embedding_size: int = 256,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 16,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_conditioning_transitions: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_conditioner = BoltzGenSingleConditioning(
            single_size=single_size,
            fourier_embedding_size=fourier_embedding_size,
            num_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.atom_attention_encoder = PackedDenoisingAtomToTokenEncoder(
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )
        self.single_to_token_norm = LayerNorm(hidden, backend=backend)
        self.single_to_token_projection = nn.Linear(hidden, hidden, bias=False)
        self.token_encoder = ConditionedTransformerEncoder(
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            hidden_size=hidden,
            conditioning_size=hidden,
            backend=backend,
        )
        self.token_norm = LayerNorm(hidden, backend=backend)
        self.atom_attention_decoder = PackedTokenToAtomDecoder(
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            backend=backend,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single_trunk: Tensor,
        noisy_positions: Tensor,
        noise_level: Tensor,
        conditioning: dict[str, Tensor],
        *,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        single, _ = self.single_conditioner(
            noise_level,
            single_trunk.repeat_interleave(multiplicity, 0),
            single_inputs.repeat_interleave(multiplicity, 0),
        )

        token_repr, atom_skip, conditioning_skip = self.atom_attention_encoder(
            conditioning["atom_query"],
            conditioning["atom_conditioning"],
            conditioning["atom_enc_bias"].repeat_interleave(multiplicity, 0),
            atom_mask,
            atom_to_token,
            token_mask.shape[-1],
            noisy_positions=noisy_positions,
            multiplicity=multiplicity,
        )
        token_repr = token_repr + self.single_to_token_projection(self.single_to_token_norm(single))

        token_repr = self.token_encoder(
            token_repr,
            single,
            conditioning["token_encoder_bias"].repeat_interleave(multiplicity, 0),
            token_mask.repeat_interleave(multiplicity, 0).float(),
        )
        token_repr = self.token_norm(token_repr)

        return self.atom_attention_decoder(
            token_repr,
            atom_skip,
            conditioning_skip,
            conditioning["atom_dec_bias"].repeat_interleave(multiplicity, 0),
            atom_mask,
            atom_to_token,
            multiplicity=multiplicity,
        )


class BoltzGenSingleConditioning(nn.Module):
    r"""
    Condition the concatenated trunk and input single representation on the diffusion noise level.
    """

    def __init__(
        self,
        single_size: int = 384,
        fourier_embedding_size: int = 256,
        num_transitions: int = 2,
        transition_expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_norm = LayerNorm(hidden, backend=backend)
        self.single_projection = nn.Linear(hidden, hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size, backend=backend)
        self.fourier_projection = nn.Linear(fourier_embedding_size, hidden, bias=False)
        self.transitions = nn.ModuleList(
            Transition(hidden, expansion=transition_expansion, backend=backend) for _ in range(num_transitions)
        )

    def forward(self, noise_level: Tensor, single_trunk: Tensor, single_inputs: Tensor) -> tuple[Tensor, Tensor]:
        return condition_single(
            single_trunk,
            single_inputs,
            noise_level,
            norm=self.single_norm,
            projection=self.single_projection,
            noise_fourier=self.noise_fourier,
            noise_norm=self.fourier_norm,
            noise_projection=self.fourier_projection,
            transitions=self.transitions,
            sample_axis=False,
        )


class BoltzGenCoordinateDiffusion(nn.Module):
    r"""
    EDM sampler wrapping a [`BoltzGenDenoiser`][ofoldx.models.boltzgen.model.BoltzGenDenoiser].
    """

    def __init__(
        self,
        denoiser: BoltzGenDenoiser,
        sigma_data: float = 16.0,
        min_noise_level: float = 4e-4,
        max_noise_level: float = 160.0,
        rho: float = 7.0,
        churn_rate: float = 0.8,
        min_churn_noise_level: float = 1.0,
        noise_scale: float = 0.95,
        step_scale: float = 1.8,
        num_sampling_steps: int = 200,
        alignment_reverse_diff: bool = True,
        sampling_schedule: str = "dilated",
        time_dilation: float = 2.667,
        time_dilation_start: float = 0.6,
        time_dilation_end: float = 0.8,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sigma_data = sigma_data
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho = rho
        self.churn_rate = churn_rate
        self.min_churn_noise_level = min_churn_noise_level
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std
        self.num_sampling_steps = num_sampling_steps
        self.alignment_reverse_diff = alignment_reverse_diff
        self.sampling_schedule = sampling_schedule
        self.time_dilation = time_dilation
        self.time_dilation_start = time_dilation_start
        self.time_dilation_end = time_dilation_end

    def preconditioned_network_forward(
        self, noisy_coordinates: Tensor, sigma: Tensor | float, *, multiplicity: int = 1, **conditioning: object
    ) -> Tensor:
        batch = noisy_coordinates.shape[0]
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device, dtype=noisy_coordinates.dtype).expand(batch)
        padded = sigma.reshape(batch, 1, 1)
        update = self.denoiser(
            noisy_positions=self.preconditioner.c_in(padded) * noisy_coordinates,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
            **conditioning,
        )
        return self.preconditioner.c_skip(padded) * noisy_coordinates + self.preconditioner.c_out(padded) * update

    def noise_distribution(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        noise = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        return self.sigma_data * (self.noise_log_mean + self.noise_log_std * noise).exp()

    def loss_weight(self, sigma: Tensor) -> Tensor:
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def sample_schedule_karras(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        inv_rho = 1 / self.rho
        ramp = torch.arange(steps, device=device, dtype=torch.float32) / max(steps - 1, 1)
        sigmas = (
            self.max_noise_level**inv_rho + ramp * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def sample_schedule_dilated(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        inv_rho = 1 / self.rho
        ramp = torch.arange(steps, device=device, dtype=torch.float32) / max(steps - 1, 1)
        dilated = _dilate_time(
            ramp,
            start=self.time_dilation_start,
            end=self.time_dilation_end,
            dilation=self.time_dilation,
        )
        sigmas = (
            self.max_noise_level**inv_rho + dilated * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def sample_schedule(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        if self.sampling_schedule == "af3":
            return self.sample_schedule_karras(num_sampling_steps, device=device)
        if self.sampling_schedule == "dilated":
            return self.sample_schedule_dilated(num_sampling_steps, device=device)
        raise ValueError(f"unknown BoltzGen sampling_schedule: {self.sampling_schedule!r}")

    @torch.no_grad()
    def sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        **conditioning: object,
    ) -> Tensor:
        device = atom_mask.device
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        step_scale_value = self.step_scale if step_scale is None else step_scale
        noise_scale_value = self.noise_scale if noise_scale is None else noise_scale

        def randn() -> Tensor:
            r"""
            Draw standard Gaussian noise of the expanded coordinate shape ``(B*multiplicity, N_atom, 3)``.
            """
            return torch.randn(shape, device=device, generator=generator)

        coords = sigmas[0] * randn()
        denoised = None
        # The loop owns the squared-noise-scale churn factor, optional mid-step alignment, dilated Karras schedule,
        # rigid augmentation, and random draws as one sampling algorithm.
        # The fixed Python loop controls denoiser invocation; schedule values stay as tensors so sampling does not
        # synchronize them to the host on every step.
        for index in range(len(sigmas) - 1):
            sigma_current, sigma_next, gamma = sigmas[index], sigmas[index + 1], gammas[index + 1]
            rotation = uniform_random_rotation(
                expanded_atom_mask.shape[0], device=device, dtype=coords.dtype, generator=generator
            )
            translation = torch.randn(expanded_atom_mask.shape[0], 1, 3, device=device, generator=generator)
            coords = torch.einsum("bmd,bds->bms", coords - coords.mean(-2, keepdim=True), rotation) + translation
            if denoised is not None:
                denoised = (
                    torch.einsum("bmd,bds->bms", denoised - denoised.mean(-2, keepdim=True), rotation) + translation
                )

            sigma_hat = sigma_current * (1 + gamma)
            noise_var = noise_scale_value**2 * (sigma_hat**2 - sigma_current**2)
            noisy = coords + noise_scale_value * torch.sqrt(torch.clamp(noise_var, min=0.0)) * randn()
            denoised = self.preconditioned_network_forward(
                noisy, sigma_hat, multiplicity=multiplicity, atom_mask=base_atom_mask, **conditioning
            )
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy.float(), target=denoised.float(), weight=expanded_atom_mask.float()
                ).to(denoised)
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale_value * (sigma_next - sigma_hat) * derivative
        return coords


class BoltzGenBroadcastGaussianSmearing(nn.Module):
    r"""
    Radial-basis distance features ``exp(coeff (d - offset)^2)`` broadcasting over a trailing singleton.
    """

    def __init__(self, start: float = 0.0, stop: float = 2.0, num_gaussians: int = 32) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Derive the RBF width from the materialized float32 center spacing. This preserves the distance-feature
        # arithmetic and avoids calling ``.item()`` on an unmaterialized meta tensor.
        spacing = torch.linspace(start, stop, num_gaussians, device="cpu")
        self.coeff = -0.5 / (spacing[1] - spacing[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, distance: Tensor) -> Tensor:
        return torch.exp(self.coeff * (distance - self.offset.view(1, 1, 1, -1)).pow(2))


class BoltzGenGaussianSmearing(nn.Module):
    r"""
    Radial-basis distance featurization with a registered ``offset`` buffer.
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 16) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Derive the RBF width from the float32 ``offset`` spacing (matching the BoltzGen reference) rather than
        # the float64 analytic spacing: for non-float32-representable spacings (e.g. 20/15) the two differ by
        # ~6e-8 relative, and that discrepancy amplifies through the 6-layer encoder to ~1e-3 in the logits.
        # Compute the spacing on a real CPU tensor so construction under ``torch.device("meta")`` (the
        # converter / hub load mode) does not call ``.item()`` on an unmaterialized meta tensor.
        spacing = torch.linspace(start, stop, num_gaussians, device="cpu")
        self.coeff = -0.5 / (spacing[1] - spacing[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, distances: Tensor) -> Tensor:
        delta = (distances.unsqueeze(-1) - self.offset.view(1, 1, -1)).flatten(start_dim=1)
        return torch.exp(self.coeff * delta.pow(2))


# Scoring heads are optional so the design-only generator keeps its base parameter count.


def _dilate_time(ts: Tensor, *, start: float, end: float, dilation: float) -> Tensor:
    r"""
    Piecewise remapping used by BoltzGen's dilated design-inference noise schedule.
    """

    x = end - start
    lower = start
    upper = 1 - end
    if (dilation - 1) * x > lower + upper:
        raise ValueError("time dilation is too large for the requested interval")

    inv_dilation = 1 / dilation
    ratio = (lower + upper + (1 - dilation) * x) / (lower + upper)
    inv_ratio = 1 / ratio
    lower_prime = lower * ratio
    x_prime = x * dilation

    lower_third = ts * inv_ratio
    middle_third = (ts - lower_prime) * inv_dilation + lower
    upper_third = (ts - (lower_prime + x_prime)) * inv_ratio + lower + x
    return torch.where(
        ts < lower_prime,
        lower_third,
        torch.where(ts < lower_prime + x_prime, middle_third, upper_third),
    )


def _center_random_augmentation(
    coords: Tensor,
    atom_mask: Tensor,
    *,
    translation_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Center masked coordinates and apply a random rotation and translation.
    """

    mask = atom_mask.unsqueeze(-1).to(coords.dtype)
    centroid = (coords * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1.0)
    centered = coords - centroid
    rotation = uniform_random_rotation(coords.shape[0], device=coords.device, dtype=coords.dtype, generator=generator)
    rotated = torch.einsum("bmd,bds->bms", centered, rotation)
    translation = (
        torch.randn(coords.shape[0], 1, 3, device=coords.device, dtype=coords.dtype, generator=generator)
        * translation_scale
    )
    return rotated + translation


def _gather_backbone4(
    token_to_bb4_atoms: Tensor,
    coords: Tensor,
    batch_size: int,
    num_tokens: int,
) -> Tensor:
    r"""
    Gather the four backbone-atom coordinates per token -> ``(B, N_token, 4, 3)``.
    """
    bb4_index = token_to_bb4_atoms.reshape(batch_size, num_tokens * 4)
    coords = coords.reshape(batch_size, -1, 3)
    backbone = coords.gather(1, bb4_index.unsqueeze(-1).expand(-1, -1, 3))
    return backbone.reshape(batch_size, num_tokens, 4, 3)


def _zero_initialize_boltzgen_updates(module: nn.Module) -> None:
    r"""
    Apply the BoltzGen zero-init scheme to every final-layer / gating / conditioning weight under ``module``.

    Zeroing each residual branch's final projection (and biasing gate projections negative) starts every
    block as a near-identity.

    Args:
        module: Root module to walk; every matching child is zero-initialized in place.
    """

    for child in module.modules():
        if isinstance(child, (PackedReferenceAtomEncoder, BoltzGenDenoiserAtomEncoder)):
            if isinstance(child, BoltzGenDenoiserAtomEncoder):
                nn.init.zeros_(child.single_trunk_projection.weight)
                nn.init.zeros_(child.pair_trunk_projection.weight)
            nn.init.zeros_(child.left_atom_projection.weight)
            nn.init.zeros_(child.right_atom_projection.weight)
            nn.init.zeros_(child.feed_forward_output_projection.weight)
        elif isinstance(child, PackedDenoisingAtomToTokenEncoder):
            nn.init.zeros_(child.noisy_position_projection.weight)
        elif isinstance(child, PackedTokenToAtomDecoder):
            nn.init.zeros_(child.token_projection.weight)
            nn.init.zeros_(child.position_update_projection.weight)
        elif isinstance(child, BoltzGenDenoiser):
            nn.init.zeros_(child.single_to_token_projection.weight)
        elif isinstance(child, BoltzGenBidirectionalTriangleMultiplication):
            nn.init.zeros_(child.input_gate_projection.weight)
            nn.init.zeros_(child.output_projection.weight)
            nn.init.zeros_(child.output_gate_projection.weight)
        elif isinstance(child, Transition):
            nn.init.zeros_(child.feed_forward.output_projection.weight)
        elif isinstance(child, GatedMultiheadAttention):
            nn.init.zeros_(child.projection.output_projection.weight)
        elif isinstance(child, ConditionedTransformerEncoderBlock):
            nn.init.zeros_(child.output_gate_projection.weight)
            if child.output_gate_projection.bias is not None:
                nn.init.constant_(child.output_gate_projection.bias, -2.0)
        elif isinstance(child, (ConditionedTransition, TripleProductConditionedTransition)):
            nn.init.zeros_(child.conditioning_gate_projection.weight)
            if child.conditioning_gate_projection.bias is not None:
                nn.init.constant_(child.conditioning_gate_projection.bias, -2.0)


def _softmax_dropout(logits: Tensor, index: Tensor, num_groups: int, probability: float) -> Tensor:
    r"""
    Apply the rejection-sampled attention-logit dropout.
    """

    for _ in range(10):
        dropped = torch.rand_like(logits) < probability
        kept = _scatter_sum((~dropped).to(torch.int64), index, dim_size=num_groups)
        if kept.all():
            return logits.masked_fill(dropped, -torch.inf)
    raise RuntimeError("attention dropout failed to keep an incoming edge for every destination and head")


def _scatter_sum(src: Tensor, index: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    r"""
    Sum ``src`` rows into buckets given by ``index``.

    Args:
        src: Source tensor whose slices along ``dim`` are summed per bucket.
        index: Destination bucket index for each slice along ``dim``.
        dim: Dimension along which to scatter.
        dim_size: Number of output buckets.

    Returns:
        Tensor with ``dim_size`` buckets along ``dim``, each the sum of its assigned ``src`` slices.
    """

    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    expanded = index
    for _ in range(src.dim() - index.dim()):
        expanded = expanded.unsqueeze(-1)
    out.scatter_add_(dim, expanded.expand_as(src), src)
    return out


def _scatter_softmax(src: Tensor, index: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    r"""
    Softmax over rows sharing a destination ``index`` (per-group / segmented softmax).

    Args:
        src: Source tensor whose slices along ``dim`` are softmaxed within each group.
        index: Group index for each slice along ``dim``.
        dim: Dimension along which to group.
        dim_size: Number of groups.

    Returns:
        Tensor the same shape as ``src`` whose slices sum to one within each group.
    """

    group_max = src.new_full((dim_size, *src.shape[1:]), float("-inf"))
    expanded = index
    for _ in range(src.dim() - index.dim()):
        expanded = expanded.unsqueeze(-1)
    group_max.scatter_reduce_(0, expanded.expand_as(src), src, reduce="amax", include_self=True)
    centered = src - group_max.gather(0, expanded.expand_as(src))
    exp = centered.exp()
    denom = _scatter_sum(exp, index, dim=dim, dim_size=dim_size).gather(0, expanded.expand_as(src))
    return exp / (denom + 1e-16)


def _build_constraint_logit_mask(
    num_nodes: int,
    aa_constraint_mask: Tensor | None,
    inverse_fold_restriction: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    r"""
    Build the additive per-position canonical-amino-acid logit mask for constrained inverse folding.

    Combines a per-residue block mask (``aa_constraint_mask``, ``(num_nodes, 20)``, ``> 0`` = blocked) with a
    global block list (``inverse_fold_restriction``, three-letter CCD codes blocked everywhere). A position with no
    allowed amino acid raises instead of silently relaxing the explicit per-residue constraints.

    Args:
        num_nodes: Number of packed valid tokens.
        aa_constraint_mask: Optional per-residue block mask ``(num_nodes, 20)``; ``None`` = no per-residue blocks.
        inverse_fold_restriction: Canonical amino acids (three-letter codes) to block at every position.
        device: Device for the returned mask.
        dtype: Floating dtype for the returned additive mask.

    Returns:
        Additive logit bias ``(num_nodes, 20)``: ``0.0`` where allowed, ``-1e6`` where blocked.

    Raises:
        ValueError: If the combined restrictions block every identity at a position.
    """

    num_aa = NUM_CANONICAL_TOKENS
    if aa_constraint_mask is None:
        per_residue_blocked = torch.zeros(num_nodes, num_aa, dtype=torch.bool, device=device)
    else:
        per_residue_blocked = aa_constraint_mask > 0

    global_blocked = torch.zeros(num_aa, dtype=torch.bool, device=device)
    for residue_type in inverse_fold_restriction:
        global_blocked[CANONICAL_TOKENS.index(residue_type)] = True

    combined = per_residue_blocked | global_blocked.unsqueeze(0)
    all_blocked = combined.all(dim=1)
    if all_blocked.any():
        raise ValueError(
            f"inverse folding has no valid amino acids at positions {torch.where(all_blocked)[0].tolist()}"
        )
    return combined.to(dtype=dtype) * -1e6


# ---------------------------------------------------------------------------
# Canonical identity layouts.
#
# These are the non-tensor leaves the design/inverse-fold Processors populate and
# the design Pipeline reads when it decodes tensor rows back into canonical molecular
# identities; the model graph above deliberately ignores them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoltzGenLayout:
    r"""
    Canonical molecular identities aligned to one ``BoltzGenModel`` tensor layout.
    """

    system: MolecularSystem | None
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    output_residues: tuple[ComponentReference | SymbolicResidueSite, ...]
    output_instance_ids: tuple[str, ...]
    placement_map: PlacementMap = field(default_factory=dict)
    instance_of: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoltzGenInverseFoldingLayout:
    r"""
    Canonical molecular identities aligned to one ``BoltzGenInverseFoldingModel`` tensor layout.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    output_residues: tuple[ComponentReference, ...]
    output_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class BoltzGenPlacements:
    r"""
    Declare the generated and retained segments of one BoltzGen assembly.

    BoltzGen expects one declaration group whose placements form the output chains.
    Ranged generated segments are resolved against the ``total_len`` supplied to the
    processor or pipeline.

    Args:
        placements: Ordered placement declaration. Placement IDs must be globally
            unique.
    """

    placements: PlacementSpec

    def __post_init__(self) -> None:
        # Every placement_id must be globally unique across the whole PlacementSpec.
        validate_placement(self.placements)


@dataclass(frozen=True)
class BoltzGenDesignMask:
    r"""
    Select existing residue positions for redesign.

    Args:
        positions: Residue occurrences whose identities may change.
    """

    positions: tuple[ComponentReference, ...]


@dataclass(frozen=True)
class BoltzGenBinding:
    r"""
    Positive and negative residue-level binding conditions.

    Args:
        binding: Residues encouraged to participate in binding.
        not_binding: Residues discouraged from participating in binding.
    """

    binding: Hotspot = Hotspot(())
    not_binding: tuple[ComponentReference, ...] = ()


@dataclass(frozen=True)
class BoltzGenSecondaryStructure:
    r"""
    Assign secondary-structure conditions to residue positions.

    Args:
        assignments: ``(residue, label)`` pairs, where the label is ``"H"``
            (helix), ``"E"`` (strand), or ``"L"`` (loop).
    """

    assignments: tuple[tuple[ComponentReference, Literal["H", "E", "L"]], ...]


@dataclass(frozen=True)
class BoltzGenStructureGroup:
    r"""
    Residues whose relative geometry remains visible during design.

    Distinct nonzero groups move independently; group zero denotes unconstrained residues.

    Args:
        members: Residues belonging to the group.
        group_id: Nonnegative group identifier.
    """

    members: tuple[ComponentReference, ...]
    group_id: int


@dataclass(frozen=True)
class BoltzGenResidueConstraint:
    r"""
    Restrict the amino-acid identities allowed at one residue.

    Args:
        position: Residue occurrence to constrain.
        allowed: Allowed three-letter amino-acid component IDs.
    """

    position: ComponentReference
    allowed: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class BoltzGenDirectionalContact:
    r"""
    Add BoltzGen's directional pocket/binder meaning to a shared contact.

    The shared [`Contact`][] owns the endpoints, cutoff, and symmetric proximity
    meaning. This wrapper owns only which ordered side is the pocket.

    Args:
        contact: Shared two-sided proximity condition.
        direction: Direction of the pocket-to-binder relation across the contact's
            declared ``side_a`` and ``side_b``.
    """

    contact: Contact
    direction: Literal["pocket>binder", "binder>pocket"]

    def _identity(self) -> tuple[object, ...]:
        contact = self.contact
        return (
            contact.side_a,
            contact.side_b,
            contact.cutoff_angstrom,
            contact.tolerance_angstrom,
            self.direction,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoltzGenDirectionalContact):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


@dataclass(frozen=True)
class BoltzGenCovalentBond:
    r"""
    A requested covalent link → derived ``cyclic_period`` (envelope over ``SymbolicBond``).
    """

    bond: SymbolicBond
    intent: Literal["hard", "soft"] = "hard"


@dataclass(frozen=True)
class BoltzGenSymmetry:
    r"""
    Tie corresponding positions across equal-length symmetry units.

    Args:
        units: Residue groups representing symmetric copies. Corresponding offsets
            are tied.
    """

    units: tuple[tuple[ComponentReference, ...], ...]


#: Condition record accepted by BoltzGen structure-design processors and pipelines.
BoltzGenSchema = (
    BoltzGenDesignMask
    | BoltzGenBinding
    | BoltzGenSecondaryStructure
    | BoltzGenStructureGroup
    | Contact
    | BoltzGenDirectionalContact
    | BoltzGenCovalentBond
    | BoltzGenSymmetry
)


@dataclass(kw_only=True, eq=False)
class BoltzGenInput:
    r"""
    Input tensors consumed by BoltzGen design models.

    Args:
        token_mask: Boolean token validity mask with shape ``(B, N_token)``.
        residue_index: Chain-local residue index used by relative-position features.
        chain_index: Chain ids mapped to contiguous integer indices.
        entity_index: Entity ids shared by chains with the same sequence.
        symmetry_index: Symmetry-copy ids for relative-position features.
        token_index: Token ids in structure order.
        atom_features: Per-atom reference features.
        atom_to_token: Token index for each atom.
        atom_mask: Boolean atom validity mask.
        reference_positions: Centered reference-conformer atom coords.
        reference_space_id: Reference-conformer group id for each atom.
        representative_atom_mask: Mask selecting the distogram representative atom per token.
        atom_token_index: Per-token atom offset for each atom.
        token_to_rep_atom: Distogram representative atom index for each token.
        residue_type_features: Query residue type one-hot features.
        msa_profile: Single-sequence MSA profile features.
        deletion_mean: Per-token deletion mean features.
        token_bonds: Token-level covalent adjacency ``(B, N, N, 1)`` (trailing singleton per the featurizer).
        type_bonds: Token-level bond type ids.
        contact_conditioning: Pairwise contact-conditioning class one-hot features.
        contact_threshold: Pairwise contact threshold values.
        molecule_type: Molecular category ids in the model vocabulary.
        center_coordinates: Per-token representative atom coords.
        token_to_bb4_atoms: Per-token ``(N, CA, C, O)`` atom indices.
        atom_coordinates: Atom coordinates used by geometric conditioning features; never a training-label fallback.
        token_distance_mask: Pairwise token-distance conditioning mask.
        target_positions: Experimental atom coords consumed exclusively by the training forward. It is optional
            on the shared input because ``design`` does not consume a training target.
        target_mask: Resolved experimental-coordinate mask consumed by the training objective. It is distinct
            from ``atom_mask`` and does not define the centroid of the rigid augmentation.
        frame_atom_index: Optional frame atom indices for frame-aligned losses.
        frame_mask: Optional frame validity mask.
        method_feature: Optional experimental-method ids.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        msa_tokens: Optional integer MSA token ids.
        has_deletion: Optional MSA deletion flags.
        deletion_value: Optional MSA deletion values.
        msa_paired: Optional paired-MSA flags.
        msa_mask: Optional MSA validity mask.
        template_residue_type_features: Optional template residue-type features.
        template_frame_rotations: Optional template frame rotations.
        template_frame_translations: Optional template frame translations.
        template_frame_mask: Optional template frame validity mask.
        template_beta_carbon_positions: Optional template beta-carbon positions.
        template_alpha_carbon_positions: Optional template alpha-carbon positions.
        template_beta_carbon_mask: Optional template beta-carbon validity mask.
        template_mask: Optional template validity mask.
        template_visibility_index: Optional template visibility-group indices.
        binding_type: Optional binding conditioning ids.
        design_mask: Optional token mask selecting designable positions.
        ss_type: Optional secondary-structure conditioning ids.
        layout: Optional molecular identities aligned to token and atom rows. The
            model graph does not consume it; result decoding requires it.
    """

    token_mask: Tensor
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
    representative_atom_mask: Tensor
    atom_token_index: Tensor
    token_to_rep_atom: Tensor
    residue_type_features: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    contact_conditioning: Tensor
    contact_threshold: Tensor
    molecule_type: Tensor
    center_coordinates: Tensor
    token_to_bb4_atoms: Tensor
    atom_coordinates: Tensor
    token_distance_mask: Tensor
    target_positions: Tensor | None = None
    target_mask: Tensor | None = None
    frame_atom_index: Tensor | None = None
    frame_mask: Tensor | None = None
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    template_residue_type_features: Tensor | None = None
    template_frame_rotations: Tensor | None = None
    template_frame_translations: Tensor | None = None
    template_frame_mask: Tensor | None = None
    template_beta_carbon_positions: Tensor | None = None
    template_alpha_carbon_positions: Tensor | None = None
    template_beta_carbon_mask: Tensor | None = None
    template_mask: Tensor | None = None
    template_visibility_index: Tensor | None = None
    binding_type: Tensor | None = None
    design_mask: Tensor | None = None
    ss_type: Tensor | None = None
    # Molecular identities aligned to tensor rows for result decoding.
    layout: BoltzGenLayout | None = None


@dataclass(kw_only=True, eq=False)
class BoltzGenInverseFoldingInput:
    r"""
    Tensor and molecular-layout input for BoltzGen inverse folding.

    ``B`` is batch size, ``N`` token count, and ``A`` atom count.

    Args:
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        atom_features: Reference atom features with shape ``(B, A, C_atom)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        atom_mask: Valid-atom mask with shape ``(B, A)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        residue_type_features: Residue-type features with shape ``(B, N, C_residue)``.
        msa_profile: Per-token MSA profile with shape ``(B, N, C_msa)``.
        deletion_mean: Per-token mean deletion values with shape ``(B, N)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N, 1)``.
        type_bonds: Token-pair bond type IDs with shape ``(B, N, N)``.
        molecule_type: Molecular category IDs with shape ``(B, N)``.
        center_coordinates: Representative token coordinates with shape ``(B, N, 3)``.
        token_to_bb4_atoms: Backbone atom indices with shape ``(B, N, 4)``.
        atom_coordinates: Atom coordinates with shape ``(B, A, 3)`` in angstroms.
        method_feature: Optional experimental-method features.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        design_mask: Optional mask selecting positions to redesign.
        aa_constraint_mask: Optional amino-acid allowance mask.
        symmetric_group: Optional sequence-symmetry group IDs.
        layout: Molecular identities aligned to the token and atom axes.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    residue_type_features: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    molecule_type: Tensor
    center_coordinates: Tensor
    token_to_bb4_atoms: Tensor
    atom_coordinates: Tensor
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None
    design_mask: Tensor | None = None
    aa_constraint_mask: Tensor | None = None
    symmetric_group: Tensor | None = None
    # Molecular identities aligned to inverse-folding rows for result decoding.
    layout: BoltzGenInverseFoldingLayout | None = None


@dataclass(eq=False)
class _BoltzGenTrunkOutput:
    r"""
    Trunk representations consumed by the diffusion and confidence heads.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    relative_position_encoding: Tensor
    distogram_logits: Tensor


@dataclass(eq=False)
class BoltzGenOutput:
    r"""
    Predictions from one BoltzGen training denoising pass.

    Args:
        coordinates: Denoised atom coordinates with shape ``(B, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        bfactor_logits: Per-token B-factor logits with shape ``(B, N, D_b)``.
        augmented_target_coordinates: Training target coordinates after
            rigid augmentation, with shape ``(B, A, 3)`` in angstroms.
        noise_level: Sampled diffusion noise levels with shape ``(B,)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor


@dataclass(eq=False)
class BoltzGenDesignOutput:
    r"""
    Predictions from one BoltzGen structure-design pass.

    Args:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        bfactor_logits: Per-token B-factor logits with shape ``(B, S, N, D_b)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor


@dataclass(eq=False)
class BoltzGenInverseFoldingOutput:
    r"""
    Logit predictions from one BoltzGen inverse-folding forward pass.

    Args:
        logits: Per-token amino-acid logits with shape ``(B, N, V)``.
    """

    logits: Tensor


@dataclass(eq=False)
class BoltzGenInverseFoldingDesignOutput:
    r"""
    Predictions from one BoltzGen inverse-folding design pass.

    Args:
        logits: Per-token amino-acid logits with shape ``(B, N, V)``.
        tokens: Sampled amino-acid token IDs with shape ``(B, N)``.
    """

    logits: Tensor
    tokens: Tensor
