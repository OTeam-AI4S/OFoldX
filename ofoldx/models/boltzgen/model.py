r"""
BoltzGen structure-design, confidence, and inverse-folding models with typed tensor contracts.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, cast

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.bonds import SymbolicBond
from ...data.conditions import Endpoint, Hotspot
from ...data.evidence import MSA_VOCAB_SIZE
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.tokenization import TOKEN_VOCAB_SIZE
from ...model import Model, _OptionalTensor, _TensorCollection
from ...modules.atom import (
    PackedAtomAttention,
    PackedAtomToTokenEncoder,
    PackedDenoisingAtomToTokenEncoder,
    PackedReferenceAtomEncoder,
    PackedTokenToAtomDecoder,
)
from ...modules.atom.pair_encoding import AtomPairEncoding
from ...modules.atom.transfer import dense_mean_to_tokens
from ...modules.atom.window import gather_key_windows
from ...modules.attention import (
    GatedMultiheadAttention,
    PairBiasedSelfAttention,
    PerBlockPairBiasProjection,
    TriangleAttention,
    TriangleMultiplication,
)
from ...modules.backend import LayerNorm
from ...modules.contact import ContactConditioning
from ...modules.diffusion.conditioning import PairwiseConditioning
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.dropout import StructuredDropout
from ...modules.embedding.factorized import FactorizedPairLinear
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.feed_forward import SwiGluFeedForward
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import PairSymmetrizedDistogramHead
from ...modules.head.tm import tm_weight_from_pae
from ...modules.transformer import (
    ConditionedTransformerEncoder,
    ConditionedTransformerEncoderBlock,
    TripleProductConditionedTransition,
)
from ...modules.transition import ConditionedTransition, Transition
from ...modules.trunk.msa import (
    MsaPairWeightedAveraging,
    OuterProductMean,
)
from ...modules.trunk.pairformer import PairformerStack
from .config import (
    BoltzGenConfidenceConfig,
    BoltzGenConfig,
    BoltzGenInverseFoldingConfig,
    BoltzGenMiniformerConfig,
    BoltzGenMiniformerResidueTypeConfig,
    BoltzGenResidueTypeConfig,
)

_NUM_BOND_TYPES = 6

# Molecular categories are protein, DNA, RNA, and non-polymer.
_NUM_MOLECULE_TYPES = 4

_NUM_RESIDUE_TYPES = TOKEN_VOCAB_SIZE

_CANONICALS_OFFSET = 0

NUM_CANONICAL_TOKENS = 20  # len(const.canonical_tokens)

# The 20 canonical amino acids (three-letter CCD codes) in the shared token-vocabulary order;
# this tuple also defines the ``aa_constraint_mask`` column order.
CANONICAL_TOKENS: tuple[str, ...] = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
)


class _BoltzGenTrainingOutput(NamedTuple):
    coordinates: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor
    residue_type_logits: Tensor | None


class _BoltzGenStructureModel(Model):
    r"""
    Own the common BoltzGen structure trunk, denoiser, sampler, and heads.

    Args:
        config: Model architecture and inference configuration.
    """

    config: BoltzGenConfig
    _miniformer = False
    _predict_residue_type = False

    def __init__(
        self,
        config: BoltzGenConfig | None = None,
    ) -> None:
        config = cast(BoltzGenConfig, self.config_class()) if config is None else config
        if type(config) is not self.config_class:
            raise TypeError(f"{type(self).__name__} requires {self.config_class.__name__}")
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
            num_method_types=input_embeddings_config.num_method_types,
            num_molecule_types=input_embeddings_config.num_molecule_types,
            add_method_conditioning=input_embeddings_config.add_method_conditioning,
            add_modified_flag=input_embeddings_config.add_modified_flag,
            add_cyclic_flag=input_embeddings_config.add_cyclic_flag,
            add_molecule_type_feature=input_embeddings_config.add_molecule_type_feature,
            add_binding_specification=input_embeddings_config.add_binding_specification,
            add_design_mask=input_embeddings_config.add_design_mask,
            add_ss_specification=input_embeddings_config.add_ss_specification,
            backend="auto",
        )

        self.single_init_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(config.num_bond_types, pair_size)
        self.contact_conditioning = ContactConditioning(
            pair_size, cutoff_min=config.conditioning_cutoff_min, cutoff_max=config.conditioning_cutoff_max
        )
        self.single_recycle_norm = LayerNorm(single_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        # Token-distance conditioning precedes template conditioning in the released constructor.
        self.token_distance_embedder: BoltzGenTokenDistanceEmbedder | BoltzGenMiniformerTokenDistanceEmbedder | None = (
            None
        )
        if design.use_token_distance:
            if self._miniformer:
                self.token_distance_embedder = BoltzGenMiniformerTokenDistanceEmbedder(
                    pair_size=pair_size,
                    token_distance_size=design.token_distance_size,
                    num_blocks=design.num_token_distance_blocks,
                    num_bins=config.template.num_distogram_bins,
                    distance_gaussian_dim=design.distance_gaussian_dim,
                    min_dist=config.template.min_distance,
                    max_dist=config.template.max_distance,
                    dropout=design.token_distance_dropout,
                    post_layer_norm=design.token_distance_post_layer_norm,
                    use_token_distance_features=design.use_token_distance_features,
                    backend="auto",
                )
            else:
                self.token_distance_embedder = BoltzGenTokenDistanceEmbedder(
                    pair_size=pair_size,
                    token_distance_size=design.token_distance_size,
                    num_blocks=design.num_token_distance_blocks,
                    num_bins=config.template.num_distogram_bins,
                    distance_gaussian_dim=design.distance_gaussian_dim,
                    min_dist=config.template.min_distance,
                    max_dist=config.template.max_distance,
                    triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
                    num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                    dropout=design.token_distance_dropout,
                    use_token_distance_features=design.use_token_distance_features,
                    backend="auto",
                )

        self.template_embedder: BoltzGenTemplateEmbedder | BoltzGenMiniformerTemplateEmbedder
        if self._miniformer:
            self.template_embedder = BoltzGenMiniformerTemplateEmbedder(
                pair_size=pair_size,
                template_size=design.template_size,
                num_blocks=design.num_template_blocks,
                vocab_size=input_embeddings_config.vocab_size,
                num_bins=config.template.num_distogram_bins,
                min_dist=config.template.min_distance,
                max_dist=config.template.max_distance,
                dropout=design.template_dropout,
                post_layer_norm=design.template_post_layer_norm,
                backend="auto",
            )
        else:
            self.template_embedder = BoltzGenTemplateEmbedder(
                pair_size=pair_size,
                template_size=design.template_size,
                num_blocks=design.num_template_blocks,
                vocab_size=input_embeddings_config.vocab_size,
                num_bins=config.template.num_distogram_bins,
                min_dist=config.template.min_distance,
                max_dist=config.template.max_distance,
                triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
                num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                dropout=design.template_dropout,
                backend="auto",
            )

        self.msa_trunk = (
            BoltzGenMsaTrunk(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                num_blocks=trunk.num_msa_blocks,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.msa_pair_dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                triangle_attention_hidden_size=trunk.msa_triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_msa_triangle_attention_heads,
                backend="auto",
            )
            if not self._miniformer
            else BoltzGenMiniformerMsaTrunk(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                num_blocks=trunk.num_msa_blocks,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.msa_pair_dropout,
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
                post_layer_norm=trunk.post_layer_norm,
                use_single_to_pair=trunk.use_single_to_pair,
                backend="auto",
            )
            if self._miniformer
            else BoltzGenPairformerStack(
                num_blocks=trunk.num_trunk_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.pair_dropout,
                post_layer_norm=trunk.post_layer_norm,
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
        denoiser_type = BoltzGenResidueTypeDenoiser if self._predict_residue_type else BoltzGenDenoiser
        denoiser = denoiser_type(
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
            transformer_size=denoiser_config.transformer_size,
            post_layer_norm=denoiser_config.post_layer_norm,
            use_qk_norm=denoiser_config.use_qk_norm,
            gaussian_random_3d_encoding_dim=denoiser_config.gaussian_random_3d_encoding_dim,
            num_residue_types=input_embeddings_config.vocab_size,
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

    @torch.no_grad()
    def _sample(
        self,
        input: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenSampleOutput | BoltzGenResidueTypeSampleOutput:
        r"""
        Draw reverse-diffusion structure samples from a preprocessed input.

        Args:
            input: Typed model input.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                configured value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional scalar or per-sample step-scale override.
            noise_scale: Optional scalar or per-sample noise-scale override.
            return_trajectory: Whether to retain the current-coordinate and
                denoised-coordinate histories.

        """

        trunk = self._run_trunk(input, recycling_steps=recycling_steps)
        conditioning = self._build_denoiser_conditioning(trunk, input)
        coordinates, residue_type_logits, coordinate_trajectory, denoised_coordinate_trajectory = (
            self.coordinate_diffusion.sample(
                input.atom_mask,
                num_sampling_steps=num_sampling_steps,
                multiplicity=num_samples,
                generator=generator,
                step_scale=step_scale,
                noise_scale=noise_scale,
                return_trajectory=return_trajectory,
                single_inputs=trunk.single_inputs,
                single_trunk=trunk.single,
                conditioning=conditioning,
                atom_to_token=input.atom_to_token,
                token_mask=input.token_mask,
            )
        )
        batch_size = input.token_mask.shape[0]
        if coordinate_trajectory is not None:
            coordinate_trajectory = coordinate_trajectory.reshape(
                batch_size,
                num_samples,
                *coordinate_trajectory.shape[1:],
            )
        if denoised_coordinate_trajectory is not None:
            denoised_coordinate_trajectory = denoised_coordinate_trajectory.reshape(
                batch_size,
                num_samples,
                *denoised_coordinate_trajectory.shape[1:],
            )
        values = {
            "coordinates": coordinates.reshape(batch_size, num_samples, *coordinates.shape[1:]),
            "distogram_logits": trunk.distogram_logits,
            "bfactor_logits": self.bfactor_head(trunk.single),
            **({"coordinate_trajectory": coordinate_trajectory} if coordinate_trajectory is not None else {}),
            **(
                {"denoised_coordinate_trajectory": denoised_coordinate_trajectory}
                if denoised_coordinate_trajectory is not None
                else {}
            ),
            "batch_size": [batch_size],
        }
        if residue_type_logits is not None:
            return BoltzGenResidueTypeSampleOutput(
                **values,
                residue_type_logits=residue_type_logits.reshape(
                    batch_size,
                    num_samples,
                    *residue_type_logits.shape[1:],
                ),
            )
        return BoltzGenSampleOutput(
            **values,
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
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                configured value.

        Returns:
            Trunk representations and distogram logits.

        """

        steps = operator.index(self._recycling_steps if recycling_steps is None else recycling_steps)
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
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
        for cycle in range(steps):
            grad_enabled = torch.is_grad_enabled() and cycle == steps - 1
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

    def _build_denoiser_conditioning(
        self,
        trunk: _BoltzGenTrunkOutput,
        input: BoltzGenInput,
    ) -> dict[str, Tensor]:
        r"""
        Precompute the per-structure diffusion conditioning reused by denoising.
        """

        return self.denoiser_conditioning(
            trunk.single,
            trunk.pair,
            trunk.relative_position_encoding,
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
        )


class _BoltzGenDenoisingModel(_BoltzGenStructureModel):
    r"""
    Generate conditioned biomolecular structures with BoltzGen.

    Accepts [`BoltzGenInput`][ofoldx.models.boltzgen.model.BoltzGenInput] and returns
    [`BoltzGenOutput`][ofoldx.models.boltzgen.model.BoltzGenOutput].

    Examples:
        >>> import torch
        >>> from ofoldx.models.boltzgen import (
        ...     BoltzGenConfig,
        ...     BoltzGenModel,
        ...     BoltzGenProcessor,
        ... )
        >>> input = BoltzGenProcessor()(None, total_len=1)
        >>> config = BoltzGenConfig(
        ...     {
        ...         "single_size": 8,
        ...         "pair_size": 8,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "num_distogram_bins": 4,
        ...         "fourier_embedding_size": 8,
        ...         "recycling_steps": 1,
        ...         "input_embeddings": {
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_encoder_attention_heads": 1,
        ...         },
        ...         "trunk": {
        ...             "num_trunk_blocks": 1, "num_pair_attention_heads": 1,
        ...             "num_msa_blocks": 1, "msa_size": 8, "msa_head_dim": 4,
        ...             "num_msa_attention_heads": 1, "outer_product_hidden_size": 4,
        ...             "msa_triangle_attention_hidden_size": 4,
        ...             "num_msa_triangle_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...             "pair_dropout": 0.0, "msa_pair_dropout": 0.0, "msa_dropout": 0.0,
        ...         },
        ...         "template": {
        ...             "num_distogram_bins": 4,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...         },
        ...         "denoiser": {
        ...             "num_token_encoder_blocks": 1,
        ...             "num_token_encoder_attention_heads": 1,
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_decoder_blocks": 1,
        ...             "num_atom_attention_heads": 1,
        ...             "atom_hidden_size": 8,
        ...             "atom_pair_hidden_size": 4,
        ...             "num_query_atoms": 4,
        ...             "num_key_atoms": 4,
        ...             "num_conditioning_transitions": 1,
        ...             "transformer_size": 8,
        ...         },
        ...         "design": {
        ...             "template_size": 8, "num_template_blocks": 1,
        ...             "template_dropout": 0.0,
        ...             "token_distance_size": 8, "num_token_distance_blocks": 1,
        ...             "token_distance_dropout": 0.0, "distance_gaussian_dim": 4,
        ...         },
        ...     }
        ... )
        >>> model = BoltzGenModel(config).eval()
        >>> with torch.inference_mode():
        ...     output = model.sample(
        ...         input, num_samples=1, num_sampling_steps=1, recycling_steps=1,
        ...     )
        >>> tuple(output.coordinates.shape)
        (1, 1, 32, 3)
        >>> tuple(output.distogram_logits.shape)
        (1, 1, 1, 4)
    """

    def _denoise_training(
        self,
        input: BoltzGenInput,
        target: BoltzGenStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _BoltzGenTrainingOutput:
        r"""
        Run stochastic denoising samples for structure training.

        ``B`` is batch size, ``S`` is ``num_samples``, ``N`` is token count,
        and ``A`` is atom count.

        Args:
            input: Inference features and canonical molecular graph.
            target: Structure-training labels with one or ``S`` conformers.
            num_samples: Number of independent noisy denoising samples per batch item.
            sample_chunk_size: Maximum samples per input evaluated by one
                denoiser call. ``None`` evaluates all samples together.
            synchronize_noise_levels: Whether all samples of one batch item share
                one sampled noise level.
            coordinate_augmentation: Whether to apply a random rigid transform to
                each repeated target after centering.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                configured value.
            generator: Random generator used for rigid augmentation, noise-level
                sampling, and coordinate noise.

        Returns:
            Internal denoising tensors used to construct the concrete public
            output type.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if sample_chunk_size is None:
            sample_chunk_size = num_samples
        else:
            sample_chunk_size = operator.index(sample_chunk_size)
            if sample_chunk_size < 1:
                raise ValueError("sample_chunk_size must be positive")
            sample_chunk_size = min(sample_chunk_size, num_samples)
        trunk = self._run_trunk(input, recycling_steps=recycling_steps)
        conditioning = self._build_denoiser_conditioning(trunk, input)
        batch_size, conformers = target.coordinates.shape[:2]
        atom_index = target.coordinate_atom_index.long()
        valid_atom_index = atom_index >= 0
        safe_atom_index = atom_index.clamp_min(0)
        target_coordinates = target.coordinates.gather(
            2,
            safe_atom_index[:, None, :, None].expand(
                -1,
                conformers,
                -1,
                target.coordinates.shape[-1],
            ),
        )
        target_coordinates = target_coordinates.masked_fill(
            ~valid_atom_index[:, None, :, None],
            0,
        )
        target_atom_mask = (
            target.coordinate_mask.gather(
                1,
                safe_atom_index,
            )
            & valid_atom_index
        )
        target_atom_mask = target_atom_mask & input.atom_mask.bool()
        target_coordinates = target_coordinates.masked_fill(
            ~target_atom_mask[:, None, :, None],
            0,
        )
        atom_count = atom_index.shape[1]
        (
            denoised_coordinates,
            augmented_target_coordinates,
            noise_level,
            residue_type_logits,
        ) = self.coordinate_diffusion.denoise_training(
            target_coordinates,
            input.atom_mask,
            multiplicity=num_samples,
            sample_chunk_size=sample_chunk_size,
            synchronize_noise_levels=synchronize_noise_levels,
            coordinate_augmentation=coordinate_augmentation,
            augmentation_translation_scale=augmentation_translation_scale,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
            generator=generator,
            single_inputs=trunk.single_inputs,
            single_trunk=trunk.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        coordinates = denoised_coordinates.reshape(batch_size, num_samples, atom_count, 3)
        augmented_target_coordinates = augmented_target_coordinates.reshape(
            batch_size,
            num_samples,
            atom_count,
            3,
        )
        noise_level = noise_level.reshape(batch_size, num_samples)
        if residue_type_logits is not None:
            residue_type_logits = residue_type_logits.reshape(
                batch_size,
                num_samples,
                *residue_type_logits.shape[1:],
            )
        return _BoltzGenTrainingOutput(
            coordinates=coordinates,
            distogram_logits=trunk.distogram_logits,
            bfactor_logits=self.bfactor_head(trunk.single),
            augmented_target_coordinates=augmented_target_coordinates,
            noise_level=noise_level,
            residue_type_logits=residue_type_logits,
        )


class BoltzGenModel(_BoltzGenDenoisingModel):
    r"""
    Generate conditioned structures with the full Pairformer BoltzGen architecture.
    """

    config_class = BoltzGenConfig
    config: BoltzGenConfig

    def sample(
        self,
        input: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenSampleOutput:
        return cast(
            BoltzGenSampleOutput,
            self._sample(
                input,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                recycling_steps=recycling_steps,
                generator=generator,
                step_scale=step_scale,
                noise_scale=noise_scale,
                return_trajectory=return_trajectory,
            ),
        )

    def forward(
        self,
        input: BoltzGenInput,
        target: BoltzGenStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenOutput:
        output = self._denoise_training(
            input,
            target,
            num_samples=num_samples,
            sample_chunk_size=sample_chunk_size,
            synchronize_noise_levels=synchronize_noise_levels,
            coordinate_augmentation=coordinate_augmentation,
            augmentation_translation_scale=augmentation_translation_scale,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
            recycling_steps=recycling_steps,
            generator=generator,
        )
        return BoltzGenOutput(
            coordinates=output.coordinates,
            distogram_logits=output.distogram_logits,
            bfactor_logits=output.bfactor_logits,
            augmented_target_coordinates=output.augmented_target_coordinates,
            noise_level=output.noise_level,
            batch_size=[output.coordinates.shape[0]],
        )


class BoltzGenResidueTypeModel(_BoltzGenDenoisingModel):
    r"""
    Generate structures and residue-type logits with the full Pairformer architecture.
    """

    config_class = BoltzGenResidueTypeConfig
    config: BoltzGenResidueTypeConfig
    _predict_residue_type = True

    def sample(
        self,
        input: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenResidueTypeSampleOutput:
        return cast(
            BoltzGenResidueTypeSampleOutput,
            self._sample(
                input,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                recycling_steps=recycling_steps,
                generator=generator,
                step_scale=step_scale,
                noise_scale=noise_scale,
                return_trajectory=return_trajectory,
            ),
        )

    def forward(
        self,
        input: BoltzGenInput,
        target: BoltzGenStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenResidueTypeOutput:
        output = self._denoise_training(
            input,
            target,
            num_samples=num_samples,
            sample_chunk_size=sample_chunk_size,
            synchronize_noise_levels=synchronize_noise_levels,
            coordinate_augmentation=coordinate_augmentation,
            augmentation_translation_scale=augmentation_translation_scale,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
            recycling_steps=recycling_steps,
            generator=generator,
        )
        return BoltzGenResidueTypeOutput(
            coordinates=output.coordinates,
            distogram_logits=output.distogram_logits,
            bfactor_logits=output.bfactor_logits,
            augmented_target_coordinates=output.augmented_target_coordinates,
            noise_level=output.noise_level,
            residue_type_logits=cast(Tensor, output.residue_type_logits),
            batch_size=[output.coordinates.shape[0]],
        )


class BoltzGenMiniformerModel(BoltzGenModel):
    r"""
    Generate conditioned structures with the reduced Miniformer architecture.
    """

    config_class = BoltzGenMiniformerConfig
    config: BoltzGenMiniformerConfig
    _miniformer = True


class BoltzGenMiniformerResidueTypeModel(BoltzGenResidueTypeModel):
    r"""
    Generate structures and residue-type logits with the reduced Miniformer architecture.
    """

    config_class = BoltzGenMiniformerResidueTypeConfig
    config: BoltzGenMiniformerResidueTypeConfig
    _miniformer = True
    _predict_residue_type = True


class BoltzGenConfidenceModel(_BoltzGenStructureModel):
    r"""
    Score explicit or newly sampled structures with the BoltzGen confidence equation.

    The model owns the ordinary BoltzGen structure trunk and sampler plus the
    independently parameterized confidence head. Its raw ``forward`` consumes
    explicit ``(B, S, A, 3)`` coordinates; ``sample_with_confidence`` samples
    and scores from one shared trunk execution.

    Examples:
        >>> import torch
        >>> from ofoldx.models.boltzgen import (
        ...     BoltzGenConfidenceConfig,
        ...     BoltzGenConfidenceModel,
        ...     BoltzGenProcessor,
        ... )
        >>> input = BoltzGenProcessor()(None, total_len=2)
        >>> config = BoltzGenConfidenceConfig(
        ...     {
        ...         "single_size": 8,
        ...         "pair_size": 8,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "num_distogram_bins": 4,
        ...         "fourier_embedding_size": 8,
        ...         "recycling_steps": 1,
        ...         "input_embeddings": {
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_encoder_attention_heads": 1,
        ...         },
        ...         "trunk": {
        ...             "num_trunk_blocks": 1, "num_pair_attention_heads": 1,
        ...             "num_msa_blocks": 1, "msa_size": 8, "msa_head_dim": 4,
        ...             "num_msa_attention_heads": 1, "outer_product_hidden_size": 4,
        ...             "msa_triangle_attention_hidden_size": 4,
        ...             "num_msa_triangle_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...             "pair_dropout": 0.0, "msa_pair_dropout": 0.0, "msa_dropout": 0.0,
        ...         },
        ...         "template": {
        ...             "num_distogram_bins": 4,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...         },
        ...         "denoiser": {
        ...             "num_token_encoder_blocks": 1,
        ...             "num_token_encoder_attention_heads": 1,
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_decoder_blocks": 1,
        ...             "num_atom_attention_heads": 1,
        ...             "atom_hidden_size": 8,
        ...             "atom_pair_hidden_size": 4,
        ...             "num_query_atoms": 4,
        ...             "num_key_atoms": 4,
        ...             "num_conditioning_transitions": 1,
        ...             "transformer_size": 8,
        ...         },
        ...         "design": {
        ...             "template_size": 8, "num_template_blocks": 1,
        ...             "template_dropout": 0.0,
        ...             "token_distance_size": 8, "num_token_distance_blocks": 1,
        ...             "token_distance_dropout": 0.0, "distance_gaussian_dim": 4,
        ...         },
        ...         "confidence": {
        ...             "num_pairformer_blocks": 1, "num_pair_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1, "pair_dropout": 0.0,
        ...             "num_distogram_bins": 4, "num_plddt_bins": 4,
        ...             "num_pae_bins": 4, "num_pde_bins": 4,
        ...         },
        ...     }
        ... )
        >>> coordinates = torch.zeros(1, 1, input.atom_mask.shape[1], 3)
        >>> coordinates[0, 0, [0, 1, 2, 3, 14, 15, 16, 17]] = torch.tensor([
        ...     [-1., 0., 0.], [0., 0., 0.], [1., 1., 0.], [1., 2., 0.],
        ...     [2., 1., 0.], [3., 1., 0.], [4., 2., 0.], [4., 3., 0.],
        ... ])
        >>> model = BoltzGenConfidenceModel(config).eval()
        >>> with torch.inference_mode():
        ...     output = model(input, coordinates=coordinates, recycling_steps=1)
        >>> tuple(output.plddt_logits.shape)
        (1, 1, 2, 4)
        >>> tuple(output.pae_logits.shape)
        (1, 1, 2, 2, 4)
    """

    config_class = BoltzGenConfidenceConfig
    config: BoltzGenConfidenceConfig

    def __init__(self, config: BoltzGenConfidenceConfig | None = None) -> None:
        config = BoltzGenConfidenceConfig() if config is None else config
        super().__init__(config)
        confidence = config.confidence
        self.confidence_head = BoltzGenConfidenceHead(
            single_size=config.single_size,
            pair_size=config.pair_size,
            num_blocks=confidence.num_pairformer_blocks,
            num_pair_attention_heads=confidence.num_pair_attention_heads,
            triangle_attention_hidden_size=confidence.triangle_attention_hidden_size,
            num_triangle_attention_heads=confidence.num_triangle_attention_heads,
            pair_dropout=confidence.pair_dropout,
            post_layer_norm=confidence.post_layer_norm,
            num_distogram_bins=confidence.num_distogram_bins,
            max_distance=confidence.max_distance,
            token_level_confidence=confidence.token_level_confidence,
            add_single_input=confidence.add_single_input,
            add_single_to_pair_product=confidence.add_single_to_pair_product,
            use_separate_heads=confidence.use_separate_heads,
            num_plddt_bins=confidence.num_plddt_bins,
            num_pae_bins=confidence.num_pae_bins,
            num_pde_bins=confidence.num_pde_bins,
            return_latent_features=confidence.return_latent_features,
            backend="auto",
        )
        _zero_initialize_boltzgen_updates(self.confidence_head)

    def forward(
        self,
        input: BoltzGenInput,
        *,
        coordinates: Tensor,
        recycling_steps: int | None = None,
    ) -> BoltzGenConfidenceOutput:
        r"""
        Score explicit sampled coordinates.

        Args:
            input: Processed BoltzGen structure input.
            coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)``.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                configured value.

        Returns:
            Complete learned logits and derived confidence tensors.
        """

        trunk = self._run_trunk(input, recycling_steps=recycling_steps)
        return self._score_confidence(input, trunk=trunk, coordinates=coordinates)

    @torch.no_grad()
    def sample_with_confidence(
        self,
        input: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenSampleWithConfidenceOutput:
        r"""
        Sample coordinates and score them without recomputing the structure trunk.
        """

        trunk = self._run_trunk(input, recycling_steps=recycling_steps)
        conditioning = self._build_denoiser_conditioning(trunk, input)
        coordinates, _, coordinate_trajectory, denoised_coordinate_trajectory = self.coordinate_diffusion.sample(
            input.atom_mask,
            num_sampling_steps=num_sampling_steps,
            multiplicity=num_samples,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
            return_trajectory=return_trajectory,
            single_inputs=trunk.single_inputs,
            single_trunk=trunk.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        batch_size = input.token_mask.shape[0]
        coordinates = coordinates.reshape(batch_size, num_samples, *coordinates.shape[1:])
        if coordinate_trajectory is not None:
            coordinate_trajectory = coordinate_trajectory.reshape(
                batch_size,
                num_samples,
                *coordinate_trajectory.shape[1:],
            )
        if denoised_coordinate_trajectory is not None:
            denoised_coordinate_trajectory = denoised_coordinate_trajectory.reshape(
                batch_size,
                num_samples,
                *denoised_coordinate_trajectory.shape[1:],
            )
        bfactor_logits = self.bfactor_head(trunk.single)
        confidence = self._score_confidence(input, trunk=trunk, coordinates=coordinates)
        return BoltzGenSampleWithConfidenceOutput(
            coordinates=coordinates,
            distogram_logits=trunk.distogram_logits,
            bfactor_logits=bfactor_logits,
            confidence=confidence,
            **({"coordinate_trajectory": coordinate_trajectory} if coordinate_trajectory is not None else {}),
            **(
                {"denoised_coordinate_trajectory": denoised_coordinate_trajectory}
                if denoised_coordinate_trajectory is not None
                else {}
            ),
            batch_size=[batch_size],
        )

    def _score_confidence(
        self,
        input: BoltzGenInput,
        *,
        trunk: _BoltzGenTrunkOutput,
        coordinates: Tensor,
    ) -> BoltzGenConfidenceOutput:
        return self.confidence_head(
            trunk.single_inputs.detach(),
            trunk.single.detach(),
            trunk.pair.detach(),
            coordinates.detach(),
            input=input,
            pred_distogram_logits=trunk.distogram_logits.detach(),
        )


class BoltzGenInverseFoldingModel(Model):
    r"""
    Design sequences for fixed biomolecular structures with BoltzGen.

    Accepts [`BoltzGenInverseFoldingInput`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldingInput] and returns
    [`BoltzGenInverseFoldingOutput`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldingOutput].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import MolecularSystem, PolymerSequence, Structure
        >>> from ofoldx.models.boltzgen import (
        ...     BoltzGenInverseFoldingConfig,
        ...     BoltzGenInverseFoldingModel,
        ...     BoltzGenInverseFoldingProcessor,
        ... )
        >>> sequence = PolymerSequence.from_one_letter("GG")
        >>> system = MolecularSystem.from_entities(
        ...     {"protein": sequence}, {"A": "protein"}
        ... )
        >>> atom_count = system.atom_symbol.numel()
        >>> component_count = system.component_symbol.numel()
        >>> structure = Structure(
        ...     system=system,
        ...     coordinates=system.reference_coordinates,
        ...     coordinate_mask=system.reference_coordinate_mask,
        ...     occupancy=torch.zeros(atom_count),
        ...     occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        ...     b_factor=torch.zeros(atom_count),
        ...     b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        ...     source_symbols=(),
        ...     residue_id=torch.full((component_count,), -1),
        ...     insertion_code=torch.full((component_count,), -1),
        ...     standard_component_id=torch.full((component_count,), -1),
        ...     batch_size=[],
        ... )
        >>> input = BoltzGenInverseFoldingProcessor()(
        ...     structure,
        ...     design=(("A", 0), ("A", 1)),
        ... )
        >>> config = BoltzGenInverseFoldingConfig(
        ...     {
        ...         "single_size": 8,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "input_embeddings": {
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_encoder_attention_heads": 1,
        ...         },
        ...         "inverse_folding": {
        ...             "node_size": 8,
        ...             "edge_size": 8,
        ...             "hidden_size": 8,
        ...             "num_encoder_blocks": 1,
        ...             "num_decoder_blocks": 1,
        ...             "num_attention_heads": 1,
        ...             "num_neighbors": 2,
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_encoder_attention_heads": 1,
        ...             "num_query_atoms": 4,
        ...             "num_key_atoms": 4,
        ...             "dropout": 0.0,
        ...             "softmax_dropout": 0.0,
        ...             "coordinate_noise": 0.0,
        ...         },
        ...     }
        ... )
        >>> model = BoltzGenInverseFoldingModel(config).eval()
        >>> output = model.design(input, temperature=None)
        >>> tuple(output.logits.shape)
        (1, 2, 38)
        >>> tuple(output.tokens.shape)
        (1, 2)
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
            num_method_types=config.input_embeddings.num_method_types,
            num_molecule_types=config.input_embeddings.num_molecule_types,
            add_method_conditioning=config.input_embeddings.add_method_conditioning,
            add_modified_flag=config.input_embeddings.add_modified_flag,
            add_cyclic_flag=config.input_embeddings.add_cyclic_flag,
            add_molecule_type_feature=config.input_embeddings.add_molecule_type_feature,
            add_binding_specification=config.input_embeddings.add_binding_specification,
            add_design_mask=config.input_embeddings.add_design_mask,
            add_ss_specification=config.input_embeddings.add_ss_specification,
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
        target_residue_type: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingOutput:
        r"""
        Predict residue-type logits from a fixed structure.

        Args:
            input: Processed inverse-folding input with ``B`` batch items and ``N``
                tokens.
            target_residue_type: Unmasked residue distributions with shape
                ``(B, N, V)`` used only for stochastic teacher forcing.
            generator: Random generator used for training-time coordinate noise
                and teacher-forcing visibility.

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
        edge_index, valid_mask, node, edge = self.inverse_folding_encoder(
            input,
            single_inputs=single_inputs,
            generator=generator,
        )
        logits = self.inverse_folding_decoder(
            node,
            edge,
            edge_index,
            valid_mask,
            target_residue_type,
            generator=generator,
        )
        return BoltzGenInverseFoldingOutput(logits=logits, batch_size=[logits.shape[0]])

    @torch.no_grad()
    def design(
        self,
        input: BoltzGenInverseFoldingInput,
        *,
        temperature: float | None = 0.1,
        tie_symmetric_sequences: bool = True,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingDesignOutput:
        r"""
        Sample residue identities for a fixed structure.

        Args:
            input: Processed inverse-folding input with ``B`` batch items and ``N``
                tokens.
            temperature: Categorical sampling temperature. ``None`` selects the
                highest-logit residue at each decoded position.
            tie_symmetric_sequences: Whether symmetric copies share one sample.
            generator: Random generator used for training-time coordinate noise,
                decode order, and categorical sampling.

        Returns:
            A [`BoltzGenInverseFoldingDesignOutput`][] with ``logits`` of shape
            ``(B, N, V)`` and sampled ``tokens`` of shape ``(B, N)``.
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
        edge_index, valid_mask, node, edge = self.inverse_folding_encoder(
            input,
            single_inputs=single_inputs,
            generator=generator,
        )
        logits, residue_type = self.inverse_folding_decoder.sample(
            node,
            edge,
            edge_index,
            valid_mask,
            input,
            temperature=temperature,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )
        return BoltzGenInverseFoldingDesignOutput(
            logits=logits,
            tokens=residue_type.argmax(dim=-1),
            batch_size=[logits.shape[0]],
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
        vocab_size: int = TOKEN_VOCAB_SIZE,
        num_method_types: int = 12,
        num_molecule_types: int = 4,
        add_method_conditioning: bool = True,
        add_modified_flag: bool = True,
        add_cyclic_flag: bool = True,
        add_molecule_type_feature: bool = True,
        add_binding_specification: bool = False,
        add_design_mask: bool = False,
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
        self.msa_profile_projection = nn.Linear(MSA_VOCAB_SIZE + 1, single_size, bias=False)
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
        if add_molecule_type_feature:
            self.molecule_type_embedding = nn.Embedding(num_molecule_types, single_size)
        if add_binding_specification:
            self.binding_specification_embedding = nn.Embedding(num_binding_types, single_size)
        if add_design_mask:
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


class BoltzGenPairUpdateStack(nn.Module):
    r"""
    Apply a homogeneous sequence of BoltzGen pair-update blocks.

    Template and token-distance embedders differ in feature construction, but
    both repeat the same pair-update equation. The stack owns that repetition
    and its gradient-checkpointing policy.
    """

    _gradient_checkpointing_func: Callable[..., Tensor]

    def __init__(self, blocks: Iterable[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.gradient_checkpointing = False

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                pair = self._gradient_checkpointing_func(
                    block.__call__,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
                pair = block(pair, pair_mask=pair_mask)
        return pair


class BoltzGenTemplateEmbedder(nn.Module):
    r"""
    Embed template geometry as an additive pair update.
    """

    def __init__(
        self,
        pair_size: int = 128,
        template_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = TOKEN_VOCAB_SIZE,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.template_feature_projection = FactorizedPairLinear(
            pair_size=num_bins + 5,
            token_size=vocab_size,
            output_size=template_size,
            bias=False,
            backend=backend,
        )
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, template_size, bias=False)
        self.pair_stack = BoltzGenPairUpdateStack(
            BoltzGenTrianglePairUpdateBlock(
                pair_size=template_size,
                triangle_multiplication_hidden_size=template_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                dropout=dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )
        self.hidden_norm = LayerNorm(template_size, backend=backend)
        self.relu = nn.ReLU()
        self.output_projection = nn.Linear(template_size, pair_size, bias=False)

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
        refined = self.pair_stack(value, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined).reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

        aggregated = (value * present[:, :, None, None, None]).sum(dim=1) / count[:, None, None, None].to(value.dtype)
        return self.output_projection(self.relu(aggregated))


class BoltzGenMiniformerTemplateEmbedder(nn.Module):
    r"""
    Embed template geometry with homogeneous BoltzGen Miniformer pair updates.
    """

    def __init__(
        self,
        pair_size: int = 128,
        template_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = TOKEN_VOCAB_SIZE,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.template_feature_projection = FactorizedPairLinear(
            pair_size=num_bins + 5,
            token_size=vocab_size,
            output_size=template_size,
            bias=False,
            backend=backend,
        )
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, template_size, bias=False)
        self.pair_stack = BoltzGenPairUpdateStack(
            BoltzGenMiniTrianglePairUpdateBlock(
                pair_size=template_size,
                dropout=dropout,
                post_layer_norm=post_layer_norm,
                backend=backend,
            )
            for _ in range(num_blocks)
        )
        self.hidden_norm = LayerNorm(template_size, backend=backend)
        self.relu = nn.ReLU()
        self.output_projection = nn.Linear(template_size, pair_size, bias=False)

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
        refined = self.pair_stack(value, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined).reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

        aggregated = (value * present[:, :, None, None, None]).sum(dim=1) / count[:, None, None, None].to(value.dtype)
        return self.output_projection(self.relu(aggregated))


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
        dropout: float = 0.25,
        use_token_distance_features: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.use_token_distance_features = use_token_distance_features
        self.distance_encoder = (
            BoltzGenDistanceTokenEncoder(
                distance_gaussian_dim=distance_gaussian_dim,
                pair_size=pair_size,
                output_size=pair_size,
            )
            if use_token_distance_features
            else None
        )
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, token_distance_size, bias=False)
        self.distance_feature_projection = nn.Linear(
            num_bins + (4 * pair_size if use_token_distance_features else 0),
            token_distance_size,
            bias=False,
        )
        self.pair_stack = BoltzGenPairUpdateStack(
            BoltzGenTrianglePairUpdateBlock(
                pair_size=token_distance_size,
                triangle_multiplication_hidden_size=token_distance_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                dropout=dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )
        self.hidden_norm = LayerNorm(token_distance_size, backend=backend)
        self.relu = nn.ReLU()
        self.output_projection = nn.Linear(token_distance_size, pair_size, bias=False)

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

        distance_features = (
            torch.cat(
                [
                    distogram,
                    cast(BoltzGenDistanceTokenEncoder, self.distance_encoder)(
                        relative_position_encoding,
                        input,
                    ),
                ],
                dim=-1,
            )
            if self.use_token_distance_features
            else distogram
        )
        distance_features = distance_features * input.token_distance_mask.unsqueeze(-1)
        value = self.pair_projection(self.pair_norm(pair)) + self.distance_feature_projection(
            distance_features.to(pair.dtype)
        )
        refined = self.pair_stack(value, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined)  # one outer residual around the (internally-residual) stack
        return self.output_projection(self.relu(value))


class BoltzGenMiniformerTokenDistanceEmbedder(nn.Module):
    r"""
    Encode inter-token backbone geometry with homogeneous Miniformer pair updates.
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
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        use_token_distance_features: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.use_token_distance_features = use_token_distance_features
        self.distance_encoder = (
            BoltzGenDistanceTokenEncoder(
                distance_gaussian_dim=distance_gaussian_dim,
                pair_size=pair_size,
                output_size=pair_size,
            )
            if use_token_distance_features
            else None
        )
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, token_distance_size, bias=False)
        self.distance_feature_projection = nn.Linear(
            num_bins + (4 * pair_size if use_token_distance_features else 0),
            token_distance_size,
            bias=False,
        )
        self.pair_stack = BoltzGenPairUpdateStack(
            BoltzGenMiniTrianglePairUpdateBlock(
                pair_size=token_distance_size,
                dropout=dropout,
                post_layer_norm=post_layer_norm,
                backend=backend,
            )
            for _ in range(num_blocks)
        )
        self.hidden_norm = LayerNorm(token_distance_size, backend=backend)
        self.relu = nn.ReLU()
        self.output_projection = nn.Linear(token_distance_size, pair_size, bias=False)

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

        distance_features = (
            torch.cat(
                [
                    distogram,
                    cast(BoltzGenDistanceTokenEncoder, self.distance_encoder)(
                        relative_position_encoding,
                        input,
                    ),
                ],
                dim=-1,
            )
            if self.use_token_distance_features
            else distogram
        )
        distance_features = distance_features * input.token_distance_mask.unsqueeze(-1)
        value = self.pair_projection(self.pair_norm(pair)) + self.distance_feature_projection(
            distance_features.to(pair.dtype)
        )
        refined = self.pair_stack(value, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined)  # one outer residual around the (internally-residual) stack
        return self.output_projection(self.relu(value))


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


class BoltzGenBroadcastGaussianSmearing(nn.Module):
    r"""
    Radial-basis distance features ``exp(coeff (d - offset)^2)`` broadcasting over a trailing singleton.
    """

    coeff: float
    offset: Tensor

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


class BoltzGenMsaTrunk(nn.Module):
    r"""
    Build BoltzGen MSA features and refine the pair representation.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        num_blocks: int = 4,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.use_paired_feature = use_paired_feature
        self.msa_projection = nn.Linear(
            MSA_VOCAB_SIZE + 2 + int(use_paired_feature),
            msa_size,
            bias=False,
        )
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            BoltzGenMsaBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                num_msa_attention_heads=num_msa_attention_heads,
                msa_head_dim=msa_head_dim,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

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
            F.one_hot(msa_tokens, num_classes=MSA_VOCAB_SIZE).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa = self.msa_projection(torch.cat(features, dim=-1))
        msa = msa + self.token_projection(single_inputs).unsqueeze(1)
        pair_mask = pair_mask_from_token_mask(token_mask)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                pair, msa = self._gradient_checkpointing_func(
                    block.__call__,
                    pair,
                    msa,
                    pair_mask,
                    msa_mask,
                )
            else:
                pair, msa = block(pair, msa, pair_mask, msa_mask)
        return pair


class BoltzGenMsaBlock(nn.Module):
    r"""
    One BoltzGen MSA block with scalar dropout per MSA row.
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
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
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
        self.msa_dropout = StructuredDropout(msa_dropout, broadcast_dims=(-2, -1))
        self.msa_transition = Transition(msa_size, expansion=expansion, backend=backend)
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.triangle_pair_update = BoltzGenTrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            backend=backend,
        )

    def forward(
        self,
        pair: Tensor,
        msa: Tensor,
        pair_mask: Tensor,
        msa_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        msa = msa + self.msa_dropout(self.pair_weighted_averaging(msa, pair, pair_mask=pair_mask))
        msa = msa + self.msa_transition(msa)
        pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return pair, msa


class BoltzGenPairformerStack(PairformerStack):
    r"""
    Homogeneous stack of BoltzGen Pairformer blocks.

    Uses the sequential block loop from
    [`PairformerStack`][ofoldx.modules.trunk.pairformer.PairformerStack] with BoltzGen blocks.
    """

    def __init__(
        self,
        num_blocks: int = 48,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            BoltzGenPairformerBlock(
                single_size=single_size,
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                expansion=expansion,
                dropout=dropout,
                post_layer_norm=post_layer_norm,
                backend=backend,
            )
            for _ in range(num_blocks)
        )


class BoltzGenPairformerBlock(nn.Module):
    r"""
    One BoltzGen Pairformer block.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_update = BoltzGenTrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
        )
        self.pair_attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)
        self.single_post_norm = LayerNorm(single_size, backend=backend) if post_layer_norm else nn.Identity()

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        pair = self.pair_update(pair, pair_mask=pair_mask)
        single = single + self.pair_attention(single, pair, pair_mask=pair_mask)
        single = single + self.single_transition(single)
        return self.single_post_norm(single), pair


class BoltzGenTrianglePairUpdateBlock(nn.Module):
    r"""
    BoltzGen pair update with one dropout scalar per pair row or column.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
            backend=backend,
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.columnwise_dropout = StructuredDropout(dropout, broadcast_dims=(-3, -1))
        self.transition = Transition(pair_size, expansion=expansion, backend=backend)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask=pair_mask))
        end_pair_mask = None if pair_mask is None else pair_mask.transpose(-1, -2)
        ending_update = self.triangle_attention_ending_node(
            pair.transpose(-3, -2),
            pair_mask=end_pair_mask,
        ).transpose(-3, -2)
        pair = pair + self.columnwise_dropout(ending_update)
        return pair + self.transition(pair)


class BoltzGenMiniformerMsaTrunk(BoltzGenMsaTrunk):
    r"""
    MSA-module variant whose pair track uses compact pair-only Miniformer blocks.

    Uses the MSA feature construction and block loop from
    [`BoltzGenMsaTrunk`][ofoldx.models.boltzgen.model.BoltzGenMsaTrunk] with Miniformer blocks.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        num_blocks: int = 3,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.use_paired_feature = use_paired_feature
        self.msa_projection = nn.Linear(
            MSA_VOCAB_SIZE + 2 + int(use_paired_feature),
            msa_size,
            bias=False,
        )
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(
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
        )


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
        self.msa_dropout = StructuredDropout(msa_dropout, broadcast_dims=(-2, -1))
        self.msa_transition = Transition(msa_size, expansion=expansion, backend=backend)
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.triangle_pair_update = BoltzGenMiniTrianglePairUpdateBlock(
            pair_size=pair_size,
            expansion=expansion,
            dropout=pair_dropout,
            backend=backend,
        )

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
    Stack of blocks in the concrete reduced BoltzGen Miniformer architecture.
    """

    def __init__(
        self,
        num_blocks: int = 12,
        single_size: int = 384,
        pair_size: int = 128,
        num_pair_attention_heads: int = 16,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        use_single_to_pair: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                BoltzGenMiniformerBlock(
                    single_size=single_size,
                    pair_size=pair_size,
                    num_pair_attention_heads=num_pair_attention_heads,
                    dropout=dropout,
                    post_layer_norm=post_layer_norm,
                    use_single_to_pair=use_single_to_pair,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                single, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    single,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
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
        post_layer_norm: bool = False,
        use_single_to_pair: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_to_pair_left_projection = (
            nn.Linear(single_size, pair_size, bias=False) if use_single_to_pair else None
        )
        self.single_to_pair_right_projection = (
            nn.Linear(single_size, pair_size, bias=False) if use_single_to_pair else None
        )
        self.triangle_multiplication = BoltzGenBidirectionalTriangleMultiplication(pair_size)
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.pair_transition = Transition(pair_size, expansion=expansion, backend=backend)
        self.attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)
        self.single_post_norm = LayerNorm(single_size, backend=backend) if post_layer_norm else nn.Identity()

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        left_projection = self.single_to_pair_left_projection
        right_projection = self.single_to_pair_right_projection
        if left_projection is not None and right_projection is not None:
            pair = pair + left_projection(single).unsqueeze(-2) + right_projection(single).unsqueeze(-3)
        pair = pair + self.rowwise_dropout(self.triangle_multiplication(pair, pair_mask=pair_mask))
        pair = pair + self.pair_transition(pair)
        single = single + self.attention(single, pair, pair_mask=pair_mask)
        single = single + self.single_transition(single)
        return self.single_post_norm(single), pair


class BoltzGenMiniTrianglePairUpdateBlock(nn.Module):
    r"""
    Pair-only Miniformer block used in structure-trained-small MSA / template stacks.
    """

    def __init__(
        self,
        pair_size: int = 128,
        expansion: int = 4,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication = BoltzGenBidirectionalTriangleMultiplication(pair_size)
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.pair_transition = Transition(pair_size, expansion=expansion, backend=backend)
        self.pair_post_norm = LayerNorm(pair_size, backend=backend) if post_layer_norm else nn.Identity()

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        pair = pair + self.rowwise_dropout(self.triangle_multiplication(pair, pair_mask=pair_mask))
        pair = pair + self.pair_transition(pair)
        return self.pair_post_norm(pair)


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
        with torch.autocast(device_type=update.device.type, enabled=False):
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
        self.atom_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )
        self.atom_decoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            backend=backend,
        )
        self.token_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=pair_size,
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            backend=backend,
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
        atom_enc_bias = self.atom_encoder_pair_bias_projection(atom_pair)
        atom_dec_bias = self.atom_decoder_pair_bias_projection(atom_pair)
        token_encoder_bias = self.token_encoder_pair_bias_projection(pair)
        return {
            "atom_query": atom_query,
            "atom_conditioning": atom_conditioning,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_encoder_bias": token_encoder_bias,
        }


class BoltzGenGaussianRandom3DEncoding(nn.Module):
    r"""
    Frozen Gaussian features around randomly initialized three-dimensional centers.
    """

    center: Tensor
    std: Tensor

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        self.register_buffer("center", torch.randn(output_size, 3))
        self.register_buffer("std", torch.rand(output_size))

    def forward(self, coordinates: Tensor) -> Tensor:
        squared_distance = (
            (coordinates.unsqueeze(-2) - self.center.view(*(1,) * (coordinates.dim() - 1), self.output_size, 3))
            .square()
            .sum(dim=-1)
        )
        return torch.exp(-squared_distance / (2 * self.std.view(*(1,) * (coordinates.dim() - 1), self.output_size)))


class BoltzGenGaussianDenoisingAtomToTokenEncoder(nn.Module):
    r"""
    Denoising atom encoder with frozen random Gaussian coordinate features.
    """

    def __init__(
        self,
        gaussian_encoding_size: int,
        atom_hidden_size: int = 128,
        transformer_size: int = 768,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        post_layer_norm: bool = False,
        use_qk_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.position_encoding = BoltzGenGaussianRandom3DEncoding(gaussian_encoding_size)
        self.noisy_position_projection = nn.Linear(3 + gaussian_encoding_size, atom_hidden_size, bias=False)
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
        position_features = torch.cat((noisy_positions, self.position_encoding(noisy_positions)), dim=-1)
        atom_query = atom_query + self.noisy_position_projection(position_features)
        conditioning = atom_conditioning.repeat_interleave(multiplicity, 0)
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


class BoltzGenResidueTypeTokenToAtomDecoder(PackedTokenToAtomDecoder):
    r"""
    Decode atom coordinate updates and summed per-token residue-type logits.
    """

    def __init__(
        self,
        *,
        num_residue_types: int,
        transformer_size: int,
        token_hidden_size: int = 384,
        atom_hidden_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        use_qk_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            transformer_size=transformer_size,
            use_qk_norm=use_qk_norm,
            backend=backend,
        )
        self.residue_type_projection = nn.Linear(atom_hidden_size, num_residue_types)

    def forward_with_residue_types(
        self,
        token_repr: Tensor,
        atom_skip: Tensor,
        conditioning_skip: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        multiplicity: int = 1,
    ) -> tuple[Tensor, Tensor]:
        token_index = atom_to_token.repeat_interleave(multiplicity, 0)
        token_projection = self.token_projection(token_repr)
        atom_query = atom_skip + token_projection.gather(
            1,
            token_index.unsqueeze(-1).expand(-1, -1, token_projection.shape[-1]),
        )
        mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_pair_bias = atom_pair_bias.repeat_interleave(multiplicity, 0)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning_skip,
            atom_pair_bias,
            atom_mask=mask.bool(),
        )
        num_tokens = token_repr.shape[-2]
        token_sum = torch.zeros(
            atom_query.shape[0],
            num_tokens,
            atom_query.shape[-1],
            device=atom_query.device,
            dtype=atom_query.dtype,
        )
        token_sum.scatter_add_(
            1,
            token_index.unsqueeze(-1).expand_as(atom_query),
            atom_query * mask.unsqueeze(-1).to(atom_query.dtype),
        )
        return (
            self.position_update_projection(self.position_norm(atom_query)),
            self.residue_type_projection(token_sum),
        )


class _BoltzGenDenoiser(nn.Module):
    r"""
    Encode noisy atom and token features shared by the concrete BoltzGen denoisers.
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
        transformer_size: int | None = None,
        post_layer_norm: bool = False,
        use_qk_norm: bool = False,
        gaussian_random_3d_encoding_dim: int = 0,
        num_residue_types: int = _NUM_RESIDUE_TYPES,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        transformer_size = 2 * single_size if transformer_size is None else transformer_size
        self.single_conditioner = BoltzGenSingleConditioning(
            single_size=single_size,
            transformer_size=transformer_size,
            fourier_embedding_size=fourier_embedding_size,
            num_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.atom_attention_encoder = (
            BoltzGenGaussianDenoisingAtomToTokenEncoder(
                gaussian_encoding_size=gaussian_random_3d_encoding_dim,
                atom_hidden_size=atom_hidden_size,
                transformer_size=transformer_size,
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
                num_blocks=num_atom_encoder_blocks,
                num_attention_heads=num_atom_encoder_attention_heads,
                post_layer_norm=post_layer_norm,
                use_qk_norm=use_qk_norm,
                backend=backend,
            )
            if gaussian_random_3d_encoding_dim > 0
            else PackedDenoisingAtomToTokenEncoder(
                atom_hidden_size=atom_hidden_size,
                token_hidden_size=single_size,
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
                num_blocks=num_atom_encoder_blocks,
                num_attention_heads=num_atom_encoder_attention_heads,
                transformer_size=transformer_size,
                post_layer_norm=post_layer_norm,
                use_qk_norm=use_qk_norm,
                backend=backend,
            )
        )
        self.single_to_token_norm = LayerNorm(transformer_size, backend=backend)
        self.single_to_token_projection = nn.Linear(transformer_size, transformer_size, bias=False)
        self.token_encoder = ConditionedTransformerEncoder(
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            hidden_size=transformer_size,
            conditioning_size=transformer_size,
            use_qk_norm=use_qk_norm,
            backend=backend,
        )
        self.token_norm = LayerNorm(transformer_size, backend=backend)
        self.atom_attention_decoder = self._build_decoder(
            num_residue_types=num_residue_types,
            transformer_size=transformer_size,
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            use_qk_norm=use_qk_norm,
            backend=backend,
        )

    def _build_decoder(
        self,
        *,
        num_residue_types: int,
        transformer_size: int,
        atom_hidden_size: int,
        token_hidden_size: int,
        num_query_atoms: int,
        num_key_atoms: int,
        num_blocks: int,
        num_attention_heads: int,
        use_qk_norm: bool,
        backend: str,
    ) -> nn.Module:
        raise NotImplementedError

    def _encode(
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

        return (
            token_repr,
            atom_skip,
            conditioning_skip,
            conditioning["atom_dec_bias"],
        )


class BoltzGenDenoiser(_BoltzGenDenoiser):
    r"""
    Predict per-atom position updates for EDM sampling.
    """

    def _build_decoder(
        self,
        *,
        num_residue_types: int,
        transformer_size: int,
        atom_hidden_size: int,
        token_hidden_size: int,
        num_query_atoms: int,
        num_key_atoms: int,
        num_blocks: int,
        num_attention_heads: int,
        use_qk_norm: bool,
        backend: str,
    ) -> PackedTokenToAtomDecoder:
        del num_residue_types
        return PackedTokenToAtomDecoder(
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=token_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            transformer_size=transformer_size,
            use_qk_norm=use_qk_norm,
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
    ) -> tuple[Tensor, None]:
        token_repr, atom_skip, conditioning_skip, atom_pair_bias = self._encode(
            single_inputs,
            single_trunk,
            noisy_positions,
            noise_level,
            conditioning,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            token_mask=token_mask,
            multiplicity=multiplicity,
        )
        return (
            self.atom_attention_decoder(
                token_repr,
                atom_skip,
                conditioning_skip,
                atom_pair_bias,
                atom_mask,
                atom_to_token,
                multiplicity=multiplicity,
            ),
            None,
        )


class BoltzGenResidueTypeDenoiser(_BoltzGenDenoiser):
    r"""
    Predict per-atom position updates and per-token residue identities.
    """

    def _build_decoder(
        self,
        *,
        num_residue_types: int,
        transformer_size: int,
        atom_hidden_size: int,
        token_hidden_size: int,
        num_query_atoms: int,
        num_key_atoms: int,
        num_blocks: int,
        num_attention_heads: int,
        use_qk_norm: bool,
        backend: str,
    ) -> BoltzGenResidueTypeTokenToAtomDecoder:
        return BoltzGenResidueTypeTokenToAtomDecoder(
            num_residue_types=num_residue_types,
            transformer_size=transformer_size,
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=token_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            use_qk_norm=use_qk_norm,
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
    ) -> tuple[Tensor, Tensor]:
        token_repr, atom_skip, conditioning_skip, atom_pair_bias = self._encode(
            single_inputs,
            single_trunk,
            noisy_positions,
            noise_level,
            conditioning,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            token_mask=token_mask,
            multiplicity=multiplicity,
        )
        decoder = cast(BoltzGenResidueTypeTokenToAtomDecoder, self.atom_attention_decoder)
        return decoder.forward_with_residue_types(
            token_repr,
            atom_skip,
            conditioning_skip,
            atom_pair_bias,
            atom_mask,
            atom_to_token,
            multiplicity=multiplicity,
        )


class BoltzGenCoordinateDiffusion(nn.Module):
    r"""
    EDM sampler wrapping a [`BoltzGenDenoiser`][ofoldx.models.boltzgen.model.BoltzGenDenoiser].
    """

    def __init__(
        self,
        denoiser: BoltzGenDenoiser | BoltzGenResidueTypeDenoiser,
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
        self.sigma_data = sigma_data
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.denoiser = denoiser
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
    ) -> tuple[Tensor, Tensor | None]:
        batch = noisy_coordinates.shape[0]
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device, dtype=noisy_coordinates.dtype).expand(batch)
        padded = sigma.reshape(batch, 1, 1)
        update, residue_type_logits = self.denoiser(
            noisy_positions=self.preconditioner.c_in(padded) * noisy_coordinates,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
            **conditioning,
        )
        coordinates = (
            self.preconditioner.c_skip(padded) * noisy_coordinates + self.preconditioner.c_out(padded) * update
        )
        return coordinates, residue_type_logits

    def noise_distribution(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        log_mean: float | None = None,
        log_std: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        log_mean = self.noise_log_mean if log_mean is None else log_mean
        log_std = self.noise_log_std if log_std is None else log_std
        noise = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        return self.sigma_data * (log_mean + log_std * noise).exp()

    def denoise_training(
        self,
        target_coordinates: Tensor,
        atom_mask: Tensor,
        *,
        multiplicity: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        generator: torch.Generator | None = None,
        single_inputs: Tensor,
        single_trunk: Tensor,
        conditioning: dict[str, Tensor],
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        r"""
        Corrupt target conformers and run one preconditioned denoising step.

        A singleton conformer is repeated for every sample; otherwise the
        conformer count must equal ``multiplicity``. Returned coordinate tensors
        and noise levels keep the flattened ``B * multiplicity`` sample axis.
        """

        sample_chunk_size = multiplicity if sample_chunk_size is None else min(sample_chunk_size, multiplicity)
        device, dtype = single_trunk.device, single_trunk.dtype
        target_coordinates = target_coordinates.to(device=device, dtype=dtype)
        batch_size, conformers, atom_count = target_coordinates.shape[:3]
        if conformers == 1:
            target_coordinates = target_coordinates.expand(
                -1,
                multiplicity,
                -1,
                -1,
            )
        elif conformers != multiplicity:
            raise ValueError(
                f"BoltzGen target conformer count must be one or equal num_samples, got {conformers} and {multiplicity}"
            )
        target_coordinates = target_coordinates.reshape(
            batch_size * multiplicity,
            atom_count,
            3,
        )
        # Centre and rigid-augment over ALL padded atoms (atom_pad_mask), matching upstream
        # ``center_random_augmentation``. The resolved-atom mask is a loss/alignment concern applied by the
        # criterion, not part of the corruption equation.
        augmentation_mask = atom_mask.to(device=device, dtype=dtype).repeat_interleave(multiplicity, dim=0)
        augmented_target_coordinates = _center_random_augmentation(
            target_coordinates,
            augmentation_mask,
            augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        noise_count = batch_size if synchronize_noise_levels else batch_size * multiplicity
        noise_level = self.noise_distribution(
            noise_count,
            device=augmented_target_coordinates.device,
            dtype=augmented_target_coordinates.dtype,
            log_mean=noise_log_mean,
            log_std=noise_log_std,
            generator=generator,
        )
        if synchronize_noise_levels:
            noise_level = noise_level.repeat_interleave(multiplicity)
        noise = torch.randn(
            augmented_target_coordinates.shape,
            device=augmented_target_coordinates.device,
            dtype=augmented_target_coordinates.dtype,
            generator=generator,
        )
        noisy_coordinates = augmented_target_coordinates + noise_level.reshape(-1, 1, 1) * noise
        noisy_by_sample = noisy_coordinates.reshape(batch_size, multiplicity, *noisy_coordinates.shape[1:])
        noise_level_by_sample = noise_level.reshape(batch_size, multiplicity)
        denoised_chunks = []
        residue_type_chunks = []
        for sample_start in range(0, multiplicity, sample_chunk_size):
            sample_end = min(sample_start + sample_chunk_size, multiplicity)
            chunk_num_samples = sample_end - sample_start
            denoised_chunk, residue_type_chunk = self.preconditioned_network_forward(
                noisy_by_sample[:, sample_start:sample_end].flatten(0, 1),
                noise_level_by_sample[:, sample_start:sample_end].flatten(),
                multiplicity=chunk_num_samples,
                atom_mask=atom_mask,
                single_inputs=single_inputs,
                single_trunk=single_trunk,
                conditioning=conditioning,
                atom_to_token=atom_to_token,
                token_mask=token_mask,
            )
            denoised_chunks.append(denoised_chunk.reshape(batch_size, chunk_num_samples, *noisy_coordinates.shape[1:]))
            if residue_type_chunk is not None:
                residue_type_chunks.append(
                    residue_type_chunk.reshape(batch_size, chunk_num_samples, *residue_type_chunk.shape[1:])
                )
        denoised_coordinates = (
            denoised_chunks[0] if len(denoised_chunks) == 1 else torch.cat(denoised_chunks, dim=1)
        ).flatten(0, 1)
        residue_type_logits = None
        if residue_type_chunks:
            residue_type_logits = (
                residue_type_chunks[0] if len(residue_type_chunks) == 1 else torch.cat(residue_type_chunks, dim=1)
            ).flatten(0, 1)
        return (
            denoised_coordinates,
            augmented_target_coordinates,
            noise_level,
            residue_type_logits,
        )

    def loss_weight(self, sigma: Tensor) -> Tensor:
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def sample_schedule_karras(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        steps = operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps)
        if steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        inv_rho = 1 / self.rho
        ramp = torch.arange(steps, device=device, dtype=dtype) / max(steps - 1, 1)
        sigmas = (
            self.max_noise_level**inv_rho + ramp * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def sample_schedule_dilated(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        steps = operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps)
        if steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        inv_rho = 1 / self.rho
        ramp = torch.arange(steps, device=device, dtype=dtype) / max(steps - 1, 1)
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

    def sample_schedule(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.sampling_schedule == "af3":
            return self.sample_schedule_karras(num_sampling_steps, device=device, dtype=dtype)
        if self.sampling_schedule == "dilated":
            return self.sample_schedule_dilated(num_sampling_steps, device=device, dtype=dtype)
        raise ValueError(f"unknown BoltzGen sampling_schedule: {self.sampling_schedule!r}")

    @torch.no_grad()
    def sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        return_trajectory: bool = False,
        **conditioning: object,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        single_trunk = cast(Tensor, conditioning["single_trunk"])
        device, dtype = single_trunk.device, single_trunk.dtype
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        step_scale_value = torch.as_tensor(
            self.step_scale if step_scale is None else step_scale,
            device=device,
            dtype=sigmas.dtype,
        )
        noise_scale_value = torch.as_tensor(
            self.noise_scale if noise_scale is None else noise_scale,
            device=device,
            dtype=sigmas.dtype,
        )
        if step_scale_value.ndim:
            step_scale_value = step_scale_value.reshape(-1, 1, 1)
        if noise_scale_value.ndim:
            noise_scale_value = noise_scale_value.reshape(-1, 1, 1)

        def randn() -> Tensor:
            r"""
            Draw standard Gaussian noise of the expanded coordinate shape ``(B*multiplicity, N_atom, 3)``.
            """
            return torch.randn(shape, device=device, dtype=dtype, generator=generator)

        coords = sigmas[0] * randn()
        denoised = None
        residue_type_logits = None
        if return_trajectory:
            coordinate_steps: list[Tensor] = [coords]
            denoised_steps: list[Tensor] = []
        # The loop owns the squared-noise-scale churn factor, optional mid-step alignment, dilated Karras schedule,
        # rigid augmentation, and random draws as one sampling algorithm.
        # The fixed Python loop controls denoiser invocation; schedule values stay as tensors so sampling does not
        # synchronize them to the host on every step.
        for index in range(len(sigmas) - 1):
            sigma_current, sigma_next, gamma = sigmas[index], sigmas[index + 1], gammas[index + 1]
            rotation = uniform_random_rotation(
                expanded_atom_mask.shape[0], device=device, dtype=coords.dtype, generator=generator
            )
            translation = torch.randn(
                expanded_atom_mask.shape[0],
                1,
                3,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            coords = torch.einsum("bmd,bds->bms", coords - coords.mean(-2, keepdim=True), rotation) + translation
            if denoised is not None:
                denoised = (
                    torch.einsum("bmd,bds->bms", denoised - denoised.mean(-2, keepdim=True), rotation) + translation
                )

            sigma_hat = sigma_current * (1 + gamma)
            noise_var = noise_scale_value**2 * (sigma_hat**2 - sigma_current**2)
            noisy = coords + noise_scale_value * torch.sqrt(torch.clamp(noise_var, min=0.0)) * randn()
            denoised, residue_type_logits = self.preconditioned_network_forward(
                noisy, sigma_hat, multiplicity=multiplicity, atom_mask=base_atom_mask, **conditioning
            )
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy,
                    target=denoised,
                    weight=expanded_atom_mask,
                )
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale_value * (sigma_next - sigma_hat) * derivative
            if return_trajectory:
                coordinate_steps.append(coords)
                denoised_steps.append(denoised)
        if not return_trajectory:
            return coords, residue_type_logits, None, None
        return coords, residue_type_logits, torch.stack(coordinate_steps, dim=1), torch.stack(denoised_steps, dim=1)


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
        self.pair_encoding = AtomPairEncoding(atom_hidden_size, atom_pair_hidden_size)
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.single_trunk_norm = LayerNorm(single_size)
        self.single_trunk_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.pair_trunk_norm = LayerNorm(pair_size)
        self.pair_trunk_projection = nn.Linear(
            pair_size,
            atom_pair_hidden_size,
            bias=False,
        )

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
        atom_pair = self.pair_encoding(
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


class BoltzGenSingleConditioning(nn.Module):
    r"""
    Condition the concatenated trunk and input single representation on the diffusion noise level.
    """

    def __init__(
        self,
        single_size: int = 384,
        transformer_size: int | None = None,
        fourier_embedding_size: int = 256,
        num_transitions: int = 2,
        transition_expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        input_size = 2 * single_size
        transformer_size = input_size if transformer_size is None else transformer_size
        self.single_norm = LayerNorm(input_size, backend=backend)
        self.input_projection = nn.Linear(input_size, transformer_size) if transformer_size != input_size else None
        self.single_projection = nn.Linear(transformer_size, transformer_size)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size, backend=backend)
        self.fourier_projection = nn.Linear(fourier_embedding_size, transformer_size, bias=False)
        self.transitions = nn.ModuleList(
            Transition(transformer_size, expansion=transition_expansion, backend=backend)
            for _ in range(num_transitions)
        )

    def forward(self, noise_level: Tensor, single_trunk: Tensor, single_inputs: Tensor) -> tuple[Tensor, Tensor]:
        single = self.single_norm(torch.cat((single_trunk, single_inputs), dim=-1))
        if self.input_projection is not None:
            single = self.input_projection(single)
        single = self.single_projection(single)
        normed_noise = self.fourier_norm(self.noise_fourier(noise_level).to(single.dtype))
        single = single + self.fourier_projection(normed_noise).unsqueeze(-2)
        for transition in self.transitions:
            single = single + transition(single)
        return single, normed_noise


class BoltzGenConfidenceHead(nn.Module):
    r"""
    BoltzGen confidence trunk and complete learned output heads.
    """

    boundaries: Tensor

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 8,
        num_pair_attention_heads: int = 16,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        pair_dropout: float = 0.25,
        post_layer_norm: bool = False,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        token_level_confidence: bool = True,
        add_single_input: bool = False,
        add_single_to_pair_product: bool = False,
        use_separate_heads: bool = False,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        return_latent_features: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_level_confidence = token_level_confidence
        self.add_single_input = add_single_input
        self.add_single_to_pair_product = add_single_to_pair_product
        self.return_latent_features = return_latent_features

        self.input_single_norm = LayerNorm(single_size, backend=backend)
        self.single_norm = LayerNorm(single_size, backend=backend)
        if add_single_input:
            self.input_single_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.single_to_pair_row = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_col = nn.Linear(single_size, pair_size, bias=False)
        if add_single_to_pair_product:
            self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
            self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
            self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)
        self.register_buffer(
            "boundaries",
            torch.linspace(2.0, max_distance, num_distogram_bins - 1),
        )
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)
        self.pairformer_stack = BoltzGenPairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            dropout=pair_dropout,
            post_layer_norm=post_layer_norm,
            backend=backend,
        )
        self.output_head = BoltzGenConfidenceOutputHead(
            single_size=single_size,
            pair_size=pair_size,
            token_level_confidence=token_level_confidence,
            use_separate_heads=use_separate_heads,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
        )

        nn.init.zeros_(self.distance_embedding.weight)
        nn.init.zeros_(self.single_to_pair_row.weight)
        nn.init.zeros_(self.single_to_pair_col.weight)
        if add_single_to_pair_product:
            nn.init.zeros_(self.single_to_pair_product_out.weight)
        if add_single_input:
            nn.init.zeros_(self.input_single_projection.weight)

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        input: BoltzGenInput,
        pred_distogram_logits: Tensor,
    ) -> BoltzGenConfidenceOutput:
        batch_size, num_samples, num_atoms, _ = coordinates.shape
        flat_coordinates = coordinates.reshape(batch_size * num_samples, num_atoms, 3)

        single_inputs = self.input_single_norm(single_inputs)
        single = self.single_norm(single)
        if self.add_single_input:
            single = single + self.input_single_projection(single_inputs)
        pair = (
            self.pair_norm(pair)
            + self.single_to_pair_row(single_inputs).unsqueeze(-2)
            + self.single_to_pair_col(single_inputs).unsqueeze(-3)
        )
        if self.add_single_to_pair_product:
            pair = pair + self.single_to_pair_product_out(
                self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
                * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
            )

        single = single.repeat_interleave(num_samples, dim=0)
        pair = pair.repeat_interleave(num_samples, dim=0)
        representative_atom_index = input.token_center_atom_index.repeat_interleave(num_samples, dim=0).long()
        representative_coordinates = flat_coordinates.gather(
            1,
            representative_atom_index.unsqueeze(-1).expand(-1, -1, 3),
        )
        representative_distances = torch.cdist(representative_coordinates, representative_coordinates)
        distance_bins = (representative_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1)
        pair = pair + self.distance_embedding(distance_bins)

        token_mask = input.token_mask.repeat_interleave(num_samples, dim=0)
        pair_mask = pair_mask_from_token_mask(token_mask)
        single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        pae_logits, pde_logits, plddt_logits, resolved_logits = self.output_head(
            single,
            pair,
            input=input,
            multiplicity=num_samples,
        )
        return _boltzgen_confidence_output(
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            confidence_single=single if self.return_latent_features else None,
            confidence_pair=pair if self.return_latent_features else None,
            coordinates=flat_coordinates,
            representative_distances=representative_distances,
            pred_distogram_logits=pred_distogram_logits,
            input=input,
            multiplicity=num_samples,
            token_level_confidence=self.token_level_confidence,
        )


class BoltzGenConfidenceOutputHead(nn.Module):
    r"""
    Project confidence representations to PAE, PDE, pLDDT, and resolved logits.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        token_level_confidence: bool = True,
        use_separate_heads: bool = False,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
    ) -> None:
        super().__init__()
        self.token_level_confidence = token_level_confidence
        self.use_separate_heads = use_separate_heads
        if use_separate_heads:
            self.pae_intra_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
            self.pae_inter_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
            self.pde_intra_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
            self.pde_inter_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        else:
            self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
            self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        plddt_size = num_plddt_bins if token_level_confidence else 23 * num_plddt_bins
        resolved_size = 2 if token_level_confidence else 23 * 2
        self.plddt_projection = nn.Linear(single_size, plddt_size, bias=False)
        self.resolved_projection = nn.Linear(single_size, resolved_size, bias=False)

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        *,
        input: BoltzGenInput,
        multiplicity: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        chain_index = input.chain_index.repeat_interleave(multiplicity, dim=0)
        if self.use_separate_heads:
            same_chain = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).to(pair).unsqueeze(-1)
            different_chain = 1.0 - same_chain
            pae_logits = (
                self.pae_intra_projection(pair) * same_chain + self.pae_inter_projection(pair) * different_chain
            )
            symmetric_pair = pair + pair.transpose(-3, -2)
            pde_logits = (
                self.pde_intra_projection(symmetric_pair) * same_chain
                + self.pde_inter_projection(symmetric_pair) * different_chain
            )
        else:
            pae_logits = self.pae_projection(pair)
            pde_logits = self.pde_projection(pair + pair.transpose(-3, -2))

        plddt_logits = self.plddt_projection(single)
        resolved_logits = self.resolved_projection(single)
        if self.token_level_confidence:
            return pae_logits, pde_logits, plddt_logits, resolved_logits

        num_bins = plddt_logits.shape[-1] // 23
        plddt_slots = plddt_logits.reshape(*plddt_logits.shape[:-1], 23, num_bins)
        resolved_slots = resolved_logits.reshape(*resolved_logits.shape[:-1], 23, 2)
        atom_to_token = input.atom_to_token.repeat_interleave(multiplicity, dim=0).long()
        atom_token_index = input.atom_token_index.repeat_interleave(multiplicity, dim=0).long()
        batch_index = torch.arange(single.shape[0], device=single.device).unsqueeze(-1)
        plddt_logits = plddt_slots[batch_index, atom_to_token, atom_token_index]
        resolved_logits = resolved_slots[batch_index, atom_to_token, atom_token_index]
        atom_mask = input.atom_mask.repeat_interleave(multiplicity, dim=0).to(plddt_logits)
        plddt_logits = plddt_logits * atom_mask.unsqueeze(-1)
        resolved_logits = resolved_logits * atom_mask.unsqueeze(-1)
        return pae_logits, pde_logits, plddt_logits, resolved_logits


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
        self.gradient_checkpointing = False
        self.num_neighbors = num_neighbors
        self.max_relative_position = max_relative_position
        self.coordinate_noise = coordinate_noise
        # geo (4x4 backbone-atom distances x 16 RBF = 256)
        # + edge attrs (2*max_relative_position+2 + same_chain + same_residue)
        # + bond (token_bonds + one-hot bond type).
        self.edge_input_size = 256 + (2 * max_relative_position + 2) + 2 + 1 + (_NUM_BOND_TYPES + 1)

        self.distance_smearing = BoltzGenGaussianSmearing(start=0.0, stop=20.0, num_gaussians=16)
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

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    @torch.no_grad()
    def build_knn_graph(self, input: BoltzGenInverseFoldingInput) -> tuple[Tensor, Tensor, Tensor]:
        valid_mask = input.token_mask.bool() & input.token_resolved_mask.bool()
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
    def edge_geometry(
        self,
        input: BoltzGenInverseFoldingInput,
        edge_index: Tensor,
        valid_mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        batch_size, num_tokens = valid_mask.shape
        coords = input.atom_coordinates
        if self.training and self.coordinate_noise > 0:
            coords = (
                coords
                + torch.randn(
                    coords.shape,
                    device=coords.device,
                    dtype=coords.dtype,
                    generator=generator,
                )
                * self.coordinate_noise
            )
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
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_index, valid_mask, bond = self.build_knn_graph(input)
        geometry = self.edge_geometry(
            input,
            edge_index,
            valid_mask,
            generator=generator,
        )
        edge_attr = self.edge_attributes(input, edge_index, valid_mask)

        node = self.node_input_projection(single_inputs[valid_mask])
        edge = self.edge_input_projection(torch.cat([geometry, edge_attr, bond], dim=-1))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                node, edge = self._gradient_checkpointing_func(
                    block.__call__,
                    node,
                    edge,
                    edge_index,
                )
            else:
                node, edge = block(node, edge, edge_index)
        return edge_index, valid_mask, node, edge


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
        self.gradient_checkpointing = False
        self.vocab_size = vocab_size
        # Each block's edge state is cat[encoder_edge, neighbour_node + residue_rep] = edge_size + node_size.
        decoder_edge_size = edge_size + node_size
        self.residue_type_projection = nn.Linear(vocab_size, node_size)
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
        self.residue_type_head = nn.Linear(node_size, vocab_size, bias=False)

    _gradient_checkpointing_func: Callable[..., Tensor]

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
        valid_mask: Tensor,
        target_residue_type: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        with torch.no_grad():
            # Teacher forcing with random per-edge visibility: an edge reveals its source residue type only when
            # rand[src] < rand[dst], so each node sees a random subset of neighbor identities.
            rand = torch.rand(node.shape[0], device=node.device, generator=generator)
            visibility = rand[src] < rand[dst]
            known = target_residue_type.bool()[valid_mask][src] * visibility[:, None]
            known = known.to(edge.dtype)
        residue_rep = self.residue_type_projection(known)
        neighbors = torch.cat([edge, node[src] + residue_rep], dim=-1)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                node = self._gradient_checkpointing_func(
                    block.__call__,
                    node,
                    neighbors,
                    edge_index,
                )
            else:
                node = block(node, neighbors, edge_index)

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
        temperature: float | None = 0.1,
        tie_symmetric_sequences: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = node.shape[0]
        design_mask = input.design_mask.bool()[valid_mask]
        decoded = node.new_zeros(num_nodes, self.vocab_size)
        logits = node.new_zeros(num_nodes, self.vocab_size)

        blocked = input.aa_constraint_mask[valid_mask].to(device=node.device) > 0
        all_blocked = blocked[design_mask].all(dim=-1)
        if all_blocked.any():
            design_positions = torch.where(design_mask)[0]
            raise ValueError(
                f"inverse folding has no valid amino acids at positions {design_positions[all_blocked].tolist()}"
            )

        order = torch.randperm(num_nodes, device=node.device, generator=generator).tolist()
        if (~design_mask).any():
            known = input.residue_type_features[valid_mask][~design_mask].to(decoded.dtype)
            decoded[~design_mask] = known
            logits[~design_mask] = known
            fixed = set(torch.where(~design_mask)[0].tolist())
            order = [position for position in order if position not in fixed]

        symmetry_groups, position_to_group = (
            self._build_symmetric_groups(input, valid_mask) if tie_symmetric_sequences else ({}, {})
        )
        group_anchors: dict[int, int] = {}
        for anchor_group_id, group_positions in symmetry_groups.items():
            fixed_positions = [position for position in group_positions if not bool(design_mask[position])]
            if not fixed_positions:
                continue
            identities = decoded[fixed_positions].argmax(dim=-1).unique()
            if identities.numel() != 1:
                raise ValueError(
                    f"inverse-folding symmetry group has conflicting fixed identities at positions {fixed_positions}"
                )
            identity = int(identities.item())
            canonical_identity = identity - _CANONICALS_OFFSET
            if not 0 <= canonical_identity < NUM_CANONICAL_TOKENS:
                raise ValueError(
                    f"inverse-folding symmetry group has a noncanonical fixed identity at positions {fixed_positions}"
                )
            group_anchors[anchor_group_id] = canonical_identity

        sampled: set[int] = set()

        for position in order:
            if position in sampled:
                continue
            group_id = position_to_group.get(position)
            positions = (
                [member for member in symmetry_groups[group_id] if bool(design_mask[member])]
                if group_id is not None
                else [position]
            )

            aggregated_logits: Tensor | None = None
            for pos in positions:
                incoming = dst == pos
                neighbors = torch.cat(
                    [edge[incoming], node[src[incoming]] + self.residue_type_projection(decoded[src[incoming]])],
                    dim=-1,
                )
                state = node[pos : pos + 1]
                for block in self.blocks:
                    state = cast(BoltzGenInverseFoldingGraphDecoderBlock, block).sample(state, neighbors)
                position_logits = self.residue_type_head(state)
                aggregated_logits = (
                    position_logits if aggregated_logits is None else aggregated_logits + position_logits
                )
            aggregated_logits = cast(Tensor, aggregated_logits) / len(positions)

            shared_blocked = blocked[positions].any(dim=0, keepdim=True)
            if bool(shared_blocked.all()):
                raise ValueError(f"inverse-folding symmetry group has no shared amino acid at positions {positions}")
            canonical = (
                aggregated_logits[:, _CANONICALS_OFFSET : _CANONICALS_OFFSET + NUM_CANONICAL_TOKENS]
                + shared_blocked.to(dtype=node.dtype) * -1e6
            )
            anchor = group_anchors.get(group_id) if group_id is not None else None
            if anchor is not None:
                if bool(shared_blocked[0, anchor]):
                    raise ValueError(
                        "inverse-folding symmetry-group constraints exclude its fixed identity at "
                        f"positions {positions}"
                    )
                chosen = canonical.new_tensor([anchor], dtype=torch.long)
            elif temperature is None:
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
        self,
        input: BoltzGenInverseFoldingInput,
        valid_mask: Tensor,
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        batch_indices = valid_mask.nonzero(as_tuple=False)[:, 0].tolist()
        group_indices = input.symmetric_group[valid_mask].tolist()
        key_to_positions: dict[tuple[int, int], list[int]] = {}
        for position, (batch_index, group) in enumerate(zip(batch_indices, group_indices, strict=True)):
            if group <= 0:  # 0 = no symmetry group.
                continue
            key_to_positions.setdefault((batch_index, group), []).append(position)

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
        self.activation = nn.GELU()
        self.hidden_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)
        )
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.activation(self.input_projection(features))
        for projection in self.hidden_projections:
            hidden = self.activation(projection(hidden))
        return self.output_projection(hidden)


class BoltzGenGaussianSmearing(nn.Module):
    r"""
    Radial-basis distance featurization with a registered ``offset`` buffer.
    """

    coeff: float
    offset: Tensor

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


def _boltzgen_confidence_output(
    *,
    pae_logits: Tensor,
    pde_logits: Tensor,
    plddt_logits: Tensor,
    resolved_logits: Tensor,
    confidence_single: Tensor | None,
    confidence_pair: Tensor | None,
    coordinates: Tensor,
    representative_distances: Tensor,
    pred_distogram_logits: Tensor,
    input: BoltzGenInput,
    multiplicity: int,
    token_level_confidence: bool,
) -> BoltzGenConfidenceOutput:
    r"""
    Derive the complete released BoltzGen confidence tensor set.
    """

    batch_size = input.token_mask.shape[0]
    token_mask = input.token_mask.repeat_interleave(multiplicity, dim=0).to(pae_logits)
    chain_index = input.chain_index.repeat_interleave(multiplicity, dim=0)
    molecule_type = input.molecule_type.repeat_interleave(multiplicity, dim=0)
    is_ligand_token = (molecule_type == 3).to(pae_logits)
    is_protein_token = (molecule_type == 0).to(pae_logits)

    plddt = _boltzgen_expected_metric(plddt_logits, end=100.0)  # 0-100, shared with AF3/Protenix
    if token_level_confidence:
        confidence_mask = token_mask
        complex_plddt = (plddt * confidence_mask).sum(dim=-1) / confidence_mask.sum(dim=-1)
        token_interface_mask = (
            (representative_distances < 8)
            & (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2))
            & ~is_ligand_token.bool().unsqueeze(-1)
        ).any(dim=-1)
        non_interface_mask = ~token_interface_mask & ~is_ligand_token.bool()
        iplddt_weight = is_ligand_token * 20.0 + token_interface_mask.to(plddt) * 10.0 + non_interface_mask.to(plddt)
    else:
        atom_mask = input.atom_mask.repeat_interleave(multiplicity, dim=0).to(plddt)
        confidence_mask = atom_mask
        complex_plddt = (plddt * confidence_mask).sum(dim=-1) / confidence_mask.sum(dim=-1)
        atom_to_token = input.atom_to_token.repeat_interleave(multiplicity, dim=0).long()
        atom_molecule_type = molecule_type.gather(1, atom_to_token)
        atom_chain_index = chain_index.gather(1, atom_to_token)
        is_ligand_atom = atom_molecule_type == 3
        atom_interface_mask = (
            (torch.cdist(coordinates, coordinates) < 8)
            & (atom_chain_index.unsqueeze(-1) != atom_chain_index.unsqueeze(-2))
            & ~is_ligand_atom.unsqueeze(-1)
        ).any(dim=-1)
        non_interface_mask = ~atom_interface_mask & ~is_ligand_atom
        iplddt_weight = (
            is_ligand_atom.to(plddt) * 20.0 + atom_interface_mask.to(plddt) * 10.0 + non_interface_mask.to(plddt)
        )
    complex_iplddt = (plddt * confidence_mask * iplddt_weight).sum(dim=-1) / (confidence_mask * iplddt_weight).sum(
        dim=-1
    )

    pde = _boltzgen_expected_metric(pde_logits, end=32.0)
    pae = _boltzgen_expected_metric(pae_logits, end=32.0)
    pred_distogram_probability = pred_distogram_logits.softmax(dim=-1).repeat_interleave(
        multiplicity,
        dim=0,
    )
    contact_probability = pred_distogram_probability[..., :20].sum(dim=-1)
    eye = torch.eye(token_mask.shape[-1], device=token_mask.device, dtype=torch.bool).unsqueeze(0)
    token_pair_mask = token_mask.bool().unsqueeze(-1) & token_mask.bool().unsqueeze(-2) & ~eye

    selected_ligand_mask = is_ligand_token.bool() & (chain_index == 1)
    protein_pair_mask = token_pair_mask & is_protein_token.bool().unsqueeze(-1) & is_protein_token.bool().unsqueeze(-2)
    ligand_pair_mask = token_pair_mask & selected_ligand_mask.unsqueeze(-1) & selected_ligand_mask.unsqueeze(-2)
    interface_protein_mask = (
        (representative_distances < 8) & is_protein_token.bool().unsqueeze(-1) & selected_ligand_mask.unsqueeze(-2)
    ).any(dim=-1)
    protein_ligand_interface_pair_mask = (
        token_pair_mask
        & interface_protein_mask.unsqueeze(-1)
        & selected_ligand_mask.unsqueeze(-2)
        & (representative_distances < 8)
    )

    def pae_average(mask: Tensor, *, contact_weighted: bool) -> Tensor:
        weight = mask.to(pae)
        if contact_weighted:
            weight = weight * contact_probability
        return (pae * weight).sum(dim=(1, 2)) / (weight.sum(dim=(1, 2)) + 1e-5)

    protein_pae_value_weighted = pae_average(protein_pair_mask, contact_weighted=True)
    protein_pae_value_unweighted = pae_average(protein_pair_mask, contact_weighted=False)
    ligand_pae_value_weighted = pae_average(ligand_pair_mask, contact_weighted=True)
    ligand_pae_value_unweighted = pae_average(ligand_pair_mask, contact_weighted=False)
    interface_pae_value_weighted = pae_average(protein_ligand_interface_pair_mask, contact_weighted=True)
    interface_pae_value_unweighted = pae_average(protein_ligand_interface_pair_mask, contact_weighted=False)

    contact_pair_weight = token_pair_mask.to(pde) * contact_probability
    complex_pde = (pde * contact_pair_weight).sum(dim=(1, 2)) / contact_pair_weight.sum(dim=(1, 2))
    interface_pair_weight = contact_pair_weight * (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(pde)
    complex_ipde = (pde * interface_pair_weight).sum(dim=(1, 2)) / (interface_pair_weight.sum(dim=(1, 2)) + 1e-5)

    design_mask = cast(Tensor, input.design_mask).bool()
    chain_design_mask = cast(Tensor, input.chain_design_mask).bool()
    has_design = design_mask.any(dim=-1, keepdim=True)
    pae_chain_design_mask = torch.where(has_design, chain_design_mask, design_mask)
    pae_target_mask = torch.where(has_design, ~chain_design_mask, design_mask)
    pae_chain_design_mask = pae_chain_design_mask.repeat_interleave(multiplicity, dim=0)
    pae_target_mask = pae_target_mask.repeat_interleave(multiplicity, dim=0)
    design_mask_repeated = design_mask.repeat_interleave(multiplicity, dim=0)
    has_design_repeated = has_design.repeat_interleave(multiplicity, dim=0).squeeze(-1)

    chain_design_target_mask = token_pair_mask & (
        (pae_chain_design_mask.unsqueeze(-1) & pae_target_mask.unsqueeze(-2))
        | (pae_target_mask.unsqueeze(-1) & pae_chain_design_mask.unsqueeze(-2))
    )
    design_target_mask = token_pair_mask & (
        (design_mask_repeated.unsqueeze(-1) & pae_target_mask.unsqueeze(-2))
        | (pae_target_mask.unsqueeze(-1) & design_mask_repeated.unsqueeze(-2))
    )
    interaction_pae = (pae * design_target_mask).sum(dim=(1, 2)) / (design_target_mask.sum(dim=(1, 2)) + 1e-5)
    min_interaction_pae = pae.masked_fill(~chain_design_target_mask, 100000.0).amin(dim=(1, 2))
    min_design_to_target_pae = pae.masked_fill(~design_target_mask, 100000.0).amin(dim=(1, 2))
    min_design_to_target_pae = torch.where(
        has_design_repeated,
        min_design_to_target_pae,
        min_design_to_target_pae.new_full(min_design_to_target_pae.shape, torch.nan),
    )

    tm_scores = _boltzgen_ptm_scores(
        pae_logits,
        coordinates=coordinates,
        input=input,
        multiplicity=multiplicity,
        chain_design_mask=chain_design_mask,
    )
    ranking_tm = torch.where(
        torch.isclose(tm_scores["iptm"], torch.zeros_like(tm_scores["iptm"])).all(),
        tm_scores["ptm"],
        tm_scores["iptm"],
    )
    confidence_score = (4 * (complex_plddt / 100.0) + ranking_tm) / 5

    def unflatten(value: Tensor) -> Tensor:
        return value.reshape(batch_size, multiplicity, *value.shape[1:])

    return BoltzGenConfidenceOutput(
        pae_logits=unflatten(pae_logits),
        pde_logits=unflatten(pde_logits),
        plddt_logits=unflatten(plddt_logits),
        resolved_logits=unflatten(resolved_logits),
        pae=unflatten(pae),
        pde=unflatten(pde),
        plddt=unflatten(plddt),
        complex_plddt=unflatten(complex_plddt),
        complex_iplddt=unflatten(complex_iplddt),
        confidence_score=unflatten(confidence_score),
        complex_pde=unflatten(complex_pde),
        complex_ipde=unflatten(complex_ipde),
        interaction_pae=unflatten(interaction_pae),
        min_design_to_target_pae=unflatten(min_design_to_target_pae),
        min_interaction_pae=unflatten(min_interaction_pae),
        protein_pae_value_weighted=unflatten(protein_pae_value_weighted),
        ligand_pae_value_weighted=unflatten(ligand_pae_value_weighted),
        interface_pae_value_weighted=unflatten(interface_pae_value_weighted),
        protein_pae_value_unweighted=unflatten(protein_pae_value_unweighted),
        ligand_pae_value_unweighted=unflatten(ligand_pae_value_unweighted),
        interface_pae_value_unweighted=unflatten(interface_pae_value_unweighted),
        ptm=unflatten(tm_scores["ptm"]),
        iptm=unflatten(tm_scores["iptm"]),
        ligand_iptm=unflatten(tm_scores["ligand_iptm"]),
        protein_iptm=unflatten(tm_scores["protein_iptm"]),
        chain_pair_iptm=unflatten(tm_scores["chain_pair_iptm"]),
        design_to_target_iptm=unflatten(tm_scores["design_to_target_iptm"]),
        design_residue_iptm=unflatten(tm_scores["design_residue_iptm"]),
        design_iptm=unflatten(tm_scores["design_iptm"]),
        design_iiptm=unflatten(tm_scores["design_iiptm"]),
        target_ptm=unflatten(tm_scores["target_ptm"]),
        design_ptm=unflatten(tm_scores["design_ptm"]),
        design_ipsae_min=unflatten(tm_scores["design_ipsae_min"]),
        design_to_target_ipsae=unflatten(tm_scores["design_to_target_ipsae"]),
        target_to_design_ipsae=unflatten(tm_scores["target_to_design_ipsae"]),
        chain_pair_ipsae=unflatten(tm_scores["chain_pair_ipsae"]),
        **({"confidence_single": unflatten(confidence_single)} if confidence_single is not None else {}),
        **({"confidence_pair": unflatten(confidence_pair)} if confidence_pair is not None else {}),
        batch_size=[batch_size],
    )


def _boltzgen_expected_metric(logits: Tensor, *, end: float = 1.0) -> Tensor:
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    centers = torch.arange(
        start=0.5 * bin_width,
        end=end,
        step=bin_width,
        device=logits.device,
        dtype=logits.dtype,
    )
    return (logits.softmax(dim=-1) * centers).sum(dim=-1)


def _boltzgen_ptm_scores(
    pae_logits: Tensor,
    *,
    coordinates: Tensor,
    input: BoltzGenInput,
    multiplicity: int,
    chain_design_mask: Tensor,
) -> dict[str, Tensor]:
    r"""
    Compute the released pTM, interface, design, chain-pair, and ipSAE values.
    """

    token_mask = input.token_mask.repeat_interleave(multiplicity, dim=0).to(pae_logits)
    frame_mask = _boltzgen_predicted_frame_mask(
        coordinates,
        input=input,
        multiplicity=multiplicity,
    ).to(pae_logits)
    chain_index = input.chain_index.repeat_interleave(multiplicity, dim=0)
    pair_mask_ptm = frame_mask.unsqueeze(-1) * token_mask.unsqueeze(-2) * token_mask.unsqueeze(-1)
    different_chain = chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)
    pair_mask_iptm = pair_mask_ptm * different_chain.to(pae_logits)

    num_bins = pae_logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_centers = torch.arange(
        start=0.5 * bin_width,
        end=32.0,
        step=bin_width,
        device=pae_logits.device,
        dtype=pae_logits.dtype,
    )
    num_residues = token_mask.sum(dim=-1, keepdim=True)
    tm_expected = tm_weight_from_pae(
        pae_logits,
        num_residues.unsqueeze(-1),
        bin_centers=pae_centers,
    )

    def aggregate(mask: Tensor) -> Tensor:
        return torch.max(
            (tm_expected * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-5),
            dim=1,
        ).values

    molecule_type = input.molecule_type.repeat_interleave(multiplicity, dim=0)
    is_ligand = (molecule_type == 3).to(pae_logits)
    is_protein = (molecule_type == 0).to(pae_logits)
    ligand_iptm_mask = pair_mask_iptm * (
        is_ligand.unsqueeze(-1) * is_protein.unsqueeze(-2) + is_protein.unsqueeze(-1) * is_ligand.unsqueeze(-2)
    )
    protein_iptm_mask = pair_mask_iptm * is_protein.unsqueeze(-1) * is_protein.unsqueeze(-2)

    design_mask = cast(Tensor, input.design_mask).bool()
    has_design = design_mask.any(dim=-1, keepdim=True)
    chain_design_mask = torch.where(has_design, chain_design_mask, design_mask)
    target_mask = ~chain_design_mask
    chain_design_mask = chain_design_mask.repeat_interleave(multiplicity, dim=0)
    target_mask = target_mask.repeat_interleave(multiplicity, dim=0)
    design_mask = design_mask.repeat_interleave(multiplicity, dim=0)

    design_iptm_mask = pair_mask_ptm * (
        (chain_design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & chain_design_mask.unsqueeze(-2))
    ).to(pae_logits)
    design_to_target_iptm_mask = pair_mask_ptm * (
        (design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & design_mask.unsqueeze(-2))
    ).to(pae_logits)
    design_residue_iptm_mask = (
        pair_mask_ptm
        * different_chain.to(pae_logits)
        * (design_mask.unsqueeze(-1).to(pae_logits) + design_mask.unsqueeze(-2).to(pae_logits)).clamp(max=1)
    )
    design_ptm_mask = pair_mask_ptm * (chain_design_mask.unsqueeze(-1) & chain_design_mask.unsqueeze(-2)).to(pae_logits)
    target_ptm_mask = pair_mask_ptm * (target_mask.unsqueeze(-1) & target_mask.unsqueeze(-2)).to(pae_logits)

    num_chains = int(input.chain_index.max().item()) + 1
    chain_pair_iptm = tm_expected.new_zeros(tm_expected.shape[0], num_chains, num_chains)
    for row in range(num_chains):
        for column in range(num_chains):
            chain_pair_mask = (
                pair_mask_ptm
                * (chain_index.unsqueeze(-1) == row).to(pae_logits)
                * (chain_index.unsqueeze(-2) == column).to(pae_logits)
            )
            chain_pair_iptm[:, row, column] = aggregate(chain_pair_mask)

    interaction_mask = _boltzgen_design_interaction_mask(
        coordinates,
        input=input,
        multiplicity=multiplicity,
        chain_design_mask=chain_design_mask,
    )
    design_iiptm_mask = pair_mask_ptm * (
        (interaction_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & interaction_mask.unsqueeze(-2))
    ).to(pae_logits)

    expected_pae = (pae_logits.softmax(dim=-1) * pae_centers).sum(dim=-1)
    design_to_target_ipsae = _boltzgen_ipsae_score(
        chain_design_mask,
        target_mask,
        expected_pae,
        frame_mask,
        token_mask,
    )
    target_to_design_ipsae = _boltzgen_ipsae_score(
        target_mask,
        chain_design_mask,
        expected_pae,
        frame_mask,
        token_mask,
    )
    chain_pair_ipsae = tm_expected.new_zeros(tm_expected.shape[0], num_chains, num_chains)
    for row in range(num_chains):
        for column in range(num_chains):
            if row != column:
                chain_pair_ipsae[:, row, column] = _boltzgen_ipsae_score(
                    chain_index == row,
                    chain_index == column,
                    expected_pae,
                    frame_mask,
                    token_mask,
                )

    return {
        "ptm": aggregate(pair_mask_ptm),
        "iptm": aggregate(pair_mask_iptm),
        "ligand_iptm": aggregate(ligand_iptm_mask),
        "protein_iptm": aggregate(protein_iptm_mask),
        "chain_pair_iptm": chain_pair_iptm,
        "design_to_target_iptm": aggregate(design_to_target_iptm_mask),
        "design_residue_iptm": aggregate(design_residue_iptm_mask),
        "design_iptm": aggregate(design_iptm_mask),
        "design_iiptm": aggregate(design_iiptm_mask),
        "target_ptm": aggregate(target_ptm_mask),
        "design_ptm": aggregate(design_ptm_mask),
        "design_ipsae_min": torch.minimum(design_to_target_ipsae, target_to_design_ipsae),
        "design_to_target_ipsae": design_to_target_ipsae,
        "target_to_design_ipsae": target_to_design_ipsae,
        "chain_pair_ipsae": chain_pair_ipsae,
    }


def _boltzgen_predicted_frame_mask(
    coordinates: Tensor,
    *,
    input: BoltzGenInput,
    multiplicity: int,
) -> Tensor:
    return _boltzgen_coordinate_frames(
        coordinates,
        input=input,
        multiplicity=multiplicity,
    )[1]


def _boltzgen_coordinate_frames(
    coordinates: Tensor,
    *,
    input: BoltzGenInput,
    multiplicity: int,
    atom_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Construct coordinate-dependent token frames and their geometry mask.

    Polymer frame indices come from the Processor. Non-polymer atom-token
    frames use the two nearest eligible atoms around each center atom, matching
    the confidence metric and PAE-target equations.
    """

    frame_index = cast(Tensor, input.frame_atom_index).repeat_interleave(multiplicity, dim=0).long().clone()
    chain_index = input.chain_index
    molecule_type = input.molecule_type
    atom_to_token = input.atom_to_token.long()
    atom_chain_index = chain_index.gather(1, atom_to_token)
    structural_atom_mask = input.atom_mask.bool()
    if atom_mask is None:
        atom_mask = structural_atom_mask.repeat_interleave(multiplicity, dim=0)
    else:
        atom_mask = atom_mask.bool()

    for batch_index in range(input.token_mask.shape[0]):
        ligand_chains = torch.unique(
            chain_index[batch_index][input.token_mask[batch_index].bool() & (molecule_type[batch_index] == 3)]
        ).tolist()
        for chain in ligand_chains:
            chain_tokens = torch.nonzero(
                input.token_mask[batch_index].bool()
                & (molecule_type[batch_index] == 3)
                & (chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            chain_atoms = torch.nonzero(
                structural_atom_mask[batch_index] & (atom_chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            if len(chain_atoms) < 3:
                continue
            for sample_index in range(multiplicity):
                row = batch_index * multiplicity + sample_index
                distances = torch.cdist(coordinates[row, chain_atoms], coordinates[row, chain_atoms])
                resolved = atom_mask[row, chain_atoms]
                resolved_pair = resolved.unsqueeze(-1) & resolved.unsqueeze(-2)
                nearest = distances.masked_fill(~resolved_pair, torch.inf).argsort(dim=-1)[:, :3]
                frame_index[row, chain_tokens] = chain_atoms[nearest][:, (1, 0, 2)]

    flat_index = frame_index.reshape(frame_index.shape[0], -1)
    frames = coordinates.gather(1, flat_index.unsqueeze(-1).expand(-1, -1, 3))
    frames = frames.reshape(*frame_index.shape, 3)
    frame_resolved = atom_mask.gather(1, flat_index).reshape(frame_index.shape).all(dim=-1)
    first = frames[..., 1, :] - frames[..., 0, :]
    second = frames[..., 1, :] - frames[..., 2, :]
    first_norm = torch.linalg.norm(first, dim=-1)
    second_norm = torch.linalg.norm(second, dim=-1)
    first_direction = first / (first_norm.unsqueeze(-1) + 1e-6)
    second_direction = second / (second_norm.unsqueeze(-1) + 1e-6)
    cosine = (first_direction * second_direction).sum(dim=-1)
    valid_geometry = (cosine.abs() < 0.9063) & (first_norm > 1e-2) & (second_norm > 1e-2)
    valid_geometry = valid_geometry & frame_resolved & input.token_mask.repeat_interleave(multiplicity, dim=0).bool()
    return frame_index, valid_geometry


def _boltzgen_design_interaction_mask(
    coordinates: Tensor,
    *,
    input: BoltzGenInput,
    multiplicity: int,
    chain_design_mask: Tensor,
) -> Tensor:
    atom_to_token = input.atom_to_token.repeat_interleave(multiplicity, dim=0).long()
    atom_mask = input.atom_mask.repeat_interleave(multiplicity, dim=0).bool()
    atom_chain_design_mask = chain_design_mask.gather(1, atom_to_token) & atom_mask
    interaction_mask = chain_design_mask.new_zeros(chain_design_mask.shape)
    for sample_index in range(coordinates.shape[0]):
        design_atoms = atom_chain_design_mask[sample_index]
        target_atoms = atom_mask[sample_index] & ~design_atoms
        if design_atoms.any() and target_atoms.any():
            interacting_atoms = (
                torch.cdist(
                    coordinates[sample_index, design_atoms],
                    coordinates[sample_index, target_atoms],
                ).amin(dim=-1)
                < 8
            )
            interacting_tokens = atom_to_token[sample_index, design_atoms][interacting_atoms]
            interaction_mask[sample_index].scatter_(
                0,
                interacting_tokens,
                torch.ones_like(interacting_tokens, dtype=torch.bool),
            )
    return interaction_mask


def _boltzgen_ipsae_score(
    source_mask: Tensor,
    target_mask: Tensor,
    pae: Tensor,
    frame_mask: Tensor,
    token_mask: Tensor,
    cutoff: float = 15.0,
) -> Tensor:
    pair_mask = (
        frame_mask.unsqueeze(-1)
        * token_mask.unsqueeze(-2)
        * token_mask.unsqueeze(-1)
        * source_mask.unsqueeze(-1).to(pae)
        * target_mask.unsqueeze(-2).to(pae)
    )
    valid_pae = (pae < cutoff).to(pae) * pair_mask
    num_valid = valid_pae.sum(dim=-1)
    d0 = 1.24 * (num_valid.clamp(min=19) - 15).pow(1.0 / 3.0) - 1.8
    d0 = d0.clamp(min=1.0)
    terms = 1.0 / (1.0 + (pae / d0.unsqueeze(-1)).square())
    per_source = (terms * valid_pae).sum(dim=-1) / (num_valid + 1e-5)
    return (per_source * source_mask.to(per_source)).max(dim=-1).values


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
    augmentation: bool = True,
    translation_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Center masked coordinates and apply a random rotation and translation.
    """

    mask = atom_mask.unsqueeze(-1).to(coords.dtype)
    centroid = (coords * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1.0)
    centered = coords - centroid
    if not augmentation:
        return centered
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
        if isinstance(child, BoltzGenDenoiserAtomEncoder):
            nn.init.zeros_(child.single_trunk_projection.weight)
            nn.init.zeros_(child.pair_trunk_projection.weight)
        if isinstance(child, AtomPairEncoding):
            nn.init.zeros_(child.left_atom_projection.weight)
            nn.init.zeros_(child.right_atom_projection.weight)
            nn.init.zeros_(child.feed_forward_output_projection.weight)
        elif isinstance(
            child,
            (PackedDenoisingAtomToTokenEncoder, BoltzGenGaussianDenoisingAtomToTokenEncoder),
        ):
            nn.init.zeros_(child.noisy_position_projection.weight)
        elif isinstance(child, PackedTokenToAtomDecoder):
            nn.init.zeros_(child.token_projection.weight)
            nn.init.zeros_(child.position_update_projection.weight)
        elif isinstance(child, _BoltzGenDenoiser):
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


# ===========================================================================
# BoltzGen design conditions, typed inputs, structure targets, and outputs.
# ===========================================================================


@dataclass(frozen=True)
class BoltzGenDesignMask:
    r"""
    Select existing residue positions for redesign.

    Attributes:
        positions: Residue occurrences whose identities may change.
    """

    positions: tuple[ComponentReference, ...]


@dataclass(frozen=True)
class BoltzGenBinding:
    r"""
    Positive and negative residue-level binding conditions.

    Attributes:
        binding: Residues encouraged to participate in binding.
        not_binding: Residues discouraged from participating in binding.
    """

    binding: Hotspot = Hotspot(())
    not_binding: tuple[ComponentReference, ...] = ()


@dataclass(frozen=True)
class BoltzGenSecondaryStructure:
    r"""
    Assign secondary-structure conditions to residue positions.

    Attributes:
        assignments: ``(residue, label)`` pairs, where the label is ``"H"``
            (helix), ``"E"`` (strand), or ``"L"`` (loop).
    """

    assignments: tuple[tuple[ComponentReference, Literal["H", "E", "L"]], ...]


@dataclass(frozen=True)
class BoltzGenStructureGroup:
    r"""
    Residues whose relative geometry remains visible during design.

    Distinct nonzero groups move independently; group zero denotes unconstrained residues.

    Attributes:
        members: Residues belonging to the group.
        group_id: Nonnegative group identifier.
    """

    members: tuple[ComponentReference, ...]
    group_id: int

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be nonnegative")


@dataclass(frozen=True)
class BoltzGenResidueConstraint:
    r"""
    Restrict the amino-acid identities allowed at one residue.

    Attributes:
        position: Residue occurrence to constrain.
        allowed: Allowed three-letter amino-acid component IDs.
    """

    position: ComponentReference
    allowed: tuple[str, ...]


@dataclass(frozen=True)
class BoltzGenContact:
    r"""
    A BoltzGen pairwise proximity condition between two endpoint sets.

    Attributes:
        side_a: Endpoints on one side of the requested contact.
        side_b: Endpoints on the other side of the requested contact.
        cutoff_angstrom: Maximum contact distance in angstroms.
        kind: Directional pocket/binder meaning across the declared sides, or
            ``"contact"`` for symmetric proximity.
    """

    side_a: tuple[Endpoint, ...]
    side_b: tuple[Endpoint, ...]
    cutoff_angstrom: float
    kind: Literal["contact", "pocket>binder", "binder>pocket"] = "contact"


@dataclass(frozen=True)
class BoltzGenSymmetry:
    r"""
    Tie corresponding positions across equal-length symmetry units.

    Attributes:
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
    | BoltzGenContact
    | SymbolicBond
    | BoltzGenSymmetry
)


@tensorclass
class BoltzGenStructureTarget:
    r"""
    Structure-training labels aligned to one [`BoltzGenInput`][].

    Attributes:
        coordinates: Complete-assembly target coordinates with shape
            ``(B, R, A_system, 3)``.
        coordinate_mask: Resolved complete-assembly mask with shape
            ``(B, A_system)``.
        coordinate_atom_index: Model-row to coordinate-source mapping with shape
            ``(B, M)`` and ``-1`` padding.
        chain_span: Half-open canonical-atom chain spans with shape ``(B, C, 2)``.
        chain_equivalence_index: Entity-and-layout equivalence ids with shape
            ``(B, C)``.
        symmetry_atom_index: Current-axis atoms in each local symmetry group
            with shape ``(B, G, W)`` and ``-1`` padding.
        symmetry_permutation: Complete carried permutations with shape
            ``(B, G, P, W)`` and ``-1`` padding.
        residue_type: Unmasked target residue distributions with shape
            ``(B, N, V)``.
        b_factor: B factor of each token's representative atom with shape ``(B, N)``,
            matching the token axis the B-factor head reads.
        b_factor_mask: Deposited-B-factor mask with shape ``(B, N)``.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    coordinate_atom_index: Tensor
    chain_span: Tensor
    chain_equivalence_index: Tensor
    symmetry_atom_index: Tensor
    symmetry_permutation: Tensor
    residue_type: Tensor
    b_factor: Tensor
    b_factor_mask: Tensor


@tensorclass
class BoltzGenInput:
    r"""
    Input tensors consumed by BoltzGen design models.

    Attributes:
        system: Canonical molecular graph shared by all tensor rows.
        atom_index: Model-row to canonical-atom identity mapping. Synthetic and
            padding rows contain ``-1``.
        token_component_index: Token-row to canonical-component mapping.
        token_atom_index: Atom-token to canonical-atom mapping; component tokens
            contain ``-1``.
        bond_atom_index: Inter-component model-row bond endpoints.
        bond_mask: Valid inter-component bond mask.
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
        atom_token_index: Per-token atom offset for each atom.
        token_center_atom_index: Center atom index for each token.
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
        frame_atom_index: Optional frame atom indices for frame-aligned losses.
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
        chain_design_mask: Optional token mask selecting the complete covalently
            connected design component. Confidence inputs must provide it.
        ss_type: Optional secondary-structure conditioning ids.
    """

    system: MolecularSystem
    atom_index: Tensor
    token_component_index: Tensor
    token_atom_index: Tensor
    bond_atom_index: Tensor
    bond_mask: Tensor
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
    atom_token_index: Tensor
    token_center_atom_index: Tensor
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
    frame_atom_index: Tensor | None = None
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
    chain_design_mask: Tensor | None = None
    ss_type: Tensor | None = None


@tensorclass
class BoltzGenInverseFoldingInput:
    r"""
    Tensor and canonical-graph input for BoltzGen inverse folding.

    ``B`` is batch size, ``N`` token count, and ``A`` atom count.

    Attributes:
        system: Canonical molecular graph shared by all tensor rows.
        atom_index: Model-row to canonical-atom identity mapping.
        token_component_index: Token-row to canonical-component mapping.
        token_atom_index: Atom-token to canonical-atom mapping; component tokens
            contain ``-1``.
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
        token_resolved_mask: Mask selecting tokens whose representative atom is
            resolved in the fixed structure.
        method_feature: Optional experimental-method features.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        design_mask: Mask selecting positions to redesign.
        aa_constraint_mask: Amino-acid block mask with shape ``(B, N, 20)``.
        symmetric_group: Sequence-symmetry group IDs with shape ``(B, N)``;
            zero means untied.
    """

    system: MolecularSystem
    atom_index: Tensor
    token_component_index: Tensor
    token_atom_index: Tensor
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
    token_resolved_mask: Tensor
    design_mask: Tensor
    aa_constraint_mask: Tensor
    symmetric_group: Tensor
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None


class _BoltzGenTrunkOutput(NamedTuple):
    r"""
    Trunk representations consumed by the diffusion and confidence heads.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    relative_position_encoding: Tensor
    distogram_logits: Tensor


class BoltzGenOutput(_TensorCollection):
    r"""
    Predictions from one BoltzGen training denoising pass.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        bfactor_logits: Per-token B-factor logits with shape ``(B, N, D_b)``.
        augmented_target_coordinates: Training target coordinates after
            rigid augmentation, with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Sampled diffusion noise levels with shape ``(B, S)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor


class BoltzGenResidueTypeOutput(BoltzGenOutput):
    r"""
    BoltzGen training predictions with required residue-type logits.

    Attributes:
        residue_type_logits: Residue-type logits with shape ``(B, S, N, V)``.
    """

    residue_type_logits: Tensor


class BoltzGenConfidenceOutput(_TensorCollection):
    r"""
    Complete raw and derived outputs from the BoltzGen confidence equation.

    Leading dimensions are ``(B, S)``. ``plddt_logits``, ``resolved_logits``,
    and ``plddt`` use the token axis when token-level confidence is configured
    and the realized atom axis for the released 23-slot atom projection.
    Chain-pair tensors use dense zero-based input chain indices.
    """

    pae_logits: Tensor
    pde_logits: Tensor
    plddt_logits: Tensor
    resolved_logits: Tensor
    pae: Tensor
    pde: Tensor
    plddt: Tensor
    complex_plddt: Tensor
    complex_iplddt: Tensor
    confidence_score: Tensor
    complex_pde: Tensor
    complex_ipde: Tensor
    interaction_pae: Tensor
    min_design_to_target_pae: Tensor
    min_interaction_pae: Tensor
    protein_pae_value_weighted: Tensor
    ligand_pae_value_weighted: Tensor
    interface_pae_value_weighted: Tensor
    protein_pae_value_unweighted: Tensor
    ligand_pae_value_unweighted: Tensor
    interface_pae_value_unweighted: Tensor
    ptm: Tensor
    iptm: Tensor
    ligand_iptm: Tensor
    protein_iptm: Tensor
    chain_pair_iptm: Tensor
    design_to_target_iptm: Tensor
    design_residue_iptm: Tensor
    design_iptm: Tensor
    design_iiptm: Tensor
    target_ptm: Tensor
    design_ptm: Tensor
    design_ipsae_min: Tensor
    design_to_target_ipsae: Tensor
    target_to_design_ipsae: Tensor
    chain_pair_ipsae: Tensor
    confidence_single: _OptionalTensor
    confidence_pair: _OptionalTensor


class BoltzGenSampleOutput(_TensorCollection):
    r"""
    Predictions from one BoltzGen reverse-diffusion sampling pass.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        bfactor_logits: Per-token B-factor logits with shape ``(B, N, D_b)``.
        coordinate_trajectory: Initial/current coordinate states with shape
            ``(B, S, K + 1, A, 3)``, when requested.
        denoised_coordinate_trajectory: Per-step denoised ``x0`` predictions
            with shape ``(B, S, K, A, 3)``, when requested.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor
    coordinate_trajectory: _OptionalTensor
    denoised_coordinate_trajectory: _OptionalTensor


class BoltzGenResidueTypeSampleOutput(BoltzGenSampleOutput):
    r"""Reverse-diffusion samples with required final residue-type logits."""

    residue_type_logits: Tensor


class BoltzGenSampleWithConfidenceOutput(_TensorCollection):
    r"""
    Coordinate samples and confidence values produced from one trunk execution.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor
    confidence: BoltzGenConfidenceOutput
    coordinate_trajectory: _OptionalTensor
    denoised_coordinate_trajectory: _OptionalTensor


class BoltzGenInverseFoldingOutput(_TensorCollection):
    r"""
    Logit predictions from one BoltzGen inverse-folding forward pass.

    Attributes:
        logits: Per-token amino-acid logits with shape ``(B, N, V)``.
    """

    logits: Tensor


class BoltzGenInverseFoldingDesignOutput(_TensorCollection):
    r"""
    Predictions from one BoltzGen inverse-folding design pass.

    Attributes:
        logits: Per-token amino-acid logits with shape ``(B, N, V)``.
        tokens: Sampled amino-acid token IDs with shape ``(B, N)``.
    """

    logits: Tensor
    tokens: Tensor
