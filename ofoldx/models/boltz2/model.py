r"""
Boltz-2 model.

The file owns the Boltz-2 input contract, trunk wiring, diffusion path, and confidence heads. Shared
OFoldX modules provide generic neural-network blocks; model-specific choices stay here so the public API has one
place to inspect the model.

The structure and affinity model variants expose typed low-level tensor inputs. The
two-stage affinity operation is assembled by ``Boltz2AffinityPipeline``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import partial
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.molecular import AtomSite, MolecularSystem
from ...data.structure import Structure
from ...model import Model
from ...modules.atom.encoders import PackedAtomAttention
from ...modules.atom.pair_encoding import encode_atom_pair
from ...modules.atom.transfer import dense_mean_to_tokens
from ...modules.atom.window import gather_key_windows
from ...modules.backend import LayerNorm
from ...modules.diffusion.conditioning import PairwiseConditioning, condition_single
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import DistogramHead
from ...modules.head.tm import tm_weight_from_pae
from ...modules.transformer import ConditionedTransformer
from ...modules.transition import Transition
from ...modules.trunk.msa import (
    CategoricalMsaTrunk,
    TrianglePairUpdateBlock,
)
from ...modules.trunk.pairformer import PairformerStack
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, ScoreResult
from .config import Boltz2AffinityConfig, Boltz2AffinityHeadConfig, Boltz2Config


class Boltz2Model(Model):
    r"""
    Boltz-2-shaped folding model composed from shared OFoldX modules.

    The model runs the standard Boltz-2 pipeline: an atom-conditioned input embedder builds the initial
    single/pair representations, a recycled trunk (template + MSA + Pairformer) refines them, an EDM diffusion
    sampler draws atom coords, and a token-level confidence head scores the sample. Subsystems are assembled
    from shared ``ofoldx.modules`` blocks so the public API has one place to inspect the wiring.

    Args:
        config: Boltz-2 model config selecting widths, block counts, and architecture flags.

    """

    config_class = Boltz2Config
    config: Boltz2Config

    def __init__(self, config: Boltz2Config | None = None) -> None:
        config = Boltz2Config() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        single_size, pair_size = config.single_size, config.pair_size
        input_embeddings_config = config.input_embeddings
        trunk = config.trunk

        self.input_embeddings = Boltz2InputEmbeddings(
            single_size=single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
            num_atom_encoder_blocks=input_embeddings_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=input_embeddings_config.num_atom_encoder_attention_heads,
            vocab_size=input_embeddings_config.vocab_size,
            msa_profile_size=input_embeddings_config.msa_profile_size,
            num_method_types=input_embeddings_config.num_method_types,
            num_molecule_types=input_embeddings_config.num_molecule_types,
            add_method_conditioning=input_embeddings_config.add_method_conditioning,
            add_modified_flag=input_embeddings_config.add_modified_flag,
            add_cyclic_flag=input_embeddings_config.add_cyclic_flag,
            add_mol_type_feat=input_embeddings_config.add_mol_type_feat,
            operator_backend="auto",
        )

        # s/z initialisation. z_init is built from the single INPUT embedding (s_inputs).
        self.single_init_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(config.num_bond_types, pair_size)
        self.contact_conditioning = Boltz2ContactConditioning(
            pair_size, cutoff_min=config.conditioning_cutoff_min, cutoff_max=config.conditioning_cutoff_max
        )

        # Recycling (zero-init gates).
        self.single_recycle_norm = LayerNorm(single_size)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        self.template_embedder: Boltz2TemplateEmbedder | None = None
        if config.use_templates:
            self.template_embedder = Boltz2TemplateEmbedder(
                pair_size=pair_size,
                hidden_size=config.template.hidden_size,
                num_blocks=config.template.num_blocks,
                vocab_size=config.template.vocab_size,
                num_bins=config.template.num_distogram_bins,
                min_distance=config.template.min_distance,
                max_distance=config.template.max_distance,
                expansion=config.template.expansion,
                triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
                num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                operator_backend="auto",
            )
        self.msa_trunk = CategoricalMsaTrunk(
            single_size=single_size,
            pair_size=pair_size,
            msa_size=trunk.msa_size,
            msa_feature_size=trunk.msa_feature_size,
            num_blocks=trunk.num_msa_blocks,
            vocab_size=input_embeddings_config.vocab_size,
            use_paired_feature=trunk.use_paired_feature,
            outer_product_hidden_size=trunk.outer_product_hidden_size,
            msa_dropout=trunk.msa_dropout,
            pair_dropout=trunk.dropout,
            num_msa_attention_heads=trunk.num_msa_attention_heads,
            msa_head_dim=trunk.msa_head_dim,
            triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
            num_triangle_attention_heads=trunk.num_triangle_attention_heads,
            operator_backend="auto",
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=trunk.num_pairformer_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
            num_triangle_attention_heads=trunk.num_triangle_attention_heads,
            num_pair_attention_heads=trunk.num_pair_attention_heads,
            dropout=trunk.dropout,
            operator_backend="auto",
        )
        self.distogram_head = DistogramHead(
            pair_size=pair_size, num_bins=config.num_distogram_bins, symmetrize_before_projection=True
        )

        denoiser_config = config.denoiser
        self.denoiser_conditioning = Boltz2DenoiserConditioning(
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
            num_token_transformer_blocks=denoiser_config.num_token_transformer_blocks,
            num_token_transformer_heads=denoiser_config.num_token_transformer_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
        )
        denoiser = Boltz2Denoiser(
            single_size=single_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            fourier_embedding_size=config.fourier_embedding_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_transformer_blocks=denoiser_config.num_token_transformer_blocks,
            num_token_transformer_heads=denoiser_config.num_token_transformer_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
            operator_backend="auto",
        )
        sampler = config.sampler
        self.coordinate_diffusion = Boltz2CoordinateDiffusion(
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
        )

        confidence = config.confidence
        self.confidence_head = Boltz2ConfidenceHead(
            single_size=single_size,
            pair_size=pair_size,
            num_blocks=confidence.num_pairformer_blocks,
            num_pair_attention_heads=confidence.num_pair_attention_heads,
            num_distogram_bins=confidence.num_distogram_bins,
            max_distance=confidence.max_distance,
            relative_position_size=config.relative_position_size,
            num_bond_types=config.num_bond_types,
            conditioning_cutoff_min=config.conditioning_cutoff_min,
            conditioning_cutoff_max=config.conditioning_cutoff_max,
            num_plddt_bins=confidence.num_plddt_bins,
            num_pae_bins=confidence.num_pae_bins,
            num_pde_bins=confidence.num_pde_bins,
            operator_backend="auto",
        )
        self.bfactor_head: Boltz2BFactorHead | None = None
        if config.predict_bfactor:
            self.bfactor_head = Boltz2BFactorHead(single_size=single_size, num_bins=config.num_distogram_bins)
        self.post_init()

    def forward(
        self,
        input: Boltz2Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Boltz2Output:
        r"""
        Training path: run the trunk, then a single EDM diffusion denoising step against ``target_positions``.

        ``forward`` is the structure-training graph in both ``train()`` and ``eval()`` modes; those modes only
        control ordinary stochastic layers such as dropout. It mirrors the Boltz-2 training diffusion step
        (``DiffusionModule.forward``): sample a per-example noise level, corrupt the (centred) target coords, and
        denoise them through the score model conditioned on the trunk. Multi-step reverse diffusion is the
        separate inference path exposed by [`fold`][].

        Args:
            input: Typed model input; training requires ``input.target_positions``.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of noised copies drawn per batch element.
            generator: Optional PyTorch random generator for the diffusion noise.

        Returns:
            Structure-prediction output carrying ``coordinates``, ``augmented_target_coordinates``, and
            ``noise_level``
            for the diffusion loss alongside the distogram logits.

        """

        representations = self._trunk(input, recycling_steps=recycling_steps)
        target_positions = cast(Tensor, input.target_positions)
        conditioning = self._build_denoiser_conditioning(representations, input)
        denoised_coordinates, sigmas, augmented_target_coordinates = self.coordinate_diffusion.denoise(
            target_positions,
            input.atom_mask,
            multiplicity=num_samples,
            generator=generator,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        batch_size = representations.single.shape[0]
        return Boltz2Output(
            coordinates=_unflatten_sample_axis(denoised_coordinates, batch_size, num_samples),
            distogram_logits=representations.distogram_logits,
            augmented_target_coordinates=_unflatten_sample_axis(augmented_target_coordinates, batch_size, num_samples),
            noise_level=sigmas,
        )

    @torch.inference_mode()
    def fold(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""Run Boltz-2 folding and decode a self-contained canonical ``FoldResult``.

        Runs the recycling trunk, draws ``num_samples`` reverse-diffusion samples, scores each with the
        confidence head, and wraps the raw
        [`Boltz2Output`][ofoldx.models.boltz2.model.Boltz2Output] into a
        [`FoldResult`][ofoldx.results.FoldResult]: one generation-ordered
        [`FoldPrediction`][ofoldx.results.FoldPrediction] per sample, carrying the sampled ``Structure``,
        per-token ``metrics`` (``plddt``, ``resolved_probability``), token-pair ``pair_metrics``
        (``pae``, ``pde``), and per-sample ``scores`` (``plddt``, ``ptm``, ``iptm``, ``ligand_iptm``,
        ``protein_iptm``, ``confidence_score``). It mirrors
        [`Boltz1Model.fold`][ofoldx.models.boltz1.model.Boltz1Model.fold].

        Args:
            input: One lowered Boltz-2 input carrying the canonical
                [`Boltz2Layout`][ofoldx.models.boltz2.model.Boltz2Layout] produced by ``Boltz2Processor``.
            num_samples: Number of reverse-diffusion samples to draw.
            generator: Optional RNG for reproducible sampling noise; resolved once at the Pipeline boundary.
            recycling_steps: Total number of trunk passes; defaults to the configured value.
            num_sampling_steps: Number of reverse-diffusion steps; defaults to the sampler config.

        Returns:
            A [`FoldResult`][ofoldx.results.FoldResult] with one prediction per requested sample.

        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if input.token_mask.shape[0] != 1:
            raise ValueError("Boltz2Model.fold accepts one molecular system per call")
        if input.layout is None:
            raise ValueError("Boltz2Model.fold requires the canonical layout produced by its Processor")

        output = self._fold(
            input,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_samples,
            generator=generator,
        )
        return _decode_boltz2_fold_result(output, input)

    def _fold(
        self,
        input: Boltz2Input,
        *,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Boltz2Output:
        r"""
        Inference verb: fold (trunk) -> sample coords (diffusion) -> score (confidence).

        Runs the recycled trunk, the multi-step EDM reverse-diffusion sampler, and the confidence head. This is
        the distinct *inference* path (the training path is
        [`forward`][ofoldx.models.boltz2.model.Boltz2Model.forward]); the returned raw
        [`Boltz2Output`][ofoldx.models.boltz2.model.Boltz2Output] is an internal carrier; the public
        [`fold`][] method decodes it into ``FoldResult``.

        Args:
            input: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            num_samples: Number of reverse-diffusion samples to draw.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output with sampled coords, distogram logits, and confidence logits.

        """

        representations = self._trunk(input, recycling_steps=recycling_steps)
        single = representations.single
        pair = representations.pair
        sample_coords_flat = self._sample_coords(
            representations,
            input,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_samples,
            generator=generator,
        )
        confidence = self.confidence_head(
            representations.single_inputs,
            single,
            pair,
            sample_coords_flat,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            token_bonds=input.token_bonds,
            type_bonds=input.type_bonds,
            contact_conditioning=input.contact_conditioning,
            contact_threshold=input.contact_threshold,
            token_to_rep_atom=input.token_to_rep_atom,
            token_mask=input.token_mask,
            cyclic_period=input.cyclic_period,
            multiplicity=num_samples,
        )
        bfactor = self.bfactor_head(single) if self.bfactor_head is not None else None
        batch_size = single.shape[0]
        return Boltz2Output(
            coordinates=_unflatten_sample_axis(sample_coords_flat, batch_size, num_samples),
            distogram_logits=representations.distogram_logits,
            plddt_logits=_unflatten_sample_axis(confidence["plddt_logits"], batch_size, num_samples),
            pae_logits=_unflatten_sample_axis(confidence["pae_logits"], batch_size, num_samples),
            pde_logits=_unflatten_sample_axis(confidence["pde_logits"], batch_size, num_samples),
            resolved_logits=_unflatten_sample_axis(confidence["resolved_logits"], batch_size, num_samples),
            bfactor_logits=(
                None
                if bfactor is None
                else bfactor.unsqueeze(1).expand(bfactor.shape[0], num_samples, *bfactor.shape[1:])
            ),
        )

    def _trunk(self, input: Boltz2Input, *, recycling_steps: int | None = None) -> _Boltz2TrunkState:
        r"""
        Run the recycled folding trunk (template + MSA + Pairformer) and emit the trunk representations.

        Args:
            input: Typed model input.
            recycling_steps: Total number of trunk passes; defaults to the configured value. The trunk runs
                ``recycling_steps`` times.

        Returns:
            Trunk representations and distogram logits.

        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        single_inputs, single_init, pair_init, relative_position_encoding = self._initial_representations(input)
        token_mask = input.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask)

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        # Boltz runs recycling_steps trunk passes.
        for cycle in range(recycling_steps):
            grad_enabled = torch.is_grad_enabled() and cycle == recycling_steps - 1
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if input.template_residue_type is not None:
                    pair = pair + cast(Boltz2TemplateEmbedder, self.template_embedder)(
                        pair,
                        residue_type=input.template_residue_type,
                        frame_rotations=cast(Tensor, input.template_frame_rotations),
                        frame_translations=cast(Tensor, input.template_frame_translations),
                        frame_mask=cast(Tensor, input.template_frame_mask),
                        beta_carbon_positions=cast(
                            Tensor,
                            input.template_beta_carbon_positions,
                        ),
                        alpha_carbon_positions=cast(
                            Tensor,
                            input.template_alpha_carbon_positions,
                        ),
                        beta_carbon_mask=cast(Tensor, input.template_beta_carbon_mask),
                        visibility_index=cast(Tensor, input.template_visibility_index),
                        template_mask=cast(Tensor, input.template_mask),
                        pair_mask=pair_mask,
                    )
                if input.msa is not None:
                    # Boltz adds the MSA module's (input-inclusive) refined pair as a residual.
                    pair = pair + self.msa_trunk(
                        pair,
                        single_inputs,
                        msa=input.msa,
                        has_deletion=cast(Tensor, input.has_deletion),
                        deletion_value=cast(Tensor, input.deletion_value),
                        is_paired=cast(Tensor, input.msa_paired),
                        msa_mask=cast(Tensor, input.msa_mask),
                        token_mask=token_mask,
                    )
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        return _Boltz2TrunkState(
            single_inputs=single_inputs,
            single=single,
            pair=pair,
            distogram_logits=self.distogram_head(pair),
            relative_position_encoding=relative_position_encoding,
        )

    def _initial_representations(self, input: Boltz2Input) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build the recycling-invariant input representations from the batch.

        Args:
            input: Typed model input.

        Returns:
            Tuple ``(single_inputs, single_init, pair_init, relative_position_encoding)`` where ``single_inputs``
            and ``single_init`` are ``(B, N, single_size)``, and ``pair_init`` and
            ``relative_position_encoding`` are ``(B, N, N, pair_size)``.
        """

        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type=input.residue_type,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
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
        pair_init = pair_init + self.token_bond_projection(input.token_bonds.unsqueeze(-1).to(pair_init.dtype))
        pair_init = pair_init + self.token_bond_type_embedding(input.type_bonds.long())
        pair_init = pair_init + self.contact_conditioning(
            input.contact_conditioning.to(pair_init.dtype), input.contact_threshold.to(pair_init.dtype)
        )
        return single_inputs, single_init, pair_init, relative_position_encoding

    def _sample_coords(
        self,
        representations: _Boltz2TrunkState,
        input: Boltz2Input,
        *,
        num_sampling_steps: int | None,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Diffusion-sample atom coords from the trunk representations.

        Args:
            representations: Trunk representations from
                [`_trunk`][ofoldx.models.boltz2.model.Boltz2Model._trunk].
            input: Typed model input.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            num_samples: Number of independent samples to draw per batch element.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Flattened sampled atom coords ``(B*num_samples, M, 3)``.
        """

        conditioning = self._build_denoiser_conditioning(representations, input)
        return self.coordinate_diffusion._sample(
            input.atom_mask,
            num_sampling_steps=num_sampling_steps,
            multiplicity=num_samples,
            generator=generator,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )

    def _build_denoiser_conditioning(self, representations: _Boltz2TrunkState, input: Boltz2Input) -> dict[str, Tensor]:
        r"""
        Precompute the per-call-invariant diffusion conditioning from the trunk representations.

        Shared by the training denoising step ([`forward`][ofoldx.models.boltz2.model.Boltz2Model.forward]) and
        the reverse-diffusion sampler ([`_sample_coords`][ofoldx.models.boltz2.model.Boltz2Model._sample_coords]).

        Args:
            representations: Trunk representations from
                [`_trunk`][ofoldx.models.boltz2.model.Boltz2Model._trunk].
            input: Typed model input.

        Returns:
            The [`Boltz2DenoiserConditioning`][ofoldx.models.boltz2.model.Boltz2DenoiserConditioning] output dict.
        """

        return self.denoiser_conditioning(
            representations.single,
            representations.pair,
            representations.relative_position_encoding,
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
        )


class Boltz2AffinityModel(Boltz2Model):
    r"""
    Extend the Boltz-2 folder with a pair-only head that scores protein-ligand binding affinity.

    The model reuses the Boltz-2 trunk and diffusion path and adds a two-head pair-only affinity ensemble.
    ``score`` returns a self-contained canonical result for one ``Boltz2Input`` produced by
    ``Boltz2AffinityProcessor``.

    Args:
        config: Boltz-2 affinity model config for the two-head architecture.

    """

    config_class = Boltz2AffinityConfig
    config: Boltz2AffinityConfig

    def __init__(self, config: Boltz2AffinityConfig | None = None) -> None:
        config = Boltz2AffinityConfig() if config is None else config
        super().__init__(config)
        self.affinity_heads = nn.ModuleList(self._build_affinity_head(head) for head in config.affinity_heads)
        self.post_init(backend=self.backend)

    def _build_affinity_head(self, affinity: Boltz2AffinityHeadConfig) -> Boltz2AffinityHead:
        r"""
        Instantiate one affinity head from an affinity-head sub-config.

        Args:
            affinity: An affinity-head sub-config (block count, distogram bins, triangle widths, etc.).

        Returns:
            A configured [`Boltz2AffinityHead`][ofoldx.models.boltz2.model.Boltz2AffinityHead].
        """

        return Boltz2AffinityHead(
            single_size=self.config.single_size,
            pair_size=self.config.pair_size,
            num_blocks=affinity.num_pairformer_blocks,
            num_distogram_bins=affinity.num_distogram_bins,
            max_distance=affinity.max_distance,
            num_conditioning_transitions=affinity.num_conditioning_transitions,
            expansion=affinity.expansion,
            triangle_attention_hidden_size=affinity.triangle_attention_hidden_size,
            num_triangle_attention_heads=affinity.num_triangle_attention_heads,
            dropout=affinity.dropout,
            operator_backend="auto",
        )

    @torch.inference_mode()
    def score(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 5,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        molecular_weight_correction: bool = False,
    ) -> ScoreResult:
        r"""Run one Boltz-2 affinity scoring pass and return canonical scores.

        This low-level call expects one second-pass batch from
        ``Boltz2AffinityProcessor``. It samples candidate complex structures,
        ranks them by iPTM, and applies the affinity ensemble only to the
        highest-iPTM sample. The affinity value is an IC50-like
        ``log10(measurement / µM)`` score (lower is stronger); the second output
        is a binder probability in ``[0, 1]``. This method does not perform a separate fold-and-rank pass.
        """

        num_samples = operator.index(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if recycling_steps is not None:
            recycling_steps = operator.index(recycling_steps)
            if recycling_steps < 1:
                raise ValueError("recycling_steps must be positive")
        if num_sampling_steps is not None:
            num_sampling_steps = operator.index(num_sampling_steps)
            if num_sampling_steps <= 0:
                raise ValueError("num_sampling_steps must be positive")
        if input.token_mask.shape[0] != 1:
            raise ValueError("Boltz2AffinityModel.score accepts one molecular system per call")
        if input.layout is None:
            raise ValueError("Boltz2AffinityModel.score requires the canonical layout produced by its Processor")
        if input.layout.affinity_instance_id is None:
            raise ValueError("Boltz2AffinityModel.score requires one canonical affinity ligand instance")

        structure_output, raw_scores = self._score(
            input,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )
        selected_index = int(raw_scores["selected_sample_index"][0].item())
        selected_coords = structure_output.coordinates[0, selected_index]
        layout = input.layout
        complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
        model_coords = selected_coords[: len(layout.atom_sites)].float().cpu().numpy()
        for site, values in zip(layout.atom_sites, model_coords):
            index = complete_index[site]
            coordinates[index] = values
            coordinate_mask[index] = True
        structure = Structure(
            system=layout.system,
            sites=tuple(complete_sites),
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
        )

        def scalar(name: str) -> float:
            return float(raw_scores[name][0, 0, 0].item())

        # Return the six affinity scores, each scoped to the selected ligand instance. The
        # ensemble ``affinity_pred_value`` carries the optional molecular-weight correction; the two per-head
        # values are uncorrected. The molecular-weight input echoes (ligand
        # molecular weight, molecular_weight_correction), recycling/sampling run-controls, the affinity ligand
        # atom mask (reconstructable from the affinity ligand scope), and the run provenance / RNG echo. The
        # first-stage per-sample confidence and its ranked selection live on the Pipeline's
        # ``structure_predictions``; this root pass returns only the second-pass ``target`` and the six Scores.
        ligand_id = cast(Boltz2Layout, input.layout).affinity_instance_id
        scope = frozenset({cast(str, ligand_id)})
        return ScoreResult(
            target=structure,
            scores={
                "affinity_pred_value": Score(
                    scalar("affinity_pred_value"),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_probability_binary": Score(
                    scalar("affinity_probability_binary"),
                    preferred="higher",
                    scope=scope,
                ),
                "affinity_pred_value1": Score(
                    scalar("affinity_pred_value1"),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_pred_value2": Score(
                    scalar("affinity_pred_value2"),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_probability_binary1": Score(
                    scalar("affinity_probability_binary1"),
                    preferred="higher",
                    scope=scope,
                ),
                "affinity_probability_binary2": Score(
                    scalar("affinity_probability_binary2"),
                    preferred="higher",
                    scope=scope,
                ),
            },
        )

    @torch.inference_mode()
    def _score(
        self,
        input: Boltz2Input,
        *,
        coordinates: Tensor | None = None,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        molecular_weight_correction: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Boltz2Output, dict[str, Tensor]]:
        r"""
        Raw affinity protocol, optionally using caller-supplied candidate coords.

        Args:
            input: Typed Boltz-2 input. ``affinity_token_mask`` may select the ligand; otherwise ligand tokens are
                inferred from ``molecule_type == 3``.
            coordinates: Optional atom coordinates to score with shape ``(B, S, M, 3)``.
            recycling_steps: Optional trunk recycling passes. Defaults to the config value.
            num_samples: Number of coordinate sets to sample when ``coordinates`` is absent.
            num_sampling_steps: Optional EDM sampling steps for internally sampled coords.
            molecular_weight_correction: Whether to apply the ligand-molecular-weight calibration.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            All sampled structures and the affinity outputs for the best-iPTM sample.

        """

        representations = self._trunk(input, recycling_steps=recycling_steps)
        batch_size = representations.single.shape[0]
        if coordinates is None:
            sample_coords_flat = self._sample_coords(
                representations,
                input,
                num_sampling_steps=num_sampling_steps,
                num_samples=num_samples,
                generator=generator,
            )
            coordinates = _unflatten_sample_axis(sample_coords_flat, batch_size, num_samples)
        else:
            num_samples = coordinates.shape[1]
            sample_coords_flat = coordinates.flatten(0, 1)

        confidence = self.confidence_head(
            representations.single_inputs,
            representations.single,
            representations.pair,
            sample_coords_flat,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            token_bonds=input.token_bonds,
            type_bonds=input.type_bonds,
            contact_conditioning=input.contact_conditioning,
            contact_threshold=input.contact_threshold,
            token_to_rep_atom=input.token_to_rep_atom,
            token_mask=input.token_mask,
            cyclic_period=input.cyclic_period,
            multiplicity=num_samples,
        )
        iptm = _boltz2_ptm_scores(
            confidence["pae_logits"],
            coordinates=sample_coords_flat,
            input=input,
            multiplicity=num_samples,
        )["iptm"].reshape(batch_size, num_samples)
        selected_sample_index = iptm.argmax(dim=1)
        selected_coords = coordinates[torch.arange(batch_size, device=coordinates.device), selected_sample_index]
        scores = self._score_affinity(
            representations,
            input,
            selected_coords.unsqueeze(1),
            molecular_weight_correction=molecular_weight_correction,
        )
        scores["iptm_by_sample"] = iptm
        scores["selected_sample_index"] = selected_sample_index
        structure_prediction = Boltz2Output(
            coordinates=coordinates,
            distogram_logits=representations.distogram_logits,
            plddt_logits=_unflatten_sample_axis(confidence["plddt_logits"], batch_size, num_samples),
            pae_logits=_unflatten_sample_axis(confidence["pae_logits"], batch_size, num_samples),
            pde_logits=_unflatten_sample_axis(confidence["pde_logits"], batch_size, num_samples),
            resolved_logits=_unflatten_sample_axis(confidence["resolved_logits"], batch_size, num_samples),
        )
        return structure_prediction, scores

    def _score_affinity(
        self,
        representations: _Boltz2TrunkState,
        input: Boltz2Input,
        coordinates: Tensor,
        *,
        affinity_token_mask: Tensor | None = None,
        affinity_mw: Tensor | None = None,
        molecular_weight_correction: bool = False,
    ) -> dict[str, Tensor]:
        r"""
        Score coords using precomputed Boltz-2 trunk representations.

        Rebuilds the affinity-specific input single, pre-masks the trunk pair to
        receptor/ligand and ligand/ligand pairs, evaluates both heads,
        and optionally applies the molecular-weight calibration.

        Args:
            representations: Trunk representations from
                [`_trunk`][ofoldx.models.boltz2.model.Boltz2Model._trunk].
            input: Typed Boltz-2 input.
            coordinates: Atom coordinates to score with shape ``(B, S, M, 3)``.
            affinity_token_mask: Optional explicit ligand token mask ``(B, N)``.
            affinity_mw: Optional per-batch molecular weight used by the molecular-weight correction.
            molecular_weight_correction: Whether to apply the ligand-molecular-weight calibration.

        Returns:
            Dict of affinity score tensors keyed by name, each with an explicit sample axis.
        """

        batch_size, multiplicity = coordinates.shape[:2]
        coords = coordinates.flatten(0, 1)
        if affinity_token_mask is None:
            affinity_token_mask = input.affinity_token_mask
        if affinity_token_mask is None:
            raise ValueError("Boltz-2 affinity scoring requires one explicit ligand token mask")
        if input.affinity_msa_profile is None or input.affinity_deletion_mean is None:
            raise ValueError("Boltz-2 affinity scoring requires its single-sequence affinity profile")
        ligand_mask = affinity_token_mask.bool() & input.token_mask.bool()
        receptor_mask = input.molecule_type == 0
        receptor_mask = receptor_mask & input.token_mask.bool()
        affinity_pair_mask = (
            ligand_mask.unsqueeze(-1) & receptor_mask.unsqueeze(-2)
            | receptor_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
            | ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
        )
        pair = representations.pair * affinity_pair_mask.unsqueeze(-1).to(representations.pair)
        affinity_single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type=input.residue_type,
            msa_profile=input.affinity_msa_profile,
            deletion_mean=input.affinity_deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
        )
        head_kwargs = {
            "single_inputs": affinity_single_inputs,
            "pair": pair,
            "coordinates": coords,
            "token_to_rep_atom": input.token_to_rep_atom,
            "token_mask": input.token_mask,
            "ligand_mask": ligand_mask,
            "receptor_mask": receptor_mask,
            "multiplicity": multiplicity,
        }
        flat_scores1, flat_scores2 = (head(**head_kwargs) for head in self.affinity_heads)
        scores1 = {
            name: _unflatten_sample_axis(tensor, batch_size, multiplicity) for name, tensor in flat_scores1.items()
        }
        scores2 = {
            name: _unflatten_sample_axis(tensor, batch_size, multiplicity) for name, tensor in flat_scores2.items()
        }
        probability1 = torch.sigmoid(scores1["affinity_logits_binary"])
        probability2 = torch.sigmoid(scores2["affinity_logits_binary"])
        scores = {
            "affinity_pred_value": (scores1["affinity_pred_value"] + scores2["affinity_pred_value"]) / 2,
            "affinity_probability_binary": (probability1 + probability2) / 2,
            "affinity_pred_value1": scores1["affinity_pred_value"],
            "affinity_logits_binary1": scores1["affinity_logits_binary"],
            "affinity_probability_binary1": probability1,
            "affinity_pred_value2": scores2["affinity_pred_value"],
            "affinity_logits_binary2": scores2["affinity_logits_binary"],
            "affinity_probability_binary2": probability2,
        }
        if molecular_weight_correction:
            affinity_mw = input.affinity_mw if affinity_mw is None else affinity_mw
            if affinity_mw is None:
                raise ValueError("molecular-weight correction requires the selected ligand molecular weight")
            affinity_pred_value = scores["affinity_pred_value"]
            mw = affinity_mw.reshape(-1)
            scores["affinity_pred_value"] = (
                1.03525938 * affinity_pred_value - 0.59992683 * mw.pow(0.3)[:, None, None] + 2.83288489
            )
        return scores


class Boltz2InputEmbeddings(nn.Module):
    r"""
    Compose atom-derived token states with token-level encodings and conditioning embeddings.

    Runs the windowed atom encoder + atom-attention encoder to pool atoms into a per-token single
    representation, then adds residue-type, MSA-profile, and the enabled ``add_*`` conditioning terms.

    Args:
        single_size: Width of the produced token single representation.
        atom_hidden_size: Width of the atom single channel.
        atom_pair_hidden_size: Width of the atom pair channel.
        atom_feature_size: Width of the per-atom reference feature vector.
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        vocab_size: Residue-type vocabulary size for the residue-type projection.
        msa_profile_size: Width of the MSA-profile feature.
        num_method_types: Number of experimental-method ids for method conditioning.
        num_molecule_types: Number of molecule types for molecule-type conditioning.
        add_method_conditioning: Whether to add the experimental-method embedding.
        add_modified_flag: Whether to add the modified-residue embedding.
        add_cyclic_flag: Whether to add the cyclic-position projection.
        add_mol_type_feat: Whether to add the molecule-type embedding.
        operator_backend: Selector for the fused-operator backend.
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
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_encoder = Boltz2AtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_pair_bias_projection = nn.Sequential(
            LayerNorm(atom_pair_hidden_size, operator_backend=operator_backend),
            nn.Linear(atom_pair_hidden_size, num_atom_encoder_blocks * num_atom_encoder_attention_heads, bias=False),
        )
        self.atom_attention_encoder = Boltz2AtomAttentionEncoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_encoder_blocks=num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
            operator_backend=operator_backend,
        )

        self.residue_type_projection = nn.Linear(vocab_size, single_size, bias=False)
        self.msa_profile_projection = nn.Linear(msa_profile_size, single_size, bias=False)
        self.method_embedding: nn.Embedding | None = None
        self.modified_embedding: nn.Embedding | None = None
        self.cyclic_projection: nn.Linear | None = None
        self.molecule_type_embedding: nn.Embedding | None = None
        if add_method_conditioning:
            self.method_embedding = nn.Embedding(num_method_types, single_size)
        if add_modified_flag:
            self.modified_embedding = nn.Embedding(2, single_size)
        if add_cyclic_flag:
            self.cyclic_projection = nn.Linear(1, single_size, bias=False)
        if add_mol_type_feat:
            self.molecule_type_embedding = nn.Embedding(num_molecule_types, single_size)
        for module in (self.method_embedding, self.modified_embedding, self.molecule_type_embedding):
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
        residue_type: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        method_feature: Tensor | None = None,
        modified: Tensor | None = None,
        cyclic_period: Tensor | None = None,
        molecule_type: Tensor | None = None,
    ) -> Tensor:
        r"""
        Pool atoms into per-token single states and add the token-level encodings and conditioning terms.

        Args:
            atom_features: Per-atom reference features ``(B, M, atom_feature_size)``.
            reference_positions: Centered reference-conformer atom coords ``(B, M, 3)``.
            atom_mask: Boolean atom validity mask ``(B, M)``.
            reference_space_id: Reference-conformer group id per atom ``(B, M)``.
            atom_to_token: Token index per atom ``(B, M)``.
            residue_type: Query residue-type one-hot ``(B, N, vocab_size)``.
            msa_profile: Single-sequence MSA profile ``(B, N, vocab_size)``.
            deletion_mean: Per-token deletion mean ``(B, N)``.
            method_feature: Optional experimental-method ids ``(B, N)``.
            modified: Optional modified-residue flags ``(B, N)``.
            cyclic_period: Optional cyclic-position periods ``(B, N)``.
            molecule_type: Optional molecule-type ids ``(B, N)``; required when molecule-type conditioning is on.

        Returns:
            Token single representation ``(B, N, single_size)``.

        """

        atom_query, atom_conditioning, atom_pair = self.atom_encoder(
            atom_features, reference_positions, atom_mask, reference_space_id
        )
        atom_pair_bias = self.atom_pair_bias_projection(atom_pair)
        token_from_atom = self.atom_attention_encoder(
            atom_query, atom_conditioning, atom_pair_bias, atom_mask, atom_to_token, residue_type.shape[-2]
        )

        single = token_from_atom
        single = single + self.residue_type_projection(residue_type.to(single.dtype))
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
        return single


class Boltz2AtomEncoder(nn.Module):
    r"""
    Build atom single states and windowed atom-pair states from reference geometry.

    Projects atom features to atom single states and encodes windowed atom-pair states from reference
    distance vectors, inverse distances, and a same-space-id validity mask. When ``structure_prediction`` is
    set, trunk single/pair representations are additionally injected (the diffusion atom encoder path).

    Args:
        atom_feature_size: Width of the per-atom reference feature vector.
        atom_hidden_size: Width of the atom single channel.
        atom_pair_hidden_size: Width of the atom pair channel.
        single_size: Width of the trunk single channel (used only when ``structure_prediction`` is set).
        pair_size: Width of the trunk pair channel (used only when ``structure_prediction`` is set).
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        structure_prediction: Whether to inject trunk single/pair states (diffusion atom encoder path).
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
        structure_prediction: bool = False,
    ) -> None:
        super().__init__()
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.single_trunk_projection: nn.Sequential | None = None
        self.pair_trunk_projection: nn.Sequential | None = None
        if structure_prediction:
            self.single_trunk_projection = nn.Sequential(
                LayerNorm(single_size), nn.Linear(single_size, atom_hidden_size, bias=False)
            )
            self.pair_trunk_projection = nn.Sequential(
                LayerNorm(pair_size),
                nn.Linear(pair_size, atom_pair_hidden_size, bias=False),
            )
        self.left_atom_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False),
        )
        self.right_atom_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False),
        )
        self.feed_forward = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
        )

    def forward(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        *,
        atom_to_token: Tensor | None = None,
        single_trunk: Tensor | None = None,
        pair_trunk: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Return atom query states, atom conditioning states, and windowed atom-pair states.

        Args:
            atom_features: Per-atom reference features ``(B, M, atom_feature_size)``.
            reference_positions: Centered reference-conformer atom coords ``(B, M, 3)``.
            atom_mask: Boolean atom validity mask ``(B, M)``.
            reference_space_id: Reference-conformer group id per atom ``(B, M)``.
            atom_to_token: Token index per atom ``(B, M)``; required when ``structure_prediction`` is set.
            single_trunk: Trunk single ``(B, N, single_size)``; injected when ``structure_prediction`` is set.
            pair_trunk: Trunk pair ``(B, N, N, pair_size)``; injected when ``structure_prediction`` is set.

        Returns:
            Tuple ``(atom_query, atom_conditioning, atom_pair)`` of atom single states ``(B, M, atom_hidden_size)``,
            atom conditioning states ``(B, M, atom_hidden_size)``, and windowed atom-pair states
            ``(B, W, Q, K, atom_pair_hidden_size)``.
        """

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

        pair_bias = None
        single_trunk_projection = self.single_trunk_projection
        if single_trunk_projection is not None:
            token_index = cast(Tensor, atom_to_token)
            single_proj = single_trunk_projection(cast(Tensor, single_trunk))
            single_from_trunk = single_proj.gather(1, token_index.unsqueeze(-1).expand(-1, -1, single_proj.shape[-1]))
            atom_single = atom_single + single_from_trunk * atom_mask.unsqueeze(-1).to(single_from_trunk.dtype)
            pair_from_trunk = cast(nn.Sequential, self.pair_trunk_projection)(cast(Tensor, pair_trunk))
            token_windows = pad_atoms(token_index)
            query_token = token_windows.view(batch, num_windows, num_query_atoms)
            key_token = to_keys(token_windows.unsqueeze(-1)).reshape(batch, num_windows, num_key_atoms)
            batch_index = torch.arange(batch, device=atom_single.device)[:, None, None, None]
            pair_gathered = pair_from_trunk[batch_index, query_token[:, :, :, None], key_token[:, :, None, :]]
            pair_bias = pair_gathered * (mask_queries & mask_keys).unsqueeze(-1).to(pair_gathered.dtype)

        single_windows = pad_atoms(atom_single)
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


class Boltz2AtomAttentionEncoder(nn.Module):
    r"""
    Windowed atom attention encoder and atom-to-token mean pool.

    Runs a packed windowed atom transformer over the atom single states, then mean-pools atoms into tokens.
    When ``structure_prediction`` is set it also adds a noisy-position projection and returns the atom skip
    states consumed by the diffusion decoder.

    Args:
        atom_hidden_size: Width of the atom single channel.
        single_size: Width of the produced token representation.
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_atom_encoder_blocks: Number of atom transformer blocks.
        num_atom_encoder_attention_heads: Head count for the atom transformer.
        structure_prediction: Whether to project noisy positions and emit diffusion skip states.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        single_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        structure_prediction: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.noisy_position_projection: nn.Linear | None = None
        if structure_prediction:
            self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_mode=False,
            transition="triple",
        )
        token_out = 2 * single_size if structure_prediction else single_size
        self.token_projection = nn.Sequential(nn.Linear(atom_hidden_size, token_out, bias=False), nn.ReLU())

    def forward(
        self,
        atom_single: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        noisy_positions: Tensor | None = None,
        multiplicity: int = 1,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        r"""
        Return token representations, and diffusion skip states when structure prediction is enabled.

        Args:
            atom_single: Atom single states ``(B, M, atom_hidden_size)``.
            atom_conditioning: Atom conditioning states ``(B, M, atom_hidden_size)``.
            atom_pair_bias: Windowed atom-pair attention bias ``(B, W, Q, K, heads)``.
            atom_mask: Boolean atom validity mask ``(B, M)``.
            atom_to_token: Token index per atom ``(B, M)``.
            num_tokens: Number of tokens ``N`` to scatter atoms into.
            noisy_positions: Noised atom coords ``(B*multiplicity, M, 3)``; required when structure prediction is on.
            multiplicity: Number of diffusion samples per batch element folded into the leading axis.

        Returns:
            Token representation ``(B, N, single_size)`` when structure prediction is off; otherwise the tuple
            ``(token_repr, atom_query, conditioning)`` where ``token_repr`` is ``(B*multiplicity, N, 2*single_size)``
            and the atom skip states are ``(B*multiplicity, M, atom_hidden_size)``.
        """

        query = atom_single
        conditioning = atom_conditioning
        mask = atom_mask
        token_index = atom_to_token
        noisy_position_projection = self.noisy_position_projection
        if noisy_position_projection is not None:
            query = query.repeat_interleave(multiplicity, 0) + noisy_position_projection(cast(Tensor, noisy_positions))
            conditioning = conditioning.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
            token_index = token_index.repeat_interleave(multiplicity, 0)

        atom_query = self.atom_transformer(
            query,
            conditioning,
            atom_pair_bias,
            atom_mask=mask.bool(),
            attention_layout="local",
        )
        token_from_atom = self.token_projection(atom_query)
        token_repr = dense_mean_to_tokens(token_from_atom, token_index, num_tokens, atom_mask=mask, count_eps=1e-6)
        if noisy_position_projection is not None:
            return token_repr, atom_query, conditioning
        return token_repr


class Boltz2AtomAttentionDecoder(nn.Module):
    r"""
    Broadcast token activations back to atoms and predict a per-atom position update.

    Scatters the token representation to atoms, adds the encoder atom skip states, runs a windowed atom
    transformer, and projects to a per-atom coordinate update.

    Args:
        atom_hidden_size: Width of the atom single channel.
        single_size: Width of the token representation broadcast to atoms (input width is ``2*single_size``).
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_atom_decoder_blocks: Number of atom transformer blocks.
        num_atom_decoder_attention_heads: Head count for the atom transformer.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        single_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(2 * single_size, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_mode=False,
            transition="triple",
        )
        self.position_update = nn.Sequential(
            LayerNorm(atom_hidden_size, operator_backend=operator_backend), nn.Linear(atom_hidden_size, 3, bias=False)
        )

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
        r"""
        Return atom-coordinate updates.

        Args:
            token_repr: Token representation ``(B*multiplicity, N, 2*single_size)``.
            atom_skip: Encoder atom skip states ``(B*multiplicity, M, atom_hidden_size)``.
            conditioning_skip: Encoder atom conditioning skip states ``(B*multiplicity, M, atom_hidden_size)``.
            atom_pair_bias: Windowed atom-pair attention bias ``(B, W, Q, K, heads)``.
            atom_mask: Boolean atom validity mask ``(B, M)``.
            atom_to_token: Token index per atom ``(B, M)``.
            multiplicity: Number of diffusion samples per batch element folded into the leading axis.

        Returns:
            Per-atom position update ``(B*multiplicity, M, 3)``.
        """

        mask = atom_mask.repeat_interleave(multiplicity, 0).bool()
        token_index = atom_to_token.repeat_interleave(multiplicity, 0)
        token_proj = self.token_projection(token_repr)
        token_to_atom = token_proj.gather(1, token_index.unsqueeze(-1).expand(-1, -1, token_proj.shape[-1]))
        atom_query = atom_skip + token_to_atom * mask.unsqueeze(-1).to(token_to_atom.dtype)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning_skip,
            atom_pair_bias,
            atom_mask=mask,
            attention_layout="local",
        )
        return self.position_update(atom_query)


CONTACT_CONDITIONING_CLASSES: tuple[str, ...] = (
    "UNSPECIFIED",
    "UNSELECTED",
    "POCKET>BINDER",
    "BINDER>POCKET",
    "CONTACT",
)


class Boltz2TemplateEmbedder(nn.Module):
    r"""
    Boltz-2 template embedder with on-the-fly distogram / unit-vector featurization.

    Unlike [`TemplateEmbedder`][ofoldx.modules.embedding.template.TemplateEmbedder] (which consumes pre-packed
    features), this computes the template distogram and per-frame unit vectors inside ``forward`` from raw
    alpha/beta-carbon positions and frames, masks pairs by visibility group, and folds templates into the batch through
    a shared pair-only pairformer with an outer residual, and aggregates by a template-count-weighted mean.

    Feature order fed to ``template_projection`` is load-bearing: ``[distogram, beta_carbon_mask, unit_vector,
    frame_mask, residue_type_i, residue_type_j]`` (= 2*vocab_size + num_bins + 5 channels).

    Args:
        pair_size: Width of the trunk pair representation this update is added into.
        hidden_size: Width of the internal template pair channel.
        num_blocks: Number of pair-only triangle update blocks in the shared pairformer.
        vocab_size: Residue-type vocabulary size (one block per side of the pair).
        num_bins: Number of distogram bins computed from CB distances.
        min_distance: Lower edge of the distogram binning range, in angstroms.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        expansion: Transition expansion factor inside each triangle update block.
        triangle_attention_hidden_size: Per-head dimension of the triangle attention.
        num_triangle_attention_heads: Number of triangle attention heads.
        dropout: Dropout probability inside each triangle update block.
        operator_backend: Operator backend selector forwarded to the triangle update blocks.

    Examples:
        >>> from ofoldx.models.boltz2.model import Boltz2TemplateEmbedder  # doctest: +SKIP
        >>> embedder = Boltz2TemplateEmbedder(pair_size=128, hidden_size=64).cuda()  # doctest: +SKIP
        >>> update = embedder(  # doctest: +SKIP
        ...     pair,
        ...     residue_type=residue_type,
        ...     frame_rotations=frame_rotations,
        ...     frame_translations=frame_translations,
        ...     frame_mask=frame_mask,
        ...     beta_carbon_positions=beta_carbon_positions,
        ...     alpha_carbon_positions=alpha_carbon_positions,
        ...     beta_carbon_mask=beta_carbon_mask,
        ...     visibility_index=visibility_index,
        ...     template_mask=template_mask,
        ... )
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = 33,
        num_bins: int = 38,
        min_distance: float = 3.25,
        max_distance: float = 50.75,
        expansion: int = 4,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        template_feature_size = 2 * vocab_size + num_bins + 5
        self.pair_norm = LayerNorm(pair_size, operator_backend=operator_backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.template_projection = nn.Linear(template_feature_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            TrianglePairUpdateBlock(
                pair_size=hidden_size,
                triangle_multiplication_hidden_size=hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                dropout=dropout,
                operator_backend=operator_backend,
            )
            for _ in range(num_blocks)
        )
        self.hidden_norm = LayerNorm(hidden_size, operator_backend=operator_backend)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def _featurize(
        self,
        residue_type: Tensor,
        frame_rotations: Tensor,
        frame_translations: Tensor,
        frame_mask: Tensor,
        beta_carbon_positions: Tensor,
        alpha_carbon_positions: Tensor,
        beta_carbon_mask: Tensor,
        visibility_index: Tensor,
    ) -> Tensor:
        r"""
        Build the packed per-template pair feature tensor from raw coordinates and frames.

        Computes the beta-carbon distance distogram and per-frame alpha-carbon unit vectors,
        masks pairs to the same visibility group, and concatenates the geometry with the broadcast per-side residue
        types in the load-bearing channel order.

        Args:
            residue_type: ``[batch, num_templates, N_token, vocab_size]`` per-token residue features.
            frame_rotations: ``[batch, num_templates, N_token, 3, 3]`` backbone frame rotations.
            frame_translations: ``[batch, num_templates, N_token, 3]`` backbone frame translations.
            frame_mask: ``[batch, num_templates, N_token]`` backbone frame validity mask.
            beta_carbon_positions: ``[batch, num_templates, N_token, 3]`` beta-carbon positions.
            alpha_carbon_positions: ``[batch, num_templates, N_token, 3]`` alpha-carbon positions.
            beta_carbon_mask: ``[batch, num_templates, N_token]`` beta-carbon validity mask.
            visibility_index: ``[batch, num_templates, N_token]`` visibility-group indices; only same-group
                pairs keep their geometry.

        Returns:
            ``[batch, num_templates, N_token, N_token, 2 * vocab_size + num_bins + 5]`` packed features.
        """
        num_tokens = residue_type.shape[2]
        pair_beta_carbon_mask = (beta_carbon_mask[..., :, None] * beta_carbon_mask[..., None, :]).unsqueeze(-1)
        pair_frame_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        same_visibility = (visibility_index[..., :, None] == visibility_index[..., None, :]).to(residue_type.dtype)

        beta_carbon_distances = torch.cdist(beta_carbon_positions, beta_carbon_positions)
        boundaries = torch.linspace(
            self.min_distance,
            self.max_distance,
            self.num_bins - 1,
            device=beta_carbon_distances.device,
        )
        distogram = (beta_carbon_distances[..., None] > boundaries).sum(dim=-1).long()
        distogram = F.one_hot(distogram, num_classes=self.num_bins).to(residue_type.dtype)

        # Per-frame unit vector of CA_j in frame_i (replicated faithfully, incl. the per-component norm).
        rotation = frame_rotations.unsqueeze(2).transpose(-1, -2)
        translation = frame_translations.unsqueeze(2).unsqueeze(-1)
        targets = alpha_carbon_positions.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(rotation, targets - translation)
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector)).squeeze(-1)

        geometry = torch.cat(
            [distogram, pair_beta_carbon_mask, unit_vector, pair_frame_mask],
            dim=-1,
        )
        geometry = geometry * same_visibility.unsqueeze(-1)
        residue_i = residue_type[:, :, :, None].expand(-1, -1, -1, num_tokens, -1)
        residue_j = residue_type[:, :, None, :].expand(-1, -1, num_tokens, -1, -1)
        return torch.cat([geometry, residue_i, residue_j], dim=-1)

    def forward(
        self,
        pair: Tensor,
        *,
        residue_type: Tensor,
        frame_rotations: Tensor,
        frame_translations: Tensor,
        frame_mask: Tensor,
        beta_carbon_positions: Tensor,
        alpha_carbon_positions: Tensor,
        beta_carbon_mask: Tensor,
        visibility_index: Tensor,
        template_mask: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return the template pair update ``u`` (added into the trunk pair representation).

        Templates are folded into the batch dim (``reshape(batch * num_templates, ...)``) and run through the
        template pairformer in one batched pass, then aggregated by a template-count-weighted mean over the
        per-template mask.

        Args:
            pair: ``[batch, N_token, N_token, pair_size]`` trunk pair representation.
            residue_type: ``[batch, num_templates, N_token, vocab_size]`` per-token residue features.
            frame_rotations: ``[batch, num_templates, N_token, 3, 3]`` backbone frame rotations.
            frame_translations: ``[batch, num_templates, N_token, 3]`` backbone frame translations.
            frame_mask: ``[batch, num_templates, N_token]`` backbone frame validity mask.
            beta_carbon_positions: ``[batch, num_templates, N_token, 3]`` beta-carbon positions.
            alpha_carbon_positions: ``[batch, num_templates, N_token, 3]`` alpha-carbon positions.
            beta_carbon_mask: ``[batch, num_templates, N_token]`` beta-carbon validity mask.
            visibility_index: ``[batch, num_templates, N_token]`` visibility-group indices.
            template_mask: ``[batch, num_templates, N_token]`` per-token template validity mask; a template
                counts as present when any token is valid.
            pair_mask: Optional ``[batch, N_token, N_token]`` pair validity mask forwarded to the blocks.

        Returns:
            ``[batch, N_token, N_token, pair_size]`` template pair update to add into ``pair``.

        Examples:
            >>> embedder = Boltz2TemplateEmbedder().cuda()  # doctest: +SKIP
            >>> update = embedder(pair, residue_type=residue_type, frame_rotations=frame_rotations,
            ...                   frame_translations=frame_translations, frame_mask=frame_mask,
            ...                   beta_carbon_positions=beta_carbon_positions,
            ...                   alpha_carbon_positions=alpha_carbon_positions,
            ...                   beta_carbon_mask=beta_carbon_mask,
            ...                   visibility_index=visibility_index,
            ...                   template_mask=template_mask)  # doctest: +SKIP
        """

        batch, num_templates = residue_type.shape[:2]
        present = template_mask.any(dim=2).to(pair.dtype)
        template_count = present.sum(dim=1).clamp(min=1)

        features = self._featurize(
            residue_type,
            frame_rotations,
            frame_translations,
            frame_mask,
            beta_carbon_positions,
            alpha_carbon_positions,
            beta_carbon_mask,
            visibility_index,
        )
        projected = self.template_projection(features)
        template_pair = self.pair_projection(self.pair_norm(pair[:, None])) + projected
        template_pair = template_pair.reshape(batch * num_templates, *template_pair.shape[2:])

        if pair_mask is not None:
            pair_mask = pair_mask[:, None].expand(-1, num_templates, -1, -1)
            pair_mask = pair_mask.reshape(batch * num_templates, *pair_mask.shape[2:])

        stacked = template_pair
        for block in self.blocks:
            stacked = block(stacked, pair_mask=pair_mask)
        template_pair = self.hidden_norm(template_pair + stacked)
        template_pair = template_pair.reshape(batch, num_templates, *template_pair.shape[1:])

        weight = present[:, :, None, None, None]
        update = (template_pair * weight).sum(dim=1) / template_count[:, None, None, None].to(pair.dtype)
        return self.output_projection(self.activation(update))


class Boltz2ContactConditioning(nn.Module):
    r"""
    Additive pair bias encoding Boltz-2 pocket and contact constraints.

    Encodes the active contact-class one-hot together with a Fourier-embedded normalized threshold into a
    pairwise bias, gated so that the ``UNSPECIFIED`` / ``UNSELECTED`` classes contribute their own learned bias.

    Args:
        pair_size: Width of the produced pair bias.
        cutoff_min: Lower distance cutoff used to normalize the contact threshold, in angstroms.
        cutoff_max: Upper distance cutoff used to normalize the contact threshold, in angstroms.
        num_classes: Number of contact-conditioning classes; defaults to
            ``len(CONTACT_CONDITIONING_CLASSES)``.
    """

    def __init__(
        self,
        pair_size: int,
        *,
        cutoff_min: float,
        cutoff_max: float,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        num_classes = len(CONTACT_CONDITIONING_CLASSES) if num_classes is None else num_classes
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max
        self.num_classes = num_classes
        self.threshold_fourier = FrozenFourierEmbedding(pair_size)
        self.constraint_encoder = nn.Linear(pair_size + num_classes - 1, pair_size)
        self.unspecified_bias = nn.Parameter(torch.zeros(pair_size))
        self.unselected_bias = nn.Parameter(torch.zeros(pair_size))

    def forward(self, contact_conditioning: Tensor, contact_threshold: Tensor) -> Tensor:
        r"""
        Encode constraints into a pairwise bias.

        Args:
            contact_conditioning: Contact-class one-hot ``(B, N, N, num_classes)``; the leading two columns are
                the ``UNSPECIFIED`` / ``UNSELECTED`` gates and the rest are the active constraint classes.
            contact_threshold: Pairwise contact thresholds ``(B, N, N)``, in angstroms.

        Returns:
            Additive pair bias ``(B, N, N, pair_size)``.
        """

        active = contact_conditioning[..., 2:]
        threshold_normalized = (contact_threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        threshold_fourier = self.threshold_fourier(threshold_normalized)

        encoder_input = torch.cat(
            [active, threshold_normalized.unsqueeze(-1), threshold_fourier],
            dim=-1,
        )
        encoded = self.constraint_encoder(encoder_input)

        gate = 1 - contact_conditioning[..., 0:2].sum(dim=-1, keepdim=True)
        return (
            encoded * gate
            + self.unspecified_bias * contact_conditioning[..., 0:1]
            + self.unselected_bias * contact_conditioning[..., 1:2]
        )


class Boltz2DenoiserConditioning(nn.Module):
    r"""
    Precompute the diffusion conditioning shared across denoiser calls (Boltz DiffusionConditioning).

    Conditions the trunk pair with the relative-position encoding, runs the (structure-prediction) atom encoder,
    and projects the atom-pair / token-pair states to per-block attention biases. The result is invariant across
    reverse-diffusion steps, so it is computed once and reused.

    Args:
        single_size: Width of the trunk single channel injected into the atom encoder.
        pair_size: Width of the trunk pair channel.
        atom_hidden_size: Width of the atom single channel.
        atom_pair_hidden_size: Width of the atom pair channel.
        atom_feature_size: Width of the per-atom reference feature vector.
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_atom_encoder_blocks: Number of atom-encoder blocks whose biases are precomputed.
        num_atom_encoder_attention_heads: Head count for the atom-encoder biases.
        num_atom_decoder_blocks: Number of atom-decoder blocks whose biases are precomputed.
        num_atom_decoder_attention_heads: Head count for the atom-decoder biases.
        num_token_transformer_blocks: Number of token-transformer blocks whose biases are precomputed.
        num_token_transformer_heads: Head count for the token-transformer biases.
        num_conditioning_transitions: Number of transitions in the pairwise conditioner.
        operator_backend: Public backend request forwarded to the pairwise conditioner.
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
        num_token_transformer_blocks: int = 24,
        num_token_transformer_heads: int = 16,
        num_conditioning_transitions: int = 2,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            operator_backend=operator_backend,
        )
        self.atom_encoder = Boltz2AtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            structure_prediction=True,
        )
        self.atom_encoder_pair_bias_projections = nn.ModuleList(
            nn.Sequential(
                LayerNorm(atom_pair_hidden_size, operator_backend=operator_backend),
                nn.Linear(atom_pair_hidden_size, num_atom_encoder_attention_heads, bias=False),
            )
            for _ in range(num_atom_encoder_blocks)
        )
        self.atom_decoder_pair_bias_projections = nn.ModuleList(
            nn.Sequential(
                LayerNorm(atom_pair_hidden_size, operator_backend=operator_backend),
                nn.Linear(atom_pair_hidden_size, num_atom_decoder_attention_heads, bias=False),
            )
            for _ in range(num_atom_decoder_blocks)
        )
        self.token_transformer_pair_bias_projections = nn.ModuleList(
            nn.Sequential(
                LayerNorm(pair_size, operator_backend=operator_backend),
                nn.Linear(pair_size, num_token_transformer_heads, bias=False),
            )
            for _ in range(num_token_transformer_blocks)
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
        r"""
        Precompute the per-call-invariant diffusion conditioning from the trunk representations.

        The trunk pair ``(B, N, N, pair_size)`` is conditioned with the relative-position encoding, then the
        atom encoder builds atom single/pair states injected with trunk single/pair; the atom-pair and token-pair
        states are projected to per-block attention biases (concatenated over blocks along the last axis).

        Args:
            single_trunk: Trunk refined single ``(B, N, single_size)``.
            pair_trunk: Trunk pair ``(B, N, N, pair_size)``.
            relative_position_encoding: Relative-position pair features ``(B, N, N, pair_size)``.
            atom_features: Per-atom reference features ``(B, M, atom_feature_size)``.
            reference_positions: Centered reference-conformer atom coords ``(B, M, 3)``.
            atom_mask: Boolean atom validity mask ``(B, M)``.
            reference_space_id: Reference-conformer group id per atom ``(B, M)``.
            atom_to_token: Token index per atom ``(B, M)``.

        Returns:
            Mapping with atom query and conditioning states plus packed encoder, decoder, and
            token-transformer attention biases. Bias channels are ordered by block, then head.
        """

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
        atom_enc_bias = torch.cat([layer(atom_pair) for layer in self.atom_encoder_pair_bias_projections], dim=-1)
        atom_dec_bias = torch.cat([layer(atom_pair) for layer in self.atom_decoder_pair_bias_projections], dim=-1)
        token_trans_bias = torch.cat([layer(pair) for layer in self.token_transformer_pair_bias_projections], dim=-1)
        return {
            "atom_query": atom_query,
            "atom_conditioning": atom_conditioning,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }


class Boltz2Denoiser(nn.Module):
    r"""
    EDM score network: ``(noisy positions, noise level, trunk, conditioning) -> position update``.

    Args:
        single_size: Width of the trunk single channel (the token channel is ``2*single_size``).
        atom_hidden_size: Width of the atom single channel.
        fourier_embedding_size: Width of the Fourier noise-level embedding.
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        num_token_transformer_blocks: Number of conditioned token-transformer blocks.
        num_token_transformer_heads: Head count for the token transformer.
        num_atom_decoder_blocks: Number of atom-attention decoder blocks.
        num_atom_decoder_attention_heads: Head count for the atom-attention decoder.
        num_conditioning_transitions: Number of transitions in the single conditioner.
        operator_backend: Selector for the fused-operator backend.
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
        num_token_transformer_blocks: int = 24,
        num_token_transformer_heads: int = 16,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_conditioning_transitions: int = 2,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_conditioner = Boltz2SingleConditioning(
            single_size=single_size,
            fourier_embedding_size=fourier_embedding_size,
            num_conditioning_transitions=num_conditioning_transitions,
            operator_backend=operator_backend,
        )
        self.atom_attention_encoder = Boltz2AtomAttentionEncoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_encoder_blocks=num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
            structure_prediction=True,
            operator_backend=operator_backend,
        )
        self.single_to_token = nn.Sequential(
            LayerNorm(hidden, operator_backend=operator_backend), nn.Linear(hidden, hidden, bias=False)
        )
        self.token_transformer = ConditionedTransformer(
            num_blocks=num_token_transformer_blocks,
            num_attention_heads=num_token_transformer_heads,
            hidden_size=hidden,
            conditioning_size=hidden,
            operator_backend=operator_backend,
        )
        self.token_norm = LayerNorm(hidden, operator_backend=operator_backend)
        self.atom_attention_decoder = Boltz2AtomAttentionDecoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_decoder_blocks=num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=num_atom_decoder_attention_heads,
            operator_backend=operator_backend,
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
        r"""
        Denoise atom coords: encode noisy atoms to tokens, run the conditioned token transformer, decode to a
        per-atom update.

        Args:
            single_inputs: Trunk input single ``(B, N, single_size)``.
            single_trunk: Trunk refined single ``(B, N, single_size)``.
            noisy_positions: Noised atom coords ``(B*multiplicity, M, 3)``.
            noise_level: Per-sample noise level ``(B*multiplicity,)`` (EDM ``c_noise(sigma)``).
            conditioning: Output dict of
                [`DiffusionConditioning`][ofoldx.models.boltz2.model.DiffusionConditioning].
            atom_mask: Boolean atom validity mask ``(B, M)``.
            atom_to_token: Token index per atom ``(B, M)``.
            token_mask: Boolean token validity mask ``(B, N)``.
            multiplicity: Number of diffusion samples per batch element folded into the leading axis.

        Returns:
            Per-atom position update ``(B*multiplicity, M, 3)``.
        """

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
        token_repr = token_repr + self.single_to_token(single)

        token_repr = self.token_transformer(
            token_repr,
            single,
            conditioning["token_trans_bias"].repeat_interleave(multiplicity, 0),
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


class Boltz2SingleConditioning(nn.Module):
    r"""
    Condition the (concatenated trunk+input) single representation on the diffusion noise level.

    Args:
        single_size: Width of each single channel; the concatenated token channel is ``2*single_size``.
        fourier_embedding_size: Width of the Fourier noise-level embedding.
        num_conditioning_transitions: Number of transitions refining the conditioned single.
        expansion: Transition expansion factor.
        operator_backend: Public backend request forwarded to every transition.
    """

    def __init__(
        self,
        single_size: int = 384,
        fourier_embedding_size: int = 256,
        num_conditioning_transitions: int = 2,
        expansion: int = 2,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_norm = LayerNorm(hidden, operator_backend=operator_backend)
        self.single_projection = nn.Linear(hidden, hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size, operator_backend=operator_backend)
        self.fourier_projection = nn.Linear(fourier_embedding_size, hidden, bias=False)
        self.transitions = nn.ModuleList(
            Transition(hidden, expansion=expansion, operator_backend=operator_backend)
            for _ in range(num_conditioning_transitions)
        )

    def forward(self, noise_level: Tensor, single_trunk: Tensor, single_inputs: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Concatenate ``single_trunk`` and ``single_inputs``, normalize/project, add the Fourier-embedded
        ``noise_level``, and refine with the transition stack.

        Args:
            noise_level: Per-sample EDM noise level ``(B*multiplicity,)``.
            single_trunk: Trunk refined single ``(B*multiplicity, N, single_size)``.
            single_inputs: Trunk input single ``(B*multiplicity, N, single_size)``.

        Returns:
            Tuple of the conditioned single ``(B*multiplicity, N, 2*single_size)`` and the normalized noise
            embedding ``(B*multiplicity, fourier_embedding_size)``.
        """

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


class Boltz2CoordinateDiffusion(nn.Module):
    r"""
    EDM sampler around a [`Boltz2Denoiser`][ofoldx.models.boltz2.model.Boltz2Denoiser].

    Wraps the score model with EDM preconditioning and a Karras noise schedule, and runs reverse diffusion
    with optional in-loop weighted rigid alignment and stochastic churn.

    Args:
        denoiser: The learned coordinate network to precondition and iterate.
        sigma_data: EDM data standard deviation used for preconditioning and to scale the schedule.
        min_noise_level: Lowest non-zero sigma before the trailing zero.
        max_noise_level: Highest sigma at the start of the schedule.
        rho: Exponent of the Karras noise schedule.
        churn_rate: Stochastic churn added per step above ``min_churn_noise_level``.
        min_churn_noise_level: Sigma threshold above which churn is applied.
        noise_scale: Multiplier on the injected churn noise.
        step_scale: Scale on the Euler update step.
        num_sampling_steps: Default number of sampling steps.
        alignment_reverse_diff: Whether to weighted-rigid-align each reverse-diffusion step.
    """

    def __init__(
        self,
        denoiser: Boltz2Denoiser,
        sigma_data: float = 16.0,
        min_noise_level: float = 1e-4,
        max_noise_level: float = 160.0,
        rho: float = 7.0,
        churn_rate: float = 0.8,
        min_churn_noise_level: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        num_sampling_steps: int = 200,
        alignment_reverse_diff: bool = True,
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
        self.num_sampling_steps = num_sampling_steps
        self.alignment_reverse_diff = alignment_reverse_diff
        # EDM log-normal training noise distribution (Karras et al.); used by the training denoise step only.
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std

    def noise_distribution(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None
    ) -> Tensor:
        r"""
        Draw per-example EDM training noise levels ``sigma = sigma_data * exp(P_mean + P_std * N(0, 1))``.

        Args:
            batch_size: Number of noise levels to draw (``B*multiplicity``).
            device: Device for the sampled tensor.
            dtype: Dtype for the sampled tensor.
            generator: Optional PyTorch random generator.

        Returns:
            Noise levels ``(batch_size,)``.
        """

        normal = torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
        return self.sigma_data * torch.exp(self.noise_log_mean + self.noise_log_std * normal)

    def loss_weight(self, sigma: Tensor) -> Tensor:
        r"""
        EDM per-example loss weight ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``.

        Args:
            sigma: Per-example noise levels ``(batch_size,)``.

        Returns:
            Loss weights ``(batch_size,)``.
        """

        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2 + 1e-8)

    def denoise(
        self,
        target_coords: Tensor,
        atom_mask: Tensor,
        *,
        multiplicity: int = 1,
        generator: torch.Generator | None = None,
        **conditioning: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        One training denoising step: centre the target, corrupt it, and denoise through the score model.

        Applies EDM noising followed by one preconditioned denoising pass. This tensor-only path omits coordinate
        augmentation and target alignment so it remains graph-break-free under ``torch.compile``.

        Args:
            target_coords: Experimental target atom coords ``(B, M, 3)``.
            atom_mask: Boolean atom validity mask ``(B, M)``.
            multiplicity: Number of noised copies drawn per batch element.
            generator: Optional PyTorch random generator for the diffusion noise.
            **conditioning: Score-model conditioning tensors (``single_inputs``, ``single_trunk``,
                ``conditioning``, ``atom_to_token``, ``token_mask``).

        Returns:
            Tuple ``(denoised, sigmas, true_coords)`` of denoised coords ``(B*multiplicity, M, 3)``, sampled noise
            levels ``(B*multiplicity,)``, and the centred target coords ``(B*multiplicity, M, 3)``.
        """

        coords = target_coords.repeat_interleave(multiplicity, 0)
        mask = atom_mask.repeat_interleave(multiplicity, 0).unsqueeze(-1).to(coords.dtype)
        center = (coords * mask).sum(dim=-2, keepdim=True) / (mask.sum(dim=-2, keepdim=True) + 1e-8)
        coords = (coords - center) * mask
        batch = coords.shape[0]
        sigmas = self.noise_distribution(batch, device=coords.device, dtype=coords.dtype, generator=generator)
        noise = torch.randn(coords.shape, device=coords.device, dtype=coords.dtype, generator=generator)
        noised = coords + sigmas.reshape(batch, 1, 1) * noise
        denoised = self.preconditioned_network_forward(
            noised, sigmas, multiplicity=multiplicity, atom_mask=atom_mask, **conditioning
        )
        return denoised, sigmas, coords

    def preconditioned_network_forward(
        self, noised_coords: Tensor, sigma: Tensor | float, *, multiplicity: int = 1, **conditioning: object
    ) -> Tensor:
        r"""
        One EDM-preconditioned denoising step: noisy coords -> denoised coords.

        Args:
            noised_coords: Noised atom coords ``(B*multiplicity, M, 3)``.
            sigma: Noise level, either a scalar or a per-sample tensor ``(B*multiplicity,)``.
            multiplicity: Number of diffusion samples per batch element folded into the leading axis.
            **conditioning: Score-model conditioning tensors (``single_inputs``, ``single_trunk``,
                ``conditioning``, ``atom_mask``, ``atom_to_token``, ``token_mask``).

        Returns:
            Denoised atom coords ``(B*multiplicity, M, 3)``.
        """

        batch = noised_coords.shape[0]
        sigma = torch.as_tensor(sigma, device=noised_coords.device, dtype=noised_coords.dtype).expand(batch)
        padded = sigma.reshape(batch, 1, 1)
        update = self.denoiser(
            noisy_positions=self.preconditioner.c_in(padded) * noised_coords,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
            **conditioning,
        )
        return self.preconditioner.c_skip(padded) * noised_coords + self.preconditioner.c_out(padded) * update

    def sample_schedule(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        r"""
        Build the EDM (Karras rho-power) noise schedule scaled by ``sigma_data``.

        The schedule decreases from ``max_noise_level`` to ``min_noise_level`` with a trailing zero.

        Args:
            num_sampling_steps: Number of sampling steps; defaults to ``self.num_sampling_steps``.
            device: Device on which to build the schedule tensor.

        Returns:
            Sigma schedule ``(steps + 1,)`` ending in a zero.
        """

        steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        inv_rho = 1 / self.rho
        # `steps` is the number of sampling steps (transitions); the trailing pad adds the final sigma=0.
        # Clamp the divisor so a single step mirrors the AF3 scheduler (ramp [0] -> a lone max_noise_level, then 0)
        # instead of dividing by zero; for steps >= 2 the divisor is steps - 1 exactly, so the schedule is unchanged.
        ramp = torch.arange(steps, device=device, dtype=torch.float32) / max(steps - 1, 1)
        sigmas = (
            self.max_noise_level**inv_rho + ramp * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    @torch.no_grad()
    def _sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        generator: torch.Generator | None = None,
        **conditioning: object,
    ) -> Tensor:
        r"""
        Run reverse diffusion (the N-step inference sampler) and return sampled atom coords.

        Args:
            atom_mask: Boolean atom validity mask ``(B, M)``.
            num_sampling_steps: Number of sampling steps; defaults to ``self.num_sampling_steps``.
            multiplicity: Number of independent samples to draw per batch element.
            generator: Optional PyTorch random generator for the diffusion noise.
            **conditioning: Score-model conditioning tensors (``single_inputs``, ``single_trunk``,
                ``conditioning``, ``atom_to_token``, ``token_mask``).

        Returns:
            Sampled atom coords ``(B*multiplicity, M, 3)``.
        """

        device = atom_mask.device
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)

        def randn() -> Tensor:
            r"""Draw standard-normal noise of shape ``(B*multiplicity, M, 3)`` on ``device`` from ``generator``."""

            return torch.randn(shape, device=device, generator=generator)

        coords = sigmas[0] * randn()
        denoised = None
        # The EDM step math matches ``CoordinateEdmSampler``: churn, Euler update, and the optional
        # ``alignment_reverse_diff`` mid-step weighted rigid alignment. The loop remains local because
        # Boltz-2 differs in three RNG-sensitive conventions: (1) the
        # Karras schedule uses the ``steps - 1`` divisor + trailing-zero pad, not EdmScheduler's ``steps``
        # divisor; (2) coordinate augmentation multiplies on the right (einsum "bmd,bds->bms", i.e. R^T @ x),
        # whereas the shared sampler's ``_rot_vec_mul`` is R @ x; and (3) noise is drawn with
        # ``torch.randn`` at model-module scope rather than the shared sampler's ``Tensor.normal_`` calls.
        # `sigmas`/`gammas` are kept as 0-dim tensors (no ``.item()``): the Python loop still drives the fixed
        # number of steps, but every per-step scalar stays a tensor so the per-step denoiser is compile-clean.
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
            # Dtype-transparent EDM churn/Euler update (no forced ``.double()``): the churn/step scalars stay in
            # the working dtype, matching the BoltzGen sampler and keeping the loop honest about its precision.
            # At high sigma an fp32 ``sigma_hat**2 - sigma_current**2`` loses a few ULPs versus fp64, which only
            # bites at unrealistically low step counts (n_step=5); at realistic step counts (n_step>=200) the
            # per-step sigma gap is small and the fp32 result tracks to ~1e-6 (see sampler_parity).
            noise_var = self.noise_scale**2 * (sigma_hat**2 - sigma_current**2)
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)) * randn()
            denoised = self.preconditioned_network_forward(
                noisy, sigma_hat, multiplicity=multiplicity, atom_mask=base_atom_mask, **conditioning
            )
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy.float(), target=denoised.float(), weight=expanded_atom_mask.float()
                ).to(denoised)
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + self.step_scale * (sigma_next - sigma_hat) * derivative
        return coords


class Boltz2AffinityHead(nn.Module):
    r"""
    Predict scalar binding affinity and a binder-class from trunk features and predicted atom geometry.

    Rebuilds a pair representation from the trunk single/pair features and a distogram of representative-atom
    distances, refines it with a pair-only Pairformer stack over the receptor/ligand pair mask, and projects to
    scalar affinity value and binder-class logits.

    Args:
        single_size: Width of the trunk single channel injected into the pair.
        pair_size: Width of the pair representation.
        num_blocks: Number of pair-only Pairformer blocks.
        num_distogram_bins: Number of distogram bins used to condition the pair.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_conditioning_transitions: Number of transitions in the pairwise conditioner.
        expansion: Transition expansion factor in the pairwise conditioner.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Triangle-attention head count.
        dropout: Dropout applied in the Pairformer stack.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 4,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        num_conditioning_transitions: int = 2,
        expansion: int = 2,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.0,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.dist_bin_pairwise_embed = nn.Embedding(num_distogram_bins, pair_size)
        nn.init.zeros_(self.dist_bin_pairwise_embed.weight)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, operator_backend=operator_backend)
        self.pair_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            expansion=expansion,
            operator_backend=operator_backend,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=0,
            pair_size=pair_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.output_head = Boltz2AffinityOutputHead(pair_size=pair_size, single_size=single_size)

    def forward(
        self,
        *,
        single_inputs: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        token_to_rep_atom: Tensor,
        token_mask: Tensor,
        ligand_mask: Tensor,
        receptor_mask: Tensor,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        r"""
        Score affinity from trunk pair features and predicted atom coords.

        Args:
            single_inputs: Trunk input single ``(B, N, single_size)``.
            pair: Trunk pair ``(B, N, N, pair_size)``.
            coordinates: Predicted atom coordinates ``(B*multiplicity, M, 3)``.
            token_to_rep_atom: Representative atom index per token ``(B, N)``.
            token_mask: Boolean token validity mask ``(B, N)``.
            ligand_mask: Boolean mask selecting ligand tokens ``(B, N)``.
            receptor_mask: Boolean mask selecting receptor tokens ``(B, N)``.
            multiplicity: Number of samples per batch element folded into the leading axis.

        Returns:
            Dict with ``affinity_pred_value`` ``(B*multiplicity, 1)`` and ``affinity_logits_binary``
            ``(B*multiplicity, 1)``.
        """

        pair = self.pair_projection(self.pair_norm(pair))
        pair = pair + self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
        pair = pair + self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        pair = pair.repeat_interleave(multiplicity, 0)

        rep_atom_index = token_to_rep_atom.repeat_interleave(multiplicity, 0).long()
        representative = coordinates.gather(
            1,
            rep_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1]),
        )
        distances = torch.cdist(representative, representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.pairwise_conditioner(pair, self.dist_bin_pairwise_embed(distogram))

        token_mask = token_mask.repeat_interleave(multiplicity, 0).bool()
        ligand_mask = ligand_mask.repeat_interleave(multiplicity, 0).bool() & token_mask
        receptor_mask = receptor_mask.repeat_interleave(multiplicity, 0).bool() & token_mask
        affinity_pair_mask = (
            ligand_mask.unsqueeze(-1) & receptor_mask.unsqueeze(-2)
            | receptor_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
            | ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
        )
        _, pair = self.pairformer_stack(None, pair, pair_mask=affinity_pair_mask)
        return self.output_head(pair, affinity_pair_mask)


class Boltz2AffinityOutputHead(nn.Module):
    r"""
    Project pooled affinity pair features to value and binder-class logits.

    Args:
        pair_size: Width of the affinity pair representation being pooled.
        single_size: Hidden width of the affinity value / score MLPs.
    """

    def __init__(self, pair_size: int = 128, single_size: int = 384) -> None:
        super().__init__()
        self.pair_feature_mlp = nn.Sequential(
            nn.Linear(pair_size, pair_size),
            nn.ReLU(),
            nn.Linear(pair_size, single_size),
            nn.ReLU(),
        )
        self.affinity_value_mlp = nn.Sequential(
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, 1),
        )
        self.affinity_score_mlp = nn.Sequential(
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, 1),
        )
        self.binder_logit_projection = nn.Linear(1, 1)

    def forward(self, pair: Tensor, affinity_pair_mask: Tensor) -> dict[str, Tensor]:
        r"""
        Mean-pool receptor/ligand and ligand/ligand pair states, excluding the diagonal.

        Args:
            pair: Refined affinity pair representation ``(B*multiplicity, N, N, pair_size)``.
            affinity_pair_mask: Boolean receptor/ligand pair mask ``(B*multiplicity, N, N)``.

        Returns:
            Dict with ``affinity_pred_value`` ``(B*multiplicity, 1)`` and ``affinity_logits_binary``
            ``(B*multiplicity, 1)``.
        """

        num_tokens = affinity_pair_mask.shape[-1]
        non_diagonal = ~torch.eye(num_tokens, device=affinity_pair_mask.device, dtype=torch.bool).unsqueeze(0)
        pool_mask = (affinity_pair_mask.bool() & non_diagonal).to(dtype=pair.dtype).unsqueeze(-1)
        pooled = torch.sum(pair * pool_mask, dim=(-3, -2)) / (torch.sum(pool_mask, dim=(-3, -2)) + 1e-7)
        pooled = self.pair_feature_mlp(pooled)
        affinity_pred_value = self.affinity_value_mlp(pooled).reshape(-1, 1)
        affinity_pred_score = self.affinity_score_mlp(pooled).reshape(-1, 1)
        affinity_logits_binary = self.binder_logit_projection(affinity_pred_score).reshape(-1, 1)
        return {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }


class Boltz2ConfidenceHead(nn.Module):
    r"""
    Score predicted structures with per-token pLDDT / resolved / PAE / PDE confidence logits.

    Rebuilds the pair representation from relative-position, bond, contact, and distogram features, injects the
    trunk single, refines with a Pairformer stack, and projects to pLDDT / resolved / PAE / PDE logits.

    Args:
        single_size: Width of the trunk single channel.
        pair_size: Width of the pair representation.
        num_blocks: Number of Pairformer blocks in the head.
        num_pair_attention_heads: Head count for the pair-biased attention.
        num_distogram_bins: Number of distogram bins used to bin representative-atom distances.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        relative_position_size: Width of the relative-position feature vector.
        num_bond_types: Bond-type vocabulary size.
        conditioning_cutoff_min: Lower distance cutoff for contact conditioning, in angstroms.
        conditioning_cutoff_max: Upper distance cutoff for contact conditioning, in angstroms.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 8,
        num_pair_attention_heads: int = 16,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        relative_position_size: int = 139,
        num_bond_types: int = 7,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)

        self.input_single_norm = LayerNorm(single_size, operator_backend=operator_backend)
        self.single_norm = LayerNorm(single_size, operator_backend=operator_backend)
        self.pair_norm = LayerNorm(pair_size, operator_backend=operator_backend)
        self.input_single_projection = nn.Linear(single_size, single_size, bias=False)
        self.single_to_pair_row = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_col = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)

        self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(num_bond_types, pair_size)
        self.contact_conditioning = Boltz2ContactConditioning(
            pair_size, cutoff_min=conditioning_cutoff_min, cutoff_max=conditioning_cutoff_max
        )

        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            num_pair_attention_heads=num_pair_attention_heads,
            operator_backend=operator_backend,
        )
        self.output_head = Boltz2ConfidenceOutputHead(
            single_size, pair_size, num_plddt_bins=num_plddt_bins, num_pae_bins=num_pae_bins, num_pde_bins=num_pde_bins
        )

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        residue_index: Tensor,
        chain_index: Tensor,
        entity_index: Tensor,
        symmetry_index: Tensor,
        token_index: Tensor,
        token_bonds: Tensor,
        type_bonds: Tensor,
        contact_conditioning: Tensor,
        contact_threshold: Tensor,
        token_to_rep_atom: Tensor,
        token_mask: Tensor,
        cyclic_period: Tensor | None = None,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        r"""
        Score predicted coords: rebuild the pair representation (relative-position, bonds, contact, and a
        binned distogram of representative-atom distances), refine with a Pairformer stack, and project to logits.

        Args:
            single_inputs: Trunk input single ``(B, N, single_size)``.
            single: Trunk refined single ``(B, N, single_size)``.
            pair: Trunk pair ``(B, N, N, pair_size)``.
            coordinates: Sampled atom coordinates ``(B*multiplicity, M, 3)``.
            residue_index, chain_index, entity_index, symmetry_index, token_index:
                Relative-position features, each ``(B, N)``.
            token_bonds: Token covalent adjacency ``(B, N, N)``.
            type_bonds: Token bond-type ids ``(B, N, N)``.
            contact_conditioning: Contact-class one-hot ``(B, N, N, num_classes)``.
            contact_threshold: Contact thresholds ``(B, N, N)``.
            token_to_rep_atom: Representative atom index per token ``(B, N)``.
            token_mask: Boolean token validity mask ``(B, N)``.
            cyclic_period: Optional cyclic-position periods ``(B, N)``.
            multiplicity: Number of samples per batch element; trunk tensors are interleaved to ``B*multiplicity``.

        Returns:
            Logit dict from [`Boltz2ConfidenceOutputHead`][ofoldx.models.boltz2.model.Boltz2ConfidenceOutputHead]:
            ``pae_logits``,
            ``pde_logits``, ``plddt_logits``, ``resolved_logits``.
        """

        single_inputs = self.input_single_norm(single_inputs)
        single = self.single_norm(single)
        single = single + self.input_single_projection(single_inputs)
        pair = self.pair_norm(pair)

        relative_features = generate_relative_position_features(
            residue_index=residue_index,
            chain_index=chain_index,
            entity_index=entity_index,
            symmetry_index=symmetry_index,
            token_index=token_index,
            cyclic_period=cyclic_period,
            dtype=single.dtype,
        )
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds.unsqueeze(-1).to(pair.dtype))
        pair = pair + self.token_bond_type_embedding(type_bonds.long())
        pair = pair + self.contact_conditioning(contact_conditioning.to(pair.dtype), contact_threshold.to(pair.dtype))

        single = single.repeat_interleave(multiplicity, 0)
        pair = (
            pair
            + self.single_to_pair_row(single_inputs).unsqueeze(-2)
            + self.single_to_pair_col(single_inputs).unsqueeze(-3)
        )
        pair = pair + self.single_to_pair_product_out(
            self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
            * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        )
        pair = pair.repeat_interleave(multiplicity, 0)

        rep_atom_index = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        representative = coordinates.gather(1, rep_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1]))
        distances = torch.cdist(representative, representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        token_mask = token_mask.repeat_interleave(multiplicity, 0)
        pair_mask = pair_mask_from_token_mask(token_mask)
        single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        chain_index = chain_index.repeat_interleave(multiplicity, 0)
        return self.output_head(single, pair, chain_index)


class Boltz2ConfidenceOutputHead(nn.Module):
    r"""
    Project the confidence single/pair representations to pLDDT / resolved / PAE / PDE logits.

    Args:
        single_size: Width of the confidence single channel.
        pair_size: Width of the confidence pair channel.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
    """

    def __init__(
        self, single_size: int, pair_size: int, num_plddt_bins: int = 50, num_pae_bins: int = 64, num_pde_bins: int = 64
    ) -> None:
        super().__init__()
        self.pae_intra_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pae_inter_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_intra_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.pde_inter_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.plddt_projection = nn.Linear(single_size, num_plddt_bins, bias=False)
        self.resolved_projection = nn.Linear(single_size, 2, bias=False)

    def forward(self, single: Tensor, pair: Tensor, chain_index: Tensor) -> dict[str, Tensor]:
        r"""
        Project the confidence single and pair representations to error logits, selecting intra- vs inter-chain
        projections per token pair via ``chain_index``. PDE uses the symmetrized pair.

        Args:
            single: Confidence single representation ``(B, N, single_size)``.
            pair: Confidence pair representation ``(B, N, N, pair_size)``.
            chain_index: Chain ids per token ``(B, N)``, used to pick intra- vs inter-chain projections.

        Returns:
            Mapping containing PAE and PDE pair logits, per-token pLDDT logits, and per-token
            resolved-state logits.
        """

        same_chain = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).float().unsqueeze(-1)
        different_chain = 1.0 - same_chain
        pae_logits = self.pae_intra_projection(pair) * same_chain + self.pae_inter_projection(pair) * different_chain
        symmetric = pair + pair.transpose(-3, -2)
        pde_logits = (
            self.pde_intra_projection(symmetric) * same_chain + self.pde_inter_projection(symmetric) * different_chain
        )
        return {
            "pae_logits": pae_logits,
            "pde_logits": pde_logits,
            "plddt_logits": self.plddt_projection(single),
            "resolved_logits": self.resolved_projection(single),
        }


class Boltz2BFactorHead(nn.Module):
    r"""
    Predict per-token B-factor logits from the trunk single representation.

    Args:
        single_size: Width of the trunk single channel.
        num_bins: Number of B-factor output bins.
    """

    def __init__(self, single_size: int = 384, num_bins: int = 64) -> None:
        super().__init__()
        self.projection = nn.Linear(single_size, num_bins)

    def forward(self, single: Tensor) -> Tensor:
        r"""
        Project the trunk single to per-token B-factor logits.

        Args:
            single: Trunk single representation ``(B, N, single_size)``.

        Returns:
            Per-token B-factor logits ``(B, N, num_bins)``.
        """

        return self.projection(single)


def _decode_boltz2_fold_result(output: Boltz2Output, input: Boltz2Input) -> FoldResult:
    r"""Decode one raw [`Boltz2Output`][ofoldx.models.boltz2.model.Boltz2Output] into a ``FoldResult``.

    Maps model atom order back onto the system's complete heavy-atom sites to build one ``Structure`` per
    sample, and lifts the confidence head's per-token pLDDT/resolved metrics, token-pair PAE/PDE, and the
    per-sample pTM/ipTM ranking scores onto the canonical value carriers aligned to the layout's token axis.
    """

    layout = cast(Boltz2Layout, input.layout)
    token_count = len(layout.token_sites)
    atom_count = len(layout.atom_sites)

    complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    model_to_complete = np.asarray([complete_index[site] for site in layout.atom_sites], dtype=np.intp)

    coordinates = output.coordinates[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
    num_samples = coordinates.shape[0]
    # Each polymer token_site is already a ``ComponentReference`` (``(instance_id, monomer_index)``) and each
    # atomized-ligand token_site is an ``AtomSite`` -- both are valid ``MetricReference`` values, so the token
    # axis is the layout's token_sites verbatim (mirrors the Boltz-1 fold decoder).
    token_axis = layout.token_sites

    # Per-token pLDDT fraction (0-1) and resolved probability aligned to the token axis.
    plddt_fraction = (output.plddt[0, :, :token_count] / 100.0).to(device="cpu", dtype=torch.float32)
    resolved_probability = (
        cast(Tensor, output.resolved_logits)[0, :, :token_count].float().softmax(dim=-1)[..., 1].cpu()
    )

    # Expected PAE/PDE in angstrom (half-bin centers over 0-32, matching _boltz2_ptm_scores).
    pae = _boltz2_expected_angstrom(cast(Tensor, output.pae_logits)[0, :, :token_count, :token_count].float().cpu())
    pde = _boltz2_expected_angstrom(cast(Tensor, output.pde_logits)[0, :, :token_count, :token_count].float().cpu())

    # pTM / ipTM metrics over the flattened sample axis (B == 1); reuse the estimator already used by
    # _fold/_score.
    ptm_scores = _boltz2_ptm_scores(
        cast(Tensor, output.pae_logits).flatten(0, 1),
        coordinates=output.coordinates.flatten(0, 1),
        input=input,
        multiplicity=num_samples,
    )
    ptm = ptm_scores["ptm"].reshape(num_samples).float().cpu()
    iptm = ptm_scores["iptm"].reshape(num_samples).float().cpu()
    ligand_iptm = ptm_scores["ligand_iptm"].reshape(num_samples).float().cpu()
    protein_iptm = ptm_scores["protein_iptm"].reshape(num_samples).float().cpu()

    # Interface confidence uses iPTM unless every iPTM value is zero, in which case it falls back to pTM.
    # The confidence score combines that interface term with complex pLDDT on the 0-1 scale.
    token_mask = input.token_mask[0, :token_count].to(device="cpu", dtype=torch.float32)
    complex_plddt = (plddt_fraction * token_mask).sum(dim=-1) / token_mask.sum().clamp_min(1.0)
    interface = ptm if torch.allclose(iptm, torch.zeros_like(iptm)) else iptm
    confidence_score = (4.0 * complex_plddt + interface) / 5.0

    predictions: list[FoldPrediction] = []
    for sample_index in range(num_samples):
        complete_coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
        complete_coordinates[model_to_complete] = coordinates[sample_index]
        coordinate_mask[model_to_complete] = True
        structure = Structure(
            system=layout.system,
            sites=tuple(complete_sites),
            coordinates=complete_coordinates,
            coordinate_mask=coordinate_mask,
        )
        predictions.append(
            FoldPrediction(
                structure=structure,
                scores={
                    "plddt": Score(float(complex_plddt[sample_index]) * 100.0, preferred="higher"),
                    "ptm": Score(float(ptm[sample_index]), preferred="higher"),
                    "iptm": Score(float(iptm[sample_index]), preferred="higher"),
                    "ligand_iptm": Score(float(ligand_iptm[sample_index]), preferred="higher"),
                    "protein_iptm": Score(float(protein_iptm[sample_index]), preferred="higher"),
                    "confidence_score": Score(float(confidence_score[sample_index]), preferred="higher"),
                },
                metrics={
                    "plddt": Metric(values=plddt_fraction[sample_index].numpy() * 100.0, axis=token_axis),
                    "resolved_probability": Metric(values=resolved_probability[sample_index].numpy(), axis=token_axis),
                },
                pair_metrics={
                    "pae": PairMetric(
                        values=pae[sample_index].numpy(),
                        row_axis=token_axis,
                        column_axis=token_axis,
                        unit="angstrom",
                    ),
                    "pde": PairMetric(
                        values=pde[sample_index].numpy(),
                        row_axis=token_axis,
                        column_axis=token_axis,
                        unit="angstrom",
                    ),
                },
            )
        )
    return FoldResult(predictions=tuple(predictions))


def _boltz2_expected_angstrom(logits: Tensor) -> Tensor:
    r"""Softmax-expected error in angstrom over half-bin centers spanning ``0--32``."""

    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    centers = torch.arange(start=0.5 * bin_width, end=32.0, step=bin_width, device=logits.device, dtype=logits.dtype)
    return (logits.softmax(dim=-1) * centers).sum(dim=-1)


def _boltz2_ptm_scores(
    pae_logits: Tensor,
    *,
    coordinates: Tensor,
    input: Boltz2Input,
    multiplicity: int,
) -> dict[str, Tensor]:
    r"""Derive the pTM and interface-pTM scores from PAE logits."""

    token_mask = input.token_mask.repeat_interleave(multiplicity, 0).float()
    frame_mask = _boltz2_predicted_frame_mask(
        coordinates,
        input=input,
        multiplicity=multiplicity,
    ).float()
    chain_index = input.chain_index.repeat_interleave(multiplicity, 0)
    pair_mask_ptm = frame_mask[:, :, None] * token_mask[:, None, :] * token_mask[:, :, None]
    pair_mask_iptm = pair_mask_ptm * (chain_index[:, None, :] != chain_index[:, :, None]).to(pae_logits)

    num_bins = pae_logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_value = torch.arange(
        start=0.5 * bin_width,
        end=32.0,
        step=bin_width,
        device=pae_logits.device,
    )
    num_residues = token_mask.sum(dim=-1, keepdim=True)
    tm_expected = tm_weight_from_pae(pae_logits, num_residues.unsqueeze(-1), bin_centers=pae_value)

    def aggregate(mask: Tensor) -> Tensor:
        return torch.max(
            (tm_expected * mask).sum(dim=-1) / (mask.sum(dim=-1) + torch.finfo(tm_expected.dtype).tiny),
            dim=1,
        ).values

    molecule_type = input.molecule_type.repeat_interleave(multiplicity, 0)
    is_ligand = (molecule_type == 3).float()
    is_protein = (molecule_type == 0).float()
    ligand_mask = pair_mask_iptm * (
        is_ligand[:, :, None] * is_protein[:, None, :] + is_protein[:, :, None] * is_ligand[:, None, :]
    )
    protein_mask = pair_mask_iptm * (is_protein[:, :, None] * is_protein[:, None, :])
    return {
        "ptm": aggregate(pair_mask_ptm),
        "iptm": aggregate(pair_mask_iptm),
        "ligand_iptm": aggregate(ligand_mask),
        "protein_iptm": aggregate(protein_mask),
    }


def _boltz2_predicted_frame_mask(
    coordinates: Tensor,
    *,
    input: Boltz2Input,
    multiplicity: int,
) -> Tensor:
    r"""Return the exact predicted-frame validity mask used by Boltz-2 pTM."""

    if input.frame_atom_index is None:
        raise ValueError("Boltz-2 confidence requires token frame atom indices")
    frame_index = input.frame_atom_index.repeat_interleave(multiplicity, 0).to(
        device=coordinates.device,
        dtype=torch.long,
    )
    molecule_type = input.molecule_type.to(device=coordinates.device)
    chain_index = input.chain_index.to(device=coordinates.device)
    token_mask = input.token_mask.to(device=coordinates.device, dtype=torch.bool)
    atom_mask = input.atom_mask.to(device=coordinates.device, dtype=torch.bool)
    atom_to_token = input.atom_to_token.to(device=coordinates.device, dtype=torch.long)
    atom_chain = chain_index.gather(1, atom_to_token)

    for batch_index in range(input.token_mask.shape[0]):
        ligand_chains = torch.unique(
            chain_index[batch_index][token_mask[batch_index] & (molecule_type[batch_index] == 3)]
        )
        for chain in ligand_chains:
            chain_tokens = torch.nonzero(
                token_mask[batch_index] & (molecule_type[batch_index] == 3) & (chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            chain_atoms = torch.nonzero(
                atom_mask[batch_index] & (atom_chain[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            if len(chain_atoms) < 3:
                continue
            for sample_index in range(multiplicity):
                row = batch_index * multiplicity + sample_index
                distances = torch.cdist(coordinates[row, chain_atoms], coordinates[row, chain_atoms])
                nearest = distances.argsort(dim=-1)[:, :3]
                nearest_atoms = chain_atoms[nearest]
                frame_index[row, chain_tokens] = nearest_atoms[:, (1, 0, 2)]

    flat_index = frame_index.reshape(frame_index.shape[0], -1)
    frames = coordinates.gather(1, flat_index.unsqueeze(-1).expand(-1, -1, 3))
    frames = frames.reshape(*frame_index.shape, 3)
    first = frames[..., 1, :] - frames[..., 0, :]
    second = frames[..., 1, :] - frames[..., 2, :]
    first_norm = torch.linalg.norm(first, dim=-1)
    second_norm = torch.linalg.norm(second, dim=-1)
    first_direction = first / (first_norm.unsqueeze(-1) + 1e-6)
    second_direction = second / (second_norm.unsqueeze(-1) + 1e-6)
    cosine = (first_direction * second_direction).sum(dim=-1)
    valid_geometry = (cosine.abs() < 0.9063) & (first_norm > 1e-2) & (second_norm > 1e-2)
    return valid_geometry & token_mask.repeat_interleave(multiplicity, 0)


def _unflatten_sample_axis(tensor: Tensor, batch_size: int, num_samples: int) -> Tensor:
    r"""
    Split a flattened leading axis ``B*S`` back into an explicit ``(B, S, ...)`` sample axis.

    Args:
        tensor: Tensor whose leading dimension is ``batch_size * num_samples``.
        batch_size: Batch size ``B``.
        num_samples: Sample count ``S``.
    Returns:
        The tensor reshaped to ``(B, S, ...)``.
    """

    return tensor.reshape(batch_size, num_samples, *tensor.shape[1:])


# =====================================================================================================
# Boltz-2 tensor boundary
#
# This section anchors the Boltz-2 tensor boundary in one place, matching the
# AlphaFold 2, AlphaFold 3, and Boltz-1 verticals: the pure-tensor
# input, the raw output, and the canonical identity layout.
# =====================================================================================================


@dataclass(frozen=True)
class Boltz2Layout:
    r"""Canonical molecular identities aligned to one Boltz-2 tensor layout."""

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    affinity_instance_id: str | None = None


@dataclass(kw_only=True)
class Boltz2Input:
    r"""
    Input tensors consumed by Boltz2Model.

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
        residue_type: Query residue type one-hot features.
        msa_profile: Single-sequence MSA profile features.
        deletion_mean: Per-token deletion mean features.
        token_bonds: Token-level covalent adjacency.
        type_bonds: Token-level bond type ids.
        contact_conditioning: Pairwise contact-conditioning class one-hot features.
        contact_threshold: Pairwise contact threshold values.
        molecule_type: Molecule type ids in the model vocabulary.
        target_positions: Optional experimental atom coords for training.
        target_mask: Optional experimental-coordinate validity mask.
        frame_atom_index: Optional frame atom indices for frame-aligned losses.
        frame_mask: Optional frame validity mask.
        method_feature: Optional experimental-method ids.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        msa: Optional integer MSA token ids.
        has_deletion: Optional MSA deletion flags.
        deletion_value: Optional MSA deletion values.
        msa_paired: Optional paired-MSA flags.
        msa_mask: Optional MSA validity mask.
        template_residue_type: Optional template residue-type features.
        template_frame_rotations: Optional template frame rotations.
        template_frame_translations: Optional template frame translations.
        template_frame_mask: Optional template frame validity mask.
        template_beta_carbon_positions: Optional template beta-carbon positions.
        template_alpha_carbon_positions: Optional template alpha-carbon positions.
        template_beta_carbon_mask: Optional template beta-carbon validity mask.
        template_mask: Optional template validity mask.
        template_visibility_index: Optional template visibility-group indices.
        affinity_token_mask: Optional mask selecting the ligand tokens to score for affinity.
        affinity_mw: Optional molecular weight for the selected ligand, used only when affinity MW correction is on.
        affinity_msa_profile: Optional single-sequence profile used by the affinity heads.
        affinity_deletion_mean: Optional deletion mean used by the affinity heads.
        layout: Canonical molecular identity aligned to the atom and token tensors.
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
    residue_type: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    contact_conditioning: Tensor
    contact_threshold: Tensor
    molecule_type: Tensor
    target_positions: Tensor | None = None
    target_mask: Tensor | None = None
    frame_atom_index: Tensor | None = None
    frame_mask: Tensor | None = None
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None
    msa: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    template_residue_type: Tensor | None = None
    template_frame_rotations: Tensor | None = None
    template_frame_translations: Tensor | None = None
    template_frame_mask: Tensor | None = None
    template_beta_carbon_positions: Tensor | None = None
    template_alpha_carbon_positions: Tensor | None = None
    template_beta_carbon_mask: Tensor | None = None
    template_mask: Tensor | None = None
    template_visibility_index: Tensor | None = None
    affinity_token_mask: Tensor | None = None
    affinity_mw: Tensor | None = None
    affinity_msa_profile: Tensor | None = None
    affinity_deletion_mean: Tensor | None = None
    layout: Boltz2Layout | None = None


@dataclass
class _Boltz2TrunkState:
    r"""
    Trunk outputs consumed by the diffusion / confidence heads.

    Args:
        single_inputs: Initial token single representations.
        single: Final token single representation.
        pair: Final token pair representation.
        distogram_logits: Trunk distogram logits.
        relative_position_encoding: Relative-position pair features.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor


@dataclass(eq=False)
class Boltz2Output:
    r"""Tensor output of one Boltz-2 forward or sampling pass."""

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor | None = None
    pae_logits: Tensor | None = None
    pde_logits: Tensor | None = None
    resolved_logits: Tensor | None = None
    bfactor_logits: Tensor | None = None
    augmented_target_coordinates: Tensor | None = None
    noise_level: Tensor | None = None

    @property
    def plddt(self) -> Tensor:
        r"""Expected pLDDT on the conventional 0--100 scale.

        The Boltz-2 confidence head bins pLDDT on the ``0--1`` fraction; the softmax-expected bin
        value is rescaled by ``100`` to the shared ``0--100`` convention.
        """

        if self.plddt_logits is None:
            raise ValueError("Boltz2Output.plddt requires confidence logits")
        num_bins = self.plddt_logits.shape[-1]
        centers = (
            torch.arange(num_bins, device=self.plddt_logits.device, dtype=self.plddt_logits.dtype) + 0.5
        ) / num_bins
        return (self.plddt_logits.softmax(dim=-1) * centers).sum(dim=-1) * 100.0
