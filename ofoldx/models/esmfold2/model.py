r"""
ESMFold2 structure-prediction and binder-design models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multimolecule import EsmCConfig, EsmCForMaskedLM, EsmCModel
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ...data.evidence import MSA, MSACollection
from ...data.molecular import AtomSite, MolecularSystem
from ...data.structure import Structure
from ...model import Model
from ...modules.adaptive_norm import AdaptiveLayerNorm
from ...modules.backend import LayerNorm
from ...modules.backend.atom_window import atom_to_token_mean as dispatch_atom_to_token_mean
from ...modules.backend.atom_window import token_to_atom_broadcast as dispatch_token_to_atom_broadcast
from ...modules.backend.attention import projected_attention
from ...modules.backend.outer_product_mean import outer_product_mean as dispatch_outer_product_mean
from ...modules.backend.pair_weighted_average import pair_weighted_average as dispatch_pair_weighted_average
from ...modules.backend.resolve import resolve_tensor_backend
from ...modules.backend.transition import transition as dispatch_transition
from ...modules.backend.triangle_multiplication import triangle_multiplication as dispatch_triangle_multiplication
from ...modules.feed_forward import SwiGluFeedForward
from ...modules.transition import ConditionedTransition
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .config import EsmFold2Config, EsmFold2ExperimentalConfig


class EsmFold2Model(Model):
    r"""
    Predict protein structures with ESMFold 2.

    Args:
        config: Model architecture and inference configuration.
        esmc: Frozen ESMC language-model trunk. ``None`` initializes one from
            ``config.esmc``.
    """

    config_class = EsmFold2Config
    config: EsmFold2Config

    def __init__(self, config: EsmFold2Config | None = None, *, esmc: EsmCModel | None = None) -> None:
        config = EsmFold2Config() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self.language_model_mask_probability = config.language_model_mask_probability
        self.esmc_bos_token_id = config.esmc.bos_token_id
        self.esmc_eos_token_id = config.esmc.eos_token_id
        self.esmc_mask_token_id = config.esmc.mask_token_id
        self.language_model_dropout = config.language_model_encoder.dropout
        self.per_loop_language_model_dropout = config.language_model_encoder.per_loop_dropout
        self.msa_encoder_overwrite = config.msa_encoder_overwrite
        input_size = config.input_embeddings.input_size
        pair_size = config.pair_size

        self.input_embeddings = EsmFold2InputEmbeddings(config)
        self.left_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.right_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.relative_position_embedder = EsmFold2RelativePositionEmbedder(
            config.num_relative_residue_bins,
            config.num_relative_chain_bins,
            pair_size,
        )
        self.bond_projection = nn.Linear(1, pair_size, bias=False)
        if esmc is None:
            esmc = EsmCModel(EsmCConfig(**dict(config.esmc.dict())))
        self.esmc = esmc
        self.esmc.requires_grad_(False)
        self.language_model_pair_encoder = EsmFold2LanguageModelPairEncoder(
            pair_size,
            config.esmc.hidden_size,
            config.esmc.num_hidden_layers,
        )

        if config.language_model_encoder.enabled:
            self.language_model_encoder: EsmFold2FoldingTrunk | None = EsmFold2FoldingTrunk(
                config.language_model_encoder.num_blocks,
                pair_size,
                expansion=4,
            )
        else:
            self.language_model_encoder = None

        msa_config = config.msa_encoder
        self.msa_encoder: EsmFold2MsaEncoder | None = None
        if msa_config.enabled:
            self.msa_encoder = EsmFold2MsaEncoder(
                msa_size=msa_config.msa_size,
                pair_size=pair_size,
                input_size=input_size,
                hidden_size=msa_config.hidden_size,
                num_blocks=msa_config.num_blocks,
                num_msa_attention_heads=msa_config.num_msa_attention_heads,
                msa_head_dim=msa_config.msa_head_dim,
            )

        folding_trunk_config = config.folding_trunk
        self.folding_trunk = EsmFold2FoldingTrunk(
            folding_trunk_config.num_blocks,
            pair_size,
            expansion=4,
        )
        self.parcae_input_norm = LayerNorm(pair_size)
        self.parcae_log_decay = nn.Parameter(torch.zeros(pair_size))
        parcae_decay_init = math.sqrt(1.0 / 5.0)
        parcae_delta_init = -math.log(parcae_decay_init)
        self.parcae_log_delta = nn.Parameter(
            torch.full(
                (pair_size,),
                parcae_delta_init + math.log(-math.expm1(-parcae_delta_init)),
                dtype=torch.float32,
            )
        )
        self.parcae_input_matrix = nn.Parameter(torch.eye(pair_size))
        self.parcae_readout = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.eye_(self.parcae_readout.weight)
        self.parcae_coda = EsmFold2FoldingTrunk(
            config.parcae.num_coda_blocks,
            pair_size,
            expansion=4,
        )

        self.diffusion_structure_head = EsmFold2DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(pair_size, config.diffusion_structure_head.num_distogram_bins, bias=True)
        self.confidence_head = EsmFold2ConfidenceHead(config)
        self.post_init()

    def set_chunk_size(self, chunk_size: int | None) -> None:
        r"""
        Set the chunk size for memory-intensive folding operations.

        Args:
            chunk_size: Number of rows processed per chunk. ``None`` disables chunking.
        """

        self.folding_trunk.set_chunk_size(chunk_size)
        if self.language_model_encoder is not None:
            self.language_model_encoder.set_chunk_size(chunk_size)
        self.parcae_coda.set_chunk_size(chunk_size)
        self.confidence_head.set_chunk_size(chunk_size)
        if self.msa_encoder is not None:
            self.msa_encoder.set_chunk_size(chunk_size)

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        delta = F.softplus(self.parcae_log_delta)
        decay = torch.exp(-delta * torch.exp(self.parcae_log_decay))
        input_matrix = delta[:, None] * self.parcae_input_matrix
        return decay, input_matrix

    def _init_pair_state(self, reference: Tensor, generator: torch.Generator | None) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * reference.shape[-1]))
        state = torch.empty_like(reference, dtype=torch.float32)
        nn.init.trunc_normal_(
            state,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
        return state.to(dtype=reference.dtype)

    def _run_one_loop(
        self,
        *,
        pair_state: Tensor,
        initial_pair: Tensor,
        language_model_pair: Tensor,
        pair_mask: Tensor,
        decay: Tensor,
        input_matrix: Tensor,
        token_mask: Tensor,
        total_steps: int,
        msa_tokens: Tensor | None,
        msa_attention_mask: Tensor | None,
        has_deletion: Tensor | None,
        deletion_value: Tensor | None,
        single_inputs: Tensor,
        msa_max_depth: int | None,
        msa_subsample_at_inference: bool,
        language_model_dropout: float | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        # Keep this loop in a helper so each iteration's pair activations can be
        # released before the distogram and sampling heads run.
        if language_model_dropout is None:
            dropout = self.language_model_dropout
            per_loop_language_model_dropout = self.per_loop_language_model_dropout and dropout > 0.0
        else:
            dropout = language_model_dropout
            per_loop_language_model_dropout = dropout > 0.0

        for _ in range(total_steps):
            if per_loop_language_model_dropout:
                loop_language_model_pair = _dropout(language_model_pair, dropout, generator)
            else:
                loop_language_model_pair = language_model_pair

            refined_language_model_pair: Tensor | None = None
            if self.language_model_encoder is not None:
                refined_language_model_pair = self.language_model_encoder(
                    loop_language_model_pair.to(initial_pair.dtype),
                    pair_attention_mask=pair_mask,
                )

            injected_pair = initial_pair
            if self.language_model_encoder is None:
                injected_pair = injected_pair + loop_language_model_pair.to(injected_pair.dtype)

            if self.msa_encoder is not None and msa_tokens is not None:
                # Fresh row subsample each iteration (column mask was applied
                # once in forward, before this loop).
                (
                    sampled_msa_tokens,
                    sampled_msa_mask,
                    sampled_has_deletion,
                    sampled_deletion_value,
                ) = _maybe_subsample_msa(
                    msa_tokens,
                    msa_attention_mask,
                    has_deletion,
                    deletion_value,
                    max_depth=msa_max_depth,
                    enabled=msa_subsample_at_inference,
                    generator=generator,
                )
                batch_size, msa_depth, sequence_length = sampled_msa_tokens.shape
                msa_residue_features = F.one_hot(
                    sampled_msa_tokens.permute(0, 2, 1).long(),
                    num_classes=_NUM_RESIDUE_TYPES,
                ).float()
                msa_attention = (
                    sampled_msa_mask.permute(0, 2, 1).float()
                    if sampled_msa_mask is not None
                    else token_mask[:, :, None].expand(-1, -1, msa_depth).float()
                )
                # The bias-free MSA projection requires zeroed padding.
                msa_residue_features = msa_residue_features * msa_attention.unsqueeze(-1)
                has_deletion_features = (
                    sampled_has_deletion.permute(0, 2, 1).float()
                    if sampled_has_deletion is not None
                    else torch.zeros(batch_size, sequence_length, msa_depth, device=sampled_msa_tokens.device)
                )
                deletion_features = (
                    sampled_deletion_value.permute(0, 2, 1).float()
                    if sampled_deletion_value is not None
                    else torch.zeros(batch_size, sequence_length, msa_depth, device=sampled_msa_tokens.device)
                )
                msa_pair_update = self.msa_encoder(
                    pair=injected_pair,
                    single_inputs=single_inputs,
                    msa_residue_features=msa_residue_features,
                    has_deletion=has_deletion_features,
                    deletion_value=deletion_features,
                    msa_attention_mask=msa_attention,
                ).to(injected_pair.dtype)
                injected_pair = msa_pair_update if self.msa_encoder_overwrite else (injected_pair + msa_pair_update)

            if refined_language_model_pair is not None:
                injected_pair = injected_pair + refined_language_model_pair.to(injected_pair.dtype)

            normalized_injected_pair = self.parcae_input_norm(injected_pair)
            pair_state = decay * pair_state + F.linear(
                normalized_injected_pair.to(pair_state.dtype),
                input_matrix,
            )
            pair_state = self.folding_trunk(pair_state, pair_attention_mask=pair_mask)

        return pair_state

    def _encode_inputs_and_trunk(
        self,
        *,
        token_index: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        symmetry_index: Tensor,
        entity_index: Tensor,
        residue_type_ids: Tensor,
        token_bonds: Tensor,
        token_mask: Tensor,
        reference_positions: Tensor,
        reference_elements: Tensor,
        reference_charges: Tensor,
        reference_atom_name_characters: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        deletion_mean: Tensor | None = None,
        msa_tokens: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        language_model_tokens: Tensor | None = None,
        language_model_mask: Tensor | None = None,
        token_to_esmc: Tensor | None = None,
        token_has_esmc: Tensor | None = None,
        esmc_hidden_states: Tensor | None = None,
        recycling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        r"""
        Run the shared trunk and emit single/pair representations.

        Both the training [`forward`][ofoldx.models.esmfold2.EsmFold2Model.forward] and the inference
        [`fold`][ofoldx.models.esmfold2.EsmFold2Model.fold] verb consume this: it returns the
        single (``single_inputs``) and pair (``pair_state``) trunk streams, plus auxiliary reference features and
        teacher-forced distogram logits.
        """

        total_steps = recycling_steps if recycling_steps is not None else self._recycling_steps
        if total_steps <= 0:
            raise ValueError("recycling_steps must be positive")

        residue_type_features = F.one_hot(residue_type_ids.long(), num_classes=_NUM_RESIDUE_TYPES).float()
        residue_type_features = residue_type_features * token_mask.unsqueeze(-1).float()

        if msa_tokens is not None:
            msa_residue_features = F.one_hot(msa_tokens.long(), num_classes=_NUM_RESIDUE_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_residue_features = msa_residue_features * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                msa_profile = msa_residue_features.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                msa_profile = msa_residue_features.mean(dim=1)
        else:
            msa_profile = residue_type_features

        if deletion_mean is None:
            deletion_mean = torch.zeros(
                residue_type_ids.shape[0], residue_type_ids.shape[1], device=residue_type_ids.device
            )

        reference_element_features = F.one_hot(reference_elements.long(), num_classes=_MAX_ATOMIC_NUMBER).float()
        reference_atom_name_character_features = F.one_hot(
            reference_atom_name_characters.long(), num_classes=_CHAR_VOCAB_SIZE
        ).float()
        # Bias-free downstream Linears require zeroed padding.
        atom_mask_f = atom_mask.float()
        reference_element_features = reference_element_features * atom_mask_f.unsqueeze(-1)
        reference_atom_name_character_features = reference_atom_name_character_features * atom_mask_f.unsqueeze(
            -1
        ).unsqueeze(-1)
        atom_to_token = atom_to_token * atom_mask.long()

        single_inputs = self.input_embeddings(
            residue_type_features=residue_type_features,
            msa_profile=msa_profile.float(),
            deletion_mean=deletion_mean.float(),
            reference_positions=reference_positions,
            atom_mask=atom_mask,
            reference_space_id=reference_space_id,
            reference_charges=reference_charges,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
        )

        initial_pair = self.left_pair_projection(single_inputs).unsqueeze(2) + self.right_pair_projection(
            single_inputs
        ).unsqueeze(1)

        relative_position_encoding = self.relative_position_embedder(
            residue_index=residue_index,
            chain_index=chain_index,
            symmetry_index=symmetry_index,
            entity_index=entity_index,
            token_index=token_index,
        )
        token_bonds_encoding = self.bond_projection(token_bonds.to(dtype=self.bond_projection.weight.dtype))
        initial_pair = initial_pair + relative_position_encoding + token_bonds_encoding

        if esmc_hidden_states is None:
            if (
                language_model_tokens is None
                or language_model_mask is None
                or token_to_esmc is None
                or token_has_esmc is None
            ):
                raise RuntimeError("ESMFold2 requires ESMC inputs or expert-provided esmc_hidden_states")
            esmc_hidden_states = _compute_esmc_hidden_states(
                self.esmc,
                language_model_tokens,
                language_model_mask,
                token_to_esmc,
                token_has_esmc,
                language_model_mask_probability=(
                    language_model_mask_probability
                    if language_model_mask_probability is not None
                    else self.language_model_mask_probability
                ),
                bos_token_id=self.esmc_bos_token_id,
                eos_token_id=self.esmc_eos_token_id,
                mask_token_id=self.esmc_mask_token_id,
                generator=generator,
            )
        language_model_pair = self.language_model_pair_encoder(esmc_hidden_states)
        del esmc_hidden_states

        pair_mask = token_mask[:, :, None].float() * token_mask[:, None, :].float()

        pair_state = self._init_pair_state(initial_pair, generator)

        decay, input_matrix = self._discretized_dynamics()
        decay = decay.view(1, 1, 1, -1).to(device=pair_state.device, dtype=pair_state.dtype)
        input_matrix = input_matrix.to(device=pair_state.device, dtype=pair_state.dtype)

        # Inference-time MSA diversity: column mask is applied once here
        # (shared across recycling loops); row subsampling is deferred to
        # per-iter inside _run_one_loop (fresh subset per loop).
        if self.msa_encoder is not None and msa_tokens is not None:
            msa_attention_mask = _maybe_apply_msa_column_masking(
                msa_attention_mask,
                rate=msa_column_mask_rate,
                generator=generator,
            )

        # Method call (not inline loop) frees per-iteration pair locals.
        pair_state = self._run_one_loop(
            pair_state=pair_state,
            initial_pair=initial_pair,
            language_model_pair=language_model_pair,
            pair_mask=pair_mask,
            decay=decay,
            input_matrix=input_matrix,
            token_mask=token_mask,
            total_steps=total_steps,
            msa_tokens=msa_tokens,
            msa_attention_mask=msa_attention_mask,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            single_inputs=single_inputs,
            msa_max_depth=msa_max_depth,
            msa_subsample_at_inference=msa_subsample_at_inference,
            language_model_dropout=language_model_dropout,
            generator=generator,
        )
        pair_state = self.parcae_readout(pair_state)
        pair_state = self.parcae_coda(pair_state, pair_attention_mask=pair_mask)
        del initial_pair, language_model_pair, decay, input_matrix

        head_dtype = self.distogram_head.weight.dtype
        pair_state = pair_state.to(dtype=head_dtype)
        single_inputs = single_inputs.to(dtype=head_dtype)
        relative_position_encoding = relative_position_encoding.to(dtype=head_dtype)
        token_bonds_encoding = token_bonds_encoding.to(dtype=head_dtype)
        with torch.autocast(device_type=pair_state.device.type, enabled=False):
            distogram_logits = self.distogram_head(pair_state + pair_state.transpose(-2, -3))

        auxiliary: dict[str, Tensor] = {
            "relative_position_encoding": relative_position_encoding,
            "token_bonds_encoding": token_bonds_encoding,
            "reference_element_features": reference_element_features,
            "reference_atom_name_character_features": reference_atom_name_character_features,
            "atom_to_token": atom_to_token,
            "distogram_logits": distogram_logits,
        }
        return single_inputs, pair_state, auxiliary

    def embeddings(
        self,
        input: EsmFold2Input,
        /,
        *,
        recycling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Compute token and pair embeddings before coordinate sampling.

        ``B`` is batch size and ``N`` is token count.

        Args:
            input: Processed molecular input.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token. ``None`` uses the configured value.
            language_model_dropout: Dropout probability for language-model pair features.
                ``None`` uses the configured value.
            msa_max_depth: Maximum number of MSA rows sampled per recurrent pass. ``None``
                keeps all rows.
            msa_column_mask_rate: Probability of masking a column in nonquery MSA rows.
            msa_subsample_at_inference: Whether to resample MSA rows on each recurrent pass
                during inference.
            generator: Random generator used for token masking, dropout, and MSA sampling.

        Returns:
            A tuple ``(token, pair)`` with shapes ``(B, N, C_s)`` and
            ``(B, N, N, C_z)``.
        """

        token_embeddings, pair_embeddings, _ = self._encode_inputs_and_trunk(
            token_index=input.token_index,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            symmetry_index=input.symmetry_index,
            entity_index=input.entity_index,
            residue_type_ids=input.residue_type_ids,
            token_bonds=input.token_bonds,
            token_mask=input.token_mask,
            reference_positions=input.reference_positions,
            reference_elements=input.reference_elements,
            reference_charges=input.reference_charges,
            reference_atom_name_characters=input.reference_atom_name_characters,
            reference_space_id=input.reference_space_id,
            atom_mask=input.atom_mask,
            atom_to_token=input.atom_to_token,
            deletion_mean=input.deletion_mean,
            msa_tokens=input.msa_tokens,
            has_deletion=input.has_deletion,
            deletion_value=input.deletion_value,
            msa_attention_mask=input.msa_attention_mask,
            language_model_tokens=input.language_model_tokens,
            language_model_mask=input.language_model_mask,
            token_to_esmc=input.token_to_esmc,
            token_has_esmc=input.token_has_esmc,
            esmc_hidden_states=input.esmc_hidden_states,
            recycling_steps=recycling_steps,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            generator=generator,
        )
        return token_embeddings, pair_embeddings

    @torch.inference_mode()
    def _fold_tensors(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        symmetry_index: Tensor,
        entity_index: Tensor,
        molecule_type: Tensor,
        residue_type_ids: Tensor,
        token_bonds: Tensor,
        token_mask: Tensor,
        reference_positions: Tensor,
        reference_elements: Tensor,
        reference_charges: Tensor,
        reference_atom_name_characters: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_index: Tensor,
        deletion_mean: Tensor | None = None,
        msa_tokens: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        language_model_tokens: Tensor | None = None,
        language_model_mask: Tensor | None = None,
        token_to_esmc: Tensor | None = None,
        token_has_esmc: Tensor | None = None,
        esmc_hidden_states: Tensor | None = None,
        recycling_steps: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        num_samples = 1 if num_diffusion_samples is None else num_diffusion_samples

        single_inputs, pair, auxiliary = self._encode_inputs_and_trunk(
            token_index=token_index,
            residue_index=residue_index,
            chain_index=chain_index,
            symmetry_index=symmetry_index,
            entity_index=entity_index,
            residue_type_ids=residue_type_ids,
            token_bonds=token_bonds,
            token_mask=token_mask,
            reference_positions=reference_positions,
            reference_elements=reference_elements,
            reference_charges=reference_charges,
            reference_atom_name_characters=reference_atom_name_characters,
            reference_space_id=reference_space_id,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            deletion_mean=deletion_mean,
            msa_tokens=msa_tokens,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_attention_mask=msa_attention_mask,
            language_model_tokens=language_model_tokens,
            language_model_mask=language_model_mask,
            token_to_esmc=token_to_esmc,
            token_has_esmc=token_has_esmc,
            esmc_hidden_states=esmc_hidden_states,
            recycling_steps=recycling_steps,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            generator=generator,
        )
        relative_position_encoding = auxiliary["relative_position_encoding"]
        token_bonds_encoding = auxiliary["token_bonds_encoding"]
        reference_element_features = auxiliary["reference_element_features"]
        reference_atom_name_character_features = auxiliary["reference_atom_name_character_features"]
        atom_to_token = auxiliary["atom_to_token"]
        distogram_logits = auxiliary["distogram_logits"]

        with torch.autocast(device_type=reference_positions.device.type, enabled=False):
            structure_output = self.diffusion_structure_head.sample(
                pair=pair,
                single_inputs=single_inputs,
                relative_position_encoding=relative_position_encoding,
                reference_positions=reference_positions,
                reference_charges=reference_charges,
                atom_mask=atom_mask,
                reference_element_features=reference_element_features,
                reference_atom_name_character_features=reference_atom_name_character_features,
                reference_space_id=reference_space_id,
                atom_to_token=atom_to_token,
                token_mask=token_mask,
                num_diffusion_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                max_inference_sigma=max_inference_sigma,
                noise_scale=noise_scale,
                step_scale=step_scale,
                return_atom_representations=False,
                denoising_early_exit_rmsd=denoising_early_exit_rmsd,
                generator=generator,
            )

        coordinates = cast(Tensor, structure_output["coordinates"])
        output: dict[str, Tensor] = {"distogram_logits": distogram_logits}
        output["coordinates"] = coordinates

        with torch.autocast(device_type=coordinates.device.type, enabled=False):
            confidence_output = self.confidence_head(
                single_inputs=single_inputs,
                pair=pair,
                coordinates=coordinates,
                distogram_atom_index=distogram_atom_index,
                token_mask=token_mask,
                atom_to_token=atom_to_token,
                atom_mask=atom_mask,
                chain_index=chain_index,
                molecule_type=molecule_type,
                num_diffusion_samples=num_samples,
                relative_position_encoding=relative_position_encoding,
                token_bonds_encoding=token_bonds_encoding,
            )
        output.update(confidence_output)
        output["atom_mask"] = atom_mask
        output["residue_index"] = residue_index
        output["entity_index"] = entity_index
        return output

    def fold(
        self,
        input: EsmFold2Input,
        /,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
    ) -> FoldResult:
        r"""
        Predict sampled structures and confidence values for one processed system.

        Args:
            input: Processed input with batch size one.
            num_samples: Number of coordinate samples and returned predictions.
            generator: PyTorch random generator controlling stochastic inference.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token. ``None`` uses the configured value.
            language_model_dropout: Dropout probability for language-model pair features.
                ``None`` uses the configured value.
            msa_max_depth: Maximum number of MSA rows sampled per recurrent pass. ``None``
                keeps all rows.
            msa_column_mask_rate: Probability of masking a column in nonquery MSA rows.
            msa_subsample_at_inference: Whether to resample MSA rows on each recurrent pass
                during inference.
            max_inference_sigma: Largest diffusion sigma included in sampling. ``None``
                keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.
            denoising_early_exit_rmsd: Aligned RMSD threshold, in angstroms, for ending
                denoising early. ``None`` disables early exit.

        Returns:
            A [`FoldResult`][ofoldx.results.FoldResult] containing ``num_samples``
            predictions in sampling order. Coordinates and PAE values are in angstroms;
            pLDDT metrics use the model's confidence scale.
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if input.token_index.shape[0] != 1:
            raise ValueError("ESMFold2 fold accepts one canonical system per call")

        raw = self._fold_tensors(
            token_index=input.token_index,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            symmetry_index=input.symmetry_index,
            entity_index=input.entity_index,
            molecule_type=input.molecule_type,
            residue_type_ids=input.residue_type_ids,
            token_bonds=input.token_bonds,
            token_mask=input.token_mask,
            reference_positions=input.reference_positions,
            reference_elements=input.reference_elements,
            reference_charges=input.reference_charges,
            reference_atom_name_characters=input.reference_atom_name_characters,
            reference_space_id=input.reference_space_id,
            atom_mask=input.atom_mask,
            atom_to_token=input.atom_to_token,
            distogram_atom_index=input.distogram_atom_index,
            deletion_mean=input.deletion_mean,
            msa_tokens=input.msa_tokens,
            has_deletion=input.has_deletion,
            deletion_value=input.deletion_value,
            msa_attention_mask=input.msa_attention_mask,
            language_model_tokens=input.language_model_tokens,
            language_model_mask=input.language_model_mask,
            token_to_esmc=input.token_to_esmc,
            token_has_esmc=input.token_has_esmc,
            esmc_hidden_states=input.esmc_hidden_states,
            recycling_steps=recycling_steps,
            num_diffusion_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            denoising_early_exit_rmsd=denoising_early_exit_rmsd,
            generator=generator,
        )

        layout = input.layout
        atom_count = len(layout.atom_sites)
        raw_coordinates = raw["coordinates"][:, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        raw_plddt = raw["plddt"].to(device="cpu", dtype=torch.float32).numpy() * 100.0
        raw_pae = raw["pae"].to(device="cpu", dtype=torch.float32).numpy()
        complete_sites = tuple(site for site, atom in layout.system.iter_atoms() if atom.element != "H")
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray(
            [complete_index[site] for site in layout.atom_sites],
            dtype=np.intp,
        )
        # Chains carry chain_index 0..n-1 in system.instances order (processor), so the
        # instance ids are the row/column axis for the per-chain-pair confidence.
        chain_axis = tuple(layout.system.instances)
        predictions: list[FoldPrediction] = []
        for sample_index in range(num_samples):
            coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
            coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
            coordinates[model_to_complete] = raw_coordinates[sample_index]
            coordinate_mask[model_to_complete] = True
            structure = Structure(
                system=layout.system,
                sites=complete_sites,
                coordinates=coordinates,
                coordinate_mask=coordinate_mask,
            )
            scores = {
                "ptm": Score(float(raw["ptm"][sample_index].item()), preferred="higher"),
                "iptm": Score(float(raw["iptm"][sample_index].item()), preferred="higher"),
                "mean_plddt": Score(
                    float(raw["mean_plddt"][sample_index].item()) * 100.0,
                    preferred="higher",
                ),
                "complex_iplddt": Score(
                    float(raw["complex_iplddt"][sample_index].item()),
                    preferred="higher",
                ),
            }
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores=scores,
                    metrics={
                        "plddt": Metric(values=raw_plddt[sample_index], axis=layout.token_sites),
                        # Per-atom pLDDT is not recoverable from token pLDDT and is
                        # therefore preserved on the AtomSite axis.
                        "plddt_per_atom": Metric(
                            values=raw["plddt_per_atom"][sample_index, :atom_count].float().cpu().numpy() * 100.0,
                            axis=layout.atom_sites,
                        ),
                    },
                    pair_metrics={
                        "pae": PairMetric(
                            values=raw_pae[sample_index],
                            row_axis=layout.token_sites,
                            column_axis=layout.token_sites,
                            unit="angstrom",
                        ),
                        # Predicted distance error on the token axes.
                        "pde": PairMetric(
                            values=raw["pde"][sample_index].float().cpu().numpy(),
                            row_axis=layout.token_sites,
                            column_axis=layout.token_sites,
                            unit="angstrom",
                        ),
                        # Per-chain-pair interface confidence on the chain (instance_id) axis.
                        "chain_pair_iptm": PairMetric(
                            values=raw["chain_pair_iptm"][sample_index].float().cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                    },
                )
            )

        return FoldResult(predictions=tuple(predictions))

    def forward(
        self,
        input: EsmFold2Input,
        *,
        recycling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        generator: torch.Generator | None = None,
    ) -> EsmFold2Output:
        r"""
        Denoise explicit noisy atom coordinates in one differentiable step.

        ``B`` is batch size, ``N`` is token count, and ``A`` is atom count.

        Args:
            input: Processed input whose ``noisy_atom_coordinates`` have shape
                ``(B, A, 3)`` in angstroms and whose ``noise_level`` has shape ``(B,)``.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token. ``None`` uses the configured value.
            language_model_dropout: Dropout probability for language-model pair features.
                ``None`` uses the configured value.
            msa_max_depth: Maximum number of MSA rows sampled per recurrent pass. ``None``
                keeps all rows.
            msa_column_mask_rate: Probability of masking a column in nonquery MSA rows.
            msa_subsample_at_inference: Whether to resample MSA rows on each recurrent pass
                during inference.
            generator: Random generator used for token masking, dropout, and MSA sampling.

        Returns:
            An [`EsmFold2Output`][] with ``denoised_coordinates`` of shape
            ``(B, A, 3)`` in angstroms and ``distogram_logits`` of shape
            ``(B, N, N, D)``.
        """

        if input.noisy_atom_coordinates is None or input.noise_level is None:
            raise ValueError("ESMFold2 forward requires explicit noisy_atom_coordinates and noise_level")

        single_inputs, pair, auxiliary = self._encode_inputs_and_trunk(
            token_index=input.token_index,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            symmetry_index=input.symmetry_index,
            entity_index=input.entity_index,
            residue_type_ids=input.residue_type_ids,
            token_bonds=input.token_bonds,
            token_mask=input.token_mask,
            reference_positions=input.reference_positions,
            reference_elements=input.reference_elements,
            reference_charges=input.reference_charges,
            reference_atom_name_characters=input.reference_atom_name_characters,
            reference_space_id=input.reference_space_id,
            atom_mask=input.atom_mask,
            atom_to_token=input.atom_to_token,
            deletion_mean=input.deletion_mean,
            msa_tokens=input.msa_tokens,
            has_deletion=input.has_deletion,
            deletion_value=input.deletion_value,
            msa_attention_mask=input.msa_attention_mask,
            language_model_tokens=input.language_model_tokens,
            language_model_mask=input.language_model_mask,
            token_to_esmc=input.token_to_esmc,
            token_has_esmc=input.token_has_esmc,
            esmc_hidden_states=input.esmc_hidden_states,
            recycling_steps=recycling_steps,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            generator=generator,
        )
        atom_to_token = auxiliary["atom_to_token"]

        denoiser_output = self.diffusion_structure_head.denoiser(
            noisy_coordinates=input.noisy_atom_coordinates,
            noise_level=input.noise_level,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            atom_mask=input.atom_mask,
            reference_element_features=auxiliary["reference_element_features"],
            reference_atom_name_character_features=auxiliary["reference_atom_name_character_features"],
            reference_space_id=input.reference_space_id,
            atom_to_token=atom_to_token,
            single_inputs=single_inputs,
            pair=pair,
            relative_position_encoding=auxiliary["relative_position_encoding"],
            token_mask=input.token_mask,
            num_diffusion_samples=1,
        )
        denoised = cast(Tensor, denoiser_output["denoised_coordinates"])

        return EsmFold2Output(
            denoised_coordinates=denoised,
            distogram_logits=auxiliary["distogram_logits"],
        )


class EsmFold2ExperimentalModel(Model):
    r"""
    Evaluate the recurrent ESMFold 2 graph used for binder design.

    Args:
        config: Model architecture and inference configuration.
        esmc: Frozen ESMC trunk or masked-language model. The masked-language-model
            head is required when binder design uses a nonzero language-model weight.
    """

    config_class = EsmFold2ExperimentalConfig
    config: EsmFold2ExperimentalConfig

    def __init__(self, config: EsmFold2ExperimentalConfig | None = None, *, esmc: EsmCModel | None = None) -> None:
        config = EsmFold2ExperimentalConfig() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self.use_msa_features = config.use_msa_features
        self.language_model_dropout = config.language_model_dropout
        self.force_language_model_dropout = config.force_language_model_dropout
        self.esmc_bos_token_id = config.esmc.bos_token_id
        self.esmc_eos_token_id = config.esmc.eos_token_id
        self.esmc_mask_token_id = config.esmc.mask_token_id
        input_size = config.input_embeddings.input_size
        pair_size = config.pair_size

        self.input_embeddings = EsmFold2InputEmbeddings(config)
        self.left_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.right_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.relative_position_embedder = EsmFold2RelativePositionEmbedder(
            config.num_relative_residue_bins,
            config.num_relative_chain_bins,
            pair_size,
        )
        self.bond_projection = nn.Linear(1, pair_size, bias=False)

        self.esmc: EsmCModel | None = None
        self.esmc_lm_head: nn.Module | None = None
        if esmc is not None:
            self.load_esmc(esmc)
        self.language_model_pair_encoder = EsmFold2LanguageModelPairEncoder(
            pair_size,
            config.esmc.hidden_size,
            config.esmc.num_hidden_layers,
        )

        msa_config = config.msa_encoder
        self.msa_encoder: EsmFold2ExperimentalMsaEncoder | None = None
        if msa_config.enabled:
            self.msa_encoder = EsmFold2ExperimentalMsaEncoder(
                msa_size=msa_config.msa_size,
                pair_size=pair_size,
                input_size=input_size,
                hidden_size=msa_config.hidden_size,
                num_blocks=msa_config.num_blocks,
                num_msa_attention_heads=msa_config.num_msa_attention_heads,
                msa_head_dim=msa_config.msa_head_dim,
            )

        folding_trunk_config = config.folding_trunk
        self.folding_trunk = EsmFold2FoldingTrunk(
            folding_trunk_config.num_blocks,
            pair_size,
            expansion=4,
        )

        # Zero-initialized pair representation carried between recurrent trunk iterations.
        self.recycling_norm = LayerNorm(pair_size)
        self.recycling_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.recycling_projection.weight)

        self.diffusion_structure_head = EsmFold2DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(pair_size, config.diffusion_structure_head.num_distogram_bins, bias=True)
        # Distogram-only configurations omit the confidence head.
        self.confidence_head: EsmFold2ExperimentalConfidenceHead | None = (
            EsmFold2ExperimentalConfidenceHead(config) if config.confidence.enabled else None
        )
        self.post_init()

    def load_esmc(self, esmc: EsmCModel | EsmCForMaskedLM) -> EsmCModel:
        r"""
        Attach a frozen ESMC model.

        Args:
            esmc: ESMC trunk or masked-language model. For a masked-language model, the
                trunk supplies folding features and the language-model head remains
                available to binder-design objectives.

        Returns:
            The attached ESMC trunk in evaluation mode.
        """

        if isinstance(esmc, EsmCForMaskedLM):
            lm_head = esmc.lm_head
            lm_head.requires_grad_(False)
            self.esmc_lm_head = lm_head.eval()
            esmc = esmc.model
        esmc.requires_grad_(False)
        self.esmc = esmc.eval()
        return self.esmc

    def set_chunk_size(self, chunk_size: int | None) -> None:
        r"""
        Set the chunk size for memory-intensive folding operations.

        Args:
            chunk_size: Number of rows processed per chunk. ``None`` disables chunking.
        """

        self.folding_trunk.set_chunk_size(chunk_size)
        if self.confidence_head is not None:
            self.confidence_head.set_chunk_size(chunk_size)
        if self.msa_encoder is not None:
            self.msa_encoder.set_chunk_size(chunk_size)

    def configure_language_model_dropout(self, dropout: float, *, force_during_inference: bool = True) -> None:
        r"""
        Configure dropout on ESMC-derived pair features.

        Args:
            dropout: Dropout probability used by subsequent calls.
            force_during_inference: Whether to apply the dropout outside training mode.
        """

        self.language_model_dropout = dropout
        self.force_language_model_dropout = force_during_inference

    def _encode_and_trunk(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None,
        recycling_steps: int | None,
        language_model_mask_probability: float,
        generator: torch.Generator | None,
    ) -> _EsmFold2ExperimentalTrunkOutput:
        r"""
        Run the recurrent trunk and return single/pair/distogram plus sampler auxiliary.
        """

        token_mask = input.token_mask
        atom_mask = input.atom_mask
        residue_type_ids = input.residue_type_ids
        total_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if total_steps <= 0:
            raise ValueError("recycling_steps must be positive")

        # Soft one-hot design sequence overrides the discrete residue types on the
        # differentiable embedder path; the atom features stay tied to the argmax
        # residues supplied in the input.
        if residue_type_probabilities is not None:
            residue_type_features = residue_type_probabilities.float()
        else:
            residue_type_features = F.one_hot(residue_type_ids.long(), num_classes=_NUM_RESIDUE_TYPES).float()
            residue_type_features = residue_type_features * token_mask.unsqueeze(-1).float()

        deletion_mean = torch.zeros(
            residue_type_ids.shape[0], residue_type_ids.shape[1], device=residue_type_ids.device
        )

        msa_tokens = input.msa_tokens
        msa_features: Tensor | None = None
        msa_attention_mask = input.msa_attention_mask
        if self.use_msa_features and residue_type_probabilities is not None:
            # Fold the soft sequence into the single-row MSA profile and track.
            msa_profile = residue_type_features
            msa_features = residue_type_features.unsqueeze(1)
            msa_attention_mask = token_mask.unsqueeze(1)
        elif msa_tokens is not None:
            msa_residue_features = F.one_hot(msa_tokens.long(), num_classes=_NUM_RESIDUE_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_residue_features = msa_residue_features * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                msa_profile = msa_residue_features.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                msa_profile = msa_residue_features.mean(dim=1)
        else:
            msa_profile = residue_type_features

        if not self.use_msa_features:
            msa_profile = torch.zeros_like(msa_profile)
            deletion_mean = torch.zeros_like(deletion_mean)
            msa_tokens = None
            msa_features = None

        reference_element_features = F.one_hot(input.reference_elements.long(), num_classes=_MAX_ATOMIC_NUMBER).float()
        reference_atom_name_character_features = F.one_hot(
            input.reference_atom_name_characters.long(), num_classes=_CHAR_VOCAB_SIZE
        ).float()
        atom_mask_f = atom_mask.float()
        reference_element_features = reference_element_features * atom_mask_f.unsqueeze(-1)
        reference_atom_name_character_features = reference_atom_name_character_features * atom_mask_f.unsqueeze(
            -1
        ).unsqueeze(-1)
        atom_to_token = input.atom_to_token * atom_mask.long()

        single_inputs = self.input_embeddings(
            residue_type_features=residue_type_features,
            msa_profile=msa_profile.float(),
            deletion_mean=deletion_mean.float(),
            reference_positions=input.reference_positions,
            atom_mask=atom_mask,
            reference_space_id=input.reference_space_id,
            reference_charges=input.reference_charges,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
        )

        initial_pair = self.left_pair_projection(single_inputs).unsqueeze(2) + self.right_pair_projection(
            single_inputs
        ).unsqueeze(1)
        relative_position_encoding = self.relative_position_embedder(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            symmetry_index=input.symmetry_index,
            entity_index=input.entity_index,
            token_index=input.token_index,
        )
        token_bonds_encoding = self.bond_projection(input.token_bonds.to(dtype=self.bond_projection.weight.dtype))
        initial_pair = initial_pair + relative_position_encoding + token_bonds_encoding

        # ESMC pair projection is constant with respect to the design logits because
        # language_model_tokens come from the argmax sequence, so it is added to the initial pair once.
        with torch.no_grad():
            esmc_hidden_states = input.esmc_hidden_states
            if esmc_hidden_states is None:
                if self.esmc is None:
                    raise RuntimeError("ESMFold2 experimental requires an attached ESMC (call load_esmc)")
                if (
                    input.language_model_tokens is None
                    or input.language_model_mask is None
                    or input.token_to_esmc is None
                ):
                    raise RuntimeError("ESMFold2 experimental requires ESMC inputs or esmc_hidden_states")
                esmc_hidden_states = _compute_esmc_hidden_states(
                    self.esmc,
                    input.language_model_tokens,
                    input.language_model_mask,
                    input.token_to_esmc,
                    input.token_has_esmc,
                    language_model_mask_probability=language_model_mask_probability,
                    bos_token_id=self.esmc_bos_token_id,
                    eos_token_id=self.esmc_eos_token_id,
                    mask_token_id=self.esmc_mask_token_id,
                    generator=generator,
                )
            language_model_pair = self.language_model_pair_encoder(esmc_hidden_states)
            if self.language_model_dropout > 0 and (self.training or self.force_language_model_dropout):
                language_model_pair = _dropout(language_model_pair, self.language_model_dropout, generator)
        initial_pair = initial_pair + language_model_pair.to(initial_pair.dtype)
        del esmc_hidden_states, language_model_pair

        pair_mask = token_mask[:, :, None].float() * token_mask[:, None, :].float()

        pair = torch.zeros_like(initial_pair)
        for _ in range(total_steps):
            pair = initial_pair + self.recycling_projection(self.recycling_norm(pair))
            if self.msa_encoder is not None and (msa_tokens is not None or msa_features is not None):
                if msa_features is not None:
                    msa_depth = msa_features.shape[1]
                    msa_residue_features = msa_features.permute(0, 2, 1, 3).float()
                else:
                    msa_depth = cast(Tensor, msa_tokens).shape[1]
                    msa_residue_features = F.one_hot(
                        cast(Tensor, msa_tokens).permute(0, 2, 1).long(),
                        num_classes=_NUM_RESIDUE_TYPES,
                    ).float()
                msa_attention = (
                    msa_attention_mask.permute(0, 2, 1).float()
                    if msa_attention_mask is not None
                    else token_mask[:, :, None].expand(-1, -1, msa_depth).float()
                )
                msa_residue_features = msa_residue_features * msa_attention.unsqueeze(-1)
                zero_deletion_features = torch.zeros(
                    msa_residue_features.shape[0],
                    msa_residue_features.shape[1],
                    msa_depth,
                    device=msa_residue_features.device,
                )
                pair = pair + self.msa_encoder(
                    pair=pair,
                    single_inputs=single_inputs,
                    msa_residue_features=msa_residue_features,
                    has_deletion=zero_deletion_features,
                    deletion_value=zero_deletion_features,
                    msa_attention_mask=msa_attention,
                ).to(pair.dtype)
            pair = self.folding_trunk(pair, pair_attention_mask=pair_mask)

        distogram_logits = self.distogram_head(pair + pair.transpose(-2, -3))

        return _EsmFold2ExperimentalTrunkOutput(
            single_inputs=single_inputs,
            pair=pair,
            distogram_logits=distogram_logits,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
        )

    def _sample_from_trunk(
        self,
        input: EsmFold2Input,
        trunk: _EsmFold2ExperimentalTrunkOutput,
        *,
        num_diffusion_samples: int,
        num_sampling_steps: int | None,
        max_inference_sigma: float | None,
        noise_scale: float | None,
        step_scale: float | None,
        generator: torch.Generator | None,
    ) -> EsmFold2ExperimentalSample:
        with torch.no_grad():
            structure_output = self.diffusion_structure_head.sample(
                pair=trunk.pair.detach(),
                single_inputs=trunk.single_inputs.detach(),
                relative_position_encoding=trunk.relative_position_encoding.detach(),
                reference_positions=input.reference_positions,
                reference_charges=input.reference_charges,
                atom_mask=input.atom_mask,
                reference_element_features=trunk.reference_element_features,
                reference_atom_name_character_features=trunk.reference_atom_name_character_features,
                reference_space_id=input.reference_space_id,
                atom_to_token=trunk.atom_to_token,
                token_mask=input.token_mask,
                num_diffusion_samples=num_diffusion_samples,
                num_sampling_steps=num_sampling_steps,
                max_inference_sigma=max_inference_sigma,
                noise_scale=noise_scale,
                step_scale=step_scale,
                return_atom_representations=False,
                generator=generator,
            )
        return EsmFold2ExperimentalSample(
            distogram_logits=trunk.distogram_logits,
            coordinates=cast(Tensor, structure_output["coordinates"]),
        )

    def _confidence_from_trunk(
        self,
        input: EsmFold2Input,
        trunk: _EsmFold2ExperimentalTrunkOutput,
        sample: EsmFold2ExperimentalSample,
        *,
        num_diffusion_samples: int,
    ) -> EsmFold2ExperimentalConfidence:
        confidence_head = self.confidence_head
        if confidence_head is None:
            raise ValueError(
                "this experimental configuration has no confidence head "
                "(confidence.enabled=false); sample_with_confidence is unavailable"
            )
        with torch.no_grad():
            return confidence_head(
                single_inputs=trunk.single_inputs.detach(),
                pair=trunk.pair.detach(),
                coordinates=sample.coordinates,
                distogram_atom_index=input.distogram_atom_index,
                token_mask=input.token_mask,
                atom_to_token=trunk.atom_to_token,
                atom_mask=input.atom_mask,
                chain_index=input.chain_index,
                molecule_type=input.molecule_type,
                relative_position_encoding=trunk.relative_position_encoding.detach(),
                token_bonds_encoding=trunk.token_bonds_encoding.detach(),
                num_diffusion_samples=num_diffusion_samples,
            )

    def forward(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None = None,
        recycling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Predict differentiable distogram logits from discrete or soft residue types.

        ``B`` is batch size and ``N`` is token count.

        Args:
            input: Processed target-binder input.
            residue_type_probabilities: Optional soft residue-type probabilities with
                shape ``(B, N, 33)``. Gradients from ``distogram_logits`` propagate to
                this tensor.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            Distogram logits with shape ``(B, N, N, D)``.
        """

        grad_enabled = residue_type_probabilities is not None
        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if generator is None and seed is not None:
            generator = torch.Generator(device=input.reference_positions.device).manual_seed(seed)

        with torch.set_grad_enabled(grad_enabled):
            trunk = self._encode_and_trunk(
                input,
                residue_type_probabilities=residue_type_probabilities,
                recycling_steps=recycling_steps,
                language_model_mask_probability=language_model_mask_probability,
                generator=generator,
            )
        return trunk.distogram_logits

    def sample(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None = None,
        recycling_steps: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> EsmFold2ExperimentalSample:
        r"""
        Predict distogram logits and sample atom coordinates.

        Args:
            input: Processed target-binder input.
            residue_type_probabilities: Optional soft residue-type probabilities with
                shape ``(B, N, 33)``.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_diffusion_samples: Number of coordinate samples. ``None`` means one.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            max_inference_sigma: Largest diffusion sigma included in sampling. ``None``
                keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            Distogram logits and sampled atom coordinates.
        """

        num_samples = 1 if num_diffusion_samples is None else num_diffusion_samples
        grad_enabled = residue_type_probabilities is not None
        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if generator is None and seed is not None:
            generator = torch.Generator(device=input.reference_positions.device).manual_seed(seed)

        with torch.set_grad_enabled(grad_enabled):
            trunk = self._encode_and_trunk(
                input,
                residue_type_probabilities=residue_type_probabilities,
                recycling_steps=recycling_steps,
                language_model_mask_probability=language_model_mask_probability,
                generator=generator,
            )
        return self._sample_from_trunk(
            input,
            trunk,
            num_diffusion_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            generator=generator,
        )

    def sample_with_confidence(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None = None,
        recycling_steps: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> EsmFold2ExperimentalPrediction:
        r"""
        Predict distogram logits, sample atom coordinates, and compute confidence.

        Args:
            input: Processed target-binder input.
            residue_type_probabilities: Optional soft residue-type probabilities with
                shape ``(B, N, 33)``.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_diffusion_samples: Number of coordinate samples. ``None`` means one.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            max_inference_sigma: Largest diffusion sigma included in sampling. ``None``
                keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            One coordinate sample carrier and its fixed confidence values.
        """

        num_samples = 1 if num_diffusion_samples is None else num_diffusion_samples
        grad_enabled = residue_type_probabilities is not None
        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if generator is None and seed is not None:
            generator = torch.Generator(device=input.reference_positions.device).manual_seed(seed)

        with torch.set_grad_enabled(grad_enabled):
            trunk = self._encode_and_trunk(
                input,
                residue_type_probabilities=residue_type_probabilities,
                recycling_steps=recycling_steps,
                language_model_mask_probability=language_model_mask_probability,
                generator=generator,
            )
        sample = self._sample_from_trunk(
            input,
            trunk,
            num_diffusion_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            generator=generator,
        )
        confidence = self._confidence_from_trunk(
            input,
            trunk,
            sample,
            num_diffusion_samples=num_samples,
        )
        return EsmFold2ExperimentalPrediction(sample=sample, confidence=confidence)


class EsmFold2InputEmbeddings(nn.Module):
    r"""
    Embeds input features including atom-level encoding via SWA attention.
    """

    def __init__(
        self,
        config: EsmFold2Config | EsmFold2ExperimentalConfig,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        atom_encoder_config = config.input_embeddings.atom_encoder

        self.atom_encoder = EsmFold2AtomEncoder(
            backend=backend,
            atom_hidden_size=atom_encoder_config.atom_hidden_size,
            token_hidden_size=atom_encoder_config.token_hidden_size,
            num_blocks=atom_encoder_config.num_blocks,
            num_attention_heads=atom_encoder_config.num_attention_heads,
            sliding_window_size=atom_encoder_config.sliding_window_size,
            expansion=atom_encoder_config.expansion,
            spatial_rope_base_frequency=atom_encoder_config.spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=atom_encoder_config.num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=atom_encoder_config.num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=atom_encoder_config.reference_space_rope_base_frequency,
        )

    def forward(
        self,
        residue_type_features: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
    ) -> Tensor:
        token_features, _atom_query, _atom_conditioning, _intermediates = self.atom_encoder(
            reference_positions=reference_positions,
            atom_mask=atom_mask,
            reference_space_id=reference_space_id,
            reference_charges=reference_charges,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
            num_tokens=residue_type_features.shape[1],
        )
        return torch.cat(
            [
                token_features,
                residue_type_features.to(dtype=token_features.dtype),
                msa_profile.to(dtype=token_features.dtype),
                deletion_mean.unsqueeze(-1).to(dtype=token_features.dtype),
            ],
            dim=-1,
        )


class EsmFold2LanguageModelPairEncoder(nn.Module):
    r"""
    Project and combine ESMC layer states into the pair representation.
    """

    def __init__(self, pair_size: int = 256, hidden_size: int = 2560, num_hidden_layers: int = 80) -> None:
        super().__init__()

        self.hidden_norm = LayerNorm(hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, pair_size, bias=False)
        self.layer_weights = nn.Parameter(torch.zeros(num_hidden_layers + 1))
        self.single_to_pair = EsmFold2SingleToPair(pair_size, pair_size, pair_size)
        self.output_norm = LayerNorm(pair_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if not torch.is_autocast_enabled(hidden_states.device.type):
            hidden_states = hidden_states.to(dtype=self.layer_weights.dtype)
        language_model_pair = self.hidden_projection(self.hidden_norm(hidden_states))  # [B, L, 81, pair_size]
        weights = self.layer_weights.softmax(0)  # [81]
        # Keep the batched-matmul reduction order. Do not squeeze:
        # ``weights @ language_model_pair`` already has shape [B, L, pair_size], including when L=1.
        language_model_pair = weights @ language_model_pair  # [B, L, pair_size]
        return self.output_norm(self.single_to_pair(language_model_pair))  # [B, L, L, pair_size]


class EsmFold2RelativePositionEmbedder(nn.Module):
    r"""
    Embed relative residue, token, chain, and entity relationships.
    """

    def __init__(
        self,
        num_relative_residue_bins: int = 32,
        num_relative_chain_bins: int = 2,
        pair_size: int = 256,
    ) -> None:
        super().__init__()
        self.num_relative_residue_bins = num_relative_residue_bins
        self.num_relative_chain_bins = num_relative_chain_bins

        num_residue_features = 2 * num_relative_residue_bins + 2
        num_token_features = 2 * num_relative_residue_bins + 2
        num_chain_features = 2 * num_relative_chain_bins + 2
        num_entity_features = 1
        num_features = num_residue_features + num_token_features + num_chain_features + num_entity_features
        self.projection = nn.Linear(num_features, pair_size, bias=False)

    def forward(
        self,
        residue_index: Tensor,
        chain_index: Tensor,
        symmetry_index: Tensor,
        entity_index: Tensor,
        token_index: Tensor,
    ) -> Tensor:
        same_chain = chain_index.unsqueeze(2) == chain_index.unsqueeze(1)
        same_residue = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
        same_entity = entity_index.unsqueeze(2) == entity_index.unsqueeze(1)

        relative_residue = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        relative_residue = torch.clip(
            relative_residue + self.num_relative_residue_bins,
            0,
            2 * self.num_relative_residue_bins,
        )
        relative_residue = torch.where(same_chain, relative_residue, 2 * self.num_relative_residue_bins + 1)
        relative_residue_features = F.one_hot(relative_residue, 2 * self.num_relative_residue_bins + 2)

        relative_token = torch.clip(
            token_index.unsqueeze(2) - token_index.unsqueeze(1) + self.num_relative_residue_bins,
            0,
            2 * self.num_relative_residue_bins,
        )
        relative_token = torch.where(
            same_chain & same_residue,
            relative_token,
            2 * self.num_relative_residue_bins + 1,
        )
        relative_token_features = F.one_hot(relative_token, 2 * self.num_relative_residue_bins + 2)

        relative_chain = torch.clip(
            symmetry_index.unsqueeze(2) - symmetry_index.unsqueeze(1) + self.num_relative_chain_bins,
            0,
            2 * self.num_relative_chain_bins,
        )
        relative_chain = torch.where(same_chain, 2 * self.num_relative_chain_bins + 1, relative_chain)
        relative_chain_features = F.one_hot(relative_chain, 2 * self.num_relative_chain_bins + 2)

        features = torch.cat(
            [
                relative_residue_features.float(),
                relative_token_features.float(),
                same_entity.float().unsqueeze(-1),
                relative_chain_features.float(),
            ],
            dim=-1,
        )

        return self.projection(features.to(dtype=self.projection.weight.dtype))


class EsmFold2SingleToPair(nn.Module):
    r"""
    Build pair features from projected single features.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.single_projection = nn.Linear(input_size, hidden_size)
        self.pair_input_projection = nn.Linear(2 * hidden_size, output_size)
        self.activation = nn.GELU()
        self.pair_output_projection = nn.Linear(output_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.single_projection(x)
        x = torch.cat(
            [(x.unsqueeze(2) * x.unsqueeze(1)), (x.unsqueeze(2) - x.unsqueeze(1))],
            dim=3,
        )
        return self.pair_output_projection(self.activation(self.pair_input_projection(x)))


class EsmFold2FoldingTrunk(nn.Module):
    r"""
    Pair-only folding trunk composed of repeated folding blocks.
    """

    def __init__(
        self,
        num_blocks: int = 24,
        pair_size: int = 256,
        expansion: int = 4,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            [EsmFold2FoldingBlock(backend, pair_size=pair_size, expansion=expansion) for _ in range(num_blocks)]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            cast(EsmFold2FoldingBlock, block).set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            fn = partial(block, pair_attention_mask=pair_attention_mask)
            if torch.is_grad_enabled():
                pair = checkpoint(fn, pair, use_reentrant=False)  # pyright: ignore
            else:
                pair = fn(pair)
        return pair


class EsmFold2MsaEncoder(nn.Module):
    r"""
    Homogeneous MSA blocks followed by one final OPM and pair update.
    """

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        input_size: int,
        hidden_size: int = 32,
        num_blocks: int = 4,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 16,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.msa_projection = nn.Linear(35, msa_size, bias=False)
        self.input_projection = nn.Linear(input_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                EsmFold2MsaEncoderBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    hidden_size=hidden_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_head_dim=msa_head_dim,
                    backend=backend,
                )
                for _ in range(num_blocks - 1)
            ]
        )
        self.final_outer_product_mean = EsmFold2OuterProductMean(
            msa_size,
            hidden_size,
            pair_size,
            backend,
        )
        self.final_pair_update = EsmFold2FoldingBlock(
            backend,
            pair_size=pair_size,
            expansion=4,
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            cast(EsmFold2MsaEncoderBlock, block).set_chunk_size(chunk_size)
        self.final_outer_product_mean.set_chunk_size(chunk_size)
        self.final_pair_update.set_chunk_size(chunk_size)

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        msa_residue_features: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        msa_features = torch.cat(
            [msa_residue_features, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)],
            dim=-1,
        )
        m = self.msa_projection(msa_features.to(dtype=self.msa_projection.weight.dtype))
        m = m + self.input_projection(single_inputs.to(dtype=self.input_projection.weight.dtype)).unsqueeze(2)
        token_mask = msa_attention_mask[:, :, 0].bool()
        pair_attention_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        for block in self.blocks:
            m, pair = block(m, pair, msa_attention_mask, pair_attention_mask)
        pair = pair + self.final_outer_product_mean(m, msa_attention_mask)
        pair = self.final_pair_update(pair, pair_attention_mask)
        return pair


# Experimental recurrent ESMFold2 architecture. It reinjects the previous pair
# representation through the recycling projection, adds the ESMC projection once,
# and predicts pLDDT and PAE from a 128-bin distogram. residue_type_probabilities keeps the
# distogram differentiable for binder design.


class EsmFold2ExperimentalMsaEncoder(nn.Module):
    r"""
    Experimental MSA encoder: embeds MSA features and updates the pair representation.
    """

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        input_size: int,
        hidden_size: int = 32,
        num_blocks: int = 4,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.msa_projection = nn.Linear(35, msa_size, bias=False)
        self.input_projection = nn.Linear(input_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                EsmFold2ExperimentalMsaEncoderBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    hidden_size=hidden_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_head_dim=msa_head_dim,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            cast(EsmFold2ExperimentalMsaEncoderBlock, block).set_chunk_size(chunk_size)

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        msa_residue_features: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        B, L, M = msa_attention_mask.shape
        msa_features = torch.cat(
            [msa_residue_features, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)],
            dim=-1,
        )
        m = self.msa_projection(msa_features.to(dtype=self.msa_projection.weight.dtype))
        m = m + self.input_projection(single_inputs.to(dtype=self.input_projection.weight.dtype)).unsqueeze(2)

        if M > 1:
            msa_track_mask = msa_attention_mask[:, :, 1:].any(dim=(1, 2))
        else:
            msa_track_mask = torch.zeros(B, dtype=torch.bool, device=pair.device)

        token_mask = msa_attention_mask[:, :, 0]
        pair_attention_mask = token_mask.unsqueeze(2) * token_mask.unsqueeze(1)
        for block in self.blocks:
            m, pair = block(m, pair, msa_attention_mask, pair_attention_mask, msa_track_mask)
        return pair * msa_track_mask[:, None, None, None].to(pair.dtype)


class EsmFold2AtomEncoder(nn.Module):
    r"""
    Encode atom features with a sliding-window atom transformer.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.backend = backend

        self.atom_projection = nn.Linear(_ATOM_FEATURE_DIM, atom_hidden_size, bias=False)
        self.input_norm = LayerNorm(atom_hidden_size)

        self.transformer = EsmFold2AtomTransformer(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=expansion,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size // 2, bias=False)

    def encode_reference(
        self,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
        num_diffusion_samples: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_atoms = reference_positions.shape[:2]
        atom_features = torch.cat(
            [
                reference_positions,
                reference_charges.unsqueeze(-1),
                atom_mask.unsqueeze(-1),
                reference_element_features,
                reference_atom_name_character_features.reshape(batch_size, num_atoms, _MAX_CHARS * _CHAR_VOCAB_SIZE),
            ],
            dim=-1,
        )
        atom_features = atom_features.to(dtype=self.atom_projection.weight.dtype)
        reference_atom_features = self.input_norm(self.atom_projection(atom_features))
        attention_cos, attention_sin = self.transformer.build_3d_rope(reference_positions, reference_space_id)
        attention_cos = attention_cos.to(dtype=reference_atom_features.dtype)
        attention_sin = attention_sin.to(dtype=reference_atom_features.dtype)
        reference_atom_features = reference_atom_features.repeat_interleave(num_diffusion_samples, 0)
        attention_cos = attention_cos.repeat_interleave(num_diffusion_samples, 0)
        attention_sin = attention_sin.repeat_interleave(num_diffusion_samples, 0)
        expanded_atom_mask = atom_mask.repeat_interleave(num_diffusion_samples, 0)
        expanded_atom_to_token = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        return (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        )

    def encode(
        self,
        reference_atom_features: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        expanded_atom_mask: Tensor,
        expanded_atom_to_token: Tensor,
        num_tokens: int,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        result = self.transformer(
            atom_query=reference_atom_features,
            atom_conditioning=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            atom_mask=expanded_atom_mask,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            atom_query, intermediates = result
        else:
            atom_query = result
            intermediates = []

        atom_to_token_update = F.relu(self.token_projection(atom_query))
        token_features = _scatter_atom_to_token(
            atom_to_token_update,
            expanded_atom_to_token,
            num_tokens,
            atom_mask=expanded_atom_mask.bool(),
            backend=self.backend,
        )

        return token_features, atom_query, reference_atom_features, intermediates

    def forward(
        self,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        ) = self.encode_reference(
            reference_positions,
            atom_mask,
            reference_space_id,
            reference_charges,
            reference_element_features,
            reference_atom_name_character_features,
            atom_to_token,
            num_diffusion_samples,
        )
        return self.encode(
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
            num_tokens,
            return_intermediates,
        )


class EsmFold2DenoisingAtomEncoder(nn.Module):
    r"""
    Encode noisy atom coordinates into full-width diffusion token states.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        nn.Module.__init__(self)
        self.backend = backend
        self.atom_projection = nn.Linear(_ATOM_FEATURE_DIM, atom_hidden_size, bias=False)
        self.input_norm = LayerNorm(atom_hidden_size)
        self.coordinate_projection = nn.Linear(6, atom_hidden_size, bias=False)
        self.transformer = EsmFold2AtomTransformer(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=expansion,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)

    def encode_reference(
        self,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
        num_diffusion_samples: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_atoms = reference_positions.shape[:2]
        atom_features = torch.cat(
            [
                reference_positions,
                reference_charges.unsqueeze(-1),
                atom_mask.unsqueeze(-1),
                reference_element_features,
                reference_atom_name_character_features.reshape(batch_size, num_atoms, _MAX_CHARS * _CHAR_VOCAB_SIZE),
            ],
            dim=-1,
        )
        atom_features = atom_features.to(dtype=self.atom_projection.weight.dtype)
        reference_atom_features = self.input_norm(self.atom_projection(atom_features))
        attention_cos, attention_sin = self.transformer.build_3d_rope(reference_positions, reference_space_id)
        attention_cos = attention_cos.to(dtype=reference_atom_features.dtype)
        attention_sin = attention_sin.to(dtype=reference_atom_features.dtype)
        reference_atom_features = reference_atom_features.repeat_interleave(num_diffusion_samples, 0)
        attention_cos = attention_cos.repeat_interleave(num_diffusion_samples, 0)
        attention_sin = attention_sin.repeat_interleave(num_diffusion_samples, 0)
        expanded_atom_mask = atom_mask.repeat_interleave(num_diffusion_samples, 0)
        expanded_atom_to_token = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        return (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        )

    def encode(
        self,
        reference_atom_features: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        expanded_atom_mask: Tensor,
        expanded_atom_to_token: Tensor,
        num_tokens: int,
        noisy_coordinates: Tensor,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        coordinate_features = torch.cat([noisy_coordinates, torch.zeros_like(noisy_coordinates)], dim=-1)
        coordinate_embedding = self.coordinate_projection(
            coordinate_features.to(dtype=self.coordinate_projection.weight.dtype)
        )
        atom_query = reference_atom_features + coordinate_embedding
        result = self.transformer(
            atom_query=atom_query,
            atom_conditioning=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            atom_mask=expanded_atom_mask,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            atom_query, intermediates = result
        else:
            atom_query = result
            intermediates = []
        atom_to_token_update = F.relu(self.token_projection(atom_query))
        token_features = _scatter_atom_to_token(
            atom_to_token_update,
            expanded_atom_to_token,
            num_tokens,
            atom_mask=expanded_atom_mask.bool(),
            backend=self.backend,
        )
        return token_features, atom_query, reference_atom_features, intermediates

    def forward(
        self,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        noisy_coordinates: Tensor,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        encoded = self.encode_reference(
            reference_positions,
            atom_mask,
            reference_space_id,
            reference_charges,
            reference_element_features,
            reference_atom_name_character_features,
            atom_to_token,
            num_diffusion_samples,
        )
        return self.encode(*encoded, num_tokens, noisy_coordinates, return_intermediates)


class EsmFold2AtomDecoder(nn.Module):
    r"""
    Decode token features into atom-coordinate updates.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.token_projection = nn.Linear(token_hidden_size, atom_hidden_size, bias=False)

        self.transformer = EsmFold2AtomTransformer(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=expansion,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

        self.output_norm = LayerNorm(atom_hidden_size)
        self.coordinate_projection = nn.Linear(atom_hidden_size, _XYZ_DIMS, bias=False)

    def forward(
        self,
        token: Tensor,
        atom_query: Tensor,
        atom_conditioning: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
        expanded_atom_to_token: Tensor,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        token_to_atom_update = self.token_projection(token)
        token_to_atom_update = _gather_token_to_atom(
            token_to_atom_update,
            expanded_atom_to_token,
            backend=self.backend,
        )
        atom_query = atom_query + token_to_atom_update

        result = self.transformer(
            atom_query=atom_query,
            atom_conditioning=atom_conditioning,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            atom_mask=atom_mask,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            atom_query, intermediates = result
        else:
            atom_query = result
            intermediates = []

        coordinate_update = self.coordinate_projection(self.output_norm(atom_query))
        return coordinate_update, intermediates


class EsmFold2AtomTransformer(nn.Module):
    r"""
    Stack of [`EsmFold2AtomBlock`] modules.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.head_dim = atom_hidden_size // num_attention_heads
        self.spatial_rope_base_frequency = spatial_rope_base_frequency
        self.num_spatial_rope_pairs_per_axis = num_spatial_rope_pairs_per_axis
        self.num_reference_space_rope_pairs = num_reference_space_rope_pairs
        self.reference_space_rope_base_frequency = reference_space_rope_base_frequency

        self.blocks = nn.ModuleList(
            [
                EsmFold2AtomBlock(
                    atom_hidden_size=atom_hidden_size,
                    num_attention_heads=num_attention_heads,
                    backend=backend,
                    half_window=sliding_window_size // 2,
                    expansion=expansion,
                )
                for _ in range(num_blocks)
            ]
        )

    def build_3d_rope(self, reference_positions: Tensor, reference_space_id: Tensor) -> tuple[Tensor, Tensor]:
        return _build_3d_rope(
            reference_positions=reference_positions,
            reference_space_id=reference_space_id,
            head_dim=self.head_dim,
            num_spatial_pairs_per_axis=self.num_spatial_rope_pairs_per_axis,
            num_reference_space_pairs=self.num_reference_space_rope_pairs,
            spatial_base_frequency=self.spatial_rope_base_frequency,
            reference_space_base_frequency=self.reference_space_rope_base_frequency,
        )

    def forward(
        self,
        atom_query: Tensor,
        atom_conditioning: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        for block in self.blocks:
            atom_query = block(atom_query, atom_conditioning, attention_cos, attention_sin, atom_mask)
            if return_intermediates:
                intermediates.append(atom_query)
        if return_intermediates:
            return atom_query, intermediates
        return atom_query


class EsmFold2Denoiser(nn.Module):
    r"""
    Denoise atom coordinates from conditioned token and pair features.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        pair_size: int = 256,
        input_size: int = 451,
        sigma_data: float = 16.0,
        fourier_embedding_size: int = 256,
        num_atom_blocks: int = 3,
        num_atom_attention_heads: int = 4,
        num_token_encoder_blocks: int = 12,
        num_token_encoder_attention_heads: int = 16,
        expansion: int = 2,
        sliding_window_size: int = 128,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)

        self.conditioning = EsmFold2DiffusionConditioning(
            backend=backend,
            pair_size=pair_size,
            single_size=token_hidden_size,  # conditioning s output is token_hidden_size
            input_size=input_size,
            sigma_data=sigma_data,
            fourier_embedding_size=fourier_embedding_size,
            expansion=expansion,
        )

        self.atom_encoder = EsmFold2DenoisingAtomEncoder(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=token_hidden_size,
            num_blocks=num_atom_blocks,
            num_attention_heads=num_atom_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=2,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

        atom_decoder = EsmFold2AtomDecoder(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=token_hidden_size,
            num_blocks=num_atom_blocks,
            num_attention_heads=num_atom_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=2,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

        single_projection = nn.Linear(token_hidden_size, token_hidden_size, bias=False)
        nn.init.zeros_(single_projection.weight)

        token_encoder = EsmFold2DiffusionTransformerEncoder(
            hidden_size=token_hidden_size,
            pair_size=pair_size,
            num_attention_heads=num_token_encoder_attention_heads,
            num_blocks=num_token_encoder_blocks,
            conditioning_size=token_hidden_size,
            expansion=expansion,
            backend=backend,
        )

        single_norm = LayerNorm(token_hidden_size)
        token_norm = LayerNorm(token_hidden_size)

        self.single_norm = single_norm
        self.single_projection = single_projection
        self.token_encoder = token_encoder
        self.token_norm = token_norm
        self.atom_decoder = atom_decoder

    def denoise(
        self,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        single_inputs: Tensor,
        conditioned_pair: Tensor,
        token_mask: Tensor,
        reference_atom_features: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        expanded_atom_mask: Tensor,
        expanded_atom_to_token: Tensor,
        num_diffusion_samples: int = 1,
        return_token_representations: bool = False,
        return_atom_representations: bool = False,
    ) -> dict[str, Tensor | None]:
        noise_level = noise_level.to(dtype=torch.float32, device=noisy_coordinates.device).reshape(-1)
        conditioning = self.conditioning.condition_single(
            noise_level,
            single_inputs,
            num_diffusion_samples,
        )

        coordinate_scale = torch.sqrt(noise_level * noise_level + self.sigma_data * self.sigma_data)
        scaled_noisy_coordinates = noisy_coordinates / coordinate_scale[:, None, None]

        token, atom_query_skip, atom_conditioning_skip, encoder_intermediates = self.atom_encoder.encode(
            reference_atom_features=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            expanded_atom_mask=expanded_atom_mask,
            expanded_atom_to_token=expanded_atom_to_token,
            num_tokens=single_inputs.shape[1],
            noisy_coordinates=scaled_noisy_coordinates,
            return_intermediates=return_atom_representations,
        )

        token = token + self.single_projection(self.single_norm(conditioning))
        token, _ = self.token_encoder(
            token,
            conditioning,
            conditioned_pair,
            attention_mask=token_mask,
            num_diffusion_samples=num_diffusion_samples,
        )
        token = self.token_norm(token)

        coordinate_update, decoder_intermediates = self.atom_decoder(
            token=token,
            atom_query=atom_query_skip,
            atom_conditioning=atom_conditioning_skip,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            atom_mask=expanded_atom_mask,
            expanded_atom_to_token=expanded_atom_to_token,
            return_intermediates=return_atom_representations,
        )

        sigma2 = self.sigma_data * self.sigma_data
        noise_level_squared = noise_level * noise_level
        denoised_coordinates = (sigma2 / (sigma2 + noise_level_squared))[:, None, None] * noisy_coordinates
        denoised_coordinates = (
            denoised_coordinates
            + ((self.sigma_data * noise_level) / torch.sqrt(sigma2 + noise_level_squared))[:, None, None]
            * coordinate_update
        )

        atom_intermediates = (
            torch.stack(encoder_intermediates + decoder_intermediates, dim=2) if return_atom_representations else None
        )

        return {
            "denoised_coordinates": denoised_coordinates,
            "token_representations": token if return_token_representations else None,
            "atom_intermediates": atom_intermediates,
        }

    def forward(
        self,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        reference_positions: Tensor,
        reference_charges: Tensor,
        atom_mask: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        single_inputs: Tensor,
        pair: Tensor,
        relative_position_encoding: Tensor,
        token_mask: Tensor,
        num_diffusion_samples: int = 1,
        return_token_representations: bool = False,
        return_atom_representations: bool = False,
    ) -> dict[str, Tensor | None]:
        conditioned_pair = self.conditioning.condition_pair(
            pair,
            relative_position_encoding,
        )
        (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        ) = self.atom_encoder.encode_reference(
            reference_positions,
            atom_mask,
            reference_space_id,
            reference_charges,
            reference_element_features,
            reference_atom_name_character_features,
            atom_to_token,
            num_diffusion_samples,
        )
        return self.denoise(
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            single_inputs=single_inputs,
            conditioned_pair=conditioned_pair,
            token_mask=token_mask,
            reference_atom_features=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            expanded_atom_mask=expanded_atom_mask,
            expanded_atom_to_token=expanded_atom_to_token,
            num_diffusion_samples=num_diffusion_samples,
            return_token_representations=return_token_representations,
            return_atom_representations=return_atom_representations,
        )


class EsmFold2FoldingBlock(nn.Module):
    r"""
    Outgoing and incoming triangle multiplication followed by a transition.
    """

    def __init__(self, backend: str = "auto", pair_size: int = 256, expansion: int = 4) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = EsmFold2TriangleMultiplication(
            pair_size,
            pair_size,
            "outgoing",
            backend,
        )
        self.triangle_multiplication_incoming = EsmFold2TriangleMultiplication(
            pair_size,
            pair_size,
            "incoming",
            backend,
        )
        self.transition = EsmFold2Transition(pair_size, backend, expansion=expansion)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.triangle_multiplication_outgoing.set_chunk_size(chunk_size)
        self.triangle_multiplication_incoming.set_chunk_size(chunk_size)
        self.transition.set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        pair = pair + self.triangle_multiplication_outgoing(pair, visibility=pair_attention_mask)
        pair = pair + self.triangle_multiplication_incoming(pair, visibility=pair_attention_mask)
        return pair + self.transition(pair)


class EsmFold2MsaEncoderBlock(nn.Module):
    r"""
    One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update.
    """

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        hidden_size: int,
        num_msa_attention_heads: int,
        msa_head_dim: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = EsmFold2OuterProductMean(msa_size, hidden_size, pair_size, backend)
        self.msa_pair_weighted_averaging = EsmFold2MsaPairWeightedAveraging(
            msa_size,
            pair_size,
            backend,
            num_msa_attention_heads,
            msa_head_dim,
        )
        self.msa_transition = EsmFold2Transition(msa_size, backend, expansion=4)
        self.triangle_multiplication_outgoing = EsmFold2TriangleMultiplication(
            pair_size,
            pair_size,
            "outgoing",
            backend,
        )
        self.triangle_multiplication_incoming = EsmFold2TriangleMultiplication(
            pair_size,
            pair_size,
            "incoming",
            backend,
        )
        self.pair_transition = EsmFold2Transition(pair_size, backend, expansion=4)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.triangle_multiplication_outgoing.set_chunk_size(chunk_size)
        self.triangle_multiplication_incoming.set_chunk_size(chunk_size)
        self.msa_transition.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
        m = m + self.msa_transition(m)
        pair = pair + self.triangle_multiplication_outgoing(pair, visibility=pair_attention_mask)
        pair = pair + self.triangle_multiplication_incoming(pair, visibility=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair


class EsmFold2ExperimentalMsaEncoderBlock(nn.Module):
    r"""
    Run an MSA update followed by outer-product-mean and triangle updates.
    """

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        hidden_size: int,
        num_msa_attention_heads: int,
        msa_head_dim: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        outer_product_mean = EsmFold2OuterProductMean(
            msa_size,
            hidden_size,
            pair_size,
            backend,
            divide_before_projection=True,
        )
        msa_pair_weighted_averaging = EsmFold2MsaPairWeightedAveraging(
            msa_size,
            pair_size,
            backend,
            num_msa_attention_heads,
            msa_head_dim,
        )
        msa_transition = EsmFold2Transition(msa_size, backend, expansion=4)
        self.msa_pair_weighted_averaging = msa_pair_weighted_averaging
        self.msa_transition = msa_transition
        self.outer_product_mean = outer_product_mean
        self.triangle_multiplication_outgoing = EsmFold2TriangleMultiplication(
            pair_size,
            pair_size,
            "outgoing",
            backend,
        )
        self.triangle_multiplication_incoming = EsmFold2TriangleMultiplication(
            pair_size,
            pair_size,
            "incoming",
            backend,
        )
        self.pair_transition = EsmFold2Transition(pair_size, backend, expansion=4)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.msa_transition.set_chunk_size(chunk_size)
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.triangle_multiplication_outgoing.set_chunk_size(chunk_size)
        self.triangle_multiplication_incoming.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
        msa_track_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        mask4d = msa_track_mask[:, None, None, None].to(m.dtype) if msa_track_mask is not None else None

        def masked(x: Tensor) -> Tensor:
            return x * mask4d if mask4d is not None else x

        m = m + masked(self.msa_pair_weighted_averaging(m, pair, pair_attention_mask))
        m = m + masked(self.msa_transition(m))
        pair = pair + masked(self.outer_product_mean(m, msa_attention_mask))
        pair = pair + masked(self.triangle_multiplication_outgoing(pair, visibility=pair_attention_mask))
        pair = pair + masked(self.triangle_multiplication_incoming(pair, visibility=pair_attention_mask))
        pair = pair + masked(self.pair_transition(pair))
        return m, pair


class EsmFold2AtomBlock(nn.Module):
    r"""
    Adaptive RMS normalization, sliding-window attention, and transition updates.
    """

    def __init__(
        self,
        atom_hidden_size: int,
        num_attention_heads: int,
        backend: str = "auto",
        half_window: int = 64,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.adaptive_norm_activation = nn.SiLU()
        self.adaptive_norm_projection = nn.Linear(atom_hidden_size, 6 * atom_hidden_size, bias=False)
        nn.init.zeros_(self.adaptive_norm_projection.weight)
        self.attention = EsmFold2SpatialAtomAttention(
            atom_hidden_size, num_attention_heads, backend, half_window=half_window
        )
        hidden_size = ((expansion * (atom_hidden_size // 3) * 2) + 255) // 256 * 256
        self.transition = SwiGluFeedForward(atom_hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        atom_conditioning: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        mod = self.adaptive_norm_projection(self.adaptive_norm_activation(atom_conditioning))
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        attn_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_a) + shift_a
        attention_update = self.attention(attn_input, attention_cos, attention_sin, atom_mask)
        x = x + gate_a * attention_update

        transition_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_f) + shift_f
        transition_update = self.transition(transition_input)
        return x + gate_f * transition_update


class EsmFold2DiffusionTransformerEncoder(nn.Module):
    r"""
    Diffusion denoising Transformer encoder with attention pair bias.
    """

    def __init__(
        self,
        hidden_size: int,
        pair_size: int,
        num_attention_heads: int,
        num_blocks: int,
        conditioning_size: int,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            EsmFold2DiffusionTransformerEncoderBlock(
                hidden_size=hidden_size,
                pair_size=pair_size,
                num_attention_heads=num_attention_heads,
                conditioning_size=conditioning_size,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        attention_mask: Tensor,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        for block in self.blocks:
            hidden = block(
                hidden,
                conditioning,
                pair,
                attention_mask=attention_mask,
                num_diffusion_samples=num_diffusion_samples,
            )
            if return_intermediates:
                intermediates.append(hidden)
        return hidden, intermediates


class EsmFold2DiffusionTransformerEncoderBlock(nn.Module):
    r"""
    One attention and transition residual unit in the diffusion Transformer encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        pair_size: int,
        num_attention_heads: int,
        conditioning_size: int,
        expansion: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.attention = EsmFold2PairBiasAttention(
            hidden_size,
            pair_size,
            num_attention_heads,
            conditioning_size,
            backend=backend,
        )
        self.transition = ConditionedTransition(
            hidden_size,
            conditioning_size,
            expansion,
            backend=backend,
        )
        nn.init.zeros_(self.transition.conditioning_gate_projection.weight)
        nn.init.constant_(self.transition.conditioning_gate_projection.bias, -2.0)

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        attention_mask: Tensor,
        num_diffusion_samples: int = 1,
    ) -> Tensor:
        hidden = hidden + self.attention(
            hidden,
            conditioning,
            pair,
            attention_mask=attention_mask,
            num_diffusion_samples=num_diffusion_samples,
        )
        return hidden + self.transition(hidden, conditioning)


_DEFAULT_CHUNK_SIZE = 64


class EsmFold2TriangleMultiplication(nn.Module):
    r"""
    Atomic triangle multiplicative transform with gated signal routing.
    """

    _FLOW_TO_EINSUM = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}
    _VALID_FLOWS = ("outgoing", "incoming")

    def __init__(
        self,
        input_channels: int,
        latent_channels: int,
        flow: str,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if flow not in self._FLOW_TO_EINSUM:
            raise ValueError(f"Invalid flow={flow!r}. Expected one of {self._VALID_FLOWS}.")

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.flow = flow
        self.backend = backend
        self._einsum_equation = self._FLOW_TO_EINSUM[flow]
        self.input_norm = LayerNorm(self.input_channels, eps=_EPS)
        self.left_value_projection = nn.Linear(self.input_channels, self.latent_channels, bias=False)
        self.right_value_projection = nn.Linear(self.input_channels, self.latent_channels, bias=False)
        self.left_gate_projection = nn.Linear(self.input_channels, self.latent_channels, bias=False)
        self.right_gate_projection = nn.Linear(self.input_channels, self.latent_channels, bias=False)
        self.output_norm = LayerNorm(self.latent_channels, eps=_EPS)
        self.output_projection = nn.Linear(self.latent_channels, self.input_channels, bias=False)
        self.output_gate_projection = nn.Linear(self.input_channels, self.input_channels, bias=False)

        # Chunked by default to bound memory on long sequences.
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def _triangular_contract(self, left_stream: Tensor, right_stream: Tensor) -> Tensor:
        return torch.einsum(self._einsum_equation, left_stream, right_stream)

    def _triangular_contract_chunked(self, left_stream: Tensor, right_stream: Tensor, chunk_size: int) -> Tensor:
        L = left_stream.shape[1] if self.flow == "outgoing" else left_stream.shape[2]
        chunks = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            if self.flow == "outgoing":
                chunk = torch.einsum(self._einsum_equation, left_stream[:, start:end], right_stream)
            else:
                chunk = torch.einsum(self._einsum_equation, left_stream[:, :, start:end], right_stream)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)

    def forward(self, pair_grid: Tensor, visibility: Tensor | None = None) -> Tensor:
        if visibility is None:
            visibility = pair_grid.new_ones(pair_grid.shape[:-1])

        resolved = resolve_tensor_backend(
            self.backend,
            tensors=(pair_grid, visibility),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            return dispatch_triangle_multiplication(
                pair_grid,
                visibility,
                outgoing=self.flow == "outgoing",
                input_norm_weight=self.input_norm.weight,
                input_norm_bias=self.input_norm.bias,
                left_value_weight=self.left_value_projection.weight,
                left_gate_weight=self.left_gate_projection.weight,
                right_value_weight=self.right_value_projection.weight,
                right_gate_weight=self.right_gate_projection.weight,
                output_norm_weight=self.output_norm.weight,
                output_norm_bias=self.output_norm.bias,
                output_weight=self.output_projection.weight,
                output_gate_weight=self.output_gate_projection.weight,
                input_norm_eps=self.input_norm.eps,
                output_norm_eps=self.output_norm.eps,
                backend="oops",
            )

        normalized_grid = self.input_norm(pair_grid)
        left_stream = self.left_value_projection(normalized_grid)
        left_stream = left_stream * torch.sigmoid(self.left_gate_projection(normalized_grid))
        right_stream = self.right_value_projection(normalized_grid)
        right_stream = right_stream * torch.sigmoid(self.right_gate_projection(normalized_grid))
        visibility = visibility.unsqueeze(-1)
        left_stream = (left_stream * visibility).float()
        right_stream = (right_stream * visibility).float()
        if self._chunk_size is not None:
            contracted = self._triangular_contract_chunked(left_stream, right_stream, self._chunk_size)
        else:
            contracted = self._triangular_contract(left_stream, right_stream)
        mixed = self.output_projection(self.output_norm(contracted.to(dtype=self.output_norm.weight.dtype)))
        output_gate = torch.sigmoid(self.output_gate_projection(normalized_grid))
        return mixed * output_gate


class EsmFold2OuterProductMean(nn.Module):
    r"""
    Outer-product mean: maps an MSA representation into a pair update.
    """

    def __init__(
        self,
        msa_size: int,
        hidden_size: int,
        pair_size: int,
        backend: str = "auto",
        divide_before_projection: bool = False,
    ) -> None:
        super().__init__()
        self.divide_before_projection = divide_before_projection
        self.backend = backend
        self.msa_norm = LayerNorm(msa_size)
        self.left_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.right_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size * hidden_size, pair_size, bias=True)
        # Unchunked by default; set a chunk size for long sequences.
        self._chunk_size: int | None = None

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, m: Tensor, msa_attention_mask: Tensor) -> Tensor:
        resolved = resolve_tensor_backend(
            self.backend,
            tensors=(m, msa_attention_mask),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            return dispatch_outer_product_mean(
                m.transpose(1, 2),
                msa_attention_mask.transpose(1, 2),
                msa_norm_weight=self.msa_norm.weight,
                msa_norm_bias=self.msa_norm.bias,
                proj_left_weight=self.left_projection.weight,
                proj_right_weight=self.right_projection.weight,
                proj_out_weight=self.output_projection.weight,
                proj_out_bias=self.output_projection.bias,
                msa_norm_eps=self.msa_norm.eps,
                denom_eps=0.0,
                divide_before_projection=self.divide_before_projection,
                clamp_denominator=True,
                backend="oops",
            )
        m_norm = self.msa_norm(m)
        mask = msa_attention_mask.unsqueeze(-1).to(m_norm.dtype)
        a = self.left_projection(m_norm) * mask
        b = self.right_projection(m_norm) * mask
        mask_f = msa_attention_mask.to(a.dtype)
        n_valid = (mask_f @ mask_f.transpose(-1, -2)).unsqueeze(-1).clamp(min=1.0)
        if self._chunk_size is None:
            outer = torch.einsum("bimc,bjmd->bijcd", a, b).flatten(-2)
            if self.divide_before_projection:
                return self.output_projection(outer / n_valid)
            return self.output_projection(outer) / n_valid
        # Chunk along the left (i) axis so the peak einsum intermediate is
        # [B, chunk, L, c, d] instead of [B, L, L, c, d].
        L = a.shape[1]
        out_chunks: list[Tensor] = []
        for s in range(0, L, self._chunk_size):
            e = min(s + self._chunk_size, L)
            outer_chunk = torch.einsum("bimc,bjmd->bijcd", a[:, s:e], b).flatten(-2)
            if self.divide_before_projection:
                out_chunks.append(self.output_projection(outer_chunk / n_valid[:, s:e]))
            else:
                out_chunks.append(self.output_projection(outer_chunk) / n_valid[:, s:e])
        return torch.cat(out_chunks, dim=1)


class EsmFold2MsaPairWeightedAveraging(nn.Module):
    r"""
    Pair-biased MSA row update (AF3 Supplement Algorithm 10).
    """

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        backend: str = "auto",
        num_attention_heads: int = 8,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.backend = backend
        self.msa_norm = LayerNorm(msa_size)
        self.pair_norm = LayerNorm(pair_size)
        self.pair_bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.value_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.gate_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.output_projection = nn.Linear(num_attention_heads * head_dim, msa_size, bias=False)

    def forward(self, msa_representation: Tensor, pair: Tensor, pair_attention_mask: Tensor) -> Tensor:
        resolved = resolve_tensor_backend(
            self.backend,
            tensors=(msa_representation, pair, pair_attention_mask),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            return dispatch_pair_weighted_average(
                msa_representation.transpose(1, 2),
                pair,
                pair_attention_mask,
                msa_norm_weight=self.msa_norm.weight,
                msa_norm_bias=self.msa_norm.bias,
                pair_norm_weight=self.pair_norm.weight,
                pair_norm_bias=self.pair_norm.bias,
                value_weight=self.value_projection.weight,
                gate_weight=self.gate_projection.weight,
                pair_logits_weight=self.pair_bias_projection.weight,
                output_weight=self.output_projection.weight,
                num_attention_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                msa_norm_eps=self.msa_norm.eps,
                pair_norm_eps=self.pair_norm.eps,
                backend="oops",
            ).transpose(1, 2)

        B, L, M, _ = msa_representation.shape
        h, dh = self.num_attention_heads, self.head_dim

        msa_normed = self.msa_norm(msa_representation)
        bias = self.pair_bias_projection(self.pair_norm(pair))  # [B, L, L, num_attention_heads]
        bias.masked_fill_(~pair_attention_mask.unsqueeze(-1).bool(), torch.finfo(bias.dtype).min)
        attn = torch.softmax(bias, dim=-2)  # softmax over j

        v = self.value_projection(msa_normed).reshape(B, L, M, h, dh)
        gate = torch.sigmoid(self.gate_projection(msa_normed)).reshape(B, L, M, h, dh)

        output = torch.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, gate)
        return self.output_projection(output.reshape(B, L, M, h * dh))


_NONPOLYMER_ID = 3


class EsmFold2Transition(nn.Module):
    r"""
    Normalized transition update branch without an outer residual.
    """

    def __init__(
        self,
        hidden_size: int,
        backend: str = "auto",
        expansion: int = 4,
        *,
        chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.input_norm = LayerNorm(hidden_size, eps=eps)
        self.feed_forward = SwiGluFeedForward(hidden_size, expansion * hidden_size)
        self._chunk_size = chunk_size

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        resolved = resolve_tensor_backend(
            self.backend,
            tensors=(x,),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            return dispatch_transition(
                x,
                input_norm_weight=self.input_norm.weight,
                input_norm_bias=self.input_norm.bias,
                gate_weight=self.feed_forward.gate_projection.weight,
                up_weight=self.feed_forward.value_projection.weight,
                down_weight=self.feed_forward.output_projection.weight,
                input_norm_eps=self.input_norm.eps,
                backend="oops",
            )
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            return self.feed_forward(self.input_norm(x))
        updates: list[Tensor] = []
        for s in range(0, x.shape[1], self._chunk_size):
            e = min(s + self._chunk_size, x.shape[1])
            chunk = x[:, s:e]
            updates.append(self.feed_forward(self.input_norm(chunk)))
        return torch.cat(updates, dim=1)


class EsmFold2PairBiasAttention(nn.Module):
    r"""
    Gated multi-head attention with pair bias conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        pair_size: int,
        num_attention_heads: int,
        conditioning_size: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5
        self.adaptive_norm = AdaptiveLayerNorm(
            hidden_size,
            conditioning_size,
            backend=backend,
        )
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, eps=1e-5)
        self.pair_bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size, bias=True)
        nn.init.zeros_(self.output_gate_projection.weight)
        nn.init.constant_(self.output_gate_projection.bias, -2.0)

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        attention_mask: Tensor,
        num_diffusion_samples: int = 1,
    ) -> Tensor:
        batch_size, num_queries, hidden_size = hidden.shape
        normalized_hidden = self.adaptive_norm(hidden, conditioning)

        num_keys = normalized_hidden.shape[1]
        query = self.query_projection(normalized_hidden).view(
            batch_size,
            num_queries,
            self.num_attention_heads,
            self.head_dim,
        )
        key = self.key_projection(normalized_hidden).view(
            batch_size,
            num_keys,
            self.num_attention_heads,
            self.head_dim,
        )
        value = self.value_projection(normalized_hidden).view(
            batch_size,
            num_keys,
            self.num_attention_heads,
            self.head_dim,
        )

        if num_diffusion_samples > 1:
            pair = pair.repeat_interleave(num_diffusion_samples, dim=0)
        if num_diffusion_samples > 1:
            attention_mask = attention_mask.repeat_interleave(num_diffusion_samples, dim=0)

        gate = torch.sigmoid(self.gate_projection(normalized_hidden)).view(
            batch_size,
            num_queries,
            self.num_attention_heads,
            self.head_dim,
        )
        pair_bias = self.pair_bias_projection(self.pair_norm(pair)).permute(0, 3, 1, 2)
        context = projected_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            biases=(pair_bias,),
            attention_mask=attention_mask.bool()[:, None, None, :],
            backend=self.backend,
            scale=self.scale,
        ).transpose(1, 2)
        context = gate * context
        output = self.output_projection(context.reshape(batch_size, num_queries, hidden_size))

        return torch.sigmoid(self.output_gate_projection(conditioning)) * output


class EsmFold2SpatialAtomAttention(nn.Module):
    r"""
    Sliding-window self-attention with three-dimensional RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        backend: str = "auto",
        half_window: int = 64,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5
        self.half_window = half_window

        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        batch_size, num_atoms = x.shape[:2]

        attention_input = x
        query = self.query_projection(x).view(batch_size, num_atoms, self.num_attention_heads, self.head_dim)
        key = self.key_projection(x).view(batch_size, num_atoms, self.num_attention_heads, self.head_dim)
        value = self.value_projection(x).view(batch_size, num_atoms, self.num_attention_heads, self.head_dim)
        query, key = _qk_norm(query), _qk_norm(key)

        query = _apply_rotary_emb_3d(query, attention_cos, attention_sin)
        key = _apply_rotary_emb_3d(key, attention_cos, attention_sin)

        valid = atom_mask
        rank = torch.cumsum(valid, dim=1) - 1
        within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window
        allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
        allowed |= torch.eye(num_atoms, dtype=torch.bool, device=query.device)
        output = projected_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attention_mask=allowed.unsqueeze(1),
            backend=self.backend,
            scale=self.scale,
        ).transpose(1, 2)
        output = output * valid.unsqueeze(-1).unsqueeze(-1)

        output = output.reshape(batch_size, num_atoms, -1)
        output = output * torch.sigmoid(self.gate_projection(attention_input))
        return self.output_projection(output)


class EsmFold2DiffusionConditioning(nn.Module):
    r"""
    Conditions pair and single representations on noise timestep.
    """

    def __init__(
        self,
        backend: str = "auto",
        pair_size: int = 256,
        single_size: int = 768,
        input_size: int = 451,
        sigma_data: float = 16.0,
        fourier_embedding_size: int = 256,
        expansion: int = 2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)

        self.pair_input_norm = LayerNorm(2 * pair_size, eps=layer_norm_eps)
        self.pair_projection = nn.Linear(2 * pair_size, pair_size, bias=False)
        self.pair_transitions = nn.ModuleList(
            EsmFold2Transition(
                pair_size,
                backend,
                expansion=expansion,
                chunk_size=None,
                eps=layer_norm_eps,
            )
            for _ in range(2)
        )

        self.single_input_norm = LayerNorm(input_size, eps=layer_norm_eps)
        self.single_projection = nn.Linear(input_size, single_size, bias=False)
        self.noise_embedding = EsmFold2FourierEmbedding(fourier_embedding_size)
        self.noise_norm = LayerNorm(fourier_embedding_size, eps=layer_norm_eps)
        self.noise_projection = nn.Linear(fourier_embedding_size, single_size, bias=False)
        self.single_transitions = nn.ModuleList(
            EsmFold2Transition(
                single_size,
                backend,
                expansion=expansion,
                chunk_size=None,
                eps=layer_norm_eps,
            )
            for _ in range(2)
        )

    def condition_pair(
        self,
        pair: Tensor,
        relative_position_encoding: Tensor,
    ) -> Tensor:
        dtype = self.pair_projection.weight.dtype
        relative_position_encoding = relative_position_encoding.to(dtype=dtype)
        conditioned_pair = torch.cat([pair.to(dtype=dtype), relative_position_encoding], dim=-1)
        conditioned_pair = self.pair_projection(self.pair_input_norm(conditioned_pair))
        for transition in self.pair_transitions:
            conditioned_pair = conditioned_pair + transition(conditioned_pair)
        return conditioned_pair

    def condition_single(
        self,
        noise_level: Tensor,
        single_inputs: Tensor,
        num_diffusion_samples: int = 1,
    ) -> Tensor:
        expanded_single_inputs = (
            single_inputs if num_diffusion_samples == 1 else single_inputs.repeat_interleave(num_diffusion_samples, 0)
        )
        conditioned_single = self.single_projection(
            self.single_input_norm(expanded_single_inputs.to(dtype=self.single_projection.weight.dtype))
        )

        noise_level = noise_level.to(dtype=torch.float32, device=conditioned_single.device).reshape(-1)
        embedded_noise_level = 0.25 * torch.log((noise_level / self.sigma_data).clamp(min=1e-20))
        noise_embedding = self.noise_embedding(embedded_noise_level)
        noise_embedding = self.noise_projection(self.noise_norm(noise_embedding))
        conditioned_single = conditioned_single + noise_embedding.unsqueeze(1)

        for transition in self.single_transitions:
            conditioned_single = conditioned_single + transition(conditioned_single)
        return conditioned_single

    def forward(
        self,
        noise_level: Tensor,
        single_inputs: Tensor,
        pair: Tensor,
        relative_position_encoding: Tensor,
        num_diffusion_samples: int = 1,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.condition_single(noise_level, single_inputs, num_diffusion_samples),
            self.condition_pair(pair, relative_position_encoding),
        )


class EsmFold2RowAttentionPooling(nn.Module):
    r"""
    Pool each pair row into a single representation.
    """

    def __init__(self, pair_size: int, single_size: int) -> None:
        super().__init__()
        self.attention_projection = nn.Linear(pair_size, 1, bias=False)
        self.output_projection = nn.Linear(pair_size, single_size, bias=False)

    def forward(self, pair: Tensor, mask: Tensor) -> Tensor:
        scores = self.attention_projection(pair).squeeze(-1)
        mask_bias = torch.where(
            mask[:, None, :].bool(),
            torch.zeros_like(scores),
            torch.full_like(scores, torch.finfo(scores.dtype).min),
        )
        scores = scores + mask_bias
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, pair)
        return self.output_projection(pooled)


class EsmFold2FourierEmbedding(nn.Module):
    r"""
    Embed noise levels with learned Fourier frequencies and phases.
    """

    frequencies: Tensor
    phases: Tensor

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(embedding_size))
        self.register_buffer("phases", torch.randn(embedding_size))

    def forward(self, noise_level: Tensor) -> Tensor:
        noise_level = noise_level.to(device=self.frequencies.device, dtype=self.frequencies.dtype).reshape(-1)
        return torch.cos(2.0 * torch.pi * (noise_level[:, None] * self.frequencies[None, :] + self.phases[None, :]))


class EsmFold2DiffusionStructureHead(nn.Module):
    r"""
    Diffusion score network and coordinate sampler.
    """

    def __init__(
        self,
        config: EsmFold2Config | EsmFold2ExperimentalConfig,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        denoiser_config = config.diffusion_structure_head.denoiser
        atom_encoder_config = config.input_embeddings.atom_encoder
        structure_head_config = config.diffusion_structure_head

        self.denoiser = EsmFold2Denoiser(
            backend=backend,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            token_hidden_size=denoiser_config.token_hidden_size,
            pair_size=denoiser_config.pair_size,
            input_size=denoiser_config.input_size,
            sigma_data=denoiser_config.sigma_data,
            fourier_embedding_size=denoiser_config.fourier_embedding_size,
            num_atom_blocks=denoiser_config.num_atom_blocks,
            num_atom_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            expansion=denoiser_config.expansion,
            sliding_window_size=atom_encoder_config.sliding_window_size,
            spatial_rope_base_frequency=atom_encoder_config.spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=atom_encoder_config.num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=atom_encoder_config.num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=atom_encoder_config.reference_space_rope_base_frequency,
        )

        # Sampling hyperparameters
        self.sigma_data = denoiser_config.sigma_data
        self.churn_rate = structure_head_config.churn_rate
        self.min_churn_noise_level = structure_head_config.min_churn_noise_level
        self.noise_scale = structure_head_config.noise_scale
        self.step_scale = structure_head_config.step_scale
        self.max_noise_level = structure_head_config.max_noise_level
        self.min_noise_level = structure_head_config.min_noise_level
        self.rho = structure_head_config.rho
        self.inference_num_steps = structure_head_config.num_sampling_steps

    def inference_noise_schedule(self, num_steps: int | None = None, device: torch.device | None = None) -> Tensor:
        steps = self.inference_num_steps if num_steps is None else num_steps
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        if steps == 1:
            return torch.tensor(
                [self.max_noise_level * self.sigma_data, 0.0],
                device=device,
                dtype=torch.float32,
            )
        inverse_rho = 1.0 / self.rho
        step_indices = torch.arange(steps, device=device, dtype=torch.float32)
        base = self.max_noise_level**inverse_rho + (step_indices / (steps - 1)) * (
            self.min_noise_level**inverse_rho - self.max_noise_level**inverse_rho
        )
        schedule = self.sigma_data * base.pow(self.rho)
        return F.pad(schedule, (0, 1), value=0.0)

    @staticmethod
    def _random_rotations(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        quaternion = torch.randn((batch_size, 4), dtype=dtype, device=device, generator=generator)
        scale = torch.sqrt((quaternion * quaternion).sum(dim=1))
        signs = torch.where(quaternion[:, 0] < 0, -scale, scale)
        quaternion = quaternion / signs[:, None]
        real, axis_x, axis_y, axis_z = torch.unbind(quaternion, dim=-1)
        inverse_norm_scale = 2.0 / (quaternion * quaternion).sum(dim=-1)
        return torch.stack(
            (
                1 - inverse_norm_scale * (axis_y * axis_y + axis_z * axis_z),
                inverse_norm_scale * (axis_x * axis_y - axis_z * real),
                inverse_norm_scale * (axis_x * axis_z + axis_y * real),
                inverse_norm_scale * (axis_x * axis_y + axis_z * real),
                1 - inverse_norm_scale * (axis_x * axis_x + axis_z * axis_z),
                inverse_norm_scale * (axis_y * axis_z - axis_x * real),
                inverse_norm_scale * (axis_x * axis_z - axis_y * real),
                inverse_norm_scale * (axis_y * axis_z + axis_x * real),
                1 - inverse_norm_scale * (axis_x * axis_x + axis_y * axis_y),
            ),
            dim=-1,
        ).reshape(batch_size, 3, 3)

    def center_random_augmentation(
        self,
        coordinates: Tensor,
        atom_mask: Tensor,
        second_coordinates: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size = coordinates.shape[0]
        mask = atom_mask.unsqueeze(-1)  # [B, A, 1]
        denominator = mask.sum(dim=1, keepdim=True).clamp(min=1)
        center = (coordinates * mask).sum(dim=1, keepdim=True) / denominator
        coordinates = coordinates - center
        if second_coordinates is not None:
            second_coordinates = second_coordinates - center

        rotation = self._random_rotations(batch_size, coordinates.dtype, coordinates.device, generator)
        coordinates = torch.einsum("bmd,bds->bms", coordinates, rotation)
        if second_coordinates is not None:
            second_coordinates = torch.einsum("bmd,bds->bms", second_coordinates, rotation)

        translation = torch.randn_like(coordinates[:, 0:1, :], generator=generator)
        coordinates = coordinates + translation
        if second_coordinates is not None:
            second_coordinates = second_coordinates + translation
        return coordinates, second_coordinates

    @staticmethod
    def _weighted_rigid_align(
        coordinates: Tensor,
        target_coordinates: Tensor,
        weights: Tensor,
        mask: Tensor,
    ) -> Tensor:
        weights = (mask * weights).unsqueeze(-1)  # [B, N, 1]
        denominator = weights.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        center = (coordinates * weights).sum(dim=-2, keepdim=True) / denominator
        target_center = (target_coordinates * weights).sum(dim=-2, keepdim=True) / denominator
        centered_coordinates = coordinates - center
        centered_target_coordinates = target_coordinates - target_center
        covariance = torch.einsum(
            "bni,bnj->bij",
            weights * centered_target_coordinates,
            centered_coordinates,
        )
        covariance_float = covariance.float()
        left_vectors, _, right_vectors = torch.linalg.svd(
            covariance_float,
            driver="gesvd" if covariance_float.is_cuda else None,
        )
        determinant = torch.linalg.det(left_vectors @ right_vectors)
        ones = torch.ones_like(determinant)
        rotation = (left_vectors @ torch.diag_embed(torch.stack([ones, ones, determinant], dim=-1)) @ right_vectors).to(
            covariance.dtype
        )
        return centered_coordinates @ rotation.transpose(-1, -2) + target_center

    @torch.inference_mode()
    def sample(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        relative_position_encoding: Tensor,
        reference_positions: Tensor,
        reference_charges: Tensor,
        atom_mask: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
        num_diffusion_samples: int = 1,
        num_sampling_steps: int | None = None,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        return_atom_representations: bool = False,
        denoising_early_exit_rmsd: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | None]:
        if num_diffusion_samples <= 0:
            raise ValueError("num_diffusion_samples must be positive")
        num_atoms = atom_to_token.shape[1]
        device = single_inputs.device
        sample_batch_size = single_inputs.shape[0] * num_diffusion_samples
        conditioned_pair = self.denoiser.conditioning.condition_pair(
            pair,
            relative_position_encoding,
        )
        (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        ) = self.denoiser.atom_encoder.encode_reference(
            reference_positions,
            atom_mask,
            reference_space_id,
            reference_charges,
            reference_element_features,
            reference_atom_name_character_features,
            atom_to_token,
            num_diffusion_samples,
        )

        schedule = self.inference_noise_schedule(num_sampling_steps, device)
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= float(max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))

        applied_noise_scale = self.noise_scale if noise_scale is None else float(noise_scale)
        applied_step_scale = self.step_scale if step_scale is None else float(step_scale)

        coordinates = schedule[0] * torch.randn(
            sample_batch_size,
            num_atoms,
            3,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        sampling_atom_mask = atom_mask.repeat_interleave(num_diffusion_samples, 0).float()

        gammas = torch.where(
            schedule > self.min_churn_noise_level,
            torch.full_like(schedule, self.churn_rate),
            torch.zeros_like(schedule),
        )

        previous_denoised_coordinates: Tensor | None = None
        token_representations: Tensor | None = None
        atom_representations: Tensor | None = None

        step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))
        num_steps = len(step_pairs)

        # The schedule is truncated at max_inference_sigma, and sampling may stop when
        # denoised-coordinate RMSD stabilizes, so the loop has a data-dependent step count.
        for step_index, (current_noise_level, next_noise_level, churn_rate) in enumerate(step_pairs):
            coordinates, previous_denoised_coordinates = self.center_random_augmentation(
                coordinates,
                sampling_atom_mask,
                second_coordinates=previous_denoised_coordinates,
                generator=generator,
            )

            current_noise = float(current_noise_level.item())
            augmented_noise = current_noise * (1.0 + float(churn_rate.item()))
            churn_noise_std = applied_noise_scale * max(augmented_noise**2 - current_noise**2, 0.0) ** 0.5
            noisy_coordinates = coordinates + churn_noise_std * torch.randn_like(coordinates, generator=generator)

            is_last_step = step_index == num_steps - 1
            request_atom_representations = return_atom_representations and (
                is_last_step or denoising_early_exit_rmsd is not None
            )

            denoiser_output = self.denoiser.denoise(
                noisy_coordinates=noisy_coordinates,
                noise_level=torch.full((sample_batch_size,), augmented_noise, device=device, dtype=torch.float32),
                single_inputs=single_inputs,
                conditioned_pair=conditioned_pair,
                token_mask=token_mask,
                reference_atom_features=reference_atom_features,
                attention_cos=attention_cos,
                attention_sin=attention_sin,
                expanded_atom_mask=expanded_atom_mask,
                expanded_atom_to_token=expanded_atom_to_token,
                num_diffusion_samples=num_diffusion_samples,
                return_token_representations=True,
                return_atom_representations=request_atom_representations,
            )

            denoised_coordinates = cast(Tensor, denoiser_output["denoised_coordinates"])
            token_representations = cast(Tensor, denoiser_output["token_representations"])
            if request_atom_representations:
                atom_representations = cast(Tensor | None, denoiser_output["atom_intermediates"])

            # Reverse diffusion alignment (Kabsch)
            with torch.autocast(device_type="cuda", enabled=False):
                noisy_coordinates = self._weighted_rigid_align(
                    noisy_coordinates.float(),
                    denoised_coordinates.float(),
                    sampling_atom_mask,
                    sampling_atom_mask,
                )
            noisy_coordinates = noisy_coordinates.to(dtype=denoised_coordinates.dtype)

            # ODE/SDE step
            next_noise = float(next_noise_level.item())
            derivative = (noisy_coordinates - denoised_coordinates) / augmented_noise
            coordinates = noisy_coordinates + applied_step_scale * (next_noise - augmented_noise) * derivative

            # Denoising early-exit: stop when consecutive predictions converge
            if denoising_early_exit_rmsd is not None and previous_denoised_coordinates is not None and step_index >= 1:
                with torch.autocast(device_type="cuda", enabled=False):
                    aligned = self._weighted_rigid_align(
                        previous_denoised_coordinates.float(),
                        denoised_coordinates.float(),
                        sampling_atom_mask,
                        sampling_atom_mask,
                    )
                coordinate_difference = (denoised_coordinates.float() - aligned) * sampling_atom_mask.unsqueeze(-1)
                per_sample_rmsd = (
                    coordinate_difference.pow(2).sum(dim=(-1, -2)) / sampling_atom_mask.sum(dim=-1).clamp(min=1)
                ).sqrt()
                if per_sample_rmsd.max().item() < denoising_early_exit_rmsd:
                    coordinates = denoised_coordinates
                    previous_denoised_coordinates = denoised_coordinates
                    break

            previous_denoised_coordinates = denoised_coordinates

        result: dict[str, Tensor | None] = {
            "coordinates": coordinates,
            "token_representations": token_representations,
        }
        if return_atom_representations:
            result["atom_representations"] = atom_representations
        return result


class EsmFold2ConfidenceHead(nn.Module):
    r"""
    Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins.
    """

    distance_boundaries: Tensor

    def __init__(self, config: EsmFold2Config, backend: str = "auto") -> None:
        super().__init__()
        self.backend = backend
        confidence_config = config.confidence
        single_size = config.single_size
        pair_size = config.pair_size
        input_size = config.input_embeddings.input_size

        distance_boundaries = torch.linspace(
            confidence_config.min_distance,
            confidence_config.max_distance,
            confidence_config.num_distogram_bins - 1,
        )
        self.register_buffer("distance_boundaries", distance_boundaries)
        self.input_norm = LayerNorm(input_size)
        self.pair_norm = LayerNorm(pair_size)
        self.left_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.right_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.left_product_projection = nn.Linear(input_size, pair_size, bias=False)
        self.right_product_projection = nn.Linear(input_size, pair_size, bias=False)
        self.product_output_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.distance_embedding = nn.Embedding(confidence_config.num_distogram_bins, pair_size)

        self.row_attention_pooling = EsmFold2RowAttentionPooling(pair_size, single_size)

        folding_trunk_config = confidence_config.folding_trunk
        self.folding_trunk = EsmFold2FoldingTrunk(
            folding_trunk_config.num_blocks,
            pair_size,
            expansion=4,
            backend=backend,
        )

        # Heads.
        self.plddt_norm = LayerNorm(single_size)
        max_atoms_per_token = 23
        self.plddt_projection_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, single_size, confidence_config.num_plddt_bins)
        )

        self.pae_norm = LayerNorm(pair_size)
        self.pae_projection = nn.Linear(pair_size, confidence_config.num_pae_bins, bias=False)

        self.pde_norm = LayerNorm(pair_size)
        self.pde_projection = nn.Linear(pair_size, confidence_config.num_pde_bins, bias=False)

        self.resolved_norm = LayerNorm(single_size)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_projection_weight = nn.Parameter(torch.zeros(max_atoms_per_token, single_size, 2))

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        return x if num_diffusion_samples == 1 else x.repeat_interleave(num_diffusion_samples, 0)

    def forward(
        self,
        single_inputs: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        distogram_atom_index: Tensor,
        token_mask: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        chain_index: Tensor,
        molecule_type: Tensor,
        relative_position_encoding: Tensor,
        token_bonds_encoding: Tensor,
        num_diffusion_samples: int = 1,
    ) -> dict[str, Tensor]:
        normalized_single_inputs = self.input_norm(single_inputs)

        base_pair = self.pair_norm(pair)
        base_pair = base_pair + relative_position_encoding + token_bonds_encoding
        base_pair = base_pair + self.left_pair_projection(normalized_single_inputs).unsqueeze(2)
        base_pair = base_pair + self.right_pair_projection(normalized_single_inputs).unsqueeze(1)
        base_pair = base_pair + self.product_output_projection(
            self.left_product_projection(normalized_single_inputs)[:, :, None, :]
            * self.right_product_projection(normalized_single_inputs)[:, None, :, :]
        )

        pair = self._repeat_batch(base_pair, num_diffusion_samples)
        expanded_atom_to_token = self._repeat_batch(atom_to_token, num_diffusion_samples)
        expanded_atom_mask = self._repeat_batch(atom_mask, num_diffusion_samples)
        expanded_representative_atom_index = self._repeat_batch(distogram_atom_index, num_diffusion_samples).long()
        expanded_token_mask = self._repeat_batch(token_mask, num_diffusion_samples)
        batch_size = pair.shape[0]

        representative_coordinates = _gather_representative_atom_coordinates(
            coordinates,
            expanded_representative_atom_index,
        )
        representative_distances = torch.cdist(
            representative_coordinates, representative_coordinates, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (representative_distances.unsqueeze(-1) > self.distance_boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram_bins)

        pair_mask = expanded_token_mask[:, :, None].float() * expanded_token_mask[:, None, :].float()

        pair_delta = self.folding_trunk(pair, pair_attention_mask=pair_mask)
        pair.add_(pair_delta)
        del pair_delta
        single = self.row_attention_pooling(pair, expanded_token_mask)

        atom_mask_float = expanded_atom_mask.float()
        atom_single = _gather_token_to_atom(
            single,
            expanded_atom_to_token,
            backend=self.backend,
        )
        normalized_atom_single = self.plddt_norm(atom_single)

        intra_token_index = _compute_intra_token_index(expanded_atom_to_token)
        intra_token_index = intra_token_index.clamp(max=self.plddt_projection_weight.shape[0] - 1)
        plddt_weights = self.plddt_projection_weight[intra_token_index]
        plddt_logits = torch.einsum("...c,...cb->...b", normalized_atom_single, plddt_weights)
        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        num_tokens = single.shape[1]
        plddt_sum = torch.zeros(batch_size, num_tokens, device=single.device, dtype=plddt_per_atom.dtype)
        atom_count = torch.zeros(batch_size, num_tokens, device=single.device, dtype=plddt_per_atom.dtype)
        plddt_atom_mask = atom_mask_float.to(plddt_per_atom.dtype)
        plddt_sum.scatter_add_(1, expanded_atom_to_token, plddt_per_atom * plddt_atom_mask)
        atom_count.scatter_add_(1, expanded_atom_to_token, plddt_atom_mask)
        plddt = plddt_sum / atom_count.clamp(min=_CONFIDENCE_EPSILON)

        mean_plddt = (plddt_per_atom * atom_mask_float).sum(dim=-1) / (
            atom_mask_float.sum(dim=-1) + _CONFIDENCE_EPSILON
        )

        expanded_type = self._repeat_batch(molecule_type, num_diffusion_samples)
        expanded_chain_index = self._repeat_batch(chain_index, num_diffusion_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).float()
        inter_chain = (expanded_chain_index.unsqueeze(-1) != expanded_chain_index.unsqueeze(-2)).float()
        near_contact = (representative_distances < 8).float()
        interface_per_token = (near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(dim=-1)
        iplddt_weight = torch.where(
            is_ligand.bool(),
            torch.full_like(interface_per_token, 2.0),
            interface_per_token,
        )
        iplddt_weight_atoms = _gather_token_to_atom(
            iplddt_weight.unsqueeze(-1),
            expanded_atom_to_token,
            backend=self.backend,
        ).squeeze(-1)
        atom_iplddt_weight = atom_mask_float * iplddt_weight_atoms
        complex_iplddt = (plddt_per_atom * atom_iplddt_weight).sum(dim=-1) / (
            atom_iplddt_weight.sum(dim=-1) + _CONFIDENCE_EPSILON
        )

        plddt_ca = plddt_per_atom.gather(1, expanded_representative_atom_index)

        # PAE
        pae_logits = self.pae_projection(self.pae_norm(pair))
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0)

        # PDE
        pde_logits = self.pde_projection(self.pde_norm(pair))
        pde = _categorical_mean(pde_logits, start=0.0, end=32.0)

        # Resolved (per-atom binary).
        resolved_atom_single = self.resolved_norm(atom_single)
        resolved_weights = self.resolved_projection_weight[intra_token_index]
        resolved_logits = torch.einsum("...c,...cb->...b", resolved_atom_single, resolved_weights)

        # pTM / ipTM from pae_logits.
        num_bins = pae_logits.shape[-1]
        bin_width = 32.0 / num_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        token_mask_float = expanded_token_mask.float()
        num_residues = token_mask_float.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (num_residues.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = token_mask_float.unsqueeze(-1) * token_mask_float.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (pair_mask_2d.sum(dim=-1) + _CONFIDENCE_EPSILON)
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (
            expanded_chain_index.unsqueeze(-1) != expanded_chain_index.unsqueeze(-2)
        ).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + _CONFIDENCE_EPSILON
        )
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_index = int(expanded_chain_index.max().item()) if batch_size > 0 else 0
        num_chains = max_chain_index + 1
        chain_pair_iptm = torch.zeros(
            batch_size,
            num_chains,
            num_chains,
            device=tm_expected.device,
            dtype=tm_expected.dtype,
        )
        for row_chain in range(num_chains):
            row_chain_mask = (expanded_chain_index == row_chain).float() * token_mask_float
            if row_chain_mask.sum() == 0:
                continue
            for column_chain in range(num_chains):
                column_chain_mask = (expanded_chain_index == column_chain).float() * token_mask_float
                chain_pair_mask = row_chain_mask.unsqueeze(-1) * column_chain_mask.unsqueeze(-2)
                denominator = chain_pair_mask.sum(dim=(-1, -2)) + _CONFIDENCE_EPSILON
                chain_pair_iptm[:, row_chain, column_chain] = (tm_expected * chain_pair_mask).sum(
                    dim=(-1, -2)
                ) / denominator

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt,
            "plddt_per_atom": plddt_per_atom,
            "plddt_ca": plddt_ca,
            "mean_plddt": mean_plddt,
            "complex_iplddt": complex_iplddt,
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm,
            "iptm": iptm,
            "chain_pair_iptm": chain_pair_iptm,
        }


class EsmFold2ExperimentalConfidenceHead(nn.Module):
    r"""
    Narrow confidence head of the experimental recurrent ESMFold2 graph.
    """

    distance_boundaries: Tensor

    def __init__(self, config: EsmFold2ExperimentalConfig, backend: str = "auto") -> None:
        super().__init__()
        self.backend = backend
        confidence_config = config.confidence
        single_size = config.single_size
        pair_size = config.pair_size
        input_size = config.input_embeddings.input_size

        distance_boundaries = torch.linspace(
            confidence_config.min_distance,
            confidence_config.max_distance,
            confidence_config.num_distogram_bins - 1,
        )
        self.register_buffer("distance_boundaries", distance_boundaries)
        self.input_norm = LayerNorm(input_size)
        self.pair_norm = LayerNorm(pair_size)
        self.left_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.right_pair_projection = nn.Linear(input_size, pair_size, bias=False)
        self.left_product_projection = nn.Linear(input_size, pair_size, bias=False)
        self.right_product_projection = nn.Linear(input_size, pair_size, bias=False)
        self.product_output_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.distance_embedding = nn.Embedding(confidence_config.num_distogram_bins, pair_size)

        self.row_attention_pooling = EsmFold2RowAttentionPooling(pair_size, single_size)

        folding_trunk_config = confidence_config.folding_trunk
        self.folding_trunk = EsmFold2FoldingTrunk(
            folding_trunk_config.num_blocks,
            pair_size,
            expansion=4,
            backend=backend,
        )

        self.plddt_norm = LayerNorm(single_size)
        max_atoms_per_token = 23
        self.plddt_projection_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, single_size, confidence_config.num_plddt_bins)
        )

        self.pae_projection = nn.Linear(pair_size, confidence_config.num_pae_bins, bias=False)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        return x if num_diffusion_samples == 1 else x.repeat_interleave(num_diffusion_samples, 0)

    def forward(
        self,
        single_inputs: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        distogram_atom_index: Tensor,
        token_mask: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        chain_index: Tensor,
        molecule_type: Tensor,
        relative_position_encoding: Tensor,
        token_bonds_encoding: Tensor,
        num_diffusion_samples: int = 1,
    ) -> EsmFold2ExperimentalConfidence:
        normalized_single_inputs = self.input_norm(single_inputs)

        base_pair = self.pair_norm(pair)
        base_pair = base_pair + relative_position_encoding + token_bonds_encoding
        base_pair = base_pair + self.left_pair_projection(normalized_single_inputs).unsqueeze(2)
        base_pair = base_pair + self.right_pair_projection(normalized_single_inputs).unsqueeze(1)
        base_pair = base_pair + self.product_output_projection(
            self.left_product_projection(normalized_single_inputs)[:, :, None, :]
            * self.right_product_projection(normalized_single_inputs)[:, None, :, :]
        )

        pair = self._repeat_batch(base_pair, num_diffusion_samples)
        expanded_atom_to_token = self._repeat_batch(atom_to_token, num_diffusion_samples)
        expanded_atom_mask = self._repeat_batch(atom_mask, num_diffusion_samples)
        expanded_representative_atom_index = self._repeat_batch(distogram_atom_index, num_diffusion_samples).long()
        expanded_token_mask = self._repeat_batch(token_mask, num_diffusion_samples)
        batch_size = pair.shape[0]

        representative_coordinates = _gather_representative_atom_coordinates(
            coordinates,
            expanded_representative_atom_index,
        )
        representative_distances = torch.cdist(
            representative_coordinates, representative_coordinates, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (representative_distances.unsqueeze(-1) > self.distance_boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram_bins)

        pair_mask = expanded_token_mask[:, :, None].float() * expanded_token_mask[:, None, :].float()

        pair_delta = self.folding_trunk(pair, pair_attention_mask=pair_mask)
        pair = pair + pair_delta
        del pair_delta
        single = self.row_attention_pooling(pair, expanded_token_mask)

        atom_mask_float = expanded_atom_mask.float()
        atom_single = _gather_token_to_atom(
            single,
            expanded_atom_to_token,
            backend=self.backend,
        )
        normalized_atom_single = self.plddt_norm(atom_single)

        intra_token_index = _compute_intra_token_index(expanded_atom_to_token)
        intra_token_index = intra_token_index.clamp(max=self.plddt_projection_weight.shape[0] - 1)
        plddt_weights = self.plddt_projection_weight[intra_token_index]
        plddt_logits = torch.einsum("...c,...cb->...b", normalized_atom_single, plddt_weights)
        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        num_tokens = single.shape[1]
        plddt_sum = torch.zeros(batch_size, num_tokens, device=single.device, dtype=plddt_per_atom.dtype)
        atom_count = torch.zeros(batch_size, num_tokens, device=single.device, dtype=plddt_per_atom.dtype)
        plddt_atom_mask = atom_mask_float.to(plddt_per_atom.dtype)
        plddt_sum.scatter_add_(1, expanded_atom_to_token, plddt_per_atom * plddt_atom_mask)
        atom_count.scatter_add_(1, expanded_atom_to_token, plddt_atom_mask)
        plddt = plddt_sum / atom_count.clamp(min=torch.finfo(atom_count.dtype).tiny)

        mean_plddt = (plddt_per_atom * atom_mask_float).sum(dim=-1) / (
            atom_mask_float.sum(dim=-1) + torch.finfo(atom_mask_float.dtype).tiny
        )

        expanded_type = self._repeat_batch(molecule_type, num_diffusion_samples)
        expanded_chain_index = self._repeat_batch(chain_index, num_diffusion_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).float()
        inter_chain = (expanded_chain_index.unsqueeze(-1) != expanded_chain_index.unsqueeze(-2)).float()
        near_contact = (representative_distances < 8).float()
        interface_per_token = (near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(dim=-1)
        iplddt_weight = torch.where(
            is_ligand.bool(),
            torch.full_like(interface_per_token, 2.0),
            interface_per_token,
        )
        iplddt_weight_atoms = _gather_token_to_atom(
            iplddt_weight.unsqueeze(-1),
            expanded_atom_to_token,
            backend=self.backend,
        ).squeeze(-1)
        atom_iplddt_weight = atom_mask_float * iplddt_weight_atoms
        complex_iplddt = (plddt_per_atom * atom_iplddt_weight).sum(dim=-1) / (
            atom_iplddt_weight.sum(dim=-1) + torch.finfo(atom_iplddt_weight.dtype).tiny
        )

        plddt_ca = plddt_per_atom.gather(1, expanded_representative_atom_index)

        pae_logits = self.pae_projection(pair)
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0)

        num_bins = pae_logits.shape[-1]
        bin_width = 32.0 / num_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        token_mask_float = expanded_token_mask.float()
        num_residues = token_mask_float.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (num_residues.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = token_mask_float.unsqueeze(-1) * token_mask_float.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (
            pair_mask_2d.sum(dim=-1) + torch.finfo(pair_mask_2d.dtype).tiny
        )
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (
            expanded_chain_index.unsqueeze(-1) != expanded_chain_index.unsqueeze(-2)
        ).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + torch.finfo(inter_chain_mask.dtype).tiny
        )
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_index = int(expanded_chain_index.max().item())
        num_chains = max_chain_index + 1
        chain_pair_iptm = torch.zeros(
            batch_size,
            num_chains,
            num_chains,
            device=tm_expected.device,
            dtype=tm_expected.dtype,
        )
        for row_chain in range(num_chains):
            row_chain_mask = (expanded_chain_index == row_chain).float() * token_mask_float
            for column_chain in range(num_chains):
                column_chain_mask = (expanded_chain_index == column_chain).float() * token_mask_float
                chain_pair_mask = row_chain_mask.unsqueeze(-1) * column_chain_mask.unsqueeze(-2)
                denominator = chain_pair_mask.sum(dim=(-1, -2)) + torch.finfo(chain_pair_mask.dtype).tiny
                chain_pair_iptm[:, row_chain, column_chain] = (tm_expected * chain_pair_mask).sum(
                    dim=(-1, -2)
                ) / denominator

        return EsmFold2ExperimentalConfidence(
            plddt_logits=plddt_logits,
            plddt=plddt,
            plddt_per_atom=plddt_per_atom,
            plddt_ca=plddt_ca,
            mean_plddt=mean_plddt,
            complex_iplddt=complex_iplddt,
            pae_logits=pae_logits,
            pae=pae,
            ptm=ptm,
            iptm=iptm,
            chain_pair_iptm=chain_pair_iptm,
        )


_CHAR_VOCAB_SIZE: int = 64
_MAX_CHARS: int = 4
_XYZ_DIMS: int = 3
_MAX_ATOMIC_NUMBER: int = 128

# Input feature dim = 3 + 1 + 1 + 128 + 64*4 = 389
_ATOM_FEATURE_DIM: int = _XYZ_DIMS + 1 + 1 + _MAX_ATOMIC_NUMBER + _CHAR_VOCAB_SIZE * _MAX_CHARS


_NUM_RESIDUE_TYPES: int = 33

_EPS = 1e-5
_CONFIDENCE_EPSILON = 1e-6


def _maybe_subsample_msa(
    msa_tokens: Tensor,
    msa_attention_mask: Tensor | None,
    has_deletion: Tensor | None,
    deletion_value: Tensor | None,
    *,
    max_depth: int | None,
    enabled: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    r"""
    Randomly subsample the MSA to ``max_depth`` rows, keeping query row 0.

    Accepts encoder-input MSA token tensors shaped ``[B, depth, L]``. No-op when
    disabled, ``max_depth`` is None, or depth <= ``max_depth``.
    """
    if not enabled or max_depth is None:
        return msa_tokens, msa_attention_mask, has_deletion, deletion_value
    depth = msa_tokens.size(1)
    if depth <= 1 or depth <= max_depth:
        return msa_tokens, msa_attention_mask, has_deletion, deletion_value
    indices = torch.zeros(max_depth, dtype=torch.long, device=msa_tokens.device)
    perm = torch.randperm(depth - 1, device=msa_tokens.device, generator=generator)[: max_depth - 1]
    indices[1:] = perm + 1
    indices = indices.sort().values
    msa_tokens = msa_tokens[:, indices]
    if msa_attention_mask is not None:
        msa_attention_mask = msa_attention_mask[:, indices]
    if has_deletion is not None:
        has_deletion = has_deletion[:, indices]
    if deletion_value is not None:
        deletion_value = deletion_value[:, indices]
    return msa_tokens, msa_attention_mask, has_deletion, deletion_value


def _maybe_apply_msa_column_masking(
    msa_attention_mask: Tensor | None,
    rate: float,
    generator: torch.Generator | None = None,
) -> Tensor | None:
    r"""
    Mask fraction ``rate`` of MSA columns in non-query rows of
    ``msa_attention_mask``, keeping query row 0. No-op when absent, ``rate <= 0``,
    or depth <= 1.
    """
    if msa_attention_mask is None or rate <= 0.0 or msa_attention_mask.size(1) <= 1:
        return msa_attention_mask
    B, _M, L = msa_attention_mask.shape
    col_keep = torch.rand(B, L, device=msa_attention_mask.device, generator=generator) >= rate
    masked = msa_attention_mask.bool() & col_keep.unsqueeze(1)
    masked[:, 0, :] = msa_attention_mask[:, 0, :].bool()
    return masked


def _gather_token_to_atom(
    token_features: Tensor,
    atom_to_token: Tensor,
    *,
    backend: str = "eager",
) -> Tensor:
    r"""
    Broadcast per-token features to per-atom features using gather.

    Args:
        token_features: [B, L, d]
        atom_to_token: [B, A] int64

    Returns:
        [B, A, d]
    """
    resolved = resolve_tensor_backend(
        backend,
        tensors=(token_features, atom_to_token),
        supported=("oops", "eager"),
        public_default="eager",
    )
    if resolved == "oops":
        return dispatch_token_to_atom_broadcast(
            token_features,
            atom_to_token,
            backend="oops",
        )
    gather_index = atom_to_token.unsqueeze(-1).expand(-1, -1, token_features.size(-1))
    return torch.gather(token_features, 1, gather_index)


def _scatter_atom_to_token(
    atom_features: Tensor,
    atom_to_token: Tensor,
    n_tokens: int,
    atom_mask: Tensor,
    *,
    backend: str = "eager",
) -> Tensor:
    r"""
    Aggregate per-atom features to per-token features (mean).

    Args:
        atom_features: [B, A, d]
        atom_to_token: [B, A] int64
        n_tokens: L
        atom_mask: [B, A] bool

    Returns:
        [B, L, d]
    """
    resolved = resolve_tensor_backend(
        backend,
        tensors=(atom_features, atom_to_token, atom_mask),
        supported=("oops", "eager"),
        public_default="eager",
    )
    if resolved == "oops":
        return dispatch_atom_to_token_mean(
            atom_features,
            atom_to_token,
            num_tokens=n_tokens,
            atom_mask=atom_mask,
            eps=1e-6,
            backend="oops",
        )
    B, A, d = atom_features.shape
    target_token_index = torch.where(atom_mask, atom_to_token, n_tokens)
    expanded_token_index = target_token_index.unsqueeze(-1).expand(B, A, d)
    out = torch.zeros(B, n_tokens + 1, d, device=atom_features.device, dtype=atom_features.dtype)
    out.scatter_reduce_(1, expanded_token_index, atom_features, reduce="mean", include_self=False)
    return out[:, :n_tokens, :]


def _gather_representative_atom_coordinates(coordinates: Tensor, representative_atom_index: Tensor) -> Tensor:
    r"""
    Gather representative atom coordinates for each token.

    Args:
        coordinates: [B, A, 3]
        representative_atom_index: [B, L] int64

    Returns:
        [B, L, 3]
    """
    coordinate_index = representative_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.size(-1))
    return torch.gather(coordinates, 1, coordinate_index)


def _compute_intra_token_index(atom_to_token: Tensor) -> Tensor:
    r"""
    Compute local atom index within each token (vectorised).

    Atoms belonging to the same token are contiguous, so this computes a
    running count that resets at each token boundary.

    Args:
        atom_to_token: [B, A] flat index mapping each atom to its token.

    Returns:
        [B, A] tensor with values in [0, max_atoms_per_token - 1].
    """
    same_as_prev = F.pad(atom_to_token[:, 1:] == atom_to_token[:, :-1], (1, 0), value=False)
    ones = torch.ones_like(atom_to_token)
    cumsum = torch.cumsum(ones, dim=-1)
    group_start = cumsum.masked_fill(same_as_prev, 0)
    group_start = torch.cummax(group_start, dim=-1).values
    return cumsum - group_start


def _categorical_mean(logits: Tensor, start: float, end: float) -> Tensor:
    r"""
    Expected value of a categorical distribution over evenly-spaced bins.

    Equivalent to ``CategoricalMixture(logits, bins=logits.shape[-1], start, end).mean()``.

    Args:
        logits: [..., n_bins]
        start: left boundary
        end: right boundary

    Returns:
        [...] expected value
    """
    n_bins = logits.shape[-1]
    edges = torch.linspace(start, end, n_bins + 1, device=logits.device, dtype=torch.float32)
    v_bins = (edges[:-1] + edges[1:]) / 2  # [n_bins]
    return (logits.float().softmax(-1) @ v_bins.unsqueeze(1)).squeeze(-1)


def _apply_rotary_emb_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    r"""
    Apply RoPE with batch-dependent cos/sin.

    Args:
        x: [B, L, H, D]
        cos: [B, L, D/2]
        sin: [B, L, D/2]
    """
    ro_dim = cos.shape[-1] * 2
    cos = cos.to(dtype=x.dtype).unsqueeze(2).repeat(1, 1, 1, 2)
    sin = sin.to(dtype=x.dtype).unsqueeze(2).repeat(1, 1, 1, 2)
    rotary = x[..., :ro_dim]
    rotary_left, rotary_right = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-rotary_right, rotary_left), dim=-1)
    return torch.cat(
        [rotary * cos + rotated * sin, x[..., ro_dim:]],
        dim=-1,
    )


def _build_3d_rope(
    reference_positions: Tensor,
    reference_space_id: Tensor,
    head_dim: int,
    num_spatial_pairs_per_axis: int = 4,
    num_reference_space_pairs: int = 2,
    spatial_base_frequency: float = 10000.0,
    reference_space_base_frequency: float = 10.0,
) -> tuple[Tensor, Tensor]:
    r"""
    Build cos/sin for 3D RoPE + reference-space RoPE.
    """
    device = reference_positions.device
    batch_size, num_atoms = reference_positions.shape[:2]
    half_dim = head_dim // 2
    num_spatial_pairs = 3 * num_spatial_pairs_per_axis

    spatial_inverse_frequency = 1.0 / (
        spatial_base_frequency
        ** (
            torch.arange(0, num_spatial_pairs_per_axis, dtype=torch.float32, device=device) / num_spatial_pairs_per_axis
        )
    )
    reference_space_inverse_frequency = 1.0 / (
        reference_space_base_frequency
        ** (torch.arange(0, num_reference_space_pairs, dtype=torch.float32, device=device) / num_reference_space_pairs)
    )

    float_positions = reference_positions.float()
    spatial_frequencies = torch.einsum("bna,k->bnak", float_positions, spatial_inverse_frequency)
    spatial_frequencies = spatial_frequencies.reshape(batch_size, num_atoms, num_spatial_pairs)

    reference_space_values = reference_space_id.float()
    reference_space_frequencies = torch.einsum(
        "bn,k->bnk",
        reference_space_values,
        reference_space_inverse_frequency,
    )

    num_active_pairs = num_spatial_pairs + num_reference_space_pairs
    frequencies = torch.cat([spatial_frequencies, reference_space_frequencies], dim=-1)

    if num_active_pairs < half_dim:
        padding = torch.zeros(
            batch_size,
            num_atoms,
            half_dim - num_active_pairs,
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.cat([frequencies, padding], dim=-1)

    return frequencies.cos(), frequencies.sin()


def _qk_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


def _compute_esmc_hidden_states(
    esmc: nn.Module,
    language_model_tokens: Tensor,
    language_model_mask: Tensor,
    token_to_esmc: Tensor,
    token_has_esmc: Tensor,
    language_model_mask_probability: float = 0.0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    mask_token_id: int = 4,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Run ESMC and gather per-layer states in structure-token order.
    """
    esmc_model = cast(Any, esmc)
    if language_model_mask_probability > 0.0:
        special_tokens = (
            (language_model_tokens == bos_token_id) | (language_model_tokens == eos_token_id) | ~language_model_mask
        )
        positions_to_mask = (
            torch.rand(
                language_model_tokens.shape,
                device=language_model_tokens.device,
                generator=generator,
            )
            < language_model_mask_probability
        ) & ~special_tokens
        language_model_tokens = language_model_tokens.masked_fill(positions_to_mask, mask_token_id)

    language_model_sequence_index = torch.zeros_like(language_model_tokens).masked_fill(
        ~language_model_mask,
        -1,
    )
    output = esmc_model(
        input_ids=language_model_tokens,
        sequence_id=language_model_sequence_index,
        output_hidden_states=True,
        return_dict=True,
    )
    recorded = output.hidden_states
    assert recorded is not None
    layers = (*recorded[:-1], output.last_hidden_state)
    hidden_states = torch.stack(layers, dim=2)
    hidden_states = hidden_states.flatten(0, 1).unsqueeze(0)
    hidden_states = hidden_states.expand(token_to_esmc.shape[0], -1, -1, -1)
    index = token_to_esmc[:, :, None, None].expand(
        -1,
        -1,
        hidden_states.shape[2],
        hidden_states.shape[3],
    )
    gathered = torch.gather(hidden_states, 1, index)
    return gathered * token_has_esmc[:, :, None, None].to(gathered.dtype)


def compute_esmc_masked_lm_logits(
    esmc: nn.Module,
    lm_head: nn.Module,
    input_token_probabilities: Tensor,
    language_model_mask: Tensor,
) -> Tensor:
    r"""
    Compute ESMC masked-language-model logits from soft token probabilities.
    """

    esmc_model = cast(Any, esmc)
    masked_lm_head = cast(Any, lm_head)
    embed_weight = esmc_model.get_input_embeddings().weight
    language_model_embeddings = input_token_probabilities.to(embed_weight.dtype) @ embed_weight
    language_model_sequence_index = torch.zeros_like(language_model_mask, dtype=torch.long).masked_fill(
        ~language_model_mask,
        -1,
    )
    output = esmc_model(
        inputs_embeds=language_model_embeddings,
        sequence_id=language_model_sequence_index,
        return_dict=True,
    )
    logits = masked_lm_head({"last_hidden_state": output.last_hidden_state}).logits
    return logits


def _dropout(input: Tensor, p: float, generator: torch.Generator | None) -> Tensor:
    if generator is None:
        return F.dropout(input, p=p, training=True)
    if p == 0.0:
        return input
    if p == 1.0:
        return torch.zeros_like(input)
    if input.is_cuda:
        with torch.random.fork_rng(devices=[input.device]):
            torch.cuda.set_rng_state(generator.get_state(), input.device)
            output = F.dropout(input, p=p, training=True)
            generator.set_state(torch.cuda.get_rng_state(input.device))
        return output
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        output = F.dropout(input, p=p, training=True)
        generator.set_state(torch.get_rng_state())
    return output


# ===========================================================================
# ESMFold2 tensor boundary and molecular layout.
# ===========================================================================


@dataclass(frozen=True)
class EsmFold2Layout:
    r"""
    Molecular identities aligned to ESMFold2 token and atom tensors.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    msa: MSA | MSACollection | None = None


@dataclass(kw_only=True, eq=False)
class EsmFold2Input:
    r"""
    Tensor and molecular-layout input for ESMFold2.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, ``M`` MSA depth,
    ``C`` protein-chain count, and ``L`` padded language-model length.

    Args:
        token_index: Token indices within each residue with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        molecule_type: Molecular category IDs with shape ``(B, N)``.
        residue_type_ids: Residue-type IDs with shape ``(B, N)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        token_mask: Valid-token mask with shape ``(B, N)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_elements: Atomic numbers with shape ``(B, A)``.
        reference_charges: Formal charges with shape ``(B, A)``.
        reference_atom_name_characters: Four encoded atom-name characters with shape
            ``(B, A, 4)``.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        atom_mask: Valid-atom mask with shape ``(B, A)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        distogram_atom_index: Representative atom index for each token with shape
            ``(B, N)``.
        token_has_esmc: Mask of tokens backed by language-model states with shape
            ``(B, N)``.
        layout: Molecular identities and evidence aligned to the token and atom axes.
        deletion_mean: Optional per-token mean deletion values with shape ``(B, N)``.
        msa_tokens: Optional MSA token IDs with shape ``(B, M, N)``.
        has_deletion: Optional MSA deletion flags with shape ``(B, M, N)``.
        deletion_value: Optional transformed MSA deletion values with shape
            ``(B, M, N)``.
        msa_attention_mask: Optional MSA validity mask with shape ``(B, M, N)``.
        language_model_tokens: Optional language-model token IDs with shape ``(C, L)``.
        language_model_mask: Optional validity mask with shape ``(C, L)``.
        token_to_esmc: Optional flattened language-model index for each structure token,
            with shape ``(B, N)``.
        esmc_hidden_states: Optional language-model states aligned to structure tokens,
            with shape ``(B, N, H, C_lm)``.
        noisy_atom_coordinates: Optional noisy atom coordinates with shape
            ``(B, A, 3)`` in angstroms for one-step denoising.
        noise_level: Optional diffusion noise level with shape ``(B,)``.
    """

    token_index: Tensor
    residue_index: Tensor
    chain_index: Tensor
    symmetry_index: Tensor
    entity_index: Tensor
    molecule_type: Tensor
    residue_type_ids: Tensor
    token_bonds: Tensor
    token_mask: Tensor
    reference_positions: Tensor
    reference_elements: Tensor
    reference_charges: Tensor
    reference_atom_name_characters: Tensor
    reference_space_id: Tensor
    atom_mask: Tensor
    atom_to_token: Tensor
    distogram_atom_index: Tensor
    token_has_esmc: Tensor
    layout: EsmFold2Layout
    deletion_mean: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_attention_mask: Tensor | None = None
    language_model_tokens: Tensor | None = None
    language_model_mask: Tensor | None = None
    token_to_esmc: Tensor | None = None
    esmc_hidden_states: Tensor | None = None
    noisy_atom_coordinates: Tensor | None = None
    noise_level: Tensor | None = None


@dataclass(eq=False)
class EsmFold2Output:
    r"""
    Predictions from one ESMFold2 denoising call.

    Args:
        denoised_coordinates: Denoised atom coordinates with shape
            ``(B, A, 3)`` in angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
    """

    denoised_coordinates: Tensor
    distogram_logits: Tensor


@dataclass(eq=False)
class _EsmFold2ExperimentalTrunkOutput:
    r"""
    Experimental trunk representations consumed by coordinate and confidence heads.
    """

    single_inputs: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor
    token_bonds_encoding: Tensor
    reference_element_features: Tensor
    reference_atom_name_character_features: Tensor
    atom_to_token: Tensor


@dataclass(eq=False)
class EsmFold2ExperimentalSample:
    r"""
    Distogram and coordinate samples from experimental ESMFold2.

    Args:
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        coordinates: Sampled atom coordinates with shape ``(B * S, A, 3)`` in
            angstroms.
    """

    distogram_logits: Tensor
    coordinates: Tensor


@dataclass(eq=False)
class EsmFold2ExperimentalConfidence:
    r"""
    Fixed confidence values for one experimental ESMFold2 sample carrier.

    Args:
        plddt_logits: Per-atom pLDDT logits.
        plddt: Per-token pLDDT values on the native ``0--1`` scale.
        plddt_per_atom: Per-atom pLDDT values on the native ``0--1`` scale.
        plddt_ca: Representative-atom pLDDT values on the native ``0--1`` scale.
        mean_plddt: Mean pLDDT values on the native ``0--1`` scale.
        complex_iplddt: Interface-weighted complex pLDDT values on the native
            ``0--1`` scale.
        pae_logits: Token-pair predicted-aligned-error logits.
        pae: Token-pair predicted aligned errors in angstroms.
        ptm: Predicted TM scores.
        iptm: Interface predicted TM scores.
        chain_pair_iptm: Directed chain-pair interface predicted TM scores.
    """

    plddt_logits: Tensor
    plddt: Tensor
    plddt_per_atom: Tensor
    plddt_ca: Tensor
    mean_plddt: Tensor
    complex_iplddt: Tensor
    pae_logits: Tensor
    pae: Tensor
    ptm: Tensor
    iptm: Tensor
    chain_pair_iptm: Tensor


@dataclass(eq=False)
class EsmFold2ExperimentalPrediction:
    r"""
    One experimental ESMFold2 coordinate sample carrier and its confidence values.

    Args:
        sample: Distogram logits and sampled atom coordinates.
        confidence: Confidence values computed for ``sample``.
    """

    sample: EsmFold2ExperimentalSample
    confidence: EsmFold2ExperimentalConfidence
