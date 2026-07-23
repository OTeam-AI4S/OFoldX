r"""
AlphaFold 2 top-level model.

This source file owns the AF2 public model class, template stack, native
StructureModule geometry, and structure-prediction contract.
"""

from __future__ import annotations

import math
import operator
from copy import copy
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import AtomSite, MolecularSystem
from ...data.structure import Structure
from ...model import Model, ModelBackend
from ...modules import residue_constants
from ...modules.attention import GatedMultiheadAttention
from ...modules.backend import LayerNorm
from ...modules.head import DistogramHead, tm_bin_weights
from ...modules.transition import RowwiseDropout
from ...modules.trunk import EvoformerStack
from ...modules.trunk.evoformer import (
    AlphaFold2OuterProductMean,
    AlphaFold2TriangleAttention,
    AlphaFold2TriangleMultiplication,
    EvoformerTransition,
)
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .config import AlphaFold2Config, AlphaFold2EvoformerConfig, AlphaFold2MultimerConfig


class AlphaFold2Model(Model):
    r"""
    AlphaFold 2 structure-prediction model with AF2-shaped module boundaries.

    The trunk is organized as input embedding, recycling embedding, optional
    template embedding, extra-MSA Evoformer stack, main Evoformer stack, and
    native AF2 structure/head modules.
    """

    config_class = AlphaFold2Config
    config: AlphaFold2Config

    def __init__(self, config: AlphaFold2Config | None = None) -> None:
        config = AlphaFold2Config() if config is None else config
        super().__init__(config)
        input_config = config.input_embeddings
        recycling_config = config.recycling
        template_config = config.template
        evoformer_config = config.evoformer
        block_config = evoformer_config.block
        structure_module_config = config.structure_module
        heads = config.heads

        self._recycling_steps = operator.index(config.recycling_steps)
        if self._recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        self.num_ensembles = operator.index(config.num_ensembles)
        if self.num_ensembles < 1:
            raise ValueError("num_ensembles must be positive")
        self._resample_msa_in_recycling = config.msa.resample_msa_in_recycling
        self._ensemble_msa_from_last = False
        self.input_embeddings = AlphaFold2InputEmbeddings(
            token_feature_size=input_config.token_feature_size,
            msa_feature_size=input_config.msa_feature_size,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            max_relative_feature=input_config.max_relative_feature,
        )
        self.recycling_embeddings = AlphaFold2RecyclingEmbeddings(
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            recycle_positions=recycling_config.recycle_positions,
            recycle_features=recycling_config.recycle_features,
            previous_positions_min_distance=recycling_config.previous_positions.min_distance,
            previous_positions_max_distance=recycling_config.previous_positions.max_distance,
            previous_positions_num_distogram_bins=recycling_config.previous_positions.num_distogram_bins,
        )
        self.template_embedder: AlphaFold2TemplateEmbedder | None = None
        self.template_msa_embedder: AlphaFold2TemplateMsaEmbedder | None = None
        if template_config.enabled:
            self.template_embedder = AlphaFold2TemplateEmbedder(
                query_size=evoformer_config.pair_size,
                template_size=template_config.template_pair_stack.pair_size,
                num_blocks=template_config.template_pair_stack.num_blocks,
                num_distogram_bins=template_config.num_distogram_bins,
                min_distance=template_config.min_distance,
                max_distance=template_config.max_distance,
                vocab_size=template_config.vocab_size,
                use_template_unit_vector=template_config.use_template_unit_vector,
                triangle_multiplication_hidden_size=(
                    template_config.template_pair_stack.triangle_multiplication_outgoing.hidden_size
                ),
                triangle_attention_hidden_size=(
                    template_config.template_pair_stack.triangle_attention_starting_node.value_dim
                ),
                num_triangle_attention_heads=(
                    template_config.template_pair_stack.triangle_attention_starting_node.num_heads
                ),
                transition_expansion=template_config.template_pair_stack.transition.expansion,
                dropout=template_config.template_pair_stack.triangle_attention_starting_node.dropout,
                num_attention_heads=template_config.attention.num_heads,
                operator_backend="auto",
            )
            if template_config.embed_torsion_angles:
                self.template_msa_embedder = AlphaFold2TemplateMsaEmbedder(
                    vocab_size=template_config.vocab_size,
                    msa_size=evoformer_config.msa_size,
                )
        self.extra_msa_embeddings = AlphaFold2ExtraMsaEmbeddings(
            extra_msa_feature_size=input_config.extra_msa_feature_size,
            extra_msa_size=evoformer_config.extra_msa_size,
        )
        self.extra_msa_stack = EvoformerStack(
            num_blocks=evoformer_config.num_extra_msa_blocks,
            msa_size=evoformer_config.extra_msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            outer_product_first=block_config.outer_product_mean.first,
            num_msa_attention_heads=block_config.msa_row_attention_with_pair_bias.num_heads,
            triangle_multiplication_hidden_size=block_config.triangle_multiplication_outgoing.hidden_size,
            triangle_attention_hidden_size=block_config.triangle_attention_starting_node.value_dim,
            num_triangle_attention_heads=block_config.triangle_attention_starting_node.num_heads,
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            is_extra_msa=True,
            operator_backend="auto",
        )
        self.evoformer_stack = EvoformerStack(
            num_blocks=evoformer_config.num_blocks,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            outer_product_first=block_config.outer_product_mean.first,
            num_msa_attention_heads=block_config.msa_row_attention_with_pair_bias.num_heads,
            triangle_multiplication_hidden_size=block_config.triangle_multiplication_outgoing.hidden_size,
            triangle_attention_hidden_size=block_config.triangle_attention_starting_node.value_dim,
            num_triangle_attention_heads=block_config.triangle_attention_starting_node.num_heads,
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            is_extra_msa=False,
            operator_backend="auto",
        )
        self.single_projection = nn.Linear(evoformer_config.msa_size, evoformer_config.single_size)
        self.structure_module = AlphaFold2StructureModule(
            single_size=evoformer_config.single_size,
            pair_size=evoformer_config.pair_size,
            num_blocks=structure_module_config.num_blocks,
            num_transition_projections=structure_module_config.num_transition_projections,
            hidden_size=structure_module_config.hidden_size,
            num_heads=structure_module_config.num_heads,
            num_scalar_qk=structure_module_config.num_scalar_qk,
            num_scalar_v=structure_module_config.num_scalar_v,
            num_point_qk=structure_module_config.num_point_qk,
            num_point_v=structure_module_config.num_point_v,
            dropout=structure_module_config.dropout,
            position_scale=structure_module_config.position_scale,
            sidechain_hidden_size=structure_module_config.sidechain.hidden_size,
            sidechain_num_blocks=structure_module_config.sidechain.num_blocks,
        )
        self.distogram_head = DistogramHead(
            pair_size=evoformer_config.pair_size,
            num_bins=heads.distogram.num_bins,
        )
        nn.init.zeros_(self.distogram_head.projection.weight)
        nn.init.zeros_(self.distogram_head.projection.bias)
        self.predicted_lddt_head = AlphaFold2PredictedLddtHead(
            input_size=structure_module_config.hidden_size,
            hidden_size=heads.predicted_lddt.hidden_size,
            num_bins=heads.predicted_lddt.num_bins,
        )
        self.predicted_aligned_error_head: AlphaFold2PredictedAlignedErrorHead | None = None
        if heads.predicted_aligned_error.enabled:
            self.predicted_aligned_error_head = AlphaFold2PredictedAlignedErrorHead(
                pair_size=evoformer_config.pair_size,
                num_bins=heads.predicted_aligned_error.num_bins,
            )
        self.experimentally_resolved_head = AlphaFold2ExperimentallyResolvedHead(
            input_size=evoformer_config.single_size,
            num_atom_types=heads.experimentally_resolved.num_atom_types,
        )
        self.masked_msa_head = AlphaFold2MaskedMsaHead(
            msa_size=evoformer_config.msa_size,
            num_output=heads.masked_msa.num_output,
        )
        self.post_init()

    def _set_backend(self, backend: ModelBackend) -> None:
        r"""Apply the public PyTorch or OOps attention path."""

        for module in self.modules():
            if isinstance(
                module,
                (GatedMultiheadAttention, AlphaFold2OuterProductMean, AlphaFold2TriangleMultiplication),
            ):
                module.operator_backend = backend

    def forward(
        self,
        input: AlphaFold2Input | AlphaFold2MultimerInput,
        *,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold2Output:
        r"""Run recycling, structure generation, and every output head.

        Gradients flow through the final recycle. This rule is independent of
        ``train()``/``eval()``; those modes affect only ordinary dropout.

        When ``input`` carries the raw alignment pool (``msa_features is None``) the
        Model realizes ``sample_msa`` / ``make_masked_msa`` / clustering /
        ``crop_extra_msa`` itself for every representation-ensemble member and
        recycle, off ``generator``. It averages ``single``, ``pair``, and
        ``msa_first_row`` before structure prediction; the monomer graph keeps the
        first member's full MSA representation.
        When ``input`` already carries realized ``msa_features`` (the precomputed-feature
        input path) the draw is skipped and the input is consumed verbatim, so
        that path is bit-identical to the pre-stochastic Model.
        """

        recycling_steps = operator.index(self._recycling_steps if recycling_steps is None else recycling_steps)
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")

        msa, single, pair, structure, realized = self._run_recycling(
            input, recycling_steps=recycling_steps, generator=generator
        )
        assert realized.msa_features is not None
        num_msa_rows = realized.msa_features.shape[-3]
        distogram_logits = self.distogram_head(pair)
        plddt_logits = self.predicted_lddt_head(structure.structure_module)
        pae_logits = None if self.predicted_aligned_error_head is None else self.predicted_aligned_error_head(pair)
        resolved_logits = self.experimentally_resolved_head(single)
        masked_msa_logits = self.masked_msa_head(msa.narrow(-3, 0, num_msa_rows))
        return AlphaFold2Output(
            single=single,
            pair=pair,
            frames=structure.traj,
            final_atom_positions=structure.final_atom_positions,
            final_atom_mask=structure.final_atom_mask,
            sidechain_angles=structure.sidechain_angles,
            sidechain_unnormalized_angles=structure.sidechain_unnormalized_angles,
            distogram_logits=distogram_logits,
            masked_msa_logits=masked_msa_logits,
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            resolved_logits=resolved_logits,
        )

    @torch.inference_mode()
    def fold(
        self,
        input: AlphaFold2Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> FoldResult:
        r"""Predict AlphaFold 2 monomer structures and return a ``FoldResult``.

        Args:
            input: Single-system ``AlphaFold2Input``. Raw MSA pools are realized per sample and
                recycle; precomputed ``msa_features`` are consumed unchanged.
            num_samples: Number of stochastic folding predictions to generate.
            generator: Optional root PyTorch generator. Each sample draws an independent sub-seed
                used for MSA realization.
            recycling_steps: Number of recycling iterations. ``None`` uses the configured value.

        Returns:
            A ``FoldResult`` containing one canonical ``FoldPrediction`` per sample.

        Notes:
            Predictions follow sample draw order. ``plddt`` and, when present, both ``pae`` axes
            follow ``input.layout.residue_sites``.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if input.residue_type.shape[0] != 1:
            raise ValueError("AlphaFold 2 fold accepts one canonical system per call")

        device = input.residue_type.device
        predictions = []
        for _ in range(num_samples):
            sample_generator = _sample_generator(generator, device)
            output = self(input, recycling_steps=recycling_steps, generator=sample_generator)
            predictions.append(self._decode_prediction(output, input))
        return FoldResult(predictions=tuple(predictions))

    def _decode_prediction(self, output: AlphaFold2Output, input: AlphaFold2Input) -> FoldPrediction:
        r"""Decode one AF2 forward pass into a canonical ``FoldPrediction``."""

        layout = input.layout
        complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        coordinates: np.ndarray = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask: np.ndarray = np.zeros(len(complete_sites), dtype=np.bool_)
        raw_coordinates = output.final_atom_positions[0].float().cpu().numpy()
        raw_coordinate_mask = output.final_atom_mask[0].bool().cpu().numpy()
        for residue_index, (instance_id, monomer_index) in enumerate(layout.residue_sites):
            for atom_name, atom37_index in residue_constants.atom_order.items():
                site = AtomSite(instance_id, monomer_index, atom_name)
                target_index = complete_index.get(site)
                if target_index is None or not raw_coordinate_mask[residue_index, atom37_index]:
                    continue
                coordinates[target_index] = raw_coordinates[residue_index, atom37_index]
                coordinate_mask[target_index] = True
        structure = Structure(
            system=layout.system,
            sites=tuple(complete_sites),
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
        )
        plddt_logits = output.plddt_logits[0].float()
        plddt_bins = (
            torch.arange(
                plddt_logits.shape[-1],
                device=plddt_logits.device,
                dtype=plddt_logits.dtype,
            )
            + 0.5
        ) / plddt_logits.shape[-1]
        plddt = (100.0 * (plddt_logits.softmax(dim=-1) * plddt_bins).sum(dim=-1)).cpu().numpy()
        pae = None if output.pae_logits is None else predicted_aligned_error(output.pae_logits[0], self.config)
        scores = {"mean_plddt": Score(float(plddt.mean()), preferred="higher")}
        metrics = {"plddt": Metric(values=plddt, axis=layout.residue_sites)}
        pair_metrics: dict[str, PairMetric] = {}
        if output.pae_logits is not None and pae is not None:
            scores["ptm"] = Score(
                alphafold2_predicted_tm_score(output.pae_logits[0], input.sequence_mask[0], self.config),
                preferred="higher",
            )
            pair_metrics["pae"] = PairMetric(
                values=pae,
                row_axis=layout.residue_sites,
                column_axis=layout.residue_sites,
                unit="angstrom",
            )
        return FoldPrediction(
            structure=structure,
            scores=scores,
            metrics=metrics,
            pair_metrics=pair_metrics,
        )

    def _run_recycling(
        self,
        input: _AlphaFold2InputT,
        *,
        recycling_steps: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, AlphaFold2StructureModuleOutput, _AlphaFold2InputT]:
        needs_realize = input.msa_features is None
        resample = needs_realize and self._resample_msa_in_recycling
        fixed_ensemble: tuple[_AlphaFold2InputT, ...] | None = None
        if needs_realize and not resample:
            fixed_ensemble = tuple(self._realize_msa(input, generator=generator) for _ in range(self.num_ensembles))
        previous_positions, previous_msa_first_row, previous_pair = self._initial_recycling(input)
        for cycle in range(recycling_steps):
            if needs_realize:
                if resample:
                    cycle_inputs = tuple(
                        self._realize_msa(input, generator=generator) for _ in range(self.num_ensembles)
                    )
                else:
                    assert fixed_ensemble is not None
                    cycle_inputs = fixed_ensemble
            else:
                cycle_inputs = (input,) * self.num_ensembles
            grad_enabled = torch.is_grad_enabled() and cycle == recycling_steps - 1
            with torch.set_grad_enabled(grad_enabled):
                msa, single, pair, msa_first_row, cycle_input = self._run_ensemble(
                    cycle_inputs,
                    previous_positions=previous_positions,
                    previous_msa_first_row=previous_msa_first_row,
                    previous_pair=previous_pair,
                )
                structure = self.structure_module(single, pair, cycle_input)
            if cycle != recycling_steps - 1:
                previous_positions = structure.final_atom_positions.detach()
                previous_msa_first_row = msa_first_row.detach()
                previous_pair = pair.detach()
        return msa, single, pair, structure, cycle_input

    def _run_ensemble(
        self,
        inputs: tuple[_AlphaFold2InputT, ...],
        *,
        previous_positions: Tensor,
        previous_msa_first_row: Tensor,
        previous_pair: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, _AlphaFold2InputT]:
        # DeepMind monomer keeps the first MSA representation; multimer's scan
        # overwrites MSA-only outputs and therefore keeps the last.
        selected_msa = None
        selected_input = inputs[0]
        single_sum = None
        pair_sum = None
        msa_first_row_sum = None
        for index, ensemble_input in enumerate(inputs):
            msa, single, pair, msa_first_row = self._embed_and_refine(
                ensemble_input,
                previous_positions=previous_positions,
                previous_msa_first_row=previous_msa_first_row,
                previous_pair=previous_pair,
            )
            if index == 0 or self._ensemble_msa_from_last:
                selected_msa = msa
                selected_input = ensemble_input
            single_sum = single if single_sum is None else single_sum + single
            pair_sum = pair if pair_sum is None else pair_sum + pair
            msa_first_row_sum = msa_first_row if msa_first_row_sum is None else msa_first_row_sum + msa_first_row

        assert selected_msa is not None
        assert single_sum is not None
        assert pair_sum is not None
        assert msa_first_row_sum is not None
        if len(inputs) == 1:
            return selected_msa, single_sum, pair_sum, msa_first_row_sum, selected_input
        scale = 1.0 / len(inputs)
        return (
            selected_msa,
            single_sum * scale,
            pair_sum * scale,
            msa_first_row_sum * scale,
            selected_input,
        )

    def _realize_msa(self, input: _AlphaFold2InputT, *, generator: torch.Generator | None) -> _AlphaFold2InputT:
        r"""Realize one MSA-feature ensemble slice from the raw pool.

        Runs MSA sampling, masking, nearest-neighbor clustering, cluster
        summarization, extra-MSA cropping, and feature assembly over the raw
        alignment tensors supplied by the shared realization boundary. Every draw
        consumes the per-sample ``generator`` so the slice is reproducible from the
        fold seed.
        """

        device = input.token_features.device
        dtype = input.token_features.dtype
        (
            msa_features,
            msa_mask,
            extra_msa_features,
            extra_msa_mask,
            bert_mask,
            true_msa,
        ) = self._realize_msa_features(
            input.msa_tokens[0],
            input.msa_deletions[0],
            input.msa_profile[0],
            generator=generator,
            device=device,
            dtype=dtype,
        )
        realized = copy(input)
        realized.msa_features = msa_features.unsqueeze(0)
        realized.msa_mask = msa_mask.unsqueeze(0)
        realized.extra_msa_features = extra_msa_features.unsqueeze(0)
        realized.extra_msa_mask = extra_msa_mask.unsqueeze(0)
        realized.bert_mask = bert_mask.unsqueeze(0)
        realized.true_msa = true_msa.unsqueeze(0)
        return realized

    def _realize_msa_features(
        self,
        msa_tokens: Tensor,
        msa_deletions: Tensor,
        msa_profile: Tensor,
        *,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _MsaRealization:
        cfg = self.config.msa
        return realize_msa_features(
            msa_tokens,
            msa_deletions,
            msa_profile,
            max_msa_clusters=cfg.max_msa_clusters,
            max_extra_msa=cfg.max_extra_msa,
            msa_cluster_features=cfg.msa_cluster_features,
            gap_agreement_weight=cfg.gap_agreement_weight,
            masked_msa_enabled=cfg.masked_msa_enabled,
            uniform_prob=cfg.masked_msa_uniform_prob,
            profile_prob=cfg.masked_msa_profile_prob,
            same_prob=cfg.masked_msa_same_prob,
            replace_fraction=cfg.masked_msa_replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _embed_and_refine(
        self,
        input: _AlphaFold2RecyclingInput,
        *,
        previous_positions: Tensor,
        previous_msa_first_row: Tensor,
        previous_pair: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        msa, pair, msa_mask, pair_mask = self.input_embeddings(input)
        msa, pair = self.recycling_embeddings(
            msa,
            pair,
            input=input,
            previous_positions=previous_positions,
            previous_msa_first_row=previous_msa_first_row,
            previous_pair=previous_pair,
        )
        pair = pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        pair = self._add_template_embedding(pair, input, pair_mask)
        extra_msa, extra_msa_mask = self.extra_msa_embeddings(input)
        extra_msa, pair = self.extra_msa_stack(
            extra_msa,
            pair,
            msa_mask=extra_msa_mask,
            pair_mask=pair_mask,
        )
        if self.template_msa_embedder is not None:
            template_activations, template_mask = self.template_msa_embedder(input)
            msa = torch.cat([msa, template_activations], dim=-3)
            msa_mask = torch.cat([msa_mask, template_mask], dim=-2)
        msa, pair = self.evoformer_stack(
            msa,
            pair,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
        )
        msa_first_row = msa.select(dim=-3, index=0)
        single = self.single_projection(msa_first_row)
        return msa, single, pair, msa_first_row

    def _initial_recycling(self, input: _AlphaFold2RecyclingInput) -> tuple[Tensor, Tensor, Tensor]:
        token_shape = input.sequence_mask.shape
        num_tokens = token_shape[-1]
        dtype = input.token_features.dtype
        device = input.token_features.device
        return (
            torch.zeros(*token_shape, residue_constants.atom_type_num, 3, device=device, dtype=dtype),
            torch.zeros(
                *token_shape,
                self.input_embeddings.msa_projection.out_features,
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                *token_shape[:-1],
                num_tokens,
                num_tokens,
                self.input_embeddings.pair_left_projection.out_features,
                device=device,
                dtype=dtype,
            ),
        )

    def _add_template_embedding(self, pair: Tensor, input: _AlphaFold2RecyclingInput, pair_mask: Tensor) -> Tensor:
        if self.template_embedder is None:
            return pair
        return pair + self.template_embedder(pair, input, pair_mask=pair_mask)


class AlphaFold2MultimerModel(AlphaFold2Model):
    r"""
    AlphaFold-Multimer structure-prediction model sharing the AF2 trunk.

    The submodules are constructed under the exact attribute names the inherited
    ``AlphaFold2Model`` forward/recycling methods reference, so those methods are
    reused verbatim across the shared trunk.
    """

    config_class = AlphaFold2MultimerConfig
    config: AlphaFold2MultimerConfig

    def __init__(self, config: AlphaFold2MultimerConfig | None = None) -> None:
        config = AlphaFold2MultimerConfig() if config is None else config
        Model.__init__(self, config)
        input_config = config.input_embeddings
        recycling_config = config.recycling
        template_config = config.template
        evoformer_config: AlphaFold2EvoformerConfig = config.evoformer
        block_config = evoformer_config.block
        structure_module_config = config.structure_module
        heads = config.heads

        self._recycling_steps = operator.index(config.recycling_steps)
        if self._recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        self.num_ensembles = operator.index(config.num_ensembles)
        if self.num_ensembles < 1:
            raise ValueError("num_ensembles must be positive")
        self._resample_msa_in_recycling = config.msa.resample_msa_in_recycling
        self._ensemble_msa_from_last = True
        self.input_embeddings = AlphaFold2MultimerInputEmbeddings(
            token_feature_size=input_config.token_feature_size,
            msa_feature_size=input_config.msa_feature_size,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            max_relative_index=input_config.max_relative_index,
            max_relative_chain=input_config.max_relative_chain,
            use_chain_relative=input_config.use_chain_relative,
        )
        self.recycling_embeddings = AlphaFold2RecyclingEmbeddings(
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            recycle_positions=recycling_config.recycle_positions,
            recycle_features=recycling_config.recycle_features,
            previous_positions_min_distance=recycling_config.previous_positions.min_distance,
            previous_positions_max_distance=recycling_config.previous_positions.max_distance,
            previous_positions_num_distogram_bins=recycling_config.previous_positions.num_distogram_bins,
        )
        self.template_embedder: AlphaFold2MultimerTemplateEmbedder | None = None
        self.template_msa_embedder = None
        if template_config.enabled:
            self.template_embedder = AlphaFold2MultimerTemplateEmbedder(
                query_size=evoformer_config.pair_size,
                template_size=template_config.template_pair_stack.pair_size,
                num_blocks=template_config.template_pair_stack.num_blocks,
                num_distogram_bins=template_config.num_distogram_bins,
                min_distance=template_config.min_distance,
                max_distance=template_config.max_distance,
                vocab_size=template_config.vocab_size,
                triangle_multiplication_hidden_size=(
                    template_config.template_pair_stack.triangle_multiplication_outgoing.hidden_size
                ),
                triangle_attention_hidden_size=(
                    template_config.template_pair_stack.triangle_attention_starting_node.value_dim
                ),
                num_triangle_attention_heads=(
                    template_config.template_pair_stack.triangle_attention_starting_node.num_heads
                ),
                transition_expansion=template_config.template_pair_stack.transition.expansion,
                dropout=template_config.template_pair_stack.triangle_attention_starting_node.dropout,
                operator_backend="auto",
            )
            self.template_msa_embedder = AlphaFold2MultimerTemplateMsaEmbedder(
                msa_size=evoformer_config.msa_size,
                vocab_size=template_config.vocab_size,
            )
        self.extra_msa_embeddings = AlphaFold2ExtraMsaEmbeddings(
            extra_msa_feature_size=input_config.extra_msa_feature_size,
            extra_msa_size=evoformer_config.extra_msa_size,
        )
        self.extra_msa_stack = EvoformerStack(
            num_blocks=evoformer_config.num_extra_msa_blocks,
            msa_size=evoformer_config.extra_msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            outer_product_first=block_config.outer_product_mean.first,
            num_msa_attention_heads=block_config.msa_row_attention_with_pair_bias.num_heads,
            triangle_multiplication_hidden_size=block_config.triangle_multiplication_outgoing.hidden_size,
            triangle_attention_hidden_size=block_config.triangle_attention_starting_node.value_dim,
            num_triangle_attention_heads=block_config.triangle_attention_starting_node.num_heads,
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            is_extra_msa=True,
            operator_backend="auto",
        )
        self.evoformer_stack = EvoformerStack(
            num_blocks=evoformer_config.num_blocks,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            outer_product_first=block_config.outer_product_mean.first,
            num_msa_attention_heads=block_config.msa_row_attention_with_pair_bias.num_heads,
            triangle_multiplication_hidden_size=block_config.triangle_multiplication_outgoing.hidden_size,
            triangle_attention_hidden_size=block_config.triangle_attention_starting_node.value_dim,
            num_triangle_attention_heads=block_config.triangle_attention_starting_node.num_heads,
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            is_extra_msa=False,
            operator_backend="auto",
        )
        self.single_projection = nn.Linear(evoformer_config.msa_size, evoformer_config.single_size)
        self.structure_module = AlphaFold2MultimerStructureModule(
            single_size=evoformer_config.single_size,
            pair_size=evoformer_config.pair_size,
            num_blocks=structure_module_config.num_blocks,
            num_transition_projections=structure_module_config.num_transition_projections,
            hidden_size=structure_module_config.hidden_size,
            num_heads=structure_module_config.num_heads,
            num_scalar_qk=structure_module_config.num_scalar_qk,
            num_scalar_v=structure_module_config.num_scalar_v,
            num_point_qk=structure_module_config.num_point_qk,
            num_point_v=structure_module_config.num_point_v,
            dropout=structure_module_config.dropout,
            position_scale=structure_module_config.position_scale,
            sidechain_hidden_size=structure_module_config.sidechain.hidden_size,
            sidechain_num_blocks=structure_module_config.sidechain.num_blocks,
        )
        self.distogram_head = DistogramHead(
            pair_size=evoformer_config.pair_size,
            num_bins=heads.distogram.num_bins,
        )
        nn.init.zeros_(self.distogram_head.projection.weight)
        nn.init.zeros_(self.distogram_head.projection.bias)
        self.predicted_lddt_head = AlphaFold2PredictedLddtHead(
            input_size=structure_module_config.hidden_size,
            hidden_size=heads.predicted_lddt.hidden_size,
            num_bins=heads.predicted_lddt.num_bins,
        )
        # The multimer TM head is always present.
        self.predicted_aligned_error_head = AlphaFold2PredictedAlignedErrorHead(
            pair_size=evoformer_config.pair_size,
            num_bins=heads.predicted_aligned_error.num_bins,
        )
        self.experimentally_resolved_head = AlphaFold2ExperimentallyResolvedHead(
            input_size=evoformer_config.single_size,
            num_atom_types=heads.experimentally_resolved.num_atom_types,
        )
        self.masked_msa_head = AlphaFold2MaskedMsaHead(
            msa_size=evoformer_config.msa_size,
            num_output=heads.masked_msa.num_output,
        )
        self.post_init()

    def _realize_msa_features(
        self,
        msa_tokens: Tensor,
        msa_deletions: Tensor,
        msa_profile: Tensor,
        *,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _MsaRealization:
        r"""Realize one multimer MSA-feature ensemble slice from the raw pool.

        Runs multimer MSA sampling, masking, nearest-neighbor clustering, and feature
        assembly over the raw paired alignment tensors supplied by the shared
        realization boundary. Every draw consumes the per-sample ``generator`` so
        the slice is reproducible from the fold seed.
        """

        cfg = self.config.msa
        return realize_multimer_msa_features(
            msa_tokens,
            msa_deletions,
            msa_profile,
            max_msa_clusters=cfg.max_msa_clusters,
            max_extra_msa=cfg.max_extra_msa,
            msa_cluster_features=cfg.msa_cluster_features,
            gap_agreement_weight=cfg.gap_agreement_weight,
            masked_msa_enabled=cfg.masked_msa_enabled,
            uniform_prob=cfg.masked_msa_uniform_prob,
            profile_prob=cfg.masked_msa_profile_prob,
            same_prob=cfg.masked_msa_same_prob,
            replace_fraction=cfg.masked_msa_replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def fold(
        self,
        input: AlphaFold2MultimerInput,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> FoldResult:
        r"""Predict AlphaFold-Multimer structures and return a ``FoldResult``.

        Args:
            input: Single-system ``AlphaFold2MultimerInput``. Raw MSA pools are realized per sample
                and recycle; precomputed ``msa_features`` are consumed unchanged.
            num_samples: Number of stochastic folding predictions to generate.
            generator: Optional root PyTorch generator. Each sample draws an independent sub-seed
                used for MSA realization.
            recycling_steps: Number of recycling iterations. ``None`` uses the configured value.

        Returns:
            A ``FoldResult`` containing one canonical ``FoldPrediction`` per sample.

        Notes:
            Predictions follow sample draw order. ``plddt`` and both ``pae`` axes follow
            ``input.layout.residue_sites``.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if input.residue_type.shape[0] != 1:
            raise ValueError("AlphaFold-Multimer fold accepts one canonical system per call")

        device = input.residue_type.device
        predictions = []
        for _ in range(num_samples):
            sample_generator = _sample_generator(generator, device)
            output = self(input, recycling_steps=recycling_steps, generator=sample_generator)
            predictions.append(self._decode_multimer_prediction(output, input))
        return FoldResult(predictions=tuple(predictions))

    def _decode_multimer_prediction(self, output: AlphaFold2Output, input: AlphaFold2MultimerInput) -> FoldPrediction:
        r"""Decode one AF-Multimer forward pass into a canonical ``FoldPrediction``."""

        layout = input.layout
        # Chain permutation is canonical-ordering only at inference: there is no
        # ground-truth assembly to permute predicted symmetric copies against, so
        # residue_sites labels come straight from the Processor's concatenation.
        complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        coordinates: np.ndarray = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask: np.ndarray = np.zeros(len(complete_sites), dtype=np.bool_)
        raw_coordinates = output.final_atom_positions[0].float().cpu().numpy()
        raw_coordinate_mask = output.final_atom_mask[0].bool().cpu().numpy()
        for residue_index, (instance_id, monomer_index) in enumerate(layout.residue_sites):
            for atom_name, atom37_index in residue_constants.atom_order.items():
                site = AtomSite(instance_id, monomer_index, atom_name)
                target_index = complete_index.get(site)
                if target_index is None or not raw_coordinate_mask[residue_index, atom37_index]:
                    continue
                coordinates[target_index] = raw_coordinates[residue_index, atom37_index]
                coordinate_mask[target_index] = True
        structure = Structure(
            system=layout.system,
            sites=tuple(complete_sites),
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
        )
        plddt_logits = output.plddt_logits[0].float()
        plddt_bins = (
            torch.arange(plddt_logits.shape[-1], device=plddt_logits.device, dtype=plddt_logits.dtype) + 0.5
        ) / plddt_logits.shape[-1]
        plddt = (100.0 * (plddt_logits.softmax(dim=-1) * plddt_bins).sum(dim=-1)).cpu().numpy()

        pae_logits = output.pae_logits
        assert pae_logits is not None  # the multimer TM head is always constructed
        pae = predicted_aligned_error(pae_logits[0], self.config)
        ptm = alphafold2_predicted_tm_score(pae_logits[0], input.sequence_mask[0], self.config)
        iptm = predicted_interface_tm_score(pae_logits[0], input.chain_index[0], input.sequence_mask[0], self.config)
        weights = self.config.heads.predicted_aligned_error
        ranking_score = weights.iptm_weight * iptm + weights.ptm_weight * ptm

        scores = {
            "mean_plddt": Score(float(plddt.mean()), preferred="higher"),
            "ptm": Score(ptm, preferred="higher"),
            "iptm": Score(iptm, preferred="higher"),
            "ranking_score": Score(ranking_score, preferred="higher"),
        }
        metrics = {"plddt": Metric(values=plddt, axis=layout.residue_sites)}
        pair_metrics = {
            "pae": PairMetric(
                values=pae,
                row_axis=layout.residue_sites,
                column_axis=layout.residue_sites,
                unit="angstrom",
            )
        }
        return FoldPrediction(
            structure=structure,
            scores=scores,
            metrics=metrics,
            pair_metrics=pair_metrics,
        )


class AlphaFold2StructureModule(nn.Module):
    r"""
    Native AlphaFold 2 StructureModule inference path.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_blocks: int,
        num_transition_projections: int,
        hidden_size: int,
        num_heads: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        sidechain_num_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.position_scale = position_scale
        self.single_norm = LayerNorm(single_size)
        self.input_projection = nn.Linear(single_size, hidden_size)
        self.pair_norm = LayerNorm(pair_size)
        self.block = AlphaFold2StructureModuleBlock(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_transition_projections=num_transition_projections,
            num_heads=num_heads,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            sidechain_hidden_size=sidechain_hidden_size,
            sidechain_num_blocks=sidechain_num_blocks,
            position_scale=position_scale,
            dropout=dropout,
        )

    def forward(self, single: Tensor, pair: Tensor, input: AlphaFold2Input) -> AlphaFold2StructureModuleOutput:
        sequence_mask = input.sequence_mask.to(dtype=single.dtype)
        residue_type = input.residue_type.clamp(min=0, max=20)
        act = self.single_norm(single)
        initial_act = act
        act = self.input_projection(act)
        pair = self.pair_norm(pair)
        affine = QuatAffine.identity(sequence_mask.shape, device=single.device, dtype=single.dtype)

        affines = []
        for _ in range(self.num_blocks):
            act, affine, angles, unnormalized_angles, atom14_positions, raw_affine = self.block(
                act=act,
                pair=pair,
                mask=sequence_mask,
                affine=affine,
                initial_act=initial_act,
                residue_type=residue_type,
            )
            affines.append(raw_affine.to_tensor())

        traj = torch.stack(affines, dim=0)
        traj_scale = traj.new_tensor(
            [1.0, 1.0, 1.0, 1.0, self.position_scale, self.position_scale, self.position_scale],
        )
        traj = traj * traj_scale
        atom14_mask = residue_type_atom14_mask(residue_type, dtype=single.dtype) * sequence_mask.unsqueeze(-1)
        atom37_positions, atom37_mask = atom14_to_atom37(atom14_positions, residue_type, sequence_mask)
        return AlphaFold2StructureModuleOutput(
            final_atom_positions=atom37_positions,
            final_atom_mask=atom37_mask,
            structure_module=act,
            final_atom14_positions=atom14_positions,
            final_atom14_mask=atom14_mask,
            final_affines=traj[-1],
            traj=traj,
            sidechain_angles=angles,
            sidechain_unnormalized_angles=unnormalized_angles,
        )


class AlphaFold2MultimerStructureModule(nn.Module):
    r"""AF-Multimer structure module.

    Same shells as the monomer :class:`AlphaFold2StructureModule` (single/pair layer norms,
    initial projection, and one shared-weight block run ``num_blocks`` times) but starting
    from an identity rotation-matrix rigid and using the multimer block.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_blocks: int,
        num_transition_projections: int,
        hidden_size: int,
        num_heads: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        sidechain_num_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.position_scale = position_scale
        self.single_norm = LayerNorm(single_size)
        self.input_projection = nn.Linear(single_size, hidden_size)
        self.pair_norm = LayerNorm(pair_size)
        self.block = AlphaFold2MultimerStructureModuleBlock(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_transition_projections=num_transition_projections,
            num_heads=num_heads,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            sidechain_hidden_size=sidechain_hidden_size,
            sidechain_num_blocks=sidechain_num_blocks,
            position_scale=position_scale,
            dropout=dropout,
        )

    def forward(self, single: Tensor, pair: Tensor, input: AlphaFold2MultimerInput) -> AlphaFold2StructureModuleOutput:
        sequence_mask = input.sequence_mask.to(dtype=single.dtype)
        residue_type = input.residue_type.clamp(min=0, max=20)
        act = self.single_norm(single)
        initial_act = act
        act = self.input_projection(act)
        pair = self.pair_norm(pair)
        rigid = Rigid.identity(sequence_mask.shape, device=single.device, dtype=single.dtype)

        frames = []
        for _ in range(self.num_blocks):
            act, rigid, angles, unnormalized_angles, atom14_positions = self.block(
                act=act,
                pair=pair,
                mask=sequence_mask,
                rigid=rigid,
                initial_act=initial_act,
                residue_type=residue_type,
            )
            scaled = rigid.scale_translation(self.position_scale)
            frames.append(
                torch.cat([scaled.rotation.reshape(*scaled.rotation.shape[:-2], 9), scaled.translation], dim=-1)
            )

        traj = torch.stack(frames, dim=0)
        atom14_mask = residue_type_atom14_mask(residue_type, dtype=single.dtype) * sequence_mask.unsqueeze(-1)
        atom37_positions, atom37_mask = atom14_to_atom37(atom14_positions, residue_type, sequence_mask)
        return AlphaFold2StructureModuleOutput(
            final_atom_positions=atom37_positions,
            final_atom_mask=atom37_mask,
            structure_module=act,
            final_atom14_positions=atom14_positions,
            final_atom14_mask=atom14_mask,
            final_affines=traj[-1],
            traj=traj,
            sidechain_angles=angles,
            sidechain_unnormalized_angles=unnormalized_angles,
        )


class AlphaFold2TemplateEmbedder(nn.Module):
    r"""Embed AF2 template pair features."""

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        use_template_unit_vector: bool,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        num_attention_heads: int,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.single_template_embedder = AlphaFold2SingleTemplateEmbedder(
            template_size=template_size,
            num_blocks=num_blocks,
            num_distogram_bins=num_distogram_bins,
            min_distance=min_distance,
            max_distance=max_distance,
            vocab_size=vocab_size,
            use_template_unit_vector=use_template_unit_vector,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.attention = GatedMultiheadAttention(
            hidden_size=query_size,
            query_size=query_size,
            key_size=template_size,
            value_size=template_size,
            num_attention_heads=num_attention_heads,
            head_dim=template_size // num_attention_heads,
            query_bias=False,
            output_bias=True,
            gating=False,
            operator_backend=operator_backend,
        )

    def forward(self, query_embedding: Tensor, input: AlphaFold2Input, *, pair_mask: Tensor) -> Tensor:
        template_batch = _template_batch(input)
        template_pair = self.single_template_embedder(template_batch, pair_mask=pair_mask)
        batch_size, num_templates, num_tokens = template_pair.shape[:3]
        template_mask = template_batch.template_mask.to(dtype=query_embedding.dtype)
        has_template = template_mask.sum(dim=-1) > 0
        safe_template_mask = torch.where(
            has_template.unsqueeze(-1),
            template_mask.to(torch.bool),
            torch.ones_like(template_mask, dtype=torch.bool),
        )

        flat_query = query_embedding.reshape(batch_size * num_tokens * num_tokens, 1, query_embedding.shape[-1])
        flat_templates = template_pair.permute(0, 2, 3, 1, 4).reshape(
            batch_size * num_tokens * num_tokens,
            num_templates,
            template_pair.shape[-1],
        )
        flat_mask = (
            safe_template_mask[:, None, None, :]
            .expand(batch_size, num_tokens, num_tokens, num_templates)
            .reshape(batch_size * num_tokens * num_tokens, 1, num_templates)
        )
        update = self.attention(flat_query, flat_templates, flat_templates, attention_mask=flat_mask)
        update = update.reshape(batch_size, num_tokens, num_tokens, query_embedding.shape[-1])
        return update * has_template[:, None, None, None].to(dtype=update.dtype)


class AlphaFold2MultimerTemplateEmbedder(nn.Module):
    r"""AF-Multimer template pair embedder.

    Runs :class:`AlphaFold2MultimerSingleTemplateEmbedder` over every template, averages
    the per-template pair activations, applies ReLU, and projects to the pair channel.
    Backbone unit vectors are built from atom37 coordinates in the model graph; the
    cross-chain ``multichain_mask`` restricts template features to same-chain residue
    pairs.
    """

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.single_template_embedder = AlphaFold2MultimerSingleTemplateEmbedder(
            query_size=query_size,
            template_size=template_size,
            num_blocks=num_blocks,
            num_distogram_bins=num_distogram_bins,
            min_distance=min_distance,
            max_distance=max_distance,
            vocab_size=vocab_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.output_projection = nn.Linear(template_size, query_size)

    def forward(self, query_embedding: Tensor, input: AlphaFold2MultimerInput, *, pair_mask: Tensor) -> Tensor:
        num_templates = input.template_residue_type.shape[1]
        chain_index = input.chain_index
        multichain_mask_2d = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).to(dtype=query_embedding.dtype)
        unit_vectors, backbone_masks = _template_backbone_unit_vector(input)

        accumulated = query_embedding.new_zeros(*query_embedding.shape[:-1], self.output_projection.in_features)
        for index in range(num_templates):
            act = self.single_template_embedder(
                query_embedding,
                residue_type=input.template_residue_type[:, index],
                pseudo_beta=input.template_pseudo_beta[:, index],
                pseudo_beta_mask=input.template_pseudo_beta_mask[:, index],
                unit_vector=unit_vectors[:, index],
                backbone_mask=backbone_masks[:, index],
                multichain_mask_2d=multichain_mask_2d,
                pair_mask=pair_mask,
            )
            accumulated = accumulated + act
        accumulated = accumulated / max(num_templates, 1)
        return self.output_projection(F.relu(accumulated))


class AlphaFold2InputEmbeddings(nn.Module):
    r"""
    Build initial token/MSA/pair activations before recycling or Evoformer blocks.
    """

    def __init__(
        self,
        *,
        token_feature_size: int,
        msa_feature_size: int,
        msa_size: int,
        pair_size: int,
        max_relative_feature: int,
    ) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.token_projection = nn.Linear(token_feature_size, msa_size)
        self.msa_projection = nn.Linear(msa_feature_size, msa_size)
        self.pair_left_projection = nn.Linear(token_feature_size, pair_size)
        self.pair_right_projection = nn.Linear(token_feature_size, pair_size)
        self.relative_position_projection = nn.Linear(2 * max_relative_feature + 1, pair_size)

    def forward(self, input: AlphaFold2Input) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        token_embedding = self.token_projection(input.token_features)
        msa = token_embedding.unsqueeze(-3) + self.msa_projection(input.msa_features)
        pair = self._build_pair_activations(input, input.token_features)
        sequence_mask = input.sequence_mask.to(torch.bool)
        pair_mask = sequence_mask.unsqueeze(-1) & sequence_mask.unsqueeze(-2)
        return msa, pair, input.msa_mask, pair_mask

    def _build_pair_activations(self, input: AlphaFold2Input, token_features: Tensor) -> Tensor:
        pair_left = self.pair_left_projection(token_features)
        pair_right = self.pair_right_projection(token_features)
        pair = pair_left.unsqueeze(-2) + pair_right.unsqueeze(-3)
        relpos = clipped_relative_position(input.residue_index, self.max_relative_feature).to(
            device=pair.device,
            dtype=pair.dtype,
        )
        return pair + self.relative_position_projection(relpos)


# ===========================================================================
# AlphaFold-Multimer top-level model.
#
# The multimer Model reuses the shared Evoformer / extra-MSA / StructureModule
# trunk, its heads, and the AF2 geometry utilities defined above. It differs from
# the monomer Model only in its chain-relative input embedder, its atom37 multimer
# template embedder, its 22-class masked-MSA head, an always-present interface-pTM
# branch, and its result-mapping (ipTM and ranking score).
#
# ``AlphaFold2MultimerModel`` subclasses ``AlphaFold2Model`` so it inherits
# ``forward`` / ``_run_recycling`` / ``_embed_and_refine`` / ``_initial_recycling`` /
# ``_add_template_embedding`` / ``set_backend`` verbatim; it only builds the multimer
# submodules under the same attribute names and overrides ``fold``. No monomer code
# is modified.
# ===========================================================================


class AlphaFold2MultimerInputEmbeddings(nn.Module):
    r"""
    Build initial token/MSA/pair activations with chain-relative positional
    encoding.

    With ``use_chain_relative`` the relative-position feature is the concatenation of a
    within-chain clipped offset one-hot (with one extra different-chain bin), a
    same-entity flag, and a cross-chain asym-difference one-hot (with one extra
    different-entity bin).
    """

    def __init__(
        self,
        *,
        token_feature_size: int,
        msa_feature_size: int,
        msa_size: int,
        pair_size: int,
        max_relative_index: int,
        max_relative_chain: int,
        use_chain_relative: bool,
    ) -> None:
        super().__init__()
        self.max_relative_index = max_relative_index
        self.max_relative_chain = max_relative_chain
        self.use_chain_relative = use_chain_relative
        self.token_projection = nn.Linear(token_feature_size, msa_size)
        self.msa_projection = nn.Linear(msa_feature_size, msa_size)
        self.pair_left_projection = nn.Linear(token_feature_size, pair_size)
        self.pair_right_projection = nn.Linear(token_feature_size, pair_size)
        self.relative_position_projection = nn.Linear(self.relative_feature_size, pair_size)

    @property
    def relative_feature_size(self) -> int:
        if self.use_chain_relative:
            return (2 * self.max_relative_index + 2) + 1 + (2 * self.max_relative_chain + 2)
        return 2 * self.max_relative_index + 1

    def forward(self, input: AlphaFold2MultimerInput) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        token_embedding = self.token_projection(input.token_features)
        msa = token_embedding.unsqueeze(-3) + self.msa_projection(input.msa_features)
        pair = self._build_pair_activations(input)
        sequence_mask = input.sequence_mask.to(torch.bool)
        pair_mask = (sequence_mask.unsqueeze(-1) & sequence_mask.unsqueeze(-2)).to(dtype=pair.dtype)
        return msa, pair, input.msa_mask, pair_mask

    def _build_pair_activations(self, input: AlphaFold2MultimerInput) -> Tensor:
        pair_left = self.pair_left_projection(input.token_features)
        pair_right = self.pair_right_projection(input.token_features)
        pair = pair_left.unsqueeze(-2) + pair_right.unsqueeze(-3)
        relative = self._relative_encoding(input).to(device=pair.device, dtype=pair.dtype)
        return pair + self.relative_position_projection(relative)

    def _relative_encoding(self, input: AlphaFold2MultimerInput) -> Tensor:
        max_relative_index = self.max_relative_index
        residue_index = input.residue_index
        chain_index = input.chain_index
        offset = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)
        if not self.use_chain_relative:
            clipped = torch.clamp(offset + max_relative_index, min=0, max=2 * max_relative_index)
            return F.one_hot(clipped.long(), num_classes=2 * max_relative_index + 1).to(torch.float32)
        clipped = torch.clamp(offset + max_relative_index, min=0, max=2 * max_relative_index)
        final_offset = torch.where(
            same_chain,
            clipped,
            torch.full_like(clipped, 2 * max_relative_index + 1),
        )
        relative_position = F.one_hot(final_offset.long(), num_classes=2 * max_relative_index + 2).to(torch.float32)

        entity_same = (input.entity_index.unsqueeze(-1) == input.entity_index.unsqueeze(-2)).to(torch.float32)

        # The relative-chain feature uses the per-entity copy index, not the global
        # chain index. These coincide only when same-entity copies are contiguously
        # ordered; for an A-B-A assembly they diverge.
        symmetry_index = input.symmetry_index
        max_chain = self.max_relative_chain
        symmetry_offset = symmetry_index.unsqueeze(-1) - symmetry_index.unsqueeze(-2)
        clipped_chain = torch.clamp(symmetry_offset + max_chain, min=0, max=2 * max_chain)
        final_chain = torch.where(
            entity_same.to(torch.bool),
            clipped_chain,
            torch.full_like(clipped_chain, 2 * max_chain + 1),
        )
        relative_chain = F.one_hot(final_chain.long(), num_classes=2 * max_chain + 2).to(torch.float32)
        return torch.cat([relative_position, entity_same.unsqueeze(-1), relative_chain], dim=-1)


class AlphaFold2RecyclingEmbeddings(nn.Module):
    r"""
    Add previous-cycle features to MSA and pair activations.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        recycle_positions: bool,
        recycle_features: bool,
        previous_positions_min_distance: float,
        previous_positions_max_distance: float,
        previous_positions_num_distogram_bins: int,
    ) -> None:
        super().__init__()
        self.recycle_positions = recycle_positions
        self.recycle_features = recycle_features
        self.previous_positions_min_distance = previous_positions_min_distance
        self.previous_positions_max_distance = previous_positions_max_distance
        self.previous_positions_num_distogram_bins = previous_positions_num_distogram_bins
        self.previous_positions_projection = nn.Linear(previous_positions_num_distogram_bins, pair_size)
        self.previous_msa_first_row_norm = LayerNorm(msa_size)
        self.previous_pair_norm = LayerNorm(pair_size)

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        input: AlphaFold2Input,
        previous_positions: Tensor,
        previous_msa_first_row: Tensor,
        previous_pair: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.recycle_features:
            msa = self._add_previous_msa_first_row(msa, previous_msa_first_row)
            pair = pair + self.previous_pair_norm(previous_pair)
        if self.recycle_positions:
            pair = pair + self.previous_positions_projection(
                self._previous_positions_distogram(input, previous_positions, pair.dtype)
            )
        return msa, pair

    def _add_previous_msa_first_row(self, msa: Tensor, previous_msa_first_row: Tensor) -> Tensor:
        first_row = msa.select(dim=-3, index=0) + self.previous_msa_first_row_norm(previous_msa_first_row)
        if msa.shape[-3] == 1:
            return first_row.unsqueeze(-3)
        return torch.cat([first_row.unsqueeze(-3), msa.narrow(-3, 1, msa.shape[-3] - 1)], dim=-3)

    def _previous_positions_distogram(
        self,
        input: AlphaFold2Input,
        previous_positions: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        representative_positions = _pseudo_beta_from_atom37(previous_positions, input)
        squared_distances = (
            (representative_positions.unsqueeze(-2).float() - representative_positions.unsqueeze(-3).float())
            .square()
            .sum(dim=-1)
            .to(dtype=dtype)
        )
        lower_breaks = torch.linspace(
            self.previous_positions_min_distance,
            self.previous_positions_max_distance,
            steps=self.previous_positions_num_distogram_bins,
            device=squared_distances.device,
            dtype=squared_distances.dtype,
        )
        lower_breaks = lower_breaks.square()
        upper_breaks = torch.cat([lower_breaks[1:], lower_breaks.new_tensor([1e8])])
        return ((squared_distances.unsqueeze(-1) > lower_breaks) & (squared_distances.unsqueeze(-1) < upper_breaks)).to(
            dtype=dtype
        )


class AlphaFold2ExtraMsaEmbeddings(nn.Module):
    r"""
    Embed extra-MSA features before the extra-MSA Evoformer stack.
    """

    def __init__(
        self,
        *,
        extra_msa_feature_size: int,
        extra_msa_size: int,
    ) -> None:
        super().__init__()
        self.extra_msa_projection = nn.Linear(extra_msa_feature_size, extra_msa_size)

    def forward(self, input: AlphaFold2Input) -> tuple[Tensor, Tensor]:
        return self.extra_msa_projection(input.extra_msa_features), input.extra_msa_mask


class AlphaFold2SingleTemplateEmbedder(nn.Module):
    r"""
    Embed and refine each template with the shared template pair stack.
    """

    def __init__(
        self,
        *,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        use_template_unit_vector: bool,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.num_distogram_bins = num_distogram_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.vocab_size = vocab_size
        self.use_template_unit_vector = use_template_unit_vector
        template_feature_size = num_distogram_bins + 1 + 2 * vocab_size + 3 + 1
        self.template_pair_projection = nn.Linear(template_feature_size, template_size)
        self.template_pair_stack = AlphaFold2TemplatePairStack(
            num_blocks=num_blocks,
            pair_size=template_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.output_norm = LayerNorm(template_size)

    def forward(self, template_batch: AlphaFold2TemplateBatch, *, pair_mask: Tensor) -> Tensor:
        features, template_pair_mask = self._features(template_batch)
        batch_size, num_templates, num_tokens = features.shape[:3]
        pair_mask = pair_mask[:, None].expand(batch_size, num_templates, num_tokens, num_tokens)
        pair_mask = pair_mask * template_pair_mask

        act = self.template_pair_projection(features).reshape(batch_size * num_templates, num_tokens, num_tokens, -1)
        flat_pair_mask = pair_mask.reshape(batch_size * num_templates, num_tokens, num_tokens)
        act = self.template_pair_stack(act, pair_mask=flat_pair_mask)
        act = self.output_norm(act)
        return act.reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        residue_type = template_batch.template_residue_type.long().clamp(min=0, max=self.vocab_size - 1)
        residue_type_one_hot = F.one_hot(residue_type, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        pseudo_beta, pseudo_beta_mask = template_batch.pseudo_beta()
        template_pair_mask = pseudo_beta_mask.unsqueeze(-1) * pseudo_beta_mask.unsqueeze(-2)
        distogram = _distogram(
            pseudo_beta,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            num_distogram_bins=self.num_distogram_bins,
        ).to(dtype=template_batch.dtype)

        residue_j = residue_type_one_hot.unsqueeze(-3).expand(*template_pair_mask.shape, self.vocab_size)
        residue_i = residue_type_one_hot.unsqueeze(-2).expand(*template_pair_mask.shape, self.vocab_size)
        unit_vector, backbone_pair_mask = template_batch.backbone_unit_vector()
        if not self.use_template_unit_vector:
            unit_vector = torch.zeros_like(unit_vector)
        features = torch.cat(
            [
                distogram,
                template_pair_mask.unsqueeze(-1),
                residue_j,
                residue_i,
                unit_vector,
                backbone_pair_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features * backbone_pair_mask.unsqueeze(-1)
        return features, backbone_pair_mask


class AlphaFold2MultimerSingleTemplateEmbedder(nn.Module):
    r"""Embed one AF-Multimer template into pair space.

    Nine per-feature linear projections sum a masked pseudo-beta distogram, the pair
    pseudo-beta mask, row/column template residue-type one-hots, backbone-frame unit
    vector components, the backbone-pair mask, and the layer-normalized query pair
    embedding. A two-block template pair stack and final layer norm refine the result.
    """

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.num_distogram_bins = num_distogram_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.vocab_size = vocab_size
        self.distogram_projection = nn.Linear(num_distogram_bins, template_size)
        self.pseudo_beta_mask_projection = nn.Linear(1, template_size)
        self.column_residue_type_projection = nn.Linear(vocab_size, template_size)
        self.row_residue_type_projection = nn.Linear(vocab_size, template_size)
        self.unit_vector_x_projection = nn.Linear(1, template_size)
        self.unit_vector_y_projection = nn.Linear(1, template_size)
        self.unit_vector_z_projection = nn.Linear(1, template_size)
        self.backbone_pair_mask_projection = nn.Linear(1, template_size)
        self.query_pair_projection = nn.Linear(query_size, template_size)
        self.query_embedding_norm = LayerNorm(query_size)
        self.template_pair_stack = AlphaFold2TemplatePairStack(
            num_blocks=num_blocks,
            pair_size=template_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
            # AF-Multimer applies triangle multiplication before attention in the
            # template pair stack; monomer keeps attention first.
            triangle_multiplication_first=True,
        )
        self.output_norm = LayerNorm(template_size)

    def forward(
        self,
        query_embedding: Tensor,
        *,
        residue_type: Tensor,
        pseudo_beta: Tensor,
        pseudo_beta_mask: Tensor,
        unit_vector: Tensor,
        backbone_mask: Tensor,
        multichain_mask_2d: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        dtype = query_embedding.dtype
        pseudo_beta_mask_2d = pseudo_beta_mask.unsqueeze(-1) * pseudo_beta_mask.unsqueeze(-2)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask_2d
        distogram = _distogram(
            pseudo_beta,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            num_distogram_bins=self.num_distogram_bins,
        ).to(dtype=dtype)
        distogram = distogram * pseudo_beta_mask_2d.unsqueeze(-1)
        act = self.distogram_projection(distogram)
        act = act + self.pseudo_beta_mask_projection(pseudo_beta_mask_2d.unsqueeze(-1).to(dtype))

        residue_type_one_hot = F.one_hot(residue_type.long().clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(
            dtype
        )
        act = act + self.column_residue_type_projection(residue_type_one_hot[..., None, :, :])
        act = act + self.row_residue_type_projection(residue_type_one_hot[..., None, :])

        backbone_mask_2d = backbone_mask.unsqueeze(-1) * backbone_mask.unsqueeze(-2)
        backbone_mask_2d = backbone_mask_2d * multichain_mask_2d
        x, y, z = (unit_vector * backbone_mask_2d.unsqueeze(-1)).to(dtype).unbind(dim=-1)
        act = act + self.unit_vector_x_projection(x.unsqueeze(-1))
        act = act + self.unit_vector_y_projection(y.unsqueeze(-1))
        act = act + self.unit_vector_z_projection(z.unsqueeze(-1))
        act = act + self.backbone_pair_mask_projection(backbone_mask_2d.unsqueeze(-1).to(dtype))

        query = self.query_embedding_norm(query_embedding)
        act = act + self.query_pair_projection(query)

        act = self.template_pair_stack(act, pair_mask=pair_mask)
        return self.output_norm(act)


class AlphaFold2TemplateMsaEmbedder(nn.Module):
    r"""
    Embed template torsion features as extra MSA rows.
    """

    def __init__(self, *, vocab_size: int, msa_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        template_feature_size = vocab_size + 14 + 14 + 7
        self.input_projection = nn.Linear(template_feature_size, msa_size)
        self.output_projection = nn.Linear(msa_size, msa_size)

    def forward(self, input: AlphaFold2Input) -> tuple[Tensor, Tensor]:
        template_batch = _template_batch(input)
        features, torsion_mask = self._features(template_batch)
        activations = F.relu(self.input_projection(features))
        activations = self.output_projection(activations)
        msa_mask = torsion_mask[..., 2] * template_batch.template_mask.unsqueeze(-1).to(dtype=torsion_mask.dtype)
        return activations, msa_mask

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        residue_type = template_batch.template_residue_type.long().clamp(min=0, max=self.vocab_size - 1)
        residue_type_one_hot = F.one_hot(residue_type, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        torsion = template_batch.template_torsion_angles_sin_cos
        alternative_torsion = template_batch.template_alternative_torsion_angles_sin_cos
        torsion_mask = template_batch.template_torsion_angles_mask
        features = torch.cat(
            [
                residue_type_one_hot,
                torsion.reshape(*residue_type.shape, 14).to(dtype=residue_type_one_hot.dtype),
                alternative_torsion.reshape(*residue_type.shape, 14).to(dtype=residue_type_one_hot.dtype),
                torsion_mask.to(dtype=residue_type_one_hot.dtype),
            ],
            dim=-1,
        )
        return features, torsion_mask.to(dtype=residue_type_one_hot.dtype)


class AlphaFold2MultimerTemplateMsaEmbedder(nn.Module):
    r"""Embed AF-Multimer template chi features as MSA rows.

    Projects per-template chi-angle features (22 residue-type one-hots + sin/cos of the four chi
    angles + chi mask = 34 channels) to the MSA channel and adds them as extra MSA rows.
    """

    def __init__(self, *, msa_size: int, vocab_size: int = 22) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        template_feature_size = vocab_size + 4 + 4 + 4
        self.input_projection = nn.Linear(template_feature_size, msa_size)
        self.output_projection = nn.Linear(msa_size, msa_size)

    def forward(self, input: AlphaFold2MultimerInput) -> tuple[Tensor, Tensor]:
        residue_type = input.template_residue_type.long().clamp(min=0, max=self.vocab_size - 1)
        residue_type_one_hot = F.one_hot(residue_type, self.vocab_size).to(
            dtype=input.template_all_atom_positions.dtype
        )
        chi, chi_mask = _template_chi_angles(
            input.template_all_atom_positions,
            input.template_all_atom_mask,
            residue_type,
        )
        features = torch.cat(
            [residue_type_one_hot, torch.sin(chi) * chi_mask, torch.cos(chi) * chi_mask, chi_mask],
            dim=-1,
        ).to(dtype=self.input_projection.weight.dtype)
        activations = F.relu(self.input_projection(features))
        activations = self.output_projection(activations)
        template_row_mask = chi_mask[..., 0] * input.template_mask.to(dtype=chi_mask.dtype)
        return activations, template_row_mask


class AlphaFold2TemplatePairStack(nn.Module):
    r"""Homogeneous stack of AF2 template-pair blocks."""

    def __init__(
        self,
        *,
        num_blocks: int,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
        triangle_multiplication_first: bool = False,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            [
                AlphaFold2TemplatePairBlock(
                    pair_size=pair_size,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    transition_expansion=transition_expansion,
                    dropout=dropout,
                    operator_backend=operator_backend,
                    triangle_multiplication_first=triangle_multiplication_first,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        for block in self.blocks:
            pair = block(pair, pair_mask=pair_mask)
        return pair


class AlphaFold2TemplatePairBlock(nn.Module):
    r"""
    Triangle update block over the template pair representation: triangle attention (starting and
    ending node), triangle multiplication (outgoing and incoming), and a transition.
    ``triangle_multiplication_first`` selects AF-Multimer ordering instead of monomer ordering.
    """

    def __init__(
        self,
        *,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
        triangle_multiplication_first: bool = False,
    ) -> None:
        super().__init__()
        self.triangle_multiplication_first = triangle_multiplication_first
        self.triangle_attention_starting_node = AlphaFold2TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            operator_backend=operator_backend,
        )
        self.triangle_attention_ending_node = AlphaFold2TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            operator_backend=operator_backend,
        )
        self.triangle_multiplication_outgoing = AlphaFold2TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            operator_backend=operator_backend,
        )
        self.triangle_multiplication_incoming = AlphaFold2TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
            operator_backend=operator_backend,
        )
        self.transition = EvoformerTransition(pair_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        # AF-Multimer's template pair stack applies triangle multiplication before
        # triangle attention; the monomer template stack keeps attention first. The
        # Evoformer operation order is unaffected because this stack is template-only.
        if self.triangle_multiplication_first:
            pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
            pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
            pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
            pair = pair + self.rowwise_dropout(self._ending_node(pair, pair_mask))
            return pair + self.transition(pair, pair_mask)
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self._ending_node(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
        return pair + self.transition(pair, pair_mask)

    def _ending_node(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        # TriangleAttention preserves the input residue-axis orientation. Ending-node attention
        # therefore transposes the pair and mask residue axes around the call, then restores them.
        end_pair_mask = pair_mask.transpose(-2, -1)
        return self.triangle_attention_ending_node(pair.transpose(-3, -2), end_pair_mask).transpose(-3, -2)


class AlphaFold2StructureModuleBlock(nn.Module):
    r"""
    One shared-weight AF2 structure-module iteration.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_transition_projections: int,
        num_heads: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        sidechain_num_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_scale = position_scale
        self.invariant_point_attention = AlphaFold2InvariantPointAttention(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_heads=num_heads,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden_size)
        self.transition = AlphaFold2StructureTransition(
            hidden_size=hidden_size,
            num_projections=num_transition_projections,
        )
        self.transition_norm = LayerNorm(hidden_size)
        self.backbone_update = nn.Linear(hidden_size, 6)
        self.sidechain = AlphaFold2Sidechain(
            hidden_size=hidden_size,
            sidechain_hidden_size=sidechain_hidden_size,
            num_blocks=sidechain_num_blocks,
        )
        nn.init.zeros_(self.backbone_update.weight)
        nn.init.zeros_(self.backbone_update.bias)

    def forward(
        self,
        *,
        act: Tensor,
        pair: Tensor,
        mask: Tensor,
        affine: QuatAffine,
        initial_act: Tensor,
        residue_type: Tensor,
    ) -> tuple[Tensor, QuatAffine, Tensor, Tensor, Tensor, QuatAffine]:
        attn = self.invariant_point_attention(inputs_1d=act, inputs_2d=pair, mask=mask, affine=affine)
        act = self.attention_norm(act + self.dropout(attn))
        act = self.transition_norm(act + self.dropout(self.transition(act)))
        affine = affine.pre_compose(self.backbone_update(act))
        angles, unnormalized_angles, atom14_positions = self.sidechain(
            affine.scale_translation(self.position_scale),
            (act, initial_act),
            residue_type,
        )
        return act, affine.detach_rotation(), angles, unnormalized_angles, atom14_positions, affine


class AlphaFold2MultimerStructureModuleBlock(nn.Module):
    r"""One shared-weight AF-Multimer structure-module iteration.

    Same block topology as the monomer :class:`AlphaFold2StructureModuleBlock` (IPA + attention layer norm,
    transition + layer norm, backbone update, side-chain resnet) but with the multimer IPA
    and the :class:`AlphaFold2MultimerBackboneUpdate` backbone update composed onto a rotation-matrix rigid.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_transition_projections: int,
        num_heads: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        sidechain_num_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_scale = position_scale
        self.invariant_point_attention = AlphaFold2MultimerInvariantPointAttention(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_heads=num_heads,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden_size)
        self.transition = AlphaFold2StructureTransition(
            hidden_size=hidden_size,
            num_projections=num_transition_projections,
        )
        self.transition_norm = LayerNorm(hidden_size)
        self.backbone_update = AlphaFold2MultimerBackboneUpdate(hidden_size)
        self.sidechain = AlphaFold2Sidechain(
            hidden_size=hidden_size,
            sidechain_hidden_size=sidechain_hidden_size,
            num_blocks=sidechain_num_blocks,
        )

    def forward(
        self,
        *,
        act: Tensor,
        pair: Tensor,
        mask: Tensor,
        rigid: Rigid,
        initial_act: Tensor,
        residue_type: Tensor,
    ) -> tuple[Tensor, Rigid, Tensor, Tensor, Tensor]:
        attn = self.invariant_point_attention(single=act, pair=pair, mask=mask, rigid=rigid)
        act = self.attention_norm(act + self.dropout(attn))
        act = self.transition_norm(act + self.dropout(self.transition(act)))
        rigid = rigid.compose(self.backbone_update(act))
        angles, unnormalized_angles, atom14_positions = self.sidechain(
            rigid.scale_translation(self.position_scale),
            (act, initial_act),
            residue_type,
        )
        rigid_next = Rigid(rotation=rigid.rotation.detach(), translation=rigid.translation)
        return act, rigid_next, angles, unnormalized_angles, atom14_positions


class AlphaFold2InvariantPointAttention(nn.Module):
    r"""
    AF2 invariant point attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_heads: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        dist_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.dist_epsilon = dist_epsilon
        self.query_scalar_projection = nn.Linear(hidden_size, num_heads * num_scalar_qk)
        self.key_value_scalar_projection = nn.Linear(hidden_size, num_heads * (num_scalar_v + num_scalar_qk))
        self.query_point_projection = nn.Linear(hidden_size, num_heads * 3 * num_point_qk)
        self.key_value_point_projection = nn.Linear(hidden_size, num_heads * 3 * (num_point_qk + num_point_v))
        self.pair_bias_projection = nn.Linear(pair_size, num_heads)
        output_size = num_heads * num_scalar_v + 4 * num_heads * num_point_v + num_heads * pair_size
        self.output_projection = nn.Linear(output_size, hidden_size)
        self.trainable_point_weights = nn.Parameter(torch.empty(num_heads))
        nn.init.constant_(self.trainable_point_weights, math.log(math.exp(1.0) - 1.0))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, *, inputs_1d: Tensor, inputs_2d: Tensor, mask: Tensor, affine: QuatAffine) -> Tensor:
        num_heads = self.num_heads
        q_scalar = self.query_scalar_projection(inputs_1d).unflatten(-1, (num_heads, self.num_scalar_qk))
        kv_scalar = self.key_value_scalar_projection(inputs_1d).unflatten(
            -1,
            (num_heads, self.num_scalar_v + self.num_scalar_qk),
        )
        k_scalar, v_scalar = kv_scalar.split((self.num_scalar_qk, self.num_scalar_v), dim=-1)

        q_point_local = (
            self.query_point_projection(inputs_1d).unflatten(-1, (3, num_heads, self.num_point_qk)).movedim(-3, -1)
        )
        q_point = affine.apply_to_point(q_point_local, extra_dims=2)
        kv_point_local = (
            self.key_value_point_projection(inputs_1d)
            .unflatten(-1, (3, num_heads, self.num_point_qk + self.num_point_v))
            .movedim(-3, -1)
        )
        kv_point = affine.apply_to_point(kv_point_local, extra_dims=2)
        k_point, v_point = kv_point.split((self.num_point_qk, self.num_point_v), dim=-2)

        scalar_weights = math.sqrt(1.0 / (3.0 * max(self.num_scalar_qk, 1)))
        q_scalar = scalar_weights * q_scalar
        attn_qk_scalar = torch.einsum("...qhd,...khd->...hqk", q_scalar, k_scalar)

        point_weights = math.sqrt(1.0 / (3.0 * max(self.num_point_qk, 1) * 9.0 / 2.0))
        point_weights = point_weights * F.softplus(self.trainable_point_weights)
        q_point = q_point.transpose(-4, -3)
        k_point = k_point.transpose(-4, -3)
        dist2 = (q_point.unsqueeze(-3) - k_point.unsqueeze(-4)).square().sum(dim=-1)
        attn_qk_point = -0.5 * (dist2 * _expand_head_weights(point_weights, dist2)).sum(dim=-1)

        attention_2d_weights = math.sqrt(1.0 / 3.0)
        attention_2d = attention_2d_weights * self.pair_bias_projection(inputs_2d).movedim(-1, -3)
        attn_logits = attn_qk_scalar + attn_qk_point + attention_2d
        pair_mask = mask.to(dtype=torch.bool).unsqueeze(-1) & mask.to(dtype=torch.bool).unsqueeze(-2)
        attn_logits = attn_logits.masked_fill(~pair_mask.unsqueeze(-3), torch.finfo(attn_logits.dtype).min)
        attn = torch.softmax(attn_logits, dim=-1)

        v = v_scalar.transpose(-3, -2)
        result_scalar = torch.einsum("...hqk,...hkd->...hqd", attn, v).transpose(-3, -2).flatten(-2)
        v_point = v_point.transpose(-4, -3)
        result_point = torch.einsum("...hqk,...hkpc->...hqpc", attn, v_point).transpose(-4, -3)
        result_point = affine.invert_point(result_point, extra_dims=2)
        result_point_norm = torch.sqrt(self.dist_epsilon + result_point.square().sum(dim=-1))
        result_attention_over_2d = torch.einsum("...hqk,...qkc->...qhc", attn, inputs_2d).flatten(-2)
        output_features = torch.cat(
            [
                result_scalar,
                result_point[..., 0].flatten(-2),
                result_point[..., 1].flatten(-2),
                result_point[..., 2].flatten(-2),
                result_point_norm.flatten(-2),
                result_attention_over_2d,
            ],
            dim=-1,
        )
        return self.output_projection(output_features)


class AlphaFold2MultimerInvariantPointAttention(nn.Module):
    r"""AF-Multimer invariant point attention (Algorithm 22, multimer variant).

    Differs from monomer invariant point attention in three ways: separate bias-free
    q/k/v scalar projections; per-head point projections in the global frame; and
    multimer logit normalization with per-term weights, ``1/√3`` scaling of the summed
    logits, additive ``inf`` masking, and softmax over the key axis.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_heads: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        inf: float = 1e5,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.inf = inf
        self.query_scalar_projection = nn.Linear(hidden_size, num_heads * num_scalar_qk, bias=False)
        self.key_scalar_projection = nn.Linear(hidden_size, num_heads * num_scalar_qk, bias=False)
        self.value_scalar_projection = nn.Linear(hidden_size, num_heads * num_scalar_v, bias=False)
        self.query_point_projection = AlphaFold2MultimerPointProjection(hidden_size, num_point_qk, num_heads)
        self.key_point_projection = AlphaFold2MultimerPointProjection(hidden_size, num_point_qk, num_heads)
        self.value_point_projection = AlphaFold2MultimerPointProjection(hidden_size, num_point_v, num_heads)
        self.pair_bias_projection = nn.Linear(pair_size, num_heads)
        self.trainable_point_weights = nn.Parameter(torch.zeros(num_heads))
        output_size = num_heads * (pair_size + num_scalar_v + num_point_v * 4)
        self.output_projection = nn.Linear(output_size, hidden_size)

    def forward(self, *, single: Tensor, pair: Tensor, mask: Tensor, rigid: Rigid) -> Tensor:
        num_heads = self.num_heads

        point_variance = max(self.num_point_qk, 1) * 9.0 / 2.0
        point_weights = math.sqrt(1.0 / point_variance)
        head_weights = F.softplus(self.trainable_point_weights)
        point_weights = point_weights * head_weights

        q_point = self.query_point_projection(single, rigid)
        k_point = self.key_point_projection(single, rigid)
        # [*, Ni, Nj, H, P]
        dist2 = (q_point.unsqueeze(-4) - k_point.unsqueeze(-5)).square().sum(dim=-1)
        attn_qk_point = -0.5 * (dist2 * point_weights[..., None, :, None]).sum(dim=-1)

        scalar_weights = math.sqrt(1.0 / max(self.num_scalar_qk, 1))
        q_scalar = self.query_scalar_projection(single).view(*single.shape[:-1], num_heads, self.num_scalar_qk)
        k_scalar = self.key_scalar_projection(single).view(*single.shape[:-1], num_heads, self.num_scalar_qk)
        q_scalar = q_scalar * scalar_weights
        attn_qk_scalar = torch.einsum("...qhc,...khc->...qkh", q_scalar, k_scalar)

        attn_logits = attn_qk_scalar + attn_qk_point + self.pair_bias_projection(pair)

        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1.0)
        attn_logits = attn_logits + square_mask.unsqueeze(-1)
        attn_logits = attn_logits * math.sqrt(1.0 / 3.0)
        # softmax over the key axis (Nj), which is dim -2 in the [*, Ni, Nj, H] layout.
        attn = torch.softmax(attn_logits, dim=-2)

        v_scalar = self.value_scalar_projection(single).view(*single.shape[:-1], num_heads, self.num_scalar_v)
        result_scalar = torch.einsum("...qkh,...khc->...qhc", attn, v_scalar)
        result_scalar = result_scalar.reshape(*result_scalar.shape[:-2], num_heads * self.num_scalar_v)

        v_point = self.value_point_projection(single, rigid)
        # [*, Ni, H, P_v, 3]
        result_point = (attn[..., None, None] * v_point.unsqueeze(-5)).sum(dim=-4)
        result_point = result_point.reshape(*result_point.shape[:-3], num_heads * self.num_point_v, 3)
        result_point = _apply_inverse_rigid_to_points(rigid, result_point)
        result_point_norm = torch.sqrt(1e-8 + result_point.square().sum(dim=-1))

        result_pair = torch.einsum("...qkh,...qkc->...qhc", attn, pair)
        result_pair = result_pair.reshape(*result_pair.shape[:-2], num_heads * pair.shape[-1])

        output_features = torch.cat(
            [
                result_scalar,
                result_point[..., 0],
                result_point[..., 1],
                result_point[..., 2],
                result_point_norm,
                result_pair,
            ],
            dim=-1,
        )
        return self.output_projection(output_features)


class AlphaFold2MultimerPointProjection(nn.Module):
    r"""Project AF-Multimer per-head points into global backbone frames.

    The flat output is read as ``[H, 3, P]`` and reordered to ``[H, P, 3]`` before
    each residue's rigid transform is applied.
    """

    def __init__(self, hidden_size: int, num_points: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.point_projection = nn.Linear(hidden_size, num_heads * 3 * num_points)

    def forward(self, single: Tensor, rigid: Rigid) -> Tensor:
        points_local = self.point_projection(single)
        points_local = points_local.view(*points_local.shape[:-1], self.num_heads, 3, self.num_points)
        points_local = points_local.movedim(-2, -1)
        return _apply_rigid_to_points(rigid, points_local)


class AlphaFold2MultimerBackboneUpdate(nn.Module):
    r"""Predict one AF-Multimer backbone rigid update.

    Predicts a per-residue rigid from six values: the first three are the imaginary part of
    a quaternion (real part fixed to 1, then normalized) giving a rotation; the last
    three are a translation.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 6)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, single: Tensor) -> Rigid:
        rigid_flat = self.projection(single)
        qx, qy, qz, tx, ty, tz = rigid_flat.unbind(dim=-1)
        quaternion = torch.stack([torch.ones_like(qx), qx, qy, qz], dim=-1)
        quaternion = F.normalize(quaternion, dim=-1)
        rotation = quat_to_rot(quaternion)
        translation = torch.stack([tx, ty, tz], dim=-1)
        return Rigid(rotation=rotation, translation=translation)


class AlphaFold2StructureTransition(nn.Module):
    r"""
    Point-wise transition on the structure-module single representation (ReLU between
    linear layers, none after the last; the final layer is zero-initialized).
    """

    def __init__(self, *, hidden_size: int, num_projections: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(num_projections))
        if self.projections:
            nn.init.zeros_(self.projections[-1].weight)
            nn.init.zeros_(self.projections[-1].bias)

    def forward(self, act: Tensor) -> Tensor:
        for index, projection in enumerate(self.projections):
            act = projection(act)
            if index != len(self.projections) - 1:
                act = F.relu(act)
        return act


class AlphaFold2Sidechain(nn.Module):
    r"""
    AF2 side-chain torsion and atom14 coordinate generator.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        sidechain_hidden_size: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.input_projections = nn.ModuleList(nn.Linear(hidden_size, sidechain_hidden_size) for _ in range(2))
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(AlphaFold2SidechainBlock(sidechain_hidden_size) for _ in range(num_blocks))
        self.angle_projection = nn.Linear(sidechain_hidden_size, 14)
        nn.init.zeros_(self.angle_projection.weight)
        nn.init.zeros_(self.angle_projection.bias)

    def forward(
        self,
        affine: QuatAffine,
        representations: tuple[Tensor, Tensor],
        residue_type: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        act = sum(proj(F.relu(rep)) for proj, rep in zip(self.input_projections, representations, strict=True))
        for block in self.blocks:
            act = block(act)
        unnormalized_angles = self.angle_projection(F.relu(act)).unflatten(-1, (7, 2))
        angles = F.normalize(unnormalized_angles, dim=-1)
        frames = torsion_angles_to_frames(residue_type, affine.to_rigid(), angles)
        atom_pos = frames_and_literature_positions_to_atom14_pos(residue_type, frames)
        return angles, unnormalized_angles, atom_pos


class AlphaFold2SidechainBlock(nn.Module):
    r"""
    Two-layer residual block (ReLU pre-activations, zero-initialized final layer) used by
    :class:`AlphaFold2Sidechain` to refine the side-chain activations.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, act: Tensor) -> Tensor:
        residual = act
        act = self.input_projection(F.relu(act))
        act = self.output_projection(F.relu(act))
        return residual + act


@dataclass
class Rigid:
    r"""
    Batched rigid transform represented by rotation matrices and translations.
    """

    rotation: Tensor
    translation: Tensor

    @classmethod
    def from_tensor4x4(cls, tensor: Tensor) -> Rigid:
        return cls(rotation=tensor[..., :3, :3], translation=tensor[..., :3, 3])

    def apply(self, points: Tensor) -> Tensor:
        return torch.matmul(self.rotation, points.unsqueeze(-1)).squeeze(-1) + self.translation

    def compose(self, other: Rigid) -> Rigid:
        rotation = self.rotation @ other.rotation
        translation = self.apply(other.translation)
        return Rigid(rotation=rotation, translation=translation)

    def compose_rotation(self, rotation: Tensor) -> Rigid:
        return Rigid(rotation=self.rotation @ rotation, translation=self.translation)

    def with_group_axis(self) -> Rigid:
        return Rigid(rotation=self.rotation.unsqueeze(-3), translation=self.translation.unsqueeze(-2))

    @classmethod
    def identity(cls, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> Rigid:
        rotation = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3).clone()
        translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
        return cls(rotation=rotation, translation=translation)

    def to_rigid(self) -> Rigid:
        # Adapter so :class:`AlphaFold2Sidechain` (which calls ``affine.to_rigid()``) can
        # consume a bare rotation-matrix rigid from the multimer structure module.
        return self

    def scale_translation(self, position_scale: float) -> Rigid:
        return Rigid(rotation=self.rotation, translation=self.translation * position_scale)

    def detach_rotation(self) -> Rigid:
        return Rigid(rotation=self.rotation.detach(), translation=self.translation)


@dataclass
class QuatAffine:
    r"""
    AF2 backbone affine represented by a quaternion, rotation, and translation.
    """

    quaternion: Tensor
    rotation: Tensor
    translation: Tensor

    @classmethod
    def identity(cls, shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> QuatAffine:
        quaternion = torch.zeros(*shape, 4, device=device, dtype=dtype)
        quaternion[..., 0] = 1.0
        rotation = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3).clone()
        translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
        return cls(quaternion=quaternion, rotation=rotation, translation=translation)

    @classmethod
    def from_tensor(cls, tensor: Tensor, *, normalize: bool = False) -> QuatAffine:
        quaternion = tensor[..., :4]
        if normalize:
            quaternion = F.normalize(quaternion, dim=-1)
        return cls(quaternion=quaternion, rotation=quat_to_rot(quaternion), translation=tensor[..., 4:7])

    def to_tensor(self) -> Tensor:
        return torch.cat([self.quaternion, self.translation], dim=-1)

    def to_rigid(self) -> Rigid:
        return Rigid(rotation=self.rotation, translation=self.translation)

    def scale_translation(self, position_scale: float) -> QuatAffine:
        return QuatAffine(
            quaternion=self.quaternion,
            rotation=self.rotation,
            translation=self.translation * position_scale,
        )

    def detach_rotation(self) -> QuatAffine:
        return QuatAffine(
            quaternion=self.quaternion.detach(),
            rotation=self.rotation.detach(),
            translation=self.translation,
        )

    def pre_compose(self, update: Tensor) -> QuatAffine:
        vector_quaternion_update = update[..., :3]
        translation_update = update[..., 3:6]
        quaternion = self.quaternion + quat_multiply_by_vec(self.quaternion, vector_quaternion_update)
        translation = self.translation + torch.matmul(self.rotation, translation_update.unsqueeze(-1)).squeeze(-1)
        quaternion = F.normalize(quaternion, dim=-1)
        return QuatAffine(quaternion=quaternion, rotation=quat_to_rot(quaternion), translation=translation)

    def apply_to_point(self, point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        return torch.matmul(rotation, point.unsqueeze(-1)).squeeze(-1) + translation

    def invert_point(self, transformed_point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        point = transformed_point - translation
        return torch.matmul(rotation.transpose(-1, -2), point.unsqueeze(-1)).squeeze(-1)


class AlphaFold2PredictedLddtHead(nn.Module):
    r"""
    Predict per-residue pLDDT confidence bin logits from the structure-module single
    representation (layer norm, two ReLU projections, zero-initialized ``num_bins`` logits).
    """

    def __init__(self, *, input_size: int, hidden_size: int, num_bins: int) -> None:
        super().__init__()
        self.input_norm = LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, num_bins)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, structure_module: Tensor) -> Tensor:
        act = self.input_norm(structure_module)
        act = F.relu(self.input_projection(act))
        act = F.relu(self.hidden_projection(act))
        return self.logits(act)


class AlphaFold2PredictedAlignedErrorHead(nn.Module):
    r"""
    Predict predicted-aligned-error bin logits per residue pair from the pair representation
    (a single zero-initialized ``num_bins`` linear projection).
    """

    def __init__(self, *, pair_size: int, num_bins: int) -> None:
        super().__init__()
        self.logits = nn.Linear(pair_size, num_bins)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, pair: Tensor) -> Tensor:
        return self.logits(pair)


class AlphaFold2ExperimentallyResolvedHead(nn.Module):
    r"""
    Predict per-atom experimentally-resolved logits from the single representation
    (one zero-initialized linear projection over ``num_atom_types``).
    """

    def __init__(self, *, input_size: int, num_atom_types: int) -> None:
        super().__init__()
        self.logits = nn.Linear(input_size, num_atom_types)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, single: Tensor) -> Tensor:
        return self.logits(single)


class AlphaFold2MaskedMsaHead(nn.Module):
    r"""
    Predict masked-MSA token logits from the MSA representation for the BERT-style reconstruction
    objective (one zero-initialized linear projection over ``num_output`` token classes).
    """

    def __init__(self, *, msa_size: int, num_output: int) -> None:
        super().__init__()
        self.logits = nn.Linear(msa_size, num_output)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, msa: Tensor) -> Tensor:
        return self.logits(msa)


# ``fold()`` returns an unrelaxed ``FoldPrediction``; callers may apply
# Amber/OpenMM structure relaxation downstream.


class _AlphaFold2RecyclingInput(Protocol):
    r"""Tensor fields shared by monomer and multimer recycling."""

    residue_type: Tensor
    token_features: Tensor
    sequence_mask: Tensor
    msa_tokens: Tensor
    msa_deletions: Tensor
    msa_profile: Tensor
    msa_features: Tensor | None
    msa_mask: Tensor | None
    extra_msa_features: Tensor | None
    extra_msa_mask: Tensor | None
    bert_mask: Tensor | None
    true_msa: Tensor | None


_AlphaFold2InputT = TypeVar("_AlphaFold2InputT", bound=_AlphaFold2RecyclingInput)
_MsaRealization = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def _sample_generator(root: torch.Generator | None, device: torch.device) -> torch.Generator | None:
    r"""Draw one independent per-sample sub-generator off the root stream.

    Returns ``None`` when ``root`` is ``None`` so per-sample draws fall through to
    the ambient default RNG (still stochastic, reproducible only under an explicit
    ``seed``/``generator``). Otherwise draws a fresh 64-bit sub-seed off ``root``
    (not ``seed + s``, which correlates low-entropy streams) and seeds a new
    generator on the compute ``device``.
    """

    if root is None:
        return None
    sub_seed = int(torch.randint(0, 2**63 - 1, (1,), generator=root, device=root.device).item())
    return torch.Generator(device=device).manual_seed(sub_seed)


def realize_msa_features(
    msa_tokens: Tensor,
    msa_deletions: Tensor,
    msa_profile: Tensor,
    *,
    max_msa_clusters: int,
    max_extra_msa: int,
    msa_cluster_features: bool,
    gap_agreement_weight: float,
    masked_msa_enabled: bool,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Realize one AF2 MSA-feature ensemble slice (unbatched).

    The per-recycle sequence is MSA row sampling, BERT masking before clustering,
    nearest-neighbor clustering, cluster summarization, extra-MSA cropping, and feature
    assembly. Inputs are ``msa_tokens [M, N]`` (long residue-type indices, row 0 the
    query, ``20`` unknown, ``21`` gap), ``msa_deletions [M, N]``, and the full-pool
    amino-acid profile ``msa_profile [N, 22]``. Returns ``msa_features [S, N, 49]``,
    ``msa_mask [S, N]``, ``extra_msa_features [E, N, 25]``, ``extra_msa_mask [E, N]``,
    ``bert_mask [S, N]`` and ``true_msa [S, N]``.
    """

    tokens = msa_tokens.to(device=device)
    deletions = msa_deletions.to(device=device, dtype=dtype)
    profile = msa_profile.to(device=device, dtype=dtype)

    sel_tokens, sel_deletions, extra_tokens, extra_deletions = _sample_msa(
        tokens, deletions, max_msa_clusters, generator
    )
    msa_mask = torch.ones(sel_tokens.shape, device=device, dtype=dtype)
    extra_mask = torch.ones(extra_tokens.shape, device=device, dtype=dtype)

    if masked_msa_enabled:
        masked_tokens, bert_mask, true_msa = _make_masked_msa(
            sel_tokens,
            profile,
            uniform_prob=uniform_prob,
            profile_prob=profile_prob,
            same_prob=same_prob,
            replace_fraction=replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        masked_tokens = sel_tokens
        bert_mask = torch.zeros(sel_tokens.shape, device=device, dtype=dtype)
        true_msa = sel_tokens

    if msa_cluster_features:
        assignment = _nearest_neighbor_clusters(
            masked_tokens, msa_mask, extra_tokens, extra_mask, gap_agreement_weight, device, dtype
        )
        cluster_profile, cluster_deletion_mean = _summarize_clusters(
            masked_tokens, msa_mask, sel_deletions, extra_tokens, extra_mask, extra_deletions, assignment, dtype
        )
    else:
        cluster_profile = None
        cluster_deletion_mean = None

    extra_tokens, extra_deletions, extra_mask = _crop_extra_msa(
        extra_tokens, extra_deletions, extra_mask, max_extra_msa, generator, device
    )

    if extra_tokens.shape[0] == 0:
        # Keep a zero-masked row when clustering consumes the entire pool. It contributes nothing to the pair
        # track while preserving a nonempty row axis for the extra-MSA Evoformer stack.
        num_res = tokens.shape[1]
        extra_tokens = tokens[:1]
        extra_deletions = torch.zeros((1, num_res), device=device, dtype=dtype)
        extra_mask = torch.zeros((1, num_res), device=device, dtype=dtype)

    msa_features = _make_msa_features(masked_tokens, sel_deletions, cluster_profile, cluster_deletion_mean, dtype)
    extra_msa_features = _make_extra_msa_features(extra_tokens, extra_deletions, dtype)
    return msa_features, msa_mask, extra_msa_features, extra_mask, bert_mask, true_msa


def _sample_msa(
    tokens: Tensor,
    deletions: Tensor,
    max_seq: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Permute MSA rows (query fixed at row 0), keep ``max_seq``, spill the rest."""

    num_seq = tokens.shape[0]
    device = tokens.device
    if num_seq > 1:
        shuffled = torch.randperm(num_seq - 1, generator=generator, device=device) + 1
        index_order = torch.cat([torch.zeros(1, dtype=torch.long, device=device), shuffled])
    else:
        index_order = torch.zeros(1, dtype=torch.long, device=device)
    num_sel = min(max_seq, num_seq)
    selected = index_order[:num_sel]
    remaining = index_order[num_sel:]
    return tokens[selected], deletions[selected], tokens[remaining], deletions[remaining]


def _make_masked_msa(
    tokens: Tensor,
    profile: Tensor,
    *,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""BERT-corrupt the sampled MSA; emit ``(bert_msa, bert_mask, true_msa)``."""

    random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=dtype, device=device)
    one_hot_msa = F.one_hot(tokens.clamp(max=21), num_classes=22).to(dtype)
    categorical = uniform_prob * random_aa + profile_prob * profile.unsqueeze(0) + same_prob * one_hot_msa
    mask_prob = 1.0 - profile_prob - same_prob - uniform_prob
    categorical = F.pad(categorical, (0, 1), value=mask_prob)
    mask_position = torch.rand(tokens.shape, generator=generator, device=device) < replace_fraction
    sampled = _shaped_categorical(categorical, generator)
    bert_msa = torch.where(mask_position, sampled, tokens)
    return bert_msa, mask_position.to(dtype), tokens


def _shaped_categorical(probs: Tensor, generator: torch.Generator | None, epsilon: float = 1e-10) -> Tensor:
    r"""Sample one class per cell from ``probs`` using ``generator`` (reproducible)."""

    shape = probs.shape
    flat = (probs + epsilon).reshape(-1, shape[-1])
    counts = torch.multinomial(flat, 1, replacement=True, generator=generator).squeeze(-1)
    return counts.reshape(shape[:-1])


def _nearest_neighbor_clusters(
    msa_tokens: Tensor,
    msa_mask: Tensor,
    extra_tokens: Tensor,
    extra_mask: Tensor,
    gap_agreement_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    r"""Assign each extra sequence to its closest cluster by weighted Hamming agreement."""

    weights = torch.cat(
        [
            torch.ones(21, device=device, dtype=dtype),
            gap_agreement_weight * torch.ones(1, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
        ]
    )
    sample_one_hot = msa_mask.unsqueeze(-1) * F.one_hot(msa_tokens.clamp(max=22), num_classes=23).to(dtype)
    extra_one_hot = extra_mask.unsqueeze(-1) * F.one_hot(extra_tokens.clamp(max=22), num_classes=23).to(dtype)
    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq = extra_one_hot.shape[0]
    agreement = torch.matmul(
        extra_one_hot.reshape(extra_num_seq, num_res * 23),
        (sample_one_hot * weights).reshape(num_seq, num_res * 23).transpose(0, 1),
    )
    return torch.argmax(agreement, dim=1).to(torch.int64)


def _unsorted_segment_sum(data: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    r"""Sum ``data`` rows into ``num_segments`` bins indexed by ``segment_ids``."""

    index = segment_ids.reshape(segment_ids.shape[0], *((1,) * (data.dim() - 1))).expand_as(data)
    output = torch.zeros((num_segments, *data.shape[1:]), device=data.device, dtype=data.dtype)
    return output.scatter_add_(0, index, data)


def _summarize_clusters(
    msa_tokens: Tensor,
    msa_mask: Tensor,
    msa_deletions: Tensor,
    extra_tokens: Tensor,
    extra_mask: Tensor,
    extra_deletions: Tensor,
    assignment: Tensor,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    r"""Average per-cluster profile and deletion mean, including the cluster center."""

    num_seq = msa_tokens.shape[0]

    def cluster_sum(values: Tensor) -> Tensor:
        r"""Sum extra-MSA rows into their assigned cluster centers (segment sum over ``assignment``)."""
        return _unsorted_segment_sum(values, assignment, num_seq)

    mask_counts = 1e-6 + msa_mask + cluster_sum(extra_mask)
    msa_sum = cluster_sum(extra_mask.unsqueeze(-1) * F.one_hot(extra_tokens.clamp(max=22), num_classes=23).to(dtype))
    msa_sum = msa_sum + F.one_hot(msa_tokens.clamp(max=22), num_classes=23).to(dtype)
    cluster_profile = msa_sum / mask_counts.unsqueeze(-1)
    deletion_sum = cluster_sum(extra_mask * extra_deletions) + msa_deletions
    cluster_deletion_mean = deletion_sum / mask_counts
    return cluster_profile, cluster_deletion_mean


def _crop_extra_msa(
    extra_tokens: Tensor,
    extra_deletions: Tensor,
    extra_mask: Tensor,
    max_extra_msa: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Randomly subselect at most ``max_extra_msa`` extra rows."""

    num_seq = extra_tokens.shape[0]
    if num_seq == 0 or not max_extra_msa:
        return extra_tokens, extra_deletions, extra_mask
    num_sel = min(max_extra_msa, num_seq)
    selected = torch.randperm(num_seq, generator=generator, device=device)[:num_sel]
    return extra_tokens[selected], extra_deletions[selected], extra_mask[selected]


def _make_msa_features(
    msa_tokens: Tensor,
    msa_deletions: Tensor,
    cluster_profile: Tensor | None,
    cluster_deletion_mean: Tensor | None,
    dtype: torch.dtype,
) -> Tensor:
    r"""Concatenate the 49-channel clustered MSA feature."""

    msa_one_hot = F.one_hot(msa_tokens.clamp(max=22), num_classes=23).to(dtype)
    has_deletion = msa_deletions.clamp(0.0, 1.0).unsqueeze(-1)
    deletion_value = (torch.atan(msa_deletions / 3.0) * (2.0 / math.pi)).unsqueeze(-1)
    parts = [msa_one_hot, has_deletion, deletion_value]
    if cluster_profile is not None and cluster_deletion_mean is not None:
        deletion_mean_value = (torch.atan(cluster_deletion_mean / 3.0) * (2.0 / math.pi)).unsqueeze(-1)
        parts.extend([cluster_profile.to(dtype), deletion_mean_value])
    return torch.cat(parts, dim=-1)


def _make_extra_msa_features(extra_tokens: Tensor, extra_deletions: Tensor, dtype: torch.dtype) -> Tensor:
    r"""Concatenate the 25-channel extra-MSA feature."""

    extra_one_hot = F.one_hot(extra_tokens.clamp(max=22), num_classes=23).to(dtype)
    has_deletion = extra_deletions.clamp(0.0, 1.0).unsqueeze(-1)
    deletion_value = (torch.atan(extra_deletions / 3.0) * (2.0 / math.pi)).unsqueeze(-1)
    return torch.cat([extra_one_hot, has_deletion, deletion_value], dim=-1)


# ===========================================================================
# AlphaFold-Multimer stochastic MSA realization.
#
# The multimer feature pipeline diverges from the monomer in the MSA
# draw itself (OpenFold ``data_transforms_multimer``): ``sample_msa`` selects rows
# by a Gumbel argsort with the query row pinned via ``cluster_bias_mask`` (not a
# plain permutation) and crops the extra axis in the same step;
# ``make_masked_msa`` corrupts over a 22-class alphabet with a Gumbel-max draw;
# ``nearest_neighbor_clusters`` uses a softmax *soft* assignment (not a hard
# argmax) and folds ``summarize_clusters`` into the same pass. These are genuine
# monomer/multimer algorithm differences, so dedicated multimer helpers keep the
# monomer path free of mode-dependent branches.
# ===========================================================================


def _gumbel_noise(
    shape: tuple[int, ...], generator: torch.Generator | None, device: torch.device, eps: float = 1e-6
) -> Tensor:
    r"""Draw Gumbel(0, 1) noise from ``generator``."""

    uniform = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return -torch.log(-torch.log(uniform + eps) + eps)


def realize_multimer_msa_features(
    msa_tokens: Tensor,
    msa_deletions: Tensor,
    msa_profile: Tensor,
    *,
    max_msa_clusters: int,
    max_extra_msa: int,
    msa_cluster_features: bool,
    gap_agreement_weight: float,
    masked_msa_enabled: bool,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Realize one AF-Multimer MSA-feature ensemble slice (unbatched).

    The multimer per-recycle sequence is MSA row sampling, 22-class Gumbel-max BERT
    masking before clustering, soft nearest-neighbor assignment with cluster
    summarization, and feature assembly. Inputs are ``msa_tokens [M, N]``
    (long residue-type indices, row 0 the concatenated query, ``20`` unknown, ``21`` gap),
    ``msa_deletions [M, N]``, and the full-pool ``msa_profile [N, 22]``. Returns
    ``msa_features [S, N, 49]``, ``msa_mask [S, N]``, ``extra_msa_features [E, N, 25]``,
    ``extra_msa_mask [E, N]``, ``bert_mask [S, N]``, and ``true_msa [S, N]``.
    """

    tokens = msa_tokens.to(device=device)
    deletions = msa_deletions.to(device=device, dtype=dtype)
    profile = msa_profile.to(device=device, dtype=dtype)
    num_seq, num_res = tokens.shape
    # The multimer pipeline pads every raw pool to at least 512 rows
    # before sampling. Padded rows carry token/deletion zeroes and a zero mask.
    min_num_seq = 512
    pool_mask = torch.ones((num_seq, num_res), device=device, dtype=dtype)
    if num_seq < min_num_seq:
        pad_rows = min_num_seq - num_seq
        tokens = F.pad(tokens, (0, 0, 0, pad_rows))
        deletions = F.pad(deletions, (0, 0, 0, pad_rows))
        pool_mask = F.pad(pool_mask, (0, 0, 0, pad_rows))

    (
        sel_tokens,
        sel_deletions,
        sel_mask,
        extra_tokens,
        extra_deletions,
        extra_mask,
    ) = _multimer_sample_msa(tokens, deletions, pool_mask, max_msa_clusters, max_extra_msa, generator, device)

    if masked_msa_enabled:
        sel_tokens, bert_mask, true_msa = _multimer_make_masked_msa(
            sel_tokens,
            sel_mask,
            profile,
            uniform_prob=uniform_prob,
            profile_prob=profile_prob,
            same_prob=same_prob,
            replace_fraction=replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        bert_mask = torch.zeros(sel_tokens.shape, device=device, dtype=dtype)
        true_msa = sel_tokens

    if msa_cluster_features:
        cluster_profile, cluster_deletion_mean = _multimer_nearest_neighbor_clusters(
            sel_tokens, sel_mask, sel_deletions, extra_tokens, extra_mask, extra_deletions, gap_agreement_weight, dtype
        )
    else:
        cluster_profile = F.one_hot(sel_tokens.clamp(max=22), num_classes=23).to(dtype)
        cluster_deletion_mean = sel_deletions

    msa_features = _multimer_make_msa_features(sel_tokens, sel_deletions, cluster_profile, cluster_deletion_mean, dtype)
    extra_msa_features = _make_extra_msa_features(extra_tokens, extra_deletions, dtype)

    if extra_tokens.shape[0] == 0:
        # A query-only / small pool leaves the extra axis empty (every row clusters).
        # The predict-mode ``make_fixed_size`` pads it to a nonzero fixed
        # size with a fully zero-masked row; the extra-MSA Evoformer stack cannot
        # reshape a 0-row tensor, so inject one zeroed masked row (token 0, mask 0).
        extra_msa_features = torch.zeros((1, num_res, 25), device=device, dtype=dtype)
        extra_msa_features[..., 0] = 1.0
        extra_mask = torch.zeros((1, num_res), device=device, dtype=dtype)

    return msa_features, sel_mask, extra_msa_features, extra_mask, bert_mask, true_msa


def _multimer_sample_msa(
    tokens: Tensor,
    deletions: Tensor,
    mask: Tensor,
    max_seq: int,
    max_extra_msa: int,
    generator: torch.Generator | None,
    device: torch.device,
    inf: float = 1e6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Gumbel-argsort row selection with the query row pinned; crop extra in one step.

    Rows with no valid position are pushed to ``-inf``, the query row (index 0) is
    pinned to ``+inf`` via ``cluster_bias_mask``, and a Gumbel argsort draws the
    remaining order. The first ``max_seq`` become clusters; the next
    ``max_extra_msa`` become the extra pool.
    """

    num_seq = tokens.shape[0]
    logits = (mask.sum(dim=-1).clamp(0.0, 1.0) - 1.0) * inf
    cluster_bias_mask = torch.zeros(num_seq, device=device, dtype=logits.dtype)
    cluster_bias_mask[0] = 1.0
    logits = logits + cluster_bias_mask * inf
    index_order = torch.argsort(logits + _gumbel_noise(logits.shape, generator, device), dim=-1, descending=True)
    selected_indices = index_order[:max_seq]
    extra_indices = index_order[max_seq:][:max_extra_msa]
    return (
        tokens[selected_indices],
        deletions[selected_indices],
        mask[selected_indices],
        tokens[extra_indices],
        deletions[extra_indices],
        mask[extra_indices],
    )


def _multimer_make_masked_msa(
    tokens: Tensor,
    mask: Tensor,
    profile: Tensor,
    *,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""22-class Gumbel-max BERT corruption; emit ``(bert_msa, bert_mask, true_msa)``.

    The categorical is a 22-class mix (uniform / full-pool profile / identity) padded
    with the mask class. The row mask gates corruption sites, and replacement uses a
    Gumbel-max draw rather than a plain multinomial. Corrupted cells can take mask
    token 22.
    """

    random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=dtype, device=device)
    one_hot_msa = F.one_hot(tokens.clamp(max=21), num_classes=22).to(dtype)
    categorical = uniform_prob * random_aa + profile_prob * profile.unsqueeze(0) + same_prob * one_hot_msa
    mask_prob = 1.0 - profile_prob - same_prob - uniform_prob
    categorical = F.pad(categorical, (0, 1), value=mask_prob)
    mask_position = torch.rand(tokens.shape, generator=generator, device=device) < replace_fraction
    mask_position = mask_position & (mask > 0)
    logits = torch.log(categorical + eps)
    sampled = torch.argmax(logits + _gumbel_noise(logits.shape, generator, device), dim=-1)
    bert_msa = torch.where(mask_position, sampled, tokens)
    bert_msa = bert_msa * mask.to(bert_msa.dtype)
    return bert_msa.long(), mask_position.to(dtype), tokens


def _multimer_nearest_neighbor_clusters(
    msa_tokens: Tensor,
    msa_mask: Tensor,
    msa_deletions: Tensor,
    extra_tokens: Tensor,
    extra_mask: Tensor,
    extra_deletions: Tensor,
    gap_agreement_weight: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    r"""Softmax soft-assign each extra row to clusters; average profile + deletion mean.

    Uses ``softmax(1e3 * agreement)`` over 23-class one-hots; the corrupted MSA may
    carry the mask token, and each cluster centre contributes to its own ``+1`` count.
    """

    device = msa_mask.device
    weights = torch.cat(
        [
            torch.ones(21, device=device, dtype=dtype),
            gap_agreement_weight * torch.ones(1, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
        ]
    )
    msa_one_hot = msa_mask.unsqueeze(-1) * F.one_hot(msa_tokens.clamp(max=22), num_classes=23).to(dtype)
    extra_one_hot = extra_mask.unsqueeze(-1) * F.one_hot(extra_tokens.clamp(max=22), num_classes=23).to(dtype)
    agreement = torch.einsum("mrc,nrc->nm", extra_one_hot, weights * msa_one_hot)
    cluster_assignment = torch.softmax(1e3 * agreement, dim=0)
    cluster_assignment = cluster_assignment * torch.einsum("mr,nr->mn", msa_mask, extra_mask)
    cluster_count = cluster_assignment.sum(dim=-1) + 1.0
    msa_sum = torch.einsum("nm,mrc->nrc", cluster_assignment, extra_one_hot) + msa_one_hot
    cluster_profile = msa_sum / cluster_count[:, None, None]
    del_sum = torch.einsum("nm,mc->nc", cluster_assignment, extra_mask * extra_deletions) + msa_deletions
    cluster_deletion_mean = del_sum / cluster_count[:, None]
    return cluster_profile, cluster_deletion_mean


def _multimer_make_msa_features(
    msa_tokens: Tensor,
    msa_deletions: Tensor,
    cluster_profile: Tensor,
    cluster_deletion_mean: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    r"""Concatenate the multimer 49-channel clustered MSA feature."""

    msa_one_hot = F.one_hot(msa_tokens.clamp(max=22), num_classes=23).to(dtype)
    has_deletion = msa_deletions.clamp(0.0, 1.0).unsqueeze(-1)
    deletion_value = (torch.atan(msa_deletions / 3.0) * (2.0 / math.pi)).unsqueeze(-1)
    deletion_mean_value = (torch.atan(cluster_deletion_mean / 3.0) * (2.0 / math.pi)).unsqueeze(-1)
    return torch.cat(
        [msa_one_hot, has_deletion, deletion_value, cluster_profile.to(dtype), deletion_mean_value], dim=-1
    )


def predicted_aligned_error(logits: Tensor, config: AlphaFold2Config) -> np.ndarray:
    r"""Return expected aligned error in Ångström for one AF2 example."""

    max_error = config.heads.predicted_aligned_error.max_error_bin
    breaks = torch.linspace(0.0, max_error, logits.shape[-1] - 1, device=logits.device)
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, breaks[-1:] + 3 * step / 2])
    return (logits.float().softmax(dim=-1) * centers).sum(dim=-1).cpu().numpy()


def alphafold2_predicted_tm_score(logits: Tensor, sequence_mask: Tensor, config: AlphaFold2Config) -> float:
    r"""Compute the monomer pTM confidence from PAE logits."""

    breaks = torch.linspace(
        0.0,
        config.heads.predicted_aligned_error.max_error_bin,
        logits.shape[-1] - 1,
        device=logits.device,
    )
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, breaks[-1:] + 3 * step / 2])
    residue_count = max(int(sequence_mask.sum().item()), 19)
    predicted = (logits.float().softmax(dim=-1) * tm_bin_weights(centers, residue_count)).sum(dim=-1)
    weights = sequence_mask.to(device=predicted.device, dtype=predicted.dtype)
    pair_mask = weights.unsqueeze(-1) * weights.unsqueeze(-2)
    alignment = (predicted * pair_mask * weights.unsqueeze(0)).sum(dim=-1) / (
        pair_mask.mul(weights.unsqueeze(0)).sum(dim=-1) + 1e-8
    )
    return float((alignment * weights).max().item())


def _residue_type_atom37_to_atom14_tables() -> tuple[np.ndarray, np.ndarray]:
    atom37_to_atom14 = np.zeros((21, residue_constants.atom_type_num), dtype=np.int64)
    atom37_exists = np.zeros((21, residue_constants.atom_type_num), dtype=np.float32)
    for residue_type_index in range(21):
        component_id = "UNK"
        if residue_type_index < residue_constants.NUM_AMINO_ACID_TYPES:
            one_letter_code = residue_constants.AMINO_ACID_ONE_LETTER_CODES[residue_type_index]
            component_id = residue_constants.AMINO_ACID_ONE_TO_THREE[one_letter_code]
        atom14_names = residue_constants.AMINO_ACID_ATOM14_NAMES[component_id]
        for atom37_index, atom_name in enumerate(residue_constants.atom_types):
            if atom_name in atom14_names:
                atom14_index = atom14_names.index(atom_name)
                if atom14_names[atom14_index]:
                    atom37_to_atom14[residue_type_index, atom37_index] = atom14_index
                    atom37_exists[residue_type_index, atom37_index] = 1.0
    return atom37_to_atom14, atom37_exists


_RESIDUE_TYPE_ATOM37_TO_ATOM14, _RESIDUE_TYPE_ATOM37_MASK = _residue_type_atom37_to_atom14_tables()


def quat_to_rot(normalized_quat: Tensor) -> Tensor:
    r"""Convert a normalized ``(w, x, y, z)`` quaternion to its ``(..., 3, 3)`` rotation matrix."""

    w, x, y, z = normalized_quat.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def quat_multiply_by_vec(quat: Tensor, vec: Tensor) -> Tensor:
    r"""Hamilton-multiply a quaternion by the pure quaternion ``(0, vec)``, returning the 4-vector product."""

    r, i, j, k = quat.unbind(dim=-1)
    x, y, z = vec.unbind(dim=-1)
    return torch.stack(
        [
            -i * x - j * y - k * z,
            r * x + j * z - k * y,
            r * y + k * x - i * z,
            r * z + i * y - j * x,
        ],
        dim=-1,
    )


def torsion_angles_to_frames(residue_type: Tensor, backb_to_global: Rigid, torsion_angles_sin_cos: Tensor) -> Rigid:
    r"""Build the 8 global side-chain rigid-group frames from the backbone frame and per-residue torsions.

    Rotates each residue type's default rigid groups by the ``(sin, cos)`` torsion angles and chains the
    dependent chi frames, returning a ``(..., 8)`` :class:`Rigid` in global coordinates.
    """

    default_frames = Rigid.from_tensor4x4(
        _gather_residue_type(
            residue_constants.AMINO_ACID_RIGID_GROUP_DEFAULT_FRAME,
            residue_type,
            dtype=torsion_angles_sin_cos.dtype,
        )
    )
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]
    sin_angles = torch.cat([torch.zeros_like(sin_angles[..., :1]), sin_angles], dim=-1)
    cos_angles = torch.cat([torch.ones_like(cos_angles[..., :1]), cos_angles], dim=-1)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)
    all_rots = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_angles, -sin_angles], dim=-1),
            torch.stack([zeros, sin_angles, cos_angles], dim=-1),
        ],
        dim=-2,
    )
    all_frames = default_frames.compose_rotation(all_rots)
    chi1 = _frame_index(all_frames, 4)
    chi2 = chi1.compose(_frame_index(all_frames, 5))
    chi3 = chi2.compose(_frame_index(all_frames, 6))
    chi4 = chi3.compose(_frame_index(all_frames, 7))
    frames_to_backbone = Rigid(
        rotation=torch.cat(
            [
                all_frames.rotation[..., :5, :, :],
                chi2.rotation.unsqueeze(-3),
                chi3.rotation.unsqueeze(-3),
                chi4.rotation.unsqueeze(-3),
            ],
            dim=-3,
        ),
        translation=torch.cat(
            [
                all_frames.translation[..., :5, :],
                chi2.translation.unsqueeze(-2),
                chi3.translation.unsqueeze(-2),
                chi4.translation.unsqueeze(-2),
            ],
            dim=-2,
        ),
    )
    return backb_to_global.with_group_axis().compose(frames_to_backbone)


def frames_and_literature_positions_to_atom14_pos(residue_type: Tensor, all_frames_to_global: Rigid) -> Tensor:
    r"""Place each residue's 14 literature atom positions into global coordinates via its rigid-group frames.

    Applies the per-atom group frame to the reference (literature) atom14 positions and masks nonexistent
    atoms, returning ``(..., 14, 3)`` predicted coordinates.
    """

    group_index = _gather_residue_type(
        residue_constants.AMINO_ACID_ATOM14_TO_RIGID_GROUP, residue_type, dtype=torch.long
    )
    group_mask = F.one_hot(group_index, num_classes=8).to(dtype=all_frames_to_global.rotation.dtype)
    atom_frames = Rigid(
        rotation=torch.einsum("...nag,...ngij->...naij", group_mask, all_frames_to_global.rotation),
        translation=torch.einsum("...nag,...ngc->...nac", group_mask, all_frames_to_global.translation),
    )
    lit_positions = _gather_residue_type(
        residue_constants.AMINO_ACID_ATOM14_RIGID_GROUP_POSITIONS,
        residue_type,
        dtype=all_frames_to_global.rotation.dtype,
    )
    pred_positions = atom_frames.apply(lit_positions)
    return pred_positions * residue_type_atom14_mask(residue_type, dtype=pred_positions.dtype).unsqueeze(-1)


def atom14_to_atom37(atom14_positions: Tensor, residue_type: Tensor, sequence_mask: Tensor) -> tuple[Tensor, Tensor]:
    r"""Scatter dense ``atom14`` coordinates into the ``atom37`` layout.

    Returns ``(atom37_positions, atom37_mask)`` where the mask combines the per-residue-type atom37
    existence table with ``sequence_mask``.
    """

    atom37_to_atom14 = _gather_residue_type(_RESIDUE_TYPE_ATOM37_TO_ATOM14, residue_type, dtype=torch.long)
    gather_index = atom37_to_atom14.unsqueeze(-1).expand(*atom37_to_atom14.shape, atom14_positions.shape[-1])
    atom37_positions = torch.gather(atom14_positions, dim=-2, index=gather_index)
    atom37_mask = _gather_residue_type(_RESIDUE_TYPE_ATOM37_MASK, residue_type, dtype=atom14_positions.dtype)
    atom37_mask = atom37_mask * sequence_mask.unsqueeze(-1).to(dtype=atom37_mask.dtype)
    return atom37_positions * atom37_mask.unsqueeze(-1), atom37_mask


def residue_type_atom14_mask(residue_type: Tensor, *, dtype: torch.dtype) -> Tensor:
    r"""Gather the per-residue ``atom14`` existence mask for each residue type."""

    return _gather_residue_type(residue_constants.AMINO_ACID_ATOM14_MASK, residue_type, dtype=dtype)


def _frame_index(rigid: Rigid, index: int) -> Rigid:
    return Rigid(rotation=rigid.rotation[..., index, :, :], translation=rigid.translation[..., index, :])


def _gather_residue_type(table: np.ndarray, residue_type: Tensor, *, dtype: torch.dtype) -> Tensor:
    source = torch.as_tensor(table, device=residue_type.device, dtype=dtype)
    index = residue_type.clamp(min=0, max=20).long()
    gathered = source.index_select(0, index.reshape(-1))
    return gathered.reshape(*index.shape, *source.shape[1:])


def _expand_head_weights(weights: Tensor, target: Tensor) -> Tensor:
    view_shape = [1] * target.ndim
    view_shape[-4] = weights.shape[0]
    return weights.view(*view_shape)


_GLY_INDEX = residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("G")


def _template_batch(input: AlphaFold2Input) -> AlphaFold2TemplateBatch:
    return AlphaFold2TemplateBatch(
        template_residue_type=input.template_residue_type,
        template_mask=input.template_mask,
        template_all_atom_positions=input.template_all_atom_positions,
        template_all_atom_mask=input.template_all_atom_mask,
        template_torsion_angles_sin_cos=input.template_torsion_angles_sin_cos,
        template_alternative_torsion_angles_sin_cos=input.template_alternative_torsion_angles_sin_cos,
        template_torsion_angles_mask=input.template_torsion_angles_mask,
    )


def _distogram(positions: Tensor, *, min_distance: float, max_distance: float, num_distogram_bins: int) -> Tensor:
    squared_distances = (positions.unsqueeze(-2) - positions.unsqueeze(-3)).square().sum(dim=-1, keepdim=True)
    lower = torch.linspace(
        min_distance,
        max_distance,
        steps=num_distogram_bins,
        device=positions.device,
        dtype=positions.dtype,
    ).square()
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])])
    return ((squared_distances > lower) & (squared_distances < upper)).to(dtype=positions.dtype)


def clipped_relative_position(residue_index: Tensor, max_relative_feature: int) -> Tensor:
    r"""One-hot encode the relative sequence separation of each residue pair, clipped to
    ``+/- max_relative_feature`` (``2 * max_relative_feature + 1`` bins)."""

    offset = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
    clipped = torch.clamp(offset + max_relative_feature, min=0, max=2 * max_relative_feature)
    return F.one_hot(clipped.long(), num_classes=2 * max_relative_feature + 1)


def _pseudo_beta_from_atom37(atom37_positions: Tensor, input: AlphaFold2Input) -> Tensor:
    ca = residue_constants.atom_order["CA"]
    cb = residue_constants.atom_order["CB"]
    gly = residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("G")
    residue_type = input.residue_type
    atom_index = torch.where(
        residue_type == gly,
        torch.full_like(residue_type, ca),
        torch.full_like(residue_type, cb),
    )
    gather_index = atom_index.unsqueeze(-1).unsqueeze(-1).expand(*atom_index.shape, 1, 3)
    return atom37_positions.gather(-2, gather_index).squeeze(-2)


def _template_backbone_unit_vector(input: AlphaFold2MultimerInput) -> tuple[Tensor, Tensor]:
    r"""Return the pairwise backbone unit vector and per-residue backbone mask per template.

    With the per-residue backbone rigid ``Tᵢ`` (rotation ``Rᵢ``, CA translation
    ``tᵢ``), the pairwise unit vector is
    ``normalize(Rᵢᵀ (tⱼ − tᵢ))`` -- the direction from residue ``i`` to residue ``j`` expressed
    in ``i``'s local frame. Shapes: unit vector ``[B, S_t, N, N, 3]``, mask ``[B, S_t, N]``.
    """

    positions = input.template_all_atom_positions
    output_dtype = positions.dtype
    compute_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
    positions = positions.to(dtype=compute_dtype)

    n = residue_constants.atom_order["N"]
    ca = residue_constants.atom_order["CA"]
    c = residue_constants.atom_order["C"]
    translation = positions[..., ca, :]
    x_axis = _multimer_normalize_vector(positions[..., c, :] - translation)
    n_axis = positions[..., n, :] - translation

    # Preserve the JAX CUDA graph's component order and single-rounding geometry.
    projection = n_axis[..., 1] * x_axis[..., 1]
    projection = torch.addcmul(projection, n_axis[..., 0], x_axis[..., 0])
    projection = torch.addcmul(projection, n_axis[..., 2], x_axis[..., 2])
    y_axis = (n_axis.to(torch.float64) - projection.to(torch.float64).unsqueeze(-1) * x_axis.to(torch.float64)).to(
        dtype=compute_dtype
    )
    y_axis = _multimer_normalize_vector(y_axis)

    x_axis_double = x_axis.to(torch.float64)
    y_axis_double = y_axis.to(torch.float64)
    z_axis = torch.stack(
        [
            x_axis_double[..., 1] * y_axis_double[..., 2] - (x_axis[..., 2] * y_axis[..., 1]).to(torch.float64),
            x_axis_double[..., 2] * y_axis_double[..., 0] - (x_axis[..., 0] * y_axis[..., 2]).to(torch.float64),
            x_axis_double[..., 0] * y_axis_double[..., 1] - (x_axis[..., 1] * y_axis[..., 0]).to(torch.float64),
        ],
        dim=-1,
    ).to(dtype=compute_dtype)
    inverse_rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1).transpose(-1, -2)

    inverse_translation = inverse_rotation[..., 0] * translation[..., 0].unsqueeze(-1)
    inverse_translation = torch.addcmul(
        inverse_translation,
        inverse_rotation[..., 1],
        translation[..., 1].unsqueeze(-1),
    )
    inverse_translation = torch.addcmul(
        inverse_translation,
        inverse_rotation[..., 2],
        translation[..., 2].unsqueeze(-1),
    )
    inverse_translation = -inverse_translation

    transformed_points = inverse_rotation[..., 1].unsqueeze(-2) * translation[..., 1].unsqueeze(-2).unsqueeze(-1)
    transformed_points = torch.addcmul(
        transformed_points,
        inverse_rotation[..., 0].unsqueeze(-2),
        translation[..., 0].unsqueeze(-2).unsqueeze(-1),
    )
    transformed_points = torch.addcmul(
        transformed_points,
        inverse_rotation[..., 2].unsqueeze(-2),
        translation[..., 2].unsqueeze(-2).unsqueeze(-1),
    )
    unit_vector = _multimer_normalize_vector(transformed_points + inverse_translation.unsqueeze(-2))
    atom_mask = input.template_all_atom_mask
    backbone_mask = atom_mask[..., n] * atom_mask[..., ca] * atom_mask[..., c]
    return unit_vector.to(dtype=output_dtype), backbone_mask.to(dtype=output_dtype)


def _multimer_normalize_vector(value: Tensor) -> Tensor:
    squared_norm = value[..., 1].square()
    squared_norm = torch.addcmul(squared_norm, value[..., 0], value[..., 0])
    squared_norm = torch.addcmul(squared_norm, value[..., 2], value[..., 2])
    return value / torch.sqrt(squared_norm.clamp_min(1e-12)).unsqueeze(-1)


def _build_chi_atom37_tables() -> tuple[np.ndarray, np.ndarray]:
    r"""Return atom37 indices ``(21, 4, 4)`` and validity mask ``(21, 4)`` for the chi angles."""

    indices = np.zeros((21, 4, 4), dtype=np.int64)
    mask = np.zeros((21, 4), dtype=np.float32)
    for residue_type_index in range(residue_constants.NUM_AMINO_ACID_TYPES):
        one_letter_code = residue_constants.AMINO_ACID_ONE_LETTER_CODES[residue_type_index]
        component_id = residue_constants.AMINO_ACID_ONE_TO_THREE[one_letter_code]
        atoms_for_chi = residue_constants.chi_angles_atoms[component_id]
        for chi_index, atom_names in enumerate(atoms_for_chi):
            for position, atom_name in enumerate(atom_names):
                indices[residue_type_index, chi_index, position] = residue_constants.atom_order[atom_name]
            mask[residue_type_index, chi_index] = 1.0
    return indices, mask


_CHI_ATOM37_INDICES, _CHI_ANGLE_MASK = _build_chi_atom37_tables()


def _template_chi_angles(positions: Tensor, atom_mask: Tensor, residue_type: Tensor) -> tuple[Tensor, Tensor]:
    r"""Compute the four backbone-independent chi angles per residue from atom37 positions.

    Returns ``(chi, chi_mask)`` with shapes ``[..., N, 4]``. ``chi`` is the signed dihedral of
    the four defining atoms; ``chi_mask`` is the product of the residue-type chi validity and the
    presence of all four atoms.
    """

    indices = torch.as_tensor(_CHI_ATOM37_INDICES, device=positions.device)
    angle_valid = torch.as_tensor(_CHI_ANGLE_MASK, device=positions.device, dtype=positions.dtype)
    # The chi tables cover residue types 0..19 plus a zero-mask UNK row at index 20;
    # clamp so gap / UNK template tokens index the inert UNK row.
    table_index = residue_type.long().clamp(min=0, max=indices.shape[0] - 1)
    per_res_indices = indices[table_index]  # [..., N, 4, 4]  (4 chi angles, 4 atoms each)
    per_res_valid = angle_valid[table_index]  # [..., N, 4]

    # Gather the four defining atom coordinates for every (residue, chi): [..., N, 4, 4, 3].
    gather_index = per_res_indices.unsqueeze(-1).expand(*per_res_indices.shape, 3)
    expanded_positions = positions.unsqueeze(-3).expand(
        *positions.shape[:-2], 4, positions.shape[-2], positions.shape[-1]
    )
    # index the atom (37) axis for each chi group
    atoms = torch.gather(expanded_positions, -2, gather_index)  # [..., N, 4chi, 4atom, 3]

    a0, a1, a2, a3 = atoms.unbind(dim=-2)
    chi = _dihedral(a0, a1, a2, a3)  # [..., N, 4]

    present = torch.gather(
        atom_mask.unsqueeze(-2).expand(*atom_mask.shape[:-1], 4, atom_mask.shape[-1]),
        -1,
        per_res_indices,
    )  # [..., N, 4chi, 4atom]
    atoms_present = present.prod(dim=-1)  # [..., N, 4]
    chi_mask = per_res_valid * atoms_present
    return chi, chi_mask


def _dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    r"""Signed dihedral angle (radians) about the ``p1-p2`` axis for four points ``[..., 3]``."""

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _apply_rigid_to_points(rigid: Rigid, points: Tensor) -> Tensor:
    r"""Map local per-head points ``[*, N, H, P, 3]`` into the global frame of each residue."""

    rotation = rigid.rotation[..., :, None, None, :, :]
    translation = rigid.translation[..., :, None, None, :]
    return torch.einsum("...ij,...j->...i", rotation, points) + translation


def _apply_inverse_rigid_to_points(rigid: Rigid, points: Tensor) -> Tensor:
    r"""Map global points ``[*, N, M, 3]`` back into each residue's local frame (``Rᵀ(p-t)``)."""

    rotation = rigid.rotation[..., :, None, :, :]
    translation = rigid.translation[..., :, None, :]
    return torch.einsum("...ji,...j->...i", rotation, points - translation)


def predicted_interface_tm_score(
    logits: Tensor,
    chain_index: Tensor,
    sequence_mask: Tensor,
    config: AlphaFold2MultimerConfig,
) -> float:
    r"""Compute interface pTM (ipTM) from PAE logits over cross-chain pairs.

    Identical to the pTM expectation except the alignment average is restricted to
    residue pairs with different ``chain_index`` values.
    """

    breaks = torch.linspace(
        0.0,
        config.heads.predicted_aligned_error.max_error_bin,
        logits.shape[-1] - 1,
        device=logits.device,
    )
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, breaks[-1:] + 3 * step / 2])
    residue_count = max(int(sequence_mask.sum().item()), 19)
    d0 = 1.24 * (residue_count - 15) ** (1.0 / 3.0) - 1.8
    tm_per_bin = 1.0 / (1.0 + (centers / d0) ** 2)
    predicted = (logits.float().softmax(dim=-1) * tm_per_bin).sum(dim=-1)

    weights = sequence_mask.to(device=predicted.device, dtype=predicted.dtype)
    chain_index = chain_index.to(device=predicted.device)
    cross_chain = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(dtype=predicted.dtype)
    pair_mask = weights.unsqueeze(-1) * weights.unsqueeze(-2) * cross_chain
    alignment = (predicted * pair_mask * weights.unsqueeze(0)).sum(dim=-1) / (
        pair_mask.mul(weights.unsqueeze(0)).sum(dim=-1) + 1e-8
    )
    # Only residues that participate in at least one cross-chain pair are eligible.
    eligible = (pair_mask.sum(dim=-1) > 0).to(dtype=predicted.dtype) * weights
    if float(eligible.sum().item()) == 0.0:
        return 0.0
    return float((alignment * eligible).max().item())


@dataclass
class AlphaFold2StructureModuleOutput:
    r"""
    Output tensors from the native AF2 structure module.
    """

    final_atom_positions: Tensor
    final_atom_mask: Tensor
    structure_module: Tensor
    final_atom14_positions: Tensor
    final_atom14_mask: Tensor
    final_affines: Tensor
    traj: Tensor
    sidechain_angles: Tensor
    sidechain_unnormalized_angles: Tensor


@dataclass
class AlphaFold2TemplateBatch:
    r"""Template tensors shared by the pair and torsion embedding modules."""

    template_residue_type: Tensor
    template_mask: Tensor
    template_all_atom_positions: Tensor
    template_all_atom_mask: Tensor
    template_torsion_angles_sin_cos: Tensor
    template_alternative_torsion_angles_sin_cos: Tensor
    template_torsion_angles_mask: Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.template_all_atom_positions.dtype

    def pseudo_beta(self) -> tuple[Tensor, Tensor]:
        ca = residue_constants.atom_order["CA"]
        cb = residue_constants.atom_order["CB"]
        is_gly = self.template_residue_type == _GLY_INDEX
        atom_index = torch.where(
            is_gly,
            torch.full_like(self.template_residue_type, ca),
            torch.full_like(self.template_residue_type, cb),
        )
        gather_index = atom_index[..., None, None].expand(*atom_index.shape, 1, 3)
        pseudo_beta = self.template_all_atom_positions.gather(-2, gather_index).squeeze(-2)
        mask_index = atom_index[..., None].expand(*atom_index.shape, 1)
        pseudo_beta_mask = self.template_all_atom_mask.gather(-1, mask_index).squeeze(-1)
        return pseudo_beta.to(dtype=self.dtype), pseudo_beta_mask.to(dtype=self.dtype)

    def backbone_unit_vector(self) -> tuple[Tensor, Tensor]:
        n = residue_constants.atom_order["N"]
        ca = residue_constants.atom_order["CA"]
        c = residue_constants.atom_order["C"]
        atom_pos = self.template_all_atom_positions.to(dtype=self.dtype)
        atom_mask = self.template_all_atom_mask.to(dtype=self.dtype)
        n_pos = atom_pos[..., n, :]
        ca_pos = atom_pos[..., ca, :]
        c_pos = atom_pos[..., c, :]
        backbone_mask = atom_mask[..., n] * atom_mask[..., ca] * atom_mask[..., c]

        x_axis = F.normalize(c_pos - ca_pos, dim=-1)
        n_axis = F.normalize(n_pos - ca_pos, dim=-1)
        z_axis = F.normalize(torch.cross(x_axis, n_axis, dim=-1), dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        vector = ca_pos.unsqueeze(-3) - ca_pos.unsqueeze(-2)
        unit_vector = torch.matmul(rotation.unsqueeze(-3).transpose(-1, -2), vector.unsqueeze(-1)).squeeze(-1)
        unit_vector = F.normalize(unit_vector, dim=-1)
        pair_mask = backbone_mask.unsqueeze(-1) * backbone_mask.unsqueeze(-2)
        unit_vector = torch.nan_to_num(unit_vector) * pair_mask.unsqueeze(-1)
        return unit_vector, pair_mask.to(dtype=self.dtype)


# ===========================================================================
# Tensor boundary and canonical semantic layout (monomer + multimer contracts).
# ===========================================================================


@dataclass(frozen=True)
class AlphaFold2Layout:
    r"""Canonical residue identities aligned to one monomer AF2 token axis."""

    system: MolecularSystem
    residue_sites: tuple[tuple[str, int], ...]
    msa: MSA | None
    templates: tuple[Template, ...]


@dataclass(kw_only=True)
class AlphaFold2Input:
    r"""Complete compute and semantic input for monomer AlphaFold 2.

    Under the stochastic-fold contract the Processor lowers only the
    **raw alignment pool** and the full-MSA amino-acid profile; the Model draws the
    clustered/extra MSA feature realizations itself, once per configured ensemble
    member and recycle, off its per-sample generator (see
    :meth:`AlphaFold2Model.fold`). The realized 49- and
    25-channel tensors are therefore *not* part of the input contract.

    Tensor shapes: ``residue_type`` and ``residue_index`` are ``[B, N]``;
    ``token_features`` are ``[B, N, 22]``. The raw pool is ``msa_tokens [B, M, N]``
    (long residue-type indices in
    ``0..21`` -- ``20`` unknown, ``21`` gap; row 0 is the query), ``msa_deletions
    [B, M, N]`` (float per-cell insertion counts), and ``msa_profile [B, N, 22]``
    (per-column amino-acid frequency of the full pool). Template tensors are always
    present because their stacks are part of the computation; zero masks
    represent absent evidence without skipping those stacks.

    ``msa_features`` / ``msa_mask`` / ``extra_msa_features`` / ``extra_msa_mask`` are
    internal per-recycle realized slots the Model fills on a shallow per-realization
    input copy; they are ``None`` on Processor output.
    """

    residue_type: Tensor
    token_features: Tensor
    residue_index: Tensor
    sequence_mask: Tensor
    msa_tokens: Tensor
    msa_deletions: Tensor
    msa_profile: Tensor
    template_mask: Tensor
    template_residue_type: Tensor
    template_all_atom_positions: Tensor
    template_all_atom_mask: Tensor
    template_torsion_angles_sin_cos: Tensor
    template_alternative_torsion_angles_sin_cos: Tensor
    template_torsion_angles_mask: Tensor
    layout: AlphaFold2Layout
    msa_features: Tensor | None = None
    msa_mask: Tensor | None = None
    extra_msa_features: Tensor | None = None
    extra_msa_mask: Tensor | None = None
    bert_mask: Tensor | None = None
    true_msa: Tensor | None = None


@dataclass
class AlphaFold2Output:
    r"""Differentiable predictions and representations from one AF2 forward pass.

    ``distogram_logits`` uses the ``num_bins - 1`` evenly spaced boundaries from
    ``config.heads.distogram.first_break`` through ``last_break``.
    """

    single: Tensor
    pair: Tensor
    frames: Tensor
    final_atom_positions: Tensor
    final_atom_mask: Tensor
    sidechain_angles: Tensor
    sidechain_unnormalized_angles: Tensor
    distogram_logits: Tensor
    masked_msa_logits: Tensor
    plddt_logits: Tensor
    pae_logits: Tensor | None
    resolved_logits: Tensor


@dataclass(frozen=True)
class AlphaFold2MultimerLayout:
    r"""Canonical residue identities aligned to one multi-chain AF2 token axis."""

    system: MolecularSystem
    residue_sites: tuple[tuple[str, int], ...]
    chain_boundaries: tuple[tuple[str, int, int], ...]
    msa: MSACollection | None
    templates: tuple[Template, ...]


@dataclass(kw_only=True)
class AlphaFold2MultimerInput:
    r"""Complete compute and semantic input for AlphaFold-Multimer.

    Tensor shapes follow the processed-feature contract over the
    concatenated multi-chain token axis. ``residue_type`` / ``residue_index`` /
    ``sequence_mask`` are ``[B, N]``; ``token_features`` is ``[B, N, 21]``; clustered MSA
    features are ``[B, M, N, 49]`` with paired-block semantics. ``residue_index``
    restarts per chain, not globally.
    ``chain_index`` / ``entity_index`` / ``symmetry_index`` are required
    chain-identity features:

    - ``chain_index`` -- distinct copy identity, one per instance;
    - ``entity_index`` -- sequence identity, one per distinct entity;
    - ``symmetry_index`` -- copy index within an entity.

    Template tensors carry atom37 positions plus the multimer pseudo-beta and
    backbone-frame projections (no template torsion MSA row).

    As with the monomer contract, under the stochastic-fold contract the
    Processor lowers only the **raw paired alignment pool** (``msa_tokens
    [B, M, N]`` long residue-type indices in ``0..21`` -- ``20`` unknown, ``21`` gap, row 0
    the concatenated query; ``msa_deletions [B, M, N]`` per-cell insertion counts;
    ``msa_profile [B, N, 22]`` the full-pool per-column frequency). The Model draws
    the clustered/extra realization itself, once per configured ensemble member and
    recycle, off its per-sample generator (see
    :meth:`AlphaFold2MultimerModel.fold`). ``msa_features`` / ``msa_mask``
    / ``extra_msa_features`` / ``extra_msa_mask`` / ``bert_mask`` / ``true_msa`` are
    internal per-recycle realized slots the Model fills on a shallow per-realization
    input copy; they are ``None`` on Processor output. When a
    realized ``msa_features`` are present (the precomputed-feature input path) the
    Model consumes the input verbatim and skips the draw.
    """

    residue_type: Tensor
    token_features: Tensor
    residue_index: Tensor
    sequence_mask: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    msa_tokens: Tensor
    msa_deletions: Tensor
    msa_profile: Tensor
    template_mask: Tensor
    template_residue_type: Tensor
    template_all_atom_positions: Tensor
    template_all_atom_mask: Tensor
    template_pseudo_beta: Tensor
    template_pseudo_beta_mask: Tensor
    template_backbone_frame: Tensor
    template_backbone_frame_mask: Tensor
    layout: AlphaFold2MultimerLayout
    msa_features: Tensor | None = None
    msa_mask: Tensor | None = None
    extra_msa_features: Tensor | None = None
    extra_msa_mask: Tensor | None = None
    bert_mask: Tensor | None = None
    true_msa: Tensor | None = None
