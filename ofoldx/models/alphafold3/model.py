r"""
AlphaFold 3 top-level model.

Assembles the native AlphaFold 3 structure-prediction path from the shared OFoldX modules: atom-to-token
input embedding, the recycled Pairformer trunk with optional MSA and template updates, the EDM coordinate
denoiser/sampler diffusion path, and the distogram and pLDDT/PAE/PDE/resolved confidence heads. ``forward`` is
the ordinary differentiable denoising graph; full sampling stays internal until canonical decoding is complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from typing_extensions import Self

from ...data.molecular import (
    HYDROGEN_ELEMENTS,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...modules.atom import (
    AtomToTokenEncoder,
    LocalAtomGeometry,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)
from ...modules.attention import GatedMultiheadAttention, TriangleMultiplication
from ...modules.backend import LayerNorm
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import CoordinateDenoiseContext, CoordinateEdmSampler, EdmScheduler
from ...modules.embedding import (
    RelativePositionBuckets,
    RelativePositionEmbedder,
    TemplateEmbedder,
    TemplateFeatures,
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask
from ...modules.head import ConfidenceHead, DistogramHead, tm_bin_weights
from ...modules.transition import Transition
from ...modules.trunk import MsaTrunk, PairformerStack
from ...modules.trunk.msa import MsaPairWeightedAveraging, OuterProductMean
from ...results import FoldPrediction, Metric, PairMetric, Score
from .config import AlphaFold3Config


class AlphaFold3Model(nn.Module):
    r"""
    AlphaFold 3 structure-prediction model assembled from the shared OFoldX modules.

    Wires the input embedder, MSA and template embedders, the recycled Pairformer trunk, the EDM
    coordinate denoiser/sampler, and the distogram and confidence heads into a single model.

    Args:
        config: Model configuration; see
            [`AlphaFold3Config`][ofoldx.models.alphafold3.AlphaFold3Config].

    """

    config_class = AlphaFold3Config

    def __init__(self, config: AlphaFold3Config | None = None) -> None:
        config = AlphaFold3Config() if config is None else config
        super().__init__()
        self.config = config
        self._recycling_steps = config.recycling_steps
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.token_feature_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.token_feature_size, config.pair_size, bias=False)
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.input_embedder = InputEmbedder(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
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
            embed_relative_position=False,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
        )
        self.coordinate_sampler = CoordinateEdmSampler(
            scheduler=EdmScheduler(
                max_noise_level=config.sampler.max_noise_level,
                min_noise_level=config.sampler.min_noise_level,
                rho=config.sampler.rho,
                sigma_data=config.denoiser.sigma_data,
            ),
            solver=cast(Literal["euler", "heun"], config.sampler.solver),
            churn_rate=config.sampler.churn_rate,
            min_churn_noise_level=config.sampler.min_churn_noise_level,
            noise_scale=config.sampler.noise_scale,
            step_scale=config.sampler.step_scale,
            center_random_augmentation=config.sampler.center_random_augmentation,
        )
        self._num_sampling_steps = config.sampler.num_sampling_steps
        self.distogram_head = DistogramHead(config.pair_size, config.distogram.num_bins, bias=False)
        self.confidence_head = ConfidenceHead(
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            distance_bin_start=config.confidence.distance_bin_start,
            distance_bin_end=config.confidence.distance_bin_end,
            distance_bin_step=config.confidence.distance_bin_step,
            num_plddt_bins=config.confidence.num_plddt_bins,
            num_resolved_bins=config.confidence.num_resolved_bins,
            max_atoms_per_token=config.confidence.max_atoms_per_token,
            num_pae_bins=config.confidence.num_pae_bins,
            num_pde_bins=config.confidence.num_pde_bins,
            num_blocks=config.confidence.num_blocks,
            num_pair_attention_heads=config.confidence.num_pair_attention_heads,
            triangle_multiplication_hidden_size=config.confidence.triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=config.confidence.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.confidence.num_triangle_attention_heads,
            expansion=config.confidence.expansion,
            dropout=config.confidence.pairformer_dropout,
            single_max_abs=config.confidence.single_max_abs,
            # AlphaFold 3 confidence flags: no extra single LayerNorm, no raw-distance projection channel,
            # and PDE symmetrizes the projected LOGITS (DeepMind ``left + swapaxes(left, -2, -3)``). The
            # distance-bin buffers are non-persistent (absent from the checkpoint).
            use_single_norm=False,
            use_distance_projection=False,
            symmetrize_pde_before_projection=False,
            persistent_distance_bins=False,
        )
        self.msa_trunk = MsaTrunk(
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            msa_dropout=config.trunk.msa_dropout,
            pair_dropout=config.trunk.pair_dropout,
        )
        self.template_embedder = TemplateEmbedder(
            pair_size=config.pair_size,
            hidden_size=config.template.hidden_size,
            num_blocks=config.template.num_blocks,
            vocab_size=config.template.vocab_size,
            num_distogram_bins=config.template.num_distogram_bins,
            triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.template.num_triangle_attention_heads,
            expansion=config.template.expansion,
            dropout=config.template.dropout,
        )
        self.reset_parameters()
        # DeepMind AlphaFold 3's ending-node triangle self-attention (``GridSelfAttention`` with
        # ``transpose=True``) computes its pair bias from the pre-transpose pair. The shared ending-node
        # module derives it from the transposed pair by default (OpenFold convention); flip every AF3
        # ending node onto the AlphaFold 3 convention so the bias orientation matches the checkpoint.
        for module in self.modules():
            ending_node = getattr(module, "triangle_attention_ending_node", None)
            if ending_node is not None:
                ending_node.transpose_bias = True
        # AF3 uses the canonical base ``ConditionedTransformerLayer`` for atom cross-attention: its
        # key/value adaptive norm consumes the raw hidden (DeepMind AF3 ``cross_attention`` adaptive-norms
        # queries and keys independently from the raw activation), so no per-family re-casting is needed.
        self.set_backend("auto")

    def reset_parameters(self) -> None:
        r"""
        Zero-initialize the recycling and PAE/PDE projection weights after construction.

        The recycling projections and the PAE/PDE output projections start at zero so the first
        recycling pass and the untrained confidence head contribute nothing, matching the upstream
        AlphaFold 3 initialization.

        Examples:
            >>> from ofoldx.models.alphafold3 import AlphaFold3Config, AlphaFold3Model  # doctest: +SKIP
            >>> model = AlphaFold3Model(AlphaFold3Config())  # doctest: +SKIP
            >>> model.reset_parameters()  # doctest: +SKIP
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)

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
        batch: AlphaFold3Input,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        recycling_steps: int | None = None,
    ) -> AlphaFold3Output:
        r"""Denoise caller-supplied coordinates at a caller-supplied EDM noise level."""

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(batch, recycling_steps=recycling_steps)
        coordinates = self.coordinate_denoiser(
            self._denoise_context(batch, representations).to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return self._assemble_output(batch, representations, coordinates)

    @torch.inference_mode()
    def _sample(
        self,
        batch: AlphaFold3Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3Output:
        r"""Run the released sampler for a future canonical fold implementation."""

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(batch, recycling_steps=recycling_steps)
        pred_coords = self.coordinate_sampler(
            self.coordinate_denoiser,
            context=self._denoise_context(batch, representations),
            num_samples=num_samples,
            num_sampling_steps=self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps,
            generator=generator,
        )
        return self._assemble_output(batch, representations, pred_coords)

    @torch.inference_mode()
    def fold(
        self,
        batch: AlphaFold3Input,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> AlphaFold3FoldResult:
        r"""Run released AlphaFold 3 folding and decode a seed-group-preserving result.

        The released pipeline runs ``S`` seeds; each seed shares one trunk pass and draws
        ``num_samples`` diffusion rollouts, and the ``S x num_samples`` samples are pooled and
        ranked by ``ranking_score`` (upstream ``model.py:487-528``). This reproduces one seed
        group: a shared trunk pass with ``num_samples`` diffusion rollouts driven off the injected
        ``generator``, preserved intact in the family-owned
        :class:`~ofoldx.models.alphafold3.model.AlphaFold3FoldResult` (frozen DESIGN 5.2 forbids
        flattening the seed groups into the shared ``FoldResult``); ranking across the pool is the
        result's ``best()``.

        Per OWNER principle P1 the model graph is a pure function of ``(tensors, generator)``: this
        method accepts an already-resolved :class:`torch.Generator` (or ``None`` for the ambient
        RNG) and never resolves an integer seed or creates a generator. The ``seed``/``generator``
        resolution lives once at the pipeline boundary in
        :func:`~ofoldx.pipelines.resolve_generator`.

        Args:
            batch: A Processor-built batch carrying ``system``, ``atom_sites``, and ``token_sites``.
            num_samples: Diffusion rollouts drawn for the seed group (default 5).
            generator: The already-resolved RNG driving the diffusion rollouts; ``None`` uses the
                ambient RNG.
            recycling_steps: Trunk recycling passes; defaults to the config value.
            num_sampling_steps: Diffusion sampling steps; defaults to the config value.

        Returns:
            The family-owned result with a single seed group.
        """

        num_samples = 5 if num_samples is None else num_samples
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        num_sampling_steps = self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        if num_sampling_steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        if batch.system is None or not batch.atom_sites or not batch.token_sites:
            raise ValueError(
                "fold requires a Processor-built batch carrying system, atom_sites, and token_sites; "
                "bare-tensor inputs can only drive forward/_sample"
            )

        group = self._decode_seed_group(
            batch,
            num_samples=num_samples,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
            seed=0,
        )
        return AlphaFold3FoldResult(seed_groups=(group,))

    def _decode_seed_group(
        self,
        batch: AlphaFold3Input,
        *,
        num_samples: int,
        recycling_steps: int,
        num_sampling_steps: int,
        generator: torch.Generator | None,
        seed: int,
    ) -> AlphaFold3SeedGroup:
        r"""Sample one seed's diffusion rollouts and decode them into one seed group.

        Shares one trunk pass across ``num_samples`` rollouts (upstream ``model.py:320-366``),
        decodes each sample's :class:`~ofoldx.data.structure.Structure`, and finalizes the full
        released confidence surface: the pTM / ipTM / ptm_iptm_average tensor summary plus the
        chain-pair / per-chain PAE and PDE aggregates (family-owned NumPy mirroring upstream
        ``confidences.py``) and the disorder / clash ranking terms
        (:mod:`~ofoldx.models.alphafold3.model`).
        """

        output = self._sample(
            batch,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

        assert batch.system is not None
        atom_count = len(batch.atom_sites)
        token_count = len(batch.token_sites)
        complete_sites = [site for site, atom in batch.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray([complete_index[site] for site in batch.atom_sites], dtype=np.intp)

        coordinates = output.coordinates[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        atom_plddt = output.plddt[0, :, :atom_count].to(device="cpu", dtype=torch.float32)
        pae_logits = output.pae_logits[0, :, :token_count, :token_count].float().cpu()
        pde_logits = output.pde_logits[0, :, :token_count, :token_count].float().cpu()
        resolved_logits = output.resolved_logits[0, :, :atom_count].float().cpu()
        distogram_logits = output.distogram_logits[0, :token_count, :token_count].float().cpu()

        pae = _expected_bin_value(pae_logits, pae_bin_centers(pae_logits.shape[-1]).cpu())
        pde = _expected_bin_value(pde_logits, _uniform_bin_centers(pde_logits.shape[-1], 0.0, 32.0))
        resolved_probability = resolved_logits.softmax(dim=-1)[..., 1]
        contact_probability = _contact_probability(distogram_logits).numpy()
        ptm, iptm, ptm_iptm_average = (value[0].cpu() for value in self.confidence_summary(output, batch))

        # Family-owned NumPy over the already-decoded per-sample tensors, mirroring upstream
        # ``confidences.py`` (chain_pair_pae / chain_pair_pde / pde_single / pae_metrics).
        full_pae = pae.numpy()
        full_pde = pde.numpy()
        asym_ids = batch.chain_index[0, :token_count].cpu().numpy().astype(np.int64)
        frame_mask = batch.token_mask[0, :token_count].to(torch.bool).cpu().numpy()
        instance_ids = tuple(batch.system.instances.keys())
        unique_chain = np.unique(asym_ids)
        chain_axis = tuple(instance_ids[int(index)] for index in unique_chain)
        multi_chain = unique_chain.shape[0] > 1

        pae_single_mask = np.tile(frame_mask[:, None], (1, token_count))
        frame_pair_mask = frame_mask[:, None] & frame_mask[None, :]
        tm_interface = tm_adjusted_pae(
            pae_logits,
            pair_mask=torch.from_numpy(frame_pair_mask).unsqueeze(0).expand(pae_logits.shape[0], -1, -1),
            asym_id=torch.from_numpy(asym_ids).unsqueeze(0).expand(pae_logits.shape[0], -1),
            interface=True,
        ).numpy()

        chain_pair_pde_mean, chain_pair_pde_min = _chain_pair_pde(full_pde, asym_ids)
        _, chain_pair_pae_min, _ = _chain_pair_pae(full_pae, asym_ids, mask=pae_single_mask)
        pde_ichain, pde_xchain, pde_full_chain = _pde_single(asym_ids, full_pde, contact_probability)
        chain_metrics = _pae_metrics(asym_ids, full_pae, pae_single_mask, contact_probability, tm_interface)
        ranking_confidence_pae = _rank_metric(
            full_pae,
            contact_probability * frame_mask[:, None].astype(float),
            rank_metric_eps=self.config.confidence.rank_metric_eps,
        )

        predictions: list[FoldPrediction] = []
        for sample_index in range(num_samples):
            complete_coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
            coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
            complete_coordinates[model_to_complete] = coordinates[sample_index]
            coordinate_mask[model_to_complete] = True
            structure = Structure(
                system=batch.system,
                sites=tuple(complete_sites),
                coordinates=complete_coordinates,
                coordinate_mask=coordinate_mask,
            )
            disorder = fraction_disordered(structure)
            clash = has_clash(structure)
            sample_ranking_score = ranking_score(float(ptm[sample_index]), float(iptm[sample_index]), disorder, clash)
            ranking_confidence = float(ptm_iptm_average[sample_index]) if multi_chain else float(ptm[sample_index])
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores={
                        "plddt": Score(float(atom_plddt[sample_index].mean()), preferred="higher"),
                        "ptm": Score(float(ptm[sample_index]), preferred="higher"),
                        "iptm": Score(float(iptm[sample_index]), preferred="higher"),
                        "ptm_iptm_average": Score(float(ptm_iptm_average[sample_index]), preferred="higher"),
                        "ranking_score": Score(sample_ranking_score, preferred="higher"),
                        "ranking_confidence": Score(ranking_confidence, preferred="higher"),
                        "ranking_confidence_pae": Score(
                            float(ranking_confidence_pae[sample_index]), preferred="higher"
                        ),
                        "fraction_disordered": Score(disorder, preferred=None),
                        "has_clash": Score(float(clash), preferred="lower"),
                    },
                    metrics={
                        "plddt": Metric(values=atom_plddt[sample_index].numpy(), axis=batch.atom_sites),
                        "resolved_probability": Metric(
                            values=resolved_probability[sample_index].numpy(), axis=batch.atom_sites
                        ),
                        "predicted_distance_error": Metric(
                            values=pde_full_chain[sample_index],
                            axis=batch.token_sites,
                            unit="angstrom",
                        ),
                        "chain_ptm": Metric(values=chain_metrics["iptm_ichain"][sample_index], axis=chain_axis),
                        "chain_iptm": Metric(values=chain_metrics["iptm_xchain"][sample_index], axis=chain_axis),
                        "pae_ichain": Metric(
                            values=chain_metrics["pae_ichain"][sample_index], axis=chain_axis, unit="angstrom"
                        ),
                        "pae_xchain": Metric(
                            values=chain_metrics["pae_xchain"][sample_index], axis=chain_axis, unit="angstrom"
                        ),
                        "intra_chain_single_pde": Metric(
                            values=pde_ichain[sample_index], axis=chain_axis, unit="angstrom"
                        ),
                        "cross_chain_single_pde": Metric(
                            values=pde_xchain[sample_index], axis=chain_axis, unit="angstrom"
                        ),
                    },
                    pair_metrics={
                        "pae": PairMetric(
                            values=pae[sample_index].numpy(),
                            row_axis=batch.token_sites,
                            column_axis=batch.token_sites,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=pde[sample_index].numpy(),
                            row_axis=batch.token_sites,
                            column_axis=batch.token_sites,
                            unit="angstrom",
                        ),
                        "contact_probs": PairMetric(
                            values=contact_probability,
                            row_axis=batch.token_sites,
                            column_axis=batch.token_sites,
                        ),
                        "chain_pair_iptm": PairMetric(
                            values=chain_metrics["chain_pair_iptm"][sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                        "chain_pair_pae_min": PairMetric(
                            values=chain_pair_pae_min[sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_pde_mean": PairMetric(
                            values=chain_pair_pde_mean[sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_pde_min": PairMetric(
                            values=chain_pair_pde_min[sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                    },
                )
            )
        return AlphaFold3SeedGroup(seed=seed, predictions=tuple(predictions))

    def _assemble_output(
        self,
        batch: AlphaFold3Input,
        representations: tuple[Tensor, Tensor, Tensor, RelativePositionBuckets],
        coordinates: Tensor,
    ) -> AlphaFold3Output:
        r"""Compute the released distogram and confidence predictions."""

        token_embedding, single, pair, _ = representations
        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            single,
            pair=pair,
            token_embedding=token_embedding,
            pred_coords=coordinates,
            token_to_rep_atom=batch.token_to_rep_atom,
            atom_to_token=batch.atom_to_token,
            atom_token_index=batch.atom_token_index,
        )
        return AlphaFold3Output(
            coordinates=coordinates,
            distogram_logits=self.distogram_head(pair),
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
        )

    def _denoise_context(
        self,
        batch: AlphaFold3Input,
        representations: tuple[Tensor, Tensor, Tensor, RelativePositionBuckets],
    ) -> CoordinateDenoiseContext:
        r"""Assemble conditioning shared by ``forward`` and the sampler."""

        token_embedding, single, pair, relative_features = representations
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
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                batch.reference_positions,
                batch.reference_space_id,
                # Real atoms are packed contiguously at the front of the flat layout; the count selects the
                # DeepMind AF3 clamped/shifted key window (upstream ``AtomCrossAtt.compute_features``), so the
                # diffusion atom transformer attends over the same 128 real keys per window as the reference.
                num_valid_atoms=int(batch.atom_mask.to(torch.bool).sum(-1).amax().item()),
            ),
        )

    def confidence_summary(
        self,
        output: AlphaFold3Output,
        batch: AlphaFold3Input,
        *,
        max_error_bin: float = 31.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Compute the pTM / ipTM / ptm_iptm_average summary from a model output.

        This is an opt-in, parameter-free post-processing step over the PAE logits already produced by
        the confidence head: callers run `forward` and then pass the output plus the batch here to
        get the scalar tensor summaries. Chains are read from ``batch.chain_index`` and the valid frame
        mask from ``batch.token_mask``.

        The third element is ``ptm_iptm_average`` -- the pure tensor ``0.8 * ipTM + 0.2 * pTM``
        (upstream ``model.py:404``). It is **not** the released ``ranking_score``: the released one
        additionally folds in DSSP disorder and clash terms (upstream ``confidences.py:194-206``),
        which require a decoded Structure and are finalized in :meth:`fold`.

        Args:
            output: Structure-prediction output carrying ``pae_logits``.
            batch: The model input the output was produced from.
            max_error_bin: Upper edge of the resolved PAE bins in Angstrom.

        Returns:
            Scalar pTM / ipTM / ptm_iptm_average summaries per leading batch and sample axis.

        Examples:
            >>> model = AlphaFold3Model(AlphaFold3Config()).eval()  # doctest: +SKIP
            >>> batch = processor("complex.cif")  # doctest: +SKIP
            >>> ptm, iptm, ptm_iptm_average = model.confidence_summary(model(batch), batch)  # doctest: +SKIP
            >>> ptm_iptm_average.shape  # doctest: +SKIP
            torch.Size([1, 1])
        """

        num_tokens = output.pae_logits.shape[-2]
        frame_mask = batch.token_mask.to(torch.bool)
        # Broadcast token-axis fields across the per-sample logit axis so the summary is per-sample.
        sample_axes = output.pae_logits.dim() - 3
        asym_id = batch.chain_index
        for _ in range(sample_axes - asym_id.dim() + 1):
            asym_id = asym_id.unsqueeze(1)
            frame_mask = frame_mask.unsqueeze(1)
        return confidence_summary(
            output.pae_logits,
            frame_mask=frame_mask.expand(*output.pae_logits.shape[:-3], num_tokens),
            asym_id=asym_id.expand(*output.pae_logits.shape[:-3], num_tokens),
            max_error_bin=max_error_bin,
        )

    def _run_trunk(
        self,
        batch: AlphaFold3Input,
        *,
        recycling_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor, RelativePositionBuckets]:
        r"""
        Build the token embedding and run the recycled trunk to final single/pair representations.

        Args:
            batch: Typed model input.
            recycling_steps: Number of recycling passes to run.

        Returns:
            Trunk representations (token embedding, single, pair, relative features, pair mask).
        """

        token_embedding = self.input_embedder(
            batch.atom_features,
            token_features=batch.token_features,
            reference_positions=batch.reference_positions,
            reference_charges=batch.reference_charges,
            reference_mask=batch.reference_mask,
            atom_to_token=batch.atom_to_token,
            num_tokens=batch.token_features.shape[-2],
            atom_mask=batch.atom_mask,
            token_mask=batch.token_mask,
            local_geometry=self.input_embedder.build_local_geometry(
                batch.reference_positions,
                batch.reference_space_id,
            ),
        )

        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            dtype=single_init.dtype,
        )
        pair_init = self._initial_pair(token_embedding, batch, relative_features)

        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        single, pair = self._run_recycling(
            batch,
            token_embedding,
            single_init,
            pair_init,
            pair_mask,
            recycling_steps,
        )

        return token_embedding, single, pair, relative_features

    def _run_recycling(
        self,
        batch: AlphaFold3Input,
        token_embedding: Tensor,
        single_init: Tensor,
        pair_init: Tensor,
        pair_mask: Tensor,
        recycling_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Run the released recycling loop as an ordinary differentiable graph."""

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for _ in range(recycling_steps):
            pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
            if batch.template_features is not None:
                pair = pair + self.template_embedder(pair, batch.template_features, pair_mask=pair_mask)
            if batch.msa_features is not None:
                _, pair = self.msa_trunk(
                    batch.msa_features, pair, token_embedding, msa_mask=batch.msa_mask, pair_mask=pair_mask
                )
            single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
            single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        return single, pair

    def _initial_pair(
        self,
        token_embedding: torch.Tensor,
        batch: AlphaFold3Input,
        relative_features: RelativePositionBuckets,
    ) -> torch.Tensor:
        r"""
        Build the initial pair representation from single projections, relative positions, and token bonds.

        Args:
            token_embedding: Initial target-feature representation ``(..., N_token, token_feature_size)``.
            batch: Typed model input (provides token bonds and the token mask).
            relative_features: Relative-position bucket features for the pair embedder.

        Returns:
            Masked initial pair representation ``(..., N_token, N_token, pair_size)``.
        """

        left = self.pair_left_projection(token_embedding).unsqueeze(-2)
        right = self.pair_right_projection(token_embedding).unsqueeze(-3)
        token_bonds = batch.token_bonds.to(device=token_embedding.device, dtype=token_embedding.dtype)
        token_bonds = torch.broadcast_to(token_bonds, (*token_embedding.shape[:-1], token_embedding.shape[-2]))
        token_bonds = token_bonds.unsqueeze(-1)
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds)

        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)


class InputEmbedder(nn.Module):
    r"""
    Build token input embeddings by encoding atoms to tokens and appending the token-feature tail.

    Runs the atom-to-token encoder over the packed atom features, then concatenates the encoder's token
    channels with the trailing per-token features to form the initial token embedding.

    Args:
        single_size: Width of the single (token) representation.
        atom_feature_size: Width of the packed per-atom feature vector.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 385,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.atom_encoder = AtomToTokenEncoder(
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
        atom_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> Tensor:
        r"""
        Return token embeddings concatenating atom-derived channels with the token-feature tail.

        Args:
            atom_features: Packed per-atom features ``(..., N_atom, atom_feature_size)``.
            reference_positions: Centered reference-conformer atom coords ``(..., N_atom, 3)``.
            reference_charges: Per-atom reference formal charges ``(..., N_atom, 1)`` (``arcsinh``-compressed
                before encoding).
            reference_mask: Boolean mask for valid reference-conformer atoms ``(..., N_atom)``.
            token_features: Packed per-token features ``(..., N_token, token_feature_size)``.
            atom_to_token: Token index for each atom ``(..., N_atom)``.
            num_tokens: Number of tokens ``N_token``.
            atom_mask: Optional boolean atom validity mask ``(..., N_atom)``.
            token_mask: Optional boolean token validity mask ``(..., N_token)``; masked tokens are zeroed.
            local_geometry: Optional precomputed local atom geometry for the atom encoder.

        Returns:
            Token embedding ``(..., N_token, single_size + token_tail)``.
        """

        encoder_reference_charges = torch.arcsinh(reference_charges)
        atom_encoder_output = self.atom_encoder(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=encoder_reference_charges,
            reference_mask=reference_mask,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
            local_geometry=local_geometry,
        )
        token_single = atom_encoder_output.token_hidden

        token_tail = token_features[..., self.single_size :].to(token_single)
        token_embedding = torch.cat([token_single, token_tail], dim=-1)

        if token_mask is not None:
            token_embedding = token_embedding * token_mask.unsqueeze(-1).to(dtype=token_embedding.dtype)
        return token_embedding

    def build_local_geometry(self, reference_positions: Tensor, reference_space_id: Tensor) -> LocalAtomGeometry:
        r"""
        Build local atom geometry using the input atom-encoder layout.

        Args:
            reference_positions: Centered reference-conformer atom coords ``(..., N_atom, 3)``.
            reference_space_id: Reference-conformer group id for each atom ``(..., N_atom)``.

        Returns:
            Local atom geometry for the input atom encoder.
        """

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)


# ipTM contributes 0.8 and pTM 0.2 to the combined ranking score (upstream ``_IPTM_WEIGHT``).
_IPTM_WEIGHT = 0.8


def _uniform_bin_centers(num_bins: int, minimum: float, maximum: float) -> Tensor:
    r"""Return the midpoints of ``num_bins`` equal bins tiling ``[minimum, maximum]``."""

    edges = torch.linspace(minimum, maximum, num_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def _expected_bin_value(logits: Tensor, centers: Tensor) -> Tensor:
    r"""Take the softmax expectation of ``logits`` over categorical distance/error ``centers``."""

    return logits.softmax(dim=-1) @ centers.to(logits.dtype)


def _contact_probability(distogram_logits: Tensor) -> Tensor:
    r"""Probability that a token pair is within 8 Angstrom under the distogram head."""

    centers = _uniform_bin_centers(distogram_logits.shape[-1], 2.3125, 21.6875)
    return distogram_logits.softmax(dim=-1)[..., centers < 8.0].sum(dim=-1)


def pae_bin_centers(num_bins: int, max_error_bin: float = 31.0, *, device: torch.device | None = None) -> Tensor:
    r"""
    Return the PAE-bin center distances used to take the expected aligned error.

    The first ``num_bins - 1`` centers evenly tile ``[0, max_error_bin]`` at half-step offsets and a
    final catch-all bin is appended one step beyond, matching the upstream ``pae_breaks`` construction.

    Args:
        num_bins: Number of PAE logit bins.
        max_error_bin: Upper edge of the resolved (non-catch-all) bins in Angstrom.
        device: Optional device for the returned tensor.

    Returns:
        ``[num_bins]`` bin-center distances in Angstrom.

    Raises:
        ValueError: If ``num_bins`` is less than two.

    Examples:
        >>> import torch
        >>> centers = pae_bin_centers(4)
        >>> centers.shape
        torch.Size([4])
        >>> centers.round(decimals=3).tolist()
        [7.75, 23.25, 38.75, 54.25]
        >>> pae_bin_centers(1)
        Traceback (most recent call last):
            ...
        ValueError: num_bins must be at least 2, got 1
    """

    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    breaks = torch.linspace(0.0, max_error_bin, num_bins - 1, device=device)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    return torch.cat([centers, centers[-1:] + step], dim=0)


def tm_adjusted_pae(
    pae_logits: Tensor,
    *,
    pair_mask: Tensor,
    asym_id: Tensor,
    interface: bool,
    max_error_bin: float = 31.0,
) -> Tensor:
    r"""
    Return the per-token-pair TM-adjusted PAE from PAE logits.

    Softmaxes the logits into a distance-error distribution, weights each bin by its TM-score term
    ``1 / (1 + (d / d0)^2)``, and sums over bins. The TM-score length scale ``d0`` uses the interface
    token count (both chains of the pair) when ``interface`` is set, else the global token count, so
    ipTM and pTM use the same tensor construction with different normalization counts (upstream
    ``tmscore_adjusted_pae_interface`` vs ``tmscore_adjusted_pae_global``).

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        asym_id: ``[..., N_token]`` chain (asym) id per token.
        interface: If ``True``, use the interface token count for ``d0``; else the global count.
        max_error_bin: Upper edge of the resolved PAE bins in Angstrom.

    Returns:
        ``[..., N_token, N_token]`` TM-adjusted PAE.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> logits = torch.randn(1, 3, 3, 4)
        >>> mask = torch.ones(1, 3, dtype=torch.bool)
        >>> pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        >>> asym_id = torch.tensor([[0, 0, 1]])
        >>> tm_adjusted_pae(logits, pair_mask=pair_mask, asym_id=asym_id, interface=False).shape
        torch.Size([1, 3, 3])
    """

    num_bins = pae_logits.shape[-1]
    centers = pae_bin_centers(num_bins, max_error_bin, device=pae_logits.device).to(pae_logits.dtype)
    probs = torch.softmax(pae_logits.float(), dim=-1).to(pae_logits.dtype)
    pair_mask = pair_mask.to(pae_logits.dtype)

    same_chain = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
    if interface:
        num_chain_tokens = (same_chain.to(pae_logits.dtype) * pair_mask).sum(dim=-1)
        num_tokens = num_chain_tokens.unsqueeze(-1) + num_chain_tokens.unsqueeze(-2)
        # Don't double-count within a single chain (upstream ``num_interface_tokens`` correction).
        num_tokens = num_tokens - same_chain.to(pae_logits.dtype) * torch.floor(num_tokens / 2)
        num_tokens = num_tokens * pair_mask
    else:
        num_tokens = pair_mask.new_full(pair_mask.shape[:-2], float(pair_mask.shape[-1])).unsqueeze(-1).unsqueeze(-1)
        num_tokens = num_tokens.expand_as(pair_mask)

    return (probs * tm_bin_weights(centers, num_tokens)).sum(dim=-1)


def predicted_tm_score(adjusted_pae: Tensor, *, pair_mask: Tensor, asym_id: Tensor, interface: bool) -> Tensor:
    r"""
    Reduce a TM-adjusted PAE matrix to a scalar pTM or ipTM.

    Restricts to cross-chain pairs when ``interface`` is set, row-normalizes the mask, takes the masked
    per-row average of the adjusted PAE, and returns the best row. Returns ``NaN`` when the (interface)
    mask is empty, matching upstream ``predicted_tm_score`` (single-chain systems have no ipTM).

    Args:
        adjusted_pae: ``[..., N_token, N_token]`` TM-adjusted PAE.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        asym_id: ``[..., N_token]`` chain id per token.
        interface: If ``True``, compute ipTM over cross-chain pairs; else pTM over all pairs.

    Returns:
        ``[...]`` pTM or ipTM score.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> adjusted = torch.rand(1, 3, 3)
        >>> mask = torch.ones(1, 3, dtype=torch.bool)
        >>> pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        >>> asym_id = torch.tensor([[0, 0, 1]])
        >>> predicted_tm_score(adjusted, pair_mask=pair_mask, asym_id=asym_id, interface=False).shape
        torch.Size([1])
    """

    mask = pair_mask.to(adjusted_pae.dtype)
    if interface:
        mask = mask * (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)).to(adjusted_pae.dtype)
    normed = mask / (torch.finfo(mask.dtype).tiny + mask.sum(dim=-1, keepdim=True))
    per_alignment = (adjusted_pae * normed).sum(dim=-1)
    score = per_alignment.max(dim=-1).values
    empty = mask.sum(dim=(-2, -1)) == 0
    return torch.where(empty, torch.full_like(score, float("nan")), score)


def confidence_summary(
    pae_logits: Tensor,
    *,
    frame_mask: Tensor,
    asym_id: Tensor,
    max_error_bin: float = 31.0,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute pTM, ipTM, and the ptm_iptm_average from PAE logits.

    The third element is ``ptm_iptm_average`` = ``0.8 * ipTM + 0.2 * pTM`` (pTM substituted for a
    NaN ipTM on single-chain systems), the pure tensor quantity of upstream ``model.py:404``. It is
    **not** the released ``ranking_score``, which additionally folds in DSSP disorder and clash terms
    (upstream ``confidences.py:194-206``); those require a decoded Structure and are added in
    :meth:`AlphaFold3Model.fold`.

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        frame_mask: ``[..., N_token]`` boolean mask of tokens with a valid frame.
        asym_id: ``[..., N_token]`` chain (asym) id per token.
        max_error_bin: Upper edge of the resolved PAE bins in Angstrom.

    Returns:
        Scalar pTM / ipTM / ptm_iptm_average summaries broadcast over the leading batch/sample axes.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> logits = torch.randn(1, 3, 3, 4)
        >>> frame_mask = torch.ones(1, 3, dtype=torch.bool)
        >>> asym_id = torch.tensor([[0, 0, 1]])
        >>> ptm, iptm, ptm_iptm_average = confidence_summary(logits, frame_mask=frame_mask, asym_id=asym_id)
        >>> ptm.shape
        torch.Size([1])
        >>> ptm_iptm_average.shape
        torch.Size([1])
    """

    frame_mask = frame_mask.to(torch.bool)
    pair_mask = frame_mask.unsqueeze(-1) & frame_mask.unsqueeze(-2)
    global_adjusted = tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, asym_id=asym_id, interface=False, max_error_bin=max_error_bin
    )
    interface_adjusted = tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, asym_id=asym_id, interface=True, max_error_bin=max_error_bin
    )
    ptm = predicted_tm_score(global_adjusted, pair_mask=pair_mask, asym_id=asym_id, interface=False)
    iptm = predicted_tm_score(interface_adjusted, pair_mask=pair_mask, asym_id=asym_id, interface=True)
    ptm_iptm_average = torch.where(torch.isnan(iptm), ptm, _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm)
    return ptm, iptm, ptm_iptm_average


# --- Family-owned NumPy confidence aggregates ---------------------------------------
# Faithful re-expressions of the released per-sample chain-pair / per-chain PAE and PDE
# summaries (upstream ``confidences.py`` ``chain_pair_pae`` / ``chain_pair_pde`` /
# ``pde_single`` / ``pae_metrics`` / ``rank_metric``), operating on the already-decoded
# ``[num_samples, num_tokens, num_tokens]`` NumPy tensors. ``asym_ids`` is the per-token
# chain index; chains are the sorted unique chain indices.


def _weighted_mean(mask: np.ndarray, value: np.ndarray, axis: object) -> np.ndarray:
    r"""Contact/frame-weighted mean, upstream ``confidences.weighted_mean``."""

    return np.mean(mask * value, axis=axis) / (np.finfo(value.dtype).tiny + np.mean(mask, axis=axis))


def _weighted_nanmean(value: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    r"""NaN-aware weighted mean; empty slices return NaN (upstream ``weighted_nanmean``)."""

    mask = mask.astype(float)
    nan_idxs = np.where(np.isnan(value))
    mask_with_nan = mask.copy()
    mask_with_nan[nan_idxs] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(value * mask_with_nan, axis=axis) / np.nanmean(mask_with_nan, axis=axis)


def _chain_pair_pde(full_pde: np.ndarray, asym_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Per chain-pair mean and min predicted distance error (upstream ``chain_pair_pde``)."""

    unique = np.unique(asym_ids)
    num_chains = len(unique)
    num_samples = full_pde.shape[0]
    mean = np.zeros((num_samples, num_chains, num_chains))
    minimum = np.zeros((num_samples, num_chains, num_chains))
    for first, asym_first in enumerate(unique):
        subset = full_pde[:, asym_ids == asym_first, :]
        for second, asym_second in enumerate(unique):
            block = subset[:, :, asym_ids == asym_second]
            mean[:, first, second] = np.mean(block, axis=(1, 2))
            minimum[:, first, second] = np.min(block, axis=(1, 2))
    return mean, minimum


def _pde_single(
    asym_ids: np.ndarray, full_pde: np.ndarray, contact_probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Contact-weighted intra/cross-chain and per-token PDE summaries (upstream ``pde_single``)."""

    unique = np.unique(asym_ids)
    num_chains = len(unique)
    num_samples = full_pde.shape[0]
    asym = asym_ids[None]
    contact = contact_probs[None]
    ichain = np.zeros((num_samples, num_chains))
    xchain = np.zeros((num_samples, num_chains))
    for index, asym_id in enumerate(unique):
        mine = asym == asym_id
        intra = mine[:, :, None] * mine[:, None, :] * contact
        cross = mine[:, :, None] * (~mine[:, None, :]) * contact
        ichain[:, index] = _weighted_mean(intra, full_pde, axis=(-2, -1))
        xchain[:, index] = _weighted_mean(cross, full_pde, axis=(-2, -1))
    full_chain = _weighted_mean(contact, full_pde, axis=(-1,))
    return ichain, xchain, full_chain


def _chain_pair_pae(
    full_pae: np.ndarray,
    asym_ids: np.ndarray,
    *,
    mask: np.ndarray,
    contact_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Per chain-pair mean/min predicted aligned error (upstream ``chain_pair_pae``)."""

    if contact_probs is None:
        contact_probs = np.ones(full_pae.shape[1:], dtype=float)
    unique = np.unique(asym_ids)
    num_chains = len(unique)
    num_samples = full_pae.shape[0]
    mean = np.zeros((num_samples, num_chains, num_chains))
    minimum = np.zeros((num_samples, num_chains, num_chains))
    for first, asym_first in enumerate(unique):
        subset = full_pae[:, asym_ids == asym_first, :]
        subset_mask = mask[asym_ids == asym_first, :]
        subset_contact = contact_probs[asym_ids == asym_first, :]
        for second, asym_second in enumerate(unique):
            block = subset[:, :, asym_ids == asym_second]
            block_mask = subset_mask[:, asym_ids == asym_second]
            block_contact = subset_contact[:, asym_ids == asym_second]
            (kept,) = np.where(block_mask.flatten() > 0)
            flat_block = block.reshape(num_samples, -1)
            flat_contact = block_contact.flatten()
            if not kept.size:
                mean[:, first, second] = np.nan
                minimum[:, first, second] = np.nan
            else:
                minimum[:, first, second] = np.min(flat_block[:, kept], axis=1)
                mean[:, first, second] = _weighted_mean(flat_contact[kept], flat_block[:, kept], axis=-1)
    return mean, minimum, unique


def _predicted_tm_score_np(
    tm_adjusted_pae: np.ndarray, pair_mask: np.ndarray, asym_id: np.ndarray, interface: bool
) -> float:
    r"""Scalar pTM / ipTM over one sample (upstream ``confidences.predicted_tm_score``)."""

    if interface:
        pair_mask = pair_mask * (asym_id[:, None] != asym_id[None, :])
    if pair_mask.sum() == 0:
        return float("nan")
    normed = pair_mask / (np.finfo(tm_adjusted_pae.dtype).tiny + np.sum(pair_mask, axis=-1, keepdims=True))
    per_alignment = np.sum(tm_adjusted_pae * normed, axis=-1)
    return float(per_alignment.max())


def _chain_pairwise_iptm(tm_adjusted_pae: np.ndarray, pair_mask: np.ndarray, asym_id: np.ndarray) -> np.ndarray:
    r"""Per chain-pair ipTM matrix for one sample (upstream ``chain_pairwise_predicted_tm_scores``)."""

    unique = list(np.unique(asym_id))
    num_chains = len(unique)
    pairwise = np.zeros((num_chains, num_chains))
    for first, chain_first in enumerate(unique):
        first_mask = asym_id == chain_first
        for offset, chain_second in enumerate(unique[first:]):
            second_mask = asym_id == chain_second
            selected = first_mask | second_mask
            (indices,) = np.where(selected)
            is_interface = chain_first != chain_second
            grid = np.ix_(indices, indices)
            value = _predicted_tm_score_np(tm_adjusted_pae[grid], pair_mask[grid], asym_id[selected], is_interface)
            pairwise[first, first + offset] = value
            pairwise[first + offset, first] = value
    return pairwise


def _reduce_chain_pair(
    chain_pair: np.ndarray, num_chain_tokens: np.ndarray, *, agg_over_col: bool
) -> tuple[np.ndarray, np.ndarray]:
    r"""Per-chain intra (diagonal) and cross-chain mean reductions (upstream ``reduce_chain_pair``).

    Uses the released ``per_chain`` weighting (unit chain weights) and mean aggregation.
    """

    num_samples, num_chains, _ = chain_pair.shape
    ichain = chain_pair.diagonal(axis1=-2, axis2=-1)
    agg_axis = -1 if agg_over_col else -2
    weight = np.ones((num_samples, num_chains, num_chains), dtype=float)
    weight -= np.eye(num_chains, dtype=float)
    xchain = _weighted_nanmean(chain_pair, mask=weight, axis=agg_axis)
    return ichain, xchain


def _pae_metrics(
    asym_ids: np.ndarray,
    full_pae: np.ndarray,
    mask: np.ndarray,
    contact_probs: np.ndarray,
    tm_adjusted_pae_interface: np.ndarray,
) -> dict[str, np.ndarray]:
    r"""Per-chain ipTM / PAE aggregates and the chain-pair ipTM matrix (upstream ``pae_metrics``)."""

    chain_pair_contact_weighted, _, unique = _chain_pair_pae(full_pae, asym_ids, mask=mask, contact_probs=contact_probs)
    chain_pair_iptm = np.stack(
        [
            _chain_pairwise_iptm(tm_adjusted_pae=sample_tm, asym_id=asym_ids, pair_mask=mask)
            for sample_tm in tm_adjusted_pae_interface
        ],
        axis=0,
    )
    num_chain_tokens = np.array([np.sum(asym_ids == asym_id) for asym_id in unique])

    def reduce(chain_pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ichain, xchain_row = _reduce_chain_pair(chain_pair, num_chain_tokens, agg_over_col=False)
        _, xchain_col = _reduce_chain_pair(chain_pair, num_chain_tokens, agg_over_col=True)
        with np.errstate(invalid="ignore"):
            xchain = np.nanmean(np.stack([xchain_row, xchain_col], axis=0), axis=0)
        return ichain, xchain

    pae_ichain, pae_xchain = reduce(chain_pair_contact_weighted)
    iptm_ichain, iptm_xchain = reduce(chain_pair_iptm)
    return {
        "chain_pair_iptm": chain_pair_iptm,
        "iptm_ichain": iptm_ichain,
        "iptm_xchain": iptm_xchain,
        "pae_ichain": pae_ichain,
        "pae_xchain": pae_xchain,
    }


def _rank_metric(full_pae: np.ndarray, contact_probs: np.ndarray, *, rank_metric_eps: float) -> np.ndarray:
    r"""Contact-weighted PAE ranking metric, higher is better (upstream ``rank_metric``)."""

    return -np.sum(full_pae * contact_probs[None, :, :], axis=(-2, -1)) / (np.sum(contact_probs) + rank_metric_eps)


# ---------------------------------------------------------------------------
# Model contract (folded in from the former ``contract.py``): the tensor
# boundary of the released AlphaFold 3 model graph.
# ---------------------------------------------------------------------------


def expected_plddt(logits: Tensor) -> Tensor:
    r"""Expected pLDDT on the conventional 0--100 scale from per-bin logits."""

    centers = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype) + 0.5
    return (logits.softmax(dim=-1) * centers * (100.0 / logits.shape[-1])).sum(dim=-1)


@dataclass(kw_only=True)
class AlphaFold3Input:
    r"""Tensors consumed by the released AlphaFold 3 graph.

    ``reference_charges`` has the unique shape ``[..., N_atom, 1]``.

    ``system``, ``atom_sites``, and ``token_sites`` carry the canonical identities
    used to reconstruct the folding result. They are optional so the pure-tensor
    model graph (``forward`` / ``_sample``) can be driven with bare tensors, but the
    :class:`AlphaFold3Processor` always populates them so :meth:`AlphaFold3Model.fold`
    can decode coordinates back into a :class:`~ofoldx.data.structure.Structure`.
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
    atom_token_index: Tensor
    token_to_rep_atom: Tensor
    token_features: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    token_bonds: Tensor
    msa_features: Tensor | None = None
    msa_mask: Tensor | None = None
    template_features: TemplateFeatures | None = None
    system: MolecularSystem | None = None
    atom_sites: tuple[AtomSite, ...] = ()
    token_sites: tuple[AtomSite | ComponentReference, ...] = ()


@dataclass
class AlphaFold3Output:
    r"""Predictions computed by one AlphaFold 3 denoising call."""

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    pae_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor

    @property
    def plddt(self) -> Tensor:
        r"""Expected pLDDT on the conventional 0--100 scale."""

        return expected_plddt(self.plddt_logits)


# ---------------------------------------------------------------------------------
# Family-owned seed-group-preserving folding result (folded from ``result.py``).
#
# The released AlphaFold 3 inference pipeline runs ``S`` seeds; each seed shares one
# trunk pass and draws ``num_diffusion_samples`` diffusion rollouts, and the
# ``S x num_samples`` samples are pooled and ranked (upstream ``model.py:487-528``).
# Per frozen DESIGN 5.2, a family that returns several seed groups owns a concrete
# result that *preserves* those groups rather than flattening them into the shared
# ``ofoldx.results.FoldResult``. The shared ``FoldPrediction`` value is reused
# unchanged; only the outer grouping is family-specific.
# ---------------------------------------------------------------------------------


@dataclass(eq=False)
class AlphaFold3SeedGroup:
    r"""One diffusion seed and its generation-ordered samples.

    Attributes:
        seed: The integer diffusion seed shared by every sample in this group.
        predictions: Generation-ordered samples drawn for this seed; tuple
            position is the sample identity within the group.
    """

    seed: int
    predictions: tuple[FoldPrediction, ...]


@dataclass(eq=False)
class AlphaFold3FoldResult:
    r"""One AlphaFold 3 folding result preserving its per-seed sample groups.

    Attributes:
        seed_groups: One :class:`AlphaFold3SeedGroup` per seed, in the order the
            seeds were run.
    """

    seed_groups: tuple[AlphaFold3SeedGroup, ...]

    def best(self, score: str) -> FoldPrediction:
        r"""Return the earliest best prediction under one explicit score.

        Pools every sample across every seed group and ranks the pool by the named
        :class:`~ofoldx.results.Score`, matching the upstream pooled ranking across
        all ``S x num_samples`` samples. This reuses the exact
        explicit-direction, identical-unit, identical-scope, NaN-rejecting,
        infinity-rankable, earliest-tie semantics of
        :meth:`~ofoldx.results.FoldResult.best` (frozen DESIGN 5.2 forbids routing
        this through a shared rank helper, so the checks are written directly).

        Args:
            score: The name of the per-sample score to rank on.

        Returns:
            The earliest pooled sample that is best under ``score``.

        Raises:
            ValueError: If the score does not consistently prefer higher or lower
                values, mixes units or scopes, or contains a NaN value.
        """

        predictions = tuple(prediction for group in self.seed_groups for prediction in group.predictions)
        scores = tuple(prediction.scores[score] for prediction in predictions)
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
            return max(predictions, key=lambda prediction: prediction.scores[score].value)
        return min(predictions, key=lambda prediction: prediction.scores[score].value)


# ---------------------------------------------------------------------------------
# Family-owned DSSP / geometry post-processing for the ranking score (folded from
# ``confidence.py``).
#
# Faithful re-expressions of the upstream confidence utilities
# (``/data/users/z/reference/alphafold3/src/alphafold3/model/confidences.py:56-206``)
# that the tensor-only ``confidence_summary`` cannot compute because they require a
# *decoded* ``ofoldx.data.structure.Structure``:
#
# - ``fraction_disordered`` -- windowed AlphaFold-RSA disorder fraction over protein
#   chains (upstream ``confidences.py:62-137``).
# - ``has_clash`` -- polymer-atom clash detection over protein / RNA / DNA chains
#   (upstream ``confidences.py:140-191``).
# - ``ranking_score`` -- the released ranking score combining the tensor
#   ``ptm_iptm_average`` with the disorder and clash terms (upstream
#   ``confidences.py:194-206``).
#
# These consume the per-sample decoded Structure, so the AlphaFold 3 ``fold`` decode
# runs them once per sample to finalize the ``ranking_score`` Score.
# ---------------------------------------------------------------------------------

# --- Ranking weights (upstream ``confidences.py:56-58``) -------------------------
_IPTM_WEIGHT = 0.8
_FRACTION_DISORDERED_WEIGHT = 0.5
_CLASH_PENALIZATION_WEIGHT = 100.0

# Maximum accessible surface areas from Sander & Rost 1994
# (https://doi.org/10.1002/prot.340200303); upstream ``confidences.py:32-52``. AF3
# owns this table because it drives the AlphaFold-RSA disorder fraction.
MAX_ACCESSIBLE_SURFACE_AREA = {
    "ALA": 106.0,
    "ARG": 248.0,
    "ASN": 157.0,
    "ASP": 163.0,
    "CYS": 135.0,
    "GLN": 198.0,
    "GLU": 194.0,
    "GLY": 84.0,
    "HIS": 184.0,
    "ILE": 169.0,
    "LEU": 164.0,
    "LYS": 205.0,
    "MET": 188.0,
    "PHE": 197.0,
    "PRO": 136.0,
    "SER": 130.0,
    "THR": 142.0,
    "TRP": 227.0,
    "TYR": 222.0,
    "VAL": 142.0,
}

# rASA disorder threshold, AlphaFold-RSA
# (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601767); upstream default.
_RASA_DISORDER_CUTOFF = 0.581
_RASA_WINDOW = 25

_POLYMER_TYPES = frozenset({"protein", "rna", "dna", "dna_rna_hybrid"})


def ranking_score(ptm: float, iptm: float, fraction_disordered_: float, has_clash_: bool) -> float:
    r"""Return the released AlphaFold 3 ranking score for one sample.

    ``ptm_iptm_average + 0.5 * fraction_disordered - 100 * has_clash``, with pTM
    substituted for the ipTM term when ipTM is ``NaN`` (single-chain systems),
    exactly matching upstream ``confidences.get_ranking_score``
    (``confidences.py:194-206``).

    Args:
        ptm: The predicted global TM score.
        iptm: The interface predicted TM score (``NaN`` for single-chain systems).
        fraction_disordered_: The AlphaFold-RSA disorder fraction in ``[0, 1]``.
        has_clash_: Whether the structure has at least one clashing chain.

    Returns:
        The scalar ranking score (higher is better).
    """

    if np.isnan(iptm):
        ptm_iptm_average = ptm
    else:
        ptm_iptm_average = _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm
    return (
        ptm_iptm_average
        + _FRACTION_DISORDERED_WEIGHT * fraction_disordered_
        - _CLASH_PENALIZATION_WEIGHT * float(has_clash_)
    )


def fraction_disordered(structure: Structure, rasa_disorder_cutoff: float = _RASA_DISORDER_CUTOFF) -> float:
    r"""Fraction of protein residues predicted disordered by windowed AlphaFold-RSA.

    Restricts to protein chains, computes a windowed (window 25) relative solvent
    accessible surface area per residue via a DSSP backend, and returns the mean
    fraction of residues whose rASA exceeds ``rasa_disorder_cutoff`` (upstream
    ``confidences.py:99-137``). Identical chain sequences reuse a cached rASA
    profile. When no DSSP backend is available (or the DSSP call fails), the rASA
    list stays empty and the function returns ``0.0``, exactly matching the
    upstream on-failure behavior (``confidences.py:132-137``).

    Args:
        structure: The decoded per-sample structure to score.
        rasa_disorder_cutoff: The rASA threshold above which a residue counts as
            disordered.

    Returns:
        The fraction of protein residues predicted disordered, in ``[0, 1]``.
    """

    rasa: list[float] = []
    sequence_rasa: dict[tuple[str, ...], np.ndarray] = {}
    for instance_id in _protein_chain_ids(structure):
        chain_sequence = _chain_sequence(structure, instance_id)
        cached = sequence_rasa.get(chain_sequence)
        if cached is not None:
            rasa.extend(cached.tolist())
            continue
        try:
            per_residue = _windowed_solvent_accessible_area(structure, instance_id)
        except (ValueError, RuntimeError, ImportError):
            continue
        sequence_rasa[chain_sequence] = per_residue
        rasa.extend(per_residue.tolist())

    if not rasa:
        return 0.0
    return float(np.mean(np.asarray(rasa) > rasa_disorder_cutoff))


def has_clash(
    structure: Structure,
    cutoff_radius: float = 1.1,
    min_clashes_for_overlap: int = 100,
    min_fraction_for_overlap: float = 0.5,
) -> bool:
    r"""Whether the structure has at least one clashing polymer chain.

    Restricts to protein / RNA / DNA polymer atoms, then flags a chain as
    overlapping when more than ``min_clashes_for_overlap`` of its atoms lie within
    ``cutoff_radius`` Angstrom of a polymer atom in another chain (or a residue more
    than one apart in the same chain), or when more than ``min_fraction_for_overlap``
    of the chain's atoms clash (upstream ``confidences.py:140-191``).

    Args:
        structure: The decoded per-sample structure to score.
        cutoff_radius: Atom-pair distances below this (Angstrom) count as a clash.
        min_clashes_for_overlap: Minimum clashing atoms for a chain to overlap.
        min_fraction_for_overlap: Minimum clashing fraction for a chain to overlap.

    Returns:
        ``True`` if at least one polymer chain overlaps another.
    """

    coordinates, residue_ids, chain_ids = _polymer_atoms(structure)
    if coordinates.shape[0] == 0:
        return False

    neighbors = _query_ball(coordinates, cutoff_radius)
    per_atom_has_clash = np.zeros(coordinates.shape[0], dtype=bool)
    for atom_index, clashing in enumerate(neighbors):
        for other in clashing:
            if abs(int(residue_ids[atom_index]) - int(residue_ids[other])) > 1 or (
                chain_ids[atom_index] != chain_ids[other]
            ):
                per_atom_has_clash[atom_index] = True
                break

    for chain_id in np.unique(chain_ids):
        mask = chain_ids == chain_id
        num_atoms = int(np.sum(mask))
        if num_atoms == 0:
            continue
        num_clashes = int(np.sum(per_atom_has_clash & mask))
        fraction_clashes = num_clashes / num_atoms
        if num_clashes > min_clashes_for_overlap or fraction_clashes > min_fraction_for_overlap:
            return True
    return False


def _windowed_solvent_accessible_area(structure: Structure, instance_id: str) -> np.ndarray:
    r"""Windowed per-residue rASA for one protein chain (AlphaFold-RSA).

    Runs the solvent-accessibility backend over the chain, normalizes each residue's
    accessible surface area by :data:`MAX_ACCESSIBLE_SURFACE_AREA`, clips to ``1.0``, and
    averages over a ``window``-wide reflect-padded window (upstream ``confidences.py:62-96``).
    Raises ``ImportError`` / ``RuntimeError`` when the backend is unavailable, which the
    caller treats as an unavailable backend and degrades to ``0.0``.
    """

    accessible = _dssp_accessible_area(structure, instance_id)
    if not accessible:
        raise RuntimeError("DSSP produced no accessible-area rows")
    rasa = []
    for residue_name, area in accessible:
        max_area = MAX_ACCESSIBLE_SURFACE_AREA.get(residue_name, MAX_ACCESSIBLE_SURFACE_AREA["ALA"])
        normalized = area / max_area
        rasa.append(min(normalized, 1.0))

    half_window = (_RASA_WINDOW - 1) // 2
    padded = np.pad(np.asarray(rasa, dtype=float), (half_window, half_window), "reflect")
    return np.convolve(padded, np.ones(_RASA_WINDOW), "valid") / _RASA_WINDOW


# Shrake-Rupley solvent-accessibility parameters. Released AF3 obtains per-residue
# accessible surface area from ``mkdssp`` (DSSP's numerical Lee-Richards integration).
# OFoldX has no DSSP binary, so it substitutes biotite's Shrake-Rupley SASA with the
# standard 1.4 Angstrom water probe and ProtOr heavy-atom radii (Tsai 1999), which is
# the accessibility model the Sander & Rost max-ASA normalization was calibrated
# against. ``fraction_disordered`` averages this over a 25-residue window and thresholds
# it, so per-residue SASA differences below the DSSP/Shrake-Rupley level wash out.
_SASA_PROBE_RADIUS = 1.4
_SASA_POINT_NUMBER = 1000


def _dssp_accessible_area(structure: Structure, instance_id: str) -> list[tuple[str, float]]:
    r"""Return per-residue ``(residue_name, accessible_area)`` for one protein chain.

    This is the solvent-accessibility backend that released AF3 delegates to ``mkdssp``.
    OFoldX bundles no DSSP binary, so accessibility comes from biotite's Shrake-Rupley
    :func:`biotite.structure.sasa` over the chain's observed heavy atoms in isolation
    (matching AF3, which runs DSSP on each chain filtered out on its own), summed per
    residue. Rows are returned in chain sequence order for residues that have at least
    one observed heavy atom; unresolved residues are skipped, mirroring DSSP's chain
    breaks. Raises ``ImportError`` when biotite is unavailable and ``RuntimeError`` when
    the chain has no observed heavy atoms, both of which :func:`fraction_disordered`
    treats as an unavailable backend and degrades to ``0.0`` (upstream on-failure path).
    """

    import biotite.structure as biotite_structure

    system = structure.system
    coordinates: list[np.ndarray] = []
    elements: list[str] = []
    residue_indices: list[int] = []
    residue_names: list[str] = []
    atom_names: list[str] = []
    order: list[int] = []
    names_by_residue: dict[int, str] = {}
    for index, site in enumerate(structure.sites):
        if site.instance_id != instance_id:
            continue
        if not bool(structure.coordinate_mask[index]):
            continue
        atom = system.atom(site)
        if atom.element in HYDROGEN_ELEMENTS:
            continue
        component = system.component((instance_id, site.component_index))
        coordinates.append(np.asarray(structure.coordinates[index], dtype=np.float32))
        elements.append(atom.element.upper())
        residue_indices.append(site.component_index)
        residue_names.append(component.component_id)
        atom_names.append(site.atom_id)
        if site.component_index not in names_by_residue:
            names_by_residue[site.component_index] = component.component_id
            order.append(site.component_index)

    if not coordinates:
        raise RuntimeError("protein chain has no observed heavy atoms")

    array = biotite_structure.AtomArray(len(coordinates))
    array.coord = np.stack(coordinates, axis=0)
    array.element = np.asarray(elements, dtype="U2")
    array.res_id = np.asarray(residue_indices, dtype=np.int64)
    array.res_name = np.asarray(residue_names, dtype="U5")
    array.atom_name = np.asarray(atom_names, dtype="U6")
    array.chain_id = np.full(len(coordinates), "A", dtype="U1")
    array.hetero = np.zeros(len(coordinates), dtype=bool)

    per_atom = biotite_structure.sasa(
        array,
        probe_radius=_SASA_PROBE_RADIUS,
        point_number=_SASA_POINT_NUMBER,
        vdw_radii="ProtOr",
    )
    residue_indices_array = np.asarray(residue_indices)
    per_atom = np.where(np.isfinite(per_atom), per_atom, 0.0)

    rows: list[tuple[str, float]] = []
    for component_index in order:
        area = float(np.sum(per_atom[residue_indices_array == component_index]))
        rows.append((names_by_residue[component_index], area))
    return rows


def _protein_chain_ids(structure: Structure) -> list[str]:
    r"""Instance ids of protein polymer chains, in system order."""

    system = structure.system
    chain_ids: list[str] = []
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            chain_ids.append(instance_id)
    return chain_ids


def _chain_sequence(structure: Structure, instance_id: str) -> tuple[str, ...]:
    r"""Ordered component ids of one chain (a cache key standing in for its sequence)."""

    entity = structure.system.entities[structure.system.instances[instance_id]]
    return tuple(component.component_id for component in entity.components)


def _polymer_atoms(structure: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Observed polymer (protein / RNA / DNA) atom coords, residue ids, and chain ids."""

    system = structure.system
    coordinates: list[np.ndarray] = []
    residue_ids: list[int] = []
    chain_ids: list[str] = []
    for index, site in enumerate(structure.sites):
        if not bool(structure.coordinate_mask[index]):
            continue
        entity = system.entities[system.instances[site.instance_id]]
        if isinstance(entity, Molecule) or getattr(entity, "polymer_type", None) not in _POLYMER_TYPES:
            continue
        coordinates.append(np.asarray(structure.coordinates[index], dtype=np.float64))
        residue_ids.append(site.component_index)
        chain_ids.append(site.instance_id)
    if not coordinates:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=object)
    return (
        np.stack(coordinates, axis=0),
        np.asarray(residue_ids, dtype=np.int64),
        np.asarray(chain_ids, dtype=object),
    )


def _query_ball(coordinates: np.ndarray, cutoff_radius: float) -> list[list[int]]:
    r"""Neighbor indices within ``cutoff_radius`` per atom.

    Uses ``scipy.spatial.cKDTree`` when available (matching upstream
    ``confidences.py:167-170``); falls back to a dense NumPy pairwise search so the
    clash test still runs without SciPy.
    """

    try:
        from scipy import spatial
    except ImportError:
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.sqrt(np.sum(differences * differences, axis=-1))
        within = distances <= cutoff_radius
        return [np.nonzero(row)[0].tolist() for row in within]

    tree = spatial.cKDTree(coordinates)
    return tree.query_ball_point(coordinates, p=2.0, r=cutoff_radius)
