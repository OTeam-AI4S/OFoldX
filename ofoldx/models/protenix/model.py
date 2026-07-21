r"""Released Protenix v0.5 default and constraint structure predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Self

from ...data.molecular import (
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...modules.adaptive_norm import AdaptiveLayerNorm
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
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation

# Re-exported for the protenix package public API (``ofoldx.models.protenix.__init__``); the shared
# confidence classes live in ``ofoldx.modules.head`` but historically resolved from ``.model``.
from ...modules.head import (  # noqa: F401
    AtomConfidenceProjector,
    ConfidenceHead,
    ConfidencePairEmbedder,
    DistogramHead,
    PairConfidenceProjector,
    tm_bin_weights,
)
from ...modules.transformer import ConditionedTransformerLayer, ProtenixConditionedTransformerLayer
from ...modules.transition import Transition
from ...modules.trunk import MsaTrunk, PairformerStack
from ...modules.trunk.msa import MsaPairWeightedAveraging, OuterProductMean
from ...results import (
    FoldPrediction,
    FoldResult,
    Metric,
    PairMetric,
    Score,
)
from .config import ProtenixConfig, ProtenixConstraintConfig


class ProtenixModel(nn.Module):
    r"""
    Protenix v0.5 structure-prediction model.

    ``ProtenixConfig(constraints=False)`` is the default v0.5.0 graph.
    ``constraints=True`` adds the complete four-branch constraint pair bias from
    the v0.5.3 constraint checkpoint. The trunk output feeds the released EDM
    sampler, distogram head, and confidence head.

    Args:
        config: Model graph configuration.

    """

    config_class = ProtenixConfig

    def __init__(self, config: ProtenixConfig | None = None) -> None:
        config = ProtenixConfig() if config is None else config
        super().__init__()
        self.config = config
        self._recycling_steps = config.recycling_steps
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
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
        self.distogram_head = DistogramHead(config.pair_size, config.distogram.num_bins)
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
            # Protenix confidence flags: an extra single LayerNorm (``input_strunk_ln``), a raw-distance
            # projection channel, and PDE symmetrizes the pair STATE before the norm/projection
            # (``pde_ln(z + z.transpose(-2, -3))``). The distance-bin buffers are persistent.
            use_single_norm=True,
            use_distance_projection=True,
            symmetrize_pde_before_projection=True,
            persistent_distance_bins=True,
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
            expansion=config.template.expansion,
            dropout=config.template.dropout,
        )
        self.constraint_embedder = ProtenixConstraintEmbedder(config.pair_size) if config.constraints else None
        # Protenix ``AttentionPairBias`` (cross_attention_mode + has_s) stacks a *second* adaptive LN on the
        # already query-normed activation for the key/value path -- the double-adaLN convention. The shared
        # atom encoder/decoder build the AF3-canonical base layer, so re-cast every cross-attention atom
        # transformer onto the Protenix-lineage subclass. Inert on the token transformer (self-attention).
        for module in self.modules():
            if isinstance(module, ConditionedTransformerLayer) and module.cross_attention_mode:
                module.__class__ = ProtenixConditionedTransformerLayer
        self.reset_parameters()
        self.set_backend("auto")

    def reset_parameters(self) -> None:
        r"""
        Zero-initialize the released zero-start projections.

        The recycle projections and confidence PAE/PDE projections start at zero.
        The constraint graph also zero-initializes every linear weight, matching
        the released implementation.
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)
        if self.constraint_embedder is not None:
            self.constraint_embedder.reset_parameters()

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
        input: ProtenixInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        recycling_steps: int | None = None,
    ) -> ProtenixOutput:
        r"""Denoise caller-supplied coordinates at a caller-supplied EDM noise level."""

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
            reference_positions=input.reference_positions,
        )
        coordinates = self.coordinate_denoiser(
            self._denoise_context(
                input,
                representations,
                reference_positions=input.reference_positions,
            ).to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return self._assemble_output(input, representations, coordinates)

    @torch.inference_mode()
    def fold(
        self,
        input: ProtenixInput,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""Run released Protenix v0.5 folding and return a canonical result."""

        num_samples = 5 if num_samples is None else num_samples
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        num_sampling_steps = self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        if num_sampling_steps < 1:
            raise ValueError("num_sampling_steps must be positive")

        reference_positions = self._augment_reference_positions(input, generator=generator)
        msa_indices = tuple(
            self._sample_msa_indices(input.msa.shape[-2], input.msa.device, generator=generator)
            for _ in range(recycling_steps)
        )
        output = self._sample(
            input,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
            reference_positions=reference_positions,
            msa_indices=msa_indices,
        )

        atom_count = len(input.atom_sites)
        token_count = len(input.token_sites)
        complete_sites = [site for site, atom in input.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray(
            [complete_index[site] for site in input.atom_sites],
            dtype=np.intp,
        )

        coordinates = output.coordinates[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        plddt_logits = output.plddt_logits[0, :, :atom_count].float()
        pae_logits = output.pae_logits[0, :, :token_count, :token_count].float()
        pde_logits = output.pde_logits[0, :, :token_count, :token_count].float()
        resolved_logits = output.resolved_logits[0, :, :atom_count].float()
        distogram_logits = output.distogram_logits[0, :token_count, :token_count].float()

        atom_plddt_unit = _protenix_bin_expectation(plddt_logits, 0.0, 1.0)
        atom_plddt = 100.0 * atom_plddt_unit
        pae, pae_probability = _protenix_bin_expectation(
            pae_logits,
            0.0,
            32.0,
            return_probability=True,
        )
        pde = _protenix_bin_expectation(pde_logits, 0.0, 32.0)
        resolved_probability = resolved_logits.softmax(dim=-1)[..., 1]
        contact_probability = _protenix_contact_probability(distogram_logits)
        token_has_frame = _protenix_token_has_frame(
            reference_positions,
            reference_mask=input.reference_mask,
            reference_space_id=input.reference_space_id,
            token_to_rep_atom=input.token_to_rep_atom,
            atom_sites=input.atom_sites,
            token_sites=input.token_sites,
        )
        token_chain = input.chain_index[0, :token_count].to(device=pae_logits.device, dtype=torch.long)
        atom_to_token = input.atom_to_token[0, :atom_count].to(device=pae_logits.device, dtype=torch.long)
        atom_is_polymer = torch.tensor(
            [
                type(input.system.entities[input.system.instances[site.instance_id]]) is PolymerSequence
                for site in input.atom_sites
            ],
            dtype=torch.bool,
            device=pae_logits.device,
        )
        ptm, iptm, chain_ptm, chain_iptm, chain_pair_iptm, chain_pair_iptm_global = _protenix_tm_scores(
            pae_probability,
            token_has_frame=token_has_frame,
            token_chain=token_chain,
            atom_to_token=atom_to_token,
            atom_is_polymer=atom_is_polymer,
        )
        gpde = (pde * contact_probability).sum(dim=(-1, -2)) / contact_probability.sum()
        chain_gpde, chain_pair_gpde = _protenix_chain_gpde(
            pde,
            contact_probability=contact_probability,
            token_chain=token_chain,
        )
        chain_plddt, chain_pair_plddt = _protenix_chain_plddt(
            atom_plddt_unit,
            token_chain=token_chain,
            atom_to_token=atom_to_token,
        )
        has_clash = _protenix_has_clash(
            output.coordinates[0, :, :atom_count].float(),
            atom_to_token=atom_to_token,
            token_chain=token_chain,
            atom_is_polymer=atom_is_polymer,
        )
        disorder = torch.zeros_like(ptm)
        ranking_score = 0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100.0 * has_clash.to(ptm.dtype)

        contact_numpy = contact_probability.cpu().numpy()
        token_instance_ids = tuple(
            site.instance_id if type(site) is AtomSite else site[0] for site in input.token_sites
        )
        token_chain_numpy = token_chain.cpu().numpy()
        chain_axis = tuple(
            token_instance_ids[int(np.argmax(token_chain_numpy == value))] for value in np.unique(token_chain_numpy)
        )
        predictions: list[FoldPrediction] = []
        for sample_index in range(num_samples):
            complete_coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
            coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
            complete_coordinates[model_to_complete] = coordinates[sample_index]
            coordinate_mask[model_to_complete] = True
            structure = Structure(
                system=input.system,
                sites=complete_sites,
                coordinates=complete_coordinates,
                coordinate_mask=coordinate_mask,
            )
            sample_plddt = atom_plddt[sample_index].cpu().numpy()
            sample_pae = pae[sample_index].cpu().numpy()
            sample_pde = pde[sample_index].cpu().numpy()
            sample_resolved_probability = resolved_probability[sample_index].cpu().numpy()
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores={
                        "plddt": Score(float(atom_plddt[sample_index].mean()), preferred="higher"),
                        "gpde": Score(float(gpde[sample_index]), unit="angstrom", preferred="lower"),
                        "ptm": Score(float(ptm[sample_index]), preferred="higher"),
                        "iptm": Score(float(iptm[sample_index]), preferred="higher"),
                        "has_clash": Score(float(has_clash[sample_index]), preferred="lower"),
                        "disorder": Score(float(disorder[sample_index]), preferred="higher"),
                        "ranking_score": Score(float(ranking_score[sample_index]), preferred="higher"),
                    },
                    metrics={
                        "plddt": Metric(values=sample_plddt, axis=input.atom_sites),
                        "resolved_probability": Metric(
                            values=sample_resolved_probability,
                            axis=input.atom_sites,
                        ),
                        "chain_plddt": Metric(
                            values=chain_plddt[sample_index].cpu().numpy(),
                            axis=chain_axis,
                        ),
                        "chain_ptm": Metric(
                            values=chain_ptm[sample_index].cpu().numpy(),
                            axis=chain_axis,
                        ),
                        "chain_iptm": Metric(
                            values=chain_iptm[sample_index].cpu().numpy(),
                            axis=chain_axis,
                        ),
                        "chain_gpde": Metric(
                            values=chain_gpde[sample_index].cpu().numpy(),
                            axis=chain_axis,
                            unit="angstrom",
                        ),
                    },
                    pair_metrics={
                        "pae": PairMetric(
                            values=sample_pae,
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=sample_pde,
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                            unit="angstrom",
                        ),
                        "contact_probs": PairMetric(
                            values=contact_numpy,
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                        ),
                        "chain_pair_iptm": PairMetric(
                            values=chain_pair_iptm[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                        "chain_pair_iptm_global": PairMetric(
                            values=chain_pair_iptm_global[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                        "chain_pair_gpde": PairMetric(
                            values=chain_pair_gpde[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_plddt": PairMetric(
                            values=chain_pair_plddt[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                    },
                )
            )

        return FoldResult(predictions=tuple(predictions))

    @torch.inference_mode()
    def _sample(
        self,
        input: ProtenixInput,
        *,
        recycling_steps: int,
        num_samples: int,
        num_sampling_steps: int,
        generator: torch.Generator | None,
        reference_positions: Tensor,
        msa_indices: tuple[Tensor, ...],
    ) -> ProtenixOutput:
        r"""Run the released Protenix trunk, sampler, and prediction heads."""

        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
            reference_positions=reference_positions,
            msa_indices=msa_indices,
        )
        coordinates = self._predict_coords(
            input,
            representations=representations,
            reference_positions=reference_positions,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )
        return self._assemble_output(input, representations, coordinates)

    def _assemble_output(
        self,
        input: ProtenixInput,
        representations: tuple[Tensor, Tensor, Tensor, RelativePositionBuckets],
        coordinates: Tensor,
    ) -> ProtenixOutput:
        r"""Compute the released distogram and confidence predictions."""

        token_embedding, single, pair, _ = representations
        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            single,
            pair=pair,
            token_embedding=token_embedding,
            pred_coords=coordinates,
            token_to_rep_atom=input.token_to_rep_atom,
            atom_to_token=input.atom_to_token,
            atom_token_index=input.atom_token_index,
        )
        return ProtenixOutput(
            coordinates=coordinates,
            distogram_logits=self.distogram_head(pair),
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
        )

    def _denoise_context(
        self,
        input: ProtenixInput,
        representations: tuple[Tensor, Tensor, Tensor, RelativePositionBuckets],
        *,
        reference_positions: Tensor,
    ) -> CoordinateDenoiseContext:
        r"""Assemble conditioning shared by forward and the sampler."""

        token_embedding, single, pair, relative_features = representations
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                reference_positions,
                input.reference_space_id,
            ),
        )

    def _run_trunk(
        self,
        input: ProtenixInput,
        *,
        recycling_steps: int,
        reference_positions: Tensor,
        msa_indices: tuple[Tensor, ...] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, RelativePositionBuckets]:
        r"""Embed the inputs and run the released recycling trunk."""

        token_embedding = self.input_embedder(
            input.atom_features,
            token_features=input.token_features,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_to_token=input.atom_to_token,
            num_tokens=input.token_features.shape[-2],
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=self.input_embedder.build_local_geometry(
                reference_positions,
                input.reference_space_id,
            ),
        )
        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            dtype=single_init.dtype,
        )
        pair_init = self._initial_pair(single_init, input, relative_features)
        pair_mask = pair_mask_from_token_mask(input.token_mask)
        single, pair = self._run_recycling(
            input,
            token_embedding,
            single_init,
            pair_init,
            pair_mask,
            recycling_steps,
            msa_indices=msa_indices,
        )
        return token_embedding, single, pair, relative_features

    def _run_recycling(
        self,
        input: ProtenixInput,
        token_embedding: Tensor,
        single_init: Tensor,
        pair_init: Tensor,
        pair_mask: Tensor,
        recycling_steps: int,
        *,
        msa_indices: tuple[Tensor, ...] | None,
    ) -> tuple[Tensor, Tensor]:
        r"""Run the released recycling loop."""

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(recycling_steps):
            pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
            indices = None if msa_indices is None else msa_indices[cycle]
            _, pair = self.msa_trunk(
                self._msa_features(input, indices),
                pair,
                token_embedding,
                pair_mask=pair_mask,
            )
            single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
            single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        return single, pair

    @staticmethod
    def _msa_features(input: ProtenixInput, indices: Tensor | None) -> Tensor:
        r"""Build the released 32+1+1 MSA feature vector for selected rows."""

        if indices is None:
            msa = input.msa
            has_deletion = input.has_deletion
            deletion_value = input.deletion_value
        else:
            msa = torch.index_select(input.msa, -2, indices)
            has_deletion = torch.index_select(input.has_deletion, -2, indices)
            deletion_value = torch.index_select(input.deletion_value, -2, indices)
        one_hot = F.one_hot(msa.long(), num_classes=32).to(dtype=input.token_features.dtype)
        return torch.cat(
            (
                one_hot,
                has_deletion.to(one_hot).unsqueeze(-1),
                deletion_value.to(one_hot).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _initial_pair(
        self,
        single: Tensor,
        input: ProtenixInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        r"""Build the initial pair representation and optional constraint bias."""

        left = self.pair_left_projection(single).unsqueeze(-2)
        right = self.pair_right_projection(single).unsqueeze(-3)
        token_bonds = input.token_bonds.to(device=single.device, dtype=single.dtype)
        token_bonds = torch.broadcast_to(token_bonds, (*single.shape[:-1], single.shape[-2]))
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds.unsqueeze(-1))
        if self.constraint_embedder is not None:
            pair = pair + self.constraint_embedder(
                input.pocket,
                input.contact,
                input.contact_atom,
                input.substructure,
            )
        pair_mask = pair_mask_from_token_mask(input.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)

    def _predict_coords(
        self,
        input: ProtenixInput,
        *,
        representations: tuple[Tensor, Tensor, Tensor, RelativePositionBuckets],
        reference_positions: Tensor,
        num_samples: int,
        num_sampling_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Run released sequential EDM sampling."""

        return self.coordinate_sampler(
            self.coordinate_denoiser,
            context=self._denoise_context(
                input,
                representations,
                reference_positions=reference_positions,
            ),
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=1,
            generator=generator,
        )

    def _sample_msa_indices(
        self,
        num_rows: int,
        device: torch.device,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Draw one recycle MSA subset without privileging the query row."""

        if self.config.constraints:
            count = int(
                torch.randint(
                    1,
                    num_rows + 1,
                    (),
                    device=device,
                    generator=generator,
                )
            )
            count = min(count, 16384)
        else:
            count = min(num_rows, 2048)
        return torch.randperm(num_rows, device=device, generator=generator)[:count]

    @staticmethod
    def _augment_reference_positions(
        input: ProtenixInput,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Apply one independent released rigid augmentation per reference space."""

        positions = input.reference_positions
        augmented = torch.zeros_like(positions)
        for batch_index in range(positions.shape[0]):
            for space_id in torch.unique(input.reference_space_id[batch_index]):
                in_space = input.reference_space_id[batch_index] == space_id
                valid = in_space & input.reference_mask[batch_index].bool()
                weight = valid.to(dtype=positions.dtype)
                center = (positions[batch_index] * weight.unsqueeze(-1)).sum(dim=-2) / weight.sum().clamp_min(1)
                translation = torch.empty(
                    3,
                    device=positions.device,
                    dtype=positions.dtype,
                ).uniform_(-1.0, 1.0, generator=generator)
                rotation = uniform_random_rotation(
                    device=positions.device,
                    dtype=positions.dtype,
                    generator=generator,
                )[0]
                transformed = torch.matmul(
                    positions[batch_index] - center + translation,
                    rotation.transpose(-1, -2),
                )
                augmented[batch_index] = torch.where(
                    valid.unsqueeze(-1),
                    transformed,
                    augmented[batch_index],
                )
        return augmented


class ProtenixConstraintModel(ProtenixModel):
    r"""Released ``protenix_base_constraint_v0.5.0`` graph (four-branch constraint pair bias).

    Adds nothing to the parity-frozen base graph beyond selecting the constraint
    configuration; the ``constraint_embedder`` submodule is built conditionally by
    ``ProtenixModel.__init__`` when ``config.constraints`` is ``True``. No numeric
    method is overridden, so parity is unaffected.
    """

    config_class = ProtenixConstraintConfig

    def __init__(self, config: ProtenixConstraintConfig | None = None) -> None:
        super().__init__(ProtenixConstraintConfig() if config is None else config)


def _protenix_bin_centers(
    tensor: Tensor,
    minimum: float,
    maximum: float,
    num_bins: int,
) -> Tensor:
    r"""Return the released evenly spaced confidence-bin centers."""

    width = (maximum - minimum) / num_bins
    return (
        torch.linspace(
            minimum,
            maximum - width,
            num_bins,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        + 0.5 * width
    )


def _protenix_bin_expectation(
    logits: Tensor,
    minimum: float,
    maximum: float,
    *,
    return_probability: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Decode released pLDDT/PAE/PDE categorical logits."""

    probability = logits.softmax(dim=-1)
    values = probability @ _protenix_bin_centers(logits, minimum, maximum, logits.shape[-1])
    if return_probability:
        return values, probability
    return values


def _protenix_contact_probability(distogram_logits: Tensor) -> Tensor:
    r"""Decode the released v0.5 probability of distance below 8 Å."""

    centers = _protenix_bin_centers(
        distogram_logits,
        2.3125,
        21.6875,
        distogram_logits.shape[-1],
    )
    return distogram_logits.softmax(dim=-1)[..., centers < 8.0].sum(dim=-1)


def _protenix_tm_scores(
    pae_probability: Tensor,
    *,
    token_has_frame: Tensor,
    token_chain: Tensor,
    atom_to_token: Tensor,
    atom_is_polymer: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Compute the released v0.5 global and chain TM summaries."""

    token_has_frame = token_has_frame.to(device=pae_probability.device, dtype=torch.bool)
    token_chain = token_chain.to(device=pae_probability.device, dtype=torch.long)

    def calculate(probability: Tensor, has_frame: Tensor, chain: Tensor) -> tuple[Tensor, Tensor]:
        if not bool(has_frame.any()):
            zeros = probability.new_zeros(probability.shape[:-3])
            return zeros, zeros
        num_tokens = has_frame.shape[0]
        centers = _protenix_bin_centers(probability, 0.0, 32.0, probability.shape[-1])
        token_pair_tm = (probability * tm_bin_weights(centers, num_tokens)).sum(dim=-1)
        ptm_score = token_pair_tm.mean(dim=-1)[..., has_frame].max(dim=-1).values
        different_chain = chain[None, :] != chain[:, None]
        iptm_per_alignment = (token_pair_tm * different_chain).sum(dim=-1) / (
            torch.finfo(token_pair_tm.dtype).tiny + different_chain.sum(dim=-1)
        )
        iptm_score = iptm_per_alignment[..., has_frame].max(dim=-1).values
        return ptm_score, iptm_score

    ptm, iptm = calculate(pae_probability, token_has_frame, token_chain)
    chains = torch.unique(token_chain)
    chain_masks = tuple(token_chain == chain for chain in chains)
    batch_shape = pae_probability.shape[:-3]
    chain_ptm = pae_probability.new_zeros(batch_shape + (len(chains),))
    chain_pair_iptm = pae_probability.new_zeros(batch_shape + (len(chains), len(chains)))
    for first_index, first_mask in enumerate(chain_masks):
        first_probability = pae_probability[..., first_mask, :, :][..., :, first_mask, :]
        chain_ptm[..., first_index] = calculate(
            first_probability,
            token_has_frame[first_mask],
            token_chain[first_mask],
        )[0]
        for second_index in range(first_index + 1, len(chains)):
            pair_mask = first_mask | chain_masks[second_index]
            pair_probability = pae_probability[..., pair_mask, :, :][..., :, pair_mask, :]
            pair_iptm = calculate(
                pair_probability,
                token_has_frame[pair_mask],
                token_chain[pair_mask],
            )[1]
            chain_pair_iptm[..., first_index, second_index] = pair_iptm
            chain_pair_iptm[..., second_index, first_index] = pair_iptm

    atom_to_token = atom_to_token.to(device=pae_probability.device, dtype=torch.long)
    atom_is_polymer = atom_is_polymer.to(device=pae_probability.device, dtype=torch.bool)
    token_is_ligand = (
        torch.zeros_like(token_chain).scatter_add(
            0,
            atom_to_token,
            (~atom_is_polymer).long(),
        )
        > 0
    )
    chain_has_frame = tuple(bool((mask & token_has_frame).any()) for mask in chain_masks)
    chain_iptm = pae_probability.new_zeros(batch_shape + (len(chains),))
    for chain_index in range(len(chains)):
        values = [
            chain_pair_iptm[..., first_index, second_index]
            for first_index in range(len(chains))
            for second_index in range(len(chains))
            if (first_index == chain_index or second_index == chain_index)
            and first_index != second_index
            and chain_has_frame[first_index]
        ]
        if values:
            chain_iptm[..., chain_index] = torch.stack(values, dim=-1).mean(dim=-1)

    chain_is_ligand = tuple(token_is_ligand[mask].sum() >= mask.sum() // 2 for mask in chain_masks)
    chain_pair_iptm_global = torch.zeros_like(chain_pair_iptm)
    for first_index in range(len(chains)):
        for second_index in range(len(chains)):
            if first_index == second_index:
                continue
            if chain_is_ligand[first_index]:
                value = chain_iptm[..., first_index]
            elif chain_is_ligand[second_index]:
                value = chain_iptm[..., second_index]
            else:
                value = 0.5 * (chain_iptm[..., first_index] + chain_iptm[..., second_index])
            chain_pair_iptm_global[..., first_index, second_index] = value
    return ptm, iptm, chain_ptm, chain_iptm, chain_pair_iptm, chain_pair_iptm_global


def _protenix_chain_gpde(
    token_pair_pde: Tensor,
    *,
    contact_probability: Tensor,
    token_chain: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute the released v0.5 chain and chain-pair gPDE summaries."""

    token_chain = token_chain.to(device=token_pair_pde.device, dtype=torch.long)
    contact_probability = contact_probability.to(device=token_pair_pde.device)
    chains = torch.unique(token_chain)
    chain_gpde = token_pair_pde.new_zeros(token_pair_pde.shape[:-2] + (len(chains),))
    chain_pair_gpde = token_pair_pde.new_zeros(token_pair_pde.shape[:-2] + (len(chains), len(chains)))

    def calculate(row_mask: Tensor, column_mask: Tensor) -> Tensor:
        contact = contact_probability[..., row_mask, :][..., column_mask]
        pde = token_pair_pde[..., row_mask, :][..., column_mask]
        return (pde * contact).sum(dim=(-1, -2)) / (contact.sum(dim=(-1, -2)) + torch.finfo(contact.dtype).tiny)

    chain_masks = tuple(token_chain == chain for chain in chains)
    for first_index, first_mask in enumerate(chain_masks):
        chain_gpde[..., first_index] = calculate(first_mask, first_mask)
        for second_index in range(first_index + 1, len(chains)):
            value = calculate(first_mask, chain_masks[second_index])
            chain_pair_gpde[..., first_index, second_index] = value
            chain_pair_gpde[..., second_index, first_index] = value
    return chain_gpde, chain_pair_gpde


def _protenix_chain_plddt(
    atom_plddt: Tensor,
    *,
    token_chain: Tensor,
    atom_to_token: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute the released v0.5 chain and chain-pair pLDDT summaries."""

    token_chain = token_chain.to(device=atom_plddt.device, dtype=torch.long)
    atom_to_token = atom_to_token.to(device=atom_plddt.device, dtype=torch.long)
    chains = torch.unique(token_chain)
    atom_chain = token_chain[atom_to_token]
    chain_plddt = atom_plddt.new_zeros(atom_plddt.shape[:-1] + (len(chains),))
    chain_pair_plddt = atom_plddt.new_zeros(atom_plddt.shape[:-1] + (len(chains), len(chains)))
    for first_index, first_chain in enumerate(chains):
        first_mask = atom_chain == first_chain
        chain_plddt[..., first_index] = atom_plddt[..., first_mask].mean(dim=-1)
        for second_index, second_chain in enumerate(chains):
            if first_index == second_index:
                continue
            pair_mask = first_mask | (atom_chain == second_chain)
            chain_pair_plddt[..., first_index, second_index] = atom_plddt[..., pair_mask].mean(dim=-1)
    return chain_plddt, chain_pair_plddt


def _protenix_token_has_frame(
    reference_positions: Tensor,
    *,
    reference_mask: Tensor,
    reference_space_id: Tensor,
    token_to_rep_atom: Tensor,
    atom_sites: tuple[AtomSite, ...],
    token_sites: tuple[AtomSite | tuple[str, int], ...],
) -> Tensor:
    r"""Reproduce v0.5 token-frame validity from the reference conformer."""

    atom_count = len(atom_sites)
    reference_positions = reference_positions[0, :atom_count].float()
    reference_mask = reference_mask[0, :atom_count].bool()
    reference_space = reference_space_id[0, :atom_count]
    representative_atom = token_to_rep_atom[0, : len(token_sites)].long()
    has_frame = torch.zeros(len(token_sites), dtype=torch.bool, device=reference_positions.device)
    for token_index, token_site in enumerate(token_sites):
        if type(token_site) is tuple:
            has_frame[token_index] = True
            continue
        center = representative_atom[token_index]
        same_reference = torch.where(reference_space == reference_space[center])[0]
        neighbors = same_reference[same_reference != center]
        if neighbors.numel() < 2:
            continue
        squared_distance = ((reference_positions[neighbors] - reference_positions[center]) ** 2).sum(dim=-1)
        nearest = neighbors[torch.argsort(squared_distance, stable=True)[:2]]
        frame_atoms = torch.cat((nearest[:1], center.reshape(1), nearest[1:]))
        if not bool(reference_mask[frame_atoms].all()):
            continue
        first = reference_positions[nearest[0]] - reference_positions[center]
        second = reference_positions[nearest[1]] - reference_positions[center]
        norm_product = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
        if bool(torch.isclose(norm_product, norm_product.new_zeros(()))):
            continue
        angle = torch.rad2deg(torch.acos(torch.dot(first, second).div(norm_product).clamp(-1.0, 1.0)))
        has_frame[token_index] = (angle > 25.0) & (angle < 155.0)
    return has_frame


def _protenix_has_clash(
    coordinates: Tensor,
    *,
    atom_to_token: Tensor,
    token_chain: Tensor,
    atom_is_polymer: Tensor,
) -> Tensor:
    r"""Apply the released AF3-style inter-polymer-chain clash rule."""

    atom_to_token = atom_to_token.to(device=coordinates.device, dtype=torch.long)
    token_chain = token_chain.to(device=coordinates.device, dtype=torch.long)
    atom_chain = token_chain[atom_to_token]
    atom_is_polymer = atom_is_polymer.to(device=coordinates.device, dtype=torch.bool)
    polymer_chains = torch.unique(atom_chain[atom_is_polymer])
    has_clash = torch.zeros(coordinates.shape[0], dtype=torch.bool, device=coordinates.device)
    for first_index, first_chain in enumerate(polymer_chains):
        first_mask = atom_is_polymer & (atom_chain == first_chain)
        for second_chain in polymer_chains[first_index + 1 :]:
            second_mask = atom_is_polymer & (atom_chain == second_chain)
            distances = torch.cdist(coordinates[:, first_mask], coordinates[:, second_mask])
            clashes = (distances < 1.1).sum(dim=(-1, -2))
            shorter_chain = min(int(first_mask.sum()), int(second_mask.sum()))
            has_clash |= (clashes > 100) | (clashes / shorter_chain > 0.5)
    return has_clash


class InputEmbedder(nn.Module):
    r"""
    Build token input embeddings from packed atom and token features.

    Runs the atom-to-token encoder to produce the leading token-single channels, then concatenates
    the trailing token-feature tail so the output width matches the token feature contract.

    Args:
        single_size: Width of the atom-derived token-single channel.
        atom_feature_size: Width of the per-atom input features consumed by the atom encoder.
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
        Return token embeddings concatenating atom-derived channels and the token-feature tail.

        Args:
            atom_features: Per-atom input features ``[..., N_atom, atom_feature_size]``.
            reference_positions: Reference conformer atom positions ``[..., N_atom, 3]``.
            reference_charges: Per-atom reference charges ``[..., N_atom, 1]`` (arcsinh-compressed here).
            reference_mask: Per-atom reference validity mask ``[..., N_atom]``.
            token_features: Packed per-token features ``[..., N_token, token_feature_size]``.
            atom_to_token: Token index per atom ``[..., N_atom]``.
            num_tokens: Number of tokens.
            atom_mask: Optional per-atom validity mask ``[..., N_atom]``.
            token_mask: Optional per-token validity mask ``[..., N_token]``; zeroes masked rows.
            local_geometry: Optional precomputed local atom geometry.

        Returns:
            Token embeddings ``[..., N_token, single_size + tail]`` with the token-feature tail appended.
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
        Build local atom geometry using the input atom encoder layout.

        Args:
            reference_positions: Reference conformer atom positions ``[..., N_atom, 3]``.
            reference_space_id: Per-atom reference-space grouping id ``[..., N_atom]``.

        Returns:
            Local atom geometry for the input atom encoder.
        """

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)


class ProtenixConstraintEmbedder(nn.Module):
    r"""Complete four-branch pair-bias embedder from the constraint checkpoint."""

    def __init__(self, pair_size: int = 128) -> None:
        super().__init__()
        self.pocket_z_embedder = nn.Linear(1, pair_size, bias=False)
        self.contact_z_embedder = nn.Linear(2, pair_size, bias=False)
        self.contact_atom_z_embedder = nn.Linear(2, pair_size, bias=False)
        self.substructure_z_embedder = ProtenixConstraintSubstructureEmbedder(pair_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Zero every released linear weight without changing PyTorch biases."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)

    def forward(
        self,
        pocket: Tensor,
        contact: Tensor,
        contact_atom: Tensor,
        substructure: Tensor,
    ) -> Tensor:
        r"""Project and sum all four mandatory constraint features."""

        return (
            self.pocket_z_embedder(pocket)
            + self.contact_z_embedder(contact)
            + self.contact_atom_z_embedder(contact_atom)
            + self.substructure_z_embedder(substructure)
        )


class ProtenixConstraintSubstructureEmbedder(nn.Module):
    r"""Released transformer over the flattened token-pair substructure grid."""

    def __init__(self, pair_size: int = 128) -> None:
        super().__init__()
        hidden_size = 128
        self.input_proj = nn.Linear(4, hidden_size, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_proj = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(self, substructure: Tensor) -> Tensor:
        r"""Embed a four-channel token-pair substructure feature."""

        shape = substructure.shape
        flattened = substructure.reshape(-1, shape[-3] * shape[-2], shape[-1])
        hidden = self.transformer(self.input_proj(flattened))
        return self.output_proj(hidden).reshape(*shape[:-1], -1)


# =====================================================================================
# Typed tensor and semantic boundary of the released Protenix v0.5 graph.
# (folded verbatim from the former ``protenix/contract.py``)
# =====================================================================================


def expected_plddt(logits: Tensor) -> Tensor:
    r"""Expected pLDDT on the conventional 0--100 scale from per-bin logits."""

    centers = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype) + 0.5
    return (logits.softmax(dim=-1) * centers * (100.0 / logits.shape[-1])).sum(dim=-1)


# Model-type entries that exist upstream (`configs/configs_model_type.py`) but whose weights are
# NOT on disk this wave. They are deliberately NOT implemented, faked, or registered. Each is a SIZE
# or internal-graph variant of the single `ofoldx.protenix` architecture: same input contract, same
# public Model/Processor/Result classes, so each lands as a CONFIG PRESET of `ProtenixConfig` selected
# through the checkpoint's `model.json` (§5.4) -- exactly like ProteinMPNN's `ca_only` graph -- and is
# published as its own future checkpoint repo (§7.1), NOT as a new `(Model, Processor)` registration.
# `mini`/`tiny` shrink the trunk dims, `esm`/`ism` add an internal ESM2-3B feature branch, and
# `protenix-v2` widens `c_z` to 256 and enables 2-block templates; every one of these is expressible as
# `ProtenixConfig` field values. Only the constraint checkpoint, which changes the INPUT contract, is a
# second architecture id (§3.7.2); a size or internal-branch change alone never earns one.
_CONFIG_PRESET_CHECKPOINTS: tuple[str, ...] = (
    "protenix_mini_default_v0.5.0",
    "protenix_tiny_default_v0.5.0",
    "protenix_mini_esm_v0.5.0",
    "protenix_mini_ism_v0.5.0",
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix-v2",
)


@dataclass(kw_only=True, eq=False, repr=False)
class ProtenixInput:
    r"""Named compute tensors and canonical identities consumed by Protenix.

    ``msa``, ``has_deletion``, and ``deletion_value`` retain every deterministic
    MSA row. ``fold`` samples rows independently for each recycle; ``forward``
    consumes all rows without drawing randomness. Full-MSA profile and deletion
    means are already present in ``token_features``.

    ``pocket``, ``contact``, ``contact_atom``, and ``substructure`` are the four
    constraint pair-bias channels of the v0.5.3 checkpoint (all zero for the default
    v0.5.0 graph, whose model ignores them). ``system``, ``atom_sites``, and
    ``token_sites`` carry the canonical identities used to reconstruct the result.
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
    msa: Tensor
    has_deletion: Tensor
    deletion_value: Tensor
    pocket: Tensor
    contact: Tensor
    contact_atom: Tensor
    substructure: Tensor
    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | ComponentReference, ...]


@dataclass(eq=False, repr=False)
class ProtenixOutput:
    r"""Predictions computed by one Protenix denoising call."""

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


# Family-owned spatial conditioning records for the Protenix v0.5.3 constraint graph.
#
# These records are owned by the Protenix family (DESIGN §3.7.2 negative contract: no
# shared cross-family ``Condition`` base). Each maps onto exactly one trained constraint
# channel of ``protenix_base_constraint_v0.5.0.pt`` (release ``a0f95f6``):
#
# * :class:`ProtenixAtomContact` -> the ``contact_atom`` branch. Upstream swapped the two
#   atom-contact channels between training and inference, so only channel 0 was ever
#   trained and it learned the *maximum* distance (see ``UPSTREAM_BUGS.md``). The record
#   therefore carries only ``max_distance``; there is no min-distance channel to fill.
# * :class:`ProtenixTokenContact` -> the ``contact`` branch, written as ``[0, max]``.
# * :class:`ProtenixPocket` -> the single ``pocket`` branch, a directed binder->residue fill.


@dataclass(frozen=True)
class ProtenixAtomContact:
    r"""Upper-bound distance between two exact atoms (the ``contact_atom`` channel).

    Only ``max_distance`` is representable: the released checkpoint trained the atom
    channel exclusively on the maximum distance, so no independent minimum exists.
    """

    atom_a: AtomSite
    atom_b: AtomSite
    max_distance: float

    def __post_init__(self) -> None:
        if not isinstance(self.atom_a, AtomSite) or not isinstance(self.atom_b, AtomSite):
            raise TypeError("ProtenixAtomContact endpoints must both be exact AtomSite atoms")
        if self.atom_a == self.atom_b:
            raise ValueError("a Protenix atom contact needs two distinct atoms")
        if self.max_distance <= 0:
            raise ValueError(
                "ProtenixAtomContact max_distance must be positive; [0, 0] is indistinguishable from no contact"
            )


@dataclass(frozen=True)
class ProtenixTokenContact:
    r"""Upper-bound token-centre distance (the ``contact`` channel), written ``[0, max]``.

    Each endpoint is either a residue (:data:`ComponentReference`) or a single
    :class:`AtomSite`; a mixed atom/residue pair is allowed. Two ``AtomSite`` endpoints
    are rejected: that pair is definitionally the atom channel
    (use :class:`ProtenixAtomContact`).
    """

    endpoint_a: ComponentReference | AtomSite
    endpoint_b: ComponentReference | AtomSite
    max_distance: float

    def __post_init__(self) -> None:
        if isinstance(self.endpoint_a, AtomSite) and isinstance(self.endpoint_b, AtomSite):
            raise ValueError(
                "a two-atom Protenix contact is the atom channel; use ProtenixAtomContact instead of "
                "ProtenixTokenContact"
            )
        if self.max_distance <= 0:
            raise ValueError(
                "ProtenixTokenContact max_distance must be positive; [0, 0] is indistinguishable from no contact"
            )


@dataclass(frozen=True)
class ProtenixPocket:
    r"""A single directed pocket: one binder chain contacting one or more residues.

    The whole ``binder`` chain is written against each residue in ``contact_residues``
    at one shared ``max_distance``. Zero-or-one pocket is representable per input.
    """

    binder: str
    contact_residues: tuple[ComponentReference, ...]
    max_distance: float

    def __post_init__(self) -> None:
        if not self.contact_residues:
            raise ValueError("ProtenixPocket requires at least one contact residue")
        if self.max_distance <= 0:
            raise ValueError("ProtenixPocket max_distance must be positive")
        for residue in self.contact_residues:
            if isinstance(residue, AtomSite):
                raise ValueError("ProtenixPocket contact residues must be residue references, not AtomSite atoms")
            if residue[0] == self.binder:
                raise ValueError("a Protenix pocket residue cannot belong to the binder chain")
