r"""
Protenix default and constraint structure predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, overload

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.molecular import (
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import (
    AtomToTokenEncoder,
    LocalAtomGeometry,
)
from ...modules.backend import LayerNorm
from ...modules.backend.attention import projected_attention
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import CoordinateDenoiseContext, CoordinateEdmSampler, EdmScheduler
from ...modules.embedding import (
    RelativePositionBuckets,
    RelativePositionEmbedder,
    TemplateEmbedder,
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask
from ...modules.head import ConfidenceHead, DistogramHead, expected_plddt, tm_bin_weights
from ...modules.trunk import MsaTrunk, PairformerStack
from ...results import (
    FoldPrediction,
    FoldResult,
    Metric,
    PairMetric,
    Score,
)
from .config import ProtenixConfig, ProtenixConstraintConfig


class ProtenixModel(Model):
    r"""
    Predict biomolecular complex structures with Protenix.

    Accepts [`ProtenixInput`][ofoldx.models.protenix.model.ProtenixInput] and returns
    [`ProtenixOutput`][ofoldx.models.protenix.model.ProtenixOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = ProtenixConfig
    config: ProtenixConfig

    def __init__(self, config: ProtenixConfig | None = None) -> None:
        config = ProtenixConfig() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self.input_embeddings = ProtenixInputEmbeddings(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
        )
        self.constraint_embeddings = ProtenixConstraintEmbeddings(config.pair_size) if config.constraints else None
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.template_embedder = TemplateEmbedder(
            pair_size=config.pair_size,
            hidden_size=config.template.hidden_size,
            num_blocks=config.template.num_blocks,
            vocab_size=config.template.vocab_size,
            num_distogram_bins=config.template.num_distogram_bins,
            expansion=config.template.expansion,
            dropout=config.template.dropout,
        )
        self.msa_trunk = MsaTrunk(
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
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
            cross_attention_key_source="normalized",
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
            # This confidence head normalizes the single representation, projects
            # raw distances, and symmetrizes pair states before PDE projection.
            use_single_norm=True,
            use_distance_projection=True,
            symmetrize_pde_before_projection=True,
            persistent_distance_bins=True,
        )
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Zero-initialize recycling, confidence, and constraint projections.

        Returns:
            ``None``.
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)
        if self.constraint_embeddings is not None:
            self.constraint_embeddings.reset_parameters()

    def forward(
        self,
        input: ProtenixInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        recycling_steps: int | None = None,
    ) -> ProtenixOutput:
        r"""
        Denoise atom coordinates at specified EDM noise levels.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            noisy_coordinates: Noisy coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: EDM noise standard deviations with shape ``(B, S)`` in
                angstroms.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.

        Returns:
            Denoised coordinates and confidence tensors with the shapes described
            by [`ProtenixOutput`][]. No diffusion sampling is performed.

        Raises:
            ValueError: If ``recycling_steps`` is not positive.
        """

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
        r"""
        Predict structures for one molecular assembly with Protenix.

        Args:
            input: Processor output for one assembly, with batch size ``B = 1`` and
                molecular identities aligned to its atom and token axes.
            num_samples: Number of coordinate samples. ``None`` uses five.
            generator: Optional generator for MSA subsampling, initial coordinates,
                stochastic churn, and per-step rigid augmentations.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.

        Returns:
            One generation-ordered prediction per sample. Structure, PAE, PDE, and
            gPDE values are in angstroms; pLDDT is on the ``0--100`` scale. Atom
            metrics follow ``input.atom_sites`` and token-pair metrics follow
            ``input.token_sites``.

        Raises:
            ValueError: If a count is not positive.
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

        reference_positions = input.reference_positions
        msa_indices = tuple(
            self._sample_msa_indices(input.msa_tokens.shape[-2], input.msa_tokens.device, generator=generator)
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
                isinstance(
                    input.system.entities[input.system.instances[site.instance_id]],
                    PolymerSequence,
                )
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
            site.instance_id if isinstance(site, AtomSite) else site[0] for site in input.token_sites
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
                sites=tuple(complete_sites),
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
                        "mean_plddt": Score(float(atom_plddt[sample_index].mean()), preferred="higher"),
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
        r"""
        Run the Protenix trunk, sampler, and prediction heads.
        """

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
        representations: _ProtenixTrunkOutput,
        coordinates: Tensor,
    ) -> ProtenixOutput:
        r"""
        Compute the distogram and confidence predictions.
        """

        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            representations.single,
            pair=representations.pair,
            token_embedding=representations.token_embedding,
            pred_coords=coordinates,
            token_to_rep_atom=input.token_to_rep_atom,
            atom_to_token=input.atom_to_token,
            atom_token_index=input.atom_token_index,
        )
        return ProtenixOutput(
            coordinates=coordinates,
            distogram_logits=self.distogram_head(representations.pair),
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
        )

    def _denoise_context(
        self,
        input: ProtenixInput,
        representations: _ProtenixTrunkOutput,
        *,
        reference_positions: Tensor,
    ) -> CoordinateDenoiseContext:
        r"""
        Assemble conditioning shared by forward and the sampler.
        """

        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=representations.token_embedding,
            single=representations.single,
            pair=representations.pair,
            relative_features=representations.relative_features,
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
    ) -> _ProtenixTrunkOutput:
        r"""
        Embed the inputs and run the recycling trunk.
        """

        token_embedding = self.input_embeddings(
            input.atom_features,
            token_features=input.token_features,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_to_token=input.atom_to_token,
            num_tokens=input.token_features.shape[-2],
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=self.input_embeddings.build_local_geometry(
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
        return _ProtenixTrunkOutput(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
        )

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
        r"""
        Run the recycling loop.
        """

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
        r"""
        Build the 32+1+1 MSA feature vector for selected rows.
        """

        if indices is None:
            msa_tokens = input.msa_tokens
            has_deletion = input.has_deletion
            deletion_value = input.deletion_value
        else:
            msa_tokens = torch.index_select(input.msa_tokens, -2, indices)
            has_deletion = torch.index_select(input.has_deletion, -2, indices)
            deletion_value = torch.index_select(input.deletion_value, -2, indices)
        one_hot = F.one_hot(msa_tokens.long(), num_classes=32).to(dtype=input.token_features.dtype)
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
        r"""
        Build the initial pair representation and optional constraint bias.
        """

        left = self.pair_left_projection(single).unsqueeze(-2)
        right = self.pair_right_projection(single).unsqueeze(-3)
        token_bonds = input.token_bonds.to(device=single.device, dtype=single.dtype)
        token_bonds = torch.broadcast_to(token_bonds, (*single.shape[:-1], single.shape[-2]))
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds.unsqueeze(-1))
        if self.constraint_embeddings is not None:
            pair = pair + self.constraint_embeddings(
                input.pocket,
                input.token_contact,
                input.atom_contact,
                input.substructure,
            )
        pair_mask = pair_mask_from_token_mask(input.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)

    def _predict_coords(
        self,
        input: ProtenixInput,
        *,
        representations: _ProtenixTrunkOutput,
        reference_positions: Tensor,
        num_samples: int,
        num_sampling_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Run sequential EDM sampling.
        """

        return self.coordinate_sampler(
            self.coordinate_denoiser,
            context=self._denoise_context(
                input,
                representations,
                reference_positions=reference_positions,
            ),
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=5,
            generator=generator,
        )

    def _sample_msa_indices(
        self,
        num_rows: int,
        device: torch.device,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Draw one recycle MSA subset without privileging the query row.
        """

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


class ProtenixConstraintModel(ProtenixModel):
    r"""
    Predict constrained biomolecular complex structures with Protenix.

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = ProtenixConstraintConfig
    config: ProtenixConstraintConfig

    def __init__(self, config: ProtenixConstraintConfig | None = None) -> None:
        super().__init__(ProtenixConstraintConfig() if config is None else config)


class ProtenixInputEmbeddings(nn.Module):
    r"""
    Build token input embeddings from packed atom and token features.
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
            cross_attention_key_source="normalized",
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
        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)


class ProtenixConstraintEmbeddings(nn.Module):
    r"""
    Project and combine pocket, token-contact, atom-contact, and substructure constraints.
    """

    def __init__(self, pair_size: int = 128, *, operator_backend: str = "auto") -> None:
        super().__init__()
        self.pocket_projection = nn.Linear(1, pair_size, bias=False)
        self.token_contact_projection = nn.Linear(2, pair_size, bias=False)
        self.atom_contact_projection = nn.Linear(2, pair_size, bias=False)
        self.substructure_embeddings = ProtenixConstraintSubstructureEmbeddings(
            pair_size, operator_backend=operator_backend
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)

    def forward(
        self,
        pocket: Tensor,
        token_contact: Tensor,
        atom_contact: Tensor,
        substructure: Tensor,
    ) -> Tensor:
        return (
            self.pocket_projection(pocket)
            + self.token_contact_projection(token_contact)
            + self.atom_contact_projection(atom_contact)
            + self.substructure_embeddings(substructure)
        )


class ProtenixConstraintSubstructureEmbeddings(nn.Module):
    r"""
    Embed the flattened token-pair substructure grid with a Transformer.
    """

    def __init__(self, pair_size: int = 128, *, operator_backend: str = "auto") -> None:
        super().__init__()
        hidden_size = 128
        self.input_projection = nn.Linear(4, hidden_size, bias=False)
        self.transformer = ProtenixConstraintSubstructureTransformer(
            hidden_size=hidden_size,
            num_blocks=1,
            num_attention_heads=4,
            transition_size=hidden_size * 4,
            dropout=0.1,
            operator_backend=operator_backend,
        )
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(self, substructure: Tensor) -> Tensor:
        shape = substructure.shape
        flattened = substructure.reshape(-1, shape[-3] * shape[-2], shape[-1])
        hidden = self.transformer(self.input_projection(flattened))
        return self.output_projection(hidden).reshape(*shape[:-1], -1)


class ProtenixConstraintSubstructureTransformer(nn.Module):
    r"""
    Transformer over flattened token-pair substructure features.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_blocks: int,
        num_attention_heads: int,
        transition_size: int,
        dropout: float,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            ProtenixConstraintSubstructureBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                transition_size=transition_size,
                dropout=dropout,
                operator_backend=operator_backend,
            )
            for _ in range(num_blocks)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class ProtenixConstraintSubstructureBlock(nn.Module):
    r"""
    Post-norm self-attention and transition residual unit.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        transition_size: int,
        dropout: float,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.attention = ProtenixConstraintSubstructureAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.attention_output_dropout = nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden_size)
        self.transition = ProtenixConstraintSubstructureTransition(
            hidden_size=hidden_size,
            transition_size=transition_size,
            dropout=dropout,
        )
        self.transition_output_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.attention_norm(hidden + self.attention_output_dropout(self.attention(hidden)))
        return self.transition_norm(hidden + self.transition_output_dropout(self.transition(hidden)))


class ProtenixConstraintSubstructureAttention(nn.Module):
    r"""
    Multi-head self-attention with separate query, key, and value projections.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.dropout = dropout
        self.operator_backend = operator_backend
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.zeros_(self.key_projection.bias)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        shape = (*hidden.shape[:-1], self.num_attention_heads, self.head_size)
        query = self.query_projection(hidden).reshape(shape).transpose(-3, -2)
        key = self.key_projection(hidden).reshape(shape).transpose(-3, -2)
        value = self.value_projection(hidden).reshape(shape).transpose(-3, -2)
        update = projected_attention(
            query,
            key,
            value,
            attention_mask=None,
            operator_backend=self.operator_backend,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output_projection(update.transpose(-3, -2).flatten(-2))


class ProtenixConstraintSubstructureTransition(nn.Module):
    r"""
    ReLU transition update with activation dropout.
    """

    def __init__(self, *, hidden_size: int, transition_size: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, transition_size)
        self.activation_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(transition_size, hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.input_projection(hidden)
        hidden = self.activation_dropout(F.relu(hidden))
        return self.output_projection(hidden)


def _protenix_bin_centers(
    tensor: Tensor,
    minimum: float,
    maximum: float,
    num_bins: int,
) -> Tensor:
    r"""
    Return the evenly spaced confidence-bin centers.
    """

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


@overload
def _protenix_bin_expectation(
    logits: Tensor,
    minimum: float,
    maximum: float,
    *,
    return_probability: Literal[False] = False,
) -> Tensor: ...


@overload
def _protenix_bin_expectation(
    logits: Tensor,
    minimum: float,
    maximum: float,
    *,
    return_probability: Literal[True],
) -> tuple[Tensor, Tensor]: ...


def _protenix_bin_expectation(
    logits: Tensor,
    minimum: float,
    maximum: float,
    *,
    return_probability: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""
    Decode pLDDT/PAE/PDE categorical logits.
    """

    probability = logits.softmax(dim=-1)
    values = probability @ _protenix_bin_centers(logits, minimum, maximum, logits.shape[-1])
    if return_probability:
        return values, probability
    return values


def _protenix_contact_probability(distogram_logits: Tensor) -> Tensor:
    r"""
    Decode the probability of distance below 8 Å.
    """

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
    r"""
    Compute global and chain TM summaries.
    """

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
    r"""
    Compute chain and chain-pair gPDE summaries.
    """

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
    r"""
    Compute chain and chain-pair pLDDT summaries.
    """

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
    r"""
    Compute token-frame validity from the reference conformer.
    """

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
    r"""
    Apply the AF3-style inter-polymer-chain clash rule.
    """

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


# =====================================================================================
# Typed tensor and semantic boundary of the Protenix graph.
# =====================================================================================


@dataclass(frozen=True, kw_only=True)
class _ProtenixTrunkOutput:
    r"""
    Named representations returned by the Protenix recycling trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets


@dataclass(frozen=True)
class ProtenixAtomContact:
    r"""
    Set an upper distance bound between two atoms.

    Args:
        atom_a: First atom occurrence.
        atom_b: Second atom occurrence.
        max_distance: Maximum distance in angstroms.
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
    r"""
    Set an upper distance bound between token-level endpoints.

    Args:
        endpoint_a: First residue or atom endpoint.
        endpoint_b: Second residue or atom endpoint.
        max_distance: Maximum token-center distance in angstroms.

    Raises:
        ValueError: If both endpoints are atoms or ``max_distance`` is not positive.
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
    r"""
    Condition one binder chain to contact a set of residues.

    Args:
        binder: Molecular instance ID of the binder chain.
        contact_residues: Target residue occurrences outside the binder chain.
        max_distance: Maximum binder-to-residue distance in angstroms.
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


@dataclass(kw_only=True, eq=False, repr=False)
class ProtenixInput:
    r"""
    Tensor and molecular-layout input for Protenix.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, and ``M`` MSA depth.

    Args:
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        token_index: Token indices within each residue with shape ``(B, N)``.
        atom_features: Reference atom features with shape ``(B, A, C_atom)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        atom_mask: Valid-atom mask with shape ``(B, A)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        atom_token_index: Atom indices within each token with shape ``(B, A)``.
        token_to_rep_atom: Representative atom index for each token with shape
            ``(B, N)``.
        token_features: Per-token input features with shape ``(B, N, C_token)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        msa_tokens: MSA token IDs with shape ``(B, M, N)``.
        has_deletion: MSA deletion flags with shape ``(B, M, N)``.
        deletion_value: Transformed MSA deletion values with shape ``(B, M, N)``.
        pocket: Directed pocket-distance features with shape ``(B, N, N, 1)``
            in angstroms.
        token_contact: Token-contact distance features with shape
            ``(B, N, N, 2)`` in angstroms.
        atom_contact: Atom-contact distance features with shape
            ``(B, N, N, 2)`` in angstroms.
        substructure: Substructure-condition features with shape
            ``(B, N, N, 4)``.
        system: Molecular system represented by the tensors.
        atom_sites: Atom occurrences aligned to the atom axis.
        token_sites: Molecular occurrences aligned to the token axis.
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
    msa_tokens: Tensor
    has_deletion: Tensor
    deletion_value: Tensor
    pocket: Tensor
    token_contact: Tensor
    atom_contact: Tensor
    substructure: Tensor
    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | ComponentReference, ...]


@dataclass(eq=False, repr=False)
class ProtenixOutput:
    r"""
    Predictions from one Protenix denoising call.

    Args:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        plddt_logits: Per-atom confidence logits with shape ``(B, S, A, D_plddt)``.
        pae_logits: Token-pair aligned-error logits with shape
            ``(B, S, N, N, D_pae)``.
        pde_logits: Token-pair distance-error logits with shape
            ``(B, S, N, N, D_pde)``.
        resolved_logits: Per-atom resolved-state logits with shape
            ``(B, S, A, D_resolved)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    pae_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor

    @property
    def plddt(self) -> Tensor:
        r"""
        Return expected per-atom pLDDT.

        Returns:
            Confidence values with shape ``(B, S, A)`` on the ``0--100`` scale.
        """

        return expected_plddt(self.plddt_logits)
