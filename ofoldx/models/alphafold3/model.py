r"""
AlphaFold 3 top-level model.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import (
    AtomToTokenEncoder,
    CoordinateAtomToTokenEncoder,
    LocalAtomGeometry,
    LocalAtomWindowing,
    TokenToAtomDecoder,
)
from ...modules.attention import PairBiasedSelfAttention, TriangleAttention, TriangleMultiplication
from ...modules.backend import LayerNorm
from ...modules.backend.attention import triangle_attention as dispatch_triangle_attention
from ...modules.denoiser import DenoisingTokenRefiner
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoisingInput,
    CoordinateEdmSampler,
    EdmPreconditioner,
    EdmScheduler,
)
from ...modules.diffusion.conditioning import condition_pair, condition_single
from ...modules.embedding import (
    FrozenFourierEmbedding,
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.embedding.factorized import FactorizedPairLinear
from ...modules.functional import pair_mask_from_msa_mask, pair_mask_from_token_mask
from ...modules.head import (
    AtomConfidenceProjector,
    DistogramHead,
    expected_plddt,
    tm_bin_weights,
)
from ...modules.head.frames import gather_atoms_by_index, insert_sample_axis
from ...modules.transition import Transition
from ...modules.trunk.msa import (
    MsaPairBlock,
    MsaPairWeightedAveraging,
    PostProjectionOuterProductMean,
    TrianglePairUpdateBlock,
)
from ...modules.trunk.pairformer import PairformerBlock, PairformerStack
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .config import AlphaFold3Config


class AlphaFold3Model(Model):
    r"""
    Predict biomolecular complex structures with AlphaFold 3.

    Accepts [`AlphaFold3Input`][ofoldx.models.alphafold3.model.AlphaFold3Input] and returns
    [`AlphaFold3Output`][ofoldx.models.alphafold3.model.AlphaFold3Output].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import PolymerSequence
        >>> from ofoldx.models.alphafold3 import AlphaFold3Config, AlphaFold3Model, AlphaFold3Processor
        >>> config = AlphaFold3Config({
        ...     "single_size": 16,
        ...     "pair_size": 16,
        ...     "template": {
        ...         "hidden_size": 8,
        ...         "num_blocks": 0,
        ...         "triangle_attention_hidden_size": 4,
        ...         "num_triangle_attention_heads": 1,
        ...     },
        ...     "trunk": {"num_pairformer_blocks": 1},
        ...     "recycling_steps": 1,
        ...     "msa_max_depth": 1,
        ...     "denoiser": {
        ...         "num_token_encoder_blocks": 1,
        ...         "num_token_encoder_attention_heads": 1,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "token_hidden_size": 16,
        ...         "num_atom_encoder_blocks": 1,
        ...         "num_atom_encoder_attention_heads": 1,
        ...         "num_atom_decoder_blocks": 1,
        ...         "num_atom_decoder_attention_heads": 1,
        ...         "num_query_atoms": 4,
        ...         "num_key_atoms": 4,
        ...     },
        ...     "distogram": {"num_bins": 4},
        ...     "confidence": {
        ...         "num_plddt_bins": 4,
        ...         "num_pae_bins": 4,
        ...         "num_pde_bins": 4,
        ...         "max_atoms_per_token": 8,
        ...         "num_blocks": 1,
        ...         "num_pair_attention_heads": 1,
        ...         "triangle_multiplication_hidden_size": 4,
        ...         "triangle_attention_hidden_size": 4,
        ...         "num_triangle_attention_heads": 1,
        ...     },
        ... })
        >>> inputs = AlphaFold3Processor()(PolymerSequence.from_one_letter("A"))
        >>> atom_count = inputs.atom_features.shape[1]
        >>> model = AlphaFold3Model(config).eval()
        >>> with torch.inference_mode():
        ...     output = model(
        ...         inputs,
        ...         noisy_coordinates=torch.zeros(1, 1, atom_count, 3),
        ...         noise_level=torch.ones(1, 1),
        ...     )
        >>> tuple(output.coordinates.shape) == (1, 1, atom_count, 3)
        True
        >>> output.distogram_logits.shape[-1]
        4
    """

    config_class = AlphaFold3Config
    config: AlphaFold3Config

    def __init__(self, config: AlphaFold3Config | None = None) -> None:
        config = AlphaFold3Config() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self._msa_max_depth = config.msa_max_depth
        self.input_embeddings = AlphaFold3InputEmbeddings(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
        )
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.token_feature_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.token_feature_size, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        # Preserve released initialization order while registering children in execution order.
        single_recycle_norm = LayerNorm(config.single_size)
        single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.template_embedder = AlphaFold3TemplateEmbedder(
            pair_size=config.pair_size,
            hidden_size=config.template.hidden_size,
            num_blocks=config.template.num_blocks,
            vocab_size=config.template.vocab_size,
            num_distogram_bins=config.template.num_distogram_bins,
            triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.template.num_triangle_attention_heads,
            expansion=config.template.expansion,
        )
        self.msa_trunk = AlphaFold3MsaTrunk(
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
        )
        self.single_recycle_norm = single_recycle_norm
        self.single_recycle_projection = single_recycle_projection
        self.pairformer_stack = AlphaFold3PairformerStack(
            num_blocks=config.trunk.num_pairformer_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
        )
        self.coordinate_denoiser = AlphaFold3CoordinateDenoiser(
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
        self.distogram_head = DistogramHead(config.pair_size, config.distogram.num_bins, bias=False)
        self.confidence_head = AlphaFold3ConfidenceHead(
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
            single_max_abs=config.confidence.single_max_abs,
        )
        # Atom cross-attention adaptively normalizes queries and keys from the raw hidden state.
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Zero-initialize the recycling and PAE/PDE output projections.

        Returns:
            ``None``.
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)

    def forward(
        self,
        input: AlphaFold3Input,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        use_conditioning: bool = True,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3Output:
        r"""
        Denoise atom coordinates at specified EDM noise levels.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            noisy_coordinates: Noisy coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: EDM noise standard deviations with shape ``(B, S)`` in
                angstroms.
            use_conditioning: Whether trunk single and pair values condition the
                diffusion head. The atom path always receives the trunk single.
            recycling_steps: Number of complete trunk passes. ``None`` uses
                the configured value.
            generator: Optional generator for the per-recycle MSA permutations.

        Returns:
            Denoised coordinates and confidence tensors with the shapes described
            by [`AlphaFold3Output`][]. No diffusion sampling is performed.

        Raises:
            ValueError: If ``recycling_steps`` is not positive.
        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(input, recycling_steps=recycling_steps, generator=generator)
        coordinates = self._denoise(
            self._denoise_context(
                input,
                representations,
                use_conditioning=use_conditioning,
            ).to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return self._assemble_output(input, representations, coordinates)

    @torch.inference_mode()
    def sample(
        self,
        input: AlphaFold3Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int | None = None,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3Output:
        r"""
        Run the recycled trunk, coordinate sampling, and prediction heads.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                configured value.
            num_samples: Number of coordinate samples. ``None`` uses five.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses
                the configured value.
            generator: Optional generator for MSA realization and coordinate diffusion.

        Returns:
            Final trunk representations, coordinate samples, distogram logits,
            and confidence logits.

        Raises:
            ValueError: If ``recycling_steps`` is not positive.
        """

        num_samples = 5 if num_samples is None else num_samples
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(input, recycling_steps=recycling_steps, generator=generator)
        pred_coords = self.coordinate_sampler(
            self._denoise,
            context=self._denoise_context(input, representations),
            num_samples=num_samples,
            num_sampling_steps=self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps,
            generator=generator,
        )
        return self._assemble_output(input, representations, pred_coords)

    @torch.inference_mode()
    def fold(
        self,
        input: AlphaFold3Input,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular assembly.

        One recycled trunk execution is shared by all coordinate samples. ``generator`` controls
        the per-recycle MSA permutations, initial Gaussian coordinates, stochastic
        churn, and per-step rigid augmentations; ``None`` uses PyTorch's ambient
        random state.

        Args:
            input: Processor output for one assembly, including molecular identities
                aligned to the atom and token axes.
            num_samples: Number of diffusion samples. ``None`` uses five.
            generator: Optional generator for MSA realization and coordinate diffusion.
            recycling_steps: Number of complete trunk passes. ``None`` uses
                the configured value.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.

        Returns:
            Generation-ordered predictions. Structure coordinates, PAE, and PDE
            values are in angstroms; pLDDT is on ``0--100``.

        Raises:
            ValueError: If a count is not positive or the input lacks molecular
                identities required to decode structures.
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
        if input.system is None or not input.atom_sites or not input.token_sites:
            raise ValueError(
                "fold requires a Processor-built input carrying system, atom_sites, and token_sites; "
                "bare-tensor inputs can only drive forward/sample"
            )

        return self._decode_fold_result(
            input,
            num_samples=num_samples,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

    def _decode_fold_result(
        self,
        input: AlphaFold3Input,
        *,
        num_samples: int,
        recycling_steps: int,
        num_sampling_steps: int,
        generator: torch.Generator | None,
    ) -> FoldResult:
        r"""
        Sample diffusion rollouts and decode one generation-ordered fold result.

        Shares one recycled trunk execution across ``num_samples`` rollouts, decodes each sample's
        [`Structure`][ofoldx.data.structure.Structure], and computes pTM, ipTM, PAE/PDE
        aggregates, disorder, clashes, and ranking scores.
        """

        output = self.sample(
            input,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

        assert input.system is not None
        atom_count = len(input.atom_sites)
        token_count = len(input.token_sites)
        complete_sites = [site for site, atom in input.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray([complete_index[site] for site in input.atom_sites], dtype=np.intp)

        coordinates = output.coordinates[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        atom_plddt = output.plddt[0, :, :atom_count].to(device="cpu", dtype=torch.float32)
        pae_logits = output.pae_logits[0, :, :token_count, :token_count].float().cpu()
        pde_logits = output.pde_logits[0, :, :token_count, :token_count].float().cpu()
        resolved_logits = output.resolved_logits[0, :, :atom_count].float().cpu()
        distogram_logits = output.distogram_logits[0, :token_count, :token_count].float().cpu()
        ptm, iptm, ptm_iptm_average = (value[0].cpu() for value in self.confidence_summary(output, input))
        del output

        pae = _expected_bin_value(pae_logits, alphafold3_pae_bin_centers(pae_logits.shape[-1]).cpu())
        pde = _expected_bin_value(pde_logits, _uniform_bin_centers(pde_logits.shape[-1], 0.0, 32.0))
        resolved_probability = resolved_logits.softmax(dim=-1)[..., 1]
        contact_probability = _contact_probability(distogram_logits).numpy()

        # Chain-level confidence summaries operate on decoded per-sample arrays.
        full_pae = pae.numpy()
        full_pde = pde.numpy()
        chain_indices = input.chain_index[0, :token_count].cpu().numpy().astype(np.int64)
        frame_mask = input.token_mask[0, :token_count].to(torch.bool).cpu().numpy()
        instance_ids = tuple(input.system.instances.keys())
        unique_chain = np.unique(chain_indices)
        chain_axis = tuple(instance_ids[int(index)] for index in unique_chain)
        multi_chain = unique_chain.shape[0] > 1

        pae_single_mask = np.tile(frame_mask[:, None], (1, token_count))
        frame_pair_mask = frame_mask[:, None] & frame_mask[None, :]
        tm_interface = alphafold3_tm_adjusted_pae(
            pae_logits,
            pair_mask=torch.from_numpy(frame_pair_mask).unsqueeze(0).expand(pae_logits.shape[0], -1, -1),
            chain_index=torch.from_numpy(chain_indices).unsqueeze(0).expand(pae_logits.shape[0], -1),
            interface=True,
        ).numpy()

        chain_pair_pde_mean, chain_pair_pde_min = _chain_pair_pde(full_pde, chain_indices)
        _, chain_pair_pae_min, _ = _chain_pair_pae(full_pae, chain_indices, mask=pae_single_mask)
        pde_ichain, pde_xchain, pde_full_chain = _pde_single(chain_indices, full_pde, contact_probability)
        chain_metrics = _pae_metrics(chain_indices, full_pae, pae_single_mask, contact_probability, tm_interface)
        ranking_confidence_pae = _rank_metric(
            full_pae,
            contact_probability * frame_mask[:, None].astype(float),
            rank_metric_eps=self.config.rank_metric_eps,
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
            disorder = alphafold3_fraction_disordered(structure)
            clash = alphafold3_has_clash(structure)
            sample_ranking_score = alphafold3_ranking_score(
                float(ptm[sample_index]), float(iptm[sample_index]), disorder, clash
            )
            ranking_confidence = float(ptm_iptm_average[sample_index]) if multi_chain else float(ptm[sample_index])
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores={
                        "mean_plddt": Score(float(atom_plddt[sample_index].mean()), preferred="higher"),
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
                        "plddt": Metric(values=atom_plddt[sample_index].numpy(), axis=input.atom_sites),
                        "resolved_probability": Metric(
                            values=resolved_probability[sample_index].numpy(), axis=input.atom_sites
                        ),
                        "predicted_distance_error": Metric(
                            values=pde_full_chain[sample_index],
                            axis=input.token_sites,
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
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=pde[sample_index].numpy(),
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                            unit="angstrom",
                        ),
                        "contact_probs": PairMetric(
                            values=contact_probability,
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
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
        return FoldResult(predictions=tuple(predictions))

    def _assemble_output(
        self,
        input: AlphaFold3Input,
        representations: _AlphaFold3TrunkOutput,
        coordinates: Tensor,
    ) -> AlphaFold3Output:
        r"""
        Compute distogram and confidence predictions.
        """

        token_embedding = representations.token_embedding
        single = representations.single
        pair = representations.pair
        confidence_dtype = self.confidence_head.pair_embedder.left_token_projection.weight.dtype
        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            single.to(dtype=confidence_dtype),
            pair=pair.to(dtype=confidence_dtype),
            token_embedding=token_embedding.to(dtype=confidence_dtype),
            pred_coords=coordinates,
            token_to_rep_atom=input.token_to_rep_atom,
            atom_to_token=input.atom_to_token,
            atom_token_index=input.atom_token_index,
        )
        with torch.autocast(device_type=pair.device.type, enabled=False):
            distogram_logits = self.distogram_head(pair.to(dtype=self.distogram_head.projection.weight.dtype))
        return AlphaFold3Output(
            single=single,
            pair=pair,
            coordinates=coordinates,
            distogram_logits=distogram_logits,
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            batch_size=[single.shape[0]],
        )

    def _denoise_context(
        self,
        input: AlphaFold3Input,
        representations: _AlphaFold3TrunkOutput,
        *,
        use_conditioning: bool = True,
    ) -> CoordinateDenoiseContext:
        r"""
        Assemble conditioning shared by ``forward`` and the sampler.
        """

        conditioning_single = representations.single if use_conditioning else 0 * representations.single
        conditioning_pair = representations.pair if use_conditioning else 0 * representations.pair
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=representations.token_embedding,
            single=representations.single,
            conditioning_single=conditioning_single,
            conditioning_pair=conditioning_pair,
            relative_features=representations.relative_features,
            local_geometry=_build_alphafold3_local_geometry(
                self.coordinate_denoiser.atom_encoder.atom_pair_encoder.local_geometry_builder.windowing,
                reference_positions=input.reference_positions,
                reference_space_id=input.reference_space_id,
                atom_to_token=input.atom_to_token,
                atom_mask=input.atom_mask,
                num_tokens=input.token_features.shape[-2],
            ),
        )

    def _denoise(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Run the FP32 diffusion head independently of trunk autocast.
        """

        with torch.autocast(device_type=inputs.noisy_coords.device.type, enabled=False):
            return self.coordinate_denoiser(inputs)

    def confidence_summary(
        self,
        output: AlphaFold3Output,
        input: AlphaFold3Input,
        *,
        max_error_bin: float = 31.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Compute pTM, ipTM, and their weighted average from PAE logits.

        Args:
            output: Predictions whose PAE logits have shape ``(B, S, N, N, D)``.
            input: The corresponding model input. ``token_mask`` and
                ``chain_index`` have shape ``(B, N)``.
            max_error_bin: Upper edge of the resolved PAE bins in angstroms.

        Returns:
            Three tensors with shape ``(B, S)``: pTM, ipTM, and
            ``0.8 * ipTM + 0.2 * pTM``. For single-chain systems, the third value
            uses pTM in place of undefined ipTM.
        """

        num_tokens = output.pae_logits.shape[-2]
        frame_mask = input.token_mask.to(torch.bool)
        # Broadcast token-axis fields across the per-sample logit axis so the summary is per-sample.
        sample_axes = output.pae_logits.dim() - 3
        chain_index = input.chain_index
        for _ in range(sample_axes - chain_index.dim() + 1):
            chain_index = chain_index.unsqueeze(1)
            frame_mask = frame_mask.unsqueeze(1)
        return alphafold3_confidence_summary(
            output.pae_logits,
            frame_mask=frame_mask.expand(*output.pae_logits.shape[:-3], num_tokens),
            chain_index=chain_index.expand(*output.pae_logits.shape[:-3], num_tokens),
            max_error_bin=max_error_bin,
        )

    def _run_trunk(
        self,
        input: AlphaFold3Input,
        *,
        recycling_steps: int,
        generator: torch.Generator | None,
    ) -> _AlphaFold3TrunkOutput:
        r"""
        Build the token embedding and run the recycled trunk to final single/pair representations.

        Args:
            input: Typed model input.
            recycling_steps: Number of complete trunk passes to run.
            generator: Generator used to realize the MSA independently for each pass.

        Returns:
            Named token embedding, single, pair, and relative-position representations.
        """

        atom_dtype = self.input_embeddings.atom_encoder.atom_conditioner.atom_feature_projection.weight.dtype
        reference_positions = input.reference_positions.to(dtype=atom_dtype)
        token_embedding = self.input_embeddings(
            input.atom_features.to(dtype=atom_dtype),
            token_features=input.token_features,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges.to(dtype=atom_dtype),
            reference_mask=input.reference_mask,
            atom_to_token=input.atom_to_token,
            num_tokens=input.token_features.shape[-2],
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=self.input_embeddings.build_local_geometry(
                reference_positions,
                input.reference_space_id,
                atom_to_token=input.atom_to_token,
                atom_mask=input.atom_mask,
                num_tokens=input.token_features.shape[-2],
            ),
        )
        # Per-atom conditioning remains float32 until entering the Evoformer projection dtype.
        token_embedding = token_embedding.to(dtype=self.single_init_projection.weight.dtype)

        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            dtype=single_init.dtype,
        )
        pair_init = self._initial_pair(token_embedding, input, relative_features)

        pair_mask = pair_mask_from_token_mask(input.token_mask)
        single, pair = self._run_recycling(
            input,
            token_embedding,
            single_init,
            pair_init,
            pair_mask,
            recycling_steps=recycling_steps,
            generator=generator,
        )

        return _AlphaFold3TrunkOutput(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            batch_size=[single.shape[0]],
        )

    def _run_recycling(
        self,
        input: AlphaFold3Input,
        token_embedding: Tensor,
        single_init: Tensor,
        pair_init: Tensor,
        pair_mask: Tensor,
        *,
        recycling_steps: int,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Run the differentiable recycling loop.
        """

        template_features = input.template_features
        if template_features is not None:
            template_features = TemplateFeatures(
                distogram=template_features.distogram.to(dtype=pair_init.dtype),
                pseudo_beta_mask=template_features.pseudo_beta_mask.to(dtype=pair_init.dtype),
                residue_type_ids=template_features.residue_type_ids,
                unit_vector=template_features.unit_vector.to(dtype=pair_init.dtype),
                backbone_frame_mask=template_features.backbone_frame_mask.to(dtype=pair_init.dtype),
                template_mask=template_features.template_mask,
                batch_size=[template_features.distogram.shape[0]],
            )
        msa_features = input.msa_features.to(dtype=pair_init.dtype)
        msa_mask = input.msa_mask

        # Recycled representations are stored in float32 and cast to the activation
        # dtype at the start of each pass.
        single = torch.zeros_like(single_init, dtype=torch.float32)
        pair = torch.zeros_like(pair_init, dtype=torch.float32)
        for _ in range(recycling_steps):
            pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair.to(dtype=pair_init.dtype)))
            if template_features is not None:
                pair = pair + self.template_embedder(pair, template_features, pair_mask=pair_mask)
            realized_msa_features, realized_msa_mask = _realize_alphafold3_msa(
                msa_features,
                msa_mask,
                max_depth=self._msa_max_depth,
                generator=generator,
            )
            _, pair = self.msa_trunk(
                realized_msa_features,
                pair,
                token_embedding,
                msa_mask=realized_msa_mask,
                pair_mask=pair_mask,
            )
            single = single_init + self.single_recycle_projection(
                self.single_recycle_norm(single.to(dtype=single_init.dtype))
            )
            single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
            single = single.float()
            pair = pair.float()

        return single, pair

    def _initial_pair(
        self,
        token_embedding: torch.Tensor,
        input: AlphaFold3Input,
        relative_features: RelativePositionBuckets,
    ) -> torch.Tensor:
        r"""
        Build the initial pair representation from single projections, relative positions, and token bonds.

        Args:
            token_embedding: Initial target-feature representation ``(..., N_token, token_feature_size)``.
            input: Typed model input (provides token bonds and the token mask).
            relative_features: Relative-position bucket features for the pair embedder.

        Returns:
            Masked initial pair representation ``(..., N_token, N_token, pair_size)``.
        """

        left = self.pair_left_projection(token_embedding).unsqueeze(-2)
        right = self.pair_right_projection(token_embedding).unsqueeze(-3)
        token_bonds = input.token_bonds.to(device=token_embedding.device, dtype=token_embedding.dtype)
        token_bonds = torch.broadcast_to(token_bonds, (*token_embedding.shape[:-1], token_embedding.shape[-2]))
        token_bonds = token_bonds.unsqueeze(-1)
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds)

        pair_mask = pair_mask_from_token_mask(input.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)


class AlphaFold3CoordinateDenoiser(nn.Module):
    r"""
    AlphaFold 3 coordinate-denoising composition.
    """

    def __init__(
        self,
        num_encoder_blocks: int = 24,
        num_encoder_attention_heads: int = 16,
        *,
        sigma_data: float = 16.0,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        token_hidden_size: int = 768,
        single_size: int = 384,
        pair_size: int = 128,
        token_feature_size: int = 449,
        atom_feature_size: int = 385,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.conditioner = AlphaFold3DenoisingConditioner(
            pair_input_size=pair_size + 139,
            pair_size=pair_size,
            single_conditioning_size=single_size + token_feature_size,
            single_size=single_size,
            sigma_data=sigma_data,
            backend=backend,
        )
        self.atom_encoder = CoordinateAtomToTokenEncoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            atom_feature_size=atom_feature_size,
            backend=backend,
        )
        self.token_refiner = DenoisingTokenRefiner(
            num_blocks=num_encoder_blocks,
            num_attention_heads=num_encoder_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.atom_decoder = TokenToAtomDecoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )

    def forward(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Denoise atom coordinates with the AlphaFold 3 diffusion-head equation.
        """

        scaled_coordinates = self.preconditioner.scale_noisy_coords(
            inputs.noisy_coords,
            inputs.noise_level,
        )
        denoising_single, denoising_pair = self.conditioner(
            inputs.noise_level,
            token_embedding=inputs.token_embedding,
            single=inputs.conditioning_single,
            pair=inputs.conditioning_pair,
            relative_features=inputs.relative_features,
        )
        atom_output, token_hidden = self.token_refiner(
            self.atom_encoder,
            atom_features=inputs.atom_features,
            atom_to_token=inputs.atom_to_token,
            single=inputs.single,
            denoising_single=denoising_single,
            denoising_pair=denoising_pair,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            reference_mask=inputs.reference_mask,
            atom_mask=inputs.atom_mask.to(dtype=inputs.atom_features.dtype),
            noisy_coords=scaled_coordinates,
            local_geometry=inputs.local_geometry,
        )
        coordinate_update = self.atom_decoder(
            atom_to_token=inputs.atom_to_token,
            token_hidden=token_hidden,
            atom_query=atom_output.atom_query,
            atom_context=atom_output.atom_context,
            atom_mask=None,
            attention_layout="local",
            num_valid_atoms=inputs.local_geometry.window_info.num_valid_atoms,
        )
        return self.preconditioner.combine_update(
            inputs.noisy_coords,
            inputs.noise_level,
            coordinate_update,
        )


class AlphaFold3DenoisingConditioner(nn.Module):
    r"""
    Concatenate raw relative-position buckets into AlphaFold 3 denoising pair conditioning.
    """

    def __init__(
        self,
        pair_input_size: int = 267,
        pair_size: int = 128,
        single_conditioning_size: int = 833,
        single_size: int = 384,
        noise_embedding_size: int = 256,
        sigma_data: float = 16.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.pair_input_norm = LayerNorm(pair_input_size, bias=False, backend=backend)
        self.pair_projection = nn.Linear(pair_input_size, pair_size, bias=False)
        self.pair_transitions = nn.ModuleList(Transition(pair_size, expansion=2, backend=backend) for _ in range(2))
        self.single_conditioning_norm = LayerNorm(single_conditioning_size, bias=False, backend=backend)
        self.single_conditioning_projection = nn.Linear(single_conditioning_size, single_size, bias=False)
        self.noise_embedding = FrozenFourierEmbedding(noise_embedding_size)
        self.noise_norm = LayerNorm(noise_embedding_size, bias=False, backend=backend)
        self.noise_projection = nn.Linear(noise_embedding_size, single_size, bias=False)
        self.single_transitions = nn.ModuleList(Transition(single_size, expansion=2, backend=backend) for _ in range(2))

    def forward(
        self,
        noise_level: Tensor,
        *,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> tuple[Tensor, Tensor]:
        denoising_pair = condition_pair(
            pair,
            _relative_position_one_hot(relative_features),
            norm=self.pair_input_norm,
            projection=self.pair_projection,
            transitions=self.pair_transitions,
        )
        denoising_single, _ = condition_single(
            single,
            token_embedding,
            torch.log(noise_level / self.sigma_data) / 4,
            norm=self.single_conditioning_norm,
            projection=self.single_conditioning_projection,
            noise_fourier=self.noise_embedding,
            noise_norm=self.noise_norm,
            noise_projection=self.noise_projection,
            transitions=self.single_transitions,
            sample_axis=True,
        )
        return denoising_single, denoising_pair


class AlphaFold3InputEmbeddings(nn.Module):
    r"""
    Build token input embeddings by encoding atoms to tokens and appending the token-feature tail.
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
        atom_dtype = self.atom_encoder.atom_conditioner.atom_feature_projection.weight.dtype
        # Reference features remain FP32 even when the Evoformer uses a lower precision.
        with torch.autocast(device_type=reference_positions.device.type, enabled=False):
            atom_encoder_output = self.atom_encoder(
                atom_features.to(dtype=atom_dtype),
                reference_positions=reference_positions.to(dtype=atom_dtype),
                reference_charges=torch.arcsinh(reference_charges.to(dtype=atom_dtype)),
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

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        *,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        num_tokens: int,
    ) -> LocalAtomGeometry:
        return _build_alphafold3_local_geometry(
            self.atom_encoder.atom_pair_encoder.local_geometry_builder.windowing,
            reference_positions=reference_positions,
            reference_space_id=reference_space_id,
            atom_to_token=atom_to_token,
            atom_mask=atom_mask,
            num_tokens=num_tokens,
        )


class AlphaFold3TemplateEmbedder(nn.Module):
    r"""
    Embed AlphaFold 3 template-derived pair features into the trunk pair representation.

    Args:
        pair_size: Width of the trunk pair representation this update is added into.
        hidden_size: Width of the internal template pair channel.
        num_blocks: Number of pair-only triangle update blocks; ``0`` disables the embedder.
        vocab_size: Residue-type vocabulary size (one block per side of the pair).
        num_distogram_bins: Number of distogram bins in the packed template features.
        triangle_attention_hidden_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Transition expansion factor inside each triangle update block.
        backend: Backend selector forwarded to the triangle update blocks.
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 0,
        vocab_size: int = 32,
        num_distogram_bins: int = 39,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.template_projection = FactorizedPairLinear(
            pair_size=num_distogram_bins + 5,
            token_size=vocab_size,
            output_size=hidden_size,
            pair_prefix_size=num_distogram_bins + 1,
            bias=False,
            backend=backend,
        )
        self.blocks = nn.ModuleList(
            AlphaFold3TrianglePairUpdateBlock(
                pair_size=hidden_size,
                triangle_multiplication_hidden_size=hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_blocks)
        )
        self.hidden_norm = LayerNorm(hidden_size, backend=backend)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(
        self,
        pair: Tensor,
        templates: TemplateFeatures,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return template pair updates for an input pair representation.

        Args:
            pair: ``[..., N_token, N_token, pair_size]`` trunk pair representation.
            templates: Pre-packed AlphaFold 3 template features.
            pair_mask: Optional ``[..., N_token, N_token]`` pair validity mask forwarded to the
                triangle update blocks.

        Returns:
            ``[..., N_token, N_token, pair_size]`` template pair update. The
            update is zero when ``num_blocks == 0`` or the template count is zero.
        """

        if self.num_blocks == 0 or templates.distogram.shape[-4] == 0:
            return pair.new_zeros(pair.shape)

        normalized_pair = self.pair_norm(pair)
        template_updates = []
        for template_index in range(templates.distogram.shape[-4]):
            template_pair = self._single_template_update(
                normalized_pair,
                templates=templates,
                template_index=template_index,
                pair_mask=pair_mask,
            )
            if templates.template_mask is not None:
                mask = templates.template_mask[..., template_index].to(
                    device=template_pair.device,
                    dtype=template_pair.dtype,
                )
                template_pair = template_pair * mask.reshape(*mask.shape, 1, 1, 1)
            template_updates.append(template_pair)

        update = torch.stack(template_updates, dim=-4).mean(dim=-4)
        return self.output_projection(self.activation(update))

    def _single_template_update(
        self,
        pair: Tensor,
        *,
        templates: TemplateFeatures,
        template_index: int,
        pair_mask: Tensor | None,
    ) -> Tensor:
        r"""
        Embed and refine one AlphaFold 3 template in the hidden pair channel.
        """

        geom, residue_one_hot = self._pack_features(templates, template_index)
        template_pair = self.pair_projection(pair)
        template_pair = template_pair + self.template_projection(geom, residue_one_hot)
        for block in self.blocks:
            template_pair = block(template_pair, pair_mask=pair_mask)
        return self.hidden_norm(template_pair)

    def _pack_features(self, templates: TemplateFeatures, template_index: int) -> tuple[Tensor, Tensor]:
        r"""
        Return geometric pair features and residue one-hot features for one template.
        """

        expected_residue_shape = templates.distogram.shape[:-2]
        if templates.residue_type_ids.shape != expected_residue_shape:
            raise ValueError(
                "residue_type_ids must have shape " f"{expected_residue_shape}; got {templates.residue_type_ids.shape}"
            )
        residue_type_ids = templates.residue_type_ids[..., template_index, :]
        residue_one_hot = F.one_hot(residue_type_ids, num_classes=self.vocab_size).to(dtype=templates.distogram.dtype)

        geom = torch.cat(
            [
                templates.distogram[..., template_index, :, :, :],
                templates.pseudo_beta_mask[..., template_index, :, :].unsqueeze(-1),
                templates.unit_vector[..., template_index, :, :, :],
                templates.backbone_frame_mask[..., template_index, :, :].unsqueeze(-1),
            ],
            dim=-1,
        )
        return geom, residue_one_hot


class AlphaFold3BinnedDistancePairEmbedder(nn.Module):
    r"""
    Add token features and binned representative-atom distances to an AlphaFold 3 pair state.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__()
        self.left_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        self.right_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        buffer_device = self.left_token_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        lower_bins = torch.arange(
            distance_bin_start,
            distance_bin_end,
            distance_bin_step,
            device=buffer_device,
            dtype=self.left_token_projection.weight.dtype,
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([torch.inf])], dim=-1)
        self.lower_bins: Tensor
        self.upper_bins: Tensor
        self.register_buffer("lower_bins", lower_bins, persistent=False)
        self.register_buffer("upper_bins", upper_bins, persistent=False)
        self.distance_one_hot_projection = nn.Linear(len(lower_bins), pair_size, bias=False)

    def forward(self, pair: Tensor, token_embedding: Tensor, representative_coords: Tensor) -> Tensor:
        pair = pair + self.left_token_projection(token_embedding).unsqueeze(-2)
        pair = pair + self.right_token_projection(token_embedding).unsqueeze(-3)
        distance = torch.cdist(representative_coords.to(torch.float32), representative_coords.to(torch.float32))
        distance_bins = ((distance[..., None] > self.lower_bins) & (distance[..., None] < self.upper_bins)).to(
            pair.dtype
        )
        return pair + self.distance_one_hot_projection(distance_bins)


class AlphaFold3MsaTrunk(nn.Module):
    r"""
    AlphaFold 3 MSA trunk with homogeneous family-owned joint blocks and a final pair-only update.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        msa_feature_size: int = 34,
        token_feature_size: int = 449,
        num_blocks: int = 4,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_projection = nn.Linear(msa_feature_size, msa_size, bias=False)
        self.token_projection = nn.Linear(token_feature_size, msa_size, bias=False)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            AlphaFold3MsaPairBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                num_msa_attention_heads=num_msa_attention_heads,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_blocks - 1)
        )
        self.final_pair_update = (
            AlphaFold3FinalMsaPairUpdate(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            if num_blocks
            else None
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        token_embedding: Tensor | None = None,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        msa = self.msa_projection(msa)
        if token_embedding is not None:
            msa = msa + self.token_projection(token_embedding).unsqueeze(-3)
        if pair_mask is None and msa_mask is not None:
            pair_mask = pair_mask_from_msa_mask(msa_mask)
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        if self.final_pair_update is not None:
            pair = self.final_pair_update(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return msa, pair


class AlphaFold3MsaPairBlock(MsaPairBlock):
    r"""
    Joint AlphaFold 3 MSA/pair block with the family-owned triangle-bias equation.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.outer_product_mean = PostProjectionOuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.msa_update = AlphaFold3MsaUpdateBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            expansion=expansion,
            backend=backend,
        )
        self.triangle_pair_update = AlphaFold3TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            backend=backend,
        )


class AlphaFold3FinalMsaPairUpdate(nn.Module):
    r"""
    Final AlphaFold 3 MSA-stage update that refines only the pair representation.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = PostProjectionOuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.triangle_pair_update = AlphaFold3TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            backend=backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        return self.triangle_pair_update(pair, pair_mask=pair_mask)


class AlphaFold3MsaUpdateBlock(nn.Module):
    r"""
    AlphaFold 3 MSA update without structured dropout.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            backend=backend,
        )
        self.transition = Transition(msa_size, expansion=expansion, backend=backend)

    def forward(self, msa: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        msa = msa + self.msa_pair_weighted_averaging(msa, pair, pair_mask=pair_mask)
        return msa + self.transition(msa)


class AlphaFold3PairformerStack(PairformerStack):
    r"""
    Homogeneous stack of AlphaFold 3 Pairformer blocks.
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
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            AlphaFold3PairformerBlock(
                single_size=single_size,
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_blocks)
        )


class AlphaFold3PairformerBlock(PairformerBlock):
    r"""
    One AlphaFold 3 Pairformer block with the family-owned triangle-bias equation.
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
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.pair_update = AlphaFold3TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            backend=backend,
        )
        self.pair_attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)


class AlphaFold3TrianglePairUpdateBlock(TrianglePairUpdateBlock):
    r"""
    AlphaFold 3 pair update with its exact ending-node triangle-bias equation.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            backend=backend,
        )
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
        self.triangle_attention_ending_node = AlphaFold3EndingNodeTriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.transition = Transition(pair_size, expansion=expansion, backend=backend)

    def triangle_update(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        pair = pair + self.triangle_multiplication_outgoing(pair, pair_mask=pair_mask)
        pair = pair + self.triangle_multiplication_incoming(pair, pair_mask=pair_mask)
        pair = pair + self.triangle_attention_starting_node(pair, pair_mask=pair_mask)
        end_pair_mask = None if pair_mask is None else pair_mask.transpose(-1, -2)
        ending_update = self.triangle_attention_ending_node(
            pair.transpose(-3, -2),
            pair_mask=end_pair_mask,
        ).transpose(-3, -2)
        return pair + ending_update


class AlphaFold3EndingNodeTriangleAttention(TriangleAttention):
    r"""
    AlphaFold 3 ending-node triangle attention with bias from the pre-transpose pair orientation.
    """

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        normalized = self.pair_norm(pair)
        bias_source = normalized.transpose(-3, -2)
        projected_bias = self.bias_projection(bias_source)
        triangle_bias = projected_bias.movedim(-1, -3).unsqueeze(-4)
        return dispatch_triangle_attention(
            normalized,
            triangle_bias,
            pair_mask,
            projection=self.attention.projection,
            backend=self.attention.backend,
        )


class AlphaFold3ConfidenceHead(nn.Module):
    r"""
    AlphaFold 3 confidence-prediction composition.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        num_blocks: int = 4,
        num_pair_attention_heads: int = 16,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        single_max_abs: float | None = 512.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_max_abs = single_max_abs
        self.pair_embedder = AlphaFold3BinnedDistancePairEmbedder(
            pair_size=pair_size,
            token_feature_size=token_feature_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )
        self.pairformer_stack = AlphaFold3PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            expansion=expansion,
            backend=backend,
        )
        self.atom_projector = AtomConfidenceProjector(
            single_size=single_size,
            num_plddt_bins=num_plddt_bins,
            num_resolved_bins=num_resolved_bins,
            max_atoms_per_token=max_atoms_per_token,
            backend=backend,
        )
        self.pair_projector = AlphaFold3PostProjectionSymmetricPairConfidenceProjector(
            pair_size=pair_size,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
        )

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        token_embedding: Tensor,
        pred_coords: Tensor,
        token_to_rep_atom: Tensor,
        atom_to_token: Tensor,
        atom_token_index: Tensor,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.single_max_abs is not None:
            single = torch.clamp(single, min=-self.single_max_abs, max=self.single_max_abs)
        representative_coords = gather_atoms_by_index(
            pred_coords,
            insert_sample_axis(token_to_rep_atom, pred_coords.shape[-3]),
        )

        outputs = []
        for representative_coords_sample in representative_coords.unbind(dim=-3):
            sample_pair = self.pair_embedder(pair, token_embedding, representative_coords_sample)
            sample_single, sample_pair = self.pairformer_stack(single, sample_pair, pair_mask=pair_mask)
            sample_plddt, sample_resolved = self.atom_projector(sample_single, atom_to_token, atom_token_index)
            sample_pae, sample_pde = self.pair_projector(sample_pair)
            outputs.append((sample_plddt, sample_pae, sample_pde, sample_resolved))

        plddt_logits, pae_logits, pde_logits, resolved_logits = zip(*outputs)
        return (
            torch.stack(plddt_logits, dim=-3),
            torch.stack(pae_logits, dim=-4),
            torch.stack(pde_logits, dim=-4),
            torch.stack(resolved_logits, dim=-3),
        )


class AlphaFold3PostProjectionSymmetricPairConfidenceProjector(nn.Module):
    r"""
    Project AlphaFold 3 pair states and symmetrize PDE after nonlinear normalization.
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
    ) -> None:
        super().__init__()
        self.pae_norm = LayerNorm(pair_size)
        self.pde_norm = LayerNorm(pair_size)
        self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        pae_pair = pair.to(dtype=self.pae_projection.weight.dtype)
        pde_pair = pair.to(dtype=self.pde_projection.weight.dtype)
        pae_logits = self.pae_projection(self.pae_norm(pae_pair))
        left_distance_logits = self.pde_projection(self.pde_norm(pde_pair))
        return pae_logits, left_distance_logits + left_distance_logits.transpose(-2, -3)


def _relative_position_one_hot(buckets: RelativePositionBuckets) -> Tensor:
    residue = F.one_hot(buckets.residue_bucket.long(), num_classes=buckets.residue_size)
    token = F.one_hot(buckets.token_bucket.long(), num_classes=buckets.residue_size)
    same_entity = buckets.same_entity.unsqueeze(-1)
    chain = F.one_hot(buckets.chain_bucket.long(), num_classes=buckets.chain_size)
    return torch.cat([residue, token, same_entity, chain], dim=-1).to(dtype=buckets.dtype)


def _realize_alphafold3_msa(
    msa_features: Tensor,
    msa_mask: Tensor,
    *,
    max_depth: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    r"""
    Independently permute valid MSA rows and gather one aligned realization.
    """

    valid_rows = msa_mask.to(torch.bool).any(dim=-1)
    permutation_keys = torch.rand(valid_rows.shape, device=valid_rows.device, generator=generator)
    permutation = permutation_keys.masked_fill(~valid_rows, -1.0).argsort(dim=-1, descending=True)
    selected_rows = permutation[..., : min(msa_features.shape[-3], max_depth)]

    feature_indices = selected_rows[..., None, None].expand(
        *selected_rows.shape,
        msa_features.shape[-2],
        msa_features.shape[-1],
    )
    mask_indices = selected_rows[..., None].expand(*selected_rows.shape, msa_mask.shape[-1])
    return (
        torch.gather(msa_features, dim=-3, index=feature_indices),
        torch.gather(msa_mask, dim=-2, index=mask_indices),
    )


def _build_alphafold3_local_geometry(
    windowing: LocalAtomWindowing,
    *,
    reference_positions: Tensor,
    reference_space_id: Tensor,
    atom_to_token: Tensor,
    atom_mask: Tensor,
    num_tokens: int,
) -> LocalAtomGeometry:
    r"""
    Build AF3 local atom geometry and key-space ids.
    """

    atom_mask = atom_mask.to(torch.bool)
    num_valid_atoms = int(atom_mask.sum(-1).amax().item())
    position_windows = windowing(reference_positions, num_valid_atoms)

    query_space_id = torch.where(atom_mask, reference_space_id, 0)
    query_space_windows = windowing(query_space_id.unsqueeze(-1), num_valid_atoms)

    # Key-space ids use a dense [token, 24] layout while positions use packed queries.
    token_space_sum = reference_space_id.new_zeros(*reference_space_id.shape[:-1], num_tokens)
    token_atom_count = reference_space_id.new_zeros(*reference_space_id.shape[:-1], num_tokens)
    token_space_sum.scatter_add_(-1, atom_to_token, torch.where(atom_mask, reference_space_id, 0))
    token_atom_count.scatter_add_(-1, atom_to_token, atom_mask.to(reference_space_id.dtype))
    token_space_id = torch.div(token_space_sum, token_atom_count.clamp_min(1), rounding_mode="floor")
    dense_space_id = token_space_id.unsqueeze(-1).expand(*token_space_id.shape, 24).flatten(-2)
    atom_count = atom_to_token.shape[-1]
    key_space_id = dense_space_id[..., :atom_count]
    key_space_windows = windowing(key_space_id.unsqueeze(-1), num_valid_atoms)

    distance_vectors = position_windows.query[..., None, :] - position_windows.key[..., None, :, :]
    distance_mask = (query_space_windows.query.unsqueeze(-2) == key_space_windows.key.unsqueeze(-3)).to(
        reference_positions.dtype
    )
    return LocalAtomGeometry(
        distance_vectors=distance_vectors,
        distance_mask=distance_mask,
        window_info=position_windows.info,
    )


# ipTM contributes 0.8 and pTM contributes 0.2 to the combined score.
_IPTM_WEIGHT = 0.8


def _uniform_bin_centers(num_bins: int, minimum: float, maximum: float) -> Tensor:
    r"""
    Return the midpoints of ``num_bins`` equal bins tiling ``[minimum, maximum]``.
    """

    edges = torch.linspace(minimum, maximum, num_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def _expected_bin_value(logits: Tensor, centers: Tensor) -> Tensor:
    r"""
    Take the softmax expectation of ``logits`` over categorical distance/error ``centers``.
    """

    return logits.softmax(dim=-1) @ centers.to(logits.dtype)


def _contact_probability(distogram_logits: Tensor) -> Tensor:
    r"""
    Probability that a token pair is within 8 Angstrom under the distogram head.
    """

    centers = _uniform_bin_centers(distogram_logits.shape[-1], 2.3125, 21.6875)
    return distogram_logits.softmax(dim=-1)[..., centers < 8.0].sum(dim=-1)


def alphafold3_pae_bin_centers(
    num_bins: int, max_error_bin: float = 31.0, *, device: torch.device | None = None
) -> Tensor:
    r"""
    Return the PAE-bin center distances used to take the expected aligned error.

    The first ``num_bins - 1`` centers evenly tile ``[0, max_error_bin]`` at
    half-step offsets; a final catch-all bin is appended one step beyond.

    Args:
        num_bins: Number of PAE logit bins.
        max_error_bin: Upper edge of the resolved (non-catch-all) bins in angstroms.
        device: Optional device for the returned tensor.

    Returns:
        ``[num_bins]`` bin-center distances in Angstrom.

    Raises:
        ValueError: If ``num_bins`` is less than two.
    """

    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    breaks = torch.linspace(0.0, max_error_bin, num_bins - 1, device=device)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    return torch.cat([centers, centers[-1:] + step], dim=0)


def alphafold3_tm_adjusted_pae(
    pae_logits: Tensor,
    *,
    pair_mask: Tensor,
    chain_index: Tensor,
    interface: bool,
    max_error_bin: float = 31.0,
) -> Tensor:
    r"""
    Return the per-token-pair TM-adjusted PAE from PAE logits.

    Softmaxes the logits into a distance-error distribution, weights each bin by its TM-score term
    ``1 / (1 + (d / d0)^2)``, and sums over bins. The TM-score length scale ``d0`` uses the interface
    token count (both chains of the pair) when ``interface`` is set, else the global
    token count. ipTM and pTM therefore differ only in normalization counts.

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        chain_index: ``[..., N_token]`` chain index per token.
        interface: If ``True``, use the interface token count for ``d0``; else the global count.
        max_error_bin: Upper edge of the resolved PAE bins in angstroms.

    Returns:
        ``[..., N_token, N_token]`` TM-adjusted PAE.
    """

    num_bins = pae_logits.shape[-1]
    centers = alphafold3_pae_bin_centers(num_bins, max_error_bin, device=pae_logits.device).to(pae_logits.dtype)
    probs = torch.softmax(pae_logits.float(), dim=-1).to(pae_logits.dtype)
    pair_mask = pair_mask.to(pae_logits.dtype)

    same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)
    if interface:
        num_chain_tokens = (same_chain.to(pae_logits.dtype) * pair_mask).sum(dim=-1)
        num_tokens = num_chain_tokens.unsqueeze(-1) + num_chain_tokens.unsqueeze(-2)
        # Tokens from one chain contribute once to the interface length.
        num_tokens = num_tokens - same_chain.to(pae_logits.dtype) * torch.floor(num_tokens / 2)
        num_tokens = num_tokens * pair_mask
    else:
        num_tokens = pair_mask.new_full(pair_mask.shape[:-2], float(pair_mask.shape[-1])).unsqueeze(-1).unsqueeze(-1)
        num_tokens = num_tokens.expand_as(pair_mask)

    return (probs * tm_bin_weights(centers, num_tokens)).sum(dim=-1)


def alphafold3_predicted_tm_score(
    adjusted_pae: Tensor, *, pair_mask: Tensor, chain_index: Tensor, interface: bool
) -> Tensor:
    r"""
    Reduce a TM-adjusted PAE matrix to a scalar pTM or ipTM.

    Restricts to cross-chain pairs when ``interface`` is set, row-normalizes the mask, takes the masked
    per-row average of the adjusted PAE, and returns the best row. Returns ``NaN`` when the (interface)
    mask is empty; single-chain systems therefore have no ipTM.

    Args:
        adjusted_pae: ``[..., N_token, N_token]`` TM-adjusted PAE.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        chain_index: ``[..., N_token]`` chain index per token.
        interface: If ``True``, compute ipTM over cross-chain pairs; else pTM over all pairs.

    Returns:
        ``[...]`` pTM or ipTM score.
    """

    mask = pair_mask.to(adjusted_pae.dtype)
    if interface:
        mask = mask * (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(adjusted_pae.dtype)
    normed = mask / (torch.finfo(mask.dtype).tiny + mask.sum(dim=-1, keepdim=True))
    per_alignment = (adjusted_pae * normed).sum(dim=-1)
    score = per_alignment.max(dim=-1).values
    empty = mask.sum(dim=(-2, -1)) == 0
    return torch.where(empty, torch.full_like(score, float("nan")), score)


def alphafold3_confidence_summary(
    pae_logits: Tensor,
    *,
    frame_mask: Tensor,
    chain_index: Tensor,
    max_error_bin: float = 31.0,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute pTM, ipTM, and the ptm_iptm_average from PAE logits.

    The third element is ``0.8 * ipTM + 0.2 * pTM``, with pTM substituted for
    NaN ipTM on single-chain systems. The final ranking score additionally includes
    DSSP disorder and clash terms in [`AlphaFold3Model.fold`][].

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        frame_mask: ``[..., N_token]`` boolean mask of tokens with a valid frame.
        chain_index: ``[..., N_token]`` chain index per token.
        max_error_bin: Upper edge of the resolved PAE bins in angstroms.

    Returns:
        Scalar pTM / ipTM / ptm_iptm_average summaries broadcast over the leading batch/sample axes.
    """

    frame_mask = frame_mask.to(torch.bool)
    pair_mask = frame_mask.unsqueeze(-1) & frame_mask.unsqueeze(-2)
    global_adjusted = alphafold3_tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, chain_index=chain_index, interface=False, max_error_bin=max_error_bin
    )
    interface_adjusted = alphafold3_tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, chain_index=chain_index, interface=True, max_error_bin=max_error_bin
    )
    ptm = alphafold3_predicted_tm_score(global_adjusted, pair_mask=pair_mask, chain_index=chain_index, interface=False)
    iptm = alphafold3_predicted_tm_score(
        interface_adjusted, pair_mask=pair_mask, chain_index=chain_index, interface=True
    )
    ptm_iptm_average = torch.where(torch.isnan(iptm), ptm, _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm)
    return ptm, iptm, ptm_iptm_average


# Per-sample chain-pair and per-chain PAE/PDE summaries over decoded NumPy arrays.


def _weighted_mean(mask: np.ndarray, value: np.ndarray, axis: object) -> np.ndarray:
    r"""
    Compute a contact/frame-weighted mean.
    """

    return np.mean(mask * value, axis=axis) / (np.finfo(value.dtype).tiny + np.mean(mask, axis=axis))


def _weighted_nanmean(value: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    r"""
    Compute a NaN-aware weighted mean; empty slices return NaN.
    """

    mask = mask.astype(float)
    nan_idxs = np.where(np.isnan(value))
    mask_with_nan = mask.copy()
    mask_with_nan[nan_idxs] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(value * mask_with_nan, axis=axis) / np.nanmean(mask_with_nan, axis=axis)


def _chain_pair_pde(full_pde: np.ndarray, chain_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute per-chain-pair mean and minimum predicted distance error.
    """

    unique = np.unique(chain_indices)
    num_chains = len(unique)
    num_samples = full_pde.shape[0]
    mean = np.zeros((num_samples, num_chains, num_chains))
    minimum = np.zeros((num_samples, num_chains, num_chains))
    for first, chain_first in enumerate(unique):
        subset = full_pde[:, chain_indices == chain_first, :]
        for second, chain_second in enumerate(unique):
            block = subset[:, :, chain_indices == chain_second]
            mean[:, first, second] = np.mean(block, axis=(1, 2))
            minimum[:, first, second] = np.min(block, axis=(1, 2))
    return mean, minimum


def _pde_single(
    chain_indices: np.ndarray, full_pde: np.ndarray, contact_probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute contact-weighted intra-chain, cross-chain, and per-token PDE.
    """

    unique = np.unique(chain_indices)
    num_chains = len(unique)
    num_samples = full_pde.shape[0]
    chain_indices_row = chain_indices[None]
    contact = contact_probs[None]
    ichain = np.zeros((num_samples, num_chains))
    xchain = np.zeros((num_samples, num_chains))
    for index, chain_index in enumerate(unique):
        mine = chain_indices_row == chain_index
        intra = mine[:, :, None] * mine[:, None, :] * contact
        cross = mine[:, :, None] * (~mine[:, None, :]) * contact
        ichain[:, index] = _weighted_mean(intra, full_pde, axis=(-2, -1))
        xchain[:, index] = _weighted_mean(cross, full_pde, axis=(-2, -1))
    full_chain = _weighted_mean(contact, full_pde, axis=(-1,))
    return ichain, xchain, full_chain


def _chain_pair_pae(
    full_pae: np.ndarray,
    chain_indices: np.ndarray,
    *,
    mask: np.ndarray,
    contact_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute per-chain-pair mean and minimum predicted aligned error.
    """

    if contact_probs is None:
        contact_probs = np.ones(full_pae.shape[1:], dtype=float)
    unique = np.unique(chain_indices)
    num_chains = len(unique)
    num_samples = full_pae.shape[0]
    mean = np.zeros((num_samples, num_chains, num_chains))
    minimum = np.zeros((num_samples, num_chains, num_chains))
    for first, chain_first in enumerate(unique):
        subset = full_pae[:, chain_indices == chain_first, :]
        subset_mask = mask[chain_indices == chain_first, :]
        subset_contact = contact_probs[chain_indices == chain_first, :]
        for second, chain_second in enumerate(unique):
            block = subset[:, :, chain_indices == chain_second]
            block_mask = subset_mask[:, chain_indices == chain_second]
            block_contact = subset_contact[:, chain_indices == chain_second]
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
    tm_adjusted_pae: np.ndarray, pair_mask: np.ndarray, chain_index: np.ndarray, interface: bool
) -> float:
    r"""
    Compute scalar pTM or ipTM for one sample.
    """

    if interface:
        pair_mask = pair_mask * (chain_index[:, None] != chain_index[None, :])
    if pair_mask.sum() == 0:
        return float("nan")
    normed = pair_mask / (np.finfo(tm_adjusted_pae.dtype).tiny + np.sum(pair_mask, axis=-1, keepdims=True))
    per_alignment = np.sum(tm_adjusted_pae * normed, axis=-1)
    return float(per_alignment.max())


def _chain_pairwise_iptm(tm_adjusted_pae: np.ndarray, pair_mask: np.ndarray, chain_index: np.ndarray) -> np.ndarray:
    r"""
    Compute the per-chain-pair ipTM matrix for one sample.
    """

    unique = list(np.unique(chain_index))
    num_chains = len(unique)
    pairwise = np.zeros((num_chains, num_chains))
    for first, chain_first in enumerate(unique):
        first_mask = chain_index == chain_first
        for offset, chain_second in enumerate(unique[first:]):
            second_mask = chain_index == chain_second
            selected = first_mask | second_mask
            (indices,) = np.where(selected)
            is_interface = chain_first != chain_second
            grid = np.ix_(indices, indices)
            value = _predicted_tm_score_np(tm_adjusted_pae[grid], pair_mask[grid], chain_index[selected], is_interface)
            pairwise[first, first + offset] = value
            pairwise[first + offset, first] = value
    return pairwise


def _reduce_chain_pair(chain_pair: np.ndarray, *, agg_over_col: bool) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Reduce chain pairs to intra-chain and cross-chain means.

    Chain weights are uniform.
    """

    num_samples, num_chains, _ = chain_pair.shape
    ichain = chain_pair.diagonal(axis1=-2, axis2=-1)
    agg_axis = -1 if agg_over_col else -2
    weight = np.ones((num_samples, num_chains, num_chains), dtype=float)
    weight -= np.eye(num_chains, dtype=float)
    xchain = _weighted_nanmean(chain_pair, mask=weight, axis=agg_axis)
    return ichain, xchain


def _pae_metrics(
    chain_indices: np.ndarray,
    full_pae: np.ndarray,
    mask: np.ndarray,
    contact_probs: np.ndarray,
    tm_adjusted_pae_interface: np.ndarray,
) -> dict[str, np.ndarray]:
    r"""
    Compute per-chain ipTM/PAE aggregates and the chain-pair ipTM matrix.
    """

    chain_pair_contact_weighted, _, _ = _chain_pair_pae(full_pae, chain_indices, mask=mask, contact_probs=contact_probs)
    chain_pair_iptm = np.stack(
        [
            _chain_pairwise_iptm(tm_adjusted_pae=sample_tm, chain_index=chain_indices, pair_mask=mask)
            for sample_tm in tm_adjusted_pae_interface
        ],
        axis=0,
    )

    def reduce(chain_pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ichain, xchain_row = _reduce_chain_pair(chain_pair, agg_over_col=False)
        _, xchain_col = _reduce_chain_pair(chain_pair, agg_over_col=True)
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
    r"""
    Compute a contact-weighted PAE ranking metric; higher is better.
    """

    return -np.sum(full_pae * contact_probs[None, :, :], axis=(-2, -1)) / (np.sum(contact_probs) + rank_metric_eps)


# DSSP and geometry terms evaluated after coordinates are decoded.
_FRACTION_DISORDERED_WEIGHT = 0.5
_CLASH_PENALIZATION_WEIGHT = 100.0

# Maximum accessible surface areas from Sander & Rost 1994
# (https://doi.org/10.1002/prot.340200303), used by AlphaFold-RSA.
_MAX_ACCESSIBLE_SURFACE_AREA = {
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

# AlphaFold-RSA rASA disorder threshold.
_RASA_DISORDER_CUTOFF = 0.581
_RASA_WINDOW = 25

_POLYMER_TYPES = frozenset({"protein", "rna", "dna", "dna_rna_hybrid"})


def alphafold3_ranking_score(ptm: float, iptm: float, fraction_disordered: float, has_clash: bool) -> float:
    r"""
    Return the AlphaFold 3 ranking score for one sample.

    ``ptm_iptm_average + 0.5 * fraction_disordered - 100 * has_clash``, with pTM
    substituted for the ipTM term when ipTM is ``NaN``.

    Args:
        ptm: The predicted global TM score.
        iptm: The interface predicted TM score (``NaN`` for single-chain systems).
        fraction_disordered: The AlphaFold-RSA disorder fraction in ``[0, 1]``.
        has_clash: Whether the structure has at least one clashing chain.

    Returns:
        The scalar ranking score (higher is better).
    """

    if np.isnan(iptm):
        ptm_iptm_average = ptm
    else:
        ptm_iptm_average = _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm
    return (
        ptm_iptm_average
        + _FRACTION_DISORDERED_WEIGHT * fraction_disordered
        - _CLASH_PENALIZATION_WEIGHT * float(has_clash)
    )


def alphafold3_fraction_disordered(structure: Structure, rasa_disorder_cutoff: float = _RASA_DISORDER_CUTOFF) -> float:
    r"""
    Fraction of protein residues predicted disordered by windowed AlphaFold-RSA.

    Restricts to protein chains, computes a windowed (window 25) relative solvent
    accessible surface area per residue with DSSP 4.4.7 semantics, and returns the mean
    fraction of residues whose rASA exceeds ``rasa_disorder_cutoff``. Repeated
    chain sequences reuse a cached rASA profile. Calculation errors propagate
    rather than being reinterpreted as an
    absent chain.

    Args:
        structure: The decoded per-sample structure to score.
        rasa_disorder_cutoff: The rASA threshold above which a residue counts as
            disordered.

    Returns:
        The fraction of protein residues predicted disordered, in ``[0, 1]``.
    """

    rasa: list[float] = []
    sequence_rasa: dict[str, np.ndarray] = {}
    for instance_id in _protein_instance_ids(structure):
        chain_sequence = _chain_sequence(structure, instance_id)
        cached = sequence_rasa.get(chain_sequence)
        if cached is not None:
            rasa.extend(cached.tolist())
            continue
        per_residue = _windowed_solvent_accessible_area(structure, instance_id)
        sequence_rasa[chain_sequence] = per_residue
        rasa.extend(per_residue.tolist())

    if not rasa:
        return 0.0
    return float(np.mean(np.asarray(rasa) > rasa_disorder_cutoff))


def alphafold3_has_clash(
    structure: Structure,
    cutoff_radius: float = 1.1,
    min_clashes_for_overlap: int = 100,
    min_fraction_for_overlap: float = 0.5,
) -> bool:
    r"""
    Whether the structure has at least one clashing polymer chain.

    Restricts to protein / RNA / DNA polymer atoms, then flags a chain as
    overlapping when more than ``min_clashes_for_overlap`` of its atoms lie within
    ``cutoff_radius`` Angstrom of a polymer atom in another chain (or a residue more
    than one apart in the same chain), or when more than ``min_fraction_for_overlap``
    of the chain's atoms clash.

    Args:
        structure: The decoded per-sample structure to score.
        cutoff_radius: Atom-pair distances below this (Angstrom) count as a clash.
        min_clashes_for_overlap: Minimum clashing atoms for a chain to overlap.
        min_fraction_for_overlap: Minimum clashing fraction for a chain to overlap.

    Returns:
        ``True`` if at least one polymer chain overlaps another.
    """

    coordinates, residue_indices, instance_ids = _polymer_atoms(structure)
    if coordinates.shape[0] == 0:
        return False

    neighbors = _query_ball(coordinates, cutoff_radius)
    per_atom_has_clash = np.zeros(coordinates.shape[0], dtype=bool)
    for atom_index, clashing in enumerate(neighbors):
        for other in clashing:
            if abs(int(residue_indices[atom_index]) - int(residue_indices[other])) > 1 or (
                instance_ids[atom_index] != instance_ids[other]
            ):
                per_atom_has_clash[atom_index] = True
                break

    for instance_id in np.unique(instance_ids):
        mask = instance_ids == instance_id
        num_atoms = int(np.sum(mask))
        if num_atoms == 0:
            continue
        num_clashes = int(np.sum(per_atom_has_clash & mask))
        fraction_clashes = num_clashes / num_atoms
        if num_clashes > min_clashes_for_overlap or fraction_clashes > min_fraction_for_overlap:
            return True
    return False


def _windowed_solvent_accessible_area(structure: Structure, instance_id: str) -> np.ndarray:
    r"""
    Windowed per-residue rASA for one protein chain (AlphaFold-RSA).

    Runs the DSSP 4.4.7 solvent-accessibility algorithm over the chain, normalizes each residue's
    accessible surface area by [`_MAX_ACCESSIBLE_SURFACE_AREA`][], clips to ``1.0``, and
    averages over a reflect-padded window.
    Raises ``RuntimeError`` when the chain has no complete DSSP residues.
    """

    accessible = _dssp_accessible_area(structure, instance_id)
    if not accessible:
        raise RuntimeError("DSSP produced no accessible-area rows")
    rasa = []
    for residue_name, area in accessible:
        max_area = _MAX_ACCESSIBLE_SURFACE_AREA.get(residue_name, _MAX_ACCESSIBLE_SURFACE_AREA["ALA"])
        normalized = area / max_area
        rasa.append(min(normalized, 1.0))

    half_window = (_RASA_WINDOW - 1) // 2
    padded = np.pad(np.asarray(rasa, dtype=float), (half_window, half_window), "reflect")
    return np.convolve(padded, np.ones(_RASA_WINDOW), "valid") / _RASA_WINDOW


# DSSP 4.4.7 accessibility parameters.
_DSSP_ATOM_RADII = {"N": 1.65, "CA": 1.87, "C": 1.76, "O": 1.4}
_DSSP_SIDE_CHAIN_RADIUS = 1.8
_DSSP_WATER_RADIUS = 1.4
_DSSP_SURFACE_POINT_COUNT = 401


def _dssp_surface_dots() -> np.ndarray:
    r"""
    Return the 401-point Fibonacci sphere used by DSSP 4.4.7.
    """

    n = (_DSSP_SURFACE_POINT_COUNT - 1) // 2
    indices = np.arange(-n, n + 1, dtype=np.int32)
    golden_ratio = np.float32((1.0 + np.sqrt(np.float32(5.0))) / 2.0)
    latitude = np.arcsin(np.float32(2.0) * indices.astype(np.float32) / np.float32(_DSSP_SURFACE_POINT_COUNT))
    longitude = np.asarray(
        np.fmod(indices.astype(np.float64), np.float64(golden_ratio)) * 2.0 * np.pi / np.float64(golden_ratio),
        dtype=np.float32,
    )
    return np.stack(
        (
            np.sin(longitude) * np.cos(latitude),
            np.cos(longitude) * np.cos(latitude),
            np.sin(latitude),
        ),
        axis=-1,
    ).astype(np.float32)


_DSSP_SURFACE_DOTS = _dssp_surface_dots()
_DSSP_SURFACE_POINT_WEIGHT = np.float32(4.0 * np.pi / _DSSP_SURFACE_POINT_COUNT)


def _dssp_accessible_area(structure: Structure, instance_id: str) -> list[tuple[str, float]]:
    r"""
    Return per-residue ``(residue_name, accessible_area)`` for one protein chain.

    This implements the DSSP 4.4.7 accessibility calculation with its atom radii,
    401-point Fibonacci sphere, water radius, overlap test, and integer output
    rounding. DSSP drops residues that
    do not contain all four backbone atoms (N, CA, C, O), then scores each retained
    residue against the complete retained chain in isolation.
    """

    residues: dict[int, tuple[str, dict[str, np.ndarray], list[np.ndarray]]] = {}
    system = structure.system
    for index, site in enumerate(structure.sites):
        if site.instance_id != instance_id:
            continue
        if not bool(structure.coordinate_mask[index]):
            continue
        atom = system.atom(site)
        if atom.element.upper() == "H":
            continue
        component = system.component((instance_id, site.component_index))
        coordinate = np.asarray(structure.coordinates[index], dtype=np.float32)
        residue_name, backbone, side_chain = residues.setdefault(
            site.component_index,
            (component.component_id, {}, []),
        )
        if site.atom_id in _DSSP_ATOM_RADII:
            backbone[site.atom_id] = coordinate
        else:
            side_chain.append(coordinate)

    complete_residues = [
        (component_index, residue_name, backbone, side_chain)
        for component_index, (residue_name, backbone, side_chain) in sorted(residues.items())
        if backbone.keys() >= _DSSP_ATOM_RADII.keys()
    ]
    if not complete_residues:
        raise RuntimeError("protein chain has no complete residues")

    atom_coordinates: list[np.ndarray] = []
    atom_radii: list[float] = []
    atom_residue_indices: list[int] = []
    residue_names: list[str] = []
    for residue_index, (_, residue_name, backbone, side_chain) in enumerate(complete_residues):
        residue_names.append(residue_name)
        for atom_name in ("N", "CA", "C", "O"):
            atom_coordinates.append(backbone[atom_name])
            atom_radii.append(_DSSP_ATOM_RADII[atom_name])
            atom_residue_indices.append(residue_index)
        atom_coordinates.extend(side_chain)
        atom_radii.extend([_DSSP_SIDE_CHAIN_RADIUS] * len(side_chain))
        atom_residue_indices.extend([residue_index] * len(side_chain))

    coordinates = np.asarray(atom_coordinates, dtype=np.float32)
    radii = np.asarray(atom_radii, dtype=np.float32)
    residue_indices = np.asarray(atom_residue_indices, dtype=np.int64)
    atom_areas = np.empty(coordinates.shape[0], dtype=np.float32)
    surface_radii = radii + np.float32(_DSSP_WATER_RADIUS)
    occlusion_radii = radii.astype(np.float64) + np.float64(np.float32(_DSSP_WATER_RADIUS))

    for atom_index, (coordinate, surface_radius, occlusion_radius) in enumerate(
        zip(coordinates, surface_radii, occlusion_radii)
    ):
        relative = coordinates - coordinate
        center_distances_sq = np.sum(relative * relative, axis=-1, dtype=np.float32)
        candidate_mask = (center_distances_sq.astype(np.float64) > 0.0001) & (
            center_distances_sq.astype(np.float64) < np.square(occlusion_radius + occlusion_radii)
        )
        candidates = relative[candidate_mask]
        candidate_radii_sq = np.square(occlusion_radii[candidate_mask])
        surface_points = _DSSP_SURFACE_DOTS * surface_radius
        if candidates.shape[0] == 0:
            free_count = _DSSP_SURFACE_POINT_COUNT
        else:
            point_distances_sq = np.sum(
                np.square(surface_points[:, None, :] - candidates[None, :, :]),
                axis=-1,
                dtype=np.float32,
            )
            free_count = int(np.sum(np.all(candidate_radii_sq < point_distances_sq.astype(np.float64), axis=-1)))
        surface = np.float32(0.0)
        for _ in range(free_count):
            surface = np.float32(surface + _DSSP_SURFACE_POINT_WEIGHT)
        atom_areas[atom_index] = np.float32(np.float32(surface * surface_radius) * surface_radius)

    rows = []
    for residue_index, residue_name in enumerate(residue_names):
        area = np.float32(0.0)
        for atom_area in atom_areas[residue_indices == residue_index]:
            area = np.float32(area + atom_area)
        rows.append((residue_name, float(np.floor(area + np.float32(0.5)))))
    return rows


def _protein_instance_ids(structure: Structure) -> list[str]:
    r"""
    Instance ids of protein polymer chains, in system order.
    """

    system = structure.system
    instance_ids: list[str] = []
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            instance_ids.append(instance_id)
    return instance_ids


def _chain_sequence(structure: Structure, instance_id: str) -> str:
    r"""
    Return the protein sequence used by the cross-chain rASA cache.
    """

    entity = structure.system.entities[structure.system.instances[instance_id]]
    return "".join(AMINO_ACID_THREE_TO_ONE.get(component.component_id, "X") for component in entity.components)


def _polymer_atoms(structure: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Observed polymer atom coordinates, component indices, and instance IDs.
    """

    system = structure.system
    coordinates: list[np.ndarray] = []
    residue_indices: list[int] = []
    instance_ids: list[str] = []
    for index, site in enumerate(structure.sites):
        if not bool(structure.coordinate_mask[index]):
            continue
        entity = system.entities[system.instances[site.instance_id]]
        if not isinstance(entity, PolymerSequence) or entity.polymer_type not in _POLYMER_TYPES:
            continue
        coordinates.append(np.asarray(structure.coordinates[index], dtype=np.float64))
        residue_indices.append(site.component_index)
        instance_ids.append(site.instance_id)
    if not coordinates:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=object)
    return (
        np.stack(coordinates, axis=0),
        np.asarray(residue_indices, dtype=np.int64),
        np.asarray(instance_ids, dtype=object),
    )


def _query_ball(coordinates: np.ndarray, cutoff_radius: float) -> list[list[int]]:
    r"""
    Find atom neighbors within ``cutoff_radius`` using a cKDTree.
    """

    from scipy import spatial

    tree = spatial.cKDTree(coordinates)
    return tree.query_ball_point(coordinates, p=2.0, r=cutoff_radius)


@tensorclass
class TemplateFeatures:
    r"""
    Template features consumed by [`AlphaFold3TemplateEmbedder`][].

    Every tensor carries a ``num_templates`` axis.

    Args:
        distogram: ``[..., num_templates, N_token, N_token, num_distogram_bins]`` template pair distogram
            features.
        pseudo_beta_mask: ``[..., num_templates, N_token, N_token]`` template pseudo-beta validity mask.
        residue_type_ids: Integer template residue type ids ``[..., num_templates, N_token]``.
        unit_vector: ``[..., num_templates, N_token, N_token, 3]`` template pair unit-vector features.
        backbone_frame_mask: ``[..., num_templates, N_token, N_token]`` template backbone-frame validity mask.
        template_mask: Optional ``[..., num_templates]`` per-template validity mask.
    """

    distogram: Tensor
    pseudo_beta_mask: Tensor
    residue_type_ids: Tensor
    unit_vector: Tensor
    backbone_frame_mask: Tensor
    template_mask: Tensor | None = None


@tensorclass(frozen=True)
class _AlphaFold3TrunkOutput:
    r"""
    Named representations returned by the AlphaFold 3 recycling trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets


@tensorclass
class AlphaFold3Input:
    r"""
    Tensor and molecular-layout input for AlphaFold 3.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, ``M`` MSA depth, and
    ``T`` template count.

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
        msa_features: MSA features with shape ``(B, M, N, C_msa)``.
        msa_mask: MSA validity mask with shape ``(B, M, N)``.
        template_features: Optional features for ``T`` structural templates.
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
    msa_features: Tensor
    msa_mask: Tensor
    template_features: TemplateFeatures | None = None
    system: MolecularSystem | None = None
    atom_sites: tuple[AtomSite, ...] = ()
    token_sites: tuple[AtomSite | ComponentReference, ...] = ()


@tensorclass
class AlphaFold3Output:
    r"""
    Predictions from one AlphaFold 3 denoising call.

    Args:
        single: Final trunk single representation with shape ``(B, N, 384)``.
        pair: Final trunk pair representation with shape ``(B, N, N, 128)``.
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

    single: Tensor
    pair: Tensor
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
