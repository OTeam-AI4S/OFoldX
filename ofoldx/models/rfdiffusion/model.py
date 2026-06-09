r"""
RfDiffusion design model.
"""

from __future__ import annotations

import math
from collections import namedtuple
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from oops import se3_basis, segment_softmax_csr, segment_sum_csr, weighted_tensor_product
from opt_einsum import contract as einsum
from torch import Tensor, nn

from ..contract import BaseInput, DesignGeneratorOutput, StructurePredictionOutput
from ..mixin import ModelMixin
from ..registry import DESIGN_GENERATOR_MODELS, MODELS
from .config import RfDiffusionConfig

if TYPE_CHECKING:
    from .sampler import NativeRfDiffusionSampler


@MODELS.register("rfdiffusion")
@DESIGN_GENERATOR_MODELS.register("rfdiffusion")
class RfDiffusionDesignGenerator(ModelMixin):
    r"""
    Denoising-diffusion model that generates novel protein backbones from a noised structure (RfDiffusion).

    Public entry point for the RfDiffusion architecture. Holds the RoseTTAFold network
    ([`RoseTTAFoldModule`][ofoldx.models.rfdiffusion.model.RoseTTAFoldModule]) as its ``network``
    submodule and delegates the diffusion sampling loop to a lazily-constructed
    [`NativeRfDiffusionSampler`][ofoldx.models.rfdiffusion.sampler.NativeRfDiffusionSampler]. Given a
    tokenized [`RfDiffusionInput`][ofoldx.models.rfdiffusion.model.RfDiffusionInput], it denoises one or
    more backbone designs.

    Args:
        config: Model configuration; defaults to a fresh
            [`RfDiffusionConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionConfig] (the reference
            ``base`` architecture) when omitted.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/rfdiffusion")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/rfdiffusion")  # doctest: +SKIP
        >>> output = model(processor("target.pdb"))  # doctest: +SKIP
    """

    config_class = RfDiffusionConfig

    def __init__(self, config: RfDiffusionConfig | None = None) -> None:
        config = RfDiffusionConfig() if config is None else config
        super().__init__(config)
        model, preprocess = config.model, config.preprocess
        self.network = RoseTTAFoldModule(
            num_extra_blocks=int(model.num_extra_blocks),
            num_main_blocks=int(model.num_main_blocks),
            num_refinement_blocks=int(model.num_refinement_blocks),
            msa_size=int(model.msa_size),
            full_msa_size=int(model.full_msa_size),
            pair_size=int(model.pair_size),
            template_size=int(model.template_size),
            num_msa_attention_heads=int(model.num_msa_attention_heads),
            num_pair_attention_heads=int(model.num_pair_attention_heads),
            num_template_attention_heads=int(model.num_template_attention_heads),
            hidden_size=int(model.hidden_size),
            template_hidden_size=int(model.template_hidden_size),
            dropout=float(model.dropout),
            template_1d_size=int(preprocess.template_1d_size),
            template_2d_size=int(preprocess.template_2d_size),
            freeze_track_motif=bool(model.freeze_track_motif),
            se3_param_full=dict(model.se3_param_full),
            se3_param_topk=dict(model.se3_param_topk),
            vocab_size=int(model.vocab_size),
        )
        self.post_init()

    def _init_module_weights(self, module: nn.Module) -> None:
        return None

    def forward(
        self,
        batch: RfDiffusionInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        diffusion_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Generate backbone-design candidates by running the diffusion sampler.

        Runs ``num_samples`` independent trajectories and stacks their coords, sequence tokens, and pLDDT
        scores into a design output. ``recycling_steps`` is accepted for API symmetry with structure
        predictors but is unused (RfDiffusion has no recycling loop).

        Args:
            batch: A tokenized [`RfDiffusionInput`][ofoldx.models.rfdiffusion.model.RfDiffusionInput].
            recycling_steps: Unused; accepted for interface compatibility.
            num_samples: Number of independent design trajectories to sample.
            num_sampling_steps: Number of denoising steps to run, capped at the diffuser's timestep count.
            diffusion_steps: Override for the diffuser's total timestep count.
            generator: Optional RNG for reproducible sampling.

        Returns:
            A [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput] carrying stacked
            ``pred_coords`` and ``sequence_tokens``, optional pLDDT scores, and per-sample metadata.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("oteam/rfdiffusion")  # doctest: +SKIP
            >>> output = model(processor("target.pdb"), num_samples=2)  # doctest: +SKIP
        """

        del recycling_steps
        sampler = self._sampler()
        results = [
            sampler.sample(
                batch,
                num_sampling_steps=num_sampling_steps,
                diffusion_steps=diffusion_steps,
                generator=generator,
            )
            for _ in range(num_samples)
        ]
        coords = torch.stack([result.coords for result in results])
        sequence_tokens = torch.stack([result.sequence_tokens for result in results])
        distogram_logits = torch.empty(0, dtype=coords.dtype, device=coords.device)
        plddt = [result.plddt for result in results if result.plddt is not None]
        scores = {"plddt": torch.stack(plddt)} if plddt else None
        metadata = {
            "architecture": "rfdiffusion",
            "backend": "native_rfdiffusion_sampler",
            "num_samples": num_samples,
            "samples": [result.metadata for result in results],
        }
        return DesignGeneratorOutput(
            sequence_tokens=sequence_tokens,
            structure_prediction=StructurePredictionOutput(
                pred_coords=coords,
                distogram_logits=distogram_logits,
            ),
            scores=scores,
            metadata=metadata,
        )

    @torch.no_grad()
    def generate(
        self,
        batch: RfDiffusionInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        diffusion_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Generate design output with gradients disabled.

        Convenience wrapper that calls
        [`forward`][ofoldx.models.rfdiffusion.model.RfDiffusionDesignGenerator.forward] under
        [torch.no_grad][], for inference where autograd is unnecessary.

        Args:
            batch: A tokenized [`RfDiffusionInput`][ofoldx.models.rfdiffusion.model.RfDiffusionInput].
            recycling_steps: Unused; accepted for interface compatibility.
            num_samples: Number of independent design trajectories to sample.
            num_sampling_steps: Number of denoising steps to run.
            diffusion_steps: Override for the diffuser's total timestep count.
            generator: Optional RNG for reproducible sampling.

        Returns:
            The same [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput] that
            ``forward`` produces.

        Examples:
            >>> output = model.generate(batch, num_samples=2)  # doctest: +SKIP
        """

        return self.forward(
            batch,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            diffusion_steps=diffusion_steps,
            generator=generator,
        )

    def sample(
        self,
        batch: RfDiffusionInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        diffusion_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Sample designs and return only the stacked backbone coords.

        Like [`forward`][ofoldx.models.rfdiffusion.model.RfDiffusionDesignGenerator.forward] but returns a
        bare coord tensor rather than a full output object.

        Args:
            batch: A tokenized [`RfDiffusionInput`][ofoldx.models.rfdiffusion.model.RfDiffusionInput].
            recycling_steps: Unused; accepted for interface compatibility.
            num_samples: Number of independent design trajectories to sample.
            num_sampling_steps: Number of denoising steps to run.
            diffusion_steps: Override for the diffuser's total timestep count.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Stacked design coords of shape `[num_samples, L, 14, 3]`.

        Examples:
            >>> coords = model.sample(batch, num_samples=2)  # doctest: +SKIP
        """

        del recycling_steps
        sampler = self._sampler()
        return torch.stack(
            [
                sampler.sample(
                    batch,
                    num_sampling_steps=num_sampling_steps,
                    diffusion_steps=diffusion_steps,
                    generator=generator,
                ).coords
                for _ in range(num_samples)
            ]
        )

    def _sampler(self) -> NativeRfDiffusionSampler:
        r"""Construct the native sampler bound to this model's config and network (imported lazily)."""

        from .sampler import NativeRfDiffusionSampler

        return NativeRfDiffusionSampler(self.config, network=self.network)


class RoseTTAFoldModule(nn.Module):
    r"""
    RoseTTAFold trunk driving one RfDiffusion denoising step.

    Ported RoseTTAFold network: embeds the MSA/pair/state representations, injects template and recycling
    features, runs the [`IterativeSimulator`][ofoldx.models.rfdiffusion.model.IterativeSimulator] track,
    and reads out amino-acid logits, pLDDT, and the denoised backbone. The
    [`NativeRfDiffusionSampler`][ofoldx.models.rfdiffusion.sampler.NativeRfDiffusionSampler] calls this
    once per diffusion timestep.

    Args:
        num_extra_blocks: Number of extra-MSA IterBlocks.
        num_main_blocks: Number of main IterBlocks.
        num_refinement_blocks: Number of top-k structure-refinement iterations.
        msa_size: Width of the latent MSA representation.
        full_msa_size: Width of the extra (full) MSA representation.
        pair_size: Width of the pair representation.
        template_size: Width of the template pair representation.
        num_msa_attention_heads: Number of MSA attention heads.
        num_pair_attention_heads: Number of pair attention heads.
        num_template_attention_heads: Number of template attention heads.
        hidden_size: Per-head hidden width shared across the attention stacks.
        template_hidden_size: Per-head hidden width used inside the template stack.
        dropout: Dropout probability applied throughout the trunk.
        template_1d_size: Width of the 1D template feature.
        template_2d_size: Width of the 2D template feature.
        freeze_track_motif: Whether to stop structure updates on motif residues in the track.
        se3_param_full: SE(3) sizing dict for the main-block structure module.
        se3_param_topk: SE(3) sizing dict for the top-k refinement structure module.
        vocab_size: Number of amino-acid tokens the sequence head predicts over.
    """

    def __init__(
        self,
        num_extra_blocks: int,
        num_main_blocks: int,
        num_refinement_blocks: int,
        msa_size: int,
        full_msa_size: int,
        pair_size: int,
        template_size: int,
        num_msa_attention_heads: int,
        num_pair_attention_heads: int,
        num_template_attention_heads: int,
        hidden_size: int,
        template_hidden_size: int,
        dropout: float,
        template_1d_size: int,
        template_2d_size: int,
        freeze_track_motif: bool,
        se3_param_full: dict | None = None,
        se3_param_topk: dict | None = None,
        vocab_size: int = 21,
    ) -> None:
        super().__init__()
        se3_full = _se3_params(se3_param_full)
        se3_topk = _se3_params(se3_param_topk)
        state_size = se3_topk["l0_out_features"]

        self.freeze_track_motif = freeze_track_motif
        self.latent_emb = MsaEmbedding(
            msa_size=msa_size,
            pair_size=pair_size,
            state_size=state_size,
            dropout=dropout,
        )
        self.full_emb = ExtraMsaEmbedding(msa_size=full_msa_size, input_feature_size=25, dropout=dropout)
        self.templ_emb = TemplateEmbedding(
            pair_size=pair_size,
            template_size=template_size,
            state_size=state_size,
            num_attention_heads=num_template_attention_heads,
            hidden_size=template_hidden_size,
            dropout=0.25,
            template_1d_size=template_1d_size,
            template_2d_size=template_2d_size,
        )
        self.recycle = Recycling(msa_size=msa_size, pair_size=pair_size, state_size=state_size)
        self.simulator = IterativeSimulator(
            num_extra_blocks=num_extra_blocks,
            num_main_blocks=num_main_blocks,
            num_refinement_blocks=num_refinement_blocks,
            msa_size=msa_size,
            full_msa_size=full_msa_size,
            pair_size=pair_size,
            hidden_size=hidden_size,
            num_msa_attention_heads=num_msa_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            se3_param_full=se3_full,
            se3_param_topk=se3_topk,
            dropout=dropout,
        )
        self.coords_6d_pred = DistanceNetwork(pair_size, dropout=dropout)
        self.aa_pred = MaskedTokenNetwork(msa_size, vocab_size=vocab_size)
        self.lddt_pred = LDDTNetwork(state_size)
        self.exp_pred = ExpResolvedNetwork(msa_size, state_size)

    def forward(
        self,
        msa_latent: Tensor,
        msa_full: Tensor,
        seq: Tensor,
        coords: Tensor,
        idx: Tensor,
        t: Tensor | None,
        template_1d: Tensor | None = None,
        template_2d: Tensor | None = None,
        template_coords: Tensor | None = None,
        alpha_t: Tensor | None = None,
        msa_prev: Tensor | None = None,
        pair_prev: Tensor | None = None,
        state_prev: Tensor | None = None,
        motif_mask: Tensor | None = None,
        cyclic_reses: Tensor | None = None,
    ):
        r"""
        Run one RoseTTAFold denoising pass over the noisy structure.

        Embeds inputs, folds in recycling and template features, runs the iterative track, applies the final
        predicted frames to the local backbone, and reads out the auxiliary heads.

        Args:
            msa_latent: Latent MSA features `[B, N, L, msa_size]`.
            msa_full: Extra (full) MSA features `[B, N_full, L, ...]`.
            seq: One-hot sequence features `[B, L, 22]`.
            coords: Noisy backbone coords `[B, L, A, 3]`.
            idx: Residue-index tensor `[B, L]` used for relative-position features.
            t: Diffusion timestep (unused here; time features are injected by the sampler).
            template_1d: 1D template features; required for inference.
            template_2d: 2D template features; required for inference.
            template_coords: Template backbone coords; required for inference.
            alpha_t: Template torsion features; required for inference.
            msa_prev: Recycled MSA state, or `None` to start from zeros.
            pair_prev: Recycled pair state, or `None` to start from zeros.
            state_prev: Recycled node state, or `None` to start from zeros.
            motif_mask: Per-residue motif mask `[L]`; motif residues are frozen when configured.
            cyclic_reses: Optional cyclic-residue mask for cyclic-peptide relative positions.

        Returns:
            A 7-tuple ``(msa_row0, pair, final_coords, state, alpha, aa_logits, plddt)`` where ``final_coords``
            is the denoised backbone, ``aa_logits`` are amino-acid logits, and ``plddt`` is the per-residue
            confidence.

        Raises:
            ValueError: If any required template feature is `None`.
        """

        batch_size, _, length = msa_latent.shape[:3]
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, cyclic_reses)
        msa_full = self.full_emb(msa_full, seq, idx)

        if msa_prev is None:
            msa_prev = torch.zeros_like(msa_latent[:, 0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)
        msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, coords, state_prev)
        msa_latent[:, 0] = msa_latent[:, 0] + msa_recycle.reshape(batch_size, length, -1)
        pair = pair + pair_recycle
        state = state + state_recycle

        # Time features are injected by the sampler's preprocess(); no timestep_embedder here.
        if template_1d is None or template_2d is None or alpha_t is None or template_coords is None:
            raise ValueError(
                "template features template_1d, template_2d, alpha_t, and template_coords "
                "are required for RFdiffusion inference"
            )
        pair, state = self.templ_emb(template_1d, template_2d, alpha_t, template_coords, pair, state)
        if motif_mask is None:
            motif_mask = torch.zeros(length, dtype=torch.bool, device=msa_latent.device)
        frozen_mask = motif_mask if self.freeze_track_motif else torch.zeros_like(motif_mask)
        msa, pair, rotations, translations, alpha_s, state = self.simulator(
            seq,
            msa_latent,
            msa_full,
            pair,
            coords[:, :, :3],
            state,
            idx,
            motif_mask=frozen_mask,
            cyclic_reses=cyclic_reses,
        )

        local = coords[:, :, :3] - coords[:, :, 1].unsqueeze(-2)
        final_coords = _apply_frames(rotations[-1], translations[-1], local)

        logits_aa = self.aa_pred(msa)
        lddt_logits = self.lddt_pred(state)
        return (
            msa[:, 0],
            pair,
            final_coords,
            state,
            alpha_s[-1],
            logits_aa.permute(0, 2, 1),
            _plddt_expectation(lddt_logits),
        )


def _plddt_expectation(logits: Tensor) -> Tensor:
    r"""Convert per-bin pLDDT logits `[B, num_bins, L]` into the expected pLDDT value `[B, L]`."""

    num_bins = logits.shape[1]
    bins = torch.linspace(1.0 / num_bins, 1.0, num_bins, dtype=logits.dtype, device=logits.device)
    return torch.sum(torch.softmax(logits, dim=1) * bins[None, :, None], dim=1)


def _chain_ids(idx: torch.Tensor, threshold: int = 35) -> torch.Tensor:
    r"""Assign a chain id per residue by detecting large jumps (> ``threshold``) in the residue index."""

    idx_np = idx.detach().cpu().numpy()
    chain_ids = np.zeros(idx_np.shape, dtype=np.int64)
    for batch_index, residue_idx in enumerate(idx_np):
        for chain_index, start in enumerate(find_breaks(residue_idx, thresh=threshold), start=1):
            chain_ids[batch_index, start:] = chain_index
    return torch.as_tensor(chain_ids, device=idx.device)


def _cyclic_mask(cyclize: torch.Tensor | None, idx: torch.Tensor) -> torch.Tensor | None:
    r"""Broadcast a cyclic-residue mask to `[B, L]` on ``idx``'s device, or return `None` when unset."""

    if cyclize is None:
        return None
    cyclize = cyclize.to(device=idx.device, dtype=torch.bool)
    if cyclize.ndim == 1:
        cyclize = cyclize.unsqueeze(0).expand(idx.shape[0], -1)
    return cyclize


def _one_hot_lookup(one_hot: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    r"""Look up embedding rows via a one-hot matmul, so soft (non-integer) sequence inputs are supported."""

    return torch.matmul(one_hot, embedding.weight)


def _zero_bias(linear: nn.Linear) -> None:
    r"""Zero the bias of a linear layer in place."""

    nn.init.zeros_(linear.bias)


def _pseudo_cbeta(coords: torch.Tensor) -> torch.Tensor:
    r"""Reconstruct the pseudo-C-beta position from backbone N/CA/C coords `[B, L, A, 3]`."""

    nitrogen, calpha, carbon = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
    ca_to_n = calpha - nitrogen
    c_to_ca = carbon - calpha
    normal = torch.cross(ca_to_n, c_to_ca, dim=-1)
    return -0.58273431 * normal + 0.56802827 * ca_to_n - 0.54067466 * c_to_ca + calpha


class PositionalEncoding2D(nn.Module):
    r"""
    Relative-position embedding added to the pair representation.

    Buckets the signed residue separation into ``minpos..maxpos`` bins and adds a learned embedding to each
    pair entry. Cyclic chains wrap the separation around the cycle length so cyclic-peptide topologies are
    encoded correctly.

    Args:
        hidden_size: Width of the pair representation the embedding is added to.
        minpos: Minimum (most negative) relative position bin.
        maxpos: Maximum relative position bin.
        dropout: Dropout probability applied to the summed pair features.
    """

    def __init__(self, hidden_size: int, minpos: int = -32, maxpos: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos) + maxpos + 1
        self.emb = nn.Embedding(self.nbin, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, pair: torch.Tensor, idx: torch.Tensor, cyclize: torch.Tensor | None = None) -> torch.Tensor:
        r"""
        Add the bucketed relative-position embedding to ``pair``.

        Args:
            pair: Pair representation `[B, L, L, hidden_size]`.
            idx: Residue-index tensor `[B, L]`.
            cyclize: Optional cyclic-residue mask; wraps separations within a cyclic chain.

        Returns:
            The pair representation with the relative-position embedding added, dropout applied.
        """

        boundaries = torch.arange(self.minpos, self.maxpos, device=pair.device)
        seqsep = idx[:, None, :] - idx[:, :, None]
        cyclic_residues = _cyclic_mask(cyclize, idx)

        if cyclic_residues is not None and cyclic_residues.any():
            chain_ids = _chain_ids(idx)
            for batch_index in range(idx.shape[0]):
                for chain_id in torch.unique(chain_ids[batch_index]):
                    is_cyclic_chain = cyclic_residues[batch_index] & (chain_ids[batch_index] == chain_id)
                    cycle_length = int(is_cyclic_chain.sum().item())
                    if cycle_length < 2:
                        continue
                    same_cycle = is_cyclic_chain[:, None] & is_cyclic_chain[None, :]
                    seqsep_slice = seqsep[batch_index]
                    seqsep_slice[same_cycle & (seqsep_slice > cycle_length // 2)] -= cycle_length
                    seqsep_slice[same_cycle & (seqsep_slice < -cycle_length // 2)] += cycle_length

        relpos = torch.bucketize(seqsep, boundaries).long()
        return self.drop(pair + self.emb(relpos))


class MsaEmbedding(nn.Module):
    r"""
    Embed the latent MSA, pair, and node-state representations from sequence features.

    Projects the raw MSA features and adds a per-residue query embedding, builds the initial pair
    representation from left/right sequence embeddings plus relative positions, and produces the initial node
    state. This is the entry embedding of the RoseTTAFold trunk.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        state_size: Width of the node-state representation.
        input_feature_size: Width of the raw per-position MSA input features.
        minpos: Minimum relative-position bin for the 2D positional encoding.
        maxpos: Maximum relative-position bin for the 2D positional encoding.
        dropout: Dropout probability applied to the embedded features.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        state_size: int = 32,
        input_feature_size: int = 22 + 22 + 2 + 2,
        minpos: int = -32,
        maxpos: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb = nn.Linear(input_feature_size, msa_size)
        self.emb_q = nn.Embedding(22, msa_size)
        self.emb_left = nn.Embedding(22, pair_size)
        self.emb_right = nn.Embedding(22, pair_size)
        self.emb_state = nn.Embedding(22, state_size)
        self.drop = nn.Dropout(dropout)
        self.pos = PositionalEncoding2D(pair_size, minpos=minpos, maxpos=maxpos, dropout=dropout)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the embedding weights to the released RoseTTAFold scheme (Lecun-normal, zero bias)."""

        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)
        self.emb_state = init_lecun_normal(self.emb_state)
        _zero_bias(self.emb)

    def forward(
        self,
        msa: torch.Tensor,
        seq: torch.Tensor,
        idx: torch.Tensor,
        cyclize: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Embed the MSA, pair, and node-state tensors.

        Args:
            msa: Raw MSA features `[B, N, L, input_feature_size]`.
            seq: One-hot sequence features `[B, L, 22]`.
            idx: Residue-index tensor `[B, L]`.
            cyclize: Optional cyclic-residue mask forwarded to the positional encoding.

        Returns:
            A tuple ``(msa, pair, state)`` of the embedded MSA `[B, N, L, msa_size]`, pair
            `[B, L, L, pair_size]`, and node-state `[B, L, state_size]` representations.
        """

        msa_depth = msa.shape[1]
        msa = self.emb(msa)
        query = _one_hot_lookup(seq, self.emb_q).unsqueeze(1)
        msa = self.drop(msa + query.expand(-1, msa_depth, -1, -1))

        left = _one_hot_lookup(seq, self.emb_left).unsqueeze(1)
        right = _one_hot_lookup(seq, self.emb_right).unsqueeze(2)
        pair = self.pos(left + right, idx, cyclize)

        state = self.drop(_one_hot_lookup(seq, self.emb_state))
        return msa, pair, state


class ExtraMsaEmbedding(nn.Module):
    r"""
    Embed the extra (full) MSA stack from its raw features.

    Projects the wider, shallower extra-MSA input and adds the per-residue query embedding, mirroring
    [`MsaEmbedding`][ofoldx.models.rfdiffusion.model.MsaEmbedding] but without the pair/state outputs.

    Args:
        msa_size: Width of the extra-MSA representation.
        input_feature_size: Width of the raw per-position extra-MSA input features.
        dropout: Dropout probability applied to the embedded features.
    """

    def __init__(
        self,
        msa_size: int = 256,
        input_feature_size: int = 22 + 1 + 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb = nn.Linear(input_feature_size, msa_size)
        self.emb_q = nn.Embedding(22, msa_size)
        self.drop = nn.Dropout(dropout)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the projection to the released RoseTTAFold scheme (Lecun-normal, zero bias)."""

        self.emb = init_lecun_normal(self.emb)
        _zero_bias(self.emb)

    def forward(self, msa: torch.Tensor, seq: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""
        Embed the extra MSA.

        Args:
            msa: Raw extra-MSA features `[B, N_full, L, input_feature_size]`.
            seq: One-hot sequence features `[B, L, 22]` used for the query embedding.
            idx: Residue-index tensor; unused, accepted for signature symmetry.

        Returns:
            The embedded extra MSA `[B, N_full, L, msa_size]`.
        """

        del idx
        msa_depth = msa.shape[1]
        msa = self.emb(msa)
        query = _one_hot_lookup(seq, self.emb_q).unsqueeze(1)
        return self.drop(msa + query.expand(-1, msa_depth, -1, -1))


class TemplatePairStack(nn.Module):
    r"""
    Stack of pair-update blocks that refines the template pair representation.

    Runs ``num_blocks`` [`PairStr2Pair`][ofoldx.models.rfdiffusion.model.PairStr2Pair] blocks over the
    flattened template batch, then layer-normalizes the result.

    Args:
        num_blocks: Number of pair-update blocks.
        template_size: Width of the template pair representation.
        num_attention_heads: Number of attention heads per block.
        hidden_size: Per-head hidden width per block.
        dropout: Dropout probability applied inside each block.
    """

    def __init__(
        self,
        num_blocks: int = 2,
        template_size: int = 64,
        num_attention_heads: int = 4,
        hidden_size: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.block = nn.ModuleList(
            PairStr2Pair(
                pair_size=template_size,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )
        self.norm = nn.LayerNorm(template_size)

    def forward(self, templ: torch.Tensor, rbf_feat: torch.Tensor) -> torch.Tensor:
        r"""
        Refine the template pair representation.

        Args:
            templ: Template pair features `[B, T, L, L, template_size]`.
            rbf_feat: Radial-basis distance features `[B*T, L, L, 36]` used as the structure bias.

        Returns:
            The refined, layer-normalized template pair features `[B, T, L, L, template_size]`.
        """

        batch_size, template_count, length = templ.shape[:3]
        templ = templ.reshape(batch_size * template_count, length, length, -1)
        for block in self.block:
            templ = block(templ, rbf_feat)
        templ = self.norm(templ)
        return templ.reshape(batch_size, template_count, length, length, -1)


class TemplateEmbedding(nn.Module):
    r"""
    Fold template features into the pair and node-state representations.

    Embeds the 1D/2D template features, refines them through a
    [`TemplatePairStack`][ofoldx.models.rfdiffusion.model.TemplatePairStack], then attends over templates
    to update the pair representation and the per-residue node state.

    Args:
        template_1d_size: Width of the 1D template feature.
        template_2d_size: Width of the 2D template feature.
        torsion_size: Width of the template torsion feature appended to the 1D state input.
        pair_size: Width of the pair representation being updated.
        state_size: Width of the node-state representation being updated.
        num_blocks: Number of pair-update blocks in the template stack.
        template_size: Internal width of the template pair representation.
        num_attention_heads: Number of attention heads.
        hidden_size: Per-head hidden width.
        dropout: Dropout probability used inside the template stack.
    """

    def __init__(
        self,
        template_1d_size: int = 21 + 1 + 1,
        template_2d_size: int = 43 + 1,
        torsion_size: int = 30,
        pair_size: int = 128,
        state_size: int = 32,
        num_blocks: int = 2,
        template_size: int = 64,
        num_attention_heads: int = 4,
        hidden_size: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.emb = nn.Linear(template_1d_size * 2 + template_2d_size, template_size)
        self.templ_stack = TemplatePairStack(
            num_blocks=num_blocks,
            template_size=template_size,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.attn = Attention(pair_size, template_size, num_attention_heads, hidden_size, pair_size)
        self.emb_template_1d = nn.Linear(template_1d_size + torsion_size, template_size)
        self.proj_template_1d = nn.Linear(template_size, template_size)
        self.attn_tor = Attention(state_size, template_size, num_attention_heads, hidden_size, state_size)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the template projections to the released RoseTTAFold scheme."""

        self.emb = init_lecun_normal(self.emb)
        _zero_bias(self.emb)
        nn.init.kaiming_normal_(self.emb_template_1d.weight, nonlinearity="relu")
        _zero_bias(self.emb_template_1d)
        self.proj_template_1d = init_lecun_normal(self.proj_template_1d)
        _zero_bias(self.proj_template_1d)

    def forward(
        self,
        template_1d: torch.Tensor,
        template_2d: torch.Tensor,
        alpha_t: torch.Tensor,
        template_coords: torch.Tensor,
        pair: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Update the pair and node-state representations with template features.

        Args:
            template_1d: 1D template features `[B, T, L, template_1d_size]`.
            template_2d: 2D template features `[B, T, L, L, template_2d_size]`.
            alpha_t: Template torsion features `[B, T, L, torsion_size]`.
            template_coords: Template backbone coords `[B, T, L, A, 3]`.
            pair: Pair representation `[B, L, L, pair_size]` to update.
            state: Node-state representation `[B, L, state_size]` to update.

        Returns:
            The updated ``(pair, state)`` representations.
        """

        batch_size, template_count, length, _ = template_1d.shape

        left = template_1d.unsqueeze(3).expand(-1, -1, -1, length, -1)
        right = template_1d.unsqueeze(2).expand(-1, -1, length, -1, -1)
        templ = self.emb(torch.cat((template_2d, left, right), dim=-1))

        template_coords = template_coords.reshape(batch_size * template_count, length, -1, 3)
        rbf_feat = rbf(torch.cdist(template_coords[:, :, 1], template_coords[:, :, 1]))
        templ = self.templ_stack(templ, rbf_feat)

        templ_state = torch.cat((template_1d, alpha_t), dim=-1)
        templ_state = self.proj_template_1d(F.relu(self.emb_template_1d(templ_state)))

        state_query = state.reshape(batch_size * length, 1, -1)
        templ_state = templ_state.permute(0, 2, 1, 3).reshape(batch_size * length, template_count, -1)
        state_update = self.attn_tor(state_query, templ_state, templ_state)
        state = state + state_update.reshape(batch_size, length, -1)

        pair_query = pair.reshape(batch_size * length * length, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(batch_size * length * length, template_count, -1)
        pair_update = self.attn(pair_query, templ, templ)
        pair = pair + pair_update.reshape(batch_size, length, length, -1)

        return pair, state


class Recycling(nn.Module):
    r"""
    Fold the previous iteration's MSA, pair, state, and structure back into the trunk inputs.

    Normalizes the recycled MSA/pair/state and adds a pair contribution derived from the previous backbone's
    pseudo-C-beta distogram, providing the recycling signal RoseTTAFold conditions on.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        state_size: Width of the node-state representation.
    """

    def __init__(self, msa_size: int = 256, pair_size: int = 128, state_size: int = 32) -> None:
        super().__init__()
        self.proj_dist = nn.Linear(36 + state_size * 2, pair_size)
        self.norm_state = nn.LayerNorm(state_size)
        self.norm_pair = nn.LayerNorm(pair_size)
        self.norm_msa = nn.LayerNorm(msa_size)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the distance projection to the released RoseTTAFold scheme (Lecun-normal, zero bias)."""

        self.proj_dist = init_lecun_normal(self.proj_dist)
        _zero_bias(self.proj_dist)

    def forward(
        self,
        seq: torch.Tensor,
        msa: torch.Tensor,
        pair: torch.Tensor,
        coords: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Build the recycled MSA, pair, and state contributions.

        Args:
            seq: Sequence features; unused, accepted for signature symmetry.
            msa: Previous-iteration MSA row `[B, L, msa_size]`.
            pair: Previous-iteration pair representation `[B, L, L, pair_size]`.
            coords: Previous-iteration backbone coords `[B, L, A, 3]`.
            state: Previous-iteration node state `[B, L, state_size]`.

        Returns:
            A tuple ``(msa, pair, state)`` of the recycled contributions to add back into the trunk inputs.
        """

        del seq
        _, length = pair.shape[:2]
        state = self.norm_state(state)
        left_state = state.unsqueeze(2).expand(-1, -1, length, -1)
        right_state = state.unsqueeze(1).expand(-1, length, -1, -1)

        cbeta = _pseudo_cbeta(coords)
        dist = rbf(torch.cdist(cbeta, cbeta))
        dist = self.proj_dist(torch.cat((dist, left_state, right_state), dim=-1))

        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair) + dist
        return msa, pair, state


DEFAULT_SE3_PARAMS = {
    "l0_in_features": 32,
    "l0_out_features": 16,
    "num_edge_features": 32,
}


def _se3_params(params: Mapping[str, int] | None) -> dict[str, int]:
    r"""Return a copy of ``params`` (or the default SE(3) sizing) as a plain mutable dict."""

    return dict(DEFAULT_SE3_PARAMS if params is None else params)


def _zeros_motif_mask(length: int, device: torch.device) -> Tensor:
    r"""Return an all-false motif mask `[L]` on ``device`` (no residues frozen)."""

    return torch.zeros(length, dtype=torch.bool, device=device)


def _query_state_update(msa: Tensor, state: Tensor) -> Tensor:
    r"""Add the projected node ``state`` into the query (row 0) slot of ``msa``."""

    query_index = torch.zeros(1, dtype=torch.long, device=state.device)
    return msa.index_add(1, query_index, state)


def _backbone_ca_distogram(coords: Tensor) -> Tensor:
    r"""Return the radial-basis distogram of the backbone CA-CA distances from coords `[B, L, A, 3]`."""

    return rbf(torch.cdist(coords[:, :, 1], coords[:, :, 1]))


def _apply_frames(rotations: Tensor, translations: Tensor, local_coords: Tensor) -> Tensor:
    r"""Apply per-residue frames ``(rotations, translations)`` to ``local_coords`` `[B, L, A, 3]`."""

    return einsum("bnij,bnaj->bnai", rotations, local_coords) + translations.unsqueeze(-2)


def _initial_frames(coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""Build identity rotations, CA translations, and CA-centered local coords from backbone coords."""

    batch_size, length = coords.shape[:2]
    rotations = torch.eye(3, device=coords.device).reshape(1, 1, 3, 3).expand(batch_size, length, -1, -1)
    translations = coords[:, :, 1].clone()
    local_coords = coords - translations.unsqueeze(-2)
    return rotations, translations, local_coords


def _quaternion_update_matrix(rotation_vector: Tensor, *, device: torch.device) -> Tensor:
    r"""Convert a per-residue rotation vector `[B, L, 3]` into a rotation matrix `[B, L, 3, 3]` via a quaternion."""

    norm = torch.sqrt(1 + torch.sum(rotation_vector * rotation_vector, dim=-1))
    q_a = 1 / norm
    q_b = rotation_vector[:, :, 0] / norm
    q_c = rotation_vector[:, :, 1] / norm
    q_d = rotation_vector[:, :, 2] / norm

    row0 = torch.stack(
        (
            q_a * q_a + q_b * q_b - q_c * q_c - q_d * q_d,
            2 * q_b * q_c - 2 * q_a * q_d,
            2 * q_b * q_d + 2 * q_a * q_c,
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            2 * q_b * q_c + 2 * q_a * q_d,
            q_a * q_a - q_b * q_b + q_c * q_c - q_d * q_d,
            2 * q_c * q_d - 2 * q_a * q_b,
        ),
        dim=-1,
    )
    row2 = torch.stack(
        (
            2 * q_b * q_d - 2 * q_a * q_c,
            2 * q_c * q_d + 2 * q_a * q_b,
            q_a * q_a - q_b * q_b - q_c * q_c + q_d * q_d,
        ),
        dim=-1,
    )
    return torch.stack((row0, row1, row2), dim=-2).to(device=device)


class MSAPairStr2MSA(nn.Module):
    r"""
    Update the MSA representation from the pair, structure, and node-state signals.

    Injects the pair bias (plus a backbone distogram) and the node state into the MSA, then runs row attention
    with pair bias, column (or global column) attention, and a feed-forward block.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation providing the attention bias.
        num_attention_heads: Number of attention heads.
        state_size: Width of the node-state representation.
        hidden_size: Per-head hidden width.
        dropout: Dropout probability applied to the row-attention update.
        use_global_attn: Whether to use global column attention (for the extra-MSA stack).
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        state_size: int = 16,
        hidden_size: int = 32,
        dropout: float = 0.15,
        use_global_attn: bool = False,
    ) -> None:
        super().__init__()
        self.norm_pair = nn.LayerNorm(pair_size)
        self.proj_pair = nn.Linear(pair_size + 36, pair_size)
        self.norm_state = nn.LayerNorm(state_size)
        self.proj_state = nn.Linear(state_size, msa_size)
        self.drop_row = Dropout(broadcast_dim=1, dropout=dropout)
        self.row_attn = MSARowAttentionWithBias(
            msa_size=msa_size, pair_size=pair_size, num_attention_heads=num_attention_heads, head_dim=hidden_size
        )
        self.col_attn = (
            MSAColGlobalAttention(msa_size=msa_size, num_attention_heads=num_attention_heads, head_dim=hidden_size)
            if use_global_attn
            else MSAColAttention(msa_size=msa_size, num_attention_heads=num_attention_heads, head_dim=hidden_size)
        )
        self.feed_forward = FeedForwardLayer(msa_size, 4, dropout=dropout)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the pair and state projections to the released RoseTTAFold scheme."""

        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa: Tensor, pair: Tensor, rbf_feat: Tensor, state: Tensor) -> Tensor:
        r"""
        Update the MSA with pair-biased row, column, and feed-forward attention.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.
            pair: Pair representation `[B, L, L, pair_size]`.
            rbf_feat: Backbone CA distogram `[B, L, L, 36]` concatenated into the pair bias.
            state: Node-state representation `[B, L, state_size]`.

        Returns:
            The updated MSA representation `[B, N, L, msa_size]`.
        """

        batch_size, _, length = msa.shape[:3]
        pair_bias = torch.cat((self.norm_pair(pair), rbf_feat), dim=-1)
        pair_bias = self.proj_pair(pair_bias)

        query_update = self.proj_state(self.norm_state(state)).reshape(batch_size, 1, length, -1)
        msa = _query_state_update(msa, query_update)

        msa = msa + self.drop_row(self.row_attn(msa, pair_bias))
        msa = msa + self.col_attn(msa)
        return msa + self.feed_forward(msa)


class PairStr2Pair(nn.Module):
    r"""
    Update the pair representation with structure-biased axial attention.

    Adds a structure bias derived from the backbone distance features, then applies row and column
    [`BiasedAxialAttention`][ofoldx.models.rfdiffusion.model.BiasedAxialAttention] followed by a
    feed-forward block.

    Args:
        pair_size: Width of the pair representation.
        num_attention_heads: Number of axial-attention heads.
        hidden_size: Per-head hidden width.
        rbf_size: Width of the radial-basis distance features feeding the structure bias.
        dropout: Dropout probability applied to the axial-attention updates.
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_attention_heads: int = 4,
        hidden_size: int = 32,
        rbf_size: int = 36,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.emb_rbf = nn.Linear(rbf_size, hidden_size)
        self.proj_rbf = nn.Linear(hidden_size, pair_size)
        self.drop_row = Dropout(broadcast_dim=1, dropout=dropout)
        self.drop_col = Dropout(broadcast_dim=2, dropout=dropout)
        self.row_attn = BiasedAxialAttention(
            pair_size, pair_size, num_attention_heads, hidden_size, dropout=dropout, is_row=True
        )
        self.col_attn = BiasedAxialAttention(
            pair_size, pair_size, num_attention_heads, hidden_size, dropout=dropout, is_row=False
        )
        self.feed_forward = FeedForwardLayer(pair_size, 2)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the RBF embedding/projection to the released RoseTTAFold scheme."""

        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity="relu")
        nn.init.zeros_(self.emb_rbf.bias)
        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair: Tensor, rbf_feat: Tensor) -> Tensor:
        r"""
        Update the pair representation with structure-biased row and column axial attention.

        Args:
            pair: Pair representation `[B, L, L, pair_size]`.
            rbf_feat: Radial-basis backbone distance features `[B, L, L, rbf_size]`.

        Returns:
            The updated pair representation `[B, L, L, pair_size]`.
        """

        structure_bias = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))
        pair = pair + self.drop_row(self.row_attn(pair, structure_bias))
        pair = pair + self.drop_col(self.col_attn(pair, structure_bias))
        return pair + self.feed_forward(pair)


class MSA2Pair(nn.Module):
    r"""
    Update the pair representation from the MSA via an outer product.

    Projects the normalized MSA to left/right features, forms their outer product over the MSA axis, and adds
    the projected result into the pair representation (the RoseTTAFold outer-product-mean update).

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation being updated.
        hidden_size: Width of the left/right projections whose outer product feeds the pair update.
        dropout: Dropout probability (accepted for signature symmetry).
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        hidden_size: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(msa_size)
        self.proj_left = nn.Linear(msa_size, hidden_size)
        self.proj_right = nn.Linear(msa_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size * hidden_size, pair_size)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the projections (left/right Lecun-normal, output zeroed) to the RoseTTAFold scheme."""

        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa: Tensor, pair: Tensor) -> Tensor:
        r"""
        Add the MSA outer-product update into the pair representation.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.
            pair: Pair representation `[B, L, L, pair_size]` to update.

        Returns:
            The updated pair representation `[B, L, L, pair_size]`.
        """

        batch_size, num_sequences, length = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa) / float(num_sequences)
        outer = einsum("bsli,bsmj->blmij", left, right).reshape(batch_size, length, length, -1)
        return pair + self.proj_out(outer)


class SCPred(nn.Module):
    r"""
    Predict sidechain and backbone torsion angles from the MSA row and node state.

    Two residual MLP blocks over the summed MSA/state features, projecting to 10 torsion angles encoded as
    ``(cos, sin)`` pairs. Used by [`Str2Str`][ofoldx.models.rfdiffusion.model.Str2Str] each structure step.

    Args:
        msa_size: Width of the MSA row input.
        state_size: Width of the node-state input.
        hidden_size: Hidden width of the residual MLP blocks.
        dropout: Dropout probability (accepted for signature symmetry).
    """

    def __init__(
        self,
        msa_size: int = 256,
        state_size: int = 32,
        hidden_size: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.norm_s0 = nn.LayerNorm(msa_size)
        self.norm_si = nn.LayerNorm(state_size)
        self.linear_s0 = nn.Linear(msa_size, hidden_size)
        self.linear_si = nn.Linear(state_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 20)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the MLP layers to the released RoseTTAFold scheme (residual branches zero-initialized)."""

        self.linear_s0 = init_lecun_normal(self.linear_s0)
        self.linear_si = init_lecun_normal(self.linear_si)
        self.linear_out = init_lecun_normal(self.linear_out)
        nn.init.zeros_(self.linear_s0.bias)
        nn.init.zeros_(self.linear_si.bias)
        nn.init.zeros_(self.linear_out.bias)

        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear_3.bias)

        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)

    def forward(self, seq: Tensor, state: Tensor) -> Tensor:
        r"""
        Predict torsion angles from the MSA row and node state.

        Args:
            seq: MSA query row `[B, L, msa_size]`.
            state: Node-state representation `[B, L, state_size]`.

        Returns:
            Torsion angles `[B, L, 10, 2]` as ``(cos, sin)`` pairs.
        """

        batch_size, length = seq.shape[:2]
        hidden = self.linear_s0(self.norm_s0(seq)) + self.linear_si(self.norm_si(state))
        hidden = hidden + self.linear_2(F.relu_(self.linear_1(F.relu_(hidden))))
        hidden = hidden + self.linear_4(F.relu_(self.linear_3(F.relu_(hidden))))
        return self.linear_out(F.relu_(hidden)).view(batch_size, length, 10, 2)


class Str2Str(nn.Module):
    r"""
    Structure-update module: refine backbone frames with an SE(3)-equivariant transformer.

    Embeds the MSA/pair/state features into node and edge features, builds a residue graph (full or top-k),
    runs the [`SE3TransformerWrapper`][ofoldx.models.rfdiffusion.model.SE3TransformerWrapper], and applies
    the predicted translation and rotation updates to the input frames. Runs with autocast disabled so the
    equivariant math stays in fp32.

    Args:
        msa_size: Width of the MSA row input.
        pair_size: Width of the pair representation.
        state_size: Width of the node-state representation.
        se3_param: SE(3) sizing dict (defaults applied when `None`).
        dropout: Dropout probability forwarded to the sidechain predictor.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        state_size: int = 16,
        se3_param: Mapping[str, int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        se3_param = _se3_params(se3_param)
        l0_in_features = se3_param["l0_in_features"]
        l0_out_features = se3_param["l0_out_features"]
        num_edge_features = se3_param["num_edge_features"]

        self.norm_msa = nn.LayerNorm(msa_size)
        self.norm_pair = nn.LayerNorm(pair_size)
        self.norm_state = nn.LayerNorm(state_size)
        self.embed_x = nn.Linear(msa_size + state_size, l0_in_features)
        self.embed_e1 = nn.Linear(pair_size, num_edge_features)
        self.embed_e2 = nn.Linear(num_edge_features + 36 + 1, num_edge_features)
        self.norm_node = nn.LayerNorm(l0_in_features)
        self.norm_edge1 = nn.LayerNorm(num_edge_features)
        self.norm_edge2 = nn.LayerNorm(num_edge_features)
        self.se3 = SE3TransformerWrapper(**se3_param)
        self.sc_predictor = SCPred(msa_size=msa_size, state_size=l0_out_features, dropout=dropout)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the node/edge embeddings to the released RoseTTAFold scheme (Lecun-normal, zero bias)."""

        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        input_rotations: Tensor,
        input_translations: Tensor,
        coords: Tensor,
        state: Tensor,
        idx: Tensor,
        motif_mask: Tensor | None,
        cyclic_reses: Tensor | None = None,
        top_k: int = 64,
        eps: float = 1e-5,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Refine backbone frames and predict torsions with the SE(3) transformer.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.
            pair: Pair representation `[B, L, L, pair_size]`.
            input_rotations: Current per-residue rotations `[B, L, 3, 3]`.
            input_translations: Current per-residue translations (CA) `[B, L, 3]`.
            coords: Current backbone coords `[B, L, A, 3]`.
            state: Node-state representation `[B, L, state_size]`.
            idx: Residue-index tensor `[B, L]`.
            motif_mask: Per-residue mask `[L]`; frame updates are zeroed on masked (motif) residues.
            cyclic_reses: Optional cyclic-residue mask for neighbor features.
            top_k: Number of neighbors for the residue graph; `0` builds the full graph.
            eps: Numerical floor (accepted for signature symmetry).

        Returns:
            A tuple ``(rotations, translations, state, alpha)`` of the updated frames, node state, and
            predicted torsion angles.
        """

        batch_size, _, length = msa.shape[:3]
        motif_mask = _zeros_motif_mask(length, coords.device) if motif_mask is None else motif_mask

        node = torch.cat((self.norm_msa(msa[:, 0]), self.norm_state(state)), dim=-1)
        node = self.norm_node(self.embed_x(node))
        pair = self.norm_edge1(self.embed_e1(self.norm_pair(pair)))

        neighbor = get_seqsep(idx, cyclic_reses)
        pair = torch.cat((pair, _backbone_ca_distogram(coords), neighbor), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))

        ca_coords = coords[:, :, 1, :]
        graph, edge_feats = (
            make_topk_graph(ca_coords, pair, idx, top_k=top_k) if top_k != 0 else make_full_graph(ca_coords, pair, idx)
        )
        vector_features = (coords - ca_coords.unsqueeze(2)).reshape(batch_size * length, -1, 3)
        shift = self.se3(graph, node.reshape(batch_size * length, -1, 1), vector_features, edge_feats)

        state = shift["0"].reshape(batch_size, length, -1)
        offset = shift["1"].reshape(batch_size, length, 2, 3)
        offset[:, motif_mask, ...] = 0

        translations = offset[:, :, 0, :] / 10.0
        rotation_vector = offset[:, :, 1, :] / 100.0
        delta_rotations = _quaternion_update_matrix(rotation_vector, device=coords.device)
        rotations = einsum("bnij,bnjk->bnik", delta_rotations, input_rotations)
        translations = translations + input_translations
        alpha = self.sc_predictor(msa[:, 0], state)
        return rotations, translations, state, alpha


class IterBlock(nn.Module):
    r"""
    One RoseTTAFold iteration block over the MSA, pair, and structure representations.

    Chains the MSA update, MSA-to-pair outer product, pair update, and one structure step, returning the
    refreshed representations and frames. The trunk stacks many of these.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        num_msa_attention_heads: Number of MSA attention heads.
        num_pair_attention_heads: Number of pair attention heads.
        use_global_attn: Whether the MSA update uses global column attention.
        hidden_size: Per-head hidden width for the pair/structure updates.
        msa_hidden_size: Per-head hidden width for the MSA update; defaults to ``hidden_size``.
        dropout: Dropout probability applied throughout the block.
        se3_param: SE(3) sizing dict for the structure step.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        num_msa_attention_heads: int = 8,
        num_pair_attention_heads: int = 4,
        use_global_attn: bool = False,
        hidden_size: int = 32,
        msa_hidden_size: int | None = None,
        dropout: float = 0.15,
        se3_param: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        se3_param = _se3_params(se3_param)
        msa_hidden_size = hidden_size if msa_hidden_size is None else msa_hidden_size
        state_size = se3_param["l0_out_features"]

        self.msa2msa = MSAPairStr2MSA(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            state_size=state_size,
            use_global_attn=use_global_attn,
            hidden_size=msa_hidden_size,
            dropout=dropout,
        )
        self.msa2pair = MSA2Pair(msa_size=msa_size, pair_size=pair_size, hidden_size=hidden_size // 2, dropout=dropout)
        self.pair2pair = PairStr2Pair(
            pair_size=pair_size, num_attention_heads=num_pair_attention_heads, hidden_size=hidden_size, dropout=dropout
        )
        self.str2str = Str2Str(
            msa_size=msa_size, pair_size=pair_size, state_size=state_size, se3_param=se3_param, dropout=dropout
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        input_rotations: Tensor,
        input_translations: Tensor,
        coords: Tensor,
        state: Tensor,
        idx: Tensor,
        motif_mask: Tensor,
        cyclic_reses: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Run one iteration block over MSA, pair, and structure.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.
            pair: Pair representation `[B, L, L, pair_size]`.
            input_rotations: Current per-residue rotations `[B, L, 3, 3]`.
            input_translations: Current per-residue translations `[B, L, 3]`.
            coords: Current backbone coords `[B, L, A, 3]`.
            state: Node-state representation `[B, L, state_size]`.
            idx: Residue-index tensor `[B, L]`.
            motif_mask: Per-residue mask `[L]` freezing motif frame updates.
            cyclic_reses: Optional cyclic-residue mask.

        Returns:
            A tuple ``(msa, pair, rotations, translations, state, alpha)`` with the updated representations,
            frames, node state, and predicted torsions.
        """

        rbf_feat = _backbone_ca_distogram(coords)
        msa = self.msa2msa(msa, pair, rbf_feat, state)
        pair = self.msa2pair(msa, pair)
        pair = self.pair2pair(pair, rbf_feat)
        rotations, translations, state, alpha = self.str2str(
            msa,
            pair,
            input_rotations,
            input_translations,
            coords,
            state,
            idx,
            motif_mask=motif_mask,
            cyclic_reses=cyclic_reses,
            top_k=0,
        )
        return msa, pair, rotations, translations, state, alpha


class IterativeSimulator(nn.Module):
    r"""
    The RoseTTAFold track: stacked iteration blocks plus a top-k structure refiner.

    Runs the extra-MSA blocks, then the main blocks, then several top-k structure-refinement steps, detaching
    frames between stages. Returns the final representations along with the per-step frame and torsion
    trajectories used to build the denoised structure.

    Args:
        num_extra_blocks: Number of extra-MSA iteration blocks.
        num_main_blocks: Number of main iteration blocks.
        num_refinement_blocks: Number of top-k structure-refinement steps.
        msa_size: Width of the latent MSA representation.
        full_msa_size: Width of the extra (full) MSA representation.
        pair_size: Width of the pair representation.
        hidden_size: Per-head hidden width shared across the blocks.
        num_msa_attention_heads: Number of MSA attention heads.
        num_pair_attention_heads: Number of pair attention heads.
        se3_param_full: SE(3) sizing dict for the main/extra structure steps.
        se3_param_topk: SE(3) sizing dict for the top-k refinement steps.
        dropout: Dropout probability applied throughout the track.
    """

    def __init__(
        self,
        num_extra_blocks: int = 4,
        num_main_blocks: int = 12,
        num_refinement_blocks: int = 4,
        msa_size: int = 256,
        full_msa_size: int = 64,
        pair_size: int = 128,
        hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        num_pair_attention_heads: int = 4,
        se3_param_full: Mapping[str, int] | None = None,
        se3_param_topk: Mapping[str, int] | None = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        se3_param_full = _se3_params(se3_param_full)
        se3_param_topk = _se3_params(se3_param_topk)

        self.num_extra_blocks = num_extra_blocks
        self.num_main_blocks = num_main_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.proj_state = nn.Linear(se3_param_topk["l0_out_features"], se3_param_full["l0_out_features"])

        if num_extra_blocks > 0:
            self.extra_block = nn.ModuleList(
                [
                    IterBlock(
                        msa_size=full_msa_size,
                        pair_size=pair_size,
                        num_msa_attention_heads=num_msa_attention_heads,
                        num_pair_attention_heads=num_pair_attention_heads,
                        msa_hidden_size=8,
                        hidden_size=hidden_size,
                        dropout=dropout,
                        use_global_attn=True,
                        se3_param=se3_param_full,
                    )
                    for _ in range(num_extra_blocks)
                ]
            )

        if num_main_blocks > 0:
            self.main_block = nn.ModuleList(
                [
                    IterBlock(
                        msa_size=msa_size,
                        pair_size=pair_size,
                        num_msa_attention_heads=num_msa_attention_heads,
                        num_pair_attention_heads=num_pair_attention_heads,
                        hidden_size=hidden_size,
                        dropout=dropout,
                        use_global_attn=False,
                        se3_param=se3_param_full,
                    )
                    for _ in range(num_main_blocks)
                ]
            )

        self.proj_state2 = nn.Linear(se3_param_full["l0_out_features"], se3_param_topk["l0_out_features"])
        if num_refinement_blocks > 0:
            self.str_refiner = Str2Str(
                msa_size=msa_size,
                pair_size=pair_size,
                state_size=se3_param_topk["l0_out_features"],
                se3_param=se3_param_topk,
                dropout=dropout,
            )

        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the state projections between stages to the released RoseTTAFold scheme."""

        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_state.bias)
        self.proj_state2 = init_lecun_normal(self.proj_state2)
        nn.init.zeros_(self.proj_state2.bias)

    def forward(
        self,
        seq: Tensor,
        msa: Tensor,
        msa_full: Tensor,
        pair: Tensor,
        coords_in: Tensor,
        state: Tensor,
        idx: Tensor,
        cyclic_reses: Tensor | None = None,
        motif_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Run the full track and return the final representations and frame trajectory.

        Args:
            seq: Sequence features (passed through to blocks).
            msa: Latent MSA representation `[B, N, L, msa_size]`.
            msa_full: Extra (full) MSA representation.
            pair: Pair representation `[B, L, L, pair_size]`.
            coords_in: Initial backbone coords `[B, L, 3, 3]`.
            state: Initial node-state representation `[B, L, ...]`.
            idx: Residue-index tensor `[B, L]`.
            cyclic_reses: Optional cyclic-residue mask.
            motif_mask: Per-residue mask `[L]` freezing motif frame updates.

        Returns:
            A tuple ``(msa, pair, rotations, translations, alpha, state)`` where ``rotations``, ``translations``
            and ``alpha`` are stacked over all track steps `[steps, ...]`.
        """

        _, length = pair.shape[:2]
        motif_mask = _zeros_motif_mask(length, coords_in.device) if motif_mask is None else motif_mask
        rotations, translations, local_coords = _initial_frames(coords_in)
        state = self.proj_state(state)

        rotation_steps: list[Tensor] = []
        translation_steps: list[Tensor] = []
        alpha_steps: list[Tensor] = []

        for block in getattr(self, "extra_block", ()):
            rotations = rotations.detach()
            translations = translations.detach()
            coords = _apply_frames(rotations, translations, local_coords)
            msa_full, pair, rotations, translations, state, alpha = block(
                msa_full,
                pair,
                rotations,
                translations,
                coords,
                state,
                idx,
                motif_mask=motif_mask,
                cyclic_reses=cyclic_reses,
            )
            rotation_steps.append(rotations)
            translation_steps.append(translations)
            alpha_steps.append(alpha)

        for block in getattr(self, "main_block", ()):
            rotations = rotations.detach()
            translations = translations.detach()
            coords = _apply_frames(rotations, translations, local_coords)
            msa, pair, rotations, translations, state, alpha = block(
                msa,
                pair,
                rotations,
                translations,
                coords,
                state,
                idx,
                motif_mask=motif_mask,
                cyclic_reses=cyclic_reses,
            )
            rotation_steps.append(rotations)
            translation_steps.append(translations)
            alpha_steps.append(alpha)

        state = self.proj_state2(state)
        for _ in range(self.num_refinement_blocks):
            rotations = rotations.detach()
            translations = translations.detach()
            coords = _apply_frames(rotations, translations, local_coords)
            rotations, translations, state, alpha = self.str_refiner(
                msa,
                pair,
                rotations,
                translations,
                coords,
                state,
                idx,
                top_k=64,
                motif_mask=motif_mask,
                cyclic_reses=cyclic_reses,
            )
            rotation_steps.append(rotations)
            translation_steps.append(translations)
            alpha_steps.append(alpha)

        return (
            msa,
            pair,
            torch.stack(rotation_steps, dim=0),
            torch.stack(translation_steps, dim=0),
            torch.stack(alpha_steps, dim=0),
            state,
        )


class FeedForwardLayer(nn.Module):
    r"""
    Pre-norm feed-forward block with a ReLU-activated expansion.

    Args:
        hidden_size: Input and output width.
        expansion: Inner-width expansion factor.
        dropout: Dropout probability applied after the activation.
    """

    def __init__(self, hidden_size: int, expansion: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * expansion)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size * expansion, hidden_size)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the block (input Kaiming, output zeroed) to the released RoseTTAFold scheme."""

        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the pre-norm feed-forward transform.

        Args:
            src: Input tensor `[..., hidden_size]`.

        Returns:
            The transformed tensor `[..., hidden_size]`.
        """

        src = self.norm(src)
        return self.linear2(self.dropout(F.relu_(self.linear1(src))))


class Attention(nn.Module):
    r"""
    Multi-head scaled dot-product attention with separate query/key sizes.

    Generic (non-axial) attention used by the template embedding to attend over templates.

    Args:
        query_size: Width of the query input.
        key_size: Width of the key/value input.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head dimension.
        output_size: Width of the projected output.
    """

    def __init__(
        self, query_size: int, key_size: int, num_attention_heads: int, head_dim: int, output_size: int
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(query_size, num_attention_heads * head_dim, bias=False)
        self.to_k = nn.Linear(key_size, num_attention_heads * head_dim, bias=False)
        self.to_v = nn.Linear(key_size, num_attention_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_attention_heads * head_dim, output_size)
        self.scaling = 1 / math.sqrt(head_dim)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the projections (Xavier for q/k/v, zeroed output) to the RoseTTAFold scheme."""

        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        r"""
        Compute multi-head attention of ``query`` over ``key``/``value``.

        Args:
            query: Query tensor `[B, Q, query_size]`.
            key: Key tensor `[B, K, key_size]`.
            value: Value tensor `[B, K, key_size]`.

        Returns:
            The attended output `[B, Q, output_size]`.
        """

        batch_size, query_len = query.shape[:2]
        key_len = key.shape[1]
        query = self.to_q(query).reshape(batch_size, query_len, self.num_attention_heads, self.head_dim)
        key = self.to_k(key).reshape(batch_size, key_len, self.num_attention_heads, self.head_dim)
        value = self.to_v(value).reshape(batch_size, key_len, self.num_attention_heads, self.head_dim)
        attn = einsum("bqhd,bkhd->bhqk", query * self.scaling, key)
        attn = F.softmax(attn, dim=-1)
        out = einsum("bhqk,bkhd->bqhd", attn, value)
        return self.to_out(out.reshape(batch_size, query_len, self.num_attention_heads * self.head_dim))


class SequenceWeight(nn.Module):
    r"""
    Per-head softmax weights over MSA rows relative to the query row.

    Produces the sequence-attention weights that reweight MSA rows inside
    [`MSARowAttentionWithBias`][ofoldx.models.rfdiffusion.model.MSARowAttentionWithBias].

    Args:
        msa_size: Width of the MSA representation.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head dimension.
        dropout: Dropout probability applied to the weights.
    """

    def __init__(self, msa_size: int, num_attention_heads: int, head_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.to_query = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.to_key = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize the query/key projections with Xavier uniform."""

        nn.init.xavier_uniform_(self.to_query.weight)
        nn.init.xavier_uniform_(self.to_key.weight)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        r"""
        Compute per-head sequence-attention weights over MSA rows.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.

        Returns:
            Softmax weights `[B, N, L, num_attention_heads, 1]` over the MSA-row axis.
        """

        batch_size, msa_depth, length = msa.shape[:3]
        query = self.to_query(msa[:, 0]).view(batch_size, 1, length, self.num_attention_heads, self.head_dim)
        key = self.to_key(msa).view(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        attn = einsum("bqihd,bkihd->bkihq", query * self.scale, key)
        return self.dropout(F.softmax(attn, dim=1))


class MSARowAttentionWithBias(nn.Module):
    r"""
    Row (per-column) MSA attention biased by the pair representation and gated.

    Attends across residues within each MSA row, adds a per-head bias projected from the pair representation,
    reweights rows by the [`SequenceWeight`][ofoldx.models.rfdiffusion.model.SequenceWeight], and gates the
    output.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation providing the attention bias.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head dimension.
    """

    def __init__(
        self, msa_size: int = 256, pair_size: int = 128, num_attention_heads: int = 8, head_dim: int = 32
    ) -> None:
        super().__init__()
        self.norm_msa = nn.LayerNorm(msa_size)
        self.norm_pair = nn.LayerNorm(pair_size)
        self.seq_weight = SequenceWeight(msa_size, num_attention_heads, head_dim, dropout=0.1)
        self.to_q = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.to_k = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.to_v = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.to_b = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.to_g = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.to_out = nn.Linear(num_attention_heads * head_dim, msa_size)
        self.scaling = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize projections to the RoseTTAFold scheme (Xavier q/k/v, gate open, zeroed output)."""

        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        self.to_b = init_lecun_normal(self.to_b)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        r"""
        Apply pair-biased, sequence-weighted, gated row attention.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.
            pair: Pair representation `[B, L, L, pair_size]` providing the per-head bias.

        Returns:
            The updated MSA representation `[B, N, L, msa_size]`.
        """

        batch_size, msa_depth, length = msa.shape[:3]
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        seq_weight = self.seq_weight(msa)
        query = self.to_q(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        key = self.to_k(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        value = self.to_v(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        bias = self.to_b(pair)
        gate = torch.sigmoid(self.to_g(msa))
        query = query * seq_weight.expand(-1, -1, -1, -1, self.head_dim)
        attn = einsum("bsqhd,bskhd->bqkh", query, key * self.scaling)
        attn = F.softmax(attn + bias, dim=-2)
        out = einsum("bqkh,bskhd->bsqhd", attn, value).reshape(batch_size, msa_depth, length, -1)
        return self.to_out(gate * out)


class MSAColAttention(nn.Module):
    r"""
    Column (per-row) MSA attention across sequences, gated.

    Attends across MSA rows at each residue position, then gates the output. Used for the latent MSA stack.

    Args:
        msa_size: Width of the MSA representation.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head dimension.
    """

    def __init__(self, msa_size: int = 256, num_attention_heads: int = 8, head_dim: int = 32) -> None:
        super().__init__()
        self.norm_msa = nn.LayerNorm(msa_size)
        self.to_q = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.to_k = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.to_v = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.to_g = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.to_out = nn.Linear(num_attention_heads * head_dim, msa_size)
        self.scaling = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize projections to the RoseTTAFold scheme (Xavier q/k/v, gate open, zeroed output)."""

        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        r"""
        Apply gated column attention across MSA rows.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.

        Returns:
            The updated MSA representation `[B, N, L, msa_size]`.
        """

        batch_size, msa_depth, length = msa.shape[:3]
        msa = self.norm_msa(msa)
        query = self.to_q(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        key = self.to_k(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        value = self.to_v(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        gate = torch.sigmoid(self.to_g(msa))
        attn = einsum("bqihd,bkihd->bihqk", query * self.scaling, key)
        attn = F.softmax(attn, dim=-1)
        out = einsum("bihqk,bkihd->bqihd", attn, value).reshape(batch_size, msa_depth, length, -1)
        return self.to_out(gate * out)


class MSAColGlobalAttention(nn.Module):
    r"""
    Global column MSA attention with a mean-pooled query, gated.

    Cheaper column attention for the deep extra-MSA stack: the query is mean-pooled over rows and the key/value
    share a single head, keeping the cost linear in MSA depth.

    Args:
        msa_size: Width of the MSA representation.
        num_attention_heads: Number of attention heads for the query/gate.
        head_dim: Per-head dimension.
    """

    def __init__(self, msa_size: int = 64, num_attention_heads: int = 8, head_dim: int = 8) -> None:
        super().__init__()
        self.norm_msa = nn.LayerNorm(msa_size)
        self.to_q = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.to_k = nn.Linear(msa_size, head_dim, bias=False)
        self.to_v = nn.Linear(msa_size, head_dim, bias=False)
        self.to_g = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.to_out = nn.Linear(num_attention_heads * head_dim, msa_size)
        self.scaling = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize projections to the RoseTTAFold scheme (Xavier q/k/v, gate open, zeroed output)."""

        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        r"""
        Apply gated global column attention with a mean-pooled query.

        Args:
            msa: MSA representation `[B, N, L, msa_size]`.

        Returns:
            The updated MSA representation broadcast to `[B, 1, L, msa_size]`.
        """

        batch_size, msa_depth, length = msa.shape[:3]
        msa = self.norm_msa(msa)
        query = (
            self.to_q(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim).mean(dim=1)
        )
        key = self.to_k(msa)
        value = self.to_v(msa)
        gate = torch.sigmoid(self.to_g(msa))
        attn = einsum("bihd,bkid->bihk", query * self.scaling, key)
        attn = F.softmax(attn, dim=-1)
        out = einsum("bihk,bkid->bihd", attn, value).reshape(batch_size, 1, length, -1)
        return self.to_out(gate * out)


class BiasedAxialAttention(nn.Module):
    r"""
    Axial (row or column) attention over the pair representation with an additive bias, gated.

    Attends along one axis of the pair map at a time (transposing for the row case), adds a per-head bias from
    a structure feature, and gates the output. The workhorse of
    [`PairStr2Pair`][ofoldx.models.rfdiffusion.model.PairStr2Pair].

    Args:
        pair_size: Width of the pair representation.
        bias_size: Width of the bias feature the per-head bias is projected from.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head dimension.
        dropout: Dropout probability (accepted for signature symmetry).
        is_row: Whether to attend along rows (`True`) or columns (`False`).
    """

    def __init__(
        self,
        pair_size: int,
        bias_size: int,
        num_attention_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        is_row: bool = True,
    ) -> None:
        super().__init__()
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(pair_size)
        self.norm_bias = nn.LayerNorm(bias_size)
        self.to_q = nn.Linear(pair_size, num_attention_heads * head_dim, bias=False)
        self.to_k = nn.Linear(pair_size, num_attention_heads * head_dim, bias=False)
        self.to_v = nn.Linear(pair_size, num_attention_heads * head_dim, bias=False)
        self.to_b = nn.Linear(bias_size, num_attention_heads, bias=False)
        self.to_g = nn.Linear(pair_size, num_attention_heads * head_dim)
        self.to_out = nn.Linear(num_attention_heads * head_dim, pair_size)
        self.scaling = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Re-initialize projections to the RoseTTAFold scheme (Xavier q/k/v, gate open, zeroed output)."""

        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        self.to_b = init_lecun_normal(self.to_b)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        r"""
        Apply biased, gated axial attention to the pair representation.

        Args:
            pair: Pair representation `[B, L, L, pair_size]`.
            bias: Bias feature `[B, L, L, bias_size]` projected into the per-head bias.

        Returns:
            The updated pair representation `[B, L, L, pair_size]`.
        """

        batch_size, length = pair.shape[:2]
        if self.is_row:
            pair = pair.permute(0, 2, 1, 3)
            bias = bias.permute(0, 2, 1, 3)
        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)
        query = self.to_q(pair).reshape(batch_size, length, length, self.num_attention_heads, self.head_dim)
        key = self.to_k(pair).reshape(batch_size, length, length, self.num_attention_heads, self.head_dim)
        value = self.to_v(pair).reshape(batch_size, length, length, self.num_attention_heads, self.head_dim)
        bias = self.to_b(bias)
        gate = torch.sigmoid(self.to_g(pair))
        attn = einsum("bnihk,bnjhk->bijh", query * self.scaling, key / math.sqrt(length))
        attn = F.softmax(attn + bias, dim=-2)
        out = einsum("bijh,bkjhd->bikhd", attn, value).reshape(batch_size, length, length, -1)
        out = self.to_out(gate * out)
        if self.is_row:
            out = out.permute(0, 2, 1, 3)
        return out


FiberEl = namedtuple("FiberEl", ["degree", "channels"])


def degree_to_dim(degree: int) -> int:
    r"""
    Return the dimension ``2*degree + 1`` of a spherical-harmonic feature of the given degree.

    Args:
        degree: SE(3) representation degree (0 for scalars, 1 for vectors, ...).

    Returns:
        The number of components a single feature of that degree occupies.

    Examples:
        >>> from ofoldx.models.rfdiffusion.model import degree_to_dim
        >>> [degree_to_dim(d) for d in (0, 1, 2)]
        [1, 3, 5]
    """

    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: Iterable[int]) -> dict[str, Tensor]:
    r"""
    Split a fused feature tensor into a per-degree dict.

    Args:
        features: Fused features `[..., sum(degree_to_dim(d) for d in degrees)]`.
        degrees: Degrees present, in the order they are packed along the last axis.

    Returns:
        A dict keyed by string degree, each holding the slice for that degree.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import unfuse_features
        >>> out = unfuse_features(torch.arange(4.0).reshape(1, 4), [0, 1])
        >>> {k: v.tolist() for k, v in out.items()}
        {'0': [[0.0]], '1': [[1.0, 2.0, 3.0]]}
    """

    degree_list = list(degrees)
    return dict(
        zip(
            [str(degree) for degree in degree_list],
            features.split([degree_to_dim(degree) for degree in degree_list], dim=-1),
        )
    )


class Fiber(dict):
    r"""
    Degree-to-channel structure of an SE(3)-equivariant feature space.

    A dict mapping each representation degree to its channel count, with helpers to enumerate degrees,
    channels, and the total flattened feature width. Ported from the NVIDIA SE(3)-Transformer.

    Args:
        structure: A ``{degree: channels}`` dict, a list of ``(degree, channels)`` pairs, or a list of
            ``FiberEl`` tuples describing the fiber.
    """

    def __init__(self, structure) -> None:
        if isinstance(structure, dict):
            structure = [
                FiberEl(int(degree), int(channels))
                for degree, channels in sorted(structure.items(), key=lambda item: item[0])
            ]
        elif not isinstance(structure[0], FiberEl):
            structure = [FiberEl(*item) for item in sorted(structure, key=lambda item: item[0])]
        self.structure = structure
        super().__init__(dict(self.structure))

    @property
    def degrees(self) -> list[int]:
        r"""Sorted list of representation degrees present in the fiber."""

        return sorted([item.degree for item in self.structure])

    @property
    def channels(self) -> list[int]:
        r"""Channel counts, one per degree in `degrees` order."""

        return [self[degree] for degree in self.degrees]

    @property
    def num_features(self) -> int:
        r"""Total flattened feature width, summing ``channels * degree_to_dim(degree)`` over all degrees."""

        return sum(item.channels * degree_to_dim(item.degree) for item in self.structure)

    @staticmethod
    def create(num_degrees: int, num_channels: int) -> Fiber:
        r"""
        Build a uniform fiber with ``num_channels`` channels for each degree ``0..num_degrees-1``.

        Args:
            num_degrees: Number of consecutive degrees (starting at 0) to include.
            num_channels: Channel count assigned to every degree.

        Returns:
            The constructed [`Fiber`][ofoldx.models.rfdiffusion.model.Fiber].

        Examples:
            >>> from ofoldx.models.rfdiffusion.model import Fiber
            >>> fiber = Fiber.create(3, 16)
            >>> fiber.degrees
            [0, 1, 2]
            >>> fiber.num_features
            144
        """

        return Fiber([(degree, num_channels) for degree in range(num_degrees)])

    def __getitem__(self, degree: int) -> int:
        r"""Return the channel count for ``degree``, or 0 if the degree is absent from the fiber."""

        return dict(self.structure).get(degree, 0)

    def __iter__(self):
        r"""Iterate over the ``FiberEl`` ``(degree, channels)`` entries."""

        return iter(self.structure)

    def __add__(self, other):
        r"""Add two fibers channel-wise per degree, or add an int to every degree's channel count."""

        if isinstance(other, Fiber):
            return Fiber({item.degree: item.channels + other[item.degree] for item in self.structure})
        if isinstance(other, int):
            return Fiber({item.degree: item.channels + other for item in self.structure})
        return NotImplemented


class Sequential(nn.Sequential):
    r"""
    An [torch.nn.Sequential][] that threads extra positional and keyword arguments to every child module.

    Needed because the SE(3) graph modules take the graph and basis alongside the running features.
    """

    def forward(self, input, *args, **kwargs):
        r"""
        Apply each child module in turn, forwarding ``*args``/``**kwargs`` to each.

        Args:
            input: The running feature passed through the stack.
            *args: Extra positional arguments forwarded to every child.
            **kwargs: Extra keyword arguments forwarded to every child.

        Returns:
            The output of the final child module.
        """

        for module in self:
            input = module(input, *args, **kwargs)
        return input


class RadialProfile(nn.Module):
    r"""
    MLP mapping invariant edge features to the radial weights of an SE(3) convolution.

    Args:
        num_freq: Number of radial frequencies produced per channel pair.
        channels_in: Number of input channels.
        channels_out: Number of output channels.
        edge_dim: Width of the invariant edge features.
        mid_dim: Hidden width of the MLP.
        use_layer_norm: Whether to layer-norm the hidden activations.
    """

    def __init__(
        self,
        num_freq: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int = 1,
        mid_dim: int = 32,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        modules = [
            nn.Linear(edge_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, num_freq * channels_in * channels_out, bias=False),
        ]
        self.net = nn.Sequential(*[module for module in modules if module is not None])

    def forward(self, features: Tensor) -> Tensor:
        r"""
        Map invariant edge ``features`` to flattened radial weights.

        Args:
            features: Invariant edge features `[E, edge_dim]`.

        Returns:
            Radial weights `[E, num_freq * channels_in * channels_out]`.
        """

        return self.net(features)


class VersatileConvSE3(nn.Module):
    r"""
    A single degree-in to degree-out SE(3) convolution parameterized by a radial profile.

    Applies the radial MLP to the edge features, then contracts the resulting weights against the
    equivariant basis (or a plain matmul when no basis is given). Ported from the NVIDIA SE(3)-Transformer.

    Args:
        freq_sum: Number of radial frequencies for this degree pair.
        channels_in: Number of input channels.
        channels_out: Number of output channels.
        edge_dim: Width of the invariant edge features.
        use_layer_norm: Whether the radial MLP layer-norms its hidden activations.
    """

    def __init__(
        self,
        freq_sum: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.radial_func = RadialProfile(
            num_freq=freq_sum,
            channels_in=channels_in,
            channels_out=channels_out,
            edge_dim=edge_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, features: Tensor, invariant_edge_feats: Tensor, basis: Tensor | None) -> Tensor:
        r"""
        Convolve ``features`` using radial weights derived from the edge features.

        Args:
            features: Per-edge input features.
            invariant_edge_feats: Invariant edge features `[E, edge_dim]` feeding the radial MLP.
            basis: Equivariant basis for the degree pair, or `None` for a plain matmul.

        Returns:
            The convolved output features.
        """

        radial_weights = self.radial_func(invariant_edge_feats).view(
            -1, self.channels_out, self.channels_in * self.freq_sum
        )
        if basis is None:
            return radial_weights @ features
        return weighted_tensor_product(features, basis, radial_weights)


class LinearSE3(nn.Module):
    r"""
    Degree-wise equivariant linear layer: an independent channel mix per representation degree.

    Ported from the NVIDIA SE(3)-Transformer.

    Args:
        fiber_in: Input fiber structure.
        fiber_out: Output fiber structure.
    """

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber) -> None:
        super().__init__()
        self.weights = nn.ParameterDict(
            {
                str(degree_out): nn.Parameter(
                    torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out])
                )
                for degree_out, channels_out in fiber_out
            }
        )

    def forward(self, features: dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
        r"""
        Apply the per-degree channel mix.

        Args:
            features: Per-degree feature dict keyed by string degree.
            *args: Ignored; present for graph-module signature symmetry.
            **kwargs: Ignored; present for graph-module signature symmetry.

        Returns:
            The mixed per-degree feature dict.
        """

        return {
            degree: torch.einsum("oc,ncd->nod", weight, features[degree]) for degree, weight in self.weights.items()
        }


class NormSE3(nn.Module):
    r"""
    Equivariant norm-nonlinearity: rescale each degree's feature by a learned function of its norm.

    Group-normalizes the per-degree feature magnitudes, applies a nonlinearity, and rescales the original
    (direction-preserving) features by the new magnitudes. Ported from the NVIDIA SE(3)-Transformer.

    Args:
        fiber: Fiber structure whose degrees are normalized.
        nonlinearity: Activation applied to the normalized magnitudes; defaults to ReLU.
    """

    NORM_CLAMP = 2**-24

    def __init__(self, fiber: Fiber, nonlinearity: nn.Module | None = None) -> None:
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nn.ReLU() if nonlinearity is None else nonlinearity
        self.group_norm = nn.GroupNorm(num_groups=len(fiber.degrees), num_channels=sum(fiber.channels))

    def forward(self, features: dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
        r"""
        Apply the equivariant norm-nonlinearity.

        Args:
            features: Per-degree feature dict keyed by string degree.
            *args: Ignored; present for graph-module signature symmetry.
            **kwargs: Ignored; present for graph-module signature symmetry.

        Returns:
            The rescaled per-degree feature dict.
        """

        output: dict[str, Tensor] = {}
        norms = [
            features[str(degree)].norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP) for degree in self.fiber.degrees
        ]
        fused_norms = torch.cat(norms, dim=-2)
        new_norms = self.nonlinearity(self.group_norm(fused_norms.squeeze(-1))).unsqueeze(-1)
        new_norms = torch.chunk(new_norms, chunks=len(self.fiber.degrees), dim=-2)
        for norm, new_norm, degree in zip(norms, new_norms, self.fiber.degrees):
            output[str(degree)] = features[str(degree)] / norm * new_norm
        return output


class ConvSE3(nn.Module):
    r"""
    SE(3)-equivariant graph convolution (message passing) between two fibers.

    Gathers source-node features (plus per-degree edge features) along the graph edges, applies a
    [`VersatileConvSE3`][ofoldx.models.rfdiffusion.model.VersatileConvSE3] per degree pair, optionally adds
    a self-interaction, and pools messages onto destination nodes. Ported from the NVIDIA SE(3)-Transformer.

    Args:
        fiber_in: Input node fiber.
        fiber_out: Output node fiber.
        fiber_edge: Edge fiber.
        pool: Whether to sum messages onto destination nodes (vs. returning per-edge messages).
        use_layer_norm: Whether the radial MLPs layer-norm their hidden activations.
        self_interaction: Whether to add a per-degree self-interaction from the destination node.
        max_degree: Maximum representation degree involved in the convolution.
        allow_fused_output: Whether a fused (un-split) output tensor may be returned when possible.
    """

    def __init__(
        self,
        fiber_in: Fiber,
        fiber_out: Fiber,
        fiber_edge: Fiber,
        pool: bool = True,
        use_layer_norm: bool = False,
        self_interaction: bool = False,
        max_degree: int = 1,
        allow_fused_output: bool = False,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.max_degree = max_degree
        self.allow_fused_output = allow_fused_output
        channels_in_set = {item.channels + fiber_edge[item.degree] * (item.degree > 0) for item in self.fiber_in}
        channels_out_set = {item.channels for item in self.fiber_out}
        unique_channels_in = len(channels_in_set) == 1
        degrees_up_to_max = list(range(max_degree + 1))
        edge_dim = fiber_edge[0] + 1

        if unique_channels_in and fiber_in.degrees == degrees_up_to_max:
            self.conv_out = nn.ModuleDict()
            for degree_out, channels_out in fiber_out:
                sum_freq = sum(degree_to_dim(min(degree_out, degree)) for degree in fiber_in.degrees)
                self.conv_out[str(degree_out)] = VersatileConvSE3(
                    sum_freq,
                    list(channels_in_set)[0],
                    channels_out,
                    edge_dim=edge_dim,
                    use_layer_norm=use_layer_norm,
                )
        else:
            self.conv_in = nn.ModuleDict()
            for degree_in, channels_in in fiber_in:
                sum_freq = sum(degree_to_dim(min(degree_in, degree)) for degree in fiber_out.degrees)
                self.conv_in[str(degree_in)] = VersatileConvSE3(
                    sum_freq,
                    channels_in,
                    list(channels_out_set)[0],
                    edge_dim=edge_dim,
                    use_layer_norm=use_layer_norm,
                )

        if self_interaction:
            self.to_kernel_self = nn.ParameterDict()
            for degree_out, channels_out in fiber_out:
                if fiber_in[degree_out]:
                    self.to_kernel_self[str(degree_out)] = nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out])
                    )

    def forward(
        self,
        node_feats: dict[str, Tensor],
        edge_feats: dict[str, Tensor],
        graph,
        basis: dict[str, Tensor],
    ) -> dict[str, Tensor] | Tensor:
        r"""
        Run the equivariant graph convolution.

        Args:
            node_feats: Per-degree node features keyed by string degree.
            edge_feats: Per-degree edge features; degree ``"0"`` carries the invariant radial input.
            graph: Sparse residue graph exposing ``edges()``, ``row_offsets``, and ``num_nodes``.
            basis: Per-degree-pair equivariant bases.

        Returns:
            The convolved per-degree node features, or a fused tensor when ``allow_fused_output`` applies.
        """

        invariant_edge_feats = edge_feats["0"].squeeze(-1)
        src, dst = graph.edges()
        in_features = []
        for degree_in in self.fiber_in.degrees:
            src_node_features = node_feats[str(degree_in)][src]
            if degree_in > 0 and str(degree_in) in edge_feats:
                src_node_features = torch.cat([src_node_features, edge_feats[str(degree_in)]], dim=1)
            in_features.append(src_node_features)

        out: dict[str, Tensor]
        if hasattr(self, "conv_out"):
            in_features_fused = torch.cat(in_features, dim=-1)
            out = {
                str(degree_out): self.conv_out[str(degree_out)](
                    in_features_fused, invariant_edge_feats, basis[f"out{degree_out}_fused"]
                )
                for degree_out in self.fiber_out.degrees
            }
        else:
            out_tensor = torch.zeros((), dtype=in_features[0].dtype, device=in_features[0].device)
            for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                out_tensor = out_tensor + self.conv_in[str(degree_in)](
                    feature, invariant_edge_feats, basis[f"in{degree_in}_fused"]
                )
            if not self.allow_fused_output or self.self_interaction or self.pool:
                out = unfuse_features(out_tensor, self.fiber_out.degrees)
            else:
                return out_tensor

        for degree_out in self.fiber_out.degrees:
            if self.self_interaction and str(degree_out) in self.to_kernel_self:
                dst_features = node_feats[str(degree_out)][dst]
                out[str(degree_out)] = out[str(degree_out)] + torch.einsum(
                    "oc,ecd->eod", self.to_kernel_self[str(degree_out)], dst_features
                )
            if self.pool:
                out[str(degree_out)] = segment_sum_csr(
                    out[str(degree_out)], graph.row_offsets, num_segments=graph.num_nodes
                )
        return out


class AttentionSE3(nn.Module):
    r"""
    Multi-head SE(3)-equivariant attention over graph edges.

    Scores each edge from the invariant projection of its key and destination query, applies a per-node
    (segment) softmax, and aggregates the value features onto destination nodes. Ported from the NVIDIA
    SE(3)-Transformer.

    Args:
        num_heads: Number of equivariant attention heads.
        key_fiber: Fiber of the key/query features.
        value_fiber: Fiber of the value features.
    """

    def __init__(self, num_heads: int, key_fiber: Fiber, value_fiber: Fiber) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber

    def forward(
        self,
        value: Tensor,
        key: Tensor,
        query: dict[str, Tensor],
        graph,
    ) -> dict[str, Tensor]:
        r"""
        Aggregate value features by edge attention onto destination nodes.

        Args:
            value: Fused per-edge value features.
            key: Fused per-edge key features.
            query: Per-degree destination-node query features.
            graph: Sparse residue graph exposing ``dst``, ``row_offsets``, and ``num_nodes``.

        Returns:
            The attended per-degree node features keyed by string degree.
        """

        key_heads = key.reshape(key.shape[0], self.num_heads, -1)
        query_tensor = torch.cat([query[str(degree)] for degree in self.key_fiber.degrees], dim=-1)
        query_heads = query_tensor.reshape(query_tensor.shape[0], self.num_heads, -1)

        edge_weights = (key_heads * query_heads[graph.dst]).sum(dim=-1) / np.sqrt(self.key_fiber.num_features)
        edge_weights = segment_softmax_csr(edge_weights, graph.row_offsets)[..., None, None]

        value_heads = value.view(value.shape[0], self.num_heads, -1, value.shape[-1])
        feat_out = segment_sum_csr(edge_weights * value_heads, graph.row_offsets, num_segments=graph.num_nodes)
        feat_out = feat_out.view(feat_out.shape[0], -1, feat_out.shape[-1])
        return unfuse_features(feat_out, self.value_fiber.degrees)


class AttentionBlockSE3(nn.Module):
    r"""
    One SE(3)-equivariant attention block: key/value convolution, query linear, attention, and projection.

    Computes fused keys and values with a [`ConvSE3`][ofoldx.models.rfdiffusion.model.ConvSE3], queries
    with a [`LinearSE3`][ofoldx.models.rfdiffusion.model.LinearSE3], runs
    [`AttentionSE3`][ofoldx.models.rfdiffusion.model.AttentionSE3], and projects the concatenation of the
    attended features and the residual node features to the output fiber. Ported from the NVIDIA
    SE(3)-Transformer.

    Args:
        fiber_in: Input node fiber.
        fiber_out: Output node fiber.
        fiber_edge: Edge fiber.
        num_heads: Number of equivariant attention heads.
        channels_div: Channel-division factor for the value fiber.
        use_layer_norm: Whether the convolutions layer-norm their radial hidden activations.
        max_degree: Maximum representation degree.
    """

    def __init__(
        self,
        fiber_in: Fiber,
        fiber_out: Fiber,
        fiber_edge: Fiber,
        num_heads: int = 4,
        channels_div: int = 2,
        use_layer_norm: bool = False,
        max_degree: int = 1,
    ) -> None:
        super().__init__()
        value_fiber = Fiber([(degree, channels // channels_div) for degree, channels in fiber_out])
        key_query_fiber = Fiber(
            [(item.degree, item.channels) for item in value_fiber if item.degree in fiber_in.degrees]
        )
        self.to_key_value = ConvSE3(
            fiber_in,
            value_fiber + key_query_fiber,
            pool=False,
            fiber_edge=fiber_edge,
            use_layer_norm=use_layer_norm,
            max_degree=max_degree,
            allow_fused_output=True,
        )
        self.to_query = LinearSE3(fiber_in, key_query_fiber)
        self.attention = AttentionSE3(num_heads, key_query_fiber, value_fiber)
        self.project = LinearSE3(value_fiber + fiber_in, fiber_out)

    def forward(
        self, node_features: dict[str, Tensor], edge_features: dict[str, Tensor], graph, basis: dict[str, Tensor]
    ):
        r"""
        Run one equivariant attention block.

        Args:
            node_features: Per-degree node features keyed by string degree.
            edge_features: Per-degree edge features.
            graph: Sparse residue graph.
            basis: Per-degree-pair equivariant bases.

        Returns:
            The per-degree output node features after attention and projection.
        """

        fused_key_value = self.to_key_value(node_features, edge_features, graph, basis)
        key, value = self._get_key_value_from_fused(fused_key_value)
        query = self.to_query(node_features)
        z = self.attention(value, key, query, graph)
        z_concat = {
            degree: torch.cat([value, node_features[degree]], dim=1) if degree in node_features else value
            for degree, value in z.items()
        }
        return self.project(z_concat)

    def _get_key_value_from_fused(self, fused_key_value: Tensor) -> tuple[Tensor, Tensor]:
        r"""Split the fused key/value tensor along the feature axis into ``(key, value)``."""

        value, key = torch.chunk(fused_key_value, chunks=2, dim=-2)
        return key, value


def get_populated_edge_features(
    relative_pos: Tensor, edge_features: dict[str, Tensor] | None = None
) -> dict[str, Tensor]:
    r"""
    Append the edge length as an invariant (degree-0) feature to the edge feature dict.

    Args:
        relative_pos: Per-edge relative position vectors `[E, 3]`.
        edge_features: Existing per-degree edge features, or `None` to start empty.

    Returns:
        A new edge-feature dict with the pairwise distance concatenated into degree ``"0"``.
    """

    edge_features = edge_features.copy() if edge_features else {}
    radius = relative_pos.norm(dim=-1, keepdim=True)
    if "0" in edge_features:
        edge_features["0"] = torch.cat([edge_features["0"], radius[..., None]], dim=1)
    else:
        edge_features["0"] = radius[..., None]
    return edge_features


class SE3Transformer(nn.Module):
    r"""
    Stack of equivariant attention blocks plus a final convolution over a residue graph.

    Interleaves [`AttentionBlockSE3`][ofoldx.models.rfdiffusion.model.AttentionBlockSE3] and
    [`NormSE3`][ofoldx.models.rfdiffusion.model.NormSE3] layers, then a self-interacting
    [`ConvSE3`][ofoldx.models.rfdiffusion.model.ConvSE3] readout. Ported from the NVIDIA SE(3)-Transformer.

    Args:
        num_layers: Number of attention layers.
        fiber_in: Input node fiber.
        fiber_hidden: Hidden node fiber.
        fiber_out: Output node fiber.
        num_heads: Number of equivariant attention heads.
        channels_div: Channel-division factor for the value fibers.
        fiber_edge: Edge fiber; empty when `None`.
        use_layer_norm: Whether the convolutions layer-norm their radial hidden activations.
        **kwargs: Accepted and ignored for construction compatibility.
    """

    def __init__(
        self,
        num_layers: int,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        num_heads: int,
        channels_div: int,
        fiber_edge: Fiber | None = None,
        use_layer_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        fiber_edge = Fiber({}) if fiber_edge is None else fiber_edge
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        graph_modules = []
        for _ in range(num_layers):
            graph_modules.append(
                AttentionBlockSE3(
                    fiber_in=fiber_in,
                    fiber_out=fiber_hidden,
                    fiber_edge=fiber_edge,
                    num_heads=num_heads,
                    channels_div=channels_div,
                    use_layer_norm=use_layer_norm,
                    max_degree=self.max_degree,
                )
            )
            graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden
        graph_modules.append(
            ConvSE3(
                fiber_in=fiber_in,
                fiber_out=fiber_out,
                fiber_edge=fiber_edge,
                self_interaction=True,
                use_layer_norm=use_layer_norm,
                max_degree=self.max_degree,
            )
        )
        self.graph_modules = Sequential(*graph_modules)

    def forward(
        self,
        graph,
        node_feats: dict[str, Tensor],
        edge_feats: dict[str, Tensor] | None = None,
        basis: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        r"""
        Run the equivariant transformer over the graph.

        Args:
            graph: Sparse residue graph exposing ``rel_pos``, ``edges()``, ``row_offsets``, and ``num_nodes``.
            node_feats: Per-degree input node features keyed by string degree.
            edge_feats: Per-degree input edge features, or `None`.
            basis: Precomputed equivariant bases; computed from ``graph.rel_pos`` when `None`.

        Returns:
            The per-degree output node features.
        """

        basis = basis or se3_basis(graph.rel_pos, max_degree=self.max_degree, use_pad_trick=False, fully_fused=False)
        edge_feats = get_populated_edge_features(graph.rel_pos, edge_feats)
        return self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)


class SE3TransformerWrapper(nn.Module):
    r"""
    RfDiffusion-facing adapter around [`SE3Transformer`][ofoldx.models.rfdiffusion.model.SE3Transformer].

    Constructs the input/hidden/output/edge fibers from the checkpoint's SE(3) sizing and exposes the plain
    ``(graph, type_0, type_1, edge)`` call signature the structure module uses.

    Args:
        num_layers: Number of attention layers.
        num_channels: Hidden channel width per degree.
        num_degrees: Number of hidden degrees.
        num_attention_heads: Number of equivariant attention heads.
        div: Channel-division factor for the value fibers.
        l0_in_features: Width of the incoming degree-0 (scalar) node feature.
        l0_out_features: Width of the outgoing degree-0 (scalar) node feature.
        l1_in_features: Number of incoming degree-1 (vector) node features.
        l1_out_features: Number of outgoing degree-1 (vector) node features.
        num_edge_features: Width of the scalar edge feature.
    """

    def __init__(
        self,
        num_layers: int = 2,
        num_channels: int = 32,
        num_degrees: int = 3,
        num_attention_heads: int = 4,
        div: int = 4,
        l0_in_features: int = 32,
        l0_out_features: int = 32,
        l1_in_features: int = 3,
        l1_out_features: int = 2,
        num_edge_features: int = 32,
    ) -> None:
        super().__init__()
        self.l1_in = l1_in_features
        fiber_edge = Fiber({0: num_edge_features})
        fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
        fiber_hidden = Fiber.create(num_degrees, num_channels)
        fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
        self.se3 = SE3Transformer(
            num_layers=num_layers,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=num_attention_heads,
            channels_div=div,
            fiber_edge=fiber_edge,
            use_layer_norm=True,
        )

    def forward(
        self,
        graph,
        type_0_features: Tensor,
        type_1_features: Tensor | None = None,
        edge_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        r"""
        Run the wrapped SE(3) transformer.

        Args:
            graph: Sparse residue graph.
            type_0_features: Degree-0 (scalar) node features.
            type_1_features: Degree-1 (vector) node features, or `None` when no vector input is used.
            edge_features: Scalar edge features.

        Returns:
            The per-degree output node features (``"0"`` state and ``"1"`` frame updates).
        """

        node_features = {"0": type_0_features}
        if self.l1_in > 0:
            node_features["1"] = type_1_features
        return self.se3(graph, node_features, {"0": edge_features})


DIST_BINS = 37
OMEGA_BINS = 37
THETA_BINS = 37
PHI_BINS = 19


def _zero_init(linear: nn.Linear) -> None:
    r"""Zero both the weight and bias of a linear head in place."""

    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


def _channel_first(logits: torch.Tensor) -> torch.Tensor:
    r"""Move the last (channel) axis of ``logits`` to position 1, matching the channel-first head convention."""

    return logits.movedim(-1, 1)


class DistanceNetwork(nn.Module):
    r"""
    Predict the RfDiffusion 6D distance/orientation logits from the pair representation.

    Projects the pair map into symmetric (distance, omega) and asymmetric (theta, phi) logits; the symmetric
    branch is symmetrized across the residue axes.

    Args:
        n_feat: Width of the input pair representation.
        dropout: Dropout probability (accepted for signature symmetry).
    """

    def __init__(self, n_feat: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj_symm = nn.Linear(n_feat, DIST_BINS + OMEGA_BINS)
        self.proj_asymm = nn.Linear(n_feat, THETA_BINS + PHI_BINS)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Zero-initialize the distance/orientation heads (the released RoseTTAFold scheme)."""

        _zero_init(self.proj_symm)
        _zero_init(self.proj_asymm)

    def forward(
        self,
        pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Predict the 6D geometry logits.

        Args:
            pair: Pair representation `[B, L, L, n_feat]`.

        Returns:
            A tuple ``(dist, omega, theta, phi)`` of channel-first logit maps.
        """

        theta_logits, phi_logits = self.proj_asymm(pair).split((THETA_BINS, PHI_BINS), dim=-1)

        symmetric_logits = self.proj_symm(pair)
        symmetric_logits = symmetric_logits + symmetric_logits.transpose(1, 2)
        dist_logits, omega_logits = symmetric_logits.split((DIST_BINS, OMEGA_BINS), dim=-1)

        return (
            _channel_first(dist_logits),
            _channel_first(omega_logits),
            _channel_first(theta_logits),
            _channel_first(phi_logits),
        )


class MaskedTokenNetwork(nn.Module):
    r"""
    Predict per-position amino-acid logits from the MSA representation.

    Args:
        n_feat: Width of the input MSA representation.
        vocab_size: Number of amino-acid tokens to predict over.
    """

    def __init__(self, n_feat: int, vocab_size: int = 21) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.proj = nn.Linear(n_feat, self.vocab_size)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Zero-initialize the token head (the released RoseTTAFold scheme)."""

        _zero_init(self.proj)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        r"""
        Predict amino-acid logits over all MSA positions.

        Args:
            msa: MSA representation `[B, N, L, n_feat]`.

        Returns:
            Channel-first logits `[B, vocab_size, N*L]`.
        """

        batch_size, msa_depth, length = msa.shape[:3]
        logits = self.proj(msa).movedim(-1, 1)
        return logits.reshape(batch_size, self.vocab_size, msa_depth * length)


class LDDTNetwork(nn.Module):
    r"""
    Predict binned per-residue pLDDT logits from the node state.

    Args:
        n_feat: Width of the input node-state representation.
        n_bin_lddt: Number of pLDDT bins.
    """

    def __init__(self, n_feat: int, n_bin_lddt: int = 50) -> None:
        super().__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Zero-initialize the pLDDT head (the released RoseTTAFold scheme)."""

        _zero_init(self.proj)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        Predict binned pLDDT logits.

        Args:
            state: Node-state representation `[B, L, n_feat]`.

        Returns:
            Channel-first pLDDT logits `[B, n_bin_lddt, L]`.
        """

        return self.proj(state).movedim(-1, 1)


class ExpResolvedNetwork(nn.Module):
    r"""
    Predict the per-residue "experimentally resolved" logit from the MSA row and node state.

    Args:
        msa_size: Width of the MSA row input.
        state_size: Width of the node-state input.
        dropout: Dropout probability (accepted for signature symmetry).
    """

    def __init__(self, msa_size: int, state_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_msa = nn.LayerNorm(msa_size)
        self.norm_state = nn.LayerNorm(state_size)
        self.proj = nn.Linear(msa_size + state_size, 1)
        self.reset_parameter()

    def reset_parameter(self) -> None:
        r"""Zero-initialize the resolved head (the released RoseTTAFold scheme)."""

        _zero_init(self.proj)

    def forward(self, seq: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        r"""
        Predict the per-residue resolved logit.

        Args:
            seq: MSA query row `[B, L, msa_size]`.
            state: Node-state representation `[B, L, state_size]`.

        Returns:
            The per-residue resolved logit `[B, L]`.
        """

        normalized_seq = self.norm_msa(seq)
        normalized_state = self.norm_state(state)
        logits = self.proj(torch.cat((normalized_seq, normalized_state), dim=-1))
        return logits.squeeze(-1)


def find_breaks(ix: np.ndarray, thresh: int = 35) -> np.ndarray:
    r"""
    Return the positions where the residue index jumps by more than ``thresh`` (chain breaks).

    Args:
        ix: 1D residue-index array.
        thresh: Minimum forward jump that counts as a chain break.

    Returns:
        A 1D array of break positions (indices into ``ix``).

    Examples:
        >>> import numpy as np
        >>> from ofoldx.models.rfdiffusion.model import find_breaks
        >>> find_breaks(np.array([1, 2, 3, 100, 101])).tolist()
        [3]
    """

    return np.where(np.diff(ix) > thresh)[0] + 1


def init_lecun_normal(module: nn.Module) -> nn.Module:
    r"""
    Re-initialize a layer's weight in place with the truncated Lecun-normal scheme, returning the layer.

    Args:
        module: A layer exposing a ``weight`` parameter.

    Returns:
        The same ``module`` with its weight replaced.
    """

    module.weight = nn.Parameter(init_lecun_normal_param(module.weight))
    return module


def init_lecun_normal_param(weight: torch.Tensor) -> torch.Tensor:
    r"""
    Draw a truncated Lecun-normal initialization tensor shaped like ``weight``.

    Args:
        weight: Tensor whose shape (and its fan-in) sets the output shape and scale.

    Returns:
        A new tensor of the same shape, drawn from the truncated Lecun-normal distribution.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import init_lecun_normal_param
        >>> _ = torch.manual_seed(0)
        >>> init_lecun_normal_param(torch.zeros(2, 3)).shape
        torch.Size([2, 3])
    """

    def truncated_normal(uniform: torch.Tensor, mu: float = 0.0, sigma: float = 1.0, a: int = -2, b: int = 2):
        r"""
        Map uniform samples to a normal distribution truncated to ``[a, b]`` via the inverse-CDF.

        Args:
            uniform: Samples drawn from `U(0, 1)` to be transformed; any shape.
            mu: Mean of the underlying (untruncated) normal.
            sigma: Standard deviation of the underlying (untruncated) normal.
            a: Lower truncation bound, in standard deviations from ``mu``.
            b: Upper truncation bound, in standard deviations from ``mu``.

        Returns:
            Truncated-normal samples of the same shape as ``uniform``, clamped to `[a, b]`.
        """

        normal = torch.distributions.normal.Normal(0, 1)
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        alpha_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_cdf + (normal.cdf(torch.tensor(beta)) - alpha_cdf) * uniform
        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        return torch.clamp(mu + sigma * np.sqrt(2) * torch.erfinv(v), a, b)

    stddev = np.sqrt(1.0 / weight.shape[-1]) / 0.87962566103423978
    return stddev * truncated_normal(torch.rand(weight.shape, device=weight.device, dtype=weight.dtype))


class Dropout(nn.Module):
    r"""
    Structured dropout that can share the drop mask along one axis.

    When ``broadcast_dim`` is set, the same Bernoulli mask is applied across that axis (e.g. dropping whole
    rows or columns of the pair map), matching the RoseTTAFold axial-dropout behavior.

    Args:
        broadcast_dim: Axis to broadcast the drop mask over, or `None` for standard elementwise dropout.
        dropout: Drop probability.
    """

    def __init__(self, broadcast_dim: int | None = None, dropout: float = 0.15) -> None:
        super().__init__()
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1 - dropout]))
        self.broadcast_dim = broadcast_dim
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply structured dropout to ``x`` (identity in eval mode).

        Args:
            x: Input tensor.

        Returns:
            The dropped, rescaled tensor while training; ``x`` unchanged otherwise.
        """

        if not self.training:
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)
        return mask * x / (1.0 - self.dropout)


def rbf(distances: torch.Tensor) -> torch.Tensor:
    r"""
    Expand pairwise ``distances`` into 36 Gaussian radial-basis features over 0-20 Angstrom.

    Args:
        distances: Pairwise distances of any shape `[...]`.

    Returns:
        Radial-basis features `[..., 36]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import rbf
        >>> rbf(torch.tensor([[0.0, 5.0]])).shape
        torch.Size([1, 2, 36])
    """

    d_min, d_max, d_count = 0.0, 20.0, 36
    d_mu = torch.linspace(d_min, d_max, d_count, device=distances.device, dtype=distances.dtype)[None, :]
    d_sigma = (d_max - d_min) / d_count
    return torch.exp(-(((distances.unsqueeze(-1) - d_mu) / d_sigma) ** 2))


def get_seqsep(idx: torch.Tensor, cyclic: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Build the signed sequence-neighbor edge feature (+/-1 for adjacent residues, 0 otherwise).

    Adjacent-in-sequence residues within a chain get a signed one; cyclic chains additionally link their first
    and last residues so the topology closes.

    Args:
        idx: Residue-index tensor `[B, L]`.
        cyclic: Optional cyclic-residue mask closing each cyclic chain.

    Returns:
        The neighbor feature `[B, L, L, 1]`.
    """

    seqsep = idx[:, None, :] - idx[:, :, None]
    sign = torch.sign(seqsep)
    neigh = torch.abs(seqsep)
    neigh[neigh > 1] = 0.0
    neigh = sign * neigh

    breaks = find_breaks(idx.squeeze().detach().cpu().numpy())
    chainids = np.zeros_like(idx.squeeze().detach().cpu().numpy())
    for i, break_index in enumerate(breaks):
        chainids[break_index:] = i + 1
    chainids = torch.from_numpy(chainids).to(device=idx.device)

    if cyclic is not None:
        for chain_id in torch.unique(chainids):
            is_chain = chainids == chain_id
            cur_cyclic = cyclic * is_chain
            cur_res = cur_cyclic.nonzero()
            if cur_cyclic.sum() >= 2:
                neigh[:, cur_res[-1], cur_res[0]] = 1
                neigh[:, cur_res[0], cur_res[-1]] = -1

    return neigh.unsqueeze(-1)


class RFGraph:
    r"""
    Sparse residue-residue graph consumed by the SE(3) transformer.

    Holds a CSR edge structure with per-edge relative positions and pair-derived edge features, built from
    backbone ``coords`` and the pair representation over the ``(src_res, dst_res)`` edge list. Edges are sorted
    by destination so ``row_offsets`` gives the CSR row boundaries for the segment reductions.

    Args:
        coords: Backbone CA coords `[B, L, 3]` used for the relative-position edges.
        pair: Pair representation `[B, L, L, C]` providing the edge features.
        batch: Per-edge batch index.
        src_res: Per-edge source residue index.
        dst_res: Per-edge destination residue index.
        batch_size: Number of graphs (batch elements).
        length: Number of residues per graph.
    """

    __slots__ = ("src", "dst", "rel_pos", "edge_feats", "num_nodes", "row_offsets")

    def __init__(
        self,
        coords: torch.Tensor,
        pair: torch.Tensor,
        batch: torch.Tensor,
        src_res: torch.Tensor,
        dst_res: torch.Tensor,
        *,
        batch_size: int,
        length: int,
    ) -> None:
        src = batch * length + src_res
        dst = batch * length + dst_res
        order = torch.argsort(dst * (batch_size * length) + src, stable=True)
        src, dst = src[order], dst[order]
        batch, src_res, dst_res = batch[order], src_res[order], dst_res[order]
        self.src = src
        self.dst = dst
        self.rel_pos = (coords[batch, dst_res, :] - coords[batch, src_res, :]).detach()
        self.edge_feats = pair[batch, src_res, dst_res][..., None]
        self.num_nodes = batch_size * length
        counts = torch.bincount(dst, minlength=self.num_nodes)
        self.row_offsets = torch.zeros(self.num_nodes + 1, dtype=torch.long, device=coords.device)
        self.row_offsets[1:] = torch.cumsum(counts, dim=0)

    def edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the ``(src, dst)`` global node-index tensors of the edge list."""

        return self.src, self.dst


def make_full_graph(
    coords: torch.Tensor,
    pair: torch.Tensor,
    idx: torch.Tensor,
) -> tuple[RFGraph, torch.Tensor]:
    r"""
    Build a fully-connected residue graph over all distinct residue pairs.

    Args:
        coords: Backbone CA coords `[B, L, 3]`.
        pair: Pair representation `[B, L, L, C]`.
        idx: Residue-index tensor `[B, L]`.

    Returns:
        A ``(graph, edge_feats)`` pair: the [`RFGraph`][ofoldx.models.rfdiffusion.model.RFGraph] and its
        per-edge features.
    """

    batch_size, length = coords.shape[:2]
    sep = idx[:, None, :] - idx[:, :, None]
    batch, src_res, dst_res = torch.where(sep.abs() > 0)
    graph = RFGraph(coords, pair, batch, src_res, dst_res, batch_size=batch_size, length=length)
    return graph, graph.edge_feats


def make_topk_graph(
    coords: torch.Tensor,
    pair: torch.Tensor,
    idx: torch.Tensor,
    top_k: int = 64,
    kmin: int = 32,
    eps: float = 1e-6,
) -> tuple[RFGraph, torch.Tensor]:
    r"""
    Build a k-nearest-neighbor residue graph (plus close-in-sequence edges).

    Keeps each residue's ``top_k`` nearest neighbors by CA distance, breaking ties by sequence separation, and
    additionally includes all pairs closer than ``kmin`` in sequence.

    Args:
        coords: Backbone CA coords `[B, L, 3]`.
        pair: Pair representation `[B, L, L, C]`.
        idx: Residue-index tensor `[B, L]`.
        top_k: Number of spatial neighbors per residue.
        kmin: Sequence-separation threshold below which pairs are always connected.
        eps: Tie-breaking weight on sequence separation.

    Returns:
        A ``(graph, edge_feats)`` pair: the [`RFGraph`][ofoldx.models.rfdiffusion.model.RFGraph] and its
        per-edge features.
    """

    batch_size, length = coords.shape[:2]
    distances = torch.cdist(coords, coords) + torch.eye(length, device=coords.device).unsqueeze(0) * 999.9
    sep = (idx[:, None, :] - idx[:, :, None]).abs() + torch.eye(length, device=coords.device).unsqueeze(0) * 999.9
    distances = distances + sep * eps
    _, edge_index = torch.topk(distances, min(top_k, length), largest=False)
    topk_matrix = torch.zeros((batch_size, length, length), device=coords.device)
    topk_matrix.scatter_(2, edge_index, 1.0)
    condition = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    batch, src_res, dst_res = torch.where(condition)
    graph = RFGraph(coords, pair, batch, src_res, dst_res, batch_size=batch_size, length=length)
    return graph, graph.edge_feats


@dataclass(slots=True, kw_only=True)
class RfDiffusionInput(BaseInput):
    r"""
    Tensor input consumed by the OFoldX-native RfDiffusion generator.

    Produced by [`RfDiffusionProcessor`][ofoldx.models.rfdiffusion.processor.RfDiffusionProcessor]. All
    tensors carry a leading batch dimension. Motif residues (fixed structure) and design residues (redesigned)
    are distinguished by the masks; optional contig/inpaint fields override the config-level specification.

    Attributes:
        token_mask: Per-token validity mask `[B, L]`.
        residue_index: Residue-position index `[B, L]`.
        chain_index: Per-token chain index `[B, L]`.
        residue_type: Per-token amino-acid type id `[B, L]`.
        motif_positions: Representative-atom (CA) coords for motif conditioning `[B, L, 3]`.
        motif_mask: Per-token mask marking fixed motif residues `[B, L]`.
        design_mask: Per-token mask marking residues to redesign `[B, L]`; `None` when unset.
        sequence_tokens: Per-token sequence ids `[B, L]`; `None` when unset.
        target_features: Optional precomputed target feature dict bypassing PDB parsing.
        contigs: Optional contig strings overriding the config-level specification.
        inpaint_seq: Optional sequence-inpainting spans.
        inpaint_str: Optional structure-inpainting spans.
        provide_seq: Optional spans whose sequence is provided (unmasked).
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    residue_type: Tensor
    motif_positions: Tensor
    motif_mask: Tensor
    design_mask: Tensor | None = None
    sequence_tokens: Tensor | None = None
    target_features: dict[str, Any] | None = None
    contigs: list[str] | None = None
    inpaint_seq: list[str] | None = None
    inpaint_str: list[str] | None = None
    provide_seq: list[str] | None = None
