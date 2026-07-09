r"""
RfDiffusion design model.
"""

from __future__ import annotations

import contextlib
import functools as fn
import glob
import math
import os
import pickle
import random
import string
from collections import namedtuple
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from oops import se3_basis, segment_softmax_csr, segment_sum_csr, weighted_tensor_product
from opt_einsum import contract as einsum
from scipy.spatial.transform import Rotation
from torch import Tensor, nn

from ..contract import BaseInput, DesignGeneratorOutput, StructurePredictionOutput
from ..mixin import ModelMixin
from ..registry import DESIGN_GENERATOR_MODELS, MODELS
from .chemical import (
    AA3_TO_1,
    ATOM14_NAMES,
    CHI_ATOM_NAMES,
    INIT_CRDS,
    MASK_TOKEN,
    NUM2AA,
    rf_all_atom_kinematics,
)
from .config import RfDiffusionConfig


@MODELS.register("rfdiffusion")
@DESIGN_GENERATOR_MODELS.register("rfdiffusion")
class RfDiffusionDesignGenerator(ModelMixin):
    r"""
    Denoising-diffusion model that generates novel protein backbones from a noised structure (RfDiffusion).

    Public entry point for the RfDiffusion architecture. Holds the RoseTTAFold network
    ([`RoseTTAFoldModule`][ofoldx.models.rfdiffusion.model.RoseTTAFoldModule]) as its ``network``
    submodule and drives the diffusion sampling loop through
    [`NativeRfDiffusionSampler`][ofoldx.models.rfdiffusion.model.NativeRfDiffusionSampler]. Given a
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

        return NativeRfDiffusionSampler(self.config, network=self.network)


class RoseTTAFoldModule(nn.Module):
    r"""
    RoseTTAFold trunk driving one RfDiffusion denoising step.

    Ported RoseTTAFold network: embeds the MSA/pair/state representations, injects template and recycling
    features, runs the [`IterativeSimulator`][ofoldx.models.rfdiffusion.model.IterativeSimulator] track,
    and reads out amino-acid logits, pLDDT, and the denoised backbone. The
    [`NativeRfDiffusionSampler`][ofoldx.models.rfdiffusion.model.NativeRfDiffusionSampler] calls this
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
    from scipy.spatial.transform import Rotation
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


class NativeRfDiffusionSampler:
    r"""
    Native RFdiffusion reverse-diffusion inference loop.

    Drives one design trajectory: builds the initial noised structure and sequence features, then repeatedly
    calls the RoseTTAFold network and steps the [`Denoiser`][ofoldx.models.rfdiffusion.model.Denoiser]
    from ``x_T`` down to the final timestep. Handles contigs, symmetry, guiding potentials, and scaffold
    guidance. The network is supplied at construction, so this module never imports it.

    Args:
        config: The [`RfDiffusionConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionConfig] controlling
            the diffuser, denoiser, and inference behavior.
        network: The RoseTTAFold network to call each step, or `None` (sampling then raises).
        device: Device to run on; inferred from ``network`` (or CUDA-if-available) when omitted.
    """

    def __init__(
        self,
        config: RfDiffusionConfig,
        *,
        network: nn.Module | None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.network = network
        self.device = torch.device(device) if device is not None else self._infer_device(network)
        self.allatom = ComputeAllAtomCoords().to(self.device)

    def sample(
        self,
        batch: Any | None = None,
        *,
        num_sampling_steps: int | None = None,
        diffusion_steps: int | None = None,
        final_step: int | None = None,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionSampleResult:
        r"""
        Run one RFdiffusion reverse-diffusion trajectory end to end.

        Initializes the noised structure, then steps from the starting timestep down to ``final_step``, calling
        the network and denoiser each step and accumulating the coord trajectory.

        Args:
            batch: A tokenized RfDiffusion input, or `None` to sample from ``inference.input_pdb``.
            num_sampling_steps: Starting timestep, capped at the diffuser's timestep count.
            diffusion_steps: Override for the diffuser's total timestep count.
            final_step: Last (smallest) timestep to denoise to; defaults to ``inference.final_step``.
            generator: Optional RNG for reproducible sampling.

        Returns:
            An [`RfDiffusionSampleResult`][ofoldx.models.rfdiffusion.model.RfDiffusionSampleResult] with
            the final coords, predicted sequence, optional pLDDT, trajectories, and metadata.

        Raises:
            RuntimeError: If no network is attached to the sampler.

        Examples:
            >>> result = sampler.sample(batch)  # doctest: +SKIP
        """

        if self.network is None:
            raise RuntimeError(
                "RFdiffusion sampler/features are available, but no native RFdiffusion network is attached. "
                "Set RfDiffusionDesignGenerator.network to a module with the official inference signature."
            )
        state = self.sample_init(batch, diffusion_steps=diffusion_steps, generator=generator)
        start_t = state.t_step_input if num_sampling_steps is None else min(int(num_sampling_steps), state.t_step_input)
        stop_t = int(final_step if final_step is not None else self.config.inference.final_step)
        pred_coords_traj: list[Tensor] = []
        noisy_coords_traj: list[Tensor] = [state.noisy_coords.detach().clone()]
        plddt: Tensor | None = None
        seq_t = state.seq_t
        noisy_coords = state.noisy_coords
        for t in range(start_t, stop_t - 1, -1):
            pred_coords, noisy_coords, seq_t, plddt = self.sample_step(
                state=state, t=t, noisy_coords=noisy_coords, seq_init=seq_t, final_step=stop_t, generator=generator
            )
            pred_coords_traj.append(pred_coords.detach().clone())
            noisy_coords_traj.append(noisy_coords.detach().clone())
        sequence_tokens = seq_t.argmax(dim=-1) if seq_t.ndim == 2 else seq_t.long()
        return RfDiffusionSampleResult(
            coords=noisy_coords,
            sequence_tokens=sequence_tokens,
            plddt=plddt,
            trajectory=torch.stack(noisy_coords_traj),
            pred_coords_trajectory=torch.stack(pred_coords_traj) if pred_coords_traj else noisy_coords[None],
            metadata={
                "architecture": "rfdiffusion",
                "backend": "native_rfdiffusion_sampler",
                "contig": state.contig_map.sampled_mask,
                "binder_length": state.binderlen,
                "sequence": seq_to_chars(sequence_tokens),
                "timesteps": list(range(start_t, stop_t - 1, -1)),
            },
        )

    def sample_init(
        self,
        batch: Any | None = None,
        *,
        diffusion_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionSampleState:
        r"""
        Build the initial noised structure ``x_T`` and sequence features for one design.

        Dispatches to the scaffold-guided or base initializer depending on the configuration.

        Args:
            batch: A tokenized RfDiffusion input, or `None` to sample from ``inference.input_pdb``.
            diffusion_steps: Override for the diffuser's total timestep count.
            generator: Optional RNG for reproducible noising.

        Returns:
            An [`RfDiffusionSampleState`][ofoldx.models.rfdiffusion.model.RfDiffusionSampleState]
            holding the noised coords, sequence, contig map, masks, and diffusion helpers.

        Examples:
            >>> state = sampler.sample_init(batch)  # doctest: +SKIP
        """

        if self._uses_scaffold_guidance():
            return self._sample_init_scaffolded(batch, diffusion_steps=diffusion_steps, generator=generator)
        return self._sample_init_base(batch, diffusion_steps=diffusion_steps, generator=generator)

    def _sample_init_base(
        self,
        batch: Any | None = None,
        *,
        diffusion_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionSampleState:
        r"""Build the initial trajectory state for a standard (non-scaffold-guided) design."""

        target_feats = self._target_features(batch)
        contig_map = self._construct_contig(target_feats, batch)
        mask_seq = torch.from_numpy(contig_map.inpaint_seq).bool()[None].to(self.device)
        mask_str = torch.from_numpy(contig_map.inpaint_str).bool()[None].to(self.device)
        binderlen = len(contig_map.inpaint)
        diffusion_mask = mask_str.squeeze(0)
        potential_manager = PotentialManager(
            self.config.potentials,
            self.config.diffuser,
            self.config.inference,
            binderlen,
        )
        diffuser = self._make_diffuser(diffusion_steps=diffusion_steps)
        coords27 = target_feats["xyz_27"].to(self.device)
        mask_27 = target_feats["mask_27"].to(self.device)
        seq_orig = target_feats["seq"].to(self.device)
        L_mapped = len(contig_map.reference)
        if self.config.diffuser.partial_timesteps:
            if coords27.shape[0] != L_mapped:
                raise ValueError("partial diffusion requires one input coordinate per contig residue")
            if contig_map.design_index0 != contig_map.reference_index0:
                raise ValueError("partial diffusion requires matching input/output residue indices")
            mapped_coords = coords27
            motif_prealign_coords = mapped_coords.clone()
            atom_mask_mapped = mask_27
        else:
            mapped_coords = torch.full((1, 1, L_mapped, 27, 3), float("nan"), device=self.device)
            mapped_coords[:, :, contig_map.design_index0] = coords27[contig_map.reference_index0]
            motif_prealign_coords = mapped_coords.squeeze(0).squeeze(0).clone()
            mapped_coords = get_init_coords(mapped_coords).squeeze(0).squeeze(0)
            atom_mask_mapped = torch.zeros((L_mapped, 27), dtype=torch.bool, device=self.device)
            atom_mask_mapped[contig_map.design_index0] = mask_27[contig_map.reference_index0]

        t_step_input = self._t_step_input(diffusion_steps)
        t_list = np.arange(1, t_step_input + 1)
        seq_t = torch.full((L_mapped,), MASK_TOKEN, dtype=torch.long, device=self.device)
        seq_t[contig_map.design_index0] = seq_orig[contig_map.reference_index0]
        if self.config.contigmap.provide_seq is not None:
            seq_t[mask_seq.squeeze(0)] = seq_orig[mask_seq.squeeze(0)]
        seq_t[~mask_seq.squeeze(0)] = MASK_TOKEN
        seq_t_oh = one_hot_sequence(seq_t)
        fa_stack, _true_coords = diffuser.diffuse_pose(
            mapped_coords,
            seq_t_oh.clone(),
            atom_mask_mapped.squeeze(),
            diffusion_mask=diffusion_mask,
            t_list=t_list,
            include_motif_sidechains=self.config.preprocess.motif_sidechain_input,
            generator=generator,
        )
        noisy_coords = fa_stack[-1, :, :14]
        self._attach_substrate_potentials(
            target_feats=target_feats,
            potential_manager=potential_manager,
            diffusion_mask=diffusion_mask,
            motif_coords=motif_prealign_coords,
            diffuser=diffuser,
        )
        denoiser = self._make_denoiser(L_mapped, diffuser=diffuser, potential_manager=potential_manager)
        symmetry = self._make_symmetry()
        if symmetry is not None:
            noisy_coords, seq_t_oh = symmetry.apply_symmetry(noisy_coords, seq_t_oh)
        chain_idx, pdb_index = self._build_output_indices(contig_map)
        cyclic_reses = self._init_cyclic_reses(mask_str, contig_map)
        return RfDiffusionSampleState(
            noisy_coords=noisy_coords,
            seq_t=seq_t_oh,
            contig_map=contig_map,
            mask_seq=mask_seq,
            mask_str=mask_str,
            diffusion_mask=diffusion_mask,
            binderlen=binderlen,
            chain_idx=chain_idx,
            pdb_index=pdb_index,
            t_step_input=t_step_input,
            diffuser=diffuser,
            denoiser=denoiser,
            potential_manager=potential_manager,
            symmetry=symmetry,
            cyclic_reses=cyclic_reses,
        )

    def sample_step(
        self,
        *,
        state: RfDiffusionSampleState,
        t: int,
        noisy_coords: Tensor,
        seq_init: Tensor,
        final_step: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        r"""
        Run one reverse-diffusion step at timestep ``t``.

        Prepares the network inputs, runs the network, reconstructs all-atom coords, applies symmetry and
        motif alignment, and denoises one step toward ``x_{t-1}`` (or writes the prediction directly at the
        final step).

        Args:
            state: The mutable trajectory state.
            t: Current timestep.
            noisy_coords: Current noised backbone coords `[L, 14, 3]`.
            seq_init: Current sequence features.
            final_step: The last timestep of the trajectory.
            generator: Optional RNG for reproducible denoising.

        Returns:
            A tuple ``(pred_coords, next_noisy_coords, next_seq, plddt)`` for this step.
        """

        features = self.preprocess(state, seq_init, noisy_coords, t)
        pdb_index = features.pdb_index
        if state.symmetry is not None:
            pdb_index, state.chain_idx = state.symmetry.res_idx_procesing(res_idx=pdb_index)
        output = self.network(  # type: ignore[misc,operator]
            features.msa_masked,
            features.msa_full,
            features.seq_in,
            features.noisy_coords_in,
            pdb_index,
            template_1d=features.template_1d,
            template_2d=features.template_2d,
            template_coords=features.template_coords,
            alpha_t=features.alpha_t,
            msa_prev=None,
            pair_prev=None,
            state_prev=None,
            t=torch.tensor(t, device=self.device),
            motif_mask=state.diffusion_mask.to(self.device),
            cyclic_reses=state.cyclic_reses,
        )
        pred_coords_raw, alpha, plddt = coerce_network_prediction(output)
        seq_tokens = torch.argmax(features.seq_in, dim=-1)
        _frames, pred_coords_full = self.allatom(
            seq_tokens, normalize_backbone_prediction(pred_coords_raw, noisy_coords), alpha
        )
        pred_coords = pred_coords_full.squeeze(0)[:, :14]
        if state.symmetry is not None and self.config.inference.symmetric_self_cond:
            pred_coords, _ = state.symmetry.apply_symmetry(pred_coords, seq_init)
        state.prev_pred = pred_coords.detach().clone()
        seq_t_1 = seq_init.clone()
        if t > final_step:
            noisy_coords_prev, pred_coords = state.denoiser.get_next_pose(
                noisy_coords=noisy_coords,
                pred_coords=pred_coords,
                t=t,
                diffusion_mask=state.diffusion_mask,
                align_motif=self.config.inference.align_motif,
                include_motif_sidechains=self.config.preprocess.motif_sidechain_input,
                generator=generator,
            )
        else:
            noisy_coords_prev = pred_coords.to(noisy_coords.device)
        if state.symmetry is not None:
            noisy_coords_prev, seq_t_1 = state.symmetry.apply_symmetry(noisy_coords_prev, seq_t_1)
        return pred_coords, noisy_coords_prev, seq_t_1, plddt

    def preprocess(
        self, state: RfDiffusionSampleState, seq: Tensor, template_coords_in: Tensor, t: int
    ) -> RfDiffusionStepFeatures:
        r"""
        Assemble the RoseTTAFold network input tensors for timestep ``t``.

        Builds the masked/full MSA, 1D/2D template features (including time, hotspot, and scaffold SS/adjacency
        features when the checkpoint expects them), template coords, and torsion features, wiring in
        self-conditioning from the previous prediction.

        Args:
            state: The mutable trajectory state.
            seq: Current one-hot sequence features `[L, 22]`.
            template_coords_in: Current template backbone coords `[L, A, 3]`.
            t: Current timestep.

        Returns:
            An [`RfDiffusionStepFeatures`][ofoldx.models.rfdiffusion.model.RfDiffusionStepFeatures]
            bundle for one network call.

        Raises:
            ValueError: If the checkpoint expects scaffold SS/adjacency features that are absent from ``state``.
        """

        L = seq.shape[0]
        msa_masked = torch.zeros((1, 1, L, 48), dtype=seq.dtype, device=self.device)
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
        msa_full = torch.zeros((1, 1, L, 25), dtype=seq.dtype, device=self.device)
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
        template_1d = torch.zeros((1, 1, L, 21), dtype=seq.dtype, device=self.device)
        seq_template_1d = seq.clone()
        mask_token_positions = seq_template_1d[:, MASK_TOKEN] == 1
        seq_template_1d[mask_token_positions, 20] = 1
        seq_template_1d[mask_token_positions, MASK_TOKEN] = 0
        template_1d[:, :, :, :21] = seq_template_1d[None, None, :, :21]
        timefeature = torch.zeros((L,), dtype=seq.dtype, device=self.device)
        timefeature[state.mask_str.squeeze(0)] = 1
        timefeature[~state.mask_str.squeeze(0)] = 1 - t / self.config.diffuser.num_timesteps
        template_1d = torch.cat((template_1d, timefeature[None, None, :, None]), dim=-1).float()

        working_coords = template_coords_in.clone()
        if self.config.preprocess.sidechain_input:
            working_coords[mask_token_positions, 3:] = float("nan")
        else:
            working_coords[~state.mask_str.squeeze(0), 3:] = float("nan")
        template_coords = working_coords[None, None]
        template_coords = torch.cat(
            (
                template_coords,
                torch.full((1, 1, L, 13, 3), float("nan"), dtype=template_coords.dtype, device=self.device),
            ),
            dim=3,
        )
        template_2d = coords_to_t2d(template_coords)
        if self.config.inference.model_runner == "SelfConditioning":
            if (
                state.prev_pred is not None
                and (t < state.diffuser.num_timesteps)
                and (t != self.config.diffuser.partial_timesteps)
            ):
                prev = state.prev_pred[:, :3][None, None]
                zeros = torch.zeros((1, 1, L, 24, 3), dtype=prev.dtype, device=prev.device)
                sidechain_coords = torch.cat((prev, zeros), dim=-2)
                template_2d[..., :44] = coords_to_t2d(sidechain_coords)
            else:
                template_2d[..., :44] = 0
        idx = torch.tensor(state.contig_map.residue_index, dtype=torch.long, device=self.device)[None]
        seq_tmp = template_1d[..., :-1].argmax(dim=-1).reshape(1, L)
        alpha_t = make_alpha_t(template_coords, seq_tmp)

        if self.config.preprocess.template_1d_size >= 24:
            hotspot_tens = torch.zeros(L, dtype=seq.dtype, device=self.device)
            hotspot_idx = []
            if self.config.ppi.hotspot_res is not None:
                hotspots = [(item[0], int(item[1:])) for item in self.config.ppi.hotspot_res]
                for idx_map, residue in enumerate(state.contig_map.contig_reference_pdb_index):
                    if residue in hotspots:
                        hotspot_idx.append(state.contig_map.design_index0[idx_map])
            hotspot_tens[hotspot_idx] = 1.0
            template_1d = torch.cat(
                (template_1d, torch.zeros_like(template_1d[..., :1]), hotspot_tens[None, None, :, None]), dim=-1
            )

        if self.config.preprocess.template_1d_size == 28:
            if state.ss is None:
                raise ValueError(
                    "checkpoint expects scaffold secondary-structure features, but no SS tensor is present"
                )
            full_ss = state.ss
            if state.target_pdb is not None:
                blank_ss = F.one_hot(torch.full((L - state.binderlen,), 3, device=self.device), num_classes=4)
                full_ss = torch.cat((state.ss, blank_ss), dim=0)
                if state.target_ss is not None:
                    full_ss[state.binderlen :] = state.target_ss
            template_1d = torch.cat((template_1d, full_ss[None, None].to(self.device).float()), dim=-1)
        if self.config.preprocess.template_2d_size == 47:
            if state.adj is None:
                raise ValueError("checkpoint expects scaffold adjacency features, but no adjacency tensor is present")
            full_adj = state.adj
            if state.target_pdb is not None:
                full_adj = torch.zeros((L, L, 3), dtype=template_2d.dtype, device=self.device)
                full_adj[:, :, -1] = 1.0
                full_adj[: state.binderlen, : state.binderlen] = state.adj.to(self.device)
                if state.target_adj is not None:
                    full_adj[state.binderlen :, state.binderlen :] = state.target_adj
            template_2d = torch.cat((template_2d, full_adj[None, None].to(self.device).float()), dim=-1)
            if state.target_pdb is not None:
                idx[:, state.binderlen :] += 200

        return RfDiffusionStepFeatures(
            msa_masked=msa_masked,
            msa_full=msa_full,
            seq_in=seq[None],
            noisy_coords_in=torch.squeeze(template_coords, dim=0),
            pdb_index=idx,
            template_1d=template_1d,
            template_2d=template_2d,
            template_coords=template_coords,
            alpha_t=alpha_t,
        )

    def _sample_init_scaffolded(
        self,
        batch: Any | None = None,
        *,
        diffusion_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionSampleState:
        r"""Build the initial trajectory state for scaffold-guided (fold-conditioned) design."""

        scaffold_length, ss, adj, target_pdb, target_ss, target_adj = self._scaffold_inputs()
        if self.config.contigmap.contigs is None:
            return self._sample_init_scaffold_auto(
                scaffold_length=scaffold_length,
                ss=ss,
                adj=adj,
                target_pdb=target_pdb,
                target_ss=target_ss,
                target_adj=target_adj,
                diffusion_steps=diffusion_steps,
                generator=generator,
            )
        state = self._sample_init_base(batch, diffusion_steps=diffusion_steps, generator=generator)
        if hasattr(state.contig_map, "ss_spec"):
            ss = ss_from_contig(state.contig_map.ss_spec).to(self.device)
            if adj is None:
                adj = _masked_adjacency(ss.shape[0], self.device)
        state.ss = ss
        state.adj = adj
        state.target_pdb = target_pdb
        state.target_ss = target_ss
        state.target_adj = target_adj
        return state

    def _scaffold_inputs(
        self,
    ) -> tuple[int, Tensor | None, Tensor | None, dict[str, Any] | None, Tensor | None, Tensor | None]:
        r"""
        Load scaffold and target inputs for scaffold-guided sampling.

        Returns:
            A tuple ``(scaffold_length, ss, adj, target_pdb, target_ss, target_adj)`` of the sampled scaffold
            length, its secondary-structure and adjacency tensors, and the optional target and its features.
        """

        if self.config.scaffoldguided.scaffold_dir is not None:
            blockadjacency = BlockAdjacency(self.config, self.config.inference.num_designs)
            scaffold_length, ss, adj_raw = blockadjacency.get_scaffold()
            ss = ss.to(self.device)
            adj = F.one_hot(adj_raw.long(), num_classes=3).to(self.device)
        else:
            scaffold_length = 100
            ss = None
            adj = None
        target = None
        target_pdb = None
        target_ss = None
        target_adj = None
        if self.config.scaffoldguided.target_pdb:
            target = Target(self.config.scaffoldguided, self.config.ppi.hotspot_res)
            target_pdb = target.get_target()
            if self.config.scaffoldguided.target_ss is not None:
                loaded_ss = torch.load(self.config.scaffoldguided.target_ss, weights_only=True).long()
                target_ss = F.one_hot(loaded_ss, num_classes=4).to(self.device)
                if self.config.scaffoldguided.contig_crop is not None:
                    target_ss = target_ss[target_pdb["crop_mask"]]
            if self.config.scaffoldguided.target_adj is not None:
                loaded_adj = torch.load(self.config.scaffoldguided.target_adj, weights_only=True).long()
                target_adj = F.one_hot(loaded_adj, num_classes=3).to(self.device)
                if self.config.scaffoldguided.contig_crop is not None:
                    target_adj = target_adj[target_pdb["crop_mask"]][:, target_pdb["crop_mask"]]
        return scaffold_length, ss, adj, target_pdb, target_ss, target_adj

    def _sample_init_scaffold_auto(
        self,
        *,
        scaffold_length: int,
        ss: Tensor | None,
        adj: Tensor | None,
        target_pdb: dict[str, Any] | None,
        target_ss: Tensor | None,
        target_adj: Tensor | None,
        diffusion_steps: int | None,
        generator: torch.Generator | None,
    ) -> RfDiffusionSampleState:
        r"""Build the trajectory state for scaffold-guided design when no contigs are given (auto contigs)."""

        if ss is None:
            ss = _masked_secondary_structure(scaffold_length, self.device)
        if adj is None:
            adj = _masked_adjacency(scaffold_length, self.device)
        noisy_coords, seq_t, atom_mask, diffusion_mask, contig_target, binderlen = self._auto_scaffold_tensors(
            scaffold_length, target_pdb
        )
        contig_map = ContigMap(
            False if contig_target is None else contig_target,
            self._auto_scaffold_contig(target_pdb, binderlen),
        )
        mask_seq = diffusion_mask[None]
        mask_str = diffusion_mask[None]
        potential_manager = PotentialManager(
            self.config.potentials,
            self.config.diffuser,
            self.config.inference,
            binderlen,
        )
        diffuser = self._make_diffuser(diffusion_steps=diffusion_steps)
        t_step_input = self._t_step_input(diffusion_steps)
        fa_stack, _true_coords = diffuser.diffuse_pose(
            noisy_coords,
            seq_t.clone(),
            atom_mask,
            diffusion_mask=diffusion_mask,
            t_list=np.arange(1, t_step_input + 1),
            include_motif_sidechains=self.config.preprocess.motif_sidechain_input,
            generator=generator,
        )
        noisy_coords = fa_stack[-1, :, :14]
        if contig_target is not None:
            self._attach_substrate_potentials(
                target_feats=contig_target,
                potential_manager=potential_manager,
                diffusion_mask=diffusion_mask,
                motif_coords=noisy_coords,
                diffuser=diffuser,
            )
        denoiser = self._make_denoiser(noisy_coords.shape[0], diffuser=diffuser, potential_manager=potential_manager)
        symmetry = self._make_symmetry()
        if symmetry is not None:
            noisy_coords, seq_t = symmetry.apply_symmetry(noisy_coords, seq_t)
        chain_idx, pdb_index = self._build_output_indices(contig_map)
        return RfDiffusionSampleState(
            noisy_coords=noisy_coords,
            seq_t=seq_t,
            contig_map=contig_map,
            mask_seq=mask_seq,
            mask_str=mask_str,
            diffusion_mask=diffusion_mask,
            binderlen=binderlen,
            chain_idx=chain_idx,
            pdb_index=pdb_index,
            t_step_input=t_step_input,
            diffuser=diffuser,
            denoiser=denoiser,
            potential_manager=potential_manager,
            symmetry=symmetry,
            cyclic_reses=self._init_cyclic_reses(mask_str, contig_map),
            ss=ss,
            adj=adj,
            target_ss=target_ss,
            target_adj=target_adj,
            target_pdb=target_pdb,
        )

    def _auto_scaffold_tensors(
        self, scaffold_length: int, target_pdb: dict[str, Any] | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any] | None, int]:
        r"""
        Build the initial coords, sequence, masks, and diffusion mask for auto scaffold-guided design.

        Returns:
            A tuple ``(noisy_coords, seq_t, atom_mask, diffusion_mask, contig_target, binderlen)`` combining
            the binder scaffold with the optional fixed target.
        """

        binderlen = int(scaffold_length)
        noisy_coords = torch.full((binderlen, 27, 3), float("nan"), dtype=torch.float32, device=self.device)
        noisy_coords = get_init_coords(noisy_coords[None, None]).squeeze(0).squeeze(0)
        seq_t = torch.full((binderlen,), MASK_TOKEN, dtype=torch.long, device=self.device)
        diffusion_mask = torch.zeros((binderlen,), dtype=torch.bool, device=self.device)
        atom_mask = torch.zeros((binderlen, 27), dtype=torch.bool, device=self.device)
        if target_pdb is None:
            return noisy_coords, one_hot_sequence(seq_t), atom_mask, diffusion_mask, None, binderlen
        target_coords = torch.full((len(target_pdb["xyz"]), 27, 3), float("nan"), device=self.device)
        target_coords[:, :14] = torch.as_tensor(target_pdb["xyz"], dtype=noisy_coords.dtype, device=self.device)
        target_mask = torch.zeros((target_coords.shape[0], 27), dtype=torch.bool, device=self.device)
        target_mask[:, :14] = torch.as_tensor(target_pdb["mask"], dtype=torch.bool, device=self.device)
        noisy_coords = torch.cat((noisy_coords, target_coords), dim=0)
        seq_t = torch.cat((seq_t, torch.as_tensor(target_pdb["seq"], dtype=torch.long, device=self.device)), dim=0)
        atom_mask = torch.cat((atom_mask, target_mask), dim=0)
        target_diffusion_mask = torch.ones(target_coords.shape[0], dtype=torch.bool, device=self.device)
        diffusion_mask = torch.cat((diffusion_mask, target_diffusion_mask))
        return noisy_coords, one_hot_sequence(seq_t), atom_mask, diffusion_mask, target_pdb, binderlen

    def _auto_scaffold_contig(self, target_pdb: dict[str, Any] | None, binderlen: int) -> list[str]:
        r"""Build the contig string for auto scaffold-guided design from the target chains and binder length."""

        if target_pdb is None:
            return [f"{binderlen}-{binderlen}"]
        fragments: list[str] = []
        pdb_idx = list(target_pdb["pdb_idx"])
        start_chain, start_residue = pdb_idx[0]
        previous_chain, previous_residue = start_chain, start_residue
        for chain, residue in pdb_idx[1:]:
            if chain != previous_chain or residue != previous_residue + 1:
                fragments.append(f"{start_chain}{start_residue}-{previous_residue}/0")
                start_chain, start_residue = chain, residue
            previous_chain, previous_residue = chain, residue
        fragments.append(f"{start_chain}{start_residue}-{previous_residue}/0")
        fragments.append(f"{binderlen}-{binderlen}")
        return [" ".join(fragments)]

    def _target_features(self, batch: Any | None) -> dict[str, Any]:
        r"""
        Resolve the target feature dict from the batch, a precomputed dict, or ``inference.input_pdb``.

        Raises:
            ValueError: If no batch tensors and no ``inference.input_pdb`` are available.
        """

        # Imported lazily: the processor pulls in the processor package, which would cycle at load time.
        from .processor import process_target

        if batch is not None and getattr(batch, "target_features", None) is not None:
            return _move_target_features(batch.target_features, self.device)
        if self.config.inference.input_pdb is not None:
            return process_target(self.config.inference.input_pdb, parse_hetatom=True, center=False)
        if batch is None:
            raise ValueError("RFdiffusion sampling needs batch target tensors or inference.input_pdb")
        residue_type = batch.residue_type[0].to(self.device)
        residue_index = batch.residue_index[0].to(self.device)
        chain_index = batch.chain_index[0].to(self.device)
        ca = batch.motif_positions[0].to(self.device)
        return target_features_from_tensors(
            residue_type=residue_type,
            residue_index=residue_index,
            chain_index=chain_index,
            ca_coords=ca,
        )

    def _construct_contig(self, target_feats: dict[str, Any], batch: Any | None) -> ContigMap:
        r"""
        Build the [`ContigMap`][ofoldx.models.rfdiffusion.model.ContigMap] from config contigs or batch masks.

        Raises:
            ValueError: If neither config contigs nor a batch are available.
        """

        contigs = self.config.contigmap.contigs
        if contigs is not None:
            return ContigMap(target_feats, **_config_kwargs(self.config.contigmap))
        if batch is None:
            raise ValueError("contigmap.contigs is required when sampling from inference.input_pdb")
        pdb_idx = list(target_feats["pdb_idx"])
        residue_index = list(range(len(pdb_idx)))
        mask_seq = (
            _numpy_bool(~batch.design_mask[0])
            if getattr(batch, "design_mask", None) is not None
            else np.ones(len(pdb_idx), dtype=bool)
        )
        mask_str = (
            _numpy_bool(batch.motif_mask[0])
            if getattr(batch, "motif_mask", None) is not None
            else np.ones(len(pdb_idx), dtype=bool)
        )
        return ContigMap(
            target_feats,
            reference_index=pdb_idx,
            design_index=pdb_idx,
            residue_index=residue_index,
            inpaint_seq_tensor=mask_seq,
            inpaint_str_tensor=mask_str,
        )

    def _make_diffuser(self, *, diffusion_steps: int | None) -> Diffuser:
        r"""Construct the forward [`Diffuser`][ofoldx.models.rfdiffusion.model.Diffuser] from config."""

        kwargs = _diffuser_kwargs(self.config.diffuser)
        if diffusion_steps is not None:
            kwargs["num_timesteps"] = int(diffusion_steps)
        schedule_dir = self.config.inference.schedule_directory_path or str(Path(".rfdiffusion_schedules"))
        kwargs.setdefault("cache_dir", schedule_dir)
        return Diffuser(**kwargs)

    def _make_denoiser(self, length: int, *, diffuser: Diffuser, potential_manager: PotentialManager) -> Denoiser:
        r"""Construct the reverse [`Denoiser`][ofoldx.models.rfdiffusion.model.Denoiser] from config."""

        kwargs = _diffuser_kwargs(self.config.diffuser)
        for diffuser_only_key in (
            "truncation_level",
            "igso3_num_omega",
            "igso3_num_sigma",
            "min_b",
            "max_b",
            "min_sigma",
            "max_sigma",
            "so3_schedule_type",
        ):
            kwargs.pop(diffuser_only_key, None)
        kwargs.update(_config_kwargs(self.config.denoiser))
        kwargs.update({"truncation_level": length, "diffuser": diffuser, "potential_manager": potential_manager})
        return Denoiser(**kwargs)

    def _make_symmetry(self) -> SymmetryGenerator | None:
        r"""Construct the [`SymmetryGenerator`][ofoldx.models.rfdiffusion.model.SymmetryGenerator], or `None`."""

        if self.config.inference.symmetry is None:
            return None
        return SymmetryGenerator(
            self.config.inference.symmetry,
            self.config.inference.recenter,
            self.config.inference.radius,
            self.config.inference.model_only_neighbors,
        )

    def _attach_substrate_potentials(
        self,
        *,
        target_feats: dict[str, Any],
        potential_manager: PotentialManager,
        diffusion_mask: Tensor,
        motif_coords: Tensor,
        diffuser: Diffuser,
    ) -> None:
        r"""
        Wire ligand HETATM coords and the motif frame into any configured ``substrate_contacts`` potential.

        Raises:
            ValueError: If the substrate potential is requested but the required HETATM records or substrate
                name are missing.
        """

        guiding = self.config.potentials.guiding_potentials
        if not guiding or not any("substrate_contacts" in value for value in guiding):
            return
        hetatm_coords = target_feats.get("xyz_het")
        info_het = target_feats.get("info_het", [])
        if hetatm_coords is None or len(hetatm_coords) == 0:
            raise ValueError("substrate_contacts requires ligand HETATM records in the input PDB")
        substrate_name = self.config.potentials.substrate
        if substrate_name is None:
            raise ValueError("potentials.substrate must name the ligand residue for substrate_contacts")
        het_names = np.asarray([str(info["name"]).strip() for info in info_het])
        substrate = torch.as_tensor(
            np.asarray(hetatm_coords)[het_names == substrate_name],
            dtype=motif_coords.dtype,
            device=self.device,
        )
        if substrate.numel() == 0:
            raise ValueError(f"expected at least one heteroatom for ligand {substrate_name!r}")
        motif_mask = diffusion_mask.to(device=motif_coords.device).bool()
        motif_only = motif_coords[motif_mask]
        for potential in potential_manager.potentials_to_apply:
            if isinstance(potential, SubstrateContacts):
                potential.motif_substrate_atoms = substrate
                potential.diffusion_mask = diffusion_mask.squeeze().to(self.device)
                potential.motif_coords = motif_only.to(self.device)

    def _uses_scaffold_guidance(self) -> bool:
        r"""Return whether the config enables scaffold guidance and supplies a scaffold or target input."""

        if not bool(_cfg_get(self.config.scaffoldguided, "scaffoldguided", False)):
            return False
        scaffold_inputs = (
            _cfg_get(self.config.scaffoldguided, "scaffold_dir", None),
            _cfg_get(self.config.scaffoldguided, "scaffold_list", None),
            _cfg_get(self.config.scaffoldguided, "target_path", None),
            _cfg_get(self.config.scaffoldguided, "target_ss", None),
            _cfg_get(self.config.scaffoldguided, "target_adj", None),
            _cfg_get(self.config.contigmap, "inpaint_str_helix", None),
            _cfg_get(self.config.contigmap, "inpaint_str_strand", None),
            _cfg_get(self.config.contigmap, "inpaint_str_loop", None),
        )
        return any(value is not None for value in scaffold_inputs) or bool(
            _cfg_get(self.config.scaffoldguided, "target_pdb", False)
        )

    def _t_step_input(self, diffusion_steps: int | None) -> int:
        r"""
        Return the starting timestep, honoring partial diffusion.

        Raises:
            ValueError: If ``partial_timesteps`` exceeds the total timestep count.
        """

        total = int(diffusion_steps if diffusion_steps is not None else self.config.diffuser.num_timesteps)
        if self.config.diffuser.partial_timesteps:
            if self.config.diffuser.partial_timesteps > total:
                raise ValueError("partial_timesteps must be <= T")
            return int(self.config.diffuser.partial_timesteps)
        return total

    def _build_output_indices(self, contig_map: ContigMap) -> tuple[list[str], list[int]]:
        r"""
        Assign output chain labels and PDB residue numbers per residue from the contig map.

        Returns:
            A ``(chain_idx, pdb_index)`` pair of per-residue chain labels and residue numbers.

        Raises:
            ValueError: If a chain block resolves to multiple chain ids, or no chain labels remain.
        """

        first_res = 0
        chain_idx: list[str] = []
        pdb_index: list[int] = []
        all_chains = {ref[0] for ref in contig_map.reference}
        available_chains = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") - all_chains)
        for last_res in contig_map.sampled_mask_length_bound:
            chain_ids = {ref[0] for ref in contig_map.reference[first_res:last_res]}
            if "_" in chain_ids:
                pdb_index += [idx + 1 for idx in range(last_res - first_res)]
                chain_ids = chain_ids - {"_"}
                if not chain_ids:
                    if not available_chains:
                        raise ValueError("no available chain ids remain")
                    chain_id = available_chains.pop(0)
                else:
                    if len(chain_ids) != 1:
                        raise ValueError(f"multiple chain ids in designed chain: {chain_ids}")
                    chain_id = list(chain_ids)[0]
                chain_idx += [chain_id] * (last_res - first_res)
            else:
                pdb_index += [int(ref[1]) for ref in contig_map.reference[first_res:last_res]]
                if len(chain_ids) != 1:
                    raise ValueError(f"multiple chain ids in fixed chain: {chain_ids}")
                chain_idx += [list(chain_ids)[0]] * (last_res - first_res)
            first_res = last_res
        return chain_idx, pdb_index

    def _init_cyclic_reses(self, mask_str: Tensor, contig_map: ContigMap) -> Tensor:
        r"""Build the per-residue cyclic-peptide mask from ``inference.cyclic`` and the configured chains."""

        mask = mask_str.squeeze(0).to(self.device).bool()
        if not self.config.inference.cyclic:
            return torch.zeros_like(mask)
        if self.config.inference.cyc_chains is None:
            return ~mask
        cyc_chains = [chain.upper() for chain in self.config.inference.cyc_chains]
        is_cyclized = torch.zeros_like(mask)
        chain_labels = [idx[0] for idx in contig_map.design]
        for chain in cyc_chains:
            chain_mask = torch.tensor([label == chain for label in chain_labels], dtype=torch.bool, device=self.device)
            is_cyclized[chain_mask] = True
        return is_cyclized

    @staticmethod
    def _infer_device(network: nn.Module | None) -> torch.device:
        r"""Infer the device from the network's parameters, falling back to CUDA-if-available then CPU."""

        if network is not None:
            with contextlib.suppress(StopIteration):
                return next(network.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def coerce_network_prediction(
    output: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    r"""
    Extract backbone ``pred_coords``, torsion angles, and pLDDT from the network's 7-tuple output.

    Args:
        output: The RoseTTAFold network output tuple.

    Returns:
        A tuple ``(pred_coords, alpha, plddt)`` where ``alpha`` and ``plddt`` are `None` if absent.

    Raises:
        TypeError: If the predicted coords entry is not a tensor.
    """

    pred_coords, alpha, plddt = output[2], output[4], output[6]
    if not torch.is_tensor(pred_coords):
        raise TypeError("RFdiffusion network output must include tensor pred_coords coords")
    return pred_coords, alpha if torch.is_tensor(alpha) else None, plddt if torch.is_tensor(plddt) else None


def normalize_backbone_prediction(pred_coords: Tensor, noisy_coords_ref: Tensor) -> Tensor:
    r"""
    Normalize network backbone ``pred_coords`` to ``(1, L, 3, 3)`` for all-atom reconstruction.

    Moves the prediction onto the reference device/dtype and squeezes stray leading singleton dimensions.

    Args:
        pred_coords: Predicted backbone coords in a `(B, L, 3, 3)`-compatible shape.
        noisy_coords_ref: Reference tensor supplying the target device and dtype.

    Returns:
        The backbone prediction reshaped to `(1, L, 3, 3)`.

    Raises:
        ValueError: If the prediction is not a single `(1, L, 3, 3)` backbone.
    """

    pred_coords = pred_coords.to(device=noisy_coords_ref.device, dtype=noisy_coords_ref.dtype)
    if pred_coords.ndim == 3:
        pred_coords = pred_coords.unsqueeze(0)
    while pred_coords.ndim > 4 and pred_coords.shape[0] == 1:
        pred_coords = pred_coords.squeeze(0)
    if pred_coords.ndim != 4 or pred_coords.shape[-2:] != (3, 3):
        raise ValueError(f"pred_coords backbone must have shape (B, L, 3, 3), got {tuple(pred_coords.shape)}")
    if pred_coords.shape[0] != 1:
        raise ValueError("RFdiffusion sampler expects one design trajectory per network call")
    return pred_coords


def _masked_secondary_structure(length: int, device: torch.device) -> Tensor:
    r"""Return an all-masked one-hot secondary-structure tensor `[L, 4]` (class 3 = masked/loop)."""

    tokens = torch.full((length,), 3, dtype=torch.long, device=device)
    return F.one_hot(tokens, num_classes=4).float()


def _masked_adjacency(length: int, device: torch.device) -> Tensor:
    r"""Return an all-masked one-hot block-adjacency tensor `[L, L, 3]` (class 2 = masked)."""

    tokens = torch.full((length, length), 2, dtype=torch.long, device=device)
    return F.one_hot(tokens, num_classes=3).float()


def _config_kwargs(config: Mapping[str, Any] | object) -> dict[str, Any]:
    r"""Return a plain dict of ``config``'s entries, handling chanfig configs, mappings, and plain objects."""

    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if isinstance(config, Mapping):
        return dict(config)
    return dict(vars(config))


def _diffuser_kwargs(config: Mapping[str, Any] | object) -> dict[str, Any]:
    r"""Return RFdiffusion diffuser config entries using the runtime constructor names."""

    kwargs = _config_kwargs(config)
    aliases = {
        "b_0": "beta_initial",
        "b_T": "beta_final",
        "so3_schedule_type": "so3_schedule_type",
    }
    for saved_name, runtime_name in aliases.items():
        if saved_name in kwargs:
            kwargs[runtime_name] = kwargs.pop(saved_name)
    return kwargs


def _cfg_get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    r"""Get ``key`` from ``config`` whether it is a mapping or a plain attribute-bearing object."""

    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _move_target_features(features: dict[str, Any], device: torch.device) -> dict[str, Any]:
    r"""Return a copy of ``features`` with every tensor value moved to ``device`` (non-tensors untouched)."""

    moved: dict[str, Any] = {}
    for key, value in features.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _numpy_bool(value: Tensor) -> np.ndarray:
    r"""Return a detached CPU boolean numpy array of ``value``."""

    return value.detach().cpu().bool().numpy()


def get_beta_schedule(
    num_timesteps: int,
    beta_initial: float,
    beta_final: float,
    schedule_type: str,
    schedule_params: dict[str, Any] | None = None,
    *,
    inference: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Build the RFdiffusion translation-noise beta, alpha, and cumulative alpha-bar schedules.

    Args:
        num_timesteps: Total number of diffusion timesteps (must be >= 15).
        beta_initial: Initial beta (rescaled by ``200 / num_timesteps``).
        beta_final: Final beta (rescaled by ``200 / num_timesteps``).
        schedule_type: Schedule shape; only `"linear"` is supported.
        schedule_params: Unused; accepted for interface compatibility.
        inference: Unused; accepted for interface compatibility.

    Returns:
        A tuple ``(beta, alpha, alpha_bar)`` of `[num_timesteps]` schedule tensors.

    Raises:
        ValueError: If ``schedule_type`` is not `"linear"` or ``num_timesteps`` is below 15.

    Examples:
        >>> from ofoldx.models.rfdiffusion.model import get_beta_schedule
        >>> beta, alpha, alpha_bar = get_beta_schedule(50, 1e-2, 7e-2, "linear")
        >>> beta.shape
        torch.Size([50])
    """

    del schedule_params, inference
    if schedule_type != "linear":
        raise ValueError(f"unsupported RFdiffusion beta schedule {schedule_type!r}")
    if num_timesteps < 15:
        raise ValueError("RFdiffusion beta schedule expects num_timesteps >= 15")
    beta0 = beta_initial * 200 / num_timesteps
    betaT = beta_final * 200 / num_timesteps
    beta = torch.linspace(beta0, betaT, num_timesteps)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


class EuclideanDiffuser:
    r"""
    Discrete Gaussian translation diffuser for backbone CA positions.

    Applies the forward variance-preserving noising kernel to CA translations across all timesteps, using the
    beta schedule from [`get_beta_schedule`][ofoldx.models.rfdiffusion.model.get_beta_schedule].

    Args:
        num_timesteps: Total number of diffusion timesteps.
        beta_initial: Initial beta of the translation schedule.
        beta_final: Final beta of the translation schedule.
        schedule_type: Schedule shape; only `"linear"` is supported.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_initial: float,
        beta_final: float,
        *,
        schedule_type: str = "linear",
    ) -> None:
        self.num_timesteps = int(num_timesteps)
        self.beta_schedule, _, _ = get_beta_schedule(
            self.num_timesteps,
            beta_initial,
            beta_final,
            schedule_type,
        )

    def diffuse_translations(
        self,
        coords: Tensor,
        *,
        diffusion_mask: Tensor | None = None,
        var_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Noise CA translations across all timesteps.

        Args:
            coords: Backbone coords `[L, A, 3]` to noise.
            diffusion_mask: Per-residue mask; masked (motif) residues stay fixed.
            var_scale: Multiplier on the per-step variance.
            generator: Optional RNG for reproducible noise.

        Returns:
            A tuple ``(noised, deltas)`` of the per-timestep noised coords and per-step deltas, each with the
            timestep axis first.
        """

        bb_stack: list[Tensor] = []
        delta_stack: list[Tensor] = []
        current = coords.clone()
        for t in range(1, self.num_timesteps + 1):
            current, delta = self.apply_kernel(
                current,
                t,
                diffusion_mask=diffusion_mask,
                var_scale=var_scale,
                generator=generator,
            )
            bb_stack.append(current)
            delta_stack.append(delta)
        return torch.stack(bb_stack).transpose(0, 1), torch.stack(delta_stack).transpose(0, 1)

    def apply_kernel(
        self,
        x: Tensor,
        t: int,
        *,
        diffusion_mask: Tensor | None = None,
        var_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Apply one forward noising step to CA translations at timestep ``t``.

        Args:
            x: Backbone coords `[L, A, 3]`.
            t: One-indexed timestep.
            diffusion_mask: Per-residue mask; masked residues stay fixed.
            var_scale: Multiplier on the step variance.
            generator: Optional RNG for reproducible noise.

        Returns:
            A tuple ``(noised, delta)`` of the noised coords and the CA delta applied.

        Raises:
            ValueError: If ``x`` is not `(L, A, 3)`.
        """

        if x.ndim != 3:
            raise ValueError("translation noising expects coords with shape (L, A, 3)")
        t_idx = t - 1
        ca_coords = x[:, 1]
        beta_t = self.beta_schedule.to(device=x.device, dtype=x.dtype)[t_idx]
        mean = torch.sqrt(1 - beta_t) * ca_coords
        std = torch.sqrt(torch.ones_like(ca_coords) * beta_t * var_scale)
        sampled = mean + torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator) * std
        delta = sampled - ca_coords
        if diffusion_mask is not None:
            delta = delta.clone()
            delta[diffusion_mask.to(device=delta.device).bool()] = 0
        return x + delta[:, None], delta


class IGSO3Diffuser:
    r"""
    Isotropic-Gaussian-on-SO(3) (IGSO3) rotation diffuser and reverse sampler.

    Diffuses residue-frame rotations forward with IGSO3 noise and samples the reverse process. Precomputes (and
    caches to disk) the IGSO3 CDFs and score norms for inverse-CDF sampling.

    Args:
        num_timesteps: Total number of diffusion timesteps.
        min_sigma: Minimum IGSO3 noise sigma.
        max_sigma: Maximum IGSO3 noise sigma.
        min_b: Minimum drift coefficient (linear schedule).
        max_b: Maximum drift coefficient (linear schedule).
        cache_dir: Directory for the precomputed IGSO3 lookup tables.
        num_omega: Number of discretized rotation angles.
        num_sigma: Number of discretized sigma levels.
        schedule: Sigma-schedule shape (`"linear"` or `"exponential"`).
        truncation_level: Number of power-series terms used in the density evaluation.

    Raises:
        ValueError: If ``schedule`` is not `"linear"` or `"exponential"`.
    """

    def __init__(
        self,
        *,
        num_timesteps: int,
        min_sigma: float,
        max_sigma: float,
        min_b: float,
        max_b: float,
        cache_dir: str | Path,
        num_omega: int = 1000,
        num_sigma: int = 500,
        schedule: str = "linear",
        truncation_level: int = 2000,
    ) -> None:
        self.num_timesteps = int(num_timesteps)
        self.schedule = schedule
        self.cache_dir = Path(cache_dir)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.min_b = float(min_b)
        self.max_b = float(max_b)
        if self.schedule == "linear":
            self.max_sigma = float(self.sigma(torch.tensor(1.0)))
        if self.schedule not in {"linear", "exponential"}:
            raise ValueError(f"unsupported IGSO3 schedule {schedule!r}")
        self.num_omega = int(num_omega)
        self.num_sigma = int(num_sigma)
        self.truncation_level = int(truncation_level)
        self.igso3_vals = self._calc_igso3_vals()
        self.step_size = 1 / self.num_timesteps

    @property
    def discrete_sigma(self) -> np.ndarray:
        r"""The discretized sigma levels of the precomputed IGSO3 tables."""

        return self.igso3_vals["discrete_sigma"]

    def sigma_idx(self, sigma: np.ndarray | Tensor | float) -> np.ndarray:
        r"""
        Return the lookup-table index for each ``sigma`` value.

        Args:
            sigma: Noise level(s) to locate in the discretized sigma grid; scalar or array-like.

        Returns:
            The zero-based table index (via `numpy.digitize`) for each ``sigma``.
        """

        return np.digitize(np.asarray(sigma), self.discrete_sigma) - 1

    def t_to_idx(self, t: int | np.ndarray | Tensor) -> np.ndarray:
        r"""
        Map a timestep (or array of timesteps) to the IGSO3 sigma-table index via the sigma schedule.

        Args:
            t: Timestep(s), normalized internally by ``num_timesteps`` before evaluating the sigma schedule.

        Returns:
            The IGSO3 table index for each timestep.
        """

        return self.sigma_idx(
            self.sigma(torch.as_tensor(t, dtype=torch.float32) / self.num_timesteps).detach().cpu().numpy()
        )

    def sigma(self, t: Tensor | float) -> Tensor:
        r"""
        Return the IGSO3 noise sigma at normalized time ``t`` in ``[0, 1]``.

        Args:
            t: Normalized time in ``[0, 1]``.

        Returns:
            The noise sigma at ``t``.

        Raises:
            ValueError: If ``t`` falls outside ``[0, 1]``.
        """

        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        if torch.any(t_tensor < 0) or torch.any(t_tensor > 1):
            raise ValueError(f"invalid IGSO3 time {t}")
        if self.schedule == "exponential":
            sigma = t_tensor * np.log10(self.max_sigma) + (1 - t_tensor) * np.log10(self.min_sigma)
            return 10**sigma
        return self.min_sigma + t_tensor * self.min_b + 0.5 * (t_tensor**2) * (self.max_b - self.min_b)

    def diffusion_coefficient(self, t: float | Tensor) -> Tensor:
        r"""
        Return the SDE diffusion coefficient (sqrt of d/dt of sigma-squared) at normalized time ``t``.

        Args:
            t: Normalized time in ``[0, 1]`` at which to evaluate the diffusion coefficient.

        Returns:
            The scalar diffusion coefficient `sqrt(d(sigma^2)/dt)` at ``t``.
        """

        t_tensor = torch.as_tensor(t, dtype=torch.float32).requires_grad_(True)
        sigma_sq = self.sigma(t_tensor) ** 2
        grad = torch.autograd.grad(sigma_sq.sum(), t_tensor)[0]
        return torch.sqrt(grad)

    def sample(self, ts: np.ndarray, *, n_samples: int = 1) -> np.ndarray:
        r"""
        Inverse-CDF sample IGSO3 rotation angles at the given timesteps.

        Args:
            ts: One-indexed timesteps to sample at.
            n_samples: Number of angle samples per timestep.

        Returns:
            Sampled rotation angles `[len(ts), n_samples]`.

        Raises:
            ValueError: If any timestep is zero (timesteps must be one-indexed).
        """

        if np.any(ts == 0):
            raise ValueError("IGSO3 sample expects one-indexed timesteps")
        all_samples = []
        for t in ts:
            sigma_idx = int(np.asarray(self.t_to_idx(int(t))).item())
            cdf = self.igso3_vals["cdf"][sigma_idx]
            all_samples.append(np.interp(np.random.rand(n_samples), cdf, self.igso3_vals["discrete_omega"]))
        return np.stack(all_samples, axis=0)

    def sample_vec(self, ts: np.ndarray, *, n_samples: int = 1) -> np.ndarray:
        r"""
        Sample IGSO3 rotation vectors (random axis times sampled angle) at the given timesteps.

        Args:
            ts: One-indexed timesteps to sample at.
            n_samples: Number of rotation vectors per timestep.

        Returns:
            Rotation vectors `[len(ts), n_samples, 3]` (unit axis scaled by the sampled angle).
        """

        axes = np.random.randn(len(ts), n_samples, 3)
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        return axes * self.sample(ts, n_samples=n_samples)[..., None]

    def score_norm(self, t: int, omega: np.ndarray) -> np.ndarray:
        r"""
        Return the IGSO3 score-function magnitude at rotation angles ``omega`` for timestep ``t``.

        Args:
            t: One-indexed timestep whose sigma level selects the score-norm table.
            omega: Rotation angles (in radians) at which to interpolate the score magnitude.

        Returns:
            The interpolated score-function magnitude at each angle in ``omega``.
        """

        sigma_idx = int(np.asarray(self.t_to_idx(t)).item())
        return np.interp(omega, self.igso3_vals["discrete_omega"], self.igso3_vals["score_norm"][sigma_idx])

    def diffuse_frames(
        self,
        coords: Tensor | np.ndarray,
        *,
        t_list: np.ndarray | None = None,
        diffusion_mask: np.ndarray | Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Noise residue frames with IGSO3 rotations across timesteps.

        Args:
            coords: Backbone coords `[L, A, 3]` (tensor or array).
            t_list: Optional one-indexed timesteps to return; all timesteps otherwise.
            diffusion_mask: Per-residue mask; masked residues keep their frame.

        Returns:
            A tuple ``(perturbed_coords, perturbed_rotations)`` with the residue axis first.
        """

        if torch.is_tensor(coords):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = coords
        t_values = np.arange(self.num_timesteps) + 1
        num_res = len(coords_np)
        N = torch.from_numpy(coords_np[None, :, 0])
        Ca = torch.from_numpy(coords_np[None, :, 1])
        C = torch.from_numpy(coords_np[None, :, 2])
        R_true, Ca_true = rigid_from_3_points(N, Ca, C)
        R_true_np = R_true[0].detach().cpu().numpy()
        Ca_np = Ca_true[0].detach().cpu().numpy()
        sampled_rots = self.sample_vec(t_values, n_samples=num_res)
        if diffusion_mask is not None:
            mask_np = (
                diffusion_mask.detach().cpu().numpy() if torch.is_tensor(diffusion_mask) else np.asarray(diffusion_mask)
            )
            sampled_rots = sampled_rots * (1 - mask_np[None, :, None])
        R_sampled = (
            Rotation.from_rotvec(sampled_rots.reshape(-1, 3)).as_matrix().reshape(self.num_timesteps, num_res, 3, 3)
        )
        R_perturbed = np.einsum("tnij,njk->tnik", R_sampled, R_true_np)
        perturbed = np.einsum("tnij,naj->tnai", R_sampled, coords_np[:, :3] - Ca_np[:, None]) + Ca_np[None, :, None]
        if t_list is not None:
            indices = [int(t) - 1 for t in t_list]
            perturbed = perturbed[indices]
            R_perturbed = R_perturbed[indices]
        return perturbed.transpose(1, 0, 2, 3), R_perturbed.transpose(1, 0, 2, 3)

    def reverse_sample_vectorized(
        self,
        rotations_t: Tensor | np.ndarray,
        rotations_0: Tensor | np.ndarray,
        t: int,
        noise_level: float,
        *,
        mask: Tensor | None = None,
        return_perturb: bool = False,
    ) -> Tensor:
        r"""
        Take one reverse IGSO3 step toward the predicted clean rotations.

        Args:
            rotations_t: Current noisy rotations `[..., 3, 3]`.
            rotations_0: Predicted clean rotations `[..., 3, 3]`.
            t: Current timestep.
            noise_level: Scale on the injected reverse noise.
            mask: Per-element mask; masked frames are not perturbed.
            return_perturb: If `True`, return the perturbation rotation instead of the updated rotation.

        Returns:
            The updated rotations, or the perturbation rotation when ``return_perturb`` is set.
        """

        R_t_t = torch.as_tensor(rotations_t, dtype=torch.float32)
        R_0_t = torch.as_tensor(rotations_0, dtype=torch.float32, device=R_t_t.device)
        R_0t = torch.einsum("...ij,...kj->...ik", R_t_t, R_0_t)
        rotvec = torch.as_tensor(
            Rotation.from_matrix(R_0t.detach().cpu().numpy()).as_rotvec(), dtype=R_t_t.dtype, device=R_t_t.device
        )
        omega_value = torch.linalg.norm(rotvec, dim=-1).detach().cpu().numpy()
        denom = np.maximum(omega_value, 1e-8)
        score = (
            rotvec
            * torch.as_tensor(self.score_norm(t, omega_value) / denom, dtype=rotvec.dtype, device=rotvec.device)[
                :, None
            ]
        )
        rot_g = self.diffusion_coefficient(t / self.num_timesteps).to(device=score.device, dtype=score.dtype)
        z = torch.randn_like(score) * noise_level
        delta_r = (rot_g**2) * self.step_size * score
        perturb_tangent = delta_r + rot_g * np.sqrt(self.step_size) * z
        if mask is not None:
            perturb_tangent = perturb_tangent * (1 - mask.long()).to(device=perturb_tangent.device)[:, None]
        perturb = exp_map(perturb_tangent)
        if return_perturb:
            return perturb
        return torch.einsum("...ij,...jk->...ik", perturb, R_t_t)

    def _calc_igso3_vals(self) -> dict[str, np.ndarray]:
        r"""Load the cached IGSO3 lookup tables, computing and caching them on a miss."""

        cache_file = self._cache_file()
        if cache_file.exists():
            from .processor import read_pickle

            return read_pickle(cache_file)  # type: ignore[return-value]
        values = calculate_igso3(
            num_sigma=self.num_sigma,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_omega=self.num_omega,
            num_terms=self.truncation_level,
        )
        write_pickle(cache_file, values)
        return values

    def _cache_file(self) -> Path:
        r"""Return the cache-file path encoding the IGSO3 schedule parameters."""

        def fmt(value: object) -> str:
            r"""
            Format a schedule value for use in the cache-file name (dots replaced with underscores).

            Args:
                value: Schedule parameter to stringify (e.g. a sigma bound or beta coefficient).

            Returns:
                The value as a filename-safe string with ``.`` replaced by ``_``.
            """

            return str(value).replace(".", "_")

        parts = [
            f"T_{self.num_timesteps}",
            f"omega_{self.num_omega}",
            f"sigma_{self.num_sigma}",
            f"min_sigma_{fmt(self.min_sigma)}",
            f"schedule_{self.schedule}",
            f"L_{self.truncation_level}",
        ]
        if self.schedule == "linear":
            parts.extend([f"min_b_{fmt(self.min_b)}", f"max_b_{fmt(self.max_b)}"])
        else:
            parts.append(f"max_sigma_{fmt(self.max_sigma)}")
        return self.cache_dir / ("_".join(parts) + ".pkl")


class Diffuser:
    r"""
    RFdiffusion forward pose diffuser coupling translation and rotation noise.

    Combines a [`EuclideanDiffuser`][ofoldx.models.rfdiffusion.model.EuclideanDiffuser] for CA
    translations and an [`IGSO3Diffuser`][ofoldx.models.rfdiffusion.model.IGSO3Diffuser] for backbone
    frames to noise a full pose forward.

    Args:
        num_timesteps: Total number of diffusion timesteps.
        beta_initial: Initial translation-noise beta.
        beta_final: Final translation-noise beta.
        min_sigma: Minimum IGSO3 sigma.
        max_sigma: Maximum IGSO3 sigma.
        min_b: Minimum IGSO3 drift coefficient.
        max_b: Maximum IGSO3 drift coefficient.
        schedule_type: Translation beta-schedule shape.
        so3_schedule_type: IGSO3 sigma-schedule shape.
        so3_type: Rotation diffuser family; only `"igso3"` is supported.
        crd_scale: Coordinate-scaling factor applied before noising.
        var_scale: Multiplier on the translation-noise variance.
        cache_dir: Directory for the IGSO3 lookup tables.
        partial_timesteps: Number of steps for partial diffusion, or `None` for full diffusion.
        truncation_level: Number of IGSO3 series terms.
        igso3_num_omega: Number of IGSO3 discretized angles.
        igso3_num_sigma: Number of IGSO3 discretized sigma levels.

    Raises:
        ValueError: If ``so3_type`` is not `"igso3"` or ``partial_timesteps`` is out of range.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_initial: float,
        beta_final: float,
        min_sigma: float,
        max_sigma: float,
        min_b: float,
        max_b: float,
        schedule_type: str,
        so3_schedule_type: str,
        so3_type: str,
        crd_scale: float,
        *,
        var_scale: float = 1.0,
        cache_dir: str | Path = ".",
        partial_timesteps: int | None = None,
        truncation_level: int = 2000,
        igso3_num_omega: int = 1000,
        igso3_num_sigma: int = 500,
    ) -> None:
        if so3_type != "igso3":
            raise ValueError(f"unsupported SO(3) diffuser {so3_type!r}")
        self.num_timesteps = int(num_timesteps)
        self.partial_timesteps = None if partial_timesteps is None else int(partial_timesteps)
        if self.partial_timesteps is not None and not 0 < self.partial_timesteps <= self.num_timesteps:
            raise ValueError("partial_timesteps must be in the interval [1, T]")
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.crd_scale = float(crd_scale)
        self.var_scale = float(var_scale)
        self.cache_dir = Path(cache_dir)
        self.so3_diffuser = IGSO3Diffuser(
            num_timesteps=self.num_timesteps,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            schedule=so3_schedule_type,
            min_b=min_b,
            max_b=max_b,
            cache_dir=self.cache_dir,
            truncation_level=truncation_level,
            num_omega=igso3_num_omega,
            num_sigma=igso3_num_sigma,
        )
        self.eucl_diffuser = EuclideanDiffuser(
            self.num_timesteps,
            beta_initial,
            beta_final,
            schedule_type=schedule_type,
        )

    def diffuse_pose(
        self,
        coords: Tensor,
        seq: Tensor,
        atom_mask: Tensor,
        *,
        include_motif_sidechains: bool = True,
        diffusion_mask: Tensor | None = None,
        t_list: np.ndarray | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Noise a full pose forward across the diffusion schedule.

        Centers on the motif (or global) centroid, noises translations and frames, recombines them, and
        optionally copies motif sidechains through.

        Args:
            coords: Backbone coords `[L, 27, 3]` to noise.
            seq: Sequence features; unused, accepted for signature symmetry.
            atom_mask: Atom mask; unused, accepted for signature symmetry.
            include_motif_sidechains: Whether to preserve fixed-motif sidechain coords.
            diffusion_mask: Per-residue mask; masked (motif) residues stay fixed.
            t_list: Optional one-indexed timesteps to return; all timesteps otherwise.
            generator: Optional RNG for reproducible noise.

        Returns:
            A tuple ``(diffused_fullatom, true_coords)`` of the noised atom27 trajectory and the centered
            clean coords.

        Raises:
            ValueError: If the backbone coords contain NaNs before diffusion.
        """

        del seq, atom_mask
        if diffusion_mask is None:
            diffusion_mask = torch.zeros(len(coords.squeeze()), dtype=torch.bool, device=coords.device)
        diffusion_mask = diffusion_mask.to(device=coords.device).bool()
        if torch.isnan(coords[:, :3]).any():
            raise ValueError("backbone coords must be initialized before diffusion")
        if diffusion_mask.any():
            motif_com = coords[diffusion_mask, 1].mean(dim=0)
            coords = coords - motif_com
        else:
            coords = coords - coords[:, 1].mean(dim=0)
        true_coords = coords.clone()
        scaled = coords * self.crd_scale
        _diffused_T, deltas = self.eucl_diffuser.diffuse_translations(
            scaled[:, :3].clone(),
            diffusion_mask=diffusion_mask,
            var_scale=self.var_scale,
            generator=generator,
        )
        deltas = deltas / self.crd_scale
        frame_coords, _frames = self.so3_diffuser.diffuse_frames(
            scaled[:, :3].clone(),
            diffusion_mask=diffusion_mask.detach().cpu().numpy(),
            t_list=None,
        )
        frame_coords = torch.from_numpy(frame_coords).to(device=coords.device, dtype=coords.dtype) / self.crd_scale
        cum_delta = deltas.cumsum(dim=1)
        diffused_bb = (frame_coords + cum_delta[:, :, None]).transpose(0, 1)
        t_steps, length = diffused_bb.shape[:2]
        diffused_fa = torch.zeros((t_steps, length, 27, 3), dtype=coords.dtype, device=coords.device)
        diffused_fa[:, :, :3] = diffused_bb
        if include_motif_sidechains:
            diffused_fa[:, diffusion_mask, :14] = true_coords[None, diffusion_mask, :14]
        if t_list is not None:
            indices = [int(t) - 1 for t in t_list]
            diffused_fa = diffused_fa[indices]
        return diffused_fa, true_coords


def get_next_frames(
    noisy_coords: Tensor,
    pred_coords: Tensor,
    t: int,
    diffuser: Diffuser,
    so3_type: str,
    diffusion_mask: Tensor,
    *,
    noise_scale: float = 1.0,
) -> Tensor:
    r"""
    Update backbone frames from ``noisy_coords`` one reverse step toward predicted ``x_0``.

    Args:
        noisy_coords: Current noised backbone coords `[L, A, 3]`.
        pred_coords: Predicted clean backbone coords `[L, A, 3]`.
        t: Current timestep.
        diffuser: The forward [`Diffuser`][ofoldx.models.rfdiffusion.model.Diffuser] whose IGSO3 reverse
            sampler is used.
        so3_type: Rotation diffuser family; only `"igso3"` is supported.
        diffusion_mask: Per-residue mask; masked (motif) frames stay fixed.
        noise_scale: Scale on the reverse rotation noise.

    Returns:
        The updated backbone coords `[L, 3, 3]`.

    Raises:
        ValueError: If ``so3_type`` is not `"igso3"`.
    """

    rotations_0, _Ca_0 = rigid_from_3_points(pred_coords[None, :, 0], pred_coords[None, :, 1], pred_coords[None, :, 2])
    rotations_t, Ca_t = rigid_from_3_points(
        noisy_coords[None, :, 0], noisy_coords[None, :, 1], noisy_coords[None, :, 2]
    )
    R_0_np = Rotation.from_matrix(rotations_0.squeeze(0).detach().cpu().numpy()).as_matrix()
    R_t_np = Rotation.from_matrix(rotations_t.squeeze(0).detach().cpu().numpy()).as_matrix()
    length = R_t_np.shape[0]
    transitions = np.broadcast_to(np.eye(3), (length, 3, 3)).copy()
    mask_cpu = diffusion_mask.detach().cpu().bool()
    mobile = ~mask_cpu.numpy()
    if so3_type == "igso3":
        if mobile.any():
            transitions[mobile] = (
                diffuser.so3_diffuser.reverse_sample_vectorized(
                    R_t_np[mobile],
                    R_0_np[mobile],
                    t,
                    noise_level=noise_scale,
                    return_perturb=True,
                )
                .detach()
                .cpu()
                .numpy()
            )
    else:
        raise ValueError(f"unsupported SO(3) diffusion type {so3_type!r}")
    transitions = transitions[:, None]
    ca_t_np = Ca_t.squeeze(0).detach().cpu().numpy()
    next_coords = (
        np.einsum("lrij,laj->lrai", transitions, noisy_coords[:, :3].detach().cpu().numpy() - ca_t_np[:, None])
        + ca_t_np[:, None, None]
    ).squeeze(1)
    return torch.as_tensor(next_coords, dtype=noisy_coords.dtype, device=noisy_coords.device)


def get_posterior_mean(
    noisy_coords: Tensor,
    pred_coords: Tensor,
    t: int,
    beta_schedule: Tensor,
    alphabar_schedule: Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute the mean and variance of ``q(x_{t-1} | noisy_coords, x_0)`` for CA translations.

    Args:
        noisy_coords: Current noised coords `[L, A, 3]`.
        pred_coords: Predicted clean coords `[L, A, 3]`.
        t: Current timestep.
        beta_schedule: The beta schedule `[T]`.
        alphabar_schedule: The cumulative alpha-bar schedule `[T]`.
        eps: Numerical floor inside the square roots.

    Returns:
        A tuple ``(mean, variance)`` of the CA posterior mean `[L, 3]` and its scalar variance.
    """

    t_idx = t - 1
    beta_schedule = beta_schedule.to(device=noisy_coords.device, dtype=noisy_coords.dtype)
    alphabar_schedule = alphabar_schedule.to(device=noisy_coords.device, dtype=noisy_coords.dtype)
    sigma = ((1 - alphabar_schedule[t_idx - 1]) / (1 - alphabar_schedule[t_idx])) * beta_schedule[t_idx]
    noisy_coords_ca = noisy_coords[:, 1]
    pred_coords_ca = pred_coords[:, 1]
    a = (
        (torch.sqrt(alphabar_schedule[t_idx - 1] + eps) * beta_schedule[t_idx])
        / (1 - alphabar_schedule[t_idx])
        * pred_coords_ca
    )
    b = (
        torch.sqrt(1 - beta_schedule[t_idx] + eps)
        * (1 - alphabar_schedule[t_idx - 1])
        / (1 - alphabar_schedule[t_idx])
        * noisy_coords_ca
    )
    return a + b, sigma


def get_next_ca(
    noisy_coords: Tensor,
    pred_coords: Tensor,
    t: int,
    diffusion_mask: Tensor,
    crd_scale: float,
    beta_schedule: Tensor,
    alphabar_schedule: Tensor,
    *,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Sample the reverse CA-translation update for one timestep.

    Args:
        noisy_coords: Current noised coords `[L, A, 3]`.
        pred_coords: Predicted clean coords `[L, A, 3]`.
        t: Current timestep.
        diffusion_mask: Per-residue mask; masked (motif) CA positions stay fixed.
        crd_scale: Coordinate-scaling factor.
        beta_schedule: The beta schedule `[T]`.
        alphabar_schedule: The cumulative alpha-bar schedule `[T]`.
        noise_scale: Scale on the injected posterior noise.
        generator: Optional RNG for reproducible sampling.

    Returns:
        A tuple ``(next_coords, delta)`` of the updated coords and the CA delta, both unscaled.
    """

    pred_coords_scaled = pred_coords * crd_scale
    noisy_coords_scaled = noisy_coords * crd_scale
    mu, sigma = get_posterior_mean(
        noisy_coords_scaled, pred_coords_scaled, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule
    )
    sampled = mu + torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator) * torch.sqrt(
        sigma * noise_scale
    )
    delta = sampled - noisy_coords_scaled[:, 1]
    if diffusion_mask is not None:
        delta = delta.clone()
        delta[diffusion_mask.to(device=delta.device).bool()] = 0
    out = noisy_coords_scaled + delta[:, None]
    return out / crd_scale, delta / crd_scale


def get_noise_schedule(num_timesteps: int, final_noise: float, initial_noise: float, schedule_type: str):
    r"""
    Return a callable mapping a timestep to its noise scale.

    Args:
        num_timesteps: Total number of timesteps (for the linear interpolation).
        final_noise: Noise scale at the final timestep.
        initial_noise: Noise scale at the first timestep.
        schedule_type: Schedule shape (`"constant"` or `"linear"`).

    Returns:
        A function ``schedule(t) -> float`` giving the noise scale at timestep ``t``.

    Raises:
        ValueError: If ``schedule_type`` is not `"constant"` or `"linear"`.

    Examples:
        >>> from ofoldx.models.rfdiffusion.model import get_noise_schedule
        >>> schedule = get_noise_schedule(50, 1.0, 1.0, "constant")
        >>> schedule(10)
        1.0
    """

    schedules = {
        "constant": lambda t: final_noise,
        "linear": lambda t: ((t - 1) / (num_timesteps - 1)) * (final_noise - initial_noise) + initial_noise,
    }
    if schedule_type not in schedules:
        raise ValueError(f"noise schedule must be one of {tuple(schedules)}")
    return schedules[schedule_type]


class NullPotentialManager:
    r"""
    No-op stand-in for the potential manager when no guiding potentials are configured.

    Exposes the same ``is_empty`` / ``compute_all_potentials`` / ``get_guide_scale`` surface as the real
    [`PotentialManager`][ofoldx.models.rfdiffusion.model.PotentialManager] so the sampler can call it
    unconditionally: every method returns the neutral value (empty, zero energy, zero guide scale), leaving the
    reverse-diffusion step unguided.

    Attributes:
        potentials_to_apply: Always the empty list; kept for interface parity with the real manager.
    """

    potentials_to_apply: list[object] = []

    def is_empty(self) -> bool:
        r"""
        Report that no potentials are registered.

        Returns:
            Always `True`; a null manager holds no potentials.
        """

        return True

    def compute_all_potentials(self, coords: Tensor) -> Tensor:
        r"""
        Return the (zero) total potential energy for ``coords``.

        Args:
            coords: Structure coordinates whose device and dtype the returned scalar matches.

        Returns:
            A zero scalar tensor on ``coords``' device/dtype (no potential to compute).
        """

        return torch.zeros((), dtype=coords.dtype, device=coords.device)

    def get_guide_scale(self, t: int) -> float:
        r"""
        Return the guidance scale at timestep ``t``.

        Args:
            t: Diffusion timestep (ignored; a null manager applies no guidance).

        Returns:
            Always `0.0`.
        """

        del t
        return 0.0


class Denoiser:
    r"""
    Reverse RFdiffusion pose updater stepping ``x_t`` toward ``x_{t-1}``.

    Combines the CA-translation posterior sample and the IGSO3 frame reverse step, optionally aligns the
    prediction onto the fixed motif, and adds guiding-potential gradients.

    Args:
        num_timesteps: Total number of diffusion timesteps.
        truncation_level: IGSO3 series truncation (also the design length here).
        diffuser: The forward [`Diffuser`][ofoldx.models.rfdiffusion.model.Diffuser].
        beta_initial: Initial translation beta.
        beta_final: Final translation beta.
        schedule_type: Translation beta-schedule shape.
        so3_type: Rotation diffuser family; only `"igso3"` is supported.
        noise_scale_ca: Initial CA-translation noise scale.
        final_noise_scale_ca: Final CA-translation noise scale.
        ca_noise_schedule_type: CA noise-schedule shape.
        noise_scale_frame: Initial frame (rotation) noise scale.
        final_noise_scale_frame: Final frame noise scale.
        frame_noise_schedule_type: Frame noise-schedule shape.
        crd_scale: Coordinate-scaling factor.
        potential_manager: Guiding-potential manager; a null manager is used when `None`.
        partial_timesteps: Number of steps for partial diffusion, or `None`.

    Raises:
        ValueError: If ``partial_timesteps`` is out of range.
    """

    def __init__(
        self,
        num_timesteps: int,
        truncation_level: int,
        diffuser: Diffuser,
        *,
        beta_initial: float = 0.001,
        beta_final: float = 0.1,
        schedule_type: str = "linear",
        so3_type: str = "igso3",
        noise_scale_ca: float = 1.0,
        final_noise_scale_ca: float = 1.0,
        ca_noise_schedule_type: str = "constant",
        noise_scale_frame: float = 0.5,
        final_noise_scale_frame: float = 0.5,
        frame_noise_schedule_type: str = "constant",
        crd_scale: float = 1 / 15,
        potential_manager: Any | None = None,
        partial_timesteps: int | None = None,
    ) -> None:
        self.num_timesteps = int(num_timesteps)
        self.truncation_level = int(truncation_level)
        self.partial_timesteps = None if partial_timesteps is None else int(partial_timesteps)
        if self.partial_timesteps is not None and not 0 < self.partial_timesteps <= self.num_timesteps:
            raise ValueError("partial_timesteps must be in the interval [1, T]")
        self.diffuser = diffuser
        self.beta_initial = float(beta_initial)
        self.beta_final = float(beta_final)
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.crd_scale = float(crd_scale)
        self.noise_scale_ca = float(noise_scale_ca)
        self.final_noise_scale_ca = float(final_noise_scale_ca)
        self.ca_noise_schedule_type = ca_noise_schedule_type
        self.noise_scale_frame = float(noise_scale_frame)
        self.final_noise_scale_frame = float(final_noise_scale_frame)
        self.frame_noise_schedule_type = frame_noise_schedule_type
        self.potential_manager: Any = NullPotentialManager() if potential_manager is None else potential_manager
        self.schedule, _, self.alphabar_schedule = get_beta_schedule(
            self.num_timesteps,
            self.beta_initial,
            self.beta_final,
            self.schedule_type,
            inference=True,
        )
        self.noise_schedule_ca = get_noise_schedule(
            self.num_timesteps,
            self.noise_scale_ca,
            self.final_noise_scale_ca,
            self.ca_noise_schedule_type,
        )
        self.noise_schedule_frame = get_noise_schedule(
            self.num_timesteps,
            self.noise_scale_frame,
            self.final_noise_scale_frame,
            self.frame_noise_schedule_type,
        )

    def align_to_xt_motif(
        self, pred_coords: Tensor, final_coords: Tensor, diffusion_mask: Tensor, *, eps: float = 1e-6
    ) -> Tensor:
        r"""
        Kabsch-align predicted coords onto the fixed motif residues in ``final_coords``.

        Args:
            pred_coords: Predicted coords `[L, A, 3]` to align.
            final_coords: Reference (noised) coords `[L, A, 3]` holding the fixed motif.
            diffusion_mask: Per-residue mask selecting the motif residues to align on.
            eps: Numerical floor (accepted for signature symmetry).

        Returns:
            The aligned predicted coords `[L, A, 3]`.

        Raises:
            ValueError: If the atom dimensions of the two coord tensors differ.
        """

        if final_coords.shape[1] != pred_coords.shape[1]:
            raise ValueError(
                f"final_coords and pred_coords atom dimensions differ: {final_coords.shape} vs {pred_coords.shape}"
            )
        atom_mask = ~torch.isnan(pred_coords)
        pred_coords_np = pred_coords.detach().cpu().numpy()
        noisy_coords_np = final_coords.detach().cpu().numpy()
        motif_mask = diffusion_mask.detach().cpu().bool().numpy()
        pred_coords_motif = pred_coords_np[motif_mask, :3].reshape(-1, 3)
        noisy_coords_motif = noisy_coords_np[motif_mask, :3].reshape(-1, 3)
        if pred_coords_motif.size == 0:
            return pred_coords
        pred_coords_mean = pred_coords_motif.mean(0)
        noisy_coords_mean = noisy_coords_motif.mean(0)
        A = noisy_coords_motif - noisy_coords_mean
        B = pred_coords_motif - pred_coords_mean
        cov = A.T @ B
        U, _S, Vt = np.linalg.svd(cov)
        handed = np.eye(3)
        handed[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
        rotation = Vt.T @ handed @ U.T
        pred_coords_np[~atom_mask.detach().cpu().numpy()] = 0
        aligned = (pred_coords_np.reshape(-1, 3) - pred_coords_mean) @ rotation + noisy_coords_mean
        aligned = aligned.reshape(pred_coords_np.shape)
        aligned[~atom_mask.detach().cpu().numpy()] = np.nan
        return torch.as_tensor(aligned, dtype=pred_coords.dtype, device=pred_coords.device)

    def get_potential_gradients(self, coords: Tensor, *, diffusion_mask: Tensor | None) -> Tensor:
        r"""
        Compute per-residue CA gradients of the summed guiding potentials.

        Args:
            coords: Backbone coords `[L, A, 3]` to differentiate against.
            diffusion_mask: Per-residue mask; masked (motif) CA gradients are zeroed.

        Returns:
            The CA gradients `[L, 3]` (zeros when no potentials are configured).

        Raises:
            RuntimeError: If the potential backward produces no coordinate gradients.
        """

        if getattr(self.potential_manager, "is_empty", lambda: True)():
            return torch.zeros(coords.shape[0], 3, dtype=coords.dtype, device=coords.device)
        coords = coords.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            potential = self.potential_manager.compute_all_potentials(coords)
            potential.backward()
        if coords.grad is None:
            raise RuntimeError("RFdiffusion potential backward produced no coordinate gradients")
        grads = coords.grad[:, 1]
        if diffusion_mask is not None:
            grads[diffusion_mask.to(device=grads.device).bool()] = 0
        if torch.isnan(grads).any():
            grads = torch.nan_to_num(grads, nan=0.0)
        return grads.detach()

    def get_next_pose(
        self,
        noisy_coords: Tensor,
        pred_coords: Tensor,
        t: int,
        diffusion_mask: Tensor,
        *,
        align_motif: bool = True,
        include_motif_sidechains: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Step the pose from ``x_t`` to ``x_{t-1}``.

        Args:
            noisy_coords: Current noised coords `[L, 14 or 27, 3]`.
            pred_coords: Predicted clean coords `[L, 14 or 27, 3]`.
            t: Current timestep.
            diffusion_mask: Per-residue mask; masked (motif) residues stay fixed.
            align_motif: Whether to Kabsch-align the prediction onto the motif first.
            include_motif_sidechains: Whether to preserve motif sidechain coords.
            generator: Optional RNG for reproducible denoising.

        Returns:
            A tuple ``(next_coords, pred_coords)`` of the atom14 next pose and the (possibly aligned)
            prediction.

        Raises:
            ValueError: If the coord tensors do not hold atom14 or atom27 coords.
        """

        if noisy_coords.shape[1] not in {14, 27} or pred_coords.shape[1] not in {14, 27}:
            raise ValueError("xt and pred_coords must contain atom14 or atom27 coords")
        diffusion_mask = diffusion_mask.to(device=noisy_coords.device).bool()
        if align_motif and diffusion_mask.any():
            pred_coords = self.align_to_xt_motif(pred_coords, noisy_coords, diffusion_mask)
        pred_coords = pred_coords.to(device=noisy_coords.device, dtype=noisy_coords.dtype)
        noise_scale_ca = self.noise_schedule_ca(t)
        _out, ca_deltas = get_next_ca(
            noisy_coords,
            pred_coords,
            t,
            diffusion_mask,
            crd_scale=self.crd_scale,
            beta_schedule=self.schedule,
            alphabar_schedule=self.alphabar_schedule,
            noise_scale=noise_scale_ca,
            generator=generator,
        )
        noise_scale_frame = self.noise_schedule_frame(t)
        frames_next = get_next_frames(
            noisy_coords,
            pred_coords,
            t,
            diffuser=self.diffuser,
            so3_type=self.so3_type,
            diffusion_mask=diffusion_mask,
            noise_scale=noise_scale_frame,
        )
        grad_ca = self.get_potential_gradients(noisy_coords.clone(), diffusion_mask=diffusion_mask)
        ca_deltas = ca_deltas + self.potential_manager.get_guide_scale(t) * grad_ca
        frames_next = frames_next + ca_deltas[:, None]
        fullatom_next = torch.full_like(noisy_coords, float("nan")).unsqueeze(0)
        fullatom_next[:, :, :3] = frames_next[None]
        if include_motif_sidechains:
            fullatom_next[:, diffusion_mask, :14] = noisy_coords[None, diffusion_mask]
        return fullatom_next.squeeze(0)[:, :14], pred_coords


class Potential:
    r"""
    Base class for scalar guiding potentials maximized during denoising.

    Subclasses implement `compute` to return a scalar whose gradient with respect to CA coords nudges the
    design toward a goal (compactness, contacts, ...).
    """

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the scalar potential value for backbone ``coords`` `[L, A, 3]`.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]` (A = 14 or 27); CA is atom index 1.

        Returns:
            The scalar potential value (subclasses define the sign convention).

        Raises:
            NotImplementedError: Always; subclasses override this.
        """

        raise NotImplementedError


class MonomerRadiusOfGyration(Potential):
    r"""
    Radius-of-gyration potential rewarding compact monomers.

    Args:
        weight: Scaling weight on the potential.
        min_distance: Floor on the per-residue centroid distance.
    """

    def __init__(self, weight: float = 1.0, min_distance: float = 15.0) -> None:
        self.weight = float(weight)
        self.min_distance = float(min_distance)

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the negative radius of gyration of the monomer CA coords.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; CA is atom index 1.

        Returns:
            The weighted negative radius of gyration (maximizing it rewards compactness).
        """

        ca = coords[:, 1]
        centroid = torch.mean(ca, dim=0, keepdim=True)
        distogram = torch.cdist(ca[None].contiguous(), centroid[None].contiguous(), p=2).squeeze(0)
        distogram = torch.maximum(torch.full_like(distogram, self.min_distance), distogram)
        return -self.weight * torch.sqrt(torch.sum(distogram.square()) / ca.shape[0])


class BinderRadiusOfGyration(Potential):
    r"""
    Radius-of-gyration potential applied to the binder chain only.

    Args:
        binderlen: Number of leading residues that form the binder.
        weight: Scaling weight on the potential.
        min_distance: Floor on the per-residue centroid distance.
    """

    def __init__(self, binderlen: int, weight: float = 1.0, min_distance: float = 15.0) -> None:
        self.binderlen = int(binderlen)
        self.weight = float(weight)
        self.min_distance = float(min_distance)

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the negative radius of gyration of the binder CA coords.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; the leading ``binderlen`` residues are the binder.

        Returns:
            The weighted negative radius of gyration of the binder chain.
        """

        ca = coords[: self.binderlen, 1]
        centroid = torch.mean(ca, dim=0, keepdim=True)
        distogram = torch.cdist(ca[None].contiguous(), centroid[None].contiguous(), p=2).squeeze(0)
        distogram = torch.maximum(torch.full_like(distogram, self.min_distance), distogram)
        return -self.weight * torch.sqrt(torch.sum(distogram.square()) / ca.shape[0])


class DimerRadiusOfGyration(Potential):
    r"""
    Radius-of-gyration potential averaged over the two chains of a dimer.

    Args:
        binderlen: Number of leading residues that form the first chain.
        weight: Scaling weight on the potential.
        min_distance: Floor on the per-residue centroid distance.
    """

    def __init__(self, binderlen: int, weight: float = 1.0, min_distance: float = 15.0) -> None:
        self.binderlen = int(binderlen)
        self.weight = float(weight)
        self.min_distance = float(min_distance)

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the mean negative radius of gyration of the two dimer chains.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; split into two chains at ``binderlen``.

        Returns:
            The mean of the two chains' weighted negative radii of gyration.
        """

        binder = coords[: self.binderlen]
        receptor = coords[self.binderlen :]
        return 0.5 * (
            BinderRadiusOfGyration(len(binder), self.weight, self.min_distance).compute(binder)
            + BinderRadiusOfGyration(len(receptor), self.weight, self.min_distance).compute(receptor)
        )


class BinderContacts(Potential):
    r"""
    Potential rewarding intra-binder CA-CA contacts.

    Args:
        binderlen: Number of leading residues that form the binder.
        weight: Scaling weight on the potential.
        contact_radius: Softening radius of the contact count.
        contact_distance: Contact distance threshold.
    """

    def __init__(
        self, binderlen: int, weight: float = 1.0, contact_radius: float = 8.0, contact_distance: float = 4.0
    ) -> None:
        self.binderlen = int(binderlen)
        self.weight = float(weight)
        self.contact_radius = float(contact_radius)
        self.contact_distance = float(contact_distance)

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the weighted count of intra-binder CA-CA contacts.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; the leading ``binderlen`` residues are the binder.

        Returns:
            The weighted soft count of CA-CA contacts within the binder.
        """

        ca = coords[: self.binderlen, 1]
        distogram = torch.cdist(ca[None].contiguous(), ca[None].contiguous(), p=2)
        return self.weight * contact_count(distogram, self.contact_distance, self.contact_radius).sum()


class InterfaceContacts(Potential):
    r"""
    Potential rewarding binder-target interface CA-CA contacts.

    Args:
        binderlen: Number of leading residues that form the binder.
        weight: Scaling weight on the potential.
        contact_radius: Softening radius of the contact count.
        contact_distance: Contact distance threshold.
    """

    def __init__(
        self, binderlen: int, weight: float = 1.0, contact_radius: float = 8.0, contact_distance: float = 6.0
    ) -> None:
        self.binderlen = int(binderlen)
        self.weight = float(weight)
        self.contact_radius = float(contact_radius)
        self.contact_distance = float(contact_distance)

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the weighted count of binder-target interface CA-CA contacts.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; residues split into binder/target at ``binderlen``.

        Returns:
            The weighted soft count of CA-CA contacts across the binder-target interface.
        """

        ca_b = coords[: self.binderlen, 1]
        ca_t = coords[self.binderlen :, 1]
        distogram = torch.cdist(ca_b[None].contiguous(), ca_t[None].contiguous(), p=2)
        return self.weight * contact_count(distogram, self.contact_distance, self.contact_radius).sum()


class MonomerContacts(Potential):
    r"""
    Potential rewarding intra-monomer CA-CA contacts.

    Args:
        weight: Scaling weight on the potential.
        contact_radius: Softening radius of the contact count.
        contact_distance: Contact distance threshold.
    """

    def __init__(self, weight: float = 1.0, contact_radius: float = 8.0, contact_distance: float = 2.0) -> None:
        self.weight = float(weight)
        self.contact_radius = float(contact_radius)
        self.contact_distance = float(contact_distance)

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the weighted count of intra-monomer CA-CA contacts.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; CA is atom index 1.

        Returns:
            The weighted soft count of CA-CA contacts within the monomer.
        """

        ca = coords[:, 1]
        distogram = torch.cdist(ca[None].contiguous(), ca[None].contiguous(), p=2)
        return self.weight * contact_count(distogram, self.contact_distance, self.contact_radius).sum()


class OligomerContacts(Potential):
    r"""
    Potential rewarding (or repelling) inter-chain contacts per an oligomer contact matrix.

    Args:
        contact_matrix: Square symmetric matrix of ``{-1, 0, 1}`` per chain pair (attract/ignore/repel).
        weight_intra: Weight on intra-chain contacts.
        weight_inter: Weight on inter-chain contacts.
        contact_radius: Softening radius of the contact count.
        contact_distance: Contact distance threshold.

    Raises:
        ValueError: If the contact matrix is not square, symmetric, or valued in ``{-1, 0, 1}``.
    """

    def __init__(
        self,
        contact_matrix: np.ndarray,
        weight_intra: float = 1.0,
        weight_inter: float = 1.0,
        contact_radius: float = 8.0,
        contact_distance: float = 2.0,
    ) -> None:
        self.contact_matrix = np.asarray(contact_matrix)
        if self.contact_matrix.ndim != 2 or self.contact_matrix.shape[0] != self.contact_matrix.shape[1]:
            raise ValueError("contact matrix must be square")
        if not np.isin(self.contact_matrix.flatten(), [-1, 0, 1]).all():
            raise ValueError("contact matrix entries must be -1, 0, or 1")
        if not np.all(self.contact_matrix == self.contact_matrix.T):
            raise ValueError("contact matrix must be symmetric")
        self.nchain = self.contact_matrix.shape[0]
        self.weight_intra = float(weight_intra)
        self.weight_inter = float(weight_inter)
        self.contact_radius = float(contact_radius)
        self.contact_distance = float(contact_distance)

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the signed, weighted inter/intra-chain contact count per the contact matrix.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; ``L`` must split evenly into ``nchain`` chains.

        Returns:
            The signed, weighted soft contact count summed over all chain pairs (attractive pairs add,
            repulsive pairs subtract).

        Raises:
            ValueError: If the design length is not divisible by the number of chains.
        """

        length = coords.shape[0]
        if length % self.nchain != 0:
            raise ValueError("oligomer potential expects equal chain lengths")
        chain_len = length // self.nchain
        total = torch.zeros((), dtype=coords.dtype, device=coords.device)
        for i in range(self.nchain):
            for j in range(i, self.nchain):
                contact_type = self.contact_matrix[i, j]
                if contact_type == 0:
                    continue
                ca_i = coords[i * chain_len : (i + 1) * chain_len, 1]
                ca_j = coords[j * chain_len : (j + 1) * chain_len, 1]
                distogram = torch.cdist(ca_i[None].contiguous(), ca_j[None].contiguous(), p=2)
                scalar = self.weight_intra / 2 if i == j else self.weight_inter
                total = (
                    total
                    + contact_count(distogram, self.contact_distance, self.contact_radius).sum()
                    * float(contact_type)
                    * scalar
                )
        return total


class SubstrateContacts(Potential):
    r"""
    Ligand-contact potential used by the RFdiffusion enzyme-design examples.

    Transforms the configured substrate atoms into the current motif frame, then attracts non-motif CA atoms
    toward the substrate while repelling clashes. The substrate atoms and motif frame are wired in by the
    sampler at initialization.

    Args:
        weight: Overall scaling weight on the potential.
        contact_radius: Softening radius of the attraction contact count.
        contact_distance: Attraction contact distance threshold.
        attraction_scale: Scaling on the attraction term.
        repulsion_radius: Radius within which repulsion acts.
        repulsion_scale: Scaling on the repulsion term.
        min_repulsion_distance: Floor distance for repulsion.
        eps: Unused; accepted for interface compatibility.
    """

    def __init__(
        self,
        weight: float = 1.0,
        contact_radius: float = 8.0,
        contact_distance: float = 2.0,
        attraction_scale: float = 1.0,
        repulsion_radius: float = 5.0,
        repulsion_scale: float = 2.0,
        min_repulsion_distance: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        del eps
        self.weight = float(weight)
        self.contact_radius = float(contact_radius)
        self.contact_distance = float(contact_distance)
        self.attraction_scale = float(attraction_scale)
        self.repulsion_radius = float(repulsion_radius)
        self.repulsion_scale = float(repulsion_scale)
        self.min_repulsion_distance = float(min_repulsion_distance)
        self.motif_substrate_atoms: Tensor | None = None
        self.diffusion_mask: Tensor | None = None
        self.motif_coords: Tensor | None = None
        self.motif_frame: Tensor | None = None
        self.motif_mapping: list[tuple[int, int]] | None = None

    def compute(self, coords: Tensor) -> Tensor:
        r"""
        Return the substrate-contact potential (attraction minus clash repulsion), or zero if unwired.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]`; non-motif CA atoms are scored against the substrate.

        Returns:
            The weighted substrate-contact potential, or a zero scalar when the substrate/motif frame has not
            been wired in by the sampler.
        """

        if self.motif_substrate_atoms is None or self.diffusion_mask is None or self.motif_coords is None:
            return torch.zeros((), dtype=coords.dtype, device=coords.device)
        self._grab_motif_residues(self.motif_coords.to(device=coords.device, dtype=coords.dtype))
        if self.motif_frame is None or self.motif_mapping is None:
            return torch.zeros((), dtype=coords.dtype, device=coords.device)
        residue_index = torch.tensor([item[0] for item in self.motif_mapping], dtype=torch.long, device=coords.device)
        atom_index = torch.tensor([item[1] for item in self.motif_mapping], dtype=torch.long, device=coords.device)
        motif_mask = self.diffusion_mask.to(device=coords.device).bool()
        new_frame = coords[motif_mask][residue_index, atom_index]
        affine, translation = recover_affine(self.motif_frame, new_frame)
        substrate = self.motif_substrate_atoms.to(device=coords.device, dtype=coords.dtype)
        substrate = substrate @ affine.T + translation
        diffusion_mask = mask_expand(motif_mask, 1)
        ca = coords[~diffusion_mask, 1]
        distogram = torch.cdist(ca[None].contiguous(), substrate[None].contiguous(), p=2)[0]
        attraction = (
            -contact_count(torch.min(distogram, dim=-1).values, self.contact_distance, self.contact_radius).sum()
            * self.attraction_scale
        )
        repulsion = poly_repulse(
            torch.min(distogram, dim=-1).values, self.repulsion_radius, self.repulsion_scale, p=1.5
        ).sum()
        return -self.weight * (attraction + repulsion)

    def _grab_motif_residues(self, motif_coords: Tensor) -> None:
        r"""Select the reference motif frame and its residue/atom mapping used to place the substrate."""

        if motif_coords.shape[0] == 0:
            self.motif_frame = None
            self.motif_mapping = None
            return
        if motif_coords.shape[0] >= 4:
            selected = torch.arange(4, device=motif_coords.device)
            self.motif_frame = motif_coords[selected, 1]
            self.motif_mapping = [(int(index), 1) for index in selected]
            return
        self.motif_frame = motif_coords[0, :4]
        self.motif_mapping = [(0, atom_idx) for atom_idx in range(4)]


IMPLEMENTED_POTENTIALS = {
    "monomer_ROG": MonomerRadiusOfGyration,
    "binder_ROG": BinderRadiusOfGyration,
    "dimer_ROG": DimerRadiusOfGyration,
    "binder_ncontacts": BinderContacts,
    "interface_ncontacts": InterfaceContacts,
    "monomer_contacts": MonomerContacts,
    "olig_contacts": OligomerContacts,
    "substrate_contacts": SubstrateContacts,
}

REQUIRE_BINDERLEN = {
    "binder_ROG",
    "dimer_ROG",
    "binder_ncontacts",
    "interface_ncontacts",
}


class PotentialManager:
    r"""
    Parse, instantiate, and aggregate RFdiffusion guiding potentials.

    Reads the configured potential strings, builds the corresponding
    [`Potential`][ofoldx.models.rfdiffusion.model.Potential] objects (injecting binder length and
    oligomer contact matrices as needed), and sums their values and gradient scales during denoising.

    Args:
        potentials_config: Potential configuration (strings, guide scale, decay, oligomer options).
        diffuser_config: Diffuser configuration supplying the timestep count.
        inference_config: Inference configuration supplying the symmetry (for oligomer contacts).
        binderlen: Number of binder residues, used by binder-specific potentials.
    """

    def __init__(
        self,
        potentials_config: Mapping[str, Any] | object,
        diffuser_config: Mapping[str, Any] | object,
        inference_config: Mapping[str, Any] | object,
        binderlen: int,
    ) -> None:
        self.potentials_config = potentials_config
        self.inference_config = inference_config
        self.guide_scale = float(_cfg_get(potentials_config, "guide_scale", 10.0))
        self.guide_decay = str(_cfg_get(potentials_config, "guide_decay", "constant"))
        settings = []
        guiding = _cfg_get(potentials_config, "guiding_potentials", None)
        if guiding is not None:
            settings = [self.parse_potential_string(value) for value in guiding]
        if binderlen > 0:
            for setting in settings:
                if setting["type"] in REQUIRE_BINDERLEN:
                    setting["binderlen"] = binderlen
        self.potentials_to_apply = self.initialize_all_potentials(settings)
        self.num_timesteps = int(_cfg_get(diffuser_config, "num_timesteps", 50))

    def is_empty(self) -> bool:
        r"""Return whether no guiding potentials are configured."""

        return len(self.potentials_to_apply) == 0

    def parse_potential_string(self, potstr: str) -> dict[str, Any]:
        r"""
        Parse one terse ``"key:value,..."`` potential string into a constructor-kwargs dict.

        Args:
            potstr: A potential specification string.

        Returns:
            A settings dict with the house-vocabulary keys and float-coerced values.
        """

        raw = {entry.split(":")[0]: entry.split(":")[1] for entry in potstr.split(",")}
        setting: dict[str, Any] = {_POTENTIAL_KWARG_RENAMES.get(key, key): value for key, value in raw.items()}
        for key, value in list(setting.items()):
            if key != "type":
                setting[key] = float(value)
        return setting

    def initialize_all_potentials(self, settings: Sequence[dict[str, Any]]) -> list[Potential]:
        r"""
        Instantiate the configured potentials from parsed settings.

        Args:
            settings: Parsed potential settings dicts.

        Returns:
            The list of instantiated [`Potential`][ofoldx.models.rfdiffusion.model.Potential] objects.

        Raises:
            ValueError: If a setting names an unimplemented potential.
        """

        to_apply: list[Potential] = []
        for setting in settings:
            potential_type = setting["type"]
            if potential_type not in IMPLEMENTED_POTENTIALS:
                raise ValueError(f"unknown RFdiffusion potential {potential_type!r}")
            kwargs = {key: value for key, value in setting.items() if key != "type"}
            symmetry = _cfg_get(self.inference_config, "symmetry", None)
            if symmetry and potential_type == "olig_contacts":
                nchain = calc_nchains(str(symmetry), components=1)
                kwargs["contact_matrix"] = make_contact_matrix(
                    nchain,
                    intra_all=bool(_cfg_get(self.potentials_config, "olig_intra_all", False)),
                    inter_all=bool(_cfg_get(self.potentials_config, "olig_inter_all", False)),
                    contact_string=_cfg_get(self.potentials_config, "olig_custom_contact", None),
                )
            to_apply.append(IMPLEMENTED_POTENTIALS[potential_type](**kwargs))
        return to_apply

    def compute_all_potentials(self, coords: Tensor) -> Tensor:
        r"""
        Return the summed value of all configured potentials for backbone ``coords``.

        Args:
            coords: Backbone atom coordinates `[L, A, 3]` passed to each registered potential.

        Returns:
            The sum of every configured potential's value, or a zero scalar when none are configured.
        """

        if not self.potentials_to_apply:
            return torch.zeros((), dtype=coords.dtype, device=coords.device)
        return torch.stack([potential.compute(coords) for potential in self.potentials_to_apply]).sum()

    def get_guide_scale(self, t: int) -> float:
        r"""
        Return the guidance gradient scale at timestep ``t`` per the configured decay.

        Args:
            t: Current diffusion timestep; scaled against ``num_timesteps`` for the non-constant decays.

        Returns:
            The guidance gradient scale to multiply the potential gradient by at timestep ``t``.

        Raises:
            ValueError: If the configured guide decay is unknown.
        """

        schedules = {
            "constant": lambda t_value: self.guide_scale,
            "linear": lambda t_value: t_value / self.num_timesteps * self.guide_scale,
            "quadratic": lambda t_value: t_value**2 / self.num_timesteps**2 * self.guide_scale,
            "cubic": lambda t_value: t_value**3 / self.num_timesteps**3 * self.guide_scale,
        }
        if self.guide_decay not in schedules:
            raise ValueError(f"unknown potential guide decay {self.guide_decay!r}")
        return float(schedules[self.guide_decay](t))


def make_contact_matrix(
    nchain: int,
    *,
    intra_all: bool | None = False,
    inter_all: bool | None = False,
    contact_string: str | None = None,
) -> np.ndarray:
    r"""
    Build the symmetric oligomer contact matrix from the intra/inter flags and a custom contact spec.

    Args:
        nchain: Number of chains.
        intra_all: Whether every chain contacts itself.
        inter_all: Whether every chain pair contacts.
        contact_string: Comma-separated ``"A&B"`` (attract) / ``"A!B"`` (repel) chain-pair specs.

    Returns:
        A `[nchain, nchain]` matrix valued in ``{-1, 0, 1}``.

    Raises:
        ValueError: If a contact spec is malformed or uses an unknown operator.
    """

    contacts = np.zeros((nchain, nchain), dtype=float)
    if intra_all:
        contacts[np.arange(nchain), np.arange(nchain)] = 1
    if inter_all:
        contacts[~np.eye(nchain, dtype=bool)] = 1
    if contact_string is not None:
        alphabet = {letter: idx for idx, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        for contact in contact_string.split(","):
            if len(contact) != 3:
                raise ValueError(f"invalid oligomer contact spec {contact!r}")
            i, j = alphabet[contact[0]], alphabet[contact[2]]
            if contact[1] == "!":
                contacts[i, j] = contacts[j, i] = -1
            elif contact[1] == "&":
                contacts[i, j] = contacts[j, i] = 1
            else:
                raise ValueError(f"invalid oligomer contact operator {contact[1]!r}")
    return contacts


def calc_nchains(symbol: str, components: int = 1) -> int:
    r"""
    Return the number of chains implied by a symmetry symbol.

    Args:
        symbol: A symmetry symbol (e.g. `"c3"`, `"d2"`, `"tetrahedral"`).
        components: Multiplier for multi-component assemblies.

    Returns:
        The total number of chains.

    Raises:
        RuntimeError: If the symmetry symbol is unrecognized.
    """

    lowered = symbol.lower()
    if lowered.startswith("c"):
        return int(lowered[1:]) * components
    if lowered.startswith("d"):
        return 2 * int(lowered[1:]) * components
    if lowered.startswith("t"):
        return 12 * components
    if lowered.startswith("o"):
        return 24 * components
    if lowered.startswith("i"):
        return 60 * components
    raise RuntimeError(f"unknown symmetry symbol {symbol!r}")


# Upstream RFdiffusion ``guiding_potentials`` config strings still use the terse tokens
# (e.g. "type:substrate_contacts,r_0:8,d_0:2,s:1"); map them onto the spelled-out house
# constructor kwargs so external configs keep working (same role as convert.py's key map).
_POTENTIAL_KWARG_RENAMES = {
    "d_0": "contact_distance",
    "r_0": "contact_radius",
    "s": "attraction_scale",
    "rep_r_0": "repulsion_radius",
    "rep_s": "repulsion_scale",
    "rep_r_min": "min_repulsion_distance",
    "min_dist": "min_distance",
}


def contact_count(distogram: Tensor, contact_distance: float, contact_radius: float) -> Tensor:
    r"""
    Return the soft (differentiable) contact count of a distance map with a smooth cutoff.

    Args:
        distogram: Pairwise distance map; any shape.
        contact_distance: Distance at which a pair is counted as a full contact.
        contact_radius: Width of the smooth transition around ``contact_distance``.

    Returns:
        The soft contact indicator per pair, shaped like ``distogram`` (near 1 inside the cutoff, near 0
        outside).
    """

    divide_by = (distogram - contact_distance) / contact_radius
    numerator = torch.pow(divide_by, 6)
    denominator = torch.pow(divide_by, 12)
    return (1 - numerator) / (1 - denominator).clamp_min(1e-8)


def poly_repulse(distogram: Tensor, repulsion_radius: float, repulsion_scale: float, p: float = 1.0) -> Tensor:
    r"""
    Return a polynomial short-range repulsion penalty for distances below ``repulsion_radius``.

    Args:
        distogram: Pairwise distance map; any shape.
        repulsion_radius: Distance below which the repulsion turns on.
        repulsion_scale: Overall strength of the repulsion.
        p: Polynomial exponent controlling how sharply the penalty grows toward zero distance.

    Returns:
        The per-pair repulsion penalty, shaped like ``distogram`` (zero at or beyond ``repulsion_radius``).
    """

    a = repulsion_scale / (p * repulsion_radius ** (p - 1))
    return (distogram < repulsion_radius) * a * torch.abs(repulsion_radius - distogram) ** p * repulsion_scale


def recover_affine(frame_from: Tensor, frame_to: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Least-squares recover the affine ``(rotation, translation)`` mapping ``frame_from`` onto ``frame_to``.

    Args:
        frame_from: Source points `[N, 3]` defining the reference frame.
        frame_to: Corresponding target points `[N, 3]` the source is mapped onto.

    Returns:
        A tuple ``(rotation, translation)`` with the `[3, 3]` linear map and `[3]` offset.
    """

    ones = torch.ones((frame_from.shape[0], 1), dtype=frame_from.dtype, device=frame_from.device)
    source = torch.cat((frame_from, ones), dim=-1)
    transform = torch.linalg.lstsq(source, frame_to).solution
    return transform[:3].T, transform[3]


def mask_expand(mask: Tensor, n: int = 1) -> Tensor:
    r"""
    Dilate a boolean ``mask`` by ``n`` positions on each side (set neighbors of true entries true).

    Args:
        mask: One-dimensional boolean mask `[L]`.
        n: Number of positions to expand each true entry by on either side.

    Returns:
        The dilated boolean mask `[L]`.
    """

    out = mask.clone()
    for idx in torch.where(mask)[0]:
        start = max(int(idx) - n, 0)
        stop = min(int(idx) + n + 1, len(mask))
        out[start:stop] = True
    return out


class ContigMap:
    r"""
    Resolve RFdiffusion contig strings into reference, output, and RF-relative residue indices.

    Expands a contig specification (fixed motif segments interleaved with variable-length designed gaps) into
    per-residue reference PDB indices, output design indices, and RF-internal residue numbering, plus the
    sequence- and structure-inpainting masks. Can also be built directly from explicit index lists.

    Args:
        parsed_pdb: Parsed target dict (with ``pdb_idx``), or `False` when none is available.
        contigs: Contig strings; required unless explicit index lists are given.
        inpaint_seq: Sequence-inpainting residue spans.
        inpaint_str: Structure-inpainting residue spans.
        length: Optional total-length spec constraining the sampled contigs.
        reference_index: Explicit reference PDB indices (bypasses contig sampling).
        design_index: Explicit output design indices (required with ``reference_index``).
        residue_index: Explicit RF residue indices (required with ``reference_index``).
        inpaint_seq_tensor: Precomputed sequence-inpaint mask overriding ``inpaint_seq``.
        inpaint_str_tensor: Precomputed structure-inpaint mask overriding ``inpaint_str``.
        topo: Whether all blocks are treated as topology (receptor) blocks.
        provide_seq: Residue spans whose sequence is provided (unmasked).
        inpaint_str_strand: Structure-inpaint spans forced to strand.
        inpaint_str_helix: Structure-inpaint spans forced to helix.
        inpaint_str_loop: Structure-inpaint spans forced to loop.

    Raises:
        ValueError: If neither contigs nor a complete explicit index mapping is given, or the specification is
            internally inconsistent.
    """

    chain_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(
        self,
        parsed_pdb: Mapping[str, Any] | bool,
        contigs: Sequence[str] | None = None,
        inpaint_seq: Sequence[str] | None = None,
        inpaint_str: Sequence[str] | None = None,
        length: str | None = None,
        reference_index: Sequence[PdbIndex] | None = None,
        design_index: Sequence[PdbIndex] | None = None,
        residue_index: Sequence[int] | None = None,
        inpaint_seq_tensor: np.ndarray | None = None,
        inpaint_str_tensor: np.ndarray | None = None,
        topo: bool = False,
        provide_seq: Sequence[str] | None = None,
        inpaint_str_strand: Sequence[str] | None = None,
        inpaint_str_helix: Sequence[str] | None = None,
        inpaint_str_loop: Sequence[str] | None = None,
    ) -> None:
        if contigs is None and reference_index is None:
            raise ValueError("must specify either contigs or explicit reference_index/design_index/residue_index")
        if any(value is not None for value in (residue_index, design_index, reference_index)) and not all(
            value is not None for value in (residue_index, design_index, reference_index)
        ):
            raise ValueError("explicit contig mapping requires reference_index, design_index, and residue_index")

        # Imported lazily: the processor pulls in the processor package, which would cycle at load time.
        from .processor import _parse_length, _parse_mask_segments

        self.parsed_pdb = parsed_pdb
        self.topo = topo
        self.length = _parse_length(length)
        self.contigs = list(contigs) if contigs is not None else None
        self.inpaint_seq_arg = _parse_mask_segments(inpaint_seq)
        self.inpaint_str_arg = _parse_mask_segments(inpaint_str)
        self.inpaint_str_helix_arg = _parse_mask_segments(inpaint_str_helix)
        self.inpaint_str_strand_arg = _parse_mask_segments(inpaint_str_strand)
        self.inpaint_str_loop_arg = _parse_mask_segments(inpaint_str_loop)

        if reference_index is None:
            if self.contigs is None:
                raise ValueError("contigs cannot be None when reference_index is not provided")
            self.sampled_mask, self.contig_length, self.n_inpaint_chains = self.get_sampled_mask()
            self.receptor_chain = self.chain_order[self.n_inpaint_chains]
            (
                self.receptor,
                self.receptor_design,
                self.receptor_residue_index,
                self.inpaint,
                self.inpaint_design,
                self.inpaint_residue_index,
                self.sampled_mask_length_bound,
            ) = self.expand_sampled_mask()
            self.reference = self.inpaint + self.receptor
            self.design = self.inpaint_design + self.receptor_design
            self.residue_index = self.inpaint_residue_index + self.receptor_residue_index
        else:
            self.reference = list(reference_index)
            self.design = list(design_index or [])
            self.residue_index = list(residue_index or [])
            self.sampled_mask = []
            self.contig_length = len(self.reference)
            self.n_inpaint_chains = 1
            self.receptor = []
            self.receptor_design = []
            self.receptor_residue_index = []
            self.inpaint = list(self.reference)
            self.inpaint_design = list(self.design)
            self.inpaint_residue_index = list(self.residue_index)
            self.sampled_mask_length_bound = _bounds_from_design(self.design)

        self.mask_1d = [residue != ("_", "_") for residue in self.reference]
        if inpaint_seq_tensor is not None:
            self.inpaint_seq = inpaint_seq_tensor
        elif self.inpaint_seq_arg is None:
            self.inpaint_seq = np.asarray(self.mask_1d, dtype=bool)
        else:
            self.inpaint_seq = self.get_inpaint_seq_str(self.inpaint_seq_arg)

        if inpaint_str_tensor is not None:
            self.inpaint_str = inpaint_str_tensor
        elif self.inpaint_str_arg is None:
            self.inpaint_str = np.asarray(self.mask_1d, dtype=bool)
        else:
            self.inpaint_str = self.get_inpaint_seq_str(self.inpaint_str_arg)

        (
            self.reference_index0,
            self.design_index0,
            self.design_index0_receptor,
        ) = self.get_index0()
        self.contig_reference_pdb_index = [value for value in self.reference if value != ("_", "_")]

        if provide_seq is not None:
            for span in provide_seq:
                if "-" in span:
                    start, stop = span.split("-", maxsplit=1)
                    self.inpaint_seq[int(start) : int(stop) + 1] = True
                else:
                    self.inpaint_seq[int(span)] = True

        if any(arg is not None for arg in (inpaint_str_helix, inpaint_str_strand, inpaint_str_loop)):
            self.ss_spec: dict[str, np.ndarray] = {}
            for key, value in (
                ("helix", self.inpaint_str_helix_arg),
                ("strand", self.inpaint_str_strand_arg),
                ("loop", self.inpaint_str_loop_arg),
            ):
                self.ss_spec[key] = (
                    ~self.get_inpaint_seq_str(value, ss=True)
                    if value is not None
                    else np.zeros(len(self.inpaint_seq), dtype=bool)
                )
            for key, mask in self.ss_spec.items():
                if np.sum(mask * self.inpaint_str) != 0:
                    raise ValueError(f"secondary-structure mask {key!r} includes residues not structure-masked")
            stack = np.vstack(list(self.ss_spec.values()))
            if np.max(np.sum(stack, axis=0)) > 1:
                raise ValueError("secondary-structure masks overlap")

    def get_sampled_mask(self) -> tuple[list[str], int, int]:
        r"""
        Sample concrete lengths for the variable contig segments until the length constraint is met.

        Returns:
            A tuple ``(sampled_mask, contig_length, n_inpaint_chains)`` of the resolved contig strings, the
            total designed length, and the number of inpaint chains.

        Raises:
            ValueError: If no length-compatible sample is found, or contigs are unset.
        """

        if self.contigs is None:
            raise ValueError("contigs are required")
        for _attempt in range(100000):
            inpaint_chains = 0
            contig_list = self.contigs[0].strip().split()
            sampled_mask: list[str] = []
            sampled_mask_length = 0
            if contig_list and all(part[0].isalpha() for part in contig_list[-1].split("/")):
                contig_list[-1] = f"{contig_list[-1]}/0"
            for contig in contig_list:
                parts = contig.split("/")
                is_receptor = (all(part[0].isalpha() for part in parts[:-1]) and parts[-1] == "0") or self.topo
                if is_receptor:
                    sampled_mask.append(contig)
                    continue
                inpaint_chains += 1
                sampled_parts: list[str] = []
                for part in parts:
                    if part[0].isalpha():
                        sampled_parts.append(part)
                        sampled_mask_length += _contig_part_length(part)
                    elif part == "0":
                        sampled_parts.append("0")
                    else:
                        sampled_length = _sample_length_part(part)
                        sampled_parts.append(f"{sampled_length}-{sampled_length}")
                        sampled_mask_length += sampled_length
                sampled_mask.append("/".join(sampled_parts))
            if self.length is None or (sampled_mask_length >= self.length[0] and sampled_mask_length < self.length[1]):
                return sampled_mask, sampled_mask_length, inpaint_chains
        raise ValueError("contig string incompatible with length range")

    def expand_sampled_mask(
        self,
    ) -> tuple[list[PdbIndex], list[PdbIndex], list[int], list[PdbIndex], list[PdbIndex], list[int], list[int]]:
        r"""
        Expand the sampled contig mask into per-residue reference/design indices and chain bounds.

        Returns:
            A 7-tuple ``(receptor, receptor_design, receptor_residue_index, inpaint, inpaint_design,
            inpaint_residue_index, bounds)`` describing the receptor and designed chains.

        Raises:
            ValueError: If a fragmented receptor block spans multiple chains or is out of PDB order.
        """

        receptor: list[PdbIndex] = []
        inpaint: list[PdbIndex] = []
        receptor_design: list[PdbIndex] = []
        inpaint_design: list[PdbIndex] = []
        receptor_idx = 1
        inpaint_idx = 1
        inpaint_chain_idx = -1
        receptor_chain_break: list[tuple[int, int]] = []
        inpaint_chain_break: list[tuple[int, int]] = []
        receptor_bounds: list[int] = []
        inpaint_bounds: list[int] = []
        for contig in self.sampled_mask:
            parts = contig.split("/")
            is_receptor = (all(part[0].isalpha() for part in parts[:-1]) and parts[-1] == "0") or self.topo
            if is_receptor:
                subcontigs = parts[:-1]
                if not subcontigs:
                    continue
                if not all(part[0] == subcontigs[0][0] for part in subcontigs):
                    raise ValueError("fragmented receptor blocks must derive from one chain")
                starts = [int(part.split("-")[0][1:]) for part in subcontigs]
                if any(first >= second for first, second in zip(starts, starts[1:], strict=False)):
                    raise ValueError("receptor fragments from one chain must be in ascending PDB order")
                for idx, part in enumerate(subcontigs):
                    ref_to_add = _expand_residue_range(part)
                    receptor.extend(ref_to_add)
                    receptor_design.extend(
                        (self.receptor_chain, value) for value in range(receptor_idx, receptor_idx + len(ref_to_add))
                    )
                    receptor_idx += len(ref_to_add)
                    if idx != len(subcontigs) - 1:
                        jump = int(subcontigs[idx + 1].split("-")[0][1:]) - int(part.split("-")[1]) - 1
                        receptor_chain_break.append((receptor_idx - 1, jump))
                    else:
                        receptor_chain_break.append((receptor_idx - 1, 200))
                receptor_bounds.append(len(receptor))
            else:
                inpaint_chain_idx += 1
                for part in parts:
                    if part[0].isalpha():
                        ref_to_add = _expand_residue_range(part)
                        inpaint.extend(ref_to_add)
                        inpaint_design.extend(
                            (self.chain_order[inpaint_chain_idx], value)
                            for value in range(inpaint_idx, inpaint_idx + len(ref_to_add))
                        )
                        inpaint_idx += len(ref_to_add)
                    else:
                        length = int(part.split("-")[0])
                        inpaint.extend(cast(list[PdbIndex], [("_", "_")] * length))
                        inpaint_design.extend(
                            (self.chain_order[inpaint_chain_idx], value)
                            for value in range(inpaint_idx, inpaint_idx + length)
                        )
                        inpaint_idx += length
                inpaint_chain_break.append((inpaint_idx - 1, 200))
                inpaint_bounds.append(len(inpaint))

        if self.topo or not inpaint_design:
            receptor_design = [(chain, residue) for chain, residue in receptor_design]
        else:
            receptor_design = [(chain, residue + inpaint_design[-1][1]) for chain, residue in receptor_design]
        inpaint_residue_index = np.arange(0, len(inpaint))
        receptor_residue_index = np.arange(len(inpaint) + 200, len(inpaint) + len(receptor) + 200)
        for ch_break in inpaint_chain_break[:-1]:
            receptor_residue_index[:] += 200
            inpaint_residue_index[ch_break[0] :] += ch_break[1]
        for ch_break in receptor_chain_break[:-1]:
            receptor_residue_index[ch_break[0] :] += ch_break[1]
        bounds = list(inpaint_bounds)
        inpaint_last_bound = inpaint_bounds[-1] if inpaint_bounds else 0
        bounds.extend(value + inpaint_last_bound for value in receptor_bounds)
        return (
            receptor,
            receptor_design,
            receptor_residue_index.tolist(),
            inpaint,
            inpaint_design,
            inpaint_residue_index.tolist(),
            bounds,
        )

    def get_inpaint_seq_str(self, inpaint_s: Sequence[str], *, ss: bool = False) -> np.ndarray:
        r"""
        Build a boolean mask marking residues named in the inpaint spans as inpainted (false).

        Args:
            inpaint_s: Residue spans (single residues or ranges) to mask.
            ss: Whether to start from an all-true mask (secondary-structure spec) rather than ``mask_1d``.

        Returns:
            A boolean mask over the reference residues.
        """

        mask = np.ones(len(self.mask_1d), dtype=bool) if ss else np.asarray(self.mask_1d, dtype=bool)
        residues: list[PdbIndex] = []
        for item in inpaint_s:
            if "-" in item:
                residues.extend(_expand_residue_range(item))
            else:
                residues.append((item[0], int(item[1:])))
        for residue in residues:
            if residue in self.reference:
                mask[self.reference.index(residue)] = False
        return np.asarray(mask, dtype=bool)

    def get_index0(self) -> tuple[list[int], list[int], list[int]]:
        r"""
        Compute the zero-based reference/design index mappings against the parsed PDB order.

        Returns:
            A tuple ``(reference_index0, design_index0, design_index0_receptor)`` of the zero-based indices.

        Raises:
            ValueError: If a referenced residue is absent from the parsed PDB.
        """

        reference_index0: list[int] = []
        design_index0: list[int] = []
        design_index0_receptor: list[int] = []
        pdb_idx: list[PdbIndex] = []
        if isinstance(self.parsed_pdb, Mapping):
            pdb_idx = cast(list[PdbIndex], list(self.parsed_pdb["pdb_idx"]))
        for idx, value in enumerate(self.reference):
            if value != ("_", "_"):
                if value not in pdb_idx:
                    raise ValueError(f"{value} is not in pdb file")
                design_index0.append(idx)
                reference_index0.append(pdb_idx.index(value))
        for idx, value in enumerate(self.receptor):
            if value != ("_", "_"):
                design_index0_receptor.append(idx)
        return (
            reference_index0,
            design_index0,
            design_index0_receptor,
        )


def _contig_part_length(part: str) -> int:
    r"""Return the residue count of a fixed contig part (single residue or ``<chain><start>-<stop>`` range)."""

    if "-" not in part:
        return 1
    start, stop = part.split("-", maxsplit=1)
    return int(stop) - int(start[1:]) + 1


def _sample_length_part(part: str) -> int:
    r"""Sample a concrete length for a variable contig part (a single value or a ``low-high`` range)."""

    if "-" not in part:
        return int(part)
    start, stop = part.split("-", maxsplit=1)
    return random.randint(int(start), int(stop))


def _expand_residue_range(part: str) -> list[PdbIndex]:
    r"""Expand a fixed contig part into its list of ``(chain, residue_number)`` PDB indices."""

    if "-" not in part:
        return [(part[0], int(part[1:]))]
    start, stop = part.split("-", maxsplit=1)
    return [(part[0], residue) for residue in np.arange(int(start[1:]), int(stop) + 1)]


def _bounds_from_design(hal: Sequence[PdbIndex]) -> list[int]:
    r"""Return the per-chain end offsets (chain boundaries) of an explicit design index list."""

    if not hal:
        return [0]
    bounds: list[int] = []
    previous_chain = hal[0][0]
    for idx, (chain, _residue) in enumerate(hal, start=1):
        if chain != previous_chain:
            bounds.append(idx - 1)
            previous_chain = chain
    bounds.append(len(hal))
    return bounds


class BlockAdjacency:
    r"""
    Sample and reshape secondary-structure and block-adjacency tensors from scaffold files.

    Loads precomputed ``*_ss.pt`` / ``*_adj.pt`` scaffold tensors, expands their loops by random insertions and
    N/C-terminal extensions, and masks them per the configured erosion, producing the SS/adjacency conditioning
    for scaffold-guided sampling.

    Args:
        conf: Configuration object holding the ``scaffoldguided`` (and ``inference``) sub-config.
        num_designs: Number of designs to generate (used for deterministic seeding).

    Raises:
        ValueError: If ``scaffold_list`` is neither a list nor a `.txt` path, or masked-loop settings conflict
            with sampled extensions.
    """

    def __init__(self, conf: Mapping[str, Any] | object, num_designs: int) -> None:
        self.conf = conf
        scaffoldguided = _cfg_get(conf, "scaffoldguided", conf)
        scaffold_list = _cfg_get(scaffoldguided, "scaffold_list", None)
        if scaffold_list is not None:
            if isinstance(scaffold_list, list):
                self.scaffold_list = scaffold_list
            elif str(scaffold_list).endswith(".txt"):
                with open(scaffold_list) as handle:
                    self.scaffold_list = [line.strip() for line in handle if line.strip()]
            else:
                raise ValueError("scaffold_list must be a list or .txt path")
        else:
            scaffold_dir = _cfg_get(scaffoldguided, "scaffold_dir", None)
            self.scaffold_list = [os.path.split(path)[1][:-6] for path in glob.glob(f"{scaffold_dir}/*_ss.pt")]
            self.scaffold_list.sort()
        self.scaffold_dir = _cfg_get(scaffoldguided, "scaffold_dir", None)
        # Imported lazily: the processor pulls in the processor package, which would cycle at load time.
        from .processor import _parse_range

        self.sampled_insertion = _parse_range(_cfg_get(scaffoldguided, "sampled_insertion", 0))
        self.sampled_N = _parse_range(_cfg_get(scaffoldguided, "sampled_N", 0))
        self.sampled_C = _parse_range(_cfg_get(scaffoldguided, "sampled_C", 0))
        self.ss_mask = int(_cfg_get(scaffoldguided, "ss_mask", 0))
        self.systematic = bool(_cfg_get(scaffoldguided, "systematic", False))
        self.num_designs = int(num_designs)
        self.num_completed = 0
        self.item_n = 0
        self.mask_loops = bool(_cfg_get(scaffoldguided, "mask_loops", True))
        if not self.mask_loops and (
            self.sampled_N != [0, 0] or self.sampled_C != [0, 0] or self.sampled_insertion != [0, 0]
        ):
            raise ValueError("cannot add sampled length when mask_loops is false")

    def get_ss_adj(self, item: str) -> tuple[Tensor, Tensor]:
        r"""
        Load the ``(ss, adj)`` secondary-structure and adjacency tensors for a scaffold ``item``.

        Args:
            item: Scaffold file stem; the ``_ss.pt`` and ``_adj.pt`` tensors are loaded from ``scaffold_dir``.

        Returns:
            The ``(ss, adj)`` secondary-structure and block-adjacency tensors.
        """

        ss = torch.load(os.path.join(self.scaffold_dir, f"{item.split('.')[0]}_ss.pt"), weights_only=True)
        adj = torch.load(os.path.join(self.scaffold_dir, f"{item.split('.')[0]}_adj.pt"), weights_only=True)
        return ss, adj

    def mask_to_segments(self, mask: Tensor) -> list[tuple[str, int]]:
        r"""
        Split a loop/SS boolean ``mask`` into a list of ``(kind, length)`` runs (``"ss"`` or ``"loop"``).

        Args:
            mask: Per-residue boolean mask `[L]`; true marks loop residues, false marks structured (SS) ones.

        Returns:
            Consecutive runs as ``(kind, length)`` pairs, ``kind`` being ``"loop"`` or ``"ss"``.
        """

        segments: list[tuple[str, int]] = []
        begin = 0
        for idx in range(1, mask.shape[0]):
            if mask[idx] != mask[idx - 1]:
                segments.append(("loop" if bool(mask[idx - 1]) else "ss", idx - begin))
                begin = idx
        segments.append(("loop" if bool(mask[-1]) else "ss", mask.shape[0] - begin))
        return segments

    def expand_mask(self, mask: Tensor, segments: list[tuple[str, int]]) -> Tensor:
        r"""
        Expand the scaffold mask with random terminal extensions and loop insertions.

        Args:
            mask: The original loop/SS boolean mask.
            segments: The ``(kind, length)`` runs from `mask_to_segments`.

        Returns:
            The expanded boolean mask.

        Raises:
            ValueError: If expansion drops secondary-structure residues.
        """

        N_add = random.randint(self.sampled_N[0], self.sampled_N[1])
        C_add = random.randint(self.sampled_C[0], self.sampled_C[1])
        output: list[bool] = [False] * N_add
        for kind, length in segments:
            if kind == "ss":
                output.extend([True] * length)
            else:
                insertion = random.randint(self.sampled_insertion[0], self.sampled_insertion[1])
                output.extend([False] * (length + insertion))
        output.extend([False] * C_add)
        expanded = torch.tensor(output, dtype=torch.bool)
        if int(expanded.sum()) != int((~mask).sum()):
            raise ValueError("expanded scaffold mask lost secondary-structure residues")
        return expanded

    def expand_ss(self, ss: Tensor, adj: Tensor, mask: Tensor, expanded_mask: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Scatter the secondary-structure and adjacency tensors into the expanded (extended) layout.

        Args:
            ss: Original per-residue secondary-structure labels.
            adj: Original block-adjacency matrix.
            mask: Original loop/SS mask.
            expanded_mask: The expanded mask marking SS positions in the new layout.

        Returns:
            The expanded ``(ss, adj)`` tensors.
        """

        ss_out = torch.ones(expanded_mask.shape[0], dtype=ss.dtype) * 3
        adj_out = torch.zeros((expanded_mask.shape[0], expanded_mask.shape[0]), dtype=adj.dtype)
        ss_out[expanded_mask] = ss[~mask]
        expanded_mask_2d = torch.ones_like(adj_out, dtype=torch.bool)
        expanded_mask_2d[~expanded_mask] = False
        expanded_mask_2d[:, ~expanded_mask] = False
        mask_2d = torch.ones_like(adj, dtype=torch.bool)
        mask_2d[mask] = False
        mask_2d[:, mask] = False
        adj_out[expanded_mask_2d] = adj[mask_2d]
        return ss_out, adj_out.reshape((expanded_mask.shape[0], expanded_mask.shape[0]))

    def mask_ss_adj(self, ss: Tensor, adj: Tensor, expanded_mask: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Erode secondary-structure segment ends and mask loop entries in the SS/adjacency tensors.

        Args:
            ss: Expanded secondary-structure labels (modified in place).
            adj: Expanded adjacency matrix (modified in place).
            expanded_mask: The expanded SS mask.

        Returns:
            The masked ``(ss, adj)`` tensors.
        """

        original = expanded_mask.clone()
        if self.ss_mask > 0:
            for idx in range(1, self.ss_mask + 1):
                expanded_mask[idx:] &= original[:-idx]
                expanded_mask[:-idx] &= original[idx:]
        if self.mask_loops:
            ss[~expanded_mask] = 3
            adj[~expanded_mask] = 0
            adj[:, ~expanded_mask] = 0
        adj[~expanded_mask] = 2
        adj[:, ~expanded_mask] = 2
        return ss, adj

    def get_scaffold(self) -> tuple[int, Tensor, Tensor]:
        r"""
        Sample one scaffold and return its length and conditioning tensors.

        Returns:
            A tuple ``(length, ss_one_hot, adj)`` of the sampled scaffold length, its one-hot
            secondary-structure tensor, and its block-adjacency matrix.
        """

        inference = _cfg_get(self.conf, "inference", self.conf)
        if bool(_cfg_get(inference, "deterministic", False)):
            torch.manual_seed(self.num_completed)
            np.random.seed(self.num_completed)
            random.seed(self.num_completed)
        if self.systematic:
            if self.item_n >= len(self.scaffold_list):
                self.item_n = 0
            item = self.scaffold_list[self.item_n]
            self.item_n += 1
        else:
            item = random.choice(self.scaffold_list)
        ss, adj = self.get_ss_adj(item)
        mask = torch.where(ss == 2, 1, 0).bool()
        segments = self.mask_to_segments(mask)
        expanded_mask = self.expand_mask(mask, segments)
        ss, adj = self.expand_ss(ss, adj, mask, expanded_mask)
        ss, adj = self.mask_ss_adj(ss, adj, expanded_mask)
        self.num_completed += 1
        return ss.shape[0], F.one_hot(ss.long(), num_classes=4), adj


class Target:
    r"""
    Target PDB loader and crop helper for scaffold-guided binder design.

    Parses the target PDB, marks its hotspot residues, and optionally crops it to a contig region while
    preserving the hotspots.

    Args:
        conf: Configuration object holding ``target_path`` and optional ``contig_crop``.
        hotspots: Hotspot residue ids as `"<chain><number>"` strings, or `None`.
    """

    def __init__(self, conf: Mapping[str, Any] | object, hotspots: list[str] | None = None) -> None:
        from .processor import parse_pdb

        self.pdb = parse_pdb(_cfg_get(conf, "target_path", None), parse_hetatom=True)
        self.hotspots = [] if hotspots is None else hotspots
        self.pdb["hotspots"] = np.asarray(
            [f"{chain}{residue}" in self.hotspots for chain, residue in self.pdb["pdb_idx"]]
        )
        contig_crop = _cfg_get(conf, "contig_crop", None)
        if contig_crop:
            self.contig_crop(contig_crop)

    def parse_contig(self, contig_crop: list[str]) -> list[list[tuple[str, int]]]:
        r"""
        Parse a contig-crop spec into per-contig lists of ``(chain, residue_number)`` indices.

        Args:
            contig_crop: Contig-crop specification; its first entry is split into space-separated contigs, each a
                ``/``-joined list of ``<chain><start>-<stop>`` ranges.

        Returns:
            One list of ``(chain, residue_number)`` pairs per contig.
        """

        contig_list: list[list[tuple[str, int]]] = []
        for contig in contig_crop[0].split(" "):
            residues: list[tuple[str, int]] = []
            for crop in contig.split("/"):
                if crop and crop[0].isalpha():
                    start, stop = crop.split("-", maxsplit=1)
                    residues.extend((crop[0], int(value)) for value in np.arange(int(start[1:]), int(stop) + 1))
            contig_list.append(residues)
        return contig_list

    def contig_crop(self, contig_crop: list[str], *, residue_offset: int = 200) -> None:
        r"""
        Crop the loaded target PDB to the residues named in the contig-crop spec, in place.

        Args:
            contig_crop: The contig-crop specification.
            residue_offset: Residue-number offset inserted between cropped chains.

        Raises:
            ValueError: If cropping would drop hotspot residues.
        """

        for idx, value in enumerate(self.pdb["pdb_idx"]):
            if idx != 0 and value != self.pdb["pdb_idx"][idx - 1]:
                self.pdb["idx"][idx:] += residue_offset + idx
        contig_list = self.parse_contig(contig_crop)
        for contig in contig_list[1:]:
            start = int(contig[0][1])
            self.pdb["idx"][start:] += residue_offset
        flat = [item for contig in contig_list for item in contig]
        mask = np.asarray([value in flat for value in self.pdb["pdb_idx"]])
        if np.sum(self.pdb["hotspots"]) != np.sum(self.pdb["hotspots"][mask]):
            raise ValueError("supplied hotspot residues are missing from the target contig")
        for key, value in list(self.pdb.items()):
            try:
                self.pdb[key] = value[mask]
            except Exception:
                self.pdb[key] = [item for idx, item in enumerate(value) if mask[idx]]
        self.pdb["crop_mask"] = mask

    def get_target(self) -> dict[str, Any]:
        r"""Return the (possibly cropped) parsed target PDB dictionary."""

        return self.pdb


def ss_from_contig(ss_masks: dict[str, np.ndarray]) -> Tensor:
    r"""
    Convert helix/strand/loop boolean masks into RFdiffusion one-hot secondary-structure features.

    Args:
        ss_masks: Dict with ``"helix"``, ``"strand"``, and ``"loop"`` boolean masks over residues.

    Returns:
        A one-hot secondary-structure tensor `[L, 4]` (class 3 = unassigned/masked).
    """

    length = len(ss_masks["helix"])
    ss = torch.zeros((length, 4), dtype=torch.long)
    ss[:, 3] = 1
    for idx, key in enumerate(("helix", "strand", "loop")):
        mask = torch.as_tensor(ss_masks[key], dtype=torch.bool)
        ss[mask, idx] = 1
        ss[mask, 3] = 0
    return ss


_SAVED_GROUPS = {
    "tetrahedral": "T",
    "octahedral": "O",
    "icosahedral": "I",
}


class SymmetryGenerator:
    r"""
    Apply RFdiffusion-style cyclic, dihedral, tetrahedral, octahedral, or icosahedral symmetry.

    Replicates the first subunit's coords and sequence onto all symmetry copies each step, and relabels the
    per-chain residue indices so the symmetric assembly is well-formed. The concrete ``apply_symmetry`` method
    is chosen at construction from ``global_sym``.

    Args:
        global_sym: Symmetry symbol (e.g. `"c3"`, `"d2"`, `"tetrahedral"`, `"octahedral"`, `"icosahedral"`,
            `"t3"`).
        recenter: Whether to recenter subunits (octahedral placement).
        radius: Placement radius for polyhedral symmetry.
        model_only_neighbors: Unused; accepted for interface compatibility.

    Raises:
        ValueError: If the symmetry symbol is unrecognized or malformed.
    """

    def __init__(self, global_sym: str, recenter: bool, radius: float, model_only_neighbors: bool = False) -> None:
        del model_only_neighbors
        self._recenter = bool(recenter)
        self._radius = float(radius)
        symbol = global_sym.lower()
        if symbol.startswith("c"):
            if not symbol[1:].isdigit():
                raise ValueError(f"invalid cyclic symmetry {global_sym!r}")
            self._init_cyclic(int(symbol[1:]))
            self.apply_symmetry = self._apply_cyclic
        elif symbol.startswith("d"):
            if not symbol[1:].isdigit():
                raise ValueError(f"invalid dihedral symmetry {global_sym!r}")
            self._init_group(f"D{int(symbol[1:])}")
            self.apply_symmetry = self._apply_cyclic
        elif symbol == "t3":
            self.sym_rots = [
                torch.eye(3, dtype=torch.float32),
                torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
                torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            ]
            self.order = 4
            self.apply_symmetry = self._apply_cyclic
        elif symbol in _SAVED_GROUPS:
            self._init_group(_SAVED_GROUPS[symbol])
            self.apply_symmetry = self._apply_octahedral if symbol == "octahedral" else self._apply_cyclic
        else:
            raise ValueError(f"unrecognized RFdiffusion symmetry {global_sym!r}")
        self.res_idx_procesing = fn.partial(self._lin_chainbreaks, num_breaks=self.order)

    def _init_cyclic(self, order: int) -> None:
        r"""Build the ``order`` rotation matrices of a cyclic group about the z-axis."""

        self.sym_rots = [
            torch.as_tensor(Rotation.from_euler("z", i * 360.0 / order, degrees=True).as_matrix(), dtype=torch.float32)
            for i in range(order)
        ]
        self.order = order

    def _init_group(self, group: str) -> None:
        r"""Build the rotation matrices of a named point group, with the identity ordered first."""

        matrices = Rotation.create_group(group).as_matrix()
        identity = np.eye(3)
        order = list(range(len(matrices)))
        identity_index = next(
            (idx for idx, matrix in enumerate(matrices) if np.isclose(((matrix - identity) ** 2).sum(), 0)), 0
        )
        order.insert(0, order.pop(identity_index))
        self.sym_rots = [torch.as_tensor(matrices[idx], dtype=torch.float32) for idx in order]
        self.order = len(self.sym_rots)

    def _apply_cyclic(self, coords_in: Tensor, seq_in: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Replicate the first subunit onto all symmetry copies by rotation.

        Raises:
            ValueError: If the sequence length is not divisible by the symmetry order.
        """

        coords_out = coords_in.clone()
        seq_out = seq_in.clone()
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(f"sequence length must be divisible by symmetry order {self.order}")
        subunit_len = seq_out.shape[0] // self.order
        base_coords = coords_out[:subunit_len]
        base_seq = seq_out[:subunit_len]
        for idx, rotation in enumerate(self.sym_rots):
            start = subunit_len * idx
            stop = subunit_len * (idx + 1)
            rot = rotation.to(device=coords_in.device, dtype=coords_in.dtype)
            coords_out[start:stop] = torch.einsum("bnj,kj->bnk", base_coords, rot)
            seq_out[start:stop] = base_seq
        return coords_out, seq_out

    def _apply_octahedral(self, coords_in: Tensor, seq_in: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Replicate the first subunit onto octahedral copies, recentering each onto a placement axis.

        Raises:
            ValueError: If the sequence length is not divisible by the symmetry order.
        """

        coords_out = coords_in.clone()
        seq_out = seq_in.clone()
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(f"sequence length must be divisible by symmetry order {self.order}")
        subunit_len = seq_out.shape[0] // self.order
        base_axis = torch.tensor([self._radius, 0.0, 0.0], dtype=coords_in.dtype, device=coords_in.device)[None]
        for idx, rotation in enumerate(self.sym_rots):
            start = subunit_len * idx
            stop = subunit_len * (idx + 1)
            rot = rotation.to(device=coords_in.device, dtype=coords_in.dtype)
            subunit = torch.einsum("bnj,kj->bnk", coords_in[:subunit_len], rot)
            if self._recenter:
                center = torch.mean(subunit[:, 1], dim=0)
                subunit = subunit - center[None, None]
                rotated_axis = torch.einsum("nj,kj->nk", base_axis, rot)
                subunit = subunit + rotated_axis[:, None]
            coords_out[start:stop] = subunit
            seq_out[start:stop] = seq_in[:subunit_len]
        return coords_out, seq_out

    def _lin_chainbreaks(self, num_breaks: int, res_idx: Tensor, offset: int | None = None) -> tuple[Tensor, list[str]]:
        r"""
        Offset per-subunit residue indices and assign chain labels so subunits form distinct chains.

        Raises:
            ValueError: If ``res_idx`` is not `(B, L)`.
        """

        if res_idx.ndim != 2:
            raise ValueError("residue indices must have shape (B, L)")
        res_idx = res_idx.clone()
        subunit_len = res_idx.shape[-1] // num_breaks
        if offset is None:
            offset = res_idx.shape[-1]
        chain_labels = list(string.ascii_uppercase) + [
            f"{first}{second}" for first in string.ascii_uppercase for second in string.ascii_uppercase
        ]
        chain_delimiters: list[str] = []
        for idx in range(num_breaks):
            start = subunit_len * idx
            stop = subunit_len * (idx + 1)
            chain_delimiters.extend([chain_labels[idx]] * subunit_len)
            res_idx[:, start:stop] = res_idx[:, start:stop] + offset * (idx + 1)
        return res_idx, chain_delimiters


def seq_to_chars(seq: Tensor | list[int] | tuple[int, ...]) -> str:
    r"""
    Convert RFdiffusion residue ids to a one-letter amino-acid sequence string.

    Args:
        seq: Residue ids as a tensor or a sequence of ints.

    Returns:
        The one-letter sequence string (unknown residues render as `"?"`).

    Examples:
        >>> from ofoldx.models.rfdiffusion.model import seq_to_chars
        >>> seq_to_chars([0, 7, 20])
        'AG?'
    """

    values = seq.detach().cpu().tolist() if isinstance(seq, Tensor) else list(seq)
    return "".join(AA3_TO_1.get(NUM2AA[int(value)], "?") for value in values)


PdbIndex = tuple[str, int]

PAIR_PARAMS = {
    "DMIN": 2.0,
    "DMAX": 20.0,
    "DBINS": 36,
    "ABINS": 36,
}


def target_features_from_tensors(
    *,
    residue_type: Tensor,
    residue_index: Tensor,
    chain_index: Tensor,
    ca_coords: Tensor,
    atom_mask: Tensor | None = None,
) -> dict[str, Any]:
    r"""
    Build a minimal RFdiffusion target dictionary from already-tokenized input tensors.

    Places the CA coords into an atom27 slot and derives ``pdb_idx`` from the chain/residue indices, giving the
    sampler a target-feature dict equivalent to a parsed PDB.

    Args:
        residue_type: Per-token amino-acid type ids `[L]`.
        residue_index: Per-token residue numbers `[L]`.
        chain_index: Per-token chain ids `[L]`.
        ca_coords: Per-token CA coords `[L, 3]`.
        atom_mask: Optional atom-present mask.

    Returns:
        A target dict with ``xyz_27``, ``mask_27``, ``seq``, and ``pdb_idx``.

    Raises:
        ValueError: If the index tensors are not 1D or ``ca_coords`` is not `(L, 3)`.
    """

    if residue_type.ndim != 1 or residue_index.ndim != 1 or chain_index.ndim != 1:
        raise ValueError("target tensor features must be unbatched 1D tensors")
    if ca_coords.shape != (residue_type.shape[0], 3):
        raise ValueError("ca_coords must have shape (L, 3)")
    coords27 = torch.full((residue_type.shape[0], 27, 3), float("nan"), dtype=ca_coords.dtype, device=ca_coords.device)
    coords27[:, 1] = ca_coords
    mask_27 = torch.zeros((residue_type.shape[0], 27), dtype=torch.bool, device=ca_coords.device)
    mask_27[:, 1] = True
    if atom_mask is not None:
        mask_27[:, : atom_mask.shape[-1]] = atom_mask.bool()
    chain_ids = _chain_ids_from_index(chain_index.detach().cpu().tolist())
    pdb_idx = [
        (chain_id, int(residue_number))
        for chain_id, residue_number in zip(chain_ids, residue_index.detach().cpu().tolist(), strict=True)
    ]
    return {
        "xyz_27": coords27,
        "mask_27": mask_27,
        "seq": residue_type.long(),
        "pdb_idx": pdb_idx,
    }


def one_hot_sequence(seq: Tensor, *, num_classes: int = 22) -> Tensor:
    r"""
    Return one-hot RFdiffusion sequence features.

    Args:
        seq: Residue ids `[L]`.
        num_classes: Number of classes (residue vocabulary size).

    Returns:
        One-hot sequence features `[L, num_classes]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import one_hot_sequence
        >>> one_hot_sequence(torch.tensor([0, 1])).shape
        torch.Size([2, 22])
    """

    return F.one_hot(seq.long().clamp(0, num_classes - 1), num_classes=num_classes).float()


def generate_cbeta(nitrogen: Tensor, alpha_carbon: Tensor, carbon: Tensor) -> Tensor:
    r"""
    Reconstruct C-beta coords from backbone N/CA/C coords.

    Args:
        nitrogen: N coords `[..., 3]`.
        alpha_carbon: CA coords `[..., 3]`.
        carbon: C coords `[..., 3]`.

    Returns:
        The reconstructed C-beta coords `[..., 3]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import generate_cbeta
        >>> n = torch.tensor([[0.0, 0.0, 0.0]])
        >>> ca = torch.tensor([[1.0, 0.0, 0.0]])
        >>> c = torch.tensor([[1.0, 1.0, 0.0]])
        >>> generate_cbeta(n, ca, c).shape
        torch.Size([1, 3])
    """

    b = alpha_carbon - nitrogen
    c = carbon - alpha_carbon
    a = torch.cross(b, c, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + alpha_carbon


def rigid_from_3_points(
    nitrogen: Tensor, alpha_carbon: Tensor, carbon: Tensor, *, eps: float = 1e-8
) -> tuple[Tensor, Tensor]:
    r"""
    Build orthonormal residue frames from backbone N/CA/C coords (Gram-Schmidt).

    Args:
        nitrogen: N coords `[..., 3]`.
        alpha_carbon: CA coords `[..., 3]`.
        carbon: C coords `[..., 3]`.
        eps: Numerical floor for the normalizations.

    Returns:
        A tuple ``(rotations, translations)`` of the per-residue rotation matrices `[..., 3, 3]` and CA
        translations `[..., 3]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import rigid_from_3_points
        >>> n = torch.tensor([[[0.0, 0.0, 0.0]]])
        >>> ca = torch.tensor([[[1.0, 0.0, 0.0]]])
        >>> c = torch.tensor([[[1.0, 1.0, 0.0]]])
        >>> rotations, translations = rigid_from_3_points(n, ca, c)
        >>> rotations.shape
        torch.Size([1, 1, 3, 3])
    """

    v1 = carbon - alpha_carbon
    v2 = nitrogen - alpha_carbon
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    u2 = v2 - (torch.einsum("...i,...i->...", e1, v2)[..., None] * e1)
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1), alpha_carbon


def pair_dist(a: Tensor, b: Tensor) -> Tensor:
    r"""
    Return the pairwise Euclidean distance matrix between point sets ``a`` and ``b``.

    Args:
        a: First point set `[..., M, 3]`.
        b: Second point set `[..., N, 3]`.

    Returns:
        The pairwise distance matrix `[..., M, N]`.
    """

    return torch.cdist(a, b, p=2)


def planar_angle(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    r"""
    Return the angle at vertex ``b`` of the triples ``(a, b, c)`` in radians.

    Args:
        a: First point(s) `[..., 3]`.
        b: Vertex point(s) `[..., 3]` where the angle is measured.
        c: Third point(s) `[..., 3]`.

    Returns:
        The planar angle `[...]` at ``b`` in radians, in ``[0, pi]``.
    """

    v = a - b
    w = c - b
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
    w = w / torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.acos(torch.clamp(torch.sum(v * w, dim=-1), -1.0, 1.0))


def dihedral(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    r"""
    Return the signed dihedral angle of the four-point sequence ``(a, b, c, d)`` in radians.

    Args:
        a: First point(s) `[..., 3]`.
        b: Second point(s) `[..., 3]` (start of the central bond).
        c: Third point(s) `[..., 3]` (end of the central bond).
        d: Fourth point(s) `[..., 3]`.

    Returns:
        The signed dihedral angle `[...]` in radians, in ``(-pi, pi]``.
    """

    beta_initial = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    v = beta_initial - torch.sum(beta_initial * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    return torch.atan2(y, x)


def coords_to_c6d(coords: Tensor, params: dict[str, float | int] = PAIR_PARAMS) -> tuple[Tensor, Tensor]:
    r"""
    Convert backbone coords into the RFdiffusion 6D distance/orientation maps.

    Args:
        coords: Backbone coords `[B, L, 3, 3]` (N/CA/C).
        params: Distance-binning parameters (``DMIN``, ``DMAX``, ``DBINS``, ``ABINS``).

    Returns:
        A tuple ``(coords_6d, mask)`` where ``coords_6d`` is `[B, L, L, 4]` (distance plus three angles) and
        ``mask`` marks valid, in-range pairs.
    """

    batch, nres = coords.shape[:2]
    N, Ca, C = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
    Cb = generate_cbeta(N, Ca, C)
    coords_6d = torch.zeros((batch, nres, nres, 4), dtype=coords.dtype, device=coords.device)
    dist = pair_dist(Cb, Cb)
    dist = torch.nan_to_num(dist, nan=999.9)
    coords_6d[..., 0] = dist + 999.9 * torch.eye(nres, dtype=coords.dtype, device=coords.device)[None]
    b_idx, i_idx, j_idx = torch.where(coords_6d[..., 0] < float(params["DMAX"]))
    if b_idx.numel() > 0:
        coords_6d[b_idx, i_idx, j_idx, 1] = dihedral(
            Ca[b_idx, i_idx], Cb[b_idx, i_idx], Cb[b_idx, j_idx], Ca[b_idx, j_idx]
        )
        coords_6d[b_idx, i_idx, j_idx, 2] = dihedral(
            N[b_idx, i_idx], Ca[b_idx, i_idx], Cb[b_idx, i_idx], Cb[b_idx, j_idx]
        )
        coords_6d[b_idx, i_idx, j_idx, 3] = planar_angle(Ca[b_idx, i_idx], Cb[b_idx, i_idx], Cb[b_idx, j_idx])
    coords_6d[..., 0][coords_6d[..., 0] >= float(params["DMAX"])] = 999.9
    mask = torch.zeros((batch, nres, nres), dtype=coords.dtype, device=coords.device)
    mask[b_idx, i_idx, j_idx] = 1.0
    return coords_6d, mask


def dist_to_onehot(dist: Tensor, params: dict[str, float | int] = PAIR_PARAMS) -> Tensor:
    r"""
    Bin a distance map into one-hot distance features (with one extra out-of-range bin).

    Args:
        dist: Pairwise distance map; NaNs are treated as out of range.
        params: Distance-binning parameters (``DMIN``, ``DMAX``, ``DBINS``).

    Returns:
        One-hot distance features `[..., DBINS + 1]`, the final bin catching out-of-range distances.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import dist_to_onehot
        >>> dist = torch.tensor([[0.0, 999.9]])
        >>> dist_to_onehot(dist).shape
        torch.Size([1, 2, 37])
    """

    dist = torch.nan_to_num(dist, nan=999.9)
    dstep = (float(params["DMAX"]) - float(params["DMIN"])) / int(params["DBINS"])
    dbins = torch.linspace(
        float(params["DMIN"]) + dstep,
        float(params["DMAX"]),
        int(params["DBINS"]),
        dtype=dist.dtype,
        device=dist.device,
    )
    bins = torch.bucketize(dist.contiguous(), dbins).long()
    return F.one_hot(bins, num_classes=int(params["DBINS"]) + 1).float()


def coords_to_t2d(template_coords: Tensor, params: dict[str, float | int] = PAIR_PARAMS) -> Tensor:
    r"""
    Convert template backbone coords into the RFdiffusion ``template_2d`` pair features.

    Concatenates one-hot distance bins, sine/cosine orientation features, and a validity flag.

    Args:
        template_coords: Template coords `[B, T, L, A, 3]`.
        params: Distance-binning parameters.

    Returns:
        The ``template_2d`` features `[B, T, L, L, C]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import coords_to_t2d
        >>> coords = torch.zeros(1, 1, 2, 3, 3)
        >>> coords_to_t2d(coords).shape[:4]
        torch.Size([1, 1, 2, 2])
    """

    B, T, length = template_coords.shape[:3]
    coords_6d, mask = coords_to_c6d(template_coords[:, :, :, :3].reshape(B * T, length, 3, 3), params=params)
    coords_6d = coords_6d.reshape(B, T, length, length, 4)
    mask = mask.reshape(B, T, length, length, 1)
    dist = dist_to_onehot(coords_6d[..., 0], params=params)
    orien = torch.cat((torch.sin(coords_6d[..., 1:]), torch.cos(coords_6d[..., 1:])), dim=-1) * mask
    valid = ~torch.isnan(coords_6d[..., 0])
    template_2d = torch.cat((dist, orien, valid.unsqueeze(-1).float()), dim=-1)
    return torch.nan_to_num(template_2d, nan=0.0)


def get_init_coords(template_coords: Tensor) -> Tensor:
    r"""
    Fill missing template coords with the RFdiffusion ideal residue frame placed near the nearest known CA.

    Args:
        template_coords: Template coords `[B, T, L, 27, 3]` with NaNs for missing atoms.

    Returns:
        The filled template coords `[B, T, L, 27, 3]`.
    """

    B, T, length = template_coords.shape[:3]
    init = (
        INIT_CRDS.to(device=template_coords.device, dtype=template_coords.dtype)
        .reshape(1, 1, 1, 27, 3)
        .repeat(B, T, length, 1, 1)
    )
    if torch.isnan(template_coords).all():
        return init
    missing = torch.isnan(template_coords[:, :, :, :3]).any(dim=-1).any(dim=-1)
    center_ca = ((~missing[:, :, :, None]) * torch.nan_to_num(template_coords[:, :, :, 1])).sum(dim=2)
    center_ca = center_ca / ((~missing[:, :, :, None]).sum(dim=2) + 1e-4)
    centered_coords = template_coords - center_ca.reshape(B, T, 1, 1, 3)
    for b_idx in range(B):
        for t_idx in range(T):
            if missing[b_idx, t_idx].all():
                continue
            existing = torch.where(~missing[b_idx, t_idx])[0]
            nearest = torch.argmin(
                (torch.arange(length, device=template_coords.device)[:, None] - existing[None]).abs(), dim=-1
            )
            gather_idx = torch.gather(existing, 0, nearest)
            offset = torch.gather(centered_coords[b_idx, t_idx, :, 1], 0, gather_idx[:, None].expand(-1, 3))
            init[b_idx, t_idx] += offset.reshape(length, 1, 3)
    return torch.where(missing.reshape(B, T, length, 1, 1), init, centered_coords)


def torsion_features(coords: Tensor, seq: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Compute the RFdiffusion alpha torsion features and their validity masks.

    Computes backbone (omega/phi/psi) and sidechain chi torsions and encodes them as ``(cos, sin)`` pairs.

    Args:
        coords: Backbone/sidechain coords `[B, L, A, 3]`.
        seq: Per-token residue ids `[B, L]` (or flattenable to it).

    Returns:
        A tuple ``(features, mask)`` of the torsion features `[B, L, 10, 2]` and the validity mask
        `[B, L, 10]`.

    Raises:
        ValueError: If ``coords`` is not `(B, L, A, 3)`.
    """

    if coords.ndim != 4:
        raise ValueError("xyz must have shape (B, L, A, 3)")
    B, L = coords.shape[:2]
    seq = seq.long().reshape(B, L).clamp(0, len(NUM2AA) - 1)
    angles = torch.zeros((B, L, 10), dtype=coords.dtype, device=coords.device)
    mask = torch.zeros((B, L, 10), dtype=torch.bool, device=coords.device)
    valid = ~torch.isnan(coords[..., 0])
    if L > 1:
        omega_valid = valid[:, :-1, 1] & valid[:, :-1, 2] & valid[:, 1:, 0] & valid[:, 1:, 1]
        phi_valid = valid[:, :-1, 2] & valid[:, 1:, 0] & valid[:, 1:, 1] & valid[:, 1:, 2]
        angles[:, :-1, 0] = dihedral(coords[:, :-1, 1], coords[:, :-1, 2], coords[:, 1:, 0], coords[:, 1:, 1])
        angles[:, 1:, 1] = dihedral(coords[:, :-1, 2], coords[:, 1:, 0], coords[:, 1:, 1], coords[:, 1:, 2])
        mask[:, :-1, 0] = omega_valid
        mask[:, 1:, 1] = phi_valid
    psi_valid = valid[:, :, 0] & valid[:, :, 1] & valid[:, :, 2] & valid[:, :, 3]
    angles[:, :, 2] = -dihedral(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2], coords[:, :, 3])
    mask[:, :, 2] = psi_valid
    for aa_idx, chi_defs in enumerate(CHI_ATOM_NAMES):
        aa_positions = seq == aa_idx
        if not aa_positions.any():
            continue
        atom_names = ATOM14_NAMES[aa_idx]
        for chi_idx, chi_atoms in enumerate(chi_defs):
            if chi_atoms is None:
                continue
            atom_indices: list[int] = []
            for atom_name in chi_atoms:
                try:
                    atom_indices.append(atom_names.index(atom_name))
                except ValueError:
                    atom_indices = []
                    break
            if len(atom_indices) != 4:
                continue
            chi_angle = dihedral(*(coords[:, :, atom_index] for atom_index in atom_indices))
            chi_valid = aa_positions
            for atom_index in atom_indices:
                chi_valid = chi_valid & valid[:, :, atom_index]
            angles[:, :, 3 + chi_idx] = torch.where(chi_valid, chi_angle, angles[:, :, 3 + chi_idx])
            mask[:, :, 3 + chi_idx] = mask[:, :, 3 + chi_idx] | chi_valid
    encoded = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    encoded = torch.nan_to_num(encoded, nan=0.0)
    return encoded, mask


def make_alpha_t(template_coords: Tensor, seq: Tensor) -> Tensor:
    r"""
    Build the flattened ``alpha_t`` torsion tensor expected by RFdiffusion inference.

    Args:
        template_coords: Template coords `[B, T, L, A, 3]`.
        seq: Per-token residue ids broadcastable to `[B*T, L]`.

    Returns:
        The flattened torsion features `[B, T, L, 30]` (angles plus masks).
    """

    B, T, length = template_coords.shape[:3]
    seq_flat = seq.reshape(B * T, length) if seq.ndim == 3 else seq.reshape(B, length).repeat_interleave(T, dim=0)
    alpha, alpha_mask = torsion_features(template_coords.reshape(B * T, length, template_coords.shape[-2], 3), seq_flat)
    alpha = alpha.reshape(B, T, length, 10, 2)
    alpha_mask = alpha_mask.reshape(B, T, length, 10, 1).float()
    return torch.cat((alpha, alpha_mask), dim=-1).reshape(B, T, length, 30)


def _chain_ids_from_index(chain_index: Sequence[int]) -> list[str]:
    r"""Map integer chain indices to one-letter chain labels (A, B, ... a, b, ...)."""

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    unique = sorted({int(value) for value in chain_index})
    mapping = {value: letters[index % len(letters)] for index, value in enumerate(unique)}
    return [mapping[int(value)] for value in chain_index]


L_DEFAULT = 2000


def exp_map(rotation_vector: Tensor) -> Tensor:
    r"""
    Map rotation vectors to rotation matrices via the SO(3) exponential map.

    Args:
        rotation_vector: Rotation vectors `[..., 3]` (axis times angle).

    Returns:
        The corresponding rotation matrices `[..., 3, 3]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import exp_map
        >>> exp_map(torch.zeros(1, 3)).shape
        torch.Size([1, 3, 3])
    """

    device, dtype = rotation_vector.device, rotation_vector.dtype
    matrix = Rotation.from_rotvec(rotation_vector.detach().cpu().numpy()).as_matrix()
    return torch.as_tensor(matrix, dtype=dtype, device=device)


def f_igso3(omega_value: Tensor, t: float | Tensor, num_terms: int = L_DEFAULT) -> Tensor:
    r"""
    Evaluate the truncated IGSO3 density power series at rotation angle ``omega``.

    Args:
        omega_value: Rotation angles `[...]`.
        t: Diffusion time (variance) parameter.
        num_terms: Number of series terms.

    Returns:
        The IGSO3 density values `[...]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import f_igso3
        >>> f_igso3(torch.tensor([0.1, 0.2]), 0.5, num_terms=50).shape
        torch.Size([2])
    """

    omega_value = omega_value.float()
    t_tensor = torch.as_tensor(t, dtype=omega_value.dtype, device=omega_value.device)
    ell = torch.arange(num_terms, dtype=omega_value.dtype, device=omega_value.device)[None]
    angles = omega_value[..., None]
    denom = torch.sin(angles / 2).clamp_min(1e-12)
    terms = (2 * ell + 1) * torch.exp(-ell * (ell + 1) * t_tensor / 2)
    return (terms * torch.sin(angles * (ell + 0.5)) / denom).sum(dim=-1)


def d_logf_d_omega(omega_value: np.ndarray, t: float, num_terms: int = L_DEFAULT) -> np.ndarray:
    r"""
    Return the derivative of the log IGSO3 density with respect to the rotation angle (the score).

    Args:
        omega_value: Rotation angles at which to evaluate the score.
        t: Diffusion time (variance) parameter.
        num_terms: Number of density-series terms.

    Returns:
        The score ``d/d_omega log f`` evaluated at each ``omega_value``, same shape as the input.

    Examples:
        >>> import numpy as np
        >>> from ofoldx.models.rfdiffusion.model import d_logf_d_omega
        >>> d_logf_d_omega(np.array([0.5, 1.0]), 0.5).shape
        (2,)
    """

    with torch.inference_mode(False), torch.enable_grad():
        omega_t = torch.tensor(omega_value, dtype=torch.float32, requires_grad=True)
        log_f = torch.log(f_igso3(omega_t, t, num_terms=num_terms).clamp_min(1e-30))
        return torch.autograd.grad(log_f.sum(), omega_t)[0].detach().cpu().numpy()


def igso3_density_angle(omega_value: np.ndarray, t: float, num_terms: int = L_DEFAULT) -> np.ndarray:
    r"""
    Return the IGSO3 angular density (density times the SO(3) angle Haar factor) at ``omega``.

    Args:
        omega_value: Rotation angles at which to evaluate the density.
        t: Diffusion time (variance) parameter.
        num_terms: Number of density-series terms.

    Returns:
        The angular density evaluated at each ``omega_value``, same shape as the input.

    Examples:
        >>> import numpy as np
        >>> from ofoldx.models.rfdiffusion.model import igso3_density_angle
        >>> igso3_density_angle(np.array([0.5, 1.0]), 0.5).shape
        (2,)
    """

    omega_t = torch.tensor(omega_value, dtype=torch.float32)
    density = f_igso3(omega_t, t, num_terms=num_terms).detach().cpu().numpy()
    return density * (1 - np.cos(omega_value)) / np.pi


def calculate_igso3(
    *,
    num_sigma: int,
    num_omega: int,
    min_sigma: float,
    max_sigma: float,
    num_terms: int = L_DEFAULT,
) -> dict[str, np.ndarray]:
    r"""
    Precompute IGSO3 CDFs and score norms over a sigma/angle grid for inverse-CDF sampling.

    Args:
        num_sigma: Number of discretized sigma levels.
        num_omega: Number of discretized rotation angles.
        min_sigma: Minimum sigma.
        max_sigma: Maximum sigma.
        num_terms: Number of density-series terms.

    Returns:
        A dict with ``cdf``, ``score_norm``, ``exp_score_norms``, ``discrete_omega``, and ``discrete_sigma``.
    """

    discrete_omega = np.linspace(0, np.pi, num_omega + 1, dtype=np.float64)[1:]
    discrete_sigma = 10 ** np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma + 1, dtype=np.float64)[1:]
    pdf_vals = np.asarray(
        [igso3_density_angle(discrete_omega, float(sigma**2), num_terms=num_terms) for sigma in discrete_sigma]
    )
    cdf_vals = np.asarray([pdf.cumsum() / num_omega * np.pi for pdf in pdf_vals])
    score_norm = np.asarray(
        [d_logf_d_omega(discrete_omega, float(sigma**2), num_terms=num_terms) for sigma in discrete_sigma]
    )
    score_moment = np.sum(score_norm**2 * pdf_vals, axis=1) / np.sum(pdf_vals, axis=1)
    exp_score_norms = np.sqrt(np.clip(score_moment, a_min=0.0, a_max=None))
    return {
        "cdf": cdf_vals,
        "score_norm": score_norm,
        "exp_score_norms": exp_score_norms,
        "discrete_omega": discrete_omega,
        "discrete_sigma": discrete_sigma,
    }


def write_pickle(path: str | Path, value: object) -> None:
    r"""
    Pickle ``value`` to ``path``, creating parent directories as needed.

    Args:
        path: Destination file path; missing parent directories are created.
        value: The object to serialize.

    Examples:
        >>> from ofoldx.models.rfdiffusion.model import write_pickle
        >>> write_pickle("igso3_cache.pkl", {"cdf": 0.0})  # doctest: +SKIP
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_rotation_x(angs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Build 4x4 homogeneous rotation matrices about the x-axis from ``(cos, sin)`` angle pairs `[B, L, 2]`.

    Args:
        angs: Per-position ``(cos, sin)`` angle pairs `[B, L, 2]`.
        eps: Floor added to the pair norm to avoid division by zero.

    Returns:
        Homogeneous rotation matrices `[B, L, 4, 4]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import make_rotation_x
        >>> make_rotation_x(torch.ones(1, 2, 2)).shape
        torch.Size([1, 2, 4, 4])
    """

    batch_size, length = angs.shape[:2]
    norm = torch.linalg.norm(angs, dim=-1) + eps
    rotations = torch.eye(4, device=angs.device, dtype=angs.dtype).repeat(batch_size, length, 1, 1)
    rotations[:, :, 1, 1] = angs[:, :, 0] / norm
    rotations[:, :, 1, 2] = -angs[:, :, 1] / norm
    rotations[:, :, 2, 1] = angs[:, :, 1] / norm
    rotations[:, :, 2, 2] = angs[:, :, 0] / norm
    return rotations


def make_rotation_z(angs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Build 4x4 homogeneous rotation matrices about the z-axis from ``(cos, sin)`` angle pairs `[B, L, 2]`.

    Args:
        angs: Per-position ``(cos, sin)`` angle pairs `[B, L, 2]`.
        eps: Floor added to the pair norm to avoid division by zero.

    Returns:
        Homogeneous rotation matrices `[B, L, 4, 4]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import make_rotation_z
        >>> make_rotation_z(torch.ones(1, 2, 2)).shape
        torch.Size([1, 2, 4, 4])
    """

    batch_size, length = angs.shape[:2]
    norm = torch.linalg.norm(angs, dim=-1) + eps
    rotations = torch.eye(4, device=angs.device, dtype=angs.dtype).repeat(batch_size, length, 1, 1)
    rotations[:, :, 0, 0] = angs[:, :, 0] / norm
    rotations[:, :, 0, 1] = -angs[:, :, 1] / norm
    rotations[:, :, 1, 0] = angs[:, :, 1] / norm
    rotations[:, :, 1, 1] = angs[:, :, 0] / norm
    return rotations


def make_rot_axis(angs: torch.Tensor, axis: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Build 4x4 homogeneous rotations about an arbitrary ``axis`` from ``(cos, sin)`` angle pairs (Rodrigues).

    Args:
        angs: Per-position ``(cos, sin)`` angle pairs `[B, L, 2]`.
        axis: Rotation axis per position `[B, L, 3]`; normalized internally.
        eps: Floor on the pair and axis norms to avoid division by zero.

    Returns:
        Homogeneous rotation matrices `[B, L, 4, 4]`.

    Examples:
        >>> import torch
        >>> from ofoldx.models.rfdiffusion.model import make_rot_axis
        >>> axis = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        >>> make_rot_axis(torch.ones(1, 2, 2), axis).shape
        torch.Size([1, 2, 4, 4])
    """

    batch_size, length = angs.shape[:2]
    norm = torch.linalg.norm(angs, dim=-1).clamp_min(eps)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    cos_theta = angs[:, :, 0] / norm
    sin_theta = angs[:, :, 1] / norm
    u0, u1, u2 = axis.unbind(dim=-1)

    rotations = torch.eye(4, device=angs.device, dtype=angs.dtype).repeat(batch_size, length, 1, 1)
    one_minus_cos = 1 - cos_theta
    rotations[:, :, 0, 0] = cos_theta + u0 * u0 * one_minus_cos
    rotations[:, :, 0, 1] = u0 * u1 * one_minus_cos - u2 * sin_theta
    rotations[:, :, 0, 2] = u0 * u2 * one_minus_cos + u1 * sin_theta
    rotations[:, :, 1, 0] = u0 * u1 * one_minus_cos + u2 * sin_theta
    rotations[:, :, 1, 1] = cos_theta + u1 * u1 * one_minus_cos
    rotations[:, :, 1, 2] = u1 * u2 * one_minus_cos - u0 * sin_theta
    rotations[:, :, 2, 0] = u0 * u2 * one_minus_cos - u1 * sin_theta
    rotations[:, :, 2, 1] = u1 * u2 * one_minus_cos + u0 * sin_theta
    rotations[:, :, 2, 2] = cos_theta + u2 * u2 * one_minus_cos
    return rotations


class ComputeAllAtomCoords(nn.Module):
    r"""
    Reconstruct RFdiffusion atom14 coords from backbone frames and predicted torsion angles.

    Composes the backbone frame with the per-torsion rigid transforms and torsion rotations from the embedded
    RoseTTAFold all-atom kinematics tables to place every sidechain atom. Holds those tables as non-persistent
    buffers.
    """

    def __init__(self) -> None:
        super().__init__()
        base_indices, rts_by_torsion, xyzs_in_base_frame = rf_all_atom_kinematics()
        self.register_buffer("base_indices", base_indices, persistent=False)
        self.register_buffer("RTs_in_base_frame", rts_by_torsion, persistent=False)
        self.register_buffer("xyzs_in_base_frame", xyzs_in_base_frame, persistent=False)

    def forward(
        self,
        seq: torch.Tensor,
        coords: torch.Tensor,
        alphas: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Reconstruct atom14 coords from backbone frames and torsions.

        Args:
            seq: Per-token residue ids `[B, L]`.
            coords: Backbone coords `[B, L, 3, 3]` (N/CA/C).
            alphas: Torsion angles `[B, L, 10, 2]`; defaults to the identity torsion when `None`.

        Returns:
            A tuple ``(frames, coords_out)`` of the per-torsion frames and the atom14 coords `[B, L, 14, 3]`.

        Raises:
            ValueError: If ``coords`` is not `(B, L, 3, 3)`.
        """

        if coords.ndim != 4 or coords.shape[-2:] != (3, 3):
            raise ValueError(f"backbone prediction must have shape (B, L, 3, 3), got {tuple(coords.shape)}")
        seq = seq.long().clamp(0, MASK_TOKEN)
        batch_size, length = coords.shape[:2]
        if alphas is None:
            alphas = torch.zeros((batch_size, length, 10, 2), dtype=coords.dtype, device=coords.device)
            alphas[..., 0] = 1.0
        alphas = alphas.to(device=coords.device, dtype=coords.dtype)

        rotations, translations = rigid_from_3_points(coords[..., 0, :], coords[..., 1, :], coords[..., 2, :])
        rtf0 = torch.eye(4, dtype=coords.dtype, device=coords.device).repeat(batch_size, length, 1, 1)
        rtf0[:, :, :3, :3] = rotations
        rtf0[:, :, :3, 3] = translations

        rts = self.RTs_in_base_frame.to(device=coords.device, dtype=coords.dtype)[seq]
        rtf1 = torch.einsum("blij,bljk,blkm->blim", rtf0, rts[:, :, 0], make_rotation_x(alphas[:, :, 0]))
        rtf2 = torch.einsum("blij,bljk,blkm->blim", rtf0, rts[:, :, 1], make_rotation_x(alphas[:, :, 1]))
        rtf3 = torch.einsum("blij,bljk,blkm->blim", rtf0, rts[:, :, 2], make_rotation_x(alphas[:, :, 2]))

        base_coords = self.xyzs_in_base_frame.to(device=coords.device, dtype=coords.dtype)[seq]
        nc_midpoint = 0.5 * (base_coords[:, :, 2, :3] + base_coords[:, :, 0, :3])
        ca_ref = base_coords[:, :, 1, :3]
        cb_ref = base_coords[:, :, 4, :3]
        cb_axis_1 = torch.cross(cb_ref - ca_ref, nc_midpoint - ca_ref, dim=-1)
        cb_axis_1 = cb_axis_1 / torch.linalg.norm(cb_axis_1, dim=-1, keepdim=True).clamp_min(1e-8)

        nc_projection = base_coords[:, :, 2, :3] - base_coords[:, :, 0, :3]
        nc_projection = nc_projection - (
            torch.sum(nc_projection * nc_midpoint, dim=-1, keepdim=True)
            / torch.sum(nc_midpoint * nc_midpoint, dim=-1, keepdim=True).clamp_min(1e-8)
            * nc_midpoint
        )
        cb_axis_2 = torch.cross(cb_ref - ca_ref, nc_projection, dim=-1)
        cb_axis_2 = cb_axis_2 / torch.linalg.norm(cb_axis_2, dim=-1, keepdim=True).clamp_min(1e-8)
        cb_rot_1 = make_rot_axis(alphas[:, :, 7], cb_axis_1)
        cb_rot_2 = make_rot_axis(alphas[:, :, 8], cb_axis_2)
        rtf8 = torch.einsum("blij,bljk,blkm->blim", rtf0, cb_rot_1, cb_rot_2)

        rtf4 = torch.einsum(
            "blij,bljk,blkm,blmn->blin",
            rtf8,
            rts[:, :, 3],
            make_rotation_x(alphas[:, :, 3]),
            make_rotation_z(alphas[:, :, 9]),
        )
        rtf5 = torch.einsum("blij,bljk,blkm->blim", rtf4, rts[:, :, 4], make_rotation_x(alphas[:, :, 4]))
        rtf6 = torch.einsum("blij,bljk,blkm->blim", rtf5, rts[:, :, 5], make_rotation_x(alphas[:, :, 5]))
        rtf7 = torch.einsum("blij,bljk,blkm->blim", rtf6, rts[:, :, 6], make_rotation_x(alphas[:, :, 6]))
        frames = torch.stack((rtf0, rtf1, rtf2, rtf3, rtf4, rtf5, rtf6, rtf7, rtf8), dim=2)

        gather_index = self.base_indices.to(device=coords.device)[seq][..., None, None].repeat(1, 1, 1, 4, 4)
        atom_frames = frames.gather(2, gather_index)
        coords_out = torch.einsum("bltij,bltj->blti", atom_frames, base_coords)
        return frames, coords_out[..., :14, :3]


@dataclass(slots=True)
class RfDiffusionStepFeatures:
    r"""
    One assembled bundle of RoseTTAFold network input tensors for a single timestep.

    Attributes:
        msa_masked: Masked MSA features `[1, 1, L, 48]`.
        msa_full: Full MSA features `[1, 1, L, 25]`.
        seq_in: Sequence features `[1, L, 22]`.
        noisy_coords_in: Noised template coords fed to the network `[1, L, A, 3]`.
        pdb_index: RF residue-index tensor `[1, L]`.
        template_1d: 1D template features.
        template_2d: 2D template features.
        template_coords: Template backbone coords.
        alpha_t: Template torsion features.
    """

    msa_masked: Tensor
    msa_full: Tensor
    seq_in: Tensor
    noisy_coords_in: Tensor
    pdb_index: Tensor
    template_1d: Tensor
    template_2d: Tensor
    template_coords: Tensor
    alpha_t: Tensor


@dataclass(slots=True)
class RfDiffusionSampleState:
    r"""
    Mutable per-trajectory state carried across RFdiffusion denoising steps.

    Attributes:
        noisy_coords: Current noised backbone coords.
        seq_t: Current sequence features.
        contig_map: The resolved [`ContigMap`][ofoldx.models.rfdiffusion.model.ContigMap].
        mask_seq: Sequence-inpainting mask.
        mask_str: Structure-inpainting mask.
        diffusion_mask: Per-residue mask marking fixed (motif) residues.
        binderlen: Number of binder residues.
        chain_idx: Per-residue output chain labels.
        pdb_index: Per-residue output PDB residue numbers.
        t_step_input: The starting timestep.
        diffuser: The forward [`Diffuser`][ofoldx.models.rfdiffusion.model.Diffuser].
        denoiser: The reverse [`Denoiser`][ofoldx.models.rfdiffusion.model.Denoiser].
        potential_manager: The guiding-potential manager.
        symmetry: Optional symmetry generator.
        cyclic_reses: Optional cyclic-residue mask.
        ss: Optional scaffold secondary-structure tensor.
        adj: Optional scaffold block-adjacency tensor.
        target_ss: Optional target secondary-structure tensor.
        target_adj: Optional target block-adjacency tensor.
        target_pdb: Optional fixed-target feature dict.
        prev_pred: Previous-step prediction for self-conditioning.
    """

    noisy_coords: Tensor
    seq_t: Tensor
    contig_map: ContigMap
    mask_seq: Tensor
    mask_str: Tensor
    diffusion_mask: Tensor
    binderlen: int
    chain_idx: list[str]
    pdb_index: list[int]
    t_step_input: int
    diffuser: Diffuser
    denoiser: Denoiser
    potential_manager: PotentialManager
    symmetry: SymmetryGenerator | None = None
    cyclic_reses: Tensor | None = None
    ss: Tensor | None = None
    adj: Tensor | None = None
    target_ss: Tensor | None = None
    target_adj: Tensor | None = None
    target_pdb: dict[str, Any] | None = None
    prev_pred: Tensor | None = None


@dataclass(slots=True)
class RfDiffusionSampleResult:
    r"""
    Output tensors and diagnostics from one completed RFdiffusion trajectory.

    Attributes:
        coords: Final designed backbone coords.
        sequence_tokens: Final designed sequence token ids.
        plddt: Final per-residue pLDDT, or `None`.
        trajectory: The stacked noised-coord trajectory over all steps.
        pred_coords_trajectory: The stacked predicted-coord trajectory over all steps.
        metadata: Run metadata (architecture, contig, sequence, timesteps, ...).
    """

    coords: Tensor
    sequence_tokens: Tensor
    plddt: Tensor | None
    trajectory: Tensor
    pred_coords_trajectory: Tensor
    metadata: dict[str, Any]
