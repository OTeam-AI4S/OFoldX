r"""RFdiffusion denoiser, sampler, and public scientific modules."""

from __future__ import annotations

import math
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from operator import index as integer_index
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from torch import Tensor, einsum, nn
from typing_extensions import Self

# The family-owned RFdiffusion spatial/placement condition records (RFdiffusionPlacements,
# Coordination, Distance, RadiusOfGyration, ReferenceCoordinates, SecondaryStructure,
# SubstrateInteraction, and PointGroupSymmetry) are defined below
# in this module (folded in from the former .conditions submodule). The frozen §3.7 placement
# records and constraint bond they build on are imported from .data.bonds / .data.placement.
from ...data.bonds import SymbolicBond
from ...data.components import get_component

# AllowedResidueIdentities / Contact / Hotspot stay shared: the frozen §3.7.1 identity record and
# the frozen §3.7.4 constraint records.
from ...data.conditions import AllowedResidueIdentities, Contact, Hotspot
from ...data.evidence import Template
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    SystemBond,
)
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    InstanceIdAllocator,
    PlacementMap,
)
from ...data.placement import PlacementSpec as PlacementMapping
from ...data.placement import SymbolicAtomSite, SymbolicResidueSite
from ...data.structure import Structure
from ...modules.backend._rfdiffusion import (
    _select_backend,
    se3_basis,
    segment_softmax_csr,
    segment_sum_csr,
    weighted_tensor_product,
)
from ...results import Metric, Score
from .chemistry import (
    ATOM14_NAMES,
    CHI_ATOM_NAMES,
    INIT_CRDS,
    MASK_TOKEN,
    NUM2AA,
    residue_token,
    rf_all_atom_kinematics,
)
from .config import RFdiffusionConfig

# The duplicated family string aliases ``ResidueReference`` / ``AtomReference`` are dropped in
# favor of the frozen reference vocabulary: an atom-granularity endpoint is
# :data:`~ofoldx.data.bonds.SymbolicBondSite` (== :data:`~ofoldx.data.conditions.AtomEndpoint`,
# ``AtomSite | SymbolicAtomSite``); a residue-granularity endpoint is
# :data:`~ofoldx.data.conditions.ResidueEndpoint` (``ComponentReference | str | SymbolicResidueSite``).

# Fixed dimensions of the released RFdiffusion IGSO(3) lookup table (num_sigma x num_omega). The
# released table is un-regenerable numerical noise, so it is baked into ``model.safetensors`` as
# persistent float64 buffers rather than recomputed; these shapes size those buffers.
_IGSO3_NUM_SIGMA = 500
_IGSO3_NUM_OMEGA = 1000
_IGSO3_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("cdf", (_IGSO3_NUM_SIGMA, _IGSO3_NUM_OMEGA)),
    ("score_norm", (_IGSO3_NUM_SIGMA, _IGSO3_NUM_OMEGA)),
    ("exp_score_norms", (_IGSO3_NUM_SIGMA,)),
    ("discrete_omega", (_IGSO3_NUM_OMEGA,)),
    ("discrete_sigma", (_IGSO3_NUM_SIGMA,)),
)


class RFdiffusionModel(nn.Module):
    r"""The released RFdiffusion RoseTTAFold denoiser and design sampler.

    ``forward`` always means one differentiable denoising pass.  ``design`` is the
    explicit reverse-diffusion task method and returns canonical Structures.
    """

    config_class = RFdiffusionConfig

    def __init__(self, config: RFdiffusionConfig | None = None) -> None:
        config = RFdiffusionConfig() if config is None else config
        super().__init__()
        self.config = config
        network, preprocess = config.network, config.preprocess
        self.network = RoseTTAFoldModule(
            num_extra_blocks=network.num_extra_blocks,
            num_main_blocks=network.num_main_blocks,
            num_refinement_blocks=network.num_refinement_blocks,
            msa_size=network.msa_size,
            full_msa_size=network.full_msa_size,
            pair_size=network.pair_size,
            template_size=network.template_size,
            num_msa_attention_heads=network.num_msa_attention_heads,
            num_pair_attention_heads=network.num_pair_attention_heads,
            num_template_attention_heads=network.num_template_attention_heads,
            hidden_size=network.hidden_size,
            template_hidden_size=network.template_hidden_size,
            dropout=network.dropout,
            template_1d_size=preprocess.template_1d_size,
            template_2d_size=preprocess.template_2d_size,
            freeze_track_motif=network.freeze_track_motif,
            se3_param_full=network.se3_param_full,
            se3_param_topk=network.se3_param_topk,
            vocab_size=network.vocab_size,
        )
        self.num_timesteps = config.diffuser.num_timesteps
        self.template_1d_size = preprocess.template_1d_size
        self.template_2d_size = preprocess.template_2d_size
        self.sidechain_input = preprocess.sidechain_input
        self.structure_self_conditioning = preprocess.structure_self_conditioning
        self.allatom = ComputeAllAtomCoords()
        # The released IGSO(3) schedule table travels with the checkpoint as persistent float64
        # buffers (baked into ``model.safetensors`` at convert time, restored by strict
        # ``load_state_dict``) instead of vendored package data. ``design`` threads them into the
        # sampler so the exact released table is used verbatim, without any disk read.
        for name, shape in _IGSO3_FIELDS:
            self.register_buffer(f"igso3_{name}", torch.zeros(shape, dtype=torch.float64), persistent=True)
        self.set_backend("auto")

    def _igso3_schedule(self) -> dict[str, np.ndarray]:
        r"""Return the model-baked IGSO(3) lookup table as the sampler's schedule dict."""

        return {name: getattr(self, f"igso3_{name}").detach().cpu().numpy() for name, _ in _IGSO3_FIELDS}

    def set_backend(self, backend: str) -> Self:
        r"""Select the operator backend used by the RFdiffusion SE(3) path."""

        if backend not in {"auto", "torch", "oops"}:
            raise ValueError("backend must be exactly one of: auto, torch, oops")
        self.backend = backend
        for module in self.modules():
            if isinstance(module, VersatileConvSE3 | ConvSE3 | AttentionSE3 | SE3Transformer):
                module.backend = backend
        return self

    def forward(self, batch: RFdiffusionInput) -> RFdiffusionOutput:
        r"""Run one ordinary differentiable denoising pass."""

        msa_masked, msa_full, template_1d, template_2d, template_coords, alpha_t = self._features(batch)
        output = self.network(
            msa_masked,
            msa_full,
            batch.sequence,
            batch.noisy_coords,
            batch.residue_index,
            template_1d=template_1d,
            template_2d=template_2d,
            template_coords=template_coords,
            alpha_t=alpha_t,
            msa_prev=None,
            pair_prev=None,
            state_prev=None,
            motif_mask=batch.fixed_structure_mask,
            cyclic_reses=batch.cyclic_mask,
        )
        return RFdiffusionOutput(*output)

    def _features(
        self,
        batch: RFdiffusionInput,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Build the exact released MSA/template feature channels."""

        sequence = batch.sequence
        batch_size, length = sequence.shape[:2]
        msa_masked = sequence.new_zeros((batch_size, 1, length, 48))
        msa_masked[..., :22] = sequence[:, None]
        msa_masked[..., 22:44] = sequence[:, None]
        msa_masked[:, :, 0, 46] = 1
        msa_masked[:, :, -1, 47] = 1
        msa_full = sequence.new_zeros((batch_size, 1, length, 25))
        msa_full[..., :22] = sequence[:, None]
        msa_full[:, :, 0, 23] = 1
        msa_full[:, :, -1, 24] = 1

        template_1d = sequence.new_zeros((batch_size, 1, length, 21))
        visible_sequence = sequence[..., :21].clone()
        masked = sequence[..., MASK_TOKEN] > 0.5
        visible_sequence[..., 20] = visible_sequence[..., 20].masked_fill(masked, 1)
        template_1d[..., :21] = visible_sequence[:, None]
        time = 1 - batch.timestep.to(sequence.dtype) / float(self.num_timesteps)
        time = (
            time[:, None]
            .expand_as(batch.fixed_structure_mask)
            .masked_fill(
                batch.fixed_structure_mask,
                1,
            )
        )
        template_1d = torch.cat((template_1d, time[:, None, :, None]), dim=-1)
        if self.template_1d_size >= 24:
            template_1d = torch.cat(
                (
                    template_1d,
                    template_1d.new_zeros((batch_size, 1, length, 1)),
                    batch.hotspot_mask[:, None, :, None].to(sequence.dtype),
                ),
                dim=-1,
            )
        if self.template_1d_size == 28:
            template_1d = torch.cat((template_1d, batch.secondary_structure[:, None]), dim=-1)

        coords = batch.noisy_coords.clone()
        if self.sidechain_input:
            coords[:, :, 3:] = coords[:, :, 3:].masked_fill(masked[:, :, None, None], float("nan"))
        else:
            coords[:, :, 3:] = coords[:, :, 3:].masked_fill(
                (~batch.fixed_structure_mask)[:, :, None, None],
                float("nan"),
            )
        template_coords = torch.cat(
            (coords[:, None], coords.new_full((batch_size, 1, length, 13, 3), float("nan"))),
            dim=3,
        )
        template_2d = coords_to_t2d(template_coords)
        alpha_t = make_alpha_t(
            template_coords,
            template_1d[..., :21].argmax(dim=-1).reshape(batch_size, length),
        )
        if self.structure_self_conditioning:
            template_2d[..., :44] = 0
            template_coords = torch.zeros_like(template_coords)
        if self.template_2d_size == 47:
            template_2d = torch.cat((template_2d, batch.block_adjacency[:, None]), dim=-1)
        return msa_masked, msa_full, template_1d, template_2d, template_coords, alpha_t

    @torch.inference_mode()
    def design(
        self,
        batch: RFdiffusionInput,
        *,
        num_candidates: int = 1,
        num_sampling_steps: int | None = None,
        final_step: int = 1,
        partial_steps: int | None = None,
        noise_scale_ca: float = 1.0,
        final_noise_scale_ca: float | None = None,
        ca_noise_schedule: Literal["constant", "linear"] = "constant",
        noise_scale_frame: float = 1.0,
        final_noise_scale_frame: float | None = None,
        frame_noise_schedule: Literal["constant", "linear"] = "constant",
        align_motif: bool = True,
        return_trajectory: bool = False,
        symmetry_recenter: bool = True,
        symmetry_radius_angstrom: float = 10.0,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
        generator: torch.Generator | None = None,
    ) -> RFdiffusionDesignResult:
        r"""Generate canonical backbone Structures, sampling variable placement lengths per candidate.

        The sampler follows the released raw final-``pX0`` contract. Canonical motif
        alignment and fixed-row restoration happen while decoding the Result. Full
        trajectories are opt-in and stored on CPU. Distinct final noise values are
        valid only with linear schedules. The caller-supplied ``generator`` (resolved once
        at the Pipeline boundary) is the sole random stream; per-step reverse-diffusion
        resampling remains model-intrinsic.
        """

        if num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer")
        if batch.sequence.shape[0] != 1:
            raise ValueError("RFdiffusion design accepts one canonical system at a time")
        prepared = [_sample_variable_lengths(batch, generator) for _ in range(num_candidates)]
        sampler = RFdiffusionSampler(
            self.config,
            network=self.network,
            allatom=self.allatom,
            igso3_schedule=self._igso3_schedule(),
            final_step=final_step,
            partial_steps=partial_steps,
            noise_scale_ca=noise_scale_ca,
            final_noise_scale_ca=final_noise_scale_ca,
            ca_noise_schedule=ca_noise_schedule,
            noise_scale_frame=noise_scale_frame,
            final_noise_scale_frame=final_noise_scale_frame,
            frame_noise_schedule=frame_noise_schedule,
            align_motif=align_motif,
            symmetry_recenter=symmetry_recenter,
            symmetry_radius_angstrom=symmetry_radius_angstrom,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
        )
        return _decode_design(
            (
                sampler.sample(
                    candidate_input,
                    num_sampling_steps=num_sampling_steps,
                    generator=generator,
                    return_trajectory=return_trajectory,
                )
                for candidate_input in prepared
            ),
            prepared,
            align_motif=align_motif,
        )

    def _resolved_backend(self, tensor: Tensor) -> str:
        return _select_backend(self.backend, tensor)


def rfdiffusion_schedule(source: str | PathLike[str] | None = None) -> dict[str, np.ndarray]:
    r"""Load the exact released RFdiffusion IGSO(3) lookup table from a convert-time source.

    This is a convert-stage loader only: the released table is baked into ``model.safetensors`` as
    the model's persistent ``igso3_*`` buffers, so nothing reads it at inference time. ``source`` is
    required (the table is NOT shipped inside the installed package -- it is a Hub release artifact):
    pass the original RFdiffusion IGSO(3) ``.pkl`` cache, or a ``.npz`` archive of the same table; a
    ``.pkl`` source is read as an original RFdiffusion IGSO(3) cache pickle and any other suffix as
    the safe NumPy archive. The reference archive is derived from
    ``schedules/T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl`` at
    RosettaCommons/RFdiffusion revision ``2d0c003df46b9db41d119321f15403dec3716cd9`` (source
    SHA-256 ``cd79d3b77b3dc80d727ae0295f932b88cdfc0e5805e0ced29ce3c719518a1e23``) and is
    redistributed under RFdiffusion's BSD 3-Clause license.
    """

    if source is None:
        raise ValueError(
            "the IGSO(3) schedule source is required: it is baked into converted checkpoints as the "
            "model's igso3_* buffers and is not shipped in the installed package. Pass the original "
            "RFdiffusion IGSO(3) .pkl cache (or an .npz archive of the same table) via schedule_source "
            "(the convert CLI's --schedule-source)."
        )
    path = Path(source)
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            values = pickle.load(handle)
        return {name: np.asarray(array) for name, array in values.items()}
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


class RoseTTAFoldModule(nn.Module):
    r"""
    RoseTTAFold trunk driving one RFdiffusion denoising step.

    Ported RoseTTAFold network: embeds the MSA/pair/state representations, injects template and recycling
    features, runs the [`IterativeSimulator`][ofoldx.models.rfdiffusion.model.IterativeSimulator] track,
    and reads out amino-acid logits, pLDDT, and the denoised backbone. The
    [`RFdiffusionSampler`][ofoldx.models.rfdiffusion.model.RFdiffusionSampler] calls this
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
        self.coords_6d_pred = DistanceNetwork(pair_size)
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
            The final denoising representations and predictions followed by the released geometry,
            pLDDT, resolved-residue, backbone-trajectory, and torsion-trajectory training outputs.

        Raises:
            ValueError: If any required template feature is `None`.
        """

        batch_size, _, length = msa_latent.shape[:3]
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, cyclic_reses)
        msa_full = self.full_emb(msa_full, seq)

        if msa_prev is None:
            msa_prev = torch.zeros_like(msa_latent[:, 0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)
        msa_recycle, pair_recycle, state_recycle = self.recycle(msa_prev, pair_prev, coords, state_prev)
        msa_latent[:, 0] = msa_latent[:, 0] + msa_recycle.reshape(batch_size, length, -1)
        pair = pair + pair_recycle
        state = state + state_recycle

        # Time features are injected by the sampler's preprocess(); no timestep_embedder here.
        pair, state = self.templ_emb(template_1d, template_2d, alpha_t, template_coords, pair, state)
        if motif_mask is None:
            motif_mask = torch.zeros(
                (batch_size, length),
                dtype=torch.bool,
                device=msa_latent.device,
            )
        elif motif_mask.ndim == 1:
            motif_mask = motif_mask.unsqueeze(0).expand(batch_size, -1)
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
        backbone_trajectory = torch.einsum("sblij,blaj->sblai", rotations, local)
        backbone_trajectory = backbone_trajectory + translations.unsqueeze(-2)
        final_coords = backbone_trajectory[-1]

        distance_logits = self.coords_6d_pred(pair)
        logits_aa = self.aa_pred(msa)
        lddt_logits = self.lddt_pred(state)
        resolved_logits = self.exp_pred(msa[:, 0], state)
        return (
            msa[:, 0],
            pair,
            final_coords,
            state,
            alpha_s[-1],
            logits_aa.permute(0, 2, 1),
            _plddt_expectation(lddt_logits),
            distance_logits,
            lddt_logits,
            resolved_logits,
            backbone_trajectory,
            alpha_s,
        )


def _plddt_expectation(logits: Tensor) -> Tensor:
    r"""Convert per-bin pLDDT logits `[B, num_bins, L]` into the expected pLDDT value `[B, L]`."""

    num_bins = logits.shape[1]
    bins = torch.linspace(1.0 / num_bins, 1.0, num_bins, dtype=logits.dtype, device=logits.device)
    return torch.sum(torch.softmax(logits, dim=1) * bins[None, :, None], dim=1)


def _chain_ids(idx: torch.Tensor, threshold: int = 35) -> torch.Tensor:
    r"""Assign a chain id per residue by detecting large jumps (> ``threshold``) in the residue index."""

    starts = (idx[:, 1:] - idx[:, :-1] > threshold).to(torch.int64)
    return torch.cat((torch.zeros_like(idx[:, :1]), starts), dim=1).cumsum(dim=1)


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

        if cyclic_residues is not None:
            chain_ids = _chain_ids(idx)
            same_chain = chain_ids[:, :, None] == chain_ids[:, None, :]
            same_cycle = same_chain & cyclic_residues[:, :, None] & cyclic_residues[:, None, :]
            cycle_length = (same_chain & cyclic_residues[:, None, :]).sum(dim=-1)[:, :, None]
            half = cycle_length // 2
            seqsep = torch.where(same_cycle & (seqsep > half), seqsep - cycle_length, seqsep)
            seqsep = torch.where(same_cycle & (seqsep < -half), seqsep + cycle_length, seqsep)

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

    def forward(self, msa: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        r"""
        Embed the extra MSA.

        Args:
            msa: Raw extra-MSA features `[B, N_full, L, input_feature_size]`.
            seq: One-hot sequence features `[B, L, 22]` used for the query embedding.

        Returns:
            The embedded extra MSA `[B, N_full, L, msa_size]`.
        """

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
        msa: torch.Tensor,
        pair: torch.Tensor,
        coords: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Build the recycled MSA, pair, and state contributions.

        Args:
            msa: Previous-iteration MSA row `[B, L, msa_size]`.
            pair: Previous-iteration pair representation `[B, L, L, pair_size]`.
            coords: Previous-iteration backbone coords `[B, L, A, 3]`.
            state: Previous-iteration node state `[B, L, state_size]`.

        Returns:
            A tuple ``(msa, pair, state)`` of the recycled contributions to add back into the trunk inputs.
        """

        _, length = pair.shape[:2]
        state = self.norm_state(state)
        left_state = state.unsqueeze(2).expand(-1, -1, length, -1)
        right_state = state.unsqueeze(1).expand(-1, length, -1, -1)

        cbeta = generate_cbeta(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2])
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
    translations = coords[:, :, 1]
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
        self.row_attn = BiasedAxialAttention(pair_size, pair_size, num_attention_heads, hidden_size, is_row=True)
        self.col_attn = BiasedAxialAttention(pair_size, pair_size, num_attention_heads, hidden_size, is_row=False)
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
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        hidden_size: int = 32,
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
    """

    def __init__(
        self,
        msa_size: int = 256,
        state_size: int = 32,
        hidden_size: int = 128,
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
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        state_size: int = 16,
        se3_param: Mapping[str, int] | None = None,
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
        self.sc_predictor = SCPred(msa_size=msa_size, state_size=l0_out_features)
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

        Returns:
            A tuple ``(rotations, translations, state, alpha)`` of the updated frames, node state, and
            predicted torsion angles.
        """

        batch_size, _, length = msa.shape[:3]
        motif_mask = torch.zeros_like(idx, dtype=torch.bool) if motif_mask is None else motif_mask
        if motif_mask.ndim == 1:
            motif_mask = motif_mask.unsqueeze(0).expand(batch_size, -1)

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
        offset = offset.masked_fill(motif_mask[:, :, None, None], 0)

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
        self.msa2pair = MSA2Pair(msa_size=msa_size, pair_size=pair_size, hidden_size=hidden_size // 2)
        self.pair2pair = PairStr2Pair(
            pair_size=pair_size, num_attention_heads=num_pair_attention_heads, hidden_size=hidden_size, dropout=dropout
        )
        self.str2str = Str2Str(
            msa_size=msa_size,
            pair_size=pair_size,
            state_size=state_size,
            se3_param=se3_param,
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

        motif_mask = torch.zeros_like(idx, dtype=torch.bool) if motif_mask is None else motif_mask
        if motif_mask.ndim == 1:
            motif_mask = motif_mask.unsqueeze(0).expand(idx.shape[0], -1)
        rotations, translations, local_coords = _initial_frames(coords_in)
        state = self.proj_state(state)

        rotation_steps: list[Tensor] = []
        translation_steps: list[Tensor] = []
        alpha_steps: list[Tensor] = []

        for block in self.extra_block:
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

        for block in self.main_block:
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
        is_row: Whether to attend along rows (`True`) or columns (`False`).
    """

    def __init__(
        self,
        pair_size: int,
        bias_size: int,
        num_attention_heads: int,
        head_dim: int,
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


class Fiber(dict[int, int]):
    r"""
    Degree-to-channel structure of an SE(3)-equivariant feature space.

    A dict mapping each representation degree to its channel count, with helpers to enumerate degrees,
    channels, and the total flattened feature width. Ported from the NVIDIA SE(3)-Transformer.

    Args:
        structure: A mapping or iterable of ``(degree, channels)`` pairs.
    """

    def __init__(self, structure: Mapping[int, int] | Iterable[tuple[int, int]]) -> None:
        super().__init__(sorted(dict(structure).items()))

    def __missing__(self, degree: int) -> int:
        r"""Return zero channels for a degree absent from the fiber."""

        return 0

    @property
    def degrees(self) -> list[int]:
        r"""Sorted list of representation degrees present in the fiber."""

        return sorted(self)

    @property
    def channels(self) -> list[int]:
        r"""Channel counts, one per degree in `degrees` order."""

        return [self[degree] for degree in self.degrees]

    @property
    def num_features(self) -> int:
        r"""Total flattened feature width, summing ``channels * degree_to_dim(degree)`` over all degrees."""

        return sum(channels * degree_to_dim(degree) for degree, channels in self.items())

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

        return Fiber((degree, num_channels) for degree in range(num_degrees))

    def __add__(self, other):
        r"""Add two fibers channel-wise per degree, or add an int to every degree's channel count."""

        if isinstance(other, Fiber):
            return Fiber({degree: channels + other[degree] for degree, channels in self.items()})
        if isinstance(other, int):
            return Fiber({degree: channels + other for degree, channels in self.items()})
        return NotImplemented


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
        self.backend = "auto"
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
        return weighted_tensor_product(features, basis, radial_weights, backend=self.backend)


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
                    torch.randn(channels_out, fiber_in[degree_out]) / math.sqrt(fiber_in[degree_out])
                )
                for degree_out, channels_out in fiber_out.items()
            }
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        r"""
        Apply the per-degree channel mix.

        Args:
            features: Per-degree feature dict keyed by string degree.

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

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        r"""
        Apply the equivariant norm-nonlinearity.

        Args:
            features: Per-degree feature dict keyed by string degree.

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
        self.backend = "auto"
        channels_in_set = {channels + fiber_edge[degree] * (degree > 0) for degree, channels in self.fiber_in.items()}
        channels_out_set = set(self.fiber_out.values())
        unique_channels_in = len(channels_in_set) == 1
        degrees_up_to_max = list(range(max_degree + 1))
        edge_dim = fiber_edge[0] + 1
        self.fused_output = unique_channels_in and fiber_in.degrees == degrees_up_to_max
        self.conv_out = nn.ModuleDict()
        self.conv_in = nn.ModuleDict()

        if self.fused_output:
            for degree_out, channels_out in fiber_out.items():
                sum_freq = sum(degree_to_dim(min(degree_out, degree)) for degree in fiber_in.degrees)
                self.conv_out[str(degree_out)] = VersatileConvSE3(
                    sum_freq,
                    list(channels_in_set)[0],
                    channels_out,
                    edge_dim=edge_dim,
                    use_layer_norm=use_layer_norm,
                )
        else:
            for degree_in, channels_in in fiber_in.items():
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
            for degree_out, channels_out in fiber_out.items():
                if fiber_in[degree_out]:
                    self.to_kernel_self[str(degree_out)] = nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out]) / math.sqrt(fiber_in[degree_out])
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
        if self.fused_output:
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
            out[str(degree_out)] = out[str(degree_out)].masked_fill(
                ~graph.edge_mask[:, None, None],
                0,
            )
            if self.pool:
                out[str(degree_out)] = segment_sum_csr(
                    out[str(degree_out)],
                    graph.row_offsets,
                    num_segments=graph.num_nodes,
                    backend=self.backend,
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
        self.backend = "auto"

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

        edge_weights = (key_heads * query_heads[graph.dst]).sum(dim=-1) / math.sqrt(self.key_fiber.num_features)
        edge_weights = edge_weights.masked_fill(
            ~graph.edge_mask[:, None],
            torch.finfo(edge_weights.dtype).min,
        )
        edge_weights = cast(
            Tensor,
            segment_softmax_csr(edge_weights, graph.row_offsets, backend=self.backend),
        )[..., None, None]

        value_heads = value.view(value.shape[0], self.num_heads, -1, value.shape[-1])
        feat_out = cast(
            Tensor,
            segment_sum_csr(
                edge_weights * value_heads,
                graph.row_offsets,
                num_segments=graph.num_nodes,
                backend=self.backend,
            ),
        )
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
        value_fiber = Fiber((degree, channels // channels_div) for degree, channels in fiber_out.items())
        key_query_fiber = Fiber(
            (degree, channels) for degree, channels in value_fiber.items() if degree in fiber_in.degrees
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
    ) -> None:
        super().__init__()
        fiber_edge = Fiber({}) if fiber_edge is None else fiber_edge
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.backend = "auto"
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
        self.graph_modules = nn.ModuleList(graph_modules)

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

        basis = basis or cast(
            "dict[str, Tensor]",
            se3_basis(graph.rel_pos, max_degree=self.max_degree, backend=self.backend),
        )
        edge_feats = get_populated_edge_features(graph.rel_pos, edge_feats)
        for block_index in range(self.num_layers):
            attention = self.graph_modules[2 * block_index]
            norm = self.graph_modules[2 * block_index + 1]
            node_feats = attention(node_feats, edge_feats, graph, basis)
            node_feats = norm(node_feats)
        output = self.graph_modules[-1]
        return output(node_feats, edge_feats, graph, basis)


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
    """

    def __init__(self, n_feat: int) -> None:
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
        self.vocab_size = vocab_size
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
    """

    def __init__(self, msa_size: int, state_size: int) -> None:
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

    if weight.is_meta:
        return torch.empty_like(weight)

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
        return torch.clamp(mu + sigma * math.sqrt(2) * torch.erfinv(v), a, b)

    stddev = math.sqrt(1.0 / weight.shape[-1]) / 0.87962566103423978
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

        if not self.training or self.dropout == 0:
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = (torch.rand(shape, device=x.device) >= self.dropout).to(x.dtype)
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

    cyclic_mask = _cyclic_mask(cyclic, idx)
    if cyclic_mask is not None:
        chain_ids = _chain_ids(idx)
        same_chain = chain_ids[:, :, None] == chain_ids[:, None, :]
        positions = torch.arange(idx.shape[1], device=idx.device)
        earlier = positions[None, None, :] < positions[None, :, None]
        later = positions[None, None, :] > positions[None, :, None]
        cyclic_peers = same_chain & cyclic_mask[:, None, :]
        first = cyclic_mask & ~(cyclic_peers & earlier).any(dim=-1)
        last = cyclic_mask & ~(cyclic_peers & later).any(dim=-1)
        forward_closure = last[:, :, None] & first[:, None, :] & same_chain
        reverse_closure = first[:, :, None] & last[:, None, :] & same_chain
        neigh = torch.where(forward_closure, torch.ones_like(neigh), neigh)
        neigh = torch.where(reverse_closure, -torch.ones_like(neigh), neigh)

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

    def __init__(
        self,
        coords: torch.Tensor,
        pair: torch.Tensor,
        batch: torch.Tensor,
        src_res: torch.Tensor,
        dst_res: torch.Tensor,
        edge_mask: torch.Tensor,
        *,
        batch_size: int,
        length: int,
    ) -> None:
        src = batch * length + src_res
        dst = batch * length + dst_res
        order = torch.argsort(dst * (batch_size * length) + src, stable=True)
        src, dst = src[order], dst[order]
        batch, src_res, dst_res = batch[order], src_res[order], dst_res[order]
        self.edge_mask = edge_mask[order]
        self.src = src
        self.dst = dst
        self.rel_pos = (coords[batch, dst_res, :] - coords[batch, src_res, :]).detach()
        self.rel_pos = self.rel_pos.masked_fill(~self.edge_mask[:, None], 0)
        self.rel_pos[:, 0] = self.rel_pos[:, 0].masked_fill(~self.edge_mask, 1)
        self.edge_feats = pair[batch, src_res, dst_res][..., None]
        self.edge_feats = self.edge_feats.masked_fill(~self.edge_mask[:, None, None], 0)
        self.num_nodes = batch_size * length
        counts = torch.zeros(self.num_nodes, dtype=torch.long, device=coords.device)
        counts.scatter_add_(0, dst, torch.ones_like(dst))
        self.row_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

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
    residues = torch.arange(length, device=coords.device)
    batch = torch.arange(batch_size, device=coords.device)[:, None, None].expand(-1, length, length).reshape(-1)
    src_res = residues[None, :, None].expand(batch_size, -1, length).reshape(-1)
    dst_res = residues[None, None, :].expand(batch_size, length, -1).reshape(-1)
    edge_mask = (idx[:, None, :] != idx[:, :, None]).reshape(-1)
    graph = RFGraph(
        coords,
        pair,
        batch,
        src_res,
        dst_res,
        edge_mask,
        batch_size=batch_size,
        length=length,
    )
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
    _, topk_targets = torch.topk(distances, min(top_k, length), largest=False)
    residues = torch.arange(length, device=coords.device)
    local_offsets = torch.arange(1 - kmin, kmin, device=coords.device)
    local_targets = residues[:, None] + local_offsets[None]
    local_valid = (local_targets >= 0) & (local_targets < length)
    local_targets = local_targets.clamp(0, length - 1)
    gathered_separation = torch.gather(
        sep,
        2,
        local_targets[None].expand(batch_size, -1, -1),
    )
    local_valid = local_valid[None] & (gathered_separation < kmin)
    local_valid = local_valid & ~(local_targets[None, :, :, None] == topk_targets[:, :, None, :]).any(dim=-1)
    targets = torch.cat(
        (topk_targets, local_targets[None].expand(batch_size, -1, -1)),
        dim=-1,
    )
    edge_mask = torch.cat((torch.ones_like(topk_targets, dtype=torch.bool), local_valid), dim=-1)
    edges_per_source = targets.shape[-1]
    batch = (
        torch.arange(batch_size, device=coords.device)[:, None, None].expand(-1, length, edges_per_source).reshape(-1)
    )
    src_res = residues[None, :, None].expand(batch_size, -1, edges_per_source).reshape(-1)
    dst_res = targets.reshape(-1)
    graph = RFGraph(
        coords,
        pair,
        batch,
        src_res,
        dst_res,
        edge_mask.reshape(-1),
        batch_size=batch_size,
        length=length,
    )
    return graph, graph.edge_feats


class RFdiffusionSymmetry:
    r"""Project declared PointGroupSymmetry units onto one exact rotation group."""

    def __init__(self, condition: PointGroupSymmetry, layout: RFdiffusionLayout) -> None:
        row_by_reference = {reference: row for row, reference in enumerate(layout.output_residues)}
        row_by_reference.update(
            {
                alias: row_by_reference[reference]
                for alias, reference in symbolic_residue_aliases(layout.output_residues).items()
            }
        )
        units = []
        for unit in condition.units:
            rows: list[int] = []
            for target in unit:
                if type(target) is str:
                    rows.extend(
                        row
                        for row, reference in enumerate(layout.output_residues)
                        if type(reference) is SymbolicResidueSite and reference.placement_id == target
                    )
                else:
                    rows.append(row_by_reference[target])
            units.append(tuple(rows))
        lengths = {len(unit) for unit in units}
        if len(lengths) != 1:
            raise ValueError("PointGroupSymmetry units must resolve to equal concrete lengths")
        if len({row for unit in units for row in unit}) != sum(map(len, units)):
            raise ValueError("PointGroupSymmetry units must not overlap")
        if {row for unit in units for row in unit} != set(range(len(layout.output_residues))):
            raise ValueError("RFdiffusion PointGroupSymmetry must cover every output residue exactly once")
        self.units = tuple(units)
        self.kind = condition.kind
        self.rotations = self.rotation_matrices(condition.kind, condition.order)

    @staticmethod
    def rotation_matrices(kind: str, order: int | None) -> tuple[Tensor, ...]:
        r"""Return RFdiffusion point-group matrices in the released upstream order."""

        if kind == "cyclic":
            angles = np.arange(order, dtype=np.float64) * (2 * np.pi / order)
            matrices = np.stack(
                [
                    np.array(
                        (
                            (np.cos(angle), -np.sin(angle), 0.0),
                            (np.sin(angle), np.cos(angle), 0.0),
                            (0.0, 0.0, 1.0),
                        )
                    )
                    for angle in angles
                ]
            )
        elif kind == "dihedral":
            angles = np.arange(order, dtype=np.float64) * (2 * np.pi / order)
            flip = np.diag((1.0, -1.0, -1.0))
            matrices = []
            for angle in angles:
                rotation = np.array(
                    (
                        (np.cos(angle), -np.sin(angle), 0.0),
                        (np.sin(angle), np.cos(angle), 0.0),
                        (0.0, 0.0, 1.0),
                    )
                )
                matrices.extend((rotation, flip @ rotation))
            matrices = np.stack(matrices)
        else:
            group = {"tetrahedral": "T", "octahedral": "O", "icosahedral": "I"}[kind]
            matrices = Rotation.create_group(group).as_matrix()
            if kind == "icosahedral":
                q = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
                matrices = np.einsum("ij,njk,lk->nil", q, matrices, q)
            rounded = np.round(matrices, decimals=12)
            canonical_order = sorted(
                range(len(matrices)),
                key=lambda index: tuple(rounded[index].reshape(-1)),
            )
            matrices = matrices[np.asarray(canonical_order)]
            permutation = {
                "tetrahedral": (10, 7, 11, 6, 5, 9, 4, 8, 3, 2, 1, 0),
                "octahedral": (19, 18, 23, 21, 14, 15, 11, 10, 16, 22, 20, 17, 3, 2, 13, 7, 5, 8, 12, 6, 9, 1, 0, 4),
                "icosahedral": (
                    59,
                    58,
                    55,
                    53,
                    57,
                    51,
                    56,
                    50,
                    54,
                    52,
                    47,
                    45,
                    49,
                    43,
                    48,
                    42,
                    46,
                    44,
                    41,
                    35,
                    39,
                    37,
                    38,
                    36,
                    40,
                    34,
                    31,
                    29,
                    33,
                    26,
                    28,
                    30,
                    32,
                    27,
                    0,
                    1,
                    9,
                    3,
                    7,
                    5,
                    6,
                    4,
                    8,
                    2,
                    17,
                    11,
                    15,
                    13,
                    14,
                    12,
                    16,
                    10,
                    23,
                    21,
                    25,
                    19,
                    24,
                    18,
                    22,
                    20,
                ),
            }[kind]
            matrices = matrices[np.asarray(permutation)]
        return tuple(torch.as_tensor(matrix, dtype=torch.float32) for matrix in matrices)

    def apply(
        self,
        coords: Tensor,
        sequence: Tensor,
        *,
        recenter: bool,
        radius_angstrom: float,
    ) -> tuple[Tensor, Tensor]:
        r"""Project coordinates and sequence from the first declared unit."""

        projected_coords = coords.clone()
        projected_sequence = sequence.clone()
        base_index = torch.tensor(self.units[0], device=coords.device)
        base_coords = coords[base_index]
        base_sequence = sequence[base_index]
        for rows, rotation in zip(self.units, self.rotations, strict=True):
            index = torch.tensor(rows, device=coords.device)
            matrix = rotation.to(device=coords.device, dtype=coords.dtype)
            rotated = torch.einsum("raj,kj->rak", base_coords, matrix)
            if self.kind == "octahedral" and recenter:
                offset = coords.new_tensor((radius_angstrom, 0.0, 0.0))
                rotated = rotated - rotated[:, 1].mean(dim=0) + torch.einsum("j,kj->k", offset, matrix)
            projected_coords[index] = rotated
            projected_sequence[index] = base_sequence
        return projected_coords, projected_sequence

    def chain_broken_residue_index(self, residue_index: Tensor) -> Tensor:
        r"""Add the released full-design-length unit breaks to residue indices."""

        result = residue_index.clone()
        offset = result.shape[-1]
        for unit_index, rows in enumerate(self.units):
            result[..., torch.tensor(rows, device=result.device)] += offset * (unit_index + 1)
        return result


class RFdiffusionGuidance:
    r"""Differentiate canonical soft geometric Conditions during reverse diffusion."""

    def __init__(
        self,
        layout: RFdiffusionLayout,
    ) -> None:
        self.layout = layout
        self.row_by_reference = {reference: row for row, reference in enumerate(layout.output_residues)}
        self.row_by_reference.update(
            {
                alias: self.row_by_reference[reference]
                for alias, reference in symbolic_residue_aliases(layout.output_residues).items()
            }
        )
        self.structure_index = (
            {} if layout.structure is None else {site: index for index, site in enumerate(layout.structure.sites)}
        )
        reference_sites = {
            site
            for condition in layout.conditions
            if type(condition) is ReferenceCoordinates
            for site in condition.sites
        }
        anchor_rows: list[int] = []
        anchor_atoms: list[int] = []
        anchor_coordinates: list[np.ndarray] = []
        if layout.structure is not None:
            for row, reference in enumerate(layout.output_residues):
                if type(reference) is not tuple:
                    continue
                sites = tuple(AtomSite(reference[0], reference[1], atom) for atom in ("N", "CA", "C"))
                if not all(site in reference_sites for site in sites):
                    continue
                for atom_index, site in enumerate(sites):
                    source = self.structure_index[site]
                    anchor_rows.append(row)
                    anchor_atoms.append(atom_index)
                    anchor_coordinates.append(layout.structure.coordinates[source])
                break
        self.anchor_rows = tuple(anchor_rows)
        self.anchor_atoms = tuple(anchor_atoms)
        self.anchor_coordinates = np.asarray(anchor_coordinates, dtype=np.float32)
        self.conditions = tuple(
            condition
            for condition in layout.conditions
            if (
                type(condition)
                in (
                    Coordination,
                    Distance,
                    RadiusOfGyration,
                    SubstrateInteraction,
                )
                and condition.intent == "soft"
            )
            or (
                type(condition) is SymbolicBond
                and {
                    (
                        condition.site_a.atom_id
                        if type(condition.site_a) is SymbolicAtomSite
                        else condition.site_a.atom_id
                    ),
                    (
                        condition.site_b.atom_id
                        if type(condition.site_b) is SymbolicAtomSite
                        else condition.site_b.atom_id
                    ),
                }
                == {"N", "C"}
            )
        )
        if any(type(condition) is SubstrateInteraction for condition in self.conditions) and not self.anchor_rows:
            raise ValueError(
                "SubstrateInteraction with external atoms requires fixed ReferenceCoordinates to define "
                "the substrate frame"
            )

    def __bool__(self) -> bool:
        return bool(self.conditions)

    def __call__(self, coords: Tensor) -> Tensor:
        potential = coords.new_zeros(())
        for condition in self.conditions:
            weight = 1.0 if type(condition) is SymbolicBond or condition.weight is None else condition.weight
            if type(condition) is Coordination:
                left = self._coordinates(condition.side_a, coords)
                right = self._coordinates(condition.side_b, coords)
                scaled = (torch.cdist(left, right) - condition.center_angstrom) / condition.scale_angstrom
                coordination = (1 / (1 + scaled.pow(6))).sum()
                direction = 1 if condition.direction == "maximize" else -1
                potential = potential + direction * weight * coordination
            elif type(condition) is Distance:
                distance = torch.linalg.vector_norm(
                    self._atom(condition.site_a, coords) - self._atom(condition.site_b, coords)
                )
                lower = max(0.0, condition.lower_angstrom - condition.tolerance_angstrom)
                upper = condition.upper_angstrom + condition.tolerance_angstrom
                violation = torch.relu(distance.new_tensor(lower) - distance).square()
                violation = violation + torch.relu(distance - distance.new_tensor(upper)).square()
                potential = potential - weight * violation
            elif type(condition) is RadiusOfGyration:
                rows = self._residue_rows(condition.targets)
                calpha = coords[torch.tensor(rows, device=coords.device), 1]
                distances = torch.linalg.vector_norm(calpha - calpha.mean(dim=0), dim=-1)
                radius = torch.sqrt(torch.clamp_min(distances, condition.upper_angstrom).square().mean())
                potential = potential - weight * radius
            elif type(condition) is SubstrateInteraction:
                rows = self._residue_rows(condition.residues)
                residue_coords = coords[torch.tensor(rows, device=coords.device), 1]
                substrate_coords = self._coordinates(condition.substrate_atoms, coords)
                nearest = torch.cdist(residue_coords, substrate_coords).amin(dim=-1)
                scaled = (nearest - condition.center_angstrom) / condition.scale_angstrom
                attraction = condition.attraction_strength / (1 + scaled.pow(6))
                exponent = 1.5
                repulsion = (
                    condition.repulsion_slope**2
                    * torch.clamp_min(condition.repulsion_cutoff_angstrom - nearest, 0).pow(exponent)
                    / (exponent * condition.repulsion_cutoff_angstrom ** (exponent - 1))
                )
                potential = potential + weight * (attraction - repulsion).sum()
            else:
                distance = torch.linalg.vector_norm(
                    self._atom(condition.site_a, coords) - self._atom(condition.site_b, coords)
                )
                potential = potential - (distance - 1.33).square()
        return potential

    def _residue_rows(self, references: Iterable[object]) -> tuple[int, ...]:
        rows: list[int] = []
        for reference in references:
            if type(reference) is str:
                rows.extend(
                    row
                    for row, output in enumerate(self.layout.output_residues)
                    if type(output) is SymbolicResidueSite and output.placement_id == reference
                )
            elif type(reference) in (tuple, SymbolicResidueSite):
                rows.append(self.row_by_reference[reference])
            elif type(reference) is SymbolicAtomSite:
                rows.append(self.row_by_reference[reference.residue])
            elif type(reference) is AtomSite:
                system = self.layout.system
                entity = system.entities[system.instances[reference.instance_id]]  # type: ignore[union-attr]
                if type(entity) is not Molecule:
                    rows.append(self.row_by_reference[(reference.instance_id, reference.component_index)])
        return tuple(dict.fromkeys(rows))

    def _coordinates(self, references: Iterable[object], coords: Tensor) -> Tensor:
        values: list[Tensor] = []
        for reference in references:
            if type(reference) in (str, tuple, SymbolicResidueSite):
                rows = self._residue_rows((reference,))
                values.extend(self._residue_coordinate(row, coords) for row in rows)
            else:
                values.append(self._atom(reference, coords))
        return torch.stack(values)

    def _residue_coordinate(self, row: int, coords: Tensor) -> Tensor:
        reference = self.layout.output_residues[row]
        if type(reference) is tuple and self.layout.system is not None:
            entity = self.layout.system.entities[self.layout.system.instances[reference[0]]]
            if entity.components[reference[1]].component_id == "GLY":
                return coords[row, 1]
        return generate_cbeta(coords[row, 0], coords[row, 1], coords[row, 2])

    def _atom(self, reference: AtomSite | SymbolicAtomSite, coords: Tensor) -> Tensor:
        if type(reference) is SymbolicAtomSite:
            row = self.row_by_reference[reference.residue]
            atom_names = ("N", "CA", "C")
            return coords[row, atom_names.index(reference.atom_id)]
        row = self.row_by_reference.get((reference.instance_id, reference.component_index))
        if row is not None:
            system = self.layout.system
            entity = system.entities[system.instances[reference.instance_id]]  # type: ignore[union-attr]
            component_id = entity.components[reference.component_index].component_id
            atom_names = ATOM14_NAMES[residue_token(component_id)]
            return coords[row, atom_names.index(reference.atom_id)]
        source_index = self.structure_index[reference]
        value = torch.tensor(
            self.layout.structure.coordinates[source_index],  # type: ignore[union-attr]
            device=coords.device,
            dtype=coords.dtype,
        )
        if not self.anchor_rows:
            raise ValueError(
                "guidance to atoms outside the RFdiffusion output requires fixed ReferenceCoordinates "
                "to define their coordinate frame"
            )
        source_anchors = torch.as_tensor(
            self.anchor_coordinates,
            device=coords.device,
            dtype=coords.dtype,
        )
        target_anchors = coords[
            torch.tensor(self.anchor_rows, device=coords.device),
            torch.tensor(self.anchor_atoms, device=coords.device),
        ]
        source_rotation, source_calpha = rigid_from_3_points(
            source_anchors[0],
            source_anchors[1],
            source_anchors[2],
        )
        target_rotation, target_calpha = rigid_from_3_points(
            target_anchors[0],
            target_anchors[1],
            target_anchors[2],
        )
        local = (value - source_calpha) @ source_rotation
        return local @ target_rotation.transpose(-1, -2) + target_calpha


class RFdiffusionSampler:
    r"""
    Reference PyTorch reverse-diffusion loop over one canonical RFdiffusion input.

    Args:
        config: Persistent checkpoint architecture and diffusion process.
        network: The RoseTTAFold denoiser called at each reverse step.
        allatom: The Model-owned atom-coordinate reconstruction module.
        igso3_schedule: The model-baked IGSO(3) lookup table used verbatim for the released schedule.
        final_step: Last reverse timestep.
        partial_steps: Optional partial-diffusion starting timestep.
        noise_scale_ca: Initial CA-translation noise scale at the first reverse step.
        final_noise_scale_ca: Optional CA-translation interpolation endpoint for a linear schedule.
        ca_noise_schedule: CA noise schedule, ``"constant"`` or ``"linear"``.
        noise_scale_frame: Initial backbone-frame noise scale at the first reverse step.
        final_noise_scale_frame: Optional backbone-frame interpolation endpoint for a linear schedule.
        frame_noise_schedule: Frame noise schedule, ``"constant"`` or ``"linear"``.
        align_motif: Whether to align each predicted clean structure onto a fixed motif.
        symmetry_recenter: Whether octahedral units are individually recentered and placed on a radius.
        symmetry_radius_angstrom: Octahedral placement radius.
        schedule_directory: Local IGSO(3) table cache.
    """

    def __init__(
        self,
        config: RFdiffusionConfig,
        *,
        network: nn.Module,
        allatom: ComputeAllAtomCoords,
        igso3_schedule: dict[str, np.ndarray] | None = None,
        final_step: int = 1,
        partial_steps: int | None = None,
        noise_scale_ca: float = 1.0,
        final_noise_scale_ca: float | None = None,
        ca_noise_schedule: Literal["constant", "linear"] = "constant",
        noise_scale_frame: float = 1.0,
        final_noise_scale_frame: float | None = None,
        frame_noise_schedule: Literal["constant", "linear"] = "constant",
        align_motif: bool = True,
        symmetry_recenter: bool = True,
        symmetry_radius_angstrom: float = 10.0,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
        schedule_directory: str | Path | None = None,
    ) -> None:
        final_step = integer_index(final_step)
        partial_steps = None if partial_steps is None else integer_index(partial_steps)
        if not 1 <= final_step <= config.diffuser.num_timesteps:
            raise ValueError("final_step must be in [1, config.diffuser.num_timesteps]")
        if partial_steps is not None and not 1 <= partial_steps <= config.diffuser.num_timesteps:
            raise ValueError("partial_steps must be in [1, config.diffuser.num_timesteps]")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if guidance_decay not in {"constant", "linear", "quadratic", "cubic"}:
            raise ValueError("guidance_decay must be constant, linear, quadratic, or cubic")
        self.config = config
        self.network = network
        self.allatom = allatom
        self.igso3_schedule = igso3_schedule
        self.final_step = final_step
        self.partial_steps = partial_steps
        self.noise_scale_ca = float(noise_scale_ca)
        self.final_noise_scale_ca = self.noise_scale_ca if final_noise_scale_ca is None else float(final_noise_scale_ca)
        self.ca_noise_schedule = ca_noise_schedule
        self.noise_scale_frame = float(noise_scale_frame)
        self.final_noise_scale_frame = (
            self.noise_scale_frame if final_noise_scale_frame is None else float(final_noise_scale_frame)
        )
        self.frame_noise_schedule = frame_noise_schedule
        if (
            ca_noise_schedule == "constant"
            and final_noise_scale_ca is not None
            and self.final_noise_scale_ca != self.noise_scale_ca
        ):
            raise ValueError("final_noise_scale_ca can differ from noise_scale_ca only for a linear schedule")
        if (
            frame_noise_schedule == "constant"
            and final_noise_scale_frame is not None
            and self.final_noise_scale_frame != self.noise_scale_frame
        ):
            raise ValueError("final_noise_scale_frame can differ from noise_scale_frame only for a linear schedule")
        self.align_motif = align_motif
        self.symmetry_recenter = symmetry_recenter
        self.symmetry_radius_angstrom = float(symmetry_radius_angstrom)
        self.guidance_scale = float(guidance_scale)
        self.guidance_decay = guidance_decay
        self.schedule_directory = None if schedule_directory is None else Path(schedule_directory)
        self.igso3_truncation_level = 2000
        self.igso3_num_omega = 1000
        self.igso3_num_sigma = 500
        self.diffuser: Diffuser | None = None

    def sample(
        self,
        batch: RFdiffusionInput,
        *,
        num_sampling_steps: int | None = None,
        final_step: int | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Run one RFdiffusion reverse-diffusion trajectory end to end.

        Initializes the noised structure, then steps from the starting timestep down to ``final_step``, calling
        the network and denoiser each step and accumulating the coord trajectory.

        Args:
            batch: One canonical RFdiffusion tensor input.
            num_sampling_steps: Number of timesteps in the active translation, rotation, and time-feature schedules.
            final_step: Last (smallest) timestep to denoise to; defaults to the constructor value.
            generator: Optional RNG for reproducible sampling.
            return_trajectory: Whether to retain every reverse step on CPU.

        Returns:
            Final upstream ``pX0`` coordinates and sequence tokens followed by
            timesteps, pLDDT, ``pX0``, and ``x_{t-1}``. The final tensors have a
            length-one timestep axis unless full CPU trajectories were requested.

        Examples:
            >>> result = sampler.sample(batch)  # doctest: +SKIP
        """

        active_steps = (
            self.config.diffuser.num_timesteps if num_sampling_steps is None else integer_index(num_sampling_steps)
        )
        if active_steps <= 0:
            raise ValueError("num_sampling_steps must be positive")
        stop_t = self.final_step if final_step is None else integer_index(final_step)
        if batch.layout.templates and (
            self.config.preprocess.template_1d_size != 28 or self.config.preprocess.template_2d_size != 47
        ):
            raise ValueError("Template evidence requires an RFdiffusion fold-conditioned checkpoint")
        if any(type(condition) is SecondaryStructure for condition in batch.layout.conditions) and (
            self.config.preprocess.template_1d_size != 28
        ):
            raise ValueError("SecondaryStructure requires an RFdiffusion fold-conditioned checkpoint")
        mask_sequence = batch.fixed_sequence_mask
        mask_structure = batch.fixed_structure_mask
        diffusion_mask = mask_structure.squeeze(0)
        diffuser = self._make_diffuser(active_steps)
        length = batch.noisy_coords.shape[1]
        coords27 = batch.noisy_coords.new_full((length, 27, 3), float("nan"))
        coords27[:, :14] = batch.noisy_coords[0]
        atom_mask = torch.isfinite(coords27).all(dim=-1)
        original_sequence = batch.sequence[0].argmax(dim=-1)
        if bool(diffusion_mask.any()):
            center = coords27[diffusion_mask, 1].mean(dim=0)
        else:
            observed_calpha = torch.isfinite(coords27[:, 1]).all(dim=-1)
            center = coords27[observed_calpha, 1].mean(dim=0) if bool(observed_calpha.any()) else coords27.new_zeros(3)
        if self.partial_steps is not None:
            if batch.layout.structure is None:
                raise ValueError("partial diffusion requires coordinates for every output residue")
            source_index = {site: index for index, site in enumerate(batch.layout.structure.sites)}
            for reference in batch.layout.output_residues:
                if type(reference) is SymbolicResidueSite:
                    raise ValueError("partial diffusion requires coordinates for every output residue")
                for atom_name in ("N", "CA", "C"):
                    source = source_index.get(AtomSite(reference[0], reference[1], atom_name))
                    if source is None or not bool(batch.layout.structure.coordinate_mask[source]):
                        raise ValueError(
                            "partial diffusion requires observed N/CA/C coordinates for every output residue"
                        )
            initial_coords = coords27
        else:
            initial_coords = get_init_coords(coords27[None, None]).squeeze(0).squeeze(0)

        trajectory_steps = active_steps
        if self.partial_steps is not None:
            if self.partial_steps > trajectory_steps:
                raise ValueError("partial_steps must not exceed the active diffusion schedule")
            trajectory_steps = self.partial_steps
        start_t = trajectory_steps
        if not 1 <= stop_t <= start_t:
            raise ValueError("final_step must be between 1 and the starting timestep")
        sequence = original_sequence.clone()
        sequence[~mask_sequence.squeeze(0)] = MASK_TOKEN
        sequence = one_hot_sequence(sequence)
        diffused, _clean = diffuser.diffuse_pose(
            initial_coords,
            atom_mask,
            diffusion_mask=diffusion_mask,
            t_list=np.arange(1, start_t + 1),
            include_motif_sidechains=self.config.preprocess.motif_sidechain_input,
            generator=generator,
        )
        noisy_coords = diffused[-1, :, :14]
        symmetry_conditions = tuple(
            condition for condition in batch.layout.conditions if type(condition) is PointGroupSymmetry
        )
        symmetry = None if not symmetry_conditions else RFdiffusionSymmetry(symmetry_conditions[0], batch.layout)
        if symmetry is not None:
            noisy_coords, sequence = symmetry.apply(
                noisy_coords,
                sequence,
                recenter=self.symmetry_recenter,
                radius_angstrom=self.symmetry_radius_angstrom,
            )
        guidance = RFdiffusionGuidance(batch.layout)
        denoiser = self._make_denoiser(
            length,
            diffuser=diffuser,
            guidance=guidance if guidance else None,
        )

        if return_trajectory:
            timesteps: list[int] = []
            plddt_trajectory: list[Tensor] = []
            px0_trajectory: list[Tensor] = []
            xt_minus_1_trajectory: list[Tensor] = []
        previous_prediction: Tensor | None = None
        residue_index = batch.residue_index
        if symmetry is not None:
            residue_index = symmetry.chain_broken_residue_index(residue_index)
        secondary_structure = batch.secondary_structure[0]
        block_adjacency = batch.block_adjacency[0]
        hotspot_mask = batch.hotspot_mask[0]
        cyclic_mask = batch.cyclic_mask[0]
        for t in range(start_t, stop_t - 1, -1):
            (
                msa_masked,
                msa_full,
                sequence_input,
                noisy_coords_input,
                template_1d,
                template_2d,
                template_coords,
                alpha_t,
            ) = self._preprocess(
                sequence,
                noisy_coords,
                t,
                mask_structure=mask_structure,
                diffuser=diffuser,
                previous_prediction=previous_prediction,
                secondary_structure=secondary_structure,
                block_adjacency=block_adjacency,
                hotspot_mask=hotspot_mask,
            )
            output = self.network(  # type: ignore[misc,operator]
                msa_masked,
                msa_full,
                sequence_input,
                noisy_coords_input,
                residue_index,
                template_1d=template_1d,
                template_2d=template_2d,
                template_coords=template_coords,
                alpha_t=alpha_t,
                msa_prev=None,
                pair_prev=None,
                state_prev=None,
                motif_mask=diffusion_mask,
                cyclic_reses=cyclic_mask,
            )
            predicted_backbone, alpha, plddt = output[2], output[4], output[6]
            sequence_tokens = sequence_input.argmax(dim=-1)
            _frames, predicted_full = self.allatom(sequence_tokens, predicted_backbone, alpha)
            pred_coords = predicted_full.squeeze(0)[:, :14]
            if symmetry is not None:
                symmetric_coords, _symmetric_sequence = symmetry.apply(
                    pred_coords,
                    sequence,
                    recenter=self.symmetry_recenter,
                    radius_angstrom=self.symmetry_radius_angstrom,
                )
                previous_prediction = symmetric_coords[:, :3].detach()
                _frames, predicted_full = self.allatom(
                    sequence_tokens,
                    previous_prediction[None],
                    alpha,
                )
                pred_coords = predicted_full.squeeze(0)[:, :14]
            else:
                previous_prediction = predicted_backbone.squeeze(0).detach()
            px0_coords = pred_coords
            if t > stop_t:
                noisy_coords, px0_coords = denoiser.get_next_pose(
                    noisy_coords=noisy_coords,
                    pred_coords=pred_coords,
                    t=t,
                    diffusion_mask=diffusion_mask,
                    align_motif=self.align_motif,
                    include_motif_sidechains=self.config.preprocess.motif_sidechain_input,
                    generator=generator,
                )
            else:
                noisy_coords = pred_coords
            if symmetry is not None:
                noisy_coords, sequence = symmetry.apply(
                    noisy_coords,
                    sequence,
                    recenter=self.symmetry_recenter,
                    radius_angstrom=self.symmetry_radius_angstrom,
                )
            final_timestep = t
            final_plddt = plddt.squeeze(0)
            final_px0 = px0_coords
            final_xt_minus_1 = noisy_coords
            if return_trajectory:
                timesteps.append(t)
                plddt_trajectory.append(final_plddt.detach().cpu())
                px0_trajectory.append(final_px0.detach().cpu())
                xt_minus_1_trajectory.append(final_xt_minus_1.detach().cpu())
        if return_trajectory:
            center_cpu = center.detach().cpu()
            timestep_values = torch.tensor(timesteps, dtype=torch.int64)
            plddt_values = torch.stack(plddt_trajectory)
            px0_values = torch.stack(px0_trajectory) + center_cpu
            xt_minus_1_values = torch.stack(xt_minus_1_trajectory) + center_cpu
        else:
            timestep_values = noisy_coords.new_tensor((final_timestep,), dtype=torch.int64)
            plddt_values = final_plddt[None]
            px0_values = (final_px0 + center)[None]
            xt_minus_1_values = (final_xt_minus_1 + center)[None]
        return (
            final_px0 + center,
            sequence.argmax(dim=-1),
            timestep_values,
            plddt_values,
            px0_values,
            xt_minus_1_values,
        )

    def _preprocess(
        self,
        seq: Tensor,
        template_coords_in: Tensor,
        t: int,
        *,
        mask_structure: Tensor,
        diffuser: Diffuser,
        previous_prediction: Tensor | None,
        secondary_structure: Tensor,
        block_adjacency: Tensor,
        hotspot_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Assemble the RoseTTAFold network input tensors for timestep ``t``.

        Builds the masked/full MSA, 1D/2D template features (including time, hotspot, and scaffold SS/adjacency
        features when the checkpoint expects them), template coords, and torsion features, wiring in
        self-conditioning from the previous prediction.

        Args:
            seq: Current one-hot sequence features `[L, 22]`.
            template_coords_in: Current template backbone coords `[L, A, 3]`.
            t: Current timestep.
        """

        L = seq.shape[0]
        msa_masked = seq.new_zeros((1, 1, L, 48))
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
        msa_full = seq.new_zeros((1, 1, L, 25))
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
        template_1d = seq.new_zeros((1, 1, L, 21))
        seq_template_1d = seq.clone()
        mask_token_positions = seq_template_1d[:, MASK_TOKEN] == 1
        seq_template_1d[mask_token_positions, 20] = 1
        seq_template_1d[mask_token_positions, MASK_TOKEN] = 0
        template_1d[:, :, :, :21] = seq_template_1d[None, None, :, :21]
        timefeature = seq.new_zeros((L,))
        timefeature[mask_structure.squeeze(0)] = 1
        timefeature[~mask_structure.squeeze(0)] = 1 - t / diffuser.num_timesteps
        template_1d = torch.cat((template_1d, timefeature[None, None, :, None]), dim=-1).float()

        working_coords = template_coords_in.clone()
        if self.config.preprocess.sidechain_input:
            working_coords[mask_token_positions, 3:] = float("nan")
        else:
            working_coords[~mask_structure.squeeze(0), 3:] = float("nan")
        template_coords = working_coords[None, None]
        template_coords = torch.cat(
            (
                template_coords,
                template_coords.new_full((1, 1, L, 13, 3), float("nan")),
            ),
            dim=3,
        )
        noisy_coords_input = template_coords.squeeze(0)
        template_2d = coords_to_t2d(template_coords)
        seq_tmp = template_1d[..., :-1].argmax(dim=-1).reshape(1, L)
        alpha_t = make_alpha_t(template_coords, seq_tmp)
        if self.config.preprocess.structure_self_conditioning:
            if previous_prediction is not None and (t < diffuser.num_timesteps) and (t != self.partial_steps):
                prev = previous_prediction[:, :3][None, None]
                zeros = prev.new_zeros((1, 1, L, 24, 3))
                template_coords = torch.cat((prev, zeros), dim=-2)
                template_2d[..., :44] = coords_to_t2d(template_coords)
            else:
                template_2d[..., :44] = 0
                template_coords = torch.zeros_like(template_coords)

        if self.config.preprocess.template_1d_size >= 24:
            hotspot_tens = hotspot_mask.to(dtype=seq.dtype)
            template_1d = torch.cat(
                (template_1d, torch.zeros_like(template_1d[..., :1]), hotspot_tens[None, None, :, None]), dim=-1
            )

        if self.config.preprocess.template_1d_size == 28:
            template_1d = torch.cat((template_1d, secondary_structure[None, None].float()), dim=-1)
        if self.config.preprocess.template_2d_size == 47:
            template_2d = torch.cat((template_2d, block_adjacency[None, None].float()), dim=-1)

        return (
            msa_masked,
            msa_full,
            seq[None],
            noisy_coords_input,
            template_1d,
            template_2d,
            template_coords,
            alpha_t,
        )

    def _make_diffuser(self, num_timesteps: int) -> Diffuser:
        r"""Construct the forward [`Diffuser`][ofoldx.models.rfdiffusion.model.Diffuser] from config."""

        if self.diffuser is not None and self.diffuser.num_timesteps == num_timesteps:
            return self.diffuser
        config = self.config.diffuser
        diffuser = Diffuser(
            num_timesteps=num_timesteps,
            beta_initial=config.beta_initial,
            beta_final=config.beta_final,
            min_sigma=config.min_rotation_sigma,
            max_sigma=config.max_rotation_sigma,
            min_b=config.min_rotation_beta,
            max_b=config.max_rotation_beta,
            schedule_type=config.schedule_type,
            so3_schedule_type=config.rotation_schedule_type,
            so3_type=config.rotation_diffuser,
            crd_scale=config.coordinate_scale,
            cache_dir=self.schedule_directory,
            partial_timesteps=self.partial_steps,
            truncation_level=self.igso3_truncation_level,
            igso3_num_omega=self.igso3_num_omega,
            igso3_num_sigma=self.igso3_num_sigma,
            igso3_schedule=self.igso3_schedule,
        )
        self.diffuser = diffuser
        return diffuser

    def _make_denoiser(
        self,
        length: int,
        *,
        diffuser: Diffuser,
        guidance: RFdiffusionGuidance | None,
    ) -> Denoiser:
        r"""Construct the reverse [`Denoiser`][ofoldx.models.rfdiffusion.model.Denoiser] from config."""

        config = self.config.diffuser
        return Denoiser(
            num_timesteps=diffuser.num_timesteps,
            truncation_level=length,
            diffuser=diffuser,
            beta_initial=config.beta_initial,
            beta_final=config.beta_final,
            schedule_type=config.schedule_type,
            so3_type=config.rotation_diffuser,
            noise_scale_ca=self.noise_scale_ca,
            final_noise_scale_ca=self.final_noise_scale_ca,
            ca_noise_schedule=self.ca_noise_schedule,
            noise_scale_frame=self.noise_scale_frame,
            final_noise_scale_frame=self.final_noise_scale_frame,
            frame_noise_schedule=self.frame_noise_schedule,
            crd_scale=config.coordinate_scale,
            partial_timesteps=self.partial_steps,
            guidance=guidance,
            guidance_scale=self.guidance_scale,
            guidance_decay=self.guidance_decay,
        )


def get_beta_schedule(
    num_timesteps: int,
    beta_initial: float,
    beta_final: float,
    schedule_type: str,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Build the RFdiffusion translation-noise beta, alpha, and cumulative alpha-bar schedules.

    Args:
        num_timesteps: Total number of diffusion timesteps (must be >= 15).
        beta_initial: Initial beta (rescaled by ``200 / num_timesteps``).
        beta_final: Final beta (rescaled by ``200 / num_timesteps``).
        schedule_type: Schedule shape; only `"linear"` is supported.

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
        self.num_timesteps = integer_index(num_timesteps)
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
        num_steps: int | None = None,
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
        current = coords
        steps = self.num_timesteps if num_steps is None else integer_index(num_steps)
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError("num_steps must be in [1, num_timesteps]")
        for t in range(1, steps + 1):
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

        """
        t_idx = t - 1
        ca_coords = x[:, 1]
        beta_t = self.beta_schedule.to(device=x.device, dtype=x.dtype)[t_idx]
        mean = torch.sqrt(1 - beta_t) * ca_coords
        std = torch.sqrt(torch.ones_like(ca_coords) * beta_t * var_scale)
        sampled = mean + torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator) * std
        delta = sampled - ca_coords
        if diffusion_mask is not None:
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
        igso3_schedule: Model-baked IGSO(3) table returned verbatim for the released schedule.

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
        cache_dir: str | Path | None,
        num_omega: int = 1000,
        num_sigma: int = 500,
        schedule: str = "linear",
        truncation_level: int = 2000,
        igso3_schedule: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.num_timesteps = integer_index(num_timesteps)
        self.schedule = schedule
        self.igso3_schedule = igso3_schedule
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.min_b = float(min_b)
        self.max_b = float(max_b)
        if self.schedule == "linear":
            self.max_sigma = float(self.sigma(torch.tensor(1.0)))
        if self.schedule not in {"linear", "exponential"}:
            raise ValueError(f"unsupported IGSO3 schedule {schedule!r}")
        self.num_omega = integer_index(num_omega)
        self.num_sigma = integer_index(num_sigma)
        self.truncation_level = integer_index(truncation_level)
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

        with torch.inference_mode(False), torch.enable_grad():
            t_tensor = torch.as_tensor(t, dtype=torch.float32).requires_grad_(True)
            sigma_sq = self.sigma(t_tensor) ** 2
            grad = torch.autograd.grad(sigma_sq.sum(), t_tensor)[0]
            return torch.sqrt(grad)

    def sample(
        self,
        ts: np.ndarray,
        *,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
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
            rng_device = torch.device("cpu") if generator is None else generator.device
            uniform = torch.rand(n_samples, device=rng_device, generator=generator).cpu().numpy()
            all_samples.append(np.interp(uniform, cdf, self.igso3_vals["discrete_omega"]))
        return np.stack(all_samples, axis=0)

    def sample_vec(
        self,
        ts: np.ndarray,
        *,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        r"""
        Sample IGSO3 rotation vectors (random axis times sampled angle) at the given timesteps.

        Args:
            ts: One-indexed timesteps to sample at.
            n_samples: Number of rotation vectors per timestep.

        Returns:
            Rotation vectors `[len(ts), n_samples, 3]` (unit axis scaled by the sampled angle).
        """

        rng_device = torch.device("cpu") if generator is None else generator.device
        axes = torch.randn((len(ts), n_samples, 3), device=rng_device, generator=generator).cpu().numpy()
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        return axes * self.sample(ts, n_samples=n_samples, generator=generator)[..., None]

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
        generator: torch.Generator | None = None,
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
        t_values = np.arange(self.num_timesteps) + 1 if t_list is None else np.asarray(t_list, dtype=np.int64)
        num_res = len(coords_np)
        N = torch.from_numpy(coords_np[None, :, 0])
        Ca = torch.from_numpy(coords_np[None, :, 1])
        C = torch.from_numpy(coords_np[None, :, 2])
        R_true, Ca_true = rigid_from_3_points(N, Ca, C)
        R_true_np = R_true[0].detach().cpu().numpy()
        Ca_np = Ca_true[0].detach().cpu().numpy()
        sampled_rots = self.sample_vec(t_values, n_samples=num_res, generator=generator)
        if diffusion_mask is not None:
            mask_np = (
                diffusion_mask.detach().cpu().numpy() if torch.is_tensor(diffusion_mask) else np.asarray(diffusion_mask)
            )
            sampled_rots = sampled_rots * (1 - mask_np[None, :, None])
        R_sampled = Rotation.from_rotvec(sampled_rots.reshape(-1, 3)).as_matrix().reshape(len(t_values), num_res, 3, 3)
        R_perturbed = np.einsum("tnij,njk->tnik", R_sampled, R_true_np)
        perturbed = np.einsum("tnij,naj->tnai", R_sampled, coords_np[:, :3] - Ca_np[:, None]) + Ca_np[None, :, None]
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
        generator: torch.Generator | None = None,
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
        rng_device = score.device if generator is None else generator.device
        z = torch.randn(score.shape, dtype=score.dtype, device=rng_device, generator=generator).to(score.device)
        z = z * noise_level
        delta_r = (rot_g**2) * self.step_size * score
        perturb_tangent = delta_r + rot_g * math.sqrt(self.step_size) * z
        if mask is not None:
            perturb_tangent = perturb_tangent * (1 - mask.long()).to(device=perturb_tangent.device)[:, None]
        perturb = exp_map(perturb_tangent)
        if return_perturb:
            return perturb
        return torch.einsum("...ij,...jk->...ik", perturb, R_t_t)

    def _calc_igso3_vals(self) -> dict[str, np.ndarray]:
        r"""Return the model-baked IGSO3 table for the released schedule, else load/compute a cache."""

        if (
            self.igso3_schedule is not None
            and self.num_timesteps == 50
            and self.num_omega == 1000
            and self.num_sigma == 500
            and self.truncation_level == 2000
            and self.schedule == "linear"
            and self.min_sigma == 0.02
            and self.min_b == 1.5
            and self.max_b == 2.5
        ):
            return self.igso3_schedule
        cache_file = self._cache_file()
        if cache_file is not None and cache_file.exists():
            with cache_file.open("rb") as handle:
                return cast(dict[str, np.ndarray], pickle.load(handle))
        values = calculate_igso3(
            num_sigma=self.num_sigma,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_omega=self.num_omega,
            num_terms=self.truncation_level,
        )
        if cache_file is not None:
            write_pickle(cache_file, values)
        return values

    def _cache_file(self) -> Path | None:
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
        return None if self.cache_dir is None else self.cache_dir / ("_".join(parts) + ".pkl")


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
        igso3_schedule: Model-baked IGSO(3) table used verbatim for the released schedule.

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
        cache_dir: str | Path | None = None,
        partial_timesteps: int | None = None,
        truncation_level: int = 2000,
        igso3_num_omega: int = 1000,
        igso3_num_sigma: int = 500,
        igso3_schedule: dict[str, np.ndarray] | None = None,
    ) -> None:
        if so3_type != "igso3":
            raise ValueError(f"unsupported SO(3) diffuser {so3_type!r}")
        self.num_timesteps = integer_index(num_timesteps)
        self.partial_timesteps = None if partial_timesteps is None else integer_index(partial_timesteps)
        if self.partial_timesteps is not None and not 0 < self.partial_timesteps <= self.num_timesteps:
            raise ValueError("partial_timesteps must be in the interval [1, T]")
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.crd_scale = float(crd_scale)
        self.var_scale = float(var_scale)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
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
            igso3_schedule=igso3_schedule,
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
            atom_mask: Atom mask used when preserving fixed motif sidechains.
            include_motif_sidechains: Whether to preserve fixed-motif sidechain coords.
            diffusion_mask: Per-residue mask; masked (motif) residues stay fixed.
            t_list: Optional one-indexed timesteps to return; all timesteps otherwise.
            generator: Optional RNG for reproducible noise.

        Returns:
            A tuple ``(diffused_fullatom, true_coords)`` of the noised atom27 trajectory and the centered
            clean coords.

        """

        if diffusion_mask is None:
            diffusion_mask = torch.zeros(len(coords.squeeze()), dtype=torch.bool, device=coords.device)
        diffusion_mask = diffusion_mask.to(device=coords.device).bool()
        if diffusion_mask.any():
            motif_com = coords[diffusion_mask, 1].mean(dim=0)
            coords = coords - motif_com
        else:
            coords = coords - coords[:, 1].mean(dim=0)
        true_coords = coords
        t_values = (
            np.arange(1, self.num_timesteps + 1, dtype=np.int64)
            if t_list is None
            else np.asarray(t_list, dtype=np.int64)
        )
        if t_values.ndim != 1 or len(t_values) == 0 or np.any(t_values < 1) or np.any(t_values > self.num_timesteps):
            raise ValueError("t_list must contain one-indexed timesteps in the active diffusion schedule")
        max_step = int(t_values.max())
        scaled = coords * self.crd_scale
        _diffused_T, deltas = self.eucl_diffuser.diffuse_translations(
            scaled[:, :3],
            diffusion_mask=diffusion_mask,
            var_scale=self.var_scale,
            num_steps=max_step,
            generator=generator,
        )
        deltas = deltas / self.crd_scale
        frame_coords, _frames = self.so3_diffuser.diffuse_frames(
            scaled[:, :3],
            diffusion_mask=diffusion_mask.detach().cpu().numpy(),
            t_list=t_values,
            generator=generator,
        )
        frame_coords = torch.from_numpy(frame_coords).to(device=coords.device, dtype=coords.dtype) / self.crd_scale
        cum_delta = deltas.cumsum(dim=1)
        requested = torch.as_tensor(t_values - 1, dtype=torch.long, device=cum_delta.device)
        cum_delta = cum_delta.index_select(1, requested)
        diffused_bb = (frame_coords + cum_delta[:, :, None]).transpose(0, 1)
        t_steps, length = diffused_bb.shape[:2]
        diffused_fa = torch.full(
            (t_steps, length, 27, 3),
            float("nan"),
            dtype=coords.dtype,
            device=coords.device,
        )
        diffused_fa[:, :, :3] = diffused_bb
        if include_motif_sidechains:
            preserve = diffusion_mask[:, None] & atom_mask.to(device=coords.device, dtype=torch.bool)
            diffused_fa = torch.where(preserve[None, :, :, None], true_coords[None], diffused_fa)
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
    generator: torch.Generator | None = None,
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
                    generator=generator,
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
        delta[diffusion_mask.to(device=delta.device).bool()] = 0
    out = noisy_coords_scaled + delta[:, None]
    return out / crd_scale, delta / crd_scale


def get_noise_schedule(num_timesteps: int, initial_noise: float, final_noise: float, schedule_type: str):
    r"""
    Return a callable mapping a timestep to its noise scale.

    Args:
        num_timesteps: Total number of timesteps (for the linear interpolation).
        initial_noise: Noise scale at the initial timestep ``t = num_timesteps``.
        final_noise: Linear-interpolation endpoint at timestep ``t = 1``.
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
        "constant": lambda t: initial_noise,
        "linear": lambda t: ((t - 1) / (num_timesteps - 1)) * (initial_noise - final_noise) + final_noise,
    }
    if schedule_type not in schedules:
        raise ValueError(f"noise schedule must be one of {tuple(schedules)}")
    if schedule_type == "constant" and final_noise != initial_noise:
        raise ValueError("a constant noise schedule cannot have different initial and final values")
    return schedules[schedule_type]


class Denoiser:
    r"""
    Reverse RFdiffusion pose updater stepping ``x_t`` toward ``x_{t-1}``.

    Combines the CA-translation posterior sample and the IGSO3 frame reverse step, optionally aligns the
    prediction onto the fixed motif.

    Args:
        num_timesteps: Total number of diffusion timesteps.
        truncation_level: IGSO3 series truncation (also the design length here).
        diffuser: The forward [`Diffuser`][ofoldx.models.rfdiffusion.model.Diffuser].
        beta_initial: Initial translation beta.
        beta_final: Final translation beta.
        schedule_type: Translation beta-schedule shape.
        so3_type: Rotation diffuser family; only `"igso3"` is supported.
        noise_scale_ca: Initial CA-translation noise scale.
        final_noise_scale_ca: Optional CA-translation endpoint for a linear schedule.
        ca_noise_schedule: CA noise-schedule shape.
        noise_scale_frame: Initial frame (rotation) noise scale.
        final_noise_scale_frame: Optional frame endpoint for a linear schedule.
        frame_noise_schedule: Frame noise-schedule shape.
        crd_scale: Coordinate-scaling factor.
        partial_timesteps: Number of steps for partial diffusion, or `None`.
        guidance: Canonical soft geometric Conditions lowered to a differentiable scalar.
        guidance_scale: Overall gradient scale.
        guidance_decay: Constant, linear, quadratic, or cubic decay over reverse time.

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
        final_noise_scale_ca: float | None = None,
        ca_noise_schedule: Literal["constant", "linear"] = "constant",
        noise_scale_frame: float = 0.5,
        final_noise_scale_frame: float | None = None,
        frame_noise_schedule: Literal["constant", "linear"] = "constant",
        crd_scale: float = 1 / 15,
        partial_timesteps: int | None = None,
        guidance: RFdiffusionGuidance | None = None,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
    ) -> None:
        self.num_timesteps = integer_index(num_timesteps)
        self.truncation_level = integer_index(truncation_level)
        self.partial_timesteps = None if partial_timesteps is None else integer_index(partial_timesteps)
        if self.partial_timesteps is not None and not 0 < self.partial_timesteps <= self.num_timesteps:
            raise ValueError("partial_timesteps must be in the interval [1, T]")
        self.diffuser = diffuser
        self.beta_initial = float(beta_initial)
        self.beta_final = float(beta_final)
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.crd_scale = float(crd_scale)
        self.noise_scale_ca = float(noise_scale_ca)
        self.final_noise_scale_ca = self.noise_scale_ca if final_noise_scale_ca is None else float(final_noise_scale_ca)
        self.ca_noise_schedule = ca_noise_schedule
        self.noise_scale_frame = float(noise_scale_frame)
        self.final_noise_scale_frame = (
            self.noise_scale_frame if final_noise_scale_frame is None else float(final_noise_scale_frame)
        )
        self.frame_noise_schedule = frame_noise_schedule
        if (
            ca_noise_schedule == "constant"
            and final_noise_scale_ca is not None
            and self.final_noise_scale_ca != self.noise_scale_ca
        ):
            raise ValueError("final_noise_scale_ca can differ from noise_scale_ca only for a linear schedule")
        if (
            frame_noise_schedule == "constant"
            and final_noise_scale_frame is not None
            and self.final_noise_scale_frame != self.noise_scale_frame
        ):
            raise ValueError("final_noise_scale_frame can differ from noise_scale_frame only for a linear schedule")
        self.guidance = guidance
        self.guidance_scale = guidance_scale
        self.guidance_decay = guidance_decay
        self.schedule, _, self.alphabar_schedule = get_beta_schedule(
            self.num_timesteps,
            self.beta_initial,
            self.beta_final,
            self.schedule_type,
        )
        self.noise_schedule_ca = get_noise_schedule(
            self.num_timesteps,
            self.noise_scale_ca,
            self.final_noise_scale_ca,
            self.ca_noise_schedule,
        )
        self.noise_schedule_frame = get_noise_schedule(
            self.num_timesteps,
            self.noise_scale_frame,
            self.final_noise_scale_frame,
            self.frame_noise_schedule,
        )

    @staticmethod
    def align_to_xt_motif(pred_coords: Tensor, final_coords: Tensor, diffusion_mask: Tensor) -> Tensor:
        r"""
        Kabsch-align predicted coords onto the fixed motif residues in ``final_coords``.

        Args:
            pred_coords: Predicted coords `[L, A, 3]` to align.
            final_coords: Reference (noised) coords `[L, A, 3]` holding the fixed motif.
            diffusion_mask: Per-residue mask selecting the motif residues to align on.

        Returns:
            The aligned predicted coords `[L, A, 3]`.

        """
        output_dtype = pred_coords.dtype
        atom_mask = ~torch.isnan(pred_coords)
        working_coords = pred_coords.float()
        motif_mask = diffusion_mask.to(device=pred_coords.device, dtype=torch.bool)
        pred_coords_motif = working_coords[motif_mask, :3].reshape(-1, 3)
        noisy_coords_motif = final_coords.to(device=pred_coords.device, dtype=torch.float32)[motif_mask, :3].reshape(
            -1, 3
        )
        if pred_coords_motif.numel() == 0:
            return pred_coords
        pred_coords_mean = pred_coords_motif.mean(0)
        noisy_coords_mean = noisy_coords_motif.mean(0)
        A = noisy_coords_motif - noisy_coords_mean
        B = pred_coords_motif - pred_coords_mean
        cov = A.T @ B
        U, _S, Vt = torch.linalg.svd(cov)
        handed = torch.eye(3, dtype=torch.float32, device=pred_coords.device)
        handed[-1, -1] = torch.sign(torch.linalg.det(Vt.T @ U.T))
        rotation = Vt.T @ handed @ U.T
        aligned = (torch.nan_to_num(working_coords).reshape(-1, 3) - pred_coords_mean) @ rotation + noisy_coords_mean
        aligned = torch.where(atom_mask, aligned.reshape_as(working_coords), torch.nan)
        return aligned.to(dtype=output_dtype)

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

        """
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
            generator=generator,
        )
        if self.guidance is not None and self.guidance_scale:
            with torch.inference_mode(False):
                guided_coords = noisy_coords.detach().clone().requires_grad_(True)
                potential = self.guidance(guided_coords)
                gradient = torch.autograd.grad(potential, guided_coords)[0][:, 1]
            gradient = gradient.masked_fill(diffusion_mask[:, None], 0)
            fraction = t / self.num_timesteps
            decay = {
                "constant": 1.0,
                "linear": fraction,
                "quadratic": fraction**2,
                "cubic": fraction**3,
            }[self.guidance_decay]
            ca_deltas = ca_deltas + self.guidance_scale * decay * gradient
        frames_next = frames_next + ca_deltas[:, None]
        fullatom_next = torch.full_like(noisy_coords, float("nan")).unsqueeze(0)
        fullatom_next[:, :, :3] = frames_next[None]
        if include_motif_sidechains:
            fullatom_next[:, diffusion_mask, :14] = noisy_coords[None, diffusion_mask]
        return fullatom_next.squeeze(0)[:, :14], pred_coords


PAIR_PARAMS = {
    "DMIN": 2.0,
    "DMAX": 20.0,
    "DBINS": 36,
    "ABINS": 36,
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

    return F.one_hot(seq.long(), num_classes=num_classes).float()


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
    v = v / torch.norm(v, dim=-1, keepdim=True)
    w = w / torch.norm(w, dim=-1, keepdim=True)
    return torch.acos(torch.sum(v * w, dim=-1))


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
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
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
    dist = torch.where(torch.isnan(dist), 999.9, dist)
    coords_6d[..., 0] = dist + 999.9 * torch.eye(nres, device=coords.device)[None]
    valid = coords_6d[..., 0] < float(params["DMAX"])
    left_ca = Ca[:, :, None]
    left_cb = Cb[:, :, None]
    right_cb = Cb[:, None, :]
    right_ca = Ca[:, None, :]
    coords_6d[..., 1] = torch.where(
        valid,
        dihedral(left_ca, left_cb, right_cb, right_ca),
        0,
    )
    coords_6d[..., 2] = torch.where(
        valid,
        dihedral(N[:, :, None], left_ca, left_cb, right_cb),
        0,
    )
    coords_6d[..., 3] = torch.where(
        valid,
        planar_angle(left_ca, left_cb, right_cb),
        0,
    )
    coords_6d[..., 0] = torch.where(valid, coords_6d[..., 0], 999.9)
    mask = valid.to(dtype=coords.dtype)
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
        coords: RFdiffusion atom27 coords `[B, L, 27, 3]`.
        seq: Per-token residue ids `[B, L]` (or flattenable to it).

    Returns:
        A tuple ``(features, mask)`` of the torsion features `[B, L, 10, 2]` and the validity mask
        `[B, L, 10]`.

    """

    def normalize(value: Tensor) -> Tensor:
        return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True) + 1e-8)

    def angle_pair(left: Tensor, right: Tensor) -> Tensor:
        left = normalize(left)
        right = normalize(right)
        cosine = torch.clamp((left * right).sum(dim=-1), -1, 1)
        sine = torch.sqrt(1 - cosine.square() + 1e-8)
        return torch.stack((cosine, sine), dim=-1)

    def dihedral_pair(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        ab = normalize(a - b)
        bc = normalize(b - c)
        cd = normalize(c - d)
        first = normalize(torch.cross(ab, bc, dim=-1))
        second = normalize(torch.cross(bc, cd, dim=-1))
        sine = (torch.cross(first, bc, dim=-1) * second).sum(dim=-1)
        cosine = (first * second).sum(dim=-1)
        return torch.stack((cosine, sine), dim=-1)

    B, L = coords.shape[:2]
    seq = seq.long().reshape(B, L)
    coords = coords.clone()
    rotation, translation = rigid_from_3_points(coords[..., 0, :], coords[..., 1, :], coords[..., 2, :])
    ideal_nitrogen = coords.new_tensor((-0.5272, 1.3593, 0.0))
    ideal_carbon = coords.new_tensor((1.5233, 0.0, 0.0))
    coords[..., 0, :] = torch.einsum("...ij,j->...i", rotation, ideal_nitrogen) + translation
    coords[..., 2, :] = torch.einsum("...ij,j->...i", rotation, ideal_carbon) + translation

    features = coords.new_zeros((B, L, 10, 2))
    mask = torch.ones((B, L, 10), dtype=torch.bool, device=coords.device)
    features[:, 0, 1, 0] = 1
    features[:, -1, 0, 0] = 1
    mask[:, 0, 1] = False
    mask[:, -1, 0] = False
    if L > 1:
        features[:, :-1, 0] = dihedral_pair(coords[:, :-1, 1], coords[:, :-1, 2], coords[:, 1:, 0], coords[:, 1:, 1])
        features[:, 1:, 1] = dihedral_pair(coords[:, :-1, 2], coords[:, 1:, 0], coords[:, 1:, 1], coords[:, 1:, 2])
    features[:, :, 2] = -dihedral_pair(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2], coords[:, :, 3])

    mask[:, :, 3:7] = False
    for aa_idx, chi_defs in enumerate(CHI_ATOM_NAMES):
        aa_positions = seq == aa_idx
        atom_names = ATOM14_NAMES[aa_idx]
        for chi_idx, chi_atoms in enumerate(chi_defs):
            if chi_atoms is None:
                continue
            atom_indices = tuple(atom_names.index(atom_name) for atom_name in chi_atoms)
            chi = dihedral_pair(*(coords[:, :, atom_index] for atom_index in atom_indices))
            features[:, :, 3 + chi_idx] = torch.where(aa_positions[..., None], chi, features[:, :, 3 + chi_idx])
            mask[:, :, 3 + chi_idx] |= aa_positions
    for aa_idx, chi_idx, atom_indices in (
        (residue_token("CYS"), 1, (1, 4, 5, 18)),
        (residue_token("HIS"), 2, (7, 8, 19, 9)),
        (residue_token("PRO"), 2, (4, 5, 6, 19)),
        (residue_token("SER"), 1, (1, 4, 5, 15)),
        (residue_token("THR"), 1, (1, 4, 5, 15)),
        (residue_token("TYR"), 2, (8, 10, 11, 22)),
    ):
        aa_positions = seq == aa_idx
        chi = dihedral_pair(*(coords[:, :, atom_index] for atom_index in atom_indices))
        features[:, :, 3 + chi_idx] = torch.where(aa_positions[..., None], chi, features[:, :, 3 + chi_idx])
        mask[:, :, 3 + chi_idx] |= aa_positions

    _, _, reference_coords = rf_all_atom_kinematics()
    reference_coords = reference_coords.to(device=coords.device, dtype=coords.dtype)
    reference_nc = 0.5 * (reference_coords[:, 0, :3] + reference_coords[:, 2, :3])
    reference_ca = reference_coords[:, 1, :3]
    reference_cb = reference_coords[:, 4, :3]
    reference_cg = reference_coords[:, 5, :3]
    reference_cb_bend = angle_pair(reference_cb - reference_ca, reference_nc - reference_ca)
    reference_nc_projection = reference_coords[:, 2, :3] - reference_coords[:, 0, :3]
    reference_nc_projection = reference_nc_projection - (
        (reference_nc_projection * reference_nc).sum(dim=-1, keepdim=True)
        / reference_nc.square().sum(dim=-1, keepdim=True)
        * reference_nc
    )
    reference_cb_twist = angle_pair(reference_cb - reference_ca, reference_nc_projection)
    reference_cg_bend = angle_pair(reference_cg, reference_coords.new_tensor((-1.0, 0.0, 0.0)))

    nc = 0.5 * (coords[:, :, 0] + coords[:, :, 2])
    ca = coords[:, :, 1]
    cb = coords[:, :, 4]
    cb_bend = angle_pair(cb - ca, nc - ca)
    nc_axis = nc - ca
    nc_projection = coords[:, :, 2] - coords[:, :, 0]
    nc_projection = nc_projection - (
        (nc_projection * nc_axis).sum(dim=-1, keepdim=True) / nc_axis.square().sum(dim=-1, keepdim=True) * nc_axis
    )
    cb_twist = angle_pair(cb - ca, nc_projection)
    cg_bend = angle_pair(coords[:, :, 5] - cb, ca - cb)
    for index, (observed, reference) in enumerate(
        ((cb_bend, reference_cb_bend), (cb_twist, reference_cb_twist), (cg_bend, reference_cg_bend)),
        start=7,
    ):
        expected = reference[seq]
        features[:, :, index] = torch.stack(
            (
                (observed * expected).sum(dim=-1),
                observed[..., 0] * expected[..., 1] - observed[..., 1] * expected[..., 0],
            ),
            dim=-1,
        )

    glycine = residue_token("GLY")
    mask[:, :, 7] = seq != glycine
    mask[:, :, 8] = seq != glycine
    mask[:, :, 9] = (
        (seq != glycine) & (seq != residue_token("ALA")) & (seq != residue_token("UNK")) & (seq != MASK_TOKEN)
    )
    features[..., 0] = torch.nan_to_num(features[..., 0], nan=1.0)
    features[..., 1] = torch.nan_to_num(features[..., 1], nan=0.0)
    return features, mask


def make_alpha_t(template_coords: Tensor, seq: Tensor) -> Tensor:
    r"""
    Build the flattened ``alpha_t`` torsion tensor expected by RFdiffusion inference.

    Args:
        template_coords: Template atom27 coords `[B, T, L, 27, 3]`.
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

    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    scale = torch.sinc(angle / math.pi)
    second_order_scale = 0.5 * torch.sinc(angle / (2 * math.pi)).square()
    x, y, z = rotation_vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(rotation_vector.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + scale[..., None] * skew + second_order_scale[..., None] * (skew @ skew)


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

        """
        seq = seq.long()
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


def _sample_variable_lengths(
    batch: RFdiffusionInput,
    generator: torch.Generator | None,
) -> RFdiffusionInput:
    # Deferred to break the model<->processor import cycle after contract.py was folded in.
    from .processor import terminal_cyclization_instance

    placements = [condition for condition in batch.layout.conditions if type(condition) is RFdiffusionPlacements]
    if not placements:
        return batch
    placement = placements[0]
    rng_device = torch.device("cpu") if generator is None else generator.device

    def draw(low: int, high: int) -> int:
        if low == high:
            return low
        return int(
            torch.randint(
                low,
                high + 1,
                (),
                device=rng_device,
                generator=generator,
            ).item()
        )

    # RFdiffusion ContigMap.length (batch.layout.combined_length): the combined generated length
    # across the placement's GeneratedSegment rows -- the family-owned §3.7 topology constraint
    # threaded from the Processor. Upstream RFdiffusion re-samples the per-subcon generated lengths
    # until their inpaint total lands in the requested [min, max] window and errors if it never can
    # (contigs.py get_sampled_mask, the `while length_compatible is False` loop). We honor it the
    # same way: reject-and-resample the per-GeneratedSegment draws until their combined length
    # satisfies the window, with an upstream-style attempt cap that raises rather than silently
    # dropping the request. When it is None the single draw below is unconstrained.
    combined_length = batch.layout.combined_length
    # placement_id -> GeneratedSegment, so a whole-placement reference (spelled as the bare
    # placement_id string, e.g. in PointGroupSymmetry units) recovers its length bounds
    # without the value carrying its own id.
    segments_by_id = {
        placement_id: value
        for group in placement.placements.values()
        for placement_id, value in group.items()
        if type(value) is GeneratedSegment
    }
    chain_lengths = placement.chain_lengths or (None,) * len(placement.placements)

    def sample_segment_lengths() -> dict[str, int]:
        sampled_lengths: dict[str, int] = {}
        for group, chain_length in zip(placement.placements.values(), chain_lengths, strict=True):
            if chain_length is None:
                continue
            segments = [
                (placement_id, value) for placement_id, value in group.items() if type(value) is GeneratedSegment
            ]
            remaining = chain_length - sum(
                value.stop - value.start for value in group.values() if type(value) is ExistingSpan
            )
            for index, (placement_id, segment) in enumerate(segments):
                following = [value for _, value in segments[index + 1 :]]
                low = max(segment.min_length, remaining - sum(value.max_length for value in following))
                high = min(segment.max_length, remaining - sum(value.min_length for value in following))
                if low > high:
                    raise ValueError("placement chain length cannot be realized by its generated segments")
                length = draw(low, high)
                sampled_lengths[placement_id] = length
                remaining -= length

        symmetry = next(
            (condition for condition in batch.layout.conditions if type(condition) is PointGroupSymmetry),
            None,
        )
        if symmetry is not None:
            unit_ranges = []
            for unit in symmetry.units:
                minimum = sum(
                    sampled_lengths.get(target, segments_by_id[target].min_length) if type(target) is str else 1
                    for target in unit
                )
                maximum = sum(
                    sampled_lengths.get(target, segments_by_id[target].max_length) if type(target) is str else 1
                    for target in unit
                )
                unit_ranges.append((minimum, maximum))
            minimum = max(value[0] for value in unit_ranges)
            maximum = min(value[1] for value in unit_ranges)
            if minimum > maximum:
                raise ValueError("placement chain lengths are incompatible with PointGroupSymmetry")
            total = draw(minimum, maximum)
            for unit in symmetry.units:
                segments = [target for target in unit if type(target) is str and target not in sampled_lengths]
                remaining = (
                    total
                    - sum(type(target) is not str for target in unit)
                    - sum(
                        sampled_lengths[target] for target in unit if type(target) is str and target in sampled_lengths
                    )
                )
                for index, placement_id in enumerate(segments):
                    segment = segments_by_id[placement_id]
                    following = [segments_by_id[value] for value in segments[index + 1 :]]
                    low = max(segment.min_length, remaining - sum(value.max_length for value in following))
                    high = min(segment.max_length, remaining - sum(value.min_length for value in following))
                    if low > high:
                        raise ValueError("placement chain lengths are incompatible with PointGroupSymmetry")
                    length = draw(low, high)
                    sampled_lengths[placement_id] = length
                    remaining -= length

        # Any GeneratedSegment not pinned by a chain-length or symmetry coupling is drawn freely
        # within its own [min_length, max_length]; hoisted here (out of the reference-building loop
        # below) so the combined generated total is known before it is checked against the window.
        for group in placement.placements.values():
            for placement_id, part in group.items():
                if type(part) is GeneratedSegment and placement_id not in sampled_lengths:
                    sampled_lengths[placement_id] = draw(part.min_length, part.max_length)
        return sampled_lengths

    # A GeneratedSegment always realizes within [min_length, max_length], so the combined generated
    # total is always within these summed bounds regardless of the per-chain / symmetry couplings;
    # if the requested window misses this envelope it can never be satisfied -- report that directly
    # instead of spinning the resampler to the cap.
    reachable_low = sum(segment.min_length for segment in segments_by_id.values())
    reachable_high = sum(segment.max_length for segment in segments_by_id.values())
    if combined_length is not None and (combined_length[1] < reachable_low or combined_length[0] > reachable_high):
        raise ValueError(
            f"combined_length {combined_length} cannot be realized: the placement's generated segments "
            f"admit a combined generated length in [{reachable_low}, {reachable_high}]"
        )
    max_attempts = 100000  # matches upstream RFdiffusion get_sampled_mask
    sampled_lengths = sample_segment_lengths()
    if combined_length is not None:
        attempt = 1
        while not combined_length[0] <= sum(sampled_lengths[pid] for pid in segments_by_id) <= combined_length[1]:
            if attempt >= max_attempts:
                raise ValueError(
                    f"combined_length {combined_length} could not be satisfied after {max_attempts} "
                    f"resampling attempts of the placement's generated segment lengths"
                )
            sampled_lengths = sample_segment_lengths()
            attempt += 1

    output_residues: list[tuple[str, int] | SymbolicResidueSite] = []
    output_instances: list[str] = []
    # Frozen §3.7 canonical realized-instance-id convention (W6 caveat b): mint the realized
    # OUTPUT instance ids with the shared `InstanceIdAllocator`, walking declaration groups in
    # order and calling `allocate()` once per realized output chain with >=1 occurrence — the
    # canonical `chain_{n}` (uniform, 0-based, no family prefix). A declaration group that
    # realizes to zero residues mints no chain and consumes no index. Identical to the processor
    # lowering in `_resolve_placement`.
    allocator = InstanceIdAllocator()
    for group in placement.placements.values():
        group_residues: list[tuple[str, int] | SymbolicResidueSite] = []
        for placement_id, part in group.items():
            if type(part) is ExistingSpan:
                references = [(part.instance_id, index) for index in range(part.start, part.stop)]
            else:
                length = sampled_lengths[placement_id]
                references = [SymbolicResidueSite(placement_id, index) for index in range(length)]
            group_residues.extend(references)
        if not group_residues:
            continue  # zero-length declaration group mints no realized chain, consumes no index
        instance_id = allocator.allocate()
        output_residues.extend(group_residues)
        output_instances.extend([instance_id] * len(group_residues))

    references = tuple(output_residues)
    instances = tuple(output_instances)
    if references == batch.layout.output_residues and instances == batch.layout.output_instance_ids:
        return batch
    if batch.layout.templates:
        raise ValueError("variable generated lengths cannot be combined with aligned Template evidence")

    old_index = {reference: index for index, reference in enumerate(batch.layout.output_residues)}
    length = len(references)
    coords = batch.noisy_coords.new_empty((1, length, 14, 3))
    sequence = batch.sequence.new_zeros((1, length, 22))
    sequence[..., MASK_TOKEN] = 1
    fixed_structure = torch.zeros((1, length), dtype=torch.bool, device=batch.noisy_coords.device)
    fixed_sequence = torch.zeros_like(fixed_structure)
    hotspots = torch.zeros_like(fixed_structure)
    for row, reference in enumerate(references):
        previous = old_index.get(reference)
        if previous is None:
            coords[0, row] = float("nan")
            continue
        coords[0, row] = batch.noisy_coords[0, previous]
        sequence[0, row] = batch.sequence[0, previous]
        fixed_structure[0, row] = batch.fixed_structure_mask[0, previous]
        fixed_sequence[0, row] = batch.fixed_sequence_mask[0, previous]
        hotspots[0, row] = batch.hotspot_mask[0, previous]
    for row, reference in enumerate(references):
        if type(reference) is SymbolicResidueSite:
            sequence[0, row].zero_()
            sequence[0, row, MASK_TOKEN] = 1
            fixed_sequence[0, row] = False
    residue_indices: list[int] = []
    previous_instance = None
    chain_index = -1
    chain_start = 0
    for row, instance_id in enumerate(instances):
        if instance_id != previous_instance:
            chain_index += 1
            chain_start = row
            previous_instance = instance_id
        residue_indices.append(200 * chain_index + row - chain_start)
    row_by_reference = {reference: row for row, reference in enumerate(references)}
    row_by_reference.update(
        {alias: row_by_reference[reference] for alias, reference in symbolic_residue_aliases(references).items()}
    )
    for condition in batch.layout.conditions:
        if type(condition) is not AllowedResidueIdentities:
            continue
        token = residue_token(next(iter(condition.component_ids)))
        rows: list[int] = []
        for target in condition.positions:
            if type(target) is str:
                rows.extend(
                    row
                    for row, reference in enumerate(references)
                    if type(reference) is SymbolicResidueSite and reference.placement_id == target
                )
            else:
                rows.append(row_by_reference[target])
        for row in rows:
            sequence[0, row].zero_()
            sequence[0, row, token] = 1
            fixed_sequence[0, row] = True
    secondary_labels = torch.full((1, length), 3, dtype=torch.int64, device=coords.device)
    label_by_kind = {"helix": 0, "strand": 1, "loop": 2}
    for condition in batch.layout.conditions:
        if type(condition) is not SecondaryStructure:
            continue
        rows: list[int] = []
        for target in condition.targets:
            if type(target) is str:
                rows.extend(
                    row
                    for row, reference in enumerate(references)
                    if type(reference) is SymbolicResidueSite and reference.placement_id == target
                )
            else:
                rows.append(row_by_reference[target])
        secondary_labels[0, torch.tensor(rows, device=coords.device)] = label_by_kind[condition.kind]
    cyclic_mask = torch.zeros((1, length), dtype=torch.bool, device=coords.device)
    row_by_reference_for_cyclization = dict(row_by_reference)
    for condition in batch.layout.conditions:
        if type(condition) is not SymbolicBond:
            continue
        instance_id = terminal_cyclization_instance(
            condition,
            row_by_reference_for_cyclization,
            instances,
        )
        cyclic_mask[0] |= torch.tensor(
            [value == instance_id for value in instances],
            device=coords.device,
        )
    return RFdiffusionInput(
        noisy_coords=coords,
        sequence=sequence,
        residue_index=torch.tensor(residue_indices, dtype=torch.int64, device=coords.device)[None],
        fixed_structure_mask=fixed_structure,
        fixed_sequence_mask=fixed_sequence,
        hotspot_mask=hotspots,
        cyclic_mask=cyclic_mask,
        secondary_structure=F.one_hot(
            secondary_labels,
            num_classes=4,
        ).to(coords.dtype),
        block_adjacency=F.one_hot(
            torch.full((1, length, length), 2, dtype=torch.int64, device=coords.device),
            num_classes=3,
        ).to(coords.dtype),
        timestep=batch.timestep,
        layout=RFdiffusionLayout(
            system=batch.layout.system,
            structure=batch.layout.structure,
            templates=batch.layout.templates,
            conditions=batch.layout.conditions,
            output_residues=references,
            output_instance_ids=instances,
            combined_length=batch.layout.combined_length,
        ),
    )


def _resolved_placement_condition(layout: RFdiffusionLayout) -> RFdiffusionPlacements | None:
    r"""The RFdiffusionPlacements that drove this layout, for placement_map materialization.

    An explicit :class:`RFdiffusionPlacements` condition is used directly; otherwise the
    trivial all-existing default placement is reconstructed from the retained input system
    exactly as :meth:`RFdiffusionProcessor.__call__` synthesizes it (each protein input
    instance -> one whole-chain ``ExistingSpan`` under its own declaration group), so the
    default protein-input path also carries a placement_map. ``None`` when there is neither a
    placements condition nor a molecular input.
    """

    for condition in layout.conditions:
        if type(condition) is RFdiffusionPlacements:
            return condition
    system = layout.system
    if system is None:
        return None
    placements: dict[str, dict[str, ExistingSpan]] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if type(entity) is PolymerSequence and entity.polymer_type == "protein":
            placements[instance_id] = {instance_id: ExistingSpan(instance_id, 0, len(entity))}
    return RFdiffusionPlacements(placements=placements) if placements else None


def _decode_design(
    samples: Iterable[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
    inputs: list[RFdiffusionInput],
    *,
    align_motif: bool,
) -> RFdiffusionDesignResult:
    # RFdiffusion is coordinate-producing and variable-topology, so each sample is wired into
    # the family-owned standalone RFdiffusionDesignCandidate/RFdiffusionDesignResult (results.py),
    # carrying the decoded Structure and per-sample values, rather than reusing or widening the
    # frozen coordinate-free shared DesignResult, which §5.3.1/§5.3.3 forbids. Diffusion
    # internals (timesteps / px0 / x_t_minus_1) are dropped as §5.1 trajectory diagnostics.
    candidates: list[RFdiffusionDesignCandidate] = []
    for sample, batch in zip(samples, inputs, strict=True):
        raw_coordinates, sequence, _timesteps, self_consistency_plddt, _px0, _xt_minus_1 = sample
        coordinates = raw_coordinates.clone()
        fixed = batch.fixed_structure_mask[0].to(device=coordinates.device, dtype=torch.bool)
        if bool(fixed.any()):
            reference = batch.noisy_coords[0].to(device=coordinates.device, dtype=coordinates.dtype)
            if align_motif:
                coordinates = Denoiser.align_to_xt_motif(coordinates, reference, fixed)
            coordinates[fixed] = reference[fixed]
        layout = batch.layout
        candidate, _identity_map, residue_map = _decode_structure(
            coordinates,
            sequence,
            batch.fixed_sequence_mask[0],
            layout,
        )
        candidate_outcomes = _condition_outcomes(candidate, layout, residue_map)
        _require_hard_conditions(layout, candidate_outcomes)
        # self_consistency_plddt is [steps, L]; the final step is the per-residue design pLDDT,
        # aligned row-for-row to the output residues (residue_map keys are the ComponentReference
        # / SymbolicResidueSite identities, its values the realized output ComponentReferences).
        # The released expectation lies on (0, 1] (bins linspace(1/num_bins, 1, num_bins)); the
        # decode multiplies by 100 so the §5.1 cross-family reserved `plddt` name carries its
        # reserved 0-100 scale. This is a mapping requirement applied only in decode -- not a
        # change to the parity-locked sampler/network.
        plddt = self_consistency_plddt[-1].detach().to(device="cpu", dtype=torch.float32).numpy() * 100.0
        plddt_axis = tuple(residue_map[reference] for reference in layout.output_residues)
        # W6 caveat a: adopt the shared grouped placement_map as the per-candidate OUTPUT
        # identity (losslessly reconstructed from the flat output_residues + residue_map;
        # exhaustive coverage enforced inside materialize_placement_map). The realized ids in
        # its ComponentReference[0] values are this candidate's Structure.system instance ids.
        from .processor import materialize_placement_map

        placement_condition = _resolved_placement_condition(layout)
        placement_map = (
            {}
            if placement_condition is None
            else materialize_placement_map(placement_condition, layout.output_residues, residue_map)[0]
        )
        candidates.append(
            RFdiffusionDesignCandidate(
                structure=candidate,
                placement_map=placement_map,
                scores={
                    "mean_plddt": Score(
                        value=float(plddt.mean()),
                        unit="0-100",
                        preferred="higher",
                    )
                },
                metrics={"plddt": Metric(values=plddt, axis=plddt_axis, unit="0-100")},
            )
        )
    return RFdiffusionDesignResult(candidates=tuple(candidates))


def _decode_structure(
    coords: Tensor,
    sequence: Tensor,
    fixed_sequence_mask: Tensor,
    layout: RFdiffusionLayout,
) -> tuple[
    Structure,
    tuple[tuple[SymbolicResidueSite, tuple[str, int]], ...],
    dict[tuple[str, int] | SymbolicResidueSite, tuple[str, int]],
]:
    source = layout.system
    rows_by_instance: dict[str, list[tuple[int, tuple[str, int] | SymbolicResidueSite]]] = {}
    for row, (reference, instance_id) in enumerate(zip(layout.output_residues, layout.output_instance_ids)):
        rows_by_instance.setdefault(instance_id, []).append((row, reference))

    aliases = symbolic_residue_aliases(layout.output_residues)
    allowed_references: set[tuple[str, int] | SymbolicResidueSite] = set()
    for condition in layout.conditions:
        if type(condition) is not AllowedResidueIdentities:
            continue
        for target in condition.positions:
            if type(target) is str:
                for reference in layout.output_residues:
                    if type(reference) is SymbolicResidueSite and reference.placement_id == target:
                        allowed_references.add(reference)
            else:
                allowed_references.add(aliases.get(target, target))

    residue_map: dict[tuple[str, int] | SymbolicResidueSite, tuple[str, int]] = {}
    output_sequences: dict[str, tuple[ChemicalComponent, ...]] = {}
    for instance_id, rows in rows_by_instance.items():
        residue_components: list[ChemicalComponent] = []
        for component_index, (row, reference) in enumerate(rows):
            token = int(sequence[row])
            if (
                type(reference) is not SymbolicResidueSite
                and bool(fixed_sequence_mask[row])
                and reference not in allowed_references
            ):
                entity = source.entities[source.instances[reference[0]]]  # type: ignore[union-attr]
                residue_components.append(entity.components[reference[1]])
            else:
                residue_components.append(get_component("UNK" if token == MASK_TOKEN else NUM2AA[token]))
            residue_map[reference] = (instance_id, component_index)
        output_sequences[instance_id] = tuple(residue_components)

    residue_map.update({alias: residue_map[reference] for alias, reference in aliases.items()})

    touched_instances = {reference[0] for reference in layout.output_residues if type(reference) is tuple}
    retained_instance_ids = (
        []
        if source is None
        else [instance_id for instance_id in source.instances if instance_id not in touched_instances]
    )
    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    entity_ids: set[str] = set()
    if source is not None:
        for instance_id in retained_instance_ids:
            entity_id = source.instances[instance_id]
            entities[entity_id] = source.entities[entity_id]
            instances[instance_id] = entity_id
            entity_ids.add(entity_id)
    for instance_id, residue_components in output_sequences.items():
        base = f"{instance_id}_rfdesign"
        entity_id = base
        suffix = 2
        while entity_id in entity_ids:
            entity_id = f"{base}_{suffix}"
            suffix += 1
        entity_ids.add(entity_id)
        entities[entity_id] = PolymerSequence("protein", residue_components)
        instances[instance_id] = entity_id

    valid_instance_ids = set(instances)
    bonds: set[SystemBond] = set()
    if source is not None:
        for bond in source.bonds:
            remapped = _remap_bond(bond, residue_map, touched_instances)
            if remapped.site_a.instance_id in valid_instance_ids and remapped.site_b.instance_id in valid_instance_ids:
                bonds.add(remapped)
    for condition in layout.conditions:
        if type(condition) is not SymbolicBond:
            continue
        endpoints: list[AtomSite] = []
        for reference in (condition.site_a, condition.site_b):
            if type(reference) is SymbolicAtomSite:
                instance_id, component_index = residue_map[reference.residue]
                endpoints.append(AtomSite(instance_id, component_index, reference.atom_id))
            else:
                mapped = residue_map.get((reference.instance_id, reference.component_index))
                endpoints.append(reference if mapped is None else AtomSite(mapped[0], mapped[1], reference.atom_id))
        bonds.add(SystemBond(endpoints[0], endpoints[1], origin="user"))
    system = MolecularSystem(entities, instances, frozenset(bonds))

    sites: list[AtomSite] = []
    coordinate_rows: list[np.ndarray] = []
    masks: list[bool] = []
    coords_cpu = coords.detach().to(device="cpu", dtype=torch.float32).numpy()
    for row, reference in enumerate(layout.output_residues):
        instance_id, component_index = residue_map[reference]
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            sites.append(AtomSite(instance_id, component_index, atom_name))
            value = np.asarray(coords_cpu[row, atom_index], dtype=np.float32)
            present = bool(np.isfinite(value).all())
            coordinate_rows.append(value if present else np.zeros(3, dtype=np.float32))
            masks.append(present)

    site_set = set(sites)
    if layout.structure is not None:
        source_index = {site: index for index, site in enumerate(layout.structure.sites)}
        for condition in layout.conditions:
            if type(condition) is not ReferenceCoordinates:
                continue
            for original in condition.sites:
                mapped_residue = residue_map[(original.instance_id, original.component_index)]
                mapped = AtomSite(mapped_residue[0], mapped_residue[1], original.atom_id)
                if mapped in site_set:
                    continue
                source_row = source_index[original]
                sites.append(mapped)
                site_set.add(mapped)
                coordinate_rows.append(np.asarray(layout.structure.coordinates[source_row], dtype=np.float32))
                masks.append(True)

    if layout.structure is not None and source is not None:
        retained_ids = set(retained_instance_ids)
        for index, site in enumerate(layout.structure.sites):
            if site.instance_id not in retained_ids or site in sites:
                continue
            sites.append(site)
            coordinate_rows.append(np.asarray(layout.structure.coordinates[index], dtype=np.float32))
            masks.append(bool(layout.structure.coordinate_mask[index]))
    structure = Structure(
        system=system,
        sites=tuple(sites),
        coordinates=np.asarray(coordinate_rows, dtype=np.float32),
        coordinate_mask=np.asarray(masks, dtype=np.bool_),
    )
    identity_map = tuple(
        (reference, residue_map[reference])
        for reference in layout.output_residues
        if type(reference) is SymbolicResidueSite
    )
    return structure, identity_map, residue_map


def _require_hard_conditions(
    layout: RFdiffusionLayout,
    outcomes: tuple[tuple[bool | None, float | None, str | None], ...],
) -> None:
    for index, (condition, outcome) in enumerate(zip(layout.conditions, outcomes, strict=True)):
        # The shared FROZEN §3.7.1/§3.7.4 records carry no ``.intent`` field -- hard/soft is
        # family-owned and rides on the family envelope -- so RFdiffusion pins each shared record's
        # fixed intent here instead of reading a ``.intent`` it does not have. The identity record
        # (AllowedResidueIdentities), the covalent link (SymbolicBond, collapsed to cyclic_mask),
        # and the single-sided distance-free target-conditioning Hotspot (realized as hotspot_mask)
        # are all consumed as HARD: a shared Hotspot IS a hard positive hotspot constraint, which is
        # the whole meaning of the frozen §3.7.4 label. The geometric Contact is a family-scorer-only
        # measurement with no released input-side potential, so it is never enforced. Every
        # family-owned spatial record still carries its own ``.intent`` (unchanged behavior).
        if type(condition) is Contact:
            is_hard = False
        elif type(condition) in (AllowedResidueIdentities, Hotspot, SymbolicBond):
            is_hard = True
        else:
            is_hard = condition.intent == "hard"
        # Raise only on a MEASURED hard violation. An unmeasured outcome is ``None`` -- the
        # conditioning-only Hotspot biases sampling through hotspot_mask but its satisfaction is
        # never verified post-hoc, so it cannot be asserted as a violation. Every family hard
        # condition and the measured shared records (AllowedResidueIdentities / SymbolicBond) report
        # a concrete bool here, so ``is False`` preserves their existing raise behavior exactly.
        if is_hard and outcome[0] is False:
            raise RuntimeError(f"RFdiffusion violated hard conditions[{index}] during result decoding")


def _remap_bond(
    bond: SystemBond,
    residue_map: Mapping[tuple[str, int] | SymbolicResidueSite, tuple[str, int]],
    touched_instances: set[str],
) -> SystemBond:
    def site(value: AtomSite) -> AtomSite:
        if value.instance_id not in touched_instances:
            return value
        mapped = residue_map.get((value.instance_id, value.component_index))
        if mapped is None:
            raise ValueError("placement omits a residue participating in an input system bond")
        return AtomSite(mapped[0], mapped[1], value.atom_id)

    stereo_sites = None if bond.stereo_sites is None else tuple(site(value) for value in bond.stereo_sites)
    return SystemBond(
        site_a=site(bond.site_a),
        site_b=site(bond.site_b),
        order=bond.order,
        aromatic=bond.aromatic,
        stereo=bond.stereo,
        stereo_sites=stereo_sites,
        origin=bond.origin,
    )


def _condition_outcomes(
    candidate: Structure,
    layout: RFdiffusionLayout,
    residue_map: Mapping[tuple[str, int] | SymbolicResidueSite, tuple[str, int]],
) -> tuple[tuple[bool | None, float | None, str | None], ...]:
    # Deferred to break the model<->processor import cycle after contract.py was folded in.
    from .processor import rfdiffusion_secondary_structure

    # TODO(§3.7↔§5 reopened): per-condition constraint measurement. Each (satisfied, value,
    # unit) triple below carries a real geometric measurement (angstrom min-distance / RMSD /
    # coordination / helix fraction) that has no frozen per-condition home; only `satisfied`
    # is consumed here, to drive the hard-condition raise. Await a §3.7↔§5 sub-freeze to
    # surface value/unit under their actual Score/Metric names.
    coordinate_index = {site: index for index, site in enumerate(candidate.sites)}
    source_index = (
        {} if layout.structure is None else {site: index for index, site in enumerate(layout.structure.sites)}
    )
    values: list[tuple[bool | None, float | None, str | None]] = []
    for condition in layout.conditions:
        if type(condition) is AllowedResidueIdentities:
            references: list[tuple[str, int] | SymbolicResidueSite] = []
            for target in condition.positions:
                if type(target) is str:
                    references.extend(
                        reference
                        for reference in layout.output_residues
                        if type(reference) is SymbolicResidueSite and reference.placement_id == target
                    )
                else:
                    references.append(target)
            component_ids = []
            for reference in references:
                instance_id, component_index = residue_map[reference]
                entity = candidate.system.entities[candidate.system.instances[instance_id]]
                component_ids.append(entity.components[component_index].component_id)
            values.append(
                (
                    all(component_id in condition.component_ids for component_id in component_ids),
                    None,
                    None,
                )
            )
            continue
        if type(condition) is RFdiffusionPlacements:
            values.append((True, None, None))
            continue
        if type(condition) is ReferenceCoordinates and layout.structure is not None:
            original_coordinates: list[np.ndarray] = []
            candidate_coordinates: list[np.ndarray] = []
            complete = True
            for original in condition.sites:
                system = layout.system
                entity = system.entities[system.instances[original.instance_id]]  # type: ignore[union-attr]
                if type(entity) is Molecule:
                    mapped = original
                else:
                    mapped_residue = residue_map.get((original.instance_id, original.component_index))
                    mapped = (
                        None
                        if mapped_residue is None
                        else AtomSite(mapped_residue[0], mapped_residue[1], original.atom_id)
                    )
                old_index = source_index.get(original)
                new_index = None if mapped is None else coordinate_index.get(mapped)
                if (
                    old_index is None
                    or new_index is None
                    or not bool(layout.structure.coordinate_mask[old_index])
                    or not bool(candidate.coordinate_mask[new_index])
                ):
                    complete = False
                    break
                original_coordinates.append(layout.structure.coordinates[old_index])
                candidate_coordinates.append(candidate.coordinates[new_index])
            distances = (
                _proper_alignment_distances(
                    np.asarray(candidate_coordinates, dtype=np.float64),
                    np.asarray(original_coordinates, dtype=np.float64),
                )
                if complete
                else None
            )
            maximum = None if distances is None else float(distances.max())
            values.append(
                (
                    maximum is not None and maximum <= condition.tolerance_angstrom,
                    maximum,
                    "angstrom",
                )
            )
            continue
        if type(condition) is Distance:
            endpoints = []
            for reference in (condition.site_a, condition.site_b):
                if type(reference) is SymbolicAtomSite:
                    mapped_residue = residue_map.get(reference.residue)
                    site = (
                        None
                        if mapped_residue is None
                        else AtomSite(mapped_residue[0], mapped_residue[1], reference.atom_id)
                    )
                else:
                    mapped_residue = residue_map.get((reference.instance_id, reference.component_index))
                    site = (
                        reference
                        if mapped_residue is None
                        else AtomSite(mapped_residue[0], mapped_residue[1], reference.atom_id)
                    )
                coordinate = None if site is None else coordinate_index.get(site)
                endpoints.append(
                    None
                    if coordinate is None or not bool(candidate.coordinate_mask[coordinate])
                    else candidate.coordinates[coordinate]
                )
            if endpoints[0] is None or endpoints[1] is None:
                values.append((None, None, None))
            else:
                distance = float(np.linalg.norm(endpoints[0] - endpoints[1]))
                lower = condition.lower_angstrom - condition.tolerance_angstrom
                upper = condition.upper_angstrom + condition.tolerance_angstrom
                values.append((lower <= distance <= upper, distance, "angstrom"))
            continue
        if type(condition) is Coordination:
            side_a = _contact_coordinates(condition.side_a, candidate, residue_map, coordinate_index)
            side_b = _contact_coordinates(condition.side_b, candidate, residue_map, coordinate_index)
            if not side_a or not side_b:
                values.append((None, None, None))
            else:
                distances = np.linalg.norm(
                    np.asarray(side_a, dtype=np.float32)[:, None] - np.asarray(side_b, dtype=np.float32)[None],
                    axis=-1,
                )
                scaled = (distances - condition.center_angstrom) / condition.scale_angstrom
                coordination = float((1 / (1 + scaled**6)).sum())
                values.append((None, coordination, None))
            continue
        if type(condition) is RadiusOfGyration:
            coordinates = _contact_coordinates(
                condition.targets,
                candidate,
                residue_map,
                coordinate_index,
            )
            if not coordinates:
                values.append((None, None, None))
            else:
                array = np.asarray(coordinates, dtype=np.float32)
                radius = float(np.sqrt(np.square(array - array.mean(axis=0)).sum(axis=-1).mean()))
                values.append(
                    (
                        radius <= condition.upper_angstrom,
                        radius,
                        "angstrom",
                    )
                )
            continue
        if type(condition) is SubstrateInteraction:
            residues = _contact_coordinates(
                condition.residues,
                candidate,
                residue_map,
                coordinate_index,
            )
            substrate = _contact_coordinates(
                condition.substrate_atoms,
                candidate,
                residue_map,
                coordinate_index,
            )
            if not residues or not substrate:
                values.append((None, None, None))
            else:
                nearest = np.linalg.norm(
                    np.asarray(residues, dtype=np.float32)[:, None] - np.asarray(substrate, dtype=np.float32)[None],
                    axis=-1,
                ).min(axis=-1)
                scaled = (nearest - condition.center_angstrom) / condition.scale_angstrom
                attraction = condition.attraction_strength / (1 + scaled**6)
                exponent = 1.5
                repulsion = (
                    condition.repulsion_slope**2
                    * np.maximum(condition.repulsion_cutoff_angstrom - nearest, 0) ** exponent
                    / (exponent * condition.repulsion_cutoff_angstrom ** (exponent - 1))
                )
                values.append((None, float((attraction - repulsion).sum()), None))
            continue
        if type(condition) is SecondaryStructure:
            ca_coordinates = []
            for reference in layout.output_residues:
                mapped = residue_map[reference]
                coordinate = coordinate_index[AtomSite(mapped[0], mapped[1], "CA")]
                ca_coordinates.append(candidate.coordinates[coordinate])
            labels = rfdiffusion_secondary_structure(torch.as_tensor(np.asarray(ca_coordinates)))
            expected = {"helix": 0, "strand": 1, "loop": 2}[condition.kind]
            target_rows = []
            for target in condition.targets:
                if type(target) is str:
                    target_rows.extend(
                        row
                        for row, reference in enumerate(layout.output_residues)
                        if type(reference) is SymbolicResidueSite and reference.placement_id == target
                    )
                else:
                    target_rows.append(layout.output_residues.index(target))
            fraction = float((labels[target_rows] == expected).to(torch.float32).mean())
            values.append((fraction == 1.0, fraction, "fraction"))
            continue
        if type(condition) is PointGroupSymmetry:
            symmetry = RFdiffusionSymmetry(condition, layout)
            units: list[np.ndarray] = []
            complete = True
            for rows in symmetry.units:
                unit = []
                for row in rows:
                    mapped = residue_map[layout.output_residues[row]]
                    residue_atoms = []
                    for atom_name in ("N", "CA", "C"):
                        coordinate = coordinate_index.get(AtomSite(mapped[0], mapped[1], atom_name))
                        if coordinate is None or not bool(candidate.coordinate_mask[coordinate]):
                            complete = False
                            break
                        residue_atoms.append(candidate.coordinates[coordinate])
                    if not complete:
                        break
                    unit.append(residue_atoms)
                if not complete:
                    break
                units.append(np.asarray(unit, dtype=np.float64))
            if complete:
                observed = np.asarray(units, dtype=np.float64)
                rotations = np.stack([rotation.cpu().numpy() for rotation in symmetry.rotations])
                rmsd = _point_group_rmsd(observed, rotations)
            else:
                rmsd = None
            values.append(
                (
                    rmsd is not None and rmsd <= condition.tolerance_angstrom,
                    rmsd,
                    "angstrom",
                )
            )
            continue
        if type(condition) is SymbolicBond:
            endpoint_sites = []
            for reference in (condition.site_a, condition.site_b):
                if type(reference) is SymbolicAtomSite:
                    mapped = residue_map[reference.residue]
                    endpoint_sites.append(AtomSite(mapped[0], mapped[1], reference.atom_id))
                else:
                    mapped = residue_map.get((reference.instance_id, reference.component_index))
                    endpoint_sites.append(
                        reference if mapped is None else AtomSite(mapped[0], mapped[1], reference.atom_id)
                    )
            bond = any({value.site_a, value.site_b} == set(endpoint_sites) for value in candidate.system.bonds)
            values.append((bond, None, None))
            continue
        if type(condition) is Contact:
            side_a = _contact_coordinates(condition.side_a, candidate, residue_map, coordinate_index)
            side_b = _contact_coordinates(condition.side_b, candidate, residue_map, coordinate_index)
            if side_a and side_b:
                distances = np.linalg.norm(
                    np.asarray(side_a, dtype=np.float32)[:, None] - np.asarray(side_b, dtype=np.float32)[None],
                    axis=-1,
                )
                minimum = float(distances.min())
                values.append(
                    (
                        minimum <= condition.cutoff_angstrom + condition.tolerance_angstrom,
                        minimum,
                        "angstrom",
                    )
                )
            else:
                values.append((None, None, None))
            continue
        values.append((None, None, None))
    return tuple(values)


def _proper_alignment_distances(moving: np.ndarray, target: np.ndarray) -> np.ndarray:
    if np.array_equal(moving, target):
        return np.zeros(len(moving), dtype=np.float64)
    moving_centered = moving - moving.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    left, _singular, right = np.linalg.svd(moving_centered.T @ target_centered)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    aligned = moving_centered @ rotation + target.mean(axis=0)
    return np.linalg.norm(aligned - target, axis=-1)


def _point_group_rmsd(observed: np.ndarray, rotations: np.ndarray) -> float:
    center = observed.reshape(-1, 3).mean(axis=0)

    def residual(parameters: np.ndarray) -> np.ndarray:
        frame = Rotation.from_rotvec(parameters[:3]).as_matrix()
        origin = parameters[3:]
        base = observed[0] - origin
        predicted = np.stack([base @ frame @ rotation.T @ frame.T + origin for rotation in rotations])
        return (predicted - observed).reshape(-1)

    starts = (
        np.zeros(3),
        np.asarray([math.pi / 2, 0.0, 0.0]),
        np.asarray([0.0, math.pi / 2, 0.0]),
        np.asarray([0.0, 0.0, math.pi / 2]),
        np.asarray([math.pi, 0.0, 0.0]),
        np.asarray([0.0, math.pi, 0.0]),
        np.asarray([0.0, 0.0, math.pi]),
    )
    fitted = least_squares(residual, np.concatenate((starts[0], center)))
    best = residual(fitted.x).reshape(-1, 3)
    for frame in starts[1:]:
        fitted = least_squares(residual, np.concatenate((frame, center)))
        values = residual(fitted.x).reshape(-1, 3)
        if np.square(values).sum() < np.square(best).sum():
            best = values
    return float(np.sqrt(np.square(best).sum(axis=-1).mean()))


def _contact_coordinates(
    references: tuple[object, ...],
    structure: Structure,
    residue_map: Mapping[tuple[str, int] | SymbolicResidueSite, tuple[str, int]],
    coordinate_index: Mapping[AtomSite, int],
) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for reference in references:
        mapped_sites: list[AtomSite] = []
        if type(reference) in (tuple, SymbolicResidueSite):
            mapped = residue_map.get(reference)
            if mapped is not None:
                coordinate = _residue_contact_coordinate(structure, mapped, coordinate_index)
                if coordinate is not None:
                    values.append(coordinate)
        elif type(reference) is str:
            for symbolic, mapped in residue_map.items():
                if type(symbolic) is SymbolicResidueSite and symbolic.placement_id == reference:
                    coordinate = _residue_contact_coordinate(structure, mapped, coordinate_index)
                    if coordinate is not None:
                        values.append(coordinate)
        elif type(reference) is SymbolicAtomSite:
            mapped = residue_map.get(reference.residue)
            if mapped is not None:
                mapped_sites.append(AtomSite(mapped[0], mapped[1], reference.atom_id))
        else:
            mapped = residue_map.get((reference.instance_id, reference.component_index))
            mapped_sites.append(reference if mapped is None else AtomSite(mapped[0], mapped[1], reference.atom_id))
        for site in mapped_sites:
            index = coordinate_index.get(site)
            if index is not None and bool(structure.coordinate_mask[index]):
                values.append(structure.coordinates[index])
    return values


def _residue_contact_coordinate(
    structure: Structure,
    residue: tuple[str, int],
    coordinate_index: Mapping[AtomSite, int],
) -> np.ndarray | None:
    entity = structure.system.entities[structure.system.instances[residue[0]]]
    if entity.components[residue[1]].component_id == "GLY":
        index = coordinate_index.get(AtomSite(residue[0], residue[1], "CA"))
        return None if index is None else structure.coordinates[index]
    indices = [coordinate_index.get(AtomSite(residue[0], residue[1], atom_name)) for atom_name in ("N", "CA", "C")]
    if any(index is None or not bool(structure.coordinate_mask[index]) for index in indices):
        return None
    nitrogen, calpha, carbon = (structure.coordinates[index] for index in indices)  # type: ignore[index]
    b = calpha - nitrogen
    c = carbon - calpha
    return -0.58273431 * np.cross(b, c) + 0.56802827 * b - 0.54067466 * c + calpha


def symbolic_residue_aliases(
    references: tuple[ComponentReference | SymbolicResidueSite, ...],
) -> dict[SymbolicResidueSite, SymbolicResidueSite]:
    r"""Map canonical negative segment offsets to the concrete positive trajectory identities."""

    segments: dict[str, list[SymbolicResidueSite]] = {}
    for reference in references:
        if type(reference) is SymbolicResidueSite:
            segments.setdefault(reference.placement_id, []).append(reference)
    return {
        SymbolicResidueSite(placement_id, index - len(rows)): reference
        for placement_id, rows in segments.items()
        for index, reference in enumerate(rows)
    }


# ---------------------------------------------------------------------------
# Folded from rfdiffusion/contract.py: RFdiffusion's exact tensor boundary and
# semantic layout.  Records moved verbatim; the imports they need are declared
# at the top of this module.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RFdiffusionLayout:
    r"""Canonical identities required to realize denoised tensor rows.

    ``output_residues[i]`` identifies row ``i`` as either an existing canonical
    residue or one symbolic residue declared by the input placement.  Chain and
    instance rows are explicit because RFdiffusion's internal residue numbers are
    not canonical identities.  This is the only non-tensor leaf of
    :class:`RFdiffusionInput` and is deliberately ignored by the denoiser graph.
    """

    system: MolecularSystem | None
    structure: Structure | None
    templates: tuple[Template, ...]
    conditions: tuple[
        AllowedResidueIdentities
        | Contact
        | Hotspot
        | RFdiffusionPlacements
        | Coordination
        | Distance
        | PointGroupSymmetry
        | RadiusOfGyration
        | ReferenceCoordinates
        | SecondaryStructure
        | SubstrateInteraction
        | SymbolicBond,
        ...,
    ]
    output_residues: tuple[ComponentReference | SymbolicResidueSite, ...]
    output_instance_ids: tuple[str, ...]
    # Family-owned §3.7 topology argument (RFdiffusion ContigMap.length): the combined
    # generated length bounds across all inpaint chains, excluding receptor ExistingSpan
    # rows. This is a concrete family operation argument (distinct from ODesign per-chain
    # lengths and BoltzGen total_len), NOT a shared placement-core invariant. Defaulted None so the
    # denoiser graph -- which ignores this non-tensor leaf -- is unaffected when absent.
    combined_length: tuple[int, int] | None = None


@dataclass(kw_only=True, eq=False)
class RFdiffusionInput:
    r"""One compile-friendly RFdiffusion denoiser input.

    Every optional official condition channel has a neutral tensor value instead
    of changing the object schema: all-false hotspots, class-3 (masked) secondary
    structure, and class-2 (masked) block adjacency.
    """

    noisy_coords: Tensor
    sequence: Tensor
    residue_index: Tensor
    fixed_structure_mask: Tensor
    fixed_sequence_mask: Tensor
    hotspot_mask: Tensor
    cyclic_mask: Tensor
    secondary_structure: Tensor
    block_adjacency: Tensor
    timestep: Tensor
    layout: RFdiffusionLayout


@dataclass(eq=False)
class RFdiffusionOutput:
    r"""All released RoseTTAFold denoising and training-head outputs."""

    single: Tensor
    pair: Tensor
    backbone_coords: Tensor
    state: Tensor
    torsion_angles: Tensor
    sequence_logits: Tensor
    plddt: Tensor
    distance_logits: tuple[Tensor, Tensor, Tensor, Tensor]
    plddt_logits: Tensor
    resolved_logits: Tensor
    backbone_trajectory: Tensor
    torsion_trajectory: Tensor


@dataclass(eq=False)
class RFdiffusionDesignCandidate:
    r"""One coordinate-producing RFdiffusion generation sample and its aligned values.

    Each candidate owns the complete decoded backbone ``Structure`` (which owns its
    ``MolecularSystem`` via ``Structure.system``). Generated residue identity is CCD ``UNK``;
    the Structure contains only the backbone sites RFdiffusion produced. The variable placement
    length across candidates is the coordinate-producing, variable-topology property that
    forbids reuse of the shared ``DesignCandidate``. Standalone -- no base class.

    ``placement_map`` is the frozen §3.7 realized output surface for this sample
    (``declaration_group -> placement_id -> realized OUTPUT occurrences``, W6 caveat a): after
    the variable generated lengths are sampled, RFdiffusion adopts the shared grouped
    ``placement_map`` as its per-candidate output identity (the same surface the other three
    families carry) rather than the FLAT pre-sample ``output_residues`` layout. It is
    losslessly reconstructed from that flat layout at decode time and exhaustively partitions
    every realized backbone residue; ``ComponentReference[0]`` values are this candidate's
    ``Structure.system`` instance ids. Empty only when there is no placement (no molecular
    input and no ``RFdiffusionPlacements``).
    """

    structure: Structure
    placement_map: PlacementMap = field(default_factory=dict)
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)


@dataclass(eq=False)
class RFdiffusionDesignResult:
    r"""One generation-ordered collection of coordinate-producing RFdiffusion designs.

    Tuple position is sample identity for one generation context; there is no auxiliary index
    field, and no input/context/placement-map/provenance field is carried.
    """

    candidates: tuple[RFdiffusionDesignCandidate, ...]

    def best(self, score: str) -> RFdiffusionDesignCandidate:
        r"""Return the earliest best candidate under one explicit score.

        Mirrors ``FoldResult.best`` / ``ESMFold2DesignResult.best`` exactly: the score's
        ``preferred`` direction must be consistently ``"higher"`` or ``"lower"``; unit and scope
        must match across every candidate; a NaN value or an undefined direction raises at
        ranking time; infinity is rankable; and the earliest candidate wins a tie.
        """

        scores = tuple(candidate.scores[score] for candidate in self.candidates)
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
            return max(self.candidates, key=lambda candidate: candidate.scores[score].value)
        return min(self.candidates, key=lambda candidate: candidate.scores[score].value)


# --------------------------------------------------------------------------------------
# Placement condition envelope (FROZEN §3.7 placement core carried with the family-owned
# ``chain_lengths`` length argument).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RFdiffusionPlacements:
    r"""One RFdiffusion generation topology: the frozen placement core plus its lengths.

    ``placements`` is the frozen §3.7 core ``PlacementSpec`` mapping
    (``declaration_group -> placement_id -> PlacementValue``); its inner ``placement_id``
    keys are globally unique and carry identity. ``chain_lengths`` is the RFdiffusion
    family-owned per-group combined-length argument (aligned to the declaration-group
    order of ``placements``); it stays on this family envelope, NEVER on the shared core
    (which carries no length field). ``intent`` is always ``"hard"`` for RFdiffusion.
    """

    placements: PlacementMapping
    chain_lengths: tuple[int | None, ...] | None = None
    intent: Literal["hard", "soft"] = "hard"


# --------------------------------------------------------------------------------------
# Family-owned spatial / placement conditions (DESIGN 3.7.2, family-owned & investigated).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCoordinates:
    r"""Fixed backbone coordinates for a set of existing atoms (consumed as ``hard``)."""

    sites: tuple[AtomSite, ...]
    tolerance_angstrom: float
    intent: Literal["hard", "soft"] = "hard"


@dataclass(frozen=True)
class Coordination:
    r"""A smooth ``1 / (1 + x**6)`` coordination-number potential between two sides."""

    side_a: tuple[ComponentReference | str | SymbolicResidueSite | AtomSite | SymbolicAtomSite, ...]
    side_b: tuple[ComponentReference | str | SymbolicResidueSite | AtomSite | SymbolicAtomSite, ...]
    center_angstrom: float
    scale_angstrom: float
    direction: Literal["maximize", "minimize"]
    weight: float | None = None
    intent: Literal["hard", "soft"] = "soft"


@dataclass(frozen=True)
class Distance:
    r"""A tolerance-padded ``[lower, upper]`` distance interval between two atoms."""

    site_a: AtomSite | SymbolicAtomSite
    site_b: AtomSite | SymbolicAtomSite
    lower_angstrom: float
    upper_angstrom: float
    tolerance_angstrom: float = 0.0
    weight: float | None = None
    intent: Literal["hard", "soft"] = "soft"


@dataclass(frozen=True)
class RadiusOfGyration:
    r"""A compactness potential over a set of residue centroids."""

    targets: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    upper_angstrom: float
    weight: float | None = None
    intent: Literal["hard", "soft"] = "soft"


@dataclass(frozen=True)
class SubstrateInteraction:
    r"""The released nearest-substrate attraction and clearance potential.

    ``repulsion_cutoff_angstrom`` is the released ``rep_r_0`` clearance radius; the
    non-default ``rep_r_min=0`` all-pairs branch is a separate control (see the family
    README) and is not represented by this default meaning.
    """

    residues: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    substrate_atoms: tuple[AtomSite | SymbolicAtomSite, ...]
    center_angstrom: float
    scale_angstrom: float
    attraction_strength: float
    repulsion_slope: float
    repulsion_cutoff_angstrom: float = 5.0
    weight: float | None = None
    intent: Literal["hard", "soft"] = "soft"


# RFdiffusion's requested covalent bond (e.g. terminal peptide cyclization) IS the frozen §3.7.4
# ``SymbolicBond``; the family declares no separate record and no compatibility alias (DESIGN §10.2
# forbids preserving prototype names through aliases). The processor still collapses a terminal
# N->C ``SymbolicBond`` to ``cyclic_mask`` (behavior unchanged), and RFdiffusion always consumes it
# as ``hard`` (the frozen ``SymbolicBond`` carries no ``intent`` field, matching that fixed meaning).


@dataclass(frozen=True)
class PointGroupSymmetry:
    r"""A point-group symmetry over equal-length residue units (consumed as ``hard``)."""

    units: tuple[tuple[ComponentReference | str | SymbolicResidueSite, ...], ...]
    kind: str
    order: int | None
    tolerance_angstrom: float
    intent: Literal["hard", "soft"] = "hard"


@dataclass(frozen=True)
class SecondaryStructure:
    r"""A per-target secondary-structure label; requires a fold-conditioned checkpoint."""

    targets: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    kind: str
    intent: Literal["hard", "soft"] = "soft"
