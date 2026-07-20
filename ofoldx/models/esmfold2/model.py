"""ESMFold2 with its MultiMolecule ESMC child and released folding head.

Quickstart::

    from ofoldx.models.esmfold2 import ESMFold2Config, ESMFold2Model, ESMFold2Processor

    processor = ESMFold2Processor(device="cuda")
    batch = processor("MQIFVKTLTGKT...")
    model = ESMFold2Model(ESMFold2Config()).cuda().eval()
    result = model.fold(batch, num_samples=1)

``fold`` is the inference verb (ESMC + trunk + diffusion sampler + confidence).
The bare ``model(batch)`` call is the differentiable single-step denoising path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chanfig import Config
from multimolecule import EsmCConfig, EsmCModel
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import create_bidirectional_mask

from ...data.evidence import MSA, MSACollection
from ...data.molecular import AtomSite, MolecularSystem
from ...data.structure import Structure
from ...modules.backend.atom_window import atom_to_token_mean as dispatch_atom_to_token_mean
from ...modules.backend.atom_window import token_to_atom_broadcast as dispatch_token_to_atom_broadcast
from ...modules.backend.outer_product_mean import outer_product_mean as dispatch_outer_product_mean
from ...modules.backend.pair_weighted_average import pair_weighted_average as dispatch_pair_weighted_average
from ...modules.backend.resolve import resolve_tensor_operator_backend
from ...modules.backend.transition import transition as dispatch_transition
from ...modules.backend.triangle_multiplication import triangle_multiplication as dispatch_triangle_multiplication
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .config import ESMFold2Config

# ===========================================================================
# ESMFold2's exact tensor boundary and canonical semantic layout.
# ===========================================================================


@dataclass(frozen=True)
class ESMFold2Layout:
    r"""Canonical identities aligned to one ESMFold2 token/atom tensor layout."""

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    msa: MSA | MSACollection | None = None


@dataclass(kw_only=True)
class ESMFold2Input:
    r"""Tensor inputs consumed by the released ESMFold2 graph.

    ESMC inputs use one batch row per protein chain. ``token_to_esmc`` stores
    flattened row/position indices that gather those states back to structure-token
    order.
    """

    token_index: Tensor
    residue_index: Tensor
    chain_index: Tensor
    symmetry_index: Tensor
    entity_index: Tensor
    mol_type: Tensor
    res_type: Tensor
    token_bonds: Tensor
    token_attention_mask: Tensor
    ref_pos: Tensor
    ref_element: Tensor
    ref_charge: Tensor
    ref_atom_name_chars: Tensor
    ref_space_uid: Tensor
    atom_attention_mask: Tensor
    atom_to_token: Tensor
    distogram_atom_idx: Tensor
    token_has_esmc: Tensor
    layout: ESMFold2Layout
    deletion_mean: Tensor | None = None
    msa: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_attention_mask: Tensor | None = None
    esmc_input_ids: Tensor | None = None
    esmc_attention_mask: Tensor | None = None
    token_to_esmc: Tensor | None = None
    esmc_hidden_states: Tensor | None = None
    noisy_atom_coordinates: Tensor | None = None
    noise_level: Tensor | None = None


@dataclass
class ESMFold2Output:
    r"""Differentiable single-step denoising output returned by ``forward``."""

    denoised_coordinates: Tensor
    distogram_logits: Tensor


class ESMFold2Model(nn.Module):
    """ESMFold2 — all-atom structure prediction with an ESMC PLM backbone.

    This is the standard released ESMFold2 architecture (uses a linear-
    recurrent trunk, internally referred to as "parcae").

    [`forward`][ofoldx.models.esmfold2.ESMFold2Model.forward] is the differentiable
    single-step diffusion-denoising primitive; [`fold`][ofoldx.models.esmfold2.ESMFold2Model.fold]
    is the inference verb. ``fold`` kwargs that callers commonly override:

    * ``num_loops`` (default ``config.num_loops``): trunk refinement
      loops.
    * ``num_samples``: parallel structure samples; the confidence head re-runs
      once per sample, so memory scales linearly.
    * ``num_sampling_steps`` (default ``config.structure_head.num_sampling_steps``):
      diffusion ODE solver steps. Lower for speed, higher for quality.
    * ``language_model_dropout``: released per-loop LM-pair dropout. Defaults to
      ``None`` → the released checkpoint config value (``0.25``), reproducing the
      shipped model; pass an explicit rate to override, or ``0`` to disable it.
    * ``noise_scale``, ``step_scale``, and ``max_inference_sigma``: direct
      diffusion sampler controls.

    Memory / perf knobs:

    * ``model.set_chunk_size(int|None)``: caps L² ops (triangle / OPM /
      pair transition) at this token-axis chunk. Default 64 — fits
      L≈2k on an 80 GB GPU. Pass ``None`` for faster inference at L<600.
    """

    config_class = ESMFold2Config

    def __init__(self, config: ESMFold2Config | None = None, *, esmc: EsmCModel | None = None) -> None:
        config = ESMFold2Config() if config is None else config
        super().__init__()
        self.config = config
        self.num_loops = config.num_loops
        self.language_model_mask_probability = config.language_model_mask_probability
        self.esmc_bos_token_id = config.esmc.bos_token_id
        self.esmc_eos_token_id = config.esmc.eos_token_id
        self.esmc_mask_token_id = config.esmc.mask_token_id
        self.language_model_dropout = config.lm_encoder.dropout
        self.per_loop_language_model_dropout = config.lm_encoder.per_loop_dropout
        self.msa_encoder_overwrite = config.msa_encoder_overwrite
        input_size = config.inputs.input_size
        pair_size = config.pair_size

        self.inputs_embedder = InputsEmbedder(config)
        self.z_init_1 = nn.Linear(input_size, pair_size, bias=False)
        self.z_init_2 = nn.Linear(input_size, pair_size, bias=False)
        self.rel_pos = ResIdxAsymIdSymIdEntityIdEncoding(
            config.num_relative_residue_bins,
            config.num_relative_chain_bins,
            pair_size,
        )
        self.token_bonds = nn.Linear(1, pair_size, bias=False)
        self.language_model = LanguageModelPairEncoder(
            pair_size,
            config.esmc.hidden_size,
            config.esmc.num_hidden_layers,
        )
        if esmc is None:
            esmc = EsmCModel(EsmCConfig(**dict(config.esmc.dict())))
        self.esmc = esmc
        self.esmc.requires_grad_(False)

        pf = config.folding_trunk
        self.folding_trunk = FoldingTrunk(pf.num_blocks, pair_size, expansion_ratio=4)
        if config.lm_encoder.enabled:
            self.lm_encoder: FoldingTrunk | None = FoldingTrunk(
                config.lm_encoder.num_blocks,
                pair_size,
                expansion_ratio=4,
            )
        else:
            self.lm_encoder = None

        self.parcae_input_norm = nn.LayerNorm(pair_size)
        self.parcae_log_a = nn.Parameter(torch.zeros(pair_size))
        parcae_decay_init = math.sqrt(1.0 / 5.0)
        parcae_delta_init = -math.log(parcae_decay_init)
        self.parcae_log_delta = nn.Parameter(
            torch.full(
                (pair_size,),
                parcae_delta_init + math.log(-math.expm1(-parcae_delta_init)),
                dtype=torch.float32,
            )
        )
        self.parcae_b_cont = nn.Parameter(torch.eye(pair_size))
        self.parcae_readout = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.eye_(self.parcae_readout.weight)
        self.parcae_coda = FoldingTrunk(
            config.parcae.num_coda_blocks,
            pair_size,
            expansion_ratio=4,
        )

        self.structure_head = DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(pair_size, config.structure_head.distogram_bins, bias=True)
        self.confidence_head = ConfidenceHead(config)

        msa_cfg = config.msa_encoder
        self.msa_encoder = None
        if msa_cfg.enabled:
            self.msa_encoder = MSAEncoder(
                msa_size=msa_cfg.msa_size,
                pair_size=pair_size,
                input_size=input_size,
                hidden_size=msa_cfg.hidden_size,
                num_blocks=msa_cfg.num_blocks,
                num_msa_attention_heads=msa_cfg.num_msa_attention_heads,
                msa_head_width=msa_cfg.msa_head_dim,
            )
        self.set_backend("auto")

    def set_backend(self, backend: str) -> ESMFold2Model:
        r"""Select the operator backend used by the folding head."""

        if backend not in {"auto", "torch", "oops"}:
            raise ValueError("backend must be exactly one of: auto, torch, oops")
        self.backend = backend
        operator_backend = "eager" if backend == "torch" else backend
        for module in self.modules():
            if isinstance(
                module,
                (
                    ESMFold2AtomEncoder,
                    ESMFold2AtomDecoder,
                    TransitionLayer,
                    TriangleMultiplicativeBlock,
                    Transition,
                    OuterProductMean,
                    MSAPairWeightedAveraging,
                    PairTransition,
                    ConfidenceHead,
                ),
            ):
                module.operator_backend = operator_backend
        return self

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)
        if self.lm_encoder is not None:
            self.lm_encoder.set_chunk_size(chunk_size)
        self.parcae_coda.set_chunk_size(chunk_size)
        self.confidence_head.set_chunk_size(chunk_size)
        if self.msa_encoder is not None:
            self.msa_encoder.set_chunk_size(chunk_size)

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        delta = F.softplus(self.parcae_log_delta)
        a = torch.exp(-delta * torch.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def _init_pair_state(self, ref: Tensor) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        state = torch.empty_like(ref, dtype=torch.float32)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=ref.dtype)

    def _run_one_loop(
        self,
        *,
        z: Tensor,
        z_init: Tensor,
        lm_z: Tensor,
        pair_mask: Tensor,
        a: Tensor,
        b_mat: Tensor,
        tok_mask: Tensor,
        total_steps: int,
        msa: Tensor | None,
        msa_attention_mask: Tensor | None,
        has_deletion: Tensor | None,
        deletion_value: Tensor | None,
        x_inputs: Tensor,
        msa_max_depth: int | None,
        msa_subsample_at_inference: bool,
        language_model_dropout: float | None,
    ) -> Tensor:
        # Helper method (not inline) so per-iter locals free on return —
        # otherwise leaks about 2 GB of pair activations into distogram/sample scope.
        # training=True forces dropout under eval(), matching the per-loop
        # dropout strategy used at train time.
        if language_model_dropout is None:
            dropout = self.language_model_dropout
            per_loop_language_model_dropout = self.per_loop_language_model_dropout and dropout > 0.0
        else:
            dropout = language_model_dropout
            per_loop_language_model_dropout = dropout > 0.0

        for _ in range(total_steps):
            if per_loop_language_model_dropout:
                lm_z_i = F.dropout(lm_z, p=dropout, training=True)
            else:
                lm_z_i = lm_z

            refined_lm_z: Tensor | None = None
            if self.lm_encoder is not None:
                refined_lm_z = self.lm_encoder(lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask)

            z_inject_pair = z_init
            if self.lm_encoder is None:
                z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)

            if self.msa_encoder is not None and msa is not None:
                # Fresh row subsample each iteration (column mask was applied
                # once in forward, before this loop).
                msa_i, mask_i, hd_i, dv_i = maybe_subsample_msa(
                    msa,
                    msa_attention_mask,
                    has_deletion,
                    deletion_value,
                    max_depth=msa_max_depth,
                    enabled=msa_subsample_at_inference,
                )
                B_msa, M, L_msa = msa_i.shape
                msa_oh = F.one_hot(msa_i.permute(0, 2, 1).long(), num_classes=NUM_RES_TYPES).float()
                msa_attn = (
                    mask_i.permute(0, 2, 1).float()
                    if mask_i is not None
                    else tok_mask[:, :, None].expand(-1, -1, M).float()
                )
                # Bias-free MSAEncoder.embed requires zeroed padding.
                msa_oh = msa_oh * msa_attn.unsqueeze(-1)
                hd = (
                    hd_i.permute(0, 2, 1).float()
                    if hd_i is not None
                    else torch.zeros(B_msa, L_msa, M, device=msa_i.device)
                )
                dv = (
                    dv_i.permute(0, 2, 1).float()
                    if dv_i is not None
                    else torch.zeros(B_msa, L_msa, M, device=msa_i.device)
                )
                msa_pair = self.msa_encoder(
                    x_pair=z_inject_pair,
                    x_inputs=x_inputs,
                    msa_oh=msa_oh,
                    has_deletion=hd,
                    deletion_value=dv,
                    msa_attention_mask=msa_attn,
                ).to(z_inject_pair.dtype)
                z_inject_pair = msa_pair if self.msa_encoder_overwrite else (z_inject_pair + msa_pair)

            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)

            injected_pair = self.parcae_input_norm(z_inject_pair)
            z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)

        return z

    def _encode_inputs_and_trunk(
        self,
        *,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        esmc_input_ids: Tensor | None = None,
        esmc_attention_mask: Tensor | None = None,
        token_to_esmc: Tensor | None = None,
        token_has_esmc: Tensor | None = None,
        esmc_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        r"""Run the shared trunk and emit single/pair representations.

        Both the training [`forward`][ofoldx.models.esmfold2.ESMFold2Model.forward] and the inference
        [`fold`][ofoldx.models.esmfold2.ESMFold2Model.fold] verb consume this: it returns the
        single (``s_inputs``) and pair (``z``) trunk streams, plus auxiliary reference features and
        teacher-forced distogram logits.
        """

        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask

        n_loops: int = num_loops if num_loops is not None else self.num_loops
        if n_loops < 0:
            raise ValueError("num_loops must be non-negative")
        total_steps = n_loops + 1

        res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
        res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()

        if msa is not None:
            msa_oh_profile = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_oh_profile = msa_oh_profile * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_oh_profile.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                profile = msa_oh_profile.mean(dim=1)
        else:
            profile = res_type_oh

        if deletion_mean is None:
            deletion_mean = torch.zeros(res_type.shape[0], res_type.shape[1], device=res_type.device)

        ref_element_oh = F.one_hot(ref_element.long(), num_classes=MAX_ATOMIC_NUMBER).float()
        ref_atom_name_chars_oh = F.one_hot(ref_atom_name_chars.long(), num_classes=CHAR_VOCAB_SIZE).float()
        # Bias-free downstream Linears require zeroed padding.
        atm_mask_f = atm_mask.float()
        ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(-1).unsqueeze(-1)
        atom_to_token = atom_to_token * atm_mask.long()

        with torch.autocast(device_type="cuda", enabled=ref_pos.is_cuda, dtype=torch.bfloat16):
            x_inputs = self.inputs_embedder(
                aatype=res_type_oh,
                profile=profile.float(),
                deletion_mean=deletion_mean.float(),
                ref_pos=ref_pos,
                atom_attention_mask=atm_mask,
                ref_space_uid=ref_space_uid,
                ref_charge=ref_charge,
                ref_element=ref_element_oh,
                ref_atom_name_chars=ref_atom_name_chars_oh,
                atom_to_token=atom_to_token,
            )

            z_init = self.z_init_1(x_inputs).unsqueeze(2) + self.z_init_2(x_inputs).unsqueeze(1)

            relative_position_encoding = self.rel_pos(
                residue_index=residue_index,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                token_index=token_index,
            )
            token_bonds_encoding = self.token_bonds(token_bonds.float())
            z_init = z_init + relative_position_encoding + token_bonds_encoding

        if esmc_hidden_states is None:
            if esmc_input_ids is None or esmc_attention_mask is None or token_to_esmc is None or token_has_esmc is None:
                raise RuntimeError("ESMFold2 requires ESMC inputs or expert-provided esmc_hidden_states")
            esmc_hidden_states = compute_esmc_hidden_states(
                self.esmc,
                esmc_input_ids,
                esmc_attention_mask,
                token_to_esmc,
                token_has_esmc,
                lm_mask_pct=(
                    language_model_mask_probability
                    if language_model_mask_probability is not None
                    else self.language_model_mask_probability
                ),
                bos_token_id=self.esmc_bos_token_id,
                eos_token_id=self.esmc_eos_token_id,
                mask_token_id=self.esmc_mask_token_id,
            )
        if not ref_pos.is_cuda:
            esmc_hidden_states = esmc_hidden_states.float()
        with torch.autocast(device_type="cuda", enabled=ref_pos.is_cuda, dtype=torch.bfloat16):
            lm_z = self.language_model(esmc_hidden_states)
        del esmc_hidden_states

        pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

        z = self._init_pair_state(z_init)

        a, b = self._discretized_dynamics()
        a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
        b_mat = b.to(device=z.device, dtype=z.dtype)

        # Inference-time MSA diversity: column mask is applied once here
        # (shared across recycling loops); row subsampling is deferred to
        # per-iter inside _run_one_loop (fresh subset per loop).
        if self.msa_encoder is not None and msa is not None:
            msa_attention_mask = maybe_apply_msa_column_masking(msa_attention_mask, rate=msa_column_mask_rate)

        # Method call (not inline loop) frees per-iteration pair locals.
        with torch.autocast(device_type="cuda", enabled=ref_pos.is_cuda, dtype=torch.bfloat16):
            z = self._run_one_loop(
                z=z,
                z_init=z_init,
                lm_z=lm_z,
                pair_mask=pair_mask,
                a=a,
                b_mat=b_mat,
                tok_mask=tok_mask,
                total_steps=total_steps,
                msa=msa,
                msa_attention_mask=msa_attention_mask,
                has_deletion=has_deletion,
                deletion_value=deletion_value,
                x_inputs=x_inputs,
                msa_max_depth=msa_max_depth,
                msa_subsample_at_inference=msa_subsample_at_inference,
                language_model_dropout=language_model_dropout,
            )
            z = self.parcae_readout(z)
            z = self.parcae_coda(z, pair_attention_mask=pair_mask)
        del z_init, lm_z, a, b_mat

        z = z.float()
        x_inputs = x_inputs.float()
        relative_position_encoding = relative_position_encoding.float()
        token_bonds_encoding = token_bonds_encoding.float()
        distogram_logits = self.distogram_head(z + z.transpose(-2, -3))

        aux: dict[str, Tensor] = {
            "relative_position_encoding": relative_position_encoding,
            "token_bonds_encoding": token_bonds_encoding,
            "ref_element_oh": ref_element_oh,
            "ref_atom_name_chars_oh": ref_atom_name_chars_oh,
            "atom_to_token": atom_to_token,
            "distogram_logits": distogram_logits,
        }
        return x_inputs, z, aux

    def embeddings(
        self,
        batch: ESMFold2Input,
        /,
        *,
        num_loops: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
    ) -> tuple[Tensor, Tensor]:
        r"""Return the released token and pair embeddings before structure sampling."""

        token_embeddings, pair_embeddings, _ = self._encode_inputs_and_trunk(
            token_index=batch.token_index,
            residue_index=batch.residue_index,
            asym_id=batch.chain_index,
            sym_id=batch.symmetry_index,
            entity_id=batch.entity_index,
            res_type=batch.res_type,
            token_bonds=batch.token_bonds,
            token_attention_mask=batch.token_attention_mask,
            ref_pos=batch.ref_pos,
            ref_element=batch.ref_element,
            ref_charge=batch.ref_charge,
            ref_atom_name_chars=batch.ref_atom_name_chars,
            ref_space_uid=batch.ref_space_uid,
            atom_attention_mask=batch.atom_attention_mask,
            atom_to_token=batch.atom_to_token,
            deletion_mean=batch.deletion_mean,
            msa=batch.msa,
            has_deletion=batch.has_deletion,
            deletion_value=batch.deletion_value,
            msa_attention_mask=batch.msa_attention_mask,
            esmc_input_ids=batch.esmc_input_ids,
            esmc_attention_mask=batch.esmc_attention_mask,
            token_to_esmc=batch.token_to_esmc,
            token_has_esmc=batch.token_has_esmc,
            esmc_hidden_states=batch.esmc_hidden_states,
            num_loops=num_loops,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
        )
        return token_embeddings, pair_embeddings

    @torch.inference_mode()
    def _fold_tensors(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_idx: Tensor,
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        esmc_input_ids: Tensor | None = None,
        esmc_attention_mask: Tensor | None = None,
        token_to_esmc: Tensor | None = None,
        token_has_esmc: Tensor | None = None,
        esmc_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
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
    ) -> dict[str, Tensor]:
        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask
        disto_idx = distogram_atom_idx
        n_samples = 1 if num_diffusion_samples is None else num_diffusion_samples

        x_inputs, z, aux = self._encode_inputs_and_trunk(
            token_index=token_index,
            residue_index=residue_index,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            res_type=res_type,
            token_bonds=token_bonds,
            token_attention_mask=token_attention_mask,
            ref_pos=ref_pos,
            ref_element=ref_element,
            ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            atom_attention_mask=atom_attention_mask,
            atom_to_token=atom_to_token,
            deletion_mean=deletion_mean,
            msa=msa,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_attention_mask=msa_attention_mask,
            esmc_input_ids=esmc_input_ids,
            esmc_attention_mask=esmc_attention_mask,
            token_to_esmc=token_to_esmc,
            token_has_esmc=token_has_esmc,
            esmc_hidden_states=esmc_hidden_states,
            num_loops=num_loops,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
        )
        relative_position_encoding = aux["relative_position_encoding"]
        token_bonds_encoding = aux["token_bonds_encoding"]
        ref_element_oh = aux["ref_element_oh"]
        ref_atom_name_chars_oh = aux["ref_atom_name_chars_oh"]
        atom_to_token = aux["atom_to_token"]
        distogram_logits = aux["distogram_logits"]

        structure_output = self.structure_head.sample(
            z_trunk=z,
            s_inputs=x_inputs,
            relative_position_encoding=relative_position_encoding,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=atm_mask,
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            ref_space_uid=ref_space_uid,
            tok_idx=atom_to_token,
            token_attention_mask=tok_mask,
            num_diffusion_samples=n_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            return_atom_repr=False,
            denoising_early_exit_rmsd=denoising_early_exit_rmsd,
        )

        sample_coords = cast(Tensor, structure_output["sample_atom_coords"])
        output: dict[str, Tensor] = {"distogram_logits": distogram_logits}
        output["sample_atom_coords"] = sample_coords

        confidence_output = self.confidence_head(
            s_inputs=x_inputs,
            z=z,
            x_pred=sample_coords,
            distogram_atom_idx=disto_idx,
            token_attention_mask=tok_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atm_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=n_samples,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
        )
        output.update(confidence_output)
        output["atom_pad_mask"] = atm_mask
        output["residue_index"] = residue_index
        output["entity_id"] = entity_id
        return output

    def fold(
        self,
        batch: ESMFold2Input,
        /,
        *,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_loops: int | None = None,
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
        r"""Run released folding and decode a self-contained canonical FoldResult."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if batch.token_index.shape[0] != 1:
            raise ValueError("ESMFold2 fold accepts one canonical system per call")

        effective_seed = seed
        if generator is not None:
            effective_seed = int(
                torch.randint(
                    0,
                    2**31,
                    (1,),
                    device=torch.device(generator.device),
                    generator=generator,
                ).item()
            )

        devices = [batch.ref_pos.device] if batch.ref_pos.is_cuda else []
        with torch.random.fork_rng(devices=devices, enabled=effective_seed is not None):
            if effective_seed is not None:
                torch.random.default_generator.manual_seed(effective_seed)
                if batch.ref_pos.is_cuda:
                    with torch.cuda.device(batch.ref_pos.device):
                        torch.cuda.manual_seed(effective_seed)
            raw = self._fold_tensors(
                token_index=batch.token_index,
                residue_index=batch.residue_index,
                asym_id=batch.chain_index,
                sym_id=batch.symmetry_index,
                entity_id=batch.entity_index,
                mol_type=batch.mol_type,
                res_type=batch.res_type,
                token_bonds=batch.token_bonds,
                token_attention_mask=batch.token_attention_mask,
                ref_pos=batch.ref_pos,
                ref_element=batch.ref_element,
                ref_charge=batch.ref_charge,
                ref_atom_name_chars=batch.ref_atom_name_chars,
                ref_space_uid=batch.ref_space_uid,
                atom_attention_mask=batch.atom_attention_mask,
                atom_to_token=batch.atom_to_token,
                distogram_atom_idx=batch.distogram_atom_idx,
                deletion_mean=batch.deletion_mean,
                msa=batch.msa,
                has_deletion=batch.has_deletion,
                deletion_value=batch.deletion_value,
                msa_attention_mask=batch.msa_attention_mask,
                esmc_input_ids=batch.esmc_input_ids,
                esmc_attention_mask=batch.esmc_attention_mask,
                token_to_esmc=batch.token_to_esmc,
                token_has_esmc=batch.token_has_esmc,
                esmc_hidden_states=batch.esmc_hidden_states,
                num_loops=num_loops,
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
            )

        layout = batch.layout
        atom_count = len(layout.atom_sites)
        raw_coordinates = raw["sample_atom_coords"][:, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        raw_plddt = raw["plddt"].to(device="cpu", dtype=torch.float32).numpy()
        raw_pae = raw["pae"].to(device="cpu", dtype=torch.float32).numpy()
        complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray(
            [complete_index[site] for site in layout.atom_sites],
            dtype=np.intp,
        )
        # Chains carry asym_id 0..n-1 in system.instances order (processor), so the
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
                "complex_plddt": Score(
                    float(raw["complex_plddt"][sample_index].item()),
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
                        # Per-atom pLDDT is finer-grained than the token pLDDT and is not
                        # recoverable from it, so it is preserved on the AtomSite axis. The
                        # raw plddt/pae/pde/resolved/distogram logits, the duplicate token
                        # 'plddt'/'pae', and the representative-atom pLDDT were pure
                        # diagnostics and are dropped.
                        "plddt_per_atom": Metric(
                            values=raw["plddt_per_atom"][sample_index, :atom_count].float().cpu().numpy(),
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
                        # Canonical §5.2 pair metric; previously emitted only as a diagnostic.
                        "pde": PairMetric(
                            values=raw["pde"][sample_index].float().cpu().numpy(),
                            row_axis=layout.token_sites,
                            column_axis=layout.token_sites,
                            unit="angstrom",
                        ),
                        # Per-chain-pair interface confidence on the chain (instance_id) axis.
                        "pair_chains_iptm": PairMetric(
                            values=raw["pair_chains_iptm"][sample_index].float().cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                    },
                )
            )

        return FoldResult(predictions=tuple(predictions))

    def forward(
        self,
        batch: ESMFold2Input,
        *,
        num_loops: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
    ) -> ESMFold2Output:
        r"""Denoise explicit noisy coordinates in one differentiable step.

        This tensor primitive does not publish an end-to-end training contract.
        [`fold`][ofoldx.models.esmfold2.ESMFold2Model.fold] runs the released sampler
        directly and does not call it.
        """

        if batch.noisy_atom_coordinates is None or batch.noise_level is None:
            raise ValueError("ESMFold2 forward requires explicit noisy_atom_coordinates and noise_level")

        x_inputs, z, aux = self._encode_inputs_and_trunk(
            token_index=batch.token_index,
            residue_index=batch.residue_index,
            asym_id=batch.chain_index,
            sym_id=batch.symmetry_index,
            entity_id=batch.entity_index,
            res_type=batch.res_type,
            token_bonds=batch.token_bonds,
            token_attention_mask=batch.token_attention_mask,
            ref_pos=batch.ref_pos,
            ref_element=batch.ref_element,
            ref_charge=batch.ref_charge,
            ref_atom_name_chars=batch.ref_atom_name_chars,
            ref_space_uid=batch.ref_space_uid,
            atom_attention_mask=batch.atom_attention_mask,
            atom_to_token=batch.atom_to_token,
            deletion_mean=batch.deletion_mean,
            msa=batch.msa,
            has_deletion=batch.has_deletion,
            deletion_value=batch.deletion_value,
            msa_attention_mask=batch.msa_attention_mask,
            esmc_input_ids=batch.esmc_input_ids,
            esmc_attention_mask=batch.esmc_attention_mask,
            token_to_esmc=batch.token_to_esmc,
            token_has_esmc=batch.token_has_esmc,
            esmc_hidden_states=batch.esmc_hidden_states,
            num_loops=num_loops,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
        )
        atom_to_token = aux["atom_to_token"]

        dm_out = self.structure_head.diffusion_module(
            x_noisy=batch.noisy_atom_coordinates,
            t_hat=batch.noise_level,
            ref_pos=batch.ref_pos,
            ref_charge=batch.ref_charge,
            ref_mask=batch.atom_attention_mask,
            ref_element=aux["ref_element_oh"],
            ref_atom_name_chars=aux["ref_atom_name_chars_oh"],
            ref_space_uid=batch.ref_space_uid,
            tok_idx=atom_to_token,
            s_inputs=x_inputs,
            z_trunk=z,
            relative_position_encoding=aux["relative_position_encoding"],
            token_attention_mask=batch.token_attention_mask,
            num_diffusion_samples=1,
        )
        denoised = cast(Tensor, dm_out["x_denoised"])

        return ESMFold2Output(
            denoised_coordinates=denoised,
            distogram_logits=aux["distogram_logits"],
        )


CHAR_VOCAB_SIZE: int = 64
MAX_CHARS: int = 4
XYZ_DIMS: int = 3
MAX_ATOMIC_NUMBER: int = 128

# Input feature dim = 3 + 1 + 1 + 128 + 64*4 = 389
ATOM_FEATURE_DIM: int = XYZ_DIMS + 1 + 1 + MAX_ATOMIC_NUMBER + CHAR_VOCAB_SIZE * MAX_CHARS


NUM_RES_TYPES: int = 33

_EPS = 1e-5

# Default for the triangle / OPM / pair-transition L² ops. Caps peak memory
# so L≈2k folds on an 80 GB GPU (~76 GB peak at chunk=128 for L=1438;
# chunk=64 leaves headroom for the largest foldbench targets). Override via
# ``model.set_chunk_size(...)``; pass None to disable chunking (faster for
# short L but OOM-prone past ~600).
_DEFAULT_CHUNK_SIZE = 64


def maybe_subsample_msa(
    msa: Tensor,
    msa_attention_mask: Tensor | None,
    has_deletion: Tensor | None,
    deletion_value: Tensor | None,
    *,
    max_depth: int | None,
    enabled: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Randomly subsample the MSA to ``max_depth`` rows, keeping query row 0.

    Accepts encoder-input MSA token tensors shaped ``[B, depth, L]``. No-op when
    disabled, ``max_depth`` is None, or depth <= ``max_depth``.
    """
    if not enabled or max_depth is None:
        return msa, msa_attention_mask, has_deletion, deletion_value
    depth = msa.size(1)
    if depth <= 1 or depth <= max_depth:
        return msa, msa_attention_mask, has_deletion, deletion_value
    indices = torch.zeros(max_depth, dtype=torch.long, device=msa.device)
    perm = torch.randperm(depth - 1, device=msa.device)[: max_depth - 1]
    indices[1:] = perm + 1
    indices = indices.sort().values
    msa = msa[:, indices]
    if msa_attention_mask is not None:
        msa_attention_mask = msa_attention_mask[:, indices]
    if has_deletion is not None:
        has_deletion = has_deletion[:, indices]
    if deletion_value is not None:
        deletion_value = deletion_value[:, indices]
    return msa, msa_attention_mask, has_deletion, deletion_value


def maybe_apply_msa_column_masking(msa_attention_mask: Tensor | None, rate: float) -> Tensor | None:
    """Mask fraction ``rate`` of MSA columns in non-query rows of
    ``msa_attention_mask``, keeping query row 0. No-op when absent, ``rate <= 0``,
    or depth <= 1.
    """
    if msa_attention_mask is None or rate <= 0.0 or msa_attention_mask.size(1) <= 1:
        return msa_attention_mask
    B, _M, L = msa_attention_mask.shape
    col_keep = torch.rand(B, L, device=msa_attention_mask.device) >= rate
    masked = msa_attention_mask.bool() & col_keep.unsqueeze(1)
    masked[:, 0, :] = msa_attention_mask[:, 0, :].bool()
    return masked


def gather_token_to_atom(
    token_features: Tensor,
    atom_to_token_idx: Tensor,
    *,
    operator_backend: str = "eager",
) -> Tensor:
    """Broadcast per-token features to per-atom features using gather.

    Args:
        token_features: [B, L, d]
        atom_to_token_idx: [B, A] int64

    Returns:
        [B, A, d]
    """
    resolved = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(token_features, atom_to_token_idx),
        supported=("oops", "eager"),
        public_default="eager",
    )
    if resolved == "oops":
        return dispatch_token_to_atom_broadcast(
            token_features,
            atom_to_token_idx,
            operator_backend="oops",
        )
    idx = atom_to_token_idx.unsqueeze(-1).expand(-1, -1, token_features.size(-1))
    return torch.gather(token_features, 1, idx)


def scatter_atom_to_token(
    atom_features: Tensor,
    atom_to_token_idx: Tensor,
    n_tokens: int,
    atom_mask: Tensor,
    *,
    operator_backend: str = "eager",
) -> Tensor:
    """Aggregate per-atom features to per-token features (mean).

    Args:
        atom_features: [B, A, d]
        atom_to_token_idx: [B, A] int64
        n_tokens: L
        atom_mask: [B, A] bool

    Returns:
        [B, L, d]
    """
    resolved = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(atom_features, atom_to_token_idx, atom_mask),
        supported=("oops", "eager"),
        public_default="eager",
    )
    if resolved == "oops":
        return dispatch_atom_to_token_mean(
            atom_features,
            atom_to_token_idx,
            num_tokens=n_tokens,
            atom_mask=atom_mask,
            eps=1e-6,
            operator_backend="oops",
        )
    B, A, d = atom_features.shape
    idx = torch.where(atom_mask, atom_to_token_idx, n_tokens)
    idx_expanded = idx.unsqueeze(-1).expand(B, A, d)
    out = torch.zeros(B, n_tokens + 1, d, device=atom_features.device, dtype=atom_features.dtype)
    out.scatter_reduce_(1, idx_expanded, atom_features, reduce="mean", include_self=False)
    return out[:, :n_tokens, :]


def gather_rep_atom_coords(coords: Tensor, rep_atom_idx: Tensor) -> Tensor:
    """Gather representative atom coordinates for each token.

    Args:
        coords: [B, A, 3]
        rep_atom_idx: [B, L] int64

    Returns:
        [B, L, 3]
    """
    idx = rep_atom_idx.unsqueeze(-1).expand(-1, -1, coords.size(-1))
    return torch.gather(coords, 1, idx)


def _compute_intra_token_idx(atom_to_token: Tensor) -> Tensor:
    """Compute local atom index within each token (vectorised).

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
    """Expected value of a categorical distribution over evenly-spaced bins.

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


class TransitionLayer(nn.Module):
    """SwiGLU transition: norm -> a_proj, b_proj -> silu(a)*b -> out_proj."""

    def __init__(self, d_model: int, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.operator_backend = "eager"
        hidden = n * d_model
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.a_proj = nn.Linear(d_model, hidden, bias=False)
        self.b_proj = nn.Linear(d_model, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        resolved = resolve_tensor_operator_backend(
            self.operator_backend,
            tensors=(x,),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            return dispatch_transition(
                x,
                norm_weight=self.norm.weight,
                norm_bias=self.norm.bias,
                gate_weight=self.a_proj.weight,
                up_weight=self.b_proj.weight,
                down_weight=self.out_proj.weight,
                eps=self.norm.eps,
                operator_backend="oops",
            )
        x = self.norm(x)
        a = self.a_proj(x)
        b = self.b_proj(x)
        return self.out_proj(F.silu(a) * b)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization (adaLN-Zero)."""

    def __init__(self, d_model: int, d_cond: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.eps = eps
        self.s_scale = nn.Parameter(torch.ones(d_cond))
        self.s_gate = nn.Linear(d_cond, d_model, bias=True)
        self.s_shift = nn.Linear(d_cond, d_model, bias=False)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        a_norm = F.layer_norm(a, (self.d_model,), None, None, self.eps)
        s_norm = F.layer_norm(s, (self.d_cond,), self.s_scale, None, self.eps)
        return torch.sigmoid(self.s_gate(s_norm)) * a_norm + self.s_shift(s_norm)


class FourierEmbedding(nn.Module):
    """Fourier embedding: cos(2*pi*(t*w + b))."""

    w: Tensor
    b: Tensor

    def __init__(self, c: int) -> None:
        super().__init__()
        self.register_buffer("w", torch.randn(c))
        self.register_buffer("b", torch.randn(c))

    def forward(self, t_hat: Tensor) -> Tensor:
        t = t_hat.to(device=self.w.device, dtype=self.w.dtype).reshape(-1)
        return torch.cos(2.0 * torch.pi * (t[:, None] * self.w[None, :] + self.b[None, :]))


class SwiGLU(nn.Module):
    """SwiGLU with packed w12 and output w3."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.hidden_features = hidden_features

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.split(self.hidden_features, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUMLP(SwiGLU):
    """SwiGLU MLP with packed weights, no bias."""

    def __init__(self, d_model: int, expansion_ratio: int = 4, bias: bool = False) -> None:
        hidden = expansion_ratio * d_model
        super().__init__(in_features=d_model, hidden_features=hidden, out_features=d_model, bias=bias)


def apply_rotary_emb_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE with batch-dependent cos/sin.

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


def build_3d_rope(
    ref_pos: Tensor,
    ref_space_uid: Tensor,
    head_dim: int,
    n_spatial_per_axis: int = 4,
    n_uid_pairs: int = 2,
    spatial_base_freq: float = 10000.0,
    uid_base_freq: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Build cos/sin for 3D RoPE + UID RoPE."""
    device = ref_pos.device
    B, N = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    spatial_inv_freq = 1.0 / (
        spatial_base_freq
        ** (torch.arange(0, n_spatial_per_axis, dtype=torch.float32, device=device) / n_spatial_per_axis)
    )
    uid_inv_freq = 1.0 / (
        uid_base_freq ** (torch.arange(0, n_uid_pairs, dtype=torch.float32, device=device) / n_uid_pairs)
    )

    pos_f32 = ref_pos.float()
    spatial_freqs = torch.einsum("bna,k->bnak", pos_f32, spatial_inv_freq)
    spatial_freqs = spatial_freqs.reshape(B, N, n_spatial_total)

    uid_f32 = ref_space_uid.float()
    uid_freqs = torch.einsum("bn,k->bnk", uid_f32, uid_inv_freq)

    n_active = n_spatial_total + n_uid_pairs
    freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)

    if n_active < half_dim:
        padding = torch.zeros(B, N, half_dim - n_active, device=device, dtype=torch.float32)
        freqs = torch.cat([freqs, padding], dim=-1)

    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    return cos, sin


def qk_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with rounded hidden size for hardware alignment."""

    def __init__(self, d_model: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        hidden_size = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.w_up = nn.Linear(d_model, 2 * hidden_size, bias=False)
        self.w_down = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.w_up.weight.dtype)
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(x1) * x2)


class SWA3DRoPEAttention(nn.Module):
    """Sliding window attention with 3D RoPE. Has Wqkv, gate_proj, out_proj."""

    def __init__(self, d_model: int, n_heads: int, half_window: int = 64) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.half_window = half_window

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        B, N = x.shape[:2]

        x_input = x
        qkv = self.Wqkv(x)
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        q, k = qk_norm(q), qk_norm(k)

        q = apply_rotary_emb_3d(q, attention_cos, attention_sin)
        k = apply_rotary_emb_3d(k, attention_cos, attention_sin)

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        valid = atom_mask
        rank = torch.cumsum(valid, dim=1) - 1
        within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window
        allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
        allowed |= torch.eye(N, dtype=torch.bool, device=q.device)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed.unsqueeze(1),
            scale=self.scale,
        ).transpose(1, 2)
        out = out * valid.unsqueeze(-1).unsqueeze(-1)

        out = out.to(input_dtype).reshape(B, N, -1)  # type: ignore[union-attr]
        out = out * torch.sigmoid(self.gate_proj(x_input))
        return self.out_proj(out)


class SWAAtomBlock(nn.Module):
    """adaLN-Zero + SWA attention + SwiGLU FFN.

    Creates adaln_modulation = Sequential(SiLU(), Linear) -> keys like adaln_modulation.1.weight
    """

    def __init__(
        self,
        d_atom: int,
        n_heads: int,
        half_window: int = 64,
        expansion_ratio: int = 2,
    ) -> None:
        super().__init__()
        adaln_linear = nn.Linear(d_atom, 6 * d_atom, bias=False)
        nn.init.zeros_(adaln_linear.weight)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), adaln_linear)

        self.attn = SWA3DRoPEAttention(d_atom, n_heads, half_window=half_window)
        self.ffn = SwiGLUFFN(d_atom, expansion_ratio)

    def forward(
        self,
        x: Tensor,
        c_l: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        mod = self.adaln_modulation(c_l)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        attn_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_a) + shift_a
        attn_out = self.attn(attn_input, attention_cos, attention_sin, atom_mask)
        x = x + gate_a * attn_out

        ffn_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_f) + shift_f
        ffn_out = self.ffn(ffn_input)
        x = x + gate_f * ffn_out
        return x


class SWAAtomTransformer(nn.Module):
    """Stack of SWAAtomBlocks."""

    def __init__(
        self,
        d_atom: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim = d_atom // n_heads
        self.spatial_rope_base_frequency = spatial_rope_base_frequency
        self.n_spatial_rope_pairs_per_axis = n_spatial_rope_pairs_per_axis
        self.n_uid_rope_pairs = n_uid_rope_pairs
        self.uid_rope_base_frequency = uid_rope_base_frequency

        self.blocks = nn.ModuleList(
            [
                SWAAtomBlock(
                    d_atom=d_atom,
                    n_heads=n_heads,
                    half_window=swa_window_size // 2,
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(n_blocks)
            ]
        )

    def build_3d_rope(self, ref_pos: Tensor, ref_space_uid: Tensor) -> tuple[Tensor, Tensor]:
        r"""Build the atom transformer's spatial and reference-space rotations."""

        return build_3d_rope(
            ref_pos=ref_pos,
            ref_space_uid=ref_space_uid,
            head_dim=self.head_dim,
            n_spatial_per_axis=self.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.n_uid_rope_pairs,
            spatial_base_freq=self.spatial_rope_base_frequency,
            uid_base_freq=self.uid_rope_base_frequency,
        )

    def forward(
        self,
        q_l: Tensor,
        c_l: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        for block in self.blocks:
            q_l = block(q_l, c_l, attention_cos, attention_sin, atom_mask)
            if return_intermediates:
                intermediates.append(q_l)
        if return_intermediates:
            return q_l, intermediates
        return q_l


class ESMFold2AtomEncoder(nn.Module):
    """SWA atom encoder with atom_linear, atom_norm, atom_to_token_linear, [coords_linear], atom_transformer.

    Args:
        d_atom: atom hidden dim
        d_token: token dim for atom_to_token aggregation
        n_blocks, n_heads, swa_window_size, expansion_ratio: transformer params
        structure_prediction: if True, creates coords_linear and uses full d_token
        spatial_rope_base_frequency, n_spatial_rope_pairs_per_axis,
        n_uid_rope_pairs, uid_rope_base_frequency: 3D RoPE config
    """

    def __init__(
        self,
        d_atom: int = 128,
        d_token: int = 768,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        structure_prediction: bool = True,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.structure_prediction = structure_prediction
        self.operator_backend = "eager"

        self.atom_linear = nn.Linear(ATOM_FEATURE_DIM, d_atom, bias=False)
        self.atom_norm = nn.LayerNorm(d_atom)

        if structure_prediction:
            self.coords_linear = nn.Linear(6, d_atom, bias=False)

        self.atom_transformer = SWAAtomTransformer(
            d_atom=d_atom,
            n_blocks=n_blocks,
            n_heads=n_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        # Output aggregation: d_token for structure prediction, d_token//2 for inputs
        out_dim = d_token if structure_prediction else d_token // 2
        self.atom_to_token_linear = nn.Linear(d_atom, out_dim, bias=False)

    def encode_reference(
        self,
        ref_pos: Tensor,
        atom_attention_mask: Tensor,
        ref_space_uid: Tensor,
        ref_charge: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_to_token: Tensor,
        num_diffusion_samples: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Encode the reference atoms once for one or more diffusion samples."""

        B, N = ref_pos.shape[:2]
        atom_features = torch.cat(
            [
                ref_pos,
                ref_charge.unsqueeze(-1),
                atom_attention_mask.unsqueeze(-1),
                ref_element,
                ref_atom_name_chars.reshape(B, N, MAX_CHARS * CHAR_VOCAB_SIZE),
            ],
            dim=-1,
        )
        reference_atom_features = self.atom_norm(self.atom_linear(atom_features))
        attention_cos, attention_sin = self.atom_transformer.build_3d_rope(ref_pos, ref_space_uid)
        reference_atom_features = reference_atom_features.repeat_interleave(num_diffusion_samples, 0)
        attention_cos = attention_cos.repeat_interleave(num_diffusion_samples, 0)
        attention_sin = attention_sin.repeat_interleave(num_diffusion_samples, 0)
        expanded_atom_mask = atom_attention_mask.repeat_interleave(num_diffusion_samples, 0)
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
        r_l: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        r"""Run atom attention from explicit reference features."""

        q = reference_atom_features

        if self.structure_prediction:
            noisy_coordinates = cast(Tensor, r_l)
            r_input = torch.cat([noisy_coordinates, torch.zeros_like(noisy_coordinates)], dim=-1)
            r_to_q = self.coords_linear(r_input)
            q = q + r_to_q

        result = self.atom_transformer(
            q_l=q,
            c_l=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            atom_mask=expanded_atom_mask,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            q, intermediates = result
        else:
            q = result
            intermediates = []

        q_to_a = F.relu(self.atom_to_token_linear(q))
        a = scatter_atom_to_token(
            q_to_a,
            expanded_atom_to_token,
            num_tokens,
            atom_mask=expanded_atom_mask.bool(),
            operator_backend=self.operator_backend,
        )

        return a, q, reference_atom_features, intermediates

    def forward(
        self,
        ref_pos: Tensor,
        atom_attention_mask: Tensor,
        ref_space_uid: Tensor,
        ref_charge: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        r_l: Tensor | None = None,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        r"""Encode reference atoms and run atom attention."""

        (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        ) = self.encode_reference(
            ref_pos,
            atom_attention_mask,
            ref_space_uid,
            ref_charge,
            ref_element,
            ref_atom_name_chars,
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
            r_l,
            return_intermediates,
        )


class ESMFold2AtomDecoder(nn.Module):
    """SWA atom decoder with token_to_atom_linear, atom_transformer, norm, output_linear."""

    def __init__(
        self,
        d_atom: int = 128,
        d_token: int = 768,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.operator_backend = "eager"
        self.token_to_atom_linear = nn.Linear(d_token, d_atom, bias=False)

        self.atom_transformer = SWAAtomTransformer(
            d_atom=d_atom,
            n_blocks=n_blocks,
            n_heads=n_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        self.norm = nn.LayerNorm(d_atom)
        self.output_linear = nn.Linear(d_atom, XYZ_DIMS, bias=False)

    def forward(
        self,
        a_i: Tensor,
        q_l: Tensor,
        c_l: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        atom_mask: Tensor,
        expanded_atom_to_token: Tensor,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Returns (r_update, intermediates)."""
        a_to_q = self.token_to_atom_linear(a_i)
        a_to_q = gather_token_to_atom(
            a_to_q,
            expanded_atom_to_token,
            operator_backend=self.operator_backend,
        )
        q_l = q_l + a_to_q

        result = self.atom_transformer(
            q_l=q_l,
            c_l=c_l,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            atom_mask=atom_mask,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            q_l, intermediates = result
        else:
            q_l = result
            intermediates = []

        r_l = self.output_linear(self.norm(q_l))
        return r_l, intermediates


class AttentionPairBias(nn.Module):
    """Gated multi-head attention with pair bias conditioning."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        num_heads: int,
        d_cond: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.adaln = AdaptiveLayerNorm(d_model, d_cond, eps=1e-5)
        self.out_gate = nn.Linear(d_cond, d_model, bias=True)
        nn.init.zeros_(self.out_gate.weight)
        nn.init.constant_(self.out_gate.bias, -2.0)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.pair_norm = nn.LayerNorm(d_pair, eps=1e-5)
        self.pair_bias_proj = nn.Linear(d_pair, num_heads, bias=False)

    def forward(
        self,
        a: Tensor,
        s: Tensor,
        z: Tensor,
        attention_mask: Tensor,
        num_diffusion_samples: int = 1,
    ) -> Tensor:
        bsz, n_queries, d_model = a.shape
        x = self.adaln(a, s)

        n_keys = x.shape[1]
        q = self.q_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(bsz, n_keys, self.num_heads, self.head_dim)
        v = v.view(bsz, n_keys, self.num_heads, self.head_dim)

        # Expand z for num_diffusion_samples
        if num_diffusion_samples > 1:
            z = z.repeat_interleave(num_diffusion_samples, dim=0)
        if num_diffusion_samples > 1:
            attention_mask = attention_mask.repeat_interleave(num_diffusion_samples, dim=0)

        g = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)
        logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * self.scale

        pair_bias = self.pair_bias_proj(self.pair_norm(z))
        logits = logits + pair_bias.to(dtype=logits.dtype)

        min_val = torch.finfo(logits.dtype).min
        mask_bias = torch.where(attention_mask.bool()[:, None, :, None], 0.0, min_val)
        logits = logits + mask_bias.to(dtype=logits.dtype)

        attn = torch.softmax(logits, dim=-2).to(dtype=v.dtype)
        ctx = torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)
        ctx = g * ctx
        out = self.out_proj(ctx.reshape(bsz, n_queries, d_model))

        return torch.sigmoid(self.out_gate(s)) * out


class ConditionedTransitionBlock(nn.Module):
    """Conditioned SwiGLU transition with adaptive layer norm."""

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        transition_multiplier: int = 2,
    ) -> None:
        super().__init__()
        hidden = transition_multiplier * d_model

        self.adaln = AdaptiveLayerNorm(d_model, d_cond, eps=1e-5)
        self.output_gate = nn.Linear(d_cond, d_model, bias=True)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.constant_(self.output_gate.bias, -2.0)

        self.lin_swish = nn.Linear(d_model, 2 * hidden, bias=False)
        self.lin_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        x = self.adaln(a, s)

        swish_a, swish_b = self.lin_swish(x).chunk(2, dim=-1)
        b = F.silu(swish_a) * swish_b
        out = self.lin_out(b)

        return torch.sigmoid(self.output_gate(s)) * out


class DiffusionTransformer(nn.Module):
    """Diffusion denoising transformer with attention pair bias."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        num_heads: int,
        num_blocks: int,
        d_cond: int,
        transition_multiplier: int = 2,
    ) -> None:
        super().__init__()

        self.attn_blocks = nn.ModuleList(
            [
                AttentionPairBias(
                    d_model=d_model,
                    d_pair=d_pair,
                    num_heads=num_heads,
                    d_cond=d_cond,
                )
                for _ in range(num_blocks)
            ]
        )
        self.transition_blocks = nn.ModuleList(
            [
                ConditionedTransitionBlock(
                    d_model=d_model,
                    d_cond=d_cond,
                    transition_multiplier=transition_multiplier,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        a: Tensor,
        s: Tensor,
        z: Tensor,
        attention_mask: Tensor,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        x = a
        for attn, transition in zip(self.attn_blocks, self.transition_blocks):
            x = x + attn(
                x,
                s,
                z,
                attention_mask=attention_mask,
                num_diffusion_samples=num_diffusion_samples,
            )
            x = x + transition(x, s)
            if return_intermediates:
                intermediates.append(x)
        return x, intermediates


class DiffusionConditioning(nn.Module):
    """Conditions pair and single representations on noise timestep."""

    def __init__(
        self,
        c_z: int = 256,
        c_s: int = 768,
        c_s_inputs: int = 451,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        transition_multiplier: int = 2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)

        self.z_input_norm = nn.LayerNorm(2 * c_z, eps=layer_norm_eps)
        self.z_proj = nn.Linear(2 * c_z, c_z, bias=False)
        self.z_transitions = nn.ModuleList(
            [TransitionLayer(c_z, n=transition_multiplier, eps=layer_norm_eps) for _ in range(2)]
        )

        self.s_input_norm = nn.LayerNorm(c_s_inputs, eps=layer_norm_eps)
        self.s_proj = nn.Linear(c_s_inputs, c_s, bias=False)
        self.fourier = FourierEmbedding(fourier_dim)
        self.noise_norm = nn.LayerNorm(fourier_dim, eps=layer_norm_eps)
        self.noise_proj = nn.Linear(fourier_dim, c_s, bias=False)
        self.s_transitions = nn.ModuleList(
            [TransitionLayer(c_s, n=transition_multiplier, eps=layer_norm_eps) for _ in range(2)]
        )

    def condition_pair(
        self,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
    ) -> Tensor:
        r"""Condition step-invariant pair features."""

        z_rel = relative_position_encoding.to(dtype=torch.float32)
        z = torch.cat([z_trunk.to(dtype=torch.float32), z_rel], dim=-1)
        z = self.z_proj(self.z_input_norm(z))
        with torch.autocast(device_type="cuda", enabled=z.is_cuda, dtype=torch.bfloat16):
            for block in self.z_transitions:
                z = z + block(z)
        return z

    def condition_single(
        self,
        t_hat: Tensor,
        s_inputs: Tensor,
        num_diffusion_samples: int = 1,
    ) -> Tensor:
        r"""Condition single features on one diffusion noise step."""

        s_inputs_eff = s_inputs if num_diffusion_samples == 1 else s_inputs.repeat_interleave(num_diffusion_samples, 0)
        s = self.s_proj(self.s_input_norm(s_inputs_eff.to(dtype=torch.float32)))

        t = t_hat.to(dtype=torch.float32, device=s.device).reshape(-1)
        t_noise = 0.25 * torch.log((t / self.sigma_data).clamp(min=1e-20))
        n = self.fourier(t_noise)
        n = self.noise_proj(self.noise_norm(n))
        s = s + n.unsqueeze(1)

        for block in self.s_transitions:
            s = s + block(s)
        return s

    def forward(
        self,
        t_hat: Tensor,
        s_inputs: Tensor,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
        num_diffusion_samples: int = 1,
    ) -> tuple[Tensor, Tensor]:
        r"""Condition single and pair features."""

        return (
            self.condition_single(t_hat, s_inputs, num_diffusion_samples),
            self.condition_pair(z_trunk, relative_position_encoding),
        )


class DiffusionModule(nn.Module):
    """Diffusion denoising module for structure prediction."""

    def __init__(
        self,
        c_atom: int = 128,
        c_token: int = 768,
        c_z: int = 256,
        c_s_inputs: int = 451,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        atom_num_blocks: int = 3,
        atom_num_heads: int = 4,
        token_num_blocks: int = 12,
        token_num_heads: int = 16,
        transition_multiplier: int = 2,
        swa_window_size: int = 128,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)

        self.conditioning = DiffusionConditioning(
            c_z=c_z,
            c_s=c_token,  # conditioning s output is c_token
            c_s_inputs=c_s_inputs,
            sigma_data=sigma_data,
            fourier_dim=fourier_dim,
            transition_multiplier=transition_multiplier,
        )

        # Atom encoder (structure_prediction=True, with coords_linear)
        self.atom_encoder = ESMFold2AtomEncoder(
            d_atom=c_atom,
            d_token=c_token,
            n_blocks=atom_num_blocks,
            n_heads=atom_num_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=2,
            structure_prediction=True,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        # Atom decoder
        self.atom_decoder = ESMFold2AtomDecoder(
            d_atom=c_atom,
            d_token=c_token,
            n_blocks=atom_num_blocks,
            n_heads=atom_num_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=2,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        self.s_to_token = nn.Linear(c_token, c_token, bias=False)
        nn.init.zeros_(self.s_to_token.weight)

        # Token transformer (DiffusionTransformer with pair bias)
        self.token_transformer = DiffusionTransformer(
            d_model=c_token,
            d_pair=c_z,
            num_heads=token_num_heads,
            num_blocks=token_num_blocks,
            d_cond=c_token,
            transition_multiplier=transition_multiplier,
        )

        self.s_step_norm = nn.LayerNorm(c_token)
        self.token_norm = nn.LayerNorm(c_token)

    def denoise(
        self,
        *,
        x_noisy: Tensor,
        t_hat: Tensor,
        s_inputs: Tensor,
        conditioned_pair: Tensor,
        token_attention_mask: Tensor,
        reference_atom_features: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        expanded_atom_mask: Tensor,
        expanded_atom_to_token: Tensor,
        num_diffusion_samples: int = 1,
        return_token_repr: bool = False,
        return_atom_repr: bool = False,
    ) -> dict[str, Tensor | None]:
        r"""Denoise from explicit step-invariant pair and atom tensors."""

        t = t_hat.to(dtype=torch.float32, device=x_noisy.device).reshape(-1)
        s = self.conditioning.condition_single(
            t,
            s_inputs,
            num_diffusion_samples,
        )

        denom = torch.sqrt(t * t + self.sigma_data * self.sigma_data)
        r_noisy = x_noisy / denom[:, None, None]

        a, q_skip, c_skip, enc_intermediates = self.atom_encoder.encode(
            reference_atom_features=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            expanded_atom_mask=expanded_atom_mask,
            expanded_atom_to_token=expanded_atom_to_token,
            num_tokens=s_inputs.shape[1],
            r_l=r_noisy,
            return_intermediates=return_atom_repr,
        )

        a = a + self.s_to_token(self.s_step_norm(s))
        a, _ = self.token_transformer(
            a,
            s,
            conditioned_pair,
            attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
        )
        a = self.token_norm(a)

        r_update, dec_intermediates = self.atom_decoder(
            a_i=a,
            q_l=q_skip,
            c_l=c_skip,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            atom_mask=expanded_atom_mask,
            expanded_atom_to_token=expanded_atom_to_token,
            return_intermediates=return_atom_repr,
        )

        sigma2 = self.sigma_data * self.sigma_data
        t2 = t * t
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((self.sigma_data * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update

        atom_intermediates = torch.stack(enc_intermediates + dec_intermediates, dim=2) if return_atom_repr else None

        return {
            "x_denoised": out,
            "token_repr": a if return_token_repr else None,
            "atom_intermediates": atom_intermediates,
        }

    def forward(
        self,
        x_noisy: Tensor,
        t_hat: Tensor,
        ref_pos: Tensor,
        ref_charge: Tensor,
        ref_mask: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        tok_idx: Tensor,
        s_inputs: Tensor,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
        token_attention_mask: Tensor,
        num_diffusion_samples: int = 1,
        return_token_repr: bool = False,
        return_atom_repr: bool = False,
    ) -> dict[str, Tensor | None]:
        r"""Compute step-invariant tensors once, then denoise one noise step."""

        conditioned_pair = self.conditioning.condition_pair(
            z_trunk,
            relative_position_encoding,
        )
        (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        ) = self.atom_encoder.encode_reference(
            ref_pos,
            ref_mask,
            ref_space_uid,
            ref_charge,
            ref_element,
            ref_atom_name_chars,
            tok_idx,
            num_diffusion_samples,
        )
        return self.denoise(
            x_noisy=x_noisy,
            t_hat=t_hat,
            s_inputs=s_inputs,
            conditioned_pair=conditioned_pair,
            token_attention_mask=token_attention_mask,
            reference_atom_features=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            expanded_atom_mask=expanded_atom_mask,
            expanded_atom_to_token=expanded_atom_to_token,
            num_diffusion_samples=num_diffusion_samples,
            return_token_repr=return_token_repr,
            return_atom_repr=return_atom_repr,
        )


class DiffusionStructureHead(nn.Module):
    """Wrapper around DiffusionModule with diffusion sampling."""

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        dm = config.structure_head.diffusion_module
        swa_cfg = config.inputs.atom_encoder
        sh = config.structure_head

        self.diffusion_module = DiffusionModule(
            c_atom=dm.atom_size,
            c_token=dm.token_size,
            c_z=dm.pair_size,
            c_s_inputs=dm.input_size,
            sigma_data=dm.sigma_data,
            fourier_dim=dm.fourier_dim,
            atom_num_blocks=dm.num_atom_blocks,
            atom_num_heads=dm.num_atom_attention_heads,
            token_num_blocks=dm.num_token_blocks,
            token_num_heads=dm.num_token_attention_heads,
            transition_multiplier=dm.transition_multiplier,
            swa_window_size=swa_cfg.sliding_window_size,
            spatial_rope_base_frequency=swa_cfg.spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=swa_cfg.num_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=swa_cfg.num_uid_rope_pairs,
            uid_rope_base_frequency=swa_cfg.uid_rope_base_frequency,
        )

        # Sampling hyperparameters
        self.sigma_data = dm.sigma_data
        self.gamma_0 = sh.gamma_0
        self.gamma_min = sh.gamma_min
        self.noise_scale = sh.noise_scale
        self.step_scale = sh.step_scale
        self.inference_s_max = sh.inference_s_max
        self.inference_s_min = sh.inference_s_min
        self.inference_p = sh.inference_p
        self.inference_num_steps = sh.num_sampling_steps

    def inference_noise_schedule(self, num_steps: int | None = None, device: torch.device | None = None) -> Tensor:
        """Karras power-law noise schedule."""
        steps = self.inference_num_steps if num_steps is None else num_steps
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        if steps == 1:
            return torch.tensor(
                [self.inference_s_max * self.sigma_data, 0.0],
                device=device,
                dtype=torch.float32,
            )
        p = self.inference_p
        inv_p = 1.0 / p
        k = torch.arange(steps, device=device, dtype=torch.float32)
        base = self.inference_s_max**inv_p + (k / (steps - 1)) * (
            self.inference_s_min**inv_p - self.inference_s_max**inv_p
        )
        schedule = self.sigma_data * base.pow(p)
        return F.pad(schedule, (0, 1), value=0.0)

    @staticmethod
    def _random_rotations(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        q = torch.randn((n, 4), dtype=dtype, device=device)
        scale = torch.sqrt((q * q).sum(dim=1))
        signs = torch.where(q[:, 0] < 0, -scale, scale)
        q = q / signs[:, None]
        r, i, j, k = torch.unbind(q, dim=-1)
        two_s = 2.0 / (q * q).sum(dim=-1)
        return torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            dim=-1,
        ).reshape(n, 3, 3)

    def center_random_augmentation(
        self, x: Tensor, atom_mask: Tensor, second_coords: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Algorithm 19: center + random rotation + translation."""
        bsz = x.shape[0]
        mask = atom_mask.unsqueeze(-1)  # [B, A, 1]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (x * mask).sum(dim=1, keepdim=True) / denom
        x = x - mean
        if second_coords is not None:
            second_coords = second_coords - mean

        r = self._random_rotations(bsz, x.dtype, x.device)
        x = torch.einsum("bmd,bds->bms", x, r)
        if second_coords is not None:
            second_coords = torch.einsum("bmd,bds->bms", second_coords, r)

        t = torch.randn_like(x[:, 0:1, :])
        x = x + t
        if second_coords is not None:
            second_coords = second_coords + t
        return x, second_coords

    @staticmethod
    def _weighted_rigid_align(x: Tensor, x_gt: Tensor, w: Tensor, mask: Tensor) -> Tensor:
        """Kabsch alignment: align x to x_gt with weights w."""
        w = (mask * w).unsqueeze(-1)  # [B, N, 1]
        denom = w.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        mu = (x * w).sum(dim=-2, keepdim=True) / denom
        mu_gt = (x_gt * w).sum(dim=-2, keepdim=True) / denom
        x_c = x - mu
        xgt_c = x_gt - mu_gt
        H = torch.einsum("bni,bnj->bij", w * xgt_c, x_c)
        H32 = H.float()
        U, _, Vh = torch.linalg.svd(H32, driver="gesvd" if H32.is_cuda else None)
        det = torch.linalg.det(U @ Vh)
        ones = torch.ones_like(det)
        R = (U @ torch.diag_embed(torch.stack([ones, ones, det], dim=-1)) @ Vh).to(H.dtype)
        return x_c @ R.transpose(-1, -2) + mu_gt

    @torch.inference_mode()
    def sample(
        self,
        z_trunk: Tensor,
        s_inputs: Tensor,
        relative_position_encoding: Tensor,
        ref_pos: Tensor,
        ref_charge: Tensor,
        ref_mask: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        tok_idx: Tensor,
        token_attention_mask: Tensor,
        num_diffusion_samples: int = 1,
        num_sampling_steps: int | None = None,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        return_atom_repr: bool = False,
        denoising_early_exit_rmsd: float | None = None,
    ) -> dict[str, Tensor | None]:
        """Diffusion sampling (Algorithm 18).

        The Karras schedule is built with ``num_sampling_steps`` entries, then
        clipped to ``max_inference_sigma`` (the high-σ tail above the cap is
        dropped and the cap prepended). The number of denoising steps actually
        run is therefore fewer than ``num_sampling_steps`` whenever the schedule
        extends above the cap.
        """
        if num_diffusion_samples <= 0:
            raise ValueError("num_diffusion_samples must be positive")
        n_atoms = tok_idx.shape[1]
        device = s_inputs.device
        target_batch = s_inputs.shape[0] * num_diffusion_samples
        conditioned_pair = self.diffusion_module.conditioning.condition_pair(
            z_trunk,
            relative_position_encoding,
        )
        (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_mask,
            expanded_atom_to_token,
        ) = self.diffusion_module.atom_encoder.encode_reference(
            ref_pos,
            ref_mask,
            ref_space_uid,
            ref_charge,
            ref_element,
            ref_atom_name_chars,
            tok_idx,
            num_diffusion_samples,
        )

        schedule = self.inference_noise_schedule(num_sampling_steps, device)
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= float(max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))

        lam = self.noise_scale if noise_scale is None else float(noise_scale)
        eta = self.step_scale if step_scale is None else float(step_scale)

        x = schedule[0] * torch.randn(target_batch, n_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()

        gammas = torch.where(
            schedule > self.gamma_min,
            torch.full_like(schedule, self.gamma_0),
            torch.zeros_like(schedule),
        )

        x_denoised_prev: Tensor | None = None
        token_repr: Tensor | None = None
        diff_atom_intermediates: Tensor | None = None

        step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))
        num_steps = len(step_pairs)

        # This loop stays separate from the shared CoordinateEdmSampler by design (Frankenstein principle):
        # ESMFold-2's sampler is data-dependent control flow -- the schedule is truncated at max_inference_sigma
        # (fewer steps than num_sampling_steps) and the loop can early-exit once the denoised-coord RMSD stops
        # moving (denoising_early_exit_rmsd). The shared sampler walks a fixed step count with no early exit, so
        # this is genuine structural divergence and is left as-is.
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(step_pairs):
            x, x_denoised_prev = self.center_random_augmentation(x, atom_mask, second_coords=x_denoised_prev)

            sigma_tm_val = float(sigma_tm.item())
            t_hat_val = sigma_tm_val * (1.0 + float(gamma.item()))
            eps_std = lam * max(t_hat_val**2 - sigma_tm_val**2, 0.0) ** 0.5
            x_noisy = x + eps_std * torch.randn_like(x)

            is_last_step = step_idx == num_steps - 1
            request_atom_repr = return_atom_repr and (is_last_step or denoising_early_exit_rmsd is not None)

            dm_out = self.diffusion_module.denoise(
                x_noisy=x_noisy,
                t_hat=torch.full((target_batch,), t_hat_val, device=device, dtype=torch.float32),
                s_inputs=s_inputs,
                conditioned_pair=conditioned_pair,
                token_attention_mask=token_attention_mask,
                reference_atom_features=reference_atom_features,
                attention_cos=attention_cos,
                attention_sin=attention_sin,
                expanded_atom_mask=expanded_atom_mask,
                expanded_atom_to_token=expanded_atom_to_token,
                num_diffusion_samples=num_diffusion_samples,
                return_token_repr=True,
                return_atom_repr=request_atom_repr,
            )

            x_denoised = cast(Tensor, dm_out["x_denoised"])
            token_repr = cast(Tensor, dm_out["token_repr"])
            if request_atom_repr:
                diff_atom_intermediates = cast(Tensor | None, dm_out["atom_intermediates"])

            # Reverse diffusion alignment (Kabsch)
            with torch.autocast(device_type="cuda", enabled=False):
                x_noisy = self._weighted_rigid_align(x_noisy.float(), x_denoised.float(), atom_mask, atom_mask)
            x_noisy = x_noisy.to(dtype=x_denoised.dtype)

            # ODE/SDE step
            sigma_t_val = float(sigma_t.item())
            denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
            x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma

            # Denoising early-exit: stop when consecutive predictions converge
            if denoising_early_exit_rmsd is not None and x_denoised_prev is not None and step_idx >= 1:
                with torch.autocast(device_type="cuda", enabled=False):
                    aligned = self._weighted_rigid_align(
                        x_denoised_prev.float(),
                        x_denoised.float(),
                        atom_mask,
                        atom_mask,
                    )
                diff = (x_denoised.float() - aligned) * atom_mask.unsqueeze(-1)
                per_sample_rmsd = (diff.pow(2).sum(dim=(-1, -2)) / atom_mask.sum(dim=-1).clamp(min=1)).sqrt()
                if per_sample_rmsd.max().item() < denoising_early_exit_rmsd:
                    x = x_denoised
                    x_denoised_prev = x_denoised
                    break

            x_denoised_prev = x_denoised

        result: dict[str, Tensor | None] = {
            "sample_atom_coords": x,
            "diff_token_repr": token_repr,
        }
        if return_atom_repr:
            result["diff_atom_intermediates"] = diff_atom_intermediates
        return result


class RowAttentionPooling(nn.Module):
    """Row-wise attention pooling: attn_proj, out_proj."""

    def __init__(self, d_pair: int, d_single: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(d_pair, 1, bias=False)
        self.out_proj = nn.Linear(d_pair, d_single, bias=False)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        scores = self.attn_proj(z).squeeze(-1)
        mask_bias = torch.where(
            mask[:, None, :].bool(),
            torch.zeros_like(scores),
            torch.full_like(scores, torch.finfo(scores.dtype).min),
        )
        scores = scores + mask_bias
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, z)
        return self.out_proj(pooled)


class InputsEmbedder(nn.Module):
    """Embeds input features including atom-level encoding via SWA attention."""

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        swa_cfg = config.inputs.atom_encoder

        self.atom_attention_encoder = ESMFold2AtomEncoder(
            d_atom=swa_cfg.atom_size,
            d_token=swa_cfg.token_size,
            n_blocks=swa_cfg.num_blocks,
            n_heads=swa_cfg.num_attention_heads,
            swa_window_size=swa_cfg.sliding_window_size,
            expansion_ratio=swa_cfg.expansion_ratio,
            structure_prediction=False,  # no coords_linear
            spatial_rope_base_frequency=swa_cfg.spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=swa_cfg.num_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=swa_cfg.num_uid_rope_pairs,
            uid_rope_base_frequency=swa_cfg.uid_rope_base_frequency,
        )

    def forward(
        self,
        aatype: Tensor,
        profile: Tensor,
        deletion_mean: Tensor,
        ref_pos: Tensor,
        atom_attention_mask: Tensor,
        ref_space_uid: Tensor,
        ref_charge: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_to_token: Tensor,
    ) -> Tensor:
        """Embed inputs into per-token features.

        Returns:
            [B, L, d_inputs] concatenation of atom encoding, aatype, profile,
            and deletion_mean.
        """
        a, _q, _c, _intermediates = self.atom_attention_encoder(
            ref_pos=ref_pos,
            atom_attention_mask=atom_attention_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_to_token=atom_to_token,
            num_tokens=aatype.shape[1],
        )
        return torch.cat([a, aatype, profile, deletion_mean.unsqueeze(-1)], dim=-1)


class ResIdxAsymIdSymIdEntityIdEncoding(nn.Module):
    """embed.weight [d_pair, n_features] where n_features = 2*(2*r_bins+2) + 1 + (2*c_bins+2).

    For default r_bins=32, c_bins=2: 2*66 + 1 + 6 = 139.
    """

    def __init__(
        self,
        n_relative_residx_bins: int = 32,
        n_relative_chain_bins: int = 2,
        d_pair: int = 256,
    ) -> None:
        super().__init__()
        self.n_relative_residx_bins = n_relative_residx_bins
        self.n_relative_chain_bins = n_relative_chain_bins

        n_feats_residue = 2 * n_relative_residx_bins + 2
        n_feats_token = 2 * n_relative_residx_bins + 2
        n_feats_chain = 2 * n_relative_chain_bins + 2
        n_feats_same_entity = 1
        total_feats = n_feats_residue + n_feats_token + n_feats_chain + n_feats_same_entity
        self.embed = nn.Linear(total_feats, d_pair, bias=False)

    def forward(
        self,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
    ) -> Tensor:
        bij_same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)
        bij_same_residue = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
        bij_same_entity = entity_id.unsqueeze(2) == entity_id.unsqueeze(1)

        dij_residue = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        dij_residue = torch.clip(
            dij_residue + self.n_relative_residx_bins,
            0,
            2 * self.n_relative_residx_bins,
        )
        dij_residue = torch.where(bij_same_chain, dij_residue, 2 * self.n_relative_residx_bins + 1)
        aij_rel_pos = F.one_hot(dij_residue, 2 * self.n_relative_residx_bins + 2)

        dij_token = torch.clip(
            token_index.unsqueeze(2) - token_index.unsqueeze(1) + self.n_relative_residx_bins,
            0,
            2 * self.n_relative_residx_bins,
        )
        dij_token = torch.where(
            bij_same_chain & bij_same_residue,
            dij_token,
            2 * self.n_relative_residx_bins + 1,
        )
        aij_rel_token = F.one_hot(dij_token, 2 * self.n_relative_residx_bins + 2)

        dij_chain = torch.clip(
            sym_id.unsqueeze(2) - sym_id.unsqueeze(1) + self.n_relative_chain_bins,
            0,
            2 * self.n_relative_chain_bins,
        )
        dij_chain = torch.where(bij_same_chain, 2 * self.n_relative_chain_bins + 1, dij_chain)
        aij_rel_chain = F.one_hot(dij_chain, 2 * self.n_relative_chain_bins + 2)

        feats = torch.cat(
            [
                aij_rel_pos.float(),
                aij_rel_token.float(),
                bij_same_entity.float().unsqueeze(-1),
                aij_rel_chain.float(),
            ],
            dim=-1,
        )

        return self.embed(feats)


class SingleToPair(nn.Module):
    """downproject, output_mlp (Sequential of Linear, GELU, Linear)."""

    def __init__(self, input_dim: int, downproject_dim: int, output_dim: int) -> None:
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * downproject_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downproject(x)
        x = torch.cat(
            [(x.unsqueeze(2) * x.unsqueeze(1)), (x.unsqueeze(2) - x.unsqueeze(1))],
            dim=3,
        )
        return self.output_mlp(x)


class LanguageModelPairEncoder(nn.Module):
    """Project and combine ESMC layer states into the pair representation.

    Contains:
    - base_z_combine: nn.Parameter [num_layers+1]
    - base_z_linear: Sequential(LayerNorm(d_model), Linear(d_model, d_z, bias=False))
    - base_z_mlp: Sequential(SingleToPair(d_z, d_z, d_z), LayerNorm(d_z))
    """

    def __init__(self, d_z: int = 256, d_model: int = 2560, num_layers: int = 80) -> None:
        super().__init__()

        self.base_z_mlp = nn.Sequential(SingleToPair(d_z, d_z, d_z), nn.LayerNorm(d_z))
        self.base_z_linear = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_z, bias=False))
        self.base_z_combine = nn.Parameter(torch.zeros(num_layers + 1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project pre-computed ESMC hidden states to pair representation.

        Args:
            hidden_states: [B, L, num_layers+1, d_model] from ESMC 6B.
        Returns:
            [B, L, L, d_pair] pair representation.
        """
        lm_z = self.base_z_linear(hidden_states)  # [B, L, 81, d_z]
        weights = self.base_z_combine.softmax(0)  # [81]
        # Keep the released batched-matmul reduction order.  Do not squeeze:
        # ``weights @ lm_z`` already has shape [B, L, d_z], including when L=1.
        lm_z = weights @ lm_z  # [B, L, d_z]
        return self.base_z_mlp(lm_z)  # [B, L, L, d_z]


def compute_esmc_hidden_states(
    esmc: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    token_to_esmc: Tensor,
    token_has_esmc: Tensor,
    lm_mask_pct: float = 0.0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    mask_token_id: int = 4,
) -> Tensor:
    """Run ESMC directly and gather Biohub-compatible per-layer states."""
    if lm_mask_pct > 0.0:
        special = (input_ids == bos_token_id) | (input_ids == eos_token_id) | ~attention_mask
        do_mask = (torch.rand(input_ids.shape, device=input_ids.device) < lm_mask_pct) & ~special
        input_ids = input_ids.masked_fill(do_mask, mask_token_id)

    with torch.autocast(device_type="cuda", enabled=input_ids.is_cuda, dtype=torch.bfloat16):
        hidden = esmc.embeddings(input_ids=input_ids)
        attention_bias = create_bidirectional_mask(
            config=esmc.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
        )
        layers: list[Tensor] = [hidden]
        for layer in esmc.encoder.layer:
            hidden = layer(hidden, attention_bias)
            layers.append(hidden)
        layers[-1] = esmc.encoder.layer_norm(hidden)
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


class TriangleMultiplicativeBlock(nn.Module):
    """Triangle multiplicative update block with gated signal routing."""

    _FLOW_TO_EINSUM = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}
    _VALID_FLOWS = ("outgoing", "incoming")

    def __init__(self, input_channels: int, latent_channels: int, flow: str) -> None:
        super().__init__()
        if flow not in self._FLOW_TO_EINSUM:
            raise ValueError(f"Invalid flow={flow!r}. Expected one of {self._VALID_FLOWS}.")

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.flow = flow
        self.operator_backend = "eager"
        self._einsum_equation = self._FLOW_TO_EINSUM[flow]
        self.norm_start = nn.LayerNorm(self.input_channels, eps=_EPS)
        self.norm_mix = nn.LayerNorm(self.latent_channels, eps=_EPS)
        self.proj_bundle = nn.Linear(self.input_channels, 4 * self.latent_channels, bias=False)
        self.proj_emit = nn.Linear(self.latent_channels, self.input_channels, bias=False)
        self.proj_gate = nn.Linear(self.input_channels, self.input_channels, bias=False)

        # Default chunked for memory on long sequences; tests override with
        # ``set_chunk_size(None)`` for the unchunked path under bit-exact bf16
        # parity checks.
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def _triangular_contract(self, left_stream: Tensor, right_stream: Tensor) -> Tensor:
        return torch.einsum(self._einsum_equation, left_stream, right_stream)

    def _triangular_contract_chunked(self, left_stream: Tensor, right_stream: Tensor, chunk_size: int) -> Tensor:
        """Compute the triangular einsum in chunks along the output i-dimension."""
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

        resolved = resolve_tensor_operator_backend(
            self.operator_backend,
            tensors=(pair_grid, visibility),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            left_value, right_value, left_gate, right_gate = self.proj_bundle.weight.chunk(4, dim=0)
            return dispatch_triangle_multiplication(
                pair_grid,
                visibility,
                outgoing=self.flow == "outgoing",
                norm_input_weight=self.norm_start.weight,
                norm_input_bias=self.norm_start.bias,
                left_value_weight=left_value,
                left_gate_weight=left_gate,
                right_value_weight=right_value,
                right_gate_weight=right_gate,
                norm_output_weight=self.norm_mix.weight,
                norm_output_bias=self.norm_mix.bias,
                output_weight=self.proj_emit.weight,
                output_gate_weight=self.proj_gate.weight,
                eps=self.norm_start.eps,
                operator_backend="oops",
            )

        normalized_grid = self.norm_start(pair_grid)
        bundled = self.proj_bundle(normalized_grid)
        signal, gate_logits = bundled.split(2 * self.latent_channels, dim=-1)
        routed = signal * torch.sigmoid(gate_logits)
        routed = routed * visibility.unsqueeze(-1)

        left_stream, right_stream = routed.float().chunk(2, dim=-1)
        if self._chunk_size is not None:
            contracted = self._triangular_contract_chunked(left_stream, right_stream, self._chunk_size)
        else:
            contracted = self._triangular_contract(left_stream, right_stream)
        mixed = self.proj_emit(self.norm_mix(contracted))
        output_gate = torch.sigmoid(self.proj_gate(normalized_grid))
        return mixed * output_gate


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangular multiplicative update with explicit incoming/outgoing orientation."""

    def __init__(self, dim: int = 128, outgoing: bool = True) -> None:
        super().__init__()
        flow = "outgoing" if outgoing else "incoming"
        self._engine = TriangleMultiplicativeBlock(input_channels=dim, latent_channels=dim, flow=flow)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._engine.set_chunk_size(chunk_size)

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        return self._engine(z, visibility=mask)


class Transition(nn.Module):
    """LN + SwiGLU FFN."""

    def __init__(self, d_model: int, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.operator_backend = "eager"
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUMLP(d_model, expansion_ratio=expansion_ratio, bias=False)
        # Default chunked; set_chunk_size(None) disables for bit-exact parity tests.
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        resolved = resolve_tensor_operator_backend(
            self.operator_backend,
            tensors=(x,),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            gate_weight, up_weight = self.ffn.w12.weight.chunk(2, dim=0)
            return x + dispatch_transition(
                x,
                norm_weight=self.norm.weight,
                norm_bias=self.norm.bias,
                gate_weight=gate_weight,
                up_weight=up_weight,
                down_weight=self.ffn.w3.weight,
                eps=self.norm.eps,
                operator_backend="oops",
            )
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            return x + self.ffn(self.norm(x))
        out_list: list[Tensor] = []
        for s in range(0, x.shape[1], self._chunk_size):
            e = min(s + self._chunk_size, x.shape[1])
            sl = x[:, s:e]
            out_list.append(sl + self.ffn(self.norm(sl)))
        return torch.cat(out_list, dim=1)


class PairUpdateBlock(nn.Module):
    """tri_mul_out, tri_mul_in, pair_transition."""

    def __init__(self, d_pair: int = 256, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=d_pair, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=d_pair, outgoing=False)
        self.pair_transition = Transition(d_pair, expansion_ratio=expansion_ratio)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = self.pair_transition(pair)
        return pair


class FoldingTrunk(nn.Module):
    """ModuleList of PairUpdateBlocks."""

    def __init__(self, n_layers: int = 24, d_pair: int = 256, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [PairUpdateBlock(d_pair=d_pair, expansion_ratio=expansion_ratio) for _ in range(n_layers)]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            cast(PairUpdateBlock, block).set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            fn = partial(block, pair_attention_mask=pair_attention_mask)
            if torch.is_grad_enabled():
                pair = checkpoint(fn, pair, use_reentrant=False)  # pyright: ignore
            else:
                pair = fn(pair)
        return pair


class OuterProductMean(nn.Module):
    """Outer-product mean: maps an MSA representation into a pair update.

    The order of the ``/ n_valid`` divide vs. the ``Wout`` projection is
    selectable via ``divide_outer_before_proj`` because different ESMFold2
    checkpoints were trained with different orderings:

    * ``False`` (default): ``Wout(outer) / n_valid`` — the projection bias
      is scaled by 1/n_valid alongside the outer product.
    * ``True``: ``Wout(outer / n_valid)`` — the projection bias is added
      unscaled, post-divide.
    """

    def __init__(
        self,
        d_msa: int,
        d_hidden: int,
        d_pair: int,
        divide_outer_before_proj: bool = False,
    ) -> None:
        super().__init__()
        self.divide_outer_before_proj = divide_outer_before_proj
        self.operator_backend = "eager"
        self.norm = nn.LayerNorm(d_msa)
        self.W = nn.Linear(d_msa, 2 * d_hidden, bias=False)
        self.Wout = nn.Linear(d_hidden * d_hidden, d_pair, bias=True)
        # Off for bit-exact bf16; ``set_chunk_size(64)`` for long sequences.
        self._chunk_size: int | None = None

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, m: Tensor, msa_attention_mask: Tensor) -> Tensor:
        resolved = resolve_tensor_operator_backend(
            self.operator_backend,
            tensors=(m, msa_attention_mask),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            left_weight, right_weight = self.W.weight.chunk(2, dim=0)
            return dispatch_outer_product_mean(
                m.transpose(1, 2),
                msa_attention_mask.transpose(1, 2),
                norm_weight=self.norm.weight,
                norm_bias=self.norm.bias,
                proj_left_weight=left_weight,
                proj_right_weight=right_weight,
                proj_out_weight=self.Wout.weight,
                proj_out_bias=self.Wout.bias,
                norm_eps=self.norm.eps,
                denom_eps=0.0,
                divide_before_projection=self.divide_outer_before_proj,
                operator_backend="oops",
            )
        m_norm = self.norm(m)
        x = self.W(m_norm) * msa_attention_mask.unsqueeze(-1).to(m_norm.dtype)
        a, b = x.chunk(2, dim=-1)
        mask_f = msa_attention_mask.to(a.dtype)
        n_valid = (mask_f @ mask_f.transpose(-1, -2)).unsqueeze(-1).clamp(min=1.0)
        if self._chunk_size is None:
            outer = torch.einsum("bimc,bjmd->bijcd", a, b).flatten(-2)
            if self.divide_outer_before_proj:
                return self.Wout(outer / n_valid)
            return self.Wout(outer) / n_valid
        # Chunk along the left (i) axis so the peak einsum intermediate is
        # [B, chunk, L, c, d] instead of [B, L, L, c, d].
        L = a.shape[1]
        out_chunks: list[Tensor] = []
        for s in range(0, L, self._chunk_size):
            e = min(s + self._chunk_size, L)
            outer_chunk = torch.einsum("bimc,bjmd->bijcd", a[:, s:e], b).flatten(-2)
            if self.divide_outer_before_proj:
                out_chunks.append(self.Wout(outer_chunk / n_valid[:, s:e]))
            else:
                out_chunks.append(self.Wout(outer_chunk) / n_valid[:, s:e])
        return torch.cat(out_chunks, dim=1)


class MSAPairWeightedAveraging(nn.Module):
    """Pair-biased MSA row update (AF3 Supplement Algorithm 10)."""

    def __init__(self, d_msa: int, d_pair: int, n_heads: int = 8, head_width: int = 32) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_width = head_width
        self.operator_backend = "eager"
        self.norm_single = nn.LayerNorm(d_msa)
        self.compute_bias = nn.Sequential(nn.LayerNorm(d_pair), nn.Linear(d_pair, n_heads, bias=False))
        self.Wv = nn.Linear(d_msa, n_heads * head_width, bias=False)
        self.Wgate = nn.Linear(d_msa, n_heads * head_width, bias=False)
        self.Wout = nn.Linear(n_heads * head_width, d_msa, bias=False)

    def forward(self, msa_repr: Tensor, pair_repr: Tensor, pair_attention_mask: Tensor) -> Tensor:
        """
        Args:
            msa_repr:           [B, L, M, d_msa]
            pair_repr:          [B, L, L, d_pair]
            pair_attention_mask:[B, L, L]
        Returns:
            [B, L, M, d_msa]
        """
        resolved = resolve_tensor_operator_backend(
            self.operator_backend,
            tensors=(msa_repr, pair_repr, pair_attention_mask),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            return dispatch_pair_weighted_average(
                msa_repr.transpose(1, 2),
                pair_repr,
                pair_attention_mask,
                norm_msa_weight=self.norm_single.weight,
                norm_msa_bias=self.norm_single.bias,
                norm_pair_weight=self.compute_bias[0].weight,
                norm_pair_bias=self.compute_bias[0].bias,
                value_weight=self.Wv.weight,
                gate_weight=self.Wgate.weight,
                pair_logits_weight=self.compute_bias[1].weight,
                output_weight=self.Wout.weight,
                num_heads=self.n_heads,
                head_dim=self.head_width,
                norm_msa_eps=self.norm_single.eps,
                norm_pair_eps=self.compute_bias[0].eps,
                operator_backend="oops",
            ).transpose(1, 2)

        B, L, M, _ = msa_repr.shape
        h, dh = self.n_heads, self.head_width

        msa_normed = self.norm_single(msa_repr)
        bias = self.compute_bias(pair_repr)  # [B, L, L, n_heads]
        bias.masked_fill_(~pair_attention_mask.unsqueeze(-1).bool(), torch.finfo(bias.dtype).min)
        attn = torch.softmax(bias, dim=-2)  # softmax over j

        v = self.Wv(msa_normed).reshape(B, L, M, h, dh)
        gate = torch.sigmoid(self.Wgate(msa_normed)).reshape(B, L, M, h, dh)

        output = torch.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, gate)
        return self.Wout(output.reshape(B, L, M, h * dh))


_NONPOLYMER_ID = 3


class PairTransition(nn.Module):
    """LayerNorm + SwiGLU feed-forward residual block on the pair representation."""

    def __init__(self, hidden_size: int, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.operator_backend = "eager"
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = SwiGLUMLP(hidden_size, expansion_ratio=expansion_ratio, bias=False)
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        resolved = resolve_tensor_operator_backend(
            self.operator_backend,
            tensors=(x,),
            supported=("oops", "eager"),
            public_default="eager",
        )
        if resolved == "oops":
            gate_weight, up_weight = self.ffn.w12.weight.chunk(2, dim=0)
            return dispatch_transition(
                x,
                norm_weight=self.norm.weight,
                norm_bias=self.norm.bias,
                gate_weight=gate_weight,
                up_weight=up_weight,
                down_weight=self.ffn.w3.weight,
                eps=self.norm.eps,
                operator_backend="oops",
            )
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            return self.ffn(self.norm(x))
        out: list[Tensor] = []
        for s in range(0, x.shape[1], self._chunk_size):
            e = min(s + self._chunk_size, x.shape[1])
            sl = x[:, s:e]
            out.append(self.ffn(self.norm(sl)))
        return torch.cat(out, dim=1)


class ConfidenceHead(nn.Module):
    """Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins."""

    boundaries: Tensor

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        self.operator_backend = "eager"
        ch = config.confidence_head
        single_size = config.single_size
        pair_size = config.pair_size
        input_size = config.inputs.input_size

        boundaries = torch.linspace(ch.min_dist, ch.max_dist, ch.distogram_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(ch.distogram_bins, pair_size)

        self.s_to_z = nn.Linear(input_size, pair_size, bias=False)
        self.s_to_z_transpose = nn.Linear(input_size, pair_size, bias=False)
        self.s_to_z_prod_in1 = nn.Linear(input_size, pair_size, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(input_size, pair_size, bias=False)
        self.s_to_z_prod_out = nn.Linear(pair_size, pair_size, bias=False)
        self.s_inputs_norm = nn.LayerNorm(input_size)
        self.z_norm = nn.LayerNorm(pair_size)

        self.row_attention_pooling = RowAttentionPooling(pair_size, single_size)

        pf = ch.folding_trunk
        self.folding_trunk = FoldingTrunk(pf.num_blocks, pair_size, expansion_ratio=4)

        # Heads.
        self.plddt_ln = nn.LayerNorm(single_size)
        max_atoms_per_token = 23
        self.plddt_weight = nn.Parameter(torch.zeros(max_atoms_per_token, single_size, ch.num_plddt_bins))

        self.pae_ln = nn.LayerNorm(pair_size)
        self.pae_head = nn.Linear(pair_size, ch.num_pae_bins, bias=False)

        self.pde_ln = nn.LayerNorm(pair_size)
        self.pde_head = nn.Linear(pair_size, ch.num_pde_bins, bias=False)

        self.resolved_ln = nn.LayerNorm(single_size)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_weight = nn.Parameter(torch.zeros(max_atoms_per_token, single_size, 2))

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        return x if num_diffusion_samples == 1 else x.repeat_interleave(num_diffusion_samples, 0)

    def forward(
        self,
        s_inputs: Tensor,
        z: Tensor,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        relative_position_encoding: Tensor,
        token_bonds_encoding: Tensor,
        num_diffusion_samples: int = 1,
    ) -> dict[str, Tensor]:
        s_inputs_normed = self.s_inputs_norm(s_inputs)

        z_base = self.z_norm(z)
        z_base = z_base + relative_position_encoding + token_bonds_encoding
        z_base = z_base + self.s_to_z(s_inputs_normed).unsqueeze(2)
        z_base = z_base + self.s_to_z_transpose(s_inputs_normed).unsqueeze(1)
        z_base = z_base + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :] * self.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :]
        )

        pair = self._repeat_batch(z_base, num_diffusion_samples)
        atom_to_token_m = self._repeat_batch(atom_to_token, num_diffusion_samples)
        atom_mask_m = self._repeat_batch(atom_attention_mask, num_diffusion_samples)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        Bm = pair.shape[0]

        rep_coords = gather_rep_atom_coords(x_pred, rep_idx_m)
        rep_distances = torch.cdist(rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist")
        distogram_bins = (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()

        with torch.autocast(device_type="cuda", enabled=pair.is_cuda, dtype=torch.bfloat16):
            pair_delta = self.folding_trunk(pair, pair_attention_mask=pair_mask)
        pair.add_(pair_delta.float())
        del pair_delta
        single = self.row_attention_pooling(pair, mask)

        atom_mask_f = atom_mask_m.float()
        s_at_atoms = gather_token_to_atom(
            single,
            atom_to_token_m,
            operator_backend=self.operator_backend,
        )
        s_at_atoms_ln = self.plddt_ln(s_at_atoms)

        intra_idx = _compute_intra_token_idx(atom_to_token_m)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        w_plddt = self.plddt_weight[intra_idx]
        plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)
        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        L = single.shape[1]
        plddt_sum = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
        atom_count = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
        atom_mask_t = atom_mask_f.to(plddt_per_atom.dtype)
        plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_t)
        atom_count.scatter_add_(1, atom_to_token_m, atom_mask_t)
        plddt = plddt_sum / atom_count.clamp(min=torch.finfo(atom_count.dtype).tiny)

        complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (
            atom_mask_f.sum(dim=-1) + torch.finfo(atom_mask_f.dtype).tiny
        )

        expanded_type = self._repeat_batch(mol_type, num_diffusion_samples)
        expanded_asym = self._repeat_batch(asym_id, num_diffusion_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).float()
        inter_chain = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float()
        near_contact = (rep_distances < 8).float()
        interface_per_token = (near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(dim=-1)
        iplddt_weight = torch.where(
            is_ligand.bool(),
            torch.full_like(interface_per_token, 2.0),
            interface_per_token,
        )
        iplddt_weight_atoms = gather_token_to_atom(
            iplddt_weight.unsqueeze(-1),
            atom_to_token_m,
            operator_backend=self.operator_backend,
        ).squeeze(-1)
        atom_iplddt_w = atom_mask_f * iplddt_weight_atoms
        complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (
            atom_iplddt_w.sum(dim=-1) + torch.finfo(atom_iplddt_w.dtype).tiny
        )

        plddt_ca = plddt_per_atom.gather(1, rep_idx_m)

        # PAE
        pae_logits = self.pae_head(self.pae_ln(pair))
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0)

        # PDE
        pde_logits = self.pde_head(self.pde_ln(pair))
        pde = _categorical_mean(pde_logits, start=0.0, end=32.0)

        # Resolved (per-atom binary).
        s_at_atoms_res = self.resolved_ln(s_at_atoms)
        w_res = self.resolved_weight[intra_idx]
        resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)

        # pTM / ipTM from pae_logits.
        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        mask_f = mask.float()
        N_res = mask_f.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (N_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (
            pair_mask_2d.sum(dim=-1) + torch.finfo(pair_mask_2d.dtype).tiny
        )
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + torch.finfo(inter_chain_mask.dtype).tiny
        )
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_id = int(expanded_asym.max().item())
        n_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(Bm, n_chains, n_chains, device=tm_expected.device, dtype=tm_expected.dtype)
        for c1 in range(n_chains):
            chain_c1 = (expanded_asym == c1).float() * mask_f
            for c2 in range(n_chains):
                chain_c2 = (expanded_asym == c2).float() * mask_f
                pair_m = chain_c1.unsqueeze(-1) * chain_c2.unsqueeze(-2)
                denom = pair_m.sum(dim=(-1, -2)) + torch.finfo(pair_m.dtype).tiny
                pair_chains_iptm[:, c1, c2] = (tm_expected * pair_m).sum(dim=(-1, -2)) / denom

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt,
            "plddt_per_atom": plddt_per_atom,
            "plddt_ca": plddt_ca,
            "complex_plddt": complex_plddt,
            "complex_iplddt": complex_iplddt,
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm,
            "iptm": iptm,
            "pair_chains_iptm": pair_chains_iptm,
        }


class MSAEncoderBlock(nn.Module):
    """One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update."""

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        hidden_size: int,
        num_msa_attention_heads: int,
        msa_head_width: int,
        is_final_block: bool = False,
    ) -> None:
        super().__init__()
        self.is_final_block = is_final_block
        self.outer_product_mean = OuterProductMean(msa_size, hidden_size, pair_size)
        if not is_final_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                msa_size,
                pair_size,
                num_msa_attention_heads,
                msa_head_width,
            )
            self.msa_transition = PairTransition(msa_size, expansion_ratio=4)
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=pair_size, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=pair_size, outgoing=False)
        self.pair_transition = PairTransition(pair_size, expansion_ratio=4)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        if not self.is_final_block:
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
        if not self.is_final_block:
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = m + self.msa_transition(m)
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair


class MSAEncoder(nn.Module):
    """Stack of [`MSAEncoderBlock`] layers that conditions the pair on an MSA."""

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        input_size: int,
        hidden_size: int = 32,
        num_blocks: int = 4,
        num_msa_attention_heads: int = 8,
        msa_head_width: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(35, msa_size, bias=False)
        self.project_inputs = nn.Linear(input_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                MSAEncoderBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    hidden_size=hidden_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_head_width=msa_head_width,
                    is_final_block=(i == num_blocks - 1),
                )
                for i in range(num_blocks)
            ]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            cast(MSAEncoderBlock, block).set_chunk_size(chunk_size)

    def forward(
        self,
        x_pair: Tensor,
        x_inputs: Tensor,
        msa_oh: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        m_feat = torch.cat([msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
        m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)
        tok_mask = msa_attention_mask[:, :, 0].bool()
        pair_attention_mask = tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_attention_mask, pair_attention_mask)
        return x_pair


# ===========================================================================
# Reserved placeholder for the second ESMFold2 architecture id
# ``ofoldx.esmfold2_experimental``.
#
# This directory follows the same two-arch-one-directory pattern as the AlphaFold 2
# family (``ofoldx.alphafold2`` / ``ofoldx.alphafold2_multimer``) and the Protenix
# family (``ofoldx.protenix`` / ``ofoldx.protenix_constraint``): one package
# exporting more than one registered architecture id when the graphs are genuinely
# different rather than checkpoint variants of a single ``Model`` class.
#
# ``ofoldx.esmfold2`` is the released single-sequence fold graph (``ESMFold2Model``
# above). The ``type="experimental"`` checkpoints used by the released
# binder-inversion notebook have a **different recurrent graph and a different
# confidence contract** (see README, and the frozen ``ESMFold2Pipeline.design``
# signature), so they are a separate architecture and are never silently loaded as a
# variant of ``ESMFold2Model``.
#
# Those experimental checkpoints are not on node4/HF yet and the recurrent graph is
# not implemented this wave. The classes below are therefore **stubs**: they reserve
# the concrete symbols and the second-arch identity so the family's shape is
# materially present in the directory, but they carry no graph. Constructing any of
# them raises ``NotImplementedError``. The paired
# ``register_model("ofoldx.esmfold2_experimental", ...)`` call is deferred (see the
# registration note in ``__init__.py``) until the recurrent-graph weights and
# implementation land; per DESIGN.md 7.2 registration is ordinary explicit Python and
# is added at that point, not simulated here.
# ===========================================================================


_PENDING = (
    "ofoldx.esmfold2_experimental is a reserved placeholder: the experimental recurrent-graph "
    "checkpoints (biohub_esm type='experimental') are not local and the graph is not "
    "implemented this wave. Only the released ofoldx.esmfold2 fold graph is available."
)


class ESMFold2ExperimentalConfig(Config):
    r"""Stub configuration for the experimental recurrent ESMFold2 graph.

    Intentionally empty of graph shape: the recurrent-trunk/confidence hyperparameters are
    fixed by the yet-to-land ``type="experimental"`` checkpoints and are not guessed here.
    ``num_coda_blocks`` is carried only so the config is a non-trivial, distinct type from
    ``ESMFold2Config``; its value is provisional and MUST NOT be treated as parity-frozen.
    """

    num_coda_blocks: int = 2


class ESMFold2ExperimentalModel(nn.Module):
    r"""Stub for the experimental recurrent ESMFold2 architecture (weights/impl pending).

    Reserves ``config_class`` and the ``ofoldx.esmfold2_experimental`` identity. The
    recurrent graph is a different architecture from ``ESMFold2Model`` (different confidence
    contract) and is not implemented this wave; construction raises ``NotImplementedError``.
    """

    config_class = ESMFold2ExperimentalConfig

    def __init__(self, config: ESMFold2ExperimentalConfig | None = None, **_: Any) -> None:
        raise NotImplementedError(_PENDING)


class ESMFold2ExperimentalProcessor:
    r"""Stub Processor paired with the experimental recurrent ESMFold2 graph.

    Placeholder only; the experimental input contract lands with the recurrent-graph
    checkpoints. Construction raises ``NotImplementedError``.
    """

    config_class = ESMFold2ExperimentalConfig

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(_PENDING)


# ===========================================================================
# ESMFold2's family-owned, coordinate-producing binder-design result carriers.
#
# The shared ``DesignResult`` (``ofoldx.results.values``) is frozen coordinate-free and
# fixed-length (DESIGN.md 5.3.1). The released ESMFold2 binder-design operation instead
# produces a folded ``target|binder`` complex whose binder length varies across
# candidates, so it cannot reuse that carrier. These concrete classes therefore live in
# the ESMFold2 family directory and are distinct types, not subclasses of the shared
# ``DesignResult``. Upstream provenance: ``biohub_esm/cookbook/tutorials/binder_design.py``.
# ===========================================================================


@dataclass(eq=False)
class ESMFold2DesignCandidate:
    r"""One folded ``target|binder`` complex and its aligned scientific values.

    Each candidate owns the complete folded ``Structure`` of the target|binder complex.
    Binder length varies across candidates, which is the coordinate-producing,
    variable-length property that forbids reuse of the shared ``DesignCandidate``.
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    pair_metrics: dict[str, PairMetric] = field(default_factory=dict)


@dataclass(eq=False)
class ESMFold2DesignResult:
    r"""One generation-ordered collection of coordinate-producing binder designs.

    Tuple position is candidate identity for one generation context; there is no
    auxiliary index field, and no input/context/provenance field is carried.
    """

    candidates: tuple[ESMFold2DesignCandidate, ...]

    def best(self, score: str) -> ESMFold2DesignCandidate:
        r"""Return the earliest best candidate under one explicit score.

        Mirrors ``FoldResult.best`` exactly: the score's ``preferred`` direction must be
        consistently ``"higher"`` or ``"lower"``; unit and scope must match across every
        candidate; a NaN value or an undefined direction raises at ranking time; infinity
        is rankable; and the earliest candidate wins a tie.
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
