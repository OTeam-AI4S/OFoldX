r"""
AlphaFold 2 template embedding modules.

The parameter-bearing module names mirror the official AF2 template tree:
``template_embedding/single_template_embedding`` for pair-template features and
``template_single_embedding`` / ``template_projection`` for torsion rows appended
to the MSA.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from oops import LayerNorm
from torch import Tensor, nn

from ...data import residue_constants
from ...modules.attention import GatedMultiheadAttention
from ...modules.transition import RowwiseDropout
from ...modules.trunk.evoformer import (
    EvoformerTransition,
    TriangleAttention,
    TriangleMultiplication,
)

_GLY_INDEX = residue_constants.restypes.index("G")


class AlphaFold2TemplateEmbedding(nn.Module):
    r"""
    Official-style AF2 template pair embedding.
    """

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_bin: float,
        max_bin: float,
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
        self.single_template_embedding = AlphaFold2SingleTemplateEmbedding(
            template_size=template_size,
            num_blocks=num_blocks,
            num_distogram_bins=num_distogram_bins,
            min_bin=min_bin,
            max_bin=max_bin,
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

    def forward(self, query_embedding: Tensor, batch: Any, *, pair_mask: Tensor) -> Tensor:
        template_batch = _template_batch(batch, reference=query_embedding)
        if template_batch is None:
            return query_embedding.new_zeros(query_embedding.shape)

        template_pair = self.single_template_embedding(template_batch, pair_mask=pair_mask)
        batch_size, num_templates, num_tokens = template_pair.shape[:3]
        template_mask = template_batch.template_mask.to(device=query_embedding.device, dtype=query_embedding.dtype)
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


class AlphaFold2SingleTemplateEmbedding(nn.Module):
    r"""
    Embed and refine each template with the shared template pair stack.
    """

    def __init__(
        self,
        *,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_bin: float,
        max_bin: float,
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
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.vocab_size = vocab_size
        self.use_template_unit_vector = use_template_unit_vector
        template_feature_size = num_distogram_bins + 1 + 2 * vocab_size + 3 + 1
        self.embedding2d = nn.Linear(template_feature_size, template_size)
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
        self.output_layer_norm = LayerNorm(template_size)

    def forward(self, template_batch: AlphaFold2TemplateBatch, *, pair_mask: Tensor) -> Tensor:
        features, template_pair_mask = self._features(template_batch)
        batch_size, num_templates, num_tokens = features.shape[:3]
        pair_mask = pair_mask[:, None].expand(batch_size, num_templates, num_tokens, num_tokens)
        pair_mask = pair_mask * template_pair_mask

        act = self.embedding2d(features).reshape(batch_size * num_templates, num_tokens, num_tokens, -1)
        flat_pair_mask = pair_mask.reshape(batch_size * num_templates, num_tokens, num_tokens)
        act = self.template_pair_stack(act, pair_mask=flat_pair_mask)
        act = self.output_layer_norm(act)
        return act.reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        aatype = template_batch.template_aatype.long().clamp(min=0, max=self.vocab_size - 1)
        aatype_one_hot = F.one_hot(aatype, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        pseudo_beta, pseudo_beta_mask = template_batch.pseudo_beta()
        template_pair_mask = pseudo_beta_mask.unsqueeze(-1) * pseudo_beta_mask.unsqueeze(-2)
        distogram = _distogram(
            pseudo_beta,
            min_bin=self.min_bin,
            max_bin=self.max_bin,
            num_bins=self.num_distogram_bins,
        ).to(dtype=template_batch.dtype)

        residue_i = aatype_one_hot.unsqueeze(-2).expand(*template_pair_mask.shape, self.vocab_size)
        residue_j = aatype_one_hot.unsqueeze(-3).expand(*template_pair_mask.shape, self.vocab_size)
        unit_vector, backbone_pair_mask = template_batch.backbone_unit_vector()
        if not self.use_template_unit_vector:
            unit_vector = torch.zeros_like(unit_vector)
        features = torch.cat(
            [
                distogram,
                template_pair_mask.unsqueeze(-1),
                residue_i,
                residue_j,
                unit_vector,
                backbone_pair_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features * backbone_pair_mask.unsqueeze(-1)
        return features, backbone_pair_mask


class AlphaFold2TemplatePairStack(nn.Module):
    r"""
    TemplatePairStack with official template block order.
    """

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
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AlphaFold2TemplatePairStackBlock(
                    pair_size=pair_size,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    transition_expansion=transition_expansion,
                    dropout=dropout,
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        for block in self.blocks:
            pair = block(pair, pair_mask=pair_mask)
        return pair


class AlphaFold2TemplatePairStackBlock(nn.Module):
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
    ) -> None:
        super().__init__()
        self.pair_stack = AlphaFold2TemplatePairStackLayer(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
        )

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        return self.pair_stack(pair, pair_mask=pair_mask)


class AlphaFold2TemplatePairStackLayer(nn.Module):
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
    ) -> None:
        super().__init__()
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            orientation="per_row",
            operator_backend=operator_backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            orientation="per_column",
            operator_backend=operator_backend,
        )
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            equation="outgoing",
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            equation="incoming",
        )
        self.feed_forward = EvoformerTransition(pair_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_ending_node(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
        return pair + self.feed_forward(pair, pair_mask)


class AlphaFold2TemplateMsaEmbedding(nn.Module):
    r"""
    Embed template torsion features as extra MSA rows.
    """

    def __init__(self, *, vocab_size: int, msa_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        template_feature_size = vocab_size + 14 + 14 + 7
        self.template_single_embedding = nn.Linear(template_feature_size, msa_size)
        self.template_projection = nn.Linear(msa_size, msa_size)

    def forward(self, batch: Any, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor] | None:
        template_batch = _template_batch(batch, dtype=dtype, device=device)
        if template_batch is None:
            return None

        features, torsion_mask = self._features(template_batch)
        activations = F.relu(self.template_single_embedding(features))
        activations = self.template_projection(activations)
        msa_mask = torsion_mask[..., 2] * template_batch.template_mask.unsqueeze(-1).to(dtype=torsion_mask.dtype)
        return activations, msa_mask

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        aatype = template_batch.template_aatype.long().clamp(min=0, max=self.vocab_size - 1)
        aatype_one_hot = F.one_hot(aatype, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        torsion = template_batch.template_torsion_angles_sin_cos
        alt_torsion = template_batch.template_alt_torsion_angles_sin_cos
        torsion_mask = template_batch.template_torsion_angles_mask
        if torsion is None or alt_torsion is None or torsion_mask is None:
            shape = (*aatype.shape, 7)
            torsion = aatype_one_hot.new_zeros(*shape, 2)
            alt_torsion = aatype_one_hot.new_zeros(*shape, 2)
            torsion_mask = template_batch.backbone_torsion_mask()
        features = torch.cat(
            [
                aatype_one_hot,
                torsion.reshape(*aatype.shape, 14).to(dtype=aatype_one_hot.dtype),
                alt_torsion.reshape(*aatype.shape, 14).to(dtype=aatype_one_hot.dtype),
                torsion_mask.to(dtype=aatype_one_hot.dtype),
            ],
            dim=-1,
        )
        return features, torsion_mask.to(dtype=aatype_one_hot.dtype)


class AlphaFold2TemplateBatch:
    def __init__(
        self,
        *,
        template_aatype: Tensor,
        template_mask: Tensor,
        template_all_atom_positions: Tensor,
        template_all_atom_masks: Tensor,
        template_pseudo_beta: Tensor | None,
        template_pseudo_beta_mask: Tensor | None,
        template_torsion_angles_sin_cos: Tensor | None,
        template_alt_torsion_angles_sin_cos: Tensor | None,
        template_torsion_angles_mask: Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        self.template_aatype = template_aatype
        self.template_mask = template_mask
        self.template_all_atom_positions = template_all_atom_positions
        self.template_all_atom_masks = template_all_atom_masks
        self.template_pseudo_beta = template_pseudo_beta
        self.template_pseudo_beta_mask = template_pseudo_beta_mask
        self.template_torsion_angles_sin_cos = template_torsion_angles_sin_cos
        self.template_alt_torsion_angles_sin_cos = template_alt_torsion_angles_sin_cos
        self.template_torsion_angles_mask = template_torsion_angles_mask
        self.dtype = dtype

    def pseudo_beta(self) -> tuple[Tensor, Tensor]:
        if self.template_pseudo_beta is not None and self.template_pseudo_beta_mask is not None:
            return self.template_pseudo_beta.to(dtype=self.dtype), self.template_pseudo_beta_mask.to(dtype=self.dtype)
        ca = residue_constants.atom_order["CA"]
        cb = residue_constants.atom_order["CB"]
        is_gly = self.template_aatype == _GLY_INDEX
        atom_index = torch.where(
            is_gly,
            torch.full_like(self.template_aatype, ca),
            torch.full_like(self.template_aatype, cb),
        )
        gather_index = atom_index[..., None, None].expand(*atom_index.shape, 1, 3)
        pseudo_beta = self.template_all_atom_positions.gather(-2, gather_index).squeeze(-2)
        mask_index = atom_index[..., None].expand(*atom_index.shape, 1)
        pseudo_beta_mask = self.template_all_atom_masks.gather(-1, mask_index).squeeze(-1)
        return pseudo_beta.to(dtype=self.dtype), pseudo_beta_mask.to(dtype=self.dtype)

    def backbone_unit_vector(self) -> tuple[Tensor, Tensor]:
        n = residue_constants.atom_order["N"]
        ca = residue_constants.atom_order["CA"]
        c = residue_constants.atom_order["C"]
        atom_pos = self.template_all_atom_positions.to(dtype=self.dtype)
        atom_mask = self.template_all_atom_masks.to(dtype=self.dtype)
        n_pos = atom_pos[..., n, :]
        ca_pos = atom_pos[..., ca, :]
        c_pos = atom_pos[..., c, :]
        backbone_mask = atom_mask[..., n] * atom_mask[..., ca] * atom_mask[..., c]

        x_axis = F.normalize(c_pos - ca_pos, dim=-1)
        n_axis = F.normalize(n_pos - ca_pos, dim=-1)
        z_axis = F.normalize(torch.cross(x_axis, n_axis, dim=-1), dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        pseudo_beta, _ = self.pseudo_beta()
        vector = pseudo_beta.unsqueeze(-3) - ca_pos.unsqueeze(-2)
        unit_vector = torch.matmul(rotation.unsqueeze(-3).transpose(-1, -2), vector.unsqueeze(-1)).squeeze(-1)
        unit_vector = F.normalize(unit_vector, dim=-1)
        pair_mask = backbone_mask.unsqueeze(-1) * backbone_mask.unsqueeze(-2)
        unit_vector = torch.nan_to_num(unit_vector) * pair_mask.unsqueeze(-1)
        return unit_vector, pair_mask.to(dtype=self.dtype)

    def backbone_torsion_mask(self) -> Tensor:
        n = residue_constants.atom_order["N"]
        ca = residue_constants.atom_order["CA"]
        c = residue_constants.atom_order["C"]
        atom_mask = self.template_all_atom_masks.to(dtype=self.dtype)
        backbone_mask = atom_mask[..., n] * atom_mask[..., ca] * atom_mask[..., c]
        torsion_mask = atom_mask.new_zeros(*self.template_aatype.shape, 7)
        torsion_mask[..., 2] = backbone_mask
        return torsion_mask


def _template_batch(
    batch: Any,
    reference: Tensor | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> AlphaFold2TemplateBatch | None:
    if batch.template_aatype is None:
        return None
    if batch.template_all_atom_positions is None or batch.template_all_atom_masks is None:
        raise ValueError("AF2 template embedding requires template_all_atom_positions and template_all_atom_masks")
    if reference is not None:
        dtype = reference.dtype
        device = reference.device
    if dtype is None or device is None:
        raise ValueError("template batch normalization requires dtype and device")

    template_aatype = _ensure_batched_templates(batch.template_aatype, 2, device=device)
    template_all_atom_positions = _ensure_batched_templates(
        batch.template_all_atom_positions,
        4,
        device=device,
        dtype=dtype,
    )
    template_all_atom_masks = _ensure_batched_templates(batch.template_all_atom_masks, 3, device=device, dtype=dtype)
    if batch.template_mask is None:
        template_mask = (template_all_atom_masks.sum(dim=(-1, -2)) > 0).to(dtype=dtype)
    else:
        template_mask = _ensure_batched_templates(batch.template_mask, 1, device=device, dtype=dtype)
    return AlphaFold2TemplateBatch(
        template_aatype=template_aatype,
        template_mask=template_mask,
        template_all_atom_positions=template_all_atom_positions,
        template_all_atom_masks=template_all_atom_masks,
        template_pseudo_beta=_optional_batched(batch.template_pseudo_beta, 3, device=device, dtype=dtype),
        template_pseudo_beta_mask=_optional_batched(batch.template_pseudo_beta_mask, 2, device=device, dtype=dtype),
        template_torsion_angles_sin_cos=_optional_batched(
            batch.template_torsion_angles_sin_cos,
            4,
            device=device,
            dtype=dtype,
        ),
        template_alt_torsion_angles_sin_cos=_optional_batched(
            batch.template_alt_torsion_angles_sin_cos,
            4,
            device=device,
            dtype=dtype,
        ),
        template_torsion_angles_mask=_optional_batched(
            batch.template_torsion_angles_mask,
            3,
            device=device,
            dtype=dtype,
        ),
        dtype=dtype,
    )


def _optional_batched(
    tensor: Tensor | None,
    unbatched_ndim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    if tensor is None:
        return None
    return _ensure_batched_templates(tensor, unbatched_ndim, device=device, dtype=dtype)


def _ensure_batched_templates(
    tensor: Tensor,
    unbatched_ndim: int,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor:
    tensor = tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)
    if tensor.dim() == unbatched_ndim:
        return tensor.unsqueeze(0)
    if tensor.dim() == unbatched_ndim + 1:
        return tensor
    raise ValueError(f"expected template tensor with {unbatched_ndim} or {unbatched_ndim + 1} dims, got {tensor.dim()}")


def _distogram(positions: Tensor, *, min_bin: float, max_bin: float, num_bins: int) -> Tensor:
    distances = torch.cdist(positions.float(), positions.float()).to(dtype=positions.dtype)
    boundaries = torch.linspace(min_bin, max_bin, steps=num_bins - 1, device=positions.device, dtype=positions.dtype)
    bins = torch.bucketize(distances, boundaries)
    return F.one_hot(bins, num_classes=num_bins)
