r"""
Template embedding modules.

Embed structural-template pair features (distograms, unit vectors, residue types, validity masks)
into the trunk pair representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..backend import LayerNorm
from ..trunk.msa import TrianglePairUpdateBlock


@dataclass
class TemplateFeatures:
    r"""
    Typed template features consumed by ``TemplateEmbedder``.

    Bundles the per-template geometric and residue features into a single typed container so they can
    be moved between devices/dtypes together and passed through the embedder as one argument. All
    tensors carry a leading ``num_templates`` axis (``[..., num_templates, ...]``).

    Args:
        distogram: ``[..., num_templates, N_token, N_token, num_distogram_bins]`` template pair distogram
            features.
        pseudo_beta_mask: ``[..., num_templates, N_token, N_token]`` template pseudo-beta validity mask.
        residue_type: Integer template residue type ids ``[..., num_templates, N_token]``.
        unit_vector: ``[..., num_templates, N_token, N_token, 3]`` template pair unit-vector features.
        backbone_frame_mask: ``[..., num_templates, N_token, N_token]`` template backbone-frame validity mask.
        template_mask: Optional ``[..., num_templates]`` per-template validity mask.
    """

    distogram: Tensor
    pseudo_beta_mask: Tensor
    residue_type: Tensor
    unit_vector: Tensor
    backbone_frame_mask: Tensor
    template_mask: Tensor | None = None


class TemplateEmbedder(nn.Module):
    r"""
    Embed template-derived pair features into pair representations.

    Consumes pre-packed [`TemplateFeatures`][ofoldx.modules.embedding.template.TemplateFeatures],
    projects each template's geometric and residue features into a hidden pair channel (added to a
    projection of the normalized trunk pair), refines it through ``num_blocks`` pair-only triangle
    update blocks, and returns the per-template mean projected back to ``pair_size``. When
    ``num_blocks == 0`` or no templates are present, a zero update is returned.

    Args:
        pair_size: Width of the trunk pair representation this update is added into.
        hidden_size: Width of the internal template pair channel.
        num_blocks: Number of pair-only triangle update blocks; ``0`` disables the embedder.
        vocab_size: Residue-type vocabulary size (one block per side of the pair).
        num_distogram_bins: Number of distogram bins in the packed template features.
        triangle_attention_hidden_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Transition expansion factor inside each triangle update block.
        dropout: Dropout probability inside each triangle update block.
        ending_node_transpose_bias: Whether ending-node attention derives its pair bias from the
            pre-transpose pair rather than the oriented attention input.
        operator_backend: Operator backend selector forwarded to the triangle update blocks.

    Examples:
        >>> import torch
        >>> embedder = TemplateEmbedder(
        ...     pair_size=8, hidden_size=4, num_blocks=0, vocab_size=2, num_distogram_bins=3
        ... )
        >>> pair = torch.zeros(1, 5, 5, 8)
        >>> templates = TemplateFeatures(
        ...     distogram=torch.zeros(1, 0, 5, 5, 3),
        ...     pseudo_beta_mask=torch.zeros(1, 0, 5, 5),
        ...     residue_type=torch.zeros(1, 0, 5, dtype=torch.long),
        ...     unit_vector=torch.zeros(1, 0, 5, 5, 3),
        ...     backbone_frame_mask=torch.zeros(1, 0, 5, 5),
        ... )
        >>> embedder(pair, templates).shape  # num_blocks == 0 -> zero update
        torch.Size([1, 5, 5, 8])
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
        dropout: float = 0.25,
        ending_node_transpose_bias: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        template_feature_size = num_distogram_bins + 1 + 2 * vocab_size + 3 + 1
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.pair_norm = LayerNorm(pair_size, operator_backend=operator_backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.template_projection = nn.Linear(template_feature_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                TrianglePairUpdateBlock(
                    pair_size=hidden_size,
                    triangle_multiplication_hidden_size=hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    expansion=expansion,
                    dropout=dropout,
                    ending_node_transpose_bias=ending_node_transpose_bias,
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            ]
        )
        self.hidden_norm = LayerNorm(hidden_size, operator_backend=operator_backend)
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

        Each template is embedded and refined independently, optionally masked by ``template_mask``,
        then averaged over templates and projected back to ``pair_size``. Returns a zero update when
        the embedder is disabled (``num_blocks == 0``) or no templates are present.

        Args:
            pair: ``[..., N_token, N_token, pair_size]`` trunk pair representation.
            templates: Pre-packed template features.
            pair_mask: Optional ``[..., N_token, N_token]`` pair validity mask forwarded to the
                triangle update blocks.

        Returns:
            ``[..., N_token, N_token, pair_size]`` template pair update to add into ``pair``.

        Examples:
            >>> import torch
            >>> embedder = TemplateEmbedder(
            ...     pair_size=8, hidden_size=4, num_blocks=0, vocab_size=2, num_distogram_bins=3
            ... )
            >>> pair = torch.zeros(1, 4, 4, 8)
            >>> templates = TemplateFeatures(
            ...     distogram=torch.zeros(1, 0, 4, 4, 3),
            ...     pseudo_beta_mask=torch.zeros(1, 0, 4, 4),
            ...     residue_type=torch.zeros(1, 0, 4, dtype=torch.long),
            ...     unit_vector=torch.zeros(1, 0, 4, 4, 3),
            ...     backbone_frame_mask=torch.zeros(1, 0, 4, 4),
            ... )
            >>> update = embedder(pair, templates)
            >>> update.shape, bool(update.abs().sum() == 0)
            (torch.Size([1, 4, 4, 8]), True)
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
        Embed and refine a single template into the hidden pair channel.

        Projects the normalized trunk pair, adds the geometric and per-side residue projections
        (gathered from the shared ``template_projection`` weight to avoid expanding the residue
        one-hots over the pair), runs the triangle update blocks, and normalizes the result.

        Args:
            pair: ``[..., N_token, N_token, pair_size]`` normalized trunk pair representation.
            templates: Pre-packed template features.
            template_index: Index of the template to embed along the template axis.
            pair_mask: Optional ``[..., N_token, N_token]`` pair validity mask for the blocks.

        Returns:
            ``[..., N_token, N_token, hidden_size]`` refined per-template pair representation.
        """
        geom, residue_one_hot = self._pack_features(templates, template_index)
        template_pair = self.pair_projection(pair)
        # Project the per-token residue types and broadcast over the pair, instead of materialising the
        # (..., N, N, 2*K) expanded one-hots. F.linear(one_hot, W_slice) == one_hot @ W_slice columns (exact).
        # The packed feature order is [distogram, pseudo_beta_mask, residue_i(K), residue_j(K), unit_vector(3),
        # frame_mask(1)], so the geometric columns are the leading (distogram+mask) plus the trailing 4.
        weight = self.template_projection.weight
        vocab_size = self.vocab_size
        leading = geom.shape[-1] - 4
        geom_weight = torch.cat([weight[:, :leading], weight[:, leading + 2 * vocab_size :]], dim=-1)
        template_pair = (
            template_pair
            + F.linear(geom, geom_weight)
            + F.linear(residue_one_hot, weight[:, leading : leading + vocab_size]).unsqueeze(-2)
            + F.linear(residue_one_hot, weight[:, leading + vocab_size : leading + 2 * vocab_size]).unsqueeze(-3)
        )
        for block in self.blocks:
            template_pair = block(template_pair, pair_mask=pair_mask)
        return self.hidden_norm(template_pair)

    def _pack_features(self, templates: TemplateFeatures, template_index: int) -> tuple[Tensor, Tensor]:
        r"""
        Return ``(geometric_pair_features, residue_one_hot)`` for a single template.

        The residue one-hot is kept per-token (``[..., N_token, vocab_size]``) so the caller can
        project-and-broadcast it rather than expanding it over the pair.

        Args:
            templates: Pre-packed template features.
            template_index: Index of the template to pack along the template axis.

        Returns:
            ``(geom, residue_one_hot)`` where ``geom`` is ``[..., N_token, N_token, num_distogram_bins + 5]``
            geometric pair features and ``residue_one_hot`` is ``[..., N_token, vocab_size]``.
        """
        expected_residue_shape = templates.distogram.shape[:-2]
        if templates.residue_type.shape != expected_residue_shape:
            raise ValueError(
                f"residue_type must have shape {expected_residue_shape}; got {templates.residue_type.shape}"
            )
        residue_type = templates.residue_type[..., template_index, :]
        residue_one_hot = F.one_hot(residue_type, num_classes=self.vocab_size).to(dtype=templates.distogram.dtype)

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
