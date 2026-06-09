r"""
Template embedding modules.

Embed structural-template pair features (distograms, unit vectors, residue types, validity masks)
into the trunk pair representation. Two variants are provided: an embedder over pre-packed template
features and a V2 embedder that featurizes raw coordinates and frames on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from oops import LayerNorm
from torch import Tensor, nn

from ..trunk.msa import TrianglePairUpdateBlock


@dataclass(slots=True)
class TemplateFeatures:
    r"""
    Typed template features consumed by ``TemplateEmbedder``.

    Bundles the per-template geometric and residue features into a single typed container so they can
    be moved between devices/dtypes together and passed through the embedder as one argument. All
    tensors carry a leading ``num_templates`` axis (``[..., num_templates, ...]``).

    Attributes:
        distogram: ``[..., num_templates, N_token, N_token, num_distogram_bins]`` template pair distogram
            features.
        pseudo_beta_mask: ``[..., num_templates, N_token, N_token]`` template pseudo-beta validity mask.
        residue_type: ``[..., num_templates, N_token]`` (or one-hot ``[..., num_templates, N_token, vocab_size]``)
            template residue type ids.
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

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> TemplateFeatures:
        r"""
        Return a copy with tensors moved to the requested device and dtype.

        Floating-point features (distogram, masks, unit vectors) are cast to ``dtype``; the integer
        ``residue_type`` and ``template_mask`` are moved but never cast, so their integer semantics are
        preserved.

        Args:
            device: Optional destination device for all tensors.
            dtype: Optional dtype for the floating-point features only.
            non_blocking: Passed through to `torch.Tensor.to` for async host-to-device copies.
            copy: Force a copy even when the move would otherwise be a no-op.

        Returns:
            A new ``TemplateFeatures`` with the same fields moved/cast as requested.

        Examples:
            >>> import torch
            >>> templates = TemplateFeatures(
            ...     distogram=torch.zeros(1, 1, 3, 3, 4),
            ...     pseudo_beta_mask=torch.zeros(1, 1, 3, 3),
            ...     residue_type=torch.zeros(1, 1, 3, dtype=torch.long),
            ...     unit_vector=torch.zeros(1, 1, 3, 3, 3),
            ...     backbone_frame_mask=torch.zeros(1, 1, 3, 3),
            ... )
            >>> moved = templates.to(dtype=torch.float64)
            >>> moved.distogram.dtype, moved.residue_type.dtype
            (torch.float64, torch.int64)
        """

        return type(self)(
            distogram=self.distogram.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            pseudo_beta_mask=self.pseudo_beta_mask.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            residue_type=self.residue_type.to(device=device, non_blocking=non_blocking, copy=copy),
            unit_vector=self.unit_vector.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            backbone_frame_mask=self.backbone_frame_mask.to(
                device=device, dtype=dtype, non_blocking=non_blocking, copy=copy
            ),
            template_mask=(
                None
                if self.template_mask is None
                else self.template_mask.to(device=device, non_blocking=non_blocking, copy=copy)
            ),
        )


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
        expansion: FFN expansion factor inside each triangle update block.
        dropout: Dropout probability inside each triangle update block.
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
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        template_feature_size = num_distogram_bins + 1 + 2 * vocab_size + 3 + 1
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.norm_pair = LayerNorm(pair_size)
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
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            ]
        )
        self.hidden_norm = LayerNorm(hidden_size)
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

        normalized_pair = self.norm_pair(pair)
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
        project-and-broadcast it rather than expanding it over the pair; ``residue_type`` that is
        already one-hot is passed through, otherwise it is one-hot encoded.

        Args:
            templates: Pre-packed template features.
            template_index: Index of the template to pack along the template axis.

        Returns:
            ``(geom, residue_one_hot)`` where ``geom`` is ``[..., N_token, N_token, num_distogram_bins + 5]``
            geometric pair features and ``residue_one_hot`` is ``[..., N_token, vocab_size]``.
        """
        residue_type = templates.residue_type[..., template_index, :]
        if residue_type.shape[-1] == self.vocab_size:
            residue_one_hot = residue_type.to(dtype=templates.distogram.dtype)
        else:
            residue_one_hot = F.one_hot(residue_type.long(), num_classes=self.vocab_size).to(
                dtype=templates.distogram.dtype
            )

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


class TemplateV2Embedder(nn.Module):
    r"""
    Template embedder with on-the-fly distogram / unit-vector featurization.

    Unlike [`TemplateEmbedder`][ofoldx.modules.embedding.template.TemplateEmbedder] (which consumes pre-packed
    features), this computes the template distogram and per-frame unit vectors inside ``forward`` from raw CA/CB
    coords and frames, masks pairs by ``visibility_ids`` (same-chain), folds templates into the batch through
    a shared pair-only pairformer with an outer residual, and aggregates by a template-count-weighted mean.

    Feature order fed to ``template_projection`` is load-bearing: ``[distogram, cb_mask, unit_vector,
    frame_mask, residue_type_i, residue_type_j]`` (= 2*vocab_size + num_bins + 5 channels).

    Args:
        pair_size: Width of the trunk pair representation this update is added into.
        hidden_size: Width of the internal template pair channel.
        num_blocks: Number of pair-only triangle update blocks in the shared pairformer.
        vocab_size: Residue-type vocabulary size (one block per side of the pair).
        num_bins: Number of distogram bins computed from CB distances.
        min_distance: Lower edge of the distogram binning range, in angstroms.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        expansion: FFN expansion factor inside each triangle update block.
        triangle_attention_hidden_size: Per-head dimension of the triangle attention.
        num_triangle_attention_heads: Number of triangle attention heads.
        dropout: Dropout probability inside each triangle update block.
        operator_backend: Operator backend selector forwarded to the triangle update blocks.

    Examples:
        >>> from ofoldx.modules.embedding.template import TemplateV2Embedder  # doctest: +SKIP
        >>> embedder = TemplateV2Embedder(pair_size=128, hidden_size=64).cuda()  # doctest: +SKIP
        >>> update = embedder(  # doctest: +SKIP
        ...     pair,
        ...     residue_type=residue_type,
        ...     frame_rotation=frame_rotation,
        ...     frame_translation=frame_translation,
        ...     frame_mask=frame_mask,
        ...     cb_coords=cb_coords,
        ...     ca_coords=ca_coords,
        ...     cb_mask=cb_mask,
        ...     visibility_ids=visibility_ids,
        ...     template_mask=template_mask,
        ... )
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = 33,
        num_bins: int = 38,
        min_distance: float = 3.25,
        max_distance: float = 50.75,
        expansion: int = 4,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        template_feature_size = 2 * vocab_size + num_bins + 5
        self.norm_pair = LayerNorm(pair_size)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.template_projection = nn.Linear(template_feature_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            TrianglePairUpdateBlock(
                pair_size=hidden_size,
                triangle_multiplication_hidden_size=hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                dropout=dropout,
                operator_backend=operator_backend,
            )
            for _ in range(num_blocks)
        )
        self.hidden_norm = LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def _featurize(
        self,
        residue_type: Tensor,
        frame_rotation: Tensor,
        frame_translation: Tensor,
        frame_mask: Tensor,
        cb_coords: Tensor,
        ca_coords: Tensor,
        cb_mask: Tensor,
        visibility_ids: Tensor,
    ) -> Tensor:
        r"""
        Build the packed per-template pair feature tensor from raw coordinates and frames.

        Computes the CB-distance distogram, the per-frame CA unit vectors, masks pairs to same-chain
        via ``visibility_ids``, and concatenates the geometry with the broadcast per-side residue
        types in the load-bearing channel order.

        Args:
            residue_type: ``[batch, num_templates, N_token, vocab_size]`` per-token residue features.
            frame_rotation: ``[batch, num_templates, N_token, 3, 3]`` backbone frame rotations.
            frame_translation: ``[batch, num_templates, N_token, 3]`` backbone frame translations.
            frame_mask: ``[batch, num_templates, N_token]`` backbone frame validity mask.
            cb_coords: ``[batch, num_templates, N_token, 3]`` CB (or pseudo-CB) coordinates.
            ca_coords: ``[batch, num_templates, N_token, 3]`` CA coordinates.
            cb_mask: ``[batch, num_templates, N_token]`` CB validity mask.
            visibility_ids: ``[batch, num_templates, N_token]`` chain/visibility ids; only same-id
                pairs keep their geometry.

        Returns:
            ``[batch, num_templates, N_token, N_token, 2 * vocab_size + num_bins + 5]`` packed features.
        """
        num_tokens = residue_type.shape[2]
        pair_cb_mask = (cb_mask[..., :, None] * cb_mask[..., None, :]).unsqueeze(-1)
        pair_frame_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        same_chain = (visibility_ids[..., :, None] == visibility_ids[..., None, :]).to(residue_type.dtype)

        cb_distances = torch.cdist(cb_coords, cb_coords)
        boundaries = torch.linspace(self.min_distance, self.max_distance, self.num_bins - 1, device=cb_distances.device)
        distogram = (cb_distances[..., None] > boundaries).sum(dim=-1).long()
        distogram = F.one_hot(distogram, num_classes=self.num_bins).to(residue_type.dtype)

        # Per-frame unit vector of CA_j in frame_i (replicated faithfully, incl. the per-component norm).
        rotation = frame_rotation.unsqueeze(2).transpose(-1, -2)
        translation = frame_translation.unsqueeze(2).unsqueeze(-1)
        targets = ca_coords.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(rotation, targets - translation)
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector)).squeeze(-1)

        geometry = torch.cat([distogram, pair_cb_mask, unit_vector, pair_frame_mask], dim=-1)
        geometry = geometry * same_chain.unsqueeze(-1)
        residue_i = residue_type[:, :, :, None].expand(-1, -1, -1, num_tokens, -1)
        residue_j = residue_type[:, :, None, :].expand(-1, -1, num_tokens, -1, -1)
        return torch.cat([geometry, residue_i, residue_j], dim=-1)

    def forward(
        self,
        pair: Tensor,
        *,
        residue_type: Tensor,
        frame_rotation: Tensor,
        frame_translation: Tensor,
        frame_mask: Tensor,
        cb_coords: Tensor,
        ca_coords: Tensor,
        cb_mask: Tensor,
        visibility_ids: Tensor,
        template_mask: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return the template pair update ``u`` (added into the trunk pair representation).

        Templates are folded into the batch dim (``reshape(batch * num_templates, ...)``) and run through the
        template pairformer in one batched pass, then aggregated by a template-count-weighted mean over the
        per-template mask.

        Args:
            pair: ``[batch, N_token, N_token, pair_size]`` trunk pair representation.
            residue_type: ``[batch, num_templates, N_token, vocab_size]`` per-token residue features.
            frame_rotation: ``[batch, num_templates, N_token, 3, 3]`` backbone frame rotations.
            frame_translation: ``[batch, num_templates, N_token, 3]`` backbone frame translations.
            frame_mask: ``[batch, num_templates, N_token]`` backbone frame validity mask.
            cb_coords: ``[batch, num_templates, N_token, 3]`` CB (or pseudo-CB) coordinates.
            ca_coords: ``[batch, num_templates, N_token, 3]`` CA coordinates.
            cb_mask: ``[batch, num_templates, N_token]`` CB validity mask.
            visibility_ids: ``[batch, num_templates, N_token]`` chain/visibility ids for the same-chain gate.
            template_mask: ``[batch, num_templates, N_token]`` per-token template validity mask; a template
                counts as present when any token is valid.
            pair_mask: Optional ``[batch, N_token, N_token]`` pair validity mask forwarded to the blocks.

        Returns:
            ``[batch, N_token, N_token, pair_size]`` template pair update to add into ``pair``.

        Examples:
            >>> embedder = TemplateV2Embedder().cuda()  # doctest: +SKIP
            >>> update = embedder(pair, residue_type=residue_type, frame_rotation=frame_rotation,
            ...                   frame_translation=frame_translation, frame_mask=frame_mask,
            ...                   cb_coords=cb_coords, ca_coords=ca_coords, cb_mask=cb_mask,
            ...                   visibility_ids=visibility_ids, template_mask=template_mask)  # doctest: +SKIP
        """

        batch, num_templates = residue_type.shape[:2]
        present = template_mask.any(dim=2).to(pair.dtype)
        template_count = present.sum(dim=1).clamp(min=1)

        features = self._featurize(
            residue_type,
            frame_rotation,
            frame_translation,
            frame_mask,
            cb_coords,
            ca_coords,
            cb_mask,
            visibility_ids,
        )
        projected = self.template_projection(features)
        template_pair = self.pair_projection(self.norm_pair(pair[:, None])) + projected
        template_pair = template_pair.reshape(batch * num_templates, *template_pair.shape[2:])

        if pair_mask is not None:
            pair_mask = pair_mask[:, None].expand(-1, num_templates, -1, -1)
            pair_mask = pair_mask.reshape(batch * num_templates, *pair_mask.shape[2:])

        stacked = template_pair
        for block in self.blocks:
            stacked = block(stacked, pair_mask=pair_mask)
        template_pair = self.hidden_norm(template_pair + stacked)
        template_pair = template_pair.reshape(batch, num_templates, *template_pair.shape[1:])

        weight = present[:, :, None, None, None]
        update = (template_pair * weight).sum(dim=1) / template_count[:, None, None, None].to(pair.dtype)
        return self.output_projection(self.activation(update))
