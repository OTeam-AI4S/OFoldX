r"""
Token encoder for coordinate denoising.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import torch
from torch import Tensor, nn

from ..backend import LayerNorm, pair_norm_projection
from ..transformer import DenoisingTransformerEncoder, DenoisingTransformerEncoderBlock


def _run_pair_biased_block(
    hidden: Tensor,
    conditioning: Tensor,
    pair: Tensor,
    pair_mask: Tensor | None,
    *,
    block: DenoisingTransformerEncoderBlock,
    norm: LayerNorm,
    projection: nn.Linear,
) -> Tensor:
    # The block accepts head-last bias and moves it back to head-major for attention.
    # Keep this as a view so OOps can write that final head-major layout directly.
    bias = pair_norm_projection(
        pair,
        norm.weight,
        norm.bias,
        projection.weight,
        norm.eps,
        output_layout="head_major",
        backend=norm.backend,
    )
    missing_prefix_dims = len(hidden.shape[:-2]) - len(pair.shape[:-3])
    bias = bias.reshape(
        *bias.shape[:-3],
        *((1,) * missing_prefix_dims),
        *bias.shape[-3:],
    ).movedim(-3, -1)
    return block(hidden, conditioning, bias, pair_mask, to_keys=None)


class DenoisingTokenEncoder(nn.Module):
    r"""
    Token encoder for coordinate denoising, conditioned on a per-block pair-derived attention bias.

    Projects the pair representation to per-block attention-bias logits (``pair_norms`` / ``bias_projections``)
    and delegates to the shared
    [`DenoisingTransformerEncoder`][ofoldx.modules.transformer.DenoisingTransformerEncoder]
    configured with a single adaptive norm and double-product transition.

    Args:
        num_blocks: Number of token-encoder blocks; each contributes one Transformer-encoder block. ``0`` makes the
            module the identity on the tokens.
        token_hidden_size: Width of the per-token channel carried through the encoder.
        conditioning_size: Width of the conditioning signal driving the adaptive norm.
        pair_size: Width of the pair representation projected to the attention bias.
        num_attention_heads: Number of attention heads per transformer block.
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        num_blocks: int = 24,
        token_hidden_size: int = 768,
        conditioning_size: int = 384,
        pair_size: int = 128,
        num_attention_heads: int = 16,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norms = nn.ModuleList(LayerNorm(pair_size, bias=False, backend=backend) for _ in range(num_blocks))
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )
        self.transformer = DenoisingTransformerEncoder(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            hidden_size=token_hidden_size,
            conditioning_size=conditioning_size,
            backend=backend,
        )

    def forward(
        self,
        token_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Norm and project the pair representation lazily into each block's per-head bias, then run the transformer.

        Each block ``i`` maps ``pair`` through its own ``pair_norms[i]`` /
        ``bias_projections[i]`` immediately before that block executes. Under
        gradient checkpointing, the pair projection is recomputed together with
        its consuming block instead of retaining every block's projected bias.
        With zero blocks the module is the identity on ``token_hidden``.

        Args:
            token_hidden: Per-token features `[..., N_token, token_hidden_size]`.
            conditioning: Conditioning signal `[..., N_token, conditioning_size]` for the adaptive norm.
            pair: Pair representation `[..., N_token, N_token, pair_size]` supplying the attention bias.
            pair_mask: Optional pair mask `[..., N_token, N_token]`; ``False`` positions are excluded.

        Returns:
            Denoised token features `[..., N_token, token_hidden_size]`.
        """

        if not self.pair_norms:  # zero blocks -> identity (matches an empty transformer stack)
            return token_hidden
        checkpoint = (
            self.transformer._gradient_checkpointing_func
            if self.transformer.gradient_checkpointing and self.training and torch.is_grad_enabled()
            else None
        )
        hidden = token_hidden
        for block, norm, projection in zip(
            self.transformer.blocks,
            self.pair_norms,
            self.bias_projections,
            strict=True,
        ):
            run_block = partial(
                _run_pair_biased_block,
                block=cast(DenoisingTransformerEncoderBlock, block),
                norm=cast(LayerNorm, norm),
                projection=cast(nn.Linear, projection),
            )
            if checkpoint is not None:
                hidden = checkpoint(run_block, hidden, conditioning, pair, pair_mask)
            else:
                hidden = run_block(hidden, conditioning, pair, pair_mask)
        return hidden
