r"""
Token transformer for coordinate denoising.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..backend import LayerNorm
from ..transformer import DenoisingTransformer


class DenoisingTokenEncoder(nn.Module):
    r"""
    Token transformer for coordinate denoising, conditioned on a per-block pair-derived attention bias.

    Projects the pair representation to per-block attention-bias logits (``pair_norms`` / ``bias_projections``)
    and delegates to the shared [`ConditionedTransformer`][ofoldx.modules.transformer.ConditionedTransformer]
    configured with a single adaptive norm and double-product transition. This mirrors
    [`AtomSelfAttention`][ofoldx.modules.atom.encoders.AtomSelfAttention] at the token level.

    Args:
        num_blocks: Number of token-encoder blocks; each contributes one transformer block. ``0`` makes the
            module the identity on the tokens.
        token_hidden_size: Width of the per-token channel carried through the transformer.
        conditioning_size: Width of the conditioning signal driving the adaptive norm.
        pair_size: Width of the pair representation projected to the attention bias.
        num_attention_heads: Number of attention heads per transformer block.
        operator_backend: Operator provider; see [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        num_blocks: int = 24,
        token_hidden_size: int = 768,
        conditioning_size: int = 384,
        pair_size: int = 128,
        num_attention_heads: int = 16,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norms = nn.ModuleList(
            LayerNorm(pair_size, bias=False, operator_backend=operator_backend) for _ in range(num_blocks)
        )
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )
        self.transformer = DenoisingTransformer(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            hidden_size=token_hidden_size,
            conditioning_size=conditioning_size,
            operator_backend=operator_backend,
        )

    def forward(
        self,
        token_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Norm and project the pair representation into a per-block, per-head bias, then run the transformer.

        Each block ``i`` maps ``pair`` through its own ``pair_norms[i]`` / ``bias_projections[i]`` to
        ``num_attention_heads`` bias logits; these are concatenated over all ``num_blocks`` blocks into
        `[..., N_token, N_token, num_blocks * num_attention_heads]` and sliced one block at a time inside the
        transformer. With zero blocks the module is the identity on ``token_hidden``.

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
        bias = torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections)], dim=-1
        )
        return self.transformer(token_hidden, conditioning, bias, pair_mask, to_keys=None)
