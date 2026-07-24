r"""
Linear projection of factorized pair features.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..backend import factorized_pair_linear, resolve_config_backend


class FactorizedPairLinear(nn.Linear):
    r"""
    Project pair features and one token feature track used on both pair axes.

    Parameters and initialization are identical to an [`nn.Linear`][torch.nn.Linear]
    over packed features ordered as
    ``[pair_prefix, token_i, token_j, pair_suffix]``. The forward input stays
    factorized, so model code does not own projection scheduling or pairwise token
    feature materialization.

    Args:
        pair_size: Width of the pairwise feature tensor.
        token_size: Width of the per-token feature tensor.
        output_size: Width of the projected pair representation.
        pair_prefix_size: Number of pair-feature columns before the two token
            blocks. ``None`` places all pair features before them.
        bias: Whether to learn an output bias.
        backend: Which backend to run; see
            [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        pair_size: int,
        token_size: int,
        output_size: int,
        *,
        pair_prefix_size: int | None = None,
        bias: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__(pair_size + 2 * token_size, output_size, bias=bias)
        self.pair_prefix_size = pair_size if pair_prefix_size is None else pair_prefix_size
        self.backend = resolve_config_backend(
            backend,
            supported=("oops", "eager"),
            public_default="eager",
        )

    def forward(self, pair_features: Tensor, token_features: Tensor) -> Tensor:
        r"""
        Project factorized pair and per-token features.

        Args:
            pair_features: Pairwise features with shape ``[..., N, N, P]``.
            token_features: Per-token features with shape ``[..., N, T]``.

        Returns:
            Projected pair features with shape ``[..., N, N, C]``.
        """

        return factorized_pair_linear(
            pair_features,
            token_features,
            self.weight,
            self.bias,
            pair_prefix_size=self.pair_prefix_size,
            backend=self.backend,
        )
