from __future__ import annotations

import pytest

from ofoldx.modules.head import ConfidenceHead


@pytest.mark.parametrize("backend", ("torch", "oops"))
def test_confidence_head_propagates_backend(backend: str) -> None:
    head = ConfidenceHead(
        single_size=8,
        pair_size=4,
        token_feature_size=6,
        num_blocks=1,
        num_pair_attention_heads=2,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        operator_backend=backend,
    )

    owners = [module.operator_backend for module in head.modules() if hasattr(module, "operator_backend")]
    assert owners
    assert set(owners) == {backend}
