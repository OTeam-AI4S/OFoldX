from __future__ import annotations

import pytest
import torch

from ofoldx.pipelines.generator import resolve_generator


def test_resolve_generator() -> None:
    supplied = torch.Generator()

    assert resolve_generator(None, None) is None
    assert resolve_generator(None, supplied) is supplied
    assert resolve_generator(17, None).initial_seed() == 17


def test_resolve_generator_rejects_seed_with_generator() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_generator(17, torch.Generator())
