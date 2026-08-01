from __future__ import annotations

import pytest
import torch

from ofoldx.pipelines import resolve_generator


class TestResolveGenerator:
    def test_default(self) -> None:
        assert resolve_generator(None, None) is None

    def test_supplied(self) -> None:
        supplied = torch.Generator()

        assert resolve_generator(None, supplied, device="cuda") is supplied

    def test_seed(self) -> None:
        generator = resolve_generator(0, None, device="cpu")

        assert generator is not None
        assert generator.device == torch.device("cpu")
        assert generator.initial_seed() == 0

    def test_conflict(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_generator(0, torch.Generator())
