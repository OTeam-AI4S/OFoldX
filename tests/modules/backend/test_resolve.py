from __future__ import annotations

import pytest
import torch

from ofoldx.modules.backend.resolve import resolve_backend, resolve_config_backend, resolve_tensor_backend


class TestResolveConfigBackend:
    @pytest.mark.parametrize("backend", ["auto", "eager"])
    def test_preserves_request(self, backend: str) -> None:
        assert (
            resolve_config_backend(
                backend,
                supported=("oops", "eager"),
                public_default="eager",
            )
            == backend
        )

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="unsupported backend 'triton'"):
            resolve_config_backend(
                "triton",
                supported=("oops", "eager"),
                public_default="eager",
            )


@pytest.mark.parametrize("backend", ["sdpa", "oops", "cueq", "apex", "deepspeed"])
def test_resolve_backend_fallback(backend: str) -> None:
    expected = "oops" if backend == "oops" else "eager"

    assert resolve_backend(backend, supported=("oops", "eager"), public_default="eager") == expected


class TestResolveTensorBackend:
    def test_auto_cpu(self) -> None:
        backend = resolve_tensor_backend(
            "auto",
            tensors=(torch.ones(1),),
            supported=("oops", "eager"),
            public_default="eager",
        )

        assert backend == "eager"

    @pytest.mark.parametrize("backend", ["eager", "oops"])
    def test_explicit(self, backend: str) -> None:
        resolved = resolve_tensor_backend(
            backend,
            tensors=(torch.ones(1),),
            supported=("oops", "eager"),
            public_default="eager",
        )

        assert resolved == backend
