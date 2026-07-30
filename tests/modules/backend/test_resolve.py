from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest
import torch

import ofoldx.modules.backend.resolve as resolve_module
from ofoldx.modules.backend import _oops
from ofoldx.modules.backend.resolve import (
    resolve_backend,
    resolve_config_backend,
    resolve_tensor_backend,
)


def _execute_oops_import(monkeypatch, missing_name: str):
    real_import = builtins.__import__

    def import_with_missing_oops(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "oops":
            raise ModuleNotFoundError(f"No module named {missing_name!r}", name=missing_name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_oops)
    spec = importlib.util.spec_from_file_location("test_oops_import_boundary", Path(_oops.__file__))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_only_a_missing_top_level_oops_package_is_optional(monkeypatch) -> None:
    module = _execute_oops_import(monkeypatch, "oops")

    assert module.OOPS_AVAILABLE is False


def test_a_broken_oops_dependency_propagates(monkeypatch) -> None:
    with pytest.raises(ModuleNotFoundError, match="oops_dependency"):
        _execute_oops_import(monkeypatch, "oops_dependency")


def test_resolve_config_backend_preserves_auto() -> None:
    backend = resolve_config_backend(
        "auto",
        supported=("oops", "eager"),
        public_default="eager",
    )

    assert backend == "auto"


def test_resolve_config_backend_preserves_eager() -> None:
    backend = resolve_config_backend(
        "eager",
        supported=("oops", "sdpa", "eager"),
        public_default="sdpa",
    )

    assert backend == "eager"


def test_explicit_unsupported_backend_fails() -> None:
    with pytest.raises(ValueError, match="unsupported backend 'triton'"):
        resolve_config_backend(
            "triton",
            supported=("oops", "eager"),
            public_default="eager",
        )


@pytest.mark.parametrize("backend", ["sdpa", "oops", "cueq", "apex", "deepspeed"])
def test_provider_uses_public_implementation_when_operation_is_not_supported(backend: str) -> None:
    expected = "oops" if backend == "oops" else "eager"

    assert resolve_backend(backend, supported=("oops", "eager"), public_default="eager") == expected


def test_resolve_tensor_backend_auto_uses_public_default_for_cpu_tensors(monkeypatch) -> None:
    monkeypatch.setattr(resolve_module, "OOPS_AVAILABLE", True)

    backend = resolve_tensor_backend(
        "auto",
        tensors=(torch.ones(1),),
        supported=("oops", "eager"),
        public_default="eager",
    )

    assert backend == "eager"


def test_resolve_tensor_backend_keeps_explicit_eager() -> None:
    backend = resolve_tensor_backend(
        "eager",
        tensors=(torch.ones(1),),
        supported=("oops", "sdpa", "eager"),
        public_default="sdpa",
    )

    assert backend == "eager"


def test_resolve_tensor_backend_keeps_explicit_oops_for_cpu_tensors() -> None:
    backend = resolve_tensor_backend(
        "oops",
        tensors=(torch.ones(1),),
        supported=("oops", "eager"),
        public_default="eager",
    )

    assert backend == "oops"
