from __future__ import annotations

import torch

import ofoldx.modules.backend.resolve as resolve_module
from ofoldx.modules.backend.resolve import resolve_config_operator_backend, resolve_tensor_operator_backend


def test_resolve_config_operator_backend_preserves_auto(monkeypatch) -> None:
    monkeypatch.setattr(resolve_module, "oops_available", lambda: True)

    backend = resolve_config_operator_backend(
        "auto",
        supported=("oops", "eager"),
        public_default="eager",
    )

    assert backend == "auto"


def test_resolve_tensor_operator_backend_auto_uses_public_default_for_cpu_tensors(monkeypatch) -> None:
    monkeypatch.setattr(resolve_module, "oops_available", lambda: True)

    backend = resolve_tensor_operator_backend(
        "auto",
        tensors=(torch.ones(1),),
        supported=("oops", "eager"),
        public_default="eager",
    )

    assert backend == "eager"


def test_resolve_tensor_operator_backend_keeps_explicit_oops_for_cpu_tensors(monkeypatch) -> None:
    monkeypatch.setattr(resolve_module, "oops_available", lambda: True)

    backend = resolve_tensor_operator_backend(
        "oops",
        tensors=(torch.ones(1),),
        supported=("oops", "eager"),
        public_default="eager",
    )

    assert backend == "oops"
