# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ofoldx._oops import MissingOopsKernel
from ofoldx.errors import InvalidRequestError
from ofoldx.modules.backend import resolve as resolver
from ofoldx.modules.backend.resolve import resolve_backend, resolve_config_backend, resolve_tensor_backend


def _kernel(*args: object) -> None:
    return None


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Callable[..., object]]:
    r"""Install the named providers; returns the OOps kernel an operator would pass (missing unless named)."""

    def install(*names: str) -> Callable[..., object]:
        for backend in ("cueq", "apex", "deepspeed"):
            monkeypatch.setattr(resolver, f"{backend.upper()}_AVAILABLE", backend in names)
        return _kernel if "oops" in names else MissingOopsKernel("kernel")

    return install


class TestResolveConfigBackend:
    @pytest.mark.parametrize("backend", ["auto", "eager", "sdpa", "oops", "cueq", "apex", "deepspeed"])
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


class TestResolveBackend:
    @pytest.mark.parametrize(
        ("installed", "expected"),
        [
            (("oops", "cueq", "apex", "deepspeed"), "oops"),
            (("cueq", "apex", "deepspeed"), "cueq"),
            (("apex", "deepspeed"), "apex"),
            (("deepspeed",), "deepspeed"),
            ((), "sdpa"),
        ],
    )
    def test_auto(self, providers, installed: tuple[str, ...], expected: str) -> None:
        oops = providers(*installed)

        assert (
            resolve_backend(
                "auto",
                supported=("oops", "cueq", "apex", "deepspeed", "sdpa", "eager"),
                public_default="sdpa",
                oops=oops,
            )
            == expected
        )

    @pytest.mark.parametrize("supported", [("cueq", "oops", "eager"), ("eager", "oops", "cueq")])
    def test_priority(self, providers, supported: tuple[str, ...]) -> None:
        oops = providers("oops", "cueq")

        assert resolve_backend("auto", supported=supported, public_default="eager", oops=oops) == supported[0]

    @pytest.mark.parametrize("backend", ["sdpa", "oops", "cueq", "apex", "deepspeed"])
    def test_unsupported(self, providers, backend: str) -> None:
        providers("cueq", "apex", "deepspeed")

        assert resolve_backend(backend, supported=("eager",), public_default="eager") == "eager"

    @pytest.mark.parametrize(
        ("installed", "backend", "expected"),
        [
            (("cueq",), "cueq", "cueq"),
            (("apex",), "cueq", "cueq"),
            (("oops", "cueq"), "eager", "eager"),
        ],
    )
    def test_explicit(self, providers, installed: tuple[str, ...], backend: str, expected: str) -> None:
        oops = providers(*installed)

        assert (
            resolve_backend(
                backend,
                supported=("oops", "cueq", "apex", "deepspeed", "eager"),
                public_default="eager",
                oops=oops,
            )
            == expected
        )

    def test_explicit_oops_missing(self, providers) -> None:
        oops = providers("cueq")

        with pytest.raises(InvalidRequestError, match="oops_operator_unavailable.*operator='kernel'"):
            resolve_backend("oops", supported=("oops", "cueq", "eager"), public_default="eager", oops=oops)


class TestResolveTensorBackend:
    def test_auto_cpu(self, providers) -> None:
        oops = providers("oops", "cueq", "apex", "deepspeed")
        backend = resolve_tensor_backend(
            "auto",
            tensors=(torch.ones(1),),
            supported=("oops", "cueq", "apex", "deepspeed", "eager"),
            public_default="eager",
            oops=oops,
        )

        assert backend == "eager"

    @pytest.mark.parametrize("installed", [("oops", "cueq"), ("cueq",), ()])
    def test_auto_cuda(self, providers, installed: tuple[str, ...]) -> None:
        oops = providers(*installed)
        with FakeTensorMode():
            hidden = torch.empty(1, device="cuda")
            backend = resolve_tensor_backend(
                "auto",
                tensors=(hidden, None),
                supported=("oops", "cueq", "sdpa", "eager"),
                public_default="sdpa",
                oops=oops,
            )

        assert backend == (installed[0] if installed else "sdpa")

    @pytest.mark.parametrize("backend", ["eager", "oops"])
    def test_explicit(self, providers, backend: str) -> None:
        oops = providers("oops", "cueq", "apex", "deepspeed")
        resolved = resolve_tensor_backend(
            backend,
            tensors=(torch.ones(1),),
            supported=("oops", "cueq", "apex", "deepspeed", "eager"),
            public_default="eager",
            oops=oops,
        )

        assert resolved == backend
