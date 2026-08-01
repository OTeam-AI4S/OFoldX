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

"""Experimental ESMFold2 repository and external language-model behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from ofoldx.auto import AutoModel, AutoPipelineForFold
from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import InvalidRequestError, RequestRef
from ofoldx.models.esmfold2.pipeline import _esmc_pseudo_perplexity_nll


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_referenced_roundtrip_and_gradients(experimental_referenced_repository, peptide, device):
    directory, reference = experimental_referenced_repository
    full = reference.model.state_dict()
    external = {name for name in full if name.startswith(("esmc.", "esmc_lm_head."))}
    assert any(name.startswith("esmc.") for name in external)
    assert any(name.startswith("esmc_lm_head.") for name in external)
    assert set(load_file(directory / "model.safetensors")) == full.keys() - external
    loaded = AutoPipelineForFold.from_pretrained(directory, backend="eager", device=device)
    reference.to(device=device)
    full = reference.model.state_dict()
    torch.testing.assert_close(loaded.model.state_dict(), full, rtol=0, atol=0)
    assert loaded.model.esmc_lm_head.decoder.bias is None
    assert loaded.model.esmc_lm_head.bias is not None
    assert not loaded.model.training and not loaded.model.esmc.training and not loaded.model.esmc_lm_head.training
    assert not any(p.requires_grad for p in loaded.model.esmc.parameters())
    assert not any(p.requires_grad for p in loaded.model.esmc_lm_head.parameters())

    inputs = loaded.processor.prepare_input(peptide("ALA", "GLY")).features
    soft = torch.randn(1, 2, len(TOKENS), device=device).softmax(-1).requires_grad_()
    other_soft = soft.detach().clone().requires_grad_()
    expected = reference.model(inputs, residue_type_probabilities=soft, num_recycles=1)
    actual = loaded.model(inputs, residue_type_probabilities=other_soft, num_recycles=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), other_soft)[0], torch.autograd.grad(expected.sum(), soft)[0], rtol=0, atol=0
    )

    design = torch.randn(1, 3, 20, device=device).softmax(-1).requires_grad_()
    other_design = design.detach().clone().requires_grad_()
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    expected = _esmc_pseudo_perplexity_nll(
        reference.model, design, mask, generator=torch.Generator(device=device).manual_seed(7)
    )
    actual = _esmc_pseudo_perplexity_nll(
        loaded.model, other_design, mask, generator=torch.Generator(device=device).manual_seed(7)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), other_design)[0],
        torch.autograd.grad(expected.sum(), design)[0],
        rtol=0,
        atol=0,
    )


def test_complete_inline_repository_still_loads(tmp_path, experimental_referenced_repository):
    _, reference = experimental_referenced_repository
    reference.model.config.esmc_repository = None
    directory = tmp_path / "complete"
    reference.save(directory)
    loaded = AutoPipelineForFold.from_pretrained(directory, backend="eager")
    torch.testing.assert_close(loaded.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["missing", "unexpected", "external"])
def test_referenced_head_rejects_invalid_state(experimental_referenced_repository, damage):
    directory, reference = experimental_referenced_repository
    path = directory / "model.safetensors"
    state = {name: value.clone() for name, value in load_file(path).items()}
    if damage == "missing":
        state.pop(next(iter(state)))
    elif damage == "unexpected":
        state["unrecognized.weight"] = torch.zeros(1)
    else:
        name = next(name for name in reference.model.state_dict() if name.startswith("esmc_lm_head."))
        state[name] = reference.model.state_dict()[name].clone()
    save_file(state, path)
    with pytest.raises(RuntimeError, match="folding-head parameters"):
        AutoModel.from_pretrained(directory)


@pytest.mark.parametrize("damage", ["missing_trunk", "missing_mlm", "unexpected", "config"])
def test_referenced_esmc_rejects_invalid_state(experimental_referenced_repository, damage):
    directory, reference = experimental_referenced_repository
    source = Path(reference.model.config.esmc_repository)
    if damage == "config":
        config_path = source / "config.json"
        config = json.loads(config_path.read_text())
        config["layer_norm_eps"] = 0.01
        config_path.write_text(json.dumps(config))
    else:
        path = source / "model.safetensors"
        state = {name: value.clone() for name, value in load_file(path).items()}
        if damage == "missing_trunk":
            state.pop(next(name for name in state if name.startswith("model.encoder.layer.0.")))
        elif damage == "missing_mlm":
            state.pop("lm_head.transform.dense.weight")
        else:
            state["unused.weight"] = torch.zeros(1)
        save_file(state, path)
    with pytest.raises(RuntimeError, match="ESMC repository"):
        AutoModel.from_pretrained(directory)


@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("remote", [False, True])
def test_conversion_writes_head_without_loading_esmc_weights(
    tmp_path, monkeypatch, experimental_referenced_repository, dtype, remote
):
    from copy import deepcopy

    from multimolecule import EsmCForMaskedLM

    from ofoldx.models.esmfold2 import convert

    _, reference = experimental_referenced_repository
    config = deepcopy(reference.model.config)
    source_esmc = config.esmc_repository
    revision = "a" * 40
    if remote:
        from ofoldx.hub import ResolvedRepository

        def snapshot(repo_id, **kwargs):
            assert repo_id == "example/esmc" and kwargs["revision"] == "main"
            assert kwargs["cache_dir"] == tmp_path / "hub-cache"
            return ResolvedRepository(Path(source_esmc), repo_id, revision)

        original_config = convert.EsmCConfig.from_pretrained
        original_model = EsmCForMaskedLM.from_pretrained

        def load_config(repo_id, **kwargs):
            if repo_id == "example/esmc":
                assert kwargs["revision"] == revision
                return original_config(source_esmc)
            return original_config(repo_id, **kwargs)

        def load_model(repo_id, **kwargs):
            assert repo_id == "example/esmc" and kwargs.pop("revision") == revision
            return original_model(source_esmc, **kwargs)

        monkeypatch.setattr(convert, "resolve_hub_repository", snapshot)
        monkeypatch.setattr(convert.EsmCConfig, "from_pretrained", load_config)
        monkeypatch.setattr(EsmCForMaskedLM, "from_pretrained", load_model)
    config.esmc_repository = None
    head = {
        name: value.detach().clone()
        for name, value in reference.model.state_dict().items()
        if not name.startswith(("esmc.", "esmc_lm_head."))
    }
    source = tmp_path / "original.safetensors"
    save_file({}, source)
    source_config = tmp_path / "config.json"
    source_config.write_text(
        json.dumps(
            {"model_type": "esmfold2", "type": "experimental", "esmc_id": "tiny", "lm_d_model": 8, "lm_num_layers": 1}
        )
    )
    monkeypatch.setattr(convert, "experimental_config_from_biohub", lambda _: deepcopy(config))
    monkeypatch.setattr(convert, "load_esmfold2_state_dict", lambda _: head)
    destination = tmp_path / "converted"
    with monkeypatch.context() as no_weights:

        def fail_load(*args, **kwargs):
            raise AssertionError("conversion should not hydrate the external ESMC")

        no_weights.setattr(EsmCForMaskedLM, "from_pretrained", fail_load)
        convert.convert_esmfold2_experimental_checkpoint(
            source,
            destination,
            esmc_source="example/esmc" if remote else source_esmc,
            esmc_revision="main" if remote else None,
            cache_dir=tmp_path / "hub-cache" if remote else None,
            config_path=source_config,
            dtype=dtype,
        )
    stored = load_file(destination / "model.safetensors")
    assert set(stored) == set(head)
    assert not any(value.is_meta for value in stored.values())
    loaded = AutoPipelineForFold.from_pretrained(destination, backend="eager")
    if dtype is not None:
        reference.model.to(dtype=dtype)
    torch.testing.assert_close(loaded.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)
    record = json.loads((destination / "conversion.json").read_text())
    assert record["sources"]["esmc"]["reference"] == ("example/esmc" if remote else source_esmc)
    assert record["sources"]["esmc"]["revision"] == (revision if remote else None)
    assert loaded.model.config.esmc_revision == (revision if remote else None)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_referenced_roundtrip_preserves_model_dtype(tmp_path, experimental_referenced_repository, dtype):
    _, reference = experimental_referenced_repository
    reference.model.to(dtype=dtype)
    directory = tmp_path / "double"
    reference.save(directory)
    loaded = AutoPipelineForFold.from_pretrained(directory, backend="eager")
    torch.testing.assert_close(loaded.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)
    assert all(parameter.dtype == dtype for parameter in loaded.model.parameters())
    overridden = AutoPipelineForFold.from_pretrained(directory, backend="eager", dtype=torch.float32)
    assert all(parameter.dtype == torch.float32 for parameter in overridden.model.parameters())


def test_referenced_different_trunk_and_mlm_dtypes(tmp_path, experimental_referenced_repository):
    _, reference = experimental_referenced_repository
    reference.model.esmc.to(dtype=torch.float16)
    reference.model.esmc_lm_head.to(dtype=torch.float64)
    directory = tmp_path / "separate"
    reference.save(directory)
    loaded = AutoPipelineForFold.from_pretrained(directory, backend="eager")
    torch.testing.assert_close(loaded.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)


def test_referenced_mixed_dtype_within_esmc_is_rejected(tmp_path, experimental_referenced_repository):
    _, reference = experimental_referenced_repository
    parameter = next(reference.model.esmc.parameters())
    parameter.data = parameter.data.to(torch.float64)
    with pytest.raises(InvalidRequestError) as caught:
        reference.save(tmp_path / "mixed")
    assert caught.value.code == "value_invalid"
    assert caught.value.refs == (RequestRef("esmc_dtype", repr({torch.float32, torch.float64})),)
    assert caught.value.detail["expected"] == "one supported floating dtype"


def test_invalid_referenced_dtype_is_rejected(experimental_referenced_repository):
    directory, _ = experimental_referenced_repository
    path = directory / "model.json"
    config = json.loads(path.read_text())
    config["esmc_dtype"] = "not-a-dtype"
    path.write_text(json.dumps(config))
    with pytest.raises(InvalidRequestError) as caught:
        AutoModel.from_pretrained(directory)
    assert caught.value.code == "value_invalid"
    assert caught.value.refs == (RequestRef("esmc_dtype", repr("not-a-dtype")),)
    assert "float32" in caught.value.detail["expected"]


def test_nondefault_experimental_esmc_requires_reference(tmp_path):
    from ofoldx.models.esmfold2 import convert

    checkpoint = tmp_path / "model.safetensors"
    save_file({}, checkpoint)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "esmfold2", "type": "experimental", "esmc_id": "biohub/ESMC-300M-250000"})
    )
    with pytest.raises(InvalidRequestError) as caught:
        convert.convert_esmfold2_experimental_checkpoint(checkpoint, tmp_path / "output")
    assert caught.value.code == "value_invalid"
    assert caught.value.refs == (RequestRef("esmc_source", repr(None)),)
    assert caught.value.detail["expected"] == "MM ESMC repository"
    assert caught.value.detail["upstream"] == "biohub/ESMC-300M-250000"
