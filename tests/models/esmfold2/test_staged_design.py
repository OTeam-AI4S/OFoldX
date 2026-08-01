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

"""Behavior of repository-backed staged ESMFold2 design."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from multimolecule import EsmCConfig, EsmCForMaskedLM

from ofoldx.errors import InvalidRequestError
from ofoldx.models.esmfold2 import (
    EsmFold2DesignPipeline,
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalPipeline,
    EsmFold2ExperimentalProcessor,
)


@pytest.fixture
def saved_design(tmp_path, esm_config):
    torch.manual_seed(91)
    template = esm_config(True)
    template.diffusion_structure_head.num_distogram_bins = 128
    first = EsmCForMaskedLM(EsmCConfig(**template.esmc.dict()))
    second = deepcopy(first)
    with torch.no_grad():
        next(second.parameters()).add_(1)
    paths = (tmp_path / "esmc-first", tmp_path / "esmc-second")
    first.save_pretrained(paths[0])
    second.save_pretrained(paths[1])
    children = {}
    for slot in EsmFold2DesignPipeline.component_layout:
        config = deepcopy(template)
        config.confidence.enabled = not slot.role.startswith("base")
        use_second = slot.role == "base6b_step250k"
        source = paths[1] if use_second else paths[0]
        esmc = deepcopy(second if use_second else first)
        if slot.role != "base300m_step250k":
            config.esmc_repository = str(source)
            config.esmc_revision = ("b" if use_second else "a") * 40
        children[slot.role] = EsmFold2ExperimentalPipeline(
            model=EsmFold2ExperimentalModel(config, esmc=esmc).eval().set_backend("eager"),
            processor=EsmFold2ExperimentalProcessor(),
        )
    original = EsmFold2DesignPipeline(**children)
    destination = tmp_path / "design"
    original.save(destination)
    return destination, original


def test_staged_load_shared_and_distinct_sources(saved_design):
    destination, original = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    assert tuple(staged.components) == tuple(slot.role for slot in original.component_layout)
    store = staged._composition_store
    assert store is not None and not store.loaded
    with store.session():
        roles = (
            "fast",
            "fast_cutoff2025",
            "experimental",
            "experimental_cutoff2025",
            "base6b_step250k",
            "base300m_step250k",
        )
        with store.use(*roles) as components:
            fast, cutoff, _, _, different, inline = components
            shared = next(fast.model.esmc.parameters())
            assert shared.data_ptr() == next(cutoff.model.esmc.parameters()).data_ptr()
            assert shared.data_ptr() != next(different.model.esmc.parameters()).data_ptr()
            assert inline.model.config.esmc_repository is None
            for slot, component in zip(roles, components):
                torch.testing.assert_close(
                    component.model.state_dict(), original.components[slot].model.state_dict(), rtol=0, atol=0
                )
    assert not store.loaded and not store.backbones


def test_staged_save_roundtrip_and_placement(saved_design, tmp_path):
    destination, original = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination).to(dtype=torch.float64).set_backend("eager")
    saved = tmp_path / "saved-again"
    staged.save(saved)
    again = EsmFold2DesignPipeline.from_pretrained(saved, dtype=torch.float64, backend="eager")
    child = again.load_component(saved, role="fast", dtype=torch.float64, backend="eager")
    expected = original.fast.model.to(dtype=torch.float64)
    torch.testing.assert_close(child.model.state_dict(), expected.state_dict(), rtol=0, atol=0)


def test_staged_session_cleans_up_after_error(saved_design):
    destination, _ = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    store = staged._composition_store
    assert store is not None
    with pytest.raises(RuntimeError, match="deliberate"), store.session(), store.use("fast"):
        raise RuntimeError("deliberate")
    assert not store.loaded and not store.backbones and not store.active
    with store.session():
        with store.use("fast") as (pipeline,):
            assert pipeline.model.config.esmc_repository is not None
        with pytest.raises(InvalidRequestError), store.session():
            pass


def test_active_repository_workflow_rejects_reentry_and_child_access(saved_design, library, monkeypatch):
    destination, original = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    request = staged.prepare("AG", num_binder_residues=1, is_antibody=False)

    def during_stage(*args, **kwargs):
        with pytest.raises(InvalidRequestError) as child_error:
            _ = staged.fast
        assert child_error.value.code == "value_invalid"
        with pytest.raises(InvalidRequestError) as reentry_error:
            staged.forward(request, seed=7, library=library)
        assert reentry_error.value.code == "value_invalid"
        assert original.fast is original.components["fast"]
        raise RuntimeError("stop staged call")

    monkeypatch.setattr(staged, "_run_design", during_stage)
    with pytest.raises(RuntimeError, match="stop staged call"):
        staged.forward(request, seed=7, library=library)
    assert staged.fast is staged.fast


def test_staged_model_matches_input_vjp(saved_design, peptide):
    destination, original = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    store = staged._composition_store
    assert store is not None
    prepared = original.fast.prepare(peptide("ALA", "GLY")).features
    soft = torch.randn(1, 2, 72).softmax(-1).requires_grad_()
    other = soft.detach().clone().requires_grad_()
    expected = original.fast.model(prepared, residue_type_probabilities=soft, num_recycles=1)
    with store.session(), store.use("fast") as (pipeline,):
        actual = pipeline.model(prepared, residue_type_probabilities=other, num_recycles=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), other)[0],
            torch.autograd.grad(expected.sum(), soft)[0],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(("use_scaling", "num_candidates"), [(False, 1), (True, 1), (False, 2)])
def test_staged_design_matches_outputs_and_rng(saved_design, library, monkeypatch, use_scaling, num_candidates):
    import numpy as np

    destination, _ = saved_design
    # Baseline independently loads every component, as the existing public
    # composite loader did before staging.
    children = {
        slot.role: EsmFold2ExperimentalPipeline.from_pretrained(destination / slot.subdirectory, backend="eager")
        for slot in EsmFold2DesignPipeline.component_layout
    }
    original = EsmFold2DesignPipeline(**children)
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    original_initial = EsmFold2DesignPipeline._initial_logits

    def initial(prompt, count, generator, device):
        logits, mask = original_initial(prompt, count, generator, device)
        with torch.no_grad():
            logits.fill_(0)
            logits[..., 0] = 0.02  # Keep the candidate inside the small CCD fixture.
            if count > 1:
                logits[1, :, 7] = 0.03  # Glycine has a different atom count from alanine.
        return logits, mask

    original_forward = EsmFold2ExperimentalPipeline.forward

    def bounded_forward(self, *args, **kwargs):
        kwargs["num_sampling_steps"] = 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(EsmFold2DesignPipeline, "_initial_logits", staticmethod(initial))
    monkeypatch.setattr(EsmFold2ExperimentalPipeline, "forward", bounded_forward)
    request = original.prepare(
        "AG", num_binder_residues=1, num_candidates=num_candidates, num_optimization_steps=1, is_antibody=False
    )
    generator_a = torch.Generator().manual_seed(771)
    generator_b = torch.Generator().manual_seed(771)
    expected = original.forward(request, use_scaling_critics=use_scaling, generator=generator_a, library=library)
    actual = staged.forward(request, use_scaling_critics=use_scaling, generator=generator_b, library=library)
    assert torch.equal(generator_a.get_state(), generator_b.get_state())
    assert len(actual.candidates) == len(expected.candidates) == num_candidates
    for index in range(num_candidates):
        assert actual.candidates[index].sequence == expected.candidates[index].sequence
        assert actual.candidates[index].trajectory == expected.candidates[index].trajectory
        torch.testing.assert_close(
            actual.candidates[index].final_logits, expected.candidates[index].final_logits, rtol=0, atol=0
        )
        for left, right in zip(actual.hero_critic_evaluations[index], expected.hero_critic_evaluations[index]):
            assert left.checkpoint == right.checkpoint
            np.testing.assert_allclose(
                left.prediction.structure.coordinates, right.prediction.structure.coordinates, rtol=0, atol=0
            )
        for left, right in zip(actual.scaling_critic_evaluations[index], expected.scaling_critic_evaluations[index]):
            assert left.checkpoint == right.checkpoint
            np.testing.assert_allclose(left.structure.coordinates, right.structure.coordinates, rtol=0, atol=0)
        assert len(actual.hero_critic_evaluations[index]) == 4
        assert len(actual.scaling_critic_evaluations[index]) == (15 if use_scaling else 0)
    if not use_scaling:
        repeated = staged.forward(
            request, use_scaling_critics=False, generator=torch.Generator().manual_seed(771), library=library
        )
        assert repeated.candidates[0].sequence == actual.candidates[0].sequence
        torch.testing.assert_close(
            repeated.candidates[0].final_logits, actual.candidates[0].final_logits, rtol=0, atol=0
        )


def test_explicit_children_keep_identity_and_independent_mutations(saved_design, tmp_path):
    destination, _ = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    fast = staged.fast
    cutoff = staged.fast_cutoff2025
    assert staged.fast is fast and staged.fast_cutoff2025 is cutoff
    before = next(cutoff.model.esmc.parameters()).detach().clone()
    original_fast = next(fast.model.esmc.parameters()).detach().clone()
    with torch.no_grad():
        next(fast.model.esmc.parameters()).add_(1)
    torch.testing.assert_close(next(cutoff.model.esmc.parameters()), before, rtol=0, atol=0)
    with torch.no_grad():
        next(fast.model.esmc.parameters()).copy_(original_fast)
        fast.model.pair_left_projection.weight.add_(1)
    staged.to(dtype=torch.float64).set_backend("eager")
    assert staged.fast is fast
    assert next(fast.model.esmc.parameters()).dtype == torch.float64
    saved = tmp_path / "mutated"
    staged.save(saved)
    reloaded = EsmFold2DesignPipeline.load_component(saved, role="fast", dtype=torch.float64, backend="eager")
    torch.testing.assert_close(reloaded.model.state_dict(), fast.model.state_dict(), rtol=0, atol=0)
    assert staged.fast is fast


def test_conversion_provenance_iterates_staged_children_without_retaining_all(saved_design, tmp_path, monkeypatch):
    import json

    from ofoldx.conversion import save_repository

    destination, _ = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    store = staged._composition_store
    assert store is not None
    load = store.load
    peak = 0

    def tracked(*args, **kwargs):
        nonlocal peak
        result = load(*args, **kwargs)
        peak = max(peak, len(store.loaded) + len(store.pinned))
        return result

    monkeypatch.setattr(store, "load", tracked)
    checkpoint = tmp_path / "source.bin"
    checkpoint.write_bytes(b"source")
    saved = tmp_path / "converted"
    save_repository(
        staged,
        saved,
        sources={"checkpoint": checkpoint},
        converter="ofoldx.models.esmfold2.convert.convert_esmfold2_checkpoint",
    )
    record = json.loads((saved / "conversion.json").read_text())
    assert set(record["components"]) == {".", *(slot.role for slot in staged.component_layout)}
    assert peak == 1 and not store.loaded and not store.pinned and not store.backbones
    assert EsmFold2DesignPipeline.from_pretrained(saved).load_component(saved, role="fast").model is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_staged_cuda_shared_sources_and_input_vjp(saved_design, peptide):
    destination, original = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, device="cuda", backend="eager")
    store = staged._composition_store
    assert store is not None
    reference = original.fast.to(device="cuda")
    prepared = reference.prepare(peptide("ALA", "GLY")).features
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        soft = torch.randn(1, 2, 72, device="cuda").softmax(-1).requires_grad_()
        other = soft.detach().clone().requires_grad_()
        expected = reference.model(prepared, residue_type_probabilities=soft, num_recycles=1)
        with store.session(), store.use("fast", "fast_cutoff2025", "base6b_step250k") as children:
            fast, cutoff, different = children
            assert next(fast.model.esmc.parameters()).data_ptr() == next(cutoff.model.esmc.parameters()).data_ptr()
            assert next(fast.model.esmc.parameters()).data_ptr() != next(different.model.esmc.parameters()).data_ptr()
            actual = fast.model(prepared, residue_type_probabilities=other, num_recycles=1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                torch.autograd.grad(actual.sum(), other)[0],
                torch.autograd.grad(expected.sum(), soft)[0],
                rtol=0,
                atol=0,
            )
    finally:
        torch.use_deterministic_algorithms(previous)


def test_conversion_provenance_failure_releases_staged_child(saved_design, tmp_path, monkeypatch):
    from pathlib import Path

    from ofoldx import conversion

    destination, _ = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    store = staged._composition_store
    assert store is not None
    original = conversion._source_record

    def fail_child(source, cache):
        if (
            isinstance(source, conversion.SourceCheckpoint)
            and source.path is not None
            and Path(source.path) == destination / "fast"
        ):
            raise RuntimeError("deliberate provenance failure")
        return original(source, cache)

    monkeypatch.setattr(conversion, "_source_record", fail_child)
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    with pytest.raises(RuntimeError, match="deliberate provenance failure"):
        conversion.save_repository(
            staged,
            tmp_path / "converted",
            sources={"checkpoint": source},
            converter="ofoldx.models.esmfold2.convert.convert_esmfold2_checkpoint",
        )
    assert not store.active and not store.loaded and not store.backbones
    with store.session(), store.use("fast") as (pipeline,):
        assert pipeline.model.config.esmc_repository is not None


def test_active_design_rejects_save_and_mutation_before_side_effects(saved_design, library, monkeypatch, tmp_path):
    destination, _ = saved_design
    staged = EsmFold2DesignPipeline.from_pretrained(destination, backend="eager")
    request = staged.prepare("AG", num_binder_residues=1, is_antibody=False)
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "pipeline.json"
    marker.write_text("original marker\n")
    missing = tmp_path / "not-created"

    def during_stage(*args, pairs, **kwargs):
        fast = pairs[0][1]
        dtype = next(fast.model.parameters()).dtype
        backend = fast.model.backend
        for operation in (
            lambda: staged.save(target),
            lambda: staged.save(missing),
            lambda: staged.to(dtype=torch.float64),
            lambda: staged.set_backend("auto"),
        ):
            with pytest.raises(InvalidRequestError) as error:
                operation()
            assert error.value.code == "value_invalid"
            assert marker.read_text() == "original marker\n"
            assert not missing.exists()
            assert next(fast.model.parameters()).dtype == dtype
            assert fast.model.backend == backend
        raise RuntimeError("stop staged call")

    monkeypatch.setattr(staged, "_run_design", during_stage)
    with pytest.raises(RuntimeError, match="stop staged call"):
        staged.forward(request, seed=7, library=library)
    staged.to(dtype=torch.float64).set_backend("auto")
    assert next(staged.fast.model.parameters()).dtype == torch.float64
    assert staged.fast.model.backend == "auto"
