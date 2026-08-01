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

"""Behavior of ofoldx.models.alphafold2.model."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.models.alphafold2 import (
    AlphaFold2Config,
    AlphaFold2Processor,
    alphafold2_predicted_interface_tm_score,
    alphafold2_predicted_tm_score,
)
from ofoldx.models.alphafold2.model import normalize_alphafold2_input


class TestBatch:
    @pytest.fixture
    def model_inputs(self, tiny_alphafold2, peptide, glycine_template, chains):
        def build(kind, shapes, templates):
            torch.manual_seed(17)
            template = (
                None
                if templates is None
                else {
                    "template_pair_stack": {
                        "num_blocks": 1,
                        "pair_size": 8,
                        "triangle_attention_starting_node": {"num_attention_heads": 1, "head_size": 4, "dropout": 0.0},
                        "triangle_multiplication_outgoing": {"hidden_size": 4},
                    },
                    "attention": {"num_attention_heads": 1},
                }
            )
            pipeline = tiny_alphafold2(
                kind,
                template=template,
                msa={
                    "max_msa_clusters": 2,
                    "max_extra_msa": 1,
                    "masked_msa_enabled": False,
                    "resample_msa_in_recycling": False,
                },
            )
            with torch.no_grad():
                pipeline.model.masked_msa_head.weight.normal_(std=0.1)
            samples = []
            for depth, length in shapes:
                if kind == "initial_guess":
                    sample = pipeline.prepare(
                        chains(A=("ALA",) * (length - 1), B=("GLY",)), binder_instance_id="B"
                    ).features
                else:
                    sample = pipeline.prepare(
                        peptide(*(["ALA"] * length)), templates=(glycine_template,) if templates else ()
                    ).features
                sample.msa_tokens = sample.msa_tokens.repeat(1, depth, 1)
                counts = torch.arange(depth, dtype=sample.msa_deletion_counts.dtype).reshape(1, depth, 1)
                sample.msa_deletion_counts = counts.expand(1, depth, length)
                samples.append(sample)
            return pipeline.model, samples

        return build

    @staticmethod
    def _stack(samples):
        batch_values = {}
        for field in fields(samples[0]):
            values = [getattr(sample, field.name) for sample in samples]
            if values[0] is None:
                batch_values[field.name] = None
            else:
                parts = [value.squeeze(0) for value in values]
                batch_values[field.name] = (
                    torch.stack(parts) if all(part.shape == parts[0].shape for part in parts) else NestedTensor(parts)
                )
        return type(samples[0])(**batch_values, batch_size=[])

    @pytest.mark.parametrize("kind", ["monomer", "multimer"])
    @pytest.mark.parametrize(
        "shapes",
        [((3, 4), (3, 4)), ((1, 4), (3, 4)), ((2, 2), (2, 4)), ((1, 2), (3, 4))],
        ids=["dense", "msa", "residues", "both"],
    )
    @pytest.mark.parametrize("templates", [None, False, True], ids=["disabled", "empty", "observed"])
    def test_values(self, kind, shapes, templates, model_inputs):
        model, samples = model_inputs(kind, shapes, templates)
        batch = self._stack(samples)
        generator = torch.Generator().manual_seed(19)
        reference_generator = torch.Generator().manual_seed(19)
        with torch.inference_mode():
            prediction = model(batch, num_recycles=2, generator=generator)
            references = [model(sample, num_recycles=2, generator=reference_generator) for sample in samples]
        torch.testing.assert_close(generator.get_state(), reference_generator.get_state(), rtol=0, atol=0)
        for name in (
            "single",
            "pair",
            "final_atom_positions",
            "plddt_logits",
            "masked_msa_logits",
            "true_msa",
            "bert_mask",
        ):
            actual = getattr(prediction, name)
            expected = tuple(getattr(reference, name).squeeze(0) for reference in references)
            torch.testing.assert_close(actual.unbind(0), expected)

    @pytest.mark.parametrize("kind", ["monomer", "multimer", "initial_guess"])
    @pytest.mark.parametrize("ragged", [False, True], ids=["dense", "ragged"])
    def test_compiled_inference(self, kind, ragged, model_inputs):
        shapes = ((1, 3), (1, 5 if ragged else 3))
        model, samples = model_inputs(kind, shapes, None if kind == "initial_guess" else True)
        model.config.msa.masked_msa_enabled = kind != "initial_guess"
        batch = self._stack(samples)
        num_recycles = 3 if kind == "multimer" else 2
        compiled = torch.compile(model, backend="aot_eager", fullgraph=kind != "multimer", dynamic=True)
        with torch.inference_mode():
            torch.manual_seed(23)
            expected = model(batch, num_recycles=num_recycles)
            torch.manual_seed(23)
            actual = compiled(batch, num_recycles=num_recycles)
        for name, reference in expected.items():
            torch.testing.assert_close(actual[name].unbind(0), reference.unbind(0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize("kind", ["monomer", "initial_guess"])
    def test_cuda_values(self, kind, model_inputs):
        model, samples = model_inputs(kind, ((1, 3), (1, 5)), None)
        model.cuda().set_backend("auto")
        samples = [sample.to("cuda") for sample in samples]
        with torch.inference_mode():
            actual = model(self._stack(samples), num_recycles=2)
            expected = [model(sample, num_recycles=2) for sample in samples]
        for name, values in actual.items():
            torch.testing.assert_close(values.unbind(0), tuple(reference[name].squeeze(0) for reference in expected))

    @pytest.mark.parametrize("kind", ["monomer", "multimer"])
    def test_precomputed_gradients(self, kind, model_inputs):
        model, samples = model_inputs(kind, ((1, 2), (2, 4)), True)
        model.double()
        for sample in samples:
            batch, depth, length = sample.msa_tokens.shape
            classes = sample.msa_profile.shape[-1]
            sample.msa_features = torch.randn(
                batch, depth, length, 2 * classes + 3, dtype=torch.float64, requires_grad=True
            )
            sample.msa_mask = torch.ones(batch, depth, length, dtype=torch.float64)
            sample.extra_msa_features = torch.zeros(batch, 1, length, classes + 2, dtype=torch.float64)
            sample.extra_msa_mask = torch.zeros(batch, 1, length, dtype=torch.float64)
            sample.true_msa = sample.msa_tokens
            sample.bert_mask = torch.zeros_like(sample.msa_mask)
        prediction = model(self._stack(samples), num_recycles=2)
        references = [model(sample, num_recycles=2) for sample in samples]
        for name in ("masked_msa_logits", "true_msa", "bert_mask"):
            expected = tuple(getattr(reference, name).squeeze(0) for reference in references)
            torch.testing.assert_close(getattr(prediction, name).unbind(0), expected)
        inputs = tuple(sample.msa_features for sample in samples)
        actual = torch.autograd.grad(prediction.masked_msa_logits.square().sum(), inputs)
        expected = torch.autograd.grad(
            sum(reference.masked_msa_logits.square().sum() for reference in references), inputs
        )
        torch.testing.assert_close(actual, expected)


class TestNormalize:
    def test_masked_msa_cells_cleared(self, peptide):
        value = AlphaFold2Processor().prepare_input(peptide("ALA", "GLY")).features
        value.msa_tokens = torch.tensor([[[1, -10000]]])
        value.residue_mask = torch.tensor([[1.0, 0.0]])
        value.msa_deletion_counts = torch.tensor([[[3.0, torch.nan]]])
        normalized = normalize_alphafold2_input(value)
        assert normalized.msa_tokens.tolist() == [[[1, 0]]]
        assert normalized.msa_deletion_counts.tolist() == [[[3.0, 0.0]]]
        assert value.msa_tokens.tolist() == [[[1, -10000]]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


@pytest.fixture(scope="class")
def tm_case(floating_dtype: torch.dtype):
    index = torch.arange(3 * 3 * 7, dtype=floating_dtype).reshape(3, 3, 7)
    logits = (index * 0.37).sin() + (index.remainder(5) * 0.173)
    weights = torch.tensor([0.25, 0.70, 1.13], dtype=floating_dtype)
    roles = torch.tensor([0, 0, 1])
    config = AlphaFold2Config({"heads": {"predicted_aligned_error": {"num_pae_bins": 7}}})
    breaks = np.linspace(0.0, 31.0, 6)
    step = breaks[1] - breaks[0]
    centers = np.r_[breaks + step / 2, breaks[-1] + 3 * step / 2]
    scale = 1.24 * (19 - 15) ** (1.0 / 3.0) - 1.8
    terms = _softmax(logits.double().numpy()) @ (1 / (1 + (centers / scale) ** 2))
    source = weights.double().numpy()
    target = np.broadcast_to(source, terms.shape)
    means = (terms * target).sum(axis=-1) / target.sum(axis=-1)
    ptm = means[np.argmax(means * source)]
    cross = target * (roles.numpy()[:, None] != roles.numpy()[None, :])
    means = (terms * cross).sum(axis=-1) / cross.sum(axis=-1)
    iptm = means[np.argmax(means * source)]
    tolerance = 2e-13 if floating_dtype == torch.float64 else 2e-6
    return logits, weights, roles, config, ptm, iptm, tolerance


class TestTmScore:
    def test_ptm(self, tm_case):
        logits, weights, _, config, expected, _, tolerance = tm_case
        score = alphafold2_predicted_tm_score(logits, weights, config, backend="eager")
        assert score == pytest.approx(expected, rel=tolerance, abs=tolerance)

    def test_iptm(self, tm_case):
        logits, weights, roles, config, _, expected, tolerance = tm_case
        score = alphafold2_predicted_interface_tm_score(logits, roles, weights, config, backend="eager")
        assert score == pytest.approx(expected, rel=tolerance, abs=tolerance)


def test_point_norm() -> None:
    from ofoldx.models.alphafold2.model import AlphaFold2MultimerInvariantPointAttention, AlphaFold2Rigid

    attention = AlphaFold2MultimerInvariantPointAttention(
        hidden_size=2,
        pair_size=1,
        num_attention_heads=1,
        scalar_query_key_size=1,
        scalar_value_size=1,
        num_point_qk=1,
        num_point_v=1,
    ).double()
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.zero_()
        attention.value_point_projection.point_projection.weight[:2].copy_(torch.eye(2))
        attention.output_projection.weight[0, 4] = 1
    single = torch.tensor(
        [[[0.0, 0.0]], [[3e-9, 4e-9]], [[3e-6, 4e-6]], [[3.0, 4.0]]], dtype=torch.float64, requires_grad=True
    )
    result = attention(
        single=single,
        pair=single.new_zeros(4, 1, 1, 1),
        mask=single.new_ones(4, 1),
        rigid=AlphaFold2Rigid.identity((4, 1), device=single.device, dtype=single.dtype),
    )[..., 0]
    torch.testing.assert_close(result, single.new_tensor([[1e-8], [1e-8], [5e-6], [5.0]]))
    gradient = torch.autograd.grad(result.sum(), single)[0]
    torch.testing.assert_close(gradient, single.new_tensor([[[0, 0]], [[0, 0]], [[0.6, 0.8]], [[0.6, 0.8]]]))
