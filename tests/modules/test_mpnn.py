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

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.modules.mpnn import MpnnProteinFeatureEncoder, MpnnSequenceLoss, validate_decoding_ranks


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_ragged_sequence_loss_gradients(label_smoothing):
    values = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [torch.nan, torch.inf, -torch.inf],
            [torch.nan, torch.nan, torch.nan],
            [0.0, 1.0, -1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    layout = NestedTensor([torch.empty(2, 3), torch.empty(3, 3)], ragged_dims=(0,))
    tokens = NestedTensor([torch.tensor([0, -1]), torch.tensor([-1, 2, 1])], ragged_dims=(0,))
    mask = NestedTensor([torch.tensor([True, False]), torch.tensor([False, True, True])], ragged_dims=(0,))
    actual = MpnnSequenceLoss(label_smoothing=label_smoothing)(layout.packed_like(values), tokens, mask)
    expected = torch.nn.functional.cross_entropy(
        values[mask.concat], tokens.concat[mask.concat], label_smoothing=label_smoothing / (1.0 + label_smoothing)
    )
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, values, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert not actual_gradient[~mask.concat].any()


def test_unscored_sequence_payload_gradients():
    logits = torch.tensor([[[1.0, 0.0, -torch.inf], [float("nan"), float("inf"), -torch.inf]]], requires_grad=True)
    tokens = torch.tensor([[0, -1]])
    loss = MpnnSequenceLoss(label_smoothing=0)(logits, tokens, torch.tensor([[True, False]]))
    torch.testing.assert_close(loss, -logits[0, 0].log_softmax(-1)[0])
    expected = torch.zeros_like(logits)
    expected[0, 0] = logits[0, 0].softmax(-1)
    expected[0, 0, 0] -= 1
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], expected)


def test_excluded_smoothed_sequence_class():
    with pytest.raises(ValueError, match="positive target mass"):
        MpnnSequenceLoss()(torch.tensor([[[0.0, -torch.inf]]]), torch.tensor([[0]]), torch.tensor([[True]]))


def test_representable_sequence_loss_mean():
    logits = torch.tensor([[[-30000.0, 0.0]] * 3], dtype=torch.float16, requires_grad=True)
    loss = MpnnSequenceLoss(label_smoothing=0)(
        logits, torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool)
    )
    torch.testing.assert_close(loss, logits.new_tensor(30000.0))
    expected = logits.new_tensor([[[-1 / 3, 1 / 3]] * 3])
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], expected)


def test_sequence_mask_is_availability():
    with pytest.raises(TypeError, match="boolean scored-residue availability"):
        MpnnSequenceLoss()(torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1))


def test_unavailable_backbone_gradients() -> None:
    model = MpnnProteinFeatureEncoder(edge_size=8, num_neighbors=2, num_rbf=3, backend="eager")
    coordinates = torch.randn(1, 3, 4, 3)
    coordinates[:, 1] = float("nan")
    coordinates.requires_grad_()
    edges, _, _ = model(
        coordinates,
        torch.tensor([[True, False, True]]),
        torch.arange(3).unsqueeze(0),
        torch.zeros(1, 3, dtype=torch.long),
    )
    assert torch.isfinite(edges).all()
    gradient = torch.autograd.grad(edges.square().sum(), coordinates)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[:, 1], torch.zeros(1, 4, 3))


@pytest.mark.parametrize(
    "ranks,groups",
    [
        ([[-5, 12, -5, 3]], [[2, 0, 2, 1]]),
        ([[8, 8, -2], [-3, 14, 14]], [[0, 0, 1], [2, 1, 1]]),
        ([[]], [[]]),
    ],
)
def test_decoding_ranks_preserve_exact_causal_units(ranks, groups) -> None:
    ranks = torch.tensor(ranks, dtype=torch.long)
    groups = torch.tensor(groups, dtype=torch.long)
    before = ranks.clone()
    validate_decoding_ranks(ranks, groups)
    # Equal ordinals identify exactly the same causal unit; rank gaps and signs
    # do not change which residues can condition on one another.
    torch.testing.assert_close(ranks, before, rtol=0, atol=0)
    assert torch.equal(ranks[:, :, None] == ranks[:, None, :], groups[:, :, None] == groups[:, None, :])


@pytest.mark.parametrize(
    "ranks,groups,error,message",
    [
        ([[0, 1]], [[0]], ValueError, "wrong exact shape"),
        ([[0.0, 1.0]], [[0, 1]], TypeError, "dtype torch.long"),
        ([[0, 1]], [[0, 0]], AssertionError, "equal decoding ranks"),
        ([[4, 4]], [[0, 1]], AssertionError, "distinct decoding ranks"),
    ],
)
def test_decoding_ranks_reject_ambiguous_causal_units(ranks, groups, error, message) -> None:
    with pytest.raises(error, match=message):
        validate_decoding_ranks(torch.tensor(ranks), torch.tensor(groups))


@pytest.mark.parametrize("family", ["proteinmpnn", "ligandmpnn", "nampnn"])
def test_explicit_causal_units_on_complete_mpnn_forward(family: str, peptide) -> None:
    from importlib import import_module

    module = import_module(f"ofoldx.models.{family}")
    names = {
        "proteinmpnn": "ProteinMpnn",
        "ligandmpnn": "LigandMpnn",
        "nampnn": "NaMpnn",
    }
    prefix = names[family]
    structure = peptide("ALA", "CYS", "ASN", "HIS")
    input = getattr(module, prefix + "Processor")(dtype=torch.float64).prepare_input(structure).features
    input.tied_group_ids = torch.tensor([[0, 0, 1, 2]])
    ranks = torch.tensor([[-7, -7, 9, 21]])
    options = {"hidden_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1, "num_neighbors": 2, "dropout": 0.0}
    if family == "nampnn":
        options.update(feature_size=8, positional_embedding_size=4, num_rbf=3, include_predicted_na_n=False)
    torch.manual_seed(71)
    model = getattr(module, prefix + "Model")(getattr(module, prefix + "Config")(options)).to(torch.float64)
    model.set_backend("eager")
    model.eval()
    output = model(input, decoding_ranks=ranks)
    assert output.logits.shape[:2] == ranks.shape
    assert not output.logits.isnan().any()
    assert torch.isfinite(output.logits).any(-1).all()
    torch.testing.assert_close(output.decoding_ranks, ranks, rtol=0, atol=0)
    with pytest.raises(AssertionError, match="equal decoding ranks"):
        model(input, decoding_ranks=torch.tensor([[-7, -6, 9, 21]]))


@pytest.mark.parametrize("family", ["proteinmpnn", "ligandmpnn", "nampnn"])
@pytest.mark.parametrize("backend", ["auto", "eager"])
def test_ragged_cpu_mpnn_forward_respects_backend(family: str, backend: str, peptide) -> None:
    from importlib import import_module

    from ofoldx.runners.collate import collate_mpnn

    module = import_module(f"ofoldx.models.{family}")
    prefix = {"proteinmpnn": "ProteinMpnn", "ligandmpnn": "LigandMpnn", "nampnn": "NaMpnn"}[family]
    processor = getattr(module, prefix + "Processor")()
    chains = (("ALA", "GLY"), ("ALA", "GLY", "ALA"))
    input = collate_mpnn([processor.prepare_input(peptide(*chain)).features for chain in chains])
    options = {"hidden_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1, "num_neighbors": 2, "dropout": 0.0}
    if family == "nampnn":
        options.update(feature_size=8, positional_embedding_size=4, num_rbf=3, include_predicted_na_n=False)
    torch.manual_seed(71)
    model = getattr(module, prefix + "Model")(getattr(module, prefix + "Config")(options)).eval().set_backend(backend)
    output = model(input)
    assert [sample.shape[0] for sample in output.logits] == [2, 3]
    assert not output.logits.concat.isnan().any()
    assert torch.isfinite(output.logits.concat).any(-1).all()
    torch.where(torch.isfinite(output.logits.concat), output.logits.concat, 0.0).square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
