from __future__ import annotations

import torch

from ofoldx.models.protenix.model import (
    ProtenixEvaluationOutput,
    ProtenixRawOutput,
    _protenix_contact_probability,
    protenix_contact_probability,
)


_EVALUATION_TENSOR_FIELDS = (
    "coordinates",
    "distogram_logits",
    "plddt_logits",
    "pae_logits",
    "pde_logits",
    "resolved_logits",
    "contact_probs",
    "per_sample_contact_probs",
)


def _evaluation_output() -> ProtenixEvaluationOutput:
    batch_size, sample_count, atom_count, token_count = 2, 3, 5, 4
    return ProtenixEvaluationOutput(
        coordinates=torch.randn(batch_size, sample_count, atom_count, 3),
        distogram_logits=torch.randn(batch_size, token_count, token_count, 7),
        plddt_logits=torch.randn(batch_size, sample_count, atom_count, 4),
        pae_logits=torch.randn(batch_size, sample_count, token_count, token_count, 6),
        pde_logits=torch.randn(batch_size, sample_count, token_count, token_count, 6),
        resolved_logits=torch.randn(batch_size, sample_count, atom_count, 2),
        contact_probs=torch.randn(batch_size, token_count, token_count),
        per_sample_contact_probs=torch.randn(batch_size, sample_count, token_count, token_count),
        batch_size=[batch_size],
    )


def test_evaluation_output_keeps_shared_logits_and_explicit_contact_axes() -> None:
    batch_size, sample_count, atom_count, token_count = 2, 3, 5, 4
    distogram_bin_count = 7
    distogram_logits = torch.randn(batch_size, token_count, token_count, distogram_bin_count)
    contact_probs = protenix_contact_probability(distogram_logits)
    per_sample_contact_probs = torch.stack(
        (
            contact_probs,
            contact_probs.transpose(-1, -2),
            contact_probs.flip((-1, -2)),
        ),
        dim=1,
    )

    output = ProtenixEvaluationOutput(
        coordinates=torch.randn(batch_size, sample_count, atom_count, 3),
        distogram_logits=distogram_logits,
        plddt_logits=torch.randn(batch_size, sample_count, atom_count, 4),
        pae_logits=torch.randn(batch_size, sample_count, token_count, token_count, 6),
        pde_logits=torch.randn(batch_size, sample_count, token_count, token_count, 6),
        resolved_logits=torch.randn(batch_size, sample_count, atom_count, 2),
        contact_probs=contact_probs,
        per_sample_contact_probs=per_sample_contact_probs,
        batch_size=[batch_size],
    )

    assert isinstance(output, ProtenixRawOutput)
    assert torch.equal(output.distogram_logits, distogram_logits)
    assert output.distogram_logits.shape == (2, 4, 4, 7)
    assert output.contact_probs.shape == (2, 4, 4)
    assert output.per_sample_contact_probs.shape == (2, 3, 4, 4)
    assert output.distogram_logits.numel() == batch_size * token_count * token_count * distogram_bin_count
    assert output.per_sample_contact_probs.numel() == batch_size * sample_count * token_count * token_count


def test_evaluation_output_to_retains_subtype_and_moves_every_tensor_field() -> None:
    output = _evaluation_output()

    moved = output.to(device=torch.device("meta"), dtype=torch.float64)

    assert isinstance(moved, ProtenixEvaluationOutput)
    assert isinstance(moved, ProtenixRawOutput)
    assert moved.batch_size == output.batch_size == torch.Size([2])
    for field in _EVALUATION_TENSOR_FIELDS:
        original = getattr(output, field)
        converted = getattr(moved, field)
        assert converted.shape == original.shape
        assert converted.device.type == "meta"
        assert converted.dtype == torch.float64


def test_evaluation_output_batch_index_retains_subtype_fields_shapes_and_device() -> None:
    output = _evaluation_output()

    indexed = output[1]

    assert isinstance(indexed, ProtenixEvaluationOutput)
    assert isinstance(indexed, ProtenixRawOutput)
    assert indexed.batch_size == torch.Size([])
    for field in _EVALUATION_TENSOR_FIELDS:
        original = getattr(output, field)
        selected = getattr(indexed, field)
        assert selected.shape == original.shape[1:]
        assert selected.device == original.device
        assert torch.equal(selected, original[1])


def test_public_contact_probability_retains_protenix_decoding_behavior() -> None:
    logits = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [4.0, 3.0, 2.0, 1.0],
            ],
            [
                [-1.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float64,
    )
    expected = logits.softmax(dim=-1)[..., :1].sum(dim=-1)

    actual = protenix_contact_probability(logits)

    assert actual.shape == (2, 2)
    assert actual.dtype == logits.dtype
    assert actual.device == logits.device
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(_protenix_contact_probability(logits), actual)
