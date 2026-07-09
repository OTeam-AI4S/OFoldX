r"""
NA-MPNN model-side pass: house verb surface, typed carriers, grad-correct training forward, and a
``torch.compile(fullgraph=True)`` forward+backward smoke test.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from ofoldx.models.contract import SequenceDesignCandidate
from ofoldx.models.nampnn import NAMPNNConfig, NAMPNNDesigner, NAMPNNInput, NAMPNNRepresentations
from ofoldx.modules.representation import Single


def _tiny_config() -> NAMPNNConfig:
    return NAMPNNConfig(
        {
            "hidden_size": 16,
            "node_feature_size": 16,
            "edge_feature_size": 16,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "k_neighbors": 4,
            "num_positional_embeddings": 8,
            "num_rbf": 4,
        }
    )


def _tiny_batch(batch_size: int = 2, length: int = 6, *, noise: torch.Tensor | None = None) -> NAMPNNInput:
    r"""A protein-only NA-MPNN batch with a fixed decoding-order noise so the training forward is deterministic."""

    return NAMPNNInput(
        atom_coords=torch.randn(batch_size, length, 16, 3),
        atom_mask=torch.ones(batch_size, length, 16),
        sequence_tokens=torch.randint(0, 20, (batch_size, length)),
        token_mask=torch.ones(batch_size, length),
        residue_index=torch.arange(length).expand(batch_size, length).clone(),
        chain_index=torch.zeros(batch_size, length, dtype=torch.long),
        polymer_type_ids=torch.zeros(batch_size, length, dtype=torch.long),
        protein_mask=torch.ones(batch_size, length),
        dna_mask=torch.zeros(batch_size, length),
        rna_mask=torch.zeros(batch_size, length),
        decoding_order_noise=noise,
    )


def _dynamo_supported() -> bool:
    torch._dynamo.reset()
    try:

        @torch.compile(fullgraph=True)
        def _probe(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        _probe(torch.zeros(1))
    except Exception:  # pragma: no cover - environment dependent (e.g. Python 3.13+)
        return False
    return True


def test_verb_surface_is_forward_plus_design() -> None:
    # forward is the training path; design is the capability verb; the public sampler was demoted to _sample.
    model = NAMPNNDesigner(_tiny_config())
    assert callable(model.forward)
    assert callable(model.design)
    assert callable(model._sample)
    assert not hasattr(model, "sample")
    assert not hasattr(model, "predict")
    assert not hasattr(model, "generate")


def test_input_class_and_keyword_call_path() -> None:
    model = NAMPNNDesigner(_tiny_config()).eval()
    assert model.input_class is NAMPNNInput
    batch = _tiny_batch(noise=torch.randn(2, 6))
    fields = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch) if getattr(batch, f.name) is not None}
    positional = model(batch)
    keyword = model(**fields)
    assert isinstance(positional, SequenceDesignCandidate)
    assert isinstance(keyword, SequenceDesignCandidate)
    torch.testing.assert_close(positional.sequence_logits, keyword.sequence_logits)


def test_input_is_pytree_registered() -> None:
    batch = _tiny_batch(noise=torch.randn(2, 6))
    leaves, spec = tree_flatten(batch)
    # Tensor fields become pytree leaves; scalar/None fields (temperature, num_samples, ...) ride along as
    # constant leaves, which dynamo handles without a graph break.
    assert any(isinstance(leaf, torch.Tensor) for leaf in leaves)
    rebuilt = tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, NAMPNNInput)
    torch.testing.assert_close(rebuilt.atom_coords, batch.atom_coords)
    torch.testing.assert_close(rebuilt.sequence_tokens, batch.sequence_tokens)


def test_representations_carrier_flattens_to_tensor_leaves() -> None:
    model = NAMPNNDesigner(_tiny_config()).eval()
    features = model._training_feature_dict(_tiny_batch(noise=torch.randn(2, 6)))
    reps = model._representations(features)
    assert isinstance(reps, NAMPNNRepresentations)
    assert isinstance(reps.single, Single)
    leaves, spec = tree_flatten(reps)
    # Single unwraps to its raw tensor child, so the carrier boundary introduces no graph break.
    assert [type(leaf).__name__ for leaf in leaves] == ["Tensor", "Tensor", "Tensor"]
    rebuilt = tree_unflatten(leaves, spec)
    assert isinstance(rebuilt.single, Single)


def test_forward_returns_teacher_forced_output() -> None:
    model = NAMPNNDesigner(_tiny_config()).eval()
    batch = _tiny_batch(noise=torch.randn(2, 6))
    output = model(batch)
    assert output.sequence_logits.shape == (2, 6, model.config.vocab_size)
    assert output.scores["log_probs"].shape == (2, 6, model.config.vocab_size)
    assert output.scores["sequence_log_likelihood"].shape == (2,)
    assert output.metadata["mode"] == "training"
    torch.testing.assert_close(output.sequence_tokens, batch.sequence_tokens.long())


def test_forward_is_grad_correct_and_one_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = NAMPNNDesigner(_tiny_config()).eval()
    batch = _tiny_batch(batch_size=2, length=6, noise=torch.randn(2, 6))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_before = -model(batch).scores["sequence_log_likelihood"].mean()
    assert torch.isfinite(loss_before)

    optimizer.zero_grad()
    loss_before.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    optimizer.step()

    loss_after = -model(batch).scores["sequence_log_likelihood"].mean()
    assert float(loss_after) < float(loss_before)


def test_design_and_score_inference_verbs_run() -> None:
    model = NAMPNNDesigner(_tiny_config()).eval()
    single_backbone = _tiny_batch(batch_size=1, length=6)
    scored = model.score(single_backbone)
    assert scored.scores["log_probs"].shape == (1, 6, model.config.vocab_size)
    designed = model.design(single_backbone, num_samples=3, temperature=0.2)
    assert designed.sequence_tokens.shape == (3, 6)


@pytest.mark.skipif(not _dynamo_supported(), reason="torch.compile/dynamo unsupported in this environment")
def test_forward_backward_compiles_fullgraph_with_no_graph_breaks() -> None:  # pragma: no cover - needs dynamo
    torch._dynamo.reset()
    model = NAMPNNDesigner(_tiny_config()).eval()
    batch = _tiny_batch(batch_size=2, length=6, noise=torch.randn(2, 6))

    # explain() raises/records on any graph break; fullgraph=True below also hard-fails on a break.
    explanation = torch._dynamo.explain(model)(batch)
    assert explanation.graph_break_count == 0

    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=True)
    output = compiled(batch)
    loss = -output.scores["sequence_log_likelihood"].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
