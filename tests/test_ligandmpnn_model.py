from __future__ import annotations

import pytest
import torch

from ofoldx.models.contract import SequenceDesignCandidate
from ofoldx.models.ligandmpnn import LigandMPNNConfig, LigandMPNNPacker, LigandMPNNPackerConfig
from ofoldx.models.ligandmpnn.model import (
    LigandMPNNDesigner,
    LigandMPNNInput,
    LigandMPNNRepresentations,
)


def _tiny_packer() -> LigandMPNNPacker:
    return LigandMPNNPacker(
        LigandMPNNPackerConfig(
            {
                "hidden_size": 16,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "num_rbf": 4,
                "k_neighbors": 3,
                "atom_context_num": 2,
                "num_positional_embeddings": 4,
            }
        )
    )


def _packer_features(batch: int = 1, length: int = 6, atoms: int = 2, *, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "X": torch.randn(batch, length, 4, 3, generator=generator),
        "S": torch.randint(0, 20, (batch, length), generator=generator),
        "mask": torch.ones(batch, length),
        "R_idx": torch.arange(length)[None].expand(batch, length).contiguous(),
        "chain_labels": torch.zeros(batch, length, dtype=torch.long),
        "Y": torch.randn(batch, length, atoms, 3, generator=generator),
        "Y_t": torch.full((batch, length, atoms), 6.0),
        "Y_m": torch.ones(batch, length, atoms),
    }


def _tiny_model() -> LigandMPNNDesigner:
    return LigandMPNNDesigner(
        LigandMPNNConfig(
            {
                "hidden_size": 16,
                "node_size": 16,
                "edge_size": 16,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "k_neighbors": 3,
                "atom_context_num": 2,
            }
        )
    )


def _tiny_batch(batch: int = 1, length: int = 6, atoms: int = 2, *, seed: int = 0) -> LigandMPNNInput:
    generator = torch.Generator().manual_seed(seed)
    return LigandMPNNInput(
        backbone_coords=torch.randn(batch, length, 4, 3, generator=generator),
        sequence_tokens=torch.randint(0, 20, (batch, length), generator=generator),
        token_mask=torch.ones(batch, length),
        residue_index=torch.arange(length)[None].expand(batch, length).contiguous(),
        chain_index=torch.zeros(batch, length, dtype=torch.long),
        ligand_atom_coords=torch.randn(batch, length, atoms, 3, generator=generator),
        ligand_atom_types=torch.full((batch, length, atoms), 6),  # carbon
        ligand_atom_mask=torch.ones(batch, length, atoms),
    )


def test_forward_is_training_path_returns_grad_logits() -> None:
    model = _tiny_model().train()
    batch = _tiny_batch()
    output = model(batch)
    assert isinstance(output, SequenceDesignCandidate)
    assert output.metadata is not None
    assert output.metadata["mode"] == "training"
    assert output.sequence_logits is not None
    assert output.sequence_logits.shape == (1, 6, 21)
    assert output.sequence_logits.requires_grad


def test_forward_gradients_flow_and_training_reduces_loss() -> None:
    # eval() -> deterministic (zeros) decode order, so the before/after comparison is like-for-like.
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = _tiny_batch()
    target = batch.sequence_tokens

    def nll() -> torch.Tensor:
        logits = model(batch).sequence_logits
        return torch.nn.functional.cross_entropy(logits.reshape(-1, 21), target.reshape(-1))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_before = nll()
    loss_before.backward()
    # Gradients reach the encoder, the decoder, and the output projection.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed through the training forward"
    assert any(g.abs().sum() > 0 for g in grads)
    for _ in range(5):
        optimizer.zero_grad()
        nll().backward()
        optimizer.step()
    loss_after = nll()
    assert loss_after.item() < loss_before.item()


def test_encode_returns_typed_carrier_representations() -> None:
    model = _tiny_model().eval()
    reps = model.encode(_tiny_batch())
    assert isinstance(reps, LigandMPNNRepresentations)
    assert reps.single.tensor.shape == (1, 6, 16)
    assert reps.edges.shape[:2] == (1, 6)
    assert reps.neighbor_idx.shape[:2] == (1, 6)


def test_design_verb_samples_sequences() -> None:
    model = _tiny_model().eval()
    generator = torch.Generator().manual_seed(1)
    output = model.design(_tiny_batch(), num_samples=2, generator=generator)
    assert isinstance(output, SequenceDesignCandidate)
    assert output.sequence_tokens is not None
    assert output.sequence_tokens.shape == (2, 6)
    assert output.scores is not None
    assert output.scores["sampling_probs"].shape == (2, 6, 20)


def test_score_and_single_residue_score_verbs() -> None:
    model = _tiny_model().eval()
    batch = _tiny_batch()
    scored = model.score(batch)
    single = model.single_residue_score(batch)
    assert scored.sequence_logits.shape == (1, 6, 21)
    assert single.sequence_logits.shape == (1, 6, 21)
    assert "sequence_log_likelihood" in scored.scores


def test_public_sample_is_demoted() -> None:
    model = _tiny_model()
    assert not hasattr(model, "sample")
    assert hasattr(model, "_sample")


def test_keyword_tensor_call_path_builds_input() -> None:
    model = _tiny_model().train()
    batch = _tiny_batch()
    output = model(
        backbone_coords=batch.backbone_coords,
        sequence_tokens=batch.sequence_tokens,
        token_mask=batch.token_mask,
        residue_index=batch.residue_index,
        chain_index=batch.chain_index,
        ligand_atom_coords=batch.ligand_atom_coords,
        ligand_atom_types=batch.ligand_atom_types,
        ligand_atom_mask=batch.ligand_atom_mask,
    )
    assert isinstance(output, SequenceDesignCandidate)
    assert output.sequence_logits is not None
    assert output.sequence_logits.shape == (1, 6, 21)


def test_packer_forward_is_grad_correct_training_path() -> None:
    packer = _tiny_packer().train()
    features = _packer_features()
    mean, concentration, mix_logits = packer(dict(features))
    assert mean.shape[:2] == (1, 6)
    assert mean.requires_grad
    (mean.square().mean() + concentration.mean() + mix_logits.square().mean()).backward()
    assert any(p.grad is not None for p in packer.parameters())


def test_packer_pack_verb_denoises_side_chains() -> None:
    packer = _tiny_packer().eval()
    packed = packer.pack(_packer_features(), num_denoising_steps=1, num_samples=2)
    assert packed["X"].shape == (1, 6, 14, 3)
    assert "b_factors" in packed


def test_forward_backward_compiles_fullgraph() -> None:
    try:
        torch._dynamo.reset()
    except Exception as exc:  # pragma: no cover - environment probe
        pytest.skip(f"torch dynamo unavailable: {exc}")
    if not torch._dynamo.is_dynamo_supported():
        pytest.skip("torch dynamo is not supported on this interpreter")

    model = _tiny_model().train()
    batch = _tiny_batch()
    compiled = torch.compile(model, fullgraph=True)
    output = compiled(batch)
    loss = output.sequence_logits.square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
