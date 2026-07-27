from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx.io import read_structure
from ofoldx.models.registry import MODELS
from ofoldx.runners import Runner
from ofoldx.runners.collate import collate_proteinmpnn
from ofoldx.runners.families import build_fresh

# Minimal backbone-only chains (N, CA, C, O per residue) -- enough for the MPNN processors to lower a
# real Input, and portable (no external data dependency).
_TRIPEPTIDE = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.332   1.550   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.970   2.850   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.480   2.700   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.000   1.580   0.000  1.00  0.00           O
ATOM      9  N   SER A   3       6.150   3.830   0.000  1.00  0.00           N
ATOM     10  CA  SER A   3       7.600   3.900   0.000  1.00  0.00           C
ATOM     11  C   SER A   3       8.100   5.330   0.000  1.00  0.00           C
ATOM     12  O   SER A   3       7.350   6.300   0.000  1.00  0.00           O
END
"""
_DIPEPTIDE = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.332   1.550   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.970   2.850   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.480   2.700   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.000   1.580   0.000  1.00  0.00           O
END
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="ascii")
    return path


def _config(root: Path, name: str, source: Path) -> dict:
    return {
        "model": {"name": name},
        "data": {"train": {"source": str(source)}},
        "epochs": 1,
        "optim": {"type": "adamw", "lr": 1e-3},
        "dataloader": {"batch_size": 1, "shuffle": False, "drop_last": False},
        "logging": {"enabled": False},
        "workspace": {"root": str(root / "workspace")},
        "ckpt": {"async_mode": "disabled"},
    }


@pytest.mark.parametrize(
    ("name", "model_cls", "objective_cls"),
    [
        ("proteinmpnn", "ProteinMpnnModel", "ProteinMpnnObjective"),
        ("nampnn", "NaMpnnModel", "NaMpnnObjective"),
    ],
)
def test_config_only_family_trains(tmp_path: Path, name: str, model_cls: str, objective_cls: str) -> None:
    source = _write(tmp_path / "chain.pdb", _TRIPEPTIDE)
    runner = Runner(_config(tmp_path, name, source))
    try:
        assert type(runner.model).__name__ == model_cls
        assert type(runner.objective).__name__ == objective_cls
        # The paired processor produced the family's own Input type.
        assert type(runner.processor) is MODELS.lookup_model(name).processor

        parameter = next(p for p in runner.model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        runner.train()

        assert runner.train_state.global_step == 1
        # A real backward + optimizer step ran: the parameter moved and stayed finite (a NaN loss
        # would have propagated into the weights).
        assert not torch.equal(parameter.detach(), before)
        assert torch.isfinite(parameter).all()
    finally:
        runner.close()


def test_untrainable_architecture_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not trainable"):
        Runner(
            {
                "model": {"name": "boltz1"},
                "logging": {"enabled": False},
                "workspace": {"root": str(tmp_path)},
                "ckpt": {"async_mode": "disabled"},
            }
        )


def test_selecting_no_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model.pretrained|model.name"):
        Runner(
            {
                "logging": {"enabled": False},
                "workspace": {"root": str(tmp_path)},
                "ckpt": {"async_mode": "disabled"},
            }
        )


def test_model_and_processor_share_config() -> None:
    registration = MODELS.lookup_model("proteinmpnn")
    model, processor, _objective, _collate = build_fresh(registration, {"ca_only": True})
    assert model.config.ca_only is True
    assert processor.config.ca_only is True


def test_collate_pads_tied_group_ids_to_own_index(tmp_path: Path) -> None:
    processor = MODELS.lookup_model("proteinmpnn").processor()
    short = processor(read_structure(_write(tmp_path / "di.pdb", _DIPEPTIDE)))
    long = processor(read_structure(_write(tmp_path / "tri.pdb", _TRIPEPTIDE)))

    batch = collate_proteinmpnn([short, long])
    length = long.sequence_tokens.shape[1]

    assert batch.sequence_tokens.shape == (2, length)
    # The shorter chain's padded rows are gated out and become their own throwaway causal units
    # (their collated row index), never merging into real unit 0.
    padded = short.sequence_tokens.shape[1]
    assert not batch.residue_mask[0, padded:].any()
    assert torch.equal(batch.tied_group_ids[0, padded:], torch.arange(padded, length))
