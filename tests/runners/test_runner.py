from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ofoldx.models.proteinmpnn as proteinmpnn
from ofoldx.io import read_structure
from ofoldx.models.registry import MODELS
from ofoldx.runners import Runner

# Minimal backbone-only chain (N, CA, C, O per residue) -- enough for the MPNN processors to lower a
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


def _write(path: Path, text: str = _TRIPEPTIDE) -> Path:
    path.write_text(text, encoding="ascii")
    return path


def _config(root: Path, name: str, source: Path, **model: object) -> dict:
    return {
        "model": {"name": name, **model},
        "data": {"train": {"source": str(source)}},
        "epochs": 1,
        "optim": {"type": "adamw", "lr": 1e-3},
        "dataloader": {"batch_size": 1, "shuffle": False, "drop_last": False},
        "logging": {"enabled": False},
        "workspace": {"root": str(root / "workspace")},
        "ckpt": {"async_mode": "disabled"},
    }


@pytest.mark.parametrize(
    ("name", "model_cls"),
    [("proteinmpnn", "ProteinMpnnModel"), ("ligandmpnn", "LigandMpnnModel"), ("nampnn", "NaMpnnModel")],
)
def test_sequence_family_trains_config_only(tmp_path: Path, name: str, model_cls: str) -> None:
    runner = Runner(_config(tmp_path, name, _write(tmp_path / "chain.pdb")))
    try:
        assert type(runner.model).__name__ == model_cls
        parameter = next(p for p in runner.model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        runner.model.eval()  # as a repository-loaded model would arrive
        runner.train()

        assert runner.train_state.global_step == 1
        assert not torch.equal(parameter.detach(), before)
        assert torch.isfinite(parameter).all()
        assert runner.model.training  # the train/eval mode guard restored train mode
    finally:
        runner.close()


def test_untrainable_architecture_is_rejected(tmp_path: Path) -> None:
    # A structure-prediction model with no `criterion` export is rejected before its model is built.
    with pytest.raises(ValueError, match="not trainable"):
        Runner(
            {
                "model": {"name": "alphafold2"},
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


def test_unsupported_stack_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stack"):
        Runner(
            {
                "model": {"name": "proteinmpnn"},
                "stack": "graph",
                "logging": {"enabled": False},
                "workspace": {"root": str(tmp_path)},
                "ckpt": {"async_mode": "disabled"},
            }
        )


def test_processor_shares_model_config(tmp_path: Path) -> None:
    # `ca_only` is one setting on the Model config; the Runner threads it into the paired Processor.
    runner = Runner(_config(tmp_path, "proteinmpnn", _write(tmp_path / "chain.pdb"), ca_only=True))
    try:
        assert runner.model.config.ca_only is True
        assert runner.processor.config.ca_only is True
    finally:
        runner.close()


def test_collate_emits_danling_batch_with_token_normalizer(tmp_path: Path) -> None:
    # The family collate produces DanLing's {input, target, loss_normalizer}; the normalizer is the
    # designable-token count, so DanLing weights ragged micro-batches by tokens (a global per-token mean).
    processor = MODELS.lookup_model("proteinmpnn").processor()
    batch = proteinmpnn.collate([processor(read_structure(_write(tmp_path / "chain.pdb")))])

    assert set(batch) == {"input", "target", "loss_normalizer"}
    assert batch["loss_normalizer"] == int((batch["input"].residue_mask & batch["input"].design_positions).sum())
