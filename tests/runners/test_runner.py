from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx.io import read_structure
from ofoldx.models.boltz1 import objectives as boltz1_objectives
from ofoldx.models.proteinmpnn import objectives as proteinmpnn_objectives
from ofoldx.models.registry import MODELS
from ofoldx.runners import Runner

# A minimal Boltz-1 (single blocks, no confidence head) so a real diffusion training step runs on CPU;
# the architecture, not its size, is what these lifecycle tests exercise.
_MINIMAL_BOLTZ1: dict = {
    "recycling_steps": 1,
    "trunk": {"num_pairformer_blocks": 1, "num_msa_blocks": 1},
    "denoiser": {
        "num_token_encoder_blocks": 1,
        "num_atom_encoder_blocks": 1,
        "num_atom_decoder_blocks": 1,
    },
    "confidence": {"enabled": False},
}

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
        "ckpt": {"enabled": False},
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


def test_diffusion_family_trains_config_only(tmp_path: Path) -> None:
    # A diffusion family lowers each structure into (input, target) via `process_training`, noises the
    # target inside its own forward, and grounds the EDM denoising loss against the model output.
    runner = Runner(_config(tmp_path, "boltz1", _write(tmp_path / "chain.pdb"), **_MINIMAL_BOLTZ1))
    try:
        assert type(runner.model).__name__ == "Boltz1Model"
        assert runner.model.config.trunk.num_pairformer_blocks == 1  # the nested override threaded through
        parameter = next(p for p in runner.model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        runner.model.eval()
        runner.train()

        assert runner.train_state.global_step == 1
        assert not torch.equal(parameter.detach(), before)
        assert torch.isfinite(parameter).all()
        assert runner.model.training
    finally:
        runner.close()


def test_untrainable_architecture_is_rejected(tmp_path: Path) -> None:
    # ODesign is a diffusion family left ungrounded on purpose: its output surfaces input-frame
    # coordinates but not the augmented training target, so it exports no `objectives.criterion`. It is
    # rejected before the model is built or any weights are downloaded.
    with pytest.raises(ValueError, match="not trainable"):
        Runner(
            {
                "model": {"name": "odesign"},
                "logging": {"enabled": False},
                "workspace": {"root": str(tmp_path)},
                "ckpt": {"enabled": False},
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
    batch = proteinmpnn_objectives.collate([processor(read_structure(_write(tmp_path / "chain.pdb")))])

    assert set(batch) == {"input", "target", "loss_normalizer"}
    assert batch["loss_normalizer"] == int((batch["input"].residue_mask & batch["input"].design_positions).sum())


def test_denoising_collate_emits_input_target_pair(tmp_path: Path) -> None:
    # The diffusion collate carries the (input, target) pair as the `input` mapping DanLing splats as
    # model(input=..., target=...), the same pair as the criterion `target`, and the atom-count normalizer.
    processor = MODELS.lookup_model("boltz1").processor()
    item = processor.process_training(read_structure(_write(tmp_path / "chain.pdb")))
    batch = boltz1_objectives.collate([item])

    assert set(batch) == {"input", "target", "loss_normalizer"}
    assert set(batch["input"]) == {"input", "target"}
    assert batch["input"] is batch["target"]
    assert batch["loss_normalizer"] == int(batch["input"]["input"].atom_mask.sum())
