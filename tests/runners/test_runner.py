from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ofoldx.models.registry import MODELS
from ofoldx.modules.diffusion.loss import resolved_atom_mask
from ofoldx.runners import Runner
from ofoldx.runners.collate import collate_denoising

# Minimal backbone-only chain -- enough for the MPNN + structure Processors to lower a real Input, and
# portable (no external data dependency).
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

# A minimal Boltz-1 (single blocks, no confidence head) so a real diffusion training step runs on CPU;
# the architecture, not its size, is what these lifecycle tests exercise.
_MINIMAL_BOLTZ1: dict = {
    "recycling_steps": 1,
    "trunk": {"num_pairformer_blocks": 1, "num_msa_blocks": 1},
    "denoiser": {"num_token_encoder_blocks": 1, "num_atom_encoder_blocks": 1, "num_atom_decoder_blocks": 1},
    "confidence": {"enabled": False},
}


def _write(path: Path, text: str = _TRIPEPTIDE) -> Path:
    path.write_text(text, encoding="ascii")
    return path


def _config(root: Path, source: Path | None = None, **model: object) -> dict:
    config: dict = {
        "model": dict(model),
        "epochs": 1,
        "optim": {"type": "adamw", "lr": 1e-3},
        "dataloader": {"batch_size": 1, "shuffle": False, "drop_last": False},
        "logging": {"enabled": False},
        "workspace": {"root": str(root / "workspace")},
        "ckpt": {"enabled": False},
    }
    if source is not None:
        config["data"] = {"train": {"source": str(source)}}
    return config


@pytest.mark.parametrize(
    ("name", "model_cls"),
    [("proteinmpnn", "ProteinMpnnModel"), ("ligandmpnn", "LigandMpnnModel"), ("nampnn", "NaMpnnModel")],
)
def test_sequence_family_trains_config_only(tmp_path: Path, name: str, model_cls: str) -> None:
    runner = Runner(_config(tmp_path, _write(tmp_path / "chain.pdb"), name=name))
    try:
        assert type(runner.model).__name__ == model_cls
        parameter = next(p for p in runner.model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        runner.train()

        assert runner.train_state.global_step == 1
        assert not torch.equal(parameter.detach(), before)
        assert torch.isfinite(parameter).all()
    finally:
        runner.close()


def test_diffusion_family_trains_config_only(tmp_path: Path) -> None:
    # A diffusion family lowers each structure into (input, target) via `process_training`, noises the
    # target inside its own forward, and grounds the EDM denoising loss (resolved-atom mask, config
    # sigma_data) against the model output.
    runner = Runner(_config(tmp_path, _write(tmp_path / "chain.pdb"), name="boltz1", **_MINIMAL_BOLTZ1))
    try:
        assert type(runner.model).__name__ == "Boltz1Model"
        parameter = next(p for p in runner.model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        runner.model.eval()
        runner.train()

        assert runner.train_state.global_step == 1
        assert not torch.equal(parameter.detach(), before)
        assert torch.isfinite(parameter).all()
    finally:
        runner.close()


# Runner objectives bind to exact architecture IDs rather than being inferred from
# a model family.
def test_architecture_without_runner_objective_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="boltz2_affinity.*has no built-in Runner objective"):
        Runner(_config(tmp_path, name="boltz2_affinity"))


def test_selecting_no_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model.pretrained|model.name"):
        Runner(_config(tmp_path))


def test_unsupported_stack_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, _write(tmp_path / "chain.pdb"), name="proteinmpnn")
    config["stack"] = "graph"
    with pytest.raises(ValueError, match="stack"):
        Runner(config)


def test_denoising_collate_normalizer_is_the_resolved_atom_count(tmp_path: Path) -> None:
    # The batch's loss_normalizer must equal the count the criterion averages over (resolved atoms), so
    # DanLing's normalizer-weighted backward forms a true per-atom mean rather than over-weighting large
    # structures.
    processor = MODELS.lookup_model("boltz1").processor()
    from ofoldx.io import read_structure

    item = processor.process_training(read_structure(_write(tmp_path / "chain.pdb")))
    batch = collate_denoising([item])
    model_input, model_target = batch["input"]["input"], batch["input"]["target"]
    assert batch["loss_normalizer"] == int(resolved_atom_mask(model_target, model_input).sum())


def test_infer_step_returns_dataclass_output(tmp_path: Path) -> None:
    # DanLing's default infer_step calls pred.squeeze(-1) and AttributeErrors on the family dataclass
    # output; the override runs the family inference entry point and returns the output(s).
    runner = Runner(_config(tmp_path, _write(tmp_path / "chain.pdb"), name="proteinmpnn"))
    try:
        # The sequence families collate to a ragged NestedTensor batch (train and infer alike); infer_step
        # forwards it and un-nests one output per structure.
        batch = runner.collate_fn([runner.datasets["train"][0]])
        output = runner.infer_step(batch)
        assert isinstance(output, list) and len(output) == 1
        assert type(output[0]).__name__ == "ProteinMpnnOutput"
    finally:
        runner.close()


def test_repository_model_loads_trains_and_rejects_processor_overrides(tmp_path: Path) -> None:
    from ofoldx.models.proteinmpnn import ProteinMpnnPipeline

    repository = tmp_path / "repo"
    registration = MODELS.lookup_model("proteinmpnn")
    model = registration.model(registration.model_config())
    processor = registration.processor(registration.processor.config_class())
    ProteinMpnnPipeline(model=model, processor=processor).save(str(repository))

    source = _write(tmp_path / "chain.pdb")
    runner = Runner(_config(tmp_path, source, pretrained=str(repository)))
    try:
        assert type(runner.model).__name__ == "ProteinMpnnModel"
        parameter = next(p for p in runner.model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        runner.model.eval()
        runner.train()
        assert runner.train_state.global_step == 1
        assert not torch.equal(parameter.detach(), before)
    finally:
        runner.close()

    # A loaded Processor's config is fixed to the published one; overrides are rejected, not ignored.
    config = _config(tmp_path, source, pretrained=str(repository))
    config["processor"] = {"ca_only": True}
    with pytest.raises(ValueError, match="published config"):
        Runner(config)

    # The same holds for the Model config: a repository model's config is published, so model overrides
    # are rejected rather than silently ignored.
    with pytest.raises(ValueError, match="published config"):
        Runner(_config(tmp_path, source, pretrained=str(repository), ca_only=True))


@pytest.mark.skipif(not hasattr(torch.distributed, "is_gloo_available"), reason="torch.distributed unavailable")
def test_diffusion_survives_ddp_second_step_via_find_unused_parameters(tmp_path: Path) -> None:
    # The single-step test cannot see it: under DDP an un-gradiented parameter crashes on the SECOND step.
    # The Runner sets find_unused_parameters=True, which locates the unused parameters by walking the
    # structured forward return. So unused modules (here confidence + distogram + sampling-only) need no freezing.
    # Build with confidence ENABLED so those parameters are genuinely present + unused.
    import torch.distributed as dist
    from tensordict import is_tensorclass

    from ofoldx.io import read_structure
    from ofoldx.models.boltz1 import Boltz1Config, Boltz1Model, Boltz1Processor
    from ofoldx.modules.diffusion.loss import edm_denoising_loss

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29529")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        config = Boltz1Config(recycling_steps=1)
        config.trunk.num_pairformer_blocks = 1
        config.trunk.num_msa_blocks = 1
        config.denoiser.num_token_encoder_blocks = 1
        config.denoiser.num_atom_encoder_blocks = 1
        config.denoiser.num_atom_decoder_blocks = 1
        config.confidence.enabled = True
        config.confidence.num_pairformer_blocks = 1
        model = Boltz1Model(config)
        model.train()
        # Exactly what DanLing's materialize_model does with the Runner's ddp.find_unused_parameters=True.
        ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model_input, model_target = Boltz1Processor().process_training(read_structure(_write(tmp_path / "chain.pdb")))
        sigma_data = float(model.config.denoiser.sigma_data)

        for _ in range(2):  # the second step is the one that would raise on an unreduced parameter
            optimizer.zero_grad()
            prediction = ddp(input=model_input, target=model_target)
            assert is_tensorclass(prediction)  # the property that lets find_unused_parameters walk the output
            resolved = resolved_atom_mask(model_target, model_input)
            edm_denoising_loss(prediction, resolved, sigma_data=sigma_data).backward()
            optimizer.step()
    finally:
        dist.destroy_process_group()
