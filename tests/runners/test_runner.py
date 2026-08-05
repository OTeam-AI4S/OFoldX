from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from danling.runners.workspace import RunnerWorkspace
from torch.utils.data import DataLoader, Dataset

import ofoldx.runners.dataset as dataset_module
from ofoldx.data import MolecularSystem, Structure
from ofoldx.models.registry import MODELS
from ofoldx.modules.diffusion.loss import resolved_atom_mask
from ofoldx.runners import Runner
from ofoldx.runners.collate import collate_denoising
from ofoldx.runners.config import RunnerConfig

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


class _StructureWorkerDataset(Dataset[Structure]):
    def __init__(self, structure: Structure) -> None:
        self.structure = structure

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Structure:
        if index != 0:
            raise IndexError(index)
        return self.structure


def _collate_structure_graph(items: list[Structure]) -> dict[str, Structure | MolecularSystem]:
    structure = items[0]
    return {"structure": structure, "system": structure.system}


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


def _uninitialized_runner(config: RunnerConfig, log_file: Path) -> Runner:
    r"""Build only the Runner state needed by construction-unit tests."""

    runner = object.__new__(Runner)
    runner.config = config
    runner.workspace = SimpleNamespace(log_file=str(log_file))
    runner._forward_cluster_indices = False
    return runner


def _workspace_runner(root: Path, **overrides: object) -> Runner:
    r"""Build the state needed to exercise workspace selection without constructing a model."""

    values = _config(root, name="proteinmpnn")
    values.update(overrides)
    config = RunnerConfig(values)
    config.boot()
    runner = object.__new__(Runner)
    runner.config = config
    runner.timestamp = "opaque-danling-suffix"
    runner.workspace = RunnerWorkspace(runner)
    return runner


def test_danling_workspace_uses_hash_identity(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    runner = _workspace_runner(
        tmp_path,
        workspace={"root": str(workspace_root), "lineage": "ofoldx"},
    )

    workspace = Path(runner.workspace.dir)
    expected_lineage = "ofoldx" if runner.code_id is None else f"ofoldx-{runner.code_id}"

    assert workspace == workspace_root / expected_lineage / runner.id
    assert workspace.name == runner.id
    assert workspace.parent.name == expected_lineage
    assert workspace.name != runner.timestamp
    assert Path(runner.workspace.log_file) == workspace / "logs" / f"{runner.timestamp}.log"
    assert Path(runner.workspace.checkpoint_dir) == workspace / "checkpoints"


def test_same_identity_reuses_hash_workspace(tmp_path: Path) -> None:
    workspace = {"root": str(tmp_path / "workspace"), "lineage": "ofoldx"}
    first = _workspace_runner(tmp_path, workspace=workspace)
    second = _workspace_runner(tmp_path, workspace=workspace)

    assert first.id == second.id
    assert first.workspace.dir == second.workspace.dir


def test_explicit_workspace_dir_is_exact(tmp_path: Path) -> None:
    explicit = tmp_path / "exact-workspace"
    runner = _workspace_runner(
        tmp_path,
        workspace={"root": str(tmp_path / "unused"), "lineage": "ofoldx", "dir": str(explicit)},
    )

    assert Path(runner.workspace.dir) == explicit


def test_explicit_logging_file_takes_precedence(tmp_path: Path) -> None:
    configured_log = tmp_path / "explicit-training.log"
    values = _config(tmp_path, name="proteinmpnn")
    values["logging"] = {"enabled": False, "file": str(configured_log)}
    config = RunnerConfig(values)
    config.boot()
    runner = object.__new__(Runner)
    runner.config = config
    runner.timestamp = "opaque-danling-suffix"
    runner.workspace = RunnerWorkspace(runner)

    assert runner.workspace.log_file == str(configured_log)


def test_runner_uses_hash_workspace_and_native_log_id(tmp_path: Path) -> None:
    values = _config(tmp_path, name="proteinmpnn")
    values["optim"] = {"type": "torch_adamw", "lr": 1e-3}
    values["logging"] = {"enabled": True}
    values["workspace"] = {"root": str(tmp_path / "workspace"), "lineage": "ofoldx"}
    runner = Runner(values)
    try:
        workspace = Path(runner.workspace.dir)
        log_file = Path(runner.workspace.log_file)
        checkpoint_dir = Path(runner.workspace.checkpoint_dir)
        expected_lineage = "ofoldx" if runner.code_id is None else f"ofoldx-{runner.code_id}"
        assert log_file.is_file()
        assert workspace.name == runner.id
        assert workspace.parent.name == expected_lineage
        assert log_file == workspace / "logs" / f"{runner.timestamp}.log"
        assert checkpoint_dir == workspace / "checkpoints"
    finally:
        runner.close()


def test_dcp_checkpoint_manager_uses_hash_workspace(tmp_path: Path) -> None:
    values = _config(tmp_path, name="proteinmpnn")
    values["optim"] = {"type": "torch_adamw", "lr": 1e-3}
    values["workspace"] = {"root": str(tmp_path / "workspace"), "lineage": "ofoldx"}
    values["ckpt"] = {"enabled": False, "backend": "dcp", "async_mode": "disabled"}
    runner = Runner(values)
    try:
        checkpoint_dir = Path(runner.workspace.checkpoint_dir)
        pointer_store_dir = Path(runner.checkpoint_manager._pointer_store.checkpoint_dir)
        assert pointer_store_dir == checkpoint_dir
        assert checkpoint_dir == Path(runner.workspace.dir) / "checkpoints"
        assert checkpoint_dir.parent.name == runner.id
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("split", "override", "configured", "expected"),
    [
        ("train", None, False, 5),
        ("validation", None, False, None),
        ("train", None, True, None),
        ("train", 3, True, 3),
        ("validation", 2, True, 2),
    ],
)
def test_runner_plumbs_split_processing_error_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    split: str,
    override: int | None,
    configured: bool,
    expected: int | None,
) -> None:
    split_values: dict[str, object] = {"source": str(tmp_path / f"{split}.cif")}
    if configured:
        split_values["max_consecutive_processing_errors"] = override
    values = _config(tmp_path, name="proteinmpnn")
    values["train_splits"] = ["train"]
    values["data"] = {split: split_values}
    config = RunnerConfig(values)
    config.boot()
    captured: dict[str, object] = {}

    class CapturingStructureDataset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(dataset_module, "StructureDataset", CapturingStructureDataset)
    runner = _uninitialized_runner(config, tmp_path / "training.log")

    runner._build_structure_dataset(
        split,
        config.data[split],
        object(),
        ccd_revision=None,
        ccd_snapshot_path=None,
    )

    assert captured["max_consecutive_processing_errors"] == expected
    assert "max_consecutive_processing_errors" not in captured["processor_kwargs"]


@pytest.mark.parametrize(("enabled", "expected"), [(True, "training.log"), (False, None)])
def test_runner_only_enables_dataset_error_sidecars_with_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enabled: bool,
    expected: str | None,
) -> None:
    values = _config(tmp_path, name="proteinmpnn")
    values["logging"] = {"enabled": enabled}
    values["train_splits"] = ["train"]
    values["data"] = {"train": {"source": str(tmp_path / "train.cif")}}
    config = RunnerConfig(values)
    config.boot()
    captured: dict[str, object] = {}

    class CapturingStructureDataset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(dataset_module, "StructureDataset", CapturingStructureDataset)
    log_file = tmp_path / "training.log"
    runner = _uninitialized_runner(config, log_file)

    runner._build_structure_dataset(
        "train",
        config.data.train,
        object(),
        ccd_revision=None,
        ccd_snapshot_path=None,
    )

    expected_path = str(log_file) if expected is not None else None
    assert captured["training_log_file"] == expected_path
    assert "training_log_file" not in captured["processor_kwargs"]


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


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_applies_model_gradient_checkpointing_switch(tmp_path: Path, enabled: bool) -> None:
    config = _config(tmp_path, name="proteinmpnn")
    config["gradient_checkpointing"] = enabled

    runner = Runner(config)
    try:
        assert runner.config.gradient_checkpointing is enabled
        assert runner.model.is_gradient_checkpointing is enabled
    finally:
        runner.close()


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


def test_dataloader_worker_round_trips_frozen_structure_graph(tmp_path: Path) -> None:
    from ofoldx.io import read_structure

    expected = read_structure(_write(tmp_path / "worker-ipc.pdb"))
    loader = DataLoader(
        _StructureWorkerDataset(expected),
        batch_size=1,
        num_workers=1,
        collate_fn=_collate_structure_graph,
    )

    [payload] = list(loader)
    structure = payload["structure"]
    system = payload["system"]

    assert isinstance(structure, Structure)
    assert isinstance(system, MolecularSystem)
    assert structure.system.symbols == expected.system.symbols
    assert torch.equal(structure.coordinates, expected.coordinates)
    assert torch.equal(system.atomic_number, expected.system.atomic_number)
    assert structure.is_locked
    assert structure.system.is_locked
    assert system.is_locked


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
