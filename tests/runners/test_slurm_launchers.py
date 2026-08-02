from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


RUNNERS_DIR = Path(__file__).parents[2] / "ofoldx" / "runners"
LAUNCHERS = ("submit_train.slurm", "submit_train_sqsh.slurm")


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _launcher_environment(
    tmp_path: Path,
    launcher_name: str,
    *,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    socket_family: str | None = None,
) -> tuple[dict[str, str], Path, Path]:
    repo_root = tmp_path / "repo"
    runner_dir = repo_root / "ofoldx" / "runners"
    runner_dir.mkdir(parents=True)
    (runner_dir / "train.py").write_text("", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_args = tmp_path / "child.args"
    capture_family = tmp_path / "child.family"
    capturing_child = r"""#!/bin/bash
printf '%s\0' "$@" > "${CAPTURE_ARGS}"
printf '%s' "${NCCL_SOCKET_FAMILY-}" > "${CAPTURE_FAMILY}"
"""
    fake_python = fake_bin / "python"
    _write_executable(fake_python, capturing_child)
    _write_executable(fake_bin / "srun", capturing_child)
    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
printf '%s\n' node0
""",
    )

    sqsh_image = tmp_path / "image.sqsh"
    sqsh_image.write_bytes(b"not-a-real-image")

    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REPO_ROOT": str(repo_root),
        "PYTHON": str(fake_python),
        "SQSH_IMAGE": str(sqsh_image),
        "CONTAINER_PYTHON": "/opt/test/bin/python",
        "CAPTURE_ARGS": str(capture_args),
        "CAPTURE_FAMILY": str(capture_family),
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "SLURM_NNODES": str(nnodes),
        "SLURM_CPUS_PER_TASK": "8",
        "NPROC_PER_NODE": str(nproc_per_node),
    }
    if socket_family is not None:
        env["NCCL_SOCKET_FAMILY"] = socket_family
    return env, capture_args, capture_family


def _run_launcher(
    tmp_path: Path,
    launcher_name: str,
    *,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    socket_family: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    env, capture_args, capture_family = _launcher_environment(
        tmp_path,
        launcher_name,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        socket_family=socket_family,
    )
    result = subprocess.run(
        ["bash", str(RUNNERS_DIR / launcher_name), "train.cif", "--print-config"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    return result, capture_args, capture_family


def _captured_arguments(path: Path) -> list[str]:
    return [part.decode() for part in path.read_bytes().split(b"\0") if part]


@pytest.mark.parametrize("launcher_name", LAUNCHERS)
@pytest.mark.parametrize(
    ("configured_family", "expected_family"),
    [(None, "AF_INET"), ("", "AF_INET"), ("AF_INET6", "AF_INET6")],
)
def test_nccl_socket_family_default_and_override_reach_child_process(
    tmp_path: Path,
    launcher_name: str,
    configured_family: str | None,
    expected_family: str,
) -> None:
    result, capture_args, capture_family = _run_launcher(
        tmp_path,
        launcher_name,
        socket_family=configured_family,
    )

    assert result.returncode == 0, result.stderr
    assert capture_args.exists()
    assert capture_family.read_text(encoding="utf-8") == expected_family
    if launcher_name == "submit_train_sqsh.slurm":
        container_env = next(
            argument for argument in _captured_arguments(capture_args) if argument.startswith("--container-env=")
        )
        assert "NCCL_SOCKET_FAMILY" in container_env.removeprefix("--container-env=").split(",")


@pytest.mark.parametrize("launcher_name", LAUNCHERS)
def test_invalid_nccl_socket_family_is_rejected_before_child_launch(
    tmp_path: Path,
    launcher_name: str,
) -> None:
    result, capture_args, capture_family = _run_launcher(
        tmp_path,
        launcher_name,
        socket_family="AF_UNSPEC",
    )

    assert result.returncode == 2
    assert "NCCL_SOCKET_FAMILY must be AF_INET or AF_INET6: AF_UNSPEC" in result.stderr
    assert not capture_args.exists()
    assert not capture_family.exists()


def test_native_multinode_srun_inherits_nccl_socket_family(tmp_path: Path) -> None:
    result, capture_args, capture_family = _run_launcher(
        tmp_path,
        "submit_train.slurm",
        nnodes=2,
        nproc_per_node=2,
        socket_family="AF_INET6",
    )

    assert result.returncode == 0, result.stderr
    arguments = _captured_arguments(capture_args)
    assert capture_family.read_text(encoding="utf-8") == "AF_INET6"
    assert arguments[:7] == [
        "--nodes",
        "2",
        "--ntasks",
        "2",
        "--ntasks-per-node=1",
        "--kill-on-bad-exit=1",
        str(tmp_path / "bin" / "python"),
    ]
    assert "torch.distributed.run" in arguments
    assert arguments[arguments.index("--nnodes") + 1] == "2"
    assert arguments[arguments.index("--nproc-per-node") + 1] == "2"
    assert arguments[arguments.index("--rdzv-endpoint") + 1] == "node0:10123"


@pytest.mark.parametrize(
    ("nnodes", "nproc_per_node", "expected_arguments", "unexpected_arguments"),
    [
        (1, 1, ["-m", "ofoldx.runners.train"], ["torch.distributed.run", "--standalone"]),
        (1, 2, ["torch.distributed.run", "--standalone", "--nproc-per-node", "2"], ["--nnodes"]),
        (
            2,
            2,
            [
                "--nodes",
                "2",
                "--ntasks",
                "2",
                "torch.distributed.run",
                "--nnodes",
                "2",
                "--nproc-per-node",
                "2",
                "--rdzv-endpoint",
                "node0:10123",
            ],
            ["--standalone"],
        ),
    ],
)
def test_sqsh_launcher_builds_pyxis_argv_for_each_launch_branch(
    tmp_path: Path,
    nnodes: int,
    nproc_per_node: int,
    expected_arguments: list[str],
    unexpected_arguments: list[str],
) -> None:
    result, capture_args, capture_family = _run_launcher(
        tmp_path,
        "submit_train_sqsh.slurm",
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        socket_family="AF_INET6",
    )

    assert result.returncode == 0, result.stderr
    arguments = _captured_arguments(capture_args)
    assert capture_family.read_text(encoding="utf-8") == "AF_INET6"
    assert all(expected in arguments for expected in expected_arguments)
    assert all(unexpected not in arguments for unexpected in unexpected_arguments)
    container_env = next(argument for argument in arguments if argument.startswith("--container-env="))
    assert "NCCL_SOCKET_FAMILY" in container_env.removeprefix("--container-env=").split(",")
