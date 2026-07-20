"""Shared path helpers for parity tooling."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the current OFoldX repository root."""

    override = os.environ.get("OFOLDX_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def log_dir() -> Path:
    """Return the parity log directory."""

    return Path(os.environ.get("OFOLDX_PARITY_LOGDIR", str(repo_root() / "artifacts/parity/slurm"))).expanduser()


def weights_root() -> Path:
    """Return the shared checkpoint root."""

    return Path(os.environ.get("OFOLDX_WEIGHTS_ROOT", "/data/users/z/weights")).expanduser()


def reference_root(name: str) -> Path:
    """Return a reference checkout path by short family name."""

    env_name = "OFOLDX_REFERENCE_" + name.upper().replace("-", "_")
    default = Path("/data/users/z/reference") / name
    return Path(os.environ.get(env_name, str(default))).expanduser()


def repo_path(*parts: str) -> Path:
    """Return a path inside the current repository."""

    return repo_root().joinpath(*parts)


def weight_path(*parts: str) -> Path:
    """Return a checkpoint path under the shared weights root."""

    return weights_root().joinpath(*parts)


def boltzgen_checkpoint(filename: str) -> Path:
    """Return a BoltzGen checkpoint path inside the local model asset directory."""

    return weight_path("boltzgen-1", filename)
