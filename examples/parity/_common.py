"""Shared helpers for local conversion/parity checks."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
from torch import Tensor

CKPT_DIR = Path("/tmp/ofoldx-ckpts")
UPSTREAM_DIR = Path("/tmp/ofoldx-upstream")


def load_pickle_checkpoint(path: Path, *, mmap: bool = False) -> object:
    """Load a Lightning/PyTorch pickle that references missing upstream classes."""

    import pickle

    class _Dummy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            if kwargs:
                self.__dict__.update(kwargs)

        def __setstate__(self, state: object) -> None:
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _TolerantUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                return _Dummy

    class _tolerant_pickle:
        Unpickler = _TolerantUnpickler
        load = pickle.load
        loads = pickle.loads

    kwargs: dict = {
        "map_location": "cpu",
        "weights_only": False,
        "pickle_module": _tolerant_pickle,
    }
    if mmap:
        kwargs["mmap"] = True
    return torch.load(path, **kwargs)


def hf_download(repo_id: str, filename: str, dest_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir = CKPT_DIR if dest_dir is None else dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id,
            filename,
            local_dir=str(dest_dir),
        )
    )


def report_convert(label: str, model: object) -> None:
    state = model.state_dict()  # type: ignore[attr-defined]
    n_tensors = sum(1 for _ in state)
    n_params = sum(value.numel() for value in state.values() if hasattr(value, "numel"))
    print(f"{label}: tensors={n_tensors} params={n_params} strict_load=ok", flush=True)


def clone_source(url: str, dest: Path, marker: str) -> Path:
    if (dest / marker).exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return dest


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def report(label: str, actual: Tensor, expected: Tensor, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    diff = (actual - expected).abs()
    print(
        f"{label}: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}",
        flush=True,
    )


def unique_backbone(batch: int, length: int, atoms: int = 4) -> Tensor:
    coords = torch.randn(batch, length, atoms, 3)
    return coords + torch.arange(length).view(1, length, 1, 1)
