"""Shared access to the official BoltzGen checkout and checkpoints."""

from __future__ import annotations

import os
import pickle as _pickle
import sys
from pathlib import Path
from typing import Any

import torch

REFERENCE_SRC = Path(os.environ.get("OFOLDX_REFERENCE_BOLTZGEN", "~/reference/boltzgen/src")).expanduser()


def configure_reference() -> None:
    """Put the real BoltzGen source tree on the import path."""

    if not REFERENCE_SRC.is_dir():
        raise FileNotFoundError(f"BoltzGen reference source not found: {REFERENCE_SRC}")
    path = str(REFERENCE_SRC.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _make_validation_stub(module: str, name: str) -> type:
    """Return a harmless placeholder class for an unused validation object.

    Every released BoltzGen Lightning checkpoint embeds a single
    ``boltzgen.model.validation.*`` validator object (``DesignValidator`` /
    ``RCSBValidator``) inside ``hyper_parameters``. Unpickling it eagerly imports
    ``boltzgen.model.validation.validator``, which pulls in ``pytorch_lightning``
    (plus ``pydssp``/``matplotlib``) at module load. The parity suite only ever
    reads ``checkpoint["state_dict"]``, so we reconstruct the validator as an
    inert stub instead of importing that (optional) dependency chain.
    """

    class _ValidationStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass

        def __setstate__(self, state: Any) -> None:
            if isinstance(state, dict):
                self.__dict__.update(state)

    _ValidationStub.__name__ = name
    _ValidationStub.__qualname__ = name
    _ValidationStub.__module__ = module
    return _ValidationStub


class _CheckpointUnpickler(_pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("boltzgen.model.validation"):
            return _make_validation_stub(module, name)
        return super().find_class(module, name)


class _CheckpointPickleModule:
    """``pickle_module`` for ``torch.load`` that stubs unused validation classes."""

    Unpickler = _CheckpointUnpickler
    Pickler = _pickle.Pickler
    load = staticmethod(_pickle.load)
    loads = staticmethod(_pickle.loads)
    dump = staticmethod(_pickle.dump)
    dumps = staticmethod(_pickle.dumps)
    HIGHEST_PROTOCOL = _pickle.HIGHEST_PROTOCOL


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint with the official BoltzGen classes available.

    The embedded validation object is stubbed (see ``_make_validation_stub``) so
    the load does not require ``pytorch_lightning``; all parameter tensors load
    byte-for-byte identically.
    """

    configure_reference()
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_CheckpointPickleModule,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected a checkpoint mapping, got {type(checkpoint).__name__}")
    return checkpoint
