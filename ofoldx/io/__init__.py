r"""Canonical structure I/O."""

from __future__ import annotations

from .msa import read_msa
from .pdb import read_pdb
from .structure import LossReport, read_structure, read_structures, write_structure
