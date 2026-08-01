r"""
FASTA, PDB, mmCIF, and MSA I/O.
"""

from __future__ import annotations

from multimolecule.io import read_fasta_records

from .structure import read_structure, write_structure

__all__ = [
    "read_fasta_records",
    "read_structure",
    "write_structure",
]
