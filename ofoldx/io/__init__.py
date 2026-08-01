r"""
FASTA, PDB, mmCIF, and MSA I/O.
"""

from __future__ import annotations

from multimolecule.io import read_fasta_records

from .msa import (
    IndexedMsaLoader,
    decode_msa_collection,
    encode_msa_collection,
    pair_msas,
    read_msa,
    read_msa_with_headers,
)
from .structure import read_structure, write_structure

__all__ = [
    "IndexedMsaLoader",
    "decode_msa_collection",
    "encode_msa_collection",
    "pair_msas",
    "read_fasta_records",
    "read_msa",
    "read_msa_with_headers",
    "read_structure",
    "write_structure",
]
