r"""
FASTA, PDB, mmCIF, and MSA input functions.
"""

from __future__ import annotations

from multimolecule.io import read_fasta_records

from .msa import pair_msa_rows_by_key, read_msa, read_msa_with_headers
from .records import AtomRecord, ParsedStructure
from .structure import LossReport, read_structure, read_structures, write_structure
