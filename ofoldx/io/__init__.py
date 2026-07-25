r"""
FASTA, PDB, mmCIF, and MSA input functions.
"""

from __future__ import annotations

from multimolecule.io import read_fasta_records

from .msa import read_msa
from .records import AtomRecord, ParsedStructure
from .structure import LossReport, read_structure, read_structures, write_structure
