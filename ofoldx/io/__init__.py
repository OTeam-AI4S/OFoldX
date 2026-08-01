r"""
FASTA, PDB, mmCIF, and MSA I/O.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multimolecule.io import read_fasta_records

    from .cif import CifDocument, parse_cif, read_cif
    from .coordinates import read_polymer_coordinates
    from .msa import (
        IndexedMsaLoader,
        decode_msa_collection,
        encode_msa_collection,
        pair_msas,
        read_msa,
        read_msa_with_headers,
    )
    from .source import (
        CifSource,
        PdbSource,
        StructureSource,
        StructureView,
        UnsupportedFormatError,
        parse_source,
        read_source,
    )
    from .structure import read_structure, write_structure
    from .template import read_template_coordinates, read_template_structure, read_template_structures

__all__ = [
    "CifDocument",
    "CifSource",
    "PdbSource",
    "parse_source",
    "StructureSource",
    "StructureView",
    "UnsupportedFormatError",
    "parse_cif",
    "read_cif",
    "read_source",
    "IndexedMsaLoader",
    "decode_msa_collection",
    "encode_msa_collection",
    "pair_msas",
    "read_fasta_records",
    "read_msa",
    "read_msa_with_headers",
    "read_structure",
    "read_template_coordinates",
    "read_polymer_coordinates",
    "read_template_structure",
    "read_template_structures",
    "write_structure",
]


_EXPORTS = {
    "read_fasta_records": ("multimolecule.io", "read_fasta_records"),
    "CifDocument": (".cif", "CifDocument"),
    "parse_cif": (".cif", "parse_cif"),
    "read_cif": (".cif", "read_cif"),
    "IndexedMsaLoader": (".msa", "IndexedMsaLoader"),
    "decode_msa_collection": (".msa", "decode_msa_collection"),
    "encode_msa_collection": (".msa", "encode_msa_collection"),
    "pair_msas": (".msa", "pair_msas"),
    "read_msa": (".msa", "read_msa"),
    "read_msa_with_headers": (".msa", "read_msa_with_headers"),
    "CifSource": (".source", "CifSource"),
    "PdbSource": (".source", "PdbSource"),
    "parse_source": (".source", "parse_source"),
    "StructureSource": (".source", "StructureSource"),
    "StructureView": (".source", "StructureView"),
    "UnsupportedFormatError": (".source", "UnsupportedFormatError"),
    "read_source": (".source", "read_source"),
    "read_structure": (".structure", "read_structure"),
    "write_structure": (".structure", "write_structure"),
    "read_template_coordinates": (".template", "read_template_coordinates"),
    "read_polymer_coordinates": (".coordinates", "read_polymer_coordinates"),
    "read_template_structure": (".template", "read_template_structure"),
    "read_template_structures": (".template", "read_template_structures"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
