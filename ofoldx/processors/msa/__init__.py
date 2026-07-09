r"""
Multiple-sequence alignment: generation and encoding.

Two complementary halves. [`search`][ofoldx.processors.msa.search] *generates* an alignment: HMMER
wrappers (``Jackhmmer``/``Nhmmer``/``Hmmbuild``/``Hmmsearch``/``Hmmalign``) that align a query sequence
against genetic databases into a3m. [`encode`][ofoldx.processors.msa.encode] *consumes* an alignment:
parse a precomputed a3m/CSV file into per-chain token rows and encode it into the ``msa`` feature tensors
the Boltz trunks read.

Examples:
    >>> from ofoldx.processors.msa import parse_a3m, encode_msa  # doctest: +SKIP
"""

from __future__ import annotations

from .encode import ParsedMsa, attach_msa, encode_msa, parse_a3m, parse_csv, parse_msa
from .search import (
    Hmmalign,
    Hmmbuild,
    Hmmsearch,
    Jackhmmer,
    MsaTool,
    MsaToolError,
    MsaToolResult,
    Nhmmer,
    convert_stockholm_to_a3m,
    resolve_binary,
    run_command,
    run_msa_tools,
    write_query_fasta,
)

__all__ = [
    "attach_msa",
    "convert_stockholm_to_a3m",
    "encode_msa",
    "Hmmalign",
    "Hmmbuild",
    "Hmmsearch",
    "Jackhmmer",
    "MsaTool",
    "MsaToolError",
    "MsaToolResult",
    "Nhmmer",
    "parse_a3m",
    "parse_csv",
    "parse_msa",
    "ParsedMsa",
    "resolve_binary",
    "run_command",
    "run_msa_tools",
    "write_query_fasta",
]
