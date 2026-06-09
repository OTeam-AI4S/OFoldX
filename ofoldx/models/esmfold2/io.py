r"""
ESMFold2 sequence input readers.
"""

from __future__ import annotations

import string
from os import PathLike
from pathlib import Path

from ...io import FastaRecord, parse_fasta, read_fasta

_FASTA_SUFFIXES = {".fa", ".faa", ".fasta", ".fna", ".ffn", ".frn"}
_PROTEIN_SEQUENCE_ALPHABET = set(string.ascii_uppercase)


def sequence_from_input(sequence_or_path: str | PathLike[str]) -> str:
    r"""
    Return one normalized protein sequence from a raw sequence string or FASTA file.

    Args:
        sequence_or_path: Raw one-letter protein sequence, pasted FASTA text, or a
            path to a FASTA file.

    Returns:
        Uppercase one-letter protein sequence.

    Raises:
        ValueError: If a filesystem path is not FASTA-like or the sequence is empty/invalid.

    Examples:
        >>> sequence_from_input(">x\nACD")
        'ACD'
        >>> sequence_from_input("AC\\nDX")
        'ACDX'
    """

    raw = str(sequence_or_path)
    path = Path(raw)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if not is_file:
        return normalize_protein_sequence(raw)
    if path.suffix.lower() in _FASTA_SUFFIXES or _first_nonempty_line(path).startswith(">"):
        records = read_fasta(path)
        return _single_sequence(records, source=str(path))
    raise ValueError(f"ESMFold2 sequence input path is not FASTA-like: {path}")


def normalize_protein_sequence(text: str) -> str:
    r"""
    Normalize raw protein sequence or FASTA text into one uppercase sequence.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(line.startswith(">") for line in lines):
        sequence = _single_sequence(parse_fasta(text), source="FASTA text")
    else:
        sequence = "".join(lines)
    sequence = sequence.upper().replace(" ", "")
    if not sequence:
        raise ValueError("ESMFold2 protein sequence must be non-empty")
    invalid = sorted({character for character in sequence if character not in _PROTEIN_SEQUENCE_ALPHABET})
    if invalid:
        raise ValueError(f"ESMFold2 protein sequence contains unsupported characters: {''.join(invalid)!r}")
    return sequence


def _single_sequence(records: tuple[FastaRecord, ...], *, source: str) -> str:
    if len(records) != 1:
        raise ValueError(f"ESMFold2 expects exactly one FASTA record in {source}, got {len(records)}")
    return records[0].sequence


def _first_nonempty_line(path: Path) -> str:
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    return ""
