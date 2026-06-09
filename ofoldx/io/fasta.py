r"""
FASTA readers.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FastaRecord:
    r"""
    One FASTA record.

    Attributes:
        name: First whitespace-delimited token after ``>``.
        description: Full header line after ``>``.
        sequence: Concatenated sequence lines.
    """

    name: str
    description: str
    sequence: str


def read_fasta(path: str | PathLike[str]) -> tuple[FastaRecord, ...]:
    r"""
    Read records from a FASTA file.

    Args:
        path: FASTA file path.

    Returns:
        Parsed FASTA records in file order.

    Raises:
        ValueError: If the file contains sequence data before the first header
            or no records.
    """

    return parse_fasta(Path(path).read_text())


def parse_fasta(text: str) -> tuple[FastaRecord, ...]:
    r"""
    Parse FASTA text into records.
    """

    records: list[FastaRecord] = []
    description: str | None = None
    sequence_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if description is not None:
                records.append(_record(description, sequence_lines))
            description = line[1:].strip()
            sequence_lines = []
            continue
        if description is None:
            raise ValueError("FASTA sequence data appeared before the first header")
        sequence_lines.append(line)
    if description is not None:
        records.append(_record(description, sequence_lines))
    if not records:
        raise ValueError("FASTA input contains no records")
    return tuple(records)


def _record(description: str, sequence_lines: list[str]) -> FastaRecord:
    sequence = "".join(sequence_lines)
    name = description.split(maxsplit=1)[0] if description else ""
    if not sequence:
        raise ValueError(f"FASTA record {name!r} has an empty sequence")
    return FastaRecord(name=name, description=description, sequence=sequence)
