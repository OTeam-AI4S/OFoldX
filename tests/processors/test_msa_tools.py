from __future__ import annotations

import io

import pytest

from ofoldx.processors.msa_tools import (
    Jackhmmer,
    MsaToolError,
    Nhmmer,
    convert_stockholm_to_a3m,
    resolve_binary,
)


def test_resolve_binary_raises_for_missing() -> None:
    with pytest.raises(MsaToolError, match="not found"):
        resolve_binary("definitely_not_a_real_binary_xyz", "jackhmmer")


def test_resolve_binary_returns_existing(tmp_path) -> None:
    binary = tmp_path / "fake_tool"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    assert resolve_binary(str(binary), "faketool") == str(binary)


def test_jackhmmer_flags_missing_binary() -> None:
    # The binary is validated at construction, so an unmet dependency fails loudly and early.
    with pytest.raises(MsaToolError, match="HMMER"):
        Jackhmmer(binary_path="definitely_not_a_real_binary_xyz", database_path="/tmp/db.fasta")


def test_nhmmer_flags_missing_binary() -> None:
    with pytest.raises(MsaToolError, match="HMMER"):
        Nhmmer(binary_path="definitely_not_a_real_binary_xyz", database_path="/tmp/db.fasta")


def test_convert_stockholm_to_a3m_keeps_query_columns() -> None:
    # Query has a gap in column 3; insert-state residues there are lowercased and '.' dropped.
    stockholm = "\n".join(
        [
            "# STOCKHOLM 1.0",
            "query AB-CD",
            "hit1  ABxCD",
            "//",
        ]
    )
    a3m = convert_stockholm_to_a3m(io.StringIO(stockholm))
    lines = a3m.splitlines()
    assert lines[0] == ">query"
    assert lines[1] == "ABCD"  # gapless query
    assert lines[2] == ">hit1"
    assert lines[3] == "ABxCD"[:2] + "x" + "CD"  # insert residue lowercased


def test_convert_stockholm_empty_returns_empty() -> None:
    assert convert_stockholm_to_a3m(io.StringIO("# STOCKHOLM 1.0\n//\n")) == ""


def test_convert_stockholm_respects_max_sequences() -> None:
    stockholm = "\n".join(["# STOCKHOLM 1.0", "query ABCD", "hit1 ABCD", "hit2 ABCD", "//"])
    a3m = convert_stockholm_to_a3m(io.StringIO(stockholm), max_sequences=2)
    names = [line for line in a3m.splitlines() if line.startswith(">")]
    assert len(names) == 2
