"""Offline CCD dataset construction using the shared runtime source decoder."""

from __future__ import annotations

from collections.abc import Iterable

from ...io.cif import read_cif
from .reader import CcdRecord, decode_component


def build_rows(components_cif_path: str) -> list[CcdRecord]:
    """Read each CCD block into the resolver record schema, preserving block order."""
    rows: list[CcdRecord] = []
    seen: set[str] = set()
    for block in read_cif(components_cif_path).values():
        row = decode_component(block)
        identifier = row["component_id"]
        if identifier in seen:
            raise ValueError(f"CCD component {identifier!r} occurs in multiple data blocks")
        seen.add(identifier)
        rows.append(row)
    return rows


def write_parquet(rows: Iterable[CcdRecord], output_path: str) -> None:
    """Serialize resolver records; Parquet is an offline dependency of this writer."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(list(rows)), output_path)
