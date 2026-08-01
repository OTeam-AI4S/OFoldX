from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ofoldx.data.ccd import resolve_ccd_component


def _snapshot(path: Path, *, charge: int = 0) -> Path:
    path.mkdir()
    row = {
        "component_id": "LIG",
        "chem_type": "non-polymer",
        "atom_ids": ["C1", "O1"],
        "elements": ["C", "O"],
        "formal_charges": [charge, 0],
        "aromatic": [False, False],
        "stereo": [None, None],
        "bond_atom_a": ["C1"],
        "bond_atom_b": ["O1"],
        "bond_orders": [1.0],
        "bond_aromatic": [False],
        "bond_stereo": [None],
        "reference_coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "leaving_atom_ids": ["O1"],
        "non_leaving_atom_ids": ["C1"],
        "previous_attachment_atom": None,
        "previous_leaving_atoms": [],
        "next_attachment_atom": None,
        "next_leaving_atoms": [],
    }
    pq.write_table(pa.Table.from_pylist([row]), path / "components.parquet")
    return path


class TestResolveCcdComponent:
    def test_local_snapshot(self, tmp_path: Path) -> None:
        component = resolve_ccd_component("LIG", snapshot_path=_snapshot(tmp_path / "ccd"))

        assert component.atoms[0].formal_charge == 0
        assert component.leaving_atom_ids == frozenset({"O1"})
        assert component.non_leaving_atom_ids == frozenset({"C1"})

    def test_relative_snapshot(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        _snapshot(first / "ccd")
        _snapshot(second / "ccd", charge=1)

        monkeypatch.chdir(first)
        neutral = resolve_ccd_component("LIG", snapshot_path="ccd")
        monkeypatch.chdir(second)
        charged = resolve_ccd_component("LIG", snapshot_path="ccd")

        assert neutral.atoms[0].formal_charge == 0
        assert charged.atoms[0].formal_charge == 1

    def test_requires_one_locator(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            resolve_ccd_component("LIG", revision="revision", snapshot_path=tmp_path)

    def test_rejects_empty_snapshot(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no Parquet shards"):
            resolve_ccd_component("LIG", snapshot_path=tmp_path)

    def test_rejects_malformed_row(self, tmp_path: Path) -> None:
        pq.write_table(pa.Table.from_pylist([{"component_id": "LIG"}]), tmp_path / "components.parquet")

        with pytest.raises(ValueError, match="missing required field"):
            resolve_ccd_component("LIG", snapshot_path=tmp_path)
