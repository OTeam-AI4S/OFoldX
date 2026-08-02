from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import MolecularSystem, Molecule
from ofoldx.data.ccd import resolver as resolver_module
from ofoldx.data.ccd.build import write_parquet
from ofoldx.data.ccd.resolver import resolve_ccd_component


def _row(component_id: str = "LIG") -> dict[str, object]:
    return {
        "component_id": component_id,
        "chem_type": "non-polymer",
        "atom_ids": ["C1", "O1"],
        "elements": ["C", "O"],
        "formal_charges": [0, 0],
        "aromatic": [False, False],
        "stereo": [None, None],
        "bond_atom_a": ["C1"],
        "bond_atom_b": ["O1"],
        "bond_orders": [1.0],
        "bond_aromatic": [False],
        "bond_stereo": [None],
        "reference_coordinates": [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        "previous_attachment_atom": None,
        "previous_leaving_atoms": [],
        "next_attachment_atom": None,
        "next_leaving_atoms": [],
    }


@pytest.fixture(autouse=True)
def _clear_resolver_cache() -> None:
    resolver_module._resolve_cached.cache_clear()
    resolver_module._resolve_ccd_snapshot.cache_clear()


def _snapshot(path: Path, *rows: dict[str, object]) -> Path:
    path.mkdir()
    write_parquet(rows, str(path / "components.parquet"))
    return path


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"revision": "2024-06-08", "snapshot_path": "/offline/ccd"},
    ),
)
def test_resolver_requires_exactly_one_locator(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_ccd_component("LIG", **kwargs)


def test_local_snapshot_resolves_offline_without_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _snapshot(tmp_path / "ccd", _row())

    def unexpected(*, revision: str | None) -> Path:
        raise AssertionError("local CCD resolution must not call the Hub")

    monkeypatch.setattr(resolver_module, "_resolve_ccd_snapshot", unexpected)
    component = resolve_ccd_component("LIG", snapshot_path=str(snapshot))
    assert component.component_id == "LIG"
    assert [atom.atom_id for atom in component.atoms] == ["C1", "O1"]
    assert component.reference_coordinates == ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0))


def test_snapshot_element_symbols_are_canonicalized(tmp_path: Path) -> None:
    row = _row()
    row["elements"] = ["NA", "CL"]
    snapshot = _snapshot(tmp_path / "ccd", row)

    component = resolve_ccd_component("LIG", snapshot_path=str(snapshot))

    assert [atom.element for atom in component.atoms] == ["Na", "Cl"]


def test_snapshot_hydrogen_isotopes_are_preserved_and_materialized(tmp_path: Path) -> None:
    row = _row()
    row["elements"] = ["d", "t"]
    snapshot = _snapshot(tmp_path / "ccd", row)

    component = resolve_ccd_component("LIG", snapshot_path=str(snapshot))

    assert [atom.element for atom in component.atoms] == ["D", "T"]
    assert component.without_hydrogens().atoms == ()

    molecule = Molecule((component,))
    system = MolecularSystem.from_entities({"ligand": molecule}, {"A": "ligand"})
    assert system.atomic_number.tolist() == [1, 1]
    assert system.isotope.tolist() == [2, 3]
    assert system.isotope_known.tolist() == [True, True]


@pytest.mark.parametrize("element", ("", "?", "Xx", "Q", "DU"))
def test_snapshot_rejects_unknown_element_symbols(tmp_path: Path, element: str) -> None:
    row = _row()
    row["elements"] = [element, "O"]
    snapshot = _snapshot(tmp_path / "ccd", row)

    with pytest.raises(ValueError, match="invalid element"):
        resolve_ccd_component("LIG", snapshot_path=str(snapshot))


def test_remote_revision_remains_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _snapshot(tmp_path / "ccd", _row())
    revisions: list[str | None] = []

    def resolve_snapshot(*, revision: str | None) -> Path:
        revisions.append(revision)
        return snapshot

    monkeypatch.setattr(resolver_module, "_resolve_ccd_snapshot", resolve_snapshot)
    assert resolve_ccd_component("LIG", revision="2024-06-08").component_id == "LIG"
    assert revisions == ["2024-06-08"]


def test_remote_snapshot_download_is_cached_across_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path / "ccd", _row("ONE"), _row("TWO"))
    calls = 0

    def snapshot_download(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        assert kwargs["revision"] == "2024-06-08"
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert resolve_ccd_component("ONE", revision="2024-06-08").component_id == "ONE"
    assert resolve_ccd_component("TWO", revision="2024-06-08").component_id == "TWO"
    assert calls == 1


def test_local_snapshot_path_must_exist_and_contain_parquet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_ccd_component("LIG", snapshot_path=str(tmp_path / "missing"))

    file_path = tmp_path / "file"
    file_path.write_text("not a snapshot")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        resolve_ccd_component("LIG", snapshot_path=str(file_path))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no Parquet shards"):
        resolve_ccd_component("LIG", snapshot_path=str(empty))


def test_unknown_and_duplicate_components_fail(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "ccd", _row())
    with pytest.raises(KeyError, match="MISSING"):
        resolve_ccd_component("MISSING", snapshot_path=str(snapshot))

    write_parquet([_row()], str(snapshot / "duplicate.parquet"))
    resolver_module._resolve_cached.cache_clear()
    with pytest.raises(ValueError, match="occurs in 2 rows"):
        resolve_ccd_component("LIG", snapshot_path=str(snapshot))


def test_canonical_local_path_and_component_are_process_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path / "ccd", _row())
    original = resolver_module._read_component_row
    calls = 0

    def read(snapshot_path: Path, component_id: str) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        return original(snapshot_path, component_id)

    monkeypatch.setattr(resolver_module, "_read_component_row", read)
    first = resolve_ccd_component("LIG", snapshot_path=str(snapshot))
    second = resolve_ccd_component("LIG", snapshot_path=str(snapshot / "."))
    assert first is second
    assert calls == 1
