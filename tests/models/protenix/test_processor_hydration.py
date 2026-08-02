from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx.data import (
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    Structure,
)
from ofoldx.models.protenix import processor as processor_module
from ofoldx.models.protenix.config import ProtenixProcessorConfig
from ofoldx.models.protenix.processor import ProtenixProcessor, _hydrate_structure, _hydrate_system


def _component(
    component_id: str,
    *,
    reference: bool,
    extra_atom: bool = False,
    hydrogen: bool = False,
    topology_complete: bool = True,
) -> ChemicalComponent:
    atoms = [ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")]
    bonds = {ChemicalBond("C1", "O1")}
    if extra_atom:
        atoms.append(ChemicalAtom("N1", "N"))
        bonds.add(ChemicalBond("C1", "N1"))
    if hydrogen:
        atoms.append(ChemicalAtom("H1", "H"))
        bonds.add(ChemicalBond("C1", "H1"))
    coordinates = tuple((float(index), 1.0, 2.0) for index in range(len(atoms))) if reference else ()
    return ChemicalComponent(
        component_id,
        tuple(atoms),
        frozenset(bonds) if topology_complete else None,
        reference_coordinates=coordinates,
    )


def _structure(*components: ChemicalComponent) -> Structure:
    system = MolecularSystem.from_entities(
        {"ligand": Molecule(components)},
        {"L": "ligand"},
    )
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    coordinates = torch.arange(atom_count * 3, dtype=torch.float32).reshape(atom_count, 3)
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.arange(atom_count, dtype=torch.float32) + 0.5,
        occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
        b_factor=torch.arange(atom_count, dtype=torch.float32) + 10.0,
        b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
        source_symbols=tuple(str(index) for index in range(component_count)),
        residue_id=torch.arange(component_count),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _processor() -> ProtenixProcessor:
    return ProtenixProcessor(ProtenixProcessorConfig(augment_reference_positions=False))


def test_reference_less_ligand_hydrates_observed_atom_induced_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _structure(_component("LIG", reference=False, topology_complete=False))
    resolved = _component("LIG", reference=True, extra_atom=True)
    calls: list[tuple[str, str]] = []

    def resolve(
        component_id: str,
        *,
        revision: str | None,
        snapshot_path: str | None,
    ) -> ChemicalComponent:
        assert snapshot_path is None
        calls.append((component_id, revision))
        return resolved

    monkeypatch.setattr(processor_module, "resolve_ccd_component", resolve)
    hydrated = _hydrate_structure(observed, revision="2026-07-23", snapshot_path=None)

    assert calls == [("LIG", "2026-07-23")]
    assert hydrated.system.atom_symbol.numel() == 2
    assert hydrated.system.reference_coordinate_mask.tolist() == [True, True]
    assert torch.equal(hydrated.coordinates, observed.coordinates)
    assert hydrated.coordinate_mask.tolist() == [True, True]
    assert hydrated.occupancy.tolist() == [0.5, 1.5]
    assert hydrated.b_factor.tolist() == [10.0, 11.0]
    assert hydrated.system.edge_index.shape[1] == 1


def test_explicit_reference_does_not_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    system = _structure(_component("LIG", reference=True)).system

    def unexpected(*args: object, **kwargs: object) -> ChemicalComponent:
        raise AssertionError("explicit references must not resolve CCD")

    monkeypatch.setattr(processor_module, "resolve_ccd_component", unexpected)
    hydrated, atom_map = _hydrate_system(system, revision=None, snapshot_path=None)
    assert hydrated is system
    assert atom_map.tolist() == [0, 1]


def test_repeated_component_id_resolves_once_per_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    system = _structure(
        _component("LIG", reference=False),
        _component("LIG", reference=False),
    ).system
    calls = 0

    def resolve(
        component_id: str,
        *,
        revision: str | None,
        snapshot_path: str | None,
    ) -> ChemicalComponent:
        nonlocal calls
        calls += 1
        assert component_id == "LIG"
        assert revision == "2026-07-23"
        assert snapshot_path is None
        return _component("LIG", reference=True, extra_atom=True)

    monkeypatch.setattr(processor_module, "resolve_ccd_component", resolve)
    hydrated, _ = _hydrate_system(system, revision="2026-07-23", snapshot_path=None)
    assert calls == 1
    assert hydrated.component_atom_ptr.tolist() == [0, 2, 4]


def test_resolver_miss_keeps_observed_topology_with_zero_reference_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _structure(_component("MISSING", reference=False))
    observed.system.formal_charge[:] = 2
    observed.system.formal_charge_known[:] = True

    def fail(
        component_id: str,
        *,
        revision: str | None,
        snapshot_path: str | None,
    ) -> ChemicalComponent:
        raise KeyError((component_id, revision, snapshot_path))

    monkeypatch.setattr(processor_module, "resolve_ccd_component", fail)
    hydrated, atom_map = _hydrate_system(observed.system, revision="2026-07-23", snapshot_path=None)
    assert hydrated is observed.system
    assert atom_map.tolist() == [0, 1]

    processor = ProtenixProcessor(
        ProtenixProcessorConfig(augment_reference_positions=False, ccd_revision="2026-07-23")
    )
    model_input, _ = processor.process_training(observed)
    assert not bool(model_input.reference_mask.any())
    assert torch.equal(model_input.reference_positions, torch.zeros_like(model_input.reference_positions))
    assert torch.equal(model_input.reference_charges, torch.zeros_like(model_input.reference_charges))


def test_training_input_and_target_share_hydrated_atom_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _structure(_component("LIG", reference=False, topology_complete=False))
    calls = 0

    def resolve(
        component_id: str,
        *,
        revision: str | None,
        snapshot_path: str | None,
    ) -> ChemicalComponent:
        nonlocal calls
        calls += 1
        assert revision == "test-revision"
        assert snapshot_path is None
        return _component(component_id, reference=True, extra_atom=True)

    monkeypatch.setattr(processor_module, "resolve_ccd_component", resolve)
    processor = ProtenixProcessor(
        ProtenixProcessorConfig(augment_reference_positions=False, ccd_revision="test-revision")
    )
    model_input, target = processor.process_training(observed)

    assert calls == 1
    assert model_input.atom_index.tolist() == [[0, 1]]
    assert target.coordinates.shape[-2] == 2
    assert target.coordinate_mask.tolist() == [[True, True]]
    assert torch.equal(target.coordinates[0, model_input.atom_index[0]], target.coordinates[0])


def test_resolved_ccd_is_heavy_only_across_system_input_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _structure(_component("LIG", reference=False, topology_complete=False))
    monkeypatch.setattr(
        processor_module,
        "resolve_ccd_component",
        lambda component_id, *, revision, snapshot_path: _component(
            component_id,
            reference=True,
            hydrogen=True,
        ),
    )

    hydrated = _hydrate_structure(observed, revision="test-revision", snapshot_path=None)
    model_input, target = ProtenixProcessor(
        ProtenixProcessorConfig(augment_reference_positions=False, ccd_revision="test-revision")
    ).process_training(observed)

    assert hydrated.system.atomic_number.tolist() == [6, 8]
    assert hydrated.system.edge_index.tolist() == [[0], [1]]
    assert hydrated.system.reference_coordinate_mask.tolist() == [True, True]
    assert torch.equal(hydrated.coordinates, observed.coordinates)
    assert model_input.atom_index.tolist() == [[0, 1]]
    assert model_input.reference_mask.tolist() == [[True, True]]
    assert target.coordinates.shape[-2] == 2
    assert target.coordinate_mask.tolist() == [[True, True]]


def test_polymer_hydration_adds_missing_heavy_atoms(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_component = ChemicalComponent(
        "ALA",
        (ChemicalAtom("N", "N"), ChemicalAtom("CA", "C")),
        None,
    )
    resolved_component = ChemicalComponent(
        "ALA",
        (
            ChemicalAtom("N", "N"),
            ChemicalAtom("CA", "C"),
            ChemicalAtom("C", "C"),
            ChemicalAtom("O", "O"),
            ChemicalAtom("H", "H"),
        ),
        frozenset(
            {
                ChemicalBond("N", "CA"),
                ChemicalBond("CA", "C"),
                ChemicalBond("C", "O"),
                ChemicalBond("N", "H"),
            }
        ),
        reference_coordinates=tuple((float(index), 0.0, 0.0) for index in range(5)),
    )
    system = MolecularSystem.from_entities(
        {"protein": PolymerSequence("protein", (observed_component,))},
        {"A": "protein"},
    )
    atom_count = system.atom_symbol.numel()
    observed = Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=torch.zeros((atom_count, 3)),
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.ones(atom_count),
        occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
        source_symbols=("1",),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([-1]),
        batch_size=[],
    )
    monkeypatch.setattr(
        processor_module,
        "resolve_ccd_component",
        lambda component_id, *, revision, snapshot_path: resolved_component,
    )

    hydrated = _hydrate_structure(observed, revision="test-revision", snapshot_path=None)

    assert hydrated.system.atomic_number.tolist() == [7, 6, 6, 8]
    assert hydrated.coordinate_mask.tolist() == [True, True, False, False]
    assert hydrated.system.reference_coordinate_mask.tolist() == [True, True, True, True]


def test_hydrated_component_without_ideal_reference_uses_zero_reference_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _structure(_component("LIG", reference=False))
    monkeypatch.setattr(
        processor_module,
        "resolve_ccd_component",
        lambda component_id, *, revision, snapshot_path: _component(
            component_id,
            reference=False,
            extra_atom=True,
        ),
    )
    model_input, _ = ProtenixProcessor(
        ProtenixProcessorConfig(augment_reference_positions=False, ccd_revision="test-revision")
    ).process_training(observed)
    assert not bool(model_input.reference_mask.any())
    assert torch.equal(model_input.reference_positions, torch.zeros_like(model_input.reference_positions))
    assert torch.equal(model_input.reference_charges, torch.zeros_like(model_input.reference_charges))


def test_reference_less_component_without_locator_uses_zero_reference_features() -> None:
    observed = _structure(_component("LIG", reference=False))
    model_input, _ = _processor().process_training(observed)
    assert not bool(model_input.reference_mask.any())
    assert torch.equal(model_input.reference_positions, torch.zeros_like(model_input.reference_positions))
    assert torch.equal(model_input.reference_charges, torch.zeros_like(model_input.reference_charges))


def test_no_locator_hydrates_from_builtin_component(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _structure(_component("LIG", reference=False, topology_complete=False))
    calls = 0

    def builtin(component_id: str) -> ChemicalComponent:
        nonlocal calls
        calls += 1
        return _component(component_id, reference=True, extra_atom=True)

    monkeypatch.setattr(processor_module, "get_component", builtin)
    monkeypatch.setattr(
        processor_module,
        "resolve_ccd_component",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no remote CCD lookup expected")),
    )

    hydrated, _ = _hydrate_system(observed.system, revision=None, snapshot_path=None)

    assert calls == 1
    assert hydrated.component_topology_complete.tolist() == [True]
    assert hydrated.edge_index.shape[1] == 1
    assert hydrated.reference_coordinate_mask.tolist() == [True, True]


def test_no_locator_hydrates_reference_less_complete_topology_from_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _structure(_component("LIG", reference=False))
    monkeypatch.setattr(
        processor_module,
        "get_component",
        lambda component_id: _component(component_id, reference=True),
    )

    hydrated, _ = _hydrate_system(observed.system, revision=None, snapshot_path=None)

    assert hydrated.component_topology_complete.tolist() == [True]
    assert hydrated.reference_coordinate_mask.tolist() == [True, True]


def test_nonpoly_ccd_with_no_observed_atom_id_overlap_is_a_safe_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _structure(_component("LIG", reference=False, topology_complete=False))
    resolved = ChemicalComponent(
        "LIG",
        (ChemicalAtom("X1", "C"), ChemicalAtom("Y1", "N")),
        frozenset({ChemicalBond("X1", "Y1")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(
        processor_module,
        "resolve_ccd_component",
        lambda component_id, *, revision, snapshot_path: resolved,
    )

    hydrated, atom_map = _hydrate_system(
        observed.system,
        revision="test-revision",
        snapshot_path=None,
    )

    assert hydrated is observed.system
    assert atom_map.tolist() == [0, 1]


def test_reference_less_unk_keeps_zero_reference_features(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _structure(_component("UNK", reference=False))

    def unexpected(*args: object, **kwargs: object) -> ChemicalComponent:
        raise AssertionError("UNK must not resolve CCD solely to obtain reference coordinates")

    monkeypatch.setattr(processor_module, "resolve_ccd_component", unexpected)
    model_input, _ = _processor().process_training(observed)

    assert not bool(model_input.reference_mask.any())
    assert torch.equal(model_input.reference_positions, torch.zeros_like(model_input.reference_positions))
    assert torch.equal(model_input.reference_charges, torch.zeros_like(model_input.reference_charges))


def test_incomplete_unk_uses_builtin_topology_without_reference_or_remote_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _structure(_component("UNK", reference=False, topology_complete=False))
    monkeypatch.setattr(
        processor_module,
        "get_component",
        lambda component_id: _component(component_id, reference=True),
    )

    def unexpected(*args: object, **kwargs: object) -> ChemicalComponent:
        raise AssertionError("UNK must not query a configured CCD locator")

    monkeypatch.setattr(processor_module, "resolve_ccd_component", unexpected)
    hydrated, _ = _hydrate_system(observed.system, revision="remote-revision", snapshot_path=None)

    assert hydrated.component_topology_complete.tolist() == [True]
    assert hydrated.edge_index.shape[1] == 1
    assert not bool(hydrated.reference_coordinate_mask.any())


def test_processor_rejects_two_ccd_locators_early() -> None:
    with pytest.raises(ValueError, match="at most one"):
        ProtenixProcessor(
            ProtenixProcessorConfig(
                ccd_revision="test-revision",
                ccd_snapshot_path="/tmp/ccd",
            )
        )


def test_standard_reference_input_performs_no_ccd_io(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _structure(_component("LIG", reference=True))

    def unexpected(*args: object, **kwargs: object) -> ChemicalComponent:
        raise AssertionError("reference-complete input must not resolve CCD")

    monkeypatch.setattr(processor_module, "resolve_ccd_component", unexpected)
    model_input, target = _processor().process_training(observed)
    assert model_input.atom_index.tolist() == [[0, 1]]
    assert target.coordinates.shape[-2] == 2


def test_local_snapshot_locator_is_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _structure(_component("LIG", reference=False))
    calls: list[tuple[str | None, str | None]] = []
    snapshot = tmp_path / "ccd"
    snapshot.mkdir()
    (snapshot / "components.parquet").touch()

    def resolve(
        component_id: str,
        *,
        revision: str | None,
        snapshot_path: str | None,
    ) -> ChemicalComponent:
        calls.append((revision, snapshot_path))
        return _component(component_id, reference=True, extra_atom=True)

    monkeypatch.setattr(processor_module, "resolve_ccd_component", resolve)
    processor = ProtenixProcessor(ProtenixProcessorConfig(ccd_snapshot_path=str(snapshot)))
    processor.process_training(observed)
    assert calls == [(None, str(snapshot.resolve()))]


def test_processor_validates_configured_local_snapshot_early(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ProtenixProcessor(ProtenixProcessorConfig(ccd_snapshot_path=str(tmp_path / "missing")))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no Parquet shards"):
        ProtenixProcessor(ProtenixProcessorConfig(ccd_snapshot_path=str(empty)))
