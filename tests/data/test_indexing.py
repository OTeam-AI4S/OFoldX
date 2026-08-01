import numpy as np
import pytest

from ofoldx.data import AtomSite, ChemicalAtom, ChemicalComponent, MolecularSystem, Molecule
from ofoldx.data._placed_atoms import PlacedAtoms
from ofoldx.data.indexing import AtomSiteTable, AtomSiteTableBuilder, MolecularAtomIndex


def test_batch_identity_matches_graph_with_shuffled_sparse_source_rows():
    component = ChemicalComponent("L", (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")), frozenset())
    system = MolecularSystem.from_entities({"L": Molecule((component, component))}, {"01": "L", "1": "L"})
    index = MolecularAtomIndex(system)
    builder = AtomSiteTableBuilder()
    builder.add_component(("1", 1), [5000, 3], ["O1", "C1"])
    builder.add_component(("01", 0), [99], ["C1"])
    source = builder.build()
    order = [99, 3, 5000]
    result = index.map_rows(source, order)
    assert [system.atom_site(int(i)) for i in result] == [source[i] for i in order]
    assert source[99] != source[3]
    assert index.find_atom(AtomSite("1", 1, "removed")) is None
    with pytest.raises(KeyError):
        index.atom_indices([0], ["removed"])
    with pytest.raises(KeyError):
        index.map_rows(source, [4])
    with pytest.raises(ValueError):
        AtomSiteTable([1, 1], [("A", 0)], [0, 0], ["C1", "O1"])


def test_placement_snapshots_transform_and_index_inputs():
    dtype = [("row_index", "i8"), ("chain_id", object), ("coordinates", "f4", (3,)), ("coordinate_present", "?")]
    rows = np.zeros(2, dtype=dtype).view(np.recarray)
    rows.chain_id = "A"
    rows.coordinates[:] = [[1, 2, 3], [0, 0, 0]]
    rows.coordinate_present = [True, False]
    transform = np.eye(4)
    transform[0, 3] = 10
    positions, copies, keys = np.array([0, 1]), np.array([0, 0]), np.array([8, 9])
    placed = PlacedAtoms(rows, positions, copies, [("copy", transform)], keys)
    first = placed.coordinates.copy()
    transform[0, 3] = 100
    positions[:] = 1
    keys[:] = 99
    np.testing.assert_array_equal(placed[:].coordinates, first)
    np.testing.assert_array_equal(first, [[11, 2, 3], [0, 0, 0]])
    np.testing.assert_array_equal(placed.row_ids, [8, 9])
    assert placed[:].transforms[0] is placed.transforms[0]
    for values in (placed.row_ids, placed.source_positions, placed.transforms[0].matrix, placed.coordinates):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(ValueError):
        PlacedAtoms(rows, [0.5], [0], [("copy", transform)])
