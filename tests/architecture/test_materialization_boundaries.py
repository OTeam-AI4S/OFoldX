"""Published plan ownership and dependencies of format-independent graph realization."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ofoldx.data._materialization.chemistry import ResolvedComponents
from ofoldx.data._materialization.plan import Occurrence, PlanBuilder, ResidueKey, ResidueTable
from ofoldx.data.indexing import AtomSiteTable
from ofoldx.data.molecular import ChemicalAtom, ChemicalComponent, Molecule
from ofoldx.io.records import _ATOM_ROW_DTYPE


def _molecule(name="L"):
    return Molecule((ChemicalComponent(name, (ChemicalAtom("C", "C"),), frozenset()),))


def test_graph_dependencies():
    code = """import importlib.abc, sys
class RejectFormats(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'ofoldx.io.pdb', 'ofoldx.io.mmcif', 'ofoldx.data.materialize',
                        'ofoldx.data._cif_chemistry', 'ofoldx.data._materialization.cif_catalog',
                        'ofoldx.data._materialization.pdb_catalog'}:
            raise AssertionError(f'common molecular layer imports a format pipeline: {fullname}')
sys.meta_path.insert(0, RejectFormats())
from ofoldx.data._materialization.pipeline import realize_structure
from ofoldx.data._component_signatures import _graph_component_signatures
from ofoldx.data.molecular import Molecule, ChemicalComponent, ChemicalAtom
from ofoldx.data._materialization.plan import PlanBuilder, Occurrence
from ofoldx.data._materialization.chemistry import ResolvedComponents
from ofoldx.data._materialization.scatter import StructureMetadata
from ofoldx.io.records import _ATOM_ROW_DTYPE
import numpy as np
entity = Molecule((ChemicalComponent('L', (ChemicalAtom('C','C'),), frozenset()),))
builder = PlanBuilder()
builder.register_entity('L', entity)
builder.add_occurrence(Occurrence(0, 'A', 'L', entity, ()))
builder.bind_atoms(('A', 0), [17], ['C'])
rows = np.empty(1, dtype=_ATOM_ROW_DTYPE).view(np.recarray)
rows.row_index = [17]
rows.element = ['C']
rows.coordinate_present = [True]
rows.coordinates = [[1, 2, 3]]
chemistry = ResolvedComponents(components={'L':entity.components[0]}, topology_complete={'L':True},
                               formal_charge_atoms={}, non_leaving_atoms={})
structure = realize_structure(rows, builder.build(), (), chemistry,
                              StructureMetadata(component_parents={}, resolutions=(), formal_charge_atoms={}))
assert structure.coordinates.reshape(-1, 3).tolist() == [[1, 2, 3]]
assert len(_graph_component_signatures(structure.system)) == 1
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=45)


def test_plan_ownership():
    builder = PlanBuilder()
    value = builder.register_entity("L", _molecule())
    instance = builder.name_instance("A")
    builder.add_occurrence(Occurrence(0, instance, "L", value, ()))
    builder.set_residue_number((instance, 0), ("007", ""))
    plan = builder.build()
    second_instance = builder.name_instance("A")
    builder.add_occurrence(Occurrence(1, second_instance, "L", value, ()))
    builder.set_residue_number((instance, 0), ("008", ""))
    assert len(builder.build().instances) == 2
    assert plan.instances == {"A": "L"}
    assert plan.residue_numbers[("A", 0)] == ("007", "")
    with pytest.raises(TypeError):
        plan.entities["X"] = value
    with pytest.raises(TypeError):
        plan.instances["B"] = "L"


def test_plan_references():
    builder = PlanBuilder()
    value = _molecule()
    builder.add_occurrence(Occurrence(0, "A", "L", value, ()))
    with pytest.raises(ValueError, match="undeclared entity"):
        builder.build()
    builder.register_entity("L", value)
    with pytest.raises(ValueError, match="unique instance"):
        builder.add_occurrence(Occurrence(1, "A", "L", value, ()))
    plan = builder.build()
    residue = ResidueKey(None, "A", 1, "1", None, "L")
    invalid_references = (
        {"residue_numbers": {("missing", 0): ("1", "")}},
        {"standard_components": {("A", 1): "L"}},
        {"standard_components": {("A", 0.5): "L"}},
        {"source_asym_by_instance": {"missing": "A"}},
        {"site_by_row_index": AtomSiteTable([0], [("A", -1)], [0], ["C"])},
        {"occurrences": (Occurrence(0, "A", "L", value, ((1, residue),)),)},
        {"occurrences": (Occurrence(0, "A", "L", value, ((0, residue), (0, residue))),)},
    )
    for change in invalid_references:
        with pytest.raises(ValueError):
            replace(plan, **change)


def test_entity_variants():
    builder = PlanBuilder()
    first = builder.register_entity("L", _molecule("L1"))
    second = _molecule("L2")
    with pytest.raises(ValueError, match="conflicting molecular definitions"):
        builder.register_entity("L", second)
    assert builder.register_entity("L", second, allow_variants=True) == second
    builder.add_occurrence(Occurrence(0, "A", "L", second, ()))
    plan = builder.build()
    assert plan.entities["L"] == first
    assert plan.instance_values["A"] == second


def test_residue_positions():
    rows = np.zeros(2, dtype=_ATOM_ROW_DTYPE).view(np.recarray)
    rows.chain_id = "A"
    rows.component_id = "L"
    rows.row_index = [20, 8]
    table = ResidueTable(rows)
    positions = next(iter(table.positions.values()))
    assert rows.row_index[positions].tolist() == [20, 8]
    assert positions.tolist() == [0, 1]
    with pytest.raises(ValueError):
        positions.setflags(write=True)
    with pytest.raises(TypeError):
        table.chains[(None, "A")] = ()


def test_catalog_ownership():
    value = _molecule().components[0]
    components = {"L": value}
    resolved = ResolvedComponents(
        components=components, topology_complete={"L": True}, formal_charge_atoms={}, non_leaving_atoms={}
    )
    components.clear()
    assert resolved.components["L"] == value
    with pytest.raises(TypeError):
        resolved.components["L"] = value


def test_assembly_ownership():
    from ofoldx.data._materialization.pdb_catalog import PdbDeclarations

    transform = np.eye(4)
    chains = {"A"}
    placements = {"1": (transform, chains)}
    declared = PdbDeclarations(
        lines=(),
        sequences={},
        missing_residues=(),
        entity_by_chain={},
        modified_residues={},
        segment_by_serial={},
        assembly_placements=placements,
    )
    transform[0, 3] = 10
    chains.add("B")
    placements.clear()
    stored_transform, stored_chains = declared.assembly_placements["1"]
    assert stored_transform[0, 3] == 0
    assert stored_chains == frozenset({"A"})
    with pytest.raises(ValueError):
        stored_transform.setflags(write=True)
