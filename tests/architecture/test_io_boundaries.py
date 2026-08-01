"""Source dependency boundaries and published observation ownership."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ofoldx.data._placed_atoms import PlacedAtoms
from ofoldx.data.indexing import AtomSiteTable, AtomSiteTableBuilder


def test_source_dependencies():
    path = Path(__file__).parents[1] / "io" / "fixtures" / "template.cif"
    code = """import importlib.abc, sys
class RejectCompute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rdkit', 'danling', 'multimolecule', 'tensordict'}:
            raise AssertionError(f'source reader imports a compute dependency: {fullname}')
sys.meta_path.insert(0, RejectCompute())
from ofoldx.io import read_source, parse_source, read_template_coordinates
source = read_source(sys.argv[1])
assert source.atoms.model_ids == (1, 2)
assert len(source.atoms.by_index(1)) == 11
assert source.declarations.entity_types
assert source.connections == ()
coordinates = read_template_coordinates(source, ['A'])['A']
assert coordinates.chains['A'].component_ids == ('ALA', 'GLY', 'MSE')
pdb = parse_source('ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \\n', format='pdb')
assert pdb.atoms.model_ids == (None,)
assert pdb.view().rows.coordinates.tolist() == [[1, 2, 3]]
assert source.declarations.select_asym_ids({'A'}).asym_entities
"""
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, timeout=45)


@pytest.mark.parametrize("module", ["ofoldx.data.materialize", "ofoldx.io", "allodesign.cif"])
def test_import_order(module):
    import importlib.util
    import subprocess
    import sys

    if module.startswith("allodesign") and importlib.util.find_spec("allodesign") is None:
        pytest.skip("optional Allodesign consumer is not installed")
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr


def test_owned_addresses():
    builder = AtomSiteTableBuilder()
    builder.add_component(("A", 0), [8], ["CA"])
    first = builder.build()
    builder.add_component(("B", 0), [9], ["N"])
    assert list(first) == [8]
    with pytest.raises(ValueError):
        first.positions([8.5])
    with pytest.raises(ValueError):
        AtomSiteTable([[8]], [("A", 0)], [0], ["CA"])
    with pytest.raises(ValueError):
        AtomSiteTable([8], [("A", 0)], [0.5], ["CA"])


def test_placement_indices():
    rows = np.zeros(2, dtype=[("coordinates", "f4", (3,)), ("coordinate_present", "?")]).view(np.recarray)
    placed = PlacedAtoms(rows, [0, 1], [0, 0], [("1", np.eye(4))])
    with pytest.raises(ValueError, match="one-dimensional"):
        placed[:, None]
    with pytest.raises(TypeError):
        placed[0.5]
    with pytest.raises(IndexError):
        placed[2]


def test_runtime_ccd_dependencies():
    code = """import importlib.abc, sys
class RejectBuild(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ofoldx.data.ccd.build':
            raise AssertionError('runtime CCD reader depends on offline build')
sys.meta_path.insert(0, RejectBuild())
from ofoldx.data.ccd.reader import decode_component
from ofoldx.io.cif import parse_cif
assert decode_component(parse_cif('data_x\\n_chem_comp.id X\\n_chem_comp.type non-polymer\\n').block)['atom_ids'] == []
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=45)


def test_address_exports():
    from ofoldx.data import AtomSite
    from ofoldx.data.molecular import AtomSite as ExistingAddress
    from ofoldx.data.references import AtomSite as Address

    assert AtomSite is Address is ExistingAddress


def test_affine_transforms():
    from ofoldx.data._placed_atoms import PlacementTransform

    first = PlacementTransform("1", np.eye(4))
    second = PlacementTransform("1", np.eye(4))
    assert first == second
    assert first != PlacementTransform("2", np.eye(4))
    with pytest.raises(TypeError):
        hash(first)
    invalid = np.eye(4)
    invalid[3, 0] = 1
    with pytest.raises(ValueError, match="affine"):
        PlacementTransform("1", invalid)


def test_coordinate_ownership():
    from ofoldx.data import ChainCoordinates, PolymerCoordinates

    coordinates = np.array([[1.123456789, 2, 3], [99, 99, 99]], dtype=np.float64)
    mask = np.array([True, False])
    chain = ChainCoordinates(
        polymer_type="rna",
        component_ids=("A", "C"),
        atom_names=("P", "P"),
        component_index=np.array([0, 1]),
        coordinates=coordinates,
        coordinate_mask=mask,
    )
    chains = {"A": chain}
    evidence = PolymerCoordinates(chains)
    coordinates[:] = 0
    mask[:] = True
    chains.clear()
    xyz, present = evidence.chains["A"].gather([0, 1], ["P", "C4'"])
    assert xyz.dtype == np.float64 and xyz[0, 0, 0] == 1.123456789
    assert present.tolist() == [[True, False], [False, False]]
    assert not xyz[1].any()
    with pytest.raises(ValueError):
        chain.coordinates.setflags(write=True)
    with pytest.raises(TypeError):
        chain.gather([0.5], ["P"])
