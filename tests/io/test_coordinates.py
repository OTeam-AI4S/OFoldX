import numpy as np
import pytest
import torch

from ofoldx.data import AtomSite, PolymerCoordinates
from ofoldx.io import (
    UnsupportedFormatError,
    parse_source,
    read_polymer_coordinates,
    read_template_coordinates,
    read_template_structure,
)


class TestPolymerCoordinates:
    def test_observation_precision(self, cif_text):
        value = 1.234567890123456
        occupancy = 0.33333333333333
        b_factor = 13.1234567890123
        text = cif_text.replace("0 1 0 1 10 7", f"{value} {-value} {value} {occupancy} {b_factor} 7", 1)
        source = parse_source(text, format="mmcif")
        direct = read_polymer_coordinates(source.view(asym_ids={"A"}))
        projected = read_template_coordinates(source, ["A"])["A"]
        structure = read_template_structure(source, "A")
        atom = structure.system.atom_index(AtomSite("A", 0, "N"))
        for field in ("coordinates", "occupancy", "b_factor"):
            assert getattr(structure, field).dtype == torch.float64
        assert structure.coordinate_mask[atom]
        assert structure.coordinates[atom].tolist() == [value, -value, value]
        assert structure.occupancy_mask[atom] and structure.occupancy[atom].item() == occupancy
        assert structure.b_factor_mask[atom] and structure.b_factor[atom].item() == b_factor
        snapshot = PolymerCoordinates.from_structure(structure)
        for coordinates in (direct, projected, snapshot):
            chain = coordinates.chains["A"]
            assert chain.coordinates.dtype == np.float64
            xyz, mask = chain.gather([0, 1], ["N"])
            assert xyz.dtype == np.float64
            np.testing.assert_array_equal(xyz, [[[value, -value, value]], [[0, 0, 0]]])
            assert mask.tolist() == [[True], [False]]

    def test_missing_declarations(self, incomplete_cif):
        category, text = incomplete_cif
        source = parse_source(text, format="mmcif")
        with pytest.raises(ValueError, match="_" + category):
            read_polymer_coordinates(source.view())

    def test_unobserved_sequence(self, cif_text):
        text = cif_text.replace("A 1\nB 2", "A 1\nC 1\nB 2")
        source = parse_source(text, format="mmcif")
        chain = read_polymer_coordinates(source.view(asym_ids={"C"})).chains["C"]
        assert chain.component_ids == ("ALA", "GLY", "MSE")
        assert chain.parent_component_ids == ((), (), ("MET",))
        assert not chain.gather([0, 1, 2], ["CA"])[1].any()

    def test_absent_chain(self, cif_text):
        source = parse_source(cif_text, format="mmcif")
        with pytest.raises(KeyError):
            read_polymer_coordinates(source.view(asym_ids={"missing"}))

    def test_assembly_unsupported(self, cif_text):
        source = parse_source(cif_text, format="mmcif")
        with pytest.raises(ValueError, match="deposited"):
            read_polymer_coordinates(source.view(assembly_id="1"))

    def test_pdb_unsupported(self):
        source = parse_source("END\n", format="pdb")
        with pytest.raises(UnsupportedFormatError):
            read_polymer_coordinates(source.view())
