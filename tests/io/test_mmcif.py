from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from ofoldx.data import (
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_ENTITY,
    BOND_ROLE_POLYMER,
    MOLECULE_BRANCHED,
    MOLECULE_MACROLIDE,
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    PolymerSequence,
    Structure,
    SystemBond,
)
from ofoldx.data._cif_chemistry import read_mmcif_components
from ofoldx.data.components import get_component
from ofoldx.data.structure import (
    RESOLUTION_KIND_EM_RECONSTRUCTION,
    RESOLUTION_KIND_REFINEMENT,
    RESOLUTION_KIND_REFLECTIONS,
)
from ofoldx.io import read_cif, read_structure, write_structure
from ofoldx.io.symmetry import _mmcif_symmetry_operator


def _structure() -> Structure:
    component = ChemicalComponent(
        "LIG",
        (
            ChemicalAtom("C1", "C", formal_charge=0),
            ChemicalAtom("O1", "O", formal_charge=-1),
        ),
        frozenset({ChemicalBond("C1", "O1")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
    )
    system = MolecularSystem.from_entities(
        {"ligand": Molecule((component,))},
        {"L": "ligand"},
    )
    return Structure(
        system=system,
        coordinates=torch.tensor([[1.0, 2.0, 3.0], [2.2, 2.0, 3.0]]),
        coordinate_mask=torch.ones(2, dtype=torch.bool),
        occupancy=torch.tensor([1.0, 0.5]),
        occupancy_mask=torch.ones(2, dtype=torch.bool),
        b_factor=torch.tensor([10.0, 20.0]),
        b_factor_mask=torch.ones(2, dtype=torch.bool),
        source_symbols=("11",),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([-1]),
        batch_size=[],
    )


def _resolved(structure: Structure) -> Structure:
    r"""Put every atom on its own point: atom_site only carries resolved coordinates."""

    count = int(structure.coordinates.shape[0])
    steps = torch.arange(count, dtype=torch.float32).unsqueeze(-1)
    return structure.replace(
        coordinates=steps * torch.tensor([1.5, 0.5, 0.25]),
        coordinate_mask=torch.ones(count, dtype=torch.bool),
    )


def _two_component_structure() -> Structure:
    components = tuple(
        ChemicalComponent(
            component_id,
            (
                ChemicalAtom("C", "C"),
                ChemicalAtom(leaving_atom, "O"),
                ChemicalAtom(leaving_hydrogen, "H"),
            ),
            frozenset(
                {
                    ChemicalBond("C", leaving_atom),
                    ChemicalBond(leaving_atom, leaving_hydrogen),
                }
            ),
        )
        for component_id, leaving_atom, leaving_hydrogen in (
            ("L1", "O1", "H1"),
            ("L2", "O2", "H2"),
        )
    )
    system = MolecularSystem.from_entities(
        {"first": Molecule((components[0],)), "second": Molecule((components[1],))},
        {"A": "first", "B": "second"},
    )
    return Structure(
        system=system,
        coordinates=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        coordinate_mask=torch.tensor([True, False, False, True, False, False]),
        occupancy=torch.ones(6),
        occupancy_mask=torch.tensor([True, False, False, True, False, False]),
        b_factor=torch.zeros(6),
        b_factor_mask=torch.tensor([True, False, False, True, False, False]),
        source_symbols=("1", "2"),
        residue_id=torch.tensor([0, 1]),
        insertion_code=torch.full((2,), -1),
        batch_size=[],
    )


def _deuterated_water() -> Structure:
    component = ChemicalComponent(
        "DOD",
        (
            ChemicalAtom("O", "O"),
            ChemicalAtom("D1", "D"),
            ChemicalAtom("D2", "D"),
        ),
        frozenset({ChemicalBond("O", "D1"), ChemicalBond("O", "D2")}),
    )
    system = MolecularSystem.from_entities({"water": Molecule((component,))}, {"A": "water"})
    return Structure(
        system=system,
        coordinates=torch.zeros(3, 3),
        coordinate_mask=torch.ones(3, dtype=torch.bool),
        occupancy=torch.ones(3),
        occupancy_mask=torch.ones(3, dtype=torch.bool),
        b_factor=torch.zeros(3),
        b_factor_mask=torch.ones(3, dtype=torch.bool),
        source_symbols=("1",),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([-1]),
        batch_size=[],
    )


def _topology_only_water() -> Structure:
    component = ChemicalComponent(
        "HOH",
        (ChemicalAtom("O", "O"), ChemicalAtom("H1", "H"), ChemicalAtom("H2", "H")),
        frozenset({ChemicalBond("O", "H1"), ChemicalBond("O", "H2")}),
    )
    system = MolecularSystem.from_entities(
        {"water": Molecule((component,), molecule_type="water")},
        {"B:18:HOH": "water"},
    )
    atom_count = system.atom_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.zeros(atom_count, 3),
        coordinate_mask=torch.zeros(atom_count, dtype=torch.bool),
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=("18",),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([-1]),
        batch_size=[],
    )


def _pna_structure() -> Structure:
    structure = Structure.from_sequence("K")
    system = MolecularSystem.from_entities(
        {"polymer": PolymerSequence("pna", (get_component("LYS"),))},
        {"A": "polymer"},
    )
    return _resolved(structure.replace(system=system))


def _charge_provenance() -> Structure:
    structure = Structure.from_sequence("K")
    structure.coordinate_mask.fill_(True)
    structure.coordinate_mask[0] = False
    structure.source_formal_charge_mask[0] = True
    return structure


#: Two embedded definitions of the same declared polymer type: ``XYZ`` states complete
#: leaving-atom chemistry and an idealized conformer, while ``QQQ`` leaves it unstated.
_EMBEDDED_COMPONENTS = """data_components
#
loop_
_chem_comp.id
_chem_comp.type
XYZ 'L-PEPTIDE LINKING'
QQQ 'L-PEPTIDE LINKING'
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
XYZ N   N N ? ? ? 1.0 0.0 0.0
XYZ CA  C N ? ? ? 2.0 0.0 0.0
XYZ C   C N ? ? ? 3.0 0.0 0.0
XYZ O   O N ? ? ? 4.0 0.0 0.0
XYZ OXT O Y ? ? ? 5.0 0.0 0.0
XYZ HXT H N ? ? ? 6.0 0.0 0.0
QQQ N   N ? 7.0 0.0 0.0 ? ? ?
QQQ CA  C ? 8.0 0.0 0.0 ? ? ?
QQQ C   C ? 9.0 0.0 0.0 ? ? ?
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
XYZ N   CA  sing
XYZ CA  C   sing
XYZ C   O   doub
XYZ C   OXT sing
XYZ OXT HXT sing
QQQ N   CA  sing
QQQ CA  C   sing
#
"""


#: Structures that differ in the ways mmCIF has to encode separately: a non-polymer
#: molecule, a protein polymer, and a nucleic polymer.
_ROUND_TRIP: dict[str, Callable[[], Structure]] = {
    "charge_provenance": _charge_provenance,
    "deuterated_water": _deuterated_water,
    "molecule": _structure,
    "pna": _pna_structure,
    "protein": lambda: _resolved(Structure.from_sequence("ACDEF")),
    "nucleic": lambda: _resolved(Structure.from_sequence("ACGU", polymer_type="rna")),
    "topology": lambda: Structure.from_sequence("ACDEF"),
    "topology_only_water": _topology_only_water,
}


class TestMmcifRoundTrip:
    @pytest.mark.parametrize("case", sorted(_ROUND_TRIP))
    def test_public_graph(
        self,
        case: str,
        tmp_path: Path,
    ) -> None:
        structure = _ROUND_TRIP[case]()
        path = tmp_path / f"{case}.cif"
        write_structure(structure, path)
        restored = read_structure(path)

        assert restored.system.symbols == structure.system.symbols
        for field in MolecularSystem.__annotations__:
            if field != "symbols":
                assert torch.equal(
                    getattr(restored.system, field),
                    getattr(structure.system, field),
                )
        assert restored.source_symbols == structure.source_symbols
        for field in Structure.__annotations__:
            if field not in {"system", "source_symbols"}:
                assert torch.equal(getattr(restored, field), getattr(structure, field))

    def test_observation_precision(self, tmp_path: Path) -> None:
        structure = Structure.from_sequence("A")
        structure = structure.replace(
            coordinates=structure.coordinates.double(),
            occupancy=structure.occupancy.double(),
            b_factor=structure.b_factor.double(),
        )
        value = 1.234567890123456
        structure.coordinate_mask[0] = True
        structure.coordinates[0] = structure.coordinates.new_tensor([value, -value, value])
        structure.occupancy[0] = 0.33333333333333
        structure.occupancy_mask[0] = True
        structure.b_factor[0] = 13.1234567890123
        structure.b_factor_mask[0] = True

        path = tmp_path / "observations.cif"
        write_structure(structure, path)
        category = read_cif(path).block["atom_site"]
        assert category.row_count == 1
        xyz = [category[name].as_array(np.float64)[0] for name in ("Cartn_x", "Cartn_y", "Cartn_z")]
        assert xyz == [value, -value, value]
        assert category["occupancy"].as_array(np.float64)[0] == structure.occupancy[0].item()
        assert category["B_iso_or_equiv"].as_array(np.float64)[0] == structure.b_factor[0].item()

        restored = read_structure(path)
        for field in ("coordinates", "occupancy", "b_factor"):
            assert getattr(restored, field).dtype == torch.float64
            torch.testing.assert_close(getattr(restored, field), getattr(structure, field), rtol=0, atol=0)
        for field in ("coordinate_mask", "occupancy_mask", "b_factor_mask"):
            assert torch.equal(getattr(restored, field), getattr(structure, field))
        assert restored.system.reference_coordinates.dtype == structure.system.reference_coordinates.dtype

    def test_provenance_reference(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "provenance.cif"
        write_structure(_charge_provenance(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["ofoldx_source_formal_charge"]["label_atom_id"] = pdbx.CIFColumn(["MISSING"])
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="references unknown atom"):
            read_structure(path)

    def test_coordinate_source(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "coordinates.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        values = cif_file.block["atom_site"]["Cartn_x"].as_array()
        values[0] = 123.0
        cif_file.block["atom_site"]["Cartn_x"] = pdbx.CIFColumn(values)
        cif_file.write(str(path))

        assert read_structure(path).coordinates[0, 0].item() == 123.0

    @pytest.mark.parametrize(
        ("category_name", "column_name", "value", "expected", "kind", "compressed_name"),
        (
            (
                "refine",
                "ls_d_res_high",
                "2.30(5)",
                2.3,
                RESOLUTION_KIND_REFINEMENT,
                "refine.cif.gz",
            ),
            (
                "em_3d_reconstruction",
                "resolution",
                1.8,
                1.8,
                RESOLUTION_KIND_EM_RECONSTRUCTION,
                "em.mmcif.gz",
            ),
            (
                "reflns",
                "d_resolution_high",
                1.8,
                1.8,
                RESOLUTION_KIND_REFLECTIONS,
                "reflns.cif.gz",
            ),
        ),
    )
    def test_resolution(
        self,
        tmp_path: Path,
        category_name: str,
        column_name: str,
        value: float | str,
        expected: float,
        kind: int,
        compressed_name: str,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "resolution.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block[category_name] = pdbx.CIFCategory({column_name: [value]})
        cif_file.write(str(path))
        compressed = tmp_path / compressed_name
        with gzip.open(compressed, "wb") as stream:
            stream.write(path.read_bytes())

        restored = read_structure(compressed)

        assert restored.resolution.tolist() == pytest.approx([expected])
        assert restored.resolution_kind.tolist() == [kind]

        output = tmp_path / f"{category_name}-roundtrip.cif"
        write_structure(restored, output)
        reread = read_structure(output)
        assert reread.resolution.tolist() == pytest.approx([expected])
        assert reread.resolution_kind.tolist() == [kind]

    def test_multiple_resolutions(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        source = tmp_path / "multiple-resolution.cif"
        write_structure(_structure(), source)
        cif_file = pdbx.CIFFile.read(str(source))
        cif_file.block["refine"] = pdbx.CIFCategory({"ls_d_res_high": [1.8, 2.0]})
        cif_file.block["reflns"] = pdbx.CIFCategory({"d_resolution_high": [2.1]})
        cif_file.write(str(source))

        structure = read_structure(source)
        target = tmp_path / "multiple-resolution-roundtrip.cif"
        write_structure(structure, target)
        restored = read_structure(target)

        assert torch.equal(restored.system_resolution_ptr, structure.system_resolution_ptr)
        assert torch.equal(restored.resolution, structure.resolution)
        assert torch.equal(restored.resolution_kind, structure.resolution_kind)


class TestMmcifReader:
    def test_polymer_microheterogeneity(
        self,
        tmp_path: Path,
    ) -> None:
        components = {
            component_id: ChemicalComponent(
                component_id,
                (ChemicalAtom("N", "N"), ChemicalAtom("C", "C")),
                frozenset({ChemicalBond("N", "C")}),
                previous_attachment=PolymerAttachment("N"),
                next_attachment=PolymerAttachment("C"),
            )
            for component_id in ("HYP", "PRO")
        }
        path = tmp_path / "polymer-microheterogeneity.cif"
        path.write_text(
            """data_microheterogeneity
loop_
_entity.id
_entity.type
1 polymer
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 HYP y
1 1 PRO y
1 2 HYP y
1 2 PRO y
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
A 1 1 HYP 13 13 ?
A 1 1 PRO 113 113 ?
A 1 2 HYP 14 14 ?
A 1 2 PRO 114 114 ?
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N A HYP A 1 1 ? 1 0 0 0.6 10 13 HYP A N 1
ATOM 2 C C A HYP A 1 1 ? 1 1 0 ? 10 13 HYP A C 1
ATOM 3 N N B PRO A 1 1 ? 101 0 0 0.4 10 113 PRO A N 1
ATOM 4 C C B PRO A 1 1 ? 101 1 0 0.4 10 113 PRO A C 1
ATOM 5 N N A HYP A 1 2 ? 2 0 0 0.1 10 14 HYP A N 1
ATOM 6 C C A HYP A 1 2 ? 2 1 0 0.1 10 14 HYP A C 1
ATOM 7 N N B PRO A 1 2 ? 102 0 0 0.8 10 114 PRO A N 1
ATOM 8 C C B PRO A 1 2 ? 102 1 0 0.8 10 114 PRO A C 1
""",
            encoding="ascii",
        )

        restored = read_structure(path, components=components)
        component_ids = tuple(
            restored.system.symbols[int(restored.system.component_symbol[index])] for index in range(2)
        )
        residue_numbers = tuple(restored.source_symbols[int(index)] for index in restored.residue_id)

        assert component_ids == ("HYP", "PRO")
        assert residue_numbers == ("13", "114")
        assert restored.coordinates[:, 0].tolist() == [1.0, 1.0, 102.0, 102.0]

    def test_polymer_type(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "polymer-type.cif"
        write_structure(Structure.from_sequence("A"), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["entity_poly"]["type"] = pdbx.CIFColumn(["other"])
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="unsupported _entity_poly.type 'other'"):
            read_structure(path)

    def test_topology_only_assembly(self, tmp_path: Path) -> None:
        path = tmp_path / "topology.cif"
        write_structure(Structure.from_sequence("ACDEF"), path)

        with pytest.raises(ValueError, match="assembly lacks its operation categories"):
            read_structure(path, assembly_id="1")
        with pytest.raises(ValueError, match="requested model_index 2"):
            read_structure(path, model_index=2)


class TestMmcifWriter:
    def test_polymer_nick(
        self,
        tmp_path: Path,
    ) -> None:
        components = PolymerSequence.from_one_letter("AA").components
        nick_system = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", components, linked=(False,))},
            {"A": "protein"},
        )
        atom_count = nick_system.atom_symbol.numel()
        nick = Structure(
            system=nick_system,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=(),
            residue_id=torch.full((2,), -1),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )
        path = tmp_path / "visible-nick.cif"
        write_structure(nick, path)
        restored = read_structure(path)
        assert not bool((restored.system.bond_role == BOND_ROLE_POLYMER).any())

    def test_hidden_hydrogens(self, tmp_path: Path) -> None:
        components = PolymerSequence.from_one_letter("AA").components
        heavy_component = components[0].without_hydrogens()
        heavy_system = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", (heavy_component,))},
            {"A": "protein"},
        )
        heavy_atom_count = heavy_system.atom_symbol.numel()
        heavy = Structure(
            system=heavy_system,
            coordinates=torch.zeros((heavy_atom_count, 3)),
            coordinate_mask=torch.ones(heavy_atom_count, dtype=torch.bool),
            occupancy=torch.zeros(heavy_atom_count),
            occupancy_mask=torch.zeros(heavy_atom_count, dtype=torch.bool),
            b_factor=torch.zeros(heavy_atom_count),
            b_factor_mask=torch.zeros(heavy_atom_count, dtype=torch.bool),
            source_symbols=(),
            residue_id=torch.full((1,), -1),
            insertion_code=torch.full((1,), -1),
            batch_size=[],
        )
        with pytest.raises(ValueError, match="does not match its authoritative definition"):
            write_structure(heavy, tmp_path / "heavy-only.cif")

    def test_masked_coordinates(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        masked = structure.replace(
            coordinates=structure.coordinates.clone(),
            coordinate_mask=torch.tensor([False, True]),
            occupancy=structure.occupancy.clone(),
            occupancy_mask=torch.tensor([False, True]),
            b_factor=structure.b_factor.clone(),
            b_factor_mask=torch.tensor([False, True]),
        )
        masked.coordinates[0] = torch.nan
        path = tmp_path / "masked.cif"

        write_structure(masked, path)
        restored = read_structure(path)

        assert not restored.coordinate_mask[0]
        assert restored.coordinates[0].tolist() == [0.0, 0.0, 0.0]

    def test_aromaticity_masks(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        unknown_atoms = structure.replace(
            system=structure.system.replace(
                atom_aromatic=torch.ones_like(structure.system.atom_aromatic),
            )
        )
        path = tmp_path / "unknown-aromaticity.cif"
        write_structure(unknown_atoms, path)
        assert not read_structure(path).system.atom_aromatic_known.any()

        partial_bond = structure.replace(
            system=structure.system.replace(
                bond_aromatic_known=torch.zeros_like(structure.system.bond_aromatic_known),
            )
        )
        bond_path = tmp_path / "partial-bond.cif"
        write_structure(partial_bond, bond_path)
        restored = read_structure(bond_path)

        assert not restored.system.bond_aromatic_known.any()
        assert restored.system.bond_order.tolist() == partial_bond.system.bond_order.tolist()


class TestMmcifComponents:
    def test_polymer_attachments(self, tmp_path: Path) -> None:

        path = tmp_path / "components.cif"
        path.write_text(_EMBEDDED_COMPONENTS)
        components, _, _, _ = read_mmcif_components(read_cif(path))

        linking = components["XYZ"]
        assert linking.leaving_atom_ids == frozenset({"OXT"})
        assert linking.non_leaving_atom_ids == frozenset({"N", "CA", "C", "O", "HXT"})
        assert linking.previous_attachment == PolymerAttachment("N")
        assert linking.next_attachment == PolymerAttachment("C", frozenset({"OXT", "HXT"}))
        assert linking.reference_coordinates == (
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
        )

    def test_unspecified_attachments(self, tmp_path: Path) -> None:

        path = tmp_path / "components.cif"
        path.write_text(_EMBEDDED_COMPONENTS)
        components, _, _, _ = read_mmcif_components(read_cif(path))

        unstated = components["QQQ"]
        assert unstated.leaving_atom_ids == frozenset()
        assert unstated.non_leaving_atom_ids == frozenset()
        assert unstated.previous_attachment is None
        assert unstated.next_attachment is None
        assert unstated.reference_coordinates == ((7.0, 0.0, 0.0), (8.0, 0.0, 0.0), (9.0, 0.0, 0.0))

    def test_invalid_leaving_flag(self, tmp_path: Path) -> None:

        path = tmp_path / "components.cif"
        path.write_text(_EMBEDDED_COMPONENTS.replace("XYZ N   N N", "XYZ N   N X", 1))
        with pytest.raises(ValueError, match="leaving_atom_flag must be Y or N"):
            read_mmcif_components(read_cif(path))

    def test_owned_components(self, tmp_path: Path) -> None:

        path = tmp_path / "components.cif"
        path.write_text(_EMBEDDED_COMPONENTS.replace("XYZ N   N N", "XYZ N   N X", 1))
        components, _, _, _ = read_mmcif_components(read_cif(path), owned={"XYZ"})
        assert set(components) == {"QQQ"}

    @pytest.mark.parametrize(
        ("flag", "leaving", "non_leaving"),
        (
            ("?", frozenset(), frozenset()),
            ("N", frozenset(), frozenset({"N"})),
            ("Y", frozenset({"N"}), frozenset()),
        ),
    )
    def test_leaving_flags(
        self,
        flag: str,
        leaving: frozenset[str],
        non_leaving: frozenset[str],
        tmp_path: Path,
    ) -> None:

        path = tmp_path / "components.cif"
        path.write_text(_EMBEDDED_COMPONENTS.replace("QQQ N   N ?", f"QQQ N   N {flag}", 1))
        components, _, _, _ = read_mmcif_components(read_cif(path))
        assert components["QQQ"].leaving_atom_ids == leaving
        assert components["QQQ"].non_leaving_atom_ids == non_leaving

    def test_known_component(self, tmp_path: Path) -> None:

        path = tmp_path / "components.cif"
        path.write_text(_EMBEDDED_COMPONENTS.replace("XYZ", "ALA"))
        components, _, _, _ = read_mmcif_components(read_cif(path))
        assert "ALA" in components

    def test_explicit_stereo(self, tmp_path: Path) -> None:

        path = tmp_path / "stereo.cif"
        path.write_text(
            """data_stereo
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_stereo_config
LIG C1 C N
LIG C2 C ?
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_stereo_config
LIG C1 C2 sing N
""",
            encoding="ascii",
        )
        components, _, _, stereo_known = read_mmcif_components(read_cif(path))

        atom_stereo_known, bond_stereo_known = stereo_known["LIG"]
        component = components["LIG"]

        assert component.atoms[0].stereo is None
        assert next(iter(component.bonds or ())).stereo is None
        assert atom_stereo_known == frozenset({"C1"})
        assert bond_stereo_known == frozenset({frozenset({"C1", "C2"})})


class TestMmcifBranches:
    def test_entity_category(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        structure = _structure()
        structure = structure.replace(
            system=structure.system.replace(
                entity_molecule_type=structure.system.entity_molecule_type.new_tensor([MOLECULE_BRANCHED])
            )
        )
        path = tmp_path / "conflicting-category.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["entity"]["type"] = pdbx.CIFColumn(["non-polymer"])
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="branched-entity definition"):
            read_structure(path)

    def test_microheterogeneity(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        def component(component_id: str, *atom_ids: str) -> ChemicalComponent:
            return ChemicalComponent(
                component_id,
                tuple(ChemicalAtom(atom_id, "C") for atom_id in atom_ids),
                frozenset(ChemicalBond(first, second) for first, second in zip(atom_ids, atom_ids[1:], strict=False)),
            )

        lgu, bem, maw, external = (
            component("LGU", "L", "Q"),
            component("BEM", "B"),
            component("MAW", "M"),
            component("EXT", "E"),
        )
        branch = Molecule(
            (lgu, bem, maw),
            frozenset(
                {
                    MoleculeBond(0, "L", 1, "B"),
                    MoleculeBond(1, "B", 2, "M"),
                }
            ),
            molecule_type="branched",
        )
        system = MolecularSystem.from_entities(
            {"branch": branch, "external": Molecule((external,))},
            {"G": "branch", "X": "external"},
            (SystemBond(AtomSite("G", 1, "B"), AtomSite("X", 0, "E")),),
        )
        atom_count = int(system.atom_symbol.numel())
        structure = Structure(
            system=system,
            coordinates=torch.arange(atom_count, dtype=torch.float32).unsqueeze(-1).repeat(1, 3),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.ones(atom_count),
            occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
            source_symbols=("1", "2", "3"),
            residue_id=torch.tensor([0, 1, 2, 0]),
            insertion_code=torch.full((4,), -1),
            batch_size=[],
        )
        path = tmp_path / "branched-microheterogeneity.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        block = cif_file.block

        def duplicate_row(
            category: pdbx.CIFCategory,
            index: int,
            *,
            before: bool = False,
            **updates: object,
        ) -> pdbx.CIFCategory:
            columns: dict[str, list[object]] = {}
            for name in category:
                values = category[name].as_array().tolist()
                value = updates.get(name, values[index])
                if before:
                    values.insert(index, value)
                else:
                    values.append(value)
                columns[name] = values
            return pdbx.CIFCategory(columns)

        branch_list = block["pdbx_entity_branch_list"]
        branch_list["hetero"] = pdbx.CIFColumn(["n", "y", "n"])
        block["pdbx_entity_branch_list"] = duplicate_row(
            branch_list,
            1,
            comp_id="LGU",
            hetero="y",
        )

        branch_scheme = block["pdbx_branch_scheme"]
        branch_scheme["hetero"] = pdbx.CIFColumn(["n", "y", "n"])
        block["pdbx_branch_scheme"] = duplicate_row(
            branch_scheme,
            1,
            mon_id="LGU",
            pdb_mon_id="LGU",
            auth_mon_id="LGU",
            hetero="y",
        )

        branch_links = block["pdbx_entity_branch_link"]
        branch_links = duplicate_row(
            branch_links,
            0,
            link_id=3,
            comp_id_2="LGU",
            atom_id_2="L",
        )
        block["pdbx_entity_branch_link"] = duplicate_row(
            branch_links,
            1,
            link_id=4,
            comp_id_1="LGU",
            atom_id_1="L",
        )

        atom_site = block["atom_site"]
        atom_components = atom_site["label_comp_id"].as_array().tolist()
        atom_site["label_alt_id"] = pdbx.CIFColumn(
            ["A" if component_id == "BEM" else "." for component_id in atom_components]
        )
        atom_site["occupancy"] = pdbx.CIFColumn(
            [0.6 if component_id == "BEM" else 1.0 for component_id in atom_components]
        )
        bem_index = atom_components.index("BEM")
        atom_site = duplicate_row(
            atom_site,
            bem_index,
            before=True,
            id=6,
            label_atom_id="Q",
            label_alt_id="B",
            label_comp_id="LGU",
            occupancy=0.4,
            auth_comp_id="LGU",
            auth_atom_id="Q",
        )
        bem_index = atom_site["label_comp_id"].as_array().tolist().index("BEM")
        block["atom_site"] = duplicate_row(
            atom_site,
            bem_index,
            before=True,
            id=7,
            label_atom_id="L",
            label_alt_id="B",
            label_comp_id="LGU",
            occupancy=0.4,
            auth_comp_id="LGU",
            auth_atom_id="L",
        )

        connections = block["struct_conn"]
        connections["pdbx_ptnr1_label_alt_id"] = pdbx.CIFColumn(["A"])
        block["struct_conn"] = duplicate_row(
            connections,
            0,
            id=2,
            ptnr1_label_comp_id="LGU",
            ptnr1_label_atom_id="L",
            ptnr1_auth_comp_id="LGU",
            ptnr1_auth_atom_id="L",
            pdbx_ptnr1_label_alt_id="B",
        )
        cif_file.write(str(path))

        def assert_realization(
            restored: Structure,
            expected_component: str,
            expected_atom: str,
            expected_occupancy: float,
        ) -> None:
            component_ids = [
                restored.system.symbols[
                    int(restored.system.component_symbol[restored.system.component_index(("G", index))])
                ]
                for index in range(3)
            ]
            assert component_ids == ["LGU", expected_component, "MAW"]
            assert (restored.system.bond_role == BOND_ROLE_ENTITY).sum().item() == 2
            assembly = restored.system.edge_index[:, restored.system.bond_role == BOND_ROLE_ASSEMBLY]
            assert assembly.shape == (2, 1)
            assert {restored.system.atom_site(int(index)) for index in assembly[:, 0]} == {
                AtomSite("G", 1, expected_atom),
                AtomSite("X", 0, "E"),
            }
            atom_index = restored.system.atom_index(AtomSite("G", 1, expected_atom))
            assert restored.occupancy[atom_index].item() == pytest.approx(expected_occupancy)

        assert_realization(read_structure(path), "BEM", "B", 0.6)
        assert_realization(read_structure(path, altloc="first"), "LGU", "L", 0.4)

    def test_scheme_identity(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        component = ChemicalComponent(
            "NAG",
            (ChemicalAtom("L", "O"), ChemicalAtom("R", "C")),
            frozenset({ChemicalBond("L", "R")}),
        )
        branch = Molecule(
            (component, component, component),
            frozenset(
                {
                    MoleculeBond(0, "R", 1, "L"),
                    MoleculeBond(1, "R", 2, "L"),
                }
            ),
            molecule_type="branched",
        )
        system = MolecularSystem.from_entities({"branch": branch}, {"B": "branch"})
        atom_count = int(system.atom_symbol.numel())
        structure = Structure(
            system=system,
            coordinates=torch.arange(atom_count, dtype=torch.float32).unsqueeze(-1).repeat(1, 3),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.ones(atom_count),
            occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
            source_symbols=("1", "2", "3"),
            residue_id=torch.tensor([0, 1, 2]),
            insertion_code=torch.full((3,), -1),
            batch_size=[],
        )
        path = tmp_path / "branch-scheme-namespaces.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        scheme = cif_file.block["pdbx_branch_scheme"]
        scheme["auth_asym_id"] = pdbx.CIFColumn(["E", "D", "B"])
        scheme["auth_seq_num"] = pdbx.CIFColumn(["3", "2", "1"])
        scheme["pdb_ins_code"] = pdbx.CIFColumn(["A", ".", "."])
        atom_site = cif_file.block["atom_site"]
        atom_site["pdbx_PDB_ins_code"] = pdbx.CIFColumn(
            ["A" if sequence_id == "1" else "." for sequence_id in atom_site["auth_seq_id"].as_array(str)]
        )
        cif_file.write(str(path))

        restored = read_structure(path)

        assert restored.system.component_symbol.numel() == 3
        assert (restored.system.bond_role == BOND_ROLE_ENTITY).sum().item() == 2


class TestMmcifConnections:
    def test_duplicate_branch_link(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        donor = ChemicalComponent(
            "DNR",
            (ChemicalAtom("O4", "O"), ChemicalAtom("HO4", "H")),
            frozenset({ChemicalBond("O4", "HO4")}),
        )
        acceptor = ChemicalComponent(
            "ACC",
            (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
            frozenset({ChemicalBond("C1", "O1")}),
        )
        external = ChemicalComponent("EXT", (ChemicalAtom("E", "C"),), frozenset())
        branch = Molecule(
            (donor, acceptor),
            frozenset(
                {
                    MoleculeBond(
                        0,
                        "O4",
                        1,
                        "C1",
                        leaving_a=frozenset({"HO4"}),
                        leaving_b=frozenset({"O1"}),
                    )
                }
            ),
            molecule_type="branched",
        )
        system = MolecularSystem.from_entities(
            {"branch": branch, "external": Molecule((external,))},
            {"B": "branch", "X": "external"},
            (SystemBond(AtomSite("B", 1, "C1"), AtomSite("X", 0, "E")),),
        )
        structure = Structure(
            system=system,
            coordinates=torch.arange(5, dtype=torch.float32).unsqueeze(-1).repeat(1, 3),
            coordinate_mask=torch.tensor([True, False, True, False, True]),
            occupancy=torch.ones(5),
            occupancy_mask=torch.tensor([True, False, True, False, True]),
            b_factor=torch.zeros(5),
            b_factor_mask=torch.tensor([True, False, True, False, True]),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1, 0]),
            insertion_code=torch.full((3,), -1),
            batch_size=[],
        )
        path = tmp_path / "duplicate-branch-link.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        connections = cif_file.block["struct_conn"]
        updates = {
            "id": "2",
            "pdbx_leaving_atom_flag": "both",
            "pdbx_ptnr1_leaving_atom_id": "HO4",
            "pdbx_ptnr2_leaving_atom_id": "O1",
            "ptnr1_label_asym_id": "B",
            "ptnr1_label_seq_id": ".",
            "ptnr1_label_comp_id": "DNR",
            "ptnr1_label_atom_id": "O4",
            "ptnr2_label_asym_id": "B",
            "ptnr2_label_seq_id": ".",
            "ptnr2_label_comp_id": "ACC",
            "ptnr2_label_atom_id": "C1",
            "ptnr1_auth_asym_id": "B",
            "ptnr1_auth_seq_id": "1",
            "ptnr1_auth_comp_id": "DNR",
            "ptnr1_auth_atom_id": "O4",
            "ptnr2_auth_asym_id": "B",
            "ptnr2_auth_seq_id": "2",
            "ptnr2_auth_comp_id": "ACC",
            "ptnr2_auth_atom_id": "C1",
        }
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                name: [*column.as_array().tolist(), updates.get(name, column.as_array().tolist()[0])]
                for name, column in connections.items()
            }
        )
        for name, branch_order, duplicate_order in (
            ("declared", "sing", "?"),
            ("duplicate", "?", "sing"),
        ):
            cif_file.block["pdbx_entity_branch_link"]["value_order"] = pdbx.CIFColumn([branch_order])
            cif_file.block["struct_conn"]["pdbx_value_order"] = pdbx.CIFColumn(["sing", duplicate_order])
            cif_file.write(str(path))

            restored = read_structure(path)
            entity_bond = restored.system.bond_role == BOND_ROLE_ENTITY

            assert entity_bond.sum().item() == 1
            assert (restored.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1
            assert restored.system.bond_order_known[entity_bond].tolist() == [True]
            assert restored.system.bond_order[entity_bond].tolist() == [1.0]
            assert restored.system.bond_aromatic_known[entity_bond].tolist() == [True]
            assert restored.system.bond_aromatic[entity_bond].tolist() == [False]

            round_trip = tmp_path / f"duplicate-branch-link-{name}-roundtrip.cif"
            write_structure(restored, round_trip)
            round_tripped = read_structure(round_trip)
            round_trip_entity_bond = round_tripped.system.bond_role == BOND_ROLE_ENTITY
            assert round_tripped.system.bond_order_known[round_trip_entity_bond].tolist() == [True]
            assert round_tripped.system.bond_order[round_trip_entity_bond].tolist() == [1.0]
            assert round_tripped.system.bond_aromatic_known[round_trip_entity_bond].tolist() == [True]
            assert round_tripped.system.bond_aromatic[round_trip_entity_bond].tolist() == [False]

        cif_file.block["pdbx_entity_branch_link"]["value_order"] = pdbx.CIFColumn(["sing"])
        cif_file.block["struct_conn"]["pdbx_value_order"] = pdbx.CIFColumn(["sing", "doub"])
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="conflicts with its branched-entity link"):
            read_structure(path)

    def test_observed_leaving_atom(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        donor = ChemicalComponent(
            "DNR",
            (ChemicalAtom("C", "C"), ChemicalAtom("O", "O")),
            frozenset({ChemicalBond("C", "O")}),
        )
        acceptor = ChemicalComponent("ACC", (ChemicalAtom("C", "C"),), frozenset())
        branch = Molecule(
            (donor, acceptor),
            frozenset({MoleculeBond(0, "C", 1, "C")}),
            molecule_type="branched",
        )
        system = MolecularSystem.from_entities({"branch": branch}, {"B": "branch"})
        atom_count = int(system.atom_symbol.numel())
        structure = Structure(
            system=system,
            coordinates=torch.zeros(atom_count, 3),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.ones(atom_count),
            occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )
        path = tmp_path / "observed-branch-leaving.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["pdbx_entity_branch_link"]["leaving_atom_id_1"] = pdbx.CIFColumn(["O"])
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="declares observed leaving atom 'O'"):
            read_structure(path)

        cif_file.block["pdbx_entity_branch_link"]["leaving_atom_id_1"] = pdbx.CIFColumn(["NOPE"])
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="invalid leaving atom 'NOPE'"):
            read_structure(path)

    def test_invalid_semantics(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "struct-conn-contract.cif"
        write_structure(_two_component_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))

        def connection(conn_type: str | None = "covale", atom_id: str = "C") -> pdbx.CIFCategory:
            columns: dict[str, list[str]] = {
                "id": ["1"],
                "pdbx_leaving_atom_flag": ["none"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["."],
                "ptnr1_label_comp_id": ["L1"],
                "ptnr1_label_atom_id": [atom_id],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["B"],
                "ptnr2_label_seq_id": ["."],
                "ptnr2_label_comp_id": ["L2"],
                "ptnr2_label_atom_id": ["C"],
                "ptnr2_symmetry": ["1_555"],
                "pdbx_value_order": ["sing"],
            }
            if conn_type is not None:
                columns["conn_type_id"] = [conn_type]
            return pdbx.CIFCategory(columns)

        cif_file.block["struct_conn"] = connection(None)
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="has no connection type"):
            read_structure(path)

        cif_file.block["struct_conn"] = connection("unknown")
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="unsupported connection type 'unknown'"):
            read_structure(path)

        cif_file.block["struct_conn"] = connection("covale", "NOPE")
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="cannot resolve endpoint 1 in the source molecular declaration"):
            read_structure(path)

        cif_file.block["struct_conn"] = connection("hydrog")
        cif_file.write(str(path))
        restored = read_structure(path)
        assert not bool((restored.system.bond_role == BOND_ROLE_ASSEMBLY).any())

    def test_component_references(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        pna_component = ChemicalComponent(
            "PNA",
            (ChemicalAtom("C1", "C"), ChemicalAtom("N1", "N")),
            frozenset({ChemicalBond("C1", "N1")}),
        )
        pna_system = MolecularSystem.from_entities({"pna": PolymerSequence("pna", (pna_component,))}, {"P": "pna"})
        hybrid_system = MolecularSystem.from_entities(
            {"hybrid": PolymerSequence("dna_rna_hybrid", (pna_component,))}, {"H": "hybrid"}
        )
        cases = (
            ("ligand", _structure(), "LIG", "non-polymer"),
            ("protein", Structure.from_sequence("A"), "ALA", "L-peptide linking"),
            ("pna", _structure().replace(system=pna_system), "PNA", "other"),
            ("hybrid", _structure().replace(system=hybrid_system), "PNA", "other"),
        )
        reference_columns = (
            ("atom_site", "label_comp_id"),
            ("entity_poly_seq", "mon_id"),
            ("pdbx_entity_branch_list", "comp_id"),
            ("pdbx_entity_nonpoly", "comp_id"),
            ("chem_comp_atom", "comp_id"),
            ("chem_comp_bond", "comp_id"),
        )
        for name, structure, component_id, component_type in cases:
            path = tmp_path / f"{name}.cif"
            write_structure(structure, path)
            block = pdbx.CIFFile.read(str(path)).block
            component_types = dict(
                zip(
                    block["chem_comp"]["id"].as_array(),
                    block["chem_comp"]["type"].as_array(),
                    strict=True,
                )
            )
            referenced = {
                str(value)
                for category_name, column_name in reference_columns
                if category_name in block
                for value in block[category_name][column_name].as_array()
            }

            assert referenced <= set(component_types)
            assert component_types[component_id].casefold() == component_type.casefold()

    @pytest.mark.parametrize("parents", [(), ("ALA",), ("ALA", "GLY")], ids=["none", "one", "multiple"])
    def test_parent_roundtrip(self, tmp_path, parents):
        structure = Structure.from_sequence("AA")
        count = len(parents)
        structure = structure.replace(
            coordinate_mask=torch.ones_like(structure.coordinate_mask),
            source_symbols=parents,
            component_parent_ptr=torch.tensor([0, count, 2 * count]),
            parent_component_id=torch.tensor(list(range(count)) * 2, dtype=torch.int64),
        )
        path = tmp_path / "parents.cif"
        write_structure(structure, path)
        restored = read_structure(path)
        assert [restored.component_parents(i) for i in range(2)] == [parents, parents]

    def test_parent_conflict(self, tmp_path):
        structure = Structure.from_sequence("AA").replace(
            source_symbols=("ALA",),
            component_parent_ptr=torch.tensor([0, 1, 1]),
            parent_component_id=torch.tensor([0]),
        )
        with pytest.raises(ValueError, match="conflicting parent declarations"):
            write_structure(structure, tmp_path / "conflicting-parents.cif")

    def test_explicit_leaving_atoms(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "leaving-atoms.cif"
        write_structure(_two_component_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                "id": ["1"],
                "conn_type_id": ["covale"],
                "pdbx_leaving_atom_flag": ["both"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["."],
                "ptnr1_label_comp_id": ["L1"],
                "ptnr1_label_atom_id": ["C"],
                "pdbx_ptnr1_leaving_atom_id": ["O1"],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["B"],
                "ptnr2_label_seq_id": ["."],
                "ptnr2_label_comp_id": ["L2"],
                "ptnr2_label_atom_id": ["C"],
                "pdbx_ptnr2_leaving_atom_id": ["O2"],
                "ptnr2_symmetry": ["1_555"],
                "pdbx_value_order": ["sing"],
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path)
        assert restored.system.atomic_number.numel() == 2
        assert (restored.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1

        cif_file.block["ofoldx_source_formal_charge"] = pdbx.CIFCategory(
            {
                "label_asym_id": ["A"],
                "component_index": [0],
                "label_atom_id": ["O1"],
            }
        )
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="references unrealized atom"):
            read_structure(path)
        del cif_file.block["ofoldx_source_formal_charge"]
        cif_file.write(str(path))

        without_hydrogens = read_structure(path, include_hydrogens=False)
        assert without_hydrogens.system.atomic_number.numel() == 2
        assert (without_hydrogens.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1

        cif_file.block["struct_conn"]["pdbx_ptnr1_leaving_atom_id"] = pdbx.CIFColumn(["?"])
        cif_file.block["struct_conn"]["pdbx_ptnr2_leaving_atom_id"] = pdbx.CIFColumn(["?"])
        component_atoms = cif_file.block["chem_comp_atom"]
        component_atoms["pdbx_leaving_atom_flag"] = pdbx.CIFColumn(
            ["N" if atom_id == "C" else "Y" for atom_id in component_atoms["atom_id"].as_array(str)]
        )
        cif_file.write(str(path))

        inferred = read_structure(path)
        assert inferred.system.atomic_number.numel() == 2

        component_atoms["pdbx_leaving_atom_flag"] = pdbx.CIFColumn(["N"] * component_atoms.row_count)
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="cannot identify endpoint 1's leaving group"):
            read_structure(path)

        cif_file.block["struct_conn"]["pdbx_ptnr1_leaving_atom_id"] = pdbx.CIFColumn(["O1"])
        cif_file.block["struct_conn"]["pdbx_ptnr2_leaving_atom_id"] = pdbx.CIFColumn(["O2"])
        component_atoms["pdbx_leaving_atom_flag"] = pdbx.CIFColumn(
            ["N" if atom_id == "O1" else "?" for atom_id in component_atoms["atom_id"].as_array(str)]
        )
        cif_file.write(str(path))
        with pytest.raises(ValueError, match="declares known non-leaving atom 'O1' as leaving"):
            read_structure(path)

    def test_exchangeable_hydrogen(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        donor = ChemicalComponent(
            "DNR",
            (
                ChemicalAtom("N", "N"),
                ChemicalAtom("H1", "H"),
                ChemicalAtom("H2", "H"),
            ),
            frozenset({ChemicalBond("N", "H1"), ChemicalBond("N", "H2")}),
            leaving_atom_ids=frozenset({"H2"}),
            non_leaving_atom_ids=frozenset({"N", "H1"}),
        )
        acceptor = ChemicalComponent(
            "ACC",
            (ChemicalAtom("C", "C"),),
            frozenset(),
            non_leaving_atom_ids=frozenset({"C"}),
        )
        system = MolecularSystem.from_entities(
            {"donor": Molecule((donor,)), "acceptor": Molecule((acceptor,))},
            {"A": "donor", "B": "acceptor"},
        )
        structure = Structure(
            system=system,
            coordinates=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            coordinate_mask=torch.tensor([True, False, True, True]),
            occupancy=torch.ones(4),
            occupancy_mask=torch.tensor([True, False, True, True]),
            b_factor=torch.zeros(4),
            b_factor_mask=torch.tensor([True, False, True, True]),
            source_symbols=("1",),
            residue_id=torch.tensor([0, 0]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )
        path = tmp_path / "exchangeable-hydrogen.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        component_atoms = cif_file.block["chem_comp_atom"]
        component_atoms["pdbx_leaving_atom_flag"] = pdbx.CIFColumn(
            ["Y" if atom_id == "H2" else "N" for atom_id in component_atoms["atom_id"].as_array(str)]
        )
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                "id": ["1"],
                "conn_type_id": ["covale"],
                "pdbx_leaving_atom_flag": ["one"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["."],
                "ptnr1_label_comp_id": ["DNR"],
                "ptnr1_label_atom_id": ["N"],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["B"],
                "ptnr2_label_seq_id": ["."],
                "ptnr2_label_comp_id": ["ACC"],
                "ptnr2_label_atom_id": ["C"],
                "ptnr2_symmetry": ["1_555"],
                "pdbx_value_order": ["sing"],
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path)

        atom_ids = [restored.system.symbols[int(index)] for index in restored.system.atom_symbol]
        assert atom_ids == ["N", "H2", "C"]
        assert int((restored.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == 1

    def test_polymer_attachment(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        residues = PolymerSequence.from_one_letter("AA").components
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", residues, linked=(False,))},
            {"A": "protein"},
        )
        observed_sites = {
            AtomSite("A", 0, "C"),
            AtomSite("A", 1, "N"),
            AtomSite("A", 1, "H2"),
        }
        structure = Structure(
            system=system,
            coordinates=torch.zeros(system.atom_symbol.numel(), 3),
            coordinate_mask=torch.tensor(
                [system.atom_site(atom_index) in observed_sites for atom_index in range(system.atom_symbol.numel())]
            ),
            occupancy=torch.zeros(system.atom_symbol.numel()),
            occupancy_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
            b_factor=torch.zeros(system.atom_symbol.numel()),
            b_factor_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )
        path = tmp_path / "missing-polymer-order.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                "id": ["1"],
                "conn_type_id": ["covale"],
                "pdbx_leaving_atom_flag": ["both"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["1"],
                "ptnr1_label_comp_id": ["ALA"],
                "ptnr1_label_atom_id": ["C"],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["A"],
                "ptnr2_label_seq_id": ["2"],
                "ptnr2_label_comp_id": ["ALA"],
                "ptnr2_label_atom_id": ["N"],
                "ptnr2_symmetry": ["1_555"],
                "pdbx_value_order": ["?"],
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path)
        polymer_bond = restored.system.bond_role == BOND_ROLE_POLYMER
        restored_sites = {
            restored.system.atom_site(atom_index) for atom_index in range(restored.system.atom_symbol.numel())
        }

        assert int(polymer_bond.sum()) == 1
        assert restored.system.bond_order_known[polymer_bond].tolist() == [True]
        assert restored.system.bond_order[polymer_bond].tolist() == [1.0]
        assert AtomSite("A", 1, "H2") in restored_sites
        assert AtomSite("A", 1, "H") not in restored_sites

        cif_file.block["struct_conn"]["pdbx_leaving_atom_flag"] = pdbx.CIFColumn(["none"])
        cif_file.write(str(path))
        without_leaving = read_structure(path)
        without_leaving_sites = {
            without_leaving.system.atom_site(atom_index)
            for atom_index in range(without_leaving.system.atom_symbol.numel())
        }
        assert AtomSite("A", 1, "H") in without_leaving_sites

    def test_altloc(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "connection-altloc.cif"
        write_structure(_two_component_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["atom_site"]["label_alt_id"] = pdbx.CIFColumn(["A", "A"])
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                "id": ["1"],
                "conn_type_id": ["covale"],
                "pdbx_leaving_atom_flag": ["none"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["."],
                "ptnr1_label_comp_id": ["L1"],
                "ptnr1_label_atom_id": ["C"],
                "pdbx_ptnr1_label_alt_id": ["B"],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["B"],
                "ptnr2_label_seq_id": ["."],
                "ptnr2_label_comp_id": ["L2"],
                "ptnr2_label_atom_id": ["C"],
                "ptnr2_symmetry": ["1_555"],
                "pdbx_value_order": ["sing"],
            }
        )
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="cannot resolve explicit alternate 'B' in source atom_site"):
            read_structure(path)

        cif_file.block["struct_conn"]["pdbx_ptnr1_label_alt_id"] = pdbx.CIFColumn(["A"])
        cif_file.write(str(path))
        matched = read_structure(path)
        assert (matched.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1

        cif_file.block["struct_conn"]["ptnr1_label_atom_id"] = pdbx.CIFColumn(["O1"])
        cif_file.block["struct_conn"]["pdbx_ptnr1_label_alt_id"] = pdbx.CIFColumn(["?"])
        cif_file.block["struct_conn"]["ptnr1_symmetry"] = pdbx.CIFColumn(["1_555"])
        cif_file.block["struct_conn"]["ptnr2_symmetry"] = pdbx.CIFColumn(["2_555"])
        cif_file.block["space_group_symop"] = pdbx.CIFCategory(
            {
                "id": ["1", "2"],
                "operation_xyz": ["1-x,y,z", "x,y,z"],
            }
        )
        cif_file.block["atom_sites"] = pdbx.CIFCategory(
            {
                "fract_transf_matrix[1][1]": [0.1],
                "fract_transf_matrix[1][2]": [0.0],
                "fract_transf_matrix[1][3]": [0.0],
                "fract_transf_matrix[2][1]": [0.0],
                "fract_transf_matrix[2][2]": [0.1],
                "fract_transf_matrix[2][3]": [0.0],
                "fract_transf_matrix[3][1]": [0.0],
                "fract_transf_matrix[3][2]": [0.0],
                "fract_transf_matrix[3][3]": [0.1],
                "fract_transf_vector[1]": [0.2],
                "fract_transf_vector[2]": [0.0],
                "fract_transf_vector[3]": [0.0],
            }
        )
        cif_file.write(str(path))

        crystallographic = read_structure(path)
        assert crystallographic.system.instance_symbol.numel() == 3
        assert crystallographic.coordinates[crystallographic.coordinate_mask, 0].tolist() == [0.0, 2.0, 6.0]
        assert (crystallographic.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1

        cif_file.block["pdbx_struct_assembly"] = pdbx.CIFCategory({"id": ["1"], "details": ["test"]})
        cif_file.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
            {
                "assembly_id": ["1", "1"],
                "oper_expression": ["2", "1"],
                "asym_id_list": ["A", "B"],
            }
        )
        cif_file.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(
            {
                "id": ["1", "2"],
                "matrix[1][1]": [1.0, -1.0],
                "matrix[1][2]": [0.0, 0.0],
                "matrix[1][3]": [0.0, 0.0],
                "matrix[2][1]": [0.0, 0.0],
                "matrix[2][2]": [1.0, 1.0],
                "matrix[2][3]": [0.0, 0.0],
                "matrix[3][1]": [0.0, 0.0],
                "matrix[3][2]": [0.0, 0.0],
                "matrix[3][3]": [1.0, 1.0],
                "vector[1]": [0.0, 6.0],
                "vector[2]": [0.0, 0.0],
                "vector[3]": [0.0, 0.0],
            }
        )
        cif_file.write(str(path))

        assembly = read_structure(path, assembly_id="1")
        assert assembly.system.instance_symbol.numel() == 2
        assert assembly.coordinates[assembly.coordinate_mask, 0].tolist() == [6.0, 2.0]
        assert (assembly.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1

    def test_symmetry_operator_one_is_identity_without_table(self) -> None:
        # Operator id "1" is, by the International Tables / PDB convention, always the
        # identity x,y,z and is listed first in every _space_group_symop table. Real PDB
        # mmCIF files routinely omit the table and give only the space-group number, yet
        # still reference "1_555" / "1_klm" partner codes, so a bare "1" must resolve to
        # the identity rather than fail for lack of an explicit table.
        assert np.array_equal(_mmcif_symmetry_operator({}, "1"), np.eye(4, dtype=np.float64))

    def test_symmetry_operator_missing_non_identity_still_raises(self) -> None:
        # Operators 2..N cannot be recovered without a table (gemmi's operator ordering
        # does not match the PDB n_klm numbering for centred / higher-symmetry groups),
        # so a missing non-identity operator must still be rejected.
        with pytest.raises(ValueError, match="unknown operation '2'"):
            _mmcif_symmetry_operator({}, "2")

    def test_symmetry_operator_prefers_tabulated_value(self) -> None:
        # A shipped table takes precedence and is returned unchanged (behaviour-preserving
        # for well-formed files); the returned array is a copy, not an alias into the table.
        tabulated = np.eye(4, dtype=np.float64)
        tabulated[0, 3] = 0.5
        resolved = _mmcif_symmetry_operator({"1": tabulated}, "1")
        assert np.array_equal(resolved, tabulated)
        resolved[0, 3] = 999.0
        assert tabulated[0, 3] == 0.5

    def test_operator_one_crystal_contact_without_symop_table(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        # A covalent crystal contact that references operator "1" with a lattice
        # translation (1_556) while the file ships no _space_group_symop table. Before
        # the identity-seed fix this aborted the whole read with "unknown operation '1'";
        # it must now resolve to identity + lattice shift and form the assembly bond.
        path = tmp_path / "operator-one-contact.cif"
        write_structure(_two_component_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                "id": ["1"],
                "conn_type_id": ["covale"],
                "pdbx_leaving_atom_flag": ["none"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["."],
                "ptnr1_label_comp_id": ["L1"],
                "ptnr1_label_atom_id": ["C"],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["B"],
                "ptnr2_label_seq_id": ["."],
                "ptnr2_label_comp_id": ["L2"],
                "ptnr2_label_atom_id": ["C"],
                "ptnr2_symmetry": ["1_556"],
                "pdbx_value_order": ["sing"],
            }
        )
        cif_file.block["atom_sites"] = pdbx.CIFCategory(
            {
                "fract_transf_matrix[1][1]": [0.1],
                "fract_transf_matrix[1][2]": [0.0],
                "fract_transf_matrix[1][3]": [0.0],
                "fract_transf_matrix[2][1]": [0.0],
                "fract_transf_matrix[2][2]": [0.1],
                "fract_transf_matrix[2][3]": [0.0],
                "fract_transf_matrix[3][1]": [0.0],
                "fract_transf_matrix[3][2]": [0.0],
                "fract_transf_matrix[3][3]": [0.1],
                "fract_transf_vector[1]": [0.0],
                "fract_transf_vector[2]": [0.0],
                "fract_transf_vector[3]": [0.0],
            }
        )
        cif_file.write(str(path))

        contact = read_structure(path)
        assert (contact.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1

    def test_unresolvable_operator_bond_is_rejected(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        # Without the numbered operation, the connected atom cannot be placed.
        # Never silently remove a declared covalent connection.
        path = tmp_path / "unresolvable-operator.cif"
        write_structure(_two_component_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                "id": ["1"],
                "conn_type_id": ["covale"],
                "pdbx_leaving_atom_flag": ["none"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["."],
                "ptnr1_label_comp_id": ["L1"],
                "ptnr1_label_atom_id": ["C"],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["B"],
                "ptnr2_label_seq_id": ["."],
                "ptnr2_label_comp_id": ["L2"],
                "ptnr2_label_atom_id": ["C"],
                "ptnr2_symmetry": ["2_555"],
                "pdbx_value_order": ["sing"],
            }
        )
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="unknown operation '2'"):
            read_structure(path)

    def test_reduced_cysteine_disulfide_keeps_modelled_thiol(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        # A declared disulfide whose cysteines still model the thiol hydrogen HG (a reduced
        # structure whose "disulfide" is not a real S-S contact). HG is an observation, not a
        # leaving atom, so the read must not abort with "leaving group contains observed atom
        # 'HG'": the modelled HG is kept and the S-S assembly bond is still formed.
        structure = Structure.from_sequence("CC")
        structure = structure.replace(coordinate_mask=torch.ones_like(structure.coordinate_mask))
        atom_count = int(structure.system.atom_symbol.numel())
        path = tmp_path / "reduced-disulfide.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"] = pdbx.CIFCategory(
            {
                "id": ["1"],
                "conn_type_id": ["disulf"],
                "pdbx_leaving_atom_flag": ["?"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["1"],
                "ptnr1_label_comp_id": ["CYS"],
                "ptnr1_label_atom_id": ["SG"],
                "ptnr1_symmetry": ["1_555"],
                "ptnr2_label_asym_id": ["A"],
                "ptnr2_label_seq_id": ["2"],
                "ptnr2_label_comp_id": ["CYS"],
                "ptnr2_label_atom_id": ["SG"],
                "ptnr2_symmetry": ["1_555"],
                "pdbx_value_order": ["sing"],
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path)
        assert int(restored.system.atom_symbol.numel()) == atom_count  # HG retained, nothing deleted
        assert (restored.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1  # S-S bond formed

    def test_topology_partner(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        components = {
            "DUM": ChemicalComponent(
                "DUM",
                (ChemicalAtom("C", "C", formal_charge=0),),
                frozenset(),
            ),
            "LIG": ChemicalComponent(
                "LIG",
                (ChemicalAtom("O", "O", formal_charge=-1),),
                frozenset(),
            ),
            "NA": ChemicalComponent(
                "NA",
                (ChemicalAtom("NA", "Na", formal_charge=1),),
                frozenset(),
            ),
        }
        system = MolecularSystem.from_entities(
            {component_id: Molecule((component,)) for component_id, component in components.items()},
            {"D": "DUM", "L": "LIG", "M": "NA"},
            (
                SystemBond(
                    AtomSite("L", 0, "O"),
                    AtomSite("M", 0, "NA"),
                    kind="dative",
                    direction="a_to_b",
                ),
            ),
        )
        atom_count = int(system.atom_symbol.numel())
        structure = Structure(
            system=system,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=torch.tensor([True, False, False]),
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1", "2", "3"),
            residue_id=torch.tensor([0, 1, 2]),
            insertion_code=torch.full((3,), -1),
            batch_size=[],
        )
        path = tmp_path / "topology-partner.cif"
        write_structure(structure, path)

        cif_file = pdbx.CIFFile.read(str(path))
        connections = cif_file.block["struct_conn"]
        for endpoint in (1, 2):
            asym_id = str(connections[f"ptnr{endpoint}_label_asym_id"].as_array()[0])
            connections[f"ptnr{endpoint}_symmetry"] = pdbx.CIFColumn(["2_555" if asym_id == "L" else "1_555"])
            connections[f"ptnr{endpoint}_role"] = pdbx.CIFColumn(["?"])
        cif_file.block["space_group_symop"] = pdbx.CIFCategory(
            {
                "id": ["1", "2"],
                "operation_xyz": ["x,y,z", "x+1,y,z"],
            }
        )
        fractional = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        cif_file.block["atom_sites"] = pdbx.CIFCategory(
            {
                **{
                    f"fract_transf_matrix[{row}][{column}]": [fractional[(row - 1) * 3 + column - 1]]
                    for row in (1, 2, 3)
                    for column in (1, 2, 3)
                },
                **{f"fract_transf_vector[{row}]": [fractional[8 + row]] for row in (1, 2, 3)},
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path, components=components)
        instance_ids = [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())]
        copied_instance = "L#2_555"
        assert copied_instance in instance_ids
        copied_index = instance_ids.index(copied_instance)
        component_start = int(restored.system.instance_component_ptr[copied_index])
        component_stop = int(restored.system.instance_component_ptr[copied_index + 1])
        atom_start = int(restored.system.component_atom_ptr[component_start])
        atom_stop = int(restored.system.component_atom_ptr[component_stop])
        assembly_bonds = torch.nonzero(restored.system.bond_role == BOND_ROLE_ASSEMBLY, as_tuple=False).flatten()

        assert len(instance_ids) == 4
        assert restored.coordinate_mask.sum().item() == 1
        assert not restored.coordinate_mask[atom_start:atom_stop].any()
        assert assembly_bonds.numel() == 1
        bond_index = int(assembly_bonds[0])
        endpoints = {
            restored.system.atom_site(int(atom)).instance_id for atom in restored.system.edge_index[:, bond_index]
        }
        assert copied_instance in endpoints
        assert not restored.coordinate_mask[restored.system.edge_index[:, bond_index]].any()
        assert int(restored.system.bond_direction[bond_index]) != 0

    def test_crystal_copy_closure(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        anchor = ChemicalComponent("ANC", (ChemicalAtom("C", "C"),), frozenset())
        seed_partner = ChemicalComponent("SED", (ChemicalAtom("O", "O"),), frozenset())
        residue = ChemicalComponent(
            "RES",
            (ChemicalAtom("N", "N"), ChemicalAtom("C", "C")),
            frozenset({ChemicalBond("N", "C")}),
        )
        components = {component.component_id: component for component in (anchor, seed_partner, residue)}
        system = MolecularSystem.from_entities(
            {
                "anchor": Molecule((anchor,)),
                "seed": Molecule((seed_partner,)),
                "protein": PolymerSequence("protein", (residue,)),
            },
            {"A": "anchor", "C": "seed", "B": "protein"},
            (
                SystemBond(AtomSite("A", 0, "C"), AtomSite("B", 0, "N")),
                SystemBond(AtomSite("A", 0, "C"), AtomSite("C", 0, "O")),
            ),
        )
        structure = Structure(
            system=system,
            coordinates=torch.zeros((4, 3)),
            coordinate_mask=torch.tensor([True, True, False, False]),
            occupancy=torch.zeros(4),
            occupancy_mask=torch.zeros(4, dtype=torch.bool),
            b_factor=torch.zeros(4),
            b_factor_mask=torch.zeros(4, dtype=torch.bool),
            source_symbols=("1", "2", "3"),
            residue_id=torch.tensor([0, 1, 2]),
            insertion_code=torch.full((3,), -1),
            batch_size=[],
        )
        path = tmp_path / "topology-polymer-closure.cif"
        write_structure(structure, path)

        cif_file = pdbx.CIFFile.read(str(path))
        connections = cif_file.block["struct_conn"]
        for row in range(connections.row_count):
            asym_ids = tuple(str(connections[f"ptnr{endpoint}_label_asym_id"].as_array()[row]) for endpoint in (1, 2))
            if set(asym_ids) != {"A", "C"}:
                continue
            for endpoint, asym_id in enumerate(asym_ids, start=1):
                if asym_id == "A":
                    values = connections[f"ptnr{endpoint}_symmetry"].as_array()
                    values[row] = "2_555"
                    connections[f"ptnr{endpoint}_symmetry"] = pdbx.CIFColumn(values)
        cif_file.block["space_group_symop"] = pdbx.CIFCategory(
            {
                "id": ["1", "2"],
                "operation_xyz": ["x,y,z", "x+1,y,z"],
            }
        )
        fractional = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        cif_file.block["atom_sites"] = pdbx.CIFCategory(
            {
                **{
                    f"fract_transf_matrix[{row}][{column}]": [fractional[(row - 1) * 3 + column - 1]]
                    for row in (1, 2, 3)
                    for column in (1, 2, 3)
                },
                **{f"fract_transf_vector[{row}]": [fractional[8 + row]] for row in (1, 2, 3)},
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path, components=components)
        instance_ids = [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())]
        copied_anchor = next(instance_id for instance_id in instance_ids if instance_id.startswith("A#2_555"))
        copied_polymer = next(instance_id for instance_id in instance_ids if instance_id.startswith("B#2_555"))
        assembly_bonds = torch.nonzero(restored.system.bond_role == BOND_ROLE_ASSEMBLY, as_tuple=False).flatten()
        endpoint_pairs = [
            {
                restored.system.atom_site(int(atom)).instance_id
                for atom in restored.system.edge_index[:, int(bond_index)]
            }
            for bond_index in assembly_bonds
        ]

        assert len(instance_ids) == 5
        assert {copied_anchor, copied_polymer} in endpoint_pairs
        assert restored.coordinate_mask.sum().item() == 3

    def test_known_leaving_groups(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        protein = PolymerSequence.from_one_letter("AA")
        system = MolecularSystem.from_entities(
            {"protein": protein},
            {"A": "protein", "B": "protein"},
            (
                SystemBond(
                    AtomSite("A", 1, "C"),
                    AtomSite("B", 0, "N"),
                    leaving_a=frozenset({"OXT", "HXT"}),
                    leaving_b=frozenset({"H2"}),
                ),
            ),
        )
        atom_count = int(system.atom_symbol.numel())
        structure = Structure(
            system=system,
            coordinates=torch.arange(atom_count, dtype=torch.float32).unsqueeze(-1).repeat(1, 3),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.ones(atom_count),
            occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
            source_symbols=(),
            residue_id=torch.full((4,), -1),
            insertion_code=torch.full((4,), -1),
            batch_size=[],
        )
        path = tmp_path / "interchain-link.cif"
        write_structure(structure, path)

        cif_file = pdbx.CIFFile.read(str(path))
        block = cif_file.block
        connections = block["struct_conn"]
        assert (system.bond_role == BOND_ROLE_POLYMER).sum().item() == 2
        assert connections.row_count == 1
        assert connections["pdbx_leaving_atom_flag"].as_array().tolist() == ["both"]
        assert connections["pdbx_ptnr1_leaving_atom_id"].as_array().tolist() == ["OXT"]
        assert connections["pdbx_ptnr2_leaving_atom_id"].as_array().tolist() == ["H2"]

        restored = read_structure(path)
        assert restored.system.symbols == system.symbols
        for field in MolecularSystem.__annotations__:
            if field != "symbols":
                assert torch.equal(getattr(restored.system, field), getattr(system, field))

        connections["pdbx_ptnr1_leaving_atom_id"] = pdbx.CIFColumn(["?"])
        connections["pdbx_ptnr2_leaving_atom_id"] = pdbx.CIFColumn(["?"])
        cif_file.write(str(path))
        inferred = read_structure(path)
        for field in MolecularSystem.__annotations__:
            if field != "symbols":
                assert torch.equal(getattr(inferred.system, field), getattr(system, field))


class TestMmcifMetadata:
    @pytest.mark.parametrize(
        ("values", "mask", "occupancy"),
        (
            (None, [True, True], [1.0, 1.0]),
            (("?", "0.5"), [False, True], [0.0, 0.5]),
        ),
    )
    def test_occupancy(
        self,
        values: tuple[str, str] | None,
        mask: list[bool],
        occupancy: list[float],
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "occupancy.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        if values is None:
            cif_file.block["atom_site"] = pdbx.CIFCategory(
                {name: atom_site[name] for name in atom_site if name != "occupancy"}
            )
        else:
            cif_file.block["atom_site"]["occupancy"] = pdbx.CIFColumn(values)
        cif_file.write(str(path))

        structure = read_structure(path)
        assert structure.occupancy_mask.tolist() == mask
        assert structure.occupancy.tolist() == occupancy

    def test_nonpoly_occurrences(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        structure_path = tmp_path / "nonpoly-occurrences.cif"
        write_structure(_structure(), structure_path)
        cif_file = pdbx.CIFFile.read(str(structure_path))
        scheme = cif_file.block["pdbx_nonpoly_scheme"]
        columns = {name: [str(scheme[name].as_array()[0])] * 2 for name in scheme}
        columns["ndb_seq_num"] = ["alpha-1", "beta-2"]
        columns["pdb_seq_num"] = ["11", "12"]
        columns["auth_seq_num"] = ["11", "12"]
        columns["pdb_ins_code"] = [".", "A"]
        cif_file.block["pdbx_nonpoly_scheme"] = pdbx.CIFCategory(columns)
        cif_file.write(str(structure_path))

        restored = read_structure(structure_path)
        assert restored.system.instance_symbol.numel() == 2
        assert restored.coordinate_mask.sum().item() == 2
        assert [restored.source_symbols[index] for index in restored.residue_id.tolist()] == ["11", "12"]

    def test_missing_author_number(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        no_author_path = tmp_path / "nonpoly-without-author-number.cif"
        write_structure(_structure(), no_author_path)
        cif_file = pdbx.CIFFile.read(str(no_author_path))
        cif_file.block["pdbx_nonpoly_scheme"]["pdb_seq_num"] = pdbx.CIFColumn(["?"])
        cif_file.block["pdbx_nonpoly_scheme"]["auth_seq_num"] = pdbx.CIFColumn(["?"])
        cif_file.write(str(no_author_path))

        no_author = read_structure(no_author_path)
        assert no_author.system.instance_symbol.numel() == 1
        assert no_author.coordinate_mask.sum().item() == 2

    def test_nonpoly_entity_conflict(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        entity_conflict = tmp_path / "nonpoly-entity-conflict.cif"
        write_structure(_structure(), entity_conflict)
        cif_file = pdbx.CIFFile.read(str(entity_conflict))
        cif_file.block["struct_asym"]["entity_id"] = pdbx.CIFColumn(["different"])
        cif_file.write(str(entity_conflict))
        with pytest.raises(ValueError, match="references undeclared entity"):
            read_structure(entity_conflict)

    def test_nonpoly_component_conflict(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        component_conflict = tmp_path / "nonpoly-component-conflict.cif"
        write_structure(_structure(), component_conflict)
        cif_file = pdbx.CIFFile.read(str(component_conflict))
        cif_file.block["pdbx_entity_nonpoly"]["comp_id"] = pdbx.CIFColumn(["OTHER"])
        cif_file.write(str(component_conflict))
        with pytest.raises(ValueError, match="disagrees with its component"):
            read_structure(component_conflict)

    def test_macrolide_round_trip(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "macrolide.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["entity"]["type"] = pdbx.CIFColumn(["macrolide"])
        cif_file.write(str(path))

        structure = read_structure(path)
        assert structure.system.entity_molecule_type.tolist() == [MOLECULE_MACROLIDE]

        output = tmp_path / "macrolide-output.cif"
        write_structure(structure, output)
        written = pdbx.CIFFile.read(str(output))
        assert written.block["entity"]["type"].as_array(str).tolist() == ["macrolide"]

        restored = read_structure(output)
        assert restored.system.entity_molecule_type.tolist() == [MOLECULE_MACROLIDE]
        assert torch.equal(restored.system.edge_index, structure.system.edge_index)
        assert structure.coordinates.dtype == torch.float64
        torch.testing.assert_close(restored.coordinates, structure.coordinates, rtol=0, atol=0)

    def test_macrolide_assembly(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "macrolide.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["entity"]["type"] = pdbx.CIFColumn(["macrolide"])
        cif_file.write(str(path))
        structure = read_structure(path)
        output = tmp_path / "macrolide-output.cif"
        write_structure(structure, output)
        written = pdbx.CIFFile.read(str(output))

        del written.block["atom_site"]
        written.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
            {"assembly_id": ["1"], "oper_expression": ["1,2"], "asym_id_list": ["L"]}
        )
        written.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(
            {
                "id": ["1", "2"],
                "matrix[1][1]": [1.0, 1.0],
                "matrix[1][2]": [0.0, 0.0],
                "matrix[1][3]": [0.0, 0.0],
                "matrix[2][1]": [0.0, 0.0],
                "matrix[2][2]": [1.0, 1.0],
                "matrix[2][3]": [0.0, 0.0],
                "matrix[3][1]": [0.0, 0.0],
                "matrix[3][2]": [0.0, 0.0],
                "matrix[3][3]": [1.0, 1.0],
                "vector[1]": [0.0, 10.0],
                "vector[2]": [0.0, 0.0],
                "vector[3]": [0.0, 0.0],
            }
        )
        topology = tmp_path / "macrolide-topology.cif"
        written.write(str(topology))

        assembly = read_structure(topology, assembly_id="1")
        assert assembly.system.instance_symbol.numel() == 2
        assert assembly.system.entity_molecule_type.tolist() == [MOLECULE_MACROLIDE]
        assert not assembly.coordinate_mask.any()
