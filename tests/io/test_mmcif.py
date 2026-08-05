from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ofoldx.data import (
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_POLYMER,
    MOLECULE_BRANCHED,
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerAttachment,
    PolymerSequence,
    Structure,
    SystemBond,
    encode_system,
    get_component,
)
from ofoldx.io import read_structure, write_structure
from ofoldx.io.mmcif import (
    _cif_bond_order,
    _cif_component_bond_order,
    _read_mmcif_entity_declarations,
    _remove_component_atoms,
    parse_mmcif,
    read_mmcif_bonds,
    read_mmcif_branch_entities,
    read_mmcif_components,
)
from ofoldx.io.structure import (
    NoRetainedStructureError,
    _canonical_polymer_leaving_sites,
    _resolve_branch_component_index,
    _select_altloc,
)


def _ligand_component(
    *,
    bond_order: float | None = 1.0,
    bond_aromatic: bool | None = False,
) -> ChemicalComponent:
    return ChemicalComponent(
        "LIG",
        (
            ChemicalAtom("C1", "C", formal_charge=0),
            ChemicalAtom("O1", "O", formal_charge=-1),
        ),
        frozenset({ChemicalBond("C1", "O1", order=bond_order, aromatic=bond_aromatic)}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
    )


def _water_component() -> ChemicalComponent:
    return ChemicalComponent(
        "HOH",
        (
            ChemicalAtom("O", "O"),
            ChemicalAtom("H1", "H"),
            ChemicalAtom("H2", "H"),
        ),
        frozenset({ChemicalBond("O", "H1"), ChemicalBond("O", "H2")}),
        reference_coordinates=((0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)),
    )


def _structure(
    *,
    bond_order: float | None = 1.0,
    bond_aromatic: bool | None = False,
) -> Structure:
    component = _ligand_component(bond_order=bond_order, bond_aromatic=bond_aromatic)
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


def _read_cif(path: Path, text: str) -> object:
    from biotite.structure.io import pdbx

    path.write_text(text)
    return pdbx.CIFFile.read(str(path))


def _microheterogeneous_polymer_components() -> dict[str, ChemicalComponent]:
    r"""Return tiny, fully observed polymer definitions without consulting CCD."""

    return {
        component_id: ChemicalComponent(
            component_id,
            (
                ChemicalAtom("N", "N"),
                ChemicalAtom("C", "C"),
            ),
            frozenset({ChemicalBond("N", "C")}),
            previous_attachment=PolymerAttachment("N"),
            next_attachment=PolymerAttachment("C"),
            reference_coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )
        for component_id in ("ALA", "GLY", "HYP", "PRO", "VAL")
    }


def _write_microheterogeneous_polymer_cif(
    path: Path,
    *,
    chain_ids: tuple[str, ...] = ("A",),
    include_assembly: bool = False,
) -> Path:
    r"""Write a 3a0m-shaped two-residue A/B polymer microheterogeneity fixture."""

    atom_rows: list[str] = []
    scheme_rows: list[str] = []
    serial = 1
    for chain_ordinal, chain_id in enumerate(chain_ids):
        for label_seq_id in (1, 2):
            for component_id, altloc_id, author_offset in (("HYP", "A", 12), ("PRO", "B", 112)):
                author_residue_id = author_offset + label_seq_id
                scheme_rows.append(
                    f"{chain_id} 1 {label_seq_id} {component_id} {author_residue_id} "
                    f"{author_residue_id} ? y"
                )
                for atom_ordinal, (atom_name, element) in enumerate((("N", "N"), ("C", "C"))):
                    x = 100.0 * (component_id == "PRO") + 10.0 * chain_ordinal + label_seq_id
                    y = float(atom_ordinal)
                    atom_rows.append(
                        f"ATOM {serial} {element} {atom_name} {altloc_id} {component_id} {chain_id} 1 "
                        f"{label_seq_id} ? {x:.1f} {y:.1f} 0.0 0.5 10.0 {author_residue_id} "
                        f"{component_id} {chain_id} {atom_name} 1"
                    )
                    serial += 1

    assembly = ""
    if include_assembly:
        assembly = f"""
loop_
_pdbx_struct_assembly.id
_pdbx_struct_assembly.details
1 'two symmetry copies'
loop_
_pdbx_struct_assembly_gen.assembly_id
_pdbx_struct_assembly_gen.oper_expression
_pdbx_struct_assembly_gen.asym_id_list
1 '(1,2)' {chain_ids[0]}
loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.type
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
1 identity 1 0 0 0 0 1 0 0 0 0 1 0
2 'crystal symmetry operation' 1 0 0 20 0 1 0 0 0 0 1 0
"""

    path.write_text(
        f"""data_microheterogeneity
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
{chr(10).join(f'{chain_id} 1' for chain_id in chain_ids)}
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
_pdbx_poly_seq_scheme.hetero
{chr(10).join(scheme_rows)}
{assembly}
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
{chr(10).join(atom_rows)}
""",
        encoding="ascii",
    )
    return path


def _instance_component_ids(structure: Structure, instance_id: str) -> tuple[str, ...]:
    instance_index = next(
        index
        for index in range(int(structure.system.instance_symbol.numel()))
        if structure.system.instance_id(index) == instance_id
    )
    start = int(structure.system.instance_component_ptr[instance_index])
    stop = int(structure.system.instance_component_ptr[instance_index + 1])
    return tuple(
        structure.system.symbols[int(structure.system.component_symbol[index])]
        for index in range(start, stop)
    )


def _instance_entity_index(structure: Structure, instance_id: str) -> int:
    instance_index = next(
        index
        for index in range(int(structure.system.instance_symbol.numel()))
        if structure.system.instance_id(index) == instance_id
    )
    return int(structure.system.instance_entity[instance_index])


def _instance_residue_numbers(structure: Structure, instance_id: str) -> tuple[str, ...]:
    instance_index = next(
        index
        for index in range(int(structure.system.instance_symbol.numel()))
        if structure.system.instance_id(index) == instance_id
    )
    start = int(structure.system.instance_component_ptr[instance_index])
    stop = int(structure.system.instance_component_ptr[instance_index + 1])
    return tuple(
        structure.source_symbols[int(structure.residue_id[index])]
        for index in range(start, stop)
    )


def _write_3a0m_shaped_polymer_block_cif(path: Path, *, link_blocks: bool = True) -> Path:
    r"""Write exact 3a0m-shaped 13--17 A/B conformers, optionally linked by GLY altlocs."""

    entity_rows = [f"1 {number} ALA n" for number in range(1, 13)]
    entity_rows.extend(
        (
            "1 13 HYP y",
            "1 13 PRO y",
            "1 14 VAL y",
            "1 14 PRO y",
            "1 15 GLY n",
            "1 16 PRO y",
            "1 16 HYP y",
            "1 17 PRO y",
            "1 17 VAL y",
        )
    )
    scheme_rows = [f"A 1 {number} ALA {number} {number} ? n" for number in range(1, 13)]
    atom_rows: list[str] = []
    serial = 1
    candidates_by_number = {
        13: (("HYP", "A", 13), ("PRO", "B", 113)),
        14: (("VAL", "A", 14), ("PRO", "B", 114)),
        15: (("GLY", "A", 15), ("GLY", "B", 15)) if link_blocks else (("GLY", None, 15),),
        16: (("PRO", "A", 16), ("HYP", "B", 116)),
        17: (("PRO", "A", 17), ("VAL", "B", 117)),
    }
    for number, candidates in candidates_by_number.items():
        if number == 15:
            scheme_rows.append("A 1 15 GLY 15 15 ? n")
        else:
            for component_id, _label, author_number in candidates:
                scheme_rows.append(
                    f"A 1 {number} {component_id} {author_number} {author_number} ? y"
                )
        for component_id, label, author_number in candidates:
            for atom_ordinal, (atom_name, element) in enumerate((("N", "N"), ("C", "C"))):
                x = number + (100.0 if label == "B" else 0.0)
                atom_rows.append(
                    f"ATOM {serial} {element} {atom_name} {label or '.'} {component_id} A 1 {number} ? "
                    f"{x:.1f} {float(atom_ordinal):.1f} 0.0 0.5 10.0 {author_number} "
                    f"{component_id} A {atom_name} 1"
                )
                serial += 1
    path.write_text(
        f"""data_3a0m_shaped_microheterogeneity
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
{chr(10).join(entity_rows)}
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
_pdbx_poly_seq_scheme.hetero
{chr(10).join(scheme_rows)}
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
{chr(10).join(atom_rows)}
""",
        encoding="ascii",
    )
    return path


def test_remove_component_atoms_without_topology_removes_only_declared_atoms() -> None:
    component = ChemicalComponent(
        "LEFT",
        (
            ChemicalAtom("A", "C"),
            ChemicalAtom("L", "O"),
            ChemicalAtom("H", "H"),
        ),
        None,
        previous_attachment=PolymerAttachment("A", frozenset({"L"})),
        next_attachment=PolymerAttachment("L"),
        reference_coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )

    retained = _remove_component_atoms(component, {"L"})

    assert tuple(atom.atom_id for atom in retained.atoms) == ("A", "H")
    assert retained.reference_coordinates == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert retained.bonds is None
    assert retained.previous_attachment == PolymerAttachment("A")
    assert retained.next_attachment is None


def test_remove_component_atoms_with_known_topology_still_expands_neighbor_hydrogen() -> None:
    component = ChemicalComponent(
        "LEFT",
        (
            ChemicalAtom("A", "C"),
            ChemicalAtom("L", "O"),
            ChemicalAtom("H", "H"),
        ),
        frozenset({ChemicalBond("A", "L"), ChemicalBond("L", "H")}),
    )

    retained = _remove_component_atoms(component, {"L"})

    assert tuple(atom.atom_id for atom in retained.atoms) == ("A",)
    assert retained.bonds == frozenset()


def test_branch_link_survives_leaving_atom_removal_without_component_topology(tmp_path: Path) -> None:
    cif_file = _read_cif(
        tmp_path / "branch-no-topology.cif",
        """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 LEFT 1
1 RIGHT 2
loop_
_pdbx_entity_branch_link.entity_id
_pdbx_entity_branch_link.entity_branch_list_num_1
_pdbx_entity_branch_link.comp_id_1
_pdbx_entity_branch_link.atom_id_1
_pdbx_entity_branch_link.leaving_atom_id_1
_pdbx_entity_branch_link.entity_branch_list_num_2
_pdbx_entity_branch_link.comp_id_2
_pdbx_entity_branch_link.atom_id_2
_pdbx_entity_branch_link.leaving_atom_id_2
_pdbx_entity_branch_link.value_order
1 1 LEFT A L 2 RIGHT B ? sing
""",
    )
    left = ChemicalComponent(
        "LEFT",
        (ChemicalAtom("A", "C"), ChemicalAtom("L", "O")),
        None,
    )
    right = ChemicalComponent("RIGHT", (ChemicalAtom("B", "N"),), None)

    entities, _ = read_mmcif_branch_entities(cif_file, {"LEFT": left, "RIGHT": right})
    branch = entities["1"]

    assert tuple(atom.atom_id for atom in branch.components[0].atoms) == ("A",)
    assert branch.components[0].bonds is None
    assert len(branch.bonds) == 1


@pytest.mark.parametrize(
    ("bem_occupancy", "reverse_declarations_and_links", "selected_component", "selected_links"),
    [
        (
            0.6,
            False,
            "BEM",
            {(1, "C1", 0, "O4"), (2, "C1", 1, "O4"), (3, "C1", 2, "O4")},
        ),
        (
            0.4,
            True,
            "LGU",
            {(1, "O4", 0, "O4"), (2, "O4", 1, "C1"), (3, "C1", 2, "O4")},
        ),
    ],
)
def test_branch_microheterogeneity_follows_selected_altloc_and_filters_links(
    tmp_path: Path,
    bem_occupancy: float,
    reverse_declarations_and_links: bool,
    selected_component: str,
    selected_links: set[tuple[int, str, int, str]],
) -> None:
    declarations = ["1 LGU 1 n", "1 BEM 2 y", "1 LGU 2 y", "1 LGU 3 n", "1 MAW 4 n"]
    links = [
        "1 2 BEM C1 ? 1 LGU O4 ? sing",
        "1 2 LGU O4 ? 1 LGU O4 ? sing",
        "1 3 LGU C1 ? 2 BEM O4 ? sing",
        "1 3 LGU O4 ? 2 LGU C1 ? sing",
        "1 4 MAW C1 ? 3 LGU O4 ? sing",
    ]
    if reverse_declarations_and_links:
        declarations.reverse()
        links.reverse()
    path = tmp_path / "branch-microheterogeneity.cif"
    path.write_text(
        f"""data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
_pdbx_entity_branch_list.hetero
{chr(10).join(declarations)}
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
B 1 LGU 1 B LGU 1
B 1 BEM 2 B BEM 2
B 1 LGU 2 B LGU 2
B 1 LGU 3 B LGU 3
B 1 MAW 4 B MAW 4
loop_
_pdbx_entity_branch_link.entity_id
_pdbx_entity_branch_link.entity_branch_list_num_1
_pdbx_entity_branch_link.comp_id_1
_pdbx_entity_branch_link.atom_id_1
_pdbx_entity_branch_link.leaving_atom_id_1
_pdbx_entity_branch_link.entity_branch_list_num_2
_pdbx_entity_branch_link.comp_id_2
_pdbx_entity_branch_link.atom_id_2
_pdbx_entity_branch_link.leaving_atom_id_2
_pdbx_entity_branch_link.value_order
{chr(10).join(links)}
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
HETATM 1 O O4 . LGU B 1 . ? 0 0 0 1.0 10 1 LGU B O4 1
HETATM 2 C C1 A BEM B 1 . ? 1 0 0 {bem_occupancy} 10 2 BEM B C1 1
HETATM 3 C C1 B LGU B 1 . ? 1 0 0 {1.0 - bem_occupancy} 10 2 LGU B C1 1
HETATM 4 C C1 . LGU B 1 . ? 2 0 0 1.0 10 3 LGU B C1 1
HETATM 5 C C1 . MAW B 1 . ? 3 0 0 1.0 10 4 MAW B C1 1
"""
    )
    rows, cif_file = parse_mmcif(path, model=1, assembly_id=None)
    selected_rows = _select_altloc(rows, "highest_occupancy")
    components = {
        component_id: ChemicalComponent(
            component_id,
            (ChemicalAtom("C1", "C"), ChemicalAtom("O4", "O")),
            frozenset(),
        )
        for component_id in ("LGU", "BEM", "MAW")
    }

    entities, _ = read_mmcif_branch_entities(
        cif_file,
        components,
        selected_rows=selected_rows,
    )
    branch = entities["1"]

    assert tuple(component.component_id for component in branch.components) == (
        "LGU",
        selected_component,
        "LGU",
        "MAW",
    )
    assert {
        (bond.component_a, bond.atom_a, bond.component_b, bond.atom_b)
        for bond in branch.bonds
    } == selected_links


def test_branch_declaration_exact_duplicate_is_idempotent(tmp_path: Path) -> None:
    cif_file = _read_cif(
        tmp_path / "duplicate-branch-declaration.cif",
        """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 X 1
1 X 1
""",
    )
    component = ChemicalComponent("X", (ChemicalAtom("A", "C"),), frozenset())

    entities, _ = read_mmcif_branch_entities(cif_file, {"X": component})

    assert entities["1"].components == (component,)


def test_branch_nonheterogeneous_conflicting_declarations_fail(tmp_path: Path) -> None:
    cif_file = _read_cif(
        tmp_path / "conflicting-branch-declaration.cif",
        """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
_pdbx_entity_branch_list.hetero
1 X 1 n
1 Y 1 y
""",
    )

    with pytest.raises(ValueError, match="conflicting non-heterogeneous declarations"):
        read_mmcif_branch_entities(cif_file, {})


def test_unobserved_branch_microheterogeneity_is_ambiguous(tmp_path: Path) -> None:
    cif_file = _read_cif(
        tmp_path / "unobserved-branch-microheterogeneity.cif",
        """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
_pdbx_entity_branch_list.hetero
1 X 1 y
1 Y 1 y
""",
    )

    with pytest.raises(ValueError, match="unobserved microheterogeneous.*ambiguous candidates"):
        read_mmcif_branch_entities(cif_file, {})


def test_branch_microheterogeneity_conflicting_instance_selections_fail(tmp_path: Path) -> None:
    cif_file = _read_cif(
        tmp_path / "conflicting-branch-instance-selections.cif",
        """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
_pdbx_entity_branch_list.hetero
1 X 1 y
1 Y 1 y
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
B 1 X 1
B 1 Y 1
C 1 X 1
C 1 Y 1
""",
    )
    selected_rows = [
        SimpleNamespace(
            entity_id="1",
            chain_id=chain_id,
            component_id=component_id,
            auth_component_id=None,
            auth_chain_id=None,
            auth_residue_id=None,
        )
        for chain_id, component_id in (("B", "X"), ("C", "Y"))
    ]

    with pytest.raises(ValueError, match="conflicting selections across source-entity instances"):
        read_mmcif_branch_entities(cif_file, {}, selected_rows=selected_rows)  # type: ignore[arg-type]


def test_struct_conn_explicit_altloc_uses_selected_conformer_identity(tmp_path: Path) -> None:
    from biotite.structure.io import pdbx

    cif_file = _read_cif(
        tmp_path / "struct-conn-altloc.cif",
        """data_connections
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_auth_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_atom_id
_struct_conn.pdbx_ptnr1_label_alt_id
_struct_conn.pdbx_ptnr2_label_alt_id
1 covale B LGU ? O4 B LGU 1 O4 B BEM ? C1 B BEM 2 C1 A A
2 covale B LGU ? O4 B LGU 1 O4 B LGU ? C1 B LGU 2 C1 B B
""",
    )

    def selected_row(row_index: int, auth_residue_id: str, component_id: str, atom_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            row_index=row_index,
            symmetry_id=None,
            chain_id="B",
            label_seq_id=None,
            insertion_code=None,
            component_id=component_id,
            atom_name=atom_name,
            auth_chain_id="B",
            auth_residue_id=auth_residue_id,
            auth_component_id=component_id,
            auth_atom_name=atom_name,
            altloc_id="A",
        )

    selected_rows = [
        selected_row(0, "1", "LGU", "C1"),
        selected_row(1, "1", "LGU", "O4"),
        selected_row(2, "2", "BEM", "C1"),
        selected_row(3, "3", "LGU", "C1"),
    ]
    site_by_row_index = {
        0: AtomSite("B", 0, "C1"),
        1: AtomSite("B", 0, "O4"),
        2: AtomSite("B", 1, "C1"),
        3: AtomSite("B", 2, "C1"),
    }
    synthesized_aliases = {
        (None, "B", "2", None, "BEM", "C1"): AtomSite("B", 2, "C1"),
    }

    bonds = read_mmcif_bonds(
        cif_file,
        selected_rows,  # type: ignore[arg-type]
        site_by_row_index,
        synthesized_aliases,
        set(),
    )

    assert bonds == (SystemBond(AtomSite("B", 0, "O4"), AtomSite("B", 1, "C1")),)

    category = cif_file.block["struct_conn"]
    category["pdbx_ptnr1_label_alt_id"] = pdbx.CIFColumn(["A", "?"])
    category["pdbx_ptnr2_label_alt_id"] = pdbx.CIFColumn(["A", "?"])
    with pytest.raises(ValueError, match="struct_conn row 1 endpoint 2 is ambiguous"):
        read_mmcif_bonds(
            cif_file,
            selected_rows,  # type: ignore[arg-type]
            site_by_row_index,
            synthesized_aliases,
            set(),
        )


@pytest.mark.parametrize(
    ("auth_residue_id", "expected_bonds"),
    [
        (
            "1",
            (SystemBond(AtomSite("B", 0, "C1"), AtomSite("M", 0, "CA")),),
        ),
        ("2", ()),
    ],
)
def test_struct_conn_explicit_altloc_uses_precise_auth_identity_for_repeated_components(
    tmp_path: Path,
    auth_residue_id: str,
    expected_bonds: tuple[SystemBond, ...],
) -> None:
    cif_file = _read_cif(
        tmp_path / f"repeated-component-altloc-{auth_residue_id}.cif",
        f"""data_connections
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_auth_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_atom_id
_struct_conn.pdbx_ptnr1_label_alt_id
_struct_conn.pdbx_ptnr2_label_alt_id
1 covale B NAG ? C1 B NAG {auth_residue_id} C1 M CA ? CA M CA {auth_residue_id} CA B B
""",
    )

    def selected_row(
        row_index: int,
        chain_id: str,
        auth_residue_id: str,
        component_id: str,
        atom_name: str,
        altloc_id: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            row_index=row_index,
            symmetry_id=None,
            chain_id=chain_id,
            label_seq_id=None,
            insertion_code=None,
            component_id=component_id,
            atom_name=atom_name,
            auth_chain_id=chain_id,
            auth_residue_id=auth_residue_id,
            auth_component_id=component_id,
            auth_atom_name=atom_name,
            altloc_id=altloc_id,
        )

    # The B conformers at author residues 1 and 3 deliberately collide under
    # each sequence-less label identity. Author residue 2 instead selected A.
    selected_rows = [
        selected_row(0, "B", "1", "NAG", "C1", "B"),
        selected_row(1, "B", "2", "NAG", "C1", "A"),
        selected_row(2, "B", "3", "NAG", "C1", "B"),
        selected_row(3, "M", "1", "CA", "CA", "B"),
        selected_row(4, "M", "2", "CA", "CA", "A"),
        selected_row(5, "M", "3", "CA", "CA", "B"),
    ]
    site_by_row_index = {
        0: AtomSite("B", 0, "C1"),
        1: AtomSite("B", 1, "C1"),
        2: AtomSite("B", 2, "C1"),
        3: AtomSite("M", 0, "CA"),
        4: AtomSite("M", 1, "CA"),
        5: AtomSite("M", 2, "CA"),
    }

    bonds = read_mmcif_bonds(
        cif_file,
        selected_rows,  # type: ignore[arg-type]
        site_by_row_index,
        {},
        set(),
    )

    assert bonds == expected_bonds


def _fully_observed_structure(system: MolecularSystem) -> Structure:
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.zeros((atom_count, 3)),
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.ones(atom_count),
        occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
        source_symbols=tuple(str(index + 1) for index in range(component_count)),
        residue_id=torch.arange(component_count),
        insertion_code=torch.full((component_count,), -1),
        batch_size=[],
    )


def _bonded_molecules() -> tuple[Structure, dict[str, ChemicalComponent]]:
    left = ChemicalComponent("LEFT", (ChemicalAtom("L", "C"),), frozenset())
    right = ChemicalComponent("RIGHT", (ChemicalAtom("R", "N"),), frozenset())
    system = MolecularSystem.from_entities(
        {"left": Molecule((left,)), "right": Molecule((right,))},
        {"A": "left", "B": "right"},
        (SystemBond(AtomSite("A", 0, "L"), AtomSite("B", 0, "R"), order=2.0),),
    )
    return (
        _fully_observed_structure(system),
        {"LEFT": left, "RIGHT": right},
    )


def test_canonical_polymer_leaving_sites_use_instance_values() -> None:
    atoms = (
        ChemicalAtom("LINK", "C"),
        ChemicalAtom("DECLARED_DROP", "O"),
        ChemicalAtom("SELECTED_DROP", "O"),
    )
    right = ChemicalComponent(
        "RIGHT",
        (ChemicalAtom("LINK", "N"),),
        frozenset(),
        previous_attachment=PolymerAttachment("LINK"),
    )
    declared = PolymerSequence(
        "protein",
        (
            ChemicalComponent(
                "LEFT",
                atoms,
                frozenset(),
                next_attachment=PolymerAttachment("LINK", frozenset({"DECLARED_DROP"})),
            ),
            right,
        ),
    )
    selected = PolymerSequence(
        "protein",
        (
            ChemicalComponent(
                "LEFT",
                atoms,
                frozenset(),
                next_attachment=PolymerAttachment("LINK", frozenset({"SELECTED_DROP"})),
            ),
            right,
        ),
    )

    sites = _canonical_polymer_leaving_sites(
        {"protein": declared},
        {"A": "protein", "B": "protein"},
        instance_values={"B": selected},
    )

    assert sites == frozenset(
        {
            AtomSite("A", 0, "DECLARED_DROP"),
            AtomSite("B", 0, "SELECTED_DROP"),
        }
    )


class TestMmcifIO:
    @pytest.mark.parametrize(
        ("seed", "selected_component", "residue_numbers", "x_offset"),
        [
            (0, "HYP", ("13", "14"), 0.0),
            (1, "PRO", ("113", "114"), 100.0),
        ],
    )
    def test_polymer_microheterogeneity_randomly_keeps_one_complete_conformer(
        self,
        tmp_path: Path,
        seed: int,
        selected_component: str,
        residue_numbers: tuple[str, str],
        x_offset: float,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(tmp_path / f"microheterogeneity-{seed}.cif")
        generator = torch.Generator(device="cpu").manual_seed(seed)

        restored = read_structure(
            path,
            components=_microheterogeneous_polymer_components(),
            generator=generator,
        )

        assert _instance_component_ids(restored, "A") == (selected_component, selected_component)
        assert _instance_residue_numbers(restored, "A") == residue_numbers
        assert restored.coordinate_mask.tolist() == [True, True, True, True]
        assert restored.occupancy.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
        assert restored.coordinates[restored.system.atom_index(AtomSite("A", 0, "N"))].tolist() == pytest.approx(
            [x_offset + 1.0, 0.0, 0.0]
        )
        assert restored.coordinates[restored.system.atom_index(AtomSite("A", 1, "C"))].tolist() == pytest.approx(
            [x_offset + 2.0, 1.0, 0.0]
        )
        assert restored.system.edge_index.shape[1] == 3
        assert (restored.system.bond_role == BOND_ROLE_COMPONENT).sum().item() == 2
        assert (restored.system.bond_role == BOND_ROLE_POLYMER).sum().item() == 1

    def test_polymer_microheterogeneity_same_seed_reproduces_identity_and_structure_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(tmp_path / "seeded-microheterogeneity.cif")
        components = _microheterogeneous_polymer_components()

        first = read_structure(
            path,
            components=components,
            generator=torch.Generator(device="cpu").manual_seed(1),
        )
        second = read_structure(
            path,
            components=components,
            generator=torch.Generator(device="cpu").manual_seed(1),
        )

        first_metadata, first_arrays = encode_system(first.system)
        second_metadata, second_arrays = encode_system(second.system)
        assert _instance_component_ids(first, "A") == _instance_component_ids(second, "A")
        assert second_metadata == first_metadata
        assert second.source_symbols == first.source_symbols
        for name, first_array in first_arrays.items():
            assert torch.equal(second_arrays[name], first_array)
        for name in (
            "coordinates",
            "coordinate_mask",
            "occupancy",
            "occupancy_mask",
            "b_factor",
            "b_factor_mask",
            "residue_id",
            "insertion_code",
        ):
            assert torch.equal(getattr(second, name), getattr(first, name))

    def test_polymer_microheterogeneity_uses_one_label_for_the_whole_residue_block(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(tmp_path / "microheterogeneous-block.cif")

        for seed in (0, 1):
            restored = read_structure(
                path,
                components=_microheterogeneous_polymer_components(),
                generator=torch.Generator(device="cpu").manual_seed(seed),
            )
            component_ids = _instance_component_ids(restored, "A")
            assert component_ids in {("HYP", "HYP"), ("PRO", "PRO")}

    @pytest.mark.parametrize(
        ("seed", "selected_label", "selected_components"),
        [
            (0, "A", ("HYP", "VAL", "GLY", "PRO", "PRO")),
            (1, "B", ("PRO", "PRO", "GLY", "HYP", "VAL")),
        ],
    )
    def test_polymer_microheterogeneity_links_across_single_monomer_altloc_residue(
        self,
        tmp_path: Path,
        seed: int,
        selected_label: str,
        selected_components: tuple[str, ...],
    ) -> None:
        path = _write_3a0m_shaped_polymer_block_cif(tmp_path / "3a0m-shaped-block.cif")

        restored = read_structure(
            path,
            components=_microheterogeneous_polymer_components(),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )

        selected_labels: list[str] = []
        for label_seq_id in range(13, 18):
            atom_index = restored.system.atom_index(AtomSite("A", label_seq_id - 1, "N"))
            assert bool(restored.coordinate_mask[atom_index])
            selected_labels.append("B" if float(restored.coordinates[atom_index, 0]) >= 100.0 else "A")
        assert selected_labels == [selected_label] * 5
        assert _instance_component_ids(restored, "A")[12:17] == selected_components

    def test_polymer_microheterogeneity_selects_two_disjoint_blocks_deterministically(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_3a0m_shaped_polymer_block_cif(
            tmp_path / "3a0m-shaped-disjoint-blocks.cif",
            link_blocks=False,
        )
        components = _microheterogeneous_polymer_components()

        restored = [
            read_structure(
                path,
                components=components,
                generator=torch.Generator(device="cpu").manual_seed(0),
            )
            for _ in range(2)
        ]

        expected = ("HYP", "VAL", "GLY", "HYP", "VAL")
        assert [_instance_component_ids(item, "A")[12:17] for item in restored] == [expected, expected]
        assert torch.equal(restored[0].coordinates, restored[1].coordinates)

    @pytest.mark.parametrize(
        ("policy", "expected_x", "expected_occupancy"),
        [("first", 1.0, 0.25), ("highest_occupancy", 101.0, 0.75)],
    )
    def test_ordinary_polymer_altloc_honors_policy_without_consuming_generator(
        self,
        tmp_path: Path,
        policy: str,
        expected_x: float,
        expected_occupancy: float,
    ) -> None:
        path = tmp_path / f"ordinary-altloc-{policy}.cif"
        path.write_text(
            """data_ordinary_altloc
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
1 1 GLY n
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
A 1 1 GLY 1 1 ?
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
ATOM 1 N N A GLY A 1 1 ? 1.0 0.0 0.0 0.25 10.0 1 GLY A N 1
ATOM 2 C C A GLY A 1 1 ? 2.0 0.0 0.0 0.25 10.0 1 GLY A C 1
ATOM 3 N N B GLY A 1 1 ? 101.0 0.0 0.0 0.75 10.0 1 GLY A N 1
ATOM 4 C C B GLY A 1 1 ? 102.0 0.0 0.0 0.75 10.0 1 GLY A C 1
""",
            encoding="ascii",
        )
        generator = torch.Generator(device="cpu").manual_seed(7)
        initial_state = generator.get_state().clone()

        restored = read_structure(
            path,
            altloc=policy,  # type: ignore[arg-type]
            components=_microheterogeneous_polymer_components(),
            generator=generator,
        )

        atom_index = restored.system.atom_index(AtomSite("A", 0, "N"))
        assert restored.coordinates[atom_index].tolist() == [expected_x, 0.0, 0.0]
        assert restored.occupancy.tolist() == pytest.approx([expected_occupancy, expected_occupancy])
        assert torch.equal(generator.get_state(), initial_state)

    def test_polymer_microheterogeneity_splits_only_distinct_realized_entities(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(
            tmp_path / "microheterogeneous-entities.cif",
            chain_ids=("A", "B", "C"),
        )

        restored = read_structure(
            path,
            components=_microheterogeneous_polymer_components(),
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

        assert _instance_component_ids(restored, "A") == ("HYP", "HYP")
        assert _instance_component_ids(restored, "B") == ("PRO", "PRO")
        assert _instance_component_ids(restored, "C") == ("PRO", "PRO")
        assert _instance_entity_index(restored, "A") != _instance_entity_index(restored, "B")
        assert _instance_entity_index(restored, "B") == _instance_entity_index(restored, "C")
        assert restored.system.entity_symbol.numel() == 2

    def test_polymer_microheterogeneity_assembly_copies_reuse_the_source_choice(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(
            tmp_path / "microheterogeneous-assembly.cif",
            include_assembly=True,
        )

        restored = read_structure(
            path,
            assembly_id="1",
            components=_microheterogeneous_polymer_components(),
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

        instance_ids = [
            restored.system.instance_id(index)
            for index in range(int(restored.system.instance_symbol.numel()))
        ]
        assert len(instance_ids) == 2
        assert len({_instance_component_ids(restored, instance_id) for instance_id in instance_ids}) == 1
        assert len({_instance_entity_index(restored, instance_id) for instance_id in instance_ids}) == 1

    def test_polymer_microheterogeneity_candidates_require_all_hetero_flags(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "invalid-polymer-microheterogeneity.cif",
            """data_microheterogeneity
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 HYP y
1 1 PRO n
""",
        )

        with pytest.raises(ValueError, match="hetero|microheterogeneous|non-heterogeneous"):
            _read_mmcif_entity_declarations(cif_file)

    def test_polymer_microheterogeneity_accepts_yes_as_an_affirmative_hetero_flag(
        self,
        tmp_path: Path,
    ) -> None:
        cif_file = _read_cif(
            tmp_path / "yes-polymer-microheterogeneity.cif",
            """data_microheterogeneity
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 HYP yes
1 1 PRO yes
""",
        )

        _, polymer_monomers, _, _, _, _ = _read_mmcif_entity_declarations(cif_file)

        assert polymer_monomers == {"1": ((1, ("HYP", "PRO")),)}

    def test_polymer_microheterogeneity_exact_duplicate_y_declaration_is_idempotent(
        self,
        tmp_path: Path,
    ) -> None:
        cif_file = _read_cif(
            tmp_path / "duplicate-polymer-microheterogeneity.cif",
            """data_microheterogeneity
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 HYP y
1 1 HYP y
1 1 PRO y
""",
        )

        _, polymer_monomers, _, _, _, _ = _read_mmcif_entity_declarations(cif_file)

        assert polymer_monomers == {"1": ((1, ("HYP", "PRO")),)}

    def test_polymer_microheterogeneity_rejects_one_altloc_label_for_multiple_monomers(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(tmp_path / "ambiguous-label-monomers.cif")
        path.write_text(path.read_text().replace(" B PRO A 1 ", " A PRO A 1 "), encoding="ascii")

        with pytest.raises(ValueError, match="maps alternate labels to multiple monomers"):
            read_structure(
                path,
                components=_microheterogeneous_polymer_components(),
                generator=torch.Generator(device="cpu").manual_seed(0),
            )

    def test_polymer_microheterogeneity_rejects_labelled_and_unlabelled_candidate_mixture(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(tmp_path / "mixed-labelled-unlabelled.cif")
        path.write_text(path.read_text().replace(" B PRO A 1 ", " . PRO A 1 "), encoding="ascii")

        with pytest.raises(ValueError, match="mixes labelled and unlabelled monomer candidates"):
            read_structure(
                path,
                components=_microheterogeneous_polymer_components(),
                generator=torch.Generator(device="cpu").manual_seed(0),
            )

    @pytest.mark.parametrize("scheme_hetero", [None, "?", "y"])
    def test_polymer_microheterogeneity_candidate_numbering_does_not_require_scheme_hetero(
        self,
        tmp_path: Path,
        scheme_hetero: str | None,
    ) -> None:
        scheme_hetero_column = "" if scheme_hetero is None else "_pdbx_poly_seq_scheme.hetero\n"
        scheme_hetero_suffix = "" if scheme_hetero is None else f" {scheme_hetero}"
        cif_file = _read_cif(
            tmp_path / f"candidate-specific-numbering-{scheme_hetero or 'omitted'}.cif",
            f"""data_microheterogeneity
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 HYP y
1 1 PRO y
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
{scheme_hetero_column}A 1 1 HYP 13 13 ?{scheme_hetero_suffix}
A 1 1 PRO 113 113 ?{scheme_hetero_suffix}
""",
        )

        _read_mmcif_entity_declarations(cif_file)

    def test_polymer_microheterogeneity_same_candidate_scheme_conflict_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        cif_file = _read_cif(
            tmp_path / "same-candidate-numbering-conflict.cif",
            """data_microheterogeneity
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 HYP y
1 1 PRO y
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
_pdbx_poly_seq_scheme.hetero
A 1 1 HYP 13 13 ? y
A 1 1 HYP 14 14 ? y
""",
        )

        with pytest.raises(ValueError, match="conflicting numbering"):
            _read_mmcif_entity_declarations(cif_file)

    @pytest.mark.parametrize("compressed_name", ["microheterogeneity.cif.gz", "microheterogeneity.mmcif.gz"])
    def test_polymer_microheterogeneity_gzip_matches_with_equal_generator_state(
        self,
        tmp_path: Path,
        compressed_name: str,
    ) -> None:
        path = _write_microheterogeneous_polymer_cif(tmp_path / "microheterogeneity.cif")
        compressed_path = tmp_path / compressed_name
        with gzip.open(compressed_path, "wb") as handle:
            handle.write(path.read_bytes())
        components = _microheterogeneous_polymer_components()

        expected = read_structure(
            path,
            components=components,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        actual = read_structure(
            compressed_path,
            components=components,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

        expected_metadata, expected_arrays = encode_system(expected.system)
        actual_metadata, actual_arrays = encode_system(actual.system)
        assert actual_metadata == expected_metadata
        assert actual.source_symbols == expected.source_symbols
        for name, expected_array in expected_arrays.items():
            assert torch.equal(actual_arrays[name], expected_array)
        for name in (
            "coordinates",
            "coordinate_mask",
            "occupancy",
            "occupancy_mask",
            "b_factor",
            "b_factor_mask",
            "residue_id",
            "insertion_code",
        ):
            assert torch.equal(getattr(actual, name), getattr(expected, name))

    @pytest.mark.parametrize("compressed_name", ["ligand.cif.gz", "ligand.mmcif.gz"])
    def test_gzip_compressed_input_matches_uncompressed_mmcif(
        self,
        tmp_path: Path,
        compressed_name: str,
    ) -> None:
        path = tmp_path / "ligand.cif"
        compressed_path = tmp_path / compressed_name
        write_structure(_structure(), path)
        with gzip.open(compressed_path, "wb") as handle:
            handle.write(path.read_bytes())

        expected = read_structure(path)
        actual = read_structure(compressed_path)

        expected_metadata, expected_arrays = encode_system(expected.system)
        actual_metadata, actual_arrays = encode_system(actual.system)
        assert actual_metadata == expected_metadata
        for name, expected_array in expected_arrays.items():
            assert torch.equal(actual_arrays[name], expected_array)
        for name in (
            "coordinates",
            "coordinate_mask",
            "occupancy",
            "occupancy_mask",
            "b_factor",
            "b_factor_mask",
            "residue_id",
            "insertion_code",
        ):
            assert torch.equal(getattr(actual, name), getattr(expected, name))
        assert actual.source_symbols == expected.source_symbols

    @pytest.mark.parametrize("name", ["output.pdb.gz", "output.cif.gz", "output.mmcif.gz"])
    def test_write_structure_rejects_gzip_paths(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError, match="structure path suffix"):
            write_structure(_structure(), tmp_path / name)

    def test_water_only_defaults_to_excluded_and_can_be_opted_in(self, tmp_path: Path) -> None:
        water = _water_component()
        system = MolecularSystem.from_entities(
            {"water": Molecule((water,), molecule_type="water")},
            {"W": "water"},
        )
        path = tmp_path / "water-only.cif"
        write_structure(_fully_observed_structure(system), path)

        with pytest.raises(NoRetainedStructureError, match="only water excluded"):
            read_structure(path, components={"HOH": water})
        retained = read_structure(path, components={"HOH": water}, include_water=True)

        assert retained.system.atomic_number.tolist() == [8]
        assert retained.system.instance_id(0) == "W"

    def test_mixed_water_defaults_to_only_the_nonwater_entity(self, tmp_path: Path) -> None:
        ligand = _ligand_component()
        water = _water_component()
        system = MolecularSystem.from_entities(
            {
                "ligand": Molecule((ligand,)),
                "water": Molecule((water,), molecule_type="water"),
            },
            {"L": "ligand", "W": "water"},
        )
        path = tmp_path / "mixed-water.cif"
        write_structure(_fully_observed_structure(system), path)

        retained = read_structure(path, components={"HOH": water})

        assert retained.system.instance_symbol.numel() == 1
        assert retained.system.instance_id(0) == "L"
        assert {
            retained.system.symbols[int(symbol)]
            for symbol in retained.system.component_symbol.tolist()
        } == {"LIG"}

    def test_deuterium_only_asym_is_not_rematerialized_after_default_filtering(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        ligand = _ligand_component()
        deuterium = ChemicalComponent("D8U", (ChemicalAtom("D", "H"),), frozenset())
        system = MolecularSystem.from_entities(
            {
                "ligand": Molecule((ligand,)),
                "deuterium": Molecule((deuterium,)),
            },
            {"L": "ligand", "D": "deuterium"},
        )
        path = tmp_path / "mixed-deuterium.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        type_symbols = atom_site["type_symbol"].as_array()
        asym_ids = atom_site["label_asym_id"].as_array()
        type_symbols[asym_ids == "D"] = "D"
        atom_site["type_symbol"] = pdbx.CIFColumn(type_symbols)
        cif_file.write(str(path))

        default = read_structure(path, components={"LIG": ligand, "D8U": deuterium})
        opted_in = read_structure(
            path,
            components={"LIG": ligand, "D8U": deuterium},
            include_hydrogens=True,
        )

        assert [
            default.system.instance_id(index)
            for index in range(int(default.system.instance_symbol.numel()))
        ] == ["L"]
        assert default.system.atomic_number.tolist() == [6, 8]
        assert [
            opted_in.system.instance_id(index)
            for index in range(int(opted_in.system.instance_symbol.numel()))
        ] == ["L", "D"]
        deuterium_index = opted_in.system.atom_index(AtomSite("D", 0, "D"))
        assert int(opted_in.system.atomic_number[deuterium_index]) == 1
        assert int(opted_in.system.isotope[deuterium_index]) == 2

    def test_unobserved_declared_heavy_nonpoly_is_still_rematerialized(self, tmp_path: Path) -> None:
        ligand = _ligand_component()
        ion = ChemicalComponent("ION", (ChemicalAtom("N", "N"),), frozenset())
        system = MolecularSystem.from_entities(
            {
                "ligand": Molecule((ligand,)),
                "ion": Molecule((ion,)),
            },
            {"L": "ligand", "U": "ion"},
        )
        structure = _fully_observed_structure(system)
        ion_atom = system.atom_index(AtomSite("U", 0, "N"))
        coordinate_mask = structure.coordinate_mask.clone()
        occupancy_mask = structure.occupancy_mask.clone()
        b_factor_mask = structure.b_factor_mask.clone()
        coordinate_mask[ion_atom] = False
        occupancy_mask[ion_atom] = False
        b_factor_mask[ion_atom] = False
        structure = structure.replace(
            coordinates=structure.coordinates.clone(),
            coordinate_mask=coordinate_mask,
            occupancy=structure.occupancy.clone(),
            occupancy_mask=occupancy_mask,
            b_factor=structure.b_factor.clone(),
            b_factor_mask=b_factor_mask,
        )
        path = tmp_path / "unobserved-heavy-nonpoly.cif"
        write_structure(structure, path)

        restored = read_structure(path, components={"LIG": ligand, "ION": ion})

        assert [
            restored.system.instance_id(index)
            for index in range(int(restored.system.instance_symbol.numel()))
        ] == ["L", "U"]
        restored_ion_atom = restored.system.atom_index(AtomSite("U", 0, "N"))
        assert int(restored.system.atomic_number[restored_ion_atom]) == 7
        assert not bool(restored.coordinate_mask[restored_ion_atom])

    def test_unobserved_declared_water_asym_is_not_rematerialized_by_default(self, tmp_path: Path) -> None:
        ligand = _ligand_component()
        water = _water_component()
        system = MolecularSystem.from_entities(
            {
                "ligand": Molecule((ligand,)),
                "water": Molecule((water,), molecule_type="water"),
            },
            {"L": "ligand", "W": "water"},
        )
        structure = _fully_observed_structure(system)
        water_start = int(system.component_atom_ptr[1])
        coordinate_mask = structure.coordinate_mask.clone()
        occupancy_mask = structure.occupancy_mask.clone()
        b_factor_mask = structure.b_factor_mask.clone()
        coordinate_mask[water_start:] = False
        occupancy_mask[water_start:] = False
        b_factor_mask[water_start:] = False
        structure = structure.replace(
            coordinates=structure.coordinates.clone(),
            coordinate_mask=coordinate_mask,
            occupancy=structure.occupancy.clone(),
            occupancy_mask=occupancy_mask,
            b_factor=structure.b_factor.clone(),
            b_factor_mask=b_factor_mask,
        )
        path = tmp_path / "declared-water.cif"
        write_structure(structure, path)

        default = read_structure(path, components={"HOH": water})
        opted_in = read_structure(path, components={"HOH": water}, include_water=True)

        assert default.system.instance_symbol.numel() == 1
        assert default.system.instance_id(0) == "L"
        assert opted_in.system.instance_symbol.numel() == 2
        water_instance = next(
            index
            for index in range(int(opted_in.system.instance_symbol.numel()))
            if opted_in.system.instance_id(index) == "W"
        )
        water_component = int(opted_in.system.instance_component_ptr[water_instance])
        water_atom = int(opted_in.system.component_atom_ptr[water_component])
        assert int(opted_in.system.atomic_number[water_atom]) == 8
        assert not bool(opted_in.coordinate_mask[water_atom])

    def test_struct_conn_whitelist_forces_single_undirected_inter_component_bond(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        structure, components = _bonded_molecules()
        path = tmp_path / "covalent.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"]["ptnr1_role"] = pdbx.CIFColumn(["metal"])
        cif_file.block["struct_conn"]["ptnr2_role"] = pdbx.CIFColumn(["donor"])
        cif_file.write(str(path))

        restored = read_structure(path, components=components)

        assert restored.system.edge_index.shape[1] == 1
        assert restored.system.bond_order.tolist() == [1.0]
        assert restored.system.bond_kind.tolist() == [BOND_KIND_COVALENT]
        assert restored.system.bond_direction.tolist() == [BOND_DIRECTION_UNDIRECTED]

    def test_struct_conn_endpoint_removed_by_canonical_polymer_link_is_filtered(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        left_atoms = (
            ChemicalAtom("LINK", "C"),
            ChemicalAtom("DROP", "O"),
            ChemicalAtom("KEEP", "C"),
        )
        right_atoms = (
            ChemicalAtom("LINK", "N"),
            ChemicalAtom("DROP", "O"),
            ChemicalAtom("KEEP", "C"),
        )
        ligand = ChemicalComponent(
            "LIG",
            (
                ChemicalAtom("L1", "N"),
                ChemicalAtom("L2", "S"),
                ChemicalAtom("L3", "P"),
            ),
            None,
        )
        raw_left = ChemicalComponent("LEFT", left_atoms, None)
        raw_right = ChemicalComponent("RIGHT", right_atoms, None)
        raw_system = MolecularSystem.from_entities(
            {
                "protein": PolymerSequence("protein", (raw_left, raw_right)),
                "ligand": Molecule((ligand,)),
            },
            {"A": "protein", "L": "ligand"},
            (
                SystemBond(AtomSite("A", 0, "DROP"), AtomSite("L", 0, "L1")),
                SystemBond(AtomSite("L", 0, "L3"), AtomSite("A", 1, "DROP")),
                SystemBond(AtomSite("A", 1, "KEEP"), AtomSite("L", 0, "L2")),
            ),
        )
        path = tmp_path / "polymer-leaving-struct-conn.cif"
        write_structure(_fully_observed_structure(raw_system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        connections = cif_file.block["struct_conn"]
        right_conflict = next(
            index
            for index, (component, atom) in enumerate(
                zip(
                    connections["ptnr1_label_comp_id"].as_array(),
                    connections["ptnr1_label_atom_id"].as_array(),
                    strict=True,
                )
            )
            if (component, atom) == ("RIGHT", "DROP")
        )
        for partner_1, partner_2 in (
            ("ptnr1_label_asym_id", "ptnr2_label_asym_id"),
            ("ptnr1_label_seq_id", "ptnr2_label_seq_id"),
            ("ptnr1_label_comp_id", "ptnr2_label_comp_id"),
            ("ptnr1_label_atom_id", "ptnr2_label_atom_id"),
            ("ptnr1_auth_asym_id", "ptnr2_auth_asym_id"),
            ("ptnr1_auth_seq_id", "ptnr2_auth_seq_id"),
            ("ptnr1_auth_comp_id", "ptnr2_auth_comp_id"),
            ("ptnr1_auth_atom_id", "ptnr2_auth_atom_id"),
            ("pdbx_ptnr1_PDB_ins_code", "pdbx_ptnr2_PDB_ins_code"),
            ("ptnr1_role", "ptnr2_role"),
            ("ptnr1_symmetry", "ptnr2_symmetry"),
        ):
            values_1 = connections[partner_1].as_array().tolist()
            values_2 = connections[partner_2].as_array().tolist()
            values_1[right_conflict], values_2[right_conflict] = (
                values_2[right_conflict],
                values_1[right_conflict],
            )
            connections[partner_1] = pdbx.CIFColumn(values_1)
            connections[partner_2] = pdbx.CIFColumn(values_2)
        assert (
            connections["ptnr1_label_comp_id"].as_array()[right_conflict],
            connections["ptnr1_label_atom_id"].as_array()[right_conflict],
            connections["ptnr2_label_comp_id"].as_array()[right_conflict],
            connections["ptnr2_label_atom_id"].as_array()[right_conflict],
        ) == ("LIG", "L3", "RIGHT", "DROP")
        cif_file.write(str(path))
        left = ChemicalComponent(
            "LEFT",
            left_atoms,
            None,
            next_attachment=PolymerAttachment("LINK", frozenset({"DROP"})),
        )
        right = ChemicalComponent(
            "RIGHT",
            right_atoms,
            None,
            previous_attachment=PolymerAttachment("LINK", frozenset({"DROP"})),
        )

        restored = read_structure(
            path,
            components={"LEFT": left, "RIGHT": right, "LIG": ligand},
        )

        edges = {
            (
                frozenset(
                    {
                        restored.system.atom_site(atom_a),
                        restored.system.atom_site(atom_b),
                    }
                ),
                role,
            )
            for (atom_a, atom_b), role in zip(
                restored.system.edge_index.T.tolist(),
                restored.system.bond_role.tolist(),
                strict=True,
            )
        }
        assert edges == {
            (
                frozenset({AtomSite("A", 0, "LINK"), AtomSite("A", 1, "LINK")}),
                BOND_ROLE_POLYMER,
            ),
            (
                frozenset({AtomSite("A", 1, "KEEP"), AtomSite("L", 0, "L2")}),
                BOND_ROLE_ASSEMBLY,
            ),
        }
        for component_index in (0, 1):
            with pytest.raises(KeyError):
                restored.system.atom_index(AtomSite("A", component_index, "DROP"))

    @pytest.mark.parametrize("connection_type", ["metalc", "covalent", "hydrog"])
    def test_struct_conn_ignores_unsupported_connection_types(
        self,
        tmp_path: Path,
        connection_type: str,
    ) -> None:
        from biotite.structure.io import pdbx

        structure, components = _bonded_molecules()
        path = tmp_path / f"unsupported-{connection_type}.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"]["conn_type_id"] = pdbx.CIFColumn([connection_type])
        cif_file.write(str(path))

        assert read_structure(path, components=components).system.edge_index.shape[1] == 0

    def test_struct_conn_ignores_same_component_links(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        structure, components = _bonded_molecules()
        path = tmp_path / "same-component.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        category = cif_file.block["struct_conn"]
        for suffix in (
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "label_atom_id",
            "auth_asym_id",
            "auth_seq_id",
            "auth_comp_id",
            "auth_atom_id",
        ):
            category[f"ptnr2_{suffix}"] = pdbx.CIFColumn(category[f"ptnr1_{suffix}"].as_array())
        cif_file.write(str(path))

        assert read_structure(path, components=components).system.edge_index.shape[1] == 0

    def test_asx_rows_and_declarations_normalize_to_asp(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("D")},
            {"A": "protein"},
        )
        path = tmp_path / "asx.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        for column_name in ("label_comp_id", "auth_comp_id"):
            atom_site[column_name] = pdbx.CIFColumn(["ASX"] * atom_site.row_count)
        for column_name in ("label_atom_id", "auth_atom_id"):
            atom_site[column_name] = pdbx.CIFColumn(
                [
                    {"OD1": "XD1", "OD2": "XD2"}.get(atom_id, atom_id)
                    for atom_id in atom_site[column_name].as_array()
                ]
            )
        atom_site["type_symbol"] = pdbx.CIFColumn(
            [
                "X" if atom_id in {"XD1", "XD2"} else element
                for atom_id, element in zip(
                    atom_site["label_atom_id"].as_array(),
                    atom_site["type_symbol"].as_array(),
                )
            ]
        )
        cif_file.block["entity_poly_seq"]["mon_id"] = pdbx.CIFColumn(["ASX"])
        cif_file.write(str(path))

        restored = read_structure(path)
        component_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.component_symbol.tolist()
        }
        atom_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.atom_symbol.tolist()
        }
        assert component_ids == {"ASP"}
        assert {"OD1", "OD2"} <= atom_ids
        assert not {"XD1", "XD2"}.intersection(atom_ids)

    def test_raw_unl_occurrence_is_removed_without_explicit_custom_component(self, tmp_path: Path) -> None:
        unl = ChemicalComponent("UNL", (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("A"), "unl": Molecule((unl,))},
            {"A": "protein", "L": "unl"},
        )
        path = tmp_path / "raw-unl.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)

        assert [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())] == [
            "A"
        ]
        restored_custom = read_structure(path, components={"UNL": unl})
        assert {
            restored_custom.system.instance_id(index)
            for index in range(restored_custom.system.instance_symbol.numel())
        } == {"A", "L"}

    @pytest.mark.parametrize("component_id", ["UNX", "UNL"])
    def test_only_raw_excluded_occurrences_raise_no_retained_structure(
        self,
        tmp_path: Path,
        component_id: str,
    ) -> None:
        component = ChemicalComponent(component_id, (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities({"excluded": Molecule((component,))}, {"L": "excluded"})
        path = tmp_path / f"raw-{component_id.lower()}.cif"
        write_structure(_fully_observed_structure(system), path)

        with pytest.raises(NoRetainedStructureError, match="excluded UNX/UNL"):
            read_structure(path)

    def test_raw_unl_branched_declaration_is_skipped_as_no_retained_structure(self, tmp_path: Path) -> None:
        component = ChemicalComponent("UNL", (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities({"excluded": Molecule((component,))}, {"L": "excluded"})
        system = system.replace(entity_molecule_type=system.entity_molecule_type.new_tensor([MOLECULE_BRANCHED]))
        path = tmp_path / "raw-unl-branched.cif"
        write_structure(_fully_observed_structure(system), path)

        with pytest.raises(NoRetainedStructureError, match="excluded UNX/UNL"):
            read_structure(path)

    def test_raw_unl_branched_declaration_does_not_break_retained_entities(self, tmp_path: Path) -> None:
        component = ChemicalComponent("UNL", (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("A"), "excluded": Molecule((component,))},
            {"A": "protein", "L": "excluded"},
        )
        molecule_types = system.entity_molecule_type.clone()
        molecule_types[1] = MOLECULE_BRANCHED
        system = system.replace(entity_molecule_type=molecule_types)
        path = tmp_path / "mixed-raw-unl-branched.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)
        assert [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())] == [
            "A"
        ]

    def test_declared_mixed_protein_is_retained_when_only_unk_is_observed(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        sequence = PolymerSequence("protein", (get_component("UNK"), get_component("ALA")))
        system = MolecularSystem.from_entities({"protein": sequence}, {"A": "protein"})
        path = tmp_path / "declared-mixed-observed-unk.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        keep = [value == "1" for value in atom_site["label_seq_id"].as_array()]
        cif_file.block["atom_site"] = pdbx.CIFCategory(
            {
                name: pdbx.CIFColumn([value for value, retained in zip(atom_site[name].as_array(), keep) if retained])
                for name in atom_site
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path)
        assert [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())] == [
            "A"
        ]
        assert [
            restored.system.symbols[int(symbol)] for symbol in restored.system.component_symbol.tolist()
        ] == ["UNK", "ALA"]
        atom_start = int(restored.system.component_atom_ptr[1])
        atom_stop = int(restored.system.component_atom_ptr[2])
        assert not bool(restored.coordinate_mask[atom_start:atom_stop].any())

    @pytest.mark.parametrize(("polymer_type", "component_id"), [("rna", "N"), ("dna", "DN")])
    def test_unknown_nucleic_polymer_is_not_removed(
        self,
        tmp_path: Path,
        polymer_type: str,
        component_id: str,
    ) -> None:
        sequence = PolymerSequence(polymer_type, (get_component(component_id),))
        system = MolecularSystem.from_entities({"polymer": sequence}, {"A": "polymer"})
        path = tmp_path / f"unknown-{component_id.lower()}.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)
        assert restored.system.instance_symbol.numel() == 1
        assert restored.system.symbols[int(restored.system.component_symbol[0])] == component_id

    def test_round_trip_preserves_the_complete_public_graph_and_coordinates(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        path = tmp_path / "ligand.cif"
        write_structure(structure, path)
        restored = read_structure(path, components={"LIG": _ligand_component()})

        expected_metadata, expected_arrays = encode_system(structure.system)
        actual_metadata, actual_arrays = encode_system(restored.system)
        assert actual_metadata == expected_metadata
        for name, expected in expected_arrays.items():
            assert torch.equal(actual_arrays[name], expected)
        assert torch.equal(restored.coordinates, structure.coordinates)
        assert torch.equal(restored.coordinate_mask, structure.coordinate_mask)
        assert torch.equal(restored.occupancy, structure.occupancy)
        assert restored.source_symbols == structure.source_symbols

    def test_atom_site_is_the_coordinate_source_of_truth(
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

    def test_ccd_miss_keeps_observed_atoms_without_embedded_chemistry(self, tmp_path: Path) -> None:
        path = tmp_path / "ccd-miss.cif"
        write_structure(_structure(), path)

        restored = read_structure(path)

        atom_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.atom_symbol.tolist()
        }
        assert atom_ids == {"C1", "O1"}
        assert restored.system.edge_index.shape[1] == 0
        assert not restored.system.reference_coordinate_mask.any()

    def test_occurrence_charge_and_isotope_are_not_shared_through_component_placeholders(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        component = ChemicalComponent("DUP", (ChemicalAtom("H1", "H"),), frozenset())
        system = MolecularSystem.from_entities(
            {"ligand": Molecule((component,))},
            {"A": "ligand", "B": "ligand"},
        )
        path = tmp_path / "occurrence-fields.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        atom_site["type_symbol"] = pdbx.CIFColumn(["D", "T"])
        atom_site["pdbx_formal_charge"] = pdbx.CIFColumn(["-1", "1"])
        cif_file.write(str(path))

        for restored in (
            read_structure(path, include_hydrogens=True),
            read_structure(path, components={"DUP": component}, include_hydrogens=True),
        ):
            first = restored.system.atom_index(AtomSite("A", 0, "H1"))
            second = restored.system.atom_index(AtomSite("B", 0, "H1"))
            assert restored.system.isotope_known[[first, second]].tolist() == [True, True]
            assert restored.system.isotope[[first, second]].tolist() == [2, 3]
            assert restored.system.formal_charge_known[[first, second]].tolist() == [True, True]
            assert restored.system.formal_charge[[first, second]].tolist() == [-1, 1]

    def test_ccd_miss_occurrence_values_and_known_masks_do_not_leak_between_instances(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        component = ChemicalComponent("DUP", (ChemicalAtom("H1", "H"),), frozenset())
        system = MolecularSystem.from_entities(
            {"ligand": Molecule((component,))},
            {"A": "ligand", "B": "ligand"},
        )
        path = tmp_path / "occurrence-known-masks.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        atom_site["type_symbol"] = pdbx.CIFColumn(["D", "H"])
        atom_site["pdbx_formal_charge"] = pdbx.CIFColumn(["-1", "?"])
        cif_file.write(str(path))

        restored = read_structure(path, include_hydrogens=True)
        first = restored.system.atom_index(AtomSite("A", 0, "H1"))
        second = restored.system.atom_index(AtomSite("B", 0, "H1"))
        assert bool(restored.system.isotope_known[first])
        assert int(restored.system.isotope[first]) == 2
        assert not bool(restored.system.isotope_known[second])
        assert int(restored.system.isotope[second]) == 0
        assert bool(restored.system.formal_charge_known[first])
        assert int(restored.system.formal_charge[first]) == -1
        assert not bool(restored.system.formal_charge_known[second])
        assert int(restored.system.formal_charge[second]) == 0

    def test_known_component_drops_extra_atom_site_rows(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "known-extra.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        columns = {}
        for name in atom_site:
            values = atom_site[name].as_array().tolist()
            extra = values[0]
            if name == "id":
                extra = "3"
            elif name in {"label_atom_id", "auth_atom_id"}:
                extra = "H1"
            elif name == "type_symbol":
                extra = "H"
            elif name == "pdbx_formal_charge":
                extra = "?"
            values.append(extra)
            columns[name] = pdbx.CIFColumn(values)
        cif_file.block["atom_site"] = pdbx.CIFCategory(columns)
        cif_file.write(str(path))

        restored = read_structure(path, components={"LIG": _ligand_component()})
        atom_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.atom_symbol.tolist()
        }
        assert atom_ids == {"C1", "O1"}

    def test_nonpoly_atom_without_coordinates_is_absent_from_the_observed_union(
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
        restored = read_structure(path, components={"LIG": _ligand_component()})

        assert restored.system.atom_symbol.numel() == 1
        assert restored.system.symbols[int(restored.system.atom_symbol[0])] == "O1"
        assert restored.coordinate_mask.tolist() == [True]

    def test_aromaticity_masks_are_not_replaced_by_filler_values(
        self,
        tmp_path: Path,
    ) -> None:
        partial_bond = _structure(bond_order=1.0, bond_aromatic=None)
        partial_path = tmp_path / "partial-bond.cif"
        write_structure(partial_bond, partial_path)
        restored = read_structure(
            partial_path,
            components={"LIG": _ligand_component(bond_order=1.0, bond_aromatic=None)},
        )
        assert restored.system.bond_order_known.all()
        assert not restored.system.bond_aromatic_known.any()

    @pytest.mark.parametrize("order", [1.0, 2.0])
    def test_component_bond_known_order_and_aromatic_true_round_trip(
        self,
        tmp_path: Path,
        order: float,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / f"aromatic-{order}.cif"
        write_structure(_structure(bond_order=order, bond_aromatic=True), path)

        cif_file = pdbx.CIFFile.read(str(path))
        bond_category = cif_file.block["chem_comp_bond"]
        assert bond_category["value_order"].as_array().tolist() == ["sing" if order == 1.0 else "doub"]
        assert bond_category["pdbx_aromatic_flag"].as_array().tolist() == ["Y"]

        restored = read_structure(
            path,
            components={"LIG": _ligand_component(bond_order=order, bond_aromatic=True)},
        )
        assert restored.system.bond_order_known.all()
        assert restored.system.bond_order.tolist() == [order]
        assert restored.system.bond_aromatic_known.all()
        assert restored.system.bond_aromatic.all()

    def test_component_bond_unknown_order_and_aromaticity_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "unknown-bond.cif"
        write_structure(_structure(bond_order=None, bond_aromatic=None), path)
        restored = read_structure(
            path,
            components={"LIG": _ligand_component(bond_order=None, bond_aromatic=None)},
        )

        assert not restored.system.bond_order_known.any()
        assert not restored.system.bond_aromatic_known.any()

    @pytest.mark.parametrize(
        ("order", "aromatic"),
        [
            ("arom", "N"),
            ("sing", "Y"),
            ("sing", "invalid"),
        ],
    )
    def test_reader_ignores_inconsistent_or_invalid_embedded_aromatic_flags(
        self,
        tmp_path: Path,
        order: str,
        aromatic: str,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "invalid-aromatic.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        bond_category = cif_file.block["chem_comp_bond"]
        bond_category["value_order"] = pdbx.CIFColumn([order])
        bond_category["pdbx_aromatic_flag"] = pdbx.CIFColumn([aromatic])
        cif_file.write(str(path))

        components, _ = read_mmcif_components(pdbx.CIFFile.read(str(path)))
        bond = next(iter(components["LIG"].bonds or ()))
        assert bond.order == (1.5 if order == "arom" else 1.0)
        assert bond.aromatic == ({"Y": True, "N": False}.get(aromatic))

    def test_reader_maps_invalid_embedded_atom_aromatic_flag_to_unknown(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "invalid-atom-aromatic.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_category = cif_file.block["chem_comp_atom"]
        atom_category["pdbx_aromatic_flag"] = pdbx.CIFColumn(["invalid"] * atom_category.row_count)
        cif_file.write(str(path))

        components, _ = read_mmcif_components(pdbx.CIFFile.read(str(path)))
        assert all(atom.aromatic is None for atom in components["LIG"].atoms)

    def test_default_reader_omits_component_edges_but_keeps_polymer_adjacency(
        self,
        tmp_path: Path,
    ) -> None:
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("AA")},
            {"A": "protein"},
        )
        path = tmp_path / "polymer.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)

        assert not restored.system.component_topology_complete.any()
        assert not bool((restored.system.bond_role == BOND_ROLE_COMPONENT).any())
        assert (restored.system.bond_role == BOND_ROLE_POLYMER).sum().item() == 1

    def test_explicit_components_preserve_component_topology(self, tmp_path: Path) -> None:
        path = tmp_path / "explicit.cif"
        write_structure(_structure(), path)

        restored = read_structure(path, components={"LIG": _ligand_component()})

        assert restored.system.component_topology_complete.tolist() == [True]
        assert (restored.system.bond_role == BOND_ROLE_COMPONENT).sum().item() == 1

    def test_struct_conn_bond_order_semantics_are_unchanged(self) -> None:
        assert _cif_bond_order("sing") == (1.0, False)
        assert _cif_bond_order("doub") == (2.0, False)
        assert _cif_component_bond_order("sing") == (1.0, None)
        assert _cif_component_bond_order("doub") == (2.0, None)

    def test_nonpoly_numbering_uses_ndb_sequence_number(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "numbering.cif",
            """data_numbering
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
_pdbx_nonpoly_scheme.pdb_seq_num
_pdbx_nonpoly_scheme.pdb_ins_code
A 1 10 10 .
A 2 20 20 ?
""",
        )

        numbering = _read_mmcif_entity_declarations(cif_file)[3]
        assert numbering == {("A", 1): ("10", None), ("A", 2): ("20", None)}

    @pytest.mark.parametrize(
        ("ndb_number", "message"),
        [("?", "missing asym_id or ndb_seq_num"), ("not-an-integer", "expected an integer identity")],
    )
    def test_nonpoly_numbering_requires_integer_ndb_sequence_number(
        self,
        tmp_path: Path,
        ndb_number: str,
        message: str,
    ) -> None:
        cif_file = _read_cif(
            tmp_path / "invalid-numbering.cif",
            f"""data_numbering
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
A {ndb_number} 10
""",
        )

        with pytest.raises(ValueError, match=message):
            _read_mmcif_entity_declarations(cif_file)

    def test_nonpoly_numbering_rejects_a_true_key_conflict(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "conflicting-numbering.cif",
            """data_numbering
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
A 1 10
A 1 20
""",
        )

        with pytest.raises(ValueError, match="conflicting numbering for \\('A', 1\\)"):
            _read_mmcif_entity_declarations(cif_file)

    def test_read_structure_accepts_multiple_nonpoly_numbers_for_one_asym_id(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "multiple-nonpoly.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))

        scheme = cif_file.block["pdbx_nonpoly_scheme"]
        expanded_scheme = {}
        for column_name in scheme:
            values = scheme[column_name].as_array().tolist()
            if column_name == "ndb_seq_num":
                values.append("2")
            elif column_name in {"auth_seq_num", "pdb_seq_num"}:
                values.append("20")
            else:
                values.append(values[0])
            expanded_scheme[column_name] = pdbx.CIFColumn(values)
        cif_file.block["pdbx_nonpoly_scheme"] = pdbx.CIFCategory(expanded_scheme)

        atom_site = cif_file.block["atom_site"]
        original_row_count = atom_site.row_count
        expanded_atom_site = {}
        for column_name in atom_site:
            values = atom_site[column_name].as_array().tolist()
            extra = list(values)
            if column_name == "id":
                extra = [str(original_row_count + index + 1) for index in range(original_row_count)]
            elif column_name == "auth_seq_id":
                extra = ["20"] * original_row_count
            values.extend(extra)
            expanded_atom_site[column_name] = pdbx.CIFColumn(values)
        cif_file.block["atom_site"] = pdbx.CIFCategory(expanded_atom_site)
        cif_file.write(str(path))

        restored = read_structure(path)
        assert restored.system.instance_entity.numel() == 2

    def test_branch_definition_cannot_override_the_entity_category(
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

    def test_branch_scheme_keeps_label_pdb_and_author_namespaces_paired(self, tmp_path: Path) -> None:
        component = ChemicalComponent("X", (ChemicalAtom("A", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "branch-namespaces.cif",
            """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 X 1
1 X 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_asym_id
_pdbx_branch_scheme.auth_mon_id
_pdbx_branch_scheme.auth_seq_num
B 1 X 1 P PX 10 A AX 20
B 1 X 2 Q QX 20 C CX 30
""",
        )
        _, candidates = read_mmcif_branch_entities(cif_file, {"X": component})

        assert candidates[("pdb", "B", "1", "PX", "P", "10")] == frozenset({0})
        assert candidates[("author", "B", "1", "AX", "A", "20")] == frozenset({0})
        assert ("pdb", "B", "1", "PX", "A", "20") not in candidates
        assert candidates[("label", "B", "1", "X", "B", None)] == frozenset({0, 1})

        row = SimpleNamespace(
            chain_id="B",
            entity_id="1",
            component_id="X",
            auth_component_id="PX",
            auth_chain_id="P",
            auth_residue_id="10",
        )
        assert _resolve_branch_component_index(row, candidates, ("B", 10)) == 0

    def test_atom_site_auth_identity_uses_branch_pdb_namespace(self, tmp_path: Path) -> None:
        component = ChemicalComponent("NAG", (ChemicalAtom("C1", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "branch-auth-collision.cif",
            """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
2 NAG 1
2 NAG 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_asym_id
_pdbx_branch_scheme.auth_mon_id
_pdbx_branch_scheme.auth_seq_num
B 2 NAG 1 B NAG 1 B NAG 0
B 2 NAG 2 B NAG 2 B NAG 1
""",
        )
        _, candidates = read_mmcif_branch_entities(cif_file, {"NAG": component})
        assert candidates[("label", "B", "2", "NAG", "B", None)] == frozenset({0, 1})
        assert candidates[("pdb", "B", "2", "NAG", "B", "1")] == frozenset({0})
        assert candidates[("author", "B", "2", "NAG", "B", "1")] == frozenset({1})

        row = SimpleNamespace(
            chain_id="B",
            entity_id="2",
            component_id="NAG",
            auth_component_id="NAG",
            auth_chain_id="B",
            auth_residue_id="1",
        )
        assert _resolve_branch_component_index(row, candidates, (None, "B", 1, "1", None, "NAG")) == 0

    def test_branch_author_namespace_does_not_disambiguate_atom_site_auth(self) -> None:
        candidates = {
            ("label", "B", "2", "NAG", "B", None): frozenset({0, 1}),
            ("author", "B", "2", "NAG", "B", "1"): frozenset({0}),
        }
        row = SimpleNamespace(
            chain_id="B",
            entity_id="2",
            component_id="NAG",
            auth_component_id="NAG",
            auth_chain_id="B",
            auth_residue_id="1",
        )
        with pytest.raises(ValueError, match="maps ambiguously"):
            _resolve_branch_component_index(row, candidates, (None, "B", 1, "1", None, "NAG"))

    def test_branch_source_numbering_uses_pdb_sequence_number(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "branch-numbering.cif",
            """data_branch
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_seq_num
_pdbx_branch_scheme.pdb_ins_code
B 1 10 20 ?
""",
        )
        _, _, _, author_numbering, _, _ = _read_mmcif_entity_declarations(cif_file)
        assert author_numbering[("B", 1)] == ("10", None)

    def test_branch_source_numbering_does_not_fallback_to_auth_sequence_number(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "branch-numbering-no-fallback.cif",
            """data_branch
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_seq_num
_pdbx_branch_scheme.pdb_ins_code
B 1 ? 20 ?
""",
        )
        _, _, _, author_numbering, _, _ = _read_mmcif_entity_declarations(cif_file)
        assert author_numbering[("B", 1)] == (None, None)

    @pytest.mark.parametrize("observe_second_nag", [False, True])
    def test_repeated_branch_component_struct_conn_uses_pdb_identity(
        self,
        tmp_path: Path,
        observe_second_nag: bool,
    ) -> None:
        second_nag = (
            "HETATM 2 C C1 . NAG B 1 . ? 2.0 0.0 0.0 1.0 10.0 ? 2 NAG B C1 1\n"
            if observe_second_nag
            else ""
        )
        ligand_serial = 3 if observe_second_nag else 2
        path = tmp_path / "repeated-branch-struct-conn.cif"
        path.write_text(
            """data_branch
loop_
_entity.id
_entity.type
1 branched
2 non-polymer
loop_
_struct_asym.id
_struct_asym.entity_id
B 1
L 2
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 NAG 1
1 NAG 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_asym_id
_pdbx_branch_scheme.auth_mon_id
_pdbx_branch_scheme.auth_seq_num
_pdbx_branch_scheme.pdb_ins_code
B 1 NAG 1 B NAG 1 B NAG 0 ?
B 1 NAG 2 B NAG 2 B NAG 1 ?
loop_
_pdbx_entity_nonpoly.entity_id
_pdbx_entity_nonpoly.comp_id
2 LIG
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.entity_id
_pdbx_nonpoly_scheme.mon_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.pdb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
_pdbx_nonpoly_scheme.pdb_ins_code
L 2 LIG 1 1 1 ?
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_auth_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_atom_id
1 covale ? ? ? ? B NAG 1 C1 L LIG . C1 L LIG 1 C1
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
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
HETATM 1 C C1 . NAG B 1 . ? 1.0 0.0 0.0 1.0 10.0 ? 1 NAG B C1 1
"""
            + second_nag
            + f"HETATM {ligand_serial} C C1 . LIG L 2 . ? 3.0 0.0 0.0 1.0 10.0 ? 1 LIG L C1 1\n"
        )
        component = ChemicalComponent("NAG", (ChemicalAtom("C1", "C"),), frozenset())
        ligand = ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset())

        structure = read_structure(path, components={"NAG": component, "LIG": ligand})
        nag_0 = structure.system.atom_index(AtomSite("B", 0, "C1"))
        nag_1 = structure.system.atom_index(AtomSite("B", 1, "C1"))
        ligand_atom = structure.system.atom_index(AtomSite("L", 0, "C1"))

        assert bool(structure.coordinate_mask[nag_0])
        assert bool(structure.coordinate_mask[nag_1]) is observe_second_nag
        assert structure.coordinates[nag_0].tolist() == [1.0, 0.0, 0.0]
        if observe_second_nag:
            assert structure.coordinates[nag_1].tolist() == [2.0, 0.0, 0.0]
        assert {
            frozenset(edge)
            for edge in structure.system.edge_index.t().tolist()
        } == {frozenset((nag_0, ligand_atom))}

    def test_unused_branch_ambiguity_is_retained_until_a_residue_uses_it(self, tmp_path: Path) -> None:
        component = ChemicalComponent("X", (ChemicalAtom("A", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "branch-ambiguity.cif",
            """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 X 1
1 X 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
B 1 X 1
B 1 X 2
""",
        )
        _, candidates = read_mmcif_branch_entities(cif_file, {"X": component})
        assert candidates[("label", "B", "1", "X", "B", None)] == frozenset({0, 1})

        row = SimpleNamespace(
            chain_id="B",
            entity_id="1",
            component_id="X",
            auth_component_id=None,
            auth_chain_id=None,
            auth_residue_id=None,
        )
        with pytest.raises(ValueError, match="maps ambiguously"):
            _resolve_branch_component_index(row, candidates, ("B", 1))

    @pytest.mark.parametrize(
        ("number", "monomer", "message"),
        [("2", "X", "is not declared"), ("1", "Y", "disagrees with declaration")],
    )
    def test_branch_scheme_num_is_authoritative_and_mon_id_must_match(
        self,
        tmp_path: Path,
        number: str,
        monomer: str,
        message: str,
    ) -> None:
        component = ChemicalComponent("X", (ChemicalAtom("A", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "invalid-branch-scheme.cif",
            f"""data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 X 1
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
B 1 {monomer} {number}
""",
        )
        with pytest.raises(ValueError, match=message):
            read_mmcif_branch_entities(cif_file, {"X": component})
