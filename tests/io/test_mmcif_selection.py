from __future__ import annotations

import pytest
import torch
from biotite.structure.io import pdbx

from ofoldx.data.components import get_component
from ofoldx.data.molecular import BOND_ROLE_ASSEMBLY
from ofoldx.io import read_source, read_structure, write_structure

from .test_mmcif import _two_component_structure


def source(tmp_path):
    path = tmp_path / "selection.cif"
    write_structure(_two_component_structure(), path)
    cif = pdbx.CIFFile.read(str(path))
    cif.block["struct_conn"] = pdbx.CIFCategory(
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
            "ptnr2_symmetry": ["1_555"],
        }
    )
    return path, cif


@pytest.mark.parametrize("order,known", [(None, True), ("?", False), (".", False), ("doub", True)])
def test_order_default_and_null(tmp_path, order, known):
    # wwPDB _struct_conn.pdbx_value_order defaults to sing only when omitted.
    path, cif = source(tmp_path)
    if order is not None:
        cif.block["struct_conn"]["pdbx_value_order"] = pdbx.CIFColumn([order])
    cif.write(str(path))
    system = read_structure(path).system
    selected = system.bond_role == BOND_ROLE_ASSEMBLY
    assert selected.sum().item() == 1
    assert system.bond_order_known[selected].tolist() == [known]
    if known:
        assert system.bond_order[selected].tolist() == [2.0 if order == "doub" else 1.0]


def test_assembly_validates_source_connections(tmp_path):
    # assembly_gen uses struct_asym IDs; struct_conn refers to deposited atom_site.
    path, cif = source(tmp_path)
    cif.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
        {
            "assembly_id": ["1"],
            "oper_expression": ["1"],
            "asym_id_list": ["A"],
        }
    )
    columns = {"id": ["1"]}
    columns.update({f"matrix[{i}][{j}]": [str(int(i == j))] for i in range(1, 4) for j in range(1, 4)})
    columns.update({f"vector[{i}]": ["0"] for i in range(1, 4)})
    cif.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(columns)
    cif.write(str(path))
    full = read_structure(path)
    subset = read_structure(path, assembly_id="1")
    assert full.system.instance_symbol.numel() == 2
    assert subset.system.instance_symbol.numel() == 1
    assert (full.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1
    assert not (subset.system.bond_role == BOND_ROLE_ASSEMBLY).any()
    cif.block["struct_conn"]["ptnr2_label_atom_id"] = pdbx.CIFColumn(["DOES_NOT_EXIST"])
    cif.write(str(path))
    with pytest.raises(ValueError, match="cannot resolve endpoint 2"):
        read_structure(path, assembly_id="1")


@pytest.fixture
def charged_source(tmp_path):
    path, cif = source(tmp_path)
    cif.block["atom_site"]["pdbx_formal_charge"] = pdbx.CIFColumn(["1", "-1"])
    cif.block["ofoldx_source_formal_charge"] = pdbx.CIFCategory(
        {"label_asym_id": ["A", "B"], "component_index": ["0", "0"], "label_atom_id": ["C", "C"]}
    )
    cif.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
        {"assembly_id": ["1"], "oper_expression": ["1,2"], "asym_id_list": ["A"]}
    )
    columns = {"id": ["1", "2"]}
    columns.update({f"matrix[{i}][{j}]": [str(int(i == j))] * 2 for i in range(1, 4) for j in range(1, 4)})
    columns.update({f"vector[{i}]": ["0", "10" if i == 1 else "0"] for i in range(1, 4)})
    cif.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(columns)
    return path, cif


@pytest.mark.parametrize("assembly", [False, True])
def test_selection_projects_charge_evidence(charged_source, assembly):
    path, cif = charged_source
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native.view())
    selected = read_structure(native.view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None))
    copies = 2 if assembly else 1
    assert selected.system.instance_symbol.numel() == copies
    assert not (selected.system.bond_role == BOND_ROLE_ASSEMBLY).any()
    full_stop = int(full.system.component_atom_ptr[1])
    for component in range(copies):
        start, stop = selected.system.component_atom_ptr[component : component + 2].tolist()
        assert selected.source_formal_charge_mask[start:stop].tolist() == [True, False, False]
        for field in ("formal_charge", "formal_charge_known", "explicit_hydrogens", "explicit_hydrogens_known"):
            torch.testing.assert_close(
                getattr(selected.system, field)[start:stop], getattr(full.system, field)[:full_stop]
            )
    assert selected.coordinates[selected.coordinate_mask, 0].tolist() == ([0.0, 10.0] if assembly else [0.0])
    assert set(native.view().rows.chain_id) == {"A", "B"}


def test_empty_provenance_stays_explicit(charged_source):
    path, cif = charged_source
    cif.block["ofoldx_source_formal_charge"] = pdbx.CIFCategory(
        {"label_asym_id": ["B"], "component_index": ["0"], "label_atom_id": ["C"]}
    )
    cif.write(str(path))
    selected = read_structure(read_source(path).view(asym_ids=frozenset({"A"})))
    assert selected.system.formal_charge[0].item() == 1
    assert not selected.source_formal_charge_mask.any()


@pytest.mark.parametrize("chain", ["A", "B", "ABSENT"])
def test_selection_rejects_invalid_provenance(charged_source, chain):
    path, cif = charged_source
    cif.block["ofoldx_source_formal_charge"] = pdbx.CIFCategory(
        {"label_asym_id": [chain], "component_index": ["0"], "label_atom_id": ["MISSING"]}
    )
    cif.write(str(path))
    with pytest.raises(ValueError, match="references unknown atom"):
        read_structure(read_source(path).view(asym_ids=frozenset({"A"})))


@pytest.fixture
def reacting_source(charged_source):
    path, cif = charged_source
    connection = cif.block["struct_conn"]
    connection["pdbx_leaving_atom_flag"] = pdbx.CIFColumn(["both"])
    connection["pdbx_ptnr1_leaving_atom_id"] = pdbx.CIFColumn(["O1"])
    connection["pdbx_ptnr2_leaving_atom_id"] = pdbx.CIFColumn(["O2"])
    return path, cif


@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("hydrogens", [False, True])
def test_boundary_preserves_reacted_chemistry(reacting_source, assembly, hydrogens):
    path, cif = reacting_source
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native.view(), include_hydrogens=hydrogens)
    selected = read_structure(
        native.view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None),
        include_hydrogens=hydrogens,
    )
    copies = 2 if assembly else 1
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == ["C"] * copies
    assert selected.system.edge_index.shape[1] == 0
    for field in (
        "atomic_number",
        "isotope",
        "isotope_known",
        "formal_charge",
        "formal_charge_known",
        "explicit_hydrogens",
        "explicit_hydrogens_known",
        "implicit_hydrogens",
        "implicit_hydrogens_known",
        "implicit_hydrogens_allowed",
        "implicit_hydrogens_allowed_known",
        "atom_aromatic",
        "atom_aromatic_known",
    ):
        torch.testing.assert_close(getattr(selected.system, field), getattr(full.system, field)[:1].repeat(copies))
    assert selected.source_formal_charge_mask.tolist() == [True] * copies
    assert selected.coordinates[:, 0].tolist() == ([0.0, 10.0] if assembly else [0.0])


@pytest.mark.parametrize("assembly", [False, True])
def test_crystal_boundary_preserves_reaction(reacting_source, assembly):
    path, cif = reacting_source
    cif.block["cell"] = pdbx.CIFCategory(
        {
            **{f"length_{axis}": [10.0] for axis in "abc"},
            **{f"angle_{angle}": [90.0] for angle in ("alpha", "beta", "gamma")},
        }
    )
    fractional = {f"fract_transf_matrix[{i}][{j}]": [0.1 if i == j else 0.0] for i in range(1, 4) for j in range(1, 4)}
    fractional.update({f"fract_transf_vector[{i}]": [0.0] for i in range(1, 4)})
    cif.block["atom_sites"] = pdbx.CIFCategory(fractional)
    cif.block["space_group_symop"] = pdbx.CIFCategory({"id": ["1", "2"], "operation_xyz": ["x,y,z", "-x+1/2,-y+1/2,z"]})
    cif.block["struct_conn"]["ptnr1_symmetry"] = pdbx.CIFColumn(["2_555"])
    for axis, values in zip("xyz", ([1.0, 2.0], [2.0, 0.0], [3.0, 0.0]), strict=True):
        cif.block["atom_site"][f"Cartn_{axis}"] = pdbx.CIFColumn(values)
    operations = cif.block["pdbx_struct_oper_list"]
    for axis in (1, 2):
        operations[f"matrix[{axis}][{axis}]"] = pdbx.CIFColumn([1.0, -1.0])
        operations[f"vector[{axis}]"] = pdbx.CIFColumn([0.0, 5.0])
    cif.write(str(path))

    native = read_source(path)
    full = read_structure(native.view())
    selected = read_structure(native.view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None))
    full_a = [
        atom
        for component, symbol in enumerate(full.system.component_symbol.tolist())
        if full.system.symbols[symbol] == "L1"
        for atom in range(*full.system.component_atom_ptr[component : component + 2].tolist())
    ]
    copies = 2 if assembly else 1
    assert [full.system.atom_site(i).atom_id for i in full_a] == ["C", "C"]
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == ["C"] * copies
    assert (full.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1
    assert selected.system.edge_index.shape[1] == 0
    assert full.system.formal_charge[full_a].tolist() == [1, 1]
    for field in (
        "formal_charge",
        "formal_charge_known",
        "explicit_hydrogens",
        "explicit_hydrogens_known",
        "implicit_hydrogens",
        "implicit_hydrogens_known",
        "implicit_hydrogens_allowed",
        "implicit_hydrogens_allowed_known",
    ):
        torch.testing.assert_close(getattr(selected.system, field), getattr(full.system, field)[full_a][:copies])
    assert full.source_formal_charge_mask[full_a].tolist() == [True, True]
    assert selected.source_formal_charge_mask.tolist() == [True] * copies
    expected = [[1.0, 2.0, 3.0], [4.0, 3.0, 3.0]]
    assert full.coordinate_mask[full_a].tolist() == [True, True]
    assert selected.coordinate_mask.tolist() == [True] * copies
    assert full.coordinates[full_a].tolist() == expected
    assert selected.coordinates.tolist() == expected[:copies]


def append_row(category, values):
    for name, column in list(category.items()):
        entries = column.as_array(str).tolist()
        category[name] = pdbx.CIFColumn([*entries, values.get(name, entries[0])])


def test_null_coordinate_retains_atom_evidence(reacting_source):
    path, cif = reacting_source
    append_row(
        cif.block["atom_site"],
        {
            "id": "3",
            "label_atom_id": "O1",
            "auth_atom_id": "O1",
            "type_symbol": "O",
            "Cartn_x": "?",
            "Cartn_y": "?",
            "Cartn_z": "?",
            "pdbx_formal_charge": "?",
        },
    )
    cif.write(str(path))
    native = read_source(path)
    assert native.view().rows.coordinate_present.tolist() == [True, True, False]
    with pytest.raises(KeyError, match="unrealized molecular atom"):
        read_structure(native.view(asym_ids=frozenset({"A"})))


def test_removed_provenance_still_fails(reacting_source):
    path, cif = reacting_source
    append_row(cif.block["ofoldx_source_formal_charge"], {"label_atom_id": "O1"})
    cif.write(str(path))
    with pytest.raises(ValueError, match="references unrealized atom"):
        read_structure(read_source(path).view(asym_ids=frozenset({"A"})))


@pytest.mark.parametrize("assembly", [False, True])
def test_unrelated_chemistry_is_not_requested(reacting_source, assembly):
    path, cif = reacting_source
    append_row(cif.block["struct_asym"], {"id": "C", "entity_id": "3"})
    append_row(cif.block["entity"], {"id": "3", "type": "non-polymer"})
    append_row(cif.block["pdbx_entity_nonpoly"], {"entity_id": "3", "comp_id": "EXC"})
    append_row(
        cif.block["pdbx_nonpoly_scheme"],
        {
            "asym_id": "C",
            "entity_id": "3",
            "mon_id": "EXC",
            "pdb_mon_id": "EXC",
            "auth_mon_id": "EXC",
            "pdb_strand_id": "C",
            "pdb_seq_num": "3",
            "auth_seq_num": "3",
            "ndb_seq_num": "3",
        },
    )
    append_row(
        cif.block["atom_site"],
        {
            "id": "3",
            "label_asym_id": "C",
            "auth_asym_id": "C",
            "label_entity_id": "3",
            "label_comp_id": "EXC",
            "auth_comp_id": "EXC",
            "auth_seq_id": "3",
        },
    )
    cif.write(str(path))

    def resolver(identifier):
        if identifier == "EXC":
            raise RuntimeError("unrelated component dictionary is unavailable")
        raise KeyError(identifier)

    view = read_source(path).view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None)
    selected = read_structure(view, component_resolver=resolver)
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == ["C"] * (
        2 if assembly else 1
    )


@pytest.mark.parametrize("altloc,reacted", [("first", True), ("highest_occupancy", False)])
@pytest.mark.parametrize("assembly", [False, True])
def test_boundary_respects_selected_conformation(reacting_source, altloc, reacted, assembly):
    path, cif = reacting_source
    atoms = cif.block["atom_site"]
    atoms["label_alt_id"] = pdbx.CIFColumn([".", "B"])
    atoms["occupancy"] = pdbx.CIFColumn(["1", "0.1"])
    alternative = {name: column.as_array(str)[1] for name, column in atoms.items()}
    alternative.update(id="3", label_alt_id="A", occupancy="0.9")
    append_row(atoms, alternative)
    cif.block["struct_conn"]["pdbx_ptnr2_label_alt_id"] = pdbx.CIFColumn(["B"])
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native.view(), altloc=altloc)
    selected = read_structure(
        native.view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None), altloc=altloc
    )
    expected = ["C"] if reacted else ["C", "O1", "H1"]
    stop = int(full.system.component_atom_ptr[1])
    assert [full.system.atom_site(i).atom_id for i in range(stop)] == expected
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == expected * (
        2 if assembly else 1
    )


@pytest.mark.parametrize("conflicting", [False, True])
def test_boundary_checks_duplicate_declarations(reacting_source, conflicting):
    path, cif = reacting_source
    values = {"id": "2"}
    if conflicting:
        values.update(pdbx_leaving_atom_flag="none", pdbx_ptnr1_leaving_atom_id="?", pdbx_ptnr2_leaving_atom_id="?")
    append_row(cif.block["struct_conn"], values)
    cif.write(str(path))
    view = read_source(path).view(asym_ids=frozenset({"A"}))
    if conflicting:
        with pytest.raises(ValueError, match="conflicting declarations"):
            read_structure(view)
    else:
        structure = read_structure(view)
        assert [structure.system.atom_site(i).atom_id for i in range(structure.system.atom_symbol.numel())] == ["C"]


@pytest.fixture
def unobserved_source(tmp_path):
    path = tmp_path / "unobserved.cif"
    path.write_text("""data_boundary
loop_
_entity.id
_entity.type
1 non-polymer
2 polymer
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
loop_
_pdbx_entity_nonpoly.entity_id
_pdbx_entity_nonpoly.comp_id
1 ZN
loop_
_entity_poly.entity_id
_entity_poly.type
2 'polypeptide(L)'
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
2 1 CYS n
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
B 2 1 CYS Q 7 7 .
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
HETATM 1 ZN ZN . ZN A 1 . ? 0 0 0 1 10 1 ZN Z ZN 1
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_atom_id
_struct_conn.ptnr2_symmetry
1 metalc A ZN ZN 1_555 ? ? SG Q 7 CYS SG 1_555
""")
    return path, pdbx.CIFFile.read(str(path))


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("author_atom", ["SG", "?"])
def test_boundary_resolves_only_required_chemistry(unobserved_source, observed, assembly, author_atom):
    path, cif = unobserved_source
    cif.block["struct_conn"]["ptnr2_auth_atom_id"] = pdbx.CIFColumn([author_atom])
    append_row(cif.block["entity_poly_seq"], {"num": "2", "mon_id": "ALA"})
    append_row(
        cif.block["pdbx_poly_seq_scheme"], {"seq_id": "2", "mon_id": "ALA", "pdb_seq_num": "8", "auth_seq_num": "8"}
    )
    append_row(cif.block["entity"], {"id": "3", "type": "non-polymer"})
    append_row(cif.block["struct_asym"], {"id": "C", "entity_id": "3"})
    append_row(cif.block["pdbx_entity_nonpoly"], {"entity_id": "3", "comp_id": "TYR"})
    if observed:
        append_row(
            cif.block["atom_site"],
            {
                "id": "2",
                "type_symbol": "C",
                "label_atom_id": "CA",
                "auth_atom_id": "CA",
                "label_comp_id": "CYS",
                "auth_comp_id": "CYS",
                "label_asym_id": "B",
                "auth_asym_id": "Q",
                "label_entity_id": "2",
                "label_seq_id": "1",
                "auth_seq_id": "7",
            },
        )
    cif.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
        {"assembly_id": ["1"], "oper_expression": ["1"], "asym_id_list": ["A"]}
    )
    columns = {"id": ["1"]}
    columns.update({f"matrix[{i}][{j}]": [str(int(i == j))] for i in range(1, 4) for j in range(1, 4)})
    columns.update({f"vector[{i}]": ["0"] for i in range(1, 4)})
    cif.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(columns)
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native)
    assert full.system.instance_symbol.numel() == 3
    assert (full.system.bond_role == BOND_ROLE_ASSEMBLY).sum().item() == 1
    calls = []

    def resolver(identifier):
        calls.append(identifier)
        assert identifier in {"ZN", "CYS"}, "excluded, unconnected chemistry must not be loaded"
        return get_component(identifier)

    view = native.view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None)
    selected = read_structure(view, component_resolver=resolver)
    assert set(calls) == {"ZN", "CYS"}
    calls.clear()
    explicit = read_structure(view, components={"CYS": get_component("CYS")}, component_resolver=resolver)
    assert set(calls) == {"ZN"}
    torch.testing.assert_close(selected.state_dict(), explicit.state_dict(), rtol=0, atol=0, equal_nan=True)
    assert selected.system.instance_symbol.numel() == 1
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == ["ZN"]
    assert selected.system.edge_index.shape[1] == 0
    assert selected.coordinate_mask.tolist() == [True]
    assert selected.coordinates.tolist() == [[0.0, 0.0, 0.0]]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "invalid", ["unknown_atom", "unknown_author_atom", "conflicting_chain", "ambiguous_occurrence"]
)
def test_boundary_rejects_invalid_identity(unobserved_source, explicit, invalid):
    path, cif = unobserved_source
    connection = cif.block["struct_conn"]
    if invalid == "unknown_atom":
        connection["ptnr2_label_atom_id"] = pdbx.CIFColumn(["MISSING"])
        message = "conflicting label/auth atom identities"
    elif invalid == "unknown_author_atom":
        connection["ptnr2_label_atom_id"] = pdbx.CIFColumn(["?"])
        message = "insufficient identity evidence"
    elif invalid == "conflicting_chain":
        connection["ptnr2_label_asym_id"] = pdbx.CIFColumn(["A"])
        message = "conflicting label/auth atom identities"
    else:
        append_row(cif.block["struct_asym"], {"id": "D", "entity_id": "2"})
        append_row(cif.block["pdbx_poly_seq_scheme"], {"asym_id": "D"})
        message = "cannot uniquely identify its label-asym placement|ambiguous"
    cif.write(str(path))
    with pytest.raises(ValueError, match=message):
        read_structure(
            read_source(path).view(asym_ids=frozenset({"A"})),
            components={"CYS": get_component("CYS")} if explicit else None,
            component_resolver=get_component,
        )


@pytest.mark.parametrize("author_atom", ["SULFUR", "SG"])
def test_boundary_checks_observed_author_atom(unobserved_source, author_atom):
    path, cif = unobserved_source
    append_row(
        cif.block["atom_site"],
        {
            "id": "2",
            "type_symbol": "S",
            "label_atom_id": "SG",
            "auth_atom_id": "SULFUR",
            "label_comp_id": "CYS",
            "auth_comp_id": "CYS",
            "label_asym_id": "B",
            "auth_asym_id": "Q",
            "label_entity_id": "2",
            "label_seq_id": "1",
            "auth_seq_id": "7",
        },
    )
    cif.block["struct_conn"]["ptnr2_auth_atom_id"] = pdbx.CIFColumn([author_atom])
    cif.write(str(path))
    view = read_source(path).view(asym_ids=frozenset({"A"}))
    if author_atom == "SULFUR":
        assert read_structure(view).system.instance_symbol.numel() == 1
    else:
        with pytest.raises(ValueError, match="conflicting label/auth atom identities"):
            read_structure(view)


@pytest.mark.parametrize("altloc,occupancy", [("first", 0.1), ("highest_occupancy", 0.9)])
def test_boundary_validation_precedes_altloc(unobserved_source, altloc, occupancy):
    path, cif = unobserved_source
    atoms = cif.block["atom_site"]
    atoms["label_alt_id"] = pdbx.CIFColumn(["B"])
    atoms["occupancy"] = pdbx.CIFColumn(["0.1"])
    append_row(atoms, {"id": "2", "label_alt_id": "A", "occupancy": "0.9"})
    cif.block["struct_conn"]["pdbx_ptnr1_label_alt_id"] = pdbx.CIFColumn(["B"])
    cif.write(str(path))
    calls = []

    def resolver(identifier):
        calls.append(identifier)
        assert identifier in {"ZN", "CYS"}
        return get_component(identifier)

    selected = read_structure(
        read_source(path).view(asym_ids=frozenset({"A"})), altloc=altloc, component_resolver=resolver
    )
    assert set(calls) == {"ZN", "CYS"}
    assert selected.system.instance_symbol.numel() == 1
    assert selected.occupancy.tolist() == [occupancy]


def test_excluded_connection_validates_source(unobserved_source):
    path, cif = unobserved_source
    append_row(cif.block["entity"], {"id": "3", "type": "non-polymer"})
    append_row(cif.block["struct_asym"], {"id": "C", "entity_id": "3"})
    append_row(cif.block["pdbx_entity_nonpoly"], {"entity_id": "3", "comp_id": "ALA"})
    cif.write(str(path))
    view = read_source(path).view(asym_ids=frozenset({"C"}))
    calls = []

    def resolver(identifier):
        calls.append(identifier)
        return get_component(identifier)

    lazy = read_structure(view, component_resolver=resolver)
    assert set(calls) == {"ALA", "CYS", "ZN"}
    explicit = read_structure(view, components={"CYS": get_component("CYS")})
    torch.testing.assert_close(lazy.state_dict(), explicit.state_dict(), rtol=0, atol=0, equal_nan=True)
    assert lazy.system.instance_symbol.numel() == 1
    assert not (lazy.system.bond_role == BOND_ROLE_ASSEMBLY).any()
    cif.block["struct_conn"]["ptnr2_label_atom_id"] = pdbx.CIFColumn(["MISSING"])
    cif.write(str(path))
    with pytest.raises(ValueError, match="conflicting label/auth atom identities"):
        read_structure(read_source(path).view(asym_ids=frozenset({"C"})))


@pytest.mark.parametrize("crystal", [False, True])
def test_atom_evidence_disambiguates_occurrence(unobserved_source, crystal):
    path, cif = unobserved_source
    append_row(cif.block["entity"], {"id": "3", "type": "non-polymer"})
    append_row(cif.block["struct_asym"], {"id": "C", "entity_id": "3"})
    append_row(cif.block["pdbx_entity_nonpoly"], {"entity_id": "3", "comp_id": "ALA"})
    cif.block["struct_conn"]["ptnr2_auth_comp_id"] = pdbx.CIFColumn(["?"])
    cif.block["struct_conn"]["ptnr2_auth_atom_id"] = pdbx.CIFColumn(["?"])
    if crystal:
        add_crystal_translation(cif)
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native, component_resolver=get_component)
    connected = (full.system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
    assert len(connected) == 1
    edge = full.system.edge_index[:, connected[0]].tolist()
    assert {full.system.atom_site(atom).atom_id for atom in edge} == {"ZN", "SG"}
    zinc = next(atom for atom in edge if full.system.atom_site(atom).atom_id == "ZN")
    sulfur = next(atom for atom in edge if full.system.atom_site(atom).atom_id == "SG")
    assert full.system.atom_site(sulfur).instance_id == "B"
    assert not full.coordinate_mask[sulfur]
    assert full.coordinates[zinc].tolist() == [5.0 if crystal else 0.0, 0.0, 0.0]
    assert full.coordinate_mask.sum().item() == (2 if crystal else 1)
    assert "HG" in {full.system.atom_site(i).atom_id for i in range(full.system.atom_symbol.numel())}
    full_explicit = read_structure(native, components={"CYS": get_component("CYS"), "ALA": get_component("ALA")})
    torch.testing.assert_close(full.state_dict(), full_explicit.state_dict(), rtol=0, atol=0, equal_nan=True)
    view = native.view(asym_ids=frozenset({"A"}))
    lazy = read_structure(view, component_resolver=get_component)
    explicit = read_structure(view, components={"CYS": get_component("CYS"), "ALA": get_component("ALA")})
    torch.testing.assert_close(lazy.state_dict(), explicit.state_dict(), rtol=0, atol=0, equal_nan=True)
    assert lazy.system.instance_symbol.numel() == 1
    assert lazy.coordinate_mask.tolist() == [True]
    assert lazy.coordinates.tolist() == [[0.0, 0.0, 0.0]]
    assert lazy.system.edge_index.shape[1] == 0


@pytest.fixture(params=["partial_polymer_scheme", "branch_without_scheme"])
def incomplete_scheme_source(unobserved_source, request):
    path, cif = unobserved_source
    if request.param == "partial_polymer_scheme":
        append_row(cif.block["entity_poly_seq"], {"num": "2", "mon_id": "ALA"})
        for field, value in {"seq_id": "2", "mon_id": "ALA", "pdb_seq_num": "8", "auth_seq_num": "8"}.items():
            cif.block["pdbx_poly_seq_scheme"][field] = pdbx.CIFColumn([value])
        number = "1"
    else:
        cif.block["entity"]["type"] = pdbx.CIFColumn(["non-polymer", "branched"])
        for category in ("entity_poly", "entity_poly_seq", "pdbx_poly_seq_scheme"):
            del cif.block[category]
        cif.block["pdbx_entity_branch_list"] = pdbx.CIFCategory({"entity_id": ["2"], "num": ["1"], "comp_id": ["CYS"]})
        number = "."
    connection = cif.block["struct_conn"]
    for field, value in {"asym_id": "B", "seq_id": number, "comp_id": "CYS"}.items():
        connection[f"ptnr2_label_{field}"] = pdbx.CIFColumn([value])
    for field in ("asym_id", "seq_id", "comp_id", "atom_id"):
        connection[f"ptnr2_auth_{field}"] = pdbx.CIFColumn(["?"])
    append_row(cif.block["entity"], {"id": "3", "type": "non-polymer"})
    append_row(cif.block["struct_asym"], {"id": "C", "entity_id": "3"})
    append_row(cif.block["pdbx_entity_nonpoly"], {"entity_id": "3", "comp_id": "TYR"})
    return path, cif


def test_boundary_resolves_incomplete_scheme(incomplete_scheme_source):
    path, cif = incomplete_scheme_source
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native)
    connected = (full.system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
    assert len(connected) == 1
    edge = full.system.edge_index[:, connected[0]].tolist()
    assert {(full.system.atom_site(i).instance_id, full.system.atom_site(i).atom_id) for i in edge} == {
        ("A", "ZN"),
        ("B", "SG"),
    }
    assert full.coordinate_mask.sum().item() == 1
    assert full.coordinates[full.coordinate_mask].tolist() == [[0.0, 0.0, 0.0]]
    assert "HG" in {full.system.atom_site(i).atom_id for i in range(full.system.atom_symbol.numel())}
    calls = []

    def resolver(identifier):
        calls.append(identifier)
        assert identifier in {"ZN", "CYS"}, "unconnected excluded components must not be loaded"
        return get_component(identifier)

    view = native.view(asym_ids=frozenset({"A"}))
    lazy = read_structure(view, component_resolver=resolver)
    assert set(calls) == {"ZN", "CYS"}
    calls.clear()
    explicit = read_structure(view, components={"CYS": get_component("CYS")}, component_resolver=resolver)
    assert calls == ["ZN"]
    torch.testing.assert_close(lazy.state_dict(), explicit.state_dict(), rtol=0, atol=0, equal_nan=True)
    assert lazy.coordinate_mask.tolist() == [True]
    assert lazy.coordinates.tolist() == [[0.0, 0.0, 0.0]]
    assert lazy.system.edge_index.shape[1] == 0


@pytest.mark.parametrize("connected", [False, True])
def test_observation_preserves_occurrence_identity(incomplete_scheme_source, connected):
    path, cif = incomplete_scheme_source
    if not connected:
        del cif.block["struct_conn"]
    cif.write(str(path))
    before = read_structure(read_source(path).view(asym_ids=frozenset({"A"})))
    append_row(
        cif.block["atom_site"],
        {
            "id": "2",
            "type_symbol": "C",
            "label_atom_id": "CA",
            "auth_atom_id": "CA",
            "label_comp_id": "CYS",
            "auth_comp_id": "CYS",
            "label_asym_id": "B",
            "auth_asym_id": "Q",
            "label_entity_id": "2",
            "label_seq_id": "1" if "entity_poly_seq" in cif.block else ".",
            "auth_seq_id": "7",
            "pdbx_PDB_ins_code": "A",
            "Cartn_x": "1",
        },
    )
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native)
    observed = {
        (full.system.atom_site(i).instance_id, full.system.atom_site(i).atom_id): full.coordinates[i].tolist()
        for i in full.coordinate_mask.nonzero().flatten().tolist()
    }
    assert observed == {("A", "ZN"): [0.0, 0.0, 0.0], ("B", "CA"): [1.0, 0.0, 0.0]}
    branch_atoms = {
        full.system.atom_site(i).atom_id: i
        for i in range(full.system.atom_symbol.numel())
        if full.system.atom_site(i).instance_id == "B"
    }
    observed_ca = next(
        i for i in full.coordinate_mask.nonzero().flatten().tolist() if full.system.atom_site(i).atom_id == "CA"
    )
    assert full.system.atom_site(observed_ca).component_index == 0
    assert not full.coordinate_mask[branch_atoms["SG"]]
    assert "HG" in branch_atoms
    edges = (full.system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
    assert len(edges) == int(connected)
    if connected:
        assert {
            (full.system.atom_site(i).instance_id, full.system.atom_site(i).atom_id)
            for i in full.system.edge_index[:, edges[0]].tolist()
        } == {("A", "ZN"), ("B", "SG")}
    calls = []

    def resolver(identifier):
        calls.append(identifier)
        assert identifier in {"ZN", "CYS"}
        return get_component(identifier)

    view = native.view(asym_ids=frozenset({"A"}))
    lazy = read_structure(view, component_resolver=resolver)
    assert set(calls) == ({"ZN", "CYS"} if connected else {"ZN"})
    calls.clear()
    explicit = read_structure(view, components={"CYS": get_component("CYS")}, component_resolver=resolver)
    assert calls == ["ZN"]
    torch.testing.assert_close(before.state_dict(), lazy.state_dict(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(lazy.state_dict(), explicit.state_dict(), rtol=0, atol=0, equal_nan=True)
    assert lazy.coordinate_mask.tolist() == [True]
    assert lazy.coordinates.tolist() == [[0.0, 0.0, 0.0]]
    assert lazy.system.edge_index.shape[1] == 0


def test_unknown_author_identity_stays_unknown(incomplete_scheme_source):
    path, cif = incomplete_scheme_source
    connection = cif.block["struct_conn"]
    for field in ("asym_id", "seq_id", "comp_id", "atom_id"):
        connection[f"ptnr2_label_{field}"] = pdbx.CIFColumn(["?"])
    for field, value in {"asym_id": "Q", "seq_id": "7", "comp_id": "CYS", "atom_id": "SG"}.items():
        connection[f"ptnr2_auth_{field}"] = pdbx.CIFColumn([value])
    cif.write(str(path))
    for chains in (None, frozenset({"A"})):
        with pytest.raises(ValueError, match="insufficient identity evidence"):
            read_structure(read_source(path).view(asym_ids=chains))


@pytest.fixture(params=["branch", "polymer"])
def ambiguous_endpoint_source(unobserved_source, request):
    path, cif = unobserved_source
    if request.param == "branch":
        cif.block["entity"]["type"] = pdbx.CIFColumn(["non-polymer", "branched"])
        for category in ("entity_poly", "entity_poly_seq", "pdbx_poly_seq_scheme"):
            del cif.block[category]
        cif.block["pdbx_entity_branch_list"] = pdbx.CIFCategory(
            {"entity_id": ["2", "2"], "num": ["1", "2"], "comp_id": ["CYS", "CYS"]}
        )
    else:
        append_row(cif.block["entity_poly_seq"], {"num": "2"})
        append_row(cif.block["pdbx_poly_seq_scheme"], {"seq_id": "2", "pdb_seq_num": "8", "auth_seq_num": "8"})
    connection = cif.block["struct_conn"]
    connection["ptnr2_label_asym_id"] = pdbx.CIFColumn(["B"])
    connection["ptnr2_label_comp_id"] = pdbx.CIFColumn(["CYS"])
    for field in ("asym_id", "seq_id", "comp_id", "atom_id"):
        connection[f"ptnr2_auth_{field}"] = pdbx.CIFColumn(["?"])
    return path, cif


@pytest.mark.parametrize("observed,connected", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("ambiguous_endpoint_source", ["branch"], indirect=True)
def test_repeated_branch_components_remain_ambiguous(ambiguous_endpoint_source, observed, connected):
    path, cif = ambiguous_endpoint_source
    if observed:
        append_row(
            cif.block["atom_site"],
            {
                "id": "2",
                "type_symbol": "C",
                "label_atom_id": "CA",
                "auth_atom_id": "CA",
                "label_comp_id": "CYS",
                "auth_comp_id": "CYS",
                "label_asym_id": "B",
                "auth_asym_id": "Q",
                "label_entity_id": "2",
                "auth_seq_id": "7",
                "pdbx_PDB_ins_code": "A",
            },
        )
    if not connected:
        del cif.block["struct_conn"]
    cif.write(str(path))
    selections = (None, frozenset({"A"})) if connected else (None,)
    for chains in selections:
        with pytest.raises(ValueError, match="ambiguous"):
            read_structure(read_source(path).view(asym_ids=chains))


@pytest.mark.parametrize("assembly", [False, True], ids=["identity", "assembly"])
@pytest.mark.parametrize("explicit", [False, True], ids=["lazy", "explicit"])
def test_projection_preserves_endpoint_validation(ambiguous_endpoint_source, assembly, explicit):
    path, cif = ambiguous_endpoint_source
    append_row(cif.block["entity"], {"id": "3", "type": "non-polymer"})
    append_row(cif.block["struct_asym"], {"id": "C", "entity_id": "3"})
    append_row(cif.block["pdbx_entity_nonpoly"], {"entity_id": "3", "comp_id": "ALA"})
    append_row(
        cif.block["atom_site"],
        {
            "id": "2",
            "type_symbol": "C",
            "label_atom_id": "CA",
            "auth_atom_id": "CA",
            "label_comp_id": "ALA",
            "auth_comp_id": "ALA",
            "label_asym_id": "C",
            "auth_asym_id": "C",
            "label_entity_id": "3",
            "auth_seq_id": "3",
            "Cartn_x": "9",
        },
    )
    if assembly:
        cif.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
            {"assembly_id": ["1"], "oper_expression": ["1"], "asym_id_list": ["A,B,C"]}
        )
        columns = {"id": ["1"]}
        columns.update({f"matrix[{i}][{j}]": [str(int(i == j))] for i in range(1, 4) for j in range(1, 4)})
        columns.update({f"vector[{i}]": ["10" if i == 1 else "0"] for i in range(1, 4)})
        cif.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(columns)
    cif.write(str(path))
    native = read_source(path)
    components = {name: get_component(name) for name in ("CYS", "ALA")} if explicit else None
    selections = (None, frozenset({"A"}), frozenset({"C"}))
    for chains in selections:
        view = native.view(asym_ids=chains, assembly_id="1" if assembly else None)
        with pytest.raises(ValueError, match="ambiguous"):
            read_structure(view, components=components, component_resolver=get_component)

    if "pdbx_entity_branch_list" in cif.block:
        cif.block["pdbx_entity_branch_list"] = pdbx.CIFCategory({"entity_id": ["2"], "num": ["1"], "comp_id": ["CYS"]})
    else:
        cif.block["struct_conn"]["ptnr2_label_seq_id"] = pdbx.CIFColumn(["1"])
    cif.write(str(path))
    native = read_source(path)
    shift = 10.0 if assembly else 0.0
    for chains in selections:
        view = native.view(asym_ids=chains, assembly_id="1" if assembly else None)
        structure = read_structure(view, components=components, component_resolver=get_component)
        system = structure.system
        expected = {}
        if chains is None or "A" in chains:
            expected["ZN"] = [shift, 0.0, 0.0]
        if chains is None or "C" in chains:
            expected["CA"] = [9.0 + shift, 0.0, 0.0]
        assert {
            system.atom_site(i).atom_id: structure.coordinates[i].tolist()
            for i in structure.coordinate_mask.nonzero().flatten().tolist()
        } == expected
        edges = (system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
        assert len(edges) == int(chains is None)
        if edges:
            endpoints = {system.atom_site(i).atom_id: i for i in system.edge_index[:, edges[0]].tolist()}
            assert set(endpoints) == {"ZN", "SG"}
            sulfur = endpoints["SG"]
            sulfur_site = system.atom_site(sulfur)
            assert sulfur_site.component_index == 0
            assert not structure.coordinate_mask[sulfur]
            assert "HG" in {
                site.atom_id
                for i in range(system.atom_symbol.numel())
                if (site := system.atom_site(i)).instance_id == sulfur_site.instance_id
                and site.component_index == sulfur_site.component_index
            }


@pytest.fixture
def alternate_disulfide_source(unobserved_source):
    path, cif = unobserved_source
    cif.block["pdbx_entity_nonpoly"]["comp_id"] = pdbx.CIFColumn(["CYS"])
    atoms = cif.block["atom_site"]
    for field, value in {
        "type_symbol": "S",
        "label_atom_id": "SG",
        "auth_atom_id": "SG",
        "label_comp_id": "CYS",
        "auth_comp_id": "CYS",
    }.items():
        atoms[field] = pdbx.CIFColumn([value])
    for identifier, alternate, occupancy in (("2", "B", "0.1"), ("3", "A", "0.9")):
        append_row(
            atoms,
            {
                "id": identifier,
                "type_symbol": "C",
                "label_atom_id": "CA",
                "auth_atom_id": "CA",
                "label_asym_id": "B",
                "auth_asym_id": "Q",
                "label_entity_id": "2",
                "label_seq_id": "1",
                "auth_seq_id": "7",
                "label_alt_id": alternate,
                "occupancy": occupancy,
            },
        )
    connection = cif.block["struct_conn"]
    connection["conn_type_id"] = pdbx.CIFColumn(["disulf"])
    connection["ptnr1_label_comp_id"] = pdbx.CIFColumn(["CYS"])
    connection["ptnr1_label_atom_id"] = pdbx.CIFColumn(["SG"])
    connection["pdbx_ptnr2_label_alt_id"] = pdbx.CIFColumn(["B"])
    append_row(cif.block["entity"], {"id": "3", "type": "non-polymer"})
    append_row(cif.block["struct_asym"], {"id": "C", "entity_id": "3"})
    append_row(cif.block["pdbx_entity_nonpoly"], {"entity_id": "3", "comp_id": "ALA"})
    return path, cif


@pytest.mark.parametrize("placement", ["identity", "crystal", "assembly"])
@pytest.mark.parametrize(
    "chains",
    [None, frozenset({"A"}), frozenset({"B"}), frozenset({"C"})],
    ids=["full", "first_partner", "second_partner", "neither_partner"],
)
@pytest.mark.parametrize("altloc", ["first", "highest_occupancy"])
@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_alternate_requires_source_atom(alternate_disulfide_source, placement, chains, altloc, explicit):
    path, cif = alternate_disulfide_source
    if placement == "crystal":
        add_crystal_translation(cif)
    elif placement == "assembly":
        cif.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
            {"assembly_id": ["1"], "oper_expression": ["1"], "asym_id_list": ["A,B,C"]}
        )
        columns = {"id": ["1"]}
        columns.update({f"matrix[{i}][{j}]": [str(int(i == j))] for i in range(1, 4) for j in range(1, 4)})
        columns.update({f"vector[{i}]": ["0"] for i in range(1, 4)})
        cif.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(columns)
    cif.write(str(path))
    view = read_source(path).view(asym_ids=chains, assembly_id="1" if placement == "assembly" else None)
    # CA_B is only a local alternate; it cannot establish an absent SG_B pointer.
    with pytest.raises(ValueError, match="cannot resolve explicit alternate 'B' in source atom_site"):
        read_structure(view, altloc=altloc, components={"CYS": get_component("CYS")} if explicit else None)


@pytest.mark.parametrize("observed_sulfur", [False, True], ids=["canonical_unobserved", "observed_alternate"])
@pytest.mark.parametrize("altloc", ["first", "highest_occupancy"])
@pytest.mark.parametrize("crystal", [False, True])
def test_alternate_controls_disulfide_reaction(alternate_disulfide_source, observed_sulfur, altloc, crystal):
    path, cif = alternate_disulfide_source
    if observed_sulfur:
        append_row(
            cif.block["atom_site"],
            {
                "id": "4",
                "label_asym_id": "B",
                "auth_asym_id": "Q",
                "label_entity_id": "2",
                "label_seq_id": "1",
                "auth_seq_id": "7",
                "label_alt_id": "B",
                "occupancy": "0.1",
            },
        )
    else:
        cif.block["struct_conn"]["pdbx_ptnr2_label_alt_id"] = pdbx.CIFColumn(["?"])
    if crystal:
        add_crystal_translation(cif)
    cif.write(str(path))
    reacts = not observed_sulfur or altloc == "first"
    for chains in (None, frozenset({"A"})):
        structure = read_structure(read_source(path).view(asym_ids=chains), altloc=altloc)
        system = structure.system
        sites = [system.atom_site(i) for i in range(system.atom_symbol.numel())]
        assert ("HG" in {site.atom_id for site in sites if site.instance_id == "A"}) is not reacts
        connected = (system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
        assert len(connected) == int(reacts and chains is None)
        if connected:
            edge = system.edge_index[:, connected[0]].tolist()
            assert {(sites[i].instance_id, sites[i].atom_id) for i in edge} == {
                ("A#2_555" if crystal else "A", "SG"),
                ("B", "SG"),
            }
            first = next(i for i in edge if sites[i].instance_id != "B")
            second = next(i for i in edge if sites[i].instance_id == "B")
            assert structure.coordinates[first].tolist() == [5.0 if crystal else 0.0, 0.0, 0.0]
            assert structure.coordinate_mask[second].item() == observed_sulfur
        if chains is None:
            assert ("HG" in {site.atom_id for site in sites if site.instance_id == "B"}) is not reacts
        assert structure.coordinate_mask.sum().item() == (
            2 + int(observed_sulfur and altloc == "first") + int(crystal and reacts) if chains is None else 1
        )


@pytest.mark.parametrize("cysteine_observed", [False, True])
@pytest.mark.parametrize("altloc", ["first", "highest_occupancy"])
@pytest.mark.parametrize("crystal", [False, True])
def test_component_selection_precedes_reaction(alternate_disulfide_source, cysteine_observed, altloc, crystal):
    path, cif = alternate_disulfide_source
    cif.block["entity_poly_seq"]["hetero"] = pdbx.CIFColumn(["y"])
    append_row(cif.block["entity_poly_seq"], {"mon_id": "SER"})
    append_row(cif.block["pdbx_poly_seq_scheme"], {"mon_id": "SER"})
    for field in ("label_comp_id", "auth_comp_id"):
        cif.block["atom_site"][field] = pdbx.CIFColumn(["CYS", "CYS", "SER"])
    if not cysteine_observed:
        cif.block["atom_site"] = pdbx.CIFCategory(
            {key: value.as_array(str)[[0, 2]].tolist() for key, value in cif.block["atom_site"].items()}
        )
    cif.block["struct_conn"]["pdbx_ptnr2_label_alt_id"] = pdbx.CIFColumn(["?"])
    if crystal:
        add_crystal_translation(cif)
    cif.write(str(path))
    reacts = cysteine_observed and altloc == "first"
    for chains in (None, frozenset({"A"})):
        structure = read_structure(read_source(path).view(asym_ids=chains), altloc=altloc)
        system = structure.system
        sites = [system.atom_site(i) for i in range(system.atom_symbol.numel())]
        assert ("HG" in {site.atom_id for site in sites if site.instance_id == "A"}) is not reacts
        connected = (system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
        assert len(connected) == int(reacts and chains is None)
        if chains is None:
            partner = {site.atom_id for site in sites if site.instance_id == "B"}
            assert ("SG" in partner) == reacts
            assert ("OG" in partner) is not reacts
            assert ("HG" in partner) is not reacts
        if connected:
            edge = system.edge_index[:, connected[0]].tolist()
            assert {(sites[i].instance_id, sites[i].atom_id) for i in edge} == {
                ("A#2_555" if crystal else "A", "SG"),
                ("B", "SG"),
            }
            sulfur = next(i for i in edge if sites[i].instance_id == "B")
            assert not structure.coordinate_mask[sulfur]
        assert structure.coordinate_mask.sum().item() == (2 + int(crystal and reacts) if chains is None else 1)


def add_crystal_translation(cif):
    cif.block["struct_conn"]["ptnr1_symmetry"] = pdbx.CIFColumn(["2_555"])
    cif.block["cell"] = pdbx.CIFCategory(
        {
            **{f"length_{axis}": [10.0] for axis in "abc"},
            **{f"angle_{angle}": [90.0] for angle in ("alpha", "beta", "gamma")},
        }
    )
    fractional = {f"fract_transf_matrix[{i}][{j}]": [0.1 if i == j else 0.0] for i in range(1, 4) for j in range(1, 4)}
    fractional.update({f"fract_transf_vector[{i}]": [0.0] for i in range(1, 4)})
    cif.block["atom_sites"] = pdbx.CIFCategory(fractional)
    cif.block["space_group_symop"] = pdbx.CIFCategory({"id": ["1", "2"], "operation_xyz": ["x,y,z", "x+1/2,y,z"]})


@pytest.mark.parametrize("observed", [False, True])
def test_crystal_resolves_author_occurrence(unobserved_source, observed):
    path, cif = unobserved_source
    add_crystal_translation(cif)
    cif.block["struct_conn"]["ptnr2_auth_atom_id"] = pdbx.CIFColumn(["?"])
    if observed:
        append_row(
            cif.block["atom_site"],
            {
                "id": "2",
                "type_symbol": "C",
                "label_atom_id": "CA",
                "auth_atom_id": "CA",
                "label_comp_id": "CYS",
                "auth_comp_id": "CYS",
                "label_asym_id": "B",
                "auth_asym_id": "Q",
                "label_entity_id": "2",
                "label_seq_id": "1",
                "auth_seq_id": "7",
            },
        )
    cif.write(str(path))
    native = read_source(path)
    full = read_structure(native)
    assert full.system.instance_symbol.numel() == 3
    connected = (full.system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
    assert len(connected) == 1
    edge = full.system.edge_index[:, connected[0]].tolist()
    assert {full.system.atom_site(atom).atom_id for atom in edge} == {"ZN", "SG"}
    zinc = next(atom for atom in edge if full.system.atom_site(atom).atom_id == "ZN")
    assert full.coordinates[zinc].tolist() == [5.0, 0.0, 0.0]
    view = native.view(asym_ids=frozenset({"A"}))
    lazy = read_structure(view, component_resolver=get_component)
    explicit = read_structure(view, components={"CYS": get_component("CYS")})
    torch.testing.assert_close(lazy.state_dict(), explicit.state_dict(), rtol=0, atol=0, equal_nan=True)
    assert lazy.system.instance_symbol.numel() == 1
    assert lazy.coordinates.tolist() == [[0.0, 0.0, 0.0]]
    assert lazy.system.edge_index.shape[1] == 0
