from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ofoldx.data.components import get_component
from ofoldx.data.molecular import BOND_ROLE_ASSEMBLY, AtomSite, ChemicalAtom, ChemicalBond, ChemicalComponent
from ofoldx.io import read_source, read_structure
from ofoldx.io.pdb import read_pdb_bonds
from ofoldx.io.records import PdbSiteKey


@pytest.fixture
def connected_pdb(tmp_path):
    path = tmp_path / "selection.pdb"
    text = (
        "REMARK 350 BIOMOLECULE: 1\n"
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
        "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
        "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000\n"
        "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
        "SSBOND   1 CYS A    1    CYS B    2                          1555   1555  2.03\n"
        "MODEL        1\n"
        "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  0.75 15.25           S\n"
        "TER       2      CYS A   1\n"
        "ATOM      3  SG  CYS B   2       2.030   0.000   0.000  1.00 10.00           S\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM     11  SG  CYS A   1      10.000   0.000   0.000  0.75 15.25           S\n"
        "TER      12      CYS A   1\n"
        "ATOM     13  SG  CYS B   2      12.030   0.000   0.000  1.00 10.00           S\n"
        "ENDMDL\n"
        "CONECT    1    3\n"
        "CONECT    3    1\n"
        "END\n"
    )
    path.write_text(text, encoding="ascii")
    return path


@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("model", [1, 2])
@pytest.mark.parametrize("hydrogens", [False, True])
def test_selection_preserves_disulfide_chemistry(connected_pdb, assembly, model, hydrogens):
    native = read_source(connected_pdb)
    full = read_structure(native.view(model_index=model), include_hydrogens=hydrogens)
    selected = read_structure(
        native.view(model_index=model, asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None),
        include_hydrogens=hydrogens,
    )
    copies = 2 if assembly else 1
    assert selected.system.instance_symbol.numel() == copies
    assert not (selected.system.bond_role == BOND_ROLE_ASSEMBLY).any()
    stop = int(full.system.component_atom_ptr[1])
    names = [full.system.atom_site(i).atom_id for i in range(stop)]
    assert "HG" not in names
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == names * copies
    for name in (
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
        torch.testing.assert_close(getattr(selected.system, name), getattr(full.system, name)[:stop].repeat(copies))
    for name in (
        "coordinate_mask",
        "occupancy",
        "occupancy_mask",
        "b_factor",
        "b_factor_mask",
        "source_formal_charge_mask",
    ):
        torch.testing.assert_close(getattr(selected, name), getattr(full, name)[:stop].repeat(copies))
    origin = 10.0 * (model - 1)
    assert selected.coordinates[selected.coordinate_mask, 0].tolist() == (
        [origin, origin + 10.0] if assembly else [origin]
    )
    assert set(native.view(model_index=model).rows.chain_id) == {"A", "B"}


@pytest.mark.parametrize("assembly", [False, True])
def test_link_selection_keeps_source_state(connected_pdb, assembly):
    lines = connected_pdb.read_text().splitlines()
    lines = [
        "LINK         SG  CYS A   1                 SG  CYS B   2" if line.startswith("SSBOND") else line
        for line in lines
    ]
    connected_pdb.write_text("\n".join(lines) + "\n", encoding="ascii")
    native = read_source(connected_pdb)
    full = read_structure(native.view())
    selected = read_structure(native.view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None))
    stop = int(full.system.component_atom_ptr[1])
    names = [full.system.atom_site(i).atom_id for i in range(stop)]
    assert "HG" in names
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == names * (
        2 if assembly else 1
    )
    assert not (selected.system.bond_role == BOND_ROLE_ASSEMBLY).any()


@pytest.mark.parametrize("record", ["unknown_serial", "missing_model", "missing_declaration"])
def test_selection_rejects_invalid_connections(connected_pdb, record):
    text = connected_pdb.read_text()
    if record == "unknown_serial":
        text = text.replace("CONECT    1    3", "CONECT    1 99999")
        error = "unknown atom serials"
    elif record == "missing_model":
        text = "\n".join(line for line in text.splitlines() if not line.startswith("ATOM     13")) + "\n"
        error = "absent from the selected model"
    else:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("SSBOND")) + "\n"
        error = "a LINK or SSBOND record is required"
    connected_pdb.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match=error):
        read_structure(read_source(connected_pdb).view(model_index=2, asym_ids=frozenset({"A"})))


@pytest.mark.parametrize("assembly", [False, True])
def test_unrelated_chemistry_is_not_requested(connected_pdb, assembly):
    text = connected_pdb.read_text().replace(
        "ENDMDL\n", "HETATM   20  X1  EXC C   3      20.000   0.000   0.000  1.00 10.00           C\nENDMDL\n"
    )
    connected_pdb.write_text(text, encoding="ascii")

    def resolver(identifier):
        if identifier == "EXC":
            raise RuntimeError("unrelated component dictionary is unavailable")
        raise KeyError(identifier)

    selected = read_structure(
        read_source(connected_pdb).view(asym_ids=frozenset({"A"}), assembly_id="1" if assembly else None),
        component_resolver=resolver,
    )
    assert selected.system.instance_symbol.numel() == (2 if assembly else 1)
    assert "HG" not in {selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())}


@pytest.mark.parametrize("altloc,reacted", [("first", True), ("highest_occupancy", False)])
@pytest.mark.parametrize("crystal", [False, True])
def test_boundary_respects_selected_conformation(connected_pdb, altloc, reacted, crystal):
    lines = []
    for line in connected_pdb.read_text().splitlines():
        if crystal and line.startswith("CONECT"):
            continue
        if crystal and line.startswith("SSBOND"):
            line = line[:59] + f"{2555:>6}" + line[65:]
        if line.startswith("ATOM") and line[21:22] == "B":
            line = line[:16] + "B" + line[17:54] + "  0.10" + line[60:]
        lines.append(line)
        if line.startswith("ATOM") and line[21:22] == "B":
            lines.append("ATOM     20  OG ASER B   2       3.000   0.000   0.000  0.90 10.00           O")
    if crystal:
        lines[:0] = [
            "REMARK 290   SMTRY1   2  1.000000  0.000000  0.000000       10.00000",
            "REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000",
            "REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000",
        ]
    connected_pdb.write_text("\n".join(lines) + "\n", encoding="ascii")
    native = read_source(connected_pdb)
    full = read_structure(native.view(), altloc=altloc)
    selected = read_structure(native.view(asym_ids=frozenset({"A"})), altloc=altloc)
    stop = int(full.system.component_atom_ptr[1])
    expected = [full.system.atom_site(i).atom_id for i in range(stop)]
    assert ("HG" not in expected) == reacted
    copies = 2 if crystal and reacted else 1
    assert selected.system.instance_symbol.numel() == copies
    assert [
        selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())
    ] == expected * copies


def test_declared_unobserved_partner(connected_pdb):
    lines = [
        line
        for line in connected_pdb.read_text().splitlines()
        if not line.startswith("CONECT") and not (line.startswith("ATOM") and line[21:22] == "B")
    ]
    connected_pdb.write_text(
        "SEQRES   1 B    1  CYS\nREMARK 465     CYS B     2\n" + "\n".join(lines) + "\n", encoding="ascii"
    )
    native = read_source(connected_pdb)
    full = read_structure(native.view())
    selected = read_structure(native.view(asym_ids=frozenset({"A"})))
    stop = int(full.system.component_atom_ptr[1])
    expected = [full.system.atom_site(i).atom_id for i in range(stop)]
    assert "HG" not in expected
    assert [selected.system.atom_site(i).atom_id for i in range(selected.system.atom_symbol.numel())] == expected


@pytest.mark.parametrize("chains", [None, frozenset({"A"}), frozenset({"B"})])
@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("symmetry", [1555, 2565])
def test_crystal_selection_preserves_reaction(connected_pdb, chains, assembly, symmetry):
    lines = []
    for line in connected_pdb.read_text().splitlines():
        if line.startswith("CONECT"):
            continue
        if line.startswith("SSBOND"):
            line = line[:59] + f"{symmetry:>6}" + line[65:]
        line = line.replace("FOLLOWING TO CHAINS: A", "FOLLOWING TO CHAINS: A, B")
        line = line.replace("10.00000", " 6.00000") if "BIOMT1   2" in line else line
        if "BIOMT2   2" in line:
            line = "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000       20.00000"
        lines.append(line)
    connected_pdb.write_text(
        "SCALE1      0.100000  0.020000  0.000000        0.00000\n"
        "SCALE2      0.000000  0.050000  0.000000        0.00000\n"
        "SCALE3      0.000000  0.000000  0.033333        0.00000\n"
        "REMARK 290   SMTRY1   2  1.000000  0.000000  0.000000       10.00000\n"
        "REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000\n"
        "REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000\n" + "\n".join(lines) + "\n",
        encoding="ascii",
    )
    structure = read_structure(read_source(connected_pdb).view(asym_ids=chains, assembly_id="1" if assembly else None))
    retained = {"A", "B"} if chains is None else set(chains)
    expected_instances = (
        {f"{chain}#{copy}" for chain in retained for copy in ("0", "1")}
        if assembly
        else retained | ({"A#2565"} if symmetry != 1555 and "A" in retained else set())
    )
    sites = [structure.system.atom_site(i) for i in range(structure.system.atom_symbol.numel())]
    assert {site.instance_id for site in sites} == expected_instances
    for instance in expected_instances:
        names = {site.atom_id for site in sites if site.instance_id == instance}
        assert "SG" in names
        assert "HG" not in names
        assert {"H", "HA", "HB2", "HB3"} <= names
    edges = torch.nonzero(structure.system.bond_role == BOND_ROLE_ASSEMBLY).flatten().tolist()
    expected_bonds = (2 if assembly and symmetry == 1555 else 1) if len(retained) == 2 else 0
    assert len(edges) == expected_bonds
    for edge in edges:
        a, b = (sites[i] for i in structure.system.edge_index[:, edge].tolist())
        assert a.atom_id == b.atom_id == "SG"
        if symmetry != 1555:
            assert {a.instance_id, b.instance_id} == ({"A#1", "B#0"} if assembly else {"A#2565", "B"})
    if symmetry != 1555 and "A" in retained:
        copy = "A#1" if assembly else "A#2565"
        atom = next(i for i, site in enumerate(sites) if site.instance_id == copy and site.atom_id == "SG")
        assert structure.coordinate_mask[atom]
        assert structure.coordinates[atom].tolist() == [6.0, 20.0, 0.0]


@pytest.fixture
def partial_link_pdb(tmp_path):
    path = tmp_path / "partial-link.pdb"
    path.write_text(
        "REMARK 350 BIOMOLECULE: 1\n"
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B\n"
        "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
        "SEQRES   1 A    1  CYS\n"
        "SEQRES   1 B    1  ALA\n"
        "LINK         SG  CYS A   1                 N   ALA B   2\n"
        "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00 10.00           S\n"
        "TER       2      CYS A   1\n"
        "ATOM      3  CA  ALA B   2       4.000   0.000   0.000  1.00 10.00           C\n"
        "TER       4      ALA B   2\n"
        "HETATM    5  X1  EXC C   3      20.000   0.000   0.000  1.00 10.00           C\n"
        "END\n",
        encoding="ascii",
    )
    return path


@pytest.mark.parametrize("chains", [frozenset({"A", "B"}), frozenset({"A"}), frozenset({"B"})])
@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("partner", ["ALA", "XAA"])
def test_partial_partner_chemistry_equivalence(partial_link_pdb, chains, assembly, partner):
    partial_link_pdb.write_text(partial_link_pdb.read_text().replace("ALA", partner), encoding="ascii")
    definitions = {"CYS": get_component("CYS"), partner: replace(get_component("ALA"), component_id=partner)}
    requested = set()

    def resolver(identifier):
        requested.add(identifier)
        if identifier == "EXC":
            raise RuntimeError("unrelated chemistry must not be requested")
        return definitions[identifier]

    view = read_source(partial_link_pdb).view(asym_ids=chains, assembly_id="1" if assembly else None)
    explicit = read_structure(view, components=definitions)
    lazy = read_structure(view, component_resolver=resolver)
    assert requested == {"CYS", partner}
    sites = [lazy.system.atom_site(i) for i in range(lazy.system.atom_symbol.numel())]
    assert sites == [explicit.system.atom_site(i) for i in range(explicit.system.atom_symbol.numel())]
    assert lazy.system.instance_symbol.numel() == len(chains)
    assert int((lazy.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == (1 if len(chains) == 2 else 0)
    if "A" in chains:
        assert any(site.atom_id == "HG" for site in sites)
    if "B" in chains:
        atom = next(i for i, site in enumerate(sites) if site.instance_id.startswith("B") and site.atom_id == "N")
        assert not lazy.coordinate_mask[atom]
    for name in ("coordinates", "coordinate_mask", "occupancy", "b_factor", "source_formal_charge_mask"):
        torch.testing.assert_close(getattr(lazy, name), getattr(explicit, name), rtol=0, atol=0)
    for name in ("atom_symbol", "edge_index", "bond_order", "bond_kind", "bond_role", "formal_charge"):
        torch.testing.assert_close(getattr(lazy.system, name), getattr(explicit.system, name), rtol=0, atol=0)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("invalid", ["occurrence", "atom"])
def test_partial_partner_rejects_unknown(partial_link_pdb, explicit, invalid):
    text = partial_link_pdb.read_text()
    text = text.replace("N   ALA B   2", "N   ALA Z   2" if invalid == "occurrence" else "XX  ALA B   2")
    partial_link_pdb.write_text(text, encoding="ascii")
    definitions = {name: get_component(name) for name in ("CYS", "ALA")}
    with pytest.raises(ValueError, match="LINK endpoint identity is missing"):
        read_structure(
            read_source(partial_link_pdb).view(asym_ids=frozenset({"A"})),
            components=definitions if explicit else None,
            component_resolver=None if explicit else definitions.__getitem__,
        )


def test_partner_resolver_failure_propagates(partial_link_pdb):
    def resolver(identifier):
        if identifier == "ALA":
            raise RuntimeError("partner chemistry unavailable")
        return get_component(identifier)

    with pytest.raises(RuntimeError, match="partner chemistry unavailable"):
        read_structure(read_source(partial_link_pdb).view(asym_ids=frozenset({"A"})), component_resolver=resolver)


@pytest.fixture
def nonpolymer_link_pdb(tmp_path):
    path = tmp_path / "nonpolymer-link.pdb"
    path.write_text(
        "REMARK 350 BIOMOLECULE: 1\n"
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B\n"
        "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
        "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000\n"
        "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
        "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
        "SEQRES   1 A    1  CYS\n"
        "LINK         SG  CYS A   1                 C1  LIG B   2\n"
        "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00 10.00           S\n"
        "TER       2      CYS A   1\n"
        "HETATM    3  C2  LIG B   2       4.000   0.000   0.000  1.00 10.00           C\n"
        "HETATM    4  X1  EXC C   3      20.000   0.000   0.000  1.00 10.00           C\n"
        "END\n",
        encoding="ascii",
    )
    definitions = {
        "CYS": get_component("CYS"),
        "LIG": ChemicalComponent(
            "LIG",
            (ChemicalAtom("C1", "C"), ChemicalAtom("C2", "C")),
            frozenset({ChemicalBond("C1", "C2")}),
        ),
        "EXC": ChemicalComponent("EXC", (ChemicalAtom("X1", "C"),), frozenset()),
    }
    return path, definitions


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize(
    "chains,assembly",
    [
        (None, False),
        (frozenset({"A", "B"}), False),
        (frozenset({"A"}), False),
        (frozenset({"B"}), False),
        (frozenset({"C"}), False),
        (None, True),
        (frozenset({"A"}), True),
        (frozenset({"B"}), True),
    ],
)
def test_nonpolymer_link_preserves_address(nonpolymer_link_pdb, observed, chains, assembly):
    path, definitions = nonpolymer_link_pdb
    if observed:
        path.write_text(
            path.read_text().replace(
                "END\n", "HETATM    5  C1  LIG B   2       2.000   0.000   0.000  1.00 10.00           C\nEND\n"
            ),
            encoding="ascii",
        )
    native = read_source(path)
    view = native.view(asym_ids=chains, assembly_id="1" if assembly else None)
    retained = {"A", "B", "C"} if chains is None else set(chains)
    if assembly:
        retained &= {"A", "B"}
    requested = set()

    def resolver(identifier):
        requested.add(identifier)
        if identifier == "EXC" and "C" not in retained:
            raise RuntimeError("unrelated chemistry must not be requested")
        return definitions[identifier]

    lazy = read_structure(view, component_resolver=resolver)
    explicit = read_structure(view, components=definitions)
    torch.testing.assert_close(lazy.state_dict(), explicit.state_dict(), rtol=0, atol=0)
    assert requested == {"CYS", "LIG"} | ({"EXC"} if "C" in retained else set())
    assert set(native.view().rows.chain_id) == {"A", "B", "C"}
    placements = {"#0": 0.0, "#1": 10.0} if assembly else {"": 0.0}
    molecule_suffixes = {"A": "", "B": ":2:LIG", "C": ":3:EXC"}
    expected_instances = {f"{chain}{suffix}{molecule_suffixes[chain]}" for chain in retained for suffix in placements}
    sites = [lazy.system.atom_site(i) for i in range(lazy.system.atom_symbol.numel())]
    assert {site.instance_id for site in sites} == expected_instances
    assert lazy.system.instance_symbol.numel() == len(expected_instances)
    edges = (lazy.system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
    expected_edges = (
        {frozenset({(f"A{suffix}", "SG"), (f"B{suffix}:2:LIG", "C1")}) for suffix in placements}
        if {"A", "B"} <= retained
        else set()
    )
    actual_edges = [
        frozenset((sites[atom].instance_id, sites[atom].atom_id) for atom in lazy.system.edge_index[:, edge].tolist())
        for edge in edges
    ]
    assert len(actual_edges) == len(expected_edges)
    assert set(actual_edges) == expected_edges
    for suffix, shift in placements.items():
        if "A" in retained:
            atoms = {site.atom_id: i for i, site in enumerate(sites) if site.instance_id == f"A{suffix}"}
            assert "HG" in atoms
            assert lazy.coordinate_mask[atoms["SG"]]
            assert lazy.coordinates[atoms["SG"]].tolist() == [shift, 0.0, 0.0]
        if "B" in retained:
            atoms = {site.atom_id: i for i, site in enumerate(sites) if site.instance_id == f"B{suffix}:2:LIG"}
            assert set(atoms) == {"C1", "C2"}
            assert bool(lazy.coordinate_mask[atoms["C1"]]) == observed
            assert lazy.coordinate_mask[atoms["C2"]]
            assert lazy.coordinates[atoms["C2"]].tolist() == [4.0 + shift, 0.0, 0.0]
            if observed:
                assert lazy.coordinates[atoms["C1"]].tolist() == [2.0 + shift, 0.0, 0.0]
            else:
                assert not lazy.occupancy_mask[atoms["C1"]]
                assert not lazy.b_factor_mask[atoms["C1"]]


def test_link_distinguishes_ligand_occurrences(nonpolymer_link_pdb):
    path, definitions = nonpolymer_link_pdb
    path.write_text(
        path.read_text().replace(
            "END\n", "HETATM    5  C2  LIG B   2A      8.000   0.000   0.000  1.00 10.00           C\nEND\n"
        ),
        encoding="ascii",
    )
    structure = read_structure(read_source(path).view(asym_ids=frozenset({"A", "B"})), components=definitions)
    sites = [structure.system.atom_site(i) for i in range(structure.system.atom_symbol.numel())]
    assert {site.instance_id for site in sites} == {"A", "B:2:LIG", "B:2A:LIG"}
    edges = (structure.system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
    assert len(edges) == 1
    assert {(sites[i].instance_id, sites[i].atom_id) for i in structure.system.edge_index[:, edges[0]].tolist()} == {
        ("A", "SG"),
        ("B:2:LIG", "C1"),
    }
    for instance, x in (("B:2:LIG", 4.0), ("B:2A:LIG", 8.0)):
        atoms = {site.atom_id: i for i, site in enumerate(sites) if site.instance_id == instance}
        assert not structure.coordinate_mask[atoms["C1"]]
        assert structure.coordinates[atoms["C2"]].tolist() == [x, 0.0, 0.0]


@pytest.mark.parametrize("chains", [None, frozenset({"A"}), frozenset({"B"}), frozenset({"C"})])
@pytest.mark.parametrize("explicit", [False, True])
def test_nonpolymer_link_rejects_unknown(nonpolymer_link_pdb, chains, explicit):
    path, definitions = nonpolymer_link_pdb
    path.write_text(path.read_text().replace("C1  LIG B", "XX  LIG B"), encoding="ascii")
    with pytest.raises(ValueError, match="LINK endpoint identity is missing"):
        read_structure(
            read_source(path).view(asym_ids=chains),
            components=definitions if explicit else None,
            component_resolver=definitions.__getitem__,
        )


@pytest.mark.parametrize("chains", [None, frozenset({"A"}), frozenset({"B"}), frozenset({"C"})])
def test_link_rejects_inferred_altloc(nonpolymer_link_pdb, chains):
    path, definitions = nonpolymer_link_pdb
    text = path.read_text().replace("LINK         SG  CYS A", "LINK         SG BCYS A")
    text = text.replace(
        "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00 10.00           S",
        "ATOM      1  CA BCYS A   1       0.000   0.000   0.000  1.00 10.00           C",
    )
    path.write_text(text, encoding="ascii")
    view = read_source(path).view(asym_ids=chains)
    for components in (definitions, None):
        with pytest.raises(ValueError, match="LINK endpoint alternate location is missing"):
            read_structure(view, components=components, component_resolver=definitions.__getitem__)


@pytest.fixture
def alternate_link_pdb(nonpolymer_link_pdb):
    path, definitions = nonpolymer_link_pdb
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("LINK"):
            line = line[:46] + "B" + line[47:]
        if line.startswith("HETATM    3"):
            lines.extend(
                (
                    "HETATM    3  C2 BLIG B   2       4.000   0.000   0.000  0.10 10.00           C",
                    "HETATM    5  C2 ALIG B   2       5.000   0.000   0.000  0.90 10.00           C",
                )
            )
        else:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path, definitions


@pytest.mark.parametrize("atom_altloc", [None, "A", "B"])
@pytest.mark.parametrize("altloc", ["first", "highest_occupancy"])
@pytest.mark.parametrize("chains", [None, frozenset({"A"}), frozenset({"B"}), frozenset({"C"})])
def test_link_requires_atom_altloc(alternate_link_pdb, atom_altloc, altloc, chains):
    path, definitions = alternate_link_pdb
    if atom_altloc is not None:
        path.write_text(
            path.read_text().replace(
                "END\n",
                f"HETATM    6  C1 {atom_altloc}LIG B   2       2.000   0.000   0.000  0.10 10.00           C\nEND\n",
            ),
            encoding="ascii",
        )
    view = read_source(path).view(asym_ids=chains)
    for components in (definitions, None):
        if atom_altloc != "B":
            with pytest.raises(ValueError, match="LINK endpoint alternate location is missing"):
                read_structure(view, altloc=altloc, components=components, component_resolver=definitions.__getitem__)
        else:
            structure = read_structure(
                view, altloc=altloc, components=components, component_resolver=definitions.__getitem__
            )
            edges = (structure.system.bond_role == BOND_ROLE_ASSEMBLY).nonzero().flatten().tolist()
            assert len(edges) == int(chains is None and altloc == "first")
            if edges:
                sites = [structure.system.atom_site(i) for i in structure.system.edge_index[:, edges[0]].tolist()]
                assert {(site.instance_id, site.atom_id) for site in sites} == {("A", "SG"), ("B:2:LIG", "C1")}


@pytest.mark.parametrize("chains", [frozenset({"A"}), frozenset({"A", "B"})])
def test_bond_reader_validates_atom_altloc(alternate_link_pdb, chains):
    path, definitions = alternate_link_pdb
    source = read_source(path)
    model_rows = source.view().rows
    source_rows = source.view(asym_ids=chains).rows
    selected_rows = source_rows[source_rows.altloc_id != "A"]
    sites = {int(row.row_index): AtomSite(str(row.chain_id), 0, str(row.atom_name)) for row in selected_rows}
    canonical_sites = {PdbSiteKey(None, "B", 2, None, "LIG", "C1"): AtomSite("B", 0, "C1")} if "B" in chains else {}
    with pytest.raises(ValueError, match="LINK endpoint alternate location is missing"):
        read_pdb_bonds(
            source.lines,
            source_rows,
            selected_rows,
            sites,
            canonical_sites,
            frozenset(),
            definitions,
            include_water=True,
            assembly_placements=None,
            model_rows=model_rows,
            selected_model_rows=model_rows[model_rows.altloc_id != "A"],
            missing_residues=(),
        )
