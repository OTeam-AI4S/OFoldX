r"""Chemical projections of the canonical molecular graph."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_KIND_DATIVE,
    BOND_ROLE_COMPONENT,
    MolecularSystem,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


def to_rdkit_component(system: MolecularSystem, component_index: int) -> Mol:
    r"""Project one complete canonical component occurrence into an RDKit molecule."""

    component_id = system.symbols[int(system.component_symbol[component_index])]
    if not bool(system.component_topology_complete[component_index]):
        raise ValueError(f"component {component_id!r} has unknown connectivity")
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    edge_mask = (
        (system.edge_index[0] >= atom_start)
        & (system.edge_index[1] < atom_stop)
        & (system.bond_role == BOND_ROLE_COMPONENT)
    )
    edge_rows = edge_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    return _to_rdkit_graph(
        system,
        range(atom_start, atom_stop),
        edge_rows,
        f"component {component_id!r}",
    )


def to_rdkit_system(system: MolecularSystem) -> Mol:
    r"""Project and chemically validate every atom and bond in a molecular system."""

    incomplete = (~system.component_topology_complete).nonzero(as_tuple=False).flatten()
    if incomplete.numel():
        component_index = int(incomplete[0])
        component_id = system.symbols[int(system.component_symbol[component_index])]
        raise ValueError(f"component {component_id!r} has unknown connectivity")
    return _to_rdkit_graph(
        system,
        range(int(system.atom_symbol.numel())),
        range(int(system.edge_index.shape[1])),
        "molecular system",
    )


def _to_rdkit_graph(
    system: MolecularSystem,
    atom_indices: Sequence[int],
    edge_rows: Sequence[int],
    description: str,
) -> Mol:
    r"""Build one RDKit graph from selected canonical tensor rows."""

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    editable = Chem.RWMol()
    atom_stereo: dict[int, str] = {}
    local_atom = {atom_index: local_index for local_index, atom_index in enumerate(atom_indices)}
    for atom_index in atom_indices:
        value = Chem.Atom(int(system.atomic_number[atom_index]))
        if bool(system.isotope_known[atom_index]):
            value.SetIsotope(int(system.isotope[atom_index]))
        if bool(system.formal_charge_known[atom_index]):
            value.SetFormalCharge(int(system.formal_charge[atom_index]))
        if bool(system.radical_electrons_known[atom_index]):
            value.SetNumRadicalElectrons(int(system.radical_electrons[atom_index]))
        if bool(system.explicit_hydrogens_known[atom_index]):
            value.SetNumExplicitHs(int(system.explicit_hydrogens[atom_index]))
        if bool(system.implicit_hydrogens_allowed_known[atom_index]):
            value.SetNoImplicit(not bool(system.implicit_hydrogens_allowed[atom_index]))
        if bool(system.atom_aromatic_known[atom_index]):
            value.SetIsAromatic(bool(system.atom_aromatic[atom_index]))
        local_index = editable.AddAtom(value)
        stereo_index = int(system.atom_stereo[atom_index])
        if stereo_index >= 0:
            atom_stereo[local_index] = system.symbols[stereo_index].upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for edge_index in edge_rows:
        first = local_atom[int(system.edge_index[0, edge_index])]
        second = local_atom[int(system.edge_index[1, edge_index])]
        if int(system.bond_kind[edge_index]) == BOND_KIND_DATIVE:
            direction = int(system.bond_direction[edge_index])
            if direction == BOND_DIRECTION_ENDPOINT_0_DONOR:
                donor, acceptor = first, second
            elif direction == BOND_DIRECTION_ENDPOINT_1_DONOR:
                donor, acceptor = second, first
            else:
                raise ValueError(f"{description} has a dative bond without a donor direction")
            editable.AddBond(donor, acceptor, Chem.BondType.DATIVE)
        else:
            if not bool(system.bond_order_known[edge_index]):
                raise ValueError(f"{description} has an unknown bond order")
            order = float(system.bond_order[edge_index])
            try:
                bond_type = bond_types[order]
            except KeyError:
                raise ValueError(f"{description} has unsupported bond order {order}") from None
            editable.AddBond(first, second, bond_type)
        rdkit_bond = editable.GetBondWithIdx(editable.GetNumBonds() - 1)
        if bool(system.bond_aromatic_known[edge_index]):
            rdkit_bond.SetIsAromatic(bool(system.bond_aromatic[edge_index]))
        stereo_index = int(system.bond_stereo[edge_index])
        if stereo_index >= 0:
            stereo_atoms_tensor = system.bond_stereo_index[:, edge_index]
            stereo_atoms = (
                None
                if bool((stereo_atoms_tensor < 0).any())
                else (
                    local_atom[int(stereo_atoms_tensor[0])],
                    local_atom[int(stereo_atoms_tensor[1])],
                )
            )
            bond_stereo.append((rdkit_bond.GetIdx(), system.symbols[stereo_index], stereo_atoms))

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(len(atom_indices))
    reference_coordinates = system.reference_coordinates.detach().cpu()
    reference_mask = system.reference_coordinate_mask.detach().cpu()
    for index, atom_index in enumerate(atom_indices):
        x, y, z = reference_coordinates[atom_index].tolist() if bool(reference_mask[atom_index]) else (0.0, 0.0, 0.0)
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    molecule.AddConformer(conformer, assignId=True)

    for index in atom_stereo:
        molecule.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    for index, desired in atom_stereo.items():
        atom = molecule.GetAtomWithIdx(index)
        assigned = atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else ""
        if assigned != desired:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    stereo_types = {
        "E": Chem.BondStereo.STEREOE,
        "TRANS": Chem.BondStereo.STEREOE,
        "Z": Chem.BondStereo.STEREOZ,
        "CIS": Chem.BondStereo.STEREOZ,
    }
    for bond_index, stereo, stereo_atoms in bond_stereo:
        bond = molecule.GetBondWithIdx(bond_index)
        if stereo_atoms is None:
            first = bond.GetBeginAtomIdx()
            second = bond.GetEndAtomIdx()
            first_neighbors = [
                atom for atom in molecule.GetAtomWithIdx(first).GetNeighbors() if atom.GetIdx() != second
            ]
            second_neighbors = [
                atom for atom in molecule.GetAtomWithIdx(second).GetNeighbors() if atom.GetIdx() != first
            ]
            stereo_atoms = (
                max(first_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
                max(second_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
            )
        bond.SetStereoAtoms(*stereo_atoms)
        try:
            stereo_type = stereo_types[stereo.upper()]
        except KeyError:
            raise ValueError(f"{description} has unsupported bond stereo {stereo!r}") from None
        bond.SetStereo(stereo_type)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    return molecule
