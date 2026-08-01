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
    from rdkit.Chem.rdchem import Atom, Bond, Mol


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


def to_rdkit_system(system: MolecularSystem, *, sanitize: bool = True) -> Mol:
    r"""Project every atom and bond, optionally applying RDKit chemical validation."""

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
        sanitize=sanitize,
    )


def validate_rdkit_system(system: MolecularSystem) -> None:
    r"""Validate complete connected molecules without joining disconnected graphs.

    Every edge, including inter-instance and dative bonds, participates in the
    partition. Entire connected molecules undergo the same RDKit projection,
    sanitization and stereo handling as :func:`to_rdkit_system`. This avoids
    superlinear ring processing over unrelated assembly copies. The canonical
    graph, atom order, coordinates and declared stereo are never modified.
    """

    import numpy as np
    from scipy.sparse import csr_array
    from scipy.sparse.csgraph import connected_components

    incomplete = (~system.component_topology_complete).nonzero(as_tuple=False).flatten()
    if incomplete.numel():
        component_id = system.symbols[int(system.component_symbol[int(incomplete[0])])]
        raise ValueError(f"component {component_id!r} has unknown connectivity")
    atom_count = int(system.atom_symbol.numel())
    edges = system.edge_index.detach().cpu().numpy()
    graph = csr_array((np.ones(edges.shape[1], dtype=np.bool_), (edges[0], edges[1])), shape=(atom_count, atom_count))
    count, labels = connected_components(graph, directed=False)
    atom_order = np.argsort(labels, kind="stable")
    atom_ptr = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
    edge_labels = labels[edges[0]]
    edge_order = np.argsort(edge_labels, kind="stable")
    edge_ptr = np.r_[0, np.cumsum(np.bincount(edge_labels, minlength=count))]
    for index in range(count):
        _to_rdkit_graph(
            system,
            atom_order[atom_ptr[index] : atom_ptr[index + 1]].tolist(),
            edge_order[edge_ptr[index] : edge_ptr[index + 1]].tolist(),
            "molecular system",
        )


def _to_rdkit_graph(
    system: MolecularSystem,
    atom_indices: Sequence[int],
    edge_rows: Sequence[int],
    description: str,
    *,
    sanitize: bool = True,
) -> Mol:
    r"""Build one RDKit graph from selected canonical tensor rows."""

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    # Materialize only selected graph columns once. Per-atom TensorClass lookups
    # and scalar tensor conversions otherwise dominate large CIF preparation.
    atom_selection = (
        slice(atom_indices.start, atom_indices.stop)
        if isinstance(atom_indices, range) and atom_indices.step == 1
        else list(atom_indices)
    )
    edge_selection = (
        slice(edge_rows.start, edge_rows.stop)
        if isinstance(edge_rows, range) and edge_rows.step == 1
        else list(edge_rows)
    )
    atom_columns = {
        name: getattr(system, name)[atom_selection].detach().cpu().tolist()
        for name in (
            "atomic_number",
            "isotope",
            "isotope_known",
            "formal_charge",
            "formal_charge_known",
            "radical_electrons",
            "radical_electrons_known",
            "explicit_hydrogens",
            "explicit_hydrogens_known",
            "implicit_hydrogens_allowed",
            "implicit_hydrogens_allowed_known",
            "atom_aromatic",
            "atom_aromatic_known",
            "atom_stereo",
        )
    }
    symbols = system.symbols
    editable = Chem.RWMol()
    atom_stereo: dict[int, str] = {}
    local_atom = {atom_index: local_index for local_index, atom_index in enumerate(atom_indices)}
    for row in range(len(atom_indices)):
        value = Chem.Atom(int(atom_columns["atomic_number"][row]))
        if bool(atom_columns["isotope_known"][row]):
            value.SetIsotope(int(atom_columns["isotope"][row]))
        if bool(atom_columns["formal_charge_known"][row]):
            value.SetFormalCharge(int(atom_columns["formal_charge"][row]))
        if bool(atom_columns["radical_electrons_known"][row]):
            value.SetNumRadicalElectrons(int(atom_columns["radical_electrons"][row]))
        if bool(atom_columns["explicit_hydrogens_known"][row]):
            value.SetNumExplicitHs(int(atom_columns["explicit_hydrogens"][row]))
        if bool(atom_columns["implicit_hydrogens_allowed_known"][row]):
            value.SetNoImplicit(not bool(atom_columns["implicit_hydrogens_allowed"][row]))
        if bool(atom_columns["atom_aromatic_known"][row]):
            value.SetIsAromatic(bool(atom_columns["atom_aromatic"][row]))
        local_index = editable.AddAtom(value)
        stereo_index = int(atom_columns["atom_stereo"][row])
        if stereo_index >= 0:
            atom_stereo[local_index] = symbols[stereo_index]

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    bond_columns = {
        name: getattr(system, name)[edge_selection].detach().cpu().tolist()
        for name in (
            "bond_kind",
            "bond_direction",
            "bond_order_known",
            "bond_order",
            "bond_aromatic_known",
            "bond_aromatic",
            "bond_stereo",
        )
    }
    edge_atoms = system.edge_index[:, edge_selection].detach().cpu().T.tolist()
    stereo_atoms_by_edge = system.bond_stereo_index[:, edge_selection].detach().cpu().T.tolist()
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for row, (atom_first, atom_second) in enumerate(edge_atoms):
        first = local_atom[atom_first]
        second = local_atom[atom_second]
        if int(bond_columns["bond_kind"][row]) == BOND_KIND_DATIVE:
            direction = int(bond_columns["bond_direction"][row])
            if direction == BOND_DIRECTION_ENDPOINT_0_DONOR:
                donor, acceptor = first, second
            elif direction == BOND_DIRECTION_ENDPOINT_1_DONOR:
                donor, acceptor = second, first
            else:
                raise ValueError(f"{description} has a dative bond without a donor direction")
            editable.AddBond(donor, acceptor, Chem.BondType.DATIVE)
        else:
            if not bool(bond_columns["bond_order_known"][row]):
                raise ValueError(f"{description} has an unknown bond order")
            order = float(bond_columns["bond_order"][row])
            try:
                bond_type = bond_types[order]
            except KeyError:
                raise ValueError(f"{description} has unsupported bond order {order}") from None
            editable.AddBond(first, second, bond_type)
        # GetNumBonds() scans the graph in RDKit; calling it for every added
        # edge makes template preparation quadratic in the number of bonds.
        rdkit_bond = editable.GetBondBetweenAtoms(first, second)
        if bool(bond_columns["bond_aromatic_known"][row]):
            rdkit_bond.SetIsAromatic(bool(bond_columns["bond_aromatic"][row]))
        stereo_index = int(bond_columns["bond_stereo"][row])
        if stereo_index >= 0:
            stereo_first, stereo_second = stereo_atoms_by_edge[row]
            stereo_atoms = (
                None
                if stereo_first < 0 or stereo_second < 0
                else (
                    local_atom[stereo_first],
                    local_atom[stereo_second],
                )
            )
            bond_stereo.append((rdkit_bond.GetIdx(), symbols[stereo_index], stereo_atoms))

    molecule = editable.GetMol()
    if not sanitize:
        return molecule
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(len(atom_indices))
    reference_coordinates = system.reference_coordinates[atom_selection].detach().cpu().tolist()
    reference_mask = system.reference_coordinate_mask[atom_selection].detach().cpu().tolist()
    for index, (coordinates, resolved) in enumerate(zip(reference_coordinates, reference_mask, strict=True)):
        x, y, z = coordinates if resolved else (0.0, 0.0, 0.0)
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    molecule.AddConformer(conformer, assignId=True)

    for index in atom_stereo:
        molecule.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
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
            if not first_neighbors or not second_neighbors:
                raise ValueError(f"{description} has bond stereo without retained defining atoms")

            if stereo.upper() not in {"E", "Z"}:
                raise ValueError(f"{description} has relative bond stereo without defining atoms")
            # A provisional reference pair fixes a geometric orientation only.
            # Exact CIP labeling below determines which orientation is E or Z.
            stereo_atoms = (first_neighbors[0].GetIdx(), second_neighbors[0].GetIdx())
        bond.SetStereoAtoms(*stereo_atoms)
        try:
            stereo_type = stereo_types[stereo.upper()]
        except KeyError:
            raise ValueError(f"{description} has unsupported bond stereo {stereo!r}") from None
        bond.SetStereo(stereo_type)
    _assign_rdkit_stereochemistry(molecule, atom_stereo, bond_stereo, description)
    # Encode declared E/Z on neighboring single bonds before downstream RDKit
    # fragment/canonical-SMILES operations can recompute stereochemistry. This
    # uses the declared bond stereo, never target or placeholder coordinates.
    if bond_stereo:
        Chem.SetDoubleBondNeighborDirections(molecule)
    return molecule


def _assign_rdkit_stereochemistry(
    molecule: Mol,
    atom_targets: dict[int, str],
    declarations: Sequence[tuple[int, str, tuple[int, int] | None]],
    description: str,
) -> None:
    r"""Construct and verify declared atom/bond stereo with exact CIP labels."""

    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    if any(stereo not in {"R", "S", "r", "s"} for stereo in atom_targets.values()):
        raise ValueError(f"{description} has unsupported atom stereo")
    bond_targets = {index: stereo.upper() for index, stereo, _ in declarations if stereo.upper() in {"E", "Z"}}
    if not atom_targets and not bond_targets:
        return
    # Dependent and pseudoasymmetric centers can change priorities when other
    # centers are oriented. Accept only a verified assignment, never a guess
    # or an approximate legacy CIP label. Nonconvergence explicitly rejects.
    for attempt in range(8):
        atoms = [(molecule.GetAtomWithIdx(i), desired) for i, desired in atom_targets.items()]
        bonds = [(molecule.GetBondWithIdx(i), desired) for i, desired in bond_targets.items()]
        stereo_elements: list[tuple[Atom | Bond, str]] = [*atoms, *bonds]
        for element, _ in stereo_elements:
            if element.HasProp("_CIPCode"):
                element.ClearProp("_CIPCode")
        try:
            rdCIPLabeler.AssignCIPLabels(
                molecule,
                atomsToLabel=list(atom_targets),
                bondsToLabel=list(bond_targets),
                maxRecursiveIterations=10_000_000,
            )
        except RuntimeError as error:
            raise ValueError(f"{description} stereo has no exact CIP assignment") from error
        atom_mismatches = []
        bond_mismatches = []
        for atom, desired in atoms:
            assigned = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
            if assigned != desired:
                atom_mismatches.append((atom, assigned, desired))
        for bond, desired in bonds:
            assigned = bond.GetProp("_CIPCode") if bond.HasProp("_CIPCode") else ""
            if assigned != desired:
                bond_mismatches.append((bond, assigned))
        if not atom_mismatches and not bond_mismatches:
            return
        if attempt < 3:
            for atom, assigned, desired in atom_mismatches:
                if assigned in {"R", "S", "r", "s"} and assigned.upper() != desired.upper():
                    atom.InvertChirality()
            for bond, assigned in bond_mismatches:
                if assigned in {"E", "Z"}:
                    bond.SetStereo(
                        Chem.BondStereo.STEREOZ
                        if bond.GetStereo() == Chem.BondStereo.STEREOE
                        else Chem.BondStereo.STEREOE
                    )
        elif attempt < 7:
            # Coupled pseudoasymmetric centers can oscillate if both are flipped
            # together. After the original bulk passes, try one residual change
            # at a time and reverify every declaration before accepting.
            atom_change = next(
                (
                    atom
                    for atom, assigned, desired in atom_mismatches
                    if assigned in {"R", "S", "r", "s"} and assigned.upper() != desired.upper()
                ),
                None,
            )
            bond_change = next((bond for bond, assigned in bond_mismatches if assigned in {"E", "Z"}), None)
            if atom_change is not None:
                atom_change.InvertChirality()
            elif bond_change is not None:
                bond_change.SetStereo(
                    Chem.BondStereo.STEREOZ
                    if bond_change.GetStereo() == Chem.BondStereo.STEREOE
                    else Chem.BondStereo.STEREOE
                )
            else:
                break
    if bond_mismatches and any(assigned not in {"E", "Z"} for _, assigned in bond_mismatches):
        raise ValueError(f"{description} has no unique E/Z assignment")
    raise ValueError(f"{description} stereo conflicts with its exact CIP assignment")
