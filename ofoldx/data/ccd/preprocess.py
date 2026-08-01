r"""CCD-backed structure preparation, in memory or as an artifact."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import partial
from typing import Literal

import torch

from ..chemistry import to_rdkit_system, validate_rdkit_system
from ..molecular import BOND_KIND_COVALENT, BOND_ROLE_COMPONENT, ChemicalComponent, MolecularSystem, form_bond
from ..structure import Structure, save_structure
from .resolver import resolve_ccd_component


def _valence_problem_atoms(system: MolecularSystem) -> frozenset[int] | None:
    r"""Return RDKit valence-problem atoms, or ``None`` for unsupported problems."""

    from rdkit import Chem

    # This pass only decides terminal-proton repairs. Full sanitization remains
    # mandatory in validate(); ring/kekulization work here can be pathological
    # for large template graphs whose valence already makes them invalid.
    problems = Chem.DetectChemistryProblems(
        to_rdkit_system(system, sanitize=False), sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
    )
    if any(problem.GetType() != "AtomValenceException" for problem in problems):
        return None
    return frozenset(int(problem.GetAtomIdx()) for problem in problems)


def _complete_observed_protons(
    structure: Structure,
    resolver: Callable[[str], ChemicalComponent],
    source_formal_charge_mask: torch.Tensor,
) -> Structure:
    r"""Realize source-observed protonation absent from the generic CCD component."""

    from rdkit import Chem

    system = structure.system
    periodic_table = Chem.GetPeriodicTable()
    completed_components: list[int] = []
    for component_index in range(int(system.component_symbol.numel())):
        if bool(system.component_topology_complete[component_index]):
            continue
        component_id = system.symbols[int(system.component_symbol[component_index])]
        try:
            reference = resolver(component_id)
        except KeyError:
            continue
        if reference.bonds is None:
            continue
        atom_start = int(system.component_atom_ptr[component_index])
        atom_stop = int(system.component_atom_ptr[component_index + 1])
        atom_indices = tuple(range(atom_start, atom_stop))
        atom_by_id = {system.symbols[int(system.atom_symbol[index])]: index for index in atom_indices}
        reference_atom_ids = {atom.atom_id for atom in reference.atoms}
        source_only = [index for atom_id, index in atom_by_id.items() if atom_id not in reference_atom_ids]
        if not source_only or any(int(system.atomic_number[index]) != 1 for index in source_only):
            continue

        current_reference_atom_ids = set(atom_by_id) & reference_atom_ids
        expected_edges = {
            frozenset((bond.atom_a, bond.atom_b))
            for bond in reference.bonds
            if {bond.atom_a, bond.atom_b} <= current_reference_atom_ids
        }
        current_edges = {
            frozenset(
                (
                    system.symbols[int(system.atom_symbol[int(system.edge_index[0, edge])])],
                    system.symbols[int(system.atom_symbol[int(system.edge_index[1, edge])])],
                )
            )
            for edge in range(int(system.edge_index.shape[1]))
            if int(system.bond_role[edge]) == BOND_ROLE_COMPONENT
            and atom_start <= int(system.edge_index[0, edge]) < atom_stop
            and atom_start <= int(system.edge_index[1, edge]) < atom_stop
            and int(system.edge_index[0, edge]) not in source_only
            and int(system.edge_index[1, edge]) not in source_only
        }
        if current_edges != expected_edges:
            continue

        inferred: list[tuple[int, int]] = []
        for hydrogen in source_only:
            if not bool(structure.coordinate_mask[hydrogen]):
                break
            if bool((system.edge_index == hydrogen).any()):
                break
            if (bool(system.formal_charge_known[hydrogen]) and int(system.formal_charge[hydrogen]) != 0) or (
                bool(system.radical_electrons_known[hydrogen]) and int(system.radical_electrons[hydrogen]) != 0
            ):
                break
            parents: list[int] = []
            for atom in atom_indices:
                if atom == hydrogen or atom in source_only or not bool(structure.coordinate_mask[atom]):
                    continue
                atomic_number = int(system.atomic_number[atom])
                if atomic_number == 1:
                    continue
                distance = float(
                    torch.linalg.vector_norm(structure.coordinates[atom] - structure.coordinates[hydrogen])
                )
                bond_length = periodic_table.GetRcovalent(1) + periodic_table.GetRcovalent(atomic_number)
                if 0.6 * bond_length <= distance <= 1.3 * bond_length:
                    parents.append(atom)
            if len(parents) != 1:
                break
            parent = parents[0]
            if (
                int(system.atomic_number[parent]) != 7
                or bool(source_formal_charge_mask[parent])
                or not bool(system.formal_charge_known[parent])
                or int(system.formal_charge[parent]) != 0
                or not bool(system.radical_electrons_known[parent])
                or int(system.radical_electrons[parent]) != 0
                or not bool(system.explicit_hydrogens_known[parent])
                or int(system.explicit_hydrogens[parent]) != 0
                or not bool(system.implicit_hydrogens_known[parent])
                or int(system.implicit_hydrogens[parent]) != 0
                or not bool(system.implicit_hydrogens_allowed_known[parent])
                or bool(system.implicit_hydrogens_allowed[parent])
            ):
                break
            has_unobserved_reference_hydrogen = False
            for edge in range(int(system.edge_index.shape[1])):
                if int(system.bond_role[edge]) != BOND_ROLE_COMPONENT:
                    continue
                first, second = (int(value) for value in system.edge_index[:, edge])
                if parent != first and parent != second:
                    continue
                other = second if first == parent else first
                if (
                    other not in source_only
                    and int(system.atomic_number[other]) == 1
                    and not bool(structure.coordinate_mask[other])
                ):
                    has_unobserved_reference_hydrogen = True
                    break
            if has_unobserved_reference_hydrogen:
                break
            inferred.append((hydrogen, parent))
        else:
            for hydrogen, parent in inferred:
                system, _ = form_bond(
                    system,
                    hydrogen,
                    parent,
                    system.edge_index.new_empty(0),
                    system.edge_index.new_empty(0),
                    role="component",
                )
            formal_charge = system.formal_charge.clone()
            formal_charge_known = system.formal_charge_known.clone()
            radical_electrons = system.radical_electrons.clone()
            radical_electrons_known = system.radical_electrons_known.clone()
            for hydrogen, parent in inferred:
                formal_charge[parent] += 1
                formal_charge_known[hydrogen] = True
                formal_charge[hydrogen] = 0
                radical_electrons_known[hydrogen] = True
                radical_electrons[hydrogen] = 0
            system = system.replace(  # type: ignore[attr-defined]
                formal_charge=formal_charge,
                formal_charge_known=formal_charge_known,
                radical_electrons=radical_electrons,
                radical_electrons_known=radical_electrons_known,
            )
            completed_components.append(component_index)

    if not completed_components:
        return structure
    topology_complete = system.component_topology_complete.clone()
    topology_complete[completed_components] = True
    return structure.replace(  # type: ignore[attr-defined]
        system=system.replace(component_topology_complete=topology_complete)  # type: ignore[attr-defined]
    )


def _repair_candidates(structure: Structure, atom_index: int) -> tuple[frozenset[int], ...]:
    r"""Enumerate ordinary unobserved terminal protons displaced at one atom."""

    system = structure.system
    incident = torch.nonzero((system.edge_index == atom_index).any(0), as_tuple=False).flatten().tolist()
    candidates: set[frozenset[int]] = set()
    participates_in_new_covalent_bond = any(
        int(system.bond_role[edge_index]) != BOND_ROLE_COMPONENT
        and int(system.bond_kind[edge_index]) == BOND_KIND_COVALENT
        for edge_index in incident
    )
    if not participates_in_new_covalent_bond:
        return ()
    for edge_index in incident:
        if int(system.bond_role[edge_index]) != BOND_ROLE_COMPONENT:
            continue
        first, second = (int(value) for value in system.edge_index[:, edge_index])
        neighbor = second if first == atom_index else first
        neighbor_edges = torch.nonzero((system.edge_index == neighbor).any(0), as_tuple=False).flatten()
        if (
            int(system.atomic_number[neighbor]) == 1
            and not bool(structure.coordinate_mask[neighbor])
            and int(neighbor_edges.numel()) == 1
            and int(system.bond_kind[edge_index]) == BOND_KIND_COVALENT
            and bool(system.bond_order_known[edge_index])
            and float(system.bond_order[edge_index]) == 1.0
            and bool(system.bond_aromatic_known[edge_index])
            and not bool(system.bond_aromatic[edge_index])
            and (not bool(system.isotope_known[neighbor]) or int(system.isotope[neighbor]) == 1)
            and bool(system.formal_charge_known[neighbor])
            and int(system.formal_charge[neighbor]) == 0
            and bool(system.radical_electrons_known[neighbor])
            and int(system.radical_electrons[neighbor]) == 0
        ):
            candidates.add(frozenset({neighbor}))
    return tuple(sorted(candidates, key=lambda group: tuple(sorted(group))))


def _chemistry_signature(system: MolecularSystem) -> tuple[object, ...]:
    r"""Return a canonical graph signature including absolute stereo facts."""

    from rdkit import Chem

    molecule = to_rdkit_system(system, sanitize=False)

    def marked_smiles(first: int, second: int | None = None) -> str:
        marked = Chem.Mol(molecule)
        marked.GetAtomWithIdx(first).SetAtomMapNum(1)
        if second is not None:
            marked.GetAtomWithIdx(second).SetAtomMapNum(2)
        return Chem.MolToSmiles(marked, canonical=True, isomericSmiles=True)

    atom_stereo = tuple(
        sorted(
            (system.symbols[int(stereo)], marked_smiles(atom_index))
            for atom_index, stereo in enumerate(system.atom_stereo)
            if int(stereo) >= 0
        )
    )
    bond_stereo: list[tuple[str, str]] = []
    for edge_index, stereo in enumerate(system.bond_stereo):
        if int(stereo) < 0:
            continue
        first, second = (int(value) for value in system.edge_index[:, edge_index])
        rooted = min(marked_smiles(first, second), marked_smiles(second, first))
        bond_stereo.append((system.symbols[int(stereo)], rooted))
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        atom_stereo,
        tuple(sorted(bond_stereo)),
    )


def _repair_occurrence_chemistry(structure: Structure) -> Structure:
    r"""Remove uniquely implied, unobserved terminal protons until the graph sanitizes."""

    def validate() -> None:
        try:
            validate_rdkit_system(structure.system)
        except ValueError as error:
            raise ValueError("prepared molecular graph is chemically invalid") from error

    while True:
        problems = _valence_problem_atoms(structure.system)
        if problems is None or not problems:
            validate()
            return structure

        repaired = False
        for problem_atom in sorted(problems):
            valid: list[tuple[frozenset[int], Structure]] = []
            for removal in _repair_candidates(structure, problem_atom):
                keep = torch.ones_like(structure.coordinate_mask)
                keep[list(removal)] = False
                candidate, old_to_new = structure.atom_subgraph(keep)
                candidate_problems = _valence_problem_atoms(candidate.system)
                if candidate_problems is None:
                    continue
                retained = torch.nonzero(old_to_new >= 0, as_tuple=False).flatten()
                candidate_problems_in_source = {int(retained[index]) for index in candidate_problems}
                if problem_atom in candidate_problems_in_source or not candidate_problems_in_source <= problems - {
                    problem_atom
                }:
                    continue
                valid.append((removal, candidate))
            # Canonical graph comparisons are needed only when multiple repairs
            # survive. A unique terminal proton needs no graph-isomorphism work.
            if len(valid) > 1 and len({_chemistry_signature(candidate.system) for _, candidate in valid}) > 1:
                atom_name = structure.system.symbols[int(structure.system.atom_symbol[problem_atom])]
                raise ValueError(f"molecular system has multiple non-equivalent repairs at atom {atom_name!r}")
            if valid:
                structure = min(valid, key=lambda item: tuple(sorted(item[0])))[1]
                repaired = True
                break
        if not repaired:
            validate()
            raise ValueError("RDKit reports unresolved valence problems despite successful sanitization")


def prepare_structure(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str] | None = None,
    *,
    revision: str | None = None,
    snapshot_path: str | os.PathLike[str] | None = None,
    model: int = 1,
    assembly_id: str | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    include_water: bool = True,
    include_hydrogens: bool = True,
    components: Mapping[str, ChemicalComponent] | None = None,
) -> Structure:
    r"""
    Complete one source structure from a pinned CCD, optionally saving it.

    Source-declared atoms, charges, bonds, and coordinates remain authoritative.
    The CCD supplies only missing component chemistry. The completed molecular graph
    is chemically validated before it is returned or saved as a native artifact.

    Args:
        source: Source PDB, gzip-compressed PDB, or mmCIF path.
        destination: Optional output path ending in ``.structure.safetensors``.
            Omit to prepare a training sample in memory.
        revision: Pinned Hub CCD revision.
        snapshot_path: Local CCD CIF, Parquet shard, or shard directory.
        model: One-based source model number.
        assembly_id: Biological assembly identifier, or ``None`` for the stored
            asymmetric unit.
        altloc: Residue-level alternate-location choice.
        include_water: Whether to retain water entities.
        include_hydrogens: Whether to retain hydrogen atoms.
        components: Explicit component chemistry, such as a source SMILES ligand.

    Returns:
        The validated Structure, also saved when ``destination`` is supplied.

    Raises:
        ValueError: If exactly one CCD locator is not supplied, the completed graph
            remains chemically incomplete, or its chemistry is invalid.
    """

    if (revision is None) == (snapshot_path is None):
        raise ValueError("exactly one of revision or snapshot_path is required")

    from ...io import read_structure

    resolver = partial(resolve_ccd_component, revision=revision, snapshot_path=snapshot_path)
    structure = read_structure(
        source,
        model_index=model,
        assembly_id=assembly_id,
        altloc=altloc,
        include_water=include_water,
        include_hydrogens=True,
        component_resolver=resolver,
        components=components,
    )
    structure = _complete_observed_protons(structure, resolver, structure.source_formal_charge_mask)
    structure = _repair_occurrence_chemistry(structure)
    if not include_hydrogens:
        structure = structure.suppress_hydrogens()
    if destination is not None:
        save_structure(structure, destination)
    return structure
