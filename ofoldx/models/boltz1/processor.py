r"""
Input processing for Boltz-1 structure prediction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from ...data.evidence import MSA, MSACollection, msa_bindings, resolve_msa_bindings
from ...data.molecular import (
    AtomSite,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...processor import Processor
from .model import Boltz1Input, Boltz1Layout, Boltz1Pocket

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

# Boltz-1 pocket one-hot classes: UNSPECIFIED, UNSELECTED, POCKET, and BINDER.
_POCKET_UNSPECIFIED = 0
_POCKET_UNSELECTED = 1
_POCKET_POCKET = 2
_POCKET_BINDER = 3
_POCKET_SIZE = 4

# The 33-token residue vocabulary places proteins at 2..21, UNKNOWN tokens at
# 22/27/32, RNA at 23..27, DNA at 28..32, and the MSA gap at 1.
_PROTEIN_RESIDUE_TYPES = {
    component_id: index
    for index, component_id in enumerate(
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ),
        start=2,
    )
}
_RNA_RESIDUE_TYPES = {"A": 23, "G": 24, "C": 25, "U": 26, "N": 27}
_DNA_RESIDUE_TYPES = {"DA": 28, "DG": 29, "DC": 30, "DT": 31, "DN": 32}
_STANDARD_COMPONENTS = {
    "protein": frozenset(_PROTEIN_RESIDUE_TYPES),
    "rna": frozenset(_RNA_RESIDUE_TYPES),
    "dna": frozenset(_DNA_RESIDUE_TYPES),
}
# Boltz parses these polymer components through its canonical residue path and
# therefore does not emit physical-potential constraints for them. MSE is
# normalized to MET before that decision.
_CONSTRAINT_FREE_COMPONENTS = {
    "protein": frozenset((*_PROTEIN_RESIDUE_TYPES, "UNK", "MSE")),
    "rna": frozenset(_RNA_RESIDUE_TYPES),
    "dna": frozenset(_DNA_RESIDUE_TYPES),
}
_MODEL_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}
_UNKNOWN_RESIDUE_TYPES = {"protein": 22, "rna": 27, "dna": 32}
_MSA_GAP = 1
# MSA cells are UPPERCASE one-letter MSA symbols (or the ``-`` gap), not ChemicalComponent IDs. The
# Processor owns the symbol->token mapping keyed by the entity's resolved polymer_type, reproducing the
# Boltz-1 ``_residue_type`` path: standard residues map to their model token and everything else to the
# model's UNKNOWN token. RNA one-letter symbols equal their component IDs directly.
_PROTEIN_MSA_RESTYPE = {
    symbol: _PROTEIN_RESIDUE_TYPES[component_id]
    for symbol, component_id in {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }.items()
}
_DNA_MSA_RESTYPE = {
    symbol: _DNA_RESIDUE_TYPES[component_id]
    for symbol, component_id in {"A": "DA", "G": "DG", "C": "DC", "T": "DT", "N": "DN"}.items()
}


def _msa_symbol_restype(symbol: str, polymer_type: str) -> int:
    r"""
    Map one UPPERCASE one-letter MSA symbol (or ``-``) to a Boltz-1 token.
    """

    if symbol == "-":
        return _MSA_GAP
    if polymer_type == "protein":
        return _PROTEIN_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["protein"])
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(symbol, _UNKNOWN_RESIDUE_TYPES["rna"])
    return _DNA_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["dna"])


class Boltz1Processor(Processor):
    r"""
    Convert supported biomolecular inputs into `Boltz1Input`.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None = None,
        pocket: Boltz1Pocket | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz1Input:
        r"""
        Prepare folding features for one molecular system.

        Args:
            input: Molecular system or structure.
            msa: Optional MSA value or collection.
            pocket: Optional binder and pocket-residue specification.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Batched token, atom, MSA, and pocket tensors. Reference coordinates
            are in angstroms; tensor axes align to ``layout``.
        """

        if not isinstance(input, MolecularSystem | Structure):
            raise TypeError("Boltz1Processor input must be a MolecularSystem or Structure")

        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        if not system.instances:
            raise ValueError("Boltz-1 requires at least one molecular instance")

        if msa is not None:
            resolve_msa_bindings(msa, system)
        return _prepare_boltz1_features(
            system,
            structure=structure,
            msa=msa,
            pocket=pocket,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )


def _prepare_boltz1_features(
    system: MolecularSystem,
    *,
    structure: Structure | None,
    msa: MSA | MSACollection | None,
    pocket: Boltz1Pocket | None = None,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Boltz1Input:
    r"""
    Build the Boltz-1 token, atom, MSA, and identity layout.
    """

    available_sites = {site for site, atom in system.iter_atoms() if atom.element not in _MODEL_HYDROGEN_ELEMENTS}
    entity_index_by_id: dict[str, int] = {}
    instance_count_by_entity: dict[str, int] = {}
    tokens: list[dict[str, object]] = []

    for chain_index, (instance_id, entity_id) in enumerate(system.instances.items()):
        entity = system.entities[entity_id]
        entity_index = entity_index_by_id.setdefault(entity_id, len(entity_index_by_id))
        symmetry_index = instance_count_by_entity.get(entity_id, 0)
        instance_count_by_entity[entity_id] = symmetry_index + 1
        if isinstance(entity, PolymerSequence):
            molecule_type = _MOLECULE_TYPES[entity.polymer_type]
            standards = _STANDARD_COMPONENTS[entity.polymer_type]
            for monomer_index, component in enumerate(entity.components):
                component_id = component.component_id
                atom_sites = tuple(
                    site
                    for atom in component.atoms
                    if atom.element not in _MODEL_HYDROGEN_ELEMENTS
                    and atom.atom_id not in {"OXT", "OP3"}
                    and (site := AtomSite(instance_id, monomer_index, atom.atom_id)) in available_sites
                )
                if not atom_sites:
                    raise ValueError(f"component {component_id!r} has no model-visible heavy atoms")
                is_standard = component_id in standards
                residue_type = (
                    _residue_type(component_id, entity.polymer_type)
                    if is_standard
                    else _UNKNOWN_RESIDUE_TYPES[entity.polymer_type]
                )
                by_name = {site.atom_id: site for site in atom_sites}
                center_name = "CA" if entity.polymer_type == "protein" else "C1'"
                if entity.polymer_type == "protein":
                    distogram_name = "CA" if component_id == "GLY" else "CB"
                    frame_names = ("N", "CA", "C") if is_standard else ("CA", "N", "C")
                else:
                    distogram_name = (
                        "C4"
                        if component_id in {"A", "G", "DA", "DG"}
                        else "C2" if component_id in {"C", "U", "DC", "DT"} else "C1'"
                    )
                    frame_names = ("C1'", "C3'", "C4'")
                center_site = by_name.get(center_name, atom_sites[0])
                distogram_site = by_name.get(distogram_name, center_site)
                tokens.append(
                    {
                        "atom_sites": atom_sites,
                        "token_site": (instance_id, monomer_index),
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": monomer_index,
                        "residue_index": monomer_index,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": molecule_type,
                        "residue_type": residue_type,
                        "center_site": center_site,
                        "distogram_site": distogram_site,
                        "frame_sites": tuple(by_name.get(name, center_site) for name in frame_names),
                    }
                )
        else:
            atom_sites = tuple(
                site
                for component_index, component in enumerate(entity.components)
                for atom in component.atoms
                if atom.element not in _MODEL_HYDROGEN_ELEMENTS
                and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available_sites
            )
            if not atom_sites:
                raise ValueError(f"molecule {instance_id!r} has no model-visible heavy atoms")
            for site in atom_sites:
                tokens.append(
                    {
                        "atom_sites": (site,),
                        "token_site": site,
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": 0,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": 3,
                        "residue_type": 22,
                        "center_site": site,
                        "distogram_site": site,
                        "frame_sites": (site, site, site),
                    }
                )

    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atom_sites"]))
    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    site_to_token = {
        site: token_index
        for token_index, token in enumerate(tokens)
        for site in cast(tuple[AtomSite, ...], token["atom_sites"])
    }
    atom_count = len(atom_sites)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)
    reference_coordinates = _reference_coordinates(system, atom_sites)
    reference_space_by_residue: dict[tuple[str, int], int] = {}
    reference_space_ids = [
        reference_space_by_residue.setdefault(
            (site.instance_id, site.component_index),
            len(reference_space_by_residue),
        )
        for site in atom_sites
    ]
    resolved = torch.zeros(atom_count, dtype=torch.bool)
    target_positions = torch.zeros(padded_atom_count, 3)
    if structure is not None:
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        for atom_index, site in enumerate(atom_sites):
            source_index = coordinate_index.get(site)
            if source_index is not None and bool(structure.coordinate_mask[source_index]):
                resolved[atom_index] = True
                target_positions[atom_index] = torch.from_numpy(structure.coordinates[source_index].copy())

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = torch.tensor(reference_coordinates)
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = torch.tensor(reference_space_ids)
    _augment_reference_positions(
        reference_positions[:atom_count],
        reference_space_id[:atom_count],
        resolved,
        generator=generator,
    )

    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count)
    atom_name_characters = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_names = _model_atom_names(system, atom_sites)
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    for atom_index, (site, atom_name) in enumerate(zip(atom_sites, atom_names)):
        atom = system.atom(site)
        atomic_number[atom_index] = periodic_table.GetAtomicNumber(atom.element)
        reference_charges[atom_index] = atom.formal_charge
        atom_name_characters[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        token_index = site_to_token[site]
        atom_to_token[atom_index] = token_index
        atom_token_index[atom_index] = cast(tuple[AtomSite, ...], tokens[token_index]["atom_sites"]).index(site)

    # Boltz-1 reference-atom features contain positions (3), charge (1), mask (1),
    # element (128), and atom-name characters (4x64=256). The mask channel at index 4
    # is the sole difference from Boltz-2's 388-wide vector.
    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            atom_mask.unsqueeze(-1).float(),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_characters, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
    token_count = len(tokens)
    token_to_rep_atom = torch.tensor(
        [site_to_atom[cast(AtomSite, token["distogram_site"])] for token in tokens],
        dtype=torch.long,
    )
    representative_atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    representative_atom_mask[token_to_rep_atom] = True
    token_bonds = _token_bond_indicator(system, site_to_token, token_count)
    msa_values = _msa_features(msa, tokens, system)
    molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
    residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
    frame_atom_index = torch.tensor(
        [
            [site_to_atom[site] for site in cast(tuple[AtomSite, AtomSite, AtomSite], token["frame_sites"])]
            for token in tokens
        ],
        dtype=torch.long,
    )
    frame_mask = torch.tensor(
        [
            all(resolved[site_to_atom[site]] for site in cast(tuple[AtomSite, ...], token["frame_sites"]))
            for token in tokens
        ]
    )
    cyclic_period = torch.tensor(
        [_cyclic_period(system, cast(str, token["instance_id"])) for token in tokens],
        dtype=torch.float32,
    )
    pocket_feature = _pocket_feature(pocket, tokens, system)
    steering_features = _steering_features(system, site_to_atom)

    tensor_values: dict[str, Tensor] = {
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
        "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
        "entity_index": torch.tensor([cast(int, token["entity_index"]) for token in tokens]),
        "symmetry_index": torch.tensor([cast(int, token["symmetry_index"]) for token in tokens]),
        "token_index": torch.arange(token_count),
        "atom_features": atom_features,
        "atom_to_token": atom_to_token,
        "atom_mask": atom_mask,
        "reference_positions": reference_positions,
        "reference_space_id": reference_space_id,
        "representative_atom_mask": representative_atom_mask,
        "atom_token_index": atom_token_index,
        "token_to_rep_atom": token_to_rep_atom,
        "residue_type_features": torch.nn.functional.one_hot(residue_types, 33).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "molecule_type": molecule_types,
        "frame_atom_index": frame_atom_index,
        "frame_mask": frame_mask,
        "cyclic_period": cyclic_period,
        "pocket_feature": pocket_feature,
        **steering_features,
        **msa_values,
    }
    if structure is not None:
        tensor_values["target_positions"] = target_positions
        tensor_values["target_mask"] = torch.nn.functional.pad(
            resolved,
            (0, padded_atom_count - atom_count),
        )
    placed = {
        name: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        for name, value in tensor_values.items()
    }
    return Boltz1Input(
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        layout=Boltz1Layout(
            system=system,
            atom_sites=atom_sites,
            token_sites=tuple(cast(AtomSite | tuple[str, int], token["token_site"]) for token in tokens),
        ),
    )


def _pocket_feature(
    pocket: Boltz1Pocket | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> Tensor:
    r"""
    Encode an optional ``Boltz1Pocket`` as a four-class ``(N, 4)`` one-hot.

    Every token of the ``binder`` instance becomes ``BINDER``; named pocket
    residues become ``POCKET``; all remaining tokens become ``UNSELECTED``.
    ``pocket=None`` yields all ``UNSPECIFIED`` tokens.
    """

    token_count = len(tokens)
    if pocket is None:
        return torch.nn.functional.one_hot(
            torch.full((token_count,), _POCKET_UNSPECIFIED, dtype=torch.long),
            _POCKET_SIZE,
        ).float()

    if pocket.binder not in system.instances:
        raise ValueError(f"pocket binder instance {pocket.binder!r} is not in the input system")

    def token_reference(token: dict[str, object]) -> ComponentReference:
        site = token["token_site"]
        if isinstance(site, AtomSite):
            return (site.instance_id, site.component_index)
        return cast(ComponentReference, site)

    pocket_refs = set(pocket.pocket)
    available = {token_reference(token) for token in tokens}
    unresolved = pocket_refs - available
    if unresolved:
        raise ValueError(f"pocket references do not resolve to any token: {sorted(unresolved)}")

    labels = []
    for token in tokens:
        if token["instance_id"] == pocket.binder:
            labels.append(_POCKET_BINDER)
        elif token_reference(token) in pocket_refs:
            labels.append(_POCKET_POCKET)
        else:
            labels.append(_POCKET_UNSELECTED)
    return torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.long), _POCKET_SIZE).float()


def _token_bond_indicator(
    system: MolecularSystem,
    site_to_token: dict[AtomSite, int],
    token_count: int,
) -> Tensor:
    r"""
    Build the symmetric inter-token bond indicator ``[N, N]`` consumed by ``token_bond_projection``.
    """

    token_bonds = torch.zeros(token_count, token_count)

    def add(site_a: AtomSite, site_b: AtomSite) -> None:
        token_a = site_to_token.get(site_a)
        token_b = site_to_token.get(site_b)
        if token_a is None or token_b is None or token_a == token_b:
            return
        token_bonds[token_a, token_b] = token_bonds[token_b, token_a] = 1

    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if not isinstance(entity, Molecule):
            continue
        for molecule_bond in entity.bonds:
            add(
                AtomSite(instance_id, molecule_bond.component_a, molecule_bond.atom_a),
                AtomSite(instance_id, molecule_bond.component_b, molecule_bond.atom_b),
            )
    for system_bond in system.bonds:
        add(system_bond.site_a, system_bond.site_b)
    return token_bonds


def _steering_features(
    system: MolecularSystem,
    site_to_atom: dict[AtomSite, int],
) -> dict[str, Tensor]:
    r"""
    Lower molecular geometry, assembly connections, and entity copies for Boltz-1x.
    """

    geometry_pairs: list[tuple[int, int]] = []
    geometry_bond_mask: list[bool] = []
    geometry_angle_mask: list[bool] = []
    geometry_upper_bounds: list[float] = []
    geometry_lower_bounds: list[float] = []
    chiral_atoms: list[tuple[int, int, int, int]] = []
    chiral_is_r: list[bool] = []
    stereo_bonds: list[tuple[int, int, int, int]] = []
    stereo_is_e: list[bool] = []
    planar_bonds: list[tuple[int, int, int, int, int, int]] = []
    constraint_scopes: list[tuple[Mol, dict[int, int]]] = []

    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            for component_index, component in enumerate(entity.components):
                if component.component_id in _CONSTRAINT_FREE_COMPONENTS[entity.polymer_type]:
                    continue
                rdkit_molecule = _canonical_rdkit_molecule(Molecule((component,)))
                atom_index = {
                    local_index: site_to_atom[site]
                    for local_index, atom in enumerate(component.atoms)
                    if (site := AtomSite(instance_id, component_index, atom.atom_id)) in site_to_atom
                }
                constraint_scopes.append((rdkit_molecule, atom_index))
        else:
            rdkit_molecule = _canonical_rdkit_molecule(entity)
            atom_index: dict[int, int] = {}
            local_index = 0
            for component_index, component in enumerate(entity.components):
                for atom in component.atoms:
                    site = AtomSite(instance_id, component_index, atom.atom_id)
                    if site in site_to_atom:
                        atom_index[local_index] = site_to_atom[site]
                    local_index += 1
            constraint_scopes.append((rdkit_molecule, atom_index))

    for rdkit_molecule, atom_index in constraint_scopes:
        (
            scope_geometry_pairs,
            scope_bond_mask,
            scope_angle_mask,
            scope_upper_bounds,
            scope_lower_bounds,
            scope_chiral_atoms,
            scope_chiral_is_r,
            scope_stereo_bonds,
            scope_stereo_is_e,
            scope_planar_bonds,
        ) = _rdkit_constraint_features(rdkit_molecule, atom_index)
        geometry_pairs.extend(scope_geometry_pairs)
        geometry_bond_mask.extend(scope_bond_mask)
        geometry_angle_mask.extend(scope_angle_mask)
        geometry_upper_bounds.extend(scope_upper_bounds)
        geometry_lower_bounds.extend(scope_lower_bounds)
        chiral_atoms.extend(scope_chiral_atoms)
        chiral_is_r.extend(scope_chiral_is_r)
        stereo_bonds.extend(scope_stereo_bonds)
        stereo_is_e.extend(scope_stereo_is_e)
        planar_bonds.extend(scope_planar_bonds)

    chain_index = {instance_id: index for index, instance_id in enumerate(system.instances)}
    connections = [bond for bond in system.bonds if bond.site_a in site_to_atom and bond.site_b in site_to_atom]
    connections.sort(
        key=lambda bond: (
            min(site_to_atom[bond.site_a], site_to_atom[bond.site_b]),
            max(site_to_atom[bond.site_a], site_to_atom[bond.site_b]),
            bond.order,
            bond.aromatic,
            bond.stereo or "",
            bond.origin,
            repr(bond.stereo_sites),
        )
    )
    connected_chains = [
        (chain_index[bond.site_a.instance_id], chain_index[bond.site_b.instance_id]) for bond in connections
    ]
    connected_atoms = [(site_to_atom[bond.site_a], site_to_atom[bond.site_b]) for bond in connections]

    instances = tuple(system.instances.items())
    symmetric_chains = [
        (first_index, second_index)
        for first_index, (_first_instance, first_entity) in enumerate(instances)
        for second_index, (_second_instance, second_entity) in enumerate(instances)
        if first_index < second_index and first_entity == second_entity
    ]

    return {
        "geometry_pair_index": _index_columns(geometry_pairs, 2),
        "geometry_bond_mask": torch.tensor(geometry_bond_mask, dtype=torch.bool),
        "geometry_angle_mask": torch.tensor(geometry_angle_mask, dtype=torch.bool),
        "geometry_upper_bounds": torch.tensor(geometry_upper_bounds, dtype=torch.float32),
        "geometry_lower_bounds": torch.tensor(geometry_lower_bounds, dtype=torch.float32),
        "chiral_atom_index": _index_columns(chiral_atoms, 4),
        "chiral_atom_is_r": torch.tensor(chiral_is_r, dtype=torch.bool),
        "stereo_bond_index": _index_columns(stereo_bonds, 4),
        "stereo_bond_is_e": torch.tensor(stereo_is_e, dtype=torch.bool),
        "planar_bond_index": _index_columns(planar_bonds, 6),
        "connected_chain_index": _index_columns(connected_chains, 2),
        "connected_atom_index": _index_columns(connected_atoms, 2),
        "symmetric_chain_index": _index_columns(symmetric_chains, 2),
    }


def _index_columns(values: list[tuple[int, ...]], width: int) -> Tensor:
    if not values:
        return torch.empty(width, 0, dtype=torch.long)
    return torch.tensor(values, dtype=torch.long).T.contiguous()


def _rdkit_constraint_features(
    molecule: Mol,
    atom_index: dict[int, int],
) -> tuple[
    list[tuple[int, int]],
    list[bool],
    list[bool],
    list[float],
    list[float],
    list[tuple[int, int, int, int]],
    list[bool],
    list[tuple[int, int, int, int]],
    list[bool],
    list[tuple[int, int, int, int, int, int]],
]:
    from rdkit import Chem
    from rdkit.Chem import HybridizationType
    from rdkit.Chem.rdchem import BondStereo
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

    molecule = Chem.AddHs(molecule)
    molecule.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(molecule)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    bounds = GetMoleculeBoundsMatrix(
        molecule,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bond_pattern = Chem.MolFromSmarts("*~*")
    angle_pattern = Chem.MolFromSmarts("*~*~*")
    bond_pairs = {tuple(sorted(match)) for match in molecule.GetSubstructMatches(bond_pattern)}
    angle_pairs = {tuple(sorted((match[0], match[2]))) for match in molecule.GetSubstructMatches(angle_pattern)}

    geometry_pairs: list[tuple[int, int]] = []
    geometry_bond_mask: list[bool] = []
    geometry_angle_mask: list[bool] = []
    geometry_upper_bounds: list[float] = []
    geometry_lower_bounds: list[float] = []
    for first in range(molecule.GetNumAtoms()):
        for second in range(first + 1, molecule.GetNumAtoms()):
            if first not in atom_index or second not in atom_index:
                continue
            geometry_pairs.append((atom_index[first], atom_index[second]))
            geometry_bond_mask.append((first, second) in bond_pairs)
            geometry_angle_mask.append((first, second) in angle_pairs)
            geometry_upper_bounds.append(float(bounds[first, second]))
            geometry_lower_bounds.append(float(bounds[second, first]))

    chiral_atoms: list[tuple[int, int, int, int]] = []
    chiral_is_r: list[bool] = []
    if all(atom.HasProp("_CIPRank") for atom in molecule.GetAtoms()):
        for center_index, orientation in Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=False,
        ):
            center = molecule.GetAtomWithIdx(center_index)
            neighbors = tuple(
                index
                for index, _rank in sorted(
                    ((neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank"))) for neighbor in center.GetNeighbors()),
                    key=lambda value: value[1],
                    reverse=True,
                )
            )
            if len(neighbors) > 4 or center.GetHybridization() != HybridizationType.SP3:
                continue
            indices = (*neighbors[:3], center_index)
            if all(index in atom_index for index in indices):
                chiral_atoms.append(
                    cast(
                        tuple[int, int, int, int],
                        tuple(atom_index[index] for index in indices),
                    )
                )
                chiral_is_r.append(orientation == "R")
            if len(neighbors) == 4:
                for skipped in range(3):
                    neighbor_set = neighbors[:skipped] + neighbors[skipped + 1 :]
                    indices = (
                        neighbor_set[::-1] + (center_index,) if skipped % 2 == 0 else neighbor_set + (center_index,)
                    )
                    if all(index in atom_index for index in indices):
                        chiral_atoms.append(
                            cast(
                                tuple[int, int, int, int],
                                tuple(atom_index[index] for index in indices),
                            )
                        )
                        chiral_is_r.append(orientation == "R")

    stereo_bonds: list[tuple[int, int, int, int]] = []
    stereo_is_e: list[bool] = []
    if all(atom.HasProp("_CIPRank") for atom in molecule.GetAtoms()):
        for bond in molecule.GetBonds():
            stereo = bond.GetStereo()
            if stereo not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
                continue
            start_index = bond.GetBeginAtomIdx()
            end_index = bond.GetEndAtomIdx()
            start_neighbors = [
                index
                for index, _rank in sorted(
                    (
                        (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                        for neighbor in molecule.GetAtomWithIdx(start_index).GetNeighbors()
                        if neighbor.GetIdx() != end_index
                    ),
                    key=lambda value: value[1],
                    reverse=True,
                )
            ]
            end_neighbors = [
                index
                for index, _rank in sorted(
                    (
                        (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                        for neighbor in molecule.GetAtomWithIdx(end_index).GetNeighbors()
                        if neighbor.GetIdx() != start_index
                    ),
                    key=lambda value: value[1],
                    reverse=True,
                )
            ]
            indices = (start_neighbors[0], start_index, end_index, end_neighbors[0])
            if all(index in atom_index for index in indices):
                stereo_bonds.append(
                    cast(
                        tuple[int, int, int, int],
                        tuple(atom_index[index] for index in indices),
                    )
                )
                stereo_is_e.append(stereo == BondStereo.STEREOE)
            if len(start_neighbors) == 2 and len(end_neighbors) == 2:
                indices = (start_neighbors[1], start_index, end_index, end_neighbors[1])
                if all(index in atom_index for index in indices):
                    stereo_bonds.append(
                        cast(
                            tuple[int, int, int, int],
                            tuple(atom_index[index] for index in indices),
                        )
                    )
                    stereo_is_e.append(stereo == BondStereo.STEREOE)

    planar_pattern = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    planar_bonds = [
        cast(
            tuple[int, int, int, int, int, int],
            tuple(atom_index[index] for index in match),
        )
        for match in molecule.GetSubstructMatches(planar_pattern)
        if all(index in atom_index for index in match)
    ]
    return (
        geometry_pairs,
        geometry_bond_mask,
        geometry_angle_mask,
        geometry_upper_bounds,
        geometry_lower_bounds,
        chiral_atoms,
        chiral_is_r,
        stereo_bonds,
        stereo_is_e,
        planar_bonds,
    )


def _msa_features(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> dict[str, Tensor]:
    residue_types = [cast(int, token["residue_type"]) for token in tokens]
    if msa is None:
        rows = torch.tensor([residue_types])
        deletion_counts = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        # The MSA holds no PolymerSequence: column ``c`` binds ``columns[c].entity_id`` to a system entity,
        # a token's monomer index is directly the aligned-row column, and cells are UPPERCASE one-letter
        # symbols (or ``-``). A bare MSA is a single-chain plan over its unpaired pool; a collection replays
        # its dense ``row_pairing`` and ``row_is_paired`` plan across the paired and unpaired pools.
        bindings = msa_bindings(msa)
        columns = tuple(value for value, _target_instance_id in bindings)
        if isinstance(msa, MSA):
            plan_rows: list[tuple[tuple[int, int, bool], ...]] = [
                ((0, row_index, False),) for row_index in range(len(msa.unpaired_sequences))
            ]
        else:
            plan_rows = [
                tuple(
                    (column_index, int(msa.row_pairing[plan_index, column_index]), bool(is_paired))
                    for column_index, is_paired in enumerate(msa.row_is_paired[plan_index])
                    if msa.row_pairing[plan_index, column_index] != -1
                )
                for plan_index in range(msa.row_pairing.shape[0])
            ]
        polymer_types = tuple(cast(PolymerSequence, system.entities[value.entity_id]).polymer_type for value in columns)
        row_values: list[Tensor] = []
        deletion_values: list[Tensor] = []
        paired_values: list[Tensor] = []
        for selections in plan_rows:
            row = torch.full((len(tokens),), _MSA_GAP, dtype=torch.long)
            deletion_row = torch.zeros(len(tokens))
            paired_row = torch.zeros(len(tokens))
            present = len(selections)
            # A taxonomy-paired plan row draws every populated cell from the paired pool; the dense
            # ``row_is_paired`` flag records exactly that; more than one populated paired column denotes
            # a taxonomy-paired row.
            taxonomic_pair = (
                isinstance(msa, MSACollection) and present > 1 and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                target_instance_id = bindings[column_index][1]
                polymer_type = polymer_types[column_index]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, token in enumerate(tokens):
                    if (
                        token["entity_id"] != value.entity_id
                        or (target_instance_id is not None and token["instance_id"] != target_instance_id)
                        or token["monomer_index"] is None
                    ):
                        continue
                    monomer_index = cast(int, token["monomer_index"])
                    row[token_index] = _msa_symbol_restype(aligned[monomer_index], polymer_type)
                    deletion_row[token_index] = float(deletion_source[monomer_index])
                    paired_row[token_index] = float(taxonomic_pair)
            row_values.append(row)
            deletion_values.append(deletion_row)
            paired_values.append(paired_row)
        rows = torch.stack(row_values)
        deletion_counts = torch.stack(deletion_values)
        paired = torch.stack(paired_values)
        rows[0] = torch.tensor(residue_types)
        paired[0] = 1
    has_deletion = deletion_counts > 0
    deletion_value = (math.pi / 2) * torch.atan(deletion_counts / 3)
    return {
        "msa_tokens": rows,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "msa_paired": paired,
        "msa_mask": torch.ones_like(rows, dtype=torch.bool),
        "msa_profile": torch.nn.functional.one_hot(rows, 33).float().mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


def _reference_coordinates(system: MolecularSystem, sites: tuple[AtomSite, ...]) -> list[tuple[float, float, float]]:
    generated: dict[str, dict[tuple[int, str], tuple[float, float, float]]] = {}
    coordinates: list[tuple[float, float, float]] = []
    for site in sites:
        entity_id = system.instances[site.instance_id]
        entity = system.entities[entity_id]
        component = entity.components[site.component_index]
        atom_index = next(index for index, atom in enumerate(component.atoms) if atom.atom_id == site.atom_id)
        coordinate = component.reference_coordinates[atom_index] if component.reference_coordinates else None
        if coordinate is None and isinstance(entity, Molecule):
            if entity_id not in generated:
                generated[entity_id] = _molecule_reference_coordinates(entity)
            coordinate = generated[entity_id][(site.component_index, site.atom_id)]
        if coordinate is None:
            raise ValueError(f"Boltz-1 requires reference coordinates for {site!r}")
        coordinates.append(coordinate)
    return coordinates


def _molecule_reference_coordinates(molecule: Molecule) -> dict[tuple[int, str], tuple[float, float, float]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rdkit_molecule = Chem.AddHs(_canonical_rdkit_molecule(molecule))
    options = AllChem.ETKDGv3()
    options.clearConfs = False
    options.randomSeed = 42
    conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    AllChem.UFFOptimizeMolecule(rdkit_molecule, confId=conformer_id, maxIters=1000)
    conformer = rdkit_molecule.GetConformer(conformer_id)
    coordinates: dict[tuple[int, str], tuple[float, float, float]] = {}
    flat_index = 0
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            position = conformer.GetAtomPosition(flat_index)
            coordinates[(component_index, atom.atom_id)] = (position.x, position.y, position.z)
            flat_index += 1
    return coordinates


def _augment_reference_positions(
    positions: Tensor,
    space_ids: Tensor,
    resolved: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> None:
    r"""
    Randomly rotate and translate each resolved reference conformer.

    Random values are drawn from ``generator``; ``None`` uses the ambient PyTorch
    RNG.
    """

    if not len(space_ids):
        return
    for space_id in range(int(space_ids.max().item()) + 1):
        included = space_ids == space_id
        mask = resolved[included]
        if not bool(mask.any()):
            continue
        values = positions[included]
        center = (values * mask[:, None]).sum(dim=0, keepdim=True) / mask.sum()
        quaternion = torch.randn(
            4,
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )
        norm = torch.sqrt((quaternion * quaternion).sum())
        quaternion = quaternion / torch.copysign(norm, quaternion[0])
        real, i, j, k = quaternion
        two = 2.0 / (quaternion * quaternion).sum()
        rotation = torch.stack(
            (
                1 - two * (j * j + k * k),
                two * (i * j - k * real),
                two * (i * k + j * real),
                two * (i * j + k * real),
                1 - two * (i * i + k * k),
                two * (j * k - i * real),
                two * (i * k - j * real),
                two * (j * k + i * real),
                1 - two * (i * i + j * j),
            )
        ).reshape(3, 3)
        positions[included] = (values - center) @ rotation + torch.randn(
            1,
            3,
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )


def _model_atom_names(system: MolecularSystem, sites: tuple[AtomSite, ...]) -> list[str]:
    names: list[str] = []
    counts: dict[tuple[str, str], int] = {}
    for site in sites:
        name = site.atom_id
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = system.atom(site).element.upper()
            key = (site.instance_id, element)
            counts[key] = counts.get(key, 0) + 1
            name = f"{element}{counts[key]}"
        if len(name) > 4:
            raise ValueError(f"Boltz-1 atom name cannot be encoded in four characters: {name!r}")
        names.append(name)
    return names


def _canonical_rdkit_molecule(molecule: Molecule) -> Mol:
    r"""
    Materialize molecule data as an RDKit graph.
    """

    from rdkit import Chem

    editable = Chem.RWMol()
    atom_index: dict[tuple[int, str], int] = {}
    atom_stereo: dict[int, str] = {}
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            rdkit_atom = Chem.Atom(atom.element)
            if atom.isotope is not None:
                rdkit_atom.SetIsotope(atom.isotope)
            if atom.formal_charge is not None:
                rdkit_atom.SetFormalCharge(atom.formal_charge)
            if atom.radical_electrons is not None:
                rdkit_atom.SetNumRadicalElectrons(atom.radical_electrons)
            if atom.explicit_hydrogens is not None:
                rdkit_atom.SetNumExplicitHs(atom.explicit_hydrogens)
            if atom.implicit_hydrogens_allowed is not None:
                rdkit_atom.SetNoImplicit(not atom.implicit_hydrogens_allowed)
            if atom.aromatic is not None:
                rdkit_atom.SetIsAromatic(atom.aromatic)
            index = editable.AddAtom(rdkit_atom)
            atom_index[(component_index, atom.atom_id)] = index
            if atom.stereo is not None:
                atom_stereo[index] = atom.stereo.upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for component_index, component in enumerate(molecule.components):
        component_bonds = sorted(
            component.bonds or (),
            key=lambda bond: (
                atom_index[(component_index, bond.atom_a)],
                atom_index[(component_index, bond.atom_b)],
            ),
        )
        for component_bond in component_bonds:
            editable.AddBond(
                atom_index[(component_index, component_bond.atom_a)],
                atom_index[(component_index, component_bond.atom_b)],
                bond_types[component_bond.order],
            )
            bond_index = editable.GetNumBonds() - 1
            editable.GetBondWithIdx(bond_index).SetIsAromatic(component_bond.aromatic)
            if component_bond.stereo is not None:
                stereo_atoms = (
                    tuple(atom_index[(component_index, atom_id)] for atom_id in component_bond.stereo_atoms)
                    if component_bond.stereo_atoms is not None
                    else None
                )
                bond_stereo.append(
                    (
                        bond_index,
                        component_bond.stereo,
                        cast(tuple[int, int] | None, stereo_atoms),
                    )
                )
    molecule_bonds = sorted(
        molecule.bonds,
        key=lambda bond: (
            atom_index[(bond.component_a, bond.atom_a)],
            atom_index[(bond.component_b, bond.atom_b)],
        ),
    )
    for molecule_bond in molecule_bonds:
        editable.AddBond(
            atom_index[(molecule_bond.component_a, molecule_bond.atom_a)],
            atom_index[(molecule_bond.component_b, molecule_bond.atom_b)],
            bond_types[molecule_bond.order],
        )
        bond_index = editable.GetNumBonds() - 1
        editable.GetBondWithIdx(bond_index).SetIsAromatic(molecule_bond.aromatic)
        if molecule_bond.stereo is not None:
            stereo_atoms = (
                tuple(atom_index[site] for site in molecule_bond.stereo_atoms)
                if molecule_bond.stereo_atoms is not None
                else None
            )
            bond_stereo.append(
                (
                    bond_index,
                    molecule_bond.stereo,
                    cast(tuple[int, int] | None, stereo_atoms),
                )
            )

    value = editable.GetMol()
    Chem.SanitizeMol(value)
    for index in atom_stereo:
        value.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)
    for index, desired in atom_stereo.items():
        atom = value.GetAtomWithIdx(index)
        assigned = atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else ""
        if assigned != desired:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)

    stereo_types = {
        "E": Chem.BondStereo.STEREOE,
        "TRANS": Chem.BondStereo.STEREOE,
        "Z": Chem.BondStereo.STEREOZ,
        "CIS": Chem.BondStereo.STEREOZ,
    }
    for index, stereo, stereo_atoms in bond_stereo:
        bond = value.GetBondWithIdx(index)
        if stereo_atoms is None:
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            begin_neighbors = [
                neighbor for neighbor in value.GetAtomWithIdx(begin).GetNeighbors() if neighbor.GetIdx() != end
            ]
            end_neighbors = [
                neighbor for neighbor in value.GetAtomWithIdx(end).GetNeighbors() if neighbor.GetIdx() != begin
            ]
            stereo_atoms = (
                max(begin_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
                max(end_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
            )
        bond.SetStereoAtoms(*stereo_atoms)
        bond.SetStereo(stereo_types[stereo.upper()])
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)
    return value


def _cyclic_period(system: MolecularSystem, instance_id: str) -> int:
    entity = system.entities[system.instances[instance_id]]
    if not isinstance(entity, PolymerSequence) or len(entity) <= 2:
        return 0
    for bond in system.bonds:
        if bond.site_a.instance_id != instance_id or bond.site_b.instance_id != instance_id:
            continue
        if {bond.site_a.component_index, bond.site_b.component_index} == {0, len(entity) - 1}:
            return len(entity)
    return 0


def _residue_type(component_id: str, polymer_type: str) -> int:
    if polymer_type == "protein":
        return _PROTEIN_RESIDUE_TYPES.get(component_id, 22)
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(component_id, 27)
    return _DNA_RESIDUE_TYPES.get(component_id, 32)
