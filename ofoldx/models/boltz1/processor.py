r"""Canonical molecular-system lowering for the released Boltz-1 structure checkpoint."""

from __future__ import annotations

import math
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from multimolecule.io import read_fasta_records
from torch import Tensor
from typing_extensions import Self

from ...data.evidence import MSA, MSACollection
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem, Molecule, PolymerSequence
from ...data.structure import Structure
from ...io.structure import read_structure
from .model import (
    _POCKET_BINDER,
    _POCKET_POCKET,
    _POCKET_SIZE,
    _POCKET_UNSELECTED,
    _POCKET_UNSPECIFIED,
    Boltz1Input,
    Boltz1Layout,
    Boltz1Pocket,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

# Boltz-1 shares the 33-token residue vocabulary with Boltz-2 (``vocab_size=33`` in both configs): protein
# residues occupy tokens 2..21, the family UNKNOWN tokens are 22/27/32, RNA 23..27, DNA 28..32, and token 1
# is the MSA gap. The Processor owns this symbol->token mapping exactly as ``boltz2/processor.py`` does.
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
_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}
_UNKNOWN_RESIDUE_TYPES = {"protein": 22, "rna": 27, "dna": 32}
_MSA_GAP = 1
# MSA cells are UPPERCASE one-letter MSA symbols (or the ``-`` gap), not ChemicalComponent IDs. The
# Processor owns the symbol->token mapping keyed by the entity's resolved polymer_type, reproducing the
# family ``_residue_type`` path: standard residues map to their model token and everything else to the
# family UNKNOWN token. RNA one-letter symbols equal their component IDs directly.
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
    r"""Map one UPPERCASE one-letter MSA symbol (or ``-``) to a Boltz-1 token."""

    if symbol == "-":
        return _MSA_GAP
    if polymer_type == "protein":
        return _PROTEIN_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["protein"])
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(symbol, _UNKNOWN_RESIDUE_TYPES["rna"])
    return _DNA_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["dna"])


class Boltz1Processor:
    r"""Lower one canonical molecular system to ``Boltz1Input``.

    Mirrors the released Boltz-1 featurizer: residue-token geometry, the 33-token
    residue/MSA vocabulary, the shared EDM reference-atom feature vector (with the
    Boltz-1 atom-pad-mask channel), the four-class pocket one-hot (all-``UNSPECIFIED``
    unless a ``Boltz1Pocket`` is supplied), and the per-token molecule/frame fields
    the confidence-ranking head consumes. The
    optional Boltz-1x physical-potential constraint surface is left empty (the
    model materializes empty steering features from the absent fields).
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update output placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        pocket: Boltz1Pocket | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz1Input:
        r"""Lower one folding input, its optional MSA evidence, and an optional pocket to ``Boltz1Input``.

        ``generator`` (``None`` means the ambient RNG) drives the reference-conformer augmentation;
        the seed→generator resolution itself lives once at the Pipeline boundary (principle P1).
        """

        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        if not isinstance(input, MolecularSystem | Structure):
            raise TypeError("Boltz1Processor input must be a MolecularSystem, Structure, or structure path")

        structure = input if isinstance(input, Structure) else None
        system = input.system if structure is not None else input
        if not system.instances:
            raise ValueError("Boltz-1 requires at least one molecular instance")

        normalized_msa = _normalize_msa(msa, system)
        return _prepare_boltz1_features(
            system,
            structure=structure,
            msa=normalized_msa,
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
    r"""Build the released Boltz-1 token, atom, MSA, and identity layout."""

    available_sites = {site for site, atom in system.iter_atoms() if atom.element != "H"}
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
                    if atom.element != "H"
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
                if atom.element != "H"
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

    ref_pos = torch.zeros(padded_atom_count, 3)
    ref_pos[:atom_count] = torch.tensor(reference_coordinates)
    ref_space = torch.zeros(padded_atom_count, dtype=torch.long)
    ref_space[:atom_count] = torch.tensor(reference_space_ids)
    _augment_reference_positions(ref_pos[:atom_count], ref_space[:atom_count], resolved, generator=generator)

    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    charge = torch.zeros(padded_atom_count)
    atom_name_chars = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_names = _model_atom_names(system, atom_sites)
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    for atom_index, (site, atom_name) in enumerate(zip(atom_sites, atom_names)):
        atom = system.atom(site)
        atomic_number[atom_index] = periodic_table.GetAtomicNumber(atom.element)
        charge[atom_index] = atom.formal_charge
        atom_name_chars[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        token_index = site_to_token[site]
        atom_to_token[atom_index] = token_index
        atom_token_index[atom_index] = cast(tuple[AtomSite, ...], tokens[token_index]["atom_sites"]).index(site)

    # Boltz-1 reference-atom feature vector (389): [ref_pos(3), charge(1), atom_pad_mask(1), element(128),
    # atom_name_chars(4x64=256)]. The atom-pad-mask channel at index 4 is the sole difference from Boltz-2's
    # 388-wide vector; the model splices nothing, so the Processor emits the full 389 here.
    atom_features = torch.cat(
        (
            ref_pos,
            charge.unsqueeze(-1),
            atom_mask.unsqueeze(-1).float(),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_chars, 64).reshape(padded_atom_count, 256).float(),
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
        "reference_positions": ref_pos,
        "reference_space_id": ref_space,
        "representative_atom_mask": representative_atom_mask,
        "atom_token_index": atom_token_index,
        "token_to_rep_atom": token_to_rep_atom,
        "residue_type": torch.nn.functional.one_hot(residue_types, 33).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "molecule_type": molecule_types,
        "chain_type": molecule_types,
        "frame_atom_index": frame_atom_index,
        "frame_mask": frame_mask,
        "cyclic_period": cyclic_period,
        "pocket_feature": pocket_feature,
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
    r"""Lower one optional ``Boltz1Pocket`` to the released four-class pocket one-hot ``[N, 4]``.

    Mirrors the released inference path (upstream ``featurizer.process_token_features``): every token of the
    ``binder`` instance becomes ``BINDER``; every token whose ``ComponentReference`` is named in ``pocket``
    becomes ``POCKET``; every remaining token becomes ``UNSELECTED``. ``pocket=None`` keeps the released
    all-``UNSPECIFIED`` default. A ``binder`` instance or a ``pocket`` reference that does not resolve in the
    input system fails here rather than silently degrading.
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
    r"""Build the symmetric inter-token bond indicator ``[N, N]`` consumed by ``token_bond_projection``."""

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
        for bond in entity.bonds:
            add(
                AtomSite(instance_id, bond.component_a, bond.atom_a),
                AtomSite(instance_id, bond.component_b, bond.atom_b),
            )
    for bond in system.bonds:
        add(bond.site_a, bond.site_b)
    return token_bonds


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
        # its dense row_pairing/row_is_paired plan across the paired and unpaired pools -- the same partition
        # the pre-migration nested-tuple plan drew, now read off the dense arrays (frozen §3.6 evidence API).
        columns = (msa,) if isinstance(msa, MSA) else msa.msas
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
        polymer_types = {
            value.entity_id: cast(PolymerSequence, system.entities[value.entity_id]).polymer_type for value in columns
        }
        row_values: list[Tensor] = []
        deletion_values: list[Tensor] = []
        paired_values: list[Tensor] = []
        for selections in plan_rows:
            row = torch.full((len(tokens),), _MSA_GAP, dtype=torch.long)
            deletion_row = torch.zeros(len(tokens))
            paired_row = torch.zeros(len(tokens))
            present = len(selections)
            # A taxonomy-paired plan row draws every populated cell from the paired pool; the dense
            # ``row_is_paired`` flag records exactly that, so ``present > 1`` paired-pool rows reproduce the
            # pre-migration taxonomic-pair semantics.
            taxonomic_pair = (
                isinstance(msa, MSACollection) and present > 1 and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                polymer_type = polymer_types[value.entity_id]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, token in enumerate(tokens):
                    if token["entity_id"] != value.entity_id or token["monomer_index"] is None:
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
        "msa": rows,
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
    r"""Apply the released conformer augmentation with a caller-injected RNG.

    Per OWNER principle P1 the seed→generator resolution lives at the Pipeline boundary; this
    reference-conformer roto-translation draws from the injected ``generator`` (``None`` means the
    ambient RNG, matching the upstream Boltz featurizer). No hardcoded seed is created here.
    """

    if not len(space_ids):
        return
    for space_id in range(int(space_ids.max().item())):
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
    r"""Materialize canonical molecule data as RDKit chemistry without consulting provenance."""

    from rdkit import Chem

    editable = Chem.RWMol()
    atom_index: dict[tuple[int, str], int] = {}
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            rdkit_atom = Chem.Atom(atom.element)
            if atom.isotope is not None:
                rdkit_atom.SetIsotope(atom.isotope)
            rdkit_atom.SetFormalCharge(atom.formal_charge)
            rdkit_atom.SetNumRadicalElectrons(atom.radical_electrons)
            rdkit_atom.SetNumExplicitHs(atom.explicit_hydrogens)
            rdkit_atom.SetNoImplicit(not atom.implicit_hydrogens_allowed)
            rdkit_atom.SetIsAromatic(atom.aromatic)
            atom_index[(component_index, atom.atom_id)] = editable.AddAtom(rdkit_atom)

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    for component_index, component in enumerate(molecule.components):
        for bond in component.bonds or ():
            editable.AddBond(
                atom_index[(component_index, bond.atom_a)],
                atom_index[(component_index, bond.atom_b)],
                bond_types[bond.order],
            )
            editable.GetBondWithIdx(editable.GetNumBonds() - 1).SetIsAromatic(bond.aromatic)
    for bond in molecule.bonds:
        editable.AddBond(
            atom_index[(bond.component_a, bond.atom_a)],
            atom_index[(bond.component_b, bond.atom_b)],
            bond_types[bond.order],
        )
        editable.GetBondWithIdx(editable.GetNumBonds() - 1).SetIsAromatic(bond.aromatic)

    value = editable.GetMol()
    Chem.SanitizeMol(value)
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


def _normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    if isinstance(msa, MSA | MSACollection):
        values = (msa,) if isinstance(msa, MSA) else msa.msas
        for value in values:
            # Evidence holds no PolymerSequence: it names one entity by ``entity_id`` and closes over the
            # primary system supplied here, so the Processor resolves the entity and checks its query width.
            entity = system.entities.get(value.entity_id)
            if not isinstance(entity, PolymerSequence):
                raise ValueError(f"MSA entity {value.entity_id!r} disagrees with the input system")
            width = len(value.unpaired_sequences[0])
            if width != len(entity.components):
                raise ValueError(
                    f"MSA for entity {value.entity_id!r} spans {width} columns but the entity has "
                    f"{len(entity.components)} components"
                )
        return msa
    path = Path(msa).expanduser()
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA is unambiguous only for one protein entity; pass MSACollection")
    records = read_fasta_records(path)
    if not records:
        raise ValueError("MSA FASTA contains no rows")
    # Project each row to the query axis (drop query-gap columns, fold ``.`` into ``-``) with all-zero
    # deletion counts; row 0 is the ungapped query.
    query = records[0].sequence.upper()
    keep = [index for index, symbol in enumerate(query) if symbol not in "-."]
    width = len(keep)
    projected: list[str] = []
    for record in records:
        sequence = record.sequence.upper()
        projected.append("".join("-" if sequence[index] in "-." else sequence[index] for index in keep))
    return MSA(
        entity_id=proteins[0],
        unpaired_sequences=tuple(projected),
        unpaired_deletion_counts=np.zeros((len(projected), width), dtype=np.uint32),
        paired_sequences=(projected[0],),
        paired_deletion_counts=np.zeros((1, width), dtype=np.uint32),
    )
