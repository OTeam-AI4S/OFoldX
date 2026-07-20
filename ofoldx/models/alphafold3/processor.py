r"""Lower canonical biomolecular inputs to the released DeepMind AlphaFold 3 graph.

This is the DeepMind AlphaFold 3 identity (distinct from the Protenix graph). The
Processor mirrors :class:`~ofoldx.models.protenix.processor.ProtenixProcessor` -- the
closest wired analog on the frozen evidence API -- but emits the AlphaFold 3 token,
MSA, and template tensors: the 31-class target restype vocabulary
(``POLYMER_TYPES_WITH_UNKNOWN_AND_GAP``), the 32-class MSA vocabulary
(``POLYMER_TYPES_WITH_ALL_UNKS_AND_GAP``), the ``msa_features`` / ``msa_mask`` stack
(one-hot + ``has_deletion`` + ``deletion_value``), and the pre-packed
``TemplateFeatures`` the released :class:`TemplateEmbedder` consumes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from multimolecule.io import read_fasta_records
from torch import Tensor
from typing_extensions import Self

from ...data.ccd import resolve_ccd_component
from ...data.components import get_component
from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...io.msa import read_msa
from ...io.structure import read_structure
from ...modules.embedding import TemplateFeatures
from ...references import classify_reference
from .config import AlphaFold3ProcessorConfig
from .model import AlphaFold3Input

# AlphaFold 3 target restype vocabulary (POLYMER_TYPES_WITH_UNKNOWN_AND_GAP, 31 classes):
# 20 standard amino acids, UNK, the gap, then A G C U (RNA) DA DG DC DT (DNA) and the shared
# unknown-nucleic N. This is the one-hot alphabet of ``create_target_feat`` and the MSA profile.
_TARGET_RESTYPE = {
    name: index
    for index, name in enumerate(
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
            "UNK",
            "-",
            "A",
            "G",
            "C",
            "U",
            "DA",
            "DG",
            "DC",
            "DT",
            "N",
        )
    )
}
_TARGET_VOCAB_SIZE = len(_TARGET_RESTYPE)  # 31
# The MSA one-hot alphabet (POLYMER_TYPES_WITH_ALL_UNKS_AND_GAP, 32 classes) extends the target
# vocabulary with a distinct unknown-DNA class DN, so ``create_msa_feat`` one-hots over 32.
_MSA_VOCAB_SIZE = _TARGET_VOCAB_SIZE + 1  # 32
_UNKNOWN = _TARGET_RESTYPE["UNK"]
_GAP = _TARGET_RESTYPE["-"]
_GLYCINE = _TARGET_RESTYPE["GLY"]
_STANDARD = {
    "protein": frozenset(tuple(_TARGET_RESTYPE)[:21]),
    "rna": frozenset(("A", "G", "C", "U", "N")),
    "dna": frozenset(("DA", "DG", "DC", "DT", "DN")),
}
# MSA cells are UPPERCASE one-letter symbols (or ``-``), not ChemicalComponent IDs. AlphaFold 3
# consumes MSA evidence only for proteins (mirroring the released Protenix gate), so this projects the
# twenty protein one-letter codes onto their target indices; ``-`` becomes the gap and every other
# symbol (ambiguity codes B Z U J O X, etc.) becomes UNK.
_MSA_RESTYPE = {
    symbol: _TARGET_RESTYPE[three]
    for symbol, three in {
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
_MSA_RESTYPE["-"] = _GAP
_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
    "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
    "Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBER = {element: index for index, element in enumerate(_ELEMENTS, start=1)}
_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".fna", ".seq"))
# Released AlphaFold 3 template distogram bins: 39 lower breaks over [3.25, 50.75] (squared).
_TEMPLATE_DISTOGRAM_MIN = 3.25
_TEMPLATE_DISTOGRAM_MAX = 50.75


def _restype_index(component_id: str, standard: bool) -> int:
    r"""Target restype index; unknown DNA folds onto the shared unknown-nucleic class N."""

    if not standard:
        return _UNKNOWN
    if component_id == "DN":
        component_id = "N"
    return _TARGET_RESTYPE.get(component_id, _UNKNOWN)


class AlphaFold3Processor:
    r"""Build the exact AlphaFold 3 atom, token, MSA, and template tensors."""

    config_class = AlphaFold3ProcessorConfig

    def __init__(
        self,
        config: AlphaFold3ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold3ProcessorConfig() if config is None else config
        if not 1 <= self.config.max_msa_rows <= 16384:
            raise ValueError("max_msa_rows must be between 1 and the released limit 16384")
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
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
    ) -> AlphaFold3Input:
        r"""Lower one canonical assembly and its MSA / template evidence."""

        system = _normalize_input(input)
        system = _resolve_components(system, revision=self.config.ccd_revision)
        normalized_msa = _normalize_msa(msa, system)
        tokens = _tokenize(system)
        return _features(
            system,
            tokens=tokens,
            msa=normalized_msa,
            templates=tuple(templates),
            max_msa_rows=self.config.max_msa_rows,
            device=self.device,
            dtype=self.dtype,
        )


def _normalize_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem:
    if isinstance(input, Structure):
        return input.system
    if isinstance(input, MolecularSystem):
        return input
    if isinstance(input, str | PathLike):
        classified = classify_reference(input)
        if isinstance(classified, Path):
            if any(suffix.lower() in _FASTA_SUFFIXES for suffix in classified.suffixes):
                sequences = tuple(
                    (
                        record.id or f"chain_{index + 1}",
                        PolymerSequence.from_one_letter(record.sequence.strip().upper()),
                    )
                    for index, record in enumerate(read_fasta_records(classified))
                )
                return _sequence_system(sequences)
            return read_structure(classified).system
        sequence = PolymerSequence.from_one_letter(classified.strip().upper())
        return _sequence_system((("protein", sequence),))
    if isinstance(input, PolymerSequence):
        return _sequence_system((("A", input),))
    if isinstance(input, Sequence):
        sequences = tuple(input)
        if not sequences or not all(isinstance(sequence, PolymerSequence) for sequence in sequences):
            raise TypeError("AlphaFold 3 sequence collections must contain PolymerSequence values")
        return _sequence_system(
            tuple(
                (f"chain_{index + 1}", sequence)
                for index, sequence in enumerate(cast(tuple[PolymerSequence, ...], sequences))
            )
        )
    raise TypeError(
        "AlphaFold 3 input must be an inline sequence, FASTA/structure path, PolymerSequence, "
        "sequence of PolymerSequence values, MolecularSystem, or Structure"
    )


def _sequence_system(named: tuple[tuple[str, PolymerSequence], ...]) -> MolecularSystem:
    used: set[str] = set()
    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    for base_id, sequence in named:
        entity_id = base_id
        suffix = 1
        while entity_id in used:
            suffix += 1
            entity_id = f"{base_id}_{suffix}"
        used.add(entity_id)
        entities[entity_id] = sequence
        instances[entity_id] = entity_id
    return MolecularSystem(entities, instances)


def _resolve_components(system: MolecularSystem, *, revision: str) -> MolecularSystem:
    r"""Embed CCD chemistry for any bare CCD-code component at the Processor boundary.

    Walks every entity's components and, for any component that is a bare CCD-code
    reference lacking the chemical graph the tokenizer requires (no ``atoms``, or no
    per-atom ``reference_coordinates`` -- the exact facts checked at
    ``_reference_features`` / ``_tokenize``), lowers it to a canonical
    :class:`~ofoldx.data.molecular.ChemicalComponent` VALUE via the shared
    ``ofoldx/ccd`` resolver and embeds that value in place. This covers CCD-code
    ligands, ions, modified residues, and glycan monomers.

    The resolution is boundary-time only: ``_features`` / ``_reference_features`` then
    read the embedded values, so the tensor-compute path performs zero live registry
    lookup (Principle 7). A component that is already fully resolved (an audited
    standard residue, an inline SMILES-imported molecule, or a structure-parsed ligand
    carrying atoms and coordinates) is left untouched -- the pass never re-resolves or
    overrides supplied chemistry. Identical CCD codes resolve once and share the value.
    """

    cache: dict[str, ChemicalComponent] = {}
    resolved_entities: dict[str, PolymerSequence | Molecule] = {}
    changed = False
    for entity_id, entity in system.entities.items():
        components: list[ChemicalComponent] = []
        entity_changed = False
        for component in entity.components:
            if _needs_ccd_resolution(component):
                resolved = cache.get(component.component_id)
                if resolved is None:
                    resolved = resolve_ccd_component(component.component_id, revision=revision)
                    cache[component.component_id] = resolved
                components.append(resolved)
                entity_changed = True
            else:
                components.append(component)
        if not entity_changed:
            resolved_entities[entity_id] = entity
            continue
        changed = True
        if isinstance(entity, PolymerSequence):
            resolved_entities[entity_id] = PolymerSequence(entity.polymer_type, tuple(components))
        else:
            resolved_entities[entity_id] = Molecule(tuple(components), entity.bonds)

    if not changed:
        return system
    return MolecularSystem(resolved_entities, system.instances, system.bonds)


def _needs_ccd_resolution(component: ChemicalComponent) -> bool:
    r"""Whether a component is a bare CCD-code reference missing tokenizer-required facts."""

    if not component.atoms:
        return True
    return not any(coordinate is not None for coordinate in component.reference_coordinates)


def _normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    if isinstance(msa, MSA | MSACollection):
        values = (msa,) if isinstance(msa, MSA) else msa.msas
        for value in values:
            entity = system.entities.get(value.entity_id)
            if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                raise ValueError("AlphaFold 3 consumes supplied MSA evidence only for proteins")
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
        raise ValueError("an aligned FASTA MSA is unambiguous only for one protein entity")
    return read_msa(path, entity_id=proteins[0])


def _is_standard(component: ChemicalComponent, polymer_type: str) -> bool:
    if component.component_id not in _STANDARD[polymer_type]:
        return False
    return component == get_component(component.component_id)


def _tokenize(system: MolecularSystem) -> list[dict[str, object]]:
    available = {site for site, atom in system.iter_atoms() if atom.element != "H"}
    tokens: list[dict[str, object]] = []
    entity_indices: dict[str, int] = {}
    entity_copies: dict[str, int] = {}
    reference_space = 0

    for chain_index, (instance_id, entity_id) in enumerate(system.instances.items()):
        entity = system.entities[entity_id]
        entity_index = entity_indices.setdefault(entity_id, len(entity_indices))
        symmetry_index = entity_copies.get(entity_id, 0)
        entity_copies[entity_id] = symmetry_index + 1
        if isinstance(entity, PolymerSequence):
            for monomer_index, component in enumerate(entity.components):
                component_id = component.component_id
                atom_sites = tuple(
                    site
                    for atom in component.atoms
                    if atom.element != "H"
                    and not (component_id in {"N", "DN"} and atom.atom_id == "N")
                    and (site := AtomSite(instance_id, monomer_index, atom.atom_id)) in available
                )
                if not atom_sites:
                    raise ValueError(
                        f"AlphaFold 3 component {component_id!r} at {instance_id}:{monomer_index} "
                        "has no model-visible heavy atoms"
                    )
                standard = _is_standard(component, entity.polymer_type)
                groups = tuple((site,) for site in atom_sites) if not standard else (atom_sites,)
                by_name = {site.atom_id: site for site in atom_sites}
                center_name = "CA" if entity.polymer_type == "protein" else "C1'"
                if entity.polymer_type == "protein":
                    representative_name = "CA" if component_id == "GLY" else "CB"
                elif component_id in {"A", "G", "DA", "DG"}:
                    representative_name = "C4"
                elif component_id in {"C", "U", "DC", "DT"}:
                    representative_name = "C2"
                else:
                    representative_name = "C1'"
                for group in groups:
                    atom_token = len(group) == 1 and not standard
                    center_site = group[0] if atom_token else by_name[center_name]
                    representative_site = group[0] if atom_token else by_name.get(representative_name, center_site)
                    tokens.append(
                        {
                            "atoms": group,
                            "site": group[0] if atom_token else (instance_id, monomer_index),
                            "instance_id": instance_id,
                            "entity_id": entity_id,
                            "monomer_index": monomer_index,
                            "residue_index": monomer_index,
                            "chain_index": chain_index,
                            "entity_index": entity_index,
                            "symmetry_index": symmetry_index,
                            "restype": _restype_index(component_id, standard),
                            "center": center_site,
                            "representative": representative_site,
                            "standard": standard,
                            "reference_space": reference_space,
                        }
                    )
                reference_space += 1
        else:
            atom_sites = tuple(
                site
                for component_index, component in enumerate(entity.components)
                for atom in component.atoms
                if atom.element != "H" and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available
            )
            if not atom_sites:
                raise ValueError(f"AlphaFold 3 molecule {instance_id!r} has no model-visible heavy atoms")
            for site in atom_sites:
                tokens.append(
                    {
                        "atoms": (site,),
                        "site": site,
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": 0,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "restype": _UNKNOWN,
                        "center": site,
                        "representative": site,
                        "standard": False,
                        "reference_space": reference_space,
                    }
                )
            reference_space += 1
    return tokens


def _features(
    system: MolecularSystem,
    *,
    tokens: list[dict[str, object]],
    msa: MSA | MSACollection | None,
    templates: tuple[Template, ...],
    max_msa_rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold3Input:
    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atoms"]))
    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    site_to_token = {
        site: token_index
        for token_index, token in enumerate(tokens)
        for site in cast(tuple[AtomSite, ...], token["atoms"])
    }
    atom_to_token = torch.tensor([site_to_token[site] for site in atom_sites], dtype=torch.long)
    reference_positions, reference_mask, reference_charges = _reference_features(system, atom_sites, site_to_atom)
    atom_names = tuple(site.atom_id for site in atom_sites)
    element_indices = torch.tensor([_ATOMIC_NUMBER[system.atom(site).element] for site in atom_sites])
    name_indices = torch.tensor(
        [[min(63, max(0, ord(character) - 32)) for character in name.ljust(4)[:4]] for name in atom_names],
        dtype=torch.long,
    )
    atom_features = torch.cat(
        (
            reference_mask[:, None].float(),
            F.one_hot(element_indices, num_classes=128).float(),
            F.one_hot(name_indices, num_classes=64).reshape(len(atom_sites), 256).float(),
        ),
        dim=-1,
    )
    atom_token_index = _atom_token_offsets(tokens, system, atom_sites, site_to_atom)

    profile, deletion_mean = _profile_features(msa, tokens, system)
    restype = torch.tensor([cast(int, token["restype"]) for token in tokens])
    token_features = torch.cat(
        (
            torch.zeros(len(tokens), 384),
            F.one_hot(restype, num_classes=_TARGET_VOCAB_SIZE).float(),
            profile,
            deletion_mean[:, None],
        ),
        dim=-1,
    )

    msa_features, msa_mask = _msa_features(msa, tokens, system, max_rows=max_msa_rows)
    template_features = _template_features(templates, tokens, system)
    token_bonds = _token_bonds(system, site_to_token, len(tokens))
    token_to_rep_atom = torch.tensor(
        [site_to_atom[cast(AtomSite, token["representative"])] for token in tokens], dtype=torch.long
    )

    def floating(value: Tensor) -> Tensor:
        return value.to(device=device, dtype=dtype).unsqueeze(0)

    def integer(value: Tensor) -> Tensor:
        return value.to(device=device).unsqueeze(0)

    return AlphaFold3Input(
        token_mask=integer(torch.ones(len(tokens), dtype=torch.bool)),
        residue_index=integer(torch.tensor([cast(int, token["residue_index"]) for token in tokens])),
        chain_index=integer(torch.tensor([cast(int, token["chain_index"]) for token in tokens])),
        entity_index=integer(torch.tensor([cast(int, token["entity_index"]) for token in tokens])),
        symmetry_index=integer(torch.tensor([cast(int, token["symmetry_index"]) for token in tokens])),
        token_index=integer(torch.arange(len(tokens))),
        atom_features=floating(atom_features),
        atom_to_token=integer(atom_to_token),
        atom_mask=integer(torch.ones(len(atom_sites), dtype=torch.bool)),
        reference_positions=floating(reference_positions),
        reference_space_id=integer(
            torch.tensor(
                [cast(int, tokens[site_to_token[site]]["reference_space"]) for site in atom_sites],
                dtype=torch.long,
            )
        ),
        atom_token_index=integer(atom_token_index),
        token_to_rep_atom=integer(token_to_rep_atom),
        token_features=floating(token_features),
        reference_mask=integer(reference_mask),
        reference_charges=floating(reference_charges[:, None]),
        token_bonds=floating(token_bonds),
        msa_features=None if msa_features is None else floating(msa_features),
        msa_mask=None if msa_mask is None else floating(msa_mask),
        template_features=(
            None if template_features is None else _template_to(template_features, device=device, dtype=dtype)
        ),
        system=system,
        atom_sites=atom_sites,
        token_sites=tuple(cast(AtomSite | ComponentReference, token["site"]) for token in tokens),
    )


def _atom_token_offsets(
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    atom_sites: tuple[AtomSite, ...],
    site_to_atom: dict[AtomSite, int],
) -> Tensor:
    atom_token_index = torch.empty(len(atom_sites), dtype=torch.long)
    for token in tokens:
        group = cast(tuple[AtomSite, ...], token["atoms"])
        if not cast(bool, token["standard"]):
            for site in group:
                atom_token_index[site_to_atom[site]] = 0
            continue
        instance_id = cast(str, token["instance_id"])
        entity = cast(PolymerSequence, system.entities[system.instances[instance_id]])
        component = entity.components[cast(int, token["monomer_index"])]
        if component.component_id == "N":
            heavy_order = {
                name: index
                for index, name in enumerate(
                    ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
                )
            }
        elif component.component_id == "DN":
            heavy_order = {
                name: index
                for index, name in enumerate(
                    ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
                )
            }
        else:
            heavy_order = {
                atom.atom_id: index
                for index, atom in enumerate(atom for atom in component.atoms if atom.element != "H")
            }
        for site in group:
            atom_token_index[site_to_atom[site]] = heavy_order[site.atom_id]
    return atom_token_index


def _reference_features(
    system: MolecularSystem,
    atom_sites: tuple[AtomSite, ...],
    site_to_atom: dict[AtomSite, int],
) -> tuple[Tensor, Tensor, Tensor]:
    positions = torch.zeros(len(atom_sites), 3)
    mask = torch.zeros(len(atom_sites), dtype=torch.bool)
    charges = torch.tensor([system.atom(site).formal_charge for site in atom_sites], dtype=torch.float32)
    grouped: dict[tuple[str, int | None], list[AtomSite]] = {}
    for site in atom_sites:
        entity = system.entities[system.instances[site.instance_id]]
        space = None if isinstance(entity, Molecule) else site.component_index
        grouped.setdefault((site.instance_id, space), []).append(site)

    for (instance_id, _space), sites in grouped.items():
        entity = system.entities[system.instances[instance_id]]
        checked: set[int] = set()
        for site in sites:
            component = entity.components[site.component_index]
            if site.component_index not in checked:
                checked.add(site.component_index)
                if not any(coordinate is not None for coordinate in component.reference_coordinates):
                    raise ValueError(
                        f"AlphaFold 3 requires explicit reference_coordinates for {component.component_id!r}; "
                        "the Processor does not generate conformers implicitly"
                    )
            atom_by_name = {atom.atom_id: index for index, atom in enumerate(component.atoms)}
            global_index = site_to_atom[site]
            coordinate = component.reference_coordinates[atom_by_name[site.atom_id]]
            if coordinate is not None:
                positions[global_index] = torch.tensor(coordinate)
                mask[global_index] = True
        indices = torch.tensor([site_to_atom[site] for site in sites])
        positions[indices] -= positions[indices].mean(dim=0)
    return positions, mask, charges


def _msa_values(msa: MSA | MSACollection | None) -> tuple[MSA, ...]:
    if msa is None:
        return ()
    return (msa,) if isinstance(msa, MSA) else msa.msas


def _msa_polymer_type(system: MolecularSystem, entity_id: str) -> str | None:
    entity = system.entities.get(entity_id)
    return entity.polymer_type if isinstance(entity, PolymerSequence) else None


def _msa_descriptors(
    msa: MSA | MSACollection | None,
    *,
    max_rows: int,
) -> list[tuple[dict[str, tuple[int, bool]], bool]]:
    # Each descriptor maps a column's entity_id to ``(source_row, is_paired)``. The bool is the query
    # flag. This is the released Protenix descriptor plan, read straight off the frozen dense arrays.
    if msa is None:
        return [({}, True)]
    if isinstance(msa, MSA):
        return [({msa.entity_id: (row, False)}, row == 0) for row in range(min(len(msa.unpaired_sequences), max_rows))]

    row_pairing = msa.row_pairing
    row_is_paired = msa.row_is_paired
    entity_ids = tuple(value.entity_id for value in msa.msas)
    plan_rows = int(row_pairing.shape[0])

    is_query = np.all(row_pairing == 0, axis=1)
    populated = np.count_nonzero(row_pairing != -1, axis=1)
    paired = [row for row in range(plan_rows) if not is_query[row] and populated[row] >= 2]
    unpaired = [row for row in range(plan_rows) if not is_query[row] and populated[row] < 2]

    pair_limit = max_rows // 2
    ordered: list[int | None] = [None]
    ordered.extend(paired[: max(0, pair_limit - 1)])
    ordered.extend(unpaired[: max_rows - len(ordered)])
    ordered = ordered[:max_rows]

    descriptors: list[tuple[dict[str, tuple[int, bool]], bool]] = []
    for row in ordered:
        if row is None:
            descriptors.append(({}, True))
            continue
        mapping: dict[str, tuple[int, bool]] = {}
        for column, entity_id in enumerate(entity_ids):
            source = int(row_pairing[row, column])
            if source == -1:
                continue
            mapping[entity_id] = (source, bool(row_is_paired[row, column]))
        descriptors.append((mapping, False))
    return descriptors


def _msa_rows_and_deletions(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> tuple[Tensor, Tensor]:
    values = _msa_values(msa)
    by_entity = {value.entity_id: value for value in values}
    descriptors = _msa_descriptors(msa, max_rows=max_rows)
    rows = torch.empty(len(descriptors), len(tokens), dtype=torch.long)
    deletions = torch.zeros(len(descriptors), len(tokens))

    for row_index, (descriptor, query) in enumerate(descriptors):
        for token_index, token in enumerate(tokens):
            entity_id = cast(str, token["entity_id"])
            monomer_index = cast(int | None, token["monomer_index"])
            value = by_entity.get(entity_id)
            if query or value is None or _msa_polymer_type(system, entity_id) != "protein":
                rows[row_index, token_index] = cast(int, token["restype"])
            elif entity_id not in descriptor:
                rows[row_index, token_index] = _GAP
            else:
                source_row, is_paired = descriptor[entity_id]
                pool = value.paired_sequences if is_paired else value.unpaired_sequences
                deletion_pool = value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts
                symbol = pool[source_row][cast(int, monomer_index)]
                rows[row_index, token_index] = _MSA_RESTYPE.get(symbol, _UNKNOWN)
                deletions[row_index, token_index] = float(deletion_pool[source_row, cast(int, monomer_index)])
    return rows, deletions


def _msa_features(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> tuple[Tensor | None, Tensor | None]:
    r"""Build the AlphaFold 3 ``msa_features`` = one-hot(32) + has_deletion + deletion_value."""

    if msa is None:
        return None, None
    rows, deletions = _msa_rows_and_deletions(msa, tokens, system, max_rows=max_rows)
    has_deletion = deletions.clamp(0.0, 1.0)
    deletion_value = torch.atan(deletions / 3.0) * (2.0 / math.pi)
    msa_one_hot = F.one_hot(rows, num_classes=_MSA_VOCAB_SIZE).float()
    features = torch.cat((msa_one_hot, has_deletion[..., None], deletion_value[..., None]), dim=-1)
    mask = torch.ones(rows.shape, dtype=torch.float32)
    return features, mask


def _profile_features(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> tuple[Tensor, Tensor]:
    r"""The target-feat MSA profile (31-class) and per-token mean deletion over the unpaired pool."""

    values = _msa_values(msa)
    by_entity = {value.entity_id: value for value in values}
    profile = torch.zeros(len(tokens), _TARGET_VOCAB_SIZE)
    deletion_mean = torch.zeros(len(tokens))
    for token_index, token in enumerate(tokens):
        entity_id = cast(str, token["entity_id"])
        monomer_index = cast(int | None, token["monomer_index"])
        value = by_entity.get(entity_id)
        if value is None or _msa_polymer_type(system, entity_id) != "protein":
            profile[token_index, cast(int, token["restype"])] = 1
            continue
        column = cast(int, monomer_index)
        encoded = [_MSA_RESTYPE.get(sequence[column], _UNKNOWN) for sequence in value.unpaired_sequences]
        profile[token_index] = F.one_hot(torch.tensor(encoded), num_classes=_TARGET_VOCAB_SIZE).float().mean(dim=0)
        deletion_mean[token_index] = torch.tensor(value.unpaired_deletion_counts[:, column], dtype=torch.float32).mean()
    return profile, deletion_mean


def _token_bonds(system: MolecularSystem, site_to_token: dict[AtomSite, int], token_count: int) -> Tensor:
    feature = torch.zeros(token_count, token_count)
    for bond in system.iter_bonds():
        first, second = bond.site_a, bond.site_b
        if first not in site_to_token or second not in site_to_token:
            continue
        overlay = bond.origin not in ("component", "molecule", "polymer_rule")
        first_entity = system.entities[system.instances[first.instance_id]]
        second_entity = system.entities[system.instances[second.instance_id]]
        keep = isinstance(first_entity, Molecule) or isinstance(second_entity, Molecule)
        if isinstance(first_entity, PolymerSequence) and isinstance(second_entity, PolymerSequence):
            if first.instance_id == second.instance_id and first.component_index == second.component_index:
                component = first_entity.components[first.component_index]
                keep = not _is_standard(component, first_entity.polymer_type)
            elif overlay and (
                first.instance_id != second.instance_id or abs(first.component_index - second.component_index) > 1
            ):
                keep = True
            else:
                keep = False
        if keep:
            first_token, second_token = site_to_token[first], site_to_token[second]
            feature[first_token, second_token] = 1
            feature[second_token, first_token] = 1
    return feature


def _template_to(features: TemplateFeatures, *, device: torch.device, dtype: torch.dtype) -> TemplateFeatures:
    return TemplateFeatures(
        distogram=features.distogram.to(device=device, dtype=dtype),
        pseudo_beta_mask=features.pseudo_beta_mask.to(device=device, dtype=dtype),
        residue_type=features.residue_type.to(device=device),
        unit_vector=features.unit_vector.to(device=device, dtype=dtype),
        backbone_frame_mask=features.backbone_frame_mask.to(device=device, dtype=dtype),
        template_mask=(
            None if features.template_mask is None else features.template_mask.to(device=device, dtype=dtype)
        ),
    )


def _template_backbone(positions: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Pseudo-beta positions/mask and per-pair unit vectors in each residue's backbone frame.

    ``positions`` / ``mask`` are ``[N, 4, *]`` over the dense backbone atoms (N, CA, C, CB); glycine
    carries no CB so its pseudo-beta falls back to CA (index 1). Frames follow the released
    ``make_backbone_rigid`` convention: x-axis along ``C - CA`` with ``N - CA`` in the xy-plane.
    """

    nitrogen, alpha, carbon, beta = positions.unbind(dim=1)
    nitrogen_mask, alpha_mask, carbon_mask, beta_mask = mask.unbind(dim=1)

    is_glycine = beta_mask == 0
    pseudo_beta = torch.where(is_glycine[:, None], alpha, beta)
    pseudo_beta_mask = torch.where(is_glycine, alpha_mask, beta_mask)

    def _normalize(vector: Tensor) -> Tensor:
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    x_axis = _normalize(carbon - alpha)
    reference = nitrogen - alpha
    reference = reference - (reference * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = _normalize(reference)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)  # columns are the frame axes

    backbone_mask = nitrogen_mask * alpha_mask * carbon_mask
    difference = alpha[None, :, :] - alpha[:, None, :]  # CA_j - CA_i
    local = torch.einsum("inm,ijn->ijm", rotation, difference)  # R_i^T (CA_j - CA_i)
    unit_vector = local / local.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask


def _template_distogram(pseudo_beta: Tensor, pseudo_beta_mask_2d: Tensor, num_bins: int) -> Tensor:
    lower = torch.linspace(_TEMPLATE_DISTOGRAM_MIN, _TEMPLATE_DISTOGRAM_MAX, num_bins) ** 2
    upper = torch.cat((lower[1:], lower.new_tensor([1e8])))
    squared = ((pseudo_beta[:, None, :] - pseudo_beta[None, :, :]) ** 2).sum(dim=-1, keepdim=True)
    dgram = ((squared > lower) & (squared < upper)).float()
    return dgram * pseudo_beta_mask_2d[..., None]


def _template_features(
    templates: tuple[Template, ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    num_distogram_bins: int = 39,
) -> TemplateFeatures | None:
    if not templates:
        return None

    token_count = len(tokens)
    # Resolve query token indices for a polymer residue named by (instance_id | entity_id, component).
    instance_residue: dict[tuple[str, int], int] = {}
    entity_residue: dict[tuple[str, int], list[int]] = {}
    for token_index, token in enumerate(tokens):
        monomer_index = cast(int | None, token["monomer_index"])
        if monomer_index is None:
            continue
        instance_residue[(cast(str, token["instance_id"]), monomer_index)] = token_index
        entity_residue.setdefault((cast(str, token["entity_id"]), monomer_index), []).append(token_index)

    per_template_positions: list[Tensor] = []
    per_template_mask: list[Tensor] = []
    per_template_restype: list[Tensor] = []
    for template in templates:
        positions = torch.zeros(token_count, 4, 3)
        atom_mask = torch.zeros(token_count, 4)
        restype = torch.full((token_count,), float(_GAP)).to(torch.long)
        coordinates = _structure_atom_lookup(template.structure)
        for alignment in template.alignments:
            template_entity_id = template.structure.system.instances[alignment.template_instance_id]
            template_entity = cast(PolymerSequence, template.structure.system.entities[template_entity_id])
            for target_component, template_component in alignment.residue_pairs:
                targets = _alignment_targets(alignment, target_component, instance_residue, entity_residue)
                if not targets:
                    continue
                component = template_entity.components[template_component]
                backbone, backbone_mask = _residue_backbone(
                    coordinates, alignment.template_instance_id, template_component, component
                )
                residue_index = _restype_index(
                    component.component_id, _is_standard(component, template_entity.polymer_type)
                )
                for target_token in targets:
                    positions[target_token] = backbone
                    atom_mask[target_token] = backbone_mask
                    restype[target_token] = residue_index
        per_template_positions.append(positions)
        per_template_mask.append(atom_mask)
        per_template_restype.append(restype)

    distograms: list[Tensor] = []
    pseudo_beta_masks: list[Tensor] = []
    unit_vectors: list[Tensor] = []
    backbone_masks: list[Tensor] = []
    for positions, atom_mask in zip(per_template_positions, per_template_mask):
        pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask = _template_backbone(positions, atom_mask)
        pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        distograms.append(_template_distogram(pseudo_beta, pseudo_beta_mask_2d, num_distogram_bins))
        pseudo_beta_masks.append(pseudo_beta_mask_2d)
        unit_vectors.append(unit_vector * backbone_mask_2d[..., None])
        backbone_masks.append(backbone_mask_2d)

    return TemplateFeatures(
        distogram=torch.stack(distograms).unsqueeze(0),
        pseudo_beta_mask=torch.stack(pseudo_beta_masks).unsqueeze(0),
        residue_type=torch.stack(per_template_restype).unsqueeze(0),
        unit_vector=torch.stack(unit_vectors).unsqueeze(0),
        backbone_frame_mask=torch.stack(backbone_masks).unsqueeze(0),
        template_mask=torch.ones(1, len(templates)),
    )


def _alignment_targets(
    alignment: object,
    target_component: int,
    instance_residue: dict[tuple[str, int], int],
    entity_residue: dict[tuple[str, int], list[int]],
) -> list[int]:
    target_instance_id = cast("str | None", getattr(alignment, "target_instance_id"))
    target_entity_id = cast(str, getattr(alignment, "target_entity_id"))
    if target_instance_id is not None:
        token_index = instance_residue.get((target_instance_id, target_component))
        return [] if token_index is None else [token_index]
    return list(entity_residue.get((target_entity_id, target_component), ()))


def _structure_atom_lookup(structure: Structure) -> dict[tuple[str, int, str], tuple[np.ndarray, bool]]:
    lookup: dict[tuple[str, int, str], tuple[np.ndarray, bool]] = {}
    for index, site in enumerate(structure.sites):
        lookup[(site.instance_id, site.component_index, site.atom_id)] = (
            structure.coordinates[index],
            bool(structure.coordinate_mask[index]),
        )
    return lookup


def _residue_backbone(
    coordinates: dict[tuple[str, int, str], tuple[np.ndarray, bool]],
    instance_id: str,
    component_index: int,
    component: ChemicalComponent,
) -> tuple[Tensor, Tensor]:
    r"""Return dense (N, CA, C, CB) positions and their validity mask for one template residue."""

    positions = torch.zeros(4, 3)
    mask = torch.zeros(4)
    for slot, atom_id in enumerate(("N", "CA", "C", "CB")):
        entry = coordinates.get((instance_id, component_index, atom_id))
        if entry is None:
            continue
        coordinate, present = entry
        if present:
            positions[slot] = torch.from_numpy(np.asarray(coordinate, dtype=np.float32))
            mask[slot] = 1.0
    return positions, mask
