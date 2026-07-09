r"""
Official Protenix inference JSON to structure conversion.

Ports the upstream inference data pipeline entry point (``protenix.data.inference.json_to_feature``
``SampleDictToFeatures.build_full_atom_array`` together with ``json_parser.build_polymer`` and
``add_entity_atom_array``): it materializes a [`Structure`][ofoldx.data.structure.Structure] directly
from the official ``sequences`` section of a Protenix job JSON (protein / DNA / RNA sequence strings,
CCD-code modifications, ligand CCD codes, chain counts and explicit ``id`` lists, and inter-entity
``covalent_bonds``). Downstream featurization (tokenization, reference features, ``ref_space_uid`` /
``distogram_rep_atom_mask`` equivalents, token frames, ``is_resolved``, and the constraint bundle) is
handled by the existing structure processor, so this module owns only the file-shaped JSON to
``Structure`` step.

Heavy atoms and their reference-conformer geometry come from the configured
[`CcdProvider`][ofoldx.data.ccd.CcdProvider]; the reference conformer supplies the coordinates for a
sequence-only input (there are no experimental coordinates), matching upstream ``add_reference_features``.
Leaving-atom trimming at internal residues (protein ``OXT``, nucleic ``OP3``) reuses the shared
[`heavy_atom_names`][ofoldx.data.ccd.heavy_atom_names] / ``LEAVING_ATOMS`` reference so the emitted
atom set matches the file-based path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...data.ccd import CcdProvider, default_ccd_provider, heavy_atom_names
from ...data.residue_constants import dna_1to3, restype_1to3_with_unknown, rna_1to3
from ...data.structure import AtomRecord, ResidueKey, Structure
from ...data.utils import index_to_chain_id

# Entity-key to molecule type, mirroring upstream ``entity_type_mapping_dict``.
_POLYMER_KEYS = frozenset({"proteinChain", "dnaSequence", "rnaSequence"})
_LIGAND_KEYS = frozenset({"ligand", "ion"})


def structure_from_protenix_json(
    sample: Mapping[str, Any],
    *,
    ccd: CcdProvider | None = None,
) -> Structure:
    r"""
    Build a [`Structure`][ofoldx.data.structure.Structure] from an official Protenix job JSON dict.

    The input dict follows the official inference schema: a ``sequences`` list of single-key entity
    dicts (``proteinChain`` / ``dnaSequence`` / ``rnaSequence`` / ``ligand`` / ``ion``), each with a
    ``sequence`` (polymer one-letter string, or CCD-code list / string for ligands), a ``count``, and
    an optional explicit ``id`` chain-label list. Optional ``modifications`` replace individual polymer
    residues with ``CCD_<code>`` components. An optional top-level ``covalent_bonds`` list adds explicit
    inter-entity bonds. MSE residues are converted to MET, matching upstream ``mse_to_met``.

    Args:
        sample: Parsed official job JSON. Only ``sequences`` and ``covalent_bonds`` are read.
        ccd: Reference component provider. Defaults to the process-wide provider.

    Returns:
        Structure with reference-conformer heavy-atom coordinates and inter-entity covalent bonds.

    Raises:
        ValueError: If ``sequences`` is missing/empty, an entity type is unknown, a chain-``id`` list is
            malformed, a residue code is unknown, or a component is absent from the CCD provider.

    Examples:
        The [`CcdProvider`][ofoldx.data.ccd.CcdProvider] protocol is just ``get(name)``, so a tiny
        in-memory provider is enough to parse a one-residue chain:

        >>> import numpy as np
        >>> from ofoldx.data import ReferenceComponent
        >>> gly = ReferenceComponent(
        ...     atom_map={"N": 0, "CA": 1, "C": 2, "O": 3},
        ...     coords=np.zeros((4, 3), "f4"), mask=np.ones(4, bool),
        ...     charge=np.zeros(4, "f4"), bonds=(), elements=("N", "C", "C", "O"))
        >>> class Ccd:
        ...     def get(self, name):
        ...         return gly
        >>> sample = {"sequences": [{"proteinChain": {"sequence": "G", "count": 1}}]}
        >>> structure = structure_from_protenix_json(sample, ccd=Ccd())
        >>> len(structure.atoms)
        4
        >>> structure_from_protenix_json({})
        Traceback (most recent call last):
            ...
        ValueError: official Protenix JSON must contain a non-empty 'sequences' list
    """

    provider = ccd if ccd is not None else default_ccd_provider()
    sequences = sample.get("sequences")
    if not sequences:
        raise ValueError("official Protenix JSON must contain a non-empty 'sequences' list")

    atoms: list[AtomRecord] = []
    # residue -> {atom_name: global atom index}, used to resolve covalent_bonds selectors.
    residue_atom_index: dict[ResidueKey, dict[str, int]] = {}
    # (entity, copy) -> chain label, used by covalent_bonds resolution.
    chain_labels: dict[tuple[int, int], str] = {}
    used_chain_ids = _validate_and_collect_chain_ids(sequences)
    auto_chain_index = 0

    for entity_zero_based, type_to_entity in enumerate(sequences):
        if len(type_to_entity) != 1:
            raise ValueError("each 'sequences' entry must hold exactly one entity type")
        ((entity_type, entity),) = type_to_entity.items()
        entity_id = entity_zero_based + 1
        explicit_ids = entity.get("id")
        count = int(entity.get("count", 1))
        residue_specs = _entity_residue_specs(entity_type, entity)
        for copy_number in range(1, count + 1):
            if explicit_ids:
                chain_label = str(explicit_ids[copy_number - 1])
            else:
                chain_label, auto_chain_index = _next_auto_chain_id(auto_chain_index, used_chain_ids)
            chain_labels[(entity_id, copy_number)] = chain_label
            _emit_chain(
                atoms,
                residue_atom_index,
                chain_label=chain_label,
                residue_specs=residue_specs,
                is_polymer=entity_type in _POLYMER_KEYS,
                provider=provider,
            )

    bonds = _covalent_bonds(sample, sequences, chain_labels, residue_atom_index)
    return Structure(tuple(atoms), bonds=bonds)


def _entity_residue_specs(entity_type: str, entity: Mapping[str, Any]) -> list[str]:
    r"""
    Return the ordered CCD component names for one entity, applying polymer modifications.
    """

    if entity_type == "proteinChain":
        codes = [_map_residue(restype_1to3_with_unknown, letter, "protein") for letter in entity["sequence"]]
        _apply_modifications(codes, entity.get("modifications"), position_key="ptmPosition", type_key="ptmType")
        return codes
    if entity_type in {"dnaSequence", "rnaSequence"}:
        table = dna_1to3 if entity_type == "dnaSequence" else rna_1to3
        codes = [_map_residue(table, letter, entity_type) for letter in entity["sequence"]]
        _apply_modifications(
            codes, entity.get("modifications"), position_key="basePosition", type_key="modificationType"
        )
        return codes
    if entity_type in _LIGAND_KEYS:
        sequence = entity["sequence"]
        codes = list(sequence) if isinstance(sequence, (list, tuple)) else [str(sequence)]
        return [str(code) for code in codes]
    raise ValueError(
        f"unknown entity type {entity_type!r}; expected one of proteinChain/dnaSequence/rnaSequence/ligand/ion"
    )


def _map_residue(table: Mapping[str, str], letter: str, kind: str) -> str:
    r"""
    Look up the CCD code for a polymer one-letter residue, raising on an unknown letter.

    Args:
        table: One-letter to CCD-code map for the polymer kind.
        letter: One-letter residue code.
        kind: Polymer kind label used only in the error message.

    Returns:
        The three-letter CCD residue code.

    Raises:
        ValueError: If the letter is not in the table.
    """

    code = table.get(letter)
    if code is None:
        raise ValueError(f"unknown {kind} residue letter {letter!r}")
    return code


def _apply_modifications(
    codes: list[str],
    modifications: Sequence[Mapping[str, Any]] | None,
    *,
    position_key: str,
    type_key: str,
) -> None:
    r"""
    Replace polymer residues in-place with the CCD component named by each ``CCD_<code>`` modification.
    """

    if not modifications:
        return
    for modification in modifications:
        index = int(modification[position_key]) - 1
        modification_type = str(modification[type_key])
        if not modification_type.startswith("CCD_"):
            raise ValueError(f"unsupported modification type {modification_type!r}; expected 'CCD_<code>'")
        codes[index] = modification_type[4:]


def _emit_chain(
    atoms: list[AtomRecord],
    residue_atom_index: dict[ResidueKey, dict[str, int]],
    *,
    chain_label: str,
    residue_specs: Sequence[str],
    is_polymer: bool,
    provider: CcdProvider,
) -> None:
    r"""
    Append every residue of one asym chain, using CCD reference conformers for heavy atoms.
    """

    num_residues = len(residue_specs)
    for residue_number, residue_name in enumerate(residue_specs, start=1):
        component = provider.get(residue_name)
        if component is None:
            raise ValueError(f"component {residue_name!r} is absent from the CCD provider")
        residue = ResidueKey(chain_id=chain_label, residue_number=residue_number)
        terminal = residue_number in (1, num_residues)
        mol_type = _residue_mol_type(residue_name, is_polymer)
        names = residue_atom_index.setdefault(residue, {})
        for name in heavy_atom_names(component, mol_type=mol_type, keep_leaving=terminal or not is_polymer):
            atom_index = component.atom_map[name]
            element = component.elements[atom_index].upper()
            atom_name, element = _mse_to_met(residue_name, name, element)
            names[atom_name] = len(atoms)
            x, y, z = (float(value) for value in component.coords[atom_index])
            atoms.append(
                AtomRecord(
                    name=atom_name,
                    element=element,
                    residue_name="MET" if residue_name == "MSE" else residue_name,
                    residue=residue,
                    x=x,
                    y=y,
                    z=z,
                    charge=float(component.charge[atom_index]) if atom_index < len(component.charge) else 0.0,
                    hetero=not is_polymer,
                    mol_type=mol_type,
                )
            )


def _residue_mol_type(residue_name: str, is_polymer: bool) -> str | None:
    r"""
    Classify a residue's molecule type (``protein`` / ``rna`` / ``dna`` / ``ligand``).

    Args:
        residue_name: Three-letter CCD residue name.
        is_polymer: Whether the residue belongs to a polymer entity.

    Returns:
        The molecule type; ``"ligand"`` for non-polymer entities.
    """

    if not is_polymer:
        return "ligand"
    if residue_name in dna_1to3.values():
        return "dna"
    if residue_name in {"A", "G", "C", "U", "N", "I"}:
        return "rna"
    return "protein"


def _mse_to_met(residue_name: str, atom_name: str, element: str) -> tuple[str, str]:
    r"""
    Rewrite the selenium ``SE`` atom of an MSE residue to sulfur ``SD`` (upstream ``mse_to_met``).
    """

    if residue_name == "MSE" and atom_name == "SE":
        return "SD", "S"
    return atom_name, element


def _validate_and_collect_chain_ids(sequences: Sequence[Mapping[str, Any]]) -> set[str]:
    r"""
    Validate explicit chain-``id`` lists and return the set of reserved labels (upstream check).
    """

    used: set[str] = set()
    for position, type_to_entity in enumerate(sequences):
        for entity in type_to_entity.values():
            ids = entity.get("id")
            if ids is None:
                continue
            if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
                raise ValueError(f"invalid 'id' field in sequences[{position}]: expected list[str]")
            if len(ids) != int(entity.get("count", 1)):
                raise ValueError(f"invalid 'id' field in sequences[{position}]: length must equal count")
            if len(set(ids)) != len(ids) or set(ids) & used:
                raise ValueError(f"invalid 'id' field in sequences[{position}]: duplicated chain IDs")
            used.update(ids)
    return used


def _next_auto_chain_id(auto_chain_index: int, used_chain_ids: set[str]) -> tuple[str, int]:
    r"""
    Return the next spreadsheet-style chain label not already reserved by an explicit ``id`` list.
    """

    while True:
        candidate = index_to_chain_id(auto_chain_index + 1)
        auto_chain_index += 1
        if candidate not in used_chain_ids:
            used_chain_ids.add(candidate)
            return candidate, auto_chain_index


def _covalent_bonds(
    sample: Mapping[str, Any],
    sequences: Sequence[Mapping[str, Any]],
    chain_labels: dict[tuple[int, int], str],
    residue_atom_index: dict[ResidueKey, dict[str, int]],
) -> tuple[tuple[int, int], ...]:
    r"""
    Resolve official ``covalent_bonds`` selectors to index pairs into the emitted atom list.

    Each bond names two atoms by ``entity`` / ``copy`` / ``position`` / ``atom`` (upstream long-form
    ``left_*`` / ``right_*`` keys are also accepted). When a copy is omitted the bond is added to every
    copy of the entity, matching upstream ``add_bonds_between_entities``. Integer atom identifiers are
    resolved through the entity ``atom_map_to_atom_name`` table.
    """

    bond_specs = sample.get("covalent_bonds")
    if not bond_specs:
        return ()
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for bond in bond_specs:
        left = _bond_side(bond, sequences, "left", "1")
        right = _bond_side(bond, sequences, "right", "2")
        copies = _bond_copies(left, right, chain_labels)
        for left_copy, right_copy in copies:
            first = _bond_atom_index(chain_labels, residue_atom_index, left, left_copy)
            second = _bond_atom_index(chain_labels, residue_atom_index, right, right_copy)
            key = (first, second) if first < second else (second, first)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return tuple(pairs)


def _bond_side(
    bond: Mapping[str, Any],
    sequences: Sequence[Mapping[str, Any]],
    long_prefix: str,
    short_suffix: str,
) -> dict[str, Any]:
    r"""
    Normalize one covalent-bond endpoint to entity/copy/position/atom-name fields.
    """

    entity = int(_bond_field(bond, long_prefix, short_suffix, "entity"))
    copy_value = _bond_field(bond, long_prefix, short_suffix, "copy")
    position = int(_bond_field(bond, long_prefix, short_suffix, "position"))
    atom = _bond_field(bond, long_prefix, short_suffix, "atom")
    if isinstance(atom, str) and atom.isdigit():
        atom = int(atom)
    if isinstance(atom, int):
        atom = _atom_map_to_name(sequences, entity, atom)
    return {
        "entity": entity,
        "copy": None if copy_value is None else int(copy_value),
        "position": position,
        "atom": str(atom),
    }


def _bond_field(bond: Mapping[str, Any], long_prefix: str, short_suffix: str, name: str) -> Any:
    r"""
    Read one covalent-bond endpoint field, accepting both the long-form and short-form key spellings.

    Args:
        bond: Covalent-bond spec dict.
        long_prefix: Endpoint prefix for the long form (e.g. ``"left"`` -> ``"left_entity"``).
        short_suffix: Endpoint suffix for the short form (e.g. ``"1"`` -> ``"entity1"``).
        name: Field name (``"entity"`` / ``"copy"`` / ``"position"`` / ``"atom"``).

    Returns:
        The field value, or ``None`` when neither key is present.
    """

    if f"{long_prefix}_{name}" in bond:
        return bond[f"{long_prefix}_{name}"]
    return bond.get(f"{name}{short_suffix}")


def _atom_map_to_name(sequences: Sequence[Mapping[str, Any]], entity: int, atom_id: int) -> str:
    r"""
    Resolve an integer atom-map identifier through the entity ``atom_map_to_atom_name`` table.
    """

    entity_entry = next(iter(sequences[entity - 1].values()))
    atom_map = entity_entry.get("atom_map_to_atom_name")
    if atom_map is None:
        raise ValueError(f"entity {entity} does not define 'atom_map_to_atom_name'")
    if atom_id in atom_map:
        return str(atom_map[atom_id])
    return str(atom_map[str(atom_id)])


def _bond_copies(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    chain_labels: dict[tuple[int, int], str],
) -> list[tuple[int, int]]:
    r"""
    Return the (left copy, right copy) pairs a bond applies to, respecting omitted copies.
    """

    left_copies = _entity_copies(left["entity"], chain_labels) if left["copy"] is None else [left["copy"]]
    right_copies = _entity_copies(right["entity"], chain_labels) if right["copy"] is None else [right["copy"]]
    if len(left_copies) != len(right_copies):
        raise ValueError("covalent bond cannot pair entities with differing copy counts")
    return list(zip(left_copies, right_copies, strict=True))


def _entity_copies(entity: int, chain_labels: dict[tuple[int, int], str]) -> list[int]:
    r"""
    Return the sorted copy numbers emitted for one entity.

    Args:
        entity: One-based entity id.
        chain_labels: ``(entity, copy) -> chain label`` map built while emitting chains.

    Returns:
        Sorted copy numbers belonging to the entity.
    """

    return sorted(copy for (bond_entity, copy) in chain_labels if bond_entity == entity)


def _bond_atom_index(
    chain_labels: dict[tuple[int, int], str],
    residue_atom_index: dict[ResidueKey, dict[str, int]],
    side: Mapping[str, Any],
    copy_number: int,
) -> int:
    r"""
    Resolve one bond endpoint to a global atom index into the emitted atom list.

    Args:
        chain_labels: ``(entity, copy) -> chain label`` map built while emitting chains.
        residue_atom_index: ``residue -> {atom_name: global index}`` map built while emitting atoms.
        side: Normalized bond endpoint (``entity`` / ``position`` / ``atom``).
        copy_number: Which copy of the entity this endpoint refers to.

    Returns:
        Global atom index of the endpoint.

    Raises:
        ValueError: If the referenced entity copy, residue, or atom is missing.
    """

    chain_label = chain_labels.get((side["entity"], copy_number))
    if chain_label is None:
        raise ValueError(f"covalent bond references missing entity {side['entity']} copy {copy_number}")
    residue = ResidueKey(chain_id=chain_label, residue_number=side["position"])
    names = residue_atom_index.get(residue)
    if names is None or side["atom"] not in names:
        raise ValueError(
            f"covalent bond references missing atom {side['atom']!r} in entity {side['entity']} "
            f"copy {copy_number} position {side['position']}"
        )
    return names[side["atom"]]
