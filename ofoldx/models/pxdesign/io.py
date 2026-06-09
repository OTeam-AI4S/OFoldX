r"""
PxDesign file I/O: JSON input parsing and structure output writing.

Parse a PxDesign (Protenix-based) design-job JSON into a typed OFoldX-native input, and write a
predicted structure back out to PDB or mmCIF.

Parsing ports the upstream PxDesign flow (`add_entity_atom_array` / `build_polymer` in
``pxdesign.data.json_parser``), which builds a biotite ``AtomArray`` per entity and attaches the design
annotations ``conditional_label``, ``coord_from_cif``, and ``coord_from_cif_is_resolved``. This module
reproduces that flow against the lightweight OFoldX
[`Structure`][ofoldx.data.structure.Structure], emitting a
[`DesignInput`][ofoldx.models.pxdesign.io.DesignInput] that pairs the parsed structure with the
per-atom design annotations that the generic structure processor would otherwise drop. Design chains are
materialized from their one-letter sequence via the CCD reference conformers (heavy atoms with zero
placeholder coords, all unresolved). Condition chains come from a structure file (``path``) plus optional
``chains``/``crop``/``hotspot`` selectors, keeping their experimental coords as the conditioning template.

Writing serializes a predicted structure to PDB or mmCIF with per-residue pLDDT confidence written into the
atom B-factor column, ported from the Boltz ``data.write`` path (``to_pdb`` / ``to_mmcif``) and adapted to
OFoldX's [`Structure`][ofoldx.data.structure.Structure] (per-atom records rather than the upstream
numpy struct-of-arrays). pLDDT is supplied per token: one value per polymer residue and one value per
ligand atom, matching the residue-token tokenization the models predict against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ...data.ccd import CcdProvider, ReferenceComponent, default_ccd_provider, heavy_atom_names
from ...data.residue_constants import dna_1to3, restype_1to3_with_unknown, rna_1to3
from ...data.structure import AtomRecord, ResidueKey, Structure, index_to_chain_id
from ...io.structure import read_structure

_MOL_TYPE_BY_ENTITY: dict[str, str] = {
    "proteinChain": "protein",
    "rnaSequence": "rna",
    "dnaSequence": "dna",
}
# Shared one-letter to CCD-code maps from the canonical residue-constants tables.
_SEQUENCE_MAP_BY_ENTITY: dict[str, dict[str, str]] = {
    "proteinChain": restype_1to3_with_unknown,
    "rnaSequence": rna_1to3,
    "dnaSequence": dna_1to3,
}

# Molecule types tokenized one token per residue; a ligand is one token per atom. This mirrors the
# residue-token tokenizer and decides ATOM vs HETATM plus how pLDDT is indexed.
_POLYMER_MOL_TYPES = frozenset({"protein", "rna", "dna"})


@dataclass(slots=True)
class DesignInput:
    r"""
    Structure paired with the design annotations parsed from a PxDesign JSON input.

    The generic structure processor consumes ``structure`` and re-derives the model-agnostic structural
    fields; the design annotations below carry the PxDesign-specific per-atom masks that the processor then
    restores. All annotation arrays are aligned to ``structure.atoms``.

    Attributes:
        structure: Parsed structure with design and condition chains concatenated in JSON order.
        conditional_label: ``[N_atom]`` bool, ``True`` for condition atoms and ``False`` for design atoms.
        coord_from_cif: ``[N_atom, 3]`` float32 conditioning coords (zeros for design atoms).
        coord_from_cif_is_resolved: ``[N_atom]`` bool, ``True`` where a real experimental coordinate exists.
        hotspot: ``[N_atom]`` bool marking hotspot atoms carried from the JSON ``hotspot`` selectors.
        name: Optional job/sample name used for output file naming.
    """

    structure: Structure
    conditional_label: np.ndarray
    coord_from_cif: np.ndarray
    coord_from_cif_is_resolved: np.ndarray
    hotspot: np.ndarray
    name: str | None = None


@dataclass(slots=True)
class _ChainAtoms:
    r"""
    Intermediate per-chain atom records plus aligned design annotations, accumulated during parsing.

    Attributes:
        atoms: Emitted atom records in JSON chain order.
        conditional_label: Per-atom condition flag (``True`` for condition atoms).
        coord_from_cif: Per-atom conditioning coordinate (zeros for design atoms).
        coord_from_cif_is_resolved: Per-atom flag marking a real experimental coordinate.
        hotspot: Per-atom hotspot flag from the JSON ``hotspot`` selectors.
    """

    atoms: list[AtomRecord] = field(default_factory=list)
    conditional_label: list[bool] = field(default_factory=list)
    coord_from_cif: list[tuple[float, float, float]] = field(default_factory=list)
    coord_from_cif_is_resolved: list[bool] = field(default_factory=list)
    hotspot: list[bool] = field(default_factory=list)


def read_pxdesign_json(
    source: str | PathLike[str] | Mapping[str, Any],
    *,
    ccd: CcdProvider | None = None,
) -> DesignInput:
    r"""
    Parse a PxDesign JSON input into a [`DesignInput`][ofoldx.models.pxdesign.io.DesignInput].

    Args:
        source: JSON file path or an already-loaded sample dict. The dict follows the PxDesign schema with a
            ``sequences`` list of entity dicts (``proteinChain`` / ``rnaSequence`` / ``dnaSequence``), each
            carrying ``sequence``, ``sequence_type`` (``"design"`` or ``"condition"``), an optional structure
            ``path`` for condition chains, and optional ``crop`` / ``hotspot`` selectors.
        ccd: Optional CCD provider used to materialize design-chain heavy atoms. Defaults to the process-wide
            provider.

    Returns:
        Parsed structure and aligned design annotations.

    Raises:
        ValueError: If the schema is malformed or an unsupported entity type is encountered.

    Examples:
        A one-residue design job with an in-memory [`CcdProvider`][ofoldx.data.ccd.CcdProvider]
        (the protocol is just ``get(name)``):

        >>> import numpy as np
        >>> from ofoldx.data import ReferenceComponent
        >>> gly = ReferenceComponent(
        ...     atom_map={"N": 0, "CA": 1, "C": 2, "O": 3},
        ...     coords=np.zeros((4, 3), "f4"), mask=np.ones(4, bool),
        ...     charge=np.zeros(4, "f4"), bonds=(), elements=("N", "C", "C", "O"))
        >>> class Ccd:
        ...     def get(self, name):
        ...         return gly
        >>> design = read_pxdesign_json({"sequences": [{"proteinChain": {"sequence": "G"}}]}, ccd=Ccd())
        >>> len(design.structure.atoms)
        4
        >>> read_pxdesign_json({})
        Traceback (most recent call last):
            ...
        ValueError: PxDesign JSON must contain a non-empty 'sequences' list
    """

    sample = _load_sample(source)
    ccd = ccd if ccd is not None else default_ccd_provider()

    sequences = sample.get("sequences")
    if not isinstance(sequences, Sequence) or not sequences:
        raise ValueError("PxDesign JSON must contain a non-empty 'sequences' list")

    used_chain_ids: set[str] = set()
    collected: _ChainAtoms = _ChainAtoms()
    for entity in sequences:
        if not isinstance(entity, Mapping) or len(entity) != 1:
            raise ValueError("each sequence entry must be a single-key entity mapping")
        entity_type, info = next(iter(entity.items()))
        if entity_type not in _MOL_TYPE_BY_ENTITY:
            raise ValueError(f"unsupported PxDesign entity type: {entity_type!r}")
        chain_id = _resolve_chain_id(info, used_chain_ids)
        _build_polymer(entity_type, info, chain_id, ccd, collected)

    structure = Structure(tuple(collected.atoms))
    return DesignInput(
        structure=structure,
        conditional_label=np.asarray(collected.conditional_label, dtype=np.bool_),
        coord_from_cif=np.asarray(collected.coord_from_cif, dtype=np.float32).reshape(-1, 3),
        coord_from_cif_is_resolved=np.asarray(collected.coord_from_cif_is_resolved, dtype=np.bool_),
        hotspot=np.asarray(collected.hotspot, dtype=np.bool_),
        name=sample.get("name"),
    )


def _load_sample(source: str | PathLike[str] | Mapping[str, Any]) -> Mapping[str, Any]:
    r"""
    Return the sample dict, loading it from a JSON file when a path is given.

    Args:
        source: An already-loaded sample mapping, or a path to a PxDesign JSON file.

    Returns:
        The sample mapping.

    Raises:
        ValueError: If the file does not decode to a JSON object.
    """

    if isinstance(source, Mapping):
        return source
    from chanfig import load

    loaded = load(source)
    if not isinstance(loaded, Mapping):
        raise ValueError("PxDesign JSON must decode to an object")
    return loaded


def _resolve_chain_id(info: Mapping[str, Any], used: set[str]) -> str:
    r"""
    Pick a chain id for one entity, preferring the JSON-supplied id and falling back to auto-assignment.

    Args:
        info: Entity info dict, optionally carrying ``json_chain_id`` / ``chain_id`` / ``id``.
        used: Chain ids already assigned; the chosen id is added to it.

    Returns:
        The resolved chain id.
    """

    chain_id = info.get("json_chain_id") or info.get("chain_id") or info.get("id")
    if chain_id is None:
        chain_id = _next_chain_id(used)
    chain_id = str(chain_id)
    used.add(chain_id)
    return chain_id


def _next_chain_id(used: set[str]) -> str:
    r"""
    Return the first spreadsheet-style chain id not already in ``used``.

    Args:
        used: Chain ids already assigned.

    Returns:
        The next free chain label (``"A"``, ``"B"``, ..., ``"AA"``, ...).

    Raises:
        ValueError: If the chain-id namespace is exhausted.
    """

    for index in range(1, 1_000_000):
        candidate = index_to_chain_id(index)
        if candidate not in used:
            return candidate
    raise ValueError("exhausted chain-id namespace")


def _build_polymer(
    entity_type: str,
    info: Mapping[str, Any],
    chain_id: str,
    ccd: CcdProvider,
    collected: _ChainAtoms,
) -> None:
    r"""
    Dispatch one entity to the sequence-materialized or file-read chain builder, appending to ``collected``.

    Args:
        entity_type: PxDesign entity key (``proteinChain`` / ``rnaSequence`` / ``dnaSequence``).
        info: Entity info dict (``sequence`` / ``sequence_type`` / ``path`` / ``crop`` / ``hotspot``).
        chain_id: Chain id assigned to this entity.
        ccd: CCD provider used to materialize design/condition heavy atoms.
        collected: Accumulator the emitted atoms and annotations are appended to.

    Raises:
        ValueError: If ``sequence_type`` is not ``"design"``/``"condition"`` or a sequence is missing.
    """

    sequence_type = str(info.get("sequence_type", "design")).lower()
    if sequence_type not in {"design", "condition"}:
        raise ValueError(f"sequence_type must be 'design' or 'condition', got {sequence_type!r}")

    crop = _parse_crop(info.get("crop"))
    hotspot_ids = {int(value) for value in info.get("hotspot", ()) or ()}

    if sequence_type == "condition" and info.get("path") is not None:
        _append_condition_from_file(info, chain_id, crop, hotspot_ids, collected)
        return

    sequence = info.get("sequence")
    if not sequence:
        raise ValueError(f"{entity_type} entity requires a 'sequence'")
    residue_names = _sequence_to_residue_names(entity_type, str(sequence))
    mol_type = _MOL_TYPE_BY_ENTITY[entity_type]
    is_condition = sequence_type == "condition"
    _append_polymer_from_sequence(residue_names, chain_id, mol_type, is_condition, crop, hotspot_ids, ccd, collected)


def _sequence_to_residue_names(entity_type: str, sequence: str) -> list[str]:
    r"""
    Map a polymer one-letter sequence to its ordered CCD residue names for the entity type.

    Args:
        entity_type: PxDesign entity key (``proteinChain`` / ``rnaSequence`` / ``dnaSequence``).
        sequence: One-letter residue sequence.

    Returns:
        Ordered three-letter CCD residue names.

    Raises:
        ValueError: If a residue code is unknown for the entity type.

    Examples:
        >>> _sequence_to_residue_names("proteinChain", "AG")
        ['ALA', 'GLY']
    """

    mapping = _SEQUENCE_MAP_BY_ENTITY[entity_type]
    residue_names: list[str] = []
    for code in sequence.strip():
        if code not in mapping:
            raise ValueError(f"unknown {entity_type} residue code: {code!r}")
        residue_names.append(mapping[code])
    return residue_names


def _append_polymer_from_sequence(
    residue_names: Sequence[str],
    chain_id: str,
    mol_type: str,
    is_condition: bool,
    crop: set[int] | None,
    hotspot_ids: set[int],
    ccd: CcdProvider,
    collected: _ChainAtoms,
) -> None:
    r"""
    Materialize a design/condition polymer chain from its residue-name sequence via CCD conformers.

    Design residues (``is_condition=False``) get zero conditioning coords and are marked unresolved via
    ``conditional_label`` / ``coord_from_cif_is_resolved``; the OFoldX tokenizer keys off the real CCD
    residue name (rather than the upstream ``xpb`` placeholder rename) so a design chain still tokenizes one
    token per residue and the distogram/frame rules apply. Reference conformer atoms come from the CCD; the
    target coords carry the zero placeholder because a sequence-only chain has no experimental structure.
    """

    num_residues = len(residue_names)
    for residue_index, residue_name in enumerate(residue_names, start=1):
        if crop is not None and residue_index not in crop:
            continue
        component = ccd.get(residue_name)
        atom_names, elements = _component_atoms(
            component, residue_name, mol_type, terminal=residue_index in (1, num_residues)
        )
        residue = ResidueKey(chain_id=chain_id, residue_number=residue_index)
        is_hotspot = residue_index in hotspot_ids
        for atom_name, element in zip(atom_names, elements):
            collected.atoms.append(
                AtomRecord(
                    name=atom_name,
                    element=element,
                    residue_name=residue_name,
                    residue=residue,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    hetero=False,
                    mol_type=mol_type,
                )
            )
            collected.conditional_label.append(is_condition)
            collected.coord_from_cif.append((0.0, 0.0, 0.0))
            collected.coord_from_cif_is_resolved.append(False)
            collected.hotspot.append(is_hotspot)


def _component_atoms(
    component: ReferenceComponent | None, residue_name: str, mol_type: str, *, terminal: bool
) -> tuple[list[str], list[str]]:
    r"""
    Return heavy-atom names/elements for a residue, matching the generic Protenix JSON emission rule.

    Internal-residue leaving atoms (protein ``OXT``, nucleic ``OP3``) and hydrogens are dropped via the shared
    [`heavy_atom_names`][ofoldx.data.ccd.heavy_atom_names]. When the CCD is unavailable the chain falls
    back to a single representative atom so it still tokenizes.
    """

    if component is not None and component.atom_map and component.elements:
        names = heavy_atom_names(component, mol_type=mol_type, keep_leaving=terminal)
        elements = [component.elements[component.atom_map[name]].upper() for name in names]
        if names:
            return names, elements
    # CCD unavailable: fall back to a single representative atom so the chain still tokenizes.
    representative = "CA" if mol_type == "protein" else "C1'"
    return [representative], [_element_from_atom_name(representative)]


def _element_from_atom_name(atom_name: str) -> str:
    r"""
    Guess the element symbol from an atom name (first alphabetic char), defaulting to carbon.

    Args:
        atom_name: PDB-style atom name.

    Returns:
        Single-letter uppercase element symbol; ``"C"`` when the name has no letters.

    Examples:
        >>> _element_from_atom_name("CA"), _element_from_atom_name("OP3")
        ('C', 'O')
    """

    for char in atom_name:
        if char.isalpha():
            return char.upper()
    return "C"


def _append_condition_from_file(
    info: Mapping[str, Any],
    chain_id: str,
    crop: set[int] | None,
    hotspot_ids: set[int],
    collected: _ChainAtoms,
) -> None:
    r"""
    Append a condition chain read from a structure file, keeping its experimental coords as the template.
    """

    structure = read_structure(info["path"])
    select_chains = _condition_chain_filter(info)
    for atom in structure.atoms:
        source_chain = atom.residue.chain_id
        if select_chains is not None and source_chain not in select_chains:
            continue
        residue_number = atom.residue.residue_number
        if crop is not None and residue_number not in crop:
            continue
        residue = ResidueKey(
            chain_id=chain_id,
            residue_number=residue_number,
            insertion_code=atom.residue.insertion_code,
        )
        collected.atoms.append(
            AtomRecord(
                name=atom.name,
                element=atom.element,
                residue_name=atom.residue_name,
                residue=residue,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                charge=atom.charge,
                hetero=atom.hetero,
                mol_type=atom.mol_type,
            )
        )
        collected.conditional_label.append(True)
        collected.coord_from_cif.append((atom.x, atom.y, atom.z))
        collected.coord_from_cif_is_resolved.append(True)
        collected.hotspot.append(residue_number in hotspot_ids)


def _condition_chain_filter(info: Mapping[str, Any]) -> set[str] | None:
    r"""
    Return the set of source-chain ids to keep for a condition chain, or ``None`` to keep all.

    Args:
        info: Condition-entity info dict, optionally carrying ``chains`` / ``json_chain_id`` / ``chain_id``.

    Returns:
        Set of chain ids to select from the source file, or ``None`` when no selector is present.
    """

    chains = info.get("chains") or info.get("json_chain_id") or info.get("chain_id")
    if chains is None:
        return None
    if isinstance(chains, str):
        return {chains}
    return {str(value) for value in chains}


def _parse_crop(raw: Any) -> set[int] | None:
    r"""
    Parse a PxDesign crop selector (``"1-116,130,140-150"``) into a residue-number set.

    Args:
        raw: Crop selector as a comma/range string, a sequence of tokens, or ``None`` for no crop.

    Returns:
        Set of one-based residue numbers to keep, or ``None`` when no crop is given.

    Raises:
        ValueError: If the selector is neither a string, a sequence, nor ``None``.

    Examples:
        >>> sorted(_parse_crop("1-3,7"))
        [1, 2, 3, 7]
    """

    if raw is None:
        return None
    if isinstance(raw, str):
        tokens = raw.replace(" ", "").split(",")
    elif isinstance(raw, Sequence):
        tokens = [str(value) for value in raw]
    else:
        raise ValueError(f"unsupported crop selector: {raw!r}")
    keep: set[int] = set()
    for token in tokens:
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            keep.update(range(int(start), int(end) + 1))
        else:
            keep.add(int(token))
    return keep


def write_structure(
    structure: Structure,
    coords: Sequence[tuple[float, float, float]],
    path: str | PathLike[str],
    *,
    plddt: Sequence[float] | None = None,
    format: str = "auto",
) -> None:
    r"""
    Write a predicted structure to a PDB or mmCIF file.

    Args:
        structure: Parsed structure defining atom identity and ordering.
        coords: Predicted Cartesian coordinates, one ``(x, y, z)`` per atom in ``structure.atoms`` order.
        path: Output file path.
        plddt: Optional per-token pLDDT in ``[0, 1]`` (one value per polymer residue, one per ligand atom),
            written to the B-factor column scaled to ``[0, 100]``. When omitted every B-factor is ``100.0``.
        format: Output format. ``"auto"`` infers from the suffix; explicit values are ``"pdb"``, ``"cif"``,
            ``"mmcif"``.

    Raises:
        ValueError: If ``coords`` length does not match the atom count, or the format is unsupported.

    Examples:
        >>> from ofoldx.io import read_structure  # doctest: +SKIP
        >>> structure = read_structure("complex.cif")  # doctest: +SKIP
        >>> write_structure(structure, pred_coords, "prediction.cif", plddt=plddt)  # doctest: +SKIP
    """

    path = Path(path)
    output_format = path.suffix.lower().lstrip(".") if format == "auto" else format.lower()
    if output_format == "pdb":
        text = to_pdb(structure, coords, plddt=plddt)
    elif output_format in {"cif", "mmcif"}:
        text = to_mmcif(structure, coords, plddt=plddt)
    else:
        raise ValueError(f"unsupported structure format: {output_format!r}")
    path.write_text(text)


def to_pdb(
    structure: Structure,
    coords: Sequence[tuple[float, float, float]],
    *,
    plddt: Sequence[float] | None = None,
) -> str:
    r"""
    Render a predicted structure as PDB text.

    Args:
        structure: Parsed structure defining atom identity and ordering.
        coords: Predicted Cartesian coordinates, one ``(x, y, z)`` per atom in ``structure.atoms`` order.
        plddt: Optional per-token pLDDT in ``[0, 1]`` written to the B-factor column scaled to ``[0, 100]``.

    Returns:
        PDB file contents.

    Raises:
        ValueError: If ``coords`` length does not match the atom count.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> residue = ResidueKey(chain_id="A", residue_number=1)
        >>> atom = AtomRecord(name="CA", element="C", residue_name="ALA", residue=residue, x=0.0, y=0.0, z=0.0)
        >>> text = to_pdb(Structure((atom,)), [(1.0, 2.0, 3.0)])
        >>> text.splitlines()[0].split()
        ['ATOM', '1', 'CA', 'ALA', 'A', '1', '1.000', '2.000', '3.000', '1.00100.00', 'C']
    """

    _check_coords(structure, coords)
    b_factors = _atom_b_factors(structure, plddt)
    lines: list[str] = []
    atom_serial = 1
    atom_to_serial: dict[int, int] = {}  # TER records consume a serial, so serial != atom_index + 1
    atoms = structure.atoms
    chain_groups = _chain_groups(atoms)
    for group_index, (chain_id, group) in enumerate(chain_groups):
        for atom_index in group:
            atom = atoms[atom_index]
            x, y, z = coords[atom_index]
            record_type = "HETATM" if _is_ligand_atom(atom) else "ATOM"
            residue_name = "LIG" if record_type == "HETATM" else atom.residue_name[:3]
            name = _pdb_atom_name(atom.name)
            lines.append(
                f"{record_type:<6}{atom_serial:>5} {name:<4}{'':>1}"
                f"{residue_name:>3} {chain_id[:1]:>1}"
                f"{atom.residue.residue_number:>4}{atom.residue.insertion_code[:1]:>1}   "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"{1.00:>6.2f}{b_factors[atom_index]:>6.2f}          "
                f"{atom.element.upper():>2}{'':>2}"
            )
            atom_to_serial[atom_index] = atom_serial
            atom_serial += 1
        if group_index < len(chain_groups) - 1:
            last = atoms[group[-1]]
            last_residue_name = "LIG" if _is_ligand_atom(last) else last.residue_name[:3]
            lines.append(
                f"{'TER':<6}{atom_serial:>5}      {last_residue_name:>3} "
                f"{chain_id[:1]:>1}{last.residue.residue_number:>4}"
            )
            atom_serial += 1
    lines.extend(_conect_records(structure, atom_to_serial))
    lines.append("END")
    lines.append("")
    return "\n".join(line.ljust(80) for line in lines)


def to_mmcif(
    structure: Structure,
    coords: Sequence[tuple[float, float, float]],
    *,
    plddt: Sequence[float] | None = None,
) -> str:
    r"""
    Render a predicted structure as mmCIF text with pLDDT in the ``B_iso_or_equiv`` column.

    Builds a biotite ``AtomArray`` (the same library [`read_mmcif`][ofoldx.io.read_mmcif] reads with) so the
    output is standards-compliant and round-trips through standard mmCIF parsers.

    Args:
        structure: Parsed structure defining atom identity and ordering.
        coords: Predicted Cartesian coordinates, one ``(x, y, z)`` per atom in ``structure.atoms`` order.
        plddt: Optional per-token pLDDT in ``[0, 1]`` written scaled to ``[0, 100]``.

    Returns:
        mmCIF file contents.

    Raises:
        ValueError: If ``coords`` length does not match the atom count.

    Examples:
        >>> import torch
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> res = ResidueKey("A", 1)
        >>> structure = Structure(tuple(
        ...     AtomRecord(n, n[0], "GLY", res, 0.0, 0.0, 0.0, mol_type="protein")
        ...     for n in ("N", "CA", "C", "O")))
        >>> text = to_mmcif(structure, torch.zeros(4, 3), plddt=torch.ones(4))
        >>> text.startswith("data_prediction")
        True
    """

    from io import StringIO

    import numpy as np
    from biotite.structure import Atom, array
    from biotite.structure.io.pdbx import CIFFile, set_structure

    _check_coords(structure, coords)
    b_factors = _atom_b_factors(structure, plddt)
    biotite_atoms = []
    for atom_index, atom in enumerate(structure.atoms):
        ligand = _is_ligand_atom(atom)
        biotite_atoms.append(
            Atom(
                np.asarray(coords[atom_index], dtype=np.float32),
                chain_id=atom.residue.chain_id,
                res_id=atom.residue.residue_number,
                ins_code=atom.residue.insertion_code,
                res_name="LIG" if ligand else atom.residue_name,
                hetero=ligand,
                atom_name=atom.name,
                element=atom.element.upper(),
                b_factor=b_factors[atom_index],
            )
        )
    atom_array = array(biotite_atoms)
    atom_array.set_annotation("b_factor", np.asarray(b_factors, dtype=np.float32))
    cif = CIFFile()
    set_structure(cif, atom_array, data_block="prediction")
    buffer = StringIO()
    cif.write(buffer)
    return buffer.getvalue()


def _check_coords(structure: Structure, coords: Sequence[tuple[float, float, float]]) -> None:
    r"""
    Validate that one coordinate triple was supplied per atom.

    Args:
        structure: Parsed structure defining the expected atom count.
        coords: Predicted coordinates, one ``(x, y, z)`` per atom.

    Raises:
        ValueError: If ``coords`` length does not match the atom count.
    """

    if len(coords) != len(structure.atoms):
        raise ValueError(f"expected {len(structure.atoms)} coordinates, got {len(coords)}")


def _is_ligand_atom(atom: AtomRecord) -> bool:
    r"""
    Return whether an atom is a (per-atom-tokenized) ligand rather than a polymer residue.

    Uses the explicit ``mol_type`` when present, otherwise falls back to the ``hetero`` flag.

    Args:
        atom: The atom record to classify.

    Returns:
        ``True`` for a ligand atom (written as ``HETATM``), ``False`` for a polymer residue atom.
    """

    if atom.mol_type is not None:
        return atom.mol_type not in _POLYMER_MOL_TYPES
    return atom.hetero


def _pdb_atom_name(name: str) -> str:
    r"""
    Return the PDB-column atom name; 4-character names fill the column, shorter names are left-padded.

    Args:
        name: Atom name.

    Returns:
        The atom name left-padded with one space unless it already fills the 4-character column.

    Examples:
        >>> _pdb_atom_name("CA"), _pdb_atom_name("FE1X")
        (' CA', 'FE1X')
    """

    return name if len(name) == 4 else f" {name}"  # noqa: PLR2004


def _chain_groups(atoms: Sequence[AtomRecord]) -> list[tuple[str, list[int]]]:
    r"""
    Group atom indices by chain id, preserving first-seen order.
    """

    groups: dict[str, list[int]] = {}
    for atom_index, atom in enumerate(atoms):
        groups.setdefault(atom.residue.chain_id, []).append(atom_index)
    return list(groups.items())


def _conect_records(structure: Structure, atom_to_serial: dict[int, int]) -> list[str]:
    r"""
    Return CONECT records for explicit inter-residue bonds, using the written atom serials.
    """

    return [f"CONECT{atom_to_serial[first]:>5}{atom_to_serial[second]:>5}" for first, second in structure.bonds]


def _atom_b_factors(structure: Structure, plddt: Sequence[float] | None) -> list[float]:
    r"""
    Broadcast per-token pLDDT to per-atom B-factors.

    One token per polymer residue (its pLDDT fills every atom of the residue) and one token per ligand
    atom, following the residue-token tokenizer. Absent pLDDT yields a full-confidence ``100.0`` column.
    """

    atoms = structure.atoms
    if plddt is None:
        return [100.0] * len(atoms)
    b_factors: list[float] = []
    token_cursor = -1
    previous_polymer_residue = None
    for atom in atoms:
        if _is_ligand_atom(atom):
            token_cursor += 1
        elif atom.residue != previous_polymer_residue:
            previous_polymer_residue = atom.residue
            token_cursor += 1
        b_factors.append(round(float(plddt[token_cursor]) * 100, 2))
    return b_factors
