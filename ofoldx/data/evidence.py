r"""
Multiple-sequence alignments and structural templates.
"""

from __future__ import annotations

import json
import operator
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .coordinates import ChainCoordinates, PolymerCoordinates
from .molecular import (
    _STANDARD_COMPONENTS,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MolecularSystem,
    one_letter_to_component,
)
from .structure import Structure
from .tokenization import (
    TOKEN_ALPHABET,
    TOKEN_GAP_INDEX,
    TOKEN_INDEX,
    TOKEN_OTHER_INDEX,
    TOKEN_VOCAB_SIZE,
    component_token_index,
    residue_token_index,
)

MSA_ALPHABET = TOKEN_ALPHABET
MSA_INDEX = TOKEN_INDEX
MSA_VOCAB_SIZE = TOKEN_VOCAB_SIZE
MSA_GAP_INDEX = TOKEN_GAP_INDEX
MSA_OTHER_INDEX = TOKEN_OTHER_INDEX
msa_residue_index = residue_token_index
msa_component_index = component_token_index

MSA_ARTIFACT_SUFFIX = ".msa.safetensors"
TEMPLATE_ARTIFACT_SUFFIX = ".templates.safetensors"

_MSA_ARTIFACT_FORMAT = "ofoldx.msa"
_TEMPLATE_ARTIFACT_FORMAT = "ofoldx.templates"

_MSA_ASCII_INDEX = {
    polymer_type: np.fromiter(
        (residue_token_index(chr(code), polymer_type) for code in range(128)),
        dtype=np.int64,
        count=128,
    )
    for polymer_type in ("protein", "rna", "dna")
}


# Canonical byte keys avoid allocating a Python integer tuple per wide MSA row.
_PROTEIN_MSA_BYTE_INDEX = bytes(msa_residue_index(chr(code), "protein") for code in range(256))

# Component IDs a one-letter alphabet names unambiguously, per polymer chemistry. Used to check an MSA
# query row against its target sequence: only standard residues on both sides can genuinely conflict.
_STANDARD_COMPONENT_IDS: dict[str, frozenset[str]] = {
    polymer_type: frozenset(mapping.values()) for polymer_type, mapping in _STANDARD_COMPONENTS.items()
}
_POLYMER_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
    ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
    ENTITY_PNA: "pna",
}

_PROTEIN_QUERY_COMPONENTS: dict[str, frozenset[str]] = {
    **{symbol: frozenset((component,)) for symbol, component in _STANDARD_COMPONENTS["protein"].items()},
    "B": frozenset(("ASP", "ASN", "ASX")),
    "Z": frozenset(("GLU", "GLN", "GLX")),
    "J": frozenset(("ILE", "LEU", "XLE")),
    "U": frozenset(("SEC",)),
    "O": frozenset(("PYL",)),
}
_KNOWN_PROTEIN_QUERY_COMPONENTS = frozenset(
    component for components in _PROTEIN_QUERY_COMPONENTS.values() for component in components
)


def encode_msa_sequences(sequences: Sequence[str], polymer_type: str) -> NDArray[np.int64]:
    r"""Encode an aligned sequence pool in the canonical token vocabulary."""

    if not sequences:
        return np.empty((0, 0), dtype=np.int64)
    width = len(sequences[0])
    if any(len(sequence) != width for sequence in sequences):
        raise ValueError("every MSA row must have the same width")
    codes = np.frombuffer("".join(sequences).encode("ascii", "replace"), dtype=np.uint8).reshape(len(sequences), width)
    table = _MSA_ASCII_INDEX.get(polymer_type)
    if table is None:
        return np.full(codes.shape, residue_token_index("?", polymer_type), dtype=np.int64)
    return table[codes]


def _instance_component_bounds(system: MolecularSystem, instance_index: int) -> tuple[int, int]:
    pointers = system.instance_component_ptr
    return int(pointers[instance_index]), int(pointers[instance_index + 1])


def _query_conflicts_with_instance(
    query: str,
    system: MolecularSystem,
    instance_index: int,
) -> int | None:
    r"""
    Locate the first query column whose residue chemistry contradicts the target sequence.

    The MSA query row is the one-letter target sequence. A column conflicts only when the query symbol
    and the bound target residue resolve to distinct known components. Protein ``B``, ``Z``, and ``J``
    accept their two IUPAC constituents; ``U`` and ``O`` name SEC and PYL exactly; only ``X`` is a wildcard.
    A modified target residue the one-letter alphabet cannot name is treated as agreeing. Returns the
    offending column index, or ``None`` when the query is consistent with the target.
    """

    entity_index = int(system.instance_entity[instance_index])
    polymer_type = _POLYMER_TYPE.get(int(system.entity_type[entity_index]))
    if polymer_type is None:
        return None
    standard_ids = (
        _STANDARD_COMPONENT_IDS["rna"] | _STANDARD_COMPONENT_IDS["dna"]
        if polymer_type == "dna_rna_hybrid"
        else _STANDARD_COMPONENT_IDS.get(polymer_type)
    )
    if standard_ids is None:
        return None
    start, stop = _instance_component_bounds(system, instance_index)
    component_symbols = system.component_symbol[start:stop].detach().cpu().tolist()
    for index, (symbol, component_symbol) in enumerate(zip(query, component_symbols)):
        target_id = system.symbols[component_symbol]
        if polymer_type == "protein":
            if target_id not in _KNOWN_PROTEIN_QUERY_COMPONENTS:
                continue
            if symbol == "X":
                continue
            accepted = _PROTEIN_QUERY_COMPONENTS.get(symbol)
            if accepted is None or target_id not in accepted:
                return index
        else:
            if target_id not in standard_ids:
                continue
            query_polymer_type = polymer_type
            if polymer_type == "dna_rna_hybrid":
                query_polymer_type = "rna" if target_id in _STANDARD_COMPONENT_IDS["rna"] else "dna"
            try:
                query_id = one_letter_to_component(symbol, query_polymer_type, policy="strict")
            except KeyError:
                continue
            if query_id != target_id:
                return index
    return None


@dataclass(eq=False)
class MSA:
    r"""
    Paired and unpaired alignments for one molecular entity.

    Sequences use uppercase one-letter MSA symbols and ``-`` for gaps. Row zero
    of both pools is the same ungapped query sequence.

    Attributes:
        entity_id: Target entity ID in the associated molecular system.
        unpaired_sequences: Query-width rows in the unpaired MSA pool.
        unpaired_deletion_counts: Unsigned deletion counts with shape
            ``(num_unpaired_rows, query_length)``.
        paired_sequences: Query-width rows in the paired MSA pool.
        paired_deletion_counts: Unsigned deletion counts with shape
            ``(num_paired_rows, query_length)``.

    Raises:
        ValueError: If query rows, row widths, deletion-count shapes, or dtypes are
            inconsistent.
    """

    entity_id: str
    unpaired_sequences: tuple[str, ...]
    unpaired_deletion_counts: NDArray[np.uint32]
    paired_sequences: tuple[str, ...]
    paired_deletion_counts: NDArray[np.uint32]

    def __post_init__(self) -> None:
        if not self.unpaired_sequences:
            raise ValueError("unpaired_sequences must contain the query row")
        if not self.paired_sequences:
            raise ValueError("paired_sequences must contain the query row")

        width = len(self.unpaired_sequences[0])
        if any(len(sequence) != width for sequence in self.unpaired_sequences):
            raise ValueError("every unpaired row must have the query width")
        if any(len(sequence) != width for sequence in self.paired_sequences):
            raise ValueError("every paired row must have the query width")

        if self.unpaired_sequences[0] != self.paired_sequences[0]:
            raise ValueError("the unpaired and paired query rows must be identical")
        if "-" in self.unpaired_sequences[0]:
            raise ValueError("the MSA query row must be ungapped")

        unpaired_shape = (len(self.unpaired_sequences), width)
        if self.unpaired_deletion_counts.shape != unpaired_shape:
            raise ValueError(f"unpaired_deletion_counts must have shape {unpaired_shape}")
        if self.unpaired_deletion_counts.dtype != np.uint32:
            raise ValueError("unpaired_deletion_counts must have dtype uint32")

        paired_shape = (len(self.paired_sequences), width)
        if self.paired_deletion_counts.shape != paired_shape:
            raise ValueError(f"paired_deletion_counts must have shape {paired_shape}")
        if self.paired_deletion_counts.dtype != np.uint32:
            raise ValueError("paired_deletion_counts must have dtype uint32")


@dataclass(eq=False)
class MsaCollection:
    r"""
    A row assembly plan spanning one or more entity MSAs.

    Column ``c`` refers to ``msas[c]``. ``row_pairing[r, c]`` selects a row from
    that MSA, or ``-1`` for a gap; ``row_is_paired[r, c]`` selects the paired or
    unpaired row pool.

    Attributes:
        msas: MSA columns participating in the assembly.
        row_pairing: Signed integer array with shape ``(num_rows, num_msas)``.
        row_is_paired: Boolean array aligned with ``row_pairing``.
        target_instance_ids: Optional exact target instance for each MSA column.
            When absent, each column applies to every instance of its entity.

    Raises:
        ValueError: If arrays, row indices, query rows, entities, or instance targets
            are inconsistent.
    """

    msas: tuple[MSA, ...]
    row_pairing: NDArray[np.int64]
    row_is_paired: NDArray[np.bool_]
    target_instance_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.msas:
            raise ValueError("msas must not be empty")
        if self.row_pairing.ndim != 2 or self.row_pairing.shape[0] == 0:
            raise ValueError("row_pairing must be a 2D array holding the query row")

        column_count = len(self.msas)
        if self.row_pairing.shape != self.row_is_paired.shape:
            raise ValueError("row_pairing and row_is_paired must have the same shape")
        if self.row_pairing.shape[1] != column_count:
            raise ValueError("row_pairing must have one column per MSA")
        if self.row_pairing.dtype != np.int64:
            raise ValueError("row_pairing must have dtype int64")
        if self.row_is_paired.dtype != np.bool_:
            raise ValueError("row_is_paired must have dtype bool")

        if not np.all(self.row_pairing[0] == 0):
            raise ValueError("the first plan row must select query row zero in every column")
        if not np.all(self.row_is_paired[0]):
            raise ValueError("the first plan row must select the paired pool in every column")

        gap = self.row_pairing == -1
        if np.any(gap & self.row_is_paired):
            raise ValueError("a gap cell cannot be marked paired")
        if np.any(np.all(gap, axis=1)):
            raise ValueError("each plan row must select at least one populated cell")

        for column_index, msa in enumerate(self.msas):
            pairing = self.row_pairing[:, column_index]
            paired = self.row_is_paired[:, column_index]
            populated = pairing != -1
            paired_cells = populated & paired
            unpaired_cells = populated & ~paired
            paired_bound = len(msa.paired_sequences)
            unpaired_bound = len(msa.unpaired_sequences)
            if np.any((pairing[paired_cells] < 0) | (pairing[paired_cells] >= paired_bound)):
                raise ValueError("row_pairing selects a paired row outside its MSA")
            if np.any((pairing[unpaired_cells] < 0) | (pairing[unpaired_cells] >= unpaired_bound)):
                raise ValueError("row_pairing selects an unpaired row outside its MSA")

        if self.target_instance_ids is None:
            entity_ids = tuple(msa.entity_id for msa in self.msas)
            if len(set(entity_ids)) != len(entity_ids):
                raise ValueError("broadcast MSAs must bind pairwise-unique entity_id columns")
        else:
            if len(self.target_instance_ids) != column_count:
                raise ValueError("target_instance_ids must have one entry per MSA")
            if len(set(self.target_instance_ids)) != len(self.target_instance_ids):
                raise ValueError("target_instance_ids must be pairwise unique")


def save_msa_collection(collection: MsaCollection, path: str | os.PathLike[str]) -> None:
    r"""Save sequence pools, deletion counts, bindings, and row pairing as one artifact.

    Args:
        collection: MSA collection whose sequence pools and pairing plan are preserved.
        path: Destination ending in ``.msa.safetensors``.

    Raises:
        ValueError: The destination suffix is invalid or sequences contain non-ASCII symbols.
    """

    destination = _msa_artifact_path(path)
    state = {
        "row_pairing": torch.from_numpy(collection.row_pairing),
        "row_is_paired": torch.from_numpy(collection.row_is_paired),
    }
    for index, msa in enumerate(collection.msas):
        prefix = f"msa.{index}."
        state[f"{prefix}unpaired_sequences"] = _sequence_tensor(msa.unpaired_sequences)
        state[f"{prefix}unpaired_deletion_counts"] = torch.from_numpy(msa.unpaired_deletion_counts)
        state[f"{prefix}paired_sequences"] = _sequence_tensor(msa.paired_sequences)
        state[f"{prefix}paired_deletion_counts"] = torch.from_numpy(msa.paired_deletion_counts)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        str(destination),
        metadata={
            "format": _MSA_ARTIFACT_FORMAT,
            "entity_ids": json.dumps([msa.entity_id for msa in collection.msas], ensure_ascii=False),
            "target_instance_ids": json.dumps(collection.target_instance_ids, ensure_ascii=False),
        },
    )


def load_msa_collection(path: str | os.PathLike[str]) -> MsaCollection:
    r"""Load sequence pools, bindings, and row pairing from an MSA collection artifact.

    Reads the complete schema written by
    [`save_msa_collection`][ofoldx.data.evidence.save_msa_collection].

    Args:
        path: Source ending in ``.msa.safetensors``.

    Returns:
        Sequence pools and deletion counts with their entity and instance bindings
            and row pairing plan.

    Raises:
        ValueError: The artifact suffix, format, metadata, or tensor schema is invalid.
    """

    source = _msa_artifact_path(path)
    with safe_open(str(source), framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata()
        if metadata.get("format") != _MSA_ARTIFACT_FORMAT:
            raise ValueError(f"{source} is not an OFoldX MSA artifact")
        state = {key: artifact.get_tensor(key) for key in artifact.keys()}

    entity_ids = cast(tuple[str, ...], _msa_metadata_strings(metadata, "entity_ids", source))
    target_instance_ids = _msa_metadata_strings(metadata, "target_instance_ids", source, optional=True)
    expected_keys = {"row_pairing", "row_is_paired"}
    for index in range(len(entity_ids)):
        expected_keys.update(
            {
                f"msa.{index}.unpaired_sequences",
                f"msa.{index}.unpaired_deletion_counts",
                f"msa.{index}.paired_sequences",
                f"msa.{index}.paired_deletion_counts",
            }
        )
    if state.keys() != expected_keys:
        raise ValueError(f"{source} does not contain the complete MsaCollection tensor schema")
    if state["row_pairing"].dtype != torch.int64 or state["row_is_paired"].dtype != torch.bool:
        raise ValueError(f"{source} contains an invalid MsaCollection tensor dtype")

    msas = []
    for index, entity_id in enumerate(entity_ids):
        prefix = f"msa.{index}."
        msas.append(
            MSA(
                entity_id=entity_id,
                unpaired_sequences=_decode_sequence_tensor(state[f"{prefix}unpaired_sequences"], source),
                unpaired_deletion_counts=state[f"{prefix}unpaired_deletion_counts"].numpy(),
                paired_sequences=_decode_sequence_tensor(state[f"{prefix}paired_sequences"], source),
                paired_deletion_counts=state[f"{prefix}paired_deletion_counts"].numpy(),
            )
        )

    return MsaCollection(
        msas=tuple(msas),
        row_pairing=state["row_pairing"].numpy(),
        row_is_paired=state["row_is_paired"].numpy(),
        target_instance_ids=target_instance_ids,
    )


def _sequence_tensor(sequences: Sequence[str]) -> Tensor:
    try:
        encoded = "".join(sequences).encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("MSA artifact sequences must contain ASCII symbols") from error
    return torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy()).reshape(len(sequences), len(sequences[0]))


def _decode_sequence_tensor(encoded: Tensor, path: Path) -> tuple[str, ...]:
    if encoded.ndim != 2 or encoded.dtype != torch.uint8:
        raise ValueError(f"{path} contains invalid MSA sequence storage")
    try:
        text = encoded.numpy().tobytes().decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} contains non-ASCII MSA sequence storage") from error
    width = encoded.shape[1]
    return tuple(text[row * width : (row + 1) * width] for row in range(encoded.shape[0]))


def _msa_artifact_path(path: str | os.PathLike[str]) -> Path:
    artifact = Path(path)
    if not artifact.name.endswith(MSA_ARTIFACT_SUFFIX):
        raise ValueError(f"MSA artifacts must end with {MSA_ARTIFACT_SUFFIX!r}")
    return artifact


def _msa_metadata_strings(
    metadata: dict[str, str],
    key: str,
    path: Path,
    *,
    optional: bool = False,
) -> tuple[str, ...] | None:
    values = json.loads(metadata[key])
    if optional and values is None:
        return None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"invalid {key!r} metadata in {path}")
    return tuple(values)


def merge_unpaired_protein_msas(msas: Sequence[MSA]) -> MSA:
    r"""Merge and canonically deduplicate protein MSA pools for one entity."""

    if not msas:
        raise ValueError("at least one MSA is required")
    source = msas[0]
    sequences: list[str] = []
    deletion_counts: list[NDArray[np.uint32]] = []
    seen: set[bytes] = set()
    for msa in msas:
        if msa.entity_id != source.entity_id:
            raise ValueError("merged MSAs must target the same entity")
        if msa.unpaired_sequences[0] != source.unpaired_sequences[0]:
            raise ValueError(f"unpaired MSA queries differ for entity {source.entity_id!r}")
        for sequence, deletions in zip(msa.unpaired_sequences, msa.unpaired_deletion_counts, strict=True):
            key = sequence.encode("ascii", "replace").translate(_PROTEIN_MSA_BYTE_INDEX)
            if key not in seen:
                seen.add(key)
                sequences.append(sequence)
                deletion_counts.append(deletions)
    counts = np.stack(deletion_counts)
    return MSA(
        source.entity_id,
        tuple(sequences),
        counts,
        (sequences[0],),
        counts[0:1].copy(),
    )


def pair_msa_rows_by_key(
    keys_by_msa: Sequence[Sequence[str]],
    *,
    max_rows_per_key: int,
) -> Iterator[NDArray[np.int64]]:
    r"""Align source-row indices for shared nonempty keys by chain coverage."""

    rows_by_key_by_msa: list[dict[str, list[int]]] = []
    key_coverage: dict[str, int] = {}
    min_rows_by_key: dict[str, int] = {}
    for keys in keys_by_msa:
        rows_by_key: dict[str, list[int]] = {}
        for row, key in enumerate(keys):
            if key:
                rows_by_key.setdefault(key, []).append(row)
        rows_by_key_by_msa.append(rows_by_key)
        for key in sorted(rows_by_key):
            key_coverage[key] = key_coverage.get(key, 0) + 1
            min_rows_by_key[key] = min(
                min_rows_by_key.get(key, max_rows_per_key),
                len(rows_by_key[key]),
            )

    for coverage in sorted(set(key_coverage.values()), reverse=True):
        blocks: list[NDArray[np.int64]] = []
        for key, key_count in key_coverage.items():
            if key_count != coverage or coverage <= 1:
                continue
            row_count = min_rows_by_key[key]
            block: NDArray[np.int64] = np.full((row_count, len(keys_by_msa)), -1, dtype=np.int64)
            for column, rows_by_key in enumerate(rows_by_key_by_msa):
                if key in rows_by_key:
                    block[:, column] = rows_by_key[key][:row_count]
            blocks.append(block)
        if blocks:
            yield np.concatenate(blocks, axis=0)


def unpaired_msa_row_plan(rows_by_msa: Sequence[Sequence[int]]) -> NDArray[np.int64]:
    r"""Build a block-diagonal unpaired plan, interleaved by source depth."""

    plan: NDArray[np.int64] = np.full(
        (sum(len(rows) for rows in rows_by_msa), len(rows_by_msa)),
        -1,
        dtype=np.int64,
    )
    output_row = 0
    for depth in range(max((len(rows) for rows in rows_by_msa), default=0)):
        for column, rows in enumerate(rows_by_msa):
            if depth < len(rows):
                plan[output_row, column] = rows[depth]
                output_row += 1
    return plan


def msa_bindings(msa: MSA | MsaCollection) -> tuple[tuple[MSA, str | None], ...]:
    r"""
    Return each MSA column with its instance target.

    Args:
        msa: One MSA or a collection of paired MSA columns.

    Returns:
        bindings (tuple[tuple[MSA, str | None], ...]):
            Pairs of MSA columns and exact instance IDs; an instance ID of `None`
            denotes entity-wide broadcast.
    """

    columns = (msa,) if isinstance(msa, MSA) else msa.msas
    targets = (
        (None,) * len(columns) if isinstance(msa, MSA) or msa.target_instance_ids is None else msa.target_instance_ids
    )
    return tuple(zip(columns, targets))


def resolve_msa_bindings(
    msa: MSA | MsaCollection,
    system: MolecularSystem,
) -> tuple[tuple[MSA, int | None, int], ...]:
    r"""
    Resolve MSA entity and instance references against a molecular system.

    Args:
        msa: One MSA or a collection of paired MSA columns.
        system: Molecular system containing the referenced entities and instances.

    Returns:
        Resolved MSA columns with their optional target instance index and entity index.

    Raises:
        ValueError: If a referenced entity or instance is invalid, an instance belongs
            to another entity, an MSA width differs from its polymer length, or the MSA
            query row disagrees with the target entity sequence.
    """

    resolved: list[tuple[MSA, int | None, int]] = []
    for column, target_instance_id in msa_bindings(msa):
        try:
            entity_index = system.entity_index(column.entity_id)
        except KeyError:
            raise ValueError(f"MSA entity {column.entity_id!r} is not a polymer entity in the input system") from None
        if int(system.entity_type[entity_index]) == ENTITY_MOLECULE:
            raise ValueError(f"MSA entity {column.entity_id!r} is not a polymer entity in the input system")
        target_instance_index = None
        if target_instance_id is not None:
            try:
                target_instance_index = system.instance_index(target_instance_id)
            except KeyError:
                raise ValueError(f"MSA targets unknown input instance {target_instance_id!r}")
            target_entity_index = int(system.instance_entity[target_instance_index])
            if target_entity_index != entity_index:
                raise ValueError(
                    f"MSA target instance {target_instance_id!r} belongs to entity "
                    f"{system.entity_id(target_entity_index)!r}, not {column.entity_id!r}"
                )
        if target_instance_index is None:
            target_instances = [
                int(index)
                for index in (system.instance_entity == entity_index).nonzero(as_tuple=False).flatten().tolist()
            ]
        else:
            target_instances = [target_instance_index]
        query = column.unpaired_sequences[0]
        width = len(query)
        for instance_index in target_instances:
            start, stop = _instance_component_bounds(system, instance_index)
            component_count = stop - start
            if width != component_count:
                raise ValueError(
                    f"MSA for entity {column.entity_id!r} spans {width} columns "
                    f"but the entity has {component_count} components"
                )
            conflict = _query_conflicts_with_instance(query, system, instance_index)
            if conflict is not None:
                raise ValueError(
                    f"MSA query for entity {column.entity_id!r} disagrees with the target sequence at column {conflict}"
                )
        resolved.append((column, target_instance_index, entity_index))
    return tuple(resolved)


@dataclass(frozen=True)
class TemplateAlignment:
    r"""
    A target-to-template polymer alignment.

    Attributes:
        target_entity_id: Target entity ID in the predicted molecular system.
        target_instance_id: Optional exact target instance; ``None`` broadcasts the
            alignment to all instances of the target entity.
        template_instance_id: Polymer instance in the template coordinate evidence.
        residue_pairs: Strictly increasing zero-based ``(target, template)`` component
            index pairs.

    Raises:
        TypeError: If a residue position is boolean or does not support integer indexing.
        ValueError: If instance identifiers or residue pairs are invalid.

    Examples:
        >>> alignment = TemplateAlignment("query", None, "A", ((0, 2), (1, 3)))
        >>> alignment.residue_pairs
        ((0, 2), (1, 3))
    """

    target_entity_id: str
    target_instance_id: str | None
    template_instance_id: str
    residue_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        pairs = tuple(tuple(pair) for pair in self.residue_pairs)
        if any(isinstance(index, (bool, np.bool_)) for pair in pairs for index in pair):
            raise TypeError("residue pair positions must be integers, not booleans")
        pairs = tuple(tuple(operator.index(index) for index in pair) for pair in pairs)
        object.__setattr__(self, "residue_pairs", pairs)
        if self.target_instance_id == "":
            raise ValueError("target_instance_id must be non-empty or None")
        if not self.template_instance_id:
            raise ValueError("template_instance_id must be non-empty")
        if not self.residue_pairs:
            raise ValueError("residue_pairs must not be empty")

        previous_target = -1
        previous_template = -1
        for target_index, template_index in self.residue_pairs:
            if target_index < 0 or template_index < 0:
                raise ValueError("residue pair indices must be non-negative")
            if target_index <= previous_target or template_index <= previous_template:
                raise ValueError("residue_pairs must be strictly increasing on both axes")
            previous_target = target_index
            previous_template = template_index


@dataclass(frozen=True, eq=False)
class Template:
    r"""
    Structural template evidence in a shared coordinate frame.

    Attributes:
        coordinates: Immutable [`PolymerCoordinates`][ofoldx.data.coordinates.PolymerCoordinates]
            containing the referenced polymers. This value is shared without copying.
            Use [`from_structure`][ofoldx.data.evidence.Template.from_structure] to
            take an explicit snapshot of a mutable Structure.
        alignments: Target-to-template polymer alignments sharing ``coordinates``. Two
            alignments may cover disjoint target ranges, and may place the same
            template position on the same target position, but they must not place
            *different* template positions on one target position: a consumer lowers
            alignments in order, so such a template would depend on their ordering.
            A broadcast alignment covers every instance of its target entity, so it
            overlaps an alignment naming one exact instance of that entity.

    Raises:
        TypeError: If coordinates is not a PolymerCoordinates value.
        ValueError: If no alignment is supplied, an alignment references an invalid
            template polymer component, or two alignments place different template
            positions on one target position.

    Examples:
        >>> source = Structure.from_sequence("AG")
        >>> alignment = TemplateAlignment("query", None, "A", ((0, 0), (1, 1)))
        >>> template = Template.from_structure(source, (alignment,))
        >>> source.coordinate_mask[:] = True
        >>> bool(template.coordinates.chains["A"].coordinate_mask.any())
        False
    """

    coordinates: PolymerCoordinates
    alignments: tuple[TemplateAlignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, PolymerCoordinates):
            raise TypeError("Template coordinates must be PolymerCoordinates")
        object.__setattr__(self, "alignments", tuple(self.alignments))
        if not self.alignments:
            raise ValueError("alignments must not be empty")
        coordinates = self.coordinates
        placements: dict[tuple[str, int], dict[str | None, tuple[str, int]]] = {}
        for alignment in self.alignments:
            chain = coordinates.chains.get(alignment.template_instance_id)
            if chain is None:
                raise ValueError("template_instance_id must reference a polymer instance")
            if alignment.residue_pairs[-1][1] >= len(chain.component_ids):
                raise ValueError("residue pair template index is outside the template instance")
            for target_index, template_index in alignment.residue_pairs:
                source = (alignment.template_instance_id, template_index)
                covered = placements.setdefault((alignment.target_entity_id, target_index), {})
                for instance_id, placed in covered.items():
                    if placed != source and (
                        instance_id is None or alignment.target_instance_id in (None, instance_id)
                    ):
                        raise ValueError(
                            f"template alignments place {placed[0]!r}:{placed[1]} and "
                            f"{source[0]!r}:{source[1]} on the same target position "
                            f"{alignment.target_entity_id!r}:{target_index}"
                        )
                covered[alignment.target_instance_id] = source

    @classmethod
    def from_structure(cls, structure: Structure, alignments: Sequence[TemplateAlignment]) -> Template:
        """Snapshot a mutable Structure once and bind its template alignments.

        Args:
            structure: One molecular system in a shared coordinate frame.
            alignments: Target-to-template component pairs.

        Returns:
            A template owning stable coordinate evidence, with no live Structure reference.

        Examples:
            >>> source = Structure.from_sequence("A")
            >>> template = Template.from_structure(source, [TemplateAlignment("query", None, "A", ((0, 0),))])
            >>> template.coordinates.chains["A"].component_ids
            ('ALA',)
        """
        return cls(PolymerCoordinates.from_structure(structure), tuple(alignments))


def template_atoms(chain: ChainCoordinates, position: int, atom_names: Sequence[str | None]) -> tuple[Tensor, Tensor]:
    """Project named observations into independent CPU tensors for a model consumer.

    Args:
        chain: Immutable coordinate evidence for one polymer instance.
        position: Zero-based sequence position.
        atom_names: Dense atom layout, with ``None`` for unused slots.

    Returns:
        coordinates: Coordinates preserving the source dtype. Model consumers choose
            their geometry calculation dtype explicitly.
        mask: Float32 validity mask.

    Examples:
        >>> chain = PolymerCoordinates.from_structure(Structure.from_sequence("A")).chains["A"]
        >>> xyz, mask = template_atoms(chain, 0, ("CA", None))
        >>> tuple(xyz.shape), mask.tolist()
        ((2, 3), [0.0, 0.0])
    """
    coordinates, mask = chain.gather((position,), atom_names)
    return torch.from_numpy(coordinates[0]), torch.from_numpy(mask[0]).float()


def _encode_template_coordinates(coordinates: PolymerCoordinates) -> tuple[dict[str, Tensor], list[dict]]:
    state, descriptors = {}, []
    for index, (identifier, chain) in enumerate(coordinates.chains.items()):
        descriptors.append(
            {
                "instance_id": identifier,
                "polymer_type": chain.polymer_type,
                "component_ids": chain.component_ids,
                "atom_names": chain.atom_names,
                "parent_component_ids": chain.parent_component_ids,
            }
        )
        for field in ("component_index", "coordinates", "coordinate_mask"):
            state[f"chain.{index}.{field}"] = torch.from_numpy(getattr(chain, field).copy())
    return state, descriptors


def _decode_template_coordinates(state: dict[str, Tensor], descriptors: list[dict]) -> PolymerCoordinates:
    if not isinstance(descriptors, list):
        raise ValueError("invalid template coordinate chain metadata")
    chains = {}
    expected = set()
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "instance_id",
            "polymer_type",
            "component_ids",
            "atom_names",
            "parent_component_ids",
        }:
            raise ValueError("invalid template coordinate chain metadata")
        identifier = descriptor["instance_id"]
        if not isinstance(identifier, str) or identifier in chains:
            raise ValueError("invalid or duplicate template coordinate chain identifier")
        fields = {}
        for field, dtype in (
            ("component_index", torch.int64),
            ("coordinates", torch.float32),
            ("coordinate_mask", torch.bool),
        ):
            key = f"chain.{index}.{field}"
            tensor = state.get(key)
            if tensor is None or (
                tensor.dtype not in (torch.float32, torch.float64) if field == "coordinates" else tensor.dtype != dtype
            ):
                raise ValueError(f"invalid template coordinate tensor {key!r}")
            fields[field] = tensor.numpy()
            expected.add(key)
        chains[identifier] = ChainCoordinates(
            polymer_type=descriptor["polymer_type"],
            component_ids=descriptor["component_ids"],
            atom_names=descriptor["atom_names"],
            parent_component_ids=descriptor["parent_component_ids"],
            component_index=fields["component_index"],
            coordinates=fields["coordinates"],
            coordinate_mask=fields["coordinate_mask"],
        )
    if state.keys() != expected:
        raise ValueError("unexpected template coordinate tensors")
    return PolymerCoordinates(chains)


def save_templates(templates: Sequence[Template], path: str | os.PathLike[str]) -> None:
    r"""Save immutable coordinate evidence and alignments as one template artifact.

    Args:
        templates: Templates whose coordinate frames, full parent declarations,
            stored precision, and alignments are preserved.
        path: Destination ending in ``.templates.safetensors``.

    Raises:
        ValueError: The destination does not have the required artifact suffix.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "empty.templates.safetensors"
        ...     save_templates((), path)
        ...     load_templates(path)
        ()
    """

    destination = _template_artifact_path(path)
    template_alignment_ptr = [0]
    alignment_residue_pair_ptr = [0]
    residue_pairs: list[tuple[int, int]] = []
    target_entity_ids = []
    target_instance_ids = []
    template_instance_ids = []
    coordinate_chains = []
    state: dict[str, Tensor] = {}

    for template_index, template in enumerate(templates):
        prefix = f"template.{template_index}."
        encoded, descriptors = _encode_template_coordinates(template.coordinates)
        coordinate_chains.append(descriptors)
        state.update({f"{prefix}{key}": value for key, value in encoded.items()})
        for alignment in template.alignments:
            target_entity_ids.append(alignment.target_entity_id)
            target_instance_ids.append(alignment.target_instance_id)
            template_instance_ids.append(alignment.template_instance_id)
            residue_pairs.extend(alignment.residue_pairs)
            alignment_residue_pair_ptr.append(len(residue_pairs))
        template_alignment_ptr.append(len(target_entity_ids))

    state.update(
        {
            "template_alignment_ptr": torch.tensor(template_alignment_ptr, dtype=torch.int64),
            "alignment_residue_pair_ptr": torch.tensor(alignment_residue_pair_ptr, dtype=torch.int64),
            "alignment_residue_pairs": torch.tensor(residue_pairs, dtype=torch.int64).reshape(-1, 2),
        }
    )
    save_file(
        state,
        str(destination),
        metadata={
            "format": _TEMPLATE_ARTIFACT_FORMAT,
            "template_coordinate_chains": json.dumps(coordinate_chains, ensure_ascii=False),
            "alignment_target_entity_ids": json.dumps(target_entity_ids, ensure_ascii=False),
            "alignment_target_instance_ids": json.dumps(target_instance_ids, ensure_ascii=False),
            "alignment_template_instance_ids": json.dumps(template_instance_ids, ensure_ascii=False),
        },
    )


def load_templates(path: str | os.PathLike[str]) -> tuple[Template, ...]:
    r"""Load immutable coordinate evidence and alignments from a template artifact.

    Args:
        path: Source ending in ``.templates.safetensors``.

    Returns:
        templates (tuple[Template, ...]):
            Stable [`Template`][ofoldx.data.evidence.Template] values with complete parent
            tuples, stored coordinate precision, and target-to-template component pairs.

    Raises:
        ValueError: The artifact suffix, format, metadata, or tensor schema is invalid.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> template = Template.from_structure(
        ...     Structure.from_sequence("A"), (TemplateAlignment("query", None, "A", ((0, 0),)),)
        ... )
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "example.templates.safetensors"
        ...     save_templates((template,), path)
        ...     load_templates(path)[0].coordinates.chains["A"].component_ids
        ('ALA',)
    """

    source = _template_artifact_path(path)
    with safe_open(str(source), framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata()
        if metadata.get("format") != _TEMPLATE_ARTIFACT_FORMAT:
            raise ValueError(f"{source} is not an OFoldX template artifact")
        state = {key: artifact.get_tensor(key) for key in artifact.keys()}

    coordinate_chains = json.loads(metadata["template_coordinate_chains"])
    if not isinstance(coordinate_chains, list) or any(not isinstance(chains, list) for chains in coordinate_chains):
        raise ValueError(f"invalid template coordinate metadata in {source}")
    target_entity_ids = cast(
        tuple[str, ...],
        _template_identifiers(metadata, "alignment_target_entity_ids", source),
    )
    target_instance_ids = _template_identifiers(
        metadata,
        "alignment_target_instance_ids",
        source,
        optional=True,
    )
    template_instance_ids = cast(
        tuple[str, ...],
        _template_identifiers(metadata, "alignment_template_instance_ids", source),
    )
    template_count = len(coordinate_chains)
    alignment_count = len(target_entity_ids)
    if len(target_instance_ids) != alignment_count or len(template_instance_ids) != alignment_count:
        raise ValueError(f"invalid template alignment identifiers in {source}")

    required_keys = {
        "template_alignment_ptr",
        "alignment_residue_pair_ptr",
        "alignment_residue_pairs",
    }
    if not required_keys <= state.keys():
        raise ValueError(f"{source} does not contain the complete template tensor schema")
    template_alignment_ptr = state["template_alignment_ptr"]
    alignment_residue_pair_ptr = state["alignment_residue_pair_ptr"]
    alignment_residue_pairs = state["alignment_residue_pairs"]
    if (
        template_alignment_ptr.dtype != torch.int64
        or alignment_residue_pair_ptr.dtype != torch.int64
        or alignment_residue_pairs.dtype != torch.int64
    ):
        raise ValueError(f"{source} contains an invalid template tensor dtype")
    if not _valid_pointer(template_alignment_ptr, template_count, alignment_count):
        raise ValueError(f"{source} contains an invalid template_alignment_ptr")
    if alignment_residue_pairs.ndim != 2 or alignment_residue_pairs.shape[1] != 2:
        raise ValueError(f"{source} contains invalid template residue pairs")
    if not _valid_pointer(alignment_residue_pair_ptr, alignment_count, alignment_residue_pairs.shape[0]):
        raise ValueError(f"{source} contains an invalid alignment_residue_pair_ptr")

    coordinate_values = []
    consumed_keys = set(required_keys)
    for template_index, descriptors in enumerate(coordinate_chains):
        prefix = f"template.{template_index}."
        coordinate_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        coordinate_values.append(_decode_template_coordinates(coordinate_state, descriptors))
        consumed_keys.update(f"{prefix}{key}" for key in coordinate_state)
    if state.keys() != consumed_keys:
        raise ValueError(f"{source} does not contain the complete template tensor schema")

    templates = []
    for template_index, coordinates in enumerate(coordinate_values):
        alignments = []
        start = int(template_alignment_ptr[template_index])
        stop = int(template_alignment_ptr[template_index + 1])
        for alignment_index in range(start, stop):
            pair_start = int(alignment_residue_pair_ptr[alignment_index])
            pair_stop = int(alignment_residue_pair_ptr[alignment_index + 1])
            alignments.append(
                TemplateAlignment(
                    target_entity_id=target_entity_ids[alignment_index],
                    target_instance_id=target_instance_ids[alignment_index],
                    template_instance_id=template_instance_ids[alignment_index],
                    residue_pairs=tuple(map(tuple, alignment_residue_pairs[pair_start:pair_stop].tolist())),
                )
            )
        templates.append(Template(coordinates=coordinates, alignments=tuple(alignments)))
    return tuple(templates)


def _template_artifact_path(path: str | os.PathLike[str]) -> Path:
    artifact = Path(path)
    if not artifact.name.endswith(TEMPLATE_ARTIFACT_SUFFIX):
        raise ValueError(f"template artifacts must end with {TEMPLATE_ARTIFACT_SUFFIX!r}")
    return artifact


def _template_identifiers(
    metadata: dict[str, str],
    key: str,
    path: Path,
    *,
    optional: bool = False,
) -> tuple[str | None, ...]:
    values = json.loads(metadata[key])
    if not isinstance(values, list) or not all(
        isinstance(value, str) or (optional and value is None) for value in values
    ):
        raise ValueError(f"invalid {key!r} metadata in {path}")
    return tuple(values)


def _valid_pointer(pointer: Tensor, item_count: int, value_count: int) -> bool:
    return bool(
        pointer.ndim == 1
        and pointer.shape[0] == item_count + 1
        and int(pointer[0]) == 0
        and not bool((pointer[1:] < pointer[:-1]).any())
        and int(pointer[-1]) == value_count
    )
