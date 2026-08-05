r"""Protenix-style asymmetric-unit structure indices and weighted samplers.

Each chain or interface row points back to one complete CIF record and carries
the exact canonical instance IDs used as crop-centre candidates. Different rows
for one CIF can therefore produce different Protenix training crops. Index
construction eagerly scans every CIF. Default structure ingestion removes
hydrogens and water. Configured ion handling and ``exclude_mol_types`` affect
sampling rows only, not the retained full structure or full-axis target.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Sampler

from ..data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MOLECULE_WATER,
    STANDARD_AMINO_ACID_IDS,
)
from .ion_components import PROTENIX_ION_COMPONENT_IDS

_STANDARD_PROTEIN_COMPONENT_IDS = STANDARD_AMINO_ACID_IDS | {"UNK"}
_STANDARD_RNA_COMPONENT_IDS = frozenset({"A", "G", "C", "U", "N"})
_STANDARD_DNA_COMPONENT_IDS = frozenset({"DA", "DG", "DC", "DT", "DN"})
_CACHE_SCHEMA = "ofoldx.cluster_indices"
_CACHE_SCHEMA_VERSION = 2
_CACHE_GENERATOR_VERSION = 2
_STRUCTURE_INGESTION_POLICY = "ccd-heavy-no-water-random-polymer-microheterogeneity-v3"
_METADATA_FIELDS = {
    "schema",
    "schema_version",
    "generator_version",
    "fingerprint",
    "record_count",
    "entry_count",
    "payload_sha256",
}


def parse_pdb_cluster_file(path: str | os.PathLike[str]) -> dict[str, tuple[str, int]]:
    r"""Parse an RCSB entity-cluster file into lower-case member lookups."""

    clusters: dict[str, tuple[str, int]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            members = [
                member
                for member in line.split()
                if not member.startswith(("AF_", "MA_"))
            ]
            if not members:
                continue
            cluster_id = f"pdb_cluster_{members[0]}"
            for member in members:
                clusters[member.lower()] = (cluster_id, len(members))
    return clusters


@dataclass(frozen=True)
class ClusterIndex:
    r"""One Protenix-style sampling row mapped back to a full structure record."""

    type: str
    pdb_id: str
    record_index: int
    entity_1_id: str
    entity_2_id: str
    chain_1_id: str
    chain_2_id: str
    mol_1_type: str
    mol_2_type: str
    cluster_1_id: str
    cluster_2_id: str
    cluster_id: str
    chain_key: str
    sample_member: str = ""
    cluster_size: int = 0
    member_num: int = 0
    weight: float = 0.0
    reference_instance_ids: tuple[str, ...] = ()


_ENTRY_FIELDS = frozenset(ClusterIndex.__dataclass_fields__)
_ENTRY_STRING_FIELDS = _ENTRY_FIELDS - {
    "record_index",
    "cluster_size",
    "member_num",
    "weight",
    "reference_instance_ids",
}


class _InvalidClusterIndicesCache(ValueError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _effective_record(record: str | Mapping[str, Any]) -> dict[str, Any]:
    r"""Return low-cost file identity metadata without reading structure contents."""

    path = Path(record if isinstance(record, str) else os.fspath(record["structure"])).expanduser().resolve()
    pdb_id = str(
        record.get("pdb_id", _cif_record_id(path)) if isinstance(record, Mapping) else _cif_record_id(path)
    ).lower()
    assembly_id = record.get("assembly_id") if isinstance(record, Mapping) else None
    stat = path.stat()
    return {
        "structure": str(path),
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def cluster_indices_fingerprint(
    records: Sequence[str | Mapping[str, Any]],
    cluster_file: str | os.PathLike[str],
    *,
    contact_cutoff: float = 5.0,
    include_all_chain_pairs: bool = False,
    exclude_mol_types: Sequence[str] = ("ions",),
    beta: Mapping[str, float] | None = None,
    alpha: Mapping[str, float] | None = None,
    ccd_revision: str | None = None,
    ccd_snapshot_path: str | os.PathLike[str] | None = None,
) -> str:
    r"""Fingerprint index settings and low-cost structure file identities."""

    if ccd_revision is not None and ccd_snapshot_path is not None:
        raise ValueError("configure at most one of ccd_revision or ccd_snapshot_path")

    cluster_path = Path(cluster_file).expanduser().resolve()
    payload = {
        "schema": _CACHE_SCHEMA,
        "schema_version": _CACHE_SCHEMA_VERSION,
        "generator_version": _CACHE_GENERATOR_VERSION,
        "structure_ingestion_policy": _STRUCTURE_INGESTION_POLICY,
        "ccd_revision": ccd_revision,
        "ccd_snapshot_path": (
            None if ccd_snapshot_path is None else str(Path(ccd_snapshot_path).expanduser().resolve())
        ),
        "records": [_effective_record(record) for record in records],
        "cluster_file": str(cluster_path),
        "cluster_sha256": hashlib.sha256(cluster_path.read_bytes()).hexdigest(),
        "contact_cutoff": float(contact_cutoff),
        "include_all_chain_pairs": bool(include_all_chain_pairs),
        "exclude_mol_types": sorted(set(exclude_mol_types)),
        "beta": sorted(
            (str(key), float(value))
            for key, value in ({"chain": 0.5, "interface": 1.0} if beta is None else beta).items()
        ),
        "alpha": sorted(
            (str(key), float(value))
            for key, value in (
                {"prot": 3.0, "nuc": 3.0, "ligand": 1.0} if alpha is None else alpha
            ).items()
        ),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def default_cluster_indices_path(
    workspace_root: str | os.PathLike[str],
    source: Any,
    cluster_file: str | os.PathLike[str],
    *,
    contact_cutoff: float = 5.0,
    include_all_chain_pairs: bool = False,
    exclude_mol_types: Sequence[str] = ("ions",),
    beta: Mapping[str, float] | None = None,
    alpha: Mapping[str, float] | None = None,
    ccd_revision: str | None = None,
    ccd_snapshot_path: str | os.PathLike[str] | None = None,
) -> Path:
    r"""Return the shared cache location for one semantic index locator."""

    if ccd_revision is not None and ccd_snapshot_path is not None:
        raise ValueError("configure at most one of ccd_revision or ccd_snapshot_path")

    source_locator = os.path.abspath(os.path.expanduser(os.fspath(source)))
    locator = {
        "generator_version": _CACHE_GENERATOR_VERSION,
        "structure_ingestion_policy": _STRUCTURE_INGESTION_POLICY,
        "ccd_revision": ccd_revision,
        "ccd_snapshot_path": (
            None if ccd_snapshot_path is None else str(Path(ccd_snapshot_path).expanduser().resolve())
        ),
        "source": os.path.normpath(source_locator),
        "cluster_file": str(Path(cluster_file).expanduser().resolve()),
        "contact_cutoff": float(contact_cutoff),
        "include_all_chain_pairs": bool(include_all_chain_pairs),
        "exclude_mol_types": sorted(set(exclude_mol_types)),
        "beta": sorted(
            (str(key), float(value))
            for key, value in ({"chain": 0.5, "interface": 1.0} if beta is None else beta).items()
        ),
        "alpha": sorted(
            (str(key), float(value))
            for key, value in (
                {"prot": 3.0, "nuc": 3.0, "ligand": 1.0} if alpha is None else alpha
            ).items()
        ),
    }
    digest = hashlib.sha256(_canonical_json(locator)).hexdigest()
    return Path(workspace_root).expanduser().resolve() / ".cache" / "ofoldx" / "cluster-indices" / f"v2-{digest}.jsonl"


def _decode_entry(payload: Any, record_count: int) -> tuple[ClusterIndex, float]:
    if not isinstance(payload, dict) or set(payload) != {"entry", "weight"}:
        raise _InvalidClusterIndicesCache("cache entry must contain exactly entry and weight")
    entry_payload, weight = payload["entry"], payload["weight"]
    if not isinstance(entry_payload, dict) or set(entry_payload) != _ENTRY_FIELDS:
        raise _InvalidClusterIndicesCache("cache ClusterIndex fields do not match the schema")
    if any(not isinstance(entry_payload[field], str) for field in _ENTRY_STRING_FIELDS):
        raise _InvalidClusterIndicesCache("cache ClusterIndex string field has an invalid type")
    if entry_payload["type"] not in {"chain", "interface"}:
        raise _InvalidClusterIndicesCache("cache ClusterIndex type must be chain or interface")
    for field in ("record_index", "cluster_size", "member_num"):
        if not isinstance(entry_payload[field], int) or isinstance(entry_payload[field], bool):
            raise _InvalidClusterIndicesCache(f"cache ClusterIndex {field} must be an integer")
    record_index = entry_payload["record_index"]
    if not 0 <= record_index < record_count:
        raise _InvalidClusterIndicesCache("cache ClusterIndex record_index is out of range")
    if entry_payload["cluster_size"] <= 0 or entry_payload["member_num"] <= 0:
        raise _InvalidClusterIndicesCache("cache weighted ClusterIndex counts must be positive")
    references = entry_payload["reference_instance_ids"]
    if not isinstance(references, list) or any(not isinstance(value, str) for value in references):
        raise _InvalidClusterIndicesCache("cache reference_instance_ids must be a list of strings")
    for value in (entry_payload["weight"], weight):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _InvalidClusterIndicesCache("cache weight must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise _InvalidClusterIndicesCache("cache weight must be finite and non-negative")
    if float(entry_payload["weight"]) != float(weight):
        raise _InvalidClusterIndicesCache("cache entry weight does not match the outer weight")
    entry_payload = dict(entry_payload)
    entry_payload["weight"] = float(entry_payload["weight"])
    entry_payload["reference_instance_ids"] = tuple(references)
    return ClusterIndex(**entry_payload), float(weight)


def _load_cluster_indices(path: Path, fingerprint: str, record_count: int) -> tuple[list[ClusterIndex], list[float]]:
    try:
        with path.open("rb") as handle:
            metadata = json.loads(handle.readline())
            if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
                raise _InvalidClusterIndicesCache("cache metadata fields do not match the schema")
            if (
                not isinstance(metadata["schema"], str)
                or not isinstance(metadata["fingerprint"], str)
                or not isinstance(metadata["payload_sha256"], str)
                or any(
                    not isinstance(metadata[field], int) or isinstance(metadata[field], bool)
                    for field in ("schema_version", "generator_version", "record_count", "entry_count")
                )
            ):
                raise _InvalidClusterIndicesCache("cache metadata has invalid field types")
            expected = {
                "schema": _CACHE_SCHEMA,
                "schema_version": _CACHE_SCHEMA_VERSION,
                "generator_version": _CACHE_GENERATOR_VERSION,
                "fingerprint": fingerprint,
                "record_count": record_count,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise _InvalidClusterIndicesCache("cache metadata does not match current inputs")
            entry_count = metadata["entry_count"]
            if not isinstance(entry_count, int) or isinstance(entry_count, bool) or entry_count < 0:
                raise _InvalidClusterIndicesCache("cache entry_count must be a non-negative integer")
            decoded = []
            payload_hash = hashlib.sha256()
            for raw_line in handle:
                payload_hash.update(raw_line)
                line = raw_line.decode("utf-8")
                if not line.strip():
                    raise _InvalidClusterIndicesCache("cache contains an empty entry row")
                decoded.append(_decode_entry(json.loads(line), record_count))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise _InvalidClusterIndicesCache(f"could not decode cluster indices cache: {error}") from error
    if len(decoded) != entry_count:
        raise _InvalidClusterIndicesCache("cache entry_count does not match its rows")
    if payload_hash.hexdigest() != metadata["payload_sha256"]:
        raise _InvalidClusterIndicesCache("cache payload checksum does not match its rows")
    entries = [item[0] for item in decoded]
    weights = [item[1] for item in decoded]
    weight_sum = sum(weights)
    if not weights or not math.isfinite(weight_sum) or weight_sum <= 0:
        raise _InvalidClusterIndicesCache("cache weights must have a finite positive sum")
    return entries, weights


def _write_cluster_indices(
    path: Path,
    fingerprint: str,
    record_count: int,
    entries: Sequence[ClusterIndex],
    weights: Sequence[float],
) -> None:
    if len(entries) != len(weights):
        raise ValueError("entries and weights must have equal length")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        payload_hash = hashlib.sha256()
        for entry, weight in zip(entries, weights):
            entry_payload = {field: getattr(entry, field) for field in _ENTRY_FIELDS}
            entry_payload["reference_instance_ids"] = list(entry.reference_instance_ids)
            payload = {"entry": entry_payload, "weight": weight}
            encoded = _canonical_json(payload) + b"\n"
            payload_hash.update(encoded)
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            temporary_name = handle.name
            metadata = {
                "schema": _CACHE_SCHEMA,
                "schema_version": _CACHE_SCHEMA_VERSION,
                "generator_version": _CACHE_GENERATOR_VERSION,
                "fingerprint": fingerprint,
                "record_count": record_count,
                "entry_count": len(entries),
                "payload_sha256": payload_hash.hexdigest(),
            }
            handle.write(_canonical_json(metadata) + b"\n")
            for entry, weight in zip(entries, weights):
                entry_payload = {field: getattr(entry, field) for field in _ENTRY_FIELDS}
                entry_payload["reference_instance_ids"] = list(entry.reference_instance_ids)
                handle.write(_canonical_json({"entry": entry_payload, "weight": weight}) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_or_build_cluster_indices(
    records: Sequence[str | Mapping[str, Any]],
    cluster_file: str | os.PathLike[str],
    indices_file: str | os.PathLike[str],
    *,
    contact_cutoff: float = 5.0,
    include_all_chain_pairs: bool = False,
    exclude_mol_types: Sequence[str] = ("ions",),
    beta: Mapping[str, float] | None = None,
    alpha: Mapping[str, float] | None = None,
    ccd_revision: str | None = None,
    ccd_snapshot_path: str | os.PathLike[str] | None = None,
) -> tuple[list[ClusterIndex], list[float], str]:
    r"""Load valid weighted indices, or atomically build and publish them."""

    import fcntl

    path = Path(indices_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = cluster_indices_fingerprint(
        records,
        cluster_file,
        contact_cutoff=contact_cutoff,
        include_all_chain_pairs=include_all_chain_pairs,
        exclude_mol_types=exclude_mol_types,
        beta=beta,
        alpha=alpha,
        ccd_revision=ccd_revision,
        ccd_snapshot_path=ccd_snapshot_path,
    )
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existed = path.exists()
        if existed:
            try:
                entries, weights = _load_cluster_indices(path, fingerprint, len(records))
                return entries, weights, "loaded"
            except _InvalidClusterIndicesCache:
                pass
        raw_entries = make_cluster_indices(
            records,
            cluster_file,
            contact_cutoff=contact_cutoff,
            include_all_chain_pairs=include_all_chain_pairs,
            exclude_mol_types=exclude_mol_types,
            ccd_revision=ccd_revision,
            ccd_snapshot_path=ccd_snapshot_path,
        )
        entries, weights = calculate_cluster_weights(raw_entries, beta=beta, alpha=alpha)
        if not entries:
            raise ValueError("cluster sampling produced no valid chain or interface index rows")
        weight_sum = sum(weights)
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("cluster sampling weights must have a finite positive sum")
        _write_cluster_indices(path, fingerprint, len(records), entries, weights)
        return entries, weights, "rebuilt" if existed else "created"


def _polymer_sequence(entity_type: int, component_ids: Sequence[str]) -> str:
    if entity_type == ENTITY_PROTEIN:
        return "".join(AMINO_ACID_THREE_TO_ONE.get(component_id, "X") for component_id in component_ids)
    if entity_type == ENTITY_RNA:
        mapping = {"A": "A", "C": "C", "G": "G", "U": "U", "I": "I", "N": "N"}
    else:
        mapping = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U", "DI": "I", "DN": "N"}
    return "".join(mapping.get(component_id, "N") for component_id in component_ids)


def _cluster_id(
    pdb_id: str,
    source_entity_id: str,
    entity_type: int,
    component_ids: Sequence[str],
    clusters: Mapping[str, tuple[str, int]],
) -> tuple[str, str]:
    if entity_type != ENTITY_MOLECULE:
        sequence = _polymer_sequence(entity_type, component_ids)
        if entity_type == ENTITY_PROTEIN:
            mol_type = "prot"
            if len(sequence) < 10:
                return mol_type, sequence
            found = clusters.get(f"{pdb_id}_{source_entity_id}".lower())
            if found is not None:
                return mol_type, found[0]
            if sequence == "X" * len(sequence):
                return mol_type, "poly_UNK" if set(component_ids) == {"UNK"} else "_".join(component_ids)
            return mol_type, "NotInClusterTxt"
        return "nuc", sequence

    mol_type = (
        "ions"
        if component_ids and all(component_id in PROTENIX_ION_COMPONENT_IDS for component_id in component_ids)
        else "ligand"
    )
    return mol_type, "_".join(component_ids)


def _is_center_site(entity_type: int, component_id: str, atom_id: str) -> bool:
    if entity_type == ENTITY_MOLECULE:
        return True
    if entity_type == ENTITY_PROTEIN and component_id in _STANDARD_PROTEIN_COMPONENT_IDS:
        return atom_id == "CA"
    if component_id in _STANDARD_RNA_COMPONENT_IDS | _STANDARD_DNA_COMPONENT_IDS:
        return atom_id in {"C1'", "C1*"}
    # Modified polymer residues are atom-tokenized, so every resolved heavy atom
    # is a centre token.
    return True


def _source_chain_id(instance_id: str, is_molecule: bool, first_component_id: str) -> str:
    r"""Reverse the reader's ``chain:residue:CCD`` non-polymer instance encoding."""

    if not is_molecule:
        return instance_id
    encoded = instance_id.rsplit(":", 2)
    if len(encoded) != 3:
        return instance_id
    source_chain_id, _residue_identity, encoded_component = encoded
    if encoded_component == first_component_id or re.fullmatch(
        rf"{re.escape(first_component_id)}_\d+",
        encoded_component,
    ):
        return source_chain_id
    return instance_id


def _cif_record_id(path: Path) -> str:
    """Return the entry stem while ignoring one optional gzip suffix."""

    source = path.with_suffix("") if path.suffix.lower() == ".gz" else path
    return source.stem


def _source_asym_entities(path: Path) -> dict[str, str]:
    """Read the source `_struct_asym` mapping independently of internal entity realization."""

    from biotite.structure.io import pdbx

    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt") as handle:
            cif_file = pdbx.CIFFile.read(handle)
    else:
        cif_file = pdbx.CIFFile.read(str(path))
    block = cif_file.block
    if "struct_asym" not in block:
        return {}
    category = block["struct_asym"]
    asym_ids = category["id"].as_array().tolist()
    entity_ids = category["entity_id"].as_array().tolist()
    mapping: dict[str, str] = {}
    for raw_asym_id, raw_entity_id in zip(asym_ids, entity_ids):
        asym_id, entity_id = str(raw_asym_id), str(raw_entity_id)
        previous = mapping.get(asym_id)
        if previous is not None and previous != entity_id:
            raise ValueError(f"mmCIF _struct_asym has conflicting entity for {asym_id!r}")
        mapping[asym_id] = entity_id
    return mapping


def _scan_resolved_atoms(structure: Any) -> tuple[dict[str, np.ndarray], set[str]]:
    r"""Group resolved heavy coordinates and centre eligibility in one site pass."""

    system = structure.system
    component_count = system.component_symbol.numel()
    atom_component = torch.repeat_interleave(
        torch.arange(component_count, dtype=torch.int64, device=system.component_symbol.device),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    instance_count = system.instance_symbol.numel()
    component_instance = torch.repeat_interleave(
        torch.arange(instance_count, dtype=torch.int64, device=system.instance_symbol.device),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    coordinate_indices: dict[str, list[int]] = defaultdict(list)
    center_instances: set[str] = set()
    resolved_heavy = torch.nonzero(structure.coordinate_mask & (system.atomic_number != 1), as_tuple=False).flatten()
    for index in resolved_heavy.detach().cpu().tolist():
        component_index = int(atom_component[index].item())
        instance_index = int(component_instance[component_index].item())
        instance_id = system.instance_id(instance_index)
        component_id = system.symbols[int(system.component_symbol[component_index].item())]
        atom_id = system.symbols[int(system.atom_symbol[index].item())]
        coordinate_indices[instance_id].append(index)
        entity_index = int(system.instance_entity[instance_index].item())
        entity_type = int(system.entity_type[entity_index].item())
        if _is_center_site(entity_type, component_id, atom_id):
            center_instances.add(instance_id)
    coordinates = {
        instance_id: structure.coordinates[indices].detach().cpu().numpy()
        for instance_id, indices in coordinate_indices.items()
    }
    return coordinates, center_instances


def _in_contact(left: np.ndarray, right: np.ndarray, cutoff: float, block_size: int = 256) -> bool:
    cutoff_squared = cutoff * cutoff
    for left_start in range(0, len(left), block_size):
        for right_start in range(0, len(right), block_size):
            left_block = left[left_start : left_start + block_size]
            right_block = right[right_start : right_start + block_size]
            delta = left_block[:, None, :] - right_block[None, :, :]
            if np.any(np.sum(delta * delta, axis=-1) < cutoff_squared):
                return True
    return False


def make_cluster_indices(
    records: Sequence[str | Mapping[str, Any]],
    cluster_file: str | os.PathLike[str],
    *,
    contact_cutoff: float = 5.0,
    include_all_chain_pairs: bool = False,
    exclude_mol_types: Sequence[str] = ("ions",),
    ccd_revision: str | None = None,
    ccd_snapshot_path: str | os.PathLike[str] | None = None,
) -> list[ClusterIndex]:
    r"""Eagerly scan asymmetric-unit CIFs into chain and interface rows."""

    if ccd_revision is not None and ccd_snapshot_path is not None:
        raise ValueError("configure at most one of ccd_revision or ccd_snapshot_path")

    from ..io import read_structure
    from ..io.structure import NoRetainedStructureError

    clusters = parse_pdb_cluster_file(cluster_file)
    entries: list[ClusterIndex] = []
    for record_index, record in enumerate(records):
        path = Path(record if isinstance(record, str) else os.fspath(record["structure"]))
        source_suffix = path.with_suffix("").suffix.lower() if path.suffix.lower() == ".gz" else path.suffix.lower()
        if source_suffix not in {".cif", ".mmcif"}:
            raise ValueError(f"cluster sampling only supports CIF/mmCIF records: {path}")
        pdb_id = str(
            record.get("pdb_id", _cif_record_id(path)) if isinstance(record, Mapping) else _cif_record_id(path)
        ).lower()
        assembly_id = record.get("assembly_id") if isinstance(record, Mapping) else None
        if assembly_id not in (None, ""):
            raise ValueError("cluster sampling supports asymmetric-unit records only; assembly_id must be empty")
        # Unit/integration adapters may supply a mocked read_structure together
        # with a non-existent locator. Preserve the legacy internal-ID fallback
        # only for that case; existing CIF files remain strictly source-validated.
        source_asym_entities = _source_asym_entities(path) if path.is_file() else None
        seed = int.from_bytes(
            hashlib.sha256(str(path.expanduser().resolve()).encode("utf-8")).digest()[:8],
            byteorder="little",
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        try:
            structure = read_structure(
                path,
                ccd_revision=ccd_revision,
                ccd_snapshot_path=ccd_snapshot_path,
                generator=generator,
            )
        except NoRetainedStructureError:
            continue
        internal_coordinates, center_instances = _scan_resolved_atoms(structure)
        system = structure.system
        source_groups: dict[str, dict[str, Any]] = {}
        for instance_index in range(system.instance_symbol.numel()):
            instance_id = system.instance_id(instance_index)
            entity_index = int(system.instance_entity[instance_index].item())
            entity_id = system.entity_id(entity_index)
            entity_type = int(system.entity_type[entity_index].item())
            component_start = int(system.instance_component_ptr[instance_index].item())
            component_stop = int(system.instance_component_ptr[instance_index + 1].item())
            component_ids = [
                system.symbols[int(system.component_symbol[index].item())]
                for index in range(component_start, component_stop)
            ]
            source_chain_id = _source_chain_id(
                instance_id,
                entity_type == ENTITY_MOLECULE,
                component_ids[0] if component_ids else "",
            )
            source_entity_id = (
                entity_id if source_asym_entities is None else source_asym_entities.get(source_chain_id)
            )
            if entity_type == ENTITY_MOLECULE:
                suffixed = re.fullmatch(r"(.+)_(\d+)", source_chain_id)
                source_chain_is_declared = (
                    source_asym_entities is not None and source_chain_id in source_asym_entities
                )
                if (
                    suffixed is not None
                    and int(suffixed.group(2)) >= 2
                    and not source_chain_is_declared
                ):
                    base_chain_id = suffixed.group(1)
                    base_group = source_groups.get(base_chain_id)
                    base_source_entity_id = (
                        entity_id
                        if source_asym_entities is None
                        else source_asym_entities.get(base_chain_id)
                    )
                    if base_group is not None and base_group["entity_id"] == base_source_entity_id and (
                        source_asym_entities is None or source_entity_id in {None, base_source_entity_id}
                    ):
                        source_chain_id = base_chain_id
                        source_entity_id = base_source_entity_id
            if source_entity_id is None:
                if entity_type != ENTITY_MOLECULE:
                    raise ValueError(
                        f"source polymer chain {source_chain_id!r} is missing from mmCIF _struct_asym"
                    )
                source_entity_id = entity_id
            group = source_groups.setdefault(
                source_chain_id,
                {
                    "entity_id": source_entity_id,
                    "entity_type": entity_type,
                    "entity_molecule_type": int(system.entity_molecule_type[entity_index].item()),
                    "instance_ids": [],
                    "component_ids": [],
                },
            )
            if group["entity_id"] != source_entity_id:
                raise ValueError(
                    f"source chain {source_chain_id!r} maps conflicting entity IDs "
                    f"{group['entity_id']!r} and {source_entity_id!r}"
                )
            group["instance_ids"].append(instance_id)
            group["component_ids"].extend(component_ids)

        chains: list[dict[str, Any]] = []
        coordinates: dict[str, np.ndarray] = {}
        for source_chain_id, group in source_groups.items():
            instance_ids = group["instance_ids"]
            if not any(instance_id in center_instances for instance_id in instance_ids):
                continue
            entity_id = group["entity_id"]
            entity_type = group["entity_type"]
            component_ids = group["component_ids"]
            if entity_type == ENTITY_MOLECULE and group["entity_molecule_type"] == MOLECULE_WATER:
                continue
            source_entity_id = entity_id
            if entity_type == ENTITY_MOLECULE:
                mol_type = (
                    "ions"
                    if component_ids
                    and all(component_id in PROTENIX_ION_COMPONENT_IDS for component_id in component_ids)
                    else "ligand"
                )
                cluster_id = "_".join(component_ids)
            else:
                mol_type, cluster_id = _cluster_id(
                    pdb_id,
                    source_entity_id,
                    entity_type,
                    component_ids,
                    clusters,
                )
            if mol_type in exclude_mol_types:
                continue
            coordinate_parts = [
                internal_coordinates[instance_id]
                for instance_id in instance_ids
                if instance_id in internal_coordinates
            ]
            coordinates[source_chain_id] = np.concatenate(coordinate_parts, axis=0)
            chains.append(
                {
                    "entity_id": source_entity_id,
                    "chain_id": source_chain_id,
                    "mol_type": mol_type,
                    "cluster_id": cluster_id,
                    "instance_ids": tuple(instance_ids),
                }
            )
            entries.append(
                ClusterIndex(
                    type="chain",
                    pdb_id=pdb_id,
                    record_index=record_index,
                    entity_1_id=source_entity_id,
                    entity_2_id="",
                    chain_1_id=source_chain_id,
                    chain_2_id="",
                    mol_1_type=mol_type,
                    mol_2_type="",
                    cluster_1_id=cluster_id,
                    cluster_2_id="",
                    cluster_id=cluster_id,
                    chain_key=source_chain_id,
                    reference_instance_ids=tuple(instance_ids),
                )
            )

        seen_entity_pairs: set[tuple[str, str]] = set()
        pair_chains = sorted(chains, key=lambda item: item["chain_id"])
        for left_index, left in enumerate(pair_chains):
            for right in pair_chains[left_index + 1 :]:
                entity_pair = tuple(sorted((left["entity_id"], right["entity_id"])))
                if entity_pair in seen_entity_pairs and not include_all_chain_pairs:
                    continue
                left_xyz, right_xyz = coordinates[left["chain_id"]], coordinates[right["chain_id"]]
                if not _in_contact(left_xyz, right_xyz, contact_cutoff):
                    continue
                ordered = sorted((left, right), key=lambda item: (item["entity_id"], item["chain_id"]))
                first, second = ordered
                entries.append(
                    ClusterIndex(
                        type="interface",
                        pdb_id=pdb_id,
                        record_index=record_index,
                        entity_1_id=first["entity_id"],
                        entity_2_id=second["entity_id"],
                        chain_1_id=first["chain_id"],
                        chain_2_id=second["chain_id"],
                        mol_1_type=first["mol_type"],
                        mol_2_type=second["mol_type"],
                        cluster_1_id=first["cluster_id"],
                        cluster_2_id=second["cluster_id"],
                        cluster_id=":".join(sorted((first["cluster_id"], second["cluster_id"]))),
                        chain_key=f"{first['chain_id']}:{second['chain_id']}",
                        reference_instance_ids=(
                            *cast(tuple[str, ...], first["instance_ids"]),
                            *cast(tuple[str, ...], second["instance_ids"]),
                        ),
                    )
                )
                seen_entity_pairs.add(entity_pair)
    return entries


def calculate_cluster_weights(
    entries: Sequence[ClusterIndex],
    *,
    beta: Mapping[str, float] | None = None,
    alpha: Mapping[str, float] | None = None,
    eps: float = 1e-9,
) -> tuple[list[ClusterIndex], list[float]]:
    r"""Apply Protenix ``calc_weights_for_df`` semantics to filtered rows."""

    beta = {"chain": 0.5, "interface": 1.0} if beta is None else dict(beta)
    alpha = {"prot": 3.0, "nuc": 3.0, "ligand": 1.0} if alpha is None else dict(alpha)
    for field, values in (("beta", beta), ("alpha", alpha)):
        for key, value in values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{field}[{key!r}] must be a finite non-negative number")
    members = []
    for entry in entries:
        entities = sorted((entry.entity_1_id or "None", entry.entity_2_id or "None"))
        members.append("_".join((entry.pdb_id, *entities)))
    member_counts = Counter(members)
    cluster_members: dict[str, set[str]] = {}
    for entry, member in zip(entries, members):
        cluster_members.setdefault(entry.cluster_id, set()).add(member)

    weighted_entries, weights = [], []
    for entry, member in zip(entries, members):
        if entry.type not in beta:
            raise ValueError(f"beta is missing sampling row type {entry.type!r}")
        types = (entry.mol_1_type, entry.mol_2_type)
        unknown = {value for value in types if value and value not in alpha}
        if unknown:
            raise ValueError(f"alpha is missing molecule type(s) {sorted(unknown)}")
        weighted_count = sum(alpha[value] for value in types if value)
        cluster_size = len(cluster_members[entry.cluster_id])
        member_num = member_counts[member]
        weight = float(beta[entry.type]) * weighted_count / (cluster_size + eps) / member_num
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"calculated weight for {entry.chain_key!r} must be finite and non-negative")
        weighted_entries.append(
            replace(
                entry,
                sample_member=member,
                cluster_size=cluster_size,
                member_num=member_num,
                weight=weight,
            )
        )
        weights.append(weight)
    return weighted_entries, weights


class ClusterWeightedSampler(Sampler[int]):
    r"""Sample weighted cluster-row ordinals."""

    def __init__(
        self,
        entries: Sequence[ClusterIndex],
        weights: Sequence[float],
        num_samples: int,
        *,
        replacement: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if not entries:
            raise ValueError("entries must be non-empty")
        try:
            weight_tensor = torch.as_tensor(weights, dtype=torch.double)
        except (TypeError, ValueError) as error:
            raise ValueError("weights must be a one-dimensional numeric sequence") from error
        if weight_tensor.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if len(entries) != len(weight_tensor):
            raise ValueError("entries and weights must have equal length")
        if not torch.isfinite(weight_tensor).all():
            raise ValueError("weights must be finite")
        if torch.any(weight_tensor < 0):
            raise ValueError("weights must be non-negative")
        if float(weight_tensor.sum()) <= 0:
            raise ValueError("weights must have a positive sum")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.entries = tuple(entries)
        self.weights = weight_tensor
        self.num_samples = int(num_samples)
        self.replacement = replacement
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.num_samples_per_replica = math.ceil(self.num_samples / self.num_replicas)
        self.total_size = self.num_samples_per_replica * self.num_replicas
        if not replacement and int(torch.count_nonzero(self.weights > 0)) < self.total_size:
            raise ValueError("non-replacement sampling requires at least total_size positive-weight rows")

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        row_indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        ).tolist()
        return iter(row_indices[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.epoch = int(state_dict["epoch"])
