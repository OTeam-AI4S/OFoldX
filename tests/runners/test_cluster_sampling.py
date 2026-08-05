from __future__ import annotations

import gzip
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ofoldx.data import (
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    Structure,
    get_component,
)
from ofoldx.io import read_structure, write_structure
from ofoldx.processor import UnresolvedTrainingCropError
from ofoldx.runners import (
    ClusterIndex,
    ClusterWeightedSampler,
    Runner,
    calculate_cluster_weights,
    make_cluster_indices,
    parse_pdb_cluster_file,
)
from ofoldx.runners import cluster as cluster_module
from ofoldx.runners.dataset import (
    ClusteredStructureDataset,
    ConsecutiveDataLoadingErrors,
    ProcessingTimeoutExhaustionError,
    StructureProcessingTimeoutError,
)


def _structure(
    system: MolecularSystem,
    sites: tuple[AtomSite, ...] | list[AtomSite],
    coordinates: np.ndarray | list[tuple[float, float, float]],
    coordinate_mask: np.ndarray | list[bool] | None = None,
) -> Structure:
    r"""Place selected test sites on the canonical tensor-native atom axis."""

    site_coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    site_mask = (
        torch.ones(len(sites), dtype=torch.bool)
        if coordinate_mask is None
        else torch.as_tensor(coordinate_mask, dtype=torch.bool)
    )
    atom_count = system.atom_symbol.numel()
    full_coordinates = torch.zeros((atom_count, 3), dtype=torch.float32)
    full_coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    for site_index, site in enumerate(sites):
        atom_index = system.atom_index(site)
        full_coordinates[atom_index] = site_coordinates[site_index]
        full_coordinate_mask[atom_index] = site_mask[site_index]
    component_count = system.component_symbol.numel()
    source_symbols = tuple(str(index + 1) for index in range(component_count))
    return Structure(
        system=system,
        coordinates=full_coordinates,
        coordinate_mask=full_coordinate_mask,
        occupancy=full_coordinate_mask.to(torch.float32),
        occupancy_mask=full_coordinate_mask.clone(),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=full_coordinate_mask.clone(),
        source_symbols=source_symbols,
        residue_id=torch.arange(component_count),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _cif(
    path: Path,
    *,
    sequences: tuple[str, ...] = ("AAAAAAAAAA", "GGGGGGGGGG"),
    origins: tuple[float, ...] = (0.0, 4.0),
    same_entity: bool = False,
) -> Path:
    entities = {
        str(index + 1): PolymerSequence.from_one_letter(sequence)
        for index, sequence in enumerate(sequences)
        if not same_entity or index == 0
    }
    instances = {
        chr(65 + index): "1" if same_entity else str(index + 1)
        for index in range(len(sequences))
    }
    system = MolecularSystem.from_entities(entities, instances)
    sites, coordinates = [], []
    for index, (instance, entity_id) in enumerate(instances.items()):
        entity = entities[entity_id]
        for residue in range(len(entity)):
            residue_origin = origins[index] + residue * 0.1
            for atom_offset, atom_id in enumerate(("N", "CA", "C", "O")):
                sites.append(AtomSite(instance, residue, atom_id))
                coordinates.append((residue_origin + atom_offset * 0.01, 0.0, 0.0))
    structure = _structure(
        system,
        tuple(sites),
        np.asarray(coordinates, dtype=np.float32),
        np.ones(len(sites), dtype=np.bool_),
    )
    write_structure(structure, path)
    return path


def _microheterogeneous_cluster_cif(path: Path) -> Path:
    r"""Write two source chains whose legal realized sequences split one source entity."""

    entity_rows: list[str] = []
    for sequence_id in range(1, 11):
        if sequence_id == 5:
            entity_rows.extend(("1 5 PRO y", "1 5 SER y"))
        else:
            entity_rows.append(f"1 {sequence_id} ALA n")
    scheme_rows: list[str] = []
    atom_rows: list[str] = []
    serial = 1
    for chain_ordinal, chain_id in enumerate(("A", "B")):
        for sequence_id in range(1, 11):
            if sequence_id == 5:
                declared_candidates = (("PRO", "A", 5), ("SER", "B", 105))
                observed_candidates = declared_candidates[:1] if chain_id == "A" else declared_candidates[1:]
            else:
                declared_candidates = (("ALA", ".", sequence_id),)
                observed_candidates = declared_candidates
            hetero = "y" if sequence_id == 5 else "n"
            for component_id, _altloc_id, author_number in declared_candidates:
                scheme_rows.append(
                    f"{chain_id} 1 {sequence_id} {component_id} {author_number} "
                    f"{author_number} ? {hetero}"
                )
            for component_id, altloc_id, author_number in observed_candidates:
                atom_rows.append(
                    f"ATOM {serial} C CA {altloc_id} {component_id} {chain_id} 1 {sequence_id} ? "
                    f"{float(sequence_id + 2 * chain_ordinal):.1f} 0.0 0.0 "
                    f"{'0.5' if sequence_id == 5 else '1.0'} 10.0 {author_number} "
                    f"{component_id} {chain_id} CA 1"
                )
                serial += 1
    path.write_text(
        f"""data_cluster_microheterogeneity
loop_
_entity.id
_entity.type
1 polymer
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 1
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
{chr(10).join(entity_rows)}
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
_pdbx_poly_seq_scheme.hetero
{chr(10).join(scheme_rows)}
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
{chr(10).join(atom_rows)}
""",
        encoding="ascii",
    )
    return path


def _genuine_suffixed_asym_nonpoly_cif(path: Path) -> Path:
    r"""Write genuine source asym IDs that resemble the reader's internal suffixes."""

    path.write_text(
        """data_genuine_suffixed_asym
loop_
_entity.id
_entity.type
1 non-polymer
loop_
_struct_asym.id
_struct_asym.entity_id
B 1
B_2 1
loop_
_pdbx_entity_nonpoly.entity_id
_pdbx_entity_nonpoly.comp_id
1 LIG
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.entity_id
_pdbx_nonpoly_scheme.mon_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.pdb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
_pdbx_nonpoly_scheme.pdb_strand_id
_pdbx_nonpoly_scheme.pdb_ins_code
B 1 LIG 1 1 1 B ?
B_2 1 LIG 1 1 1 B_2 ?
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
HETATM 1 C C1 . LIG B 1 . ? 0.0 0.0 0.0 1.0 10.0 1 LIG B C1 1
HETATM 2 C C1 . LIG B_2 1 . ? 20.0 0.0 0.0 1.0 10.0 1 LIG B_2 C1 1
""",
        encoding="ascii",
    )
    return path


def _entry(**overrides: object) -> ClusterIndex:
    values: dict[str, object] = {
        "type": "chain",
        "pdb_id": "1abc",
        "record_index": 0,
        "entity_1_id": "1",
        "entity_2_id": "",
        "chain_1_id": "A",
        "chain_2_id": "",
        "mol_1_type": "prot",
        "mol_2_type": "",
        "cluster_1_id": "cluster1",
        "cluster_2_id": "",
        "cluster_id": "cluster1",
        "chain_key": "A",
    }
    values.update(overrides)
    return ClusterIndex(**values)  # type: ignore[arg-type]


def test_cluster_indices_cache_created_then_loaded_without_rescanning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cif = tmp_path / "1abc.cif"
    cif.write_text("cif", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    cache = tmp_path / "indices.jsonl"
    calls = 0

    def make(*args: object, **kwargs: object) -> list[ClusterIndex]:
        nonlocal calls
        calls += 1
        return [_entry()]

    monkeypatch.setattr(cluster_module, "make_cluster_indices", make)
    first = ClusteredStructureDataset(cif, lambda value: value, cluster_file=str(cluster), cluster_indices_file=cache)
    assert first.cluster_indices_status == "created"
    assert first.cluster_indices_file == cache.resolve()
    lines = cache.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    assert metadata["schema"] == "ofoldx.cluster_indices"
    assert metadata["schema_version"] == 2
    assert metadata["generator_version"] == 2
    assert metadata["entry_count"] == 1
    assert json.loads(lines[1])["entry"]["reference_instance_ids"] == []

    monkeypatch.setattr(cluster_module, "make_cluster_indices", lambda *args, **kwargs: pytest.fail("rescanned"))
    second = ClusteredStructureDataset(cif, lambda value: value, cluster_file=str(cluster), cluster_indices_file=cache)
    assert second.cluster_indices_status == "loaded"
    assert second.entries == first.entries
    assert second.weights == first.weights
    assert calls == 1


def test_cluster_indices_cache_rebuilds_for_input_change_and_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cif = tmp_path / "1abc.cif"
    cif.write_text("cif", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    cache = tmp_path / "indices.jsonl"
    calls = 0

    def make(*args: object, **kwargs: object) -> list[ClusterIndex]:
        nonlocal calls
        calls += 1
        return [_entry()]

    monkeypatch.setattr(cluster_module, "make_cluster_indices", make)
    kwargs = {"cluster_file": str(cluster), "cluster_indices_file": cache}
    assert ClusteredStructureDataset(cif, lambda value: value, **kwargs).cluster_indices_status == "created"
    cluster.write_text("1ABC_1 2DEF_1\n", encoding="ascii")
    assert ClusteredStructureDataset(cif, lambda value: value, **kwargs).cluster_indices_status == "rebuilt"
    cache.write_text('{"schema":"broken"}\n', encoding="utf-8")
    assert ClusteredStructureDataset(cif, lambda value: value, **kwargs).cluster_indices_status == "rebuilt"
    lines = cache.read_text(encoding="utf-8").splitlines()
    invalid_row = json.loads(lines[1])
    invalid_row["entry"]["weight"] = -1.0
    invalid_row["weight"] = -1.0
    cache.write_text(lines[0] + "\n" + json.dumps(invalid_row) + "\n", encoding="utf-8")
    assert ClusteredStructureDataset(cif, lambda value: value, **kwargs).cluster_indices_status == "rebuilt"
    assert calls == 4


def test_structure_ingestion_policy_changes_cache_fingerprint_and_default_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cif = tmp_path / "1abc.cif"
    cif.write_text("cif", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    records = (str(cif),)

    fingerprint = cluster_module.cluster_indices_fingerprint(records, cluster)
    path = cluster_module.default_cluster_indices_path(tmp_path, cif, cluster)
    monkeypatch.setattr(cluster_module, "_STRUCTURE_INGESTION_POLICY", "test-policy-change")

    assert cluster_module.cluster_indices_fingerprint(records, cluster) != fingerprint
    assert cluster_module.default_cluster_indices_path(tmp_path, cif, cluster) != path


def test_cluster_indices_failed_atomic_rebuild_preserves_previous_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cif = tmp_path / "1abc.cif"
    cif.write_text("cif", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    cache = tmp_path / "indices.jsonl"
    monkeypatch.setattr(cluster_module, "make_cluster_indices", lambda *args, **kwargs: [_entry()])
    kwargs = {"cluster_file": str(cluster), "cluster_indices_file": cache}
    ClusteredStructureDataset(cif, lambda value: value, **kwargs)
    previous = cache.read_bytes()
    cluster.write_text("changed\n", encoding="ascii")

    def fail_publish(*args: object) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(cluster_module.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failed"):
        ClusteredStructureDataset(cif, lambda value: value, **kwargs)
    assert cache.read_bytes() == previous


def test_loaded_cluster_indices_preserve_seeded_sampler_ordinals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cif = tmp_path / "1abc.cif"
    cif.write_text("cif", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    cache = tmp_path / "indices.jsonl"
    rows = [_entry(chain_key="A"), _entry(chain_key="B", entity_1_id="2", cluster_id="cluster2")]
    monkeypatch.setattr(cluster_module, "make_cluster_indices", lambda *args, **kwargs: rows)
    first = ClusteredStructureDataset(cif, lambda value: value, cluster_file=str(cluster), cluster_indices_file=cache)
    second = ClusteredStructureDataset(cif, lambda value: value, cluster_file=str(cluster), cluster_indices_file=cache)
    samplers = [ClusterWeightedSampler(dataset.entries, dataset.weights, 20, seed=17) for dataset in (first, second)]
    for sampler in samplers:
        sampler.set_epoch(4)
    assert list(samplers[0]) == list(samplers[1])


@pytest.mark.parametrize("tamper", ["record_index", "weights"])
def test_cluster_indices_payload_checksum_rejects_legal_value_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    records = [tmp_path / "first.cif", tmp_path / "second.cif"]
    for record in records:
        record.write_text("cif", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("", encoding="ascii")
    cache = tmp_path / "indices.jsonl"
    calls = 0

    def make(*args: object, **kwargs: object) -> list[ClusterIndex]:
        nonlocal calls
        calls += 1
        return [_entry(record_index=0), _entry(record_index=1, chain_key="B", entity_1_id="2")]

    monkeypatch.setattr(cluster_module, "make_cluster_indices", make)
    kwargs = {"cluster_file": str(cluster), "cluster_indices_file": cache}
    ClusteredStructureDataset(tmp_path, lambda value: value, **kwargs)
    lines = cache.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    if tamper == "record_index":
        row["entry"]["record_index"] = 1
    else:
        row["entry"]["weight"] = 0.25
        row["weight"] = 0.25
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    cache.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rebuilt = ClusteredStructureDataset(tmp_path, lambda value: value, **kwargs)
    assert rebuilt.cluster_indices_status == "rebuilt"
    assert calls == 2


def test_structure_file_identity_invalidates_cache_after_same_size_same_mtime_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cif = tmp_path / "1abc.cif"
    cif.write_text("old", encoding="ascii")
    original = cif.stat()
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("", encoding="ascii")
    cache = tmp_path / "indices.jsonl"
    calls = 0

    def make(*args: object, **kwargs: object) -> list[ClusterIndex]:
        nonlocal calls
        calls += 1
        return [_entry()]

    monkeypatch.setattr(cluster_module, "make_cluster_indices", make)
    kwargs = {"cluster_file": str(cluster), "cluster_indices_file": cache}
    ClusteredStructureDataset(cif, lambda value: value, **kwargs)
    cif.write_text("new", encoding="ascii")
    deadline = time.monotonic() + 2.0
    while True:
        os.utime(cif, ns=(original.st_atime_ns, original.st_mtime_ns))
        replaced = cif.stat()
        if replaced.st_ctime_ns != original.st_ctime_ns:
            break
        if time.monotonic() >= deadline:
            pytest.skip("filesystem does not expose a timely ctime change for an in-place rewrite")
        time.sleep(0.01)
    replaced = cif.stat()
    assert replaced.st_size == original.st_size
    assert replaced.st_mtime_ns == original.st_mtime_ns
    rebuilt = ClusteredStructureDataset(cif, lambda value: value, **kwargs)
    assert rebuilt.cluster_indices_status == "rebuilt"
    assert calls == 2


def test_concurrent_cache_build_rechecks_after_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cif = tmp_path / "1abc.cif"
    cif.write_text("cif", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("", encoding="ascii")
    cache = tmp_path / "indices.jsonl"
    building = threading.Event()
    release = threading.Event()
    calls = 0
    statuses: list[str] = []
    errors: list[BaseException] = []

    def make(*args: object, **kwargs: object) -> list[ClusterIndex]:
        nonlocal calls
        calls += 1
        building.set()
        assert release.wait(timeout=5)
        return [_entry()]

    def construct() -> None:
        try:
            dataset = ClusteredStructureDataset(
                cif,
                lambda value: value,
                cluster_file=str(cluster),
                cluster_indices_file=cache,
            )
            statuses.append(dataset.cluster_indices_status)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(cluster_module, "make_cluster_indices", make)
    first = threading.Thread(target=construct)
    second = threading.Thread(target=construct)
    first.start()
    assert building.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert calls == 1
    assert sorted(statuses) == ["created", "loaded"]


def _mock_cif_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, structure: Structure) -> tuple[Path, Path]:
    cif = tmp_path / "mock.cif"
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("", encoding="ascii")
    monkeypatch.setattr("ofoldx.io.read_structure", lambda _, **kwargs: structure)
    return cif, cluster


def test_cluster_parser_is_case_insensitive_and_skips_predictions(tmp_path: Path) -> None:
    path = tmp_path / "clusters.txt"
    path.write_text("\n1ABC_1 2def_2 AF_Q9_1 ma_model_1\n\nAF_ONLY\n3GHI_1\n", encoding="ascii")
    parsed = parse_pdb_cluster_file(path)
    assert parsed["1abc_1"] == ("pdb_cluster_1ABC_1", 3)
    assert parsed["2def_2"] == ("pdb_cluster_1ABC_1", 3)
    assert parsed["ma_model_1"] == ("pdb_cluster_1ABC_1", 3)
    assert parsed["3ghi_1"] == ("pdb_cluster_3GHI_1", 1)
    assert "af_q9_1" not in parsed


def test_indices_include_chains_contacting_pair_and_sorted_clusters(tmp_path: Path) -> None:
    cif = _cif(tmp_path / "1abc.cif")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n1ABC_2\n", encoding="ascii")
    rows = make_cluster_indices([str(cif)], cluster)
    assert [(row.type, row.chain_key) for row in rows] == [("chain", "A"), ("chain", "B"), ("interface", "A:B")]
    assert [row.entity_1_id for row in rows[:2]] == ["1", "2"]
    assert {row.record_index for row in rows} == {0}
    pair = rows[-1]
    assert pair.cluster_id == ":".join(sorted((pair.cluster_1_id, pair.cluster_2_id)))
    assert pair.record_index == 0 and pair.pdb_id == "1abc"


def test_indices_exclude_noncontact_and_deduplicate_same_entity_pair(tmp_path: Path) -> None:
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    distant = _cif(tmp_path / "far.cif", origins=(0.0, 20.0))
    assert all(row.type == "chain" for row in make_cluster_indices([str(distant)], cluster))

    same = _cif(
        tmp_path / "same.cif",
        sequences=("AAAAAAAAAA", "AAAAAAAAAA", "AAAAAAAAAA"),
        origins=(0.0, 3.0, 4.0),
        same_entity=True,
    )
    rows = make_cluster_indices([str(same)], cluster)
    assert sum(row.type == "interface" for row in rows) == 1
    all_rows = make_cluster_indices([str(same)], cluster, include_all_chain_pairs=True)
    assert sum(row.type == "interface" for row in all_rows) == 3


def test_interface_representative_is_stable_by_source_chain_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entity = PolymerSequence.from_one_letter("AAAAAAAAAA")
    instances = {"C": "1", "A": "1", "B": "1"}
    sites = tuple(
        AtomSite(instance_id, component_index, "CA")
        for instance_id in instances
        for component_index in range(10)
    )
    structure = _structure(
        MolecularSystem.from_entities({"1": entity}, instances),
        sites,
        np.zeros((len(sites), 3), dtype=np.float32),
        np.ones(len(sites), dtype=np.bool_),
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    rows = make_cluster_indices([str(cif)], cluster)
    assert [row.chain_key for row in rows if row.type == "chain"] == ["C", "A", "B"]
    assert [row.chain_key for row in rows if row.type == "interface"] == ["A:B"]


def test_standard_protein_requires_resolved_ca_for_chain_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entity = PolymerSequence.from_one_letter("A")
    structure = _structure(
        MolecularSystem.from_entities({"1": entity}, {"A": "1"}),
        (AtomSite("A", 0, "CA"), AtomSite("A", 0, "CB")),
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32),
        np.asarray((False, True), dtype=np.bool_),
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    assert make_cluster_indices([str(cif)], cluster) == []


def test_cluster_scan_skips_all_unk_l_polypeptide_but_keeps_mixed_chain(tmp_path: Path) -> None:
    all_unknown = PolymerSequence("protein", (get_component("UNK"), get_component("UNK")))
    mixed = PolymerSequence("protein", (get_component("UNK"), get_component("ALA")))

    def write_polymer(path: Path, sequence: PolymerSequence) -> Path:
        system = MolecularSystem.from_entities({"1": sequence}, {"A": "1"})
        sites = [system.atom_site(index) for index in range(system.atom_symbol.numel())]
        write_structure(
            _structure(system, sites, np.zeros((len(sites), 3), dtype=np.float32)),
            path,
        )
        return path

    all_unknown_path = write_polymer(tmp_path / "allunk.cif", all_unknown)
    mixed_path = write_polymer(tmp_path / "mixed.cif", mixed)
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("ALLUNK_1\nMIXED_1\n", encoding="ascii")

    assert make_cluster_indices([str(all_unknown_path)], cluster) == []
    mixed_rows = make_cluster_indices([str(mixed_path)], cluster)
    assert [row.chain_key for row in mixed_rows if row.type == "chain"] == ["A"]


def test_molecule_instances_use_source_chain_ids_and_stable_interfaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protein = PolymerSequence.from_one_letter("AAAAAAAAAA")
    sulfate = Molecule((ChemicalComponent("SO4", (ChemicalAtom("S", "S"),), frozenset()),))
    heme = Molecule((ChemicalComponent("HEM", (ChemicalAtom("FE", "Fe"),), frozenset()),))
    entities = {"1": protein, "2": sulfate, "3": heme}
    instances = {"A": "1", "B:157:SO4": "2", "C:155:HEM": "3"}
    sites = [AtomSite("A", index, "CA") for index in range(10)]
    sites.extend((AtomSite("B:157:SO4", 0, "S"), AtomSite("C:155:HEM", 0, "FE")))
    coordinates = [(0.0, 0.0, 0.0)] * 10
    coordinates.extend(((-4.0, 0.0, 0.0), (4.0, 0.0, 0.0)))
    structure = _structure(
        MolecularSystem.from_entities(entities, instances),
        tuple(sites),
        np.asarray(coordinates, dtype=np.float32),
        np.ones(len(sites), dtype=np.bool_),
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    cluster.write_text("MOCK_1\n", encoding="ascii")
    rows = make_cluster_indices([str(cif)], cluster)
    assert [(row.type, row.chain_key) for row in rows] == [
        ("chain", "A"),
        ("chain", "B"),
        ("chain", "C"),
        ("interface", "A:B"),
        ("interface", "A:C"),
    ]
    assert [(row.entity_1_id, row.chain_1_id) for row in rows[:3]] == [("1", "A"), ("2", "B"), ("3", "C")]


def test_molecule_instances_on_one_source_chain_are_grouped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ligand = Molecule((ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset()),))
    structure = _structure(
        MolecularSystem.from_entities(
            {"2": ligand},
            {"B:1:LIG": "2", "B:2:LIG": "2"},
        ),
        (AtomSite("B:1:LIG", 0, "C1"), AtomSite("B:2:LIG", 0, "C1")),
        np.asarray(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), dtype=np.float32),
        np.ones(2, dtype=np.bool_),
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    rows = make_cluster_indices([str(cif)], cluster)
    assert len(rows) == 1
    assert (rows[0].chain_1_id, rows[0].entity_1_id, rows[0].cluster_1_id) == ("B", "2", "LIG_LIG")


def test_reader_suffix_is_grouped_only_with_matching_base_entity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ligand = Molecule((ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset()),))
    drug = Molecule((ChemicalComponent("DRG", (ChemicalAtom("C1", "C"),), frozenset()),))
    system = MolecularSystem.from_entities(
        {"2": ligand, "3": drug},
        {"B": "2", "B_2": "2", "B_3": "3"},
    )
    structure = _structure(
        system,
        (
            AtomSite("B", 0, "C1"),
            AtomSite("B_2", 0, "C1"),
            AtomSite("B_3", 0, "C1"),
        ),
        [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)],
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    rows = [row for row in make_cluster_indices([str(cif)], cluster) if row.type == "chain"]
    assert [(row.chain_key, row.entity_1_id, row.cluster_1_id) for row in rows] == [
        ("B", "2", "LIG_LIG"),
        ("B_3", "3", "DRG"),
    ]


def test_source_chain_with_conflicting_entities_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left = Molecule((ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset()),))
    right = Molecule((ChemicalComponent("DRG", (ChemicalAtom("C1", "C"),), frozenset()),))
    structure = _structure(
        MolecularSystem.from_entities(
            {"2": left, "3": right},
            {"B:1:LIG": "2", "B:2:DRG": "3"},
        ),
        (AtomSite("B:1:LIG", 0, "C1"), AtomSite("B:2:DRG", 0, "C1")),
        np.zeros((2, 3), dtype=np.float32),
        np.ones(2, dtype=np.bool_),
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    with pytest.raises(ValueError, match="source chain 'B' maps conflicting entity IDs"):
        make_cluster_indices([str(cif)], cluster)


def test_contact_distance_blocks_both_atom_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[int, ...]] = []
    original_sum = np.sum

    def recording_sum(value, *args, **kwargs):
        observed.append(value.shape)
        return original_sum(value, *args, **kwargs)

    monkeypatch.setattr(cluster_module.np, "sum", recording_sum)
    left = np.zeros((11, 3), dtype=np.float32)
    right = np.full((13, 3), 100.0, dtype=np.float32)
    assert not cluster_module._in_contact(left, right, 5.0, block_size=4)
    assert observed
    assert all(shape[0] <= 4 and shape[1] <= 4 for shape in observed)


def test_ion_ccd_table_classification_and_default_filtering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    components = {
        "molecule_fe": ChemicalComponent("FE", (ChemicalAtom("FE", "Fe"),), frozenset()),
        "molecule_g6d": ChemicalComponent(
            "G6D",
            (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
            frozenset(),
        ),
        "molecule_mg": ChemicalComponent("MG", (ChemicalAtom("MG", "Mg"),), frozenset()),
        "molecule_lig": ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset()),
    }
    entities = {entity_id: Molecule((component,)) for entity_id, component in components.items()}
    system = MolecularSystem.from_entities(
        entities,
        {"F": "molecule_fe", "I": "molecule_g6d", "G": "molecule_mg", "L": "molecule_lig"},
    )
    sites = (
        AtomSite("F", 0, "FE"),
        AtomSite("I", 0, "C1"),
        AtomSite("I", 0, "O1"),
        AtomSite("G", 0, "MG"),
        AtomSite("L", 0, "C1"),
    )
    structure = _structure(
        system,
        sites,
        np.zeros((len(sites), 3), dtype=np.float32),
        np.ones(len(sites), dtype=np.bool_),
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    all_rows = make_cluster_indices([str(cif)], cluster, exclude_mol_types=())
    assert [(row.chain_key, row.mol_1_type) for row in all_rows if row.type == "chain"] == [
        ("F", "ligand"),
        ("I", "ions"),
        ("G", "ligand"),
        ("L", "ligand"),
    ]
    filtered = make_cluster_indices([str(cif)], cluster)
    assert all(row.chain_1_id != "I" and row.chain_2_id != "I" for row in filtered)
    assert [(row.chain_key, row.mol_1_type) for row in filtered if row.type == "chain"] == [
        ("F", "ligand"),
        ("G", "ligand"),
        ("L", "ligand"),
    ]


@pytest.mark.parametrize(
    "entity_id",
    ("polymer_alpha", "polymer_1#2", "custom_alpha"),
)
def test_entity_id_is_preserved_for_cluster_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entity_id: str,
) -> None:
    entity = PolymerSequence.from_one_letter("AAAAAAAAAA")
    structure = _structure(
        MolecularSystem.from_entities({entity_id: entity}, {"A": entity_id}),
        tuple(AtomSite("A", index, "CA") for index in range(10)),
        np.zeros((10, 3), dtype=np.float32),
        np.ones(10, dtype=np.bool_),
    )
    cif, cluster = _mock_cif_scan(monkeypatch, tmp_path, structure)
    cluster.write_text(f"MOCK_{entity_id}\n", encoding="ascii")
    row = make_cluster_indices([str(cif)], cluster)[0]
    assert row.entity_1_id == entity_id
    assert row.cluster_1_id == f"pdb_cluster_MOCK_{entity_id}"


def test_cluster_scan_uses_source_entity_for_split_microheterogeneous_sequences(
    tmp_path: Path,
) -> None:
    cif = _microheterogeneous_cluster_cif(tmp_path / "mock.cif")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("MOCK_1\n", encoding="ascii")

    realized = read_structure(cif)
    instance_indices = {
        realized.system.instance_id(index): index
        for index in range(int(realized.system.instance_symbol.numel()))
    }
    assert set(instance_indices) == {"A", "B"}
    assert int(realized.system.instance_entity[instance_indices["A"]]) != int(
        realized.system.instance_entity[instance_indices["B"]]
    )
    realized_components: dict[str, str] = {}
    for instance_id, instance_index in instance_indices.items():
        position_five = int(realized.system.instance_component_ptr[instance_index]) + 4
        realized_components[instance_id] = realized.system.symbols[
            int(realized.system.component_symbol[position_five])
        ]
    assert realized_components == {"A": "PRO", "B": "SER"}

    rows = make_cluster_indices([str(cif)], cluster)
    chain_rows = [row for row in rows if row.type == "chain"]

    assert [row.chain_key for row in chain_rows] == ["A", "B"]
    assert {row.entity_1_id for row in chain_rows} == {"1"}
    assert {row.cluster_1_id for row in chain_rows} == {"pdb_cluster_MOCK_1"}
    assert [row.reference_instance_ids for row in chain_rows] == [("A",), ("B",)]


def test_cluster_scan_preserves_genuine_suffixed_source_asym_ids(tmp_path: Path) -> None:
    cif = _genuine_suffixed_asym_nonpoly_cif(tmp_path / "genuine_suffix.cif")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("", encoding="ascii")

    rows = make_cluster_indices([str(cif)], cluster)
    chain_rows = [row for row in rows if row.type == "chain"]

    assert [row.chain_key for row in chain_rows] == ["B", "B_2"]
    assert [row.chain_1_id for row in chain_rows] == ["B", "B_2"]
    assert [row.entity_1_id for row in chain_rows] == ["1", "1"]
    assert [row.reference_instance_ids for row in chain_rows] == [("B",), ("B_2",)]
    assert [row.cluster_1_id for row in chain_rows] == ["LIG", "LIG"]


def test_weight_formula_includes_cluster_unique_members_and_duplicate_correction() -> None:
    rows = [
        _entry(),
        _entry(chain_1_id="B", chain_key="B"),
        _entry(pdb_id="2def", record_index=1, entity_1_id="2", chain_1_id="C", chain_key="C"),
        _entry(
            type="interface",
            entity_2_id="3",
            chain_2_id="D",
            mol_2_type="ligand",
            cluster_2_id="lig",
            cluster_id="cluster1",
            chain_key="A:D",
        ),
    ]
    weighted, weights = calculate_cluster_weights(rows)
    # cluster1 has three unique sample members; the duplicated 1abc chain member is divided by two.
    assert weights == pytest.approx([0.25, 0.25, 0.5, 4.0 / 3.0])
    assert [row.cluster_size for row in weighted] == [3, 3, 3, 3]
    assert [row.member_num for row in weighted] == [2, 2, 1, 1]


def test_cluster_dataset_rejects_manifest_assembly_id(tmp_path: Path) -> None:
    cif = _cif(tmp_path / "1abc.cif", sequences=("AAAAAAAAAA",), origins=(0.0,))
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps({"structure": str(cif), "assembly_id": "1"}) + "\n", encoding="utf-8")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    with pytest.raises(ValueError, match="asymmetric-unit"):
        ClusteredStructureDataset(manifest, lambda structure: structure, cluster_file=str(cluster))


def test_cluster_dataset_rejects_non_cif_record(tmp_path: Path) -> None:
    pdb = tmp_path / "1abc.pdb"
    pdb.write_text("", encoding="ascii")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("", encoding="ascii")
    with pytest.raises(ValueError, match="only supports CIF"):
        ClusteredStructureDataset(pdb, lambda structure: structure, cluster_file=str(cluster))


def test_weighted_sampler_reproducibility_frequency_state_and_ddp_stride() -> None:
    entries = [
        _entry(record_index=17),
        _entry(record_index=17, chain_key="B"),
        _entry(record_index=3, chain_key="C"),
    ]
    weights = [1.0, 4.0, 2.0]
    first = ClusterWeightedSampler(entries, weights, 1000, seed=7)
    second = ClusterWeightedSampler(entries, weights, 1000, seed=7)
    assert list(first) == list(second)
    assert Counter(first)[1] > Counter(first)[0]
    first.set_epoch(1)
    assert list(first) != list(second)
    state = first.state_dict()
    restored = ClusterWeightedSampler(entries, weights, 1000, seed=7)
    restored.load_state_dict(state)
    assert list(first) == list(restored)

    ranks = [
        ClusterWeightedSampler(entries, weights, 5, seed=9, num_replicas=3, rank=rank)
        for rank in range(3)
    ]
    assert [len(sampler) for sampler in ranks] == [2, 2, 2]
    generator = torch.Generator().manual_seed(9)
    global_rows = torch.multinomial(torch.tensor(weights, dtype=torch.double), 6, True, generator=generator).tolist()
    expected = [global_rows[rank::3] for rank in range(3)]
    assert [list(sampler) for sampler in ranks] == expected


def test_weighted_sampler_distinguishes_rows_for_the_same_record() -> None:
    entries = [_entry(record_index=0), _entry(record_index=0, chain_key="B")]
    sampler = ClusterWeightedSampler(entries, [0.0, 1.0], 4, seed=3)
    assert list(sampler) == [1, 1, 1, 1]


@pytest.mark.parametrize(
    ("weights", "match"),
    (
        ([0.0, 0.0], "positive sum"),
        ([-1.0, 2.0], "non-negative"),
        ([float("nan"), 1.0], "finite"),
        ([[1.0, 2.0]], "one-dimensional"),
    ),
)
def test_weighted_sampler_rejects_invalid_weights(weights, match: str) -> None:
    entries = [_entry(record_index=0), _entry(record_index=1, chain_key="B")]
    with pytest.raises(ValueError, match=match):
        ClusterWeightedSampler(entries, weights, 2)


def test_nonreplacement_requires_enough_positive_weight_rows() -> None:
    entries = [_entry(record_index=0), _entry(record_index=1, chain_key="B")]
    with pytest.raises(ValueError, match="positive-weight"):
        ClusterWeightedSampler(entries, [1.0, 0.0], 2, replacement=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"beta": {"interface": 1.0}}, "beta.*chain"),
        ({"alpha": {"nuc": 3.0, "ligand": 1.0}}, "alpha.*prot"),
        ({"beta": {"chain": float("nan")}}, "beta.*finite"),
        ({"alpha": {"prot": -1.0}}, "alpha.*non-negative"),
    ),
)
def test_weight_calculation_validates_beta_and_alpha(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        calculate_cluster_weights([_entry()], **kwargs)


def test_cluster_dataset_preserves_manifest_msa_and_reserved_config(tmp_path: Path) -> None:
    cif = _cif(tmp_path / "1abc.cif", sequences=("AAAAAAAAAA",), origins=(0.0,))
    msa = tmp_path / "a.a3m"
    msa.write_text(">query\nAAAAAAAAAA\n>hit\nAAAA-AAAAA\n", encoding="ascii")
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        json.dumps({"structure": str(cif), "pdb_id": "explicit", "msa": {"path": str(msa), "entity_id": "1"}})
        + "\n",
        encoding="utf-8",
    )
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("EXPLICIT_1\n", encoding="ascii")
    seen = {}

    def processor(structure, **kwargs):
        seen.update(kwargs)
        return structure

    dataset = ClusteredStructureDataset(manifest, processor, cluster_file=str(cluster))
    assert dataset.entries[0].pdb_id == "explicit"
    dataset[0]
    assert seen["msa"].entity_id == "1"
    assert "indices" not in seen


def test_cluster_dataset_maps_row_ordinal_and_optionally_forwards_indices(tmp_path: Path) -> None:
    cif = _cif(tmp_path / "1abc.cif")
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n1ABC_2\n", encoding="ascii")
    seen = []

    def processor(structure, **kwargs):
        seen.append(kwargs["indices"])
        return structure

    dataset = ClusteredStructureDataset(
        cif,
        processor,
        cluster_file=str(cluster),
        forward_indices=True,
    )
    assert len(dataset) == len(dataset.entries)
    dataset[0]
    dataset[1]
    assert seen == [dataset.entries[0], dataset.entries[1]]
    assert seen[0].record_index == seen[1].record_index == 0
    assert seen[0].reference_instance_ids != seen[1].reference_instance_ids


@pytest.mark.parametrize("compound_suffix", [".cif.gz", ".mmcif.gz"])
def test_cluster_dataset_accepts_gzip_cif_and_matches_uncompressed_indices(
    tmp_path: Path,
    compound_suffix: str,
) -> None:
    plain = _cif(
        tmp_path / "1abc.cif",
        sequences=("AAAAAAAAAA",),
        origins=(0.0,),
    )
    compressed = tmp_path / f"1abc{compound_suffix}"
    compressed.write_bytes(gzip.compress(plain.read_bytes()))
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")

    expected = ClusteredStructureDataset(plain, lambda value: value, cluster_file=str(cluster))
    observed = ClusteredStructureDataset(compressed, lambda value: value, cluster_file=str(cluster))

    assert observed.records == [str(compressed)]
    assert observed.entries == expected.entries
    assert observed.weights == expected.weights


def test_cluster_dataset_retry_advances_row_and_forwards_replacement_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(chain_key="A", reference_instance_ids=("A",))
    second = _entry(chain_key="B", entity_1_id="2", reference_instance_ids=("B",))
    assert first.record_index == second.record_index == 0
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.records = ["shared.cif"]
    dataset.entries = [first, second]
    dataset.forward_indices = True
    dataset.max_consecutive_processing_errors = None
    dataset.training_log_file = None
    calls: list[tuple[int, ClusterIndex]] = []

    def process(record_index: int, **kwargs: object) -> str:
        indices = kwargs["indices"]
        assert isinstance(indices, ClusterIndex)
        calls.append((record_index, indices))
        if indices is first:
            raise UnresolvedTrainingCropError("first cluster row is unresolved")
        return indices.chain_key

    monkeypatch.setattr(dataset, "_process_record", process)

    assert dataset[0] == "B"
    assert calls == [(0, first), (0, second)]


@pytest.mark.parametrize("stage", ["read_structure", "read_msa"])
def test_cluster_dataset_record_wide_timeout_skips_rows_for_same_cif_within_request(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stage: str,
) -> None:
    first = _entry(chain_key="A", reference_instance_ids=("A",))
    same_cif = _entry(chain_key="B", entity_1_id="2", reference_instance_ids=("B",))
    next_cif = _entry(record_index=1, pdb_id="2xyz", chain_key="C", reference_instance_ids=("C",))
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.records = ["shared.cif", "success.cif"]
    dataset.entries = [first, same_cif, next_cif]
    dataset.forward_indices = True
    dataset.processing_timeout_seconds = 1.0
    dataset.max_consecutive_processing_errors = None
    dataset.training_log_file = None
    calls: list[tuple[int, ClusterIndex]] = []

    def process(record_index: int, **kwargs: object) -> str:
        indices = kwargs["indices"]
        assert isinstance(indices, ClusterIndex)
        calls.append((record_index, indices))
        if record_index == 0:
            raise StructureProcessingTimeoutError("shared CIF timed out", stage=stage)
        return indices.chain_key

    monkeypatch.setattr(dataset, "_process_record", process)

    with caplog.at_level("WARNING", logger="ofoldx.runners.dataset"):
        assert dataset[0] == "C"
    assert calls == [(0, first), (1, next_cif)]
    assert len(caplog.messages) == 1
    assert "logical index 0" in caplog.messages[0]
    assert "backing record index 0/path 'shared.cif'" in caplog.messages[0]
    assert f"stage {stage}" in caplog.messages[0]
    assert "next sample" not in caplog.messages[0]


@pytest.mark.parametrize("stage", ["processor", "unknown"])
def test_cluster_dataset_row_specific_timeout_allows_later_row_for_same_cif(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    first = _entry(chain_key="A", reference_instance_ids=("A",))
    same_cif = _entry(chain_key="B", entity_1_id="2", reference_instance_ids=("B",))
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.records = ["shared.cif"]
    dataset.entries = [first, same_cif]
    dataset.forward_indices = True
    dataset.processing_timeout_seconds = 1.0
    dataset.max_consecutive_processing_errors = None
    dataset.training_log_file = None
    calls: list[tuple[int, ClusterIndex]] = []

    def process(record_index: int, **kwargs: object) -> str:
        indices = kwargs["indices"]
        assert isinstance(indices, ClusterIndex)
        calls.append((record_index, indices))
        if indices is first:
            raise StructureProcessingTimeoutError("row-specific timeout", stage=stage)
        return indices.chain_key

    monkeypatch.setattr(dataset, "_process_record", process)

    assert dataset[0] == "B"
    assert calls == [(0, first), (0, same_cif)]


def test_cluster_dataset_timeout_exhaustion_reports_scanned_and_processed_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(chain_key="A", reference_instance_ids=("A",))
    same_cif = _entry(chain_key="B", entity_1_id="2", reference_instance_ids=("B",))
    next_cif = _entry(record_index=1, pdb_id="2xyz", chain_key="C", reference_instance_ids=("C",))
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.records = ["shared.cif", "other.cif"]
    dataset.entries = [first, same_cif, next_cif]
    dataset.forward_indices = True
    dataset.processing_timeout_seconds = 1.0
    dataset.max_consecutive_processing_errors = None
    dataset.training_log_file = None
    calls: list[int] = []

    def timeout(record_index: int, **kwargs: object) -> None:
        calls.append(record_index)
        raise StructureProcessingTimeoutError(f"record {record_index} timed out", stage="read_structure")

    monkeypatch.setattr(dataset, "_process_record", timeout)

    with pytest.raises(ProcessingTimeoutExhaustionError) as exc_info:
        dataset[0]

    assert calls == [0, 1]
    assert "scanning 3 logical candidates and processing 2 attempts" in str(exc_info.value)
    assert exc_info.value.timeout_count == 2
    assert exc_info.value.scanned_candidates == 3
    assert exc_info.value.processed_attempts == 2


def test_cluster_dataset_deduplicates_recoverable_read_error_for_shared_backing_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(chain_key="A", reference_instance_ids=("A",))
    same_cif = _entry(chain_key="B", entity_1_id="2", reference_instance_ids=("B",))
    next_cif = _entry(record_index=1, pdb_id="2xyz", chain_key="C", reference_instance_ids=("C",))
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.records = ["shared.cif", "success.cif"]
    dataset.entries = [first, same_cif, next_cif]
    dataset.forward_indices = True
    dataset.processing_timeout_seconds = None
    dataset.max_consecutive_processing_errors = 5
    dataset.training_log_file = None
    dataset.ccd_revision = None
    dataset.ccd_snapshot_path = None
    dataset.seq_to_pdb_index = None
    dataset.msa_path = None
    dataset.processor_kwargs = {}
    read_calls: list[str] = []
    processor_calls: list[ClusterIndex] = []
    structure = SimpleNamespace(system=SimpleNamespace(entity_type=[]))

    def read(path: str, **kwargs: object) -> object:
        read_calls.append(path)
        if path == "shared.cif":
            raise ValueError("duplicate source atom declaration")
        return structure

    def processor(value: object, **kwargs: object) -> str:
        assert value is structure
        indices = kwargs["indices"]
        assert isinstance(indices, ClusterIndex)
        processor_calls.append(indices)
        return indices.chain_key

    monkeypatch.setattr("ofoldx.io.read_structure", read)
    dataset.processor = processor

    assert dataset[0] == "C"
    assert read_calls == ["shared.cif", "success.cif"]
    assert processor_calls == [next_cif]


def test_cluster_dataset_recoverable_processor_error_remains_row_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(chain_key="A", reference_instance_ids=("A",))
    same_cif = _entry(chain_key="B", entity_1_id="2", reference_instance_ids=("B",))
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.records = ["shared.cif"]
    dataset.entries = [first, same_cif]
    dataset.forward_indices = True
    dataset.processing_timeout_seconds = None
    dataset.max_consecutive_processing_errors = 5
    dataset.training_log_file = None
    dataset.ccd_revision = None
    dataset.ccd_snapshot_path = None
    dataset.seq_to_pdb_index = None
    dataset.msa_path = None
    dataset.processor_kwargs = {}
    read_calls: list[str] = []
    processor_calls: list[ClusterIndex] = []
    structure = SimpleNamespace(system=SimpleNamespace(entity_type=[]))

    def read(path: str, **kwargs: object) -> object:
        read_calls.append(path)
        return structure

    def processor(value: object, **kwargs: object) -> str:
        indices = kwargs["indices"]
        assert isinstance(indices, ClusterIndex)
        processor_calls.append(indices)
        if indices is first:
            raise ValueError("row-specific processor failure")
        return indices.chain_key

    monkeypatch.setattr("ofoldx.io.read_structure", read)
    dataset.processor = processor

    assert dataset[0] == "B"
    assert read_calls == ["shared.cif", "shared.cif"]
    assert processor_calls == [first, same_cif]


def test_cluster_dataset_unique_backing_exhaustion_preserves_last_processed_row_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(chain_key="A", reference_instance_ids=("A",))
    same_cif = _entry(chain_key="B", entity_1_id="2", reference_instance_ids=("B",))
    next_cif = _entry(record_index=1, pdb_id="2xyz", chain_key="C", reference_instance_ids=("C",))
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.records = ["shared.cif", "other.cif"]
    dataset.entries = [first, same_cif, next_cif]
    dataset.forward_indices = True
    dataset.processing_timeout_seconds = None
    dataset.max_consecutive_processing_errors = 5
    dataset.training_log_file = None
    dataset.ccd_revision = None
    dataset.ccd_snapshot_path = None
    dataset.seq_to_pdb_index = None
    dataset.msa_path = None
    dataset.processor_kwargs = {}
    calls: list[str] = []

    def read(path: str, **kwargs: object) -> object:
        calls.append(path)
        raise ValueError(f"bad backing record {path}")

    monkeypatch.setattr("ofoldx.io.read_structure", read)
    dataset.processor = lambda value, **kwargs: value

    with pytest.raises(ConsecutiveDataLoadingErrors) as exc_info:
        dataset[0]
    assert calls == ["shared.cif", "other.cif"]
    assert type(exc_info.value.__cause__) is ValueError
    assert str(exc_info.value.__cause__) == "bad backing record other.cif"


def test_runner_uses_cluster_sampler_without_forwarding_reserved_config(tmp_path: Path) -> None:
    cif = _cif(tmp_path / "1abc.cif", sequences=("AAAAAAAAAA",), origins=(0.0,))
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("1ABC_1\n", encoding="ascii")
    runner = Runner(
        {
            "model": {"name": "proteinmpnn"},
            "steps": 1,
            "seed": 11,
            "data": {
                "train": {
                    "source": str(cif),
                    "cluster_sampling": {
                        "enabled": True,
                        "cluster_file": str(cluster),
                        "epoch_size": 3,
                    },
                }
            },
            "optim": {"type": "torch_adamw", "lr": 1e-3},
            "dataloader": {"batch_size": 1, "drop_last": False, "num_workers": 0},
            "logging": {"enabled": False},
            "workspace": {"root": str(tmp_path / "workspace")},
            "ckpt": {"enabled": False},
        }
    )
    try:
        runner.build_dataloaders()
        sampler = runner.dataloaders["train"].sampler
        assert isinstance(sampler, ClusterWeightedSampler)
        assert runner.datasets["train"].cluster_indices_status == "created"
        assert runner.datasets["train"].cluster_indices_file.parent == (
            tmp_path / "workspace" / ".cache" / "ofoldx" / "cluster-indices"
        )
        assert runner._forward_cluster_indices is False
        assert len(sampler) == 3
        batch = next(iter(runner.dataloaders["train"]))
        assert batch is not None
    finally:
        runner.close()


def test_runner_routes_protenix_split_crop_args_and_cluster_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class Processor:
        def process_training(self, structure, **kwargs):
            return structure, kwargs

    class Split(dict):
        source = "train.cif"
        cluster_sampling = SimpleNamespace(
            enabled=True,
            cluster_file="clusters.txt",
            contact_cutoff=5.0,
            include_all_chain_pairs=False,
            exclude_mol_types=("ions",),
            beta={"chain": 0.5, "interface": 1.0},
            alpha={"prot": 3.0, "nuc": 3.0, "ligand": 1.0},
        )

    class Config(dict):
        data = {
            "train": Split(
                source="train.cif",
                max_tokens=128,
                crop_method_weights=(0.2, 0.4, 0.4),
                cluster_sampling=Split.cluster_sampling,
            )
        }

    marker = object()

    def clustered_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return marker

    monkeypatch.setattr("ofoldx.runners.dataset.ClusteredStructureDataset", clustered_dataset)
    fake_runner = SimpleNamespace(
        processor=Processor(),
        _process_training=True,
        _forward_cluster_indices=True,
        config=Config(),
    )
    datasets = Runner.build_datasets(fake_runner)  # type: ignore[arg-type]
    assert datasets == {"train": marker}
    assert captured["args"][1].__self__ is fake_runner.processor
    assert captured["kwargs"]["processor_kwargs"] == {
        "max_tokens": 128,
        "crop_method_weights": (0.2, 0.4, 0.4),
    }
    assert captured["kwargs"]["forward_indices"] is True


def test_runner_ordinary_split_keeps_default_sampler(tmp_path: Path) -> None:
    cif = _cif(tmp_path / "plain.cif", sequences=("AAAAAAAAAA",), origins=(0.0,))
    runner = Runner(
        {
            "model": {"name": "proteinmpnn"},
            "steps": 1,
            "data": {"train": {"source": str(cif)}},
            "optim": {"type": "torch_adamw", "lr": 1e-3},
            "dataloader": {"batch_size": 1, "drop_last": False, "num_workers": 0},
            "logging": {"enabled": False},
            "workspace": {"root": str(tmp_path / "workspace")},
            "ckpt": {"enabled": False},
        }
    )
    try:
        runner.build_dataloaders()
        assert not isinstance(runner.dataloaders["train"].sampler, ClusterWeightedSampler)
        assert next(iter(runner.dataloaders["train"])) is not None
    finally:
        runner.close()


def test_runner_rejects_cluster_sampling_on_validation_split(tmp_path: Path) -> None:
    cif = _cif(tmp_path / "val.cif", sequences=("AAAAAAAAAA",), origins=(0.0,))
    cluster = tmp_path / "clusters.txt"
    cluster.write_text("VAL_1\n", encoding="ascii")
    with pytest.raises(ValueError, match="only supported for training"):
        Runner(
            {
                "model": {"name": "proteinmpnn"},
                "steps": 1,
                "data": {
                    "val": {
                        "source": str(cif),
                        "cluster_sampling": {"enabled": True, "cluster_file": str(cluster)},
                    }
                },
                "optim": {"type": "torch_adamw", "lr": 1e-3},
                "dataloader": {"batch_size": 1, "drop_last": False, "num_workers": 0},
                "logging": {"enabled": False},
                "workspace": {"root": str(tmp_path / "workspace")},
                "ckpt": {"enabled": False},
            },
        )
