#!/usr/bin/env python
r"""Compare OFoldX ProteinMpnn with a local dauparas/ProteinMPNN checkout.

This is a manual, on-demand scientific diagnostic. It uses the checkout's 5L33
fixture, writes no report, and is not a test, CI, registration, or publication
gate. It exercises the frozen §4.7 / §4.7.1 public surface: the two Processor
arms, differentiable ``forward`` returning ``MpnnOutput``, public ``encode``,
the four ``MpnnProbabilityResult`` methods, the unified ``sample``, the
``global_omit_tokens`` lowering, and the ``ProteinMpnnDesignAndPackPipeline``
composition. Numerics are still cross-checked against the upstream checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ofoldx import AutoModel, AutoProcessor
from ofoldx import pipeline as load_pipeline  # noqa: E402
from ofoldx.data.conditions import (  # noqa: E402
    AllowedResidueIdentities,
    TiedResidueIdentities,
)
from ofoldx.data.molecular import (  # noqa: E402
    AtomSite,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ofoldx.data.structure import Structure  # noqa: E402
from ofoldx.io import read_pdb  # noqa: E402
from ofoldx.models.mpnn_packer import (  # noqa: E402
    MpnnPackerConfig,
    MpnnPackerModel,
    MpnnPackerPipeline,
    MpnnPackerProcessor,
)
from ofoldx.models.mpnn_packer.convert import convert_mpnn_packer_state_dict  # noqa: E402
from ofoldx.models.proteinmpnn import (  # noqa: E402
    ProteinMpnnConfig,
    ProteinMpnnDesignAndPackPipeline,
    ProteinMpnnDesignResult,
    ProteinMpnnInput,
    ProteinMpnnModel,
    ProteinMpnnPipeline,
    ProteinMpnnProcessor,
    ProteinMpnnProcessorConfig,
)
from ofoldx.models.proteinmpnn.convert import (  # noqa: E402
    convert_checkpoint,
    convert_state_dict,
)
from ofoldx.models.proteinmpnn.model import tied_groups_from_ids  # noqa: E402
from ofoldx.models.proteinmpnn.pipeline import (  # noqa: E402
    _candidate_protein_sequences,
    _global_omit_token_mask,
)
from ofoldx.modules.mpnn import (  # noqa: E402
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
    MpnnOutput,
    MpnnSamplingOutput,
)
from ofoldx.results import MpnnProbabilityResult  # noqa: E402
from ofoldx.results import (  # noqa: E402
    DesignCandidate,
    PackPrediction,
    PackResult,
    Score,
    ScoreResult,
)
from scripts.parity.paths import reference_root, weight_path  # noqa: E402

FLOAT32_ATOL = 1e-5
FLOAT32_RTOL = 1.3e-6


def _check(label: str, passed: bool | Tensor) -> None:
    passed = bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    if not passed:
        raise AssertionError(label)


def _compare(
    label: str,
    ours: Tensor,
    reference: Tensor,
    *,
    exact: bool = False,
) -> None:
    ours = ours.detach().cpu()
    reference = reference.detach().cpu()
    if ours.shape != reference.shape:
        print(f"[FAIL] {label} shape={tuple(ours.shape)} " f"reference_shape={tuple(reference.shape)}")
        raise AssertionError(label)
    if ours.dtype != reference.dtype:
        print(f"[FAIL] {label} dtype={ours.dtype} " f"reference_dtype={reference.dtype}")
        raise AssertionError(label)

    if exact:
        passed = torch.equal(ours, reference)
        detail = f"shape={tuple(ours.shape)}"
    else:
        passed = torch.allclose(
            ours,
            reference,
            atol=FLOAT32_ATOL,
            rtol=FLOAT32_RTOL,
        )
        delta = (ours.float() - reference.float()).abs()
        max_error = 0.0 if delta.numel() == 0 else float(delta.max())
        detail = f"shape={tuple(ours.shape)} max_abs={max_error:.3e}"
    print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}")
    if not passed:
        raise AssertionError(label)


def _component_signature(component: ChemicalComponent) -> tuple:
    attachments = tuple(
        None if attachment is None else (attachment.atom_id, tuple(sorted(attachment.leaving_atom_ids)))
        for attachment in (component.previous_attachment, component.next_attachment)
    )
    bonds = (
        None
        if component.bonds is None
        else tuple(
            sorted(
                (
                    bond.atom_a,
                    bond.atom_b,
                    bond.order,
                    bond.aromatic,
                    bond.stereo,
                    bond.stereo_atoms,
                )
                for bond in component.bonds
            )
        )
    )
    return (
        component.component_id,
        tuple(
            (
                atom.atom_id,
                atom.element,
                atom.isotope,
                atom.formal_charge,
                atom.radical_electrons,
                atom.explicit_hydrogens,
                atom.implicit_hydrogens,
                atom.implicit_hydrogens_allowed,
                atom.aromatic,
                atom.stereo,
            )
            for atom in component.atoms
        ),
        bonds,
        attachments,
        component.reference_coordinates,
    )


def _entity_signature(entity: PolymerSequence | Molecule) -> tuple:
    if isinstance(entity, PolymerSequence):
        return (
            "polymer",
            entity.polymer_type,
            tuple(_component_signature(component) for component in entity.components),
        )
    return (
        "molecule",
        tuple(_component_signature(component) for component in entity.components),
        tuple(
            sorted(
                (
                    bond.component_a,
                    bond.atom_a,
                    bond.component_b,
                    bond.atom_b,
                    bond.order,
                    bond.aromatic,
                    bond.stereo,
                    bond.stereo_atoms,
                )
                for bond in entity.bonds
            )
        ),
    )


def _site_signature(site: AtomSite) -> tuple[str, int, str]:
    return site.instance_id, site.component_index, site.atom_id


def _system_signature(system: MolecularSystem) -> tuple:
    return (
        tuple((entity_id, _entity_signature(entity)) for entity_id, entity in system.entities.items()),
        tuple(system.instances.items()),
        tuple(
            sorted(
                (
                    _site_signature(bond.site_a),
                    _site_signature(bond.site_b),
                    bond.order,
                    bond.aromatic,
                    bond.stereo,
                    None if bond.stereo_sites is None else tuple(_site_signature(site) for site in bond.stereo_sites),
                    bond.origin,
                )
                for bond in system.bonds
            )
        ),
    )


def _compare_structure(label: str, ours: Structure, reference: Structure) -> None:
    _check(f"{label}: concrete type", type(ours) is Structure and type(reference) is Structure)
    _check(
        f"{label}: molecular system",
        _system_signature(ours.system) == _system_signature(reference.system),
    )
    _check(
        f"{label}: atom sites",
        tuple(_site_signature(site) for site in ours.sites) == tuple(_site_signature(site) for site in reference.sites),
    )
    _compare(
        f"{label}: coordinates",
        torch.as_tensor(ours.coordinates),
        torch.as_tensor(reference.coordinates),
        exact=True,
    )
    _compare(
        f"{label}: coordinate mask",
        torch.as_tensor(ours.coordinate_mask),
        torch.as_tensor(reference.coordinate_mask),
        exact=True,
    )
    _check(
        f"{label}: occupancy presence",
        (ours.occupancy is None) == (reference.occupancy is None),
    )
    if ours.occupancy is not None and reference.occupancy is not None:
        _compare(
            f"{label}: occupancy",
            torch.as_tensor(ours.occupancy),
            torch.as_tensor(reference.occupancy),
            exact=True,
        )

    def residue_number_signature(structure: Structure) -> tuple | None:
        if structure.residue_numbers is None:
            return None
        return tuple(
            sorted(
                (
                    component_reference,
                    residue_number.number,
                    residue_number.insertion_code,
                )
                for component_reference, residue_number in structure.residue_numbers.items()
            )
        )

    _check(
        f"{label}: residue numbers",
        residue_number_signature(ours) == residue_number_signature(reference),
    )


def _reference_signature(reference: str | tuple[str, int] | AtomSite) -> tuple:
    if isinstance(reference, str):
        return "name", reference
    if isinstance(reference, tuple):
        return "component", reference[0], reference[1]
    return "atom", *_site_signature(reference)


def _scope_signature(scope: frozenset | None) -> tuple | None:
    if scope is None:
        return None
    return tuple(sorted((_reference_signature(reference) for reference in scope), key=repr))


def _compare_score(label: str, ours: Score, reference: Score) -> None:
    _check(f"{label}: concrete type", type(ours) is Score and type(reference) is Score)
    _compare(
        f"{label}: value",
        torch.tensor(ours.value),
        torch.tensor(reference.value),
    )
    _check(f"{label}: unit", ours.unit == reference.unit)
    _check(f"{label}: preference", ours.preferred == reference.preferred)
    _check(
        f"{label}: scope",
        _scope_signature(ours.scope) == _scope_signature(reference.scope),
    )
    _check(
        f"{label}: evaluations presence",
        (ours.evaluations is None) == (reference.evaluations is None),
    )
    if ours.evaluations is not None and reference.evaluations is not None:
        _compare(
            f"{label}: evaluations",
            torch.from_numpy(ours.evaluations),
            torch.from_numpy(reference.evaluations),
        )


def _compare_score_result(
    label: str,
    ours: ScoreResult,
    reference: ScoreResult,
) -> None:
    _check(
        f"{label}: concrete type",
        type(ours) is ScoreResult and type(reference) is ScoreResult,
    )
    _compare_structure(f"{label}: target", ours.target, reference.target)
    _check(f"{label}: score names", set(ours.scores) == set(reference.scores))
    for score_name in ours.scores:
        _compare_score(
            f"{label}: {score_name}",
            ours.scores[score_name],
            reference.scores[score_name],
        )
    _check(f"{label}: no metrics", not ours.metrics and not reference.metrics)


def _compare_proteinmpnn_design_result(
    label: str,
    ours: ProteinMpnnDesignResult,
    reference: ProteinMpnnDesignResult,
) -> None:
    _check(
        f"{label}: concrete type",
        type(ours) is ProteinMpnnDesignResult and type(reference) is ProteinMpnnDesignResult,
    )
    _check(f"{label}: candidate count", len(ours.candidates) == len(reference.candidates))
    for index, (candidate, expected) in enumerate(zip(ours.candidates, reference.candidates)):
        candidate_label = f"{label}: candidate {index}"
        _check(
            f"{candidate_label}: concrete type",
            type(candidate) is DesignCandidate and type(expected) is DesignCandidate,
        )
        _check(
            f"{candidate_label}: molecular system",
            _system_signature(candidate.system) == _system_signature(expected.system),
        )
        _check(
            f"{candidate_label}: score names",
            set(candidate.scores) == set(expected.scores),
        )
        for score_name in candidate.scores:
            _compare_score(
                f"{candidate_label}: {score_name}",
                candidate.scores[score_name],
                expected.scores[score_name],
            )
    _compare_score_result(f"{label}: input score", ours.input_score, reference.input_score)


def _load_reference(root: Path) -> tuple[ModuleType, str, bool]:
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    path = root / "protein_mpnn_utils.py"
    spec = importlib.util.spec_from_file_location("proteinmpnn_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import the ProteinMPNN reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, revision, dirty


def _load_models(
    label: str,
    checkpoint_path: Path,
    reference: ModuleType,
    device: torch.device,
) -> tuple[ProteinMpnnModel, ProteinMpnnProcessor, torch.nn.Module]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint["model_state_dict"]
    config = ProteinMpnnConfig(
        hidden_size=state_dict["W_s.weight"].shape[1],
        num_encoder_layers=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("encoder_layers.")}),
        num_decoder_layers=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("decoder_layers.")}),
        num_neighbors=checkpoint["num_edges"],
        ca_only="W_v.weight" in state_dict,
    )

    model = ProteinMpnnModel(config)
    model.load_state_dict(convert_state_dict(state_dict), strict=True, assign=True)
    checkpoint_dtype = state_dict["W_s.weight"].dtype
    _check(
        f"{label}: converted checkpoint dtype",
        model.token_embedding.weight.dtype == checkpoint_dtype,
    )
    model = model.to(device).eval()

    upstream = reference.ProteinMPNN(
        num_letters=21,
        node_features=config.hidden_size,
        edge_features=config.hidden_size,
        hidden_dim=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        k_neighbors=config.num_neighbors,
        augment_eps=0.0,
        dropout=config.dropout,
        ca_only=config.ca_only,
    )
    upstream.load_state_dict(state_dict, strict=True, assign=True)
    upstream = upstream.to(device).eval()
    processor = ProteinMpnnProcessor(
        ProteinMpnnProcessorConfig(ca_only=config.ca_only),
        device=device,
        dtype=checkpoint_dtype,
    )
    print(f"[checkpoint] PASS {label}: strict conversion and load")
    return model, processor, upstream


def _repeat(value: Tensor, count: int) -> Tensor:
    return value.repeat_interleave(count, dim=0)


def _decoding_ranks(batch: ProteinMpnnInput, count: int) -> Tensor:
    length = batch.sequence_tokens.shape[1]
    tied = tuple(tuple(sorted(group)) for group in tied_groups_from_ids(batch.tied_group_ids)[0])
    grouped = {member for group in tied for member in group}
    units = (*tied, *((index,) for index in range(length) if index not in grouped))
    design = (batch.valid_mask & batch.design_mask)[0].tolist()
    fixed = [unit for unit in units if not all(design[index] for index in unit)]
    designed = [unit for unit in units if all(design[index] for index in unit)]
    ranks = torch.empty(
        count,
        length,
        dtype=torch.long,
        device=batch.sequence_tokens.device,
    )
    for row in range(count):
        fixed_offset = row % len(fixed) if fixed else 0
        designed_offset = row % len(designed) if designed else 0
        order = fixed[fixed_offset:] + fixed[:fixed_offset] + designed[designed_offset:] + designed[:designed_offset]
        for rank, unit in enumerate(order):
            ranks[row, list(unit)] = rank
    return ranks


def _reference_features(
    reference: ModuleType,
    pdb: Path,
    device: torch.device,
    *,
    ca_only: bool,
) -> tuple:
    parsed = reference.parse_PDB(str(pdb), ca_only=ca_only)[0]
    return reference.tied_featurize(
        [parsed],
        device,
        None,
        ca_only=ca_only,
    )


def _compare_processor(
    label: str,
    batch: ProteinMpnnInput,
    upstream: tuple,
) -> None:
    coordinates, sequence, valid = upstream[0], upstream[1], upstream[2]
    chain_mask, chains = upstream[4], upstream[5]
    position_mask, residue_indices = upstream[10], upstream[12]
    _compare(f"{label}: coordinates", batch.backbone_coordinates, coordinates)
    _compare(f"{label}: sequence tokens", batch.sequence_tokens, sequence, exact=True)
    _compare(f"{label}: valid mask", batch.valid_mask, valid.bool(), exact=True)
    _compare(
        f"{label}: design mask",
        batch.design_mask,
        (chain_mask * position_mask * valid).bool(),
        exact=True,
    )
    _compare(
        f"{label}: residue indices",
        batch.residue_indices,
        residue_indices,
        exact=True,
    )
    _compare(
        f"{label}: chain indices",
        batch.chain_indices + 1,
        chains,
        exact=True,
    )


def _pdb_line(
    record: str,
    serial: int,
    atom_id: str,
    component_id: str,
    instance_id: str,
    number: int,
    insertion_code: str,
    coordinates: tuple[float, float, float],
    occupancy: float,
    element: str,
) -> str:
    x, y, z = coordinates
    return (
        f"{record:<6}{serial:>5} {atom_id:<4} {component_id:>3} {instance_id:1}"
        f"{number:>4}{insertion_code:1}   {x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{occupancy:>6.2f}{20.0:>6.2f}{'':10}{element:>2}\n"
    )


def _processor_checks(
    label: str,
    processor: ProteinMpnnProcessor,
    reference: ModuleType,
    pdb: Path,
    batch: ProteinMpnnInput,
) -> dict[str, tuple[str, ...]]:
    # `processor(path)` uses the original fixed-column ProteinMPNN arm and must
    # match upstream `tied_featurize` exactly.
    upstream = _reference_features(
        reference,
        pdb,
        batch.backbone_coordinates.device,
        ca_only=processor.config.ca_only,
    )
    _compare_processor(f"{label}: reference PDB", batch, upstream)

    # §4.7 (~2929): `processor(path) != processor(mpnn.read_pdb(path))`. The record
    # is `eq=False`, so identity `!=` is trivial; the meaningful contract is that
    # the two arms produce genuinely different lowerings. The path arm assigns
    # `100 * c + i` positions while the Structure arm derives positions from source
    # residue numbering, so `residue_indices` diverges on the same fixture.
    structure_arm = processor(read_pdb(pdb))
    _check(
        f"{label}: two arms are ProteinMpnnInput",
        type(batch) is ProteinMpnnInput and type(structure_arm) is ProteinMpnnInput,
    )
    _check(
        f"{label}: path and Structure arms are not equivalent (§4.7)",
        batch.residue_indices.shape == structure_arm.residue_indices.shape
        and not torch.equal(batch.residue_indices, structure_arm.residue_indices),
    )

    first_reference = next(
        component_reference
        for component_reference, valid in zip(
            batch.component_references[0],
            batch.valid_mask[0].tolist(),
        )
        if component_reference is not None and valid
    )
    instance_id = first_reference[0]
    system = batch.structures[0].system
    entity = system.entities[system.instances[instance_id]]
    component_ids = [component.component_id for component in entity.components]
    replacement = "GLY" if component_ids[first_reference[1]] == "ALA" else "ALA"
    component_ids[first_reference[1]] = replacement
    alternate_sequence = {instance_id: tuple(component_ids)}
    alternate = processor(batch.structures[0], sequence=alternate_sequence)
    row = alternate.component_references[0].index(first_reference)
    _check(
        f"{label}: alternate sequence and canonical target",
        int(alternate.sequence_tokens[0, row]) == AMINO_ACID_TOKEN_BY_COMPONENT[replacement]
        and alternate.structures[0].system.component(first_reference).component_id == replacement,
    )

    rows = (
        ("ATOM", "N", "ALA", "A", 1, "", (0.0, 0.0, 0.0), 1.0, "N"),
        ("ATOM", "CA", "ALA", "A", 1, "", (1.0, 0.0, 0.0), 0.0, "C"),
        ("ATOM", "CA", "ALA", "A", 1, "", (9.0, 9.0, 9.0), 1.0, "C"),
        ("ATOM", "C", "ALA", "A", 1, "", (2.0, 0.0, 0.0), 1.0, "C"),
        ("ATOM", "O", "ALA", "A", 1, "", (3.0, 0.0, 0.0), 1.0, "O"),
        ("HETATM", "N", "MSE", "A", 1, "A", (4.0, 0.0, 0.0), 1.0, "N"),
        ("HETATM", "CA", "MSE", "A", 1, "A", (5.0, 0.0, 0.0), 1.0, "C"),
        ("HETATM", "C", "MSE", "A", 1, "A", (6.0, 0.0, 0.0), 1.0, "C"),
        ("HETATM", "O", "MSE", "A", 1, "A", (7.0, 0.0, 0.0), 1.0, "O"),
        ("ATOM", "N", "GLY", "A", 3, "", (8.0, 0.0, 0.0), 1.0, "N"),
        ("ATOM", "CA", "GLY", "A", 3, "", (9.0, 0.0, 0.0), 1.0, "C"),
        ("ATOM", "C", "GLY", "A", 3, "", (10.0, 0.0, 0.0), 1.0, "C"),
        ("ATOM", "N", "VAL", "B", 1, "", (0.0, 4.0, 0.0), 1.0, "N"),
        ("ATOM", "CA", "VAL", "B", 1, "", (1.0, 4.0, 0.0), 1.0, "C"),
        ("ATOM", "C", "VAL", "B", 1, "", (2.0, 4.0, 0.0), 1.0, "C"),
        ("ATOM", "O", "VAL", "B", 1, "", (3.0, 4.0, 0.0), 1.0, "O"),
    )
    with tempfile.TemporaryDirectory() as directory:
        tiny_path = Path(directory) / "tiny.pdb"
        tiny_path.write_text(
            "".join(_pdb_line(record, serial, *values) for serial, (record, *values) in enumerate(rows, start=1))
        )
        tiny = processor(tiny_path)
        tiny_upstream = _reference_features(
            reference,
            tiny_path,
            tiny.backbone_coordinates.device,
            ca_only=processor.config.ca_only,
        )
        _compare_processor(f"{label}: tiny PDB", tiny, tiny_upstream)

    expected_references = (("A", 0), ("A", 1), None, ("A", 2), ("B", 0))
    expected_valid = (True, True, False, True, True) if processor.config.ca_only else (True, True, False, False, True)
    _check(
        f"{label}: gap and insertion references",
        tiny.component_references[0] == expected_references,
    )
    _compare(
        f"{label}: source-number residue indices",
        tiny.residue_indices,
        torch.tensor([[0, 1, 2, 3, 104]], device=tiny.residue_indices.device),
        exact=True,
    )
    _compare(
        f"{label}: full/CA validity",
        tiny.valid_mask,
        torch.tensor([expected_valid], device=tiny.valid_mask.device),
        exact=True,
    )
    _check(
        f"{label}: MSE identity and MET token",
        tiny.structures[0].system.component(("A", 1)).component_id == "MSE"
        and int(tiny.sequence_tokens[0, 1]) == AMINO_ACID_TOKEN_BY_COMPONENT["MET"],
    )
    first_ca = tiny.backbone_coordinates[0, 0] if processor.config.ca_only else tiny.backbone_coordinates[0, 0, 1]
    _compare(
        f"{label}: first atom wins and occupancy is ignored",
        first_ca,
        first_ca.new_tensor((1.0, 0.0, 0.0)),
        exact=True,
    )
    return alternate_sequence


def _upstream_forward(
    upstream: torch.nn.Module,
    batch: ProteinMpnnInput,
    ranks: Tensor,
    *,
    sequence_tokens: Tensor | None = None,
    backbone_noise: float = 0.0,
    seed: int = 0,
) -> Tensor:
    r"""Sequence-conditioned autoregressive log-probabilities for a full order."""

    count = ranks.shape[0]
    dtype = batch.backbone_coordinates.dtype
    sequence_tokens = _repeat(batch.sequence_tokens, count) if sequence_tokens is None else sequence_tokens
    upstream.features.augment_eps = backbone_noise
    torch.manual_seed(seed)
    try:
        return upstream(
            _repeat(batch.backbone_coordinates, count),
            sequence_tokens,
            _repeat(batch.valid_mask.to(dtype), count),
            _repeat(batch.design_mask.to(dtype), count),
            _repeat(batch.residue_indices, count),
            _repeat(batch.chain_indices + 1, count),
            torch.zeros_like(ranks, dtype=dtype),
            use_input_decoding_order=True,
            decoding_order=torch.argsort(ranks, dim=-1),
        )
    finally:
        upstream.features.augment_eps = 0.0


def _upstream_sequence_free_forward(
    upstream: torch.nn.Module,
    reference: ModuleType,
    batch: ProteinMpnnInput,
    ranks: Tensor,
) -> Tensor:
    r"""Reconstruct sequence-free autoregressive log-probabilities from upstream layers.

    ProteinMPNN exposes no single method for the frozen
    ``sequence_free_autoregressive_probabilities`` computation: it keeps the causal
    forward mask (a position sees only earlier-decoded neighbors) while never
    conditioning the decoder on any token identity. `unconditional_probs` is the
    order-free sequence-free pass (``order_mask_backward = 0``), and `forward`
    conditions on the sequence, so neither is a clean analogue. This helper recomposes
    ProteinMPNN's own encoder, decoder, and einsum permutation mask with
    ``mask_bw = 0`` and zero sequence embeddings, exercising the same upstream weights
    and mask construction as an independent cross-check.
    """

    count = ranks.shape[0]
    dtype = batch.backbone_coordinates.dtype
    device = batch.backbone_coordinates.device
    coordinates = _repeat(batch.backbone_coordinates, count)
    mask = _repeat(batch.valid_mask.to(dtype), count)
    residue_indices = _repeat(batch.residue_indices, count)
    chain_indices = _repeat(batch.chain_indices + 1, count)
    decoding_order = torch.argsort(ranks, dim=-1)

    edge_features, neighbor_indices = upstream.features(coordinates, mask, residue_indices, chain_indices)
    node_states = torch.zeros(
        edge_features.shape[0],
        edge_features.shape[1],
        edge_features.shape[-1],
        device=device,
    )
    edge_states = upstream.W_e(edge_features)
    attend = reference.gather_nodes(mask.unsqueeze(-1), neighbor_indices).squeeze(-1)
    attend = mask.unsqueeze(-1) * attend
    for layer in upstream.encoder_layers:
        node_states, edge_states = layer(node_states, edge_states, neighbor_indices, mask, attend)

    encoder_edges = reference.cat_neighbors_nodes(torch.zeros_like(node_states), edge_states, neighbor_indices)
    encoder_context = reference.cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
    mask_size = neighbor_indices.shape[1]
    permutation = F.one_hot(decoding_order, num_classes=mask_size).to(dtype)
    order_mask_backward = torch.einsum(
        "ij,biq,bjp->bqp",
        1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
        permutation,
        permutation,
    )
    backward_attend = torch.gather(order_mask_backward, 2, neighbor_indices).unsqueeze(-1)
    mask_1d = mask.view(mask.shape[0], mask.shape[1], 1, 1)
    forward_context = mask_1d * (1.0 - backward_attend) * encoder_context
    for layer in upstream.decoder_layers:
        node_states = layer(node_states, forward_context, mask)
    return F.log_softmax(upstream.W_out(node_states), dim=-1)


def _encode_checks(
    label: str,
    model: ProteinMpnnModel,
    batch: ProteinMpnnInput,
) -> None:
    r"""Exercise the public `encode` and the differentiable `forward` for `B >= 1`."""

    length = batch.sequence_tokens.shape[1]
    hidden = model.config.hidden_size
    node_states, edge_states, neighbor_indices = model.encode(batch)
    _check(
        f"{label}: encode returns three tensors",
        isinstance(node_states, Tensor) and isinstance(edge_states, Tensor) and isinstance(neighbor_indices, Tensor),
    )
    _check(
        f"{label}: encode node/edge/neighbor shapes",
        tuple(node_states.shape) == (1, length, hidden)
        and tuple(edge_states.shape) == (1, length, model.config.num_neighbors, hidden)
        and tuple(neighbor_indices.shape) == (1, length, model.config.num_neighbors),
    )

    # `encode` and `forward` accept a heterogeneous B >= 1 batch. Build a two-row
    # batch by aligning every tensor field and reuse the same canonical metadata.
    doubled = ProteinMpnnInput(
        backbone_coordinates=_repeat(batch.backbone_coordinates, 2),
        sequence_tokens=_repeat(batch.sequence_tokens, 2),
        valid_mask=_repeat(batch.valid_mask, 2),
        residue_indices=_repeat(batch.residue_indices, 2),
        chain_indices=_repeat(batch.chain_indices, 2),
        design_mask=_repeat(batch.design_mask, 2),
        structures=batch.structures * 2,
        component_references=batch.component_references * 2,
        tied_group_ids=_repeat(batch.tied_group_ids, 2),
    )
    node_states_batch, _edge_states_batch, _neighbors_batch = model.encode(doubled)
    _check(
        f"{label}: encode accepts B >= 1",
        tuple(node_states_batch.shape) == (2, length, hidden),
    )
    _compare(
        f"{label}: encode row 0 matches single input",
        node_states_batch[:1],
        node_states,
        exact=True,
    )

    ranks = _decoding_ranks(doubled, 2)
    output = model(doubled, decoding_ranks=ranks)
    _check(f"{label}: forward returns MpnnOutput", type(output) is MpnnOutput)
    _check(
        f"{label}: forward B>=1 logit and rank shapes",
        tuple(output.logits.shape) == (2, length, 21) and tuple(output.decoding_ranks.shape) == (2, length),
    )
    _compare(f"{label}: forward returns exact ranks", output.decoding_ranks, ranks, exact=True)
    _compare(
        f"{label}: forward log-probability property",
        output.log_probabilities,
        output.logits.log_softmax(dim=-1),
        exact=True,
    )
    _compare(
        f"{label}: forward probability property",
        output.probabilities,
        output.logits.softmax(dim=-1),
        exact=True,
    )


def _score_checks(
    label: str,
    model: ProteinMpnnModel,
    upstream: torch.nn.Module,
    batch: ProteinMpnnInput,
    num_orders: int,
) -> None:
    ranks = _decoding_ranks(batch, num_orders)
    ours = model.score(
        batch,
        num_decoding_orders=num_orders,
        decoding_ranks=ranks,
    )
    log_probabilities = _upstream_forward(upstream, batch, ranks)
    sequence = _repeat(batch.sequence_tokens, num_orders)
    losses = -torch.gather(
        log_probabilities,
        -1,
        sequence.unsqueeze(-1),
    ).squeeze(-1)
    valid = _repeat(batch.valid_mask, num_orders).to(losses.dtype)
    designed = _repeat(
        batch.valid_mask & batch.design_mask,
        num_orders,
    ).to(losses.dtype)
    designed_evaluations = (losses * designed).sum(-1) / designed.sum(-1)
    global_evaluations = (losses * valid).sum(-1) / valid.sum(-1)

    ours_designed = ours.scores["designed_sequence_nll"]
    ours_global = ours.scores["global_sequence_nll"]
    _compare(
        f"{label}: designed score evaluations",
        torch.from_numpy(ours_designed.evaluations),
        designed_evaluations,
    )
    _compare(
        f"{label}: global score evaluations",
        torch.from_numpy(ours_global.evaluations),
        global_evaluations,
    )
    _compare(
        f"{label}: designed score mean",
        torch.tensor(ours_designed.value, dtype=designed_evaluations.dtype),
        designed_evaluations.mean(),
    )
    _compare(
        f"{label}: global score mean",
        torch.tensor(ours_global.value, dtype=global_evaluations.dtype),
        global_evaluations.mean(),
    )


def _probability_method_checks(
    label: str,
    model: ProteinMpnnModel,
    upstream: torch.nn.Module,
    reference: ModuleType,
    batch: ProteinMpnnInput,
    num_orders: int,
) -> None:
    r"""Cross-check the four frozen MpnnProbabilityResult methods against upstream.

    Each frozen method maps to one upstream computation. Backbone noise is zero, so
    every comparison is deterministic and needs no random-stream matching.
    """

    dtype = batch.backbone_coordinates.dtype
    length = batch.sequence_tokens.shape[1]
    designed = (batch.valid_mask & batch.design_mask)[0]

    # 1. sequence-conditioned autoregressive  <->  upstream `forward`
    #    (teacher-forced logits under one supplied full decoding order).
    ranks = _decoding_ranks(batch, num_orders)
    autoregressive = model.autoregressive_probabilities(
        batch,
        num_decoding_orders=num_orders,
        decoding_ranks=ranks,
    )
    _check(
        f"{label}: autoregressive result type",
        type(autoregressive) is MpnnProbabilityResult,
    )
    _check(
        f"{label}: autoregressive logits [E, L, 21] and ranks [E, L]",
        tuple(autoregressive.logits.shape) == (num_orders, length, 21)
        and tuple(autoregressive.decoding_ranks.shape) == (num_orders, length),
    )
    _compare(
        f"{label}: autoregressive vs upstream forward",
        autoregressive.log_probabilities,
        _upstream_forward(upstream, batch, ranks),
    )
    _compare(
        f"{label}: autoregressive returns supplied ranks",
        autoregressive.decoding_ranks,
        ranks,
        exact=True,
    )
    _check(
        f"{label}: autoregressive target identity",
        autoregressive.target is batch.structures[0]
        and autoregressive.component_references == batch.component_references[0],
    )

    # 2. order-dependent sequence-free autoregressive  <->  reconstructed upstream
    #    (no single upstream method; see `_upstream_sequence_free_forward`).
    sequence_free = model.sequence_free_autoregressive_probabilities(
        batch,
        num_decoding_orders=num_orders,
        decoding_ranks=ranks,
    )
    _check(
        f"{label}: sequence-free logits [E, L, 21] and ranks [E, L]",
        tuple(sequence_free.logits.shape) == (num_orders, length, 21)
        and tuple(sequence_free.decoding_ranks.shape) == (num_orders, length),
    )
    _compare(
        f"{label}: sequence-free autoregressive vs reconstructed upstream",
        sequence_free.log_probabilities,
        _upstream_sequence_free_forward(upstream, reference, batch, ranks),
    )

    # 3 & 4. per-residue conditional (target moved last) and unconditional
    #        (target moved first) over one base order. Upstream `conditional_probs`
    #        only fills designed (chain_M == 1) rows, so parity restricts to them;
    #        the frozen methods additionally evaluate every other row.
    base_ranks = _decoding_ranks(batch, num_orders)
    priorities = 1 + base_ranks.to(dtype) / (length + 1)
    upstream_args = (
        _repeat(batch.backbone_coordinates, num_orders),
        _repeat(batch.sequence_tokens, num_orders),
        _repeat(batch.valid_mask.to(dtype), num_orders),
        _repeat(batch.design_mask.to(dtype), num_orders),
        _repeat(batch.residue_indices, num_orders),
        _repeat(batch.chain_indices + 1, num_orders),
        priorities,
    )

    conditional = model.conditional_probabilities(
        batch,
        num_decoding_orders=num_orders,
        base_decoding_ranks=base_ranks,
    )
    _check(
        f"{label}: conditional logits [E, L, 21] and effective ranks [E, L, L]",
        tuple(conditional.logits.shape) == (num_orders, length, 21)
        and tuple(conditional.decoding_ranks.shape) == (num_orders, length, length),
    )
    _compare(
        f"{label}: conditional target-last vs upstream conditional_probs (designed rows)",
        conditional.log_probabilities[:, designed],
        upstream.conditional_probs(*upstream_args, backbone_only=False)[:, designed],
    )
    conditional_diagonal = conditional.decoding_ranks.diagonal(dim1=1, dim2=2)
    _check(
        f"{label}: conditional moves each designed target last",
        bool(torch.all((conditional_diagonal == conditional.decoding_ranks.amax(dim=-1))[:, designed])),
    )

    unconditional = model.unconditional_probabilities(
        batch,
        num_decoding_orders=num_orders,
        base_decoding_ranks=base_ranks,
    )
    _check(
        f"{label}: unconditional logits [E, L, 21] and effective ranks [E, L, L]",
        tuple(unconditional.logits.shape) == (num_orders, length, 21)
        and tuple(unconditional.decoding_ranks.shape) == (num_orders, length, length),
    )
    _compare(
        f"{label}: unconditional target-first vs upstream conditional_probs (designed rows)",
        unconditional.log_probabilities[:, designed],
        upstream.conditional_probs(*upstream_args, backbone_only=True)[:, designed],
    )
    unconditional_diagonal = unconditional.decoding_ranks.diagonal(dim1=1, dim2=2)
    _check(
        f"{label}: unconditional moves each designed target first",
        bool(torch.all((unconditional_diagonal == -1)[:, designed])),
    )


def _sample_checks(
    label: str,
    model: ProteinMpnnModel,
    upstream: torch.nn.Module,
    batch: ProteinMpnnInput,
    *,
    num_samples: int,
    seed: int,
    controlled: bool,
    pssm_threshold: float | None,
) -> None:
    device = batch.backbone_coordinates.device
    dtype = batch.backbone_coordinates.dtype
    temperature = 0.2 if controlled else 0.1
    backbone_noise = 0.02 if controlled else 0.0
    pssm_mixture = 0.4 if controlled else 0.0

    global_bias = (
        torch.linspace(
            -0.1,
            0.1,
            len(AMINO_ACID_COMPONENT_IDS) + 1,
            device=device,
            dtype=dtype,
        )
        if controlled
        else None
    )
    global_omit = torch.zeros(
        len(AMINO_ACID_COMPONENT_IDS) + 1,
        dtype=torch.bool,
        device=device,
    )
    global_omit[AMINO_ACID_UNKNOWN_TOKEN] = True
    if controlled:
        global_omit[AMINO_ACID_TOKEN_BY_COMPONENT["TYR"]] = True

    ranks = _decoding_ranks(batch, num_samples)
    ours = model.sample(
        batch,
        num_samples=num_samples,
        temperature=temperature,
        backbone_noise=backbone_noise,
        decoding_ranks=ranks,
        global_logit_bias=global_bias,
        global_omit_token_mask=global_omit,
        pssm_mixture=pssm_mixture,
        pssm_log_odds_threshold=pssm_threshold,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    _check(f"{label}: sample returns MpnnSamplingOutput", type(ours) is MpnnSamplingOutput)

    valid = _repeat(batch.valid_mask, num_samples)
    design = _repeat(batch.design_mask, num_samples)
    effective_design = valid & design
    priorities = (ranks.to(dtype) + 1) / (effective_design.to(dtype) + 0.0001)
    omit_by_residue = (
        torch.zeros(
            num_samples,
            batch.sequence_tokens.shape[1],
            len(AMINO_ACID_COMPONENT_IDS) + 1,
            device=device,
            dtype=dtype,
        )
        if batch.allowed_token_mask is None
        else (~_repeat(batch.allowed_token_mask, num_samples)).to(dtype)
    )
    pssm_coefficients = (
        torch.zeros_like(effective_design, dtype=dtype)
        if batch.pssm_coefficients is None
        else _repeat(batch.pssm_coefficients, num_samples)
    )
    pssm_probabilities = (
        torch.zeros_like(omit_by_residue)
        if batch.pssm_probabilities is None
        else _repeat(batch.pssm_probabilities, num_samples)
    )
    pssm_support = (
        torch.ones_like(omit_by_residue)
        if pssm_threshold is None or batch.pssm_log_odds is None
        else (_repeat(batch.pssm_log_odds, num_samples) > pssm_threshold).to(dtype)
    )
    residue_bias = (
        torch.zeros_like(omit_by_residue)
        if batch.residue_logit_bias is None
        else _repeat(batch.residue_logit_bias, num_samples)
    )

    upstream.features.augment_eps = backbone_noise
    torch.manual_seed(seed)
    try:
        expected = upstream.sample(
            _repeat(batch.backbone_coordinates, num_samples),
            priorities,
            _repeat(batch.sequence_tokens, num_samples),
            design.to(dtype),
            _repeat(batch.chain_indices + 1, num_samples),
            _repeat(batch.residue_indices, num_samples),
            mask=valid.to(dtype),
            temperature=temperature,
            omit_AAs_np=global_omit.to(dtype).cpu().numpy(),
            bias_AAs_np=(torch.zeros_like(global_omit, dtype=dtype) if global_bias is None else global_bias)
            .cpu()
            .numpy(),
            chain_M_pos=torch.ones_like(design, dtype=dtype),
            omit_AA_mask=omit_by_residue,
            pssm_coef=pssm_coefficients,
            pssm_bias=pssm_probabilities,
            pssm_multi=pssm_mixture,
            pssm_log_odds_flag=pssm_threshold is not None,
            pssm_log_odds_mask=pssm_support,
            pssm_bias_flag=batch.pssm_probabilities is not None,
            bias_by_res=residue_bias,
        )
    finally:
        upstream.features.augment_eps = 0.0

    mode = "controlled" if controlled else "threshold-only" if pssm_threshold is not None else "neutral"
    _compare(f"{label}: {mode} sample tokens", ours.sequences, expected["S"], exact=True)
    _compare(
        f"{label}: {mode} sample probabilities",
        ours.sampling_probabilities,
        expected["probs"],
    )
    _compare(
        f"{label}: {mode} sample order",
        torch.argsort(ours.decoding_ranks, dim=-1),
        expected["decoding_order"],
        exact=True,
    )
    _compare(
        f"{label}: {mode} returned ranks",
        ours.decoding_ranks,
        ranks,
        exact=True,
    )
    raw_expected = _upstream_forward(
        upstream,
        batch,
        ranks,
        sequence_tokens=expected["S"],
        backbone_noise=backbone_noise,
        seed=seed,
    )
    _compare(
        f"{label}: {mode} raw in-pass log probabilities",
        ours.log_probabilities,
        raw_expected,
    )


def _tied_sample_checks(
    label: str,
    model: ProteinMpnnModel,
    processor: ProteinMpnnProcessor,
    pdb: Path,
    seed: int,
) -> None:
    r"""Exercise the unified sampler on a transitive tied group.

    §4.7 replaced the old `tied_sample` with one `sample` that resolves tied and
    untied inputs on the same causal path. Tied members must share a rank and a
    sampled identity in every drawn sequence, which a flattened permutation could
    not encode without the upstream member-order bug.
    """

    base = processor(pdb)
    first, second, third = tuple(
        component_reference
        for component_reference, valid in zip(
            base.component_references[0],
            base.valid_mask[0].tolist(),
        )
        if component_reference is not None and valid
    )[:3]
    tied = processor(
        pdb,
        positions=(first, second, third),
        conditions=(TiedResidueIdentities(groups=((first, second), (second, third))),),
    )
    groups = tied_groups_from_ids(tied.tied_group_ids)[0]
    _check(
        f"{label}: transitive closure forms one tied group of three",
        len(groups) == 1 and len(groups[0]) == 3,
    )
    rows = list(groups[0])
    sampled = model.sample(
        tied,
        num_samples=2,
        temperature=0.2,
        generator=torch.Generator(device=base.backbone_coordinates.device).manual_seed(seed),
    )
    _check(
        f"{label}: tied members share exact ranks",
        bool(torch.all(sampled.decoding_ranks[:, rows] == sampled.decoding_ranks[:, rows[:1]])),
    )
    _check(
        f"{label}: tied members share sampled identity",
        bool(torch.all(sampled.sequences[:, rows] == sampled.sequences[:, rows[:1]])),
    )
    _compare(
        f"{label}: tied members share the sampling distribution",
        sampled.sampling_probabilities[:, rows],
        sampled.sampling_probabilities[:, rows[:1]].expand(-1, len(rows), -1),
    )


def _generated_class_checks(
    label: str,
    model: ProteinMpnnModel,
    upstream: torch.nn.Module,
    batch: ProteinMpnnInput,
    *,
    num_samples: int,
    seed: int,
) -> None:
    r"""Demonstrate the frozen §4.7 twenty-first-class (X / UNK) contract.

    The raw Model can place probability on the twenty-first generatable class
    (index 20 = ``AMINO_ACID_UNKNOWN_TOKEN`` = X / UNK) when
    ``global_omit_token_mask`` is ``None``, matching upstream ProteinMPNN with an
    empty ``omit_AAs``. Omitting X — the Pipeline's default ``("X",)`` lowering —
    drives that mass to zero and matches upstream with ``omit_AAs`` marking X. The
    two layers are therefore numerically distinct rather than the old core that
    killed the 21st class unconditionally in ``_hard_support``.
    """

    device = batch.backbone_coordinates.device
    dtype = batch.backbone_coordinates.dtype
    temperature = 0.1
    vocabulary = len(AMINO_ACID_COMPONENT_IDS) + 1
    length = batch.sequence_tokens.shape[1]
    unknown = AMINO_ACID_UNKNOWN_TOKEN

    ranks = _decoding_ranks(batch, num_samples)
    valid = _repeat(batch.valid_mask, num_samples)
    design = _repeat(batch.design_mask, num_samples)
    effective_design = valid & design
    priorities = (ranks.to(dtype) + 1) / (effective_design.to(dtype) + 0.0001)

    omit_x = torch.zeros(vocabulary, dtype=torch.bool, device=device)
    omit_x[unknown] = True
    allow_all = torch.zeros(vocabulary, dtype=torch.bool, device=device)

    def run_model(mask: Tensor | None) -> MpnnSamplingOutput:
        return model.sample(
            batch,
            num_samples=num_samples,
            temperature=temperature,
            decoding_ranks=ranks,
            global_omit_token_mask=mask,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

    def run_upstream(omit: Tensor) -> dict:
        torch.manual_seed(seed)
        return upstream.sample(
            _repeat(batch.backbone_coordinates, num_samples),
            priorities,
            _repeat(batch.sequence_tokens, num_samples),
            design.to(dtype),
            _repeat(batch.chain_indices + 1, num_samples),
            _repeat(batch.residue_indices, num_samples),
            mask=valid.to(dtype),
            temperature=temperature,
            omit_AAs_np=omit.to(dtype).cpu().numpy(),
            bias_AAs_np=torch.zeros(vocabulary, device=device, dtype=dtype).cpu().numpy(),
            chain_M_pos=torch.ones_like(design, dtype=dtype),
            omit_AA_mask=torch.zeros(num_samples, length, vocabulary, device=device, dtype=dtype),
            pssm_coef=torch.zeros(num_samples, length, device=device, dtype=dtype),
            pssm_bias=torch.zeros(num_samples, length, vocabulary, device=device, dtype=dtype),
            pssm_multi=0.0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=torch.ones(num_samples, length, vocabulary, device=device, dtype=dtype),
            pssm_bias_flag=False,
            bias_by_res=torch.zeros(num_samples, length, vocabulary, device=device, dtype=dtype),
        )

    enabled = run_model(None)
    omitted = run_model(omit_x)
    upstream_enabled = run_upstream(allow_all)
    upstream_omitted = run_upstream(omit_x)

    enabled_x = enabled.sampling_probabilities[..., unknown]
    omitted_x = omitted.sampling_probabilities[..., unknown]
    _check(
        f"{label}: raw Model places probability on the 21st class (X/UNK) with mask=None",
        bool((enabled_x[effective_design] > 0).all()),
    )
    _check(
        f"{label}: Pipeline-default X omission removes all 21st-class probability",
        bool((omitted_x[effective_design] == 0).all()),
    )
    _check(
        f"{label}: enabling versus omitting X yields distinct distributions",
        not torch.equal(enabled.sampling_probabilities, omitted.sampling_probabilities),
    )
    _compare(
        f"{label}: raw Model mask=None equals upstream empty omit_AAs (probabilities)",
        enabled.sampling_probabilities,
        upstream_enabled["probs"],
    )
    _compare(
        f"{label}: raw Model mask=None equals upstream empty omit_AAs (tokens)",
        enabled.sequences,
        upstream_enabled["S"],
        exact=True,
    )
    _compare(
        f"{label}: raw Model X-omission equals upstream omit_AAs=['X'] (probabilities)",
        omitted.sampling_probabilities,
        upstream_omitted["probs"],
    )


def _tied_weight_checks(
    label: str,
    model: ProteinMpnnModel,
    processor: ProteinMpnnProcessor,
    upstream: torch.nn.Module,
    pdb: Path,
    seed: int,
) -> None:
    r"""Prove the frozen §4.7 tied-weight coefficient semantics.

    ``ProteinMpnnInput.tied_weights`` stores the final signed neural-logit
    coefficients: a tied group's combined logits are ``sum(coeff[i] * logits[i])``.
    The processor's default coefficient ``1/len(G)`` reproduces the equal mean (which
    is exactly upstream tied decoding's ``beta/len(G)`` averaging with unit betas),
    while an explicit all-ones assignment reproduces the plain sum of member logits.

    The combination is proven self-contained from the sampler's own per-member
    ``log_probabilities`` (``softmax`` is invariant to the per-member normalizer, so a
    uniform mean/sum of member log-softmax rows recovers the mean/sum of the raw
    logits). The leading tied member has no earlier same-group member and therefore
    shares its all-native context with an upstream teacher-forced forward pass, giving
    a direct upstream anchor; OFoldX deliberately hides same-group members from each
    other, avoiding upstream ``tied_sample``'s zero-embedding member-order artifact.
    """

    base = processor(pdb)
    device = base.backbone_coordinates.device
    temperature = 0.2

    first, second, third = tuple(
        component_reference
        for component_reference, valid in zip(
            base.component_references[0],
            base.valid_mask[0].tolist(),
        )
        if component_reference is not None and valid
    )[:3]
    members = (first, second, third)
    conditions = (TiedResidueIdentities(groups=((first, second), (second, third))),)

    default_input = processor(pdb, positions=members, conditions=conditions)
    summed_input = processor(
        pdb,
        positions=members,
        conditions=conditions,
        tied_weights={member: 1.0 for member in members},
    )
    groups = tied_groups_from_ids(default_input.tied_group_ids)[0]
    _check(
        f"{label}: tied-weight group forms one unit of three",
        len(groups) == 1 and len(groups[0]) == 3,
    )
    rows = list(groups[0])
    representative = rows[0]

    default = model.sample(
        default_input,
        num_samples=1,
        temperature=temperature,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    summed = model.sample(
        summed_input,
        num_samples=1,
        temperature=temperature,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    mean_distribution = default.sampling_probabilities[:, representative]
    sum_distribution = summed.sampling_probabilities[:, representative]
    member_log_probs = default.log_probabilities[:, rows]

    _compare(
        f"{label}: default (None) tied weights combine member logits by mean",
        mean_distribution,
        F.softmax(member_log_probs.mean(dim=1) / temperature, dim=-1),
    )
    _compare(
        f"{label}: explicit all-ones tied weights combine member logits by sum",
        sum_distribution,
        F.softmax(member_log_probs.sum(dim=1) / temperature, dim=-1),
    )
    _check(
        f"{label}: mean (default) and sum (all-ones) tied combinations are distinct",
        not torch.allclose(mean_distribution, sum_distribution, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL),
    )

    upstream_forward = _upstream_forward(
        upstream,
        default_input,
        default.decoding_ranks,
        sequence_tokens=default_input.sequence_tokens,
    )
    _compare(
        f"{label}: leading tied member averages upstream member log-probs",
        default.log_probabilities[:, representative],
        upstream_forward[:, representative],
    )


def _representative_checks(
    label: str,
    model: ProteinMpnnModel,
    processor: ProteinMpnnProcessor,
    upstream: torch.nn.Module,
    reference: ModuleType,
    pdb: Path,
    batch: ProteinMpnnInput,
    num_samples: int,
    seed: int,
) -> None:
    alternate_sequence = _processor_checks(
        label,
        processor,
        reference,
        pdb,
        batch,
    )

    _encode_checks(label, model, batch)

    ranks = _decoding_ranks(batch, 1)
    noise = 0.05
    output = model(
        batch,
        backbone_noise=noise,
        decoding_ranks=ranks,
        generator=torch.Generator(device=batch.backbone_coordinates.device).manual_seed(seed),
    )
    expected = _upstream_forward(
        upstream,
        batch,
        ranks,
        backbone_noise=noise,
        seed=seed,
    )
    _check(f"{label}: nonzero-noise forward MpnnOutput", type(output) is MpnnOutput)
    _compare(f"{label}: nonzero-noise forward log-probabilities", output.log_probabilities, expected)
    _compare(f"{label}: nonzero-noise forward returns ranks", output.decoding_ranks, ranks, exact=True)

    _score_checks(label, model, upstream, batch, num_samples)
    alternate = processor(batch.structures[0], sequence=alternate_sequence)
    _score_checks(
        f"{label}: alternate sequence",
        model,
        upstream,
        alternate,
        num_samples,
    )

    _probability_method_checks(label, model, upstream, reference, batch, num_samples)

    _sample_checks(
        label,
        model,
        upstream,
        batch,
        num_samples=num_samples,
        seed=seed + 2,
        controlled=False,
        pssm_threshold=None,
    )
    _sample_checks(
        label,
        model,
        upstream,
        batch,
        num_samples=num_samples,
        seed=seed + 3,
        controlled=False,
        pssm_threshold=0.0,
    )

    component_reference = next(
        component_reference
        for component_reference, valid in zip(
            batch.component_references[0],
            batch.valid_mask[0].tolist(),
        )
        if component_reference is not None and valid
    )
    probabilities = torch.zeros(
        len(AMINO_ACID_COMPONENT_IDS) + 1,
        device=batch.sequence_tokens.device,
        dtype=batch.backbone_coordinates.dtype,
    )
    probabilities[AMINO_ACID_TOKEN_BY_COMPONENT["ALA"]] = 0.6
    probabilities[AMINO_ACID_TOKEN_BY_COMPONENT["GLY"]] = 0.4
    log_odds = torch.full_like(probabilities, -1.0)
    log_odds[AMINO_ACID_TOKEN_BY_COMPONENT["ALA"]] = 1.0
    log_odds[AMINO_ACID_TOKEN_BY_COMPONENT["GLY"]] = 1.0
    # Allowed support is exactly the nonzero PSSM ALA/GLY support; omitted TYR
    # has zero PSSM mass, so neither recorded upstream leakage defect is crossed.
    controlled = processor(
        pdb,
        conditions=(
            AllowedResidueIdentities(
                (component_reference,),
                frozenset({"ALA", "GLY"}),
            ),
        ),
        residue_logit_bias={
            component_reference: torch.linspace(
                -0.15,
                0.15,
                len(AMINO_ACID_COMPONENT_IDS) + 1,
                device=batch.sequence_tokens.device,
                dtype=batch.backbone_coordinates.dtype,
            )
        },
        pssm_coefficients={component_reference: 0.6},
        pssm_probabilities={component_reference: probabilities},
        pssm_log_odds={component_reference: log_odds},
    )
    _sample_checks(
        label,
        model,
        upstream,
        controlled,
        num_samples=num_samples,
        seed=seed + 4,
        controlled=True,
        pssm_threshold=0.0,
    )
    _tied_sample_checks(label, model, processor, pdb, seed + 10)
    _generated_class_checks(
        label,
        model,
        upstream,
        batch,
        num_samples=num_samples,
        seed=seed + 20,
    )
    _tied_weight_checks(label, model, processor, upstream, pdb, seed + 30)


def _load_packer(
    checkpoint_path: Path,
    device: torch.device,
) -> MpnnPackerPipeline:
    r"""Build a native MpnnPacker Pipeline from a released side-chain checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint["model_state_dict"]
    model = MpnnPackerModel(
        MpnnPackerConfig(
            k_neighbors=int(checkpoint["num_edges"]),
            atom_context_num=int(checkpoint["atom_context_num"]),
        )
    )
    model.load_state_dict(convert_mpnn_packer_state_dict(state_dict), strict=True, assign=True)
    model = model.to(device).eval()
    processor = MpnnPackerProcessor(device=device)
    return MpnnPackerPipeline(model=model, processor=processor)


def _design_and_pack_checks(
    label: str,
    model: ProteinMpnnModel,
    processor: ProteinMpnnProcessor,
    packer_checkpoint: Path,
    pdb: Path,
    device: torch.device,
    seed: int,
) -> None:
    r"""Smoke and design-parity for the composed ProteinMpnnDesignAndPackPipeline."""

    if not packer_checkpoint.exists():
        print(f"[skip] {label}: packer checkpoint {packer_checkpoint} is absent")
        return

    designer = ProteinMpnnPipeline(model=model, processor=processor)
    packer = _load_packer(packer_checkpoint, device)
    composed = ProteinMpnnDesignAndPackPipeline(designer=designer, packer=packer)
    _check(
        f"{label}: composed pipeline preserves its children",
        composed.designer is designer and composed.packer is packer,
    )

    structure = read_pdb(pdb)
    num_candidates = 2
    num_packs = 3
    design_result, packs = composed.design(
        structure,
        num_candidates=num_candidates,
        num_decoding_orders=1,
        num_packs_per_candidate=num_packs,
        num_denoising_steps=2,
        num_torsion_samples=4,
        seed=seed,
    )
    _check(
        f"{label}: design-and-pack returns a design result and per-candidate packs",
        type(design_result) is ProteinMpnnDesignResult
        and isinstance(packs, tuple)
        and len(design_result.candidates) == num_candidates
        and len(packs) == num_candidates,
    )
    _check(
        f"{label}: each candidate owns num_packs_per_candidate predictions",
        all(type(pack) is PackResult and len(pack.predictions) == num_packs for pack in packs)
        and all(type(prediction) is PackPrediction for pack in packs for prediction in pack.predictions),
    )

    # The composed design portion consumes the supplied stream first, so it must
    # reproduce a standalone Model.design over the identically lowered Structure
    # input with a fresh generator on the same seed.
    design_input = designer.processor.process_structure(structure)
    standalone = designer.model.design(
        design_input,
        num_candidates=num_candidates,
        num_decoding_orders=1,
        global_omit_token_mask=_global_omit_token_mask(("X",), device=device),
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    _compare_proteinmpnn_design_result(
        f"{label}: composed design vs standalone Model.design",
        design_result,
        standalone,
    )

    # `_candidate_protein_sequences` maps only the protein instances to realized
    # component-ID sequences; the released reader also retains non-protein
    # molecules, so the keys are the protein subset of the candidate system.
    candidate_system = design_result.candidates[0].system
    expected_protein_instances = {
        instance_id
        for instance_id, entity_id in candidate_system.instances.items()
        if isinstance(candidate_system.entities[entity_id], PolymerSequence)
        and candidate_system.entities[entity_id].polymer_type == "protein"
    }
    protein_sequences = _candidate_protein_sequences(candidate_system)
    _check(
        f"{label}: candidate protein sequence mapping",
        isinstance(protein_sequences, dict)
        and bool(protein_sequences)
        and set(protein_sequences) == expected_protein_instances
        and all(
            isinstance(instance_id, str)
            and isinstance(sequence, tuple)
            and len(sequence) == len(candidate_system.entities[candidate_system.instances[instance_id]].components)
            for instance_id, sequence in protein_sequences.items()
        ),
    )


def _public_vertical_checks(
    label: str,
    model: ProteinMpnnModel,
    processor: ProteinMpnnProcessor,
    pdb: Path,
    batch: ProteinMpnnInput,
    checkpoint_path: Path,
    packer_checkpoint: Path,
    device: torch.device,
    seed: int,
) -> None:
    direct_pipeline = ProteinMpnnPipeline(model=model, processor=processor)
    _check(
        f"{label}: direct Pipeline preserves its pair",
        direct_pipeline.model is model and direct_pipeline.processor is processor,
    )
    expected_vocabulary = tuple(
        "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    )
    _check(
        f"{label}: exact vocabulary and output width",
        AMINO_ACID_COMPONENT_IDS == expected_vocabulary
        and AMINO_ACID_TOKEN_BY_COMPONENT
        == {component_id: index for index, component_id in enumerate(expected_vocabulary)}
        and AMINO_ACID_UNKNOWN_TOKEN == 20
        and model.output_projection.out_features == 21,
    )

    # `global_omit_tokens` lowering: the default `("X",)` omits the generated X /
    # UNK class (index 20), while an empty selection lowers to `None`. Per frozen
    # §4.7 the raw Model retains the twenty-first class as reachable support, so an
    # empty selection genuinely re-enables X rather than aliasing the default; the
    # `_generated_class_checks` demonstration proves the two layers are distinct.
    _check(
        f"{label}: global_omit_tokens lowering",
        _global_omit_token_mask((), device=device) is None
        and torch.equal(
            torch.nonzero(_global_omit_token_mask(("X",), device=device)).flatten(),
            torch.tensor([AMINO_ACID_UNKNOWN_TOKEN], device=device),
        ),
    )

    with torch.inference_mode():
        design_result = model.design(
            batch,
            num_candidates=2,
            num_decoding_orders=2,
            global_omit_token_mask=_global_omit_token_mask(("X",), device=device),
            generator=torch.Generator(device=batch.backbone_coordinates.device).manual_seed(seed),
        )
        pipeline_design_result = direct_pipeline.design(
            pdb,
            num_candidates=2,
            num_decoding_orders=2,
            seed=seed,
        )
        _compare_proteinmpnn_design_result(
            f"{label}: Model/Pipeline design",
            pipeline_design_result,
            design_result,
        )

        empty_pipeline_design = direct_pipeline.design(
            pdb,
            num_candidates=2,
            num_decoding_orders=2,
            global_omit_tokens=(),
            seed=seed,
        )
        empty_model_design = model.design(
            batch,
            num_candidates=2,
            num_decoding_orders=2,
            global_omit_token_mask=None,
            generator=torch.Generator(device=batch.backbone_coordinates.device).manual_seed(seed),
        )
        _compare_proteinmpnn_design_result(
            f"{label}: empty global_omit_tokens lowers to None",
            empty_pipeline_design,
            empty_model_design,
        )

        input_score_names = {"designed_sequence_nll", "global_sequence_nll"}
        candidate_score_metadata = {
            "designed_sequence_nll": ("nat/residue", "lower"),
            "global_sequence_nll": ("nat/residue", "lower"),
            "sequence_recovery": (None, "higher"),
        }
        valid = batch.valid_mask[0].detach().cpu().tolist()
        designed = (batch.valid_mask & batch.design_mask)[0].detach().cpu().tolist()
        references = batch.component_references[0]
        designed_scope = frozenset(
            reference for reference, selected in zip(references, designed) if reference is not None and selected
        )
        global_scope = frozenset(
            reference for reference, present in zip(references, valid) if reference is not None and present
        )
        _check(f"{label}: design candidate count", len(design_result.candidates) == 2)
        _check(
            f"{label}: input score contract",
            design_result.input_score.target is batch.structures[0]
            and set(design_result.input_score.scores) == input_score_names
            and all(
                score.evaluations is not None
                and score.evaluations.shape == (2,)
                and score.value == float(score.evaluations.mean())
                for score in design_result.input_score.scores.values()
            )
            and (
                design_result.input_score.scores["designed_sequence_nll"].unit,
                design_result.input_score.scores["designed_sequence_nll"].preferred,
                design_result.input_score.scores["designed_sequence_nll"].scope,
            )
            == ("nat/residue", "lower", designed_scope)
            and (
                design_result.input_score.scores["global_sequence_nll"].unit,
                design_result.input_score.scores["global_sequence_nll"].preferred,
                design_result.input_score.scores["global_sequence_nll"].scope,
            )
            == ("nat/residue", "lower", global_scope),
        )
        source_system = batch.structures[0].system
        for candidate_index, candidate in enumerate(design_result.candidates):
            _check(
                f"{label}: candidate {candidate_index} direct contract",
                set(candidate.scores) == set(candidate_score_metadata)
                and all(score.evaluations is None for score in candidate.scores.values())
                and all(
                    (candidate.scores[name].unit, candidate.scores[name].preferred) == metadata
                    for name, metadata in candidate_score_metadata.items()
                )
                and candidate.scores["designed_sequence_nll"].scope == designed_scope
                and candidate.scores["sequence_recovery"].scope == designed_scope
                and candidate.scores["global_sequence_nll"].scope == global_scope
                and all(
                    candidate.system.component(reference).component_id in expected_vocabulary
                    for reference in designed_scope
                )
                and tuple(candidate.system.instances) == tuple(source_system.instances)
                and all(
                    len(candidate.system.entities[candidate.system.instances[instance_id]].components)
                    == len(source_system.entities[source_system.instances[instance_id]].components)
                    for instance_id in source_system.instances
                ),
            )

        direct_score = model.score(
            batch,
            num_decoding_orders=2,
            generator=torch.Generator(device=batch.backbone_coordinates.device).manual_seed(seed + 1),
        )
        pipeline_score = direct_pipeline.score(pdb, num_decoding_orders=2, seed=seed + 1)
        _compare_score_result(
            f"{label}: Pipeline score delegation",
            pipeline_score,
            direct_score,
        )

        # The four probability computations delegate through the concrete Pipeline.
        # With zero noise a shared seed fixes the internally drawn base order, so the
        # Model and Pipeline logits and effective ranks match exactly.
        for method_name in (
            "autoregressive_probabilities",
            "sequence_free_autoregressive_probabilities",
            "conditional_probabilities",
            "unconditional_probabilities",
        ):
            direct_result = getattr(model, method_name)(
                batch,
                num_decoding_orders=2,
                generator=torch.Generator(device=batch.backbone_coordinates.device).manual_seed(seed + 6),
            )
            pipeline_result = getattr(direct_pipeline, method_name)(pdb, num_decoding_orders=2, seed=seed + 6)
            _check(
                f"{label}: Pipeline {method_name} result type",
                type(pipeline_result) is MpnnProbabilityResult,
            )
            _compare(
                f"{label}: Pipeline {method_name} logits delegation",
                pipeline_result.logits,
                direct_result.logits,
                exact=True,
            )
            _compare(
                f"{label}: Pipeline {method_name} rank delegation",
                pipeline_result.decoding_ranks,
                direct_result.decoding_ranks,
                exact=True,
            )

    with tempfile.TemporaryDirectory(prefix="ofoldx-proteinmpnn-") as directory:
        repository = Path(directory)
        convert_checkpoint(checkpoint_path, repository)
        expected_files = {
            "model.json",
            "processor.json",
            "model.safetensors",
            "pipeline.json",
        }
        _check(
            f"{label}: repository exact files",
            {path.name for path in repository.iterdir()} == expected_files,
        )

        model_values = json.loads((repository / "model.json").read_text(encoding="utf-8"))
        processor_values = json.loads((repository / "processor.json").read_text(encoding="utf-8"))
        marker_values = json.loads((repository / "pipeline.json").read_text(encoding="utf-8"))
        expected_model_values = dict(model.config)
        expected_model_values["architecture"] = "ofoldx.proteinmpnn"
        _check(
            f"{label}: repository model architecture and config",
            model_values == expected_model_values,
        )
        _check(
            f"{label}: repository processor config",
            processor_values == dict(processor.config),
        )
        _check(
            f"{label}: repository concrete Pipeline marker",
            marker_values == {"pipeline": "ofoldx.proteinmpnn"},
        )

        loaded_model = AutoModel(repository, device=device)
        loaded_processor = AutoProcessor(
            repository,
            device=device,
            dtype=processor.dtype,
        )
        _check(f"{label}: AutoModel concrete type", type(loaded_model) is ProteinMpnnModel)
        _check(
            f"{label}: AutoProcessor concrete type",
            type(loaded_processor) is ProteinMpnnProcessor,
        )
        _check(
            f"{label}: AutoModel concrete config",
            type(loaded_model.config) is ProteinMpnnConfig and dict(loaded_model.config) == dict(model.config),
        )
        _check(
            f"{label}: AutoProcessor concrete config",
            type(loaded_processor.config) is ProteinMpnnProcessorConfig
            and dict(loaded_processor.config) == dict(processor.config),
        )
        _check(f"{label}: AutoModel eval mode", not loaded_model.training)
        expected_runtime_tensors = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        loaded_runtime_tensors = {
            **dict(loaded_model.named_parameters()),
            **dict(loaded_model.named_buffers()),
        }
        _check(
            f"{label}: AutoModel placement",
            set(loaded_runtime_tensors) == set(expected_runtime_tensors)
            and all(
                loaded_runtime_tensors[name].device == tensor.device
                and loaded_runtime_tensors[name].dtype == tensor.dtype
                for name, tensor in expected_runtime_tensors.items()
            ),
        )
        _check(
            f"{label}: AutoProcessor placement",
            loaded_processor.device == processor.device and loaded_processor.dtype == processor.dtype,
        )
        torch.testing.assert_close(
            loaded_model.state_dict(),
            model.state_dict(),
            rtol=0,
            atol=0,
        )
        print(f"[PASS] {label}: AutoModel exact state mapping")

        design_pipeline = load_pipeline(
            "design",
            model=repository,
            device=device,
            dtype=processor.dtype,
        )
        score_pipeline = load_pipeline(
            "score",
            model=repository,
            device=device,
            dtype=processor.dtype,
        )
        for task, loaded_pipeline in (
            ("design", design_pipeline),
            ("score", score_pipeline),
        ):
            _check(
                f"{label}: pipeline({task!r}) concrete type",
                type(loaded_pipeline) is ProteinMpnnPipeline,
            )
            _check(
                f"{label}: pipeline({task!r}) exact pair",
                type(loaded_pipeline.model) is ProteinMpnnModel
                and type(loaded_pipeline.processor) is ProteinMpnnProcessor,
            )
            _check(
                f"{label}: pipeline({task!r}) configs",
                dict(loaded_pipeline.model.config) == dict(model.config)
                and dict(loaded_pipeline.processor.config) == dict(processor.config),
            )
            _check(
                f"{label}: pipeline({task!r}) eval mode",
                not loaded_pipeline.model.training,
            )
            pipeline_runtime_tensors = {
                **dict(loaded_pipeline.model.named_parameters()),
                **dict(loaded_pipeline.model.named_buffers()),
            }
            _check(
                f"{label}: pipeline({task!r}) placement",
                loaded_pipeline.processor.device == processor.device
                and loaded_pipeline.processor.dtype == processor.dtype
                and set(pipeline_runtime_tensors) == set(expected_runtime_tensors)
                and all(
                    tensor.device == expected_runtime_tensors[name].device
                    and tensor.dtype == expected_runtime_tensors[name].dtype
                    for name, tensor in pipeline_runtime_tensors.items()
                ),
            )

        hydrated_design_result = design_pipeline.design(
            pdb,
            num_candidates=2,
            num_decoding_orders=2,
            seed=seed,
        )
        _compare_proteinmpnn_design_result(
            f"{label}: hydrated Pipeline design",
            hydrated_design_result,
            design_result,
        )

    _design_and_pack_checks(
        label,
        model,
        processor,
        packer_checkpoint,
        pdb,
        device,
        seed + 70,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--packer-checkpoint",
        type=Path,
        default=None,
        help="Side-chain packer checkpoint for the ProteinMpnnDesignAndPackPipeline check.",
    )
    args = parser.parse_args()

    root = reference_root("ProteinMPNN")
    weights = weight_path("mpnn-weights")
    packer_checkpoint = (
        args.packer_checkpoint.expanduser()
        if args.packer_checkpoint is not None
        else weights / "ligandmpnn_sc_v_32_002_16.pt"
    )
    reference, revision, dirty = _load_reference(root)
    pdb = root / "inputs" / "PDB_monomers" / "pdbs" / "5L33.pdb"
    checkpoints = (
        ("vanilla-002", weights / "proteinmpnn_v_48_002.pt"),
        ("vanilla-010", weights / "proteinmpnn_v_48_010.pt"),
        ("vanilla-020", weights / "proteinmpnn_v_48_020.pt"),
        ("vanilla-030", weights / "proteinmpnn_v_48_030.pt"),
        ("soluble-002", weights / "solublempnn_v_48_002.pt"),
        ("soluble-010", weights / "solublempnn_v_48_010.pt"),
        ("soluble-020", weights / "solublempnn_v_48_020.pt"),
        ("soluble-030", weights / "solublempnn_v_48_030.pt"),
        ("ca-002", root / "ca_model_weights" / "v_48_002.pt"),
        ("ca-010", root / "ca_model_weights" / "v_48_010.pt"),
        ("ca-020", root / "ca_model_weights" / "v_48_020.pt"),
        (
            "training-example",
            root / "training" / "exp_020" / "model_weights" / "epoch_last.pt",
        ),
    )

    print(f"[source] dauparas/ProteinMPNN checkout={root} revision={revision} dirty={dirty}")
    print(f"[fixture] {pdb}")
    print(f"[packer] {packer_checkpoint}")
    device = torch.device(args.device)
    for checkpoint_index, (label, checkpoint_path) in enumerate(checkpoints):
        model, processor, upstream = _load_models(
            label,
            checkpoint_path,
            reference,
            device,
        )
        with torch.inference_mode():
            batch = processor(pdb)
            ranks = _decoding_ranks(batch, 1)
            output = model(batch, decoding_ranks=ranks)
            expected = _upstream_forward(upstream, batch, ranks)
            _check(f"{label}: explicit-rank forward MpnnOutput", type(output) is MpnnOutput)
            _compare(f"{label}: explicit-rank forward", output.log_probabilities, expected)
            if label in {"vanilla-020", "ca-020"}:
                _representative_checks(
                    label,
                    model,
                    processor,
                    upstream,
                    reference,
                    pdb,
                    batch,
                    args.num_samples,
                    args.seed + 100 * checkpoint_index,
                )
        if label == "vanilla-020":
            _public_vertical_checks(
                label,
                model,
                processor,
                pdb,
                batch,
                checkpoint_path,
                packer_checkpoint,
                device,
                args.seed + 100 * checkpoint_index + 50,
            )

    print("[summary] PASS complete 12-checkpoint ProteinMpnn diagnostic")


if __name__ == "__main__":
    main()
