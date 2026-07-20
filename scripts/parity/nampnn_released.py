#!/usr/bin/env python
r"""Compare both released NA-MPNN checkpoints with the native full vertical.

The native vertical exposes the frozen ``ofoldx.models.nampnn`` surface: a typed
``NaMpnnInput``, a ``NaMpnnModel`` whose operations take an explicit
``decoding_ranks`` tensor (never a persisted decode-order noise), the shared
``MpnnOutput``/``MpnnProbabilityResult``/``MpnnSamplingOutput`` carriers, and the
registered convert -> save -> ``AutoModel``/``AutoProcessor`` round trip. Each
released operation is driven by reconstructing the exact upstream decoding order
as ranks so both sides evaluate one identical causal path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ofoldx import AutoModel, AutoProcessor  # noqa: E402
from ofoldx.data.components import get_component  # noqa: E402
from ofoldx.data.conditions import AllowedResidueIdentities, TiedResidueIdentities  # noqa: E402
from ofoldx.data.molecular import MolecularSystem, PolymerSequence  # noqa: E402
from ofoldx.data.structure import Structure  # noqa: E402
from ofoldx.models.nampnn import (  # noqa: E402
    NaMpnnConfig,
    NaMpnnInput,
    NaMpnnModel,
    NaMpnnProcessor,
    NaMpnnProcessorConfig,
    nll_loss,
    smoothed_loss,
)
from ofoldx.models.nampnn.convert import convert_nampnn_checkpoint, convert_nampnn_state_dict  # noqa: E402

_VOCAB_SIZE = 33
# The released graph runs in float32; the incremental sampler and the low-temperature
# softmax place two independent float32 implementations a few ULP apart. This is the
# shared MPNN-family parity tolerance (see scripts/parity/proteinmpnn.py).
_FLOAT32_ATOL = 1e-5
_ATOM_NAMES = (
    "N",
    "CA",
    "C",
    "O",
    "OP1",
    "OP2",
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
)
_RESTYPES = (
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
    "DA",
    "DC",
    "DG",
    "DT",
    "DX",
    "A",
    "C",
    "G",
    "U",
    "RX",
    "MAS",
    "PAD",
)


def _numerical_check(error: float, atol: float) -> dict[str, float | bool]:
    return {
        "max_absolute_error": error,
        "absolute_tolerance": atol,
        "passed": error <= atol,
    }


def _exact_check(exact: bool) -> dict[str, bool]:
    return {"exact": exact, "passed": exact}


def _checks_passed(checks: dict[str, dict[str, float | bool]]) -> bool:
    return all(bool(check["passed"]) for check in checks.values())


# ----------------------------------------------------------------------
# Decoding-order reconstruction. The frozen surface never stores decode-order
# noise on the Input; operations accept an explicit ``decoding_ranks`` tensor.
# These helpers reconstruct the exact upstream permutation as ranks so the native
# operation walks one causal path identical to the released call.
# ----------------------------------------------------------------------
def _ranks_from_order(order: torch.Tensor) -> torch.Tensor:
    r"""Invert a decode-order permutation into lower-is-earlier causal ranks."""

    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(order.shape[1], device=order.device).expand_as(order),
    )
    return ranks


def _ranks_from_noise(effective_chain_mask: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    r"""Reproduce the released ``argsort((chain_mask+eps)*|randn|)`` order as ranks."""

    order = torch.argsort((effective_chain_mask + 0.0001) * noise.abs(), dim=1)
    return _ranks_from_order(order)


def _shared_units(batch: NaMpnnInput) -> list[list[int]]:
    r"""Group aligned rows into causal units, fixed units before designed units."""

    ids = batch.token_group_ids[0].tolist()
    groups: dict[int, list[int]] = {}
    for index, group_id in enumerate(ids):
        groups.setdefault(int(group_id), []).append(index)
    design = (batch.valid_mask & batch.design_mask)[0].tolist()
    units = list(groups.values())
    # Fixed units first, then singletons, then tied units. Decoding a tied unit last
    # keeps native's intended permutation-invariant divergence (it does not copy the
    # released member-order coupling) off every untied row's context.
    units.sort(key=lambda members: (all(design[member] for member in members), len(members) > 1, min(members)))
    return units


def _shared_sample_ranks(batch: NaMpnnInput) -> torch.Tensor:
    r"""One shared causal ranking where tied members share their unit's rank."""

    length = batch.sequence_tokens.shape[1]
    ranks = torch.zeros(1, length, dtype=torch.long, device=batch.sequence_tokens.device)
    for rank, members in enumerate(_shared_units(batch)):
        for member in members:
            ranks[0, member] = rank
    return ranks


def _upstream_randn(ranks: torch.Tensor, effective_chain_mask: torch.Tensor) -> torch.Tensor:
    r"""Released ``randn`` whose ``argsort((chain_mask+eps)*|randn|)`` equals ``ranks``."""

    return (ranks.to(torch.float64) + 1.0) / (effective_chain_mask.to(torch.float64) + 0.0001)


def _effective_chain_mask(batch: NaMpnnInput) -> torch.Tensor:
    return (batch.valid_mask & batch.design_mask).to(torch.float64)


def _upstream_bias(batch: NaMpnnInput) -> torch.Tensor:
    r"""Released per-position log bias: the aligned residue bias, -1e8 where disallowed."""

    length = batch.sequence_tokens.shape[1]
    base = batch.residue_logit_bias
    if base is None:
        base = torch.zeros(1, length, _VOCAB_SIZE, dtype=torch.float32, device=batch.sequence_tokens.device)
    return torch.where(batch.allowed_token_mask, base, torch.full_like(base, -1e8))


def _fixture() -> Structure:
    definitions = (
        ("protein", "protein", ("ALA", "GLY")),
        ("dna", "dna", ("DA", "DT")),
        ("rna", "rna", ("A", "U")),
    )
    entities = {
        entity_id: PolymerSequence(
            polymer_type,
            tuple(get_component(component_id) for component_id in component_ids),
        )
        for entity_id, polymer_type, component_ids in definitions
    }
    instances = {entity_id.upper(): entity_id for entity_id, _, _ in definitions}
    system = MolecularSystem(entities, instances)
    sites = tuple(site for site, atom in system.iter_atoms() if atom.element != "H" and site.atom_id in _ATOM_NAMES)
    coordinates = np.arange(len(sites) * 3, dtype=np.float32).reshape(-1, 3) / 10
    return Structure(system, sites, coordinates, np.ones(len(sites), dtype=np.bool_))


def _official_model(
    module: object,
    *,
    shared_nucleic_tokens: bool = True,
    k_neighbors: int = 32,
) -> torch.nn.Module:
    restype_to_int = dict(zip(_RESTYPES, range(33)))
    if shared_nucleic_tokens:
        restype_to_int.update({"A": 21, "C": 22, "G": 23, "U": 24, "RX": 25})
    return module.ProteinMPNN(  # type: ignore[attr-defined]
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        model_type="na_mpnn",
        vocab=33,
        num_letters=33,
        atom_dict=dict(zip(_ATOM_NAMES, range(16))),
        restype_to_int=restype_to_int,
        polytype_to_int=dict(zip(("PP", "DNA", "RNA", "UNK", "MAS", "PAD"), range(6))),
    )


def _official_training_model(
    module: object,
    *,
    decode_protein_first: bool,
    dropout: float = 0.1,
) -> torch.nn.Module:
    restype_to_int = dict(zip(_RESTYPES, range(33)))
    restype_to_int.update({"A": 21, "C": 22, "G": 23, "U": 24, "RX": 25})
    return module.ProteinMPNN(  # type: ignore[attr-defined]
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=32,
        dropout=dropout,
        vocab=33,
        num_letters=33,
        atom_dict=dict(zip(_ATOM_NAMES, range(16))),
        restype_to_int=restype_to_int,
        polytype_to_int=dict(zip(("PP", "DNA", "RNA", "UNK", "MAS", "PAD"), range(6))),
        decode_protein_first=int(decode_protein_first),
    )


def _features(batch: NaMpnnInput) -> dict[str, torch.Tensor]:
    return {
        "X": batch.atom_coordinates,
        "X_m": batch.atom_mask,
        "S": batch.sequence_tokens,
        "mask": batch.valid_mask.float(),
        "R_idx": batch.residue_indices,
        "chain_labels": batch.chain_indices,
        "chain_mask": batch.design_mask.float(),
        "protein_mask": batch.protein_mask.float(),
        "dna_mask": batch.dna_mask.float(),
        "rna_mask": batch.rna_mask.float(),
        "R_polymer_type": batch.polymer_type_ids,
    }


def _candidate_tokens(candidate: MolecularSystem) -> list[int]:
    protein = {name: index for index, name in enumerate(_RESTYPES[:20])}
    nucleic = {"DA": 21, "DC": 22, "DG": 23, "DT": 24, "A": 21, "C": 22, "G": 23, "U": 24}
    values: list[int] = []
    for entity_id in candidate.instances.values():
        entity = candidate.entities[entity_id]
        if not isinstance(entity, PolymerSequence):
            continue
        mapping = protein if entity.polymer_type == "protein" else nucleic
        values.extend(mapping[component.component_id] for component in entity.components)
    return values


def _training_loss(
    log_probabilities: torch.Tensor,
    batch: NaMpnnInput,
    probabilities: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    shared_nucleic_tokens: bool,
) -> torch.Tensor:
    r"""Reimplement released ``loss_smoothed`` with an explicit smoothing target.

    Mirrors upstream ``na_model_utils.loss_smoothed`` at ``tokens=6000``, including
    the PPM-injection path where ``probabilities`` is an aligned distribution rather
    than a one-hot label.
    """

    protein_tokens = log_probabilities.new_zeros(33)
    protein_tokens[:21] = 1
    dna_tokens = log_probabilities.new_zeros(33)
    dna_tokens[21:26] = 1
    if shared_nucleic_tokens:
        rna_tokens = dna_tokens
    else:
        rna_tokens = log_probabilities.new_zeros(33)
        rna_tokens[26:31] = 1
    smoothing = (
        batch.protein_mask[..., None] * protein_tokens * (0.1 / 21.0)
        + batch.dna_mask[..., None] * dna_tokens * (0.1 / 5.0)
        + batch.rna_mask[..., None] * rna_tokens * (0.1 / 5.0)
    )
    probabilities = probabilities.clone()
    probabilities[..., (protein_tokens + dna_tokens + rna_tokens).bool()] *= 0.9
    per_token = -((probabilities + smoothing) * log_probabilities).sum(dim=-1)
    return (per_token * loss_mask).sum() / 6000.0


def _one_hot_target(batch: NaMpnnInput) -> torch.Tensor:
    return torch.nn.functional.one_hot(batch.sequence_tokens, 33).to(torch.float64)


def _valid_identity_mask(batch: NaMpnnInput) -> torch.Tensor:
    valid_identity = ~torch.isin(batch.sequence_tokens, torch.tensor((20, 25, 30, 31, 32)))
    return batch.valid_mask * valid_identity


def _upstream_smoothed_loss(
    module: object,
    batch: NaMpnnInput,
    log_probabilities: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    shared_nucleic_tokens: bool,
    ppm_mask: torch.Tensor,
    aligned_ppm: torch.Tensor,
) -> torch.Tensor:
    protein_tokens = torch.zeros(33)
    protein_tokens[:21] = 1
    dna_tokens = torch.zeros(33)
    dna_tokens[21:26] = 1
    if shared_nucleic_tokens:
        rna_tokens = dna_tokens
    else:
        rna_tokens = torch.zeros(33)
        rna_tokens[26:31] = 1
    _per_token, loss = module.loss_smoothed(  # type: ignore[attr-defined]
        batch.sequence_tokens,
        log_probabilities,
        loss_mask,
        polymer_masks={
            "protein": batch.protein_mask,
            "dna": batch.dna_mask,
            "rna": batch.rna_mask,
        },
        polymer_restype_masks={
            "protein": protein_tokens,
            "dna": dna_tokens,
            "rna": rna_tokens,
        },
        polymer_restype_nums={"protein": 21, "dna": 5, "rna": 5},
        weight=0.1,
        tokens=6000.0,
        num_letters=33,
        ppm_mask=ppm_mask,
        aligned_ppm=aligned_ppm,
    )
    return loss


def _load_upstream(reference: Path) -> tuple[object, object]:
    inference_spec = importlib.util.spec_from_file_location(
        "nampnn_upstream_model_utils",
        reference / "inference" / "model_utils.py",
    )
    inference_module = importlib.util.module_from_spec(inference_spec)
    assert inference_spec.loader is not None
    inference_spec.loader.exec_module(inference_module)
    training_spec = importlib.util.spec_from_file_location(
        "nampnn_upstream_training_utils",
        reference / "na_model_utils.py",
    )
    training_module = importlib.util.module_from_spec(training_spec)
    assert training_spec.loader is not None
    training_spec.loader.exec_module(training_module)
    return inference_module, training_module


def _released_variant(
    inference_module: object,
    training_module: object,
    reference: Path,
    relative_path: str,
    temperature: float,
    is_specificity: bool,
    batch: NaMpnnInput,
    specificity_batch: NaMpnnInput,
    fixed_noise: torch.Tensor,
    atol: float,
) -> dict[str, object]:
    checkpoint = torch.load(reference / relative_path, map_location="cpu", weights_only=True)
    official = _official_model(inference_module).eval()
    official.load_state_dict(checkpoint["model_state_dict"], strict=True)
    converted = convert_nampnn_state_dict(checkpoint["model_state_dict"])

    # Registered convert -> save -> load round trip (previously broken because the
    # architecture was unregistered).
    with TemporaryDirectory() as temporary_directory:
        repository = Path(temporary_directory) / "repository"
        convert_nampnn_checkpoint(reference / relative_path, repository)
        native = AutoModel(repository).eval()
        repository_processor = AutoProcessor(repository)
    assert type(native) is NaMpnnModel
    assert type(repository_processor) is NaMpnnProcessor
    repository_batch = repository_processor(_fixture())
    repository_checks = {
        "repository_atom_coordinates": _exact_check(
            torch.equal(repository_batch.atom_coordinates, batch.atom_coordinates)
        ),
        "repository_sequence_tokens": _exact_check(
            torch.equal(repository_batch.sequence_tokens, batch.sequence_tokens)
        ),
    }

    conditional_ranks = _ranks_from_noise(_effective_chain_mask(batch), fixed_noise)
    target_probabilities = _one_hot_target(batch)
    target_loss_mask = _valid_identity_mask(batch)
    with torch.inference_mode():
        expected_forward = official.score({**_features(batch), "randn": fixed_noise, "batch_size": 1})["log_probs"]
        actual_forward = native(batch, decoding_ranks=conditional_ranks).log_probabilities
        expected_unconditional = official.unconditional_probs({**_features(batch), "batch_size": 1})["log_probs"]
        actual_unconditional = native.unconditional_probabilities(batch).log_probabilities

    actual_training_loss = _training_loss(
        actual_forward,
        batch,
        target_probabilities,
        target_loss_mask,
        shared_nucleic_tokens=True,
    )
    zero_ppm = torch.zeros_like(batch.valid_mask)
    expected_training_loss = _upstream_smoothed_loss(
        training_module,
        batch,
        expected_forward,
        target_loss_mask,
        shared_nucleic_tokens=True,
        ppm_mask=zero_ppm,
        aligned_ppm=target_probabilities,
    )

    ppm_training_error = None
    if is_specificity:
        ppm_probabilities = target_probabilities.clone()
        ppm_probabilities[0, 4] = 0
        ppm_probabilities[0, 4, 21:25] = torch.tensor((0.1, 0.2, 0.3, 0.4))
        actual_ppm_loss = _training_loss(
            actual_forward,
            batch,
            ppm_probabilities,
            target_loss_mask,
            shared_nucleic_tokens=True,
        )
        ppm_mask = torch.zeros_like(batch.valid_mask)
        ppm_mask[0, 4] = True
        expected_ppm_loss = _upstream_smoothed_loss(
            training_module,
            batch,
            expected_forward,
            target_loss_mask,
            shared_nucleic_tokens=True,
            ppm_mask=ppm_mask,
            aligned_ppm=ppm_probabilities,
        )
        ppm_training_error = float((expected_ppm_loss - actual_ppm_loss).abs())

    # Sampling with a dense adjacent-token pair bias, an explicit shared decode
    # order, and a matching released randn.
    seed = 77
    design_batch = specificity_batch if is_specificity else batch
    length = design_batch.sequence_tokens.shape[1]
    pair_bias = torch.linspace(-0.2, 0.2, length * 33 * length * 33).reshape(1, length, 33, length, 33)
    design_batch = replace(design_batch, pair_logit_bias=pair_bias)
    sample_ranks = _shared_sample_ranks(design_batch)
    sample_randn = _upstream_randn(sample_ranks, _effective_chain_mask(design_batch)).float()
    design_features = {
        **_features(design_batch),
        "randn": sample_randn,
        "batch_size": 1,
        "temperature": temperature,
        "bias": _upstream_bias(design_batch),
        "pair_bias": pair_bias,
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
    }
    torch.manual_seed(seed)
    with torch.inference_mode():
        expected_design = official.sample(design_features)
    actual_sampling = native.sample(
        design_batch,
        temperature=temperature,
        decoding_ranks=sample_ranks,
        generator=torch.Generator().manual_seed(seed),
    )

    # design() realizes exactly what sample() produces at one shared seed; verify
    # the realized MolecularSystem reproduces those tokens.
    realize_seed = 89
    realize_reference = native.sample(
        design_batch, temperature=temperature, generator=torch.Generator().manual_seed(realize_seed)
    )
    realized = native.design(
        design_batch, temperature=temperature, generator=torch.Generator().manual_seed(realize_seed)
    )

    profile_error = None
    if is_specificity:
        profile_seed = 103
        profile_samples = 4
        profile_batch = replace(specificity_batch, pair_logit_bias=pair_bias)
        na_design = profile_batch.design_mask & ~profile_batch.protein_mask
        fixed_input = replace(profile_batch, design_mask=na_design)
        reference_sample = native.sample(
            fixed_input,
            num_samples=profile_samples,
            temperature=0.6,
            generator=torch.Generator().manual_seed(profile_seed),
        )
        reference_mean = reference_sample.sampling_probabilities.to(torch.float64).mean(dim=0)
        profile_result = native.specificity_profile(
            profile_batch,
            num_samples=profile_samples,
            temperature=0.6,
            generator=torch.Generator().manual_seed(profile_seed),
        )
        profile_error = _specificity_profile_error(profile_result, fixed_input, reference_mean)

    # Independent decoding orders scored one released call at a time (the released
    # batched score path reuses only the first order; OFoldX evaluates each order).
    score_seed = 91
    torch.manual_seed(score_seed)
    score_noise = torch.randn(4, length)
    score_ranks = torch.cat(
        [_ranks_from_noise(batch.valid_mask.to(torch.float64), noise[None]) for noise in score_noise],
        dim=0,
    )
    with torch.inference_mode():
        expected_log_probabilities = torch.cat(
            [
                official.score(
                    {
                        **_features(batch),
                        "chain_mask": batch.valid_mask.float(),
                        "randn": noise[None],
                        "batch_size": 1,
                    }
                )["log_probs"]
                for noise in score_noise
            ]
        )
        actual_score = native.score(batch, num_decoding_orders=4, decoding_ranks=score_ranks)
    expected_nll = -torch.gather(
        expected_log_probabilities,
        -1,
        batch.sequence_tokens.repeat(4, 1).unsqueeze(-1),
    ).squeeze(-1)
    expected_order_nll = expected_nll.mean(dim=-1).numpy()
    actual_nll = -torch.gather(
        actual_score.log_probabilities,
        -1,
        batch.sequence_tokens.repeat(4, 1).unsqueeze(-1),
    ).squeeze(-1)
    actual_order_nll = actual_nll.mean(dim=-1).numpy()

    expected_probabilities = expected_design["sampling_probs"].numpy()
    actual_probabilities = actual_sampling.sampling_probabilities.numpy()
    designed_rows = design_batch.design_mask[0].numpy()
    checks = {
        **repository_checks,
        "conditional_log_probabilities": _numerical_check(
            float((expected_forward - actual_forward).abs().max()),
            atol,
        ),
        "unconditional_log_probabilities": _numerical_check(
            float((expected_unconditional - actual_unconditional).abs().max()),
            atol,
        ),
        "training_loss": _numerical_check(
            float((expected_training_loss - actual_training_loss).abs()),
            atol,
        ),
        "design_probabilities": _numerical_check(
            float(np.max(np.abs(expected_probabilities[:, designed_rows] - actual_probabilities[:, designed_rows]))),
            atol,
        ),
        "design_tokens": _exact_check(
            _candidate_tokens(realized.candidates[0].system) == realize_reference.sequences[0].tolist()
        ),
        "raw_sampling_tokens": _exact_check(torch.equal(expected_design["S"], actual_sampling.sequences)),
        "raw_sampling_log_probabilities": _numerical_check(
            float((expected_design["log_probs"] - actual_sampling.log_probabilities).abs().max()),
            atol,
        ),
        "raw_sampling_decoding_orders": _exact_check(
            torch.equal(
                expected_design["decoding_order"],
                torch.argsort(actual_sampling.decoding_ranks, dim=-1),
            )
        ),
        "score_order_nll": _numerical_check(
            float(np.max(np.abs(expected_order_nll - actual_order_nll))),
            atol,
        ),
    }
    if profile_error is not None:
        checks["specificity_profile"] = _numerical_check(profile_error, atol)
    if ppm_training_error is not None:
        checks["specificity_ppm_training_loss"] = _numerical_check(ppm_training_error, atol)
    return {
        "checkpoint": relative_path,
        "checkpoint_step": checkpoint["step"],
        "tensor_count": len(converted),
        "temperature": temperature,
        "design_probability_scope": "designable_rows",
        "nonzero_pair_bias": True,
        "checks": checks,
        "passed": _checks_passed(checks),
    }


def _specificity_profile_error(
    result: object,
    fixed_input: NaMpnnInput,
    reference_mean: torch.Tensor,
) -> float:
    r"""Compare the released base metrics with a direct fixed-protein sample mean."""

    references = fixed_input.component_references[0]
    polymer_types = fixed_input.polymer_types[0]
    selected = (fixed_input.design_mask[0] & fixed_input.valid_mask[0]).tolist()
    na_rows = [
        index
        for index, (reference, polymer, keep) in enumerate(zip(references, polymer_types, selected))
        if reference is not None and keep and polymer in ("dna", "rna")
    ]
    # The released shared-token specificity model reads the four DNA base columns.
    base_columns = {"adenine": 21, "cytosine": 22, "guanine": 23, "thymine_uracil": 24}
    error = 0.0
    for base, column in base_columns.items():
        expected = np.array([float(reference_mean[row, column]) for row in na_rows], dtype=np.float64)
        actual = result.metrics[f"{base}_probability"].values  # type: ignore[attr-defined]
        error = max(error, float(np.max(np.abs(expected - actual))))
    return error


def run(reference: Path, atol: float) -> dict[str, object]:
    reference_revision = subprocess.run(
        ("git", "-C", str(reference), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reference_dirty = bool(
        subprocess.run(
            ("git", "-C", str(reference), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    inference_module, training_module = _load_upstream(reference)

    batch = NaMpnnProcessor()(_fixture())
    specificity_batch = NaMpnnProcessor()(
        _fixture(),
        conditions=(
            AllowedResidueIdentities((("PROTEIN", 0),), frozenset({"ALA"})),
            AllowedResidueIdentities((("PROTEIN", 1),), frozenset({"GLY"})),
        ),
    )
    # Single-identity conditions lock a row to one generatable token; the row stays
    # designable but its support is a singleton.
    assert int(specificity_batch.allowed_token_mask[0, 0].sum()) == 1
    assert int(specificity_batch.allowed_token_mask[0, 1].sum()) == 1
    fixed_noise = torch.tensor([[0.7, 0.1, 0.5, 0.4, 0.3, 0.2]])

    comparisons: dict[str, object] = {
        variant: _released_variant(
            inference_module,
            training_module,
            reference,
            relative_path,
            temperature,
            variant == "specificity",
            batch,
            specificity_batch,
            fixed_noise,
            atol,
        )
        for variant, relative_path, temperature in (
            ("design", "models/design_model/s_19137.pt", 0.1),
            ("specificity", "models/specificity_model/s_70114.pt", 0.6),
        )
    }

    controls = _controls(
        inference_module,
        training_module,
        reference,
        batch,
        fixed_noise,
        atol,
    )
    passed = all(bool(comparison["passed"]) for comparison in comparisons.values()) and all(
        bool(control["passed"]) for control in controls.values()
    )

    return {
        "reference": {
            "repository": "https://github.com/baker-laboratory/NA-MPNN",
            "revision": reference_revision,
            "dirty": reference_dirty,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "fixture": {
            "polymer_types": ["protein", "dna", "rna"],
            "component_ids": ["ALA", "GLY", "DA", "DT", "A", "U"],
            "atom_order": list(_ATOM_NAMES),
            "shared_nucleic_tokens": True,
        },
        "score_protocol": (
            "Each stochastic decoding order is reconstructed as ranks and compared with one released "
            "score call. The released batched score path builds its autoregressive mask before repeating "
            "the graph and therefore reuses only the first order; OFoldX evaluates the requested "
            "independent orders instead of preserving that batching bug."
        ),
        "absolute_tolerance": atol,
        "comparisons": comparisons,
        "controls": controls,
        "passed": passed,
    }


def _controls(
    inference_module: object,
    training_module: object,
    reference: Path,
    batch: NaMpnnInput,
    fixed_noise: torch.Tensor,
    atol: float,
) -> dict[str, object]:
    control_checkpoint = torch.load(
        reference / "models/design_model/s_19137.pt",
        map_location="cpu",
        weights_only=True,
    )
    control_state = convert_nampnn_state_dict(control_checkpoint["model_state_dict"])
    length = batch.sequence_tokens.shape[1]

    # ------------------------------------------------------------------
    # Separate-token RNA alphabet: conditional score, smoothed/NLL objectives,
    # and sampling.
    # ------------------------------------------------------------------
    separate_batch = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=False))(_fixture())
    separate_official = _official_model(inference_module, shared_nucleic_tokens=False).eval()
    separate_official.load_state_dict(control_checkpoint["model_state_dict"], strict=True)
    separate_native = NaMpnnModel(NaMpnnConfig()).eval()
    separate_native.load_state_dict(control_state, strict=True)

    separate_ranks = _ranks_from_noise(_effective_chain_mask(separate_batch), fixed_noise)
    separate_target = _one_hot_target(separate_batch)
    separate_loss_mask = _valid_identity_mask(separate_batch)
    with torch.inference_mode():
        separate_expected_forward = separate_official.score(
            {**_features(separate_batch), "randn": fixed_noise, "batch_size": 1}
        )["log_probs"]
        separate_actual_forward = separate_native(separate_batch, decoding_ranks=separate_ranks).log_probabilities
    separate_actual_loss = _training_loss(
        separate_actual_forward,
        separate_batch,
        separate_target,
        separate_loss_mask,
        shared_nucleic_tokens=False,
    )
    zero_ppm = torch.zeros_like(separate_batch.valid_mask)
    separate_expected_loss = _upstream_smoothed_loss(
        training_module,
        separate_batch,
        separate_expected_forward,
        separate_loss_mask,
        shared_nucleic_tokens=False,
        ppm_mask=zero_ppm,
        aligned_ppm=separate_target,
    )
    # Frozen objectives: smoothed_loss reproduces the released label smoothing over
    # the separate-token spans, and nll_loss the released NLL.
    objective_smoothed = smoothed_loss(
        separate_batch.sequence_tokens,
        separate_actual_forward,
        separate_batch.polymer_type_ids,
        weight=0.1,
    )
    objective_smoothed_loss = (objective_smoothed * separate_loss_mask).sum() / 6000.0
    objective_nll = nll_loss(separate_batch.sequence_tokens, separate_actual_forward)
    objective_nll_av = (objective_nll * separate_loss_mask).sum() / separate_loss_mask.sum()
    _loss, expected_nll_av, _accuracy = training_module.loss_nll(
        separate_batch.sequence_tokens,
        separate_expected_forward,
        separate_loss_mask,
    )

    separate_sample_ranks = _shared_sample_ranks(separate_batch)
    separate_sample_randn = _upstream_randn(separate_sample_ranks, _effective_chain_mask(separate_batch)).float()
    separate_seed = 131
    separate_features = {
        **_features(separate_batch),
        "randn": separate_sample_randn,
        "batch_size": 1,
        "temperature": 0.1,
        "bias": _upstream_bias(separate_batch),
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
    }
    torch.manual_seed(separate_seed)
    with torch.inference_mode():
        separate_expected_sample = separate_official.sample(separate_features)
    separate_actual_sample = separate_native.sample(
        separate_batch,
        temperature=0.1,
        decoding_ranks=separate_sample_ranks,
        generator=torch.Generator().manual_seed(separate_seed),
    )

    separate_checks = {
        "sequence_tokens": _exact_check(separate_batch.sequence_tokens[0].tolist() == [0, 7, 21, 24, 26, 29]),
        "conditional_log_probabilities": _numerical_check(
            float((separate_expected_forward - separate_actual_forward).abs().max()),
            atol,
        ),
        "training_loss": _numerical_check(
            float((separate_expected_loss - separate_actual_loss).abs()),
            atol,
        ),
        "objective_smoothed_loss": _numerical_check(
            float((separate_expected_loss - objective_smoothed_loss).abs()),
            atol,
        ),
        "objective_nll_loss": _numerical_check(
            float((expected_nll_av - objective_nll_av).abs()),
            atol,
        ),
        "sampling_probabilities": _numerical_check(
            float(
                (separate_expected_sample["sampling_probs"] - separate_actual_sample.sampling_probabilities).abs().max()
            ),
            atol,
        ),
        "sampling_tokens": _exact_check(torch.equal(separate_expected_sample["S"], separate_actual_sample.sequences)),
    }

    # ------------------------------------------------------------------
    # Position bias on the free rows and a weighted, tied symmetry group.
    # ------------------------------------------------------------------
    position_bias = torch.linspace(-0.3, 0.3, length * 33).reshape(-1, 33)
    reference_refs = NaMpnnProcessor()(
        _fixture(),
        conditions=(TiedResidueIdentities(((("DNA", 0), ("DNA", 1)),)),),
    ).component_references[0]
    # The native sampler sums the aligned residue bias over a tied unit's members
    # while the released sampler reads only the last member's bias; keep the tied
    # DNA pair unbiased so both agree, and bias the free rows.
    tied_positions = {("DNA", 0), ("DNA", 1)}
    symmetry_residue_bias = {
        reference: position_bias[index]
        for index, reference in enumerate(reference_refs)
        if reference not in tied_positions
    }
    symmetry_batch = NaMpnnProcessor()(
        _fixture(),
        conditions=(TiedResidueIdentities(((("DNA", 0), ("DNA", 1)),)),),
        residue_logit_bias=symmetry_residue_bias,
        tied_weights={("DNA", 0): 0.3, ("DNA", 1): 1.7},
    )
    symmetry_ranks = _shared_sample_ranks(symmetry_batch)
    symmetry_randn = _upstream_randn(symmetry_ranks, _effective_chain_mask(symmetry_batch)).float()
    symmetry_seed = 149
    symmetry_features = {
        **_features(symmetry_batch),
        "randn": symmetry_randn,
        "batch_size": 1,
        "temperature": 0.1,
        "bias": _upstream_bias(symmetry_batch),
        "symmetry_residues": [[2, 3]],
        "symmetry_weights": [[0.3, 1.7]],
    }
    shared_official = _official_model(inference_module).eval()
    shared_official.load_state_dict(control_checkpoint["model_state_dict"], strict=True)
    torch.manual_seed(symmetry_seed)
    with torch.inference_mode():
        symmetry_expected = shared_official.sample(symmetry_features)
    symmetry_actual = separate_native.sample(
        symmetry_batch,
        temperature=0.1,
        decoding_ranks=symmetry_ranks,
        generator=torch.Generator().manual_seed(symmetry_seed),
    )
    # The tied unit is decoded last, so every untied row's probability matches the
    # released sampler exactly; the tied rows intentionally diverge because native's
    # simultaneous unit is permutation-invariant (§4.10) and does not copy the
    # released member-order coupling. The weighted combination and its draw are still
    # verified against the released tokens.
    group_ids = symmetry_batch.token_group_ids[0].tolist()
    group_counts = Counter(group_ids)
    untied_rows = [index for index, group_id in enumerate(group_ids) if group_counts[group_id] == 1]
    tied_rows = [index for index, group_id in enumerate(group_ids) if group_counts[group_id] > 1]
    symmetry_checks = {
        "tied_group_ids": _exact_check(group_ids == [0, 1, 2, 2, 3, 4]),
        "untied_sampling_probabilities": _numerical_check(
            float(
                (
                    symmetry_expected["sampling_probs"][:, untied_rows]
                    - symmetry_actual.sampling_probabilities[:, untied_rows]
                )
                .abs()
                .max()
            ),
            atol,
        ),
        "tied_members_share_token": _exact_check(
            bool((symmetry_actual.sequences[0, tied_rows] == symmetry_actual.sequences[0, tied_rows[0]]).all())
        ),
        "sampling_tokens": _exact_check(torch.equal(symmetry_expected["S"], symmetry_actual.sequences)),
    }

    # ------------------------------------------------------------------
    # Neighbor-count override.
    # ------------------------------------------------------------------
    neighbor_official = _official_model(inference_module, k_neighbors=3).eval()
    neighbor_official.load_state_dict(control_checkpoint["model_state_dict"], strict=True)
    neighbor_native = NaMpnnModel(NaMpnnConfig(k_neighbors=3)).eval()
    neighbor_native.load_state_dict(control_state, strict=True)
    neighbor_ranks = _ranks_from_noise(_effective_chain_mask(batch), fixed_noise)
    with torch.inference_mode():
        neighbor_expected = neighbor_official.score({**_features(batch), "randn": fixed_noise, "batch_size": 1})[
            "log_probs"
        ]
        neighbor_actual = neighbor_native(batch, decoding_ranks=neighbor_ranks).log_probabilities
    neighbor_checks = {
        "conditional_log_probabilities": _numerical_check(
            float((neighbor_expected - neighbor_actual).abs().max()),
            atol,
        )
    }

    # ------------------------------------------------------------------
    # Protein-first teacher forcing.
    # ------------------------------------------------------------------
    decode_official = _official_training_model(training_module, decode_protein_first=True).eval()
    decode_official.load_state_dict(control_checkpoint["model_state_dict"], strict=True)
    decode_native = NaMpnnModel(NaMpnnConfig()).eval()
    decode_native.load_state_dict(control_state, strict=True)
    decode_seed = 163
    torch.manual_seed(decode_seed)
    with torch.inference_mode():
        decode_expected = decode_official(_features(batch))[0]
    torch.manual_seed(decode_seed)
    decode_noise = torch.randn(batch.valid_mask.shape)
    decode_chain_mask = batch.valid_mask.to(torch.float64).masked_fill(batch.protein_mask.bool(), 0.0)
    decode_ranks = _ranks_from_noise(decode_chain_mask, decode_noise)
    with torch.inference_mode():
        decode_actual = decode_native(batch, decoding_ranks=decode_ranks).log_probabilities
    # The frozen decode_protein_first argument must decode every protein unit before
    # any nucleic unit.
    protein_first_ranks = decode_native(
        batch,
        decode_protein_first=True,
        generator=torch.Generator().manual_seed(decode_seed),
    ).decoding_ranks
    protein_rows = batch.protein_mask[0]
    protein_first = bool(protein_first_ranks[0][protein_rows].max() < protein_first_ranks[0][~protein_rows].min())
    decode_checks = {
        "conditional_log_probabilities": _numerical_check(
            float((decode_expected - decode_actual).abs().max()),
            atol,
        ),
        "protein_decoded_first": _exact_check(protein_first),
    }

    # ------------------------------------------------------------------
    # Explicit training-mode coordinate augmentation.
    # ------------------------------------------------------------------
    augmentation_official = _official_training_model(
        training_module,
        decode_protein_first=False,
        dropout=0.0,
    ).train()
    augmentation_official.load_state_dict(control_checkpoint["model_state_dict"], strict=True)
    augmentation_native = NaMpnnModel(NaMpnnConfig(dropout=0.0)).train()
    augmentation_native.load_state_dict(control_state, strict=True)
    augmentation_seed = 181
    torch.manual_seed(augmentation_seed)
    augmentation_expected = augmentation_official(_features(batch))[0]
    augmentation_generator = torch.Generator().manual_seed(augmentation_seed)
    polymer_mask = (batch.protein_mask + batch.dna_mask + batch.rna_mask).to(batch.atom_coordinates.dtype)
    augmented_coordinates = batch.atom_coordinates + (
        batch.atom_mask[..., None]
        * polymer_mask[..., None, None]
        * 0.1
        * torch.randn(
            batch.atom_coordinates.shape,
            dtype=batch.atom_coordinates.dtype,
            device=batch.atom_coordinates.device,
            generator=augmentation_generator,
        )
    )
    augmentation_noise = torch.randn(
        batch.valid_mask.shape,
        device=batch.atom_coordinates.device,
        generator=augmentation_generator,
    )
    augmentation_ranks = _ranks_from_noise(batch.valid_mask.to(torch.float64), augmentation_noise)
    augmented_batch = replace(batch, atom_coordinates=augmented_coordinates)
    augmentation_actual = augmentation_native(
        augmented_batch,
        decoding_ranks=augmentation_ranks,
    ).log_probabilities.detach()
    augmentation_checks = {
        "conditional_log_probabilities": _numerical_check(
            float((augmentation_expected.detach() - augmentation_actual).abs().max()),
            atol,
        )
    }

    return {
        "separate_rna_tokens": {
            "sequence_tokens": separate_batch.sequence_tokens[0].tolist(),
            "checks": separate_checks,
            "passed": _checks_passed(separate_checks),
        },
        "position_bias_and_weighted_symmetry": {
            "checks": symmetry_checks,
            "passed": _checks_passed(symmetry_checks),
        },
        "k_neighbors_override": {
            "checks": neighbor_checks,
            "passed": _checks_passed(neighbor_checks),
        },
        "decode_protein_first": {
            "checks": decode_checks,
            "passed": _checks_passed(decode_checks),
        },
        "explicit_training_augmentation": {
            "checks": augmentation_checks,
            "passed": _checks_passed(augmentation_checks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path.home() / "reference" / "NA-MPNN",
    )
    parser.add_argument("--atol", type=float, default=_FLOAT32_ATOL)
    args = parser.parse_args()
    if args.atol < 0:
        parser.error("--atol must be non-negative")
    report = run(args.reference, args.atol)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
