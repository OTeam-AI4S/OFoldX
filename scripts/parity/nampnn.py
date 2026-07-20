#!/usr/bin/env python
r"""On-demand NA-MPNN parity: native vertical vs released checkpoints and upstream.

Run under the real-weight environment:

    /data/miniforge3/envs/zyc/bin/python scripts/parity/nampnn.py

    PYTHONPATH=/data/node4/home/zhiyuan/ofoldx
    TRITON_CACHE_DIR=/tmp/claude-2667/triton

The script covers, for both released checkpoints, the folded encoder/decoder
numerics against the upstream released model (deterministic unconditional and
teacher-forced score passes over a mixed protein/DNA/RNA graph, so the RNA
16-atom O2' layout, the predicted nucleobase-N augmentation, and the RNA polymer
one-hot are all driven), the mixed-polymer lowering with source-number
running-count offsets and shared/separate tokens (RNA A/C/G/U onto DA/DC/DG/DT
21..24 or onto its own 26..29 span), the occupancy demotion of an RNA monomer
missing O2', exact ties vs token symmetry, the simultaneous bias equation, the
design/score/unconditional/specificity operations (with the specificity RNA rows
read from the shared DA/DC/DG/DT base columns), the training objectives, and the
``convert -> save -> AutoModel/Pipeline`` round trip that previously failed at
``_registration_for_pair``.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ofoldx import AutoModel, AutoProcessor  # noqa: E402
from ofoldx.data.components import get_component  # noqa: E402
from ofoldx.data.conditions import AllowedResidueIdentities, TiedResidueIdentities  # noqa: E402
from ofoldx.data.molecular import AtomSite, MolecularSystem, PolymerSequence  # noqa: E402
from ofoldx.data.structure import ResidueNumber, Structure  # noqa: E402
from ofoldx.models.nampnn import (  # noqa: E402
    NaMpnnConfig,
    NaMpnnModel,
    NaMpnnPipeline,
    NaMpnnProcessor,
    NaMpnnProcessorConfig,
    nll_loss,
    smoothed_loss,
)
from ofoldx.models.nampnn.convert import (  # noqa: E402
    convert_nampnn_checkpoint,
    convert_nampnn_state_dict,
)
from ofoldx.models.nampnn.model import MIXED_POLYMER_TOKEN_BY_COMPONENT  # noqa: E402
from ofoldx.models.nampnn.model import NaMpnnInput  # noqa: E402
from ofoldx.modules.mpnn import _decoding_order_to_ranks  # noqa: E402

_REFERENCE = Path("/data/node4/home/zhiyuan/reference/NA-MPNN")
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
_ATOM_INDEX = {name: index for index, name in enumerate(_ATOM_NAMES)}
_CHECKPOINTS = {
    "design": "models/design_model/s_19137.pt",
    "specificity": "models/specificity_model/s_70114.pt",
}
_PROTEIN_ATOMS = ("N", "CA", "C", "O")
_DNA_ATOMS = ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
# RNA carries the full 16-atom nucleic backbone including O2', which DNA lacks; the
# released featurize demotes any nucleic monomer missing a required backbone atom.
_RNA_ATOMS = ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")


def _load_upstream():
    spec = importlib.util.spec_from_file_location("na_upstream_model", _REFERENCE / "inference" / "model_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _polytype_and_restype():
    from ofoldx.models.nampnn.model import MIXED_POLYMER_COMPONENT_IDS

    atom_dict = dict(zip(_ATOM_NAMES, range(len(_ATOM_NAMES))))
    polytypes = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
    polytype_to_int = dict(zip(polytypes, range(len(polytypes))))
    restype_to_int = {name: index for index, name in enumerate(MIXED_POLYMER_COMPONENT_IDS)}
    return atom_dict, polytype_to_int, restype_to_int


def _synthetic_features(device):
    torch.manual_seed(7)
    protein_rows = 6
    dna_rows = 4
    rna_rows = 4
    length = protein_rows + dna_rows + rna_rows
    coordinates = torch.zeros(1, length, 16, 3)
    atom_mask = torch.zeros(1, length, 16)
    for row in range(length):
        center = torch.randn(3) * 4 + row * 3.2
        if row < protein_rows:
            names = _PROTEIN_ATOMS
        elif row < protein_rows + dna_rows:
            names = _DNA_ATOMS
        else:
            names = _RNA_ATOMS
        for name in names:
            coordinates[0, row, _ATOM_INDEX[name]] = center + torch.randn(3) * 1.3
            atom_mask[0, row, _ATOM_INDEX[name]] = 1.0
    valid = torch.ones(1, length)
    residue_indices = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 4]], dtype=torch.long)
    chain_indices = torch.tensor([[0] * protein_rows + [1] * dna_rows + [2] * rna_rows], dtype=torch.long)
    protein_mask = torch.tensor([[1] * protein_rows + [0] * (dna_rows + rna_rows)], dtype=torch.float32)
    dna_mask = torch.tensor([[0] * protein_rows + [1] * dna_rows + [0] * rna_rows], dtype=torch.float32)
    rna_mask = torch.tensor([[0] * (protein_rows + dna_rows) + [1] * rna_rows], dtype=torch.float32)
    polymer_type_ids = torch.tensor([[0] * protein_rows + [1] * dna_rows + [2] * rna_rows], dtype=torch.long)
    tokens = [int(torch.randint(0, 20, (1,))) for _ in range(protein_rows)]
    tokens += [int(torch.randint(21, 25, (1,))) for _ in range(dna_rows)]
    # Released NA_SHARED_TOKENS=1 maps RNA A/C/G/U onto the DA/DC/DG/DT columns 21..24,
    # so a faithful RNA row carries a shared nucleic token in that span.
    tokens += [int(torch.randint(21, 25, (1,))) for _ in range(rna_rows)]
    sequence_tokens = torch.tensor([tokens], dtype=torch.long)
    return dict(
        atom_coordinates=coordinates.to(device),
        atom_mask=atom_mask.to(device),
        valid_mask=valid.to(device).bool(),
        residue_indices=residue_indices.to(device),
        chain_indices=chain_indices.to(device),
        protein_mask=protein_mask.to(device).bool(),
        dna_mask=dna_mask.to(device).bool(),
        rna_mask=rna_mask.to(device).bool(),
        polymer_type_ids=polymer_type_ids.to(device),
        sequence_tokens=sequence_tokens.to(device),
        length=length,
    )


def _native_input(features) -> NaMpnnInput:
    length = features["length"]
    return NaMpnnInput(
        atom_coordinates=features["atom_coordinates"],
        atom_mask=features["atom_mask"].bool(),
        sequence_tokens=features["sequence_tokens"],
        valid_mask=features["valid_mask"],
        residue_indices=features["residue_indices"],
        chain_indices=features["chain_indices"],
        polymer_type_ids=features["polymer_type_ids"],
        protein_mask=features["protein_mask"],
        dna_mask=features["dna_mask"],
        rna_mask=features["rna_mask"],
        design_mask=torch.ones(1, length, dtype=torch.bool, device=features["valid_mask"].device),
        token_group_ids=torch.arange(length, device=features["valid_mask"].device).unsqueeze(0),
        structures=(),
        component_references=((None,) * length,),
        polymer_types=(("protein",) * length,),
        conditions=((),),
    )


def _upstream_feature_dict(features):
    return {
        "X": features["atom_coordinates"],
        "X_m": features["atom_mask"],
        "S": features["sequence_tokens"],
        "mask": features["valid_mask"].float(),
        "R_idx": features["residue_indices"],
        "chain_labels": features["chain_indices"],
        "chain_mask": torch.ones_like(features["valid_mask"]).float(),
        "protein_mask": features["protein_mask"].float(),
        "dna_mask": features["dna_mask"].float(),
        "rna_mask": features["rna_mask"].float(),
        "R_polymer_type": features["polymer_type_ids"],
        "batch_size": 1,
        "randn": torch.randn(1, features["length"]),
    }


def encoder_parity(checkpoint_name: str) -> None:
    r"""Compare native encoder/decoder numerics against the upstream released model."""

    device = torch.device("cpu")
    upstream_module = _load_upstream()
    atom_dict, polytype_to_int, restype_to_int = _polytype_and_restype()
    checkpoint = torch.load(_REFERENCE / _CHECKPOINTS[checkpoint_name], map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    upstream = upstream_module.ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=32,
        vocab=33,
        num_letters=33,
        model_type="na_mpnn",
        atom_dict=atom_dict,
        restype_to_int=restype_to_int,
        polytype_to_int=polytype_to_int,
    ).eval()
    upstream.load_state_dict(state_dict)

    native = NaMpnnModel(NaMpnnConfig()).eval()
    native.load_state_dict(convert_nampnn_state_dict(state_dict), strict=True, assign=True)

    features = _synthetic_features(device)
    native_input = _native_input(features)
    upstream_dict = _upstream_feature_dict(features)

    with torch.inference_mode():
        upstream_unconditional = upstream.unconditional_probs(upstream_dict)["log_probs"]
        node, edge, neighbor = native._encode(native_input)
        native_unconditional = native._decode_unconditional_logits(
            node, edge, neighbor, native_input.valid_mask
        ).log_softmax(dim=-1)
        unconditional_delta = (upstream_unconditional - native_unconditional).abs().max().item()

        upstream_score = upstream.score(upstream_dict)
        order = upstream_score["decoding_order"].unsqueeze(0)
        ranks = _decoding_order_to_ranks(order)
        native_score = native._decode_logits(
            node, edge, neighbor, native_input.sequence_tokens, native_input.valid_mask, ranks
        ).log_softmax(dim=-1)
        score_delta = (upstream_score["log_probs"] - native_score).abs().max().item()

    print(
        f"[{checkpoint_name}] encoder parity: "
        f"unconditional max|delta|={unconditional_delta:.3e} "
        f"teacher_forced_score max|delta|={score_delta:.3e}"
    )
    assert unconditional_delta < 1e-4, "unconditional log-prob parity failed"
    assert score_delta < 1e-4, "teacher-forced score parity failed"

    with tempfile.TemporaryDirectory() as directory:
        convert_nampnn_checkpoint(_REFERENCE / _CHECKPOINTS[checkpoint_name], directory)
        loaded = AutoModel(directory).eval()
        loaded_processor = AutoProcessor(directory)
        assert type(loaded) is NaMpnnModel and type(loaded_processor) is NaMpnnProcessor
    print(f"[{checkpoint_name}] convert -> save -> AutoModel round trip OK")


def _build_structure() -> Structure:
    protein = PolymerSequence("protein", tuple(get_component(c) for c in ("ALA", "GLY", "SER", "LEU", "VAL")))
    dna = PolymerSequence("dna", tuple(get_component(c) for c in ("DA", "DC", "DG", "DT")))
    # Chain C is RNA: four complete monomers (A/C/G/U, each with the 16-atom layout
    # including O2') plus a trailing monomer whose O2' is zero-occupancy, so the
    # released occupancy filter demotes it to an invalid UNK row missing a required
    # nucleic atom.
    rna = PolymerSequence("rna", tuple(get_component(c) for c in ("A", "C", "G", "U", "A")))
    demoted = ("C", len(rna) - 1)
    system = MolecularSystem({"A": protein, "B": dna, "C": rna}, {"A": "A", "B": "B", "C": "C"})
    rng = np.random.default_rng(3)
    sites, coordinates, occupancy, residue_numbers = [], [], [], {}
    for entity_id, entity, atoms, offset in (
        ("A", protein, _PROTEIN_ATOMS, 0.0),
        ("B", dna, _DNA_ATOMS, 25.0),
        ("C", rna, _RNA_ATOMS, 50.0),
    ):
        for component_index in range(len(entity)):
            residue_numbers[(entity_id, component_index)] = ResidueNumber(component_index + 1, "")
            for name in atoms:
                sites.append(AtomSite(entity_id, component_index, name))
                coordinates.append(rng.standard_normal(3) * 4 + component_index * 3 + offset)
                occupancy.append(0.0 if ((entity_id, component_index) == demoted and name == "O2'") else 1.0)
    return Structure(
        system,
        tuple(sites),
        np.asarray(coordinates, dtype=np.float32),
        np.ones(len(sites), dtype=np.bool_),
        np.asarray(occupancy, dtype=np.float32),
        residue_numbers,
    )


def _assert_specificity_reads_shared_columns(model, batch, specificity) -> None:
    r"""Prove the specificity profile reads the shared DA/DC/DG/DT columns for RNA.

    The released specificity checkpoint is ``NA_SHARED_TOKENS=1``, so both DNA and
    RNA rows realize their base probability in columns 21..24 (``_RNA_SHARED_TOKENS
    == _DNA_TOKENS``). Reproduce the fixed-protein sample mean the profile averages
    and confirm every reported base metric equals that mean at columns 21..24 for
    every nucleic row, RNA included, which drives the RNA branch of
    ``NaMpnnModel.specificity_profile``.
    """

    na_design = batch.design_mask & ~batch.protein_mask
    fixed_input = replace(batch, design_mask=na_design)
    reference_mean = (
        model.sample(
            fixed_input,
            num_samples=6,
            temperature=0.6,
            generator=torch.Generator(device=fixed_input.atom_coordinates.device).manual_seed(14),
        )
        .sampling_probabilities.to(torch.float64)
        .mean(dim=0)
    )
    references = batch.component_references[0]
    polymer_types = batch.polymer_types[0]
    selected = (na_design[0] & batch.valid_mask[0]).tolist()
    na_rows = [
        index
        for index, (reference, polymer, keep) in enumerate(zip(references, polymer_types, selected))
        if reference is not None and keep and polymer in ("dna", "rna")
    ]
    assert any(polymer_types[row] == "rna" for row in na_rows), "specificity axis must include RNA rows"
    base_columns = {
        "adenine_probability": 21,
        "cytosine_probability": 22,
        "guanine_probability": 23,
        "thymine_uracil_probability": 24,
    }
    worst = 0.0
    for metric_name, column in base_columns.items():
        expected = np.array([float(reference_mean[row, column]) for row in na_rows], dtype=np.float64)
        actual = specificity.metrics[metric_name].values
        worst = max(worst, float(np.max(np.abs(expected - actual))))
    assert worst < 1e-9, f"specificity RNA/DNA base columns 21..24 mismatch: {worst:.3e}"


def lowering_and_operations() -> None:
    r"""Exercise lowering, symmetry, the operations, objectives, and the round trip."""

    structure = _build_structure()
    device = torch.device("cpu")
    model = NaMpnnModel(NaMpnnConfig(hidden_size=48, feature_size=48, k_neighbors=5)).eval().to(device)
    pipeline = NaMpnnPipeline(model=model, processor=NaMpnnProcessor())

    shared = pipeline.processor(structure)
    separate = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=False))(structure)
    # Rows: protein A[0:5], DNA B[5:9], RNA C[9:14] whose trailing monomer (index 13)
    # is demoted for a zero-occupancy O2'. Running-count renumber restarts per chain.
    expected_residue_indices = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 5]]
    assert shared.residue_indices.tolist() == expected_residue_indices, "running-count renumber offsets"
    shared_tokens = shared.sequence_tokens[0].tolist()
    separate_tokens = separate.sequence_tokens[0].tolist()
    # RNA is typed from its canonical component, not from observed O2': the four valid
    # RNA monomers map onto shared DA/DC/DG/DT columns 21..24, or onto their own
    # 26..29 columns when shared_nucleic_tokens is off.
    assert shared_tokens[9:13] == [21, 22, 23, 24], "shared-token RNA maps A/C/G/U onto DA/DC/DG/DT"
    assert separate_tokens[9:13] == [26, 27, 28, 29], "separate-token RNA owns columns 26..29"
    assert shared_tokens != separate_tokens, "RNA rows distinguish shared from separate tokens"
    # The O2'-missing RNA monomer is demoted: invalid, UNK polymer type, RX token.
    assert shared.valid_mask[0].tolist() == [True] * 13 + [False], "occupancy filter demotes RNA missing O2'"
    assert shared_tokens[13] == MIXED_POLYMER_TOKEN_BY_COMPONENT["RX"], "demoted RNA row lowers to RX"
    assert separate_tokens[13] == MIXED_POLYMER_TOKEN_BY_COMPONENT["RX"], "demoted RNA row lowers to RX"
    assert shared.polymer_type_ids[0].tolist() == [0] * 5 + [1] * 4 + [2] * 4 + [3], "RNA polymer typing + UNK demotion"
    assert shared.rna_mask[0].tolist() == [False] * 9 + [True] * 4 + [False], "rna_mask excludes the demoted row"
    print("[ops] RNA lowering: O2' layout, polymer typing, shared/separate tokens, demotion OK")

    design = pipeline.design(structure, num_candidates=3, seed=11)
    best = design.best("designed_sequence_confidence")
    assert len(design.candidates) == 3 and best is not None
    for candidate in design.candidates:
        for reference in ("A", "B", "C"):
            entity = candidate.system.entities[candidate.system.instances[reference]]
            assert all(component.component_id != "PAD" for component in entity.components)
    # The realized RNA chain uses canonical RNA components, never DNA identities.
    for candidate in design.candidates:
        rna_entity = candidate.system.entities[candidate.system.instances["C"]]
        assert all(component.component_id in ("A", "C", "G", "U") for component in rna_entity.components)
    print("[ops] design candidates + realization OK")

    score = pipeline.score(structure, num_decoding_orders=4, seed=12)
    assert tuple(score.logits.shape) == (4, 14, 33)
    unconditional = pipeline.unconditional_probabilities(structure, num_decoding_orders=2, seed=13)
    assert torch.unique(unconditional.decoding_ranks).numel() == 1
    print("[ops] score + unconditional MpnnProbabilityResult OK")

    specificity = pipeline.specificity_profile(structure, num_samples=6, seed=14)
    assert set(specificity.metrics) == {
        "adenine_probability",
        "cytosine_probability",
        "guanine_probability",
        "thymine_uracil_probability",
    }
    # Axis spans every valid nucleic design row: four DNA plus four RNA (the demoted
    # RNA monomer is excluded).
    assert len(specificity.metrics["adenine_probability"].axis) == 8
    _assert_specificity_reads_shared_columns(model, shared, specificity)
    print("[ops] specificity_profile ScoreResult + RNA shared-column faithfulness OK")

    exact = (
        pipeline.design(
            structure,
            conditions=[TiedResidueIdentities(groups=((("A", 0), ("A", 2)),))],
            tied_weights={("A", 0): 1.5, ("A", 2): -0.5},
            num_candidates=1,
            seed=15,
        )
        .candidates[0]
        .system
    )
    assert exact.component(("A", 0)).component_id == exact.component(("A", 2)).component_id
    restricted = (
        pipeline.design(
            structure,
            conditions=[AllowedResidueIdentities(positions=(("A", 1),), component_ids=frozenset({"TRP"}))],
            num_candidates=1,
            seed=16,
        )
        .candidates[0]
        .system
    )
    assert restricted.component(("A", 1)).component_id == "TRP"
    print("[ops] exact ties, token symmetry, allowed identities OK")

    log_probabilities = torch.log_softmax(torch.randn(1, shared.sequence_tokens.shape[1], 33), dim=-1)
    assert nll_loss(shared.sequence_tokens, log_probabilities).shape == shared.sequence_tokens.shape
    assert torch.allclose(
        smoothed_loss(shared.sequence_tokens, log_probabilities, shared.polymer_type_ids, weight=0.0),
        nll_loss(shared.sequence_tokens, log_probabilities),
        atol=1e-5,
    )
    print("[ops] nll_loss / smoothed_loss OK")

    with tempfile.TemporaryDirectory() as directory:
        pipeline.save(directory)
        loaded_model = AutoModel(directory).eval()
        loaded_processor = AutoProcessor(directory)
        assert type(loaded_model) is NaMpnnModel and type(loaded_processor) is NaMpnnProcessor
    print("[ops] convert -> save -> AutoModel/Pipeline round trip OK")


def main() -> None:
    for name in _CHECKPOINTS:
        encoder_parity(name)
    lowering_and_operations()
    print("NAMPNN_PARITY_OK")


if __name__ == "__main__":
    main()
