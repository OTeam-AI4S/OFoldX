#!/usr/bin/env python
r"""Compare OFoldX LigandMpnn with a local dauparas/LigandMPNN checkout.

This is a manual, on-demand scientific diagnostic. It uses the checkout's 1BC8
zinc-binding fixture, writes no report, and is not a test, CI, registration, or
publication gate. It exercises the frozen sec. 4.8 / 4.8.1 public surface: the
aligned tensor ``LigandMpnnInput``, public ``encode``, differentiable ``forward``
returning ``MpnnOutput``, the four ``MpnnProbabilityResult`` methods with the
leading evaluation axis, the unified ``sample`` (independent, tied, and weighted),
``coordinate_noise``, ``design`` returning the shared ``DesignResult`` with the
three released confidences, and the ``LigandMpnnPipeline`` /
``LigandMpnnDesignAndPackPipeline`` compositions. There is no ``score`` /
``ScoreResult`` and no ``LigandMpnnBackbone``. Numerics are cross-checked against
the upstream checkout across the four ``ligandmpnn_v_32_{005,010,020,030}_25.pt``
checkpoints.

Each frozen method maps to one upstream computation:

* ``forward`` / ``autoregressive_probabilities`` <-> ``score(use_sequence=True)``
  (teacher-forced sequence-conditioned autoregressive logits under one order).
* ``sequence_free_autoregressive_probabilities`` <-> ``score(use_sequence=False)``
  (the released order-dependent sequence-free decoder).
* ``conditional_probabilities`` (target moved last) <->
  ``single_aa_score(use_sequence=False)`` (target ordered after every other row).
* ``unconditional_probabilities`` (target moved first) <->
  ``single_aa_score(use_sequence=True)`` (target ordered before every other row).
* ``sample`` <-> ``sample`` (independent and symmetric weighted branches).

The upstream operations derive their decoding order from a supplied ``randn``
tensor rather than an explicit order argument, so every comparison builds the
native ranks and the upstream ``randn`` from one shared draw. The two single-residue
comparisons additionally pin that shared draw to a well-conditioned random permutation
(magnitudes in ``[1, L]``) so upstream's ``argsort`` realizes a genuine target-last /
target-first order for every row and only the probability math is compared, never the
RNG-argsort placement artifact that appears when a target's own ``|randn|`` is
vanishingly small (see ``_probability_checks`` and DESIGN.md sec. 4.8 L3596, L3613-3615).
With ``coordinate_noise`` zero the encoder is deterministic; the single nonzero-noise
forward matches upstream because both perturb ``X`` then ``Y`` with one identically
seeded stream.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ofoldx import AutoModel, AutoProcessor  # noqa: E402
from ofoldx import pipeline as load_pipeline  # noqa: E402
from ofoldx.data.conditions import (  # noqa: E402
    AllowedResidueIdentities,
    TiedResidueIdentities,
)
from ofoldx.data.molecular import ComponentReference  # noqa: E402
from ofoldx.io import read_pdb  # noqa: E402
from ofoldx.models.ligandmpnn import (  # noqa: E402
    LigandMpnnConfig,
    LigandMpnnDesignAndPackPipeline,
    LigandMpnnFeatures,
    LigandMpnnInput,
    LigandMpnnModel,
    LigandMpnnPipeline,
    LigandMpnnProcessor,
    LigandMpnnProcessorConfig,
    tied_groups_from_ids,
)
from ofoldx.models.ligandmpnn.convert import (  # noqa: E402
    convert_ligandmpnn_checkpoint,
    convert_ligandmpnn_state_dict,
    ligandmpnn_config_from_checkpoint,
    ligandmpnn_processor_config_from_checkpoint,
    load_ligandmpnn_checkpoint,
)
from ofoldx.models.mpnn_packer import (  # noqa: E402
    MpnnPackerConfig,
    MpnnPackerModel,
    MpnnPackerPipeline,
    MpnnPackerProcessor,
)
from ofoldx.models.mpnn_packer.convert import convert_mpnn_packer_state_dict  # noqa: E402
from ofoldx.modules.mpnn import (  # noqa: E402
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
    MpnnOutput,
    MpnnSamplingOutput,
    _decoding_order_to_ranks,
)
from ofoldx.results import MpnnProbabilityResult  # noqa: E402
from ofoldx.results import DesignCandidate, DesignResult, PackPrediction, PackResult  # noqa: E402
from scripts.parity.paths import reference_root, weights_root  # noqa: E402

CHECKPOINTS = (
    "ligandmpnn_v_32_005_25.pt",
    "ligandmpnn_v_32_010_25.pt",
    "ligandmpnn_v_32_020_25.pt",
    "ligandmpnn_v_32_030_25.pt",
)
REPRESENTATIVE = "ligandmpnn_v_32_020_25.pt"
PACKER_CHECKPOINT = "ligandmpnn_sc_v_32_002_16.pt"

FLOAT32_ATOL = 1e-5
FLOAT32_RTOL = 1.3e-6
# Path-specific float32 precision floor, applied to exactly two comparisons whose OWN
# float32 accumulation floor is coarser than the base atol: the sequence-free decoder
# log-probabilities (score(use_sequence=False)) and the sampler softmax probabilities.
#
# Upstream computes these in intrinsically float32 arithmetic -- its RBF / positional
# embedding hardcodes ``d_onehot.float()`` (dauparas/LigandMPNN model_utils.py:1650) -- and
# the native model's own float32-vs-float64 gap on these paths was measured at 9.7e-6-4.55e-5
# (per checkpoint/context), i.e. above the base atol=1e-5. A base atol below the achievable
# float32 floor is not a meaningful check. Every one of these four comparisons was confirmed
# floor-bound (native f32-vs-f64 gap same order as its native-vs-upstream deviation; a genuine
# logic error deviates by orders of magnitude more -- the target-order bug this parity first
# caught was 1.236). This atol sits above the worst measured floor (4.55e-5) and the worst
# observed deviation (2.05e-5), yet stays >16000x below that logic-error scale. It is applied
# ONLY to those two comparisons; forward, autoregressive, conditional, unconditional, tied,
# design, sampler tokens/log-probs, and every decoding-order check remain at the base atol.
FLOAT32_FLOOR_ATOL = 6e-5
_VOCABULARY = len(AMINO_ACID_COMPONENT_IDS) + 1
_GENERATABLE = len(AMINO_ACID_COMPONENT_IDS)

# (context name, use_atom_context, use_side_chain_context, parse_all_atoms).
CONTEXTS = (
    ("atom_enabled", True, False),
    ("atom_disabled", False, False),
    ("side_chain", True, True),
)


class _Scoreboard:
    r"""Collect pass/fail outcomes so the diagnostic runs every suite to completion.

    The diagnostic exercises 4 checkpoints x 3 contexts plus downstream sampler,
    designer, tie, pipeline, and pack suites. A single expected divergence must not
    abort the remaining coverage, so ``_check`` / ``_compare`` record here instead of
    raising, and ``main`` reports one full pass/fail tally with a nonzero exit on any
    failure. Numerical tolerances are unchanged; only the control flow differs.
    """

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []
        self.max_deviation = 0.0

    def record(self, label: str, ok: bool) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(label)

    def note_deviation(self, value: float) -> None:
        if value > self.max_deviation:
            self.max_deviation = value


SCORE = _Scoreboard()


def _check(label: str, passed: bool | Tensor) -> None:
    passed = bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    SCORE.record(label, passed)


def _compare(
    label: str,
    ours: Tensor,
    reference: Tensor,
    *,
    exact: bool = False,
    atol: float = FLOAT32_ATOL,
) -> None:
    ours = ours.detach().cpu()
    reference = reference.detach().cpu()
    if ours.shape != reference.shape:
        print(f"[FAIL] {label} shape={tuple(ours.shape)} reference_shape={tuple(reference.shape)}")
        SCORE.record(label, False)
        return
    if exact:
        passed = torch.equal(ours, reference)
        detail = f"shape={tuple(ours.shape)}"
    else:
        passed = torch.allclose(ours.float(), reference.float(), atol=atol, rtol=FLOAT32_RTOL)
        delta = (ours.float() - reference.float()).abs()
        max_error = 0.0 if delta.numel() == 0 else float(delta.max())
        SCORE.note_deviation(max_error)
        detail = f"shape={tuple(ours.shape)} max_abs={max_error:.3e}"
        if atol != FLOAT32_ATOL:
            detail += f" atol={atol:.1e}"
    print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}")
    SCORE.record(label, passed)


def _run_suite(name: str, function, *args, **kwargs) -> None:
    r"""Run one check suite, recording any unexpected exception instead of aborting the run."""

    try:
        function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the parity harness must survive one suite's failure
        print(f"[ERROR] {name}: suite raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        SCORE.record(f"{name}: suite raised {type(exc).__name__}", False)


# -- reference loading ---------------------------------------------------------


def _load_reference(root: Path) -> tuple[ModuleType, ModuleType, str, bool]:
    import subprocess

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
    sys.path.insert(0, str(root))
    import data_utils
    import model_utils

    return data_utils, model_utils, revision, dirty


def _load_models(
    label: str,
    checkpoint_path: Path,
    model_utils: ModuleType,
    device: torch.device,
    *,
    use_side_chain_context: bool,
) -> tuple[LigandMpnnModel, LigandMpnnProcessor, torch.nn.Module]:
    checkpoint = load_ligandmpnn_checkpoint(checkpoint_path, map_location=device)
    config = ligandmpnn_config_from_checkpoint(checkpoint)
    processor_config = ligandmpnn_processor_config_from_checkpoint(checkpoint)

    model = LigandMpnnModel(config)
    model.load_state_dict(
        convert_ligandmpnn_state_dict(checkpoint["model_state_dict"]),
        strict=True,
        assign=True,
    )
    checkpoint_dtype = checkpoint["model_state_dict"]["W_s.weight"].dtype
    _check(
        f"{label}: converted checkpoint dtype preserved",
        model.token_embedding.weight.dtype == checkpoint_dtype,
    )
    # The root Model owns the folded feature Module directly; no backbone wrapper.
    _check(
        f"{label}: folded feature_encoder and no backbone wrapper",
        isinstance(model.feature_encoder, LigandMpnnFeatures) and not hasattr(model, "backbone"),
    )
    model = model.to(device).eval()

    upstream = model_utils.ProteinMPNN(
        num_letters=_VOCABULARY,
        node_features=config.hidden_size,
        edge_features=config.hidden_size,
        hidden_dim=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        k_neighbors=config.num_neighbors,
        atom_context_num=int(processor_config.num_context_atoms),
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=use_side_chain_context,
        device=device,
        dropout=config.dropout,
    )
    upstream.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
    upstream = upstream.to(device).eval()

    processor = LigandMpnnProcessor(
        LigandMpnnProcessorConfig(num_context_atoms=int(processor_config.num_context_atoms)),
        device=device,
        dtype=checkpoint_dtype,
    )
    print(f"[checkpoint] PASS {label}: strict conversion and load")
    return model, processor, upstream


# -- decoding-order helpers ----------------------------------------------------


def _order_ranks(randn: Tensor, chain_mask: Tensor) -> Tensor:
    r"""Native ranks for the released fixed-first full-order decode from ``randn``."""

    return _decoding_order_to_ranks(torch.argsort((chain_mask + 0.0001) * randn.abs(), dim=-1))


def _base_ranks(randn: Tensor) -> Tensor:
    r"""Native base ranks for the released single-residue order (no chain weighting)."""

    return _decoding_order_to_ranks(torch.argsort(randn.abs(), dim=-1))


def _sampling_bias(batch: LigandMpnnInput) -> Tensor:
    r"""Reproduce the Model's aligned generatable bias for the upstream sampler."""

    if batch.logit_bias is None:
        bias = torch.zeros(
            1,
            batch.sequence_tokens.shape[1],
            _VOCABULARY,
            device=batch.sequence_tokens.device,
            dtype=batch.backbone_coordinates.dtype,
        )
    else:
        bias = batch.logit_bias
    if batch.allowed_token_mask is not None:
        bias = bias + torch.where(
            batch.allowed_token_mask,
            torch.zeros((), dtype=bias.dtype, device=bias.device),
            torch.full((), -1e8, dtype=bias.dtype, device=bias.device),
        )
    return bias


def _feature_dict(
    features: dict,
    batch: LigandMpnnInput,
    *,
    batch_size: int,
    randn: Tensor,
    temperature: float = 0.1,
    score_all: bool = False,
    symmetry_residues: list[list[int]] | None = None,
    symmetry_weights: list[list[float]] | None = None,
) -> dict:
    r"""Assemble the upstream feature dict from featurize output and native controls."""

    feature_dict = dict(features)
    feature_dict.update(
        batch_size=batch_size,
        bias=_sampling_bias(batch),
        chain_mask=(batch.valid_mask if score_all else batch.design_mask).to(dtype=batch.backbone_coordinates.dtype),
        randn=randn,
        symmetry_residues=symmetry_residues if symmetry_residues is not None else [[]],
        symmetry_weights=symmetry_weights if symmetry_weights is not None else [[]],
        temperature=temperature,
    )
    return feature_dict


def _upstream_features(
    data_utils: ModuleType,
    pdb: Path,
    batch: LigandMpnnInput,
    device: torch.device,
    *,
    use_atom_context: bool,
    use_side_chain_context: bool,
) -> dict:
    r"""Parse and featurize the upstream input, pinning the design mask to the native batch."""

    source, *_ = data_utils.parse_PDB(
        str(pdb),
        device=str(device),
        parse_all_atoms=use_side_chain_context,
    )
    source["chain_mask"] = batch.design_mask[0].to(dtype=source["mask"].dtype)
    return data_utils.featurize(
        source,
        number_of_ligand_atoms=int(batch.context_coordinates.shape[2]),
        model_type="ligand_mpnn",
        use_atom_context=use_atom_context,
    )


# -- processor parity ----------------------------------------------------------


def _processor_checks(
    label: str,
    batch: LigandMpnnInput,
    features: dict,
    *,
    compare_side_chain: bool,
) -> None:
    r"""Cross-check the aligned tensor input against the upstream featurize output."""

    _check(f"{label}: LigandMpnnInput concrete type", type(batch) is LigandMpnnInput)
    _compare(f"{label}: backbone coordinates", batch.backbone_coordinates, features["X"])
    _compare(f"{label}: sequence tokens", batch.sequence_tokens, features["S"].long(), exact=True)
    _compare(f"{label}: valid mask", batch.valid_mask, features["mask"].bool(), exact=True)
    _compare(
        f"{label}: design mask",
        batch.design_mask,
        (features["chain_mask"] * features["mask"]).bool(),
        exact=True,
    )
    _compare(f"{label}: context coordinates", batch.context_coordinates, features["Y"])
    _compare(f"{label}: context atomic numbers", batch.context_atomic_numbers, features["Y_t"].long(), exact=True)
    _compare(f"{label}: context mask", batch.context_mask, features["Y_m"].bool(), exact=True)
    _compare(f"{label}: ligand interface mask", batch.ligand_interface_mask, features["mask_XY"].bool(), exact=True)
    _compare(
        f"{label}: residue offsets",
        batch.residue_indices[:, :, None] - batch.residue_indices[:, None, :],
        features["R_idx"][:, :, None] - features["R_idx"][:, None, :],
        exact=True,
    )
    _compare(
        f"{label}: same-chain equality",
        batch.chain_indices[:, :, None] == batch.chain_indices[:, None, :],
        features["chain_labels"][:, :, None] == features["chain_labels"][:, None, :],
        exact=True,
    )
    _check(
        f"{label}: flat non-None component references",
        len(batch.component_references) == batch.sequence_tokens.shape[1]
        and all(reference is not None for reference in batch.component_references),
    )
    _check(
        f"{label}: residue context slots [1, L, 32, 3] excluding CB",
        tuple(batch.residue_context_coordinates.shape) == (1, batch.sequence_tokens.shape[1], 32, 3)
        and tuple(batch.residue_context_mask.shape) == (1, batch.sequence_tokens.shape[1], 32),
    )
    if compare_side_chain:
        native_mask = batch.residue_context_mask.bool()
        upstream_mask = features["xyz_37"][:, :, 5:].shape == batch.residue_context_coordinates.shape
        _check(f"{label}: residue context aligns with atom37 slots 5:37", upstream_mask)
        _compare(
            f"{label}: residue context mask",
            native_mask,
            features["xyz_37_m"][:, :, 5:].bool(),
            exact=True,
        )
        _compare(
            f"{label}: masked residue context coordinates",
            batch.residue_context_coordinates * native_mask[..., None],
            features["xyz_37"][:, :, 5:] * features["xyz_37_m"][:, :, 5:].bool()[..., None],
        )


# -- encode / forward ----------------------------------------------------------


def _encode_checks(label: str, model: LigandMpnnModel, batch: LigandMpnnInput) -> None:
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
    again = model.encode(batch)[0]
    _compare(f"{label}: encode is deterministic at zero noise", again, node_states, exact=True)


def _forward_checks(
    label: str,
    model: LigandMpnnModel,
    upstream: torch.nn.Module,
    batch: LigandMpnnInput,
    features: dict,
    *,
    use_side_chain_context: bool,
    device: torch.device,
    seed: int,
) -> None:
    length = batch.sequence_tokens.shape[1]
    chain_mask = (batch.valid_mask & batch.design_mask).to(dtype=batch.backbone_coordinates.dtype)
    randn = torch.randn(1, length, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    ranks = _order_ranks(randn, chain_mask)

    output = model(batch, decoding_ranks=ranks, use_side_chain_context=use_side_chain_context)
    _check(f"{label}: forward returns MpnnOutput", type(output) is MpnnOutput)
    _check(
        f"{label}: forward logit and rank shapes",
        tuple(output.logits.shape) == (1, length, _VOCABULARY) and tuple(output.decoding_ranks.shape) == (1, length),
    )
    _compare(f"{label}: forward returns supplied ranks", output.decoding_ranks, ranks, exact=True)
    _compare(
        f"{label}: forward log-probability property",
        output.log_probabilities,
        output.logits.log_softmax(dim=-1),
        exact=True,
    )
    feature_dict = _feature_dict(features, batch, batch_size=1, randn=randn)
    upstream_log_probs = upstream.score(feature_dict, use_sequence=True)["log_probs"]
    _compare(f"{label}: forward vs upstream score(use_sequence=True)", output.log_probabilities, upstream_log_probs)


def _coordinate_noise_checks(
    label: str,
    model: LigandMpnnModel,
    upstream: torch.nn.Module,
    batch: LigandMpnnInput,
    features: dict,
    *,
    device: torch.device,
    seed: int,
) -> None:
    r"""One nonzero-noise forward: matched X-then-Y stream against upstream augment_eps."""

    length = batch.sequence_tokens.shape[1]
    noise = 0.1
    chain_mask = (batch.valid_mask & batch.design_mask).to(dtype=batch.backbone_coordinates.dtype)
    randn = torch.randn(1, length, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    ranks = _order_ranks(randn, chain_mask)

    clean = model(batch, decoding_ranks=ranks).logits
    noised = model(
        batch,
        decoding_ranks=ranks,
        coordinate_noise=noise,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).logits
    repeat = model(
        batch,
        decoding_ranks=ranks,
        coordinate_noise=noise,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).logits
    _compare(f"{label}: coordinate_noise is deterministic under a seeded generator", noised, repeat, exact=True)
    _check(
        f"{label}: coordinate_noise perturbs the logits",
        not torch.allclose(clean.float(), noised.float(), atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL),
    )

    feature_dict = _feature_dict(features, batch, batch_size=1, randn=randn)
    upstream.features.augment_eps = noise
    torch.manual_seed(seed)
    try:
        upstream_log_probs = upstream.score(feature_dict, use_sequence=True)["log_probs"]
    finally:
        upstream.features.augment_eps = 0.0
    _compare(
        f"{label}: coordinate_noise forward vs upstream augment_eps",
        noised.log_softmax(dim=-1),
        upstream_log_probs,
    )


# -- probability methods -------------------------------------------------------


def _probability_checks(
    label: str,
    model: LigandMpnnModel,
    upstream: torch.nn.Module,
    batch: LigandMpnnInput,
    features: dict,
    *,
    num_orders: int,
    use_side_chain_context: bool,
    device: torch.device,
    seed: int,
) -> None:
    length = batch.sequence_tokens.shape[1]
    valid = batch.valid_mask[0]
    chain_mask = (batch.valid_mask & batch.design_mask).to(dtype=batch.backbone_coordinates.dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    randn = torch.randn(num_orders, length, device=device, generator=generator)
    order_ranks = _order_ranks(randn, chain_mask)

    # Order-matched single-residue comparison (conditional / unconditional).
    #
    # The frozen contract makes ``conditional_probabilities`` move each target *exactly*
    # last and ``unconditional_probabilities`` move it *exactly* first, preserving every
    # other rank relation (DESIGN.md sec. 4.8 L3596 and L3613-3615). Upstream
    # ``single_aa_score`` has no explicit-order argument; it realizes "target last/first"
    # only through ``argsort((order_mask +/- 1) * |randn|)``. That realization silently
    # fails when the target's own |randn| is vanishingly small: with ``order_mask[t]=1``
    # the target key is ``1.0001 * |randn[t]|`` while every other key is
    # ``0.0001 * |randn[p]|``, so a target magnitude below ~1e-4 * max|randn| is overtaken
    # and the target is NOT placed last (symmetrically for the target-first branch). That
    # is an RNG-argsort placement artifact of the reference, not a property of the
    # probability math this parity tests. To compare only the probability computation,
    # drive BOTH sides from one shared, well-conditioned decode order: a random permutation
    # whose upstream magnitudes lie in ``[1, L]`` (so the +/-1 target boost dominates
    # ``0.0001 * max|randn|`` by ~100x and the target is genuinely last/first for every
    # row), with native's ``base_decoding_ranks`` derived from the SAME permutation. This
    # supplies the identical target-last/first order to upstream through its only order
    # control (``randn``) and to the model through its frozen ``base_decoding_ranks`` input
    # ("Supplying ranks skips only their random draw", DESIGN.md sec. 4.8 L3617). It is not
    # a reseed-to-dodge: any permutation is admissible because the magnitudes are
    # well-conditioned by construction. Deterministic target-last/first placement is
    # verified separately by the diagonal checks below.
    single_residue_order = torch.stack(
        [torch.randperm(length, device=device, generator=generator) for _ in range(num_orders)]
    )
    base_ranks = _decoding_order_to_ranks(single_residue_order)
    single_residue_randn = (base_ranks + 1).to(dtype=batch.backbone_coordinates.dtype)

    kwargs = {"use_side_chain_context": use_side_chain_context}

    # 1. native-sequence-conditioned autoregressive  <->  score(use_sequence=True)
    autoregressive = model.autoregressive_probabilities(
        batch,
        num_decoding_orders=num_orders,
        decoding_ranks=order_ranks,
        **kwargs,
    )
    _check(f"{label}: autoregressive result type", type(autoregressive) is MpnnProbabilityResult)
    _check(
        f"{label}: autoregressive logits [E, L, 21] and ranks [E, L]",
        tuple(autoregressive.logits.shape) == (num_orders, length, _VOCABULARY)
        and tuple(autoregressive.decoding_ranks.shape) == (num_orders, length),
    )
    _check(
        f"{label}: probability result carries the one target identity",
        autoregressive.target is batch.structure and autoregressive.component_references == batch.component_references,
    )
    upstream_autoregressive = upstream.score(
        _feature_dict(features, batch, batch_size=num_orders, randn=randn),
        use_sequence=True,
    )["log_probs"]
    _compare(
        f"{label}: autoregressive vs upstream score(use_sequence=True)",
        autoregressive.log_probabilities,
        upstream_autoregressive,
    )
    _compare(f"{label}: autoregressive returns supplied ranks", autoregressive.decoding_ranks, order_ranks, exact=True)

    # 2. order-dependent sequence-free autoregressive  <->  score(use_sequence=False)
    sequence_free = model.sequence_free_autoregressive_probabilities(
        batch,
        num_decoding_orders=num_orders,
        decoding_ranks=order_ranks,
        **kwargs,
    )
    _check(
        f"{label}: sequence-free logits [E, L, 21] and ranks [E, L]",
        tuple(sequence_free.logits.shape) == (num_orders, length, _VOCABULARY)
        and tuple(sequence_free.decoding_ranks.shape) == (num_orders, length),
    )
    upstream_sequence_free = upstream.score(
        _feature_dict(features, batch, batch_size=num_orders, randn=randn),
        use_sequence=False,
    )["log_probs"]
    _compare(
        f"{label}: sequence-free vs upstream score(use_sequence=False)",
        sequence_free.log_probabilities,
        upstream_sequence_free,
        atol=FLOAT32_FLOOR_ATOL,  # float32-floor path: see FLOAT32_FLOOR_ATOL rationale
    )

    # 3. conditional (target moved last)  <->  single_aa_score(use_sequence=False)
    conditional = model.conditional_probabilities(
        batch,
        num_decoding_orders=num_orders,
        base_decoding_ranks=base_ranks,
        **kwargs,
    )
    _check(
        f"{label}: conditional logits [E, L, 21] and effective ranks [E, L, L]",
        tuple(conditional.logits.shape) == (num_orders, length, _VOCABULARY)
        and tuple(conditional.decoding_ranks.shape) == (num_orders, length, length),
    )
    upstream_conditional = upstream.single_aa_score(
        _feature_dict(features, batch, batch_size=num_orders, randn=single_residue_randn),
        use_sequence=False,
    )["log_probs"]
    _compare(
        f"{label}: conditional vs upstream single_aa_score(use_sequence=False)",
        conditional.log_probabilities[:, valid],
        upstream_conditional[:, valid],
    )
    conditional_diagonal = conditional.decoding_ranks.diagonal(dim1=1, dim2=2)
    _check(
        f"{label}: conditional moves each target last",
        bool(torch.all(conditional_diagonal == conditional.decoding_ranks.amax(dim=-1))),
    )

    # 4. unconditional (target moved first)  <->  single_aa_score(use_sequence=True)
    unconditional = model.unconditional_probabilities(
        batch,
        num_decoding_orders=num_orders,
        base_decoding_ranks=base_ranks,
        **kwargs,
    )
    _check(
        f"{label}: unconditional logits [E, L, 21] and effective ranks [E, L, L]",
        tuple(unconditional.logits.shape) == (num_orders, length, _VOCABULARY)
        and tuple(unconditional.decoding_ranks.shape) == (num_orders, length, length),
    )
    upstream_unconditional = upstream.single_aa_score(
        _feature_dict(features, batch, batch_size=num_orders, randn=single_residue_randn),
        use_sequence=True,
    )["log_probs"]
    _compare(
        f"{label}: unconditional vs upstream single_aa_score(use_sequence=True)",
        unconditional.log_probabilities[:, valid],
        upstream_unconditional[:, valid],
    )
    unconditional_diagonal = unconditional.decoding_ranks.diagonal(dim1=1, dim2=2)
    _check(
        f"{label}: unconditional moves each target first",
        bool(torch.all(unconditional_diagonal == -1)),
    )


# -- sampling ------------------------------------------------------------------


def _sampling_compare(
    label: str,
    native: MpnnSamplingOutput,
    upstream: dict,
) -> None:
    _check(f"{label}: sample returns MpnnSamplingOutput", type(native) is MpnnSamplingOutput)
    _compare(f"{label}: sample tokens", native.sequences, upstream["S"], exact=True)
    _compare(
        f"{label}: sample sampling probabilities",
        native.sampling_probabilities,
        upstream["sampling_probs"],
        atol=FLOAT32_FLOOR_ATOL,  # float32-floor path: see FLOAT32_FLOOR_ATOL rationale
    )
    _compare(f"{label}: sample log probabilities", native.log_probabilities, upstream["log_probs"])
    _compare(
        f"{label}: sample decoding order",
        torch.argsort(native.decoding_ranks, dim=-1),
        upstream["decoding_order"],
        exact=True,
    )


def _sample_checks(
    label: str,
    model: LigandMpnnModel,
    upstream: torch.nn.Module,
    batch: LigandMpnnInput,
    features: dict,
    *,
    num_samples: int,
    device: torch.device,
    seed: int,
) -> None:
    length = batch.sequence_tokens.shape[1]
    temperature = 0.1
    chain_mask = (batch.valid_mask & batch.design_mask).to(dtype=batch.backbone_coordinates.dtype)
    randn = torch.randn(num_samples, length, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    ranks = _order_ranks(randn, chain_mask)

    native = model.sample(
        batch,
        num_samples=num_samples,
        temperature=temperature,
        decoding_ranks=ranks,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    torch.manual_seed(seed)
    expected = upstream.sample(
        _feature_dict(features, batch, batch_size=num_samples, randn=randn, temperature=temperature)
    )
    _sampling_compare(label, native, expected)
    _check(
        f"{label}: independent decoding orders are distinct",
        torch.unique(native.decoding_ranks, dim=0).shape[0] > 1,
    )
    _check(
        f"{label}: fixed rows retain zero sampling arrays",
        bool((native.sampling_probabilities[:, ~(batch.valid_mask & batch.design_mask)[0]] == 0).all()),
    )


def _controlled_sample_checks(
    label: str,
    model: LigandMpnnModel,
    processor: LigandMpnnProcessor,
    upstream: torch.nn.Module,
    data_utils: ModuleType,
    pdb: Path,
    *,
    num_samples: int,
    device: torch.device,
    seed: int,
) -> None:
    r"""Allowed-support and residue-bias sampling routed through the released bias axis."""

    base = processor(pdb)
    reference = next(
        component_reference
        for component_reference, valid in zip(base.component_references, base.valid_mask[0].tolist())
        if valid
    )
    residue_bias = torch.zeros(_GENERATABLE, device=device, dtype=base.backbone_coordinates.dtype)
    residue_bias[AMINO_ACID_TOKEN_BY_COMPONENT["ALA"]] = 1.5
    residue_bias[AMINO_ACID_TOKEN_BY_COMPONENT["GLY"]] = -0.5
    batch = processor(
        pdb,
        conditions=(AllowedResidueIdentities((reference,), frozenset({"ALA", "GLY", "SER"})),),
        residue_logit_bias={reference: residue_bias},
    )
    _check(
        f"{label}: allowed support and residue bias lower onto the aligned tensors",
        batch.allowed_token_mask is not None and batch.logit_bias is not None,
    )
    row = batch.component_references.index(reference)
    _check(
        f"{label}: unknown slot is inert in the public bias",
        float(batch.logit_bias[0, row, AMINO_ACID_UNKNOWN_TOKEN]) == 0.0,
    )
    length = batch.sequence_tokens.shape[1]
    temperature = 0.15
    chain_mask = (batch.valid_mask & batch.design_mask).to(dtype=batch.backbone_coordinates.dtype)
    randn = torch.randn(num_samples, length, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    ranks = _order_ranks(randn, chain_mask)

    features = _upstream_features(
        data_utils,
        pdb,
        batch,
        device,
        use_atom_context=True,
        use_side_chain_context=False,
    )
    native = model.sample(
        batch,
        num_samples=num_samples,
        temperature=temperature,
        decoding_ranks=ranks,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    torch.manual_seed(seed)
    expected = upstream.sample(
        _feature_dict(features, batch, batch_size=num_samples, randn=randn, temperature=temperature)
    )
    _sampling_compare(f"{label}: allowed+bias", native, expected)
    allowed = {AMINO_ACID_TOKEN_BY_COMPONENT[name] for name in ("ALA", "GLY", "SER")}
    disallowed = torch.tensor(
        [token for token in range(_GENERATABLE) if token not in allowed],
        device=device,
    )
    _check(
        f"{label}: hard support removes every omitted identity at the restricted row",
        bool((native.sampling_probabilities[:, row][:, disallowed] == 0).all()),
    )


def _tied_sample_checks(
    label: str,
    model: LigandMpnnModel,
    processor: LigandMpnnProcessor,
    upstream: torch.nn.Module,
    data_utils: ModuleType,
    pdb: Path,
    *,
    num_samples: int,
    device: torch.device,
    seed: int,
) -> None:
    r"""Weighted tied sampling: one group decision copied to members, matched to upstream."""

    base = processor(pdb)
    designable = [
        component_reference
        for component_reference, keep in zip(
            base.component_references,
            (base.valid_mask & base.design_mask)[0].tolist(),
        )
        if keep
    ]
    first, second = designable[0], designable[1]
    weights = (0.25, 0.75)
    batch = processor(
        pdb,
        conditions=(TiedResidueIdentities(groups=((first, second),)),),
        tied_weights={first: weights[0], second: weights[1]},
    )
    groups = tied_groups_from_ids(batch.tied_group_ids)
    _check(
        f"{label}: tie closure forms one group of two",
        len(groups) == 1 and len(groups[0]) == 2,
    )
    rows = list(groups[0])
    length = batch.sequence_tokens.shape[1]
    temperature = 0.2
    chain_mask = (batch.valid_mask & batch.design_mask).to(dtype=batch.backbone_coordinates.dtype)
    randn = torch.randn(num_samples, length, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    ranks = _order_ranks(randn, chain_mask)

    features = _upstream_features(
        data_utils,
        pdb,
        batch,
        device,
        use_atom_context=True,
        use_side_chain_context=False,
    )
    native = model.sample(
        batch,
        num_samples=num_samples,
        temperature=temperature,
        decoding_ranks=ranks,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    torch.manual_seed(seed)
    expected = upstream.sample(
        _feature_dict(
            features,
            batch,
            batch_size=num_samples,
            randn=randn,
            temperature=temperature,
            symmetry_residues=[rows],
            symmetry_weights=[list(weights)],
        )
    )
    _sampling_compare(f"{label}: weighted tie", native, expected)
    _check(
        f"{label}: tied members share the sampled identity",
        bool(torch.all(native.sequences[:, rows] == native.sequences[:, rows[:1]])),
    )
    group_ranks = native.decoding_ranks[:, rows]
    _check(
        f"{label}: tied members decode as one consecutive unit",
        bool(torch.all(group_ranks.amax(dim=-1) - group_ranks.amin(dim=-1) == len(rows) - 1)),
    )
    _compare(
        f"{label}: one group distribution copied to members",
        native.sampling_probabilities[:, rows],
        native.sampling_probabilities[:, rows[:1]].expand(-1, len(rows), -1),
    )
    # Raw log probabilities remain each member's individual unadjusted distribution.
    _check(
        f"{label}: members keep individual raw log-probability rows",
        not torch.allclose(
            native.log_probabilities[:, rows[0]].float(),
            native.log_probabilities[:, rows[1]].float(),
            atol=FLOAT32_ATOL,
            rtol=FLOAT32_RTOL,
        ),
    )


# -- design --------------------------------------------------------------------


def _candidate_tokens(candidate: DesignCandidate, references: tuple[ComponentReference, ...]) -> list[int]:
    return [
        AMINO_ACID_TOKEN_BY_COMPONENT[candidate.system.component(reference).component_id] for reference in references
    ]


def _design_checks(
    label: str,
    model: LigandMpnnModel,
    batch: LigandMpnnInput,
    *,
    num_candidates: int,
    device: torch.device,
    seed: int,
) -> None:
    references = batch.component_references
    designed = (batch.valid_mask & batch.design_mask)[0]
    interface = designed & batch.ligand_interface_mask[0].bool()
    designed_scope = frozenset(reference for reference, keep in zip(references, designed.tolist()) if keep)
    interface_scope = frozenset(reference for reference, keep in zip(references, interface.tolist()) if keep)

    result = model.design(
        batch,
        num_candidates=num_candidates,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    sample = model.sample(
        batch,
        num_samples=num_candidates,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    _check(f"{label}: design returns the shared DesignResult (no subtype)", type(result) is DesignResult)
    _check(f"{label}: candidate count", len(result.candidates) == num_candidates)
    expected_scores = {"designed_sequence_confidence", "ligand_interface_confidence", "sequence_recovery"}
    for index, candidate in enumerate(result.candidates):
        _check(f"{label}: candidate {index} DesignCandidate", type(candidate) is DesignCandidate)
        _check(
            f"{label}: candidate {index} score set and preference",
            set(candidate.scores) == expected_scores
            and all(score.preferred == "higher" and score.unit is None for score in candidate.scores.values()),
        )
        _check(
            f"{label}: candidate {index} confidence scopes",
            candidate.scores["designed_sequence_confidence"].scope == designed_scope
            and candidate.scores["sequence_recovery"].scope == designed_scope
            and candidate.scores["ligand_interface_confidence"].scope == interface_scope,
        )
        _compare(
            f"{label}: candidate {index} tokens equal the equal-seed sample",
            torch.tensor(_candidate_tokens(candidate, references), device=device),
            sample.sequences[index],
            exact=True,
        )
        _check(
            f"{label}: candidate {index} preserves instances and coordinate-free system",
            tuple(candidate.system.instances) == tuple(batch.structure.system.instances),
        )
    losses = -torch.gather(sample.log_probabilities, -1, sample.sequences.unsqueeze(-1)).squeeze(-1)
    designed_row = designed.to(dtype=losses.dtype)
    expected_confidence = torch.exp(-(losses * designed_row).sum(-1) / designed_row.sum(-1))
    actual_confidence = torch.tensor(
        [candidate.scores["designed_sequence_confidence"].value for candidate in result.candidates],
        device=device,
    )
    _compare(
        f"{label}: designed_sequence_confidence equals exp(-mean nll[D])",
        actual_confidence,
        expected_confidence,
    )


# -- pipeline and composition --------------------------------------------------


def _candidate_signature(candidate: DesignCandidate) -> tuple:
    system = candidate.system
    return (
        tuple(system.instances.items()),
        tuple(
            (instance_id, tuple(component.component_id for component in system.entities[entity_id].components))
            for instance_id, entity_id in system.instances.items()
        ),
        tuple(sorted((name, round(score.value, 6)) for name, score in candidate.scores.items())),
    )


def _pipeline_checks(
    label: str,
    model: LigandMpnnModel,
    processor: LigandMpnnProcessor,
    pdb: Path,
    batch: LigandMpnnInput,
    checkpoint_path: Path,
    *,
    num_orders: int,
    device: torch.device,
    seed: int,
) -> None:
    structure = read_pdb(pdb)
    pipeline = LigandMpnnPipeline(model=model, processor=processor)
    _check(
        f"{label}: LigandMpnnPipeline preserves its pair",
        pipeline.model is model and pipeline.processor is processor,
    )
    _check(
        f"{label}: no score / ScoreResult surface",
        not hasattr(model, "score") and not hasattr(pipeline, "score"),
    )

    pipeline_design = pipeline.design(structure, num_candidates=2, seed=seed)
    model_design = model.design(
        processor(structure),
        num_candidates=2,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    _check(
        f"{label}: Pipeline design delegates to Model design",
        tuple(_candidate_signature(candidate) for candidate in pipeline_design.candidates)
        == tuple(_candidate_signature(candidate) for candidate in model_design.candidates),
    )

    for method_name in (
        "autoregressive_probabilities",
        "sequence_free_autoregressive_probabilities",
        "conditional_probabilities",
        "unconditional_probabilities",
    ):
        pipeline_result = getattr(pipeline, method_name)(structure, num_decoding_orders=num_orders, seed=seed)
        model_result = getattr(model, method_name)(
            processor(structure),
            num_decoding_orders=num_orders,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        _check(f"{label}: Pipeline {method_name} result type", type(pipeline_result) is MpnnProbabilityResult)
        _compare(
            f"{label}: Pipeline {method_name} logits delegation",
            pipeline_result.logits,
            model_result.logits,
            exact=True,
        )
        _compare(
            f"{label}: Pipeline {method_name} rank delegation",
            pipeline_result.decoding_ranks,
            model_result.decoding_ranks,
            exact=True,
        )

    with tempfile.TemporaryDirectory(prefix="ofoldx-ligandmpnn-") as directory:
        repository = Path(directory)
        convert_ligandmpnn_checkpoint(checkpoint_path, repository)
        loaded_model = AutoModel(repository, device=device)
        loaded_processor = AutoProcessor(repository, device=device, dtype=processor.dtype)
        _check(
            f"{label}: AutoModel / AutoProcessor concrete types",
            type(loaded_model) is LigandMpnnModel and type(loaded_processor) is LigandMpnnProcessor,
        )
        _check(
            f"{label}: hydrated configs match",
            dict(loaded_model.config) == dict(model.config) and dict(loaded_processor.config) == dict(processor.config),
        )
        torch.testing.assert_close(loaded_model.state_dict(), model.state_dict(), rtol=0, atol=0)
        design_pipeline = load_pipeline("design", model=repository, device=device, dtype=processor.dtype)
        _check(
            f"{label}: pipeline('design') resolves the concrete LigandMpnnPipeline",
            type(design_pipeline) is LigandMpnnPipeline,
        )
        hydrated = design_pipeline.design(structure, num_candidates=2, seed=seed)
        _check(
            f"{label}: hydrated Pipeline design reproduces the live pair",
            tuple(_candidate_signature(candidate) for candidate in hydrated.candidates)
            == tuple(_candidate_signature(candidate) for candidate in pipeline_design.candidates),
        )


def _load_packer(checkpoint_path: Path, device: torch.device) -> MpnnPackerPipeline:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = checkpoint["model_state_dict"]
    model = MpnnPackerModel(
        MpnnPackerConfig(
            k_neighbors=int(checkpoint["num_edges"]),
            atom_context_num=int(checkpoint["atom_context_num"]),
        )
    )
    model.load_state_dict(convert_mpnn_packer_state_dict(state_dict), strict=True, assign=True)
    model = model.to(device).eval()
    return MpnnPackerPipeline(model=model, processor=MpnnPackerProcessor(device=device))


def _design_and_pack_checks(
    label: str,
    model: LigandMpnnModel,
    processor: LigandMpnnProcessor,
    packer_checkpoint: Path,
    pdb: Path,
    *,
    device: torch.device,
    seed: int,
) -> None:
    if not packer_checkpoint.exists():
        print(f"[skip] {label}: packer checkpoint {packer_checkpoint} is absent")
        return
    designer = LigandMpnnPipeline(model=model, processor=processor)
    packer = _load_packer(packer_checkpoint, device)
    composed = LigandMpnnDesignAndPackPipeline(designer=designer, packer=packer)
    _check(
        f"{label}: composition preserves its child Pipelines",
        composed.designer is designer and composed.packer is packer,
    )

    structure = read_pdb(pdb)
    num_candidates = 2
    num_packs = 2
    design_result, packs = composed.design(
        structure,
        num_candidates=num_candidates,
        num_packs_per_candidate=num_packs,
        num_denoising_steps=2,
        num_torsion_samples=4,
        seed=seed,
    )
    _check(
        f"{label}: design-and-pack returns a DesignResult and per-candidate packs",
        type(design_result) is DesignResult
        and isinstance(packs, tuple)
        and len(design_result.candidates) == num_candidates
        and len(packs) == num_candidates,
    )
    _check(
        f"{label}: each candidate owns num_packs_per_candidate predictions",
        all(type(pack) is PackResult and len(pack.predictions) == num_packs for pack in packs)
        and all(type(prediction) is PackPrediction for pack in packs for prediction in pack.predictions),
    )


# -- absence checks ------------------------------------------------------------


def _absence_checks(model: LigandMpnnModel) -> None:
    _check(
        "surface: no score / Backbone / prepared-input surface",
        not hasattr(model, "score")
        and not hasattr(model, "backbone")
        and not hasattr(LigandMpnnModel, "score")
        and not hasattr(LigandMpnnInput, "layout"),
    )
    _check(
        "surface: LigandMpnnConfig holds one width and no false axes",
        set(dict(LigandMpnnConfig()))
        == {"hidden_size", "num_encoder_layers", "num_decoder_layers", "num_neighbors", "dropout"},
    )
    _check(
        "surface: LigandMpnnProcessorConfig holds only the context extent",
        set(dict(LigandMpnnProcessorConfig())) == {"num_context_atoms"},
    )


# -- entry point ---------------------------------------------------------------


def main() -> None:
    reference = reference_root("LigandMPNN")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=reference)
    parser.add_argument("--checkpoint-dir", type=Path, default=weights_root() / "mpnn-weights")
    parser.add_argument("--checkpoint", type=Path, help="Run a single checkpoint instead of all four.")
    parser.add_argument("--context", choices=("all", *(name for name, *_ in CONTEXTS)), default="all")
    parser.add_argument("--pdb", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-orders", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    if args.num_orders < 2:
        parser.error("--num-orders must be at least 2 to exercise independent decoding orders")

    pdb = args.pdb if args.pdb is not None else args.reference / "inputs" / "1BC8.pdb"
    data_utils, model_utils, revision, dirty = _load_reference(args.reference)
    device = torch.device(args.device)
    structure = read_pdb(pdb)

    contexts = CONTEXTS if args.context == "all" else tuple(item for item in CONTEXTS if item[0] == args.context)
    checkpoints = (
        (args.checkpoint,) if args.checkpoint is not None else tuple(args.checkpoint_dir / name for name in CHECKPOINTS)
    )
    packer_checkpoint = args.checkpoint_dir / PACKER_CHECKPOINT

    print(f"[source] dauparas/LigandMPNN checkout={args.reference} revision={revision} dirty={dirty}")
    print(f"[fixture] {pdb}")

    _run_suite("surface:absence", _absence_checks, LigandMpnnModel(LigandMpnnConfig()))

    for checkpoint_path in checkpoints:
        if not checkpoint_path.is_file():
            print(f"[ERROR] missing checkpoint {checkpoint_path}")
            SCORE.record(f"{checkpoint_path.name}: checkpoint present", False)
            continue
        representative = checkpoint_path.name == REPRESENTATIVE
        for context_name, use_atom_context, use_side_chain_context in contexts:
            label = f"{checkpoint_path.name}:{context_name}"
            try:
                model, processor, upstream = _load_models(
                    label,
                    checkpoint_path,
                    model_utils,
                    device,
                    use_side_chain_context=use_side_chain_context,
                )
            except Exception as exc:  # noqa: BLE001 - record load failure and keep going
                print(f"[ERROR] {label}: model load raised {type(exc).__name__}: {exc}")
                traceback.print_exc()
                SCORE.record(f"{label}: model load", False)
                continue
            batch = None
            try:
                with torch.inference_mode():
                    batch = processor(structure, use_atom_context=use_atom_context)
                    features = _upstream_features(
                        data_utils,
                        pdb,
                        batch,
                        device,
                        use_atom_context=use_atom_context,
                        use_side_chain_context=use_side_chain_context,
                    )
                    _run_suite(
                        f"{label}:processor",
                        _processor_checks,
                        label,
                        batch,
                        features,
                        compare_side_chain=use_side_chain_context,
                    )
                    _run_suite(
                        f"{label}:forward",
                        _forward_checks,
                        label,
                        model,
                        upstream,
                        batch,
                        features,
                        use_side_chain_context=use_side_chain_context,
                        device=device,
                        seed=args.seed,
                    )
                    _run_suite(
                        f"{label}:probabilities",
                        _probability_checks,
                        label,
                        model,
                        upstream,
                        batch,
                        features,
                        num_orders=args.num_orders,
                        use_side_chain_context=use_side_chain_context,
                        device=device,
                        seed=args.seed + 1,
                    )
                    # The context branches (atom_disabled, side_chain) are validated
                    # numerically by the context-aware forward and four probability
                    # methods above. The sampler and designer share that one encoder, so
                    # their surface is exercised once on the atom_enabled context.
                    if representative and context_name == "atom_enabled":
                        _run_suite(f"{label}:encode", _encode_checks, label, model, batch)
                        _run_suite(
                            f"{label}:coordinate_noise",
                            _coordinate_noise_checks,
                            label,
                            model,
                            upstream,
                            batch,
                            features,
                            device=device,
                            seed=args.seed + 2,
                        )
                        _run_suite(
                            f"{label}:sample",
                            _sample_checks,
                            label,
                            model,
                            upstream,
                            batch,
                            features,
                            num_samples=args.num_orders,
                            device=device,
                            seed=args.seed + 3,
                        )
                        _run_suite(
                            f"{label}:design",
                            _design_checks,
                            label,
                            model,
                            batch,
                            num_candidates=args.num_orders,
                            device=device,
                            seed=args.seed + 4,
                        )
                        _run_suite(
                            f"{label}:controlled_sample",
                            _controlled_sample_checks,
                            label,
                            model,
                            processor,
                            upstream,
                            data_utils,
                            pdb,
                            num_samples=args.num_orders,
                            device=device,
                            seed=args.seed + 5,
                        )
                        _run_suite(
                            f"{label}:tied_sample",
                            _tied_sample_checks,
                            label,
                            model,
                            processor,
                            upstream,
                            data_utils,
                            pdb,
                            num_samples=args.num_orders,
                            device=device,
                            seed=args.seed + 6,
                        )
            except Exception as exc:  # noqa: BLE001 - record prerequisite failure and keep going
                print(f"[ERROR] {label}: context prerequisites raised {type(exc).__name__}: {exc}")
                traceback.print_exc()
                SCORE.record(f"{label}: context prerequisites", False)
                continue
            if representative and context_name == "atom_enabled":
                _run_suite(
                    f"{label}:pipeline",
                    _pipeline_checks,
                    label,
                    model,
                    processor,
                    pdb,
                    batch,
                    checkpoint_path,
                    num_orders=args.num_orders,
                    device=device,
                    seed=args.seed + 7,
                )
                _run_suite(
                    f"{label}:design_and_pack",
                    _design_and_pack_checks,
                    label,
                    model,
                    processor,
                    packer_checkpoint,
                    pdb,
                    device=device,
                    seed=args.seed + 8,
                )

    print("\n[scorecard] ============================================================")
    print(
        f"[scorecard] PASS={SCORE.passed} FAIL={SCORE.failed} "
        f"max_abs_deviation={SCORE.max_deviation:.3e} "
        f"tolerance(atol={FLOAT32_ATOL:.1e}, rtol={FLOAT32_RTOL:.1e})"
    )
    if SCORE.failures:
        print(f"[scorecard] {SCORE.failed} FAILED checks:")
        for failure in SCORE.failures:
            print(f"[scorecard][FAIL] {failure}")
        print("[summary] FAIL LigandMpnn frozen-surface diagnostic")
        sys.exit(1)
    print("[summary] PASS complete LigandMpnn frozen-surface diagnostic")


if __name__ == "__main__":
    main()
