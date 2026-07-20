r"""Upstream parity for the BoltzGen extend-an-existing-input lowering (frozen §3.7 X1-A + §3.7.2).

Covers the three Processor gates that used to raise ``NotImplementedError``:

* ``total_len`` / ``BoltzGenPlacements`` extension of an EXISTING input system (carried source rows
  + minted non-colliding ``chain_{n}`` generated rows, in released ``concatenate`` order), and
* ``BoltzGenStructureGroup`` -> ``token_distance_mask`` (§3.7.2).

The reference oracle is the RELEASED BoltzGen data pipeline itself (``OFOLDX_REFERENCE_BOLTZGEN``):
we build the exact released protein-binder structure
(``Structure.concatenate(empty_protein(target), empty_protein(binder))``), set ``design_mask`` /
``structure_group`` exactly as ``task/predict/data_protein_binder.py`` does, and run the released
``Tokenizer`` + ``featurizer.process_token_features``. We then lower the equivalent request through
``ofoldx.models.boltzgen.BoltzGenProcessor`` and compare, channel by channel:

  design_mask        (per token)      -- which rows are generated/designed
  structure ordering (per token)      -- source carried first, generated appended
  token_distance_mask (N x N)         -- the §3.7.2 structure-group channel

No neural weights are needed: this validates the DATA LOWERING against the released featurizer code,
which is what the frozen §3.7 X1-A / §3.7.2 contract governs. Numbers are exact (integer/boolean
tensors), so parity is reported as an exact max-abs difference of 0.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from scripts.parity.boltzgen import reference


def _stub_optional_featurizer_deps() -> None:
    """Stub ``pydssp`` (secondary-structure assignment) so importing the featurizer does not pull
    an optional dependency the token-feature stage never uses. ``process_token_features`` /
    ``process_atom_features`` never call ``pydssp`` — it is only used by the training SS-labeling
    path — so a bare module placeholder is exact for this parity.
    """

    import sys  # noqa: PLC0415
    import types as _types  # noqa: PLC0415

    if "pydssp" not in sys.modules:
        sys.modules["pydssp"] = _types.ModuleType("pydssp")


def build_reference_token_features(target_len: int, binder_len: int, *, structure_condition: bool):
    """Run the RELEASED tokenizer + process_token_features on the released binder structure.

    Returns (design_mask[N] int, structure_group[N] int, token_distance_mask[N,N] float, res_names[N]).
    """

    reference.configure_reference()
    _stub_optional_featurizer_deps()
    from boltzgen.data import const  # noqa: PLC0415
    from boltzgen.data.data import Input, Structure  # noqa: PLC0415
    from boltzgen.data.feature.featurizer import process_token_features  # noqa: PLC0415
    from boltzgen.data.tokenize.tokenizer import Tokenizer  # noqa: PLC0415

    # Released protein-binder construction (task/predict/data_protein_binder.py):
    #   str_native  -> existing target (carried verbatim)
    #   str_prot    -> Structure.empty_protein(seq_len) generated binder chain
    #   structure = Structure.concatenate(str_native, str_prot)   # target first, binder appended
    str_native = Structure.empty_protein(seq_len=target_len)
    str_prot = Structure.empty_protein(seq_len=binder_len)
    structure = Structure.concatenate(str_native, str_prot)

    tokenized = Tokenizer().tokenize(structure)
    tokens = tokenized.tokens

    # "Set the last chain to be designed" (verbatim from data_protein_binder.py).
    tokens["design_mask"] = (tokens["asym_id"] == tokens["asym_id"][-1]).astype(tokens["asym_id"].dtype)
    if structure_condition:
        # target_structure_condition: condition on the whole target (all non-design tokens) -> group 1.
        tokens["structure_group"][~tokens["design_mask"].astype(bool)] = 1

    input_data = Input(
        tokens=tokens,
        bonds=tokenized.bonds,
        token_to_res=tokenized.token_to_res,
        structure=structure,
        msa={},
        templates=None,
    )
    feats = process_token_features(input_data, np.random.default_rng(0), override_method="X-RAY DIFFRACTION")
    design_mask = feats["design_mask"].long().cpu().numpy()
    structure_group = feats["structure_group"].long().cpu().numpy()
    token_distance_mask = feats["token_distance_mask"].float().cpu().numpy()
    res_names = tokens["res_name"].tolist()
    _ = const
    return design_mask, structure_group, token_distance_mask, res_names


def build_ofoldx_token_features(target_len: int, binder_len: int, *, structure_condition: bool):
    """Lower the equivalent extend-existing request through the ofoldx BoltzGenProcessor."""

    from ofoldx.data.molecular import MolecularSystem, PolymerSequence  # noqa: PLC0415
    from ofoldx.models.boltzgen import BoltzGenProcessor  # noqa: PLC0415
    from ofoldx.models.boltzgen.model import BoltzGenStructureGroup  # noqa: PLC0415

    # Existing target: one GLY protein chain of `target_len` (matches Structure.empty_protein).
    target = PolymerSequence.from_one_letter("G" * target_len, polymer_type="protein")
    system = MolecularSystem({"target_entity": target}, {"A": "target_entity"})

    conditions = ()
    if structure_condition:
        members = tuple(("A", index) for index in range(target_len))
        conditions = (BoltzGenStructureGroup(members=members, group_id=1),)

    batch = BoltzGenProcessor()(system, total_len=binder_len, conditions=conditions)
    design_mask = batch.design_mask[0].long().cpu().numpy()
    token_distance_mask = batch.token_distance_mask[0].float().cpu().numpy()
    instance_ids = batch.layout.output_instance_ids
    return design_mask, token_distance_mask, instance_ids


def reference_token_distance_mask(structure_group: list[int]) -> np.ndarray:
    """Run the RELEASED process_token_features on a length-N structure with an OVERRIDDEN
    per-token structure_group vector, returning its token_distance_mask. This exercises every
    branch of the released algorithm (all-group-1 efficient path, multi-group, all-free)."""

    reference.configure_reference()
    _stub_optional_featurizer_deps()
    from boltzgen.data.data import Input, Structure  # noqa: PLC0415
    from boltzgen.data.feature.featurizer import process_token_features  # noqa: PLC0415
    from boltzgen.data.tokenize.tokenizer import Tokenizer  # noqa: PLC0415

    structure = Structure.empty_protein(seq_len=len(structure_group))
    tokenized = Tokenizer().tokenize(structure)
    tokens = tokenized.tokens
    tokens["structure_group"] = np.asarray(structure_group, dtype=tokens["structure_group"].dtype)
    input_data = Input(
        tokens=tokens,
        bonds=tokenized.bonds,
        token_to_res=tokenized.token_to_res,
        structure=structure,
        msa={},
        templates=None,
    )
    feats = process_token_features(input_data, np.random.default_rng(0), override_method="X-RAY DIFFRACTION")
    return feats["token_distance_mask"].float().cpu().numpy()


def run_algorithm_parity() -> bool:
    """Direct token_distance_mask algorithm parity across representative structure_group vectors."""

    from ofoldx.models.boltzgen.processor import _token_distance_mask  # noqa: PLC0415

    vectors = {
        "all_free_0": [0, 0, 0, 0],
        "all_group_1 (efficient branch)": [1, 1, 1, 1, 1],
        "single_group_1": [1, 1, 0, 0, 1],
        "two_groups": [1, 1, 2, 2, 0, 1],
        "high_group_ids": [0, 3, 3, 5, 5, 5, 0],
    }
    print("=== token_distance_mask algorithm parity (released process_token_features) ===")
    ok = True
    for name, vector in vectors.items():
        ref = reference_token_distance_mask(vector)
        ofo = _token_distance_mask(torch.tensor(vector, dtype=torch.long)).cpu().numpy()
        diff = float(np.abs(ref - ofo).max())
        status = "PASS" if diff == 0.0 else "FAIL"
        print(f"  {name:34s} ref_sum={ref.sum():5.0f}  max_abs_diff={diff:.3e}  {status}")
        ok = ok and diff == 0.0
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_case(target_len: int, binder_len: int, *, structure_condition: bool) -> bool:
    title = f"target_len={target_len} binder_len={binder_len} " f"structure_condition={structure_condition}"
    print(f"=== {title} ===")

    ref_design, ref_group, ref_tdm, ref_names = build_reference_token_features(
        target_len, binder_len, structure_condition=structure_condition
    )
    ofo_design, ofo_tdm, ofo_ids = build_ofoldx_token_features(
        target_len, binder_len, structure_condition=structure_condition
    )

    n = target_len + binder_len
    assert ref_tdm.shape == (n, n), ref_tdm.shape
    assert ofo_tdm.shape == (n, n), ofo_tdm.shape

    design_diff = int(np.abs(ref_design - ofo_design).max())
    tdm_diff = float(np.abs(ref_tdm - ofo_tdm).max())

    # Ordering: released carries the target (first target_len rows) then the appended binder.
    ordering_ok = (
        all(i == "A" for i in ofo_ids[:target_len])
        and len(set(ofo_ids[target_len:])) == 1
        and ofo_ids[target_len] not in ("A",)
    )

    print(f"  ref design_mask sum={int(ref_design.sum())}  ofoldx design_mask sum={int(ofo_design.sum())}")
    print(f"  ref structure_group unique={sorted(set(ref_group.tolist()))}")
    print(f"  token_distance_mask: ref sum={ref_tdm.sum():.0f} ofoldx sum={ofo_tdm.sum():.0f}")
    print(f"  [design_mask]         max_abs_diff = {design_diff}")
    print(f"  [token_distance_mask] max_abs_diff = {tdm_diff:.3e}")
    print(f"  [ordering] source-first / generated-appended non-colliding id: {'OK' if ordering_ok else 'FAIL'}")
    print(f"  generated instance id = {ofo_ids[target_len]!r} (source id = 'A')")

    ok = design_diff == 0 and tdm_diff == 0.0 and ordering_ok
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-len", type=int, default=12)
    parser.add_argument("--binder-len", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    results = [
        run_algorithm_parity(),
        # No structure conditioning: token_distance_mask is all-zero on both sides (design-only).
        run_case(args.target_len, args.binder_len, structure_condition=False),
        # target_structure_condition: whole target -> group 1 -> intra-target token_distance_mask block.
        run_case(args.target_len, args.binder_len, structure_condition=True),
        # Small + asymmetric shapes.
        run_case(5, 3, structure_condition=True),
        run_case(1, 20, structure_condition=True),
    ]
    print()
    print(f"[summary] {sum(results)}/{len(results)} cases PASS")
    if not all(results):
        raise SystemExit(1)
    print("ALL_OK")


if __name__ == "__main__":
    main()
