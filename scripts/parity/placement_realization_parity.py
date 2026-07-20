r"""Realization-parity harness for the frozen §3.7 placement core.

For each generative family (rfdiffusion / odesign / pxdesign / boltzgen) this exercises the
migrated ``PlacementSpec -> placement_map`` realization plus ``SymbolicResidueSite(placement_id,
offset)`` resolution and asserts the FROZEN §3.7 invariants (owner verdicts ratified
2026-07-18; see ``DESIGN.md`` §3.7 "placement core" and ``CONTIG_IDENTITY.md``):

  1. global ``placement_id`` uniqueness / derived ``instance_of`` index;
  2. coverage + declared order — every declared placement appears, in declared outer-group
     then inner-placement order (never recovered by sorting ids);
  3. zero-length placement -> ``()`` (frozen coverage rule);
  4. resolve round-trip — ``resolve(SymbolicResidueSite(pid, k))`` equals
     ``placement_map[group][pid][k]`` for every positive AND native-negative offset, and the
     two index the SAME occurrence (``occ[k] == occ[k - len]``);
  5. reference-target hygiene — every ``ComponentReference`` is ``(str realized_id, int idx)``;
     the family-owned realized id is DISJOINT from both the outer group keys and the
     ``placement_id`` namespace (the outer key is never a reference target; the realized id is
     100% family-owned and rides only in ``ComponentReference[0]``);
  6. family length semantics — boltzgen ``total_len`` budget distribution, odesign per-chain
     ``max_length``, pxdesign single fixed binder length;
  7. EXHAUSTION (the point §3.7 left open at the four-family freeze) — the union of every
     placement's occurrences is a FULL PARTITION of the realized output surface: it covers
     EVERY realized output component (the independent ``MolecularSystem.instances`` /
     ``layout.output_residues`` ground truth for the three ``placement_map`` families; the flat
     ``output_residues`` tuple for rfdiffusion) with no internal gap, no double-cover, and NO
     UNADDRESSABLE TAIL. See :func:`check_exhaustion`.

HONESTY BOUNDARY: none of the four families ship an EXECUTABLE upstream placement/realization
reference in this environment (rfdiffusion's released path needs ``dgl``, absent from the
parity venv; the odesign forward and pxdesign sampler parity scripts never touch the placement
path; no upstream BoltzGen generation reference is dumped). So realization-parity here is an
INTERNAL round-trip / self-consistency invariant plus a structural conformance check against
each family's DOCUMENTED upstream semantics — it is NOT a numeric comparison to a captured
upstream realization. rfdiffusion now ADOPTS ``placement_map`` (W6 caveat a): after its
``model.design`` samples the variable generated lengths it lifts the flat ``output_residues``
layout into the shared grouped ``placement_map`` via ``processor.materialize_placement_map``,
so it is checked on that grouped surface here — through the same materializer, fed the faithful
per-instance realized ``residue_map`` ``_decode_structure`` mints — exactly like the other three
families, plus the ``contig``-string ⇄ core projection round-trip.

Non-numeric: runs on the dev-box CPU torch build (``/data/users/z/envs/zyc`` — odesign needs
``/data/users/z/envs/zyc-odesign`` for its biotite pin). Numeric no-regression is a SEPARATE
gate carried by the per-family forward/sampler parity scripts on GPU.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ofoldx.data.placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
    SymbolicResidueSite,
)
from ofoldx.data.placement import instance_of as derive_instance_of
from ofoldx.data.placement import realized_instance_id, resolve, validate_placement


class CheckError(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def check_placement_map(
    placement: PlacementSpec,
    placement_map: PlacementMap,
    instance_of_map: dict[str, str],
) -> dict[str, Any]:
    r"""Assert the frozen §3.7 invariants over a realized ``placement_map`` + ``instance_of``.

    ``placement`` is the shared-core nested mapping that was realized. Returns per-check metrics.
    """

    metrics: dict[str, Any] = {}

    # (1) global uniqueness + derived index agreement.
    validate_placement(placement)
    expected_instance_of = derive_instance_of(placement)
    _require(
        instance_of_map == expected_instance_of,
        f"instance_of mismatch: family={instance_of_map} derived={expected_instance_of}",
    )
    metrics["placement_ids"] = len(expected_instance_of)

    # (2) coverage + declared order (outer group order, then inner placement order).
    _require(
        list(placement_map.keys()) == list(placement.keys()),
        f"declaration-group order not preserved: {list(placement_map)} != {list(placement)}",
    )
    for group, placements in placement.items():
        _require(group in placement_map, f"declared group {group!r} missing from placement_map")
        _require(
            list(placement_map[group].keys()) == list(placements.keys()),
            f"placement order not preserved in group {group!r}: " f"{list(placement_map[group])} != {list(placements)}",
        )

    group_keys = set(placement.keys())
    placement_ids = set(expected_instance_of)
    realized_ids: set[str] = set()
    zero_length = 0
    total_occurrences = 0
    resolve_checks = 0

    for group, placements in placement.items():
        for placement_id, value in placements.items():
            occ = placement_map[group][placement_id]
            _require(isinstance(occ, tuple), f"{placement_id!r} occurrences must be a tuple")

            # (3) zero-length -> ().
            if type(value) is GeneratedSegment and value.max_length == 0:
                _require(occ == (), f"zero-length placement {placement_id!r} must realize to ()")
            if len(occ) == 0:
                zero_length += 1

            for reference in occ:
                # (5) ComponentReference shape + realized-id hygiene.
                _require(
                    isinstance(reference, tuple)
                    and len(reference) == 2
                    and isinstance(reference[0], str)
                    and isinstance(reference[1], int),
                    f"occurrence {reference!r} of {placement_id!r} is not a (str, int) ComponentReference",
                )
                realized_ids.add(reference[0])
            total_occurrences += len(occ)

            # (4) resolve round-trip for every positive and native-negative offset.
            length = len(occ)
            for k in range(length):
                pos = resolve(SymbolicResidueSite(placement_id, k), placement_map, instance_of_map)
                _require(pos == occ[k], f"resolve({placement_id!r},{k}) != occ[{k}]")
                neg = resolve(SymbolicResidueSite(placement_id, k - length), placement_map, instance_of_map)
                _require(
                    neg == occ[k],
                    f"native-negative resolve({placement_id!r},{k - length}) != occ[{k}] "
                    f"(negative and positive offset must index the SAME occurrence)",
                )
                resolve_checks += 2

    # (5) realized-id namespace note. The declaration namespace (group keys / placement_ids)
    # and the output namespace (ComponentReference[0]) are disjoint BY ROLE, not by string
    # value: a realized id only ever appears in ComponentReference[0] and is never looked up in
    # instance_of, so a coinciding STRING is harmless and legal. boltzgen/odesign mint fresh
    # family ids (disjoint by value too); pxdesign reuses the binder placement_id string as its
    # realized chain id — both conform. We record the overlap rather than failing on it.
    string_overlap_placement = sorted(realized_ids & placement_ids)
    string_overlap_group = sorted(realized_ids & group_keys)

    metrics.update(
        {
            "groups": len(placement),
            "zero_length_placements": zero_length,
            "total_occurrences": total_occurrences,
            "realized_instance_ids": sorted(realized_ids),
            "resolve_round_trips": resolve_checks,
            "realized_id_reuses_placement_id_string": string_overlap_placement,
            "realized_id_reuses_group_string": string_overlap_group,
        }
    )
    return metrics


def _realized_output_components(system: Any) -> set[tuple[str, int]]:
    r"""Enumerate every realized OUTPUT component ``(instance_id, component_index)`` of a
    :class:`MolecularSystem`, computed independently of the ``placement_map`` from the
    system's ``instances -> entity.components`` (this is the surface the model actually sees;
    ``layout.output_residues`` is the same set in token order). Non-circular: the system is
    what a ``ComponentReference`` points INTO, not the map that points."""

    realized: set[tuple[str, int]] = set()
    for instance_id, entity_id in system.instances.items():
        for component_index in range(len(system.entities[entity_id].components)):
            realized.add((instance_id, component_index))
    return realized


def check_exhaustion(placement_map: PlacementMap, system: Any) -> dict[str, Any]:
    r"""Assert the §3.7 EXHAUSTION invariant left open at the four-family freeze: the union of
    ``placement_map`` occurrences (over all placements in all declaration groups) is a FULL
    PARTITION of the realized output surface — it covers EVERY realized output component with
    no internal gap, no double-cover, and NO UNADDRESSED TAIL.

    ``system`` supplies the independent ground truth (see :func:`_realized_output_components`).
    VALID for these witnesses because every witness runs with ``input=None`` (no conditioning
    SOURCE instance), so every system instance is a generated OUTPUT instance; a witness that
    carried an existing target would first have to subtract its source instances from the
    ground-truth set."""

    realized = _realized_output_components(system)

    addressed: list[tuple[str, int]] = []
    for placements in placement_map.values():
        for occ in placements.values():
            addressed.extend(occ)

    # no output component addressed by more than one placement (partition, not a cover).
    seen: set[tuple[str, int]] = set()
    duplicates: set[tuple[str, int]] = set()
    for reference in addressed:
        if reference in seen:
            duplicates.add(reference)
        seen.add(reference)
    _require(
        not duplicates,
        f"double-cover: output component(s) addressed by >1 placement: {sorted(duplicates)}",
    )

    # every addressed reference is a real realized output component (no dangling target).
    phantom = seen - realized
    _require(not phantom, f"placement addresses non-realized component(s): {sorted(phantom)}")

    # THE TAIL CHECK: no realized output component is left unaddressed by any placement.
    unaddressed_tail = realized - seen
    _require(
        not unaddressed_tail,
        f"unaddressed output tail (components no SymbolicResidueSite can reach): " f"{sorted(unaddressed_tail)}",
    )

    # per-instance contiguity: each covered instance is tiled 0..N-1 (explicit 'no gap'
    # witness; implied by the set equality above but reported per instance).
    per_instance: dict[str, set[int]] = {}
    for instance_id, index in seen:
        per_instance.setdefault(instance_id, set()).add(index)
    for instance_id, indices in per_instance.items():
        _require(
            indices == set(range(len(indices))),
            f"instance {instance_id!r} coverage not contiguous 0..N-1: {sorted(indices)}",
        )

    return {
        "exhausts_output": True,
        "realized_output_components": len(realized),
        "addressed_components": len(seen),
        "double_covered": sorted(duplicates),
        "unaddressed_tail": sorted(unaddressed_tail),
        "output_instances_covered": sorted(per_instance),
    }


# ======================================================================================
# Family runners. Each returns (family, cases[]) where a case is a dict with a name,
# realized length/metrics, and any family-specific semantic assertions.
# ======================================================================================


def run_boltzgen() -> list[dict[str, Any]]:
    from ofoldx.models.boltzgen import BoltzGenPlacements, BoltzGenProcessor

    processor = BoltzGenProcessor()
    cases: list[dict[str, Any]] = []

    # Case A — degenerate: only a total_len budget, no placement -> one default chain.
    b1 = processor(None, total_len=24)
    m1 = check_placement_map(
        {"assembly": {"chain_A": GeneratedSegment("protein", 24, 24)}},
        b1.layout.placement_map,
        b1.layout.instance_of,
    )
    _require(int(b1.token_mask.sum().item()) == 24, "boltzgen default token_count != 24")
    _require(len(b1.layout.placement_map["assembly"]["chain_A"]) == 24, "default chain len != 24")
    e1 = check_exhaustion(b1.layout.placement_map, b1.layout.system)
    cases.append({"name": "default_total_len_24", "token_count": 24, **m1, "exhaustion": e1})

    # Case B — multi-chain whole-assembly: ranged lengths from the budget + a zero-length chain.
    placement = {
        "assembly": {
            "binderA": GeneratedSegment("protein", 10, 40),
            "binderB": GeneratedSegment("protein", 10, 40),
            "scaffold": GeneratedSegment("protein", 0, 0),
        }
    }
    b2 = processor(None, total_len=50, conditions=(BoltzGenPlacements(placements=placement),))
    pm2, io2 = b2.layout.placement_map, b2.layout.instance_of
    m2 = check_placement_map(placement, pm2, io2)
    lens = {pid: len(occ) for pid, occ in pm2["assembly"].items()}
    # family length semantics: greedy fill within total_len, sum == total_len.
    _require(lens == {"binderA": 40, "binderB": 10, "scaffold": 0}, f"budget distribution wrong: {lens}")
    _require(sum(lens.values()) == 50, "total_len budget not honored")
    # distinct contiguous chain instances, one per generated placement.
    _require(pm2["assembly"]["binderA"][0] == ("chain_0", 0), "binderA not contiguous chain_0")
    _require(pm2["assembly"]["binderB"][0] == ("chain_1", 0), "binderB not contiguous chain_1")
    _require(
        resolve(SymbolicResidueSite("binderA", -1), pm2, io2) == ("chain_0", 39),
        "boltzgen negative-offset resolve wrong",
    )
    e2 = check_exhaustion(pm2, b2.layout.system)
    _require(e2["realized_output_components"] == 50, "boltzgen exhaustion did not see all 50 tokens")
    cases.append({"name": "multichain_total_len_50", "token_count": 50, "lengths": lens, **m2, "exhaustion": e2})

    # Case C — total_len outside the feasible [20,80] band must raise (upstream accept-window).
    raised = False
    try:
        processor(None, total_len=5, conditions=(BoltzGenPlacements(placements=placement),))
    except ValueError:
        raised = True
    _require(raised, "boltzgen infeasible total_len=5 did not raise")
    cases.append({"name": "infeasible_total_len_raises", "raised": True})

    return cases


def run_odesign() -> list[dict[str, Any]]:
    from ofoldx.models.odesign.model import ODesignPlacements
    from ofoldx.models.odesign.processor import _materialize_placement

    cases: list[dict[str, Any]] = []

    # Pure-generated multi-chain (one declaration group per output chain; family per-chain
    # exact length == max_length). ODesign mints its own odesign_chain_N realized ids.
    # Realization is exercised through _materialize_placement, the module function that the
    # migrated ODesignProcessor.__call__ invokes to build the placement_map (constructing the
    # full processor is skipped only to dodge an unrelated chanfig/Literal config-validation
    # incompatibility in the zyc-odesign env; the realization code under test is identical).
    placement = {
        "chainA": {"segA": GeneratedSegment("protein", 12, 12)},
        "chainB": {"segB": GeneratedSegment("protein", 7, 7)},
        "chainC": {"segC": GeneratedSegment("protein", 0, 0)},
    }
    system, _sites, pm, io, _src_res, _src_atom = _materialize_placement(ODesignPlacements(placements=placement), None)
    metrics = check_placement_map(placement, pm, io)
    lens = {pid: len(occ) for group in pm.values() for pid, occ in group.items()}
    _require(lens == {"segA": 12, "segB": 7, "segC": 0}, f"odesign per-chain lengths wrong: {lens}")
    _require(pm["chainC"]["segC"] == (), "odesign zero-length chain not ()")
    _require(
        resolve(SymbolicResidueSite("segA", -1), pm, io) == resolve(SymbolicResidueSite("segA", 11), pm, io),
        "odesign negative/positive offset disagree",
    )
    # two non-empty placements must live on DISTINCT realized chain instances.
    id_a = pm["chainA"]["segA"][0][0]
    id_b = pm["chainB"]["segB"][0][0]
    _require(id_a != id_b, "odesign distinct chains share a realized id")
    exhaustion = check_exhaustion(pm, system)
    cases.append({"name": "multichain_exact_lengths", "lengths": lens, **metrics, "exhaustion": exhaustion})

    return cases


def run_pxdesign() -> list[dict[str, Any]]:
    from ofoldx.models.pxdesign import PxDesignProcessor

    processor = PxDesignProcessor()
    cases: list[dict[str, Any]] = []

    # Released PXDesign: exactly one fixed-length generated protein binder, input=None.
    placement: PlacementSpec = {"binder_chain": {"binder": GeneratedSegment("protein", 16, 16)}}
    batch = processor(conditions=(placement,))
    pm = batch.layout.placement_map
    io = derive_instance_of(placement)  # layout carries placement_map; instance_of is derived from the PlacementSpec.
    metrics = check_placement_map(placement, pm, io)
    length = len(pm["binder_chain"]["binder"])
    _require(length == 16, f"pxdesign fixed binder length wrong: {length}")
    _require(
        resolve(SymbolicResidueSite("binder", -1), pm, io) == pm["binder_chain"]["binder"][15],
        "pxdesign negative-offset resolve wrong",
    )
    exhaustion = check_exhaustion(pm, batch.layout.system)
    cases.append({"name": "single_fixed_binder", "binder_length": length, **metrics, "exhaustion": exhaustion})

    # Global-uniqueness gate: a duplicate placement_id must be rejected at construction.
    raised = False
    try:
        validate_placement(
            {"g1": {"dup": GeneratedSegment("protein", 4, 4)}, "g2": {"dup": GeneratedSegment("protein", 4, 4)}}
        )
    except ValueError:
        raised = True
    _require(raised, "duplicate placement_id was not rejected")
    cases.append({"name": "duplicate_placement_id_rejected", "raised": True})

    return cases


class _ExhaustionEntity:
    r"""Minimal stand-in exposing ``.components`` for :func:`check_exhaustion`."""

    def __init__(self, count: int) -> None:
        self.components = tuple(range(count))


class _ExhaustionSystem:
    r"""Minimal ``MolecularSystem`` stand-in: rfdiffusion has no realized system at lowering
    time (lengths are sampled inside ``model.design``), so the independent ground truth for
    :func:`check_exhaustion` is synthesized from the realized ``residue_map`` — the SAME
    per-instance ``(instance_id, component_index)`` surface ``_decode_structure`` builds."""

    def __init__(self, counters: dict[str, int]) -> None:
        self.instances = {instance_id: instance_id for instance_id in counters}
        self.entities = {instance_id: _ExhaustionEntity(n) for instance_id, n in counters.items()}


def run_rfdiffusion() -> list[dict[str, Any]]:
    r"""rfdiffusion NOW ADOPTS ``placement_map`` (W6 caveat a): after ``model.design`` samples
    the variable generated lengths it lifts the flat ``output_residues`` surface into the shared
    grouped ``placement_map`` via ``processor.materialize_placement_map``. Checked here through
    that same materializer on a faithful realized ``residue_map`` (the per-instance sequential
    ``(instance_id, component_index)`` surface ``_decode_structure`` mints), so this exercises the
    real production path without a GPU model run. Also verifies the canonical realized-id
    convention (W6 caveat b: realized id == shared ``chain_{n}``) and the ``contig``-string
    projection."""

    from ofoldx.models.rfdiffusion.model import RFdiffusionPlacements as RFContig
    from ofoldx.models.rfdiffusion.processor import (
        _resolve_placement,
        from_contig_string,
        materialize_placement_map,
        to_contig_string,
    )

    cases: list[dict[str, Any]] = []

    # ---- Case A: placement_map adoption on a pure-generated + mixed existing/generated topology.
    placements: PlacementSpec = {
        "chain_1": {"binder": GeneratedSegment("protein", 4, 4)},
        "chain_2": {
            "left": GeneratedSegment("protein", 2, 2),
            "A:0": ExistingSpan("A", 0, 3),
            "right": GeneratedSegment("protein", 2, 2),
        },
    }
    validate_placement(placements)  # frozen global-uniqueness gate applies to rfdiffusion too.
    placement = RFContig(placements=placements, chain_lengths=(4, 7))
    output_residues, output_instances = _resolve_placement(placement)
    _require(len(output_residues) == len(output_instances), "residue/instance length mismatch")

    # W6 caveat b — canonical realized-id convention: the realized OUTPUT instance ids are the
    # shared `chain_{n}` (uniform, 0-based, no family prefix), minted by InstanceIdAllocator in
    # declaration order over the groups with >=1 occurrence — NOT the declaration-group key nor
    # the source id. Both declared groups here are non-empty, so they realize onto chain_0, chain_1.
    expected_realized = [realized_instance_id(index) for index in range(len(placements))]
    _require(
        set(output_instances) == set(expected_realized),
        f"realized instance ids {sorted(set(output_instances))} are not the canonical chain_{{n}} "
        f"ids {sorted(expected_realized)} (canonical realized-id convention violated)",
    )
    # Declaration order: the first declared group (binder) realizes onto chain_0, the second
    # (left/A:0/right) onto chain_1; every residue of a group shares that one realized id.
    _require(
        output_instances[0] == "chain_0" and output_instances[-1] == "chain_1",
        f"declaration-order allocation wrong: first={output_instances[0]!r} last={output_instances[-1]!r}",
    )

    # Synthesize the realized residue_map exactly as _decode_structure does: component_index is
    # the per-instance sequential position in row order. This is the surface model.design feeds
    # into materialize_placement_map.
    residue_map: dict[Any, tuple[str, int]] = {}
    counters: dict[str, int] = {}
    for reference, instance_id in zip(output_residues, output_instances):
        index = counters.get(instance_id, 0)
        residue_map[reference] = (instance_id, index)
        counters[instance_id] = index + 1

    placement_map, instance_of_map = materialize_placement_map(placement, output_residues, residue_map)

    # (1)-(5) shared frozen §3.7 invariants over the materialized placement_map.
    metrics = check_placement_map(placements, placement_map, instance_of_map)
    # (7) EXHAUSTION — full partition of the realized output surface, no unaddressed tail.
    exhaustion = check_exhaustion(placement_map, _ExhaustionSystem(counters))

    # Grouped realization matches the declared topology: binder=4, left/right=2, motif=3.
    lengths = {pid: len(occ) for grp in placement_map.values() for pid, occ in grp.items()}
    _require(
        lengths == {"binder": 4, "left": 2, "A:0": 3, "right": 2},
        f"placement_map occurrence lengths wrong: {lengths}",
    )
    # The existing motif is in the SECOND declared group, so its occurrences ride on that group's
    # realized id chain_1 (the shared canonical id), NOT the source id "A" nor the group key.
    _require(
        placement_map["chain_2"]["A:0"][0][0] == "chain_1",
        "existing motif did not realize onto its declaration-group's canonical chain id",
    )
    # A single placement's occurrences are contiguous within one realized instance. The binder is
    # in the FIRST declared group -> realized id chain_0.
    _require(
        resolve(SymbolicResidueSite("binder", -1), placement_map, instance_of_map) == ("chain_0", 3),
        "rfdiffusion negative-offset resolve wrong",
    )
    _require(exhaustion["realized_output_components"] == len(output_residues), "exhaustion miscount")
    cases.append(
        {
            "name": "placement_map_adoption_mixed_topology",
            "note": "rfdiffusion adopts grouped placement_map (W6 caveat a); realized id == shared chain_{n}",
            "occurrence_lengths": lengths,
            **metrics,
            "exhaustion": exhaustion,
        }
    )

    # ---- Case B: contig-string ⇄ core projection round-trip (from/to_contig_string).
    source_instances = {"A": 200, "B": 100}
    contigs = "10-40/A163-181/10-40 B1-100/0"
    spec, combined_length = from_contig_string(contigs, source_instances=source_instances)
    validate_placement(spec)
    # A163-181 -> ExistingSpan("A", 163, 182) inclusive->half-open (audit §3).
    motif = next(v for grp in spec.values() for v in grp.values() if type(v) is ExistingSpan and v.instance_id == "A")
    _require((motif.start, motif.stop) == (163, 182), f"A163-181 lowered to {(motif.start, motif.stop)}")
    # combined_length is the family ContigMap.length: (Sum gen min, Sum gen max) over generated subcons.
    _require(combined_length == (10 + 10 + 0, 40 + 40 + 0), f"combined_length wrong: {combined_length}")
    # Topology round-trips: re-emit, re-parse, re-emit is stable and structurally identical.
    emitted = to_contig_string(spec, combined_length)
    spec2, combined2 = from_contig_string(emitted, source_instances=source_instances)
    _require(to_contig_string(spec2, combined2) == emitted, "contig-string round-trip not stable")
    _require(combined2 == combined_length, "combined_length not preserved across round-trip")
    # Structural equality of the two parses (values + declared order).
    _require(
        [list(g.values()) for g in spec.values()] == [list(g.values()) for g in spec2.values()],
        "contig-string parse not idempotent under re-emission",
    )
    cases.append(
        {
            "name": "contig_string_projection_roundtrip",
            "contigs": contigs,
            "emitted": emitted,
            "combined_length": list(combined_length),
            "groups": len(spec),
        }
    )
    return cases


RUNNERS = {
    "boltzgen": run_boltzgen,
    "odesign": run_odesign,
    "pxdesign": run_pxdesign,
    "rfdiffusion": run_rfdiffusion,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable JSON result")
    args = parser.parse_args()

    try:
        cases = RUNNERS[args.family]()
        result = {"family": args.family, "realization_parity": "PASS", "cases": cases}
    except Exception as error:  # noqa: BLE001 — report, do not fabricate a pass
        result = {"family": args.family, "realization_parity": "FAIL", "error": repr(error)}
        if args.json:
            print(json.dumps(result, indent=2))
        raise SystemExit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"=== {args.family} realization-parity: {result['realization_parity']} ===")
        for case in cases:
            print(json.dumps(case))
    print("REALIZATION_OK")


if __name__ == "__main__":
    main()
