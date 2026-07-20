r"""X1-B sanity check: BoltzGen whole-assembly total_len realization -> placement_map.

Not a numerical parity test (no upstream generation reference is dumped). It exercises the
frozen §3.7 X1-B path end to end: the previously-gated fully-generated ``input=None`` branch
now realizes a whole-assembly PlacementSpec into a synthesized MolecularSystem, a placement_map, and
a runnable BoltzGenInput, and a SymbolicResidueSite resolves through it.
"""

from __future__ import annotations

from ofoldx.data.placement import GeneratedSegment, SymbolicResidueSite, resolve
from ofoldx.models.boltzgen import (
    BoltzGenPlacements,
    BoltzGenProcessor,
    DesignMaskResidue,
    from_design_mask,
    to_design_mask,
)


def _dump(title: str, batch, placement_map, instance_of) -> None:
    print(f"=== {title} ===")
    print("token_count:", int(batch.token_mask.sum().item()))
    print("atom_features:", tuple(batch.atom_features.shape))
    print("instance_of:", instance_of)
    for group, placements in placement_map.items():
        print(f"group {group!r}:")
        for placement_id, occ in placements.items():
            head = occ[:2]
            tail = occ[-1:] if occ else ()
            print(f"  {placement_id!r}: len={len(occ)} head={head} tail={tail}")


def main() -> None:
    processor = BoltzGenProcessor()

    # Case 1 — degenerate whole-assembly: only a total_len budget, one generated chain.
    batch1 = processor(None, total_len=24)
    _dump("case1 total_len=24 (no placement)", batch1, batch1.layout.placement_map, batch1.layout.instance_of)
    pm1 = batch1.layout.placement_map
    assert pm1 == {"assembly": {"chain_A": tuple(("chain_0", i) for i in range(24))}}, pm1
    assert int(batch1.token_mask.sum().item()) == 24

    # Case 2 — multi-chain assembly, ranged lengths drawn from the total_len budget, a symmetric
    # twin, and a zero-length placement (-> () under the frozen coverage rule).
    placement = BoltzGenPlacements(
        placements={
            "assembly": {
                "binderA": GeneratedSegment("protein", 10, 40),
                "binderB": GeneratedSegment("protein", 10, 40),
                "scaffold": GeneratedSegment("protein", 0, 0),
            }
        }
    )
    # budget: min 20, max 80; ask for 50 -> greedy: binderA 40, binderB 10, scaffold 0.
    batch2 = processor(None, total_len=50, conditions=(placement,))
    pm2 = batch2.layout.placement_map
    io2 = batch2.layout.instance_of
    _dump("case2 total_len=50 multi-chain", batch2, pm2, io2)
    lens = {pid: len(occ) for pid, occ in pm2["assembly"].items()}
    assert lens == {"binderA": 40, "binderB": 10, "scaffold": 0}, lens
    assert sum(lens.values()) == 50
    assert pm2["assembly"]["scaffold"] == ()
    # distinct realized chain instances, each contiguous from 0:
    assert pm2["assembly"]["binderA"][0] == ("chain_0", 0)
    assert pm2["assembly"]["binderB"][0] == ("chain_1", 0)
    assert int(batch2.token_mask.sum().item()) == 50

    # resolve() through the frozen core: last residue of binderA, first of binderB.
    assert resolve(SymbolicResidueSite("binderA", -1), pm2, io2) == ("chain_0", 39)
    assert resolve(SymbolicResidueSite("binderB", 0), pm2, io2) == ("chain_1", 0)
    print("resolve binderA[-1]:", resolve(SymbolicResidueSite("binderA", -1), pm2, io2))
    print("resolve binderB[0] :", resolve(SymbolicResidueSite("binderB", 0), pm2, io2))

    # Design-mask projection round-trip (CONTIG_GENERALITY_AUDIT.md §3): to_design_mask over the
    # realized rows -> all designed (every placement is a GeneratedSegment); from_design_mask rebuilds
    # the PlacementSpec at EXACT realized lengths (a mask fixes exact lengths; the zero-length scaffold
    # has no realized row and so does not reappear -- honest lossless-for-realized-rows).
    mask, groups = to_design_mask(placement.placements, pm2)
    assert mask == [True] * 50, mask
    assert groups == ["assembly"] * 50, groups
    # Build per-row (group, placement_id) from the realized map, in declared order, to reproject.
    rows = [
        (group, placement_id)
        for group, placements in pm2.items()
        for placement_id, occ in placements.items()
        for _ in occ
    ]
    residues = [DesignMaskResidue(designed=True, polymer_type="protein") for _ in rows]
    rebuilt = from_design_mask(
        residues,
        group_of=lambda i: rows[i][0],
        placement_id_of=lambda i: rows[i][1],
    )
    assert rebuilt == {
        "assembly": {
            "binderA": GeneratedSegment("protein", 40, 40),
            "binderB": GeneratedSegment("protein", 10, 10),
        }
    }, rebuilt
    print("design_mask round-trip:", rebuilt)

    # Case 3 — total_len outside the feasible band raises.
    try:
        processor(None, total_len=5, conditions=(placement,))
    except ValueError as error:
        print("case3 infeasible total_len raised:", error)
    else:
        raise AssertionError("total_len=5 should be infeasible for a [20,80] assembly")

    print("ALL_OK")


if __name__ == "__main__":
    main()
