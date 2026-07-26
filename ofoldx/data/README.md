# data

`ofoldx.data` contains the canonical molecular values and the small evidence or
condition records shared by real model families.

`MolecularSystem` is a coordinate-free assembly. `Structure` is one coordinate
realization over that system, aligned by exact `AtomSite` values. The canonical
carrier stores coordinates, their mask, and optional occupancy; it does not store
source-file identity, atom-layout policy, B-factors, confidence, provenance, or a
generic annotations bag.

Low-level chemical values protect identity, topology, and referential closure.
They do not perform generic plausibility, valence, tensor, or lifecycle validation.
Arrays are retained by reference and explicit caller dtypes are not repaired.

There is no global `Condition` class or registry. The currently frozen shared
records are:

- `AllowedResidueIdentities` and `TiedResidueIdentities` for fixed-topology
  sequence design.
- The placement core (`placement.py`): `GeneratedSegment`, `ExistingSpan`,
  `SymbolicResidueSite`, `SymbolicAtomSite`, the `PlacementSpec` /
  `PlacementValue` / `PlacementMap` type aliases, and the `instance_of`,
  `validate_placement`, and `resolve` helpers. A PlacementSpec is an ordered
  `declaration_group -> placement_id -> value` nested mapping whose inner
  `placement_id` is globally unique across the whole PlacementSpec and is the sole
  pre-realization reference target. Length is re-homed per family (no shared
  `chain_lengths` / `total_len`).

Spatial conditioning is family-owned: Protenix, Boltz, RFdiffusion, ODesign, and
other families do not share one `Contact`, `AtomDistance`, or `Pocket` invariant.
Their official controls remain mandatory on the concrete family APIs once each
complete Tik freezes. Absence from this shared list is not permission to omit a
released capability.

## Public modules

- `molecular.py` — coordinate-free chemical values and exact occurrence references.
- `structure.py` — one canonical coordinate realization.
- `components.py` — package-audited standard component access.
- `evidence.py` — MSA and template evidence.
- `conditions.py` — the two frozen plain identity-condition records above.
- `placement.py` — the frozen placement core (generation-topology identity).
