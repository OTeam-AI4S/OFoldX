# data

`ofoldx.data` contains the canonical molecular values and the small evidence or
condition records shared by real model families.

`MolecularSystem` is the coordinate-free tensor graph over system, entity,
instance, component, atom, and edge axes. `Structure` is one coordinate
realization aligned to its complete canonical atom axis. It stores coordinate,
occupancy, and B-factor values with explicit masks, plus the source residue and
insertion labels aligned to the component axis. It does not store source-file
identity, a second atom layout, confidence, or a generic annotations bag.

`ChemicalComponent`, `PolymerSequence`, and `Molecule` are construction values
lowered once into that graph. `Molecule` retains the irreducible
non-polymer/branched/water entity category; component count does not infer it.
`ChemicalComponent.without_hydrogens()` is the exact H/D/T induced-subgraph
operation for component-definition inspection. Structure readers instead materialize
the complete occurrence graph before `suppress_hydrogens()` folds the remaining
hydrogen vertices into heavy-atom counts. `ComponentReference` and `AtomSite` are symbolic
boundary addresses; model inputs carry canonical integer maps instead of object
views. Low-level chemical values protect identity, topology, and referential
closure. They do not perform generic plausibility, valence, or lifecycle
validation. Their module-like `.to(dtype=...)` casts floating values and preserves
the integer and Boolean graph schema; it does not infer or repair a floating dtype.

`tokenization.py` owns the canonical biological vocabulary and the one shared
atom-to-token partition operation. A `TokenLayout` retains canonical component
and atom indices; architecture metadata may choose component or atom tokens, but
families do not implement another graph traversal. `crop.py` owns replaceable
training views over that layout. A Cropper selects token rows while the complete
`Structure` remains the coordinate truth; it does not parse files, lower model
features, or know family condition types. Configure an included policy with
`functools.partial`, or pass any callable with the same signature. The family
adapter supplies two distinct masks: every `required_token_mask` row must remain,
while the selected view only needs one row from `anchor_token_mask`. Resolved
coordinates are therefore a diffusion-family anchor, not a universal crop rule.
The included policies never split the atom tokens of one canonical component;
when the next complete component does not fit, a crop may contain fewer than the
configured maximum number of tokens.

There is no global `Condition` class or registry. The currently frozen shared
records are:

- `AllowedResidueIdentities` and `TiedResidueIdentities` for fixed-topology
  sequence design.
- `Hotspot` for the shared distance-free hotspot label used by concrete design
  families.
- The placement core (`placement.py`): `GeneratedSegment`, `ExistingSpan`,
  `SymbolicResidueSite`, `SymbolicAtomSite`, the `PlacementSpec` /
  `PlacementValue` / `PlacementMap` type aliases, and direct indexing helpers. A
  PlacementSpec is an ordered
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

- `molecular.py` — construction values, the canonical tensor graph, exact
  occurrence references, and graph operations.
- `structure.py` — one canonical coordinate realization over the complete atom
  axis.
- `components.py` — package-audited standard component access.
- `evidence.py` — MSA and template evidence.
- `tokenization.py` — canonical token vocabulary and structural partition.
- `crop.py` — injectable continuous, spatial, and interface crop policies.
- `conditions.py` — the two frozen plain identity-condition records above.
- `placement.py` — the frozen placement core (generation-topology identity).
