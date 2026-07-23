# Upstream bugs

This document records confirmed bugs in upstream reference implementations,
including both corrected OFoldX divergences and defects intentionally preserved
for strict numerical parity. Each entry states which policy applies. It is an
evidence log, not a compatibility matrix or a list of unsupported features.

## Boltz reference augmentation skips the highest space ID

**Upstream:** `jwohlwend/boltz` and `HannesStark/boltzgen`

**Affected families:** Boltz-1, Boltz-2, and BoltzGen.

**Classification:** confirmed inference-preprocessing defect, intentionally
preserved by OFoldX for strict upstream parity.

The released reference augmentation loops over `range(space_ids.max())`. Because
the upper bound is exclusive, the highest reference-space ID is never augmented;
when zero is the only ID, no reference space is augmented. OFoldX preserves this
exact loop in all three affected Processors so identical inputs retain identical
reference features and downstream numerical outputs.

This behavior may be corrected only after the project deliberately selects a
corrected upstream target and re-establishes end-to-end numerical parity against
that target. It must not be changed as an isolated cleanup.

## Boltz-2 affinity molecular weight drops implicit hydrogens

**Upstream:** `jwohlwend/boltz`

**Affected reference:** present in the Boltz-2.0.0 implementation at
[`afe4aa6`](https://github.com/jwohlwend/boltz/commit/afe4aa6) and still present at
OFoldX's audited revision
[`c46ac60`](https://github.com/jwohlwend/boltz/tree/c46ac60c128875eac7ecc722937894edb4162f82).

**Classification:** independently reproduced upstream inference-preprocessing bug.
This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

For an affinity ligand supplied as SMILES, the upstream parser standardizes the
SMILES, calls `Chem.AddHs`, removes hydrogens with
`Chem.RemoveHs(..., sanitize=False)`, and then calls `Descriptors.MolWt` on the
unsanitized result. The corresponding CCD branch calls `Descriptors.MolWt` directly
on the CCD molecule. See the audited
[`schema.py`](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/parse/schema.py#L1185-L1272).

`RemoveHs(..., sanitize=False)` leaves the remaining atoms with no implicit-hydrogen
property cache. `MolWt` therefore omits ordinary implicit hydrogens. The same
chemical graph consequently receives a different value depending on whether it
entered through SMILES or CCD.

The behavior is reproducible with RDKit:

| SMILES | Upstream SMILES-path value | Molecular weight after sanitization |
| ------ | -------------------------: | ----------------------------------: |
| `CC`   |                     24.022 |                              30.070 |
| `CCC`  |                     36.033 |                              44.097 |
| `CCO`  |                     40.021 |                              46.069 |

The incorrect value survives the pickle round-trip used by the upstream molecule
cache. Updating the property cache or sanitizing the molecule restores the expected
molecular weight. Explicit hydrogen atoms such as those in `[NH4+]` remain counted,
which further identifies stale implicit-hydrogen state as the cause rather than an
intentional heavy-atom-mass definition.

### Impact

The value is used only by Boltz-2's optional post-head molecular-weight correction.
It is not an input to the learned affinity heads and does not occur in their
checkpoint tensors. The upstream command-line interface disables that correction by
default in its
[prediction documentation](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/docs/prediction.md).

The released correction coefficients are hard-coded after the affinity ensemble.
Public upstream material says that they were fitted on a holdout set using ligand
molecular weight in the
[Boltz-2 technical report](https://jeremywohlwend.com/assets/boltz2.pdf), but the
preprocessing used to fit them has not been published.
It is therefore not known whether those coefficients were fitted with correct
molecular weights or with the faulty inference representation. This uncertainty is
not evidence that the faulty value is intentional.

### OFoldX behavior

OFoldX retains ChEMBL standardization for raw affinity SMILES. After parsing, SMILES
and CCD inputs both become the same canonical `Molecule`, and molecular weight is
derived consistently from that standardized graph. Input provenance does not alter
the computation.

OFoldX does not expose the faulty value as `exact_mw`, does not ask users for a
`ligand_molecular_weight`, and does not add a legacy or source-selection switch.
The Boltz-2 parity script must identify molecular-weight-corrected outputs as an
intentional bug-fix divergence until the upstream calibration preprocessing is
available. Any generated comparison report remains local.

## Boltz-2 leaks component-internal bond order into token-pair diagonals

**Upstream:** `jwohlwend/boltz`

**Affected reference:** OFoldX's audited Boltz-2 revision
[`c46ac60`](https://github.com/jwohlwend/boltz/tree/c46ac60c128875eac7ecc722937894edb4162f82).

**Classification:** independently reproduced training/inference feature-construction
bug. This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

Both Boltz-2 input parsers append every component-internal residue bond to the
structure bond table. The inference-schema path does so in
[`parse_boltz_schema`](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/parse/schema.py#L1472-L1485);
the training mmCIF path does the same in
[`parse_mmcif`](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/parse/mmcif.py#L1125-L1142).

`tokenize_structure` maps every atom of a standard residue to one residue token at
[lines 252–255](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/tokenize/boltz2.py#L252-L255)
and does the same for a modified polymer residue at
[lines 352–355](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/tokenize/boltz2.py#L352-L355).
It then converts every atom bond to token indices without rejecting equal token
indices at
[lines 359–371](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/tokenize/boltz2.py#L359-L371).

Consequently every internal bond of one residue becomes a self-edge. The feature
builder writes those self-edges onto `token_bonds[i, i]` and
`type_bonds[i, i]` at
[lines 688–705](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/feature/featurizerv2.py#L688-L705).
When a component has multiple internal bonds, the final bond in RDKit/parser
iteration order overwrites every preceding type on the same diagonal cell. Reversing
the declaration order of the same undirected chemical graph therefore changes the
model input.

A minimal reproduction is a one-residue protein input such as alanine. All alanine
atoms map to token zero, while its component bonds produce multiple
`TokenBondV2(0, 0, type)` rows. Applying the released feature loop in forward order
leaves the final type at `type_bonds[0, 0]`; applying the same graph rows in reverse
order leaves a different type whenever the component contains more than one bond
type. The graph and tokenization are otherwise unchanged.

The released model consumes the resulting diagonal through both its binary bond
projection and, when enabled, its bond-type embedding at
[`Boltz2.forward`](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/model/models/boltz2.py#L420-L429).
Both audited released affinity and confidence checkpoints contain
`token_bonds.weight` and `token_bonds_type.weight`.

### Explicit same-token connections are different

This defect does not justify deleting every same-token edge.

The training mmCIF parser appends explicit source `struct_conn` covalent connections
after component bonds at
[`parse_mmcif` lines 1176–1194](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/parse/mmcif.py#L1176-L1194).
The inference schema likewise appends user `constraints[].bond` connections after
component bonds, assigning `COVALENT`, at
[`parse_boltz_schema` lines 1755–1760](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/parse/schema.py#L1755-L1760).

Neither path excludes a connection whose endpoints belong to the same residue
token. Such an explicit connection deterministically writes bond type 6 after the
component rows. It represents a declared source/user relationship rather than an
arbitrary component-bond iteration artifact. The training parser implements this
semantic even though the frequency of same-token examples in the unpublished
training corpus is unknown.

A minimal inference fixture adds a non-intrinsic bond constraint between two atoms
of one residue, for example `A:1:N` and `A:1:O`. Both endpoints map to token zero,
and the final diagonal type is `COVALENT + 1 == 6`.

### OFoldX behavior

OFoldX derives token topology from the complete canonical topology while retaining
bond origin.

A component-origin bond whose endpoints map to the same token is omitted. OFoldX
does not add declaration order to `ChemicalComponent.bonds`, because iteration order
is not chemical meaning and must not become a second canonical identity.

An explicit source- or user-origin `SystemBond` remains a Boltz-2 covalent feature
even when its endpoints map to the same token. Default `polymer_rule` links are not
added because the audited Boltz-2 parser does not add ordinary adjacent polymer
bonds to this feature. A geometry-inferred overlay is rejected by the Boltz-2
Processor rather than silently discarded or represented as a declared connection.

### Parity impact

Raw released-weight parity intentionally differs for standard and modified polymer
token diagonals: OFoldX supplies zero for the component-internal self-edge where the
upstream feature depends on arbitrary component bond order. This difference enters
the initial pair representation and can propagate through the trunk, diffusion, and
confidence outputs.

Parity tooling must:

1. report the raw-reference diagonal and output difference as an intentional bug-fix
   divergence;
2. compare exact behavior against a patched reference that skips
   component-derived `token_1 == token_2` rows; and
3. include an explicit source/user same-token connection fixture proving that the
   meaningful `COVALENT=6` feature remains present.

No compatibility switch or ordered canonical bond representation is added.

## Boltz reuses taxonomically paired MSA rows as unpaired rows

**Upstream:** `jwohlwend/boltz`

**Affected reference:** OFoldX's audited Boltz revision
[`c46ac60`](https://github.com/jwohlwend/boltz/tree/c46ac60c128875eac7ecc722937894edb4162f82).

**Classification:** independently reproduced inference feature-construction bug.
This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

The MSA pairing code first represents each retained taxonomy as
`(taxonomy_id, [(chain_id, sequence_index), ...])`. It then attempts to collect the
visited `(chain_id, sequence_index)` pairs before constructing the pool of ordinary
unpaired rows. Both the original and v2 featurizers instead construct
`(taxonomy_id, (chain_id, sequence_index))` values. See
[`featurizer.py` lines 202–215](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/feature/featurizer.py#L202-L215)
and
[`featurizerv2.py` lines 316–329](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/feature/featurizerv2.py#L316-L329).

The resulting set can never match the later `(chain_id, sequence_index)` membership
test. Every non-query row therefore remains available even when it was already used
in a taxonomic pair. Such a row may be emitted once with its paired flag set and
again as an unpaired filler or ordinary unpaired row.

For example, with
`taxonomy_map = [(7, [(0, 1), (1, 1)])]`, the released comprehension produces
`{(7, (0, 1)), (7, (1, 1))}`. Neither `(0, 1)` nor `(1, 1)` is marked visited. The
intended flattening produces exactly those two chain/row pairs and removes them from
the ordinary available pools.

### Impact

The duplicate occurrence is not harmless padding. The paired flag is an explicit
Boltz-2 trunk input, so the same biological sequence can contribute twice with
different pairing semantics and can displace other rows under the released MSA
limits or random crop.

### OFoldX behavior

`MSACollection` records the complete raw row assembly plan and whether each populated
cell is paired. A source row used for taxonomic pairing is excluded from the ordinary
available pool. It may also occur as unpaired only when it independently belongs to
the supplied raw unpaired pool; reuse is explicit in the row plan rather than an
accidental consequence of a malformed visited set.

OFoldX does not add a compatibility switch for the duplicate-row behavior. Boltz
parity tooling records the raw divergence and compares canonical MSA features against
a reference with the visited-set flattening corrected.

## Boltz drops MSA deletion features after slicing the query row

**Upstream:** `jwohlwend/boltz`

**Affected reference:** OFoldX's audited Boltz revision
[`c46ac60`](https://github.com/jwohlwend/boltz/tree/c46ac60c128875eac7ecc722937894edb4162f82).

**Classification:** independently reproduced inference feature-construction bug.
This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

Each chain stores one global deletion-record array and gives every MSA row offsets
into that original array. The released loop initializes a local variable from the
global array once, then replaces that variable with the current row's slice. The
next row applies its original global offsets to the already sliced previous row.
See
[`featurizer.py` lines 313–323](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/feature/featurizer.py#L313-L323)
and
[`featurizerv2.py` lines 427–437](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/feature/featurizerv2.py#L427-L437).

The first row is the query and normally has an empty deletion interval. It therefore
turns the local array into an empty slice; every later slice is also empty, and the
constructed deletion lookup contains no values for the non-query rows. Even without
an empty first interval, later offsets address the wrong sliced array.

### Impact

The MSA token identities and pairing plan remain present, which makes the defect easy
to miss, but the model receives zero/default deletion features where the source MSA
declared insertions. This changes an explicit checkpoint input for every affected
row.

### OFoldX behavior

OFoldX projects each raw MSA row to its entity axis together with that row's deletion
counts. Boltz lowering indexes every row from the original bound MSA rather than
mutating the source array across iterations. No compatibility path reproduces the
stateful slicing bug.

Boltz parity tooling compares both the released zeroed deletion features and a
patched reference that slices the original global deletion array for every row;
generated reports remain local.

## ESMFold2 and Protenix count A3M insertion gaps as inserted residues

**Upstreams:** `Biohub/esm` and `bytedance/Protenix`

**Affected references:** ESM revision
[`ba4d712`](https://github.com/Biohub/esm/tree/ba4d7124864eed323a93bf3cfefcd958f573b75a)
and Protenix revision
[`c3bfc36`](https://github.com/bytedance/Protenix/tree/c3bfc365b3e1341a11935eddfe7bfdc308092147).

**Classification:** independently reproduced A3M feature-construction bugs. This
classification records OFoldX's technical finding; it does not claim
acknowledgement from either upstream maintainer.

### Behavior

A3M is A2M with gaps aligned to insertion columns optionally omitted. Lowercase
letters are inserted residues, whereas `.` is only the gap that occupies such an
insertion column. HH-suite's canonical formatter both describes this distinction
and removes every `.` when emitting A3M. See the inspected
[`reformat.pl`](https://github.com/soedinglab/hh-suite/blob/43095e46ada4ec2a8a47d47ef5ad7e38b1429f7b/scripts/reformat.pl).

ESM's parser instead classifies both lowercase letters and `.` as insertions and
increments the deletion count for both. See
[`a3m_deletion_counts`](https://github.com/Biohub/esm/blob/ba4d7124864eed323a93bf3cfefcd958f573b75a/esm/utils/msa/msa.py#L27-L43).
Protenix treats every character absent from its aligned alphabet as an inserted
residue, so `.` likewise increments its cumulative deletion count. It then silently
truncates or zero-pads rows whose projected width differs from the query width. See
[`sequences_to_array`](https://github.com/bytedance/Protenix/blob/c3bfc365b3e1341a11935eddfe7bfdc308092147/protenix/data/msa/msa_utils.py#L97-L118).

For example, `Ac.DE` and `AcDE` are equivalent A3M representations: both contain
one inserted residue `c` before `D`. Both upstream parsers assign a count of two to
the first representation and one to the second because the optional dot changes
their feature.

### Impact

Semantically equivalent A3M files produce different checkpoint inputs. In
Protenix, malformed or wrong-query-width rows can additionally become plausible
zero-padded rows instead of failing at the evidence boundary.

### OFoldX behavior

`read_msa` removes `.` without incrementing deletion counts, counts only lowercase
residues, and requires every biological row to project to the exact entity length.
Family parity tooling records the raw upstream divergence and compares against a
reference parser with insertion-gap handling corrected; no compatibility switch
preserves the format-dependent feature.

## RFdiffusion skips fixed-motif restoration on the final sampling step

**Upstream:** `RosettaCommons/RFdiffusion`

**Affected reference:** OFoldX's audited RFdiffusion revision
[`2d0c003`](https://github.com/RosettaCommons/RFdiffusion/tree/2d0c003df46b9db41d119321f15403dec3716cd9).

**Classification:** independently reproduced inference-output bug. This
classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

For every nonfinal reverse-diffusion step, the upstream sampler calls
`get_next_pose`, which proper-aligns the predicted `pX0` to all fixed motif N/CA/C
atoms jointly and copies the fixed motif rows from the current pose. See
[`get_next_pose`](https://github.com/RosettaCommons/RFdiffusion/blob/2d0c003df46b9db41d119321f15403dec3716cd9/rfdiffusion/inference/utils.py#L406-L501).

At `t == final_step`, `sample_step` bypasses that operation and assigns the raw
network `pX0` directly to `x_t_1`. See
[`sample_step`](https://github.com/RosettaCommons/RFdiffusion/blob/2d0c003df46b9db41d119321f15403dec3716cd9/rfdiffusion/inference/model_runners.py#L769-L794).
The command-line writer then serializes the final `x_t_1`, not a separately aligned
or restored value. See
[`run_inference.py`](https://github.com/RosettaCommons/RFdiffusion/blob/2d0c003df46b9db41d119321f15403dec3716cd9/scripts/run_inference.py#L93-L100)
and its
[`writepdb` call](https://github.com/RosettaCommons/RFdiffusion/blob/2d0c003df46b9db41d119321f15403dec3716cd9/scripts/run_inference.py#L133-L145).

The raw prediction is not constrained to reproduce the supplied motif coordinates
exactly. The omission therefore defeats the same fixed-motif restoration that the
sampler applies at every preceding step.

### Impact

A motif-scaffolding result can move its nominally fixed rows on the final step. The
input diffusion path also subtracts the joint motif C-alpha centroid, so returning
that raw final value without restoring the caller frame is incompatible with a
complete Structure that retains external substrate or other non-output context at
its input coordinates.

### OFoldX behavior

OFoldX treats fixed motif coordinates as coordinates in the caller's frame. It
proper-aligns the final predicted structure into that frame as an internal
RFdiffusion operation, restores every fixed row exactly, adds back the caller-frame
center, and only then constructs the canonical Structure. Restoration is part of
sampling construction; OFoldX does not add a generic post-hoc condition gate.

The RFdiffusion parity tooling records the raw upstream final value and the corrected
canonical result as an intentional bug-fix divergence; generated reports remain
local.

## RFdiffusion inpainting masks only the first repeated source placement

**Upstream:** `RosettaCommons/RFdiffusion`

**Affected reference:** OFoldX's audited RFdiffusion revision
[`2d0c003`](https://github.com/RosettaCommons/RFdiffusion/tree/2d0c003df46b9db41d119321f15403dec3716cd9).

**Classification:** independently reproduced inpainting-selection bug. This
classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

RFdiffusion permits the same source residue to occur more than once in an output
Contig. While lowering `inpaint_seq` and `inpaint_str`,
`ContigMap.get_inpaint_seq_str` finds a selected source residue with
`self.ref.index(res)` and changes only that first row. For example, placing `A1`
twice and selecting `A1` for sequence inpainting leaves the second placement's
sequence visible. A selected residue absent from the output is silently ignored.

### Impact

The public selection syntax names a source residue, not one numbered placement of
that residue. Applying it to only the first copy makes two chemically identical
placements receive different masks without any caller-visible way to request or
explain that distinction. Silently ignoring an absent selection can also discard
an explicit scientific request.

### OFoldX behavior

`inpaint_sequence` accepts ordinary source `ComponentReference` values. A selected
source occurrence masks every realized placement; duplicates are idempotent, and a
selected occurrence absent from the realized output is rejected. Placement-specific
masking, when required by another released operation such as partial-diffusion
`provide_seq`, uses that operation's eventual placement-aware contract rather than
reinterpreting a source reference.

## Protenix v0.5.3 swaps atom-contact channels between training and inference

**Upstream:** `bytedance/Protenix`

**Affected reference:** the first formal constraint-model release, annotated tag
`v0.5.3`, peeled commit
[`a0f95f6`](https://github.com/bytedance/Protenix/tree/a0f95f62430093b46c677f04e3d292ccc5d28d64),
with `protenix_base_constraint_v0.5.0.pt`.

**Classification:** independently reproduced training/inference preprocessing and
checkpoint-contract bug. This classification records OFoldX's technical finding;
it does not claim an acknowledgement from the upstream maintainers.

### Behavior

The constraint checkpoint has a two-channel atom-contact projection intended to
encode minimum and maximum distances. The training path samples tuples in
`(atom_i, atom_j, min_distance, max_distance)` order in
[`_sample_contacts`](https://github.com/bytedance/Protenix/blob/a0f95f62430093b46c677f04e3d292ccc5d28d64/protenix/data/constraint_featurizer.py#L1558-L1565)
and preserves that order while expanding residue tokens in
[`expand_tokens`](https://github.com/bytedance/Protenix/blob/a0f95f62430093b46c677f04e3d292ccc5d28d64/protenix/data/constraint_featurizer.py#L674-L679).

`ContactAtomFeaturizer.generate_spec_constraint`, however, unpacks every tuple as
`(atom_i, atom_j, max_distance, min_distance)`, then writes the purported maximum
to channel 1 and minimum to channel 0. See the
[`generate_spec_constraint` implementation](https://github.com/bytedance/Protenix/blob/a0f95f62430093b46c677f04e3d292ccc5d28d64/protenix/data/constraint_featurizer.py#L1971-L1988).
Consequently, training writes the sampled maximum into channel 0 and the sampled
minimum—normally zero—into channel 1.

The JSON inference path constructs tuples in the opposite
`(max_distance, min_distance)` order at
[`generate_from_json`](https://github.com/bytedance/Protenix/blob/a0f95f62430093b46c677f04e3d292ccc5d28d64/protenix/data/constraint_featurizer.py#L240-L246).
It therefore writes the user-facing maximum into channel 1 instead of the channel
that carried maximum distances during training.

The released checkpoint makes the mismatch observable rather than merely
notational. All 128 values in
`constraint_embedder.contact_atom_z_embedder.weight[:, 0]` are nonzero, with an
L2 norm of approximately `5.78885`; channel 1 is identically zero. The ordinary
token-contact projection has the opposite expected pattern: channel 0 is zero and
all 128 channel-1 values are nonzero. Thus an atom-contact maximum produced by the
official JSON path is multiplied by an all-zero checkpoint column and has no learned
effect.

### Impact

The advertised atom-level `max_distance` constraint is invisible to the released
checkpoint through its official inference preprocessing. The advertised nonzero
`min_distance` also has no trained independent channel: the only active atom-contact
column learned maximum-distance values. Reproducing the JSON tensor layout would
therefore claim conditioning that the checkpoint does not perform.

### OFoldX behavior

OFoldX maps atom-contact upper-bound conditioning to the checkpoint column that was
actually trained on upper bounds. It rejects a nonzero atom-contact lower bound
instead of pretending that the all-zero column consumes it. It does not expose a
second path that reproduces the faulty upstream JSON preprocessing.

The maintained Protenix parity script separately demonstrates the upstream channel
mismatch, the checkpoint column support, and the corrected conditioned computation.
Ordinary zero-constraint and token-contact comparisons remain exact; generated
reports remain local.

## Protenix v0.5.3 drops the first substructure distance bin

**Upstream:** `bytedance/Protenix`

**Affected reference:** the first formal constraint-model release, annotated tag
`v0.5.3`, peeled commit
[`a0f95f6`](https://github.com/bytedance/Protenix/tree/a0f95f62430093b46c677f04e3d292ccc5d28d64),
with `protenix_base_constraint_v0.5.0.pt`.

**Classification:** independently reproduced training-featurization bug plus an
unreleased, unusable inference helper. Nonzero substructure conditioning was not a
documented public function of this release.

### Behavior

Training declares substructure distance boundaries of `0`, `4`, `8`, `16`, and
infinity Å. After bucketizing, however, the featurizer treats only class indices
greater than zero as valid. The intended `[0, 4)` Å class is therefore erased along
with genuinely absent pairs. See the
[`SubStructureFeaturizer` implementation](https://github.com/bytedance/Protenix/blob/a0f95f62430093b46c677f04e3d292ccc5d28d64/protenix/data/constraint_featurizer.py#L2003-L2092).

The released checkpoint confirms the training behavior. Every value in column 0 of
the substructure input projection is exactly zero; the four column norms are
approximately `0`, `0.511945`, `0.783432`, and `0.679077`. Consequently an absent
pair, a selected pair below 4 Å, a diagonal pair, and a one-token selection all have
the same all-zero input representation.

The nominal nonempty inference helper has a separate error: it writes Å-valued
distances directly into an integer class-index tensor without bucketizing them before
one-hot encoding. Values of at least 4 Å exceed the four-class range. See
[`generate_spec_constraint`](https://github.com/bytedance/Protenix/blob/a0f95f62430093b46c677f04e3d292ccc5d28d64/protenix/data/constraint_featurizer.py#L2386-L2412).
This path is not reached by the released JSON interface: its `structure` branch is a
`TODO` and always emits an empty substructure feature.

### Impact

The checkpoint never learned the advertised first distance channel, and the source
does not define a usable nonzero inference contract. Reconstructing a public
substructure-conditioning API from these paths would invent semantics that the
release neither published nor implemented.

### OFoldX behavior

OFoldX does not expose nonzero substructure conditioning for this release. It still
passes an all-zero `[T, T, 4]` tensor and executes the complete released substructure
Transformer for every constraint-model inference. That execution is mandatory:
the trained bias and normalization parameters turn the all-zero input into a
nonzero pair baseline, so skipping the module would change the checkpoint function.

## PXDesign accepts incompatible Protenix releases

**Upstream:** `bytedance/PXDesign`

**Affected reference:** PXDesign's audited revision
[`f788441`](https://github.com/bytedance/PXDesign/tree/f788441313c84c3074fe9596ac2433f96b15c763),
including its released `pxdesign_v0.1.0.pt` inference path.

**Classification:** independently reproduced packaging and dependency-constraint
bug. This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

PXDesign declares `protenix>=0.1.0` without an upper bound. Its source actually
calls the dictionary-based `AtomAttentionEncoder` and `DiffusionModule` interfaces
present in Protenix 0.6.3. Protenix 0.7.3 changed those interfaces: atom features
became explicit arguments, and the diffusion call acquired the required
`pair_z`, `p_lm`, and `c_l` arguments.

Protenix 0.7.3 was released before the audited PXDesign revision. Consequently, a
fresh installation could already resolve a version inside PXDesign's declared
dependency range that its inference code could not call. Current Protenix 2.x has
the same incompatible expanded interface. The failure occurs during condition
embedding or the first diffusion step rather than checkpoint loading.

### Impact

This is an installation/runtime compatibility failure, not a scientific
checkpoint difference. The released 732-tensor checkpoint targets the PXDesign graph
constructed against Protenix 0.6.3. The maintained parity script compares its
condition embedding and one-step denoiser on the same canonical features; generated
reports remain local.

OFoldX's audit invoked the compatible upstream components directly to isolate
numerical behavior. That is an audit method, not an upstream-supported workaround
for the broken dependency range.

### OFoldX behavior

OFoldX implements the released PXDesign graph natively and does not import
Protenix at runtime. It therefore needs neither a version-dependent compatibility
branch nor a hidden provider switch. Reference validation uses Protenix 0.6.3 as
the compatible numerical baseline and records the exact upstream revisions used.

## PXDesign suite can save final Protenix coordinates with stale metrics

**Upstream:** `bytedance/PXDesign`

**Affected reference:** PXDesign's audited revision
[`f788441`](https://github.com/bytedance/PXDesign/tree/f788441313c84c3074fe9596ac2433f96b15c763),
using PXDesignBench
[`f6d0d72`](https://github.com/bytedance/PXDesignBench/tree/f6d0d72496c23ca1374df5943754f3454ae8c549).

**Classification:** independently reproduced inference-output bug. This
classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

The advertised extended pipeline initially evaluates Protenix with two diffusion
steps, selects candidates, and then reruns passing candidates with 20 steps before
final serialization. The result of `rerun_ptx` is assigned to `_` and never merged
back into the selected table. The final Protenix structure therefore comes from the
20-step prediction while `summary.csv` retains the earlier two-step metrics. See
[`process_extended_results`](https://github.com/bytedance/PXDesign/blob/f788441313c84c3074fe9596ac2433f96b15c763/pxdesign/runner/helpers.py#L409-L479)
and
[`rerun_ptx`](https://github.com/bytedance/PXDesign/blob/f788441313c84c3074fe9596ac2433f96b15c763/pxdesign/runner/helpers.py#L786-L851).

The mismatch was reproduced by injecting a rerun result with a distinct
`ptx_iptm`: the saved summary retained the initial value.

### Impact

Confidence and geometry values in the final table do not describe the final saved
coordinates. This can corrupt downstream filtering, reporting, and comparisons
that assume a structure and its metrics are one result.

### OFoldX behavior

Each final structure and its final metrics must come from the same fold. Screening
metrics may also be retained, but only under names that state they belong to the
screening prediction. OFoldX does not reproduce the upstream mismatch.

## PXDesign suite ignores mini-only Protenix results during ranking

**Upstream:** `bytedance/PXDesign`

**Affected reference:** PXDesign
[`f788441`](https://github.com/bytedance/PXDesign/tree/f788441313c84c3074fe9596ac2433f96b15c763)
with PXDesignBench
[`f6d0d72`](https://github.com/bytedance/PXDesignBench/tree/f6d0d72496c23ca1374df5943754f3454ae8c549).

**Classification:** independently reproduced ranking bug in the reachable custom
mini-only configuration. It does not affect the default extended preset, which
uses full Protenix. This classification does not claim upstream acknowledgement.

### Behavior

PXDesignBench defines only the strict `ptx_mini_success` filter for the mini model.
PXDesign's ranking resolver nevertheless requires all three columns
`ptx_mini_success`, `ptx_mini_basic_success`, and `ptx_mini_iptm` before it recognizes
mini output. No mini-basic filter produces the required middle column. See
[`resolve_ptx_columns`](https://github.com/bytedance/PXDesign/blob/f788441313c84c3074fe9596ac2433f96b15c763/pxdesign/runner/helpers.py#L108-L130)
and the
[`ptx_mini` filter definition](https://github.com/bytedance/PXDesignBench/blob/f6d0d72496c23ca1374df5943754f3454ae8c549/pxdbench/pxd_configs/eval.py#L84-L100).

The actual resolver was reproduced with `ptx_mini_success=True` and
`ptx_mini_iptm=0.91`; it classified the table as preview data and ignored the
available mini result.

### Impact

Mini-only Protenix evaluation can run successfully but then be omitted from
selection and final summary semantics.

### OFoldX behavior

If OFoldX publishes mini-only evaluation, ranking uses the filter that actually
exists. It does not invent a `mini-basic` result to satisfy the broken resolver.

## PXDesign suite exposes output controls that do not work

**Upstream:** `bytedance/PXDesign`

**Affected reference:** PXDesign
[`f788441`](https://github.com/bytedance/PXDesign/tree/f788441313c84c3074fe9596ac2433f96b15c763)
with PXDesignBench
[`f6d0d72`](https://github.com/bytedance/PXDesignBench/tree/f6d0d72496c23ca1374df5943754f3454ae8c549).

**Classification:** independently reproduced public configuration and output
serialization bugs. These findings do not claim upstream acknowledgement.

### Behavior

Two controls disagree with the serializer:

- `return_topk` is declared by both presets and the command-line parser but has no
  consumer. Actual result cardinality is governed by other selection limits. See
  [`presets.py`](https://github.com/bytedance/PXDesign/blob/f788441313c84c3074fe9596ac2433f96b15c763/pxdesign/runner/presets.py#L1-L27)
  and the
  [`--return_topk` declaration](https://github.com/bytedance/PXDesign/blob/f788441313c84c3074fe9596ac2433f96b15c763/pxdesign/runner/pipeline.py#L238-L246).
- PXDesignBench exposes multiple AF2 `model_ids`, and its evaluation path produces
  per-model outputs. PXDesign's final saver globs all matching model files and then
  requires exactly one match. The actual saver was reproduced raising
  `FileNotFoundError` when two configured model predictions were present. See the
  [`model_ids` configuration](https://github.com/bytedance/PXDesignBench/blob/f6d0d72496c23ca1374df5943754f3454ae8c549/pxdbench/pxd_configs/eval.py#L43-L53)
  and
  [`save_af2_docked`](https://github.com/bytedance/PXDesign/blob/f788441313c84c3074fe9596ac2433f96b15c763/pxdesign/utils/pipeline.py#L179-L205).

### Impact

The first option silently does nothing. The second makes a configuration accepted
by evaluation fail only during final output serialization. The released default
uses one AF2 model and does not trigger the latter failure.

### OFoldX behavior

OFoldX does not expose dead cardinality controls. A published multi-model AF2 path
must preserve and explicitly select among all predictions; otherwise its API
accepts exactly one model rather than accepting a list that fails at serialization.

## LigandMPNN inverts the documented single-residue `use_sequence` flag

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed public-semantics bug. This
classification does not claim upstream acknowledgement.

### Behavior

The scoring CLI says `--use_sequence=1` computes
`p(AA_i | backbone, all other residues)` and `0` uses backbone information only.
The implementation swaps those meanings. The `False` branch places the target
nominally last, so the always-active teacher-forced decoder sees earlier sequence
tokens and computes the conditional path. The `True` branch places the target
nominally first, so no sequence token is causally visible at that target and the
same decoder computes the unconditional path. See
[`single_aa_score`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L471-L558)
and the contradictory
[`--use_sequence` help](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/score.py#L527-L532).

The separately recorded unbounded-random-priority defect can still move a target
across either intended boundary. It is a second defect rather than the cause of
this systematic inversion.

### Impact

Both outputs are plausible normalized distributions, but each flag value returns
the opposite documented scientific computation. A caller can silently label
conditional probabilities unconditional or vice versa. The ordinary
autoregressive scoring path does not have this inversion.

### OFoldX behavior

OFoldX exposes `conditional_probabilities` and
`unconditional_probabilities` directly. Conditional evaluation puts each target
exactly last; unconditional evaluation puts that target exactly first. Both retain
the released repeated-evaluation axis, aligned `[E, L, 21]` logits, and the exact
`[E, L, L]` effective target ranks while preserving every non-target relative
order. OFoldX does not collapse the unconditional operation into one all-equal-rank
forward merely because some target-first values can coincide. No inverted Boolean
or compatibility mode survives.

## LigandMPNN symmetry groups crash when weights are omitted

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed command-to-Model default bug. The
released weighted symmetric sampler remains usable when every group has an
explicit same-length weight list. This classification does not claim upstream
acknowledgement.

### Behavior

The design CLI documents `--symmetry_residues` and `--symmetry_weights` as
separate optional arguments. When groups are supplied without weights, `run.py`
constructs the nonempty groups but sets the weight value to `[[]]`. The symmetric
sampler initializes every residue weight to one and then unconditionally replaces
each grouped member from that empty nested list, raising `IndexError` before any
sequence is sampled. See the
[`run.py` defaults](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L320-L360)
and the
[`sample` weight lowering](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L351-L356).

The sampler's initial all-one tensor establishes the ordinary omitted-member
default. The distinct homo-oligomer convenience path explicitly supplies
`1 / group_size` for each member; that averaging policy is not an implicit
default for arbitrary symmetry groups.

### Impact

A documented symmetric-design input fails unless the caller also discovers and
supplies a complete parallel weight structure. The failure is caused by the CLI
sentinel rather than an unsupported scientific operation.

### OFoldX behavior

`TiedResidueIdentities` expresses the symmetry groups directly. A missing
`tied_weights` mapping gives every tied member weight one, while supplied entries
override only their named members. The Model uses the released signed weighted
sum and does not normalize caller weights. Homo-oligomer averaging is expressed
explicitly with `1 / group_size`; OFoldX does not preserve the crashing empty-list
sentinel or add a second symmetry representation.

## LigandMPNN tied sampling applies only the last member's residue bias

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced tied-sampling implementation bug.
This classification does not claim upstream acknowledgement.

### Behavior

The tied sampler correctly forms a weighted sum of the group members' model
logits, but it overwrites `bias_t` while visiting the members and adds that bias
only after the loop. The effective residue bias therefore comes entirely from the
last member of each group and changes when the same group is written in another
order. See the
[`sample` tied branch](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L413-L452).

### Impact

Per-residue amino-acid bias on every tied member except the last is ignored. A
tied design can consequently ignore a preference encoded at another member, and
merely reordering the same group can change the result.

### OFoldX behavior

Symmetry weights apply only to neural logits. Per-residue bias must be elementwise
identical at every tied member after sparse defaults and is applied exactly once to
the group distribution. Unequal member biases are ambiguous evidence and fail at
the Processor boundary. OFoldX neither preserves last-member selection nor invents
a weighted-bias formula unsupported by the released operation.

## LigandMPNN gives tied-group members an autoregressive order

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced tied-decoder implementation bug.
This classification does not claim upstream acknowledgement.

### Behavior

The symmetric sampler expands every tied group into a flat decoding permutation
before constructing its causal mask. It then visits the members sequentially and
writes each member's decoder-layer state before evaluating the next member. A later
member can therefore attend to an earlier member's updated hidden state even though
the group identity has not yet been sampled. See the
[`sample` group expansion and decode loop](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L357-L452).

With otherwise identical random Model state, inputs, unit member weights, and zero
bias, reversing one two-member group from `[0, 1]` to `[1, 0]` changed its aggregate
sampling distribution by a maximum absolute probability of `0.00109355`. The
difference is separate from the last-member residue-bias bug: it remains when all
external soft controls are neutral.

### Impact

An unordered equality constraint acquires an artificial direction. Group member
order, which has no biological meaning, changes the probability of the shared
identity and can therefore change the sampled sequence.

### OFoldX behavior

All members of a tied group have the same causal rank. They see only earlier
groups, are evaluated from the same pre-publication decoder state, and publish one
shared sampled identity simultaneously. Missing member weights retain the released
ordinary default of one; OFoldX does not silently replace their signed sum with an
average. Homo-oligomer averaging remains an explicit `1 / group_size` choice.

## LigandMPNN homo-oligomer order depends on Python hash iteration

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed command-level determinism bug. This
classification does not claim upstream acknowledgement.

### Behavior

The `homo_oligomer` path constructs `chain_letters_set` as
`list(set(chain_letters_list))`, selects element zero as the reference chain, and
uses that same hash-dependent list as every tied group's member order. Python set
iteration is not insertion ordered and can differ across interpreter hash seeds.
The chosen reference chain and group order are therefore not controlled by the
LigandMPNN random seed. See the
[`run.py` homo-oligomer lowering](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L341-L358).

Because the released tied decoder evaluates members sequentially, the separately
recorded member-order defect makes this more than an output-order difference: two
processes given the same PDB and scientific seed can form a different conditional
distribution.

### Impact

Homo-oligomer design is not reproducible from the documented seed alone, and the
arbitrarily selected reference chain can change which residue-number set defines
the generated ties.

### OFoldX behavior

OFoldX expresses equality through explicitly ordered `TiedResidueIdentities` and
never derives a reference or member order through a set. All group members are
evaluated at one causal rank from the same pre-publication state, so their declared
order cannot change the shared distribution.

## LigandMPNN mixed fixed/design ties can return unequal identities

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed tied-sampling state-update bug. This
classification does not claim upstream acknowledgement.

### Behavior

After drawing one token for a tied group, the sampler walks its members in listed
order. At every fixed member it overwrites the shared temporary token with that
member's native identity, then immediately publishes the current value to that
member. A fixed member visited before a designed member anchors the remaining
group, but a designed member visited first keeps the sampled token while a later
fixed member receives its native token. The supposedly tied outputs can therefore
differ. Conflicting fixed members likewise resolve by traversal rather than by a
defined equality constraint. See the
[`sample` publication loop](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L452-L461).

### Impact

The same unordered biological constraint changes meaning when its members are
reordered, and a call can return a sequence that violates the requested tie without
raising or reporting the contradiction.

### OFoldX behavior

A fixed member anchors the complete tied group before any member is published.
Every fixed anchor must name the same exact generatable component identity;
otherwise the request is contradictory. All members receive the one anchored or
sampled identity simultaneously, independent of group traversal order.

## LigandMPNN random priorities do not enforce categorical decode order

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed autoregressive-ordering bug affecting
design and both score algorithms. This classification does not claim upstream
acknowledgement.

### Behavior

Design sampling and ordinary scoring intend fixed rows to precede designed rows,
but sort `(chain_mask + 0.0001) * abs(randn)`. Single-residue scoring intends the
current target to be categorically first or last, but uses the same construction
with a target mask. See the generated orders in
[`sample`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L218-L221),
[`score`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L580-L585),
and
[`single_aa_score`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L497-L510).

The random factor is unbounded and can be arbitrarily close to zero, so the nominal
priority categories overlap. For example, a fixed row with `abs(randn)=2` has key
`0.0002`, while a designed row with `abs(randn)=0.00001` has key
`0.000010001` and is decoded first. Ratios beyond roughly 10,001 can likewise move
a single-residue target across the intended first/last boundary.

### Impact

A designed token may fail to condition on native fixed context. A supposedly
sequence-conditioned single-residue distribution may omit sequence context, while
a backbone-only distribution may leak it. The scientific meaning changes under a
rare random crossing rather than only the requested mode.

### OFoldX behavior

Generated ranks partition fixed causal units before designed units exactly. For
single-residue scoring, the target is moved exactly after all other sequence
positions or before all of them while their relative order is preserved. Randomness
orders units only within a category; explicit caller ranks remain authoritative.

## Enabling LigandMPNN packing changes sequence-design randomness

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced composition RNG bug. This
classification does not claim upstream acknowledgement.

### Behavior

The design command seeds global Torch RNG before constructing Models. When packing
is enabled, it constructs the optional Packer before sequence sampling. The Packer
initializes every multi-dimensional parameter with Xavier random values and then
loads a checkpoint over them, so scientifically unused initialization advances the
same global stream consumed later by sequence decoding. See the
[`run.py` seed and optional construction](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L31-L35)
and
[`Packer` initialization](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L334-L336).

At seed 111, the first four baseline uniform draws were approximately
`[0.7156, 0.9140, 0.2819, 0.2581]`; after constructing the Packer they were
`[0.4293, 0.1964, 0.1923, 0.9805]`. The optional downstream stage can therefore
change upstream designed sequences before it is ever called.

### Impact

Turning on packing is not a pure post-design choice. A caller cannot compare packed
and unpacked runs at one seed while holding their designed candidates fixed.

### OFoldX behavior

Models are constructed and loaded before an operation begins. A composition seed
creates its operation generator afterward; every sequence-design draw happens
before packing advances the same generator. Merely including the already-loaded
Packer never changes design candidates.

## LigandMPNN fixed-side-chain mode moves observed fixed atoms

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced Packer realization bug. This is
separate from materializing unobserved atoms in fixed residues and does not claim
upstream acknowledgement.

### Behavior

With `repack_everything=False`, upstream extracts observed fixed torsions but then
reconstructs the complete atom14 coordinates from ideal rigid frames and literature
positions. The reconstructed array replaces the original feature coordinates, so
fixed atoms are not copied from the input. See
[`make_torsion_features`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L149-L223)
and the repeated
[`frame-to-atom14 reconstruction`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L108-L123).

On the bundled 1BC8 example with every residue fixed, 683 of 789 observed atoms
moved more than `1e-3 Å`; mean displacement was `0.0983 Å` and maximum displacement
was `0.8056 Å`. CA remained exact, while observed N, C, and O moved by as much as
`0.2214 Å`, `0.02275 Å`, and `0.04811 Å` respectively.

### Impact

A mode documented as preserving fixed side chains silently changes fixed backbone
and side-chain observations. The output is neither the source structure nor a
selected-only repack.

### OFoldX behavior

Unselected sites, including all source backbone coordinates, remain exact.
Selected packing overwrites only selected generated side-chain sites and appends
missing selected heavy atoms. OFoldX does not route fixed observations through
ideal-coordinate reconstruction.

## LigandMPNN cannot use zero as an explicit seed

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** confirmed command-line seed-sentinel bug. This classification
does not claim upstream acknowledgement.

### Behavior

The public argument accepts an integer seed and defaults to zero, but the command
tests `if args.seed`. An explicit zero is false and is replaced by a random NumPy
integer rather than being used. See the
[`run.py` seed branch](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L31-L37).

### Impact

Seed zero cannot reproduce an operation and is observationally indistinguishable
from asking for a random seed.

### OFoldX behavior

`seed=0` is the ordinary deterministic integer zero. `seed=None` means no explicit
seed. No truthiness sentinel or compatibility switch is retained.

## LigandMPNN cannot directly pack the sequence produced by its parser

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced dtype mismatch between the parser and
side-chain packer. This classification does not claim upstream acknowledgement.

### Behavior

The parser encodes residue identities as NumPy `int32`, and `featurize` preserves
that dtype when adding a batch dimension. The side-chain packer passes the resulting
tensor directly to `torch.nn.functional.one_hot`, which accepts only an integer index
tensor with PyTorch's `int64` dtype. Calling the released packer directly on parsed
features therefore raises `RuntimeError: one_hot is only applicable to index tensor`.
See the parser's
[`int32` sequence construction](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L826-L831)
and the packer's
[`one_hot` call](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L149-L164).

The official combined design-and-pack path does not expose the failure because it
replaces the parsed sequence with the sequence model's `int64` sample immediately
before packing. See
[`run.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L520-L532).

### Impact

The released side-chain model cannot be used directly to repack a parsed structure
through the otherwise natural `parse_PDB` → `featurize` → `pack_side_chains` path.
Its advertised combined sequence-design workflow remains reachable because that
workflow happens to replace the incompatible tensor first.

### OFoldX behavior

`MpnnPackerProcessor` emits residue indices as `torch.int64`, so direct
structure packing and packing after sequence design share one input contract. The
parity harness converts the reference parser's sequence to `int64` before invoking
the upstream packer, reproducing the dtype supplied by upstream's own working
combined workflow without modifying upstream source.

## LigandMPNN's documented zero-pack fast path returns no packing

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced documentation/control-flow mismatch.
This classification does not claim upstream acknowledgement.

### Behavior

The first side-chain example says that
`--number_of_packs_per_design 0` returns one fast packing solution; see the
[`README`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/README.md#L538-L550).
The implementation uses that value directly in both the generation
[`range`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L517-L535)
and the output-writing
[`range`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L613-L619).
With zero, neither loop executes and no packed structure is produced.

### Impact

The documented fast path silently completes without the result it promises. Zero
does not select a cheaper scientific algorithm; it only makes the output
cardinality empty.

### OFoldX behavior

`num_samples` is the positive number of independent packed Structures and defaults
to one. OFoldX does not give zero a hidden sentinel meaning. Denoising steps and
torsion proposals have their own explicit positive counts.

## LigandMPNN materializes unobserved atoms in fixed side chains

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced observation-mask bug in side-chain
packing. This classification does not claim upstream acknowledgement.

### Behavior

The parser records the actual per-atom observation mask as `xyz_37_m`; see
[`data_utils.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L802-L807).
When `repack_everything=False`, the packer derives fixed torsions from the parsed
coordinates but supplies the theoretical residue-type
`atom37_atom_exists` mask instead of `xyz_37_m`; see
[`sc_utils.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L179-L190).
Missing coordinates are stored as zeros, so they participate as if they had been
observed. The output mask is likewise rebuilt from theoretical atom existence and
the generated atom14 coordinates; see
[`sc_utils.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L201-L223).

### Impact

An unselected side chain with missing atoms can acquire torsions inferred from zero
coordinates and be emitted as a complete observed side chain. The output no longer
preserves which atoms were actually present in the input, even though that residue
was meant to remain fixed context.

### OFoldX behavior

Unselected sites retain their original coordinates and observation masks. Missing
unselected side-chain atoms remain unresolved rather than being materialized from
a theoretical atom-existence mask. Selected residues alone receive packed
side-chain coordinates.

## LigandMPNN's Packer KNN can drop a valid neighbor on an invalid-distance tie

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced nearest-neighbor masking defect. This
classification does not claim upstream acknowledgement.

### Behavior

The released Packer zeros invalid pair distances, takes the maximum remaining
distance in each row, then assigns that same maximum to every invalid pair before
`topk`. See
[`ProteinFeatures._dist`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L847-L857).
An invalid pair can therefore tie the farthest valid pair. `torch.topk` has no
stable tie-order guarantee: for the distance row `[1.0, 0.001, 1.0]`, where the
first maximum is invalid and the second is valid, the reproduced two-neighbor
selection returned indices `[1, 0]` and discarded valid index 2.

### Impact

A valid residue can receive fewer valid graph neighbors than requested whenever an
invalid row is present. Later attention masking removes the selected invalid edge
but cannot restore the valid edge that `topk` discarded. The resulting prediction
remains finite and plausible, and the chosen graph can vary with tie ordering.

### OFoldX behavior

`MpnnPackerFeatures` assigns invalid pair distances positive infinity before
nearest-neighbor selection. Every finite valid pair is therefore selected first.
When fewer valid rows than requested exist, the remaining invalid fill indices are
still excluded by the same boolean source-by-neighbor mask in both encoder and
decoder computation.

## LigandMPNN's Packer decoder attends to invalid residue neighbors

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed inference-mask omission. This
classification does not claim upstream acknowledgement.

### Behavior

The Packer encoder gathers the validity of each residue neighbor and passes the
source-by-neighbor attend mask to every encoder layer. Its decoder uses the same
neighbor graph and a decoder layer that accepts an attend mask, but calls each
layer with only the source-node mask. See the released
[`encode`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L337-L352),
[`decode`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L367-L385),
and the decoder layer's unused
[`mask_attend`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/model_utils.py#L1654-L1685)
argument.

An invalid neighbor's learned edge and sequence features therefore contribute to a
valid residue's decoder message. This is reachable whenever the fixed-size neighbor
graph contains an invalid row, including short or partially observed inputs where
the requested neighbor count reaches the row count.

### Impact

Missing-backbone rows can change otherwise valid side-chain distributions without
an error or an explicit missing-data signal. The output remains numerically
plausible, so ordinary shape and finiteness checks do not expose the contamination.

### OFoldX behavior

`MpnnPackerModel.decode` gathers the same boolean source-by-neighbor validity mask
as the encoder and passes it to every residue decoder layer. Invalid rows remain on
the explicit tensor axis but cannot contribute messages to valid residues.

## LigandMPNN's Packer erases valid unknown-residue backbones

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed unknown-token coordinate-mask defect.
This classification does not claim upstream acknowledgement.

### Behavior

The parser can retain a nonstandard protein occurrence with a valid N/CA/C/O
backbone and encode its identity as the Packer's unknown token. During torsion
initialization, that token maps to AlphaFold's unknown residue, whose atom14 and
atom37 existence masks are entirely false. The released code then multiplies the
reconstructed atom14 coordinates by that theoretical mask before encoding; see
[`make_torsion_features`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L151-L225).

The row-level residue mask remains true, but the Model receives zero N, CA, C, and
O coordinates for that row. Because both theoretical and gathered atom14 masks are
false, a missing-atom comparison cannot recognize the row as partial fixed state.

### Impact

A valid unselected modified residue becomes a zero-coordinate graph neighbor and
can distort distances and side-chain predictions for surrounding standard
residues. The computation still returns ordinary tensors.

### OFoldX behavior

For an unselected unknown-token protein row, `MpnnPackerModel.sample` places the
observed N, CA, C, and O coordinates and masks into atom14 slots 0–3. The existing
fixed-state path reapplies that observation after every denoising update. Unsupported
unknown side-chain atoms remain masked, and the checkpoint still cannot select an
unknown identity for packing.

## MpnnPacker leaks atomic identities when atom context is disabled

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently reproduced inference feature-construction bug in
the side-chain Packer path. This classification does not claim upstream
acknowledgement.

### Behavior

The released `featurize(..., use_atom_context=False)` path clears `Y_m` but retains
the selected atoms' `Y_t` values. See
[`data_utils.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L926-L957).

The Packer decoder masks atom-distance RBFs with `Y_m`, then concatenates
`one_hot(Y_t)` without applying that mask before its learned projection. See
[`sc_utils.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L1120-L1139).
Consequently the option described as masking all ligand atoms still exposes their
element identities. With the released checkpoint and a zero context mask, replacing
the retained atomic types by padding types changes the predicted mixture values;
the largest reproduced changes were 0.0335 for means, 14.6858 for concentrations,
and 0.1851 for mixture logits.

The upstream no-context fallback already uses both `Y_m=0` and `Y_t=0`, so the
padding representation supplies the unambiguous intended no-context baseline.

### Impact

Side-chain packing requested without non-protein atom context remains conditioned
on which ligand, ion, DNA, or RNA elements happened to be selected. The flag can
therefore change less of the scientific computation than its public meaning says.

### OFoldX behavior

`MpnnPackerProcessor(..., use_atom_context=False)` emits the ordinary padding
representation for every non-protein context row: a false mask and atomic number
zero. It preserves the molecules in the decoded Structure and does not remove fixed
protein side-chain context.

The maintained Packer parity script compares the released context-enabled path.
It reports this context-disabled behavior as an intentional exclusion rather than
patching upstream tensors to manufacture equality. No compatibility switch
preserves the identity leak.

## LigandMPNN silently drops molybdenum atom context

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed element-table typo in PDB lowering.
This classification does not claim upstream acknowledgement.

### Behavior

The released PDB parser's periodic element list spells molybdenum as `Mb` rather
than the chemical symbol `Mo`; see
[`data_utils.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L528-L648).
Elements absent from that list are assigned atomic number zero and then removed
from atom context; see the
[`Y_t` construction and mask](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L832-L847).
A standard PDB `Mo` atom is therefore silently treated as unknown padding rather
than atomic number 42.

### Impact

Protein environments containing a molybdenum ion or molybdenum-bearing cofactor
lose that atom from both LigandMPNN and side-chain-Packer context. The computation
remains plausible, so the missing chemical evidence is not exposed by a tensor
shape or dtype failure.

### OFoldX behavior

OFoldX's canonical element table uses the standard symbol `Mo`, and
`LigandMpnnProcessor` and `MpnnPackerProcessor` retain it as atomic number 42.
Unknown element symbols fail during lowering instead of being silently recast as
padding. OFoldX provides no `Mb` compatibility alias.

## LigandMPNN's `force_hetatm` option does not force HETATM records

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed CLI and writer implementation
mismatch. This classification does not claim upstream acknowledgement.

### Behavior

The CLI describes `--force_hetatm` as forcing ligand atoms to be written as
`HETATM`; see
[`run.py`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L962-L970).
When enabled, the writer instead copies each atom's existing `hetatm` flag back
onto the output group; see
[`write_full_PDB`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L477-L484).
The option therefore leaves record categories unchanged.

### Impact

A caller can explicitly request HETATM output and receive ordinary `ATOM` records
for non-protein atoms whose input flags were false. Downstream PDB tools may then
classify those atoms differently despite the apparent successful completion of
the command.

### OFoldX behavior

The canonical `Structure` and `PackResult` do not encode a false successful
`force_hetatm` operation. PDB exchange writing remains a separate I/O contract;
when that contract is frozen, it must either implement an explicit record-category
choice or omit the option. OFoldX does not preserve this no-op as a compatibility
flag.

## LigandMPNN's packed-PDB B column is not chi density for fixed or chi-free atoms

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed inference-output serialization bug.
This classification does not claim upstream acknowledgement.

### Behavior

The released README states that the packed-PDB B factor shows log-probability
density per chi-angle group. The Packer instead replaces every fixed row's four
densities with the literal value `2.0`. It then clamps atom14 rigid groups below
four to group four before selecting one of the four density heads. See
[`pack_side_chains`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/sc_utils.py#L128-L140).

Consequently all fixed atoms receive `2.0`, regardless of any probability
distribution. Backbone atoms receive the chi-one head. Selected ALA and GLY also
receive that head even though those residues have no physical chi angle and the
head does not control their coordinates.

### Impact

The B column mixes physical final-sample log densities, a fixed-row sentinel, and
values from an unused output head. A viewer cannot interpret the column according
to the documented meaning, and preserving it would require task output to retain
pack/fixed state plus nonphysical ALA/GLY values after all scientific chi metrics
have already been decoded.

### OFoldX behavior

`PackPrediction` retains only physical `chi1_log_density` through
`chi4_log_density` Metrics. The family packed-PDB writer maps those values only to
atoms in the corresponding real chi rigid group and writes zero for fixed,
backbone, chi-free, and nonprotein atoms. OFoldX does not add a raw B-factor payload
or compatibility switch for the released sentinel and unused-head values. Manual
parity reports identify this as an intentional bug-fix divergence.

## LigandMPNN writes residue initials as packed-PDB protein elements

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed PDB serialization bug. This
classification does not claim upstream acknowledgement.

### Behavior

While constructing the packed protein AtomGroup, the writer assigns every atom the
first character of the three-letter residue name rather than the atom's element;
see
[`write_full_PDB`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L448-L453).
For example every atom of ARG receives `A`, and every atom of GLU receives `G`,
including nitrogen, oxygen, sulfur, and carbon sites.

### Impact

The PDB element column can contain invalid symbols or a valid but chemically wrong
element. Downstream chemistry and visualization software may misclassify atoms even
though their names and coordinates are otherwise correct.

### OFoldX behavior

The family writer obtains the element from the exact canonical `ChemicalAtom` for
each `AtomSite`. It never infers an element from a residue or atom-name prefix.

## LigandMPNN's packed-PDB insertion-code default is unusable

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de).

**Classification:** independently confirmed public-function default bug. This
classification does not claim upstream acknowledgement.

### Behavior

`write_full_PDB` declares `icodes=None`, but its protein loop unconditionally reads
`icodes[i]`; see
[`write_full_PDB`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L236-L258)
and the unconditional access at
[`lines 448-457`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L448-L457).
Calling the advertised default therefore raises instead of writing a PDB.

### OFoldX behavior

The family writer consumes a complete `PackPrediction`. It uses a supplied
canonical residue insertion code when present and an ordinary blank code when no
source number fact exists; it has no redundant insertion-code argument.

## ODesign writes generated ligand atoms with zero occupancy

**Upstream:** `OTeam-AI4S/ODesign`

**Affected reference:** ODesign revision
[`cc95c2f`](https://github.com/OTeam-AI4S/ODesign/tree/cc95c2f8f915af4d87b1db3f44c4db2df8566a41).

**Classification:** independently reproduced inference-output serialization bug.
This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

ODesign creates each generated ligand atom with `condition_token_mask=False` and
appends those atoms to the design assembly. See the
[`ligand` placeholder construction](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/inference/inference_utils.py#L56-L69)
and the
[`build_from_chain_sequence` generated branch](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/inference/inference_utils.py#L146-L168).

The reachable output path passes that `AtomArray` to `save_structure_cif`, whose
writer replaces `_atom_site.occupancy` with the integer-valued condition mask. See
the dumper's
[`save_structure_cif` call](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/inference/dumper.py#L141-L147)
and the writer's
[`occupancy` assignment](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/data/misc.py#L434-L444).
Consequently every generated ligand slot is written with occupancy `0`, although
the file contains a generated coordinate for it. In PDBx/mmCIF, occupancy is the
[fraction of the atom type present at a site](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_atom_site.occupancy.html),
not a conditioning indicator.

### Impact

The mmCIF misstates generated ligand atoms as having zero occupancy. Consumers that
honor occupancy may treat the predicted atoms as absent or down-weight them, and the
file no longer faithfully represents the generated structure.

### OFoldX behavior

The typed ODesign expert output preserves generated coordinates and the condition
mask as separate facts. OFoldX does not serialize condition status as occupancy.
Parity records the inspected upstream CIF behavior as an intentional serialization
divergence; any generated report remains local.

## ODesign truncates the permitted `RARE` ligand class to `RA`

**Upstream:** `OTeam-AI4S/ODesign`

**Affected reference:** ODesign revision
[`cc95c2f`](https://github.com/OTeam-AI4S/ODesign/tree/cc95c2f8f915af4d87b1db3f44c4db2df8566a41).

**Classification:** independently reproduced ligand inference-output serialization
bug. This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

The ligand tokenizer defines `RARE` as a model output class, and the released
inference path explicitly permits and decodes that class. See the
[`LigandTokenizer` vocabulary](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/modules/invfold/src/datasets/featurizer_ligand.py#L17-L31)
and the
[`allow_rare=True` decoding path](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/modules/invfold/evaluation_tools/tools.py#L1132-L1164).

The main inference runner forwards the decoded token to `DataDumper`, which copies
`atom_array.element` with its existing string width and assigns the upper-case token
into that array. See the
[`InferRunner` ligand path](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/inference/infer_runner.py#L203-L259)
and the dumper's
[`element` patch](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/inference/dumper.py#L157-L195).
ODesign pins Biotite 1.0.1 in its
[`requirements.txt`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/requirements.txt#L26-L28),
whose `AtomArray.element` annotation has fixed NumPy dtype
[`U2`](https://github.com/biotite-dev/biotite/blob/0f02222e3cf1be5abd2140f67463761b9d1ee261/src/biotite/structure/atoms.py#L44-L53).
Assigning `RARE` therefore stores `RA`, and that truncated value is what the main
writer serializes.

A separate module contains a helper that would map `RARE` to no element, but that
helper is not used by this path. The reachable runner imports `inference` from
`evaluation_tools.tools` and writes through `DataDumper`; compare its
[`imports`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/inference/infer_runner.py#L14-L21)
with the unused
[`_normalize_new_elems`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/modules/invfold/evaluation_tools/inference_utils.py#L798-L808).

### Impact

The output loses the model's unknown-element-category result and writes `RA` as if
it were an element symbol. A downstream chemistry reader can consequently interpret
the generated atom as radium rather than as the model's `RARE` class.

### OFoldX behavior

The typed ODesign expert output preserves the decoded `RARE` class and its logits as
model facts. OFoldX does not coerce that class into a fixed-width element field or
serialize it as `RA`. Parity records the faulty upstream writer as an intentional
serialization divergence; any generated report remains local.

## ODesign drops its computed distogram from the returned inference output

**Upstream:** `OTeam-AI4S/ODesign`

**Affected reference:** ODesign revision
[`cc95c2f`](https://github.com/OTeam-AI4S/ODesign/tree/cc95c2f8f915af4d87b1db3f44c4db2df8566a41).

**Classification:** independently reproduced inference-output aggregation bug.
This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

For every model seed, `_main_inference_loop` runs the pairwise head, places
`pairwise_output.distogram` in its `ODesignOutput`, and returns that object. See the
[`pairwise-head update`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/odesign.py#L515-L535)
and the
[`_main_inference_loop` return](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/odesign.py#L571-L588).

`main_inference_loop` then aggregates coordinates, bond logits, and the bond mask
into a new `ODesignOutput`, but omits `distogram`. See the complete
[`model-seed aggregation`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/odesign.py#L442-L461).
The public output type defines the omitted field with a default of `None`, so the
core-computed tensor is absent from the object received by the inference runner.
See
[`ODesignOutput`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/api/model_interface.py#L498-L546).

### Impact

Released inference computes the distogram head but callers cannot retrieve its
result. The head consumes compute and memory while its prediction is discarded at
the model-seed aggregation boundary.

### OFoldX behavior

The typed ODesign expert output preserves the computed distogram alongside
coordinates and bond predictions. OFoldX does not reproduce the faulty aggregation.
Parity compares the head output before the upstream omission and records the absent
upstream return field as an intentional divergence; any generated report remains
local.

## Boltz-2 affinity exposes conflicting diffusion-sample defaults

**Upstream:** `jwohlwend/boltz`

**Affected reference:** Boltz revision
[`c46ac60`](https://github.com/jwohlwend/boltz/tree/c46ac60c128875eac7ecc722937894edb4162f82).

**Classification:** independently confirmed public-interface default conflict. This
classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

The released Click option declares `--diffusion_samples_affinity` with default `5`,
and the prediction documentation also lists `5`. See the
[`click.option` declaration](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/main.py#L1002-L1007)
and the
[`prediction` option table](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/docs/prediction.md#L158-L161).
The decorated Python callback signature instead declares
`diffusion_samples_affinity=3`; see the
[`predict` signature](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/main.py#L1041-L1055).

Consequently a normal CLI call that omits the option receives `5`, while reading or
calling the underlying Python signature suggests `3` for the same omitted argument.

### Impact

The two entry points can perform different amounts of stochastic affinity sampling
without the caller supplying a value. This changes runtime, memory use, and the set
from which the maximum-iPTM affinity structure is selected.

### OFoldX behavior

OFoldX follows the released CLI and user documentation: the public affinity Pipeline
default is `5`. Direct tensor methods retain explicit family-level sampling controls
rather than inheriting the contradictory callback default. The real-checkpoint
parity script passes the count explicitly and also records the omitted-CLI default.

## ODesign inverse-fold scores change scale when beam search is enabled

**Upstream:** `OTeam-AI4S/ODesign`

**Affected reference:** ODesign revision
[`cc95c2f`](https://github.com/OTeam-AI4S/ODesign/tree/cc95c2f8f915af4d87b1db3f44c4db2df8566a41).

**Classification:** independently confirmed inference-output scale inconsistency.
This classification records OFoldX's technical finding; it does not claim an
acknowledgement from the upstream maintainers.

### Behavior

The ligand inverse-fold model produces ordinary per-position log probabilities.
Without beam search, inference reports the sum of the selected token log
probabilities. See the
[`greedy` score](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/modules/invfold/evaluation_tools/tools.py#L1144-L1154).

With beam search, candidate construction accumulates the same sum but then divides
it by the fixed sequence length before returning it. See
[`beam_search_logp`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/modules/invfold/evaluation_tools/tools.py#L316-L351)
and the
[`beam` inference branch](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/model/modules/invfold/evaluation_tools/tools.py#L1155-L1168).
Both values are returned through the same `scores` slot and retained under the same
per-chain key by the main inference runner; see
[`InferRunner.predict`](https://github.com/OTeam-AI4S/ODesign/blob/cc95c2f8f915af4d87b1db3f44c4db2df8566a41/src/utils/inference/infer_runner.py#L200-L256).

Consequently the same element assignment has a score differing by a factor equal to
its atom count solely because beam search was toggled.

### Impact

Scores are not comparable between greedy and beam inference or across different
generated atom counts. A caller can mistake a change in normalization for a change
in model likelihood.

### OFoldX behavior

`ODesignLigandCandidate.log_probability` always stores the summed token log
probability. Greedy output already has that meaning. For beam output, OFoldX gathers
and sums the selected token log probabilities, equivalently multiplying the upstream
score by the fixed candidate length. This preserves beam ordering while giving one
field one scale. The real-checkpoint parity script compares greedy directly and beam
after this documented normalization.

## RDKit silently drops declared allene chirality from SMILES

**Upstream:** `rdkit/rdkit`

**Affected dependency:** RDKit
[`2025.09.3`](https://github.com/rdkit/rdkit/releases/tag/Release_2025_09_3),
including the version required by the inspected Protenix revision
[`c3bfc365`](https://github.com/bytedance/Protenix/blob/c3bfc365b3e1341a11935eddfe7bfdc308092147/requirements.txt#L14).

**Classification:** independently reproduced standards-level silent semantic-loss
bug. Protenix, Boltz, and BoltzGen inherit the dependency behavior when they lower
SMILES through RDKit; this is not a claim about their learned weights. The
classification does not claim acknowledgement from the upstream maintainers.

### Behavior

OpenSMILES defines the allene stereochemical class `AL`, including the two
configurations `@AL1` and `@AL2`; see the
[`OpenSMILES specification`](https://opensmiles.org/opensmiles.pdf).

In RDKit 2025.09.3, both of these valid inputs parse successfully:

```text
FC=[C@AL1]=C(Cl)Br
FC=[C@AL2]=C(Cl)Br
```

For both inputs, every atom has `CHI_UNSPECIFIED`, every bond has `STEREONONE`, and
isomeric serialization returns the same achiral string:

```text
FC=C=C(Cl)Br
```

The shorthand allene forms using `@` and `@@` are lost in the same way. The parser
raises no error or warning that the declared stereochemical distinction has been
discarded.

### Impact

Two distinct allene stereoisomers collapse to one canonical graph before a model or
converter can preserve their identity. Downstream conformer generation and model
features can therefore operate on an unintended or unspecified stereochemical input
while the caller receives no indication that meaning was lost.

### OFoldX behavior

OFoldX does not claim support for a descriptor that its audited chemistry path cannot
retain. SMILES import rejects declared allene chirality, including explicit
`@AL1`/`@AL2` and their allene shorthand forms, before RDKit can erase it. The
canonical core does not add an unused `AL` abstraction. Support begins only when an
audited importer and at least one released consumer can round-trip and lower the
descriptor without semantic loss.

## ProteinMPNN tied sampling uses only the last member's controls

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced tied-sampling implementation bug.
This classification does not claim upstream acknowledgement.

### Behavior

The tied sampler aggregates model logits from every member of a tied group, but the
loop variable `t` escapes the member loop. Residue bias, PSSM coefficient and
probability, PSSM log-odds mask, per-residue omission, and the fixed/design decision
therefore all come from the last member. See the member aggregation and subsequent
uses of `t` in
[`tied_sample`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1238-L1287).

Reversing the same two-member group changes which controls take effect. A residue
identity forbidden on the first member remains sampleable, while the same
restriction on the last member is enforced. A fixed member can likewise be
overwritten when a later member is designable.

### Impact

A sequence reported as tied can violate the constraints of any member except the
last. The result depends on an incidental list order that has no biological meaning,
and a fixed residue does not reliably anchor its group.

### OFoldX behavior

OFoldX treats a tied group as one constrained random variable. In canonical-row
order it averages the signed-weighted member neural logits, without normalizing by
the weight sum. Tied weights affect no other evidence. Residue bias, PSSM
coefficient, and PSSM probabilities must be one elementwise group-constant soft
control and are applied once; the Processor rejects unequal member values rather
than guessing an aggregation. Thresholded PSSM log odds and all other hard support
intersect across members, any fixed member anchors the group, and fixed identities
must agree. Hard support is final, so signed weights cannot erase or invert it.

## ProteinMPNN gives tied-group members an autoregressive order

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced tied-decoder implementation bug. This
classification does not claim upstream acknowledgement.

### Behavior

The tied sampler first expands each tied group into an ordinary flat decoding
permutation, then builds its causal mask from that flattened order. A later member
can therefore attend to an earlier member even though the shared identity has not
yet been published. Merely reversing the members changes the returned probabilities
with all external controls neutral. See the group expansion and mask construction in
[`tied_sample`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1203-L1223).

The same block derives the group order only from batch row zero and repeats it over
the whole batch. Random order values supplied for other candidates do not determine
their own group order.

### Impact

An unordered equality constraint acquires an artificial within-group direction, and
batched candidates do not receive the independent decoding orders implied by their
random inputs. Group member order can change the sampling distribution.

### OFoldX behavior

All members of a tied group have the same causal rank. They see only earlier groups,
are evaluated from the same pre-publication state, and publish the shared sampled
identity simultaneously. Each candidate constructs its own group order. Real-weight
parity records the intentional divergence and checks member-permutation invariance.

## ProteinMPNN's PSSM threshold leaks forbidden identities

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced probability-mask bug, also reported by
the still-open upstream
[`PR #56`](https://github.com/dauparas/ProteinMPNN/pull/56).

### Behavior

After multiplying probabilities by the PSSM log-odds mask, both ordinary and tied
sampling add `0.001 * probabilities` before normalization. A forbidden identity is
therefore assigned nonzero probability and can even become probability one when it
was the only identity with nonzero pre-mask mass. See the ordinary
[`sample`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1171-L1179)
and
[`tied_sample`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1273-L1281)
branches.

### Impact

The documented PSSM threshold is not a hard restriction. Designed sequences can
contain identities whose log odds are below the caller's threshold.

### OFoldX behavior

PSSM log-odds support is hard. OFoldX applies the final support mask and normalizes
once without probability leakage. Empty support fails explicitly because silently
sampling a forbidden identity would be a plausible but scientifically wrong result.

## ProteinMPNN's PSSM mixture can restore globally omitted identities

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced sampling-constraint ordering bug. This
classification does not claim upstream acknowledgement.

### Behavior

The sampler removes globally omitted amino acids in the model softmax, then mixes
the resulting probabilities with the PSSM distribution. It never reapplies the
global support after that mixture. With a full PSSM mixture, a PSSM concentrated on
a globally omitted identity restores that identity to probability one. See the
softmax followed by PSSM mixing in
[`sample`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1164-L1175).

### Impact

The global omission argument does not define the final sampling support whenever a
PSSM is active. A sequence can contain an identity the caller excluded globally.

### OFoldX behavior

OFoldX first constructs and mixes the model and PSSM probability distributions,
then applies the intersection of global, per-residue, tied-member, and PSSM
log-odds support and normalizes once. Soft evidence can change relative probability;
it cannot reintroduce a hard-forbidden identity.

## ProteinMPNN silently truncates requested sequence counts

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced command-line batching bug. This
classification does not claim upstream acknowledgement.

### Behavior

The released command computes the number of batches with integer floor division:
[`num_seq_per_target // batch_size`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_run.py#L60-L61).
Every generation and repeated-scoring loop uses that value. A non-divisible request
drops the remainder; a request smaller than the batch size runs zero batches.

### Impact

The command can return fewer sequences or score evaluations than explicitly
requested without raising an error. Output cardinality changes with a compute-only
batch-size choice.

### OFoldX behavior

`num_candidates` and `num_decoding_orders` specify exact scientific cardinality.
`batch_size` only chunks the computation, and the final chunk may be smaller. The
returned count never changes merely because the requested count is not divisible by
the compute batch size.

## ProteinMPNN alternate FASTA scoring retains a stale suffix

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced alternate-sequence scoring bug. This
classification does not claim upstream acknowledgement.

### Behavior

The released `--score_only --path_to_fasta` path creates one mutable sequence tensor,
then overwrites only the prefix covered by each FASTA sequence. It neither constructs
a fresh tensor nor requires an exact length. Positions beyond a short sequence keep
the native sequence or the preceding alternate sequence. See the in-place prefix
assignment in
[`protein_mpnn_run.py`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_run.py#L239-L279).

### Impact

The reported alternate-sequence score can describe a hybrid sequence that the user
never supplied. Its suffix may also depend on which FASTA record was scored before
it.

### OFoldX behavior

Every alternate sequence is lowered independently and must match the exact canonical
protein positions being scored. No mutable tensor is reused across alternatives, so
the score and returned target always describe the caller's complete sequence.

## ProteinMPNN conditional probabilities do not enforce target order

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced conditional-decoder ordering bug.
This classification does not claim upstream acknowledgement.

### Behavior

For every target residue, `conditional_probs` intends to place that target after
all other residues when sequence context is enabled, or before all other residues
when `backbone_only=True`. It instead sorts
`(order_mask + 0.0001) * abs(randn)`. The random factor is unbounded and can be
arbitrarily close to zero, so the two nominal priority groups overlap rather than
having an exact order. See the mask and ordering construction in
[`conditional_probs`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1323-L1334).

For ordinary conditional probabilities, a non-target value greater than roughly
10,001 times the target value places that non-target after the target. For
backbone-only probabilities, the reverse ratio can place a non-target before the
target. Both events have nonzero probability under `abs(randn)`.

### Impact

An ordinary conditional probability can omit part of the intended sequence
context. A backbone-only probability can leak sequence context. The reported
meaning therefore depends on a rare random priority crossing rather than only on
the requested conditioning mode.

### OFoldX behavior

Target order is exact: the target follows every sequence-context position for
ordinary conditional probabilities and precedes every sequence-context position
for backbone-only probabilities. Randomness may order positions within the other
group, but it cannot cross the target boundary.

## ProteinMPNN fixed residues are not guaranteed to precede designed residues

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced autoregressive-ordering bug. This
classification does not claim upstream acknowledgement.

### Behavior

When no exact decoding order is supplied, `forward`, `sample`, and `tied_sample`
intend to decode fixed context before designed variables. They sort
`(design_mask + 0.0001) * abs(randn)`, so their nominal priority groups overlap.
See the generated orders in
[`forward`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1080-L1086),
[`sample`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1118-L1123),
and
[`tied_sample`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1203-L1220).

For example, a fixed row with `abs(randn)=2` receives key `0.0002`, while a
designed row with `abs(randn)=0.00001` receives key `0.000010001`. The designed
row is therefore placed first.

### Impact

The designed row cannot condition on that fixed row's native sequence token. The
same fixed/design partition can produce different causal contexts solely because
the random values cross an approximate priority boundary.

### OFoldX behavior

Generated decoding orders partition fixed context before designed variables
exactly, then randomize only within each partition. An explicitly supplied exact
decoding order remains authoritative. Random scaling is never used as a substitute
for categorical priority.

## LigandMPNN and NA-MPNN report perfect confidence for an empty scope

**Upstream:** `dauparas/LigandMPNN` and `baker-laboratory/NA-MPNN`

**Affected references:** LigandMPNN
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de)
and NA-MPNN
[`aafa32c`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

**Classification:** independently reproduced decoded-score bug. This classification
does not claim upstream acknowledgement.

### Behavior

Both implementations compute the mean sequence negative log-likelihood as

```python
sum(loss_per_residue * mask) / (sum(mask) + 1e-8)
```

and publish `exp(-mean_loss)` as confidence. See LigandMPNN
[`get_score`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/data_utils.py#L219-L233)
and NA-MPNN
[`get_score`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/data_utils.py#L36-L52).
When no residue belongs to the selected design or ligand-interface scope, the
numerator is zero and the result becomes `exp(0) == 1.0`.

### Impact

Absence of evaluated residues is reported as maximum confidence. Sequence recovery
on the same empty scope uses ordinary zero-over-zero arithmetic and becomes `NaN`,
so the two published summaries also disagree about whether the empty evaluation is
defined.

### OFoldX behavior

An empty selected or interface scope remains explicit: `scope=()`, and its mean,
confidence, and recovery are `NaN`. OFoldX adds no epsilon, fallback, or special
exception that turns missing evidence into a favorable score.

## NA-MPNN specificity output marks fixed rows as predicted profiles

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN
[`aafa32c`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

**Classification:** independently reproduced specificity-output mask bug. This
classification does not claim upstream acknowledgement.

### Behavior

The sampler multiplies `sampling_probs` by `chain_mask`, so fixed rows contain an
all-zero vector. The specificity writer averages those arrays but stores the
structural validity `mask`, not `chain_mask`, beside `predicted_ppm`; see
[`model_utils.py`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/model_utils.py#L208-L218)
and
[`run.py`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/run.py#L427-L441).

### Impact

A downstream consumer using the saved mask can treat an all-zero vector at a fixed
residue as a present probability distribution. The vector is neither normalized nor
a scientific prediction.

### OFoldX behavior

Specificity Metric axes contain selected valid nucleic-acid positions only. Fixed
rows are absent rather than represented by a zero probability vector. The typed raw
sampling output may retain its exact tensor mask for expert callers.

## NA-MPNN's specificity evaluator reads the wrong RNA probability columns

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN
[`aafa32c`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

**Classification:** independently reproduced RNA specificity-evaluation bug. This
classification does not claim upstream acknowledgement.

### Behavior

With the released default shared-token mode, the residue-to-index mapping rewrites
RNA A/C/G/U to the DNA-token columns 21–24. `compute_human_readable_ppm` uses that
mapping and is correct. `score_specificity_prediction` instead creates a boolean
mask by testing the original 33-entry token-name list for literal RNA names, which
selects legacy columns 26–29; see the shared mapping in
[`na_eval_utils.py`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/evaluation/na_eval_utils.py#L642-L690)
and the evaluator mask at
[`na_eval_utils.py`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/evaluation/na_eval_utils.py#L6004-L6012).

### Impact

The released RNA specificity evaluation compares reference profiles with columns
that do not contain the shared-token RNA predictions, so its MAE, RMSE, and
cross-entropy do not evaluate the profile emitted by the default inference path.

### OFoldX behavior

RNA probability Metrics use the canonical A/C/G/U identities and the actual shared
sampling columns. Offline evaluation aligns those named Metrics rather than
reconstructing token columns from a stale literal list.

## NA-MPNN permits chemically incompatible token classes at designed positions

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN
[`aafa32c`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

**Classification:** independently reproduced sampling-support bug. This
classification does not claim upstream acknowledgement.

### Behavior

The released command's default `--omit_AA` excludes only `X`; shared-token mode
also appends the legacy RNA characters. The sampler later removes unknown and
special tokens, but it does not restrict protein rows to amino-acid tokens or
nucleic-acid rows to their chemically compatible bases. See
[`run.py`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/run.py#L227-L234)
and
[`model_utils.py`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/model_utils.py#L198-L211).

### Impact

Probability mass may remain on token classes that cannot be decoded as the
polymer chemistry at that canonical position. A specificity profile can therefore
sum to less than one over the four chemically meaningful bases, and a sampled token
may not define a valid replacement component.

### OFoldX behavior

The final sampling support intersects caller restrictions with the concrete
position's polymer-compatible component identities before normalization. Soft bias
cannot restore an incompatible class. This is a scientific support rule, not a
fallback decoder or provenance-dependent repair.

## ProteinMPNN exposes width combinations its graph cannot execute

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced architecture-constructor bug. This
classification does not claim upstream acknowledgement.

### Behavior

`ProteinMPNN.__init__` exposes `node_features`, `edge_features`, and `hidden_dim`
as independent arguments. It passes `node_features` into the first positional
width of `ProteinFeatures`/`CA_ProteinFeatures`, so the feature graph emits that
width, while `W_e` accepts `edge_features` and the encoder node state/layers assume
`hidden_dim`; see
[`protein_mpnn_utils.py`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py#L1024-L1055).
The CA graph additionally constructs `W_v`, `node_embedding`, and `norm_nodes`, but
never reads them in forward computation.

### Impact

Non-equal advertised widths fail at ordinary matrix or message-passing shape
boundaries. Five CA checkpoint tensors are dead state. Treating those arguments as
independent architecture axes creates configs that cannot represent an executable
released Model.

### OFoldX behavior

`ProteinMpnnConfig` stores one `hidden_size`. Conversion maps every live tensor and
explicitly discards only the five proven-unused CA tensors. It does not retain
false width axes or dead Modules for checkpoint-shape compatibility.

## ProteinMPNN input scoring changes with generation batch size

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
[`8907e66`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).

**Classification:** independently reproduced decoded-score cardinality bug. This
classification does not claim upstream acknowledgement.

### Behavior

The command clones one target `batch_size` times, draws one decoding order per
clone, computes the native input scores over those clones once, and prints their
mean. The number of input-sequence evaluations is therefore `batch_size`, even
though that option is documented as a memory/performance control for generation;
see
[`protein_mpnn_run.py`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_run.py#L58-L61)
and
[`protein_mpnn_run.py`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_run.py#L320-L337).

### Impact

Changing only a generation chunk size changes the number of random evaluations,
their arithmetic mean, and the reported input score. The value is not invariant to
an operational batching choice and is coupled to candidate generation cardinality.

### OFoldX behavior

`ProteinMpnnModel.score` owns an explicit `num_decoding_orders`. Design records one
input `ScoreResult` using that scientific count independently of candidate count or
execution chunking. No batching option silently changes the score definition.

## Boltz YAML parsing merges differently modified or cyclic polymers

**Upstream:** `jwohlwend/boltz`

**Affected reference:** Boltz revision
[`c46ac60`](https://github.com/jwohlwend/boltz/tree/c46ac60c128875eac7ecc722937894edb4162f82).

**Classification:** independently reproduced input-grouping bug. This classification
does not claim upstream acknowledgement.

### Behavior

The YAML parser says it groups entries with the same type, sequence, and
modifications, but the actual key contains only `(entity_type, sequence)`; see
[`schema.py`](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/parse/schema.py#L1009-L1035).
After grouping, polymer construction reads `modifications` and `cyclic` only from
the first entry in the group; see
[`schema.py`](https://github.com/jwohlwend/boltz/blob/c46ac60c128875eac7ecc722937894edb4162f82/src/boltz/data/parse/schema.py#L1155-L1171).

The documented YAML contract instead describes one entry per unique chain or
molecule and uses an ID list for multiple identical entities. It also exposes
`modifications` and `cyclic` as part of a polymer entry.

### Impact

Two entries with the same raw sequence but different residue modifications or
cyclic state are collapsed into one entity. Every chain in the group receives the
first entry's chemistry, so the second entry's declared molecular identity is lost
before feature construction.

### OFoldX behavior

Canonical entity identity uses the complete stored `PolymerSequence` value. A
Boltz Processor may reproduce its documented grouping rule, but it never groups
entries using raw sequence alone when their modifications or cyclic chemistry
differ. OFoldX does not preserve this upstream parser defect as a canonical rule.

## BoltzGen relaxes explicit residue constraints when their intersection is empty

**Upstream:** `HannesStark/boltzgen`

**Affected reference:** BoltzGen revision
[`31d9d9b`](https://github.com/HannesStark/boltzgen/tree/31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0).

**Classification:** independently reproduced inverse-folding constraint bug. This
classification does not claim upstream acknowledgement.

### Behavior

BoltzGen combines the caller's per-residue forbidden-amino-acid mask with its
global `inverse_fold_avoid` list. If the intersection leaves no valid amino acid at
a position, the released helper warns and discards the complete per-residue mask at
that position before trying again. See
[`build_constraint_logit_mask`](https://github.com/HannesStark/boltzgen/blob/31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0/src/boltzgen/model/modules/inverse_fold.py#L42-L108).

The same helper also warns and ignores a per-residue mask whose shape is wrong.
Both branches turn explicit user constraints into a different operation instead of
failing.

### Impact

A residue may be sampled as an identity the caller explicitly prohibited. The
warning is not a scientific fallback: there is no principled precedence between
the per-residue and global exclusions, and choosing one silently changes the
design problem.

### OFoldX behavior

OFoldX preserves both explicit exclusion sources and raises when their union blocks
every amino acid at any position. It never relaxes or ignores the supplied mask.
Shape remains a direct sampling-contract check because native broadcasting could
otherwise apply restrictions to the wrong positions. Explicit device and dtype are
not repaired; ordinary PyTorch behavior remains visible to the caller.

## ProDy's water flag collides with existing CCD components

**Upstream:** `prody/ProDy`

**Affected reference:** ProDy
[`v2.6.1`](https://github.com/prody/ProDy/tree/v2.6.1).

**Classification:** independently confirmed component-classification bug. This
classification does not claim upstream acknowledgement.

### Behavior

ProDy 2.6.1 adds `SPC`, `SOL`, and `T3P` to its name-only `water` flag. Those IDs
already name non-water wwPDB CCD components: SPC contains C/N/O/S, SOL is
L-sorbose, and T3P is thymidine-3'-phosphate. A selection such as LigandMPNN's
`not protein and not water` therefore removes those real components before atom
context is constructed. ProDy 2.4.1 used a smaller nine-name set, demonstrating
that this is dependency-version policy rather than model architecture.

### Impact

Running the same structure through a later ProDy can silently delete ligand or
nucleic-acid atoms from model context. The resulting tensors remain well formed,
so the scientific input changes without an ordinary parsing or tensor failure.

### OFoldX behavior

OFoldX does not pin or import ProDy semantics. LigandMpnn and MpnnPacker use a
stable bounded predicate: one of the twelve candidate aliases is water only when
its declared atoms include O and no element outside H/O. This retains ordinary
partial water observations and preserves the colliding CCD components without a
Water class, generic chemistry inference, or CCD lookup.

## LigandMPNN delegates protein identity to a mutable ProDy flag

**Upstream:** `dauparas/LigandMPNN` and `prody/ProDy`

**Affected references:** LigandMPNN
[`26ec57a`](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de)
with ProDy 2.4.1, and the changed ProDy 2.6.1 flag table.

**Classification:** independently confirmed input-classification bug. This
classification does not claim upstream acknowledgement.

### Behavior

LigandMPNN delegates protein selection to ProDy. ProDy's amino-acid names differ
between 2.4.1 and 2.6.1 and may also be changed through ProDy settings. Several
force-field aliases in those tables collide with real non-polymer CCD components;
for example ARN and TYM can themselves contain an atom named CA. Conversely, the
CCD peptide placeholder `UNK` is omitted.

The flag also requires a CA atom. LigandMPNN applies its positive-occupancy filter
before selecting `protein`, so an absent or filtered CA can make the occurrence's
remaining atoms satisfy `not protein and not water` and enter atom context.

### Impact

Dependency version, process-global settings, record occupancy, or a colliding
three-character name can silently change whether the same source occurrence is a
protein row, ligand context, or omitted. Every resulting tensor remains well
formed, so the semantic change need not trigger a parser or PyTorch failure.

### OFoldX behavior

The family PDB reader owns a stable collision-aware rule. Chemically peptide
component IDs, including `UNK`, are protein directly. Force-field aliases that
collide with non-polymers require an ATOM declaration. Once imported, protein
identity is canonical and never changes when CA or occupancy changes; those facts
control only Model row observation. OFoldX imports no ProDy classifier, mutable
settings, generic registry, or runtime CCD lookup.

## AlphaFold 3 user CCD overrides leak into later jobs

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
[`7b197fe`](https://github.com/google-deepmind/alphafold3/tree/7b197fe859790fc3e04d03ea70dd0b9ba48881c9).

**Classification:** independently confirmed process-global state mutation in the
released input path. This classification does not claim upstream acknowledgement.

### Behavior

AlphaFold 3 caches the loaded base CCD with `functools.cache`; the cached function
returns a mutable dictionary. Every `Ccd` instance stores that exact dictionary as
`self._dict`. When `user_ccd` is supplied, its parsed components are applied with
`self._dict.update(...)` rather than to an instance-local mapping. See the
[`Ccd` loader and constructor](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/constants/chemical_components.py#L32-L82).

Consequently, overriding or adding a component in one `Ccd` mutates the cached
base object. A later `Ccd()` constructed without user data, or with a different
override, observes state left by the earlier job. This contradicts the class's
immutable-mapping description and its identity-based hash assumption.

### Impact

The component graph used to interpret an otherwise identical AlphaFold 3 input can
depend on which jobs previously ran in the Python process. Long-lived workers can
therefore leak custom ligand definitions between jobs and produce order-dependent
featurization without any input or repository change.

### OFoldX behavior

The base CCD may be cached for read-only reuse, but every call that supplies user
CCD data applies those exact-key overrides to an ordinary call-local shallow copy
or overlay. Later calls see only the base plus their own explicit override. The
override remains Processor input data, not Model repository state, a mutable
global registry, Provider abstraction, or lifecycle hook.

## AlphaFold 3 cannot select uncompressed `model.bin.N` shards

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
[`7b197fe`](https://github.com/google-deepmind/alphafold3/tree/7b197fe859790fc3e04d03ea70dd0b9ba48881c9).

**Classification:** independently confirmed filename-selection bug. This
classification does not claim upstream acknowledgement.

### Behavior

The released selector intends to recognize both `model.N.bin` and `model.bin.N`
for uncompressed parameter shards. Its latter regular expression is
`(?P<model_name>.*)\.bin]\.[0-9]+$`, with an unmatched literal `]` after `bin`.
An ordinary filename such as `af3.bin.0` therefore cannot match that branch. See
the
[`select_model_files` patterns](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/model/params.py#L185-L209).

### Impact

A correctly named uncompressed sharded model is reported as absent unless it also
happens to match another supported layout. Single-file and compressed layouts are
unaffected.

### OFoldX behavior

The AlphaFold 3 converter uses the intended `\.bin\.[0-9]+$` spelling and accepts
the released shard layouts without a typo-compatibility filename. It does not
manufacture a literal-`]` alias.

## AlphaFold 3 documents chirality matches as mismatches

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
[`7b197fe`](https://github.com/google-deepmind/alphafold3/tree/7b197fe859790fc3e04d03ea70dd0b9ba48881c9).

**Classification:** independently confirmed public-function documentation bug.
This classification does not claim upstream acknowledgement.

### Behavior

`compare_chirality` documents its returned Boolean as indicating that a ligand's
chirality _mismatches_ the reference. The implementation stores `_chiral_match(...)`
directly, whose Boolean is true only when every compared centre matches. Its local
mapping is correspondingly named `chiral_match_by_chain_id`. See
[`compare_chirality`](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/model/scoring/chirality.py#L149-L199).

### Impact

A caller following the return documentation can invert ligand stereochemistry
quality, treating a matching prediction as a mismatch and vice versa.

### OFoldX behavior

OFoldX preserves the actual scientific computation and names/documents the Boolean
as a chirality match. It does not negate the value merely to reproduce the
upstream prose error.

## AlphaFold 3 duplicate seeds overwrite earlier outputs

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
[`7b197fe`](https://github.com/google-deepmind/alphafold3/tree/7b197fe859790fc3e04d03ea70dd0b9ba48881c9).

**Classification:** independently confirmed output-identity bug. This
classification does not claim upstream acknowledgement.

### Behavior

The released input contract requires a nonempty seed sequence but does not require
unique values. Prediction preserves every occurrence in order. Output directories,
however, are named only from the numeric seed and sample index:
`seed-{seed}_sample-{sample_idx}`; seed-level embedding and distogram directories
use the same seed-only identity. See the
[`Input` seed check](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/common/folding_input.py#L969-L984)
and
[`write_outputs`](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/run_alphafold.py#L634-L665).

When the same seed appears twice, the later seed occurrence writes to the earlier
occurrence's paths. Files are replaced or mixed instead of representing both valid
executions.

### Impact

The in-memory operation performs both requested seed groups, but its released disk
output loses their distinct occurrence identity. Repeated seeds can silently erase
predictions, embeddings, or distograms.

### OFoldX behavior

`AlphaFold3Result` retains an ordered sequence of `(seed, FoldResult)` groups, so
duplicates remain distinct. A future family writer includes occurrence position in
its path identity while retaining the numeric seed as metadata. OFoldX does not
reject valid duplicate seeds or collapse them into a mapping.

## AlphaFold 3 declares a template-date cutoff that its featurizer never reads

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
[`7b197fe`](https://github.com/google-deepmind/alphafold3/tree/7b197fe859790fc3e04d03ea70dd0b9ba48881c9).

**Classification:** independently confirmed dead configuration field. This
classification does not claim upstream acknowledgement.

### Behavior

`WholePdbPipeline.Config.max_template_date` is documented as preventing template
leakage and is stored on the configuration, but `WholePdbPipeline` never reads it.
Its template featurizer receives only `max_templates`; the only date that this
pipeline reads is the distinct `ref_max_modified_date`, which controls whether CCD
model coordinates may be used after other conformer sources fail. See the
[`WholePdbPipeline.Config` declaration](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/model/pipeline/pipeline.py#L91-L163)
and its
[`Templates.compute_features` call](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/model/pipeline/pipeline.py#L350-L365).

The separate search data pipeline does enforce
`DataPipelineConfig.max_template_date` while filtering search hits. Templates
already supplied on an input bypass that search/filtering route, so the dead
`WholePdbPipeline` field cannot enforce its documented cutoff for them.

### Impact

A caller can set the documented featurizer cutoff and still use a supplied template
newer than that date, introducing validation leakage without an error. The live
search cutoff and CCD fallback date continue to work, but they have different
ownership and scientific meanings.

### OFoldX behavior

OFoldX does not preserve the no-op `WholePdbPipeline` field. The live search-result
template cutoff and the distinct CCD model-coordinate fallback date remain separate
arguments at the operations that consume them. A convenience path may pass one
caller date to both deliberately, but the underlying API does not merge their
meanings or claim that the dead upstream field was effective.

## AlphaFold 3 silently converts DSSP failure into zero disorder

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
[`7b197fe`](https://github.com/google-deepmind/alphafold3/tree/7b197fe859790fc3e04d03ea70dd0b9ba48881c9).

**Classification:** independently confirmed scientific fallback defect. This
classification does not claim upstream acknowledgement.

### Behavior

AlphaFold 3 computes `fraction_disordered` by running mkdssp separately for each
protein chain. It catches `ValueError` and `RuntimeError`, logs a warning, and
omits the failed chain. If every chain fails, the accumulated accessibility list
is empty and the function returns `0.0`. See
[`fraction_disordered`](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/model/confidences.py#L99-L137).

The same numeric result is therefore used for a genuinely ordered prediction, a
structure with no protein, a malformed DSSP conversion, or an unavailable/broken
mkdssp or `components.cif` installation. Partial chain failure silently computes a
fraction over only the surviving chains.

### Impact

`fraction_disordered` contributes `0.5 * fraction_disordered` to the final ranking
score. Dependency or conversion failure can therefore change global seed/sample
selection while leaving a plausible score and otherwise complete output.

### OFoldX behavior

OFoldX retains the exact AF-RSA computation when it succeeds and lets an ordinary
DSSP failure propagate. A structure with genuinely no protein returns zero; a
failed protein-chain calculation does not masquerade as that biological case. No
retry, substitute estimator, warning-only path, or hidden ranking fallback is
added.

## AlphaFold 3 input documentation contradicts its accepted JSON versions

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
[`7b197fe`](https://github.com/google-deepmind/alphafold3/tree/7b197fe859790fc3e04d03ea70dd0b9ba48881c9).

**Classification:** independently confirmed public documentation bug. This
classification does not claim upstream acknowledgement.

### Behavior

The input reference first says the top-level `version` must be 1 or 2. Its next
section says the field may be 1, 2, or 3, then immediately documents a version-4
chain-description feature. The parser's authoritative `JSON_VERSIONS` constant is
`(1, 2, 3, 4)`. See the
[`version` documentation](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/docs/input.md#L154-L169)
and the
[`JSON_VERSIONS` declaration](https://github.com/google-deepmind/alphafold3/blob/7b197fe859790fc3e04d03ea70dd0b9ba48881c9/src/alphafold3/common/folding_input.py#L47-L52).

### Impact

A caller following either prose restriction may reject a valid version-3 or
version-4 input, including `userCCDPath` or chain descriptions, despite the
released parser supporting it.

### OFoldX behavior

The AlphaFold3 JSON compatibility path accepts versions 1 through 4 and preserves
their released fields. It does not add a compatibility alias or silently rewrite
the declared version; the code-supported format is documented directly.

## NA-MPNN scores every requested decoding order with the first causal mask

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed batched-scoring bug. This
classification does not claim upstream acknowledgement.

### Behavior

The released `score` operation derives `decoding_order` for the input batch and
constructs one causal mask before repeating the encoded tensors for
`batch_size` evaluations. It neither repeats nor reconstructs the mask from the
requested per-evaluation orders, and it returns only `decoding_order[0]`. See
`inference/model_utils.py:375-423` in the affected revision.

Consequently, opposite requested orders can be evaluated with the same first
causal mask while the output reports only one order. The returned probabilities
remain numerically plausible, so shape checks do not expose the semantic mismatch.

### Impact

Sequence likelihood and residue probabilities depend on the autoregressive causal
order. Reusing another evaluation's mask silently changes the conditioning graph
and makes repeated-order statistics incorrect.

### OFoldX behavior

NaMpnn constructs one causal mask for each exact decoding-rank row and returns all
rank rows with the corresponding probability tensor. It does not repeat a first
mask or discard order identity merely to preserve the upstream batching shortcut.

## NA-MPNN gives tied residues an order and last-member controls

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed symmetry-sampling bug. This
classification does not claim upstream acknowledgement.

### Behavior

The tied sampler derives every candidate's grouped traversal from
`decoding_order[0]`, flattens each group back into a strict residue order, and then
decodes the members sequentially. Although neural logits are accumulated across
members, the residue bias and adjacent-token pair bias used for the final group
distribution are the values left by the last loop member. See
`inference/model_utils.py:226-326` in the affected revision.

Reversing member order can therefore change a supposedly symmetric distribution,
and earlier members can become causal context for later members even though one
token is sampled for the entire group.

### Impact

The result is not invariant to the textual order of a symmetry group. A caller's
signed weights and per-residue controls can affect a different computation from
the one described by the tied-token request.

### OFoldX behavior

NaMpnn gives token-symmetry members equal causal ranks and combines their neural
logits with the released signed, unnormalized weights. A group-constant residue
bias is applied once and hard support is intersected. The exact interaction with
the released adjacent-token pair potential remains an explicit family Tik question;
OFoldX will neither copy last-member traversal nor silently drop that official
control.

## NA-MPNN's numeric order priorities do not guarantee fixed-first decoding

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed causal-order bug. This classification
does not claim upstream acknowledgement.

### Behavior

Sampling and scoring sort `(design_mask + 0.0001) * abs(random_value)` to place
fixed rows first. Training uses the same construction after setting protein rows
to zero for its protein-first option. See `inference/model_utils.py:129,389` and
`na_model_utils.py:620-623` in the affected revision.

The two numeric ranges overlap: a sufficiently large fixed-row draw can exceed a
small designed-row draw. The same is true for a nominally first protein row and a
small non-protein draw.

### Impact

Fixed sequence or protein context may appear after a row that was supposed to
condition on it. This changes the autoregressive graph without changing tensor
shape or producing an ordinary failure.

### OFoldX behavior

NaMpnn constructs categorical causal ranks: required context groups precede sampled
groups exactly, and random values order rows only within their category. Explicit
caller ranks are obeyed directly rather than repaired through numeric offsets.

## NA-MPNN's smoothed loss crashes with its own default arguments

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed public-function default bug. This
classification does not claim upstream acknowledgement.

### Behavior

`loss_smoothed` declares both `ppm_mask=None` and `aligned_ppm=None`, then
unconditionally evaluates `ppm_mask.bool()` and indexes `aligned_ppm`. Calling the
function with its documented defaults therefore raises before computing the
ordinary one-hot smoothed objective. See `na_model_utils.py:111-145` in the
affected revision.

### Impact

The public training objective cannot be used without experimental probability
profiles even though its signature says those inputs are optional.

### OFoldX behavior

The NaMpnn smoothed objective uses one-hot targets when both experimental-profile
arguments are absent. When both are supplied it replaces exactly the selected
target rows. Supplying only one is scientifically incomplete and fails directly;
there is no empty placeholder or fallback profile.

## NA-MPNN exposes two feature widths but swaps them internally

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed architecture-constructor bug. This
classification does not claim upstream acknowledgement.

### Behavior

Both inference and training constructors accept `node_features` and
`edge_features`, but pass them positionally to feature constructors whose first two
parameters are declared in the opposite order. The feature encoder therefore emits
node width `edge_features` and edge width `node_features`, while the following
projections expect the original widths. See `inference/model_utils.py:8-48,426-485`
and `na_model_utils.py:349-419,519-567` in the affected revision.

Released commands and both checkpoints set the two values equal, hiding the swap.
Unequal values fail at the projections rather than representing two usable public
architecture axes.

### Impact

The constructor advertises configurations that cannot execute, and a converted
config could persist two apparently meaningful values unsupported by any released
model.

### OFoldX behavior

NaMpnn exposes one node/edge `feature_size` plus the independently live
`hidden_size` proven by the executable graph. It does not preserve false separate
node and edge axes, add an equality validator, or guess support for unreleased
unequal node/edge widths.

## NA-MPNN infers RNA identity from the presence of O2-prime coordinates

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed polymer-realization bug. This
classification does not claim upstream acknowledgement.

### Behavior

With incomplete-residue loading enabled, the parser correctly classifies polymer
type from the component name. It separately stores observed O2-prime presence as
`rna_mask_for_token_conversion`. The output sequence and PDB paths later use that
coordinate-presence mask—not the canonical polymer type—to decide whether a shared
nucleic token should be rendered as RNA or DNA. See
`inference/data_utils.py:302-320` and `inference/run.py:395-396,468-472` in the
affected revision.

An RNA residue missing O2-prime coordinates is thus decoded as DA, DC, DG, or DT
even though it was admitted and modeled as RNA.

### Impact

Missing experimental coordinates silently change chemical identity. The output can
replace ribose with deoxyribose while retaining a plausible base and backbone.

### OFoldX behavior

Canonical polymer occurrence identity is authoritative. Coordinate masks determine
Model validity and observed geometry only; they never reclassify RNA as DNA.
Shared tokens are realized through each row's stored polymer type.

## NA-MPNN training has no full-precision execution branch

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed training-control-flow bug. This
classification does not claim upstream acknowledgement.

### Behavior

The training loop defines `log_probs`, `probs`, and the smoothed loss only inside
`if params["MIXED_PRECISION"]`. Code immediately after that branch consumes
`log_probs` unconditionally. Setting the released configuration flag false reaches
an unbound variable instead of ordinary full-precision forward/backward. See
`na_run.py:216-240` in the affected revision.

### Impact

The advertised non-AMP training mode cannot run, and users cannot obtain ordinary
full-precision optimization through the released control.

### OFoldX behavior

The unified Runner uses an ordinary Model call when AMP is not requested and the
normal autocast/scaler route when it is. OFoldX does not force AMP, change an
explicit dtype, or add a precision-repair layer.

## NA-MPNN's `all` metric mode misspells canonical accuracy

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`.

**Classification:** independently confirmed metric-registration bug. This
classification does not claim upstream acknowledgement.

### Behavior

`get_metric_manager(..., metrics_to_compute="all")` registers
`canonialBasePairAccuracy`, while accumulation later looks up
`canonicalBasePairAccuracy`. See `na_metric_manager.py:84-85,199-210` in the
affected revision.

### Impact

The public `all` metrics mode raises `KeyError` instead of reporting canonical
base-pair accuracy.

### OFoldX behavior

The canonical-base-pair objective and metric use one correctly spelled public
name. Runner composition refers to that function directly; no misspelled alias or
compatibility registry is retained.

## AlphaFold 3 accepts `--num_seeds=1` and then rejects it

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed command-contract bug. This
classification does not claim upstream acknowledgement.

### Behavior

`run_alphafold.py` declares `--num_seeds` with `lower_bound=1` and describes every
set value as an expansion of a one-seed input. It then passes the value to
`Input.with_multiple_seeds`, which rejects every value less than or equal to one.
See `run_alphafold.py:352-364,1005-1009` and
`src/alphafold3/common/folding_input.py:1493-1504` in the affected revision.

Consequently, a value accepted by flag parsing and implied by the help text fails
only after input loading.

### Impact

The public command has contradictory bounds and runtime semantics for the ordinary
single-seed case.

### OFoldX behavior

An explicit ordered one-seed request remains one seed. A convenience seed-count
argument, if retained by the final AF3 contract, treats one as the identity rather
than routing through a helper that requires plurality.

## AlphaFold 3's optional CCD date can be compared with `None`

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed fallback-path bug. This classification
does not claim upstream acknowledgement.

### Behavior

`WholePdbPipeline.Config.ref_max_modified_date` defaults to `None` and is passed
unchanged into reference-coordinate construction. When ideal coordinates contain
unknown values and the CCD provides modification dates, the helper evaluates
`max_modified_date < ref_max_modified_date`. See
`src/alphafold3/model/pipeline/pipeline.py:108-112,148,360-385` and
`src/alphafold3/model/features.py:1505-1534` in the affected revision.

That reachable fallback compares a `datetime.date` with `None` and raises
`TypeError`. The released CLI normally supplies a date, which hides the invalid
Config default.

### Impact

A documented optional configuration fails only for chemical components that need
the model-coordinate fallback, making behavior depend on CCD contents rather than
the declared API.

### OFoldX behavior

The final AF3 Tik will freeze the exact cutoff signature. If `None` is accepted, it
means no date cutoff and is handled explicitly; it is never passed into a date
comparison.

## AlphaFold 3 marks unpredicted atoms as occupied zero coordinates

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed structure-and-ranking bug. This
classification does not claim upstream acknowledgement.

### Behavior

When the complete output atom layout contains atoms absent from the Model output
layout, gather conversion inserts `(0, 0, 0)`. `get_predicted_structure` warns but
then assigns occupancy `1.0` to every output atom. The resulting structures are
passed directly to clash detection and fraction-disordered computation, both of
which feed the final ranking score. See
`src/alphafold3/model/model.py:79-149,382,477-493` in the affected revision.

### Impact

An atom the Model never predicted is represented as an observed atom at the
origin. It can create false clashes, alter solvent accessibility and disorder, and
therefore change which diffusion sample is selected as best.

### OFoldX behavior

Decoded Structures mark unpredicted rows false in `coordinate_mask`. Confidence,
clash, solvent-accessibility, and ranking operations do not treat those fabricated
coordinates as observations. This is an intentional scientific correction and is
reported separately from predicted-coordinate parity.

## AlphaFold 3's bucket docstring contradicts dynamic bucket creation

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed documentation bug. This classification
does not claim upstream acknowledgement.

### Behavior

`process_fold_input` says an input larger than the largest explicit bucket raises
an error. `calculate_bucket_size` instead logs a warning and returns the exact
input token count as a new bucket; the command flag help and performance guide
describe that implemented behavior. See `run_alphafold.py:755-771`,
`src/alphafold3/model/pipeline/pipeline.py:43-71`, and
`docs/performance.md:262-291` in the affected revision.

### Impact

Callers reading the function contract may incorrectly precompute or reject large
inputs even though the released implementation supports dynamic compilation.

### OFoldX behavior

Explicit buckets retain the released dynamic extension behavior: an input beyond
the largest bucket uses its exact token length. The public documentation states
that behavior directly.

## AlphaFold 3 cannot read its own empty template-map JSON

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed JSON round-trip bug. This
classification does not claim upstream acknowledgement.

### Behavior

`ProteinChain.to_dict` serializes an empty `query_to_template_map` as
`queryIndices: []` and `templateIndices: null`. The corresponding reader passes
those two values directly to `zip`, so the emitted `null` is not iterable. See
`src/alphafold3/common/folding_input.py:375-389,408-416` in the affected revision.

An otherwise valid template with an empty alignment map therefore writes
successfully but raises `TypeError` when read back.

### Impact

The official normalized JSON writer does not satisfy semantic round-trip for a
representable public `Template` value.

### OFoldX behavior

An empty map writes both `queryIndices` and `templateIndices` as empty lists. The
normalized version-4 document remains readable and preserves the empty map.

## AlphaFold 3 exposes DNA modifications as a bound method

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed public-value bug. This classification
does not claim upstream acknowledgement.

### Behavior

`ProteinChain` and `RnaChain` expose their stored modifications through ordinary
properties. `DnaChain.modifications` has the same body but omits `@property`, so
ordinary attribute access returns a bound method rather than the modification
sequence. See `src/alphafold3/common/folding_input.py:518-524,665-740` in the
affected revision.

### Impact

The three polymer values have inconsistent public semantics, and generic code
reading `chain.modifications` silently receives the wrong kind only for DNA.

### OFoldX behavior

DNA modifications are ordinary canonical molecular-system data and are read as a
sequence, exactly like protein and RNA modifications. No compatibility bound
method is retained.

## AlphaFold 3 reuses solvent accessibility across equal-sequence chains

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed ranking bug. This classification does
not claim upstream acknowledgement.

### Behavior

`fraction_disordered` caches each chain's DSSP-derived accessibility vector by its
single-letter sequence. A later chain with the same sequence reuses that vector
without running DSSP on its own predicted coordinates. See
`src/alphafold3/model/confidences.py:102-136` in the affected revision.

Equal-sequence copies can have different conformations and solvent exposure, so the
cache key omits a scientific input to AF-RSA.

### Impact

The disorder term, final ranking score, and globally selected diffusion sample can
depend on which equal-sequence chain is encountered first.

### OFoldX behavior

AF-RSA is computed independently for every predicted protein chain. OFoldX does
not cache a coordinate-dependent result by sequence.

## AlphaFold 3 can freeze global selection on an initial NaN

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed output-selection bug. This
classification does not claim upstream acknowledgement.

### Behavior

The writer stores the first ranking score unconditionally and replaces it only
when a later value is strictly greater. If the first value is NaN, every later
comparison against it is false, so a finite higher-ranked result can never become
the root best output. See `run_alphafold.py:629-649` in the affected revision.

### Impact

One invalid first score poisons selection for the entire ordered seed/sample set
and publishes the wrong root prediction.

### OFoldX behavior

Global selection rejects any NaN ranking value. Among valid equal values it keeps
the earliest seed-major, sample-major result.

## AlphaFold 3 documents `pad_num_chains` but never reads it

**Upstream:** `google-deepmind/alphafold3`

**Affected reference:** AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.

**Classification:** independently confirmed no-op configuration bug. This
classification does not claim upstream acknowledgement.

### Behavior

`WholePdbPipeline.Config.pad_num_chains` explicitly documents the chain dimension
it claims to pad. The value is copied into `PaddingShapes.num_chains`, but no
feature computation reads that field. See
`src/alphafold3/model/pipeline/pipeline.py:89-101,142,286-292` and
`src/alphafold3/model/features.py:60-66` in the affected revision.

Changing `pad_num_chains` leaves every released feature shape and value unchanged.

### Impact

Direct users can configure and serialize a documented padding behavior that was
never executed.

### OFoldX behavior

The native Processor config omits `pad_num_chains`; no compatibility setting,
warning, or dead-option validator is added.

## ProteinMPNN training ignores `num_decoder_layers`

**Upstream:** `dauparas/ProteinMPNN`

**Affected reference:** ProteinMPNN revision
`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`.

**Classification:** independently confirmed training-constructor bug. This
classification does not claim upstream acknowledgement.

### Behavior

The training command declares independent `--num_encoder_layers` and
`--num_decoder_layers` arguments, but Model construction passes
`args.num_encoder_layers` to both constructor fields. See
[`training/training.py:76-83`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/training/training.py#L76-L83)
and the two public arguments at
[`training/training.py:240-241`](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/training/training.py#L240-L241).

Changing only the advertised decoder-layer count therefore has no effect on the
constructed training graph.

### Impact

A training configuration can record a decoder depth that was never executed or
saved, so its apparent architecture does not describe the checkpoint.

### OFoldX behavior

`ProteinMpnnConfig` retains independent encoder and decoder layer counts because
both are real checkpoint shapes. Construction passes each field directly; there is
no equality repair or compatibility alias for the upstream command bug.

## LigandMPNN maps conflicting membrane labels to `other`

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de`.

**Classification:** independently confirmed membrane-label lowering bug. This
classification does not claim upstream acknowledgement.

### Behavior

The design and score commands accept independent lists of transmembrane buried and
interface residues. They lower labels as
`2 * buried * (1 - interface) + interface * (1 - buried)`. A residue present in
both public lists therefore receives zero, the same value as `other`, rather than
either requested membrane class or an explicit contradiction. See
[`run.py:254-274`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L254-L274)
and
[`score.py:136-156`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/score.py#L136-L156).

### Impact

Conflicting scientific annotations silently erase each other and produce a valid
but unintended Model input.

### OFoldX behavior

The per-residue membrane Processor accepts one label per component reference:
`other`, `interface`, or `buried`. Ordinary mapping ownership makes a component's
value singular, so the contradictory two-list state is not representable and no
generic validation layer is needed.

## LigandMPNN misnames duplicate overall confidence as ligand confidence

**Upstream:** `dauparas/LigandMPNN`

**Affected reference:** LigandMPNN revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de`.

**Classification:** independently confirmed decoded-metric naming bug. This
classification does not claim upstream acknowledgement.

### Behavior

The design command computes overall confidence over valid designed rows. Only the
`ligand_mpnn` model type narrows a second mask to the ligand interface; every other
model type assigns the second mask to the exact overall mask. The FASTA writer
still calls that duplicate value `ligand_confidence`. See
[`run.py:428-443`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L428-L443)
and
[`run.py:651-670`](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/run.py#L651-L670).

### Impact

Protein, soluble, and membrane outputs claim a ligand-specific metric even though
no ligand scope entered the calculation, and duplicate the overall value under a
scientifically false name.

### OFoldX behavior

Only the ligand-aware route publishes `ligand_interface_confidence` with its exact
component scope. Protein, soluble, and membrane routes publish overall designed
sequence confidence once; no duplicate alias is retained.

## NA-MPNN's NA-only mode designs protein rows in mixed chains

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
[`aafa32c`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

**Classification:** independently confirmed row-selection bug. This
classification does not claim upstream acknowledgement.

### Behavior

The parser records an NA chain letter whenever any valid DNA or RNA reference row
uses that PDB chain ID. `design_na_only` then filters chain letters and constructs
`chain_mask` by testing every Model row's chain letter. See
[`data_utils.py:371-381`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/data_utils.py#L371-L381)
and
[`run.py:293-310`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/run.py#L293-L310).

If one source chain contains both a valid protein row and a valid nucleic-acid
row, the NA row admits that chain and the protein row also receives
`chain_mask=1`. The released specificity workflow invokes this mode specifically
to sample NA while retaining protein structural context.

### Impact

Protein context can become a design target merely because it shares a source chain
identifier with nucleic acid. The sequence and specificity profile then no longer
implement “NA only”.

### OFoldX behavior

NaMpnn specificity selects exact valid DNA and RNA `ComponentReference` values.
Protein rows remain fixed context even when they share a source chain or canonical
instance with nucleic-acid rows. Generic design uses the caller's exact
`positions`; it never infers biological selection from another row in the chain.

## NA-MPNN's missing-atom flag can remove complete modified residues

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
[`aafa32c`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

**Classification:** independently confirmed non-monotonic missing-atom bug. This
classification does not claim upstream acknowledgement.

### Behavior

A Model row exists only when the parser observes its protein CA or nucleic-acid
C1-prime reference atom. The flag cannot restore a row whose reference atom is
absent. With `load_residues_with_missing_atoms=0`, protein, DNA, and RNA validity
is computed from observed backbone completeness. With the flag enabled, that
computation is replaced by membership in three short hard-coded standard-resname
lists. See
[`data_utils.py:259-318`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/data_utils.py#L259-L318).

The switch is not monotonic: an incomplete standard ALA with CA can become valid,
while a complete ProDy protein such as MSE or a complete modified nucleic residue
such as PSU can become invalid because its resname is absent from those lists.

### Impact

A flag described as loading residues with missing atoms can instead remove complete
modified residues, changing polymer features, selection, sampling, scoring, and
specificity alignment.

### OFoldX behavior

The public operation is `allow_incomplete_backbone`. Row membership still requires
an observed CA for protein or C1-prime for DNA/RNA. When false, validity requires
the exact polymer-specific backbone slots. When true, every canonical protein,
DNA, or RNA row already on that axis is valid despite missing remaining slots.
Missing coordinates retain false atom masks; canonical polymer identity, not a
source-resname whitelist, determines polymer class.

## NA-MPNN makes modified-nucleic identity depend on missing coordinates

**Upstream:** `baker-laboratory/NA-MPNN`

**Affected reference:** NA-MPNN revision
[`aafa32c`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

**Classification:** independently confirmed identity-lowering bug. This
classification does not claim upstream acknowledgement.

### Behavior

The parser derives protein, DNA, RNA, and unknown polymer IDs from backbone
completeness masks. It then chooses the fallback sequence token from those same
masks: protein `UNK`, DNA `DX`, or RNA `RX`. See
[`data_utils.py:313-341`](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/inference/data_utils.py#L313-L341).

Standard alphabet resnames still map directly on invalid rows. The defect concerns
out-of-alphabet modified nucleic residues: with a complete nucleic backbone the
residue falls back to `DX` or `RX`; when one required coordinate is missing all
polymer masks become zero and the same chemical residue falls back to protein
`UNK` and polymer type `UNK`.

### Impact

Coordinate observation changes chemical identity. The native sequence,
statistics output, explicit incomplete-backbone execution, and dense pair bias can
observe the changed token even when the ordinary graph mask suppresses the row.

### OFoldX behavior

NaMpnn derives polymer type from canonical polymer chemistry independently of
`valid_mask`. An out-of-alphabet DNA component lowers to `DX` and an
out-of-alphabet RNA component lowers to `RX` whether its backbone is complete or
incomplete. The validity mask alone controls graph participation; missing
coordinates never relabel nucleic chemistry as protein unknown.
