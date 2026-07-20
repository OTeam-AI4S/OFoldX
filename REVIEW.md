# Design review

This is the short review surface for high-impact findings made while `DESIGN.md`
is being implemented. It is not a second specification: accepted contracts live
in `DESIGN.md`, and confirmed reference defects live in `UPSTREAM_BUGS.md`.

An item appears here when it changes a public contract, invalidates an earlier
premise, or materially changes the route to a complete family. Routine
implementation details do not belong here.

## ESMFold2, MultiMolecule ESMC, and explicit dtype

MultiMolecule's `EsmCPreTrainedModel` inherits Transformers'
`PreTrainedModel`. The current MultiMolecule checkout uses Transformers 5, whose
`from_pretrained` path constructs models on the meta device. MultiMolecule's shared
`RotaryEmbedding` consequently owns lazy materialization of its non-persistent
`inv_freq` buffer.

An independent probe against MultiMolecule revision
`531216db0fdbe2065485a3f2cab0b3bdc3b0b80c` constructed a two-layer ESMC on meta,
loaded an ordinary state dict with `strict=True, assign=True`, and ran a CPU
forward pass. Its two `inv_freq` buffers remained on meta immediately after the
state load, then were rebuilt on CPU by the first forward; no meta tensor remained
and the output was ordinary FP32. Therefore ESMC is **not** evidence against
OFoldX's generic meta-and-assign repository loader, and OFoldX should not add an
ESMC materializer or a generic post-load tensor scan.

A separate probe showed that explicit
`RotaryEmbedding(64).to(torch.bfloat16)` changes `inv_freq` and position arithmetic
to BF16. At sequence length 2,048, 1,407 adjacent position pairs then have identical
complete cosine rows; FP32 has zero. This is not an OFoldX or MultiMolecule bug:
the caller explicitly requested conversion of every floating parameter and buffer.
The ESM reference's FP32 rotary arithmetic describes its released mixed-precision
policy, not a reason to violate ordinary `nn.Module.to` semantics.

Current conclusion:

- keep OFoldX's ordinary `nn.Module.to` contract and generic meta loader;
- retain the serialized mixed dtypes when no dtype is requested;
- recommend AMP or explicit per-module choices when a researcher wants mixed
  precision;
- obey an explicit whole-model dtype without a guard, warning, hidden exception,
  or babysitting override; and
- remove the erroneous ESMC entry from `UPSTREAM_BUGS.md`.

This is the owner-reviewed decision. No MultiMolecule file has been changed from
the OFoldX worktree.

## Explicit caller intent audit

Applying the same decision beyond ESMFold2 exposed several remaining exceptions.
They have been removed from the frozen shared behavior:

- repository and config construction now both obey a final `device="meta"` through
  ordinary Model `.to(...)`; repository loading still hydrates first and does not
  invent a second skip-weight mode;
- an explicit Processor dtype applies to every floating tensor it emits, with no
  hidden field-specific FP32 policy;
- Boltz-2 no longer warns about its 56-atom training distribution, preserves
  affinity molecular weight in a hidden dtype, or automatically repairs explicit
  affinity tensors supplied to the Model;
- ProteinMPNN, LigandMPNN, and NA-MPNN no longer move or cast explicit sampling
  tensors to match the processed batch; and
- BoltzGen no longer warns and relaxes explicit residue exclusions when their
  intersection is empty. It raises because its finite additive mask would otherwise
  sample a scientifically forbidden identity.

The BoltzGen relaxation is inherited from the audited upstream behavior and is recorded in
`UPSTREAM_BUGS.md`. This audit does not authorize deleting evidence-backed FP32
numerical formulas or semantic boolean/index casts. It removes placement repair and
hidden policy, not the released scientific computation.

## AF3 and LigandMPNN derived buffers

AF3 confidence-bin edges and LigandMPNN element lookup tables are deterministic
config/code constants absent from every released checkpoint. Making them persistent
would create a second artifact authority. Leaving them on meta would make OFoldX's
internal low-memory construction unusable.

The frozen minimal solution is owner-local: create ordinary Parameters first; use
their construction device for the derived buffers; and use CPU only when that
internal device is meta. The buffers remain `persistent=False`, and later explicit
`.to(...)` follows ordinary PyTorch semantics. No loader scan, hook, materializer,
family flag, `_apply` override, or persistent duplicate was added.

## Minimal Structure carrier

The earlier Structure/I/O proposal mixed one coordinate realization with source
round-trip policy. The released families justify a smaller canonical carrier:
`system`, `sites`, `coordinates`, `coordinate_mask`, optional `occupancy`, and the
one independently evidenced source fact `residue_numbers`.

Occupancy is not generic file metadata: AlphaFold 3's public ligand-template path
passes `ligand_atom_occupancies` into model features. B-factor has no equivalent
input consumer in the audited families. AF3, Boltz, and Protenix use the B-factor
column as an export encoding for predicted confidence, while ModelCIF owns
separate QA categories. Confidence therefore remains a Result score or metric.

The frozen carrier has no atom-layout flag, B-factor, altloc IDs, source bag,
provenance, or generic annotations. One source model/altloc realization becomes
one Structure; exact parsing, assembly selection, exchange identity, and lossy
writing remain a separate open I/O decision. The implementation follows explicit
NumPy dtypes, retains arrays by reference, and performs no finiteness/range scan or
defensive copy.

## Structure path inputs lose model-significant residue numbering

The five-field `Structure` carrier correctly removed a generic source bag, but a
released-family audit found one source fact that is not cosmetic. ProteinMPNN uses
PDB residue numbers and insertion codes to insert transient masked rows across
integer gaps, then gives that expanded tensor a dense `residue_idx`. LigandMPNN,
NA-MPNN, and the side-chain Packer instead derive `R_idx` directly from source
numbers, with their released insertion-code or duplicate-number handling. Released
fixtures contain numbering gaps, so these distinct transforms are reachable and
can change the tensor graph.

Current `AtomSite(instance_id, component_index, atom_id)` and the five Structure
fields cannot recover that source numbering. Three tempting repairs are rejected:

- inserting `UNK` components at every numeric gap asserts molecular chemistry that
  a PDB numbering discontinuity does not prove;
- silently using dense component indices changes released model computation; and
- repairing only path calls inside each Processor makes `processor(path)` differ
  from `processor(read_structure(path))` without an explicit API distinction.

This falsifies the already-reopened Structure reader proposal more strongly and
reopens the `str | PathLike` part of the MPNN Processor contracts. The direct
`Structure` tensor boundary and Model math remain valid. The bounded
source-numbering Tik selected the smallest explicit carrier below and kept every
family-specific gap/index transform in its concrete Processor. The old `_codec.py`,
private JSON-in-mmCIF payload, broad
`SourceAtomIdentity`, and generic annotations map do not solve this narrower fact
and remain rejected.

The frozen bounded value is deliberately smaller than source identity:

~~~python
@dataclass(frozen=True)
class ResidueNumber:
    number: int
    insertion_code: str = ""

class Structure:
    residue_numbers: dict[ComponentReference, ResidueNumber] | None
~~~

`ComponentReference` already identifies the instance, so a second `asym_id` would
be redundant for MPNN lowering. `None` means that a programmatic or decoded
Structure supplies no source numbering and each family uses its documented dense
fallback. A supplied map is authoritative for a consuming family: missing row keys
fail naturally instead of falling back per row. It may contain extra references
and is retained by reference without a coverage scan. ProteinMpnn creates only
transient non-canonical gap rows, while the other three families apply their own
released `R_idx` transforms. Fixed-topology pack/refine outputs preserve the map;
new folds have none. The record needs no custom validation, reverse index,
normalization, or cache.

This value is not a complete PDBx identity. mmCIF distinguishes label and auth
asym/sequence namespaces, and `auth_seq_id` need not be an integer. The owner
decision for a future general reader remains open. The three-field
`ResidueId(asym_id, sequence_id, insertion_code)` proposal is rejected because it
would conflate those namespaces while pretending to be general.

## Partial observed chemistry before Structure reading

Released LigandMPNN and NA-MPNN paths accept arbitrary non-water HETATM rows and
consume only their element and coordinates. Requiring a complete CCD graph would
remove official input behavior; treating an unknown graph as known bondless would
fabricate chemistry. The frozen narrow slice in DESIGN Section 3.2.1 therefore uses
`ChemicalComponent.bonds=None` for an open partial observation and
`frozenset()` for a declared complete bondless component. It needs no
`PartialComponent`, completeness enum, provider, provenance bag, or generic
validator.

The earlier proposal to collapse explicit and implicit hydrogen state into one
`hydrogens` field was too broad. MPNN does not consume those fields, while the
SMILES and Boltz RDKit boundaries consume their distinction. DESIGN now retains the
three existing field names and only allows `None` for unknown facts. Complete
catalog and SMILES chemistry continue to supply explicit zero and boolean values;
topology-dependent families reject partial chemistry at their own scientific
boundary.

This does not settle label/auth identity, altloc and microheterogeneity,
source-bond unknown order, assembly materialization, or packed-PDB output, so it is
not authorization to implement a generic reader. Concrete MPNN family readers may
materialize arbitrary observed HETATM rows as `bonds=None` without first solving
that broader contract.

## Condition vocabulary subtraction without feature loss

The former `conditions.py` exposed twelve records, a global `Condition` union,
hard/soft intent, weights, tolerances, symbolic references, Contig, and a generic
cross-condition validator. The audited families do not share that framework, and
several mappings were scientifically wrong: RFdiffusion length is global across
inpaint chains rather than per output instance; BoltzGen structure groups are not
fixed coordinates; and ODesign hotspots are not synthetic 8 Å contacts.

The frozen shared implementation now contains only the two plain records with
proven cross-family consumers: `AllowedResidueIdentities` and
`TiedResidueIdentities`. They have no custom constructor validation, generic
intent/weight/tolerance, satisfaction outcome, or global `Condition` alias.
The earlier brief attempt to publish Protenix-shaped `Contact`, `AtomDistance`,
and `Pocket` records as shared data was falsified by the Boltz and Protenix audits.
Every positive spatial-conditioning API remains family-owned; each stays reopened
until its complete family Tik freezes, with one exception already ratified —
BoltzGen's structure-group contract (per-token `structure_group` → `token_distance_mask`,
`reference_space_id` left per-residue) is frozen under §3.7.2, validated exact-equal
to the released featurizer (`process_token_features`, 2026-07-19).

This subtraction does **not** remove official functionality. RFdiffusion
potentials/scaffold controls, BoltzGen binding and non-binding labels, structure
groups and secondary structure, ODesign hotspots/length controls, and every other
released conditioning path remain mandatory on the concrete family API. Their
records and lowering stay open until each complete family Tik can place them
without inventing a false shared abstraction.

## MpnnPacker side-chain packing contract

The upstream ownership audit corrected the earlier name and package boundary.
`run.py` loads the side-chain Packer independently from the selected sequence
Model, and the official README explicitly composes it with any of five available
designer types. Its graph has no LigandMPNN dependency. OFoldX therefore names the
independent concrete family `MpnnPacker`, rather than a LigandMpnn-owned Packer; `pack`,
`PackPipeline`, and `PackResult` remain the generic task vocabulary. The official
cross-model B × pack workflow is a composition question and does not move the
Packer back under LigandMPNN.

The audited released implementation contradicts several current OFoldX assumptions:

- an explicit target sequence may replace identities already known in the input;
  the packer still does not *choose* residue identities;
- every changed identity must also be selected for packing;
- the standalone selection is just `positions`; `repack_everything` is a
  design-to-pack composition control, not a second Model/Processor selection API;
- all 20 standard amino acids are selectable, including ALA and GLY;
- non-protein atom context includes ligands, ions, DNA, and RNA, not merely
  ligands; and
- upstream-public `encode` and `decode` should remain public.

A later full callable-axis audit found three further contract facts before Tok:

- feature `edge_size` is independent from the hidden width and must remain in the
  Packer config;
- training-time backbone coordinate noise is a real public Model control even
  though official inference uses zero; checkpoint `noise_level` is training
  metadata and does not select that runtime value; and
- upstream `use_atom_context=False` clears only the mask while its decoder still
  consumes the unmasked atomic-number one-hot. OFoldX uses the ordinary padding
  atomic number as well as a false mask.

The same parser audit found one previously omitted public axis. Before building
protein CA rows, atom37 observations, and non-protein context, the released path
drops atoms whose occupancy is not positive; a filtered CA therefore removes the
entire residue row. Its CLI can explicitly keep all occupancy values. The canonical
spelling is `filter_by_occupancy=True`: the exact `> 0` predicate also drops
negative and NaN values, while `False` ignores occupancy. One shared observation
mask drives all three tensor domains. The reader and packed Result retain supplied
occupancy facts; generated selected side-chain sites have unknown occupancy.

The path-input counterreview also rejected treating ProDy's mutable `protein` flag
as chemistry. Its name table changes by version, omits the CCD protein placeholder
`UNK`, and contains force-field aliases that collide with real non-polymer CCD
components. It also makes protein identity depend on an observed CA after occupancy
filtering, so removing one CA can turn the remaining protein atoms into apparent
ligand context. OFoldX instead freezes two reader-local literal groups: audited
peptide component IDs are protein directly, while colliding force-field aliases
require an ATOM declaration. CA and occupancy then affect Model observation, never
canonical chemical identity. This intentional upstream correction is recorded in
`UPSTREAM_BUGS.md`; it adds no runtime CCD provider, ProDy dependency, classifier
registry, or public taxonomy.

The resulting file API is one public family function,
`ofoldx.models.mpnn_packer.read_pdb(path, *, chains=None)`. It reads plain or gzip
PDB, the first physical model, blank/A altlocs, source order, hybrid-36 residue
numbers, exact element columns, and occupancy without applying the Processor's
filter. `chains` removes all source rows outside the selected raw chain IDs before
materialization and therefore cannot be replaced by packing `positions`. The
Processor and Pipeline accept the common Structure-or-path union; only their path
branch delegates to this reader. There is no generic reader resurrection, parser
class, row dataclass, source bag, format registry, or mmCIF claim.

Upstream's nominal `node_features` is not independently usable from `hidden_dim`,
`num_chain_embeddings` is inert, and `atom37_order` is a generic-layout switch
replaced by OFoldX's explicit atom14 tensor boundary. The official B × pack
design-to-pack cardinality remains mandatory on the complete concrete composition;
it is not attributed to LigandMPNN ownership or the standalone Packer pair. Whether
the five designer routes share one concrete class remains reopened.

A later coordinate-state counterexample prevents simplifying the public encoder to
`encode(input)`. The released sampler reconstructs atom14 from rigid frames and
literature positions before calling the encoder; the N/CA/C/O values read there are
therefore not guaranteed to equal `input.backbone_coordinates`. The minimal exact
boundary keeps explicit `atom14_coordinates` on `encode`, and adds the equally real
`atom14_mask` consumed by the released decoder to `decode` and `forward`. These are
ordinary Tensor arguments, not an `Atom14State` wrapper or another prepared input.

The learned graph and raw sampling path are batch-general. Canonical `pack` is not:
one `MpnnPackerInput` owns one Structure and one row mapping, so it must reject
`B != 1` before realization rather than silently decode batch zero. A future
design-to-pack Pipeline owns B × pack grouping through the raw operation. The
standalone torsion-proposal default is 16, matching the released CLI; the upstream
helper's default 10 remains an explicit value, while the combined command's four
packs per design remains composition state.

The dtype audit also found unconditional FP32 staging in the Processor and raw
mixture outputs. The correction preserves source precision until the caller's
explicit Processor conversion and leaves `forward`/`sample` in the Model
computation dtype. A BF16 counterexample then falsified the stronger Result claim:
PyTorch cannot convert BF16 directly to NumPy, and canonical Structure/Metric
values are NumPy-backed. `pack` therefore uses ordinary detach/CPU/NumPy conversion
without a hidden cast and may naturally fail for an explicitly unsupported dtype.
For representable dtypes it combines source and sampled coordinates without
downcast and preserves supplied occupancy. The manual BF16 diagnostic covers raw tensor
operations, not NumPy Result construction.
Checkpoint conversion likewise preserves the official weight dtype. A converter
`dtype` option would persist cast Model weights without a matching persistent
Processor runtime dtype and create a pair that is not runnable by default; callers
instead cast the assembled pair explicitly.

The rigid-geometry audit separated algebra from execution policy. OpenFold writes
out 3x3 matrix and matrix-vector products by hand specifically to evade AMP
downcasting. MpnnPacker uses ordinary PyTorch matrix multiplication instead: the
algebra is unchanged, default-FP32 parity is tolerance-based rather than bitwise,
and an experienced caller's dtype, autocast, and matmul-precision choices are
obeyed. This removes a hidden precision exception rather than changing the model.

Two further counterexamples corrected the lowering design. First, the shared Mpnn
alphabet cannot own `MSE -> MET`: only ProteinMPNN performs that rewrite, while
LigandMPNN, NA-MPNN, and the Packer use their unknown protein token. Packer result
realization must therefore parse atom14 identities only for selected canonical
rows, leaving unselected modified rows untouched. This is released behavior, not
an upstream bug. Second, ProDy's water-name table changes by version, and the later
12-name table collides with real CCD components `SPC`, `SOL`, and `T3P`. Neither a
9-name dependency snapshot nor the 12-name table is a stable scientific contract.
LigandMpnn and MpnnPacker now use one direct predicate: a candidate alias must
declare oxygen and no element outside H/O. This keeps O-only partial water without
dropping the colliding inhibitor, L-sorbose, or nucleotide. It adds no Water type,
graph inference, CCD provider, provenance, or dependency pin.

The two `atom_context_num` configs also do not express one duplicated value. The
Processor count controls emitted nearest atoms and the Model count controls decoder
projection width. Upstream accepts a wider Processor extent and uses the extra
atoms in the encoder; only a shorter extent reaches a projection-shape failure.
The released converter writes 16 to both, but OFoldX adds no pair validator,
PairConfig, compatibility matrix, or synchronization layer.

The smallest lossless shared output is one `PackPrediction(structure, metrics)` per
sample inside `PackResult(predictions)`. It has no provenance, diagnostics, score,
or `best()` method. The exact Packer API and decoded carrier are now frozen in
Sections 4.6 and 5.5 of `DESIGN.md`.

Three additional upstream defects were confirmed during this audit: the documented
zero-pack fast path returns no packed structure, fixed side chains use a theoretical
atom-existence mask instead of the parsed observation mask, and disabled atom
context still leaks atomic identities. They are recorded separately in
`UPSTREAM_BUGS.md`.

A later adversarial review reopened and corrected four parts of this contract.
First, complete audited residue components inside one gapped PDB chain silently
created peptide adjacency between consecutive observed rows. Residue-number gaps
cannot repair this: a skipped number can still be bonded, while a same-chain jump
can span unresolved residues. The bounded reader now keeps one chain axis but makes
every protein occurrence topology-open, with only the atom identities the Packer
can consume or generate. No generic segment or inferred-break abstraction was
added.
Programmatic topology-open standard residues receive the same minimal atom14
declarations once per entity, without sites, coordinates, bonds, attachments, or
optional chemical claims; this closes Result identities without making chemistry
depend on instance-local selection. Complete components are never silently
supplemented.

Second, explicit PDB deuterium and tritium are canonical hydrogen isotopes, not new
elements. Blank or invalid element columns for arbitrary partial atoms now fail at
the importer boundary, while audited standard atom IDs can supply their known
element without inferring from the atom name. Alternate locations are filtered
before identity evidence, so discarded conformers no longer control chemistry.

Third, boolean observation masks had been cast to the Processor's floating dtype.
Under AMP this promoted masked activations back to FP32. Masks now remain boolean,
and the residue decoder receives the same source-by-neighbor validity mask as the
encoder. Valid unknown-token protein rows explicitly retain their observed
backbone during every denoising update instead of becoming plausible zero-coordinate
neighbors. The latter two inherited scientific defects are recorded in
`UPSTREAM_BUGS.md`.

Fourth, canonical `Structure.sites` order is caller-owned; rebuilding every result
in `system.iter_atoms()` order silently reordered unrelated observations. Packed
results retain source-row order and append only newly generated heavy sites in the
Model's deterministic row/atom14 order.

The official packed-PDB chi-density visualization remains a required file-output
path in the later Structure/I/O Tik. It does not justify restoring `b_factors` as a
scientific raw Model output or canonical Structure annotation.

## Rewriting models preserves behavior, not source layout

The pre-release refactor may completely rewrite Models, Processors, converters,
and Modules. The committed implementations are not a file-preservation constraint;
their public-output parity and implemented capabilities are behavioral evidence.
The complete released upstream surface remains authoritative; the exact revision
recorded by an audit makes that evidence reproducible without pinning an OFoldX
dependency or public behavior.

Each family therefore keeps three independent accounts: committed parity-backed
behavior, upstream behavior still missing from that baseline, and integration into
the new Model/Processor/Hub/Pipeline architecture. Large source churn can be the
correct consequence of removing dispersed and duplicated implementations. It does
not establish completion by itself, and it is not a reason to restore the old
layout.

## Canonical entity splitting is family-owned

The earlier frozen rule grouped realized `PolymerSequence` values using ordinary
dataclass equality. A counterexample disproved that as chemical entity identity:
the current equality changes when reference coordinates differ, and it treats the
two declaration directions of the same undirected `ChemicalBond` as unequal.
Reference conformations do not define PDBx entity identity.

The earlier `frozenset[SystemBond]` claim also implicitly froze equality and hashing
before the complete public bond fields were known. Reversed endpoints currently
produce unequal values, while the final order, stereo, and origin domains remain
open.

Current conclusion:

- Section 3.4.1 freezes only explicit entity-to-instance ownership, positional
  component occurrence identity, exact `AtomSite`, and direct component lookup;
- paired polymer attachments separately freeze which leaving atoms are retained;
- complete `SystemBond` semantics and materialized graph traversal remain open;
- canonical values never infer entity identity from value equality or Python object
  identity; and
- each fixed-length family explicitly constructs its output partition from facts
  established by that released operation.

For MpnnPacker, the family-owned fact is the complete target
component-ID tuple within each original entity. The deterministic suffix-allocation
rule is frozen in Section 4.6. Other inverse-folding families still require their
own Tik rather than inheriting it accidentally.

## RFdiffusion has three distinct sequence/structure controls

The released RFdiffusion contract disproves the earlier selection-wrapper
simplification. `inpaint_seq` hides sequence on existing source residues,
`inpaint_str` independently hides their coordinates, and partial diffusion's
`provide_seq` reveals sequence by realized output row and can apply to generated
contig rows. They are not aliases and cannot share one biological wrapper without
losing official behavior.

Only the first bounded argument is now frozen:
`inpaint_sequence: Sequence[ComponentReference] = ()`. It uses ordinary canonical
references, masks every realized placement of a selected source occurrence, and
rejects a selected occurrence absent from the output. The complete RFdiffusion Tik
must still place `provide_seq`, structure inpainting, placement-aware secondary
structure, scaffold block adjacency, and every released potential and sampling
control before the family can publish.

Upstream uses `self.ref.index(res)` for sequence and structure inpainting, so a
source residue placed twice masks only its first copy. That behavior is recorded in
`UPSTREAM_BUGS.md`; OFoldX applies the only meaning expressible by a source
occurrence reference and masks all placements.

## Protein, Ligand, and Membrane Mpnn contract resolves the expanded inventory

The earlier bounded inventory correctly falsified the distinct-CA-pair design.
Full-backbone and CA-only checkpoints have the same scientific operations and
differ only in the coordinate/feature graph, so one Model–Processor pair owns an
explicit `ca_only` field on each config. The two fields are not synchronized by a
pair validator; converted repositories write the matching facts and manual
mismatches fail ordinarily.

The expanded Protein/Ligand/Membrane inventory falsified the earlier surface and
then established its complete replacement. All three Models expose four named raw
computations: autoregressive, sequence-free autoregressive, conditional target-last,
and unconditional target-first probabilities. Every command-owned computation
retains its repeated evaluation axis and effective ranks. Protein alone additionally
owns the original scalar NLL `score`. The original Protein sampler has 21 reachable
classes and can generate X when explicitly allowed; Ligand and Membrane retain 21
raw output classes but generate only the first 20.

The rewrite deletes the nested `ProteinMpnnLayout`, one-field output wrapper,
Backbone forwarding wrapper, generic feature dictionary, generic inverse-folding
helper, condition outcomes, diagnostics, and provenance. One plain Input owns its
tensors and exact canonical associations. The Model itself is the ordinary
PyTorch graph; `ProteinMpnnFeatures`, `CaProteinMpnnFeatures`, and the actual
message-passing layers remain public Modules.

The new `Structure.residue_numbers` fact remains necessary, but it does not close
all path lowering. Original ProteinMPNN inserts numeric-gap rows and applies its
designed-chain-first ordering; the updated route retains physical CA rows without
those synthetic gaps. `ProteinMpnnProcessor.process_pdb` owns the one-consumer
original fixed-column lowering directly to `ProteinMpnnInput`; it does not pretend
those transient gaps are canonical Structure facts. Direct Structures, including
the shared public `mpnn.read_pdb` output, use the updated canonical lowering.
Ordinary `processor(path)` selects the original all-chain route, while
`process_pdb(path, chains=...)` preserves the official graph-removing chain filter.

The checkpoint inventory also proves one additional raw architecture rather than
four variant Models. Vanilla, soluble, and the tracked training checkpoint are one
full-backbone ProteinMpnn graph; CA-only remains the same root pair with its
different feature graph selected by architecture config. Global-label and
per-residue-label membrane checkpoints share a second exact graph and therefore
one `MembraneMpnnModel`. Their biological inputs differ: one Processor lowers a
whole-structure Boolean, while the other lowers explicit per-component
other/interface/buried labels. They require two concrete Processors and Pipelines,
not a Model mode, enum/spec wrapper, or `SolubleMpnnModel`.

Tied coefficients are final signed neural-logit coefficients in
`sum(c_i * logits_i)`, never normalized by their sum. Protein defaults omitted
members to `1 / len(group)`; Ligand and Membrane default them to one. Original
Protein `tied_beta` converts to `c_i = tied_beta_i / len(group)`, while updated
symmetry weights convert directly. Updated protein/soluble parity therefore passes
ones explicitly rather than changing defaults by input route. Signed coefficients
touch neural logits only; group-constant residue/PSSM soft controls apply once,
hard support intersects last, fixed anchors agree, and tied members share one
causal rank.

The complete callable audit additionally preserves upstream caller-controlled
autoregressive order as one Model-only `decoding_ranks` tensor; random priorities
and permutation aliases are rejected. Sparse PSSM mappings are one paired control:
supplied probability rows default to coefficient one, absent rows to zero, and a
coefficient cannot exist without its probability distribution.
Tied weights are accepted only on real non-singleton tied groups, and sparse
log-odds absence is represented by positive infinity under a finite threshold;
neither control can become silently inert through a lowering default.

State evidence distinguishes CA by its 167-wide edge input, full Protein by
416-wide edges without `W_v`, Ligand by context-layer keys, and Membrane by
416-wide edges plus live node-label projection. Global and per-residue membrane
artifacts have identical key sets and shapes, so two explicit converters select
their two concrete Pipeline markers; filename or state guessing cannot. The
current worktree implements only the earlier bounded pair and is not publication
evidence. All 18 unique sequence-design artifacts require on-demand conversion,
load/save, route, and numerical parity before publication; this is never a
frequent-CI gate.

An adversarial post-freeze review found four public-boundary counterexamples, so
the affected clauses return to Tik before family Tok. First, Membrane proves that
a Processor may have no persistent lowering config, while the current Auto,
registration, and repository contracts require a concrete Processor config class.
An empty config cannot be invented to satisfy that framework premise. Second,
Protein raw training `forward` is genuinely batched, while sampling, probabilities,
score, design, and their canonical Results describe one biological target; the
per-method batch boundary must be explicit rather than silently selecting target
zero. Third, the released combined side-chain-packing route is reachable for
protein, soluble, ligand, global-membrane, and per-residue-membrane designers; the
then-current Ligand-only contract omitted four official call paths even though
`MpnnPackerPipeline.pack_sequences` supplies their common implementation handoff.
Fourth, Ligand probability Results contain no interface mask or interface metric,
so `ligand_interface_cutoff` on those four Pipeline methods is inert and must be
removed.

The same subtraction review found that Protein, Ligand, and Membrane raw outputs
have identical fields and meanings, as do their sampling outputs. Three family
spellings do not encode a domain invariant. One concrete shared `MpnnOutput` and
one `MpnnSamplingOutput` in `ofoldx.models.mpnn` are the smaller honest carriers;
their family-specific vocabulary extents remain observable tensor shapes. This is
the same bounded multiple-consumer justification as `MpnnProbabilityResult`, not
a generic output hierarchy.

A follow-up exact audit of all five released combined-command `model_type` routes
resolves the design-to-pack counterexample. Protein and soluble weights share one
`ProteinMpnnDesignAndPackPipeline`; ligand, global-membrane, and
per-residue-membrane routes each require their own explicitly named class. These
four live-only compositions retain their concrete designer signatures and share
only `MpnnPackerPipeline.pack_sequences`. Every path is parsed once with
`mpnn.read_pdb`; notably, the Protein composition therefore selects the updated
Structure arm rather than the standalone Protein Pipeline's original path arm.
This evidence proves neither a generic composition base nor an aggregate artifact.
The exact contracts are frozen in Sections 4.6–4.9 of `DESIGN.md`.

## Complete upstream surface audit reopens incomplete family routes

An independent read of the audited official repositories invalidated the narrower
surfaces described by several current family cards. Those cards and existing raw
Models are implementation material under review; none may redefine a family as the
convenient subset already ported.

The Protein/Ligand/Membrane finding from that audit is now resolved by the frozen
contract above: full-backbone and CA-only graphs share one explicit
config-controlled Protein pair, while Ligand and Membrane own their concrete
surfaces. The current code still lacks that complete probability, sampling,
coefficient, reader, and membrane implementation, so resolution of Tik does not
make the worktree complete.

| Family | Official surface currently missing or wrongly deferred | Required ownership consequence |
| --- | --- | --- |
| LigandMPNN | Per-residue membrane, global membrane, soluble, and protein model types | One Protein graph for ordinary/soluble weights, one Membrane Model graph, and two concrete membrane Processors/Pipelines because whole-structure and per-residue labels have different biological signatures. |
| RFdiffusion | Independent structure inpainting, `provide_seq`, scaffold SS/block adjacency/crop, and the `rep_r_min=0` all-pairs substrate branch | Concrete RFdiffusion controls and potential Modules; the branch is scientific behavior, not an escape hatch. |
| Boltz-2 | Forced contact/pocket/template guidance, complete fold/confidence outputs, embeddings, and two-stage affinity | Family-owned force/threshold semantics and the released concrete affinity Pipeline. |
| Protenix | Nine released v1/v2/ESM/ISM variants, RNA-MSA/template behavior, and TFG | Complete variant pairs and explicit TFG sampling/potential surface. |
| ESMFold2 | Experimental graph, soft-sequence inversion, its checkpoints, and released binder-design composition | Separate Experimental pair(s) and a concrete binder-design Pipeline. |
| AlphaFold 2 | AlphaFold-Multimer, all monomer/PTM presets, ensemble/recycle controls, and relaxation | Distinct Multimer pair(s), complete heads and controls, and an explicit released postprocess path. |
| AlphaFold 3 | Canonical Processor/fold, seed × sample grouping, embeddings, contact probabilities, and complete confidence summaries/matrices | A complete concrete fold surface rather than only the raw tensor graph. |
| Boltz-1 | Canonical fold, steering/FK potentials, embeddings, and complete confidence outputs | Concrete fold arguments, public potential Modules, and complete Result decoding. |
| BoltzGen | Released design → inverse-fold → fold/affinity → analysis/filter/ranking system and complete schema controls | Raw pairs plus one explicit multi-pair BoltzGen Pipeline; no generic workflow or Condition bag. |
| ODesign | OInvFold modality checkpoints and full beam/temperature/top-k composition | Separate OInvFold pairs plus the concrete ODesign Pipeline; uncertain de novo ligand identity must stay honest in a family Result. |
| PXDesign | ProteinMPNN sequencing, AF2-IG/optional Protenix evaluation, ranking, filtering, and all controls | The released concrete multi-pair PXDesign Pipeline. |
| NA-MPNN | Mixed-polymer token support, source residue ordering, incomplete-residue chemistry, token symmetry, pair bias, opposite-order scoring, specificity, core objectives, and two released weight distributions are not represented correctly by the current worktree | One direct mixed-polymer Model and two weight repositories; a concrete Processor/Pipeline and typed probability/sampling outputs, with training-data ownership frozen separately rather than hidden in Model config. |

This also reopens the apparent shared `Contact`/`Pocket` contract: Protenix
constraints and Boltz forced guidance use similar nouns but do not yet have proven
identical semantics. The common top-level fold API must not erase Boltz's `force`
bit or reinterpret another family's conditioning. A dedicated Tik is deciding
whether any smaller shared value survives.

The earlier 35–40% estimate mixed shared Hub infrastructure progress with a family
denominator that this audit proved incomplete. It is no longer a valid overall
completion percentage. No replacement numerator or denominator is inferred from
package count: implementation, registration, manual parity evidence, and external
Hub publication are distinct facts. The concrete current boundary is recorded
below.

## Current Model Hub completion boundary

The shared root-pair loader/writer and registries exist. A one-pair concrete
Pipeline repository now writes an exact `pipeline.json` marker last, resolves the
registered Pipeline and requested task before weight hydration, loads the root
Model/Processor pair, and constructs the Pipeline through its ordinary explicit
constructor. `AutoModel` and `AutoProcessor` deliberately ignore that marker. An
on-demand node4 run of the maintained MpnnPacker script exercised all of those
paths through a fresh interpreter; no model-side test or committed parity artifact
was added.

The current worktree has built-in registrations for ProteinMpnn and MpnnPacker.
Only the independently frozen MpnnPacker registration currently represents a
complete bounded vertical; ProteinMpnn registration is implementation material
under its reopened family audit. Neither state implies that weights have been
published to the Hub. Remaining families stay open until their own complete surfaces
provide affirmative evidence.

The multi-pair concrete repository lifecycle is now RATIFIED and frozen (DESIGN §7.4
"Multi-pair concrete Pipeline repositories"). A multi-pair Pipeline declares a
class-level pair-layout — an ordered set of named Model/Processor pairs (a runnable
root pair plus fixed sub-directories such as `structure/`, `protein_designer/`,
`ligand_designer/`, `ifold/`, and optional `fold/` / `affinity/`) mapping to its
constructor keywords — and carries a shared `save` / `from_repository`
(`ofoldx/pipelines/_composite.py`) built on the ordinary single-pair writer/loader.
`pipeline.json` stays the frozen one-field marker (the layout lives in Python, never
serialized), sub-pairs carry no marker, and the repository factory (`api/tasks.py`)
now routes a marked multi-pair repository to `from_repository` on the pinned snapshot
instead of the one-pair `(model, processor)` call, with one-pair routing unchanged.
`Boltz2AffinityPipeline`, `ODesignPipeline`, and `BoltzGenPipeline` are the three
materially different verticals that prove it; all three are promoted from
owner-reserved-but-deferred ids to routable `_BUILTIN_PIPELINE_MODULES` entries and
`register_pipeline`-ed, with owner squat-reservation preserved. A node4 run exercised
`save` → on-disk root + named sub-pairs + one-field marker → factory `from_repository`
reconstruction with per-pair weight round-trip through a fresh interpreter for each of
the three; this is implementation evidence, not a committed test or publication gate.
Multi-pair *placement* (DESIGN §6.3) and complete family checkpoint publication stay
open.

Shared Hub infrastructure has substantial implementation, but no overall percentage
is defensible until every official family surface has a complete inventory.
Committed-era public-output E2E parity is valuable behavioral evidence, but it does
not prove current Processor, Result, repository, task-Pipeline, or complete-family
behavior. Model rewrites compare against that evidence without preserving the old
source organization.

## MpnnPacker subtraction fixed geometry ownership and completed registration

A real-checkpoint baseline proved exact conversion of all 215 tensors, the complete
Processor boundary, public encode/decode/forward, all eight sampling fields, and
canonical PackResult semantics. The subsequent performance audit found that fixed
atom masks, chi indices, rigid frames, group indices, and literature positions were
still rebuilt from Python/NumPy inside every sample and denoising step. A separate
Geometry Module or cache would add an entity without another consumer.

The accepted design instead makes the root Model own eight derived non-persistent
buffers already arranged in its 20-amino-acid-plus-unknown token order. This removes
the mutable global remapping Tensor, repeated host construction and device copies,
and any need for lazy mutation or loader hooks. The ordinary internal meta loader
constructs these tiny constants on CPU because `assign=True` does not hydrate
non-persistent buffers; later `.to(...)` retains normal PyTorch ownership. The
state-dict keyspace stays unchanged.

The first post-subtraction parity run exposed a numerically meaningful false
simplification: replacing the released `Rᵀp + (-Rᵀt)` sequence with the algebraically
equivalent `Rᵀ(p - t)` changed normalized FP32 torsions by up to `1.67e-6`, and the
iterative local argmax amplified that into different mixture concentrations. The
released two-matmul order was restored. A complete rerun then matched all 215
converted tensors, 11 Processor tensors, public encode/decode/forward, all eight
sampling fields, canonical PackResult behavior, the PDB writer, and the local
save/Auto/concrete-Pipeline repository round-trip.

A BF16 repository probe raised a separate ownership question. Persisting the eight
tables would partially serialize a prior runtime cast while device, Processor dtype,
AMP state, and other runtime placement remain deliberately unsaved; it would still
not make an arbitrary cast pair runnable by default. The accepted contract keeps
`dtype=None` literal: serialized Model state retains its stored dtypes, derived
constants and Processor retain constructor defaults, and no weight is treated as a
whole-pair dtype oracle. Experienced callers pass `dtype=` for an explicit whole-pair
cast or use AMP; an unrequested incompatible mixture fails in ordinary PyTorch.

The pair and its concrete Pipeline are registered as `ofoldx.mpnn_packer`. An
ephemeral converted repository round-trip loaded the exact registered Model and
Processor through Auto, selected `MpnnPackerPipeline` through `pipeline("pack",
...)`, performed one canonical pack, preserved all 215 learned state keys, and
applied an explicit BF16 request to the Processor, learned weights, and floating
derived buffers. This is implementation evidence, not a committed test or
publication gate.

## MpnnPacker packed-PDB output does not justify another Result payload

The complete-file audit falsified the earlier claim that the released packed-PDB B
column could be reproduced from the physical chi metrics without first deciding
whether its nonphysical values were part of the contract. The released code writes
literal `2.0` for every fixed row and maps backbone plus selected ALA/GLY atoms to
head zero even though those atoms have no corresponding physical chi density. Two
predictions with the same canonical Structure and physical metrics can therefore
require different raw upstream B columns.

The subtraction review rejected both family-specific Result clones and a
`pdb_b_factors` field. Persisting the raw column would promote serialization defects
to public task output. The frozen repair keeps `PackPrediction(structure, metrics)`:
the family `write_pdb(prediction, path)` uses the existing physical chi metrics only
for atoms in their actual rigid groups and writes zero elsewhere. One prediction is
one explicitly named file; the writer never asks for Model input or raw sampling
state and never guesses multi-sample filenames.

Chain allocation is likewise defined only over instances that emit records. The
real 1BC8 input contains more than 62 water-only instances; allocating identities
before the required water omission would fail on objects that have no serialized
chain field. Retained one-character identities still win, and every other emitted
instance receives an unused chain deterministically in MolecularSystem order.

The first writer review rejected an alphanumeric-only chain pool, whitespace-bearing
identity fields, unrestricted insertion characters, and unchecked occurrence-key
collisions. PDB chain IDs can retain printable punctuation and the blank chain is an
available code; fixed-column padding cannot silently collapse canonical component or
atom IDs, and two component occurrences cannot share one emitted `(chain, resSeq,
iCode)`. These are local representability invariants, not a generic validation
layer. The smallest repair uses the full bounded character pool and one local emitted
occurrence map; no helper class, identity sidecar, or new Result field is justified.

The same audit confirmed that the current family reader intentionally discards
nonprotein author-chain identity and source PDB charge fields even though canonical
`ChemicalAtom` can represent formal charge. Reopening the entire canonical Structure source
model merely to claim byte/source round-trip would be disproportionate and would
not repair the many identities the upstream writer itself drops. The accepted
boundary is a deterministic canonical visualization with truthful atoms,
coordinates, occupancy, elements, record categories, and available residue
numbers—not a PDB source round-trip. The exact upstream divergences are recorded in
`UPSTREAM_BUGS.md` for owner review and manual parity reporting.

## Architecture family is the Tik–Tok coordination unit

Every concrete released Model first receives its own checkpoint, callable, output,
and command-path inventory. The architecture-family Tik then computes and tries to
falsify their exact shared implementation intersection before any sibling Tok
changes that shared boundary. This prevents ProteinMpnn, LigandMpnn, and NaMpnn
from independently inventing incompatible copies while another sibling is still
revealing what is genuinely common. Unrelated families may still progress in
parallel.

Family coordination creates no runtime family object, registry field, inherited
config, generic Model, or publication identity. Concrete Models retain their own
complete scientific contracts and files; only proven shared Modules, geometry
primitives, parsers, or literal converter mappings receive one integration owner.
After the shared boundary and every affected concrete contract are frozen,
model-owned Tok work may proceed in parallel. Contrary evidence reopens only the
affected family boundary and concrete contracts, never unrelated families.

## MPNN family sharing stops at graph primitives

A three-way comparison against ProteinMPNN revision `8907e667`, LigandMPNN
revision `26ec57ac`, and NA-MPNN revision `aafa32c8` falsified a shared Model,
Input, config base, sampler, output, or request contract. Their exact common graph
is smaller: encoder and decoder message-passing layers, feed-forward/message MLPs,
relative-position encoding, gather operations, residue KNN, the 2–22 Å radial
basis, virtual Cβ, and the underlying teacher-forced/per-step decoder equation.
Token embeddings, output heads, feature projections, feature encoders, sampling
controls, chemistry lowering, and decoded outputs differ.

The family Tok may extend the existing public graph primitives only where the
result has multiple concrete consumers and removes code without callback or
union-option machinery. It must not introduce `MpnnModel`, an inherited config,
or a generic sampler merely because most checkpoint tensors have compatible
shapes. Converter code may share literal key-suffix tables; it never shares
weights or makes physical-key similarity a public architecture.

The same audit found operation choices currently stored in config. Ligand side-chain
context is selected inside the Model after coordinate noise and residue KNN, while
external atom-context enablement and the ligand-interface cutoff are invocation
choices. NA sampling temperature is likewise an invocation choice; its released
design and specificity repositories do not require distinct architecture configs.
These fields must leave persistent config in the concrete family contracts.

LigandMPNN's four released probability computations also cannot be collapsed into
the current scalar `ScoreResult` or hidden behind a conditioning Boolean. They are
sequence-conditioned autoregressive score, sequence-free autoregressive
probabilities, conditional probabilities, and unconditional probabilities. The
autoregressive pair and both single-residue computations retain their released
repeated-evaluation axes. Unconditional probabilities put each target first and
retain aligned effective ranks; they are not collapsed into one all-equal-rank
backbone pass merely because some values can coincide. The upstream single-residue
`use_sequence` Boolean is inverted, as recorded in
`UPSTREAM_BUGS.md`; OFoldX names the four computations directly.

NA-MPNN provides a counterexample to treating every MPNN symmetry as exact
chemical-identity equality. In its released shared-token mode, DNA and RNA rows
may be tied while retaining different canonical components: DA/A, DC/C, DG/G, and
DT/U share model tokens. `TiedResidueIdentities` cannot express that operation
because it promises the same component identity at every member. Rejecting all
cross-polymer ties would delete a reachable official symmetry mode.

The concrete NaMpnn API therefore needs an ordinary family-owned
`symmetry_groups` argument over component references in addition to any exact-
identity conditions it still proves. It constrains sampled token equality; decode
realizes the same shared token through each row's polymer type. Validity follows
from the intersection of per-row hard token support, so a shared-token DNA/RNA
group may succeed while an incompatible separate-token group fails. This adds no
SymmetrySpec, Condition subtype, generic tie semantics, or cross-family request.

The complete Na audit also falsified separate design/specificity Model variants.
Both released checkpoints have the same 123-tensor architecture; `mode` selects a
weight distribution and CLI sampling defaults, not a config branch. Node and edge
feature widths are not two live axes because upstream swaps them, but one shared
`feature_size` and one independently live `hidden_size` do execute and must not be
collapsed. Temperature, neighbor override, decoding order, protein-first teacher
forcing, sample count, bias, and seed/generator remain operation arguments. The
three polymer-specific coordinate noises belong only to `encode` and the
teacher-forced `forward`; released inference does not expose augmentation controls
on score, sampling, design, specificity, or unconditional probabilities.
Processor persistence proves only `shared_nucleic_tokens`; `rna_as_dna` belongs to
training-data lowering rather than the released inference Processor.

Na can consume none of the current `models/inverse_folding.py` functions. They
hardcode the Protein/Ligand alphabet, protein-only row selection, synthetic residue
indices, and component-equality realization. Its PDB route instead owns physical
CA/C1-prime row order and source-integer arithmetic. Insertion code is retained as
source identity but does not participate in the released feature-index arithmetic.
RNA/DNA identity comes from the canonical component, never observed O2-prime
presence. Threading Na through that helper with callbacks would preserve the wrong
abstraction. The concrete Model directly owns its feature encoder and graph; a
prefix-only `NaMpnnBackbone`, `Layout`, feature dictionary, stored random noise, or
diagnostic wrapper has no independent consumer.

The audit inventoried the released training objectives, preprocessing functions,
and dataset transformations rather than using “unified Runner later” to delete
them. The bounded Model-Hub Tok may implement the complete inference pair and core
objectives while dataset/Runner ownership remains a separate Tik, but the family is
not declared complete or published until those scientific functions have an
accepted owner. Model-side synthetic tests remain excluded; real behavior evidence
belongs in `scripts/parity` and Slurm.

**Verdict: FREEZE the Model Hub contract.** The complete follow-up audit resolved exact PDB
row lowering, modified and hybrid chemistry, batched raw-forward axes, canonical
candidate partitioning, total-count versus chunk-size multiplicity, and
PDB/statistics output ownership. One `NaMpnnInput` remains justified for its aligned
tensors and direct canonical realization references; graph operations touch only
its tensors. A dense `token_group_ids[B,L]` replaces nested Python symmetry groups
in the sampling hot path. Aligned residue bias and tied weights use ordinary
zero/one tensors, while the dominant `[B,L,V,L,V]` pair tensor remains `None` when
absent.

The subtraction pass deletes the one-field `NaMpnnSpecificityOutput` in favor of a
plain `[B,L,V]` Tensor, deletes the speculative FASTA adapter, and does not freeze
publisher-selected Hub repository names. Tensor/config/count/numerical failures
belong to Python and Torch. Only ambiguous hybrid chemistry, conflicting exact or
token equality, inconsistent fixed anchors/ranks, seed-versus-generator ownership,
and biological design on `B != 1` justify explicit scientific failures.

The simultaneous bias equation is now explicit. `global_logit_bias[B,V]` applies
once per token group; `residue_logit_bias[B,L,V]` sums over member target rows;
signed tied weights multiply neural logits only. A full
`pair_logit_bias[b,i,a,j,c]` is a directed additive contribution to target row `i`'s
candidate `a` when context row `j` carries `c`. A simultaneous group sums every
member target row, keeps self at PAD, uses the pre-rank-block state for external
context, and evaluates another internal member at the candidate token. Both
directions of a mirrored internal pair therefore contribute without averaging or
deduplication. Hard support intersects last, then temperature and one group draw
apply. This reduces exactly to the released untied equation for a singleton and
requires no provenance scan, aggregation enum, policy object, or fallback mode.

## LigandMpnn remains a bounded reopened vertical

The complete LigandMPNN audit proves one released graph width across the four
ligand-aware checkpoints. Independent root node and edge widths are false axes,
although the public feature encoder may retain independently useful module widths.
Checkpoint noise labels are training provenance. Runtime atom-context selection,
side-chain-context selection, and coordinate noise are operation arguments rather
than persistent Model identity.

Processor and Model context extents are separate live facts. The Processor chooses
the initial external atoms; the Model reselects the final extent after fixed-residue
side-chain atoms have been merged. Neither value consumes learned state, and no
validator synchronizes them. The direct input owns one Structure and aligned
tensors, not a fake batch, Backbone, Layout, generic feature dictionary, stored
noise, provenance bag, diagnostic wrapper, or outcome layer.

The later adversarial audit and released
`ligandmpnn_v_32_010_25.pt` counterexamples falsified the frozen probability
surface. Correct target-last conditional logits changed by `0.04179287` when
only the relative order of other residues changed, so conditional
`[N, L, 21]` logits and `[N, L, L]` effective ranks are scientific output.
Correct target-first unconditional logits were invariant, and one all-equal-rank
pass matched all per-target-first rows exactly. Sequence-free autoregressive
probabilities are not unconditional: under order `[0, 1, 2, 3]`, their
per-target maximum logit differences from unconditional were
`[0.0, 0.1511825, 0.3885993, 0.9284916]` because that equation masks earlier
encoder neighbors.

The minimal public surface is therefore `score`,
`backbone_autoregressive_probabilities`, `conditional_probabilities`, and
`unconditional_probabilities`, with no public conditioning Boolean. The first
three retain their genuine decoding-order evaluation axis; unconditional is one
simultaneous pass. One aligned `LigandMpnnProbabilityResult` suffices and adds no
operation tag or axis-dependent aggregate API. Canonical component references
still make path and Structure outputs use the same biological row alignment.

One `sample` method dispatches tied and untied groups internally. Tied neural
logits use the released signed, unnormalized weighted sum; group-constant external
bias is applied once, hard support is intersected last, and members receive equal
causal ranks. The adjusted sampling probabilities remain part of the sampling
output. There is no parallel `tied_sample` method or condition Boolean.

The parse route is not Packer-specific: LigandMpnn, the membrane variants, and
MpnnPacker consume the same released PDB lowering. The public
`ofoldx.models.mpnn.read_pdb` therefore has multiple concrete consumers, while the
packed visualization writer remains Packer-owned. Conversely,
`models/inverse_folding.py` encodes a false generic protein realization and must
not survive as the route through which the family is unified.

The released design-to-pack surface is four explicit live classes, not another
Model artifact: Protein/soluble, Ligand, global Membrane, and per-residue Membrane.
Each delegates placement to its two already assembled child Pipelines but owns no
aggregate save, Hub marker, manifest, or loader hook. The reusable batch is
`MpnnPackerPipeline.pack_sequences`: one source Structure, one packing scope, and
ordered target sequences. Each composition hands off only canonical source,
candidate identities, design scope, and one RNG stream; no tensor bridge,
BatchInput, workflow object, composite Result, or generic base survives
subtraction.

The complete Ligand pair and all four composition routes are now frozen in
Sections 4.6–4.9 of `DESIGN.md`. Real-checkpoint evidence remains an on-demand
script under `scripts/parity`, never a model-side test or frequent-CI gate.

## AlphaFold3 direct AF-RSA is frozen; CCD distribution remains open

**Verdict: FREEZE direct AF-RSA and REOPEN only the CCD asset decision.** An exact audit of AlphaFold 3
revision `7b197fe859790fc3e04d03ea70dd0b9ba48881c9`, tag `v3.0.3`, and the local
released checkpoint has now closed the tensor, document, operation, decoded-result,
writer, converter, and ranking-RSA contracts. Full-family Tok still cannot finish:
complete canonical input support needs one exact CCD snapshot with reviewed
provenance, redistribution terms, and offline packaging. The released ranking path
does not require a runtime DSSP or libcifpp dependency.

The released AlphaFold 3 operation does not fit the standard single-Processor-call
fold route. For each user seed, upstream uses that same integer for randomized
featurization and Model sampling, then produces one seed-level trunk/distogram/
contact/optional-embedding output and several diffusion/confidence samples. The
complete call retains every seed group and chooses the global best ranking score in
seed-major, sample-major order. Duplicate seeds are permitted and order is
observable.

The current OFoldX `AlphaFold3Model.forward` is only one denoising step, not the
released full operation. The audited replacement is the complete Model call:
eleven trunk executions by default, per-recycle MSA reshuffle and crop, five
diffusion samples, 200 diffusion steps, confidence for every sample, seed-level
contact/distogram output, and optional single/pair embeddings. Registration uses a
concrete `AlphaFold3Pipeline` marker with `standard_tasks=()`; the standard
single-Processor-call fold route is lossy for this family.

`AlphaFold3Document` is justified by real JSON and writer consumers. It preserves
the canonical `MolecularSystem`, job name, ordered duplicate seeds, descriptions,
the `None` versus explicit-empty MSA/template distinction, and raw user CCD. It
reads official JSON versions 1 through 4 and AlphaFold Server JSON and writes
normalized inline version 4. A generic `FoldRequest` remains unjustified.

One unbatched `AlphaFold3Input` has 59 compute tensors in the official MSA,
template, token, prediction-layout, reference, nine gather, and frame groups. One
public `AlphaFold3Layout` retains only canonical identity and decode layout and
never enters Model placement. The exact raw output owns sample atom coordinates
and masks, expected pLDDT/resolution, full and average PDE, full and TM-adjusted
PAE, contact probabilities, optional distogram and embeddings, and the persistent
model identifier. The present logits-shaped output is therefore replaced, not
wrapped.

A named `AlphaFold3SeedResult` is now justified because every seed owns its exact
processed input, typed raw output, and decoded `FoldResult`; it is not merely a
name for `(seed, FoldResult)`. `AlphaFold3Result` preserves ordered duplicate seed
groups and the originating document. Global ranking flattens them only transiently
in seed-major, sample-major order.

The AF3 artifact audit also found two constants missing from the existing frozen
loader rule. Official noise-level Fourier weight and bias are literal 256-value
source constants, deliberately fixed to avoid future JAX RNG drift, and are absent
from the released 405-record parameter schema. Current OFoldX instead stores them
as non-trainable `nn.Linear` Parameters and makes the converter manufacture two
target state keys. That creates artifact authority unsupported by the checkpoint.
Conversely, the checkpoint `__identifier__` record is live: the runner injects it
into raw Model output and every final ModelCIF. It remains persistent Model state,
appears in the typed raw output, and reaches every family writer; it cannot be
grouped with the eight materialized-but-unused floating records or discarded by
conversion.
AF3 owns the exact constants in one public `AlphaFold3NoiseEmbedding`: fixed width
256, two FP32 nonpersistent buffers, and the released cosine computation.
Converters must not serialize it. The denoiser must be constructed with the
correct Module directly; construct-then-replace would create and discard random
state and obscure the actual module tree. This does not change Boltz-family Fourier
projections merely because they currently reuse the same
`FrozenFourierEmbedding` spelling. Their own checkpoint evidence decides their
state ownership, so a shared Module does not acquire a family-mode branch, factory,
protocol, or config axis to hide the difference.

The local `~/weights/af3.bin.zst` is present and readable; weight availability is
not a blocker. It contains 405 records and 368,384,602 raw elements. Eight complete
floating records totalling 8,320 elements belong to pair features computed by
`_per_atom_conditioning` whose result both released callers discard. The separate
64-byte `uint8` identifier is live output and writer state. The corrected native
persistent total is 368,313,306 elements, of which 368,313,242 are learned floating
state.

The fourth released MSA block has a second bounded dead tail. Its outer-product
pair update runs first and remains live, but its later MSA attention and transition
produce an updated MSA that no subsequent operation consumes. Those 62,976
elements need not exist in the native graph. Conversely, the current PyTorch graph
duplicates four officially shared pair-input LayerNorms across atom and diffusion
blocks, leaving 3,040 excess persistent elements after excluding its 512
manufactured Fourier Parameters. A strict load into that target keyspace therefore
did not prove graph parity. AF3 Tok must rewrite the graph and converter together;
it cannot preserve the current state layout merely because conversion once loaded
strictly.

The released declarations `conditioning.prob`, `eval_batch_size`,
`eval_batch_dim_shard_size`, `no_embedding_prob`, and the Evoformer/diffusion
rematerialization fields have no executable reads in the official inference path.
They are subtraction evidence, not separate claimed scientific bugs, and do not
enter OFoldX config. The separately documented but unread `pad_num_chains` is a
public no-op bug and is recorded in `UPSTREAM_BUGS.md`.

AF3 ranking is a complete-output requirement. The released score is not only
`0.8 * ipTM + 0.2 * pTM - 100 * has_clash`; it also adds
`0.5 * fraction_disordered`. Upstream computes that fraction from AF-RSA windowed
relative solvent accessibility through its compiled mkdssp binding for every
sample. The binding is built against DSSP 4.4.7/libcifpp, but the consumed legacy
output is only one integer accessible-surface-area value per complete protein
residue. AF3 does not consume secondary structure, hydrogen bonds, or CCD radii
from that path. The current local Protenix path hardcodes disorder to zero and is
not AF3 parity evidence.

A complete AF3 vertical must preserve public windowed solvent-accessible-area and
fraction-disordered operations and use the exact full ranking formula before
global seed/sample selection. It may reuse a dependency only after equivalence is
proven; it may not omit the term, silently substitute zero, or relabel an
approximation. Every protein chain is computed independently, DSSP failure for a
protein propagates, no-protein naturally returns zero, and global best selection
rejects any NaN before retaining earliest ties.

The direct family-owned algorithm is now exact enough to freeze. Each protein
chain is evaluated independently from coordinates cast to float32 and formatted to
three decimal places. Hydrogen is ignored. A residue participates only when N,
CA, C, and O are all present. Accessibility uses the 401 golden-spiral directions
indexed from -200 through 200, a 1.4 Angstrom probe, intrinsic radii N=1.65,
CA=1.87, C=1.76, O=1.40, and 1.80 for every other non-hydrogen atom. Candidate
occluders satisfy `d**2 < (R + S)**2` and `d**2 > 1e-4`; a surface point is
occluded at distance squared `<= S**2`. Per-dot area is `4*pi/401`.

The resulting floating ASA is quantized through the legacy `%4.0f` spelling before
division by the fixed 20-amino-acid maxima and clipping at one. Unknown or modified
protein identity uses the ALA maximum. AF3 then applies its reflected window-25
smoothing and the strict `smoothed > 0.581` fraction. These facts are available
directly from canonical observed atoms. Serializing a Structure to CIF merely so
libcifpp can parse it again is accidental plumbing; OFoldX implements this narrow
AF-RSA operation directly and does not claim a general DSSP replacement.

The released identical-sequence cache is scientifically wrong because coordinates
are omitted from its key, as recorded in `UPSTREAM_BUGS.md`; OFoldX computes every
chain independently. Generic DSSP may lowercase a disulfide CYS, but released AF3
predicted outputs remove polymer-polymer bonds before this calculation and retain
uppercase CYS. No current AF3-output consumer justifies importing that wider DSSP
connection behavior.

The full CCD remains external chemical authority rather than code OFoldX can
recreate. OFoldX may own its parser, index, component lookup, and call-local
`user_ccd` overlay, but must obtain one exact snapshot with clear provenance and
redistribution terms. AF-RSA itself does not read `components.cif` or use a bond
graph. A clear vectorized NumPy or PyTorch reference implementation comes first; a
native or OOps kernel follows only from measured performance evidence. OOps remains
optional.

Depending on the complete official `alphafold3==3.0.3` package would force Python
>=3.12, JAX/CUDA12, Haiku, Tokamax, a fixed RDKit, Linux-oriented native builds,
five CMake `FetchContent` repositories, and a separate post-install `build_data`
step merely to obtain DSSP and CCD. A system `mkdssp` adds path/version/process
state and has no exact-output evidence. Both are larger system contracts than the
narrow family-owned alternatives.

AF3 remains reopened before complete family Tok only for exact CCD provenance,
redistribution terms, and offline asset inclusion. The direct AF-RSA implementation
is authorized by the recovered algorithm above; on-demand parity under
`scripts/parity` compares the official per-residue integer ASA, boundary cases,
window behavior, and complete ranking without becoming CI, committed validation,
or a publication gate. If direct equivalence is falsified, Tik returns to the exact
narrow DSSP/libcifpp binding; it does not add an approximate mode or fallback
chain. Any unsupported platform is removed honestly from the AF3 publication claim
rather than receiving a partial implementation.

## AlphaFold3 canonical and distribution boundaries

The official mmCIF input path uses deposited coordinates to derive topology,
bioassembly, modifications, and bonds, but does not condition prediction on those
coordinate observations. Ordinary canonical folding therefore accepts a
coordinate-free `MolecularSystem`; it must not accept a `Structure` and silently
discard its coordinates. The released mmCIF route remains a clearly named family
conversion whose topology-only meaning is visible at the call site. A caller with
a `Structure` can make the same choice explicitly through `.system`.

The official output gather has the inverse ambiguity: atoms absent from the Model
layout are inserted into the complete output layout at `(0, 0, 0)`, warned about,
and then assigned occupancy `1.0` with every other atom. Canonical decode must not
call fabricated zeros observed coordinates. However, those atoms currently enter
clash, AF-RSA, and therefore ranking. A truthful coordinate mask is consequently a
scientific correction, not a decode-only parity exception. The family Tik must
freeze which confidence metrics and global selection use the corrected mask and
record the intentional divergence before Tok.

Two upstream date names also cannot be merged. The data-search
`max_template_date` filters structural-template hits, while
`ref_max_modified_date` controls whether CCD model coordinates may be a fallback
after conformer/ideal-coordinate failure. The similarly named field on
`WholePdbPipeline.Config` is dead and is recorded in `UPSTREAM_BUGS.md`; OFoldX
keeps only the two live operation arguments and does not preserve the no-op field.
The released `ref_max_modified_date=None` default can nevertheless reach a
date-to-`None` comparison on that fallback path; whether `None` means no cutoff or
is forbidden remains a family contract decision and the upstream failure is
recorded separately.

AF-RSA failure is not a scientific zero. Upstream catches DSSP failures per protein
chain, omits those chains, and returns `0.0` when every chain fails. That turns a
missing or broken DSSP/components installation into a plausible ranking score.
OFoldX retains the exact successful computation but lets the dependency failure
fail; only a structure with no protein has a natural zero disorder fraction.

Finally, the Apache-2.0 source license does not license the released weights or
their outputs. The model-parameter terms prohibit publishing or sharing weights
outside the permitted organization, while outputs carry separate terms. OFoldX
may convert a user's authorized local weights into a local repository, but it does
not ship or upload them. The released writer adds ModelCIF metadata and a legal
comment to every sample and root CIF, in addition to a root terms file. Exact
seed/sample directories, filled input, confidence JSON, ranking CSV, optional
embeddings/distogram, compression, identifier, and truthful OFoldX software
attribution remain part of the reopened writer contract. None belongs in
`FoldResult`, provenance, a manifest fingerprint, or generic lifecycle machinery.

<!-- proposed 2026-07-16 via §3.3 freeze-drafting investigation; OWNER VERDICT 2026-07-16: REOPENED — freeze REJECTED; DESIGN §3.3 corrected back to reopened and the reopened candidate rewritten with the blockers below -->
## Owner review — §3.3 value-shape slice (+ §3.4.3 `iter_bonds()`, §3.5.2 atom-access)

**Verdict: REOPENED.** The §3.3 value-shape freeze is rejected. The committed `ofoldx/data/molecular.py` already implements frozen §3.4.1's dict `MolecularSystem`, so a migration landing spot for the stale io/processor calls is still needed — but the proposed frozen shape was wrong on the points below, and DESIGN §3.3's premature `Status: frozen` was corrected back to reopened (it does not authorize Tok).

Freeze-blockers (must be resolved before any §3.3 freeze):

- **`instances` must be an ordered entry array**, not an unordered `dict[str, str]` — otherwise order is lost on round-trip, because `io/structure.py:1503` serializes with `json.dumps(..., sort_keys=True)` (which reorders object keys).
- **Keep the 5-member `PolymerType` including `"pna"`.** The proposed 4-member alias makes `pna` unrepresentable before family lowering resolves it.
- **`from_one_letter` must support** protein `X → UNK`, RNA `N → N`, and DNA `N → DN`.
- **`from_smiles` emits one `UNL` `ChemicalComponent`** carrying its atoms plus `ChemicalBond` edges — not a separate `MoleculeBond` type resolved against a side table.
- **Import classification is declaration-driven:** `PolymerSequence.polymer_type` is a fact the producer declares, with a format-local component-ID fallback permitted *inside* the importer — never a reusable public chemistry-keyed classifier.
- **MSA/Template entity serialization is not owned here.** `Template.target`/`.monomers` are not fields on the committed shapes, so every "`encode_msa`/`encode_template` reuse verbatim" claim is void; that serialization moves to reopened §3.6.
- **`iter_bonds()` is not complete.** Its incident-bond-removal predicate (coupled to `iter_atoms()`) and cross-layer physical-pair dedup remain unresolved. (The `BondOrigin` domain is now resolved: `geometry_inferred` was deleted, so committed code and §3.4.3 both carry the 5 members `component/molecule/polymer_rule/source/user`.)

Preserved factual/migration findings: the stale callers are `io/structure.py` plus six processors invoking a pre-dict API (`system.entity()`, `system.components`, `EntityInstance`, `AtomSite(inst, atom_id, monomer_index)`, `entity.monomers`, `iter_bonds()`). The migration is not a rename: iterating a bare dict yields KEYS (silently always-False `isinstance` at `esmfold2:200`, `odesign:200,946`, `io:1663`, `pxdesign:537,589`); `AtomSite` is a transpose to `(instance_id, component_index, atom_id)`; `component_index` is a required non-`None` int (so `is None` polymer/nonpolymer tests are always-False); `system.component(ref)` takes `(instance_id, component_index)`; there is no system-level component table; `Molecule` has no `.atoms`/`.reference_coordinates`; `EntityInstance` is gone; `SystemBond.provenance → .origin`. Two consumers stay separately blocked on reopened facets: `odesign` cyclic closure (`processor.py:662-693`, reads removed `PolymerAttachment.order/.aromatic` and synthesizes cyclic `SystemBond`s) and `protenix` SMILES-origin routing (`processor.py:615`, committed `Molecule` has no `provenance`).

<!-- proposed 2026-07-16 via §3.5.1 freeze-drafting investigation; OWNER VERDICT 2026-07-16: ACCEPTED (carrier only) -->
## Owner review — §3.5.1 Structure value carrier

**Verdict: ACCEPTED (carrier only).** The six-field `Structure` (`system`, `sites`, `coordinates`, `coordinate_mask`, `occupancy`, `residue_numbers`) plus `ResidueNumber(number: int, insertion_code: str = "")` carrier is frozen in DESIGN §3.5.1 exactly as committed in `ofoldx/data/structure.py`. This covers ONLY the carrier: the writer policy (label/auth column selection, B-factor/altloc placeholders, one-based fallback), the exact codec JSON shape, the "author when present, else label" rule, and "nonpolymer extra entries are harmless" are removed from DESIGN and stay in the reopened §3.5 Structure I/O contract. This freeze does NOT authorize implementing the codec (`encode_structure`/`decode_structure`) or the writer.

Sequencing (factual): this carrier does not by itself make `_codec.py` or `io/structure.py` compile — both also target the pre-§3.4.1 `MolecularSystem` and must land together with the §3.3/§3.4.1 codec-system migration and the reopened §3.5 I/O rewrite; §3.5.1 does not unblock §3.3 (the dependency runs the other way).

<!-- proposed 2026-07-16 via §3.7.2 Protenix conditioning freeze-drafting; OWNER VERDICT 2026-07-16: REOPENED — positive API rejected, v0.5 constraint slice only -->
## Owner review — §3.7.2 Protenix conditioning

**Verdict: REOPENED — v0.5 constraint SLICE only (not Protenix end-to-end).** The proposed positive conditioning API (shared-looking min/max-band records, `ProtenixLayout`, a heterogeneous `conditions` bag) is rejected. The DESIGN §3.7.2 negative cross-family contract (no shared `Condition` record) stays frozen. Refined direction for the eventual v0.5 constraint slice:

- **Minimal interface:** `processor(input, *, atom_contacts=(), token_contacts=(), pocket=None, generator=None)` — named params, not a heterogeneous `conditions` union bag. `pocket` is a single optional value (one binder, one `max_distance`, N contact residues), not multiple records.
- **`ProtenixAtomContact` has ONLY `max_distance`** (no `min_distance`): upstream swapped the two atom-contact channels, so the checkpoint trained only channel 0 and it learned the max distance. Write `max → channel 0`, leave channel 1 zero, and reject a nonzero min (UPSTREAM_BUGS.md:470-473).
- **A token contact must NOT allow both endpoints to be `AtomSite`** (that pair is definitionally the atom channel), while it must ALLOW a single-atom (mixed atom/residue) endpoint.
- **Substructure is ALWAYS an all-zero `[T,T,4]`**, yet the full substructure embedder still runs (so the pair baseline is nonzero regardless). DELETE the extension that builds a nonzero token-centre distogram from input coords (`processor.py:921-951`).
- **The v0.5 Processor must NOT silently ignore `Structure` coords:** topology input explicitly passes `structure.system`.
- **Token/pocket lowering uses a PyTorch RNG (NOT unseeded).** Generator ownership and exact draw order must be frozen, and the Processor + Model must consume ONE advancing `generator`. Multi-token-component / pocket-residue token selection stays open processor policy, not a record field.
- **DELETE `ProtenixLayout`, `ProtenixConstraintFeatures`, and the computed-then-discarded `_protenix_condition_measurements`.**
- **The default (non-constraint) checkpoint must REJECT non-empty constraint params.**
- **Release evidence:** `a0f95f6` + `protenix_base_constraint_v0.5.0.pt` (with `c3bfc365` supplementary). TFG and `protenix-v2` stay reopened / non-target.

Recorded code bugs (preserved; fix separately, NOT now):

1. **The Protenix processor is NON-IMPORTABLE.** `processor.py:37` imports `ProtenixConstraintFeatures`, which `contract.py` does not define (ImportError); `Contact`/`Distance`/`ReferenceCoordinates` are used in `isinstance` at `processor.py:329,338,346,809` but never imported (NameError).
2. The `atom_layout`/decode regressions are being fixed separately.

<!-- proposed 2026-07-16 via §3.7.3 Contig investigation; STILL_PARTIAL — direction only, awaiting owner -->
<!-- SUPERSEDED 2026-07-18: the placement/contig contract was ratified + FROZEN (placement axis DONE; exhaustion = full partition; realized-id unified to chain_{n}). See "§3.7 caveats — NOW RESOLVED" and "OWNER VERDICTS (2026-07-18)" below. The per-family chain_lengths re-home finding recorded below remains valid. -->
## Owner review — §3.7.3 Contig / placement

**Verdict: REOPENED (STILL_PARTIAL).** The synthesized candidate resolves its core (shared placement records, pre-realization symbolic references, a `placement_map` output surface), but the Contig core is NOT frozen — it is gated on two blockers:

- **X1 — BoltzGen has no in-tree realization.** BoltzGen is in scope but ships only `config/convert/model/README` (no Processor/contract/schema/conditions); its `model.py` consumes pre-tensorized `binding_type`/`visibility_ids` with no Contig. So the coverage/exhaustion (partition) invariant cannot be witnessed against a whole-assembly `total_len` family, and BoltzGen's variable-tail `SymbolicResidueSite` rejection rule cannot be pinned. Either scope BoltzGen out of the Contig freeze, or build its realization path first.
- **X2 — outer Contig key = handle vs. authoritative id.** No in-tree family honors a caller outer key as the realized instance id (RFdiffusion source-derives it, ODesign synthesizes `odesign_chain_N`, PXDesign carries the source id / `segment_id`). Resolving toward "the outer key is a caller-side declaration + `placement_map` lookup handle, not the realized instance id" is load-bearing and interacts with the already-frozen `inpaint_sequence` (which addresses existing residues by source `ComponentReference`); it needs owner ratification, not a unilateral freeze.

Preserved factual findings: the realization paths differ — ODesign copies-and-renumbers (fresh `odesign_chain_N` + indices, `processor.py:446,470-473`), while RFdiffusion and PXDesign carry existing spans verbatim as `(source_instance_id, source_index)` (`model.py:6352`; `processor.py:296-303`). The in-tree `Contig.chain_lengths` read by both `odesign/processor.py:716` and `rfdiffusion/model.py:6270` is a false shared invariant and must re-home per family. Generated-topology guarantees that DO hold: declared outer/inner order is preserved; every declared placement appears exactly once (zero-length → `()`); every `placement_map` `ComponentReference` names a realized-output occurrence. The per-declared-instance exhaustion/partition invariant — recorded here as withdrawn while this audit was open — was later **ratified as exhaustive (a full partition)** by the 2026-07-18 owner verdict (see "§3.7 caveats — NOW RESOLVED" and "OWNER VERDICTS (2026-07-18)" below); this audit line is superseded on that point. `SymbolicBond` stays a separate open item, not a Contig-core gate.

<!-- proposed 2026-07-16 via §5.3.4 coordinate-design freeze-drafting; OWNER VERDICT 2026-07-16: ACCEPTED (simpler shared carrier) -->
## §5.3.4 Coordinate-bearing design — owner review

**Verdict: ACCEPTED (simpler shared carrier).** The three coordinate-generating families (RFdiffusion, ODesign, PXDesign) currently shove realized `Structure` objects into `DesignResult.candidates` behind `# type: ignore[arg-type]` (`rfdiffusion/model.py:6518`, `pxdesign/model.py:168`) or an unsuppressed mismatch (`odesign/model.py:243`). The accepted fix widens the shared `DesignCandidate` with both an optional coordinate realization and a metrics map:

```python
@dataclass(eq=False)
class DesignCandidate:
    system: MolecularSystem
    structure: Structure | None = None            # coordinate realization; None => coordinate-free
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
```

- Producer rule: `structure.system is system` (a coordinate-generating family sets `system = structure.system`). NO defensive `__post_init__`.
- §5.3.3's "do not widen `DesignCandidate`" sentence is amended to drop only `structure`/coordinate-realization and `metrics`, keeping the pair-metrics/union/optional-input/placement/context exclusions.
- MPNN families are unaffected — `structure`/`metrics` default empty, so existing coordinate-free construction is unchanged.

**REJECT `RFdiffusionDesignCandidate`.** The proposed family subtype is rejected: adding a non-defaulted field after the base's defaulted fields would raise at class-creation time; `self_consistency` misnames upstream's per-residue predicted-lDDT ("Predicted lDDT of x0"); and the pLDDT scale (0-1 vs. scaling) and Metric axis are not frozen. RFdiffusion's predicted-lDDT (decoded at `model.py:4172`, currently unpacked-then-dropped at `model.py:6499`) now has a home in the shared `metrics` map, but its exact Metric axis/unit — and whether RFdiffusion also publishes a derived scalar Score — stay a reopened RFdiffusion family decision.

<!-- proposed 2026-07-16 via §5.4.3 affinity investigation; OWNER VERDICT 2026-07-16: ACCEPTED (narrow fact only) -->
## §5.4.3 Boltz-2 affinity — owner review

**Verdict: ACCEPTED (narrow fact only).** Narrow accepted fact: `Boltz2AffinityModel.score(...) -> ScoreResult` — the released affinity output is scalar Scores over one selected coordinate-conditioned target and fits the already-frozen shared `ScoreResult(target, scores, metrics)` with no affinity-specific Result type. Committed `model.py:716-750` already conforms.

Full composition is REOPENED because:

- **First-stage coordinates ARE consumed** by the second-stage spatial crop — they are not "unconsumed", so the sweeping "drop `structure_predictions`" claim is withdrawn.
- **The concrete Pipeline must return BOTH** the first-stage `FoldResult` and the second-stage `ScoreResult`, and the join shape (tuple, two returns, or a thin Pipeline result) is undecided.
- First-stage predictions keep generation order.
- Score names, probability unit, and ligand scope are not unified across families.
- The private `_score` path is a run-intermediate audit hook, not a normative guarantee.

## Owner review — §3.7.2/§3.7.3 ODesign conditioning

**Verdict: SPLIT — freeze two family-owned §3.7.2 spatial records now; the placement/contig and covalent-link records stay BLOCKED.** ODesign is a coordinate-GENERATING design family, so its conditioning straddles both facets: the §3.7.2 spatial signals (fixed coordinates, hotspots) are family-owned over EXISTING input atoms with zero placement coupling and are freezable today; everything that names or materializes GENERATED topology is gated on the reopened §3.7.3 Contig (STILL_PARTIAL, X1 BoltzGen-scope + X2 outer-key), and the cyclic covalent link is separately gated on §3.3 + `SymbolicBond`. The §3.7.2 negative contract stands: ODesign owns its records, there is no shared `Condition`, no heterogeneous `conditions` union bag.

**Non-importable fact (fix, but the shape below is the target).** All three ODesign files import removed records from `ofoldx.data.conditions`, which now holds only the two §3.7.1 records. Reproduced: `import ofoldx.models.odesign.model` → `ImportError: cannot import name 'Contact' from 'ofoldx.data.conditions'` (contract.py:12-18, model.py:17-25, processor.py:23-32). Unbreaking must re-home each record per its disposition below, NOT restore a shared module.

### A. Freezable NOW — §3.7.2 family-owned spatial (no Contig coupling)

**A1. `ODesignReferenceCoordinates` (from `ReferenceCoordinates`) — fixed/constraint coordinates.** Endpoints are EXISTING input `AtomSite`s only: the processor rejects any reference site whose coordinates are not observed in the input (`processor.py:298-299`), so this record has zero symbolic/generated coupling. It drives two learned/sampler surfaces that are pure §3.7.2: `fixed_coordinate_mask` (hard intent → EDM `condition_mask`, `model.py:286`) and `constraint_coordinate_mask` → the pairwise `constraint_distogram` learned channel (`processor.py:328-334`, `ConstraintDistogramEmbedder` in `model.py:454-497`).
- Concrete record (family-owned, e.g. `odesign/conditions.py`): `sites: tuple[AtomSite, ...]`, `intent: Literal["hard", "soft"]`, `tolerance_angstrom: float`, `weight: float | None`. The validation is family-owned, not a shared protocol: `"preference"` is rejected and `weight` must be `None`/`1.0` (`processor.py:134-137`); `"hard"` also sets `fixed_coordinate_mask` (`processor.py:296-297`) and is verified post-generation (§B).
- Migration: processor named param `reference_coordinates: Sequence[ODesignReferenceCoordinates] = ()`.

**A2. `ODesignHotspot` (the LEARNED half of `Contact`) — soft per-residue hotspot flag.** The only tensor the model consumes is `token_features[..., 452] = hotspot` (`processor.py:320`), a per-residue binary label over EXISTING input residues/atoms — upstream `is_hotspot_residue`. `_hotspot_tokens` (`processor.py:1108-1136`) SKIPS every symbolic/generated reference (`processor.py:1122`) and requires ≥1 input hotspot (`processor.py:1135`). Per REVIEW.md:196 and DESIGN §3.7.3:1402, an ODesign hotspot is a residue label, NOT a synthetic 8 Å contact; the two-sided `side_a`/`side_b` + `cutoff_angstrom` framing is a shared-`Contact` artifact and must NOT be preserved as the learned record.
- Concrete record: a hotspot set naming existing input residues/atoms — `residues: tuple[AtomSite | tuple[str, int], ...]` (matching the `AtomSite` / `(instance_id, component_index)` endpoints the lowering accepts at `processor.py:162,169`). Consumed strictly as a soft signal (`processor.py:139-142` forces `intent == "soft"`, `weight` `None`/`1.0`); the protein-only-unless-exact-atom rule stays family-owned (`processor.py:158-172`).
- Migration: processor named param `hotspots: Sequence[ODesignHotspot] = ()`.

### B. REOPENED — §3.7↔§5 hard-condition satisfaction (not a learned channel, defer)

`_require_hard_conditions` (`model.py:518-574`) is a POST-generation verification pass (Kabsch RMSD for `ReferenceCoordinates`, min contact distance ≤ `cutoff_angstrom + tolerance_angstrom` for `Contact`), not a model input. The per-condition measurement `value` (angstrom RMSD / min-distance) has NO frozen carrier (`model.py:531-534` TODO) — the same §3.7↔§5 measurement/satisfaction surface flagged for Protenix. Freeze only the family-owned raise-on-violation gate with the A1/A2 records; the measurement carrier stays reopened. NOTE the coupling: the Contact half of this pass (`_contact_sites`, `model.py:598-622`) DOES resolve `GeneratedSegment`/`SymbolicResidueSite`/`SymbolicAtomSite` endpoints, so hard-contact verification over GENERATED endpoints is Contig-coupled (§C) even though the learned hotspot (A2) is not.

### C. BLOCKED on §3.7.3 Contig — X1 BoltzGen + X2 outer-key (STILL_PARTIAL)

`Contig`, `GeneratedSegment`, `ExistingSpan`, `SymbolicResidueSite`, `SymbolicAtomSite` drive generation-topology realization (`_materialize_contig` `processor.py:407-605`; `_realize_ranges` `processor.py:703-943`; the layout identity maps `contract.py:29-32`). These CANNOT re-home to a frozen public shape — three concrete ODesign facts land squarely on the two open §3.7.3 blockers, and the current record shapes are the ones DESIGN §3.7.3 explicitly withholds:
- **X2 (outer-key = handle vs. id).** ODesign synthesizes fresh `odesign_chain_N` / `odesign_entity_N` ids and renumbers (`processor.py:446,470-473,495`) instead of honoring a caller outer key — the exact case REVIEW.md:1172 leaves unratified.
- **`chain_lengths` re-home.** `Contig.chain_lengths` (`processor.py:716`) is the false shared invariant; ODesign owns per-chain exact lengths (DESIGN §3.7.3:1364, REVIEW.md:1174). Must become a family-owned field, not a shared one.
- **Record shape contradicts the DESIGN candidate core.** Current `SymbolicResidueSite(segment_id, offset)` is 2-field (`processor.py:511-512,780-781`), but DESIGN §3.7.3:1389 requires 3-field `(instance_id, placement_id, offset)`; current `GeneratedSegment` carries an embedded `.segment_id` field (`processor.py:156,718,774`), but DESIGN §3.7.3:1355 gives placements identity via the mapping KEY, not a value field ("the same ordinary value may occur more than once without acquiring a hidden ID"); current `Contig.chains` is `tuple[tuple[part, ...], ...]` where DESIGN's candidate is a nested ordered dict. Re-homing these as-is would freeze precisely the shapes X2 is meant to decide. Keep them family-local and non-public (private placeholders, explicitly TODO'd) until X1/X2 resolve. `ExistingSpan(instance_id, start, stop)` alone matches DESIGN §3.7.3:1352, but it has no standalone use outside the blocked `Contig`.

### D. SEPARATELY BLOCKED — `DesiredCovalentLink` (§3.3 cyclic closure + §3.7.3 `SymbolicBond`)

Cyclic head-to-tail closure (`_materialize_cyclic_links` `processor.py:608-700`; `model.py:568`). Two independent gates, neither of them Contig:
- The closure SystemBond synthesis is blocked on the reopened §3.3 cyclic-closure `SystemBond` contract: committed `PolymerAttachment` no longer carries `order`/`aromatic`, so the synthesized bond falls back to polymer-rule defaults and the attachment-chemistry compatibility checks are dropped (`processor.py:640-644` TODO; REVIEW.md:1137 preserved finding).
- The request record with a GENERATED endpoint (`SymbolicAtomSite`) is blocked on the §3.7.3 `SymbolicBond` contract (DESIGN §3.7.3:1407-1411: no accepted endpoint/chemistry/ownership). A link between EXISTING `AtomSite`s is already `MolecularSystem` topology (DESIGN §3.7.3:1409), so no condition record is needed there — `DesiredCovalentLink` cannot be frozen as a public record in either form.

### Removed-record site inventory (migration per site)

| Record | ODesign sites | Disposition |
| --- | --- | --- |
| `ReferenceCoordinates` | contract.py:16,29; processor.py:29,124,133,288,299,610; model.py:22,535,571 | **Freeze now** → `ODesignReferenceCoordinates` (§A1), named param `reference_coordinates=` |
| `Contact` (learned) | processor.py:24,124,138,159,172,1118,1135 | **Freeze now** → `ODesignHotspot` (§A2), named param `hotspots=`; drop `side_a/side_b/cutoff` learned framing |
| `Contact` (hard-distance check) | model.py:18,551,598-622 | **Reopened §3.7↔§5** (§B); symbolic endpoints Contig-coupled |
| `Contig` | contract.py:14,29; processor.py:25,124,129,131,143,408,417,610,704; model.py:19,568 | **BLOCKED §3.7.3** (§C); shape unresolved, `chain_lengths` re-home |
| `GeneratedSegment` | processor.py:28,156,164,455,464,711,718,735,773,778,1122,1181; model.py:21,616 | **BLOCKED §3.7.3** (§C); drop embedded `segment_id`, identity moves to key |
| `ExistingSpan` | processor.py:27,464,482 | **BLOCKED §3.7.3** (§C); no standalone use outside `Contig` |
| `SymbolicResidueSite` | contract.py:17,30; processor.py:31,166,413,437,511-512,611,780-781,1187; model.py:24,578 | **BLOCKED §3.7.3** (§C); 2-field → 3-field is the X2 decision |
| `SymbolicAtomSite` | processor.py:30,162,1122,1178; model.py:23,607 | **BLOCKED §3.7.3** (§C) + `SymbolicBond` (§D) |
| `DesiredCovalentLink` | contract.py:15,29; processor.py:26,124,143,610,646; model.py:20,568 | **BLOCKED §3.3 + §3.7.3** (§D); not a public record |

### Layout / signature migration (Protenix corrected direction applied)

- **Delete the heterogeneous union bag.** `ODesignLayout.conditions: tuple[Contact | Contig | DesiredCovalentLink | ReferenceCoordinates, ...]` (`contract.py:29`) and the processor's `conditions: Sequence[...] = ()` (`processor.py:124`) are exactly the rejected shape from the Protenix review — replace with named params (`reference_coordinates=`, `hotspots=`, and blocked `contig=`/`covalent_links=` placeholders). The layout already carries the realized identity maps (`generated_identity_map`, `source_residue_map`, `source_atom_map`); the `conditions` tuple survives only to feed the §B satisfaction pass and must move onto the family-owned records / §3.7↔§5 carrier, not a union field.
- **No shared `Condition`, no `intent`/`weight`/`tolerance` protocol.** `intent`/`weight`/`tolerance_angstrom` become validated fields of the family-owned §A records, checked inline (as they already are at `processor.py:132-144`, `model.py:573`), never a universal base.
- **Records re-home into the ODesign package, not `data.conditions`.** The freezable §A records become public family records; the §C/§D records stay family-local and non-public until §3.7.3 (X1/X2) and §3.3/`SymbolicBond` resolve.

<!-- proposed 2026-07-17 via CCD asset-distribution investigation (AF3 full-ligand + Protenix/Boltz ligand blocker); OWNER VERDICT 2026-07-17: ACCEPTED -->
## Owner review — CCD asset distribution (§3.2 / §3.5 importer resolution)

**Verdict: ACCEPTED.** The wwPDB Chemical Component Dictionary is distributed as a Hub `dataset` repo `ofoldx/ccd` — Parquet, one component per row, revision = the wwPDB snapshot date — loaded via the existing `ofoldx/hub.py` `snapshot_download` and lowered to `ChemicalComponent` VALUES by an IMPORT-TIME resolver `ofoldx/data/ccd/resolver.py::resolve_ccd_component(component_id, *, revision)`. This reuses MultiMolecule's dataset DISTRIBUTION pattern (Hub-hosted Parquet + snapshot download), NOT its runtime `Dataset` class — no runtime dataset object enters tensor compute.

`get_component` STAYS audited-only + zero-I/O (Principle 7): the resolver runs at import time, never at tensor-compute time, so there is no live registry lookup on the compute path. The deleted prototype `ccd.py` `CcdProvider` (a live registry lookup at tensor-compute time) violated Principle 7 and MUST NOT return.

Rationale/impact: unblocks the AF3 full-ligand path and the Protenix/Boltz ligand paths, which need CCD component atoms/bonds resolved before lowering. Flows into DESIGN §3.2 / §3.5 (importer resolution) when those sections freeze.

<!-- proposed 2026-07-17 via AlphaFold2 family-boundary investigation; OWNER VERDICT 2026-07-17: ACCEPTED — one family, two arch ids -->
## Owner review — AlphaFold2 / AF-Multimer family boundary (§1.1 / §7.1)

**Verdict: ACCEPTED — ONE `alphafold2` family, TWO registered arch ids.** Package `ofoldx.models.alphafold2` holds the shared Evoformer / ExtraMSA / IPA trunk and registers two arch ids / two main entry points: `ofoldx.alphafold2` (monomer + pTM) and `ofoldx.alphafold2_multimer` (paired MSA, asym/entity/sym features, ipTM, chain permutation, `model_*_multimer_v3`). Per §1.1 "no partial family" they ship TOGETHER — no monomer-first release.

Rationale: multimer is a genuinely different concrete INPUT/OUTPUT type — it REQUIRES chain-identity features + paired-MSA semantics and emits ipTM — so it is a second arch id, not a config flag on the monomer processor. It is named to stay unambiguously AF2 (both ids live under the one family). This resolves under the §7.1 registration contract (a pair is not inferred from tensor shape/heads/tasks/keys but from the concrete input/output type) and the §1.1 complete-family gate.

<!-- proposed 2026-07-17 via entry-point/arch-split criterion elevation (generalized from AF2 vs multimer); OWNER VERDICT 2026-07-17: ACCEPTED — reusable criterion -->
## Owner review — General entry-point / arch-split rule (§7.1)

**Verdict: ACCEPTED as a reusable owner criterion.** Split a family into SEPARATE registered (arch, entry-point) pairs ONLY when the Processor call SIGNATURE / required input-output TYPE differs. A checkpoint-selected graph variant with the SAME Processor signature stays ONE arch with a config field.

- SPLIT precedents: AF2 monomer vs multimer (required chain-identity features + paired-MSA + ipTM); MembraneMPNN global `transmembrane: bool` vs per-residue `membrane_labels: Mapping` (different Processor signature — two pairs that share one Model).
- STAY-ONE precedent: ProteinMPNN `ca_only`, vanilla/soluble (same Processor signature, checkpoint-selected graph → config field).

This is consistent with §7.1 ("registration does not infer a pair from tensor shape, a head checklist, tasks, variants, or keys"): the discriminator is the caller-visible Processor signature, not internal tensor structure. Flows into §7.1 as the standing split criterion.

<!-- proposed 2026-07-17 via ProteinMPNN ca_only config-vs-axis review; OWNER VERDICT 2026-07-17: ACCEPTED — ca_only stays, no §4.7 change -->
## Owner review — ProteinMPNN `ca_only` stays config-only (§4.7)

**Verdict: ACCEPTED — `ca_only` stays; NO change to §4.7.** `ca_only` is config-only: it is never a runtime call arg, it is auto-derived at conversion from the checkpoint's edge-embedding input width (416 CA-only vs 167 full-backbone), and it is read from `model.json` exactly like a transformers `is_decoder` config boolean.

Rationale: the two alternatives are both worse and already rejected. Splitting the arch is rejected because both graphs share ONE Processor signature (the split rule above), and a magic edge-width int at compute time is a "false axis" already rejected under §4.7. Keeping `ca_only` as a converted config field is the minimal correct encoding. No §4.7 change.

<!-- proposed 2026-07-17 via ESMFold2 trunk/child weight-provenance investigation; OWNER VERDICT 2026-07-17: ACCEPTED — narrow fact -->
## Owner review — ESMFold2 ESMC child + trunk weight provenance

**Verdict: ACCEPTED (narrow fact).** ESMFold2's ESMC child is `multimolecule/esmc-6b` (Hub). The ESMFold2 / ESMFold2-Fast trunk weights are NOT available locally → they are node4 / HF-gated, and parity for the ESMFold2 trunk must run against the gated weights there.

<!-- proposed 2026-07-17 via MPNN biological-op batch-semantics reconciliation (DESIGN-frozen vs ROADMAP-reopened gap); OWNER VERDICT 2026-07-17: ACCEPTED — ratifies _require_single_target -->
## Owner review — MPNN biological-op batch semantics (§4.7 / §4.10)

**Verdict: ACCEPTED — keep B=1 for one-target biological ops.** The one-target biological ops (`sample` / `score` / `design` / `probabilities` / `specificity_profile`) in ProteinMPNN, LigandMPNN, and NA-MPNN keep B=1. Raw `forward` / `encode` stay B>=1 (training / collate). Multi-target bsz>=1 is deliberately NOT added — revisit ONLY on a concrete, measured serving / screening need.

This ratifies the currently-implemented `_require_single_target` guard and closes the DESIGN-frozen-vs-ROADMAP-reopened gap on §4.7 / §4.10 batch semantics: the biological ops are single-target by contract, so the guard is correct rather than a limitation to lift.

<!-- 2026-07-18 PARITY-PUSH-2 — parity results log (not an owner decision); details/tables in SESSION_NOTES.md §2 -->
## 2026-07-18 PARITY-PUSH-2 — parity findings

Second parity push. Numbers are GPU (A100) with relative tolerance where applicable.

- **af2-MONOMER — VALIDATED (deep-forward).** The full trunk deep-forward matches OpenFold to **1.4e-4 Å** on positions along a deterministic fixed-feature path. This closes the earlier "FAIL vs OpenFold" state, which was the unimplemented stochastic MSA/ensemble path (not a defect in the deterministic core).
- **af2-MULTIMER — [SUPERSEDED — see the two sections below].** This dated bullet is stale. af2-multimer strict weight-load is now **CLEAN** (0/0/0, 93.2M params) and the IPA/QuatRigid structure module is **bit-exact** (max_abs 0.0) vs OpenFold v2.2.0; full deep-forward golden is in progress. The "3 buckets not implemented" claim no longer holds — all three are implemented.
- **AF3 native trunk — [SUPERSEDED — see the two sections below].** This dated bullet is stale. The pair-pathway defect (pair cos 0.011, absmax ~73,791) was **CONFIRMED REAL and FIXED** (4 converter/orientation bugs). It is no longer "suspected".
- **esmfold2 — determinism SOLVED; global PASS; per-atom at floor.** Non-determinism is solved (**det==0**). Global scores **PASS** (~**2e-4**). Per-atom coords/pae sit at an **irreducible cross-impl floor** (**0.215 Å** coords / **0.108** pae) and will not reach 1e-4 — a different-implementation floor, not a bug.
- **Checkpoint sweep — mostly PASS.** PASS for **boltz1, boltz2, odesign (×2), boltzgen (3 of 4)**; **FAIL boltzgen `structuretrained_small`** (a *different architecture*, expected); **pxdesign** covered sampler-only.
- **Contract drafts.** Written to `CONTRACT_DRAFTS.md` with **3 owner decisions pending**.


<!-- 2026-07-18 CONSOLIDATION — supersedes PARITY-PUSH-2 af2/af3 status; parity findings log + owner verdicts -->
## 2026-07-18 — af2/af3 status update + owner verdicts (supersedes PARITY-PUSH-2)

This section reflects the TRUE current state and supersedes the af2/af3 lines in the 2026-07-18 PARITY-PUSH-2
findings above.

### AF3 native trunk — real pair-pathway BUG FOUND + FIXED (not "suspected")

The earlier "suspected real pair-pathway defect" (pair cos 0.011, absmax ~73,791) was CONFIRMED as a real bug and
FIXED. Root cause was **four** distinct defects on the pair pathway:

1. **Triangle-multiplicative even/odd factor-split interleave** — the two projected factors were split as a
   *contiguous chunk* instead of the correct *even/odd interleave*. This was the source of the magnitude blow-up.
2. **Incoming-swap** on the triangle-multiplicative incoming pathway.
3. **`target_feat` channel rotation-by-63** — a channel roll that scrambled the MSA projection.
4. **Ending-node attention-bias transpose** — wrong transpose orientation on the ending-node attention bias.

Post-fix parity: pair cos **0.011 → 0.9985**, distogram cos **0.10 → 0.9993**, single **→ 0.99998**,
forward-mode regression **max_abs 0.0**. Residual pair relL2 **0.054** sits just over the **0.05** gate; every op
is cos=1.0 in isolation, so this residual is **deep-stack cross-framework floating-point accumulation** — the
implementation is structurally faithful, it is simply not gate-green on the aggregate relL2 metric.

Files: `ofoldx/models/alphafold3/converters/deepmind.py`, `ofoldx/models/alphafold3/model.py`,
`ofoldx/modules/attention/triangle.py`.

This trunk fix is **CONFIRMED-STABLE and will not regress**. Downstream **denoiser / confidence /
trunk-at-real-tolerance** numeric validation is **IN PROGRESS** (was previously covered only by `isfinite` / 5e-1).

### AF2 — monomer + stochastic fold validated; multimer strict-load CLEAN (all CONFIRMED-STABLE)

- **af2-monomer** — deep-forward VALIDATED (**1.42e-4 Å** on positions vs OpenFold, deterministic fixed-feature).
- **af2-stochastic fold** — IMPLEMENTED + VALIDATED (Draft 1): Stage-1 feature parity **bitwise 0.0** vs OpenFold
  per-stage, Stage-2 trunk **4.16e-5 Å**, adversarial RNG-aligned.
- **af2-multimer** — strict weight-load now **CLEAN**: 0 missing / 0 unexpected / 0 shape mismatches, **93.2M**
  params. All three architectural buckets are implemented: **Bucket1** tri-mult fused→unfused converter-split
  verified; **Bucket2** multimer IPA / QuatRigid **max_abs 0.0** vs OpenFold v2.2.0; **Bucket3**
  TemplateEmbeddingMultimer implemented and loads. Deep-forward is all-finite; monomer shows no regression.
  **REMAINING**: the full end-to-end multimer golden — needs an OpenFold ≥ 2.0 harness plus the processor's real
  Gram-Schmidt `template_backbone_frame` (currently a placeholder identity).
  Files: `ofoldx/models/alphafold2/multimer_model.py`, `ofoldx/models/alphafold2/converters/deepmind_multimer.py`.

### OWNER VERDICTS (2026-07-18)

1. **AF2 fold → stochastic: YES.** Proceed to the stochastic monomer fold path (separate workflow, in progress).
2. **Contig X1 → build BoltzGen first, then freeze all 4.** Build the BoltzGen placement realization first, then
   freeze the Contig/placement contract across all four design families.
3. **Contig X2 → placement-keyed, global-unique `placement_id`, 2-field.**
   `SymbolicResidueSite(placement_id, offset)` — the "one correct way". (Now FROZEN in DESIGN.md §3.7.)

### §3.7 caveats — NOW RESOLVED (placement axis DONE, 2026-07-18)

The three earlier open caveats are closed; the placement axis is DONE (contig→placement rename to `PlacementSpec`,
`placement_map` adopted 4/4, exhaustive coverage, 6 projections, realized-id unified to `chain_{n}`; rfdiffusion
coords byte-identical). See "2026-07-18 OWNER PRINCIPLES + VERDICTS" below.

a. **rfdiffusion — RESOLVED.** `placement_map` is now adopted **4/4** (all four design families use the frozen
   placement_map surface).
b. **Realized-id minting — RESOLVED.** Unified to a single `chain_{n}` spelling across families.
c. **Coverage-vs-exhaustion — RESOLVED.** Exhaustive realization (the full partition) is the ratified rule.

<!-- 2026-07-18 OWNER PRINCIPLES + VERDICTS — records the binding owner principles and the confirmed-stable verdicts; principle detail also lives in CLEANUP_TRACKER.md -->
## 2026-07-18 OWNER PRINCIPLES + VERDICTS

### Owner principles (binding)

- **P1 — Model GRAPH = pure function of `(tensors, generator)`.** The graph takes a `generator` object only — never
  an int seed, never touches global RNG. `seed→Generator` resolution and the stochastic driver (fold/design
  seed-resolve + decode) live in the pipeline/runner, not the model. `seed`, `dtype`, and `device` are
  pipeline/runner concerns.
- **P2 — dtype-transparent model math.** Any forced fp32/fp64 inside sensitive math must be EXPLICIT and
  contract-documented (measured, not a silent `.float()`); decode-boundary `.float().cpu().numpy()` is fine.
  Precision/autocast is runner-level policy.
- **P3 — One correct way (Zen).** One canonical spelling per concept; no borrowed-lineage jargon promoted to a shared
  abstraction.
- **P4 — Math-equivalence is the unification bar.** Do not mirror upstream implementation structure; collapse only
  when the math/results are identical.
- **P5 — Overseer coordinates shared-code work.** Set the shared decision first, then parallel-apply; steer running
  agents; reconcile at integration points. Do not silo files to dodge conflicts.
- **P6 — Verify-first.** Rule out confounds (recycle / precision / RNG) before claiming a bug; parity on GPU with
  relative tolerance. (Caught: protenix false-alarm retracted; af3 real bug found + fixed; boltzgen `noise_scale`
  faithful.)

### Confirmed-stable results these verdicts rest on (will not regress)

| result | status | numbers |
| --- | --- | --- |
| af3 pair-pathway bug | CONFIRMED REAL + FIXED (stable) | single cos **0.99998**, pair cos **0.9985** (relL2 0.054, deep-stack fp accumulation), distogram cos **0.9993**; downstream denoiser/confidence/trunk-at-real-tol validation IN PROGRESS |
| af2 monomer | deep-forward VALIDATED | **1.42e-4 Å** vs OpenFold (deterministic fixed-feature) |
| af2 stochastic fold | IMPLEMENTED + VALIDATED (Draft 1) | Stage-1 feature parity **bitwise 0.0** vs OpenFold per-stage, Stage-2 trunk **4.16e-5 Å**, adversarial RNG-aligned |
| af2 multimer | strict-load CLEAN | **0/0/0**, 93.2M params; IPA/QuatRigid bit-exact **max_abs 0.0** vs OpenFold v2.2.0; full deep-forward golden IN PROGRESS |
| placement axis | DONE | contig→placement (`PlacementSpec`), `placement_map` 4/4, exhaustive coverage, 6 projections, realized-id `chain_{n}` (rfdiffusion coords byte-identical) |
| component-cleanup-1 | DONE (parity-gated, no regression) | `_best()`/`expected_plddt()`/`_component_from_raw`; `asym_id`→`chain_index`; `hhblits_profile`→`msa_profile`; `Boltz2AffinityResult`→`AffinityResult` |
| esmfold2 | determinism SOLVED; global parity | det==0, global ~**2e-4**; per-atom coords/pae at an irreducible cross-impl floor (**0.215 Å**) — characterized, not a bug |
| checkpoint sweep | mostly PASS | boltz1 / boltz2 / odesign (×2) / boltzgen (3 of 4) PASS; boltzgen `structuretrained_small` different-arch FAIL; pxdesign sampler-only |

### Owner verdicts

1. **AF2 fold → stochastic: YES.** The stochastic monomer fold path is the ratified direction (now implemented +
   validated, Draft 1).
2. **Contig X1 — BoltzGen-first.** Build the BoltzGen placement realization first, then freeze the placement contract
   across all four design families.
3. **Contig X2 — placement-keyed identity.** Global-unique `placement_id`; 2-field `SymbolicResidueSite(placement_id,
   offset)`; realized instance id unified to `chain_{n}`. This is the "one correct way"; the placement axis is DONE
   and this supersedes the earlier "Open §3.7 caveats".
4. **MultiMolecule (MM) is a data-point to critically evaluate, not a template.** MM validates the
   orchestration-boundary principle (every MM model is non-stochastic), but OFoldX sits ABOVE it: per-family
   `contract.py` typed tensor boundary (MM has none) and self-owned `FoldResult`/contract types (MM borrows HF
   `ModelOutput`/`AutoModelFor*` — do not import HF vocabulary). Adopt MM's discipline; EXCEED its global-seed
   mechanism via generator threading.

<!-- 2026-07-19 MIGRATION/CLEANUP ROUND — COMPLETE — final consolidation of the coverage/validation bug-hunt + cross-cutting cleanup wave + contract freezes; authoritative status lives in CLEANUP_TRACKER.md, per-family env/results in SESSION_NOTES.md -->
## 2026-07-19 MIGRATION/CLEANUP ROUND — COMPLETE

This round closed the coverage/validation bug-hunt, the cross-cutting cleanup wave (RNG / dtype / naming /
unification / OOps), and the two remaining contract freezes. Everything below is **CONFIRMED-STABLE** (validated this
session; will not regress) except the final "Remaining" table. Numbers are GPU (A100) with relative tolerance where
applicable.

### 1. Real OFoldX bugs found + fixed by the coverage/validation hunt (~12)

The coverage thesis — "unvalidated areas hide real bugs" — was vindicated: **14 distinct root causes across 6 areas**,
all fixed and parity-gated, plus one shared-module regression the cleanup itself introduced and then fixed by
subclassing.

| area | count | root causes | post-fix |
| --- | --- | --- | --- |
| af3 pair-pathway | 4 | tri-mult even/odd factor-split interleave (magnitude blow-up), incoming-swap, `target_feat` channel rotation-by-63, ending-node bias transpose | pair cos 0.011→**0.9985**, distogram 0.10→**0.9993**, single→**0.99998** |
| af3 confidence | 3 | `single_max_abs`=512 clamp (→None; corrupted pLDDT/resolved), PDE symmetrize-logits-not-pair, token channel-order rotation on confidence projections | confidence cos **0.99999**, trunk cos ≥**0.998** |
| af3 denoiser | 2 | cross-attn key double-norm (norm raw hidden), key-window gather zero-pad→clamp-inward | denoiser cos 0.278→**1.0** |
| boltz2 processor | 1 | folding `method_feature` 4 ('other', copied from affinity)→**1** ('x-ray') — perturbed every folding activation via method_embedding | all tensors **0.0** |
| af2-multimer | 3 | rel-encoding used `chain_index` not `symmetry_index`, template embedder missing `tri_mul_first`, Evoformer+extraMSA missing `opm-first` (+ processor Gram-Schmidt `template_backbone_frame`) | rel-L2 **0.0** all quantities; monomer no-regress **4.16e-5** |
| boltz1 | 1 | `count.clamp_min(1e-5)`→`+1e-5` on 7 denominators (`complex_pde` deliberately kept — upstream no-eps) | ptm **1e-8** / iptm **1e-10** |

**Cross-attn shared-module regression** (introduced by the af3 denoiser key-double-norm fix; caught by the
cross-family re-check): Protenix/odesign/pxdesign are `query_normed` (upstream double-adaLN), AF3 is raw. Fixed with a
per-family `ProtenixConditionedTransformerLayer` **subclass** rather than a `cross_attn_key_norm` flag on the shared
`ConditionedTransformerLayer` (Frankenstein principle). Recovery: Protenix coords 0.66→**2.15e-6**, odesign
7651.9→rel **2.9e-7**, pxdesign hidden denoiser rel-L2 0.373→**1.03e-6** (had been masked by a fake-denoiser
sampler_parity). Boltz/BoltzGen/ESMFold inert. New real parity scripts added: `pxdesign/denoiser_parity.py`,
`boltz1/`+`boltz2/processor_parity.py`, `boltzgen/structure_diffusion_parity.py`, `odesign/reverse_sampler_parity.py`.

### 2. All fold+design cores numerically validated

| family | status | numbers |
| --- | --- | --- |
| alphafold2 | mono + multi + stochastic VALIDATED | monomer **1.42e-4 Å**; stochastic Stage-1 bitwise **0.0** / Stage-2 **4.16e-5 Å**; multimer rel-L2 **0.0** (strict-load 0/0/0, 93.2M) |
| alphafold3 | trunk + confidence + denoiser VALIDATED | single **0.99998** / distogram **0.9993** / confidence **0.99999** / denoiser **1.0**; pair rel-L2 **0.0587** marginal (see caveats) |
| protenix | FAITHFUL | per-op bisect **0–1e-6**; query_normed cross-attn confirmed |
| boltz1 / boltz2 | PASS (4/4, 5/5) | **0–8.6e-5**; boltz2 method_feature + dtype-transparency fixes |
| boltzgen | FAITHFUL | denoiser **2.4e-4**, reverse-EDM **1.1e-4** |
| odesign | FAITHFUL | 40-step **4.9e-4 Å**, 100-step **4.3e-4 Å** |
| pxdesign | FAITHFUL | real denoiser rel-L2 **1e-6** |
| rfdiffusion | PASS (8/8) | worst **5.5e-4 Å** |
| proteinmpnn / ligandmpnn / mpnn_packer | validated | pre-/this-session real-weight |

**Caveats (recorded, not defects):** esmfold2 per-atom coords at a **0.215 Å** cross-impl floor (global scores PASS
~**2e-4**); af3 residual pair **rel-L2 0.0587** marginally over the 0.05 gate = deep-stack cross-framework fp
accumulation (every op cos=1.0 in isolation, structurally faithful); **NA-MPNN in contract-transition** (released-ckpt
re-validate pending); **MembraneMpnn deferred** by owner.

### 3. Contracts FROZEN (both)

- **Placement (§3.7):** contig→placement rename (`PlacementSpec`), `placement_map` adopted **4/4**, exhaustive
  coverage, 6 projections, realized-id unified to `chain_{n}` (rfdiffusion coords byte-identical; 3/4
  numeric-confirmed).
- **Constraint (§3.7.4):** `SymbolicBond` (`data/bonds.py` — D1 explicit-leaving / D2 unordered→SystemBond) +
  `Hotspot`/`Contact` (`data/conditions.py`). All 4 design families migrated off family-local placeholders,
  parity-gated no-regression.

### 4. Cross-cutting cleanup (all landed, parity-gated)

| axis | outcome |
| --- | --- |
| RNG (X-wave) | model graph = pure `(tensors, generator)`; shared `resolve_generator` (`pipelines/_generator.py`); esmfold2 documented **P1-exception** (single ordered global stream + integer-seeded golden) |
| dtype-transparency | boltz2 `.double()` **removed**, measured **7.58e-7** @ n_step=200; boltzgen `noise_scale²` faithful preserved |
| naming (A0–A5) | contig→placement, asym_id→chain_index, Boltz2AffinityResult→AffinityResult, hhblits_profile→msa_profile, one-letter table (A5/B-nv1) |
| unification | shared `ConfidenceHead` (B-eq1), pTM kernel (B-nv8, max_abs **0.0**), Trunk triangle/OPM (B-nv2/3/4 — AF2 = thin biased **subclass**, ~21Å einsum byte-exact **0.0**), converter schema (B-nv10) |
| softmax sentinel | 4 magic magnitudes → one dtype-aware `finfo.min` |
| OOps backend | VALIDATED (`scripts/parity/backend/oops_parity.py`, fp32 rel-L2 ~1e-6/0) + audited (integration healthy) |

### 5. Structural principles confirmed

- **Frankenstein principle** — genuine algorithm differences become **subclasses**, never flags on a shared module:
  `ProtenixConditionedTransformerLayer` (cross-attn), AF2 biased trunk subclasses, `BoltzGenCoordinateEdmSampler`.
  eps/config differences stay config and are never a reason to fork.
- **eps analysis** — only the softmax-sentinel is math-unifiable (applied); per-family confidence eps is
  upstream-faithful and kept; recorded in the family READMEs + `EPSILON_ANALYSIS.md`.
- **Diffusion samplers** — stay per-family (measured non-unifiable: rotation `R·v` vs `Rᵀ·v`, schedule divisor, RNG
  call-sites); shareable step-math extracted as opt-in primitives only.

### 6. Remaining (explicitly NOT done — none blocks this round)

| item | why deferred |
| --- | --- |
| OP2 AdaLN → OOps fused kernel | perf-only, optional (converter already emits the fused layout) |
| surface per-family confidence eps as config | hygiene |
| NA-MPNN released-ckpt re-validate | contract in-transition |
| esmfold2 pure-P1 RNG | owner-gated golden-regeneration |
| MembraneMpnn | deferred by owner |
| publication-level §9.1 | all-checkpoint coverage, model cards, Hub repos (separate large scope) |

<!-- 2026-07-19 DESIGN-GOAL PUSH — closes the autonomous items still open after the MIGRATION/CLEANUP ROUND; from here only owner-gated / no-runnable-path items remain. Authoritative status in CLEANUP_TRACKER.md. -->
## 2026-07-19 DESIGN-GOAL PUSH — autonomous items complete

The final push closed every remaining **autonomous** item from the cleanup tracker. The codebase is now at the
**COMPLETE design-goal state**: all fold+design cores validated, both contracts frozen, cross-cutting cleanup landed,
and the last hygiene/unification/OOps items done. Everything below is CONFIRMED-STABLE; what is left (last section) is
strictly owner-gated or has no runnable path — not autonomous work.

### 1. Autonomous items closed this push

| id | item | outcome |
| --- | --- | --- |
| X7 | softmax mask sentinel | 4 magic magnitudes → one dtype-aware `finfo.min` |
| X8 / eps | AF2 OPM eps + per-family confidence eps | eps analysis (`EPSILON_ANALYSIS.md`): only the softmax-sentinel is math-unifiable (applied); per-family eps upstream-faithful, kept and config-surfaced in the family READMEs |
| X5 | boltz2 `.double()` | removed (dtype-transparent, measured **7.58e-7** @ n_step=200); boltzgen `noise_scale²` faithful preserved |
| A3 | `sample_schedule_af3` → `karras` | renamed |
| A5 | residue_constants neutral maps | moved |
| A6 / A7 | triangle direction spelling / spelling nits | unified (folded into trunk / B) |
| B-nv1 | one-letter table | unified |
| B-nv8 | pTM kernel extract | done (`modules/head/tm.py`; max_abs **0.0** across 5 families) |
| B-nv10 | converter schema | frozen ordinary-pair schema (per-family README schema tables + `HUB_PUBLISH_PLAN.md` §7.3) |
| OP2 | AdaptiveLayerNorm → OOps fused `adaptive_layer_norm` | re-wired to consume the converter's fused `conditioning_projection (2*hidden,cond)` layout — the #1 converter-weight-surgery win; consumers (Protenix/Boltz/ESMFold cond) parity-gated no-regression |
| OP3 | TriangleAttention → OOps flash `triangle_attention` (+ varlen) | routed (Trunk wave); OP1 parity covers TriangleAttention e2e |
| OP4 | PWA dual-eps | gated `norm_pair_eps` guard landed (Trunk wave) — was rel-L2 0.125 when eps-unequal |
| C9 | NA-MPNN released-ckpt re-validate | VALIDATED under the frozen API |
| — | all-checkpoint coverage | §9.1 checkpoint inventory covers every published variant (per-family lists in `HUB_PUBLISH_PLAN.md`) |
| — | biotite venv fix | per-family venv pins isolated (1.4.0 / 1.7.1 / 1.0.1) — parity imports OFoldX + upstream in one process |
| — | boltz2 `contract.py` | typed tensor boundary frozen (with the method_feature 4→1 + dtype-transparency fixes) |
| — | mpnn_packer README | authored model card / side-chain-packing contract (`processor.json` `atom_context_num`) |
| — | §9.1 publication docs | documented limitations + licensing/redistribution facts per family (READMEs + `HUB_PUBLISH_PLAN.md`) |

### 2. FINAL remaining — owner-gated or no runnable path (NOT autonomous)

These are recorded as terminal for autonomous work; reopening requires an owner decision or an upstream/runnable path
that does not exist today.

- **Hub publishing** (`HUB_PUBLISH_PLAN.md`) — an owner-gated proposal; **nothing is published**. GREEN families
  (boltz1/2, boltzgen, esmfold2 folding weights, ligandmpnn, mpnn_packer, proteinmpnn, protenix, pxdesign,
  rfdiffusion, and the `ofoldx/ccd` dataset) are publish-ready pending owner authorization. **RED — af3** (non-commercial
  weights, no third-party redistribution grant) and **af2 DeepMind-parameter variant** (owner/upstream confirmation
  required; the OpenFold-trained AF2 weights are GREEN) — RED requires an explicit owner (and, for af3, DeepMind)
  decision. **nampnn / odesign** are GREEN-by-license but publication is blocked on registration / owner-gated
  qualification, not terms.
- **MembraneMpnn** — deferred by owner (absent by design this round).
- **esmfold2-Fast** — broken reference env (no clean upstream to parity against). **esmfold2 pure-P1 RNG** —
  golden-regeneration is an owner call (current `fork_rng` scopes the single-ordered-global-stream mutation;
  documented P1-exception).
- **af2 `model_2_ptm`** — divergence is **chaos-not-bug** (harness note: RNG / feature-order sensitivity, not an
  OFoldX defect).
- **boltzgen `structuretrained_small` / MembraneMpnn / ligandmpnn side-chain-packer** — **no runnable path**
  (contract-level: different arch / deferred / bounded-reopened vertical), not defects.
