# OFoldX Architecture

Status: design investigation for the first public OFoldX release.

Until the first intentional public release, accepted parts of this contract replace
prototypes directly and carry no backward-compatibility requirement. The document
remains falsifiable: authoritative model behavior, biological semantics, ordinary
user intuition, or a simpler complete design may require changing it before
changing production code.

This document is not currently an accepted architecture as a whole. Multiple core
sections have already been falsified by upstream and chemical evidence and remain
under revision. Presence in a technical subsection is not acceptance. A technical
implementation contract is **investigation** unless it begins with an explicit
`Status: frozen` line naming its current evidence. A frozen contract is the
implementation basis for one bounded Tik–Tok cycle, not a claim that future evidence
cannot improve it. An investigation or falsified contract does not authorize its
production implementation. Practice is evidence: implementation friction or new
authoritative behavior reopens the affected contract when it invalidates a premise.

Project principles, error boundaries, completeness requirements, CI limits,
pre-release migration rules, and the review process are governing constraints, not
technical candidates waiting for a `frozen` label. They apply while technical
contracts are investigated and implemented.

This document records the candidate cross-family architecture. Family READMEs record
the independently audited released behavior, revisions, checkpoints, equations,
defaults, limitations, and supporting evidence; they do not define or narrow the
upstream surface. Contributor process lives in AGENTS.md. Confirmed upstream defects
live in UPSTREAM_BUGS.md. High-impact findings and decisions that need concise owner
review are indexed in REVIEW.md rather than hidden in this document's full history.

## 1. Scope and principles

### 1.1 Product scope

OFoldX is MultiMolecule's structure-side sister project and the flagship of the two projects. MultiMolecule focuses on biological sequence models. OFoldX focuses on:

- biomolecular structure prediction;
- protein, nucleic-acid, backbone, and all-atom design;
- scientific scoring, including interaction scores;
- protein side-chain packing; and
- interactions among proteins, nucleic acids, ligands, ions, and other molecules.

The goal is a common model ecosystem for biomolecular structure: predictable at the entrance, chemically honest in the difficult cases, ordinary PyTorch at the tensor boundary, and independent of a private optimized backend.

The four shared task words are fold, design, score, and pack. They are routing vocabulary, not a taxonomy of architectures and not the completeness boundary of a model. A published family implements every scientific function of the released model, including relevant heads, conditioning modes, sampling modes, scoring paths, and checkpoint variants. OFoldX publishes no partial family under a weaker label.

Training orchestration is in scope. Differentiable Models remain ordinary
trainable PyTorch modules; the Runner (§10.3) is the config-only Danling consumer
that trains, evaluates, and infers with them. It never leaks training state into a
frozen Model, Processor, or config: a Model exposes forward-time predictions and
unreduced loss primitives, and the Runner-side objective owns the reduction,
dataset, optimizer, schedule, and metrics.

### 1.2 First principles

The following principles govern every public and internal decision:

1. Biological identity and topology survive every conversion.
2. A public noun exists only when it encodes a real invariant or serves more than one real production consumer.
3. One recurring job has one mechanism and one name.
4. The simplest complete design wins; a smaller incomplete design does not.
5. Public PyTorch defines correctness. Optimized backends implement the same semantics.
6. Models, Processors, repositories, task Pipelines, and Results have distinct ownership.
7. Persistent values contain their own meaning; they do not query a registry, Model, Processor, or source file later.
8. Architecture identity is a registered construction contract: an ID selects one
   concrete Model–Processor pair, and its typed configuration defines graph
   construction, state interpretation, and tensor meaning. Task words, optional
   heads, and one checkpoint's key set are not independent identity tests.
9. A Hub repository and its selected revision own publication history. OFoldX does not create a second identity or history system.
10. MultiMolecule is the baseline for ordinary Python, PyTorch, loading, and Runner design. Additional OFoldX machinery requires a structural, chemical, backend, or publication invariant.
11. OFoldX is research software for experienced researchers. Preserve scientific
    truth at public and persistence boundaries, then trust explicit caller intent
    and established Python and PyTorch contracts.
12. Performance is a scientific requirement. Defensive work that copies, scans, or synchronizes tensors needs evidence that it prevents a plausible wrong result.
13. Complete functionality is binary: a family is published or absent. Publication
    includes the distributed importable package, public reexports, user-facing docs
    or model cards, an OFoldX Hub repository, and registration. There are no
    maturity or reduced-support classes on any of those surfaces. Software
    feature-completeness is a first-class goal and is distinct from
    weight-redistribution eligibility. Converter-only publication relaxes ONLY the
    weight-redistribution requirement — OFoldX need not ship the Hub checkpoint — and
    never relaxes feature-completeness: the family's software must still be complete
    end-to-end (architecture, converter, typed contracts, task Pipeline, Result
    carrier, docs, and registration all working), independently of whether OFoldX may
    itself redistribute that family's weights. "Publication includes an OFoldX Hub
    repository" is therefore one satisfiable option, not the sole one — for a
    restricted-weight family the same bar is met in converter-only self-serve form:
    ship the complete software plus a converter that users run on weights they obtain
    themselves, with no OFoldX Hub checkpoint. That converter-only mode is a
    first-class, complete publication, not a lower maturity tier and not a license to
    leave the software functionally incomplete (AlphaFold3 — distributed internally
    while public users self-convert their own weights — is the canonical example).

Public Python identifiers follow ordinary CapWords capitalization: an initialism is
treated as one word, so the MPNN family classes use `Mpnn`, not `MPNN`. Package and
architecture IDs remain lowercase. Prose, citations, and provenance retain an
upstream project's official spelling such as ProteinMPNN, LigandMPNN, or NA-MPNN.
OFoldX-owned constants use names for the value they contain, such as
`AMINO_ACID_COMPONENT_IDS`; constant style is not a reason to reintroduce `MPNN`
into an OFoldX identifier.

Removing a shared abstraction never removes the released computation it previously
misrepresented. Every official function, conditioning mode, sampling control,
head, decoded output, and checkpoint variant remains required on the concrete
family surface that actually owns it. Only a confirmed upstream defect may differ,
and that divergence is recorded in `UPSTREAM_BUGS.md` and parity evidence.

#### 1.2.1 Code organization consistency

Status: frozen — MultiMolecule's model implementation guide and the owner's
explicit cross-model consistency correction support this bounded source-organization
contract for the current Tik–Tok cycle.

One semantic concept uses one canonical name across config fields,
Model/Processor/Pipeline signatures, Model attributes, Results, converters, and
repository operations. Scientific differences remain distinct; consistency never
flattens them into one misleading abstraction or term.

Every model package follows the same `config.py`, `model.py`, `processor.py`,
`pipeline.py`, `convert.py`, and `__init__.py` ownership grammar. An extra file
requires a concrete scientific or I/O ownership reason, not an upstream source
layout or local preference.

In a single-file `model.py`, declarations follow MultiMolecule execution order:
public Model variants; top-level modules; embeddings or input projections;
encoders or trunks; repeated layers or blocks; lower-level attention, MLP,
normalization, geometry, and diffusion components; heads; helpers; and typed
inputs, outputs, and Results last. Sibling families place equivalent large
substructures in the same relative position and use the same term. Within every
`__init__`, child modules are assigned in their `forward` execution order.

`Layer` names one atomic transform. `Block` names a repeatable composite unit
with multiple coupled transforms or residual paths. A repeated unit's class,
container attribute, and count use the same noun: `Layer/layers/num_layers` or
`Block/blocks/num_blocks`. Homogeneous repetition alone does not make a
composite block a layer.

The remaining package files are similarly predictable: `config.py` follows model
and component construction order; `processor.py` follows input-to-tensor lowering;
`pipeline.py` follows call-to-Result execution; `convert.py` follows load,
translate, validate, and save; and `__init__.py` groups public imports and exports
before registration. Source-only reordering or renaming must preserve state-dict
key meaning and exact numerical behavior. Conversion alone owns upstream names and
translates them into canonical OFoldX names. Each model's `convert.py` owns its
complete source-checkpoint mapping; `models/` contains no cross-model conversion
script. Format-independent artifact loading and saving may be shared, while
appropriate repetition of source-specific key and tensor rules keeps each model's
conversion contract local and readable.

### 1.3 Think before coding

Now is better than never, although never is better than right now.

OFoldX develops in explicit Tik–Tok cycles:

- Tik investigates one bounded problem and freezes the simplest complete contract
  justified by the evidence needed for that implementation.
- Tok implements that accepted contract and lets practice examine its ergonomics,
  ownership, performance, and integration.
- Contrary evidence returns the affected work to Tik; before the first release, no
  compatibility layer preserves a rejected direction.

Before one bounded Tok begins, its Tik freezes the premises and contract that
determine that work, together with acceptance criteria capable of falsifying it:

- authoritative upstream revision and checkpoint variants;
- every released scientific input, function, output, and evaluation effect;
- canonical input, Condition, tensor, and Result mappings;
- unresolved upstream ambiguities and intentional OFoldX divergences;
- the exact public surface and ownership boundaries; and
- the evidence that would demonstrate or falsify the finished behavior.

Implementation produces the acceptance evidence; a pre-implementation checklist is
not itself proof. Shared infrastructure may be established from its library contract
and current consumers. Any decision that owns a Model, Processor, Result, or
scientific family behavior requires at least one real family vertical. A bounded
family Tik inventories the complete released family surface; it cannot declare a
single convenient method or checkpoint to be the family merely to close the cycle.

Unknown facts outside that bounded implementation do not keep Tik open forever.
Unknown facts that would change its public meaning, scientific correctness,
ownership, or acceptance evidence do. Code is not a substitute for thinking, but
practice is how a considered design is tested. Already-spent effort is never
evidence for preserving a bad abstraction.

Parallel Tik–Tok work is grouped by architecture family. Every concrete released
Model still receives an independent checkpoint, command, callable, and output
inventory; the family Tik then computes and tries to falsify their shared
implementation intersection. Different families may be in Tik and Tok at the same
time, and a frozen family does not wait for unrelated audits. Shared files have one
implementation owner during Tok. `family` is review and source ownership, not a
public registry, manifest field, base class, or runtime value.

AGENTS.md defines the delegation and review procedure. This document does not duplicate subagent prompts, implementation phases, progress lists, or private-helper review rules.

### 1.4 Error boundaries

Validation belongs where semantic information would otherwise be lost:

- canonical values protect identity, topology, and referential closure;
- Processors protect structural and family-wide encodability of biological inputs,
  evidence, and Conditions;
- repository I/O protects architecture dispatch and persistent file meaning; and
- task methods reject operation-specific unsupported modes and protect sampling
  controls whose misuse could silently change the scientific algorithm.

Internal tensor values are trusted contracts. A concrete Model input or module does not recheck every type, shape, dtype, device, or finite value before ordinary tensor operations that already require them. Model code does not catch and rewrite native PyTorch failures merely to stabilize error wording. Backend dispatch does not duplicate the selected operator's tensor validation.

Let Python and PyTorch fail. Add a guard only when native execution would accept an input and produce a scientifically plausible but semantically wrong answer.

The public API introduces no parallel molecular representation or generic orchestration framework. A complex is simply a multi-instance MolecularSystem or Structure.

## 2. Public verbs and factories

### 2.1 Four biological verbs

OFoldX exposes exactly four top-level task functions:

~~~python
ofoldx.fold(...)
ofoldx.design(...)
ofoldx.score(...)
ofoldx.pack(...)
~~~

The same verbs name Model task methods and task-Pipeline methods. Predict, generate, sample, repack, structure_prediction, binding_affinity, and side_chain_packing are not parallel routing names. A family may expose precisely named expert methods for released computations that do not form another cross-family task.

Pack means protein side-chain packing in a fixed sequence and structural environment. It does not rotate ligands or redesign residue identity.

A heterogeneous input is an explicit MolecularSystem. A tuple that mixes sequences, Molecules, and ad hoc strings is not another input language. Bare SMILES is never guessed from a string; use Molecule.from_smiles.

The common signatures are:

~~~python
def fold(
    input: str | PathLike[str] | PolymerSequence | MolecularSystem | Structure,
    *, model: str | PathLike[str], revision: str | None = None,
    token: str | None = None, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    msa: MSA | MSACollection | None = None, templates: Sequence[Template] = (),
    num_samples: int = 1,
    seed: int | None = None, generator: torch.Generator | None = None,
) -> Any: ...

def design(
    input: MolecularSystem | Structure | str | PathLike[str],
    *, model: str | PathLike[str],
    revision: str | None = None, token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None, msa: MSA | MSACollection | None = None,
    templates: Sequence[Template] = (),
    conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
    num_candidates: int = 1, seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any: ...

def score(
    input: MolecularSystem | Structure | str | PathLike[str],
    *, model: str | PathLike[str], revision: str | None = None,
    token: str | None = None, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    seed: int | None = None, generator: torch.Generator | None = None,
) -> Any: ...

def pack(
    input: Structure | str | PathLike[str],
    *, model: str | PathLike[str], revision: str | None = None,
    token: str | None = None, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    positions: Sequence[ComponentReference] | None = None, num_samples: int = 1,
    seed: int | None = None, generator: torch.Generator | None = None,
) -> Any: ...
~~~

The `Any` return annotation is confined to these dynamically dispatched one-shot
functions. The selected standard Pipeline returns the shared task Result. A
concrete family Pipeline also returns that shared Result when it can represent the
released decoded output without loss; only an irreducible family output justifies
a named family Result. Explicit Model and Pipeline classes publish their exact
return types. OFoldX does not introduce a nominal Result base, Protocol, or closed
union solely to make this dynamic boundary look more precise than it is.

These signatures are common-case one-shot conveniences, not a union of every
released family's controls. Model is required because there is no scientifically
universal default. Counts are positive. Seed and generator are mutually exclusive.
A top-level verb passes only the arguments shown above. It has no generic `scope`,
`options`, or unchecked keyword bag. Complete family support does not require
copying every family argument onto the top-level function: a concrete family
Pipeline exposes the precise biological signature of its released operation, and
the concrete Model exposes its precise tensor-level controls.

One-shot dispatch constructs its known call arguments explicitly. Neutral common
defaults that carry no supplied evidence — `msa=None`, empty `templates`, empty
`conditions`, and `seed=None` or `generator=None` — are omitted. A nonempty MSA,
Template, or Condition is forwarded, so a concrete method that does not own it
fails through its ordinary Python signature. Dispatch does not inspect signatures,
catch `TypeError`, or add `**kwargs` to concrete methods.

For example, sequence-scoring Pipelines may accept an explicit sequence, selected
positions, and a number of autoregressive decoding orders, while an affinity
Pipeline may require a ligand and separate structure-sampling counts. Those are
different scientific arguments and retain their family-owned names rather than
being flattened into generic `num_samples` or `scope` parameters. Standard task
Pipelines cover the same common case as the top-level verb; concrete family
Pipelines cover the complete operation. Signature details remain investigation
until each family inventory is complete. If an operation has no scientifically
unambiguous common-case default for a required family argument, callers use its
concrete Pipeline; the top-level convenience does not guess.

Variable-topology design uses the concrete family's explicit topology argument,
such as a `contig`, rather than overloading `conditions` or `input=None` in the
common signature. A supplied MSA, Template, coordinate observation, or Condition
is consumed or rejected before tensor computation; it is never silently discarded.

The design union contains the two exact fixed-topology identity values shared by
its current consumers. Common fold has no generic `conditions` argument: Protenix,
Boltz-1, Boltz-2, RFdiffusion, ODesign, and BoltzGen own different spatial,
guidance, topology, and potential semantics. Their concrete Pipelines expose exact
family arguments. OFoldX defines no global `Condition`, `FoldCondition`, or
`DesignCondition` alias merely to avoid repeating concrete names. PlacementSpec and
family controls are ordinary arguments rather than Conditions.

A partial Structure supplies observed geometric evidence. Its present coordinates
are not implicitly fixed and are not silently treated as a template. Use a
MolecularSystem for topology-only input and Template for aligned template evidence.
A family that can fix observed coordinates exposes its exact concrete control once
that output-frame contract freezes; the common verb does not guess.

#### 2.1.1 Common pack boundary

Status: frozen — the released side-chain Packer, its use across the official
designer matrix, and the family Tik in Sections 4.6 and 5.5 support this bounded
top-level and standard-Pipeline contract.

For pack, `sequence=None` derives identities from the Structure. A bare string is
valid only for one unambiguous polymer. A mapping is keyed by canonical
`instance_id`; each value is either an ordinary one-letter biological sequence or
an exact ordered sequence of component IDs. The sequence declares the already
chosen target identities: it may replace known identities, but the packer never
chooses identities itself. Different instances of one input entity may receive
different target sequences; the concrete packer explicitly constructs the
resulting entity-to-instance mapping rather than asking canonical values to infer
it from equality.

`positions` is an ordinary ordered sequence of canonical `ComponentReference`
values. `positions=None` selects every residue the concrete packer can pack, while
an empty sequence selects none. Every identity changed by `sequence` must also be
selected for packing; otherwise the new component identity would retain side-chain
coordinates and chemistry belonging to the old identity. Packing returns the
system after target-identity realization and preserves its assembly, resolvable
topology, backbone, ligands, and unselected observations. These ordinary
containers do not justify `SequenceAssignment` or `ResidueSelection` wrapper types.

A `str | PathLike` pack input means a local structure format accepted by the
selected concrete family; it is not a universal PDB/mmCIF promise. The common
boundary performs no format detection or conversion beyond that family contract.

### 2.2 Task-Pipeline factory

Status: reopened — task-first dispatch and concrete class routing are frozen in
Sections 7.2 and 7.4, while the exact repository-to-concrete-class construction
callable still requires two completed verticals.

The task-first factory is:

~~~python
def pipeline(
    task: Literal["fold", "design", "score", "pack"],
    *, model: str | PathLike[str],
    revision: str | None = None, token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Any: ...
~~~

Both task and model are required. The model reference selects a repository whose
root is a runnable Model–Processor pair; task selects the biological verb. The
factory never guesses one from the other. With no `pipeline.json`, it returns the
matching standard task Pipeline when the requested task is in the root
registration's `standard_tasks`. With `pipeline.json`, the registered concrete
class is authoritative for that repository. The requested task must name a callable
public method on that class; an unknown marker, a missing method, or concrete
construction failure does not fall back to the root pair. In every successful case
the caller still says the task explicitly. `task` selects a class but creates no
selected-task state: one concrete object may publicly expose several shared verbs.
Here runnable means that ordinary explicit runtime placement can execute the complete
pair. It does not mean that omitting `dtype` reconciles an arbitrarily cast Model
artifact with the Processor's ordinary constructor dtype.

These forms are deliberately invalid:

~~~python
ofoldx.pipeline("ofoldx/rfdiffusion")
ofoldx.pipeline(model="ofoldx/rfdiffusion")
ofoldx.pipeline("design", model=loaded_model)
~~~

Loaded live objects use the explicit constructors in Section 6.

### 2.3 Auto construction and loading

Status: frozen — the two Model constructor forms, repository-only Processor
dispatch, repository-only compatibility spelling, standard root-pair loader, the
config-less MembraneMpnn counterexample, and one-pair concrete repository vertical
in Section 7 support this bounded API. This does not freeze multi-pair Pipeline
construction or family publication.

OFoldX owns construction, architecture dispatch, loading, and Model/Processor pairing:

~~~python
AutoModel(
    pretrained: str | PathLike[str], *, revision: str | None = None,
    token: str | None = None, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ConcreteModel
AutoModel(
    config: ConcreteModelConfig, *, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ConcreteModel
AutoProcessor(
    pretrained: str | PathLike[str], *, revision: str | None = None,
    token: str | None = None, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ConcreteProcessor
~~~

String or PathLike means a repository. A typed config means fresh construction and
selects its concrete Model class by exact registered config-class identity. Bare
mappings are not guessed as Model configs. A Processor with persistent lowering
configuration is already named by its concrete config class and is constructed
directly as `ConcreteProcessor(config, device=..., dtype=...)`; a config-less
Processor is constructed directly without a positional config. `AutoProcessor`
does not maintain a second Processor-config-to-class registry merely to spell that
ordinary constructor indirectly. Processor and Config above are collective terms,
not required nominal base classes. Every registered Model inherits the minimal
public `ofoldx.Model` base.

`Model.__init__` performs ordinary `nn.Module` initialization once, stores the
concrete config, and initializes the backend policy. Its default
`reset_parameters()` delegates to each child Module's native PyTorch initializer;
an architecture overrides that method when its released initialization differs or
when it owns parameters directly. The base does not claim a singular device or
dtype for a potentially mixed or sharded module, and it does not duplicate pair-
and Pipeline-owned repository writing with a Model-only save API. Placement uses
`nn.Module.to`; repository writing is specified in Section 7.5.

AutoModel(pretrained) returns an eval-mode Model. AutoModel(config) returns a normal
fresh Model without changing its training flag. Runtime device and dtype are not
config fields.

`AutoModel.from_pretrained(pretrained, ...)` and
`AutoProcessor.from_pretrained(pretrained, ...)` are guarded, repository-only thin
compatibility spellings for the corresponding repository constructors above. They
accept only `str | PathLike[str]`; passing a config fails before dispatch rather
than silently becoming another constructor form. They add no option or lifecycle
behavior. The constructors remain the primary API.

### 2.4 Reference classification

Status: investigation — one small public classifier replaces the previous private
reference hierarchy; its use at each overloaded boundary still requires call-site
audit.

Classification exists only at genuinely overloaded boundaries. It has one public
function and no `Reference` class or parallel value hierarchy:

~~~python
# ofoldx.references
def classify_reference(reference: str | PathLike[str]) -> str | Path: ...
~~~

A PathLike, an existing string path, or a string with an explicit local spelling is
returned as an expanded `Path`. Explicit local spellings include POSIX absolute and
relative prefixes (`/`, `./`, `../`, and `~/`), Windows drive prefixes, and UNC
paths, independent of the host platform. Every other string is returned unchanged.
The classifier does not validate, normalize, or special-case the remaining string,
including an empty string; the consuming boundary decides its meaning and lets its
ordinary parser fail. `classify_reference` need not be reexported from the package
root.

At a repository boundary, a remaining string is a Hub repository ID. Use PathLike
or an explicit local spelling to express a missing local repository unambiguously.
Repository resolution may keep path, Hub ID, and resolved commit aligned in one
small internal NamedTuple. That loader-only value is justified transport state, not
a public reference abstraction.

The fold input is the only biological string-or-path overload. PathLike and an
existing string are paths; a remaining valid protein sequence is inline sequence;
any other string is passed to the ordinary structure reader and naturally raises
its file error. Path-only readers and writers use normal Python path semantics:
an input need not exist to be recognized as a path, and an output path may be
created. They do not apply repository classification or require `./` merely to
name `target.cif` or `out.cif`.

Revision applies to Hub repositories and selects one Hub snapshot. It is not a second local-directory identity.

### 2.5 Direct and repeated use

Status: investigation — the ordinary direct and repeated-use forms are stable in
intent, while family call signatures and repository-loading details remain open.

Repeated biological inference loads once:

~~~python
pipe = ofoldx.pipeline(
    "fold",
    model="ofoldx/alphafold3",
    device="cuda",
    dtype=torch.bfloat16,
)
result = pipe.fold("target.cif", num_samples=4, seed=7)
~~~

Direct paired use loads the objects explicitly:

~~~python
model = AutoModel("ofoldx/alphafold3", device="cuda")
processor = AutoProcessor("ofoldx/alphafold3", device="cuda")

inputs = processor(structure)
result = model.fold(inputs)
~~~

The Processor return is the concrete static and biological input for that Model.
Each concrete Model documents its own explicit raw `forward` signature; the raw
call may also require Model-owned dynamic tensors, so OFoldX does not prescribe a
universal `model(inputs)` spelling. Raw forward inherits the caller's grad and
autocast contexts. The one-invocation Model task method owns its scientific
inference scopes.

Two direct Hub Auto calls resolve independently and therefore do not promise one
shared immutable snapshot. Callers who require repository-level pairing use
`pipeline(...)` or explicitly load both objects from the same resolved local
snapshot. Models and Processors do not retain hidden repository state to repair
independent calls after construction.

## 3. Canonical identity and topology

### 3.1 One domain representation

Status: reopened — the earlier `EntityInstance` wrapper was unnecessary, but the
subsequent instance-only representation was falsified by Protenix, AlphaFold 3,
and Boltz. Those Models consume same-entity and copy-ordinal features. The minimal
system-local grouping below is the current investigation, not a frozen contract.

I/O, Processors, Results, evaluation, and users should share one molecular
representation. Families lower it to their tensor layouts; they do not introduce a
second public structure object.

Canonical references are flat:

~~~python
ComponentReference = tuple[str, int]
AtomSite(instance_id: str, component_index: int, atom_id: str)
~~~

`ComponentReference` is `(instance_id, component_index)`. Resolution reads the
instance's entity ID from `system.instances`, then the ordinary molecular value
from `system.entities`, and finally its component position. No `Entity`,
`EntityInstance`, or component-provider object intervenes. A `Molecule` occurrence is already
realized, while a `PolymerSequence` occurrence receives the pure graph resolution
described below. `AtomSite` adds an `atom_id` local to that occurrence and
identifies only an atom retained by realization. `component_index` is zero-based in
the occurrence's complete declaration. It never disappears or changes meaning by
molecule kind and is not derived from the subset of components or atoms that
happened to have coordinates.

An entity ID is a system-local declaration that occurrences are copies of the same
molecular entity. It is scientific input: released Models use it to distinguish
copies of one entity from chemically equal independent entities. Equality or
Python object identity never infers this relation. Two entity keys may deliberately
hold equal values, including the same Python object, and remain independent.
`EntityInstance` is still absent because the two strings it would wrap are already
the keys and values of the instance mapping.

Source `label_entity_id` is evidence for an importer's partition, not a universal
canonical key that must be copied verbatim. Importers preserve its source identity
for round-trip I/O while constructing the canonical system-local grouping required
by the realized chemistry.

Source instance-ID allocation is importer policy. A common bare string or
`PolymerSequence` is one generated polymer with entity ID `"A"` and instance ID
`"A"`; its public Result references are therefore predictable. The common
signature does not accept `Sequence[PolymerSequence]`, whose meaning is ambiguous
between batching and a multi-chain assembly. Callers construct `MolecularSystem`
explicitly for more than one instance, so families cannot invent incompatible IDs
behind one top-level call.

### 3.2 Chemical components

Status: reopened — required chemical facts and partial-graph semantics are
evidence-backed, but stereochemical interpretation and transform round-trip remain
unresolved. This is an investigation contract, not frozen. Section 3.2.1 freezes
only the smaller partial-observation slice already consumed by released families.

The current minimum chemical values are:

~~~python
ChemicalAtom(
    atom_id: str,
    element: str,
    isotope: int | None = None,
    formal_charge: int | None = None,
    radical_electrons: int | None = None,
    explicit_hydrogens: int | None = None,
    implicit_hydrogens: int | None = None,
    implicit_hydrogens_allowed: bool | None = None,
    aromatic: bool | None = None,
    stereo: str | None = None,
)
ChemicalBond(
    atom_a: str,
    atom_b: str,
    order: Literal[1, 2, 3, 4, "dative"] | None = 1,
    aromatic: bool | None = False,
    stereo: str | None = None,
)
PolymerAttachment(
    atom_id: str,
    leaving_atom_ids: frozenset[str],
)
ChemicalComponent(
    component_id: str,
    atoms: tuple[ChemicalAtom, ...],
    bonds: frozenset[ChemicalBond] | None,
    previous_attachment: PolymerAttachment | None = None,
    next_attachment: PolymerAttachment | None = None,
    reference_coordinates: tuple[tuple[float, float, float] | None, ...] | None = None,
)
~~~

`ChemicalComponent` has no polymer-membership field. Chain context belongs to the
containing `PolymerSequence`; any family-specific row classification, including
NA-MPNN lowering of hybrid occurrences, belongs to that family's later Processor
investigation rather than reusable component chemistry.

`PolymerAttachment` declares only its retained endpoint and leaving atom IDs. It
does not carry a second copy of bond order or aromaticity: when both adjacent
attachments are present, the default polymer rule owns the resulting
non-aromatic single bond.

`atom_id` and `element` are always known. Optional chemical facts use `None` when
they are unknown; absence does not silently mean neutral, zero, non-aromatic, or
stereochemically unspecified by observation. Explicit and implicit hydrogen counts
and permission for implicit hydrogen completion remain distinct because the SMILES
and Boltz RDKit boundaries consume those separate facts. They are not collapsed
merely because MPNN atom-context lowering ignores them. Deuterium and tritium are
explicit hydrogen atom nodes with isotope facts.

`bonds=None` means that the component is an open, partial atom observation: neither
complete atom inventory nor complete topology is asserted. An empty frozenset
means that the declared component is complete and is known to have no bonds. This
distinction is required because atom-context consumers can use partial components
while topology-dependent Processors cannot. The latter reject unsupported partial
input at their own scientific boundary; the canonical value does not pretend that
all consumers require complete chemistry.

Atom IDs are unique in the component. Every declared bond endpoint and attachment
atom resolves there. Numeric bond order and aromaticity are independent facts;
localized aromatic bonds may retain a numeric order, while a delocalized aromatic
edge may use `order=None, aromatic=True`. `order=None, aromatic=False` is a known
non-aromatic edge of unknown order; `aromatic=None` means aromaticity itself is
unknown. The ordinary default declares a non-aromatic bond, so an importer lacking
that evidence passes `None` explicitly.

For numeric or unknown order, `atom_a` precedes `atom_b` in the component atom
declaration. For a dative bond, `atom_a` is the donor and `atom_b` the acceptor,
regardless of declaration order. A component contains at most one bond for an
unordered physical endpoint pair. The importer or caller supplies the required
orientation and rejects duplicates; the value does not rewrite itself during
construction. The exact atom- and bond-stereo descriptor languages, neighbor
ordering, and post-transform recomputation are not yet accepted.

Reference coordinates, when present, align with atom declaration order, contain an
Ångström xyz triple or `None` per atom, and contain at least one xyz triple.
`reference_coordinates=None` is the one ordinary no-conformer state. An importer
turns an all-`None` source conformer into this value at its boundary; the component
does not mutate after construction. Reference coordinates are component geometry,
not an assembly coordinate observation.

These are ordinary values. No custom equality or hashing, constructor mutation,
hidden comparison mode, `same_component_chemistry`, provider, cache, or transform
history is part of the canonical model. Importers return exact known facts. A
transform that needs an atom mapping returns that mapping directly rather than
storing it in the component.

#### 3.2.1 Partial component observations

Status: frozen — the released LigandMPNN and NA-MPNN parsers accept arbitrary
non-water HETATM observations and consume their atom IDs, elements, and
coordinates without requiring a CCD graph. SMILES and Boltz independently require
the distinction between known zero/false chemistry and an unknown fact.

`ChemicalAtom.atom_id` and `element` are required. Its isotope, charge, radical,
explicit- and implicit-hydrogen, implicit-hydrogen-permission, aromaticity, and
stereo fields default to `None`; package-audited components and SMILES importers
pass every known zero and boolean explicitly. `None` never silently means neutral,
zero, non-aromatic, or permitted.

`ChemicalComponent.bonds` is a required
`frozenset[ChemicalBond] | None`. `None` is an open partial atom observation: its
atom inventory and topology are not asserted complete. `frozenset()` is the
different, positive statement that the complete component is bondless. Declared
bonds are checked for endpoint and physical-edge closure only when the field is
not `None`; atom-ID and attachment closure remain independent. There is no
`PartialComponent`, completeness enum, provenance bag, provider, default bond set,
or generic validation framework. This bounded slice does not freeze the unresolved
stereo descriptor languages, graph transforms, or complete Structure I/O contract.

### 3.3 Polymers and molecules

Status: reopened — per the owner review this §3.3 value-shape slice is NOT frozen and does
not authorize implementation. The committed `ofoldx/data/molecular.py` value shape
(`PolymerSequence`, `Molecule`, `MoleculeBond`, their `from_one_letter`/`from_smiles`
factories, `PolymerType`) and the endorsed core below are the candidate direction: an
entity ID is never a field, `EntityInstance` is deleted, an entity holds its exact
`ChemicalComponent` occurrences directly, and heavy-atom filtering is consumer-side. But
blocking corrections keep it an investigation candidate: `PolymerType` retains an
independent `"pna"` (5 members); `instances` must serialize as an ordered entry array, not
a sort-key-reordered JSON object; `from_one_letter` must resolve the audited ambiguity
symbols (protein `X→UNK`, RNA `N→N`, DNA `N→DN`); `from_smiles` produces a single `UNL`
`ChemicalComponent` using `ChemicalBond`, not a `MoleculeBond`; import classification may
do format-local inference when a source declaration is missing (§3.5.2); MSA and Template
entity ownership moves to reopened §3.6; and `iter_bonds()` cross-layer duplicate-edge
resolution and intrinsic-stereo completeness stay unresolved (§3.4.3). The
linear-adjacency graph-edit transform, cyclic closure, and family lowering listed under
"Open questions" below are likewise not frozen.

Polymer and non-polymer values hold their chemistry directly as ordered exact
`ChemicalComponent` occurrences rather than component-ID strings resolved against a side
table (when a source declaration is absent, an importer may still fall back to
format-local, component-ID-keyed inference — AF3 `guess_polymer_type`, OFoldX
`io/structure.py:_component_polymer_type`; §3.5.2):

~~~python
PolymerType = Literal["protein", "rna", "dna", "dna_rna_hybrid", "pna"]   # 5 members; "pna" independent — recognized+lowered-differently by released parsers; no shipped from_one_letter path, but explicit ChemicalComponents still construct it

@dataclass(frozen=True)
class PolymerSequence:
    polymer_type: PolymerType
    components: tuple[ChemicalComponent, ...]
    @classmethod
    def from_one_letter(cls, sequence: str, *, polymer_type: PolymerType = "protein") -> PolymerSequence: ...
    def __len__(self) -> int: ...          # == len(self.components)

@dataclass(frozen=True)
class MoleculeBond:
    component_a: int
    atom_a: str
    component_b: int
    atom_b: str
    order: float = 1.0
    aromatic: bool = False
    stereo: str | None = None
    stereo_atoms: tuple[tuple[int, str], tuple[int, str]] | None = None

@dataclass(frozen=True)
class Molecule:
    components: tuple[ChemicalComponent, ...]
    bonds: frozenset[MoleculeBond] = frozenset()
    @staticmethod
    def from_smiles(smiles: str, *, atom_ids: tuple[str, ...] | None = None) -> Molecule: ...
~~~

The ordered tuples hold exact `ChemicalComponent` occurrences, not component IDs that
require a separate lookup table. Repeated components are ordinary repeated values, and
occurrence index disambiguates them. A D-polypeptide remains `polymer_type="protein"`;
handedness belongs to exact component stereochemistry, not a second polymer kind. The
candidate `PolymerType` retains an independent `"pna"` (a confirmed-correct 5th member):
PNA is recognized and lowered *differently* by the released *parsing* pipelines (AF3 →
peptide/protein via `mmcif_names.py`; Protenix → nucleic via `constants.py`), so folding
it into another polymer kind would erase a real canonical distinction. `"pna"` is
nonetheless **not constructible via the shipped `from_one_letter` convenience** — there is
no PNA one-letter map and no shipped PNA `ChemicalComponent`s for it to resolve — but an
explicit `Molecule`/`PolymerSequence` assembled from explicit `ChemicalComponent`s can
still carry a PNA occurrence. So its `from_one_letter` producer and concrete per-family
lowering stay reopened.

**Entity ID is not a field.** Neither value carries an `entity_id`. An entity ID is
exclusively the *key* of `MolecularSystem.entities` and the corresponding *value* of
`MolecularSystem.instances` (§3.1, §3.4.1). A bare value carries no identity; whoever
inserts it chooses its key (default `"A"`, §3.1). `EntityInstance` remains absent because
the two strings it would wrap are already the key and value of the instance mapping.

**`components` replaces `monomers`.** The prior `PolymerSequence.monomers` (component-ID
strings resolved against a system-level component table) is removed, and the system-level
`polymer_components` table is removed with it. Occurrence `i`'s old ID string is
`components[i].component_id`; repeated components are ordinary repeated values
disambiguated by index. A `PolymerSequence` occurrence is realized to retained atoms by
the frozen §3.4.2 predicate; a `Molecule` occurrence is already realized.

**Factories.** `from_one_letter` takes keyword-only `polymer_type` (default `"protein"`)
and no entity ID. It rejects `"dna_rna_hybrid"` (which requires explicit occurrences).
For every other symbol it resolves via `data.components.get_component`, which yields a
fully-built component directly resolvable by `system.component()`/`iter_atoms()` with no
side table. It must resolve the audited ambiguity symbols the released parsers accept:
protein `X → UNK`, RNA `N → N`, and DNA `N → DN`. A symbol absent from the one-letter map
**or** mapping to a non-audited ID raises `ValueError`; because `get_component` is limited
to the audited set, the non-audited one-letter targets (protein `U→SEC`, `O→PYL`; rna `I`;
dna `I/DI/DU`) all raise. The package-audited set is exactly **31 IDs** (21 protein incl.
`UNK`, 10 nucleic); modified or non-audited residues must be supplied through `components=`.
`from_smiles` is a `staticmethod` taking keyword-only `atom_ids` and no entity ID; it
returns a `Molecule` holding a single `UNL` `ChemicalComponent` whose intra-component
connectivity is `ChemicalBond` edges built by the private RDKit importer — it emits no
`MoleculeBond`, because one SMILES lowers to one component occurrence and `MoleculeBond`
only bridges two occurrences. Ordinary RDKit failures fail.

`MoleculeBond` carries the same numeric `order` (float, default `1.0`), `aromatic`, and
stereo facts as `ChemicalBond`, across two *distinct* occurrences in one `Molecule`.
`Molecule.__post_init__` validates endpoint resolution, differing occurrences, no
self-loops, and no duplicate physical edges; constructing a `MoleculeBond` adds a pure
edge and never edits or removes atoms. `ChemicalBond` owns topology inside one component;
`MoleculeBond` owns intrinsic topology between two component occurrences in one
`Molecule`, and its flat endpoint fields avoid another site type.

**Codec facet — paired entity serialization (candidate).** Because an entity carries no
ID, `_encode_entity(entity)` serializes only the entity's own fields and must not emit
`entity_id` or `monomers`:

~~~
PolymerSequence -> {"kind":"polymer_sequence", "polymer_type": …, "components": [_encode_component(c) …]}
Molecule        -> {"kind":"molecule", "components": [_encode_component(c) …],
                    "bonds": [_encode_molecule_bond(b) for b in sorted(entity.bonds, key=_molecule_bond_key)]}
~~~

`_decode_entity(value)` reconstructs with no `entity_id` argument; the required-key sets
are `{"kind","polymer_type","components"}` and `{"kind","components","bonds"}`. The entity
ID lives only at the system-mapping level: `encode_system` keys entities by their
`system.entities` key. Because `instances` order determines each entity's copy ordinal, it
must serialize as an **ordered entry array**, never a plain JSON object: encoding
`instances` as a JSON object plus the writer's recursive `json.dumps(..., sort_keys=True)`
on the `ofoldx_structure` system payload (`ofoldx/io/structure.py:1503`) erases instance
order on every `Structure` round-trip. Serialize instances as an ordered
`(instance_id, entity_id)` array; this ordered entry array is the single normative
requirement, because it makes order intrinsic to the payload regardless of any writer
flag (the recursive `sort_keys=True` writer above is what corrupts a plain object, not an
equal alternative fix). `decode_system` rebuilds `entities` and reads the instance
order back from that array. The codec also drops `ChemicalComponent.polymer_types` and
`PolymerAttachment.order`/`.aromatic` (removed fields, §3.2). MSA and Template entity
serialization is not owned here; it moves to reopened §3.6 alongside the current
`evidence.py`/`_codec.py` field mismatch.

Open questions (this candidate is not frozen):

- `iter_bonds()` cross-layer duplicate-physical-edge resolution and intrinsic-stereo
  completeness — the accessor cannot claim a complete graph while duplicate physical edges
  across owner layers are not deduped and a consumer must still read the owner
  `ChemicalBond`/`MoleculeBond` declaration for missing stereo (§3.4.3);
- the exact ordered-entry-array persistence shape for `instances` and its decode contract,
  which depend on the reopened system ordering/persistence facet;
- MSA and Template entity ownership and their `evidence.py`/`_codec.py` field mismatch —
  moved to reopened §3.6;
- the linear-adjacency pure graph edit's public signature, its returned atom mapping,
  `reference_coordinates` filtering, and stereo preservation or recomputation — only the
  §3.4.2 retained-atom predicate is settled, not the transform itself;
- cyclic closure as an explicit `SystemBond` (terminal edit plus closure edge) and exact
  protein, RNA, DNA, hybrid, and PNA cyclic semantics;
- `dna_rna_hybrid` and `pna` family lowering — the canonical domain keeps an independent
  `"pna"` precisely because AF3 and Protenix lower it differently, but the concrete
  per-family lowering stays reopened;
- `Molecule` source-format or provenance (a separate provenance/conditions facet); this
  contract deliberately gives `Molecule` no `provenance` field.

### 3.4 Assemblies

Status: reopened — direct system ownership and bond origins have survived, but the
instance-only candidate did not. Cross-component bond chemistry, cyclic
materialization, full topology traversal, and I/O remain unresolved. The bounded
entity/instance identity slice below is frozen independently.

#### 3.4.1 Entity, instance, and occurrence identity

Status: frozen — PDBx/ModelCIF and released family identity consumers support the
explicit entity-to-instance mapping and flat positional occurrence references
below. This subsection freezes only that ownership, declaration order,
`ComponentReference`, `AtomSite`, and direct component lookup. Retained-atom lookup
is frozen separately by the paired-adjacency slice. `SystemBond` chemistry and
equality, complete graph materialization, and decoder inference of output entity
partitions remain reopened.

An assembly stores ordinary molecular entity definitions and an insertion-ordered
mapping from stable instance IDs to those definitions:

~~~python
MolecularSystem(
    entities: dict[str, PolymerSequence | Molecule],
    instances: dict[str, str],  # instance_id -> entity_id
    bonds: frozenset[SystemBond] = frozenset(),
)
~~~

At construction, every instance resolves to one entity and every stored entity has
at least one instance. Equivalently,
`set(entities) == set(instances.values())`; an empty system is valid. Several
instances may name one entity, but this copy relation never implies a design tie.
Two equal values, and even the same Python value object, under different entity IDs
remain distinct entities. Instance insertion order is declaration order, not
molecular identity. It gives each family Processor the order from which to derive
its own copy ordinal. The system does not sort IDs, infer grouping, retain a hidden
reverse index or cache, or maintain a component table.

`MolecularSystem` is an ordinary public class with object-identity equality. Its
constructor checks the mapping invariant and shallow-copies the two outer
dictionaries with `dict(...)`; nested molecular declarations are not deep-copied.
The public dictionaries remain ordinary live state. Later caller mutation may
invalidate the construction-time mapping or any bond closure; the caller owns that
mutation. The system does not install a proxy, hidden backing mapping, cache,
normalizer, or automatic rescan.

The bounded direct component surface is:

~~~python
ComponentReference = tuple[str, int]

@dataclass(frozen=True)
class AtomSite:
    instance_id: str
    component_index: int
    atom_id: str

system.component(reference: ComponentReference) -> ChemicalComponent
~~~

An `AtomSite` uses ordinary generated equality and hashing over all three fields;
it does not sort, normalize case, or rewrite identifiers. It identifies an atom in
the canonical molecular graph, not an alternate coordinate location or ensemble
model row. A component occurrence is identified only by
`(instance_id, component_index)`. Equal component values, and even one shared
Python object, at two positions remain two occurrences.

`component()` follows the two ordinary mappings, resolves the occurrence, and
returns its stored component value. The component index is non-negative and
zero-based; `component()` rejects a negative value because native Python indexing
would silently select a different scientific occurrence. Unknown IDs and positive
out-of-range indices fail through ordinary `KeyError` and `IndexError`.

`PolymerSequence` owns an ordered tuple of exact component declaration values.
`Molecule` owns an ordered tuple plus its intrinsic cross-component bond
declarations. Entity IDs, instance IDs, source identity, and occurrence identity do
not belong to either value. Their ordinary equality, where available, is exact
declaration equality; it is not graph isomorphism, chemical equivalence, or entity
identity. A canonical system never infers entity grouping from that equality or
from Python object identity.

Stored-bond closure remains with the reopened `SystemBond` and topology contract
because its final fields and ordinary value semantics are not yet frozen. A decoder
explicitly constructs its output `entities` and `instances`. Each fixed-length
family freezes its own partition and entity-ID allocation only after defining which
released output facts establish that two realized copies remain one molecular
entity.

Core importers and Processors do not share one grouping heuristic. They translate
their authoritative input path: PDBx and Protenix preserve explicit source entity
relations, while AF3 and Boltz may derive relations according to their released
input rules. The canonical system itself never infers grouping from equality.

#### 3.4.2 Paired-adjacency retained-atom lookup

Status: frozen — standard polymer attachment semantics and the released family
consumers support this bounded retained-atom rule. It does not freeze complete bond
materialization.

For adjacent polymer components `i` and `i + 1`, the pair is active only when the
left component declares `next_attachment` and the right component declares
`previous_attachment`. An active pair hides the left attachment's
`leaving_atom_ids` and the right attachment's `leaving_atom_ids`; both attachment
endpoint atoms remain. If either attachment is absent, no bond is inferred and no
leaving atom is hidden. Terminal leaving groups therefore remain exposed.

This rule applies only to default linear `PolymerSequence` adjacency. It does not
apply to `Molecule`, and a stored `SystemBond` does not activate an attachment or
remove a leaving atom.

~~~python
system.atom(site: AtomSite) -> ChemicalAtom
system.iter_atoms() -> Iterator[tuple[AtomSite, ChemicalAtom]]
~~~

`atom()` resolves a declared atom that survives this one retained-atom predicate.
`iter_atoms()` follows instance insertion order, component order, and atom
declaration order while applying the same predicate. Both are local, uncached, and
non-mutating. Callers filter `ChemicalAtom.element` when they do not want hydrogen;
there is no `include_hydrogens` mode or eager `sites()` materialization.

Incident-bond removal, new polymer bond records, reference-coordinate filtering,
atom mapping, stereochemistry, cyclic closure, and complete graph traversal remain
reopened.

#### 3.4.3 Topology materialization

Status: reopened — per the owner review, `iter_bonds()` cannot freeze as a complete-graph
accessor. Two completeness gaps remain unresolved: it does not dedup a duplicate physical
edge that appears across owner layers (an assembly overlay duplicating an intrinsic
`component`/`molecule`/`polymer_rule` edge), and it lifts no intrinsic stereo onto assembly
`AtomSite`s, so a consumer needing stereo must still read the owner
`ChemicalBond`/`MoleculeBond` declaration — an accessor that claims completeness while
sending consumers back to the owner declaration for missing facts cannot be frozen. The
committed `molecular.py` `SystemBond`/`BondOrigin` shape, the io read and write paths
(`io/structure.py:1177`, `:1597`) that already call `system.iter_bonds()`, and the
single-covalent-edge accessor with its four edge classes below are the endorsed candidate
direction, but the accessor stays an investigation contract. Assembly-level stereo lifting,
the cross-component aromatic/stereo/dative field domain, the cross-layer at-most-once
physical-pair invariant, and cyclic closure also stay reopened (see "Stays reopened"
below).

`SystemBond` is one resolved edge between exact `AtomSite` endpoints. Its origin is one
of the five members:

~~~python
BondOrigin = Literal[
    "component", "molecule", "polymer_rule", "source", "user",
]
~~~

These map exactly to the sources `iter_bonds()` can emit: the three intrinsic classes
(`component`, `molecule`, `polymer_rule`) and the stored assembly overlays (`source`,
`user`). Geometry proximity is not covalent evidence and never creates a `SystemBond`, so
no geometry-inferred origin exists; `iter_bonds()` passes through whatever origin an
overlay stored.

`MolecularSystem.bonds` stores ONLY explicit assembly overlays (`source`, `user`, or a
stored origin); it never duplicates intrinsic component, `MoleculeBond`, or default
linear-polymer edges. A known source edge over a partial component is stored as a
`source` overlay; `ChemicalComponent.bonds=None` does not erase known source
connectivity. Persistence round-trips exactly this stored overlay set — never a
materialized graph.

The per-owner-layer covalent-edge projection is obtained on demand from one uncached,
non-mutating accessor added to committed `molecular.py`, parallel to `iter_atoms()` — a
non-deduplicated stream, not a completed graph (cross-layer duplicate-edge dedup and
intrinsic-stereo lifting stay reopened, per the Status above):

~~~python
def iter_bonds(self) -> Iterator[SystemBond]: ...
~~~

It yields a `SystemBond` for each of four edge classes:

- a retained `ChemicalBond` on component `i` of an instance → a `component` edge over
  `(inst,i,atom_a)`–`(inst,i,atom_b)`, carrying its `order`/`aromatic`;
- a `MoleculeBond` on a `Molecule` instance → a `molecule` edge, carrying its
  `order`/`aromatic`;
- an active adjacent polymer pair (the §3.4.2 predicate: left declares `next_attachment`,
  right declares `previous_attachment`) → one `polymer_rule` edge between
  `left.next_attachment.atom_id` and `right.previous_attachment.atom_id`, with
  `order = 1.0` and `aromatic = False` (attachments carry no order or aromaticity, §3.2);
- each stored overlay in `self.bonds`, yielded unchanged with its stored
  `origin`/`order`/`aromatic`/`stereo`/`stereo_sites`.

An intrinsic edge is materialized only when **both endpoints are realized atoms** under
the frozen §3.4.2 retained-atom predicate — an edge incident to a hidden leaving atom is
not yielded (the only incident-bond removal this candidate carries). `iter_bonds()` reuses the exact
predicate that `iter_atoms()`/`atom()` implement, so the two are coupled by construction
and a future incident-bond-removal refinement moves both together. For the three
intrinsic classes it sets `stereo`/`stereo_sites` to `None` (lifting an intrinsic stereo
descriptor onto assembly `AtomSite`s is reopened; a consumer needing it reads the
`ChemicalBond`/`MoleculeBond` declaration directly). Stored overlays pass through
unlifted.

`iter_bonds()` preserves each edge's supplied orientation, performs no canonical
reorientation, and emits each owner-layer edge **exactly once per owner layer**; it does
**not** dedup one physical endpoint pair across the four classes. Consumers apply their
own selection as ordinary inline post-filters over the stream (io-read keys by endpoint
pair; io-write reorients and sorts; token-feature Processors skip backbone or exclude
cyclic links) — none of which belong in the accessor. An importer or caller supplies the
required orientation; an algorithm that needs an undirected comparison or a stable
serialized orientation performs it locally. There is no custom bond equality or hashing,
constructor endpoint rewrite, provider, or `same_component_chemistry` policy. Partial
components remain traversable as partial known graphs; completeness is a consumer
requirement, not a fabricated global fact.

A monomer, homomer, heteromer, protein-ligand system, protein-nucleic-acid system, and
covalent assembly are all `MolecularSystem` values.

Stays reopened:

- atom and bond stereo descriptor semantics, assembly-level stereo-site lifting for
  `component`/`molecule` edges, neighbor ordering, dative direction, and round-trip
  fixtures; the aromatic, stereochemical, and dative field domain on `MoleculeBond` and
  overlays;
- the cross-layer at-most-once physical-pair invariant and construction-time rejection of
  an overlay that duplicates an intrinsic edge (`iter_bonds()` does not dedup across
  classes and committed `__init__` does not enforce it);
- general incident-bond removal beyond the candidate's "edge with a non-realizable endpoint is
  not materialized";
- cyclic-closure materialization and its interaction with the terminal edit, including
  exact protein, RNA, DNA, hybrid, and PNA behavior;
- the cross-component bond domain beyond intrinsic `MoleculeBond`, and full `SystemBond`
  chemistry and equality;
- hybrid and PNA family lowering, and Structure/source identity and PDB/mmCIF I/O over
  partial components and branch declarations.

### 3.5 Structures and source identities

Status: frozen — the earlier `atom_layout`/complete-chemistry/round-trip design was
falsified and is gone, and the Structure and Structure I/O contract that replaced it is
now accepted per the owner review. It matches the committed `ofoldx/data/structure.py`,
`ofoldx/data/_codec.py`, and `ofoldx/io/structure.py`. §3.5.1 freezes the bounded
`Structure` value carrier; §3.5.2 freezes the import-classification and heavy-atom
data-model boundary; §3.5.3 freezes the read/exchange-write/loss-report I/O surface
(`read_structure`, `read_structures`, `write_structure`, `LossReport`, the PDB and mmCIF
parsers, altloc/model/assembly materialization, the mmCIF exchange writer, and the
importer's source-category derivation). Together they resolve the source-atom-identity,
microheterogeneity, altloc, assembly, partial-observation, site-materialization, and
lossy-write questions the falsified design left open.

The single remaining open sub-point is not an I/O question. It is the codec's
system-ordering/persistence facet — whether `MolecularSystem.instances` serializes as an
ordered `(instance_id, entity_id)` entry array rather than a plain JSON object whose keys
the exchange writer's recursive `json.dumps(..., sort_keys=True)` reorders (§3.4.1, "Open
questions"). That facet governs only instance *declaration order* carried through the
`_ofoldx_structure` payload, not the read/exchange-write/`LossReport` surface frozen in
§3.5.3. Git history preserves the discarded proposal.

#### 3.5.1 One canonical coordinate realization

Status: frozen — the committed `ofoldx/data/structure.py` carrier (the six-field
`Structure` plus `ResidueNumber`) is accepted per the owner review. This subsection freezes
that bounded value carrier. It is supported by PDBx atom-site identity, ModelCIF model and
QA ownership, AlphaFold 3 ligand-template occupancy, and the decoded coordinate outputs of
AF2, AF3, ESMFold2, Protenix, Boltz, RFdiffusion, ODesign, and LigandMPNN. The codec
(`encode_structure`/`decode_structure`) and the exchange writer that consume this carrier
are now frozen as well: the structure-payload JSON shape and the mmCIF numbering-axis
selection are settled in §3.5.3, and parsing, exchange-identity writing,
altloc/model/assembly materialization, lossy-write reporting, and the importer's
source-category derivation are frozen there too. The one facet that stays open is the
codec's *system-ordering* persistence — the `MolecularSystem.instances` ordered-entry-array
shape owned by §3.4.1 — which fixes instance declaration order but changes neither this
carrier nor the §3.5.3 read/write/`LossReport` surface (see "Frozen by §3.5.3; one facet
stays open" below).

A Structure is one coordinate realization of one already-materialized
MolecularSystem. The one bounded source fact required by released MPNN families is
an instance-local residue number:

~~~python
@dataclass(frozen=True)
class ResidueNumber:
    number: int
    insertion_code: str = ""

class Structure:
    system: MolecularSystem
    sites: tuple[AtomSite, ...]
    coordinates: NDArray
    coordinate_mask: NDArray[np.bool_]
    occupancy: NDArray | None
    residue_numbers: dict[ComponentReference, ResidueNumber] | None

    def __init__(
        self,
        system: MolecularSystem,
        sites: tuple[AtomSite, ...],
        coordinates: NDArray,
        coordinate_mask: NDArray[np.bool_],
        occupancy: NDArray | None = None,
        residue_numbers: dict[ComponentReference, ResidueNumber] | None = None,
    ) -> None: ...
~~~

`sites` is the complete and only row axis of the three arrays. Its order is
caller-owned and need not equal system traversal order. Every site is unique and
resolves through `system.atom(site)`. The axis may omit atoms from the system and
may include rows whose coordinate mask is false; those two representations of
missing coordinates have no distinct canonical meaning.

`residue_numbers` maps exact polymer component occurrences to their source integer
number and insertion code. `ResidueNumber.number` means only the integer
residue-number axis a specific producer chose for a specific path; the general
mmCIF label-vs-auth axis decision is not this carrier's to make and is settled by the
importer in §3.5.3 (author number when present, else label).
It belongs to Structure because two instances of one entity may carry different
source numbering; coordinate-free chemistry does not. `None` means that no
source-number facts are supplied and does not mean numbering starts at one. A
supplied mapping is authoritative: a family that needs a polymer occurrence
accesses it directly and a missing key fails naturally when the authoritative
residue number is missing, rather than mixing source and fabricated dense
numbering. Number and insertion-code values are retained exactly, with no range,
width, sorting, or normalization policy. The mapping is not atom serials, record
categories, model or altloc identity, a source bag, or a general PDBx round-trip
claim.

Coordinates are in Å, have shape `(N, 3)`, and have a real non-boolean NumPy
numerical dtype. `coordinate_mask` has shape `(N,)` and dtype exactly `bool`. A
true row carries a coordinate in this realization. A false row carries no
coordinate meaning, regardless of its stored numerical values. At least one row
is true; a coordinate-free value is a MolecularSystem rather than an all-false
Structure.

Occupancy is the only atom-row annotation currently justified by a released
family: AlphaFold 3 ligand templates consume it directly. When supplied, it has
shape `(N,)`, a real non-boolean numerical dtype, and aligns with `sites`. `None`
means that the carrier supplies no occupancy facts. NaN may represent an unknown
value on an individual row. The carrier imposes no range or finiteness policy; an
importer applies source-format defaults and a family Processor enforces only the
occupancy facts that operation requires.

Structure retains `system`, `sites`, every array, and `residue_numbers` by
reference. It does not coerce dtype, copy, make arrays contiguous or read-only, or
move data. Explicit user dtypes are obeyed. Later caller mutation may invalidate
construction-time alignment or closure, and the caller owns that mutation.

Construction checks only row shapes, coordinate and occupancy numerical kinds,
the boolean mask dtype, site uniqueness, site resolution, and that at least one
coordinate is present. It does not scan, complete, copy, normalize, or validate
`residue_numbers`; producer and consuming-family contracts own its coverage. It
performs no coordinate or occupancy finiteness scan, range scan, normalization,
defensive copy, or input-type preflight that ordinary attribute access already
supplies. Structure has object-identity equality; it has no NumPy-aware custom
equality or hashing.

A decoder explicitly constructs its output system and row axis. It marks a row
true only for a coordinate produced or deliberately preserved by that one
operation and that one sample. It may retain unchanged input coordinates in an
operation such as packing, but it does not merge coordinates from different
stochastic predictions, fill missing atoms from reference conformers, or infer
coordinates from chemistry merely to claim a complete layout. A false row is
sufficient when fixed tensor alignment is useful.

`sites` and `coordinate_mask` already state materialization, so the following are
explicitly removed and must not be reintroduced as a `Structure` field without a proven
released-family *input* consumer:

| Removed | Rationale |
|---|---|
| `atom_layout` | `sites` + `coordinate_mask` already state materialization; a mask-false row and an omitted row have no distinct canonical meaning. The `observed`/`complete_heavy`/`complete_all` knob is an I/O element-filter choice, not a carrier fact. |
| `b_factor` | No audited family consumes a B-factor **input**. AF3/Boltz/Protenix use the B column only as an *export encoding* for predicted confidence; ModelCIF owns separate QA categories. Confidence stays a Result Score or Metric — it is not a canonical B-factor merely because a PDB or mmCIF exporter encodes it in that column. |
| `altloc_ids` | One source altloc realization becomes one Structure; altloc materialization belongs to the §3.5.3 I/O reader (`read_structure` selects one residue-consistent altloc; `read_structures` expands the combinations), not the coordinate carrier. |
| `source_identities: tuple[SourceAtomIdentity, ...]` **and the `SourceAtomIdentity` class** | The broad per-atom label/auth asym/seq/comp/atom/group bag is rejected. The one independently evidenced source fact is instance-local residue numbering, now carried by `residue_numbers`; `ComponentReference` already names the instance. |

One source file may contain several models, altloc realizations, or assemblies.
Materializing those values and retaining exchange identities belong to the §3.5.3
Structure I/O contract, not to this carrier. They do not turn one canonical Structure
into an ensemble or a source-file container.

Frozen by §3.5.3; one facet stays open. The questions this list previously deferred are
now resolved by the §3.5.3 I/O contract: the importer source-category derivation (which
rows populate `residue_numbers`, the author-else-label axis, and the numberless-row rule);
altloc, model, and assembly materialization; the `atom_layout` read-time
site-materialization modes; and exchange writing plus `LossReport` reporting (the
`_ofoldx_structure` JSON-in-mmCIF payload, `_decode_canonical_mmcif`, and the placeholder
B/altloc export columns). What stays deliberately out of scope is unchanged: no byte- or
source-exact PDBx/PDB round-trip is ever guaranteed (only the `_ofoldx_structure` payload
round-trips canonically); a broad per-atom source-identity provenance bag stays rejected
(the removed `SourceAtomIdentity` above); non-integer `auth_seq_id` and any general
`ResidueId(asym_id, sequence_id, insertion_code)` value stay rejected; and per-family path
lowering stays with each family. The single genuinely-open item is the codec's
system-ordering/persistence facet — serializing `MolecularSystem.instances` as an ordered
`(instance_id, entity_id)` entry array so the exchange writer's recursive `sort_keys=True`
cannot erase instance declaration order (§3.4.1, "Open questions"). It bounds instance
ordinals inside the decoded system, not the §3.5.3 read/write/`LossReport` surface.

#### 3.5.2 Import classification and heavy-atom realization

Status: frozen — the committed `molecular.py` `iter_atoms()` predicate, the released
family I/O paths, and the REVIEW.md owner review freeze the data-model boundary:
heavy-atom realization is an inline element filter, and polymer classification on import
is producer/source-driven. The exact source parser categories are frozen in §3.5.3.

Heavy-atom realization is an inline element filter, not an API. There is no
`MolecularSystem.sites(include_hydrogens=…)` and no `atom_layout`-driven eager site
materialization. A consumer wanting a hydrogen-free axis filters inline over
`iter_atoms()`:

~~~python
heavy_sites = {site for site, atom in system.iter_atoms() if atom.element != "H"}
~~~

`iter_atoms()` already applies the §3.4.2 retained-atom predicate, so this also excludes
hidden leaving atoms; the consumer adds only the tests it needs (`element != "H"` plus
any per-family name exclusion such as `OXT`/`OP3`). There is no separate heavy-atom,
complete-heavy, or complete-all traversal.

Polymer classification on import is producer/source-driven, not chemistry-driven.
`ChemicalComponent` carries no polymer-membership fact (§3.2), and the package exposes no
reusable component- or module-level polymer-classification helper.
`PolymerSequence.polymer_type` is a fact *declared by the producer* that constructs the
sequence; the primary import path takes it from the producer's own source declaration,
not from the chemistry of its component occurrences. When that declaration is absent, an
importer may fall back to a format-local, component-ID-keyed heuristic (AF3
`guess_polymer_type`; OFoldX `io/structure.py:_component_polymer_type`). The frozen
boundary is only that `ChemicalComponent` carries no polymer-membership fact and no
*public* classifier is exposed — never that a component ID is unusable inside an
importer. The codec serializes `PolymerSequence.polymer_type` but emits and requires
no `ChemicalComponent.polymer_types` key.

Frozen in §3.5.3: the exact source categories an importer consults (mmCIF
`label_asym_id`/`label_seq_id`/`label_entity_id`/`*_scheme` vs. PDB `ATOM`/`HETATM`) — the
data-model boundary is frozen here, and the concrete classification logic is frozen with
the §3.5.3 I/O contract. Any `polymer_type` recovery heuristic for a bare PDB without a
producer declaration stays importer-internal and non-public
(`io/structure.py:_component_polymer_type`), never a reusable public helper.

#### 3.5.3 Structure I/O: read, exchange write, and loss reporting

Status: frozen — the committed `ofoldx/io/structure.py` (`read_structure`,
`read_structures`, `write_structure`, `LossReport`, the PDB/mmCIF parsers, altloc
selection/expansion, the mmCIF exchange writer, and the source-category derivation) and the
`ofoldx/data/_codec.py` structure-payload shape are accepted per the owner review. The
public surface exported from `ofoldx/io/__init__.py` is `read_structure`,
`read_structures`, `write_structure`, and `LossReport`. Format is chosen by suffix: `.pdb`
is PDB, `.cif`/`.mmcif` are mmCIF (`_FORMATS`); any other suffix is rejected. Reading and
writing go through Biotite; the domain values are the §3.5.1 `Structure` over the §3.4.1
`MolecularSystem`. This resolves each question §3.5.1 previously left reopened, as follows.

**Source-atom identity on read.** A parser lowers each source atom record to an internal
`_AtomRow` that retains both the label and the author namespace separately
(`_rows_from_atom_array`): `label_asym_id`/`label_seq_id`/`label_entity_id` and
`auth_asym_id`/`auth_seq_id`/`auth_comp_id`/`auth_atom_id`, plus element, formal charge,
occupancy, altloc, insertion code, assembly symmetry ID, and the raw atom serial. mmCIF is
read with `use_author_fields=False`, so the label namespace is primary and the author
fields are captured as extra columns; PDB has one namespace, mirrored into both. The
canonical grouping axis is the label chain (`label_asym_id`, else Biotite `chain_id`, else
`"_"`) and the label residue number (`label_seq_id`, else Biotite `res_id`). This bounded
capture is not a general per-atom provenance bag: it feeds grouping, `residue_numbers`, and
bond endpoint resolution only, consistent with §3.5.1's rejection of `SourceAtomIdentity`.

**Source-category derivation.** The importer partitions each chain into at most one polymer
occurrence-run plus its nonpolymer occurrences using the importer-internal,
component-ID-keyed classifier `_component_polymer_type`: an audited protein component is
`protein`, an audited nucleic component is `dna` when its ID starts with `D` else `rna`,
and everything else is a non-polymer. This is the §3.5.2 format-local fallback (no producer
`_entity_poly` declaration is consulted), and it stays non-public. A chain's polymer run
becomes one `PolymerSequence` instance (entity ID `polymer_{label_entity_id}` for mmCIF,
`polymer_{chain_id}` otherwise); each remaining residue becomes a single-component
`Molecule`, except mmCIF branched entities (`pdbx_entity_branch_list`/`_branch_link`/
`pdbx_branch_scheme`), which reassemble into one multi-component `Molecule` with its
declared cross-component bonds and leaving-atom removals (`_read_mmcif_branch_entities`).
Groups are emitted in first-source-row order; equal entity definitions under one source
entity ID are shared and conflicting ones are rejected (`_reuse_entity`). `residue_numbers`
is populated for polymer occurrences only: the value is the author residue number when
present, else the label number (`ResidueNumber(number, insertion_code)`); a numberless
polymer row contributes no key, and non-polymer occurrences never contribute one. Every
selected row's element (and, when the source states it, formal charge) is checked against
its resolved canonical atom, so identity disagreement fails loudly rather than silently
mislabeling.

**Altloc, model, and assembly materialization.** `read_structure(model=1, altloc=…)`
materializes exactly one model and one residue-consistent alternate-location choice:
`highest_occupancy` (default) picks, per residue, the altloc label with the greatest
summed occupancy; `first` picks the first-seen label (`_select_altloc`). `read_structures`
instead expands every residue-consistent combination as the Cartesian product over each
residue's altloc labels, returning a tuple of `Structure` values (`_expand_altlocs`); with
no altlocs it returns the single realization. Both refuse a source with no atoms for the
requested model/assembly. `assembly_id` selects a Biotite assembly; symmetry copies carry a
distinct `sym_id`, which becomes a distinct instance (`chain_id#symmetry_id`), so an
assembly materializes its copies as separate instances rather than collapsing them.

**Site materialization (`atom_layout`).** `observed` (default) uses exactly the selected
source rows as the site axis. `complete_heavy` uses every non-hydrogen atom of the resolved
system, and `complete_all` uses every atom after materializing component hydrogens
(`_materialize_component_hydrogens`); rows with no source coordinate are mask-false. This is
the §3.5.2 read-time element/completeness choice realized over `iter_atoms()`, not a carrier
field or a system-level API.

**Source bonds.** Inter-residue source bonds are read and overlaid onto the intrinsic
component/polymer topology (`_read_source_bonds`): PDB `CONECT` records (multiplicity →
bond order, `_read_pdb_bonds`) and mmCIF `struct_conn` `covale`/`disulf` links in the
`1_555` frame, resolved through either the label or the author endpoint key
(`_read_mmcif_bonds`). A source bond that merely restates intrinsic topology is dropped;
one that conflicts with it is rejected; endpoints absent from the selected assembly/model
fail. Bonds are stored with `origin="source"`.

**Exchange write — numbering-axis selection.** mmCIF is the lossless exchange format.
`_write_mmcif` emits one `atom_site` loop carrying both namespaces plus canonical identity
columns. The label axis is a synthesized dense axis: `label_asym_id` is the (deduplicated)
instance ID, `label_seq_id` is `component_index + 1` for polymer atoms and `.` for
non-polymer atoms, and `label_entity_id` is the entity ID. The author axis carries source
numbering where known: `auth_asym_id` mirrors `label_asym_id`, and `auth_seq_id` is the
`residue_numbers` value for that occurrence, falling back to `component_index + 1` when
none is supplied; `pdbx_PDB_ins_code` is the stored insertion code or `?`. Every row is a
single conformer of a single model (`label_alt_id` = `.`, `pdbx_PDB_model_num` = 1), and
`B_iso_or_equiv` is the placeholder `.` — §3.5.1 removed the B-factor carrier, so no
confidence is exported in that column. Four custom columns
(`ofoldx_instance_id`/`ofoldx_component_index`/`ofoldx_atom_id`/`ofoldx_coordinate_mask`)
carry the exact canonical `AtomSite` identity and the coordinate mask, and a sibling
`ofoldx_structure` category carries the full codec payload as one JSON string; source bonds
plus derived topology are written to `struct_conn` with both endpoint namespaces
(`_output_bonds`). On read-back, `_parse_mmcif` detects the `ofoldx_structure` category
(then requires model 1 and no assembly), and `_decode_canonical_mmcif` rebuilds the arrays
from the `atom_site` columns keyed by the payload's array-reference names, decodes the
system, and verifies that every `atom_site` row's canonical identity and element match the
payload — so an OFoldX-written mmCIF round-trips exactly, while an ordinary PDBx/PDB file
round-trips only up to the loss the writer reports.

**Lossy-write reporting.** `LossReport(dropped_fields: tuple[str, ...],
rewritten_fields: tuple[str, ...])` is the machine-readable loss record. `write_structure`
returns `None` unless `allow_lossy=True`, in which case it returns the report. mmCIF is
always lossless (it returns an empty `LossReport`). PDB is guarded: `_pdb_loss_report`
computes the dropped/rewritten field tokens, and when any are present and `allow_lossy` is
false the write raises with the sorted token set rather than silently degrading. The PDB
tokens are a fixed, sorted vocabulary derived from the committed carrier and dict-based
system — e.g. `system.instances`, `system.polymer_sequence`, `system.bonds`,
`system.bond_origins`, `system.bond_semantics`, `system.nonpolymer_graph`,
`system.nonstandard_components`, `system.formal_charge`, `system.atom_ids`,
`system.component_ids`, `system.instance_ids`, `structure.coordinate_mask`,
`structure.coordinates`, `structure.occupancy`, `structure.author_residue_ids`, and
`structure.insertion_codes` — and PDB field-range overflows (serial, coordinate, chain
count, atom-name collisions) raise outright.

**Codec structure-payload shape (frozen) and the one open facet.** `encode_structure`
returns `({"kind":"structure", "system": …, "sites":[{instance_id, component_index,
atom_id}…], "coordinates": ref, "coordinate_mask": ref, "occupancy": ref | null,
"residue_numbers": [{instance_id, component_index, number, insertion_code}…] | null},
arrays)`; the coordinate/mask/occupancy arrays are referenced by name and stored beside the
payload (in mmCIF, as the `atom_site` columns), and `decode_structure` is its exact
inverse. That structure/site/`residue_numbers`/array-reference shape is frozen. The one
part not frozen here is inherited from §3.4.1: `encode_system` currently serializes
`instances` as a plain JSON object, and the exchange writer's recursive
`json.dumps(payload, sort_keys=True, …)` (`ofoldx/io/structure.py:_write_mmcif`) reorders
its keys, so instance declaration order is not preserved across an `_ofoldx_structure`
round-trip. Freezing that — serializing `instances` as an ordered `(instance_id,
entity_id)` entry array read back in order — is the sole remaining open sub-point (§3.4.1,
"Open questions"). It changes only instance ordinals inside the decoded system; the read,
exchange-write, and `LossReport` surface above is frozen independent of it.

### 3.6 Evidence

Status: frozen — parser behavior audited against MultiMolecule
`read_fasta_records`, HH-suite `43095e46`, OpenFold `42e71db`, Boltz `c46ac60`,
ESM `ba4d712`, and Protenix `c3bfc365`. The entity-identity audit resolved to
`entity_id` / `target_entity_id` string binding that closes over the primary
`MolecularSystem` (evidence holds no `PolymerSequence` object and no registry), with
`MSA` carrying separate unpaired and paired pools and `MSACollection` using dense
`int64` / `bool` plan arrays. The `evidence.py` and `read_msa` implementations
conform by construction and numeric parity (ESMFold2 lowered-feature parity is
byte-identical; the Protenix and consumer migrations are faithful re-expressions).

MSA, MSACollection, and Template are ordinary evidence records supplied beside the
primary input. Their explicit entity and instance IDs close over that primary
`MolecularSystem`; they do not retain Python object identity or query a registry:

~~~python
MSA(
    entity_id: str,
    unpaired_sequences: tuple[str, ...],
    unpaired_deletion_counts: NDArray[np.uint32],
    paired_sequences: tuple[str, ...],
    paired_deletion_counts: NDArray[np.uint32],
)
MSACollection(
    msas: tuple[MSA, ...],
    row_pairing: NDArray[np.int64],
    row_is_paired: NDArray[np.bool_],
    target_instance_ids: tuple[str, ...] | None = None,
)
TemplateAlignment(
    target_entity_id: str,
    target_instance_id: str | None,
    template_instance_id: str,
    residue_pairs: tuple[tuple[int, int], ...],
)
Template(
    structure: Structure,
    alignments: tuple[TemplateAlignment, ...],
)
~~~

An MSA is bound to one entity ID in the primary `MolecularSystem`. It retains the
complete unpaired and paired source pools separately, including rows not selected
by a later assembly plan. Both pools are projected to query width; insertions are
removed and their deletion-count arrays store counts on aligned columns while
trailing insertions are discarded. Row zero of each pool is the ungapped query; a
pool with no non-query rows still contains that query row. Column `i` binds
positionally to the entity's `components[i]`. The Processor resolves the entity and
verifies exact query chemistry; evidence does not duplicate a `PolymerSequence` as
identity.

Aligned cells are uppercase one-letter MSA symbols or the `-` gap, not ChemicalComponent IDs. Ambiguity symbols such as B, Z, U, J, O, X, and nucleic-acid ambiguity are preserved because released families map them differently. A Processor owns that family mapping and verifies the query under the same mapping it uses for model tokens. There is no framework-wide component-to-MSA-symbol normalization and no alphabet field.

Strings are the packed public sequence-row representation; a separate character
tensor or NumPy grid would expose packing without adding meaning.
Each deletion-count value is an ordinary CPU NumPy uint32 array with shape
`(pool rows, entity length)`. Evidence owns no torch device, dtype movement,
immutable snapshot, custom numerical equality, or hidden packed form. Construction
checks row lengths, query agreement, and array shape/dtype without scanning every
matrix cell; family mapping handles actual symbols.

`MSACollection` columns align one-to-one with `msas`. `row_pairing` is the complete
raw, pre-cleanup, pre-crop assembly plan; a non-negative integer selects a source
row and `-1` supplies a gap for that column. `row_is_paired` has the same shape and,
for each populated cell, selects the paired rather than unpaired pool and records
the paired feature semantics. Both are dense CPU arrays rather than Python object
matrices because released plans can contain tens of thousands of rows. Source rows
may be reused or unused; intentional reuse of one biological sequence in both roles
is represented by its explicit presence in both pools. The first plan row selects
query row zero from the paired pool in every column. Keeping both complete pools is
necessary because AlphaFold 3 and Protenix calculate profile/deletion features from
the full unpaired pool before pairing, cleanup, or crop, while Boltz consumes the
assembled mixed rows.

With `target_instance_ids=None`, each column broadcasts by its explicit
`MSA.entity_id` to every instance of that entity. An explicit
`target_instance_ids` tuple binds every column to one exact instance and must agree
with the column's entity ID. This all-or-none mode is required when copies carry
different MSAs and avoids mixed-scope precedence. Equal but independently declared
entities never share evidence merely because their sequences compare equal.
In broadcast mode, column `MSA.entity_id` values are unique. In explicit mode,
`target_instance_ids` are unique. Otherwise two columns would claim the same
target without defining whether their evidence is merged, replaced, or duplicated.

One Template is one coordinate frame. Its alignments bind one or more target entities or exact target instances to polymer instances in that same Structure. residue_pairs are increasing zero-based (target component index, template component index) pairs. Grouping multiple chain alignments under one Template preserves an interface frame without duplicating the Structure.

Search is a separate acquisition operation. Source database, taxonomy, search settings, row labels, and acquisition history belong to that operation's output or logs, not the final model evidence. Processor and standard Pipeline calls do not hide network or database access.

The candidate ordinary file boundary is:

~~~python
read_msa(path, *, entity_id: str) -> MSA
~~~

It uses MultiMolecule sequence-file records and produces an unpaired pool plus a
query-only paired pool. A separate acquisition/pairing operation may supply paired
rows without discarding the unpaired source. The query row establishes width; the Processor later
checks that width and chemistry against the named entity in the primary system.
This signature remains open until the complete evidence API freezes. Search and
pairing operations consume row headers before producing final MSA/MSACollection
evidence; `read_msa` does not retain them. Template release dates, search scores,
database/method metadata, and family-specific force or threshold controls are
acquisition or Condition facts rather than fields on final Template evidence.

The filename selects the alignment grammar. A case-insensitive `.a3m` suffix means
A3M; MultiMolecule's `.fa`, `.fas`, `.fasta`, `.fna`, `.ffn`, `.frn`, and `.faa`
suffixes mean aligned FASTA. Other suffixes fail instead of guessing from content.
The implementation calls MultiMolecule's public `read_fasta_records`, preserving
record order and duplicate rows; it does not restore a second FASTA parser in
OFoldX. Record identifiers and comments are used only to discard standard HH-suite
A3M annotation records (`ss_*`, `sa_*`, `aa_*`, and `*_consensus`) before choosing
the first biological row as query.

Aligned FASTA is rectangular before projection. Letters are uppercased and `.` is
normalized to the ordinary alignment gap `-`. Columns in which the normalized query
is a gap are insertion columns and are removed from every row. A non-gap residue in
such a column increments the deletion count attached to the next retained query
column; a gap does not. Leading insertions attach to column zero and trailing
insertions are discarded. Every row must project to the query's retained width.
Only the Processor, which has the primary `MolecularSystem`, checks that width and
chemistry against the named entity.

In A3M, uppercase letters and `-` emit aligned cells. Lowercase letters are inserted
residues: they are removed and counted on the next emitted cell. `.` is an optional
gap used to align insertion columns: it is removed but never counted. Leading
lowercase runs attach to column zero and trailing runs are discarded. The query may
contain `.` but no lowercase residue or emitted `-`, and every biological row must
project to exactly the query's retained width. Readers never truncate, pad, repair, deduplicate,
crop, interpret taxonomy, or map the family alphabet. They build an ordinary CPU
`np.uint32` deletion matrix and otherwise let MultiMolecule file and decoding errors
propagate. Invalid alignment grammar or width raises `ValueError` without a new
exception hierarchy.

### 3.7 Variable topology and conditions

Status: the placement core is frozen (owner verdicts ratified 2026-07-18;
see the "placement core" subsection immediately below), and §3.7.1–§3.7.4
remain frozen as recorded. §3.7.4 adds the frozen constraint layer
(`SymbolicBond` + `Hotspot` + `Contact`, owner verdicts D1–D3 ratified
2026-07-19). The former generic selection wrappers, universal hard/soft intent,
weight, tolerance, satisfaction, and Condition serialization frameworks stay
rejected. BoltzGen's structure-group positive contract is now ratified/frozen under
§3.7.2 (validated 2026-07-19). The remaining open spatial, adjacency, and
secondary-structure records named in this subsection stay investigation and are not
authorized by this freeze.

#### Placement core

Status: FROZEN — owner verdicts ratified 2026-07-18. Two verdicts fix this core.
The X2 identity model is **placement-keyed**: a globally-unique `placement_id` is
the sole reference target, the outer declaration-group key is pure grouping/order
and is never a reference target, and the realized instance id is 100% family-owned
and surfaces only inside `ComponentReference` values in `placement_map`. The X1
build scope is **X1-B**: BoltzGen's whole-assembly `total_len` realization is built
first, then all four families (RFdiffusion, ODesign, PXDesign, BoltzGen) freeze
together. Design rationale, the four-model comparison, and worked examples A–C are
in `CONTIG_IDENTITY.md`; read it before implementing. This freeze covers the shared
placement records, their identity and reference model, the `placement_map` output
surface, per-family length re-homing, and composition with `inpaint_sequence`. It
does not freeze the open spatial, adjacency, secondary-structure, or `SymbolicBond`
records named later in this subsection.

Naming: the shared core type is `PlacementSpec` (module `ofoldx.data.placement`).
"Contig" is not the core — it is RFdiffusion's projection name (its `contig`-string
DSL, e.g. `10-40/A163-181/10-40`). Each family carries its own envelope over the
shared core (`RFdiffusionPlacements`, `ODesignPlacements`, `BoltzGenPlacements`; PXDesign
consumes the bare `PlacementSpec`), and the `placement_id` / `placement_map` /
`PlacementValue` / `GeneratedSegment` / `ExistingSpan` / `SymbolicResidueSite` vocabulary
is shared and unchanged.

A PlacementSpec is an ordered mapping from a declaration-group label to an ordered mapping
from a placement identity to one placement value:

~~~python
PolymerType        = Literal["protein", "rna", "dna", "dna_rna_hybrid"]
ComponentReference = tuple[str, int]        # (realized_instance_id, component_index); UNCHANGED

@dataclass(frozen=True)
class GeneratedSegment:
    polymer_type: PolymerType
    min_length: int            # min_length == 0 is meaningful
    max_length: int            # min_length == max_length fixes the length
    # NO embedded id — identity is the declaration-mapping KEY (segment_id is dropped)

@dataclass(frozen=True)
class ExistingSpan:
    instance_id: str           # names the SOURCE input system; half-open [start, stop)
    start: int
    stop: int

PlacementValue = GeneratedSegment | ExistingSpan

PlacementSpec = dict[str, dict[str, PlacementValue]]
#         declaration_group -> placement_id (globally unique) -> value
~~~

**Identity.** A placement's identity is its caller-chosen `placement_id`, the inner
mapping key, which is **globally unique across the whole PlacementSpec** and validated as
such at construction (a duplicate inner key anywhere is a construction-time error).
The mapping key, not the `GeneratedSegment` or `ExistingSpan` value, carries
identity, so the same ordinary value may occur more than once without acquiring a
hidden ID. The outer key is a declaration-group and ordering label only: it groups
placements into a declared output instance and fixes declared order, and it is
**never named by any reference**. `ExistingSpan` refers to the primary input system
and copies its half-open source interval `[start, stop)`; `GeneratedSegment`
describes a polymer interval to be realized, and `min_length == 0` is meaningful.
ODesign's ligand `num_atoms` remains a family argument and does not justify a shared
`GeneratedMolecule`.

**Reference model — two fields, one canonical target.** A pre-realization reference
names only `(placement_id, offset)`; `placement_id` is globally unique so the
reference resolves without the outer key:

~~~python
@dataclass(frozen=True)
class SymbolicResidueSite:
    placement_id: str          # the globally-unique inner key; resolves WITHOUT the outer group
    offset: int                # >=0 counts from the placement start, <0 from its realized end

@dataclass(frozen=True)
class SymbolicAtomSite:
    residue: SymbolicResidueSite
    atom_id: str
~~~

This is a **rename** of today's 2-field `(segment_id, offset)`, not a widening; the
`segment_id → placement_id` rename keeps the arity, so every family reference site
migrates 2→2, not 2→3. The offset is relative to the placement; non-negative offsets
count from its start and negative offsets from its realized end, with bounds resolved
against the realized length.

**One correct way (owner's Zen principle).** `placement_id` is the SOLE reference
target and there is exactly one canonical way to express a reference. There is no
alternative addressing: the outer declaration-group key is never a reference target,
no reference carries a composite `(group, placement)` key, and no realized instance
id is ever authored or predicted by a reference. There is no alias, no
positive/negative double-key map, and no parallel `instance_binding`/
`symbolic_residue_alias` table — those exist today only because the current
`SymbolicResidueSite` is used as a dict key rather than a tuple index, and they are
deleted, not migrated, when `placement_map` values become offset-indexed ordered
tuples with native negative indexing. A whole-placement reference is spelled as the
bare `placement_id` (no offset); reusing a `GeneratedSegment` value as an id is
forbidden.

**Output surface.** Declaration order is semantic and is never recovered by sorting
ids. After realization a result exposes:

~~~python
placement_map: dict[str, dict[str, tuple[ComponentReference, ...]]]
#              declaration_group -> placement_id -> realized OUTPUT occurrences, in order
~~~

The outer and inner mappings preserve the declared group and placement order. Every
declared placement appears, including a zero-length placement whose value is an empty
tuple, and every `ComponentReference` identifies a realized **output** occurrence
rather than the `ExistingSpan` source. The realized instance id rides
per-occurrence in `ComponentReference[0]`, so one declared placement may expand into
occurrences spanning more than one realized instance id (symmetry/copy expansion is
expressed natively, with no pre-keying). Resolution is `placement_map[group][placement_id][offset]`
with plain Python negative indexing; because `placement_id` is globally unique, a
reference recovers its group through a trivial derived index
`instance_of: dict[placement_id, declaration_group]` built once from the PlacementSpec, and
needs no outer key of its own. Coverage — "every declared placement appears;
zero-length → `()`" — is frozen. Whether declared placements must also **exhaust**
every output component (a variable, unaddressable tail versus a full partition) was
the one point left open; it is now **resolved to a full partition (exhaustive
realization)**. BoltzGen's `total_len` realization — the witness X1-B was to build
first — was built and validated, and the four-family placement freeze ratified
exhaustive coverage (REVIEW.md, ratified 2026-07-18: `placement_map` adopted 4/4,
exhaustive coverage, realized-id unified to `chain_{n}`).

**Length re-homes per family (false shared invariant).** There is no shared
`length_group`, `chain_lengths`, or total-length field on the core. RFdiffusion's
released `ContigMap.length` constrains the combined inpaint length across inpaint
chains and excludes receptor chains; ODesign owns per-chain exact lengths; BoltzGen
owns a whole-assembly `total_len`. These are three distinct concrete-family operation
arguments, not three spellings of one placement-core invariant, and each is carried by the
family that consumes it, never by the shared core. A family resolves its own segment
lengths from its own length argument at realization.

**Composition with `inpaint_sequence` (disjoint, source-side).** The frozen
`inpaint_sequence` (§3.7.3) addresses **existing** residues by source
`ComponentReference` and is untouched by this model: its namespace is the source
input system, while `placement_id` is the declaration namespace and the realized
`ComponentReference` is the output namespace — all three are disjoint, so no
reference can be confused for another. If one source occurrence is placed more than
once by the PlacementSpec, selecting that source with `inpaint_sequence` masks every
realized placement of it (§3.7.3 semantics are preserved verbatim). Because the outer
key is never a reference target, it cannot collide with source-side addressing — the
reconciliation the rejected handle-keyed draft required does not arise.

**BoltzGen whole-assembly handling (X1-B).** A BoltzGen assembly is a single
declaration group whose inner placements are its chains; the outer key is a pure
grouping label that no reference names, and per-chain lengths are drawn from the
family-owned `total_len` at realization (the core carries no total-length field).
Symmetry ties between placements expand into extra `ComponentReference` occurrences
in a placement's tuple, whose realized ids ride per-occurrence in
`ComponentReference[0]`, so one declared unit becomes many physical copies with no
pre-keying. Because references key on `placement_id` and never on the single outer
key, the degenerate "is the one outer key the caller handle or the one realized id?"
question that defeated the handle-keyed draft cannot be asked here.

RFdiffusion scaffold guidance also consumes block adjacency, not only a secondary-
structure label. BoltzGen structure groups expose geometry only within the same
nonzero group and allow groups to move independently; they are not fixed
coordinates. ODesign hotspot labels include ligand components and are not a
synthetic fixed-distance Contact — that hotspot-vs-Contact split is now frozen in
§3.7.4 (`Hotspot` is a first-class distance-free label; a geometric `Contact` is a
strictly larger separate record). These official capabilities remain mandatory on
their concrete family APIs while their exact contracts are audited; they are not
deleted merely because the shared candidate vocabulary does not represent them.

Generated endpoints also establish a need for a symbolic covalent-bond declaration.
That record — `SymbolicBond` — is now frozen in §3.7.4 (endpoints reference the
frozen placement model, explicit leaving atoms, unordered physical identity,
lowering to `SystemBond`). Exact bonds between existing AtomSite values remain
MolecularSystem topology. A covalent link is not a Condition, and the rejected
public wrapper for requesting one is not restored.

#### 3.7.1 Fixed-topology identity conditions

Status: frozen — ProteinMPNN, LigandMPNN, NA-MPNN, and BoltzGen directly consume
these two biological values. This subsection freezes only their public records;
each concrete Processor owns reference resolution, family alphabet support, and
lowering.

~~~python
@dataclass(frozen=True)
class AllowedResidueIdentities:
    positions: tuple[ComponentReference, ...]
    component_ids: frozenset[str]

@dataclass(frozen=True)
class TiedResidueIdentities:
    groups: tuple[tuple[ComponentReference, ...], ...]
~~~

These are plain immutable records with no `intent`, `weight`, tolerance,
satisfaction method, `__post_init__`, or generic validator. `positions` remains the
ordinary operation argument selecting which residues a design operation changes;
an identity restriction does not also select residues. A concrete Processor
resolves references and rejects a semantic contradiction only when its native
lowering would otherwise accept it and silently change the scientific operation.
These two records are the proven cross-family intersection, not an exhaustive list
of every concrete family argument. In particular, NaMpnn token symmetry permits
DNA and RNA rows to share one sampled token while retaining different chemical
components; it is a direct family-owned `symmetry_groups` argument and cannot be
renamed into exact-identity `TiedResidueIdentities`.

#### 3.7.2 Spatial conditioning is family-owned

Status: frozen — Protenix revision `c3bfc365`, Boltz revision `c46ac60`, their
released checkpoint tables, schemas, featurizers, and samplers disprove a shared
spatial-condition record. This subsection freezes the negative cross-family
contract; it also ratifies the FIRST positive family API — BoltzGen's structure-group
contract (frozen, validated 2026-07-19; see "BoltzGen structure-group positive
contract" below). Each remaining positive family API stays investigation until its
complete Tik.

Protenix's v0.5 constraint checkpoint, Boltz-1, and Boltz-2 do not share one
`Contact`, `AtomDistance`, or `Pocket` invariant. Their endpoint domains, distance
predicates, same-chain support, pocket cardinality, cutoffs, learned-conditioning
channels, and sampler guidance differ. In particular, Protenix supports mixed
atom/component token contacts; Boltz-1 has one unary pocket at a checkpoint-fixed
6 Å meaning; and each Boltz-2 contact or pocket independently owns `force`, which
adds a sampler potential without replacing learned conditioning. Boltz-2 template
`force` and `threshold`, its separate physical-potential switch, and Protenix TFG
are further independent controls.

`protenix-v2` is not the version-two constraint format: that released checkpoint
has constraint embedding disabled. The learned contact, atom-distance, pocket, and
substructure channels belong to the v0.5 constraint checkpoint. A concrete
Processor accepts only the controls consumed by its exact registered pair; it does
not accept a condition that the checkpoint would silently ignore.

Therefore shared data and common fold expose no spatial-condition record or
generic condition bag. Each concrete family owns its exact public records or
ordinary arguments, lowering, potential Modules, and outputs. Similar biological
nouns do not justify a union endpoint or base class when the geometric predicate
differs. No family may lose contact, pocket, atom-distance, force,
template-guidance, TFG, or physical-potential behavior as a consequence of this
subtraction.

**BoltzGen structure-group positive contract (ratified/frozen, validated
2026-07-19).** The first positive family API to freeze under this subsection is
BoltzGen's structure-group conditioning. `BoltzGenStructureGroup{members, group_id}`
assigns its `group_id` to every named member, producing a per-token `structure_group`
vector; the default is `0` (free / ungrouped) and a designed/generated residue cannot
join a nonzero group (its backbone is not fixed, so it cannot anchor distance
conditioning). That per-token vector lowers to the released `token_distance_mask`:
cell `(i, j)` is `1` iff `structure_group[i] > 0` and
`structure_group[i] == structure_group[j]` (distinct nonzero groups move
independently; group `0` is never tied), with an all-ones efficient branch when every
token is group `1`. `reference_space_id` remains per-residue — the released featurizer
does NOT regroup it. This lowering is validated exact-equal (max_abs 0.000) against
the released BoltzGen featurizer (`process_token_features`) and is the real code in
`ofoldx/models/boltzgen/processor.py` (`_structure_group_ids` / `_token_distance_mask`).
This ratification is BoltzGen-only; the negative cross-family contract above is
unchanged and every other family's positive spatial-condition API stays investigation.

#### 3.7.3 RFdiffusion existing-residue sequence inpainting

Status: frozen — RFdiffusion revision
`2d0c003df46b9db41d119321f15403dec3716cd9`, its documented
`contigmap.inpaint_seq` path, and canonical source-occurrence identity support this
bounded Processor argument. This subsection freezes neither partial-diffusion
sequence provision, structure inpainting, secondary-structure specification, nor
the complete PlacementSpec contract.

~~~python
inpaint_sequence: Sequence[ComponentReference] = ()
~~~

Each reference names one existing protein component occurrence in the primary
input whose identity is hidden from the denoiser. Existing-span identities remain
visible unless selected; generated placements remain sequence-masked independently
of this argument.

An empty sequence selects nothing. Selection order does not affect tensor order,
and duplicate references are idempotent rather than rejected. If one source
occurrence is placed more than once by the PlacementSpec, selecting that source masks
every realized placement. A selected source occurrence absent from the realized
output is rejected rather than silently discarded.

The concrete Processor resolves this family-local request directly. There is no
`ResidueSelection` value, generic validator, serialization format, or selection
helper. Upstream `inpaint_str` is an independent coordinate mask. `provide_seq` is
a third operation over realized output rows during partial diffusion and can reveal
sequence on generated rows; it requires its own placement-aware complete-family
contract and is not represented by `inpaint_sequence`.

Further evidence justifies separate investigation of `BindingResidues`,
`NonBindingResidues`, `FixedCoordinates`, and `SecondaryStructure`.
`NonBindingResidues` cannot be collapsed into omission because BoltzGen publishes
an explicit `NOT_BINDING` state. Fixed-coordinate output-frame semantics,
variable-topology references, and the complete secondary-structure/scaffold input
must freeze before those records are implemented. The official family functions
remain required while those public records are open.

There is no public Condition base class, global `Condition` alias, closed runtime
registry, generic validation pass, intent/weight/tolerance protocol, satisfaction
result, or Condition persistence framework. Common design spells its two shared
identity records directly; common fold has no condition bag. A concrete Pipeline
and Processor type the exact family controls they consume; a Processor with no
consumer does not accept and reject generic conditions merely for signature
symmetry.

Ordinary operation positions and family controls remain ordinary arguments. Sample
counts, diffusion schedules, guidance or potential weights, symmetry controls, and
ODesign ligand atom counts do not become Conditions merely because they affect a
generation call. Family-specific potential equations likewise do not justify a
generic intent, weight, tolerance, or satisfaction protocol.

The Processor owns family-specific canonicalization, variable-length realization,
and tensorization because those steps determine the actual ModelInput layout. Any
length sampling consumes caller-controlled run-level randomness; the Model consumes
the already-realized layout and performs the family's tensor computation without
reinterpreting the PlacementSpec. A concrete Pipeline owns its released call multiplicity
and coordinates the concrete family's required random streams without unintended
reseeding. This does not imply that CPU, accelerator, Python, or third-party draws
share one `torch.Generator`. RFdiffusion candidates, ODesign model seeds and
coordinate samples, and BoltzGen batches therefore do not share a universal
`num_candidates` axis. The exact seed/generator-bearing signatures, stream bridge,
and transport of realization metadata through family inputs and outputs remain to
be frozen; neither a generic context bag nor a PreparedInput is implied.

#### 3.7.4 Frozen constraint layer — SymbolicBond, Hotspot, Contact

Status: **FROZEN** — owner verdicts D1–D3 ratified 2026-07-19. This subsection
freezes the *input-side* constraint records that ride on top of the frozen
placement core (`#### Placement core`) and the exact covalent surface
(`ofoldx.data.molecular`): the pre-realization covalent link `SymbolicBond`
(`ofoldx.data.bonds`) and the target-conditioning pair `Hotspot` / `Contact`
(`ofoldx.data.conditions`). All four design families (RFdiffusion, ODesign,
PXDesign, BoltzGen) consume these records; each concrete Processor owns reference
resolution, legality, and lowering. This freeze covers only the coordinate-free
INPUT declaration — it does **not** unify the per-lineage output confidence
metrics (see "Do not unify" below).

**Reference, never mint.** Every endpoint is one of the frozen placement-reference
nouns; no constraint mints a new `placement_id` or names a new reference target, so
the placement freeze's uniqueness and `resolve()` invariants carry the constraints
for free. Two endpoint granularities are shared, both pure re-uses of the placement
vocabulary:

~~~python
# atom-granularity endpoint (ofoldx.data.bonds; re-exported as AtomEndpoint in conditions)
#   AtomSite       — an EXISTING realized atom; its (instance_id, component_index) IS a ComponentReference
#   SymbolicAtomSite — a GENERATED atom over the frozen 2-field SymbolicResidueSite(placement_id, offset)
SymbolicBondSite: TypeAlias = AtomSite | SymbolicAtomSite

# residue-granularity endpoint (ofoldx.data.conditions)
ResidueEndpoint: TypeAlias = ComponentReference | str | SymbolicResidueSite
#   ComponentReference   — an EXISTING realized residue (instance_id, component_index)
#   str                  — a WHOLE generated placement: the bare globally-unique placement_id
#   SymbolicResidueSite  — one GENERATED residue (placement_id, offset)
AtomEndpoint: TypeAlias = AtomSite | SymbolicAtomSite      # == SymbolicBondSite, for atom-reaching families
Endpoint:     TypeAlias = ResidueEndpoint | AtomEndpoint
~~~

The resolution chain is closed by the placement freeze:
`SymbolicAtomSite → SymbolicResidueSite → resolve() → ComponentReference → AtomSite`.

**D1 / D2 — `SymbolicBond` (frozen `ofoldx.data.bonds`).** A covalent link declared
BEFORE realization, whose endpoints may be generated atoms.

~~~python
@dataclass(frozen=True)
class SymbolicBond:
    site_a: SymbolicBondSite
    site_b: SymbolicBondSite
    order: float = 1.0
    aromatic: bool = False
    stereo: str | None = None
    leaving_a: frozenset[str] = frozenset()   # atom_ids deleted from site_a on bond formation
    leaving_b: frozenset[str] = frozenset()   # atom_ids deleted from site_b
~~~

- **D2 — physical identity is the UNORDERED endpoint set** (`endpoints` → the
  `frozenset` of both endpoints), mirroring `SystemBond` (whose identity is enforced
  as a `frozenset` on `MolecularSystem`). `site_a` / `site_b` positional order is
  retained ONLY to anchor per-endpoint leaving-atom attribution — `leaving_a` pairs
  with `site_a`, `leaving_b` with `site_b` — and carries no directional chemistry. A
  bond that connects an atom to itself is a construction-time error.
- **D1 — leaving atoms are EXPLICIT on the bond.** A generated-atom endpoint has no
  finalized `ChemicalComponent` to carry `PolymerAttachment.leaving_atom_ids`, so the
  atoms deleted on bond formation ride explicitly as `leaving_a` / `leaving_b`. This
  is lossless and re-derivable by a family helper later; inference-only could not be
  added back without breaking identity.
- **Chemistry kept** (`order` / `aromatic` / `stereo`) so lowering is lossless —
  ODesign's previously hardcoded `1.0` / `False` becomes explicit rather than lost.
- **No `intent` field.** Hard/soft is family-owned and rides on the family condition
  envelope, exactly as the placement freeze re-homed `intent` / `chain_lengths`.
- **Lowering.** `SymbolicBond.resolve(placement_map, instance_of)` resolves each
  endpoint (via `resolve_bond_site`) and lowers to
  `SystemBond(site_a, site_b, order, aromatic, stereo, origin="user")`. Leaving atoms
  are NOT placed on `SystemBond` (which has no leaving-atom concept); the family
  strips them from the realized components separately.

**D3 — `Hotspot` and `Contact` are two orthogonal records, not one toggle** (frozen
`ofoldx.data.conditions`).

~~~python
@dataclass(frozen=True)
class Hotspot:
    residues: tuple[Endpoint, ...]            # per-residue target label; NO distance; single-sided

@dataclass(frozen=True)
class Contact:
    side_a: tuple[Endpoint, ...]
    side_b: tuple[Endpoint, ...]
    cutoff_angstrom: float                    # identity IS the distance; symmetric in side_a/side_b
    tolerance_angstrom: float = 0.0
~~~

- **`Hotspot` is a first-class distance-free label**, not a degenerate `Contact`. The
  generated partner is implicit (the whole design). All four families already own the
  concept and lower it to a per-token signal (RFdiffusion `hotspot_mask` / PXDesign
  per-token bool / ODesign `token_features[..., 452]` / BoltzGen `binding`). Endpoints
  MAY be non-protein (e.g. an ODesign ligand `ComponentReference` such as `E/505`).
- **`Contact` is the strictly larger two-sided geometric record**, whose identity is a
  real `cutoff_angstrom`. A `Contact` degenerated to one side would still carry a
  cutoff every label consumer ignores — the required-but-meaningless field the freeze
  rejects — so `Hotspot` is the primitive and `Contact` the superset. The constraint
  is symmetric: swapping `side_a` / `side_b` names the same physical contact.
- **Family-owned, kept OFF the frozen core:** `weight` (px/od), directional `kind`
  (BoltzGen `pocket>binder` / `binder>pocket`), the negative `not_binding` label
  (BoltzGen), the allowed cutoff/threshold *values* and their enforcement, hard/soft
  `intent`, and the lowering itself. A family that needs any of these wraps the frozen
  record in its own condition envelope.

**Composition with `inpaint_sequence` (orthogonal).** `inpaint_sequence` (§3.7.3)
addresses EXISTING residues by source `ComponentReference` and selects which sequence
positions are masked; a `Hotspot` / `Contact` / `SymbolicBond` selects which
residues/atoms are conditioned-on or bonded. They can name the same residue with no
conflict — one masks its identity, the other constrains its geometry/bonding.
Constraints are declared pre-realization and materialize post-realization through the
same `resolve(site, placement_map, instance_of)` the placement freeze provides, so a
constraint over a doubly-placed source sees every realized copy identically and
inherits the §3.7.3 "mask/constrain all realized placements" behavior rather than
contradicting it.

**Do not unify (load-bearing).** This freeze constrains only the INPUT spec. It is
NOT license to unify the *output-side* confidence metrics: the per-lineage
`has_clash`, the pTM-with-`d0` variants, and the family post-generation
verifications (ODesign's §B hard-distance check that reads `Contact.cutoff_angstrom` /
`tolerance_angstrom`; BoltzGen's N×N `contact_conditioning` classes) stay per-lineage
faithful to each upstream and remain distinct. The constraint layer freezes the shared
*question* ("the input asks for these residues to be within `cutoff`"); each family's
metric independently answers "did my output satisfy it, by my faithful definition."
A shared `Contact.cutoff_angstrom` is one input field with three faithful family
lowerings (px hard-pins 8 Å, od §B verifier, bg threshold matrix), not evidence the
metrics should merge.

### 3.8 Fixed-length inverse-folding canonical slice

Status: reopened — the current canonical Tok exposed the missing entity/copy
relation, family-specific sampling support, invalid-row context, mixed-tie, and
replacement-dependency questions that
must be resolved against the audited evidence below before this subsection is
frozen again. Evidence: ProteinMPNN
`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, LigandMPNN
`26ec57ac976ade5379920dbd43c7f97a91cf82de`, NA-MPNN
`aafa32c87b65046a02db152f6e6235a7078e5fef`, and the PDBx distinction between
entity identity and deposited-coordinate assembly occurrences. This is the
bounded contract needed to implement their fixed-length canonical inputs and
realizations. Practice or contrary family evidence reopens it before the affected
Tok continues.

The expanded family inventory additionally falsifies one shared
ProteinMPNN/LigandMPNN path lowerer. Original ProteinMPNN inserts transient numeric
gap rows and reorders designed chains before fixed chains; the updated
LigandMPNN-owned protein, soluble, and membrane routes retain physical CA rows and
do not insert those gaps. They also differ in whether X is generatable. Those
behaviors return to concrete family readers and samplers rather than becoming a
parser mode, generic option, or lowest-common-denominator support mask.

The candidate public surface retained from the previous Tik is:

~~~python
PolymerType = Literal["protein", "rna", "dna", "dna_rna_hybrid"]
ComponentReference = tuple[str, int]

PolymerSequence(polymer_type, components)
Molecule(components, bonds=frozenset())
MolecularSystem(entities, instances, bonds=frozenset())
AtomSite(instance_id, component_index, atom_id)

system.component(reference: ComponentReference) -> ChemicalComponent
system.atom(site: AtomSite) -> ChemicalAtom

Structure(
    system,
    sites,
    coordinates,
    coordinate_mask,
    occupancy=None,
    residue_numbers=None,
)
~~~

Only the following chemical-leaf fields remain candidates in this slice because
its resolution and MPNN consumers use them directly:

~~~python
ChemicalAtom.atom_id: str
ChemicalAtom.element: str
PolymerAttachment.atom_id: str
PolymerAttachment.leaving_atom_ids: frozenset[str]
ChemicalComponent.component_id: str
ChemicalComponent.atoms: tuple[ChemicalAtom, ...]
ChemicalComponent.previous_attachment: PolymerAttachment | None
ChemicalComponent.next_attachment: PolymerAttachment | None
MoleculeBond.component_a: int
MoleculeBond.atom_a: str
MoleculeBond.component_b: int
MoleculeBond.atom_b: str
SystemBond.site_a: AtomSite
SystemBond.site_b: AtomSite
SystemBond.stereo_sites: tuple[AtomSite, AtomSite] | None
~~~

`ChemicalComponent.polymer_types` does not exist: polymer membership belongs to a
`PolymerSequence` occurrence. `PolymerAttachment` carries neither `order` nor
`aromatic`; the paired linear rule owns its non-aromatic single bond. These
deletions survived the previous implementation cycle. Other chemistry fields
remain investigation and are neither accepted nor rejected by this slice.
Component atom IDs are unique. Every
attachment endpoint and leaving ID resolves in its component, and its retained
endpoint is not also a leaving atom. A `MoleculeBond` joins two distinct component
occurrences and both local atom endpoints resolve. Within a Molecule or the stored
system-bond overlay, an unordered physical endpoint pair occurs at most once and a
bond never joins an endpoint to itself. Every AtomSite carried by a stored
`SystemBond`, including optional stereo reference sites, resolves through
`system.atom` after the paired linear-adjacency edit. These are local
referential-closure checks, not authorization for the reopened complete chemistry
or unified bond traversal.

The previous instance-only candidate is no longer accepted. The minimal candidate
above uses `MolecularSystem(entities, instances, bonds=frozenset())`, where `instances` maps
instance ID to entity ID as specified in Section 3.4. Same-entity is an explicit
copy relation consumed by released Models, but it is not an implicit design tie.
Dictionary insertion order is declaration order only; a family Processor may
choose another tensor order and then owns the exact ordered `ComponentReference`
mapping. Component indices are non-negative, zero-based positions in the exact
named instance; negative Python indexing is not canonical identity.

`PolymerSequence` declares an ordered sequence of exact component occurrences.
Polymer membership belongs to that occurrence context, never to a reusable
`ChemicalComponent`. Construction does not require every adjacent pair to have
attachments. A default linear adjacency consumes leaving atoms only when the left
component declares `next_attachment` and the right component declares
`previous_attachment`; if either is missing, no default edge or leaving-atom
removal is inferred. Exposed terminal leaving groups remain. A stored `SystemBond`
is only an explicit assembly edge and never consumes an attachment, removes a
leaving group, or triggers another graph transform.

Direct resolution is local and uncached. `component(reference)` returns that exact
stored occurrence. `atom(site)` resolves a declared atom that remains after only
the paired linear-adjacency rule above. This cycle does not publish an eager
`sites()` view, a hydrogen-filtering traversal mode, or a unified `iter_bonds()`;
their full graph and chemistry semantics remain reopened.

The bounded canonical Structure carrier is frozen in Section 3.5.1. This family
slice adds no layout flags, source identities, annotations, array copies, or
stricter numerical policy to it.

No generic public `replace_polymer_components` contract is accepted. A design
decoder must construct the output system explicitly because occurrences of one
input entity may be designed independently. Canonical values do not infer an output
partition from `PolymerSequence` equality. Each family Tik must define which exact
released output facts keep copies in one entity, how new entity IDs are allocated,
and how stored topology remains resolvable. Explicit released ties constrain the
realized values before that partition; same-entity identity alone is not a tie. A
generic helper does not invent entity IDs. Instance IDs, component indices, and
untouched topology remain exact, while the concrete family owns realization of its
output values. NA-MPNN likewise owns the exact ribose or deoxyribose component
chosen for each hybrid position.

For the shared ProteinMPNN/LigandMPNN lowering in this cycle, coordinate validity
and design selection are separate Processor facts. Their Processors accept the
ordinary family argument:

~~~python
positions: Sequence[ComponentReference] | None = None
~~~

Validity belongs to the concrete Processor/Model pair; selection is user intent.
For example, the released full-backbone ProteinMPNN path requires finite N/CA/C/O,
its distinct CA-only path requires finite CA, and LigandMPNN derives validity from
its released atom-presence policy. Those exact policies return to their family Tik
rather than being guessed by this shared lowering.

`None` selects every valid protein row. An explicit sequence selects exactly those
protein occurrences; an empty sequence selects none, and order or duplicates do
not reorder or duplicate tensor rows. Non-protein or selected invalid references
fail rather than silently becoming fixed. A valid nonstandard protein occurrence
remains selectable: ProteinMpnn lowers its input identity to X and may redesign it
to a generated canonical identity. The lowerer receives validity and selection
separately and returns their conjunction as the Model design mask. Every other row
retains its input token. Valid unselected rows and invalid rows remain
native-sequence decoder context; validity still excludes invalid rows from their
structural node/update path.
Identity Conditions do not double as a residue selection language: a singleton
allowed set remains a designable but deterministic sampling position and therefore
does not have fixed-context decode semantics.

Allowed-identity records intersect at the same position. Ties form an equivalence
relation, and restrictions intersect across the resulting group. A fixed member of
a mixed fixed/designable group anchors the identity; conflicting fixed anchors are
a semantic contradiction. The concrete Processor intersects the remaining set with
the exact identities that its released sampler can generate. A singleton allowed
set remains a deterministic design request rather than becoming fixed context.

Everything beyond those relations is family-owned. ProteinMPNN, LigandMPNN, and
NA-MPNN have different alphabets, scoring protocols, sampling controls, PSSM/bias
semantics, and RNG behavior. Every such official function remains required in the
complete concrete family vertical; it does not belong in a speculative shared
inverse-folding framework. The shared lowering does not scan every possible
identity against assembly bonds, precompute topology-preserving alphabets, scan
tensor finiteness, or report Condition satisfaction. A decoder constructs the
sampled canonical output; an actually dangling explicit bond fails there through
ordinary reference resolution.

Immutable chemical and reference leaves may use ordinary value semantics.
`MolecularSystem`, `Structure`, and array-bearing Results claim no scientific deep
equality: their implementation uses ordinary object identity rather than custom
NumPy equality, dict-order normalization, hidden comparison fields, or constructor
mutation.

This reopened slice freezes no complete family tensor lowering. Section 3.7.1
independently authorizes only the two plain identity records. ProteinMPNN,
LigandMPNN, and NA-MPNN each require their complete family Tik before the candidate
selection, tie, alphabet, sampling, scoring, and decode behavior above can enter
Tok. The same is true of complete component stereochemistry, cyclic graph
transforms, full atom/bond traversal, SMILES, PNA, source identity, structure I/O,
and variable-topology placement.

#### 3.8.1 Family-owned MSE token lowering

Status: frozen — audited released parser behavior and direct parser counterexamples
establish this narrow model-token rule independently of the reopened complete
inverse-folding contract.

Canonical chemistry retains the exact `MSE` component identity. The shared MPNN
checkpoint alphabet contains only its native 20 amino-acid tokens and owns no
modified-residue aliases. ProteinMpnn alone lowers `MSE` to its `MET` model token;
LigandMpnn, NaMpnn, and MpnnPacker lower it to their unknown protein token. The
ProteinMPNN source performs this rewrite only for `HETATM` records, but canonical
Structure deliberately has no ATOM/HETATM record-category field, so ProteinMpnn
applies the chemical-identity rule to every canonical `MSE` occurrence.

MpnnPacker parses atom14 identity only for selected canonical 20-amino-acid rows.
Unknown or modified unselected rows remain unchanged context and are never indexed
through the native alphabet. A caller still cannot select `MSE` for packing with
the released checkpoint. This family split is released behavior, not an upstream
bug or a generic modified-residue normalization framework.

#### 3.8.2 Mpnn atom-context water filtering

Status: frozen — the released operations exclude water from atom context, while
ProDy version drift and existing CCD-name collisions falsify either dependency
version's name-only table as an OFoldX scientific contract.

LigandMpnn and MpnnPacker treat a component as water only when its ID is one of
`HOH`, `DOD`, `WAT`, `TIP3`, `H2O`, `OH2`, `TIP`, `TIP2`, `TIP4`, `SPC`, `SOL`, or
`T3P`, its declared atoms include oxygen, and every declared atom has element H or
O. The rule reads `ChemicalComponent.atoms`, not coordinate-present sites, so
occupancy and missing coordinates do not change component identity. It does not
inspect bonds, completeness, hydrogen count, charge, or provenance.

This bounded alias-and-elements rule retains ordinary O-only partial water and
DOD's hydrogen isotope while preserving real CCD components whose colliding IDs
include C, N, P, or S. It is one shared predicate, not a canonical Water type,
generic chemistry classifier, CCD lookup, or ProDy dependency. A partial colliding
component that declares only H/O remains indistinguishable from the corresponding
water-model spelling and follows the explicit importer declaration.

## 4. Model and Processor boundary

### 4.1 Concrete contracts, not family cardinality

Status: reopened — this is an investigation contract. The earlier one-Model,
one-Processor, one-input, and one-output family rule was falsified by released
families with several public computation contracts. Model or checkpoint count does
not determine whether a Pipeline's public contract is standard or concrete.

Each registered Model has one ordinary, explicit raw `forward` contract and one
stable meaning for its raw output. This statement applies to that registered
Model, not to an entire family or repository. It does not imply a universal call
signature, a single positional argument, or framework-wide input and output
classes.

A family, repository, or Pipeline may contain as many public Models,
Processors, and reusable `torch.nn.Module` components as its released scientific
operation requires. Each public expert Module owns its own ordinary signature and
output meaning. No family-wide cardinality or shared-output rule follows from
registration. Which concrete Models and Processors form repository pairs remains
a family inventory question rather than a convention to assume.

`Processor` and `ModelConfig` are collective terms. `ProcessorConfig` is collective
only for Processors with persistent lowering values; a config-less Processor does
not invent an empty config. Every registered Model inherits the minimal public
`ofoldx.Model` lifecycle base, which stores its concrete config and owns backend
selection without defining or hiding its scientific signature. Processors and
Configs do not require nominal bases.

Every concrete Model that is constructible through `AutoModel` has the following
Hub-loadable constructor shape:

~~~python
class ConcreteModel(ofoldx.Model):
    config_class = ConcreteModelConfig

    def __init__(self, config: ConcreteModelConfig | None = None) -> None: ...
~~~

`config=None` means the concrete default config.

#### 4.1.1 Ordinary Model placement

Status: frozen — ordinary PyTorch semantics, ESMFold2's audited mixed release, its
real 6.59-billion-parameter graph, and independent loader review falsified the
former family `.to` override. This slice freezes public Model placement only;
the shared internal repository materialization algorithm is frozen separately in
Section 7.6, while family eligibility and converter artifact dtypes remain
family-owned evidence questions.

After construction or repository hydration, Auto calls the Model's ordinary
`.to(device=..., dtype=...)` only when placement was requested. A config-constructed
Model with omitted dtype retains its constructor dtypes. A repository-loaded Model
with omitted dtype retains every serialized tensor dtype and every non-persistent
constant's ordinary constructor dtype. Omitted dtype means no runtime dtype request;
it does not infer one Model dtype from the first weight, reconcile derived constants
with assigned state, or change the paired Processor's constructor default. An
explicit dtype uses ordinary `nn.Module.to` semantics for all floating and complex
parameters and buffers; a device-only request preserves dtypes. Caller-injected
child Modules have the same ownership semantics. OFoldX does not warn, refuse, or
preserve hidden FP32 exceptions when an experienced caller explicitly requests a
whole-Model dtype; mixed precision uses AMP or the caller's explicit per-module
choices.

Repository serialization preserves configs and persistent Model state, not runtime
placement. It does not snapshot device, Processor dtype, training mode, AMP state,
or prior whole-Model casts of non-persistent constants. A caller who wants one
runtime dtype for a loaded pair passes `dtype=` to `pipeline(...)` or applies
ordinary `.to(...)`; an unrequested incompatible mixture fails in PyTorch.

Artifact mixed precision, family autocast, and state placement are distinct. The
canonical ESMFold2 release stores its head in FP32 and ESMC child in BF16; ordinary
device movement preserves that mixture, while an explicit dtype intentionally
converts the complete Model. ESMFold2 does not override `.to`, and no family
reinterprets a runtime dtype as a checkpoint policy. OFoldX does not thread
placement through every leaf, scan parameters, or publish duplicate cached device
or dtype properties.

Fresh `device="meta"` through post-construction `.to("meta")` does not promise
allocation-free construction; it first follows the concrete Model's ordinary
constructor. A repository uses the internal meta path only after that family's
persistent and non-persistent tensors satisfy the eligibility obligations in
Section 7.6. Peak-memory optimization belongs once in repository loading rather
than in every Model constructor.

### 4.2 Processor

Status: reopened — this is an investigation contract. The biological-to-tensor
boundary is necessary, but task neutrality, blanket determinism, and universal
randomness ownership are not established.

A concrete Processor accepts canonical biological values and the explicit
lowering arguments required by its consumer. One call returns one concrete,
self-contained static and biological input value. Its tensor positions are aligned
with canonical molecular identity inside that value, so a Model task method does
not need the original source object or a hidden side channel to interpret them.

Here static means fixed for one Model task invocation. It does not mean globally
immutable, deterministic across invocations, or shared by every task in a family.

The returned value contains only the static and biological facts established by
processing. A dynamic tensor created by a Model-side algorithm is an explicit
argument of the concrete Model or expert-Module signature that consumes it; it is
not inserted into the processed value, mutable Processor state, or a generic
context bag.

There is no public `PreparedInput`, generic context, mapping-unpack hook, required
`Layout` class or field, or framework tree protocol. A concrete input carries only
the identity associations and other fields that its real consumers require. The
words concrete and typed specify stable public meaning, not representation: they
mandate no dataclass, custom equality or hashing, tree registration, or particular
container. The Processor does not construct a canonical Result.

Concrete Processors accept runtime device and dtype requests at construction and
support ordinary mutable placement through:

~~~python
processor.to(device=..., dtype=...) -> processor
~~~

Tensor outputs are produced on the requested device. Device and runtime dtype are
not `ProcessorConfig` fields, and placement is not inferred by scanning Model
weights. An explicit Processor dtype applies to every floating tensor it emits;
integer, boolean, and identity values retain their semantic types. It is an output
placement request, not an internal-compute or checkpoint policy. OFoldX does not
preserve hidden FP32 output fields on the caller's behalf; an unsuitable explicit
dtype follows ordinary execution and may fail.

A Processor may be deterministic or stochastic according to the released
lowering algorithm. If it draws randomness, that draw and its explicit RNG input
belong to the Processor operation that performs it; randomness is not reassigned
by a framework-wide category.

### 4.3 Concrete inputs and raw outputs

Status: reopened — this is an investigation contract. Concrete typed values have
survived review; a family-wide input/output naming layer and one-per-family
multiplicity have not.

The Processor return is one concrete value that carries the processed tensors and
the canonical identity associations needed by its actual consumers. It is the
self-contained static and biological boundary, not necessarily the sole argument
to raw forward. Model-owned noise, timesteps, sampled states, or other dynamic
tensors appear explicitly in the signature that uses them.

Each registered Model's raw forward returns exactly the concrete value produced by
that computation. Its fields have stable meanings for direct tensor callers and
training; it is never a union or superset of every task and expert output in the
family. A task or expert computation that needs another computation calls an
explicit public Model method or Module with its own signature and output value.

A shared Model task method constructs the concrete decoded task Result for one
Model invocation. A public expert method or Module may instead return the typed raw
output that its scientific callers need. Neither kind of output is padded with the
other kind merely to make a family-wide type.

There is no framework-wide public `decode` method or decoder protocol. A Model
task method already has its processed input and raw values and constructs its
decoded Result directly. A family exposes a separate decoder only when a real
public scientific operation requires one, with its own complete concrete
signature.

### 4.4 Model task methods and operation ownership

Status: reopened — this is an investigation contract. Model-side and Pipeline-side
ownership is evidence-backed, but exact family method signatures remain under
inventory.

A registered Model is a normal `torch.nn.Module`. Its concrete `forward` signature
is the raw tensor API. OFoldX does not override `nn.Module.__call__` to unpack a
mapping, infer a task, move inputs, or choose a decoder. Raw forward inherits the
caller's grad and autocast contexts.

A Model implements only the shared task methods it actually performs and every
released family-specific scientific method that belongs to its checkpoint
contract. A task method owns the Model-side loop for one invocation, its local
inference or gradient policy, semantic decoding, and construction of that
invocation's concrete decoded Result. It calls raw forward or public expert Modules
according to the real algorithm; it does not execute a useless forward to satisfy
a framework hook.

A Pipeline owns the exact number and order of Model task and expert-operation
invocations required by the user-visible operation, composition and handoff across
Models and Processors, and selection across invocations. These responsibilities do
not move a Model's tensor computation into the Pipeline.

RNG ownership follows the draw. The Processor or Model operation that actually
draws consumes and advances its explicit RNG input. There is no assumed single RNG
spanning CPU, accelerator, and non-PyTorch streams. The exact bridge from a public
seed into the streams required by a family, and how a composed Pipeline coordinates
them without restarting a stream, remain open.

Task Results contain scientific output, not unverifiable execution history. An
ordinary mutable Model can change after Hub loading through training,
`load_state_dict`, direct tensor mutation, compilation, or transformation, so a
repository revision cannot identify the weights that produced a later Result.
Likewise one seed cannot describe a stochastic Processor, an advancing Generator,
or a multi-stage operation. Results therefore carry no generic run provenance.
The Hub commit remains transient loader state used to pin one repository snapshot.
If a future experiment-record consumer appears, that work starts from its concrete
requirements rather than attaching lifecycle state to scientific Results now.

ModelConfig contains graph and algorithm values needed to construct the concrete
architecture. When a Processor has persistent values needed to construct its
lowering, they belong to its concrete ProcessorConfig; when it has none, no config
object or empty file is created. Device, runtime dtype, compile or quantization
state, Hub transport, repository source, and Pipeline flow are not config fields.
Generic Result provenance does not exist.

### 4.5 PyTorch transformations and open boundaries

Status: reopened — this is an investigation contract. Ordinary PyTorch
transformation paths are required, while the scope of compilation and precision
policy still require family evidence.

Each concrete raw `forward` is an ordinary eligible framework compilation boundary. OFoldX
adds no generic input wrapper or mapping hook, automatic movement, parameter scan,
or duplicated device/dtype preflight inside forward. Canonical identity metadata
needed outside tensor computation does not drive Python control flow in the tensor
graph. Exact compile guarantees, supported modes, and graph breaks remain a
family-level evidence question.

Any `torchao` transformation supported by a family operates on its ordinary public
PyTorch Model. OFoldX does not infer Processor placement or dtype from transformed
weights, scan parameters to publish duplicate runtime properties, or serialize a
transformed runtime graph. Which transformations preserve each family's scientific
and compilation semantics remains an affirmative family-evidence question; the
existence of an ordinary `nn.Module` is not by itself a blanket support claim.

The following must remain open until family inventories provide affirmative
evidence:

- exact Model–Processor pairings and repository cardinality;
- the seed/generator bridge, including non-PyTorch randomness;
- the raw-forward meaning and arguments of diffusion architectures;
- batched processed-input and canonical-Result semantics; and
- whether Model task methods, rather than their raw tensor kernels, are
  compilation targets.

### 4.6 MpnnPacker side-chain packing vertical

Status: frozen — an audit of the released Packer using upstream revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de` as reproducibility evidence, the
`ligandmpnn_sc_v_32_002_16.pt` checkpoint, official cross-designer workflow, and
independent callable-axis and numerical counterreview support this complete
standalone Model, shared MPNN PDB reader, family-owned writer, Processor, and
Result contract. This
section freezes neither a generic PDB/mmCIF reader nor any sequence designer. The
four explicit design-to-pack compositions that consume the Packer are frozen in
Sections 4.7.1, 4.8.1, 4.9.1, and 4.9.2. The inspected revision identifies the audit;
it is not an OFoldX dependency or public-behavior pin.

The upstream CLI loads `Packer` and its checkpoint independently from the selected
sequence Model. It then feeds the Packer sequences produced by ProteinMPNN,
LigandMPNN, per-residue membrane MPNN, global membrane MPNN, or soluble MPNN. The
Packer graph has no dependency on a LigandMPNN Model or model-type branch. It is
therefore the independent `MpnnPacker` family in Python, with package
`ofoldx.models.mpnn_packer` and architecture identity `ofoldx.mpnn_packer`; it is
not a LigandMPNN child or mode. The upstream repository and checkpoint names remain
provenance rather than Python ownership.

The released side-chain packer is one independent Model–Processor pair. It is
registered only after the complete family is implemented, and it does
not require a composition manifest when called alone. The identical released PDB
parse route also feeds LigandMpnn and the membrane MPNN variants, so it belongs to
the existing public MPNN module rather than to Packer ownership or a rejected
generic structure reader:

~~~python
# ofoldx.io
def read_mpnn_pdb(
    path: str | PathLike[str],
    *,
    chains: Sequence[str] | None = None,
) -> Structure: ...
~~~

The Processor accepts the common input union and delegates only its path branch:

~~~python
def __call__(
    self,
    input: Structure | str | PathLike[str],
    *,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    positions: Sequence[ComponentReference] | None = None,
    use_atom_context: bool = True,
    filter_by_occupancy: bool = True,
) -> MpnnPackerInput: ...
~~~

`processor(path, ...)` is exactly `processor(read_mpnn_pdb(path), ...)`; no path-only
payload survives canonical construction. The reader interprets plain PDB text and
a `.gz` path as gzip-compressed PDB text. It does not inspect content to guess a
format, accept mmCIF, fall back to another parser, or introduce a format enum.
Callers with mmCIF or another realization pass an already constructed `Structure`.

`read_pdb` materializes the first physical model and retains blank/A altloc rows in
source order. Model, chain, and altloc selection occur before occurrence identity,
component-conflict, or ATOM-evidence construction: a discarded alternate location
cannot change the retained identity or protein classification. `chains=None`
retains every parsed chain ID; a supplied ordinary
sequence filters all ATOM/HETATM rows by those IDs before protein, water, or context
construction. This is the released `parse_these_chains_only` meaning, not residue
packing selection: it removes unselected protein and environment atoms together.
An empty sequence therefore reaches the ordinary nonempty-Structure failure. The
reader has no model, altloc, assembly, occupancy-policy, device, context, or generic
options argument. Fixed-column padding is not identity: the one-character chain
field is stripped, so the blank chain ID is the ordinary empty string.

PDB protein classification is stable OFoldX chemistry rather than a ProDy flag
table. Standard amino acids plus `ASX`, `GLX`, `CME`, `CSB`, `CSO`, `GLH`, `HIP`,
`HSE`, `LYN`, `MEN`, `MSE`, `PHD`, `PTR`, `SEC`, `SEP`, `TPO`, and `UNK` are
protein component IDs independent of ATOM/HETATM spelling or observed CA. The
force-field aliases `ASH`, `ARN`, `CYX`, `HID`, `HIE`, `HSD`, `HSP`, `TYM`, `XAA`,
and `XLE` collide with non-polymer CCD IDs and are protein only when that source
occurrence contains an ATOM record. The two literal sets are reader implementation
facts, not public registries or user-extensible classification state.

The shared MPNN reader materializes a bounded coordinate observation, not complete
polymer topology. Each source protein chain becomes one `PolymerSequence` because
that is the exact chain-label and occurrence-order axis consumed by the released
Packer. Every reader-created protein component has `bonds=None`,
`previous_attachment=None`, and `next_attachment=None`. Its tuple declares ordered
known protein occurrences; it does not assert a complete source sequence or a
peptide bond between consecutive stored occurrences. Numeric gaps, insertion
codes, `TER`, and coordinate proximity neither add nor remove topology. A
topology-dependent consumer rejects these open components at its own boundary.
No segment, break, or inferred-adjacency entity is introduced.

For each standard amino acid, the reader declares the smallest atom vocabulary
closed under Packer realization: all package-audited non-hydrogen atoms except
`OXT`, which is exactly that residue's nonblank atom14 set, plus every retained
source atom ID that resolves in the audited component. `UNK` declares only its
retained source atom IDs that resolve in the audited component because the
checkpoint cannot generate an unknown side chain. Atom declaration order follows
the audited component; sparse `Structure.sites` independently retain coordinate
rows in source order. An unobserved atom14 declaration is target chemical identity
needed for possible packing, not a coordinate observation.

These open components do not reuse the complete audited `ChemicalAtom` facts. Each
declared atom carries only its known atom ID, element, and explicit isotope; charge,
aromaticity, stereo, hydrogen counts, and other optional chemistry remain unknown.
The audited component supplies only authoritative atom-ID-to-element resolution and
order, avoiding unsupported protonation, tautomer, reference-conformer, or complete
topology claims from a three-letter PDB name.

Other protein and nonprotein occurrences use ordinary open partial components with
their unique retained source atom IDs in source order. Their element comes only
from PDB columns 77–78 and ordinary case normalization. `D` and `T` lower to
element `H` with isotope 2 and 3 respectively. A blank or invalid element for an
otherwise retained partial atom fails at the reader boundary; this family never
infers an element from an atom name. A standard-amino-acid or `UNK` atom resolved
by the audited table may use that known element when the source field is blank;
an incompatible supplied element fails. The reader never declares a peptide
attachment, component bond, reference conformer, `CONECT` edge, or
geometry-inferred edge.

Each parsed protein chain uses its parsed ID as both entity and instance ID. Each
nonprotein occurrence receives `molecule_1`, `molecule_2`, and so on in first
retained source-occurrence order, again as both entity and instance ID. Distinct
chains or nonprotein occurrences are not grouped into a shared entity because
legacy coordinate rows provide no authoritative copy relation. Conflicting
component IDs among retained blank/A rows or duplicate retained canonical atom
sites fail instead of selecting a first or last row.

Residue and site order follow first source occurrence and are never sorted by
number. The public Biotite hybrid-36 decoder reads the four-character residue
number; insertion code remains separate. Blank occupancy becomes NaN, explicit
zero remains zero, and every retained PDB coordinate row has a true coordinate
mask. `residue_numbers` retains only protein occurrence numbering because that is
the bounded fact canonical Structure currently carries; this reader makes no
general PDB round-trip or nonpolymer source-identity claim.

Protein rows are exactly observed protein CA sites in `Structure.sites` order. The
same observation predicate is used for row membership and atom37 values. A missing
or filtered CA removes that component occurrence from the Model row axis but never
changes its canonical protein identity or moves its remaining atoms into ligand
context. A retained CA with missing or filtered N, C, or O remains a row with a
false residue mask.

When `residue_numbers is None`, direct programmatic input uses the zero-based
canonical component index as `residue_index`; it does not fabricate PDB numbering.
When the mapping is supplied, every Model row accesses it directly. The released
transform then walks the final CA row axis globally with a cumulative counter: the
counter increments whenever the current source integer number equals the
immediately previous number, never resets at a chain boundary, and each emitted
index is `number + counter`. Insertion code is retained as source identity but is
not itself used in this arithmetic. The shared MPNN reader populates this mapping
directly. It never fabricates gap chemistry, silently densifies source numbering,
or stores a hidden parser side channel.

`MpnnPackerConfig` retains `edge_size` independently from `hidden_size`:
the released feature-edge projections produce the former and the top-level
projections consume it into the latter. Encoder node width is `hidden_size`.
Although upstream exposes a separate `node_features` constructor argument, its
context projection requires that value to equal `hidden_dim`, so it is not an
independent working architecture axis.

`MpnnPackerConfig.atom_context_num` is a Model architecture axis: it fixes the
input width of the decoder's atom14-to-context projection.
`MpnnPackerProcessorConfig.atom_context_num` is a separate lowering axis: it fixes
how many nearest context atoms the Processor emits. The released route uses 16 for
both, so conversion writes 16 to both configs. The encoder consumes every emitted
context atom, while the decoder consumes the first Model-configured count. A
shorter Processor extent reaches the ordinary projection-shape failure; a wider
extent is accepted and changes encoder context exactly as upstream. OFoldX adds no
pair config, validator, inference, or synchronization layer.

Checkpoint conversion preserves the released tensor dtypes. It translates the
official key space and writes the paired configs; it does not expose an artifact
`dtype` option that casts only Model weights while leaving Processor runtime dtype
outside the repository. An experienced caller changes computation precision later
through ordinary Model or Pipeline placement.

The config also owns `coordinate_noise: float = 0.0`, the standard deviation of
Gaussian noise added to backbone coordinates while the Model is in training mode.
It is the explicit scientific control named `augment_eps` upstream. The released
checkpoint's `noise_level=0.02` is training metadata; official inference still
constructs the Packer with `augment_eps=0.0`, so conversion does not silently turn
that metadata into runtime augmentation. `num_chain_embeddings` is unused, device
is ordinary Module placement, and the upstream `atom37_order` layout switch is
absent because the public OFoldX tensor boundary explicitly consumes atom14
coordinates rather than an ambiguous generic atom array.

Rigid-frame vector and matrix products use ordinary PyTorch matrix multiplication.
The OpenFold helper's hand-expanded products are an execution policy that prevents
AMP downcasting, not a different rigid-body algebra. OFoldX does not preserve that
hidden exception: explicit dtype, caller autocast, and PyTorch matmul-precision
settings apply normally. Default-FP32 parity is numerical within the declared
family tolerance rather than a bitwise ordering contract.

`sequence` and `positions` have the semantics frozen in Section 2.1. The target
sequence may replace known identities. All changed positions must be packed, and
distinct copies of one entity may receive different targets. For this protein-only
packer, output partitioning is exact and local: within each original entity,
instances are grouped in original instance order by their complete target
component-ID tuple. Groups from different original entities never merge. A group
whose tuple equals the original component-ID tuple retains the original entity ID;
if none does, the first group retains it. Further groups receive
`<original_id>_2`, `<original_id>_3`, and so on while skipping allocated IDs.
Complete and nonstandard unchanged occurrences retain their stored component
declarations. After target identity and entity partitioning, the Processor
supplements every topology-open standard-amino-acid component in a protein entity
with each missing nonblank atom14 heavy-atom declaration. Existing declaration
order is preserved and missing atoms append in atom14 order. Each added
`ChemicalAtom` carries only its audited atom ID and element; every optional fact is
unknown. `bonds` remains `None`, existing attachments and system bonds are
preserved, no attachment or bond is inferred, and an existing reference-coordinate
tuple receives aligned `None` entries for the additions.

This normalization is once per entity component and independent of `positions`,
coordinate availability, and instance-local selection. Instances sharing an entity
therefore keep one chemical declaration. It creates no `Structure.site`, coordinate,
mask, or occupancy row: selection decides which declared sites a packed Result
materializes. A programmatic backbone-only open ALA can therefore declare CB
without pretending CB was observed. A component with `bonds is not None` asserts a
complete inventory and is never supplemented. `positions=None` excludes such a
component when an expected atom14 site is absent; explicit selection fails at the
packing boundary.

Identity replacement preserves three independent source knowledge states: whether
`bonds` is open, whether a previous attachment is present, and whether a next
attachment is present. An open target keeps `bonds=None`, uses the same minimal
standard atom vocabulary, retains source sites that resolve in the target, and
declares any atoms required by retained target-specific attachments. A complete
source receives the package-audited target atom and bond graph. A present source
attachment is replaced by the corresponding target attachment; an absent one
stays absent. This is family-owned target realization, not a generic graph
transform. A target that leaves a stored system-bond or stereo endpoint unresolved
fails through ordinary canonical construction.

The released checkpoint can select all 20 standard amino acids, including ALA and GLY;
ALA still needs a generated CB, while GLY simply has no side-chain chi metric.
Selected residues require observed N, CA, C, and O coordinates. Nonstandard
residues may remain as unselected context but this checkpoint does not pack them.
The Processor trusts the observation mask and does not add a coordinate-finiteness
scan.

A valid unselected nonstandard protein row retains its observed N, CA, C, and O in
the sampler's atom14 backbone slots even though its sequence token is unknown. The
same fixed observation state is reapplied after every denoising update; unknown
identity never turns a valid row into a zero-coordinate neighbor. Invalid source or
neighbor rows are excluded from both encoder and decoder messages. Semantic masks
remain boolean through Model feature and message computation, so multiplying a
caller-autocast activation by a hidden FP32 mask cannot defeat AMP.

The neighbor search assigns every invalid residue pair positive infinity before
`topk`, so every finite valid pair precedes every invalid pair. The released Packer
instead assigns invalid pairs the row's maximum valid distance. An invalid pair can
therefore tie and displace the farthest valid neighbor because `topk` has no stable
tie contract. OFoldX intentionally corrects that upstream defect. The returned
neighbor distances are not consumed; if fewer than `k_neighbors` valid rows exist,
the remaining invalid indices are harmless fill and the boolean source-by-neighbor
mask excludes them from both encoder and decoder messages.

The released parser filters the complete atom collection with `occupancy > 0`
before constructing protein rows, atom37 observations, or non-protein context, and
its public CLI can opt out. Canonical Structure retains occupancy independently.
With `filter_by_occupancy=True`, supplied occupancy is intersected with
`coordinate_mask` using the exact `> 0` predicate, which also excludes negative
and NaN values. `False` ignores occupancy and uses `coordinate_mask` alone.
`occupancy=None` likewise supplies no filter evidence. The input Structure is not
mutated. Packed output preserves supplied occupancy on retained source sites;
generated selected side-chain sites carry unknown occupancy rather than inheriting
an unrelated source value.

For every protein row, atom context is the nearest `atom_context_num` observed non-protein,
non-water, non-hydrogen atoms to its virtual CB, without a distance cutoff. This
includes ligands, ions, DNA, and RNA. Fixed protein side chains are separate atom14
context and are not part of those atoms. The released pair uses 16.
`use_atom_context=False` lowers every
non-protein context row to the ordinary padding representation: its coordinates and
atomic number are zero and its mask is false. The Processor does not enumerate or
rank molecular context for this disabled path. Zeroing only the mask is an upstream bug because the
Packer decoder concatenates an unmasked one-hot atomic type. The flag does not
delete molecules from the decoded Structure or remove fixed side chains. It is
therefore a call argument, not `ProcessorConfig` state.

`MpnnPackerInput` directly owns the tensors and decode associations needed by
the Model:

~~~python
@dataclass(eq=False)
class MpnnPackerInput:
    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    context_coordinates: Tensor
    context_atomic_numbers: Tensor
    context_mask: Tensor
    atom37_coordinates: Tensor
    atom37_mask: Tensor
    pack_mask: Tensor
    structure: Structure
    component_references: tuple[ComponentReference, ...]
~~~

`structure` is the canonical target-identity Structure after the explicit
partition above.
`component_references` gives the canonical identity of tensor rows. There is no
Layout, prepared-input wrapper, source bag, selection wrapper, or provenance field.
`chain_index` is a dense zero-based label for protein instances in first observed
protein-row order. The Model consumes only equality of these labels. An upstream
parser may offset the same protein partition by non-protein chains that are absent
from the residue tensor, so parity compares the induced same-chain relation rather
than treating arbitrary absolute integers as scientific values.

The public tensor operations are:

~~~python
def encode(
    self,
    input: MpnnPackerInput,
    *,
    atom14_coordinates: Tensor,
) -> tuple[Tensor, Tensor, Tensor]: ...

def decode(
    self,
    input: MpnnPackerInput,
    *,
    atom14_coordinates: Tensor,
    atom14_mask: Tensor,
    node_states: Tensor,
    edge_states: Tensor,
    neighbor_indices: Tensor,
) -> MpnnPackerOutput: ...

def forward(
    self,
    input: MpnnPackerInput,
    *,
    atom14_coordinates: Tensor,
    atom14_mask: Tensor,
) -> MpnnPackerOutput: ...

def sample(
    self,
    input: MpnnPackerInput,
    *,
    num_denoising_steps: int = 3,
    num_torsion_samples: int = 16,
    generator: torch.Generator | None = None,
) -> MpnnPackerSamplingOutput: ...

def pack(
    self,
    input: MpnnPackerInput,
    *,
    num_samples: int = 1,
    num_denoising_steps: int = 3,
    num_torsion_samples: int = 16,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> PackResult: ...
~~~

`encode` and `decode` are public in the released Model and remain public here.
The explicit coordinates on `encode` are not redundant with
`input.backbone_coordinates`: the released sampler reconstructs atom14 from rigid
frames and literature positions before encoding, and the reconstructed N/CA/C/O
state need not equal the observed Processor input. `decode` consumes both the
current atom14 coordinates and their explicit mask; it does not infer that dynamic
state again from `pack_mask`. `forward` performs one encode/decode calculation for
the supplied atom14 coordinate and mask state.
`sample` encodes once, repeats decode and coordinate update
`num_denoising_steps` times, and at each step draws `num_torsion_samples` proposals
per residue and chi before choosing the locally highest-density proposal. It
returns only the final step, not a fabricated trajectory. `pack` repeats that
complete stochastic sample `num_samples` times. All three counts are positive;
zero is not a sentinel. Seed and generator are mutually exclusive.

The Model directly owns the eight small fixed geometry tables used by this
sampling path: atom14 and atom37 existence, atom14-to-atom37 indices, chi atom
indices and masks, rigid-group default frames, atom14-to-rigid-group indices, and
literature atom14 positions. Every table is arranged once in the Model's
`AMINO_ACID_COMPONENT_IDS` plus unknown token order and registered as a derived
`persistent=False` buffer. There is no mutable module-global token-remapping
Tensor, geometry cache, load hook, lazy first-forward mutation, or separate
Geometry object. `sample` indexes the tables once for its input identities and
reuses those indexed tensors across torsion extraction and every denoising step;
the pure geometry functions receive only the tensors they actually consume.
State-dict and converted-checkpoint keys are unchanged.

Result realization preserves the relative order of every retained
`input.structure.sites` row. Existing selected atom14 side-chain sites stay in
place and are overwritten, while N, CA, C, and O remain fixed source facts; sites
that belong only to a replaced identity and selected hydrogens
are removed. Missing generated heavy sites append after all retained rows in Model
component-reference order and then atom14 order. Global `system.iter_atoms()` order
does not rewrite caller-owned coordinate-row order. Generated selected sites have
unknown occupancy; all retained occupancy facts stay aligned to their original
rows.

Raw `encode`, `decode`, `forward`, and `sample` preserve arbitrary tensor batch
size. `MpnnPackerInput` owns one canonical `Structure` and one row-to-component
mapping, so `pack` accepts exactly batch size one before realizing a `PackResult`;
it never silently takes row zero from a larger batch. The four frozen
design-to-pack compositions call the batch-general tensor operation and own their
B x pack grouping rather than flattening that meaning into the standalone Result.

The public default `num_torsion_samples=16` follows the released CLI operation.
The lower-level upstream helper's default of 10 remains expressible by passing 10
explicitly and does not justify inconsistent OFoldX defaults. The combined command's
default of four packs per designed sequence belongs to each concrete composition;
standalone `pack` continues to default to one result.

The raw outputs retain the values needed by direct scientific callers:

~~~python
@dataclass(eq=False)
class MpnnPackerOutput:
    mean: Tensor
    concentration: Tensor
    mix_logits: Tensor

@dataclass(eq=False)
class MpnnPackerSamplingOutput:
    atom14_coordinates: Tensor
    atom14_mask: Tensor
    sampled_chi_angles: Tensor
    chi_log_densities: Tensor
    mean: Tensor
    concentration: Tensor
    mix_logits: Tensor
    input_chi_sin_cos: Tensor
~~~

These dataclasses name related tensors; they retain ordinary positional or keyword
construction and ordinary dataclass representations. `eq=False` is the one
necessary exception because Tensor equality is not a scalar record comparison.
They do not claim immutability. Callers may reassign fields or mutate tensors using
ordinary PyTorch semantics. `frozen=True` would block only attribute assignment
while leaving tensor mutation untouched, so it would add a misleading defensive
restriction rather than an invariant.

There is no `b_factors` tensor: the upstream field is a visualization encoding
derived from chi log densities, not another scientific Model output. The released
writer's fixed-row sentinel and its mapping of backbone and chi-free residues to an
unused density head are serialization defects recorded in `UPSTREAM_BUGS.md`; they
do not justify retaining another raw tensor or a file payload in the task Result.
Raw floating outputs retain the Model computation dtype; OFoldX does not silently
restore upstream's unconditional FP32 casts after an explicit caller cast.

#### 4.6.1 Packed-PDB visualization

Status: frozen — the released one-file-per-pack operation, the final physical chi
metrics already retained by `PackPrediction`, the canonical site/coordinate axis,
and the audited writer defects support this bounded family visualization export.
This subsection freezes neither generic Structure exchange nor a source-file
round-trip contract.

The public family writer is symmetric with `read_pdb` and consumes one complete
task output:

~~~python
# ofoldx.models.mpnn_packer.pdb, reexported by ofoldx.models.mpnn_packer
def write_pdb(
    prediction: PackPrediction,
    path: str | PathLike[str],
) -> None: ...
~~~

One prediction maps to one file. A caller with several samples loops over
`result.predictions` and chooses each path explicitly. The writer does not accept a
`PackResult`, select the first prediction, invent filenames, add an output path to
the Pipeline, or require the original input or raw sampling output. Output alone is
therefore sufficient without a PreparedInput lifecycle.

The PDB B column is an explicit visualization channel, not a physical thermal
factor. For an atom whose audited atom14 rigid group is a real chi group 1 through
4, the writer uses the corresponding `chi1_log_density` through
`chi4_log_density` Metric when that exact component occurrence is present on the
Metric axis. Every atom without a produced physical chi density receives `0.0`.
This includes fixed residues, backbone atoms, ALA, GLY, `OXT`, nonprotein atoms,
and retained modified or unknown residues. OFoldX does not reproduce the released
writer's literal `2.0` fixed-row sentinel or its arbitrary head-zero value for
backbone and chi-free residues, and it adds no compatibility switch or
`pdb_b_factors` Result field for those defects.

The writer serializes every true coordinate row in `prediction.structure.sites`
order except water, which the released packed view omits. It preserves canonical
component and atom identity, coordinates, supplied occupancy, and source residue
numbers/insertion codes when present; an unknown occupancy is written as an unknown
field rather than fabricated as one. Protein atoms use `ATOM`, and all other atoms
use `HETATM`; the broken upstream `force_hetatm` option is absent. Elements come
from the canonical `ChemicalAtom`, never from a residue-name initial.

This is a deterministic visualization of the canonical prediction, not a PDB
source round-trip. The shared MPNN reader does not retain nonprotein author-chain
identity, record flags, atom serials, altlocs, segments, source PDB charge fields,
or source B values; this packed view does not claim general chemistry persistence.
The empty instance ID maps to the blank PDB chain. Any other unique one-character
printable, non-whitespace ASCII instance ID is retained, including punctuation.
Among instances that emit at least one retained record, remaining instances receive
unused chain characters in instance order: uppercase letters, lowercase letters,
digits, ASCII punctuation, then blank. Omitted water and all-false instances consume
no chain ID because they have no serialized record.

A supplied `residue_numbers` entry is used directly, while an occurrence without
one uses its one-based component position. Distinct emitted component occurrences
must remain distinct by `(chain, residue number, insertion code)`. A nonempty
insertion code is one ASCII letter; component and atom IDs contain no whitespace.
An identity or numeric value selected by this view that PDB cannot encode fails
rather than being truncated. Calling this explicitly named family PDB writer selects
that fixed view; it does not require a generic `allow_lossy`, `LossReport`, source
bag, or hidden identity map.

Because the released call controls exceed the common pack signature, its repository
uses `MpnnPackerPipeline`. The root pair is still exactly the packer Model
and Processor:

~~~python
def pack(
    self,
    input: Structure | str | PathLike[str],
    *,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    positions: Sequence[ComponentReference] | None = None,
    use_atom_context: bool = True,
    filter_by_occupancy: bool = True,
    num_samples: int = 1,
    num_denoising_steps: int = 3,
    num_torsion_samples: int = 16,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> PackResult: ...
~~~

`repack_everything` is not a second standalone selection mechanism. A composed
design-to-pack Pipeline may translate its concrete designer selection to
`positions` or `positions=None` before calling this pair.

The released combined command can batch distinct designed sequences and produce
several packs for each. The Packer is shared across five official designer types.
The complete command inventory proves four explicit live compositions:
`ProteinMpnnDesignAndPackPipeline` owns both protein and soluble weights,
`LigandMpnnDesignAndPackPipeline` owns ligand design, and
`GlobalMembraneMpnnDesignAndPackPipeline` and
`PerResidueMembraneMpnnDesignAndPackPipeline` retain their distinct biological
label inputs. Thus B × pack cardinality belongs to each concrete composition,
not to LigandMPNN ownership or the standalone Packer call. Their identical packing
tail and handoff justify `pack_sequences`; they do not justify a generic
composition base, protocol, mode, factory, or manifest.

The Packer Pipeline nevertheless owns the one biological batch operation proven by
all five routes:

~~~python
def pack_sequences(
    self,
    input: Structure | str | PathLike[str],
    *,
    sequences: Sequence[str | Mapping[str, str | Sequence[str]]],
    positions: Sequence[ComponentReference] | None = None,
    use_atom_context: bool = True,
    filter_by_occupancy: bool = True,
    num_samples: int = 1,
    num_denoising_steps: int = 3,
    num_torsion_samples: int = 16,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[PackResult, ...]: ...
~~~

This is one source Structure, one selected-position set, and an ordered set of
target protein sequences. The path branch parses once. Each target is lowered by
the ordinary singleton Processor contract, then their compatible tensors are
batched internally. Sampling runs pack index outermost across every sequence and
the return is transposed to input-sequence order: result `i` owns `num_samples`
predictions for `sequences[i]`. `pack` remains the singleton operation; batch
grouping does not add a BatchInput, plural canonical metadata, Model task method,
or Ligand-specific Packer. The nontrivial same-family realization shared by these
two Pipeline routes may be one private implementation helper in the existing
package; it has two real consumers but no independent scientific callable.

### 4.7 ProteinMpnn fixed-backbone design vertical

Status: frozen — an adversarial inventory of ProteinMPNN revision
`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, LigandMPNN revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de`, every active protein/soluble/CA
checkpoint, the tracked executable training checkpoint, both command surfaces,
public Model methods, training objectives, decoded outputs, and state schemas
supports the bounded contract below. The revisions make the evidence
reproducible; they are not OFoldX dependencies or public-behavior pins.

The unique Protein artifacts are four vanilla `v_48_{002,010,020,030}.pt`, four
soluble `v_48_{002,010,020,030}.pt`, three CA-only
`v_48_{002,010,020}.pt`, and
`training/exp_020/model_weights/epoch_last.pt`. LigandMPNN's protein and soluble
downloads are byte-identical aliases. Vanilla, soluble, and training state have
one 118-key full-backbone schema; CA state has a 123-key schema and a real
different coordinate-feature graph. They share one scientific Model–Processor
pair. Solubility and training provenance are learned weights, not runtime modes;
CA-only is an architecture and lowering fact:

~~~python
class ProteinMpnnConfig(Config):
    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 48
    dropout: float = 0.1
    ca_only: bool = False

class ProteinMpnnProcessorConfig(Config):
    ca_only: bool = False
~~~

The released graph has only one usable width. Its nominal node, edge, and hidden
dimensions fail when unequal. OFoldX therefore has no false `edge_feature_size`
axis and discards only the five proven-unused CA tensors: `W_v.weight`,
`W_v.bias`, `features.node_embedding.weight`, `features.norm_nodes.weight`, and
`features.norm_nodes.bias`. Full-backbone edge input width is 416 and has no
`W_v`; CA edge input width is 167. The presence of `W_v` alone does not identify
CA because membrane checkpoints have a live `W_v`. A converted repository writes
matching `ca_only` values to both configs. No pair validator synchronizes manually
constructed objects; an incompatible pair fails through ordinary tensor shape
behavior.

The family-owned input is one plain record rather than a nested Layout or prepared
wrapper. Protein, Ligand, and Membrane also have two exactly identical raw tensor
outputs, so those three real consumers use one concrete shared spelling in
`ofoldx.modules.mpnn` rather than six family aliases:

~~~python
@dataclass(eq=False)
class ProteinMpnnInput:
    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    valid_mask: Tensor
    residue_indices: Tensor
    chain_indices: Tensor
    design_mask: Tensor
    structures: tuple[Structure, ...]
    component_references: tuple[
        tuple[ComponentReference | None, ...], ...
    ]
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    residue_logit_bias: Tensor | None = None
    pssm_coefficients: Tensor | None = None
    pssm_probabilities: Tensor | None = None
    pssm_log_odds: Tensor | None = None

@dataclass(eq=False)
class MpnnOutput:
    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor: ...

    @property
    def probabilities(self) -> Tensor: ...

@dataclass(eq=False)
class MpnnSamplingOutput:
    sequences: Tensor
    log_probabilities: Tensor
    sampling_probabilities: Tensor
    decoding_ranks: Tensor
~~~

These are concrete MPNN tensor records, not bases or a generic output framework.
The raw output meaning is identical across the three graphs. Sampling vocabulary
extent remains observable in tensor shape: Protein has 21 adjusted classes while
Ligand and Membrane have 20; that difference needs no nominal wrapper.

One Protein Processor call returns `B=1`. The plural `structures` and nested
`component_references` remain because a later Runner may collate heterogeneous
training examples into the same raw input without a BatchInput or PreparedInput.
Only `encode` and differentiable raw `forward` accept such `B >= 1` input;
`MpnnOutput.logits` is `[B, L, 21]` and its ranks are `[B, L]`. Canonical metadata
does not drive Python control flow in that graph.

`sample`, all four probability methods, `score`, and `design` require `B=1`.
Their leading `N` or `E` dimension is candidate or evaluation identity for one
target, never an execution-batch axis, and their Results own one Structure. A
direct `B != 1` fails before those operations draw randomness; otherwise the
method could silently select row or target zero and return a scientifically
plausible wrong Result. This is the one necessary biological-axis check, not a
generic shape-validation pass. Released inference batches only clones of one
target; heterogeneous batching is used by training raw `forward`.

Optional evidence remains `None`; the common path does not allocate neutral
`[B, L, 21]` tensors merely to simplify internal branching. A `None` component
reference is one transient numeric-gap row from the original path arm, never
canonical chemistry. A direct Structure arm has no such row. Causal output is
represented by ranks: an untied order has distinct ranks and tied members have the
same rank. A flattened permutation cannot encode simultaneous tied sampling and
would reintroduce the upstream member-order bug.

`tied_group_ids` is aligned `torch.long [B, L]`. The Processor computes transitive
closure once, gives every causal unit a dense nonnegative ID in first-row order,
and gives tied members the same ID; an untied row is its own unit. Sampling and
probability hot paths consume only this Tensor and aligned final coefficients,
never nested Python groups, `.tolist()`, or data-dependent compile guards.
`tied_weights`, when present, is an aligned floating Tensor `[B, L]`; `None`
selects the Protein family default after group IDs are known.

The Processor has two explicit lowering arms. `processor(path)` means original
ProteinMPNN fixed-column semantics. `processor(structure)` means the updated
canonical Structure semantics used by LigandMPNN's protein and soluble routes.
The public expression selects the arm; no provenance bit or runtime reader mode is
stored. Their real path-specific controls live on explicit methods rather than a
sentinel or an argument that is inert for half of a union.

~~~python
def __call__(
    self,
    input: Structure | str | PathLike[str],
    *,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    positions: Sequence[ComponentReference] | None = None,
    conditions: Sequence[
        AllowedResidueIdentities | TiedResidueIdentities
    ] = (),
    tied_weights: Mapping[ComponentReference, float] | None = None,
    residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
    pssm_coefficients: Mapping[ComponentReference, float] | None = None,
    pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
    pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
) -> ProteinMpnnInput: ...

def process_structure(
    self,
    structure: Structure,
    *,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    positions: Sequence[ComponentReference] | None = None,
    conditions: Sequence[
        AllowedResidueIdentities | TiedResidueIdentities
    ] = (),
    tied_weights: Mapping[ComponentReference, float] | None = None,
    residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
    pssm_coefficients: Mapping[ComponentReference, float] | None = None,
    pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
    pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
    filter_by_occupancy: bool = True,
) -> ProteinMpnnInput: ...

def process_pdb(
    self,
    path: str | PathLike[str],
    *,
    chains: Sequence[str] | None = None,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    positions: Sequence[ComponentReference] | None = None,
    conditions: Sequence[
        AllowedResidueIdentities | TiedResidueIdentities
    ] = (),
    tied_weights: Mapping[ComponentReference, float] | None = None,
    residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
    pssm_coefficients: Mapping[ComponentReference, float] | None = None,
    pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
    pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
) -> ProteinMpnnInput: ...
~~~

`__call__(path)` delegates to `process_pdb(path, chains=None)` and
`__call__(structure)` delegates to
`process_structure(structure, filter_by_occupancy=True)`. The explicit Structure
method preserves the updated zero-occupancy operation; the path method exposes only
the original chain filter and never pretends occupancy affects its parser. Chain
filtering is kept because original `parse_PDB(input_chain_list=...)` removes other
chains from the graph and design positions cannot express that operation. The
fixed-column row parser is a private implementation in
`proteinmpnn/processor.py` with this one consumer; there is no public parsed-PDB
wrapper or Protein-specific `read_pdb`.

The one-pair concrete Pipeline owns design, the original scalar score protocol,
and four distinct probability computations:

~~~python
class ProteinMpnnPipeline:
    def __init__(
        self,
        *,
        model: ProteinMpnnModel,
        processor: ProteinMpnnProcessor,
    ) -> None: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        backbone_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnDesignResult: ...

    def score(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult: ...

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...
~~~

Each method lowers once and calls one matching Model operation once. `sequence`
belongs only to alternate-sequence scalar scoring. `positions` selects generated
rows, the designed score scope, or the fixed/design partition retained in aligned
probability output; every probability computation still evaluates all Model rows.
Tied conditions affect the two autoregressive orders, while the released
single-residue path ignores ties. Conditions, tied coefficients, residue evidence,
global bias or omission, temperature, and PSSM sampling controls belong only to
design. Design's `num_decoding_orders` controls its independent native input score.
`global_omit_tokens=("X",)` preserves the released command default; an explicit
empty sequence enables original ProteinMPNN's reachable 21st generated class.
Every alternate sequence is lowered independently and must exactly cover the
canonical protein occurrences selected for that scored target; no prefix update
can retain a native or previously scored suffix.

There are no reject-only MSA, template, coordinate-free-system, generic Condition,
or generic reader arguments. The original path arm has these exact observable
semantics:

- only ATOM records plus HETATM MSE rewritten to MET participate;
- fixed-column decimal residue integers are used; gzip, hybrid36, MODEL, altloc,
  and occupancy semantics are not added;
- all model and altloc rows are visited and the first coordinate for an atom name
  at one integer/insertion occurrence wins;
- default chain order is `A`–`Z`, `a`–`z`, then `0`–`9`, with the blank chain
  absent; `process_pdb(chains=...)` uses the caller's exact selected chains;
- designed chains are lexically sorted before lexically sorted fixed chains;
- within a chain, every integer from minimum to maximum is emitted, insertion
  occurrences are sorted, and every absent integer creates one transient invalid
  X row with a `None` canonical reference;
- full validity requires finite N/CA/C/O, CA validity requires finite CA, and
  nonfinite values are zeroed after that mask is derived; and
- for global tensor row `i` and zero-based tensor-chain ordinal `c`, positional
  value is `100 * c + i`, while chain labels follow tensor-chain order.

The Structure arm has the deliberately different updated semantics:

- Model rows are static protein occurrences with a retained observed CA, in
  physical CA/source order; occurrences without CA are absent and no numeric gaps
  or designed-chain reordering are fabricated;
- every row has a real `ComponentReference`, MSE lowers to X, and default
  occupancy accepts an absent value or a value strictly greater than zero;
  `filter_by_occupancy=False` ignores occupancy;
- validity is exact observation of N/CA/C/O in full mode or CA in CA mode, without
  an added finiteness scan;
- chain labels are dense in first retained-row instance order; and
- without source numbering the positional value is component index. With source
  integers, one global cumulative counter increments whenever the current integer
  equals the immediately previous integer, never resets at a chain boundary, and
  is added to every subsequent emitted integer.

Composing the shared `ofoldx.io.read_mpnn_pdb(path, chains=...)` selects the
Structure arm and therefore also selects that reader's first-model, blank/A-altloc,
gzip, hybrid36, static-chemistry, and retained-occupancy semantics. The two public
expressions are intentionally not equivalent:

~~~python
processor(path) != processor(read_mpnn_pdb(path))
~~~

Validity and selection remain independent. `positions=None` selects all valid
protein occurrences, an empty sequence selects none, duplicates are idempotent,
and an explicit nonprotein or invalid occurrence fails. A valid nonstandard
protein occurrence is selectable through X. In the path arm a chain is designed
when it contains any selected occurrence; in the Structure arm selection never
reorders rows.

Allowed identities intersect per position. Ties form transitive groups, and hard
support intersects across each group. A fixed group member anchors exact chemical
identity; conflicting anchors fail. There is no candidate-topology pre-scan.
Per-residue bias and PSSM inputs use canonical-reference mappings because source
gap expansion makes raw caller row tensors ambiguous. Conditions are not retained
in Model input or Results.

The public Model surface is:

~~~python
def encode(
    self,
    input: ProteinMpnnInput,
    *,
    backbone_noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]: ...

def forward(
    self,
    input: ProteinMpnnInput,
    *,
    decoding_ranks: Tensor | None = None,
    backbone_noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> MpnnOutput: ...

def sample(
    self,
    input: ProteinMpnnInput,
    *,
    num_samples: int = 1,
    temperature: float = 0.1,
    decoding_ranks: Tensor | None = None,
    backbone_noise: float = 0.0,
    global_logit_bias: Tensor | None = None,
    global_omit_token_mask: Tensor | None = None,
    pssm_mixture: float = 0.0,
    pssm_log_odds_threshold: float | None = None,
    generator: torch.Generator | None = None,
) -> MpnnSamplingOutput: ...

def autoregressive_probabilities(
    self,
    input: ProteinMpnnInput,
    *,
    num_decoding_orders: int = 1,
    decoding_ranks: Tensor | None = None,
    backbone_noise: float = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def sequence_free_autoregressive_probabilities(
    self,
    input: ProteinMpnnInput,
    *,
    num_decoding_orders: int = 1,
    decoding_ranks: Tensor | None = None,
    backbone_noise: float = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def conditional_probabilities(
    self,
    input: ProteinMpnnInput,
    *,
    num_decoding_orders: int = 1,
    base_decoding_ranks: Tensor | None = None,
    backbone_noise: float = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def unconditional_probabilities(
    self,
    input: ProteinMpnnInput,
    *,
    num_decoding_orders: int = 1,
    base_decoding_ranks: Tensor | None = None,
    backbone_noise: float = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def score(
    self,
    input: ProteinMpnnInput,
    *,
    num_decoding_orders: int = 1,
    decoding_ranks: Tensor | None = None,
    backbone_noise: float = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> ScoreResult: ...

def design(
    self,
    input: ProteinMpnnInput,
    *,
    num_candidates: int = 1,
    num_decoding_orders: int = 1,
    temperature: float = 0.1,
    backbone_noise: float = 0.0,
    global_logit_bias: Tensor | None = None,
    global_omit_token_mask: Tensor | None = None,
    pssm_mixture: float = 0.0,
    pssm_log_odds_threshold: float | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> ProteinMpnnDesignResult: ...
~~~

Each operation that encodes coordinates exposes `backbone_noise`; it is an
inference operation control in Å, not checkpoint training metadata. Nonzero noise
is drawn independently for each candidate or decoding-order evaluation before
feature construction. `forward` is the single-evaluation differentiable
sequence-conditioned core. The four named methods preserve four algorithms rather
than one Boolean-selected method. `sequence_free_autoregressive_probabilities`
remains order-dependent; it is not unconditional. One `sample` handles tied and
untied inputs, so a second sampler method would add no scientific operation.

Generated causal ranks place every fixed row before every designed row, then
randomize only within those categories. Autoregressive masks compare neighbor and
query ranks directly; they do not materialize a dense one-hot/einsum permutation.

Caller-controlled autoregressive order is represented only by optional ranks,
never by raw random priorities, a flattened permutation, or a second order
concept. Ranks are `torch.long` on the input device; lower values are earlier, and
equal values mean simultaneous members of one tied group. For a direct batch,
`forward` ranks are `[B, L]`. For one biological input, autoregressive and
sequence-free results have logits `[E, L, 21]` and ranks `[E, L]`, where
`E = num_decoding_orders`. Conditional and unconditional results have logits
`[E, L, 21]` and effective ranks `[E, L, L]`, with evaluation, target, and causal
row axes. Sampling has ranks `[N, L]`. There is no implicit broadcasting.

Protein sampling returns `sequences [N, L]`, raw untempered and unbiased
`log_probabilities [N, L, 21]`, adjusted
`sampling_probabilities [N, L, 21]`, and exact ranks `[N, L]`. Fixed rows retain
zero sampling and raw arrays because no token is sampled there. The raw Model can
generate every class in `ACDEFGHIKLMNPQRSTVWYX`; the Pipeline's default X omission
is an operation default, not a permanent support reduction. A generated X realizes
canonical `UNK`. Unsupported fixed source identities remain exact when not
replaced.

Generated ranks place every `valid_mask & design_mask` causal unit after every
other unit. A transitive tied group is one unit; a mixed fixed/design group is an
anchored fixed unit. Invalid rows retain an aligned rank but are ignored by masks.
Explicit ranks are experienced-caller intent and are used without fixed-first
repair; tied members must already have equal ranks and distinct causal units have
distinct ranks. Supplying ranks skips only the random draw for the order. Backbone
noise and token sampling still consume the supplied generator, and a sampling
output returns the exact effective ranks.

Conditional and unconditional probabilities interpret supplied ranks as one base
order. For each target, conditional evaluation moves only that target exactly
last; unconditional evaluation moves only it exactly first. Both preserve every
non-target relative order and return the effective target-specific ranks.
`design` does not accept ranks because it combines candidate sampling with an
independent input score and one tensor would be ambiguous. The concrete Pipeline
also does not expose internal tensor-row ranks; exact control remains available
through typed Model methods.
Sampling collects raw, untempered and unbiased conditional log probabilities in
the autoregressive pass and does not rerun forward. Hard support is intersected
once after global bias, residue bias, and PSSM soft mixing, so thresholding and
omissions cannot leak or be restored by later soft operations.

For a transitive tied group `G`, `tied_weights` stores final signed neural-logit
coefficients and the group logits are
`sum(tied_weight[i] * logits[i] for i in G)`. They are not normalized by their sum
and affect neural logits only. ProteinMpnn's omitted member coefficient is
`1 / len(G)`. Original `tied_beta[i]` compatibility therefore lowers to
`tied_beta[i] / len(G)`. Updated LigandMPNN protein/soluble sum behavior remains
exactly expressible by explicitly assigning coefficient one; it is not chosen
from path origin or hidden provenance. Global and residue logit bias, PSSM
mixture, hard support, and fixed anchors are not multiplied by coefficients.
A supplied weight reference must belong to a final non-singleton transitive tied
group; otherwise the Processor rejects a scientific parameter that would be
silently inert. Unspecified members use the Protein default `1 / len(G)`.

A tied group has one soft control. After ordinary untied defaults are applied,
`residue_logit_bias`, `pssm_coefficients`, and `pssm_probabilities` must be
elementwise identical at every member and are applied exactly once after neural
logit aggregation. The per-reference input mappings remain; callers explicitly
assign the same values to the members. The Processor rejects unequal values after
transitive closure. It does not broadcast one member, invent a group wrapper, or
offer an aggregation policy. A caller combining distinct external evidence does so
under its own scientific semantics before passing the resulting group value.

Sparse PSSM probabilities and coefficients form one paired soft control. Empty
mappings carry no evidence and are equivalent to `None`. A nonempty probability
mapping materializes both dense tensors: probabilities and coefficients are zero
outside its references, each supplied probability receives coefficient one by
default, and explicit coefficient entries override those defaults. Coefficient
keys must be a subset of probability keys; a coefficient without the same
probability fails, and `pssm_log_odds` is not a substitute distribution. When
probabilities are absent or empty, a nonempty coefficient mapping fails. When both
are absent or empty, both tensors remain `None`.

Tied equality is checked after those defaults, without member broadcast. At the
direct dense Model boundary, `pssm_coefficients=None` with a dense probability
tensor means coefficient one at every row; a manually constructed dense input
represents per-row absence explicitly with coefficient zero. The Model does not
infer presence from probability values.

`pssm_log_odds` becomes hard evidence only when thresholded and may differ by
member. Its support intersects with every member's allowed identities, global
omissions, and any fixed identity. Hard support is applied last, without epsilon,
restoration, or fallback. Multiple fixed anchors must have the same exact
generatable chemical identity. This removes the released last-member and member-
order bugs without guessing a numerical composition that neither the paper nor
the audited implementation defines.
A sparse log-odds mapping lowers absent references to positive infinity, which is
neutral under ordinary finite thresholds. `None` disables threshold evidence; an
explicit numerical `pssm_log_odds_threshold`, including infinity or NaN, is caller
intent and participates in the ordinary Tensor comparison without a numerical
preflight or repair. When the mapping itself is absent or empty, it remains `None`
and a supplied threshold is likewise neutral; the common path does not allocate an
all-infinity tensor merely to represent no evidence.

ProteinMpnn uniquely returns its native input sequence score with a design:

~~~python
@dataclass(eq=False)
class ProteinMpnnDesignResult(DesignResult):
    input_score: ScoreResult
~~~

`score` returns only `designed_sequence_nll` and `global_sequence_nll`, each with
ordered decoding-order values in `Score.evaluations`, explicit canonical scope,
`unit="nat/residue"`, and `preferred="lower"`. It does not invent residue-NLL,
uncertainty, diagnostics, provenance, or condition outcomes. Every candidate owns
single-value designed/global NLL and sequence recovery. Candidate likelihood uses
the raw autoregressive log probabilities, not the adjusted sampling distribution.
The updated protein/soluble confidence is exactly
`exp(-designed_sequence_nll)` and does not justify a duplicate Score.

Realization partitions instances within each original entity by their complete
output component-ID sequence, never merges across original entities, preserves the
unchanged partition's ID when present and otherwise the first partition's ID, and
allocates `_2`, `_3`, and so on without collision. It preserves untouched entities,
instance IDs, and resolvable stored bonds. An alternate-sequence score target
rebuilds the system and retains only observations that resolve in the new chemistry;
changed protein occurrences retain the N/CA/C/O observations used by full mode or
CA in CA-only mode, not stale side-chain sites.

Only ProteinMPNN's inspected training surface proves objective ownership. It
publishes two ordinary differentiable functions in
`ofoldx.models.proteinmpnn.objectives`:

~~~python
def nll_loss(sequence_tokens: Tensor, log_probabilities: Tensor) -> Tensor: ...

def smoothed_loss(
    sequence_tokens: Tensor,
    log_probabilities: Tensor,
    *,
    weight: float = 0.1,
) -> Tensor: ...
~~~

Both return the unreduced per-row loss with shape `sequence_tokens.shape`.
`smoothed_loss` adds `weight / 21` to the one-hot target and renormalizes each row
before cross entropy. Masked mean, accuracy, and the training command's fixed
`/ 2000` aggregate belong to the caller or a later Runner; they do not create
competing core objectives. Ligand/Membrane inference code proves no additional
training-objective contract merely because graph primitives are shared. The
tracked training Model weights convert with the family now, while optimizer
resume, datasets, scheduling, and Runner composition receive a later Tik.

Conversion loads with `weights_only=True` and `mmap=True`, classifies CA by the
167-wide edge input rather than `W_v`, derives every live architecture fact,
preserves serialized dtype, translates every live tensor, strict-loads with
`assign=True`, and saves the paired concrete Pipeline artifact. It exposes no
dtype conversion option. The pair and concrete Pipeline use exact ID
`ofoldx.proteinmpnn`; saving writes that exact Pipeline marker. Hub publication is
a separate external action and no publication declaration is made until all
active Protein artifacts strict-load and existing real-weight parity evidence
covers both lowering arms, every public operation, the 21-class support case, and
accepted intentional divergences. The on-demand parity script and observations do
not enter frequent CI or committed package history.

#### 4.7.1 ProteinMpnn design-and-pack composition

Status: frozen — the released combined command's protein and soluble `model_type`
routes, the updated Structure lowering frozen above, the independent Packer batch
operation in Section 4.6, candidate × pack cardinality, selected-only/full
repacking, and exact RNG order support this one explicit live composition. Soluble
weights change the assembled designer artifact, not the composition class.

~~~python
class ProteinMpnnDesignAndPackPipeline:
    def __init__(
        self,
        *,
        designer: ProteinMpnnPipeline,
        packer: MpnnPackerPipeline,
    ) -> None: ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        backbone_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ProteinMpnnDesignResult, tuple[PackResult, ...]]: ...
~~~

A path is parsed exactly once with `ofoldx.io.read_mpnn_pdb`; that canonical
`Structure` is the only source given to both stages. The designer stage lowers it
through `designer.processor.process_structure`, including the explicit
`filter_by_occupancy` value, physical-CA order, and updated MSE semantics, before
invoking the matching Model design operation. It intentionally does not call
`ProteinMpnnPipeline.design(path)`, which would select the original fixed-column,
numeric-gap, designed-chain-first path arm. The following expressions are therefore
not interchangeable:

~~~python
ProteinMpnnDesignAndPackPipeline(...).design(path)
ProteinMpnnPipeline(...).design(path)
~~~

The first expression is the released shared-parser composition route; the second
is the standalone original Protein route. This is observable scientific lowering,
not an implementation detail to repair with a second parse or hidden source state.

With `repack_everything=False`, packing positions are exactly the canonical
references in the designer's effective `D = valid_mask & design_mask` scope. A
deterministic singleton allowed identity remains designed and is packed. With
`True`, the composition passes `positions=None` to the Packer. It forwards every
candidate's complete protein component-ID sequences to one
`packer.pack_sequences(...)` call; `use_packing_atom_context` maps only to that
call's `use_atom_context`. For candidate `i`, returned element `packs[i]` owns
`num_packs_per_candidate` predictions.

A supplied `seed` creates one operation generator. Design consumes it first, then
packing advances the same stream pack-major across all candidates. A supplied
generator is advanced in place; neither child restarts it or derives a stage seed.
`to(...)` delegates the same ordinary placement request to the named `designer`
and `packer`, updates in place, and returns `self`.

The class owns no aggregate save, registration ID, Pipeline marker, manifest,
loader hook, or generic composition parent. Its two already assembled children
remain independently loadable Hub artifacts; the composition is explicit and
live-only.

### 4.8 LigandMpnn context-aware sequence-design pair

Status: frozen — an adversarial audit of LigandMPNN revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de`, its four active
`ligandmpnn_v_32_{005,010,020,030}_25.pt` checkpoints, design and score commands,
public Model/Module methods, all four probability computations, context lowering,
sampling support, decoded outputs, and state schemas supports the bounded pair
below. Its independently audited design-to-MpnnPacker route is one of the four
explicit compositions frozen in Sections 4.7.1–4.9.2. The inspected revision is
reproducible evidence, not a runtime
dependency or behavior pin.

The four checkpoints were trained at four coordinate-noise levels, but that label
does not change the executable graph. They share one pair:

~~~python
class LigandMpnnConfig(Config):
    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 32
    dropout: float = 0.0

class LigandMpnnProcessorConfig(Config):
    num_context_atoms: int = 25
~~~

The released graph has one usable node/edge/hidden width. Its nominal independent
width arguments fail when unequal, so OFoldX retains one `hidden_size` rather than
three false axes. `num_context_atoms` fixes only the Processor's padded nearest-
context extent; it controls no learned tensor shape. The Model reads that extent
from its input. `use_atom_context`, `use_side_chain_context`, interface cutoff,
coordinate noise, temperature, and sampling counts are operation controls rather
than persistent configuration.

Checkpoint `noise_level` is training metadata. Official inference leaves
`augment_eps=0`; conversion never turns the metadata into runtime noise. Every
operation that offers noise calls it `coordinate_noise` because the released
feature computation perturbs both backbone coordinates and the already selected
nonprotein context coordinates before residue-neighbor and atom-context features.
Fixed side-chain coordinates are not perturbed.

The direct input is an ordinary aligned tensor record. Raw forward and sampling
return the shared `MpnnOutput` and `MpnnSamplingOutput` frozen in Section 4.7:

~~~python
@dataclass(eq=False)
class LigandMpnnInput:
    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    valid_mask: Tensor
    residue_indices: Tensor
    chain_indices: Tensor
    context_coordinates: Tensor
    context_atomic_numbers: Tensor
    context_mask: Tensor
    residue_context_coordinates: Tensor
    residue_context_mask: Tensor
    design_mask: Tensor
    ligand_interface_mask: Tensor
    structure: Structure
    component_references: tuple[ComponentReference, ...]
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    logit_bias: Tensor | None = None
~~~

The canonical Processor operation owns one Structure, so its tensor batch is one.
`backbone_coordinates` is `[1, L, 4, 3]` in N/CA/C/O order. Context tensors are
`[1, L, A, ...]`, where `A` is the Processor extent. The Model consumes only the
32 official residue-context slots at atom37 indices 5:37; carrying a redundant
full atom37 array would duplicate backbone coordinates and unused slots. The
tensors are therefore `[1, L, 32, 3]` and `[1, L, 32]` in the released CG-through-
OXT slot order. This slice is called residue context because it deliberately
excludes CB and includes OXT; a field named side-chain atoms would be false. A
manually constructed input may use the same tensor contract, but one
input never claims associations for several unrelated Structures.

`tied_group_ids` is aligned `torch.long [1, L]`. The Processor closes transitive
ties once, assigns dense causal-unit IDs in first-row order, and gives each untied
row its own ID. The Model hot path receives no nested Python group structure.
`tied_weights`, when present, is aligned `[1, L]`; `None` selects coefficient one.

Sampling returns `sequences [N, L]`, raw untempered and unbiased
`log_probabilities [N, L, 21]`, adjusted generatable
`sampling_probabilities [N, L, 20]`, and exact causal
`decoding_ranks [N, L]`. Fixed rows retain zero sampling and raw-log-probability
arrays as in the released operation. The unknown token remains a raw output class
and native context token but is never generatable.

A probability operation returns the concrete shared `MpnnProbabilityResult`
frozen in Section 5.4.2. With `E = num_decoding_orders`, both autoregressive
operations return `logits [E, L, 21]` and effective ranks `[E, L]`. Conditional
and unconditional probabilities return `logits [E, L, 21]` and effective ranks
`[E, L, L]`, with evaluation, target, and causal-row axes. Every operation retains
the released leading evaluation axis, including unconditional target-first
evaluation. The official command saves raw aligned distributions, their mean, and
population standard deviation rather than an aggregate NLL; `ScoreResult` would
lose the actual output.

Optional evidence remains `None`; the common path does not allocate neutral
per-position masks or bias tensors. These dataclasses bundle aligned tensors only.
They add no frozen mutation, custom equality, hidden validation, representation
hook, Layout, Backbone wrapper, prepared input, provenance, or diagnostics.
`LigandMpnnFeatures` and every other reusable `nn.Module` class remain public.
The root Model owns the released feature Module directly; a
`LigandMpnnBackbone` whose only purpose is a checkpoint prefix does not exist.

The Processor consumes only inputs used by this pair:

~~~python
def __call__(
    self,
    input: Structure | str | PathLike[str],
    *,
    positions: Sequence[ComponentReference] | None = None,
    conditions: Sequence[
        AllowedResidueIdentities | TiedResidueIdentities
    ] = (),
    tied_weights: Mapping[ComponentReference, float] | None = None,
    logit_bias: Tensor | None = None,
    residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
    use_atom_context: bool = True,
    ligand_interface_cutoff: float = 8.0,
    filter_by_occupancy: bool = True,
) -> LigandMpnnInput: ...
~~~

There is no alternate-sequence argument: the released LigandMPNN scoring command
scores the input sequence only. There are no MSA, template, MolecularSystem,
generic Condition, generic options, or reject-only symmetry arguments. Coordinates
are required. `use_side_chain_context` is Model-owned because it changes feature
construction after coordinate noise and residue KNN; the Processor always emits
the observed side-chain facts.

`logit_bias` is one generatable-token vector broadcast to selected rows;
`residue_logit_bias` supplies sparse row additions. The Processor sums them once
into the optional aligned `input.logit_bias`. The upstream unknown-token bias is
inert because sampling removes unknown and renormalizes the 20 retained values, so
the public bias axis contains only those 20 generatable identities. Global hard
omission is not a second tensor argument: one `AllowedResidueIdentities` record
over the selected positions expresses it exactly and combines with every other
hard restriction through the one support intersection.

The path branch is exactly `processor(read_pdb(path), ...)` using the shared public
reader frozen in Section 4.6. PDB chain filtering remains the reader's `chains`
argument rather than a path-only Processor option; callers needing it invoke the
reader explicitly. Direct Structure and path inputs then share one lowering.

Protein rows are exactly retained protein occurrences with an observed CA, in CA-
site order. A missing CA removes the occurrence from the Model axis but never turns
its other atoms into ligand context. A retained row is valid exactly when N, CA, C,
and O are observed; there is no extra finite-value scan. Protein identity excludes
all atoms of every protein occurrence from nonprotein context even when that
occurrence has no Model row or is invalid. MSE lowers to the unknown token.

Without source numbering, `residue_indices` use zero-based component indices. With
the reader's PDB numbering, the released transform walks final CA rows globally:
a cumulative counter increments whenever the current integer equals the immediately
previous integer, does not reset at a chain boundary, and is added to every emitted
integer. Chain indices are dense labels in first Model-row instance order; only
their equality is scientific.

With `filter_by_occupancy=True`, an observed coordinate participates only when
occupancy is absent or strictly greater than zero; false ignores occupancy.
The Processor adds no positive/finite cutoff gate. For every row it selects the
nearest `num_context_atoms` observed nonprotein, nonwater, nonhydrogen atoms to the
virtual Cβ, with padding and no distance cutoff. Ligands, ions, DNA, and RNA are
ordinary context. `ligand_interface_mask` is computed first using the closest such
atom and the released strict `< ligand_interface_cutoff` predicate. It remains
available when `use_atom_context=False`, which sets the emitted context mask false
without changing the scoring scope or deleting chemistry from the Structure.

`positions` has the shared selection meaning in Section 3.8. `None` selects every
valid protein row, an empty sequence selects none, duplicates are idempotent, and
an explicit nonprotein or invalid occurrence fails. Unmodeled protein occurrences
remain unchanged in every candidate. Allowed identities intersect at each selected
row and across transitive tied groups. A fixed tied member anchors exact identity;
conflicting fixed anchors or an anchor outside the 20 generated identities are
contradictions. No topology pre-scan or satisfaction report is added.

The public Model operations are:

~~~python
def encode(
    self,
    input: LigandMpnnInput,
    *,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]: ...

def forward(
    self,
    input: LigandMpnnInput,
    *,
    decoding_ranks: Tensor | None = None,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    generator: torch.Generator | None = None,
) -> MpnnOutput: ...

def sample(
    self,
    input: LigandMpnnInput,
    *,
    num_samples: int = 1,
    temperature: float = 0.1,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    decoding_ranks: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> MpnnSamplingOutput: ...

def autoregressive_probabilities(
    self,
    input: LigandMpnnInput,
    *,
    num_decoding_orders: int = 1,
    decoding_ranks: Tensor | None = None,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def sequence_free_autoregressive_probabilities(
    self,
    input: LigandMpnnInput,
    *,
    num_decoding_orders: int = 1,
    decoding_ranks: Tensor | None = None,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def conditional_probabilities(
    self,
    input: LigandMpnnInput,
    *,
    num_decoding_orders: int = 1,
    base_decoding_ranks: Tensor | None = None,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def unconditional_probabilities(
    self,
    input: LigandMpnnInput,
    *,
    num_decoding_orders: int = 1,
    base_decoding_ranks: Tensor | None = None,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> MpnnProbabilityResult: ...

def design(
    self,
    input: LigandMpnnInput,
    *,
    num_candidates: int = 1,
    temperature: float = 0.1,
    coordinate_noise: float = 0.0,
    use_side_chain_context: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> DesignResult: ...
~~~

LigandMpnn exposes four named probability computations.
`autoregressive_probabilities` is native-sequence-conditioned autoregressive
evaluation. `sequence_free_autoregressive_probabilities` is the released
sequence-free decoder and remains rank-dependent because earlier-ranked encoder
neighbors are masked. `conditional_probabilities` moves each target exactly after
every other residue and uses the normal sequence decoder.
`unconditional_probabilities` moves each target exactly before every other
residue. `forward` is the ordinary sequence-conditioned teacher-forced graph.
There is no conditioning flag or public raw-distribution method called `score`.

All four probability methods are ordinary differentiable Model methods. They
preserve the caller's grad, autocast, and compile context. The released command
chooses `torch.no_grad()` at its application boundary; that is not a Model contract
and does not justify hidden `inference_mode` decorators on scientific probability
operations.

Generated ranks put every fixed causal unit before every designed unit. A tied
group is one unit and every member has the same rank. Explicit ranks are `long`,
lower means earlier, equal means simultaneous tied members, and no implicit
broadcast or fixed-first repair changes experienced-caller input. Ordinary forward,
sampling, and both autoregressive operations have ranks `[N, L]`. Conditional and
unconditional base ranks have exactly `[N, L]`; for every target the operation
moves only that target last or first respectively, preserves every other rank
relation, and returns effective `[N, L, L]` ranks. One call generates base ranks
before encoding, applies coordinate noise independently per evaluation, and
retains every released repeated evaluation. Supplying ranks skips only their
random draw.

When fixed side-chain context is enabled, only source-fixed rows contribute. The
first 16 residue KNN neighbors supply their observed 32-slot side-chain atoms.
Those atoms are concatenated with nonprotein context and the nearest `A` combined
atoms are reselected around the possibly noise-perturbed virtual Cβ. Disabling
nonprotein context does not disable fixed-side-chain context.

For a tied group `G`, `tied_weights` stores final signed coefficients and neural
logits are exactly `sum(tied_weight[i] * logits[i] for i in G)`. They are neither
divided by group size nor normalized by weight sum. Missing member coefficients
are one, the released Ligand default. Homo-oligomer averaging is expressed
explicitly by assigning `1 / len(G)` to every member. Coefficients apply to neural
logits only. The aligned global-plus-residue bias must be
elementwise identical at every member after sparse defaults and applies once;
hard allowed support intersects across the group and applies last. Unknown is
always outside generated support. All members are evaluated from the same pre-
publication decoder state. Raw log probabilities remain each member's individual
unadjusted distribution; one group sampling distribution is copied to designable
members while fixed members retain zero arrays. This removes last-member and
member-order behavior without inventing weighted evidence aggregation.

For each candidate, let `D = valid_mask & design_mask` and
`I = D & ligand_interface_mask`. Raw in-pass log probabilities define:

- `designed_sequence_confidence = exp(-mean(nll[D]))`;
- `ligand_interface_confidence = exp(-mean(nll[I]))`; and
- `sequence_recovery = mean(sampled[D] == input[D])`.

All three candidate Scores are unitless, prefer higher, and carry exact canonical
component-reference scopes. An empty scope retains the key with `NaN`; no epsilon
turns absence into perfect confidence. The shared `DesignResult` is complete, so
LigandMpnn adds no subtype. Candidate systems are coordinate-free. They preserve
untouched entities, instances, and resolvable stored bonds, and partition changed
copies within each original entity by complete output component sequence exactly
as frozen for ProteinMpnn; entities from different originals never merge.

The one-pair concrete Pipeline exposes only the released biological operations:

~~~python
class LigandMpnnPipeline:
    def __init__(
        self,
        *,
        model: LigandMpnnModel,
        processor: LigandMpnnProcessor,
    ) -> None: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult: ...

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...
~~~

Each Pipeline method processes once and invokes one matching Model task method.
`ligand_interface_cutoff` belongs only to design, whose Result publishes an
interface-scoped confidence. Probability Results contain neither interface mask nor
interface metric, so their methods do not expose an inert cutoff.
`positions` in conditional and unconditional evaluation determines which source
rows count as fixed side-chain context; the released algorithm still evaluates
every Model row and the Result retains `design_mask` so the caller can select its
intended scope. The released single-residue path ignores symmetry, so those two
Pipeline methods expose no tie argument. Tensor-level ranks remain on direct Model
methods rather than leaking tensor row identity into the biological Pipeline. All
four methods own `num_decoding_orders`; execution chunking never changes `E`.

Conversion loads with `weights_only=True` and `mmap=True`, identifies the unique
context-layer schema, derives the one live width, layer counts, and `num_neighbors`
from state and metadata, writes `num_context_atoms` only to Processor config,
preserves serialized tensor dtype, translates every live key, strict-loads the
flat Model with `assign=True`, and saves the paired concrete Pipeline artifact. It
does not preserve the upstream `backbone.` prefix, expose a dtype conversion
option, store training noise metadata as runtime config, or add artifact versions.
The pair and Pipeline use exact ID `ofoldx.ligandmpnn`. No publication declaration
is made until all four active checkpoints strict-load and existing real-weight
parity evidence covers every public operation, context branch, sampling support,
tied-group invariant, and accepted intentional divergence. The on-demand parity
script and observations do not enter frequent CI or committed package history.

#### 4.8.1 LigandMpnn design-and-pack composition

Status: frozen — the released combined command, the independent Packer contract in
Section 4.6, all candidate × pack axes, three distinct context controls, selected-
only/full repacking, RNG order, and an independent subtraction review support this
one of four explicit live compositions. It freezes no aggregate repository,
generic workflow, shared composition base, or designer mode.

The composition holds the two already assembled public child Pipelines:

~~~python
class LigandMpnnDesignAndPackPipeline:
    def __init__(
        self,
        *,
        designer: LigandMpnnPipeline,
        packer: MpnnPackerPipeline,
    ) -> None: ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[DesignResult, tuple[PackResult, ...]]: ...
~~~

The constructor accepts child Pipelines rather than four loose Models/Processors or
a component mapping. Selecting this class already selects design plus packing, so
there is no `pack_side_chains` Boolean. Design and packing retain three genuinely
different controls: `use_atom_context` affects nonprotein context in the sequence
designer, `use_side_chain_context` affects fixed protein residue context in that
designer, and `use_packing_atom_context` affects the independent Packer's
nonprotein context. They are not aliases.

A path is parsed once through the shared public MPNN reader. The designer completes
all candidates first. For each candidate, the composition reads the complete
protein component-ID sequence of every canonical instance; it never passes logits,
probabilities, ranks, interface masks, feature tensors, a PreparedInput, or a file
between stages. The Packer independently realizes each target identity and lowers
its own 16-atom context rather than reusing LigandMpnn's 25-atom features.

With `repack_everything=False`, packing positions are exactly the designer's
canonical `D = valid_mask & design_mask` scope. A deterministic singleton allowed
identity is still a designed position and is packed. With `True`, the composition
passes `positions=None` to pack every residue the Packer checkpoint can express;
unsupported modified residues are not reinterpreted as packable. No second public
packing-position argument competes with this released choice. Unselected source
sites and backbone coordinates remain exact under the Packer contract.

For `N` candidates and `P` packs, `packs[i].predictions[j]` is candidate `i`, pack
`j`. The public return is directly unpacked as `designs, packs`; a two-field family
Result would add a noun without adding information. Computation retains the
released efficient order: all design draws, then pack 0 for all candidates, pack 1
for all candidates, and so on. Results are transposed only after computation to
the candidate-major association above.

A supplied `seed` creates one operation generator after both child Models have
already been constructed and loaded. That generator is passed through every design
draw and then every pack draw without restart or stage-derived seeds. A supplied
generator is advanced in place. The composition does not touch Python or NumPy RNG.
Callers who require independent stage streams invoke the two child Pipelines
explicitly.

`to(...)` delegates the same ordinary request to `designer` and `packer`, updates
in place, and returns `self`. It does not discover components, move stages during a
call, roll back, infer autocast, or preserve hidden dtypes. Experienced callers may
place and invoke child Pipelines independently instead.

This class is live-only. Neither child Model–Processor pair owns the other's
scientific output, and the Packer is shared by five released designers. The class
therefore has no aggregate `save`, architecture ID, Pipeline marker, `pipeline.json`,
or component manifest and is not a `pipeline(task, model=...)` target. Each child
retains its ordinary Hub artifact. A future request for one aggregate Hub ID is a
new Tik rather than permission to invent a multi-pair loader now.

### 4.9 MembraneMpnn label-conditioned sequence-design vertical

Status: frozen — the global-label
`global_label_membrane_mpnn_v_48_020.pt` and per-residue-label
`per_residue_label_membrane_mpnn_v_48_020.pt` artifacts, LigandMPNN revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de`, both design and score command paths,
public Model/Module methods, all four probability computations, membrane label
lowering, decoded outputs, and exact state schemas support one Model and two
biological pairs. The two checkpoints have identical 123-key names and shapes but
different tensor values. State shape cannot determine their biological subtype.

Both artifacts execute one graph:

~~~python
class MembraneMpnnConfig(Config):
    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 48
    dropout: float = 0.0

@dataclass(eq=False)
class MembraneMpnnInput:
    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    valid_mask: Tensor
    residue_indices: Tensor
    chain_indices: Tensor
    membrane_labels: Tensor
    design_mask: Tensor
    structure: Structure
    component_references: tuple[ComponentReference, ...]
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    logit_bias: Tensor | None = None
~~~

Raw forward and sampling return the shared `MpnnOutput` and
`MpnnSamplingOutput`; membrane labels change the input graph, not the output
meaning.

`membrane_labels` is `torch.long [1, L]` with fixed class values `other=0`,
`interface=1`, and `buried=2`. These classes are learned node inputs, not a
generic condition taxonomy. The Model has no global/per-residue subtype field,
mode object, or empty Processor config.

`tied_group_ids` is aligned `torch.long [1, L]`, with dense causal-unit IDs in
first-row order, equal IDs for transitive tied members, and a distinct ID for each
untied row. The Processor performs closure; the Model hot path receives no nested
Python groups or host-side list conversion. `tied_weights`, when present, is
aligned `[1, L]`; `None` selects coefficient one.

The distinct scientific signatures require two Processors:

~~~python
class GlobalMembraneMpnnProcessor:
    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: ...

    def __call__(
        self,
        input: Structure | str | PathLike[str],
        *,
        transmembrane: bool = False,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> MembraneMpnnInput: ...

class PerResidueMembraneMpnnProcessor:
    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: ...

    def __call__(
        self,
        input: Structure | str | PathLike[str],
        *,
        membrane_labels: Mapping[
            ComponentReference,
            Literal["other", "interface", "buried"],
        ] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> MembraneMpnnInput: ...
~~~

Neither Processor declares `config_class` or exposes an empty `.config`; there is
no persistent lowering axis. Concrete fresh construction uses the ordinary
constructors above, while a repository architecture ID selects the exact pair.
The global Processor broadcasts label zero when `transmembrane=False` and label
one when true. It never emits buried. The per-residue Processor maps absent keys
and explicit `other` to zero, `interface` to one, and `buried` to two. A single
mapping makes the upstream contradictory interface-plus-buried annotation
unrepresentable without a validator or label-spec wrapper.

Both path arms are exactly `processor(read_pdb(path), ...)` using the shared reader
from Section 4.6; chain filtering is expressed by composing
`read_pdb(path, chains=...)`. Protein row selection, MSE-to-X lowering, occupancy,
validity, residue indices, chain labels, selection, realization, hard support, and
global-plus-residue bias are exactly the Structure-arm semantics frozen for
LigandMpnn, without atom context or an interface mask.

The one root Model owns the released reusable membrane feature Module directly:

~~~python
class MembraneMpnnModel(ofoldx.Model):
    def encode(
        self,
        input: MembraneMpnnInput,
        *,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        input: MembraneMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnOutput: ...

    def sample(
        self,
        input: MembraneMpnnInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnSamplingOutput: ...

    def autoregressive_probabilities(
        self,
        input: MembraneMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def sequence_free_autoregressive_probabilities(
        self,
        input: MembraneMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def conditional_probabilities(
        self,
        input: MembraneMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def unconditional_probabilities(
        self,
        input: MembraneMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def design(
        self,
        input: MembraneMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult: ...
~~~

Every operation is the corresponding LigandMPNN updated algorithm without atom or
side-chain context. `forward` is one differentiable sequence-conditioned
evaluation. The four named probability operations retain the same exact
evaluation and rank axes frozen in Section 5.4.2. Sampling returns
`sequences [N, L]`, raw untempered and unbiased
`log_probabilities [N, L, 21]`, adjusted generatable
`sampling_probabilities [N, L, 20]`, and ranks `[N, L]`. Fixed rows have zero raw
and sampling arrays; X may condition the graph but is never generated.

For each tied group, dense `tied_weights` contains final signed coefficients in
`sum(tied_weight[i] * logits[i])`. Missing coefficients are one, the released
membrane default. Coefficients are not divided by group size or normalized by
their sum; explicit `1 / len(G)` values express homo-oligomer averaging. Bias is
group-constant and applied once, hard support intersects last, fixed members
anchor exact identity, and tied members have equal ranks.

The two Pipelines expose the same operations while retaining their distinct label
arguments. Their exact signatures are:

~~~python
class GlobalMembraneMpnnPipeline:
    def __init__(
        self,
        *,
        model: MembraneMpnnModel,
        processor: GlobalMembraneMpnnProcessor,
    ) -> None: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        transmembrane: bool = False,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult: ...

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        transmembrane: bool = False,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        transmembrane: bool = False,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        transmembrane: bool = False,
        positions: Sequence[ComponentReference] | None = None,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        transmembrane: bool = False,
        positions: Sequence[ComponentReference] | None = None,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

class PerResidueMembraneMpnnPipeline:
    def __init__(
        self,
        *,
        model: MembraneMpnnModel,
        processor: PerResidueMembraneMpnnProcessor,
    ) -> None: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        membrane_labels: Mapping[
            ComponentReference,
            Literal["other", "interface", "buried"],
        ] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult: ...

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        membrane_labels: Mapping[
            ComponentReference,
            Literal["other", "interface", "buried"],
        ] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        membrane_labels: Mapping[
            ComponentReference,
            Literal["other", "interface", "buried"],
        ] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        membrane_labels: Mapping[
            ComponentReference,
            Literal["other", "interface", "buried"],
        ] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        membrane_labels: Mapping[
            ComponentReference,
            Literal["other", "interface", "buried"],
        ] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult: ...
~~~

The repeated spelling reflects two concrete biological APIs, not a shared Pipeline
base or runtime request object. Tensor-level rank arguments remain on direct Model
methods.

For every candidate, `D = valid_mask & design_mask`. Raw in-pass log probabilities
define unitless higher-preferred
`designed_sequence_confidence = exp(-mean(nll[D]))` and
`sequence_recovery = mean(sampled[D] == input[D])`, each with exact component
scope. An empty scope retains both keys with `NaN`. Neither membrane Pipeline
publishes the upstream duplicate value mislabeled ligand confidence.

State conversion loads with `weights_only=True` and `mmap=True`, requires the
416-wide membrane edge schema and live three-class node projection, preserves
serialized dtype, translates every tensor, and strict-loads with `assign=True`.
Because global and per-residue state schemas are indistinguishable, the public
entry points in `ofoldx.models.membranempnn.convert` are explicitly:

~~~python
def convert_global_membrane_mpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> None: ...

def convert_per_residue_membrane_mpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> None: ...
~~~

Neither guesses from a path, filename, or tensor shape. They save the matching
concrete Pipeline marker and expose no dtype-conversion argument. The
registration/Pipeline IDs are exactly `ofoldx.global_membrane_mpnn` and
`ofoldx.per_residue_membrane_mpnn`.

No membrane publication declaration is made until both artifacts strict-load and
existing real-weight parity evidence covers both explicit converters, label
lowerings, every public operation, 20/21 support, ties, metrics, save/load, and
accepted intentional divergences. The on-demand parity script and observations do
not enter frequent CI or committed package history.

Shared ownership stops at concrete graph primitives in `ofoldx.modules.mpnn`: the
gather functions, `MpnnProteinFeatures`, relative-position, feed-forward, message,
encoder, and decoder Modules; the public MPNN reader lives in `ofoldx.io`
and the amino-acid/element/water facts in `ofoldx.data`. `CaProteinMpnnFeatures` remains Protein-owned,
`LigandMpnnFeatures` remains Ligand-owned, and `MembraneMpnnFeatures` remains
Membrane-owned and public because its live three-class node projection is a
structure-domain difference. No shared Model, Input, Config, sampler, Pipeline,
Result hierarchy, mode vocabulary, or checkpoint-prefix-only Backbone follows
from those primitives. The broad `ofoldx.modules.message_passing` owner is deleted
after its remaining consumers import the one shared concrete owner; code files do
not define `__all__`.

#### 4.9.1 GlobalMembraneMpnn design-and-pack composition

Status: frozen — the released `global_label_membrane_mpnn` combined-command route,
its whole-structure Boolean, Section 4.6's batch Packer handoff, selected-only/full
repacking, candidate × pack association, and exact RNG order support this explicit
live composition.

~~~python
class GlobalMembraneMpnnDesignAndPackPipeline:
    def __init__(
        self,
        *,
        designer: GlobalMembraneMpnnPipeline,
        packer: MpnnPackerPipeline,
    ) -> None: ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        transmembrane: bool = False,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[DesignResult, tuple[PackResult, ...]]: ...
~~~

The composition parses a path once with `ofoldx.io.read_mpnn_pdb` and gives
the resulting `Structure` to both child Pipelines. `transmembrane` remains the
global designer's biological input; it is not translated to a per-residue label
mapping or a composition mode. With `repack_everything=False`, the Packer receives
the designer's exact canonical `D = valid_mask & design_mask` references; with
`True`, it receives `positions=None`. `use_packing_atom_context` controls only
`packer.pack_sequences(..., use_atom_context=...)`.

Design advances one seed-created or caller-supplied generator first. Packing then
advances the same stream pack-major across all candidates, and returned element
`packs[i]` owns the packs for design candidate `i`. `to(...)` delegates directly
to `designer` and `packer`. The class is live-only and owns no save, registration,
marker, manifest, loader hook, or generic composition parent.

#### 4.9.2 PerResidueMembraneMpnn design-and-pack composition

Status: frozen — the released `per_residue_label_membrane_mpnn`
combined-command route, its component-aligned three-class labels, Section 4.6's
batch Packer handoff, selected-only/full repacking, candidate × pack association,
and exact RNG order support this distinct explicit live composition.

~~~python
class PerResidueMembraneMpnnDesignAndPackPipeline:
    def __init__(
        self,
        *,
        designer: PerResidueMembraneMpnnPipeline,
        packer: MpnnPackerPipeline,
    ) -> None: ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        membrane_labels: Mapping[
            ComponentReference,
            Literal["other", "interface", "buried"],
        ] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[
            AllowedResidueIdentities | TiedResidueIdentities
        ] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[DesignResult, tuple[PackResult, ...]]: ...
~~~

The path, selected-position, target-sequence, candidate association, generator,
placement, and live-only ownership semantics are exactly those in Section 4.9.1.
The irreducible difference is the public `membrane_labels` mapping: it is passed
unchanged to `PerResidueMembraneMpnnPipeline` and never collapsed to a global
Boolean. That difference proves a second concrete class rather than a mode on the
global composition. Neither membrane composition adds a second packing-position
argument, composite Result, component mapping, generic helper, or serialized root.

### 4.10 NaMpnn mixed-polymer sequence-design pair

Status: frozen — a follow-up audit and adversarial review of NA-MPNN revision
`aafa32c87b65046a02db152f6e6235a7078e5fef`, both released checkpoints, inference,
training objectives, preprocessing, evaluation, and output paths resolve every
open boundary below and support the complete `ofoldx.nampnn` Model, Processor, and
Pipeline. The family is a registered `MODELS.register`/`PIPELINES.register` pair so
`convert -> save -> AutoModel/Pipeline` round-trips; contrary upstream fact or
execution evidence reopens only this subsection.

The two released checkpoints are one 33-token architecture with different learned
weights and released sampling defaults, not design and specificity Model variants.
One shared `feature_size` node/edge projection width and one independently live
`hidden_size` survive; the upstream node/edge feature-width pair is false because
its constructor swaps those arguments, so `NaMpnnConfig` carries a single
`feature_size` and never a node/edge pair. Model config owns checkpoint-shape
facts only: `vocab_size`, `hidden_size`, `feature_size`, layer counts,
`k_neighbors`, `dropout`, positional/RBF widths, `num_polymer_types`,
`num_atom_types`, and `include_predicted_na_n`. Temperature, neighbor override,
protein/DNA/RNA coordinate noise, causal ranks, protein-first teacher forcing,
sample count, signed tied weights, and seed or generator are operation arguments;
`decode_protein_first` and the three polymer noises live only on the training
`forward`. `NaMpnnProcessorConfig` persists only the released `shared_nucleic_tokens`
choice; `rna_as_dna` is a training-data transformation, not an inference setting.

Exact `TiedResidueIdentities` and NA-MPNN token symmetry are different operations.
In shared-token mode, DNA and RNA rows may sample the same token while realizing as
different chemical components. The concrete Processor therefore takes a direct
family-owned `symmetry_groups: Sequence[Sequence[ComponentReference]]` argument
rather than another Condition subtype or a generic symmetry object. Exact ties,
token symmetry groups, same-entity shared instances, and bonded fixing all lower
into one dense `token_group_ids[B, L]`; exact identity ties may not cross polymer
types, while token symmetry may, and validity then follows from the intersection
of each row's hard token support. Aligned `tied_weights[B, L]` and
`residue_logit_bias[B, L, 33]` are ordinary zero/one tensors, `allowed_token_mask
[B, L, 33]` is the polymer hard support intersected with the global omissions, and
the dominant `pair_logit_bias[B, L, 33, L, 33]` adjacent-token tensor stays `None`
when absent. The single `NaMpnnInput` owns these aligned tensors together with the
one target `Structure`, its `component_references`, `polymer_types`, and
`conditions`; graph operations touch only its tensors.

The simultaneous bias equation is permutation-invariant and preserves the official
combination without copying its member-order defect. For one causal unit, the
group logit at candidate token `a` sums each member's signed
`tied_weights`-scaled neural logit, adds `global_logit_bias[a]` once, sums
`residue_logit_bias` over every member target row, and sums the directed
`pair_logit_bias` contribution over every member: external context rows use the
pre-rank-block sampled state, `self` stays at PAD, and another internal member is
evaluated at the shared candidate token, so both directions of a mirrored internal
pair contribute without averaging or deduplication. Hard support intersects last,
then temperature and one group draw apply. A singleton reduces exactly to the
released untied equation, so symmetry is driven entirely from the dense
`token_group_ids`.

The released route uses physical CA/C1-prime rows, occupancy and missing-atom
demotion to an invalid unknown token via `Structure.occupancy` and
`Structure.coordinate_mask`, and derives `residue_indices` from
`Structure.residue_numbers` with the upstream running-count renumber for repeats
and insertion codes rather than canonical component order; RNA/DNA identity comes
from the canonical component, never observed O2-prime presence. The training
`forward` returns `MpnnOutput` and takes `decode_protein_first` and the three
polymer noises, while every one-target inference operation rejects a batch axis.
Candidate realization partitions copies within each original entity by the
complete realized sequence into distinct entities, never merges original entities,
and never raises on divergent shared instances. The Processor is a two-arm boundary:
a design/probability arm lowers conditions, symmetry, and bias, and a score arm
lowers an explicit observed-sequence override.

The Model publishes `sample -> MpnnSamplingOutput`, `score` (sequence-conditioned
autoregressive over `num_decoding_orders`) and `unconditional_probabilities`
(sequence-free all-equal-rank) returning the shared `MpnnProbabilityResult`,
`design -> DesignResult`, and `specificity_profile -> ScoreResult`. The family
comparison still proves only shared public Mpnn graph primitives and the shared
`MpnnOutput`, `MpnnSamplingOutput`, and `MpnnProbabilityResult` carriers; it proves
no shared Model, Input, Config base, sampler, request, or family runtime value. The
`nll_loss` and polymer-specific `smoothed_loss` objectives are required scientific
functions, while dataset and Danling Runner composition receives a later Tik rather
than leaking training provenance into Model or Processor config.

## 5. Results

### 5.1 Shared scientific values

Status: frozen — this bounded carrier contract is supported by decoded outputs from
AF2, ESMFold2, Protenix, ProteinMPNN, LigandMPNN, NA-MPNN, and Boltz-2, depends on
the canonical identity slice in Section 3.4.1, and survived an independent
counterreview. It freezes only Score, Metric, PairMetric, their canonical
references, and producer ownership of alignment. Family-composite Results,
variable-topology design, and multi-seed grouping remain reopened. PackResult is
frozen separately in Section 5.5.

The shared FoldResult, fixed-length DesignResult, and ScoreResult are the exact task
Results whenever they can represent a released decoded output without loss. A
standard Pipeline returns its Model Result unchanged. Direct Model task methods
and concrete family Pipelines use the shared types whenever their decoded output
fits; a family Result exists only when a released complete operation has
irreducible decoded output that the shared type cannot carry honestly. The
top-level verbs and `pipeline(...)` remain dynamically typed as described in
Section 2. Result is a collective word, not a required nominal base class or
Protocol.

One metric reference names one level of canonical identity:

~~~python
MetricReference = str | ComponentReference | AtomSite
~~~

A string is an exact instance_id. ComponentReference is an (instance_id, component_index) pair for either a PolymerSequence monomer or a Molecule component occurrence. AtomSite is an exact atom. This instance → component → atom hierarchy is sufficient for atom, residue, ligand-token, chain, and chain-pair outputs; there is no InstanceSite or generic axis descriptor.

The shared scientific values are:

~~~python
@dataclass(eq=False)
class Score:
    value: float
    unit: str | None = None
    preferred: Literal["higher", "lower"] | None = None
    scope: frozenset[MetricReference] | None = None
    evaluations: np.ndarray | None = None

@dataclass(eq=False)
class Metric:
    values: np.ndarray
    axis: tuple[MetricReference, ...]
    unit: str | None = None

@dataclass(eq=False)
class PairMetric:
    values: np.ndarray
    row_axis: tuple[MetricReference, ...]
    column_axis: tuple[MetricReference, ...]
    unit: str | None = None
~~~

Score is one named scalar scientific value. scope=None means the whole containing
target; a frozenset names the exact unordered canonical subtarget to which the
score applies, and an empty frozenset is an explicitly empty scope. Scope is output
meaning, not a generic `score(..., scope=...)` input language. `evaluations` is a
one-dimensional CPU NumPy array preserving released repeated scalar evaluations in
actual execution order; `None` means that operation publishes no repeated
evaluation, while a length-one array is still one published evaluation. In the
released MPNN score protocols, its values are the per-decoding-order scores and
`value` is their released aggregate, currently the arithmetic mean. They are real
scientific evaluations, not a diagnostic tensor or a fabricated uncertainty
field. Other families define their own released aggregate; the framework does not
silently recompute it. A repeated stochastic evaluation is not mislabeled as
calibrated uncertainty.

Metric is exactly a one-dimensional numerical array aligned to one canonical axis. PairMetric is exactly a two-dimensional array aligned to two canonical axes. Mixed reference levels are valid when the released tokenization genuinely mixes them, such as polymer-component and atomized-ligand tokens. Raw logits, vocabulary axes, mixture components, trajectories, timesteps, layout indices, duplicate coordinates, and run controls cannot enter these values; they remain in typed raw Model or expert outputs.

Task Results contain only decoded canonical values. Implementations transfer only
returned values to detached CPU NumPy arrays and do not retain autograd graphs.
They do not copy arrays merely to enforce deep immutability, wrap mappings in
read-only proxies, define custom numerical equality, or constrain runtime values
to persistence-friendly dtypes. `eq=False` avoids NumPy's ambiguous automatic
boolean equality; there is no custom equality. NaN and infinity are model
observations and remain visible rather than causing a Result constructor to discard
an expensive inference.

These dataclasses are plain named carriers with no `__post_init__`, target traversal,
shape, duplicate, dtype, finiteness, or membership scans. A family decoder creates
values and references from the same canonical layout and owns their alignment.
Family parity scripts provide evidence for that mapping without entering frequent
CI or committed package history. A frozen family may require existing real-weight
parity evidence before making its own publication declaration, as Sections
4.7–4.9 do for the completed MPNN families; this is a family qualification, not
generic automation or a framework lifecycle.

Common names have one cross-family scale: plddt and mean_plddt use 0–100; PAE and PDE use Å; pTM, ipTM, and probabilities use 0–1. Family READMEs define additional names and units. These conventions are mapping requirements, not another runtime scan.

Conditions and runtime controls remain inputs and are not copied into Results. A
released condition-related measurement is returned under its actual Score or
Metric name. Repository source, mutable Model history, backend, placement, seed,
Generator state, and composition history do not become a generic metadata bag.

### 5.2 FoldResult

Status: frozen — this bounded flat result is supported by AF2, ESMFold2, and
Protenix single-invocation outputs. AF3 multi-seed grouping and family-composite
folding remain reopened.

~~~python
@dataclass(eq=False)
class FoldPrediction:
    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    pair_metrics: dict[str, PairMetric] = field(default_factory=dict)

@dataclass(eq=False)
class FoldResult:
    predictions: tuple[FoldPrediction, ...]
~~~

Predictions belong to one seed or other common family sampling context and are
generation ordered; tuple position is sample identity. Each prediction owns the
Structure produced by that family decoder. This Result slice does not freeze the
still-reopened Structure/site-decoding or I/O contracts. Common values use the same
typed mappings as every other decoded scientific value: prediction.metrics["plddt"],
prediction.pair_metrics["pae"], prediction.pair_metrics["pde"], and
prediction.scores["ptm"].

FoldPrediction carries no auxiliary per-seed or per-sample index fields. Ordering
already identifies samples within this one invocation. A family such as AF3 that
returns several seed groups owns a concrete result preserving those groups rather
than flattening them or adding generic fields here.

`FoldResult.best(score)` performs explicit ranking using the named Score's preferred
direction. It has no default score and no implicit first or best property. Missing
values fail through ordinary lookup. Undefined or conflicting directions,
inconsistent units or scopes, and NaN values fail when ranking is requested, not
when the Result is constructed. Units and frozenset scopes must match exactly and
are never normalized or converted. Infinity remains rankable, and a tie returns
the earliest prediction. The short operation is written directly; it does not
justify a private generic rank helper, Protocol, mixin, or base Result.

### 5.3 Fixed-length inverse-folding DesignResult

Status: reopened — the common fixed-length carrier is frozen below, while concrete
ProteinMPNN, LigandMPNN, and NA-MPNN family results and score mappings remain part
of their family Tik. Variable-topology design and ODesign's released M × S × K
output remain separate investigations.

#### 5.3.1 Shared fixed-length carrier

Status: frozen — ProteinMPNN, LigandMPNN, and NA-MPNN all publish
generation-ordered, coordinate-free fixed-length designs. This slice freezes only
their common carrier and ranking behavior; it does not claim to represent
RFdiffusion, ODesign, PXDesign, BoltzGen, packing, or any coordinate-generating
design operation.

The values needed by this bounded cycle are:

~~~python
@dataclass(eq=False)
class DesignCandidate:
    system: MolecularSystem
    scores: dict[str, Score] = field(default_factory=dict)

@dataclass(eq=False)
class DesignResult:
    candidates: tuple[DesignCandidate, ...]
~~~

ProteinMPNN, LigandMPNN, and NA-MPNN publish sequence design without predicted
coordinates. Each generation-ordered candidate therefore owns a complete
coordinate-free `MolecularSystem`, including unchanged instances and stored bonds.
Fixed-length replacement preserves every `ComponentReference`; the decoder applies
its family-owned output partition, so the candidate system is its own complete
occurrence mapping and needs no generated identity map. The shared Result does not
infer or prescribe that partition from value equality. Candidate scores belong to
that candidate system. Fixed-length inverse-folding scopes contain component
references; this is an ownership rule for these family Results, not a narrower
definition of shared `Score`. Vocabulary-axis probabilities, logits, decoding
orders, and other tensor diagnostics remain in the concrete raw Model output rather
than being forced into canonical metrics.

`DesignResult.best(score)` has the same explicit direction, exact unit, identical
frozenset scope, NaN, infinity, and earliest-tie behavior as `FoldResult.best`.
There is no default. Comparing different residue subsets as though they were
commensurate fails at ranking time. The implementation is another short direct
method rather than a shared private ranking abstraction.

The shared result has no input, input-score, context, placement, generated-map, or
provenance field. A family whose one invocation has irreducible additional output
owns a concrete public subtype or result; it does not make every design consumer
carry an optional generic field.

#### 5.3.2 Family score mappings

Status: frozen — the exact ProteinMPNN and LigandMPNN revisions, all active
Protein/Ligand/Membrane design commands, decoded scores, empty-scope behavior, and
the frozen family contracts in Sections 4.7–4.9 support the mappings below. This
freezes no NaMpnn result or wider shared carrier.

ProteinMPNN uniquely publishes native input scores during the same design
invocation. ProteinMpnn preserves them without changing candidate RNG or
recomputing a later score call. Its family-specific Result is an ordinary
constructible dataclass:

~~~python
@dataclass(eq=False)
class ProteinMpnnDesignResult(DesignResult):
    input_score: ScoreResult
~~~

That `ScoreResult.target` is the exact conditioning Structure, so every scope owns
its geometry without adding input fields to the shared `DesignResult`.
`designed_sequence_nll` and `global_sequence_nll` retain the ordered repeated
decoding-order arrays in `Score.evaluations`; `Score.value` is their released
arithmetic mean. Each generated candidate has its own single
`designed_sequence_nll`, `global_sequence_nll`, and `sequence_recovery`, so those
candidate Scores have `evaluations=None`. The word global follows the released
model and still receives an explicit scope containing all valid protein tensor
rows; `scope=None` would incorrectly include ligands or unobserved residues in a
heterogeneous target. Both NLL values use `unit="nat/residue"` and
`preferred="lower"`; sequence recovery has no unit and uses `preferred="higher"`.

Candidate likelihood values use the raw, untempered, unbiased Model
`log_softmax`, not the temperature- or bias-adjusted sampling distribution. Let
`D` be selected valid design rows and `I` the LigandMpnn ligand-interface subset.
LigandMpnn and both membrane Pipelines publish:

- `designed_sequence_confidence = exp(-mean(nll[D]))`, scoped to `D`, unitless,
  and preferred higher; and
- `sequence_recovery`, scoped to `D`, unitless, and preferred higher.

LigandMpnn additionally publishes unitless,
higher-preferred `ligand_interface_confidence = exp(-mean(nll[I]))`, scoped to `I`.
Its interface uses the released strict `< cutoff` test, normally 8 Å, from the
virtual Cβ to the nearest parsed context atom. Membrane publishes no interface
score. For protein, soluble, and membrane upstream routes, the field printed as
`ligand_confidence` is evaluated on the same `D` mask as overall confidence and is
exactly redundant; the canonical Result does not duplicate it as a second
scientific Score. ProteinMpnn instead retains its more informative
`designed_sequence_nll`; its updated-route confidence is the exact derived value
`exp(-designed_sequence_nll)` and is not duplicated.

An empty selected or interface scope remains an ordinary undefined mean: the score
key is present, `scope=frozenset()`, and `value=NaN`. No epsilon turns absence into
perfect confidence, and no special exception or fallback is added. The released
LigandMPNN behavior is recorded as an upstream bug.

#### 5.3.3 NaMpnn score investigation

Status: frozen — the NaMpnn family Tik in Section 4.10 froze sampling support,
pair-bias composition, complete Pipeline outputs, and the score mapping below.

NA-MPNN candidates publish the same unitless, higher-preferred
`designed_sequence_confidence` and `sequence_recovery` meanings over selected valid
rows `D`, computed from the raw untempered unbiased Model `log_softmax` rather than
the sampling distribution. The released default may include protein and
nucleic-acid rows; an explicit NA-only design mode restricts `D` accordingly.
ProteinMPNN is the only one of these released design commands that also publishes
scalar input sequence scores; the NaMpnn `score` and `unconditional_probabilities`
paths return the typed `MpnnProbabilityResult` instead of fabricating a shared
input-score field.

An empty NaMpnn selected scope likewise retains its keys with `NaN`: the score key
is present, `scope=frozenset()`, and `value=NaN`, and the released epsilon behavior
is recorded as an upstream bug.

NA-MPNN's released specificity profile averages its controlled sampling
probabilities across samples after temperature, hard omissions, and biases. It is
not duplicated on every candidate. The minimal candidate is a separate
`specificity_profile(...) -> ScoreResult` whose one-dimensional Metrics are named
by canonical DNA or RNA base probability and whose axes contain selected NA
`ComponentReference` values only. Raw 33-token arrays and per-sample profiles stay
in the typed sampling output. The polymer-specific `allowed_token_mask` hard
support keeps probability mass off chemically incompatible token classes, so the
averaged base Metrics are well defined.

Coordinate-producing and variable-topology families investigate concrete family
Results from their own released hierarchy and topology evidence. They do not widen
the frozen shared `DesignCandidate` or `DesignResult` with unions, optional input,
metrics, pair metrics, placement maps, or generic context fields.

### 5.4 ScoreResult

Status: reopened — the shared coordinate-conditioned carrier is frozen below.
Boltz-2 composition and remaining concrete family mappings remain in family Tik;
the MPNN aligned probability carrier is frozen separately below.

#### 5.4.1 Shared coordinate-conditioned carrier

Status: frozen — ProteinMPNN's released scalar outputs and Boltz-2's released
coordinate-conditioned scalar outputs both own the exact Structure whose
coordinates condition their canonical values. The complete ProteinMpnn mapping is
frozen in Section 4.7 without changing this independently proven common carrier.
LigandMpnn, MembraneMpnn, and NaMpnn probability operations are explicitly
excluded: their vocabulary and decoding-order axes remain concrete typed tensor
outputs rather than scalar `ScoreResult` values. No audited consumer of this
carrier currently returns a canonical pair matrix or a coordinate-free target.

~~~python
@dataclass(eq=False)
class ScoreResult:
    target: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
~~~

Target is the Structure on which the returned scores were evaluated. This Result
carrier does not itself prove the still-reopened Structure/site-decoding contract.
The target is not necessarily the caller's input: an operation may generate and
select a Structure before evaluating it. ProteinMPNN FASTA scoring replaces the
sequence in the system while retaining the backbone geometry actually used, so its
target is that scored Structure rather than the unmodified input PDB. There is no
coordinate-free union, duplicated system, optional Structure, or pair-metric field
without a consumer. The producer constructs every metric reference and Score scope
against `target.system`; MPNN designed and global scores may coexist with different
scopes.

The existence of a CLI path named score does not select this carrier. A concrete
operation returns `ScoreResult` only when its decoded scientific result consists of
canonical scalar Scores or Metrics over one coordinate-conditioned target.
LigandMpnn's, MembraneMpnn's, and NaMpnn's released conditional, unconditional,
and autoregressive probability operations instead retain vocabulary and, where
scientific, decoding-order axes; their concrete family outputs must not be averaged
merely to fit this shared Result.

#### 5.4.2 MPNN aligned probability Result

Status: frozen — ProteinMPNN revision
`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, LigandMPNN revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de`, the Protein, Ligand, global-membrane,
and per-residue-membrane score commands, all four public computations, and their
released repeated tensor outputs support one concrete carrier with four real
Model consumers, the fourth being the NaMpnn `score` and
`unconditional_probabilities` operations frozen in Section 4.10:

~~~python
@dataclass(eq=False)
class MpnnProbabilityResult:
    target: Structure
    component_references: tuple[ComponentReference | None, ...]
    sequence_tokens: Tensor
    valid_mask: Tensor
    design_mask: Tensor
    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor: ...

    @property
    def probabilities(self) -> Tensor: ...

    @property
    def mean_probabilities(self) -> Tensor: ...

    @property
    def probability_standard_deviation(self) -> Tensor: ...
~~~

This constructible carrier is defined in `ofoldx.results` and used directly by
`ProteinMpnnModel`, `LigandMpnnModel`, `MembraneMpnnModel`, and `NaMpnnModel`. It
is not a base class, family hierarchy, registry identity, Protocol, or generic
probability
framework. It is the typed expert tensor output expressly excluded from the
decoded NumPy `ScoreResult`/`Metric` rules in Section 5.1; differentiable Model
calls retain Tensor placement and autograd, and the concrete Pipelines return that
same scientific output without averaging it. Protein's original path may align a
transient numeric-gap row with
`None`; Structure-arm Protein, Ligand, and Membrane Results have only real
component references. Identity tensors are unbatched `[L]` values aligned to the
one target Structure.

For `E = num_decoding_orders`, sequence-conditioned autoregressive and
sequence-free autoregressive operations have `logits [E, L, 21]` and
`decoding_ranks [E, L]`. Conditional target-last and unconditional target-first
operations have `logits [E, L, 21]` and effective
`decoding_ranks [E, L, L]`, whose axes are evaluation, target, and causal row.
Ranks are `torch.long`; lower is earlier and equal values mean simultaneous tied
members. Every operation retains the leading evaluation axis even when repeated
unconditional logits happen to be equal. Execution batching is not a scientific
axis and never changes `E`.

`probabilities` is `softmax(logits, dim=-1)` and `log_probabilities` is
`log_softmax(logits, dim=-1)`. `mean_probabilities` reduces the leading evaluation
axis. `probability_standard_deviation` uses the population definition
`probabilities.std(dim=0, correction=0)`. These are derived views rather than
stored duplicate tensors. The Result never averages away the vocabulary, target,
or causal-rank axes to fit `ScoreResult`.

The four operations have no conditioning flag. Autoregressive uses native sequence
context in causal order; sequence-free autoregressive remains order-dependent;
conditional moves each target exactly last; unconditional moves each target
exactly first. Conditional and unconditional preserve the non-target relations of
their base ranks and return their exact effective ranks. The released command's
misdocumented single-residue switch therefore does not survive as compatibility
state or another method pair.

#### 5.4.3 Boltz-2 affinity composition

Status: reopened — the two-stage released/extended operation and exact ownership
of all decoded first- and second-stage outputs require a dedicated family Tik.

Boltz-2 affinity is the concrete counterexample to treating score as a scalar head
over the input. Its complete released operation has one additional decoded output,
so its concrete Pipeline returns the only necessary family Result here:

~~~python
@dataclass(eq=False)
class Boltz2AffinityResult:
    target: Structure
    structure_predictions: tuple[FoldPrediction, ...]
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
~~~

The target, scores, and metrics remain directly accessible; they are not nested in
another ScoreResult. The additional tuple
retains every decoded first-stage structure/confidence prediction and is ordered by
descending released confidence score, not generation order.

For one first-stage call, the released ranking is exactly:

~~~python
interface_score = (
    iptm
    if not torch.allclose(iptm, torch.zeros_like(iptm))
    else ptm
)
confidence_score = (4 * complex_plddt + interface_score) / 5
~~~

The all-zero iPTM decision is made once over the complete call tensor. It is not a
per-sample fallback: if any iPTM entry makes that tensor nonzero, every sample is
ranked with its iPTM entry. The Pipeline sorts all first-stage predictions by
descending confidence_score, stores that order in structure_predictions, and
passes the first prediction's canonical Structure through the root affinity
Processor.

The root affinity pass then independently samples its own candidate coordinates
and selects the candidate with maximum iPTM. This second-stage rule has no pTM
fallback. The inherited ScoreResult.target is that selected second-pass Structure,
with the complete MolecularSystem retained. Sites outside the affinity crop have
coordinate_mask=False. OFoldX does not merge first-pass coordinates into those
rows, because such a merge would present coordinates from two different
predictions as one coherent model observation.

The released Boltz-2 affinity result has exactly six Scores. Every one has
`scope=frozenset({selected_ligand_instance_id})`, and they retain the official names and
meanings:

- affinity_pred_value is the two-head ensemble affinity, including the released
  molecular-weight correction when enabled, reported as log10(IC50) for IC50 in
  micromolar; its preferred direction is lower;
- affinity_probability_binary is the arithmetic mean of the two head binder
  probabilities in [0, 1], has no unit, and prefers higher;
- affinity_pred_value1 and affinity_pred_value2 are the two individual uncorrected
  affinity-head values on the same log10(IC50) scale and prefer lower; and
- affinity_probability_binary1 and affinity_probability_binary2 are the sigmoid
  binder probabilities from the two individual heads, have no unit, and prefer
  higher.

The two ensemble values do not erase the four released component values. The
molecular-weight correction applies to affinity_pred_value, not silently to the
individual head values. The selected-ligand scope states which ligand the affinity
property describes; it is a focus, not a claim that computation is isolated from
the receptor or the rest of the retained target. Score is not synonymous with
binding affinity: every Model declares the properties it computes, their units,
ranking direction, scope, and aggregation semantics. A released repeated-evaluation
algorithm uses a precise concrete Model or family Pipeline argument, not a
framework-wide averaging rule.

### 5.5 PackResult

Status: frozen — the released MpnnPacker proves the sample-to-metric grouping and
ordinary NumPy decode boundary below. That
grouping is a domain invariant, so one real consumer is sufficient to justify the
shared carrier. A future packer uses it only when the carrier represents its
complete decoded output without loss; the routing word alone does not force it.

~~~python
@dataclass(eq=False)
class PackPrediction:
    structure: Structure
    metrics: dict[str, Metric] = field(default_factory=dict)

@dataclass(eq=False)
class PackResult:
    predictions: tuple[PackPrediction, ...]
~~~

Each prediction binds one stochastic packed Structure to the decoded metrics
produced for that exact sample. Its Structure contains the explicit target sequence
and producer-owned entity mapping. It preserves instance IDs, component
indices, assembly, resolvable topology, backbone, ligands, and every unselected
site, coordinate, mask, and supplied occupancy fact. Selected expected heavy side-chain sites
are replaced or supplemented by the packed sample; sites belonging only to a
replaced old identity are removed. The packer does not generate hydrogens and does
not materialize missing unselected atoms merely to create a complete-heavy layout.
Retained rows preserve their relative input `Structure.sites` order; generated
selected heavy atoms append in Model-row then atom14 order. Selected hydrogens are
removed because their coordinates would no longer describe the packed side chain.

Processor staging preserves source coordinate precision until the one explicit
Processor dtype conversion. Raw `forward` and `sample` tensors retain the caller's
Model computation dtype. Canonical `pack` decoding uses ordinary
`tensor.detach().cpu().numpy()` and never inserts a hidden FP32 cast. For dtypes
NumPy represents, a packed Structure uses the NumPy result type of source and
sampled coordinates; supplied occupancy is retained without downcast, generated
sites use a dtype that represents unknown occupancy, and Metric arrays retain the
sampled dtype. PyTorch BF16 has no ordinary NumPy dtype, so explicit whole-Model
BF16 may naturally fail at canonical Result decoding. OFoldX does not babysit that
caller choice or claim that every torch dtype has a NumPy-backed Result.

MpnnPacker exposes up to four residue-axis metrics named `chi1_log_density` through
`chi4_log_density`. Each axis contains only selected target residues for which that
chi exists, in Model row order. The unit is `nat/radian`; a key with an empty axis is
absent. Values come from the final mixture that produced that sample. Chi density
is never presented as a physical B-factor or another Structure annotation. The
explicit family `write_pdb(prediction, path)` boundary described in Section 4.6.1
encodes only these physical chi metrics as a documented visualization in the PDB B
column and uses zero elsewhere. Raw mixture parameters, fixed-row sentinels, and
other expert tensors remain outside the canonical Result.

The per-prediction grouping avoids parallel Structure/metric tuples. No torsion
identity type or family-specific clone is required. The released packer defines no
cross-sample score or ranking, so `PackPrediction` has no scores and `PackResult`
has no `best()` method.

### 5.6 No Result persistence protocol

V1 defines no Result.save/load API or result archive format. No production consumer currently requires one, and persistence must not force CPU copies, safetensors dtypes, codec versions, or strict schema machinery into ordinary runtime outputs. Structure exchange uses PDB/mmCIF; model repositories use the protocol in Section 7. A future result archive starts from a concrete durability or interchange consumer rather than extending the runtime dataclasses speculatively.

## 6. Pipelines

### 6.1 Standard and concrete Pipelines

Status: reopened — the classification below corrects the earlier signature-only
rule, but exact family execution structures remain subject to complete inventory.

A Pipeline is standard exactly when all four statements are true:

1. the common biological signature expresses the complete operation without loss;
2. one call to one Processor produces the input for its registered paired Model;
3. one call to that Model's matching shared task method performs the operation; and
4. the shared task Result represents the complete decoded output without loss.

The Model task method may internally run the scientific sampling loop and call raw
forward more than once; it is still one Model task invocation. A standard Pipeline
does not compose pairs, repeat task invocations, or select among invocation Results.
Its complete route is:

~~~text
biological input -> Processor -> concrete Model input -> Model task method -> Result
~~~

Its four direct-construction classes are:

~~~python
FoldPipeline(*, model, processor)
DesignPipeline(*, model, processor)
ScorePipeline(*, model, processor)
PackPipeline(*, model, processor)
~~~

Direct construction binds the supplied live objects as public model and processor
attributes. It does not change Model training state, redispatch through architecture
registration, or infer a repository source. AutoModel(pretrained) is already in
eval mode; the owner of a manually supplied Model chooses its state. The caller
supplies a compatible pair, and ordinary Processor and Model calls expose an
incompatible pair naturally.

A concrete Pipeline is required when any standard criterion is false. It may bind
one pair or several explicitly named pairs. A one-pair end-to-end Model is standard
when one task-method call, the common signature, and the shared Result are all
lossless; the same end-to-end Model uses a concrete Pipeline when its biological
signature or decoded output is family-specific. Conversely, a multi-call or
multi-pair operation is concrete even when it can return the shared Result. Model,
Module, checkpoint, or head count alone does not decide the class.

Each of the four generic standard classes owns one matching task. A concrete family
class may instead expose several public shared task methods when the same pair or
fixed composition completely implements them. ProteinMPNN, LigandMPNN, and NA-MPNN
each require one concrete class with both `design` and `score`; expert probability
methods remain additional precisely named methods on that same object. Factory task
selection never creates a task wrapper, mode field, or partial view of the class.

A standard class exposes the common-case signature of its matching top-level verb,
minus repository-loading arguments. Every concrete constructor names the Models
and Processors it actually needs. Neither accepts a component mapping, generic
options, or a graph description.

Each shared Model task method owns the tensor computation, Model-side loop,
decoding, and local inference policy for one invocation and constructs that
invocation's decoded Result. A public expert operation may instead return the exact
typed family value required by another stage.

The Pipeline owns exact invocation multiplicity and order, composition, explicit
handoffs, and selection across invocations. A handoff may be a canonical domain
value or an exact typed family value; the actual consumer decides, and neither is
hidden in a generic context, `PreparedInput`, or intermediate-Result wrapper.

When there is one Model task invocation, that method constructs and returns the task
Result unchanged; the Pipeline does not decode, rebuild, copy, or mutate it. A
Pipeline constructs a new final Result only when it combines multiple invocation
Results or typed expert outputs into the complete scientific output.

A standard Pipeline returns the shared task Result by definition. A concrete
Pipeline also returns it when its signature or execution structure, rather than its
decoded output, requires a concrete class. It returns a named family Result only
when the complete decoded output cannot fit the shared Result without loss. Dynamic
factories do not force these exact classes behind a generic Pipeline or Result
type.

Pipeline is a collective term; no public generic Pipeline base is required.

### 6.2 Repository construction and Boltz-2 affinity

Status: reopened — concrete identity registration and observable marker routing are
frozen in Section 7. One-pair repository-to-class construction is frozen there from
the complete MpnnPacker vertical. Multi-pair root ownership, repository
construction, and saving remain candidates until a complete materially different
vertical proves them.

`pipeline(task, model=...)` follows the one routing algorithm in Section 7.4. A
marker selects one complete concrete family class, not one task-specific wrapper;
that class may expose several shared verbs. The marker never supplies task metadata,
a Python import path, or a component graph. A one-pair class receives the loaded
root Model and Processor directly. How a multi-pair class receives one resolved
repository snapshot is deliberately not yet a contract; no private hook or public
lifecycle helper is implied.

AutoModel(repository) and AutoProcessor(repository) always load the root pair,
whether or not pipeline.json exists. They never return or implicitly construct a
Pipeline. AutoModel plus AutoProcessor plus an explicit standard or concrete
Pipeline constructor remains the live-object builder path. This preserves one name
and one meaning: Auto constructs one object, while pipeline constructs the
published biological call boundary.

Every repository that publishes a concrete Pipeline has one independently runnable
root pair that owns the published operation: its Model operation defines the
operation's primary scientific contract and the other pairs, if any, are explicit
dependencies of that call. The root is not an arbitrary first component chosen to
fit the file format. `pipeline.json` annotates that owning pair's repository with a
concrete biological call boundary; it is not a standalone composition manifest.

Boltz2AffinityPipeline is the concrete example that requires two ordinary pairs:

~~~python
Boltz2AffinityPipeline(
    *,
    model=affinity_model,
    processor=affinity_processor,
    structure_model=structure_confidence_model,
    structure_processor=structure_confidence_processor,
)
~~~

The root pair is the affinity pass. The structure/ repository directory holds the
structure/confidence pair. The Pipeline first runs the structure/confidence pair,
applies the whole-call all-zero-iPTM confidence formula in Section 5.4, and passes
the highest-ranked canonical Structure through the root affinity Processor. It does
not pass an upstream file, private tensor layout, PreparedInput, or generic context
between the two pairs. The root affinity Model then performs its independent
second-pass structure sampling, maximum-iPTM selection without a pTM fallback, and
six-score evaluation. The concrete Pipeline returns Boltz2AffinityResult so the
complete confidence-ranked first-stage predictions are not discarded. There is no
composite Boltz-2 Model containing both checkpoints.

The concrete score signature must name its ligand, MSA/template/Condition inputs,
and each scientifically distinct sampling count that the complete upstream
inventory proves. Their exact types, names, defaults, and which Conditions enter
which pass remain investigation; neither a generic scope argument nor one
overloaded count may hide those decisions. Processor routing is likewise still
investigation: the public ownership and per-pass lowering of MSA, templates,
constraints, method selection, and MSA subsampling must be traced before the
signature is frozen. The bounded two-stage selection and Result contract above do
not freeze those unresolved call details.

The resolved immutable Hub commit exists only while loading, so every pair in the
operation comes from one snapshot. Model, Processor, Pipeline, and Result do not
gain hidden repository source fields. Once constructed, they are ordinary mutable
Python and PyTorch objects; pretending the original snapshot still identifies
later execution would be false provenance.

The top-level verbs are one-shot conveniences over this construction. Repeated
calls keep the Pipeline. Every route explicitly maps biological arguments to the
owned Processor call and tensor controls to the owned Model task method. It does not
inspect signatures or forward an unchecked bag. A standard Pipeline omits optional
arguments left at their common empty default, so a concrete Processor declares only
the inputs it actually consumes. Supplying a nonempty unsupported argument reaches
that ordinary concrete signature and fails; Processors do not add reject-only
parameters for API symmetry.

### 6.3 Placement

Status: frozen — ordinary PyTorch placement, the existing standard and concrete
one-pair Pipeline consumers, and the four MPNN design-and-pack compositions prove
the bounded delegations below. Placement for every other multi-pair operation
remains family-owned investigation pending its actual memory and execution evidence.

Any one-pair Pipeline, standard or concrete, supports
`to(device=..., dtype=...)`. It updates in place, returns itself, and delegates the
same request to its public `model` and `processor`. Whether the Pipeline is concrete
because it has a family-specific signature or Result does not change this ordinary
one-pair behavior.

Each of `ProteinMpnnDesignAndPackPipeline`,
`LigandMpnnDesignAndPackPipeline`,
`GlobalMembraneMpnnDesignAndPackPipeline`, and
`PerResidueMembraneMpnnDesignAndPackPipeline` likewise updates in place, returns
itself, and delegates the same request to its explicitly named `designer` and
`packer` child Pipelines. None performs discovery or adds stage-wise placement
policy.

Every multi-pair Pipeline is necessarily concrete because it fails the
one-Processor, one-paired-Model route in Section 6.1. It may expose a shorthand `to`
only after the placement of that complete operation is audited. Every such
delegation is written against explicitly named child Pipelines, Models, or
Processors; no common implementation discovers components by scanning attributes.
Model dtype means
floating parameter and buffer placement. Processor dtype applies to every floating
tensor it emits rather than its internal compute dtype. One user request may supply
both while the two meanings remain distinct.

Boltz2AffinityPipeline placement remains investigation. In particular, the design
does not yet decide whether its shorthand co-locates both large pairs or whether
its public call exposes explicit stage-wise placement. This also leaves the exact
meaning of `pipeline(..., device=..., dtype=...)` open for that concrete class. No
automatic swapping, offload policy, or runtime-policy abstraction is inferred
while that question is open.

Pipeline does not scan Model parameters, infer placement, emulate transactions,
roll back partial movement, or wrap native movement failures. Sharded and offloaded
runtimes can place their components explicitly. Pipeline never derives autocast
from Processor dtype. A direct forward caller owns autocast; a task method owns only
the ordinary inference-mode scope declared by that public method.

### 6.4 No composition language

Status: frozen — standard one-pair Pipelines, family-owned concrete Pipelines, and
the four explicit live MPNN design-and-pack compositions prove this negative
framework boundary. It freezes no future aggregate repository; a request for one
must enter its own Tik with a concrete released operation.

`pipeline.json` identifies one registered concrete class; it does not serialize
composition. That class fixes its named components, execution order, explicit
handoffs, and selection rules in ordinary Python. A handoff is the canonical domain
value or exact typed family value required by the next concrete consumer; it is not
forced through a canonical-only rule or hidden in a generic intermediate container.
The marker contains no component list, path mapping, workflow nodes, version,
digest, contract fingerprint, or arbitrary options.

A concrete Pipeline is justified by failure of any part of the standard skeleton:
one common signature, one Processor call for its paired Model, one matching Model
task invocation, and one lossless shared Result. Checkpoint, Model, Module, or head
count is not a substitute for that test. One pair may therefore need a concrete
Pipeline, while an end-to-end Model that hides a complicated computation behind
one lossless task invocation may remain standard.

A released operation with one scientific root owner may publish its family-owned
concrete Pipeline under Section 7. A live composition with no single owning
Model–Processor pair can still be constructed explicitly from its components, but it
has no serialized root, no `pipeline.json`, and no implied aggregate repository
format. OFoldX does not make users express either case as a workflow or DAG.

## 7. Architecture identity, registration, and repositories

### 7.1 Identity

Status: reopened — the revision-local dispatch meaning below is the candidate
contract; its implementation waits for the registration and repository verticals
in the following subsections.

An architecture ID is an exact, revision-local dispatch key. In `model.json` it
selects one registered concrete Model–Processor pair and the Model's concrete
config class. The remaining Model config values select that Model's graph;
together the pair and config define how serialized state and raw tensor inputs and
outputs are interpreted. A Processor with persistent lowering configuration owns
`processor.json`; a config-less Processor needs no second file because the
architecture already selects its exact class. The ID is not a second
semantic-version or history system: reproduction selects the Hub revision and
corresponding OFoldX source revision.

Family names, upstream checkpoint labels, shared task words, backend choice, public
head presence, and the exact state-dict key set produced by one config are not
architecture-identity tests. An architecture ID simply names one explicitly
registered Model–Processor and Model-config construction contract in that revision.
Config-controlled graph choices, heads, and tensor extents remain config when the
same stable typed input and output contracts express them; AlphaFold 3's
`num_pae_bins` is a concrete example. A genuinely different concrete input or
output type may require another registered pair, but registration does not infer
that decision from tensor shape, a head checklist, tasks, variants, or keys.

A concrete Pipeline ID is separate routing identity for `pipeline.json`; it never
replaces the root architecture ID in `model.json`. A family remains documentation
and source ownership rather than a dispatch layer. Family documentation inventories
released checkpoints and scientific behavior without turning those labels into a
framework identity hierarchy.

### 7.2 Explicit registration

Status: reopened — the two identity registries and one-pair concrete construction
are frozen below. Multi-pair construction remains open until a complete repository
vertical establishes its callable.

#### Model registration

Status: frozen — the standard root-pair Hub vertical, exact Model-config dispatch,
the config-less MembraneMpnn pairs, and ProteinMPNN, LigandMPNN, NA-MPNN, and
Boltz-2 counterexamples to method-based standard routing support this bounded
registry contract.

Model registration is normal explicit Python:

~~~python
MODELS.register(
    "ofoldx.alphafold3",
    model=AlphaFold3Model,
    processor=AlphaFold3Processor,
    standard_tasks=(),
)
~~~

Its complete method signature on the `MODELS` registry is:

~~~python
MODELS.register(
    id: str,
    *,
    model: type[ofoldx.Model],
    processor: type,
    standard_tasks: Collection[Literal["fold", "design", "score", "pack"]] = (),
) -> None
~~~

`id` is any nonempty string and is unique by exact string equality. Registration
does not normalize it or impose a naming regex. `standard_tasks` is converted to a
set whose only permitted values are the four shared task words; the empty set is
valid. It records only whether an unmarked root pair may be wrapped by the matching
generic standard Pipeline. It is not a capability list and does not claim to
exhaust the Model's scientific surface. Public Model task methods cannot replace
this declaration: a family method may have a concrete signature, multi-stage
operation, or family Result that is not losslessly standard.

The Model declares one concrete `config_class`. Registration maintains exact
Model-config-to-implementation consistency: one Model config class always selects
the same Model class. A Processor may declare a concrete `config_class` when it has
persistent lowering values, but registration does not dispatch fresh Processors by
that class and does not impose global Processor-config identity. The exact `(model
class, processor class)` pair may be registered under only one architecture ID.
Reusing the same Model class with a different Processor class is valid when the
exact pairs differ. These are the only cross-registration uniqueness rules.

The registration stores the ID, exact pair, the Model config class, and
`standard_tasks`. The selected Processor class itself is the authority for whether
it has a config class. Registration does not duplicate that optional declaration
or inspect method signatures, annotations, config fields, heads, state-dict keys,
movement methods, or package layout. Exact indexes by architecture ID, Model config
class, and Model–Processor pair serve the corresponding dispatch operations
directly; lookup does not linearly scan registrations or infer a match.

#### Concrete Pipeline registration

Status: reopened — the class-only registration mechanism remains the simplest
candidate, but the prior evidence incorrectly treated unfinished ProteinMpnn,
LigandMpnn, NaMpnn, and Boltz-2 family contracts as completed real surfaces. The
architecture-family audits must first prove at least two complete concrete
Pipelines and their loading/writing routes; no incomplete family is implementation
evidence for this subsection.

Concrete registration is normal explicit Python:

~~~python
PIPELINES.register(
    "ofoldx.mpnn_packer",
    pipeline=MpnnPackerPipeline,
)
~~~

Its complete method signature on the `PIPELINES` registry is:

~~~python
PIPELINES.register(id: str, *, pipeline: type) -> None
~~~

The ID is an exact nonempty string. ID and exact class are each unique; the reverse
class-to-ID index gives a concrete writer one unambiguous marker value. Registration
does not require a generic Pipeline base and does not inspect task methods,
signatures, annotations, components, subdirectories, or repository layout. It
stores no task set. At task-first routing time, support means only:

~~~python
callable(getattr(pipeline_class, task, None))
~~~

An inherited ordinary public method counts; instance-only dynamic `__getattr__`
does not hide a task contract from the class. A methodless registered class is not a
scientific error and needs no usefulness gate: every task request simply fails
before weights are downloaded. The class itself owns its named components, fixed
subdirectories, call signatures, execution order, explicit handoffs, and selection.
The registry and repository marker do not repeat those facts.

#### Lazy built-ins and publication

Status: frozen — `model.json` architecture lookup and `pipeline.json`
concrete-class lookup are distinct exact-ID routes, and plain-import built-in
loading requires each route to reach its owner without eagerly importing every
family. This supports two literal ID-to-owner-module maps.

Built-ins use exactly two literal lazy-import indexes: architecture ID to owning
module, and concrete Pipeline ID to owning module. On an exact registry miss, the
corresponding literal index supplies the one module to import and lookup is retried.
Those maps contain no class, task, config, or component metadata and are not second
registries. OFoldX performs no package-tree, filesystem, entry-point, or installed-
distribution scan and does not eagerly import every family.

Registration is permitted only after the complete released family behavior is
implemented. A small or empty `standard_tasks` set does not make an incomplete
family publishable.

Registration is not the only publication surface. The governing list is Principle
13; incomplete development code cannot ship as an importable but unregistered lower
tier.

Third-party packages may expose an ordinary importable registration function that
the application calls explicitly. A repository never names Python to import.

### 7.3 Repository format

Status: reopened — the ordinary pair schema and one-field marker are frozen for
standard and one-pair concrete repositories. Additional-pair ownership, fixed
sub-directory paths, writing, and loading are frozen in Section 7.4's multi-pair
subsection; the pair format and one-field marker are unchanged there.

Every serializable repository has one runnable root Model–Processor pair:

~~~text
model-repository/
├── model.json
├── processor.json        # only when the Processor has persistent config
└── model.safetensors
~~~

Runnable describes the complete scientific pair and its explicit placement surface;
it is not a promise that every user-cast artifact runs with omitted placement. The
files do not serialize device, Processor runtime dtype, AMP state, or a uniform
Model dtype.

Weights use exactly the single-file or standard sharded safetensors alternatives
defined in Section 7.4. A published Hub repository additionally contains
`README.md`; documentation authorship is not part of serialization. The selected
Hub snapshot is the publication and history unit.

`model.json` is one top-level JSON object with no format envelope. Its reserved
`architecture` field contains the exact nonempty architecture ID; after removing
that field, its members are exactly the concrete Model config serialization. A
configful Processor adds `processor.json`, likewise one top-level JSON object whose
members are exactly its concrete config serialization and which does not repeat
`architecture`. A config-less Processor has no `processor.json`; an empty object
would serialize no fact and the architecture already selects its class.

The registration selected by `model.json` supplies the concrete pair and Model
config class. A declared Processor config class accepts the `processor.json`
mapping. Individual scientific values may themselves be nested JSON objects or
arrays. A Model and a configful Processor expose the concrete config used to
construct them as `.config`; that value supports ordinary `dict(config)`, whose
result is the JSON object written here. A config-less Processor exposes neither an
empty config nor a file. This is a small structural convention, not a required
Config base class or a runtime-checked Protocol. Registration does not preflight
serialization methods; an incompatible config fails at the ordinary load or save
call that uses it.

One scientific axis has one owner, determined by its actual consumer rather than
equal field spelling or defaults. A lowering-only value belongs to ProcessorConfig;
a graph- or state-shape value belongs to ModelConfig. Section 4.6's two
`atom_context_num` fields control different sides of the Packer tensor boundary;
the released route merely gives them the same value. OFoldX adds no shared pair
config, generic pair validator, compatibility matrix, or hidden synchronization.

A repository that publishes a concrete Pipeline adds exactly one optional root
file whose complete JSON object is:

~~~json
{ "pipeline": "ofoldx.boltz2_affinity" }
~~~

`pipeline.json` is a marker for the registered concrete call boundary, not a
manifest or task record. It identifies the complete class; its public methods state
which shared tasks that class supports. It has that one field and no other field;
unknown IDs or extra members fail. A standard root Pipeline has no marker. A
one-pair end-to-end operation may still carry a concrete marker when its signature
or Result cannot satisfy the standard skeleton. Concrete therefore does not imply
composition.

A published concrete Pipeline must have one root pair that scientifically owns the
operation. The root is independently runnable and is not an arbitrary component
chosen to satisfy the format. Additional ordinary Model–Processor pairs may live at
fixed class-owned subdirectories; each uses the same pair format without its own
marker and is not listed in the root marker. A composition with no single root
owner remains live-only under Section 6.4 and cannot acquire a marker by choosing a
convenient component.

For `Boltz2AffinityPipeline` the layout is:

~~~text
model-repository/
├── model.json
├── processor.json
├── model.safetensors
├── pipeline.json
└── structure/
    ├── model.json
    ├── processor.json
    └── model.safetensors
~~~

The root pair is the affinity pass and `structure/` is the structure/confidence
pair. That meaning belongs to the registered class. `pipeline.json` contains no
component names, paths, order, versions, digests, fingerprints, schema identifiers,
or options. `AutoModel` and `AutoProcessor` continue to see only the root pair.

OFoldX adds no parallel format or compatibility revision and no repository-wide integrity metadata. The Hub revision already selects history. Historical reproduction records that Hub revision and the corresponding OFoldX source revision when needed.

OFoldX targets the latest repository snapshot. Before the first intentional public
release and compatibility commitment, breaking code and repository changes update
together. Older states remain addressable through explicit revisions rather than
runtime compatibility branches.

### 7.4 Loading and pairing

Status: reopened — strict physical-key loading and snapshot pairing are specified
below. The exact standard root algorithm is frozen in Section 7.6, one-pair concrete
construction is frozen below, and the multi-pair concrete-repository pair-layout,
writer, snapshot loader, and factory routing are frozen in the multi-pair subsection
below. Multi-pair placement (Section 6.3) and complete family publication remain open.

`AutoModel(repository, device=..., dtype=...)` reads root `model.json`, resolves its
exact registration and constructs `ModelConfig`. The allocation-free construction,
strict state loading, and placement order remain solely in the reopened Section
7.6; this subsection does not preserve the rejected move-before-copy algorithm.
After hydration, explicit placement follows the ordinary Model semantics frozen in
Section 4.1.1, and Auto calls `eval()` once.

`AutoProcessor(repository, device=..., dtype=...)` reads root `model.json` only to
select the paired Processor class. When that class declares `config_class`, Auto
constructs it from root `processor.json` and passes config, device, and dtype to the
ordinary constructor. Otherwise `processor.json` must be absent and Auto calls the
config-less constructor with only device and dtype. It loads no weights. Neither
Auto factory reads `pipeline.json` to change the type it returns. Config-based Auto
construction exists only for Models: it calls `Model(config)` and then ordinary
`.to(...)` only when requested; a fresh Model retains its normal initial training
state. The aggregate header, shard-path, exact-key, placement, and state-loading
algorithm is stated once in Section 7.6. This subsection does not maintain a second
filename or construction-order specification.

For a Pipeline, a mutable Hub reference is resolved once to one immutable commit;
every metadata, root-weight, and fixed-subdirectory download is pinned to that
commit. This subsection is the sole repository Pipeline routing algorithm:

- without `pipeline.json`, the requested task must be in the root registration's
  `standard_tasks`; the factory then loads the root pair and constructs the matching
  standard class;
- with `pipeline.json`, the factory resolves its exact registered class and requires
  a callable public method on the class named by the requested task; that class is
  authoritative; a one-pair class receives the loaded root pair, while a future
  multi-pair class must load every class-owned pair from the same immutable
  snapshot; and
- an unknown Pipeline ID, extra marker field, unsupported requested method, missing
  class-owned subrepository, or concrete construction failure fails without a
  standard-root fallback.

This selection finishes before weight hydration. Users who intentionally want the
independently runnable root pair of a marked repository use AutoModel plus
AutoProcessor and an explicit standard constructor; Auto ignores the marker.
`pipeline.json` never names Python to import. A package-owned literal Pipeline ID
may trigger the known built-in lazy import declared by Section 7.2.

Local directories are trusted mutable inputs. Loading applies the same schema and
aggregate key rules but adds no locking, transaction, race detector, or rewritten
PyTorch error report. If a caller mutates files during a load, the resulting
ordinary file or state-loading failure is sufficient.

Direct Auto calls are independent load operations and may resolve a moving Hub
reference at different commits. Callers needing guaranteed Model–Processor or
multi-pair pairing use one Pipeline construction or pass one already resolved local
snapshot. All pairs of a concrete Hub Pipeline load from its one resolved commit;
that commit is transient transport state, not a claim about later mutable execution.

Runtime device and dtype are loading arguments, not serialized fields. Structural
config overrides do not mutate a pretrained repository load; construct a typed
config for a new Model instead.

#### One-pair concrete Pipeline repositories

Status: frozen — the standard root-pair snapshot loader, concrete Pipeline
registration, and complete MpnnPacker vertical support this
bounded contract. This subsection does not freeze multi-pair repository
construction.

A repository-owned concrete Pipeline that requires only its runnable root pair has
the ordinary live constructor:

~~~python
class FamilyPipeline:
    def __init__(
        self,
        *,
        model: FamilyModel,
        processor: FamilyProcessor,
    ) -> None: ...
~~~

The supplied objects remain available as public `model` and `processor` attributes.
Construction does not redispatch them, change Model state, infer a repository, or
retain repository transport state.

For a marked one-pair repository, `pipeline(task, model=...)`:

1. resolves root metadata and `pipeline.json` from one snapshot;
2. parses the exact one-field marker, resolves its registered class, and checks
   `callable(getattr(pipeline_class, task, None))` before requesting weights;
3. loads the root Processor and Model from that same snapshot through the ordinary
   root-pair algorithms; and
4. calls `pipeline_class(model=model, processor=processor)`.

It does not call `from_repository`, `_from_repository`, a loader hook, constructor
introspection, or a component factory. It does not store the requested task on the
instance. The constructor and ordinary Python calls own compatibility failures;
the factory adds no pair validator or signature scan. A construction failure never
falls back to a standard Pipeline. AutoModel and AutoProcessor ignore the marker
and load only the root pair.

The repository is the ordinary root pair plus `pipeline.json`. A registered
one-pair concrete Pipeline exposes
`save(directory, *, max_shard_size="5GB")`. It calls the ordinary pair writer,
which first removes a stale marker, and writes its exact registered one-field
marker only after the pair succeeds. It adds no Pipeline config, task field,
component list, source state, or second manifest.

#### Multi-pair concrete Pipeline repositories

Status: frozen — the multi-pair pair-layout, writer, snapshot loader, and factory
routing below are proven by three materially different verticals
(`Boltz2AffinityPipeline`, `ODesignPipeline`, `BoltzGenPipeline`). This freezes only
repository construction and routing; multi-pair *placement* stays reopened in Section
6.3, and complete family checkpoint publication stays reopened in Sections 7.5/7.6.

A concrete Pipeline whose live constructor binds more than one Model–Processor pair
cannot be built by the one-pair `pipeline_class(model=..., processor=...)` call. Such
a class instead declares its layout as ordinary Python: a class-level ordered
**pair-layout**. Each entry is one class-owned pair and fixes

- its repository-relative directory — the empty string is the independently runnable
  root pair that scientifically owns the operation (Section 6.2), any other value is a
  fixed class-owned sub-directory (`structure/`, `protein_designer/`, `ligand_designer/`,
  `ifold/`, `fold/`, `affinity/`);
- the exact constructor keyword(s) the reloaded pair fills — either the two keywords
  carrying the Model and the Processor, or, when the pair is itself a read-only
  single-pair child Pipeline (the ODesign OInvFold sequence designers), the one keyword
  that receives that reloaded pair rewrapped in its own concrete Pipeline class; and
- whether the pair is required or optional. An optional pair (BoltzGen's injected
  boltz2 `fold/` / `affinity/` stages) is written only when present and reloaded only
  when its directory exists.

The pair-layout lives in the class, never in the repository. `pipeline.json` remains
the frozen one-field marker of Section 7.3; it gains no component list, path map,
order, or options. Sub-pairs use the ordinary pair format of Section 7.3 and carry no
marker of their own. The registry stores no layout either — the class alone owns its
named components and their directories, exactly as required by Section 6.4.

A registered multi-pair concrete Pipeline exposes the symmetric writer
`save(directory, *, max_shard_size="5GB", repo_id=None, ...)`. It writes every
present class-owned pair with the ordinary single-pair repository writer of Section
7.5 — the root pair at the repository root and each named pair in its fixed
sub-directory — and then writes its exact registered one-field marker only after all
pairs succeed. Because the ordinary pair writer first removes a stale root marker, and
the marker is written last, a failed write never advertises a complete multi-pair
layout. A missing required pair fails the write. Optional Hub upload reuses the same
uploader as the one-pair writer.

It also exposes the classmethod `from_repository(directory, *, revision=None,
device=None, dtype=None)`. This is the multi-pair counterpart the one-pair path
explicitly does not use. It resolves the reference to one immutable snapshot, loads
every class-owned pair from that single snapshot through the ordinary root-pair
loaders of Section 7.6, maps each to its declared constructor keyword — rewrapping a
child-Pipeline pair in its concrete class — skips an absent optional pair, and
constructs the Pipeline through its ordinary multi-pair constructor. A missing
required pair fails without a standard-root fallback. Device and dtype are the
ordinary co-location loading arguments applied to every reloaded pair; whether a
concrete class instead exposes stage-wise placement remains the Section 6.3 question.

The repository routing algorithm of this subsection is unchanged for one-pair classes.
When the marker resolves to a class that declares a pair-layout, the factory dispatches
to that class's `from_repository` on the pinned snapshot instead of the one-pair
`(model, processor)` call; when it does not, the one-pair call is used exactly as
frozen above. Both routes resolve the exact registered class and require a callable
public method named by the requested task before any weight download. `Boltz2AffinityPipeline`
therefore reloads the affinity root pair and the `structure/` pair; `ODesignPipeline`
reloads its backbone root pair and the `protein_designer/` and `ligand_designer/`
sequence designers; and `BoltzGenPipeline` reloads its design root pair, the `ifold/`
pair, and any present optional boltz2 `fold/` / `affinity/` stages.

### 7.5 Writing and conversion

Status: reopened — ordinary writing and marker order are specified below. The
writer must preserve the exact public `state_dict` key set; a family whose standard
safetensors writer elides tied aliases returns to Tik rather than inventing a
framework alias protocol.

The paired Model+Processor repository writer is:

~~~python
ofoldx.artifacts.save_model(
    directory,
    model=model,
    processor=processor,
)
~~~

`save_model` requires one exact registered Model–Processor class pair. It writes
`model.json` and, only for a configful Processor, `processor.json` as specified in
Section 7.3. For a config-less Processor it removes any stale `processor.json`
rather than writing an empty object. It then uses the standard Hugging Face
`save_torch_state_dict` safetensors path with the
`model{suffix}.safetensors` filename pattern. That helper owns sharding and removal
of stale standard single/index/shard weight artifacts. README authorship and Hub
publishing remain separate. The repository adds no compile, quantization, backend,
or mutable Pipeline metadata. V1's writer contract is an ordinary untransformed
registered Model; it does not attempt to detect or reinterpret a transformed state
dict.

Every Pipeline with a serialized root owner has the symmetric writer:

~~~python
pipeline.save(directory)
~~~

Before writing any ordinary pair, `save_model` removes an existing
`pipeline.json` at that pair root. A standard task Pipeline delegates its save to
that writer and produces no marker. A repository-owned concrete Pipeline first
removes the destination root marker, writes its root pair and every class-owned pair
at fixed subdirectories, and writes the exact one-field root marker only after all
pairs succeed. It serializes neither an execution graph nor runtime state.
`Boltz2AffinityPipeline.save` therefore writes the affinity pair at the root, the
structure/confidence pair at `structure/`, and its marker last. An ownerless live
composition has no aggregate `save`; its ordinary component pairs may be saved
separately.

The writer creates or updates the target directory, replaces the artifacts it owns,
and leaves unrelated publication files such as `README.md` alone. Marker-last
ordering prevents a failed write from advertising a complete concrete layout; it
does not make the write atomic. OFoldX adds no staging directory, rollback,
transaction, or concurrent-writer protocol around the standard helper.

The writer snapshots `set(model.state_dict())` before calling the standard helper,
because that helper may mutate its input and elide shared-storage aliases. Before
reporting success it reads only the resulting single-file header or index
`weight_map` and requires the stored physical key set to equal that snapshot. Alias
metadata is not interpreted: it is caller-overridable, cannot safely distinguish a
tied target from an untied one, and has no symmetric standard sharded loader. A
Model whose public state dict exposes one canonical key per stored tensor works
normally; a Model whose keys are elided fails explicitly and returns to a real
family Tik. OFoldX does not scan storages, clone tied tensors, add an alias registry,
or weaken global key checks.

Converters map one explicitly identified upstream checkpoint/config contract into the native ModelConfig and state dict. They recognize the released source format directly and use strict PyTorch loading as the target authority. A converter does not guess missing scientific defaults, create a reduced alternate repository format, or add a second compatibility layer.

Publishing to the Hub is a separate operation. Model cards distinguish OFoldX code licensing from upstream code and weight terms. A family with unknown or prohibited redistribution terms can still be implemented completely, but its weights are not relicensed by OFoldX.

The Hub design remains reopened until all of the following have affirmative
evidence: the concrete-Pipeline registration and loader protocol through at least
two different verticals; every family's complete checkpoint/pair inventory and
class-owned repository layout; ordinary Model placement through real family
movement policies; exact public-state-dict key preservation;
multi-pair Pipeline placement; and the minimum `huggingface_hub` and `safetensors`
versions whose exact helpers implement the selected behavior.

### 7.6 Standard root-pair Hub slice

Status: reopened — the root schema, registration indexes, strict JSON, standard
materialization algorithm, shard ownership checks, writer postconditions, standard
task delegation, and one-pair concrete marker loading have bounded contracts below
and in Section 7.4. Multi-pair repository construction and routing are now frozen in
Section 7.4's multi-pair subsection; multi-pair placement, complete family publication,
and dependency-floor claims remain reopened. The shared materialization algorithm
is frozen independently of family eligibility: RFdiffusion and PXDesign prove it
with real checkpoints. AlphaFold 3, the LigandMpnn sequence pair, and the independent
MpnnPacker pair have the bounded
family-owned non-persistent-constant work frozen below. MultiMolecule ESMC already
rebuilds its non-persistent rotary buffer on the first real-device forward, so it
is not a loader exception; complete ESMFold2 eligibility still requires its own
full checkpoint evidence.

#### References and one snapshot

`classify_reference(str | PathLike) -> str | Path` is the only public reference
classifier. A PathLike, existing string path, or explicit POSIX, home, Windows
drive, or UNC local spelling becomes an expanded `Path`; any other string remains
unchanged and is a Hub ID at repository boundaries. An empty string is not
special-cased. A local reference rejects `revision`; after classification,
ordinary file reads own missing-path and not-a-directory failures rather than a
second defensive preflight.

One small loader-only record `(path, repository, commit)` carries resolved
transport state. A Hub load resolves metadata once, takes the immutable commit from
that snapshot, and pins every later download to it. This is necessary transport
state, not a public Reference or snapshot abstraction. Local loads carry no
repository or commit.

#### Registration and Auto

`MODELS.register(id, *, model, processor, standard_tasks=())` records one exact pair,
the Model's declared config class, and its set of shared task words. It protects
only dispatch meaning: the ID is a nonempty string; Model, Processor, and Model
config are classes of the required ordinary kinds; standard tasks are among fold,
design, score, and pack; and exact ID, Model-config, or pair mappings cannot
conflict. Exact indexes serve ID, Model-config, and pair lookup. The
`MODELS.register(...)` call does not inspect signatures, annotations, methods,
fields, state keys, package layout, or runtime placement and performs no package
scan, owner-module import, locking, or plugin discovery. Exact architecture lookup
may import only the literal built-in owner on a miss, as specified in Section 7.2.

A registered root pair has one of these direct Processor construction contracts:

~~~python
model_class(model_config)
processor_class(processor_config, *, device=None, dtype=None)
processor_class(*, device=None, dtype=None)  # config-less
~~~

Config-based `AutoModel` dispatches by exact config class. Fresh Model construction
calls `model_class(config)`, applies ordinary `.to(...)` only when requested, and
preserves the class's normal training state. Fresh Processors use their concrete
constructors; there is no config-based `AutoProcessor`. Repository Auto reads
`model.json` to select the pair and calls `eval()` after a successful Model load.
Repository Model construction uses the frozen materialization subsection below.
Processor loading reads no weights. `from_pretrained` remains only a thin
repository-only spelling. Bare mappings are not guessed. Auto never reads
`pipeline.json`.

Fresh and repository-backed Model construction follow the same ordinary PyTorch
placement. The repository loader hydrates checkpoint values and then obeys an
explicit final `device="meta"` through ordinary `.to(...)`; it does not reinterpret
that request as a skip-hydration optimization. Internal meta construction remains
only the allocation mechanism of the frozen repository loader and is not a repair
for family tensors absent from `state_dict`. Repository `AutoProcessor` remains
independent and follows its concrete constructor's ordinary placement.

#### Root files and aggregate loading

One ordinary root contains `model.json`, exactly one of `model.safetensors` or
`model.safetensors.index.json`, and `processor.json` exactly when the selected
Processor declares `config_class`. `model.json` is a strict JSON object containing
the exact registered `architecture` plus `dict(model.config)`. For a configful
Processor, `processor.json` is exactly `dict(processor.config)` and does not repeat
architecture. For a config-less Processor that file is absent. JSON `NaN`,
`Infinity`, duplicate object keys, and floating literals that overflow to a
non-finite Python value fail during decoding. There is no schema, package,
contract, digest, fingerprint, compatibility, or README field.

For an indexed checkpoint, `weight_map` is a JSON object whose values are string
paths. Each path must be lexically contained in the repository: it has no POSIX or
Windows anchor and no `..` component. Nested relative paths and arbitrary shard
names are valid. Resolution is deliberately lexical because a normal Hub snapshot
uses repository entries that may be symlinks to blobs outside the snapshot
directory. Filename patterns, root-only placement, ordinal coverage, shard totals,
and glob discovery are writer conventions rather than reader invariants.

Hub hydration first requests only the two fixed single-file and index alternatives
at the pinned commit and applies the exactly-one rule before requesting any shard.
If the index is the selected alternative, it parses `weight_map` and calls
`hf_hub_download(repo_id=repository, filename=path, revision=commit)` once for each
distinct referenced relative path. `filename` is an exact Hub file name, not a
glob. It never feeds indexed names to `snapshot_download(..., allow_patterns=...)`
or downloads `model-*.safetensors` by wildcard.

#### Standard Model materialization

Status: frozen — actual RFdiffusion and PXDesign checkpoints, the serialized-BF16
counterexample to eager copy, inventories of every current public Model constructor,
PyTorch `assign=True` behavior, and two independent harsh reviews support this
bounded algorithm. It freezes the shared loader for eligible registered Models; it
does not declare any unaudited family eligible.

A Hub-loadable family establishes all of the following before registration:

- its Model constructs under `torch.device("meta")` without data-dependent
  constructor work that requires real tensor values;
- learned and otherwise persistent tensor state is in `state_dict`; any tensor
  deliberately absent from it is a small deterministic family-owned constant that
  has real storage after internal meta construction and correct device and dtype
  semantics after ordinary `.to(...)` and on first forward;
- the Model is assign-safe: replacing its registered Parameter and Buffer objects
  does not leave semantic references to the constructor objects in a plain
  container, closure, optimizer, or other state, and registered names do not rely
  on object or storage aliasing that assignment would split; and
- any family fix for internal meta construction preserves ordinary CPU, accelerator,
  and explicit-device construction semantics.

These are family evidence obligations, not registration fields or runtime checks.
The loader does not scan object references or storages, advertise a capability,
invoke a materializer, repair a family after loading, or fall back to eager copy.
A future custom Parameter, parametrization, `load_state_dict` override, or load hook
must establish its own assign semantics before that family publishes.

The exact materialization path is:

~~~python
with torch.device("meta"):
    model = model_class(config)

state_dict = ...  # selected safetensors files, loaded once each on CPU
model.load_state_dict(state_dict, strict=True, assign=True)
del state_dict

if device is not None or dtype is not None:
    model.to(device=device, dtype=dtype)
model.eval()
~~~

The meta context ends before loading state. The call above is one ordinary public
`Model.load_state_dict` call, so native subclass and pre/post-hook semantics remain
authoritative. Native strict loading owns target keys, shapes, and assignment
compatibility; OFoldX does not first construct another expected-key set or rewrite
its failure. Auto creates a fresh Model before any user optimizer exists and does
not change PyTorch's global swap-module future.

A single-file checkpoint is loaded once on CPU. For a sharded checkpoint, every
distinct file referenced by `weight_map` is loaded exactly once on CPU and merged
shallowly. During that merge the loader rejects a physical key occurring in more
than one shard, aggregate keys different from the index keys, and a key physically
stored under a different full relative path than its indexed owner. Those failures
could otherwise select a plausible wrong tensor. There is no separate header pass
and no second open merely to repeat these checks.

`assign=True` preserves every serialized tensor dtype, including a mixed-precision
checkpoint. Only an explicit public `dtype` request converts floating and complex
state, through the ordinary Model placement contract after hydration. Deleting the
aggregate dictionary immediately after assignment releases its additional
references before placement or casting; the Model itself continues to own the
mapped CPU tensors until ordinary `.to(...)` replaces them. The loader promises no
exact RSS or page-cache bound and does not clone mmap-backed tensors to manufacture
one.

The loader adds no pickle fallback, hash, tensor-value or finite scan, duplicate
shape check, alias registry, policy object, marker, or rewritten PyTorch failure.
Compilation and quantization happen only after an ordinary loaded Model exists.

The reader compares only physical tensor keys and ignores alias-like safetensors
metadata. A Model whose writer elides public state-dict keys is outside this slice
and returns to Tik; the reader never fabricates missing keys or weakens exact key
matching.

##### Family-owned non-persistent constants

Status: frozen — AlphaFold 3 revision
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`, LigandMPNN revision
`26ec57ac976ade5379920dbd43c7f97a91cf82de`, the four released LigandMPNN
sequence checkpoints, the released MpnnPacker checkpoint, and meta-construction
probes support this bounded rule.

AlphaFold 3 confidence distance-bin edges and its public
`AlphaFold3NoiseEmbedding` Fourier weight and bias, the shared MPNN periodic-table
values, LigandMpnn side-chain element values, and MpnnPacker atom masks, chi
indices and masks, rigid frames, group indices, and literature positions are small
deterministic values derived entirely from config or package constants. They are
not learned state, and none occurs in the released checkpoints. They remain
`persistent=False` buffers; converters and repository artifacts do not manufacture
a second copy.

Each owning Module creates its ordinary Parameters first and uses one such
Parameter's construction device for these constants. When that device is meta only
because the repository loader is performing its internal allocation-free
construction, the tiny deterministic constants are built on CPU. Ordinary CPU or
accelerator construction builds them on that ordinary device. Later explicit
`.to(device=..., dtype=...)` applies normal PyTorch semantics, including converting
floating bin edges when the caller requests a whole-Model dtype.
With omitted dtype, `assign=True` state hydration does not retroactively choose the
constructor dtype of non-persistent constants from any learned tensor. A serialized
checkpoint may intentionally contain mixed dtypes, and no first-weight rule can
recover a Processor dtype or one whole-Model policy from it. Callers use explicit
pair placement or AMP; an unrequested incompatible mixture fails normally.

The shared MPNN periodic-table factory and LigandMpnn side-chain-atom factory accept
only an ordinary optional `device=` argument. There is no load hook, `_apply`
override, lazy forward mutation,
materializer, loader scan, family flag, or persistent duplicate. State-dict keys
and converted checkpoints remain unchanged.

MultiMolecule's ESMC rotary buffer uses its owner's existing lazy first-forward
construction and needs no OFoldX action. An explicit whole-Model dtype request is
obeyed; OFoldX does not preserve hidden FP32 exceptions on behalf of an experienced
caller.

#### Writing and a standard task Pipeline

##### Standard writer

Status: frozen — Hugging Face/safetensors single, empty, forced-sharded, stale-file,
and alias-elision behavior, real 1.5–2.5 GB family state inventories, and two
independent loader reviews support this bounded writer contract. It freezes only
ordinary root-pair serialization and shard-size control, not Model materialization,
concrete Pipelines, converter correctness, or publication of any family.

`save_model(directory, *, model, processor, max_shard_size: int | str = "5GB")`
requires one exact registered pair,
always writes strict `model.json`, writes strict `processor.json` only for a
configful Processor, and delegates safetensors sharding and the helper's
recognized stale-weight namespace to
`save_torch_state_dict(..., filename_pattern="model{suffix}.safetensors",
max_shard_size=max_shard_size)`. OFoldX
does not move or mutate the Model and does not build its own CPU copy of the state
dict; the serializer owns any device transfer needed to write bytes. Files outside
the selected helper weight namespace, including README, are preserved. A stale root
`pipeline.json` is removed before advertising an ordinary pair. `max_shard_size`
uses the helper's ordinary int-or-size-string language and default; OFoldX adds no
size parser. A single tensor may exceed the requested shard size under the helper's
normal semantics.

An empty state dict is valid. After the standard helper performs its ordinary
cleanup, the writer emits an empty `model.safetensors` with safetensors directly
when the helper produced no shard. The reader likewise accepts an empty index or
single-file key set exactly when the constructed Model's state dict is empty; it
does not add a nonempty-map guard.

Before invoking the helper, the writer snapshots the public state-dict key set.
After empty-state handling, it requires exactly one of the root single-file and
index alternatives. It then reads the single-file header or sharded index key
manifest and requires that serialized key set to equal the snapshot. These are successful-
write postconditions: the Hub helper may warn rather than raise when stale cleanup
fails and may remove shared-storage aliases from its input. The writer does not
reopen tensor values, interpret metadata, scan storage, roll back, or perform a
second reader validation pass.

If the helper elides public state keys because several names share storage, the
write fails. That family cannot publish until its Model/converter exposes a
canonical one-key-per-stored-tensor state surface; the writer does not scan, clone,
or invent an alias protocol. The ordinary standard Pipeline
`.save(directory, *, max_shard_size="5GB")` exposes and forwards the same keyword.

##### Standard Pipeline binding

Status: frozen — direct delegation, the frozen standard Model materialization, and
one-pair repository routing support this binding. This does not freeze multi-pair
placement or construction.

The four standard task Pipelines bind the supplied public `.model` and
`.processor` unchanged. Their `.to(device=..., dtype=...)` delegates the same
arguments to both and returns `self`; `.save(...)` delegates to the frozen writer.
A shared internal
implementation is justified by these four real consumers but does not create a
public Pipeline base or lifecycle framework. There is no repository-specific
constructor or `_from_repository` lifecycle: Hub construction creates the ordinary
Pipeline after loading one pinned snapshot.

For `pipeline(task, model=...)`, absence of `pipeline.json` selects the matching
standard class after exact task support is checked. The Processor is constructed
from the metadata snapshot before weight hydration; the Model then loads from that
same snapshot. The Pipeline returns the Model Result unchanged; it neither
re-decodes, copies, nor injects loader history.

One-pair concrete marker loading follows the exact routing and construction
algorithm frozen in Section 7.4. Auto ignores the marker and a standard save
removes it. Standard Pipeline binding adds no second marker parser, component
manifest, lifecycle hook, or fallback.

Exact dependency floors remain a package compatibility decision and must be proved
against the APIs and behavior used here; this artifact format does not encode them
as schema or compatibility versions.

One ephemeral, uncommitted tiny-pair script has exercised config and repository
Auto, fresh versus loaded mode, Processor-without-weights, single, empty-state, and
forced-sharded round trips, arbitrary nested shard names, exact Model keys,
duplicate and overflow JSON rejection, single-file alias elision, marker removal,
and preservation of an unrelated README. It produced useful evidence but is not a
publication gate and did not prove large-model placement or memory behavior. A
separate BF16 round trip established that repository serialization is not a runtime
placement snapshot: omitted dtype preserves serialized state and constructor
defaults, while explicit pair dtype remains the caller's control. Diagnostics stay
uncommitted and no Model-side tests are added.

## 8. PyTorch and OOps backends

### 8.1 Reference semantics

OOps is co-developed with OFoldX and distributed as ooperators. It is optional. Every optimized operator has an always-available public PyTorch reference defining:

- tensor inputs and output schema;
- mask and numerical meaning;
- supported dtype and device behavior; and
- autograd semantics.

Models call an OFoldX dispatcher, not OOps directly. Backends cannot change Model configs, state-dict keys, concrete raw output types, or Results.

The dispatcher and OOps adapter are implementation details. The first release has no third-party backend extension ABI.

### 8.2 Selection

For every Model, the user may select an operator policy:

- auto: use a compatible optimized implementation, otherwise PyTorch;
- torch: require the public reference; or
- oops: require OOps and fail if unavailable.

Every Model exposes the following method. The base initializes ``auto``; a concrete
constructor may select a stricter documented initial policy after its operator graph
is built:

~~~python
model.set_backend(name) -> model
~~~

A torch-only Model records the same policy but has no dispatched operation to
reconfigure. A task Pipeline does not duplicate backend state. Users select the
backend on every explicitly named Model: `pipe.model` for a standard Pipeline, and
each named Model such as `pipe.model` and `pipe.structure_model` for a concrete
multi-pair Pipeline.

Auto dispatch inspects only facts needed to select an implementation. It forwards the documented valid tensor arguments without another full validation pass. Invalid-input failure wording and order are not a cross-backend API.

### 8.3 Compile and transformation lifecycle

The ordinary order is:

~~~text
load
  -> select backend
  -> place Model and Processor
  -> compile where useful
  -> execute
~~~

Backend lookup resolves outside or during tracing so the captured graph contains a stable callable or registered operator. PyTorch guards and recompilation retain their normal meaning. OFoldX does not inspect or mutate private compiler fields.

Model placement uses ordinary public `.to` semantics. Serialized mixed dtype and
family autocast are separate from state placement. `load_state_dict`,
`torch.compile`, and torchao transformations retain ordinary library semantics.
OFoldX does not wrap them in a lifecycle manager or maintain duplicate
compiled/quantized flags.

The public PyTorch path works without OOps. OOps qualification compares representative forward values and gradients against that path. Backend optimization is never a reason to weaken the reference implementation.

## 9. Quality and release evidence

### 9.1 Publication quality

Every released family satisfies one quality standard. There is no lower class for
an incomplete port, converter, architecture, or checkpoint.

Before a family appears on any publication surface in Principle 13, review
establishes:

- inspected source revisions and checkpoint inventory;
- every released scientific function and checkpoint variant;
- exact canonical and tensor contracts;
- faithful public Result construction;
- complete native checkpoint conversion: every upstream tensor is interpreted,
  every live target tensor is assigned exactly, and a proven dead upstream tensor
  is discarded explicitly rather than retained as a dead Module or hidden by loose
  loading;
- ordinary direct PyTorch execution and family-evidenced `torch.compile` and
  torchao friendliness where those transformations are claimed;
- a subtraction and performance pass that removes unproved tensor scans, copies,
  host synchronizations, Python wrappers, and lifecycle state from hot paths;
- documented limitations and intentional upstream divergences; and
- licensing and redistribution facts.

The family is absent from every publication surface in Principle 13 until this
account is complete. A raw state-dict conversion, one forward tensor, one standard
verb, or one working checkpoint does not establish family support.

### 9.2 Frequent CI

Frequent CI is cheap, self-contained, and runs many times a day. It uses no network, production checkpoint, external reference checkout, private OOps installation, large fixture, or required GPU.

It covers shared infrastructure at its owning layer:

- wheel and source distribution build and clean import;
- canonical domain identity, topology, structure I/O, and shared Result semantics;
- repository classification, loading, writing, pairing, and registration with tiny synthetic fixtures;
- standard Pipeline delegation and placement;
- public PyTorch operator behavior and dispatcher fallback;
- documentation/API examples that do not require remote assets.

Frequent CI contains no family model-side validation suite and does not freeze partially understood behavior through tests. Shared synthetic fixtures may exercise shared infrastructure; they are not claims of family scientific parity.

CI does not download hundreds of gigabytes of weights. Real-checkpoint execution is never smuggled into a marker, optional test, or hidden validation job.

### 9.3 Parity tooling

Real-reference comparisons live only under scripts/parity. They record the exact upstream revision, checkpoint, inputs, random controls, output mappings, and tolerances used for each comparison. Those facts make the evidence reproducible; they do not pin an OFoldX dependency or public behavior to that upstream version. Local reference checkouts and weights remain outside the repository.

Parity scripts are capable of covering all released functions and checkpoint
variants, not only a headline forward pass. Inability to compare one path directly
changes the diagnostic method, not OFoldX's required support surface. Final public
biological inputs and Results matter more than intermediate tensor agreement.
Stochastic comparisons control the same meaningful random choices where the
upstream implementation permits it.

Scripts print or write local diagnostic output. Generated reports, checkpoint
summaries, logs, and dated acceptance records are not committed. Parity is an
on-demand development diagnostic. Its observed results may be inspected and cited
as evidence for specific Section 9.1 claims; neither running a parity script nor
producing or passing a parity report is a CI, registration, publication, or
review-completion gate. Parity is never a frequent-CI check, automated release job,
committed validation artifact, or hidden release marker.

Confirmed upstream defects are documented in UPSTREAM_BUGS.md. OFoldX implements the scientifically correct behavior after the contract is frozen; it does not hide an upstream defect behind compatibility logic.

## 10. Documentation and migration rules

### 10.1 Documentation ownership

The root README is the public entrance. This document investigates cross-family
architecture and explicitly marks the bounded subsections that have become accepted
contracts. README templates and rendering are polishing work after the current
Model Hub contract. Family READMEs explain user-facing semantics, usage,
intentional divergence, limitations, and citations without manually duplicating
facts that can be derived from registration or `model.json`/`processor.json`. Their
eventual complete evidence includes:

- repository ID and architecture;
- inspected upstream revisions;
- every released function and standard task;
- exact biological inputs and Results;
- Model, Processor, Pipeline, and top-level usage;
- checkpoint variants and conversion;
- family-specific scientific equations and defaults;
- available parity tooling, when it exists, and its factual limitations;
- upstream code and weight terms; and
- citations.

UPSTREAM_BUGS.md records confirmed upstream defects and OFoldX's chosen correct behavior. Parity operational details live beside scripts/parity. Backlogs and progress belong in issues, not this normative document.

### 10.2 Direct pre-release migration

Before the first intentional public release and compatibility promise, OFoldX
replaces incorrect prototype contracts directly. It does not preserve them through
aliases, fallback branches, duplicate registries, shadow modules, or files whose
only purpose is old behavior.

Migration is complete only when production imports and docs use the new contract. Dead prototype code is deleted rather than renamed out of sight. Public imports are expressed by actual module locations and reexports; source files do not maintain a second public-surface list through __all__.

Private files, helpers, and dataclasses are not default organization tools. A helper
needs a concrete invariant or current consumer that ordinary inline Python cannot
express more clearly; repeated calls alone are not proof. A meaningful reusable
scientific Module or value is public by default, and a private name must additionally
explain why that meaning should not be imported. A dataclass is justified only by a
natural record; `kw_only=True`, suppressed representation, custom `__post_init__`
mutation, hidden equality, and `compare=False` each carry their own burden of
proof. Tensor carriers may use `eq=False` because Tensor equality is not scalar,
but that fact does not justify changing unrelated dataclass semantics. This
document intentionally does not prescribe exact helper names, nominal subclass
checks, validation order, or error text.

### 10.3 The Runner training contract

The Runner is the config-only Danling consumer that trains, evaluates, and infers
with any registered family. Construction takes one `RunnerConfig` — a
`danling.RunnerConfig` extended with a `model` section (a registered architecture
id plus typed-config overrides, or a repository to fine-tune) and a `data` section
(each split's raw-structure source plus Processor arguments). The Runner builds the
Model from the model registry and the Processor as the featurizer; danling owns the
optimizer, scheduler, dataloader, checkpoint, EMA, and metric machinery from the
inherited config sections. No training state — dataset, objective, optimizer, or
schedule — is ever written into a Model, Processor, or Model config.

A `StructureDataset` lowers each raw structure through the family Processor on
demand, passing the same reference inference passes so training and inference share
one featurization. A generic dataclass collate concatenates per-structure Inputs
along the batch axis, right-padding ragged sequence and atom axes to the batch
maximum; a family whose Input needs bespoke padding registers an override.

The training loss lives only in an objective, never in a Model: a Model's forward
returns predictions or unreduced loss primitives, and the objective reduces one
batch to a scalar. Objectives are grounded, not speculative — the MPNN families
masked-mean their in-repo label-smoothed per-token loss over the scored positions,
and the diffusion families apply the EDM-weighted denoising error over valid atoms.
An architecture with no faithful in-repo objective is absent from the objective
table and raises rather than fabricating a loss (Principle 13). `Runner.infer`
drives the Model forward graph in batches; the Result-producing task surface
remains the Pipelines'.

The exact class, method, and helper names remain ordinary implementation detail:
this section fixes the contract — config-only construction, the Model/objective
split, and no training state in frozen surfaces — not a signature.

### 10.4 Review questions

The governing review principles are scientific completeness, predictable public
meaning, canonical identity, ordinary PyTorch, one mechanism per job, and a second
pass that starts from deletion. Any extra noun, guard, private name, dataclass, or
layer must name the concrete invariant or current consumer that ordinary
Python/PyTorch/MultiMolecule cannot express more directly. When that answer is
unclear, investigate first. AGENTS.md owns the procedure; this section creates no
checklist, report, approval artifact, or additional gate.
