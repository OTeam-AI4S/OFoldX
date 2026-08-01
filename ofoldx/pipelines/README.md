# Pipelines

A pipeline performs one biological operation using a concrete scientific workflow.
The task identifies its purpose; the pipeline class defines how it runs. Pipelines
may own one Model–Processor pair or several named child workflows.

## Usage

`pipeline`, `AutoPipelineForFold`, `AutoPipelineForDesign`, `AutoPipelineForScore` and
`AutoPipelineForPack` use the same loader. An ordinary model repository selects the
explicitly registered default for its architecture and requested task. A composite
repository declares its concrete workflow and named children in `pipeline.json`.
An architecture-registered saved paired workflow can select another registered task
for that architecture; a composition retains its declared workflow and actual children.

A processor's `prepare_input` returns features and decoding context. Its `prepare` combines input and optional
truth preparation into a pair; a pipeline's `prepare` assembles the state required by that complete workflow.

## Complete and Staged Calls

Each concrete pipeline exposes its scientific parameters and prepared, raw-output
and result types. The complete call runs preparation, execution and postprocessing.
Workflows whose preparation draws random values use one generator through preparation
and execution; consult their concrete signatures. The `fold`, `design`, `score` and
`pack` methods provide the corresponding complete calls.

`postprocess` receives preparation information to preserve component identity,
source coordinates and scientific scope. `forward` may run sampling, optimize
with gradients or execute complete child pipelines. Each stage describes a responsibility;
the concrete implementation determines its device and inference or gradient mode.

Preparation policies travel with the prepared value. Set shared observation policies
during preparation when composing stages.

## Composition

`CompositePipeline.components` exposes its named children. A composition can contain
another composition, provided it satisfies the role's scientific input and output
requirements. Equal task names alone do not establish compatibility: a packing stage
needs the original backbone and exact designed positions as well as the candidate
sequence; a differentiable design stage needs gradient-capable execution.

`save` writes the local repository; supplying `repo_id` also publishes that
complete repository to the Hub. Saving preserves each child's workflow identity
in its own directory. Loading reconstructs the actual saved children, including explicitly substituted or
nested pipelines. Custom serializable workflows register a stable Pipeline ID;
explicit in-memory components use their normal Python scientific interfaces.
Custom task names are supported through the same registration and loader.

## Components and Repositories

`PairedComponent` provides loading, saving and placement for a Model–Processor
pair. `PairedPipeline` adds the executable Pipeline interface and saved identity.
`CompositePipeline` owns named children, including nested compositions, and
delegates their lifecycle. Its `component_layout` contains `ComponentSpec`
descriptions; `.components` contains the actual child instances. Each child has a
nonempty repository directory. Saving retains each executable child's concrete
Pipeline ID so loading reconstructs the selected child workflow; non-executable
pairs use their declared component type.

## Writing a Pipeline

An executable pipeline declares its intrinsic `task` on the class. A subclass
inherits that task before registration. Register a stable serialized ID with
`PIPELINES.register(id, pipeline=Class)`; pass `model_architecture=...` only to
declare an architecture's default for that task. Registration reads the class's
task rather than accepting a separate task argument. Subclass an existing
scientific Pipeline to reuse its preparation, complete call and repository lifecycle while changing a typed stage.
The abstract `Pipeline` base defines those author responsibilities for a new
workflow.

## Inference Compute

Scientific controls belong to the concrete family workflow. A custom pipeline can
combine compatible child stages, score their candidates and decide whether to
allocate additional computation. Preserve the required source structures, scopes,
output meanings and random streams when handing candidates between operations.

Each composition defines which compatible stages it consumes. A design workflow's
differentiable model computations remain distinct from inference-only sampling
stages. Inspect the concrete stage signatures and scientific outputs when replacing a child.

A repository saved with `save` contains the workflow and its model components. Continuing an active search or
optimization also requires its optimizer state, random stream, and schedule. Consult the concrete workflow's API
for continuation support; returned logits and trajectories describe scientific outputs rather than that full
execution state.
