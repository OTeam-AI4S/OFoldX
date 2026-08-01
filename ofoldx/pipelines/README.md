# Pipelines

A pipeline performs one biological operation using a concrete scientific workflow.
The task identifies its purpose; the pipeline class defines how it runs. Pipelines
may own one Model–Processor pair or several named child workflows.

## Usage

Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact before reading a structure.

```python
from ofoldx import AutoPipelineForDesign, pipeline

designer = pipeline("design", model="oteam/proteinmpnn")
designer = AutoPipelineForDesign("oteam/proteinmpnn")
print(designer.id, designer.task)
```

`AutoPipelineForFold`, `AutoPipelineForScore` and `AutoPipelineForPack` provide the
corresponding task-specific entrypoints. Both forms use the same loader. An
ordinary model repository selects the explicitly registered default for its
architecture and requested task. For example, ProteinMPNN design and score load
from the same model weights. A composite repository declares its concrete
workflow and named children in `pipeline.json`. An architecture-registered saved
paired workflow can select another registered task for that architecture; a
composition retains its declared workflow and actual children.

A processor's `prepare_input` returns features and decoding context. Its `prepare` combines input and optional
truth preparation into a pair; a pipeline's `prepare` assembles the state required by that complete workflow.

## Complete and Staged Calls

Each concrete pipeline exposes its scientific parameters and prepared, raw-output
and result types. The complete call runs preparation, execution and postprocessing:

```python
from ofoldx.io import read_structure

structure, report = read_structure("backbone.cif")
result = designer(structure, num_candidates=4, seed=7)
```

For ProteinMPNN, the same operation can be run in stages:

```python
import torch

prepared = designer.prepare(structure)
output = designer.forward(prepared.features, num_candidates=4, generator=torch.Generator().manual_seed(7))
result = designer.postprocess(prepared, output)
```

The generator in this example is for a CPU pipeline. Use a generator on the
sampling device when running elsewhere. Workflows whose preparation draws random
values use one generator through preparation and execution; consult their
concrete signatures. Existing `fold`, `design`, `score` and `pack` methods remain
available as the corresponding complete calls.

`postprocess` receives preparation information to preserve component identity,
source coordinates and scientific scope. `forward` may run sampling, optimize
with gradients or execute complete child pipelines. Each stage describes a responsibility; the concrete implementation determines its device and inference or gradient
mode.

Preparation policies also travel with the prepared value. For example, the MPNN
design-and-pack workflows retain `filter_by_occupancy=False` from `prepare` for
subsequent packing. Set the shared observation policy during preparation so
design and packing use the same observations.

## Composition

A design-and-pack workflow can be assembled from existing components:

```python
from ofoldx.models.proteinmpnn import ProteinMpnnDesignAndPackPipeline

flow = ProteinMpnnDesignAndPackPipeline(designer=designer, packer=pipeline("pack", model="oteam/mpnn-packer"))
result = flow(structure, num_candidates=4, seed=7)
```

`flow.components` exposes its named children. A composition can contain another
composition, provided it satisfies the role's scientific input and output
requirements. Equal task names alone do not establish compatibility: a packing
stage needs the original backbone and exact designed positions as well as the
candidate sequence; a differentiable design stage needs gradient-capable execution.

```python
flow.save("./design-and-pack")
loaded = AutoPipelineForDesign("./design-and-pack")
```

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

This example keeps the sampled ProteinMPNN candidate with the lowest existing
`designed_sequence_nll` score. It retains that candidate's molecular system,
scores, design scope and design record, together with the original input-sequence score. The
composition then packs the selected sequence on the retained source backbone.
The model components below are loaded from the Hub; the custom composition is
saved locally. Supply your observed structure as `backbone.cif`.

```python
from ofoldx import AutoPipelineForDesign
from ofoldx.chemistry import ComponentLibrary
from ofoldx.models.mpnn_packer import MpnnPackerPipeline
from ofoldx.models.preparation import PreparedInput
from ofoldx.models.proteinmpnn import (
    ProteinMpnnDesignAndPackPipeline,
    ProteinMpnnDesignOutput,
    ProteinMpnnDesignResult,
    ProteinMpnnInput,
    ProteinMpnnPipeline,
)
from ofoldx.pipelines import PIPELINES


class LowestNllProteinMpnnPipeline(ProteinMpnnPipeline):
    def postprocess(
        self,
        prepared: PreparedInput[ProteinMpnnInput],
        output: ProteinMpnnDesignOutput,
        *,
        library: ComponentLibrary | None = None,
    ) -> ProteinMpnnDesignResult:
        result = super().postprocess(prepared, output, library=library)
        candidate = result.best_by("designed_sequence_nll")
        index = next(i for i, item in enumerate(result.candidates) if item is candidate)
        return ProteinMpnnDesignResult(
            candidates=(candidate,),
            input_score=result.input_score,
            records=(result.records[index],),
        )


pair = ProteinMpnnPipeline.from_pretrained("oteam/proteinmpnn")
designer = LowestNllProteinMpnnPipeline(model=pair.model, processor=pair.processor, source=pair.source)
print(designer.task)  # "design", inherited before registration
PIPELINES.register("lowest_nll_proteinmpnn", pipeline=LowestNllProteinMpnnPipeline)

flow = ProteinMpnnDesignAndPackPipeline(
    designer=designer,
    packer=MpnnPackerPipeline.from_pretrained("oteam/mpnn-packer"),
)
result = flow(structure, num_candidates=4, num_packs_per_candidate=1, seed=7)
flow.save("./lowest-nll-design-and-pack")

loaded = AutoPipelineForDesign("./lowest-nll-design-and-pack")
print(loaded.components["designer"].id)  # "lowest_nll_proteinmpnn"
result = loaded(structure, num_candidates=4, num_packs_per_candidate=1, seed=7)
```

Sampling still draws four candidates; the overridden stage selects one before
packing. Stable score ties retain the original sampling order. The custom registration leaves the architecture
default unchanged for ordinary ProteinMPNN repositories. In a fresh process, import the module that defines and registers
the custom class before loading its saved composition.

## Inference Compute

OFoldX exposes concrete ways to control inference-time computation:

| Technique                        | Existing scientific operation                              |
| -------------------------------- | ---------------------------------------------------------- |
| Candidate sampling and selection | Family sample counts and Result `best` / `best_by`         |
| Model ensembles                  | AlphaFold2 checkpoint ensemble and ESMFold2 design critics |
| Gradient optimization            | ESMFold2 binder sequence optimization                      |
| Beam and particle methods        | OInvFold beam decoding and Boltz sampling controls         |
| Early exit                       | ESMFold2 diffusion RMSD stopping controls                  |

These controls belong to the concrete family workflow. A custom pipeline can
combine compatible child stages, score their candidates and decide whether to
allocate additional computation. Preserve the required source structures, scopes,
output meanings and random streams when handing candidates between operations.

Each composition defines which compatible stages it consumes. For example,
AlphaFold2 CASP14 prepares one shared input for its five checkpoints, while
BoltzGen prepares its diverse and adherence workflows separately. A design
workflow's differentiable model computations remain distinct from inference-only
sampling stages. Inspect the concrete stage signatures and scientific outputs
when replacing a child.

A repository saved with `save` contains the workflow and its model components. Continuing an active search or
optimization also requires its optimizer state, random stream, and schedule. Consult the concrete workflow's API
for continuation support; returned logits and trajectories describe scientific outputs rather than that full
execution state.
