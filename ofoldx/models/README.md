# Models

Use OFoldX models to predict structures, design molecules, score candidates and
pack side chains. Each family guide describes the scientific method, available
checkpoints, inputs and results.

## Choose a Model

| Family                       | Tasks             | Input → Output                                         | Weights                                                  |
| ---------------------------- | ----------------- | ------------------------------------------------------ | -------------------------------------------------------- |
| [AlphaFold 2](alphafold2)    | `fold`            | Protein system + MSA/templates → structure             | [Hub](https://huggingface.co/oteam/alphafold2)           |
| [AlphaFold 3](alphafold3)    | `fold`            | Molecular system + evidence → all-atom structure       | [Model card](alphafold3)                                 |
| [Boltz-1](boltz1)            | `fold`            | Molecular complex → structure + confidence             | [Hub](https://huggingface.co/oteam/boltz1)               |
| [Boltz-2](boltz2)            | `fold`, `score`   | Molecular complex → structure or affinity              | [Hub](https://huggingface.co/oteam/boltz2)               |
| [ESMFold2](esmfold2)         | `fold`, `design`  | Molecular input / binder target → structure / sequence | [Hub](https://huggingface.co/oteam/esmfold2)             |
| [Protenix](protenix)         | `fold`            | Molecular system + evidence → all-atom structure       | [Hub](https://huggingface.co/oteam/protenix)             |
| [RFdiffusion](rfdiffusion)   | `design`          | Contigs + optional motifs → protein backbone           | [Hub](https://huggingface.co/oteam/rfdiffusion-base)     |
| [ODesign](odesign)           | `design`          | Molecular context → polymer or ligand designs          | [Hub](https://huggingface.co/oteam/odesign-protein-flex) |
| [PXDesign](pxdesign)         | `design`          | Target + binder specification → protein designs        | [Hub](https://huggingface.co/oteam/pxdesign)             |
| [BoltzGen](boltzgen)         | `design`          | Binder specification → ranked molecular designs        | [Hub](https://huggingface.co/oteam/boltzgen)             |
| [ProteinMPNN](proteinmpnn)   | `design`, `score` | Protein backbone → sequence or sequence NLL            | [Hub](https://huggingface.co/oteam/proteinmpnn)          |
| [LigandMPNN](ligandmpnn)     | `design`, `score` | Backbone + atomic context → sequence or NLL            | [Hub](https://huggingface.co/oteam/ligandmpnn)           |
| [MembraneMPNN](membranempnn) | `design`, `score` | Backbone + membrane labels → sequence or NLL           | [Hub](https://huggingface.co/oteam/membranempnn-global)  |
| [NA-MPNN](nampnn)            | `design`, `score` | Protein/DNA/RNA backbone → sequence or NLL             | [Hub](https://huggingface.co/oteam/nampnn)               |
| [MPNN Packer](mpnn_packer)   | `pack`            | Protein backbone + sequence → side-chain coordinates   | [Hub](https://huggingface.co/oteam/mpnn-packer)          |
| [gRNAde](grnade)             | `design`          | RNA conformations + base pairs → RNA sequences         | [Hub](https://huggingface.co/oteam/grnade)               |
| [OInvFold](oinvfold)         | `design`          | Fixed coordinates → polymer or atom identities         | [Hub](https://huggingface.co/oteam/oinvfold-protein)     |

The Hub links point to an official repository for each family. AlphaFold 3 and
Protenix-v2 use locally converted parameters. Follow the family guide to select
a variant and its task-specific inputs. Composite workflows package their
required child models together.

## Load and Run a Model

Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact before reading a structure.

Read a protein backbone from `backbone.cif` and score its sequence with the
official ProteinMPNN model:

```python
from ofoldx import pipeline
from ofoldx.io import read_structure

structure, report = read_structure("backbone.cif")
scorer = pipeline("score", model="oteam/proteinmpnn")
result = scorer(structure, seed=7)
nll = result.scores["designed_sequence_nll"]
print(nll.value, nll.unit)
```

Model files are downloaded and cached on first use. The [standalone scoring example](../../examples/pipelines/proteinmpnn.py)
writes named scores, their units and molecular scopes to JSON. Its `--model`
argument also accepts a native local export.

The [pipeline guide](../pipelines/README.md) describes preparation, execution,
postprocessing and workflow composition. Use a family's concrete classes when
you need direct access to its model or input processor.

To load the model and processor separately:

```python
from ofoldx import AutoModel, AutoProcessor

model = AutoModel("oteam/proteinmpnn")
processor = AutoProcessor("oteam/proteinmpnn")
```

To initialize a new model instead of loading weights, pass its configuration:

```python
from ofoldx.models.proteinmpnn import ProteinMpnnConfig

model = AutoModel(ProteinMpnnConfig())
```

A model created from configuration starts in training mode; a model loaded from a repository is in evaluation mode.
`from_pretrained` is also available on the Auto classes and concrete Pipelines for Hugging Face-style loading.

## Processor Data Preparation

Processors use three data operations for preparation:

```python
input = processor.prepare_input(structure, **task_options)
target = processor.prepare_target(truth, prepared=input)
input, target = processor.prepare(structure, target=truth, **task_options)
```

Use the processor and task options for your chosen family. `prepare_input` returns a `PreparedInput` with
`features` for the task and a `context` that maps them to the molecular structure. `prepare_target` aligns
truth to that context, reusing the same crop and molecular correspondence. The supported targets depend on
the model family; processors without target preparation raise an error when a target is supplied.

`prepare` combines these operations and returns `(input, target)`. Without an explicit target, it returns
`(input, None)`. To use the source structure as truth, pass `target=structure` explicitly. Targets contain
supervision values separately from the input features. The [Runner guide](../runners/README.md#data-preparation)
explains how objectives use these values for training and evaluation.

## Model Classes and Molecular Inputs

Each family provides implementations of three public interfaces:

- `Model` performs the tensor computation used in training and inference.
- `Processor` builds the model's inputs from molecular data and its supporting evidence.
- `Pipeline` combines preparation, model execution and scientific results for one task.

The [chemistry guide](../chemistry/README.md) explains how to construct molecular
inputs from sequences, components and explicit connections. The
[data guide](../data/README.md) describes the molecular values shared across
families.
