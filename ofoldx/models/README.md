# Models

Use OFoldX models to predict structures, design molecules, score candidates and
pack side chains. Each family guide describes the scientific method, available
checkpoints, inputs and results.

## Choose a Model

| Family                     | Tasks  | Input → Output                                       | Weights                                                                             |
| -------------------------- | ------ | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [MPNN Packer](mpnn_packer) | `pack` | Protein backbone + sequence → side-chain coordinates | [Hub](https://huggingface.co/oteam/mpnn-packer) · [Convert](mpnn_packer/convert.py) |

The Hub links point to an official repository for each family. Follow the family guide to select
a variant and its task-specific inputs. Composite workflows package their
required child models together.

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
supervision values separately from the input features.

## Model Classes and Molecular Inputs

Each family provides implementations of three public interfaces:

- `Model` performs the tensor computation used in training and inference.
- `Processor` builds the model's inputs from molecular data and its supporting evidence.
- `Pipeline` combines preparation, model execution and scientific results for one task.

The [chemistry guide](../chemistry/README.md) explains how to construct molecular
inputs from sequences, components and explicit connections. The
[data guide](../data/README.md) describes the molecular values shared across
families.
