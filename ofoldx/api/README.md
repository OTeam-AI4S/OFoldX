# Task API

Load and run folding, design, scoring and side-chain packing workflows through
`ofoldx.api`. Use `pipeline` to keep a model loaded for repeated calls, or use
`score` and `pack` for a single complete operation.

## Usage

Read the protein backbone in `backbone.cif`, load ProteinMPNN and inspect its
sequence score:

```python
from ofoldx import pipeline
from ofoldx.io import read_structure

structure, report = read_structure("backbone.cif")
scorer = pipeline("score", model="oteam/proteinmpnn")
result = scorer(structure, seed=7)
nll = result.scores["designed_sequence_nll"]
print(nll.value, nll.unit)
```

Keep `scorer` to evaluate more structures with the same model. For one call,
`score` reads the file, loads the workflow and returns its result:

```python
from ofoldx import score

result = score("backbone.cif", model="oteam/proteinmpnn", seed=7)
print(result.scores["designed_sequence_nll"].value)
```

Model files and the [CCD artifact](../reference/ccd/README.md) are downloaded and cached
on first use.

## Tasks and Inputs

| Task     | Operation                                 | Typical Input                              |
| -------- | ----------------------------------------- | ------------------------------------------ |
| `fold`   | Predict coordinates                       | A family's sequence or molecular input     |
| `design` | Generate sequence identities or structure | A family's design input and controls       |
| `score`  | Evaluate a supplied target                | An observed structure and scoring controls |
| `pack`   | Generate side-chain coordinates           | A structure and selected packing positions |

The repository and task select a concrete pipeline. Each family guide explains
its inputs, controls and result type. For options such as ProteinMPNN's number
of decoding orders, call the loaded pipeline with those arguments.

## Reading Structures

The `score` and `pack` convenience functions accept a `Structure`, an mmCIF/PDB
path, or a native OFoldX structure artifact ending in `.safetensors`. For an
assembly, altloc selection or another reading policy, call `read_structure`
first and retain its report. The [I/O guide](../io/README.md) explains those
choices and the handling of conflicting evidence.

## Loading and Sampling Options

For a Hub model, `revision` selects a branch, tag or commit. `device` and `dtype`
set model and processor placement, while `backend` selects an operator provider;
see the [backend guide](../modules/backend/README.md).

Use a seed to initialize sampling or pass a device-compatible `torch.Generator`
to keep a random stream across calls. The family guide describes its sampling
controls.

## Related Modules

- [Models](../models/README.md): model families, checkpoints and conversion.
- [Pipelines](../pipelines/README.md): staged execution, composition and repository saving.
- [I/O](../io/README.md): structure reads, reports and exported files.
- [Results](../results/README.md): candidates, scores and molecular metrics.
- [Hub](../hub/README.md): repository snapshots and file publication.
