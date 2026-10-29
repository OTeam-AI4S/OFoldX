# Results

`ofoldx.results` holds scientific outputs from biological pipelines: predicted structures, designed targets,
side-chain packing samples, scalar scores, and metrics aligned to molecular axes. Concrete families may extend
these values with their own scientific records.

## Usage

After a ProteinMPNN design call, select a candidate by sequence NLL and save its target:

```python
from ofoldx import pipeline
from ofoldx.io import read_structure, save_structure

structure, report = read_structure("backbone.cif")
designer = pipeline("design", model="oteam/proteinmpnn")
result = designer(structure, num_candidates=4, seed=7)
candidate = result.best_by("designed_sequence_nll")
print(candidate.scores["designed_sequence_nll"].value)
save_structure(candidate.target, "candidate.safetensors")
```

The [CCD artifact](../reference/ccd/README.md) and official model are downloaded and cached on first use. A sequence
design's target contains its designed chemistry and available guide coordinates. Use a packing workflow when the
next step needs generated side-chain coordinates.

## Result Values

| Value                 | Contents                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------- |
| `FoldResult`          | Generation-ordered `FoldPrediction` values, each with a `structure`, scores, and metrics  |
| `DesignResult`        | Generation-ordered `DesignCandidate` values, each with a final `target` and scores        |
| `ScoreResult`         | The scored `target`, named scalar scores, and available metrics                           |
| `PackResult`          | Side-chain packing predictions, each with its generated `structure` and available metrics |
| `DesignAndPackResult` | Sequence designs and the packing results aligned to those candidates                      |

Inspect the concrete family's result type for additional records. Some ligand-generation workflows return
provisional coordinates and atom assignments before a canonical chemical graph is determined; their family
guides describe that boundary.

## Scores and Molecular Axes

`Score` carries a scalar `value`, its `unit`, a ranking direction (`preferred`), the molecular `scope`, and optional
individual `evaluations`. Score names belong to the scientific operation: a sequence NLL and an affinity score
have different meanings even though both are scalar values.

`best_by(name)` and `best(values, name)` use the named score's ranking direction. They require matching units,
scopes, and directions, preserve generation order for ties, and skip undefined NaN values when another candidate
can be ranked. Consult the API reference for empty and unrankable inputs.

`Metric` has one molecular axis. `PairMetric` has separate row and column axes. Keep those references and units
alongside the numerical arrays when exporting confidence values or constructing analysis tables.
`AutoregressiveProbabilityResult` similarly retains the canonical component references, masks, logits, and
decoding ranks for each sequence evaluation.

## Saving and Reusing Outputs

Use `save_structure` and `load_structure` for the full canonical `Structure` value, or `write_structure` for mmCIF
and PDB exchange. The latter returns a write report describing format limitations. Scores and metric arrays remain
in the Result; include their names, units, scopes, and axes in any JSON, CSV, or NumPy export you create.

A pipeline's `save` method persists its executable workflow and model components. Reload that repository for
operations supported by its registered task and architecture.

## Related Modules

- [Pipelines](../pipelines/README.md): complete calls, stages, and design-and-pack composition.
- [I/O](../io/README.md): native structures and format-aware writes.
- [Metrics](../metrics/README.md): scientific measurements on predictions.
- [Analysis](../analysis/README.md): sequence, structure, and diversity analyses.
