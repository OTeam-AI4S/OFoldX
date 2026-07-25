# models

`ofoldx.models` is the minimal model-registration namespace. Callers import a
concrete family explicitly, for example `ofoldx.models.proteinmpnn`.

Each architecture keeps its source, configuration, processor, converters, and
scientific documentation under its family package. Registration is a publication
boundary: the presence of a package, prototype, or import path does not claim that
the complete family contract has been satisfied. Family READMEs own released
architecture and variant facts; this package README does not duplicate a status
inventory.

## Contents

- `registry.py` — the small Model/Processor identity mapping exposed as `MODELS`. Its
  `register` method records architecture identity and the exact Model/Processor pair,
  not family method signatures, tensor contracts, config fields, PyTrees, or placement.
- one subpackage per model family, containing its public architecture, conversion,
  processing, and scientific documentation when those parts exist.
