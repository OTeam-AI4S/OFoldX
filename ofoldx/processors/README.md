# processors

`ofoldx.processors` turns raw structures — mmCIF/PDB files or in-memory `Structure`
objects — into the typed tensor inputs a model consumes, and holds the cross-family
machinery that resolves and round-trips those processors: `AutoProcessor`, the `PROCESSORS`
registry, the `ProcessorMixin` save/load protocol, and the shared featurizers for
structure tokenization, MSAs, and templates.

The boundary is deliberate. This package knows about **zero specific models**. A
model-specific processor lives beside its family in
[`ofoldx.models`](../models)`.<family>.processor` and self-registers with `PROCESSORS`;
resolve it by architecture through `AutoProcessor`, or import it directly. Only the
processing steps genuinely **shared across two or more families** — structure tokenization,
MSA parsing/featurization, template search — belong here. See
[`DESIGN_PRINCIPLES.md`](../../DESIGN_PRINCIPLES.md) for the full contract.

## Contents

- `auto.py`, `registry.py` — `AutoProcessor` dispatch and the `PROCESSORS` registry, keyed by a config's `architecture`.
- `mixin.py` — `ProcessorMixin`, the Hugging Face-style pretrained save/load round-trip.
- `structure.py` — `StructureProcessor`, the shared structure-to-fields featurizer.
- `residue_token.py` — `ResidueTokenStructureProcessor`, the residue-token single-sequence input contract.
- `msa.py` — parse precomputed a3m/CSV alignments (`parse_a3m`, `parse_csv`) and featurize them (`featurize_msa`).
- `msa_tools.py` — HMMER wrappers (`Jackhmmer`, `Nhmmer`, `Hmmbuild`, `Hmmsearch`, `Hmmalign`) for MSA generation.
- `template.py` — structural-template search, hit parsing, and date/max-hits filtering.
- `utils.py` — shared structure I/O, `StructuralFields`/`StructureToken`, and the `ProcessorOutput` protocol.
