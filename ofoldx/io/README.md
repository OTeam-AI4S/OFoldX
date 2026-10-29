# I/O

`ofoldx.io` reads and writes mmCIF, PDB, and native OFoldX structures. Read reports retain source metadata and
chemical definitions; write reports describe information that an exchange format cannot preserve.

## Usage

Read a structure and write it as mmCIF with the definitions used during the read:

```python
from ofoldx.io import read_structure, write_structure

structure, report = read_structure("input.cif")
write_report = write_structure(structure, "output.cif", definitions=report.definitions)
print(structure.coordinates.shape)
print(len(write_report.records))
```

Supply your `input.cif`. Chemical definitions come from the cached `oteam/ccd` dataset by default.
The [CCD guide](../reference/ccd/README.md) explains local library overrides and offline use. An explicit `library=` can
also be supplied to a read or write.

## Reading Structures

[`read_structure`][ofoldx.io.read.read_structure] returns a `Structure` and a `ReadReport`. It accepts a file path
or a parsed source. Parse an ensemble once when reading several models:

```python
from ofoldx.io import read_source, read_structure

source = read_source("ensemble.pdb")
reads = [read_structure(source, model=number) for number in source.models]
```

| Option        | Selection                                                            |
| ------------- | -------------------------------------------------------------------- |
| `model`       | Model number; the first model by default                             |
| `assembly`    | mmCIF assembly ID or PDB biomolecule; deposited instances by default |
| `altloc`      | Highest occupancy or a named alternate location                      |
| `components`  | Caller-supplied component definitions                                |
| `on_conflict` | Raise on conflicts or retain affected facts as undetermined records  |
| `library`     | An explicit CCD library                                              |
| `assumptions` | Explicit consumer assumptions; the default read applies none         |

For example, `components={"LIG": read_smiles("CCO", component_id="LIG").component}` supplies a ligand definition
created with [`read_smiles`][ofoldx.chemistry.smiles.read_smiles].

The default `on_conflict="raise"` reports conflicting declarations with `ContradictionError(read_conflicts)`.
Connections admitting several readings, such as a blank-symmetry `SSBOND` combined with a `CONECT` declaration of
the same pair, raise `InsufficientInformationError(read_ambiguous_connections)`. `on_conflict="record"` retains
those findings and leaves the affected facts undetermined.

The report provides `conflicts()`, `unknowns()`, and `unrealised()` views, document metadata, component definitions,
and the outcome of connection declarations. Select chains, water, and hydrogens on the returned structure with
`structure.select(chains=..., water=False, hydrogens=False)`; see [Data](../data/README.md).

## Writing and Native Artifacts

[`write_structure`][ofoldx.io.write.write_structure] returns a `WriteReport`. Its records identify information that
mmCIF or PDB cannot carry, aggregated by chain instance for repeated atom or residue losses. Required identifiers
must be present, or explicitly assigned with `identity="assign"`; assignments are included in the report.

For a structure read from an exchange format, writing that format and reading it under the same CCD library,
assumptions, and configuration preserves supported facts. The report identifies losses, including a frame stated
only through `_atom_sites.Cartn_transf_*` and unknown declared branch-link orders. Projections, chemical edits, and
model outputs are written with the information that the chosen format supports.

Pass `definitions=report.definitions` when writing mmCIF to retain definitions that differ from the selected CCD
library. A definition equal to its library entry can be recovered from that library on the next read.

Use native artifacts to retain the full `Structure` value:

```python
from ofoldx.io import load_definitions, load_structure, save_structure

save_structure(structure, "structure.safetensors", definitions=report.definitions)
restored = load_structure("structure.safetensors")
definitions = load_definitions("structure.safetensors")
print(restored.coordinates.shape)
```

## CIF Syntax

[`read_cif`][ofoldx.io.cif.read_cif] and [`parse_cif`][ofoldx.io.cif.parse_cif] expose CIF blocks and categories.
`TokenState` distinguishes a stated value, unknown `?`, inapplicable `.`, and an item absent from its category.
Sequence alignments have their own readers in [MSA](../msa/README.md).

[ofoldx.chemistry.smiles.read_smiles]: https://ofoldx.valtech.ai/chemistry/smiles/#ofoldx.chemistry.smiles.read_smiles
[ofoldx.io.cif.parse_cif]: https://ofoldx.valtech.ai/io/cif/#ofoldx.io.cif.parse_cif
[ofoldx.io.cif.read_cif]: https://ofoldx.valtech.ai/io/cif/#ofoldx.io.cif.read_cif
[ofoldx.io.read.read_structure]: https://ofoldx.valtech.ai/io/read/#ofoldx.io.read.read_structure
[ofoldx.io.write.write_structure]: https://ofoldx.valtech.ai/io/write/#ofoldx.io.write.write_structure
