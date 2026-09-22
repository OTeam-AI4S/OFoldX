# Data

`ofoldx.data` represents molecular identity, chemical connectivity, coordinates, and the evidence behind them.
Use these values to select chains, combine molecular systems, and pass structures between readers and models.

## Usage

Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact before reading a structure.

Read a structure and select author chain `A`, omitting water and folding explicit hydrogens into hydrogen counts:

```python
from ofoldx.io import read_structure

structure, report = read_structure("input.cif")
protein = structure.select(chains=["A"], water=False, hydrogens=False)
print(protein.coordinates.shape)
print(protein.coordinate_mask.sum())
```

Supply a structure containing author chain `A`. Selection returns the selected `Structure` and leaves the source unchanged. Use `namespace="label"` when selecting label asym IDs instead of author chain names.

## Molecular Values

| Value                                                      | Purpose                                                                                       |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [`MolecularSystem`][ofoldx.data.molecular.MolecularSystem] | Molecular graph and identities, independent of coordinates                                    |
| [`Structure`][ofoldx.data.structure.Structure]             | A molecular system with coordinates, observations, source identity, and an experimental frame |
| [`TokenLayout`][ofoldx.data.tokenization.TokenLayout]      | Correspondence between canonical components and the tokens consumed by a model                |

The molecular hierarchy is system → entity → instance → component → atom. An entity describes a molecule;
instances describe its physical copies. Bonds, external connections, stereochemistry, and reference conformers
retain their relationship to that hierarchy.

Values use immutable NumPy columns. [`ArrayRecord`][ofoldx.data.schema.ArrayRecord] validates published copies;
build buffers provide writable arrays for constructing a new value. Symbol codes refer to per-family vocabularies; operations preserve their meaning when remapping or merging tables. See the [schema reference][ofoldx.data.schema] for column declarations and validation.

## Evidence and Missing Information

A fact's state distinguishes known, unknown, inapplicable, and conflicting information, and records its basis.
Unavailable values use canonical fill values; inspect the state before treating a payload as a measurement.
Coordinate evidence separately identifies observed, predicted, generated, reference, and unavailable positions.

Operations preserve known facts where applicable and report facts invalidated by a change. Selecting atoms can
narrow component completeness. A bond cut between components becomes an external connection; removing a hydrogen
folds it into the retained atom's count. Selecting atom-level fragments does not reassemble a complete component.
The [projection reference][ofoldx.data.projection] describes these cases and their returned index maps.

## Working with Values

| Operation                                                        | Use                                                              |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| `select`, `project_atoms`, `project_instances`, `fold_hydrogens` | Select molecular views while retaining identity and connectivity |
| `union`, `replicate`, `batch`, `unbatch`                         | Join systems, place copies, and group independent examples       |
| `apply_edit`                                                     | Apply an explicit chemical edit with declared changes            |
| `replace_coordinates`                                            | Replace a selected range of coordinates and update its evidence  |
| `tokenize`, `token_classes`                                      | Build token layouts and sequence classes for model input         |

[Assembly][ofoldx.data.assembly] reconnects matching external identities only at the stated placement.
[Editing][ofoldx.data.editing] applies the supplied chemical declarations. [Stereochemistry][ofoldx.data.stereo]
documents ordered neighbor slots, parity transport, and reasons a stereocenter may become undetermined.

Invalid requests, unsupported representations, and insufficient placement information use the named errors in
[`ofoldx.errors`][ofoldx.errors]. Each operation documents its failure conditions.

## Related Modules

- [I/O](../io/README.md): read, write, and persist structures.
- [Chemistry](../chemistry/README.md): component definitions and construction recipes.
