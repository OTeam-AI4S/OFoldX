# Chemistry

`ofoldx.chemistry` resolves chemical component definitions, constructs molecular graphs from explicit recipes, and
imports SMILES. It gives model preparation a chemical graph with stated identities, bonds, and knowledge states.
Coordinates and observations belong to the canonical values in [Data](../data/README.md).

## Usage

Import a ligand definition directly from SMILES:

```python
from ofoldx.chemistry import read_smiles

imported = read_smiles("CCO", component_id="LIG")
component = imported.component
print(component.component_id, component.atom_names)
print(len(imported.records))
```

The import returns a component definition and records describing the read. To construct a polymer, state its
sequence and physical instances. Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

```python
from ofoldx.chemistry import Recipe, build_system, polymer_sequence
from ofoldx.data import MonomerType, Structure

sequence = polymer_sequence("GA", MonomerType.PROTEIN)
recipe = Recipe(entities={"protein": sequence}, instances={"A": "protein"})
system, definitions, records = build_system(recipe)
target = Structure.unobserved(system)
```

`system` holds the molecular graph. `target` makes its unavailable coordinates explicit; prediction pipelines can
prepare such a target according to their own scientific requirements.

## Definitions and Recipes

- `ChemicalComponent` describes one component's atoms, bonds, stereochemistry, reference geometry, and knowledge
  states. `ComponentLibrary` reads a versioned CCD artifact; `Definitions` records the definitions used in a build
  or read.
- `PolymerSequence` gives an entity's ordered component identities. `polymer_sequence` converts an explicitly typed
  protein, RNA, or DNA alphabet into CCD component IDs. Ambiguous letters are refused where no single component is
  determined.
- `Recipe` separates entity definitions from their named physical instances. Reusing one entity in several
  instances describes copies of the same sequence. `RecipeBond` states additional bonds and their declared leaving
  atoms.
- `Molecule` lists a non-polymer entity's components. Caller definitions, including `read_smiles(...).component`,
  can be supplied to `build_system` through its `components` argument.

SMILES syntax, chemical identity, reference conformers, and observed coordinates carry distinct evidence. The
definition and construction APIs retain unknown facts and report contradictions. A default structure read applies
no consumer assumptions; select explicit `AssumptionSet` rules when a task requires them.

Reference conformer embedding and RDKit conversion operate on the chemical information their concrete functions
require. A generated conformer does not become an experimental observation.

## Related Modules

- [Data](../data/README.md): molecular values, graph edits, projections, and coordinate evidence.
- [I/O](../io/README.md): read source definitions and structure observations together.
- [Models](../models/README.md): family-specific preparation and chemical capacity.
