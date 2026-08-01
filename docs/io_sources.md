# Sources and projections

[`read_source`][ofoldx.io.source.read_source] reads CIF or PDB, including gzip
input. [`parse_source`][ofoldx.io.source.parse_source] accepts text in memory.
Both retain the source document for independent consumers without requiring
chemical completion, a model processor, or a training filter.

```python
from ofoldx.io import read_source, read_structure

source = read_source("input.cif.gz")
view = source.view(model_index=1)
rows = view.rows
structure = read_structure(view, include_water=False, include_hydrogens=False)
```

[`read_structure`][ofoldx.io.structure.read_structure] constructs a molecular
graph and aligned coordinate evidence. It can reject insufficient or
contradictory chemical evidence while the source remains available to inspect.
An explicit `components` mapping or `component_resolver` supplies chemistry.
[`write_structure`][ofoldx.io.structure.write_structure] writes that molecular
projection; it does not reproduce the original source document.

## Source values

A [`CifSource`][ofoldx.io.source.CifSource] retains CIF categories through
`source.block`; a [`PdbSource`][ofoldx.io.source.PdbSource] retains `text` and
`lines`. The common view exposes selection and deposited rows only. CIF declarations
and namespace indices belong to the source: `source.declarations`,
`source.connections`, and `source.atoms.index(model_index)`. PDB projection
through a CIF-only consumer raises
[`UnsupportedFormatError`][ofoldx.io.source.UnsupportedFormatError].

```python
from ofoldx.io import parse_cif

column = parse_cif("data_x\nloop_\n_a.v\n?\n.\n'?'\n1.25\n").block["a"]["v"]
assert column.values == (None, None, "?", "1.25")
assert column.mask.tolist() == [2, 1, 0, 0]
```

[`CifColumn`][ofoldx.io.cif.CifColumn] exposes original tokens as `raw_values`
and decoded text as `values`. Whitespace and quoted literals are preserved.
Its mask distinguishes `PRESENT=0`, `INAPPLICABLE=1`, and `UNKNOWN=2`;
`name in category` separately detects an absent item. Constructors take decoded
values by default: `None` means unknown and `False` means inapplicable.
Use `raw=True` for lexical tokens.

Numeric conversion requires an explicit fill when nulls occur. Present values
must be valid, unquoted numbers; an invalid number cannot become missing merely
because a fill was provided. `uncertainty` and `uncertainty_present` retain
standard uncertainties. For example, `1.23e2(4)` and `1.23(4)e2` both describe
123 with uncertainty 4. Numeric source columns use float64 or signed int64
before conversion to the requested dtype.

Source values are read-only snapshots. A missing coordinate axis makes the
observation invalid; its other original axis values remain inspectable in CIF.
Author residue identifiers and atom serials remain text, including leading zeros.
An absent CIF occupancy item has its declared default of 1; explicit `?` and `.`
cells remain missing.

## Models and views

`source.view(model_index=2)` selects the second model in source order;
`source.view(model_id=12)` selects stored identifier 12. These options are
exclusive. The default is the first model. A PDB without MODEL records has ID
`None` and ordinal 1. Interleaved atom rows do not change model selection.

`asym_ids` selects CIF label chain IDs or native PDB chain IDs. View rows retain
deposited observations and all alternate locations. `assembly_id` requests
expansion by a graph consumer; it does not transform source rows. When a view
is supplied to `read_structure`, model and assembly options belong to that view.

CIF entity declarations distinguish an absent polymer sequence from an empty
one. Selection retains the entities and numbering referenced by selected chains.
Label and author identity evidence must agree; filtering a chain cannot hide a
contradictory connection declaration.

## Polymer coordinates and parents

[`read_template_coordinates`][ofoldx.io.template.read_template_coordinates]
projects declared protein chains without constructing a graph:

```python
from ofoldx.io import read_template_coordinates

chains = read_template_coordinates(source, ["A"], chain_id_kind="label")
chain = chains["A"].chains["A"]
coordinates, present = chain.gather([0, 1], ["N", "CA", "C"])
```

The mapping uses requested chain identifiers; each result retains label chain
IDs. `chain_id_kind="auth"` requests author naming and rejects ambiguous
matches. Individual chain errors may be handled with `on_error(chain, error)`;
source parsing, malformed selected-model observations, contradictory chain
authorship and missing declarations raise before callbacks. Unknown alternative
author names cannot establish a unique chain match.

[`read_polymer_coordinates`][ofoldx.io.coordinates.read_polymer_coordinates]
also supports declared nucleotide polymers. This projection requires CIF
sequence declarations and deposited chains. PDB observations are available as
`source.view().rows`; their SEQRES correspondence remains a graph consumer's
responsibility.

[`ChainCoordinates`][ofoldx.data.coordinates.ChainCoordinates] retains declared
sequence positions, observed atom names, coordinates and masks. A missing atom
has a false mask in `gather`; it is not an observed zero. Each position's
`parent_component_ids` contains every declared parent, including an empty tuple
when none is declared. No parent is chosen implicitly.

[`read_component_parents`][ofoldx.io.declarations.read_component_parents] is the
shared CIF decoder. Structure stores these relationships as
`component_parent_ptr` and `parent_component_id`, with IDs referring to
`source_symbols`. Converting to coordinate evidence preserves each full parent
list. Multiple parents describe constituent relationships, not alternative monomer
choices. Consumers requiring single-letter sequences must explicitly handle inputs
that their alphabet cannot represent.

[`Template`][ofoldx.data.evidence.Template] binds immutable coordinate evidence
to alignments. Use `from_structure` to take one explicit snapshot of a mutable
Structure; later changes to that Structure do not alter the template.

```python
from ofoldx.data import Template, TemplateAlignment

alignment = TemplateAlignment("query", None, "A", ((0, 0), (1, 1)))
template = Template(coordinates=chains["A"], alignments=(alignment,))
snapshot = Template.from_structure(structure, alignments=(alignment,))
```

Model processors choose their own dense atom layouts. The explicit
`read_template_structure` and `read_template_structures` conveniences remain
available for callers requiring a Structure representation.
