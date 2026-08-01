# I/O

[`read_source`][ofoldx.io.source.read_source] and
[`parse_source`][ofoldx.io.source.parse_source] retain original CIF/PDB facts for
training, inference and independent model consumers. Source reading preserves
models, alternate locations, identity and missing-value evidence without
constructing a chemical graph.

[`read_structure`][ofoldx.io.structure.read_structure] materializes a selected
source into a molecular graph and aligned coordinates.
[`write_structure`][ofoldx.io.structure.write_structure] writes that projection
through mmCIF or PDB, rejecting evidence the selected format cannot represent.

[`read_template_coordinates`][ofoldx.io.template.read_template_coordinates] and
[`read_polymer_coordinates`][ofoldx.io.coordinates.read_polymer_coordinates]
provide declared polymer sequences and observed coordinates without graph
construction. Parent relationships retain complete lists; model processors
choose their own atom layouts and canonical-monomer policies.

See [sources and projections](../../docs/io_sources.md) for model selection,
source values, parent relationships and reuse. `msa.py` provides MSA reading,
pairing and indexed acquisition.
