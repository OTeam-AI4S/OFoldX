# io

`ofoldx.io` exposes canonical PDB/mmCIF structure exchange (including
gzip-compressed `.pdb.gz` input) through
`read_structure`, `read_structures`, `write_structure`, and `LossReport`.
Readers construct referentially closed `MolecularSystem`/`Structure` values from
built-in or explicitly supplied component graphs. An observed layout represents an
otherwise unknown component from its source atoms without inventing connectivity.
Readers expose explicit model, assembly, atom-layout, and alternate-location
policies. mmCIF preserves the complete canonical value; PDB refuses unacknowledged
loss.

## Contents

- `structure.py` - public suffix dispatch, alternate-location and atom-layout
  policies, canonical Structure materialization, and the `LossReport` facade.
- `records.py` - public format-neutral `AtomRecord` and `ParsedStructure`
  values, record normalization, source-site mapping, identity generation, and
  output-bond ordering shared by PDB and mmCIF.
- `pdb.py` - generic PDB parsing, CONECT bonds, lossy-field analysis, and writing.
- `mmcif.py` - generic mmCIF parsing, canonical payload decoding, branched entities,
  `struct_conn` bonds, and writing.
- `msa.py` - `read_msa`.

Canonical value serialization is the public `ofoldx.data.codec` boundary. It
encodes and decodes `MolecularSystem`, `Structure`, MSA evidence, and templates;
the mmCIF exchange path uses it for embedded canonical payloads.

Model Processors lower canonical `Structure` values into their own tensor and
scientific input contracts; the I/O namespace has no model-specific file dialect.
