# io

`ofoldx.io` exposes canonical PDB/mmCIF structure exchange (including
gzip-compressed `.pdb.gz` input) through
`read_structure` and `write_structure`. Readers construct a referentially closed
tensor-native `MolecularSystem` and a `Structure` aligned to its complete atom
axis. Built-in CCD components supply known chemistry; source atoms define an
unknown component without inventing connectivity. Reading exposes explicit model,
assembly, alternate-location, water, and hydrogen choices.

## Contents

- `structure.py` - public suffix dispatch and canonical tensor graph
  materialization.
- `records.py` - internal source-row normalization and identity helpers shared by
  PDB and mmCIF.
- `pdb.py` - standard PDB parsing, CONECT bonds, and writing.
- `mmcif.py` - standard mmCIF parsing, entities, assemblies, `struct_conn` bonds,
  and writing.
- `msa.py` - MSA parsing, biological-source pairing, indexed acquisition, and
  JSON exchange for explicit pairing plans.

PDB and mmCIF are the standard exchange formats for `MolecularSystem` and
`Structure`; neither format carries a private OFoldX payload. Runner manifests
may reference an `MsaCollection` encoded by `ofoldx.io.encode_msa_collection`.

Model Processors lower canonical `Structure` values into their own tensor and
scientific input contracts; the I/O namespace has no model-specific file dialect.
