# io

`ofoldx.io` exposes complete canonical PDB/mmCIF structure I/O. Sequence-format I/O,
including FASTA, belongs to the MultiMolecule dependency rather than being wrapped
again here.
Readers construct referentially closed `MolecularSystem`/`Structure` values from
package-audited or explicitly supplied component graphs, with explicit model,
assembly, atom-layout, and alternate-location policies. mmCIF preserves the complete
canonical value; PDB fails by default when it cannot and returns a `LossReport` only
under explicit `allow_lossy=True`.

## Contents

- `structure.py` — canonical `read_structure`, altloc-expanding `read_structures`,
  `write_structure`, and `LossReport`. This is the general Structure-exchange entry
  point: PDB and mmCIF, explicit model / assembly / altloc / atom-layout policies,
  and an explicit fidelity contract.
- `pdb.py` — `read_pdb`, the narrower policy-free reader the released MPNN models
  parse their inputs through (hybrid-36 decoding, single-model / first-altloc,
  audited protein-component reconstruction). It is what the model Pipelines call and
  is **not** interchangeable with `read_structure`; neither reader delegates to the
  other. The symmetric packed-PDB *writer* is a bounded family visualization and
  lives with its model as `ofoldx.models.mpnn_packer.write_pdb`, not here.
- `msa.py` — `read_msa`.
