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
  `write_structure`, and `LossReport`.
