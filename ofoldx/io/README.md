# io

`ofoldx.io` reads structure files into OFoldX's typed
[`Structure`](../data/structure.py) — heavy atoms plus explicit inter-residue bonds. Featurization
belongs to the structure processor, not here; the JSON input parsers and the predicted-structure
writers are model-specific and live with their families
(`models/protenix/json_input.py`, `models/pxdesign/json_input.py`,
`models/pxdesign/write.py`, `models/ligandmpnn/write.py`).

## Contents

- `structure.py` — `read_structure` / `read_mmcif` / `read_pdb`: parse PDB / mmCIF into a `Structure`.
- `fasta.py` — `read_fasta` / `parse_fasta`: parse FASTA records.
