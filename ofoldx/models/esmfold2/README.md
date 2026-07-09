# ESMFold2

OFoldX implementation of the released ESMFold2 structure-prediction head.

Current status:

- `ESMFold2Config` matches the public Hub `config.json`
- `ESMFold2Model` strict-loads the released folding-head checkpoint
- `io.py` adapts raw protein sequence / FASTA inputs through `ofoldx.io.read_fasta`
- `ESMFold2Processor` builds a protein-only tensor batch for smoke/e2e checks
- `convert.py` accepts a local or Hub `model.safetensors` source and writes an
  OFoldX artifact
- `AutoConfig`, `AutoModel`, and `AutoModelForStructurePrediction` can load the
  converted artifact

The released checkpoint expects ESMC-6B language-model representations. OFoldX
keeps that boundary explicit: callers can pass `lm_hidden_states` in
`ESMFold2Input`, attach a frozen language model with `set_language_model()`, or
call `load_esmc()` to load ESMC through MultiMolecule's `AutoTokenizer` /
`AutoModel` surface.

Full model parity is tracked as same-input/same-output comparison against the
reference release. Folding-head parity uses the same ESMC hidden states on both
sides; sequence-to-structure parity additionally covers the MultiMolecule ESMC
component.
