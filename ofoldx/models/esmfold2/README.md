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
`ESMFold2Input`, or attach a compatible frozen language model with
`set_language_model()`. `load_esmc()` intentionally raises until OFoldX has a
native ESMC implementation.

Full model parity still requires a native ESMC path and a same-input/same-output
comparison against the reference release before publication.
