# ProteinMPNN

ProteinMPNN is an inverse-folding model that designs and scores amino-acid
sequences for a fixed protein backbone. A geometric message-passing encoder reads
the backbone over a k-nearest-neighbor residue graph, and an autoregressive decoder
produces per-residue probabilities, sequence scores, and sampled sequences.

Distinctive here are its backbone-feature encoder, encoder/decoder message-passing
stacks, and the full sampling surface: standard and single-residue scoring,
autoregressive sampling with fixed positions, per-residue bias, PSSM, and omit-AA
conditioning, plus symmetric and tied-position decoding.

Shared building blocks live in `ofoldx.modules`; this directory holds the
family's config, typed I/O, checkpoint conversion, and model.
