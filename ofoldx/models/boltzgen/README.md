# BoltzGen

BoltzGen is a structure-based biomolecular design family with two generators: a
diffusion structure designer that samples complex coordinates under design
conditioning, and an inverse-folding designer that decodes per-token residue
types from an input backbone.

The structure designer adds a v1 template module, a token-distance module, and
binding / design-mask / secondary-structure conditioning in the input embedder,
with opt-in confidence and affinity scoring heads. The inverse-folding path
encodes the structure into a kNN token graph and autoregressively samples
sequences.

Shared building blocks live in `ofoldx.modules`; this directory holds the
family's config, typed I/O, checkpoint conversion, and model.
