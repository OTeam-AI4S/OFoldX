# Boltz-2

Boltz-2 is a Boltz-2-shaped folding model: an atom-conditioned input embedder builds
the initial single/pair representations, a recycled trunk (template + MSA + Pairformer)
refines them, an EDM diffusion sampler draws atom coordinates, and a token-level
confidence head scores the sample.

Distinctive to this family are its atom-conditioned input embedder, V2 template embedder,
the EDM sampler with in-loop weighted rigid alignment, and the `Boltz2AffinityModel`
variant that adds a pair-only head scoring protein-ligand binding affinity.

Shared building blocks live in `ofoldx.modules`; this directory holds the family's
config, typed I/O, checkpoint conversion, and model.
