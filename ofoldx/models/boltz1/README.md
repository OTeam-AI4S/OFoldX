# Boltz-1

Boltz-1 predicts all-atom structure for tokenized biomolecular complexes: a token/atom
input embedder feeds a recycling representation trunk (Pairformer plus an optional MSA
module) and a distogram head, then an EDM reverse-diffusion sampler denoises coordinates,
which the Boltz-1 confidence/ranking head scores.

Distinctive to this family are its atom-attention input embedder, precomputed diffusion
conditioning, the EDM coordinate sampler, the confidence/ranking head, and optional Boltz-1x
inference-time steering (Feynman-Kac particles and physical-potential guidance).

Shared neural network components live in `ofoldx.modules`; this directory holds the
family's config, typed I/O, checkpoint conversion, and model.
