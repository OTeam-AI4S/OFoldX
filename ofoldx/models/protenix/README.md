# Protenix

Protenix is the AlphaFold 3-shaped Protenix v0.5 structure predictor. The base checkpoint maps onto the
AF3-style trunk, EDM diffusion sampler, and distogram/confidence heads. The model directory also owns the
Protenix constraint pair-bias path: pocket, token-contact, atom-contact, and substructure features can be
embedded and added to the initial pair representation before the recycling trunk, and inference can
optionally apply Protenix training-free guidance.

Distinctive to this family are its guided EDM sampling loop, ten-cycle recycling trunk with no template
stack by default, and optional aggregate constraint pair-bias embedder.

Shared neural network components live in `ofoldx.modules`; this directory holds the family's
config, typed I/O, checkpoint conversion, and model.
