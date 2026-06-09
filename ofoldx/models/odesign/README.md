# ODesign

ODesign is an all-atom structure-prediction model for protein design: a recycling
trunk (MSA, template, constraint-distogram, and Pairformer stacks) feeds an EDM
coordinate diffusion sampler, with distogram and confidence heads.

It uses the ODesign token schema and adds its own input embedder, a token
bond-type reconstruction head, an optional constraint-distogram embedder for
conditioning generation on distances, and a design-generator wrapper.

Shared neural network components live in `ofoldx.modules`; this directory holds
the family's config, typed I/O, structure processor, and model.
