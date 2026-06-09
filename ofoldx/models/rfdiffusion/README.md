# RFdiffusion

RFdiffusion is a denoising-diffusion model that generates novel protein backbones from a
noised structure. A RoseTTAFold trunk denoises the backbone at each timestep, and a native
reverse-diffusion sampler drives one design trajectory from noise down to a finished backbone.

Its distinctive pieces live here: the ported RoseTTAFold network with MSA/pair/template embedders
and an SE(3)-transformer structure module, plus a sampler with the Euclidean and IGSO3 diffusers,
denoiser, contig map, symmetry, and guiding potentials.

Shared neural network components stay in `ofoldx.modules`; this directory holds RFdiffusion's
config, typed I/O, checkpoint conversion, and model.
