# LigandMPNN

LigandMPNN is a message-passing inverse-folding model that designs and scores
amino-acid sequences for a protein-ligand complex, conditioning on backbone
geometry and nearby ligand/context atoms.

Its `LigandMPNNDesignGenerator` runs a k-nearest-neighbor graph encoder and an
autoregressive decoder that scores a given sequence, scores one masked residue
at a time, or samples sequences. The family also ships a `Packer` that predicts
side-chain chi angles as a Von Mises mixture and denoises them into atom
coordinates.

Shared neural network components stay in `ofoldx.modules`; this directory
holds the family config, typed I/O, official checkpoint conversion, and model.
