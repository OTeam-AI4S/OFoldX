# AlphaFold 3 Model

This directory owns the AlphaFold 3-shaped model.

- `config.py` defines the architecture-specific `chanfig.Config` classes.
- `model.py` assembles shared `ofoldx.modules` building blocks into `AlphaFold3Model`.

Shared neural network components stay in `ofoldx.modules`; this directory only contains AlphaFold 3
configuration and composition.

## Checkpoint conversion

AlphaFold 3 uses a source-aware converter:

```bash
python -m ofoldx.models.alphafold3.convert --source deepmind ~/models official-af3 --inspect-only
```

`--source deepmind` reads the official local AlphaFold 3 parameter directory for
inspection/loading, without bundling or downloading the weights. Official DeepMind
parameter conversion is intentionally kept behind that source hook until the
Haiku record keys are mapped to the native OFoldX `AlphaFold3Model` state dict.

Protenix checkpoints use the separate `ofoldx.models.protenix.convert` entry point so converted
artifacts keep the `protenix` architecture identity.
