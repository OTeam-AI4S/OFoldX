# Examples

Examples of inference, training, and model export.

## Score a Protein Sequence

Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact before reading a structure.

Provide a backbone file to the [ProteinMPNN example](pipelines/proteinmpnn.py):

```bash
python examples/pipelines/proteinmpnn.py --structure backbone.cif --output scores.json
```

The script uses `oteam/proteinmpnn` and writes `designed_sequence_nll` and
`global_sequence_nll` to `scores.json`. Each record contains the score's value,
unit, ranking direction and molecular scope. For these NLL scores, the unit is
`nat/residue`, lower values are preferred, and the scope names the scored
residues by instance and component ordinal.

Omitting `--structure` uses the bundled [tripeptide](data/tripeptide.pdb).
`--device` defaults to `cpu`; use `--device cuda` for a CUDA model.
`--seed` initializes decoding-order sampling.

## Use a Local Checkpoint

Pass `--model` to select another Hub repository or a native local export. To
convert a downloaded ProteinMPNN checkpoint and score it:

```bash
python -m ofoldx.models.proteinmpnn.convert ./checkpoints/v_48_020.pt ./models/proteinmpnn
python examples/pipelines/proteinmpnn.py --model ./models/proteinmpnn --output scores.json
```

The [model guide](../ofoldx/models/README.md) lists the other families and their
converters.

## Train and Export a Model

These examples initialize a model locally and train for one step on the bundled
structure:

```bash
python examples/runners/proteinmpnn.py
python examples/runners/alphafold3.py
python examples/runners/odesign.py
```

The ProteinMPNN example exports a native repository, reloads it and scores the
input sequence. Use that repository with the standalone scoring script:

```bash
python examples/pipelines/proteinmpnn.py --model ./runs/proteinmpnn/native --output scores.json
```

The one-step runs demonstrate training and export. For pretrained inference,
select a checkpoint from the model guides. The [training guide](../ofoldx/runners/README.md)
describes the available objectives and dataset configuration.
