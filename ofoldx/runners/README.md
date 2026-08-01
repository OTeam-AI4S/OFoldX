# Runners

Train a supported model family with a configured objective and export its model and processor for inference.
The one-step examples use a bundled structure and initialize models locally.

## Usage

Run these commands from the repository root:

```bash
python examples/runners/proteinmpnn.py
python examples/runners/alphafold3.py
python examples/runners/odesign.py
```

See the complete [ProteinMPNN](../../examples/runners/proteinmpnn.py),
[AlphaFold 3](../../examples/runners/alphafold3.py), and
[ODesign](../../examples/runners/odesign.py) configurations. ProteinMPNN trains
sequence recovery over designable residues. AlphaFold 3, all Protenix variants,
Boltz-1, Boltz-2, `boltzgen`, and `boltzgen_miniformer` use the
shared, rigidly aligned EDM coordinate error over resolved atoms. These generic
objectives do not reproduce the families' complete losses, training data policy,
optimizers, or curricula. Protenix's Runner disables auxiliary heads and omits
its molecule weights, `1/3` coordinate normalization, and independent mini-rollout
supervision; see its [training details](../models/protenix/README.md#training-details).
BoltzGen separately exposes its [complete coordinate loss](../models/boltzgen/README.md#coordinate-loss)
through [`diffusion_loss`](../models/boltzgen/objective.py).
ODesign uses its released five-term aggregate, normalized once per example.
Families without a built-in Runner objective remain available through their
model and pipeline APIs.

The ProteinMPNN example reloads its exported model to score the observed
sequence. Scoring needs the supplied backbone and residue classes; declaring a
new chemical design additionally needs sufficient boundary chemistry.

## Data Preparation

Processors expose three data operations:

- `prepare_input(structure, ...)` returns features and their preparation context.
- `prepare_target(truth, prepared=input, ...)` aligns supervision to that same layout.
- `prepare(structure, target=truth, ...)` combines the two and returns an ordinary `(input, target)` pair.

Without an explicit target, the second result is `None`. Training and labelled validation use the same operations;
no preparation method infers a train/eval mode. Supported target types depend on the model family. The Runner's
objective declares its target source: coordinate and
sequence objectives use an independent manifest truth or explicitly request the source as truth, while AlphaFold 2
masked-MSA supervision is constructed from the input alignment by its objective.

Input features and supervision remain separate. Objectives construct teacher history or noisy model inputs
without changing the prepared features. Autoregressive objectives require known truth throughout the generated
sequence; missing truth or a known class outside the model's output vocabulary raises an error.

The automatic cropper is added only to training splits; other scientific options and random generators remain
explicit. Validation evaluates the same objective without a parameter update. Use a pipeline for full structure
prediction or generation.

## MSA and Target Evidence

Per-sample MSA evidence is explicit in a JSON-lines dataset manifest:

```json
{"structure": "structures/1abc.safetensors", "msa": {"path": "msa/1abc_A.a3m", "entity_id": "A"}}
{"structure": "structures/input.safetensors", "target": {"path": "targets/truth.safetensors"}}
{"structure": "contexts/ligand_site.safetensors", "target": {"path": "targets/ligand_complex.safetensors", "instance_ids": ["L"]}}
```

Paths are relative to the manifest. A record's `msa` (A3M or aligned FASTA) is the
pools of the named entity, bound to every instance of that entity. For supervised
training and validation, `target.path` supplies independent truth matched by instance name,
component ordinal and atom name; missing atoms remain unobserved. Input chemistry
and conditioning come from `structure`. ODesign provisional ligand training
(`odesign_ligand`) additionally requires `target.instance_ids`, the generated
instances in their supervision order. Its atom count comes from those instances;
context and generated truth must share a Cartesian frame. Fixed context coordinates
come from `structure`. Structure and target paths are native artifacts (`.safetensors`, written
by `save_structure`); the Dataset refuses other formats. Read mmCIF or PDB once beforehand
with `read_structure` and save the structures. For a sequence-indexed corpus, configure
the Dataset split directly:

```yaml
data:
  train:
    source: structures/
    msa_index: sequence_to_directory.json
    msa_root: msa/
```

The index maps each instance's query sequence to its directory under `msa_root`, in the
indexed layout (`pairing.a3m` and `non_pairing.a3m` per directory; see
`ofoldx.msa.IndexedMsaLoader`). The Dataset passes the pools by instance to the
Processor, which pairs them after its crop. It does not infer sidecars from structure
filenames.

## Weighted Training Views

Weighted training views use a separate JSON-lines index whose rows identify source
records by structure path rather than by mutable list position:

```json
{"structure": "structures/1abc.safetensors", "sample_type": "chain", "instance_ids": ["A"], "weight": 2.0}
{"structure": "structures/2xyz.safetensors", "sample_type": "interface", "instance_ids": ["A", "B"], "weight": 1.0}
```

Relative paths are resolved from the sampling index. Configure it with
`data.train.sampling.index`; optional `num_epoch_samples` and `replacement` select the global
draw count and weighted-replacement policy. Chain IDs restrict the eligible crop
anchors. An interface row requires a resolved contact between exactly the two named
instances; it never substitutes another interface. Chain and interface rows require
a cropper passed to the Runner.

## Export for Inference

Runner checkpoints retain training state. Export the trained Model and its
processor as a native repository before loading a prediction pipeline; see the
[`Runner`][ofoldx.runners.runner.Runner] export example.

[ofoldx.runners.runner.Runner]: https://ofoldx.valtech.ai/runners/runner/#ofoldx.runners.runner.Runner
