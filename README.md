# OFoldX

OFoldX is a research toolkit for biomolecular structure prediction and design. It
follows a small, consistent shape: processors turn structures into typed model inputs,
models consume those inputs, and pipelines package the processor/model boundary for
inference.

OFoldX is pre-release. A family package or prototype is not a support claim;
registration is reserved for a family only after all of its released scientific
functions, checkpoint variants, canonical mappings, and conversion paths are
complete.

Training and evaluation run through one `ofoldx.runners.Runner` (Danling-backed).
It resolves the exact registered architecture, builds the Model and a
config-consistent Processor, and applies the exact built-in objective and batching
behavior owned by the Runner. Training support is not a model-registration
requirement.

## Get Started

Install OFoldX:

```bash
pip install ofoldx
```

For development, install from source:

```bash
git clone https://github.com/OTeam-AI4S/OFoldX
cd OFoldX
pip install -e .
```

Design operations have family-specific scientific controls. Load the concrete
Pipeline selected by the repository and call its exact interface:

```python
import ofoldx
from ofoldx.io import read_structure

structure = read_structure("input.cif")
pipeline = ofoldx.pipeline("design", model="oteam/proteinmpnn")
result = pipeline.design(structure)
candidate = result.candidates[0]
system = candidate.system
for instance in range(system.instance_symbol.numel()):
    start = int(system.instance_component_ptr[instance])
    stop = int(system.instance_component_ptr[instance + 1])
    component_ids = [
        str(system.symbols[int(symbol)])
        for symbol in system.component_symbol[start:stop]
    ]
    print(system.instance_id(instance), component_ids)
```

The same concrete Pipeline exposes that family's complete controls:

```python
result = pipeline.design(structure, num_candidates=4)
```

Use the `Auto*` factories only when you need the raw registered pair:

```python
from ofoldx import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("oteam/model-name")
model = AutoModel.from_pretrained("oteam/model-name")
```

Train a family with a built-in Runner objective from one ordinary configuration.
The runnable one-step examples use a bundled structure and reduced CPU models:

```bash
python examples/runners/proteinmpnn.py
python examples/runners/alphafold3.py
python examples/runners/odesign.py
```

See the complete [ProteinMPNN](examples/runners/proteinmpnn.py),
[AlphaFold 3](examples/runners/alphafold3.py), and
[ODesign](examples/runners/odesign.py) configurations. ProteinMPNN trains
sequence recovery over designable residues. AlphaFold 3 uses the EDM-weighted
coordinate-denoising loss over resolved atoms. ODesign uses its released five-term
aggregate, normalized once per example.

The AlphaFold 3 objective does not reproduce the unpublished upstream corpus,
auxiliary confidence and distogram losses, optimizer, or curriculum. Families
without a faithful built-in objective remain available through their Model and
Pipeline APIs; the Runner does not fabricate a loss for them.

Per-sample MSA evidence is explicit in a JSON-lines dataset manifest:

```json
{"structure": "structures/1abc.cif", "msa": {"path": "msa/1abc_A.a3m", "entity_id": "A"}}
{"structure": "structures/2xyz.cif", "msa_collection": "msa/2xyz_pairing.json"}
{"structure": "contexts/ligand_site.cif", "target": {"path": "targets/ligand_complex.cif", "instance_ids": ["L"]}}
```

Paths are relative to the manifest. `msa_collection` is encoded with
`ofoldx.io.encode_msa_collection`. `target` supplies a distinct per-sample truth
structure and the exact molecular instances supervised as generated atoms. For a
sequence-indexed corpus, configure the
Dataset split directly:

```yaml
data:
  train:
    source: structures/
    msa_index: sequence_to_directory.json
    msa_root: msa/
```

The Dataset reads each structure, acquires the indexed alignments, and constructs
the canonical `MsaCollection` before calling the Processor. It does not infer
sidecars from structure filenames; `read_msa()` remains the single-file parser.

Weighted training views use a separate JSON-lines index whose rows identify source
records by structure path rather than by mutable list position:

```json
{"structure": "structures/1abc.cif", "sample_type": "chain", "instance_ids": ["A"], "weight": 2.0}
{"structure": "structures/2xyz.cif", "sample_type": "interface", "instance_ids": ["A", "B"], "weight": 1.0}
```

Relative paths are resolved from the sampling index. Configure it with
`data.train.sampling.index`; optional `epoch_size` and `replacement` select the global
draw count and weighted-replacement policy. Chain IDs restrict the eligible crop
anchors. An interface row requires a resolved contact between exactly the two named
instances; it never substitutes another interface. Chain and interface rows require
a cropper passed to the Runner.

Write one registered Model/Processor pair as a standard repository with the sole
public artifact writer:

```python
from ofoldx.artifacts import save_model

save_model("./model-repository", model=model, processor=processor)
```

The writer emits `model.json`, `processor.json`, and standard safetensors weights.
A registered concrete Pipeline additionally writes `pipeline.json` through its
own `save()` method and is reconstructed by `ofoldx.pipeline(...)`. Model-card
authorship and Hub publishing remain separate.

Canonical PDB/mmCIF structure I/O and A3M/aligned-FASTA MSA-evidence I/O live
in `ofoldx.io`; standalone biological-sequence formats remain in MultiMolecule.

`AutoModel.from_pretrained()` and `AutoProcessor.from_pretrained()` accept either
a local model directory or a Hugging Face Hub repository ID.
`AutoModel.from_config()` constructs a fresh registered model. Pipeline repositories
persist only the concrete Pipeline ID; the class owns how its model/processor pair
or fixed composition is loaded.

## Explore

| Package                                | Purpose                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| [`ofoldx.models`](ofoldx/models)       | Model registration; concrete family packages are imported explicitly.                 |
| `ofoldx.artifacts`                     | The one public writer for a validated standard Model/Processor repository.            |
| `ofoldx.hub`                           | Hugging Face Hub repository resolution, publication, and loading.                     |
| [`ofoldx.pipelines`](ofoldx/pipelines) | Concrete family Pipeline registration and shared paired/composite persistence.        |
| [`ofoldx.modules`](ofoldx/modules)     | Shared neural-network building blocks: attention, trunk, heads, denoising, diffusion. |
| [`ofoldx.io`](ofoldx/io)               | PDB/mmCIF structure and A3M/aligned-FASTA MSA-evidence I/O.                           |
| [`ofoldx.data`](ofoldx/data)           | Canonical molecular, evidence, placement, and scientific domain values.               |
| `ofoldx.runners`                       | Config-only Danling Runner for unified training and evaluation.                       |

The low-level values exported by `ofoldx.data` are part of the domain API: atom, bond,
component, attachment, entity, instance, and site values are what make an in-memory
`MolecularSystem` and `Structure` referentially closed.

## Status

OFoldX is pre-release research software under the AGPL-3.0 license. V1 is not complete
or released until all normative verbs, inputs, Results, and complete PDB/mmCIF I/O
pass their automated release gates.
