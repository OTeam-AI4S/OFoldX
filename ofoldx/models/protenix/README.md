---
language:
  - protein
  - rna
  - dna
tags:
  - Biology
  - Protein
  - RNA
  - DNA
  - Structure Prediction
license: agpl-3.0
base_model: bytedance/Protenix
library_name: ofoldx
pipeline_tag: other
---

# Protenix

Protenix is OFoldX's AlphaFold 3-shaped all-atom structure-prediction family for
proteins, nucleic acids, small molecules, and their complexes. OFoldX implements
the released v0.5 default and constraint checkpoint graphs as distinct
Model/Processor/Pipeline pairs.

## Disclaimer

This is an unofficial OFoldX conversion of
[Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
by the ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, et al. The
upstream authors did not write, review, or endorse this conversion or model card.

The official repository is
[bytedance/Protenix](https://github.com/bytedance/Protenix). OFoldX converts
released weights into its native parameter layout; it does not train, relicense,
or claim ownership of the upstream models.

## Model Details

`ProtenixModel` embeds token and atom features, refines single, pair, and MSA
representations through a recycled Pairformer trunk, denoises atom coordinates
with an EDM sampler, and predicts distogram and confidence outputs. The
constraint graph adds a four-branch pair-bias embedder before recycling.

`ProtenixPipeline` and `ProtenixConstraintPipeline` own the concrete `fold`
operation for their respective pairs. Both return a canonical `FoldResult`.
Only `ProtenixConstraintProcessor` accepts atom contacts, token contacts, and a
pocket; the default `ProtenixProcessor` omits those keywords, so passing them to
the default pair raises `TypeError` at the call boundary.

### Variants

| Architecture                 | Released checkpoint                  | Inspected upstream source                          | Constraints                                                                       | Concrete Pipeline            |
| ---------------------------- | ------------------------------------ | -------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------- |
| `ofoldx.protenix`            | `protenix_base_default_v0.5.0.pt`    | v0.5.0, `26816e9fcb2cd1ff16dc9eaa8feb913f853b9528` | none                                                                              | `ProtenixPipeline`           |
| `ofoldx.protenix_constraint` | `protenix_base_constraint_v0.5.0.pt` | v0.5.3, `a0f95f62430093b46c677f04e3d292ccc5d28d64` | atom contact, token contact, pocket; substructure branch present with a zero grid | `ProtenixConstraintPipeline` |

The constraint checkpoint contains all four constraint embedders as one graph;
the branches are not independently selectable. The default checkpoint contains
none of them.

The upstream model-type inventory also names
`protenix_mini_default_v0.5.0`, `protenix_tiny_default_v0.5.0`,
`protenix_mini_esm_v0.5.0`, `protenix_mini_ism_v0.5.0`,
`protenix_base_default_v1.0.0`, `protenix_base_20250630_v1.0.0`, and
`protenix-v2`. Current OFoldX code does not define or convert those graphs, so
they are not aliases or supported checkpoint variants of the two pairs above.

### Model Specification

| Property                               | Native v0.5 default             |
| -------------------------------------- | ------------------------------- |
| Token / single / pair size             | 449 / 384 / 128                 |
| Trunk Pairformer Blocks                | 48 (`num_pairformer_blocks`)    |
| Denoiser token encoder Blocks          | 24 (`num_token_encoder_blocks`) |
| Denoiser atom encoder / decoder Blocks | 3 / 3                           |
| Confidence Pairformer Blocks           | 4 (`num_blocks`)                |
| Template Blocks                        | 0 (disabled)                    |
| Recycling passes                       | 10 (`recycling_steps`)          |
| Reverse-diffusion steps                | 200 (`num_sampling_steps`)      |
| Default fold samples / sample chunk    | 5 / 5                           |
| Distogram / PAE / PDE bins             | 64 / 64 / 64                    |
| pLDDT bins                             | 50                              |

The names above follow the native OFoldX graph: repeated units are Blocks,
stored under `blocks` and counted by fields ending in `num_blocks`; channel MLP
units are Transitions. Released source names appear only in conversion mappings.

Both v0.5 graphs use an Euler EDM sampler with `sigma_data=16`, `rho=7`, noise
levels from `160` to `4e-4`, `churn_rate=0.8`,
`min_churn_noise_level=1`, `noise_scale=1.003`, and `step_scale=1.5`. Reference
coordinates receive random centering/rotation/translation augmentation. The
default graph samples up to 2048 MSA rows per recycle; the constraint graph draws
a row count and caps it at 16,384. A caller-owned generator controls both
Processor and Model randomness.

The constraint records are family-specific:

- `ProtenixAtomContact(atom_a, atom_b, max_distance)` encodes one upper bound
  between exact atoms. There is no public minimum-distance channel.
- `ProtenixTokenContact(endpoint_a, endpoint_b, max_distance)` encodes `[0, max]`
  between residue/token endpoints. Two exact-atom endpoints must use
  `ProtenixAtomContact`.
- `ProtenixPocket(binder, contact_residues, max_distance)` encodes one directed
  binder-to-pocket declaration. At most one pocket is accepted per input.

The constraint Processor always emits a zero substructure grid because current
OFoldX exposes no substructure condition. Both v0.5 graphs reject template
evidence because their native configuration has zero template Blocks.

Each `FoldPrediction` contains atom-axis pLDDT on the 0-100 scale, resolved
probability, token-pair PAE/PDE in angstroms, contact probabilities, and chain or
chain-pair pLDDT, pTM, ipTM, and gPDE summaries. Scalar scores include
`mean_plddt`, `gpde`, `ptm`, `iptm`, `has_clash`, `disorder`, and
`ranking_score`. Current decoding sets `disorder=0` and uses

```text
ranking_score = 0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100 * has_clash
```

### Links

- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/protenix)
- [Official source](https://github.com/bytedance/Protenix)
- [Protenix paper](https://doi.org/10.1101/2025.01.08.631967)
- [Upstream model repository](https://huggingface.co/bytedance/Protenix)
- Developed by the ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang,
  et al.

## Usage

```python
from ofoldx import AutoModel, AutoProcessor, fold, pipeline
from ofoldx.data import AtomSite
from ofoldx.models.protenix import ProtenixAtomContact, ProtenixPocket

default_repository = "./protenix"
constraint_repository = "./protenix-constraint"

default_result = fold(
    "ACDEFGHIK",
    model=default_repository,
    device="cuda",
    num_samples=5,
    seed=7,
)

system = ...
constraint_folder = pipeline(
    "fold",
    model=constraint_repository,
    device="cuda",
)
constraint_result = constraint_folder.fold(
    system,
    atom_contacts=(
        ProtenixAtomContact(
            atom_a=AtomSite("A", 0, "CA"),
            atom_b=AtomSite("L", 0, "C1"),
            max_distance=8.0,
        ),
    ),
    pocket=ProtenixPocket(
        binder="L",
        contact_residues=(("A", 10), ("A", 11)),
        max_distance=10.0,
    ),
    num_samples=5,
    seed=7,
)

model = AutoModel(default_repository, device="cuda")
processor = AutoProcessor(default_repository, device="cuda")
direct_result = model.fold(processor("ACDEFGHIK"))
```

Inputs may also be FASTA or structure paths, `PolymerSequence` values, sequence
collections, `MolecularSystem`, or `Structure`. Pipeline calls accept `seed` or a
caller-owned `torch.Generator`; direct Model calls accept the generator.

## Checkpoints and Conversion

Convert each released graph into its own repository:

```bash
python -m ofoldx.models.protenix.convert \
  protenix_base_default_v0.5.0.pt \
  ./protenix

python -m ofoldx.models.protenix.convert \
  protenix_base_constraint_v0.5.0.pt \
  ./protenix-constraint
```

The converter detects the constraint graph from its tensors, constructs the
matching Model, translates every released source tensor into native Block and
Transition names, requires every source tensor to participate, and loads with
`strict=True`. It pairs the Model with the matching Processor and writes
`model.json`, `processor.json`, and safetensors weights. The constraint repository
also records `ProtenixConstraintPipeline` so its extra fold keywords survive
repository loading. `--dtype` may cast floating tensors, and `--repo-id`
optionally uploads the converted repository.

## Training Details

### Training Data

The Protenix v0.5.0 release reports a 2021-09-30 training-data cutoff. Its data
pipeline covers released wwPDB/mmCIF bioassemblies, MSAs, CCD cache data, and
RCSB sequence clusters. OFoldX does not redistribute the training corpus.

### Training Procedure

Protenix trains the trunk, distogram, confidence, and coordinate-denoising graph
jointly. The diffusion objective denoises noised ground-truth coordinates, while
the auxiliary heads supervise geometry and confidence. The default and
constraint artifacts are full checkpoints containing their shared trunk weights;
the constraint checkpoint additionally trains all four pair-bias embedders.

The complete upstream optimizer, schedule, data mixture, and loss reductions are
not reproduced by the converter. OFoldX translates and strict-loads released
weights without retraining them.

## Limitations

- Current OFoldX support is limited to the two named v0.5 base graphs. The seven
  other upstream model-type entries listed above are not implemented or
  parity-qualified.
- Both implemented graphs disable templates. Supplying a non-empty `templates`
  argument raises `ValueError`.
- The default pair cannot consume constraints. The constraint pair exposes only
  upper-bound atom/token contacts and one pocket; its substructure branch remains
  a zero grid.
- `ProtenixAtomContact` intentionally maps `max_distance` to the checkpoint's
  trained atom-contact channel and exposes no legacy channel-order mode. This
  corrects the documented v0.5.3 training/inference channel swap.
- Folding is stochastic because MSA subsets, reference augmentation, and
  diffusion sampling advance the caller's generator.
- TM and chain-gPDE empty denominators use the current dtype's
  `torch.finfo(dtype).tiny`; this is a Protenix-local numerical behavior.
- Recorded GPU comparisons for matched v0.5 inputs found per-operation
  differences from 0 to `1e-6`. That evidence is specific to the two inspected
  revisions, checkpoints, inputs, and precision; it does not establish parity
  for every padded input, constraint combination, or unimplemented variant.

## Citation

```bibtex
@article{protenix2025advancing,
  author = {ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and others},
  title = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  year = {2025},
  doi = {10.1101/2025.01.08.631967},
  journal = {bioRxiv}
}
```

If OFoldX supports your research, also cite the OFoldX software:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card. Questions about the
upstream model should be directed to the authors of the
[Protenix paper](https://doi.org/10.1101/2025.01.08.631967).

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and model parameters:** Apache-2.0 under the
  [Protenix license](https://github.com/bytedance/Protenix/blob/main/LICENSE),
  which permits academic research and commercial use subject to its terms.
- `protenix_base_default_v0.5.0.pt` and
  `protenix_base_constraint_v0.5.0.pt` may be redistributed with the upstream
  Apache-2.0 notice preserved. Converted weights are not relicensed by OFoldX.

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
