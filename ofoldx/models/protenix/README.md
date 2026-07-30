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
proteins, nucleic acids, small molecules, and their complexes. OFoldX exposes
all ten released checkpoint variants through four exact
Model/Processor/Pipeline pairs.

## Disclaimer

This is an unofficial OFoldX implementation of
[Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
by the ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, et al. The
upstream authors did not write, review, or endorse this implementation or model card.

The official repository is
[bytedance/Protenix](https://github.com/bytedance/Protenix).

## Model Details

Protenix is an AlphaFold 3-style all-atom structure predictor for proteins,
nucleic acids, small molecules, ions, and their complexes. It combines atom and
token embeddings with an MSA-conditioned recycled Pairformer trunk, EDM
coordinate diffusion, and distogram and confidence heads. The constraint variant
adds learned pair biases for atom contacts, token contacts, pockets, and
substructures.

### Released Variants

Conversion requires one of these exact identifiers; OFoldX does not infer the
variant from a filename or tensor shape:

- `protenix_base_default_v0.5.0`
- `protenix_base_constraint_v0.5.0`
- `protenix_mini_default_v0.5.0`
- `protenix_mini_tmpl_v0.5.0`
- `protenix_tiny_default_v0.5.0`
- `protenix_mini_esm_v0.5.0`
- `protenix_mini_ism_v0.5.0`
- `protenix_base_default_v1.0.0`
- `protenix_base_20250630_v1.0.0`
- `protenix-v2`

The variants map to ordinary registered pairs:

| Pair                                                 | Model, Processor, Pipeline                                                                                     | Released variants                                                     |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Default (`protenix`)                                 | `ProtenixModel`, `ProtenixProcessor`, `ProtenixPipeline`                                                       | base default v0.5, mini default, tiny default, mini ESM, and mini ISM |
| Template (`protenix_template`)                       | `ProtenixModel`, `ProtenixTemplateProcessor`, `ProtenixTemplatePipeline`                                       | both base v1 releases and `protenix-v2`                               |
| Structure condition (`protenix_structure_condition`) | `ProtenixStructureConditionModel`, `ProtenixStructureConditionProcessor`, `ProtenixStructureConditionPipeline` | mini template v0.5                                                    |
| Constraint (`protenix_constraint`)                   | `ProtenixConstraintModel`, `ProtenixConstraintProcessor`, `ProtenixConstraintPipeline`                         | base constraint v0.5                                                  |

The two v1 releases share one graph and differ in weights. Mini ESM and mini ISM
also share one graph and package their selected ESM2 companion weights inside the
model artifact. The template pair owns the released template and RNA-MSA input
paths. The constraint pair exposes atom-contact, token-contact, and pocket
controls. Its fourth learned branch accepts a `Structure` through
`substructure=` and digitizes resolved token-center distances into the released
four-channel representation.

The mini template checkpoint is not an ordinary template model. Its distinct
pair accepts `structure_condition=` as an in-memory `Structure`, aligns resolved
representative-atom coordinates to the model token axis, and applies the
checkpoint's noisy-structure pair update on every recycling pass.
Direct `ProtenixNoisyStructureEmbedder` callers may enable its released
`add_noise` distance perturbation and supply a `torch.Generator` for that draw.

### Links

- **Code**: [ofoldx.models.protenix](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/protenix)
- **Paper**: [Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
- **Developed by**: ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, et al.
- **Model type**: Pairformer-based all-atom biomolecular structure predictor with optional learned constraints
- **Original Repository**: [bytedance/Protenix](https://github.com/bytedance/Protenix)

## Conversion

Conversion requires the exact released variant rather than inferring it from
the checkpoint filename:

```bash
python -m ofoldx.models.protenix.convert \
  released-protenix.pt /path/to/converted-protenix \
  --variant protenix_base_default_v0.5.0
```

The mini ESM and mini ISM variants additionally require their released ESM2
companion through `--esm-checkpoint`.

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Structure Prediction

Load any converted repository through its concrete folding Pipeline:

```python
import ofoldx

pipeline = ofoldx.pipeline("fold", model="/path/to/converted-protenix")
output = pipeline.fold("input.cif")
```

Template variants additionally accept `templates=`. Constraint repositories
route to `ProtenixConstraintPipeline`, whose concrete `fold` method accepts
`atom_contacts`, `token_contacts`, and `pocket`. Direct
`ProtenixConstraintProcessor` use additionally accepts the released
`substructure` coordinate condition. Every concrete Protenix Pipeline accepts
per-call `recycling_steps`, `num_sampling_steps`, `churn_rate`, and
`step_scale`; `None` uses the converted checkpoint configuration.

Training-free chemical guidance is a sampling-time operation. Pass the released
enabled configuration explicitly:

```python
import ofoldx

from ofoldx.models.protenix import ProtenixTfgGuidance

pipeline = ofoldx.pipeline(
    "fold",
    model="/path/to/converted-protenix",
)
output = pipeline.fold(
    "input.cif",
    tfg_guidance=ProtenixTfgGuidance(),
)
```

Leaving `tfg_guidance` as `None` disables guidance. The guidance value exposes
the eight named chemical terms, schedules, Monte Carlo smoothing, and iteration
counts; setting one term to `None` removes that term. TFG is family-specific and
therefore remains an argument of the concrete Pipeline.

## Training Details

`ProtenixProcessor.process_training(structure, ...)` returns a
`ProtenixInput` and separate `ProtenixStructureTarget`. Calling the model with
those values runs one differentiable stochastic denoising pass and returns
denoised coordinates, augmented target coordinates, sampled noise levels, the
distogram, and all four confidence logits. `model.denoise(...)` retains the
explicit noisy-coordinate operation, while `model.sample(...)` exposes raw
reverse-diffusion output and `model.fold(...)` decodes it.

OFoldX reproduces the complete released Model graph, not Protenix's complete
four-stage training program. Dataset policy, objectives, stage weights,
condition-dropout curriculum, optimizer, and training runner remain the
responsibility of the training application.

## Limitations

The built-in Runner objective covers the EDM coordinate-denoising term, not
Protenix's complete four-stage training program or its auxiliary-head and
condition-dropout schedules. The selected converted checkpoint determines
which of the four concrete input contracts is available.

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

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
