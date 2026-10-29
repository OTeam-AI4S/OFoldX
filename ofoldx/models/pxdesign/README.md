---
tags:
  - Biology
  - Protein
  - Protein Design
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# PXDesign

Pre-trained model for protein backbone generation and binder evaluation through Preview and Extended workflows.

## Disclaimer

This is an UNOFFICIAL implementation of [PXDesign](https://github.com/bytedance/PXDesign) by ByteDance Research.

The OFFICIAL repository of PXDesign is at [bytedance/PXDesign](https://github.com/bytedance/PXDesign).

**The team releasing PXDesign did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

PXDesign-d is an all-atom diffusion model for de novo and target-conditioned protein backbone generation. The complete
workflow assigns a ProteinMPNN sequence to each backbone, evaluates the complex and unbound binder with AlphaFold 2,
and filters and ranks the candidates.

Extended evaluation additionally uses Protenix base or the structure-conditioned Protenix mini model, following the
released [PXDesignBench](https://github.com/bytedance/PXDesignBench) composition. Please refer to the
[Training Details](#training-details) section for more information on the training process.

### Variants

- **PXDesign backbone**: Independently callable PXDesign-d coordinate generation through `PxDesignBackbonePipeline`.
- **PXDesign Preview**: Designs 100 backbones by default, assigns sequences with ProteinMPNN, and evaluates candidates with AlphaFold 2.
- **PXDesign Extended**: Designs 500 backbones by default and adds Protenix evaluation and screening.

### Model Specification

| Single Size | Pair Size | Token Denoiser Blocks | Atom Encoder Blocks | Atom Decoder Blocks | Num Parameters (M) |
| ----------- | --------- | --------------------- | ------------------- | ------------------- | ------------------ |
| 384         | 128       | 16                    | 4                   | 4                   | 139.028            |

The parameter count is for PXDesign-d.

### Links

- **Code**: [ofoldx.models.pxdesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/pxdesign)
- **Weights**: [oteam/pxdesign](https://huggingface.co/oteam/pxdesign)
- **Developed by**: ByteDance Research
- **Model type**: Target-conditioned all-atom protein backbone diffusion model
- **Original Repository**: [bytedance/PXDesign](https://github.com/bytedance/PXDesign)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### Preview Design

You can use this model directly with a pipeline for 100-residue binder design with Preview evaluation:

```python
import ofoldx
from ofoldx.io import read_structure
from ofoldx.models.pxdesign import from_binder

structure, _ = read_structure("target.cif")
pipeline = ofoldx.pipeline("design", model="oteam/pxdesign")
result = pipeline(structure, placement=from_binder(100, [("target", [(0, 150)])]), seed=7)
```

The result contains the generated candidates and ranked selections. `from_binder` specifies the target instances and
zero-based component spans.

#### Extended Design

You can use this model directly with a pipeline for binder design with Extended evaluation and full-target MSA evidence:

```python
import ofoldx
from ofoldx.io import read_structure
from ofoldx.models.pxdesign import from_binder
from ofoldx.msa import read_msa

structure, _ = read_structure("target.cif")
msa = {"target": read_msa("target.a3m", entity_id="1")}
pipeline = ofoldx.pipeline("design", model="oteam/pxdesign")
result = pipeline(
    structure,
    placement=from_binder(100, [("target", [(0, 150)])]),
    msa=msa,
    mode="extended",
    seed=7,
)
best = result.ranked_candidates[0]
```

Here `target` is the target instance ID and `1` is its entity ID; use the IDs in your input structure.

### Downstream Use

#### Backbone Generation

Use `PxDesignBackbonePipeline` to generate backbones without sequence assignment or evaluation. See the
[pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) and [`PxDesignPipeline`][ofoldx.models.pxdesign.PxDesignPipeline] for component loading and workflow options.

## Training Details

PXDesign-d learns protein backbone generation with an EDM coordinate-denoising objective.

### Training Data

PXDesign-d is trained on [PDB](https://www.rcsb.org/) structures up to 1 May 2021 and predicted structures from
[AlphaFold DB](https://alphafold.ebi.ac.uk/) and [MGnify](https://www.ebi.ac.uk/metagenomics/).

### Training Procedure

PXDesign-d is trained in two stages, first emphasizing monomer structures and then target-conditioned complexes.

## Citation

Please cite the PXDesign work and repository described by the [upstream project](https://github.com/bytedance/PXDesign).

> [!NOTE]
> The artifacts distributed in this repository are part of the OFoldX project.
> If OFoldX supports your research, please cite the OFoldX project as follows:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for any questions or comments on the model implementation or model card.

Please contact the authors of the [PXDesign project](https://github.com/bytedance/PXDesign) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.pxdesign.PxDesignPipeline]: https://ofoldx.valtech.ai/models/pxdesign/#ofoldx.models.pxdesign.PxDesignPipeline
