---
tags:
  - Biology
  - Protein
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# ESMFold2

Pre-trained models for all-atom structure prediction and gradient-based protein binder design.

## Disclaimer

This is an UNOFFICIAL implementation of the [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf) by Salvatore Candido, et al.

The OFFICIAL repository of ESMFold2 is at [Biohub/esm](https://github.com/Biohub/esm).

**The team releasing ESMFold2 did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

ESMFold2 uses a frozen ESMC language model to supply sequence representations to a recurrent folding trunk.
A diffusion head generates atomic coordinates, and distogram and confidence heads score the prediction. Supported
checkpoints also accept protein MSA evidence. Experimental variants expose differentiable inversion for protein binder
sequence design.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/esmfold2](https://huggingface.co/oteam/esmfold2)**: The full 48-block folding model with optional MSA conditioning.
- **[oteam/esmfold2-fast](https://huggingface.co/oteam/esmfold2-fast)**: The 24-block single-sequence folding model.
- **[oteam/esmfold2-experimental](https://huggingface.co/oteam/esmfold2-experimental)**: The complete binder-design workflow containing all 19 required model-processor pairs.
- **[oteam/esmfold2-experimental-fast](https://huggingface.co/oteam/esmfold2-experimental-fast)**: The experimental fast model used for binder sequence optimization.

The design workflow includes Experimental-Fast, Experimental-Fast-Cutoff2025, Experimental, Experimental-Cutoff2025,
and 15 `Fast-base{300M,600M,6B}-step{250,500,750,1000,1500}k` scaling critics.

### Model Specification

| Variant           | Folding Blocks | MSA | ESMC | Single Size | Pair Size | Atom Size | Token Size | Recurrent Updates | Sampling Steps | Num Parameters (M) |
| ----------------- | -------------- | --- | ---- | ----------- | --------- | --------- | ---------- | ----------------- | -------------- | ------------------ |
| ESMFold2          | 48             | yes | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6579.742           |
| ESMFold2 Fast     | 24             | no  | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6533.723           |
| Experimental      | 48             | yes | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6577.328           |
| Experimental Fast | 24             | no  | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6530.863           |

Parameter counts include the frozen ESMC encoder.

### Links

- **Code**: [ofoldx.models.esmfold2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/esmfold2)
- **Paper**: [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf)
- **Developed by**: Salvatore Candido, Thomas Hayes, Alexander Derry, Roshan Rao, Zeming Lin, Robert Verkuil, Bryan Wu,
  Jin Sub Lee, Elise S. Bruguera, Jehan A. Keval, Mykhailo Kopylov, John E. Pak, Wesley Wu, Neil Thomas, Samson
  Mataraso, Alvin Hsu, Ashton C. Trotman-Grant, Kilian Fatras, Allan dos Santos Costa, Rohil Badkundri, Halil Akın,
  Deniz Oktay, Jonathan Deaton, Elizabeth Montabana, Hrishita Sitwala, Yue Yu, Marius Wiggert, Dylan Alexander Carlin,
  Anthony W. Goering, Tomasz Blazejewski, McCullen Sandora, Michael Hla, Tina Z. Jia, Leon H. Kloker, Nicholas J.
  Sofroniew, Masatoshi Uehara, Jassi Pannu, Sharrol Bachas, Daniel S. Liu, Tom Sercu, and Alexander Rives
- **Model type**: ESMC-conditioned recurrent all-atom folding model with diffusion coordinate generation
- **Original Repository**: [Biohub/esm](https://github.com/Biohub/esm)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Structure Prediction

You can use this model directly with a pipeline for all-atom structure prediction:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("input.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/esmfold2")
result = pipeline(structure, seed=7)
```

The result contains predicted structures and confidence values. The standard and Fast models share the frozen
[MultiMolecule ESMC-6B checkpoint](https://huggingface.co/multimolecule/esmc-6b); its exact revision is recorded in the
model configuration and downloaded through the Hugging Face cache. Protein sequences are also accepted directly.

#### Binder Design

You can use this model directly with a pipeline for experimental binder sequence optimization:

```python
import ofoldx

pipeline = ofoldx.pipeline("design", model="oteam/esmfold2-experimental")
result = pipeline("MKTAYIAKQRQISFVKSHFSRQ", num_binder_residues=80, seed=7)
```

The workflow optimizes binder sequences with gradients and ranks candidates using folding predictions. All components
of the experimental repository are required, including when scaling critics are disabled. Antibody designs are detected
automatically unless `is_antibody` is specified.
Antibody detection and CDR scoring use ANARCI and require HMMER's `hmmscan` executable on `PATH`.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution and composition.

## Training Details

ESMFold2 combines pretrained folding models with learned confidence and scaling components for structure prediction.

### Training Data

Biohub reports training on Protein Data Bank (PDB) and AlphaFold DB (AFDB) data, with a September 2021 cutoff for
ESMFold2 and ESMFold2-Fast. ESMC was pretrained separately; consult its selected model card for the sequence corpus.
OFoldX does not redistribute the upstream training corpora.

### Training Procedure

The paper describes diffusion MSE, smooth lDDT, polymer-ligand bond, distogram, and confidence objectives. The released
source and model cards do not publish every target-construction, masking, mini-rollout, batching, and reduction rule.

OFoldX exposes a differentiable single-step denoising path. It does not provide a complete reproduction of the upstream
head-training program. Binder sequence optimization is an inference workflow described under [Usage](#usage).

## Citation

```bibtex
@misc{candido2026language,
  title = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and others},
  year = {2026},
  url = {https://biohub.ai/papers/esm_protein.pdf},
  note = {Preprint}
}
```

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

Please contact the authors of the [ESMFold2 paper](https://biohub.ai/papers/esm_protein.pdf) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
