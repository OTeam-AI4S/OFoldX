---
language: protein
tags:
  - Biology
  - Protein
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# ESMFold2

All-atom biomolecular structure prediction and protein binder design driven by
ESMC representations, with optional protein MSA evidence for supported variants.

## Disclaimer

This is an unofficial OFoldX implementation of
[ESMFold2](https://github.com/Biohub/esm), described in
[Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf).
The upstream authors did not write, review, or endorse this implementation,
or model card.

## Model Details

ESMFold2 predicts all-atom biomolecular structures from ESMC representations and
optional protein MSA evidence. A frozen ESMC language model supplies sequence
representations, a recurrent folding trunk updates token and pair features, and
a diffusion head generates coordinates before distogram and confidence heads
score the result. Experimental variants expose the differentiable inversion
model used for protein binder sequence design.

### Variants

- **[oteam/esmfold2](https://huggingface.co/oteam/esmfold2)**: Full 48-block folding model with optional MSA conditioning.
- **[oteam/esmfold2-fast](https://huggingface.co/oteam/esmfold2-fast)**: Faster 24-block single-sequence folding model.
- **[oteam/esmfold2-experimental](https://huggingface.co/oteam/esmfold2-experimental)**: Complete binder-design artifact containing the fixed 19 Model--Processor pairs used by the released protocol.
- **[oteam/esmfold2-experimental-fast](https://huggingface.co/oteam/esmfold2-experimental-fast)**: Experimental fast model used for binder sequence optimization.
- **ESMFold2 experimental critics**: Experimental-Fast-Cutoff2025,
  Experimental, Experimental-Cutoff2025, and the 15 released
  `Fast-base{300M,600M,6B}-step{250,500,750,1000,1500}k` scaling checkpoints.

### Model Specification

| Variant           | Folding Blocks | MSA | ESMC | Single Size | Pair Size | Atom Size | Token Size | Recurrent Updates | Sampling Steps | Num Parameters (M) | FLOPs (G) |
| ----------------- | -------------- | --- | ---- | ----------- | --------- | --------- | ---------- | ----------------- | -------------- | ------------------ | --------- |
| ESMFold2          | 48             | yes | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6586.242           | 4908.593  |
| ESMFold2 Fast     | 24             | no  | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6548.432           | 3620.103  |
| Experimental      | 48             | yes | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6568.980           | 4335.312  |
| Experimental Fast | 24             | no  | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6531.170           | 3046.822  |

### Links

- **Code**: [ofoldx.models.esmfold2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/esmfold2)
- **Paper**: [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf)
- **Developed by**: Salvatore Candido, Thomas Hayes, Alexander Derry, et al.
- **Model type**: ESMC-conditioned recurrent all-atom folding model with diffusion coordinate generation
- **Original Repository**: [Biohub/esm](https://github.com/Biohub/esm)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Structure Prediction

Load the concrete folding Pipeline:

```python
import ofoldx

pipeline = ofoldx.pipeline("fold", model="oteam/esmfold2")
output = pipeline.fold("MKTAYIAKQRQISFVKSHFSRQ")
```

### Binder Design

The repository marker selects the experimental ESMFold2 design Pipeline. Its
root pair is Experimental-Fast. The other hero roles live at
`fast_cutoff2025/`, `experimental/`, and `experimental_cutoff2025/`; the 15
scaling roles live at
`scaling/base{300m,600m,6b}_step{250k,500k,750k,1000k,1500k}/`. Every slot is
required even when a call skips scaling-critic computation. Each saved Model
owns its complete frozen ESMC trunk and masked-language-model head, so loading
performs no hidden checkpoint download.

```python
import ofoldx

pipeline = ofoldx.pipeline("design", model="oteam/esmfold2-experimental")
output = pipeline.design("target.fasta", binder_length=80)
```

The default call performs 150 inversion steps, retains each candidate's best
actual low-temperature iPTM prediction, evaluates four hero and 15 scaling
critics, applies the released non-antibody pI filter, and stably selects at most
84 designs. `use_scaling_critics=False` skips only the 15 scaling evaluations;
their score component becomes zero. The scaling checkpoints publish no confidence
head, so their evaluations contain structures and distogram proxies; the four hero
evaluations also contain confidence. `output.candidates` preserves both evaluation
groups, while `output.selected` uses each selected candidate's coherent
Experimental-Cutoff2025 structure and confidence values. When `is_antibody` is not
specified, the binder prompt is detected using the released antibody numbering rule.

## Training Details

### Training Data

Biohub reports that ESMFold2 was trained on sequences from the Protein Data
Bank (PDB) and AlphaFold DB (AFDB). The release model cards give a September
2021 data cutoff for ESMFold2 and ESMFold2-Fast. ESMC was pretrained separately
on large protein-sequence corpora; consult the selected ESMC artifact's model
card for its training data. OFoldX does not
redistribute the upstream training corpora.

### Training Procedure

The paper describes diffusion MSE, smooth-LDDT, polymer-ligand bond,
distogram, and confidence losses for the folding system. The released source
and model cards do not publish every target construction, mask, mini-rollout,
batching rule, and reduction needed to reproduce head training exactly.
OFoldX exposes the differentiable single-step denoising path but does not
present it as a complete reproduction of the unpublished training pipeline.

Binder design is inference-time optimization rather than model training.
It performs gradient descent over soft binder-sequence logits under the
released distogram contact objective and ESMC pseudo-perplexity regularizer.
Low-temperature coordinate samples are scored by actual iPTM, and each
candidate retains its best observed sequence and structure.

## Limitations

The released sources do not specify every target, mask, mini-rollout, batching,
and reduction needed for a faithful training objective. OFoldX therefore
provides no built-in Runner objective for ESMFold2; binder design is the
documented inference-time optimization procedure, not model training.

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card. Contact the Biohub
authors for questions about ESMFold2 or the paper.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
