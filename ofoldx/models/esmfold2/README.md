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
score the result. Experimental variants reuse the folding model as a
differentiable critic for protein binder sequence design.

### Variants

- **[oteam/esmfold2](https://huggingface.co/oteam/esmfold2)**: Full 48-block folding model with optional MSA conditioning.
- **[oteam/esmfold2-fast](https://huggingface.co/oteam/esmfold2-fast)**: Faster 24-block single-sequence folding model.
- **[oteam/esmfold2-experimental](https://huggingface.co/oteam/esmfold2-experimental)**: Experimental MSA-conditioned model used for binder design and refolding.
- **[oteam/esmfold2-experimental-fast](https://huggingface.co/oteam/esmfold2-experimental-fast)**: Experimental fast model used for binder sequence optimization.

### Model Specification

| Variant           | Folding Blocks | MSA | ESMC | Single Size | Pair Size | Atom Size | Token Size | Recurrent Updates | Sampling Steps | Num Parameters (M) | FLOPs (G) |
| ----------------- | -------------- | --- | ---- | ----------- | --------- | --------- | ---------- | ----------------- | -------------- | ------------------ | --------- |
| ESMFold2          | 48             | yes | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6586.242           | 4908.593  |
| ESMFold2 Fast     | 24             | no  | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6548.432           | 3620.103  |
| Experimental      | 48             | yes | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6568.980           | 4335.312  |
| Experimental Fast | 24             | no  | 6B   | 384         | 256       | 128       | 768        | 4                 | 14             | 6531.170           | 3046.822  |

Parameter counts include the frozen ESMC-6B backbone. FLOPs use PyTorch's
`FlopCounterMode`, with each multiply-add counted as two operations. They are
measured in bfloat16 for one batch of 128 residues (640 atom rows), one
query-only MSA row, one recurrent update, and one denoising step.

### Links

- **Code**: [ofoldx.models.esmfold2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/esmfold2)
- **Paper**: [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf)
- **Developed by**: Salvatore Candido, Thomas Hayes, Alexander Derry, et al.
- **Model type**: ESMC-conditioned recurrent all-atom folding model with diffusion coordinate generation
- **Original Repository**: [Biohub/esm](https://github.com/Biohub/esm)

## Usage

Install OFoldX:

```bash
pip install ofoldx
```

You can predict a protein structure from its sequence with the high-level
`fold` API:

```python
import ofoldx

output = ofoldx.fold("MKTAYIAKQRQISFVKSHFSRQ", model="oteam/esmfold2")
```

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
released distogram contact objective and ESMC pseudo-perplexity regularizer,
then refolds a discrete selected sequence.

## Limitations

- ESMFold2 predictions are computational hypotheses, not experimental
  measurements. Structures and designed binders require independent
  experimental validation.
- PDB and AFDB contain uneven coverage of protein families, conformational
  states, modifications, disorder, and non-protein biomolecules. Accuracy may
  degrade for underrepresented chemistry and biology.
- Quality depends on inference-time sampling and recurrent refinement.
  Reducing samples or steps trades accuracy for speed, while pair and atom
  computations make large systems memory-intensive.
- The ESMC dependency has a 2048-token context. The Processor runs one ESMC row
  per protein chain; MSA evidence applies only to protein entities, and Fast
  variants reject it.
- Canonical small-molecule inputs must carry an explicit, complete reference
  conformer. The Processor does not generate geometry from SMILES. Observed
  coordinates in a `Structure` are not folding features.
- The released source emits distance-conditioning tensors that its released
  folding forward path does not consume. OFoldX rejects this ineffective
  conditioning rather than advertising it as supported.
- Binder design currently targets one single-chain protein and one binder per
  call. It does not automatically run the multi-inversion, hero-critic, and
  scaling-critic ensemble from the released workflow.

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
