# Metrics

`ofoldx.metrics` computes sequence and structure comparisons, confidence expectations, and diversity scores.
Use the individual functions when comparing predictions or implementing a model's evaluation workflow.

## Usage

Compare a designed sequence with a reference of the same length:

```python
from ofoldx.metrics.sequence import sequence_recovery

recovery = sequence_recovery("ACDF", "ACDE")
print(recovery)  # 0.75
```

For coordinate and confidence measurements, supply the observation mask, bins, and molecular scope required by the
function. Learned prediction heads are available in [Modules](../modules/README.md).

## Choosing a Metric

| Module                                  | Measurements                                                                                   |
| --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [Confidence][ofoldx.metrics.confidence] | Categorical expectations, cutoff probabilities, pLDDT, and resolved probabilities              |
| [TM Scores][ofoldx.metrics.tm]          | TM length scaling, pTM/ipTM, chain-pair reductions, and directional iPSAE                      |
| [Structure][ofoldx.metrics.structure]   | Aligned RMSD, lDDT/TM, secondary structure, label agreement, and binding-site adherence        |
| [Sequence][ofoldx.metrics.sequence]     | Composition, recovery, NLL, normalized alignment scores, hydrophobicity, and isoelectric point |
| [Diversity][ofoldx.metrics.diversity]   | Vendi entropy and normalized PSD projections of alignment or TM similarities                   |

Vendi scores use valid positive-semidefinite (PSD) kernels. The diversity helpers provide explicit normalized PSD
projections when working with alignment or TM similarities.

## Interpreting Confidence Scores

pTM selects the best eligible source frame after averaging target-pair TM expectations. Target weights define
each mean; reference weights select the frame without rescaling its reported mean. An empty eligible set returns
zero, while an explicitly eligible frame without a target domain has an undefined mean. See
[`predicted_tm_score`][ofoldx.metrics.tm.predicted_tm_score] for the axes and weighting rules.

Chain-pair scores answer different questions: `chain_pair_iptm` selects source frames in one chain and targets in
another, while `chain_pair_complex_iptm` evaluates both chains with pair-local length calibration and is symmetric.
Its diagonal is zero; `chain_ptm` supplies intra-chain scores. Partner-mean functions average each eligible distinct
partner once, omit nonfinite scores, and return NaN for an empty partner domain. The [TM reference][ofoldx.metrics.tm]
describes each reduction.

## Related Modules

- [Results](../results/README.md): retain units and molecular axes alongside scores.
- [Modules](../modules/README.md): learned heads that produce confidence logits.
