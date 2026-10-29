# Analysis

`ofoldx.analysis` helps inspect and select generated sequences and structures. It combines molecular-site
matching, sequence annotations, diversity measures, and candidate ranking for use in model workflows.

## Usage

Screen a peptide sequence for composition and motif annotations:

```python
from ofoldx.analysis.sequence import sequence_liabilities

annotations = sequence_liabilities("ACDEFGHIKLMNPQRSTVWY", modality="peptide")
for annotation in annotations:
    print(annotation.motif, annotation.position, annotation.severity)
```

The returned annotations include their sequence positions and integer screening weights. These weights are
uncalibrated heuristics. `OddCysteineCount` describes composition; disulfide pairing requires additional evidence.
`BasicResidueExcess` reports K+R-D-E counts rather than a charge calculated at a specified pH.

## Choosing an Analysis

| Module                                 | Use                                                                                                    |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [Structure][ofoldx.analysis.structure] | Match molecular sites, compare coordinates, assign secondary structure, and measure contacts or burial |
| [Sequence][ofoldx.analysis.sequence]   | Annotate peptide and antibody sequences with motif and composition screens                             |
| [Diversity][ofoldx.analysis.diversity] | Deduplicate candidates, make greedy selections, and summarize sequence or structure diversity          |
| [Ranking][ofoldx.analysis.ranking]     | Filter candidates by named measurements and combine their ranks                                        |

Structure analysis includes RMSD/lDDT comparisons, DSSP secondary structure, noncovalent contacts, target burial,
hydrophobic patches, and binding-site adherence. Molecular-site matching keeps comparisons aligned to the intended
parts of each structure.

Diversity analysis includes Vendi scores and Foldseek novelty. For Foldseek, supply an in-memory operation that
returns reference matches; process execution and reference data are configured by the calling workflow.

[`CandidateFilter`][ofoldx.analysis.ranking.CandidateFilter] and
[`rank_candidates`][ofoldx.analysis.ranking.rank_candidates] operate on named measurements. A model family's
pipeline combines those measurements with its own candidate and result types.

## Related Modules

- [Metrics](../metrics/README.md): individual score formulas and confidence reductions.
- [Results](../results/README.md): candidate structures, scores, and their molecular scope.
- [Models](../models/README.md): the analyses used by each scientific workflow.

[ofoldx.analysis.diversity]: https://ofoldx.valtech.ai/analysis/diversity/#ofoldx.analysis.diversity
[ofoldx.analysis.ranking]: https://ofoldx.valtech.ai/analysis/ranking/#ofoldx.analysis.ranking
[ofoldx.analysis.ranking.CandidateFilter]: https://ofoldx.valtech.ai/analysis/ranking/#ofoldx.analysis.ranking.CandidateFilter
[ofoldx.analysis.ranking.rank_candidates]: https://ofoldx.valtech.ai/analysis/ranking/#ofoldx.analysis.ranking.rank_candidates
[ofoldx.analysis.sequence]: https://ofoldx.valtech.ai/analysis/sequence/#ofoldx.analysis.sequence
[ofoldx.analysis.structure]: https://ofoldx.valtech.ai/analysis/structure/#ofoldx.analysis.structure
