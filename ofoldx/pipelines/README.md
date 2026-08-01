# pipelines

OFoldX publishes family-owned concrete Pipelines. Every routable repository carries
one `pipeline.json` marker naming its registered class; that class owns repository
loading, saving, placement, and its exact biological method signatures.

`PairedPipeline` implements this lifecycle for one Model–Processor pair.
`CompositePipeline` implements it for a concrete class's fixed named pairs. Every
current family artifact uses one of these two ordinary lifecycles; complete child
Models such as ESMFold2's ESMC-backed graph remain ordinary pair state.

```python
import ofoldx

folding = ofoldx.pipeline("fold", model="your-organization/published-fold-model")
result = folding.fold("MKT...")
```

The task remains explicit and is checked against the selected class. The factory
does not infer repository layout, load a root pair itself, or fall back when a
marker or requested public method is missing. There are no generic Fold, Design,
Score, or Pack Pipeline classes hiding family signature differences.
