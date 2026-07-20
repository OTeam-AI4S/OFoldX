# pipelines

OFoldX exposes one concrete Pipeline for each standard biological task:

- `FoldPipeline`
- `DesignPipeline`
- `ScorePipeline`
- `PackPipeline`

Each package export is the same class object as its top-level export. A standard
Pipeline binds a Model to its Processor, delegates ordinary `.to(...)` calls, and
exposes only its matching verb. The repository-backed factory validates registration,
pairing, and the requested task; a direct constructor trusts the pair supplied
by its caller.

```python
from ofoldx import AutoModel, AutoProcessor, FoldPipeline

model_repository = "your-organization/published-fold-model"
model = AutoModel(model_repository)
processor = AutoProcessor(model_repository)
folding = FoldPipeline(model=model, processor=processor)
result = folding.fold("MKT...")
```

For repository-backed construction, the task stays explicit and the model ID selects
the paired Model and Processor:

```python
import ofoldx

folding = ofoldx.pipeline("fold", model="your-organization/published-fold-model")
result = folding.fold("MKT...")
```

There is no public generic `Pipeline` base, task inference, default model, or factory
form that accepts live Model/Processor objects. Standard Pipelines have no independent
manifest or serialized composition format.
