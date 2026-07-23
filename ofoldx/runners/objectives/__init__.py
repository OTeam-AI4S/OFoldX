r"""Training objectives — the loss reducers, keyed by objective name.

An objective is a callable ``(model, batch) -> (output, loss)`` registered into
`ofoldx.runners.registry.OBJECTIVES` when its module is imported. The Model's
forward returns predictions or unreduced loss primitives; the objective reduces one
batch to a scalar (the loss lives here, never in the Model).
"""
