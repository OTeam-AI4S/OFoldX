r"""
Command entry points for OFoldX runners.
"""

from __future__ import annotations

from typing import Any

import danling as dl

from ofoldx.runners import RUNNERS, Config


def train() -> Any:
    r"""
    Train an OFoldX model with the configured runner.

    Builds a [`Config`][ofoldx.runners.Config] from the command line (falling back to the
    ``config`` default), constructs the matching runner from the [`RUNNERS`][ofoldx.runners]
    registry, and drives its training loop. This is the ``otrain`` console entry point.

    Returns:
        Whatever the selected runner's ``train`` method returns (typically the final training
        metrics or checkpoint summary).

    Examples:
        >>> from ofoldx.api import train
        >>> train()  # doctest: +SKIP
    """

    config = Config()
    config.parse(default_config="config")
    with dl.debug(config.get("debug", False)):
        runner = RUNNERS.build(config)
        print(runner)
        return runner.train()


def evaluate() -> Any:
    r"""
    Evaluate an OFoldX model with the configured runner.

    Parses a [`Config`][ofoldx.runners.Config] from the command line, builds the matching
    runner from the [`RUNNERS`][ofoldx.runners] registry, and runs a single evaluation pass
    over the configured dataset. This is the ``oevaluate`` console entry point.

    Returns:
        The evaluation result produced by the selected runner's ``evaluate`` method (typically a
        mapping of metric names to values).

    Examples:
        >>> from ofoldx.api import evaluate
        >>> evaluate()  # doctest: +SKIP
    """

    config = Config()
    config.parse(default_config="config")
    runner = RUNNERS.build(config)
    print(runner)
    result = runner.evaluate()
    print(result)
    return result


def infer() -> Any:
    r"""
    Run inference with the configured runner.

    Parses a [`Config`][ofoldx.runners.Config] from the command line, builds the matching
    runner from the [`RUNNERS`][ofoldx.runners] registry, and runs its inference pipeline.
    When the runner returns a list of prediction batches their count is echoed to stdout. This is
    the ``oinfer`` console entry point.

    Returns:
        The inference result produced by the selected runner's ``infer`` method, commonly a list
        of per-batch prediction outputs.

    Examples:
        >>> from ofoldx.api import infer
        >>> infer()  # doctest: +SKIP
    """

    config = Config()
    config.parse(default_config="config")
    runner = RUNNERS.build(config)
    print(runner)
    result = runner.infer()
    if isinstance(result, list):
        print(f"infer: {len(result)} batch(es)")
    return result
