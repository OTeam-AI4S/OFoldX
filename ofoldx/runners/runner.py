r"""
DanLing runner integration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias, cast

import danling as dl
import torch
from danling.runners.utils import RunnerMode
from torch import Tensor, nn

from ..data import StructureDataset
from ..data.structure import Structure
from ..models import (
    AutoModel,
    BaseInput,
    StructurePredictionOutput,
)
from ..modules.loss import CompositeLoss
from ..processors import AutoProcessor
from .config import Config, DataConfig
from .registry import RUNNERS

_ModelBatch: TypeAlias = BaseInput


@RUNNERS.register("ofoldx", default=True)
class Runner(dl.Runner):
    r"""
    Train, evaluate, and run inference with OFoldX models.

    A [`danling.Runner`][danling.Runner] specialization that boots the config, builds the model via
    [`AutoModel`][ofoldx.models.AutoModel], wires up the composite loss, and constructs the configured
    structure datasets — leaving dataloader construction, distributed setup, and the epoch loop to danling.
    Registered as the default ``"ofoldx"`` runner in [`RUNNERS`][ofoldx.runners.registry.RUNNERS].

    Args:
        config: Runner [`Config`][ofoldx.runners.config.Config] object, or a raw mapping that is wrapped
            into one.

    Examples:
        >>> from ofoldx.runners import Runner  # doctest: +SKIP
        >>> runner = Runner({"model": {"architecture": "alphafold3"}})  # doctest: +SKIP
        >>> runner.train()  # doctest: +SKIP
    """

    config: Config
    model: nn.Module
    criterion: CompositeLoss

    def __init__(self, config: Config | Mapping[str, Any]) -> None:
        if not isinstance(config, Config):
            config = Config(config)
        config.boot()
        super().__init__(config)
        self.model = cast(nn.Module, AutoModel.from_config(config.model))
        self.criterion = CompositeLoss(weights=config.loss.weights)
        self.datasets.update(self.build_datasets())

    def build_datasets(self) -> dict[str, StructureDataset[_ModelBatch]]:
        r"""
        Build the configured structure datasets, one per split.

        Reads [`DataConfig`][ofoldx.runners.config.DataConfig] from the runner config, builds the model's
        [`AutoProcessor`][ofoldx.processors.AutoProcessor] featurizer, and wraps each configured split in a
        [`StructureDataset`][ofoldx.data.StructureDataset]. Returns an empty mapping when no data is
        configured; dataloader construction is deferred to danling.

        Returns:
            Mapping from split name to its structure dataset, empty when no data is configured.

        Examples:
            >>> runner.build_datasets()  # doctest: +SKIP
            {'train': StructureDataset(...)}
        """

        data_config = self.config.data
        if data_config is None:
            return {}
        if not isinstance(data_config, DataConfig):
            data_config = DataConfig(data_config)
        model_config = cast(Any, self.model).config
        processor = cast(Callable[[Structure], _ModelBatch], AutoProcessor.from_config(model_config))
        return {
            split: StructureDataset(
                _split_root(source, root=data_config.root),
                format=data_config.format,
                max_tokens=data_config.max_tokens,
                processor=processor,
            )
            for split, source in data_config.splits().items()
        }

    def train_step(self, batch: _ModelBatch) -> tuple[StructurePredictionOutput, Tensor]:
        r"""
        Run one training step: forward, loss, backward, and optimizer step.

        Moves the batch to the runner device, runs the model under the train context with the configured
        training-sampling kwargs, evaluates the composite loss, and backpropagates it.

        Args:
            batch: Typed model input batch.

        Returns:
            The model [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput] and the scalar
            total training loss.

        Examples:
            >>> runner.train_step(batch)  # doctest: +SKIP
        """

        batch = self.to_device(batch)
        with self.train_context():
            output = cast(StructurePredictionOutput, self.model(batch, **self.config.train_sampling.model_kwargs()))
            loss = self.criterion(output, batch)
            self.backward(loss.total)
            self.step()
        return output, loss.total

    @torch.inference_mode()
    def evaluate_step(self, batch: _ModelBatch) -> tuple[StructurePredictionOutput, Tensor]:
        r"""
        Run one evaluation step under inference mode.

        Prefers the EMA model when available, runs it under the inference context with the inference-sampling
        kwargs, and computes the composite loss without tracking gradients.

        Args:
            batch: Typed model input batch.

        Returns:
            The model [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput] and the scalar
            total evaluation loss.

        Examples:
            >>> runner.evaluate_step(batch)  # doctest: +SKIP
        """

        batch = self.to_device(batch)
        model = self.ema or self.model
        with self.infer_context():
            output = cast(StructurePredictionOutput, model(batch, **self.config.infer_sampling.model_kwargs()))
            loss = self.criterion(output, batch)
        return output, loss.total

    @torch.inference_mode()
    def infer_step(self, batch: _ModelBatch) -> StructurePredictionOutput:
        r"""
        Run one inference step under inference mode, returning only the prediction.

        Like [`evaluate_step`][ofoldx.runners.runner.Runner.evaluate_step] but without loss evaluation:
        it moves the batch to device and runs the EMA-or-base model with the inference-sampling kwargs.

        Args:
            batch: Typed model input batch.

        Returns:
            The [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput] for the batch.

        Examples:
            >>> runner.infer_step(batch)  # doctest: +SKIP
        """

        batch = self.to_device(batch)
        model = self.ema or self.model
        with self.infer_context():
            return cast(StructurePredictionOutput, model(batch, **self.config.infer_sampling.model_kwargs()))

    def infer(
        self,
        split: str = "infer",
        *,
        steps: int | None = None,
        stream: bool = False,
    ) -> list[StructurePredictionOutput] | Iterator[StructurePredictionOutput]:
        r"""
        Run inference over a configured dataloader split.

        Switches the runner into infer mode on ``split`` and consumes its dataloader, marking heartbeat progress
        and periodically collecting garbage between batches. With ``stream=True`` the generator is returned lazily
        so callers can process outputs without holding every prediction in memory.

        Args:
            split: Dataloader split name.
            steps: Optional maximum number of batches to consume; ``None`` runs the whole split.
            stream: If ``True``, return an iterator instead of materializing a list.

        Returns:
            Inference outputs as a materialized list, or a streaming iterator when ``stream`` is ``True``.

        Raises:
            ValueError: If ``steps`` is negative.

        Examples:
            >>> outputs = runner.infer("infer", steps=8)  # doctest: +SKIP
            >>> for output in runner.infer(stream=True):  # doctest: +SKIP
            ...     handle(output)
        """

        self.mode = RunnerMode.infer
        self.split = split
        loader = self.dataloaders[split]
        if steps is not None and steps < 0:
            raise ValueError(f"invalid steps: expected a non-negative value, got {steps}")

        def iterator() -> Iterator[StructurePredictionOutput]:
            r"""
            Yield inference outputs for the selected split.

            Yields:
                Structure-prediction outputs.
            """

            for iteration, batch in enumerate(loader):
                if steps is not None and iteration >= steps:
                    break
                output = self.infer_step(batch)
                self.supervisor.mark_heartbeat_progress()
                yield output
                self.supervisor.maybe_collect_garbage(iteration + 1, scope=f"infer:{split}")

        if stream:
            return iterator()
        return list(iterator())

    @staticmethod
    def collate_fn(batch: list[Any]) -> Any:
        r"""
        Collate dataloader items while preserving already-batched model inputs.

        Processor outputs ([`BaseInput`][ofoldx.models.BaseInput]) arrive pre-batched, so they must pass
        through untouched rather than being re-stacked; this requires ``dataloader.batch_size=1``. Anything else
        falls back to DanLing's default tensor collation.

        Args:
            batch: Raw dataloader item list.

        Returns:
            The single typed model input when the batch is a prepared input, otherwise the default DanLing
            collated batch.

        Raises:
            ValueError: If multiple already-batched model inputs are collated together.

        Examples:
            >>> Runner.collate_fn([prepared_input])  # doctest: +SKIP
            <prepared_input>
        """

        if batch and _is_prepared_model_input(batch[0]):
            if len(batch) != 1:
                raise ValueError("model-local input batches are already batched; use dataloader.batch_size=1")
            return batch[0]
        return dl.TorchRunner.collate_fn(batch)


def _split_root(path: str | PathLike[str], *, root: str | PathLike[str] | None) -> Path:
    r"""
    Resolve one split directory against an optional shared ``root``.

    Args:
        path: Split path from the data config; may be relative or absolute, with ``~`` expanded.
        root: Optional common parent prepended only when ``path`` is relative.

    Returns:
        The absolute-or-relative split directory as a [`Path`][pathlib.Path].
    """

    split_root = Path(path).expanduser()
    if root is not None and not split_root.is_absolute():
        split_root = Path(root).expanduser() / split_root
    return split_root


def _is_prepared_model_input(value: object) -> bool:
    r"""
    Return whether ``value`` looks like one already-batched model input.

    Runner collation only needs to preserve processor/model batches. Raw structures still flow through the dataset
    processor, and plain tensors should keep using the default tensor collator.

    Args:
        value: Candidate dataloader item.

    Returns:
        ``True`` when ``value`` is a [`BaseInput`][ofoldx.models.BaseInput] processor/model batch.
    """

    return isinstance(value, BaseInput)
