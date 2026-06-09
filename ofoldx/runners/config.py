r"""
Runner configuration.
"""

from __future__ import annotations

import danling as dl
from chanfig import Config as ChanfigConfig


class DataConfig(ChanfigConfig):
    r"""
    Dataset split configuration for runner fine-tuning.

    Points the runner at the on-disk structure directories that back each dataloader split. Splits may be
    given individually (``train`` / ``validation`` / ``test`` / ``infer``) or collapsed to a single ``root``,
    in which case the whole tree is treated as the training split.

    Attributes:
        root: Common parent directory prepended to relative split paths; also the sole training source when no
            explicit split is set.
        format: Structure-file format hint forwarded to the dataset (``"auto"`` sniffs it per file).
        train: Directory of training structures, relative to ``root`` when not absolute.
        validation: Directory of validation structures.
        test: Directory of held-out test structures.
        infer: Directory of structures to run inference over.
        max_tokens: Optional cap on tokens per structure; larger structures are dropped by the dataset.

    Examples:
        >>> config = DataConfig()
        >>> config.format
        'auto'
        >>> config.splits()
        {}
        >>> DataConfig(train="train_dir").splits()
        {'train': 'train_dir'}
    """

    root: str | None = None
    format: str = "auto"
    train: str | None = None
    validation: str | None = None
    test: str | None = None
    infer: str | None = None
    max_tokens: int | None = None

    def splits(self) -> dict[str, str]:
        r"""
        Resolve the configured dataset splits to their structure directories.

        Explicit per-split directories take precedence; when none are set but a shared ``root`` is configured,
        the whole tree collapses to a single ``train`` split rooted at ``"."``.

        Returns:
            Mapping from split name to configured structure directory, empty when nothing is configured.

        Examples:
            >>> DataConfig(validation="val", test="test").splits()
            {'validation': 'val', 'test': 'test'}
            >>> DataConfig(root="/data").splits()
            {'train': '.'}
            >>> DataConfig().splits()
            {}
        """

        sources = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "infer": self.infer,
        }
        splits = {split: source for split, source in sources.items() if source is not None}
        if splits:
            return splits
        if self.root is not None:
            return {"train": "."}
        return {}


class LossConfig(ChanfigConfig):
    r"""
    Training loss configuration.

    Selects how the composite training loss weights its individual terms.

    Attributes:
        weights: Optional mapping from loss-term name to scalar weight; ``None`` lets
            [`CompositeLoss`][ofoldx.modules.loss.CompositeLoss] apply its built-in defaults.

    Examples:
        >>> config = LossConfig()
        >>> config.weights is None
        True
        >>> LossConfig(weights={"distogram": 0.3}).weights["distogram"]
        0.3
    """

    weights: dict[str, float] | None = None


class SamplingConfig(ChanfigConfig):
    r"""
    Model sampling options for one runner phase.

    Holds the trunk-recycling and diffusion-sampling knobs applied to the model forward in a single phase
    (training vs. inference each own an instance). Only the fields the phase sets are forwarded, so the model
    falls back to its own defaults for the rest.

    Attributes:
        recycling_steps: Number of trunk recycling iterations; ``None`` uses the model default.
        num_sampling_steps: Number of diffusion sampling steps; ``None`` uses the model default.
        num_samples: Number of structure samples to draw per input (always forwarded).

    Examples:
        >>> config = SamplingConfig()
        >>> config.num_samples
        1
        >>> config.model_kwargs()
        {'num_samples': 1}
    """

    recycling_steps: int | None = None
    num_sampling_steps: int | None = None
    num_samples: int = 1

    def model_kwargs(self) -> dict[str, int]:
        r"""
        Build the sampling keyword arguments forwarded to a top-level model forward call.

        ``num_samples`` is always included; ``recycling_steps`` and ``num_sampling_steps`` are added only when
        explicitly configured, so unset knobs defer to the model's own defaults.

        Returns:
            Sampling keyword arguments whose values are explicitly configured.

        Examples:
            >>> SamplingConfig().model_kwargs()
            {'num_samples': 1}
            >>> SamplingConfig(recycling_steps=3, num_sampling_steps=5).model_kwargs()
            {'num_samples': 1, 'recycling_steps': 3, 'num_sampling_steps': 5}
        """

        kwargs = {"num_samples": self.num_samples}
        if self.recycling_steps is not None:
            kwargs["recycling_steps"] = self.recycling_steps
        if self.num_sampling_steps is not None:
            kwargs["num_sampling_steps"] = self.num_sampling_steps
        return kwargs


class Config(dl.RunnerConfig):
    r"""
    OFoldX runner configuration.

    Top-level [`RunnerConfig`][danling.RunnerConfig] for the OFoldX runner: it selects the runner and model
    architecture, points at the dataset, and carries the loss weights plus the per-phase sampling knobs. Training
    and inference use separate [`SamplingConfig`][ofoldx.runners.config.SamplingConfig] instances so the two
    phases can recycle and sample differently.

    Attributes:
        runner: Registry key selecting the runner class (``"ofoldx"``).
        model: Model sub-config; ``architecture`` chooses the model family built by
            [`AutoModel`][ofoldx.models.AutoModel].
        data: Optional [`DataConfig`][ofoldx.runners.config.DataConfig]; ``None`` builds no datasets.
        loss: Composite-loss weighting.
        train_sampling: Sampling knobs applied during training steps.
        infer_sampling: Sampling knobs applied during evaluation and inference steps.

    Examples:
        >>> config = Config()
        >>> config.runner
        'ofoldx'
        >>> config.model["architecture"]
        'alphafold3'
        >>> config.train_sampling.num_sampling_steps
        20
        >>> config.infer_sampling.recycling_steps
        10
    """

    runner: str = "ofoldx"
    model: ChanfigConfig = ChanfigConfig({"architecture": "alphafold3"})
    data: DataConfig | None = None
    loss: LossConfig = LossConfig()
    train_sampling: SamplingConfig = SamplingConfig(num_sampling_steps=20)
    infer_sampling: SamplingConfig = SamplingConfig(recycling_steps=10)
