from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.runners.config import (
    DataConfig,
    DatasetConfig,
    DatasetMixtureConfig,
    MixtureComponentConfig,
    RunnerConfig,
)


def _config(data: dict[str, object]) -> RunnerConfig:
    return RunnerConfig({"model": {"name": "proteinmpnn"}, "data": data})


def test_data_config_coerces_only_dataset_mappings_to_mixtures() -> None:
    data = DataConfig(
        {
            "train": {"datasets": {"pdb": {"source": "train.cif"}}},
            "val": {"source": "val.cif"},
            "test": "test.cif",
        }
    )

    assert isinstance(data.train, DatasetMixtureConfig)
    assert isinstance(data.train.datasets["pdb"], MixtureComponentConfig)
    assert data.train.datasets["pdb"].sampling_ratio == 1.0
    assert data.train.epoch_size is None
    assert data.train.replacement is True
    assert data.train.seed is None
    assert isinstance(data.val, DatasetConfig)
    assert isinstance(data.test, DatasetConfig)


def test_valid_mixture_configuration_posts() -> None:
    config = _config(
        {
            "train": {
                "datasets": {
                    "PDB-1": {
                        "source": "train.cif",
                        "sampling_ratio": 2.5,
                        "max_tokens": 384,
                        "seq_to_pdb_index": "index.json",
                        "msa_path": "msa",
                        "cluster_sampling": {"enabled": True, "cluster_file": "clusters.txt"},
                    },
                    "synthetic_set": {"source": "synthetic.cif"},
                },
                "epoch_size": 100,
                "replacement": False,
                "seed": 0,
            }
        }
    )

    config.post()


def test_component_processing_controls_preserve_values_nulls_and_absence() -> None:
    config = _config(
        {
            "train": {
                "datasets": {
                    "bounded": {
                        "source": "bounded.cif",
                        "processing_timeout_seconds": 12.5,
                        "max_consecutive_processing_errors": 7,
                    },
                    "disabled": {
                        "source": "disabled.cif",
                        "processing_timeout_seconds": None,
                        "max_consecutive_processing_errors": None,
                    },
                    "defaults": {"source": "defaults.cif"},
                }
            }
        }
    )

    config.post()

    bounded = config.data.train.datasets["bounded"]
    assert bounded.processing_timeout_seconds == 12.5
    assert bounded.max_consecutive_processing_errors == 7
    disabled = config.data.train.datasets["disabled"]
    assert "processing_timeout_seconds" in disabled
    assert disabled.processing_timeout_seconds is None
    assert "max_consecutive_processing_errors" in disabled
    assert disabled.max_consecutive_processing_errors is None
    defaults = config.data.train.datasets["defaults"]
    assert "processing_timeout_seconds" not in defaults
    assert "max_consecutive_processing_errors" not in defaults


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("processing_timeout_seconds", True),
        ("processing_timeout_seconds", 0),
        ("processing_timeout_seconds", -0.5),
        ("processing_timeout_seconds", float("inf")),
        ("processing_timeout_seconds", "30"),
        ("max_consecutive_processing_errors", True),
        ("max_consecutive_processing_errors", 0),
        ("max_consecutive_processing_errors", -1),
        ("max_consecutive_processing_errors", float("inf")),
        ("max_consecutive_processing_errors", "5"),
        ("max_consecutive_processing_errors", 1.5),
    ],
    ids=[
        "timeout-bool",
        "timeout-zero",
        "timeout-negative",
        "timeout-infinite",
        "timeout-string",
        "limit-bool",
        "limit-zero",
        "limit-negative",
        "limit-infinite",
        "limit-string",
        "limit-noninteger",
    ],
)
def test_invalid_component_processing_controls_name_component_and_key(
    key: str,
    value: object,
) -> None:
    config = _config(
        {
            "train": {
                "datasets": {
                    "broken_component": {"source": "x.cif", key: value},
                }
            }
        }
    )

    with pytest.raises(ValueError) as error:
        config.post()

    message = str(error.value)
    assert "broken_component" in message
    assert key in message


@pytest.mark.parametrize(
    ("split", "mixture", "message"),
    [
        ("val", {"datasets": {"pdb": {"source": "x.cif"}}}, "only supported for training"),
        ("train", {"datasets": {}}, "nonempty mapping"),
        ("train", {"datasets": {"bad.name": {"source": "x.cif"}}}, "invalid name"),
        ("train", {"datasets": {"pdb": {"source": ""}}}, "source must be a nonempty"),
        ("train", {"datasets": {"pdb": {"source": 7}}}, "source must be a nonempty"),
        ("train", {"datasets": {"pdb": {"source": "x", "sampling_ratio": 0}}}, "strictly positive"),
        ("train", {"datasets": {"pdb": {"source": "x", "sampling_ratio": True}}}, "strictly positive"),
        ("train", {"datasets": {"pdb": {"source": "x", "sampling_ratio": float("inf")}}}, "finite"),
        ("train", {"datasets": {"pdb": {"source": "x", "seq_to_pdb_index": "i"}}}, "provided together"),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "seq_to_pdb_index": "", "msa_path": "msa"}}},
            "seq_to_pdb_index must be a nonempty string",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "seq_to_pdb_index": "index", "msa_path": " "}}},
            "msa_path must be a nonempty string",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "seq_to_pdb_index": 1, "msa_path": "msa"}}},
            "seq_to_pdb_index must be a nonempty string",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"enabled": 1}}}},
            "enabled must be a bool",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"enabled": True}}}},
            "cluster_file is required",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"epoch_size": 2}}}},
            "epoch_size must be None",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"seed": 2}}}},
            "seed must be None",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"replacement": False}}}},
            "replacement must remain True",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"cluster_file": ""}}}},
            "cluster_file must be a nonempty string",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"indices_file": 7}}}},
            "indices_file must be a nonempty string",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"contact_cutoff": True}}}},
            "contact_cutoff must be finite",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"include_all_chain_pairs": 1}}}},
            "include_all_chain_pairs must be a bool",
        ),
        (
            "train",
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"exclude_mol_types": ["ions", 1]}}}},
            "exclude_mol_types must be a sequence",
        ),
        ("train", {"datasets": {"pdb": {"source": "x"}}, "epoch_size": 0}, "positive integer"),
        ("train", {"datasets": {"pdb": {"source": "x"}}, "epoch_size": "8"}, "positive integer"),
        ("train", {"datasets": {"pdb": {"source": "x"}}, "seed": -1}, "nonnegative integer"),
        ("train", {"datasets": {"pdb": {"source": "x"}}, "seed": "7"}, "nonnegative integer"),
        ("train", {"datasets": {"pdb": {"source": "x"}}, "replacement": 1}, "must be a bool"),
    ],
)
def test_invalid_mixture_configuration_is_rejected(split: str, mixture: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config({split: mixture}).post()


@pytest.mark.parametrize(
    "mixture",
    [
        {"datasets": {"pdb": {"source": "x"}}, "source": "also-x"},
        {"datasets": {"pdb": {"source": "x", "datasets": {"nested": {"source": "y"}}}}},
    ],
)
def test_nested_or_ambiguous_mixture_configuration_is_rejected(mixture: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="datasets"):
        _config({"train": mixture}).post()


def test_duplicate_explicit_cluster_indices_paths_are_rejected() -> None:
    mixture = {
        "datasets": {
            "first": {"source": "a", "cluster_sampling": {"indices_file": "cache/indices.jsonl"}},
            "second": {"source": "b", "cluster_sampling": {"indices_file": "cache/../cache/indices.jsonl"}},
        }
    }

    with pytest.raises(ValueError, match="duplicates the explicit cache path"):
        _config({"train": mixture}).post()


def test_custom_train_split_allows_a_mixture() -> None:
    config = RunnerConfig(
        {
            "model": {"name": "proteinmpnn"},
            "train_splits": ["fit"],
            "data": {"fit": {"datasets": {"pdb": {"source": "train.cif"}}}},
        }
    )

    config.post()


def test_scalar_custom_train_split_allows_a_mixture() -> None:
    config = RunnerConfig(
        {
            "model": {"name": "proteinmpnn"},
            "train_splits": "fit",
            "data": {"fit": {"datasets": {"pdb": {"source": "train.cif"}}}},
        }
    )

    config.post()


def test_pathlike_component_paths_are_normalized(tmp_path: Path) -> None:
    config = _config(
        {
            "train": {
                "datasets": {
                    "pdb": {
                        "source": tmp_path / "mmcif",
                        "seq_to_pdb_index": tmp_path / "seq.json",
                        "msa_path": tmp_path / "msa",
                        "cluster_sampling": {
                            "cluster_file": tmp_path / "clusters.txt",
                            "indices_file": tmp_path / "indices.jsonl",
                        },
                    }
                }
            }
        }
    )

    config.post()

    component = config.data.train.datasets["pdb"]
    assert component.source == str(tmp_path / "mmcif")
    assert component.seq_to_pdb_index == str(tmp_path / "seq.json")
    assert component.msa_path == str(tmp_path / "msa")
    assert component.cluster_sampling.cluster_file == str(tmp_path / "clusters.txt")
    assert component.cluster_sampling.indices_file == str(tmp_path / "indices.jsonl")


@pytest.mark.parametrize(
    ("mixture", "message"),
    [
        ({"datasets": {"pdb": {"source": "x"}}, "epch_size": 10}, "unsupported mixture keys"),
        ({"datasets": {"pdb": {"source": "x", "seed": 1}}}, "mixture-level controls"),
        (
            {"datasets": {"pdb": {"source": "x", "cluster_sampling": {"cluser_file": "clusters"}}}},
            "unsupported keys",
        ),
    ],
)
def test_mixture_control_typos_and_misplacement_are_rejected(
    mixture: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config({"train": mixture}).post()


def test_legacy_data_config_behavior_is_unchanged(tmp_path: Path) -> None:
    data = DataConfig(
        {
            "train": tmp_path / "train",
            "val": {"source": tmp_path / "val", "max_tokens": 256},
            "test": None,
        }
    )

    assert isinstance(data.train, DatasetConfig)
    assert not isinstance(data.train, DatasetMixtureConfig)
    assert data.train.source == str(tmp_path / "train")
    assert data.val.source == str(tmp_path / "val")
    assert data.val.max_tokens == 256
    assert data.test.source is None


def test_runner_config_boot_recurses_into_components_and_interpolates_processor_values() -> None:
    config = RunnerConfig(
        {
            "model": {"name": "proteinmpnn"},
            "processor": {"max_tokens": 384},
            "data": {
                "train": {
                    "datasets": {
                        "PDB-1": {
                            "source": "train.cif",
                            "seq_to_pdb_index": "index.json",
                            "msa_path": "msa",
                            "max_tokens": "${processor.max_tokens}",
                            "cluster_sampling": {
                                "enabled": True,
                                "cluster_file": "clusters.txt",
                                "indices_file": "indices.jsonl",
                            },
                        }
                    }
                }
            },
        }
    )

    config.boot()

    component = config.data.train.datasets["PDB-1"]
    assert isinstance(component, MixtureComponentConfig)
    assert component.source == "train.cif"
    assert component.seq_to_pdb_index == "index.json"
    assert component.msa_path == "msa"
    assert component.cluster_sampling.enabled is True
    assert component.cluster_sampling.cluster_file == "clusters.txt"
    assert component.max_tokens == 384


def test_boot_interpolates_and_normalizes_all_mixture_control_types(tmp_path: Path) -> None:
    config = RunnerConfig(
        {
            "model": {"name": "proteinmpnn"},
            "mixture_values": {
                "source": tmp_path / "mmcif",
                "ratio": 2,
                "seq_index": tmp_path / "seq.json",
                "msa": tmp_path / "msa",
                "cluster": tmp_path / "clusters.txt",
                "indices": tmp_path / "indices.jsonl",
                "epoch_size": 80,
                "replacement": False,
                "seed": 7,
            },
            "data": {
                "train": {
                    "datasets": {
                        "pdb": {
                            "source": "${mixture_values.source}",
                            "sampling_ratio": "${mixture_values.ratio}",
                            "seq_to_pdb_index": "${mixture_values.seq_index}",
                            "msa_path": "${mixture_values.msa}",
                            "cluster_sampling": {
                                "enabled": True,
                                "cluster_file": "${mixture_values.cluster}",
                                "indices_file": "${mixture_values.indices}",
                            },
                        }
                    },
                    "epoch_size": "${mixture_values.epoch_size}",
                    "replacement": "${mixture_values.replacement}",
                    "seed": "${mixture_values.seed}",
                }
            },
        }
    )

    config.boot()

    mixture = config.data.train
    component = mixture.datasets["pdb"]
    assert mixture.epoch_size == 80
    assert mixture.replacement is False
    assert mixture.seed == 7
    assert component.source == str(tmp_path / "mmcif")
    assert component.sampling_ratio == 2.0
    assert component.seq_to_pdb_index == str(tmp_path / "seq.json")
    assert component.msa_path == str(tmp_path / "msa")
    assert component.cluster_sampling.cluster_file == str(tmp_path / "clusters.txt")
    assert component.cluster_sampling.indices_file == str(tmp_path / "indices.jsonl")


@pytest.mark.parametrize(
    ("values", "component", "message"),
    [
        ({"bad": 7}, {"source": "${values.bad}"}, "source must be a nonempty"),
        (
            {"bad": "not-a-ratio"},
            {"source": "x", "sampling_ratio": "${values.bad}"},
            "sampling_ratio must be finite",
        ),
        (
            {"bad": 7},
            {"source": "x", "seq_to_pdb_index": "${values.bad}", "msa_path": "msa"},
            "seq_to_pdb_index must be a nonempty",
        ),
        (
            {"bad": 7},
            {
                "source": "x",
                "cluster_sampling": {"enabled": True, "cluster_file": "${values.bad}"},
            },
            "cluster_file must be a nonempty",
        ),
    ],
)
def test_boot_rejects_invalid_interpolated_component_values(
    values: dict[str, object], component: dict[str, object], message: str
) -> None:
    config = RunnerConfig(
        {
            "model": {"name": "proteinmpnn"},
            "values": values,
            "data": {"train": {"datasets": {"pdb": component}}},
        }
    )

    with pytest.raises(ValueError, match=message):
        config.boot()
