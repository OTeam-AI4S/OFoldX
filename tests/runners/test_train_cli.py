from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import ofoldx.runners.train as canonical


def _load_wrapper() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("ofoldx_test_train_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compatibility_wrapper_reexports_canonical_cli() -> None:
    wrapper = _load_wrapper()

    assert wrapper.parse_args is canonical.parse_args
    assert wrapper.build_config is canonical.build_config
    assert wrapper.main is canonical.main


@pytest.mark.parametrize(
    ("flag", "key", "value"),
    [
        ("--ccd-revision", "ccd_revision", "ccd-v1"),
        ("--ccd-snapshot-path", "ccd_snapshot_path", "/data/ccd"),
    ],
)
def test_explicit_ccd_locator_maps_to_a_copied_processor_config(
    flag: str,
    key: str,
    value: str,
) -> None:
    args = canonical.parse_args(["train.cif", "--processor-config", '{"feature": 1}', flag, value])
    original = args.processor_config

    config = canonical.build_config(args)

    assert config["processor"] == {"feature": 1, key: value}
    assert config["processor"] is not original
    assert original == {"feature": 1}


def test_default_processor_config_does_not_override_reader_ingestion_policy() -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif"]))

    assert config["processor"] == {}
    assert "include_hydrogens" not in config["processor"]
    assert "include_water" not in config["processor"]
    assert "bond_policy" not in config["processor"]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], False), (["--gradient-checkpointing"], True), (["--no-gradient-checkpointing"], False)],
)
def test_gradient_checkpointing_cli_maps_to_top_level_config(flags: list[str], expected: bool) -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif", *flags]))

    assert config["gradient_checkpointing"] is expected


def test_gradient_checkpointing_is_independent_of_training_state_checkpointing() -> None:
    config = canonical.build_config(
        canonical.parse_args(["train.cif", "--gradient-checkpointing", "--no-checkpoint"])
    )

    assert config["gradient_checkpointing"] is True
    assert config["ckpt"]["enabled"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["train.cif", "--ccd-revision", "one", "--ccd-snapshot-path", "two"],
        ["train.cif", "--ccd-revision", "one", "--processor-config", '{"ccd_revision": null}'],
        ["train.cif", "--ccd-revision", "one", "--processor-config", '{"ccd_snapshot_path": null}'],
        ["train.cif", "--pretrained", "repo", "--ccd-revision", "one"],
        ["train.cif", "--pretrained", "repo", "--processor-config", '{"feature": 1}'],
        [
            "train.cif",
            "--processor-config",
            '{"ccd_revision": "one", "ccd_snapshot_path": "two"}',
        ],
        ["train.cif", "--ccd-revision", ""],
        ["train.cif", "--ccd-snapshot-path", "   "],
        ["train.cif", "--processor-config", '{"ccd_revision": ""}'],
        ["train.cif", "--processor-config", '{"ccd_snapshot_path": "   "}'],
    ],
)
def test_ccd_locator_conflicts_and_empty_values_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(argv)


def test_pretrained_revision_remains_distinct_from_ccd_revision() -> None:
    args = canonical.parse_args(["train.cif", "--pretrained", "repo", "--revision", "model-rev"])
    config = canonical.build_config(args)

    assert config["model"]["revision"] == "model-rev"
    assert config["processor"] == {}


def test_print_config_does_not_construct_runner(capsys: pytest.CaptureFixture[str]) -> None:
    assert canonical.main(["train.cif", "--ccd-revision", "ccd-v1", "--print-config"]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["processor"] == {"ccd_revision": "ccd-v1"}


@pytest.mark.parametrize("model", [None, "protenix", "protenix_template"])
def test_known_protenix_defaults_to_baseline_objective_recipe(model: str | None) -> None:
    arguments = ["train.cif"]
    if model is not None:
        arguments.extend(["--model", model])

    config = canonical.build_config(canonical.parse_args(arguments))

    assert config["objective_recipe"] == "protenix_baseline_v1"


def test_known_protenix_retains_explicit_edm_objective_recipe() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            ["train.cif", "--model", "protenix", "--objective-recipe", "ofoldx_edm_v1"]
        )
    )

    assert config["objective_recipe"] == "ofoldx_edm_v1"


@pytest.mark.parametrize("model", ["ligandmpnn", "proteinmpnn"])
@pytest.mark.parametrize("recipe_arguments", [[], ["--objective-recipe", "ofoldx_edm_v1"]])
def test_known_non_protenix_omits_absent_or_edm_objective_recipe(
    model: str,
    recipe_arguments: list[str],
) -> None:
    config = canonical.build_config(
        canonical.parse_args(["train.cif", "--model", model, *recipe_arguments])
    )

    assert "objective_recipe" not in config


@pytest.mark.parametrize("model", ["ligandmpnn", "proteinmpnn"])
def test_known_non_protenix_rejects_protenix_baseline_objective_recipe(model: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(
            [
                "train.cif",
                "--model",
                model,
                "--objective-recipe",
                "protenix_baseline_v1",
            ]
        )


@pytest.mark.parametrize(
    ("recipe_arguments", "expected_recipe"),
    [
        ([], None),
        (["--objective-recipe", "ofoldx_edm_v1"], "ofoldx_edm_v1"),
        (["--objective-recipe", "protenix_baseline_v1"], "protenix_baseline_v1"),
    ],
)
def test_unresolved_pretrained_objective_recipe_remains_absent_or_explicit(
    recipe_arguments: list[str],
    expected_recipe: str | None,
) -> None:
    config = canonical.build_config(
        canonical.parse_args(
            ["train.cif", "--pretrained", "organization/model", *recipe_arguments]
        )
    )

    if expected_recipe is None:
        assert "objective_recipe" not in config
    else:
        assert config["objective_recipe"] == expected_recipe


def test_pdb_yaml_config_print_pins_historical_edm_objective_recipe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path(__file__).parents[2] / "configs" / "pdb.yaml"

    assert canonical.main(["--config", str(config_path), "--print-config"]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["model"]["name"] == "protenix"
    assert config["objective_recipe"] == "ofoldx_edm_v1"


def test_legacy_logging_config_is_exact_and_new_backends_are_absent_by_default() -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif", "--no-logging", "--log-interval", "7"]))

    assert config["logging"] == {"enabled": False, "interval": 7}
    assert "tensorboard" not in config
    assert "wandb" not in config


def test_tensorboard_default_and_explicit_config_mapping() -> None:
    default = canonical.build_config(canonical.parse_args(["train.cif", "--tensorboard"]))
    explicit = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--tensorboard",
                "--tensorboard-log-dir",
                "events/custom",
                "--tensorboard-flush-secs",
                "15",
            ]
        )
    )

    assert default["tensorboard"] == {"enabled": True}
    assert explicit["tensorboard"] == {
        "enabled": True,
        "log_dir": "events/custom",
        "flush_secs": 15,
    }


def test_wandb_full_config_mapping_preserves_tag_order() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--wandb",
                "--wandb-mode",
                "offline",
                "--wandb-project",
                "folding",
                "--wandb-entity",
                "research",
                "--wandb-id",
                "run-123",
                "--wandb-name",
                "baseline",
                "--wandb-group",
                "ablation",
                "--wandb-resume",
                "allow",
                "--wandb-tag",
                "first",
                "--wandb-tag",
                "second",
            ]
        )
    )

    assert config["wandb"] == {
        "enabled": True,
        "mode": "offline",
        "project": "folding",
        "entity": "research",
        "id": "run-123",
        "name": "baseline",
        "group": "ablation",
        "resume": "allow",
        "tags": ["first", "second"],
    }


def test_logging_backends_are_independent_and_combinable() -> None:
    config = canonical.build_config(
        canonical.parse_args(["train.cif", "--no-logging", "--tensorboard", "--wandb"])
    )

    assert config["logging"] == {"enabled": False, "interval": None}
    assert config["tensorboard"] == {"enabled": True}
    assert config["wandb"] == {"enabled": True}


@pytest.mark.parametrize(
    "backend_arguments",
    [
        ["--tensorboard-log-dir", "events"],
        ["--tensorboard-flush-secs", "10"],
        ["--no-tensorboard", "--tensorboard-log-dir", "events"],
        ["--wandb-mode", "offline"],
        ["--wandb-project", "project"],
        ["--wandb-entity", "entity"],
        ["--wandb-id", "id"],
        ["--wandb-name", "name"],
        ["--wandb-group", "group"],
        ["--wandb-resume", "auto"],
        ["--wandb-tag", "tag"],
        ["--no-wandb", "--wandb-mode", "offline"],
    ],
)
def test_backend_specific_options_require_enabled_backend(backend_arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", *backend_arguments])


@pytest.mark.parametrize(
    "backend_arguments",
    [
        ["--tensorboard", "--tensorboard-log-dir", ""],
        ["--tensorboard", "--tensorboard-log-dir", "   "],
        ["--tensorboard", "--tensorboard-flush-secs", "0"],
        ["--tensorboard", "--tensorboard-flush-secs", "-1"],
        ["--wandb", "--wandb-mode", "invalid"],
        ["--wandb", "--wandb-project", ""],
        ["--wandb", "--wandb-entity", "   "],
        ["--wandb", "--wandb-id", ""],
        ["--wandb", "--wandb-name", ""],
        ["--wandb", "--wandb-group", ""],
        ["--wandb", "--wandb-resume", "invalid"],
        ["--wandb", "--wandb-tag", "   "],
    ],
)
def test_invalid_backend_values_are_rejected(backend_arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", *backend_arguments])


def test_print_config_with_optional_backends_does_not_import_optional_logger_packages(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("tensorboard", "wandb")):
            raise AssertionError(f"unexpected optional logger import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert canonical.main(
        ["train.cif", "--tensorboard", "--wandb", "--wandb-mode", "offline", "--print-config"]
    ) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["tensorboard"] == {"enabled": True}
    assert config["wandb"] == {"enabled": True, "mode": "offline"}


def test_slurm_launcher_uses_job_name_and_job_id_log_paths() -> None:
    launcher = (Path(__file__).parents[2] / "ofoldx" / "runners" / "submit_train.slurm").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --output=/data/users/qinghanw313/AlloDesign/logs/%x-%j.out" in launcher
    assert "#SBATCH --error=/data/users/qinghanw313/AlloDesign/logs/%x-%j.err" in launcher


def test_crop_settings_are_absent_by_default_from_train_and_val() -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif", "--val-source", "val.cif"]))

    assert "max_tokens" not in config["data"]["train"]
    assert "crop_method_weights" not in config["data"]["train"]
    assert "max_tokens" not in config["data"]["val"]
    assert "crop_method_weights" not in config["data"]["val"]


def test_max_tokens_uses_continuous_crop_without_cluster_sampling() -> None:
    args = canonical.parse_args(["train.cif", "--max-tokens", "384"])
    config = canonical.build_config(args)

    assert args.crop_method_weights == [1.0, 0.0, 0.0]
    assert config["data"]["train"]["max_tokens"] == 384
    assert config["data"]["train"]["crop_method_weights"] == [1.0, 0.0, 0.0]


def test_max_tokens_uses_mixed_crop_with_cluster_sampling() -> None:
    args = canonical.parse_args(
        ["train.cif", "--max-tokens", "384", "--cluster-file", "clusters.txt"]
    )
    config = canonical.build_config(args)

    assert args.crop_method_weights == [0.2, 0.4, 0.4]
    assert config["data"]["train"]["crop_method_weights"] == [0.2, 0.4, 0.4]


def test_explicit_continuous_weights_allow_explicit_protenix_without_cluster_sampling() -> None:
    args = canonical.parse_args(
        [
            "train.cif",
            "--model",
            "protenix",
            "--max-tokens",
            "384",
            "--crop-method-weights",
            "2",
            "0",
            "0",
        ]
    )
    config = canonical.build_config(args)

    assert config["data"]["train"]["crop_method_weights"] == [2.0, 0.0, 0.0]


def test_explicit_crop_weights_map_only_to_training_split() -> None:
    args = canonical.parse_args(
        [
            "train.cif",
            "--val-source",
            "val.cif",
            "--eval-source",
            "benchmark=test.cif",
            "--max-tokens",
            "256",
            "--crop-method-weights",
            "0.5",
            "0.25",
            "0.25",
            "--cluster-file",
            "clusters.txt",
        ]
    )
    config = canonical.build_config(args)

    assert config["data"]["train"]["max_tokens"] == 256
    assert config["data"]["train"]["crop_method_weights"] == [0.5, 0.25, 0.25]
    assert config["data"]["val"] == {"source": "val.cif"}
    assert config["data"]["benchmark"] == {"source": "test.cif"}


def test_crop_size_alias_maps_to_max_tokens() -> None:
    args = canonical.parse_args(["train.cif", "--crop-size", "128"])
    config = canonical.build_config(args)

    assert args.max_tokens == 128
    assert config["data"]["train"]["max_tokens"] == 128


def test_pretrained_repository_is_allowed_when_crop_architecture_is_not_known_yet() -> None:
    args = canonical.parse_args(["train.cif", "--pretrained", "organization/model", "--max-tokens", "128"])
    config = canonical.build_config(args)

    assert config["model"]["pretrained"] == "organization/model"
    assert config["data"]["train"]["crop_method_weights"] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "argv",
    [
        ["train.cif", "--max-tokens", "0"],
        ["train.cif", "--max-tokens", "-1"],
        ["train.cif", "--crop-method-weights", "1", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "0", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "1", "-1", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "nan", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "inf", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "-inf", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "1", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "1", "0", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "0", "1", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "0", "0", "1"],
        ["train.cif", "--model", "ligandmpnn", "--max-tokens", "128"],
    ],
)
def test_invalid_crop_options_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(argv)


def test_print_config_includes_normalized_crop_settings(capsys: pytest.CaptureFixture[str]) -> None:
    assert canonical.main(["train.cif", "--max-tokens", "192", "--print-config"]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["data"]["train"]["max_tokens"] == 192
    assert config["data"]["train"]["crop_method_weights"] == [1.0, 0.0, 0.0]


def test_inline_train_dataset_mixture_builds_normalized_composite_config() -> None:
    args = canonical.parse_args(
        [
            "--train-datasets",
            '{"structures": {"source": "structures"}, '
            '"clustered": {"source": "cifs", "sampling_ratio": 2.5, '
            '"cluster_sampling": {"enabled": true, "cluster_file": "clusters.txt"}}}',
            "--train-epoch-size",
            "120",
            "--seed",
            "17",
        ]
    )

    config = canonical.build_config(args)

    assert config["train_splits"] == ["train"]
    assert config["data"]["train"] == {
        "datasets": {
            "structures": {"source": "structures", "sampling_ratio": 1},
            "clustered": {
                "source": "cifs",
                "sampling_ratio": 2.5,
                "cluster_sampling": {"enabled": True, "cluster_file": "clusters.txt"},
            },
        },
        "epoch_size": 120,
        "replacement": True,
        "seed": 17,
    }


def test_component_processing_controls_build_and_print_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    specification = json.dumps(
        {
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
    )

    built = canonical.build_config(canonical.parse_args(["--train-datasets", specification]))
    built_datasets = built["data"]["train"]["datasets"]
    assert built_datasets["bounded"]["processing_timeout_seconds"] == 12.5
    assert built_datasets["bounded"]["max_consecutive_processing_errors"] == 7
    assert built_datasets["disabled"]["processing_timeout_seconds"] is None
    assert built_datasets["disabled"]["max_consecutive_processing_errors"] is None
    assert "processing_timeout_seconds" not in built_datasets["defaults"]
    assert "max_consecutive_processing_errors" not in built_datasets["defaults"]

    assert canonical.main(["--train-datasets", specification, "--print-config"]) == 0
    printed_datasets = json.loads(capsys.readouterr().out)["data"]["train"]["datasets"]
    assert printed_datasets == built_datasets


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
def test_cli_rejects_invalid_component_processing_controls_with_precise_location(
    capsys: pytest.CaptureFixture[str],
    key: str,
    value: object,
) -> None:
    specification = json.dumps(
        {"broken_component": {"source": "train.cif", key: value}}
    )

    with pytest.raises(SystemExit) as error:
        canonical.parse_args(["--train-datasets", specification])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "broken_component" in message
    assert key in message


def test_train_dataset_mixture_can_be_loaded_from_json_file(tmp_path: Path) -> None:
    specification = tmp_path / "datasets.json"
    specification.write_text(json.dumps({"primary": {"source": "train.cif"}}), encoding="utf-8")

    config = canonical.build_config(
        canonical.parse_args(["--train-datasets", str(specification)])
    )

    assert config["data"]["train"]["datasets"] == {
        "primary": {"source": "train.cif", "sampling_ratio": 1}
    }
    assert config["data"]["train"]["epoch_size"] is None


def test_mixture_accepts_complete_cluster_sampling_schema() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "--train-datasets",
                '{"clustered": {"source": "cifs", "cluster_sampling": {'
                '"enabled": true, "cluster_file": "clusters.txt", '
                '"indices_file": "cache/indices.jsonl", "epoch_size": null, '
                '"replacement": true, "contact_cutoff": 6.5, '
                '"include_all_chain_pairs": true, "exclude_mol_types": ["ions", "water"], '
                '"seed": null, "beta": {"chain": 0.25, "interface": 2}, '
                '"alpha": {"prot": 4, "nuc": 2.5, "ligand": 0}}}}',
            ]
        )
    )

    expected_indices_file = str(Path("cache/indices.jsonl").resolve())
    assert config["data"]["train"]["datasets"]["clustered"]["cluster_sampling"] == {
        "enabled": True,
        "cluster_file": "clusters.txt",
        "indices_file": expected_indices_file,
        "epoch_size": None,
        "replacement": True,
        "contact_cutoff": 6.5,
        "include_all_chain_pairs": True,
        "exclude_mol_types": ["ions", "water"],
        "seed": None,
        "beta": {"chain": 0.25, "interface": 2},
        "alpha": {"prot": 4, "nuc": 2.5, "ligand": 0},
    }


@pytest.mark.parametrize(
    "cluster_sampling",
    [
        '{"unknown": 1}',
        '{"enabled": 1}',
        '{"enabled": true, "cluster_file": []}',
        '{"indices_file": ""}',
        '{"indices_file": 4}',
        '{"epoch_size": 10}',
        '{"seed": 10}',
        '{"replacement": false}',
        '{"replacement": 1}',
        '{"contact_cutoff": 0}',
        '{"contact_cutoff": true}',
        '{"contact_cutoff": "5"}',
        '{"include_all_chain_pairs": 1}',
        '{"exclude_mol_types": "ions"}',
        '{"exclude_mol_types": ["ions", ""]}',
        '{"beta": []}',
        '{"beta": {"unknown": 1}}',
        '{"beta": {"chain": -1}}',
        '{"beta": {"interface": true}}',
        '{"alpha": []}',
        '{"alpha": {"protein": 1}}',
        '{"alpha": {"prot": -1}}',
        '{"alpha": {"ligand": "1"}}',
    ],
)
def test_mixture_rejects_invalid_nested_cluster_sampling_schema(
    cluster_sampling: str,
) -> None:
    specification = '{"one": {"source": "x", "cluster_sampling": ' + cluster_sampling + "}}"

    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["--train-datasets", specification])


@pytest.mark.parametrize(
    "specification",
    [
        '{"one": {"source": "first"}, "one": {"source": "second"}}',
        '{"one": {"source": "first", "source": "second"}}',
        '{"one": {"source": "x", "cluster_sampling": {"enabled": false, "enabled": true}}}',
    ],
)
def test_inline_json_rejects_duplicate_object_keys_at_any_level(specification: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["--train-datasets", specification])


def test_json_file_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    specification = tmp_path / "duplicate.json"
    specification.write_text(
        '{"one": {"source": "first"}, "one": {"source": "second"}}',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["--train-datasets", str(specification)])


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["train.cif", "--train-datasets", '{"other": {"source": "other.cif"}}'],
        ["train.cif", "--train-epoch-size", "10"],
        ["--train-datasets", "{}"],
        ["--train-datasets", '{"bad.name": {"source": "x"}}'],
        ["--train-datasets", '{"one": "x"}'],
        ["--train-datasets", '{"one": {"source": ""}}'],
        ["--train-datasets", '{"one": {"source": 3}}'],
        ["--train-datasets", '{"one": {"source": "x", "sampling_ratio": 0}}'],
        ["--train-datasets", '{"one": {"source": "x", "sampling_ratio": true}}'],
        ["--train-datasets", '{"one": {"source": "x", "sampling_ratio": "1"}}'],
        ["--train-datasets", '{"one": {"source": "x", "unknown": 1}}'],
        ["--train-datasets", '{"one": {"source": "x", "seq_to_pdb_index": "index"}}'],
        ["--train-datasets", '{"one": {"source": "x", "msa_path": "msa"}}'],
        [
            "--train-datasets",
            '{"one": {"source": "x", "cluster_sampling": true}}',
        ],
        [
            "--train-datasets",
            '{"one": {"source": "x", "cluster_sampling": null}}',
        ],
        [
            "--train-datasets",
            '{"one": {"source": "x", "cluster_sampling": {"enabled": "yes"}}}',
        ],
        [
            "--train-datasets",
            '{"one": {"source": "x", "cluster_sampling": {"enabled": true}}}',
        ],
    ],
)
def test_invalid_train_dataset_mixtures_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(argv)


@pytest.mark.parametrize(
    "cluster_arguments",
    [
        ["--cluster-file", "clusters.txt"],
        ["--cluster-seed", "3"],
        ["--no-cluster-replacement"],
        ["--cluster-all-pairs"],
    ],
)
def test_mixture_rejects_legacy_cluster_controls(cluster_arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(
            [
                "--train-datasets",
                '{"one": {"source": "one.cif"}}',
                *cluster_arguments,
            ]
        )


def test_mixture_msa_inheritance_local_override_and_eval_global_pair() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "--train-datasets",
                '{"inherited": {"source": "one.cif"}, '
                '"local": {"source": "two.cif", "seq_to_pdb_index": "local-index.json", '
                '"msa_path": "local-msa"}}',
                "--seq-to-pdb-index",
                "global-index.json",
                "--msa-path",
                "global-msa",
                "--val-source",
                "val.cif",
            ]
        )
    )

    assert config["data"]["train"]["datasets"]["inherited"] == {
        "source": "one.cif",
        "sampling_ratio": 1,
        "seq_to_pdb_index": "global-index.json",
        "msa_path": "global-msa",
    }
    assert config["data"]["train"]["datasets"]["local"] == {
        "source": "two.cif",
        "sampling_ratio": 1,
        "seq_to_pdb_index": "local-index.json",
        "msa_path": "local-msa",
    }
    assert config["data"]["val"] == {
        "source": "val.cif",
        "seq_to_pdb_index": "global-index.json",
        "msa_path": "global-msa",
    }


@pytest.mark.parametrize("flag", ["--seq-to-pdb-index", "--msa-path"])
@pytest.mark.parametrize("empty_value", ["", "   ", "\t"])
def test_global_msa_paths_must_be_nonempty_before_config_generation(
    flag: str,
    empty_value: str,
) -> None:
    companion = "--msa-path" if flag == "--seq-to-pdb-index" else "--seq-to-pdb-index"

    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", flag, empty_value, companion, "valid"])


def test_mixture_print_config_uses_global_msa_pair_for_inheritance_and_all_evaluation_splits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert canonical.main(
        [
            "--train-datasets",
            '{"inherited": {"source": "train.cif"}}',
            "--seq-to-pdb-index",
            "global-index.json",
            "--msa-path",
            "global-msa",
            "--val-source",
            "val.cif",
            "--eval-source",
            "benchmark=test.cif",
            "--print-config",
        ]
    ) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["data"]["train"]["datasets"]["inherited"]["seq_to_pdb_index"] == (
        "global-index.json"
    )
    assert config["data"]["train"]["datasets"]["inherited"]["msa_path"] == "global-msa"
    for split in ("val", "benchmark"):
        assert config["data"][split]["seq_to_pdb_index"] == "global-index.json"
        assert config["data"][split]["msa_path"] == "global-msa"


def test_mixture_print_config_resolves_component_indices_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    indices_file = tmp_path / "cache" / "indices.jsonl"
    specification = json.dumps(
        {
            "clustered": {
                "source": "train.cif",
                "cluster_sampling": {
                    "enabled": True,
                    "cluster_file": "clusters.txt",
                    "indices_file": str(tmp_path / "cache" / "." / "indices.jsonl"),
                },
            }
        }
    )

    assert canonical.main(["--train-datasets", specification, "--print-config"]) == 0

    config = json.loads(capsys.readouterr().out)
    cluster_sampling = config["data"]["train"]["datasets"]["clustered"]["cluster_sampling"]
    assert cluster_sampling["indices_file"] == str(indices_file.resolve())


def test_mixture_rejects_component_indices_file_aliases(tmp_path: Path) -> None:
    indices_file = tmp_path / "cache" / "indices.jsonl"
    alias = tmp_path / "cache" / "nested" / ".." / "indices.jsonl"
    specification = json.dumps(
        {
            "first": {
                "source": "first.cif",
                "cluster_sampling": {"indices_file": str(indices_file)},
            },
            "second": {
                "source": "second.cif",
                "cluster_sampling": {"indices_file": str(alias)},
            },
        }
    )

    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["--train-datasets", specification])


def test_train_datasets_json_file_expands_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = tmp_path / "datasets.json"
    specification.write_text('{"primary": {"source": "train.cif"}}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    args = canonical.parse_args(["--train-datasets", "~/datasets.json"])

    assert args.train_datasets["primary"]["source"] == "train.cif"


def test_mixture_crop_settings_are_copied_to_every_component() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "--train-datasets",
                '{"one": {"source": "one.cif"}, "two": {"source": "two.cif"}}',
                "--max-tokens",
                "256",
            ]
        )
    )

    for component in config["data"]["train"]["datasets"].values():
        assert component["max_tokens"] == 256
        assert component["crop_method_weights"] == [1.0, 0.0, 0.0]


def test_mixture_default_crop_is_mixed_only_when_every_component_is_clustered() -> None:
    args = canonical.parse_args(
        [
            "--train-datasets",
            '{"one": {"source": "one.cif", "cluster_sampling": '
            '{"enabled": true, "cluster_file": "one.txt"}}, '
            '"two": {"source": "two.cif", "cluster_sampling": '
            '{"enabled": true, "cluster_file": "two.txt"}}}',
            "--max-tokens",
            "256",
        ]
    )

    assert args.crop_method_weights == [0.2, 0.4, 0.4]


def test_mixture_spatial_crop_requires_every_component_to_be_clustered() -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(
            [
                "--train-datasets",
                '{"clustered": {"source": "one.cif", "cluster_sampling": '
                '{"enabled": true, "cluster_file": "one.txt"}}, '
                '"plain": {"source": "two.cif"}}',
                "--max-tokens",
                "256",
                "--crop-method-weights",
                "0",
                "1",
                "0",
            ]
        )


def test_mixture_print_config_does_not_construct_runner_or_check_source_paths(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnexpectedRunner:
        def __init__(self, config: dict[str, Any]) -> None:
            raise AssertionError("Runner must not be constructed by --print-config")

    monkeypatch.setattr("ofoldx.runners.runner.Runner", UnexpectedRunner)

    assert canonical.main(
        [
            "--train-datasets",
            '{"missing": {"source": "/definitely/not/present.cif"}}',
            "--print-config",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["data"]["train"]["datasets"]["missing"]["source"] == (
        "/definitely/not/present.cif"
    )


def test_mixture_cluster_cache_diagnostics_include_component_names() -> None:
    class Component:
        def __init__(self, path: str | None, status: str = "unknown") -> None:
            self.cluster_indices_file = path
            self.cluster_indices_status = status

    class Mixture:
        datasets = {
            "clustered": Component("cache/clustered.jsonl", "created"),
            "ordinary": Component(None),
        }

    assert canonical._cluster_cache_diagnostics(Mixture()) == [
        ("clustered", "cache/clustered.jsonl", "created")
    ]


def test_multiple_named_evaluation_sources_preserve_order_and_select_first_score() -> None:
    args = canonical.parse_args(
        [
            "train.cif",
            "--eval-source",
            "validation=val.cif",
            "--eval-source",
            "benchmark=data/path=with=equals.cif",
        ]
    )

    config = canonical.build_config(args)

    assert args.eval_sources == [
        ("validation", "val.cif"),
        ("benchmark", "data/path=with=equals.cif"),
    ]
    assert config["evaluate_splits"] == ["validation", "benchmark"]
    assert list(config["data"]) == ["train", "validation", "benchmark"]
    assert config["data"]["validation"] == {"source": "val.cif"}
    assert config["data"]["benchmark"] == {"source": "data/path=with=equals.cif"}
    assert config["score"] == {"split": "validation", "metric": "loss"}


def test_val_source_coexists_with_named_evaluation_and_has_score_priority() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--eval-source",
                "benchmark=test.cif",
                "--val-source",
                "val.cif",
            ]
        )
    )

    assert config["evaluate_splits"] == ["val", "benchmark"]
    assert config["score"] == {"split": "val", "metric": "loss"}


def test_explicit_score_split_selects_one_configured_evaluation_split() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--val-source",
                "val.cif",
                "--eval-source",
                "benchmark=test.cif",
                "--score-split",
                "benchmark",
            ]
        )
    )

    assert config["score"] == {"split": "benchmark", "metric": "loss"}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--eval-source", "missing-separator"],
        ["--eval-source", "=path.cif"],
        ["--eval-source", "1invalid=path.cif"],
        ["--eval-source", "invalid.name=path.cif"],
        ["--eval-source", "train=path.cif"],
        ["--eval-source", "empty="],
        ["--eval-source", "empty=   "],
        ["--eval-source", "dup=one.cif", "--eval-source", "dup=two.cif"],
        ["--val-source", "one.cif", "--eval-source", "val=two.cif"],
        ["--score-split", "val"],
        ["--eval-source", "validation=val.cif", "--score-split", "unknown"],
    ],
)
def test_invalid_named_evaluation_options_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", *arguments])


def test_main_checks_every_evaluation_split_and_closes_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"backend": None, "closed": False, "trained": False}

    class FakeRunner:
        rank = 1

        def __init__(self, config: dict[str, Any]) -> None:
            assert config["evaluate_splits"] == ["first", "empty"]
            self.datasets = {"train": [object()], "first": [object()], "empty": []}

        def set_backend(self, backend: str) -> None:
            assert backend == "oops"
            state["backend"] = backend

        def train(self) -> None:
            state["trained"] = True

        def close(self) -> None:
            state["closed"] = True

    monkeypatch.setattr("ofoldx.runners.runner.Runner", FakeRunner)

    with pytest.raises(ValueError, match="evaluation source 'empty'.*produced an empty dataset"):
        canonical.main(
            [
                "train.cif",
                "--eval-source",
                "first=first.cif",
                "--eval-source",
                "empty=empty.cif",
                "--device",
                "cpu",
            ]
        )

    assert state == {"backend": "oops", "closed": True, "trained": False}


def test_main_distributed_error_fails_fast_without_closing_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"backend": None, "closed": 0, "fail_fast": 0}

    class FailFastExit(BaseException):
        pass

    class FakeRunner:
        distributed = True
        rank = 1

        def __init__(self, config: dict[str, Any]) -> None:
            self.datasets = {"train": [object()]}

        def set_backend(self, backend: str) -> None:
            assert backend == "oops"
            state["backend"] = backend

        def train(self) -> None:
            raise RuntimeError("loader failed")

        def close(self) -> None:
            state["closed"] += 1

    def fail_fast() -> None:
        state["fail_fast"] += 1
        raise FailFastExit

    monkeypatch.setattr("ofoldx.runners.runner.Runner", FakeRunner)
    monkeypatch.setattr(canonical, "_distributed_fail_fast", fail_fast)

    with pytest.raises(FailFastExit):
        canonical.main(["train.cif", "--device", "cpu"])

    assert state == {"backend": "oops", "closed": 0, "fail_fast": 1}


def test_distributed_fail_fast_reports_flushes_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    class ExitCalled(BaseException):
        pass

    class RecordingStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def write(self, value: str) -> int:
            events.append((self.name, value))
            return len(value)

        def flush(self) -> None:
            events.append((f"{self.name}_flush", None))

    def exit_process(status: int) -> None:
        events.append(("exit", status))
        raise ExitCalled

    monkeypatch.setattr(canonical.sys, "stdout", RecordingStream("stdout"))
    monkeypatch.setattr(canonical.sys, "stderr", RecordingStream("stderr"))
    monkeypatch.setattr(canonical.os, "_exit", exit_process)

    try:
        raise RuntimeError("loader failed")
    except RuntimeError:
        with pytest.raises(ExitCalled):
            canonical._distributed_fail_fast()

    stderr = "".join(value for name, value in events if name == "stderr" and isinstance(value, str))
    assert "RuntimeError: loader failed" in stderr
    assert events[-3:] == [("stdout_flush", None), ("stderr_flush", None), ("exit", 1)]


def test_main_success_closes_runner_once(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"backend": None, "closed": 0, "trained": 0}

    class FakeRunner:
        rank = 1

        def __init__(self, config: dict[str, Any]) -> None:
            self.datasets = {"train": [object()]}

        def set_backend(self, backend: str) -> None:
            assert backend == "oops"
            state["backend"] = backend

        def train(self) -> None:
            state["trained"] += 1

        def close(self) -> None:
            state["closed"] += 1

    monkeypatch.setattr("ofoldx.runners.runner.Runner", FakeRunner)

    assert canonical.main(["train.cif", "--device", "cpu"]) == 0
    assert state == {"backend": "oops", "closed": 1, "trained": 1}
