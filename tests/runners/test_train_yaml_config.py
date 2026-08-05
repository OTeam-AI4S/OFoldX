from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import ofoldx.runners.train as canonical


def _write_config(tmp_path: Path, contents: str, *, name: str = "train.yaml") -> Path:
    config_path = tmp_path / name
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def _parse_config(config_path: Path, *extra: str) -> Any:
    return canonical.parse_args(["--config", str(config_path), *extra])


def _assert_config_error(config_path: Path, *extra: str) -> None:
    with pytest.raises(SystemExit) as error:
        _parse_config(config_path, *extra)

    assert error.value.code == 2


def test_yaml_multidataset_config_matches_equivalent_cli(tmp_path: Path) -> None:
    datasets = {
        "afdb": {"source": "data/afdb", "sampling_ratio": 4},
        "pdb": {
            "source": "data/pdb",
            "sampling_ratio": 1.5,
            "cluster_sampling": {"enabled": True, "cluster_file": "data/clusters.txt"},
        },
    }
    config_path = _write_config(
        tmp_path,
        """train_datasets:
  afdb:
    source: data/afdb
    sampling_ratio: 4
  pdb:
    source: data/pdb
    sampling_ratio: 1.5
    cluster_sampling:
      enabled: true
      cluster_file: data/clusters.txt
train_epoch_size: 320
eval_source:
  - validation=data/validation
  - benchmark=data/benchmark
score_split: benchmark
model: protenix
model_config:
  num_blocks: 4
processor_config:
  feature_mode: compact
steps: 20
lr: 0.002
deterministic: true
max_tokens: 256
crop_method_weights: [1, 0, 0]
wandb: true
wandb_mode: offline
wandb_tag: [afdb, mixture]
""",
    )

    yaml_args = _parse_config(config_path)
    cli_args = canonical.parse_args(
        [
            "--train-datasets",
            json.dumps(datasets),
            "--train-epoch-size",
            "320",
            "--eval-source",
            "validation=data/validation",
            "--eval-source",
            "benchmark=data/benchmark",
            "--score-split",
            "benchmark",
            "--model",
            "protenix",
            "--model-config",
            '{"num_blocks": 4}',
            "--processor-config",
            '{"feature_mode": "compact"}',
            "--steps",
            "20",
            "--lr",
            "0.002",
            "--deterministic",
            "--max-tokens",
            "256",
            "--crop-method-weights",
            "1",
            "0",
            "0",
            "--wandb",
            "--wandb-mode",
            "offline",
            "--wandb-tag",
            "afdb",
            "--wandb-tag",
            "mixture",
        ]
    )

    assert yaml_args.train_datasets == cli_args.train_datasets
    assert yaml_args.model_config == {"num_blocks": 4}
    assert yaml_args.processor_config == {"feature_mode": "compact"}
    assert yaml_args.eval_source == ["validation=data/validation", "benchmark=data/benchmark"]
    assert yaml_args.eval_sources == [
        ("validation", "data/validation"),
        ("benchmark", "data/benchmark"),
    ]
    assert yaml_args.wandb_tag == ["afdb", "mixture"]
    assert yaml_args.crop_method_weights == [1.0, 0.0, 0.0]
    assert canonical.build_config(yaml_args) == canonical.build_config(cli_args)


def test_yaml_component_processing_controls_preserve_values_nulls_and_absence(
    tmp_path: Path,
) -> None:
    args = _parse_config(
        _write_config(
            tmp_path,
            """train_datasets:
  bounded:
    source: bounded.cif
    processing_timeout_seconds: 12.5
    max_consecutive_processing_errors: 7
  disabled:
    source: disabled.cif
    processing_timeout_seconds: null
    max_consecutive_processing_errors: null
  defaults:
    source: defaults.cif
""",
        )
    )

    datasets = canonical.build_config(args)["data"]["train"]["datasets"]
    assert datasets["bounded"]["processing_timeout_seconds"] == 12.5
    assert datasets["bounded"]["max_consecutive_processing_errors"] == 7
    assert "processing_timeout_seconds" in datasets["disabled"]
    assert datasets["disabled"]["processing_timeout_seconds"] is None
    assert "max_consecutive_processing_errors" in datasets["disabled"]
    assert datasets["disabled"]["max_consecutive_processing_errors"] is None
    assert "processing_timeout_seconds" not in datasets["defaults"]
    assert "max_consecutive_processing_errors" not in datasets["defaults"]


@pytest.mark.parametrize(
    ("key", "rendered_value"),
    [
        ("processing_timeout_seconds", "true"),
        ("processing_timeout_seconds", "0"),
        ("processing_timeout_seconds", "-0.5"),
        ("processing_timeout_seconds", "1e309"),
        ("processing_timeout_seconds", '"30"'),
        ("max_consecutive_processing_errors", "true"),
        ("max_consecutive_processing_errors", "0"),
        ("max_consecutive_processing_errors", "-1"),
        ("max_consecutive_processing_errors", "1e309"),
        ("max_consecutive_processing_errors", '"5"'),
        ("max_consecutive_processing_errors", "1.5"),
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
def test_yaml_rejects_invalid_component_processing_controls_with_precise_location(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    key: str,
    rendered_value: str,
) -> None:
    config_path = _write_config(
        tmp_path,
        "train_datasets:\n"
        "  broken_component:\n"
        "    source: train.cif\n"
        f"    {key}: {rendered_value}\n",
    )

    _assert_config_error(config_path)

    error = capsys.readouterr().err
    assert "broken_component" in error
    assert key in error


def test_yaml_print_config_does_not_construct_runner_or_check_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(
        tmp_path,
        """train_source: /definitely/not/present.cif
processor_config:
  ccd_revision: ccd-v1
""",
    )

    class UnexpectedRunner:
        def __init__(self, config: dict[str, Any]) -> None:
            raise AssertionError("Runner must not be constructed by --print-config")

    monkeypatch.setattr("ofoldx.runners.runner.Runner", UnexpectedRunner)

    assert canonical.main(["-c", str(config_path), "--print-config"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["data"]["train"]["source"] == "/definitely/not/present.cif"
    assert printed["processor"] == {"ccd_revision": "ccd-v1"}


@pytest.mark.parametrize("short_option", [False, True], ids=["long", "short"])
def test_missing_yaml_config_is_rejected(tmp_path: Path, short_option: bool) -> None:
    option = "-c" if short_option else "--config"

    with pytest.raises(SystemExit) as error:
        canonical.parse_args([option, str(tmp_path / "missing.yaml")])

    assert error.value.code == 2


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    _assert_config_error(_write_config(tmp_path, "train_source: [unterminated\n"))


def test_invalid_utf8_yaml_is_reported_as_a_cli_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-utf8.yaml"
    config_path.write_bytes(b"train_source: \xff\xfe\n")

    _assert_config_error(config_path)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "train.cif\n",
        "- train.cif\n",
    ],
    ids=["empty", "scalar", "sequence"],
)
def test_yaml_root_must_be_a_mapping(tmp_path: Path, contents: str) -> None:
    _assert_config_error(_write_config(tmp_path, contents))


@pytest.mark.parametrize(
    "unknown_key",
    ["unknown_option", "crop_size", "cluster_contact_cutoff", "print_config", "config"],
)
def test_yaml_rejects_unknown_and_noncanonical_top_level_keys(
    tmp_path: Path,
    unknown_key: str,
) -> None:
    _assert_config_error(
        _write_config(tmp_path, f"train_source: train.cif\n{unknown_key}: value\n")
    )


@pytest.mark.parametrize(
    ("case", "contents"),
    [
        ("duplicate", "train_source: first.cif\ntrain_source: second.cif\n"),
        (
            "nested-duplicate",
            "train_source: train.cif\nmodel_config:\n  width: 32\n  width: 64\n",
        ),
        (
            "alias",
            "train_source: &source train.cif\nval_source: *source\n",
        ),
        ("custom-tag", "train_source: !env HOME\n"),
        (
            "multiple-documents",
            "train_source: first.cif\n---\ntrain_source: second.cif\n",
        ),
    ],
)
def test_yaml_loader_rejects_ambiguous_or_unsafe_constructs(
    tmp_path: Path,
    case: str,
    contents: str,
) -> None:
    del case
    _assert_config_error(_write_config(tmp_path, contents))


@pytest.mark.parametrize(
    "contents",
    [
        "train_source: train.cif\n1: value\n",
        "train_source: train.cif\nmodel_config:\n  1: value\n",
        "train_datasets:\n  1:\n    source: train.cif\n",
    ],
    ids=["root", "open-mapping", "nested-mapping"],
)
def test_yaml_mapping_keys_must_be_strings(tmp_path: Path, contents: str) -> None:
    _assert_config_error(_write_config(tmp_path, contents))


@pytest.mark.parametrize(
    "contents",
    [
        "train_datasets: '{}'\n",
        "train_source: train.cif\nmodel_config: '{}'\n",
        "train_source: train.cif\nprocessor_config: configs/processor.json\n",
    ],
    ids=["train-datasets", "model-config", "processor-config"],
)
def test_yaml_mapping_options_require_native_mappings(tmp_path: Path, contents: str) -> None:
    _assert_config_error(_write_config(tmp_path, contents))


def test_yaml_exponent_float_matches_equivalent_cli(tmp_path: Path) -> None:
    yaml_args = _parse_config(
        _write_config(tmp_path, "train_source: train.cif\nlr: 1e-3\n")
    )
    cli_args = canonical.parse_args(["train.cif", "--lr", "1e-3"])

    assert yaml_args.lr == pytest.approx(0.001)
    assert canonical.build_config(yaml_args) == canonical.build_config(cli_args)


@pytest.mark.parametrize(
    "scalar",
    ["010", "1:30"],
    ids=["leading-zero", "sexagesimal"],
)
def test_yaml_legacy_numeric_spellings_cannot_change_values(
    tmp_path: Path,
    scalar: str,
) -> None:
    _assert_config_error(
        _write_config(tmp_path, f"train_source: train.cif\nseed: {scalar}\n")
    )


def test_yaml_timestamp_like_plain_scalar_remains_a_string(tmp_path: Path) -> None:
    args = _parse_config(_write_config(tmp_path, "train_source: 2026-08-03\n"))

    assert args.train_source == "2026-08-03"
    assert isinstance(args.train_source, str)


def test_yaml_timestamp_like_scalar_still_follows_target_option_type(tmp_path: Path) -> None:
    _assert_config_error(
        _write_config(tmp_path, "train_source: train.cif\nseed: 2026-08-03\n")
    )


@pytest.mark.parametrize(
    "fragment",
    [
        "deterministic: !!int 1\n",
        "batch_size: !!float 2\n",
        "train_source: !!timestamp 2026-08-03\n",
    ],
    ids=["int", "float", "timestamp"],
)
def test_explicit_yaml_scalar_tags_cannot_bypass_schema(
    tmp_path: Path,
    fragment: str,
) -> None:
    prefix = "" if fragment.startswith("train_source:") else "train_source: train.cif\n"
    _assert_config_error(_write_config(tmp_path, prefix + fragment))


@pytest.mark.parametrize(
    "contents",
    [
        "!!map {train_source: train.cif}\n",
        "!!seq [train.cif]\n",
        "train_source: train.cif\nmodel_config: !!map {width: 64}\n",
        "train_source: train.cif\nmodel_config: {widths: !!seq [32, 64]}\n",
    ],
    ids=["root-map", "root-sequence", "nested-map", "nested-sequence"],
)
def test_explicit_yaml_collection_tags_are_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str,
) -> None:
    _assert_config_error(_write_config(tmp_path, contents))

    assert "explicit YAML tags are not allowed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "contents",
    [
        "<<:\n  train_source: train.cif\n",
        "train_source: train.cif\nmodel_config:\n  <<:\n    width: 64\n",
    ],
    ids=["top-level", "nested"],
)
def test_yaml_merge_keys_are_rejected(tmp_path: Path, contents: str) -> None:
    _assert_config_error(_write_config(tmp_path, contents))


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("boolean-integer", "deterministic: 1\n"),
        ("boolean-string", 'deterministic: "true"\n'),
        ("yaml-1.1-boolean", "deterministic: yes\n"),
        ("integer-boolean", "batch_size: true\n"),
        ("integer-float", "batch_size: 1.5\n"),
        ("integer-string", 'batch_size: "2"\n'),
        ("nonpositive-integer", "batch_size: 0\n"),
        ("float-boolean", "lr: true\n"),
        ("float-string", 'lr: "0.1"\n'),
        ("nonfinite-float", "lr: .inf\n"),
        ("nonpositive-float", "lr: 0\n"),
        ("invalid-choice", "device: gpu\n"),
        ("mapping-as-list", "model_config: []\n"),
        ("append-list-as-string", "wandb: true\nwandb_tag: tag\n"),
        ("append-list-item-type", "wandb: true\nwandb_tag: [valid, 2]\n"),
        ("fixed-nargs-length", "max_tokens: 64\ncrop_method_weights: [1, 0]\n"),
        ("fixed-nargs-item-type", 'max_tokens: 64\ncrop_method_weights: [1, "0", 0]\n'),
    ],
)
def test_yaml_values_follow_strict_argparse_types_and_choices(
    tmp_path: Path,
    name: str,
    fragment: str,
) -> None:
    del name
    _assert_config_error(_write_config(tmp_path, "train_source: train.cif\n" + fragment))


@pytest.mark.parametrize(
    "fragment",
    [
        "lr: null\n",
        "deterministic: null\n",
        "eval_source: null\n",
        "model_config: null\n",
        "wandb_tag: null\n",
        "scheduler: null\n",
    ],
    ids=["float", "boolean", "append-list", "mapping", "tag-list", "choice"],
)
def test_yaml_null_is_rejected_for_nonnullable_options(tmp_path: Path, fragment: str) -> None:
    _assert_config_error(_write_config(tmp_path, "train_source: train.cif\n" + fragment))


def test_yaml_null_means_omitted_for_nullable_options(tmp_path: Path) -> None:
    args = _parse_config(
        _write_config(
            tmp_path,
            """train_source: train.cif
steps: null
precision: null
max_tokens: null
max_grad_norm: null
val_source: null
run_name: null
tensorboard_flush_secs: null
""",
        )
    )

    assert args.steps is None
    assert args.epochs == 1
    assert args.precision is None
    assert args.max_tokens is None
    assert args.max_grad_norm is None
    assert args.val_source is None
    assert args.run_name is None
    assert args.tensorboard_flush_secs is None


@pytest.mark.parametrize(
    "contents",
    [
        "train_source: ${DATA_ROOT}/train.cif\n",
        "train_source: train.cif\nmodel_config:\n  nested:\n    value: ${MODEL_WIDTH}\n",
    ],
    ids=["top-level", "nested"],
)
def test_yaml_interpolation_markers_are_rejected(tmp_path: Path, contents: str) -> None:
    _assert_config_error(_write_config(tmp_path, contents))


@pytest.mark.parametrize("excluded", [[], ["ions", "water"]])
def test_yaml_zero_or_more_nargs_list_is_preserved(
    tmp_path: Path,
    excluded: list[str],
) -> None:
    rendered = json.dumps(excluded)
    args = _parse_config(
        _write_config(
            tmp_path,
            "train_source: train.cif\n"
            "cluster_file: clusters.txt\n"
            f"cluster_exclude_mol_types: {rendered}\n",
        )
    )

    assert args.cluster_exclude_mol_types == excluded
    cluster_sampling = canonical.build_config(args)["data"]["train"]["cluster_sampling"]
    assert cluster_sampling["exclude_mol_types"] == excluded


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        (
            "training-source-alternative",
            "train_source: train.cif\ntrain_datasets:\n  other:\n    source: other.cif\n",
        ),
        ("training-boundary", "train_source: train.cif\nepochs: 2\nsteps: 10\n"),
        (
            "ccd-locator",
            "train_source: train.cif\nccd_revision: one\nccd_snapshot_path: two\n",
        ),
        (
            "resume-source",
            "train_source: train.cif\nresume: true\nresume_from: checkpoint.pt\n",
        ),
        (
            "msa-pair",
            "train_source: train.cif\nseq_to_pdb_index: seq-to-pdb.json\n",
        ),
        (
            "pretrained-model-config",
            "train_source: train.cif\npretrained: org/model\nmodel_config:\n  width: 64\n",
        ),
        (
            "backend-gating",
            "train_source: train.cif\ntensorboard_log_dir: events\n",
        ),
        (
            "crop-coupling",
            "train_source: train.cif\ncrop_method_weights: [1, 0, 0]\n",
        ),
    ],
)
def test_yaml_preserves_mutual_exclusions_and_coupled_constraints(
    tmp_path: Path,
    name: str,
    contents: str,
) -> None:
    del name
    _assert_config_error(_write_config(tmp_path, contents))


@pytest.mark.parametrize(
    "extra",
    [
        ["other.cif"],
        ["--steps", "2"],
        ["--no-checkpoint"],
        ["--eval-source", "validation=val.cif"],
    ],
    ids=["positional", "scalar", "boolean", "append"],
)
def test_yaml_config_rejects_mixed_legacy_cli_arguments(
    tmp_path: Path,
    extra: list[str],
) -> None:
    _assert_config_error(_write_config(tmp_path, "train_source: train.cif\n"), *extra)


@pytest.mark.parametrize(
    "selectors",
    [
        ["--config", "{path}", "--config", "{path}"],
        ["-c", "{path}", "--config", "{path}"],
        ["--config={path}", "-c", "{path}"],
    ],
)
def test_multiple_config_selectors_are_rejected(
    tmp_path: Path,
    selectors: list[str],
) -> None:
    config_path = _write_config(tmp_path, "train_source: train.cif\n")
    argv = [selector.format(path=config_path) for selector in selectors]

    with pytest.raises(SystemExit) as error:
        canonical.parse_args(argv)

    assert error.value.code == 2


@pytest.mark.parametrize("with_positional", [False, True], ids=["alone", "mixed-positional"])
def test_abbreviated_long_config_option_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    with_positional: bool,
) -> None:
    config_path = _write_config(tmp_path, "train_source: from-config.cif\n")
    argv = (["legacy.cif"] if with_positional else []) + ["--conf", str(config_path)]

    with pytest.raises(SystemExit) as error:
        canonical.parse_args(argv)

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr
    assert "--conf" in stderr


def test_config_path_expands_user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path, "train_source: train.cif\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    args = canonical.parse_args(["--config", f"~/{config_path.name}"])

    assert args.train_source == "train.cif"


def test_unresolvable_config_user_home_is_reported_as_a_cli_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved_config = "~missing-ofoldx-yaml-user/train.yaml"
    original_expanduser = Path.expanduser

    def fail_for_missing_user(path: Path) -> Path:
        if str(path) == unresolved_config:
            raise RuntimeError("synthetic missing user")
        return original_expanduser(path)

    monkeypatch.setattr(Path, "expanduser", fail_for_missing_user)

    with pytest.raises(SystemExit) as error:
        canonical.parse_args(["--config", unresolved_config])

    assert error.value.code == 2


def test_attached_short_config_option_is_supported(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "train_source: train.cif\n")

    args = canonical.parse_args([f"-c{config_path}"])

    assert args.train_source == "train.cif"


def test_double_dash_shields_config_spelling_as_legacy_positional() -> None:
    args = canonical.parse_args(["--", "--config"])

    assert args.train_source == "--config"


def test_double_dash_does_not_turn_shielded_print_config_into_a_config_control(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path, "train_source: train.cif\n")

    with pytest.raises(SystemExit) as error:
        canonical.parse_args(["--config", str(config_path), "--", "--print-config"])

    assert error.value.code == 2


@pytest.mark.parametrize("help_option", ["-h", "--help"])
def test_help_is_an_allowed_config_mode_control(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    help_option: str,
) -> None:
    config_path = _write_config(tmp_path, "train_source: train.cif\n")

    with pytest.raises(SystemExit) as error:
        canonical.parse_args([f"-c{config_path}", help_option])

    assert error.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_legacy_cli_without_config_remains_supported() -> None:
    args = canonical.parse_args(["train.cif", "--steps", "3", "--no-checkpoint"])

    assert args.train_source == "train.cif"
    assert args.steps == 3
    assert args.checkpoint is False
