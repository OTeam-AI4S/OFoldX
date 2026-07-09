from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME
from ofoldx.models.alphafold2.convert import (
    convert_alphafold2_checkpoint,
    convert_official_alphafold2_state_dict,
    convert_openfold_alphafold2_state_dict,
    inspect_official_alphafold2_checkpoint,
    inspect_official_alphafold2_mapping,
    load_official_alphafold2_params,
    main,
    normalize_official_param_key,
    normalize_openfold_state_key,
    official_param_filename,
    unflatten_official_params,
)


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "alphafold/alphafold_iteration/evoformer/preprocess_1d//weights": np.arange(6, dtype=np.float32).reshape(2, 3),
        "alphafold/alphafold_iteration/evoformer/pair_activiations//weights": np.arange(10, dtype=np.float32).reshape(
            5, 2
        ),
        "alphafold/alphafold_iteration/structure_module/layer_norm//scale": np.ones(4, dtype=np.float32),
        "alphafold/alphafold_iteration/structure_module/layer_norm//offset": np.zeros(4, dtype=np.float32),
    }
    np.savez(path, **arrays)


def test_official_param_filename_validates_model_names() -> None:
    assert official_param_filename("model_1_ptm") == "params_model_1_ptm.npz"

    with pytest.raises(ValueError, match="unsupported AlphaFold 2 model name"):
        official_param_filename("not_af2")


def test_normalize_official_param_key_maps_haiku_names_to_current_tree() -> None:
    assert (
        normalize_official_param_key("alphafold/alphafold_iteration/evoformer/preprocess_1d//weights")
        == "input_embedding.preprocess_1d.weight"
    )
    assert (
        normalize_official_param_key("alphafold/alphafold_iteration/evoformer/pair_activiations//weights")
        == "input_embedding.relpos_projection.weight"
    )
    assert (
        normalize_official_param_key("alphafold/alphafold_iteration/evoformer/single_activations//weights")
        == "single_activations.weight"
    )
    assert (
        normalize_official_param_key(
            "alphafold/alphafold_iteration/evoformer/evoformer_iteration/__layer_stack_no_state/"
            "msa_transition/layer_norm//scale"
        )
        == "evoformer.blocks.stacked.msa_stack.feed_forward.layer_norm.weight"
    )
    assert (
        normalize_official_param_key(
            "alphafold/alphafold_iteration/evoformer/extra_msa_stack/__layer_stack_no_state/"
            "msa_row_attention_with_pair_bias/attention//query_w"
        )
        == "extra_msa_stack.blocks.stacked.msa_stack.msa_row_attention_with_pair_bias."
        "attention.projection.query_projection.weight"
    )
    assert (
        normalize_official_param_key("alphafold/alphafold_iteration/distogram_head/half_logits//bias")
        == "distogram_head.linear.bias"
    )
    assert (
        normalize_official_param_key(
            "alphafold/alphafold_iteration/structure_module/fold_iteration/"
            "invariant_point_attention/q_scalar//weights"
        )
        == "structure_module.fold_iteration.invariant_point_attention.q_scalar.weight"
    )
    assert (
        normalize_official_param_key(
            "alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//weights"
        )
        == "structure_module.fold_iteration.transition.layers.1.weight"
    )
    assert (
        normalize_official_param_key(
            "alphafold/alphafold_iteration/structure_module/fold_iteration/"
            "rigid_sidechain/input_projection_1//weights"
        )
        == "structure_module.fold_iteration.sidechain.input_projection.1.weight"
    )
    assert (
        normalize_official_param_key(
            "alphafold/alphafold_iteration/structure_module/fold_iteration/" "rigid_sidechain/resblock2_1//bias"
        )
        == "structure_module.fold_iteration.sidechain.resblocks.1.resblock2.bias"
    )


def test_convert_official_state_dict_transposes_linear_weights_and_casts() -> None:
    source = {
        "alphafold/alphafold_iteration/evoformer/preprocess_1d//weights": torch.arange(6).reshape(2, 3).float(),
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/__layer_stack_no_state/"
        "msa_row_attention_with_pair_bias/attention//query_w": torch.arange(24).reshape(2, 3, 4).float(),
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/__layer_stack_no_state/"
        "outer_product_mean//output_w": torch.arange(60).reshape(3, 4, 5).float(),
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/__layer_stack_no_state/"
        "triangle_attention_starting_node//feat_2d_weights": torch.arange(16).reshape(8, 2).float(),
        "alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//weights": (
            torch.arange(8).reshape(2, 4).float()
        ),
        "alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/"
        "input_projection_1//bias": torch.zeros(3),
    }

    converted = convert_official_alphafold2_state_dict(source, dtype=torch.float16)

    assert sorted(converted) == [
        "evoformer.blocks.stacked.msa_stack.msa_row_attention_with_pair_bias."
        "attention.projection.query_projection.weight",
        "evoformer.blocks.stacked.outer_product_mean.output_projection.weight",
        "evoformer.blocks.stacked.pair_stack.triangle_attention_starting_node.feat_2d_weights.weight",
        "input_embedding.preprocess_1d.weight",
        "structure_module.fold_iteration.sidechain.input_projection.1.bias",
        "structure_module.fold_iteration.transition.layers.1.weight",
    ]
    assert converted["input_embedding.preprocess_1d.weight"].shape == (3, 2)
    assert converted["input_embedding.preprocess_1d.weight"].dtype is torch.float16
    torch.testing.assert_close(
        converted["input_embedding.preprocess_1d.weight"],
        torch.arange(6).reshape(2, 3).t().half(),
    )
    assert converted[
        "evoformer.blocks.stacked.msa_stack.msa_row_attention_with_pair_bias."
        "attention.projection.query_projection.weight"
    ].shape == (12, 2)
    assert converted["evoformer.blocks.stacked.outer_product_mean.output_projection.weight"].shape == (5, 12)
    assert converted[
        "evoformer.blocks.stacked.pair_stack.triangle_attention_starting_node.feat_2d_weights.weight"
    ].shape == (2, 8)
    assert converted["structure_module.fold_iteration.transition.layers.1.weight"].shape == (4, 2)
    assert converted["structure_module.fold_iteration.sidechain.input_projection.1.bias"].shape == (3,)


def test_normalize_openfold_state_key_maps_pytorch_names_to_current_tree() -> None:
    assert normalize_openfold_state_key("input_embedder.linear_tf_m.weight") == "input_embedding.preprocess_1d.weight"
    assert (
        normalize_openfold_state_key("evoformer.blocks.0.msa_att_row.mha.linear_q.weight")
        == "evoformer.blocks.0.msa_stack.msa_row_attention_with_pair_bias."
        "attention.projection.query_projection.weight"
    )
    assert (
        normalize_openfold_state_key("extra_msa_stack.blocks.1.msa_att_col.global_attention.linear_k.weight")
        == "extra_msa_stack.blocks.1.msa_stack.msa_column_global_attention."
        "attention.projection.key_projection.weight"
    )
    assert (
        normalize_openfold_state_key("template_pair_stack.blocks.1.tri_mul_out.linear_a_p.bias")
        == "template_embedder.single_template_embedding.template_pair_stack.blocks.1.pair_stack."
        "triangle_multiplication_outgoing.left_projection.bias"
    )
    assert (
        normalize_openfold_state_key("structure_module.ipa.linear_q.weight")
        == "structure_module.fold_iteration.invariant_point_attention.q_scalar.weight"
    )
    assert (
        normalize_openfold_state_key("structure_module.transition.layers.0.linear_3.bias")
        == "structure_module.fold_iteration.transition.layers.2.bias"
    )
    assert (
        normalize_openfold_state_key("structure_module.angle_resnet.layers.1.linear_2.weight")
        == "structure_module.fold_iteration.sidechain.resblocks.1.resblock2.weight"
    )
    assert normalize_openfold_state_key("aux_heads.tm.linear.bias") == "predicted_aligned_error_head.logits.bias"


def test_convert_openfold_state_dict_preserves_pytorch_layout_and_casts() -> None:
    source = {
        "input_embedder.linear_tf_m.weight": torch.arange(6).reshape(3, 2).float(),
        "evoformer.blocks.0.core.outer_product_mean.linear_out.weight": torch.arange(12).reshape(3, 4).float(),
        "template_pointwise_att.mha.linear_o.bias": torch.ones(3),
    }

    converted = convert_openfold_alphafold2_state_dict(source, dtype=torch.float16)

    assert sorted(converted) == [
        "evoformer.blocks.0.outer_product_mean.output_projection.weight",
        "input_embedding.preprocess_1d.weight",
        "template_embedder.attention.projection.output_projection.bias",
    ]
    torch.testing.assert_close(
        converted["input_embedding.preprocess_1d.weight"],
        torch.arange(6).reshape(3, 2).half(),
    )
    assert converted["input_embedding.preprocess_1d.weight"].dtype is torch.float16


def test_inspect_openfold_checkpoint_infers_template_and_ptm_heads(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "finetuning_no_templ_ptm_1.pt"
    torch.save(
        {
            "state_dict": {
                "input_embedder.linear_tf_m.weight": torch.zeros(2, 2),
                "aux_heads.tm.linear.weight": torch.zeros(2, 2),
            }
        },
        checkpoint_path,
    )

    report = convert_alphafold2_checkpoint(
        checkpoint_path,
        tmp_path / "out",
        source="openfold",
        inspect_only=True,
    )

    assert report["model_name"] == "finetuning_no_templ_ptm_1"
    assert report["selected_state_key"] == "state_dict"
    assert report["template_enabled"] is False
    assert report["ptm_head_enabled"] is True


def test_convert_official_state_dict_splits_real_stacked_evoformer_tensors() -> None:
    source = {
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
        "msa_row_attention_with_pair_bias/attention//query_w": torch.arange(48).reshape(2, 3, 2, 4).float(),
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
        "msa_row_attention_with_pair_bias/attention//gating_b": torch.arange(16).reshape(2, 2, 4).float(),
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
        "msa_transition/transition1//weights": torch.arange(24).reshape(2, 3, 4).float(),
    }

    converted = convert_official_alphafold2_state_dict(source)

    assert sorted(converted) == [
        "evoformer.blocks.0.msa_stack.feed_forward.transition1.weight",
        "evoformer.blocks.0.msa_stack.msa_row_attention_with_pair_bias.attention.projection.gate_projection.bias",
        "evoformer.blocks.0.msa_stack.msa_row_attention_with_pair_bias.attention.projection.query_projection.weight",
        "evoformer.blocks.1.msa_stack.feed_forward.transition1.weight",
        "evoformer.blocks.1.msa_stack.msa_row_attention_with_pair_bias.attention.projection.gate_projection.bias",
        "evoformer.blocks.1.msa_stack.msa_row_attention_with_pair_bias.attention.projection.query_projection.weight",
    ]
    assert converted[
        "evoformer.blocks.0.msa_stack.msa_row_attention_with_pair_bias." "attention.projection.query_projection.weight"
    ].shape == (8, 3)
    assert converted[
        "evoformer.blocks.1.msa_stack.msa_row_attention_with_pair_bias." "attention.projection.gate_projection.bias"
    ].shape == (8,)
    assert converted["evoformer.blocks.0.msa_stack.feed_forward.transition1.weight"].shape == (4, 3)


def test_convert_official_state_dict_splits_template_pair_stack_tensors() -> None:
    source = {
        "alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/"
        "template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//query_w": (
            torch.arange(32).reshape(2, 4, 2, 2).float()
        ),
        "alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/"
        "template_pair_stack/__layer_stack_no_state/pair_transition/transition1//weights": (
            torch.arange(48).reshape(2, 4, 6).float()
        ),
    }

    converted = convert_official_alphafold2_state_dict(source)

    assert sorted(converted) == [
        "template_embedder.single_template_embedding.template_pair_stack.blocks.0.pair_stack."
        "feed_forward.transition1.weight",
        "template_embedder.single_template_embedding.template_pair_stack.blocks.0.pair_stack."
        "triangle_attention_starting_node.attention.projection.query_projection.weight",
        "template_embedder.single_template_embedding.template_pair_stack.blocks.1.pair_stack."
        "feed_forward.transition1.weight",
        "template_embedder.single_template_embedding.template_pair_stack.blocks.1.pair_stack."
        "triangle_attention_starting_node.attention.projection.query_projection.weight",
    ]
    assert converted[
        "template_embedder.single_template_embedding.template_pair_stack.blocks.0.pair_stack."
        "triangle_attention_starting_node.attention.projection.query_projection.weight"
    ].shape == (4, 4)
    assert converted[
        "template_embedder.single_template_embedding.template_pair_stack.blocks.1.pair_stack."
        "feed_forward.transition1.weight"
    ].shape == (6, 4)


def test_inspect_official_mapping_reports_strict_load_ready() -> None:
    source = {
        "alphafold/alphafold_iteration/evoformer/preprocess_1d//weights": torch.zeros(2, 3),
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/__layer_stack_no_state/"
        "msa_transition/layer_norm//scale": torch.zeros(4),
        "alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/"
        "embedding2d//weights": torch.zeros(2, 3),
        "alphafold/alphafold_iteration/structure_module/fold_iteration/attention_layer_norm//scale": torch.zeros(4),
        "alphafold/alphafold_iteration/distogram_head/half_logits//bias": torch.zeros(64),
    }

    summary = inspect_official_alphafold2_mapping(source)

    assert summary["strict_load_ready"] is True
    assert summary["target_group_counts"] == {
        "evoformer": 1,
        "heads": 1,
        "input_embedding": 1,
        "structure_module": 1,
        "template": 1,
    }
    assert summary["strict_load_status_counts"]["strict_load_ready"] == 5
    assert summary["blocked_group_counts"] == {}
    assert summary["blocker_counts"] == {}


def test_unflatten_official_params_matches_haiku_scope_split() -> None:
    tensor = torch.ones(1)
    nested = unflatten_official_params({"scope/path//weights": tensor})

    assert nested == {"scope/path": {"weights": tensor}}


def test_load_official_params_from_download_root_and_tar(tmp_path: Path) -> None:
    npz_path = tmp_path / "download" / "params" / "params_model_1.npz"
    _write_npz(npz_path)

    loaded = load_official_alphafold2_params(tmp_path / "download", model_name="model_1")

    assert "alphafold/alphafold_iteration/evoformer/preprocess_1d//weights" in loaded
    tar_path = tmp_path / "alphafold_params_2022-12-06.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(npz_path, arcname="params_model_1.npz")

    tar_loaded = load_official_alphafold2_params(tar_path)

    torch.testing.assert_close(
        tar_loaded["alphafold/alphafold_iteration/evoformer/preprocess_1d//weights"],
        torch.arange(6).reshape(2, 3).float(),
    )


def test_inspect_only_does_not_write_bundle(tmp_path: Path) -> None:
    npz_path = tmp_path / "params_model_1.npz"
    _write_npz(npz_path)

    report = inspect_official_alphafold2_checkpoint(npz_path)

    assert report["model_name"] == "model_1"
    assert report["num_tensors"] == 4
    assert not (tmp_path / "out").exists()


def test_convert_checkpoint_rejects_partial_runnable_model_artifacts(tmp_path: Path) -> None:
    npz_path = tmp_path / "params" / "params_model_1.npz"
    _write_npz(npz_path)

    with pytest.raises(ValueError, match="refusing to write a partial runnable model artifact"):
        convert_alphafold2_checkpoint(tmp_path / "params", tmp_path / "out", dtype=torch.float16)

    assert not (tmp_path / "out" / ARTIFACT_MANIFEST_NAME).exists()


def test_converter_cli_rejects_inspect_only_push_to_hub(tmp_path: Path) -> None:
    npz_path = tmp_path / "params_model_1.npz"
    _write_npz(npz_path)

    with pytest.raises(SystemExit, match="cannot be combined"):
        main(
            [
                str(npz_path),
                str(tmp_path / "out"),
                "--inspect-only",
                "--push-to-hub",
                "--repo-id",
                "oteam/alphafold2",
            ]
        )
