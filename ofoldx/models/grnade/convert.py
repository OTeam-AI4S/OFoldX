r"""Convert eleven released gRNAde checkpoints or the separate ODesign artifact."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor

from .config import GrnadeConfig, GrnadeV032ProcessorConfig
from .model import GrnadeModel

GrnadeArtifact: TypeAlias = Literal[
    "odesign-grnade-v1",
    "gRNAde_drop3d@0.75_maxlen@500",
    "gRNAde_ARv1_1state_all",
    "gRNAde_ARv1_1state_das",
    "gRNAde_ARv1_1state_multi",
    "gRNAde_ARv1_2state_all",
    "gRNAde_ARv1_3state_all",
    "gRNAde_ARv1_3state_das",
    "gRNAde_ARv1_3state_multi",
    "gRNAde_ARv1_5state_all",
    "gRNAde_ARv1_5state_das",
    "gRNAde_ARv1_5state_multi",
]

_ARTIFACTS: dict[GrnadeArtifact, int | None] = {
    "odesign-grnade-v1": None,
    "gRNAde_drop3d@0.75_maxlen@500": None,
    "gRNAde_ARv1_1state_all": 1,
    "gRNAde_ARv1_1state_das": 1,
    "gRNAde_ARv1_1state_multi": 1,
    "gRNAde_ARv1_2state_all": 2,
    "gRNAde_ARv1_3state_all": 3,
    "gRNAde_ARv1_3state_das": 3,
    "gRNAde_ARv1_3state_multi": 3,
    "gRNAde_ARv1_5state_all": 5,
    "gRNAde_ARv1_5state_das": 5,
    "gRNAde_ARv1_5state_multi": 5,
}

_PLACEHOLDER_KEYS = frozenset(
    {
        "W_v.1.dummy_param",
        "W_e.1.dummy_param",
        "W_out.dummy_param",
    }
).union(
    {
        f"{family}.{block}.{path}.dummy_param"
        for family in ("encoder_layers", "decoder_layers")
        for block in range(4)
        for path in (
            "conv.message_func.0",
            "conv.message_func.1",
            "conv.message_func.2",
            "dropout.0.vdropout",
            "dropout.1.vdropout",
            "ff_func.0",
            "ff_func.1",
        )
    }
)


def convert_grnade_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    artifact: GrnadeArtifact,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Strict-load one explicitly identified released artifact and save its pair."""

    from .pipeline import GrnadePipeline, GrnadeV032Pipeline
    from .processor import GrnadeProcessor, GrnadeV032Processor

    state_dict = cast(
        Mapping[str, Tensor],
        torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        ),
    )
    conformation_count = _ARTIFACTS[artifact]
    model = GrnadeModel(
        GrnadeConfig(
            edge_scalar_size=132 if conformation_count is None else 131,
            num_token_embeddings=5 if conformation_count is None else 4,
        )
    )
    model.load_state_dict(
        convert_grnade_state_dict(state_dict),
        strict=True,
        assign=True,
    )
    model.eval()
    if conformation_count is None:
        GrnadePipeline(model=model, processor=GrnadeProcessor()).save(
            output_dir,
            repo_id=repo_id,
            token=token,
            private=private,
        )
    else:
        GrnadeV032Pipeline(
            model=model,
            processor=GrnadeV032Processor(GrnadeV032ProcessorConfig(max_num_conformations=conformation_count)),
        ).save(
            output_dir,
            repo_id=repo_id,
            token=token,
            private=private,
        )


def convert_grnade_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    r"""Translate every live released tensor to the native gRNAde layout."""

    return {
        _convert_key(source_key): value
        for source_key, value in state_dict.items()
        if source_key not in _PLACEHOLDER_KEYS
    }


def _convert_key(key: str) -> str:
    top_level = {
        "W_v.": "node_embedding.",
        "W_e.": "edge_embedding.",
        "encoder_layers.": "encoder_blocks.",
        "W_s.": "token_embedding.",
        "decoder_layers.": "decoder_blocks.",
        "W_out.": "output_projection.",
    }
    for source, target in top_level.items():
        if key.startswith(source):
            key = target + key.removeprefix(source)
            break
    else:
        raise KeyError(f"unrecognized gRNAde checkpoint tensor: {key}")

    for source, target in (
        (".conv.message_func.", ".message_passing.message."),
        (".norm.", ".norms."),
        (".ff_func.", ".transition."),
        (".wh.", ".hidden_vector_projection."),
        (".ws.", ".scalar_projection."),
        (".wv.", ".output_vector_projection."),
        (".wsv.", ".vector_gate_projection."),
    ):
        key = key.replace(source, target)
    return key


def main(argv: list[str] | None = None) -> None:
    r"""Convert one released gRNAde checkpoint from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", choices=tuple(_ARTIFACTS))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)
    convert_grnade_checkpoint(
        args.checkpoint,
        args.output_dir,
        artifact=args.artifact,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
