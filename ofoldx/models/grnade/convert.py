# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
gRNAde checkpoint conversion.

Converts eleven released gRNAde checkpoints or the separate ODesign artifact.

Notes:
    The implementation follows gRNAde v1.0.0 revision
    ``2453b18778a1981ac25f314790e5395f1e9ab218`` and v0.3.2 revision
    ``e473636bef037be1e1b2516cf00af71e3334bd4f``. Explicit artifact IDs distinguish the independent v1 checkpoint,
    the ODesign ``grnade.h5`` artifact, and ten v0.3.2 autoregressive checkpoints. Each v0.3.2 pair retains its
    released one-, two-, three-, or five-conformation Processor setting; split names do not alter the model
    equation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor

from ...conversion import save_repository
from ...errors import RequestRef, fail
from ..vocabulary import canonical_word_embeddings
from .config import GrnadeConfig, GrnadeV032ProcessorConfig
from .model import GRNADE_INPUT_VOCABULARY, GRNADE_VOCABULARY, GrnadeModel
from .pipeline import GrnadePipeline, GrnadeV032Pipeline
from .processor import GrnadeProcessor, GrnadeV032Processor

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

# Both released autoregressive graphs bypass the first message normalization.
# Its affine tensors are serialized by the common GVPConvLayer constructor but unused.
_UNUSED_DECODER_NORM_SUFFIXES = frozenset({"norm.0.scalar_norm.weight", "norm.0.scalar_norm.bias"})


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

    state_dict = cast(
        Mapping[str, Tensor],
        torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        ),
    )
    num_conformations = _ARTIFACTS[artifact]
    model = GrnadeModel(GrnadeConfig(edge_scalar_size=132 if num_conformations is None else 131))
    model.load_state_dict(
        convert_grnade_state_dict(state_dict),
        strict=True,
        assign=True,
    )
    model.eval()
    if num_conformations is None:
        save_repository(
            GrnadePipeline(model=model, processor=GrnadeProcessor()),
            output_dir,
            sources={"checkpoint": checkpoint_path},
            converter=__name__ + ".convert_grnade_checkpoint",
            options={"artifact": artifact},
            repo_id=repo_id,
            token=token,
            private=private,
        )
    else:
        save_repository(
            GrnadeV032Pipeline(
                model=model,
                processor=GrnadeV032Processor(GrnadeV032ProcessorConfig(max_num_conformations=num_conformations)),
            ),
            output_dir,
            sources={"checkpoint": checkpoint_path},
            converter=__name__ + ".convert_grnade_checkpoint",
            options={"artifact": artifact},
            repo_id=repo_id,
            token=token,
            private=private,
        )


def convert_grnade_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    r"""
    Translate live tensors, discarding the verified inactive decoder norm and GVP placeholders, with the decoder input
    rows (the v1 graph's five, v0.3.2's four without the unknown row) and the output projection regenerated onto the
    72 canonical classes.
    """

    converted: dict[str, Tensor] = {}
    for source_key, value in state_dict.items():
        if source_key in _PLACEHOLDER_KEYS:
            continue
        if source_key.startswith("decoder_layers."):
            block, separator, suffix = source_key.removeprefix("decoder_layers.").partition(".")
            if separator and block.isdecimal() and suffix in _UNUSED_DECODER_NORM_SUFFIXES:
                continue
        converted[_convert_key(source_key)] = value
    embedding = converted["token_embedding.weight"]
    (converted["token_embedding.weight"],) = canonical_word_embeddings(
        embedding, vocabulary=GRNADE_INPUT_VOCABULARY[: embedding.shape[0]]
    )
    weight, bias = "output_projection.scalar_projection.weight", "output_projection.scalar_projection.bias"
    converted[weight], converted[bias] = canonical_word_embeddings(
        converted[weight], converted[bias], vocabulary=GRNADE_VOCABULARY
    )
    return converted


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
        fail("value_invalid", RequestRef("checkpoint", key))

    for source, target in (
        (".conv.message_func.", ".message_passing.message."),
        (".norm.0.", ".message_norm."),
        (".norm.1.", ".transition.input_norm."),
        (".ff_func.", ".transition.feed_forward."),
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
