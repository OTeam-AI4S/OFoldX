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
Strict conversion of the two released membrane MPNN checkpoints.

Notes:
    The released files are ``global_label_membrane_mpnn_v_48_020.pt`` and
    ``per_residue_label_membrane_mpnn_v_48_020.pt``. Their label vocabularies differ: global conditioning uses
    soluble/transmembrane labels, while residue conditioning uses exposed/other, interface, and buried labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ...conversion import save_repository
from ...errors import RequestRef, fail
from ..proteinmpnn.convert import convert_proteinmpnn_state_dict
from .config import MembraneLabelMode, MembraneMpnnConfig, MembraneMpnnProcessorConfig
from .model import MembraneMpnnModel
from .pipeline import MembraneMpnnPipeline
from .processor import MembraneMpnnProcessor


def convert_membranempnn_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""Translate every live membrane tensor without dropping node conditioning."""
    node_names = {
        "features.node_embedding.weight": "node_embedding.weight",
        "features.norm_nodes.weight": "node_norm.weight",
        "features.norm_nodes.bias": "node_norm.bias",
        "W_v.weight": "node_projection.weight",
        "W_v.bias": "node_projection.bias",
    }
    converted = convert_proteinmpnn_state_dict({key: value for key, value in state.items() if key not in node_names})
    converted.update({target: state[source] for source, target in node_names.items()})
    return converted


def convert_membranempnn_model(
    checkpoint_path: str | PathLike[str],
    *,
    label_mode: MembraneLabelMode,
    dtype: torch.dtype | None = None,
) -> MembraneMpnnModel:
    r"""Load one membrane checkpoint, retaining its exact learned label meaning."""
    if label_mode not in ("global", "per_residue"):
        fail("value_invalid", RequestRef("label_mode", str(label_mode)))
    checkpoint: dict[str, Any] = torch.load(
        Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=True, mmap=True
    )
    state = checkpoint["model_state_dict"]
    config = MembraneMpnnConfig(
        label_mode=label_mode,
        hidden_size=state["W_s.weight"].shape[1],
        num_encoder_blocks=len({key.split(".")[1] for key in state if key.startswith("encoder_layers.")}),
        num_decoder_blocks=len({key.split(".")[1] for key in state if key.startswith("decoder_layers.")}),
        num_neighbors=checkpoint["num_edges"],
    )
    model = MembraneMpnnModel(config)
    model.load_state_dict(convert_membranempnn_state_dict(state), strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    return model


def convert_membranempnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    label_mode: MembraneLabelMode,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Save one membrane Model and its label-aware Processor as a native pair."""
    save_repository(
        MembraneMpnnPipeline(
            model=convert_membranempnn_model(checkpoint_path, label_mode=label_mode),
            processor=MembraneMpnnProcessor(MembraneMpnnProcessorConfig(label_mode=label_mode)),
        ),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter=__name__ + ".convert_membranempnn_checkpoint",
        options={"label_mode": label_mode},
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label-mode", required=True, choices=("global", "per_residue"))
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)
    convert_membranempnn_checkpoint(
        args.checkpoint,
        args.output_dir,
        label_mode=args.label_mode,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
