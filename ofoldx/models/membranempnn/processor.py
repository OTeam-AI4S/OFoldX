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

r"""Backbone and membrane-label preparation on ProteinMPNN's prepared rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ...data.crop import Cropper
from ...data.molecular import ComponentRef
from ...data.structure import Structure
from ...errors import RequestRef, fail
from ...model import Model
from ...processor import Processor
from ..conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..design import PositionSelection
from ..layout import LayoutTable
from ..preparation import PreparedInput
from ..proteinmpnn.model import ProteinMpnnInput
from ..proteinmpnn.processor import PROTEINMPNN_LAYOUT, ProteinMpnnProcessor
from .config import MembraneMpnnConfig, MembraneMpnnProcessorConfig


class MembraneMpnnProcessor(ProteinMpnnProcessor):
    r"""
    Prepare N, CA, C, O backbones and scalar or per-residue membrane labels.

    The global checkpoint takes one scalar label: zero = soluble, one = transmembrane. The per-residue checkpoint
    accepts a scalar broadcast to every row or a mapping that leaves unnamed residues zero: zero = exposed or other,
    one = interface, two = buried. The Processor and its paired Model must state the same label mode. A tied group's
    members default to coefficient one (upstream LigandMPNN repository), not ProteinMPNN's ``1 / len(group)``.
    """

    config_class = MembraneMpnnProcessorConfig
    config: MembraneMpnnProcessorConfig

    def __init__(
        self,
        config: MembraneMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = MembraneMpnnProcessorConfig() if config is None else config
        Processor.__init__(self, device=device, dtype=dtype)

    @property
    def layout(self) -> LayoutTable:
        return PROTEINMPNN_LAYOUT

    def validate_model_input_layout(self, model: Model) -> None:
        mode = self.config.label_mode
        model_mode = cast(MembraneMpnnConfig, model.config).label_mode
        if mode not in ("global", "per_residue") or model_mode != mode or self.config.ca_only:
            fail("value_invalid", RequestRef("label_mode", str(mode)), model=model_mode, ca_only=self.config.ca_only)

    def prepare_input(  # type: ignore[override]
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        design: PositionSelection = PositionSelection(),
        sequence: str | Mapping[str, str] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""Prepare ProteinMPNN rows with membrane labels.

        See `ProteinMpnnProcessor.prepare_input` for the remaining arguments.
        """
        mode = self.config.label_mode
        if mode not in ("global", "per_residue"):
            fail("value_invalid", RequestRef("label_mode", str(mode)), reason="mode_required")
        if mode == "global" and isinstance(membrane_labels, Mapping):
            fail("value_invalid", RequestRef("membrane_labels", "global mode requires one label"))

        def label(value: object) -> int:
            maximum = 1 if mode == "global" else 2
            if not isinstance(value, Real):
                fail("value_invalid", RequestRef("membrane_labels", str(value)), label_mode=mode)
            number = float(cast(float, value))
            if not np.isfinite(number) or not 0 <= number <= maximum or not number.is_integer():
                fail("value_invalid", RequestRef("membrane_labels", str(value)), label_mode=mode)
            return int(number)

        validated = (
            {reference: label(value) for reference, value in membrane_labels.items()}
            if isinstance(membrane_labels, Mapping)
            else label(membrane_labels)
        )
        prepared = super().prepare_input(
            structure,
            design=design,
            sequence=sequence,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            filter_by_occupancy=filter_by_occupancy,
            cropper=cropper,
            generator=generator,
        )
        system = structure.system
        input = prepared.features
        rows = input.component_index[0].cpu().numpy()
        if isinstance(validated, Mapping):
            named = system.component_index(list(validated))
            if (named < 0).any():
                reference = list(validated)[int(np.flatnonzero(named < 0)[0])]
                fail("request_invalid_positions", RequestRef("membrane_labels", str(reference)))
            labels = np.zeros(len(system.component_symbol), dtype=np.int64)
            labels[named] = list(validated.values())
            input.membrane_labels = torch.as_tensor(labels[rows], device=self.device).unsqueeze(0)
        else:
            input.membrane_labels = torch.full_like(input.sequence_tokens, validated)
        if tied_weights:
            row_of = {component: row for row, component in enumerate(rows.tolist())}
            weights = torch.ones_like(cast(Tensor, input.tied_weights))
            for component, weight in zip(system.component_index(list(tied_weights)).tolist(), tied_weights.values()):
                weights[0, row_of[component]] = weight
            input.tied_weights = weights
        return prepared
