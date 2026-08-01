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

r"""Membrane-label-conditioned protein sequence model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...model import Model
from ...modules.backend import LayerNorm
from ...modules.mpnn import MpnnDecoderBlock, MpnnEncoderBlock, MpnnProteinFeatureEncoder
from ..processing import project_model_categories
from ..proteinmpnn.model import (
    PROTEINMPNN_VOCABULARY,
    ProteinMpnnInput,
    ProteinMpnnOutput,
    ProteinMpnnSamplingOutput,
    ProteinMpnnSequenceModel,
)
from .config import MembraneMpnnConfig


class MembraneMpnnModel(ProteinMpnnSequenceModel):
    r"""Predict protein identities conditioned on backbone and membrane labels.

    Label zero means exposed/other, one interface and two buried for the
    per-residue checkpoint. The global checkpoint uses zero for soluble and one
    for transmembrane. Both checkpoints execute the same three-class graph.
    """

    config_class = MembraneMpnnConfig
    config: MembraneMpnnConfig

    def __init__(self, config: MembraneMpnnConfig | None = None) -> None:
        config = MembraneMpnnConfig() if config is None else config
        Model.__init__(self, config)
        self.gradient_checkpointing = False
        self.feature_encoder = MpnnProteinFeatureEncoder(config.hidden_size, config.num_neighbors)
        self.node_embedding = nn.Linear(3, config.hidden_size, bias=False)
        self.node_norm = LayerNorm(config.hidden_size)
        self.node_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.edge_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            MpnnEncoderBlock(config.hidden_size, config.hidden_size * 2, dropout=config.dropout)
            for _ in range(config.num_encoder_blocks)
        )
        self.token_embedding = nn.Embedding(len(TOKENS), config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            MpnnDecoderBlock(config.hidden_size, config.hidden_size * 3, dropout=config.dropout)
            for _ in range(config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(config.hidden_size, len(TOKENS))
        trained = torch.zeros(len(TOKENS), dtype=torch.bool)
        trained[[TOKEN_INDEX[token] for token in PROTEINMPNN_VOCABULARY]] = True
        self.register_buffer("trained_classes", trained, persistent=False)
        self.post_init()

    def _initial_node_states(self, input: ProteinMpnnInput, edge_states: Tensor) -> Tensor:
        if input.membrane_labels is None:
            fail("value_invalid", RequestRef("membrane_labels", "None"))
        mode = self.config.label_mode
        if mode not in ("global", "per_residue"):
            fail("value_invalid", RequestRef("label_mode", str(mode)), reason="mode_required")
        labels = project_model_categories(
            input.membrane_labels,
            input.residue_mask,
            num_classes=2 if mode == "global" else self.node_embedding.in_features,
            feature="membrane labels",
        )
        if mode == "global":
            present = input.residue_mask.bool()
            low = torch.where(present, labels, 2).amin(dim=-1)
            high = torch.where(present, labels, -1).amax(dim=-1)
            uniform = (~present.any(dim=-1) | (low == high)).all()
            if torch.compiler.is_compiling():
                torch._assert_async(uniform, "global membrane labels must be uniform")
            elif not bool(uniform):
                fail("value_invalid", RequestRef("membrane_labels", "global labels must be uniform"))
        labels = F.one_hot(labels, num_classes=self.node_embedding.in_features)
        labels = labels.to(dtype=self.node_embedding.weight.dtype)
        return self.node_projection(self.node_norm(self.node_embedding(labels)))

    @staticmethod
    def _repeat_input(input: ProteinMpnnInput, num_repeats: int) -> ProteinMpnnInput:
        repeated = ProteinMpnnSequenceModel._repeat_input(input, num_repeats)
        if repeated.tied_weights is None:
            repeated.tied_weights = torch.ones_like(repeated.residue_mask, dtype=repeated.backbone_coordinates.dtype)
        return repeated

    def score(
        self,
        input: ProteinMpnnInput,
        *,
        use_sequence: bool = True,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
        r"""Evaluate sequence-conditioned or sequence-free autoregressive logits."""
        operation = (
            self.autoregressive_probabilities if use_sequence else self.sequence_free_autoregressive_probabilities
        )
        return operation(
            input,
            num_decoding_orders=num_decoding_orders,
            decoding_ranks=decoding_ranks,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    @torch.inference_mode()
    def sample(
        self,
        input: ProteinMpnnInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_token_mask: Tensor | None = None,
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnSamplingOutput:
        r"""Sample protein identities, omitting unknown by default."""
        if global_omit_token_mask is None:
            global_omit_token_mask = torch.arange(len(TOKENS), device=input.residue_mask.device) == TOKEN_INDEX["X"]
        return super().sample(
            input,
            num_samples=num_samples,
            temperature=temperature,
            decoding_ranks=decoding_ranks,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=global_omit_token_mask,
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )

    @torch.inference_mode()
    def design(
        self,
        input: ProteinMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_token_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnSamplingOutput:
        r"""Sample identities without a separate input-sequence evaluation."""
        return self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=global_omit_token_mask,
            generator=generator,
        )
