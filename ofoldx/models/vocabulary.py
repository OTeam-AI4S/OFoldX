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

r"""Model vocabulary and feature-axis adaptation.

Every model's token axis is the 72 canonical classes of ``ofoldx.data.tokenization.TOKENS``. A checkpoint trained on
an upstream vocabulary is regenerated onto that axis once, at conversion, with MultiMolecule's
``convert_word_embeddings``: classes the checkpoint knows are copied, new classes drawn from N(0, 0.02),
ambiguity codes and the unknown tokens the mean of their member rows. A layer whose input is normalised over the
whole feature vector keeps its upstream width and reads the canonical planes through ``folding_source_map``.

Attributes:
    FOLDING_SOURCE_TOKEN_INDEX: The 33-row upstream folding alphabet (``<pad>``, gap, protein, RNA, DNA) in canonical
        spelling.
    FOLDING_SOURCE_ALIASES: Canonical classes the upstream folding alphabet reads as another of its rows: ``OTHER``
        is its unknown protein token, which upstream gives every per-atom token, ligands included.
    VOCAB_MAPPING: Per canonical class with members, its member classes (MultiMolecule's protein, RNA and DNA tables
        in canonical spelling).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from multimolecule.tokenisers.dna.utils import VOCAB_MAPPING as DNA_VOCAB_MAPPING
from multimolecule.tokenisers.protein.utils import VOCAB_MAPPING as PROTEIN_VOCAB_MAPPING
from multimolecule.tokenisers.rna.utils import VOCAB_MAPPING as RNA_VOCAB_MAPPING
from multimolecule.tokenisers.utils import convert_word_embeddings
from torch import Tensor

from ..data.tokenization import TOKEN_INDEX, TOKENS

FOLDING_SOURCE_TOKEN_INDEX = {
    token: index
    for index, token in enumerate(
        ("<pad>", "-", *"ARNDCQEGHILKMFPSTWYVX", *(f"RNA_{x}" for x in "AGCUN"), *(f"DNA_{x}" for x in "AGCTN"))
    )
}
FOLDING_SOURCE_ALIASES: Mapping[str, str] = {"OTHER": "X"}

VOCAB_MAPPING: Mapping[str, tuple[str, ...]] = {
    **{token: tuple(members) for token, members in PROTEIN_VOCAB_MAPPING.items()},
    **{
        f"{prefix}{token}": tuple(f"{prefix}{member}" for member in members)
        for prefix, table in (("RNA_", RNA_VOCAB_MAPPING), ("DNA_", DNA_VOCAB_MAPPING))
        for token, members in table.items()
        if f"{prefix}{token}" in TOKEN_INDEX
    },
}


def canonical_word_embeddings(*embeddings: Tensor, vocabulary: Sequence[str], dim: int = 0) -> tuple[Tensor, ...]:
    r"""
    Regenerate checkpoint tensors indexed by ``vocabulary`` along ``dim`` onto the 72 canonical classes.

    ``vocabulary`` names each checkpoint row in canonical spelling. Rows of classes the checkpoint knows are copied,
    new classes are drawn from N(0, 0.02) (biases zero), and ambiguity codes and unknown tokens are the mean of their
    member rows (MultiMolecule ``convert_word_embeddings``, seed 1016). Dtypes are kept.
    """

    moved = [embedding.movedim(dim, 0) for embedding in embeddings]
    converted = convert_word_embeddings(
        *moved, old_vocab=list(vocabulary), new_vocab=list(TOKENS), vocab_mapping=VOCAB_MAPPING
    )
    return tuple(new.to(old.dtype).movedim(0, dim).contiguous() for new, old in zip(converted, embeddings, strict=True))


def folding_word_embeddings(*embeddings: Tensor, dim: int = 0) -> tuple[Tensor, ...]:
    r"""
    ``canonical_word_embeddings`` of tensors indexed by the upstream folding alphabet, with each alias class
    (``FOLDING_SOURCE_ALIASES``) a copy of the row it is read as.
    """

    converted = canonical_word_embeddings(*embeddings, vocabulary=tuple(FOLDING_SOURCE_TOKEN_INDEX), dim=dim)
    for tensor in converted:
        for alias, token in FOLDING_SOURCE_ALIASES.items():
            tensor.select(dim, TOKEN_INDEX[alias]).copy_(tensor.select(dim, TOKEN_INDEX[token]))
    return converted


def folding_source_map() -> Tensor:
    r"""
    ``[33, 72]``: per canonical class (column), its coefficients on the upstream folding alphabet — a class the
    alphabet holds is itself, an alias the row it is read as, a class whose members it holds is their mean, every
    other class is zero.
    """

    source = len(FOLDING_SOURCE_TOKEN_INDEX)
    mapping = torch.zeros(source, len(TOKENS))
    for column, token in enumerate(TOKENS):
        token = FOLDING_SOURCE_ALIASES.get(token, token)
        members = (token,) if token in FOLDING_SOURCE_TOKEN_INDEX else VOCAB_MAPPING.get(token, ())
        rows = [FOLDING_SOURCE_TOKEN_INDEX[member] for member in members if member in FOLDING_SOURCE_TOKEN_INDEX]
        if rows and len(rows) == len(members):
            mapping[rows, column] = 1 / len(rows)
    return mapping


def map_token_feature_planes(features: Tensor, vocabulary_map: Tensor, prefix_size: int) -> Tensor:
    r"""Map two categorical feature planes while retaining their real prefix and suffix.

    ``vocabulary_map[source, canonical]`` gives each canonical category's source
    coefficients. The feature layout is ``[prefix | token | profile | suffix]``;
    both categorical planes have the map's canonical width. Elementwise products
    and sums preserve feature arithmetic dtype, including under caller AMP.
    """

    canonical_vocab_size = vocabulary_map.shape[-1]
    planes_end = prefix_size + 2 * canonical_vocab_size
    planes = features[..., prefix_size:planes_end].unflatten(-1, (2, canonical_vocab_size))
    mapped = (planes.unsqueeze(-1) * vocabulary_map.to(features).T).sum(dim=-2).flatten(-2)
    return torch.cat((features[..., :prefix_size], mapped, features[..., planes_end:]), dim=-1)
