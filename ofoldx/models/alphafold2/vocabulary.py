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

r"""AlphaFold 2's categorical axes on the canonical vocabulary.

Every AlphaFold 2 embedding axis — the target features, the MSA one-hot and its profile — is the 72 canonical classes;
the BERT mask class is the canonical ``<mask>``. The upstream letter rules the featuriser applies to a given input stay
the family's: an MSA or template letter is read through the HHblits aliases (``B``→``D``, ``Z``→``E``, ``U``→``C``,
``J`` and ``O``→``X``), a target residue other than the 20 standard amino acids is ``X``. The masked-MSA head keeps its
upstream width; ``masked_msa_targets`` maps a canonical class onto it.

Attributes:
    NUM_CLASSES: The canonical classes, specials included.
    GAP: The canonical gap class.
    MASK: The canonical BERT mask class.
    AMINO_ACIDS: The 20 standard amino-acid classes in upstream residue-type order (``ARNDCQEGHILKMFPSTWYV``).
    SOURCE_MSA_ALPHABET: The upstream MSA rows in canonical spelling (``ARNDCQEGHILKMFPSTWYVX-``).
    SOURCE_MASKED_MSA_ALPHABET: ``SOURCE_MSA_ALPHABET`` and the mask: the monomer masked-MSA head and the upstream
        clustered-MSA one-hot.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...data import residue_constants
from ...data.tokenization import TOKEN_INDEX, TOKENS

NUM_CLASSES = len(TOKENS)
GAP = TOKEN_INDEX["-"]
MASK = TOKEN_INDEX["<mask>"]
AMINO_ACIDS = np.array([TOKEN_INDEX[letter] for letter in residue_constants.AMINO_ACID_ONE_LETTER_CODES])
SOURCE_MSA_ALPHABET = (*residue_constants.AMINO_ACID_ONE_LETTER_CODES, "X", "-")
SOURCE_MASKED_MSA_ALPHABET = (*SOURCE_MSA_ALPHABET, "<mask>")

_X = TOKEN_INDEX["X"]
#: Per canonical class, the class an upstream MSA or template letter reads it as (HHblits ``B D``, ``Z E``, ``U C``,
#: ``J X``, ``O X``); every other class is itself.
HHBLITS_CLASS = np.arange(NUM_CLASSES)
HHBLITS_CLASS[[TOKEN_INDEX[letter] for letter in "BZUJO"]] = [TOKEN_INDEX[letter] for letter in "DECXX"]
#: Per canonical class, its upstream residue type (``ARNDCQEGHILKMFPSTWYV``, then 20 for ``X`` and every other class).
RESIDUE_TYPE = np.full(NUM_CLASSES, residue_constants.NUM_AMINO_ACID_TYPES)
RESIDUE_TYPE[AMINO_ACIDS] = np.arange(residue_constants.NUM_AMINO_ACID_TYPES)


def target_class(token_class: NDArray) -> NDArray[np.int64]:
    r"""A target residue's class as the upstream featuriser reads it: a standard amino acid, else ``X``."""
    token_class = np.asarray(token_class, dtype=np.int64)
    return np.where(np.isin(token_class, AMINO_ACIDS), token_class, _X)


def masked_msa_targets(reference: Tensor) -> Tensor:
    r"""``[72, H]``: the one-hot target of each canonical class on the first ``H`` rows of the masked-MSA head (a class
    the head has no row for is zero)."""
    width = reference.shape[-1]
    targets = torch.zeros(NUM_CLASSES, width, dtype=reference.dtype, device=reference.device)
    for row, token in enumerate(SOURCE_MASKED_MSA_ALPHABET[:width]):
        targets[TOKEN_INDEX[token], row] = 1
    return targets
