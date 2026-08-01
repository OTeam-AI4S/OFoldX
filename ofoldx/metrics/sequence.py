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
Sequence composition, recovery, alignment scores, and hydrophobicity metrics.
"""

from __future__ import annotations

from itertools import zip_longest

import numpy as np
import torch
from torch import Tensor

from ..data.residue_constants import AMINO_ACID_ONE_TO_THREE
from ..data.tokenization import TOKEN_INDEX
from ..modules.functional import weighted_mean

_HYDROPHOBICITY = {
    "W": {"Rc": 12.25, "Rc1": 11.1, "Rc2": 11.8, "Rn": 12.25, "Rn1": 12.1},
    "F": {"Rc": 10.90, "Rc1": 7.5, "Rc2": 9.5, "Rn": 10.90, "Rn1": 10.3},
    "L": {"Rc": 9.30, "Rc1": 5.55, "Rc2": 7.4, "Rn": 9.30, "Rn1": 9.3},
    "I": {"Rc": 8.00, "Rc1": 5.2, "Rc2": 6.6, "Rn": 8.00, "Rn1": 7.7},
    "M": {"Rc": 6.20, "Rc1": 4.4, "Rc2": 5.7, "Rn": 6.20, "Rn1": 6.0},
    "V": {"Rc": 5.00, "Rc1": 2.9, "Rc2": 3.4, "Rn": 5.00, "Rn1": 4.2},
    "Y": {"Rc": 4.85, "Rc1": 3.7, "Rc2": 4.5, "Rn": 4.85, "Rn1": 4.4},
    "C": {"Rc": 0.45, "Rc1": 0.9, "Rc2": 0.2, "Rn": 0.45, "Rn1": -0.5},
    "P": {"Rc": 2.10, "Rc1": 2.1, "Rc2": 2.1, "Rn": 2.10, "Rn1": 2.1},
    "A": {"Rc": 1.10, "Rc1": 0.35, "Rc2": 0.5, "Rn": 1.10, "Rn1": -0.1},
    "E": {"Rc": 0.95, "Rc1": 1.0, "Rc2": 0.0, "Rn": 0.95, "Rn1": -0.1},
    "T": {"Rc": 0.65, "Rc1": 0.8, "Rc2": 0.6, "Rn": 0.65, "Rn1": 0.0},
    "D": {"Rc": 0.15, "Rc1": 0.5, "Rc2": 0.4, "Rn": 0.15, "Rn1": -0.5},
    "Q": {"Rc": -0.40, "Rc1": -0.7, "Rc2": -0.2, "Rn": -0.40, "Rn1": -1.1},
    "S": {"Rc": -0.15, "Rc1": 0.8, "Rc2": -0.1, "Rn": -0.15, "Rn1": -1.2},
    "G": {"Rc": -0.35, "Rc1": 0.2, "Rc2": 0.15, "Rn": -0.35, "Rn1": -0.7},
    "R": {"Rc": -1.40, "Rc1": 0.5, "Rc2": -1.1, "Rn": -1.30, "Rn1": -1.1},
    "N": {"Rc": -0.85, "Rc1": 0.2, "Rc2": -0.2, "Rn": -0.85, "Rn1": -1.1},
    "H": {"Rc": -1.45, "Rc1": -0.1, "Rc2": -0.2, "Rn": -1.45, "Rn1": -1.7},
    "K": {"Rc": -2.05, "Rc1": -0.6, "Rc2": -1.5, "Rn": -1.90, "Rn1": -1.45},
}


_NEIGHBOR_PENALTY = {"W": 0.15, "F": 0.10, "L": 0.30, "I": 0.15, "V": 0.20, "Y": 0.05}


def amino_acid_fractions(sequence: str) -> dict[str, float]:
    r"""
    Return all 20 shared three-letter amino-acid fraction scores.

    ``sequence`` uses the one-letter designed sequence retained for
    deduplication and diversity selection.
    """

    return {
        f"{three_letter}_fraction": sequence.count(one_letter) / len(sequence)
        for one_letter, three_letter in AMINO_ACID_ONE_TO_THREE.items()
    }


def sequence_recovery(designed: str, native: str) -> float:
    r"""
    Return positional sequence recovery against a native designed sequence.
    """

    if len(designed) != len(native):
        raise ValueError("sequence recovery requires paired sequences of equal length")
    designed_tokens = torch.tensor([ord(residue) for residue in designed], dtype=torch.long)
    native_tokens = torch.tensor([ord(residue) for residue in native], dtype=torch.long)
    return float(token_recovery(designed_tokens, native_tokens, torch.ones_like(designed_tokens, dtype=torch.bool)))


def known_recovery_mask(native_tokens: Tensor, mask: Tensor) -> Tensor:
    r"""Select canonical protein, DNA and RNA identities that can be compared for sequence recovery.

    Unknown and ambiguous classes have no definite native identity. Encoded parent classes and
    shared RNA/DNA output classes remain eligible when they resolve to a standard residue or base.
    """
    known = (
        *(TOKEN_INDEX[letter] for letter in AMINO_ACID_ONE_TO_THREE),
        *(TOKEN_INDEX[f"DNA_{letter}"] for letter in "ACGT"),
        *(TOKEN_INDEX[f"RNA_{letter}"] for letter in "ACGU"),
    )
    return mask.bool() & torch.isin(native_tokens, native_tokens.new_tensor(known))


def token_recovery(designed_tokens: Tensor, native_tokens: Tensor, mask: Tensor) -> Tensor:
    r"""Return positional token recovery over the selected final sequence axis.

    The token arrays and mask broadcast over any leading candidate axes. Empty
    observation scopes report NaN.
    """

    matches, selected = torch.broadcast_tensors(designed_tokens == native_tokens, mask.bool())
    count = selected.sum(dim=-1)
    value = (matches & selected).sum(dim=-1) / count.clamp_min(1)
    return torch.where(count > 0, value, value.new_full((), float("nan")))


def sequence_nll(log_probabilities: Tensor, sequence_tokens: Tensor, mask: Tensor) -> Tensor:
    r"""Return selected-token negative log likelihood, in nat per scored token.

    ``log_probabilities`` has shape ``(..., N, V)``; tokens and the Boolean
    observation mask broadcast over ``(..., N)``. The result retains the leading
    candidate axes. Empty scopes report NaN with zero log-probability gradients;
    values at excluded positions are never included in the reduction.
    """

    log_probabilities, tokens, selected = torch.broadcast_tensors(
        log_probabilities, sequence_tokens.unsqueeze(-1), mask.bool().unsqueeze(-1)
    )
    # An excluded observation carries no token identity. Select its payload out
    # before integer lookup, including sentinel/out-of-range ignored token IDs.
    tokens = torch.where(selected, tokens, 0)[..., :1]
    in_vocabulary = ((tokens >= 0) & (tokens < log_probabilities.shape[-1])).all()
    if torch.compiler.is_compiling():
        torch._assert_async(in_vocabulary, "scored sequence tokens are outside the vocabulary")
    elif not bool(in_vocabulary):
        raise ValueError("scored sequence tokens are outside the vocabulary")
    values = torch.where(selected, log_probabilities, 0.0)
    errors = -values.gather(-1, tokens).squeeze(-1)
    selected = selected[..., 0]
    return weighted_mean(errors, selected)


def sequence_alignment_score(first: str, second: str) -> float:
    r"""
    Return corresponding-chain alignment score normalized by their lengths.

    The shared aligner uses global alignment with match ``1``, mismatch ``0``,
    and a ``-1`` score for each gap position. Colons separate ordered chains.
    Corresponding chains align independently; unmatched chains add their length
    without a score. Delimiters add no residues.
    """

    score = 0
    num_positions = 0
    for first_chain, second_chain in zip_longest(first.split(":"), second.split(":")):
        num_positions += max(len(first_chain or ""), len(second_chain or ""))
        if first_chain is None or second_chain is None:
            continue
        previous = -np.arange(len(second_chain) + 1, dtype=np.int64)
        for first_index, first_residue in enumerate(first_chain, start=1):
            current = np.empty(len(second_chain) + 1, dtype=np.int64)
            current[0] = -first_index
            for index, second_residue in enumerate(second_chain, start=1):
                current[index] = max(
                    previous[index - 1] + int(first_residue == second_residue),
                    previous[index] - 1,
                    current[index - 1] - 1,
                )
            previous = current
        score += int(previous[-1])
    return score / num_positions if num_positions else float("nan")


def sequence_hydrophobicity(sequence: str) -> float:
    r"""
    Return the shared Thermo Fisher-like sequence hydrophobicity score.
    """

    sequence = (sequence or "").strip().upper()
    if not sequence or "X" in sequence:
        return float("nan")

    score = 0.0
    for index, residue in enumerate(sequence):
        if residue not in _HYDROPHOBICITY:
            raise ValueError(f"unknown residue {residue!r} in {sequence!r}")
        if index == 0:
            coefficient = "Rc1"
        elif index == 1:
            coefficient = "Rc2"
        elif index == len(sequence) - 1:
            coefficient = "Rn"
        elif index == len(sequence) - 2:
            coefficient = "Rn1"
        else:
            coefficient = "Rc"
        score += _HYDROPHOBICITY[residue][coefficient]

    for index, residue in enumerate(sequence):
        if residue in "HRK":
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(sequence):
                    score -= _NEIGHBOR_PENALTY.get(sequence[neighbor], 0.0)

    index = 0
    while index < len(sequence):
        if sequence[index] != "P":
            index += 1
            continue
        end = index
        while end < len(sequence) and sequence[end] == "P":
            end += 1
        num_prolines = end - index
        if num_prolines >= 4:
            score -= 5.0
        elif num_prolines == 3:
            score -= 3.5
        elif num_prolines == 2:
            score -= 1.2
        index = end

    if len(sequence) < 8:
        score *= 1.0 - 0.055 * (8 - len(sequence))
    elif len(sequence) > 20:
        score *= 1.0 / (1.0 + 0.027 * (len(sequence) - 20))

    if score <= 20:
        adjusted = score
    elif score <= 30:
        adjusted = score - 0.27 * (score - 18.0)
    elif score <= 40:
        adjusted = score - 0.33 * (score - 18.0)
    elif score <= 50:
        adjusted = score - 0.38 * (score - 18.0)
    else:
        adjusted = score - 0.447 * (score - 18.0)
    return round(adjusted, 4)


def sequence_isoelectric_point(sequence: str) -> float:
    r"""
    Solve for the zero of the Bjellqvist protein charge model.

    The pKa values match the inspected Biopython model. Expand the bisection
    bracket so highly acidic or basic sequences retain their actual zero;
    stopping at a fixed 4.05--12 boundary does not solve their charge equation.
    """

    positive_pks = {"Nterm": 7.5, "K": 10.0, "R": 12.0, "H": 5.98}
    negative_pks = {"Cterm": 3.55, "D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
    n_terminal_pks = {
        "A": 7.59,
        "M": 7.0,
        "S": 6.93,
        "P": 8.36,
        "T": 6.82,
        "V": 7.44,
        "E": 7.7,
    }
    c_terminal_pks = {"D": 4.55, "E": 4.75}
    if sequence[0] in n_terminal_pks:
        positive_pks["Nterm"] = n_terminal_pks[sequence[0]]
    if sequence[-1] in c_terminal_pks:
        negative_pks["Cterm"] = c_terminal_pks[sequence[-1]]

    charged_content = {
        amino_acid: float(sequence.count(amino_acid)) for amino_acid in ("K", "R", "H", "D", "E", "C", "Y")
    }
    charged_content["Nterm"] = 1.0
    charged_content["Cterm"] = 1.0

    def charge(ph: float) -> float:
        positive_charge = sum(
            charged_content[amino_acid] / (10 ** (ph - pk) + 1.0) for amino_acid, pk in positive_pks.items()
        )
        negative_charge = sum(
            charged_content[amino_acid] / (10 ** (pk - ph) + 1.0) for amino_acid, pk in negative_pks.items()
        )
        return positive_charge - negative_charge

    minimum = 4.05
    maximum = 12.0
    while charge(minimum) < 0.0:
        minimum -= 1.0
    while charge(maximum) > 0.0:
        maximum += 1.0
    ph = (minimum + maximum) / 2
    while maximum - minimum > 0.0001:
        if charge(ph) > 0.0:
            minimum = ph
        else:
            maximum = ph
        ph = (minimum + maximum) / 2
    return ph
