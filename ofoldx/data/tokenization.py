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
The unified canonical vocabulary, the component → class encoding with recorded loss, and the one vectorised atom →
token partition.

72 classes, MultiMolecule's standard alphabets with its special tokens first: 0–5 ``<pad> <cls> <eos> <unk> <mask>
<null>``, 6–31 protein ``ACDEFGHIKLMNPQRSTVWYXZBJUO``, 32–48 ``RNA_`` + ``ACGUNRYSWKMBDHVIX``, 49–65 ``DNA_`` +
``ACGTNRYSWKMBDHVIX``, 66–70 ``| . * - ?``, 71 ``OTHER``. ``<unk>`` (an input symbol outside the vocabulary), ``X`` (a
residue that maps to no other class) and ``OTHER`` (not a monomer of the three polymer types, or of an undeterminable
class) are three different classes. The encoding is a representation rule, never a chemical claim: the component keeps
its identity and the loss is recorded per component.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, StructureRef, fail
from ..provenance import known
from .molecular import EntityType, MolecularSystem, MonomerType
from .schema import check_mask, check_rows

_PROTEIN, _NUCLEIC, _SYMBOLS = "ACDEFGHIKLMNPQRSTVWYXZBJUO", "NRYSWKMBDHVIX", "|.*-?"
TOKENS: tuple[str, ...] = (
    ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>")
    + tuple(_PROTEIN)
    + tuple(f"RNA_{letter}" for letter in "ACGU" + _NUCLEIC)
    + tuple(f"DNA_{letter}" for letter in "ACGT" + _NUCLEIC)
    + tuple(_SYMBOLS)
    + ("OTHER",)
)
TOKEN_INDEX: Mapping[str, int] = MappingProxyType({token: index for index, token in enumerate(TOKENS)})


class Encoding(IntEnum):
    r"""How a component reached its class: its own id, its parent, the other pyrimidine, its class's unknown token, or
    no polymer class (``OTHER``)."""

    EXACT = 0
    PARENT = 1
    CROSS_PYRIMIDINE = 2
    UNMAPPED = 3
    NON_POLYMER = 4


_AMINO = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
_STANDARD = {
    MonomerType.PROTEIN: {
        **{name: letter for letter, name in _AMINO.items()},
        **{"UNK": "X", "ASX": "B", "GLX": "Z", "XLE": "J", "SEC": "U", "PYL": "O"},
    },
    MonomerType.RNA: {name: f"RNA_{name}" for name in "ACGUIN"},
    MonomerType.DNA: {f"D{letter}": f"DNA_{letter}" for letter in "ACGTIN"},
}
_CROSS = {MonomerType.DNA: {"DU": "DNA_T"}}  # DNA reads U as T; the component stays DU
_UNKNOWN = {MonomerType.PROTEIN: "X", MonomerType.RNA: "RNA_X", MonomerType.DNA: "DNA_X"}
_POLYMER_CLASS = {
    "polypeptide(L)": MonomerType.PROTEIN,
    "polypeptide(D)": MonomerType.PROTEIN,
    "polyribonucleotide": MonomerType.RNA,
    "polydeoxyribonucleotide": MonomerType.DNA,
}
_NOT_POLYMER = (EntityType.NON_POLYMER, EntityType.BRANCHED, EntityType.WATER, EntityType.MACROLIDE)
_ALPHABET = {
    MonomerType.PROTEIN: {letter: letter for letter in _PROTEIN},
    MonomerType.RNA: {letter: f"RNA_{letter}" for letter in "ACGU" + _NUCLEIC} | {"T": "RNA_U"},
    MonomerType.DNA: {letter: f"DNA_{letter}" for letter in "ACGT" + _NUCLEIC} | {"U": "DNA_T"},
}


@dataclass(frozen=True)
class TokenClasses:
    r"""Per component: its canonical class (a ``TOKENS`` index) and how it was reached (``Encoding``)."""

    token: NDArray[np.uint8]
    encoding: NDArray[np.uint8]


def _lookup(table: tuple[str, ...], monomer: MonomerType, rules: Mapping) -> NDArray[np.int64]:
    r"""Per symbol of a family, its class under ``monomer`` by ``rules`` (``-1`` none)."""
    mapping = rules.get(monomer, {})
    return np.array([TOKEN_INDEX[mapping[text]] if text in mapping else -1 for text in table], dtype=np.int64)


def token_classes(system: MolecularSystem) -> TokenClasses:
    r"""
    The canonical class of every component, in this order.

    The component's monomer class is its own (``component_monomer_type``, KNOWN and not ``NONE``), else the one its
    instance's typed polymer supplies (``polypeptide(L/D)``, ``polyribonucleotide``, ``polydeoxyribonucleotide``). (1)
    A standard id of that class → its class, ``EXACT`` (``DU`` → ``DNA_T``, ``CROSS_PYRIMIDINE``); (2) exactly one
    parent of the component's definition (``component_definition_parent``: upstream maps a modified residue by the
    CCD's ``mon_nstd_parent_comp_id``, never by the occurrence's annotation) that is a standard id of
    that class → the parent's class, ``PARENT``; (3) a determined class → its
    unknown token (``X``, ``RNA_X``, ``DNA_X``), ``UNMAPPED``; (4) otherwise ``OTHER``: ``NON_POLYMER`` for components
    of non-polymer, branched, macrolide and water entities, ``UNMAPPED`` otherwise.
    """

    s, symbols = system.arrays(), system.symbols()
    count = len(s["component_symbol"])
    component_instance = np.repeat(np.arange(len(s["instance_symbol"])), np.diff(s["instance_component_ptr"]))
    entity = s["instance_entity"][component_instance]
    own = np.where(known(s["component_monomer_type_state"]), s["component_monomer_type"], 0).astype(np.int64)
    polymer = np.array([_POLYMER_CLASS.get(text, 0) for text in symbols["polymer_type"]] + [0], dtype=np.int64)
    supplied = np.where(known(s["entity_polymer_type_state"]), polymer[s["entity_polymer_type"]], 0)[entity]
    monomer = np.where(own > 0, own, supplied)
    token = np.full(count, TOKEN_INDEX["OTHER"], dtype=np.int64)
    encoding = np.full(count, Encoding.UNMAPPED, dtype=np.int64)
    single = np.diff(s["component_definition_parent_ptr"]) == 1
    parents, start = s["component_definition_parent"], s["component_definition_parent_ptr"][:-1]
    parent = np.where(single, parents[np.minimum(start, len(parents) - 1)], -1) if len(parents) else np.full(count, -1)
    for kind in (MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA):
        mine = monomer == kind
        exact = np.append(_lookup(symbols["component"], kind, _STANDARD), -1)[np.where(mine, s["component_symbol"], -1)]
        cross = np.append(_lookup(symbols["component"], kind, _CROSS), -1)[np.where(mine, s["component_symbol"], -1)]
        by_parent = np.append(_lookup(symbols["parent"], kind, _STANDARD), -1)[np.where(mine, parent, -1)]
        for found, how in ((by_parent, Encoding.PARENT), (cross, Encoding.CROSS_PYRIMIDINE), (exact, Encoding.EXACT)):
            token, encoding = np.where(found >= 0, found, token), np.where(found >= 0, how, encoding)
        unmapped = mine & (exact < 0) & (cross < 0) & (by_parent < 0)
        token = np.where(unmapped, TOKEN_INDEX[_UNKNOWN[kind]], token)
    typed = known(s["entity_type_state"][entity])
    outside = (monomer == 0) & typed & np.isin(s["entity_type"][entity], _NOT_POLYMER)
    encoding = np.where(outside, Encoding.NON_POLYMER, encoding)
    return TokenClasses(token.astype(np.uint8), encoding.astype(np.uint8))


def encode_sequence(text: str, monomer: MonomerType) -> NDArray[np.uint8]:
    r"""
    The classes of a one-letter sequence of one monomer type; RNA reads ``T`` as ``U``, DNA reads ``U`` as ``T``.

    Shared symbols ``| . * - ?`` are valid in every type; any other symbol outside the alphabet is ``<unk>``
    (letters are case-sensitive: lowercase is outside the alphabet).

    Examples:
        >>> [TOKENS[i] for i in encode_sequence("ACU-", MonomerType.DNA)]
        ['DNA_A', 'DNA_C', 'DNA_T', '-']
    """

    if monomer not in _ALPHABET:
        fail("value_invalid", RequestRef("monomer", str(monomer)), invariant="polymer_monomer")
    alphabet = _ALPHABET[MonomerType(monomer)]
    table = {**{symbol: TOKEN_INDEX[symbol] for symbol in _SYMBOLS}, **{c: TOKEN_INDEX[t] for c, t in alphabet.items()}}
    return np.array([table.get(character, TOKEN_INDEX["<unk>"]) for character in text], dtype=np.uint8)


_PREFIX_MONOMER = {"RNA_": MonomerType.RNA, "DNA_": MonomerType.DNA}
#: Per class, its monomer type (``NONE`` for specials, shared symbols and ``OTHER``).
CLASS_MONOMER = np.array(
    [_PREFIX_MONOMER.get(token[:4], MonomerType.PROTEIN if token in set(_PROTEIN) else 0) for token in TOKENS],
    dtype=np.uint8,
)
#: Per class, its sequence letter (``""`` for a class without one).
_CLASS_LETTER = tuple(
    "" if not monomer else "N" if token in ("RNA_X", "DNA_X") else token.removeprefix("RNA_").removeprefix("DNA_")
    for token, monomer in zip(TOKENS, CLASS_MONOMER)
)


def class_sequence(token_class: NDArray[np.uint8], ref: StructureRef | None = None) -> str:
    r"""
    The one sequence of canonical classes: each class's letter with its ``RNA_``/``DNA_`` prefix stripped; ``X`` stays
    ``X``, ``RNA_X`` and ``DNA_X`` read ``N``. It is the query sequence of a polymer instance (the MSA index key and
    the MSA query check) and of a template chain.

    Raises:
        UnsupportedError: ``query_class_unrepresentable`` for ``OTHER``, a special token or a shared symbol, at
            ``ref`` (the instance whose classes these are) and the position in ``token_class``.

    Examples:
        >>> class_sequence(np.array([TOKEN_INDEX[t] for t in ("M", "RNA_A", "DNA_X")], dtype=np.uint8))
        'MAN'
    """

    codes = check_rows("token_class", token_class, len(TOKENS), shape=(None,))
    letters = [_CLASS_LETTER[code] for code in codes.tolist()]
    if "" in letters:
        position = letters.index("")
        refs = () if ref is None else (ref,)
        fail("query_class_unrepresentable", *refs, position=position, token=TOKENS[int(token_class[position])])
    return "".join(letters)


@dataclass(frozen=True)
class TokenLayout:
    r"""
    One structural token axis over a system's canonical atoms.

    ``token_atoms[token_atom_ptr[t] : token_atom_ptr[t + 1]]`` are the atoms of token ``t``; ``atom_token`` is the
    token of every canonical atom (``-1`` for an atom in no token); ``token_class`` indexes ``TOKENS``.
    """

    token_component: NDArray[np.int64]
    token_centre_atom: NDArray[np.int64]
    token_class: NDArray[np.uint8]
    atom_token: NDArray[np.int64]
    token_atom_ptr: NDArray[np.int64]
    token_atoms: NDArray[np.int64]


def tokenize(
    system: MolecularSystem,
    *,
    residue_token: NDArray[np.bool_],
    centre_atom: NDArray[np.int64],
    atom_mask: NDArray[np.bool_] | None = None,
) -> TokenLayout:
    r"""
    Partition the visible atoms (``atom_mask``, default every atom) into tokens without changing canonical indices.

    A component with ``residue_token`` set is one token of all its visible atoms, centred on ``centre_atom`` (which must
    be one of them); a component with no visible atom has no token. Every other component is one token per visible atom.
    Tokens follow component order, then atom order; each token's class is its component's ``token_classes``.
    """

    n, count = len(system.atom_symbol), len(system.component_symbol)
    residue_token = check_mask("residue_token", residue_token, count)
    centre_atom = np.asarray(centre_atom)  # read only for one-token components, each checked as a visible member
    if centre_atom.shape != (count,) or (count and centre_atom.dtype.kind not in "iu"):
        fail("value_invalid", RequestRef("centre_atom", f"{centre_atom.dtype}{list(centre_atom.shape)}"), rows=count)
    visible = np.ones(n, dtype=bool) if atom_mask is None else check_mask("atom_mask", atom_mask, n)
    pointer = system.component_atom_ptr
    atom_component = np.repeat(np.arange(count), np.diff(pointer))
    shown = np.bincount(atom_component[visible], minlength=count)
    grouped = residue_token & (shown > 0)
    safe = np.clip(centre_atom, 0, max(n - 1, 0))
    member = (centre_atom >= pointer[:-1]) & (centre_atom < pointer[1:]) & (visible[safe] if n else False)
    if (grouped & ~member).any():
        component = int(np.flatnonzero(grouped & ~member)[0])
        fail("value_invalid", RequestRef("centre_atom", str(component)), invariant="visible_member")
    per_component = np.where(grouped, 1, np.where(residue_token, 0, shown))
    first_token = np.concatenate(([0], np.cumsum(per_component)))[:-1]
    atoms = np.flatnonzero(visible)
    owner = atom_component[atoms]
    rank = np.arange(len(atoms)) - np.concatenate(([0], np.cumsum(shown)))[:-1][owner]
    token_of = np.where(grouped[owner], first_token[owner], first_token[owner] + rank)
    total = int(per_component.sum())
    atom_token = np.full(n, -1, dtype=np.int64)
    atom_token[atoms] = token_of
    token_component = np.repeat(np.arange(count, dtype=np.int64), per_component)
    centre = np.zeros(total, dtype=np.int64)
    centre[token_of] = atoms
    centre = np.where(grouped[token_component], centre_atom[token_component], centre)
    classes = token_classes(system).token
    return TokenLayout(
        token_component=token_component,
        token_centre_atom=centre.astype(np.int64),
        token_class=classes[token_component],
        atom_token=atom_token,
        token_atom_ptr=np.concatenate(([0], np.cumsum(np.bincount(token_of, minlength=total)))).astype(np.int64),
        token_atoms=atoms.astype(np.int64),
    )


def select_tokens(layout: TokenLayout, keep: NDArray[np.bool_]) -> TokenLayout:
    r"""The tokens ``keep`` with their atoms; atoms of dropped tokens are in no token."""
    keep = check_mask("keep", keep, len(layout.token_component))
    new = np.where(keep, np.cumsum(keep) - 1, -1)
    counts = np.diff(layout.token_atom_ptr)
    atom_keep = np.repeat(keep, counts)
    atom_token = np.append(new, -1)[layout.atom_token]
    return TokenLayout(
        token_component=layout.token_component[keep],
        token_centre_atom=layout.token_centre_atom[keep],
        token_class=layout.token_class[keep],
        atom_token=atom_token.astype(np.int64),
        token_atom_ptr=np.concatenate(([0], np.cumsum(counts[keep]))).astype(np.int64),
        token_atoms=layout.token_atoms[atom_keep],
    )
