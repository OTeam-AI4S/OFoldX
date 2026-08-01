r"""
Canonical structural tokenization of molecular systems.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import NamedTuple

import torch
from torch import Tensor

from .molecular import (
    _STANDARD_COMPONENTS,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    STANDARD_AMINO_ACID_IDS,
    MolecularSystem,
)

TOKEN_ALPHABET: tuple[str, ...] = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
    "Z",
    "B",
    "J",
    "U",
    "O",
    "-",
    "RNA_A",
    "RNA_G",
    "RNA_C",
    "RNA_U",
    "RNA_N",
    "DNA_A",
    "DNA_G",
    "DNA_C",
    "DNA_T",
    "DNA_N",
    "OTHER",
)
TOKEN_INDEX: dict[str, int] = {symbol: index for index, symbol in enumerate(TOKEN_ALPHABET)}
TOKEN_VOCAB_SIZE = len(TOKEN_ALPHABET)
TOKEN_GAP_INDEX = TOKEN_INDEX["-"]
TOKEN_OTHER_INDEX = TOKEN_INDEX["OTHER"]

_PROTEIN_TOKEN_INDEX = {symbol: TOKEN_INDEX[symbol] for symbol in "ACDEFGHIKLMNPQRSTVWYXZBJUO"}
_RNA_TOKEN_INDEX = {symbol: TOKEN_INDEX[f"RNA_{symbol}"] for symbol in "AGCU"}
_DNA_TOKEN_INDEX = {symbol: TOKEN_INDEX[f"DNA_{symbol}"] for symbol in "AGCT"}
_PROTEIN_TOKEN_SYMBOL_BY_COMPONENT = {
    **{component: symbol for symbol, component in _STANDARD_COMPONENTS["protein"].items()},
    "UNK": "X",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "SEC": "U",
    "PYL": "O",
}
_RNA_TOKEN_SYMBOL_BY_COMPONENT = {component: symbol for symbol, component in _STANDARD_COMPONENTS["rna"].items()}
_DNA_TOKEN_SYMBOL_BY_COMPONENT = {component: symbol for symbol, component in _STANDARD_COMPONENTS["dna"].items()}
_STANDARD_POLYMER_COMPONENTS = {
    ENTITY_PROTEIN: STANDARD_AMINO_ACID_IDS | {"UNK"},
    ENTITY_RNA: frozenset({"A", "C", "G", "U", "N"}),
    ENTITY_DNA: frozenset({"DA", "DC", "DG", "DT", "DN"}),
    ENTITY_DNA_RNA_HYBRID: frozenset({"A", "C", "G", "U", "N", "DA", "DC", "DG", "DT", "DN"}),
}
_POLYMER_TYPE_BY_ENTITY_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
    ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
}


def residue_token_index(symbol: str, polymer_type: str | None) -> int:
    r"""Resolve one aligned residue symbol in the canonical token vocabulary."""

    if symbol == "-":
        return TOKEN_GAP_INDEX
    if polymer_type == "protein":
        return _PROTEIN_TOKEN_INDEX.get(symbol, TOKEN_INDEX["X"])
    if polymer_type == "rna":
        return _RNA_TOKEN_INDEX.get(symbol, TOKEN_INDEX["RNA_N"])
    if polymer_type == "dna":
        return _DNA_TOKEN_INDEX.get(symbol, TOKEN_INDEX["DNA_N"])
    return TOKEN_OTHER_INDEX


def component_token_index(component_id: str, polymer_type: str | None) -> int:
    r"""Resolve one chemical component in the canonical token vocabulary."""

    if polymer_type == "protein":
        symbol = _PROTEIN_TOKEN_SYMBOL_BY_COMPONENT.get(component_id, "X")
    elif polymer_type == "rna":
        symbol = _RNA_TOKEN_SYMBOL_BY_COMPONENT.get(component_id, "N")
    elif polymer_type == "dna":
        symbol = _DNA_TOKEN_SYMBOL_BY_COMPONENT.get(component_id, "N")
    elif polymer_type == "dna_rna_hybrid":
        if component_id in _RNA_TOKEN_SYMBOL_BY_COMPONENT or component_id == "N":
            return component_token_index(component_id, "rna")
        if component_id in _DNA_TOKEN_SYMBOL_BY_COMPONENT or component_id == "DN":
            return component_token_index(component_id, "dna")
        return TOKEN_OTHER_INDEX
    else:
        return TOKEN_OTHER_INDEX
    return residue_token_index(symbol, polymer_type)


class TokenLayout(NamedTuple):
    r"""
    Canonical indices describing one structural token axis.

    ``atom_index[token_atom_ptr[i] : token_atom_ptr[i + 1]]`` contains the
    canonical atoms represented by token ``i``. ``token_atom_index`` is the
    canonical atom index for an atom token and ``-1`` for a component token;
    ``token_type_index`` indexes ``TOKEN_ALPHABET``.
    """

    token_component_index: Tensor
    token_atom_index: Tensor
    token_type_index: Tensor
    token_instance_index: Tensor
    token_entity_index: Tensor
    token_residue_index: Tensor
    token_symmetry_index: Tensor
    token_center_atom_index: Tensor
    token_atom_ptr: Tensor
    atom_index: Tensor


def tokenize_molecular_system(
    system: MolecularSystem,
    *,
    atomize_nonstandard_polymers: bool = True,
    expanded_components: Collection[int] = (),
    component_mask: Tensor | None = None,
    atom_mask: Tensor | None = None,
) -> TokenLayout:
    r"""
    Build a structural token view without changing canonical graph indices.

    Standard polymer components form one token unless explicitly expanded.
    Non-standard polymer components may either form one component token or one
    token per visible atom. Molecules always form one token per visible atom.

    Args:
        system: Canonical molecular graph.
        atomize_nonstandard_polymers: Whether non-standard polymer components
            form atom tokens. Set to ``False`` for an all-component polymer view.
        expanded_components: Canonical component indices that must form atom
            tokens, including otherwise standard polymer components.
        component_mask: Optional mask selecting canonical components.
        atom_mask: Optional mask over the canonical atom axis. Hydrogens are
            excluded independently of this mask.

    Returns:
        Canonical token-to-component and token-to-atom mappings.
    """

    visible_atom_mask = system.atomic_number != 1
    if atom_mask is not None:
        visible_atom_mask = visible_atom_mask & atom_mask

    component_atom_ptr = system.component_atom_ptr.detach().cpu().tolist()
    instance_component_ptr = system.instance_component_ptr.detach().cpu().tolist()
    instance_entity = system.instance_entity.detach().cpu().tolist()
    entity_type = system.entity_type.detach().cpu().tolist()
    component_symbol = system.component_symbol.detach().cpu().tolist()
    atom_symbol = system.atom_symbol.detach().cpu().tolist()
    selected_components = (
        [True] * len(component_symbol) if component_mask is None else component_mask.detach().cpu().tolist()
    )
    visible_atoms = visible_atom_mask.detach().cpu().tolist()
    expanded = set(expanded_components)

    token_component_index: list[int] = []
    token_atom_index: list[int] = []
    token_type_index: list[int] = []
    token_instance_index: list[int] = []
    token_entity_index: list[int] = []
    token_residue_index: list[int] = []
    token_symmetry_index: list[int] = []
    token_center_atom_index: list[int] = []
    token_atom_ptr = [0]
    atom_index: list[int] = []
    entity_copies: dict[int, int] = {}

    for instance_index, entity_index in enumerate(instance_entity):
        current_entity_type = int(entity_type[entity_index])
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        instance_token_start = len(token_component_index)
        instance_has_selected_component = False
        symmetry_index = entity_copies.get(entity_index, 0)
        entity_copies[entity_index] = symmetry_index + 1

        for component_index in range(component_start, component_stop):
            if not selected_components[component_index]:
                continue
            instance_has_selected_component = True
            residue_index = component_index - component_start
            atoms = tuple(
                atom
                for atom in range(
                    int(component_atom_ptr[component_index]),
                    int(component_atom_ptr[component_index + 1]),
                )
                if visible_atoms[atom]
            )
            if not atoms:
                if current_entity_type == ENTITY_MOLECULE:
                    continue
                component_id = system.symbols[int(component_symbol[component_index])]
                raise ValueError(
                    f"polymer component {component_id!r} at canonical component "
                    f"{component_index} has no visible heavy atoms"
                )

            component_id = system.symbols[int(component_symbol[component_index])]
            polymer_type = _POLYMER_TYPE_BY_ENTITY_TYPE.get(current_entity_type)
            type_index = component_token_index(component_id, polymer_type)
            standard = component_id in _STANDARD_POLYMER_COMPONENTS.get(
                current_entity_type,
                frozenset(),
            )
            atomized = (
                current_entity_type == ENTITY_MOLECULE
                or component_index in expanded
                or (atomize_nonstandard_polymers and not standard)
            )
            center_name = (
                "CA"
                if current_entity_type == ENTITY_PROTEIN
                else "C1'" if current_entity_type in {ENTITY_RNA, ENTITY_DNA, ENTITY_DNA_RNA_HYBRID} else None
            )
            by_name = {system.symbols[int(atom_symbol[atom])]: atom for atom in atoms}
            component_center = atoms[0] if center_name is None else by_name.get(center_name, atoms[0])
            groups = ((atom,) for atom in atoms) if atomized else (atoms,)
            for group in groups:
                token_component_index.append(component_index)
                token_atom_index.append(group[0] if atomized else -1)
                token_type_index.append(type_index)
                token_instance_index.append(instance_index)
                token_entity_index.append(entity_index)
                token_residue_index.append(residue_index)
                token_symmetry_index.append(symmetry_index)
                token_center_atom_index.append(group[0] if atomized else component_center)
                atom_index.extend(group)
                token_atom_ptr.append(len(atom_index))

        if (
            current_entity_type == ENTITY_MOLECULE
            and instance_has_selected_component
            and len(token_component_index) == instance_token_start
        ):
            raise ValueError(f"molecule at canonical instance {instance_index} has no visible heavy atoms")

    index = system.component_atom_ptr.new_tensor
    return TokenLayout(
        token_component_index=index(token_component_index),
        token_atom_index=index(token_atom_index),
        token_type_index=index(token_type_index),
        token_instance_index=index(token_instance_index),
        token_entity_index=index(token_entity_index),
        token_residue_index=index(token_residue_index),
        token_symmetry_index=index(token_symmetry_index),
        token_center_atom_index=index(token_center_atom_index),
        token_atom_ptr=index(token_atom_ptr),
        atom_index=index(atom_index),
    )


def select_tokens(layout: TokenLayout, selection: Tensor) -> TokenLayout:
    r"""Select a token view and rebuild its ragged token-to-atom mapping."""

    selection = selection.to(device=layout.token_atom_ptr.device, dtype=torch.int64)
    starts = layout.token_atom_ptr[selection]
    counts = layout.token_atom_ptr[selection + 1] - starts
    token_atom_ptr = torch.cat((layout.token_atom_ptr.new_zeros(1), counts.cumsum(0)))
    source_offset = torch.repeat_interleave(starts - token_atom_ptr[:-1], counts)
    source_index = torch.arange(source_offset.numel(), device=token_atom_ptr.device) + source_offset

    return TokenLayout(
        token_component_index=layout.token_component_index[selection],
        token_atom_index=layout.token_atom_index[selection],
        token_type_index=layout.token_type_index[selection],
        token_instance_index=layout.token_instance_index[selection],
        token_entity_index=layout.token_entity_index[selection],
        token_residue_index=layout.token_residue_index[selection],
        token_symmetry_index=layout.token_symmetry_index[selection],
        token_center_atom_index=layout.token_center_atom_index[selection],
        token_atom_ptr=token_atom_ptr,
        atom_index=layout.atom_index[source_index],
    )
