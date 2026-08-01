"""Coordinate-free occurrence addresses shared by readers and molecular graphs."""

from dataclasses import dataclass
from typing import TypeAlias

#: Molecular-system component occurrence as ``(instance_id, component_index)``.
ComponentReference: TypeAlias = tuple[str, int]


@dataclass(frozen=True)
class AtomSite:
    r"""
    An atom occurrence in a molecular system.

    Attributes:
        instance_id: Molecular instance containing the atom.
        component_index: Zero-based component occurrence within the instance.
        atom_id: Component-local atom identifier.
    """

    instance_id: str
    component_index: int
    atom_id: str
