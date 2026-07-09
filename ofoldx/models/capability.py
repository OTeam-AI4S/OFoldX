r"""
Capability protocols: the typed acts a model or pipeline performs.

A capability is a *verb*, not a class hierarchy. A provider (a model or a task pipeline) *is* a
[`Folder`][ofoldx.models.capability.Folder] because it implements ``fold`` -- structural typing, checked
with ``isinstance`` at runtime, never inheritance. The four house verbs are the single public name for
each act at every layer (facade, CLI, pipeline, docs); when a computer-science term and a biological term
tie, the biological one wins, so the acts are ``fold`` (predict structure), ``design`` (sequence/backbone),
``score`` (affinity/ranking), and ``pack`` (side chains). ``score`` and ``pack`` also match
Rosetta/PyRosetta vocabulary.

These protocols carry no implementation and impose no base class, so a family model file stays
self-contained; the Tier-0 verb facade and pipelines dispatch on them.

Examples:
    >>> from ofoldx.models.capability import Folder
    >>> class Toy:
    ...     def fold(self, inputs):
    ...         return inputs
    >>> isinstance(Toy(), Folder)
    True
    >>> isinstance(object(), Folder)
    False
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Folder(Protocol):
    r"""A provider that predicts a structure from sequence/complex inputs."""

    def fold(self, inputs: Any, /, **kwargs: Any) -> Any:
        r"""Predict a structure for ``inputs`` and return a structure-prediction result."""
        ...


@runtime_checkable
class Designer(Protocol):
    r"""A provider that designs sequences or backbones."""

    def design(self, inputs: Any, /, **kwargs: Any) -> Any:
        r"""Design sequence(s) or backbone(s) for ``inputs`` and return a design result."""
        ...


@runtime_checkable
class Scorer(Protocol):
    r"""A provider that scores a complex (e.g. binding affinity or ranking)."""

    def score(self, inputs: Any, /, **kwargs: Any) -> Any:
        r"""Score ``inputs`` and return the scored result."""
        ...


@runtime_checkable
class Packer(Protocol):
    r"""A provider that places side-chain atoms onto a backbone."""

    def pack(self, inputs: Any, /, **kwargs: Any) -> Any:
        r"""Pack side chains for ``inputs`` and return the packed result."""
        ...
