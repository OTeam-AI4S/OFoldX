r"""
Feature packing helpers for embedding modules.

These concatenate the per-atom and per-token raw features into the single dense channel
layout the downstream embedders expect, validating the trailing channel widths so a
mis-shaped feature fails loudly at the pack site rather than deep inside a projection.
"""

from __future__ import annotations

import torch
from torch import Tensor


def pack_atom_features(
    atom_mask: Tensor,
    element_features: Tensor,
    atom_name_characters: Tensor,
    *,
    element_channels: int = 128,
    atom_name_characters_count: int = 4,
    atom_name_character_channels: int = 64,
) -> Tensor:
    r"""
    Pack atom features as mask, element, and atom-name character channels.

    Args:
        atom_mask: ``[..., N_atom]`` atom-validity mask.
        element_features: ``[..., N_atom, element_channels]`` element one-hot features.
        atom_name_characters: ``[..., N_atom, atom_name_characters_count, atom_name_character_channels]`` atom-name
            character features.
        element_channels: Expected element feature width.
        atom_name_characters_count: Expected number of encoded atom-name characters.
        atom_name_character_channels: Expected character feature width.

    Returns:
        ``[..., N_atom, 1 + element_channels + atom_name_characters_count * atom_name_character_channels]``
        packed atom features, on the device and dtype of ``element_features``.

    Raises:
        ValueError: If feature shapes do not match the configured widths or atom axis.

    Examples:
        >>> import torch
        >>> atom_mask = torch.ones(2, 3)
        >>> element_features = torch.zeros(2, 3, 4)
        >>> atom_name_characters = torch.zeros(2, 3, 2, 5)
        >>> packed = pack_atom_features(
        ...     atom_mask,
        ...     element_features,
        ...     atom_name_characters,
        ...     element_channels=4,
        ...     atom_name_characters_count=2,
        ...     atom_name_character_channels=5,
        ... )
        >>> packed.shape  # 1 (mask) + 4 (element) + 2 * 5 (atom-name) = 15
        torch.Size([2, 3, 15])
        >>> pack_atom_features(atom_mask, torch.zeros(2, 3, 8), atom_name_characters)
        Traceback (most recent call last):
            ...
        ValueError: element_features must have shape (..., N_atom, 128), got (2, 3, 8) for atom shape (2, 3)
    """

    atom_name_channels = atom_name_characters_count * atom_name_character_channels
    atom_shape = atom_mask.shape
    mask_channel = atom_mask.unsqueeze(-1)

    if element_features.shape[:-1] != atom_shape or element_features.shape[-1] != element_channels:
        raise ValueError(
            f"element_features must have shape (..., N_atom, {element_channels}), got "
            f"{tuple(element_features.shape)} for atom shape {tuple(atom_shape)}"
        )

    if atom_name_characters.shape[:-2] != atom_shape or atom_name_characters.shape[-2:] != (
        atom_name_characters_count,
        atom_name_character_channels,
    ):
        raise ValueError(
            "atom_name_characters must have shape "
            f"(..., N_atom, {atom_name_characters_count}, {atom_name_character_channels}), got "
            f"{tuple(atom_name_characters.shape)} for atom shape {tuple(atom_shape)}"
        )
    atom_name_features = atom_name_characters.reshape(*atom_shape, atom_name_channels)

    dtype = element_features.dtype
    return torch.cat(
        [
            mask_channel.to(device=element_features.device, dtype=dtype),
            element_features,
            atom_name_features.to(device=element_features.device, dtype=dtype),
        ],
        dim=-1,
    )


def pack_token_tail(
    residue_type: Tensor,
    profile: Tensor,
    deletion_mean: Tensor,
    *,
    residue_type_channels: int = 32,
    profile_channels: int = 32,
    deletion_mean_channels: int = 1,
) -> Tensor:
    r"""
    Pack token-tail features as residue type, profile, and deletion mean channels.

    Args:
        residue_type: ``[..., N_token, residue_type_channels]`` residue-type features.
        profile: ``[..., N_token, profile_channels]`` profile features.
        deletion_mean: ``[..., N_token, deletion_mean_channels]`` deletion-mean features.
        residue_type_channels: Expected residue-type feature width.
        profile_channels: Expected profile feature width.
        deletion_mean_channels: Expected deletion-mean feature width.

    Returns:
        ``[..., N_token, residue_type_channels + profile_channels + deletion_mean_channels]`` concatenated
        token-tail features, cast to the dtype and device of ``residue_type``.

    Raises:
        ValueError: If trailing widths or leading dimensions do not match.

    Examples:
        >>> import torch
        >>> residue_type = torch.zeros(2, 3, 4)
        >>> profile = torch.zeros(2, 3, 4)
        >>> deletion_mean = torch.zeros(2, 3, 1)
        >>> packed = pack_token_tail(
        ...     residue_type,
        ...     profile,
        ...     deletion_mean,
        ...     residue_type_channels=4,
        ...     profile_channels=4,
        ...     deletion_mean_channels=1,
        ... )
        >>> packed.shape  # 4 + 4 + 1 = 9
        torch.Size([2, 3, 9])
    """

    if any(
        (
            residue_type.shape[-1] != residue_type_channels,
            profile.shape[-1] != profile_channels,
            deletion_mean.shape[-1] != deletion_mean_channels,
        )
    ):
        raise ValueError(
            "residue_type, profile, and deletion_mean must end with "
            f"{residue_type_channels}, {profile_channels}, and {deletion_mean_channels} channels, got "
            f"{residue_type.shape[-1]}, {profile.shape[-1]}, and {deletion_mean.shape[-1]}"
        )
    if residue_type.shape[:-1] != profile.shape[:-1] or residue_type.shape[:-1] != deletion_mean.shape[:-1]:
        raise ValueError("residue_type, profile, and deletion_mean leading dimensions must match")
    return torch.cat([residue_type, profile.to(residue_type), deletion_mean.to(residue_type)], dim=-1)
