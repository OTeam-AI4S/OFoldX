r"""
Chi log-density projection for MpnnPacker predictions.

PDB carries one per-atom scalar, its B-factor column. [`structure_with_chi_density`][]
projects a prediction's per-residue chi log densities onto that field so an ordinary
[`ofoldx.io.write_structure`][] call visualizes them; the reinterpretation of the field
happens here, once, rather than inside a family-owned serializer.
"""

from __future__ import annotations

import torch

from ...data import ENTITY_PROTEIN
from ...data import residue_constants as rc
from ...data.molecular import AMINO_ACID_ONE_TO_THREE
from ...data.structure import Structure
from ...results import PackPrediction

#: Rigid group index of the frame each chi angle rotates.
_FIRST_CHI_RIGID_GROUP = 4


def structure_with_chi_density(prediction: PackPrediction) -> Structure:
    r"""
    Return the packed structure with per-atom chi log density in ``b_factor``.

    When chi metrics are supplied, each resolved side-chain atom takes the log density
    of the chi angle whose rigid group it belongs to. Backbone atoms, non-protein atoms,
    unscored side-chain atoms and atoms without coordinates have ``b_factor_mask=False``.
    For written atoms, [`write_structure`][ofoldx.io.structure.write_structure] represents
    this missing score as a blank PDB B-factor field or a mmCIF ``.`` value. PDB rejects
    scores that its field with two decimal places would round.

    Args:
        prediction: Packed structure and its per-residue chi metrics.

    Returns:
        The prediction's structure with ``b_factor`` and ``b_factor_mask`` replaced.
            If no chi metrics are supplied, returns the original structure unchanged.
    """

    structure = prediction.structure
    system = structure.system

    density_by_group = {}
    for chi_index in range(4):
        metric = prediction.metrics.get(f"chi{chi_index + 1}_log_density")
        if metric is not None:
            density_by_group[chi_index + _FIRST_CHI_RIGID_GROUP] = dict(zip(metric.axis, metric.values, strict=True))
    if not density_by_group:
        return structure

    amino_acid_index = {
        AMINO_ACID_ONE_TO_THREE[code]: index for index, code in enumerate(rc.AMINO_ACID_ONE_LETTER_CODES)
    }
    component_atom_ptr = system.component_atom_ptr.detach().cpu()
    component_symbol = system.component_symbol.detach().cpu()
    atom_symbol = system.atom_symbol.detach().cpu()
    instance_component_ptr = system.instance_component_ptr.detach().cpu()
    instance_entity = system.instance_entity.detach().cpu()
    entity_type = system.entity_type.detach().cpu()
    component_instance = torch.repeat_interleave(
        torch.arange(instance_component_ptr.numel() - 1),
        instance_component_ptr[1:] - instance_component_ptr[:-1],
    )

    b_factor = torch.zeros_like(structure.b_factor)
    b_factor_mask = torch.zeros_like(structure.b_factor_mask)
    resolved = structure.coordinate_mask
    for component_index in range(component_symbol.numel()):
        component_id = system.symbols[int(component_symbol[component_index])]
        index = amino_acid_index.get(component_id)
        if index is None:
            continue
        instance_index = int(component_instance[component_index])
        entity_index = int(instance_entity[instance_index])
        if int(entity_type[entity_index]) != ENTITY_PROTEIN:
            continue
        reference = system.component_reference(component_index)
        atom14_names = rc.AMINO_ACID_ATOM14_NAMES[component_id]
        start = int(component_atom_ptr[component_index])
        stop = int(component_atom_ptr[component_index + 1])
        for atom_index in range(start, stop):
            if not bool(resolved[atom_index]):
                continue
            atom_id = system.symbols[int(atom_symbol[atom_index])]
            if atom_id not in atom14_names:
                continue
            rigid_group = int(rc.AMINO_ACID_ATOM14_TO_RIGID_GROUP[index, atom14_names.index(atom_id)])
            density = density_by_group.get(rigid_group, {}).get(reference)
            if density is None:
                continue
            b_factor[atom_index] = float(density)
            b_factor_mask[atom_index] = True

    return structure.replace(  # type: ignore[attr-defined]
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
    )
