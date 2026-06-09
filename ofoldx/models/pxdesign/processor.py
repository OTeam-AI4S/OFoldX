r"""
Featurize structures and PxDesign JSON design inputs into the model's tensor contract.

Converts either a pre-tokenized structure or a PxDesign JSON input into a [`PxDesignInput`][]. On top of the
model-agnostic structural fields, this processor preserves the Protenix-coupled featurization that a generic
residue processor would lossily substitute: the AF3 ``distogram_rep_atom_mask``, the reference-space uid
(``ref_space_uid``), and per-token frames. It also restores the PxDesign design annotations
(``conditional_label`` / ``is_resolved``) rather than dropping them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.ccd import CcdProvider, default_ccd_provider
from ...data.structure import ResidueKey, Structure
from ...processors.registry import PROCESSORS
from ...processors.structure import StructureProcessor, default_residue_vocabulary
from ...processors.utils import StructuralFields, StructureInput, structure_from_input
from .io import DesignInput, read_pxdesign_json
from .model import PxDesignInput

_PXDESIGN_NAMES = frozenset({"xpb", "xpa", "rbb", "raa"})
_PXDESIGN_DESIGN_RESTYPE_IDS = {"xpb": 32, "xpa": 33, "rbb": 34, "raa": 35}

# AF3 SI 4.4 distogram representative-atom rule. Purines use C4, pyrimidines C2, unknown nucleotides C1'.
# Protein residues use CB except glycine and design ("xpb") tokens which use CA. Ligands are single-atom
# tokens and are all selected.
_PURINE_NAMES = frozenset({"DA", "DG", "A", "G"})
_PYRIMIDINE_NAMES = frozenset({"DC", "DT", "C", "U"})
_UNKNOWN_NUCLEOTIDE_NAMES = frozenset({"DN", "N"})

PxDesignProcessorInput = StructureInput | DesignInput | Mapping[str, Any]


@PROCESSORS.register("pxdesign")
class PxDesignProcessor(StructureProcessor):
    r"""
    Convert structure or PxDesign JSON inputs into [`PxDesignInput`][].

    The processor emits the PxDesign condition schema for structures that already contain condition/design
    tokens. Design positions can be supplied three ways, in priority order: a [`DesignInput`][] (or JSON
    path / dict) carrying per-atom ``conditional_label`` design annotations; residues renamed to the PxDesign
    ``xpb`` placeholder; or the ``design_residues`` selector. ``hotspot_residues`` sets PxDesign hotspot
    features when a JSON input does not already carry them.

    Args:
        single_size: Width of the single (token) representation the model expects.
        token_feature_size: Width of the per-token feature vector.
        atom_feature_size: Width of the per-atom reference feature vector.
        residue_vocabulary: Residue names in vocabulary order; defaults to the standard set.
        element_vocabulary: Element symbols for element features, or ``None`` for the default.
        element_channels: Number of element-embedding channels.
        atom_name_characters_count: Number of atom-name characters featurized per atom.
        atom_name_character_channels: Number of channels per atom-name character.
        ccd: Chemical Component Dictionary provider, or ``None`` for the default provider.
        reference_conformer_augment: Whether to randomly augment reference conformers.
        augment_seed: Optional seed for reference-conformer augmentation.
        ligand_atom_rename: Whether to rename ligand atoms to canonical names.
        materialize_missing_atoms: Whether to add placeholder atoms for missing ones.
        vocab_size: Restype vocabulary size, including PxDesign design placeholders.
        design_residues: ``(chain_id, residue_number)`` selectors marking design positions.
        hotspot_residues: ``(chain_id, residue_number)`` selectors marking hotspot positions.
        design_protein_token_id: Restype id assigned to selector-marked protein design tokens.
        restype_one_hot: Whether to emit restype features as one-hot instead of ids.
    """

    input_class = PxDesignInput

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> PxDesignProcessor:
        r"""
        Build a PxDesign processor from a model config.

        Reads the width fields (and the condition-embedder ``vocab_size``) off the model
        config, applies them as processor defaults, and lets ``kwargs`` override any of them.

        Args:
            config: A model config (or config-like object) exposing the PxDesign width fields.
            **kwargs: Overrides passed through to the processor constructor.

        Returns:
            A configured [`PxDesignProcessor`][ofoldx.models.pxdesign.processor.PxDesignProcessor].

        Examples:
            >>> from ofoldx.models.pxdesign.config import PxDesignConfig
            >>> processor = PxDesignProcessor.from_config(PxDesignConfig())  # doctest: +SKIP
        """

        condition_embedder = getattr(config, "condition_embedder", None)
        parameters: dict[str, Any] = {
            "single_size": int(_config_scalar(config, "single_size", 384)),
            "token_feature_size": int(_config_scalar(config, "token_feature_size", 449)),
            "atom_feature_size": int(_config_scalar(config, "atom_feature_size", 385)),
            "vocab_size": int(_config_scalar(condition_embedder, "vocab_size", 36)),
        }
        parameters.update(kwargs)
        return cls(**parameters)

    def __init__(
        self,
        *,
        single_size: int = 384,
        token_feature_size: int = 449,
        atom_feature_size: int = 385,
        residue_vocabulary: Sequence[str] | None = None,
        element_vocabulary: Sequence[str] | None = None,
        element_channels: int = 128,
        atom_name_characters_count: int = 4,
        atom_name_character_channels: int = 64,
        ccd: CcdProvider | None = None,
        reference_conformer_augment: bool = False,
        augment_seed: int | None = None,
        ligand_atom_rename: bool = False,
        materialize_missing_atoms: bool = False,
        vocab_size: int = 36,
        design_residues: Iterable[Sequence[Any]] = (),
        hotspot_residues: Iterable[Sequence[Any]] = (),
        design_protein_token_id: int = 32,
        restype_one_hot: bool = True,
    ) -> None:
        super().__init__(
            single_size=single_size,
            token_feature_size=token_feature_size,
            atom_feature_size=atom_feature_size,
            residue_vocabulary=default_residue_vocabulary() if residue_vocabulary is None else residue_vocabulary,
            element_vocabulary=element_vocabulary,
            element_channels=element_channels,
            atom_name_characters_count=atom_name_characters_count,
            atom_name_character_channels=atom_name_character_channels,
            ccd=ccd,
            reference_conformer_augment=reference_conformer_augment,
            augment_seed=augment_seed,
            ligand_atom_rename=ligand_atom_rename,
            materialize_missing_atoms=materialize_missing_atoms,
        )
        self.vocab_size = vocab_size
        self.design_residues = _normalize_residue_set(design_residues)
        self.hotspot_residues = _normalize_residue_set(hotspot_residues)
        self.design_protein_token_id = design_protein_token_id
        self.restype_one_hot = restype_one_hot

    def __call__(self, structure: PxDesignProcessorInput) -> PxDesignInput:  # type: ignore[override]
        r"""
        Return a batched PxDesign input for one structure or JSON design input.

        Args:
            structure: A parsed [`Structure`][ofoldx.data.structure.Structure], a structure file path, a
                [`DesignInput`][ofoldx.models.pxdesign.io.DesignInput], a PxDesign JSON file path, or a
                decoded JSON sample dict.

        Returns:
            Batched PxDesign model input with leading batch size 1.

        Examples:
            >>> processor = PxDesignProcessor()  # doctest: +SKIP
            >>> batch = processor("design.json")  # doctest: +SKIP
        """

        design_input = self._resolve_design_input(structure)
        parsed = (
            design_input.structure
            if design_input is not None
            else structure_from_input(cast(StructureInput, structure))
        )
        ccd = self._ccd if self._ccd is not None else default_ccd_provider()
        fields = self._structural_fields(parsed, ccd)
        # Materialization may have expanded the structure; re-align annotations to fields.structure.
        parsed = fields.structure
        residues = fields.residues
        token_mask = fields.token_mask

        conditional_label = self._conditional_label(design_input, parsed)
        design_token_mask = self._design_token_mask(fields, residues, conditional_label)
        condition_token_mask = token_mask & ~design_token_mask
        distogram_rep_atom_mask = _distogram_rep_atom_mask(parsed, fields, design_token_mask)
        condition_atom_mask = condition_token_mask[fields.atom_to_token] & fields.target_mask
        conditional_templ, conditional_templ_mask = _conditional_template(fields, condition_token_mask, design_input)
        hotspot = self._hotspot_feature(design_input, residues)
        is_resolved = self._is_resolved(design_input, fields)
        condition_coords = self._condition_coords(design_input, fields, condition_atom_mask)

        return PxDesignInput(
            token_mask=token_mask.unsqueeze(0),
            residue_index=torch.tensor(
                [residue.residue_number for residue in residues],
                dtype=torch.long,
            ).unsqueeze(0),
            chain_index=torch.tensor(
                [fields.chain_to_index[residue.chain_id] for residue in residues],
                dtype=torch.long,
            ).unsqueeze(0),
            entity_index=torch.tensor(fields.entity_values, dtype=torch.long).unsqueeze(0),
            symmetry_index=torch.zeros(1, fields.num_tokens, dtype=torch.long),
            token_index=torch.arange(fields.num_tokens, dtype=torch.long).unsqueeze(0),
            atom_features=self._atom_features(parsed, self._atom_names(parsed)).unsqueeze(0),
            atom_to_token=fields.atom_to_token.unsqueeze(0),
            atom_mask=fields.atom_mask.unsqueeze(0),
            reference_positions=fields.reference_positions.unsqueeze(0),
            reference_space_id=torch.tensor(fields.reference_space_values, dtype=torch.long).unsqueeze(0),
            reference_mask=fields.reference_mask.unsqueeze(0),
            reference_charges=fields.reference_charges.unsqueeze(0),
            restype=self._restype_features(residues, fields.residue_names, design_token_mask).unsqueeze(0),
            conditional_templ=conditional_templ.unsqueeze(0),
            conditional_templ_mask=conditional_templ_mask.unsqueeze(0),
            design_token_mask=design_token_mask.unsqueeze(0),
            condition_token_mask=condition_token_mask.unsqueeze(0),
            condition_atom_mask=condition_atom_mask.unsqueeze(0),
            condition_coords=condition_coords.unsqueeze(0),
            backbone_atom_indices=_backbone_atom_indices(fields, parsed).unsqueeze(0),
            plddt=torch.zeros(1, fields.num_tokens, dtype=torch.float32),
            hotspot=hotspot.unsqueeze(0),
            distogram_rep_atom_mask=distogram_rep_atom_mask.unsqueeze(0),
            frame_atom_index=torch.tensor(fields.frame_atom_index_values, dtype=torch.long)
            .reshape(fields.num_tokens, 3)
            .unsqueeze(0),
            frame_mask=torch.tensor(fields.frame_mask_values, dtype=torch.bool).unsqueeze(0),
            conditional_label=torch.from_numpy(conditional_label).unsqueeze(0),
            is_resolved=is_resolved.unsqueeze(0),
        )

    def _resolve_design_input(self, structure: PxDesignProcessorInput) -> DesignInput | None:
        r"""
        Return a [`DesignInput`][] when the input is a JSON design input, else ``None``.

        A plain structure or structure file path returns ``None`` and follows the legacy naming-based path.

        Args:
            structure: A structure, structure path, [`DesignInput`][], JSON path, or JSON dict.

        Returns:
            The parsed [`DesignInput`][] for JSON/mapping inputs, otherwise ``None``.
        """

        ccd = self._ccd if self._ccd is not None else None
        if isinstance(structure, DesignInput):
            return structure
        if isinstance(structure, Mapping):
            return read_pxdesign_json(structure, ccd=ccd)
        if isinstance(structure, (str, PathLike)) and str(structure).lower().endswith(".json"):
            return read_pxdesign_json(structure, ccd=ccd)
        return None

    def _processor_config(self) -> dict[str, object]:
        r"""
        Return the serializable processor configuration.

        Extends the base structure-processor config with the PxDesign-specific fields
        (vocabulary size, design/hotspot selectors, and restype encoding flags) so the
        processor round-trips through ``save``/``from_pretrained``.

        Returns:
            A dict of processor settings suitable for serialization.
        """

        config = super()._processor_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "design_residues": [list(value) for value in sorted(self.design_residues)],
                "hotspot_residues": [list(value) for value in sorted(self.hotspot_residues)],
                "design_protein_token_id": self.design_protein_token_id,
                "restype_one_hot": self.restype_one_hot,
            }
        )
        return config

    def _conditional_label(self, design_input: DesignInput | None, structure: Structure) -> np.ndarray:
        r"""
        Return the per-atom ``conditional_label`` (``True`` = condition, ``False`` = design).

        JSON inputs carry it directly; otherwise every atom is a condition atom unless its residue is a
        PxDesign ``xpb``-family design placeholder.

        Args:
            design_input: Optional parsed JSON design input.
            structure: The parsed structure whose atoms are labeled.

        Returns:
            A ``[N_atom]`` boolean array; ``True`` marks condition atoms.
        """

        if design_input is not None and design_input.conditional_label.shape[0] == len(structure.atoms):
            return design_input.conditional_label.astype(np.bool_)
        values = [atom.residue_name.lower() not in _PXDESIGN_NAMES for atom in structure.atoms]
        return np.asarray(values, dtype=np.bool_)

    def _design_token_mask(
        self,
        fields: StructuralFields,
        residues: tuple[ResidueKey, ...],
        conditional_label: np.ndarray,
    ) -> Tensor:
        r"""
        Return the per-token design mask, merging JSON design annotations with the legacy selectors.

        A token is a design token when any of its atoms is a design atom (from
        ``conditional_label``) or when the residue matches a ``design_residues`` selector.

        Args:
            fields: Structural fields carrying the atom-to-token mapping.
            residues: Per-token residue keys, in token order.
            conditional_label: Per-atom condition/design labels (``True`` = condition).

        Returns:
            A ``[N_token]`` boolean tensor; ``True`` marks design tokens.
        """

        atom_design = torch.from_numpy(~conditional_label)
        token_design = torch.zeros(fields.num_tokens, dtype=torch.bool)
        token_design[fields.atom_to_token[atom_design]] = True
        selector_values = [
            (residue.chain_id, int(residue.residue_number)) in self.design_residues for residue in residues
        ]
        return token_design | torch.tensor(selector_values, dtype=torch.bool)

    def _hotspot_feature(self, design_input: DesignInput | None, residues: tuple[ResidueKey, ...]) -> Tensor:
        r"""
        Return the per-token hotspot feature.

        Uses the JSON hotspot flags (reduced to per-residue) when present, otherwise the
        ``hotspot_residues`` selectors.

        Args:
            design_input: Optional parsed JSON design input.
            residues: Per-token residue keys, in token order.

        Returns:
            A ``[N_token]`` float tensor of hotspot indicators.
        """

        if design_input is not None and design_input.hotspot.any():
            # Reduce per-atom hotspot to per-token via first-atom order; residues carry the JSON flag.
            hotspot_by_residue = _per_residue_hotspot(design_input)
            values = [hotspot_by_residue.get((r.chain_id, int(r.residue_number)), False) for r in residues]
            return torch.tensor(values, dtype=torch.float32)
        values = [(residue.chain_id, int(residue.residue_number)) in self.hotspot_residues for residue in residues]
        return torch.tensor(values, dtype=torch.float32)

    def _is_resolved(self, design_input: DesignInput | None, fields: StructuralFields) -> Tensor:
        r"""
        Return the per-atom resolved mask (atoms with a real experimental coordinate).

        Args:
            design_input: Optional parsed JSON design input carrying the resolved flags.
            fields: Structural fields whose target mask is the fallback.

        Returns:
            A ``[N_atom]`` boolean tensor; ``True`` marks resolved atoms.
        """

        if design_input is not None and design_input.coord_from_cif_is_resolved.shape[0] == fields.num_atoms:
            return torch.from_numpy(design_input.coord_from_cif_is_resolved.astype(np.bool_))
        return fields.target_mask.to(dtype=torch.bool)

    def _condition_coords(
        self, design_input: DesignInput | None, fields: StructuralFields, condition_atom_mask: Tensor
    ) -> Tensor:
        r"""
        Return the ``[N_atom, 3]`` condition coords, zeroed outside resolved condition atoms.

        Prefers the JSON ``coord_from_cif`` coordinates when present, otherwise the
        structure's target positions.

        Args:
            design_input: Optional parsed JSON design input carrying condition coords.
            fields: Structural fields whose target positions are the fallback.
            condition_atom_mask: ``[N_atom]`` mask selecting condition atoms.

        Returns:
            A ``[N_atom, 3]`` float tensor with non-condition atoms zeroed.
        """

        if design_input is not None and design_input.coord_from_cif.shape[0] == fields.num_atoms:
            coords = torch.from_numpy(design_input.coord_from_cif.astype(np.float32))
        else:
            coords = fields.target_positions
        return coords * condition_atom_mask.to(dtype=coords.dtype).unsqueeze(-1)

    def _restype_features(
        self,
        residues: tuple[ResidueKey, ...],
        residue_names: dict[ResidueKey, str],
        design_token_mask: Tensor,
    ) -> Tensor:
        r"""
        Return the per-token restype features (ids or one-hot).

        Args:
            residues: Per-token residue keys, in token order.
            residue_names: Mapping from residue key to residue name.
            design_token_mask: ``[N_token]`` mask marking design tokens.

        Returns:
            Restype ids ``[N_token]`` when ``restype_one_hot`` is ``False``, otherwise a
            one-hot ``[N_token, vocab_size]`` float tensor.
        """

        restype_ids = self._restype_ids(residues, residue_names, design_token_mask)
        if not self.restype_one_hot:
            return restype_ids
        return F.one_hot(restype_ids, num_classes=self.vocab_size).to(dtype=torch.float32)

    def _restype_ids(
        self,
        residues: tuple[ResidueKey, ...],
        residue_names: dict[ResidueKey, str],
        design_token_mask: Tensor,
    ) -> Tensor:
        r"""
        Return the per-token restype ids.

        Assigns the PxDesign design placeholder ids to ``xpb``-family residues, the
        ``design_protein_token_id`` to other design tokens, and the vocabulary index
        otherwise.

        Args:
            residues: Per-token residue keys, in token order.
            residue_names: Mapping from residue key to residue name.
            design_token_mask: ``[N_token]`` mask marking design tokens.

        Returns:
            A ``[N_token]`` long tensor of restype ids clamped into vocabulary range.
        """

        values = []
        for index, residue in enumerate(residues):
            residue_name = residue_names[residue]
            design_id = _PXDESIGN_DESIGN_RESTYPE_IDS.get(residue_name.lower())
            if design_id is not None:
                values.append(design_id)
            elif bool(design_token_mask[index]):
                values.append(self.design_protein_token_id)
            else:
                values.append(self._residue_index(residue_name))
        return torch.tensor(values, dtype=torch.long).clamp(min=0, max=self.vocab_size - 1)

    def _atom_names(self, structure: Any) -> list[str]:
        r"""
        Return atom names used for atom-name character features.

        Uses the raw structure atom names unless ligand renaming is enabled, in which
        case it defers to the base processor's canonicalized names.

        Args:
            structure: The parsed structure whose atom names are read.

        Returns:
            A list of atom-name strings in atom order.
        """

        if not self._ligand_atom_rename:
            return [atom.name for atom in structure.atoms]
        return cast(list[str], super()._atom_names(structure))


def _distogram_rep_atom_mask(structure: Structure, fields: StructuralFields, design_token_mask: Tensor) -> Tensor:
    r"""
    Return the AF3 distogram representative-atom mask (``[N_atom]`` bool).

    Faithful port of Protenix ``add_distogram_rep_atom_mask``: CB for protein (CA for glycine and design
    tokens), C4 for purines, C2 for pyrimidines, C1' for unknown nucleotides, and all atoms for ligand/atom
    tokens. A design token is a protein residue flagged in ``design_token_mask`` (OFoldX carries the
    design status via ``conditional_label`` rather than the upstream ``xpb`` residue rename). Residues that
    end up without a representative atom fall back to the token representative atom so every token keeps
    exactly one distogram atom.

    Args:
        structure: The parsed structure supplying per-atom residue names.
        fields: Structural fields carrying token definitions and atom counts.
        design_token_mask: ``[N_token]`` mask marking design tokens.

    Returns:
        A ``[N_atom]`` boolean mask selecting one distogram representative atom per token.
    """

    from ...processors.utils import _STANDARD_RESIDUE_NAMES  # local import avoids widening the public surface

    standard = _STANDARD_RESIDUE_NAMES
    mask = torch.zeros(fields.num_atoms, dtype=torch.bool)
    for token_index, token in enumerate(fields.tokens):
        residue_name = structure.atoms[token.atom_indices[0]].residue_name
        upper = residue_name.upper()
        is_design = residue_name.lower() in _PXDESIGN_NAMES or bool(design_token_mask[token_index])
        is_protein = len(upper) == 3 and upper in standard
        if upper not in standard:
            # Non-standard / ligand token: every atom is its own distogram atom.
            for atom_index in token.atom_indices:
                mask[atom_index] = True
            continue
        if is_design and is_protein:
            _select_atom_named(structure, token.atom_indices, "CA", mask, token.representative_atom_index)
            continue
        target = _standard_distogram_atom(upper)
        _select_atom_named(structure, token.atom_indices, target, mask, token.representative_atom_index)
    return mask


def _standard_distogram_atom(residue_name: str) -> str:
    r"""
    Return the AF3 distogram representative-atom name for a standard residue.

    Applies the AF3 SI 4.4 rule: glycine uses ``CA``, purines ``C4``, pyrimidines
    ``C2``, unknown nucleotides ``C1'``, other three-letter protein residues ``CB``,
    and everything else falls back to ``C1'``.

    Args:
        residue_name: Upper-cased residue name.

    Returns:
        The representative atom name.

    Examples:
        >>> _standard_distogram_atom("GLY")
        'CA'
        >>> _standard_distogram_atom("ALA")
        'CB'
        >>> _standard_distogram_atom("DA")
        'C4'
    """

    if residue_name == "GLY":
        return "CA"
    if residue_name in _PURINE_NAMES:
        return "C4"
    if residue_name in _PYRIMIDINE_NAMES:
        return "C2"
    if residue_name in _UNKNOWN_NUCLEOTIDE_NAMES:
        return "C1'"
    if len(residue_name) == 3:  # standard protein residue
        return "CB"
    return "C1'"


def _select_atom_named(
    structure: Structure,
    atom_indices: tuple[int, ...],
    target_name: str,
    mask: Tensor,
    fallback_index: int,
) -> None:
    r"""
    Set the mask bit for the atom named ``target_name`` within a token, else the fallback.

    Args:
        structure: The parsed structure supplying atom names.
        atom_indices: Atom indices belonging to the token.
        target_name: Atom name to select.
        mask: ``[N_atom]`` boolean mask mutated in place.
        fallback_index: Atom index selected when no atom matches ``target_name``.
    """

    for atom_index in atom_indices:
        if structure.atoms[atom_index].name == target_name:
            mask[atom_index] = True
            return
    mask[fallback_index] = True


def _per_residue_hotspot(design_input: DesignInput) -> dict[tuple[str, int], bool]:
    r"""
    Reduce per-atom JSON hotspot flags to a per-residue mapping.

    Args:
        design_input: Parsed JSON design input with per-atom hotspot flags.

    Returns:
        A mapping from ``(chain_id, residue_number)`` to whether any of its atoms is a hotspot.
    """

    hotspot: dict[tuple[str, int], bool] = {}
    for atom, flag in zip(design_input.structure.atoms, design_input.hotspot):
        key = (atom.residue.chain_id, int(atom.residue.residue_number))
        hotspot[key] = hotspot.get(key, False) or bool(flag)
    return hotspot


def _conditional_template(
    fields: StructuralFields,
    condition_token_mask: Tensor,
    design_input: DesignInput | None,
) -> tuple[Tensor, Tensor]:
    r"""
    Build the condition-condition distance-bin template over resolved condition tokens.

    Condition coords come from the JSON ``coord_from_cif`` when available (so an unresolved condition
    residue is excluded), otherwise from the structure's target coords.

    Args:
        fields: Structural fields carrying token/atom mappings and target coords.
        condition_token_mask: ``[N_token]`` mask marking condition tokens.
        design_input: Optional parsed JSON design input carrying condition coords.

    Returns:
        A ``(conditional_templ, conditional_templ_mask)`` pair of ``[N_token, N_token]`` tensors:
        distance-bin ids (long) and the boolean mask of conditioned pairs.
    """

    num_tokens = fields.num_tokens
    conditional_templ = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
    conditional_templ_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
    if num_tokens == 0:
        return conditional_templ, conditional_templ_mask

    rep_index = fields.token_to_rep_atom.to(dtype=torch.long)
    if design_input is not None and design_input.coord_from_cif.shape[0] == fields.num_atoms:
        atom_coords = torch.from_numpy(design_input.coord_from_cif.astype(np.float32))
        atom_resolved = torch.from_numpy(design_input.coord_from_cif_is_resolved.astype(np.bool_))
    else:
        atom_coords = fields.target_positions
        atom_resolved = fields.target_mask.to(dtype=torch.bool)
    rep_coords = atom_coords[rep_index]
    rep_mask = atom_resolved[rep_index]
    valid = condition_token_mask & rep_mask
    if not bool(valid.any()):
        return conditional_templ, conditional_templ_mask

    coords = rep_coords[valid]
    distances = torch.cdist(coords, coords)
    boundaries = torch.linspace(start=2.0, end=22.0, steps=63, dtype=distances.dtype)
    bins = torch.sum(distances.unsqueeze(dim=-1) > boundaries, dim=-1).to(dtype=torch.long)
    indices = torch.nonzero(valid, as_tuple=True)[0]
    row, col = torch.meshgrid(indices, indices, indexing="ij")
    conditional_templ[row, col] = bins
    conditional_templ_mask[row, col] = True
    return conditional_templ, conditional_templ_mask


def _normalize_residue_set(values: Iterable[Sequence[Any]] | Mapping[Any, Any]) -> set[tuple[str, int]]:
    r"""
    Normalize residue selectors into a set of ``(chain_id, residue_number)`` tuples.

    Args:
        values: Iterable of selectors, or a mapping of ``chain_id -> residue_number`` as produced by
            ``chanfig`` for saved selector lists.

    Returns:
        A set of ``(str chain_id, int residue_number)`` tuples.

    Raises:
        ValueError: If any selector has fewer than two elements.

    Examples:
        >>> sorted(_normalize_residue_set([("A", 10), ("B", "20")]))
        [('A', 10), ('B', 20)]
    """

    normalized: set[tuple[str, int]] = set()
    if isinstance(values, Mapping):
        for chain_id, residue_number in values.items():
            normalized.add((str(chain_id), int(residue_number)))
        return normalized
    for value in values:
        if isinstance(value, Mapping):
            for chain_id, residue_number in value.items():
                normalized.add((str(chain_id), int(residue_number)))
            continue
        if len(value) < 2:
            raise ValueError("residue selectors must contain at least chain id and residue number")
        normalized.add((str(value[0]), int(value[1])))
    return normalized


def _backbone_atom_indices(fields: StructuralFields, structure: Structure) -> Tensor:
    r"""
    Return the ``N/CA/C/O`` backbone atom indices per token for ProteinMPNN.

    Args:
        fields: Structural fields carrying the atom-to-token mapping and token count.
        structure: The parsed structure supplying atom names.

    Returns:
        A ``[N_token, 4]`` long tensor of atom indices in ``(N, CA, C, O)`` slot order,
        with ``-1`` marking a missing backbone atom.
    """

    atom_name_to_slot = {"N": 0, "CA": 1, "C": 2, "O": 3}
    indices = torch.full((fields.num_tokens, 4), -1, dtype=torch.long)
    for atom_index, atom in enumerate(structure.atoms):
        slot = atom_name_to_slot.get(str(atom.name).strip().upper())
        if slot is None:
            continue
        token_index = int(fields.atom_to_token_values[atom_index])
        if 0 <= token_index < fields.num_tokens:
            indices[token_index, slot] = atom_index
    return indices


def _config_scalar(config: Any, name: str, default: Any) -> Any:
    r"""
    Read a scalar field from a config object or mapping, with a default.

    Treats an empty-mapping value (a chanfig placeholder) as absent and returns the default.

    Args:
        config: A config object, mapping, or ``None``.
        name: Field/key to read.
        default: Value returned when the field is absent or an empty mapping.

    Returns:
        The field value, or ``default``.

    Examples:
        >>> _config_scalar({"single_size": 128}, "single_size", 0)
        128
        >>> _config_scalar(None, "single_size", 384)
        384
    """

    if config is None:
        return default
    if isinstance(config, Mapping):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    if isinstance(value, Mapping) and not value:
        return default
    return value


__all__ = ["PxDesignProcessor"]
