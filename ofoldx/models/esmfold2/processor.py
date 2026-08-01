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
ESMFold2 input processing, written against the shared preparation contract.

Everything that produces the model's inputs is OFoldX's own (layout, gating, MSA pairing, reference conformers); this
module keeps only the ESMFold2 featurisation of those inputs, which matches upstream for identical inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from abnumber import Chain, ChainParseError
from multimolecule import ProteinTokenizer
from numpy.typing import NDArray
from torch import Tensor

from ...data.molecular import ELEMENTS, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, TokenLayout, token_classes
from ...errors import RequestRef, fail
from ...msa import MSA, MsaCollection, MsaRows, msa_rows, pair_msas
from ...processor import Processor
from ...provenance import Records
from ..layout import LayoutTable, model_tokens, slot_map, token_layout, token_representatives
from ..preparation import (
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    SlotMap,
    component_records,
    gate,
    missing_reference,
    reference_conformers,
)
from ..processing import project_model_features
from ..tensors import to_tensors
from .config import EsmFold2ExperimentalProcessorConfig, EsmFold2ProcessorConfig
from .model import EsmFold2Input

if TYPE_CHECKING:
    from ...model import Model
    from .model import EsmFold2ExperimentalModel, EsmFold2Model

_PROTEIN, _RNA, _DNA = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA

ESMFOLD2_LAYOUT = LayoutTable(
    family="esmfold2",
    monomer_types={_PROTEIN: 0, _DNA: 1, _RNA: 2},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=True,
    centre_atom={_PROTEIN: "CA", _DNA: "C1'", _RNA: "C1'"},
    frame_atoms={},
    representative={
        **{TOKEN_INDEX[token]: "C4" for token in ("RNA_A", "RNA_G", "DNA_A", "DNA_G")},
        **{TOKEN_INDEX[token]: "C1'" for token in ("RNA_N", "DNA_N")},
    },
    representative_default={_PROTEIN: "CB", _DNA: "C2", _RNA: "C2"},
)
ESMFOLD2_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
_NONPOLYMER = 3

# Binder-design prompts use amino-acid letters for fixed positions and ``#`` for
# mutable positions. Antibody CDRs use Chothia numbering.
_MUTABLE_TOKEN = "#"
_ANTIBODY_CDR_SCHEME = "chothia"


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class EsmFold2Processor(Processor):
    r"""
    Prepare one structure for ESMFold2.

    Args:
        config: MSA-processing options. Defaults to ``EsmFold2ProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = EsmFold2ProcessorConfig
    config: EsmFold2ProcessorConfig

    def __init__(
        self,
        config: EsmFold2ProcessorConfig | None = None,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = self.config_class() if config is None else self.config_class(config)
        if seed is not None:
            self.config.seed = seed
        super().__init__(device=device, dtype=dtype)
        self.tokenizer = ProteinTokenizer()

    @property
    def seed(self) -> int:
        r"""Seed used to embed missing reference conformers."""
        return self.config.seed

    @seed.setter
    def seed(self, seed: int) -> None:
        self.config.seed = seed

    def validate_model_input_layout(self, model: Model) -> None:
        r"""Require the Processor and folding graph to agree on MSA conditioning."""

        model = cast("EsmFold2Model", model)
        if self.config.msa_conditioning != model.config.msa_encoder.enabled:
            fail("value_invalid", RequestRef("processor.msa_conditioning", str(self.config.msa_conditioning)))

    def prepare_input(
        self, structure: Structure, *, msa: Mapping[str, MSA] | MsaCollection | None = None
    ) -> PreparedInput[EsmFold2Input]:
        r"""
        Prepare one structure: layout, slots, gating, reference conformers, MSA rows and the ESMFold2 tensors. The
        scope is every atom in a new frame.

        Args:
            structure: The target; its chemistry is consumed as read (no library).
            msa: Protein MSA pools by instance (paired by ``pair_msas``), an exact plan, or ``None``. A checkpoint
                without MSA conditioning refuses it (``model_capacity``).
        """

        if msa is not None and not self.config.msa_conditioning:
            fail("model_capacity", RequestRef("msa", "a single-sequence checkpoint"))
        system = structure.system
        layout = token_layout(system, ESMFOLD2_LAYOUT)
        slots = slot_map(system, layout, ESMFOLD2_LAYOUT)
        accepted = gate(structure, slots, ESMFOLD2_GATING)
        references, sources, embedded = reference_conformers(
            structure, slots, embed=missing_reference(system), seed=self.seed
        )
        tokens, losses = model_tokens(system, layout)
        features = _features(
            system,
            slots,
            tokens,
            references,
            _msa_rows(msa, system, layout),
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=self.dtype,
        )
        conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
        conformers[_owner(system.component_atom_ptr)[slots.atom]] = sources
        return PreparedInput(
            features,
            PreparationContext(
                structure,
                slots,
                RegenerationScope.all(system),
                PreparationRecord(
                    Records.concat((component_records("encoding_loss", losses.component), accepted, embedded)),
                    conformers,
                ),
                spec=self.preparation_spec,
            ),
        )


class EsmFold2ExperimentalProcessor(EsmFold2Processor):
    r"""
    Prepare one structure for the experimental binder-design graphs.

    Args:
        config: Design-time MSA-processing options. Defaults to ``EsmFold2ExperimentalProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = EsmFold2ExperimentalProcessorConfig
    config: EsmFold2ExperimentalProcessorConfig

    def __init__(
        self,
        config: EsmFold2ExperimentalProcessorConfig | None = None,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config, seed=seed, device=device, dtype=dtype)

    def validate_model_input_layout(self, model: Model) -> None:
        r"""Require the Processor and experimental graph to agree on MSA inputs."""

        model = cast("EsmFold2ExperimentalModel", model)
        if self.config.msa_conditioning != model.config.use_msa_features:
            fail("value_invalid", RequestRef("processor.msa_conditioning", str(self.config.msa_conditioning)))


def _msa_rows(
    msa: Mapping[str, MSA] | MsaCollection | None, system: MolecularSystem, layout: TokenLayout
) -> MsaRows | None:
    r"""
    Pools are paired by the one OFoldX rule; an exact plan is used as given. Upstream conditions on protein
    alignments only: a row letter read as a nucleic class is ``model_capacity``.
    """

    if msa is None:
        return None
    plan = msa if isinstance(msa, MsaCollection) else pair_msas(msa, system, layout)
    if plan is None:
        return None
    rows = msa_rows(plan, system, layout)
    if (CLASS_MONOMER[rows.token[1:]] > _PROTEIN).any():
        fail("model_capacity", RequestRef("msa", "a nucleic alignment"))
    return rows


def _msa_features(rows: MsaRows | None, num_tokens: int) -> dict[str, Tensor]:
    if rows is None:
        return {"deletion_mean": torch.zeros(num_tokens)}
    token = torch.from_numpy(rows.token.astype(np.int64))
    deletions = torch.from_numpy(rows.deletions.astype(np.float32))
    deletion_value = (math.pi / 2) * torch.atan(deletions / 3)
    return {
        "msa_tokens": token,
        "has_deletion": deletions > 0,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_value.mean(dim=0),
        "msa_mask": torch.ones_like(token, dtype=torch.bool),
    }


def _atom_names(system: MolecularSystem, atoms: NDArray[np.int64]) -> list[str]:
    r"""
    Atom names in four characters; a longer or non-printable name becomes its element and a per-instance count, cut to
    four characters.
    """
    names = system.text("atom", system.atom_symbol[atoms]).tolist()
    instance = _owner(system.instance_component_ptr)[_owner(system.component_atom_ptr)[atoms]].tolist()
    elements = system.atomic_number[atoms].tolist()
    counts: dict[tuple[int, str], int] = {}
    encoded = []
    for name, owner, number in zip(names, instance, elements):
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = ELEMENTS[number].upper()
            counts[owner, element] = counts.get((owner, element), 0) + 1
            name = f"{element}{counts[owner, element]}"[:4]
        encoded.append(name)
    return encoded


def _token_bonds(system: MolecularSystem, layout: TokenLayout) -> Tensor:
    r"""``[N, N, 1]``: 1 between two tokens any connection joins, whatever its kind, order or role."""
    first, second = system.arrays()["edge_index"].astype(np.int64)
    a, b = layout.atom_token[first], layout.atom_token[second]
    linked = (a >= 0) & (b >= 0) & (a != b)
    bonds = torch.zeros(len(layout.token_component), len(layout.token_component))
    bonds[torch.from_numpy(a[linked]), torch.from_numpy(b[linked])] = 1
    bonds[torch.from_numpy(b[linked]), torch.from_numpy(a[linked])] = 1
    return bonds.unsqueeze(-1)


def _language_model(
    system: MolecularSystem, layout: TokenLayout, molecule_type: NDArray[np.int64], tokenizer: ProteinTokenizer
) -> dict[str, Tensor]:
    r"""
    One isolated ESMC row per instance holding a protein token (its whole sequence; a component of no protein class
    reads upstream's unknown residue ``X``) and, per protein token, its flattened row position.
    """

    pointer = system.instance_component_ptr
    token_instance = _owner(pointer)[layout.token_component]
    proteins = np.unique(token_instance[molecule_type == 0])
    classes = token_classes(system).token
    width = int(np.diff(pointer)[proteins].max()) + 2 if len(proteins) else 2
    tokens = torch.full((max(len(proteins), 1), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros(tokens.shape, dtype=torch.bool)
    rows = [[tokenizer.bos_token_id, tokenizer.eos_token_id]] if not len(proteins) else []
    for instance in proteins.tolist():
        letters = [
            TOKENS[k] if CLASS_MONOMER[k] == _PROTEIN else "X"
            for k in classes[pointer[instance] : pointer[instance + 1]]
        ]
        rows.append([tokenizer.bos_token_id, *tokenizer.convert_tokens_to_ids(letters), tokenizer.eos_token_id])
    for row, ids in enumerate(rows):
        tokens[row, : len(ids)] = torch.tensor(ids)
        mask[row, : len(ids)] = True
    row_of = np.full(len(pointer) - 1, -1, dtype=np.int64)
    row_of[proteins] = np.arange(len(proteins))
    has_esmc = molecule_type == 0
    position = row_of[token_instance] * width + layout.token_component - pointer[token_instance] + 1
    return {
        "language_model_tokens": tokens,
        "language_model_mask": mask,
        "token_to_esmc": torch.from_numpy(np.where(has_esmc, position, 0)),
        "token_has_esmc": torch.from_numpy(has_esmc),
    }


def _features(
    system: MolecularSystem,
    slots: SlotMap,
    tokens: NDArray[np.int64],
    references: NDArray[np.float64],
    rows: MsaRows | None,
    *,
    tokenizer: ProteinTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> EsmFold2Input:
    r"""The ESMFold2 tensors of one prepared task (upstream featurisation over the slot axis)."""
    s = system.arrays()
    layout = slots.layout
    atom = slots.atom
    num_atoms, num_tokens = len(atom), len(layout.token_component)
    atom_component = _owner(s["component_atom_ptr"])
    pointer = s["instance_component_ptr"]
    token_instance = _owner(pointer)[layout.token_component]
    entity = s["instance_entity"]
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    molecule_type = np.array(
        [ESMFOLD2_LAYOUT.monomer_types.get(MonomerType(m), _NONPOLYMER) for m in CLASS_MONOMER[tokens].tolist()],
        dtype=np.int64,
    )
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[atom] = np.arange(num_atoms)
    names = _atom_names(system, atom)
    canonical = to_tensors(system, ("atomic_number", "formal_charge"), device=torch.device("cpu"))
    tensors: dict[str, Tensor] = {
        "token_index": torch.arange(num_tokens),
        "residue_index": torch.from_numpy(layout.token_component - pointer[token_instance]),
        "chain_index": torch.from_numpy(token_instance),
        "symmetry_index": torch.from_numpy(copy[token_instance]),
        "entity_index": torch.from_numpy(entity[token_instance].astype(np.int64) + 1),
        "molecule_type": torch.from_numpy(molecule_type),
        "residue_type_ids": torch.from_numpy(tokens),
        "token_bonds": _token_bonds(system, layout),
        "reference_positions": torch.from_numpy(references),
        "reference_elements": canonical["atomic_number"][torch.from_numpy(atom)].long(),
        "reference_charges": canonical["formal_charge"][torch.from_numpy(atom)].double(),
        "reference_atom_name_characters": torch.tensor(
            [[ord(c) - 32 if c != " " else 0 for c in name.ljust(4)] for name in names], dtype=torch.long
        ).reshape(num_atoms, 4),
        "reference_space_id": torch.from_numpy(atom_component[atom]),
        "atom_to_token": torch.from_numpy(slots.token),
        "token_representative_atom_index": torch.from_numpy(
            slot_of[token_representatives(system, layout, ESMFOLD2_LAYOUT)]
        ),
        **_msa_features(rows, num_tokens),
    }
    language_model = _language_model(system, layout, molecule_type, tokenizer)
    tensors["token_has_esmc"] = language_model.pop("token_has_esmc")
    tensors["token_to_esmc"] = language_model.pop("token_to_esmc")
    placed = {
        name: (
            project_model_features(
                value, torch.ones_like(value, dtype=torch.bool), dtype=dtype, feature=f"ESMFold2 prepared {name}"
            ).to(device=device)
            if value.is_floating_point()
            else value.to(device=device)
        )
        for name, value in tensors.items()
    }
    return EsmFold2Input(  # type: ignore[call-arg]
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        **{name: value.to(device=device) for name, value in language_model.items()},
    )


# Antibody Fv prompts use ANARCI Chothia numbering. CDR-H1/H2/H3 and
# CDR-L1/L2/L3 are mutable; framework positions remain fixed.


def _antibody_domains(sequence: str) -> list[Chain]:
    r"""Number antibody domains with the released ANARCI workflow."""
    try:
        return Chain.multiple_domains(
            sequence,
            scheme=_ANTIBODY_CDR_SCHEME,
            cdr_definition=_ANTIBODY_CDR_SCHEME,
            allowed_species=None,
            use_anarcii=False,
        )
    except FileNotFoundError as error:
        if error.filename != "hmmscan":
            raise
        raise FileNotFoundError(
            "ANARCI requires HMMER 3: install HMMER and make its hmmscan executable available on PATH."
        ) from error


def is_antibody_sequence(sequence: str) -> bool:
    r"""
    Check whether a sequence contains only antibody variable domains.

    Args:
        sequence: Amino-acid sequence. Mutable ``#`` positions are treated as alanine
            during numbering.

    Returns:
        ``True`` when numbering succeeds and every detected domain is a heavy or light
            antibody variable domain; otherwise ``False``.
    """

    resolved = sequence.replace(_MUTABLE_TOKEN, "A")
    try:
        chains = _antibody_domains(resolved)
    except ChainParseError:
        return False
    valid_chain_types = {"H", "K", "L"}
    return all(chain.chain_type in valid_chain_types for chain in chains)


def antibody_cdr_indices(sequence: str) -> list[int]:
    r"""
    Return the Chothia CDR positions in an antibody Fv sequence.

    Args:
        sequence: Antibody Fv amino-acid sequence.

    Returns:
        Zero-based sequence indices for CDR1, CDR2, and CDR3 of each variable domain,
            with the heavy domain listed before the light domain when both are present.
    """

    chains = _antibody_domains(sequence)
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = sequence.find(cdr)
            if start < 0:
                fail("value_invalid", RequestRef("sequence", sequence), cdr=cdr)
            indices.extend(range(start, start + len(cdr)))
    return indices


def antibody_cdr_prompt(fv_sequence: str) -> str:
    r"""
    Mark the CDR residues of an antibody Fv sequence as designable.

    Args:
        fv_sequence: Antibody Fv amino-acid sequence.

    Returns:
        A sequence-length prompt with each Chothia CDR residue replaced by ``#`` and
            each framework residue preserved.
    """

    cdr_indices = set(antibody_cdr_indices(fv_sequence))
    return "".join(_MUTABLE_TOKEN if i in cdr_indices else residue for i, residue in enumerate(fv_sequence))
