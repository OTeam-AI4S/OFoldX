from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data import MSA, MSACollection, Template, TemplateAlignment
from ofoldx.data.molecular import (
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    PolymerSequence,
)
from ofoldx.data.structure import Structure

# §3.6 (frozen): evidence records hold no PolymerSequence and no registry. Entity and
# instance identities are plain strings that close over the primary MolecularSystem;
# they carry no Python object identity. The tests bind IDs as bare strings accordingly.


def _msa(entity_id: str = "a") -> MSA:
    r"""A width-3 MSA: three unpaired rows (row 0 the ungapped query) + query-only paired pool."""

    unpaired = ("AGV", "AG-", "A-V")
    return MSA(
        entity_id=entity_id,
        unpaired_sequences=unpaired,
        unpaired_deletion_counts=np.zeros((3, 3), dtype=np.uint32),
        paired_sequences=(unpaired[0],),
        paired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
    )


def _plan() -> tuple[np.ndarray, np.ndarray]:
    r"""A valid two-column plan: first row selects the paired query, later rows the unpaired pool."""

    row_pairing = np.array([[0, 0], [1, 2], [2, -1]], dtype=np.int64)
    row_is_paired = np.array([[True, True], [False, False], [False, False]], dtype=np.bool_)
    return row_pairing, row_is_paired


def _residue(component_id: str) -> ChemicalComponent:
    return ChemicalComponent(component_id, atoms=(ChemicalAtom("CA", "C"),), bonds=frozenset())


def _template_structure(*, instance_id: str = "TMPL", entity_id: str = "tmpl_entity") -> Structure:
    r"""A minimal two-residue polymer instance, coordinates unobserved-but-present."""

    entity = PolymerSequence("protein", (_residue("R0"), _residue("R1")))
    system = MolecularSystem(entities={entity_id: entity}, instances={instance_id: entity_id})
    sites = tuple(site for site, _ in system.iter_atoms())
    return Structure(
        system,
        sites,
        np.zeros((len(sites), 3), dtype=np.float32),
        np.ones(len(sites), dtype=np.bool_),
    )


def test_msa_keeps_separate_pools_and_checks_shape_and_query_agreement() -> None:
    msa = _msa("a")

    # entity_id is a bare string, not an object reference.
    assert msa.entity_id == "a"
    assert isinstance(msa.entity_id, str)
    # Row 0 of each pool is the identical ungapped query.
    assert msa.unpaired_sequences[0] == "AGV"
    assert msa.paired_sequences == ("AGV",)
    assert msa.unpaired_deletion_counts.shape == (3, 3)
    assert msa.unpaired_deletion_counts.dtype == np.uint32
    assert msa.paired_deletion_counts.shape == (1, 3)

    with pytest.raises(ValueError, match="query width"):
        MSA(
            entity_id="a",
            unpaired_sequences=("AGV", "AG"),
            unpaired_deletion_counts=np.zeros((2, 3), dtype=np.uint32),
            paired_sequences=("AGV",),
            paired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
        )
    with pytest.raises(ValueError, match="query rows must be identical"):
        MSA(
            entity_id="a",
            unpaired_sequences=("AGV",),
            unpaired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
            paired_sequences=("VGA",),
            paired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
        )
    with pytest.raises(ValueError, match="query row must be ungapped"):
        MSA(
            entity_id="a",
            unpaired_sequences=("A-V",),
            unpaired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
            paired_sequences=("A-V",),
            paired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
        )
    with pytest.raises(ValueError, match="unpaired_deletion_counts must have shape"):
        MSA(
            entity_id="a",
            unpaired_sequences=("AGV",),
            unpaired_deletion_counts=np.zeros((1, 2), dtype=np.uint32),
            paired_sequences=("AGV",),
            paired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
        )
    with pytest.raises(ValueError, match="dtype uint32"):
        MSA(
            entity_id="a",
            unpaired_sequences=("AGV",),
            unpaired_deletion_counts=np.zeros((1, 3), dtype=np.int64),
            paired_sequences=("AGV",),
            paired_deletion_counts=np.zeros((1, 3), dtype=np.uint32),
        )


def test_msa_collection_columns_align_and_broadcast_by_unique_entity() -> None:
    row_pairing, row_is_paired = _plan()
    collection = MSACollection((_msa("a"), _msa("b")), row_pairing, row_is_paired)

    # One column per MSA; broadcast mode carries no explicit instance binding.
    assert len(collection.msas) == collection.row_pairing.shape[1]
    assert collection.target_instance_ids is None
    # The first plan row selects query row zero from the paired pool in every column.
    assert list(collection.row_pairing[0]) == [0, 0]
    assert bool(collection.row_is_paired[0].all())

    # Broadcast columns must bind pairwise-unique entity IDs.
    with pytest.raises(ValueError, match="pairwise-unique entity_id"):
        MSACollection((_msa("a"), _msa("a")), row_pairing, row_is_paired)
    # A gap cell (-1) can never be marked paired.
    with pytest.raises(ValueError, match="gap cell cannot be marked paired"):
        MSACollection(
            (_msa("a"), _msa("b")),
            np.array([[0, 0], [1, -1]], dtype=np.int64),
            np.array([[True, True], [False, True]], dtype=np.bool_),
        )
    # The first plan row must select query row zero everywhere.
    with pytest.raises(ValueError, match="first plan row must select query row zero"):
        MSACollection(
            (_msa("a"), _msa("b")),
            np.array([[1, 0], [1, 2], [2, -1]], dtype=np.int64),
            row_is_paired,
        )


def test_msa_collection_explicit_instances_bind_one_to_one_and_are_unique() -> None:
    row_pairing, row_is_paired = _plan()
    # Two independently declared copies of one entity may carry different MSAs, bound
    # to exact instances; equal entity IDs never share evidence in explicit mode.
    collection = MSACollection(
        (_msa("a"), _msa("a")),
        row_pairing,
        row_is_paired,
        target_instance_ids=("a_1", "a_2"),
    )
    assert collection.target_instance_ids == ("a_1", "a_2")

    with pytest.raises(ValueError, match="pairwise unique"):
        MSACollection(
            (_msa("a"), _msa("a")),
            row_pairing,
            row_is_paired,
            target_instance_ids=("a_1", "a_1"),
        )
    with pytest.raises(ValueError, match="one entry per MSA"):
        MSACollection(
            (_msa("a"), _msa("a")),
            row_pairing,
            row_is_paired,
            target_instance_ids=("a_1",),
        )


def test_template_alignment_requires_increasing_zero_based_residue_pairs() -> None:
    alignment = TemplateAlignment("tgt", None, "TMPL", ((0, 0), (2, 1)))

    assert alignment.target_entity_id == "tgt"
    assert alignment.target_instance_id is None
    assert alignment.residue_pairs == ((0, 0), (2, 1))

    with pytest.raises(ValueError, match="strictly increasing"):
        TemplateAlignment("tgt", None, "TMPL", ((1, 0), (0, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        TemplateAlignment("tgt", None, "TMPL", ((-1, 0),))
    with pytest.raises(ValueError, match="residue_pairs must not be empty"):
        TemplateAlignment("tgt", None, "TMPL", ())
    with pytest.raises(ValueError, match="non-empty or None"):
        TemplateAlignment("tgt", "", "TMPL", ((0, 0),))


def test_template_binds_alignments_to_polymer_instances_in_its_own_frame() -> None:
    structure = _template_structure(instance_id="TMPL", entity_id="tmpl_entity")
    template = Template(
        structure,
        (TemplateAlignment("tgt", None, "TMPL", ((0, 0), (1, 1))),),
    )

    assert template.structure is structure
    assert template.alignments[0].template_instance_id == "TMPL"

    # A template component index beyond the template polymer length is rejected.
    with pytest.raises(ValueError, match="outside the template instance"):
        Template(structure, (TemplateAlignment("tgt", None, "TMPL", ((0, 0), (1, 5))),))
    # template_instance_id must name an instance in this Structure's own system.
    with pytest.raises(KeyError):
        Template(structure, (TemplateAlignment("tgt", None, "missing", ((0, 0),)),))
    with pytest.raises(ValueError, match="alignments must not be empty"):
        Template(structure, ())
