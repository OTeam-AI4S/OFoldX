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

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.chemistry.definitions import StereoBasis
from ofoldx.chemistry.rdkit import (
    EmbeddingFailed,
    SymmetryKind,
    arbitrate_stereo,
    constraint_features,
    embed_conformer,
    molecular_weight,
    symmetry_candidates,
    to_rdkit,
)
from ofoldx.data.molecular import ReferenceSource
from ofoldx.errors import InsufficientInformationError, ResourceLimitError, UnsupportedError
from ofoldx.provenance import Basis, Resolution, state

from .builders import edit_atom

# The parity verdicts on the subset, as the previous build decided them (shipped parquet, inverted_parity_atom_ids and
# refused_parity_atom_ids): C49's conformer disagrees with its label and its InChI, 8D8's InChI leaves C06 undefined.
INCOMPATIBLE, UNDECIDED = ("C49", "C1'"), ("8D8", "C06")
AGREEING = ("ALA", "ASN", "CYS", "HIS", "LYS", "MSE", "PRO", "SME", "NAG", "FRU", "DA", "A")


def row(component, name: str) -> int:
    return component.stereo_centre.tolist().index(component.atom_names.index(name))


class TestArbitrateStereo:
    def test_stale_label_stands_on_conformer(self, ccd, define):  # the InChI agrees with the conformer, not the label
        relabelled = edit_atom(ccd["ALA"], "pdbx_stereo_config", "CA", b"R")
        first = define(relabelled)
        result = arbitrate_stereo(first, relabelled.descriptors)
        assert (result.inverted, result.refused, result.contradicted) == ((), (), False)
        assert result.parity.tolist() == first.stereo_parity.tolist()
        assert result.basis.tolist() == [StereoBasis.CONFORMER]

    @pytest.mark.parametrize("described", [True, False])
    def test_insufficient_evidence_refused(self, archive, ccd, define, described):
        component_id, atom = UNDECIDED if described else INCOMPATIBLE
        descriptors = ccd[component_id].descriptors if described else ()
        result = arbitrate_stereo(archive(component_id), descriptors)
        assert (result.inverted, result.refused, result.contradicted) == ((), (atom,), False)
        component = define(ccd[component_id], stereo=result)
        r = row(component, atom)
        assert (component.stereo_parity[r], component.stereo_basis[r]) == (0, StereoBasis.REFUSED)

    def test_incompatible_conformer_contradicted(self, archive, ccd, define):
        component_id, atom = INCOMPATIBLE
        result = arbitrate_stereo(archive(component_id), ccd[component_id].descriptors)
        assert (result.inverted, result.refused, result.contradicted) == ((atom,), (), True)
        component = define(ccd[component_id], stereo=result)
        r = row(component, atom)
        assert component.stereo_basis[r] == StereoBasis.DESCRIPTOR_VERIFIED
        assert component.stereo_parity[r] == 3 - archive(component_id).stereo_parity[r]
        assert component.reference_source == ReferenceSource.CONTRADICTED

    @pytest.mark.parametrize("component_id", AGREEING)
    def test_agrees_with_rdkit(self, archive, ccd, component_id):
        r"""Locks the neighbour-slot convention (incl. the -1 slot of SME's sulfoxide S) against RDKit's 3-D
        perception: a disagreement would surface as ``derivation_differs``."""
        result = arbitrate_stereo(archive(component_id), ccd[component_id].descriptors)
        assert [atom for atom, reason in result.unverified if reason == "derivation_differs"] == []
        assert result.parity.tolist() == archive(component_id).stereo_parity.tolist()
        assert (result.inverted, result.refused, result.contradicted) == ((), (), False)


ALANINE = "N[C@@H](C)C(=O)O"


class TestToRdkit:
    @pytest.mark.parametrize("smiles", [ALANINE, "F/C=C/F", "Oc1ccccc1", "C[N+](C)(C)C"])
    def test_round_trip_smiles(self, ligand, smiles):
        from rdkit import Chem

        assert Chem.MolToSmiles(Chem.RemoveHs(to_rdkit(ligand(smiles)))) == Chem.CanonSmiles(smiles)

    def test_cut_selection_refused(self, ligand):
        with pytest.raises(UnsupportedError) as caught:
            to_rdkit(ligand("CC"), np.array([0]))
        assert caught.value.code == "rdkit_unsupported"

    def test_unknown_parity_unset(self, ligand, edit):
        from rdkit import Chem

        unknown = state(Resolution.UNKNOWN, Basis.DERIVED)
        system = edit(
            ligand(ALANINE),
            stereocentre_state=(0, unknown),
            stereocentre_parity=(0, 0),
            stereocentre_neighbors=(0, -1),
        )
        assert to_rdkit(system).GetAtomWithIdx(1).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    def test_incomplete_topology_refused(self, ligand, edit):
        system = edit(ligand("CC"), component_topology_complete=(0, False))
        with pytest.raises(InsufficientInformationError) as caught:
            to_rdkit(system)
        assert caught.value.code == "rdkit_incomplete"


class TestEmbedConformer:
    def test_coordinates_per_atom(self, ligand):
        system = ligand("OCC")
        conformer = embed_conformer(system, np.arange(len(system.atom_symbol)), seed=7)
        assert conformer.coordinates.shape == (9, 3)
        assert conformer.method == "etkdg_v3"

    def test_refused_projection_value(self, ligand, edit):
        system = edit(ligand("CC"), component_topology_complete=(0, False))
        assert embed_conformer(system, np.arange(8), seed=7) == EmbeddingFailed("projection_refused", 0)


class TestConstraintFeatures:
    def test_pairs_and_bonds(self, ligand):
        system = ligand("OCC")
        features = constraint_features(system, np.arange(9))
        assert len(features.pairs) == 36
        assert features.bond_pair.sum() == 8
        assert (features.lower <= features.upper).all()

    def test_chiral_from_parity(self, ligand):
        system = ligand(ALANINE)
        features = constraint_features(system, np.arange(len(system.atom_symbol)))
        assert len(features.chiral) == 4
        assert set(features.chiral[:, 3].tolist()) == {1}

    def test_trans_stereobond(self, ligand):
        system = ligand("F/C=C/F")
        features = constraint_features(system, np.arange(len(system.atom_symbol)))
        assert features.stereo_trans.tolist() == [True]

    def test_linear_triple(self, ligand):
        system = ligand("CC#N")
        features = constraint_features(system, np.arange(len(system.atom_symbol)))
        assert sorted(features.linear.reshape(-1).tolist()) == [0, 1, 2]

    def test_carbonyl_impropers(self, ligand):
        system = ligand("CC(=O)O")
        features = constraint_features(system, np.arange(len(system.atom_symbol)))
        assert features.planar_improper[:, 2].tolist() == [1, 1, 1]


class TestMolecularWeight:
    def test_hydrogens_counted(self, ligand):
        system = ligand("OCC")
        assert molecular_weight(system, np.arange(len(system.atom_symbol))) == pytest.approx(46.069, abs=1e-3)


class TestSymmetryCandidates:
    def candidates(self, system, resonance=False, pinned=None):
        pinned = np.zeros(len(system.atom_symbol), dtype=bool) if pinned is None else pinned
        return symmetry_candidates(system, np.array([0]), resonance=resonance, pinned=pinned)

    def test_benzene_twelve_automorphisms(self, ligand):
        candidates, _ = self.candidates(ligand("c1ccccc1"))
        assert len(candidates.kind) == 12
        assert set(candidates.kind.tolist()) == {SymmetryKind.AUTOMORPHISM}

    def test_pinned_atom_fixed(self, ligand):
        system = ligand("c1ccccc1")
        pinned = np.zeros(len(system.atom_symbol), dtype=bool)
        pinned[0] = True
        candidates, _ = self.candidates(system, pinned=pinned)
        assert len(candidates.kind) == 2
        assert set(candidates.permutation[:: len(system.atom_symbol)].tolist()) == {0}

    def test_carboxylate_resonance_only(self, ligand):
        system = ligand("CC(=O)[O-]")
        assert len(self.candidates(system)[0].kind) == 0
        candidates, _ = self.candidates(system, resonance=True)
        assert candidates.kind.tolist() == [SymmetryKind.AUTOMORPHISM, SymmetryKind.RESONANCE]

    def test_hydrogens_keep_their_isotope(self, ligand):
        assert len(self.candidates(ligand("[2H]C([2H])([2H])OC"))[0].kind) == 0

    def test_search_limit_raises(self, ligand):
        with pytest.raises(ResourceLimitError) as caught:
            self.candidates(ligand("CC(C)(C)C(C(C)(C)C)(C(C)(C)C)C(C)(C)C"))
        assert caught.value.code == "symmetry_search_limit"

    def test_incomplete_component_recorded(self, ligand, edit):
        system = edit(ligand("c1ccccc1"), component_topology_complete=(0, False))
        candidates, records = self.candidates(system)
        assert len(candidates.component) == 0
        assert records.counts() == {"symmetry_unknown_incomplete": 1}
