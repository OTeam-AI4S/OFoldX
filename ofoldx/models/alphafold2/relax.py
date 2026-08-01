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
Amber/OpenMM relaxation and AlphaFold 2 structural-violation evaluation.

The equations and default minimization parameters follow DeepMind AlphaFold
revision ``c77e5d2a8961d1a353632c462914ff0a32a950f6``. OpenMM and PDBFixer are
imported only when relaxation is requested; violation evaluation itself is an
ordinary PyTorch calculation over a canonical [`Structure`][].
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...data import residue_constants
from ...data.molecular import ELEMENTS, MolecularSystem
from ...data.structure import CoordinateEvidence, Structure, replace_coordinates
from ...data.tokenization import Encoding, token_classes
from ...errors import RequestRef, StructureRef, fail
from ...provenance import known
from .vocabulary import AMINO_ACIDS, RESIDUE_TYPE

_CA_C_N_COSINE = -0.4473
_CA_C_N_STDDEV = 0.0311
_C_N_CA_COSINE = -0.5203
_C_N_CA_STDDEV = 0.0353
_C_N_LENGTH = (1.329, 1.341)
_C_N_STDDEV = (0.014, 0.016)
_VAN_DER_WAALS_RADIUS = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}
_STEREO_CHEMICAL_PROPS = r"""
Bond			Residue		Mean		StdDev
CA-CB			ALA		1.520		0.021
N-CA			ALA		1.459		0.020
CA-C			ALA		1.525		0.026
C-O			ALA		1.229		0.019
CA-CB			ARG		1.535		0.022
CB-CG			ARG		1.521		0.027
CG-CD			ARG		1.515		0.025
CD-NE			ARG		1.460		0.017
NE-CZ			ARG		1.326		0.013
CZ-NH1			ARG		1.326		0.013
CZ-NH2			ARG		1.326		0.013
N-CA			ARG		1.459		0.020
CA-C			ARG		1.525		0.026
C-O			ARG		1.229		0.019
CA-CB			ASN		1.527		0.026
CB-CG			ASN		1.506		0.023
CG-OD1			ASN		1.235		0.022
CG-ND2			ASN		1.324		0.025
N-CA			ASN		1.459		0.020
CA-C			ASN		1.525		0.026
C-O			ASN		1.229		0.019
CA-CB			ASP		1.535		0.022
CB-CG			ASP		1.513		0.021
CG-OD1			ASP		1.249		0.023
CG-OD2			ASP		1.249		0.023
N-CA			ASP		1.459		0.020
CA-C			ASP		1.525		0.026
C-O			ASP		1.229		0.019
CA-CB			CYS		1.526		0.013
CB-SG			CYS		1.812		0.016
N-CA			CYS		1.459		0.020
CA-C			CYS		1.525		0.026
C-O			CYS		1.229		0.019
CA-CB			GLU		1.535		0.022
CB-CG			GLU		1.517		0.019
CG-CD			GLU		1.515		0.015
CD-OE1			GLU		1.252		0.011
CD-OE2			GLU		1.252		0.011
N-CA			GLU		1.459		0.020
CA-C			GLU		1.525		0.026
C-O			GLU		1.229		0.019
CA-CB			GLN		1.535		0.022
CB-CG			GLN		1.521		0.027
CG-CD			GLN		1.506		0.023
CD-OE1			GLN		1.235		0.022
CD-NE2			GLN		1.324		0.025
N-CA			GLN		1.459		0.020
CA-C			GLN		1.525		0.026
C-O			GLN		1.229		0.019
N-CA			GLY		1.456		0.015
CA-C			GLY		1.514		0.016
C-O			GLY		1.232		0.016
CA-CB			HIS		1.535		0.022
CB-CG			HIS		1.492		0.016
CG-ND1			HIS		1.369		0.015
CG-CD2			HIS		1.353		0.017
ND1-CE1			HIS		1.343		0.025
CD2-NE2			HIS		1.415		0.021
CE1-NE2			HIS		1.322		0.023
N-CA			HIS		1.459		0.020
CA-C			HIS		1.525		0.026
C-O			HIS		1.229		0.019
CA-CB			ILE		1.544		0.023
CB-CG1			ILE		1.536		0.028
CB-CG2			ILE		1.524		0.031
CG1-CD1			ILE		1.500		0.069
N-CA			ILE		1.459		0.020
CA-C			ILE		1.525		0.026
C-O			ILE		1.229		0.019
CA-CB			LEU		1.533		0.023
CB-CG			LEU		1.521		0.029
CG-CD1			LEU		1.514		0.037
CG-CD2			LEU		1.514		0.037
N-CA			LEU		1.459		0.020
CA-C			LEU		1.525		0.026
C-O			LEU		1.229		0.019
CA-CB			LYS		1.535		0.022
CB-CG			LYS		1.521		0.027
CG-CD			LYS		1.520		0.034
CD-CE			LYS		1.508		0.025
CE-NZ			LYS		1.486		0.025
N-CA			LYS		1.459		0.020
CA-C			LYS		1.525		0.026
C-O			LYS		1.229		0.019
CA-CB			MET		1.535		0.022
CB-CG			MET		1.509		0.032
CG-SD			MET		1.807		0.026
SD-CE			MET		1.774		0.056
N-CA			MET		1.459		0.020
CA-C			MET		1.525		0.026
C-O			MET		1.229		0.019
CA-CB			PHE		1.535		0.022
CB-CG			PHE		1.509		0.017
CG-CD1			PHE		1.383		0.015
CG-CD2			PHE		1.383		0.015
CD1-CE1			PHE		1.388		0.020
CD2-CE2			PHE		1.388		0.020
CE1-CZ			PHE		1.369		0.019
CE2-CZ			PHE		1.369		0.019
N-CA			PHE		1.459		0.020
CA-C			PHE		1.525		0.026
C-O			PHE		1.229		0.019
CA-CB			PRO		1.531		0.020
CB-CG			PRO		1.495		0.050
CG-CD			PRO		1.502		0.033
CD-N			PRO		1.474		0.014
N-CA			PRO		1.468		0.017
CA-C			PRO		1.524		0.020
C-O			PRO		1.228		0.020
CA-CB			SER		1.525		0.015
CB-OG			SER		1.418		0.013
N-CA			SER		1.459		0.020
CA-C			SER		1.525		0.026
C-O			SER		1.229		0.019
CA-CB			THR		1.529		0.026
CB-OG1			THR		1.428		0.020
CB-CG2			THR		1.519		0.033
N-CA			THR		1.459		0.020
CA-C			THR		1.525		0.026
C-O			THR		1.229		0.019
CA-CB			TRP		1.535		0.022
CB-CG			TRP		1.498		0.018
CG-CD1			TRP		1.363		0.014
CG-CD2			TRP		1.432		0.017
CD1-NE1			TRP		1.375		0.017
NE1-CE2			TRP		1.371		0.013
CD2-CE2			TRP		1.409		0.012
CD2-CE3			TRP		1.399		0.015
CE2-CZ2			TRP		1.393		0.017
CE3-CZ3			TRP		1.380		0.017
CZ2-CH2			TRP		1.369		0.019
CZ3-CH2			TRP		1.396		0.016
N-CA			TRP		1.459		0.020
CA-C			TRP		1.525		0.026
C-O			TRP		1.229		0.019
CA-CB			TYR		1.535		0.022
CB-CG			TYR		1.512		0.015
CG-CD1			TYR		1.387		0.013
CG-CD2			TYR		1.387		0.013
CD1-CE1			TYR		1.389		0.015
CD2-CE2			TYR		1.389		0.015
CE1-CZ			TYR		1.381		0.013
CE2-CZ			TYR		1.381		0.013
CZ-OH			TYR		1.374		0.017
N-CA			TYR		1.459		0.020
CA-C			TYR		1.525		0.026
C-O			TYR		1.229		0.019
CA-CB			VAL		1.543		0.021
CB-CG1			VAL		1.524		0.021
CB-CG2			VAL		1.524		0.021
N-CA			VAL		1.459		0.020
CA-C			VAL		1.525		0.026
C-O			VAL		1.229		0.019
-

Angle			Residue		Mean		StdDev
N-CA-CB			ALA		110.1		1.4
CB-CA-C			ALA		110.1		1.5
N-CA-C			ALA		111.0		2.7
CA-C-O			ALA		120.1		2.1
N-CA-CB			ARG		110.6		1.8
CB-CA-C			ARG		110.4		2.0
CA-CB-CG		ARG		113.4		2.2
CB-CG-CD		ARG		111.6		2.6
CG-CD-NE		ARG		111.8		2.1
CD-NE-CZ		ARG		123.6		1.4
NE-CZ-NH1		ARG		120.3		0.5
NE-CZ-NH2		ARG		120.3		0.5
NH1-CZ-NH2		ARG		119.4		1.1
N-CA-C			ARG		111.0		2.7
CA-C-O			ARG		120.1		2.1
N-CA-CB			ASN		110.6		1.8
CB-CA-C			ASN		110.4		2.0
CA-CB-CG		ASN		113.4		2.2
CB-CG-ND2		ASN		116.7		2.4
CB-CG-OD1		ASN		121.6		2.0
ND2-CG-OD1		ASN		121.9		2.3
N-CA-C			ASN		111.0		2.7
CA-C-O			ASN		120.1		2.1
N-CA-CB			ASP		110.6		1.8
CB-CA-C			ASP		110.4		2.0
CA-CB-CG		ASP		113.4		2.2
CB-CG-OD1		ASP		118.3		0.9
CB-CG-OD2		ASP		118.3		0.9
OD1-CG-OD2		ASP		123.3		1.9
N-CA-C			ASP		111.0		2.7
CA-C-O			ASP		120.1		2.1
N-CA-CB			CYS		110.8		1.5
CB-CA-C			CYS		111.5		1.2
CA-CB-SG		CYS		114.2		1.1
N-CA-C			CYS		111.0		2.7
CA-C-O			CYS		120.1		2.1
N-CA-CB			GLU		110.6		1.8
CB-CA-C			GLU		110.4		2.0
CA-CB-CG		GLU		113.4		2.2
CB-CG-CD		GLU		114.2		2.7
CG-CD-OE1		GLU		118.3		2.0
CG-CD-OE2		GLU		118.3		2.0
OE1-CD-OE2		GLU		123.3		1.2
N-CA-C			GLU		111.0		2.7
CA-C-O			GLU		120.1		2.1
N-CA-CB			GLN		110.6		1.8
CB-CA-C			GLN		110.4		2.0
CA-CB-CG		GLN		113.4		2.2
CB-CG-CD		GLN		111.6		2.6
CG-CD-OE1		GLN		121.6		2.0
CG-CD-NE2		GLN		116.7		2.4
OE1-CD-NE2		GLN		121.9		2.3
N-CA-C			GLN		111.0		2.7
CA-C-O			GLN		120.1		2.1
N-CA-C			GLY		113.1		2.5
CA-C-O			GLY		120.6		1.8
N-CA-CB			HIS		110.6		1.8
CB-CA-C			HIS		110.4		2.0
CA-CB-CG		HIS		113.6		1.7
CB-CG-ND1		HIS		123.2		2.5
CB-CG-CD2		HIS		130.8		3.1
CG-ND1-CE1		HIS		108.2		1.4
ND1-CE1-NE2		HIS		109.9		2.2
CE1-NE2-CD2		HIS		106.6		2.5
NE2-CD2-CG		HIS		109.2		1.9
CD2-CG-ND1		HIS		106.0		1.4
N-CA-C			HIS		111.0		2.7
CA-C-O			HIS		120.1		2.1
N-CA-CB			ILE		110.8		2.3
CB-CA-C			ILE		111.6		2.0
CA-CB-CG1		ILE		111.0		1.9
CB-CG1-CD1		ILE		113.9		2.8
CA-CB-CG2		ILE		110.9		2.0
CG1-CB-CG2		ILE		111.4		2.2
N-CA-C			ILE		111.0		2.7
CA-C-O			ILE		120.1		2.1
N-CA-CB			LEU		110.4		2.0
CB-CA-C			LEU		110.2		1.9
CA-CB-CG		LEU		115.3		2.3
CB-CG-CD1		LEU		111.0		1.7
CB-CG-CD2		LEU		111.0		1.7
CD1-CG-CD2		LEU		110.5		3.0
N-CA-C			LEU		111.0		2.7
CA-C-O			LEU		120.1		2.1
N-CA-CB			LYS		110.6		1.8
CB-CA-C			LYS		110.4		2.0
CA-CB-CG		LYS		113.4		2.2
CB-CG-CD		LYS		111.6		2.6
CG-CD-CE		LYS		111.9		3.0
CD-CE-NZ		LYS		111.7		2.3
N-CA-C			LYS		111.0		2.7
CA-C-O			LYS		120.1		2.1
N-CA-CB			MET		110.6		1.8
CB-CA-C			MET		110.4		2.0
CA-CB-CG		MET		113.3		1.7
CB-CG-SD		MET		112.4		3.0
CG-SD-CE		MET		100.2		1.6
N-CA-C			MET		111.0		2.7
CA-C-O			MET		120.1		2.1
N-CA-CB			PHE		110.6		1.8
CB-CA-C			PHE		110.4		2.0
CA-CB-CG		PHE		113.9		2.4
CB-CG-CD1		PHE		120.8		0.7
CB-CG-CD2		PHE		120.8		0.7
CD1-CG-CD2		PHE		118.3		1.3
CG-CD1-CE1		PHE		120.8		1.1
CG-CD2-CE2		PHE		120.8		1.1
CD1-CE1-CZ		PHE		120.1		1.2
CD2-CE2-CZ		PHE		120.1		1.2
CE1-CZ-CE2		PHE		120.0		1.8
N-CA-C			PHE		111.0		2.7
CA-C-O			PHE		120.1		2.1
N-CA-CB			PRO		103.3		1.2
CB-CA-C			PRO		111.7		2.1
CA-CB-CG		PRO		104.8		1.9
CB-CG-CD		PRO		106.5		3.9
CG-CD-N			PRO		103.2		1.5
CA-N-CD			PRO		111.7		1.4
N-CA-C			PRO		112.1		2.6
CA-C-O			PRO		120.2		2.4
N-CA-CB			SER		110.5		1.5
CB-CA-C			SER		110.1		1.9
CA-CB-OG		SER		111.2		2.7
N-CA-C			SER		111.0		2.7
CA-C-O			SER		120.1		2.1
N-CA-CB			THR		110.3		1.9
CB-CA-C			THR		111.6		2.7
CA-CB-OG1		THR		109.0		2.1
CA-CB-CG2		THR		112.4		1.4
OG1-CB-CG2		THR		110.0		2.3
N-CA-C			THR		111.0		2.7
CA-C-O			THR		120.1		2.1
N-CA-CB			TRP		110.6		1.8
CB-CA-C			TRP		110.4		2.0
CA-CB-CG		TRP		113.7		1.9
CB-CG-CD1		TRP		127.0		1.3
CB-CG-CD2		TRP		126.6		1.3
CD1-CG-CD2		TRP		106.3		0.8
CG-CD1-NE1		TRP		110.1		1.0
CD1-NE1-CE2		TRP		109.0		0.9
NE1-CE2-CD2		TRP		107.3		1.0
CE2-CD2-CG		TRP		107.3		0.8
CG-CD2-CE3		TRP		133.9		0.9
NE1-CE2-CZ2		TRP		130.4		1.1
CE3-CD2-CE2		TRP		118.7		1.2
CD2-CE2-CZ2		TRP		122.3		1.2
CE2-CZ2-CH2		TRP		117.4		1.0
CZ2-CH2-CZ3		TRP		121.6		1.2
CH2-CZ3-CE3		TRP		121.2		1.1
CZ3-CE3-CD2		TRP		118.8		1.3
N-CA-C			TRP		111.0		2.7
CA-C-O			TRP		120.1		2.1
N-CA-CB			TYR		110.6		1.8
CB-CA-C			TYR		110.4		2.0
CA-CB-CG		TYR		113.4		1.9
CB-CG-CD1		TYR		121.0		0.6
CB-CG-CD2		TYR		121.0		0.6
CD1-CG-CD2		TYR		117.9		1.1
CG-CD1-CE1		TYR		121.3		0.8
CG-CD2-CE2		TYR		121.3		0.8
CD1-CE1-CZ		TYR		119.8		0.9
CD2-CE2-CZ		TYR		119.8		0.9
CE1-CZ-CE2		TYR		119.8		1.6
CE1-CZ-OH		TYR		120.1		2.7
CE2-CZ-OH		TYR		120.1		2.7
N-CA-C			TYR		111.0		2.7
CA-C-O			TYR		120.1		2.1
N-CA-CB			VAL		111.5		2.2
CB-CA-C			VAL		111.4		1.9
CA-CB-CG1		VAL		110.9		1.5
CA-CB-CG2		VAL		110.9		1.5
CG1-CB-CG2		VAL		110.9		1.6
N-CA-C			VAL		111.0		2.7
CA-C-O			VAL		120.1		2.1
-

Non-bonded distance     Minimum Dist    Tolerance
C-C                     3.4             1.5
C-N                     3.25            1.5
C-S                     3.5             1.5
C-O                     3.22            1.5
N-N                     3.1             1.5
N-S                     3.35            1.5
N-O                     3.07            1.5
O-S                     3.32            1.5
O-O                     3.04            1.5
S-S                     2.03            1.0
-

"""


@dataclass(eq=False)
class AlphaFold2RelaxationResult:
    r"""
    One relaxed AlphaFold 2 structure and its minimization values.

    Attributes:
        structure: Relaxed canonical structure, including any canonical atoms
            materialized by PDBFixer.
        structural_violations: Per-residue post-relaxation violation mask.
        initial_energy: Initial Amber potential energy in kcal/mol.
        final_energy: Final Amber potential energy in kcal/mol.
        rmsd: All-atom displacement from the final minimization start in angstroms.
        num_iterations: Number of violation-informed minimization iterations.
    """

    structure: Structure
    structural_violations: NDArray[np.float32]
    initial_energy: float
    final_energy: float
    rmsd: float
    num_iterations: int


class AlphaFold2AmberRelaxation:
    r"""
    Restrained Amber minimization used by released AlphaFold 2 inference.

    Args:
        max_iterations: Maximum L-BFGS iterations per minimization. ``0`` leaves
            the iteration count unlimited.
        tolerance: OpenMM minimization force tolerance in kcal/(mol nm).
        stiffness: Heavy-atom restraint stiffness in kcal/(mol Å²).
        exclude_residues: Zero-based residues excluded from the initial restraints.
        max_outer_iterations: Maximum violation-informed minimization iterations.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 0,
        tolerance: float = 2.39,
        stiffness: float = 10.0,
        exclude_residues: Sequence[int] = (),
        max_outer_iterations: int = 3,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.stiffness = stiffness
        self.exclude_residues = tuple(exclude_residues)
        self.max_outer_iterations = max_outer_iterations

    def relax(
        self,
        structure: Structure,
        *,
        platform: str = "CPU",
        platform_properties: Mapping[str, str] | None = None,
    ) -> AlphaFold2RelaxationResult:
        r"""
        Relax one decoded monomer structure.

        Args:
            structure: Unrelaxed standard linear AlphaFold 2 monomer prediction.
                Actual canonical heavy connectivity must match the standard sequence;
                nicks, cycles and additional links are rejected. PDBFixer supplies
                missing coordinates. Canonical charges and realised hydrogens must
                be known and agree with the force-field topology; protonation changes
                are rejected.
            platform: Exact OpenMM platform name, such as ``"CPU"`` or ``"CUDA"``.
            platform_properties: Properties passed directly to the selected OpenMM
                platform.

        Returns:
            Relaxed structure, energies, displacement, and per-residue violations.
        """

        residue_types = _standard_linear_protein(structure.system)
        _known_protonation(structure.system)
        source_pdb, locations = _structure_pdb(structure)
        chemistry = _relaxation_chemistry(structure.system, locations)
        pdb_string = _clean_pdb(source_pdb)
        excluded = set(self.exclude_residues)
        final: _MinimizationResult | None = None
        violations = np.ones(len(residue_types), dtype=np.float32)
        num_iterations = 0
        while bool(violations.any()) and num_iterations < self.max_outer_iterations:
            final = _minimize(
                pdb_string,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                stiffness=self.stiffness,
                exclude_residues=excluded,
                platform=platform,
                platform_properties=platform_properties,
                chemistry=chemistry,
            )
            relaxed = _structure_from_openmm(
                structure,
                final.topology,
                final.positions,
                locations,
            )
            violations = alphafold2_structural_violations(relaxed)
            excluded.update(np.flatnonzero(violations).tolist())
            relaxed_pdb, _ = _structure_pdb(relaxed)
            pdb_string = _clean_pdb(relaxed_pdb)
            num_iterations += 1
        if final is None:
            fail("value_invalid", RequestRef("max_outer_iterations", str(self.max_outer_iterations)))
        relaxed = _structure_from_openmm(
            structure,
            final.topology,
            final.positions,
            locations,
        )
        displacement = final.initial_positions - final.positions
        rmsd = float(np.sqrt(np.square(displacement).sum() / len(displacement)))
        return AlphaFold2RelaxationResult(
            structure=relaxed,
            structural_violations=violations,
            initial_energy=final.initial_energy,
            final_energy=final.final_energy,
            rmsd=rmsd,
            num_iterations=num_iterations,
        )


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _standard_linear_protein(system: MolecularSystem) -> NDArray[np.int64]:
    r"""
    The residue type of each component, after proving the heavy-atom chemistry AF2's violation tables assume: one
    chain of standard amino acids whose topology, heavy atoms and connections are known, joined only by the peptide
    bond from the C of each component to the N of the next (no nick, cycle, crosslink or external connection).
    """

    s = system.arrays()
    count = len(s["component_symbol"])
    if len(s["instance_symbol"]) != 1:
        fail("model_capacity", RequestRef("instances", str(len(s["instance_symbol"]))))
    classes = token_classes(system)
    standard = np.isin(classes.token, AMINO_ACIDS) & (classes.encoding == Encoding.EXACT)
    if not standard.all():
        fail("model_capacity", StructureRef(instance=0, component=int(np.flatnonzero(~standard)[0])))
    complete = s["component_topology_complete"] & s["component_heavy_atoms_complete"]
    complete &= s["instance_connections_complete"][0]
    if not complete.all():
        fail("prepare_incomplete_topology", StructureRef(instance=0, component=int(np.flatnonzero(~complete)[0])))
    owner = _owner(s["component_atom_ptr"])
    names = system.text("atom", s["atom_symbol"])
    first, second = s["edge_index"].astype(np.int64)
    between = owner[first] != owner[second]
    first, second = first[between], second[between]
    earlier = np.where(owner[first] < owner[second], first, second)
    later = np.where(owner[first] < owner[second], second, first)
    peptide = (owner[later] == owner[earlier] + 1) & (names[earlier] == "C") & (names[later] == "N")
    chain = peptide.all() and np.array_equal(np.sort(owner[earlier]), np.arange(count - 1))
    if not chain or len(s["external_atom"]):
        fail("model_capacity", RequestRef("relaxation", "a protein graph other than one linear chain"))
    return RESIDUE_TYPE[classes.token.astype(np.int64)]


def _known_protonation(system: MolecularSystem) -> None:
    r"""Refuse a value whose hydrogens, formal charges or implicit hydrogens are not known: relaxation compares them
    with the force-field topology."""

    s = system.arrays()
    complete = system.component_hydrogens_complete
    if not complete.all():
        fail("prepare_incomplete_topology", StructureRef(instance=0, component=int(np.flatnonzero(~complete)[0])))
    for column in ("formal_charge", "implicit_hydrogens"):  # a canonical zero is never compared as a value
        unknown = ~known(s[f"{column}_state"])
        if unknown.any():
            fail("prepare_unknown_feature", StructureRef(atom=int(np.flatnonzero(unknown)[0])), column=column)


def alphafold2_structural_violations(structure: Structure) -> NDArray[np.float32]:
    r"""
    Return AlphaFold 2's hard structural-violation mask for one monomer.

    The mask combines peptide bond/angle violations, between-residue clashes, and within-residue bond, angle, and
    clash violations. Its row order follows the monomer's component order. A chemistry other than one standard linear
    protein chain is refused before the linear-neighbour terms are computed.
    """

    system = structure.system
    residue_types = _standard_linear_protein(system)
    ids = system.text("component", system.component_symbol).tolist()
    names = system.text("atom", system.atom_symbol).tolist()
    pointer = system.component_atom_ptr
    positions = torch.zeros(len(ids), 14, 3, dtype=torch.float32)
    exists = torch.zeros(len(ids), 14, dtype=torch.bool)
    for residue, component_id in enumerate(ids):
        slots = residue_constants.AMINO_ACID_ATOM14_NAMES[component_id]
        for atom in range(int(pointer[residue]), int(pointer[residue + 1])):
            if names[atom] in slots and structure.coordinate_mask[atom]:
                positions[residue, slots.index(names[atom])] = torch.from_numpy(structure.coordinates[atom])
                exists[residue, slots.index(names[atom])] = True
    residue_types = torch.from_numpy(residue_types)
    connection = _connection_violations(positions, exists, residue_types)
    between = _between_residue_clashes(positions, exists, residue_types)
    within = _within_residue_violations(positions, exists, residue_types)
    return torch.stack((connection, between, within)).any(dim=0).to(torch.float32).numpy()


@dataclass(frozen=True)
class _Bond:
    atom1: str
    atom2: str
    length: float
    stddev: float


@cache
def _atom14_bounds() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    lines = iter(_STEREO_CHEMICAL_PROPS.strip().splitlines())
    next(lines)
    bonds: dict[str, list[_Bond]] = {}
    for line in lines:
        if line.strip() == "-":
            break
        bond_atoms, residue, bond_length, bond_stddev = line.split()
        atom1, atom2 = bond_atoms.split("-")
        bonds.setdefault(residue, []).append(_Bond(atom1, atom2, float(bond_length), float(bond_stddev)))
    next(lines)
    next(lines)
    angles: dict[str, list[tuple[str, str, str, float, float]]] = {}
    for line in lines:
        if line.strip() == "-":
            break
        angle_atoms, residue, degrees, stddev_degrees = line.split()
        atom1, atom2, atom3 = angle_atoms.split("-")
        angles.setdefault(residue, []).append(
            (
                atom1,
                atom2,
                atom3,
                math.radians(float(degrees)),
                math.radians(float(stddev_degrees)),
            )
        )

    lower = np.zeros((21, 14, 14), dtype=np.float32)
    upper = np.zeros((21, 14, 14), dtype=np.float32)
    for residue_type, one_letter in enumerate(residue_constants.AMINO_ACID_ONE_LETTER_CODES):
        residue = residue_constants.AMINO_ACID_ONE_TO_THREE[one_letter]
        atom_names = residue_constants.AMINO_ACID_ATOM14_NAMES[residue]
        for atom1_index, atom1 in enumerate(atom_names):
            if not atom1:
                continue
            for atom2_index, atom2 in enumerate(atom_names):
                if not atom2 or atom1_index == atom2_index:
                    continue
                lower[residue_type, atom1_index, atom2_index] = (
                    _VAN_DER_WAALS_RADIUS[atom1[0]] + _VAN_DER_WAALS_RADIUS[atom2[0]] - 1.5
                )
                upper[residue_type, atom1_index, atom2_index] = 1e10

        bond_by_atoms = {frozenset((bond.atom1, bond.atom2)): bond for bond in bonds[residue]}
        bounded = list(bonds[residue])
        for atom1, atom2, atom3, angle, angle_stddev in angles[residue]:
            bond1 = bond_by_atoms[frozenset((atom1, atom2))]
            bond2 = bond_by_atoms[frozenset((atom2, atom3))]
            length = math.sqrt(bond1.length**2 + bond2.length**2 - 2.0 * bond1.length * bond2.length * math.cos(angle))
            scale = 0.5 / length
            derivative_angle = 2.0 * bond1.length * bond2.length * math.sin(angle) * scale
            derivative_bond1 = (2.0 * bond1.length - 2.0 * bond2.length * math.cos(angle)) * scale
            derivative_bond2 = (2.0 * bond2.length - 2.0 * bond1.length * math.cos(angle)) * scale
            stddev = math.sqrt(
                (derivative_angle * angle_stddev) ** 2
                + (derivative_bond1 * bond1.stddev) ** 2
                + (derivative_bond2 * bond2.stddev) ** 2
            )
            bounded.append(_Bond(atom1, atom3, length, stddev))
        for bond in bounded:
            atom1_index = atom_names.index(bond.atom1)
            atom2_index = atom_names.index(bond.atom2)
            lower_bound = bond.length - 12.0 * bond.stddev
            upper_bound = bond.length + 12.0 * bond.stddev
            lower[residue_type, atom1_index, atom2_index] = lower_bound
            lower[residue_type, atom2_index, atom1_index] = lower_bound
            upper[residue_type, atom1_index, atom2_index] = upper_bound
            upper[residue_type, atom2_index, atom1_index] = upper_bound
    return lower, upper


def _connection_violations(positions: Tensor, exists: Tensor, residue_types: Tensor) -> Tensor:
    num_residues = positions.shape[0]
    this_ca = positions[:-1, 1]
    this_c = positions[:-1, 2]
    next_n = positions[1:, 0]
    next_ca = positions[1:, 1]
    c_n_length = torch.sqrt(1e-6 + torch.square(this_c - next_n).sum(dim=-1))
    is_proline = residue_types[1:] == residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("P")
    expected_length = torch.where(is_proline, _C_N_LENGTH[1], _C_N_LENGTH[0])
    length_stddev = torch.where(is_proline, _C_N_STDDEV[1], _C_N_STDDEV[0])
    c_n = (
        exists[:-1, 2]
        & exists[1:, 0]
        & (torch.sqrt(1e-6 + torch.square(c_n_length - expected_length)) > 12.0 * length_stddev)
    )

    ca_c_length = torch.sqrt(1e-6 + torch.square(this_ca - this_c).sum(dim=-1))
    n_ca_length = torch.sqrt(1e-6 + torch.square(next_n - next_ca).sum(dim=-1))
    c_ca_unit = (this_ca - this_c) / ca_c_length[:, None]
    c_n_unit = (next_n - this_c) / c_n_length[:, None]
    n_ca_unit = (next_ca - next_n) / n_ca_length[:, None]
    ca_c_n = (
        exists[:-1, 1]
        & exists[:-1, 2]
        & exists[1:, 0]
        & (torch.sqrt(1e-6 + torch.square((c_ca_unit * c_n_unit).sum(dim=-1) - _CA_C_N_COSINE)) > 12.0 * _CA_C_N_STDDEV)
    )
    c_n_ca = (
        exists[:-1, 2]
        & exists[1:, 0]
        & exists[1:, 1]
        & (
            torch.sqrt(1e-6 + torch.square(((-c_n_unit) * n_ca_unit).sum(dim=-1) - _C_N_CA_COSINE))
            > 12.0 * _C_N_CA_STDDEV
        )
    )
    pair_violations = c_n | ca_c_n | c_n_ca
    per_residue = torch.zeros(num_residues, dtype=torch.bool)
    per_residue[:-1] |= pair_violations
    per_residue[1:] |= pair_violations
    return per_residue


@cache
def _atom14_radii() -> Tensor:
    radius_by_residue_type = torch.zeros(21, 14)
    for residue_type, one_letter in enumerate(residue_constants.AMINO_ACID_ONE_LETTER_CODES):
        component_id = residue_constants.AMINO_ACID_ONE_TO_THREE[one_letter]
        for atom_index, atom_name in enumerate(residue_constants.AMINO_ACID_ATOM14_NAMES[component_id]):
            if atom_name:
                radius_by_residue_type[residue_type, atom_index] = _VAN_DER_WAALS_RADIUS[atom_name[0]]
    return radius_by_residue_type


def _between_residue_clashes(positions: Tensor, exists: Tensor, residue_types: Tensor) -> Tensor:
    """Check heavy-atom overlaps after proving standard linear connectivity.

    The admitted graph has consecutive peptide C--N bonds and no disulfide
    crosslinks. Only those actual inter-residue bonds are excluded here.
    """
    num_residues = positions.shape[0]
    distances = torch.sqrt(
        1e-10 + torch.square(positions[:, None, :, None, :] - positions[None, :, None, :, :]).sum(dim=-1)
    )
    residue_index = torch.arange(num_residues)
    valid = (
        exists[:, None, :, None]
        & exists[None, :, None, :]
        & (residue_index[:, None, None, None] < residue_index[None, :, None, None])
    )
    radii = _atom14_radii()[residue_types] * exists
    neighbour = residue_index[:, None] + 1 == residue_index[None, :]
    c_n_bond = torch.zeros_like(valid)
    c_n_bond[:, :, 2, 0] = neighbour
    valid &= ~c_n_bond
    lower_bound = radii[:, None, :, None] + radii[None, :, None, :]
    clashes = valid & (distances < lower_bound - 1.5)
    per_atom = clashes.any(dim=(0, 2)) | clashes.any(dim=(1, 3))
    return per_atom.any(dim=-1)


def _within_residue_violations(positions: Tensor, exists: Tensor, residue_types: Tensor) -> Tensor:
    lower, upper = _atom14_bounds()
    lower_bound = torch.from_numpy(lower)[residue_types]
    upper_bound = torch.from_numpy(upper)[residue_types]
    distances = torch.sqrt(1e-10 + torch.square(positions[:, :, None, :] - positions[:, None, :, :]).sum(dim=-1))
    valid = ~torch.eye(14, dtype=torch.bool)[None] & exists[:, :, None] & exists[:, None, :]
    violations = valid & ((distances < lower_bound) | (distances > upper_bound))
    return (violations.any(dim=1) | violations.any(dim=2)).any(dim=-1)


def _structure_pdb(structure: Structure) -> tuple[str, dict[int, tuple[str, str, str]]]:
    r"""The placed heavy atoms as a PDB exchange file, and the (chain, number, insertion) address of each component."""
    system = structure.system
    s = system.arrays()
    count = len(s["component_symbol"])
    if count > 9999:
        fail("model_capacity", RequestRef("residues", str(count)), limit=9999)
    preferred = str(system.text("instance", s["instance_symbol"])[0])
    chain_id = preferred if len(preferred) == 1 and preferred.isalnum() else "A"
    ids = system.text("component", s["component_symbol"]).tolist()
    names = system.text("atom", s["atom_symbol"]).tolist()
    occupancy = np.where(known(structure.occupancy_state), structure.occupancy, 1.0)
    b_factor = np.where(known(structure.b_factor_state), structure.b_factor, 0.0)
    lines: list[str] = []
    locations: dict[int, tuple[str, str, str]] = {}
    serial = 0
    for component in range(count):
        # OpenMM addresses this temporary exchange file by chain/number/insertion. Use unique local addresses and keep
        # their component mapping; source author identifiers can repeat or be non-numeric and stay on the Structure.
        number = component + 1
        locations[component] = (chain_id, str(number), "")
        for atom in range(int(s["component_atom_ptr"][component]), int(s["component_atom_ptr"][component + 1])):
            atomic_number = int(s["atomic_number"][atom])
            if atomic_number == 1 or not structure.coordinate_mask[atom]:
                continue
            serial += 1
            element = ELEMENTS[atomic_number].upper()
            name = names[atom][:4]
            if len(name) < 4:
                name = f" {name:<3}" if len(element) == 1 and not name[0].isdigit() else f"{name:<4}"
            charge = int(s["formal_charge"][atom])
            charge_text = "  " if charge == 0 or abs(charge) > 9 else f"{abs(charge)}{'+' if charge > 0 else '-'}"
            x, y, z = structure.coordinates[atom].tolist()
            lines.append(
                f"ATOM  {serial:>5} {name} {ids[component][:3]:>3} {chain_id}{number:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy[atom]:>6.2f}{b_factor[atom]:>6.2f}          "
                f"{element:>2}{charge_text:>2}"
            )
    lines.append("END")
    return "\n".join(lines) + "\n", locations


def _clean_pdb(pdb_string: str) -> str:
    import pdbfixer
    from openmm import app

    fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_string))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()
    output = io.StringIO()
    app.PDBFile.writeFile(fixer.topology, fixer.positions, output, keepIds=True)
    return output.getvalue()


_ExchangeAtom = tuple[str, str, str, str]


@dataclass(frozen=True)
class _RelaxationChemistry:
    heavy_bonds: frozenset[frozenset[_ExchangeAtom]]
    hydrogens: dict[_ExchangeAtom, int]
    formal_charge: int


def _relaxation_chemistry(system: MolecularSystem, locations: dict[int, tuple[str, str, str]]) -> _RelaxationChemistry:
    r"""
    The canonical chemistry the force-field topology must reproduce: the bonds between heavy atoms, each heavy atom's
    realised hydrogens (its graph hydrogens and its KNOWN implicit hydrogens) and the total formal charge.
    """

    s = system.arrays()
    owner = _owner(s["component_atom_ptr"])
    names = system.text("atom", s["atom_symbol"]).tolist()
    heavy = s["atomic_number"] != 1
    addresses = {int(atom): (*locations[int(owner[atom])], names[atom]) for atom in np.flatnonzero(heavy)}
    hydrogens = s["implicit_hydrogens"].astype(np.int64)
    first, second = s["edge_index"].astype(np.int64)
    np.add.at(hydrogens, first[heavy[first] & ~heavy[second]], 1)
    np.add.at(hydrogens, second[heavy[second] & ~heavy[first]], 1)
    edges = frozenset(
        frozenset((addresses[a], addresses[b]))
        for a, b in zip(first.tolist(), second.tolist())
        if a in addresses and b in addresses
    )
    counts = {address: int(hydrogens[atom]) for atom, address in addresses.items()}
    return _RelaxationChemistry(edges, counts, int(s["formal_charge"].astype(np.int64).sum()))


def _check_openmm_chemistry(topology, chemistry: _RelaxationChemistry) -> None:
    def address(atom) -> _ExchangeAtom:
        return atom.residue.chain.id, atom.residue.id, atom.residue.insertionCode.strip(), atom.name

    heavy_bonds = set()
    hydrogens = {address(atom): 0 for atom in topology.atoms() if atom.element.atomic_number != 1}
    for first, second in topology.bonds():
        if first.element.atomic_number == 1:
            hydrogens[address(second)] += 1
        elif second.element.atomic_number == 1:
            hydrogens[address(first)] += 1
        else:
            heavy_bonds.add(frozenset((address(first), address(second))))
    if heavy_bonds != chemistry.heavy_bonds or hydrogens != chemistry.hydrogens:
        fail(
            "model_capacity", RequestRef("relaxation", "the force-field topology changes the connectivity or hydrogens")
        )


@dataclass(frozen=True)
class _MinimizationResult:
    topology: _OpenMMTopology
    positions: NDArray
    initial_positions: NDArray
    initial_energy: float
    final_energy: float


class _OpenMMChain(Protocol):
    id: str


class _OpenMMResidue(Protocol):
    chain: _OpenMMChain
    id: str
    insertionCode: str


class _OpenMMAtom(Protocol):
    residue: _OpenMMResidue
    name: str


class _OpenMMTopology(Protocol):
    def atoms(self) -> Iterator[_OpenMMAtom]: ...


def _minimize(
    pdb_string: str,
    *,
    max_iterations: int,
    tolerance: float,
    stiffness: float,
    exclude_residues: set[int],
    platform: str,
    platform_properties: Mapping[str, str] | None,
    chemistry: _RelaxationChemistry,
) -> _MinimizationResult:
    import openmm
    from openmm import app, unit

    pdb = app.PDBFile(io.StringIO(pdb_string))
    _check_openmm_chemistry(pdb.topology, chemistry)
    force_field = app.ForceField("amber99sb.xml")
    system = force_field.createSystem(pdb.topology, constraints=app.HBonds)
    nonbonded = next(force for force in system.getForces() if isinstance(force, openmm.NonbondedForce))
    net_charge = sum(
        nonbonded.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
        for index in range(nonbonded.getNumParticles())
    )
    if not np.isclose(net_charge, chemistry.formal_charge, rtol=0.0, atol=1e-4):
        fail("model_capacity", RequestRef("relaxation", "the force-field charge differs from the formal charge"))
    if stiffness > 0.0:
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness * unit.kilocalories_per_mole / unit.angstroms**2)
        for parameter in ("x0", "y0", "z0"):
            force.addPerParticleParameter(parameter)
        for atom_index, atom in enumerate(pdb.topology.atoms()):
            if atom.residue.index not in exclude_residues and atom.element.name != "hydrogen":
                force.addParticle(atom_index, pdb.positions[atom_index])
        system.addForce(force)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    selected_platform = openmm.Platform.getPlatformByName(platform)
    if platform_properties is None:
        simulation = app.Simulation(pdb.topology, system, integrator, selected_platform)
    else:
        simulation = app.Simulation(
            pdb.topology,
            system,
            integrator,
            selected_platform,
            dict(platform_properties),
        )
    simulation.context.setPositions(pdb.positions)
    initial = simulation.context.getState(getEnergy=True, getPositions=True)
    simulation.minimizeEnergy(
        maxIterations=max_iterations,
        tolerance=tolerance * unit.kilocalories_per_mole / unit.nanometer,
    )
    final = simulation.context.getState(getEnergy=True, getPositions=True)
    positions = final.getPositions(asNumpy=True)
    return _MinimizationResult(
        topology=simulation.topology,
        positions=np.asarray(positions.value_in_unit(unit.angstroms)),
        initial_positions=np.asarray(initial.getPositions(asNumpy=True).value_in_unit(unit.angstroms)),
        initial_energy=float(initial.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)),
        final_energy=float(final.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)),
    )


def _structure_from_openmm(
    source: Structure,
    topology: _OpenMMTopology,
    positions: NDArray,
    locations: dict[int, tuple[str, str, str]],
) -> Structure:
    r"""
    The relaxed coordinates written over every atom of the source (evidence PREDICTED: a relaxed prediction is still
    one); an atom OpenMM does not address has no coordinates afterwards.
    """

    system = source.system
    names = system.text("atom", system.atom_symbol).tolist()
    pointer = system.component_atom_ptr
    atom_by_address = {
        (*address, names[atom]): atom
        for component, address in locations.items()
        for atom in range(int(pointer[component]), int(pointer[component + 1]))
    }
    atoms, coordinates = [], []
    for openmm_atom, coordinate in zip(topology.atoms(), positions, strict=True):
        residue = openmm_atom.residue
        atom = atom_by_address.get((residue.chain.id, residue.id, residue.insertionCode.strip(), openmm_atom.name))
        if atom is not None:
            atoms.append(atom)
            coordinates.append(coordinate)
    every = np.ones(len(names), dtype=bool)
    relaxed, _ = replace_coordinates(
        source,
        np.array(atoms, dtype=np.int64),
        np.array(coordinates, dtype=np.float64).reshape(-1, 3),
        scope=every,
        evidence=CoordinateEvidence.PREDICTED,
    )
    return relaxed
