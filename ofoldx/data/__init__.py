r"""
Public molecular data types and codecs.
"""

from __future__ import annotations

from .codec import (
    decode_msa,
    decode_msa_collection,
    decode_structure,
    decode_system,
    decode_template,
    encode_msa,
    encode_msa_collection,
    encode_structure,
    encode_system,
    encode_template,
)
from .components import get_component
from .conditions import (
    AllowedResidueIdentities,
    AtomEndpoint,
    Endpoint,
    Hotspot,
    ResidueEndpoint,
    TiedResidueIdentities,
)
from .elements import ATOMIC_NUMBER_BY_ELEMENT
from .evidence import MSA, MSACollection, Template, TemplateAlignment
from .molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_KIND_DATIVE,
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    BOND_ROLE_POLYMER,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MOLECULE_BRANCHED,
    MOLECULE_NON_POLYMER,
    MOLECULE_WATER,
    STANDARD_AMINO_ACID_IDS,
    AtomSite,
    BondDirection,
    BondKind,
    BondRole,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    MoleculeType,
    PolymerAttachment,
    PolymerSequence,
    PolymerType,
    SystemBond,
    copy_components,
    disjoint_union,
    form_bond,
    instance_subgraph,
    is_water_component,
    replace_components,
    water_component_mask,
)
from .placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
    SymbolicAtomSite,
    SymbolicResidueSite,
    validate_placement,
)
from .structure import Structure
