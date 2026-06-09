r"""
Hugging Face Hub integration helpers.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from os import PathLike
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterator, cast

from chanfig import Config

from .artifacts import artifact_kind, artifact_task, entrypoint, read_artifact_manifest

MODEL_CARD_NAME = "README.md"
_PROJECT_REPOSITORY = "https://github.com/OTeam-AI4S/OFoldX"
_PROJECT_ISSUES = "https://github.com/OTeam-AI4S/OFoldX/issues"

_DEFAULT_TAGS = ("biology", "biomolecular-design", "protein", "rna", "dna")
_TASK_TAGS = {
    "binding_affinity_prediction": ("binding-affinity-prediction", "affinity-prediction", "protein-ligand"),
    "design_generation": ("design-generation", "protein-design"),
    "side_chain_packing": ("side-chain-packing", "protein-design"),
    "structure_prediction": ("structure-prediction", "protein-structure"),
}
_PUBLIC_ARCHITECTURE_NAMES = {
    "boltz2_affinity": "boltz2-affinity",
    "boltzgen_ifold": "boltzgen-inversefolding",
    "boltzgen_structuretrained_small": "boltzgen-structure-small",
    "ligandmpnn_packer": "ligandmpnn-packer",
}
_DEFAULT_WIDGET_TEXT = {
    "binding_affinity_prediction": (
        {
            "example_title": "Protein-ligand affinity scoring",
            "text": "complex_structure: complex.cif\nligand_chain: L",
            "input_format": "structure_path",
        },
    ),
    "design_generation": (
        {
            "example_title": "Backbone sequence design",
            "text": "input_structure: backbone.cif\ndesign_chains: A",
            "input_format": "structure_path",
        },
        {
            "example_title": "Binder design",
            "text": "target_structure: target.cif\ntarget_chains: A\ndesign_chains: B",
            "input_format": "structure_path",
        },
    ),
    "structure_prediction": (
        {
            "example_title": "Single-chain protein",
            "text": ">A\nMKTAYIAKQRQISFVKSHFSRQDILD",
            "input_format": "fasta",
        },
        {
            "example_title": "Protein complex",
            "text": ">A\nMKTAYIAKQRQISFVKSHFSRQDILD\n>B\nGSHMRYFVTAVSRPGRGEPRFI",
            "input_format": "fasta",
        },
    ),
    "side_chain_packing": (
        {
            "example_title": "Pack side chains",
            "text": "input_structure: backbone.cif\nsequence: MKTAYIAKQRQISFVKSHFSRQDILD",
            "input_format": "structure_path",
        },
    ),
}
_ARCHITECTURE_WIDGET_TEXT = {
    "boltzgen_ifold": (
        {
            "example_title": "Inverse folding",
            "text": "input_structure: backbone.cif\ndesign_chains: A",
            "input_format": "structure_path",
        },
    ),
    "boltzgen_structuretrained_small": (
        {
            "example_title": "Structure-conditioned generation",
            "text": "input_structure: seed_structure.cif\nnum_designs: 8",
            "input_format": "structure_path",
        },
    ),
    "rfdiffusion": (
        {
            "example_title": "Backbone generation",
            "text": "contig: 100-150\nhotspot_residues: []",
            "input_format": "structure_path",
        },
        {
            "example_title": "Motif scaffolding",
            "text": "input_structure: motif.cif\ncontig: A10-25/80-120",
            "input_format": "structure_path",
        },
    ),
}
_TASK_SUMMARIES = {
    "binding_affinity_prediction": "protein-ligand binding-affinity prediction",
    "design_generation": "biomolecular design generation",
    "side_chain_packing": "protein side-chain packing",
    "structure_prediction": "biomolecular structure prediction",
}
_ARCHITECTURE_SUMMARIES = {
    "alphafold2": (
        "AlphaFold 2 structure-prediction entry point for the OFoldX runtime. Strict checkpoint loading is a "
        "conversion gate; same-input/same-output numerical parity is tracked separately."
    ),
    "alphafold3": "Official AlphaFold 3 structure-prediction architecture entry point for the OFoldX runtime.",
    "boltz2": "Boltz-2-shaped structure-prediction model converted for the OFoldX runtime.",
    "boltz2_affinity": "Boltz-2 affinity model with a pair-only head for protein-ligand affinity scoring.",
    "boltzgen_adherence": "BoltzGen design generator variant optimized for adherence to conditioning signals.",
    "boltzgen_diverse": "BoltzGen design generator variant optimized for diverse structure-conditioned generation.",
    "boltzgen_ifold": "BoltzGen inverse-folding generator for sequence design from backbone or structure inputs.",
    "boltzgen_structuretrained_small": "Small BoltzGen structure-conditioned generation checkpoint.",
    "esmfold2": "Biohub ESMFold2 structure-prediction architecture converted for the OFoldX runtime.",
    "ligandmpnn": "LigandMPNN sequence-design model with ligand/atom context features.",
    "ligandmpnn_packer": "LigandMPNN side-chain-packing model for packing rotamers around a fixed sequence.",
    "protenix": "Protenix v0.5 AlphaFold 3-shaped structure-prediction model converted for the OFoldX runtime.",
    "proteinmpnn": "ProteinMPNN sequence-design model for protein backbones, including soluble and membrane variants.",
    "rfdiffusion": "RFdiffusion backbone-generation model for protein design and motif scaffolding.",
}
_BOLTZ2_CITATION = """\
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}"""
_BOLTZGEN_CITATION = """\
@article{stark2025boltzgen,
  author = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and others},
  title = {BoltzGen: Toward Universal Binder Design},
  year = {2025},
  doi = {10.1101/2025.11.20.689494},
  journal = {bioRxiv}
}"""
_LIGANDMPNN_CITATION = """\
@article{dauparas2025atomic,
  author = {Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and others},
  title = {Atomic context-conditioned protein sequence design using LigandMPNN},
  journal = {Nature Methods},
  year = {2025},
  doi = {10.1038/s41592-025-02626-1}
}"""
_PROTEINMPNN_CITATION = """\
@article{dauparas2022robust,
  author = {Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and others},
  title = {Robust deep learning-based protein sequence design using ProteinMPNN},
  journal = {Science},
  volume = {378},
  number = {6615},
  pages = {49--56},
  year = {2022},
  doi = {10.1126/science.add2187}
}"""
_SOLUBLEMPNN_CITATION = """\
@article{solublempnn2024membrane,
  title = {Computational design of soluble and functional membrane protein analogues},
  journal = {Nature},
  year = {2024},
  doi = {10.1038/s41586-024-07601-y}
}"""
_ALPHAFOLD2_CITATION = """\
@article{jumper2021highly,
  author = {Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  title = {Highly accurate protein structure prediction with AlphaFold},
  journal = {Nature},
  volume = {596},
  pages = {583--589},
  year = {2021},
  doi = {10.1038/s41586-021-03819-2}
}"""
_ALPHAFOLD3_CITATION = """\
@article{abramson2024accurate,
  author = {Abramson, Josh and Adler, Jonas and Dunger, Jack and others},
  title = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  journal = {Nature},
  volume = {630},
  pages = {493--500},
  year = {2024},
  doi = {10.1038/s41586-024-07487-w}
}"""
_PROTENIX_CITATION = """\
@article{protenix2025advancing,
  author = {ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and others},
  title = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  year = {2025},
  doi = {10.1101/2025.01.08.631967},
  journal = {bioRxiv}
}"""
_ESMFOLD2_CITATION = """\
@misc{candido2026language,
  title = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and others},
  year = {2026},
  url = {https://biohub.ai/papers/esm_protein.pdf},
  note = {Preprint}
}"""
_RFDIFFUSION_CITATION = """\
@article{watson2023denovo,
  author = {Watson, Joseph L. and Juergens, David and Bennett, Nathaniel R. and others},
  title = {De novo design of protein structure and function with RFdiffusion},
  journal = {Nature},
  volume = {620},
  number = {7976},
  pages = {1089--1100},
  year = {2023},
  doi = {10.1038/s41586-023-06415-8}
}"""
_UPSTREAM_METADATA = {
    "alphafold2": {
        "project": "AlphaFold 2",
        "summary": (
            "AlphaFold 2 structure-prediction architecture entry point for OFoldX. Strict checkpoint loading is "
            "a conversion gate, not final public-output parity."
        ),
        "repository": "https://github.com/google-deepmind/alphafold",
        "paper": "Highly accurate protein structure prediction with AlphaFold",
        "paper_url": "https://doi.org/10.1038/s41586-021-03819-2",
        "license": "Apache-2.0 for upstream AlphaFold 2 code; CC BY 4.0 for official model parameters",
        "training_data": (
            "The upstream AlphaFold 2 release describes training on PDB structures and sequence databases. "
            "OFoldX does not redistribute the training corpus."
        ),
        "training_procedure": (
            "OFoldX records AlphaFold 2 as a distinct architecture and converts official monomer/pTM parameters "
            "into a strict-loadable PyTorch parameter tree. Strict loading is a conversion gate, not final "
            "public-output parity."
        ),
        "citation": _ALPHAFOLD2_CITATION,
    },
    "alphafold3": {
        "project": "AlphaFold 3",
        "summary": "Official AlphaFold 3 structure-prediction architecture entry point for OFoldX.",
        "repository": "https://github.com/google-deepmind/alphafold3",
        "paper": "Accurate structure prediction of biomolecular interactions with AlphaFold 3",
        "paper_url": "https://doi.org/10.1038/s41586-024-07487-w",
        "license": "Apache-2.0 for upstream AlphaFold 3 code; official parameter terms apply to model weights",
        "training_data": (
            "The upstream AlphaFold 3 publication describes training on biomolecular structure and sequence "
            "resources. OFoldX does not redistribute the training corpus."
        ),
        "training_procedure": (
            "OFoldX records AlphaFold 3 as a distinct architecture. Official checkpoint conversion requires "
            "the local DeepMind parameter release and must not upload official parameters."
        ),
        "citation": _ALPHAFOLD3_CITATION,
    },
    "esmfold2": {
        "project": "ESMFold2",
        "summary": "Biohub ESMFold2 all-atom structure-prediction architecture converted for OFoldX.",
        "repository": "https://github.com/Biohub/esm",
        "paper": "Language Modeling Materializes a World Model of Protein Biology",
        "paper_url": "https://biohub.ai/papers/esm_protein.pdf",
        "license": "MIT for the Biohub ESM package; Apache-2.0 for the Biohub Transformers model code",
        "training_data": (
            "The upstream ESMFold2 model card describes training on Protein Data Bank and AlphaFold DB "
            "structures. OFoldX does not redistribute the training corpus."
        ),
        "training_procedure": (
            "OFoldX converts the released Biohub ESMFold2 checkpoint into `model.safetensors` plus an "
            "OFoldX manifest; it does not run ESMFold2 training."
        ),
        "citation": _ESMFOLD2_CITATION,
    },
    "protenix": {
        "project": "Protenix",
        "summary": (
            "Converted Protenix v0.5.0 structure-prediction weights in an AlphaFold 3-shaped OFoldX " "implementation."
        ),
        "repository": "https://github.com/bytedance/Protenix",
        "paper": "Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction",
        "paper_url": "https://doi.org/10.1101/2025.01.08.631967",
        "license": "Apache-2.0 for upstream Protenix code and model parameters",
        "training_data": (
            "The upstream Protenix v0.5.0 release reports training with a PDB cutoff date of 2021-09-30. "
            "OFoldX does not include the Protenix training corpus."
        ),
        "training_procedure": (
            "OFoldX converts the released `protenix_base_default_v0.5.0.pt` checkpoint to the OFoldX "
            "artifact format; it does not retrain the model."
        ),
        "citation": _PROTENIX_CITATION,
    },
    "boltz2": {
        "project": "Boltz-2",
        "summary": "Converted Boltz-2 structure-prediction checkpoint for biomolecular interaction modeling.",
        "repository": "https://github.com/jwohlwend/boltz",
        "paper": "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction",
        "paper_url": "https://doi.org/10.1101/2025.06.14.659707",
        "license": "MIT for upstream Boltz code and weights",
        "training_data": (
            "The upstream Boltz-2 report describes a biomolecular interaction training set with expanded "
            "structure data, molecular-dynamics ensembles, self-distilled predictions, and affinity supervision. "
            "OFoldX does not redistribute the training set."
        ),
        "training_procedure": (
            "OFoldX converts the upstream Boltz-2 checkpoint into `model.safetensors` plus an OFoldX manifest; "
            "it does not run Boltz-2 training."
        ),
        "citation": _BOLTZ2_CITATION,
    },
    "boltz2_affinity": {
        "project": "Boltz-2",
        "summary": "Converted Boltz-2 affinity checkpoint for protein-ligand binding-affinity prediction.",
        "repository": "https://github.com/jwohlwend/boltz",
        "paper": "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction",
        "paper_url": "https://doi.org/10.1101/2025.06.14.659707",
        "license": "MIT for upstream Boltz code and weights",
        "training_data": (
            "The upstream Boltz-2 affinity heads use binding-affinity supervision described in the Boltz-2 "
            "technical report. OFoldX does not redistribute the training set."
        ),
        "training_procedure": (
            "OFoldX converts the upstream Boltz-2 affinity checkpoint into `model.safetensors` plus an OFoldX "
            "manifest; it does not run Boltz-2 training."
        ),
        "citation": _BOLTZ2_CITATION,
    },
    "boltzgen_adherence": {
        "project": "BoltzGen",
        "summary": "Converted BoltzGen adherence generator checkpoint for controllable binder and protein design.",
        "repository": "https://github.com/HannesStark/boltzgen",
        "paper": "BoltzGen: Toward Universal Binder Design",
        "paper_url": "https://doi.org/10.1101/2025.11.20.689494",
        "license": "MIT for upstream BoltzGen code",
        "training_data": (
            "The upstream BoltzGen training release includes target structures, MSAs, and molecule dictionaries "
            "distributed through the `boltzgen/boltzgen1_train` and `boltzgen/inference-data` Hub datasets."
        ),
        "training_procedure": (
            "OFoldX converts the released BoltzGen adherence checkpoint; it does not run BoltzGen training."
        ),
        "citation": _BOLTZGEN_CITATION,
    },
    "boltzgen_diverse": {
        "project": "BoltzGen",
        "summary": "Converted BoltzGen diverse generator checkpoint for diverse structure-conditioned design.",
        "repository": "https://github.com/HannesStark/boltzgen",
        "paper": "BoltzGen: Toward Universal Binder Design",
        "paper_url": "https://doi.org/10.1101/2025.11.20.689494",
        "license": "MIT for upstream BoltzGen code",
        "training_data": (
            "The upstream BoltzGen training release includes target structures, MSAs, and molecule dictionaries "
            "distributed through the `boltzgen/boltzgen1_train` and `boltzgen/inference-data` Hub datasets."
        ),
        "training_procedure": (
            "OFoldX converts the released BoltzGen diverse checkpoint; it does not run BoltzGen training."
        ),
        "citation": _BOLTZGEN_CITATION,
    },
    "boltzgen_ifold": {
        "project": "BoltzGen",
        "summary": "Converted BoltzGen inverse-folding checkpoint for designing sequences from structure inputs.",
        "repository": "https://github.com/HannesStark/boltzgen",
        "paper": "BoltzGen: Toward Universal Binder Design",
        "paper_url": "https://doi.org/10.1101/2025.11.20.689494",
        "license": "MIT for upstream BoltzGen code",
        "training_data": (
            "The upstream BoltzGen training release includes target structures, MSAs, and molecule dictionaries "
            "distributed through the `boltzgen/boltzgen1_train` and `boltzgen/inference-data` Hub datasets."
        ),
        "training_procedure": (
            "OFoldX converts the released BoltzGen inverse-folding checkpoint; it does not run BoltzGen training."
        ),
        "citation": _BOLTZGEN_CITATION,
    },
    "boltzgen_structuretrained_small": {
        "project": "BoltzGen",
        "summary": "Converted small BoltzGen structure-trained checkpoint for structure-conditioned generation.",
        "repository": "https://github.com/HannesStark/boltzgen",
        "paper": "BoltzGen: Toward Universal Binder Design",
        "paper_url": "https://doi.org/10.1101/2025.11.20.689494",
        "license": "MIT for upstream BoltzGen code",
        "training_data": (
            "The upstream BoltzGen training release includes target structures, MSAs, and molecule dictionaries "
            "distributed through the `boltzgen/boltzgen1_train` and `boltzgen/inference-data` Hub datasets."
        ),
        "training_procedure": (
            "OFoldX converts the released BoltzGen structure-trained checkpoint; it does not run BoltzGen training."
        ),
        "citation": _BOLTZGEN_CITATION,
    },
    "ligandmpnn": {
        "project": "LigandMPNN",
        "summary": "Converted LigandMPNN sequence-design checkpoint with ligand and non-protein atom context.",
        "repository": "https://github.com/dauparas/LigandMPNN",
        "paper": "Atomic context-conditioned protein sequence design using LigandMPNN",
        "paper_url": "https://doi.org/10.1038/s41592-025-02626-1",
        "license": "MIT for upstream LigandMPNN code and model parameters",
        "training_data": (
            "The LigandMPNN paper describes training on PDB assemblies, including non-protein atomic context "
            "such as ligands, nucleic acids, and metals. OFoldX does not redistribute the training set."
        ),
        "training_procedure": (
            "OFoldX converts released LigandMPNN checkpoints into `model.safetensors`; it does not run "
            "LigandMPNN training."
        ),
        "citation": _LIGANDMPNN_CITATION,
    },
    "ligandmpnn_packer": {
        "project": "LigandMPNN",
        "summary": "Converted LigandMPNN side-chain packing checkpoint.",
        "repository": "https://github.com/dauparas/LigandMPNN",
        "paper": "Atomic context-conditioned protein sequence design using LigandMPNN",
        "paper_url": "https://doi.org/10.1038/s41592-025-02626-1",
        "license": "MIT for upstream LigandMPNN code and model parameters",
        "training_data": (
            "The side-chain packing checkpoint is distributed with the LigandMPNN model parameters and shares "
            "the upstream LigandMPNN release provenance."
        ),
        "training_procedure": (
            "OFoldX converts the released LigandMPNN side-chain packing checkpoint; it does not run training."
        ),
        "citation": _LIGANDMPNN_CITATION,
    },
    "proteinmpnn": {
        "project": "ProteinMPNN / LigandMPNN release",
        "summary": "Converted ProteinMPNN-family sequence-design checkpoint for fixed-backbone inverse folding.",
        "repository": "https://github.com/dauparas/LigandMPNN",
        "paper": "Robust deep learning-based protein sequence design using ProteinMPNN",
        "paper_url": "https://doi.org/10.1126/science.add2187",
        "license": "MIT for upstream ProteinMPNN and LigandMPNN code/model parameters",
        "training_data": (
            "ProteinMPNN was trained for fixed-backbone protein sequence design. Soluble and membrane variants "
            "come from the upstream LigandMPNN parameter release; OFoldX does not redistribute the training set."
        ),
        "training_procedure": (
            "OFoldX converts released ProteinMPNN-family checkpoints into `model.safetensors`; it does not run "
            "ProteinMPNN training."
        ),
        "citation": _PROTEINMPNN_CITATION,
    },
    "rfdiffusion": {
        "project": "RFdiffusion",
        "summary": "Converted RFdiffusion checkpoint for backbone generation, motif scaffolding, and binder design.",
        "repository": "https://github.com/RosettaCommons/RFdiffusion",
        "paper": "De novo design of protein structure and function with RFdiffusion",
        "paper_url": "https://doi.org/10.1038/s41586-023-06415-8",
        "license": "BSD for upstream RFdiffusion code and referenced model weights",
        "training_data": (
            "RFdiffusion training and use are described in the upstream Nature paper and repository. OFoldX "
            "does not redistribute the training set."
        ),
        "training_procedure": (
            "OFoldX converts released RFdiffusion checkpoints into `model.safetensors`; it does not run "
            "RFdiffusion training."
        ),
        "citation": _RFDIFFUSION_CITATION,
    },
}
_SOURCE_CHECKPOINTS_BY_REPO = {
    "alphafold2": "alphafold_params_2022-12-06.tar",
    "boltz2": "boltz2_conf_final.ckpt",
    "boltz2-affinity": "boltz2_aff.ckpt",
    "boltzgen-adherence": "boltzgen1_adherence.ckpt",
    "boltzgen-diverse": "boltzgen1_diverse.ckpt",
    "boltzgen-inversefolding": "boltzgen1_ifold.ckpt",
    "boltzgen-structure-small": "boltzgen1_structuretrained_small.ckpt",
    "esmfold2": "biohub/ESMFold2 model.safetensors",
    "protenix-base": "protenix_base_default_v0.5.0.pt",
}
_UPSTREAM_REFINEMENTS = {
    "alphafold2": {
        "hub_license": "cc-by-4.0",
        "source_release": "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
        "training_procedure": (
            "The upstream AlphaFold 2 release provides official model parameters under CC BY 4.0. OFoldX "
            "converts official monomer and pTM `params_model_*.npz` checkpoints into `model.safetensors` "
            "with a strict-loadable AF2 PyTorch parameter tree. Strict loading is a conversion gate, not final "
            "public-output parity."
        ),
    },
    "alphafold3": {
        "source_release": "https://github.com/google-deepmind/alphafold3",
        "training_procedure": (
            "OFoldX can inspect local official AlphaFold 3 parameter archives for user-side conversion work. "
            "Official DeepMind AlphaFold 3 parameters are not uploaded by the OFoldX converter."
        ),
    },
    "esmfold2": {
        "base_model": "biohub/ESMFold2",
        "hub_license": "mit",
        "source_release": "https://huggingface.co/biohub/ESMFold2",
    },
    "protenix": {
        "hub_license": "apache-2.0",
        "source_release": "https://github.com/bytedance/Protenix/blob/main/docs/supported_models.md",
        "training_data": (
            "The upstream Protenix v0.5.0 release reports a 2021-09-30 training-data cutoff. The Protenix "
            "training-data pipeline covers released wwPDB/mmCIF bioassemblies, MSAs, CCD cache data, and "
            "RCSB sequence clusters. OFoldX does not redistribute the training corpus."
        ),
        "training_procedure": (
            "The upstream `protenix_base_default_v0.5.0` model is an AF3-shaped all-atom structure-prediction "
            "checkpoint. OFoldX converts `protenix_base_default_v0.5.0.pt` to the OFoldX artifact format; it "
            "does not retrain the model."
        ),
    },
    "boltz2": {
        "base_model": "boltz-community/boltz-2",
        "hub_license": "mit",
        "source_release": "https://huggingface.co/boltzgen/boltzgen-1",
        "training_data": (
            "The Boltz-2 report describes structure data from PDB structures before 2023-06-01, experimental "
            "NMR collections, MD trajectory datasets, AF2 monomer distillation, Boltz-1 complex distillation, "
            "and affinity data from PubChem, ChEMBL, BindingDB, and related filtered assay sources. OFoldX does "
            "not redistribute the training set."
        ),
        "training_procedure": (
            "Upstream Boltz-2 training is staged across structure, confidence, and affinity objectives. This "
            "OFoldX artifact converts a released checkpoint into `model.safetensors` plus an OFoldX manifest; "
            "it does not run Boltz-2 training."
        ),
    },
    "boltz2_affinity": {
        "base_model": "boltz-community/boltz-2",
        "hub_license": "mit",
        "source_release": "https://huggingface.co/boltzgen/boltzgen-1",
        "training_data": (
            "The Boltz-2 affinity heads use filtered assay data from sources such as PubChem, ChEMBL, and "
            "BindingDB, with decoy generation and structural-confidence filters described by the upstream report. "
            "OFoldX does not redistribute the training set."
        ),
        "training_procedure": (
            "Upstream affinity training detaches the trunk and optimizes regression and binary binder/decoy "
            "objectives. OFoldX converts the released affinity checkpoint into `model.safetensors` plus an "
            "OFoldX manifest; it does not run Boltz-2 training."
        ),
    },
    "boltzgen_adherence": {
        "base_model": "boltz-community/boltz-2",
        "datasets": ("boltzgen/boltzgen1_train", "boltzgen/inference-data"),
        "hub_license": "mit",
        "source_release": "https://huggingface.co/boltzgen/boltzgen-1",
        "training_data": (
            "BoltzGen builds on the Boltz-2 data pipeline. The public training release includes `targets.zip` "
            "and `msa.zip` in the `boltzgen/boltzgen1_train` dataset and molecule dictionaries in "
            "`boltzgen/inference-data`; the full large-model recipe may also use additional distillation data."
        ),
        "training_procedure": (
            "Upstream BoltzGen trains a diffusion objective over randomly cropped biomolecular structures with "
            "randomly selected design and conditioning regions. OFoldX converts the released adherence checkpoint; "
            "it does not run BoltzGen training."
        ),
    },
    "boltzgen_diverse": {
        "base_model": "boltz-community/boltz-2",
        "datasets": ("boltzgen/boltzgen1_train", "boltzgen/inference-data"),
        "hub_license": "mit",
        "source_release": "https://huggingface.co/boltzgen/boltzgen-1",
        "training_data": (
            "BoltzGen builds on the Boltz-2 data pipeline. The public training release includes `targets.zip` "
            "and `msa.zip` in the `boltzgen/boltzgen1_train` dataset and molecule dictionaries in "
            "`boltzgen/inference-data`; the full large-model recipe may also use additional distillation data."
        ),
        "training_procedure": (
            "Upstream BoltzGen trains a diffusion objective over randomly cropped biomolecular structures with "
            "randomly selected design and conditioning regions. OFoldX converts the released diverse checkpoint; "
            "it does not run BoltzGen training."
        ),
    },
    "boltzgen_ifold": {
        "base_model": "boltz-community/boltz-2",
        "datasets": ("boltzgen/boltzgen1_train", "boltzgen/inference-data"),
        "hub_license": "mit",
        "source_release": "https://huggingface.co/boltzgen/boltzgen-1",
        "training_data": (
            "BoltzGen builds on the Boltz-2 data pipeline. The public training release includes `targets.zip` "
            "and `msa.zip` in the `boltzgen/boltzgen1_train` dataset and molecule dictionaries in "
            "`boltzgen/inference-data`; inverse folding is released as a distinct upstream training target."
        ),
        "training_procedure": (
            "Upstream BoltzGen trains a model suite covering folding, design, and inverse folding. OFoldX converts "
            "the released inverse-folding checkpoint; it does not run BoltzGen training."
        ),
    },
    "boltzgen_structuretrained_small": {
        "base_model": "boltz-community/boltz-2",
        "datasets": ("boltzgen/boltzgen1_train", "boltzgen/inference-data"),
        "hub_license": "mit",
        "source_release": "https://huggingface.co/boltzgen/boltzgen-1",
        "training_data": (
            "BoltzGen builds on the Boltz-2 data pipeline. The public training release includes `targets.zip` "
            "and `msa.zip` in the `boltzgen/boltzgen1_train` dataset and molecule dictionaries in "
            "`boltzgen/inference-data`; this artifact corresponds to the released small structure-trained checkpoint."
        ),
    },
    "ligandmpnn": {
        "hub_license": "mit",
        "training_data": (
            "LigandMPNN was trained on PDB assemblies as of 2022-12-16 from X-ray crystallography or cryo-EM "
            "structures better than 3.5 A and shorter than 6,000 residues, with 30% sequence-identity clustering "
            "and non-protein atomic context such as ligands, nucleic acids, and metals. OFoldX does not redistribute "
            "the training set."
        ),
        "training_procedure": (
            "Upstream LigandMPNN uses categorical cross-entropy sequence-design training with Adam, mixed "
            "precision/checkpointing, and released noisy-backbone checkpoint variants. OFoldX converts released "
            "LigandMPNN checkpoints into `model.safetensors`; it does not run LigandMPNN training."
        ),
    },
    "ligandmpnn_packer": {
        "hub_license": "mit",
        "training_procedure": (
            "The upstream packer predicts side-chain chi torsions from backbone, ligand atoms, and sequence with "
            "a LigandMPNN-style architecture. OFoldX converts the released side-chain packing checkpoint; it does "
            "not run training."
        ),
    },
    "proteinmpnn": {
        "hub_license": "mit",
        "repository": "https://github.com/dauparas/ProteinMPNN",
        "source_release": "https://github.com/dauparas/LigandMPNN",
        "training_data": (
            "ProteinMPNN was trained for fixed-backbone protein sequence design on PDB biounits with a 2021-08-02 "
            "data snapshot. OFoldX does not redistribute the training set."
        ),
        "training_procedure": (
            "Upstream ProteinMPNN trains graph-neural inverse-folding models with 48-neighbor checkpoints and "
            "multiple Gaussian backbone-noise variants. OFoldX converts released ProteinMPNN-family checkpoints "
            "into `model.safetensors`; it does not run ProteinMPNN training."
        ),
    },
    "rfdiffusion": {
        "hub_license": "bsd-3-clause",
        "source_release": "https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh",
        "training_data": (
            "RFdiffusion fine-tunes a RoseTTAFold-style structure network as a denoising diffusion model over "
            "PDB protein structures. OFoldX does not redistribute the training set."
        ),
        "training_procedure": (
            "Upstream RFdiffusion noises residue frames with Gaussian C-alpha translation noise and rotational "
            "Brownian motion, trains denoising to true frames with self-conditioning, and uses checkpoint-specific "
            "inference configs. OFoldX converts released RFdiffusion checkpoints into `model.safetensors`; it does "
            "not run RFdiffusion training."
        ),
    },
}


def resolve_pretrained_path(
    pretrained_name_or_path: str | PathLike[str],
    *,
    repo_type: str = "model",
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    r"""
    Return a local directory, downloading a Hub repo snapshot when needed.

    Args:
        pretrained_name_or_path: Local directory path or Hugging Face Hub repository id.
        repo_type: Hub repository type passed to ``snapshot_download``.
        revision: Optional branch, tag, or commit id to resolve.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: If ``True``, never contact the Hub and resolve only local cache entries.
        token: Optional Hugging Face authentication token.
        allow_patterns: Optional file patterns to include when downloading a Hub snapshot.

    Returns:
        Local directory path containing the requested pretrained artifact.

    Raises:
        ImportError: If a Hub repository is requested but ``huggingface-hub`` is unavailable.

    Examples:
        >>> import tempfile, os
        >>> d = tempfile.mkdtemp()
        >>> resolve_pretrained_path(d) == Path(d)
        True
        >>> resolve_pretrained_path("oteam/protein-model")  # doctest: +SKIP
        PosixPath('/home/user/.cache/huggingface/hub/models--oteam--protein-model/snapshots/...')
    """

    local_path = Path(pretrained_name_or_path).expanduser()
    if local_path.exists():
        return local_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency.
        raise ImportError("loading from the Hugging Face Hub requires huggingface-hub") from error

    return Path(
        snapshot_download(
            repo_id=str(pretrained_name_or_path),
            repo_type=repo_type,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
            token=token,
            allow_patterns=None if allow_patterns is None else list(allow_patterns),
        )
    )


def pretrained_allow_patterns(
    *patterns: str,
    trust_remote_code: bool = False,
) -> tuple[str, ...]:
    r"""
    Return file patterns needed to download a pretrained artifact.

    Args:
        *patterns: Required artifact files.
        trust_remote_code: If ``True``, include Python source files from the artifact repository.

    Returns:
        Tuple of allow-pattern strings for ``huggingface_hub.snapshot_download``. When remote code
        is trusted the ``*.py`` and ``**/*.py`` globs are appended so custom model code is fetched.

    Examples:
        >>> pretrained_allow_patterns("config.json", "*.safetensors")
        ('config.json', '*.safetensors')
        >>> pretrained_allow_patterns("config.json", trust_remote_code=True)
        ('config.json', '*.py', '**/*.py')
    """

    if not trust_remote_code:
        return patterns
    return (*patterns, "*.py", "**/*.py")


@contextmanager
def remote_code_import_path(path: str | PathLike[str], *, trust_remote_code: bool) -> Iterator[None]:
    r"""
    Temporarily add a pretrained artifact directory to ``sys.path`` when remote code is trusted.

    When ``trust_remote_code`` is ``False`` this is a no-op that leaves ``sys.path`` untouched; the
    artifact directory is inserted at the front of ``sys.path`` only for the duration of the block
    and removed again on exit, so custom model modules do not leak into later imports.

    Args:
        path: Local artifact directory.
        trust_remote_code: Whether importing Python code from ``path`` is explicitly allowed.

    Yields:
        ``None`` while imports may resolve modules from ``path``.

    Examples:
        >>> with remote_code_import_path("/tmp/model", trust_remote_code=True):  # doctest: +SKIP
        ...     from modeling_custom import CustomModel
    """

    if not trust_remote_code:
        yield
        return
    code_path = str(Path(path).resolve())
    sys.path.insert(0, code_path)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        with suppress(ValueError):
            sys.path.remove(code_path)
        importlib.invalidate_caches()


def upload_folder_to_hub(
    folder_path: str | PathLike[str],
    *,
    repo_id: str,
    repo_type: str = "model",
    private: bool | None = None,
    token: str | bool | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
    create_pr: bool = False,
    delete_existing: bool = False,
    tag: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
) -> Any:
    r"""
    Create/update a Hub repo with the contents of ``folder_path``.

    Args:
        folder_path: Local directory to upload.
        repo_id: Destination Hugging Face Hub repository id.
        repo_type: Hub repository type passed to ``HfApi``.
        private: Optional privacy setting used when creating the repository.
        token: Optional Hugging Face authentication token.
        revision: Optional target branch for the upload.
        commit_message: Optional commit message for the upload.
        create_pr: If ``True``, upload changes through a Hub pull request.
        delete_existing: If ``True``, delete an existing destination repo before recreating it.
        tag: Optional Hub tag to create after the upload.
        allow_patterns: Optional file patterns to include in the upload.
        ignore_patterns: Optional file patterns to exclude from the upload.

    Returns:
        The object returned by ``huggingface_hub.HfApi.upload_folder`` (the commit info or the
        pull-request URL when ``create_pr`` is set).

    Raises:
        ImportError: If ``huggingface-hub`` is unavailable.

    Examples:
        >>> upload_folder_to_hub("checkpoints/run-1", repo_id="oteam/protein-model")  # doctest: +SKIP
    """

    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency.
        raise ImportError("pushing to the Hugging Face Hub requires huggingface-hub") from error

    api = HfApi()
    if delete_existing:
        api.delete_repo(repo_id=repo_id, repo_type=repo_type, token=token, missing_ok=True)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, token=token, exist_ok=True)
    upload = api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=commit_message,
        token=token,
        create_pr=create_pr,
        allow_patterns=None if allow_patterns is None else list(allow_patterns),
        ignore_patterns=None if ignore_patterns is None else list(ignore_patterns),
    )
    if tag is not None:
        api.create_tag(repo_id=repo_id, repo_type=repo_type, token=token, tag=tag)
    return upload


def publish_artifact_to_hub(
    folder_path: str | PathLike[str],
    *,
    repo_id: str,
    repo_type: str = "model",
    private: bool | None = None,
    token: str | bool | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
    create_pr: bool = False,
    delete_existing: bool = False,
    tag: str | None = None,
    model_card: bool = True,
    model_card_metadata: Mapping[str, object] | None = None,
    widget_examples: Sequence[Mapping[str, object]] | None = None,
    card_body: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
) -> Any:
    r"""
    Prepare and upload a pretrained artifact directory to the Hugging Face Hub.

    This is the artifact-aware publish path: it optionally writes ``README.md`` from the root
    ``manifest.json`` and then delegates the transport step to
    [`upload_folder_to_hub`][ofoldx.hub.upload_folder_to_hub].

    Args:
        folder_path: Local artifact directory to publish.
        repo_id: Destination Hugging Face Hub repository id.
        repo_type: Hub repository type.
        private: Optional privacy setting used when creating the repository.
        token: Optional Hugging Face authentication token.
        revision: Optional target branch for the upload.
        commit_message: Optional commit message for the upload.
        create_pr: If ``True``, upload changes through a Hub pull request.
        delete_existing: If ``True``, delete an existing destination repo before recreating it.
        tag: Optional Hub tag to create after the upload.
        model_card: If ``True``, write/update ``README.md`` from the artifact manifest before upload.
        model_card_metadata: Extra model-card front matter values.
        widget_examples: Optional widget examples. ``None`` uses task defaults; an empty sequence disables widgets.
        card_body: Optional Markdown body for the generated model card.
        allow_patterns: Optional file patterns to include in the upload.
        ignore_patterns: Optional file patterns to exclude from the upload.

    Returns:
        The object returned by ``huggingface_hub.HfApi.upload_folder``.
    """

    folder = Path(folder_path)
    if model_card:
        write_model_card(
            folder,
            repo_id=repo_id,
            metadata=model_card_metadata,
            widget_examples=widget_examples,
            body=card_body,
        )
    return upload_folder_to_hub(
        folder,
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        token=token,
        revision=revision,
        commit_message=commit_message,
        create_pr=create_pr,
        delete_existing=delete_existing,
        tag=tag,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def write_model_card(
    folder_path: str | PathLike[str],
    *,
    repo_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    widget_examples: Sequence[Mapping[str, object]] | None = None,
    body: str | None = None,
) -> Path:
    r"""
    Write ``README.md`` for a saved pretrained artifact.

    The card metadata is derived from the root ``manifest.json`` and emitted as Hub-compatible YAML front
    matter. Existing Markdown body text is preserved while generated metadata is refreshed.

    Args:
        folder_path: Artifact directory containing ``manifest.json``.
        repo_id: Optional Hub repo id used in the generated title and load snippet.
        metadata: Extra front matter values to merge into the generated metadata.
        widget_examples: Optional widget examples. ``None`` uses task defaults; an empty sequence disables widgets.
        body: Optional Markdown body. When omitted, an existing body is preserved or a compact default body is used.

    Returns:
        Path to the written ``README.md``.

    Raises:
        FileNotFoundError: If the directory is missing ``manifest.json``.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME, build_artifact_manifest
        >>> with TemporaryDirectory() as directory:
        ...     build_artifact_manifest(
        ...         kind="model",
        ...         entrypoint="demo.model.DemoModel",
        ...         architecture="demo",
        ...         task="structure_prediction",
        ...     ).json(Path(directory) / ARTIFACT_MANIFEST_NAME)
        ...     path = write_model_card(directory, repo_id="oteam/demo")
        ...     path.name
        'README.md'
    """

    folder = Path(folder_path)
    manifest = read_artifact_manifest(folder)
    card_path = folder / MODEL_CARD_NAME
    existing_body = ""
    if card_path.is_file():
        _, existing_body = _split_model_card(card_path.read_text())
    card_body = (
        body
        if body is not None
        else existing_body
        or _default_model_card_body(
            manifest,
            repo_id=repo_id,
            folder_path=folder,
        )
    )
    card_metadata = model_card_metadata(
        manifest,
        repo_id=repo_id,
        metadata=metadata,
        widget_examples=widget_examples,
    )
    card_path.write_text(_format_model_card(card_metadata, card_body))
    return card_path


def model_card_metadata(
    manifest: Config,
    *,
    repo_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    widget_examples: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    r"""
    Build Hub model-card front matter for an artifact manifest.

    Args:
        manifest: Parsed artifact manifest.
        repo_id: Optional Hub repo id.
        metadata: Extra metadata values. ``tags`` are merged with generated tags.
        widget_examples: Optional widget examples. ``None`` uses task defaults; an empty sequence disables widgets.

    Returns:
        Plain dictionary suitable for YAML front matter.
    """

    task = artifact_task(manifest)
    kind = artifact_kind(manifest)
    producer_package = _producer_package(manifest)
    tags: list[object] = [*_DEFAULT_TAGS, kind]
    architecture = manifest.get("architecture")
    architecture_key = architecture if isinstance(architecture, str) else None
    title = repo_id.rsplit("/", 1)[-1] if repo_id else str(architecture or kind)
    upstream = _upstream_metadata(architecture_key, title=title, artifact_config={})
    if isinstance(architecture, str) and architecture:
        tags.append(public_architecture_name(architecture))
    if task is not None:
        tags.extend(_TASK_TAGS.get(task, (task.replace("_", "-"),)))
    card: dict[str, object] = {
        "library_name": producer_package,
        "tags": _dedupe_strings(tags),
        "artifact_kind": kind,
    }
    if repo_id is not None:
        card["repo_id"] = repo_id
    hub_license = upstream.get("hub_license")
    if isinstance(hub_license, str) and hub_license:
        card["license"] = hub_license
    base_model = upstream.get("base_model")
    if isinstance(base_model, str) and base_model:
        card["base_model"] = base_model
    datasets = upstream.get("datasets")
    if isinstance(datasets, str):
        card["datasets"] = [datasets]
    elif isinstance(datasets, Sequence) and not isinstance(datasets, str | bytes):
        card["datasets"] = _dedupe_strings(datasets)
    if task is not None:
        card["pipeline_tag"] = "other"
        card["task"] = task
    card["model-index"] = [{"name": title, "results": []}]
    if widget_examples is None:
        widget_examples = default_widget_examples(
            task,
            architecture=architecture if isinstance(architecture, str) else None,
        )
    if widget_examples:
        card["widget"] = [dict(example) for example in widget_examples]
    for key, value in ({} if metadata is None else dict(metadata)).items():
        if key == "tags":
            generated_tags = cast(Sequence[object], card["tags"])
            card["tags"] = _dedupe_strings([*generated_tags, *_as_sequence(value)])
        elif key == "widget" and value is None:
            card.pop("widget", None)
        else:
            card[key] = value
    return card


def default_widget_examples(
    task: str | None,
    *,
    architecture: str | None = None,
) -> tuple[dict[str, object], ...]:
    r"""
    Return built-in widget examples for a task.

    Args:
        task: Manifest task name, with hyphens or underscores accepted.
        architecture: Optional architecture key used to specialize otherwise broad task examples.

    Returns:
        Tuple of widget example dictionaries. Unknown tasks return an empty tuple.
    """

    if task is None:
        return ()
    normalized = task.strip().lower().replace("-", "_")
    architecture_key = None if architecture is None else architecture.strip().lower()
    if architecture_key is not None and architecture_key in _ARCHITECTURE_WIDGET_TEXT:
        examples = _ARCHITECTURE_WIDGET_TEXT[architecture_key]
    else:
        examples = _DEFAULT_WIDGET_TEXT.get(normalized, ())
    return tuple({"pipeline_tag": "other", "task": normalized, **example} for example in examples)


def public_architecture_name(architecture: str) -> str:
    r"""
    Return the public card/tag name for an internal architecture key.

    Internal registry keys use Python-friendly underscores; Hub-facing metadata should read like repository names.
    """

    key = architecture.strip().lower()
    return _PUBLIC_ARCHITECTURE_NAMES.get(key, key.replace("_", "-"))


def add_hub_upload_arguments(parser: Any) -> None:
    r"""
    Add common Hugging Face Hub upload arguments to a conversion CLI parser.

    Args:
        parser: ``argparse.ArgumentParser``-compatible object.
    """

    group = parser.add_argument_group("Hub upload")
    group.add_argument("--push-to-hub", action="store_true", help="Upload the converted artifact to the Hub")
    group.add_argument("--repo-id", default=None, help="Destination Hub repository id, e.g. oteam/model-name")
    group.add_argument("--private", action="store_true", help="Create the destination repo as private")
    group.add_argument("--token", default=None, help="Hugging Face token")
    group.add_argument("--revision", default=None, help="Target branch for upload_folder")
    group.add_argument("--create-pr", action="store_true", help="Upload through a Hub pull request")
    group.add_argument("--delete-existing", action="store_true", help="Delete an existing repo before upload")
    group.add_argument("--tag", default=None, help="Create this Hub tag after upload")
    group.add_argument(
        "--no-model-card",
        dest="model_card",
        action="store_false",
        default=True,
        help="Skip generated README.md/model-card metadata",
    )


def upload_artifact_from_args(folder_path: str | PathLike[str], args: Any, *, commit_message: str | None = None) -> Any:
    r"""
    Upload a converted artifact using attributes populated by ``add_hub_upload_arguments``.

    Args:
        folder_path: Artifact directory to upload.
        args: Parsed CLI namespace.
        commit_message: Optional fallback commit message.

    Returns:
        ``None`` when ``--push-to-hub`` was not set; otherwise the upload result.
    """

    if not getattr(args, "push_to_hub", False):
        return None
    repo_id = getattr(args, "repo_id", None)
    if not repo_id:
        raise ValueError("--repo-id is required with --push-to-hub")
    return publish_artifact_to_hub(
        folder_path,
        repo_id=repo_id,
        private=True if getattr(args, "private", False) else None,
        token=getattr(args, "token", None),
        revision=getattr(args, "revision", None),
        commit_message=commit_message,
        create_pr=getattr(args, "create_pr", False),
        delete_existing=getattr(args, "delete_existing", False),
        tag=getattr(args, "tag", None),
        model_card=getattr(args, "model_card", True),
    )


def _producer_package(manifest: Config) -> str:
    producer = manifest.get("producer")
    if isinstance(producer, Mapping):
        package = producer.get("package")
        if isinstance(package, str) and package:
            return package
    return "ofoldx"


def _default_model_card_body(
    manifest: Config,
    *,
    repo_id: str | None,
    folder_path: str | PathLike[str] | None = None,
) -> str:
    kind = artifact_kind(manifest)
    task = artifact_task(manifest)
    architecture = manifest.get("architecture")
    title = repo_id.rsplit("/", 1)[-1] if repo_id else str(architecture or kind)
    package = _producer_package(manifest)
    loader = _loader_snippet(kind, package, repo_id)
    architecture_key = architecture if isinstance(architecture, str) else None
    architecture_name = None if architecture_key is None else public_architecture_name(architecture_key)
    task_summary = _task_summary(task)
    repo_url = None if repo_id is None else f"https://huggingface.co/{repo_id}"
    source = repo_id or "./artifact"
    artifact_config = _read_artifact_config(manifest, folder_path=folder_path)
    upstream = _upstream_metadata(architecture_key, title=title, artifact_config=artifact_config)
    code_link = _code_link(manifest)
    specification = _model_specification_rows(
        manifest,
        repo_id=repo_id,
        artifact_config=artifact_config,
        upstream=upstream,
    )
    provenance_note = _provenance_note(title, upstream=upstream, artifact_config=artifact_config)
    lines = [
        f"# {title}",
        "",
        _opening_summary(kind=kind, task_summary=task_summary, architecture_name=architecture_name),
        "",
        "## Disclaimer",
        "",
        f"This model card was generated by the OFoldX team for an OFoldX `{kind}` artifact.",
        "The upstream model authors did not write this card unless explicitly stated otherwise.",
        "",
        "OFoldX is pre-alpha research software. Check the source checkpoint, upstream release, and local validation",
        "before using the artifact for scientific or operational decisions.",
        "",
        "## Model Details",
        "",
        _architecture_summary(architecture_key, kind=kind, task_summary=task_summary),
        "",
        _upstream_summary(upstream, kind=kind),
        "",
        "### Model Provenance",
        "",
        *[f"- **{field}**: {value}" for field, value in _provenance_items(upstream)],
        "",
        "### Model Specification",
        "",
        "| Field | Value |",
        "| ----- | ----- |",
        *[f"| {field} | {value} |" for field, value in specification],
    ]
    if provenance_note is not None:
        lines.extend(["", "> [!NOTE]", f"> {provenance_note}"])
    lines.extend(
        [
            "",
            "### Links",
            "",
        ]
    )
    if repo_url is not None:
        lines.append(f"- **Hub repository**: [{repo_id}]({repo_url})")
    upstream_links = _upstream_links(upstream)
    if upstream_links:
        lines.extend(upstream_links)
    lines.extend(
        [
            f"- **Code**: {code_link}",
            f"- **Project repository**: [{_PROJECT_REPOSITORY}]({_PROJECT_REPOSITORY})",
            f"- **Issues**: [{_PROJECT_ISSUES}]({_PROJECT_ISSUES})",
            "",
            "## Usage",
            "",
            f"The artifact depends on the [`{package}`]({_PROJECT_REPOSITORY}) library. Install it with pip:",
            "",
            "```bash",
            f"pip install {package}",
            "```",
            "",
            "### Pipeline Usage",
            "",
        ]
    )
    if loader:
        lines.extend(
            [
                f"Load the artifact from `{source}` with the OFoldX task pipeline. Use `AutoModel` or `AutoProcessor` "
                "only when you need lower-level control:",
                "",
                "```python",
                loader,
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "When a matching processor is available, load it with `AutoProcessor.from_pretrained(...)` and pass the",
            "processed batch to the model.",
            "",
            "### Interface",
            "",
            f"- **Task**: `{task or 'unspecified'}`",
            f"- **Artifact kind**: `{kind}`",
            f"- **Architecture**: `{architecture_name or 'unspecified'}`",
            f"- **Runtime files**: {_runtime_files(manifest)}",
            "",
            "## Training Details",
            "",
            "OFoldX did not train these weights. This repository contains a converted checkpoint and OFoldX runtime",
            "metadata for loading it.",
            "",
            "### Training Data",
            "",
            _metadata_text(upstream, "training_data"),
            "",
            "### Training Procedure",
            "",
            _metadata_text(upstream, "training_procedure"),
            "",
            "## Evaluation",
            "",
            (
                "OFoldX conversion reports and contract tests validate artifact structure and checkpoint loading. "
                "Task-level scientific evaluation should be checked against the corresponding upstream model release "
                "or paper."
            ),
            "",
            "## Limitations",
            "",
            "- This artifact is distributed for research use.",
            "- Inputs must match the model-specific processor and expected biomolecular representation.",
            "- OFoldX is pre-alpha, so APIs and artifact metadata may still change before a stable release.",
            "",
            "## Citation",
            "",
            _citation_intro(upstream),
            "",
            *_citation_lines(upstream),
            "",
            "## Contact",
            "",
            f"Please use [OFoldX GitHub issues]({_PROJECT_ISSUES}) for questions or comments about this model card.",
            "",
            "## License",
            "",
            "The Hub `license` metadata, when present, reflects the source checkpoint or upstream project license. "
            "The OFoldX project license is not yet finalized.",
            _license_text(upstream),
        ]
    )
    return "\n".join(lines) + "\n"


def _loader_snippet(kind: str, package: str, repo_id: str | None) -> str:
    source = json.dumps(repo_id or "./artifact")
    if kind == "model":
        return (
            f"from {package}.pipelines import Pipeline\n\n"
            f"pipeline = Pipeline.from_pretrained({source})\n"
            f"output = pipeline(...)\n\n"
            f"from {package} import AutoModel\n\n"
            f"model = AutoModel.from_pretrained({source})"
        )
    if kind == "processor":
        return f"from {package} import AutoProcessor\n\nprocessor = AutoProcessor.from_pretrained({source})"
    if kind == "pipeline":
        return f"from {package}.pipelines import Pipeline\n\npipeline = Pipeline.from_pretrained({source})"
    return ""


def _task_summary(task: str | None) -> str:
    if task is None:
        return "biomolecular pretrained-artifact loading"
    return _TASK_SUMMARIES.get(task, task.replace("_", " "))


def _opening_summary(*, kind: str, task_summary: str, architecture_name: str | None) -> str:
    if architecture_name is None:
        return f"OFoldX `{kind}` artifact for {task_summary}."
    return f"OFoldX `{kind}` artifact for {task_summary}, using the `{architecture_name}` architecture."


def _architecture_summary(architecture: str | None, *, kind: str, task_summary: str) -> str:
    if architecture is None:
        return f"This repository contains a biomolecular `{kind}` artifact for {task_summary}."
    return _ARCHITECTURE_SUMMARIES.get(
        architecture,
        f"This repository contains a biomolecular `{kind}` artifact for {task_summary}.",
    )


def _read_artifact_config(
    manifest: Config,
    *,
    folder_path: str | PathLike[str] | None,
) -> Mapping[str, object]:
    if folder_path is None:
        return {}
    files = manifest.get("files")
    config_name = "config.json"
    if isinstance(files, Mapping):
        configured_name = files.get("config")
        if isinstance(configured_name, str) and configured_name:
            config_name = configured_name
    config_path = Path(folder_path) / config_name
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, Mapping) else {}


def _upstream_metadata(
    architecture: str | None,
    *,
    title: str,
    artifact_config: Mapping[str, object],
) -> dict[str, object]:
    key = None if architecture is None else architecture.strip().lower()
    metadata: dict[str, object] = dict(cast(Mapping[str, object], _UPSTREAM_METADATA.get(key or "", {})))
    metadata.update(cast(Mapping[str, object], _UPSTREAM_REFINEMENTS.get(key or "", {})))
    metadata.update(_repo_upstream_overrides(title))
    source_checkpoint = _source_checkpoint(title, artifact_config=artifact_config)
    if source_checkpoint is not None:
        metadata["source_checkpoint"] = source_checkpoint
    return metadata


def _repo_upstream_overrides(title: str) -> dict[str, object]:
    if title.startswith("solublempnn-noise"):
        return {
            "project": "SolubleMPNN",
            "summary": "Converted SolubleMPNN sequence-design checkpoint for soluble protein backbones.",
            "repository": "https://github.com/dauparas/ProteinMPNN/tree/main/soluble_model_weights",
            "source_release": "https://github.com/dauparas/LigandMPNN",
            "paper": "Computational design of soluble and functional membrane protein analogues",
            "paper_url": "https://doi.org/10.1038/s41586-024-07601-y",
            "training_data": (
                "The SolubleMPNN work trains like ProteinMPNN on PDB assemblies as of 2021-08-02, filtered to "
                "X-ray/cryo-EM structures better than 3.5 A and fewer than 10,000 residues, while excluding "
                "annotated transmembrane PDB entries. OFoldX does not redistribute the training set."
            ),
            "training_procedure": (
                "Upstream SolubleMPNN follows ProteinMPNN-style fixed-backbone inverse-folding training with "
                "48-neighbor noisy-backbone checkpoint variants. OFoldX converts released SolubleMPNN checkpoints "
                "into `model.safetensors`; it does not run SolubleMPNN training."
            ),
            "citation": _SOLUBLEMPNN_CITATION,
        }
    if title == "proteinmpnn-membrane-global-noise020":
        return {
            "project": "Membrane ProteinMPNN",
            "summary": "Converted membrane ProteinMPNN checkpoint with a global soluble/membrane label.",
            "repository": "https://github.com/dauparas/LigandMPNN",
            "training_data": (
                "This checkpoint is distributed in the LigandMPNN parameter release. The global-label model adds "
                "a binary soluble/membrane feature to the ProteinMPNN-style input representation. OFoldX does not "
                "redistribute the training set."
            ),
            "training_procedure": (
                "Upstream training follows the ProteinMPNN fixed-backbone inverse-folding setup with a global "
                "membrane label and a 0.20 Angstrom noisy-backbone checkpoint. OFoldX only converts the released "
                "checkpoint."
            ),
        }
    if title == "proteinmpnn-membrane-per-residue-noise020":
        return {
            "project": "Membrane ProteinMPNN",
            "summary": "Converted membrane ProteinMPNN checkpoint with per-residue membrane labels.",
            "repository": "https://github.com/dauparas/LigandMPNN",
            "training_data": (
                "This checkpoint is distributed in the LigandMPNN parameter release. The per-residue model adds "
                "three membrane-context classes per residue: buried, interface, and other. OFoldX does not "
                "redistribute the training set."
            ),
            "training_procedure": (
                "Upstream training follows the ProteinMPNN fixed-backbone inverse-folding setup with per-residue "
                "membrane labels and a 0.20 Angstrom noisy-backbone checkpoint. OFoldX only converts the released "
                "checkpoint."
            ),
        }
    return {}


def _source_checkpoint(title: str, *, artifact_config: Mapping[str, object]) -> str | None:
    if title in _SOURCE_CHECKPOINTS_BY_REPO:
        return _SOURCE_CHECKPOINTS_BY_REPO[title]
    variant = artifact_config.get("checkpoint_variant")
    if not isinstance(variant, str) or not variant:
        return None
    if variant.startswith("model_"):
        return f"alphafold_params_2022-12-06.tar::params_{variant}.npz"
    if title.startswith("rfdiffusion-"):
        return f"{variant}.pt"
    return f"{variant}.pt"


def _metadata_text(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return "Not recorded in this OFoldX artifact. See the upstream release for authoritative details."


def _runtime_files(manifest: Config) -> str:
    files = ["manifest.json"]
    manifest_files = manifest.get("files")
    if isinstance(manifest_files, Mapping):
        files.extend(str(value) for value in manifest_files.values() if isinstance(value, str))
    weights = manifest.get("weights")
    if isinstance(weights, Mapping):
        files.extend(str(value) for value in weights.values() if isinstance(value, str))
    seen: set[str] = set()
    unique: list[str] = []
    for file in files:
        if file in seen:
            continue
        seen.add(file)
        unique.append(file)
    return ", ".join(f"`{file}`" for file in unique)


def _upstream_summary(metadata: Mapping[str, object], *, kind: str) -> str:
    summary = metadata.get("summary")
    if isinstance(summary, str) and summary:
        return summary
    return f"This OFoldX `{kind}` artifact records runtime metadata for a converted upstream checkpoint."


def _provenance_items(metadata: Mapping[str, object]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    project = metadata.get("project")
    if isinstance(project, str) and project:
        rows.append(("Upstream Project", project))
    source_checkpoint = metadata.get("source_checkpoint")
    if isinstance(source_checkpoint, str) and source_checkpoint:
        rows.append(("Source Checkpoint", f"`{source_checkpoint}`"))
    source_release = metadata.get("source_release")
    if isinstance(source_release, str) and source_release:
        rows.append(("Source Release", f"[{source_release}]({source_release})"))
    paper = metadata.get("paper")
    paper_url = metadata.get("paper_url")
    if isinstance(paper, str) and paper:
        rows.append(("Primary Paper", _markdown_link_or_text(paper, paper_url)))
    license_name = metadata.get("license")
    if isinstance(license_name, str) and license_name:
        rows.append(("Upstream License", license_name))
    if not rows:
        rows.append(("Upstream Project", "Not recorded in this OFoldX artifact"))
    return rows


def _upstream_links(metadata: Mapping[str, object]) -> list[str]:
    links: list[str] = []
    paper = metadata.get("paper")
    paper_url = metadata.get("paper_url")
    if isinstance(paper, str) and paper and isinstance(paper_url, str) and paper_url:
        links.append(f"- **Upstream paper**: [{paper}]({paper_url})")
    repository = metadata.get("repository")
    if isinstance(repository, str) and repository:
        project = metadata.get("project")
        label = project if isinstance(project, str) and project else repository
        links.append(f"- **Upstream repository**: [{label}]({repository})")
    source_release = metadata.get("source_release")
    if isinstance(source_release, str) and source_release:
        links.append(f"- **Source checkpoint release**: [{source_release}]({source_release})")
    return links


def _markdown_link_or_text(label: str, url: object) -> str:
    if isinstance(url, str) and url:
        return f"[{label}]({url})"
    return label


def _citation_intro(metadata: Mapping[str, object]) -> str:
    project = metadata.get("project")
    if isinstance(project, str) and project:
        return (
            f"Please cite the upstream {project} work for the source checkpoint. If OFoldX supports your work, "
            "please also cite or link the OFoldX project repository."
        )
    return (
        "Please cite the upstream model or paper associated with the source checkpoint. If OFoldX supports your work, "
        "please also cite or link the OFoldX project repository."
    )


def _citation_lines(metadata: Mapping[str, object]) -> list[str]:
    citation = metadata.get("citation")
    if not isinstance(citation, str) or not citation:
        return []
    return ["```bibtex", dedent(citation).strip(), "```"]


def _license_text(metadata: Mapping[str, object]) -> str:
    license_name = metadata.get("license")
    if isinstance(license_name, str) and license_name:
        return (
            f"The source checkpoint is associated with the upstream license noted above: {license_name}. "
            "Review both OFoldX and upstream terms before redistribution or production use."
        )
    return (
        "The source checkpoint may carry additional upstream license terms; review both OFoldX and upstream terms "
        "before redistribution or production use."
    )


def _model_specification_rows(
    manifest: Config,
    *,
    repo_id: str | None,
    artifact_config: Mapping[str, object] | None = None,
    upstream: Mapping[str, object] | None = None,
) -> list[tuple[str, str]]:
    kind = artifact_kind(manifest)
    task = artifact_task(manifest)
    architecture = manifest.get("architecture")
    files = manifest.get("files")
    weights = manifest.get("weights")
    artifact_config = {} if artifact_config is None else artifact_config
    upstream = {} if upstream is None else upstream
    rows = [
        ("Repository", f"`{repo_id}`" if repo_id else "`local artifact`"),
        ("Artifact Kind", f"`{kind}`"),
        ("Task", f"`{task}`" if task else "`unspecified`"),
        (
            "Architecture",
            (
                f"`{public_architecture_name(architecture)}`"
                if isinstance(architecture, str) and architecture
                else "`unspecified`"
            ),
        ),
        ("Entrypoint", f"`{entrypoint(manifest)}`"),
    ]
    if isinstance(files, Mapping):
        config = files.get("config")
        if isinstance(config, str):
            rows.append(("Config", f"`{config}`"))
        conversion_config = files.get("conversion_config")
        if isinstance(conversion_config, str):
            rows.append(("Conversion Config", f"`{conversion_config}`"))
    if isinstance(weights, Mapping):
        default_weights = weights.get("default")
        if isinstance(default_weights, str):
            rows.append(("Weights", f"`{default_weights}`"))
    source_checkpoint = upstream.get("source_checkpoint")
    if isinstance(source_checkpoint, str) and source_checkpoint:
        rows.append(("Source Checkpoint", f"`{source_checkpoint}`"))
    checkpoint_variant = artifact_config.get("checkpoint_variant")
    if isinstance(checkpoint_variant, str) and checkpoint_variant:
        rows.append(("Checkpoint Variant", f"`{checkpoint_variant}`"))
    for key, label in (
        ("k_neighbors", "k-Neighbors"),
        ("atom_context_num", "Atom Context Count"),
        ("node_feature_size", "Node Feature Size"),
        ("num_mix", "Side-chain Mixture Components"),
        ("num_rbf", "Side-chain RBF Features"),
    ):
        value = artifact_config.get(key)
        if isinstance(value, int | float | str):
            rows.append((label, f"`{value}`"))
    noise_level = artifact_config.get("noise_level")
    if isinstance(noise_level, int | float):
        rows.append(("Training Noise", f"`{_format_noise_angstrom(noise_level)}`"))
    return rows


def _code_link(manifest: Config) -> str:
    target = entrypoint(manifest)
    module_name, _, _ = target.rpartition(".")
    path = module_name.replace(".", "/") + ".py"
    if path.startswith("ofoldx/"):
        return f"[`{path}`]({_PROJECT_REPOSITORY}/tree/main/{path})"
    return f"`{target}`"


def _provenance_note(
    title: str,
    *,
    upstream: Mapping[str, object],
    artifact_config: Mapping[str, object],
) -> str | None:
    if title == "protenix-base":
        return (
            "Source checkpoint: `protenix_base_default_v0.5.0.pt`; `v0.5.0` is the upstream Protenix "
            "checkpoint release version, not part of the public repo id."
        )
    source_checkpoint = upstream.get("source_checkpoint")
    if isinstance(source_checkpoint, str) and source_checkpoint:
        noise = _noise_from_config_or_checkpoint(artifact_config, source_checkpoint)
        if noise is not None:
            return (
                f"Source checkpoint: `{source_checkpoint}`; the public repo suffix records the upstream "
                f"Gaussian noise variant ({noise})."
            )
        return f"Source checkpoint: `{source_checkpoint}`."
    if title.startswith(("proteinmpnn-noise", "solublempnn-noise")):
        return "Checkpoint metadata: `k_neighbors=48`; the `noiseXXX` suffix identifies the training-noise variant."
    if title.startswith("proteinmpnn-membrane-"):
        return "Checkpoint metadata: `k_neighbors=48`; the `noiseXXX` suffix identifies the training-noise variant."
    if title.startswith("ligandmpnn-noise"):
        return (
            "Checkpoint metadata: `k_neighbors=32`, `atom_context_num=25`; the `noiseXXX` suffix identifies "
            "the training-noise variant."
        )
    if title.startswith("ligandmpnn-packer-noise"):
        return (
            "Checkpoint metadata: `k_neighbors=32`, `atom_context_num=16`; the `noiseXXX` suffix identifies "
            "the training-noise variant."
        )
    return None


def _noise_from_config_or_checkpoint(
    artifact_config: Mapping[str, object],
    source_checkpoint: str,
) -> str | None:
    noise_level = artifact_config.get("noise_level")
    if isinstance(noise_level, int | float):
        return _format_noise_angstrom(noise_level)
    return _noise_from_checkpoint(source_checkpoint)


def _noise_from_checkpoint(source_checkpoint: str) -> str | None:
    stem = source_checkpoint.removesuffix(".pt").removesuffix(".ckpt")
    parts = stem.split("_")
    token = next((part for part in reversed(parts) if part.isdigit() and len(part) == 3), "")
    if not token.isdigit() or len(token) != 3:
        return None
    return _format_noise_angstrom(int(token) / 100)


def _format_noise_angstrom(noise_level: int | float) -> str:
    return f"{float(noise_level):.2f} Angstrom"


def _split_model_card(content: str) -> tuple[str, str]:
    if not content.startswith("---\n"):
        return "", content
    end = content.find("\n---\n", 4)
    if end == -1:
        return "", content
    return content[4:end], content[end + len("\n---\n") :]


def _format_model_card(metadata: Mapping[str, object], body: str) -> str:
    body = body if body.endswith("\n") else f"{body}\n"
    return "---\n" + "\n".join(_yaml_lines(metadata)) + "\n---\n\n" + body.lstrip("\n")


def _yaml_lines(value: object, *, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if _is_scalar(item):
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_lines(item, indent=indent + 2))
        return lines
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if not value:
            return [f"{pad}[]"]
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{pad}- {_yaml_scalar(item)}")
            elif isinstance(item, Mapping):
                item_lines = _yaml_lines(item, indent=indent + 2)
                if item_lines:
                    first = item_lines[0].lstrip()
                    lines.append(f"{pad}- {first}")
                    lines.extend(item_lines[1:])
                else:
                    lines.append(f"{pad}- {{}}")
            else:
                lines.append(f"{pad}-")
                lines.extend(_yaml_lines(item, indent=indent + 2))
        return lines
    return [f"{pad}{_yaml_scalar(value)}"]


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value))


def _dedupe_strings(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return (value,)
