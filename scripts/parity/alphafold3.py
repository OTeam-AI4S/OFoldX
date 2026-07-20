#!/usr/bin/env python
r"""DeepMind AlphaFold 3 parity and public-surface acceptance harness.

This is the DeepMind AlphaFold 3 identity (distinct from the Protenix graph). The
native surface mirrors the wired :mod:`scripts.parity.protenix` sibling; the
cross-environment capture mirrors :mod:`scripts.parity.alphafold2`, because the
AlphaFold 3 reference is DeepMind's JAX / Haiku implementation rather than a
PyTorch checkout and can only run inside its own conda environment.

Because the reference is a different framework, parity follows the
RFdiffusion-proven ``reference`` -> capture -> ``forward`` compare flow. The
``reference`` (``--dump-reference``) stage runs the released DeepMind forward in
its own environment and serialises the *deterministic trunk* tensors --- the
Evoformer ``single``/``pair`` embeddings and the distogram logits (exposed with
``return_embeddings=True`` / ``return_distogram=True``) --- together with the exact
featurised inputs, into a NumPy pickle (the AlphaFold 3 environment has no torch).
The ``forward`` (``--load-reference``) stage then runs in the OFoldX (zyc)
environment: it converts the official checkpoint and drives the native head on the
*reference's own* trunk tensors, so the two frameworks meet on byte-identical input
and the distogram-logit comparison is a real forward-vs-forward check at
``FORWARD_ATOL`` / ``FORWARD_RTOL`` = 1e-4.

Two stages need no reference and no weights and are always runnable in zyc CPU:
``semantic`` exercises the ``AlphaFold3Processor`` featurisation invariants on a
canonical mixed system, and ``smoke`` drives the full
processor -> explicit-noise ``forward`` -> ``fold`` -> ``FoldResult`` surface with
tiny random weights.

Honest scope: the ``forward`` comparison is exact because it feeds the reference's
own ``pair`` embedding into the native distogram head, so it does not depend on
featurisation agreement. The full native trunk (native ``single``/``pair`` from the
native processor) is NOT compared bitwise: AlphaFold 3 sampling is stochastic, its
JAX diffusion RNG does not match the PyTorch EDM sampler, and the DeepMind data
pipeline and the OFoldX ``AlphaFold3Processor`` do not yet share a byte-identical
featurisation, so the native trunk input differs from the reference trunk input.
Those sampled-surface identities stay on the separate coarse ``SURFACE`` tolerance.
Captures and JSON reports are local run outputs; keep them under the ignored
``artifacts/parity`` directory or another scratch directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.paths import reference_root, weight_path  # noqa: E402

# The official checkpoint is the single gated ``<name>.bin.zst`` DeepMind release.
# The native converter accepts the file directly; the reference JAX loader wants the
# directory that contains it (:func:`_reference_model_dir` isolates one via a symlink).
DEFAULT_MODEL_DIR = weight_path("af3.bin.zst")
REFERENCE_ROOT = reference_root("alphafold3")
SEQUENCE = "ACDEFGHIK"
MODEL_NAME = "af3"
SEED = 7
RECYCLING_STEPS = 1
# Seed for the FIXED denoiser initial-noise tensor dumped by ``reference-ext``.  The actual
# noise array is serialised alongside the seed so the torch side reuses byte-identical values.
NOISE_SEED = 42

# The reference capture carries the Evoformer ``single`` / ``pair`` embeddings for provenance;
# the distogram logits are the byte-identical-input forward-vs-forward numeric target that the
# ``forward`` comparison drives through the native head.
#
# Chosen before executing the maintained matrix.  All comparisons are FP32.  The FORWARD
# tolerance governs the distogram head fed the reference's own ``pair`` embedding.
FORWARD_ATOL = 1e-4
FORWARD_RTOL = 1e-4
# Sampled-surface agreement across JAX (reference) and PyTorch (native) is coarse:
# the diffusion RNG and the two featurisers differ, so the pooled predicted
# structure and confidences are only shape/coarse-numerically comparable.
SURFACE_ATOL = 5e-1
SURFACE_RTOL = 5e-1
# The ``processor`` mode checks that the native AlphaFold3Processor packs the same channels as
# DeepMind's featurise_input on the same system: integer / one-hot channels must be exact, float
# channels agree at PROCESSOR_ATOL.  Chain/entity/residue indices are compared modulo a constant
# offset (DeepMind is 1-based, native 0-based) because the model only reads their differences.
PROCESSOR_ATOL = 1e-5

# These gate only the coarse SURFACE (sampled-structure) ``compare`` verdict.  The
# ``forward`` distogram comparison is exact and is NOT gated by them.
CURRENT_BLOCKERS = (
    "AlphaFold 3 diffusion sampling is stochastic and its JAX RNG does not match "
    "the native PyTorch EDM sampler, so pooled coordinates cannot match bitwise.",
    "The reference DeepMind data pipeline and the OFoldX AlphaFold3Processor do not "
    "yet share a byte-identical featurisation, so the native trunk input differs from "
    "the reference trunk input; native single/pair are captured but not compared bitwise.",
)


class Blocked(RuntimeError):
    """A required production surface is not implemented or not runnable here."""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=(
            "semantic",
            "smoke",
            "native",
            "reference",
            "reference-ext",
            "forward",
            "native-trunk",
            "ext-compare",
            "processor-reference",
            "processor",
            "compare",
        ),
    )
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--recycling-steps", type=int, default=RECYCLING_STEPS)
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=NOISE_SEED,
        help="reference-ext mode: JAX PRNG seed for the FIXED denoiser initial-noise tensor",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flash-attention", default="xla", help="reference JAX attention kernel (xla for CPU)")
    parser.add_argument(
        "--golden",
        action="store_true",
        help="reference mode: forward the DeepMind-shipped test_data/featurised_example.pkl "
        "instead of a live featurisation (the live pipeline needs external MSA/CCD state)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument(
        "--dump-reference", type=Path, help="reference mode: path for the reference forward-tensor capture"
    )
    parser.add_argument(
        "--load-reference", type=Path, help="forward mode: the reference forward-tensor capture to compare against"
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--atol", type=float, default=SURFACE_ATOL)
    parser.add_argument("--rtol", type=float, default=SURFACE_RTOL)
    args = parser.parse_args()

    sequence = args.sequence.strip().upper()
    if not sequence:
        parser.error("--sequence must not be empty")

    if args.mode == "semantic":
        semantic_only()
    elif args.mode == "smoke":
        smoke_surface(sequence=sequence, seed=args.seed, device_name=args.device)
    elif args.mode == "native":
        if args.output is None:
            parser.error("native mode requires --output")
        native_capture(
            sequence=sequence,
            model_dir=args.model_dir,
            model_name=args.model_name,
            seed=args.seed,
            device_name=args.device,
            output=args.output,
        )
    elif args.mode == "reference":
        output = args.dump_reference or args.output
        if output is None:
            parser.error("reference mode requires --dump-reference (or --output)")
        reference_dump(
            reference=args.reference_root,
            sequence=sequence,
            model_dir=args.model_dir,
            model_name=args.model_name,
            seed=args.seed,
            recycling_steps=args.recycling_steps,
            flash_attention=args.flash_attention,
            golden=args.golden,
            output=output,
        )
    elif args.mode == "reference-ext":
        output = args.dump_reference or args.output
        if output is None:
            parser.error("reference-ext mode requires --dump-reference (or --output)")
        reference_ext_dump(
            sequence=sequence,
            model_dir=args.model_dir,
            model_name=args.model_name,
            seed=args.seed,
            noise_seed=args.noise_seed,
            recycling_steps=args.recycling_steps,
            flash_attention=args.flash_attention,
            golden=args.golden,
            output=output,
        )
    elif args.mode == "forward":
        reference_output = args.load_reference or args.reference_output
        if reference_output is None:
            parser.error("forward mode requires --load-reference (the reference forward-tensor capture)")
        forward_compare(
            reference_output=reference_output,
            model_dir=args.model_dir,
            model_name=args.model_name,
            device_name=args.device,
            json_path=args.json,
        )
    elif args.mode == "native-trunk":
        reference_output = args.load_reference or args.reference_output
        if reference_output is None:
            parser.error("native-trunk mode requires --load-reference (the reference golden capture)")
        native_trunk_compare(
            reference_output=reference_output,
            model_dir=args.model_dir,
            model_name=args.model_name,
            device_name=args.device,
            recycling_steps=args.recycling_steps,
            json_path=args.json,
        )
    elif args.mode == "ext-compare":
        reference_output = args.load_reference or args.reference_output
        if reference_output is None:
            parser.error("ext-compare mode requires --load-reference (the reference-ext golden capture)")
        ext_compare(
            reference_output=reference_output,
            model_dir=args.model_dir,
            model_name=args.model_name,
            device_name=args.device,
            recycling_steps=args.recycling_steps,
            json_path=args.json,
        )
    elif args.mode == "processor-reference":
        output = args.dump_reference or args.output
        if output is None:
            parser.error("processor-reference mode requires --dump-reference (or --output)")
        processor_reference_dump(sequence=sequence, seed=args.seed, output=output)
    elif args.mode == "processor":
        reference_output = args.load_reference or args.reference_output
        if reference_output is None:
            parser.error("processor mode requires --load-reference (the DeepMind featurisation capture)")
        processor_compare(
            reference_output=reference_output,
            sequence=sequence,
            device_name=args.device,
            json_path=args.json,
        )
    else:
        if args.reference_output is None or args.output is None:
            parser.error("compare mode requires --reference-output (reference) and --output (native)")
        compare_captures(
            reference_output=args.reference_output,
            native_output=args.output,
            json_path=args.json,
            atol=args.atol,
            rtol=args.rtol,
        )
    return 0


def semantic_only() -> None:
    """Exercise the AlphaFold 3 featurisation invariants without weights."""

    import numpy as np

    from ofoldx.data import (
        MSA,
        AtomSite,
        ChemicalAtom,
        ChemicalBond,
        ChemicalComponent,
        MolecularSystem,
        Molecule,
        PolymerSequence,
        SystemBond,
    )
    from ofoldx.models.alphafold3 import AlphaFold3Processor
    from ofoldx.models.alphafold3.model import AlphaFold3Input

    protein = PolymerSequence.from_one_letter("AC", polymer_type="protein")
    rna = PolymerSequence.from_one_letter("AG", polymer_type="rna")
    dna = PolymerSequence.from_one_letter("CT", polymer_type="dna")
    ligand = Molecule(
        components=(
            ChemicalComponent(
                component_id="LIG",
                atoms=(
                    ChemicalAtom("C1", "C", formal_charge=0),
                    ChemicalAtom("O1", "O", formal_charge=0),
                ),
                bonds=frozenset({ChemicalBond("C1", "O1", order=2.0)}),
                reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
            ),
        ),
    )
    system = MolecularSystem(
        entities={"protein": protein, "rna": rna, "dna": dna, "ligand": ligand},
        instances={"P": "protein", "R": "rna", "D": "dna", "L": "ligand"},
        bonds=frozenset({SystemBond(AtomSite("P", 1, "SG"), AtomSite("L", 0, "C1"))}),
    )
    msa = MSA(
        entity_id="protein",
        unpaired_sequences=("AC", "AG"),
        unpaired_deletion_counts=np.array([[0, 0], [0, 1]], dtype=np.uint32),
        paired_sequences=("AC",),
        paired_deletion_counts=np.zeros((1, 2), dtype=np.uint32),
    )

    batch = AlphaFold3Processor()(system, msa=msa)
    if not isinstance(batch, AlphaFold3Input):
        raise AssertionError(f"processor returned {type(batch).__name__}, not AlphaFold3Input")

    # The released AlphaFold 3 target restype one-hot spans 31 classes and the MSA one-hot
    # stack spans 32 classes plus has_deletion + deletion_value (34).  token_features is
    # [zeros(384) | restype(31) | profile(31) | deletion_mean(1)] = 447.
    token_width = int(batch.token_features.shape[-1])
    if token_width != 447:
        raise AssertionError(f"token_features width {token_width} != released 447")
    if batch.msa_features is None or batch.msa_mask is None:
        raise AssertionError("supplied protein MSA did not populate msa_features / msa_mask")
    msa_width = int(batch.msa_features.shape[-1])
    if msa_width != 34:
        raise AssertionError(f"msa_features width {msa_width} != one-hot(32) + has_deletion + deletion_value")
    if batch.msa_features.shape[-2] != batch.token_features.shape[-2]:
        raise AssertionError("msa_features token axis is not aligned to the token count")

    # A small self-template exercises the pre-packed TemplateFeatures the released
    # TemplateEmbedder consumes (39 distogram bins over the aligned residue pairs).
    _semantic_template(np)
    print(
        "semantic AlphaFold3 mixed protein/RNA/DNA/ligand: "
        f"tokens={batch.token_features.shape[-2]} atoms={batch.atom_features.shape[-2]} "
        f"token_feat={token_width} msa_feat={msa_width}",
        flush=True,
    )


def _semantic_template(np: Any) -> None:
    import torch

    from ofoldx.data import PolymerSequence
    from ofoldx.data.evidence import Template, TemplateAlignment
    from ofoldx.data.molecular import MolecularSystem
    from ofoldx.data.structure import Structure
    from ofoldx.models.alphafold3 import AlphaFold3Processor
    from ofoldx.modules.embedding import TemplateFeatures

    target = PolymerSequence.from_one_letter("ACD", polymer_type="protein")
    template_seq = PolymerSequence.from_one_letter("ACD", polymer_type="protein")
    template_system = MolecularSystem(entities={"T": template_seq}, instances={"T": "T"})
    sites = tuple(site for site, _atom in template_system.iter_atoms())
    coordinates = np.zeros((len(sites), 3), dtype=np.float32)
    for index, site in enumerate(sites):
        coordinates[index] = (float(site.component_index), 0.0, 0.0)
    template_structure = Structure(
        system=template_system,
        sites=sites,
        coordinates=coordinates,
        coordinate_mask=np.ones(len(sites), dtype=np.bool_),
    )
    template = Template(
        structure=template_structure,
        alignments=(
            TemplateAlignment(
                target_entity_id="A",
                target_instance_id=None,
                template_instance_id="T",
                residue_pairs=((0, 0), (1, 1), (2, 2)),
            ),
        ),
    )
    batch = AlphaFold3Processor()(target, templates=(template,))
    if not isinstance(batch.template_features, TemplateFeatures):
        raise AssertionError("templates did not pre-pack a TemplateFeatures")
    distogram_bins = int(batch.template_features.distogram.shape[-1])
    if distogram_bins != 39:
        raise AssertionError(f"template distogram bins {distogram_bins} != released 39")
    if int(batch.template_features.distogram.shape[1]) != 1:
        raise AssertionError("expected exactly one packed template frame")
    _ = torch  # imported for its side effect of asserting the tensor backend is present


def smoke_surface(*, sequence: str, seed: int, device_name: str) -> None:
    """Drive processor -> explicit-noise forward -> fold -> FoldResult with tiny random weights."""

    import torch

    from ofoldx.models.alphafold3 import AlphaFold3Model, AlphaFold3Processor
    from ofoldx.models.alphafold3.model import AlphaFold3FoldResult

    device = _device(device_name)
    config = _tiny_config()
    torch.manual_seed(seed)
    model = AlphaFold3Model(config).set_backend("torch").to(device).eval()
    processor = AlphaFold3Processor(device=device)
    batch = processor(sequence)

    atom_count = batch.atom_features.shape[-2]
    noisy = torch.randn(1, 1, atom_count, 3, generator=torch.Generator().manual_seed(seed + 1)).to(device)
    noise = torch.ones(1, 1, device=device)
    with torch.inference_mode():
        output = model(batch, noisy_coordinates=noisy, noise_level=noise, recycling_steps=1)
    for name in ("coordinates", "distogram_logits", "plddt_logits", "pae_logits", "pde_logits", "resolved_logits"):
        value = getattr(output, name)
        if not torch.isfinite(value).all():
            raise AssertionError(f"explicit-noise forward produced non-finite {name}")

    fold_generator = torch.Generator(device=batch.atom_features.device).manual_seed(seed)
    result = model.fold(batch, num_samples=2, generator=fold_generator, recycling_steps=1, num_sampling_steps=3)
    _check_fold_result(result, num_samples=2)
    if not isinstance(result, AlphaFold3FoldResult):
        raise AssertionError("fold did not return an AlphaFold3FoldResult")
    best_ranking = result.best("ranking_score").scores["ranking_score"].value
    prediction_count = sum(len(group.predictions) for group in result.seed_groups)
    print(
        f"smoke AlphaFold3 surface: tokens={len(batch.token_sites)} atoms={len(batch.atom_sites)} "
        f"predictions={prediction_count} best_ranking={best_ranking:.6g}",
        flush=True,
    )


def native_capture(
    *,
    sequence: str,
    model_dir: Path,
    model_name: str,
    seed: int,
    device_name: str,
    output: Path,
) -> None:
    """Serialise the converted-weight native surface for cross-environment comparison."""

    import numpy as np
    import torch

    from ofoldx.models.alphafold3 import AlphaFold3Processor
    from ofoldx.models.alphafold3.convert import convert_alphafold3_checkpoint

    device = _device(device_name)
    # The official DeepMind AlphaFold 3 checkpoint is mixed precision (most trunk
    # weights are bfloat16, the rest float32).  The reference casts trunk activations
    # into a bfloat16 context at run time; the native surface instead upcasts every
    # parameter to float32 so the whole graph runs homogeneously at the CPU/fp32
    # precision this parity half targets.  Without this the fp32 processor features
    # meet bfloat16 Linear weights and the forward raises a dtype mismatch.
    model = convert_alphafold3_checkpoint(
        _local_path(model_dir, "model checkpoint"), model_name=model_name, dtype=torch.float32
    )
    model = model.to(device).eval().set_backend("torch")
    processor = AlphaFold3Processor(device=device)
    batch = processor(sequence)

    fold_generator = torch.Generator(device=batch.atom_features.device).manual_seed(seed)
    result = model.fold(batch, num_samples=1, generator=fold_generator)
    _check_fold_result(result, num_samples=1)
    prediction = result.seed_groups[0].predictions[0]
    outputs = {
        "predicted_atom_positions": torch.from_numpy(prediction.structure.coordinates.copy()),
        "predicted_atom_mask": torch.from_numpy(prediction.structure.coordinate_mask.copy()),
        "plddt": torch.from_numpy(np.asarray(prediction.metrics["plddt"].values, dtype=np.float32)),
        "pae": torch.from_numpy(np.asarray(prediction.pair_metrics["pae"].values, dtype=np.float32)),
        "ptm": torch.tensor(float(prediction.scores["ptm"].value)),
        "iptm": torch.tensor(float(prediction.scores["iptm"].value)),
        "ranking_score": torch.tensor(float(prediction.scores["ranking_score"].value)),
        "mean_plddt": torch.tensor(float(prediction.scores["plddt"].value)),
    }
    payload = {
        "metadata": _metadata(sequence=sequence, model_name=model_name, seed=seed),
        "golden_inputs": _golden_inputs(batch),
        "outputs": {name: value.detach().float().cpu() for name, value in outputs.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"mode": "native", "output": str(output), **payload["metadata"]}, indent=2, sort_keys=True))


def reference_dump(
    *,
    reference: Path,
    sequence: str,
    model_dir: Path,
    model_name: str,
    seed: int,
    recycling_steps: int,
    flash_attention: str,
    golden: bool,
    output: Path,
) -> None:
    r"""Run the released DeepMind AlphaFold 3 forward in its own env and dump trunk tensors.

    Serialises the deterministic Evoformer ``single`` / ``pair`` embeddings and the distogram
    logits (``return_embeddings`` / ``return_distogram``) plus the exact featurised inputs, as a
    NumPy pickle --- the AlphaFold 3 JAX environment ships no torch, so the capture stays
    framework-neutral and the zyc ``forward`` half reloads it with NumPy.
    """

    import pickle

    _activate_reference_env()
    try:
        import haiku as hk
        import jax
        import numpy as np
        from alphafold3.model import model as af3_model
        from alphafold3.model import params as af3_params
    except Exception as error:  # noqa: BLE001 -- absent JAX env degrades to a clean Blocked
        raise Blocked(f"the AlphaFold 3 JAX environment is unavailable: {type(error).__name__}: {error}") from error

    # tokamax (pulled in by the diffusion transformer's gated-linear-unit) lazily forces absl to
    # parse ``sys.argv`` the first time a tokamax config flag is read during the forward; our argv
    # carries argparse flags (``--golden`` etc.) that absl does not recognise, so pre-mark FLAGS as
    # parsed against the program name alone to keep that lazy reparse a no-op.
    from absl import flags as _absl_flags

    if not _absl_flags.FLAGS.is_parsed():
        _absl_flags.FLAGS([sys.argv[0]])

    provenance = _reference_provenance(af3_model)
    print(f"SOURCE alphafold3 module={provenance['module']} version={provenance['version']}", flush=True)

    batch, feature_source = _reference_features(sequence=sequence, seed=seed, golden=golden)

    config = af3_model.Model.Config()
    config.num_recycles = max(recycling_steps - 1, 0)  # reference runs num_recycles + 1 trunk passes
    config.return_embeddings = True
    config.return_distogram = True
    config.global_config.flash_attention_implementation = flash_attention
    config.heads.diffusion.eval.num_samples = 1
    # The distogram/trunk tensors are independent of sampling; cut diffusion steps so the
    # capture stays tractable on CPU without touching the deterministic tensors we compare.
    config.heads.diffusion.eval.steps = 20
    hk_params = af3_params.get_model_haiku_params(model_dir=_reference_model_dir(model_dir, model_name))

    @hk.transform
    def _forward(data: Any) -> Any:
        return af3_model.Model(config)(data)

    result = _forward.apply(hk_params, jax.random.PRNGKey(seed), batch)
    result = jax.tree.map(np.asarray, result)
    if "single_embeddings" not in result or "pair_embeddings" not in result:
        raise Blocked("the reference forward did not return trunk embeddings; return_embeddings ignored")
    distogram = result["distogram"]
    if not hasattr(distogram, "get") or distogram.get("distogram") is None:
        raise Blocked("the reference forward did not return distogram logits; return_distogram ignored")

    def _f32(value: Any) -> Any:
        return np.asarray(value, dtype=np.float32)

    forward_tensors = {
        "single_embeddings": _f32(result["single_embeddings"]),
        "pair_embeddings": _f32(result["pair_embeddings"]),
        "distogram_logits": _f32(distogram["distogram"]),
        "contact_probs": _f32(distogram["contact_probs"]),
    }
    features = {
        key: np.asarray(value)
        for key, value in batch.items()
        if hasattr(value, "dtype") and getattr(value, "dtype").kind in "fiub"
    }
    metadata = {**_metadata(sequence=sequence, model_name=model_name, seed=seed), "feature_source": feature_source}
    payload = {
        "metadata": metadata,
        "provenance": provenance,
        "recycling_steps": recycling_steps,
        "forward_tensors": forward_tensors,
        "features": features,
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(payload, handle)
    shapes = {name: list(value.shape) for name, value in forward_tensors.items()}
    print(
        json.dumps(
            {"mode": "reference", "output": str(output), "feature_source": feature_source, "forward_tensors": shapes},
            indent=2,
            sort_keys=True,
        )
    )


def reference_ext_dump(
    *,
    sequence: str,
    model_dir: Path,
    model_name: str,
    seed: int,
    noise_seed: int,
    recycling_steps: int,
    flash_attention: str,
    golden: bool,
    output: Path,
) -> None:
    r"""Extend the DeepMind AlphaFold 3 golden with denoiser + confidence-head tensors.

    Runs the released DeepMind forward in its own JAX / Haiku environment on the SAME
    featurised example as :func:`reference_dump`, but instead of stopping at the trunk it
    additionally captures, from the *reference's own* modules under the ``diffuser`` parameter
    scope (so the torch side mirrors byte-identical inputs):

    1. DIFFUSION DENOISER one step: a FIXED initial-noise tensor (``jax.random.normal`` seeded by
       ``noise_seed``, scaled by ``noise_schedule(0)`` and masked by the dense atom mask) is fed at
       ``noise_level = noise_schedule(0)`` into ``DiffusionHead.__call__`` together with the trunk
       ``single`` / ``pair`` / ``target_feat`` embeddings and ``use_conditioning=True`` --- exactly
       one ``denoising_step`` evaluation (no sampler augmentation / churn). The exact
       ``positions_noisy`` tensor and ``noise_level`` scalar are serialised so the comparison is
       deterministic.
    2. CONFIDENCE HEAD logits: the reference's own one-step denoiser prediction (deterministic,
       finite, DeepMind missing-atom=0 masked) is fed into ``ConfidenceHead.__call__``; a
       ``hk.intercept_methods`` interceptor captures the raw pre-softmax logits of the
       ``plddt_logits`` / ``pae_logits`` / ``experimentally_resolved_logits`` /
       ``left_half_distance_logits`` linears (the head itself only returns post-softmax scalars).
       The full stochastic sampler is intentionally avoided: its JAX RNG is not reproducible on the
       torch side and at a low step count it diverges to NaN.
    3. The trunk ``single`` / ``pair`` (and ``target_feat``, newly needed by both heads) are dumped.

    Everything runs inside a :class:`Model` subclass named ``diffuser`` so the Haiku parameter
    scoping is byte-identical to the released checkpoint; the base golden's trunk pipeline is
    reproduced verbatim (recycle loop over the Evoformer) so the embeddings match ``reference``.
    """

    import pickle

    _activate_reference_env()
    try:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        import numpy as np
        from alphafold3.model import feat_batch
        from alphafold3.model import model as af3_model
        from alphafold3.model import params as af3_params
        from alphafold3.model.network import confidence_head, diffusion_head
        from alphafold3.model.network import evoformer as evoformer_network
    except Exception as error:  # noqa: BLE001 -- absent JAX env degrades to a clean Blocked
        raise Blocked(f"the AlphaFold 3 JAX environment is unavailable: {type(error).__name__}: {error}") from error

    # See reference_dump: tokamax forces a lazy absl reparse of sys.argv; pre-mark FLAGS parsed.
    from absl import flags as _absl_flags

    if not _absl_flags.FLAGS.is_parsed():
        _absl_flags.FLAGS([sys.argv[0]])

    provenance = _reference_provenance(af3_model)
    print(f"SOURCE alphafold3 module={provenance['module']} version={provenance['version']}", flush=True)

    batch, feature_source = _reference_features(sequence=sequence, seed=seed, golden=golden)

    config = af3_model.Model.Config()
    config.num_recycles = max(recycling_steps - 1, 0)  # reference runs num_recycles + 1 trunk passes
    config.return_embeddings = True
    config.return_distogram = True
    config.global_config.flash_attention_implementation = flash_attention
    config.heads.diffusion.eval.num_samples = 1
    # The diffusion sample only supplies confidence-head input coordinates; keep it cheap on CPU.
    config.heads.diffusion.eval.steps = 20

    logit_leaf_names = {
        "plddt_logits",
        "pae_logits",
        "experimentally_resolved_logits",
        "left_half_distance_logits",
    }

    class _ExtModel(af3_model.Model):
        """Model subclass that stops for a fixed-noise one-step denoise + confidence-logit capture."""

        def __call__(self, data: Any, key: Any = None) -> Any:  # type: ignore[override]
            if key is None:
                key = hk.next_rng_key()
            fb = feat_batch.Batch.from_data_dict(data)

            embedding_module = evoformer_network.Evoformer(self.config.evoformer, self.global_config)
            target_feat = af3_model.create_target_feat_embedding(
                batch=fb, config=embedding_module.config, global_config=self.global_config
            )

            def recycle_body(_, args):
                prev, rng = args
                rng, subkey = jax.random.split(rng)
                emb = embedding_module(batch=fb, prev=prev, target_feat=target_feat, key=subkey)
                emb["pair"] = emb["pair"].astype(jnp.float32)
                emb["single"] = emb["single"].astype(jnp.float32)
                return emb, rng

            num_res = fb.num_res
            embeddings = {
                "pair": jnp.zeros([num_res, num_res, self.config.evoformer.pair_channel], dtype=jnp.float32),
                "single": jnp.zeros([num_res, self.config.evoformer.seq_channel], dtype=jnp.float32),
                "target_feat": target_feat,
            }
            num_iter = self.config.num_recycles + 1
            embeddings, _ = hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))

            # (1) One-step denoiser on a FIXED initial-noise tensor at noise_schedule(0).
            atom_mask = fb.predicted_structure_info.atom_mask  # [num_res, max_atoms]
            noise_levels = diffusion_head.noise_schedule(
                jnp.linspace(0.0, 1.0, self.config.heads.diffusion.eval.steps + 1)
            )
            sigma0 = noise_levels[0]
            raw_noise = jax.random.normal(jax.random.PRNGKey(noise_seed), atom_mask.shape + (3,))
            positions_noisy = raw_noise * sigma0 * atom_mask[..., None]
            noise_level = jnp.asarray(sigma0, dtype=jnp.float32)
            denoised = self.diffusion_module(
                positions_noisy,
                noise_level,
                batch=fb,
                embeddings=embeddings,
                use_conditioning=True,
            )

            # (2) Confidence head on the reference's OWN one-step denoiser prediction.
            # The full stochastic sampler is deliberately not used: its RNG does not match the torch
            # EDM sampler (so its coordinates are not reproducible cross-framework), and at a low
            # step count it diverges to NaN. The one-step denoised coordinates are deterministic,
            # finite, masked to DeepMind's missing-atom=0 convention, and are serialised verbatim so
            # the torch confidence head is driven by byte-identical input.
            confidence_coords = denoised
            captured: dict[str, Any] = {}

            def interceptor(next_f: Any, args: Any, kwargs: Any, context: Any) -> Any:
                out = next_f(*args, **kwargs)
                if context.method_name == "__call__":
                    leaf = context.module.module_name.split("/")[-1]
                    if leaf in logit_leaf_names and leaf not in captured:
                        captured[leaf] = out
                return out

            with hk.intercept_methods(interceptor):
                confidence = confidence_head.ConfidenceHead(self.config.heads.confidence, self.global_config)(
                    dense_atom_positions=confidence_coords,
                    embeddings=embeddings,
                    seq_mask=fb.token_features.mask,
                    token_atoms_to_pseudo_beta=fb.pseudo_beta_info.token_atoms_to_pseudo_beta,
                    asym_id=fb.token_features.asym_id,
                )

            return {
                "single_embeddings": embeddings["single"],
                "pair_embeddings": embeddings["pair"],
                "target_feat": embeddings["target_feat"],
                "positions_noisy": positions_noisy,
                "noise_level": noise_level,
                "atom_mask": atom_mask,
                "denoised_coords": denoised,
                "confidence_coords": confidence_coords,
                "plddt_logits": captured["plddt_logits"],
                "pae_logits": captured["pae_logits"],
                "resolved_logits": captured["experimentally_resolved_logits"],
                "left_half_distance_logits": captured["left_half_distance_logits"],
                "confidence_predicted_lddt": confidence["predicted_lddt"],
                "confidence_full_pae": confidence["full_pae"],
                "confidence_full_pde": confidence["full_pde"],
                "confidence_resolved_prob": confidence["predicted_experimentally_resolved"],
            }

    @hk.transform
    def _forward(data: Any) -> Any:
        return _ExtModel(config)(data)

    hk_params = af3_params.get_model_haiku_params(model_dir=_reference_model_dir(model_dir, model_name))
    result = _forward.apply(hk_params, jax.random.PRNGKey(seed), batch)
    result = jax.tree.map(np.asarray, result)

    def _f32(value: Any) -> Any:
        return np.asarray(value, dtype=np.float32)

    # The "pde" logits DeepMind actually consumes are the symmetrised distance-error logits
    # (left_half + its (-2,-3) transpose); keep the raw half alongside for provenance.
    left_half = _f32(result["left_half_distance_logits"])
    pde_logits = left_half + np.swapaxes(left_half, -2, -3)

    trunk_tensors = {
        "single_embeddings": _f32(result["single_embeddings"]),
        "pair_embeddings": _f32(result["pair_embeddings"]),
        "target_feat": _f32(result["target_feat"]),
    }
    denoiser_tensors = {
        "positions_noisy": _f32(result["positions_noisy"]),
        "noise_level": _f32(result["noise_level"]),
        "atom_mask": np.asarray(result["atom_mask"]).astype(bool),
        "denoised_coords": _f32(result["denoised_coords"]),
    }
    confidence_tensors = {
        "confidence_coords": _f32(result["confidence_coords"]),
        "plddt_logits": _f32(result["plddt_logits"]),
        "pae_logits": _f32(result["pae_logits"]),
        "pde_logits": _f32(pde_logits),
        "left_half_distance_logits": left_half,
        "resolved_logits": _f32(result["resolved_logits"]),
        "predicted_lddt": _f32(result["confidence_predicted_lddt"]),
        "full_pae": _f32(result["confidence_full_pae"]),
        "full_pde": _f32(result["confidence_full_pde"]),
        "resolved_prob": _f32(result["confidence_resolved_prob"]),
    }
    noise_spec = {
        "noise_seed": int(noise_seed),
        "noise_sigma": float(np.asarray(result["noise_level"])),
        "noise_source": "jax.random.normal(jax.random.PRNGKey(noise_seed), atom_mask.shape+(3,)) "
        "* noise_schedule(0) * atom_mask[...,None]",
        "noise_level_source": "noise_schedule(0) == SIGMA_DATA * smax == 2560.0",
    }
    features = {
        key: np.asarray(value)
        for key, value in batch.items()
        if hasattr(value, "dtype") and getattr(value, "dtype").kind in "fiub"
    }
    metadata = {**_metadata(sequence=sequence, model_name=model_name, seed=seed), "feature_source": feature_source}
    payload = {
        "metadata": metadata,
        "provenance": provenance,
        "recycling_steps": recycling_steps,
        "noise_spec": noise_spec,
        "trunk_tensors": trunk_tensors,
        "denoiser_tensors": denoiser_tensors,
        "confidence_tensors": confidence_tensors,
        "features": features,
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(payload, handle)

    def _shapes(group: dict[str, Any]) -> dict[str, list[int]]:
        return {name: list(np.asarray(value).shape) for name, value in group.items()}

    print(
        json.dumps(
            {
                "mode": "reference-ext",
                "output": str(output),
                "feature_source": feature_source,
                "noise_spec": noise_spec,
                "trunk_tensors": _shapes(trunk_tensors),
                "denoiser_tensors": _shapes(denoiser_tensors),
                "confidence_tensors": _shapes(confidence_tensors),
            },
            indent=2,
            sort_keys=True,
        )
    )


def forward_compare(
    *,
    reference_output: Path,
    model_dir: Path,
    model_name: str,
    device_name: str,
    json_path: Path | None,
) -> None:
    r"""Compare the native forward against a reference trunk capture at FORWARD tolerance.

    The comparison feeds the reference's *own* ``pair`` embedding into the native distogram head
    (converted official weights), so the two frameworks meet on byte-identical input and the
    distogram-logit agreement is a genuine forward-vs-forward numeric check --- independent of any
    featurisation difference.  The native full trunk is not compared here (its input would come
    from the native processor, which does not yet reproduce the reference featurisation bitwise).
    """

    import pickle

    import numpy as np
    import torch

    from ofoldx.models.alphafold3.convert import convert_alphafold3_checkpoint

    with open(_local_file(reference_output, "reference capture"), "rb") as handle:
        reference = pickle.load(handle)
    forward_tensors = reference.get("forward_tensors")
    if not forward_tensors or "pair_embeddings" not in forward_tensors or "distogram_logits" not in forward_tensors:
        raise ValueError(
            f"reference capture lacks forward_tensors pair_embeddings/distogram_logits: {reference_output}"
        )

    device = _device(device_name)
    checkpoint = model_dir.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"local AlphaFold 3 checkpoint does not exist: {checkpoint}")
    model = convert_alphafold3_checkpoint(checkpoint, model_name=model_name, dtype=torch.float32)
    model = model.to(device).eval().set_backend("torch")

    pair = torch.from_numpy(np.asarray(forward_tensors["pair_embeddings"], dtype=np.float32)).to(device)
    reference_distogram = torch.from_numpy(np.asarray(forward_tensors["distogram_logits"], dtype=np.float32))
    with torch.inference_mode():
        native_distogram = model.distogram_head(pair.unsqueeze(0)).squeeze(0).float().cpu()

    comparisons = {
        "distogram_logits": _forward_metrics(
            reference_distogram, native_distogram, atol=FORWARD_ATOL, rtol=FORWARD_RTOL
        ),
    }
    # single/pair are captured for provenance; the native trunk that produces them needs a
    # byte-identical featurised input, which does not yet exist (see CURRENT_BLOCKERS[1]).
    provenance_only = {
        name: {"reference_shape": list(np.asarray(forward_tensors[name]).shape), "compared": False}
        for name in ("single_embeddings", "pair_embeddings")
        if name in forward_tensors
    }
    numerical_match = all(value["passed"] for value in comparisons.values())
    report = {
        "reference_repository": "https://github.com/google-deepmind/alphafold3",
        "reference_provenance": reference.get("provenance"),
        "metadata": reference.get("metadata"),
        "tolerance": {"atol": FORWARD_ATOL, "rtol": FORWARD_RTOL},
        "comparisons": comparisons,
        "reference_only_tensors": provenance_only,
        "numerical_match": numerical_match,
        "passed": numerical_match,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if json_path is not None:
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text, encoding="utf-8")
    raise SystemExit(int(not numerical_match))


def native_trunk_compare(
    *,
    reference_output: Path,
    model_dir: Path,
    model_name: str,
    device_name: str,
    recycling_steps: int,
    json_path: Path | None,
) -> None:
    r"""Drive OFoldX's *native* AlphaFold 3 trunk on the reference's own featurised example.

    This is the meaningful end-to-end trunk validation that ``forward`` deliberately does not do.
    ``forward`` feeds the reference's own ``pair`` embedding into the native distogram head, so it
    never exercises the native Pairformer / MSA / template trunk. Here we instead map DeepMind's
    already-featurised example (the ``features`` dict captured alongside the golden trunk tensors)
    onto the OFoldX :class:`AlphaFold3Input`, run the *native* recycled trunk on those exact
    features (converted official weights), and compare the native ``single`` / ``pair`` embeddings
    (and the distogram driven through the native trunk pair) against the golden reference embeddings
    with a RELATIVE metric (rel L2 + cosine). The native featuriser is bypassed entirely: every
    channel comes from DeepMind's captured features, so the two frameworks meet on byte-identical
    trunk input and any disagreement is the native trunk graph, not a featurisation gap.
    """

    import pickle

    import numpy as np
    import torch

    from ofoldx.models.alphafold3.convert import convert_alphafold3_checkpoint

    with open(_local_file(reference_output, "reference golden capture"), "rb") as handle:
        reference = pickle.load(handle)
    features = reference.get("features")
    golden = reference.get("forward_tensors")
    if not features:
        raise ValueError(f"reference capture lacks the DeepMind featurised example (features): {reference_output}")
    if not golden or "single_embeddings" not in golden or "pair_embeddings" not in golden:
        raise ValueError(f"reference capture lacks golden single/pair embeddings: {reference_output}")

    device = _device(device_name)
    batch, mapping = _alphafold3_input_from_reference(features, device=device)

    checkpoint = model_dir.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"local AlphaFold 3 checkpoint does not exist: {checkpoint}")
    model = convert_alphafold3_checkpoint(checkpoint, model_name=model_name, dtype=torch.float32)
    model = model.to(device).eval().set_backend("torch")

    with torch.inference_mode():
        _token_embedding, single, pair, _relative = model._run_trunk(batch, recycling_steps=recycling_steps)
        distogram = model.distogram_head(pair)
    native = {
        "single_embeddings": single[0].float().cpu(),
        "pair_embeddings": pair[0].float().cpu(),
        "distogram_logits": distogram[0].float().cpu(),
    }

    # A faithful native trunk on byte-identical input reproduces the golden trunk up to fp
    # reduction-order noise: near-unit cosine AND a small relative L2. The distogram is the native
    # distogram head driven by the *native* trunk pair (distinct from ``forward``, which drives it
    # with the golden pair); it is reported for completeness and tracks whatever pair passes.
    passing_cosine = 0.999
    passing_relative_l2 = 5e-2
    comparisons: dict[str, Any] = {}
    for name in ("single_embeddings", "pair_embeddings", "distogram_logits"):
        if name not in golden:
            continue
        reference_tensor = torch.from_numpy(np.asarray(golden[name], dtype=np.float32))
        metrics = _relative_metrics(reference_tensor, native[name])
        metrics["passed"] = bool(
            metrics.get("cosine", 0.0) >= passing_cosine
            and metrics.get("relative_l2", float("inf")) <= passing_relative_l2
        )
        comparisons[name] = metrics

    trunk_names = [name for name in ("single_embeddings", "pair_embeddings") if name in comparisons]
    validated = bool(trunk_names) and all(comparisons[name]["passed"] for name in trunk_names)
    report = {
        "reference_repository": "https://github.com/google-deepmind/alphafold3",
        "reference_provenance": reference.get("provenance"),
        "metadata": reference.get("metadata"),
        "recycling_steps": recycling_steps,
        "feature_mapping": mapping,
        "metric": "relative (rel L2 + cosine over flattened fp64 tensors)",
        "tolerance": {"cosine_min": passing_cosine, "relative_l2_max": passing_relative_l2},
        "comparisons": comparisons,
        "native_trunk_validated": validated,
        "passed": validated,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if json_path is not None:
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text, encoding="utf-8")
    raise SystemExit(int(not validated))


def ext_compare(
    *,
    reference_output: Path,
    model_dir: Path,
    model_name: str,
    device_name: str,
    recycling_steps: int,
    json_path: Path | None,
) -> None:
    r"""Compare OFoldX's denoiser + confidence heads against the extended golden at RELATIVE tol.

    Consumes the ``reference-ext`` capture (``trunk_tensors`` / ``denoiser_tensors`` /
    ``confidence_tensors`` + ``features``).  The native trunk is *bypassed*: the reference's own
    ``single`` / ``pair`` / ``target_feat`` embeddings are injected as the denoiser + confidence
    conditioning, so the two frameworks meet on byte-identical trunk representations AND byte-identical
    input coordinates (the serialised fixed-noise tensor for the denoiser; the serialised one-step
    denoised coordinates for the confidence head).  Any disagreement is therefore purely the native
    denoiser / confidence graph, not a trunk or featurisation gap.

    (1) DENOISER: the reference's fixed initial-noise tensor at ``noise_schedule(0)`` is flattened to
        the native per-atom layout and fed through ``coordinate_denoiser`` on the reference trunk
        conditioning; the one-step denoised coordinates are compared against the golden
        ``denoised_coords`` (rel L2 + cosine over the 954 valid atoms).
    (2) CONFIDENCE: the golden one-step denoised coordinates are fed through ``confidence_head`` on the
        reference trunk conditioning; the raw pLDDT / PAE / PDE / resolved logits are compared against
        the golden pre-softmax logits.

    Both use a RELATIVE metric with a tight threshold: near-unit cosine and small relative L2.
    """

    import pickle

    import numpy as np
    import torch

    from ofoldx.models.alphafold3.convert import convert_alphafold3_checkpoint
    from ofoldx.modules.embedding import generate_relative_position_features

    with open(_local_file(reference_output, "reference-ext golden capture"), "rb") as handle:
        reference = pickle.load(handle)
    features = reference.get("features")
    trunk = reference.get("trunk_tensors")
    denoiser_ref = reference.get("denoiser_tensors")
    confidence_ref = reference.get("confidence_tensors")
    if not features or not trunk or not denoiser_ref or not confidence_ref:
        raise ValueError(
            f"reference-ext capture lacks trunk/denoiser/confidence tensors + features: {reference_output}"
        )

    device = _device(device_name)
    batch, mapping = _alphafold3_input_from_reference(features, device=device)

    # Rebuild the two confidence-head atom index channels that the trunk mapping stubs out (the trunk
    # never reads them).  We reproduce the EXACT (token, slot) flat atom order that
    # ``_alphafold3_input_from_reference`` builds -- token-major over the valid ``ref_mask`` slots.
    ref_mask = np.asarray(features["ref_mask"]).astype(bool)  # [T, 24]
    token_count, slot_count = ref_mask.shape
    slot_of_atom: list[int] = []
    flat_index_of: dict[tuple[int, int], int] = {}
    for token in range(token_count):
        for slot in range(slot_count):
            if ref_mask[token, slot]:
                flat_index_of[(token, slot)] = len(slot_of_atom)
                slot_of_atom.append(slot)
    atom_count = len(slot_of_atom)

    # atom_token_index := the dense slot (== DeepMind's per-token atom offset that indexes the
    # per-offset pLDDT / resolved weight tables).
    atom_token_index = torch.tensor(slot_of_atom, dtype=torch.long, device=device)
    # token_to_rep_atom := global flat index of each token's pseudo-beta atom.  DeepMind's
    # ``token_atoms_to_pseudo_beta`` gathers into the flattened dense [T*24] array; slot = idx % 24.
    pseudo_beta_idx = np.asarray(features["token_atoms_to_pseudo_beta:gather_idxs"]).astype(np.int64)
    rep_flat = []
    for token in range(token_count):
        slot = int(pseudo_beta_idx[token]) % slot_count
        rep_flat.append(flat_index_of[(token, slot)])
    token_to_rep_atom = torch.tensor(rep_flat, dtype=torch.long, device=device)
    batch.atom_token_index = atom_token_index.unsqueeze(0)
    batch.token_to_rep_atom = token_to_rep_atom.unsqueeze(0)

    # Reference trunk conditioning (batch axis added); these REPLACE the native trunk.
    def _ref(name: str, group: dict[str, Any]) -> torch.Tensor:
        return torch.from_numpy(np.asarray(group[name], dtype=np.float32)).to(device)

    single = _ref("single_embeddings", trunk).unsqueeze(0)  # [1, T, 384]
    pair = _ref("pair_embeddings", trunk).unsqueeze(0)  # [1, T, T, 128]
    # DeepMind's ``target_feat`` is ordered [restype/profile/deletion tail (63), atom-encoder single (384)];
    # OFoldX's InputEmbedder emits the token embedding in the opposite [atom (384), tail (63)] order (and the
    # converted target_feat-consuming projections expect that native order).  Rotate the golden reference into
    # the native order so the injected token embedding matches what OFoldX production would feed.
    _target_feat_dm = _ref("target_feat", trunk)  # [T, 447] DeepMind [tail|atom]
    token_embedding = torch.cat([_target_feat_dm[..., 63:], _target_feat_dm[..., :63]], dim=-1).unsqueeze(0)

    # Flatten the dense [T, 24, 3] coordinate tensors to the native [A, 3] valid-atom layout.
    def _flatten_dense(dense: np.ndarray) -> torch.Tensor:
        flat = np.empty((atom_count, 3), dtype=np.float32)
        for (token, slot), j in flat_index_of.items():
            flat[j] = dense[token, slot]
        return torch.from_numpy(flat).to(device)

    positions_noisy = _flatten_dense(np.asarray(denoiser_ref["positions_noisy"], dtype=np.float32))
    denoised_ref = _flatten_dense(np.asarray(denoiser_ref["denoised_coords"], dtype=np.float32))
    confidence_coords = _flatten_dense(np.asarray(confidence_ref["confidence_coords"], dtype=np.float32))
    noise_level = float(np.asarray(denoiser_ref["noise_level"]))

    checkpoint = model_dir.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"local AlphaFold 3 checkpoint does not exist: {checkpoint}")
    model = convert_alphafold3_checkpoint(checkpoint, model_name=model_name, dtype=torch.float32)
    model = model.to(device).eval().set_backend("torch")

    relative_features = generate_relative_position_features(
        residue_index=batch.residue_index,
        chain_index=batch.chain_index,
        entity_index=batch.entity_index,
        symmetry_index=batch.symmetry_index,
        token_index=batch.token_index,
        dtype=single.dtype,
    )
    representations = (token_embedding, single, pair, relative_features)

    comparisons: dict[str, Any] = {}
    with torch.inference_mode():
        # (1) DENOISER one step on the reference trunk conditioning + fixed noise.
        context = model._denoise_context(batch, representations)
        noisy = positions_noisy.unsqueeze(0).unsqueeze(0)  # [1, 1, A, 3]
        nl = torch.full((1, 1), noise_level, dtype=single.dtype, device=device)
        denoised_native = model.coordinate_denoiser(context.to_denoising_input(noisy_coords=noisy, noise_level=nl))
        denoised_native = denoised_native.reshape(atom_count, 3).float().cpu()
        comparisons["denoised_coords"] = _relative_metrics(denoised_ref.float().cpu(), denoised_native)

        # (2) CONFIDENCE head on the reference trunk conditioning + golden one-step denoised coords.
        pred_coords = confidence_coords.unsqueeze(0).unsqueeze(0)  # [1, 1, A, 3]
        plddt_logits, pae_logits, pde_logits, resolved_logits = model.confidence_head(
            single,
            pair=pair,
            token_embedding=token_embedding,
            pred_coords=pred_coords,
            token_to_rep_atom=batch.token_to_rep_atom,
            atom_to_token=batch.atom_to_token,
            atom_token_index=batch.atom_token_index,
        )

    # pLDDT / resolved are per-atom in native flat layout; gather the golden dense logits to the same
    # (token, slot) order to compare atom-for-atom.
    def _gather_dense_logits(dense: np.ndarray) -> torch.Tensor:
        out = np.empty((atom_count, dense.shape[-1]), dtype=np.float32)
        for (token, slot), j in flat_index_of.items():
            out[j] = dense[token, slot]
        return torch.from_numpy(out)

    plddt_ref = _gather_dense_logits(np.asarray(confidence_ref["plddt_logits"], dtype=np.float32))
    resolved_ref = _gather_dense_logits(np.asarray(confidence_ref["resolved_logits"], dtype=np.float32))
    pae_ref = torch.from_numpy(np.asarray(confidence_ref["pae_logits"], dtype=np.float32))
    pde_ref = torch.from_numpy(np.asarray(confidence_ref["pde_logits"], dtype=np.float32))

    comparisons["plddt_logits"] = _relative_metrics(plddt_ref, plddt_logits.reshape(atom_count, -1).float().cpu())
    comparisons["resolved_logits"] = _relative_metrics(
        resolved_ref, resolved_logits.reshape(atom_count, -1).float().cpu()
    )
    comparisons["pae_logits"] = _relative_metrics(
        pae_ref, pae_logits.reshape(token_count, token_count, -1).float().cpu()
    )
    comparisons["pde_logits"] = _relative_metrics(
        pde_ref, pde_logits.reshape(token_count, token_count, -1).float().cpu()
    )

    # Tight relative gate: a faithful graph on byte-identical input reproduces the reference up to fp
    # reduction-order noise -- near-unit cosine AND a small relative L2.
    passing_cosine = 0.999
    passing_relative_l2 = 5e-3
    for name, metrics in comparisons.items():
        metrics["passed"] = bool(
            metrics.get("cosine", 0.0) >= passing_cosine
            and metrics.get("relative_l2", float("inf")) <= passing_relative_l2
        )
    validated = all(metrics["passed"] for metrics in comparisons.values())
    report = {
        "reference_repository": "https://github.com/google-deepmind/alphafold3",
        "reference_provenance": reference.get("provenance"),
        "metadata": reference.get("metadata"),
        "noise_spec": reference.get("noise_spec"),
        "recycling_steps": recycling_steps,
        "feature_mapping": mapping,
        "metric": "relative (rel L2 + cosine over flattened fp64 tensors)",
        "tolerance": {"cosine_min": passing_cosine, "relative_l2_max": passing_relative_l2},
        "comparisons": comparisons,
        "ext_validated": validated,
        "passed": validated,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if json_path is not None:
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text, encoding="utf-8")
    raise SystemExit(int(not validated))


def _relative_metrics(reference: Any, native: Any) -> dict[str, Any]:
    r"""Relative L2 and cosine between two tensors, plus shape / scale context, over flattened fp64."""

    import torch

    left = reference.detach().to(torch.float64).cpu()
    right = native.detach().to(torch.float64).cpu()
    if left.shape != right.shape:
        return {"reference_shape": list(left.shape), "native_shape": list(right.shape), "comparable": False}
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    reference_norm = float(flat_left.norm())
    native_norm = float(flat_right.norm())
    difference = flat_right - flat_left
    relative_l2 = float(difference.norm()) / reference_norm if reference_norm > 0 else float("inf")
    denominator = reference_norm * native_norm
    cosine = float(torch.dot(flat_left, flat_right)) / denominator if denominator > 0 else 0.0
    return {
        "shape": list(left.shape),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_absolute_error": float(difference.abs().max()) if difference.numel() else 0.0,
        "reference_norm": reference_norm,
        "native_norm": native_norm,
    }


def _alphafold3_input_from_reference(features: dict[str, Any], *, device: Any) -> tuple[Any, dict[str, Any]]:
    r"""Map DeepMind's captured featurised example onto an OFoldX :class:`AlphaFold3Input`.

    DeepMind stores atoms in a dense ``[N_token, 24-slot]`` layout; OFoldX consumes a flat
    ``[N_atom]`` layout. We flatten the dense atoms in ``(token, slot)`` order over ``ref_mask``
    (which keeps atoms grouped by token, exactly the contiguous layout the native atom encoder
    expects), then repack every native channel directly from DeepMind's arrays -- the native
    featuriser is never invoked. Index channels (residue / chain / entity / symmetry / token) are
    forwarded verbatim: the native relative-position embedder reads only their differences and
    equalities, both invariant to DeepMind's constant 1-based offset. Token/atom feature packings
    reproduce the native concatenations bit-for-bit (already channel-verified by ``processor``).
    """

    import math

    import numpy as np
    import torch
    import torch.nn.functional as F

    from ofoldx.models.alphafold3.model import AlphaFold3Input
    from ofoldx.models.alphafold3.processor import _template_backbone, _template_distogram
    from ofoldx.modules.embedding import TemplateFeatures

    def field(name: str) -> np.ndarray:
        return np.asarray(features[name])

    ref_mask = field("ref_mask").astype(bool)  # [T, 24]
    ref_pos = field("ref_pos").astype(np.float32)  # [T, 24, 3]
    ref_charge = field("ref_charge").astype(np.float32)  # [T, 24]
    ref_element = field("ref_element").astype(np.int64)  # [T, 24] atomic number Z
    ref_name = field("ref_atom_name_chars").astype(np.int64)  # [T, 24, 4] (ord(c) - 32)
    ref_space = field("ref_space_uid").astype(np.int64)  # [T, 24]
    token_count = int(ref_mask.shape[0])

    # Flatten the dense atoms in (token, slot) order over the valid mask -> the native flat layout.
    flat_token, flat_pos, flat_charge, flat_space, flat_element, flat_name = [], [], [], [], [], []
    for token in range(token_count):
        for slot in range(ref_mask.shape[1]):
            if ref_mask[token, slot]:
                flat_token.append(token)
                flat_pos.append(ref_pos[token, slot])
                flat_charge.append(ref_charge[token, slot])
                flat_space.append(ref_space[token, slot])
                flat_element.append(ref_element[token, slot])
                flat_name.append(ref_name[token, slot])
    atom_count = len(flat_token)

    atom_to_token = torch.tensor(flat_token, dtype=torch.long)
    reference_positions = torch.tensor(np.stack(flat_pos), dtype=torch.float32)
    reference_charges = torch.tensor(np.asarray(flat_charge), dtype=torch.float32)
    reference_space_id = torch.tensor(np.asarray(flat_space), dtype=torch.long)
    reference_mask = torch.ones(atom_count, dtype=torch.bool)
    element_index = torch.tensor(np.asarray(flat_element), dtype=torch.long).clamp_(0, 127)
    name_index = torch.tensor(np.stack(flat_name), dtype=torch.long).clamp_(0, 63)  # [A, 4]
    # atom_features = [reference_mask(1) | element one-hot(128) | atom-name one-hot(4 x 64 = 256)] = 385.
    atom_features = torch.cat(
        (
            reference_mask[:, None].float(),
            F.one_hot(element_index, num_classes=128).float(),
            F.one_hot(name_index, num_classes=64).reshape(atom_count, 256).float(),
        ),
        dim=-1,
    )

    # token_features = [zeros(384) | restype one-hot(31) | profile(31) | deletion_mean(1)] = 447.
    aatype = torch.tensor(field("aatype").astype(np.int64))
    profile = torch.tensor(field("profile").astype(np.float32))
    deletion_mean = torch.tensor(field("deletion_mean").astype(np.float32))
    token_features = torch.cat(
        (
            torch.zeros(token_count, 384),
            F.one_hot(aatype, num_classes=31).float(),
            profile,
            deletion_mean[:, None],
        ),
        dim=-1,
    )

    residue_index = torch.tensor(field("residue_index").astype(np.int64))
    chain_index = torch.tensor(field("asym_id").astype(np.int64))
    entity_index = torch.tensor(field("entity_id").astype(np.int64))
    symmetry_index = torch.tensor(field("sym_id").astype(np.int64))
    token_index = torch.tensor(field("token_index").astype(np.int64))
    token_mask = torch.tensor(field("seq_mask").astype(bool))

    # Token bonds: DeepMind carries an atomised-ligand bond list as a (row, 2-endpoint) token gather;
    # each fully masked row is one undirected token--token bond. Reconstruct the dense adjacency.
    token_bonds = torch.zeros(token_count, token_count)
    bond_edges = 0
    for gather_name in ("tokens_to_ligand_ligand_bonds", "tokens_to_polymer_ligand_bonds"):
        idxs = field(f"{gather_name}:gather_idxs")
        mask = field(f"{gather_name}:gather_mask").astype(bool)
        for row in range(idxs.shape[0]):
            if mask[row].all():
                first, second = int(idxs[row, 0]), int(idxs[row, 1])
                if 0 <= first < token_count and 0 <= second < token_count:
                    token_bonds[first, second] = 1.0
                    token_bonds[second, first] = 1.0
                    bond_edges += 1

    # MSA: one-hot(32) + has_deletion + deletion_value over the populated rows only (rest are padding).
    msa = field("msa").astype(np.int64)
    msa_mask_full = field("msa_mask").astype(bool)
    deletion_matrix = field("deletion_matrix").astype(np.float32)
    populated_rows = np.where(msa_mask_full.any(axis=1))[0]
    msa_rows = torch.tensor(msa[populated_rows], dtype=torch.long).clamp_(0, 31)
    deletions = torch.tensor(deletion_matrix[populated_rows], dtype=torch.float32)
    has_deletion = deletions.clamp(0.0, 1.0)
    deletion_value = torch.atan(deletions / 3.0) * (2.0 / math.pi)
    msa_features = torch.cat(
        (
            F.one_hot(msa_rows, num_classes=32).float(),
            has_deletion[..., None],
            deletion_value[..., None],
        ),
        dim=-1,
    )
    msa_mask = torch.tensor(msa_mask_full[populated_rows], dtype=torch.float32)

    # Templates: pre-pack the native TemplateFeatures from DeepMind's dense template arrays using the
    # native processor's own backbone/distogram helpers (identical frame + squared-distance binning),
    # reading N/CA/C/CB from dense slots 0/1/2/4 (the verified protein backbone layout).
    template_positions = field("template_atom_positions").astype(np.float32)  # [F, T, 24, 3]
    template_atom_mask = field("template_atom_mask").astype(np.float32)  # [F, T, 24]
    template_aatype = field("template_aatype").astype(np.int64)  # [F, T]
    num_frames = int(template_positions.shape[0])
    backbone_slots = [0, 1, 2, 4]  # N, CA, C, CB
    distograms, pseudo_beta_masks, unit_vectors, backbone_masks, residue_types = [], [], [], [], []
    for frame in range(num_frames):
        positions4 = torch.tensor(template_positions[frame][:, backbone_slots, :])  # [T, 4, 3]
        mask4 = torch.tensor(template_atom_mask[frame][:, backbone_slots])  # [T, 4]
        pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask = _template_backbone(positions4, mask4)
        pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        distograms.append(_template_distogram(pseudo_beta, pseudo_beta_mask_2d, 39))
        pseudo_beta_masks.append(pseudo_beta_mask_2d)
        unit_vectors.append(unit_vector * backbone_mask_2d[..., None])
        backbone_masks.append(backbone_mask_2d)
        residue_types.append(torch.tensor(template_aatype[frame]))
    template_features = TemplateFeatures(
        distogram=torch.stack(distograms).unsqueeze(0).to(device=device, dtype=torch.float32),
        pseudo_beta_mask=torch.stack(pseudo_beta_masks).unsqueeze(0).to(device=device, dtype=torch.float32),
        residue_type=torch.stack(residue_types).unsqueeze(0).to(device=device),
        unit_vector=torch.stack(unit_vectors).unsqueeze(0).to(device=device, dtype=torch.float32),
        backbone_frame_mask=torch.stack(backbone_masks).unsqueeze(0).to(device=device, dtype=torch.float32),
        template_mask=torch.ones(1, num_frames, device=device),
    )

    def floating(value: Any) -> Any:
        return value.to(device=device, dtype=torch.float32).unsqueeze(0)

    def integer(value: Any) -> Any:
        return value.to(device=device).unsqueeze(0)

    batch = AlphaFold3Input(
        token_mask=integer(token_mask),
        residue_index=integer(residue_index),
        chain_index=integer(chain_index),
        entity_index=integer(entity_index),
        symmetry_index=integer(symmetry_index),
        token_index=integer(token_index),
        atom_features=floating(atom_features),
        atom_to_token=integer(atom_to_token),
        atom_mask=integer(torch.ones(atom_count, dtype=torch.bool)),
        reference_positions=floating(reference_positions),
        reference_space_id=integer(reference_space_id),
        # atom_token_index / token_to_rep_atom drive only the confidence head; the trunk never reads
        # them, so trunk-irrelevant placeholders keep the mapping honest about what is exercised.
        atom_token_index=integer(torch.zeros(atom_count, dtype=torch.long)),
        token_to_rep_atom=integer(torch.zeros(token_count, dtype=torch.long)),
        token_features=floating(token_features),
        reference_mask=integer(reference_mask),
        reference_charges=floating(reference_charges[:, None]),
        token_bonds=floating(token_bonds),
        msa_features=floating(msa_features),
        msa_mask=floating(msa_mask),
        template_features=template_features,
    )
    mapping = {
        "tokens": token_count,
        "atoms": atom_count,
        "msa_rows_used": int(len(populated_rows)),
        "template_frames": num_frames,
        "token_bond_edges": bond_edges,
        "source_fields_bypassing_native_featuriser": sorted(features),
    }
    return batch, mapping


def processor_reference_dump(*, sequence: str, seed: int, output: Path) -> None:
    r"""Featurise a single protein query with DeepMind and dump its per-token / per-atom channels.

    This is the independent ground truth for the ``processor`` check: the native
    ``AlphaFold3Processor`` mirrors the Protenix processor, so it can be Protenix-faithful yet
    AlphaFold3-wrong; only DeepMind's own ``featurise_input`` output can discriminate.
    """

    import pickle

    import numpy as np

    _activate_reference_env()
    batch, source = _reference_features(sequence=sequence, seed=seed, golden=False)
    features = {
        key: np.asarray(value)
        for key, value in batch.items()
        if hasattr(value, "dtype") and np.asarray(value).dtype.kind in "fiub"
    }
    payload = {"metadata": {"sequence": sequence, "seed": seed, "feature_source": source}, "features": features}
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(payload, handle)
    print(
        json.dumps(
            {"mode": "processor-reference", "output": str(output), "channels": sorted(features)},
            indent=2,
            sort_keys=True,
        )
    )


def processor_compare(*, reference_output: Path, sequence: str, device_name: str, json_path: Path | None) -> None:
    r"""Compare the native AlphaFold3Processor channels against a DeepMind featurisation capture.

    Runs the native processor on the same single-protein query-only system, aligns atoms by
    (token, atom-name), and checks every channel it can map: integer / one-hot channels exactly,
    float channels at ``PROCESSOR_ATOL``.  Index channels are constant-offset-aware.  ``ref_pos`` is
    reported as an unverified residual because DeepMind augments the reference conformer per RNG
    seed, so there is no deterministic target on the reference side.
    """

    import pickle

    import numpy as np

    from ofoldx.data import MSA, MolecularSystem, PolymerSequence
    from ofoldx.models.alphafold3 import AlphaFold3Processor

    with open(_local_file(reference_output, "processor capture"), "rb") as handle:
        reference = pickle.load(handle)
    deepmind = reference["features"]

    device = _device(device_name)
    width = len(sequence)
    protein = PolymerSequence.from_one_letter(sequence, polymer_type="protein")
    system = MolecularSystem(entities={"A": protein}, instances={"A": "A"})
    msa = MSA(
        entity_id="A",
        unpaired_sequences=(sequence,),
        unpaired_deletion_counts=np.zeros((1, width), dtype=np.uint32),
        paired_sequences=(sequence,),
        paired_deletion_counts=np.zeros((1, width), dtype=np.uint32),
    )
    batch = AlphaFold3Processor(device=device)(system, msa=msa)

    comparisons, unverified = _compare_processor_channels(deepmind, batch)
    strict = {name: value for name, value in comparisons.items() if value.get("strict", True)}
    processor_af3_faithful = all(value["passed"] for value in strict.values())
    report = {
        "reference_repository": "https://github.com/google-deepmind/alphafold3",
        "metadata": reference.get("metadata"),
        "tolerance": {"float_atol": PROCESSOR_ATOL, "integer": "exact"},
        "channels": comparisons,
        "unverified_channels": unverified,
        "processor_af3_faithful": processor_af3_faithful,
        "passed": processor_af3_faithful,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if json_path is not None:
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text, encoding="utf-8")
    raise SystemExit(int(not processor_af3_faithful))


def _decode_atom_name(chars: Any) -> str:
    r"""Decode DeepMind ``ref_atom_name_chars`` (``ord(c) - 32`` per char, 0-padded) to a string."""

    return "".join(chr(int(code) + 32) for code in chars if int(code) > 0)


def _align_processor_atoms(deepmind: dict[str, Any], batch: Any) -> tuple[list[tuple[int, int, int]], list, list]:
    r"""Match DeepMind dense (token, slot) atoms to native flat atoms by (token index, atom name)."""

    import numpy as np

    ref_mask = np.asarray(deepmind["ref_mask"])
    ref_name_chars = np.asarray(deepmind["ref_atom_name_chars"])
    dm_atoms: dict[tuple[int, str], int] = {}
    for token in range(ref_mask.shape[0]):
        for slot in range(ref_mask.shape[1]):
            if ref_mask[token, slot]:
                dm_atoms[(token, _decode_atom_name(ref_name_chars[token, slot]))] = slot
    atom_to_token = batch.atom_to_token[0].detach().cpu().numpy()
    native_atoms: dict[tuple[int, str], int] = {
        (int(atom_to_token[index]), site.atom_id): index for index, site in enumerate(batch.atom_sites)
    }
    matched = [
        (token, dm_atoms[(token, name)], native_atoms[(token, name)])
        for (token, name) in dm_atoms
        if (token, name) in native_atoms
    ]
    dm_only = [key for key in dm_atoms if key not in native_atoms]
    native_only = [key for key in native_atoms if key not in dm_atoms]
    return matched, dm_only, native_only


def _compare_processor_channels(deepmind: dict[str, Any], batch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Return per-channel {tolerance, max_deviation, verdict, passed} plus unverified residuals."""

    import numpy as np

    token_features = batch.token_features[0].detach().cpu().numpy()
    channels: dict[str, Any] = {}

    def integer_channel(name: str, native: Any, reference: Any, *, note: str = "") -> None:
        native_array = np.asarray(native).astype(np.int64)
        reference_array = np.asarray(reference).astype(np.int64)
        deviation = int(np.abs(native_array - reference_array).max()) if native_array.size else 0
        channels[name] = {
            "tolerance": "exact",
            "max_deviation": deviation,
            "passed": deviation == 0,
            "verdict": "MATCH" if deviation == 0 else "MISMATCH",
            **({"note": note} if note else {}),
        }

    def float_channel(name: str, native: Any, reference: Any, *, note: str = "", strict: bool = True) -> None:
        native_array = np.asarray(native, dtype=np.float64)
        reference_array = np.asarray(reference, dtype=np.float64)
        deviation = float(np.abs(native_array - reference_array).max()) if native_array.size else 0.0
        channels[name] = {
            "tolerance": PROCESSOR_ATOL,
            "max_deviation": deviation,
            "passed": deviation <= PROCESSOR_ATOL,
            "verdict": "MATCH" if deviation <= PROCESSOR_ATOL else "MISMATCH",
            "strict": strict,
            **({"note": note} if note else {}),
        }

    # Token-level target features: restype one-hot, MSA profile, mean deletion.
    integer_channel("restype", token_features[:, 384:415].argmax(-1), deepmind["aatype"])
    float_channel("restype_onehot", token_features[:, 384:415], np.eye(31)[np.asarray(deepmind["aatype"])])
    float_channel("profile", token_features[:, 415:446], deepmind["profile"])
    float_channel(
        "deletion_mean",
        token_features[:, 446],
        deepmind["deletion_mean"],
        note="all ~0 for query-only (low-discriminating)",
    )

    # MSA stack: 32-class one-hot plus the two deletion channels.
    if batch.msa_features is not None:
        msa_features = batch.msa_features.detach().cpu().numpy()
        while msa_features.ndim > 2:
            msa_features = msa_features[0]
        integer_channel("msa_onehot", msa_features[:, :32].argmax(-1), np.asarray(deepmind["msa"])[0])
        deletion = np.asarray(deepmind["deletion_matrix"])[0].astype(np.float64)
        float_channel(
            "msa_has_deletion", msa_features[:, 32], np.clip(deletion, 0.0, 1.0), note="all ~0 (low-discriminating)"
        )
        float_channel(
            "msa_deletion_value",
            msa_features[:, 33],
            np.arctan(deletion / 3.0) * (2.0 / np.pi),
            note="all ~0 (low-discriminating)",
        )

    # Index channels: constant DeepMind(1-based) vs native(0-based) offset is difference-invariant.
    for name, native_key, reference_key in (
        ("residue_index", "residue_index", "residue_index"),
        ("chain_index", "chain_index", "asym_id"),
        ("entity_index", "entity_index", "entity_id"),
        ("symmetry_index", "symmetry_index", "sym_id"),
        ("token_index", "token_index", "token_index"),
    ):
        native = getattr(batch, native_key)[0].detach().cpu().numpy().astype(np.int64)
        reference = np.asarray(deepmind[reference_key]).astype(np.int64)
        offset = reference - native
        constant = bool(np.all(offset == offset.flat[0])) if offset.size else True
        channels[name] = {
            "tolerance": "exact-up-to-constant-offset",
            "constant_offset": int(offset.flat[0]) if constant and offset.size else None,
            "passed": constant,
            "verdict": "MATCH (difference-invariant offset)" if constant else "MISMATCH",
        }

    # Atom-level features aligned by (token, atom name).
    matched, dm_only, native_only = _align_processor_atoms(deepmind, batch)
    channels["atom_set"] = {
        "tolerance": "exact",
        "matched": len(matched),
        "dm_only": len(dm_only),
        "native_only": len(native_only),
        "passed": not dm_only and not native_only,
        "verdict": "MATCH" if not dm_only and not native_only else "PARTIAL",
    }
    atom_features = batch.atom_features[0].detach().cpu().numpy()
    reference_charges = batch.reference_charges[0].detach().cpu().numpy().reshape(-1)
    native_element = atom_features[:, 1:129].argmax(-1)
    native_name = atom_features[:, 129:385].reshape(len(atom_features), 4, 64).argmax(-1)
    ref_element = np.asarray(deepmind["ref_element"])
    ref_charge = np.asarray(deepmind["ref_charge"])
    ref_name_chars = np.asarray(deepmind["ref_atom_name_chars"])
    element_native = np.array([native_element[i] for (_t, _s, i) in matched])
    element_reference = np.array([ref_element[t, s] for (t, s, _i) in matched])
    integer_channel(
        "atom_element",
        element_native,
        element_reference,
        note="both one-hot the atomic number Z into 128 classes (index Z); the native atom-feature "
        "projection uses DeepMind's embed_ref_element weight verbatim, so the index must be Z, not Z-1",
    )
    integer_channel(
        "atom_name",
        np.array([native_name[i] for (_t, _s, i) in matched]),
        np.array([ref_name_chars[t, s] for (t, s, _i) in matched]),
    )
    float_channel(
        "ref_charge",
        np.array([reference_charges[i] for (_t, _s, i) in matched]),
        np.array([ref_charge[t, s] for (t, s, _i) in matched]),
        note="all ~0 for this system (low-discriminating)",
    )

    unverified = {
        "ref_pos": (
            "DeepMind applies stochastic per-RNG-seed reference-conformer augmentation (pairwise "
            "distances vary ~1 Angstrom across seeds), so there is no deterministic target; the native "
            "fixed conformer differs by ~0.3 Angstrom RMSD, within the augmentation envelope."
        ),
    }
    return channels, unverified


def compare_captures(
    *,
    reference_output: Path,
    native_output: Path,
    json_path: Path | None,
    atol: float,
    rtol: float,
) -> None:
    """Compare a reference capture with a native capture at a fixed FP32 tolerance."""

    import torch

    reference = torch.load(_local_file(reference_output, "reference capture"), map_location="cpu", weights_only=True)
    native = torch.load(_local_file(native_output, "native capture"), map_location="cpu", weights_only=True)
    if reference["metadata"] != native["metadata"]:
        raise ValueError(f"capture inputs differ: {reference['metadata']!r} != {native['metadata']!r}")

    comparisons = _compare_outputs(reference["outputs"], native["outputs"], atol=atol, rtol=rtol)
    numerical_match = all(value["passed"] for value in comparisons.values())
    report = {
        "reference_repository": "https://github.com/google-deepmind/alphafold3",
        "reference_revision": reference.get("reference_revision"),
        "reference_dirty": reference.get("reference_dirty"),
        "metadata": reference["metadata"],
        "tolerance": {"atol": atol, "rtol": rtol},
        "comparisons": comparisons,
        "numerical_match": numerical_match,
        "blockers": CURRENT_BLOCKERS,
        "passed": numerical_match and not CURRENT_BLOCKERS,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text, encoding="utf-8")
    raise SystemExit(2 if CURRENT_BLOCKERS else int(not numerical_match))


def _tiny_config() -> Any:
    from ofoldx.models.alphafold3 import AlphaFold3Config

    config = AlphaFold3Config()
    config.single_size = 16
    config.pair_size = 8
    config.recycling_steps = 1
    config.trunk.num_pairformer_blocks = 0
    config.trunk.pairformer_dropout = 0.0
    config.confidence.num_blocks = 0
    config.confidence.num_pair_attention_heads = 2
    config.confidence.triangle_multiplication_hidden_size = 8
    config.confidence.triangle_attention_hidden_size = 4
    config.confidence.num_triangle_attention_heads = 2
    config.denoiser.num_token_encoder_blocks = 0
    config.denoiser.num_token_encoder_attention_heads = 2
    config.denoiser.atom_hidden_size = 8
    config.denoiser.atom_pair_hidden_size = 4
    config.denoiser.token_hidden_size = 16
    config.denoiser.num_atom_encoder_blocks = 1
    config.denoiser.num_atom_encoder_attention_heads = 2
    config.denoiser.num_atom_decoder_blocks = 1
    config.denoiser.num_atom_decoder_attention_heads = 2
    config.denoiser.num_query_atoms = 4
    config.denoiser.num_key_atoms = 8
    config.template.num_blocks = 0
    return config


def _check_fold_result(result: Any, *, num_samples: int) -> None:
    from ofoldx.models.alphafold3.model import AlphaFold3FoldResult
    from ofoldx.results import Metric, PairMetric, Score

    if type(result) is not AlphaFold3FoldResult:
        raise AssertionError(f"public fold returned {type(result).__name__}, not AlphaFold3FoldResult")
    predictions = tuple(prediction for group in result.seed_groups for prediction in group.predictions)
    if len(predictions) != num_samples:
        raise AssertionError(f"public fold returned {len(predictions)} predictions, expected {num_samples}")
    for prediction in predictions:
        required_scores = {"plddt", "ptm", "iptm", "ranking_score"}
        if not required_scores <= prediction.scores.keys():
            raise AssertionError(f"AlphaFold3 scores incomplete: {sorted(prediction.scores.keys())}")
        if any(type(prediction.scores[name]) is not Score for name in required_scores):
            raise AssertionError("AlphaFold3 scores must be Score values")
        if {"plddt", "resolved_probability"} - prediction.metrics.keys():
            raise AssertionError(f"AlphaFold3 site metrics incomplete: {sorted(prediction.metrics.keys())}")
        if {"pae", "pde", "contact_probs"} - prediction.pair_metrics.keys():
            raise AssertionError(f"AlphaFold3 pair metrics incomplete: {sorted(prediction.pair_metrics.keys())}")
        if type(prediction.metrics["plddt"]) is not Metric or type(prediction.pair_metrics["pae"]) is not PairMetric:
            raise AssertionError("AlphaFold3 FoldPrediction lacks public pLDDT or PAE")
    if result.best("ranking_score") not in predictions:
        raise AssertionError("FoldResult.best did not rank AlphaFold3 predictions")


def _golden_inputs(batch: Any) -> dict[str, Any]:
    from dataclasses import fields

    import torch

    golden: dict[str, Any] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            golden[field.name] = value.detach().cpu()
    return golden


def _compare_outputs(
    reference: dict[str, Any],
    native: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, Any]]:
    import torch

    comparisons: dict[str, dict[str, Any]] = {}
    shared = sorted(set(reference) & set(native))
    if set(reference) != set(native):
        comparisons["output_names"] = {
            "reference": sorted(reference),
            "native": sorted(native),
            "passed": False,
        }
    for name in shared:
        left = reference[name].detach().float().cpu()
        right = native[name].detach().float().cpu()
        if left.shape != right.shape:
            comparisons[name] = {
                "reference_shape": list(left.shape),
                "native_shape": list(right.shape),
                "passed": False,
            }
            continue
        difference = (left - right).abs()
        relative = difference / left.abs().clamp_min(1e-9)
        comparisons[name] = {
            "shape": list(left.shape),
            "max_absolute_error": float(difference.max()) if difference.numel() else 0.0,
            "max_relative_error": float(relative.max()) if relative.numel() else 0.0,
            "passed": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        }
    return comparisons


def _metadata(*, sequence: str, model_name: str, seed: int) -> dict[str, Any]:
    return {"sequence": sequence, "model_name": model_name, "msa": "query-only", "seed": seed}


def _activate_reference_env() -> None:
    r"""Make the installed DeepMind ``alphafold3`` package importable.

    Two shadows have to be cleared. First, running this file as ``scripts/parity/alphafold3.py``
    puts its own directory on ``sys.path[0]``; because that directory holds *this* ``alphafold3.py``
    module, ``import alphafold3`` would otherwise resolve to the parity script rather than the
    installed package. Second, a scikit-build *editable* install registers a ``sys.meta_path``
    finder that maps every ``alphafold3`` submodule onto its original build tree; when that tree is
    another user's permission-restricted checkout the finder shadows a fully built, group-readable
    copy that also lives in ``site-packages``. Dropping the script directory and the finder lets the
    ordinary path finder resolve the readable installed package (compiled ``cpp`` extension included).
    Both steps are no-ops when the shadow is absent.
    """

    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if entry and Path(entry).resolve() != script_dir]
    survivors = [finder for finder in sys.meta_path if type(finder).__module__ != "_alphafold3_editable"]
    if len(survivors) != len(sys.meta_path):
        sys.meta_path[:] = survivors


def _reference_provenance(af3_model: Any) -> dict[str, str]:
    module = getattr(af3_model, "__file__", "unknown")
    version = "unknown"
    try:
        import alphafold3

        version = str(getattr(alphafold3, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        pass
    return {"module": str(module), "version": version}


def _reference_features(*, sequence: str, seed: int, golden: bool) -> tuple[Any, str]:
    r"""Return a featurised reference batch and a label describing where it came from.

    The reference forward expects the featurised dict cleaned exactly as ``ModelRunner.run_inference``
    cleans it: ``remove_invalidly_typed_feats`` drops the string / object features (e.g. the
    ``token_atoms_layout`` object) that would otherwise desynchronise the atom-cross-attention gather
    layout, and the arrays are materialised as a JAX pytree.
    """

    import jax

    try:
        from alphafold3.model.components import utils
    except Exception as error:  # noqa: BLE001
        raise Blocked(f"the AlphaFold 3 data pipeline is unavailable: {type(error).__name__}: {error}") from error

    if golden:
        import pickle

        from alphafold3.common import resources

        path = resources.ROOT / "test_data" / "featurised_example.pkl"
        raw = pickle.loads(path.read_bytes())[0]
        source = f"golden:{path.name}"
    else:
        try:
            from alphafold3.common import folding_input
            from alphafold3.constants import chemical_components
            from alphafold3.data import featurisation
        except Exception as error:  # noqa: BLE001
            raise Blocked(f"the AlphaFold 3 data pipeline is unavailable: {type(error).__name__}: {error}") from error

        fold_input = folding_input.Input(
            name="ofoldx_parity",
            chains=[
                folding_input.ProteinChain(
                    id="A",
                    sequence=sequence,
                    ptms=[],
                    unpaired_msa=f">query\n{sequence}\n",
                    paired_msa="",
                    templates=[],
                )
            ],
            rng_seeds=[seed],
        )
        try:
            raw = featurisation.featurise_input(
                fold_input=fold_input, ccd=chemical_components.cached_ccd(), buckets=None
            )[0]
        except Exception as error:  # noqa: BLE001 -- the live pipeline depends on external CCD/MSA state
            raise Blocked(
                "reference live featurisation failed "
                f"({type(error).__name__}: {error}); rerun with --golden to forward the shipped example"
            ) from error
        source = "live:query-only"

    batch = jax.tree_util.tree_map(jax.numpy.asarray, utils.remove_invalidly_typed_feats(raw))
    return batch, source


def _reference_model_dir(model_dir: Path, model_name: str) -> Path:
    r"""Return a directory the reference JAX loader can scan for exactly one model.

    ``get_model_haiku_params`` scans a *directory*; the shared weights root holds unrelated
    checkpoints, so when ``model_dir`` points straight at the ``.bin.zst`` file we isolate it in a
    scratch directory via a single symlink to avoid the loader's "multiple models matched" guard.
    """

    import tempfile

    model_dir = model_dir.expanduser().resolve()
    if model_dir.is_dir():
        return model_dir
    if not model_dir.is_file():
        raise FileNotFoundError(f"local AlphaFold 3 checkpoint does not exist: {model_dir}")
    isolated = Path(tempfile.mkdtemp(prefix="af3_model_")) / f"{model_name}.bin.zst"
    isolated.symlink_to(model_dir)
    return isolated.parent


def _forward_metrics(reference: Any, native: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    left = reference.detach().float().cpu()
    right = native.detach().float().cpu()
    if left.shape != right.shape:
        return {"reference_shape": list(left.shape), "native_shape": list(right.shape), "passed": False}
    difference = (left - right).abs()
    relative = difference / left.abs().clamp_min(1e-9)
    return {
        "shape": list(left.shape),
        "max_absolute_error": float(difference.max()) if difference.numel() else 0.0,
        "max_relative_error": float(relative.max()) if relative.numel() else 0.0,
        "passed": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
    }


def _local_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"local {label} does not exist: {path}")
    return path


def _local_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"local {label} does not exist: {path}")
    return path


def _local_path(path: Path, label: str) -> Path:
    r"""Resolve a checkpoint given as either the ``.bin.zst`` file or its directory."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"local {label} does not exist: {path}")
    return path


def _device(name: str) -> Any:
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


if __name__ == "__main__":
    # Keep every diagnostic on stdout so a batch scheduler captures one self-contained stream.
    sys.stderr = sys.stdout
    try:
        raise SystemExit(main())
    except Blocked as error:
        print(f"BLOCKED alphafold3: {error}", flush=True)
        raise SystemExit(2) from None
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"FAIL alphafold3: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1) from None
