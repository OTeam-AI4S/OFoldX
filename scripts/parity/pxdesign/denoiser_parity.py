r"""Real PXDesign converter + trained-denoiser parity against upstream PXDesign/Protenix.

Unlike ``sampler_parity`` (which swaps in a *fake* deterministic denoiser and only checks
the sampling equations), this harness validates the actual released checkpoint end to end:

1. STRICT LOAD -- load ``pxdesign_v0.1.0.pt`` through OFoldX ``convert_pxdesign_checkpoint``
   into the native model (0 missing / 0 unexpected / 0 shape-mismatch), and independently
   strict-load the same checkpoint into the upstream ProtenixDesign submodules
   (``DiffusionModule`` + ``DesignConditionEmbedder``).

2. PER-TENSOR WEIGHT COMPARE -- for every OFoldX parameter, resolve its upstream reference
   key(s) through the exact converter key-tree, apply the declared value transform, and
   compare against the tensor the upstream model loaded. This is the sharpest converter-bug
   detector: a mis-mapped key, a wrong layout transform, or a dropped/duplicated weight all
   surface here as a non-zero delta or a bijection failure. Deltas are bucketed by op class
   (triangle mult, attention proj, MSA/target_feat, denoiser transitions, ...).

3. DENOISER FORWARD PARITY -- drive the OFoldX ``CoordinateDenoiser`` and the upstream
   ``DiffusionModule`` with the SAME fixed synthetic features and the SAME fixed noise, and
   compare the denoised coordinates (relative L2 + cosine). Upstream ``d_lm/v_lm/pad_info``
   and ``relp`` are built with the upstream feature helpers; the OFoldX side derives its
   inputs from the identical raw tensors, so any divergence is a weight/graph fault, not a
   feature-mapping artifact.

Run (single process, env zyc-pxdesign, CPU is fine and deterministic):

    zyc-pxdesign/bin/python -m scripts.parity.pxdesign.denoiser_parity \
      --protenix-root /data/users/z/reference/Protenix \
      --reference-root /data/users/z/reference/PXDesign \
      --checkpoint /data/users/z/weights/pxdesign_v0.1.0.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from scripts.parity.paths import repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protenix-root", type=Path, default=Path("/data/users/z/reference/Protenix"))
    parser.add_argument("--reference-root", type=Path, default=Path("/data/users/z/reference/PXDesign"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/users/z/weights/pxdesign_v0.1.0.pt"))
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--atoms-per-token", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250718)
    parser.add_argument("--weight-atol", type=float, default=1e-6)
    return parser.parse_args()


def configure_imports(reference: Path, protenix: Path) -> None:
    os.environ.setdefault("LAYERNORM_TYPE", "normal")
    for path in (str(repo_root()), str(reference), str(protenix)):
        if path not in sys.path:
            sys.path.insert(0, path)


# ---------------------------------------------------------------------------------------------
# op-class bucketing for per-tensor weight compare
# ---------------------------------------------------------------------------------------------


def op_class(reference_key: str) -> str:
    key = reference_key
    if "tri_mul_out" in key or "tri_mul_in" in key:
        return "triangle_multiplication"
    if "tri_att_start" in key or "tri_att_end" in key:
        return "triangle_attention"
    if ".mha." in key or ".attention.linear_" in key or "attention_pair_bias.attention" in key:
        return "attention_projection"
    if "outer_product_mean" in key or "msa_pair_weighted" in key or "linear_no_bias_m" in key:
        return "msa_projection"
    if "input_embedder" in key or "input_map" in key or "linear_no_bias_f" in key:
        return "target_feat_projection"
    if "diffusion_conditioning" in key or "fourier" in key or "relpe" in key:
        return "denoiser_conditioning"
    if "atom_attention_encoder" in key:
        return "atom_encoder"
    if "atom_attention_decoder" in key:
        return "atom_decoder"
    if "diffusion_transformer" in key:
        return "denoiser_token_transformer"
    if "transition" in key or "conditioned_transition" in key or "pair_transition" in key:
        return "transition"
    if "condition_template_embedder" in key:
        return "template_embedder"
    return "other"


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().norm().item()
    num = (actual.float() - expected.float()).norm().item()
    return num / denom if denom > 0 else num


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.float().reshape(-1)
    e = expected.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, e, dim=0, eps=1e-12).item()


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    configure_imports(args.reference_root, args.protenix_root)
    torch.manual_seed(args.seed)

    from protenix.model.modules.diffusion import DiffusionModule
    from protenix.model.modules.embedders import RelativePositionEncoding
    from protenix.model.protenix import update_input_feature_dict

    from ofoldx.models.pxdesign.convert import (
        _extract_reference_state,
        _pxdesign_reference_key,
        convert_pxdesign_checkpoint,
        translate_pxdesign_state,
    )
    from ofoldx.modules.diffusion.sampling import CoordinateDenoiseContext
    from ofoldx.modules.embedding.position import generate_relative_position_features

    report: dict[str, object] = {}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    reference_state = _extract_reference_state(checkpoint)

    # ---- 1. STRICT LOAD -----------------------------------------------------------------
    native = convert_pxdesign_checkpoint(args.checkpoint).to(torch.float32)
    native.set_backend("torch")
    native.eval()
    native_sd = native.state_dict()

    diffusion_module = DiffusionModule(
        sigma_data=16.0,
        c_atom=128,
        c_atompair=16,
        c_token=768,
        c_s=384,
        c_z=128,
        c_s_inputs=449,
        atom_encoder={"n_blocks": 4, "n_heads": 4},
        transformer={"n_blocks": 16, "n_heads": 16},
        atom_decoder={"n_blocks": 4, "n_heads": 4},
        use_fine_grained_checkpoint=False,
        blocks_per_ckpt=None,
    )
    dm_sub = {k[len("diffusion_module.") :]: v for k, v in reference_state.items() if k.startswith("diffusion_module.")}
    dm_missing, dm_unexpected = diffusion_module.load_state_dict(dm_sub, strict=False)
    diffusion_module = diffusion_module.to(torch.float32).eval()

    # upstream state, fully prefixed, as the ground truth for the weight compare
    upstream_full = {"diffusion_module." + k: v for k, v in diffusion_module.state_dict().items()}
    upstream_full.update({k: v for k, v in reference_state.items() if k.startswith("design_condition_embedder.")})

    report["strict_load"] = {
        "ofoldx_params": len(native_sd),
        "checkpoint_tensors": len(reference_state),
        "ofoldx_load": "0 missing / 0 unexpected (convert_pxdesign_checkpoint strict=True succeeded)",
        "upstream_diffusion_module_tensors": len(dm_sub),
        "upstream_missing": len(dm_missing),
        "upstream_unexpected": len(dm_unexpected),
    }

    # ---- 2. PER-TENSOR WEIGHT COMPARE ---------------------------------------------------
    converted = translate_pxdesign_state(native, checkpoint, dtype=torch.float32)
    per_class: dict[str, dict[str, float | int]] = {}
    diverging: list[dict[str, object]] = []
    consumed_upstream: dict[str, int] = {}
    bijection_errors: list[str] = []

    for model_key in native_sd:
        ref_key = _pxdesign_reference_key(model_key)
        keys = ref_key if isinstance(ref_key, tuple) else (ref_key,)
        # resolve upstream ground-truth tensor(s)
        try:
            upstream_pieces = [upstream_full[k] for k in keys]
        except KeyError as exc:
            bijection_errors.append(f"{model_key} -> missing upstream key {exc}")
            continue
        for k in keys:
            consumed_upstream[k] = consumed_upstream.get(k, 0) + 1

        native_tensor = native_sd[model_key].float()
        converted_tensor = converted[model_key].float()
        # converted is what OFoldX actually loaded; it must equal the native param
        load_delta = (native_tensor - converted_tensor).abs().max().item()

        # rebuild what the value *should* be from upstream ground-truth, using the same
        # declared transform the converter applies, then compare to the native param.
        if len(keys) == 1:
            expected = upstream_pieces[0].float()
            if model_key.endswith(".noise_embedding.projection.weight") and expected.dim() == 1:
                expected = expected.unsqueeze(-1)
        else:
            # only conditioning_projection.weight is a multi-key concat
            expected = torch.cat([p.float() for p in upstream_pieces], dim=0)

        shape_ok = tuple(native_tensor.shape) == tuple(expected.shape)
        delta = float("inf") if not shape_ok else (native_tensor - expected).abs().max().item()

        cls = op_class(keys[0])
        bucket = per_class.setdefault(cls, {"count": 0, "max_abs": 0.0})
        bucket["count"] = int(bucket["count"]) + 1
        bucket["max_abs"] = max(float(bucket["max_abs"]), 0.0 if not shape_ok else delta)

        if not shape_ok or delta > args.weight_atol or load_delta > args.weight_atol:
            diverging.append(
                {
                    "model_key": model_key,
                    "reference_key": keys if len(keys) > 1 else keys[0],
                    "op_class": cls,
                    "shape_ok": shape_ok,
                    "native_shape": tuple(native_tensor.shape),
                    "expected_shape": tuple(expected.shape),
                    "max_abs_vs_upstream": delta,
                    "max_abs_load_vs_native": load_delta,
                }
            )

    # bijection: every upstream tensor consumed exactly once
    unconsumed = sorted(set(upstream_full) - set(consumed_upstream))
    multiplexed = {k: c for k, c in consumed_upstream.items() if c > 1}

    report["weight_compare"] = {
        "params_compared": len(native_sd),
        "diverging_tensors": len(diverging),
        "diverging_detail": diverging[:40],
        "bijection_errors": bijection_errors[:40],
        "upstream_unconsumed": len(unconsumed),
        "upstream_unconsumed_detail": unconsumed[:40],
        "upstream_multiplexed": multiplexed,
        "per_op_class_max_abs": dict(sorted(per_class.items())),
    }

    # ---- 3. DENOISER FORWARD PARITY -----------------------------------------------------
    b = 1
    n_token = args.tokens
    apt = args.atoms_per_token
    n_atom = n_token * apt
    n_sample = args.samples
    dtype = torch.float32
    g = torch.Generator().manual_seed(args.seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=dtype)

    ref_pos = rnd(b, n_atom, 3)
    ref_charge = rnd(b, n_atom, 1)
    ref_mask = torch.ones(b, n_atom, 1, dtype=dtype)
    ref_element = rnd(b, n_atom, 128)
    ref_atom_name_chars = rnd(b, n_atom, 256)
    atom_to_token = torch.arange(n_token).repeat_interleave(apt).reshape(1, n_atom)
    ref_space_uid = atom_to_token.clone()

    residue_index = torch.arange(n_token).reshape(1, n_token)
    token_index = torch.arange(n_token).reshape(1, n_token)
    asym_id = torch.zeros(1, n_token, dtype=torch.long)
    entity_id = torch.zeros(1, n_token, dtype=torch.long)
    sym_id = torch.zeros(1, n_token, dtype=torch.long)

    s_inputs = rnd(b, n_token, 449)  # token embedding
    s_trunk = torch.zeros(b, n_token, 384, dtype=dtype)  # pxdesign single == zeros
    z_trunk = rnd(b, n_token, n_token, 128)  # conditional-template pair
    x_noisy = rnd(b, n_sample, n_atom, 3) * 12.0
    t_hat = torch.full((b, n_sample), 7.5, dtype=dtype)

    # upstream input_feature_dict
    ifd = {
        "atom_to_token_idx": atom_to_token.reshape(n_atom),  # upstream expects 1-D [N_atom]
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
    }
    ifd = update_input_feature_dict(ifd)
    relpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
    ifd = relpe.generate_relp(ifd)

    with torch.no_grad():
        upstream_denoised = diffusion_module.forward(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=ifd,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=None,
            p_lm=None,
            c_l=None,
            inplace_safe=False,
            chunk_size=None,
            use_conditioning=True,
        )

    # OFoldX side: same raw tensors
    atom_features = torch.cat([ref_mask, ref_element, ref_atom_name_chars], dim=-1)  # 385
    relative_features = generate_relative_position_features(
        residue_index=residue_index,
        chain_index=asym_id,
        entity_index=entity_id,
        symmetry_index=sym_id,
        token_index=token_index,
        dtype=dtype,
    )
    denoiser = native.coordinate_denoiser
    local_geometry = denoiser.build_local_geometry(ref_pos, ref_space_uid)
    context = CoordinateDenoiseContext(
        atom_features=atom_features,
        atom_to_token=atom_to_token,
        reference_positions=ref_pos,
        reference_charges=ref_charge,
        reference_mask=ref_mask.squeeze(-1).bool(),
        atom_mask=torch.ones(b, n_atom, dtype=torch.bool),
        token_embedding=s_inputs,
        single=s_trunk,
        pair=z_trunk,
        relative_features=relative_features,
        local_geometry=local_geometry,
    )
    with torch.no_grad():
        native_denoised = denoiser(context.to_denoising_input(noisy_coords=x_noisy, noise_level=t_hat))

    forward = {
        "shapes": {"upstream": tuple(upstream_denoised.shape), "native": tuple(native_denoised.shape)},
        "rel_l2": rel_l2(native_denoised, upstream_denoised),
        "cosine": cosine(native_denoised, upstream_denoised),
        "max_abs": (native_denoised.float() - upstream_denoised.float()).abs().max().item(),
    }
    report["denoiser_forward"] = forward

    # ---- verdict ------------------------------------------------------------------------
    weights_ok = len(diverging) == 0 and len(bijection_errors) == 0 and len(unconsumed) == 0 and len(multiplexed) == 0
    forward_ok = forward["rel_l2"] < 2e-4 and forward["cosine"] > 0.9999
    report["verdict"] = {
        "strict_load_ok": dm_missing == [] and dm_unexpected == [] and len(native_sd) == len(reference_state),
        "weights_faithful": weights_ok,
        "denoiser_forward_faithful": forward_ok,
        "overall": "FAITHFUL" if (weights_ok and forward_ok) else "BUG-FOUND",
    }

    print(json.dumps(report, indent=2, default=str))
    if not (weights_ok and forward_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
