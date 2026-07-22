r"""
OFoldX AlphaFold 3 parameter-key schema for checkpoint conversion.

The DeepMind checkpoint converter
([`deepmind`][ofoldx.models.alphafold3.converters.deepmind]) uses
[`resolve`][ofoldx.models.alphafold3.converters._schema.resolve] to map each OFoldX model key to its
submodule ``section``, within-section ``context``, leaf ``rest``, and optional stack ``block`` index.
DeepMind parameter names and tensor-layout transforms remain in the converter; this module describes only
the OFoldX model-key tree.

The block index is carried separately from ``rest`` because stacked Haiku tensors consume it as an array
index rather than as part of the source parameter name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AF3Node:
    r"""One resolved OFoldX-AF3 model key.

    Attributes:
        section: The top-level submodule the key belongs to (e.g. ``"pairformer"``, ``"atom_encoder"``).
        context: A within-section discriminator (e.g. the pairformer stack that hosts the block, or the
            ``"input"`` / ``"diffusion"`` atom-encoder instance). Empty when the section has a single context.
        rest: The leaf key inside the resolved submodule, with the section prefix and stack index removed.
        block: The stack/layer index when the section is stacked, otherwise ``None``.
    """

    section: str
    context: str
    rest: str
    block: int | None


_MSA_STACK_PREFIX = "msa_stack."
_PAIRFORMER_BLOCK = re.compile(r"blocks\.(\d+)\.(.+)")


def resolve(model_key: str) -> AF3Node | None:
    r"""Route an OFoldX-AF3 model key to its ``(section, context, rest, block)`` schema node.

    Returns ``None`` for keys outside the shared AF3 tree (model-specific heads are handled by each converter
    before it consults this schema).
    """

    if model_key.startswith("input_embeddings.atom_encoder."):
        return AF3Node("atom_encoder", "input", model_key.removeprefix("input_embeddings.atom_encoder."), None)
    if model_key.startswith("pairformer_stack."):
        match = _PAIRFORMER_BLOCK.fullmatch(model_key.removeprefix("pairformer_stack."))
        if match is not None:
            return AF3Node("pairformer", "trunk", match.group(2), int(match.group(1)))
    if model_key.startswith("msa_trunk."):
        return _resolve_msa(model_key.removeprefix("msa_trunk."))
    if model_key.startswith("template_embedder."):
        rest = model_key.removeprefix("template_embedder.")
        if match := _PAIRFORMER_BLOCK.fullmatch(rest):
            return AF3Node("pairformer", "template", match.group(2), int(match.group(1)))
        return AF3Node("template", "", rest, None)
    if model_key.startswith("confidence_head."):
        rest = model_key.removeprefix("confidence_head.")
        if match := re.fullmatch(r"pairformer_stack\.blocks\.(\d+)\.(.+)", rest):
            return AF3Node("pairformer", "confidence", match.group(2), int(match.group(1)))
        return AF3Node("confidence", "", rest, None)
    if model_key.startswith("coordinate_denoiser."):
        return _resolve_denoiser(model_key.removeprefix("coordinate_denoiser."))
    return AF3Node("top", "", model_key, None)


def _resolve_msa(rest: str) -> AF3Node | None:
    if rest in ("msa_projection.weight", "token_projection.weight"):
        return AF3Node("msa", "", rest, None)
    match = _PAIRFORMER_BLOCK.fullmatch(rest)
    if match is None:
        return None
    block = int(match.group(1))
    inner = match.group(2)
    if inner.startswith("outer_product_mean."):
        return AF3Node("msa_outer_product_mean", "", inner.removeprefix("outer_product_mean."), block)
    averaging_prefix = _MSA_STACK_PREFIX + "msa_pair_weighted_averaging."
    if inner.startswith(averaging_prefix):
        return AF3Node("msa_attention", "", inner.removeprefix(averaging_prefix), block)
    if inner.startswith(_MSA_STACK_PREFIX + "feed_forward."):
        return AF3Node("msa_transition", "", inner.removeprefix(_MSA_STACK_PREFIX + "feed_forward."), block)
    if inner.startswith("pair_stack."):
        return AF3Node("pairformer", "msa_pair", inner.removeprefix("pair_stack."), block)
    return None


def _resolve_denoiser(rest: str) -> AF3Node | None:
    if rest.startswith("conditioner."):
        return AF3Node("denoiser_conditioner", "", rest.removeprefix("conditioner."), None)
    if rest.startswith("atom_encoder."):
        return AF3Node("atom_encoder", "diffusion", rest.removeprefix("atom_encoder."), None)
    if rest.startswith("token_refiner."):
        return AF3Node("token_refiner", "", rest.removeprefix("token_refiner."), None)
    if rest.startswith("atom_decoder."):
        return AF3Node("atom_decoder", "", rest.removeprefix("atom_decoder."), None)
    return None
