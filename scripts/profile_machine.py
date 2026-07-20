"""Self-contained machine benchmark for OFoldX synthetic workloads.

This script does not import tests, reference code, checkpoints, or benchmark data.
It can either run a random AlphaFold3-shaped OFoldX workload or a pure
synthetic GPU transformer workload. For multi-GPU runs, launch with ``torchrun``.

Examples:

    python scripts/profile_machine.py --preset small --precision tf32 --pass forward

    python scripts/profile_machine.py --workload gpu --preset medium --precision bf16 --pass train

    python scripts/profile_machine.py --workload gpu --gpu-kernel matmul --matrix-size 8192

    torchrun --standalone --nproc_per_node=8 scripts/profile_machine.py \
      --workload gpu --preset medium --precision bf16 --pass train --ddp

    torchrun --standalone --nproc_per_node=8 scripts/profile_machine.py \
      --workload comm --comm-size-mib 1024
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@dataclass(frozen=True)
class Preset:
    """Model/input size preset."""

    batch_size: int
    n_tokens: int
    n_atoms: int
    num_samples: int
    recycling_steps: int
    pairformer_blocks: int
    diffusion_transformer_blocks: int
    confidence_blocks: int


PRESETS: dict[str, Preset] = {
    "tiny": Preset(
        batch_size=1,
        n_tokens=16,
        n_atoms=80,
        num_samples=1,
        recycling_steps=1,
        pairformer_blocks=1,
        diffusion_transformer_blocks=1,
        confidence_blocks=0,
    ),
    "small": Preset(
        batch_size=1,
        n_tokens=64,
        n_atoms=512,
        num_samples=1,
        recycling_steps=1,
        pairformer_blocks=4,
        diffusion_transformer_blocks=2,
        confidence_blocks=1,
    ),
    "medium": Preset(
        batch_size=1,
        n_tokens=128,
        n_atoms=1024,
        num_samples=1,
        recycling_steps=2,
        pairformer_blocks=8,
        diffusion_transformer_blocks=4,
        confidence_blocks=1,
    ),
    "base": Preset(
        batch_size=1,
        n_tokens=256,
        n_atoms=2048,
        num_samples=1,
        recycling_steps=4,
        pairformer_blocks=48,
        diffusion_transformer_blocks=24,
        confidence_blocks=4,
    ),
}


class AlphaFold3SyntheticWorkload(nn.Module):
    """A synthetic end-to-end workload around AlphaFold3Model."""

    def __init__(
        self,
        model: nn.Module,
        batch: Any,
        *,
        recycling_steps: int,
        num_samples: int,
        pass_kind: str,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.recycling_steps = recycling_steps
        self.num_samples = num_samples
        self.pass_kind = pass_kind
        self.autocast_dtype = autocast_dtype

    def forward(self, _ddp_input: Tensor | None = None) -> Tensor:
        context = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )
        with context:
            output = self.model(self.batch, recycling_steps=self.recycling_steps, num_samples=self.num_samples)
            loss = synthetic_loss(output)
        if self.pass_kind == "forward":
            return loss.detach()
        loss.backward()
        return loss.detach()


class GpuTransformerBlock(nn.Module):
    """Self-contained transformer block for hardware-only GPU benchmarking."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.up = nn.Linear(hidden_size, ffn_size * 2, bias=False)
        self.down = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, hidden_size = x.shape
        residual = x
        qkv = self.qkv(self.norm_1(x))
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_size)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = F.scaled_dot_product_attention(query, key, value)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        x = residual + self.out(attn)

        residual = x
        gate, hidden = self.up(self.norm_2(x)).chunk(2, dim=-1)
        x = residual + self.down(F.silu(gate) * hidden)
        return x


class GpuSyntheticModel(nn.Module):
    """Hardware-only transformer model with no OFoldX imports."""

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        depth: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer("x", torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.02)
        self.blocks = nn.ModuleList(
            GpuTransformerBlock(hidden_size=hidden_size, num_heads=num_heads, ffn_size=ffn_size) for _ in range(depth)
        )
        self.head = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, _ddp_input: Tensor | None = None) -> Tensor:
        del _ddp_input
        x = self.x
        for block in self.blocks:
            x = block(x)
        return self.head(x).float().square().mean()


class GpuSyntheticWorkload(nn.Module):
    """Hardware-only forward/backward workload with no OFoldX imports."""

    def __init__(
        self,
        model: nn.Module,
        *,
        pass_kind: str,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.pass_kind = pass_kind
        self.autocast_dtype = autocast_dtype

    def forward(self, _ddp_input: Tensor | None = None) -> Tensor:
        context = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )
        with context:
            loss = self.model(_ddp_input)
        if self.pass_kind == "forward":
            return loss.detach()
        loss.backward()
        return loss.detach()


class MatmulWorkload(nn.Module):
    """Dense GEMM-only workload for the most stable hardware baseline."""

    def __init__(
        self,
        *,
        matrix_size: int,
        depth: int,
        pass_kind: str,
        autocast_dtype: torch.dtype | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.pass_kind = pass_kind
        self.autocast_dtype = autocast_dtype
        self.scale = matrix_size**-0.5
        self.left = nn.Parameter(torch.randn(matrix_size, matrix_size, device=device) * 0.02)
        self.right = nn.Parameter(torch.randn(matrix_size, matrix_size, device=device) * 0.02)

    def forward(self, _ddp_input: Tensor | None = None) -> Tensor:
        context = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )
        with context:
            x = self.left
            for _ in range(self.depth):
                x = (x @ self.right) * self.scale
            loss = x.float().square().mean()
        if self.pass_kind == "forward":
            return loss.detach()
        loss.backward()
        return loss.detach()


class CommWorkload(nn.Module):
    """Distributed all-reduce workload for multi-GPU communication benchmarking."""

    def __init__(self, *, size_mib: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        element_size = torch.empty((), dtype=dtype).element_size()
        numel = size_mib * 1024 * 1024 // element_size
        self.register_buffer("tensor", torch.zeros(numel, device=device, dtype=dtype))

    def forward(self, _ddp_input: Tensor | None = None) -> Tensor:
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            raise RuntimeError("--workload comm requires torchrun with at least 2 ranks")
        torch.distributed.all_reduce(self.tensor)
        return self.tensor[:1].float().sum().detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("ofoldx", "gpu", "comm"), default="ofoldx")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-tokens", type=int)
    parser.add_argument("--n-atoms", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--recycling-steps", type=int)
    parser.add_argument("--pairformer-blocks", type=int)
    parser.add_argument("--diffusion-transformer-blocks", type=int)
    parser.add_argument("--confidence-blocks", type=int)
    parser.add_argument("--gpu-kernel", choices=("transformer", "matmul"), default="transformer")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--matmul-depth", type=int, default=4)
    parser.add_argument("--comm-size-mib", type=int, default=1024)
    parser.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="tf32")
    parser.add_argument("--pass", dest="pass_kind", choices=("forward", "train"), default="forward")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--ddp", action="store_true", help="Wrap the model in DistributedDataParallel when using torchrun."
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--no-nvidia-smi", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/profile/machine_profile.json")
    return parser.parse_args()


def resolve_preset(args: argparse.Namespace) -> Preset:
    preset = PRESETS[args.preset]
    return Preset(
        batch_size=args.batch_size or preset.batch_size,
        n_tokens=args.n_tokens or preset.n_tokens,
        n_atoms=args.n_atoms or preset.n_atoms,
        num_samples=args.num_samples or preset.num_samples,
        recycling_steps=args.recycling_steps or preset.recycling_steps,
        pairformer_blocks=args.pairformer_blocks or preset.pairformer_blocks,
        diffusion_transformer_blocks=args.diffusion_transformer_blocks or preset.diffusion_transformer_blocks,
        confidence_blocks=args.confidence_blocks if args.confidence_blocks is not None else preset.confidence_blocks,
    )


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def configure_precision(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return None
    if precision == "tf32":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return None
    if precision == "bf16":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {precision}")


def build_config(preset: Preset) -> Any:
    from ofoldx.models.alphafold3.config import AlphaFold3Config
    from ofoldx.models.config import ConfidenceConfig, DenoiserConfig, TrunkConfig

    return AlphaFold3Config(
        confidence=ConfidenceConfig(num_blocks=preset.confidence_blocks),
        trunk=TrunkConfig(num_pairformer_blocks=preset.pairformer_blocks),
        denoiser=DenoiserConfig(num_token_encoder_blocks=preset.diffusion_transformer_blocks),
        recycling_steps=preset.recycling_steps,
    )


def synthetic_batch(preset: Preset, config: Any, device: torch.device) -> Any:
    from ofoldx.models.alphafold3 import AlphaFold3Input

    batch_size = preset.batch_size
    n_tokens = preset.n_tokens
    n_atoms = preset.n_atoms
    atom_to_token = torch.arange(n_atoms, device=device, dtype=torch.long).remainder(n_tokens)
    atom_to_token = atom_to_token.expand(batch_size, n_atoms).clone()
    rep_mask, atom_token_index, token_to_rep_atom = confidence_atom_fields(atom_to_token[0], n_tokens)

    token_features = torch.randn(batch_size, n_tokens, config.token_feature_size, device=device) * 0.02
    atom_features = torch.randn(batch_size, n_atoms, config.atom_feature_size, device=device) * 0.02
    reference_mask = torch.ones(batch_size, n_atoms, device=device, dtype=torch.bool)
    reference_positions = torch.randn(batch_size, n_atoms, 3, device=device) * 0.1

    return AlphaFold3Input(
        token_features=token_features,
        token_mask=torch.ones(batch_size, n_tokens, device=device, dtype=torch.bool),
        residue_index=torch.arange(n_tokens, device=device, dtype=torch.long).expand(batch_size, n_tokens).clone(),
        chain_index=torch.zeros(batch_size, n_tokens, device=device, dtype=torch.long),
        entity_index=torch.zeros(batch_size, n_tokens, device=device, dtype=torch.long),
        symmetry_index=torch.zeros(batch_size, n_tokens, device=device, dtype=torch.long),
        token_index=torch.arange(n_tokens, device=device, dtype=torch.long).expand(batch_size, n_tokens).clone(),
        atom_features=atom_features,
        atom_to_token=atom_to_token,
        atom_mask=reference_mask,
        reference_positions=reference_positions,
        reference_mask=reference_mask,
        reference_charges=torch.zeros(batch_size, n_atoms, 1, device=device),
        reference_space_id=torch.arange(n_atoms, device=device, dtype=torch.long).expand(batch_size, n_atoms).clone(),
        representative_atom_mask=rep_mask.to(device=device),
        atom_token_index=atom_token_index.to(device=device),
        token_to_rep_atom=token_to_rep_atom.to(device=device).expand(batch_size, n_tokens).clone(),
        token_bonds=torch.zeros(batch_size, n_tokens, n_tokens, device=device),
    )


def confidence_atom_fields(atom_to_token: Tensor, n_tokens: int) -> tuple[Tensor, Tensor, Tensor]:
    seen = torch.zeros(n_tokens, dtype=torch.bool)
    counts = torch.zeros(n_tokens, dtype=torch.long)
    rep_mask = torch.zeros(atom_to_token.shape[0], dtype=torch.bool)
    atom_token_index = torch.zeros(atom_to_token.shape[0], dtype=torch.long)
    token_to_rep_atom = torch.zeros(n_tokens, dtype=torch.long)
    for atom_idx, token_idx_tensor in enumerate(atom_to_token.cpu()):
        token_idx = int(token_idx_tensor)
        atom_token_index[atom_idx] = counts[token_idx]
        counts[token_idx] += 1
        if not seen[token_idx]:
            rep_mask[atom_idx] = True
            token_to_rep_atom[token_idx] = atom_idx
            seen[token_idx] = True
    return rep_mask, atom_token_index, token_to_rep_atom


def synthetic_loss(output: Any) -> Tensor:
    loss = output.pred_coords.float().square().mean()
    loss = loss + output.distogram_logits.float().square().mean()
    for tensor in (
        output.plddt_logits,
        output.pae_logits,
        output.pde_logits,
        output.resolved_logits,
        output.denoised_coords,
    ):
        if tensor is not None:
            loss = loss + tensor.float().square().mean()
    return loss


def build_ofoldx_workload(
    args: argparse.Namespace,
    preset: Preset,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    *,
    local_rank: int,
    world_size: int,
) -> nn.Module:
    from ofoldx.models.alphafold3 import AlphaFold3Model

    config = build_config(preset)
    model: nn.Module = AlphaFold3Model(config).to(device)
    model.train(args.pass_kind == "train")
    if args.ddp and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    return AlphaFold3SyntheticWorkload(
        model,
        synthetic_batch(preset, config, device),
        recycling_steps=preset.recycling_steps,
        num_samples=preset.num_samples,
        pass_kind=args.pass_kind,
        autocast_dtype=autocast_dtype,
    ).to(device)


def build_gpu_workload(
    args: argparse.Namespace,
    preset: Preset,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    *,
    local_rank: int,
    world_size: int,
) -> nn.Module:
    depth = max(1, preset.pairformer_blocks)
    if args.gpu_kernel == "matmul":
        workload: nn.Module = MatmulWorkload(
            matrix_size=args.matrix_size,
            depth=args.matmul_depth,
            pass_kind=args.pass_kind,
            autocast_dtype=autocast_dtype,
            device=device,
        ).to(device)
    else:
        model: nn.Module = GpuSyntheticModel(
            batch_size=preset.batch_size,
            seq_len=preset.n_tokens,
            hidden_size=args.hidden_size,
            num_heads=args.heads,
            ffn_size=args.ffn_size,
            depth=depth,
            device=device,
        ).to(device)
        if args.ddp and world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        workload = GpuSyntheticWorkload(
            model,
            pass_kind=args.pass_kind,
            autocast_dtype=autocast_dtype,
        ).to(device)
    if args.gpu_kernel == "matmul" and args.ddp and world_size > 1:
        workload = torch.nn.parallel.DistributedDataParallel(workload, device_ids=[local_rank])
    return workload


def build_comm_workload(
    args: argparse.Namespace, device: torch.device, autocast_dtype: torch.dtype | None
) -> nn.Module:
    dtype = autocast_dtype if autocast_dtype is not None else torch.float32
    return CommWorkload(size_mib=args.comm_size_mib, dtype=dtype, device=device).to(device)


def run_once(workload: nn.Module, *, pass_kind: str) -> float:
    if isinstance(workload, torch.nn.parallel.DistributedDataParallel):
        device = next(workload.parameters()).device
        ddp_input = torch.empty((), device=device)
        call_args = (ddp_input,)
    else:
        call_args = ()
    if pass_kind == "train":
        for parameter in workload.parameters():
            parameter.grad = None
        loss = workload(*call_args)
    else:
        with torch.no_grad():
            loss = workload(*call_args)
    return float(loss.detach())


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    holder: dict[str, float] = {}
    started = time.perf_counter()
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - started


def benchmark(workload: nn.Module, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    losses = []
    for _ in range(args.warmup):
        losses.append(run_once(workload, pass_kind=args.pass_kind))
        synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(args.iters):
        started = time.perf_counter()
        losses.append(run_once(workload, pass_kind=args.pass_kind))
        synchronize(device)
        timings.append(time.perf_counter() - started)

    return {
        "loss": losses[-1],
        "seconds_mean": statistics.mean(timings),
        "seconds_median": statistics.median(timings),
        "seconds_min": min(timings),
        "seconds_max": max(timings),
        "samples_per_second": args.iters / sum(timings),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def device_metadata(device: torch.device, *, query_smi: bool) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    metadata: dict[str, Any] = {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_index": index,
        "cuda_device_name": props.name,
        "cuda_capability": f"{props.major}.{props.minor}",
        "cuda_total_memory_gib": props.total_memory / 1024**3,
        "cuda_multi_processor_count": props.multi_processor_count,
    }
    if query_smi:
        smi = query_nvidia_smi(index)
        if smi is not None:
            metadata["nvidia_smi"] = smi
    return metadata


def query_nvidia_smi(index: int) -> dict[str, str] | None:
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
    ]
    command = [
        "nvidia-smi",
        f"--id={index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    values = [value.strip() for value in completed.stdout.strip().split(",", maxsplit=len(fields) - 1)]
    if len(values) != len(fields):
        return None
    return dict(zip(fields, values, strict=True))


def aggregate_results(local_result: dict[str, Any], rank: int, world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return [local_result]
    payload = json.dumps(local_result).encode("utf-8")
    payload_tensor = torch.tensor(list(payload), device="cuda", dtype=torch.uint8)
    size_tensor = torch.tensor([payload_tensor.numel()], device="cuda", dtype=torch.long)
    sizes = [torch.zeros_like(size_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, size_tensor)
    max_size = int(torch.stack(sizes).max().item())
    padded = torch.zeros(max_size, device="cuda", dtype=torch.uint8)
    padded[: payload_tensor.numel()] = payload_tensor
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded)
    if rank != 0:
        return []
    results = []
    for tensor, size in zip(gathered, sizes, strict=True):
        data = bytes(tensor[: int(size.item())].cpu().tolist()).decode("utf-8")
        results.append(json.loads(data))
    return results


def summarize_ranks(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    means = [result["metrics"]["seconds_mean"] for result in results]
    memories = [result["metrics"]["peak_memory_gib"] for result in results]
    summary = {
        "num_ranks": len(results),
        "seconds_mean_max": max(means),
        "seconds_mean_min": min(means),
        "seconds_mean_avg": statistics.mean(means),
        "aggregate_samples_per_second": sum(result["metrics"]["samples_per_second"] for result in results),
        "peak_memory_gib_max": max(memories),
    }
    payload_bandwidths = [
        result["metrics"]["payload_gib_per_second"]
        for result in results
        if "payload_gib_per_second" in result["metrics"]
    ]
    if payload_bandwidths:
        summary["payload_gib_per_second_min"] = min(payload_bandwidths)
        summary["payload_gib_per_second_max"] = max(payload_bandwidths)
        summary["payload_gib_per_second_avg"] = statistics.mean(payload_bandwidths)
    return summary


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("profile_machine.py requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    autocast_dtype = configure_precision(args.precision)
    preset = resolve_preset(args)

    if args.workload == "ofoldx":
        workload = build_ofoldx_workload(
            args,
            preset,
            device,
            autocast_dtype,
            local_rank=local_rank,
            world_size=world_size,
        )
    elif args.workload == "gpu":
        workload = build_gpu_workload(
            args,
            preset,
            device,
            autocast_dtype,
            local_rank=local_rank,
            world_size=world_size,
        )
    elif args.workload == "comm":
        workload = build_comm_workload(args, device, autocast_dtype)
    else:
        raise ValueError(f"unsupported workload: {args.workload}")

    compile_seconds = None
    if args.compile:
        if rank == 0:
            print("Compiling workload...", flush=True)
        with timed() as timer:
            workload = torch.compile(
                workload,
                mode=args.compile_mode,
                fullgraph=args.fullgraph,
                dynamic=args.dynamic,
            )
            run_once(workload, pass_kind=args.pass_kind)
            synchronize(device)
        compile_seconds = timer["seconds"]

    if rank == 0:
        print(
            f"Benchmarking workload={args.workload} precision={args.precision} "
            f"pass={args.pass_kind} world_size={world_size}...",
            flush=True,
        )
    metrics = benchmark(workload, args, device)
    if args.workload == "comm":
        metrics["all_reduce_size_mib"] = args.comm_size_mib
        metrics["payload_gib_per_second"] = (args.comm_size_mib / 1024) / metrics["seconds_mean"]
    local_result: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device_metadata(device, query_smi=not args.no_nvidia_smi),
        "workload": args.workload,
        "preset": asdict(preset),
        "gpu_workload": (
            {
                "kernel": args.gpu_kernel,
                "hidden_size": args.hidden_size,
                "heads": args.heads,
                "ffn_size": args.ffn_size,
                "depth": max(1, preset.pairformer_blocks),
                "matrix_size": args.matrix_size,
                "matmul_depth": args.matmul_depth,
            }
            if args.workload == "gpu"
            else None
        ),
        "comm_workload": {"size_mib": args.comm_size_mib} if args.workload == "comm" else None,
        "precision": args.precision,
        "pass": args.pass_kind,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "compile_first_call_seconds": compile_seconds,
        "ddp": args.ddp and world_size > 1,
        "warmup": args.warmup,
        "iters": args.iters,
        "metrics": metrics,
    }

    rank_results = aggregate_results(local_result, rank, world_size)
    if rank == 0:
        summary = {
            "results": rank_results,
            "summary": summarize_ranks(rank_results),
            "argv": sys.argv,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
