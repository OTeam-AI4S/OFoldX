"""Benchmark the Protenix core workload across precision and compile modes.

This script is intended for comparing GPUs without assuming the GPU model in
advance. It records device metadata, timing, memory, optional Dynamo graph-break
counts, and optional Chrome traces.

Examples:

    python scripts/benchmark_protenix.py --input 2orb --variants eager-fp32 eager-tf32 eager-bf16 compiled-tf32

    python scripts/benchmark_protenix.py --input 2orb --variants compiled-bf16 --iters 2 --explain

    python scripts/benchmark_protenix.py --input parity --variants eager-tf32 --trace-iters 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.profile_protenix import (  # noqa: E402
    DEFAULT_PROTENIX_2ORB_SAMPLE_PATH,
    DEFAULT_PROTENIX_BASE_CKPT_PATH,
    DEFAULT_PROTENIX_REFERENCE_PATH,
    BackwardWorkload,
    build_workload,
    configure_data_paths,
    run_once,
    synchronize,
    workload_metadata,
)


@dataclass(frozen=True)
class Variant:
    """A benchmark variant parsed from strings like ``compiled-bf16``."""

    mode: str
    precision: str

    @property
    def label(self) -> str:
        return f"{self.mode}_{self.precision}"


class AutocastWorkload(nn.Module):
    """Wrap a workload in CUDA autocast."""

    def __init__(self, module: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.module = module
        self.dtype = dtype
        self.requires_grad = getattr(module, "requires_grad", False)

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return self.module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/profile/protenix_benchmark")
    parser.add_argument("--input", choices=("parity", "2orb"), default="2orb")
    parser.add_argument(
        "--workload",
        choices=("core", "trunk", "diffusion", "confidence", "forward", "train"),
        default="core",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["eager-fp32", "eager-tf32", "eager-bf16", "compiled-tf32"],
        help=(
            "Benchmark variants. Format: {eager,compiled}-{fp32,tf32,bf16}. "
            "compiled-bf16 can have very high first-compile cost."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--recycling-steps", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="Crop real pkl inputs to the first N tokens.")
    parser.add_argument("--backward", action="store_true", help="Run synthetic loss backward for segment workloads.")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--explain", action="store_true", help="Run torch._dynamo.explain before timing each variant.")
    parser.add_argument("--trace-iters", type=int, default=0, help="Export a profiler trace for each variant when > 0.")
    parser.add_argument("--with-stack", action="store_true", help="Include Python stacks in profiler traces.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddp", action="store_true", help="Run under DistributedDataParallel with torchrun.")
    parser.add_argument("--dist-backend", default="nccl", help="Distributed backend for --ddp.")
    parser.add_argument("--json", type=Path, default=None, help="Optional explicit summary JSON path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PROTENIX_BASE_CKPT_PATH)
    parser.add_argument("--protenix-root", type=Path, default=DEFAULT_PROTENIX_REFERENCE_PATH)
    parser.add_argument("--sample-2orb", type=Path, default=DEFAULT_PROTENIX_2ORB_SAMPLE_PATH)
    return parser.parse_args()


def parse_variant(raw: str) -> Variant:
    try:
        mode, precision = raw.split("-", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"variant must look like eager-tf32, got {raw!r}") from error
    if mode not in {"eager", "compiled"}:
        raise argparse.ArgumentTypeError(f"variant mode must be eager or compiled, got {mode!r}")
    if precision not in {"fp32", "tf32", "bf16"}:
        raise argparse.ArgumentTypeError(f"variant precision must be fp32, tf32, or bf16, got {precision!r}")
    return Variant(mode=mode, precision=precision)


def distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_rank_zero() -> bool:
    return distributed_rank() == 0


def init_distributed_if_needed(args: argparse.Namespace) -> torch.device:
    if not args.ddp:
        return torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--ddp requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend)
    return torch.device(f"cuda:{local_rank}")


def gather_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not (dist.is_available() and dist.is_initialized()):
        return [result]
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, result)
    return [item for item in gathered if item is not None]


def aggregate_rank_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"world_size": len(results), "per_rank": results}
    numeric_keys = sorted(
        {
            key
            for result in results
            for key, value in result.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        values = [float(result[key]) for result in results if isinstance(result.get(key), int | float)]
        if not values:
            continue
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
        aggregate[f"{key}_mean"] = sum(values) / len(values)
    return aggregate


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "hostname": platform.node(),
        "rank": distributed_rank(),
        "world_size": distributed_world_size(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cuda_device_index": index,
                "cuda_device_name": props.name,
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gib": props.total_memory / 1024**3,
                "cuda_multi_processor_count": props.multi_processor_count,
            }
        )
        metadata["nvidia_smi"] = query_nvidia_smi(index)
    return metadata


def query_nvidia_smi(index: int) -> dict[str, str] | None:
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "temperature.gpu",
        "power.limit",
        "clocks.max.sm",
        "clocks.max.memory",
    ]
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    values = [part.strip() for part in completed.stdout.strip().split(",", maxsplit=len(fields) - 1)]
    if len(values) != len(fields):
        return None
    return dict(zip(fields, values, strict=True))


def git_metadata() -> dict[str, Any]:
    """Return lightweight repository state for reproducibility."""

    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def configure_precision(precision: str, device: torch.device) -> None:
    if precision == "fp32":
        torch.set_float32_matmul_precision("highest")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    elif precision == "tf32":
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif precision == "bf16":
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        raise ValueError(f"unsupported precision: {precision}")


def build_variant_workload(
    variant: Variant,
    device: torch.device,
    *,
    input_name: str,
    num_samples: int,
    recycling_steps: int,
    sampling_steps: int,
    max_tokens: int | None,
    workload_name: str,
    backward: bool,
    ddp: bool,
) -> nn.Module:
    configure_precision(variant.precision, device)
    workload: nn.Module = build_workload(
        device,
        input_name=input_name,
        num_samples=num_samples,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        max_tokens=max_tokens,
        workload_name=workload_name,
    )
    if backward:
        if workload_name == "train":
            raise ValueError("--workload train already includes backward; do not also pass --backward")
        workload = BackwardWorkload(workload).train()
    if variant.precision == "bf16":
        if device.type != "cuda":
            raise RuntimeError("bf16 autocast benchmark requires CUDA")
        workload = AutocastWorkload(workload, torch.bfloat16)
        if workload.requires_grad:
            workload.train()
        else:
            workload.eval()
    if ddp:
        # The workload is called with no forward arguments (it owns its batch and
        # places it on `device` itself), so DDP must NOT scatter inputs. Passing
        # device_ids/output_device makes DDP._pre_forward index an empty
        # moved_inputs tuple -> "IndexError: tuple index out of range". With
        # device_ids=None DDP only synchronizes gradients and leaves device
        # placement to the module, which is what this harness already does.
        # find_unused_parameters=True: the synthetic benchmark loss does not touch
        # every model parameter (some heads/branches are unused), so DDP must
        # tolerate params that receive no gradient instead of erroring on the
        # reducer. Costs a small graph traversal per iter, applied to both devices.
        workload = DistributedDataParallel(workload, find_unused_parameters=True)
    return workload


def explain_workload(workload: nn.Module) -> dict[str, Any]:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None or not hasattr(dynamo, "explain"):
        return {"available": False}
    dynamo.reset()
    try:
        with torch.no_grad():
            explanation = dynamo.explain(workload)()
    finally:
        dynamo.reset()
    return {
        "available": True,
        "graph_count": getattr(explanation, "graph_count", None),
        "graph_break_count": getattr(explanation, "graph_break_count", None),
        "op_count": getattr(explanation, "op_count", None),
        "break_reasons": [str(reason) for reason in getattr(explanation, "break_reasons", ())],
    }


@contextmanager
def timed_section() -> Iterator[dict[str, float]]:
    started = time.perf_counter()
    holder: dict[str, float] = {}
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - started


def benchmark(workload: nn.Module, *, warmup: int, iters: int, device: torch.device) -> dict[str, float]:
    # DDP does not forward the workload's custom ``requires_grad`` attribute, so
    # unwrap it; otherwise a train workload (which calls loss.backward() inside its
    # forward) runs under no_grad and dies with "does not require grad".
    base = workload.module if isinstance(workload, DistributedDataParallel) else workload
    context = torch.enable_grad() if getattr(base, "requires_grad", False) else torch.no_grad()
    with context:
        for _ in range(warmup):
            run_once(workload)
            synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        for _ in range(iters):
            run_once(workload)
            synchronize(device)
        elapsed = time.perf_counter() - started

    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    current_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    current_reserved = torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
    return {
        "iters": float(iters),
        "total_seconds": elapsed,
        "seconds_per_iter": elapsed / iters,
        "peak_memory_gib": peak_memory / 1024**3,
        "peak_memory_reserved_gib": peak_reserved / 1024**3,
        "final_memory_allocated_gib": current_allocated / 1024**3,
        "final_memory_reserved_gib": current_reserved / 1024**3,
    }


def export_trace(
    workload: nn.Module,
    *,
    label: str,
    out_dir: Path,
    trace_iters: int,
    device: torch.device,
    with_stack: bool,
) -> Path:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    trace_path = out_dir / f"{label}.trace.json"
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
        with_flops=True,
    ) as profiler:
        for _ in range(trace_iters):
            run_once(workload)
            profiler.step()
    synchronize(device)
    profiler.export_chrome_trace(str(trace_path))
    return trace_path


def run_variant(variant: Variant, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    result: dict[str, Any] = {
        "label": variant.label,
        "mode": variant.mode,
        "precision": variant.precision,
        "input": args.input,
        "workload": args.workload,
        "backward": args.backward,
        "includes_backward": args.workload == "train" or args.backward,
        "num_samples": args.num_samples,
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "max_tokens": args.max_tokens,
    }
    workload = build_variant_workload(
        variant,
        device,
        input_name=args.input,
        num_samples=args.num_samples,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        max_tokens=args.max_tokens,
        workload_name=args.workload,
        backward=args.backward,
        ddp=args.ddp,
    )
    result["workload_metadata"] = workload_metadata(workload)
    result["rank"] = distributed_rank()
    result["world_size"] = distributed_world_size()

    if args.explain:
        result["dynamo_explain"] = explain_workload(workload)

    if variant.mode == "compiled":
        with timed_section() as timer:
            workload = torch.compile(
                workload,
                mode=args.compile_mode,
                fullgraph=args.fullgraph,
                dynamic=args.dynamic,
            )
            run_once(workload)
            synchronize(device)
        result["compile_first_call_seconds"] = timer["seconds"]
        warmup = 0
    else:
        warmup = args.warmup

    result.update(benchmark(workload, warmup=warmup, iters=args.iters, device=device))

    if args.trace_iters > 0:
        trace_path = export_trace(
            workload,
            label=variant.label,
            out_dir=args.out_dir,
            trace_iters=args.trace_iters,
            device=device,
            with_stack=args.with_stack,
        )
        result["trace"] = str(trace_path)

    return result


def main() -> None:
    args = parse_args()
    if args.recycling_steps is None:
        args.recycling_steps = 4 if args.workload == "train" else 10
    if args.sampling_steps is None:
        args.sampling_steps = 20 if args.workload == "train" else 200
    variants = [parse_variant(raw) for raw in args.variants]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.json or (args.out_dir / "summary.json")
    configure_data_paths(
        checkpoint=args.checkpoint,
        protenix_root=args.protenix_root,
        sample_2orb=args.sample_2orb,
    )

    device = init_distributed_if_needed(args)
    torch.set_grad_enabled(False)

    summary: dict[str, Any] = {
        "metadata": device_metadata(device),
        "git": git_metadata(),
        "argv": sys.argv,
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "LOCAL_RANK",
                "RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCHINDUCTOR_FX_GRAPH_CACHE",
                "TORCH_LOGS",
            )
        },
        "settings": {
            "input": args.input,
            "workload": args.workload,
            "backward": args.backward,
            "includes_backward": args.workload == "train" or args.backward,
            "warmup": args.warmup,
            "iters": args.iters,
            "num_samples": args.num_samples,
            "recycling_steps": args.recycling_steps,
            "sampling_steps": args.sampling_steps,
            "max_tokens": args.max_tokens,
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
            "dynamic": args.dynamic,
            "trace_iters": args.trace_iters,
            "ddp": args.ddp,
            "dist_backend": args.dist_backend,
            "checkpoint": str(args.checkpoint),
            "protenix_root": str(args.protenix_root),
            "sample_2orb": str(args.sample_2orb),
        },
        "results": [],
    }

    for variant in variants:
        result = run_variant(variant, args, device)
        rank_results = gather_results(result)
        if is_rank_zero():
            aggregate = aggregate_rank_results(rank_results)
            aggregate.update(
                {
                    "label": result["label"],
                    "mode": result["mode"],
                    "precision": result["precision"],
                    "input": result["input"],
                    "workload": result["workload"],
                    "backward": result["backward"],
                    "includes_backward": result["includes_backward"],
                    "num_samples": result["num_samples"],
                    "recycling_steps": result["recycling_steps"],
                    "sampling_steps": result["sampling_steps"],
                    "max_tokens": result["max_tokens"],
                }
            )
            summary["results"].append(aggregate)
            print(json.dumps(aggregate, indent=2), flush=True)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if is_rank_zero():
        print(f"summary: {summary_path}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
