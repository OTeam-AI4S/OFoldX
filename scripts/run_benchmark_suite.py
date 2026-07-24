"""
Run the full OFoldX machine benchmark suite.

The suite is meant for procurement-style comparisons. It runs real
Protenix workloads, operator profiles, multi-GPU communication tests,
and multi-GPU synthetic DDP tests, then writes one top-level summary.

Example:

    python scripts/run_benchmark_suite.py

    python scripts/run_benchmark_suite.py --gpu-counts 8 16 --out-dir artifacts/profile/ppu_suite
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]


@dataclass
class TaskResult:
    """
    One child benchmark command result.
    """

    name: str
    command: list[str]
    returncode: int
    seconds: float
    stdout: str
    stderr: str
    summary: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = REPO / "artifacts/profile" / f"suite_{platform.node()}_{timestamp()}"
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--gpu-counts", nargs="*", type=int, default=None)
    parser.add_argument("--real-warmup", type=int, default=20)
    parser.add_argument("--real-iters", type=int, default=100)
    parser.add_argument("--train-warmup", type=int, default=20)
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=100)
    parser.add_argument("--machine-warmup", type=int, default=20)
    parser.add_argument("--machine-iters", type=int, default=100)
    parser.add_argument("--tokens", nargs="*", type=int, default=[128, 160, 192, 196, 200, 224, 256])
    parser.add_argument("--profile-tokens", type=int, default=196)
    parser.add_argument("--ddp-real-tokens", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "data/benchmark/protenix_base_default_v0.5.0.pt")
    parser.add_argument("--sample-2orb", type=Path, default=REPO / "data/benchmark/2orb.pkl.gz")
    parser.add_argument("--protenix-root", type=Path, default=REPO / "data/benchmark/Protenix")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--skip-multigpu", action="store_true")
    parser.add_argument("--real-ddp", action="store_true", help="Also run real Protenix train under DDP.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def detect_gpu_counts() -> list[int]:
    count = torch.cuda.device_count()
    if count <= 0:
        return []
    if count >= 16:
        return [8, 16]
    if count >= 8:
        return [8]
    return [count]


def run_task(name: str, command: list[str], *, out_dir: Path, dry_run: bool) -> TaskResult:
    task_dir = out_dir / name
    task_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = task_dir / "stdout.txt"
    stderr_path = task_dir / "stderr.txt"
    command_path = task_dir / "command.json"
    command = [str(part) for part in command]
    command_path.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")

    started = time.perf_counter()
    if dry_run:
        result = TaskResult(name, command, 0, 0.0, str(stdout_path), str(stderr_path))
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return result

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    seconds = time.perf_counter() - started
    summary_path = task_dir / "summary.json"
    summary = str(summary_path) if summary_path.exists() else None
    return TaskResult(name, command, completed.returncode, seconds, str(stdout_path), str(stderr_path), summary)


def python_command(*args: str | Path) -> list[str]:
    return [sys.executable, *[str(arg) for arg in args]]


def torchrun_command(gpus: int, script: str | Path, *args: str | Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        str(script),
        *[str(arg) for arg in args],
    ]


def append_common_paths(command: list[str], args: argparse.Namespace) -> list[str]:
    return [
        *command,
        "--checkpoint",
        str(args.checkpoint),
        "--sample-2orb",
        str(args.sample_2orb),
        "--protenix-root",
        str(args.protenix_root),
    ]


def collect_tasks(args: argparse.Namespace, gpu_counts: list[int]) -> list[tuple[str, list[str]]]:
    tasks: list[tuple[str, list[str]]] = []

    tasks.append(
        (
            "preflight_compileall",
            python_command(
                "-m",
                "compileall",
                "-q",
                "scripts/profile_machine.py",
                "scripts/profile_protenix.py",
                "scripts/benchmark_protenix.py",
            ),
        )
    )
    tasks.append(("nvidia_smi", ["nvidia-smi"]))
    tasks.append(("nvidia_smi_topo", ["nvidia-smi", "topo", "-m"]))

    if not args.skip_real:
        forward = python_command(
            "scripts/benchmark_protenix.py",
            "--input",
            "2orb",
            "--workload",
            "forward",
            "--max-tokens",
            "640",
            "--variants",
            "eager-tf32",
            "eager-bf16",
            "compiled-tf32",
            "--warmup",
            args.real_warmup,
            "--iters",
            args.real_iters,
            "--out-dir",
            args.out_dir / "real_forward",
        )
        tasks.append(("real_forward", append_common_paths(forward, args)))

        core = python_command(
            "scripts/benchmark_protenix.py",
            "--input",
            "2orb",
            "--workload",
            "core",
            "--max-tokens",
            "640",
            "--variants",
            "eager-fp32",
            "eager-tf32",
            "eager-bf16",
            "compiled-tf32",
            "--warmup",
            args.real_warmup,
            "--iters",
            args.real_iters,
            "--out-dir",
            args.out_dir / "real_core",
        )
        tasks.append(("real_core", append_common_paths(core, args)))

        for token_count in args.tokens:
            train = python_command(
                "scripts/benchmark_protenix.py",
                "--input",
                "2orb",
                "--workload",
                "train",
                "--variants",
                "eager-bf16",
                "--max-tokens",
                token_count,
                "--warmup",
                args.train_warmup,
                "--iters",
                args.train_iters,
                "--out-dir",
                args.out_dir / f"real_train_{token_count}tok",
            )
            tasks.append((f"real_train_{token_count}tok", append_common_paths(train, args)))

    if not args.skip_profiles:
        for workload in ("trunk", "confidence", "diffusion"):
            profile = python_command(
                "scripts/profile_protenix.py",
                "--input",
                "2orb",
                "--workload",
                workload,
                "--mode",
                "eager",
                "--max-tokens",
                args.profile_tokens,
                "--warmup",
                args.profile_warmup,
                "--iters",
                args.profile_iters,
                "--trace-iters",
                "1",
                "--operator-profiler",
                "dispatch",
                "--out-dir",
                args.out_dir / f"profile_{workload}",
            )
            tasks.append((f"profile_{workload}", append_common_paths(profile, args)))

    if not args.skip_multigpu:
        for gpus in gpu_counts:
            tasks.append(
                (
                    f"{gpus}gpu_comm1024",
                    torchrun_command(
                        gpus,
                        "scripts/profile_machine.py",
                        "--workload",
                        "comm",
                        "--comm-size-mib",
                        "1024",
                        "--warmup",
                        args.machine_warmup,
                        "--iters",
                        args.machine_iters,
                        "--out",
                        args.out_dir / f"{gpus}gpu_comm1024.json",
                    ),
                )
            )
            tasks.append(
                (
                    f"{gpus}gpu_synthetic_train_bf16",
                    torchrun_command(
                        gpus,
                        "scripts/profile_machine.py",
                        "--workload",
                        "gpu",
                        "--preset",
                        "medium",
                        "--precision",
                        "bf16",
                        "--pass",
                        "train",
                        "--ddp",
                        "--warmup",
                        args.machine_warmup,
                        "--iters",
                        args.machine_iters,
                        "--out",
                        args.out_dir / f"{gpus}gpu_synthetic_train_bf16.json",
                    ),
                )
            )

            if args.real_ddp:
                real_ddp = torchrun_command(
                    gpus,
                    "scripts/benchmark_protenix.py",
                    "--ddp",
                    "--input",
                    "2orb",
                    "--workload",
                    "train",
                    "--variants",
                    "eager-bf16",
                    "--max-tokens",
                    args.ddp_real_tokens,
                    "--warmup",
                    "2",
                    "--iters",
                    "10",
                    "--out-dir",
                    args.out_dir / f"{gpus}gpu_real_train_{args.ddp_real_tokens}tok",
                )
                tasks.append((f"{gpus}gpu_real_train_{args.ddp_real_tokens}tok", append_common_paths(real_ddp, args)))

    return tasks


def system_metadata(gpu_counts: list[int], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "created_at": timestamp(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_counts": gpu_counts,
        "argv": sys.argv,
        "settings": {
            "real_warmup": args.real_warmup,
            "real_iters": args.real_iters,
            "train_warmup": args.train_warmup,
            "train_iters": args.train_iters,
            "machine_warmup": args.machine_warmup,
            "machine_iters": args.machine_iters,
            "profile_warmup": args.profile_warmup,
            "profile_iters": args.profile_iters,
            "tokens": args.tokens,
            "profile_tokens": args.profile_tokens,
            "ddp_real_tokens": args.ddp_real_tokens,
            "checkpoint": str(args.checkpoint),
            "sample_2orb": str(args.sample_2orb),
            "protenix_root": str(args.protenix_root),
        },
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_IB_DISABLE",
                "NCCL_SOCKET_IFNAME",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCHINDUCTOR_FX_GRAPH_CACHE",
            )
        },
    }


def main() -> None:
    args = parse_args()
    gpu_counts = args.gpu_counts if args.gpu_counts is not None else detect_gpu_counts()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "suite_summary.json"

    summary: dict[str, Any] = {
        "metadata": system_metadata(gpu_counts, args),
        "results": [],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    tasks = collect_tasks(args, gpu_counts)
    for index, (name, command) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {name}", flush=True)
        result = run_task(name, command, out_dir=args.out_dir, dry_run=args.dry_run)
        result_dict = asdict(result)
        result_dict["ok"] = result.ok
        summary["results"].append(result_dict)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        status = "ok" if result.ok else f"failed:{result.returncode}"
        print(f"  {status} {result.seconds:.2f}s", flush=True)

    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
