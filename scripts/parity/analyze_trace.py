"""Stream a (large) torch.profiler chrome trace and aggregate kernel/op time.

The trace can be many GB, so we stream the top-level "traceEvents" array item by item
with ijson instead of json.load. We aggregate:
  - GPU kernels (cat == "kernel"): self device time by name -> the real compute cost
  - CPU ops (cat in {"cpu_op","operator"}): self host time by name -> launch/overhead cost
and bucket kernels into coarse families (attention / matmul / layernorm / elementwise / etc).
"""

from __future__ import annotations

import argparse
import collections
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    kernel_us: dict[str, float] = collections.defaultdict(float)
    kernel_calls: dict[str, int] = collections.defaultdict(int)
    cpu_us: dict[str, float] = collections.defaultdict(float)
    cpu_calls: dict[str, int] = collections.defaultdict(int)
    total_kernel_us = 0.0
    n_events = 0

    # Dependency-free streaming: the chrome trace is one giant JSON object with a
    # "traceEvents": [ {..}, {..}, ... ] array. We don't json.load the whole thing
    # (it's GB-scale); instead we incrementally decode one event object at a time
    # with raw_decode, advancing past the array.
    import json as _json

    decoder = _json.JSONDecoder()

    def iter_events(path: str):
        buf = ""
        with open(path, "r") as fh:
            # find the start of the traceEvents array
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    return
                buf += chunk
                key = buf.find('"traceEvents"')
                if key != -1:
                    lb = buf.find("[", key)
                    if lb != -1:
                        buf = buf[lb + 1 :]
                        break
            while True:
                buf = buf.lstrip()
                while buf[:1] in (",",):
                    buf = buf[1:].lstrip()
                if buf[:1] == "]":
                    return
                try:
                    obj, idx = decoder.raw_decode(buf)
                    buf = buf[idx:]
                    yield obj
                except ValueError:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        return
                    buf += chunk

    for ev in iter_events(args.trace):
        n_events += 1
        cat = ev.get("cat")
        if cat is None:
            continue
        name = ev.get("name", "?")
        dur = ev.get("dur")
        if dur is None:
            continue
        dur = float(dur)
        if cat == "kernel":
            kernel_us[name] += dur
            kernel_calls[name] += 1
            total_kernel_us += dur
        elif cat in ("cpu_op", "operator", "user_annotation"):
            cpu_us[name] += dur
            cpu_calls[name] += 1

    def bucket(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ("attention", "fmha", "flash", "softmax", "scaled_dot")):
            return "attention/softmax"
        if any(k in n for k in ("gemm", "matmul", "cutlass", "sgemm", "wgrad", "dgrad", "ampere", "cublas", "addmm")):
            return "matmul/gemm"
        if any(k in n for k in ("layer_norm", "layernorm", "norm")):
            return "layernorm/norm"
        if any(k in n for k in ("elementwise", "vectorized", "pointwise", "mul", "add", "copy", "cast", "to_copy")):
            return "elementwise/copy"
        if "reduce" in n or "sum" in n or "mean" in n:
            return "reduction"
        return "other"

    fam_us: dict[str, float] = collections.defaultdict(float)
    for name, us in kernel_us.items():
        fam_us[bucket(name)] += us

    print(f"events_scanned={n_events}")
    print(f"total_gpu_kernel_ms={total_kernel_us/1e3:.1f}")
    print("\n=== GPU time by family (% of kernel time) ===")
    for fam, us in sorted(fam_us.items(), key=lambda x: -x[1]):
        print(f"  {fam:22s} {us/1e3:10.1f} ms  {100*us/max(total_kernel_us,1):5.1f}%")

    print(f"\n=== top {args.top} GPU kernels by total self time ===")
    for name, us in sorted(kernel_us.items(), key=lambda x: -x[1])[: args.top]:
        print(f"  {us/1e3:10.1f} ms  x{kernel_calls[name]:<7d} {name[:90]}")

    print(f"\n=== top {args.top} CPU ops by total self host time ===")
    for name, us in sorted(cpu_us.items(), key=lambda x: -x[1])[: args.top]:
        print(f"  {us/1e3:10.1f} ms  x{cpu_calls[name]:<7d} {name[:90]}")


if __name__ == "__main__":
    main()
