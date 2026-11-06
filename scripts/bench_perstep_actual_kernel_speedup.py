#!/usr/bin/env python3
"""Benchmark lookup-selected kernels for per-step oracle allocations.

This uses the existing per-batch lookup tables only to choose the kernel
configuration. It then directly measures those selected kernels and averages
the measured projection cost over each group/step allocation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from kernels.shared_mask_batched_triton_min_runtime import (  # noqa: E402
    _build_active_k_bitmap,
    _get_prepacked_weight_block_major,
    _prepacked_block_sparse_gemm_from_active_blocks_splitk,
    _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload,
    _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload_splitk,
)

PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROJ_SHAPES = {
    "q_proj": "qkvo",
    "k_proj": "qkvo",
    "v_proj": "qkvo",
    "o_proj": "qkvo",
    "gate_proj": "gate_up",
    "up_proj": "gate_up",
    "down_proj": "down",
}
SHAPES = {
    "qkvo": (4096, 4096),
    "gate_up": (4096, 11008),
    "down": (11008, 4096),
}
NUM_LAYERS = 32
BLOCK_K = 64


def _nearest_density(shape_rows: dict, density: float) -> str:
    return min(shape_rows.keys(), key=lambda k: abs(float(k) - float(density)))


def _bench_cg(fn, warmup: int, iters: int) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def _bench_median(fn, warmup: int, iters: int, repeats: int) -> float:
    return float(statistics.median(_bench_cg(fn, warmup, iters) for _ in range(repeats)))


def _active_blocks(d_in: int, density: float) -> torch.Tensor:
    total_k = math.ceil(d_in / BLOCK_K)
    n_k = max(1, min(total_k, int(math.floor(float(density) * total_k))))
    return torch.randperm(total_k, device="cuda")[:n_k].sort()[0].to(torch.int32)


def _sparse_fn(selected: dict, x: torch.Tensor, weight: torch.Tensor, active: torch.Tensor,
               bitmap: torch.Tensor, out: torch.Tensor, total_k: int, cache_prefix: str):
    mask = str(selected.get("mask", selected.get("best_kern", "")))
    split_k = int(selected.get("K", selected.get("best_K", 1)) or 1)
    tile = selected.get("tile", selected.get("best_tile")) or [None, None, None, None]
    tile_b, tile_n, tile_warps, tile_stages = tile

    if mask == "dense":
        return None
    if mask == "naive":
        return lambda: _prepacked_block_sparse_gemm_from_active_blocks_splitk(
            x, weight, active, out, cache_prefix=cache_prefix, num_split_k=split_k,
            tile_b=tile_b, tile_n=tile_n, tile_warps=tile_warps, tile_stages=tile_stages)
    if mask in {"bitmap", "bpl"} and split_k <= 1:
        return lambda: _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload(
            x, weight, active, out, cache_prefix=cache_prefix,
            total_k_blocks=total_k, prebuilt_bitmap=bitmap)
    if mask in {"bitmap", "bpl"}:
        return lambda: _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload_splitk(
            x, weight, active, out, cache_prefix=cache_prefix,
            total_k_blocks=total_k, prebuilt_bitmap=bitmap, num_split_k=split_k,
            tile_b=tile_b, tile_n=tile_n, tile_warps=tile_warps, tile_stages=tile_stages)
    raise ValueError(f"unknown selected kernel mask={mask}")


def _complete_batches(rows: list[dict], expected_groups: int, expected_steps: int) -> set[int]:
    seen = defaultdict(set)
    for row in rows:
        seen[int(row["batch_size"])].add((int(row["group_id"]), int(row["step"])))
    return {bs for bs, keys in seen.items() if len(keys) >= expected_groups * expected_steps}


def _selected_items_for_row(sweep: dict, matrix: list[list[float]]) -> list[dict]:
    items = []
    for pi, proj in enumerate(PROJS):
        shape = PROJ_SHAPES[proj]
        shape_rows = sweep[shape]
        for li in range(NUM_LAYERS):
            density = max(0.0, min(1.0, 1.0 - float(matrix[pi][li])))
            density_key = _nearest_density(shape_rows, density)
            selected = dict(shape_rows[density_key])
            items.append({
                "layer": li,
                "proj": proj,
                "shape": shape,
                "density": density,
                "density_key": density_key,
                "mask": selected.get("mask", selected.get("best_kern")),
                "K": int(selected.get("K", selected.get("best_K", 1)) or 1),
                "tile": selected.get("tile", selected.get("best_tile")),
            })
    return items


def _case_key(item: dict) -> tuple:
    return (item["shape"], item["density_key"], item["mask"], int(item["K"]), tuple(item["tile"] or []))


def _benchmark_case(batch_size: int, item: dict, warmup: int, iters: int, repeats: int) -> dict:
    shape = item["shape"]
    d_in, d_out = SHAPES[shape]
    density = float(item["density_key"])
    total_k = math.ceil(d_in / BLOCK_K)
    x = torch.randn(batch_size, 1, d_in, device="cuda", dtype=torch.float16)
    weight = torch.randn(d_out, d_in, device="cuda", dtype=torch.float16)
    wt = weight.t().contiguous()
    out = torch.zeros(batch_size, 1, d_out, device="cuda", dtype=torch.float16)
    active = _active_blocks(d_in, density)
    bitmap = _build_active_k_bitmap(active, total_k, "cuda")
    cache_prefix = f"perstep_actual_B{batch_size}_{shape}_{density}_{item['mask']}_{item['K']}"
    _get_prepacked_weight_block_major(weight, BLOCK_K, cache_prefix=cache_prefix)

    dense_fn = lambda: torch.mm(x.view(batch_size, d_in), wt)
    sparse_fn = _sparse_fn(item, x, weight, active, bitmap, out, total_k, cache_prefix)
    dense_ms = _bench_median(dense_fn, warmup, iters, repeats)
    sparse_ms = dense_ms if sparse_fn is None else _bench_median(sparse_fn, warmup, iters, repeats)
    return {
        **item,
        "batch_size": batch_size,
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
        "speedup": dense_ms / sparse_ms if sparse_ms > 0 else 1.0,
    }


def _plot_original_style(rows: list[dict], speedup_rows: list[dict], out_dir: Path) -> None:
    batch_sizes = sorted({int(r["batch_size"]) for r in rows})
    sparsity_means = []
    for bs in batch_sizes:
        vals = [float(r["mean_sparsity"]) for r in rows if int(r["batch_size"]) == bs]
        sparsity_means.append(float(np.mean(vals)) if vals else float("nan"))

    speedups = {int(r["batch_size"]): float(r["kernel_speedup"]) for r in speedup_rows}
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax1.plot(batch_sizes, sparsity_means, "-o", color="#1f77b4", lw=2.0, label="Mean Achievable Sparsity")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(batch_sizes)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Sparsity", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    sx = [b for b in batch_sizes if b in speedups]
    if sx:
        sy = [speedups[b] for b in sx]
        ax2.plot(sx, sy, "-s", color="#d62728", lw=2.0, label="Kernel Speedup")
        for b, s in zip(sx, sy):
            ax2.annotate(f"{s:.2f}x", (b, s), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8, color="#d62728")
    ax2.set_ylabel("Kernel Speedup over Dense", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Mean Achievable Sparsity and Kernel Speedup")
    fig.tight_layout()
    fig.savefig(out_dir / "perstep_sparsity_speedup_vs_batch.png", dpi=150)
    fig.savefig(out_dir / "perstep_sparsity_speedup_vs_batch.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perstep_json", type=Path, required=True)
    parser.add_argument("--lookup_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_sizes", default=None)
    parser.add_argument("--expected_groups", type=int, default=5)
    parser.add_argument("--expected_steps", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out_dir = args.out_dir or args.perstep_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.perstep_json.read_text())
    rows = [r for r in data.get("rows", []) if r.get("record_type") == "group_step"]
    matrices = data.get("S", {})
    complete = _complete_batches(rows, args.expected_groups, args.expected_steps)
    requested = {int(x) for x in args.batch_sizes.split(",") if x.strip()} if args.batch_sizes else complete
    target_batches = sorted(complete & requested)

    all_step_rows = []
    all_case_rows = []
    speedup_rows = []
    for bs in target_batches:
        lookup_path = args.lookup_dir / f"llama2_kernel_best_sweep_B{bs}.json"
        if not lookup_path.exists():
            continue
        sweep = json.loads(lookup_path.read_text())
        step_items = {}
        unique = {}
        for row in rows:
            if int(row["batch_size"]) != bs:
                continue
            key = f"B{bs}_g{int(row['group_id'])}_step{int(row['step'])}"
            if key not in matrices:
                continue
            items = _selected_items_for_row(sweep, matrices[key])
            step_items[key] = (row, items)
            for item in items:
                unique.setdefault(_case_key(item), item)

        bench = {}
        print(f"B={bs}: benchmarking {len(unique)} unique kernel cases over {len(step_items)} group/steps", flush=True)
        for idx, (case_key, item) in enumerate(unique.items(), start=1):
            rec = _benchmark_case(bs, item, args.warmup, args.iters, args.repeats)
            bench[case_key] = rec
            all_case_rows.append(rec)
            print(
                f"  [{idx}/{len(unique)}] {item['shape']} d={item['density_key']} "
                f"{item['mask']}/K={item['K']} speedup={rec['speedup']:.3f}x",
                flush=True,
            )

        batch_step_speedups = []
        for key, (row, items) in step_items.items():
            dense_total = 0.0
            sparse_total = 0.0
            for item in items:
                rec = bench[_case_key(item)]
                dense_total += float(rec["dense_ms"])
                sparse_total += float(rec["sparse_ms"])
            speedup = dense_total / sparse_total if sparse_total > 0 else 1.0
            batch_step_speedups.append(speedup)
            all_step_rows.append({
                "batch_size": bs,
                "group_id": int(row["group_id"]),
                "step": int(row["step"]),
                "mean_sparsity": float(row["mean_sparsity"]),
                "kernel_speedup": float(speedup),
                "dense_ms": float(dense_total),
                "sparse_ms": float(sparse_total),
            })

        speedup_rows.append({
            "batch_size": bs,
            "kernel_speedup": float(np.mean(batch_step_speedups)),
            "mean_speedup": float(np.mean(batch_step_speedups)),
            "n_group_steps": int(len(batch_step_speedups)),
        })
        print(f"B={bs}: mean actual kernel speedup={speedup_rows[-1]['kernel_speedup']:.3f}x", flush=True)

    out_json = out_dir / "perstep_actual_kernel_speedup.json"
    out_json.write_text(json.dumps({
        "config": {
            "perstep_json": str(args.perstep_json),
            "lookup_dir": str(args.lookup_dir),
            "batch_sizes": target_batches,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        "rows": speedup_rows,
        "step_rows": all_step_rows,
    }, indent=2))

    if all_step_rows:
        with (out_dir / "perstep_actual_kernel_speedup_step_rows.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_step_rows)
    if all_case_rows:
        with (out_dir / "perstep_actual_kernel_speedup_cases.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_case_rows)

    _plot_original_style(rows, speedup_rows, out_dir)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
