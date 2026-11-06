#!/usr/bin/env python3
"""Profile block-sparse kernel configurations for a target GPU."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ["TEAL_EXP_ENABLE_SPLIT_K_1D"] = "0"
os.environ["TEAL_EXP_USE_RUN_COMPRESSED_METADATA"] = "0"
os.environ["TEAL_EXP_SORT_ACTIVE_BLOCKS"] = "0"

from kernels.shared_mask_batched_triton_min_runtime import (  # noqa: E402
    _PREPACK_AUTOTUNE_CONFIGS,
    _build_active_k_bitmap,
    _get_prepacked_weight_block_major,
    _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload,
    _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload_splitk,
    _prepacked_block_sparse_gemm_from_active_blocks_splitk,
    prepacked_block_sparse_gemm_bitmap_preload_batched_kernel,
)

SHAPES = {
    "qkvo": (4096, 4096),
    "gate_up": (4096, 11008),
    "down": (11008, 4096),
}
BLOCK_K = 64


def measure_cuda_graph(fn, warmup: int, iterations: int, repeats: int) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()

    measurements = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        measurements.append(start.elapsed_time(end) / iterations)
    return float(statistics.median(measurements))


def split_k_candidates(active_blocks: int, sm_count: int) -> list[int]:
    maximum = min(active_blocks, sm_count)
    return [2**power for power in range(int(math.log2(maximum)) + 1)]


def tile_configs(batch_size: int) -> list[tuple[int, int, int, int]]:
    batch_tile_cap = max(batch_size, 16)
    return list(
        dict.fromkeys(
            (
                config.kwargs["BLOCK_B"],
                config.kwargs["BLOCK_N"],
                config.num_warps,
                config.num_stages,
            )
            for config in _PREPACK_AUTOTUNE_CONFIGS
            if config.kwargs["BLOCK_B"] <= batch_tile_cap
        )
    )


def profile_density(
    batch_size: int,
    shape_name: str,
    d_in: int,
    d_out: int,
    density: float,
    weight: torch.Tensor,
    x: torch.Tensor,
    dense_ms: float,
    warmup: int,
    iterations: int,
    repeats: int,
    sm_count: int,
) -> dict:
    if density >= 1.0:
        return {
            "dense_ms": dense_ms,
            "sparse_ms": dense_ms,
            "speedup": 1.0,
            "mask": "dense",
            "block_k": None,
            "K": 1,
            "tile": None,
        }

    total_blocks = math.ceil(d_in / BLOCK_K)
    active_count = max(1, int(math.floor(density * total_blocks)))
    active = torch.randperm(total_blocks, device="cuda")[:active_count].sort()[0].to(torch.int32)
    bitmap = _build_active_k_bitmap(active, total_blocks, "cuda")
    cache_prefix = f"profile_b{batch_size}_{shape_name}"
    _get_prepacked_weight_block_major(weight, BLOCK_K, cache_prefix=cache_prefix)

    best_ms = float("inf")
    best = None

    output = torch.zeros(batch_size, 1, d_out, device="cuda", dtype=weight.dtype)

    def bitmap_k1() -> None:
        _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload(
            x,
            weight,
            active,
            output,
            cache_prefix=cache_prefix,
            total_k_blocks=total_blocks,
            prebuilt_bitmap=bitmap,
        )

    bitmap_ms = measure_cuda_graph(bitmap_k1, warmup, iterations, repeats)
    config = prepacked_block_sparse_gemm_bitmap_preload_batched_kernel.best_config
    best_ms = bitmap_ms
    best = {
        "mask": "bitmap",
        "block_k": BLOCK_K,
        "K": 1,
        "tile": [
            config.kwargs["BLOCK_B"],
            config.kwargs["BLOCK_N"],
            config.num_warps,
            config.num_stages,
        ],
    }

    def naive_k1() -> None:
        _prepacked_block_sparse_gemm_from_active_blocks_splitk(
            x, weight, active, output, cache_prefix=cache_prefix, num_split_k=1
        )

    naive_ms = measure_cuda_graph(naive_k1, warmup, iterations, repeats)
    if naive_ms < best_ms:
        best_ms = naive_ms
        best = {"mask": "naive", "block_k": BLOCK_K, "K": 1, "tile": None}

    for split_k in split_k_candidates(active_count, sm_count):
        if split_k == 1:
            continue
        for block_b, block_n, warps, stages in tile_configs(batch_size):
            output = torch.zeros(batch_size, 1, d_out, device="cuda", dtype=weight.dtype)

            def bitmap_split() -> None:
                _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload_splitk(
                    x,
                    weight,
                    active,
                    output,
                    cache_prefix=cache_prefix,
                    total_k_blocks=total_blocks,
                    prebuilt_bitmap=bitmap,
                    num_split_k=split_k,
                    tile_b=block_b,
                    tile_n=block_n,
                    tile_warps=warps,
                    tile_stages=stages,
                )

            elapsed = measure_cuda_graph(bitmap_split, warmup, iterations, repeats)
            if elapsed < best_ms:
                best_ms = elapsed
                best = {
                    "mask": "bitmap",
                    "block_k": BLOCK_K,
                    "K": split_k,
                    "tile": [block_b, block_n, warps, stages],
                }

            def naive_split() -> None:
                _prepacked_block_sparse_gemm_from_active_blocks_splitk(
                    x,
                    weight,
                    active,
                    output,
                    cache_prefix=cache_prefix,
                    num_split_k=split_k,
                    tile_b=block_b,
                    tile_n=block_n,
                    tile_warps=warps,
                    tile_stages=stages,
                )

            elapsed = measure_cuda_graph(naive_split, warmup, iterations, repeats)
            if elapsed < best_ms:
                best_ms = elapsed
                best = {
                    "mask": "naive",
                    "block_k": BLOCK_K,
                    "K": split_k,
                    "tile": [block_b, block_n, warps, stages],
                }

    best.update(dense_ms=dense_ms, sparse_ms=best_ms, speedup=dense_ms / best_ms)
    return best


def profile_batch(args: argparse.Namespace, batch_size: int, densities: list[float]) -> dict:
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    results = {}
    for shape_name, (d_in, d_out) in SHAPES.items():
        torch.cuda.empty_cache()
        weight = torch.randn(d_out, d_in, device="cuda", dtype=dtype)
        transposed = weight.t().contiguous()
        x = torch.randn(batch_size, 1, d_in, device="cuda", dtype=dtype)
        dense_ms = measure_cuda_graph(
            lambda: torch.mm(x.view(batch_size, d_in), transposed),
            args.warmup,
            args.iterations,
            args.repeats,
        )
        shape_results = {}
        for density in densities:
            record = profile_density(
                batch_size,
                shape_name,
                d_in,
                d_out,
                density,
                weight,
                x,
                dense_ms,
                args.warmup,
                args.iterations,
                args.repeats,
                sm_count,
            )
            shape_results[str(density)] = record
            print(
                f"B={batch_size:>2} {shape_name:>7} density={density:.1f} "
                f"{record['mask']}/K={record['K']} {record['speedup']:.3f}x",
                flush=True,
            )
        results[shape_name] = shape_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_sizes", default="2,4,8,16,32,64")
    parser.add_argument("--densities", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    batches = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    densities = [float(value) for value in args.densities.split(",") if value.strip()]

    device = torch.cuda.get_device_properties(0)
    metadata = {
        "gpu": device.name,
        "compute_capability": f"{device.major}.{device.minor}",
        "sm_count": device.multi_processor_count,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "block_k": BLOCK_K,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
    }
    (args.out_dir / "profile_metadata.json").write_text(json.dumps(metadata, indent=2))

    for batch_size in batches:
        results = profile_batch(args, batch_size, densities)
        output = args.out_dir / f"llama2_kernel_best_sweep_B{batch_size}.json"
        output.write_text(json.dumps(results, indent=2))
        print(output)


if __name__ == "__main__":
    main()
