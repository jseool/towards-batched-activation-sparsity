#!/usr/bin/env python3
"""Compute offline kernel-replay throughput from measured per-step kernel times.

This is not end-to-end LLM tokens/s. It aggregates the measured dense and sparse
linear-projection kernel latencies for the precomputed oracle schedules and
reports the corresponding replay throughput in batch tokens per second.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _summarize(step_rows: list[dict]) -> list[dict]:
    out = []
    for b in sorted({int(r["batch_size"]) for r in step_rows}):
        rows = [r for r in step_rows if int(r["batch_size"]) == b]
        n_steps = len(rows)
        token_count = int(b * n_steps)
        dense_ms = float(np.sum([float(r.get("dense_ms", r.get("total_dense_ms"))) for r in rows]))
        sparse_ms = float(np.sum([float(r.get("sparse_ms", r.get("total_sparse_ms"))) for r in rows]))
        mean_sparsity = float(np.mean([float(r["mean_sparsity"]) for r in rows]))
        out.append({
            "batch_size": int(b),
            "n_group_steps": int(n_steps),
            "batch_tokens": int(token_count),
            "mean_sparsity": mean_sparsity,
            "dense_kernel_ms_total": dense_ms,
            "sparse_kernel_ms_total": sparse_ms,
            "dense_kernel_replay_tokens_per_s": float(token_count / (dense_ms / 1000.0)),
            "sparse_kernel_replay_tokens_per_s": float(token_count / (sparse_ms / 1000.0)),
            "throughput_speedup": float(dense_ms / sparse_ms),
            "dense_kernel_ms_per_batch_token": float(dense_ms / token_count),
            "sparse_kernel_ms_per_batch_token": float(sparse_ms / token_count),
        })
    return out


def _plot(rows: list[dict], out_dir: Path) -> None:
    batches = [int(r["batch_size"]) for r in rows]
    dense = [float(r["dense_kernel_replay_tokens_per_s"]) for r in rows]
    sparse = [float(r["sparse_kernel_replay_tokens_per_s"]) for r in rows]
    speedup = [float(r["throughput_speedup"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(batches, dense, "-o", color="#555555", lw=2.0, label="Dense Kernel Replay")
    ax.plot(batches, sparse, "-s", color="#d62728", lw=2.0, label="Sparse Kernel Replay")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Kernel Replay Throughput (tokens/s)")
    ax.set_title("Kernel Replay Throughput from Precomputed Schedules")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "perstep_kernel_replay_throughput_vs_batch.png", dpi=150)
    fig.savefig(out_dir / "perstep_kernel_replay_throughput_vs_batch.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(batches, speedup, "-s", color="#d62728", lw=2.0)
    for b, s in zip(batches, speedup):
        ax.annotate(f"{s:.2f}x", (b, s), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color="#d62728")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput Speedup over Dense")
    ax.set_title("Kernel Replay Throughput Speedup")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perstep_kernel_replay_throughput_speedup_vs_batch.png", dpi=150)
    fig.savefig(out_dir / "perstep_kernel_replay_throughput_speedup_vs_batch.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actual_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.actual_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = json.loads(args.actual_json.read_text())
    rows = _summarize(obj["step_rows"])

    out_json = out_dir / "perstep_kernel_replay_throughput.json"
    out_json.write_text(json.dumps({
        "config": {
            "actual_json": str(args.actual_json),
            "note": "Kernel replay throughput aggregates measured projection-kernel latencies only; it is not end-to-end LLM tokens/s.",
        },
        "rows": rows,
    }, indent=2))
    with (out_dir / "perstep_kernel_replay_throughput.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, out_dir)

    for r in rows:
        print(
            f"B={r['batch_size']:>3d} dense={r['dense_kernel_replay_tokens_per_s']:.1f} tok/s "
            f"sparse={r['sparse_kernel_replay_tokens_per_s']:.1f} tok/s "
            f"speedup={r['throughput_speedup']:.3f}x",
            flush=True,
        )
    print(out_json)


if __name__ == "__main__":
    main()
