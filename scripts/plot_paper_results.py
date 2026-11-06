#!/usr/bin/env python3
"""Recreate thesis Figures 4.2--4.4 from committed result snapshots."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def oracle_summary(oracle: dict) -> tuple[list[int], list[float], dict[int, list[float]]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in oracle["rows"]:
        if row.get("record_type", "group_step") == "group_step":
            grouped[int(row["batch_size"])].append(float(row["mean_sparsity"]))
    batches = sorted(grouped)
    means = [float(np.mean(grouped[b])) for b in batches]
    return batches, means, grouped


def plot_heatmap(oracle: dict, out: Path, key: str) -> None:
    if key not in oracle["S"]:
        raise KeyError(f"unknown heatmap key {key!r}; try one of {list(oracle['S'])[:5]}")
    matrix = np.asarray(oracle["S"][key], dtype=float)
    fig, ax = plt.subplots(figsize=(15, 3.6))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=0.9, cmap="Blues")
    ax.set_yticks(range(len(PROJS)), labels=PROJS)
    ax.set_xticks(range(matrix.shape[1]), labels=range(matrix.shape[1]), fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Projection")
    ax.set_title(f"Oracle block sparsity: {key} (mean={matrix.mean():.3f})")
    fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012, label="Sparsity")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_oracle(oracle: dict, out: Path) -> tuple[list[int], list[float]]:
    batches, means, grouped = oracle_summary(oracle)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for batch in batches:
        # Thesis points summarize each batch group over its decode steps.
        by_group: dict[int, list[float]] = defaultdict(list)
        for row in oracle["rows"]:
            if int(row["batch_size"]) == batch:
                by_group[int(row["group_id"])].append(float(row["mean_sparsity"]))
        values = [float(np.mean(v)) for v in by_group.values()]
        ax.scatter([batch] * len(values), values, color="#1f77b4", alpha=0.65, s=34)
    ax.plot(batches, means, "-o", color="#d62728", lw=2, label="Mean over groups/steps")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Achievable Block Sparsity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return batches, means


def plot_joint(batches: list[int], means: list[float], kernel: dict, out: Path) -> dict[int, float]:
    speedups = {int(row["batch_size"]): float(row["kernel_speedup"]) for row in kernel["rows"]}
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax1.plot(batches, means, "-o", color="#1f77b4", lw=2)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(batches)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Achievable Block Sparsity", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    common = [b for b in batches if b in speedups]
    ax2.plot(common, [speedups[b] for b in common], "-s", color="#d62728", lw=2)
    ax2.set_ylabel("Kernel Speedup over Dense", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return speedups


def check_claims(batches: list[int], means: list[float], speedups: dict[int, float]) -> None:
    sparsity = dict(zip(batches, means))
    checks = [
        ("B16 sparsity near 0.64", abs(sparsity[16] - 0.64) <= 0.03),
        ("B64 sparsity above 0.50", sparsity[64] > 0.50),
        ("B2 kernel speedup near 1.95x", abs(speedups[2] - 1.95) <= 0.08),
        ("B16 kernel speedup near 1.60x", abs(speedups[16] - 1.60) <= 0.08),
        ("B64 kernel speedup near 1.20x", abs(speedups[64] - 1.20) <= 0.08),
    ]
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    print("\nMeasured values")
    for batch in batches:
        print(f"B={batch:>2}: sparsity={sparsity[batch]:.3f}, kernel_speedup={speedups.get(batch, float('nan')):.3f}x")
    if not all(passed for _, passed in checks):
        raise SystemExit("one or more thesis consistency checks failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reproduced_figures")
    parser.add_argument("--heatmap-key", default="B16_g0_step0")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oracle = load_json(ROOT / "results/oracle/perstep_batch_sweep.json")
    kernel = load_json(ROOT / "results/kernel/perstep_actual_kernel_speedup.json")
    plot_heatmap(oracle, args.out_dir / "figure_4_2_oracle_heatmap.png", args.heatmap_key)
    batches, means = plot_oracle(oracle, args.out_dir / "figure_4_3_achievable_sparsity.png")
    speedups = plot_joint(batches, means, kernel, args.out_dir / "figure_4_4_sparsity_speedup.png")
    check_claims(batches, means, speedups)


if __name__ == "__main__":
    main()
