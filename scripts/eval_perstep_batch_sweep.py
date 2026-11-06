#!/usr/bin/env python3
"""Per-step batch-size sweep for the batch-shared block oracle.

For each batch size and length-sorted Alpaca group, this runs an independent
round-robin oracle at each selected decode step. The quality budget is measured
at the same step using TEAL-0.5 channel masking.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_perstep_budget_sweep import (
    CACHE_DIR,
    MODEL_NAME,
    NUM_LAYERS,
    PROJS,
    _build,
    _collect_stats,
    _length_groups,
    _load_alpaca_examples,
    _load_thresholds_for_budgets,
    _round_robin_step,
    _teal_step_budget,
)


def _plot_batch_sparsity(rows: List[dict], out_dir: Path) -> None:
    if not rows:
        return
    batch_sizes = sorted({int(r["batch_size"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    means = []
    for bs in batch_sizes:
        vals = [
            float(r["mean_sparsity"])
            for r in rows
            if int(r["batch_size"]) == bs and r["record_type"] == "group_step"
        ]
        group_vals = []
        for gid in sorted({int(r["group_id"]) for r in rows if int(r["batch_size"]) == bs}):
            gv = [
                float(r["mean_sparsity"])
                for r in rows
                if int(r["batch_size"]) == bs
                and int(r["group_id"]) == gid
                and r["record_type"] == "group_step"
            ]
            if gv:
                group_vals.append(float(np.mean(gv)))
        ax.scatter([bs] * len(group_vals), group_vals, color="#1f77b4", alpha=0.65, s=34, zorder=3)
        means.append(float(np.mean(vals)) if vals else float("nan"))
    ax.plot(batch_sizes, means, "-o", color="#d62728", lw=2.0, label="Mean over groups/steps", zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Sparsity")
    ax.set_title("Mean Achievable Sparsity (11 Steps Averaged)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "perstep_batch_sweep_oracle_mean_sparsity.png", dpi=150)
    fig.savefig(out_dir / "perstep_batch_sweep_oracle_mean_sparsity.pdf")
    plt.close(fig)


def _plot_sparsity_speedup(rows: List[dict], speedup_json: Path | None, out_dir: Path) -> None:
    if not rows:
        return
    batch_sizes = sorted({int(r["batch_size"]) for r in rows})
    sparsity_means = []
    for bs in batch_sizes:
        vals = [
            float(r["mean_sparsity"])
            for r in rows
            if int(r["batch_size"]) == bs and r["record_type"] == "group_step"
        ]
        sparsity_means.append(float(np.mean(vals)) if vals else float("nan"))

    speedups = {}
    if speedup_json and speedup_json.exists():
        obj = json.loads(speedup_json.read_text())
        for r in obj.get("rows", obj if isinstance(obj, list) else []):
            if "batch_size" in r:
                for key in ["speedup", "kernel_speedup", "mean_speedup"]:
                    if key in r:
                        speedups[int(r["batch_size"])] = float(r[key])
                        break

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
    if speedups:
        sx = [b for b in batch_sizes if b in speedups]
        sy = [speedups[b] for b in sx]
        ax2.plot(sx, sy, "-s", color="#d62728", lw=2.0, label="Kernel Speedup")
    ax2.set_ylabel("Kernel Speedup over Dense", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Mean Achievable Sparsity and Kernel Speedup")
    fig.tight_layout()
    fig.savefig(out_dir / "perstep_sparsity_speedup_vs_batch.png", dpi=150)
    fig.savefig(out_dir / "perstep_sparsity_speedup_vs_batch.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_sizes", default="2,4,8,16,32,64,128")
    ap.add_argument("--num_groups", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--steps", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--block_k", type=int, default=64)
    ap.add_argument("--budget_sparsity", type=float, default=0.5)
    ap.add_argument("--grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--speedup_json", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    out_json = args.out_dir / "perstep_batch_sweep.json"

    print("[1/5] tokenizer + length groups", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    samples = {}
    group_meta = []
    for bs in batch_sizes:
        examples = _load_alpaca_examples(
            tok,
            num_samples=bs * args.num_groups,
            seed=args.seed * 100003 + bs,
            decode=args.decode,
            max_prompt_len=args.max_prompt_len,
        )
        groups = _length_groups(examples, bs, args.num_groups)
        samples[bs] = (examples, groups)
        for gid, idx in enumerate(groups):
            lens = [examples[i]["prompt_len"] for i in idx]
            group_meta.append({
                "batch_size": int(bs),
                "group_id": int(gid),
                "dataset_indices": [int(examples[i]["dataset_idx"]) for i in idx],
                "prompt_len_min": int(np.min(lens)),
                "prompt_len_mean": float(np.mean(lens)),
                "prompt_len_max": int(np.max(lens)),
            })
        lens_all = [m["prompt_len_mean"] for m in group_meta if m["batch_size"] == bs]
        print(f"  B={bs}: {args.num_groups} groups, mean prompt len range {min(lens_all):.1f}-{max(lens_all):.1f}", flush=True)

    print("[2/5] TEAL-0.5 thresholds", flush=True)
    thresholds = _load_thresholds_for_budgets([float(args.budget_sparsity)], args.device)[float(args.budget_sparsity)]

    print("[3/5] dense model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2",
    ).eval()

    if out_json.exists():
        result = json.loads(out_json.read_text())
        result.setdefault("config", vars(args))
        result.setdefault("groups", group_meta)
        result.setdefault("rows", [])
        result.setdefault("S", {})
        result["resume_config"] = vars(args)
        print(f"  resume from {out_json}: {len(result['rows'])} existing rows", flush=True)
    else:
        result = {"config": vars(args), "groups": group_meta, "rows": [], "S": {}}
        out_json.write_text(json.dumps(result, indent=2, default=str))

    completed = {
        (int(r["batch_size"]), int(r["group_id"]), int(r["step"]))
        for r in result["rows"]
        if r.get("record_type") == "group_step"
    }

    print("[4/5] per-step batch sweep", flush=True)
    for bs in batch_sizes:
        examples, groups = samples[bs]
        for gid, idx in enumerate(groups):
            if all((int(bs), int(gid), int(step)) in completed for step in steps):
                print(f"  B={bs} g{gid}: skip complete", flush=True)
                continue
            ids, am, starts = _build(examples, idx, int(tok.pad_token_id), args.decode, args.device)
            print(f"  B={bs} g{gid}: collect decode stats", flush=True)
            decode_stats = _collect_stats(model, ids, am, starts, args.decode, args.block_k, args.device)
            for step in steps:
                if (int(bs), int(gid), int(step)) in completed:
                    print(f"    B={bs} g{gid} step={step:2d}: skip complete", flush=True)
                    continue
                budget_nll = _teal_step_budget(model, ids, am, starts, step, thresholds, args.device)
                alloc, dense_nll, final_nll = _round_robin_step(
                    model, ids, am, starts, step, decode_stats, args.block_k, args.device, grid, budget_nll)
                mat = np.zeros((len(PROJS), NUM_LAYERS), dtype=np.float64)
                for pi, proj in enumerate(PROJS):
                    for li in range(NUM_LAYERS):
                        mat[pi, li] = float(alloc[(li, proj)])
                key = f"B{bs}_g{gid}_step{step}"
                result["S"][key] = mat.tolist()
                row = {
                    "record_type": "group_step",
                    "batch_size": int(bs),
                    "group_id": int(gid),
                    "step": int(step),
                    "budget_sparsity": float(args.budget_sparsity),
                    "dense_nll": float(dense_nll),
                    "budget_nll": float(budget_nll),
                    "final_nll": float(final_nll),
                    "dense_ppl": float(math.exp(dense_nll)),
                    "budget_ppl": float(math.exp(budget_nll)),
                    "final_ppl": float(math.exp(final_nll)),
                    "delta_nll_vs_budget": float(final_nll - budget_nll),
                    "mean_sparsity": float(mat.mean()),
                }
                result["rows"].append(row)
                completed.add((int(bs), int(gid), int(step)))
                out_json.write_text(json.dumps(result, indent=2, default=str))
                _plot_batch_sparsity(result["rows"], args.out_dir)
                _plot_sparsity_speedup(result["rows"], args.speedup_json, args.out_dir)
                print(
                    f"    B={bs} g{gid} step={step:2d}: meanS={row['mean_sparsity']:.3f} "
                    f"dense={dense_nll:.3f} budget={budget_nll:.3f} final={final_nll:.3f}",
                    flush=True,
                )
            del ids, am, starts, decode_stats
            gc.collect()
            torch.cuda.empty_cache()

    print("[5/5] done:", out_json, flush=True)


if __name__ == "__main__":
    main()
