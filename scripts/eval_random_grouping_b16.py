#!/usr/bin/env python3
"""Compare random and length-sorted Alpaca groups at batch size 16."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_perstep_budget_sweep import (  # noqa: E402
    MODEL_NAME, CACHE_DIR, NUM_LAYERS, PROJS,
    _collect_stats, _teal_step_budget, _round_robin_step,
    _load_thresholds_for_budgets,
)

# Dataset indices used by the five length-sorted B16 groups.
POOL = [41764, 9396, 15077, 30465, 24657, 51562, 32373, 3034, 11313, 25111, 41068, 3142, 42479, 4834, 44849, 22327,
        5089, 44940, 40720, 13292, 21827, 50198, 1173, 6587, 38212, 37866, 26376, 11622, 44863, 48302, 43939, 32366,
        237, 10317, 34029, 20407, 43652, 17131, 46606, 13128, 3539, 50984, 3956, 49937, 11781, 19089, 9672, 10175,
        28875, 4877, 12812, 23292, 25697, 23507, 35215, 7366, 45219, 43708, 12518, 34520, 38526, 14410, 14475, 26933,
        39630, 24357, 14603, 8696, 45835, 18292, 11244, 20506, 25953, 15634, 17142, 46215, 35155, 6960, 40468, 21489]
LENGTH_GROUP_B16_MEAN = 0.638  # mean oracle sparsity across g0..g4 at B16


def _prompt(row) -> str:
    inst = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    if inp:
        return ("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n")
    return ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{inst}\n\n### Response:\n")


def _build_batch(tok, ds, indices, decode, device):
    exs = []
    for idx in indices:
        row = ds[int(idx)]
        p = tok(_prompt(row), add_special_tokens=True).input_ids
        r = tok((row.get("output") or "").strip(), add_special_tokens=False).input_ids
        exs.append({"prompt_ids": p, "response_ids": r[:decode], "prompt_len": len(p)})
    bsz = len(exs)
    seq_len = max(e["prompt_len"] for e in exs) + decode
    pad_id = int(tok.pad_token_id)
    ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
    am = torch.zeros((bsz, seq_len), dtype=torch.long)
    starts = torch.zeros(bsz, dtype=torch.long)
    for bi, e in enumerate(exs):
        p = torch.tensor(e["prompt_ids"], dtype=torch.long)
        r = torch.tensor(e["response_ids"], dtype=torch.long)
        ids[bi, :p.numel()] = p
        ids[bi, p.numel():p.numel() + r.numel()] = r
        am[bi, :p.numel() + r.numel()] = 1
        starts[bi] = int(p.numel())
    return ids.to(device), am.to(device), starts.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--steps", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--block_k", type=int, default=64)
    ap.add_argument("--budget_sparsity", type=float, default=0.5)
    ap.add_argument("--grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=CACHE_DIR)
    thr = _load_thresholds_for_budgets([float(args.budget_sparsity)], args.device)[float(args.budget_sparsity)]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        cache_dir=CACHE_DIR, attn_implementation="flash_attention_2").eval()

    seed_results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(POOL, size=16, replace=False).tolist())
        ids, am, starts = _build_batch(tok, ds, idx, args.decode, args.device)
        plens = starts.tolist()
        print(f"\n[seed {seed}] prompt_lens min/mean/max = {min(plens)}/{statistics.mean(plens):.1f}/{max(plens)}", flush=True)
        decode_stats = _collect_stats(model, ids, am, starts, args.decode, args.block_k, args.device)
        step_S = []
        for step in steps:
            budget_nll = _teal_step_budget(model, ids, am, starts, step, thr, args.device)
            alloc, _dn, _fn = _round_robin_step(model, ids, am, starts, step, decode_stats,
                                                args.block_k, args.device, grid, budget_nll)
            meanS = float(np.mean([alloc[(li, p)] for li in range(NUM_LAYERS) for p in PROJS]))
            step_S.append(meanS)
            print(f"  step={step:2d} meanS={meanS:.3f}", flush=True)
        m = statistics.mean(step_S)
        seed_results.append({"seed": seed, "indices": idx, "prompt_lens": plens,
                             "per_step_meanS": step_S, "mean_sparsity": m})
        print(f"[seed {seed}] mean oracle sparsity over {len(steps)} steps = {m:.3f}", flush=True)

    rand_mean = statistics.mean(r["mean_sparsity"] for r in seed_results)
    out = {"config": vars(args), "length_group_b16_mean": LENGTH_GROUP_B16_MEAN,
           "seed_results": seed_results, "random_mean_sparsity": rand_mean}
    (args.out_dir / "random_grouping_b16.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n=== Random vs length grouping (Alpaca B16, oracle mean sparsity) ===")
    for r in seed_results:
        print("  random seed %d: %.3f  (prompt_len %d-%d)" %
              (r["seed"], r["mean_sparsity"], min(r["prompt_lens"]), max(r["prompt_lens"])))
    print("  random mean (%d seeds): %.3f" % (len(seeds), rand_mean))
    print("  length-group B16 mean : %.3f" % LENGTH_GROUP_B16_MEAN)
    print("  difference            : %+.3f" % (rand_mean - LENGTH_GROUP_B16_MEAN))


if __name__ == "__main__":
    main()
