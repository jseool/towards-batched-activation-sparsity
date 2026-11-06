#!/usr/bin/env python3
"""Per-step oracle sweep over TEAL budget sparsity.

This is the step-aligned version of a quality-budget sensitivity experiment:
for each length-sorted Alpaca group and decode step, both the TEAL budget and
the round-robin block-sparsity oracle are evaluated on the same single-token
prediction.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
HIST_PATH = os.environ.get("TEAL_HIST_PATH", "histograms/Llama-2-7B")
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NUM_LAYERS = 32


def _prompt(row) -> str:
    inst = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{inst}\n\n### Response:\n"
    )


def _load_alpaca_examples(tokenizer, num_samples: int, seed: int, decode: int, max_prompt_len: int):
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=CACHE_DIR)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    rows = []
    for idx in order:
        row = ds[int(idx)]
        out = (row.get("output") or "").strip()
        if not out:
            continue
        prompt_ids = tokenizer(_prompt(row), add_special_tokens=True).input_ids
        if len(prompt_ids) > max_prompt_len:
            continue
        response_ids = tokenizer(out, add_special_tokens=False).input_ids
        if len(response_ids) < decode:
            continue
        rows.append({
            "dataset_idx": int(idx),
            "prompt_ids": prompt_ids,
            "response_ids": response_ids[:decode],
            "prompt_len": int(len(prompt_ids)),
        })
        if len(rows) >= num_samples:
            break
    if len(rows) < num_samples:
        raise RuntimeError(f"usable Alpaca rows {len(rows)} < requested {num_samples}")
    return rows


def _length_groups(examples: List[dict], batch_size: int, num_groups: int) -> List[List[int]]:
    order = sorted(range(len(examples)), key=lambda i: examples[i]["prompt_len"])
    groups = [order[i * batch_size:(i + 1) * batch_size] for i in range(num_groups)]
    if any(len(g) != batch_size for g in groups):
        raise RuntimeError("not enough examples for requested groups")
    return groups


def _build(examples: List[dict], indices: Iterable[int], pad_id: int, decode: int, device: str):
    indices = list(indices)
    bsz = len(indices)
    max_prompt = max(int(examples[i]["prompt_len"]) for i in indices)
    seq_len = max_prompt + decode
    ids = torch.full((bsz, seq_len), int(pad_id), dtype=torch.long)
    am = torch.zeros((bsz, seq_len), dtype=torch.long)
    starts = torch.zeros(bsz, dtype=torch.long)
    for bi, ei in enumerate(indices):
        ex = examples[ei]
        p = torch.tensor(ex["prompt_ids"], dtype=torch.long)
        r = torch.tensor(ex["response_ids"][:decode], dtype=torch.long)
        ids[bi, :p.numel()] = p
        ids[bi, p.numel():p.numel() + r.numel()] = r
        am[bi, :p.numel() + r.numel()] = 1
        starts[bi] = int(p.numel())
    return ids.to(device), am.to(device), starts.to(device)


def _get_module(model, li: int, proj: str):
    layer = model.model.layers[li]
    if proj in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, proj)
    return getattr(layer.mlp, proj)


def _collect_stats(model, ids, am, starts, decode: int, block_k: int, device: str):
    stats = {}
    handles = []
    bsz = ids.shape[0]
    ib = torch.arange(bsz, device=device)[:, None]
    it = starts[:, None] + torch.arange(decode, device=device)[None, :]
    for li in range(NUM_LAYERS):
        for proj in PROJS:
            mod = _get_module(model, li, proj)
            key = (li, proj)

            def _hook(_module, inputs, k=key):
                x = inputs[0].detach()
                if x.dim() == 2:
                    x = x.reshape(bsz, -1, x.shape[-1])
                d = x.shape[-1]
                nb = d // block_k
                deff = nb * block_k
                xd = x[ib, it][:, :, :deff]
                bn = xd.reshape(bsz, decode, nb, block_k).float().norm(dim=3)
                stats[k] = bn.mean(dim=0).cpu().numpy()

            handles.append(mod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        _ = model(input_ids=ids, attention_mask=am, use_cache=False)
    for h in handles:
        h.remove()
    return stats


def _make_mask(li: int, proj: str, sp: float, step: int, decode_stats, block_k: int, d_in: int, device: str):
    score = decode_stats[(li, proj)][step]
    nb = int(score.shape[0])
    k_keep = max(1, min(nb, int(round((1.0 - float(sp)) * nb))))
    keep = np.argpartition(score, nb - k_keep)[nb - k_keep:]
    mask = torch.zeros(nb * block_k, device=device)
    for b in keep:
        mask[int(b) * block_k:(int(b) + 1) * block_k] = 1.0
    if mask.numel() < d_in:
        mask = torch.cat([mask, torch.ones(d_in - mask.numel(), device=device)])
    return mask


def _eval_step_nll(model, ids, am, starts, step: int, alloc, decode_stats, block_k: int, device: str):
    bsz = ids.shape[0]
    pos = starts + int(step)
    handles = []
    for (li, proj), sp in alloc.items():
        if float(sp) <= 0.0:
            continue
        mod = _get_module(model, li, proj)
        d_in = int(mod.weight.shape[1])
        mask = _make_mask(li, proj, float(sp), step, decode_stats, block_k, d_in, device)

        def _hook(_module, inputs, _mask=mask, _pos=pos):
            x = inputs[0]
            orig = x.shape
            if x.dim() == 2:
                x = x.reshape(bsz, -1, x.shape[-1])
            out = x.clone()
            bi = torch.arange(bsz, device=x.device)
            out[bi, _pos, :] = out[bi, _pos, :] * _mask.to(x.dtype)
            return (out.reshape(orig),) + inputs[1:]

        handles.append(mod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
    for h in handles:
        h.remove()
    bi = torch.arange(bsz, device=device)
    return float(F.cross_entropy(logits[bi, pos, :].float(), ids[bi, pos + 1], reduction="mean").item())


def _extract_thresholds(teal_model):
    thresholds = {}
    for li in range(NUM_LAYERS):
        attn = teal_model.model.layers[li].self_attn
        mlp = teal_model.model.layers[li].mlp
        for proj, key in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")]:
            thresholds[(li, proj)] = float(attn.sparse_fns[key].get_threshold())
        for proj, key in [("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")]:
            thresholds[(li, proj)] = float(mlp.sparse_fns[key].get_threshold())
    return thresholds


def _load_thresholds_for_budgets(budget_sparsities: List[float], device: str):
    teal_root = os.environ.get("TEAL_ROOT")
    if not teal_root:
        raise RuntimeError("set TEAL_ROOT to a TEAL checkout before running the oracle")
    sys.path.insert(0, teal_root)
    import utils.utils as teal_utils

    teal = teal_utils.get_sparse_model(MODEL_NAME, device=device, histogram_path=HIST_PATH, grab_acts=False)
    teal.eval()
    out = {}
    for sp in budget_sparsities:
        teal.set_uniform_sparsity(float(sp))
        out[float(sp)] = _extract_thresholds(teal)
    del teal
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _teal_step_budget(model, ids, am, starts, step: int, thresholds, device: str):
    bsz = ids.shape[0]
    pos = starts + int(step)
    handles = []
    for (li, proj), thr in thresholds.items():
        mod = _get_module(model, li, proj)

        def _hook(_module, inputs, _thr=thr, _pos=pos):
            x = inputs[0]
            orig = x.shape
            if x.dim() == 2:
                x = x.reshape(bsz, -1, x.shape[-1])
            out = x.clone()
            bi = torch.arange(bsz, device=x.device)
            sel = out[bi, _pos, :]
            out[bi, _pos, :] = sel * (sel.abs() > _thr).to(x.dtype)
            return (out.reshape(orig),) + inputs[1:]

        handles.append(mod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
    for h in handles:
        h.remove()
    bi = torch.arange(bsz, device=device)
    return float(F.cross_entropy(logits[bi, pos, :].float(), ids[bi, pos + 1], reduction="mean").item())


def _round_robin_step(model, ids, am, starts, step: int, decode_stats, block_k: int, device: str, grid, budget: float):
    keys = [(li, proj) for li in range(NUM_LAYERS) for proj in PROJS]
    dense_nll = _eval_step_nll(model, ids, am, starts, step, {k: 0.0 for k in keys}, decode_stats, block_k, device)
    alloc = {k: 0.0 for k in keys}
    running_nll = dense_nll
    for level in grid:
        accepted = 0
        for k in keys:
            if alloc[k] >= float(level):
                continue
            cand = {**alloc, k: float(level)}
            nll = _eval_step_nll(model, ids, am, starts, step, cand, decode_stats, block_k, device)
            if nll <= budget + 1.0e-9:
                alloc[k] = float(level)
                running_nll = nll
                accepted += 1
        print(f"      step={step:2d} level={level:.1f} accepted={accepted}", flush=True)
    return alloc, dense_nll, running_nll


def _plot_summary(rows: List[dict], out_dir: Path):
    if not rows:
        return
    budgets = sorted({float(r["budget_sparsity"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    means = []
    for sp in budgets:
        vals = [float(r["mean_sparsity"]) for r in rows if float(r["budget_sparsity"]) == sp]
        means.append(float(np.mean(vals)))
        ax.scatter([sp] * len(vals), vals, color="#1f77b4", alpha=0.5, s=24)
    ax.plot(budgets, means, "-o", color="#d62728", label="Mean over groups/steps")
    ax.set_xlabel("TEAL budget sparsity")
    ax.set_ylabel("Oracle mean block sparsity")
    ax.set_title("Per-step oracle sparsity vs. TEAL quality budget")
    ax.set_ylim(0.0, max(0.9, max(means) * 1.08))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "budget_sweep_mean_sparsity.png", dpi=150)
    fig.savefig(out_dir / "budget_sweep_mean_sparsity.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_groups", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--steps", default="0,1,2,4,8,12,16,20,24,28,30")
    ap.add_argument("--block_k", type=int, default=64)
    ap.add_argument("--budget_sparsities", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "perstep_budget_sweep.json"

    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    budget_sparsities = [float(x) for x in args.budget_sparsities.split(",") if x.strip()]
    grid = [float(x) for x in args.grid.split(",") if x.strip()]

    print("[1/5] tokenizer + Alpaca length groups", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    examples = _load_alpaca_examples(
        tok,
        num_samples=args.batch_size * args.num_groups,
        seed=args.seed,
        decode=args.decode,
        max_prompt_len=args.max_prompt_len,
    )
    groups = _length_groups(examples, args.batch_size, args.num_groups)
    group_meta = []
    for gid, idx in enumerate(groups):
        lens = [examples[i]["prompt_len"] for i in idx]
        group_meta.append({
            "group_id": gid,
            "group_indices": [int(i) for i in idx],
            "dataset_indices": [int(examples[i]["dataset_idx"]) for i in idx],
            "prompt_len_min": int(np.min(lens)),
            "prompt_len_mean": float(np.mean(lens)),
            "prompt_len_max": int(np.max(lens)),
        })
        print(f"  group {gid}: prompt_len min/mean/max={min(lens)}/{np.mean(lens):.1f}/{max(lens)}", flush=True)

    print("[2/5] TEAL thresholds for budget sparsities", flush=True)
    thresholds_by_budget = _load_thresholds_for_budgets(budget_sparsities, args.device)

    print("[3/5] dense model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2",
    ).eval()

    result = {
        "config": vars(args),
        "groups": group_meta,
        "rows": [],
        "S": {},
    }
    out_json.write_text(json.dumps(result, indent=2, default=str))

    print("[4/5] per-group stats + per-step budget sweep", flush=True)
    for gid, idx in enumerate(groups):
        ids, am, starts = _build(examples, idx, int(tok.pad_token_id), args.decode, args.device)
        print(f"  group {gid}: collect decode stats", flush=True)
        decode_stats = _collect_stats(model, ids, am, starts, args.decode, args.block_k, args.device)
        for bsp in budget_sparsities:
            print(f"  group {gid}: budget_sparsity={bsp:.1f}", flush=True)
            for step in steps:
                budget_nll = _teal_step_budget(model, ids, am, starts, step, thresholds_by_budget[bsp], args.device)
                alloc, dense_nll, final_nll = _round_robin_step(
                    model, ids, am, starts, step, decode_stats, args.block_k, args.device, grid, budget_nll)
                mat = np.zeros((len(PROJS), NUM_LAYERS), dtype=np.float64)
                for pi, proj in enumerate(PROJS):
                    for li in range(NUM_LAYERS):
                        mat[pi, li] = float(alloc[(li, proj)])
                key = f"bsp{bsp:.1f}_g{gid}_step{step}"
                result["S"][key] = mat.tolist()
                row = {
                    "budget_sparsity": float(bsp),
                    "group_id": int(gid),
                    "step": int(step),
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
                out_json.write_text(json.dumps(result, indent=2, default=str))
                _plot_summary(result["rows"], args.out_dir)
                print(
                    f"    g{gid} bsp={bsp:.1f} step={step:2d}: "
                    f"meanS={row['mean_sparsity']:.3f} dense={dense_nll:.3f} "
                    f"budget={budget_nll:.3f} final={final_nll:.3f}",
                    flush=True,
                )

    print("[5/5] done:", out_json, flush=True)


if __name__ == "__main__":
    main()
