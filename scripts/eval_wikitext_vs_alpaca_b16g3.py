#!/usr/bin/env python3
"""Compare WikiText and Alpaca using a matched batch-16 length profile."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_perstep_budget_sweep import (  # noqa: E402
    MODEL_NAME, CACHE_DIR, NUM_LAYERS, PROJS,
    _collect_stats, _teal_step_budget, _round_robin_step,
    _load_thresholds_for_budgets,
)

# Alpaca g3 per-sample prompt lengths (= starts), sorted as built.
G3_PROMPT_LENS = [54, 54, 55, 56, 60, 61, 66, 66, 67, 69, 69, 70, 71, 71, 74, 74]
ALPACA_G3 = {"dense_nll": 1.1527, "teal_nll": 1.2269, "oracle_nll": 1.2155,
             "oracle_meanS": 0.642, "polar_nll": 1.2017}


def _build_wikitext_batch(tok, prompt_lens, decode, device, seed):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=CACHE_DIR)
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids_all = tok(text, add_special_tokens=False).input_ids
    bos = tok.bos_token_id
    rng = np.random.default_rng(seed)
    # pick non-overlapping windows large enough for prompt_len-1 (+bos) + decode
    bsz = len(prompt_lens)
    seq_len = max(prompt_lens) + decode
    pad_id = int(tok.pad_token_id)
    ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
    am = torch.zeros((bsz, seq_len), dtype=torch.long)
    starts = torch.zeros(bsz, dtype=torch.long)
    cursor = int(rng.integers(0, 1000))
    for bi, L in enumerate(prompt_lens):
        need = (L - 1) + decode  # we prepend BOS to mimic add_special_tokens
        chunk = ids_all[cursor:cursor + need]
        cursor += need + int(rng.integers(50, 200))  # gap, distinct contexts
        toks = [bos] + list(chunk)  # length L + decode
        p = torch.tensor(toks[:L], dtype=torch.long)
        r = torch.tensor(toks[L:L + decode], dtype=torch.long)
        ids[bi, :L] = p
        ids[bi, L:L + r.numel()] = r
        am[bi, :L + r.numel()] = 1
        starts[bi] = L
    return ids.to(device), am.to(device), starts.to(device)


def _polar_step_nll(model, ids, am, starts, step, k_keep, device):
    bsz = ids.shape[0]
    pos = starts + int(step)
    nh = model.config.num_attention_heads
    hd = model.config.hidden_size // nh
    handles = []
    for li in range(NUM_LAYERS):
        o_proj = model.model.layers[li].self_attn.o_proj

        def _hook(_m, inputs, _pos=pos):
            x = inputs[0]
            orig = x.shape
            if x.dim() == 2:
                x = x.reshape(bsz, -1, x.shape[-1])
            out = x.clone()
            bi = torch.arange(bsz, device=x.device)
            sel = out[bi, _pos, :].view(bsz, nh, hd)
            norms = sel.float().norm(dim=-1)
            _, topk = norms.topk(k_keep, dim=-1)
            mask = torch.zeros_like(norms, dtype=torch.bool)
            mask.scatter_(-1, topk, True)
            sel = sel * mask.unsqueeze(-1).to(sel.dtype)
            out[bi, _pos, :] = sel.view(bsz, nh * hd)
            return (out.reshape(orig),) + inputs[1:]
        handles.append(o_proj.register_forward_pre_hook(_hook))
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
    for h in handles:
        h.remove()
    bi = torch.arange(bsz, device=device)
    return float(F.cross_entropy(logits[bi, pos, :].float(), ids[bi, pos + 1], reduction="mean").item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--steps", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--block_k", type=int, default=64)
    ap.add_argument("--budget_sparsity", type=float, default=0.5)
    ap.add_argument("--grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--head_density", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    grid = [float(x) for x in args.grid.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids, am, starts = _build_wikitext_batch(tok, G3_PROMPT_LENS, args.decode, args.device, args.seed)
    print(f"wikitext batch: ids={tuple(ids.shape)} starts={starts.tolist()}", flush=True)

    thr = _load_thresholds_for_budgets([float(args.budget_sparsity)], args.device)[float(args.budget_sparsity)]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        cache_dir=CACHE_DIR, attn_implementation="flash_attention_2").eval()
    nh = model.config.num_attention_heads
    k_keep = int(round(args.head_density * nh))

    decode_stats = _collect_stats(model, ids, am, starts, args.decode, args.block_k, args.device)

    rows = []
    for step in steps:
        budget_nll = _teal_step_budget(model, ids, am, starts, step, thr, args.device)
        alloc, dense_nll, final_nll = _round_robin_step(
            model, ids, am, starts, step, decode_stats, args.block_k, args.device, grid, budget_nll)
        meanS = float(np.mean([alloc[(li, p)] for li in range(NUM_LAYERS) for p in PROJS]))
        polar_nll = _polar_step_nll(model, ids, am, starts, step, k_keep, args.device)
        row = {"step": step, "dense_nll": dense_nll, "teal_nll": budget_nll,
               "oracle_nll": final_nll, "oracle_meanS": meanS, "polar_nll": polar_nll}
        rows.append(row)
        print(f"  step={step:2d} dense={dense_nll:.4f} teal={budget_nll:.4f} "
              f"oracle={final_nll:.4f} meanS={meanS:.3f} polar={polar_nll:.4f}", flush=True)

    def m(k):
        return statistics.mean(r[k] for r in rows)

    wiki = {c: m(f"{c}_nll") for c in ["dense", "teal", "oracle", "polar"]}
    wiki["oracle_meanS"] = m("oracle_meanS")
    out = {"config": vars(args), "rows": rows, "wikitext_summary": wiki, "alpaca_g3": ALPACA_G3}
    (args.out_dir / "wikitext_vs_alpaca_b16g3.json").write_text(json.dumps(out, indent=2, default=str))

    print("\n=== WikiText vs Alpaca g3 (B16, matched lengths, 11 steps) ===")
    print("  cond     |  wikitext NLL / PPL   |  alpaca NLL / PPL")
    for c in ["dense", "teal", "oracle", "polar"]:
        wn = wiki[c]
        an = ALPACA_G3[f"{c}_nll"]
        print("  %-8s |  %.4f / %.4f      |  %.4f / %.4f" % (c, wn, math.exp(wn), an, math.exp(an)))
    print("  oracle mean sparsity:  wikitext=%.3f   alpaca=%.3f" % (wiki["oracle_meanS"], ALPACA_G3["oracle_meanS"]))


if __name__ == "__main__":
    main()
