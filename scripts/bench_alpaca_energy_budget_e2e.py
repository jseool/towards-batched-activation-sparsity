#!/usr/bin/env python3
"""Energy-budget online top-k: fully-adaptive block sparsity, no sparsity target.

Same setting as bench_alpaca_uniform_online_topk_e2e.py (Alpaca, B=16, group3,
full-prompt prefill with padding/attention mask, 32-token decode, decode-phase
timing, no CUDA graph, HF forward path).

Instead of keeping a fixed *fraction* of K-blocks per linear (uniform density),
each linear keeps, at every decode step, only as many top-energy blocks as are
needed to retain a fixed fraction X of that step's activation energy:

  1. block energy e[b] = ||x_block||^2   (mean over batch tokens)
  2. sort descending, cumulative sum
  3. keep the smallest k blocks whose cumulative energy >= X * total energy

No per-layer target, no global sparsity target: per-cell and overall sparsity
emerge from the data.  The only knob is the scalar energy budget X.

Deployment uses the real block-sparse kernel (same path as the other practical
benchmarks): the energy-budget selection picks the active K-blocks, the realised
per-cell density picks the kernel config, and the sparse GEMM runs.  We report
absolute PPL (dense and sparse), tps speedup over dense, and -- when --profile is
set -- a wall-time breakdown of energy-selection vs sparse-GEMM so the slow part
is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity / debugging)
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from bench_roundrobin_e2e_decode import (  # noqa: E402
    INDEX_BLOCK_META_SIZE,
    PROJ_TO_SHAPE,
    _get_module,
    _get_prepacked_weight_block_major,
    _nearest_cfg,
    _restore,
    _sparse_linear,
)
from bench_alpaca_uniform_online_topk_e2e import (  # noqa: E402
    _build_tensors,
    _decode_measure_alpaca,
    _load_alpaca_batch,
    _load_group_indices,
    _summarize,
)
from eval_perstep_budget_sweep import CACHE_DIR, MODEL_NAME, NUM_LAYERS, PROJS  # noqa: E402


def _block_energy(x3: torch.Tensor, block_k: int, total_k: int) -> torch.Tensor:
    """Mean over batch tokens of per-block squared L2 energy. Shape [total_k]."""
    d_eff = total_k * block_k
    return x3[..., :d_eff].float().reshape(-1, total_k, block_k).pow(2).sum(dim=2).mean(dim=0)


def _energy_budget_active(energy: torch.Tensor, energy_keep: float, min_keep: int = 1) -> torch.Tensor:
    total_k = int(energy.numel())
    order = torch.argsort(energy, descending=True)
    cum = torch.cumsum(energy[order], dim=0)
    thresh = float(energy_keep) * float(cum[-1].item())
    k = int(torch.searchsorted(cum, torch.tensor(thresh, device=cum.device)).item()) + 1
    k = max(int(min_keep), min(total_k, k))
    return order[:k].sort()[0].to(torch.int32)


def _topk_active(energy: torch.Tensor, density: float, min_keep: int = 1) -> torch.Tensor:
    total_k = int(energy.numel())
    k = max(int(min_keep), min(total_k, int(round(float(density) * total_k))))
    return torch.topk(energy, k=k).indices.sort()[0].to(torch.int32)


def _patch_adaptive(
    model,
    sweep,
    block_k: int,
    mode: str,
    value: float,
    profile: bool,
    stats: dict,
    min_sparse_sparsity: float,
):
    """mode='energy' -> keep X=value energy fraction; mode='uniform' -> density=1-value."""
    originals = []
    for li in range(NUM_LAYERS):
        for proj in PROJS:
            key = (li, proj)
            stats[key] = {"kept": 0, "total_k": 0}
            mod = _get_module(model, li, proj)
            shape = PROJ_TO_SHAPE[proj]
            total_k = int(mod.weight.shape[1]) // block_k
            prefix = f"eb_l{li}_{proj}"
            _get_prepacked_weight_block_major(mod.weight, block_k, cache_prefix=prefix)
            bitmap_buf = torch.empty(total_k, dtype=torch.uint8, device=mod.weight.device)
            orig_forward = mod.forward

            def forward(x, _mod=mod, _orig=orig_forward, _shape=shape, _prefix=prefix, _buf=bitmap_buf, _tk=total_k, _key=key):
                x3 = x.reshape(x.shape[0], 1, x.shape[-1]) if x.dim() == 2 else x
                if x3.shape[1] != 1:  # prefill stays dense, only decode is sparsified
                    return _orig(x)
                if profile:
                    torch.cuda.synchronize()
                    t_sel = time.perf_counter()
                energy = _block_energy(x3, block_k, _tk)
                if mode == "energy":
                    active = _energy_budget_active(energy, value)
                else:
                    active = _topk_active(energy, 1.0 - value)
                if profile:
                    torch.cuda.synchronize()
                    stats["select_wall_s"] = stats.get("select_wall_s", 0.0) + (time.perf_counter() - t_sel)
                density = float(active.numel()) / float(_tk)
                candidate_sparsity = 1.0 - density
                stats[_key]["candidate_kept"] = stats[_key].get("candidate_kept", 0) + int(active.numel())
                stats[_key]["candidate_total_k"] = stats[_key].get("candidate_total_k", 0) + _tk
                if candidate_sparsity < float(min_sparse_sparsity):
                    stats["dense_fallback_calls"] = stats.get("dense_fallback_calls", 0) + 1
                    stats[_key]["kept"] += _tk
                    stats[_key]["total_k"] += _tk
                    return _orig(x)
                stats[_key]["kept"] += int(active.numel())
                stats[_key]["total_k"] += _tk
                cfg = _nearest_cfg(sweep, _shape, density)
                return _sparse_linear(
                    x, _mod.weight, density, cfg, _prefix, block_k, stats, profile, _buf,
                    None, 1.0, 0.1, active,
                )

            mod.forward = forward
            originals.append((mod, orig_forward))
    return originals


def _sparsity_report(stats: dict) -> dict:
    per_cell = {}
    sp_list = []
    candidate_sp_list = []
    for k, s in stats.items():
        if not isinstance(k, tuple):
            continue
        if s["total_k"] <= 0:
            continue
        density = s["kept"] / s["total_k"]
        per_cell[f"{k[0]}.{k[1]}"] = float(density)
        sp_list.append(1.0 - density)
        if int(s.get("candidate_total_k", 0)) > 0:
            cand_density = s["candidate_kept"] / s["candidate_total_k"]
            candidate_sp_list.append(1.0 - cand_density)
    return {
        "mean_sparsity": float(sum(sp_list) / len(sp_list)) if sp_list else 0.0,
        "min_sparsity": float(min(sp_list)) if sp_list else 0.0,
        "max_sparsity": float(max(sp_list)) if sp_list else 0.0,
        "candidate_mean_sparsity": float(sum(candidate_sp_list) / len(candidate_sp_list)) if candidate_sp_list else 0.0,
        "per_cell_density": per_cell,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--perstep_json", type=Path, default=ROOT_DIR / "results/oracle/perstep_batch_sweep.json")
    ap.add_argument("--sweep_json", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--group_id", type=int, default=3)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--energy_keeps", default="0.99,0.97,0.95,0.90,0.85,0.80")
    ap.add_argument("--uniform_sparsities", default="0.3,0.5,0.7")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--profile", action="store_true", help="extra pass timing energy-selection vs sparse-GEMM")
    ap.add_argument("--min_sparse_sparsity", type=float, default=0.0, help="fallback to dense when the selected blocks imply less sparsity than this")
    ap.add_argument("--energy_stop_ppl_ratio", type=float, default=0.0, help="stop energy sweep after first point whose sparse PPL reaches this multiple of dense PPL")
    ap.add_argument("--energy_stop_ppl_delta", type=float, default=0.0, help="stop energy sweep after first point whose sparse PPL is at least dense PPL plus this value")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dataset_indices = _load_group_indices(args.perstep_json, args.batch_size, args.group_id)
    rows = _load_alpaca_batch(tok, dataset_indices, args.decode)
    prompt_ids, prompt_am, last_pos, resp, max_prompt = _build_tensors(rows, int(tok.pad_token_id), args.decode, args.device)
    sweep = json.loads(args.sweep_json.read_text())

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2",
    ).eval()

    print("[1/3] dense Alpaca decode", flush=True)
    dense_records = _decode_measure_alpaca(model, prompt_ids, prompt_am, last_pos, resp, args.repeats, args.warmup)
    dense_summary = _summarize(dense_records)
    print(f"dense ppl={dense_summary['ppl']:.4f} tps={dense_summary['decode_tps']:.1f} elapsed={dense_summary['decode_elapsed_s']:.4f}s", flush=True)

    candidates = []
    for x in [float(v) for v in args.energy_keeps.split(",") if v.strip()]:
        candidates.append(("energy", f"energy_budget_X{x:.2f}", x))
    for s in [float(v) for v in args.uniform_sparsities.split(",") if v.strip()]:
        candidates.append(("uniform", f"uniform_s{s:.2f}", s))

    methods = []
    stop_energy = False
    for mode, name, value in candidates:
        if mode == "energy" and stop_energy:
            continue
        print(f"[2/3] {name}", flush=True)
        # clean speed + quality (no profiling sync overhead)
        stats: dict = {"calls": 0, "wall_s": 0.0, "select_wall_s": 0.0, "kernel_wall_s": 0.0}
        originals = _patch_adaptive(model, sweep, block_k, mode, value, False, stats, args.min_sparse_sparsity)
        rec = _decode_measure_alpaca(model, prompt_ids, prompt_am, last_pos, resp, args.repeats, args.warmup)
        _restore(originals)
        summ = _summarize(rec)
        sp = _sparsity_report(stats)

        breakdown = None
        if args.profile:
            pstats: dict = {"calls": 0, "wall_s": 0.0, "select_wall_s": 0.0, "kernel_wall_s": 0.0}
            poriginals = _patch_adaptive(model, sweep, block_k, mode, value, True, pstats, args.min_sparse_sparsity)
            prec = _decode_measure_alpaca(model, prompt_ids, prompt_am, last_pos, resp, 1, 0)
            _restore(poriginals)
            calls = max(1, int(pstats.get("calls", 0)))
            breakdown = {
                "profiled_decode_elapsed_s": float(prec[0]["decode_elapsed_s"]),
                "select_wall_s": float(pstats.get("select_wall_s", 0.0)),
                "kernel_wall_s": float(pstats.get("kernel_wall_s", 0.0)),
                "sparse_linear_calls": int(calls),
                "avg_select_us_per_call": float(pstats.get("select_wall_s", 0.0) / calls * 1e6),
                "avg_kernel_us_per_call": float(pstats.get("kernel_wall_s", 0.0) / calls * 1e6),
            }

        row = {
            "method": name,
            "mode": mode,
            "knob": float(value),
            "mean_sparsity": float(sp["mean_sparsity"]),
            "candidate_mean_sparsity": float(sp["candidate_mean_sparsity"]),
            "min_sparsity": float(sp["min_sparsity"]),
            "max_sparsity": float(sp["max_sparsity"]),
            "dense_fallback_calls": int(stats.get("dense_fallback_calls", 0)),
            "dense_ppl": float(dense_summary["ppl"]),
            "sparse_ppl": float(summ["ppl"]),
            "delta_ppl_vs_dense": float(summ["ppl"] - dense_summary["ppl"]),
            "delta_nll_vs_dense": float(summ["nll"] - dense_summary["nll"]),
            "dense_tps": float(dense_summary["decode_tps"]),
            "sparse_tps": float(summ["decode_tps"]),
            "tps_speedup_over_dense": float(summ["decode_tps"] / dense_summary["decode_tps"]),
            "profile_breakdown": breakdown,
            "per_cell_density": sp["per_cell_density"],
        }
        methods.append(row)
        head = {k: row[k] for k in ["method", "mean_sparsity", "dense_ppl", "sparse_ppl", "tps_speedup_over_dense"]}
        if breakdown is not None:
            head["select_us"] = round(breakdown["avg_select_us_per_call"], 1)
            head["kernel_us"] = round(breakdown["avg_kernel_us_per_call"], 1)
        print(json.dumps(head, indent=2), flush=True)
        if (
            mode == "energy"
            and float(args.energy_stop_ppl_ratio) > 0.0
            and float(row["sparse_ppl"]) >= float(args.energy_stop_ppl_ratio) * float(dense_summary["ppl"])
        ):
            stop_energy = True
            print(
                f"[energy-stop] {name}: sparse_ppl={row['sparse_ppl']:.4f} >= "
                f"{args.energy_stop_ppl_ratio:.3f} * dense_ppl={dense_summary['ppl']:.4f}",
                flush=True,
            )
        if (
            mode == "energy"
            and float(args.energy_stop_ppl_delta) > 0.0
            and float(row["sparse_ppl"]) >= float(dense_summary["ppl"]) + float(args.energy_stop_ppl_delta)
        ):
            stop_energy = True
            print(
                f"[energy-stop] {name}: sparse_ppl={row['sparse_ppl']:.4f} >= "
                f"dense_ppl={dense_summary['ppl']:.4f} + {args.energy_stop_ppl_delta:.3f}",
                flush=True,
            )

    out = {
        "config": {**vars(args), "perstep_json": str(args.perstep_json), "sweep_json": str(args.sweep_json)},
        "group": {
            "batch_size": int(args.batch_size),
            "group_id": int(args.group_id),
            "dataset_indices": dataset_indices,
            "prompt_len_mean": float(sum(r["prompt_len"] for r in rows) / len(rows)),
            "max_prompt_padded": int(max_prompt),
        },
        "dense_summary": dense_summary,
        "methods": methods,
    }
    out_path = args.out_dir / "alpaca_energy_budget_e2e.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[3/3] saved", out_path, flush=True)


if __name__ == "__main__":
    main()
