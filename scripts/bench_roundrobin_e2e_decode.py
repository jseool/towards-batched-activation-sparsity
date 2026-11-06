#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

os.environ["TEAL_EXP_ENABLE_SPLIT_K_1D"] = "0"
os.environ["TEAL_EXP_USE_RUN_COMPRESSED_METADATA"] = "0"
os.environ["TEAL_EXP_SORT_ACTIVE_BLOCKS"] = "0"

from kernels.shared_mask_batched_triton_min_runtime import (  # noqa: E402
    INDEX_BLOCK_META_SIZE,
    _build_active_k_bitmap,
    _get_prepacked_weight_block_major,
    _prepacked_block_sparse_gemm_from_active_blocks,
    _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload,
    _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload_splitk,
    _prepacked_block_sparse_gemm_from_active_blocks_splitk,
)

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROJ_TO_SHAPE = {
    "q_proj": "qkvo",
    "k_proj": "qkvo",
    "v_proj": "qkvo",
    "o_proj": "qkvo",
    "gate_proj": "gate_up",
    "up_proj": "gate_up",
    "down_proj": "down",
}


def _load_wikitext_batches(tokenizer, batch_size: int, prefill: int, decode: int, num_batches: int):
    total_len = prefill + decode + 1
    parts = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=CACHE_DIR)
        parts.extend(row["text"].strip() for row in ds if row["text"].strip())
    ids_all = tokenizer(" ".join(parts), return_tensors="pt").input_ids[0]
    chunks = [ids_all[i : i + total_len] for i in range(0, len(ids_all) - total_len + 1, total_len)]
    need = batch_size * num_batches
    if len(chunks) < need:
        raise RuntimeError(f"need {need} chunks, got {len(chunks)}")
    batches = []
    for i in range(0, need, batch_size):
        batches.append(torch.stack(chunks[i : i + batch_size]))
    return batches


def _load_alloc(path: Path, batch_idx: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["batches"]
    row = next((r for r in rows if int(r["batch_idx"]) == int(batch_idx)), rows[0])
    return row["allocation_density"], row


def _nearest_cfg(sweep: dict, shape: str, density: float):
    rows = sweep[shape]
    key = min(rows.keys(), key=lambda k: abs(float(k) - float(density)))
    cfg = dict(rows[key])
    cfg["density_key"] = key
    return cfg


def _get_module(model, layer_idx: int, proj: str):
    layer = model.model.layers[layer_idx]
    if proj in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, proj)
    return getattr(layer.mlp, proj)


def _select_active_blocks(
    x3: torch.Tensor,
    density: float,
    block_k: int,
    block_prior: torch.Tensor | None = None,
    prior_alpha: float = 1.0,
    prior_eps: float = 0.1,
    block_score_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    d_in = x3.shape[-1]
    total_k = d_in // block_k
    d_eff = total_k * block_k
    n_keep = max(1, min(total_k, int(round(float(density) * total_k))))
    xb = x3[..., :d_eff].float().reshape(-1, total_k, block_k)
    if block_score_scale is None:
        score = xb.norm(dim=2).mean(dim=0)
    else:
        scale = block_score_scale.to(device=x3.device, dtype=xb.dtype)[:d_eff].reshape(1, total_k, block_k)
        score = (xb.pow(2) * scale).sum(dim=2).mean(dim=0)
    if block_prior is not None:
        prior = block_prior.to(device=score.device, dtype=score.dtype)
        if int(prior.numel()) == int(total_k):
            score = score * torch.clamp(prior + float(prior_eps), min=1e-6).pow(float(prior_alpha))
    return torch.topk(score, k=n_keep).indices.sort()[0].to(torch.int32)


def _topk_energy_ratio(
    x,
    density: float,
    block_k: int,
    block_prior: torch.Tensor | None = None,
    prior_alpha: float = 1.0,
    prior_eps: float = 0.1,
) -> float:
    x3 = x.reshape(x.shape[0], 1, x.shape[-1]) if x.dim() == 2 else x
    if x3.shape[1] != 1 or density >= 1.0:
        return 1.0
    d_in = x3.shape[-1]
    total_k = d_in // block_k
    d_eff = total_k * block_k
    n_keep = max(1, min(total_k, int(round(float(density) * total_k))))
    energy = x3[..., :d_eff].float().reshape(-1, total_k, block_k).pow(2).sum(dim=2).mean(dim=0)
    score = energy
    if block_prior is not None:
        prior = block_prior.to(device=score.device, dtype=score.dtype)
        if int(prior.numel()) == int(total_k):
            score = score * torch.clamp(prior + float(prior_eps), min=1e-6).pow(float(prior_alpha))
    active = torch.topk(score, k=n_keep).indices
    return float(energy[active].sum().item() / max(float(energy.sum().item()), 1e-12))


def _energy_expanded_active_blocks(
    x,
    density: float,
    block_k: int,
    energy_keep: float,
    block_prior: torch.Tensor | None = None,
    prior_alpha: float = 1.0,
    prior_eps: float = 0.1,
):
    x3 = x.reshape(x.shape[0], 1, x.shape[-1]) if x.dim() == 2 else x
    if x3.shape[1] != 1 or density >= 1.0:
        return None, 1.0, 1.0
    d_in = x3.shape[-1]
    total_k = d_in // block_k
    d_eff = total_k * block_k
    n_keep = max(1, min(total_k, int(round(float(density) * total_k))))
    energy = x3[..., :d_eff].float().reshape(-1, total_k, block_k).pow(2).sum(dim=2).mean(dim=0)
    score = energy
    if block_prior is not None:
        prior = block_prior.to(device=score.device, dtype=score.dtype)
        if int(prior.numel()) == int(total_k):
            score = score * torch.clamp(prior + float(prior_eps), min=1e-6).pow(float(prior_alpha))
    order = torch.argsort(score, descending=True)
    sorted_energy = energy[order]
    total_energy = max(float(energy.sum().item()), 1e-12)
    cum = torch.cumsum(sorted_energy, dim=0)
    base_ratio = float(cum[n_keep - 1].item() / total_energy)
    if base_ratio >= float(energy_keep):
        active = order[:n_keep]
        return active.sort()[0].to(torch.int32), base_ratio, float(n_keep / total_k)
    threshold = float(energy_keep) * total_energy
    expanded_keep = int(torch.searchsorted(cum, torch.tensor(threshold, device=cum.device)).item()) + 1
    expanded_keep = max(n_keep, min(total_k, expanded_keep))
    active = order[:expanded_keep]
    expanded_ratio = float(cum[expanded_keep - 1].item() / total_energy)
    return active.sort()[0].to(torch.int32), expanded_ratio, float(expanded_keep / total_k)


def _sparse_linear(
    x,
    weight,
    density: float,
    cfg: dict,
    cache_prefix: str,
    block_k: int,
    stats: dict,
    profile_sparse_parts: bool,
    bitmap_buf: torch.Tensor | None,
    block_prior: torch.Tensor | None = None,
    prior_alpha: float = 1.0,
    prior_eps: float = 0.1,
    fixed_active_blocks: torch.Tensor | None = None,
    block_score_scale: torch.Tensor | None = None,
):
    orig_shape = x.shape
    if x.dim() == 2:
        x3 = x.reshape(x.shape[0], 1, x.shape[1])
    else:
        x3 = x
    if x3.shape[1] != 1 or density >= 1.0:
        return F.linear(x, weight)

    d_in = int(weight.shape[1])
    d_out = int(weight.shape[0])
    total_k = d_in // block_k
    if profile_sparse_parts:
        torch.cuda.synchronize()
        t_select0 = time.perf_counter()
    if fixed_active_blocks is not None:
        active_blocks = fixed_active_blocks.to(device=x3.device, dtype=torch.int32)
    else:
        active_blocks = _select_active_blocks(x3, density, block_k, block_prior, prior_alpha, prior_eps, block_score_scale)
    if profile_sparse_parts:
        torch.cuda.synchronize()
        stats["select_wall_s"] = stats.get("select_wall_s", 0.0) + (time.perf_counter() - t_select0)
    out = torch.empty(x3.shape[0], x3.shape[1], d_out, device=x3.device, dtype=x3.dtype)

    kern = str(cfg["best_kern"])
    split_k = int(cfg.get("best_K") or 1)
    tile = cfg.get("best_tile") or [None, None, None, None]
    tb, tn, nw, ns = tile
    if profile_sparse_parts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    else:
        t0 = None
    if kern == "naive" and split_k <= 1:
        _prepacked_block_sparse_gemm_from_active_blocks(
            x3, weight, active_blocks, out, cache_prefix=cache_prefix, use_run_compressed=False
        )
    elif kern == "naive":
        _prepacked_block_sparse_gemm_from_active_blocks_splitk(
            x3,
            weight,
            active_blocks,
            out,
            cache_prefix=cache_prefix,
            num_split_k=split_k,
            tile_b=tb,
            tile_n=tn,
            tile_warps=nw,
            tile_stages=ns,
        )
    else:
        if bitmap_buf is None or int(bitmap_buf.numel()) != int(total_k):
            bitmap = _build_active_k_bitmap(active_blocks, total_k, x3.device)
        else:
            bitmap = bitmap_buf
            bitmap.zero_()
            if int(active_blocks.numel()) > 0:
                bitmap[active_blocks.to(torch.long)] = 1
        if split_k <= 1:
            _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload(
                x3,
                weight,
                active_blocks,
                out,
                cache_prefix=cache_prefix,
                total_k_blocks=total_k,
                prebuilt_bitmap=bitmap,
            )
        else:
            _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload_splitk(
                x3,
                weight,
                active_blocks,
                out,
                cache_prefix=cache_prefix,
                total_k_blocks=total_k,
                prebuilt_bitmap=bitmap,
                num_split_k=split_k,
                tile_b=tb,
                tile_n=tn,
                tile_warps=nw,
                tile_stages=ns,
            )
    stats["calls"] += 1
    if profile_sparse_parts:
        torch.cuda.synchronize()
        stats["wall_s"] += time.perf_counter() - float(t0)
        stats["kernel_wall_s"] = stats.get("kernel_wall_s", 0.0) + (time.perf_counter() - float(t0))
    if x.dim() == 2:
        return out.reshape(orig_shape[0], d_out)
    return out


def _patch_roundrobin_sparse(
    model,
    alloc,
    sweep,
    profile_sparse_parts: bool,
    block_priors: dict | None = None,
    prior_alpha: float = 1.0,
    prior_eps: float = 0.1,
    reuse_masks: bool = False,
    energy_gate_keep: float | None = None,
    energy_gate_keep_by_cell: dict[tuple[int, str], float] | None = None,
    energy_gate_mode: str = "dense",
    energy_gate_expand_projs: set[str] | None = None,
    energy_gate_fallback_sparsity_scale: float = 0.5,
    weight_aware_score: bool = False,
    step_density_alloc: dict[tuple[int, str], list[float]] | None = None,
):
    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    stats = {"calls": 0, "wall_s": 0.0}
    cell_energy_stats: dict[str, dict] = {}
    stats["cell_energy_stats"] = cell_energy_stats
    mask_cache: dict[tuple[int, str], torch.Tensor] = {}
    decode_call_counts: dict[tuple[int, str], int] = {}
    originals = []
    for li in range(len(model.model.layers)):
        for proj in PROJS:
            mod = _get_module(model, li, proj)
            density = float(alloc[str(li)][proj])
            schedule = (step_density_alloc or {}).get((li, proj))
            block_score_scale = None
            if weight_aware_score and density < 1.0:
                block_score_scale = mod.weight.detach().float().pow(2).sum(dim=0)
            cfg = _nearest_cfg(sweep, PROJ_TO_SHAPE[proj], density)
            cache_prefix = f"rr_e2e_l{li}_{proj}_{cfg['best_kern']}"
            _get_prepacked_weight_block_major(mod.weight, block_k, cache_prefix=cache_prefix)
            if schedule is not None:
                _get_prepacked_weight_block_major(mod.weight, block_k, cache_prefix=f"rr_e2e_l{li}_{proj}_naive")
                _get_prepacked_weight_block_major(mod.weight, block_k, cache_prefix=f"rr_e2e_l{li}_{proj}_bpl")
            bitmap_buf = None
            if (str(cfg["best_kern"]) == "bpl" and density < 1.0) or schedule is not None:
                total_k = int(mod.weight.shape[1]) // block_k
                bitmap_buf = torch.empty(total_k, dtype=torch.uint8, device=mod.weight.device)
            orig_forward = mod.forward

            def forward(
                x,
                _mod=mod,
                _density=density,
                _cfg=cfg,
                _prefix=cache_prefix,
                _bitmap_buf=bitmap_buf,
                _prior=(block_priors or {}).get((li, proj)),
                _mask_key=(li, proj),
                _proj=proj,
                _orig=orig_forward,
                _block_score_scale=block_score_scale,
                _schedule=schedule,
            ):
                prior = _prior
                active_density = float(_density)
                active_cfg = _cfg
                active_prefix = _prefix
                x3_for_step = x.reshape(x.shape[0], 1, x.shape[-1]) if x.dim() == 2 else x
                if _schedule is not None and x3_for_step.shape[1] == 1:
                    raw_decode_calls = int(decode_call_counts.get(_mask_key, 0))
                    decode_call_counts[_mask_key] = raw_decode_calls + 1
                    # The first response-token loss is produced by the dense prefill
                    # pass. Sparse decode calls therefore start at schedule index 1.
                    # Wrap the counter so warmup/repeat timing runs reuse the same
                    # per-step schedule instead of drifting to the final step value.
                    if len(_schedule) > 1:
                        decode_idx = (raw_decode_calls % (len(_schedule) - 1)) + 1
                    else:
                        decode_idx = 0
                    if decode_idx < len(_schedule):
                        active_density = float(_schedule[decode_idx])
                    else:
                        active_density = float(_schedule[-1])
                    active_cfg = _nearest_cfg(sweep, PROJ_TO_SHAPE[_proj], active_density)
                    active_prefix = f"rr_e2e_l{_mask_key[0]}_{_proj}_{active_cfg['best_kern']}"
                cell_energy_gate_keep = None
                if energy_gate_keep_by_cell is not None and _mask_key in energy_gate_keep_by_cell:
                    cell_energy_gate_keep = float(energy_gate_keep_by_cell[_mask_key])
                elif energy_gate_keep is not None:
                    cell_energy_gate_keep = float(energy_gate_keep)
                if (
                    cell_energy_gate_keep is not None
                    and active_density < 1.0
                    and not (reuse_masks and _mask_key in mask_cache)
                ):
                    ratio = _topk_energy_ratio(x, active_density, block_k, prior, prior_alpha, prior_eps)
                    stats["energy_gate_checks"] = stats.get("energy_gate_checks", 0) + 1
                    stats["energy_gate_ratio_sum"] = stats.get("energy_gate_ratio_sum", 0.0) + float(ratio)
                    cell_key = f"{_mask_key[0]}:{_mask_key[1]}"
                    cell_stat = cell_energy_stats.setdefault(
                        cell_key,
                        {
                            "layer": int(_mask_key[0]),
                            "proj": str(_mask_key[1]),
                            "density": float(active_density),
                            "checks": 0,
                            "ratio_sum": 0.0,
                            "ratio_min": 1.0,
                            "ratio_max": 0.0,
                            "dense_fallbacks": 0,
                            "expands": 0,
                            "scaled_fallbacks": 0,
                        },
                    )
                    cell_stat["checks"] += 1
                    cell_stat["ratio_sum"] += float(ratio)
                    cell_stat["ratio_min"] = min(float(cell_stat["ratio_min"]), float(ratio))
                    cell_stat["ratio_max"] = max(float(cell_stat["ratio_max"]), float(ratio))
                    if ratio < float(cell_energy_gate_keep):
                        use_expand = energy_gate_mode == "expand" or (
                            energy_gate_mode == "hybrid" and energy_gate_expand_projs is not None and _proj in energy_gate_expand_projs
                        )
                        if use_expand:
                            active, expanded_ratio, expanded_density = _energy_expanded_active_blocks(
                                x,
                                active_density,
                                block_k,
                                float(cell_energy_gate_keep),
                                prior,
                                prior_alpha,
                                prior_eps,
                            )
                            stats["energy_gate_expands"] = stats.get("energy_gate_expands", 0) + 1
                            stats["energy_gate_expanded_ratio_sum"] = stats.get("energy_gate_expanded_ratio_sum", 0.0) + float(expanded_ratio)
                            stats["energy_gate_expanded_density_sum"] = stats.get("energy_gate_expanded_density_sum", 0.0) + float(expanded_density)
                            cell_stat["expands"] += 1
                            cfg2 = _nearest_cfg(sweep, PROJ_TO_SHAPE[_proj], float(expanded_density))
                            return _sparse_linear(
                                x,
                                _mod.weight,
                                float(expanded_density),
                                cfg2,
                                f"rr_e2e_l{_mask_key[0]}_{_proj}_{cfg2['best_kern']}",
                                block_k,
                                stats,
                                profile_sparse_parts,
                                _bitmap_buf,
                                None,
                                prior_alpha,
                                prior_eps,
                                active,
                                _block_score_scale,
                            )
                        if energy_gate_mode == "scaled":
                            base_sparsity = max(0.0, min(1.0, 1.0 - float(active_density)))
                            fallback_sparsity = max(0.0, min(base_sparsity, base_sparsity * float(energy_gate_fallback_sparsity_scale)))
                            fallback_density = max(float(active_density), min(1.0, 1.0 - fallback_sparsity))
                            stats["energy_gate_scaled_fallbacks"] = stats.get("energy_gate_scaled_fallbacks", 0) + 1
                            stats["energy_gate_scaled_density_sum"] = stats.get("energy_gate_scaled_density_sum", 0.0) + float(fallback_density)
                            cell_stat["scaled_fallbacks"] += 1
                            cfg2 = _nearest_cfg(sweep, PROJ_TO_SHAPE[_proj], float(fallback_density))
                            return _sparse_linear(
                                x,
                                _mod.weight,
                                float(fallback_density),
                                cfg2,
                                f"rr_e2e_l{_mask_key[0]}_{_proj}_{cfg2['best_kern']}",
                                block_k,
                                stats,
                                profile_sparse_parts,
                                _bitmap_buf,
                                prior,
                                prior_alpha,
                                prior_eps,
                                None,
                                _block_score_scale,
                            )
                        stats["energy_gate_dense_fallbacks"] = stats.get("energy_gate_dense_fallbacks", 0) + 1
                        cell_stat["dense_fallbacks"] += 1
                        return _orig(x)
                if reuse_masks and _mask_key in mask_cache:
                    prior = None
                    # The cached mask is passed through the prior slot with a
                    # sentinel alpha below; _sparse_linear handles it as fixed.
                    fixed_blocks = mask_cache[_mask_key]
                    return _sparse_linear(
                        x,
                        _mod.weight,
                        active_density,
                        active_cfg,
                        active_prefix,
                        block_k,
                        stats,
                        profile_sparse_parts,
                        _bitmap_buf,
                        None,
                        prior_alpha,
                        prior_eps,
                        fixed_blocks,
                        _block_score_scale,
                    )
                out = _sparse_linear(
                    x,
                    _mod.weight,
                    active_density,
                    active_cfg,
                    active_prefix,
                    block_k,
                    stats,
                    profile_sparse_parts,
                    _bitmap_buf,
                    prior,
                    prior_alpha,
                    prior_eps,
                    None,
                    _block_score_scale,
                )
                if reuse_masks and active_density < 1.0 and _mask_key not in mask_cache:
                    x3 = x.reshape(x.shape[0], 1, x.shape[-1]) if x.dim() == 2 else x
                    if x3.shape[1] == 1:
                        mask_cache[_mask_key] = _select_active_blocks(x3, active_density, block_k, prior, prior_alpha, prior_eps, _block_score_scale)
                return out

            mod.forward = forward
            originals.append((mod, orig_forward))
    return originals, stats


def _restore(originals):
    for mod, orig in originals:
        mod.forward = orig


def _decode_measure(model, ids, prefill: int, decode: int, device: str, repeats: int, warmup: int):
    ids = ids.to(device)
    batch = int(ids.shape[0])
    records = []
    total_runs = warmup + repeats
    for rep in range(total_runs):
        torch.cuda.synchronize()
        with torch.inference_mode():
            out = model(input_ids=ids[:, :prefill], use_cache=True, return_dict=False)
            logits, past = out[0], out[1]
            total_nll = 0.0
            total_tok = 0
            first_loss = F.cross_entropy(logits[:, -1, :].float(), ids[:, prefill], reduction="sum")
            total_nll += float(first_loss.item())
            total_tok += batch
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for t in range(1, decode):
                dec = model(input_ids=ids[:, prefill + t - 1 : prefill + t], past_key_values=past, use_cache=True, return_dict=False)
                logits_t, past = dec[0], dec[1]
                loss = F.cross_entropy(logits_t[:, -1, :].float(), ids[:, prefill + t], reduction="sum")
                total_nll += float(loss.item())
                total_tok += batch
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        if rep >= warmup:
            records.append({
                "repeat": rep - warmup,
                "decode_elapsed_s": elapsed,
                "decode_tps": float((batch * max(decode - 1, 1)) / max(elapsed, 1e-12)),
                "nll": total_nll / max(total_tok, 1),
                "ppl": math.exp(total_nll / max(total_tok, 1)),
            })
    return records


def _summarize(records):
    keys = ["decode_elapsed_s", "decode_tps", "nll", "ppl"]
    return {k: float(sum(r[k] for r in records) / len(records)) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allocation_json", type=Path, required=True)
    p.add_argument("--sweep_json", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch_idx", type=int, default=5)
    p.add_argument("--num_batches", type=int, default=9)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--prefill", type=int, default=1024)
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile_sparse_parts", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    batches = _load_wikitext_batches(tok, args.batch_size, args.prefill, args.decode, args.num_batches)
    ids = batches[int(args.batch_idx)]
    alloc, alloc_meta = _load_alloc(args.allocation_json, args.batch_idx)
    sweep = json.loads(args.sweep_json.read_text(encoding="utf-8"))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print("[1/3] dense e2e decode", flush=True)
    dense_records = _decode_measure(model, ids, args.prefill, args.decode, args.device, args.repeats, args.warmup)
    dense_summary = _summarize(dense_records)
    print(dense_summary, flush=True)

    print("[2/3] patch sparse projection decode", flush=True)
    originals, sparse_stats = _patch_roundrobin_sparse(model, alloc, sweep, bool(args.profile_sparse_parts))
    sparse_records = _decode_measure(model, ids, args.prefill, args.decode, args.device, args.repeats, args.warmup)
    sparse_summary = _summarize(sparse_records)
    _restore(originals)
    print(sparse_summary, flush=True)

    result = {
        "config": vars(args),
        "allocation_meta": {
            "batch_idx": int(alloc_meta["batch_idx"]),
            "mean_density": float(alloc_meta["mean_density"]),
            "budget_ppl": float(alloc_meta.get("budget_ppl", 0.0)),
            "ppl_final": float(alloc_meta.get("ppl_final", 0.0)),
        },
        "dense_records": dense_records,
        "sparse_records": sparse_records,
        "dense_summary": dense_summary,
        "sparse_summary": sparse_summary,
        "decode_speedup_dense_over_sparse": dense_summary["decode_elapsed_s"] / sparse_summary["decode_elapsed_s"],
        "tps_speedup_sparse_over_dense": sparse_summary["decode_tps"] / dense_summary["decode_tps"],
        "delta_nll_sparse_minus_dense": sparse_summary["nll"] - dense_summary["nll"],
        "delta_ppl_sparse_minus_dense": sparse_summary["ppl"] - dense_summary["ppl"],
        "sparse_kernel_stats": sparse_stats,
    }
    out = args.out_dir / "roundrobin_e2e_decode_actual.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[3/3] saved", out, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
