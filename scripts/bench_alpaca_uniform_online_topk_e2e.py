#!/usr/bin/env python3
"""Practical online block-topk decode benchmark on Alpaca.

This intentionally avoids oracle per-cell sparsity schedules and quality
budgets.  For a fixed global sparsity, every projection uses the same density,
and the active K-blocks are selected online from the current decode activation
using batch-mean block L2 scores.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from bench_roundrobin_e2e_decode import (  # noqa: E402
    INDEX_BLOCK_META_SIZE,
    _get_module,
    _patch_roundrobin_sparse,
    _restore,
    _select_active_blocks,
)
from eval_perstep_budget_sweep import CACHE_DIR, MODEL_NAME, NUM_LAYERS, PROJS, _prompt  # noqa: E402


def _load_group_indices(perstep_json: Path, batch_size: int, group_id: int) -> list[int]:
    obj = json.loads(perstep_json.read_text())
    for group in obj["groups"]:
        if int(group["batch_size"]) == int(batch_size) and int(group["group_id"]) == int(group_id):
            return [int(x) for x in group["dataset_indices"]]
    raise RuntimeError(f"group not found: B={batch_size}, group={group_id}")


def _load_alpaca_batch(tokenizer, dataset_indices: list[int], decode: int):
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=CACHE_DIR)
    rows = []
    for idx in dataset_indices:
        row = ds[int(idx)]
        prompt_ids = tokenizer(_prompt(row), add_special_tokens=True).input_ids
        response_ids = tokenizer((row.get("output") or "").strip(), add_special_tokens=False).input_ids
        if len(response_ids) < decode:
            raise RuntimeError(f"dataset_idx={idx} response shorter than decode={decode}")
        rows.append({
            "dataset_idx": int(idx),
            "prompt_ids": prompt_ids,
            "response_ids": response_ids[:decode],
            "prompt_len": int(len(prompt_ids)),
        })
    return rows


def _build_tensors(rows: list[dict], pad_id: int, decode: int, device: str):
    bsz = len(rows)
    max_prompt = max(r["prompt_len"] for r in rows)
    prompt_ids = torch.full((bsz, max_prompt), int(pad_id), dtype=torch.long)
    prompt_am = torch.zeros((bsz, max_prompt), dtype=torch.long)
    last_pos = torch.zeros(bsz, dtype=torch.long)
    resp = torch.empty((bsz, decode), dtype=torch.long)
    for bi, r in enumerate(rows):
        p = torch.tensor(r["prompt_ids"], dtype=torch.long)
        prompt_ids[bi, :p.numel()] = p
        prompt_am[bi, :p.numel()] = 1
        last_pos[bi] = p.numel() - 1
        resp[bi] = torch.tensor(r["response_ids"], dtype=torch.long)
    return (
        prompt_ids.to(device),
        prompt_am.to(device),
        last_pos.to(device),
        resp.to(device),
        max_prompt,
    )


def _build_full_teacher_tensors(rows: list[dict], pad_id: int, decode: int, device: str):
    bsz = len(rows)
    max_len = max(int(r["prompt_len"]) + decode for r in rows)
    ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
    am = torch.zeros((bsz, max_len), dtype=torch.long)
    starts = torch.zeros(bsz, dtype=torch.long)
    for bi, r in enumerate(rows):
        p = torch.tensor(r["prompt_ids"], dtype=torch.long)
        y = torch.tensor(r["response_ids"][:decode], dtype=torch.long)
        ids[bi, :p.numel()] = p
        ids[bi, p.numel():p.numel() + y.numel()] = y
        am[bi, :p.numel() + y.numel()] = 1
        starts[bi] = p.numel()
    return ids.to(device), am.to(device), starts.to(device)


def _uniform_alloc(density: float, enabled_projs: set[str] | None = None) -> dict:
    enabled = enabled_projs or set(PROJS)
    return {
        str(li): {proj: float(density) if proj in enabled else 1.0 for proj in PROJS}
        for li in range(NUM_LAYERS)
    }


def _projection_sparsity_alloc(spec: str) -> tuple[dict, dict]:
    sparsity_by_proj = {}
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"projection sparsity item must be proj:sparsity, got {item!r}")
        proj, val = item.split(":", 1)
        proj = proj.strip()
        if proj not in PROJS:
            raise ValueError(f"unknown projection {proj!r}; choices={PROJS}")
        sparsity_by_proj[proj] = max(0.0, min(0.95, float(val)))
    alloc = {}
    for li in range(NUM_LAYERS):
        alloc[str(li)] = {}
        for proj in PROJS:
            alloc[str(li)][proj] = 1.0 - float(sparsity_by_proj.get(proj, 0.0))
    mean_s = sum(float(sparsity_by_proj.get(proj, 0.0)) for proj in PROJS) / len(PROJS)
    return alloc, {
        "kind": "projection_sparsity",
        "projection_sparsities": dict(sorted(sparsity_by_proj.items())),
        "mean_sparsity": float(mean_s),
    }


def _cell_sparsity_alloc(path: Path) -> tuple[dict, dict]:
    import csv

    alloc = _uniform_alloc(1.0)
    cell_rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            proj = str(row["proj"]).strip()
            if not (0 <= layer < NUM_LAYERS):
                raise ValueError(f"bad layer {layer}")
            if proj not in PROJS:
                raise ValueError(f"unknown projection {proj!r}; choices={PROJS}")
            s = max(0.0, min(0.95, float(row["sparsity"])))
            alloc[str(layer)][proj] = float(1.0 - s)
            out = {"layer": layer, "proj": proj, "sparsity": float(s)}
            for key, val in row.items():
                if key not in out:
                    out[key] = val
            cell_rows.append(out)
    flat_s = [1.0 - float(alloc[str(li)][proj]) for li in range(NUM_LAYERS) for proj in PROJS]
    return alloc, {
        "kind": "cell_sparsity_csv",
        "source_csv": str(path),
        "mean_sparsity": float(sum(flat_s) / len(flat_s)),
        "min_sparsity": float(min(flat_s)),
        "max_sparsity": float(max(flat_s)),
        "cell_rows": cell_rows,
    }


def _step_cell_sparsity_alloc(path: Path) -> tuple[dict, dict, dict[tuple[int, str], list[float]]]:
    import csv

    raw: dict[tuple[int, str], dict[int, float]] = {}
    max_step = 0
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"step", "layer", "proj", "sparsity"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"step cell CSV missing columns: {sorted(missing)}")
        for row in reader:
            step = int(row["step"])
            layer = int(row["layer"])
            proj = str(row["proj"]).strip()
            if not (0 <= layer < NUM_LAYERS):
                raise ValueError(f"bad layer {layer}")
            if proj not in PROJS:
                raise ValueError(f"unknown projection {proj!r}; choices={PROJS}")
            s = max(0.0, min(0.95, float(row["sparsity"])))
            raw.setdefault((layer, proj), {})[step] = s
            max_step = max(max_step, step)
            rows.append({"step": step, "layer": layer, "proj": proj, "sparsity": float(s)})

    step_density: dict[tuple[int, str], list[float]] = {}
    alloc = _uniform_alloc(1.0)
    flat = []
    for li in range(NUM_LAYERS):
        for proj in PROJS:
            by_step = raw.get((li, proj), {})
            vals = []
            last = 0.0
            for step in range(max_step + 1):
                if step in by_step:
                    last = float(by_step[step])
                vals.append(last)
            if not vals:
                vals = [0.0]
            densities = [1.0 - max(0.0, min(0.95, v)) for v in vals]
            step_density[(li, proj)] = densities
            mean_s = sum(vals) / len(vals)
            alloc[str(li)][proj] = 1.0 - mean_s
            flat.extend(vals)

    return alloc, {
        "kind": "step_cell_sparsity_csv",
        "source_csv": str(path),
        "max_step": int(max_step),
        "mean_sparsity": float(sum(flat) / len(flat)),
        "min_sparsity": float(min(flat)),
        "max_sparsity": float(max(flat)),
        "cell_rows": rows,
    }, step_density


def _parse_projection_scale_overrides(spec: str) -> dict[str, float]:
    scales: dict[str, float] = {}
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"projection scale item must be proj:scale, got {item!r}")
        proj, val = item.split(":", 1)
        proj = proj.strip()
        if proj not in PROJS:
            raise ValueError(f"unknown projection {proj!r}; choices={PROJS}")
        scales[proj] = float(val)
    return scales


def _load_cell_scale_csv(path: Path | None) -> dict[tuple[str, int], float]:
    if path is None or not str(path).strip():
        return {}
    import csv

    scales: dict[tuple[str, int], float] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            proj = str(row["proj"]).strip()
            layer = int(row["layer"])
            scale = float(row["scale"])
            if proj not in PROJS:
                raise ValueError(f"unknown projection {proj!r}; choices={PROJS}")
            scales[(proj, layer)] = scale
    return scales


def _load_cell_gate_csv(path: Path | None) -> dict[tuple[int, str], float]:
    if path is None or not str(path).strip():
        return {}
    import csv

    gates: dict[tuple[int, str], float] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            proj = str(row["proj"]).strip()
            layer = int(row["layer"])
            gate = float(row["gate"])
            if proj not in PROJS:
                raise ValueError(f"unknown projection {proj!r}; choices={PROJS}")
            gates[(layer, proj)] = gate
    return gates


def _static_layerproj_alloc_from_perstep(
    perstep_json: Path,
    batch_size: int,
    exclude_group_id: int,
    scale: float,
) -> tuple[dict, dict]:
    """Build a calibration-derived static per-layer/proj density schedule.

    We average oracle achievable sparsities over other groups and all recorded
    steps, then scale the resulting sparsity matrix by a single global factor.
    The held-out target group is excluded so this is not using its per-cell
    oracle schedule.
    """
    obj = json.loads(perstep_json.read_text())
    mats = []
    prefix = f"B{int(batch_size)}_g"
    for key, mat in obj.get("S", {}).items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        gid = int(rest.split("_", 1)[0])
        if gid == int(exclude_group_id):
            continue
        mats.append(mat)
    if not mats:
        raise RuntimeError(f"no calibration schedules for B={batch_size}, excluding group {exclude_group_id}")

    # S has shape [projection, layer].
    sums = [[0.0 for _ in range(NUM_LAYERS)] for _ in PROJS]
    for mat in mats:
        for pi in range(len(PROJS)):
            for li in range(NUM_LAYERS):
                sums[pi][li] += float(mat[pi][li])
    denom = float(len(mats))
    sp = [[max(0.0, min(0.9, (sums[pi][li] / denom) * float(scale))) for li in range(NUM_LAYERS)] for pi in range(len(PROJS))]
    alloc = {}
    for li in range(NUM_LAYERS):
        alloc[str(li)] = {}
        for pi, proj in enumerate(PROJS):
            alloc[str(li)][proj] = float(1.0 - sp[pi][li])
    flat = [x for row in sp for x in row]
    meta = {
        "num_calibration_heatmaps": int(len(mats)),
        "exclude_group_id": int(exclude_group_id),
        "scale": float(scale),
        "mean_sparsity": float(sum(flat) / len(flat)),
        "min_sparsity": float(min(flat)),
        "max_sparsity": float(max(flat)),
    }
    return alloc, meta


def _calibration_sparsity_stack(perstep_json: Path, batch_size: int, exclude_group_id: int):
    obj = json.loads(perstep_json.read_text())
    mats = []
    prefix = f"B{int(batch_size)}_g"
    for key, mat in obj.get("S", {}).items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        gid = int(rest.split("_", 1)[0])
        if gid == int(exclude_group_id):
            continue
        mats.append(mat)
    if not mats:
        raise RuntimeError(f"no calibration schedules for B={batch_size}, excluding group {exclude_group_id}")
    return mats


def _alloc_from_sparsity_matrix(sp, meta: dict) -> tuple[dict, dict]:
    alloc = {}
    flat = []
    for li in range(NUM_LAYERS):
        alloc[str(li)] = {}
        for pi, proj in enumerate(PROJS):
            val = max(0.0, min(0.9, float(sp[pi][li])))
            flat.append(val)
            alloc[str(li)][proj] = float(1.0 - val)
    return alloc, {
        **meta,
        "mean_sparsity": float(sum(flat) / len(flat)),
        "min_sparsity": float(min(flat)),
        "max_sparsity": float(max(flat)),
    }


def _static_layerproj_quantile_alloc_from_perstep(
    perstep_json: Path,
    batch_size: int,
    exclude_group_id: int,
    quantile: float,
    scale: float,
    projection_scales: dict[str, float] | None = None,
    cell_scales: dict[tuple[str, int], float] | None = None,
) -> tuple[dict, dict]:
    """Robust calibration schedule using a per-cell lower quantile.

    Unlike the global-scale schedule, this keeps cells with consistently high
    oracle sparsity relatively aggressive and suppresses unstable cells.
    """
    import numpy as np

    mats = _calibration_sparsity_stack(perstep_json, batch_size, exclude_group_id)
    arr = np.asarray(mats, dtype=np.float64)  # [num_calib_maps, proj, layer]
    q = max(0.0, min(1.0, float(quantile)))
    sp = np.quantile(arr, q, axis=0) * float(scale)
    projection_scales = projection_scales or {}
    cell_scales = cell_scales or {}
    for pi, proj in enumerate(PROJS):
        sp[pi, :] *= float(projection_scales.get(proj, 1.0))
        for li in range(NUM_LAYERS):
            sp[pi, li] *= float(cell_scales.get((proj, li), 1.0))
    return _alloc_from_sparsity_matrix(
        sp,
        {
            "kind": "static_layerproj_quantile",
            "num_calibration_heatmaps": int(arr.shape[0]),
            "exclude_group_id": int(exclude_group_id),
            "quantile": q,
            "scale": float(scale),
            "projection_scales": dict(sorted(projection_scales.items())),
            "cell_scale_csv_applied": bool(cell_scales),
        },
    )


def _static_layerproj_meanstd_alloc_from_perstep(
    perstep_json: Path,
    batch_size: int,
    exclude_group_id: int,
    std_mult: float,
    scale: float,
) -> tuple[dict, dict]:
    """Robust calibration schedule using mean - lambda * std per cell."""
    import numpy as np

    mats = _calibration_sparsity_stack(perstep_json, batch_size, exclude_group_id)
    arr = np.asarray(mats, dtype=np.float64)
    sp = (arr.mean(axis=0) - float(std_mult) * arr.std(axis=0)) * float(scale)
    return _alloc_from_sparsity_matrix(
        sp,
        {
            "kind": "static_layerproj_mean_minus_std",
            "num_calibration_heatmaps": int(arr.shape[0]),
            "exclude_group_id": int(exclude_group_id),
            "std_mult": float(std_mult),
            "scale": float(scale),
        },
    )


def _decode_measure_alpaca(model, prompt_ids, prompt_am, last_pos, resp, repeats: int, warmup: int):
    batch = int(prompt_ids.shape[0])
    decode = int(resp.shape[1])
    records = []
    total_runs = int(warmup) + int(repeats)
    arange_b = torch.arange(batch, device=prompt_ids.device)
    for rep in range(total_runs):
        torch.cuda.synchronize()
        with torch.inference_mode():
            out = model(input_ids=prompt_ids, attention_mask=prompt_am, use_cache=True, return_dict=False)
            logits, past = out[0], out[1]
            total_nll = float(F.cross_entropy(logits[arange_b, last_pos, :].float(), resp[:, 0], reduction="sum").item())
            total_tok = batch

            attn = torch.cat([prompt_am, torch.ones(batch, 1, dtype=prompt_am.dtype, device=prompt_am.device)], dim=1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for t in range(1, decode):
                dec = model(
                    input_ids=resp[:, t - 1:t],
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=False,
                )
                logits_t, past = dec[0], dec[1]
                total_nll += float(F.cross_entropy(logits_t[:, -1, :].float(), resp[:, t], reduction="sum").item())
                total_tok += batch
                if t + 1 < decode:
                    attn = torch.cat([attn, torch.ones(batch, 1, dtype=attn.dtype, device=attn.device)], dim=1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        if rep >= warmup:
            nll = total_nll / max(total_tok, 1)
            records.append({
                "repeat": int(rep - warmup),
                "decode_elapsed_s": float(elapsed),
                "decode_tps": float((batch * max(decode - 1, 1)) / max(elapsed, 1e-12)),
                "nll": float(nll),
                "ppl": float(math.exp(nll)),
            })
    return records


def _collect_calibration_block_priors(
    model,
    tokenizer,
    perstep_json: Path,
    batch_size: int,
    target_group_id: int,
    decode: int,
    alloc: dict,
    device: str,
) -> tuple[dict, dict]:
    """Collect block survival frequency from held-out calibration groups.

    The model remains dense: hooks only observe decode-time activations and
    count which K-blocks would be selected by activation top-k under `alloc`.
    """
    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    counts: dict[tuple[int, str], torch.Tensor] = {}
    totals: dict[tuple[int, str], int] = {}
    originals = []

    for li in range(NUM_LAYERS):
        for proj in PROJS:
            mod = _get_module(model, li, proj)
            density = float(alloc[str(li)][proj])
            total_k = int(mod.weight.shape[1]) // block_k
            key = (li, proj)
            counts[key] = torch.zeros(total_k, dtype=torch.float32, device=device)
            totals[key] = 0
            orig_forward = mod.forward

            def forward(x, _orig=orig_forward, _density=density, _key=key):
                if _density < 1.0:
                    x3 = x.reshape(x.shape[0], 1, x.shape[-1]) if x.dim() == 2 else x
                    if x3.shape[1] == 1:
                        active = _select_active_blocks(x3, _density, block_k)
                        counts[_key][active.to(torch.long)] += 1.0
                        totals[_key] += 1
                return _orig(x)

            mod.forward = forward
            originals.append((mod, orig_forward))

    meta_groups = []
    try:
        for gid in range(5):
            if int(gid) == int(target_group_id):
                continue
            idx = _load_group_indices(perstep_json, batch_size, gid)
            rows = _load_alpaca_batch(tokenizer, idx, decode)
            prompt_ids, prompt_am, last_pos, resp, _ = _build_tensors(rows, int(tokenizer.pad_token_id), decode, device)
            _ = _decode_measure_alpaca(model, prompt_ids, prompt_am, last_pos, resp, repeats=1, warmup=0)
            meta_groups.append({"group_id": int(gid), "num_sequences": int(len(idx))})
    finally:
        _restore(originals)

    priors = {}
    for key, c in counts.items():
        denom = max(1, int(totals[key]))
        priors[key] = (c / float(denom)).detach()
    meta = {
        "kind": "calibration_block_frequency",
        "calibration_groups": meta_groups,
        "target_group_id": int(target_group_id),
        "decode": int(decode),
        "block_k": int(block_k),
        "num_cells": int(len(priors)),
        "mean_observations_per_cell": float(sum(totals.values()) / max(1, len(totals))),
    }
    return priors, meta


def _warmup_local_error_alloc(
    model,
    ids: torch.Tensor,
    am: torch.Tensor,
    starts: torch.Tensor,
    tau: float,
    block_k: int,
) -> tuple[dict, dict]:
    """Choose per-cell sparsity from one dense teacher-forced warmup token.

    This uses no labels and no PPL oracle.  For each layer/projection, it
    compares the dense linear output with sparse-input linear outputs selected
    by dynamic activation top-k on the current batch.
    """
    saved: dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    bsz = int(ids.shape[0])
    arange_b = torch.arange(bsz, device=ids.device)

    for li in range(NUM_LAYERS):
        for proj in PROJS:
            mod = _get_module(model, li, proj)

            def _h(_module, inputs, _key=(li, proj)):
                x = inputs[0].detach()
                if x.dim() == 2:
                    x = x.reshape(bsz, -1, x.shape[-1])
                saved[_key] = x[arange_b, starts, :].detach()

            handles.append(mod.register_forward_pre_hook(_h))
    with torch.no_grad():
        _ = model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=False)
    for h in handles:
        h.remove()

    alloc = {}
    cell_rows = []
    candidate_s = [round(x / 10.0, 1) for x in range(0, 10)]
    for li in range(NUM_LAYERS):
        alloc[str(li)] = {}
        for proj in PROJS:
            x = saved[(li, proj)]
            mod = _get_module(model, li, proj)
            w = mod.weight.detach()
            d_in = int(w.shape[1])
            total_k = int(d_in // block_k)
            d_eff = total_k * block_k
            xb = x[:, :d_eff].float().reshape(x.shape[0], total_k, block_k)
            score = xb.norm(dim=2).mean(dim=0)
            dense_out = torch.nn.functional.linear(x, w).float()
            dense_norm = torch.linalg.vector_norm(dense_out).clamp_min(1.0e-12)
            best_s = 0.0
            best_rel = 0.0
            rel_by_s = {}
            for s in candidate_s:
                k_keep = max(1, min(total_k, int(round((1.0 - float(s)) * total_k))))
                if k_keep >= total_k:
                    rel = 0.0
                else:
                    keep = torch.topk(score, k=k_keep).indices
                    mask = torch.zeros(d_eff, device=x.device, dtype=x.dtype)
                    for b in keep:
                        bb = int(b.item())
                        mask[bb * block_k:(bb + 1) * block_k] = 1
                    if d_eff < d_in:
                        mask = torch.cat([mask, torch.ones(d_in - d_eff, device=x.device, dtype=x.dtype)])
                    sparse_out = torch.nn.functional.linear(x * mask, w).float()
                    rel = float((torch.linalg.vector_norm(dense_out - sparse_out) / dense_norm).item())
                rel_by_s[f"{s:.1f}"] = rel
                if rel <= float(tau) + 1.0e-12:
                    best_s = float(s)
                    best_rel = rel
            alloc[str(li)][proj] = float(1.0 - best_s)
            cell_rows.append({"layer": int(li), "proj": proj, "sparsity": float(best_s), "rel_error": float(best_rel), **rel_by_s})
    flat = [1.0 - float(alloc[str(li)][proj]) for li in range(NUM_LAYERS) for proj in PROJS]
    meta = {
        "kind": "warmup_local_linear_error",
        "tau": float(tau),
        "warmup_position": "first_response_token",
        "mean_sparsity": float(sum(flat) / len(flat)),
        "min_sparsity": float(min(flat)),
        "max_sparsity": float(max(flat)),
        "cell_rows": cell_rows,
    }
    return alloc, meta


def _warmup_logit_teacher_alloc(
    model,
    ids: torch.Tensor,
    am: torch.Tensor,
    starts: torch.Tensor,
    tau: float,
    block_k: int,
    candidate_s: list[float],
    top_vocab: int,
) -> tuple[dict, dict]:
    """Choose per-cell sparsity from first-token dense-logit agreement.

    This is a label-free calibration signal.  For each layer/projection cell,
    the dense model's logits at the first response-token position are treated
    as a teacher.  We sparsify only that cell using the current activation
    top-k block mask and choose the largest sparsity whose teacher top-vocab
    KL stays below `tau`.
    """
    bsz = int(ids.shape[0])
    arange_b = torch.arange(bsz, device=ids.device)
    pos = starts
    with torch.no_grad():
        dense_logits_all = model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=False)[0]
        dense_logits = dense_logits_all[arange_b, pos, :].float()
        vocab_k = max(1, min(int(top_vocab), int(dense_logits.shape[-1])))
        teacher_idx = torch.topk(dense_logits, k=vocab_k, dim=-1).indices
        teacher_logits = torch.gather(dense_logits, dim=-1, index=teacher_idx)
        teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
        teacher_p = teacher_logp.exp()

    def _masked_forward_kl(layer: int, proj: str, sparsity: float) -> float:
        if float(sparsity) <= 0.0:
            return 0.0
        mod = _get_module(model, layer, proj)
        d_in = int(mod.weight.shape[1])
        total_k = int(d_in // block_k)
        d_eff = total_k * block_k
        if total_k <= 1:
            return 0.0

        def _hook(_module, inputs):
            x = inputs[0]
            if x.dim() == 2:
                # This path is not expected for the teacher-forced warmup, but
                # keep it dense rather than applying an ambiguous position mask.
                return inputs
            x_pos = x[arange_b, pos, :]
            xb = x_pos[:, :d_eff].float().reshape(bsz, total_k, block_k)
            score = xb.norm(dim=2).mean(dim=0)
            k_keep = max(1, min(total_k, int(round((1.0 - float(sparsity)) * total_k))))
            if k_keep >= total_k:
                return inputs
            keep = torch.topk(score, k=k_keep).indices
            block_mask = torch.zeros(total_k, device=x.device, dtype=x.dtype)
            block_mask[keep.to(torch.long)] = 1
            mask = block_mask.repeat_interleave(block_k)
            if d_eff < d_in:
                mask = torch.cat([mask, torch.ones(d_in - d_eff, device=x.device, dtype=x.dtype)])
            x_mod = x.clone()
            x_mod[arange_b, pos, :] = x_pos * mask
            return (x_mod, *inputs[1:])

        handle = mod.register_forward_pre_hook(_hook)
        try:
            with torch.no_grad():
                sparse_logits_all = model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=False)[0]
                sparse_logits = sparse_logits_all[arange_b, pos, :].float()
                sparse_teacher_logits = torch.gather(sparse_logits, dim=-1, index=teacher_idx)
                sparse_logq = torch.log_softmax(sparse_teacher_logits, dim=-1)
                kl = (teacher_p * (teacher_logp - sparse_logq)).sum(dim=-1).mean()
                return float(kl.item())
        finally:
            handle.remove()

    alloc = {}
    cell_rows = []
    candidate_s = sorted({max(0.0, min(0.9, float(s))) for s in candidate_s})
    for li in range(NUM_LAYERS):
        alloc[str(li)] = {}
        for proj in PROJS:
            best_s = 0.0
            best_kl = 0.0
            kl_by_s = {}
            for s in candidate_s:
                kl = _masked_forward_kl(li, proj, s)
                kl_by_s[f"{s:.2f}"] = kl
                if kl <= float(tau) + 1.0e-12:
                    best_s = float(s)
                    best_kl = kl
            alloc[str(li)][proj] = float(1.0 - best_s)
            cell_rows.append({"layer": int(li), "proj": proj, "sparsity": float(best_s), "kl": float(best_kl), **kl_by_s})
            if (len(cell_rows) % 32) == 0:
                print(f"[calibration] logit teacher processed {len(cell_rows)}/{NUM_LAYERS * len(PROJS)} cells", flush=True)
    flat = [1.0 - float(alloc[str(li)][proj]) for li in range(NUM_LAYERS) for proj in PROJS]
    meta = {
        "kind": "warmup_dense_logit_teacher_kl",
        "tau": float(tau),
        "warmup_position": "first_response_token",
        "candidate_sparsities": candidate_s,
        "top_vocab": int(top_vocab),
        "mean_sparsity": float(sum(flat) / len(flat)),
        "min_sparsity": float(min(flat)),
        "max_sparsity": float(max(flat)),
        "cell_rows": cell_rows,
    }
    return alloc, meta


def _warmup_logit_teacher_alloc_from_csv(path: Path, tau: float) -> tuple[dict, dict]:
    import csv

    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"empty warmup logit teacher CSV: {path}")
    sparsity_cols = sorted(
        [c for c in rows[0].keys() if c not in {"layer", "proj", "sparsity", "kl"}],
        key=lambda x: float(x),
    )
    alloc = {str(li): {} for li in range(NUM_LAYERS)}
    cell_rows = []
    for row in rows:
        li = int(row["layer"])
        proj = str(row["proj"])
        best_s = 0.0
        best_kl = 0.0
        for col in sparsity_cols:
            s = float(col)
            kl = float(row[col])
            if kl <= float(tau) + 1.0e-12:
                best_s = s
                best_kl = kl
        alloc[str(li)][proj] = float(1.0 - best_s)
        cell_rows.append({"layer": li, "proj": proj, "sparsity": float(best_s), "kl": float(best_kl)})
    flat = [1.0 - float(alloc[str(li)][proj]) for li in range(NUM_LAYERS) for proj in PROJS]
    meta = {
        "kind": "warmup_dense_logit_teacher_kl_from_csv",
        "tau": float(tau),
        "source_csv": str(path),
        "candidate_sparsities": [float(c) for c in sparsity_cols],
        "mean_sparsity": float(sum(flat) / len(flat)),
        "min_sparsity": float(min(flat)),
        "max_sparsity": float(max(flat)),
        "cell_rows": cell_rows,
    }
    return alloc, meta


def _teacher_top_vocab(model, ids: torch.Tensor, am: torch.Tensor, starts: torch.Tensor, top_vocab: int):
    bsz = int(ids.shape[0])
    arange_b = torch.arange(bsz, device=ids.device)
    with torch.no_grad():
        dense_logits_all = model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=False)[0]
        dense_logits = dense_logits_all[arange_b, starts, :].float()
        vocab_k = max(1, min(int(top_vocab), int(dense_logits.shape[-1])))
        teacher_idx = torch.topk(dense_logits, k=vocab_k, dim=-1).indices
        teacher_logits = torch.gather(dense_logits, dim=-1, index=teacher_idx)
        teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
        teacher_p = teacher_logp.exp()
    return teacher_idx, teacher_logp, teacher_p


def _teacher_kl_with_alloc(
    model,
    ids: torch.Tensor,
    am: torch.Tensor,
    starts: torch.Tensor,
    alloc: dict,
    block_k: int,
    teacher_idx: torch.Tensor,
    teacher_logp: torch.Tensor,
    teacher_p: torch.Tensor,
) -> float:
    bsz = int(ids.shape[0])
    arange_b = torch.arange(bsz, device=ids.device)
    handles = []
    for li in range(NUM_LAYERS):
        for proj in PROJS:
            density = float(alloc[str(li)][proj])
            if density >= 1.0:
                continue
            mod = _get_module(model, li, proj)
            d_in = int(mod.weight.shape[1])
            total_k = int(d_in // block_k)
            d_eff = total_k * block_k
            if total_k <= 1:
                continue

            def _hook(_module, inputs, _density=density, _d_in=d_in, _total_k=total_k, _d_eff=d_eff):
                x = inputs[0]
                if x.dim() == 2:
                    return inputs
                x_pos = x[arange_b, starts, :]
                xb = x_pos[:, :_d_eff].float().reshape(bsz, _total_k, block_k)
                score = xb.norm(dim=2).mean(dim=0)
                k_keep = max(1, min(_total_k, int(round(float(_density) * _total_k))))
                if k_keep >= _total_k:
                    return inputs
                keep = torch.topk(score, k=k_keep).indices
                block_mask = torch.zeros(_total_k, device=x.device, dtype=x.dtype)
                block_mask[keep.to(torch.long)] = 1
                mask = block_mask.repeat_interleave(block_k)
                if _d_eff < _d_in:
                    mask = torch.cat([mask, torch.ones(_d_in - _d_eff, device=x.device, dtype=x.dtype)])
                x_mod = x.clone()
                x_mod[arange_b, starts, :] = x_pos * mask
                return (x_mod, *inputs[1:])

            handles.append(mod.register_forward_pre_hook(_hook))
    try:
        with torch.no_grad():
            sparse_logits_all = model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=False)[0]
            sparse_logits = sparse_logits_all[arange_b, starts, :].float()
            sparse_teacher_logits = torch.gather(sparse_logits, dim=-1, index=teacher_idx)
            sparse_logq = torch.log_softmax(sparse_teacher_logits, dim=-1)
            kl = (teacher_p * (teacher_logp - sparse_logq)).sum(dim=-1).mean()
            return float(kl.item())
    finally:
        for h in handles:
            h.remove()


def _warmup_interaction_logit_teacher_alloc(
    model,
    ids: torch.Tensor,
    am: torch.Tensor,
    starts: torch.Tensor,
    tau: float,
    block_k: int,
    candidate_s: list[float],
    top_vocab: int,
) -> tuple[dict, dict]:
    """Greedy interaction-aware dense-teacher calibration.

    Unlike the independent teacher test, each candidate is evaluated with all
    previously accepted sparse cells active at the same time.  It therefore
    measures cumulative teacher-logit drift under the current schedule.
    """
    teacher_idx, teacher_logp, teacher_p = _teacher_top_vocab(model, ids, am, starts, top_vocab)
    alloc = _uniform_alloc(1.0)
    candidate_s = sorted({max(0.0, min(0.9, float(s))) for s in candidate_s})
    candidate_s = [s for s in candidate_s if s > 0.0]
    attempts = []
    accepted = 0
    for li in range(NUM_LAYERS):
        for proj in PROJS:
            current_s = 0.0
            for s in candidate_s:
                if s <= current_s:
                    continue
                old_density = float(alloc[str(li)][proj])
                alloc[str(li)][proj] = float(1.0 - s)
                kl = _teacher_kl_with_alloc(
                    model,
                    ids,
                    am,
                    starts,
                    alloc,
                    block_k,
                    teacher_idx,
                    teacher_logp,
                    teacher_p,
                )
                ok = kl <= float(tau) + 1.0e-12
                attempts.append({"layer": int(li), "proj": proj, "try_sparsity": float(s), "kl": float(kl), "accepted": bool(ok)})
                if ok:
                    current_s = float(s)
                    accepted += 1
                else:
                    alloc[str(li)][proj] = old_density
                    break
            if (len(attempts) % 32) == 0:
                flat_now = [1.0 - float(alloc[str(l)][p]) for l in range(NUM_LAYERS) for p in PROJS]
                print(
                    f"[calibration] interaction teacher attempts={len(attempts)} accepted={accepted} meanS={sum(flat_now)/len(flat_now):.3f}",
                    flush=True,
                )
    final_kl = _teacher_kl_with_alloc(
        model,
        ids,
        am,
        starts,
        alloc,
        block_k,
        teacher_idx,
        teacher_logp,
        teacher_p,
    )
    flat = [1.0 - float(alloc[str(li)][proj]) for li in range(NUM_LAYERS) for proj in PROJS]
    cell_rows = [{"layer": int(li), "proj": proj, "sparsity": float(1.0 - alloc[str(li)][proj])} for li in range(NUM_LAYERS) for proj in PROJS]
    meta = {
        "kind": "warmup_interaction_dense_logit_teacher_kl",
        "tau": float(tau),
        "warmup_position": "first_response_token",
        "candidate_sparsities": candidate_s,
        "top_vocab": int(top_vocab),
        "final_warmup_kl": float(final_kl),
        "num_attempts": int(len(attempts)),
        "num_accepted_increments": int(accepted),
        "mean_sparsity": float(sum(flat) / len(flat)),
        "min_sparsity": float(min(flat)),
        "max_sparsity": float(max(flat)),
        "cell_rows": cell_rows,
        "attempt_rows": attempts,
    }
    return alloc, meta


def _summarize(records: list[dict]) -> dict:
    return {
        k: float(sum(float(r[k]) for r in records) / len(records))
        for k in ["decode_elapsed_s", "decode_tps", "nll", "ppl"]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--perstep_json", type=Path, default=Path(__file__).resolve().parents[1] / "results/oracle/perstep_batch_sweep.json")
    ap.add_argument("--sweep_json", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--group_id", type=int, default=3)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--sparsities", default="0.3,0.5,0.7")
    ap.add_argument("--static_schedule_scales", default="")
    ap.add_argument(
        "--static_schedule_quantiles",
        default="",
        help="comma-separated q[:scale] specs. Uses per-cell quantile_q over calibration heatmaps excluding target group.",
    )
    ap.add_argument(
        "--static_schedule_meanstd",
        default="",
        help="comma-separated lambda[:scale] specs. Uses per-cell mean - lambda*std over calibration heatmaps.",
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile_sparse_parts", action="store_true")
    ap.add_argument("--freq_weighted_static_scales", default="")
    ap.add_argument("--reuse_static_scales", default="")
    ap.add_argument("--prior_alpha", type=float, default=1.0)
    ap.add_argument("--prior_eps", type=float, default=0.1)
    ap.add_argument("--quantile_block_frequency_prior", action="store_true", help="apply held-out calibration block-frequency prior to static quantile schedules")
    ap.add_argument("--energy_gate_keep", type=float, default=None, help="fallback to dense unless selected top-k blocks retain this fraction of activation energy")
    ap.add_argument("--energy_gate_cell_csv", type=Path, default=None, help="CSV with columns proj,layer,gate overriding energy gate per cell")
    ap.add_argument("--energy_gate_mode", choices=["dense", "expand", "hybrid", "scaled"], default="dense", help="dense: fallback to dense; expand: add blocks until energy gate is met; hybrid: expand only selected projections; scaled: reduce sparsity and rerun dynamic top-k")
    ap.add_argument("--energy_gate_expand_projs", default="", help="comma-separated projection names expanded under hybrid energy gate mode")
    ap.add_argument("--energy_gate_fallback_sparsity_scale", type=float, default=0.5, help="for scaled energy gate mode, multiply current sparsity by this value after gate failure")
    ap.add_argument("--weight_aware_score", action="store_true", help="score blocks by activation energy weighted by projection column norms")
    ap.add_argument("--enabled_projs", default="", help="comma-separated projection subset for uniform sparsity; empty means all projections")
    ap.add_argument("--projection_sparsities", default="", help="comma-separated proj:sparsity spec, e.g. k_proj:0.45,o_proj:0.45,q_proj:0.2")
    ap.add_argument("--cell_sparsity_csv", type=Path, default=None, help="CSV with layer,proj,sparsity; block identities are still selected online by activation top-k")
    ap.add_argument("--step_cell_sparsity_csv", type=Path, default=None, help="CSV with step,layer,proj,sparsity; density is updated per decode step while block identities remain online activation top-k")
    ap.add_argument("--projection_scale_overrides", default="", help="comma-separated proj:scale multipliers applied to static quantile schedules")
    ap.add_argument("--cell_scale_csv", type=Path, default=None, help="CSV with columns proj,layer,scale applied to static quantile schedules")
    ap.add_argument("--warmup_error_taus", default="", help="comma-separated local linear relative-error thresholds for first-token warmup calibration")
    ap.add_argument("--warmup_logit_kl_taus", default="", help="comma-separated dense-logit teacher KL thresholds for first-token warmup calibration")
    ap.add_argument("--warmup_logit_candidate_s", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9", help="comma-separated sparsity candidates for logit-teacher warmup")
    ap.add_argument("--warmup_logit_top_vocab", type=int, default=128, help="teacher top-vocab size used for logit-teacher KL")
    ap.add_argument("--warmup_logit_cell_csv", type=Path, default=None, help="reuse a warmup logit teacher cell CSV instead of recomputing calibration")
    ap.add_argument("--warmup_interaction_logit_kl_taus", default="", help="comma-separated cumulative dense-logit teacher KL thresholds")
    ap.add_argument("--warmup_interaction_candidate_s", default="0.0,0.2,0.4", help="comma-separated sparsity candidates for interaction-aware teacher warmup")
    ap.add_argument("--warmup_interaction_top_vocab", type=int, default=256, help="teacher top-vocab size used for interaction-aware teacher KL")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dataset_indices = _load_group_indices(args.perstep_json, args.batch_size, args.group_id)
    rows = _load_alpaca_batch(tok, dataset_indices, args.decode)
    prompt_ids, prompt_am, last_pos, resp, max_prompt = _build_tensors(rows, int(tok.pad_token_id), args.decode, args.device)
    full_ids, full_am, full_starts = _build_full_teacher_tensors(rows, int(tok.pad_token_id), args.decode, args.device)
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
    print(dense_summary, flush=True)

    methods = []
    candidates = []
    enabled_projs = {x.strip() for x in args.enabled_projs.split(",") if x.strip()}
    if not enabled_projs:
        enabled_projs = set(PROJS)
    if args.projection_sparsities.strip():
        alloc, meta = _projection_sparsity_alloc(args.projection_sparsities)
        candidates.append(("online_projection_sparsity", float(meta["mean_sparsity"]), alloc, meta))
    if args.cell_sparsity_csv:
        alloc, meta = _cell_sparsity_alloc(args.cell_sparsity_csv)
        candidates.append(("online_cell_sparsity_csv", float(meta["mean_sparsity"]), alloc, meta))
    if args.step_cell_sparsity_csv:
        alloc, meta, step_density = _step_cell_sparsity_alloc(args.step_cell_sparsity_csv)
        candidates.append(("online_step_cell_sparsity_csv", float(meta["mean_sparsity"]), alloc, meta, None, step_density))
    projection_scale_overrides = _parse_projection_scale_overrides(args.projection_scale_overrides)
    cell_scales = _load_cell_scale_csv(args.cell_scale_csv)
    cell_gates = _load_cell_gate_csv(args.energy_gate_cell_csv)
    for s in [float(x) for x in args.sparsities.split(",") if x.strip()]:
        density = 1.0 - float(s)
        effective_s = float(s) * (len(enabled_projs) / len(PROJS))
        suffix = "" if set(enabled_projs) == set(PROJS) else "_" + "-".join(sorted(enabled_projs))
        candidates.append((
            f"online_uniform_s{s:.2f}{suffix}",
            effective_s,
            _uniform_alloc(density, enabled_projs),
            {"kind": "uniform", "sparsity": float(s), "enabled_projs": sorted(enabled_projs), "effective_global_sparsity": effective_s},
        ))
    for scale in [float(x) for x in args.static_schedule_scales.split(",") if x.strip()]:
        alloc, meta = _static_layerproj_alloc_from_perstep(args.perstep_json, args.batch_size, args.group_id, scale)
        candidates.append((f"online_static_layerproj_mean_x{scale:.2f}", float(meta["mean_sparsity"]), alloc, {"kind": "static_layerproj_mean", **meta}, None))
    for spec in [x.strip() for x in args.static_schedule_quantiles.split(",") if x.strip()]:
        if ":" in spec:
            q_text, scale_text = spec.split(":", 1)
            q = float(q_text)
            scale = float(scale_text)
        else:
            q = float(spec)
            scale = 1.0
        alloc, meta = _static_layerproj_quantile_alloc_from_perstep(
            args.perstep_json,
            args.batch_size,
            args.group_id,
            q,
            scale,
            projection_scale_overrides,
            cell_scales,
        )
        proj_suffix = "_projscale" if projection_scale_overrides else ""
        cell_suffix = "_cellscale" if cell_scales else ""
        candidates.append((
            f"online_static_layerproj_q{q:.2f}_x{scale:.2f}{proj_suffix}{cell_suffix}",
            float(meta["mean_sparsity"]),
            alloc,
            meta,
            "collect_prior" if bool(args.quantile_block_frequency_prior) else None,
        ))
    for spec in [x.strip() for x in args.static_schedule_meanstd.split(",") if x.strip()]:
        if ":" in spec:
            lam_text, scale_text = spec.split(":", 1)
            lam = float(lam_text)
            scale = float(scale_text)
        else:
            lam = float(spec)
            scale = 1.0
        alloc, meta = _static_layerproj_meanstd_alloc_from_perstep(args.perstep_json, args.batch_size, args.group_id, lam, scale)
        candidates.append((
            f"online_static_layerproj_mean_minus_{lam:.2f}std_x{scale:.2f}",
            float(meta["mean_sparsity"]),
            alloc,
            meta,
            None,
        ))
    for i, item in enumerate(list(candidates)):
        if len(item) == 4:
            name, effective_sparsity, alloc, meta = item
            candidates[i] = (name, effective_sparsity, alloc, meta, None, None)
        elif len(item) == 5:
            name, effective_sparsity, alloc, meta, priors = item
            candidates[i] = (name, effective_sparsity, alloc, meta, priors, None)

    prior_cache = None
    prior_meta = None
    for scale in [float(x) for x in args.freq_weighted_static_scales.split(",") if x.strip()]:
        alloc, meta = _static_layerproj_alloc_from_perstep(args.perstep_json, args.batch_size, args.group_id, scale)
        if prior_cache is None:
            print(f"[calibration] collect block-frequency prior for scale={scale:.2f}", flush=True)
            prior_cache, prior_meta = _collect_calibration_block_priors(
                model,
                tok,
                args.perstep_json,
                args.batch_size,
                args.group_id,
                args.decode,
                alloc,
                args.device,
            )
            print(json.dumps(prior_meta, indent=2), flush=True)
        candidates.append((
            f"freq_weighted_static_x{scale:.2f}_a{args.prior_alpha:.1f}",
            float(meta["mean_sparsity"]),
            alloc,
            {"kind": "freq_weighted_static_layerproj_mean", **meta, "prior": prior_meta, "prior_alpha": float(args.prior_alpha), "prior_eps": float(args.prior_eps)},
            prior_cache,
        ))
    for scale in [float(x) for x in args.reuse_static_scales.split(",") if x.strip()]:
        alloc, meta = _static_layerproj_alloc_from_perstep(args.perstep_json, args.batch_size, args.group_id, scale)
        candidates.append((
            f"step0_reuse_static_x{scale:.2f}",
            float(meta["mean_sparsity"]),
            alloc,
            {"kind": "step0_reuse_static_layerproj_mean", **meta},
            "reuse",
        ))
    for tau in [float(x) for x in args.warmup_error_taus.split(",") if x.strip()]:
        print(f"[calibration] warmup local linear error tau={tau:g}", flush=True)
        alloc, meta = _warmup_local_error_alloc(
            model,
            full_ids,
            full_am,
            full_starts,
            float(tau),
            max(16, int(INDEX_BLOCK_META_SIZE)),
        )
        cell_rows = meta.pop("cell_rows", [])
        import csv
        cell_csv = args.out_dir / f"warmup_local_error_tau{tau:.3f}_cells.csv"
        if cell_rows:
            with cell_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
                writer.writeheader()
                writer.writerows(cell_rows)
        meta["cell_csv"] = str(cell_csv)
        candidates.append((
            f"warmup_local_error_tau{tau:.3f}",
            float(meta["mean_sparsity"]),
            alloc,
            meta,
            None,
        ))
    logit_candidate_s = [float(x) for x in args.warmup_logit_candidate_s.split(",") if x.strip()]
    for tau in [float(x) for x in args.warmup_logit_kl_taus.split(",") if x.strip()]:
        if args.warmup_logit_cell_csv:
            print(f"[calibration] reuse warmup dense-logit teacher CSV tau={tau:g}", flush=True)
            alloc, meta = _warmup_logit_teacher_alloc_from_csv(args.warmup_logit_cell_csv, float(tau))
        else:
            print(f"[calibration] warmup dense-logit teacher KL tau={tau:g}", flush=True)
            alloc, meta = _warmup_logit_teacher_alloc(
                model,
                full_ids,
                full_am,
                full_starts,
                float(tau),
                max(16, int(INDEX_BLOCK_META_SIZE)),
                logit_candidate_s,
                int(args.warmup_logit_top_vocab),
            )
        cell_rows = meta.pop("cell_rows", [])
        import csv
        cell_csv = args.out_dir / f"warmup_logit_teacher_kl_tau{tau:.5f}_cells.csv"
        if cell_rows:
            with cell_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
                writer.writeheader()
                writer.writerows(cell_rows)
        meta["cell_csv"] = str(cell_csv)
        candidates.append((
            f"warmup_logit_teacher_kl_tau{tau:.5f}",
            float(meta["mean_sparsity"]),
            alloc,
            meta,
            None,
        ))
    interaction_candidate_s = [float(x) for x in args.warmup_interaction_candidate_s.split(",") if x.strip()]
    for tau in [float(x) for x in args.warmup_interaction_logit_kl_taus.split(",") if x.strip()]:
        print(f"[calibration] warmup interaction dense-logit teacher KL tau={tau:g}", flush=True)
        alloc, meta = _warmup_interaction_logit_teacher_alloc(
            model,
            full_ids,
            full_am,
            full_starts,
            float(tau),
            max(16, int(INDEX_BLOCK_META_SIZE)),
            interaction_candidate_s,
            int(args.warmup_interaction_top_vocab),
        )
        cell_rows = meta.pop("cell_rows", [])
        attempt_rows = meta.pop("attempt_rows", [])
        import csv
        cell_csv = args.out_dir / f"warmup_interaction_logit_teacher_tau{tau:.5f}_cells.csv"
        if cell_rows:
            with cell_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
                writer.writeheader()
                writer.writerows(cell_rows)
        attempt_csv = args.out_dir / f"warmup_interaction_logit_teacher_tau{tau:.5f}_attempts.csv"
        if attempt_rows:
            with attempt_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(attempt_rows[0].keys()))
                writer.writeheader()
                writer.writerows(attempt_rows)
        meta["cell_csv"] = str(cell_csv)
        meta["attempt_csv"] = str(attempt_csv)
        candidates.append((
            f"warmup_interaction_logit_teacher_tau{tau:.5f}",
            float(meta["mean_sparsity"]),
            alloc,
            meta,
            None,
        ))

    for name, effective_sparsity, alloc, meta, priors, step_density_alloc in candidates:
        print(f"[2/3] {name}", flush=True)
        if priors == "collect_prior":
            print(f"[calibration] collect block-frequency prior for {name}", flush=True)
            priors, prior_meta_local = _collect_calibration_block_priors(
                model,
                tok,
                args.perstep_json,
                args.batch_size,
                args.group_id,
                args.decode,
                alloc,
                args.device,
            )
            meta = {**meta, "prior": prior_meta_local, "prior_alpha": float(args.prior_alpha), "prior_eps": float(args.prior_eps)}
            name = f"{name}_freqprior_a{args.prior_alpha:.1f}"
        originals, stats = _patch_roundrobin_sparse(
            model,
            alloc,
            sweep,
            bool(args.profile_sparse_parts),
            block_priors=None if priors == "reuse" else priors,
            prior_alpha=float(args.prior_alpha),
            prior_eps=float(args.prior_eps),
            reuse_masks=(priors == "reuse"),
            energy_gate_keep=args.energy_gate_keep,
            energy_gate_keep_by_cell=cell_gates or None,
            energy_gate_mode=str(args.energy_gate_mode),
            energy_gate_expand_projs={x.strip() for x in str(args.energy_gate_expand_projs).split(",") if x.strip()},
            energy_gate_fallback_sparsity_scale=float(args.energy_gate_fallback_sparsity_scale),
            weight_aware_score=bool(args.weight_aware_score),
            step_density_alloc=step_density_alloc,
        )
        rec = _decode_measure_alpaca(model, prompt_ids, prompt_am, last_pos, resp, args.repeats, args.warmup)
        summ = _summarize(rec)
        _restore(originals)
        row = {
            "method": name,
            "sparsity": float(effective_sparsity),
            "schedule_meta": meta,
            "records": rec,
            "summary": summ,
            "tps_speedup_over_dense": float(summ["decode_tps"] / dense_summary["decode_tps"]),
            "latency_speedup_over_dense": float(dense_summary["decode_elapsed_s"] / summ["decode_elapsed_s"]),
            "delta_nll_vs_dense": float(summ["nll"] - dense_summary["nll"]),
            "delta_ppl_vs_dense": float(summ["ppl"] - dense_summary["ppl"]),
            "sparse_kernel_stats": stats,
            "energy_gate_keep": args.energy_gate_keep,
            "energy_gate_cell_csv": str(args.energy_gate_cell_csv) if args.energy_gate_cell_csv else "",
            "energy_gate_mode": str(args.energy_gate_mode),
            "energy_gate_expand_projs": [x.strip() for x in str(args.energy_gate_expand_projs).split(",") if x.strip()],
            "energy_gate_fallback_sparsity_scale": float(args.energy_gate_fallback_sparsity_scale),
            "weight_aware_score": bool(args.weight_aware_score),
        }
        methods.append(row)
        print(json.dumps({k: row[k] for k in ["method", "tps_speedup_over_dense", "delta_nll_vs_dense"]}, indent=2), flush=True)

    out = {
        "config": {**vars(args), "sweep_json": str(args.sweep_json), "perstep_json": str(args.perstep_json)},
        "group": {
            "batch_size": int(args.batch_size),
            "group_id": int(args.group_id),
            "dataset_indices": dataset_indices,
            "prompt_len_min": int(min(r["prompt_len"] for r in rows)),
            "prompt_len_mean": float(sum(r["prompt_len"] for r in rows) / len(rows)),
            "prompt_len_max": int(max(r["prompt_len"] for r in rows)),
            "max_prompt_padded": int(max_prompt),
        },
        "dense_records": dense_records,
        "dense_summary": dense_summary,
        "methods": methods,
    }
    out_path = args.out_dir / "alpaca_uniform_online_topk_e2e.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[3/3] saved", out_path, flush=True)


if __name__ == "__main__":
    main()
