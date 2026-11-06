"""Triton kernels for batch-shared block-sparse matrix multiplication."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl


INDEX_BLOCK_META_SIZE = int(os.environ.get("TEAL_EXP_INDEX_BLOCK_META_SIZE", "64"))
INDEX_SORT_BLOCK_SIZE = int(os.environ.get("TEAL_EXP_INDEX_SORT_BLOCK_SIZE", "64"))
INDEX_BLOCK_OUT_META_SIZE = int(os.environ.get("TEAL_EXP_INDEX_BLOCK_OUT_META_SIZE", "64"))

RUNTIME_DEFAULT_DTYPE = os.environ.get("TEAL_EXP_RUNTIME_DTYPE", "bf16").strip().lower()
if RUNTIME_DEFAULT_DTYPE not in {"bf16", "fp16"}:
    RUNTIME_DEFAULT_DTYPE = "bf16"

# Decode defaults.
USE_DELTA_ACTIVE_BLOCKS_DEFAULT = os.environ.get("TEAL_EXP_USE_DELTA_ACTIVE_BLOCKS", "1") not in ("0", "false", "False")
USE_COMPACT_2D_OUTPUT_DEFAULT = os.environ.get("TEAL_EXP_USE_COMPACT_2D_OUTPUT", "0") not in ("0", "false", "False")
SCATTER_COMPACT_TO_DENSE_DEFAULT = os.environ.get("TEAL_EXP_SCATTER_COMPACT_TO_DENSE", "0") not in ("0", "false", "False")

INDEXED_BLOCK_B = int(os.environ.get("TEAL_EXP_INDEXED_BLOCK_B", "32"))
INDEXED_BLOCK_N = int(os.environ.get("TEAL_EXP_INDEXED_BLOCK_N", "128"))
INDEXED_BLOCK_K = int(os.environ.get("TEAL_EXP_INDEXED_BLOCK_K", "64"))

INDEXED_BLOCK_B_GEMV = int(os.environ.get("TEAL_EXP_INDEXED_BLOCK_B_GEMV", "16"))
INDEXED_BLOCK_N_GEMV = int(os.environ.get("TEAL_EXP_INDEXED_BLOCK_N_GEMV", str(INDEXED_BLOCK_N)))
INDEXED_BLOCK_K_GEMV = int(os.environ.get("TEAL_EXP_INDEXED_BLOCK_K_GEMV", str(INDEXED_BLOCK_K)))

INDEXED_NUM_WARPS = int(os.environ.get("TEAL_EXP_INDEXED_NUM_WARPS", "0"))
INDEXED_NUM_STAGES = int(os.environ.get("TEAL_EXP_INDEXED_NUM_STAGES", "0"))

# Runtime fast-path controls.
SORT_ACTIVE_BLOCKS_DEFAULT = os.environ.get("TEAL_EXP_SORT_ACTIVE_BLOCKS", "1") not in ("0", "false", "False")
# Cheap CPU-bitmask sort: for N <= threshold, sort on CPU to avoid GPU kernel launch overhead.
# TEAL_EXP_CHEAP_SORT_THRESHOLD=0 disables (always use torch.sort on GPU).
_CHEAP_SORT_THRESHOLD = int(os.environ.get("TEAL_EXP_CHEAP_SORT_THRESHOLD", "0"))
# Default ON: active block ids are run-compressed for 1D prepacked path.
USE_RUN_COMPRESSED_METADATA_DEFAULT = os.environ.get("TEAL_EXP_USE_RUN_COMPRESSED_METADATA", "1") not in ("0", "false", "False")
FAST_BLOCK_KERNEL_PATH = os.environ.get("TEAL_EXP_FAST_BLOCK_KERNEL_PATH", "prepack_blocks_2d").strip().lower()
if FAST_BLOCK_KERNEL_PATH not in {"prepack_runs", "prepack_blocks", "prepack_blocks_2d", "prepack_bitmap_tilewalk", "blockmeta"}:
    FAST_BLOCK_KERNEL_PATH = "prepack_blocks_2d"
USE_BLOCK_BASEPTR_METADATA_DEFAULT = os.environ.get("TEAL_EXP_USE_BLOCK_BASEPTR_METADATA", "0") not in ("0", "false", "False")
ENABLE_N_VALID_EARLY_EXIT_DEFAULT = os.environ.get("TEAL_EXP_ENABLE_N_VALID_EARLY_EXIT", "1") not in ("0", "false", "False")
# Default ON: q/k/v/o -> 1D path, gate/up/down -> global path (typically 2D).
ENABLE_PROJ_FAMILY_DISPATCH_DEFAULT = os.environ.get("TEAL_EXP_ENABLE_PROJ_FAMILY_DISPATCH", "1") not in ("0", "false", "False")
ENABLE_PROJ_FAMILY_TUNED_DEFAULT = os.environ.get("TEAL_EXP_ENABLE_PROJ_FAMILY_TUNED", "1") not in ("0", "false", "False")
ENABLE_LEGACY_BLOCKMETA_PATH = os.environ.get("TEAL_EXP_ENABLE_LEGACY_BLOCKMETA_PATH", "0") not in ("0", "false", "False")
ENABLE_SPLIT_K_1D_DEFAULT = os.environ.get("TEAL_EXP_ENABLE_SPLIT_K_1D", "1") not in ("0", "false", "False")
MAX_SPLIT_K_1D = max(1, int(os.environ.get("TEAL_EXP_MAX_SPLIT_K_1D", "4")))
SPLIT_K_1D_MIN_ACTIVE_BLOCKS = max(1, int(os.environ.get("TEAL_EXP_SPLIT_K_1D_MIN_ACTIVE_BLOCKS", "96")))
SPLIT_K_1D_MAX_BT = max(1, int(os.environ.get("TEAL_EXP_SPLIT_K_1D_MAX_BT", "32")))
SPLIT_K_1D_MIN_N_OUT = max(1, int(os.environ.get("TEAL_EXP_SPLIT_K_1D_MIN_N_OUT", "2048")))
_SPLIT_K_1D_REDUCE_MODE = os.environ.get("TEAL_EXP_SPLIT_K_1D_REDUCE_MODE", "auto").strip().lower()
if _SPLIT_K_1D_REDUCE_MODE not in {"auto", "atomic", "two_pass"}:
    _SPLIT_K_1D_REDUCE_MODE = "auto"
ENABLE_SPLIT_K_2D_DEFAULT = os.environ.get("TEAL_EXP_ENABLE_SPLIT_K_2D", "1") not in ("0", "false", "False")
MAX_SPLIT_K_2D = max(1, int(os.environ.get("TEAL_EXP_MAX_SPLIT_K_2D", "4")))
SPLIT_K_2D_MIN_ACTIVE_K_BLOCKS = max(1, int(os.environ.get("TEAL_EXP_SPLIT_K_2D_MIN_ACTIVE_K_BLOCKS", "96")))
SPLIT_K_2D_MAX_BT = max(1, int(os.environ.get("TEAL_EXP_SPLIT_K_2D_MAX_BT", "32")))
SPLIT_K_2D_N_TILE_REPEAT = max(1, int(os.environ.get("TEAL_EXP_SPLIT_K_2D_N_TILE_REPEAT", "1")))
SPLIT_K_2D_K_CHUNK = max(1, int(os.environ.get("TEAL_EXP_SPLIT_K_2D_K_CHUNK", "1")))
ENABLE_SPLIT_K_BPL_DEFAULT = os.environ.get("TEAL_EXP_ENABLE_SPLIT_K_BPL", "1") not in ("0", "false", "False")
MAX_SPLIT_K_BPL = max(1, int(os.environ.get("TEAL_EXP_MAX_SPLIT_K_BPL", "4")))

_INDEX_PACK_CACHE: Dict[str, dict] = {
    "blocks": {},
    "hits": 0,
    "misses": 0,
}
_WEIGHT_PREPACK_CACHE: Dict[str, dict] = {
    "gemm": {},
    "hits": 0,
    "misses": 0,
}
_RUN_META_CACHE: Dict[str, dict] = {
    "runs": {},
    "hits": 0,
    "misses": 0,
}
_BLOCK_BITMAP_CACHE: Dict[str, dict] = {
    "maps": {},
    "hits": 0,
    "misses": 0,
}
_DELTA_BLOCK_STATE_CACHE: Dict[str, dict] = {
    "states": {},
    "hits": 0,
    "misses": 0,
    "updates": 0,
}
_DELTA_N_BLOCK_STATE_CACHE: Dict[str, dict] = {
    "states": {},
    "hits": 0,
    "misses": 0,
    "updates": 0,
}
_SPLITK_ACCUM_CACHE: Dict[str, dict] = {
    "bufs": {},
    "hits": 0,
    "misses": 0,
}
_SPLITK_PARTIAL_CACHE: Dict[str, dict] = {
    "bufs": {},
    "hits": 0,
    "misses": 0,
}
_TEMP_TENSOR_CACHE: Dict[str, dict] = {
    "bufs": {},
    "hits": 0,
    "misses": 0,
}


def _pack_cache_key(prefix: str, mask_u8: torch.Tensor) -> tuple:
    return (
        prefix,
        str(mask_u8.device),
        int(mask_u8.numel()),
        int(mask_u8.data_ptr()),
    )


def _sort_active_idx_by_block(active_idx: torch.Tensor, block_size: int) -> torch.Tensor:
    if int(active_idx.numel()) <= 1 or block_size <= 1:
        return active_idx
    block = torch.div(active_idx, int(block_size), rounding_mode="floor")
    order = torch.argsort(block, stable=True)
    return active_idx.index_select(0, order)


def _sort_active_blocks_asc(active_blocks: torch.Tensor) -> torch.Tensor:
    n = int(active_blocks.numel())
    if n <= 1:
        return active_blocks
    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    # Cheap CPU bitmask sort: for small arrays, avoid GPU kernel launch.
    # Values are block indices in [0, N_K_BLOCKS), so a Python int bitmask suffices.
    # .cpu() synchronizes the GPU, but avoids a separate CUDA kernel launch + its
    # scheduling overhead, which dominates for N<=128 elements.
    if 1 < n <= _CHEAP_SORT_THRESHOLD:
        vals = active_blocks.cpu().tolist()
        mask = 0
        for v in vals:
            mask |= 1 << int(v)
        sorted_vals = []
        tmp = mask
        while tmp:
            lsb = tmp & (-tmp)
            sorted_vals.append(lsb.bit_length() - 1)
            tmp ^= lsb
        return torch.tensor(sorted_vals, dtype=torch.int32, device=active_blocks.device)
    # Default: GPU sort via torch.sort (launches a CUDA kernel).
    return torch.sort(active_blocks.contiguous())[0].contiguous()


def _sort_prefix_keep_shape(active_blocks: torch.Tensor, valid_len: int) -> torch.Tensor:
    """
    Sort only valid prefix [0:valid_len), keep full tensor length unchanged.
    Useful when metadata is padded to a fixed length and only num_active varies.
    """
    n = int(active_blocks.numel())
    v = max(0, min(int(valid_len), n))
    if v <= 1:
        return active_blocks
    if v == n:
        return _sort_active_blocks_asc(active_blocks)
    out = active_blocks.clone()
    out[:v] = torch.sort(out[:v].contiguous())[0]
    return out


def _build_block_start_and_base_tables(
    active_blocks: torch.Tensor,
    valid_len: int,
    block_k: int,
    packed_stride_blk: int,
    sort_prefix: bool,
    keep_padded_shape: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build kernel-friendly K-axis metadata:
      - k_starts: absolute K start index for each active block
      - k_bases:  packed-weight base offset for each active block
    """
    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()
    n = int(active_blocks.numel())
    v = max(0, min(int(valid_len), n))
    if sort_prefix:
        active_blocks = _sort_prefix_keep_shape(active_blocks, v)
    src = active_blocks if keep_padded_shape else active_blocks[:v]
    if int(src.numel()) <= 0:
        empty = torch.empty((0,), device=active_blocks.device, dtype=torch.int32)
        return empty, empty
    k_starts = (src.to(dtype=torch.int32) * int(block_k)).contiguous()
    k_bases = (src.to(dtype=torch.int32) * int(packed_stride_blk)).contiguous()
    return k_starts, k_bases


def _build_n_start_table(
    active_n_blocks: torch.Tensor,
    valid_len: int,
    block_n: int,
    sort_prefix: bool,
    keep_padded_shape: bool = False,
) -> torch.Tensor:
    """
    Build kernel-friendly N-axis metadata:
      - n_starts: absolute N start index for each active output block
    """
    if active_n_blocks.dtype != torch.int32:
        active_n_blocks = active_n_blocks.to(dtype=torch.int32)
    if not active_n_blocks.is_contiguous():
        active_n_blocks = active_n_blocks.contiguous()
    n = int(active_n_blocks.numel())
    v = max(0, min(int(valid_len), n))
    if sort_prefix:
        active_n_blocks = _sort_prefix_keep_shape(active_n_blocks, v)
    src = active_n_blocks if keep_padded_shape else active_n_blocks[:v]
    if int(src.numel()) <= 0:
        return torch.empty((0,), device=active_n_blocks.device, dtype=torch.int32)
    return (src.to(dtype=torch.int32) * int(block_n)).contiguous()


def _compress_active_block_runs(active_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress sorted active block ids into (start_block, run_length_in_blocks).
    """
    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    active_blocks = _sort_active_blocks_asc(active_blocks.contiguous())
    n = int(active_blocks.numel())
    if n <= 0:
        empty = torch.empty((0,), device=active_blocks.device, dtype=torch.int32)
        return empty, empty
    if n == 1:
        return active_blocks, torch.ones((1,), device=active_blocks.device, dtype=torch.int32)

    delta = active_blocks[1:] - active_blocks[:-1]
    cut = torch.nonzero(delta.ne(1), as_tuple=False).flatten().to(dtype=torch.int64) + 1
    starts_idx = torch.cat(
        [torch.zeros((1,), device=active_blocks.device, dtype=torch.int64), cut], dim=0
    )
    ends_idx = torch.cat(
        [cut, torch.tensor([n], device=active_blocks.device, dtype=torch.int64)], dim=0
    )
    run_starts = active_blocks.index_select(0, starts_idx.to(dtype=torch.int32))
    run_lens = (ends_idx - starts_idx).to(dtype=torch.int32)
    return run_starts.contiguous(), run_lens.contiguous()


def _get_cached_active_block_runs(
    active_blocks: torch.Tensor,
    prefix: str,
    sort_blocks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if sort_blocks:
        active_blocks = _sort_active_blocks_asc(active_blocks)
    key = (
        prefix,
        str(active_blocks.device),
        int(active_blocks.numel()),
        int(active_blocks.data_ptr()),
        bool(sort_blocks),
    )
    ent = _RUN_META_CACHE["runs"].get(key)
    if ent is not None:
        _RUN_META_CACHE["hits"] += 1
        return ent["run_starts"], ent["run_lengths"]
    _RUN_META_CACHE["misses"] += 1
    run_starts, run_lengths = _compress_active_block_runs(active_blocks)
    # Keep references directly to avoid clone/copy overhead in hot path.
    _RUN_META_CACHE["runs"][key] = {
        "run_starts": run_starts,
        "run_lengths": run_lengths,
    }
    return run_starts, run_lengths


def _get_cached_active_block_bitmap(
    active_blocks: torch.Tensor,
    num_total_blocks: int,
    prefix: str,
) -> torch.Tensor:
    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()
    key = (
        prefix,
        str(active_blocks.device),
        int(active_blocks.data_ptr()),
        int(active_blocks.numel()),
        int(num_total_blocks),
    )
    ent = _BLOCK_BITMAP_CACHE["maps"].get(key)
    if ent is not None:
        _BLOCK_BITMAP_CACHE["hits"] += 1
        return ent["bitmap"]
    _BLOCK_BITMAP_CACHE["misses"] += 1
    nt = max(0, int(num_total_blocks))
    bm = torch.zeros((nt,), device=active_blocks.device, dtype=torch.uint8)
    if nt > 0 and int(active_blocks.numel()) > 0:
        ab = active_blocks.to(dtype=torch.long)
        ab = ab[(ab >= 0) & (ab < nt)]
        if int(ab.numel()) > 0:
            bm.index_fill_(0, ab, 1)
    _BLOCK_BITMAP_CACHE["maps"][key] = {"bitmap": bm.detach().clone()}
    return bm


def _active_blocks_from_mask(mask_u8: torch.Tensor, block_size: int) -> torch.Tensor:
    m = int(mask_u8.numel())
    if m <= 0:
        return torch.empty((0,), device=mask_u8.device, dtype=torch.int32)
    block = max(1, int(block_size))
    n_blocks = (m + block - 1) // block
    if (m % block) != 0:
        pad = torch.zeros((n_blocks * block,), device=mask_u8.device, dtype=torch.uint8)
        pad[:m] = mask_u8
        mask_u8 = pad
    block_nonempty = mask_u8.view(n_blocks, block).sum(dim=1).gt(0)
    active_blocks = torch.nonzero(block_nonempty, as_tuple=False).flatten().to(dtype=torch.int32)
    return _sort_active_blocks_asc(active_blocks)


def _get_cached_active_blocks(mask_u8: torch.Tensor, prefix: str, block_size: int) -> torch.Tensor:
    if bool(USE_DELTA_ACTIVE_BLOCKS_DEFAULT):
        return _get_cached_active_blocks_delta(mask_u8=mask_u8, prefix=prefix, block_size=block_size)
    key = _pack_cache_key(prefix, mask_u8) + (int(block_size),)
    ent = _INDEX_PACK_CACHE["blocks"].get(key)
    if ent is not None:
        _INDEX_PACK_CACHE["hits"] += 1
        return _sort_active_blocks_asc(ent["active_blocks"])
    _INDEX_PACK_CACHE["misses"] += 1
    active_blocks = _active_blocks_from_mask(mask_u8, block_size)
    active_blocks = _sort_active_blocks_asc(active_blocks)
    _INDEX_PACK_CACHE["blocks"][key] = {"active_blocks": active_blocks.detach().clone()}
    return active_blocks


def _delta_state_key(prefix: str, mask_u8: torch.Tensor, block_size: int) -> tuple:
    return (
        prefix,
        str(mask_u8.device),
        int(mask_u8.numel()),
        int(block_size),
    )


def _init_delta_state_from_mask(mask_u8: torch.Tensor, block_size: int) -> dict:
    m = int(mask_u8.numel())
    block = max(1, int(block_size))
    n_blocks = (m + block - 1) // block
    padded = mask_u8
    if (m % block) != 0:
        padded = torch.zeros((n_blocks * block,), device=mask_u8.device, dtype=torch.uint8)
        padded[:m] = mask_u8
    counts = padded.view(n_blocks, block).sum(dim=1).to(dtype=torch.int32).contiguous()
    active_blocks = torch.nonzero(counts.gt(0), as_tuple=False).flatten().to(dtype=torch.int32).contiguous()
    return {
        "prev_mask": mask_u8.detach().clone(),
        "counts": counts,
        "active_blocks": _sort_active_blocks_asc(active_blocks),
        "block_size": int(block),
    }


def _get_cached_active_blocks_delta(mask_u8: torch.Tensor, prefix: str, block_size: int) -> torch.Tensor:
    key = _delta_state_key(prefix, mask_u8, block_size)
    st = _DELTA_BLOCK_STATE_CACHE["states"].get(key)
    if st is None:
        _DELTA_BLOCK_STATE_CACHE["misses"] += 1
        st = _init_delta_state_from_mask(mask_u8=mask_u8, block_size=block_size)
        _DELTA_BLOCK_STATE_CACHE["states"][key] = st
        return st["active_blocks"]

    _DELTA_BLOCK_STATE_CACHE["hits"] += 1
    prev_mask = st["prev_mask"]
    if prev_mask.shape != mask_u8.shape or prev_mask.device != mask_u8.device:
        st = _init_delta_state_from_mask(mask_u8=mask_u8, block_size=block_size)
        _DELTA_BLOCK_STATE_CACHE["states"][key] = st
        return st["active_blocks"]

    changed = torch.nonzero(mask_u8.ne(prev_mask), as_tuple=False).flatten().to(dtype=torch.int64)
    if int(changed.numel()) == 0:
        return st["active_blocks"]

    _DELTA_BLOCK_STATE_CACHE["updates"] += 1
    block = int(st["block_size"])
    old_vals = prev_mask.index_select(0, changed).to(dtype=torch.int32)
    new_vals = mask_u8.index_select(0, changed).to(dtype=torch.int32)
    delta_vals = (new_vals - old_vals).to(dtype=torch.int32)
    nz = torch.nonzero(delta_vals.ne(0), as_tuple=False).flatten()
    if int(nz.numel()) > 0:
        changed_nz = changed.index_select(0, nz)
        delta_nz = delta_vals.index_select(0, nz)
        block_ids = torch.div(changed_nz, int(block), rounding_mode="floor").to(dtype=torch.int64)
        uniq_blocks, inv = torch.unique(block_ids, sorted=False, return_inverse=True)
        blk_delta = torch.zeros((int(uniq_blocks.numel()),), device=mask_u8.device, dtype=torch.int32)
        blk_delta.scatter_add_(0, inv, delta_nz)
        counts = st["counts"]
        counts.index_add_(0, uniq_blocks, blk_delta)
        counts.clamp_(min=0, max=int(block))
        st["active_blocks"] = _sort_active_blocks_asc(
            torch.nonzero(counts.gt(0), as_tuple=False).flatten().to(dtype=torch.int32).contiguous()
        )
    prev_mask.index_copy_(0, changed, mask_u8.index_select(0, changed))
    return st["active_blocks"]


def _delta_n_state_key(prefix: str, mask_u8: torch.Tensor, block_size: int) -> tuple:
    return (
        prefix,
        str(mask_u8.device),
        int(mask_u8.numel()),
        int(block_size),
    )


def _init_delta_n_state_from_mask(mask_u8: torch.Tensor, block_size: int) -> dict:
    m = int(mask_u8.numel())
    block = max(1, int(block_size))
    n_blocks = (m + block - 1) // block
    padded = mask_u8
    if (m % block) != 0:
        padded = torch.zeros((n_blocks * block,), device=mask_u8.device, dtype=torch.uint8)
        padded[:m] = mask_u8
    counts = padded.view(n_blocks, block).sum(dim=1).to(dtype=torch.int32).contiguous()
    active_blocks = torch.nonzero(counts.gt(0), as_tuple=False).flatten().to(dtype=torch.int32).contiguous()
    return {
        "prev_mask": mask_u8.detach().clone(),
        "counts": counts,
        "active_blocks": _sort_active_blocks_asc(active_blocks),
        "block_size": int(block),
    }


def _get_cached_active_n_blocks_delta(mask_u8: torch.Tensor, prefix: str, block_size: int) -> torch.Tensor:
    key = _delta_n_state_key(prefix, mask_u8, block_size)
    st = _DELTA_N_BLOCK_STATE_CACHE["states"].get(key)
    if st is None:
        _DELTA_N_BLOCK_STATE_CACHE["misses"] += 1
        st = _init_delta_n_state_from_mask(mask_u8=mask_u8, block_size=block_size)
        _DELTA_N_BLOCK_STATE_CACHE["states"][key] = st
        return st["active_blocks"]

    _DELTA_N_BLOCK_STATE_CACHE["hits"] += 1
    prev_mask = st["prev_mask"]
    if prev_mask.shape != mask_u8.shape or prev_mask.device != mask_u8.device:
        st = _init_delta_n_state_from_mask(mask_u8=mask_u8, block_size=block_size)
        _DELTA_N_BLOCK_STATE_CACHE["states"][key] = st
        return st["active_blocks"]

    changed = torch.nonzero(mask_u8.ne(prev_mask), as_tuple=False).flatten().to(dtype=torch.int64)
    if int(changed.numel()) == 0:
        return st["active_blocks"]

    _DELTA_N_BLOCK_STATE_CACHE["updates"] += 1
    block = int(st["block_size"])
    old_vals = prev_mask.index_select(0, changed).to(dtype=torch.int32)
    new_vals = mask_u8.index_select(0, changed).to(dtype=torch.int32)
    delta_vals = (new_vals - old_vals).to(dtype=torch.int32)
    nz = torch.nonzero(delta_vals.ne(0), as_tuple=False).flatten()
    if int(nz.numel()) > 0:
        changed_nz = changed.index_select(0, nz)
        delta_nz = delta_vals.index_select(0, nz)
        block_ids = torch.div(changed_nz, int(block), rounding_mode="floor").to(dtype=torch.int64)
        uniq_blocks, inv = torch.unique(block_ids, sorted=False, return_inverse=True)
        blk_delta = torch.zeros((int(uniq_blocks.numel()),), device=mask_u8.device, dtype=torch.int32)
        blk_delta.scatter_add_(0, inv, delta_nz)
        counts = st["counts"]
        counts.index_add_(0, uniq_blocks, blk_delta)
        counts.clamp_(min=0, max=int(block))
        st["active_blocks"] = _sort_active_blocks_asc(
            torch.nonzero(counts.gt(0), as_tuple=False).flatten().to(dtype=torch.int32).contiguous()
        )

    prev_mask.index_copy_(0, changed, mask_u8.index_select(0, changed))
    return st["active_blocks"]


def _active_blocks_delta_update(
    prev_active_blocks: torch.Tensor,
    add_idx: torch.Tensor | None,
    remove_idx: torch.Tensor | None,
    block_size: int,
) -> torch.Tensor:
    """
    Delta update helper for decode runtime:
    update active blocks using only add/remove index sets.
    """
    block = max(1, int(block_size))
    if prev_active_blocks.dtype != torch.int32:
        prev_active_blocks = prev_active_blocks.to(dtype=torch.int32)
    prev_active_blocks = _sort_active_blocks_asc(prev_active_blocks.contiguous())

    def _to_blocks(idx: torch.Tensor | None) -> torch.Tensor:
        if idx is None or int(idx.numel()) <= 0:
            return torch.empty((0,), device=prev_active_blocks.device, dtype=torch.int32)
        if idx.dtype != torch.int64 and idx.dtype != torch.int32:
            idx = idx.to(dtype=torch.int64)
        blocks = torch.div(idx.to(dtype=torch.int64), int(block), rounding_mode="floor").to(dtype=torch.int32)
        return torch.unique(blocks, sorted=True)

    add_blocks = _to_blocks(add_idx)
    remove_blocks = _to_blocks(remove_idx)

    if int(add_blocks.numel()) <= 0 and int(remove_blocks.numel()) <= 0:
        return prev_active_blocks

    keep_mask = torch.ones((int(prev_active_blocks.numel()),), device=prev_active_blocks.device, dtype=torch.bool)
    if int(remove_blocks.numel()) > 0:
        keep_mask = ~torch.isin(prev_active_blocks, remove_blocks)
    kept = prev_active_blocks[keep_mask]
    if int(add_blocks.numel()) > 0:
        merged = torch.cat([kept, add_blocks], dim=0)
        merged = torch.unique(merged, sorted=True)
        return merged.to(dtype=torch.int32).contiguous()
    return kept.to(dtype=torch.int32).contiguous()


def _indexed_launch_meta(bsz: int, n_dim: int, k_dim: int) -> tuple[int, int, int, int, int]:
    block_b = INDEXED_BLOCK_B_GEMV
    block_n = INDEXED_BLOCK_N_GEMV
    block_k = INDEXED_BLOCK_K_GEMV
    if bsz < block_b:
        block_b = 16 if bsz <= 16 else 32
    num_warps = 8 if (block_n >= 256 or block_b >= 64) else 4
    if INDEXED_NUM_WARPS > 0:
        num_warps = INDEXED_NUM_WARPS
    num_stages = 2 if INDEXED_NUM_STAGES <= 0 else INDEXED_NUM_STAGES
    return block_b, block_n, block_k, num_warps, num_stages


@triton.jit
def indexed_sparse_gemm_batched_kernel(
    Y,
    A,
    X,
    ACTIVE_IDX,
    K,
    N_OUT,
    M,
    A_STRIDE_ROW,
    A_STRIDE_COL,
    A_ROW_OFFSET,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    k0 = 0
    while k0 < K:
        rk = k0 + tl.arange(0, BLOCK_K)
        idx = tl.load(ACTIVE_IDX + rk, mask=rk < K, other=0).to(tl.int32)

        x_ptrs = X + rb[:, None] * M + idx[None, :]
        a_ptrs = A + idx[:, None] * A_STRIDE_COL + (A_ROW_OFFSET + rn[None, :]) * A_STRIDE_ROW

        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & (rk[None, :] < K), other=0.0)
        a = tl.load(a_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        k0 += BLOCK_K

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.jit
def indexed_sparse_blockmeta_gemm_batched_kernel(
    Y,
    A,
    X,
    MASK_U8,
    ACTIVE_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_STRIDE_ROW,
    A_STRIDE_COL,
    A_ROW_OFFSET,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_BLOCKS:
        block_id = tl.load(ACTIVE_BLOCKS + block_it).to(tl.int32)
        rk = block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)
        mask_vals = tl.load(MASK_U8 + rk, mask=valid_k, other=0).to(tl.int32)
        x = tl.where(mask_vals[None, :] != 0, x, tl.zeros_like(x))

        a_ptrs = A + rk[:, None] * A_STRIDE_COL + (A_ROW_OFFSET + rn[None, :]) * A_STRIDE_ROW
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


_PREPACK_AUTOTUNE_CONFIGS = [
    # Small/skinny decode shapes.
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 32},  num_warps=2, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 32},  num_warps=2, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 32},  num_warps=2, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 32},  num_warps=2, num_stages=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=2, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=2, num_stages=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=5),
    # Larger batched/throughput-heavy shapes.
    triton.Config({"BLOCK_B": 64, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": 64},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    # Very large batch (B=256~512).
    triton.Config({"BLOCK_B": 256, "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 256, "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 256, "BLOCK_N": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 512, "BLOCK_N": 32},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 512, "BLOCK_N": 64},  num_warps=8, num_stages=2),
]

_PREPACK_2D_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 128, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=8, num_stages=4),
]


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_batched_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_BLOCKS:
        block_id = tl.load(ACTIVE_BLOCKS + block_it).to(tl.int32)
        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.jit
def prepacked_block_sparse_gemm_batched_kernel_explicit(
    Y,
    A_PACKED,
    X,
    ACTIVE_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_BLOCKS:
        block_id = tl.load(ACTIVE_BLOCKS + block_it).to(tl.int32)
        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


# Bitmap-addressed static-loop kernel.

_PREPACK_BITMAP_PRELOAD_AUTOTUNE_CONFIGS = [
    # Standard shapes.
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=2, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    # Very large batch (B=256~512).
    triton.Config({"BLOCK_B": 256, "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 256, "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 256, "BLOCK_N": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 512, "BLOCK_N": 32},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 512, "BLOCK_N": 64},  num_warps=8, num_stages=2),
]


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_bitmap_preload_batched_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BITMAP,       # u8 bitmap: shape [N_TOTAL_K_BLOCKS], 1=active 0=inactive
    NUM_ACTIVE_BLOCKS,     # runtime: number of set bits in bitmap
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_TOTAL_K_BLOCKS: tl.constexpr,  # constexpr: compile-time loop bound for pipelining
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    # tile addresses are affine in block_id (block_id * stride + base),
    # so Triton num_stages can pipeline loads across iterations.
    for block_id in range(N_TOTAL_K_BLOCKS):
        # Scalar load with compile-time-known address: no serial dependency
        is_active = tl.load(ACTIVE_K_BITMAP + block_id) != 0

        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(
            x_ptrs,
            mask=is_active & (rb[:, None] < BATCHSIZE) & valid_k[None, :],
            other=0.0,
        )

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=is_active & valid_k[:, None] & (rn[None, :] < N_OUT),
            other=0.0,
        )
        acc += tl.dot(x, a)

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.jit
def prepacked_block_sparse_gemm_bitmap_preload_batched_kernel_explicit(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BITMAP,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_TOTAL_K_BLOCKS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for block_id in range(N_TOTAL_K_BLOCKS):
        is_active = tl.load(ACTIVE_K_BITMAP + block_id) != 0

        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(
            x_ptrs,
            mask=is_active & (rb[:, None] < BATCHSIZE) & valid_k[None, :],
            other=0.0,
        )

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=is_active & valid_k[:, None] & (rn[None, :] < N_OUT),
            other=0.0,
        )
        acc += tl.dot(x, a)

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.autotune(
    configs=_PREPACK_BITMAP_PRELOAD_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_bitmap_regpreload_batched_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BITMAP,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_TOTAL_K_BLOCKS: tl.constexpr,
    PADDED_K_BLOCKS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    # Preload entire bitmap as a register vector (one coalesced load before loop).
    bm_offs = tl.arange(0, PADDED_K_BLOCKS)
    bm = tl.load(ACTIVE_K_BITMAP + bm_offs, mask=bm_offs < N_TOTAL_K_BLOCKS, other=0)

    for block_id in range(N_TOTAL_K_BLOCKS):
        # Triton does not support tensor[scalar] extraction; use masked sum instead.
        is_active = tl.sum(bm.to(tl.int32) * (bm_offs == block_id).to(tl.int32)) != 0

        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(
            x_ptrs,
            mask=is_active & (rb[:, None] < BATCHSIZE) & valid_k[None, :],
            other=0.0,
        )

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=is_active & valid_k[:, None] & (rn[None, :] < N_OUT),
            other=0.0,
        )
        acc += tl.dot(x, a)

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


def _prepacked_block_sparse_gemm_from_active_blocks_bitmap_regpreload(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_bpl_rp",
    total_k_blocks: int | None = None,
    prebuilt_bitmap: torch.Tensor | None = None,
) -> None:
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_active = int(active_blocks.numel())
    if num_active <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    if total_k_blocks is None:
        total_k_blocks = (m_dim + block_k - 1) // block_k

    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if prebuilt_bitmap is not None:
        bitmap = prebuilt_bitmap
    else:
        bitmap = _build_active_k_bitmap(active_blocks, total_k_blocks, x.device)

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    padded_k = 1
    while padded_k < total_k_blocks:
        padded_k <<= 1

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )
    prepacked_block_sparse_gemm_bitmap_regpreload_batched_kernel[grid](
        y2d, packed, x2d, bitmap, num_active,
        n_dim, m_dim,
        packed.stride(0), packed.stride(1), packed.stride(2),
        bt,
        BLOCK_K=block_k,
        N_TOTAL_K_BLOCKS=total_k_blocks,
        PADDED_K_BLOCKS=padded_k,
    )


USE_PRELOADED_BITMAP_KERNEL = os.environ.get("TEAL_EXP_USE_PRELOADED_BITMAP_KERNEL", "0") not in ("0", "false", "False")


def _build_active_k_bitmap(active_blocks: torch.Tensor, total_k_blocks: int, device) -> torch.Tensor:
    """Build u8 bitmap from sorted active block index list."""
    bitmap = torch.zeros(total_k_blocks, dtype=torch.uint8, device=device)
    if int(active_blocks.numel()) > 0:
        bitmap[active_blocks.to(torch.long)] = 1
    return bitmap


def _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_bpl",
    total_k_blocks: int | None = None,
    prebuilt_bitmap: torch.Tensor | None = None,
) -> None:
    """
    Optimized 1D sparse GEMM using bitmap preload + static loop unrolling.
    Drop-in replacement for _prepacked_block_sparse_gemm_from_active_blocks
    at low active-block counts.

    Requirements:
    - active_blocks: sorted int32 tensor of active K-block indices
    - total_k_blocks must be small enough for static loop unrolling (<=128 recommended)

    For CUDA graph compatibility: pass prebuilt_bitmap (a pre-allocated uint8 tensor
    of shape [total_k_blocks]) so bitmap construction happens outside the captured region.
    """
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_active = int(active_blocks.numel())
    if num_active <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    if total_k_blocks is None:
        total_k_blocks = (m_dim + block_k - 1) // block_k

    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    # Build bitmap (small, fits in L1 cache permanently).
    # If prebuilt_bitmap is provided (e.g., for CUDA graph capture where
    # fancy-index scatter is not allowed inside the capture region), use it directly.
    if prebuilt_bitmap is not None:
        bitmap = prebuilt_bitmap
    else:
        bitmap = _build_active_k_bitmap(active_blocks, total_k_blocks, x.device)

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )

    prepacked_block_sparse_gemm_bitmap_preload_batched_kernel[grid](
        y2d,
        packed,
        x2d,
        bitmap,
        num_active,
        n_dim,
        m_dim,
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        bt,
        BLOCK_K=block_k,
        N_TOTAL_K_BLOCKS=total_k_blocks,
    )


# ---------------------------------------------------------------------------
# Bitmap-preload split-K kernel
#
# Splits K across NUM_SPLIT_K CTAs (program_id(2)).
# pid_sk handles K-blocks [pid_sk*CHUNK .. (pid_sk+1)*CHUNK).
# Both N_TOTAL_K_BLOCKS and NUM_SPLIT_K are constexpr → CHUNK is constexpr
# → inner loop bound is compile-time-known → num_stages pipelining works.
# ---------------------------------------------------------------------------

@triton.jit
def prepacked_block_sparse_gemm_bitmap_preload_splitk_kernel(
    Y_PARTIAL,          # float32 [NUM_SPLIT_K, bt, n_out]
    A_PACKED,
    X,
    ACTIVE_K_BITMAP,    # u8 [N_TOTAL_K_BLOCKS]
    N_OUT,
    M,
    YP_STRIDE_SK,
    YP_STRIDE_B,
    YP_STRIDE_N,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_TOTAL_K_BLOCKS: tl.constexpr,
    NUM_SPLIT_K: tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_b  = tl.program_id(1)
    pid_sk = tl.program_id(2)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    CHUNK_SIZE: tl.constexpr = (N_TOTAL_K_BLOCKS + NUM_SPLIT_K - 1) // NUM_SPLIT_K
    start = pid_sk * CHUNK_SIZE

    for i in range(CHUNK_SIZE):
        block_id = start + i
        in_range = block_id < N_TOTAL_K_BLOCKS
        is_active = tl.load(ACTIVE_K_BITMAP + block_id, mask=in_range, other=0) != 0

        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = in_range & (rk < M)

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(
            x_ptrs,
            mask=is_active & (rb[:, None] < BATCHSIZE) & valid_k[None, :],
            other=0.0,
        )
        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=is_active & valid_k[:, None] & (rn[None, :] < N_OUT),
            other=0.0,
        )
        acc += tl.dot(x, a)

    y_ptrs = (
        Y_PARTIAL
        + pid_sk * YP_STRIDE_SK
        + rb[:, None] * YP_STRIDE_B
        + rn[None, :] * YP_STRIDE_N
    )
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


def _choose_split_k_bpl(bt: int, total_k_blocks: int, n_out: int) -> int:
    if not bool(ENABLE_SPLIT_K_BPL_DEFAULT):
        return 1
    block_n_approx = max(16, int(INDEX_BLOCK_OUT_META_SIZE))
    n_tiles = (int(n_out) + block_n_approx - 1) // block_n_approx
    b_tiles = max(1, (int(bt) + 15) // 16)
    current_ctas = n_tiles * b_tiles
    N_SM = torch.cuda.get_device_properties(0).multi_processor_count
    target_ctas = N_SM * 2
    if current_ctas >= target_ctas:
        return 1
    sk = max(1, (target_ctas + current_ctas - 1) // current_ctas)
    max_sk = max(1, int(total_k_blocks) // 8)  # min 8 K-blocks per split
    return min(sk, max_sk, int(MAX_SPLIT_K_BPL))


def _prepacked_block_sparse_gemm_from_active_blocks_bitmap_preload_splitk(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_bpl_sk",
    total_k_blocks: int | None = None,
    prebuilt_bitmap: torch.Tensor | None = None,
    num_split_k: int | None = None,
    tile_b: int | None = None,
    tile_n: int | None = None,
    tile_warps: int | None = None,
    tile_stages: int | None = None,
) -> None:
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_active = int(active_blocks.numel())
    if num_active <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    if total_k_blocks is None:
        total_k_blocks = (m_dim + block_k - 1) // block_k

    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    if prebuilt_bitmap is not None:
        bitmap = prebuilt_bitmap
    else:
        bitmap = _build_active_k_bitmap(active_blocks, total_k_blocks, x.device)

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    if num_split_k is None:
        num_split_k = _choose_split_k_bpl(bt, total_k_blocks, n_dim)

    if num_split_k <= 1:
        grid = lambda META: (
            triton.cdiv(n_dim, META["BLOCK_N"]),
            triton.cdiv(bt, META["BLOCK_B"]),
        )
        prepacked_block_sparse_gemm_bitmap_preload_batched_kernel[grid](
            y2d, packed, x2d, bitmap, num_active,
            n_dim, m_dim,
            packed.stride(0), packed.stride(1), packed.stride(2),
            bt, BLOCK_K=block_k, N_TOTAL_K_BLOCKS=total_k_blocks,
        )
        return

    split_block_n = tile_n     if tile_n     is not None else max(16, int(INDEX_BLOCK_OUT_META_SIZE))
    split_b       = tile_b     if tile_b     is not None else 16
    split_warps   = tile_warps  if tile_warps  is not None else 2
    split_stages  = tile_stages if tile_stages is not None else 3

    y_part = _get_splitk_partial_buffer(
        prefix=f"{cache_prefix}_bplsk_part",
        num_split_k=num_split_k,
        bt=bt,
        n_dim=n_dim,
        device=out.device,
    )

    grid_sk = (triton.cdiv(n_dim, split_block_n), triton.cdiv(bt, split_b), num_split_k)
    prepacked_block_sparse_gemm_bitmap_preload_splitk_kernel[grid_sk](
        y_part,
        packed,
        x2d,
        bitmap,
        n_dim,
        m_dim,
        y_part.stride(0),
        y_part.stride(1),
        y_part.stride(2),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        bt,
        BLOCK_B=split_b,
        BLOCK_N=split_block_n,
        BLOCK_K=block_k,
        N_TOTAL_K_BLOCKS=total_k_blocks,
        NUM_SPLIT_K=num_split_k,
        num_warps=split_warps,
        num_stages=split_stages,
    )

    grid_reduce = (triton.cdiv(n_dim, split_block_n), triton.cdiv(bt, split_b))
    prepacked_block_sparse_gemm_splitk_reduce_kernel[grid_reduce](
        y2d,
        y_part,
        num_split_k,
        n_dim,
        y_part.stride(0),
        y_part.stride(1),
        y_part.stride(2),
        bt,
        BLOCK_B=split_b,
        BLOCK_N=split_block_n,
        num_warps=split_warps,
        num_stages=split_stages,
    )


def _prepacked_block_sparse_gemm_from_active_blocks_splitk(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_1d_sk",
    num_split_k: int | None = None,
    tile_b: int | None = None,
    tile_n: int | None = None,
    tile_warps: int | None = None,
    tile_stages: int | None = None,
) -> None:
    """Naive 1D sparse GEMM with split-K. Uses _choose_split_k_bpl for split factor."""
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_blocks = int(active_blocks.numel())
    if num_blocks <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    total_k_blocks = (m_dim + block_k - 1) // block_k
    if num_split_k is None:
        num_split_k = _choose_split_k_bpl(bt, total_k_blocks, n_dim)

    split_block_n = tile_n     if tile_n     is not None else max(16, int(INDEX_BLOCK_OUT_META_SIZE))
    split_b       = tile_b     if tile_b     is not None else 16
    split_warps   = tile_warps  if tile_warps  is not None else 2
    split_stages  = tile_stages if tile_stages is not None else 3

    if num_split_k <= 1:
        grid = lambda META: (
            triton.cdiv(n_dim, META["BLOCK_N"]),
            triton.cdiv(bt, META["BLOCK_B"]),
        )
        prepacked_block_sparse_gemm_batched_kernel[grid](
            y2d, packed, x2d, active_blocks, num_blocks,
            n_dim, m_dim,
            packed.stride(0), packed.stride(1), packed.stride(2),
            bt, BLOCK_K=block_k,
        )
        return

    y_part = _get_splitk_partial_buffer(
        prefix=f"{cache_prefix}_1dsk_part",
        num_split_k=num_split_k,
        bt=bt,
        n_dim=n_dim,
        device=out.device,
    )

    grid_write = (triton.cdiv(n_dim, split_block_n), triton.cdiv(bt, split_b), num_split_k)
    prepacked_block_sparse_gemm_splitk_write_kernel[grid_write](
        y_part,
        packed,
        x2d,
        active_blocks,
        num_blocks,
        num_split_k,
        n_dim,
        m_dim,
        y_part.stride(0),
        y_part.stride(1),
        y_part.stride(2),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        bt,
        BLOCK_B=split_b,
        BLOCK_N=split_block_n,
        BLOCK_K=block_k,
        num_warps=split_warps,
        num_stages=split_stages,
    )

    grid_reduce = (triton.cdiv(n_dim, split_block_n), triton.cdiv(bt, split_b))
    prepacked_block_sparse_gemm_splitk_reduce_kernel[grid_reduce](
        y2d,
        y_part,
        num_split_k,
        n_dim,
        y_part.stride(0),
        y_part.stride(1),
        y_part.stride(2),
        bt,
        BLOCK_B=split_b,
        BLOCK_N=split_block_n,
        num_warps=split_warps,
        num_stages=split_stages,
    )


# Sorted scan over all K blocks; inactive blocks are skipped uniformly.

_PREPACK_SORTED_SCAN_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=5),
]


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_sorted_scan_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BITMAP,       # u8 [N_TOTAL_K_BLOCKS]: 1=active, 0=inactive
    NUM_ACTIVE_BLOCKS,     # runtime, for autotune key
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_TOTAL_K_BLOCKS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for block_id in tl.static_range(N_TOTAL_K_BLOCKS):
        # Compile-time address → GPU can pipeline all N_TOTAL_K_BLOCKS bitmap
        # byte reads. Result is a uniform scalar (same for all threads).
        is_active = tl.load(ACTIVE_K_BITMAP + block_id)

        # Uniform branch: all threads in the CTA take the same path.
        # When inactive: entire block body is skipped (no tl.dot, no memory).
        if is_active:
            rk_local = tl.arange(0, BLOCK_K)
            rk = block_id * BLOCK_K + rk_local
            valid_k = rk < M

            x_ptrs = X + rb[:, None] * M + rk[None, :]
            x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

            a_ptrs = (
                A_PACKED
                + block_id * A_PACK_STRIDE_BLK
                + rk_local[:, None] * A_PACK_STRIDE_K
                + rn[None, :] * A_PACK_STRIDE_N
            )
            a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
            acc += tl.dot(x, a)

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


def _prepacked_block_sparse_gemm_from_active_blocks_sorted_scan(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_ss",
    total_k_blocks: int | None = None,
    prebuilt_bitmap: torch.Tensor | None = None,
) -> None:
    """
    Sorted-scan 1D sparse GEMM.
    active_blocks must be sorted ascending (guaranteed by .sort()[0]).
    Iterates all K-blocks in order with `if is_active:` uniform branch →
    truly skips inactive blocks, not just predicated zeros.
    """
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_active = int(active_blocks.numel())
    if num_active <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    if total_k_blocks is None:
        total_k_blocks = (m_dim + block_k - 1) // block_k

    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    if prebuilt_bitmap is not None:
        bitmap = prebuilt_bitmap
    else:
        bitmap = _build_active_k_bitmap(active_blocks, total_k_blocks, x.device)

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )

    prepacked_block_sparse_gemm_sorted_scan_kernel[grid](
        y2d, packed, x2d, bitmap,
        num_active, n_dim, m_dim,
        packed.stride(0), packed.stride(1), packed.stride(2),
        bt,
        BLOCK_K=block_k,
        N_TOTAL_K_BLOCKS=total_k_blocks,
    )


# Preload active block IDs to avoid a scalar dependency chain.

_PREPACK_PRELOAD_IDS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=5),
]


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_doublebuf_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Double-buffer prefetch: issue ACTIVE_BLOCKS[i+1] load at the start of
    iteration i so it arrives before iteration i+1 needs it.
    Reduces block_id load stall but weight tile load dependency remains.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_id = tl.load(ACTIVE_BLOCKS).to(tl.int32)

    block_it = 0
    while block_it < NUM_ACTIVE_BLOCKS:
        next_block_id = tl.load(
            ACTIVE_BLOCKS + block_it + 1,
            mask=block_it + 1 < NUM_ACTIVE_BLOCKS,
            other=0,
        ).to(tl.int32)

        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)

        block_id = next_block_id
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


def _prepacked_block_sparse_gemm_from_active_blocks_doublebuf(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_db",
) -> None:
    """Double-buffer prefetch: ACTIVE_BLOCKS[i+1] issued during iteration i."""
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_active = int(active_blocks.numel())
    if num_active <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )

    prepacked_block_sparse_gemm_doublebuf_kernel[grid](
        y2d, packed, x2d, active_blocks,
        num_active, n_dim, m_dim,
        packed.stride(0), packed.stride(1), packed.stride(2),
        bt,
        BLOCK_K=block_k,
    )


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_preload_ids_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_BLOCKS,         # int32 [N_TOTAL_K_BLOCKS]: padded to power-of-2
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_TOTAL_K_BLOCKS: tl.constexpr,
):
    """
    Preload ALL block IDs into registers via ONE vectorized load at kernel start.
    Loop body accesses block_id via register arithmetic (no global memory per iter).
    Purpose: isolate whether ACTIVE_BLOCKS load is the bottleneck.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    scan = tl.arange(0, N_TOTAL_K_BLOCKS)
    preloaded_ids = tl.load(
        ACTIVE_BLOCKS + scan,
        mask=scan < NUM_ACTIVE_BLOCKS,
        other=0,
    ).to(tl.int32)

    block_it = 0
    while block_it < NUM_ACTIVE_BLOCKS:
        block_id = tl.sum(tl.where(scan == block_it, preloaded_ids, tl.zeros_like(preloaded_ids)))

        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


def _prepacked_block_sparse_gemm_from_active_blocks_preload_ids(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_pi",
    total_k_blocks: int | None = None,
) -> None:
    """
    Preload ALL block IDs into registers at kernel start.
    Eliminates ACTIVE_BLOCKS global memory access inside loop entirely.
    Used to isolate whether ACTIVE_BLOCKS load contributes to Long Scoreboard stall.
    N_TOTAL_K_BLOCKS must be power-of-2 (Triton reduction constraint).
    """
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_active = int(active_blocks.numel())
    if num_active <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    if total_k_blocks is None:
        total_k_blocks = (m_dim + block_k - 1) // block_k

    assert num_active <= total_k_blocks, "num_active > total_k_blocks"

    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    if active_blocks.numel() < total_k_blocks:
        padded = torch.zeros(total_k_blocks, dtype=torch.int32, device=active_blocks.device)
        padded[:num_active] = active_blocks
        active_blocks_padded = padded
    else:
        active_blocks_padded = active_blocks

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )

    prepacked_block_sparse_gemm_preload_ids_kernel[grid](
        y2d, packed, x2d, active_blocks_padded,
        num_active, n_dim, m_dim,
        packed.stride(0), packed.stride(1), packed.stride(2),
        bt,
        BLOCK_K=block_k,
        N_TOTAL_K_BLOCKS=total_k_blocks,
    )


@triton.jit
def prepacked_block_sparse_gemm_splitk_accum_kernel(
    Y_ACC,
    A_PACKED,
    X,
    ACTIVE_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    NUM_SPLIT_K,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sk = tl.program_id(2)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = pid_sk
    while block_it < NUM_ACTIVE_BLOCKS:
        block_id = tl.load(ACTIVE_BLOCKS + block_it).to(tl.int32)
        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += NUM_SPLIT_K

    y_ptrs = Y_ACC + rb[:, None] * N_OUT + rn[None, :]
    tl.atomic_add(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.jit
def prepacked_block_sparse_gemm_splitk_write_kernel(
    Y_PARTIAL,
    A_PACKED,
    X,
    ACTIVE_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    NUM_SPLIT_K,
    N_OUT,
    M,
    YP_STRIDE_SK,
    YP_STRIDE_B,
    YP_STRIDE_N,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sk = tl.program_id(2)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = pid_sk
    while block_it < NUM_ACTIVE_BLOCKS:
        block_id = tl.load(ACTIVE_BLOCKS + block_it).to(tl.int32)
        rk_local = tl.arange(0, BLOCK_K)
        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += NUM_SPLIT_K

    y_ptrs = (
        Y_PARTIAL
        + pid_sk * YP_STRIDE_SK
        + rb[:, None] * YP_STRIDE_B
        + rn[None, :] * YP_STRIDE_N
    )
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.jit
def prepacked_block_sparse_gemm_splitk_reduce_kernel(
    Y,
    Y_PARTIAL,
    NUM_SPLIT_K,
    N_OUT,
    YP_STRIDE_SK,
    YP_STRIDE_B,
    YP_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    sk = 0
    while sk < NUM_SPLIT_K:
        yp_ptrs = (
            Y_PARTIAL
            + sk * YP_STRIDE_SK
            + rb[:, None] * YP_STRIDE_B
            + rn[None, :] * YP_STRIDE_N
        )
        acc += tl.load(yp_ptrs, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT), other=0.0)
        sk += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_TOTAL_K_BLOCKS", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_bitmap_tilewalk_batched_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BITMAP,
    NUM_TOTAL_K_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Dense-like tile walk over all K-blocks:
      - iterate block_id in ascending order
      - skip inactive tiles using ACTIVE_K_BITMAP[block_id]
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_id = 0
    active_seen = 0
    while block_id < NUM_TOTAL_K_BLOCKS and active_seen < NUM_ACTIVE_BLOCKS:
        is_active = tl.load(ACTIVE_K_BITMAP + block_id, mask=block_id < NUM_TOTAL_K_BLOCKS, other=0).to(tl.int32)
        if is_active != 0:
            rk_local = tl.arange(0, BLOCK_K)
            rk = block_id * BLOCK_K + rk_local
            valid_k = rk < M

            x_ptrs = X + rb[:, None] * M + rk[None, :]
            x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

            a_ptrs = (
                A_PACKED
                + block_id * A_PACK_STRIDE_BLK
                + rk_local[:, None] * A_PACK_STRIDE_K
                + rn[None, :] * A_PACK_STRIDE_N
            )
            a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
            acc += tl.dot(x, a)

            active_seen += 1
        block_id += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_precomputed_offsets_kernel(
    Y,
    A_PACKED,
    X,
    PRECOMP_W_OFFSETS,   # int64 [NUM_ACTIVE_BLOCKS]: active_blocks[i] * A_PACK_STRIDE_BLK
    PRECOMP_X_OFFSETS,   # int32 [NUM_ACTIVE_BLOCKS]: active_blocks[i] * BLOCK_K
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Precomputed-offsets 1D sparse GEMM.
    Moves block_id * stride multiplications outside the kernel (onto CPU at dispatch time).
    Dependency chain: load(W_OFFSETS[i]) -> weight_ptrs -> load(weight)
    Same structure as naive but eliminates in-kernel multiply for weight/x addressing.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_BLOCKS:
        w_offset = tl.load(PRECOMP_W_OFFSETS + block_it).to(tl.int64)
        x_col    = tl.load(PRECOMP_X_OFFSETS + block_it).to(tl.int32)

        rk_local = tl.arange(0, BLOCK_K)
        rk = x_col + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + w_offset
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


def _prepacked_block_sparse_gemm_from_active_blocks_precomputed_offsets(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_po",
) -> None:
    """
    Precomputed-offsets 1D sparse GEMM.
    Moves block_id * stride multiplications to CPU at dispatch time.
    """
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_active = int(active_blocks.numel())
    if num_active <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    x_offsets = (active_blocks * block_k).contiguous()
    w_offsets = (active_blocks.to(torch.int64) * int(packed.stride(0))).contiguous()

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )

    prepacked_block_sparse_gemm_precomputed_offsets_kernel[grid](
        y2d, packed, x2d,
        w_offsets, x_offsets,
        num_active, n_dim, m_dim,
        packed.stride(1), packed.stride(2),
        bt,
        BLOCK_K=block_k,
    )


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_BLOCKS", "K_COMPACT"],
)
@triton.jit
def prepacked_block_sparse_gemm_compactin_batched_kernel(
    Y,
    A_PACKED,
    X_COMPACT,
    ACTIVE_BLOCKS,
    NUM_ACTIVE_BLOCKS,
    N_OUT,
    M,
    K_COMPACT,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    K-compact input variant:
      X_COMPACT shape = [BATCHSIZE, K_COMPACT]
      K_COMPACT typically equals NUM_ACTIVE_BLOCKS * BLOCK_K.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_BLOCKS:
        block_id = tl.load(ACTIVE_BLOCKS + block_it).to(tl.int32)
        rk_local = tl.arange(0, BLOCK_K)
        rk_orig = block_id * BLOCK_K + rk_local
        rk_comp = block_it * BLOCK_K + rk_local
        valid_orig = rk_orig < M
        valid_comp = rk_comp < K_COMPACT
        valid_k = valid_orig & valid_comp

        x_ptrs = X_COMPACT + rb[:, None] * K_COMPACT + rk_comp[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.autotune(
    configs=_PREPACK_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_RUNS"],
)
@triton.jit
def prepacked_block_sparse_gemm_runs_batched_kernel(
    Y,
    A_PACKED,
    X,
    RUN_STARTS,
    RUN_LENGTHS,
    NUM_RUNS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    run_it = 0
    while run_it < NUM_RUNS:
        start_block = tl.load(RUN_STARTS + run_it).to(tl.int32)
        run_len = tl.load(RUN_LENGTHS + run_it).to(tl.int32)
        blk_local = 0
        while blk_local < run_len:
            rk_local = tl.arange(0, BLOCK_K)
            block_id = start_block + blk_local
            rk = block_id * BLOCK_K + rk_local
            valid_k = rk < M

            x_ptrs = X + rb[:, None] * M + rk[None, :]
            x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

            a_ptrs = (
                A_PACKED
                + block_id * A_PACK_STRIDE_BLK
                + rk_local[:, None] * A_PACK_STRIDE_K
                + rn[None, :] * A_PACK_STRIDE_N
            )
            a = tl.load(a_ptrs, mask=valid_k[:, None] & (rn[None, :] < N_OUT), other=0.0)
            acc += tl.dot(x, a)
            blk_local += 1
        run_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(y_ptrs, acc, mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT))


@triton.autotune(
    configs=_PREPACK_2D_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_K_BLOCKS", "NUM_ACTIVE_N_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_2d_batched_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BLOCKS,
    ACTIVE_K_STARTS,
    ACTIVE_K_BASES,
    NUM_ACTIVE_K_BLOCKS,
    ACTIVE_N_BLOCKS,
    ACTIVE_N_STARTS,
    NUM_ACTIVE_N_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PRECOMPUTED_PTRS: tl.constexpr,
    ENABLE_EARLY_EXIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    if ENABLE_EARLY_EXIT:
        if pid_n >= NUM_ACTIVE_N_BLOCKS:
            return

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    n_valid = pid_n < NUM_ACTIVE_N_BLOCKS
    if USE_PRECOMPUTED_PTRS:
        n_start = tl.load(ACTIVE_N_STARTS + pid_n, mask=n_valid, other=0).to(tl.int32)
    else:
        n_block_id = tl.load(ACTIVE_N_BLOCKS + pid_n, mask=n_valid, other=0).to(tl.int32)
        n_start = n_block_id * BLOCK_N
    rn = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_K_BLOCKS:
        rk_local = tl.arange(0, BLOCK_K)
        if USE_PRECOMPUTED_PTRS:
            rk_start = tl.load(ACTIVE_K_STARTS + block_it).to(tl.int32)
            a_base = tl.load(ACTIVE_K_BASES + block_it).to(tl.int32)
        else:
            block_id = tl.load(ACTIVE_K_BLOCKS + block_it).to(tl.int32)
            rk_start = block_id * BLOCK_K
            a_base = block_id * A_PACK_STRIDE_BLK
        rk = rk_start + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + a_base
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=valid_k[:, None] & (rn[None, :] < N_OUT) & n_valid,
            other=0.0,
        )
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(
        y_ptrs,
        acc,
        mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT) & n_valid,
    )


# Sequential-bitmap kernel for pipeline-friendly K-block traversal.

_PREPACK_2D_BITMAP_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 64, "BLOCK_N": max(16, int(INDEX_BLOCK_OUT_META_SIZE))}, num_warps=8, num_stages=4),
]


@triton.autotune(
    configs=_PREPACK_2D_BITMAP_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_K_BLOCKS", "NUM_ACTIVE_N_BLOCKS", "TOTAL_K_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_2d_bitmap_batched_kernel(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BITMAP,        # uint8 [TOTAL_K_BLOCKS]: 1=active, 0=inactive (precomputed)
    NUM_ACTIVE_K_BLOCKS,    # runtime int (autotune key / early-exit)
    ACTIVE_N_BLOCKS,        # int32 [NUM_ACTIVE_N_BLOCKS]: active N-block IDs
    ACTIVE_N_STARTS,        # int32 [NUM_ACTIVE_N_BLOCKS]: precomputed n_start values
    NUM_ACTIVE_N_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOTAL_K_BLOCKS: tl.constexpr,   # constexpr: compile-time loop bound for pipelining
    USE_PRECOMPUTED_PTRS: tl.constexpr,
    ENABLE_EARLY_EXIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    if ENABLE_EARLY_EXIT:
        if pid_n >= NUM_ACTIVE_N_BLOCKS:
            return

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    n_valid = pid_n < NUM_ACTIVE_N_BLOCKS
    if USE_PRECOMPUTED_PTRS:
        n_start = tl.load(ACTIVE_N_STARTS + pid_n, mask=n_valid, other=0).to(tl.int32)
    else:
        n_block_id = tl.load(ACTIVE_N_BLOCKS + pid_n, mask=n_valid, other=0).to(tl.int32)
        n_start = n_block_id * BLOCK_N
    rn = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    # Sequential K-block loop.  All addresses are functions of the loop counter
    # alone (no data dependency from previous iterations), so Triton's software
    # pipeline (num_stages) can issue loads for tile i+k while computing tile i.
    rk_local = tl.arange(0, BLOCK_K)
    for block_id in range(TOTAL_K_BLOCKS):
        # Bitmap load address is sequential too → can be prefetched
        is_active = tl.load(ACTIVE_K_BITMAP + block_id) != 0

        rk = block_id * BLOCK_K + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x_tile = tl.load(
            x_ptrs,
            mask=is_active & (rb[:, None] < BATCHSIZE) & valid_k[None, :],
            other=0.0,
        )

        a_ptrs = (
            A_PACKED
            + block_id * A_PACK_STRIDE_BLK
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a_tile = tl.load(
            a_ptrs,
            mask=is_active & valid_k[:, None] & (rn[None, :] < N_OUT) & n_valid,
            other=0.0,
        )
        acc += tl.dot(x_tile, a_tile)

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(
        y_ptrs,
        acc,
        mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT) & n_valid,
    )


USE_2D_BITMAP_KERNEL = os.environ.get("TEAL_EXP_USE_2D_BITMAP_KERNEL", "0") not in ("0", "false", "False")


@triton.autotune(
    configs=_PREPACK_2D_AUTOTUNE_CONFIGS,
    key=["N_OUT", "M", "BATCHSIZE", "NUM_ACTIVE_K_BLOCKS", "NUM_ACTIVE_N_BLOCKS"],
)
@triton.jit
def prepacked_block_sparse_gemm_2d_compact_batched_kernel(
    Y_COMPACT,
    A_PACKED,
    X,
    ACTIVE_K_BLOCKS,
    ACTIVE_K_STARTS,
    ACTIVE_K_BASES,
    NUM_ACTIVE_K_BLOCKS,
    ACTIVE_N_BLOCKS,
    ACTIVE_N_STARTS,
    NUM_ACTIVE_N_BLOCKS,
    N_OUT,
    M,
    YC_STRIDE_B,
    YC_STRIDE_N,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PRECOMPUTED_PTRS: tl.constexpr,
    ENABLE_EARLY_EXIT: tl.constexpr,
):
    """
    2D sparse kernel that writes compact output for active N blocks only.
    Y_COMPACT shape: [BATCHSIZE, NUM_ACTIVE_N_BLOCKS * BLOCK_N]
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    if ENABLE_EARLY_EXIT:
        if pid_n >= NUM_ACTIVE_N_BLOCKS:
            return

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn_local = tl.arange(0, BLOCK_N)
    n_valid = pid_n < NUM_ACTIVE_N_BLOCKS
    if USE_PRECOMPUTED_PTRS:
        n_start = tl.load(ACTIVE_N_STARTS + pid_n, mask=n_valid, other=0).to(tl.int32)
    else:
        n_block_id = tl.load(ACTIVE_N_BLOCKS + pid_n, mask=n_valid, other=0).to(tl.int32)
        n_start = n_block_id * BLOCK_N
    rn = n_start + rn_local
    out_col = pid_n * BLOCK_N + rn_local

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_K_BLOCKS:
        rk_local = tl.arange(0, BLOCK_K)
        if USE_PRECOMPUTED_PTRS:
            rk_start = tl.load(ACTIVE_K_STARTS + block_it).to(tl.int32)
            a_base = tl.load(ACTIVE_K_BASES + block_it).to(tl.int32)
        else:
            block_id = tl.load(ACTIVE_K_BLOCKS + block_it).to(tl.int32)
            rk_start = block_id * BLOCK_K
            a_base = block_id * A_PACK_STRIDE_BLK
        rk = rk_start + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + a_base
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=valid_k[:, None] & (rn[None, :] < N_OUT) & n_valid,
            other=0.0,
        )
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y_COMPACT + rb[:, None] * YC_STRIDE_B + out_col[None, :] * YC_STRIDE_N
    tl.store(
        y_ptrs,
        acc,
        mask=(rb[:, None] < BATCHSIZE) & n_valid,
    )


@triton.jit
def prepacked_block_sparse_gemm_2d_batched_kernel_tuned(
    Y,
    A_PACKED,
    X,
    ACTIVE_K_BLOCKS,
    ACTIVE_K_STARTS,
    ACTIVE_K_BASES,
    NUM_ACTIVE_K_BLOCKS,
    ACTIVE_N_BLOCKS,
    ACTIVE_N_STARTS,
    NUM_ACTIVE_N_BLOCKS,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PRECOMPUTED_PTRS: tl.constexpr,
    ENABLE_EARLY_EXIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    if ENABLE_EARLY_EXIT:
        if pid_n >= NUM_ACTIVE_N_BLOCKS:
            return

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    n_valid = pid_n < NUM_ACTIVE_N_BLOCKS
    if USE_PRECOMPUTED_PTRS:
        n_start = tl.load(ACTIVE_N_STARTS + pid_n, mask=n_valid, other=0).to(tl.int32)
    else:
        n_block_id = tl.load(ACTIVE_N_BLOCKS + pid_n, mask=n_valid, other=0).to(tl.int32)
        n_start = n_block_id * BLOCK_N
    rn = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_K_BLOCKS:
        rk_local = tl.arange(0, BLOCK_K)
        if USE_PRECOMPUTED_PTRS:
            rk_start = tl.load(ACTIVE_K_STARTS + block_it).to(tl.int32)
            a_base = tl.load(ACTIVE_K_BASES + block_it).to(tl.int32)
        else:
            block_id = tl.load(ACTIVE_K_BLOCKS + block_it).to(tl.int32)
            rk_start = block_id * BLOCK_K
            a_base = block_id * A_PACK_STRIDE_BLK
        rk = rk_start + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + a_base
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=valid_k[:, None] & (rn[None, :] < N_OUT) & n_valid,
            other=0.0,
        )
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y + rb[:, None] * N_OUT + rn[None, :]
    tl.store(
        y_ptrs,
        acc,
        mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT) & n_valid,
    )


@triton.jit
def prepacked_block_sparse_gemm_2d_compact_batched_kernel_tuned(
    Y_COMPACT,
    A_PACKED,
    X,
    ACTIVE_K_BLOCKS,
    ACTIVE_K_STARTS,
    ACTIVE_K_BASES,
    NUM_ACTIVE_K_BLOCKS,
    ACTIVE_N_BLOCKS,
    ACTIVE_N_STARTS,
    NUM_ACTIVE_N_BLOCKS,
    N_OUT,
    M,
    YC_STRIDE_B,
    YC_STRIDE_N,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PRECOMPUTED_PTRS: tl.constexpr,
    ENABLE_EARLY_EXIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    if ENABLE_EARLY_EXIT:
        if pid_n >= NUM_ACTIVE_N_BLOCKS:
            return

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn_local = tl.arange(0, BLOCK_N)
    n_valid = pid_n < NUM_ACTIVE_N_BLOCKS
    if USE_PRECOMPUTED_PTRS:
        n_start = tl.load(ACTIVE_N_STARTS + pid_n, mask=n_valid, other=0).to(tl.int32)
    else:
        n_block_id = tl.load(ACTIVE_N_BLOCKS + pid_n, mask=n_valid, other=0).to(tl.int32)
        n_start = n_block_id * BLOCK_N
    rn = n_start + rn_local
    out_col = pid_n * BLOCK_N + rn_local

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = 0
    while block_it < NUM_ACTIVE_K_BLOCKS:
        rk_local = tl.arange(0, BLOCK_K)
        if USE_PRECOMPUTED_PTRS:
            rk_start = tl.load(ACTIVE_K_STARTS + block_it).to(tl.int32)
            a_base = tl.load(ACTIVE_K_BASES + block_it).to(tl.int32)
        else:
            block_id = tl.load(ACTIVE_K_BLOCKS + block_it).to(tl.int32)
            rk_start = block_id * BLOCK_K
            a_base = block_id * A_PACK_STRIDE_BLK
        rk = rk_start + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + a_base
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=valid_k[:, None] & (rn[None, :] < N_OUT) & n_valid,
            other=0.0,
        )
        acc += tl.dot(x, a)
        block_it += 1

    y_ptrs = Y_COMPACT + rb[:, None] * YC_STRIDE_B + out_col[None, :] * YC_STRIDE_N
    tl.store(
        y_ptrs,
        acc,
        mask=(rb[:, None] < BATCHSIZE) & n_valid,
    )


@triton.jit
def prepacked_block_sparse_gemm_2d_splitk_accum_kernel(
    Y_ACC,
    A_PACKED,
    X,
    ACTIVE_K_BLOCKS,
    ACTIVE_K_STARTS,
    ACTIVE_K_BASES,
    NUM_ACTIVE_K_BLOCKS,
    ACTIVE_N_BLOCKS,
    ACTIVE_N_STARTS,
    NUM_ACTIVE_N_BLOCKS,
    NUM_SPLIT_K,
    N_OUT,
    M,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PRECOMPUTED_PTRS: tl.constexpr,
    ENABLE_EARLY_EXIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sk = tl.program_id(2)

    if ENABLE_EARLY_EXIT:
        if pid_n >= NUM_ACTIVE_N_BLOCKS:
            return

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    n_valid = pid_n < NUM_ACTIVE_N_BLOCKS
    if USE_PRECOMPUTED_PTRS:
        n_start = tl.load(ACTIVE_N_STARTS + pid_n, mask=n_valid, other=0).to(tl.int32)
    else:
        n_block_id = tl.load(ACTIVE_N_BLOCKS + pid_n, mask=n_valid, other=0).to(tl.int32)
        n_start = n_block_id * BLOCK_N
    rn = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    block_it = pid_sk
    while block_it < NUM_ACTIVE_K_BLOCKS:
        rk_local = tl.arange(0, BLOCK_K)
        if USE_PRECOMPUTED_PTRS:
            rk_start = tl.load(ACTIVE_K_STARTS + block_it).to(tl.int32)
            a_base = tl.load(ACTIVE_K_BASES + block_it).to(tl.int32)
        else:
            block_id = tl.load(ACTIVE_K_BLOCKS + block_it).to(tl.int32)
            rk_start = block_id * BLOCK_K
            a_base = block_id * A_PACK_STRIDE_BLK
        rk = rk_start + rk_local
        valid_k = rk < M

        x_ptrs = X + rb[:, None] * M + rk[None, :]
        x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

        a_ptrs = (
            A_PACKED
            + a_base
            + rk_local[:, None] * A_PACK_STRIDE_K
            + rn[None, :] * A_PACK_STRIDE_N
        )
        a = tl.load(
            a_ptrs,
            mask=valid_k[:, None] & (rn[None, :] < N_OUT) & n_valid,
            other=0.0,
        )
        acc += tl.dot(x, a)
        block_it += NUM_SPLIT_K

    y_ptrs = Y_ACC + rb[:, None] * N_OUT + rn[None, :]
    tl.atomic_add(
        y_ptrs,
        acc,
        mask=(rb[:, None] < BATCHSIZE) & (rn[None, :] < N_OUT) & n_valid,
    )


@triton.jit
def prepacked_block_sparse_gemm_2d_compact_splitk_accum_kernel(
    YC_ACC,
    A_PACKED,
    X,
    ACTIVE_K_BLOCKS,
    ACTIVE_K_STARTS,
    ACTIVE_K_BASES,
    NUM_ACTIVE_K_BLOCKS,
    ACTIVE_N_BLOCKS,
    ACTIVE_N_STARTS,
    NUM_ACTIVE_N_BLOCKS,
    NUM_SPLIT_K,
    N_OUT,
    M,
    YC_STRIDE_B,
    YC_STRIDE_N,
    A_PACK_STRIDE_BLK,
    A_PACK_STRIDE_K,
    A_PACK_STRIDE_N,
    BATCHSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_TILE_REPEAT: tl.constexpr,
    K_CHUNK: tl.constexpr,
    USE_PRECOMPUTED_PTRS: tl.constexpr,
    ENABLE_EARLY_EXIT: tl.constexpr,
):
    pid_n_tile = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sk = tl.program_id(2)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn_local = tl.arange(0, BLOCK_N)
    if ENABLE_EARLY_EXIT:
        pid_n_base = pid_n_tile * N_TILE_REPEAT
        if pid_n_base >= NUM_ACTIVE_N_BLOCKS:
            return

    for rep in tl.static_range(0, N_TILE_REPEAT):
        pid_n = pid_n_tile * N_TILE_REPEAT + rep
        n_valid = pid_n < NUM_ACTIVE_N_BLOCKS
        if USE_PRECOMPUTED_PTRS:
            n_start = tl.load(ACTIVE_N_STARTS + pid_n, mask=n_valid, other=0).to(tl.int32)
        else:
            n_block_id = tl.load(ACTIVE_N_BLOCKS + pid_n, mask=n_valid, other=0).to(tl.int32)
            n_start = n_block_id * BLOCK_N
        rn = n_start + rn_local
        out_col = pid_n * BLOCK_N + rn_local
        acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        block_it = pid_sk
        while block_it < NUM_ACTIVE_K_BLOCKS:
            for krep in tl.static_range(0, K_CHUNK):
                block_idx = block_it + krep * NUM_SPLIT_K
                block_valid = block_idx < NUM_ACTIVE_K_BLOCKS
                rk_local = tl.arange(0, BLOCK_K)
                if USE_PRECOMPUTED_PTRS:
                    rk_start = tl.load(ACTIVE_K_STARTS + block_idx, mask=block_valid, other=0).to(tl.int32)
                    a_base = tl.load(ACTIVE_K_BASES + block_idx, mask=block_valid, other=0).to(tl.int32)
                else:
                    block_id = tl.load(ACTIVE_K_BLOCKS + block_idx, mask=block_valid, other=0).to(tl.int32)
                    rk_start = block_id * BLOCK_K
                    a_base = block_id * A_PACK_STRIDE_BLK
                rk = rk_start + rk_local
                valid_k = (rk < M) & block_valid

                x_ptrs = X + rb[:, None] * M + rk[None, :]
                x = tl.load(x_ptrs, mask=(rb[:, None] < BATCHSIZE) & valid_k[None, :], other=0.0)

                a_ptrs = (
                    A_PACKED
                    + a_base
                    + rk_local[:, None] * A_PACK_STRIDE_K
                    + rn[None, :] * A_PACK_STRIDE_N
                )
                a = tl.load(
                    a_ptrs,
                    mask=valid_k[:, None] & (rn[None, :] < N_OUT) & n_valid,
                    other=0.0,
                )
                acc += tl.dot(x, a)
            block_it += NUM_SPLIT_K * K_CHUNK

        y_ptrs = YC_ACC + rb[:, None] * YC_STRIDE_B + out_col[None, :] * YC_STRIDE_N
        tl.atomic_add(
            y_ptrs,
            acc,
            mask=(rb[:, None] < BATCHSIZE) & n_valid,
        )


def _indexed_sparse_gemm_from_active_idx(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_idx: torch.Tensor,
    out: torch.Tensor,
    weight_row_offset: int = 0,
) -> None:
    bsz, _, m_dim = x.shape
    n_dim = out.shape[-1]
    k_dim = int(active_idx.numel())
    if k_dim <= 0:
        out.zero_()
        return

    if active_idx.dtype != torch.int32:
        active_idx = active_idx.to(dtype=torch.int32)
    active_idx = active_idx.contiguous()

    x2d = x.view(bsz, m_dim)
    y2d = out.view(bsz, n_dim)

    block_b, block_n, block_k, num_warps, num_stages = _indexed_launch_meta(bsz, n_dim, k_dim)

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bsz, META["BLOCK_B"]),
    )
    indexed_sparse_gemm_batched_kernel[grid](
        y2d,
        weight,
        x2d,
        active_idx,
        k_dim,
        n_dim,
        m_dim,
        weight.stride(0),
        weight.stride(1),
        weight_row_offset,
        bsz,
        BLOCK_B=block_b,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _indexed_sparse_gemm_from_active_blocks_blockmeta(
    x: torch.Tensor,
    weight: torch.Tensor,
    mask_u8: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    weight_row_offset: int = 0,
) -> None:
    bsz, _, m_dim = x.shape
    n_dim = out.shape[-1]
    num_blocks = int(active_blocks.numel())
    if num_blocks <= 0:
        out.zero_()
        return

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    active_blocks = _sort_active_blocks_asc(active_blocks.contiguous())

    x2d = x.view(bsz, m_dim)
    y2d = out.view(bsz, n_dim)

    block_b, block_n, _unused_block_k, num_warps, num_stages = _indexed_launch_meta(
        bsz, n_dim, max(1, int(mask_u8.sum().item()))
    )
    block_k = max(16, int(INDEX_BLOCK_META_SIZE))

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bsz, META["BLOCK_B"]),
    )
    indexed_sparse_blockmeta_gemm_batched_kernel[grid](
        y2d,
        weight,
        x2d,
        mask_u8,
        active_blocks,
        num_blocks,
        n_dim,
        m_dim,
        weight.stride(0),
        weight.stride(1),
        weight_row_offset,
        bsz,
        BLOCK_B=block_b,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _proj_tuned_launch_meta(proj_tag: str, block_n: int) -> tuple[int, int, int]:
    """
    Projection-aware launch tuning for 2D sparse path.
    Returns (block_b, num_warps, num_stages).
    """
    p = str(proj_tag or "").lower()
    if p in {"q", "k", "v", "q_proj", "k_proj", "v_proj", "qkv"}:
        return 32, 4, 2
    if p in {"gate", "up", "gate_proj", "up_proj", "dual"}:
        return 64, 8, 3
    if p in {"down", "down_proj"}:
        return 64, 8, 3
    if p in {"o", "o_proj"}:
        return 32, 4, 2
    if int(block_n) >= 128:
        return 32, 8, 3
    return 32, 4, 2


def _choose_split_k_2d(bt: int, num_active_k_blocks: int) -> int:
    if not bool(ENABLE_SPLIT_K_2D_DEFAULT):
        return 1
    if int(bt) > int(SPLIT_K_2D_MAX_BT):
        return 1
    if int(num_active_k_blocks) < int(SPLIT_K_2D_MIN_ACTIVE_K_BLOCKS):
        return 1
    if int(num_active_k_blocks) >= 256:
        return min(int(MAX_SPLIT_K_2D), 8)
    if int(num_active_k_blocks) >= 192:
        return min(int(MAX_SPLIT_K_2D), 4)
    return min(int(MAX_SPLIT_K_2D), 2)


def _proj_family_1d(proj_tag: str) -> str:
    p = str(proj_tag or "").lower()
    if p in {"down", "down_proj"}:
        return "down"
    if p in {"o", "o_proj"}:
        return "o"
    if p in {"q", "k", "v", "q_proj", "k_proj", "v_proj", "qkv"}:
        return "qkv"
    return "other"


def _choose_split_k_1d(bt: int, num_active_blocks: int, n_out: int, use_run_compressed: bool, proj_tag: str = "") -> int:
    if not bool(ENABLE_SPLIT_K_1D_DEFAULT):
        return 1
    if bool(use_run_compressed):
        return 1
    if int(bt) > int(SPLIT_K_1D_MAX_BT):
        return 1
    if int(n_out) < int(SPLIT_K_1D_MIN_N_OUT):
        return 1

    fam = _proj_family_1d(proj_tag)
    min_active = int(SPLIT_K_1D_MIN_ACTIVE_BLOCKS)
    max_split = int(MAX_SPLIT_K_1D)
    if fam == "o":
        min_active = max(min_active + 16, int(SPLIT_K_1D_MIN_ACTIVE_BLOCKS))
        max_split = min(max_split, 2)
    elif fam == "qkv":
        min_active = max(min_active + 32, int(SPLIT_K_1D_MIN_ACTIVE_BLOCKS))
        max_split = min(max_split, 2)
    elif fam == "other":
        return 1

    if int(num_active_blocks) < int(min_active):
        return 1
    if int(num_active_blocks) >= 256:
        return min(int(max_split), 8)
    if int(num_active_blocks) >= 192:
        return min(int(max_split), 4)
    return min(int(max_split), 2)


def _choose_split_k_1d_reduce_mode(bt: int, n_out: int, num_split_k: int, proj_tag: str = "") -> str:
    mode = str(_SPLIT_K_1D_REDUCE_MODE)
    if mode in {"atomic", "two_pass"}:
        return mode
    fam = _proj_family_1d(proj_tag)
    if int(num_split_k) <= 1:
        return "atomic"
    if fam == "down" and int(bt) <= 32 and int(n_out) >= 4096:
        return "two_pass"
    if fam == "qkv" and int(n_out) >= 8192:
        return "two_pass"
    return "atomic"


def _get_splitk_accum_buffer(prefix: str, bt: int, n_dim: int, device: torch.device) -> torch.Tensor:
    cache = _SPLITK_ACCUM_CACHE.setdefault("bufs", {})
    key = (str(prefix), str(device), int(bt), int(n_dim))
    buf = cache.get(key)
    if buf is None:
        _SPLITK_ACCUM_CACHE["misses"] += 1
        buf = torch.zeros((bt, n_dim), device=device, dtype=torch.float32)
        cache[key] = buf
    else:
        _SPLITK_ACCUM_CACHE["hits"] += 1
        buf.zero_()
    return buf


def _get_splitk_partial_buffer(prefix: str, num_split_k: int, bt: int, n_dim: int, device: torch.device) -> torch.Tensor:
    cache = _SPLITK_PARTIAL_CACHE.setdefault("bufs", {})
    key = (str(prefix), str(device), int(num_split_k), int(bt), int(n_dim))
    buf = cache.get(key)
    if buf is None:
        _SPLITK_PARTIAL_CACHE["misses"] += 1
        buf = torch.empty((num_split_k, bt, n_dim), device=device, dtype=torch.float32)
        cache[key] = buf
    else:
        _SPLITK_PARTIAL_CACHE["hits"] += 1
    return buf


def _get_temp_tensor_buffer(
    prefix: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    zero_fill: bool = False,
) -> torch.Tensor:
    cache = _TEMP_TENSOR_CACHE.setdefault("bufs", {})
    key = (str(prefix), str(device), str(dtype), tuple(int(v) for v in shape))
    buf = cache.get(key)
    if buf is None:
        _TEMP_TENSOR_CACHE["misses"] += 1
        buf = torch.empty(shape, device=device, dtype=dtype)
        cache[key] = buf
    else:
        _TEMP_TENSOR_CACHE["hits"] += 1
    if bool(zero_fill):
        buf.zero_()
    return buf


def _get_prepacked_weight_block_major(weight: torch.Tensor, block_k: int, cache_prefix: str = "gemm") -> torch.Tensor:
    cache = _WEIGHT_PREPACK_CACHE.setdefault(cache_prefix, {})
    n_dim, m_dim = weight.shape
    key = (
        cache_prefix,
        int(weight.data_ptr()),
        str(weight.device),
        str(weight.dtype),
        int(n_dim),
        int(m_dim),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(block_k),
    )
    cached = cache.get(key)
    if cached is not None:
        _WEIGHT_PREPACK_CACHE["hits"] += 1
        return cached

    _WEIGHT_PREPACK_CACHE["misses"] += 1
    num_blocks = (m_dim + block_k - 1) // block_k
    padded_m = num_blocks * block_k
    wt = weight.t().contiguous()
    if padded_m != m_dim:
        wt_pad = torch.zeros((padded_m, n_dim), device=weight.device, dtype=weight.dtype)
        wt_pad[:m_dim, :] = wt
    else:
        wt_pad = wt
    packed = wt_pad.view(num_blocks, block_k, n_dim).contiguous()
    cache[key] = packed
    return packed


def _prepacked_block_sparse_gemm_from_active_blocks(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm",
    proj_tag: str = "",
    sort_active_blocks: bool = True,
    use_run_compressed: bool = True,
    active_blocks_limit: int | None = None,
) -> None:
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_blocks = int(active_blocks.numel()) if active_blocks_limit is None else int(active_blocks_limit)
    if num_blocks <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()
    if int(active_blocks.numel()) > num_blocks:
        active_blocks = active_blocks[:num_blocks].contiguous()
    if sort_active_blocks:
        active_blocks = _sort_active_blocks_asc(active_blocks)

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )
    num_split_k = int(
        _choose_split_k_1d(
            bt=bt,
            num_active_blocks=num_blocks,
            n_out=n_dim,
            use_run_compressed=bool(use_run_compressed),
            proj_tag=str(proj_tag or ""),
        )
    )
    use_split_k = num_split_k > 1
    if use_run_compressed:
        run_starts, run_lengths = _get_cached_active_block_runs(
            active_blocks[:num_blocks], prefix=f"{cache_prefix}_runs", sort_blocks=sort_active_blocks
        )
        num_runs = int(run_starts.numel())
        if num_runs <= 0:
            out.zero_()
            return
        prepacked_block_sparse_gemm_runs_batched_kernel[grid](
            y2d,
            packed,
            x2d,
            run_starts,
            run_lengths,
            num_runs,
            n_dim,
            m_dim,
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            bt,
            BLOCK_K=block_k,
        )
        return

    if use_split_k:
        reduce_mode = _choose_split_k_1d_reduce_mode(
            bt=bt,
            n_out=n_dim,
            num_split_k=num_split_k,
            proj_tag=str(proj_tag or ""),
        )
        split_block_n = max(16, int(INDEX_BLOCK_OUT_META_SIZE))
        if split_block_n >= 256:
            split_b, split_warps, split_stages = 64, 8, 3
        elif split_block_n >= 128:
            split_b, split_warps, split_stages = 32, 4, 2
        else:
            split_b, split_warps, split_stages = 32, 4, 2
        if reduce_mode == "two_pass":
            y_part = _get_splitk_partial_buffer(
                prefix=f"{cache_prefix}_splitk1d_part",
                num_split_k=num_split_k,
                bt=bt,
                n_dim=n_dim,
                device=out.device,
            )
            grid_write = (triton.cdiv(n_dim, split_block_n), triton.cdiv(bt, split_b), num_split_k)
            prepacked_block_sparse_gemm_splitk_write_kernel[grid_write](
                y_part,
                packed,
                x2d,
                active_blocks,
                num_blocks,
                num_split_k,
                n_dim,
                m_dim,
                y_part.stride(0),
                y_part.stride(1),
                y_part.stride(2),
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                bt,
                BLOCK_B=split_b,
                BLOCK_N=split_block_n,
                BLOCK_K=block_k,
                num_warps=split_warps,
                num_stages=split_stages,
            )
            grid_reduce = (triton.cdiv(n_dim, split_block_n), triton.cdiv(bt, split_b))
            prepacked_block_sparse_gemm_splitk_reduce_kernel[grid_reduce](
                y2d,
                y_part,
                num_split_k,
                n_dim,
                y_part.stride(0),
                y_part.stride(1),
                y_part.stride(2),
                bt,
                BLOCK_B=split_b,
                BLOCK_N=split_block_n,
                num_warps=split_warps,
                num_stages=split_stages,
            )
        else:
            y_acc_2d = _get_splitk_accum_buffer(prefix=f"{cache_prefix}_splitk1d", bt=bt, n_dim=n_dim, device=out.device)
            grid_split = (triton.cdiv(n_dim, split_block_n), triton.cdiv(bt, split_b), num_split_k)
            prepacked_block_sparse_gemm_splitk_accum_kernel[grid_split](
                y_acc_2d,
                packed,
                x2d,
                active_blocks,
                num_blocks,
                num_split_k,
                n_dim,
                m_dim,
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                bt,
                BLOCK_B=split_b,
                BLOCK_N=split_block_n,
                BLOCK_K=block_k,
                num_warps=split_warps,
                num_stages=split_stages,
            )
            if y2d.dtype == torch.float32:
                y2d.copy_(y_acc_2d)
            else:
                y2d.copy_(y_acc_2d.to(dtype=y2d.dtype))
        return

    prepacked_block_sparse_gemm_batched_kernel[grid](
        y2d,
        packed,
        x2d,
        active_blocks,
        num_blocks,
        n_dim,
        m_dim,
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        bt,
        BLOCK_K=block_k,
    )


def _prepacked_block_sparse_gemm_from_active_blocks_bitmap_tilewalk(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_tilewalk",
    sort_active_blocks: bool = True,
    active_blocks_limit: int | None = None,
) -> None:
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_blocks = int(active_blocks.numel()) if active_blocks_limit is None else int(active_blocks_limit)
    if num_blocks <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()
    if int(active_blocks.numel()) > num_blocks:
        active_blocks = active_blocks[:num_blocks].contiguous()
    if sort_active_blocks:
        active_blocks = _sort_active_blocks_asc(active_blocks)

    total_k_blocks = int((m_dim + block_k - 1) // block_k)
    bitmap = _get_cached_active_block_bitmap(
        active_blocks=active_blocks,
        num_total_blocks=total_k_blocks,
        prefix=f"{cache_prefix}_bitmap",
    )

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )
    prepacked_block_sparse_gemm_bitmap_tilewalk_batched_kernel[grid](
        y2d,
        packed,
        x2d,
        bitmap,
        total_k_blocks,
        num_blocks,
        n_dim,
        m_dim,
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        bt,
        BLOCK_K=block_k,
    )


def _prepacked_block_sparse_gemm_from_compact_input_blocks(
    x_compact: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    cache_prefix: str = "gemm_compactin",
    sort_active_blocks: bool = True,
) -> None:
    """
    K-compact runtime path:
      - input uses compact active-block order (no K gather via original indices)
      - weight stays persistent block-major packed
    """
    bsz, seqlen, k_comp = x_compact.shape
    n_dim = out.shape[-1]
    num_blocks = int(active_blocks.numel())
    if num_blocks <= 0:
        out.zero_()
        return

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()
    if sort_active_blocks:
        active_blocks = _sort_active_blocks_asc(active_blocks)

    bt = bsz * seqlen
    x2d = x_compact.view(bt, k_comp)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid = lambda META: (
        triton.cdiv(n_dim, META["BLOCK_N"]),
        triton.cdiv(bt, META["BLOCK_B"]),
    )
    prepacked_block_sparse_gemm_compactin_batched_kernel[grid](
        y2d,
        packed,
        x2d,
        active_blocks,
        num_blocks,
        n_dim,
        int(weight.shape[1]),
        int(k_comp),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        bt,
        BLOCK_K=block_k,
    )


def _scatter_compact_to_dense_output(
    compact_out: torch.Tensor,
    active_n_blocks: torch.Tensor,
    block_n: int,
    out_dense: torch.Tensor,
) -> None:
    """
    Scatter compact [B,S,num_active_blocks*block_n] output back to dense N axis.
    """
    if int(active_n_blocks.numel()) <= 0:
        return
    if active_n_blocks.dtype != torch.int64:
        active_n_blocks = active_n_blocks.to(dtype=torch.int64)
    bsz, seqlen, compact_w = compact_out.shape
    bt = bsz * seqlen
    out2d = out_dense.view(bt, out_dense.shape[-1])
    comp2d = compact_out.view(bt, compact_w)
    blk = int(block_n)
    for i, block_id in enumerate(active_n_blocks.tolist()):
        src0 = int(i * blk)
        src1 = min(src0 + blk, int(compact_w))
        dst0 = int(block_id * blk)
        dst1 = min(dst0 + blk, int(out2d.shape[-1]))
        width = min(src1 - src0, dst1 - dst0)
        if width > 0:
            out2d[:, dst0 : dst0 + width] = comp2d[:, src0 : src0 + width]


def _prepacked_block_sparse_gemm_2d_from_active_blocks(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_k_blocks: torch.Tensor,
    active_n_blocks: torch.Tensor | None,
    out: torch.Tensor,
    cache_prefix: str = "gemm2d",
    proj_tag: str = "",
    sort_active_k_blocks: bool = True,
    sort_active_n_blocks: bool = True,
    zero_fill_out: bool = False,
    compact_output: bool = False,
    scatter_compact_to_dense: bool = False,
    active_n_mask_u8: torch.Tensor | None = None,
    active_k_blocks_limit: int | None = None,
    active_n_blocks_limit: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_k_blocks = int(active_k_blocks.numel()) if active_k_blocks_limit is None else int(active_k_blocks_limit)
    if num_k_blocks <= 0:
        out.zero_()
        empty = torch.empty((0,), device=out.device, dtype=torch.int32)
        return None, empty

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    block_n_meta = max(16, int(INDEX_BLOCK_OUT_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)

    if active_k_blocks.dtype != torch.int32:
        active_k_blocks = active_k_blocks.to(dtype=torch.int32)
    if not active_k_blocks.is_contiguous():
        active_k_blocks = active_k_blocks.contiguous()
    if int(active_k_blocks.numel()) > num_k_blocks:
        active_k_blocks = active_k_blocks[:num_k_blocks].contiguous()
    if sort_active_k_blocks:
        active_k_blocks = _sort_active_blocks_asc(active_k_blocks)

    if active_n_blocks is None and active_n_mask_u8 is not None:
        active_n_blocks = _get_cached_active_n_blocks_delta(
            mask_u8=active_n_mask_u8,
            prefix=f"{cache_prefix}_nout",
            block_size=block_n_meta,
        )

    if active_n_blocks is None:
        n_blocks = (int(n_dim) + block_n_meta - 1) // block_n_meta
        active_n_blocks = torch.arange(n_blocks, device=out.device, dtype=torch.int32)
    elif active_n_blocks.dtype != torch.int32:
        active_n_blocks = active_n_blocks.to(dtype=torch.int32)
    if not active_n_blocks.is_contiguous():
        active_n_blocks = active_n_blocks.contiguous()
    if sort_active_n_blocks:
        active_n_blocks = _sort_active_blocks_asc(active_n_blocks)

    num_n_blocks = int(active_n_blocks.numel()) if active_n_blocks_limit is None else int(active_n_blocks_limit)
    if num_n_blocks <= 0:
        out.zero_()
        return None, active_n_blocks
    fixed_grid = os.environ.get("TEAL_EXP_FIXED_GRID_FROM_PADDED_META", "1") not in ("0", "false", "False")
    if (not fixed_grid) and int(active_n_blocks.numel()) > num_n_blocks:
        active_n_blocks = active_n_blocks[:num_n_blocks].contiguous()
    active_n_blocks_ret = active_n_blocks[:num_n_blocks].contiguous()
    use_precomputed_ptrs = bool(USE_BLOCK_BASEPTR_METADATA_DEFAULT)
    enable_n_valid_early_exit = bool(ENABLE_N_VALID_EARLY_EXIT_DEFAULT)

    if use_precomputed_ptrs:
        active_k_starts, active_k_bases = _build_block_start_and_base_tables(
            active_blocks=active_k_blocks,
            valid_len=num_k_blocks,
            block_k=block_k,
            packed_stride_blk=int(packed.stride(0)),
            sort_prefix=False,
            keep_padded_shape=False,
        )
        active_n_starts = _build_n_start_table(
            active_n_blocks=active_n_blocks,
            valid_len=num_n_blocks,
            block_n=block_n_meta,
            sort_prefix=False,
            keep_padded_shape=False,
        )
    else:
        # Dummy pointers for unified kernel signature.
        active_k_starts = active_k_blocks
        active_k_bases = active_k_blocks
        active_n_starts = active_n_blocks

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    grid_n = int(active_n_blocks.numel()) if (fixed_grid and active_n_blocks_limit is not None) else int(num_n_blocks)
    grid = lambda META: (grid_n, triton.cdiv(bt, META["BLOCK_B"]))
    num_split_k = int(_choose_split_k_2d(bt=bt, num_active_k_blocks=num_k_blocks))
    use_split_k = num_split_k > 1

    if bool(compact_output):
        block_n = int(block_n_meta)
        compact_width = int((grid_n if (fixed_grid and active_n_blocks_limit is not None) else num_n_blocks) * block_n)
        if use_split_k:
            y_compact_acc_2d = _get_temp_tensor_buffer(
                prefix=f"{cache_prefix}_splitk2d_compact_acc",
                shape=(bt, compact_width),
                dtype=torch.float32,
                device=out.device,
                zero_fill=True,
            )
            split_b, split_warps, split_stages = _proj_tuned_launch_meta(proj_tag=proj_tag, block_n=block_n)
            n_tile_repeat = int(SPLIT_K_2D_N_TILE_REPEAT)
            k_chunk = int(SPLIT_K_2D_K_CHUNK)
            grid_split_n = triton.cdiv(grid_n, n_tile_repeat)
            grid_split = (grid_split_n, triton.cdiv(bt, split_b), num_split_k)
            prepacked_block_sparse_gemm_2d_compact_splitk_accum_kernel[grid_split](
                y_compact_acc_2d,
                packed,
                x2d,
                active_k_blocks,
                active_k_starts,
                active_k_bases,
                num_k_blocks,
                active_n_blocks,
                active_n_starts,
                num_n_blocks,
                num_split_k,
                n_dim,
                m_dim,
                y_compact_acc_2d.stride(0),
                y_compact_acc_2d.stride(1),
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                bt,
                BLOCK_B=split_b,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                N_TILE_REPEAT=n_tile_repeat,
                K_CHUNK=k_chunk,
                USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
                ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
                num_warps=split_warps,
                num_stages=split_stages,
            )
            y_compact_2d = _get_temp_tensor_buffer(
                prefix=f"{cache_prefix}_splitk2d_compact_out",
                shape=(bt, compact_width),
                dtype=out.dtype,
                device=out.device,
                zero_fill=False,
            )
            y_compact_2d.copy_(y_compact_acc_2d)
        else:
            need_tail_zero = bool(
                fixed_grid
                and (active_n_blocks_limit is not None)
                and (grid_n > int(num_n_blocks))
            )
            y_compact_2d = _get_temp_tensor_buffer(
                prefix=f"{cache_prefix}_compact2d_out",
                shape=(bt, compact_width),
                dtype=out.dtype,
                device=out.device,
                zero_fill=need_tail_zero,
            )
        tuned_b, tuned_warps, tuned_stages = _proj_tuned_launch_meta(proj_tag=proj_tag, block_n=block_n)
        p = str(proj_tag or "").lower()
        if not use_split_k:
            if bool(ENABLE_PROJ_FAMILY_TUNED_DEFAULT) and p in {"down", "down_proj", "o", "o_proj"}:
                prepacked_block_sparse_gemm_2d_compact_batched_kernel_tuned[grid](
                    y_compact_2d,
                    packed,
                    x2d,
                    active_k_blocks,
                    active_k_starts,
                    active_k_bases,
                    num_k_blocks,
                    active_n_blocks,
                    active_n_starts,
                    num_n_blocks,
                    n_dim,
                    m_dim,
                    y_compact_2d.stride(0),
                    y_compact_2d.stride(1),
                    packed.stride(0),
                    packed.stride(1),
                    packed.stride(2),
                    bt,
                    BLOCK_B=tuned_b,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
                    ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
                    num_warps=tuned_warps,
                    num_stages=tuned_stages,
                )
            else:
                prepacked_block_sparse_gemm_2d_compact_batched_kernel[grid](
                    y_compact_2d,
                    packed,
                    x2d,
                    active_k_blocks,
                    active_k_starts,
                    active_k_bases,
                    num_k_blocks,
                    active_n_blocks,
                    active_n_starts,
                    num_n_blocks,
                    n_dim,
                    m_dim,
                    y_compact_2d.stride(0),
                    y_compact_2d.stride(1),
                    packed.stride(0),
                    packed.stride(1),
                    packed.stride(2),
                    bt,
                    BLOCK_K=block_k,
                    USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
                    ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
                )
        y_compact_3d = y_compact_2d.view(bsz, seqlen, compact_width)
        if bool(scatter_compact_to_dense):
            if bool(zero_fill_out):
                out.zero_()
            _scatter_compact_to_dense_output(
                compact_out=y_compact_3d,
                active_n_blocks=active_n_blocks,
                block_n=block_n,
                out_dense=out,
            )
        return y_compact_3d, active_n_blocks_ret

    if bool(zero_fill_out) and (not use_split_k):
        out.zero_()
    tuned_b, tuned_warps, tuned_stages = _proj_tuned_launch_meta(proj_tag=proj_tag, block_n=block_n_meta)
    p = str(proj_tag or "").lower()
    if use_split_k:
        y_acc_2d = _get_temp_tensor_buffer(
            prefix=f"{cache_prefix}_splitk2d_acc",
            shape=(bt, n_dim),
            dtype=torch.float32,
            device=out.device,
            zero_fill=True,
        )
        grid_split = (grid_n, triton.cdiv(bt, tuned_b), num_split_k)
        prepacked_block_sparse_gemm_2d_splitk_accum_kernel[grid_split](
            y_acc_2d,
            packed,
            x2d,
            active_k_blocks,
            active_k_starts,
            active_k_bases,
            num_k_blocks,
            active_n_blocks,
            active_n_starts,
            num_n_blocks,
            num_split_k,
            n_dim,
            m_dim,
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            bt,
            BLOCK_B=tuned_b,
            BLOCK_N=block_n_meta,
            BLOCK_K=block_k,
            USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
            ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
            num_warps=tuned_warps,
            num_stages=tuned_stages,
        )
        y2d.copy_(y_acc_2d)
    else:
        if bool(ENABLE_PROJ_FAMILY_TUNED_DEFAULT) and p in {"down", "down_proj", "o", "o_proj"}:
            prepacked_block_sparse_gemm_2d_batched_kernel_tuned[grid](
                y2d,
                packed,
                x2d,
                active_k_blocks,
                active_k_starts,
                active_k_bases,
                num_k_blocks,
                active_n_blocks,
                active_n_starts,
                num_n_blocks,
                n_dim,
                m_dim,
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                bt,
                BLOCK_B=tuned_b,
                BLOCK_N=block_n_meta,
                BLOCK_K=block_k,
                USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
                ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
                num_warps=tuned_warps,
                num_stages=tuned_stages,
            )
        elif bool(USE_2D_BITMAP_KERNEL):
            # Experimental: sequential-bitmap kernel (TEAL_EXP_USE_2D_BITMAP_KERNEL=1)
            total_k_blocks = (m_dim + block_k - 1) // block_k
            _bmp_key = (int(active_k_blocks.data_ptr()), total_k_blocks)
            _bmp_store = _BLOCK_BITMAP_CACHE.setdefault("_2d_bmp_main", {})
            if _bmp_key not in _bmp_store:
                _bmp_store[_bmp_key] = _build_active_k_bitmap(active_k_blocks, total_k_blocks, out.device)
            active_k_bitmap = _bmp_store[_bmp_key]
            prepacked_block_sparse_gemm_2d_bitmap_batched_kernel[grid](
                y2d,
                packed,
                x2d,
                active_k_bitmap,
                num_k_blocks,
                active_n_blocks,
                active_n_starts,
                num_n_blocks,
                n_dim,
                m_dim,
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                bt,
                BLOCK_K=block_k,
                TOTAL_K_BLOCKS=total_k_blocks,
                USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
                ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
            )
        else:
            prepacked_block_sparse_gemm_2d_batched_kernel[grid](
                y2d,
                packed,
                x2d,
                active_k_blocks,
                active_k_starts,
                active_k_bases,
                num_k_blocks,
                active_n_blocks,
                active_n_starts,
                num_n_blocks,
                n_dim,
                m_dim,
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                bt,
                BLOCK_K=block_k,
                USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
                ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
            )
    return None, active_n_blocks_ret


def _prepacked_block_sparse_gemm_2d_bitmap_from_active_blocks(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_k_blocks: torch.Tensor,
    active_n_blocks: torch.Tensor | None,
    out: torch.Tensor,
    cache_prefix: str = "gemm2d_bmp",
    sort_active_n_blocks: bool = True,
    zero_fill_out: bool = False,
    active_n_blocks_limit: int | None = None,
) -> tuple[None, torch.Tensor]:
    """
    2D sparse GEMM using the sequential-bitmap kernel.

    Instead of iterating over active K-block IDs (serial dependency chain),
    this iterates over ALL total_k K-tiles sequentially with a precomputed
    bitmap mask.  Sequential addresses break the serial dependency, allowing
    Triton's software pipeline (num_stages) to overlap memory latency with
    tensor-core compute.

    Enable via: TEAL_EXP_USE_2D_BITMAP_KERNEL=1
    """
    bsz, seqlen, m_dim = x.shape
    n_dim = out.shape[-1]
    num_k_blocks = int(active_k_blocks.numel())
    if num_k_blocks <= 0:
        out.zero_()
        empty = torch.empty((0,), device=out.device, dtype=torch.int32)
        return None, empty

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    block_n_meta = max(16, int(INDEX_BLOCK_OUT_META_SIZE))
    packed = _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)
    total_k_blocks = (m_dim + block_k - 1) // block_k

    if active_k_blocks.dtype != torch.int32:
        active_k_blocks = active_k_blocks.to(dtype=torch.int32)
    if not active_k_blocks.is_contiguous():
        active_k_blocks = active_k_blocks.contiguous()

    # Build bitmap: uint8 [total_k_blocks], 1=active 0=inactive.
    # Cache by (data_ptr, total_k_blocks) so CUDA graph capture (which
    # replays the same fixed-pointer tensors) doesn't re-run fancy-indexing.
    _bmp_key = (int(active_k_blocks.data_ptr()), total_k_blocks)
    _bmp_store = _BLOCK_BITMAP_CACHE.setdefault("_2d_bmp", {})
    if _bmp_key not in _bmp_store:
        _bmp_store[_bmp_key] = _build_active_k_bitmap(active_k_blocks, total_k_blocks, out.device)
    active_k_bitmap = _bmp_store[_bmp_key]

    if active_n_blocks is None:
        n_blocks = (int(n_dim) + block_n_meta - 1) // block_n_meta
        active_n_blocks = torch.arange(n_blocks, device=out.device, dtype=torch.int32)
    elif active_n_blocks.dtype != torch.int32:
        active_n_blocks = active_n_blocks.to(dtype=torch.int32)
    if not active_n_blocks.is_contiguous():
        active_n_blocks = active_n_blocks.contiguous()
    if sort_active_n_blocks:
        active_n_blocks = _sort_active_blocks_asc(active_n_blocks)

    num_n_blocks = int(active_n_blocks.numel()) if active_n_blocks_limit is None else int(active_n_blocks_limit)
    if num_n_blocks <= 0:
        out.zero_()
        return None, active_n_blocks

    active_n_blocks_ret = active_n_blocks[:num_n_blocks].contiguous()
    use_precomputed_ptrs = bool(USE_BLOCK_BASEPTR_METADATA_DEFAULT)
    enable_n_valid_early_exit = bool(ENABLE_N_VALID_EARLY_EXIT_DEFAULT)

    if use_precomputed_ptrs:
        active_n_starts = _build_n_start_table(
            active_n_blocks=active_n_blocks,
            valid_len=num_n_blocks,
            block_n=block_n_meta,
            sort_prefix=False,
            keep_padded_shape=False,
        )
    else:
        active_n_starts = active_n_blocks  # dummy

    bt = bsz * seqlen
    x2d = x.view(bt, m_dim)
    y2d = out.view(bt, n_dim)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not y2d.is_contiguous():
        y2d = y2d.contiguous()

    if bool(zero_fill_out):
        out.zero_()

    grid_n = int(num_n_blocks)
    grid = lambda META: (grid_n, triton.cdiv(bt, META["BLOCK_B"]))

    prepacked_block_sparse_gemm_2d_bitmap_batched_kernel[grid](
        y2d,
        packed,
        x2d,
        active_k_bitmap,
        num_k_blocks,
        active_n_blocks,
        active_n_starts,
        num_n_blocks,
        n_dim,
        m_dim,
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        bt,
        BLOCK_K=block_k,
        TOTAL_K_BLOCKS=total_k_blocks,
        USE_PRECOMPUTED_PTRS=use_precomputed_ptrs,
        ENABLE_EARLY_EXIT=enable_n_valid_early_exit,
    )
    return None, active_n_blocks_ret


def _prepacked_block_sparse_gemm_2d_compact_from_active_blocks(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_k_blocks: torch.Tensor,
    active_n_blocks: torch.Tensor | None,
    cache_prefix: str = "gemm2d_compact",
    proj_tag: str = "",
    sort_active_k_blocks: bool = True,
    sort_active_n_blocks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compact/index-aware output path:
      returns compact output + active output block ids.
    """
    bsz, seqlen, _ = x.shape
    if active_n_blocks is not None and active_n_blocks.dtype != torch.int32:
        active_n_blocks = active_n_blocks.to(dtype=torch.int32)
    block_n = max(16, int(INDEX_BLOCK_OUT_META_SIZE))
    num_n_blocks = int(active_n_blocks.numel()) if active_n_blocks is not None else 0
    if num_n_blocks <= 0:
        out_empty = torch.empty((bsz, seqlen, 0), device=x.device, dtype=x.dtype)
        empty_blocks = torch.empty((0,), device=x.device, dtype=torch.int32)
        return out_empty, empty_blocks
    out_compact = torch.empty((bsz, seqlen, num_n_blocks * block_n), device=x.device, dtype=x.dtype)
    compact_out, active_blocks = _prepacked_block_sparse_gemm_2d_from_active_blocks(
        x=x,
        weight=weight,
        active_k_blocks=active_k_blocks,
        active_n_blocks=active_n_blocks,
        out=out_compact,
        cache_prefix=cache_prefix,
        proj_tag=proj_tag,
        sort_active_k_blocks=sort_active_k_blocks,
        sort_active_n_blocks=sort_active_n_blocks,
        zero_fill_out=False,
        compact_output=True,
        scatter_compact_to_dense=False,
    )
    if compact_out is None:
        compact_out = out_compact
    return compact_out, active_blocks


def _indexed_sparse_gemm_from_active_blocks(
    x: torch.Tensor,
    weight: torch.Tensor,
    mask_u8: torch.Tensor | None,
    active_blocks: torch.Tensor,
    out: torch.Tensor,
    active_blocks_limit: int | None = None,
    weight_row_offset: int = 0,
    active_out_blocks: torch.Tensor | None = None,
    active_out_blocks_limit: int | None = None,
    cache_prefix: str = "gemm",
    proj_tag: str = "",
    return_compact: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Runtime entry point used by decode patch path.
    Main fast path is prepacked block-sparse; blockmeta path remains for reference.
    """
    path = FAST_BLOCK_KERNEL_PATH
    proj = str(proj_tag or "").lower()
    # Projection-family fixed dispatch:
    # - q/k/v/o default to 1D K-block sparse path (stable, lower metadata overhead)
    # - gate/up/down keep global path (typically 2D for compact-chain gains)
    if bool(ENABLE_PROJ_FAMILY_DISPATCH_DEFAULT) and path == "prepack_blocks_2d" and proj in {"q", "k", "v", "q_proj", "k_proj", "v_proj", "o", "o_proj"}:
        path = os.environ.get("TEAL_EXP_ATTN_KERNEL_PATH", "prepack_blocks").strip().lower()
        if path not in {"prepack_runs", "prepack_blocks", "prepack_blocks_2d", "prepack_bitmap_tilewalk", "blockmeta"}:
            path = "prepack_blocks"
    sort_blocks = bool(SORT_ACTIVE_BLOCKS_DEFAULT)
    use_run = bool(USE_RUN_COMPRESSED_METADATA_DEFAULT)
    # If 1D prepacked path is selected, transparently switch to run-compressed
    # variant to reduce ACTIVE_BLOCKS indirection overhead.
    if path == "prepack_blocks" and use_run:
        path = "prepack_runs"
    use_compact_2d = bool(USE_COMPACT_2D_OUTPUT_DEFAULT)
    scatter_compact_to_dense = bool(SCATTER_COMPACT_TO_DENSE_DEFAULT)
    if int(weight_row_offset) != 0:
        # Keep offset semantics correct without changing packed layout assumptions.
        weight = weight[int(weight_row_offset) : int(weight_row_offset) + int(out.shape[-1]), :]
        weight_row_offset = 0
    if path == "prepack_blocks_2d":
        need_dense_scatter = bool(active_out_blocks is not None) and (not bool(return_compact))
        compact_out, _active_n = _prepacked_block_sparse_gemm_2d_from_active_blocks(
            x=x,
            weight=weight,
            active_k_blocks=active_blocks,
            active_n_blocks=active_out_blocks,
            out=out,
            cache_prefix=f"{cache_prefix}_2d",
            proj_tag=proj_tag,
            sort_active_k_blocks=sort_blocks,
            sort_active_n_blocks=sort_blocks,
            zero_fill_out=bool(scatter_compact_to_dense or need_dense_scatter),
            compact_output=use_compact_2d,
            scatter_compact_to_dense=bool(scatter_compact_to_dense or need_dense_scatter),
            active_k_blocks_limit=active_blocks_limit,
            active_n_blocks_limit=active_out_blocks_limit,
        )
        if bool(return_compact):
            return compact_out, _active_n
        if compact_out is not None and int(out.shape[-1]) == int(compact_out.shape[-1]):
            out.copy_(compact_out)
        elif compact_out is not None and _active_n is not None and int(_active_n.numel()) > 0:
            out.zero_()
            _scatter_compact_to_dense_output(
                compact_out=compact_out,
                active_n_blocks=_active_n,
                block_n=max(16, int(INDEX_BLOCK_OUT_META_SIZE)),
                out_dense=out,
            )
        return None, _active_n
    if path == "blockmeta":
        if not bool(ENABLE_LEGACY_BLOCKMETA_PATH):
            raise RuntimeError(
                "Legacy blockmeta/mask_u8 path is disabled by default. "
                "Set TEAL_EXP_ENABLE_LEGACY_BLOCKMETA_PATH=1 to enable."
            )
        if mask_u8 is None:
            mask_u8 = _active_blocks_to_mask_u8(
                active_blocks=active_blocks,
                dim=int(x.size(-1)),
                block_k=max(16, int(INDEX_BLOCK_META_SIZE)),
                device=str(x.device),
            )
        _indexed_sparse_gemm_from_active_blocks_blockmeta(
            x=x,
            weight=weight,
            mask_u8=mask_u8,
            active_blocks=active_blocks,
            out=out,
            weight_row_offset=weight_row_offset,
        )
        return None, None

    if path == "prepack_bitmap_tilewalk":
        _prepacked_block_sparse_gemm_from_active_blocks_bitmap_tilewalk(
            x=x,
            weight=weight,
            active_blocks=active_blocks,
            out=out,
            cache_prefix=f"{cache_prefix}_bitmap",
            sort_active_blocks=sort_blocks,
            active_blocks_limit=active_blocks_limit,
        )
        return None, None

    _prepacked_block_sparse_gemm_from_active_blocks(
        x=x,
        weight=weight,
        active_blocks=active_blocks,
        out=out,
        cache_prefix=f"{cache_prefix}_k1d",
        proj_tag=proj_tag,
        sort_active_blocks=sort_blocks,
        use_run_compressed=(use_run and path == "prepack_runs"),
        active_blocks_limit=active_blocks_limit,
    )
    return None, None


def _bench_dense_mm(x2d: torch.Tensor, w_t: torch.Tensor, out2d: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        torch.mm(x2d, w_t, out=out2d)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        torch.mm(x2d, w_t, out=out2d)
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e) / max(1, iters))


def _active_blocks_to_mask_u8(active_blocks: torch.Tensor, dim: int, block_k: int, device: str) -> torch.Tensor:
    mask = torch.zeros((int(dim),), device=device, dtype=torch.uint8)
    if int(active_blocks.numel()) <= 0:
        return mask
    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    starts = active_blocks.to(dtype=torch.int64) * int(block_k)
    for s in starts.tolist():
        e = min(int(dim), int(s + block_k))
        mask[int(s):int(e)] = 1
    return mask


def _bench_sparse_path(
    x3d: torch.Tensor,
    weight: torch.Tensor,
    active_blocks: torch.Tensor,
    active_out_blocks: torch.Tensor | None,
    out3d: torch.Tensor,
    warmup: int,
    iters: int,
    cache_prefix: str,
    kernel_path: str,
    sort_active_blocks: bool,
    use_run_compressed: bool,
    compact_output: bool,
    scatter_compact_to_dense: bool,
) -> float:
    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    proj_tag = str(cache_prefix).split("_", 1)[0]
    _get_prepacked_weight_block_major(weight, block_k, cache_prefix=cache_prefix)
    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(dtype=torch.int32)
    active_blocks = active_blocks.contiguous()
    if sort_active_blocks:
        active_blocks = _sort_active_blocks_asc(active_blocks)
    mask_u8 = _active_blocks_to_mask_u8(active_blocks, int(x3d.size(-1)), block_k, str(x3d.device))

    def _run_once() -> None:
        if kernel_path == "blockmeta":
            _indexed_sparse_gemm_from_active_blocks_blockmeta(
                x=x3d,
                weight=weight,
                mask_u8=mask_u8,
                active_blocks=active_blocks,
                out=out3d,
            )
            return
        if kernel_path == "prepack_blocks_2d":
            _prepacked_block_sparse_gemm_2d_from_active_blocks(
                x=x3d,
                weight=weight,
                active_k_blocks=active_blocks,
                active_n_blocks=active_out_blocks,
                out=out3d,
                cache_prefix=cache_prefix,
                proj_tag=proj_tag,
                sort_active_k_blocks=sort_active_blocks,
                sort_active_n_blocks=sort_active_blocks,
                zero_fill_out=bool(scatter_compact_to_dense),
                compact_output=bool(compact_output),
                scatter_compact_to_dense=bool(scatter_compact_to_dense),
            )
            return
        if kernel_path == "prepack_bitmap_tilewalk":
            _prepacked_block_sparse_gemm_from_active_blocks_bitmap_tilewalk(
                x=x3d,
                weight=weight,
                active_blocks=active_blocks,
                out=out3d,
                cache_prefix=f"{cache_prefix}_bitmap",
                sort_active_blocks=sort_active_blocks,
            )
            return
        _prepacked_block_sparse_gemm_from_active_blocks(
            x=x3d,
            weight=weight,
            active_blocks=active_blocks,
            out=out3d,
            cache_prefix=cache_prefix,
            proj_tag=proj_tag,
            sort_active_blocks=sort_active_blocks,
            use_run_compressed=bool(use_run_compressed and kernel_path == "prepack_runs"),
        )

    for _ in range(warmup):
        _run_once()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        _run_once()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e) / max(1, iters))


def _make_keep_blocks(dim: int, block_k: int, keep_ratio: float, mode: str, device: str) -> torch.Tensor:
    n_blocks = (dim + block_k - 1) // block_k
    keep_blocks = max(1, int(round(n_blocks * keep_ratio)))
    ids = torch.arange(n_blocks, device=device, dtype=torch.int32)
    if mode in {"regular", "regular_sorted"}:
        stride = max(1, int(round(1.0 / max(keep_ratio, 1e-6))))
        sel = ids[(ids % stride) == 0]
        if int(sel.numel()) < keep_blocks:
            extra = ids[~torch.isin(ids, sel)][: keep_blocks - int(sel.numel())]
            sel = torch.cat([sel, extra], dim=0)
        return _sort_active_blocks_asc(sel[:keep_blocks].contiguous())
    if mode == "regular_unsorted":
        stride = max(1, int(round(1.0 / max(keep_ratio, 1e-6))))
        sel = ids[(ids % stride) == 0]
        if int(sel.numel()) < keep_blocks:
            extra = ids[~torch.isin(ids, sel)][: keep_blocks - int(sel.numel())]
            sel = torch.cat([sel, extra], dim=0)
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        p = torch.randperm(int(sel.numel()), generator=g, device=device, dtype=torch.int64)
        return sel.index_select(0, p)[:keep_blocks].to(torch.int32).contiguous()
    g = torch.Generator(device=device)
    g.manual_seed(1234)
    perm = torch.randperm(n_blocks, generator=g, device=device, dtype=torch.int64)
    sel = perm[:keep_blocks].to(torch.int32).contiguous()
    if mode in {"random_sorted"}:
        sel = _sort_active_blocks_asc(sel)
    return sel


def _run_kernel_only_bench(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = args.device
    if str(args.dtype).strip().lower() == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    bsz = int(args.batch_size)
    seq_tile = int(args.seq_tile)
    hidden = int(args.hidden)
    interm = int(args.intermediate)
    qkv_out = int(args.qkv_out)
    o_out = int(args.o_out)
    warmup = int(args.warmup)
    iters = int(args.iters)

    bt = bsz * seq_tile
    x_hidden = torch.randn((bsz, seq_tile, hidden), device=device, dtype=dtype)
    x_interm = torch.randn((bsz, seq_tile, interm), device=device, dtype=dtype)
    xh2 = x_hidden.view(bt, hidden).contiguous()
    xi2 = x_interm.view(bt, interm).contiguous()

    w_qkv = torch.randn((qkv_out, hidden), device=device, dtype=dtype)
    w_o = torch.randn((o_out, hidden), device=device, dtype=dtype)
    w_gate = torch.randn((interm, hidden), device=device, dtype=dtype)
    w_up = torch.randn((interm, hidden), device=device, dtype=dtype)
    w_down = torch.randn((hidden, interm), device=device, dtype=dtype)

    out_dense_qkv = torch.empty((bt, qkv_out), device=device, dtype=dtype)
    out_dense_o = torch.empty((bt, o_out), device=device, dtype=dtype)
    out_dense_gate = torch.empty((bt, interm), device=device, dtype=dtype)
    out_dense_up = torch.empty((bt, interm), device=device, dtype=dtype)
    out_dense_down = torch.empty((bt, hidden), device=device, dtype=dtype)

    out_sparse_qkv = torch.empty((bsz, seq_tile, qkv_out), device=device, dtype=dtype)
    out_sparse_o = torch.empty((bsz, seq_tile, o_out), device=device, dtype=dtype)
    out_sparse_gate = torch.empty((bsz, seq_tile, interm), device=device, dtype=dtype)
    out_sparse_up = torch.empty((bsz, seq_tile, interm), device=device, dtype=dtype)
    out_sparse_down = torch.empty((bsz, seq_tile, hidden), device=device, dtype=dtype)

    block_k = max(16, int(INDEX_BLOCK_META_SIZE))
    keep = float(1.0 - float(args.sparsity))
    out_sparsity = float(args.sparsity) if float(args.n_sparsity) < 0.0 else float(args.n_sparsity)
    keep_out = float(1.0 - out_sparsity)
    block_out_meta = max(16, int(INDEX_BLOCK_OUT_META_SIZE))

    dense_ms = {
        "qkv": _bench_dense_mm(xh2, w_qkv.t().contiguous(), out_dense_qkv, warmup, iters),
        "o": _bench_dense_mm(xh2, w_o.t().contiguous(), out_dense_o, warmup, iters),
        "gate": _bench_dense_mm(xh2, w_gate.t().contiguous(), out_dense_gate, warmup, iters),
        "up": _bench_dense_mm(xh2, w_up.t().contiguous(), out_dense_up, warmup, iters),
        "down": _bench_dense_mm(xi2, w_down.t().contiguous(), out_dense_down, warmup, iters),
    }

    if bool(args.locality_compare):
        patterns = [
            ("random_unsorted", "random", False),
            ("random_sorted", "random", True),
            ("regular_sorted", "regular", True),
        ]
    else:
        patterns = [("single", str(args.block_select_mode), bool(int(args.sort_active_blocks)))]

    rows = []
    summary_rows = []
    for pattern_name, select_mode, sort_blocks in patterns:
        ab_hidden = _make_keep_blocks(hidden, block_k, keep, select_mode, device)
        ab_interm = _make_keep_blocks(interm, block_k, keep, select_mode, device)
        ab_qkv_out = _make_keep_blocks(qkv_out, block_out_meta, keep_out, select_mode, device)
        ab_o_out = _make_keep_blocks(o_out, block_out_meta, keep_out, select_mode, device)
        ab_interm_out = _make_keep_blocks(interm, block_out_meta, keep_out, select_mode, device)
        ab_hidden_out = _make_keep_blocks(hidden, block_out_meta, keep_out, select_mode, device)
        splitk_plan = {
            "qkv": _choose_split_k_1d(
                bt=bt,
                num_active_blocks=int(ab_hidden.numel()),
                n_out=qkv_out,
                use_run_compressed=bool(int(args.use_run_compressed_metadata) and str(args.kernel_path) == "prepack_runs"),
                proj_tag="qkv",
            ),
            "o": _choose_split_k_1d(
                bt=bt,
                num_active_blocks=int(ab_hidden.numel()),
                n_out=o_out,
                use_run_compressed=bool(int(args.use_run_compressed_metadata) and str(args.kernel_path) == "prepack_runs"),
                proj_tag="o",
            ),
            "gate": _choose_split_k_1d(
                bt=bt,
                num_active_blocks=int(ab_hidden.numel()),
                n_out=interm,
                use_run_compressed=bool(int(args.use_run_compressed_metadata) and str(args.kernel_path) == "prepack_runs"),
                proj_tag="gate",
            ),
            "up": _choose_split_k_1d(
                bt=bt,
                num_active_blocks=int(ab_hidden.numel()),
                n_out=interm,
                use_run_compressed=bool(int(args.use_run_compressed_metadata) and str(args.kernel_path) == "prepack_runs"),
                proj_tag="up",
            ),
            "down": _choose_split_k_1d(
                bt=bt,
                num_active_blocks=int(ab_interm.numel()),
                n_out=hidden,
                use_run_compressed=bool(int(args.use_run_compressed_metadata) and str(args.kernel_path) == "prepack_runs"),
                proj_tag="down",
            ),
        }
        splitk_reduce_plan = {
            proj: _choose_split_k_1d_reduce_mode(bt=bt, n_out=(qkv_out if proj == "qkv" else o_out if proj == "o" else hidden if proj == "down" else interm), num_split_k=int(sk), proj_tag=proj)
            for proj, sk in splitk_plan.items()
        }
        sparse_ms = {
            "qkv": _bench_sparse_path(
                x_hidden,
                w_qkv,
                ab_hidden,
                ab_qkv_out,
                out_sparse_qkv,
                warmup,
                iters,
                f"qkv_{pattern_name}",
                kernel_path=str(args.kernel_path),
                sort_active_blocks=sort_blocks,
                use_run_compressed=bool(int(args.use_run_compressed_metadata)),
                compact_output=bool(int(args.compact_output)),
                scatter_compact_to_dense=bool(int(args.scatter_compact_to_dense)),
            ),
            "o": _bench_sparse_path(
                x_hidden,
                w_o,
                ab_hidden,
                ab_o_out,
                out_sparse_o,
                warmup,
                iters,
                f"o_{pattern_name}",
                kernel_path=str(args.kernel_path),
                sort_active_blocks=sort_blocks,
                use_run_compressed=bool(int(args.use_run_compressed_metadata)),
                compact_output=bool(int(args.compact_output)),
                scatter_compact_to_dense=bool(int(args.scatter_compact_to_dense)),
            ),
            "gate": _bench_sparse_path(
                x_hidden,
                w_gate,
                ab_hidden,
                ab_interm_out,
                out_sparse_gate,
                warmup,
                iters,
                f"gate_{pattern_name}",
                kernel_path=str(args.kernel_path),
                sort_active_blocks=sort_blocks,
                use_run_compressed=bool(int(args.use_run_compressed_metadata)),
                compact_output=bool(int(args.compact_output)),
                scatter_compact_to_dense=bool(int(args.scatter_compact_to_dense)),
            ),
            "up": _bench_sparse_path(
                x_hidden,
                w_up,
                ab_hidden,
                ab_interm_out,
                out_sparse_up,
                warmup,
                iters,
                f"up_{pattern_name}",
                kernel_path=str(args.kernel_path),
                sort_active_blocks=sort_blocks,
                use_run_compressed=bool(int(args.use_run_compressed_metadata)),
                compact_output=bool(int(args.compact_output)),
                scatter_compact_to_dense=bool(int(args.scatter_compact_to_dense)),
            ),
            "down": _bench_sparse_path(
                x_interm,
                w_down,
                ab_interm,
                ab_hidden_out,
                out_sparse_down,
                warmup,
                iters,
                f"down_{pattern_name}",
                kernel_path=str(args.kernel_path),
                sort_active_blocks=sort_blocks,
                use_run_compressed=bool(int(args.use_run_compressed_metadata)),
                compact_output=bool(int(args.compact_output)),
                scatter_compact_to_dense=bool(int(args.scatter_compact_to_dense)),
            ),
        }
        for proj in ("qkv", "o", "gate", "up", "down"):
            rows.append(
                {
                    "pattern": pattern_name,
                    "projection": proj,
                    "kernel_path": str(args.kernel_path),
                    "dtype": str(args.dtype),
                    "sorted_blocks": bool(sort_blocks),
                    "run_compressed_requested": bool(int(args.use_run_compressed_metadata)),
                    "run_compressed_metadata": bool(
                        int(args.use_run_compressed_metadata) and str(args.kernel_path) == "prepack_runs"
                    ),
                    "compact_output": bool(int(args.compact_output)),
                    "scatter_compact_to_dense": bool(int(args.scatter_compact_to_dense)),
                    "block_k": int(block_k),
                    "block_out_meta": int(block_out_meta),
                    "n_sparsity": float(out_sparsity),
                    "split_k_1d_selected": int(splitk_plan[proj]),
                    "split_k_1d_reduce_mode_selected": str(splitk_reduce_plan[proj]),
                    "dense_ms_per_call": float(dense_ms[proj]),
                    "sparse_ms_per_call": float(sparse_ms[proj]),
                    "dense_over_sparse_speedup": float(dense_ms[proj] / max(sparse_ms[proj], 1e-12)),
                }
            )
        dense_total = sum(dense_ms.values())
        sparse_total = sum(sparse_ms.values())
        summary_rows.append(
            {
                "pattern": pattern_name,
                "kernel_path": str(args.kernel_path),
                "dtype": str(args.dtype),
                "sorted_blocks": bool(sort_blocks),
                "run_compressed_requested": bool(int(args.use_run_compressed_metadata)),
                "run_compressed_metadata": bool(
                    int(args.use_run_compressed_metadata) and str(args.kernel_path) == "prepack_runs"
                ),
                "compact_output": bool(int(args.compact_output)),
                "scatter_compact_to_dense": bool(int(args.scatter_compact_to_dense)),
                "block_k": int(block_k),
                "block_out_meta": int(block_out_meta),
                "n_sparsity": float(out_sparsity),
                "dense_total_ms_per_step": float(dense_total),
                "sparse_total_ms_per_step": float(sparse_total),
                "dense_over_sparse_speedup": float(dense_total / max(sparse_total, 1e-12)),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "kernel_only_projection_bench.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_csv = out_dir / "kernel_only_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    meta = {
        "mode": "kernel_only_projection_bench",
        "config": {
            "device": device,
            "dtype": str(args.dtype),
            "seed": int(args.seed),
            "batch_size": bsz,
            "seq_tile": seq_tile,
            "sparsity": float(args.sparsity),
            "n_sparsity": float(out_sparsity),
            "keep_ratio": keep,
            "keep_ratio_out": keep_out,
            "block_k": block_k,
            "block_out_meta": int(block_out_meta),
            "block_select_mode": args.block_select_mode,
            "locality_compare": bool(args.locality_compare),
            "kernel_path": str(args.kernel_path),
            "sort_active_blocks": bool(int(args.sort_active_blocks)),
            "use_run_compressed_metadata": bool(int(args.use_run_compressed_metadata)),
            "compact_output": bool(int(args.compact_output)),
            "scatter_compact_to_dense": bool(int(args.scatter_compact_to_dense)),
            "split_k_1d_enabled_env_default": bool(ENABLE_SPLIT_K_1D_DEFAULT),
            "split_k_1d_reduce_mode_env_default": str(_SPLIT_K_1D_REDUCE_MODE),
            "split_k_1d_min_active_blocks": int(SPLIT_K_1D_MIN_ACTIVE_BLOCKS),
            "split_k_1d_min_n_out": int(SPLIT_K_1D_MIN_N_OUT),
            "split_k_2d_n_tile_repeat": int(SPLIT_K_2D_N_TILE_REPEAT),
            "split_k_2d_k_chunk": int(SPLIT_K_2D_K_CHUNK),
            "hidden": hidden,
            "intermediate": interm,
            "qkv_out": qkv_out,
            "o_out": o_out,
            "warmup": warmup,
            "iters": iters,
        },
        "summary": summary_rows,
        "kernel_path_info": {
            "runtime_fast_block_kernel_path_env_default": FAST_BLOCK_KERNEL_PATH,
            "runtime_sort_active_blocks_env_default": bool(SORT_ACTIVE_BLOCKS_DEFAULT),
            "runtime_use_run_compressed_env_default": bool(USE_RUN_COMPRESSED_METADATA_DEFAULT),
            "runtime_use_delta_active_blocks_env_default": bool(USE_DELTA_ACTIVE_BLOCKS_DEFAULT),
            "runtime_use_compact_2d_output_env_default": bool(USE_COMPACT_2D_OUTPUT_DEFAULT),
            "runtime_use_block_baseptr_metadata_env_default": bool(USE_BLOCK_BASEPTR_METADATA_DEFAULT),
            "autotune_enabled_for_prepacked": True,
            "autotune_note": "Triton selects launch config internally; selected config is not exposed by this script.",
        },
        "cache_stats": {
            "weight_prepack_hits": int(_WEIGHT_PREPACK_CACHE["hits"]),
            "weight_prepack_misses": int(_WEIGHT_PREPACK_CACHE["misses"]),
            "run_meta_hits": int(_RUN_META_CACHE["hits"]),
            "run_meta_misses": int(_RUN_META_CACHE["misses"]),
            "delta_state_hits": int(_DELTA_BLOCK_STATE_CACHE["hits"]),
            "delta_state_misses": int(_DELTA_BLOCK_STATE_CACHE["misses"]),
            "delta_state_updates": int(_DELTA_BLOCK_STATE_CACHE["updates"]),
            "delta_n_state_hits": int(_DELTA_N_BLOCK_STATE_CACHE["hits"]),
            "delta_n_state_misses": int(_DELTA_N_BLOCK_STATE_CACHE["misses"]),
            "delta_n_state_updates": int(_DELTA_N_BLOCK_STATE_CACHE["updates"]),
            "splitk_accum_hits": int(_SPLITK_ACCUM_CACHE["hits"]),
            "splitk_accum_misses": int(_SPLITK_ACCUM_CACHE["misses"]),
            "splitk_partial_hits": int(_SPLITK_PARTIAL_CACHE["hits"]),
            "splitk_partial_misses": int(_SPLITK_PARTIAL_CACHE["misses"]),
            "temp_tensor_hits": int(_TEMP_TENSOR_CACHE["hits"]),
            "temp_tensor_misses": int(_TEMP_TENSOR_CACHE["misses"]),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out_csv)
    print(summary_csv)
    print(out_dir / "run_meta.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal runtime Triton kernel-only benchmark (no grouping/shared-score).")
    p.add_argument("--kernel_only_bench", action="store_true", help="Run kernel-only benchmark mode.")
    p.add_argument("--out_dir", default="outputs/kernel_only_projection_bench")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default=RUNTIME_DEFAULT_DTYPE)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_tile", type=int, default=1)
    p.add_argument("--sparsity", type=float, default=0.4)
    p.add_argument(
        "--n_sparsity",
        type=float,
        default=-1.0,
        help="Output-axis sparsity for 2D block sparse mode; negative uses --sparsity.",
    )
    p.add_argument(
        "--block_select_mode",
        choices=["regular", "random", "regular_sorted", "regular_unsorted", "random_sorted", "random_unsorted"],
        default="regular",
    )
    p.add_argument("--locality_compare", action="store_true", help="Run locality patterns: random-unsorted/random-sorted/regular-sorted.")
    p.add_argument("--sort_active_blocks", type=int, choices=[0, 1], default=1)
    p.add_argument("--use_run_compressed_metadata", type=int, choices=[0, 1], default=1)
    p.add_argument(
        "--compact_output",
        type=int,
        choices=[0, 1],
        default=1 if USE_COMPACT_2D_OUTPUT_DEFAULT else 0,
        help="Enable compact/index-aware output in 2D sparse path.",
    )
    p.add_argument(
        "--scatter_compact_to_dense",
        type=int,
        choices=[0, 1],
        default=1 if SCATTER_COMPACT_TO_DENSE_DEFAULT else 0,
        help="Scatter compact 2D output back to dense output buffer (for compatibility).",
    )
    p.add_argument(
        "--kernel_path",
        choices=["prepack_runs", "prepack_blocks", "prepack_blocks_2d", "prepack_bitmap_tilewalk", "blockmeta"],
        default="prepack_blocks_2d",
    )
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--intermediate", type=int, default=11008)
    p.add_argument("--qkv_out", type=int, default=12288)
    p.add_argument("--o_out", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=80)
    p.add_argument("--iters", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.kernel_only_bench:
        _run_kernel_only_bench(args)
    else:
        print("No action. Use --kernel_only_bench to run the kernel-only benchmark.")
