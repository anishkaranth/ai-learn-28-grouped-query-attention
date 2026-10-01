"""Grouped Query Attention (GQA) and Multi-Query Attention (MQA) helpers.

Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints" (2023). Used in Llama-2 / Llama-3, Mistral, etc.

Standard multi-head attention (MHA) stores a separate K and V projection per
query head. That dominates KV-cache memory at decode time.

GQA / MQA share K/V across groups of query heads:
  - MHA: n_kv_heads == n_q_heads  (no sharing)
  - GQA: 1 < n_kv_heads < n_q_heads, and n_q_heads % n_kv_heads == 0
  - MQA: n_kv_heads == 1           (all query heads share one K/V)

At attention time, each KV head is expanded (repeated) ``n_rep`` times so the
head axis matches the query heads: n_rep = n_q_heads // n_kv_heads.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

Mode = Literal["mha", "gqa", "mqa"]


def validate_heads(n_q_heads: int, n_kv_heads: int) -> None:
    if n_q_heads < 1 or n_kv_heads < 1:
        raise ValueError("n_q_heads and n_kv_heads must be >= 1")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )


def n_rep(n_q_heads: int, n_kv_heads: int) -> int:
    """How many query heads share each KV head."""
    validate_heads(n_q_heads, n_kv_heads)
    return n_q_heads // n_kv_heads


def mode_from_heads(n_q_heads: int, n_kv_heads: int) -> Mode:
    validate_heads(n_q_heads, n_kv_heads)
    if n_kv_heads == n_q_heads:
        return "mha"
    if n_kv_heads == 1:
        return "mqa"
    return "gqa"


def repeat_kv(x: np.ndarray, n_repeat: int) -> np.ndarray:
    """Expand KV along the head axis by repeating each KV head ``n_repeat`` times.

    Parameters
    ----------
    x : (B, n_kv_heads, T, d_head) or (n_kv_heads, T, d_head)
    n_repeat : n_q_heads // n_kv_heads

    Returns
    -------
    out : (B, n_q_heads, T, d_head)  — or without batch if input had none
    """
    x = np.asarray(x, dtype=np.float64)
    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")
    if n_repeat == 1:
        return x.copy()

    if x.ndim == 3:
        # (n_kv, T, Dh) -> (n_kv, 1, T, Dh) -> broadcast -> reshape
        n_kv, T, Dh = x.shape
        x = x[:, None, :, :]
        x = np.broadcast_to(x, (n_kv, n_repeat, T, Dh))
        return np.ascontiguousarray(x.reshape(n_kv * n_repeat, T, Dh))

    if x.ndim == 4:
        B, n_kv, T, Dh = x.shape
        x = x[:, :, None, :, :]
        x = np.broadcast_to(x, (B, n_kv, n_repeat, T, Dh))
        return np.ascontiguousarray(x.reshape(B, n_kv * n_repeat, T, Dh))

    raise ValueError(f"expected 3D or 4D KV tensor, got shape {x.shape}")


def kv_cache_bytes(
    *,
    batch: int,
    seq_len: int,
    n_kv_heads: int,
    d_head: int,
    dtype_bytes: int = 4,
) -> int:
    """Bytes to store K and V for a full sequence (both tensors).

    Cache layout (per layer): 2 * B * T * n_kv_heads * d_head * dtype_bytes.
    """
    if min(batch, seq_len, n_kv_heads, d_head, dtype_bytes) < 1:
        raise ValueError("all kv_cache_bytes args must be >= 1")
    return int(2 * batch * seq_len * n_kv_heads * d_head * dtype_bytes)


def kv_cache_elements(
    *,
    batch: int,
    seq_len: int,
    n_kv_heads: int,
    d_head: int,
) -> int:
    """Float elements for K+V (dtype-agnostic count)."""
    return int(2 * batch * seq_len * n_kv_heads * d_head)


def kv_savings_table(
    n_q_heads: int,
    *,
    n_kv_gqa: int,
    batch: int = 1,
    seq_len: int = 2048,
    d_head: int = 64,
    dtype_bytes: int = 4,
) -> dict:
    """Structural KV-cache comparison for MHA / GQA / MQA."""
    validate_heads(n_q_heads, n_kv_gqa)
    validate_heads(n_q_heads, 1)

    rows = {}
    for name, n_kv in (("mha", n_q_heads), ("gqa", n_kv_gqa), ("mqa", 1)):
        elems = kv_cache_elements(
            batch=batch, seq_len=seq_len, n_kv_heads=n_kv, d_head=d_head
        )
        nbytes = kv_cache_bytes(
            batch=batch,
            seq_len=seq_len,
            n_kv_heads=n_kv,
            d_head=d_head,
            dtype_bytes=dtype_bytes,
        )
        rows[name] = {
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv,
            "n_rep": n_rep(n_q_heads, n_kv),
            "elements": elems,
            "bytes": nbytes,
            "mib": round(nbytes / (1024 * 1024), 4),
        }

    mha_b = rows["mha"]["bytes"]
    for name in rows:
        rows[name]["vs_mha_ratio"] = round(mha_b / rows[name]["bytes"], 4)
        rows[name]["fraction_of_mha"] = round(rows[name]["bytes"] / mha_b, 4)

    return {
        "batch": batch,
        "seq_len": seq_len,
        "d_head": d_head,
        "dtype_bytes": dtype_bytes,
        "n_q_heads": n_q_heads,
        "n_kv_gqa": n_kv_gqa,
        "modes": rows,
        "headline": {
            "gqa_reduction_vs_mha": rows["gqa"]["vs_mha_ratio"],
            "mqa_reduction_vs_mha": rows["mqa"]["vs_mha_ratio"],
            "expected_gqa": float(n_q_heads / n_kv_gqa),
            "expected_mqa": float(n_q_heads),
        },
    }


def expand_kv_for_gqa(
    K: np.ndarray,
    V: np.ndarray,
    n_q_heads: int,
    n_kv_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat K and V so head count matches ``n_q_heads``."""
    rep = n_rep(n_q_heads, n_kv_heads)
    return repeat_kv(K, rep), repeat_kv(V, rep)
