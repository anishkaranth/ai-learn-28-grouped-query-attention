"""GQA/MQA experiments: KV savings, output fidelity vs MHA, matching helpers."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from attention import GQAAttention, make_attention, softmax
from gqa import kv_savings_table, mode_from_heads, repeat_kv, validate_heads


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_cosine_batched(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine over batch x time flattened per-token vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    # (B, T, D) -> per (B,T) cosine
    flat_a = a.reshape(-1, a.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    sims = []
    for i in range(flat_a.shape[0]):
        sims.append(cosine_sim(flat_a[i], flat_b[i]))
    return float(np.mean(sims))


def structural_kv_report(
    n_q_heads: int = 8,
    n_kv_gqa: int = 2,
    *,
    batch: int = 1,
    seq_len: int = 2048,
    d_head: int = 64,
) -> Dict[str, Any]:
    """Table of KV-cache bytes for MHA / GQA / MQA."""
    table = kv_savings_table(
        n_q_heads,
        n_kv_gqa=n_kv_gqa,
        batch=batch,
        seq_len=seq_len,
        d_head=d_head,
        dtype_bytes=4,
    )
    h = table["headline"]
    table["checks"] = {
        "gqa_ratio_matches_theory": abs(h["gqa_reduction_vs_mha"] - h["expected_gqa"]) < 1e-6,
        "mqa_ratio_matches_theory": abs(h["mqa_reduction_vs_mha"] - h["expected_mqa"]) < 1e-6,
    }
    return table


def expand_roundtrip_check(
    n_q_heads: int = 8,
    n_kv_heads: int = 2,
    *,
    batch: int = 2,
    seq_len: int = 16,
    d_head: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify repeat_kv expands head axis correctly and preserves values."""
    validate_heads(n_q_heads, n_kv_heads)
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(batch, n_kv_heads, seq_len, d_head))
    rep = n_q_heads // n_kv_heads
    K_exp = repeat_kv(K, rep)
    ok_shape = K_exp.shape == (batch, n_q_heads, seq_len, d_head)
    # Each group of ``rep`` consecutive query heads should match the KV head
    group_ok = True
    for hkv in range(n_kv_heads):
        for r in range(rep):
            hq = hkv * rep + r
            if not np.allclose(K_exp[:, hq], K[:, hkv]):
                group_ok = False
                break
    return {
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "n_rep": rep,
        "in_shape": list(K.shape),
        "out_shape": list(K_exp.shape),
        "shape_ok": bool(ok_shape),
        "values_ok": bool(group_ok),
        "pass": bool(ok_shape and group_ok),
        "mode": mode_from_heads(n_q_heads, n_kv_heads),
    }


def random_weight_fidelity(
    d_model: int = 64,
    n_q_heads: int = 8,
    n_kv_gqa: int = 2,
    *,
    seq_len: int = 32,
    batch: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare GQA/MQA attention outputs to an MHA reference with shared init.

    Story: start from an MHA model. Collapse K/V by *averaging* within each
    query-head group (GQA) or across all heads (MQA), then expand via repeat.
    Cosine fidelity measures how much sharing changes the attention output
    before any training - GQA (milder sharing) should stay closer to MHA than MQA.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, seq_len, d_model)) * 0.5
    mha = make_attention(
        d_model, "mha", n_q_heads=n_q_heads, n_kv_gqa=n_kv_gqa, rng=rng, scale=0.15
    )
    out_mha = mha.forward(x)

    def collapse_and_run(n_kv: int) -> np.ndarray:
        """Average MHA KV weight columns within groups -> GQA/MQA, then forward."""
        attn = make_attention(
            d_model,
            mode_from_heads(n_q_heads, n_kv),
            n_q_heads=n_q_heads,
            n_kv_gqa=n_kv,
            scale=0.0,
        )
        attn.W_q = mha.W_q.copy()
        attn.W_o = mha.W_o.copy()
        attn.b_o = mha.b_o.copy()
        # mha W_k: (D, Hq*Dh); reshape to (D, Hq, Dh), average groups
        Dh = mha.d_head
        Wk = mha.W_k.reshape(d_model, n_q_heads, Dh)
        Wv = mha.W_v.reshape(d_model, n_q_heads, Dh)
        rep = n_q_heads // n_kv
        Wk_g = Wk.reshape(d_model, n_kv, rep, Dh).mean(axis=2)
        Wv_g = Wv.reshape(d_model, n_kv, rep, Dh).mean(axis=2)
        attn.W_k = Wk_g.reshape(d_model, n_kv * Dh)
        attn.W_v = Wv_g.reshape(d_model, n_kv * Dh)
        return attn.forward(x)

    out_gqa = collapse_and_run(n_kv_gqa)
    out_mqa = collapse_and_run(1)

    # Also: independent random GQA/MQA (no collapse) - lower fidelity expected
    rng2 = np.random.default_rng(seed + 7)
    gqa_rand = make_attention(
        d_model, "gqa", n_q_heads=n_q_heads, n_kv_gqa=n_kv_gqa, rng=rng2, scale=0.15
    )
    # Match Q/O to MHA so only KV sharing differs structurally with expand
    gqa_rand.W_q = mha.W_q.copy()
    gqa_rand.W_o = mha.W_o.copy()
    gqa_rand.b_o = mha.b_o.copy()
    out_gqa_rand = gqa_rand.forward(x)

    cos_gqa = mean_cosine_batched(out_gqa, out_mha)
    cos_mqa = mean_cosine_batched(out_mqa, out_mha)
    cos_gqa_rand = mean_cosine_batched(out_gqa_rand, out_mha)

    return {
        "d_model": d_model,
        "n_q_heads": n_q_heads,
        "n_kv_gqa": n_kv_gqa,
        "seq_len": seq_len,
        "batch": batch,
        "cosine_gqa_collapsed": round(cos_gqa, 6),
        "cosine_mqa_collapsed": round(cos_mqa, 6),
        "cosine_gqa_random_kv": round(cos_gqa_rand, 6),
        "gqa_closer_than_mqa": bool(cos_gqa > cos_mqa),
        "mha_self_cosine": 1.0,
    }


# Re-export matching helpers (stable import path for train_task / notebooks)
from matching_data import (  # noqa: E402
    _query_logits,
    attention_mass_on_target,
    make_copy_match_batch,
    matching_accuracy,
)


def train_matching_task(*args, **kwargs):
    from train_task import train_matching_task as _train

    return _train(*args, **kwargs)


def compare_modes_trained(*args, **kwargs):
    from train_task import compare_modes_trained as _cmp

    return _cmp(*args, **kwargs)
