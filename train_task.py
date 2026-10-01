"""Train loop for planted-key matching under MHA / GQA / MQA."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from attention import GQAAttention, make_attention, softmax
from experiments import (
    _query_logits,
    attention_mass_on_target,
    make_copy_match_batch,
    matching_accuracy,
)


def train_matching_task(
    mode: str,
    d_model: int = 64,
    seq_len: int = 16,
    n_q_heads: int = 8,
    n_kv_gqa: int = 2,
    n_train: int = 640,
    n_test: int = 160,
    steps: int = 220,
    lr: float = 0.18,
    seed: int = 42,
    n_distractors: int = 2,
    logit_scale: float = 12.0,
) -> Dict[str, Any]:
    """Train Q/K (and V lightly) so attention at query selects the planted key."""
    mode_seed = {"mha": 0, "gqa": 1, "mqa": 2}[mode]
    rng = np.random.default_rng(seed + mode_seed)
    rng_init = np.random.default_rng(seed)  # shared init scale/seed base
    # Different mode -> different W_k/W_v shapes; share W_q init via copy from MHA
    base = make_attention(
        d_model,
        "mha",
        n_q_heads=n_q_heads,
        n_kv_gqa=n_kv_gqa,
        rng=rng_init,
        scale=0.08,
    )
    attn = make_attention(
        d_model,
        mode,  # type: ignore[arg-type]
        n_q_heads=n_q_heads,
        n_kv_gqa=n_kv_gqa,
        rng=np.random.default_rng(seed + 100 + mode_seed),
        scale=0.08,
    )
    # Share Q and O init so capacity differences are mainly in KV sharing
    attn.W_q = base.W_q.copy()
    attn.W_o = base.W_o.copy()
    attn.b_o = base.b_o.copy()
    # Initialize KV from averaged MHA columns (fairer start for GQA/MQA)
    Dh = attn.d_head
    Wk = base.W_k.reshape(d_model, n_q_heads, Dh)
    Wv = base.W_v.reshape(d_model, n_q_heads, Dh)
    n_kv = attn.n_kv_heads
    rep = n_q_heads // n_kv
    attn.W_k = Wk.reshape(d_model, n_kv, rep, Dh).mean(axis=2).reshape(d_model, n_kv * Dh)
    attn.W_v = Wv.reshape(d_model, n_kv, rep, Dh).mean(axis=2).reshape(d_model, n_kv * Dh)

    Xtr, Ytr, Qtr = make_copy_match_batch(
        n_train, seq_len, d_model, rng, n_distractors=n_distractors
    )
    Xte, Yte, Qte = make_copy_match_batch(
        n_test, seq_len, d_model, rng, n_distractors=n_distractors
    )

    history: List[Dict[str, Any]] = []
    for step in range(steps):
        idx = rng.integers(0, n_train, size=80)
        xb, yb, qb = Xtr[idx], Ytr[idx], Qtr[idx]
        B = len(qb)
        Dh = attn.d_head
        Hq = attn.n_q_heads
        Hkv = attn.n_kv_heads

        Q = (xb @ attn.W_q).reshape(B, seq_len, Hq, Dh).transpose(0, 2, 1, 3)
        K = (xb @ attn.W_k).reshape(B, seq_len, Hkv, Dh).transpose(0, 2, 1, 3)
        # Expand K for GQA/MQA
        if attn.n_rep > 1:
            K_exp = np.repeat(K, attn.n_rep, axis=1)
        else:
            K_exp = K
        scale = logit_scale / np.sqrt(Dh)
        logits = (Q @ K_exp.transpose(0, 1, 3, 2)) * scale  # (B,Hq,T,T)
        logits_mean = logits.mean(axis=1)
        row_logits = logits_mean[np.arange(B), qb.astype(int), :].copy()
        row_logits[np.arange(B), qb.astype(int)] = -1e9
        row_prob = softmax(row_logits, axis=-1)
        loss = -float(np.mean(np.log(row_prob[np.arange(B), yb] + 1e-12)))
        grad_row = row_prob.copy()
        grad_row[np.arange(B), yb] -= 1.0
        grad_row /= B

        g_logits = np.zeros_like(logits)
        for b in range(B):
            qi = int(qb[b])
            g_logits[b, :, qi, :] = grad_row[b][None, :] / Hq

        gQ = scale * (g_logits @ K_exp)
        gK_exp = scale * (g_logits.transpose(0, 1, 3, 2) @ Q)

        # Collapse grads on expanded KV heads back to n_kv heads (sum group)
        if attn.n_rep > 1:
            gK = gK_exp.reshape(B, Hkv, attn.n_rep, seq_len, Dh).sum(axis=2)
        else:
            gK = gK_exp

        gQ_flat = gQ.transpose(0, 2, 1, 3).reshape(B, seq_len, Hq * Dh)
        gK_flat = gK.transpose(0, 2, 1, 3).reshape(B, seq_len, Hkv * Dh)
        gWq = xb.reshape(-1, d_model).T @ gQ_flat.reshape(-1, Hq * Dh)
        gWk = xb.reshape(-1, d_model).T @ gK_flat.reshape(-1, Hkv * Dh)
        attn.W_q -= lr * gWq
        attn.W_k -= lr * gWk

        if step % 20 == 0 or step == steps - 1:
            acc = matching_accuracy(attn, Xte, Yte, Qte, logit_scale)
            mass = attention_mass_on_target(attn, Xte, Yte, Qte, logit_scale)
            history.append(
                {
                    "step": step,
                    "loss": round(loss, 6),
                    "acc": round(acc, 6),
                    "mass": round(mass, 6),
                }
            )

    final_acc = matching_accuracy(attn, Xte, Yte, Qte, logit_scale)
    final_mass = attention_mass_on_target(attn, Xte, Yte, Qte, logit_scale)
    # Demo attention row for first test example
    rows = _query_logits(attn, Xte[:1], Qte[:1], logit_scale)
    probs = softmax(rows, axis=-1)[0]
    return {
        "mode": mode,
        "n_q_heads": n_q_heads,
        "n_kv_heads": attn.n_kv_heads,
        "n_rep": attn.n_rep,
        "seq_len": seq_len,
        "d_model": d_model,
        "steps": steps,
        "final_acc": round(final_acc, 6),
        "final_mass": round(final_mass, 6),
        "chance_acc": round(1.0 / (seq_len - 1), 6),
        "history": history,
        "attn_row": [round(float(v), 6) for v in probs.tolist()],
        "label": int(Yte[0]),
        "query_idx": int(Qte[0]),
    }


def compare_modes_trained(
    d_model: int = 64,
    seq_len: int = 16,
    n_q_heads: int = 8,
    n_kv_gqa: int = 2,
    n_train: int = 640,
    n_test: int = 160,
    steps: int = 220,
    lr: float = 0.18,
    seed: int = 42,
    n_distractors: int = 2,
    logit_scale: float = 12.0,
) -> Dict[str, Any]:
    """Train MHA, GQA, MQA on the same matching task; return side-by-side metrics."""
    results = []
    for mode in ("mha", "gqa", "mqa"):
        results.append(
            train_matching_task(
                mode,
                d_model=d_model,
                seq_len=seq_len,
                n_q_heads=n_q_heads,
                n_kv_gqa=n_kv_gqa,
                n_train=n_train,
                n_test=n_test,
                steps=steps,
                lr=lr,
                seed=seed,
                n_distractors=n_distractors,
                logit_scale=logit_scale,
            )
        )
    by_mode = {r["mode"]: r for r in results}
    return {
        "modes": results,
        "by_mode": {
            m: {"final_acc": by_mode[m]["final_acc"], "final_mass": by_mode[m]["final_mass"],
                "n_kv_heads": by_mode[m]["n_kv_heads"]}
            for m in ("mha", "gqa", "mqa")
        },
        "gqa_near_mha": bool(
            by_mode["gqa"]["final_acc"] >= by_mode["mha"]["final_acc"] - 0.08
        ),
        "gqa_beats_mqa": bool(
            by_mode["gqa"]["final_acc"] >= by_mode["mqa"]["final_acc"] - 1e-9
        ),
        "all_above_chance": bool(
            all(by_mode[m]["final_acc"] > by_mode[m]["chance_acc"] + 0.15 for m in by_mode)
        ),
    }
