"""Matching-task data helpers for GQA experiments."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from attention import GQAAttention, softmax


def make_copy_match_batch(
    n: int,
    seq_len: int,
    d_model: int,
    rng: np.random.Generator,
    *,
    n_distractors: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planted-key matching: query must attend to the matching key token.

    For each example pick query index q and target key k_star, plant the *same*
    unit content vector on both. Distractors get a noisy version of that vector
    so ranking needs the projection to separate exact matches - capacity from
    more KV heads helps (MHA >= GQA >= MQA).
    """
    if seq_len < 4:
        raise ValueError("seq_len must be >= 4")
    X = rng.normal(size=(n, seq_len, d_model)) * 0.15
    labels = np.zeros(n, dtype=np.int64)
    q_idxs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        positions = rng.choice(seq_len, size=2 + n_distractors, replace=False)
        q = int(positions[0])
        k_star = int(positions[1])
        distractors = [int(p) for p in positions[2:]]
        q_idxs[i] = q
        labels[i] = k_star
        src = rng.normal(size=d_model)
        src /= np.linalg.norm(src) + 1e-12
        X[i, q] = src * 2.5
        X[i, k_star] = src * 2.5  # exact content match
        for k in distractors:
            d = src + rng.normal(size=d_model) * 0.4
            d /= np.linalg.norm(d) + 1e-12
            X[i, k] = d * 2.5
    return X, labels, q_idxs


def _query_logits(
    attn: GQAAttention,
    X: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
    *,
    mask_self: bool = True,
) -> np.ndarray:
    """(B, T) mean-over-heads logits from each example's query position."""
    _, _, _, logits = attn.qkv_logits(X)  # (B, Hq, T, T)
    logits = logits.mean(axis=1) * logit_scale
    rows = logits[np.arange(len(q_idxs)), q_idxs.astype(int), :].copy()
    if mask_self:
        rows[np.arange(len(q_idxs)), q_idxs.astype(int)] = -1e9
    return rows


def matching_accuracy(
    attn: GQAAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
) -> float:
    rows = _query_logits(attn, X, q_idxs, logit_scale)
    return float(np.mean(np.argmax(rows, axis=-1) == labels))


def attention_mass_on_target(
    attn: GQAAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
) -> float:
    rows = _query_logits(attn, X, q_idxs, logit_scale)
    probs = softmax(rows, axis=-1)
    return float(np.mean(probs[np.arange(len(labels)), labels]))
